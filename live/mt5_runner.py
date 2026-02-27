"""MetaTrader 5 live runner: bridges MT5 bar data to Python strategy engine.

Mirrors the bar-by-bar loop from the backtest engine but reads bars
from MT5's terminal via the Python API (or mt5linux on Mac).

Usage:
    python3 live/mt5_runner.py [--symbol "MES-MICRO"] [--contracts 4]

Mac setup (one-time):
    1. Install MT5 via CrossOver/Wine
    2. Inside Wine Python: pip install MetaTrader5 mt5linux rpyc
    3. Start bridge server: wine python -m mt5linux /path/to/wine/python.exe
    4. Run this script natively on Mac
"""

from __future__ import annotations

import signal as os_signal
import sys
import time as _time
from datetime import datetime, date, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    StrategyParams,
    ElevatorParams,
    ExitParams,
    RiskParams,
    SessionTimes,
    ESContractSpec,
)
from core.indicators import compute_velocity
from core.signals import Signal
from live.mt5_bridge import MT5Bridge, MT5Config
from strategy.entry_manager import EntryManager
from strategy.exit_manager import ExitManager, ExitAction, ExitPhase, TradePosition
from strategy.mancini_long import ManciniLongStrategy
from strategy.position_manager import PositionManager
from strategy.risk_manager import RiskManager


# Production params (chop 13-15, Monday filter, 6/6 validated)
PRODUCTION_STRATEGY = StrategyParams(
    swing_low_order=15,
    multi_hour_rally_min_pts=22.5,
    level_reclaim_min_touches=4,
    acceptance_min_hold_bars=7,
    acceptance_min_hold_bars_deep=8,
    acceptance_max_dip_pts=3.0,
    true_breakdown_abort_bars=12,
    fb_stop_buffer_pts=5.5,
    lr_stop_buffer_pts=5.0,
    non_acceptance_min_recovery_pts=5.0,
)
PRODUCTION_ELEVATOR = ElevatorParams(
    min_velocity_pts_per_min=0.75,
    min_levels_broken=2,
    higher_low_lookback=4,
)
PRODUCTION_EXIT = ExitParams(
    t1_exit_fraction=1.0,
    trailing_stop_pts=7.0,
)
PRODUCTION_RISK = RiskParams(max_trades_per_day=4)
PRODUCTION_SESSION = SessionTimes(
    chop_zone_start=time(13, 0),
    chop_zone_end=time(15, 0),
)
MES_CONTRACT = ESContractSpec(
    symbol="MES",
    tick_size=0.25,
    tick_value=1.25,
    point_value=5.0,
    margin_initial=1_265.0,
    margin_maintenance=1_150.0,
    exchange="CME",
)


class MT5Runner:
    """Main live runner: bridges MetaTrader 5 data to Python strategy engine.

    Polls MT5 for new 1-minute bars, runs the full strategy pipeline,
    and executes orders via the MT5 API.
    """

    def __init__(
        self,
        mt5_config: MT5Config = MT5Config(),
        strategy_params: StrategyParams = PRODUCTION_STRATEGY,
        elevator_params: ElevatorParams = PRODUCTION_ELEVATOR,
        exit_params: ExitParams = PRODUCTION_EXIT,
        risk_params: RiskParams = PRODUCTION_RISK,
        session_times: SessionTimes = PRODUCTION_SESSION,
        contract: ESContractSpec = MES_CONTRACT,
        min_rr_ratio: float = 1.0,
    ):
        self.bridge = MT5Bridge(mt5_config)
        self.contract = contract

        # Reuse ManciniLongStrategy's sub-components
        self.strategy = ManciniLongStrategy(
            strategy_params=strategy_params,
            elevator_params=elevator_params,
            exit_params=exit_params,
            risk_params=risk_params,
            session_times=session_times,
            contract=contract,
            min_rr_ratio=min_rr_ratio,
        )
        self.signal_aggregator = self.strategy.signal_aggregator
        self.entry_manager = EntryManager(
            session=session_times,
            exit_params=exit_params,
            risk_params=risk_params,
        )
        self.exit_manager = ExitManager(params=exit_params, contract=contract)
        self.position_manager = PositionManager(risk_params=risk_params)
        self.risk_manager = RiskManager(
            risk_params=risk_params,
            session=session_times,
            contract=contract,
        )

        # State
        self._df: Optional[pd.DataFrame] = None
        self._position: Optional[TradePosition] = None
        self._position_ticket: Optional[int] = None
        self._pattern_type: str = ""
        self._current_signal: Optional[Signal] = None
        self._bar_count: int = 0
        self._running: bool = False
        self._session_date: date = date.today()

    def run(self) -> None:
        """Main event loop. Blocks until session ends or shutdown signal."""
        logger.info("=" * 60)
        logger.info("MANCINI MT5 RUNNER STARTING")
        logger.info(f"  Symbol: {self.bridge.config.symbol}")
        logger.info("=" * 60)

        # Setup
        self._running = True
        os_signal.signal(os_signal.SIGINT, self._handle_shutdown)
        os_signal.signal(os_signal.SIGTERM, self._handle_shutdown)

        # Check if Monday
        self._session_date = date.today()
        if self._session_date.weekday() == 0:
            logger.info("Monday detected — skipping session (Monday filter active)")
            return

        # Connect to MT5
        if not self.bridge.connect():
            logger.error("Failed to connect to MT5")
            return

        # Initialize session
        if not self._initialize_session():
            logger.error("Session initialization failed")
            self.bridge.disconnect()
            return

        # Main loop: poll for bars
        logger.info("Waiting for bars from MT5...")
        try:
            while self._running:
                bar = self.bridge.get_latest_bar()
                if bar is not None:
                    self._process_bar(bar)
                    self._check_eod(bar)
                else:
                    _time.sleep(self.bridge.config.poll_interval_sec)

                # Sync position with MT5 periodically
                self._sync_position()

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        self._log_session_summary()
        self.bridge.disconnect()
        logger.info("MT5 Runner stopped")

    # ── Session initialization ───────────────────────────────────────

    def _initialize_session(self) -> bool:
        """Initialize strategy from MT5 historical bars.

        1. Get prior day bars for level initialization
        2. Get current day bars for catchup
        3. Initialize levels
        4. Check for existing position (crash recovery)
        """
        logger.info("Initializing session from MT5 data...")

        # Get prior day data for level calculation
        prior_day_df = self.bridge.get_prior_day_bars()
        if prior_day_df is not None:
            logger.info(f"Prior day: {len(prior_day_df)} bars loaded")
        else:
            logger.warning("No prior day data available")

        # Get current day bars (for catchup if restarting mid-session)
        all_bars = self.bridge.get_bars(count=400)
        current_day_df = None
        if all_bars is not None:
            today_mask = all_bars.index.date == self._session_date
            current_day_df = all_bars[today_mask]
            if current_day_df.empty:
                current_day_df = None

        # Reset strategy state
        self.strategy.reset()
        self.position_manager.start_session(datetime.now())

        if current_day_df is not None:
            self._df = current_day_df
            self._bar_count = len(current_day_df)
        else:
            self._df = pd.DataFrame()
            self._bar_count = 0

        # Initialize levels from prior day
        self.signal_aggregator.initialize_levels(
            self._df if self._df is not None and len(self._df) > 0 else pd.DataFrame(),
            prior_day_df,
        )

        # Catch up on current-day bars (update state, don't trade)
        if self._df is not None and len(self._df) > 0:
            logger.info(f"Catching up on {len(self._df)} bars from session start")
            velocity = compute_velocity(self._df, window=5)
            for i in range(len(self._df)):
                vel = float(velocity.iat[i]) if not np.isnan(velocity.iat[i]) else 0.0
                self.signal_aggregator.update(
                    bar_idx=i,
                    timestamp=self._df.index[i],
                    open_=float(self._df["open"].iat[i]),
                    high=float(self._df["high"].iat[i]),
                    low=float(self._df["low"].iat[i]),
                    close=float(self._df["close"].iat[i]),
                    volume=float(self._df["volume"].iat[i]),
                    velocity=vel,
                    df=self._df,
                )

        # Check for existing position (crash recovery)
        pos_data = self.bridge.get_position()
        if pos_data and pos_data.get("market_position") == "long":
            logger.warning("Existing position detected — recovering state")
            self._recover_position(pos_data)

        # Log account info
        acct = self.bridge.get_account_info()
        if acct:
            logger.info(f"Account: balance=${acct['balance']:,.0f}, "
                         f"equity=${acct['equity']:,.0f}, "
                         f"server={acct['server']}")

        logger.info(f"Session initialized: {self._bar_count} bars, "
                     f"levels initialized from prior day")
        return True

    # ── Bar processing ───────────────────────────────────────────────

    def _process_bar(self, bar: dict) -> None:
        """Process a new bar from MT5.

        Same logic as the backtest engine:
        1. Append bar to DataFrame
        2. Compute velocity
        3. Check exits on open position
        4. Check for new signals
        """
        self._bar_count += 1
        ts_str = bar.get("timestamp", "")
        try:
            timestamp = pd.Timestamp(ts_str)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("US/Eastern")
        except (ValueError, TypeError):
            timestamp = pd.Timestamp.now(tz="US/Eastern")

        current_time = timestamp.time()
        open_ = float(bar.get("open", 0))
        high = float(bar.get("high", 0))
        low = float(bar.get("low", 0))
        close = float(bar.get("close", 0))
        volume = float(bar.get("volume", 0))

        # Append to DataFrame
        new_row = pd.DataFrame(
            {"open": [open_], "high": [high], "low": [low],
             "close": [close], "volume": [volume]},
            index=pd.DatetimeIndex([timestamp]),
        )
        if self._df is None or len(self._df) == 0:
            self._df = new_row
        else:
            self._df = pd.concat([self._df, new_row])
            # Keep last 400 bars
            if len(self._df) > 400:
                self._df = self._df.iloc[-400:]

        if len(self._df) < 6:
            return

        # Compute velocity
        velocity = compute_velocity(self._df, window=5)
        vel = float(velocity.iat[-1]) if not np.isnan(velocity.iat[-1]) else 0.0

        # Step 1: Check exits on existing position
        if self._position is not None and self._position.is_open:
            exit_action = self.exit_manager.update(self._position, high, low, close)
            if exit_action is not None:
                self._handle_exit_action(exit_action, timestamp)
                if self._position is not None and not self._position.is_open:
                    self.position_manager.close_position(
                        exit_price=exit_action.exit_price,
                        timestamp=timestamp,
                        exit_reason=exit_action.reason,
                        pattern_type=self._pattern_type,
                    )
                    self._position = None
                    self._position_ticket = None
                return

        # Step 2: Check for new signals (only if no position and not done)
        if self._position is None or not self._position.is_open:
            if self.position_manager.is_done_for_day:
                return

            bar_idx = self._bar_count - 1
            signal = self.signal_aggregator.update(
                bar_idx=bar_idx,
                timestamp=timestamp,
                open_=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                velocity=vel,
                df=self._df,
            )

            if signal is not None:
                self._evaluate_and_enter(signal, current_time, timestamp)

    def _evaluate_and_enter(self, signal: Signal, current_time: time, timestamp: datetime) -> None:
        """Evaluate signal through risk/entry gates, execute if approved."""
        # Risk check
        risk_check = self.risk_manager.validate_entry(
            signal, current_time, self.position_manager
        )
        if not risk_check.passed:
            logger.info(f"Signal rejected by risk: {risk_check.reason}")
            return

        # Entry decision
        entry = self.entry_manager.evaluate(
            signal=signal,
            current_time=current_time,
            trades_today=self.position_manager.trades_today,
            is_in_profit_protection=self.position_manager.is_profit_protection,
            daily_pnl_pts=self.position_manager.daily_pnl_pts,
        )
        if not entry.should_enter:
            logger.info(f"Entry declined: {entry.reason}")
            return

        # Execute via MT5
        ticket = self.bridge.send_entry(
            volume=entry.contracts,
            sl=entry.stop_price,
            tp=signal.target_1,
            comment=f"Mancini:{signal.signal_type.name}",
        )

        if ticket is None:
            logger.error("Entry order rejected by MT5")
            return

        # Create local position tracking
        self._position = self.exit_manager.create_position(
            entry_price=entry.entry_price,
            stop_price=entry.stop_price,
            target_1=signal.target_1,
            target_2=signal.target_2,
            contracts=entry.contracts,
        )
        self._position_ticket = ticket
        self._pattern_type = signal.pattern.pattern_type
        self._current_signal = signal
        self.position_manager.open_position(self._position, timestamp, self._pattern_type)

        logger.info(
            f"ENTRY: {entry.contracts} {self.contract.symbol} @ {entry.entry_price:.2f} "
            f"stop={entry.stop_price:.2f} T1={signal.target_1:.2f} "
            f"R:R={signal.rr_ratio_t1:.1f} [{signal.signal_type.name}]"
        )

    def _handle_exit_action(self, action: ExitAction, timestamp: datetime) -> None:
        """Translate ExitAction from ExitManager into MT5 orders."""
        if self._position_ticket is None:
            return

        if action.new_phase == ExitPhase.CLOSED:
            self.bridge.flatten(reason=action.reason)
            logger.info(f"EXIT: flatten — {action.reason}")

        elif action.reason.startswith("Target 1"):
            if action.contracts_to_close > 0 and self._position and self._position.remaining_contracts > 0:
                self.bridge.partial_exit(
                    position_ticket=self._position_ticket,
                    volume=action.contracts_to_close,
                    new_sl=action.new_stop,
                    reason=action.reason,
                )
            else:
                self.bridge.flatten(reason=action.reason)
            logger.info(f"EXIT: partial {action.contracts_to_close} @ T1, "
                         f"new stop={action.new_stop:.2f}")

        elif action.reason.startswith("Target 2"):
            self.bridge.partial_exit(
                position_ticket=self._position_ticket,
                volume=action.contracts_to_close,
                new_sl=action.new_stop,
                reason=action.reason,
            )

        else:
            # Stop update (trailing)
            self.bridge.update_stop(
                position_ticket=self._position_ticket,
                new_sl=action.new_stop,
                reason=action.reason,
            )

    def _sync_position(self) -> None:
        """Sync local position state with MT5 actual position.

        Handles cases where MT5's bracket order (stop/target) fires
        independently of Python's exit logic.
        """
        if self._position is None or not self._position.is_open:
            return

        mt5_pos = self.bridge.get_position()
        if mt5_pos is None:
            # Position closed on MT5 side (stop or target hit)
            logger.info("Position closed on MT5 side (bracket order filled)")
            self._position.remaining_contracts = 0
            self._position.phase = ExitPhase.CLOSED
            self.position_manager.close_position(
                exit_price=0,  # Unknown exact fill price
                timestamp=datetime.now(),
                exit_reason="MT5 bracket fill",
                pattern_type=self._pattern_type,
            )
            self._position = None
            self._position_ticket = None

    def _recover_position(self, pos_data: dict) -> None:
        """Reconstruct TradePosition from MT5 position on restart."""
        entry = pos_data.get("price_open", 0)
        stop = pos_data.get("sl", entry - 5.5)
        target = pos_data.get("tp", entry + 10)
        qty = int(pos_data.get("volume", 1))
        ticket = pos_data.get("ticket", 0)

        self._position = TradePosition(
            entry_price=entry,
            stop_price=stop,
            target_1=target,
            target_2=target + 10,
            total_contracts=qty,
            remaining_contracts=qty,
        )
        self._position_ticket = ticket
        self._pattern_type = "recovered"
        logger.warning(f"Recovered position: ticket={ticket}, {qty} @ {entry:.2f}, "
                        f"SL={stop:.2f}, TP={target:.2f}")

    # ── EOD and session management ───────────────────────────────────

    def _check_eod(self, bar: dict) -> None:
        """Check for EOD flatten time (15:55 ET)."""
        ts_str = bar.get("timestamp", "")
        try:
            timestamp = pd.Timestamp(ts_str)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("US/Eastern")

            if timestamp.time() >= time(15, 55):
                if self._position is not None and self._position.is_open:
                    self.bridge.flatten(reason="eod_flatten")
                    logger.info("EOD flatten sent (15:55 ET)")
                    self._position.remaining_contracts = 0
                    self._position.phase = ExitPhase.CLOSED
                    self.position_manager.close_position(
                        exit_price=float(bar.get("close", 0)),
                        timestamp=timestamp,
                        exit_reason="EOD flatten",
                        pattern_type=self._pattern_type,
                    )
                    self._position = None
                    self._position_ticket = None
                if timestamp.time() >= time(16, 0):
                    logger.info("Session complete (16:00 ET)")
                    self._running = False
        except (ValueError, TypeError):
            pass

    # ── Shutdown ─────────────────────────────────────────────────────

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle SIGINT/SIGTERM."""
        logger.info("Shutdown signal received")
        if self._position is not None and self._position.is_open:
            self.bridge.flatten(reason="runner_shutdown")
        self._running = False

    def _log_session_summary(self) -> None:
        """Log end-of-session statistics."""
        if self.position_manager.session is None:
            return
        s = self.position_manager.session
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info(f"  Date:    {self._session_date}")
        logger.info(f"  Bars:    {self._bar_count}")
        logger.info(f"  Trades:  {s.trade_count}")
        logger.info(f"  Wins:    {s.wins}")
        logger.info(f"  Losses:  {s.losses}")
        logger.info(f"  PnL:     {s.daily_pnl_pts:+.1f} pts (${s.daily_pnl_dollars:+,.0f})")
        logger.info(f"  State:   {s.state.name}")
        logger.info("=" * 60)


def main():
    """Run the MT5 bridge with production params."""
    import argparse

    parser = argparse.ArgumentParser(description="Mancini MT5 Runner")
    parser.add_argument("--symbol", default="MES-MICRO",
                        help="MT5 symbol name (broker-specific)")
    parser.add_argument("--contracts", type=int, default=4,
                        help="Number of contracts to trade")
    parser.add_argument("--host", default="localhost",
                        help="mt5linux server host (Mac/Linux only)")
    parser.add_argument("--port", type=int, default=18812,
                        help="mt5linux server port (Mac/Linux only)")
    parser.add_argument("--magic", type=int, default=20260209,
                        help="EA magic number for order identification")
    args = parser.parse_args()

    config = MT5Config(
        symbol=args.symbol,
        magic=args.magic,
        host=args.host,
        port=args.port,
    )

    exit_params = ExitParams(
        default_contracts=args.contracts,
        t1_exit_fraction=1.0,
        trailing_stop_pts=7.0,
    )

    runner = MT5Runner(
        mt5_config=config,
        exit_params=exit_params,
    )
    runner.run()


if __name__ == "__main__":
    main()
