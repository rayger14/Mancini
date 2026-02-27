"""NinjaTrader live runner: bridges NT bar data to Python strategy engine.

Mirrors the bar-by-bar loop from LiveSessionManager._on_bar() but reads
bars from NT's shared directory instead of Databento.

Usage:
    python3 live/nt_runner.py [--shared-dir C:\\ManciniShared] [--instrument "MES 03-26"]
"""

from __future__ import annotations

import signal as os_signal
import sys
import threading
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
from live.nt_bridge import NTBridge, NTBridgeConfig
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


class NTRunner:
    """Main live runner: bridges NinjaTrader data to Python strategy engine.

    Polls for bar files written by NinjaTrader, runs the full strategy
    pipeline, and writes signal files for NT to execute.
    """

    def __init__(
        self,
        bridge_config: NTBridgeConfig = NTBridgeConfig(),
        strategy_params: StrategyParams = PRODUCTION_STRATEGY,
        elevator_params: ElevatorParams = PRODUCTION_ELEVATOR,
        exit_params: ExitParams = PRODUCTION_EXIT,
        risk_params: RiskParams = PRODUCTION_RISK,
        session_times: SessionTimes = PRODUCTION_SESSION,
        contract: ESContractSpec = MES_CONTRACT,
        min_rr_ratio: float = 1.0,
    ):
        self.bridge = NTBridge(bridge_config)
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
        self._bars: list[dict] = []
        self._df: Optional[pd.DataFrame] = None
        self._position: Optional[TradePosition] = None
        self._pattern_type: str = ""
        self._current_signal: Optional[Signal] = None
        self._bar_count: int = 0
        self._running: bool = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._session_date: date = date.today()

    def run(self) -> None:
        """Main event loop. Blocks until session ends or shutdown signal."""
        logger.info("=" * 60)
        logger.info("MANCINI NT RUNNER STARTING")
        logger.info(f"  Instrument: {self.bridge.config.instrument}")
        logger.info(f"  Shared dir: {self.bridge.config.shared_dir}")
        logger.info("=" * 60)

        # Setup
        self.bridge.ensure_directories()
        self._running = True
        os_signal.signal(os_signal.SIGINT, self._handle_shutdown)
        os_signal.signal(os_signal.SIGTERM, self._handle_shutdown)

        # Check if Monday
        self._session_date = date.today()
        if self._session_date.weekday() == 0:
            logger.info("Monday detected — skipping session (Monday filter active)")
            return

        # Initialize session
        if not self._initialize_session():
            logger.error("Session initialization failed")
            return

        # Start heartbeat
        self._start_heartbeat()

        # Main loop: poll for bars
        logger.info("Waiting for bars from NinjaTrader...")
        try:
            while self._running:
                bar = self.bridge.poll_new_bar()
                if bar is not None:
                    self._process_bar(bar)

                    # Check for stale signals
                    stale = self.bridge.check_stale_signals(self._bar_count)
                    for sig_id in stale:
                        logger.warning(f"Signal {sig_id} has no fill after {self.bridge.config.signal_timeout_bars} bars")

                    # Check EOD
                    self._check_eod(bar)
                else:
                    _time.sleep(self.bridge.config.poll_interval_sec)

                # Check NT heartbeat periodically
                if not self.bridge.check_nt_heartbeat():
                    if self._bar_count > 0:
                        logger.warning("NinjaTrader heartbeat stale — pausing signal generation")

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._heartbeat_thread is not None:
            self._heartbeat_thread = None
        self._log_session_summary()
        logger.info("NT Runner stopped")

    # ── Session initialization ───────────────────────────────────────

    def _initialize_session(self) -> bool:
        """Initialize strategy from NT's history file.

        1. Wait for history file
        2. Build DataFrames
        3. Initialize levels
        4. Check for crash recovery (existing position)
        """
        logger.info("Waiting for NinjaTrader history file...")
        if not self.bridge.wait_for_history(self._session_date, timeout_sec=300):
            return False

        prior_day_df, current_day_df = self.bridge.read_history(self._session_date)

        # Reset strategy state
        self.strategy.reset()
        self.position_manager.start_session(datetime.now())

        # Initialize levels from prior day
        if current_day_df is not None and len(current_day_df) > 0:
            self._df = current_day_df
            self._bars = current_day_df.reset_index().to_dict("records")
            self._bar_count = len(self._bars)
        else:
            self._df = None
            self._bars = []
            self._bar_count = 0

        self.signal_aggregator.initialize_levels(
            self._df if self._df is not None else pd.DataFrame(),
            prior_day_df,
        )

        # Process any current-day bars that arrived before Python started
        if self._df is not None and len(self._df) > 0:
            logger.info(f"Catching up on {len(self._df)} bars from session start")
            velocity = compute_velocity(self._df, window=5)
            for i in range(len(self._df)):
                vel = float(velocity.iat[i]) if not np.isnan(velocity.iat[i]) else 0.0
                # Just update signal aggregator state, don't trade on historical bars
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
        pos_data = self.bridge.read_position()
        if pos_data and pos_data.get("market_position") == "long":
            logger.warning("Existing position detected — recovering state")
            self._recover_position(pos_data)

        logger.info(f"Session initialized: {self._bar_count} bars loaded, "
                     f"levels initialized from prior day")
        return True

    # ── Bar processing ───────────────────────────────────────────────

    def _process_bar(self, bar: dict) -> None:
        """Process a new bar from NinjaTrader.

        Follows the same logic as LiveSessionManager._on_bar():
        1. Append bar to DataFrame
        2. Compute velocity
        3. Check exits on open position
        4. Check for new signals
        5. Sync fills from NT
        """
        self._bar_count += 1
        ts_str = bar.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            timestamp = datetime.now()

        current_time = timestamp.time()
        open_ = float(bar.get("open", 0))
        high = float(bar.get("high", 0))
        low = float(bar.get("low", 0))
        close = float(bar.get("close", 0))
        volume = float(bar.get("volume", 0))

        # Append to bar history and rebuild DataFrame
        self._bars.append(bar)
        self._rebuild_df()

        if self._df is None or len(self._df) < 6:
            return

        # Compute velocity
        velocity = compute_velocity(self._df, window=5)
        vel = float(velocity.iat[-1]) if not np.isnan(velocity.iat[-1]) else 0.0

        # Step 1: Sync any fills from NT
        fills = self.bridge.read_fills()
        if fills:
            self._sync_from_fills(fills)

        # Step 2: Check exits on existing position
        if self._position is not None and self._position.is_open:
            exit_action = self.exit_manager.update(self._position, high, low, close)
            if exit_action is not None:
                self._handle_exit_action(exit_action)
                # If fully closed, record trade
                if self._position is not None and not self._position.is_open:
                    self.position_manager.close_position(
                        exit_price=exit_action.exit_price,
                        timestamp=timestamp,
                        exit_reason=exit_action.reason,
                        pattern_type=self._pattern_type,
                    )
                    self._position = None
                return

        # Step 3: Check for new signals (only if no position and not done)
        if self._position is None or not self._position.is_open:
            if self.position_manager.is_done_for_day:
                return

            # Don't enter on pending signals
            if self.bridge.get_pending_signals():
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
        """Evaluate signal through risk/entry gates, write entry signal if approved."""
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

        # Create local position tracking (will sync with NT fills)
        self._position = self.exit_manager.create_position(
            entry_price=entry.entry_price,
            stop_price=entry.stop_price,
            target_1=signal.target_1,
            target_2=signal.target_2,
            contracts=entry.contracts,
        )
        self._pattern_type = signal.pattern.pattern_type
        self._current_signal = signal
        self.position_manager.open_position(self._position, timestamp, self._pattern_type)

        # Write signal for NinjaTrader to execute
        self.bridge.write_entry_signal(
            quantity=entry.contracts,
            entry_price=entry.entry_price,
            stop_price=entry.stop_price,
            target_price=signal.target_1,
            signal_type=signal.signal_type.name,
            rr_ratio=signal.rr_ratio_t1,
        )

        logger.info(
            f"ENTRY SIGNAL: {entry.contracts} {self.contract.symbol} @ {entry.entry_price:.2f} "
            f"stop={entry.stop_price:.2f} T1={signal.target_1:.2f} "
            f"R:R={signal.rr_ratio_t1:.1f} [{signal.signal_type.name}]"
        )

    def _handle_exit_action(self, action: ExitAction) -> None:
        """Translate ExitAction from ExitManager into NT bridge signals."""
        if action.new_phase == ExitPhase.CLOSED:
            # Full exit — but NT bracket may handle this already
            # Write flatten as safety
            self.bridge.write_flatten(reason=action.reason)
            logger.info(f"EXIT SIGNAL: flatten — {action.reason}")

        elif action.reason.startswith("Target 1"):
            # T1 hit — exit partial, update stop to breakeven
            if action.contracts_to_close > 0 and self._position and self._position.remaining_contracts > 0:
                self.bridge.write_partial_exit(
                    quantity=action.contracts_to_close,
                    new_stop_price=action.new_stop,
                    reason=action.reason,
                )
            else:
                self.bridge.write_flatten(reason=action.reason)
            logger.info(f"EXIT SIGNAL: partial exit {action.contracts_to_close} @ T1, "
                         f"new stop={action.new_stop:.2f}")

        elif action.reason.startswith("Target 2"):
            self.bridge.write_partial_exit(
                quantity=action.contracts_to_close,
                new_stop_price=action.new_stop,
                reason=action.reason,
            )

        else:
            # Stop update (trailing)
            self.bridge.write_stop_update(
                new_stop_price=action.new_stop,
                reason=action.reason,
            )

    def _sync_from_fills(self, fills: list[dict]) -> None:
        """Process fill confirmations from NinjaTrader."""
        for fill in fills:
            action = fill.get("action", "")
            price = fill.get("price", 0)
            qty = fill.get("quantity", 0)
            sig_id = fill.get("signal_id", "")

            if action == "entry_fill":
                logger.info(f"FILL: Entry {qty} @ {price:.2f} [NT confirmed]")
                # Position already tracked locally; fill confirms it

            elif action == "exit_fill":
                logger.info(f"FILL: Exit {qty} @ {price:.2f} [NT confirmed]")

            elif action == "rejected":
                reason = fill.get("reason", "unknown")
                logger.error(f"ORDER REJECTED: {reason} [signal: {sig_id}]")
                # Reset position if entry was rejected
                if self._position is not None:
                    self._position = None

    def _recover_position(self, pos_data: dict) -> None:
        """Reconstruct TradePosition from NT's position.json on restart."""
        entry = pos_data.get("avg_entry_price", 0)
        stop = pos_data.get("working_stop", entry - 5.5)
        target = pos_data.get("working_target", entry + 10)
        qty = pos_data.get("quantity", 1)

        self._position = TradePosition(
            entry_price=entry,
            stop_price=stop,
            target_1=target,
            target_2=target + 10,
            total_contracts=qty,
            remaining_contracts=qty,
        )
        self._pattern_type = "recovered"
        logger.warning(f"Recovered position: {qty} @ {entry:.2f}, stop={stop:.2f}, target={target:.2f}")

    # ── EOD and session management ───────────────────────────────────

    def _check_eod(self, bar: dict) -> None:
        """Check for EOD flatten time (15:55 ET)."""
        ts_str = bar.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(ts_str)
            if timestamp.time() >= time(15, 55):
                if self._position is not None and self._position.is_open:
                    self.bridge.write_flatten(reason="eod_flatten")
                    logger.info("EOD flatten signal sent (15:55 ET)")
                    self._position.remaining_contracts = 0
                    self._position.phase = ExitPhase.CLOSED
                    self.position_manager.close_position(
                        exit_price=float(bar.get("close", 0)),
                        timestamp=timestamp,
                        exit_reason="EOD flatten",
                        pattern_type=self._pattern_type,
                    )
                    self._position = None
                if timestamp.time() >= time(16, 0):
                    logger.info("Session complete (16:00 ET)")
                    self._running = False
        except (ValueError, TypeError):
            pass

    def _rebuild_df(self) -> None:
        """Rebuild DataFrame from accumulated bars."""
        if not self._bars:
            self._df = None
            return
        df = pd.DataFrame(self._bars)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            if df.index.tz is None:
                df.index = df.index.tz_localize("US/Eastern")
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Keep last 400 bars to limit memory
        if len(df) > 400:
            df = df.iloc[-400:]
        self._df = df

    # ── Heartbeat ────────────────────────────────────────────────────

    def _start_heartbeat(self) -> None:
        """Start background thread writing heartbeat every 5 seconds."""
        def _heartbeat_loop():
            while self._running:
                self.bridge.write_heartbeat(
                    bars_processed=self._bar_count,
                    session_date=self._session_date,
                )
                _time.sleep(self.bridge.config.heartbeat_interval_sec)

        self._heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    # ── Shutdown ─────────────────────────────────────────────────────

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle SIGINT/SIGTERM."""
        logger.info("Shutdown signal received")
        if self._position is not None and self._position.is_open:
            self.bridge.write_flatten(reason="runner_shutdown")
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
    """Run the NinjaTrader bridge with production params."""
    import argparse

    parser = argparse.ArgumentParser(description="Mancini NinjaTrader Bridge Runner")
    parser.add_argument("--shared-dir", default=r"C:\ManciniShared",
                        help="Shared directory for NT communication")
    parser.add_argument("--instrument", default="MES 03-26",
                        help="Instrument name")
    parser.add_argument("--contracts", type=int, default=4,
                        help="Number of contracts to trade")
    args = parser.parse_args()

    config = NTBridgeConfig(
        shared_dir=args.shared_dir,
        instrument=args.instrument,
    )

    exit_params = ExitParams(
        default_contracts=args.contracts,
        t1_exit_fraction=1.0,
        trailing_stop_pts=7.0,
    )

    runner = NTRunner(
        bridge_config=config,
        exit_params=exit_params,
    )
    runner.run()


if __name__ == "__main__":
    main()
