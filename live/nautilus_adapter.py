"""NautilusTrader adapter: wraps Mancini strategy into NautilusTrader's Strategy class.

This adapter bridges our existing Mancini MES futures day trading strategy
into NautilusTrader's event-driven execution framework. It translates
NautilusTrader bar events into calls to our SignalAggregator, RiskManager,
EntryManager, and ExitManager pipeline, and submits orders through
NautilusTrader's execution engine.

Broker: Interactive Brokers (via nautilus_trader.adapters.interactive_brokers)
Instrument: MES (Micro E-mini S&P 500) on CME

INSTALLATION REQUIREMENTS
=========================
NautilusTrader v1.222.0 requires:
  - Python 3.12, 3.13, or 3.14
  - macOS ARM64 (Apple Silicon) for pre-built wheels, OR
  - Linux x86_64/ARM64, Windows x86_64 for pre-built wheels
  - Building from source requires Rust toolchain (rustc + cargo)

** macOS Intel x86_64 is NOT supported **
  - No pre-built wheels exist for macOS x86_64 on PyPI
  - Source build fails: Rust linker errors in nautilus-cryptography crate
    ("ld: symbol(s) not found for architecture x86_64")
  - This is a known limitation of NautilusTrader v1.222.0

To install on a supported platform:
  uv pip install "nautilus_trader[ib]"

Or with standard pip:
  python3 -m pip install "nautilus_trader[ib]"

Usage:
  python3 live/nautilus_adapter.py \\
      --ib-host 127.0.0.1 \\
      --ib-port 7497 \\
      --account DU123456 \\
      --contracts 4

Requires TWS or IB Gateway running with API connections enabled.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, time
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
from strategy.entry_manager import EntryManager
from strategy.exit_manager import ExitManager, ExitAction, ExitPhase, TradePosition
from strategy.mancini_long import ManciniLongStrategy
from strategy.position_manager import PositionManager
from strategy.risk_manager import RiskManager

# --------------------------------------------------------------------------
# Conditional NautilusTrader imports (graceful degradation if not installed)
# --------------------------------------------------------------------------
try:
    from nautilus_trader.config import StrategyConfig as NautilusStrategyConfig
    from nautilus_trader.config import TradingNodeConfig
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.model import Bar, BarType
    from nautilus_trader.model import InstrumentId
    from nautilus_trader.model.enums import OrderSide, TimeInForce
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.trading.strategy import Strategy as NautilusStrategy

    # Interactive Brokers adapter
    from nautilus_trader.adapters.interactive_brokers.config import (
        InteractiveBrokersDataClientConfig,
        InteractiveBrokersExecClientConfig,
        InteractiveBrokersInstrumentProviderConfig,
        SymbologyMethod,
    )
    from nautilus_trader.adapters.interactive_brokers.common import IBContract
    from nautilus_trader.adapters.interactive_brokers.factories import (
        InteractiveBrokersLiveDataClientFactory,
        InteractiveBrokersLiveExecClientFactory,
    )

    NAUTILUS_AVAILABLE = True

except ImportError as e:
    NAUTILUS_AVAILABLE = False
    _import_error = str(e)
    logger.warning(
        f"NautilusTrader not available: {_import_error}. "
        "Install with: uv pip install 'nautilus_trader[ib]' "
        "(requires Python 3.12+ and macOS ARM64, Linux, or Windows)"
    )

    # Stub base class so the module can still be imported for testing
    class NautilusStrategy:  # type: ignore[no-redef]
        pass

    class NautilusStrategyConfig:  # type: ignore[no-redef]
        pass


# --------------------------------------------------------------------------
# Production parameters (same as mt5_runner.py / nt_runner.py)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# NautilusTrader Strategy Config
# --------------------------------------------------------------------------
if NAUTILUS_AVAILABLE:

    class ManciniNautilusConfig(NautilusStrategyConfig, frozen=True):
        """Configuration for ManciniNautilusStrategy.

        NautilusTrader requires a frozen (immutable) config dataclass.
        Strategy parameters are set here and passed to the strategy at init.
        """

        bar_type: str = ""  # e.g. "MES.CME-1-MINUTE-LAST-EXTERNAL"
        instrument_id_str: str = ""  # e.g. "MES.CME"
        contracts: int = 4
        min_rr_ratio: float = 1.0
        skip_monday: bool = True

else:

    class ManciniNautilusConfig:  # type: ignore[no-redef]
        """Stub config when NautilusTrader is not installed."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# --------------------------------------------------------------------------
# NautilusTrader Strategy Implementation
# --------------------------------------------------------------------------
class ManciniNautilusStrategy(NautilusStrategy):
    """NautilusTrader Strategy adapter for the Mancini MES day trading method.

    On each 1-minute bar close from NautilusTrader's data engine, this adapter:
    1. Appends the bar to a rolling pandas DataFrame
    2. Computes velocity indicator
    3. Checks for exits on any open position (via ExitManager)
    4. Runs SignalAggregator.update() for new signal detection
    5. Evaluates signals through RiskManager and EntryManager
    6. Submits bracket orders (entry + stop + target) via NautilusTrader

    The actual strategy logic is 100% delegated to our existing Python modules.
    This class only handles the NautilusTrader event interface and order routing.
    """

    def __init__(self, config: ManciniNautilusConfig) -> None:
        super().__init__(config)

        self._config = config

        # Mancini strategy sub-components
        self._strategy = ManciniLongStrategy(
            strategy_params=PRODUCTION_STRATEGY,
            elevator_params=PRODUCTION_ELEVATOR,
            exit_params=ExitParams(
                default_contracts=config.contracts,
                t1_exit_fraction=1.0,
                trailing_stop_pts=7.0,
            ),
            risk_params=PRODUCTION_RISK,
            session_times=PRODUCTION_SESSION,
            contract=MES_CONTRACT,
            min_rr_ratio=config.min_rr_ratio,
        )
        self._signal_aggregator = self._strategy.signal_aggregator
        self._entry_manager = EntryManager(
            session=PRODUCTION_SESSION,
            exit_params=ExitParams(
                default_contracts=config.contracts,
                t1_exit_fraction=1.0,
                trailing_stop_pts=7.0,
            ),
            risk_params=PRODUCTION_RISK,
        )
        self._exit_manager = ExitManager(
            params=ExitParams(
                default_contracts=config.contracts,
                t1_exit_fraction=1.0,
                trailing_stop_pts=7.0,
            ),
            contract=MES_CONTRACT,
        )
        self._position_manager = PositionManager(risk_params=PRODUCTION_RISK)
        self._risk_manager = RiskManager(
            risk_params=PRODUCTION_RISK,
            session=PRODUCTION_SESSION,
            contract=MES_CONTRACT,
        )

        # Bar accumulation state
        self._bars_data: list[dict] = []
        self._df: Optional[pd.DataFrame] = None
        self._bar_count: int = 0

        # Position tracking (mirrors MT5/NT runners)
        self._position: Optional[TradePosition] = None
        self._pattern_type: str = ""
        self._current_signal: Optional[Signal] = None
        self._session_date: Optional[date] = None

        # NautilusTrader identifiers (set in on_start)
        self._instrument_id: Optional[InstrumentId] = None
        self._bar_type: Optional[BarType] = None

    # ------------------------------------------------------------------
    # NautilusTrader lifecycle callbacks
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Called when the strategy is started by the TradingNode.

        Sets up subscriptions and initializes session state.
        """
        self.log.info("ManciniNautilusStrategy starting...")

        # Parse instrument and bar type from config
        self._instrument_id = InstrumentId.from_str(self._config.instrument_id_str)
        self._bar_type = BarType.from_str(self._config.bar_type)

        # Verify instrument exists in cache
        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            self.log.error(
                f"Could not find instrument for {self._instrument_id}. "
                "Check that the IB instrument provider loaded MES."
            )
            self.stop()
            return

        self.log.info(f"Instrument loaded: {instrument}")

        # Monday filter
        self._session_date = date.today()
        if self._config.skip_monday and self._session_date.weekday() == 0:
            self.log.info("Monday detected -- skipping session (Monday filter)")
            self.stop()
            return

        # Initialize session
        self._strategy.reset()
        self._position_manager.start_session(datetime.now())
        self._bars_data.clear()
        self._df = None
        self._bar_count = 0

        # Request historical bars for level initialization and state catchup
        self.request_bars(self._bar_type)

        # Subscribe to live 1-minute bars
        self.subscribe_bars(self._bar_type)

        self.log.info(
            f"Subscribed to {self._bar_type}. "
            f"Contracts: {self._config.contracts}, "
            f"Min R:R: {self._config.min_rr_ratio}"
        )

    def on_stop(self) -> None:
        """Called when the strategy is stopped.

        Flattens any open position and logs session summary.
        """
        self.log.info("ManciniNautilusStrategy stopping...")

        # Flatten any open position
        if self._position is not None and self._position.is_open:
            self.log.warning("Flattening open position on shutdown")
            self.close_all_positions(self._instrument_id)

        # Log session summary
        self._log_session_summary()

        # Unsubscribe
        if self._bar_type is not None:
            self.unsubscribe_bars(self._bar_type)

    def on_historical_data(self, data) -> None:
        """Process historical bars for level initialization.

        NautilusTrader calls this with batches of bars from request_bars().
        We use these to initialize price levels (prior day) and catch up
        on current-day state.
        """
        # Historical bars are automatically used to update registered indicators.
        # We also need them for our custom level initialization.
        if not isinstance(data, list):
            return

        self.log.info(f"Received {len(data)} historical bars for initialization")

        for bar in data:
            self._append_bar_to_df(bar)

        # Initialize levels from accumulated history
        if self._df is not None and len(self._df) > 0:
            today = date.today()
            today_mask = self._df.index.date == today
            prior_mask = self._df.index.date < today

            prior_day_df = self._df[prior_mask]
            current_day_df = self._df[today_mask]

            if prior_day_df.empty:
                prior_day_df = None
            if current_day_df.empty:
                current_day_df = None

            # Initialize levels from prior day data
            self._signal_aggregator.initialize_levels(
                current_day_df if current_day_df is not None else pd.DataFrame(),
                prior_day_df,
            )

            # Catch up on current-day bars (update state only, no trading)
            if current_day_df is not None and len(current_day_df) >= 6:
                self.log.info(
                    f"Catching up on {len(current_day_df)} current-day bars"
                )
                velocity = compute_velocity(current_day_df, window=5)
                for i in range(len(current_day_df)):
                    vel = (
                        float(velocity.iat[i])
                        if not np.isnan(velocity.iat[i])
                        else 0.0
                    )
                    self._signal_aggregator.update(
                        bar_idx=i,
                        timestamp=current_day_df.index[i].to_pydatetime(),
                        open_=float(current_day_df["open"].iat[i]),
                        high=float(current_day_df["high"].iat[i]),
                        low=float(current_day_df["low"].iat[i]),
                        close=float(current_day_df["close"].iat[i]),
                        volume=float(current_day_df["volume"].iat[i]),
                        velocity=vel,
                        df=current_day_df,
                    )
                self._bar_count = len(current_day_df)

    def on_bar(self, bar: Bar) -> None:
        """Called on each new 1-minute bar close.

        This is the main strategy loop. It mirrors the logic in
        mt5_runner.py and nt_runner.py _process_bar() methods.
        """
        self._bar_count += 1

        # Extract OHLCV from NautilusTrader Bar object
        timestamp = bar.ts_event  # nanosecond timestamp
        ts_datetime = pd.Timestamp(timestamp, unit="ns", tz="UTC")
        ts_eastern = ts_datetime.tz_convert("US/Eastern")

        open_ = float(bar.open)
        high = float(bar.high)
        low = float(bar.low)
        close = float(bar.close)
        volume = float(bar.volume)
        current_time = ts_eastern.time()

        # Append to DataFrame
        self._append_bar_to_df(bar)

        if self._df is None or len(self._df) < 6:
            return

        # Check EOD flatten
        if current_time >= time(15, 55):
            if self._position is not None and self._position.is_open:
                self.log.info("EOD flatten (15:55 ET) -- closing all positions")
                self.close_all_positions(self._instrument_id)
                self._force_close_local_position(close, ts_eastern.to_pydatetime())
            if current_time >= time(16, 0):
                self.log.info("Session complete (16:00 ET)")
                self.stop()
            return

        # Compute velocity
        velocity = compute_velocity(self._df, window=5)
        vel = float(velocity.iat[-1]) if not np.isnan(velocity.iat[-1]) else 0.0

        # Step 1: Check exits on existing position
        if self._position is not None and self._position.is_open:
            exit_action = self._exit_manager.update(
                self._position, high, low, close
            )
            if exit_action is not None:
                self._handle_exit_action(exit_action, ts_eastern.to_pydatetime())
                if self._position is not None and not self._position.is_open:
                    self._position_manager.close_position(
                        exit_price=exit_action.exit_price,
                        timestamp=ts_eastern.to_pydatetime(),
                        exit_reason=exit_action.reason,
                        pattern_type=self._pattern_type,
                    )
                    self._position = None
                return

        # Step 2: Check for new signals (only if flat and not done)
        if self._position is None or not self._position.is_open:
            if self._position_manager.is_done_for_day:
                return

            bar_idx = self._bar_count - 1
            signal = self._signal_aggregator.update(
                bar_idx=bar_idx,
                timestamp=ts_eastern.to_pydatetime(),
                open_=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                velocity=vel,
                df=self._df,
            )

            if signal is not None:
                self._evaluate_and_enter(signal, current_time, ts_eastern.to_pydatetime())

    def on_order_filled(self, event) -> None:
        """Called when an order is filled by the venue.

        Provides fill confirmation logging.
        """
        self.log.info(f"Order filled: {event}")

    def on_order_rejected(self, event) -> None:
        """Called when an order is rejected by the venue.

        Resets local position tracking if entry was rejected.
        """
        self.log.error(f"Order rejected: {event}")
        # If our entry was rejected, clean up local state
        if self._position is not None and self._position.is_open:
            self._position = None
            self._pattern_type = ""

    # ------------------------------------------------------------------
    # Signal evaluation and order submission
    # ------------------------------------------------------------------

    def _evaluate_and_enter(
        self, signal: Signal, current_time: time, timestamp: datetime
    ) -> None:
        """Evaluate signal through risk/entry gates, submit bracket order if approved."""
        # Risk check
        risk_check = self._risk_manager.validate_entry(
            signal, current_time, self._position_manager
        )
        if not risk_check.passed:
            self.log.info(f"Signal rejected by risk: {risk_check.reason}")
            return

        # Entry decision
        entry = self._entry_manager.evaluate(
            signal=signal,
            current_time=current_time,
            trades_today=self._position_manager.trades_today,
            is_in_profit_protection=self._position_manager.is_profit_protection,
            daily_pnl_pts=self._position_manager.daily_pnl_pts,
        )
        if not entry.should_enter:
            self.log.info(f"Entry declined: {entry.reason}")
            return

        # Create local position tracking
        self._position = self._exit_manager.create_position(
            entry_price=entry.entry_price,
            stop_price=entry.stop_price,
            target_1=signal.target_1,
            target_2=signal.target_2,
            contracts=entry.contracts,
        )
        self._pattern_type = signal.pattern.pattern_type
        self._current_signal = signal
        self._position_manager.open_position(
            self._position, timestamp, self._pattern_type
        )

        # Submit bracket order via NautilusTrader execution engine
        self._submit_bracket_order(
            contracts=entry.contracts,
            stop_price=entry.stop_price,
            target_price=signal.target_1,
            signal_type=signal.signal_type.name,
        )

        self.log.info(
            f"ENTRY: {entry.contracts} MES @ {entry.entry_price:.2f} "
            f"stop={entry.stop_price:.2f} T1={signal.target_1:.2f} "
            f"R:R={signal.rr_ratio_t1:.1f} [{signal.signal_type.name}]"
        )

    def _submit_bracket_order(
        self,
        contracts: int,
        stop_price: float,
        target_price: float,
        signal_type: str,
    ) -> None:
        """Submit a bracket order (market entry + stop + target) via NautilusTrader.

        Uses NautilusTrader's OrderFactory.bracket() to create an OUO
        (One-Updates-Other) bracket: market entry, limit take-profit,
        stop-market stop-loss.
        """
        if not NAUTILUS_AVAILABLE or self._instrument_id is None:
            self.log.error("Cannot submit order: NautilusTrader not available")
            return

        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            self.log.error(f"Instrument not found: {self._instrument_id}")
            return

        # Round prices to tick size (MES = 0.25)
        tick_size = float(instrument.price_increment)
        stop_rounded = round(round(stop_price / tick_size) * tick_size, 2)
        target_rounded = round(round(target_price / tick_size) * tick_size, 2)

        bracket_order = self.order_factory.bracket(
            instrument_id=self._instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_int(contracts),
            # Entry: market order (immediate fill)
            entry_order_type="MARKET",
            time_in_force=TimeInForce.GTC,
            entry_tags=[f"MANCINI:{signal_type}"],
            # Take-profit: limit order at T1
            tp_price=Price.from_str(f"{target_rounded}"),
            tp_order_type="LIMIT",
            tp_time_in_force=TimeInForce.GTC,
            # Stop-loss: stop market below sweep level
            sl_trigger_price=Price.from_str(f"{stop_rounded}"),
            sl_time_in_force=TimeInForce.GTC,
        )

        # Submit all orders in the bracket
        self.submit_order_list(bracket_order)
        self.log.info(
            f"Bracket order submitted: BUY {contracts} MES, "
            f"TP={target_rounded}, SL={stop_rounded}"
        )

    # ------------------------------------------------------------------
    # Exit handling
    # ------------------------------------------------------------------

    def _handle_exit_action(self, action: ExitAction, timestamp: datetime) -> None:
        """Translate ExitAction into NautilusTrader order operations.

        For the current t1_exit_fraction=1.0 configuration, the ExitManager
        will either stop out (close all) or hit T1 (close all). The bracket
        order on the venue side handles this automatically, but we sync
        local state.
        """
        if action.new_phase == ExitPhase.CLOSED:
            # Full exit -- close all via NautilusTrader
            self.close_all_positions(self._instrument_id)
            self.log.info(f"EXIT: {action.reason}")
        elif action.reason.startswith("Target 1") and action.contracts_to_close > 0:
            # T1 hit with 100% exit -- bracket TP handles this on venue side
            # Just log; venue bracket will auto-close
            self.log.info(
                f"EXIT: T1 hit, {action.contracts_to_close} contracts, "
                f"new stop={action.new_stop:.2f}"
            )
        else:
            # Trailing stop update -- modify the SL order
            # With t1_exit_fraction=1.0, this branch is rarely reached
            self.log.info(f"Stop update: {action.new_stop:.2f} ({action.reason})")

    def _force_close_local_position(self, price: float, timestamp: datetime) -> None:
        """Force-close local position tracking (for EOD flatten, shutdown, etc.)."""
        if self._position is not None and self._position.is_open:
            self._position.remaining_contracts = 0
            self._position.phase = ExitPhase.CLOSED
            self._position_manager.close_position(
                exit_price=price,
                timestamp=timestamp,
                exit_reason="EOD flatten",
                pattern_type=self._pattern_type,
            )
            self._position = None

    # ------------------------------------------------------------------
    # DataFrame management
    # ------------------------------------------------------------------

    def _append_bar_to_df(self, bar) -> None:
        """Append a NautilusTrader Bar to the rolling DataFrame."""
        if NAUTILUS_AVAILABLE:
            ts = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").tz_convert(
                "US/Eastern"
            )
            open_ = float(bar.open)
            high = float(bar.high)
            low = float(bar.low)
            close = float(bar.close)
            volume = float(bar.volume)
        else:
            # Fallback for testing without NautilusTrader
            ts = pd.Timestamp.now(tz="US/Eastern")
            open_ = high = low = close = volume = 0.0

        new_row = pd.DataFrame(
            {
                "open": [open_],
                "high": [high],
                "low": [low],
                "close": [close],
                "volume": [volume],
            },
            index=pd.DatetimeIndex([ts]),
        )

        if self._df is None or len(self._df) == 0:
            self._df = new_row
        else:
            self._df = pd.concat([self._df, new_row])
            # Keep last 400 bars to limit memory
            if len(self._df) > 400:
                self._df = self._df.iloc[-400:]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_session_summary(self) -> None:
        """Log end-of-session statistics."""
        if self._position_manager.session is None:
            return
        s = self._position_manager.session
        self.log.info("=" * 60)
        self.log.info("SESSION SUMMARY")
        self.log.info(f"  Date:    {self._session_date}")
        self.log.info(f"  Bars:    {self._bar_count}")
        self.log.info(f"  Trades:  {s.trade_count}")
        self.log.info(f"  Wins:    {s.wins}")
        self.log.info(f"  Losses:  {s.losses}")
        self.log.info(
            f"  PnL:     {s.daily_pnl_pts:+.1f} pts (${s.daily_pnl_dollars:+,.0f})"
        )
        self.log.info(f"  State:   {s.state.name}")
        self.log.info("=" * 60)


# ==========================================================================
# TradingNode configuration and launcher
# ==========================================================================

IB_VENUE = None  # Set after imports


def build_trading_node_config(
    ib_host: str = "127.0.0.1",
    ib_port: int = 7497,
    ib_account: str = "",
    contracts: int = 4,
    min_rr_ratio: float = 1.0,
) -> dict:
    """Build the NautilusTrader TradingNode configuration dict.

    Parameters
    ----------
    ib_host : str
        TWS/Gateway API host (default: 127.0.0.1)
    ib_port : int
        TWS/Gateway API port (7497 = paper, 7496 = live)
    ib_account : str
        IB account ID (e.g. "DU123456" for paper)
    contracts : int
        Number of MES contracts to trade
    min_rr_ratio : float
        Minimum risk:reward ratio for entries

    Returns
    -------
    dict
        Configuration suitable for TradingNodeConfig
    """
    if not NAUTILUS_AVAILABLE:
        raise RuntimeError(
            "NautilusTrader is not installed. "
            "Install with: uv pip install 'nautilus_trader[ib]' "
            "(requires Python 3.12+ and a supported platform)"
        )

    # The MES continuous futures contract on CME
    mes_contract = IBContract(
        secType="CONTFUT",
        exchange="CME",
        symbol="MES",
        build_futures_chain=False,  # We want the front-month continuous
    )

    instrument_provider_config = InteractiveBrokersInstrumentProviderConfig(
        symbology_method=SymbologyMethod.IB_SIMPLIFIED,
        build_futures_chain=False,
        load_contracts=frozenset([mes_contract]),
        cache_validity_days=1,
    )

    data_client_config = InteractiveBrokersDataClientConfig(
        ibg_host=ib_host,
        ibg_port=ib_port,
        ibg_client_id=1,
        instrument_provider=instrument_provider_config,
    )

    exec_client_config = InteractiveBrokersExecClientConfig(
        ibg_host=ib_host,
        ibg_port=ib_port,
        ibg_client_id=2,
        account_id=ib_account,
        instrument_provider=instrument_provider_config,
    )

    # NautilusTrader node configuration
    config = TradingNodeConfig(
        trader_id="MANCINI-001",
        data_clients={"IB": data_client_config},
        exec_clients={"IB": exec_client_config},
    )

    return config


def create_strategy(
    contracts: int = 4,
    min_rr_ratio: float = 1.0,
) -> ManciniNautilusStrategy:
    """Create a ManciniNautilusStrategy instance with production parameters.

    Parameters
    ----------
    contracts : int
        Number of MES contracts
    min_rr_ratio : float
        Minimum R:R ratio for entries

    Returns
    -------
    ManciniNautilusStrategy
    """
    # The instrument_id and bar_type use NautilusTrader's simplified IB symbology
    # MES continuous futures on CME via IB
    config = ManciniNautilusConfig(
        bar_type="MES.CME-1-MINUTE-LAST-EXTERNAL",
        instrument_id_str="MES.CME",
        contracts=contracts,
        min_rr_ratio=min_rr_ratio,
        skip_monday=True,
    )
    return ManciniNautilusStrategy(config=config)


def run_live(
    ib_host: str = "127.0.0.1",
    ib_port: int = 7497,
    ib_account: str = "",
    contracts: int = 4,
    min_rr_ratio: float = 1.0,
) -> None:
    """Run the Mancini strategy live via NautilusTrader + Interactive Brokers.

    Parameters
    ----------
    ib_host : str
        TWS/Gateway API host
    ib_port : int
        TWS/Gateway API port (7497=paper, 7496=live)
    ib_account : str
        IB account ID
    contracts : int
        Number of MES contracts
    min_rr_ratio : float
        Minimum R:R ratio
    """
    if not NAUTILUS_AVAILABLE:
        logger.error(
            "NautilusTrader is not installed. Cannot run live trading.\n"
            "Install with: uv pip install 'nautilus_trader[ib]'\n"
            "Requires: Python 3.12+, macOS ARM64 / Linux / Windows"
        )
        sys.exit(1)

    global IB_VENUE
    IB_VENUE = Venue("IB")

    node_config = build_trading_node_config(
        ib_host=ib_host,
        ib_port=ib_port,
        ib_account=ib_account,
        contracts=contracts,
        min_rr_ratio=min_rr_ratio,
    )

    node = None
    try:
        node = TradingNode(config=node_config)
        node.add_data_client_factory("IB", InteractiveBrokersLiveDataClientFactory)
        node.add_exec_client_factory("IB", InteractiveBrokersLiveExecClientFactory)
        node.build()

        # Set IB as the specific venue for portfolio
        node.portfolio.set_specific_venue(IB_VENUE)

        # Add our strategy
        strategy = create_strategy(
            contracts=contracts,
            min_rr_ratio=min_rr_ratio,
        )
        node.trader.add_strategy(strategy)

        logger.info("=" * 60)
        logger.info("MANCINI NAUTILUS RUNNER STARTING")
        logger.info(f"  Broker:     Interactive Brokers ({ib_host}:{ib_port})")
        logger.info(f"  Account:    {ib_account}")
        logger.info(f"  Instrument: MES (Micro E-mini S&P 500)")
        logger.info(f"  Contracts:  {contracts}")
        logger.info(f"  Min R:R:    {min_rr_ratio}")
        logger.info("=" * 60)

        # Run (blocks until shutdown)
        node.run()

    except KeyboardInterrupt:
        logger.info("Shutdown requested (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if node:
            node.dispose()
            logger.info("TradingNode disposed")


# ==========================================================================
# CLI entry point
# ==========================================================================

def main():
    """Command-line entry point for the NautilusTrader live runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Mancini MES Strategy - NautilusTrader Live Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requirements:
  - Python 3.12+ with nautilus_trader[ib] installed
  - TWS or IB Gateway running with API connections enabled
  - macOS ARM64, Linux, or Windows (macOS Intel x86_64 NOT supported)

Examples:
  # Paper trading (TWS default paper port)
  python3 live/nautilus_adapter.py --ib-port 7497 --account DU123456

  # Live trading (use with extreme caution!)
  python3 live/nautilus_adapter.py --ib-port 7496 --account U123456 --contracts 1
        """,
    )
    parser.add_argument(
        "--ib-host",
        default="127.0.0.1",
        help="TWS/Gateway API host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--ib-port",
        type=int,
        default=7497,
        help="TWS/Gateway API port (7497=paper, 7496=live)",
    )
    parser.add_argument(
        "--account",
        default="",
        help="IB account ID (e.g. DU123456 for paper trading)",
    )
    parser.add_argument(
        "--contracts",
        type=int,
        default=4,
        help="Number of MES contracts to trade (default: 4)",
    )
    parser.add_argument(
        "--min-rr",
        type=float,
        default=1.0,
        help="Minimum risk:reward ratio for entries (default: 1.0)",
    )
    args = parser.parse_args()

    if not NAUTILUS_AVAILABLE:
        print(
            "\nERROR: NautilusTrader is not installed on this system.\n"
            "\n"
            "Installation failed because:\n"
            "  - This machine is macOS Intel (x86_64)\n"
            "  - NautilusTrader only provides macOS ARM64 (Apple Silicon) wheels\n"
            "  - Building from source fails: Rust linker error in nautilus-cryptography\n"
            "\n"
            "Options:\n"
            "  1. Run on a Mac with Apple Silicon (M1/M2/M3/M4)\n"
            "  2. Run on Linux (x86_64 or ARM64)\n"
            "  3. Run on Windows (x86_64)\n"
            "  4. Use Docker with a Linux image:\n"
            "     docker run -it python:3.12 bash\n"
            "     pip install 'nautilus_trader[ib]'\n"
            "\n"
            "For this machine, consider using the existing runners:\n"
            "  - live/mt5_runner.py  (MetaTrader 5 via mt5linux bridge)\n"
            "  - live/nt_runner.py   (NinjaTrader via file bridge)\n"
        )
        sys.exit(1)

    run_live(
        ib_host=args.ib_host,
        ib_port=args.ib_port,
        ib_account=args.account,
        contracts=args.contracts,
        min_rr_ratio=args.min_rr,
    )


if __name__ == "__main__":
    main()
