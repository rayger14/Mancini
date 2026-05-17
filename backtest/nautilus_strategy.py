"""NautilusTrader Strategy wrapper for the Mancini Method.

Delegates all signal detection, entry evaluation, and risk management
to existing Mancini components.  NautilusTrader handles order routing,
fills, slippage, and commissions.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from enum import Enum, auto
from typing import Optional

import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.enums import OrderSide, TimeInForce, TriggerType
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId, ClientOrderId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from config.settings import (
    StrategyParams,
    ElevatorParams,
    ExitParams,
    RiskParams,
    DEFAULT_STRATEGY,
    DEFAULT_ELEVATOR,
    DEFAULT_EXIT,
    DEFAULT_RISK,
    DEFAULT_SESSION,
    DEFAULT_CONTRACT,
)
from core.signals import SignalAggregator
from strategy.entry_manager import EntryManager
from strategy.position_manager import PositionManager, TradeRecord
from strategy.risk_manager import RiskManager


class _Phase(Enum):
    """Internal position phase tracking."""
    FLAT = auto()
    PENDING_ENTRY = auto()
    INITIAL = auto()       # full position, initial stop
    AFTER_T1 = auto()      # 75% exited, stop at breakeven
    AFTER_T2 = auto()      # runner only, trailing stop
    CLOSED = auto()


class ManciniNautilusConfig(StrategyConfig, frozen=True):
    """Configuration for ManciniNautilusStrategy."""

    instrument_id: str = "ES.GLBX"
    bar_type: str = "ES.GLBX-1-MINUTE-LAST-EXTERNAL"

    # Serialised param dicts — frozen dataclasses can't be pydantic fields
    strategy_params: dict = {}
    elevator_params: dict = {}
    exit_params: dict = {}
    risk_params: dict = {}

    min_rr_ratio: float = 1.5

    # Prior day OHLCV summary for level initialization
    prior_day_data: dict | None = None


def _reconstruct_params(cfg: ManciniNautilusConfig):
    """Rebuild frozen dataclass param objects from config dicts."""
    sp = StrategyParams(**cfg.strategy_params) if cfg.strategy_params else DEFAULT_STRATEGY
    ep = ElevatorParams(**cfg.elevator_params) if cfg.elevator_params else DEFAULT_ELEVATOR
    xp_raw = dict(cfg.exit_params) if cfg.exit_params else {}
    # trailing_tighten_thresholds serialises as list-of-lists; fix up
    if "trailing_tighten_thresholds" in xp_raw:
        xp_raw["trailing_tighten_thresholds"] = [
            tuple(x) for x in xp_raw["trailing_tighten_thresholds"]
        ]
    xp = ExitParams(**xp_raw) if xp_raw else DEFAULT_EXIT
    rp = RiskParams(**cfg.risk_params) if cfg.risk_params else DEFAULT_RISK
    return sp, ep, xp, rp


class ManciniNautilusStrategy(Strategy):
    """NautilusTrader strategy that reuses Mancini signal/entry/risk logic.

    Order lifecycle:
        on_bar  → signal detected → market BUY
        on_order_filled (entry) → submit T1 limit, T2 limit, stop-market
        on_order_filled (T1)    → amend stop to breakeven, reduce qty
        on_order_filled (T2)    → switch to trailing stop
        on_order_filled (stop)  → cancel remaining limits, record trade
    """

    def __init__(self, config: ManciniNautilusConfig) -> None:
        super().__init__(config)

        sp, ep, xp, rp = _reconstruct_params(config)

        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._bar_type_str = config.bar_type
        self._exit_params = xp
        self._contract_spec = DEFAULT_CONTRACT
        self._min_rr_ratio = config.min_rr_ratio
        self._prior_day_data = config.prior_day_data

        # Existing Mancini components
        self._signal_agg = SignalAggregator(
            strategy_params=sp,
            elevator_params=ep,
            exit_params=xp,
            min_rr_ratio=config.min_rr_ratio,
        )
        self._entry_mgr = EntryManager(
            session=DEFAULT_SESSION,
            exit_params=xp,
            risk_params=rp,
        )
        self._risk_mgr = RiskManager(
            risk_params=rp,
            session=DEFAULT_SESSION,
            contract=DEFAULT_CONTRACT,
        )
        self._pos_mgr = PositionManager(risk_params=rp)

        # Bar accumulation
        self._bars: list[dict] = []
        self._velocity_deque: deque[float] = deque(maxlen=5)

        # Position / order tracking
        self._phase = _Phase.FLAT
        self._entry_price: float = 0.0
        self._stop_price: float = 0.0
        self._target_1: float = 0.0
        self._target_2: float = 0.0
        self._total_contracts: int = 0
        self._remaining_contracts: int = 0
        self._highest_since_entry: float = 0.0
        self._pattern_type: str = ""
        self._entry_time: Optional[datetime] = None
        self._total_commission: float = 0.0

        # Order ID tracking
        self._entry_order_id: Optional[ClientOrderId] = None
        self._t1_order_id: Optional[ClientOrderId] = None
        self._t2_order_id: Optional[ClientOrderId] = None
        self._stop_order_id: Optional[ClientOrderId] = None

        # Completed trades
        self._completed_trades: list[TradeRecord] = []

        # Realized PnL accumulator — updated on every fill (T1, T2, stop).
        # Each fill adds (fill_price - entry_price) * fill_qty for longs.
        self._realized_pnl_pts: float = 0.0
        self._final_exit_price: float = 0.0  # last fill that closed the position
        self._exit_reason_hint: str = "Stop loss hit"  # updated based on which order fills last

        # Levels initialised flag
        self._levels_initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        from nautilus_trader.model.data import BarType

        bar_type = BarType.from_str(self._bar_type_str)
        self.subscribe_bars(bar_type)

        # Start a new PositionManager session
        self._pos_mgr.start_session(datetime.now())

    def on_bar(self, bar) -> None:
        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            return

        # Bar timestamps from Nautilus are nanoseconds since epoch (UTC).
        # Our level_store and pattern detectors compare against tz-aware
        # timestamps from the source parquet (US/Eastern), so convert.
        ts = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").tz_convert("US/Eastern").to_pydatetime()
        o = float(bar.open)
        h = float(bar.high)
        lo = float(bar.low)
        c = float(bar.close)
        v = float(bar.volume)

        # Accumulate bar
        self._bars.append({
            "open": o, "high": h, "low": lo, "close": c, "volume": v,
            "timestamp": ts,
        })

        # Velocity from last 5 closes
        self._velocity_deque.append(c)
        if len(self._velocity_deque) >= 2:
            velocity = (self._velocity_deque[-1] - self._velocity_deque[0]) / len(self._velocity_deque)
        else:
            velocity = 0.0

        bar_idx = len(self._bars) - 1

        # One-time level initialization after we have enough bars
        if not self._levels_initialized and len(self._bars) >= 2:
            df = self._bars_to_dataframe()
            prior_df = self._prior_day_to_dataframe()
            self._signal_agg.initialize_levels(df, prior_df)
            self._levels_initialized = True

        # Update trailing stop if runner active
        if self._phase == _Phase.AFTER_T2:
            if h > self._highest_since_entry:
                self._highest_since_entry = h
                self._update_trailing_stop(instrument)

        # Skip signal detection if position open or session done
        if self._phase not in (_Phase.FLAT, _Phase.CLOSED):
            # Track highest for trailing
            if self._phase in (_Phase.INITIAL, _Phase.AFTER_T1) and h > self._highest_since_entry:
                self._highest_since_entry = h
            return

        if self._pos_mgr.is_done_for_day:
            return

        # Build partial df for incremental level detection
        df = self._bars_to_dataframe()

        signal = self._signal_agg.update(
            bar_idx=bar_idx,
            timestamp=ts,
            open_=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            velocity=velocity,
            df=df,
        )

        if signal is None:
            return

        current_time = ts.time()

        # Risk check
        risk_check = self._risk_mgr.validate_entry(
            signal, current_time, self._pos_mgr,
        )
        if not risk_check.passed:
            return

        # Entry evaluation
        entry = self._entry_mgr.evaluate(
            signal=signal,
            current_time=current_time,
            trades_today=self._pos_mgr.trades_today,
            is_in_profit_protection=self._pos_mgr.is_profit_protection,
            daily_pnl_pts=self._pos_mgr.daily_pnl_pts,
        )

        if not entry.should_enter:
            return

        # Submit market buy
        self._total_contracts = entry.contracts
        self._stop_price = signal.stop_price
        self._target_1 = signal.target_1
        self._target_2 = signal.target_2
        self._pattern_type = signal.pattern.pattern_type
        self._phase = _Phase.PENDING_ENTRY
        self._total_commission = 0.0

        order = self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_int(entry.contracts),
            time_in_force=TimeInForce.FOK,
        )
        self._entry_order_id = order.client_order_id
        self.submit_order(order)

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def on_order_filled(self, event: OrderFilled) -> None:
        oid = event.client_order_id

        # Accumulate commission
        if event.commission is not None:
            self._total_commission += float(event.commission)

        if oid == self._entry_order_id:
            self._on_entry_filled(event)
        elif oid == self._t1_order_id:
            self._on_t1_filled(event)
        elif oid == self._t2_order_id:
            self._on_t2_filled(event)
        elif oid == self._stop_order_id:
            self._on_stop_filled(event)

    def _on_entry_filled(self, event: OrderFilled) -> None:
        instrument = self.cache.instrument(self._instrument_id)
        self._entry_price = float(event.last_px)
        self._entry_time = pd.Timestamp(event.ts_event, unit="ns").to_pydatetime()
        self._remaining_contracts = self._total_contracts
        self._highest_since_entry = self._entry_price
        self._phase = _Phase.INITIAL

        # Register with PositionManager
        from strategy.exit_manager import TradePosition
        pos = TradePosition(
            entry_price=self._entry_price,
            stop_price=self._stop_price,
            target_1=self._target_1,
            target_2=self._target_2,
            total_contracts=self._total_contracts,
            remaining_contracts=self._total_contracts,
        )
        self._pos_mgr.open_position(pos, self._entry_time, self._pattern_type)

        # Submit bracket orders
        self._submit_bracket_orders(instrument)

    def _submit_bracket_orders(self, instrument) -> None:
        """Place T1 limit, T2 limit, and stop-market."""
        t1_qty = round(self._total_contracts * self._exit_params.t1_exit_fraction)
        t2_qty = round(self._total_contracts * self._exit_params.t2_exit_fraction)
        # Ensure at least 1 runner remains
        runner_qty = max(1, self._total_contracts - t1_qty - t2_qty)
        # Adjust t2 if total doesn't add up
        t2_qty = self._total_contracts - t1_qty - runner_qty

        # T1 limit sell
        if t1_qty > 0:
            t1_order = self.order_factory.limit(
                instrument_id=self._instrument_id,
                order_side=OrderSide.SELL,
                quantity=Quantity.from_int(t1_qty),
                price=Price(self._round_to_tick(self._target_1), precision=2),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
            self._t1_order_id = t1_order.client_order_id
            self.submit_order(t1_order)

        # T2 limit sell
        if t2_qty > 0:
            t2_order = self.order_factory.limit(
                instrument_id=self._instrument_id,
                order_side=OrderSide.SELL,
                quantity=Quantity.from_int(t2_qty),
                price=Price(self._round_to_tick(self._target_2), precision=2),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
            self._t2_order_id = t2_order.client_order_id
            self.submit_order(t2_order)

        # Stop-market sell for full position
        stop_order = self.order_factory.stop_market(
            instrument_id=self._instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_int(self._total_contracts),
            trigger_price=Price(self._round_to_tick(self._stop_price), precision=2),
            trigger_type=TriggerType.DEFAULT,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )
        self._stop_order_id = stop_order.client_order_id
        self.submit_order(stop_order)

    def _on_t1_filled(self, event: OrderFilled) -> None:
        """T1 hit → record realized PnL, move stop to breakeven, reduce stop qty."""
        instrument = self.cache.instrument(self._instrument_id)
        filled_qty = int(event.last_qty)
        fill_px = float(event.last_px)
        # Accumulate realized PnL for the T1 partial exit
        self._realized_pnl_pts += (fill_px - self._entry_price) * filled_qty
        self._final_exit_price = fill_px
        self._exit_reason_hint = "T1 target hit"
        self._remaining_contracts -= filled_qty
        self._phase = _Phase.AFTER_T1

        # Move stop to breakeven (production: -3pt = entry - 3pt for long)
        be_buf = getattr(self._exit_params, "breakeven_buffer_pts", 0.0)
        be_stop = self._entry_price + be_buf  # buf is negative for "BE-3"
        self._stop_price = self._round_to_tick(be_stop)

        # If T1 closed the whole position (rare — only if t1_exit_fraction=1.0)
        if self._remaining_contracts <= 0:
            self._cancel_order_if_active(self._t2_order_id)
            self._cancel_order_if_active(self._stop_order_id)
            self._close_trade(fill_px, event)
            return

        # Cancel and resubmit stop with new qty and price
        self._replace_stop_order(instrument, self._remaining_contracts, self._stop_price)

    def _on_t2_filled(self, event: OrderFilled) -> None:
        """T2 hit → record realized PnL, switch to trailing stop on runner."""
        instrument = self.cache.instrument(self._instrument_id)
        filled_qty = int(event.last_qty)
        fill_px = float(event.last_px)
        self._realized_pnl_pts += (fill_px - self._entry_price) * filled_qty
        self._final_exit_price = fill_px
        self._exit_reason_hint = "T2 target hit"
        self._remaining_contracts -= filled_qty
        self._phase = _Phase.AFTER_T2

        if self._remaining_contracts <= 0:
            self._cancel_order_if_active(self._stop_order_id)
            self._close_trade(fill_px, event)
            return

        # Set initial trailing stop
        trail_stop = self._compute_trail_stop()
        self._stop_price = self._round_to_tick(trail_stop)
        self._replace_stop_order(instrument, self._remaining_contracts, self._stop_price)

    def _on_stop_filled(self, event: OrderFilled) -> None:
        """Stop hit → record realized PnL on remaining contracts, close trade."""
        filled_qty = int(event.last_qty)
        fill_px = float(event.last_px)
        self._realized_pnl_pts += (fill_px - self._entry_price) * filled_qty
        self._final_exit_price = fill_px
        # Distinguish initial stop (loss) vs trailed stop (runner profit-stop)
        if self._phase == _Phase.AFTER_T2:
            self._exit_reason_hint = "Runner trailing stop"
        elif self._phase == _Phase.AFTER_T1:
            self._exit_reason_hint = "Breakeven stop after T1"
        else:
            self._exit_reason_hint = "Stop loss hit"
        self._remaining_contracts -= filled_qty

        # Cancel any remaining limit orders
        self._cancel_order_if_active(self._t1_order_id)
        self._cancel_order_if_active(self._t2_order_id)

        self._close_trade(fill_px, event)

    def _replace_stop_order(self, instrument, new_qty: int, new_price: float) -> None:
        """Cancel existing stop and submit a new one."""
        if new_qty <= 0:
            return

        # Cancel old stop
        self._cancel_order_if_active(self._stop_order_id)

        # Submit new stop
        stop_order = self.order_factory.stop_market(
            instrument_id=self._instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_int(new_qty),
            trigger_price=Price(self._round_to_tick(new_price), precision=2),
            trigger_type=TriggerType.DEFAULT,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )
        self._stop_order_id = stop_order.client_order_id
        self.submit_order(stop_order)

    def _cancel_order_if_active(self, order_id: Optional[ClientOrderId]) -> None:
        """Cancel an order if it exists and is still open."""
        if order_id is None:
            return
        order = self.cache.order(order_id)
        if order is not None and order.is_open:
            self.cancel_order(order)

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------

    def _update_trailing_stop(self, instrument) -> None:
        """Tighten trailing stop based on profit thresholds."""
        new_trail = self._compute_trail_stop()
        new_price = self._round_to_tick(new_trail)
        if new_price > self._stop_price:
            self._stop_price = new_price
            self._replace_stop_order(instrument, self._remaining_contracts, self._stop_price)

    def _compute_trail_stop(self) -> float:
        """Dynamic trailing stop distance based on profit."""
        profit = self._highest_since_entry - self._entry_price
        trail_pts = self._exit_params.trailing_stop_pts  # default 4.0

        for threshold, tighter in self._exit_params.trailing_tighten_thresholds:
            if profit >= threshold:
                trail_pts = tighter

        return self._highest_since_entry - trail_pts

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def _close_trade(self, exit_price: float, event: OrderFilled) -> None:
        """Build TradeRecord from accumulated realized PnL and reset state."""
        exit_time = pd.Timestamp(event.ts_event, unit="ns",
                                 tz="UTC").tz_convert("US/Eastern").to_pydatetime()

        # PnL pts is the running accumulator built up across T1/T2/stop fills,
        # divided by total contracts to get per-contract pts (matching the
        # backtest/runner.py legacy convention of pts as per-contract).
        if self._total_contracts > 0:
            pnl_pts_per_contract = self._realized_pnl_pts / self._total_contracts
        else:
            pnl_pts_per_contract = 0.0
        # Total dollar PnL: realized_pts_total × point_value − commissions
        point_value = self._contract_spec.point_value
        pnl_dollars = (self._realized_pnl_pts * point_value) - self._total_commission

        record = TradeRecord(
            entry_time=self._entry_time or exit_time,
            exit_time=exit_time,
            entry_price=self._entry_price,
            avg_exit_price=self._final_exit_price or exit_price,
            contracts=self._total_contracts,
            pnl_pts=pnl_pts_per_contract,
            pnl_dollars=pnl_dollars,
            pattern_type=self._pattern_type,
            exit_reason=self._exit_reason_hint,
        )
        self._completed_trades.append(record)
        # Best-effort sync with PositionManager (for daily PnL tracking)
        if self._pos_mgr.session and self._pos_mgr.session.active_position is not None:
            try:
                self._pos_mgr.close_position(
                    exit_price=self._final_exit_price or exit_price,
                    timestamp=exit_time,
                    exit_reason=self._exit_reason_hint,
                    pattern_type=self._pattern_type,
                )
            except Exception:
                pass

        self._reset_position_state()

    def _reset_position_state(self) -> None:
        """Reset all position tracking for next trade."""
        self._phase = _Phase.FLAT
        self._entry_price = 0.0
        self._stop_price = 0.0
        self._target_1 = 0.0
        self._target_2 = 0.0
        self._total_contracts = 0
        self._remaining_contracts = 0
        self._highest_since_entry = 0.0
        self._pattern_type = ""
        self._entry_time = None
        self._total_commission = 0.0
        self._entry_order_id = None
        self._t1_order_id = None
        self._t2_order_id = None
        self._stop_order_id = None
        # Realized-PnL accumulator and exit metadata
        self._realized_pnl_pts = 0.0
        self._final_exit_price = 0.0
        self._exit_reason_hint = "Stop loss hit"

    # ------------------------------------------------------------------
    # Position closed event (catch-all cleanup)
    # ------------------------------------------------------------------

    def on_position_closed(self, event) -> None:
        """Ensure cleanup when Nautilus reports position closed."""
        # Cancel any straggler orders
        self._cancel_order_if_active(self._t1_order_id)
        self._cancel_order_if_active(self._t2_order_id)
        self._cancel_order_if_active(self._stop_order_id)

        if self._phase != _Phase.FLAT:
            self._reset_position_state()

    def on_stop(self) -> None:
        self.cancel_all_orders(self._instrument_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _round_to_tick(self, price: float) -> float:
        """Round price to nearest ES tick (0.25)."""
        return round(price / 0.25) * 0.25

    def _bars_to_dataframe(self) -> pd.DataFrame:
        """Convert accumulated bars to DataFrame."""
        if not self._bars:
            return pd.DataFrame()
        df = pd.DataFrame(self._bars)
        df.index = pd.DatetimeIndex(df.pop("timestamp"))
        return df

    def _prior_day_to_dataframe(self) -> Optional[pd.DataFrame]:
        """Reconstruct prior-day DataFrame from config dict."""
        if self._prior_day_data is None:
            return None
        data = self._prior_day_data
        if "bars" not in data:
            return None
        bars = data["bars"]
        df = pd.DataFrame(bars)
        if "timestamp" in df.columns:
            df.index = pd.DatetimeIndex(df.pop("timestamp"))
        return df

    @property
    def completed_trades(self) -> list[TradeRecord]:
        """All completed trade records."""
        return list(self._completed_trades)
