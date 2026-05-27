"""NautilusTrader Strategy wrapper for the Mancini Method.

Delegates ALL exit decisions to ``strategy.exit_manager.ExitManager`` — the
authoritative implementation of Mancini's three-stage 75/15/10 exit ladder
(T1 → T2 → structure-trailed runner) with multi-session hold for the 10%
post-T2 slice. The Nautilus strategy here is the thin event-driven adapter:

  * On each bar we feed (high, low, close) to ``ExitManager.update()`` and
    translate any returned ``ExitAction`` into Nautilus order ops:
      - T1 / T2 / stop close   → market SELL for ``contracts_to_close``
      - stop migration only    → cancel + resubmit stop at ``new_stop``
  * The original stop-market (placed at entry) is the safety net that
    catches stop-outs Nautilus might fill faster than our bar-level check.
  * EOD logic mirrors live/ib_runner.py:
      - INITIAL  / AFTER_T1: flatten at EOD (15:55 ET) every day.
      - AFTER_T2: hold across EOD when ``multi_session_runner=True`` and
        the survived-sessions counter is below the safety cap; otherwise
        flatten with reason ``eod_flatten_max_days``. At each EOD we still
        call ``update_prior_day_low()`` so the structural trail ratchets
        under each session's low.

This eliminates the previous duplicate exit state machine (the old
``_Phase`` enum, ``_compute_trail_stop``, and per-fill state mutations) so
Nautilus backtests pick up every Mancini exit refinement automatically.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
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
from strategy.exit_manager import ExitAction, ExitManager, ExitPhase, TradePosition
from strategy.position_manager import PositionManager, TradeRecord
from strategy.risk_manager import RiskManager


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
    """NautilusTrader strategy that delegates exit decisions to ExitManager.

    Order lifecycle:
        on_bar  → SignalAggregator detects entry → market BUY
        on_order_filled (entry) → create TradePosition + place safety stop
        on_bar  (position open) → ExitManager.update() returns ExitAction:
            * contracts_to_close > 0 → market SELL (T1, T2, or stop)
            * stop migration         → replace safety stop order
        EOD (15:55 ET):
            * INITIAL/AFTER_T1 → flatten
            * AFTER_T2 + multi_session_runner → hold + update PDL trail
            * cap hit → force-flatten
    """

    def __init__(self, config: ManciniNautilusConfig) -> None:
        super().__init__(config)

        sp, ep, xp, rp = _reconstruct_params(config)

        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._bar_type_str = config.bar_type
        self._strategy_params = sp
        self._exit_params = xp
        self._risk_params = rp
        self._contract_spec = DEFAULT_CONTRACT
        self._session_times = DEFAULT_SESSION
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
        # Single authoritative exit decision engine.
        self._exit_mgr = ExitManager(
            params=xp, contract=DEFAULT_CONTRACT, strategy_params=sp,
        )

        # Bar accumulation
        self._bars: list[dict] = []
        self._velocity_deque: deque[float] = deque(maxlen=5)

        # Active position: TradePosition is the single source of exit state.
        # When None, we're flat. When set, ExitManager drives all exits.
        self._position: Optional[TradePosition] = None
        # Trade metadata kept alongside the position (Nautilus needs these
        # for the close-trade record, but they don't belong on TradePosition).
        self._pattern_type: str = ""
        self._entry_time: Optional[datetime] = None
        self._total_commission: float = 0.0
        # Realized PnL accumulator (points * contracts, sign-aware via direction).
        # Updated on every partial fill so the final TradeRecord reflects the
        # actual round-trip P&L including slippage.
        self._realized_pnl_pts: float = 0.0
        self._final_exit_price: float = 0.0
        self._exit_reason_hint: str = "Stop loss hit"

        # Day-rollover / EOD bookkeeping.
        # We detect day boundary by comparing the bar's ET date against the
        # previous bar's. ``_runner_sessions_held`` tracks how many sessions
        # an AFTER_T2 runner has survived (for the max-days safety cap),
        # mirroring live/ib_runner.py._runner_sessions_held.
        self._current_session_date: Optional[date] = None
        self._runner_sessions_held: int = 0
        self._session_low: float = float("inf")
        self._session_high: float = 0.0
        self._eod_processed_for_date: Optional[date] = None

        # Order ID tracking. Only the entry market + the safety stop are
        # placed up-front; exits are fired as market orders from on_bar
        # when ExitManager decides.
        self._entry_order_id: Optional[ClientOrderId] = None
        self._stop_order_id: Optional[ClientOrderId] = None
        # Reduce-only market exits we've issued from the bar loop. Tracked
        # so on_order_filled can credit them as exit fills (rather than
        # accidental entries).
        self._pending_exit_order_ids: set[ClientOrderId] = set()

        # In-flight signal / entry — stashed between submit_order(entry)
        # and the subsequent OrderFilled event so _on_entry_filled has the
        # information it needs to build the TradePosition.
        self._pending_signal = None
        self._pending_entry = None

        # Completed trades
        self._completed_trades: list[TradeRecord] = []

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

        # Track session high/low for the PDL trail on cross-session runners.
        if lo < self._session_low:
            self._session_low = lo
        if h > self._session_high:
            self._session_high = h

        # Day-rollover detection: when the bar's date moves forward we hand
        # off the just-finished session's low/high to the runner (if any)
        # and bump the survived-sessions counter for AFTER_T2 holds.
        bar_date = ts.date()
        if self._current_session_date is None:
            self._current_session_date = bar_date
        elif bar_date != self._current_session_date:
            self._on_session_rollover(prior_date=self._current_session_date)
            self._current_session_date = bar_date

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

        # ── Exit path: position open → let ExitManager drive ──────────
        if self._position is not None and self._position.is_open:
            self._process_exit_bar(bar, h, lo, c, ts, instrument)
            # If the bar closed the position, fall through to EOD check
            # only when explicitly flat; otherwise we're done for the bar.
            if self._position is not None and self._position.is_open:
                self._maybe_handle_eod(ts, c, instrument)
                return
            # else position was just closed by the exit; fall through to
            # allow re-entry on a fresh signal if same bar generates one
            # (rare in Mancini, but mancini_long.py allows it).

        if self._pos_mgr.is_done_for_day:
            return

        # ── Entry path ────────────────────────────────────────────────
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

        # Stash the in-flight signal for use in _on_entry_filled.
        self._pending_signal = signal
        self._pending_entry = entry
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
    # Exit driver — ExitManager translates ExitActions to Nautilus ops.
    # ------------------------------------------------------------------

    def _process_exit_bar(
        self,
        bar,
        high: float,
        low: float,
        close: float,
        ts: datetime,
        instrument,
    ) -> None:
        """Feed the bar to ExitManager and act on any returned ExitAction.

        ExitAction shape (see strategy/exit_manager.py):
          * contracts_to_close > 0 → fire market SELL/BUY for that qty
          * new_stop > 0 (different from current stop on the order) →
            replace the safety stop order at that price
          * reason → use as exit-reason hint for the eventual TradeRecord

        ExitManager may also mutate ``position.stop_price`` without
        returning an action (e.g. structure-trail ratchet between T2
        and the stop being hit). We compare against the order's stored
        trigger to detect those cases too.
        """
        pos = self._position
        prev_stop = pos.stop_price
        action = self._exit_mgr.update(pos, high, low, close)

        # Always reconcile the safety stop with the position's current
        # stop_price, even when no action returned (structure trail
        # ratchet only updates the field).
        if action is not None:
            if action.contracts_to_close > 0:
                self._submit_market_exit(
                    action.contracts_to_close,
                    action.exit_price,
                    action.reason,
                    pos.direction,
                )
            new_stop = action.new_stop if action.new_stop > 0 else pos.stop_price
            if new_stop != prev_stop and pos.is_open:
                self._replace_stop_order(instrument, pos.remaining_contracts, new_stop)
        elif pos.stop_price != prev_stop and pos.is_open:
            # Structure-trail-only ratchet (no scale-out fired).
            self._replace_stop_order(instrument, pos.remaining_contracts, pos.stop_price)

    def _submit_market_exit(
        self,
        contracts: int,
        intended_price: float,
        reason: str,
        direction: str,
    ) -> None:
        """Issue a reduce-only market exit for the given quantity.

        ``intended_price`` is the price the bar-loop logic expected to
        fill at (e.g. T1, T2, stop) — used purely as the exit-reason
        hint. The actual fill from Nautilus may differ (slippage), and
        the realized P&L is updated from the OrderFilled event.
        """
        if contracts <= 0:
            return
        side = OrderSide.SELL if direction == "long" else OrderSide.BUY
        order = self._build_market_order(
            order_side=side, quantity=contracts, reduce_only=True,
        )
        self._pending_exit_order_ids.add(order.client_order_id)
        self._exit_reason_hint = reason
        self._final_exit_price = intended_price
        self._submit(order)

    # ------------------------------------------------------------------
    # Thin wrappers around the Cython-slot Strategy operations.
    # These exist so unit tests can swap the order_factory / submit_order
    # plumbing via subclass / monkey-patch (the Nautilus Strategy base
    # class's attributes are read-only Cython slots).
    # ------------------------------------------------------------------

    def _build_market_order(self, *, order_side, quantity: int, reduce_only: bool):
        return self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=order_side,
            quantity=Quantity.from_int(quantity),
            time_in_force=TimeInForce.FOK,
            reduce_only=reduce_only,
        )

    def _build_stop_market_order(
        self, *, order_side, quantity: int, trigger_price: float, reduce_only: bool,
    ):
        return self.order_factory.stop_market(
            instrument_id=self._instrument_id,
            order_side=order_side,
            quantity=Quantity.from_int(quantity),
            trigger_price=Price(self._round_to_tick(trigger_price), precision=2),
            trigger_type=TriggerType.DEFAULT,
            time_in_force=TimeInForce.GTC,
            reduce_only=reduce_only,
        )

    def _submit(self, order) -> None:
        self.submit_order(order)

    def _cancel(self, order) -> None:
        self.cancel_order(order)

    def _lookup_order(self, order_id: Optional[ClientOrderId]):
        if order_id is None:
            return None
        return self.cache.order(order_id)

    def _lookup_instrument(self):
        return self.cache.instrument(self._instrument_id)

    # ------------------------------------------------------------------
    # Session rollover + EOD handling
    # ------------------------------------------------------------------

    def _on_session_rollover(self, prior_date: date) -> None:
        """Day boundary detected. Bump the runner survived-sessions counter
        for any held position (so max-days cap accumulates), then reset
        session high/low for the new day.

        Mirrors live/ib_runner.py._check_session_rollover for the part that
        matters for exits — we don't need pattern-state snapshotting here
        because Nautilus runs each backtest config as one engine pass.

        When eod_flatten_enabled=False (new default), ANY held position
        bumps the counter — INITIAL/AFTER_T1/AFTER_T2 all participate.
        When eod_flatten_enabled=True (legacy), only AFTER_T2 with
        multi_session_runner=True holds, so only that case bumps.
        """
        pos = self._position
        if pos is not None and pos.is_open:
            eod_flatten_enabled = getattr(
                self._exit_params, "eod_flatten_enabled", False
            )
            multi_session_enabled = getattr(
                self._exit_params, "multi_session_runner", False
            )
            should_bump = (
                (not eod_flatten_enabled) or
                (pos.phase == ExitPhase.AFTER_T2 and multi_session_enabled)
            )
            if should_bump:
                self._runner_sessions_held += 1

        # Reset session extremes — they only ever describe one ET date.
        self._session_low = float("inf")
        self._session_high = 0.0
        self._eod_processed_for_date = None

    def _maybe_handle_eod(self, ts: datetime, close: float, instrument) -> None:
        """If we're past 15:55 ET, run the EOD branch once per session.

        Mirrors live/ib_runner.py._check_eod:

        Default (eod_flatten_enabled=False):
          - INITIAL / AFTER_T1 / AFTER_T2 all HOLD across EOD via their
            existing stops, up to the multi_session_runner_max_days cap.
          - update_prior_day_low() is called so the runner trail ratchets
            for AFTER_T1/AFTER_T2 (no-op for INITIAL).

        Legacy (eod_flatten_enabled=True):
          - INITIAL / AFTER_T1 → flatten.
          - AFTER_T2 + multi_session_runner + under cap → update PDL trail,
            keep position open.
          - AFTER_T2 + multi_session disabled or cap hit → flatten with
            an "eod_flatten_max_days" reason when the cap is the trigger.
        """
        pos = self._position
        if pos is None or not pos.is_open:
            return
        t = ts.time()
        if not self._session_times.past_eod_flatten(t):
            return
        bar_date = ts.date()
        if self._eod_processed_for_date == bar_date:
            return  # already ran for this session

        eod_flatten_enabled = getattr(
            self._exit_params, "eod_flatten_enabled", False
        )
        multi_session_enabled = getattr(
            self._exit_params, "multi_session_runner", False
        )
        max_days = getattr(
            self._exit_params, "multi_session_runner_max_days", 5
        )
        if not eod_flatten_enabled:
            hold_across_eod = self._runner_sessions_held < max_days
        else:
            hold_across_eod = (
                multi_session_enabled
                and pos.phase == ExitPhase.AFTER_T2
                and self._runner_sessions_held < max_days
            )

        if hold_across_eod:
            # Update the structural trail under today's session low.
            # No-op for INITIAL (update_prior_day_low ignores pre-T1 phases).
            if self._session_low < float("inf") and self._session_low > 0:
                action = self._exit_mgr.update_prior_day_low(
                    pos, self._session_low
                )
                if action is not None and action.new_stop > 0 and pos.is_open:
                    self._replace_stop_order(
                        instrument, pos.remaining_contracts, pos.stop_price
                    )
            self._eod_processed_for_date = bar_date
            return

        # Flatten path:
        #   - eod_flatten_enabled=False with max-days cap hit (any phase)
        #   - eod_flatten_enabled=True legacy semantics
        if self._runner_sessions_held >= max_days:
            reason = "EOD flatten (max-days cap)"
        else:
            reason = f"EOD flatten ({pos.phase.name})"
        self._submit_market_exit(
            pos.remaining_contracts, close, reason, pos.direction
        )
        self._eod_processed_for_date = bar_date

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
        elif oid == self._stop_order_id:
            self._on_stop_filled(event)
        elif oid in self._pending_exit_order_ids:
            self._pending_exit_order_ids.discard(oid)
            self._on_exit_fill(event)

    def _on_entry_filled(self, event: OrderFilled) -> None:
        """Create the TradePosition and place the safety stop order."""
        instrument = self.cache.instrument(self._instrument_id)
        entry_px = float(event.last_px)
        self._entry_time = pd.Timestamp(event.ts_event, unit="ns").to_pydatetime()
        signal = self._pending_signal
        entry = self._pending_entry
        self._pattern_type = signal.pattern.pattern_type

        # Build the TradePosition that ExitManager will drive.
        self._position = self._exit_mgr.create_position(
            entry_price=entry_px,
            stop_price=signal.stop_price,
            target_1=signal.target_1,
            target_2=signal.target_2,
            contracts=entry.contracts,
            direction="long",
            is_double_dip=getattr(signal.pattern, "is_double_dip", False),
        )
        # Register with PositionManager for daily trade-count and PnL tracking.
        self._pos_mgr.open_position(
            self._position, self._entry_time, self._pattern_type
        )
        # Reset per-trade accounting
        self._realized_pnl_pts = 0.0
        self._final_exit_price = 0.0
        self._exit_reason_hint = "Stop loss hit"
        self._runner_sessions_held = 0
        self._pending_signal = None
        self._pending_entry = None

        # Safety stop — Nautilus may execute this faster than our bar loop
        # in some markets. ExitManager replaces this stop as price evolves.
        self._submit_safety_stop(instrument)

    def _submit_safety_stop(self, instrument) -> None:
        pos = self._position
        if pos is None or not pos.is_open:
            return
        side = OrderSide.SELL if pos.direction == "long" else OrderSide.BUY
        stop_order = self._build_stop_market_order(
            order_side=side,
            quantity=pos.remaining_contracts,
            trigger_price=pos.stop_price,
            reduce_only=True,
        )
        self._stop_order_id = stop_order.client_order_id
        self._submit(stop_order)

    def _replace_stop_order(self, instrument, new_qty: int, new_price: float) -> None:
        """Cancel existing stop and submit a new one with updated qty/price."""
        if new_qty <= 0:
            return
        self._cancel_order_if_active(self._stop_order_id)
        pos = self._position
        if pos is None or not pos.is_open:
            return
        side = OrderSide.SELL if pos.direction == "long" else OrderSide.BUY
        stop_order = self._build_stop_market_order(
            order_side=side,
            quantity=new_qty,
            trigger_price=new_price,
            reduce_only=True,
        )
        self._stop_order_id = stop_order.client_order_id
        self._submit(stop_order)

    def _cancel_order_if_active(self, order_id: Optional[ClientOrderId]) -> None:
        """Cancel an order if it exists and is still open."""
        order = self._lookup_order(order_id)
        if order is not None and order.is_open:
            self._cancel(order)

    def _on_exit_fill(self, event: OrderFilled) -> None:
        """Reduce-only market exit fill (T1, T2, EOD flatten). Credit PnL."""
        pos = self._position
        if pos is None:
            return
        filled_qty = int(event.last_qty)
        fill_px = float(event.last_px)
        if pos.direction == "short":
            self._realized_pnl_pts += (pos.entry_price - fill_px) * filled_qty
        else:
            self._realized_pnl_pts += (fill_px - pos.entry_price) * filled_qty
        self._final_exit_price = fill_px

        # If this was the final-contracts exit, finalise the trade record.
        if pos.remaining_contracts <= 0:
            self._cancel_order_if_active(self._stop_order_id)
            self._close_trade(fill_px, event)

    def _on_stop_filled(self, event: OrderFilled) -> None:
        """Safety stop hit → close trade. ExitManager would also detect
        this via low<=stop on the bar that triggered it, but the Nautilus
        stop-market may execute first. We mirror that into the
        ExitManager state by zeroing remaining_contracts and setting
        phase=CLOSED so subsequent bars don't try to scale again.
        """
        pos = self._position
        if pos is None:
            return
        filled_qty = int(event.last_qty)
        fill_px = float(event.last_px)
        if pos.direction == "short":
            self._realized_pnl_pts += (pos.entry_price - fill_px) * filled_qty
        else:
            self._realized_pnl_pts += (fill_px - pos.entry_price) * filled_qty
        self._final_exit_price = fill_px
        if pos.phase == ExitPhase.AFTER_T2:
            self._exit_reason_hint = "Runner trailing stop"
        elif pos.phase == ExitPhase.AFTER_T1:
            self._exit_reason_hint = "Breakeven stop after T1"
        else:
            self._exit_reason_hint = "Stop loss hit"
        # Mirror Nautilus's reality onto the TradePosition so ExitManager
        # won't try to scale-out a position we've already flatted.
        pos.remaining_contracts = max(0, pos.remaining_contracts - filled_qty)
        pos.phase = ExitPhase.CLOSED
        self._cancel_order_if_active(self._stop_order_id)
        self._close_trade(fill_px, event)

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def _close_trade(self, exit_price: float, event: OrderFilled) -> None:
        """Build TradeRecord from accumulated realized PnL and reset state."""
        pos = self._position
        if pos is None:
            return
        exit_time = pd.Timestamp(event.ts_event, unit="ns",
                                 tz="UTC").tz_convert("US/Eastern").to_pydatetime()

        total_contracts = pos.total_contracts
        if total_contracts > 0:
            pnl_pts_per_contract = self._realized_pnl_pts / total_contracts
        else:
            pnl_pts_per_contract = 0.0
        point_value = self._contract_spec.point_value
        pnl_dollars = (self._realized_pnl_pts * point_value) - self._total_commission

        record = TradeRecord(
            entry_time=self._entry_time or exit_time,
            exit_time=exit_time,
            entry_price=pos.entry_price,
            avg_exit_price=self._final_exit_price or exit_price,
            contracts=total_contracts,
            pnl_pts=pnl_pts_per_contract,
            pnl_dollars=pnl_dollars,
            pattern_type=self._pattern_type,
            exit_reason=self._exit_reason_hint,
            direction=pos.direction,
            is_runner=pos.t1_hit,
            is_double_dip=pos.is_double_dip,
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
        self._position = None
        self._pattern_type = ""
        self._entry_time = None
        self._total_commission = 0.0
        self._entry_order_id = None
        self._stop_order_id = None
        self._pending_exit_order_ids = set()
        self._realized_pnl_pts = 0.0
        self._final_exit_price = 0.0
        self._exit_reason_hint = "Stop loss hit"
        self._runner_sessions_held = 0

    # ------------------------------------------------------------------
    # Position closed event (catch-all cleanup)
    # ------------------------------------------------------------------

    def on_position_closed(self, event) -> None:
        """Ensure cleanup when Nautilus reports position closed."""
        # Cancel any straggler orders
        self._cancel_order_if_active(self._stop_order_id)
        for oid in list(self._pending_exit_order_ids):
            self._cancel_order_if_active(oid)

        if self._position is not None:
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
