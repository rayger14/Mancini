"""Unit tests for ManciniNautilusStrategy's exit delegation to ExitManager.

The Nautilus strategy in ``backtest/nautilus_strategy.py`` no longer owns its
own exit state machine — it delegates every decision to
``strategy.exit_manager.ExitManager`` and translates the returned
``ExitAction`` objects into Nautilus order operations. These tests verify
that translation:

  * T1 hit          → market SELL fired with the T1 fraction, stop migrated
  * T2 hit (post-T1)→ market SELL fired with the T2 fraction, structure
                       trail engaged
  * Structure trail → stop ratchets up as new swing lows form
  * Multi-session   → AFTER_T2 survives EOD, AFTER_T1 still flattens, the
                       max-days cap force-flattens
  * T1-only path    → with t2_exit_fraction=0 the position holds at AFTER_T1

The Nautilus Strategy base class is Cython and exposes ``order_factory``,
``submit_order``, ``cancel_order``, and ``cache`` as read-only slots, so we
can't monkey-patch those directly on an instance. Instead we subclass
ManciniNautilusStrategy with ``_build_market_order``, ``_build_stop_market_order``,
``_submit``, ``_cancel``, ``_lookup_order``, ``_lookup_instrument`` overridden
to return / record fakes. The underlying ExitManager is the real one so the
three-stage logic exercised is identical to production.
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


try:
    from backtest.nautilus_strategy import ManciniNautilusStrategy
    from strategy.exit_manager import ExitAction, ExitManager, ExitPhase, TradePosition
    nautilus_available = True
except ImportError:
    nautilus_available = False

pytestmark = pytest.mark.skipif(
    not nautilus_available,
    reason="nautilus_trader not installed",
)

from config.settings import (
    DEFAULT_CONTRACT,
    DEFAULT_SESSION,
    DEFAULT_STRATEGY,
    ExitParams,
    StrategyParams,
)


# ---------------------------------------------------------------------------
# Test subclass: replace the Nautilus-slot operations with recorders.
# ---------------------------------------------------------------------------


class _FakeOrder:
    """Lightweight stand-in for a Nautilus Order with the fields the
    strategy actually reads: client_order_id, quantity, trigger_price,
    is_open. ``reduce_only`` is recorded for assertions."""

    __slots__ = (
        "client_order_id",
        "order_side",
        "quantity",
        "trigger_price",
        "reduce_only",
        "is_open",
    )

    def __init__(
        self,
        client_order_id: str,
        order_side,
        quantity: int,
        trigger_price: float = 0.0,
        reduce_only: bool = False,
    ):
        self.client_order_id = client_order_id
        self.order_side = order_side
        self.quantity = quantity
        self.trigger_price = trigger_price
        self.reduce_only = reduce_only
        self.is_open = True


if nautilus_available:

    class _RecordingStrategy(ManciniNautilusStrategy):
        """Subclass that records every order op without hitting Nautilus."""

        def _init_test_state(self):
            self._submitted_market_orders: list[_FakeOrder] = []
            self._submitted_stop_orders: list[_FakeOrder] = []
            self._cancelled_order_ids: list[str] = []
            self._next_order_seq: int = 0
            # Mirrors of submitted orders by id for _lookup_order.
            self._order_book: dict[str, _FakeOrder] = {}

        def _next_id(self) -> str:
            self._next_order_seq += 1
            return f"ORD-{self._next_order_seq}"

        def _build_market_order(self, *, order_side, quantity, reduce_only):
            order = _FakeOrder(
                client_order_id=self._next_id(),
                order_side=order_side,
                quantity=int(quantity),
                reduce_only=reduce_only,
            )
            self._submitted_market_orders.append(order)
            self._order_book[order.client_order_id] = order
            return order

        def _build_stop_market_order(
            self, *, order_side, quantity, trigger_price, reduce_only,
        ):
            order = _FakeOrder(
                client_order_id=self._next_id(),
                order_side=order_side,
                quantity=int(quantity),
                trigger_price=float(trigger_price),
                reduce_only=reduce_only,
            )
            self._submitted_stop_orders.append(order)
            self._order_book[order.client_order_id] = order
            return order

        def _submit(self, order) -> None:
            # No-op — order recorded at build time.
            pass

        def _cancel(self, order) -> None:
            self._cancelled_order_ids.append(order.client_order_id)
            order.is_open = False

        def _lookup_order(self, order_id):
            if order_id is None:
                return None
            return self._order_book.get(order_id)

        def _lookup_instrument(self):
            return SimpleNamespace(symbol="ES")

else:
    _RecordingStrategy = None  # type: ignore


def _make_strategy_stub(
    *,
    exit_params: ExitParams | None = None,
    strategy_params: StrategyParams | None = None,
):
    """Build a _RecordingStrategy bypassing Strategy.__init__.

    We only wire the attributes the exit-path methods touch.
    """
    ep = exit_params or ExitParams()
    sp = strategy_params or DEFAULT_STRATEGY

    strat = _RecordingStrategy.__new__(_RecordingStrategy)
    strat._init_test_state()
    strat._exit_params = ep
    strat._strategy_params = sp
    strat._contract_spec = DEFAULT_CONTRACT
    strat._session_times = DEFAULT_SESSION
    strat._min_rr_ratio = 1.5
    strat._prior_day_data = None

    strat._exit_mgr = ExitManager(
        params=ep, contract=DEFAULT_CONTRACT, strategy_params=sp,
    )
    strat._instrument_id = SimpleNamespace(symbol="ES")

    # Position / day tracking
    strat._position = None
    strat._pattern_type = ""
    strat._entry_time = None
    strat._total_commission = 0.0
    strat._realized_pnl_pts = 0.0
    strat._final_exit_price = 0.0
    strat._exit_reason_hint = "Stop loss hit"
    strat._current_session_date = None
    strat._runner_sessions_held = 0
    strat._session_low = float("inf")
    strat._session_high = 0.0
    strat._eod_processed_for_date = None
    strat._entry_order_id = None
    strat._stop_order_id = None
    strat._pending_exit_order_ids = set()
    strat._completed_trades = []
    strat._pos_mgr = SimpleNamespace(session=None)

    return strat


def _open_long_position(
    strat,
    *,
    entry_price: float = 5800.0,
    stop_price: float = 5790.0,
    target_1: float = 5810.0,
    target_2: float = 5825.0,
    contracts: int = 20,
):
    """Open a long TradePosition on the strategy stub and place safety stop."""
    pos = strat._exit_mgr.create_position(
        entry_price=entry_price,
        stop_price=stop_price,
        target_1=target_1,
        target_2=target_2,
        contracts=contracts,
        direction="long",
    )
    strat._position = pos
    strat._pattern_type = "TestFB"
    strat._entry_time = datetime(2026, 5, 12, 9, 35)
    strat._submit_safety_stop(instrument=None)
    return pos


def _reduce_only_market_sells(strat) -> list[_FakeOrder]:
    return [o for o in strat._submitted_market_orders if o.reduce_only]


# ---------------------------------------------------------------------------
# T1 scale-out
# ---------------------------------------------------------------------------


class TestT1ScaleOut:
    """T1 fires → market SELL for t1_fraction, stop migrates."""

    def test_t1_fires_market_sell_75pct(self):
        strat = _make_strategy_stub()
        pos = _open_long_position(strat, contracts=20)

        # Bar that prints through T1 (5810): high=5811, low=5805
        strat._process_exit_bar(
            bar=None, high=5811.0, low=5805.0, close=5810.5,
            ts=datetime(2026, 5, 12, 9, 40), instrument=None,
        )

        sells = _reduce_only_market_sells(strat)
        assert len(sells) == 1
        assert sells[0].quantity == 15  # 75% of 20
        assert pos.phase == ExitPhase.AFTER_T1
        assert pos.remaining_contracts == 5
        # Stop should have been replaced — old one cancelled, new one submitted
        assert len(strat._submitted_stop_orders) >= 2

    def test_t1_only_path_when_t2_fraction_zero(self):
        """When t2_exit_fraction is 0, T2 phase still flips but no further
        market sell is fired — runner stays at 25% under the legacy trail.
        This is the t2-disabled ablation path."""
        ep = ExitParams(
            t1_exit_fraction=0.75,
            t2_exit_fraction=0.0,
            runner_fraction=0.25,
            structure_trail_enabled=False,  # use legacy fixed trail
        )
        strat = _make_strategy_stub(exit_params=ep)
        pos = _open_long_position(strat, contracts=20)

        # Hit T1
        strat._process_exit_bar(
            bar=None, high=5811.0, low=5805.0, close=5810.5,
            ts=datetime(2026, 5, 12, 9, 40), instrument=None,
        )
        assert pos.phase == ExitPhase.AFTER_T1
        assert pos.remaining_contracts == 5  # 25% runner left

        # Drive price through T2 — phase should flip but no extra market sell.
        sells_before = len(_reduce_only_market_sells(strat))
        strat._process_exit_bar(
            bar=None, high=5830.0, low=5820.0, close=5828.0,
            ts=datetime(2026, 5, 12, 9, 45), instrument=None,
        )
        sells_after = len(_reduce_only_market_sells(strat))
        assert sells_after == sells_before
        assert pos.remaining_contracts == 5


# ---------------------------------------------------------------------------
# T2 scale-out
# ---------------------------------------------------------------------------


class TestT2ScaleOut:
    """T2 fires (after T1) → market SELL for t2_fraction, structure trail."""

    def test_t2_sells_15pct_after_t1(self):
        strat = _make_strategy_stub()
        pos = _open_long_position(strat, contracts=20)

        # Hit T1
        strat._process_exit_bar(
            bar=None, high=5811.0, low=5805.0, close=5810.5,
            ts=datetime(2026, 5, 12, 9, 40), instrument=None,
        )
        # Hit T2
        strat._process_exit_bar(
            bar=None, high=5826.0, low=5820.0, close=5825.5,
            ts=datetime(2026, 5, 12, 9, 50), instrument=None,
        )

        sells = _reduce_only_market_sells(strat)
        assert len(sells) == 2
        assert sells[0].quantity == 15  # T1 (75% of 20)
        assert sells[1].quantity == 3   # T2 (15% of 20)
        assert pos.phase == ExitPhase.AFTER_T2
        assert pos.remaining_contracts == 2  # 10% runner


# ---------------------------------------------------------------------------
# Structure trail ratchet
# ---------------------------------------------------------------------------


class TestStructureTrail:
    """Post-T2 structure trail ratchets the stop up under new swing lows."""

    def test_stop_ratchets_up_on_new_swing_low(self):
        strat = _make_strategy_stub()
        pos = _open_long_position(strat, contracts=20)

        # T1 + T2 to get into AFTER_T2.
        strat._process_exit_bar(
            bar=None, high=5811.0, low=5805.0, close=5810.5,
            ts=datetime(2026, 5, 12, 9, 40), instrument=None,
        )
        strat._process_exit_bar(
            bar=None, high=5826.0, low=5820.0, close=5825.5,
            ts=datetime(2026, 5, 12, 9, 50), instrument=None,
        )
        assert pos.phase == ExitPhase.AFTER_T2
        stop_before_swings = pos.stop_price

        # Build a swing low at 5831 in a window of 11 bars.
        # ExitManager._find_recent_swing_low needs `2*order+1` bars where
        # the candidate is order-bars from each end of its window.
        # With order=3 the candidate must be at index 3 or deeper, with
        # at least 3 bars above on each side.
        bar_times = [datetime(2026, 5, 12, 10, i) for i in range(11)]
        levels = [5835, 5834, 5833, 5832, 5831, 5832, 5833, 5834, 5835, 5836, 5837]
        for ts, lo in zip(bar_times, levels):
            strat._process_exit_bar(
                bar=None, high=lo + 2.0, low=lo, close=lo + 1.0,
                ts=ts, instrument=None,
            )

        buf = strat._exit_params.structure_trail_buffer_pts
        # The swing low identified should be one of {5831, 5832} depending
        # on which qualifies first in the newest-first scan. Stop should
        # be at swing_low - buf in either case.
        assert pos.stop_price >= 5831.0 - buf, (
            f"Expected stop≥{5831.0 - buf}, got {pos.stop_price}"
        )
        assert pos.stop_price > stop_before_swings


# ---------------------------------------------------------------------------
# Multi-session runner hold
# ---------------------------------------------------------------------------


class TestMultiSessionRunner:
    """AFTER_T2 holds across EOD; AFTER_T1 still flattens."""

    def test_after_t2_survives_eod_with_flag_on(self):
        ep = ExitParams(multi_session_runner=True, multi_session_runner_max_days=5)
        strat = _make_strategy_stub(exit_params=ep)
        pos = _open_long_position(strat, contracts=20)

        # Drive to AFTER_T2
        strat._process_exit_bar(
            bar=None, high=5811.0, low=5805.0, close=5810.5,
            ts=datetime(2026, 5, 12, 9, 40), instrument=None,
        )
        strat._process_exit_bar(
            bar=None, high=5826.0, low=5820.0, close=5825.5,
            ts=datetime(2026, 5, 12, 9, 50), instrument=None,
        )
        assert pos.phase == ExitPhase.AFTER_T2
        runners_before = pos.remaining_contracts

        # Track session low so the EOD trail has something to update.
        strat._session_low = 5818.0
        strat._session_high = 5828.0

        # Bar at 16:00 ET (past eod_flatten_time=15:55) should NOT flatten.
        eod_ts = datetime(2026, 5, 12, 16, 0)
        strat._maybe_handle_eod(eod_ts, close=5825.0, instrument=None)

        sells = _reduce_only_market_sells(strat)
        assert len(sells) == 2  # T1 + T2 only — no EOD flatten
        assert pos.is_open
        assert pos.remaining_contracts == runners_before

    def test_after_t1_flattens_at_eod_even_when_multi_session_on(self):
        ep = ExitParams(multi_session_runner=True, multi_session_runner_max_days=5)
        strat = _make_strategy_stub(exit_params=ep)
        pos = _open_long_position(strat, contracts=20)

        # AFTER_T1 only
        strat._process_exit_bar(
            bar=None, high=5811.0, low=5805.0, close=5810.5,
            ts=datetime(2026, 5, 12, 9, 40), instrument=None,
        )
        assert pos.phase == ExitPhase.AFTER_T1

        eod_ts = datetime(2026, 5, 12, 16, 0)
        strat._session_low = 5800.0
        strat._maybe_handle_eod(eod_ts, close=5808.0, instrument=None)

        sells = _reduce_only_market_sells(strat)
        assert len(sells) == 2  # T1 + EOD flatten
        assert sells[1].quantity == 5  # 25% remaining after T1 flushed

    def test_max_days_cap_force_flattens(self):
        ep = ExitParams(multi_session_runner=True, multi_session_runner_max_days=2)
        strat = _make_strategy_stub(exit_params=ep)
        pos = _open_long_position(strat, contracts=20)

        # Drive to AFTER_T2
        strat._process_exit_bar(
            bar=None, high=5811.0, low=5805.0, close=5810.5,
            ts=datetime(2026, 5, 12, 9, 40), instrument=None,
        )
        strat._process_exit_bar(
            bar=None, high=5826.0, low=5820.0, close=5825.5,
            ts=datetime(2026, 5, 12, 9, 50), instrument=None,
        )
        # Pretend the runner has already survived max_days sessions.
        strat._runner_sessions_held = 2
        strat._session_low = 5820.0

        eod_ts = datetime(2026, 5, 12, 16, 0)
        strat._maybe_handle_eod(eod_ts, close=5825.0, instrument=None)

        sells = _reduce_only_market_sells(strat)
        assert len(sells) == 3  # T1 + T2 + force-flatten
        assert sells[2].quantity == 2  # the 10% runner being flatted

    def test_session_rollover_bumps_runner_counter(self):
        ep = ExitParams(multi_session_runner=True, multi_session_runner_max_days=5)
        strat = _make_strategy_stub(exit_params=ep)
        pos = _open_long_position(strat, contracts=20)

        # Force to AFTER_T2
        strat._process_exit_bar(
            bar=None, high=5811.0, low=5805.0, close=5810.5,
            ts=datetime(2026, 5, 12, 9, 40), instrument=None,
        )
        strat._process_exit_bar(
            bar=None, high=5826.0, low=5820.0, close=5825.5,
            ts=datetime(2026, 5, 12, 9, 50), instrument=None,
        )

        assert strat._runner_sessions_held == 0
        strat._on_session_rollover(prior_date=date(2026, 5, 12))
        assert strat._runner_sessions_held == 1
        # Session extremes reset
        assert strat._session_low == float("inf")
        assert strat._session_high == 0.0

    def test_session_rollover_does_not_bump_when_flag_off(self):
        ep = ExitParams(multi_session_runner=False)
        strat = _make_strategy_stub(exit_params=ep)
        pos = _open_long_position(strat, contracts=20)

        strat._process_exit_bar(
            bar=None, high=5811.0, low=5805.0, close=5810.5,
            ts=datetime(2026, 5, 12, 9, 40), instrument=None,
        )
        strat._process_exit_bar(
            bar=None, high=5826.0, low=5820.0, close=5825.5,
            ts=datetime(2026, 5, 12, 9, 50), instrument=None,
        )
        strat._on_session_rollover(prior_date=date(2026, 5, 12))
        assert strat._runner_sessions_held == 0


# ---------------------------------------------------------------------------
# Stop migration (ExitManager → Nautilus stop order replacement)
# ---------------------------------------------------------------------------


class TestStopMigration:
    """Whenever ExitManager moves the stop, the safety stop order is replaced."""

    def test_stop_replaced_on_t1(self):
        strat = _make_strategy_stub()
        pos = _open_long_position(strat, contracts=20)

        # Initial stop placed at 5790
        initial_stops = list(strat._submitted_stop_orders)
        assert len(initial_stops) == 1
        assert initial_stops[0].trigger_price == 5790.0

        # T1 fires → stop migrates to entry + breakeven_buffer_pts (-3) = 5797
        strat._process_exit_bar(
            bar=None, high=5811.0, low=5805.0, close=5810.5,
            ts=datetime(2026, 5, 12, 9, 40), instrument=None,
        )

        assert initial_stops[0].client_order_id in strat._cancelled_order_ids
        assert len(strat._submitted_stop_orders) == 2
        new_stop = strat._submitted_stop_orders[1]
        assert new_stop.trigger_price == 5797.0
        # New stop covers the remaining 5 contracts
        assert new_stop.quantity == 5
