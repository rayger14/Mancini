"""Tests for the venue-side scale-out + marketable-limit entry.

The free workaround for the 12-min data delay: push the exit ladder to the
exchange (which always executes in real time) so the runner can't be stranded
by the delay, and cap entry slippage with a marketable-limit order.

- marketable_limit_price: prices the entry cap_pts through the signal so it
  fills immediately but never worse than cap_pts of adverse slippage.
- build_bracket_orders: parent = marketable LIMIT; TP + SL in one OCA group
  with reduce-on-fill (ocaType=2) so the stop auto-shrinks to the runner.
- classify_position_sync: decides whether a venue-side position change is a
  full close, a T1 partial (runner remains), or no change.
"""
from __future__ import annotations

from live.ib_bridge import (
    build_bracket_orders,
    marketable_limit_price,
    runner_split,
)
from live.ib_runner import classify_position_sync, venue_t1_pnl_and_stop


class TestVenueT1Booking:
    def test_long_pnl_and_breakeven_stop(self):
        pnl, stop = venue_t1_pnl_and_stop(
            direction="long", entry_price=7000.0, fill_price=7012.0, filled=3,
            breakeven_buffer_pts=-3.0, prior_day_low=0.0, pdl_buffer_pts=1.0)
        assert pnl == 36.0          # (7012-7000)*3
        assert stop == 6997.0       # entry - 3

    def test_short_pnl_and_breakeven_stop(self):
        pnl, stop = venue_t1_pnl_and_stop(
            direction="short", entry_price=7553.25, fill_price=7518.5, filled=1,
            breakeven_buffer_pts=-8.0, prior_day_low=0.0, pdl_buffer_pts=1.0)
        assert pnl == 34.75         # (7553.25-7518.5)*1
        assert stop == 7545.25      # entry - 8

    def test_prior_day_low_widens_stop_when_lower(self):
        # A long whose prior-day-low stop is lower than breakeven uses the
        # wider (lower) stop so the runner has room.
        _, stop = venue_t1_pnl_and_stop(
            direction="long", entry_price=7000.0, fill_price=7012.0, filled=3,
            breakeven_buffer_pts=-3.0, prior_day_low=6990.0, pdl_buffer_pts=1.0)
        assert stop == 6989.0       # min(6997, 6990-1)


class TestRunnerSplit:
    def test_four_contracts(self):
        assert runner_split(4, 0.75) == (3, 1)

    def test_two_contracts(self):
        assert runner_split(2, 0.75) == (1, 1)

    def test_one_contract_no_runner(self):
        assert runner_split(1, 0.75) == (1, 0)

    def test_three_contracts(self):
        assert runner_split(3, 0.75) == (2, 1)


class TestMarketableLimitPrice:
    def test_long_pays_up_to_cap_above_signal(self):
        assert marketable_limit_price("long", 7000.0, 3.0) == 7003.0

    def test_short_sells_down_to_cap_below_signal(self):
        assert marketable_limit_price("short", 7553.25, 4.0) == 7549.25

    def test_zero_cap_is_the_signal_price(self):
        assert marketable_limit_price("long", 7000.0, 0.0) == 7000.0


class TestBuildBracketOrders:
    def _build(self, quantity=4, direction="long", tp_fraction=0.75):
        return build_bracket_orders(
            parent_id=10, tp_id=11, sl_id=12,
            quantity=quantity, tp_fraction=tp_fraction,
            entry_limit_price=7003.0, sl=6980.0, tp=7030.0,
            direction=direction, comment="Mancini:FB",
        )

    def test_parent_is_marketable_limit(self):
        parent, _, _ = self._build()
        assert parent.orderType == "LMT"
        assert parent.lmtPrice == 7003.0
        assert parent.totalQuantity == 4
        assert parent.action == "BUY"

    def test_none_limit_price_falls_back_to_market(self):
        parent, _, _ = build_bracket_orders(
            parent_id=1, tp_id=2, sl_id=3, quantity=2, tp_fraction=0.75,
            entry_limit_price=None, sl=6980.0, tp=7030.0,
            direction="long", comment="x",
        )
        assert parent.orderType == "MKT"

    def test_short_flips_actions(self):
        parent, tp, sl = self._build(direction="short")
        assert parent.action == "SELL"
        assert tp.action == "BUY"
        assert sl.action == "BUY"

    def test_tp_is_fraction_sl_is_full(self):
        _, tp, sl = self._build(quantity=4)
        assert tp.totalQuantity == 3      # floor(4*0.75)
        assert sl.totalQuantity == 4      # full position

    def test_runner_split_two_contracts(self):
        _, tp, sl = self._build(quantity=2)
        assert tp.totalQuantity == 1      # leaves a 1-ct runner
        assert sl.totalQuantity == 2

    def test_one_contract_no_runner(self):
        _, tp, sl = self._build(quantity=1)
        assert tp.totalQuantity == 1
        assert sl.totalQuantity == 1

    def test_tp_and_sl_share_oca_group_with_reduce(self):
        _, tp, sl = self._build()
        assert tp.ocaGroup == sl.ocaGroup
        assert tp.ocaGroup  # non-empty
        assert tp.ocaType == 2   # reduce-with-block: SL shrinks when TP fills
        assert sl.ocaType == 2

    def test_children_are_parented_and_priced(self):
        _, tp, sl = self._build()
        assert tp.parentId == 10
        assert sl.parentId == 10
        assert tp.lmtPrice == 7030.0
        assert sl.auxPrice == 6980.0  # StopOrder stores stop as auxPrice

    def test_prices_are_tick_rounded(self):
        parent, tp, sl = build_bracket_orders(
            parent_id=1, tp_id=2, sl_id=3, quantity=2, tp_fraction=0.75,
            entry_limit_price=7003.10, sl=6980.30, tp=7030.40,
            direction="long", comment="x",
        )
        assert parent.lmtPrice == 7003.0
        assert sl.auxPrice == 6980.25
        assert tp.lmtPrice == 7030.5


class TestClassifyPositionSync:
    def test_none_is_full_close(self):
        assert classify_position_sync(local_remaining=4, ib_volume=None,
                                      t1_booked=False) == "full_close"

    def test_zero_is_full_close(self):
        assert classify_position_sync(4, 0, False) == "full_close"

    def test_reduced_before_t1_is_venue_partial(self):
        # 4 -> 1 with T1 not yet booked: the venue filled the TP fraction,
        # a runner remains. Book the partial, keep the runner.
        assert classify_position_sync(4, 1, False) == "venue_t1_partial"

    def test_reduced_after_t1_is_no_change(self):
        # T1 already handled; a later reduction is the runner's own business,
        # resolved by the full-close path when it hits zero.
        assert classify_position_sync(1, 1, True) == "no_change"

    def test_unchanged_is_no_change(self):
        assert classify_position_sync(4, 4, False) == "no_change"

    def test_larger_than_expected_is_no_change(self):
        # Defensive: never act on an unexpected increase.
        assert classify_position_sync(2, 4, False) == "no_change"
