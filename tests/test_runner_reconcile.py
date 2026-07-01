"""Regression tests for the venue T1 + full-close reconcile race that silently
collapsed the 10% runner (trade 622, 2026-07-01).

Sequence that lost the runner:
  1. Venue OCA bracket: TP=3 lots @ T1, SL=4 (reduces to a 1-lot runner on fill).
  2. Price hits T1. _sync_position reads an intermediate/lagged volume (3, not
     the runner's 1), so _reconcile_venue_t1's `ib_volume == expected_runner`
     guard refuses to book — t1 stays UNBOOKED.
  3. IB then reports the position flat (full close) via that TP fill. The
     full-close path books the whole still-recorded 4 lots at the T1 price,
     silently erasing the scale-out and the runner leg.

`plan_full_close_legs` makes that race explicit: a TP-typed full close with T1
never booked must be booked as a T1 leg + a runner leg (so the collapsed runner
is visible), not one lump at T1.
"""
from live.ib_runner import plan_full_close_legs, venue_t1_booking_plan


class TestVenueT1BookingPlan:
    """Book T1 off the TP-order fill confirmation, NOT the laggy position-volume
    read. This is the actual fix for the 622 race: the old guard only booked
    when ib_volume == expected_runner (exactly 1), so a lagged/partial read (3)
    made it refuse and the runner got swallowed by the later full-close. Now T1
    is booked the moment the TP order confirms Filled, using the INTENDED split
    (tp_qty / runner_qty) rather than a lag-derived difference."""

    def test_tp_confirmed_books_intended_split(self):
        # TP order Filled, T1 not yet booked, 4 lots → book 3, hold runner 1.
        assert venue_t1_booking_plan(
            tp_confirmed=True, t1_booked=False,
            total_contracts=4, t1_fraction=0.75) == (True, 3, 1)

    def test_tp_not_confirmed_does_not_book(self):
        # A bare volume dip without a confirmed TP fill must NOT book T1.
        assert venue_t1_booking_plan(
            tp_confirmed=False, t1_booked=False,
            total_contracts=4, t1_fraction=0.75) == (False, 0, 0)

    def test_already_booked_is_idempotent(self):
        assert venue_t1_booking_plan(
            tp_confirmed=True, t1_booked=True,
            total_contracts=4, t1_fraction=0.75) == (False, 0, 0)

    def test_single_contract_has_no_runner_to_hold(self):
        assert venue_t1_booking_plan(
            tp_confirmed=True, t1_booked=False,
            total_contracts=1, t1_fraction=0.75) == (False, 0, 0)

    def test_two_contracts_split_one_and_one(self):
        assert venue_t1_booking_plan(
            tp_confirmed=True, t1_booked=False,
            total_contracts=2, t1_fraction=0.75) == (True, 1, 1)

    def test_booked_plus_held_equals_total(self):
        ok, filled, runner = venue_t1_booking_plan(
            tp_confirmed=True, t1_booked=False,
            total_contracts=4, t1_fraction=0.75)
        assert ok and filled + runner == 4


def test_tp_full_close_with_unbooked_t1_splits_into_t1_and_runner():
    # The 622 race: 4 lots, TP fill, T1 never booked → must split 3 + 1.
    legs = plan_full_close_legs(
        exit_type="TP", t1_booked=False,
        remaining_contracts=4, total_contracts=4, t1_fraction=0.75)
    assert legs == [("t1", 3), ("runner", 1)]


def test_real_stop_out_books_full_size():
    # A genuine SL close before any target — book the whole position, no split.
    legs = plan_full_close_legs(
        exit_type="SL", t1_booked=False,
        remaining_contracts=4, total_contracts=4, t1_fraction=0.75)
    assert legs == [("full", 4)]


def test_runner_hitting_target_after_t1_books_full_remaining():
    # T1 already booked; the 1-lot runner later closes — book it as one leg.
    legs = plan_full_close_legs(
        exit_type="TP", t1_booked=True,
        remaining_contracts=1, total_contracts=4, t1_fraction=0.75)
    assert legs == [("full", 1)]


def test_single_contract_never_splits():
    # 1 contract has no runner (runner_split leaves 0) — never split.
    legs = plan_full_close_legs(
        exit_type="TP", t1_booked=False,
        remaining_contracts=1, total_contracts=1, t1_fraction=0.75)
    assert legs == [("full", 1)]


def test_partial_already_booked_is_not_resplit():
    # remaining < total means some scale-out already booked — don't re-split.
    legs = plan_full_close_legs(
        exit_type="TP", t1_booked=False,
        remaining_contracts=1, total_contracts=4, t1_fraction=0.75)
    assert legs == [("full", 1)]


def test_leg_contracts_sum_to_remaining():
    # Whatever the split, total booked contracts must equal remaining (no P&L
    # created or destroyed — only attribution changes).
    for exit_type in ("TP", "SL"):
        for t1_booked in (True, False):
            legs = plan_full_close_legs(
                exit_type=exit_type, t1_booked=t1_booked,
                remaining_contracts=4, total_contracts=4, t1_fraction=0.75)
            assert sum(qty for _, qty in legs) == 4
