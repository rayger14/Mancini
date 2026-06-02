"""Tests for the T2 scale-down + structure-based trailing stop.

Two Mancini quotes drive the behaviour under test:

  2025-08-05: "lock in 75% profits at the first level, leave a 25% runner,
    then lock in more at second level up, and let a 10% runner go and
    initiate the trailing stop methodology."

  2025-05-14: "I lock in 75% profits at the first level up, leaving a 25%
    runner. I then lock in more at the 2nd level up, leaving a 10% runner
    to trail, at this point, I typically move my stop up, often to below
    wherever structure is."

Together they imply:
  * T2 must actually scale the position down — not just flip the phase. With
    the default 75/15/10 split, a T2 fire on 4 contracts after T1 has filled
    sells another ~15% (1 contract), leaving the ~10% runner.
  * Post-T2 trail follows market structure: the most recent swing low minus
    a small buffer. It only ratchets UP, never down.
"""

from __future__ import annotations

import pytest

from config.settings import ExitParams, StrategyParams
from strategy.exit_manager import ExitManager, ExitPhase, TradePosition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    *,
    structure_trail_enabled: bool = True,
    structure_trail_buffer_pts: float = 3.0,
    structure_trail_lookback_bars: int = 30,
    structure_trail_swing_order: int = 3,
    t2_exit_fraction: float = 0.15,
    runner_fraction: float = 0.10,
    t1_exit_fraction: float = 0.75,
    trailing_stop_pts: float = 12.0,
) -> ExitManager:
    """Build an ExitManager with the supplied params, defaults match prod."""
    params = ExitParams(
        t1_exit_fraction=t1_exit_fraction,
        t2_exit_fraction=t2_exit_fraction,
        runner_fraction=runner_fraction,
        structure_trail_enabled=structure_trail_enabled,
        structure_trail_buffer_pts=structure_trail_buffer_pts,
        structure_trail_lookback_bars=structure_trail_lookback_bars,
        structure_trail_swing_order=structure_trail_swing_order,
        trailing_stop_pts=trailing_stop_pts,
    )
    return ExitManager(params=params)


def _hit_t1(manager: ExitManager, pos: TradePosition) -> None:
    """Drive the position through T1 so we can isolate T2 behaviour."""
    manager.update(pos, high=pos.target_1 + 1.0, low=pos.target_1 - 1.0,
                   close=pos.target_1)
    assert pos.phase == ExitPhase.AFTER_T1, "T1 must have filled to set up T2"


# ---------------------------------------------------------------------------
# Part 1: T2 scale-down
# ---------------------------------------------------------------------------


class TestPartialFillRounding:
    """Regression tests for the banker's-rounding bug.

    Python's `round()` half-to-even turned `round(2 * 0.75) = round(1.5)` into
    2, which closed the entire 2-contract position at T1 and bypassed the
    runner entirely. This actually happened in production on the 2026-05-29
    Friday FB long: the bot entered 2 contracts, hit T1 at 7611.75 on Sunday
    night, and flattened all 2 — no T2, no runner trail. Using `math.floor`
    fixes it (floor(1.5) = 1).
    """

    def test_t1_with_two_contracts_leaves_runner(self):
        """The exact production scenario — must close 1, not 2."""
        mgr = _make_manager()
        pos = TradePosition(
            entry_price=7594.25,
            stop_price=7571.50,
            target_1=7611.75,
            target_2=7621.75,
            total_contracts=2,
            remaining_contracts=2,
            direction="long",
        )
        action = mgr.update(pos, high=7612.0, low=7610.0, close=7611.50)
        assert action is not None, "T1 should fire when high >= target_1"
        assert action.contracts_to_close == 1, (
            f"Expected 1 contract closed at T1 for 2-contract trade, got "
            f"{action.contracts_to_close} — banker's rounding regression"
        )
        assert pos.remaining_contracts == 1, "1 contract must survive for runner"
        assert pos.phase == ExitPhase.AFTER_T1

    def test_t1_with_one_contract_closes_full(self):
        """1 contract has no runner to preserve — close the 1 at T1."""
        mgr = _make_manager()
        pos = TradePosition(
            entry_price=100.0,
            stop_price=90.0,
            target_1=110.0,
            target_2=115.0,
            total_contracts=1,
            remaining_contracts=1,
            direction="long",
        )
        action = mgr.update(pos, high=111.0, low=109.0, close=110.5)
        assert action is not None
        assert action.contracts_to_close == 1
        assert pos.remaining_contracts == 0

    def test_t1_with_four_contracts_closes_three(self):
        """4 * 0.75 = 3.0 (no rounding ambiguity)."""
        mgr = _make_manager()
        pos = TradePosition(
            entry_price=100.0,
            stop_price=90.0,
            target_1=110.0,
            target_2=115.0,
            total_contracts=4,
            remaining_contracts=4,
            direction="long",
        )
        action = mgr.update(pos, high=111.0, low=109.0, close=110.5)
        assert action is not None
        assert action.contracts_to_close == 3
        assert pos.remaining_contracts == 1

    def test_t1_with_ten_contracts_closes_seven(self):
        """10 * 0.75 = 7.5; floor(7.5) = 7 (round would also give 8 here)."""
        mgr = _make_manager()
        pos = TradePosition(
            entry_price=100.0,
            stop_price=90.0,
            target_1=110.0,
            target_2=115.0,
            total_contracts=10,
            remaining_contracts=10,
            direction="long",
        )
        action = mgr.update(pos, high=111.0, low=109.0, close=110.5)
        assert action is not None
        assert action.contracts_to_close == 7
        assert pos.remaining_contracts == 3

    def test_t1_short_with_two_contracts_leaves_runner(self):
        """Same regression must hold for short side."""
        mgr = _make_manager()
        pos = TradePosition(
            entry_price=100.0,
            stop_price=110.0,
            target_1=90.0,
            target_2=85.0,
            total_contracts=2,
            remaining_contracts=2,
            direction="short",
        )
        action = mgr.update(pos, high=91.0, low=89.0, close=89.5)
        assert action is not None
        assert action.contracts_to_close == 1
        assert pos.remaining_contracts == 1


class TestT2ScaleDown:
    """T2 must actually sell ``t2_exit_fraction`` of total — not just flip phase."""

    def test_t2_sells_15pct_after_t1_on_20_contracts(self):
        """20 contracts: 15 at T1 (75%) + 3 at T2 (15%) → 2 runner (10%)."""
        mgr = _make_manager()
        pos = mgr.create_position(
            entry_price=5800.0, stop_price=5790.0,
            target_1=5810.0, target_2=5825.0, contracts=20,
        )
        _hit_t1(mgr, pos)
        assert pos.remaining_contracts == 5  # 25% runner = 5

        action = mgr.update(pos, high=5826.0, low=5824.0, close=5825.5)
        assert action is not None, "T2 must produce an ExitAction"
        assert action.contracts_to_close == 3, "Sell 15% of 20 = 3 contracts at T2"
        assert pos.remaining_contracts == 2, "10% runner remaining"
        assert pos.phase == ExitPhase.AFTER_T2
        assert pos.t2_hit is True
        assert "Target 2" in action.reason

    def test_t2_default_fraction_is_15pct(self):
        """Confirms the default t2_exit_fraction is 0.15."""
        mgr = ExitManager()  # all defaults
        assert mgr._t2_exit_fraction == pytest.approx(0.15)

    def test_t2_realized_pnl_includes_scale_down(self):
        """T2 scale-down P&L must accumulate into realized_pnl_pts."""
        mgr = _make_manager()
        pos = mgr.create_position(
            entry_price=5800.0, stop_price=5790.0,
            target_1=5810.0, target_2=5825.0, contracts=20,
        )
        _hit_t1(mgr, pos)
        pnl_after_t1 = pos.realized_pnl_pts
        # T1: 15 contracts * 10 pts = 150 pts
        assert pnl_after_t1 == pytest.approx(150.0)

        mgr.update(pos, high=5826.0, low=5824.0, close=5825.5)
        # T2: 3 contracts * 25 pts = 75 pts → running total 225
        assert pos.realized_pnl_pts == pytest.approx(225.0)

    def test_t2_with_small_position_falls_back_to_phase_only(self):
        """4 contracts at default 75/15/10: T1 sells 3, leaving 1. T2 fraction
        rounds to 1 but the runner floor is also 1 — so we never oversell; T2
        becomes a phase-only transition with no contracts sold."""
        mgr = _make_manager()
        pos = mgr.create_position(
            entry_price=5020.0, stop_price=5015.0,
            target_1=5030.0, target_2=5040.0, contracts=4,
        )
        _hit_t1(mgr, pos)
        assert pos.remaining_contracts == 1

        action = mgr.update(pos, high=5041.0, low=5039.0, close=5040.5)
        # Already at runner floor, so no sell — but phase must advance.
        assert action is None
        assert pos.phase == ExitPhase.AFTER_T2
        assert pos.t2_hit is True
        assert pos.remaining_contracts == 1

    def test_t2_never_oversells_runner_floor(self):
        """Even with a huge t2_exit_fraction, the runner floor is preserved."""
        mgr = _make_manager(t2_exit_fraction=0.50, runner_fraction=0.10)
        pos = mgr.create_position(
            entry_price=5800.0, stop_price=5790.0,
            target_1=5810.0, target_2=5825.0, contracts=20,
        )
        _hit_t1(mgr, pos)
        # 5 remaining; runner_floor = 2. Max sellable = 3. t2_target = 10.
        # contracts_to_exit must clamp to 3.
        action = mgr.update(pos, high=5826.0, low=5824.0, close=5825.5)
        assert action is not None
        assert action.contracts_to_close == 3
        assert pos.remaining_contracts == 2

    def test_t2_does_not_fire_below_target(self):
        """Price must actually reach target_2 — no early scale-down."""
        mgr = _make_manager()
        pos = mgr.create_position(
            entry_price=5800.0, stop_price=5790.0,
            target_1=5810.0, target_2=5825.0, contracts=20,
        )
        _hit_t1(mgr, pos)
        action = mgr.update(pos, high=5824.0, low=5820.0, close=5823.0)
        assert action is None
        assert pos.phase == ExitPhase.AFTER_T1
        assert pos.remaining_contracts == 5

    def test_t2_sets_phase_to_after_t2_and_t2_hit_flag(self):
        mgr = _make_manager()
        pos = mgr.create_position(
            entry_price=5800.0, stop_price=5790.0,
            target_1=5810.0, target_2=5825.0, contracts=20,
        )
        _hit_t1(mgr, pos)
        mgr.update(pos, high=5826.0, low=5824.0, close=5825.5)
        assert pos.phase == ExitPhase.AFTER_T2
        assert pos.t2_hit is True


# ---------------------------------------------------------------------------
# Part 2: Structure-based trail
# ---------------------------------------------------------------------------


class TestStructureTrail:
    """Post-T2 stop migrates to ``swing_low - buffer`` and only ratchets up."""

    def _drive_to_after_t2(
        self, mgr: ExitManager, bars: list[tuple[float, float]],
    ) -> TradePosition:
        """Build a position, drive it through T1 + T2, then feed the supplied
        bars. ``bars`` is a list of (high, low) tuples representing price
        action after T2 has fired."""
        pos = mgr.create_position(
            entry_price=5800.0, stop_price=5790.0,
            target_1=5810.0, target_2=5825.0, contracts=20,
        )
        _hit_t1(mgr, pos)
        mgr.update(pos, high=5826.0, low=5824.0, close=5825.5)
        assert pos.phase == ExitPhase.AFTER_T2
        for high, low in bars:
            mgr.update(pos, high=high, low=low, close=(high + low) / 2.0)
        return pos

    def test_structure_trail_finds_swing_low(self):
        """A clear local minimum surrounded by higher lows is picked up."""
        mgr = _make_manager(
            structure_trail_swing_order=2,
            structure_trail_buffer_pts=3.0,
        )
        # Bars after T2: form a V-shape with low at the centre.
        # idx        : 0      1      2      3      4
        # lows       : 5830, 5828, 5820, 5828, 5832  (swing low @ 5820)
        # Then a few more bars to confirm the swing and keep ratcheting.
        post_t2_bars = [
            (5832.0, 5830.0),
            (5831.0, 5828.0),
            (5825.0, 5820.0),  # swing low
            (5832.0, 5828.0),
            (5836.0, 5832.0),
            (5840.0, 5836.0),  # confirm the swing
            (5842.0, 5838.0),
        ]
        pos = self._drive_to_after_t2(mgr, post_t2_bars)
        # Stop should sit at 5820 - 3 = 5817 (the structure trail).
        assert pos.stop_price == pytest.approx(5817.0)

    def test_structure_trail_only_ratchets_up(self):
        """A later, lower swing low must NOT pull the stop back down."""
        mgr = _make_manager(
            structure_trail_swing_order=2,
            structure_trail_buffer_pts=3.0,
            # Bigger lookback so the earlier swing stays inside the window
            # for the full sequence we replay below.
            structure_trail_lookback_bars=50,
        )
        # First swing low at 5825 (confirmed by following higher lows).
        # Then price tags the runner highs and a later (lower) swing forms.
        # We must keep the original 5825-buffer stop.
        post_t2_bars: list[tuple[float, float]] = [
            (5832.0, 5830.0),
            (5831.0, 5828.0),
            (5830.0, 5825.0),  # first swing low candidate
            (5836.0, 5830.0),
            (5840.0, 5835.0),  # confirms the 5825 swing
            (5845.0, 5840.0),
            (5848.0, 5844.0),
            (5846.0, 5840.0),
            # Pull-back: form a NEW swing low at 5820 (LOWER than 5825).
            (5844.0, 5836.0),
            (5840.0, 5820.0),  # lower swing candidate
            (5836.0, 5832.0),
            (5840.0, 5834.0),  # confirms 5820 swing
            (5844.0, 5840.0),
        ]
        pos = self._drive_to_after_t2(mgr, post_t2_bars)
        # Stop must NOT have dropped — should still reflect the higher
        # 5825-based trail (5822). Even if internal logic identifies the
        # newer 5820 swing low, the ratchet-up guard blocks the move.
        assert pos.stop_price >= 5822.0 - 1e-6, (
            f"Stop trailed DOWN to {pos.stop_price} — must never decrease"
        )

    def test_structure_trail_disabled_falls_back_to_fixed(self):
        """With structure_trail_enabled=False, behaviour matches the legacy
        fixed-distance trail (highest_price - trailing_stop_pts)."""
        mgr = _make_manager(
            structure_trail_enabled=False,
            trailing_stop_pts=10.0,
        )
        post_t2_bars = [
            (5840.0, 5830.0),
            (5850.0, 5842.0),  # new high → fixed trail bumps up
            (5852.0, 5848.0),
        ]
        pos = self._drive_to_after_t2(mgr, post_t2_bars)
        # highest_price_since_entry == 5852 → fixed trail = 5852 - 10 = 5842
        assert pos.stop_price == pytest.approx(5852.0 - 10.0)

    def test_structure_trail_no_swing_yet_keeps_existing_stop(self):
        """Before enough history accumulates for a swing low, the post-T2
        stop must not drop below its prior level."""
        mgr = _make_manager(structure_trail_swing_order=5)
        # Only feed two post-T2 bars → not enough for a swing of order 5.
        pos = self._drive_to_after_t2(mgr, [(5830.0, 5828.0), (5832.0, 5830.0)])
        # Stop should still be whatever was set when T2 fired (the fallback
        # since no swing was available). The key invariant: it didn't drop
        # below the post-T1 stop of `entry + breakeven_buffer = 5800 + -3 = 5797`.
        assert pos.stop_price >= 5797.0 - 1e-6

    def test_structure_trail_ratchets_up_on_new_higher_swing(self):
        """As price makes higher swing lows, the stop should follow up."""
        mgr = _make_manager(
            structure_trail_swing_order=2,
            structure_trail_buffer_pts=3.0,
            structure_trail_lookback_bars=50,
        )
        # First swing at 5828 → expect stop ≈ 5825.
        # Then a higher swing at 5840 → expect stop ≈ 5837.
        post_t2_bars: list[tuple[float, float]] = [
            (5836.0, 5832.0),
            (5834.0, 5830.0),
            (5832.0, 5828.0),  # swing #1 (5828)
            (5836.0, 5832.0),
            (5840.0, 5836.0),  # confirms swing #1
        ]
        pos = self._drive_to_after_t2(mgr, post_t2_bars)
        first_stop = pos.stop_price
        assert first_stop == pytest.approx(5825.0)

        # Now a higher swing low at 5840.
        more_bars: list[tuple[float, float]] = [
            (5848.0, 5844.0),
            (5852.0, 5848.0),
            (5854.0, 5840.0),  # candidate higher swing low
            (5858.0, 5848.0),
            (5860.0, 5852.0),  # confirms the 5840 swing
            (5862.0, 5854.0),
        ]
        mgr_pos = pos  # alias for clarity
        for high, low in more_bars:
            mgr.update(mgr_pos, high=high, low=low, close=(high + low) / 2.0)
        assert mgr_pos.stop_price == pytest.approx(5837.0)
        assert mgr_pos.stop_price > first_stop


# ---------------------------------------------------------------------------
# Part 3: Backward compat — existing T1-only flows still work
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """The legacy fixed-distance trail still works when structure trail is off."""

    def test_t1_only_behavior_with_structure_disabled(self):
        """Disabling the structure trail must not affect the T1 exit path."""
        mgr = _make_manager(structure_trail_enabled=False)
        pos = mgr.create_position(
            entry_price=5020.0, stop_price=5015.0,
            target_1=5030.0, target_2=5040.0, contracts=4,
        )
        action = mgr.update(pos, high=5031.0, low=5028.0, close=5030.5)
        assert action is not None
        assert action.contracts_to_close == 3  # 75% of 4
        assert pos.phase == ExitPhase.AFTER_T1
        # Stop should have moved to entry + breakeven_buffer (-3) = 5017
        assert pos.stop_price == pytest.approx(5017.0)

    def test_stop_loss_still_closes_all_with_structure_enabled(self):
        """The initial stop-out path remains unaffected by the new trail."""
        mgr = _make_manager(structure_trail_enabled=True)
        pos = mgr.create_position(
            entry_price=5020.0, stop_price=5015.0,
            target_1=5030.0, target_2=5040.0, contracts=4,
        )
        action = mgr.update(pos, high=5018.0, low=5014.0, close=5015.5)
        assert action is not None
        assert action.contracts_to_close == 4
        assert pos.phase == ExitPhase.CLOSED

    def test_bar_history_capped_to_lookback(self):
        """The rolling bar history must never exceed structure_trail_lookback_bars."""
        mgr = _make_manager(structure_trail_lookback_bars=5)
        pos = mgr.create_position(
            entry_price=5800.0, stop_price=5790.0,
            target_1=5810.0, target_2=5825.0, contracts=20,
        )
        # Feed 20 bars with no T1 hit (keep below target_1).
        for i in range(20):
            mgr.update(pos, high=5805.0, low=5802.0, close=5804.0)
        assert len(pos.bar_history) == 5

    def test_structure_trail_short_positions_no_op(self):
        """Structure trail (long-only per the Mancini quote) returns None for shorts."""
        mgr = _make_manager()
        pos = mgr.create_position(
            entry_price=6000.0, stop_price=6010.0,
            target_1=5990.0, target_2=5975.0, contracts=20,
            direction="short",
        )
        # Drive through T1 + T2 on the short side.
        mgr.update(pos, high=5995.0, low=5989.0, close=5990.0)
        assert pos.phase == ExitPhase.AFTER_T1
        mgr.update(pos, high=5980.0, low=5974.0, close=5975.0)
        assert pos.phase == ExitPhase.AFTER_T2
        # Feed bars; structure_trail_stop should be None (returns None for shorts),
        # so the short-side fallback trail logic still controls the stop.
        struct = mgr._compute_structure_trail_stop(pos)
        assert struct is None
