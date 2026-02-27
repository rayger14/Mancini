"""Tests for exit management (75/15/10 split, trailing stops, prior-day-low runner trail)."""

from __future__ import annotations

import pytest

from config.settings import ExitParams
from strategy.exit_manager import ExitManager, ExitPhase


@pytest.fixture
def exit_manager():
    return ExitManager()


@pytest.fixture
def exit_manager_full_exit():
    """Exit manager with 100% T1 exit (no runner) for backward compat tests."""
    params = ExitParams(t1_exit_fraction=1.0, runner_fraction=0.0)
    return ExitManager(params=params)


@pytest.fixture
def sample_position(exit_manager):
    """A sample position: 4 contracts, entry=5020, stop=5015, T1=5030, T2=5040."""
    return exit_manager.create_position(
        entry_price=5020.0,
        stop_price=5015.0,
        target_1=5030.0,
        target_2=5040.0,
        contracts=4,
    )


class TestExitManager:
    """Test the 75/15/10 exit system."""

    def test_initial_state(self, sample_position):
        assert sample_position.phase == ExitPhase.INITIAL
        assert sample_position.remaining_contracts == 4
        assert sample_position.is_open

    def test_stop_loss_closes_all(self, exit_manager, sample_position):
        """Stop hit → close all contracts."""
        action = exit_manager.update(
            sample_position,
            high=5022.0, low=5014.5, close=5015.0
        )
        assert action is not None
        assert action.contracts_to_close == 4
        assert action.new_phase == ExitPhase.CLOSED
        assert not sample_position.is_open

    def test_t1_exits_75pct(self, exit_manager, sample_position):
        """Target 1 hit → exit 75% (3 of 4 contracts), keep 1 runner."""
        action = exit_manager.update(
            sample_position,
            high=5031.0, low=5028.0, close=5030.5
        )
        assert action is not None
        assert action.contracts_to_close == 3  # 75% of 4
        assert sample_position.remaining_contracts == 1
        assert sample_position.phase == ExitPhase.AFTER_T1

    def test_t1_100pct_exits_all(self, exit_manager_full_exit):
        """With t1_exit_fraction=1.0, T1 hit closes all."""
        pos = exit_manager_full_exit.create_position(
            entry_price=5020.0, stop_price=5015.0,
            target_1=5030.0, target_2=5040.0, contracts=4,
        )
        action = exit_manager_full_exit.update(pos, high=5031.0, low=5028.0, close=5030.5)
        assert action is not None
        assert action.contracts_to_close == 4
        assert pos.remaining_contracts == 0

    def test_t1_stop_moves_under_breakeven(self, exit_manager, sample_position):
        """After T1, stop should be several pts UNDER breakeven (Mancini method)."""
        exit_manager.update(
            sample_position,
            high=5031.0, low=5028.0, close=5030.5
        )
        # Default breakeven_buffer_pts = -3.0 → stop at 5020 + (-3) = 5017
        assert sample_position.stop_price == pytest.approx(5017.0)

    def test_t2_exits_to_runner(self, exit_manager, sample_position):
        """T1 → T2 → only runner remains."""
        # T1 hit: exit 3 of 4
        exit_manager.update(sample_position, high=5031.0, low=5028.0, close=5030.5)
        assert sample_position.remaining_contracts == 1
        # T2 hit: with 1 contract remaining and runner_fraction=0.10,
        # runner_contracts = max(1, round(4 * 0.10)) = 1
        # contracts_to_exit = 1 - 1 = 0, so no further exit (already at runner size)
        action = exit_manager.update(sample_position, high=5041.0, low=5038.0, close=5040.0)
        assert action is None  # already at runner size, just phase change
        assert sample_position.phase == ExitPhase.AFTER_T2

    def test_trailing_stop_tightens(self, exit_manager, sample_position):
        """Trailing stop should tighten as profit grows (fallback intraday trail)."""
        sample_position.remaining_contracts = 1
        sample_position.phase = ExitPhase.AFTER_T2
        sample_position.stop_price = 5027.0

        # Runner at 5045 → profit = 25 pts → trail should tighten to 2 pts
        exit_manager.update(
            sample_position,
            high=5045.0, low=5043.0, close=5044.0
        )
        expected_stop = 5045.0 - 2.0
        assert sample_position.stop_price == expected_stop

    def test_pnl_tracking_75pct(self, exit_manager, sample_position):
        """Realized P&L accumulates: 75% at T1 = 3 contracts * 10 pts."""
        exit_manager.update(
            sample_position,
            high=5031.0, low=5028.0, close=5030.5
        )
        # 3 contracts * 10 pts = 30 pts realized
        assert sample_position.realized_pnl_pts == pytest.approx(30.0, abs=1.0)

    def test_no_action_when_closed(self, exit_manager, sample_position):
        """No action if position is already closed."""
        exit_manager.update(
            sample_position,
            high=5022.0, low=5014.0, close=5015.0
        )
        assert not sample_position.is_open
        action = exit_manager.update(
            sample_position,
            high=5025.0, low=5020.0, close=5022.0
        )
        assert action is None


class TestPriorDayLowTrail:
    """Test the Mancini prior-day-low runner trailing."""

    def test_update_prior_day_low_ratchets_up(self, exit_manager):
        """Prior day low trail should only ratchet up for longs."""
        pos = exit_manager.create_position(
            entry_price=5020.0, stop_price=5015.0,
            target_1=5030.0, target_2=5040.0, contracts=4,
        )
        # Simulate T1 hit
        exit_manager.update(pos, high=5031.0, low=5028.0, close=5030.5)
        assert pos.phase == ExitPhase.AFTER_T1

        # Set prior day low: 5010 → stop at 5010 - 1.0 = 5009
        action = exit_manager.update_prior_day_low(pos, 5010.0)
        # 5009 < current stop (5017) → should NOT ratchet down
        assert action is None
        assert pos.stop_price == pytest.approx(5017.0)

        # Set prior day low: 5020 → stop at 5019 > 5017 → ratchet up
        action = exit_manager.update_prior_day_low(pos, 5020.0)
        assert action is not None
        assert pos.stop_price == pytest.approx(5019.0)

    def test_runner_no_intraday_trail_with_pdl(self, exit_manager):
        """With prior_day_low set, runner stop stays fixed during session."""
        pos = exit_manager.create_position(
            entry_price=5020.0, stop_price=5015.0,
            target_1=5030.0, target_2=5040.0, contracts=4,
        )
        exit_manager.update(pos, high=5031.0, low=5028.0, close=5030.5)
        # Go to AFTER_T2
        pos.phase = ExitPhase.AFTER_T2
        pos.prior_day_low = 5010.0
        pos.stop_price = 5009.0  # under prior day low

        # Price makes new highs — stop should NOT trail (PDL set)
        exit_manager.update(pos, high=5060.0, low=5055.0, close=5058.0)
        assert pos.stop_price == pytest.approx(5009.0)  # unchanged

    def test_short_direction(self, exit_manager):
        """Short positions track direction correctly."""
        pos = exit_manager.create_position(
            entry_price=5040.0, stop_price=5050.0,
            target_1=5030.0, target_2=5020.0, contracts=4,
            direction="short",
        )
        # T1 hit for short (price drops to target)
        action = exit_manager.update(pos, high=5035.0, low=5029.0, close=5030.0)
        assert action is not None
        assert action.contracts_to_close == 3
        assert pos.phase == ExitPhase.AFTER_T1
