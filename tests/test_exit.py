"""Tests for exit management (75/15/10 split, trailing stops)."""

from __future__ import annotations

import pytest

from strategy.exit_manager import ExitManager, ExitPhase


@pytest.fixture
def exit_manager():
    return ExitManager()


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
            high=5022.0, low=5014.5, close=5015.0  # low below stop
        )
        assert action is not None
        assert action.contracts_to_close == 4
        assert action.new_phase == ExitPhase.CLOSED
        assert not sample_position.is_open

    def test_t1_exits_full_position(self, exit_manager, sample_position):
        """Target 1 hit → exit 100% (all contracts) with t1_exit_fraction=1.0."""
        action = exit_manager.update(
            sample_position,
            high=5031.0, low=5028.0, close=5030.5
        )
        assert action is not None
        assert action.contracts_to_close == 4  # 100% of 4
        assert sample_position.remaining_contracts == 0
        assert sample_position.phase == ExitPhase.AFTER_T1

    def test_t1_closes_position_fully(self, exit_manager, sample_position):
        """With t1_exit_fraction=1.0, T1 hit closes the full position."""
        exit_manager.update(
            sample_position,
            high=5031.0, low=5028.0, close=5030.5
        )
        # With 100% exit at T1, position should be fully closed
        assert sample_position.remaining_contracts == 0
        assert not sample_position.is_open

    def test_trailing_stop_tightens(self, exit_manager, sample_position):
        """Trailing stop should tighten as profit grows (for runner scenarios)."""
        # Manually set up a runner scenario (with partial exit fraction < 1.0)
        sample_position.remaining_contracts = 1
        sample_position.phase = ExitPhase.AFTER_T2
        sample_position.stop_price = 5027.0

        # Runner at 5045 → profit = 25 pts → trail should tighten to 2 pts
        exit_manager.update(
            sample_position,
            high=5045.0, low=5043.0, close=5044.0
        )
        # At 25 pts profit, trail tightens to 2.0 pts
        expected_stop = 5045.0 - 2.0
        assert sample_position.stop_price == expected_stop

    def test_pnl_tracking(self, exit_manager, sample_position):
        """Realized P&L should accumulate across exits."""
        # T1: exit 4 contracts at 5030 (10 pts profit each) with 100% T1 exit
        exit_manager.update(
            sample_position,
            high=5031.0, low=5028.0, close=5030.5
        )
        # 4 contracts * 10 pts = 40 pts realized
        assert sample_position.realized_pnl_pts == pytest.approx(40.0, abs=1.0)

    def test_no_action_when_closed(self, exit_manager, sample_position):
        """No action if position is already closed."""
        # Stop out
        exit_manager.update(
            sample_position,
            high=5022.0, low=5014.0, close=5015.0
        )
        assert not sample_position.is_open

        # Another bar should return None
        action = exit_manager.update(
            sample_position,
            high=5025.0, low=5020.0, close=5022.0
        )
        assert action is None
