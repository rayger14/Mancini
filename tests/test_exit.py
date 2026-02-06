"""Tests for exit management (75/15/10 split, trailing stops)."""

from __future__ import annotations

import pytest

from config.settings import ExitParams, ESContractSpec
from strategy.exit_manager import ExitManager, ExitPhase, TradePosition


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

    def test_t1_exits_75_percent(self, exit_manager, sample_position):
        """Target 1 hit → exit 75% (3 contracts), move stop to breakeven."""
        action = exit_manager.update(
            sample_position,
            high=5031.0, low=5028.0, close=5030.5
        )
        assert action is not None
        assert action.contracts_to_close == 3  # 75% of 4
        assert sample_position.remaining_contracts == 1
        assert sample_position.phase == ExitPhase.AFTER_T1
        # Stop should be at breakeven + 1 tick
        expected_stop = 5020.0 + 0.25
        assert sample_position.stop_price == expected_stop

    def test_t2_then_runner(self, exit_manager, sample_position):
        """After T1, hitting T2 should leave runner with trailing stop."""
        # First hit T1
        exit_manager.update(
            sample_position,
            high=5031.0, low=5028.0, close=5030.5
        )
        assert sample_position.phase == ExitPhase.AFTER_T1
        assert sample_position.remaining_contracts == 1

        # Now hit T2 — only 1 contract left, so it becomes runner
        action = exit_manager.update(
            sample_position,
            high=5041.0, low=5038.0, close=5040.5
        )
        # With 1 contract remaining and runner_fraction wanting 1, no exit
        # It should transition to AFTER_T2
        assert sample_position.phase == ExitPhase.AFTER_T2

    def test_trailing_stop_tightens(self, exit_manager, sample_position):
        """Trailing stop should tighten as profit grows."""
        # Skip to runner phase
        exit_manager.update(
            sample_position,
            high=5031.0, low=5028.0, close=5030.5  # T1
        )
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
        # T1: exit 3 contracts at 5030 (10 pts profit each)
        exit_manager.update(
            sample_position,
            high=5031.0, low=5028.0, close=5030.5
        )
        # 3 contracts * 10 pts = 30 pts realized
        assert sample_position.realized_pnl_pts == pytest.approx(30.0, abs=1.0)

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
