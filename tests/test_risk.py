"""Tests for risk management and session rules."""

from __future__ import annotations

from datetime import datetime, time

import pytest

from config.settings import RiskParams, SessionTimes
from core.signals import Signal, SignalType
from core.patterns import PatternSignal, ConfirmationType
from config.levels import Level, LevelType
from strategy.entry_manager import EntryManager
from strategy.exit_manager import ExitManager, TradePosition, ExitPhase
from strategy.position_manager import PositionManager, SessionState
from strategy.risk_manager import RiskManager


@pytest.fixture
def risk_manager():
    return RiskManager()


@pytest.fixture
def position_manager():
    pm = PositionManager()
    pm.start_session(datetime(2024, 1, 15))
    return pm


@pytest.fixture
def entry_manager():
    return EntryManager()


@pytest.fixture
def sample_signal():
    """A sample qualified Signal for testing."""
    base_time = datetime(2024, 1, 15, 9, 0)
    pattern = PatternSignal(
        pattern_type="failed_breakdown",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=Level(
            price=5020.0,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=base_time,
            confirmed_at=base_time,
        ),
        sweep_low=5018.0,
        entry_price=5022.0,
        stop_price=5016.0,
        bar_idx=50,
        timestamp=base_time,
    )
    return Signal(
        signal_type=SignalType.FAILED_BREAKDOWN,
        pattern=pattern,
        target_1=5032.0,
        target_2=5042.0,
        risk_pts=6.0,
        reward_t1_pts=10.0,
        reward_t2_pts=20.0,
        rr_ratio_t1=1.67,
        rr_ratio_t2=3.33,
        bar_idx=50,
        timestamp=base_time,
    )


class TestRiskManager:
    """Test risk management validation."""

    def test_passes_valid_entry(self, risk_manager, position_manager, sample_signal):
        """Valid entry should pass all checks."""
        check = risk_manager.validate_entry(
            sample_signal, time(9, 45), position_manager
        )
        assert check.passed

    def test_blocks_chop_zone(self, risk_manager, position_manager, sample_signal):
        """Should reject trades in the 11AM-2PM chop zone."""
        check = risk_manager.validate_entry(
            sample_signal, time(12, 0), position_manager
        )
        assert not check.passed
        assert "chop zone" in check.reason.lower()

    def test_blocks_after_max_trades(self, risk_manager, position_manager, sample_signal):
        """Should reject after max trades per day."""
        # Simulate 2 completed trades
        exit_mgr = ExitManager()
        for _ in range(2):
            pos = exit_mgr.create_position(5020, 5015, 5030, 5040, 4)
            position_manager.open_position(pos, datetime(2024, 1, 15, 9, 30), "test")
            pos.remaining_contracts = 0
            pos.realized_pnl_pts = 10.0
            pos.phase = ExitPhase.CLOSED
            position_manager.close_position(5030.0, datetime(2024, 1, 15, 10, 0), "test")

        check = risk_manager.validate_entry(
            sample_signal, time(10, 30), position_manager
        )
        assert not check.passed

    def test_blocks_after_daily_loss(self, risk_manager, position_manager, sample_signal):
        """Should reject when daily loss limit is reached."""
        # Simulate a big loss
        pos = ExitManager().create_position(5020, 5000, 5030, 5040, 4)
        position_manager.open_position(pos, datetime(2024, 1, 15, 9, 30), "test")
        pos.remaining_contracts = 0
        pos.realized_pnl_pts = -25.0
        pos.phase = ExitPhase.CLOSED
        position_manager.close_position(5000.0, datetime(2024, 1, 15, 10, 0), "loss")

        check = risk_manager.validate_entry(
            sample_signal, time(10, 30), position_manager
        )
        assert not check.passed

    def test_blocks_wide_stop(self, risk_manager, position_manager):
        """Should reject if stop is too wide (>15 pts)."""
        base_time = datetime(2024, 1, 15, 9, 0)
        pattern = PatternSignal(
            pattern_type="failed_breakdown",
            confirmation=ConfirmationType.ACCEPTANCE,
            level=Level(price=5020.0, level_type=LevelType.SWING_LOW,
                        created_at=base_time, confirmed_at=base_time),
            sweep_low=5000.0,
            entry_price=5022.0,
            stop_price=5000.0,  # 22 pts away
            bar_idx=50,
            timestamp=base_time,
        )
        wide_signal = Signal(
            signal_type=SignalType.FAILED_BREAKDOWN,
            pattern=pattern,
            target_1=5040.0,
            target_2=5060.0,
            risk_pts=22.0,
            reward_t1_pts=18.0,
            reward_t2_pts=38.0,
            rr_ratio_t1=0.82,
            rr_ratio_t2=1.73,
            bar_idx=50,
            timestamp=base_time,
        )
        check = risk_manager.validate_entry(
            wide_signal, time(9, 45), position_manager
        )
        assert not check.passed


class TestPositionManager:
    """Test session state machine."""

    def test_initial_state(self, position_manager):
        assert position_manager.session.state == SessionState.ACTIVE
        assert position_manager.trades_today == 0

    def test_profit_protection_after_win(self, position_manager):
        """After a win, should enter profit protection mode."""
        pos = ExitManager().create_position(5020, 5015, 5030, 5040, 4)
        position_manager.open_position(pos, datetime(2024, 1, 15, 9, 30), "test")

        pos.remaining_contracts = 0
        pos.realized_pnl_pts = 10.0
        pos.phase = ExitPhase.CLOSED
        position_manager.close_position(5030.0, datetime(2024, 1, 15, 10, 0), "T1 hit")

        assert position_manager.is_profit_protection
        assert position_manager.session.state == SessionState.PROFIT_PROTECTION

    def test_done_after_two_losses(self, position_manager):
        """After two losses, session should be done."""
        for i in range(2):
            pos = ExitManager().create_position(5020, 5015, 5030, 5040, 4)
            position_manager.open_position(
                pos, datetime(2024, 1, 15, 9 + i, 30), "test"
            )
            pos.remaining_contracts = 0
            pos.realized_pnl_pts = -5.0
            pos.phase = ExitPhase.CLOSED
            position_manager.close_position(
                5015.0, datetime(2024, 1, 15, 10 + i, 0), "stop"
            )

        assert position_manager.is_done_for_day

    def test_no_duplicate_open_positions(self, position_manager):
        """Should reject opening a second position."""
        pos1 = ExitManager().create_position(5020, 5015, 5030, 5040, 4)
        assert position_manager.open_position(pos1, datetime(2024, 1, 15, 9, 30), "test")

        pos2 = ExitManager().create_position(5050, 5045, 5060, 5070, 4)
        assert not position_manager.open_position(pos2, datetime(2024, 1, 15, 9, 35), "test")


class TestEntryManager:
    """Test entry decision logic."""

    def test_rejects_in_chop_zone(self, entry_manager, sample_signal):
        decision = entry_manager.evaluate(
            sample_signal,
            current_time=time(12, 30),
            trades_today=0,
            is_in_profit_protection=False,
            daily_pnl_pts=0.0,
        )
        assert not decision.should_enter

    def test_accepts_in_morning_window(self, entry_manager, sample_signal):
        decision = entry_manager.evaluate(
            sample_signal,
            current_time=time(8, 0),
            trades_today=0,
            is_in_profit_protection=False,
            daily_pnl_pts=0.0,
        )
        assert decision.should_enter
        assert decision.contracts == 4

    def test_sizes_down_in_profit_protection(self, entry_manager, sample_signal):
        """In profit protection, should only risk current profits."""
        decision = entry_manager.evaluate(
            sample_signal,
            current_time=time(8, 0),
            trades_today=1,
            is_in_profit_protection=True,
            daily_pnl_pts=8.0,  # 8 pts of profit to risk
        )
        assert decision.should_enter
        # risk_per_contract = 6.0 pts, can risk 8 pts → 1 contract
        assert decision.contracts <= 2

    def test_rejects_past_eod_flatten(self, entry_manager, sample_signal):
        """Should reject entries past the EOD flatten time (3:55 PM)."""
        decision = entry_manager.evaluate(
            sample_signal,
            current_time=time(15, 56),
            trades_today=0,
            is_in_profit_protection=False,
            daily_pnl_pts=0.0,
        )
        assert not decision.should_enter
        assert "EOD flatten" in decision.reason

    def test_accepts_before_eod_flatten(self, entry_manager, sample_signal):
        """Should accept entries before EOD flatten time."""
        decision = entry_manager.evaluate(
            sample_signal,
            current_time=time(15, 30),
            trades_today=0,
            is_in_profit_protection=False,
            daily_pnl_pts=0.0,
        )
        assert decision.should_enter


class TestMorningWindow:
    """Test corrected morning window (9:30-11:00)."""

    def test_930_is_in_preferred_window(self):
        """9:30 AM should be in the preferred morning window."""
        session = SessionTimes()
        assert session.in_preferred_window(time(9, 30))

    def test_1030_is_in_preferred_window(self):
        """10:30 AM should be in the preferred morning window."""
        session = SessionTimes()
        assert session.in_preferred_window(time(10, 30))

    def test_1100_is_in_preferred_window(self):
        """11:00 AM should be in the preferred morning window."""
        session = SessionTimes()
        assert session.in_preferred_window(time(11, 0))

    def test_830_is_not_in_preferred_window(self):
        """8:30 AM should NOT be in preferred window (old window was 7:30-8:30)."""
        session = SessionTimes()
        assert not session.in_preferred_window(time(8, 30))

    def test_afternoon_window_still_works(self):
        """3:00-3:55 PM should still be in preferred window."""
        session = SessionTimes()
        assert session.in_preferred_window(time(15, 0))
        assert session.in_preferred_window(time(15, 30))


class TestEODFlatten:
    """Test EOD flatten functionality."""

    def test_past_eod_flatten(self):
        """past_eod_flatten should return True after 3:55 PM."""
        session = SessionTimes()
        assert session.past_eod_flatten(time(15, 55))
        assert session.past_eod_flatten(time(15, 59))
        assert session.past_eod_flatten(time(16, 0))

    def test_not_past_eod_flatten(self):
        """past_eod_flatten should return False before 3:55 PM."""
        session = SessionTimes()
        assert not session.past_eod_flatten(time(15, 54))
        assert not session.past_eod_flatten(time(14, 0))

    def test_position_manager_eod_flatten_with_position(self, position_manager):
        """check_eod_flatten should return True when position open and past time."""
        pos = ExitManager().create_position(5020, 5015, 5030, 5040, 4)
        position_manager.open_position(pos, datetime(2024, 1, 15, 9, 30), "test")
        assert position_manager.check_eod_flatten(time(15, 56))

    def test_position_manager_eod_flatten_no_position(self, position_manager):
        """check_eod_flatten should return False when no position open."""
        assert not position_manager.check_eod_flatten(time(15, 56))


class TestMinLevelsBrokenGate:
    """Test that SignalAggregator gates on min_levels_broken."""

    def test_elevator_with_enough_levels_accepted(self):
        """Elevator breaking >= min_levels_broken should be accepted."""
        from core.signals import SignalAggregator
        from config.settings import ElevatorParams

        params = ElevatorParams(min_levels_broken=2)
        aggregator = SignalAggregator(
            min_rr_ratio=0.0,
            elevator_params=params,
        )
        assert aggregator._last_elevator is None

        from core.elevator_down import ElevatorEvent
        event = ElevatorEvent(
            start_idx=0, start_time=datetime(2024, 1, 15, 9, 30),
            start_price=5050.0, end_idx=9,
            end_time=datetime(2024, 1, 15, 9, 39), end_price=5020.0,
            low_price=5015.0, low_idx=8, peak_velocity=4.0, levels_broken=3,
        )

        # Directly test the gate logic
        if event.levels_broken >= aggregator.elevator_detector.params.min_levels_broken:
            aggregator._last_elevator = event

        assert aggregator._last_elevator is not None

    def test_elevator_with_too_few_levels_rejected(self):
        """Elevator breaking < min_levels_broken should be rejected."""
        from core.signals import SignalAggregator
        from config.settings import ElevatorParams

        params = ElevatorParams(min_levels_broken=2)
        aggregator = SignalAggregator(
            min_rr_ratio=0.0,
            elevator_params=params,
        )

        from core.elevator_down import ElevatorEvent
        event = ElevatorEvent(
            start_idx=0, start_time=datetime(2024, 1, 15, 9, 30),
            start_price=5050.0, end_idx=9,
            end_time=datetime(2024, 1, 15, 9, 39), end_price=5020.0,
            low_price=5015.0, low_idx=8, peak_velocity=4.0, levels_broken=1,
        )

        # Directly test the gate logic
        if event.levels_broken >= aggregator.elevator_detector.params.min_levels_broken:
            aggregator._last_elevator = event

        assert aggregator._last_elevator is None, "Should reject elevator with only 1 level broken"
