"""Tests for short suppression features: move exhaustion + elevator→FB cycle."""

from datetime import datetime, timedelta

import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.signals import SignalAggregator, SignalType
from core.patterns import PatternSignal, ConfirmationType
from core.elevator_down import ElevatorEvent


def _ts(minute: int) -> datetime:
    base = datetime(2024, 6, 15, 9, 30)
    return base + timedelta(minutes=minute)


def _make_level(price: float, lt: LevelType = LevelType.PRIOR_DAY_LOW) -> Level:
    ts = datetime(2024, 6, 15, 9, 0)
    return Level(price=price, level_type=lt, created_at=ts, confirmed_at=ts)


def _make_short_pattern(
    level_price: float = 7100.0,
    entry_price: float = 7095.0,
    stop_price: float = 7106.0,
    sweep_depth_pts: float = 5.0,
) -> PatternSignal:
    return PatternSignal(
        pattern_type="breakdown_short",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=_make_level(level_price),
        sweep_low=7085.0,
        entry_price=entry_price,
        stop_price=stop_price,
        bar_idx=100,
        timestamp=_ts(60),
        sweep_depth_pts=sweep_depth_pts,
        direction="short",
        sweep_high=level_price + 5,
    )


def _make_agg(**extra_params) -> SignalAggregator:
    extra_params.setdefault("block_pdl_shorts", False)  # test other gates in isolation
    params = StrategyParams(
        allow_breakdown_short=True,
        bd_short_min_rr=0.5,  # low floor so R:R doesn't interfere
        min_signal_rr=0.3,
        **extra_params,
    )
    agg = SignalAggregator(strategy_params=params)
    agg.level_store = LevelStore()
    # Support targets below entry for short signals
    agg.level_store.add(_make_level(7070.0))
    agg.level_store.add(_make_level(7050.0))
    return agg


class TestMoveExhaustion:
    """Suppress shorts when the session low already hit the target."""

    def test_short_blocked_when_target_already_hit(self):
        """If session low already reached T1, the move is done — skip."""
        agg = _make_agg()
        # Simulate session where price already dropped to 7060 (below T1 ~7070)
        agg._session_low = 7060.0

        pattern = _make_short_pattern(entry_price=7095.0, stop_price=7106.0)
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)

        assert signal is None
        # Should have logged a shadow event
        exhaustion_events = [e for e in agg.shadow_events if e["feature"] == "move_exhaustion"]
        assert len(exhaustion_events) == 1
        assert exhaustion_events[0]["session_low"] == 7060.0

    def test_short_allowed_when_target_not_hit(self):
        """If session low hasn't reached T1, the short is valid."""
        agg = _make_agg()
        # Session low is above the target
        agg._session_low = 7090.0

        pattern = _make_short_pattern(entry_price=7095.0, stop_price=7106.0)
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)

        assert signal is not None
        assert signal.signal_type == SignalType.BREAKDOWN_SHORT

    def test_short_blocked_when_target_exactly_hit(self):
        """Edge case: session low equals T1 exactly — still exhausted."""
        agg = _make_agg()
        agg._session_low = 7070.0  # exactly at T1

        pattern = _make_short_pattern(entry_price=7095.0, stop_price=7106.0)
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)

        assert signal is None


class TestElevatorFBCycleDetection:
    """Suppress shorts when elevator→FB squeeze is active."""

    def _make_elevator(self, start_price=7160.0, low_price=7080.0) -> ElevatorEvent:
        return ElevatorEvent(
            start_idx=10,
            start_time=_ts(10),
            start_price=start_price,
            end_idx=50,
            end_time=_ts(50),
            end_price=low_price + 20,
            low_price=low_price,
            low_idx=45,
            peak_velocity=5.0,
            levels_broken=3,
        )

    def test_short_suppressed_during_squeeze(self):
        """After 80pt elevator + 50%+ recovery, shorts are suppressed."""
        agg = _make_agg()
        agg._session_low = 7200.0  # high enough to not trigger move exhaustion
        # Set up elevator→FB active state directly
        agg._elevator_fb_active = True
        agg._elevator_recovery_low = 7080.0
        agg._elevator_recovery_drop = 80.0

        pattern = _make_short_pattern(entry_price=7130.0, stop_price=7141.0)
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)

        assert signal is None
        squeeze_events = [e for e in agg.shadow_events if e["feature"] == "elevator_fb_squeeze_suppression"]
        assert len(squeeze_events) == 1
        assert squeeze_events[0]["elevator_drop_pts"] == 80.0

    def test_short_allowed_when_no_elevator(self):
        """No elevator event — shorts proceed normally."""
        agg = _make_agg()
        agg._session_low = 7200.0
        agg._elevator_fb_active = False

        pattern = _make_short_pattern(entry_price=7095.0, stop_price=7106.0)
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)

        assert signal is not None

    def test_elevator_fb_activates_on_recovery(self):
        """process_bar should activate elevator→FB when price recovers 50%+."""
        agg = _make_agg()
        # Plant a completed elevator event
        elevator = self._make_elevator(start_price=7160.0, low_price=7080.0)
        agg._last_elevator = elevator

        # Process a bar where close is above 50% recovery
        # Drop = 80 pts, 50% recovery = 7080 + 40 = 7120
        # Close at 7130 = 62.5% recovery → should activate
        agg.update(
            bar_idx=60, timestamp=_ts(60),
            open_=7128.0, high=7132.0, low=7126.0, close=7130.0,
            volume=5000, velocity=0.5,
        )

        assert agg._elevator_fb_active is True

    def test_elevator_fb_not_active_on_small_recovery(self):
        """If recovery is < 50% of the drop, don't activate."""
        agg = _make_agg()
        elevator = self._make_elevator(start_price=7160.0, low_price=7080.0)
        agg._last_elevator = elevator

        # Close at 7100 = 25% recovery → should NOT activate
        agg.update(
            bar_idx=60, timestamp=_ts(60),
            open_=7098.0, high=7102.0, low=7096.0, close=7100.0,
            volume=5000, velocity=0.5,
        )

        assert agg._elevator_fb_active is False

    def test_elevator_fb_not_active_on_small_drop(self):
        """Elevator drop < 40 pts doesn't activate suppression."""
        agg = _make_agg()
        # Only a 30 pt drop
        elevator = self._make_elevator(start_price=7110.0, low_price=7080.0)
        agg._last_elevator = elevator

        # Full recovery — but drop was too small
        agg.update(
            bar_idx=60, timestamp=_ts(60),
            open_=7108.0, high=7112.0, low=7106.0, close=7110.0,
            volume=5000, velocity=0.5,
        )

        assert agg._elevator_fb_active is False

    def test_elevator_fb_deactivates_on_new_low(self):
        """If price falls back below elevator low, deactivate."""
        agg = _make_agg()
        # Activate the cycle
        agg._elevator_fb_active = True
        agg._elevator_recovery_low = 7080.0
        agg._elevator_recovery_drop = 80.0
        # No active elevator anymore (expired)
        agg._last_elevator = None

        # Price falls below elevator low
        agg.update(
            bar_idx=200, timestamp=_ts(200),
            open_=7082.0, high=7085.0, low=7075.0, close=7078.0,
            volume=5000, velocity=-2.0,
        )

        assert agg._elevator_fb_active is False

    def test_velocity_short_also_suppressed(self):
        """Velocity shorts should also be suppressed during squeeze."""
        agg = _make_agg()
        agg._session_low = 7200.0
        agg._elevator_fb_active = True
        agg._elevator_recovery_low = 7080.0
        agg._elevator_recovery_drop = 80.0

        pattern = _make_short_pattern(entry_price=7110.0, stop_price=7120.0)
        signal = agg._qualify_short_signal(pattern, SignalType.VELOCITY_SHORT)

        assert signal is None
