"""Tests for short-side patterns: FailedRally + LevelRejection."""

from datetime import datetime, timedelta

import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.elevator_up import ElevatorUpDetector, ElevatorUpEvent, ElevatorUpState
from core.patterns import PatternState, ConfirmationType
from core.patterns_short import FailedRally, LevelRejection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(minute: int) -> datetime:
    return datetime(2024, 6, 15, 9, 30) + timedelta(minutes=minute)


def _make_resistance_store(price: float = 6000.0) -> LevelStore:
    store = LevelStore()
    store.add(Level(
        price=price,
        level_type=LevelType.PRIOR_DAY_HIGH,
        created_at=_ts(0),
        confirmed_at=_ts(0),
    ))
    return store


# ---------------------------------------------------------------------------
# Elevator Up
# ---------------------------------------------------------------------------

class TestElevatorUp:

    def test_detects_sharp_rally(self):
        detector = ElevatorUpDetector()
        store = LevelStore()
        store.add(Level(
            price=5990.0, level_type=LevelType.SWING_LOW,
            created_at=_ts(0), confirmed_at=_ts(0),
        ))
        store.add(Level(
            price=5995.0, level_type=LevelType.SWING_LOW,
            created_at=_ts(0), confirmed_at=_ts(0),
        ))

        # Bar 0: Rally starts at 6000, velocity = 2.0
        detector.update(0, _ts(0), high=6002.0, low=5999.0, close=6001.0,
                       velocity=2.0, level_store=store)
        assert detector.state == ElevatorUpState.ACTIVE

        # Bar 1-3: Rally continues, velocity stays high
        for i in range(1, 4):
            price = 6001.0 + i * 2.0
            detector.update(i, _ts(i), high=price + 1, low=price - 1,
                           close=price, velocity=2.5, level_store=store)
        assert detector.state == ElevatorUpState.ACTIVE

        # Bar 4-8: Velocity drops, lower high forms
        event = None
        for i in range(4, 9):
            price = 6007.0 - (i - 4) * 0.5
            event = detector.update(i, _ts(i), high=price, low=price - 1.5,
                                    close=price - 0.5, velocity=0.3, level_store=store)
            if event is not None:
                break

        assert event is not None
        assert event.is_complete
        assert event.high_price > 6005.0

    def test_no_detection_in_calm_market(self):
        detector = ElevatorUpDetector()
        store = LevelStore()

        for i in range(20):
            event = detector.update(i, _ts(i), high=6001.0, low=5999.0,
                                    close=6000.0, velocity=0.1, level_store=store)
            assert event is None
        assert detector.state == ElevatorUpState.IDLE


# ---------------------------------------------------------------------------
# Failed Rally
# ---------------------------------------------------------------------------

class TestFailedRally:

    def test_full_sequence_acceptance(self):
        """Elevator up sweeps resistance, price falls back below, holds = short signal."""
        params = StrategyParams(
            short_acceptance_min_hold_bars=3,
            short_acceptance_max_dip_pts=3.0,
            fr_stop_buffer_pts=4.5,
        )
        fr = FailedRally(params)
        store = _make_resistance_store(6000.0)

        # Create a completed elevator up event
        elevator = ElevatorUpEvent(
            start_idx=0, start_time=_ts(0), start_price=5990.0,
            end_idx=5, end_time=_ts(5), end_price=6002.0,
            high_price=6003.0, high_idx=3, peak_velocity=3.0, levels_broken=2,
        )

        # Bar 6: Elevator delivers, price swept above 6000, closes below.
        # Fast-track: sweep detected + recovery in same bar → acceptance watch.
        # Hold starts at 2 (entry bar + fast-track counts).
        signal = fr.update(6, _ts(6), high=6001.0, low=5997.0, close=5998.0,
                           level_store=store, elevator_event=elevator)
        assert fr.state == PatternState.ACCEPTANCE_WATCH

        # Bar 7: One more bar below level → hold=3 → confirmed
        signal = fr.update(7, _ts(7), high=5999.5, low=5997.0, close=5998.0,
                           level_store=store)

        assert signal is not None
        assert signal.pattern_type == "failed_rally"
        assert signal.direction == "short"
        assert signal.stop_price == 6000.0 + 4.5  # level + buffer

    def test_no_signal_without_elevator(self):
        """No FR signal without an elevator up (when allow_level_sweep_fb is False)."""
        params = StrategyParams(allow_level_sweep_fb=False)
        fr = FailedRally(params)
        store = _make_resistance_store(6000.0)

        for i in range(20):
            signal = fr.update(i, _ts(i), high=6001.0, low=5998.0, close=5999.0,
                               level_store=store)
            assert signal is None

    def test_reset_clears_state(self):
        fr = FailedRally()
        fr.state = PatternState.SWEEP_DETECTED
        fr.reset()
        assert fr.state == PatternState.IDLE


# ---------------------------------------------------------------------------
# Level Rejection
# ---------------------------------------------------------------------------

class TestLevelRejection:

    def test_rejection_from_above(self):
        """H/SR level with 4+ touches, price rejects from above = short signal."""
        params = StrategyParams(
            level_reclaim_min_touches=4,
            short_acceptance_min_hold_bars=3,
            lj_stop_buffer_pts=4.5,
        )
        lj = LevelRejection(params)

        store = LevelStore()
        store.add(Level(
            price=6000.0,
            level_type=LevelType.HORIZONTAL_SR,
            created_at=_ts(0),
            confirmed_at=_ts(0),
            touch_count=5,
        ))

        # Bar 0: Price was above, now closes below (rejection)
        signal = lj.update(0, _ts(0), high=6001.0, low=5998.0, close=5999.0,
                           level_store=store)
        assert signal is None  # need confirmation

        # Bars 1-3: Hold below level
        for i in range(1, 4):
            signal = lj.update(i, _ts(i), high=5999.5, low=5997.0, close=5998.5,
                               level_store=store)

        assert signal is not None
        assert signal.pattern_type == "level_rejection"
        assert signal.direction == "short"
        assert signal.stop_price == 6000.0 + 4.5

    def test_no_signal_if_spike_too_high(self):
        """Abort if price spikes too far above level during confirmation."""
        params = StrategyParams(
            level_reclaim_min_touches=4,
            short_acceptance_max_dip_pts=3.0,
            lj_stop_buffer_pts=4.5,
        )
        lj = LevelRejection(params)

        store = LevelStore()
        store.add(Level(
            price=6000.0,
            level_type=LevelType.HORIZONTAL_SR,
            created_at=_ts(0),
            confirmed_at=_ts(0),
            touch_count=5,
        ))

        # Bar 0: Rejection
        lj.update(0, _ts(0), high=6001.0, low=5998.0, close=5999.0,
                  level_store=store)

        # Bar 1: Spike 5 pts above level (> acceptance_max_dip_pts=3)
        signal = lj.update(1, _ts(1), high=6005.0, low=5998.0, close=5999.0,
                           level_store=store)

        assert signal is None
        assert lj.state == PatternState.IDLE  # aborted
