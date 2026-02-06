"""Tests for Failed Breakdown + Level Reclaim pattern detection."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams, ElevatorParams, DEFAULT_STRATEGY
from core.elevator_down import ElevatorDownDetector, ElevatorState
from core.indicators import compute_velocity
from core.patterns import (
    FailedBreakdown,
    LevelReclaim,
    PatternState,
    ConfirmationType,
)
from core.signals import SignalAggregator, SignalType
from tests.conftest import make_bars, make_selloff_then_recovery


class TestElevatorDown:
    """Test elevator down detection."""

    def test_detects_sharp_selloff(self, elevator_params):
        """Elevator down should be detected during a sharp selloff."""
        # Create a sharp selloff: 3+ pts/bar for 5+ bars
        prices = []
        p = 5050.0
        for i in range(15):
            o = p
            c = p - 3.5  # >2 pts/min velocity
            h = o + 0.5
            l = c - 0.5
            prices.append((o, h, l, c))
            p = c

        # Then a recovery (higher lows)
        for i in range(5):
            o = p
            c = p + 2.0
            h = c + 1.0
            l = o - 0.25
            prices.append((o, h, l, c))
            p = c

        df = make_bars(prices)
        velocity = compute_velocity(df, window=5)

        store = LevelStore()
        base_time = datetime(2024, 1, 15, 9, 0)
        store.add(Level(price=5040.0, level_type=LevelType.SWING_LOW,
                        created_at=base_time, confirmed_at=base_time))
        store.add(Level(price=5020.0, level_type=LevelType.SWING_LOW,
                        created_at=base_time, confirmed_at=base_time))

        detector = ElevatorDownDetector(elevator_params)
        events = detector.detect_all(df, velocity, store)

        assert len(events) >= 1
        event = events[0]
        assert event.is_complete
        assert event.total_drop_pts > 10

    def test_no_detection_in_calm_market(self, elevator_params):
        """No elevator down in a calm, sideways market."""
        prices = []
        p = 5000.0
        for i in range(30):
            o = p
            c = p + np.sin(i * 0.5) * 0.5  # tiny oscillation
            h = max(o, c) + 0.25
            l = min(o, c) - 0.25
            prices.append((o, h, l, c))
            p = c

        df = make_bars(prices)
        velocity = compute_velocity(df, window=5)
        store = LevelStore()

        detector = ElevatorDownDetector(elevator_params)
        events = detector.detect_all(df, velocity, store)

        assert len(events) == 0


class TestFailedBreakdown:
    """Test Failed Breakdown state machine."""

    def test_full_sequence_acceptance(self):
        """Test complete: elevator → sweep → recovery → acceptance confirmation."""
        df = make_selloff_then_recovery(
            start_price=5050.0,
            selloff_bars=10,
            selloff_rate=3.0,
            sweep_below=1.5,
            recovery_bars=5,
            hold_bars=5,
        )

        aggregator = SignalAggregator(min_rr_ratio=0.0)  # no R:R filter for this test

        # Add levels that will be swept
        base_time = datetime(2024, 1, 15, 9, 0)
        significant_low = 5050.0 - 10 * 3.0  # = 5020.0
        aggregator.level_store.add(Level(
            price=significant_low,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=base_time,
            confirmed_at=base_time,
            rally_from_low_pts=25.0,
        ))
        # Add resistance targets above
        aggregator.level_store.add(Level(
            price=5035.0,
            level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time,
            confirmed_at=base_time,
            touch_count=3,
        ))
        aggregator.level_store.add(Level(
            price=5050.0,
            level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time,
            confirmed_at=base_time,
            touch_count=3,
        ))

        velocity = compute_velocity(df, window=5)
        signals = []

        for i in range(len(df)):
            vel = float(velocity.iat[i]) if not np.isnan(velocity.iat[i]) else 0.0
            signal = aggregator.update(
                bar_idx=i,
                timestamp=df.index[i],
                open_=float(df["open"].iat[i]),
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                volume=float(df["volume"].iat[i]),
                velocity=vel,
            )
            if signal is not None:
                signals.append(signal)

        assert len(signals) >= 1, "Expected at least one signal from selloff → recovery"
        sig = signals[0]
        assert sig.signal_type == SignalType.FAILED_BREAKDOWN
        assert sig.entry_price > significant_low
        assert sig.stop_price < significant_low

    def test_no_signal_without_elevator(self, strategy_params):
        """No signal if there was no elevator down first."""
        fb = FailedBreakdown(strategy_params)

        # Just a small dip and recovery, no elevator
        prices = []
        p = 5020.0
        for i in range(5):
            o = p
            c = p - 0.5
            h = o + 0.25
            l = c - 0.25
            prices.append((o, h, l, c))
            p = c

        for i in range(5):
            o = p
            c = p + 0.5
            h = c + 0.25
            l = o - 0.25
            prices.append((o, h, l, c))
            p = c

        df = make_bars(prices)
        store = LevelStore()
        base_time = datetime(2024, 1, 15, 9, 0)
        store.add(Level(price=5018.0, level_type=LevelType.SWING_LOW,
                        created_at=base_time, confirmed_at=base_time))

        for i in range(len(df)):
            result = fb.update(
                bar_idx=i,
                timestamp=df.index[i],
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                level_store=store,
                elevator_event=None,
            )
            assert result is None, "Should not signal without elevator"

    def test_reset_clears_state(self, strategy_params):
        """Reset should return to IDLE."""
        fb = FailedBreakdown(strategy_params)
        fb.state = PatternState.SWEEP_DETECTED
        fb.reset()
        assert fb.state == PatternState.IDLE


class TestLevelReclaim:
    """Test Level Reclaim state machine."""

    def test_reclaim_from_below(self, strategy_params):
        """Level reclaim: price below S/R → crosses above → holds → signal."""
        lr = LevelReclaim(strategy_params)
        base_time = datetime(2024, 1, 15, 9, 0)

        store = LevelStore()
        store.add(Level(
            price=5020.0,
            level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time,
            confirmed_at=base_time,
            touch_count=5,
        ))
        store.add(Level(
            price=5040.0,
            level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time,
            confirmed_at=base_time,
            touch_count=3,
        ))

        # Bar 0: below the level, then cross above and hold for 5+ bars
        prices = [
            (5015.0, 5018.0, 5014.0, 5016.0),  # below 5020
            (5016.0, 5022.0, 5015.0, 5021.0),  # crosses above (low < 5020, close > 5020)
            (5021.0, 5023.0, 5019.5, 5022.0),  # holds above (dip only 0.5 pts)
            (5022.0, 5024.0, 5020.5, 5023.0),  # continues holding
            (5023.0, 5025.0, 5020.5, 5024.0),  # hold bar 3
            (5024.0, 5026.0, 5021.0, 5025.0),  # hold bar 4
            (5025.0, 5027.0, 5021.0, 5026.0),  # hold bar 5 — confirmation
        ]
        df = make_bars(prices, start=base_time + pd.Timedelta(minutes=30))

        signal = None
        for i in range(len(df)):
            result = lr.update(
                bar_idx=i,
                timestamp=df.index[i],
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                level_store=store,
            )
            if result is not None:
                signal = result

        assert signal is not None, "Expected a level reclaim signal"
        assert signal.pattern_type == "level_reclaim"
        assert signal.entry_price > 5020.0

    def test_no_signal_if_dip_too_deep(self, strategy_params):
        """No signal if price dips more than acceptance_max_dip_pts below level."""
        lr = LevelReclaim(strategy_params)
        base_time = datetime(2024, 1, 15, 9, 0)

        store = LevelStore()
        store.add(Level(
            price=5020.0,
            level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time,
            confirmed_at=base_time,
            touch_count=5,
        ))

        prices = [
            (5015.0, 5018.0, 5014.0, 5016.0),
            (5016.0, 5022.0, 5015.0, 5021.0),  # reclaim
            (5021.0, 5021.5, 5015.0, 5016.0),  # deep dip (5 pts below level)
        ]
        df = make_bars(prices, start=base_time + pd.Timedelta(minutes=30))

        signals = []
        for i in range(len(df)):
            result = lr.update(
                bar_idx=i,
                timestamp=df.index[i],
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                level_store=store,
            )
            if result is not None:
                signals.append(result)

        # Should have reset due to deep dip
        assert len(signals) == 0 or lr.state == PatternState.IDLE


class TestSignalAggregator:
    """Test full signal aggregation pipeline."""

    def test_run_bars_returns_signals(self):
        """run_bars should process a full DataFrame and return signals."""
        df = make_selloff_then_recovery(
            start_price=5060.0,
            selloff_bars=12,
            selloff_rate=3.0,
            sweep_below=2.0,
            recovery_bars=6,
            hold_bars=5,
        )

        aggregator = SignalAggregator(min_rr_ratio=0.0)
        base_time = datetime(2024, 1, 15, 9, 0)
        significant_low = 5060.0 - 12 * 3.0  # = 5024.0

        aggregator.level_store.add(Level(
            price=significant_low,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=base_time,
            confirmed_at=base_time,
            rally_from_low_pts=25.0,
        ))
        aggregator.level_store.add(Level(
            price=5045.0,
            level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time,
            confirmed_at=base_time,
            touch_count=3,
        ))
        aggregator.level_store.add(Level(
            price=5060.0,
            level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time,
            confirmed_at=base_time,
            touch_count=3,
        ))

        velocity = compute_velocity(df, window=5)
        signals = []
        for i in range(len(df)):
            vel = float(velocity.iat[i]) if not np.isnan(velocity.iat[i]) else 0.0
            sig = aggregator.update(
                bar_idx=i,
                timestamp=df.index[i],
                open_=float(df["open"].iat[i]),
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                volume=float(df["volume"].iat[i]),
                velocity=vel,
            )
            if sig is not None:
                signals.append(sig)

        # We should get at least one signal from the selloff→recovery pattern
        assert len(signals) >= 1


class TestTrueBreakdownAbort:
    """Test that the FB state machine aborts after 5+ bars closing below level."""

    def test_aborts_after_consecutive_bars_below(self, strategy_params):
        """If price stays below level for true_breakdown_abort_bars, reset to IDLE."""
        from core.elevator_down import ElevatorEvent

        fb = FailedBreakdown(strategy_params)
        base_time = datetime(2024, 1, 15, 9, 30)
        store = LevelStore()
        level = Level(
            price=5020.0,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=base_time,
            confirmed_at=base_time,
            rally_from_low_pts=25.0,
        )
        store.add(level)

        # Simulate a completed elevator that swept 5020
        elevator = ElevatorEvent(
            start_idx=0,
            start_time=base_time,
            start_price=5050.0,
            end_idx=9,
            end_time=base_time,
            end_price=5015.0,
            low_price=5010.0,
            low_idx=8,
            peak_velocity=4.0,
            levels_broken=3,
        )

        # First bar: elevator completes, sweep detected, but close below level
        result = fb.update(
            bar_idx=10,
            timestamp=base_time,
            high=5018.0,
            low=5010.0,
            close=5015.0,
            level_store=store,
            elevator_event=elevator,
        )
        assert fb.state == PatternState.SWEEP_DETECTED

        # 5 consecutive bars closing below level → should abort
        for i in range(strategy_params.true_breakdown_abort_bars):
            t = datetime(2024, 1, 15, 9, 31 + i)
            result = fb.update(
                bar_idx=11 + i,
                timestamp=t,
                high=5018.0,
                low=5012.0,
                close=5015.0,  # below 5020
                level_store=store,
                elevator_event=elevator,
            )

        assert fb.state == PatternState.IDLE, "Should abort after 5 bars below level"
        assert result is None

    def test_no_abort_if_recovery_interrupts(self, strategy_params):
        """If price closes above level mid-countdown, counter resets."""
        from core.elevator_down import ElevatorEvent

        fb = FailedBreakdown(strategy_params)
        base_time = datetime(2024, 1, 15, 9, 30)
        store = LevelStore()
        level = Level(
            price=5020.0,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=base_time,
            confirmed_at=base_time,
            rally_from_low_pts=25.0,
        )
        store.add(level)

        elevator = ElevatorEvent(
            start_idx=0, start_time=base_time, start_price=5050.0,
            end_idx=9, end_time=base_time, end_price=5015.0,
            low_price=5010.0, low_idx=8, peak_velocity=4.0, levels_broken=3,
        )

        # Trigger sweep detection with close below level
        fb.update(
            bar_idx=10, timestamp=base_time, high=5018.0, low=5010.0,
            close=5015.0, level_store=store, elevator_event=elevator,
        )
        assert fb.state == PatternState.SWEEP_DETECTED

        # 3 bars below (not enough to abort)
        for i in range(3):
            fb.update(
                bar_idx=11 + i,
                timestamp=datetime(2024, 1, 15, 9, 31 + i),
                high=5018.0, low=5012.0, close=5015.0,
                level_store=store, elevator_event=elevator,
            )
        assert fb.state == PatternState.SWEEP_DETECTED

        # Close above level → counter resets, state transitions to recovery
        result = fb.update(
            bar_idx=14,
            timestamp=datetime(2024, 1, 15, 9, 34),
            high=5025.0, low=5018.0, close=5022.0,
            level_store=store, elevator_event=elevator,
        )
        assert fb.state != PatternState.IDLE, "Should NOT have aborted"


class TestShallowVsDeepFlush:
    """Test that sweep depth affects acceptance hold requirements."""

    def test_shallow_flush_uses_standard_hold(self, strategy_params):
        """Shallow flush (<20 pts) uses acceptance_min_hold_bars (5)."""
        from core.elevator_down import ElevatorEvent

        fb = FailedBreakdown(strategy_params)
        base_time = datetime(2024, 1, 15, 9, 30)
        store = LevelStore()
        level = Level(
            price=5020.0, level_type=LevelType.MULTI_HOUR_LOW,
            created_at=base_time, confirmed_at=base_time,
            rally_from_low_pts=25.0,
        )
        store.add(level)

        # Sweep only 5 pts below (shallow)
        elevator = ElevatorEvent(
            start_idx=0, start_time=base_time, start_price=5050.0,
            end_idx=9, end_time=base_time, end_price=5022.0,
            low_price=5015.0, low_idx=8, peak_velocity=4.0, levels_broken=3,
        )

        # Trigger sweep + recovery in one bar (close above level)
        fb.update(
            bar_idx=10, timestamp=base_time, high=5023.0, low=5015.0,
            close=5022.0, level_store=store, elevator_event=elevator,
        )

        # Hold bars: need 5 for shallow
        signal = None
        for i in range(strategy_params.acceptance_min_hold_bars):
            t = datetime(2024, 1, 15, 9, 31 + i)
            result = fb.update(
                bar_idx=11 + i, timestamp=t,
                high=5024.0, low=5020.5, close=5023.0,
                level_store=store, elevator_event=elevator,
            )
            if result is not None:
                signal = result

        assert signal is not None, "Shallow flush should confirm after 5 hold bars"
        assert signal.sweep_depth_pts == 5.0

    def test_deep_flush_needs_more_hold_bars(self, strategy_params):
        """Deep flush (>=20 pts) requires acceptance_min_hold_bars_deep (15)."""
        from core.elevator_down import ElevatorEvent

        fb = FailedBreakdown(strategy_params)
        base_time = datetime(2024, 1, 15, 9, 30)
        store = LevelStore()
        level = Level(
            price=5020.0, level_type=LevelType.MULTI_HOUR_LOW,
            created_at=base_time, confirmed_at=base_time,
            rally_from_low_pts=25.0,
        )
        store.add(level)

        # Sweep 25 pts below (deep)
        elevator = ElevatorEvent(
            start_idx=0, start_time=base_time, start_price=5080.0,
            end_idx=9, end_time=base_time, end_price=5022.0,
            low_price=4995.0, low_idx=8, peak_velocity=6.0, levels_broken=5,
        )

        # Trigger sweep + recovery
        fb.update(
            bar_idx=10, timestamp=base_time, high=5023.0, low=4995.0,
            close=5022.0, level_store=store, elevator_event=elevator,
        )

        # Only 5 hold bars — should NOT confirm for deep flush
        signal = None
        for i in range(strategy_params.acceptance_min_hold_bars):
            t = datetime(2024, 1, 15, 9, 31 + i)
            result = fb.update(
                bar_idx=11 + i, timestamp=t,
                high=5024.0, low=5020.5, close=5023.0,
                level_store=store, elevator_event=elevator,
            )
            if result is not None:
                signal = result

        assert signal is None, "Deep flush should NOT confirm after only 5 bars"
        assert fb.state in (
            PatternState.ACCEPTANCE_WATCH,
            PatternState.NON_ACCEPTANCE_WATCH,
        )


class TestSweepDepthTracking:
    """Test that sweep_depth_pts is correctly recorded on signals."""

    def test_sweep_depth_on_fb_signal(self):
        """Failed breakdown signal should record how far below level the sweep went."""
        df = make_selloff_then_recovery(
            start_price=5050.0,
            selloff_bars=10,
            selloff_rate=3.0,
            sweep_below=3.0,
            recovery_bars=5,
            hold_bars=5,
        )
        aggregator = SignalAggregator(min_rr_ratio=0.0)
        base_time = datetime(2024, 1, 15, 9, 0)
        significant_low = 5050.0 - 10 * 3.0  # 5020.0

        aggregator.level_store.add(Level(
            price=significant_low, level_type=LevelType.MULTI_HOUR_LOW,
            created_at=base_time, confirmed_at=base_time,
            rally_from_low_pts=25.0,
        ))
        aggregator.level_store.add(Level(
            price=5035.0, level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time, confirmed_at=base_time, touch_count=3,
        ))
        aggregator.level_store.add(Level(
            price=5050.0, level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time, confirmed_at=base_time, touch_count=3,
        ))

        velocity = compute_velocity(df, window=5)
        signals = []
        for i in range(len(df)):
            vel = float(velocity.iat[i]) if not np.isnan(velocity.iat[i]) else 0.0
            sig = aggregator.update(
                bar_idx=i, timestamp=df.index[i],
                open_=float(df["open"].iat[i]),
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                volume=float(df["volume"].iat[i]),
                velocity=vel,
            )
            if sig is not None:
                signals.append(sig)

        assert len(signals) >= 1
        pattern = signals[0].pattern
        assert pattern.sweep_depth_pts > 0, "sweep_depth_pts should be positive"

    def test_level_reclaim_has_zero_sweep_depth(self, strategy_params):
        """Level reclaim signals should have sweep_depth_pts = 0."""
        lr = LevelReclaim(strategy_params)
        base_time = datetime(2024, 1, 15, 9, 0)

        store = LevelStore()
        store.add(Level(
            price=5020.0, level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time, confirmed_at=base_time, touch_count=5,
        ))
        store.add(Level(
            price=5040.0, level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time, confirmed_at=base_time, touch_count=3,
        ))

        prices = [
            (5015.0, 5018.0, 5014.0, 5016.0),
            (5016.0, 5022.0, 5015.0, 5021.0),
        ] + [
            (5021.0 + i, 5023.0 + i, 5020.5, 5022.0 + i) for i in range(6)
        ]
        df = make_bars(prices, start=base_time + pd.Timedelta(minutes=30))

        signal = None
        for i in range(len(df)):
            result = lr.update(
                bar_idx=i, timestamp=df.index[i],
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                level_store=store,
            )
            if result is not None:
                signal = result

        assert signal is not None
        assert signal.sweep_depth_pts == 0.0


class TestStopBuffer:
    """Test that the 5-point stop buffer is applied correctly."""

    def test_fb_stop_is_sweep_low_minus_5(self, strategy_params):
        """FB stop should be sweep_low - 5.0 pts."""
        from core.elevator_down import ElevatorEvent

        fb = FailedBreakdown(strategy_params)
        base_time = datetime(2024, 1, 15, 9, 30)
        store = LevelStore()
        store.add(Level(
            price=5020.0, level_type=LevelType.MULTI_HOUR_LOW,
            created_at=base_time, confirmed_at=base_time,
            rally_from_low_pts=25.0,
        ))

        elevator = ElevatorEvent(
            start_idx=0, start_time=base_time, start_price=5050.0,
            end_idx=9, end_time=base_time, end_price=5022.0,
            low_price=5015.0, low_idx=8, peak_velocity=4.0, levels_broken=3,
        )

        # Trigger sweep + recovery
        fb.update(
            bar_idx=10, timestamp=base_time, high=5023.0, low=5015.0,
            close=5022.0, level_store=store, elevator_event=elevator,
        )

        # Hold until confirmation
        signal = None
        for i in range(strategy_params.acceptance_min_hold_bars + 1):
            t = datetime(2024, 1, 15, 9, 31 + i)
            result = fb.update(
                bar_idx=11 + i, timestamp=t,
                high=5024.0, low=5020.5, close=5023.0,
                level_store=store, elevator_event=elevator,
            )
            if result is not None:
                signal = result

        assert signal is not None
        assert signal.stop_price == signal.sweep_low - 5.0

    def test_lr_stop_is_level_minus_5(self, strategy_params):
        """Level reclaim stop should be level_price - 5.0 pts."""
        lr = LevelReclaim(strategy_params)
        base_time = datetime(2024, 1, 15, 9, 0)

        store = LevelStore()
        store.add(Level(
            price=5020.0, level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time, confirmed_at=base_time, touch_count=5,
        ))
        store.add(Level(
            price=5040.0, level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time, confirmed_at=base_time, touch_count=3,
        ))

        prices = [
            (5015.0, 5018.0, 5014.0, 5016.0),
            (5016.0, 5022.0, 5015.0, 5021.0),
        ] + [
            (5021.0 + i, 5023.0 + i, 5020.5, 5022.0 + i) for i in range(6)
        ]
        df = make_bars(prices, start=base_time + pd.Timedelta(minutes=30))

        signal = None
        for i in range(len(df)):
            result = lr.update(
                bar_idx=i, timestamp=df.index[i],
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                level_store=store,
            )
            if result is not None:
                signal = result

        assert signal is not None
        assert signal.stop_price == 5020.0 - 5.0
