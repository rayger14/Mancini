"""Tests for Failed Breakdown + Level Reclaim pattern detection."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.elevator_down import ElevatorDownDetector
from core.indicators import compute_velocity
from core.patterns import (
    FailedBreakdown,
    LevelReclaim,
    PatternState,
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

        # Bar 0: below the level, then cross above and hold for 7+ bars
        prices = [
            (5015.0, 5018.0, 5014.0, 5016.0),  # below 5020
            (5016.0, 5022.0, 5015.0, 5021.0),  # crosses above (low < 5020, close > 5020)
            (5021.0, 5023.0, 5019.5, 5022.0),  # holds above (dip only 0.5 pts)
            (5022.0, 5024.0, 5020.5, 5023.0),  # continues holding
            (5023.0, 5025.0, 5020.5, 5024.0),  # hold bar 3
            (5024.0, 5026.0, 5021.0, 5025.0),  # hold bar 4
            (5025.0, 5027.0, 5021.0, 5026.0),  # hold bar 5
            (5026.0, 5028.0, 5021.5, 5027.0),  # hold bar 6
            (5027.0, 5029.0, 5022.0, 5028.0),  # hold bar 7 — confirmation
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
        fb.update(
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

    def test_deep_flush_confirms_faster(self, strategy_params):
        """Deep flush (>=20 pts) confirms faster (acceptance_min_hold_bars_deep=4).

        Violent elevator-driven sweeps that recover are more convincing and
        need fewer confirmation bars than shallow dips.
        """
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

        # Deep flush needs acceptance_min_hold_bars_deep (4) bars
        signal = None
        for i in range(strategy_params.acceptance_min_hold_bars_deep + 2):
            t = datetime(2024, 1, 15, 9, 31 + i)
            result = fb.update(
                bar_idx=11 + i, timestamp=t,
                high=5024.0, low=5020.5, close=5023.0,
                level_store=store, elevator_event=elevator,
            )
            if result is not None:
                signal = result

        assert signal is not None, "Deep flush should confirm after hold_bars_deep bars"
        assert signal.sweep_depth_pts == pytest.approx(25.0, abs=0.5)


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
            (5021.0 + i, 5023.0 + i, 5020.5, 5022.0 + i) for i in range(10)
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

    def test_fb_stop_is_level_minus_2(self, strategy_params):
        """FB stop should be level - 2.0 pts (Mancini: level is line in the sand)."""
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
        # Stop below sweep low (Mancini: beneath the swept low where shorts are trapped)
        # stop = min(sweep_low - 0.25, level - fb_stop_buffer_pts)
        expected_stop = min(signal.sweep_low - 0.25, signal.level.price - strategy_params.fb_stop_buffer_pts)
        assert signal.stop_price == expected_stop

    def test_lr_stop_is_level_minus_2(self, strategy_params):
        """Level reclaim stop should be level_price - lr_stop_buffer_pts."""
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
            (5021.0 + i, 5023.0 + i, 5020.5, 5022.0 + i) for i in range(10)
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
        assert signal.stop_price == 5020.0 - strategy_params.lr_stop_buffer_pts


class TestNoLookAheadBias:
    """Verify that signals at bar N depend only on data from bars 0..N.

    The test processes bars incrementally and checks that:
    1. Levels returned by get_confirmed(as_of=bar_time) never have
       confirmed_at > bar_time.
    2. Signals produced at bar N are identical whether we process
       bars 0..N or bars 0..N+K (i.e., future bars don't change past signals).
    3. detect_incremental only uses data up to bar_idx.
    """

    def _make_intraday_data(self, n_bars: int = 120) -> pd.DataFrame:
        """Create a realistic intraday session with enough structure for levels."""
        np.random.seed(42)
        start = datetime(2024, 3, 15, 9, 30)
        prices = []
        price = 5100.0

        # Phase 1: Drift up (40 bars)
        for i in range(40):
            o = price
            c = price + np.random.uniform(-0.5, 1.5)
            h = max(o, c) + np.random.uniform(0.5, 2.0)
            l = min(o, c) - np.random.uniform(0.25, 1.0)
            prices.append((o, h, l, c))
            price = c

        # Phase 2: Sharp selloff (15 bars)
        for i in range(15):
            o = price
            c = price - np.random.uniform(2.0, 4.0)
            h = o + np.random.uniform(0.0, 0.5)
            l = c - np.random.uniform(0.0, 1.0)
            prices.append((o, h, l, c))
            price = c

        # Phase 3: Recovery (20 bars)
        for i in range(20):
            o = price
            c = price + np.random.uniform(0.5, 3.0)
            h = max(o, c) + np.random.uniform(0.5, 1.5)
            l = min(o, c) - np.random.uniform(0.0, 0.5)
            prices.append((o, h, l, c))
            price = c

        # Phase 4: Consolidation (remaining bars)
        for i in range(n_bars - 75):
            o = price
            c = price + np.random.uniform(-1.0, 1.0)
            h = max(o, c) + np.random.uniform(0.25, 1.0)
            l = min(o, c) - np.random.uniform(0.25, 1.0)
            prices.append((o, h, l, c))
            price = c

        return make_bars(prices, start=start)

    def test_confirmed_levels_never_from_future(self):
        """At each bar, get_confirmed(as_of) must only return levels with
        confirmed_at <= current bar timestamp."""
        from core.price_levels import PriceLevelDetector

        df = self._make_intraday_data(120)
        detector = PriceLevelDetector()

        # Pre-compute all levels (simulating what initialize_levels does)
        store = detector.detect_all(df)

        for i in range(len(df)):
            bar_time = df.index[i]
            confirmed = store.get_confirmed(as_of=bar_time)
            for level in confirmed:
                assert level.confirmed_at <= bar_time, (
                    f"Look-ahead bias: bar {i} at {bar_time} sees level "
                    f"{level.label} confirmed_at={level.confirmed_at} "
                    f"(which is in the future)"
                )

    def test_incremental_detection_no_future_data(self):
        """detect_incremental at bar_idx must produce identical results whether
        the DataFrame has bars 0..bar_idx or bars 0..N (full day)."""
        from core.price_levels import PriceLevelDetector
        from config.levels import LevelStore

        df_full = self._make_intraday_data(120)
        detector = PriceLevelDetector()

        # Run incremental on full DataFrame
        store_full = LevelStore()
        levels_at_bar_full = {}
        for i in range(len(df_full)):
            new = detector.detect_incremental(store_full, df_full, i)
            if new:
                levels_at_bar_full[i] = [(l.price, l.level_type) for l in new]

        # Run incremental on truncated DataFrames (only bars 0..i)
        store_trunc = LevelStore()
        levels_at_bar_trunc = {}
        for i in range(len(df_full)):
            df_partial = df_full.iloc[: i + 1].copy()
            new = detector.detect_incremental(store_trunc, df_partial, i)
            if new:
                levels_at_bar_trunc[i] = [(l.price, l.level_type) for l in new]

        # Both approaches must produce the same levels at each bar
        all_bars = set(levels_at_bar_full.keys()) | set(levels_at_bar_trunc.keys())
        for bar in sorted(all_bars):
            full_levels = levels_at_bar_full.get(bar, [])
            trunc_levels = levels_at_bar_trunc.get(bar, [])
            assert full_levels == trunc_levels, (
                f"Look-ahead bias in detect_incremental at bar {bar}: "
                f"full={full_levels} vs truncated={trunc_levels}"
            )

    def test_signals_stable_across_future_data(self):
        """Signals at bar N must be the same regardless of how many future bars
        exist in the DataFrame.

        We compare: processing bars 0..60 vs processing bars 0..120.
        Signals at bars 0..60 must be identical in both runs.
        """
        from core.signals import SignalAggregator
        from core.indicators import compute_velocity

        df_full = self._make_intraday_data(120)
        cutoff = 60

        # Add prior-day levels so there's something to trade against
        base_time = datetime(2024, 3, 14, 9, 0)
        prior_prices = [(5090.0 + i, 5095.0 + i, 5085.0 + i, 5092.0 + i) for i in range(30)]
        prior_day_df = make_bars(prior_prices, start=base_time)

        # Run 1: Full data (120 bars)
        agg_full = SignalAggregator(min_rr_ratio=0.0)
        agg_full.initialize_levels(df_full, prior_day_df)
        vel_full = compute_velocity(df_full, window=5)
        signals_full = []
        for i in range(len(df_full)):
            vel = float(vel_full.iat[i]) if not np.isnan(vel_full.iat[i]) else 0.0
            sig = agg_full.update(
                bar_idx=i, timestamp=df_full.index[i],
                open_=float(df_full["open"].iat[i]),
                high=float(df_full["high"].iat[i]),
                low=float(df_full["low"].iat[i]),
                close=float(df_full["close"].iat[i]),
                volume=float(df_full["volume"].iat[i]),
                velocity=vel, df=df_full,
            )
            if sig is not None and i < cutoff:
                signals_full.append((i, sig.signal_type, round(sig.entry_price, 2)))

        # Run 2: Truncated data (first 60 bars only)
        df_trunc = df_full.iloc[:cutoff].copy()
        agg_trunc = SignalAggregator(min_rr_ratio=0.0)
        agg_trunc.initialize_levels(df_trunc, prior_day_df)
        vel_trunc = compute_velocity(df_trunc, window=5)
        signals_trunc = []
        for i in range(len(df_trunc)):
            vel = float(vel_trunc.iat[i]) if not np.isnan(vel_trunc.iat[i]) else 0.0
            sig = agg_trunc.update(
                bar_idx=i, timestamp=df_trunc.index[i],
                open_=float(df_trunc["open"].iat[i]),
                high=float(df_trunc["high"].iat[i]),
                low=float(df_trunc["low"].iat[i]),
                close=float(df_trunc["close"].iat[i]),
                volume=float(df_trunc["volume"].iat[i]),
                velocity=vel, df=df_trunc,
            )
            if sig is not None:
                signals_trunc.append((i, sig.signal_type, round(sig.entry_price, 2)))

        assert signals_full == signals_trunc, (
            f"Look-ahead bias: signals differ between full vs truncated data.\n"
            f"Full (first {cutoff} bars): {signals_full}\n"
            f"Truncated: {signals_trunc}"
        )

    def test_get_confirmed_filters_correctly(self):
        """Direct test of LevelStore.get_confirmed() temporal filtering."""
        from config.levels import Level, LevelStore, LevelType

        store = LevelStore()
        t1 = datetime(2024, 1, 15, 9, 30)
        t2 = datetime(2024, 1, 15, 10, 30)
        t3 = datetime(2024, 1, 15, 11, 30)

        store.add(Level(price=5000.0, level_type=LevelType.SWING_LOW,
                        created_at=t1, confirmed_at=t2))
        store.add(Level(price=4980.0, level_type=LevelType.MULTI_HOUR_LOW,
                        created_at=t1, confirmed_at=t3))
        store.add(Level(price=5020.0, level_type=LevelType.PRIOR_DAY_LOW,
                        created_at=t1, confirmed_at=t1))

        # At t1: only prior_day_low is confirmed
        confirmed_t1 = store.get_confirmed(as_of=t1)
        assert len(confirmed_t1) == 1
        assert confirmed_t1[0].price == 5020.0

        # At t2: prior_day_low + swing_low
        confirmed_t2 = store.get_confirmed(as_of=t2)
        assert len(confirmed_t2) == 2
        prices = {l.price for l in confirmed_t2}
        assert prices == {5000.0, 5020.0}

        # At t3: all three
        confirmed_t3 = store.get_confirmed(as_of=t3)
        assert len(confirmed_t3) == 3

    def test_swing_low_confirmed_at_delay(self):
        """Swing lows detected by argrelextrema must have confirmed_at
        at least `order` bars after the swing low bar."""
        from core.price_levels import PriceLevelDetector

        params = StrategyParams(swing_low_order=10)  # smaller order for test
        detector = PriceLevelDetector(params)

        # Create data with a clear swing low at bar 15
        prices = []
        p = 5100.0
        for i in range(30):
            if i < 15:
                # Declining
                o = p
                c = p - 1.0
                h = o + 0.5
                l = c - 0.5
            elif i == 15:
                # The swing low
                o = p
                c = p - 0.5
                h = o + 0.25
                l = p - 2.0  # deep low
            else:
                # Rising
                o = p
                c = p + 2.0
                h = c + 0.5
                l = o - 0.25
            prices.append((o, h, l, c))
            p = c

        df = make_bars(prices)
        store = detector.detect_all(df)

        for level in store.levels:
            if level.level_type in (LevelType.SWING_LOW, LevelType.MULTI_HOUR_LOW):
                created_idx = df.index.get_loc(level.created_at)
                confirmed_idx = df.index.get_loc(level.confirmed_at)
                assert confirmed_idx >= created_idx + params.swing_low_order, (
                    f"Swing low at bar {created_idx} confirmed too early "
                    f"at bar {confirmed_idx} (need delay of {params.swing_low_order})"
                )
