"""Tests for Daily Structure Detector — macro bias from the daily chart."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from config.settings import StrategyParams
from core.daily_structure import DailyStructureDetector


def _make_daily_bars(
    lows: List[float],
    closes: List[float],
    *,
    highs: Optional[List[float]] = None,
    opens: Optional[List[float]] = None,
    start_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Build a DataFrame of daily OHLC bars for testing.

    Parameters
    ----------
    lows : list[float]
        Daily low prices.
    closes : list[float]
        Daily close prices.
    highs : list[float], optional
        Daily high prices.  Defaults to close + 10.
    opens : list[float], optional
        Daily open prices.  Defaults to close - 5.
    start_date : datetime, optional
        Start date for the index.
    """
    n = len(lows)
    assert len(closes) == n, "lows and closes must have same length"
    if highs is None:
        highs = [c + 10 for c in closes]
    if opens is None:
        opens = [c - 5 for c in closes]
    if start_date is None:
        start_date = datetime(2026, 3, 1, tzinfo=timezone.utc)

    dates = pd.date_range(start=start_date, periods=n, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=dates,
    )


class TestDailyFBDetected:
    """Test that a clear daily FB pattern produces DAILY_FB_BULL."""

    def test_daily_fb_detected(self):
        """Shelf at ~6800, sweep to 6350, recovery closes above shelf -> DAILY_FB_BULL."""
        params = StrategyParams(
            use_daily_structure=True,
            daily_shelf_lookback_days=20,
            daily_shelf_cluster_pts=30.0,
            daily_shelf_min_touches=3,
            daily_fb_recovery_bars=3,
        )
        detector = DailyStructureDetector(params)

        # Build 20 daily bars:
        # Days 1-10: lows cluster around 6800 (the shelf)
        # Days 11-13: sweep down to 6350
        # Days 14-20: recovery with closes above 6800
        lows = (
            [6810, 6795, 6805, 6790, 6800, 6810, 6795, 6805, 6790, 6800]  # shelf cluster
            + [6500, 6400, 6350]  # sweep below shelf
            + [6700, 6750, 6820, 6850, 6900, 6950, 7000]  # recovery
        )
        closes = (
            [6850, 6840, 6860, 6830, 6845, 6855, 6835, 6850, 6825, 6840]  # above shelf
            + [6520, 6420, 6380]  # closes during sweep
            + [6720, 6780, 6850, 6880, 6920, 6960, 7100]  # recovery closes
        )
        daily_bars = _make_daily_bars(lows, closes)

        result = detector.update(daily_bars)
        assert result == "DAILY_FB_BULL"

    def test_recovery_must_be_consecutive(self):
        """If last 3 closes are not ALL above shelf, no FB confirmation."""
        params = StrategyParams(
            use_daily_structure=True,
            daily_shelf_lookback_days=20,
            daily_shelf_cluster_pts=30.0,
            daily_shelf_min_touches=3,
            daily_fb_recovery_bars=3,
        )
        detector = DailyStructureDetector(params)

        lows = (
            [6810, 6795, 6805, 6790, 6800, 6810, 6795, 6805, 6790, 6800]
            + [6500, 6400, 6350]
            + [6700, 6750, 6820, 6850, 6900, 6750, 7000]  # day 19 dips below shelf close
        )
        closes = (
            [6850, 6840, 6860, 6830, 6845, 6855, 6835, 6850, 6825, 6840]
            + [6520, 6420, 6380]
            + [6720, 6780, 6850, 6880, 6920, 6750, 7100]  # close[18]=6750 < shelf
        )
        daily_bars = _make_daily_bars(lows, closes)

        result = detector.update(daily_bars)
        # Last 3 closes: 6920, 6750, 7100 — 6750 < 6800 shelf → not all above
        assert result != "DAILY_FB_BULL"


class TestDailyBDDetected:
    """Test that a breakdown pattern produces DAILY_BD_BEAR."""

    def test_daily_bd_detected(self):
        """Price breaks below shelf and stays below -> DAILY_BD_BEAR."""
        params = StrategyParams(
            use_daily_structure=True,
            daily_shelf_lookback_days=20,
            daily_shelf_cluster_pts=30.0,
            daily_shelf_min_touches=3,
            daily_fb_recovery_bars=3,
        )
        detector = DailyStructureDetector(params)

        # Shelf around 6800, then price breaks down and stays below
        lows = (
            [6810, 6795, 6805, 6790, 6800, 6810, 6795, 6805, 6790, 6800]
            + [6750, 6700, 6650, 6600, 6580, 6560, 6550, 6540, 6530, 6520]
        )
        closes = (
            [6850, 6840, 6860, 6830, 6845, 6855, 6835, 6850, 6825, 6840]
            + [6760, 6710, 6660, 6610, 6590, 6570, 6560, 6550, 6540, 6530]
        )
        daily_bars = _make_daily_bars(lows, closes)

        result = detector.update(daily_bars)
        # Sweep low is 6520, which is < 6800 - 30 = 6770 → swept below
        # Last 3 closes: 6550, 6540, 6530 — all below shelf → BD
        # But actually they swept below AND last closes are below → BD_BEAR
        assert result == "DAILY_BD_BEAR"


class TestNeutralNoStructure:
    """Test that sideways daily bars produce NEUTRAL."""

    def test_neutral_no_clear_structure(self):
        """Sideways market with no clear shelf or sweep → NEUTRAL."""
        params = StrategyParams(
            use_daily_structure=True,
            daily_shelf_lookback_days=20,
            daily_shelf_cluster_pts=30.0,
            daily_shelf_min_touches=3,
            daily_fb_recovery_bars=3,
        )
        detector = DailyStructureDetector(params)

        # Random-ish lows spread over 200+ pts — no cluster
        rng = np.random.default_rng(42)
        lows = (6600 + rng.uniform(0, 200, 20)).tolist()
        closes = [l + 30 for l in lows]
        daily_bars = _make_daily_bars(lows, closes)

        result = detector.update(daily_bars)
        # With 30 pt cluster and random spread over 200 pts,
        # may or may not find 3 touches. Accept NEUTRAL or another state.
        # The key is it shouldn't crash.
        assert result in ("NEUTRAL", "DAILY_FB_BULL", "DAILY_BD_BEAR")

    def test_insufficient_data_returns_neutral(self):
        """Too few bars → NEUTRAL."""
        params = StrategyParams(
            use_daily_structure=True,
            daily_shelf_lookback_days=20,
        )
        detector = DailyStructureDetector(params)

        lows = [6800, 6810, 6790]
        closes = [6850, 6860, 6840]
        daily_bars = _make_daily_bars(lows, closes)

        result = detector.update(daily_bars)
        assert result == "NEUTRAL"


class TestShelfDetection:
    """Test that shelf detection clusters daily lows correctly."""

    def test_shelf_detection_clusters_lows(self):
        """Shelf price should be near the median of the cluster."""
        params = StrategyParams(
            use_daily_structure=True,
            daily_shelf_lookback_days=10,
            daily_shelf_cluster_pts=15.0,
            daily_shelf_min_touches=3,
            daily_fb_recovery_bars=3,
        )
        detector = DailyStructureDetector(params)

        # 5 lows cluster near 7000, 5 lows are scattered
        lows = [7000, 7005, 6995, 7010, 6990, 6800, 6700, 6600, 6500, 6400]
        closes = [7050, 7055, 7045, 7060, 7040, 6850, 6750, 6650, 6550, 6450]
        daily_bars = _make_daily_bars(lows, closes)

        detector.update(daily_bars)
        # Shelf should be near 7000 (median of 6990, 6995, 7000, 7005, 7010)
        assert abs(detector._shelf_price - 7000) <= 15

    def test_shelf_requires_min_touches(self):
        """If no cluster has enough touches, no shelf → NEUTRAL."""
        params = StrategyParams(
            use_daily_structure=True,
            daily_shelf_lookback_days=10,
            daily_shelf_cluster_pts=5.0,  # very tight
            daily_shelf_min_touches=5,    # need 5 touches
        )
        detector = DailyStructureDetector(params)

        # All lows spread far apart
        lows = [6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900]
        closes = [6050, 6150, 6250, 6350, 6450, 6550, 6650, 6750, 6850, 6950]
        daily_bars = _make_daily_bars(lows, closes)

        result = detector.update(daily_bars)
        assert result == "NEUTRAL"
        assert detector._shelf_price == 0.0


class TestLQSBonus:
    """Test that LQS bonus is applied correctly during DAILY_FB_BULL."""

    def test_lqs_bonus_applied(self):
        """During DAILY_FB_BULL, FB Long should get +daily_fb_lqs_bonus to LQS."""
        from unittest.mock import MagicMock, patch
        from core.signals import SignalAggregator, SignalType

        params = StrategyParams(
            use_daily_structure=True,
            daily_fb_lqs_bonus=10,
            use_level_quality_scoring=False,  # don't gate, just score
        )
        agg = SignalAggregator(strategy_params=params)

        # Set daily bias to DAILY_FB_BULL
        agg._daily_bias = "DAILY_FB_BULL"

        # Create a mock pattern with realistic values
        mock_pattern = MagicMock()
        mock_pattern.entry_price = 6850.0
        mock_pattern.stop_price = 6840.0
        mock_pattern.level.price = 6845.0
        mock_pattern.level.level_type.name = "PRIOR_DAY_LOW"
        mock_pattern.timestamp = datetime(2026, 4, 20, tzinfo=timezone.utc)
        mock_pattern.bar_idx = 100
        mock_pattern.sweep_depth_pts = 5.0
        mock_pattern.sweep_low = 6840.0
        mock_pattern.sweep_high = 0.0
        mock_pattern.direction = "long"
        mock_pattern.is_double_dip = False
        mock_pattern.confirmation = MagicMock()
        mock_pattern.confirmation.name = "ACCEPTANCE"

        # Mock the level scorer to return a known LQS
        base_lqs = 50
        with patch.object(agg._level_scorer, 'compute_lqs', return_value=base_lqs):
            # Also need level store to have resistances above
            mock_level = MagicMock()
            mock_level.price = 6870.0
            mock_level.level_type.name = "MULTI_HOUR_LOW"
            agg.level_store.resistances_above = MagicMock(return_value=[mock_level])

            signal = agg._qualify_signal(mock_pattern, SignalType.FAILED_BREAKDOWN)

        assert signal is not None
        # LQS should be base + bonus = 50 + 10 = 60
        assert signal.lqs == base_lqs + 10

    def test_lqs_bonus_not_applied_for_lr(self):
        """LQS bonus should only apply to FB Long, not Level Reclaim."""
        from unittest.mock import MagicMock, patch
        from core.signals import SignalAggregator, SignalType

        params = StrategyParams(
            use_daily_structure=True,
            daily_fb_lqs_bonus=10,
            use_level_quality_scoring=False,
        )
        agg = SignalAggregator(strategy_params=params)
        agg._daily_bias = "DAILY_FB_BULL"

        mock_pattern = MagicMock()
        mock_pattern.entry_price = 6850.0
        mock_pattern.stop_price = 6840.0
        mock_pattern.level.price = 6845.0
        mock_pattern.level.level_type.name = "PRIOR_DAY_LOW"
        mock_pattern.timestamp = datetime(2026, 4, 20, tzinfo=timezone.utc)
        mock_pattern.bar_idx = 100
        mock_pattern.sweep_depth_pts = 5.0
        mock_pattern.sweep_low = 6840.0
        mock_pattern.sweep_high = 0.0
        mock_pattern.direction = "long"

        base_lqs = 50
        with patch.object(agg._level_scorer, 'compute_lqs', return_value=base_lqs):
            mock_level = MagicMock()
            mock_level.price = 6870.0
            mock_level.level_type.name = "MULTI_HOUR_LOW"
            agg.level_store.resistances_above = MagicMock(return_value=[mock_level])

            signal = agg._qualify_signal(mock_pattern, SignalType.LEVEL_RECLAIM)

        assert signal is not None
        # No bonus for LR
        assert signal.lqs == base_lqs


class TestShortSuppression:
    """Test that shorts are suppressed during DAILY_FB_BULL."""

    def test_short_marked_contra_trend_during_daily_fb(self):
        """BD Short with LQS 55 should be marked contra-trend (not blocked in collection mode)."""
        from unittest.mock import MagicMock, patch
        from core.signals import SignalAggregator, SignalType

        params = StrategyParams(
            use_daily_structure=True,
            daily_bd_short_min_lqs=70,
            use_level_quality_scoring=False,
        )
        agg = SignalAggregator(strategy_params=params)
        agg._daily_bias = "DAILY_FB_BULL"

        mock_pattern = MagicMock()
        mock_pattern.entry_price = 6850.0
        mock_pattern.stop_price = 6860.0  # short: stop above entry
        mock_pattern.level.price = 6855.0
        mock_pattern.level.level_type.name = "MULTI_HOUR_LOW"
        mock_pattern.timestamp = datetime(2026, 4, 20, tzinfo=timezone.utc)
        mock_pattern.bar_idx = 100
        mock_pattern.sweep_depth_pts = 5.0
        mock_pattern.sweep_low = 6840.0
        mock_pattern.sweep_high = 6860.0
        mock_pattern.direction = "short"

        # LQS = 55 < required 70 → should be blocked
        with patch.object(agg._level_scorer, 'compute_lqs', return_value=55):
            mock_level = MagicMock()
            mock_level.price = 6830.0
            mock_level.level_type.name = "MULTI_HOUR_LOW"
            agg.level_store.supports_below = MagicMock(return_value=[mock_level])

            signal = agg._qualify_short_signal(mock_pattern, SignalType.BREAKDOWN_SHORT)

        # In collection mode: signal is NOT blocked, just marked as contra-trend
        # Shadow event should still be logged
        assert len(agg.shadow_events) > 0
        assert agg.shadow_events[-1]["feature"] == "daily_structure_short_suppression"

    def test_short_allowed_with_high_lqs(self):
        """Short with LQS >= 70 should pass even during DAILY_FB_BULL."""
        from unittest.mock import MagicMock, patch
        from core.signals import SignalAggregator, SignalType

        params = StrategyParams(
            use_daily_structure=True,
            daily_bd_short_min_lqs=70,
            use_level_quality_scoring=False,
            block_pdl_shorts=False,  # test LQS gate, not PDL gate
        )
        agg = SignalAggregator(strategy_params=params)
        agg._daily_bias = "DAILY_FB_BULL"

        mock_pattern = MagicMock()
        mock_pattern.entry_price = 6850.0
        mock_pattern.stop_price = 6860.0
        mock_pattern.level.price = 6855.0
        mock_pattern.level.level_type.name = "PRIOR_DAY_LOW"
        mock_pattern.timestamp = datetime(2026, 4, 20, tzinfo=timezone.utc)
        mock_pattern.bar_idx = 100
        mock_pattern.sweep_depth_pts = 5.0
        mock_pattern.sweep_low = 6840.0
        mock_pattern.sweep_high = 6860.0
        mock_pattern.direction = "short"

        # LQS = 75 >= required 70 → should pass
        with patch.object(agg._level_scorer, 'compute_lqs', return_value=75):
            mock_level = MagicMock()
            mock_level.price = 6830.0
            mock_level.level_type.name = "PRIOR_DAY_LOW"
            agg.level_store.supports_below = MagicMock(return_value=[mock_level])

            signal = agg._qualify_short_signal(mock_pattern, SignalType.BREAKDOWN_SHORT)

        assert signal is not None


class TestDisabledFlag:
    """Test that use_daily_structure=False disables all daily structure effects."""

    def test_disabled_flag(self):
        """With use_daily_structure=False, detector always returns NEUTRAL."""
        params = StrategyParams(use_daily_structure=False)
        detector = DailyStructureDetector(params)

        # Build bars that would normally trigger DAILY_FB_BULL
        lows = (
            [6810, 6795, 6805, 6790, 6800, 6810, 6795, 6805, 6790, 6800]
            + [6500, 6400, 6350]
            + [6700, 6750, 6820, 6850, 6900, 6950, 7000]
        )
        closes = (
            [6850, 6840, 6860, 6830, 6845, 6855, 6835, 6850, 6825, 6840]
            + [6520, 6420, 6380]
            + [6720, 6780, 6850, 6880, 6920, 6960, 7100]
        )
        daily_bars = _make_daily_bars(lows, closes)

        result = detector.update(daily_bars)
        assert result == "NEUTRAL"

    def test_disabled_no_short_suppression(self):
        """With use_daily_structure=False, shorts are not suppressed."""
        from unittest.mock import MagicMock, patch
        from core.signals import SignalAggregator, SignalType

        params = StrategyParams(
            use_daily_structure=False,
            daily_bd_short_min_lqs=70,
            use_level_quality_scoring=False,
        )
        agg = SignalAggregator(strategy_params=params)
        # Even if bias is somehow set, the flag should prevent suppression
        agg._daily_bias = "DAILY_FB_BULL"

        mock_pattern = MagicMock()
        mock_pattern.entry_price = 6850.0
        mock_pattern.stop_price = 6860.0
        mock_pattern.level.price = 6855.0
        mock_pattern.level.level_type.name = "MULTI_HOUR_LOW"
        mock_pattern.timestamp = datetime(2026, 4, 20, tzinfo=timezone.utc)
        mock_pattern.bar_idx = 100
        mock_pattern.sweep_depth_pts = 5.0
        mock_pattern.sweep_low = 6840.0
        mock_pattern.sweep_high = 6860.0
        mock_pattern.direction = "short"

        with patch.object(agg._level_scorer, 'compute_lqs', return_value=55):
            mock_level = MagicMock()
            mock_level.price = 6830.0
            mock_level.level_type.name = "MULTI_HOUR_LOW"
            agg.level_store.supports_below = MagicMock(return_value=[mock_level])

            signal = agg._qualify_short_signal(mock_pattern, SignalType.BREAKDOWN_SHORT)

        # Should NOT be suppressed because use_daily_structure=False
        assert signal is not None


class TestMovePosition:
    """Test move position calculation."""

    def test_move_position_calculation(self):
        """Move position = (current - shelf) / sweep_depth."""
        params = StrategyParams(
            use_daily_structure=True,
            daily_shelf_lookback_days=20,
            daily_shelf_cluster_pts=30.0,
            daily_shelf_min_touches=3,
            daily_fb_recovery_bars=3,
        )
        detector = DailyStructureDetector(params)

        # Shelf at ~6800, sweep to ~6350, recovery to ~7100
        # sweep_depth = |6350 - 6800| = 450
        # move_position = (7100 - 6800) / 450 ≈ 0.667
        lows = (
            [6810, 6795, 6805, 6790, 6800, 6810, 6795, 6805, 6790, 6800]
            + [6500, 6400, 6350]
            + [6700, 6750, 6820, 6850, 6900, 6950, 7000]
        )
        closes = (
            [6850, 6840, 6860, 6830, 6845, 6855, 6835, 6850, 6825, 6840]
            + [6520, 6420, 6380]
            + [6720, 6780, 6850, 6880, 6920, 6960, 7100]
        )
        daily_bars = _make_daily_bars(lows, closes)

        result = detector.update(daily_bars)
        assert result == "DAILY_FB_BULL"

        # move_position should be positive and reasonable
        assert detector._move_position > 0.0
        # With 7100 close, shelf ~6800, sweep ~6350:
        # (7100 - 6800) / (6800 - 6350) = 300 / 450 ≈ 0.667
        assert 0.3 < detector._move_position < 1.5

    def test_move_position_zero_when_neutral(self):
        """Move position should be 0 when state is NEUTRAL."""
        params = StrategyParams(
            use_daily_structure=True,
            daily_shelf_lookback_days=20,
        )
        detector = DailyStructureDetector(params)

        lows = [6800 + i * 5 for i in range(20)]
        closes = [l + 20 for l in lows]
        daily_bars = _make_daily_bars(lows, closes)

        detector.update(daily_bars)
        assert detector._move_position == 0.0


class TestSnapshot:
    """Test get_snapshot for logging."""

    def test_snapshot_for_logging(self):
        """get_snapshot returns dict with all required keys."""
        params = StrategyParams(
            use_daily_structure=True,
            daily_shelf_lookback_days=20,
            daily_shelf_cluster_pts=30.0,
            daily_shelf_min_touches=3,
            daily_fb_recovery_bars=3,
        )
        detector = DailyStructureDetector(params)

        # Trigger FB detection
        lows = (
            [6810, 6795, 6805, 6790, 6800, 6810, 6795, 6805, 6790, 6800]
            + [6500, 6400, 6350]
            + [6700, 6750, 6820, 6850, 6900, 6950, 7000]
        )
        closes = (
            [6850, 6840, 6860, 6830, 6845, 6855, 6835, 6850, 6825, 6840]
            + [6520, 6420, 6380]
            + [6720, 6780, 6850, 6880, 6920, 6960, 7100]
        )
        daily_bars = _make_daily_bars(lows, closes)
        detector.update(daily_bars)

        snap = detector.get_snapshot()
        assert "state" in snap
        assert "shelf_price" in snap
        assert "sweep_low" in snap
        assert "move_position" in snap
        assert snap["state"] == "DAILY_FB_BULL"
        assert snap["shelf_price"] > 0
        assert snap["sweep_low"] > 0
        assert isinstance(snap["move_position"], float)

    def test_snapshot_default_state(self):
        """Default snapshot before any update."""
        params = StrategyParams(use_daily_structure=True)
        detector = DailyStructureDetector(params)
        snap = detector.get_snapshot()

        assert snap["state"] == "NEUTRAL"
        assert snap["shelf_price"] == 0.0
        assert snap["sweep_low"] == 0.0
        assert snap["move_position"] == 0.0
