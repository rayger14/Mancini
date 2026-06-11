"""Tests for 5-min bar aggregation and shelf-of-lows detection."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.bar_aggregator import BarAggregator
from core.price_levels import PriceLevelDetector
from tests.conftest import make_bars


# ---------------------------------------------------------------------------
# BarAggregator unit tests
# ---------------------------------------------------------------------------


class TestBarAggregator:
    """Unit tests for BarAggregator."""

    def _make_1min_bars(self, n_bars: int = 20) -> pd.DataFrame:
        """Create n 1-min bars."""
        prices = []
        p = 5800.0
        for _ in range(n_bars):
            o = p
            h = p + 1.0
            l = p - 0.5
            c = p + 0.5
            prices.append((o, h, l, c))
            p = c
        return make_bars(prices)

    def test_resample_basic(self):
        """Resample 10 1-min bars into 2 5-min bars."""
        df = self._make_1min_bars(10)
        agg = BarAggregator(period_minutes=5)
        result = agg.resample(df)
        assert len(result) == 2
        assert "open" in result.columns
        assert "high" in result.columns
        assert "volume" in result.columns

    def test_resample_preserves_ohlc(self):
        """OHLC aggregation is correct: first open, max high, min low, last close."""
        prices = [
            (100.0, 105.0, 98.0, 102.0),
            (102.0, 106.0, 101.0, 104.0),
            (104.0, 107.0, 99.0, 103.0),
            (103.0, 108.0, 100.0, 105.0),
            (105.0, 110.0, 104.0, 109.0),
        ]
        df = make_bars(prices)
        agg = BarAggregator(period_minutes=5)
        result = agg.resample(df)
        assert len(result) == 1
        assert result.iloc[0]["open"] == 100.0
        assert result.iloc[0]["high"] == 110.0
        assert result.iloc[0]["low"] == 98.0
        assert result.iloc[0]["close"] == 109.0
        assert result.iloc[0]["volume"] == 5000

    def test_resample_empty_df(self):
        """Empty input returns empty output."""
        agg = BarAggregator()
        result = agg.resample(pd.DataFrame())
        assert len(result) == 0

    def test_resample_none_input(self):
        """None input returns empty output."""
        agg = BarAggregator()
        result = agg.resample(None)
        assert len(result) == 0

    def test_update_incremental_drops_incomplete(self):
        """Incomplete trailing 5-min bar is dropped."""
        df = self._make_1min_bars(12)  # 2 complete + 2 leftover
        agg = BarAggregator(period_minutes=5)
        result = agg.update_incremental(df)
        assert len(result) == 2  # only 2 complete bars

    def test_update_incremental_exact_multiple(self):
        """When bar count is exact multiple of period, all bars returned."""
        df = self._make_1min_bars(15)  # 3 complete bars
        agg = BarAggregator(period_minutes=5)
        result = agg.update_incremental(df)
        assert len(result) == 3

    def test_update_incremental_too_few_bars(self):
        """Fewer than period bars returns empty."""
        df = self._make_1min_bars(3)
        agg = BarAggregator(period_minutes=5)
        result = agg.update_incremental(df)
        assert len(result) == 0

    def test_update_incremental_trimmed_window_excludes_forming_bar(self):
        """A rolling window trimmed mid-bucket must not leak the forming 5-min bar.

        Live, ib_runner keeps the last 400 1-min bars, so the DF rarely starts
        on a 5-min boundary. 10 bars from 09:32 → buckets 09:30 (partial),
        09:35 (complete), 09:40 (forming at 09:41). Only 09:35 is returnable.
        """
        prices = [(5800.0, 5801.0, 5799.5, 5800.5)] * 10
        df = make_bars(prices, start=datetime(2024, 1, 15, 9, 32))
        agg = BarAggregator(period_minutes=5)
        result = agg.update_incremental(df)
        assert len(result) == 1
        assert result.index[0] == pd.Timestamp(datetime(2024, 1, 15, 9, 35))

    def test_update_incremental_drops_partial_leading_bucket(self):
        """A leading bucket missing its first minutes is dropped, complete one kept.

        7 bars from 09:33 → bucket 09:30 has only 09:33-09:34 (wrong OHLC),
        bucket 09:35 is complete through 09:39.
        """
        prices = [(5800.0, 5801.0, 5799.5, 5800.5)] * 7
        df = make_bars(prices, start=datetime(2024, 1, 15, 9, 33))
        agg = BarAggregator(period_minutes=5)
        result = agg.update_incremental(df)
        assert len(result) == 1
        assert result.index[0] == pd.Timestamp(datetime(2024, 1, 15, 9, 35))

    def test_update_incremental_gap_does_not_leak_forming_bar(self):
        """Completeness is decided by timestamps, not row count, across data gaps.

        5 bars 09:30-09:34, then 5 bars 09:36-09:40 (gap at 09:35). Bucket
        09:30 is complete, bucket 09:35 ended (covered through 09:39), bucket
        09:40 is still forming at 09:40 and must be dropped — even though
        len(df) % 5 == 0.
        """
        prices = [(5800.0, 5801.0, 5799.5, 5800.5)] * 10
        first = make_bars(prices[:5], start=datetime(2024, 1, 15, 9, 30))
        second = make_bars(prices[5:], start=datetime(2024, 1, 15, 9, 36))
        df = pd.concat([first, second])
        agg = BarAggregator(period_minutes=5)
        result = agg.update_incremental(df)
        assert list(result.index) == [
            pd.Timestamp(datetime(2024, 1, 15, 9, 30)),
            pd.Timestamp(datetime(2024, 1, 15, 9, 35)),
        ]


# ---------------------------------------------------------------------------
# Catch-up look-ahead tests
# ---------------------------------------------------------------------------


class TestCatchUpNoLookahead:
    """During startup catch-up, update() receives the full session DF while
    replaying early bars. 5-min level detection must not see future buckets."""

    def test_update_with_future_bars_does_not_create_future_5min_levels(self):
        from core.signals import SignalAggregator

        # 120 flat 1-min bars at ~5800, except a swing low to 5750 at
        # minutes 85-89. That is 5-min bucket 17 — exactly order(6) buckets
        # behind the last completed bucket (23), so a full-df call confirms
        # it as a swing low even while replaying bar 10 (09:40).
        prices = []
        for i in range(120):
            if 85 <= i <= 89:
                prices.append((5752.0, 5753.0, 5750.0, 5752.0))
            else:
                prices.append((5800.0, 5801.0, 5799.5, 5800.5))
        df = make_bars(prices, start=datetime(2024, 1, 15, 9, 30))

        params = StrategyParams(use_5min_levels=True)
        agg = SignalAggregator(strategy_params=params)

        # Catch-up style call: replaying bar 10 (09:40) with the FULL df
        agg.update(
            bar_idx=10,
            timestamp=df.index[10],
            open_=float(df["open"].iat[10]),
            high=float(df["high"].iat[10]),
            low=float(df["low"].iat[10]),
            close=float(df["close"].iat[10]),
            volume=1000.0,
            velocity=0.0,
            df=df,
        )

        future_levels = [
            l for l in agg.level_store.levels if abs(l.price - 5750.0) < 2.0
        ]
        assert future_levels == [], (
            f"5-min swing low from the future (10:30 dip) leaked into the "
            f"level store while replaying 09:40: {future_levels}"
        )


# ---------------------------------------------------------------------------
# Shelf-of-lows detection tests
# ---------------------------------------------------------------------------


class TestShelfDetection:
    """Tests for PriceLevelDetector._detect_shelf_levels."""

    def _make_shelf_bars_5min(self, n_bars: int = 24) -> pd.DataFrame:
        """Create 5-min bars with a shelf of lows around 5800.

        Multiple bars touch ~5800 low (within 3 pts), creating a shelf.
        """
        start = datetime(2024, 1, 15, 9, 30)
        prices = []
        for i in range(n_bars):
            # Every other bar touches the shelf zone (5798-5801)
            if i % 2 == 0:
                o = 5810.0
                h = 5815.0
                l = 5799.0 + (i % 3) * 0.5  # 5799.0, 5799.5, 5800.0
                c = 5808.0
            else:
                o = 5808.0
                h = 5812.0
                l = 5805.0  # doesn't touch shelf
                c = 5810.0
            prices.append((o, h, l, c))
        return make_bars(prices, start=start, freq="5min")

    def test_shelf_detected(self):
        """Shelf of lows is detected when enough bars touch the zone."""
        params = StrategyParams(
            detect_shelf_levels=True,
            shelf_min_touches=4,
            shelf_proximity_pts=3.0,
            shelf_min_bars=12,
        )
        detector = PriceLevelDetector(params)
        store = LevelStore()
        df = self._make_shelf_bars_5min(24)

        levels = detector._detect_shelf_levels(store, df, len(df) - 1)
        assert len(levels) >= 1
        shelf = levels[0]
        assert shelf.level_type == LevelType.HORIZONTAL_SR
        assert shelf.touch_count >= 4
        assert 5798.0 <= shelf.price <= 5801.0

    def test_shelf_not_detected_too_few_bars(self):
        """Shelf not detected when bar_idx < shelf_min_bars."""
        params = StrategyParams(
            detect_shelf_levels=True,
            shelf_min_bars=12,
        )
        detector = PriceLevelDetector(params)
        store = LevelStore()
        df = self._make_shelf_bars_5min(24)

        levels = detector._detect_shelf_levels(store, df, 5)  # too few bars
        assert len(levels) == 0

    def test_shelf_not_detected_too_few_touches(self):
        """No shelf when touches < shelf_min_touches."""
        start = datetime(2024, 1, 15, 9, 30)
        prices = []
        for i in range(20):
            # Only 2 bars touch the zone — not enough
            if i < 2:
                prices.append((5810.0, 5815.0, 5800.0, 5808.0))
            else:
                prices.append((5810.0, 5815.0, 5808.0, 5812.0))
        df = make_bars(prices, start=start, freq="5min")

        params = StrategyParams(
            detect_shelf_levels=True,
            shelf_min_touches=4,
            shelf_proximity_pts=3.0,
            shelf_min_bars=12,
        )
        detector = PriceLevelDetector(params)
        store = LevelStore()

        levels = detector._detect_shelf_levels(store, df, len(df) - 1)
        assert len(levels) == 0

    def test_shelf_not_duplicated_with_existing_level(self):
        """Shelf not added when existing level is within 1 pt."""
        params = StrategyParams(
            detect_shelf_levels=True,
            shelf_min_touches=4,
            shelf_proximity_pts=3.0,
            shelf_min_bars=12,
        )
        detector = PriceLevelDetector(params)
        store = LevelStore()
        # Pre-add a level at 5799.5
        store.add(Level(
            price=5799.5,
            level_type=LevelType.SWING_LOW,
            created_at=datetime(2024, 1, 14, 10, 0),
            confirmed_at=datetime(2024, 1, 14, 10, 0),
        ))
        df = self._make_shelf_bars_5min(24)

        levels = detector._detect_shelf_levels(store, df, len(df) - 1)
        assert len(levels) == 0  # suppressed by existing level


# ---------------------------------------------------------------------------
# 5-min swing detection integration tests
# ---------------------------------------------------------------------------


class TestFiveMinSwingDetection:
    """Tests for swing detection on 5-min data via detect_incremental."""

    def test_5min_swing_low_detected(self):
        """Swing low detected on 5-min data when use_5min_levels=True."""
        params = StrategyParams(
            use_5min_levels=True,
            swing_low_order_5min=3,  # small order for test
        )
        detector = PriceLevelDetector(params)
        store = LevelStore()

        # Create 5-min bars with a clear swing low at bar 5
        # Down to bar 5, then rally
        prices_5min = []
        for i in range(15):
            if i < 5:
                # Selling
                p = 5900.0 - i * 10
                prices_5min.append((p, p + 2, p - 3, p - 2))
            elif i == 5:
                # The low
                prices_5min.append((5850.0, 5852.0, 5840.0, 5845.0))
            else:
                # Rally
                p = 5845.0 + (i - 5) * 8
                prices_5min.append((p, p + 5, p - 1, p + 3))
        df_5min = make_bars(prices_5min, freq="5min")

        # Create matching 1-min bars (75 bars)
        prices_1min = []
        p = 5900.0
        for i in range(75):
            if i < 25:
                p = 5900.0 - i * 2
            elif i == 25:
                p = 5840.0
            else:
                p = 5840.0 + (i - 25) * 1.5
            prices_1min.append((p, p + 1, p - 0.5, p + 0.5))
        df_1min = make_bars(prices_1min)

        # Run at bar_idx where swing should be confirmed (bar 5 + order 3 = bar 8 on 5min)
        bar_idx_5min = 8
        bar_idx_1min = 44  # corresponding 1-min bar

        levels = detector._detect_swing_lows_on_df(
            store, df_5min, bar_idx_5min,
            order=3,
            current_close_1min=float(df_1min["close"].values[bar_idx_1min]),
        )
        assert len(levels) >= 1

    def test_5min_disabled_uses_1min(self):
        """When use_5min_levels=False, detect_incremental uses 1-min data."""
        params = StrategyParams(use_5min_levels=False)
        detector = PriceLevelDetector(params)
        store = LevelStore()

        # Create enough 1-min bars for swing detection (order 30 * 2 = 60)
        prices = []
        for i in range(80):
            if i < 30:
                p = 5900.0 - i * 2
            elif i == 30:
                p = 5830.0
            else:
                p = 5830.0 + (i - 30) * 1.5
            prices.append((p, p + 1, p - 0.5, p + 0.5))
        df = make_bars(prices)

        # Call detect_incremental without 5-min data (use_5min_levels=False)
        new_levels = detector.detect_incremental(store, df, 75)
        # Should work fine — no 5-min data needed
        assert isinstance(new_levels, list)


# ---------------------------------------------------------------------------
# Micro sweep threshold tests
# ---------------------------------------------------------------------------


class TestMicroSweepThreshold:
    """Tests that shelf levels with high touch counts allow smaller sweeps."""

    def test_shelf_level_allows_micro_sweep(self):
        """Level with touch_count >= shelf_min_touches uses shelf_sweep_min_pts."""
        from core.patterns import FailedBreakdown, PatternState

        params = StrategyParams(
            sweep_min_ticks=8,  # standard = 2.0 pts
            shelf_min_touches=4,
            shelf_sweep_min_pts=1.0,
        )
        fb = FailedBreakdown(params)

        store = LevelStore()
        shelf_level = Level(
            price=5800.0,
            level_type=LevelType.CLUSTER_LOW,
            created_at=datetime(2024, 1, 15, 9, 0),
            confirmed_at=datetime(2024, 1, 15, 9, 0),
            touch_count=5,  # >= shelf_min_touches
        )
        store.add(shelf_level)

        # Low of 5799.0 = 1 pt below level. Standard sweep (2 pts) would reject.
        # But shelf allows 1 pt min.
        fb._scan_for_sweep(
            low=5799.0,
            close=5799.5,
            level_store=store,
            timestamp=datetime(2024, 1, 15, 10, 0),
            bar_idx=100,
        )
        assert fb.state == PatternState.SWEEP_DETECTED

    def test_normal_level_requires_standard_sweep(self):
        """Level with low touch count uses standard sweep_min_ticks."""
        from core.patterns import FailedBreakdown, PatternState

        params = StrategyParams(
            sweep_min_ticks=8,  # standard = 2.0 pts
            shelf_min_touches=4,
            shelf_sweep_min_pts=1.0,
        )
        fb = FailedBreakdown(params)

        store = LevelStore()
        normal_level = Level(
            price=5800.0,
            level_type=LevelType.CLUSTER_LOW,
            created_at=datetime(2024, 1, 15, 9, 0),
            confirmed_at=datetime(2024, 1, 15, 9, 0),
            touch_count=2,  # < shelf_min_touches
        )
        store.add(normal_level)

        # Low of 5799.0 = only 1 pt below. Standard requires 2 pts.
        fb._scan_for_sweep(
            low=5799.0,
            close=5799.5,
            level_store=store,
            timestamp=datetime(2024, 1, 15, 10, 0),
            bar_idx=100,
        )
        assert fb.state != PatternState.SWEEP_DETECTED


# ---------------------------------------------------------------------------
# Config flag backward compatibility
# ---------------------------------------------------------------------------


class TestConfigBackwardCompat:
    """Verify that use_5min_levels=False preserves exact existing behavior."""

    def test_defaults_off(self):
        """All 5-min features default to off."""
        params = StrategyParams()
        assert params.use_5min_levels is False
        assert params.detect_shelf_levels is False

    def test_production_config_has_5min_params(self):
        """Production runs 5-min level + shelf detection with tuned params.

        Enabled 2026-06-11 after sign-off (fully-gated backtest: +638 pts,
        PF +0.20, WR +4.0% on 315 plan sessions). The live resample path
        was fixed end-to-end in PR #35.
        """
        from live.ib_runner import PRODUCTION_STRATEGY
        assert PRODUCTION_STRATEGY.use_5min_levels is True
        assert PRODUCTION_STRATEGY.swing_low_order_5min == 6
        assert PRODUCTION_STRATEGY.detect_shelf_levels is True
        assert PRODUCTION_STRATEGY.shelf_min_touches == 8
        assert PRODUCTION_STRATEGY.shelf_sweep_min_pts == 2.0
