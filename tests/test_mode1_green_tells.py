"""Tests for the Mode 1 Green detector's data-backed tells.

5-year study (2026-06-10, 177 labeled trend-up days of 1292 sessions):
- shallow-fast dips (<=8 pts, recovered <=20 bars): green median 6/day vs 1
- breakdown squeezes (break below 60-bar low, reclaim, new high): 17% of
  green days vs 7% of normal days
- consecutive bars above PDH: green median 180 vs 1

Also: resistance levels (swing highs etc.) were only detected when a short
pattern was enabled, starving the green detector's resistance tell in
longs-only configs.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from config.levels import LevelStore, LevelType
from config.settings import StrategyParams
from core.mode1_green_detector import Mode1GreenDetector
from core.price_levels import PriceLevelDetector


_T0 = datetime(2024, 6, 17, 9, 30)


def _params(**overrides) -> StrategyParams:
    base = dict(
        use_mode1_green_detection=True,
        mode1_green_bullish_pressure_bars=500,  # neutralize pressure tell
        mode1_green_bars_above_pdh=500,         # neutralize PDH tell
    )
    base.update(overrides)
    return StrategyParams(**base)


def _feed(det: Mode1GreenDetector, bars: list[tuple[float, float, float]]):
    """bars: list of (high, low, close)."""
    store = LevelStore()
    for i, (h, l, c) in enumerate(bars):
        det.update(
            bar_idx=i,
            close=c,
            high=h,
            low=l,
            level_store=store,
            timestamp=_T0 + timedelta(minutes=i),
        )
    return det.state


def _dip_episode(high: float, depth: float, dip_bars: int):
    """Rise to `high`, dip `depth` pts for `dip_bars` bars, recover."""
    bars = [(high, high - 1.0, high - 0.25)]
    for _ in range(dip_bars):
        bars.append((high - depth + 1.0, high - depth, high - depth + 0.5))
    bars.append((high + 0.25, high - 1.0, high))  # reclaim the high
    return bars


class TestShallowDipTell:
    def test_shallow_fast_dips_counted(self):
        det = Mode1GreenDetector(_params())
        bars = []
        h = 5800.0
        for i in range(5):
            h += 5.0
            bars.extend(_dip_episode(h, depth=5.0, dip_bars=6))
        state = _feed(det, bars)
        assert state.shallow_fast_dips == 5
        assert state.condition_shallow_dips is True  # tuned default min is 5

    def test_deep_dip_not_counted(self):
        det = Mode1GreenDetector(_params())
        state = _feed(det, _dip_episode(5800.0, depth=15.0, dip_bars=6))
        assert state.shallow_fast_dips == 0

    def test_slow_recovery_not_counted(self):
        det = Mode1GreenDetector(_params())
        state = _feed(det, _dip_episode(5800.0, depth=5.0, dip_bars=30))
        assert state.shallow_fast_dips == 0


class TestSqueezeTell:
    def _squeeze_session(self, follow_through: bool = True):
        # 61 flat bars to fill the rolling-low window
        bars = [(5801.0, 5800.0, 5800.5)] * 61
        # breakdown: 3 pts below the 60-bar low
        bars.append((5799.0, 5797.0, 5797.5))
        # reclaim within a few bars
        bars.append((5799.5, 5798.0, 5799.0))
        bars.append((5800.75, 5799.5, 5800.5))
        if follow_through:
            # new session high after the reclaim
            bars.append((5803.0, 5800.5, 5802.5))
        return bars

    def test_breakdown_squeeze_counted(self):
        det = Mode1GreenDetector(_params())
        state = _feed(det, self._squeeze_session(follow_through=True))
        assert state.squeezes == 1
        assert state.condition_squeeze is True

    def test_no_follow_through_no_squeeze(self):
        det = Mode1GreenDetector(_params())
        state = _feed(det, self._squeeze_session(follow_through=False))
        assert state.squeezes == 0


class TestTwoOfFiveConfirm:
    def test_new_tells_alone_confirm_green(self):
        """4 shallow dips + 1 squeeze = 2 conditions = MODE 1 GREEN,
        with PDH/pressure/resistances all neutralized."""
        det = Mode1GreenDetector(_params())
        bars = [(5801.0, 5800.0, 5800.5)] * 61
        bars.append((5799.0, 5797.0, 5797.5))
        bars.append((5800.75, 5799.5, 5800.5))
        bars.append((5803.0, 5800.5, 5802.5))   # squeeze complete
        h = 5803.0
        for _ in range(4):
            h += 5.0
            bars.extend(_dip_episode(h, depth=5.0, dip_bars=6))
        state = _feed(det, bars)
        assert state.squeezes >= 1
        assert state.shallow_fast_dips >= 4
        assert state.conditions_met >= 2
        assert state.is_mode1_green is True


class TestPressureSemantics:
    def test_pressure_counts_only_new_high_bars(self):
        """The docstring says 'higher highs for 60+ bars' but the counter
        also incremented on bars that made NO new high (any bar within 60
        bars of the last high counted). That made the condition nearly
        free — it fired on 71% of all sessions in the 5y replay."""
        params = _params(mode1_green_bullish_pressure_bars=60)
        det = Mode1GreenDetector(params)
        # 70 bars: only the first 10 make new highs, then flat drift
        bars = []
        h = 5800.0
        for i in range(10):
            h += 1.0
            bars.append((h, h - 1.0, h - 0.25))
        bars.extend([(h - 2.0, h - 3.0, h - 2.5)] * 60)
        state = _feed(det, bars)
        assert state.bullish_pressure_bars == 10
        assert state.condition_pressure is False


class TestResistanceStarvationFix:
    def test_swing_highs_detected_with_green_on_shorts_off(self):
        params = StrategyParams(
            use_mode1_green_detection=True,
            allow_short_fr=False,
            allow_short_lj=False,
            allow_breakdown_short=False,
            allow_backtest_short=False,
            swing_low_order=3,
        )
        detector = PriceLevelDetector(params)
        store = LevelStore()
        # clear swing high at bar 5: rise into 5820, fall away after
        prices = [5800, 5805, 5810, 5815, 5818, 5820, 5818, 5815, 5810, 5805,
                  5800, 5798, 5796, 5795, 5794]
        idx = pd.date_range("2024-06-17 09:30", periods=len(prices),
                            freq="1min", tz="US/Eastern")
        df = pd.DataFrame({
            "open": prices, "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices], "close": prices,
            "volume": [1000] * len(prices),
        }, index=idx)
        for i in range(len(df)):
            detector.detect_incremental(store, df, i)
        swing_highs = [l for l in store.levels
                       if l.level_type == LevelType.SWING_HIGH]
        assert swing_highs, (
            "green detection enabled must produce resistance levels even "
            "with every short pattern disabled"
        )
