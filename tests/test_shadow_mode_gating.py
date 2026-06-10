"""Tests for shadow_mode_features gating of VBD and Back-Test shorts.

The IB production config documents allow_velocity_short / allow_backtest_short
as "Shadow: log, don't trade", but the v2 short pipeline returned both as
live signals (the 2026-06-09 backtest_short traded real money and lost 47
pts). With shadow_mode_features=True these patterns must log a shadow event
(with entry/stop/target so phantom outcome tracking works) and return None.
Backtest behavior (shadow_mode_features=False) is unchanged.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.patterns import ConfirmationType, PatternSignal
from core.signals import SignalAggregator, SignalType


_LEVEL_TS = datetime(2024, 6, 15, 9, 30)


def _make_level(price: float, level_type=LevelType.CLUSTER_LOW) -> Level:
    return Level(
        price=price,
        level_type=level_type,
        created_at=_LEVEL_TS,
        touch_count=5,
    )


def _make_short_pattern(pattern_type: str) -> PatternSignal:
    return PatternSignal(
        pattern_type=pattern_type,
        confirmation=ConfirmationType.ACCEPTANCE,
        level=_make_level(5800.0),
        sweep_low=5790.0,
        entry_price=5799.0,
        stop_price=5805.0,
        bar_idx=100,
        timestamp=datetime(2024, 6, 15, 10, 30),
        sweep_depth_pts=5.0,
        direction="short",
        sweep_high=5805.0,
    )


def _make_agg(shadow: bool, **flags) -> SignalAggregator:
    params = StrategyParams(
        shadow_mode_features=shadow,
        block_pdl_shorts=False,
        block_capitulation_shorts=False,
        min_session_range_pts=0.0,
        **flags,
    )
    agg = SignalAggregator(strategy_params=params)
    agg.level_store = LevelStore()
    # Support targets below the ~5799 short entry
    agg.level_store.add(_make_level(5785.0))
    agg.level_store.add(_make_level(5770.0))
    return agg


def _run_one_bar(agg: SignalAggregator):
    return agg.update(
        bar_idx=100,
        timestamp=datetime(2024, 6, 15, 10, 30),
        open_=5800.0,
        high=5801.0,
        low=5798.0,
        close=5799.0,
        volume=5000.0,
        velocity=-3.0,
    )


class TestVelocityShortShadowGating:
    def test_shadow_mode_logs_but_does_not_trade(self):
        agg = _make_agg(shadow=True, allow_velocity_short=True)
        agg.velocity_breakdown.update = lambda **kw: _make_short_pattern("velocity_short")

        signal = _run_one_bar(agg)

        assert signal is None, "shadow mode must not return a tradeable signal"
        events = [e for e in agg.shadow_events if e["feature"] == "velocity_short"]
        assert len(events) == 1
        assert events[0]["entry_price"] == 5799.0
        assert events[0]["stop_price"] == 5805.0

    def test_live_when_shadow_off(self):
        agg = _make_agg(shadow=False, allow_velocity_short=True)
        agg.velocity_breakdown.update = lambda **kw: _make_short_pattern("velocity_short")

        signal = _run_one_bar(agg)

        assert signal is not None
        assert signal.signal_type == SignalType.VELOCITY_SHORT


class TestBacktestShortShadowGating:
    def test_shadow_mode_logs_but_does_not_trade(self):
        agg = _make_agg(shadow=True, allow_backtest_short=True)
        agg.backtest_short.update = lambda **kw: _make_short_pattern("backtest_short")

        signal = _run_one_bar(agg)

        assert signal is None, "shadow mode must not return a tradeable signal"
        events = [e for e in agg.shadow_events if e["feature"] == "backtest_short"]
        assert len(events) == 1
        # entry/stop/target required for phantom outcome tracking
        assert events[0]["entry_price"] == 5799.0
        assert events[0]["stop_price"] == 5805.0
        # target derives from the 5785 support (plus qualifier buffer)
        assert 5770.0 <= events[0]["target_1"] < 5799.0

    def test_live_when_shadow_off(self):
        agg = _make_agg(shadow=False, allow_backtest_short=True)
        agg.backtest_short.update = lambda **kw: _make_short_pattern("backtest_short")

        signal = _run_one_bar(agg)

        assert signal is not None
        assert signal.signal_type == SignalType.BACKTEST_SHORT
