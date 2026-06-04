"""Tests for Mancini LLM plan level injection.

Without injection, the plan only *filters* (danger zones, no_trade_below)
and *boosts LQS* — it does NOT cause the engine to trigger a FB at
Mancini's specific levels unless the engine independently classified
the price as a swing/cluster/PDL low. That meant tonight's setup at
7538 was invisible.

After injection, each LONG planned_setup is pushed into the level store
as a CUSTOM level. The FB pattern detector's _HIGH_QUALITY_LEVELS
whitelist already includes CUSTOM, so these are fireable immediately.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest

from config.levels import LevelStore, LevelType
from live.ib_runner import IBRunner


_ET = timezone(timedelta(hours=-4))


@dataclass
class _FakeSetup:
    setup_type: str
    level_price: float
    direction: str
    context: str = ""
    conviction: str = "low"


@dataclass
class _FakePlan:
    lean: str = "bullish"
    mode: str = "range"
    planned_setups: list = None
    danger_zones: list = None
    no_trade_above: float = None
    no_trade_below: float = None

    def __post_init__(self):
        if self.planned_setups is None:
            self.planned_setups = []
        if self.danger_zones is None:
            self.danger_zones = []


def _make_runner_stub() -> SimpleNamespace:
    """Build the minimal object exposed to _inject_plan_levels."""
    store = LevelStore()
    agg = SimpleNamespace(level_store=store)
    runner = SimpleNamespace(
        signal_aggregator=agg,
        # _inject_plan_levels is bound; we'll call it directly
    )
    return runner, store


class TestPlanLevelInjection:
    def test_high_conviction_fb_long_gets_injected(self):
        runner, store = _make_runner_stub()
        plan = _FakePlan(planned_setups=[
            _FakeSetup("failed_breakdown", 7538.0, "long",
                       "FB of medium support", "medium"),
        ])
        n = IBRunner._inject_plan_levels(runner, plan)
        assert n == 1
        active = store.get_active(LevelType.CUSTOM)
        assert len(active) == 1
        assert active[0].price == 7538.0
        assert active[0].mancini_confirmed is True
        assert active[0].mancini_conviction == 2  # medium=2
        assert active[0].confirmed_at is not None  # immediately usable

    def test_level_reclaim_long_is_injected(self):
        runner, store = _make_runner_stub()
        plan = _FakePlan(planned_setups=[
            _FakeSetup("level_reclaim", 7565.0, "long", "LR support", "low"),
        ])
        n = IBRunner._inject_plan_levels(runner, plan)
        assert n == 1
        assert store.get_active(LevelType.CUSTOM)[0].price == 7565.0

    def test_short_setups_are_skipped(self):
        runner, store = _make_runner_stub()
        plan = _FakePlan(planned_setups=[
            _FakeSetup("breakdown_short", 7558.0, "short", "bear case", "low"),
            _FakeSetup("failed_breakdown", 7538.0, "long", "FB", "high"),
        ])
        n = IBRunner._inject_plan_levels(runner, plan)
        assert n == 1  # only the long FB
        assert store.get_active(LevelType.CUSTOM)[0].price == 7538.0

    def test_trend_continuation_is_skipped(self):
        """trend_continuation is a manual Mancini setup, not auto-tradeable
        by the engine's FB detector."""
        runner, store = _make_runner_stub()
        plan = _FakePlan(planned_setups=[
            _FakeSetup("trend_continuation", 7538.0, "long", "bid slow grind", "low"),
        ])
        n = IBRunner._inject_plan_levels(runner, plan)
        assert n == 0
        assert len(store.get_active(LevelType.CUSTOM)) == 0

    def test_conviction_mapping(self):
        runner, store = _make_runner_stub()
        plan = _FakePlan(planned_setups=[
            _FakeSetup("failed_breakdown", 7527.0, "long", "high", "high"),
            _FakeSetup("failed_breakdown", 7517.0, "long", "med", "medium"),
            _FakeSetup("failed_breakdown", 7485.0, "long", "low", "low"),
        ])
        IBRunner._inject_plan_levels(runner, plan)
        by_price = {l.price: l for l in store.get_active(LevelType.CUSTOM)}
        assert by_price[7527.0].mancini_conviction == 3
        assert by_price[7517.0].mancini_conviction == 2
        assert by_price[7485.0].mancini_conviction == 1

    def test_zero_price_is_skipped(self):
        runner, store = _make_runner_stub()
        plan = _FakePlan(planned_setups=[
            _FakeSetup("failed_breakdown", 0.0, "long", "bad data", "low"),
        ])
        n = IBRunner._inject_plan_levels(runner, plan)
        assert n == 0

    def test_empty_setups_returns_zero(self):
        runner, store = _make_runner_stub()
        n = IBRunner._inject_plan_levels(runner, _FakePlan(planned_setups=[]))
        assert n == 0

    def test_real_world_june_4_plan(self):
        """The actual Mancini plan for 2026-06-04 — verifies which levels
        the bot would now track."""
        runner, store = _make_runner_stub()
        plan = _FakePlan(planned_setups=[
            _FakeSetup("failed_breakdown", 7573.0, "long", "range support", "low"),
            _FakeSetup("failed_breakdown", 7563.0, "long", "daily low", "high"),
            _FakeSetup("failed_breakdown", 7538.0, "long", "bid only on slow grind", "medium"),
            _FakeSetup("failed_breakdown", 7527.0, "long", "Thursday 10am low", "high"),
            _FakeSetup("failed_breakdown", 7517.0, "long", "major multi-touch", "high"),
            _FakeSetup("level_reclaim", 7587.0, "long", "add-on for strength", "low"),
            _FakeSetup("breakdown_short", 7558.0, "short", "bear case", "low"),
            _FakeSetup("breakdown_short", 7604.0, "short", "resistance short", "low"),
        ])
        n = IBRunner._inject_plan_levels(runner, plan)
        # 5 FB longs + 1 LR long = 6
        assert n == 6
        active_prices = sorted(l.price for l in store.get_active(LevelType.CUSTOM))
        assert active_prices == [7517.0, 7527.0, 7538.0, 7563.0, 7573.0, 7587.0]
