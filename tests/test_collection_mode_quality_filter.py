"""Tests for the collection_mode quality filter.

Before this change, `bypass_session_gates=True` caused the bot to take
ANY signal it qualified, as long as only a time-of-day gate was in the
way. That produced trade #15161 (LR @ 7607, mid-range, no plan match,
lost $425) and ~30-40% of the historical entry trades.

After this change, collection_mode requires ONE of:
  - high-quality structural level type (PDL, MHL, IL, CUSTOM)
  - level matches a Mancini plan setup within 2 pt

If neither, the trade is rejected even though the time gate would have
been bypassed.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from live.ib_runner import IBRunner


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeLevelType:
    name: str


@dataclass
class _FakeLevel:
    price: float
    level_type: _FakeLevelType


@dataclass
class _FakePattern:
    level: _FakeLevel


@dataclass
class _FakeSignal:
    pattern: _FakePattern
    entry_price: float = 0.0
    signal_type: object = None


@dataclass
class _FakeSetup:
    level_price: float
    direction: str = "long"
    setup_type: str = "failed_breakdown"
    conviction: str = "high"


@dataclass
class _FakePlan:
    planned_setups: list


def _runner_with_plan(setups: list[_FakeSetup] | None = None):
    runner = SimpleNamespace()
    runner._mancini_llm_plan = _FakePlan(planned_setups=setups or [])
    return runner


def _signal_at(level_price: float, level_type_name: str) -> _FakeSignal:
    return _FakeSignal(
        pattern=_FakePattern(
            level=_FakeLevel(
                price=level_price,
                level_type=_FakeLevelType(name=level_type_name),
            )
        ),
        entry_price=level_price + 1.0,
    )


# ---------------------------------------------------------------------------
# Path 1: high-quality structural level types
# ---------------------------------------------------------------------------


class TestHighQualityLevelTypes:
    @pytest.mark.parametrize("lvl_type", [
        "PRIOR_DAY_LOW",
        "PRIOR_DAY_HIGH",
        "MULTI_HOUR_LOW",
        "MULTI_HOUR_HIGH",
        "INTRADAY_LOW",
        "CUSTOM",
    ])
    def test_high_quality_types_accepted_even_without_plan(self, lvl_type):
        """No plan loaded, no match — but the level type alone is enough."""
        runner = _runner_with_plan([])
        sig = _signal_at(7530.0, lvl_type)
        assert IBRunner._collection_mode_is_quality_setup(runner, sig) is True

    @pytest.mark.parametrize("lvl_type", [
        "SWING_LOW",
        "SWING_HIGH",
        "CLUSTER_LOW",
        "CLUSTER_HIGH",
        "HORIZONTAL_SR",
        "VWAP",
    ])
    def test_low_quality_types_rejected_without_plan_match(self, lvl_type):
        runner = _runner_with_plan([])
        sig = _signal_at(7530.0, lvl_type)
        assert IBRunner._collection_mode_is_quality_setup(runner, sig) is False


# ---------------------------------------------------------------------------
# Path 2: Mancini plan match
# ---------------------------------------------------------------------------


class TestManciniPlanMatch:
    def test_exact_plan_match_accepts_swing_low(self):
        runner = _runner_with_plan([_FakeSetup(level_price=7530.0)])
        sig = _signal_at(7530.0, "SWING_LOW")
        assert IBRunner._collection_mode_is_quality_setup(runner, sig) is True

    def test_within_tolerance_accepts(self):
        """Default tolerance is 2 pt."""
        runner = _runner_with_plan([_FakeSetup(level_price=7530.0)])
        sig = _signal_at(7531.5, "SWING_LOW")
        assert IBRunner._collection_mode_is_quality_setup(runner, sig) is True

    def test_outside_tolerance_rejects(self):
        runner = _runner_with_plan([_FakeSetup(level_price=7530.0)])
        sig = _signal_at(7533.0, "SWING_LOW")  # 3pt off
        assert IBRunner._collection_mode_is_quality_setup(runner, sig) is False

    def test_no_plan_loaded_rejects_low_quality(self):
        runner = SimpleNamespace()
        runner._mancini_llm_plan = None
        sig = _signal_at(7530.0, "SWING_LOW")
        assert IBRunner._collection_mode_is_quality_setup(runner, sig) is False


# ---------------------------------------------------------------------------
# Production scenario: today's loser at LR 7607
# ---------------------------------------------------------------------------


class TestProductionScenario:
    def test_trade_15161_rejected(self):
        """The 15:20 ET LEVEL_RECLAIM at 7607.25 that lost $425 must now
        be rejected. Today's plan had LR longs at 7587 and 7604 (low
        conviction). Closest plan level to 7607.25 is 7604 — that's 3.25
        pt away, outside the 2pt tolerance. Level type was a swing-like
        engine-derived level, not a structural one."""
        runner = _runner_with_plan([
            _FakeSetup(level_price=7587.0, setup_type="level_reclaim"),
            _FakeSetup(level_price=7604.0, setup_type="level_reclaim"),
            _FakeSetup(level_price=7563.0, setup_type="failed_breakdown"),
            _FakeSetup(level_price=7538.0, setup_type="failed_breakdown"),
            _FakeSetup(level_price=7527.0, setup_type="failed_breakdown"),
            _FakeSetup(level_price=7517.0, setup_type="failed_breakdown"),
        ])
        sig = _signal_at(7607.25, "HORIZONTAL_SR")
        assert IBRunner._collection_mode_is_quality_setup(runner, sig) is False, (
            "trade #15161 LR @ 7607.25 must be rejected — no plan match "
            "(closest 7604 is 3.25pt off) and not a high-quality level type"
        )

    def test_overnight_fb_at_7550_close_enough(self):
        """Yesterday night's winning FB at 7550.50 entered after a sweep
        through ~7548-7550. Engine detected this as INTRADAY_LOW (a
        valid high-quality type)."""
        runner = _runner_with_plan([])  # no plan match needed
        sig = _signal_at(7548.0, "INTRADAY_LOW")
        assert IBRunner._collection_mode_is_quality_setup(runner, sig) is True

    def test_fb_at_planned_7517_takes_through_evening_block(self):
        """A FB long at 7517 (Mancini's HIGH-conviction Thursday FB level)
        firing at 9pm Globex evening block. Time gate would skip in
        production but the setup is high quality — collection mode
        SHOULD take it for the dataset."""
        runner = _runner_with_plan([
            _FakeSetup(level_price=7517.0, conviction="high"),
        ])
        # Level type CUSTOM (injected by plan loader) — both paths pass
        sig = _signal_at(7517.0, "CUSTOM")
        assert IBRunner._collection_mode_is_quality_setup(runner, sig) is True

    def test_missing_pattern_or_level_rejects(self):
        runner = _runner_with_plan([_FakeSetup(level_price=7530.0)])
        sig = _FakeSignal(pattern=None, entry_price=7530.0)
        assert IBRunner._collection_mode_is_quality_setup(runner, sig) is False


# ---------------------------------------------------------------------------
# Bypass entry sizing
# ---------------------------------------------------------------------------


class TestBypassEntrySizing:
    """Collection-mode bypass entries ride only on bypassed TIME gates —
    they must not also bypass position sizing.

    Trade #16229 (2026-06-05, -130 pts): signal had position_size_factor
    0.25 for a 31-pt stop, but the bypass path rebuilt the EntryDecision
    with default_contracts=4. Bypass entries are now risk-floored at 1
    contract — the collection-data value is identical at minimum size.
    """

    def test_bypass_entries_are_one_contract(self):
        runner = SimpleNamespace()
        sig = SimpleNamespace(entry_price=7474.0, stop_price=7443.0)
        entry = IBRunner._build_bypass_entry(
            runner, sig, ["In chop zone (1PM-3PM)", "In chop zone (11AM-2PM)"]
        )
        assert entry.should_enter is True
        assert entry.contracts == 1
        assert entry.entry_price == 7474.0
        assert entry.stop_price == 7443.0
        assert "chop zone" in entry.reason.lower()
