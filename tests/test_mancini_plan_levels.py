"""Tests for the shared Mancini plan-level builder.

This is the one source of truth both live (_inject_plan_levels) and the backtest
use to inject Mancini's called levels — closing the #1 backtest-fidelity gap.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from config.levels import LevelType
from core.mancini_plan_levels import build_plan_levels

_NOW = datetime(2026, 6, 30, 9, 30)


def _setup(**k):
    return SimpleNamespace(**k)


def _plan(setups):
    return SimpleNamespace(planned_setups=setups)


def test_builds_long_fb_lr_as_custom_levels():
    plan = _plan([
        _setup(direction="long", setup_type="failed_breakdown",
               level_price=7423.0, conviction="high", context="ATM"),
        _setup(direction="long", setup_type="level_reclaim",
               level_price=7453.0, conviction="low", context="reclaim"),
    ])
    lvls = build_plan_levels(plan, _NOW)
    assert len(lvls) == 2
    assert all(l.level_type == LevelType.CUSTOM and l.mancini_confirmed for l in lvls)
    assert {round(l.price) for l in lvls} == {7423, 7453}
    fb = next(l for l in lvls if "failed_breakdown" in l.label)
    assert fb.mancini_conviction == 3        # high


def test_skips_shorts_and_other_types():
    plan = _plan([
        _setup(direction="short", setup_type="breakdown_short",
               level_price=7369.0, conviction="low"),
        _setup(direction="long", setup_type="other",
               level_price=7530.0, conviction="low"),
        _setup(direction="long", setup_type="failed_breakdown",
               level_price=7409.0, conviction="high"),
    ])
    lvls = build_plan_levels(plan, _NOW)
    assert len(lvls) == 1 and round(lvls[0].price) == 7409


def test_skips_nonpositive_price_and_empty_plan():
    assert build_plan_levels(_plan([]), _NOW) == []
    bad = _plan([_setup(direction="long", setup_type="failed_breakdown",
                        level_price=0.0, conviction="high")])
    assert build_plan_levels(bad, _NOW) == []
