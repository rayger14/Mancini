"""Build Mancini LLM-plan levels — the ONE source of truth shared by the live
engine and the backtest, so they can't drift.

Live injects Mancini's called LONG setups into the level store as CUSTOM levels
that the FB detector can fire on (`live/ib_runner.py::_inject_plan_levels`). For
the backtest to faithfully reproduce live, it must inject the SAME levels — so
both call this builder instead of each constructing levels independently.
"""
from __future__ import annotations

from datetime import datetime

from config.levels import Level, LevelType

_CONV_SCORE = {"high": 3, "medium": 2, "low": 1}
_ACCEPT_TYPES = {"failed_breakdown", "level_reclaim"}


def build_plan_levels(plan, now: datetime) -> list:
    """Return the LONG failed_breakdown / level_reclaim setups from a Mancini
    LLM plan as CUSTOM levels (mancini_confirmed). Short and trend-continuation
    setups are skipped — they run via their own detectors. Mirrors the level
    construction in live `_inject_plan_levels`."""
    out: list = []
    for setup in (getattr(plan, "planned_setups", None) or []):
        if (getattr(setup, "direction", "") or "").lower() != "long":
            continue
        stype = (getattr(setup, "setup_type", "") or "").lower()
        if stype not in _ACCEPT_TYPES:
            continue
        price = float(getattr(setup, "level_price", 0.0) or 0.0)
        if price <= 0:
            continue
        conv = (getattr(setup, "conviction", "") or "").lower()
        out.append(Level(
            price=price,
            level_type=LevelType.CUSTOM,
            created_at=now,
            confirmed_at=now,
            touch_count=1,
            label=f"MANCINI_PLAN:{setup.setup_type}@{price:.2f}",
            mancini_confirmed=True,
            mancini_side="support",
            mancini_conviction=_CONV_SCORE.get(conv, 1),
            mancini_tags=["llm_plan", f"conv:{conv}", f"type:{setup.setup_type}"],
        ))
    return out
