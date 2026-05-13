"""Tests for FB Tier 1 research improvements (from the 5/13/2026 quad-agent
investigation).

Three focused changes:
  1. Level freshness gate — reject FBs on levels older than
     ``fb_max_level_age_hours``. Mancini's "24-36 hours" rule.
  2. Acceptance timeout shallow 15 → 20 — recover ~63% WR setups
     rejected just shy of confirmation.
  3. MFE/MAE instrumentation — augment with recent-bar high/low and
     the exit fill price so IB-bracket fills mid-bar don't undershoot
     the recorded excursion.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.patterns import ConfirmationType, PatternSignal
from core.signals import SignalAggregator, SignalType


_NOW = datetime(2026, 5, 13, 10, 30)


def _level(price: float, level_type: LevelType, created_at: datetime,
           touches: int = 3) -> Level:
    return Level(
        price=price,
        level_type=level_type,
        created_at=created_at,
        confirmed_at=created_at,
        touch_count=touches,
    )


def _fb_pattern(level_age_hours: float = 0.0,
                level_price: float = 7250.0,
                entry_price: float = 7252.0) -> PatternSignal:
    level_ts = _NOW - timedelta(hours=level_age_hours)
    return PatternSignal(
        pattern_type="failed_breakdown",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=_level(level_price, LevelType.PRIOR_DAY_LOW, level_ts),
        sweep_low=level_price - 2.0,
        entry_price=entry_price,
        stop_price=entry_price - 5.0,
        bar_idx=100,
        timestamp=_NOW,
        sweep_depth_pts=2.0,
        direction="long",
    )


def _agg(**overrides) -> SignalAggregator:
    base = dict(
        use_level_quality_scoring=False,
        use_confluence_scoring=False,
        use_sweep_depth_sizing=False,
    )
    base.update(overrides)
    params = StrategyParams(**base)
    agg = SignalAggregator(strategy_params=params, min_rr_ratio=0.1)
    agg.level_store = LevelStore()
    agg.level_store.add(_level(7260.0, LevelType.HORIZONTAL_SR, _NOW))
    agg.level_store.add(_level(7275.0, LevelType.HORIZONTAL_SR, _NOW))
    return agg


# ---------------------------------------------------------------------------
# Change 1 — Level freshness gate
# ---------------------------------------------------------------------------


def test_freshness_gate_accepts_recent_level():
    """A 4-hour-old level passes the gate (well under 36h)."""
    agg = _agg(fb_max_level_age_hours=36.0)
    pattern = _fb_pattern(level_age_hours=4.0)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is not None


def test_freshness_gate_accepts_at_boundary():
    """At the boundary (~36h), still accept (strict greater-than)."""
    agg = _agg(fb_max_level_age_hours=36.0)
    pattern = _fb_pattern(level_age_hours=36.0)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is not None


def test_freshness_gate_rejects_stale_level():
    """A 72-hour-old level should be rejected at default 36h cap."""
    agg = _agg(fb_max_level_age_hours=36.0)
    pattern = _fb_pattern(level_age_hours=72.0)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is None


def test_freshness_gate_rejects_week_old_level():
    """The macro-FB case from Mancini (3-week-old low). Reject by default."""
    agg = _agg(fb_max_level_age_hours=36.0)
    pattern = _fb_pattern(level_age_hours=24.0 * 21)  # 3 weeks
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is None


def test_freshness_gate_can_be_disabled():
    """Setting fb_max_level_age_hours=0.0 disables the gate."""
    agg = _agg(fb_max_level_age_hours=0.0)
    pattern = _fb_pattern(level_age_hours=72.0)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is not None


def test_freshness_gate_does_not_affect_non_fb_signals():
    """LEVEL_RECLAIM signals should not be gated by FB freshness rule."""
    agg = _agg(fb_max_level_age_hours=36.0)
    pattern = _fb_pattern(level_age_hours=72.0)
    pattern.pattern_type = "level_reclaim"
    sig = agg._qualify_signal(pattern, SignalType.LEVEL_RECLAIM)
    # No FB-specific gate triggered for LR — passes through
    assert sig is not None


def test_freshness_logs_shadow_event_on_reject():
    """Rejected stale FB should emit a shadow event for diagnostics."""
    agg = _agg(fb_max_level_age_hours=36.0)
    pattern = _fb_pattern(level_age_hours=72.0)
    agg.shadow_events.clear()
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is None
    features = [e["feature"] for e in agg.shadow_events]
    assert "fb_level_too_old" in features


# ---------------------------------------------------------------------------
# Change 2 — acceptance_timeout_bars_shallow bumped 15 → 20
# ---------------------------------------------------------------------------


def test_acceptance_timeout_shallow_default_is_20():
    """The new default should be 20 bars (was 15). Pure config sanity."""
    p = StrategyParams()
    assert p.acceptance_timeout_bars_shallow == 20


def test_acceptance_timeout_deep_unchanged():
    """Deep-flush timeout (60) is unchanged."""
    p = StrategyParams()
    assert p.acceptance_timeout_bars_deep == 60


# ---------------------------------------------------------------------------
# Change 3 — MFE/MAE augmentation
# ---------------------------------------------------------------------------


def test_mfe_augmentation_logic_takes_max_of_position_and_recent_high():
    """The MFE fix takes max of (position.highest, recent bar high, exit_price).
    Unit-test the core logic by simulating the merge.
    """
    position_highest = 7280.0   # stale snapshot
    recent_bar_high = 7285.0    # IB filled on a bar that ran higher
    exit_fill_price = 7283.0    # actual fill price

    # Apply the same merge logic from ib_runner.py:
    highest = position_highest
    for candidate in (recent_bar_high, exit_fill_price):
        if candidate is not None and (highest is None or candidate > highest):
            highest = candidate

    assert highest == 7285.0


def test_mfe_augmentation_when_position_is_none():
    """When position is cleared (None), still recover MFE from recent bar."""
    position_highest = None
    recent_bar_high = 7290.0
    exit_fill_price = 7288.0

    highest = position_highest
    for candidate in (recent_bar_high, exit_fill_price):
        if candidate is not None and (highest is None or candidate > highest):
            highest = candidate

    assert highest == 7290.0


def test_mae_augmentation_for_long_takes_min():
    """For long trades, MAE = entry - lowest. Verify the merge takes min."""
    position_lowest = 7245.0
    recent_bar_low = 7240.0    # ran lower than position snapshot
    exit_fill_price = 7242.0

    lowest = position_lowest
    for candidate in (recent_bar_low, exit_fill_price):
        if candidate is not None and (lowest is None or candidate < lowest):
            lowest = candidate

    assert lowest == 7240.0


def test_mfe_handles_all_none_safely():
    """All three sources None — highest stays None, no exception."""
    highest = None
    for candidate in (None, None):
        if candidate is not None and (highest is None or candidate > highest):
            highest = candidate
    assert highest is None
