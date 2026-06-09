"""Tests for the type-based FB level-age-cap exemption.

Mancini's "24-36 hour" freshness rule rejects FB longs whose underlying
swept low is older than ``fb_max_level_age_hours``. That cap makes sense
for engine-derived intraday clusters (CLUSTER_LOW, SWING_LOW) which
quickly go stale. It does NOT match Mancini's own behaviour on
structural multi-day shelves: he routinely holds runners off
PRIOR_DAY_LOW / MULTI_HOUR_LOW / INTRADAY_LOW shelves for days, and
levels he names in his own plan (loaded as ``CUSTOM``) can be valid for
an entire week (e.g. the 7517 shelf he flagged "since last Tuesday").

When ``StrategyParams.fb_age_cap_exempt_high_quality_levels`` is True,
the age check is bypassed for those four level types. All other types
(SWING_LOW, CLUSTER_LOW, HORIZONTAL_SR, ...) remain subject to the cap.
The flag is False by default — these tests both pin the default
behaviour and prove the opt-in works.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.patterns import ConfirmationType, PatternSignal
from core.signals import SignalAggregator, SignalType


_NOW = datetime(2026, 6, 8, 10, 30)


def _level(
    price: float,
    level_type: LevelType,
    created_at: datetime,
    touches: int = 3,
) -> Level:
    return Level(
        price=price,
        level_type=level_type,
        created_at=created_at,
        confirmed_at=created_at,
        touch_count=touches,
    )


def _fb_pattern(
    level_age_hours: float,
    level_type: LevelType,
    level_price: float = 7517.0,
    entry_price: float = 7519.0,
) -> PatternSignal:
    level_ts = _NOW - timedelta(hours=level_age_hours)
    return PatternSignal(
        pattern_type="failed_breakdown",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=_level(level_price, level_type, level_ts),
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
        fb_max_level_age_hours=36.0,
        # Disable the macro-VIX carve-out so the only thing that can
        # bypass the freshness gate is the type-based exemption itself.
        fb_macro_vix_threshold=0.0,
    )
    base.update(overrides)
    params = StrategyParams(**base)
    agg = SignalAggregator(strategy_params=params, min_rr_ratio=0.1)
    agg.level_store = LevelStore()
    # Provide a couple of upside reference levels so target/R:R math
    # doesn't bail before we reach the freshness gate.
    agg.level_store.add(_level(7527.0, LevelType.HORIZONTAL_SR, _NOW))
    agg.level_store.add(_level(7540.0, LevelType.HORIZONTAL_SR, _NOW))
    return agg


# ---------------------------------------------------------------------------
# Default-off (flag False): current behaviour is preserved
# ---------------------------------------------------------------------------


def test_default_flag_is_false():
    """Pure config sanity — the new flag is off by default."""
    assert StrategyParams().fb_age_cap_exempt_high_quality_levels is False


def test_default_off_old_pdl_still_rejected():
    """Flag False (default): an old PRIOR_DAY_LOW is rejected, matching
    the existing freshness-gate behaviour."""
    agg = _agg(fb_age_cap_exempt_high_quality_levels=False)
    pattern = _fb_pattern(level_age_hours=72.0, level_type=LevelType.PRIOR_DAY_LOW)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is None


# ---------------------------------------------------------------------------
# Flag On: structural multi-day levels are exempt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level_type",
    [
        LevelType.PRIOR_DAY_LOW,
        LevelType.MULTI_HOUR_LOW,
        LevelType.INTRADAY_LOW,
        LevelType.CUSTOM,
    ],
)
def test_flag_on_exempt_types_pass_when_old(level_type):
    """Flag True: the four high-quality structural types bypass the cap
    even when the level is well past 36h old."""
    agg = _agg(fb_age_cap_exempt_high_quality_levels=True)
    pattern = _fb_pattern(level_age_hours=24.0 * 6, level_type=level_type)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is not None, (
        f"{level_type.name} should be exempt from the age cap"
    )


def test_flag_on_old_pdl_passes():
    """Concrete spelling of the parametrize for the 7517-style case:
    a 6-day-old PRIOR_DAY_LOW must be allowed when the flag is on."""
    agg = _agg(fb_age_cap_exempt_high_quality_levels=True)
    pattern = _fb_pattern(level_age_hours=24.0 * 6, level_type=LevelType.PRIOR_DAY_LOW)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is not None


def test_flag_on_old_mhl_passes():
    agg = _agg(fb_age_cap_exempt_high_quality_levels=True)
    pattern = _fb_pattern(level_age_hours=24.0 * 4, level_type=LevelType.MULTI_HOUR_LOW)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is not None


def test_flag_on_old_intraday_low_passes():
    agg = _agg(fb_age_cap_exempt_high_quality_levels=True)
    pattern = _fb_pattern(level_age_hours=72.0, level_type=LevelType.INTRADAY_LOW)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is not None


def test_flag_on_old_custom_passes():
    """Mancini's own plan-named levels are injected as CUSTOM and must
    remain valid for the whole week he flags them."""
    agg = _agg(fb_age_cap_exempt_high_quality_levels=True)
    pattern = _fb_pattern(level_age_hours=24.0 * 5, level_type=LevelType.CUSTOM)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is not None


# ---------------------------------------------------------------------------
# Flag On: non-exempt types are still gated
# ---------------------------------------------------------------------------


def test_flag_on_old_swing_low_still_rejected():
    """SWING_LOW is NOT in the exempt set — old swings still get gated
    even when the flag is on."""
    agg = _agg(fb_age_cap_exempt_high_quality_levels=True)
    pattern = _fb_pattern(level_age_hours=72.0, level_type=LevelType.SWING_LOW)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is None


def test_flag_on_old_cluster_low_still_rejected():
    """CLUSTER_LOW is NOT in the exempt set — the noisy mid-range
    cluster type stays under the cap."""
    agg = _agg(fb_age_cap_exempt_high_quality_levels=True)
    pattern = _fb_pattern(level_age_hours=72.0, level_type=LevelType.CLUSTER_LOW)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is None


# ---------------------------------------------------------------------------
# Regression: fresh levels under the cap pass regardless of type/flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level_type",
    [
        LevelType.PRIOR_DAY_LOW,
        LevelType.MULTI_HOUR_LOW,
        LevelType.INTRADAY_LOW,
        LevelType.CUSTOM,
        LevelType.SWING_LOW,
        LevelType.CLUSTER_LOW,
    ],
)
@pytest.mark.parametrize("flag", [False, True])
def test_fresh_level_passes_regardless_of_type_and_flag(level_type, flag):
    """A 4-hour-old level is well under the 36h cap, so it passes the
    freshness gate for every level type whether the flag is on or off.
    This is the regression guard — the exemption must never change
    behaviour for fresh levels."""
    agg = _agg(fb_age_cap_exempt_high_quality_levels=flag)
    pattern = _fb_pattern(level_age_hours=4.0, level_type=level_type)
    sig = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert sig is not None, (
        f"fresh {level_type.name} should pass (flag={flag})"
    )
