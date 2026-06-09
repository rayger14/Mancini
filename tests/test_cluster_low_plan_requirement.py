"""Tests for the CLUSTER_LOW plan-match requirement gate.

Per the 5y leak audit (2026-06-08), engine-derived CLUSTER_LOW levels
generate ~98% of acceptance-protocol FB long entries and account for
roughly $94K in losses. Mancini's playbook warns "mid-range entries are
noisy" — CLUSTER_LOW is that noisy mid-range cluster.

When ``strategy_params.cluster_low_requires_plan_match`` is True, an
FB-long / LR-long whose underlying ``pattern.level.level_type`` is
``CLUSTER_LOW`` may only fire if Mancini's plan has a matching long
setup within ``mancini_llm_setup_match_tolerance_pts``. Otherwise the
signal is rejected.

These tests cover:
- Flag False (default) → CLUSTER_LOW signals pass unchanged.
- Flag True + no plan → CLUSTER_LOW long is rejected.
- Flag True + plan with matching long setup within tolerance → passes.
- Flag True + plan with no matching setup → rejected.
- Flag True + plan with matching SHORT setup at same price → rejected
  (direction must match long).
- Flag True + PRIOR_DAY_LOW with no plan → passes (only CLUSTER_LOW
  is gated; other types untouched).
- Flag True + FAILED_BREAKDOWN short on CLUSTER_LOW → passes (only
  long signals are gated).
"""

from __future__ import annotations

from datetime import datetime

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.patterns import ConfirmationType, PatternSignal
from core.signals import SignalAggregator, SignalType

from live.mancini_llm_extract import ManciniPlan, PlannedSetup


_TS = datetime(2026, 6, 8, 10, 30)


def _level(price: float, level_type: LevelType = LevelType.CLUSTER_LOW,
           touch_count: int = 5) -> Level:
    return Level(
        price=price,
        level_type=level_type,
        created_at=_TS,
        confirmed_at=_TS,
        touch_count=touch_count,
    )


def _fb_pattern(
    level_price: float = 7517.0,
    entry_price: float = 7519.0,
    stop_price: float = 7512.5,
    level_type: LevelType = LevelType.CLUSTER_LOW,
    direction: str = "long",
) -> PatternSignal:
    return PatternSignal(
        pattern_type="failed_breakdown",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=_level(level_price, level_type=level_type),
        sweep_low=level_price - 2.0,
        entry_price=entry_price,
        stop_price=stop_price,
        bar_idx=100,
        timestamp=_TS,
        sweep_depth_pts=2.0,
        direction=direction,
    )


def _agg(
    cluster_low_requires_plan_match: bool = False,
    use_mancini_llm_plan: bool = True,
    **extra,
) -> SignalAggregator:
    """SignalAggregator wired for the gate-under-test.

    We disable LQS / confluence / sweep-depth sizing so the gate's
    pass/reject outcome is the only thing _qualify_signal turns on.
    Resistance levels are pre-seeded so target finding succeeds for
    the passing cases.
    """
    params = StrategyParams(
        cluster_low_requires_plan_match=cluster_low_requires_plan_match,
        use_mancini_llm_plan=use_mancini_llm_plan,
        use_level_quality_scoring=False,
        use_confluence_scoring=False,
        use_sweep_depth_sizing=False,
        **extra,
    )
    agg = SignalAggregator(strategy_params=params, min_rr_ratio=0.1)
    agg.level_store = LevelStore()
    # Resistance levels above the typical CLUSTER_LOW entry so target
    # finding succeeds and we don't accidentally reject for missing T1.
    agg.level_store.add(_level(7530.0, LevelType.HORIZONTAL_SR, touch_count=3))
    agg.level_store.add(_level(7545.0, LevelType.HORIZONTAL_SR, touch_count=3))
    return agg


# ---------------------------------------------------------------------------
# Flag off → gate is a no-op
# ---------------------------------------------------------------------------


def test_flag_off_cluster_low_no_plan_passes():
    """Default (flag False): CLUSTER_LOW FB long with no plan loaded
    must pass — the gate is a pure no-op until the flag is flipped on.
    """
    agg = _agg(cluster_low_requires_plan_match=False)
    pattern = _fb_pattern()
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is not None


# ---------------------------------------------------------------------------
# Flag on → rejection paths
# ---------------------------------------------------------------------------


def test_flag_on_cluster_low_no_plan_rejects():
    """Flag True + no plan loaded => CLUSTER_LOW long is rejected."""
    agg = _agg(cluster_low_requires_plan_match=True)
    pattern = _fb_pattern()
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is None


def test_flag_on_cluster_low_plan_no_match_rejects():
    """Flag True + plan loaded but no setup within tolerance => reject."""
    agg = _agg(cluster_low_requires_plan_match=True)
    agg.set_mancini_llm_plan(ManciniPlan(
        planned_setups=[
            # Far away from our CLUSTER_LOW @ 7517 (default tolerance 2 pts)
            PlannedSetup(
                setup_type="failed_breakdown",
                level_price=7400.0,
                direction="long",
                context="far below",
                conviction="medium",
            ),
        ],
    ))
    pattern = _fb_pattern(level_price=7517.0)
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is None


def test_flag_on_cluster_low_plan_matching_short_setup_rejects():
    """Direction discipline: a SHORT setup at the same price doesn't
    bless a CLUSTER_LOW long entry — the gate requires direction='long'.
    """
    agg = _agg(cluster_low_requires_plan_match=True)
    agg.set_mancini_llm_plan(ManciniPlan(
        planned_setups=[
            PlannedSetup(
                setup_type="breakdown_short",
                level_price=7517.0,  # exact match on price
                direction="short",   # but wrong direction
                context="break of 7517 shorts",
                conviction="high",
            ),
        ],
    ))
    pattern = _fb_pattern(level_price=7517.0)
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is None


# ---------------------------------------------------------------------------
# Flag on → pass paths
# ---------------------------------------------------------------------------


def test_flag_on_cluster_low_plan_matching_long_setup_passes():
    """Mancini blessed an FB long at this level => CLUSTER_LOW passes."""
    agg = _agg(cluster_low_requires_plan_match=True)
    agg.set_mancini_llm_plan(ManciniPlan(
        planned_setups=[
            PlannedSetup(
                setup_type="failed_breakdown",
                level_price=7518.0,  # within 2pt tolerance of 7517 level
                direction="long",
                context="FB of 7518 cluster",
                conviction="high",
            ),
        ],
    ))
    pattern = _fb_pattern(level_price=7517.0)
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is not None


def test_flag_on_prior_day_low_no_plan_passes():
    """Only CLUSTER_LOW is gated. PRIOR_DAY_LOW longs must still flow
    even when no plan is loaded — they're Mancini's primary FB setup.
    """
    agg = _agg(cluster_low_requires_plan_match=True)
    pattern = _fb_pattern(level_type=LevelType.PRIOR_DAY_LOW)
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is not None


def test_flag_on_cluster_low_short_signal_passes():
    """The gate only applies to long signals (per the audit, the leak
    is acceptance-protocol FB *longs* on CLUSTER_LOW). A FAILED_BREAKDOWN
    short at a CLUSTER_LOW level — even with no plan — is not gated here.
    """
    agg = _agg(cluster_low_requires_plan_match=True)
    # We just verify the gate method itself doesn't reject the short.
    # (_qualify_signal has other short-side logic we don't want to
    # entangle this test with.)
    pattern = _fb_pattern(direction="short")
    reason = agg._check_cluster_low_plan_requirement(
        pattern, SignalType.FAILED_BREAKDOWN
    )
    assert reason is None


def test_flag_on_cluster_low_lr_long_no_plan_rejects():
    """LR longs are gated the same way as FB longs (both are the noisy
    CLUSTER_LOW path in the leak analysis).
    """
    agg = _agg(cluster_low_requires_plan_match=True)
    pattern = _fb_pattern()  # CLUSTER_LOW long
    reason = agg._check_cluster_low_plan_requirement(
        pattern, SignalType.LEVEL_RECLAIM
    )
    assert reason is not None
    assert "CLUSTER_LOW" in reason


def test_flag_on_other_signal_type_passes_through():
    """Signal types outside FB/LR aren't gated — e.g. a BREAKDOWN_SHORT
    on a CLUSTER_LOW shouldn't even reach this filter.
    """
    agg = _agg(cluster_low_requires_plan_match=True)
    pattern = _fb_pattern()
    reason = agg._check_cluster_low_plan_requirement(
        pattern, SignalType.BREAKDOWN_SHORT
    )
    assert reason is None
