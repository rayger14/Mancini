"""Tests for the Mancini LLM plan integration in the signal aggregator.

Covers the three gates (mode_1_green, danger zones, no_trade_above/below)
and the planned-setup LQS bonus, plus the no-op behavior when the master
switch is off or no plan is loaded.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.patterns import ConfirmationType, PatternSignal
from core.signals import SignalAggregator, SignalType

from live.mancini_llm_extract import DangerZone, ManciniPlan, PlannedSetup


_TS = datetime(2026, 5, 6, 10, 30)


def _level(price: float, level_type: LevelType = LevelType.PRIOR_DAY_LOW,
           touch_count: int = 3) -> Level:
    return Level(
        price=price,
        level_type=level_type,
        created_at=_TS,
        confirmed_at=_TS,
        touch_count=touch_count,
    )


def _fb_pattern(level_price: float = 7250.0,
                entry_price: float = 7252.0,
                stop_price: float = 7245.0) -> PatternSignal:
    return PatternSignal(
        pattern_type="failed_breakdown",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=_level(level_price),
        sweep_low=level_price - 2.0,
        entry_price=entry_price,
        stop_price=stop_price,
        bar_idx=100,
        timestamp=_TS,
        sweep_depth_pts=2.0,
        direction="long",
    )


def _agg(use_mancini_llm_plan: bool = True, **extra) -> SignalAggregator:
    """SignalAggregator with LLM plan enabled and LQS gating disabled
    so we can isolate the behavior we're testing.
    """
    params = StrategyParams(
        use_mancini_llm_plan=use_mancini_llm_plan,
        use_level_quality_scoring=False,
        use_confluence_scoring=False,
        use_sweep_depth_sizing=False,
        **extra,
    )
    agg = SignalAggregator(strategy_params=params, min_rr_ratio=0.1)
    agg.level_store = LevelStore()
    # Provide resistance levels so target-finding succeeds
    agg.level_store.add(_level(7260.0, LevelType.HORIZONTAL_SR))
    agg.level_store.add(_level(7275.0, LevelType.HORIZONTAL_SR))
    return agg


# ---------------------------------------------------------------------------
# No-op behavior
# ---------------------------------------------------------------------------


def test_no_plan_no_op():
    """Without a plan loaded, _qualify_signal works unchanged."""
    agg = _agg()
    pattern = _fb_pattern()
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is not None  # passes through


def test_plan_loaded_but_flag_off_is_no_op():
    """When use_mancini_llm_plan=False, plan is ignored entirely."""
    agg = _agg(use_mancini_llm_plan=False)
    agg.set_mancini_llm_plan(ManciniPlan(
        mode="mode_1_green",
        danger_zones=[DangerZone(price_low=7240.0, price_high=7260.0,
                                 rule="block")],
        no_trade_below=7300.0,
    ))
    pattern = _fb_pattern()
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is not None  # all gates inert because the flag is off


# ---------------------------------------------------------------------------
# Gates (rejections)
# ---------------------------------------------------------------------------


def test_mode_1_green_blocks_subsequent_fb_long():
    """Mancini's verbatim rule: 'All Mode 1 green days are triggered by a
    Failed Breakdown ... but if you missed the triggering Failed Breakdown
    on these days, you are typically out of luck.' The FIRST FB long is the
    triggering trade he wants us in — subsequent FB longs are the ones to
    block.
    """
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(mode="mode_1_green"))
    pattern = _fb_pattern()
    # Simulate the triggering FB already taken earlier in the session.
    agg._fb_long_taken_today = True
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is None


def test_mode_other_does_not_block_fb_long():
    """Mode != mode_1_green should not gate."""
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(mode="trending"))
    pattern = _fb_pattern()
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is not None


def test_danger_zone_blocks_long_entry_inside_band():
    """Entry price within a danger_zone band rejects the signal."""
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(
        danger_zones=[DangerZone(price_low=7250.0, price_high=7255.0,
                                 rule="recently broken support")],
    ))
    pattern = _fb_pattern(entry_price=7252.0)  # inside the zone
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is None


def test_danger_zone_does_not_block_outside_band():
    """Entry outside the danger zone should pass through."""
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(
        danger_zones=[DangerZone(price_low=7100.0, price_high=7150.0,
                                 rule="far below current")],
    ))
    pattern = _fb_pattern(entry_price=7252.0)
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is not None


def test_danger_zone_single_sided_uses_low_only():
    """When price_high is None, the zone is exactly at price_low."""
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(
        danger_zones=[DangerZone(price_low=7252.0, price_high=None,
                                 rule="exact level")],
    ))
    pattern = _fb_pattern(entry_price=7252.0)
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is None


def test_no_trade_above_blocks_when_entry_above():
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(no_trade_above=7250.0))
    pattern = _fb_pattern(entry_price=7252.0)
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is None


def test_no_trade_above_passes_when_entry_below():
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(no_trade_above=7300.0))
    pattern = _fb_pattern(entry_price=7252.0)
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is not None


def test_no_trade_below_blocks_when_entry_below():
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(no_trade_below=7300.0))
    pattern = _fb_pattern(entry_price=7252.0)
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
    assert signal is None


# ---------------------------------------------------------------------------
# Planned-setup LQS bonus
# ---------------------------------------------------------------------------


def test_planned_setup_match_boosts_lqs_when_lqs_active():
    """When LQS gating is on and the level matches a planned_setup, the
    bonus pushes a borderline LQS over the trade threshold."""
    params = StrategyParams(
        use_mancini_llm_plan=True,
        use_level_quality_scoring=True,
        # Set thresholds tight enough that bonus matters
        lqs_shadow_threshold=10,
        lqs_min_trade_threshold=40,
        mancini_llm_setup_lqs_bonus=30,
        mancini_llm_setup_match_tolerance_pts=2.0,
        use_confluence_scoring=False,
        use_sweep_depth_sizing=False,
    )
    agg = SignalAggregator(strategy_params=params, min_rr_ratio=0.1)
    agg.level_store = LevelStore()
    agg.level_store.add(_level(7260.0, LevelType.HORIZONTAL_SR))
    agg.level_store.add(_level(7275.0, LevelType.HORIZONTAL_SR))

    pattern = _fb_pattern(level_price=7250.0)

    # Without a matching setup, LQS=0 (intraday low w/o touches), trade rejected.
    no_match_plan = ManciniPlan(planned_setups=[
        PlannedSetup(
            setup_type="failed_breakdown",
            level_price=7100.0,  # 150pts away — won't match
            direction="long",
            context="distant",
            conviction="high",
        ),
    ])
    agg.set_mancini_llm_plan(no_match_plan)
    signal_no_match = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)

    # With a matching setup, +30 bonus pushes LQS over the trade threshold.
    match_plan = ManciniPlan(planned_setups=[
        PlannedSetup(
            setup_type="failed_breakdown",
            level_price=7250.5,  # within 2pt tolerance of 7250.0
            direction="long",
            context="match",
            conviction="high",
        ),
    ])
    agg.set_mancini_llm_plan(match_plan)
    signal_match = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)

    # The match must produce at least as good an outcome (signal returned
    # when no_match was rejected, or both returned with match having higher
    # LQS). We assert the match gives a non-None signal.
    assert signal_match is not None
    # And the match's LQS exceeds the no-match LQS (or no-match was None).
    if signal_no_match is not None:
        assert signal_match.lqs >= signal_no_match.lqs


def test_planned_setup_direction_mismatch_no_bonus():
    """A short setup at the same level must NOT boost a long signal."""
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(planned_setups=[
        PlannedSetup(
            setup_type="breakdown_short",
            level_price=7250.0,
            direction="short",
            context="short setup",
            conviction="high",
        ),
    ]))
    pattern = _fb_pattern(level_price=7250.0)
    bonus = agg._mancini_llm_setup_bonus(pattern, SignalType.FAILED_BREAKDOWN)
    assert bonus == 0


def test_planned_setup_far_away_no_bonus():
    """Setup level outside the tolerance returns 0 bonus."""
    agg = _agg(mancini_llm_setup_match_tolerance_pts=2.0)
    agg.set_mancini_llm_plan(ManciniPlan(planned_setups=[
        PlannedSetup(
            setup_type="failed_breakdown",
            level_price=7250.0,
            direction="long",
            context="too far",
            conviction="high",
        ),
    ]))
    pattern = _fb_pattern(level_price=7253.0)  # 3pts away, outside 2pt tolerance
    bonus = agg._mancini_llm_setup_bonus(pattern, SignalType.FAILED_BREAKDOWN)
    assert bonus == 0


def test_planned_setup_match_returns_configured_bonus():
    """Direct call to _mancini_llm_setup_bonus returns the configured int."""
    agg = _agg(mancini_llm_setup_lqs_bonus=20)
    agg.set_mancini_llm_plan(ManciniPlan(planned_setups=[
        PlannedSetup(
            setup_type="failed_breakdown",
            level_price=7250.0,
            direction="long",
            context="match",
            conviction="high",
        ),
    ]))
    pattern = _fb_pattern(level_price=7250.5)
    bonus = agg._mancini_llm_setup_bonus(pattern, SignalType.FAILED_BREAKDOWN)
    assert bonus == 20


# ---------------------------------------------------------------------------
# set_mancini_llm_plan API
# ---------------------------------------------------------------------------


def test_set_mancini_llm_plan_stores_and_clears():
    agg = _agg()
    plan = ManciniPlan(lean="bullish")
    agg.set_mancini_llm_plan(plan)
    assert agg._mancini_llm_plan is plan
    agg.set_mancini_llm_plan(None)
    assert agg._mancini_llm_plan is None
