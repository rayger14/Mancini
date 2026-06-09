"""Tests for Mancini's Mode 1 Green "first FB only" rule.

Mancini's verbatim rule:
    "All Mode 1 green days are triggered by a Failed Breakdown (but not all
    Failed Breakdowns trigger Mode 1 green days, a key distinction), but if
    you missed the triggering Failed Breakdown on these days, you are
    typically out of luck."

The triggering FB is the trade he WANTS us in. Only subsequent FB longs on
the same session should be blocked. These tests exercise the refined gate
in ``SignalAggregator._check_mancini_llm_gates`` together with the
``_fb_long_taken_today`` session-scoped flag set inside ``_qualify_signal``.
"""

from __future__ import annotations

from datetime import datetime

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.patterns import ConfirmationType, PatternSignal
from core.signals import SignalAggregator, SignalType

from live.mancini_llm_extract import ManciniPlan


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
                stop_price: float = 7245.0,
                direction: str = "long") -> PatternSignal:
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
        direction=direction,
    )


def _lr_pattern(level_price: float = 7250.0,
                entry_price: float = 7252.0,
                stop_price: float = 7245.0) -> PatternSignal:
    return PatternSignal(
        pattern_type="level_reclaim",
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
    """SignalAggregator with LLM plan enabled and LQS/confluence gates off
    so we can isolate the Mode 1 Green gate behavior.
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
# The "first FB only" rule on Mode 1 Green
# ---------------------------------------------------------------------------


def test_mode_1_green_allows_first_fb_long():
    """The first (triggering) FB long of a Mode 1 Green session must pass.

    Mancini: the open-to-close trend day is BORN from a Failed Breakdown.
    Blocking the very signal that kicks it off would defeat the strategy.
    """
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(mode="mode_1_green"))
    assert agg._fb_long_taken_today is False

    pattern = _fb_pattern()
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)

    assert signal is not None
    # The flag must flip to True after a fully-qualified FB long.
    assert agg._fb_long_taken_today is True


def test_mode_1_green_blocks_second_fb_long():
    """After the triggering FB long, all subsequent FB longs are blocked
    on the same Mode 1 Green session.
    """
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(mode="mode_1_green"))
    # Simulate the triggering FB already taken earlier this session.
    agg._fb_long_taken_today = True

    pattern = _fb_pattern()
    signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)

    assert signal is None


def test_mode_1_green_blocks_after_first_fb_long_via_qualify():
    """End-to-end: run two FB longs back-to-back on the same Mode 1 Green
    plan. First passes, second is rejected — without manually setting the
    flag. Exercises the wire-up between ``_qualify_signal`` (which flips
    the flag) and ``_check_mancini_llm_gates`` (which reads it).
    """
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(mode="mode_1_green"))

    first = agg._qualify_signal(_fb_pattern(), SignalType.FAILED_BREAKDOWN)
    second = agg._qualify_signal(_fb_pattern(), SignalType.FAILED_BREAKDOWN)

    assert first is not None
    assert second is None


# ---------------------------------------------------------------------------
# Regression: non-mode_1_green plans must not be affected
# ---------------------------------------------------------------------------


def test_non_mode_1_green_plan_unaffected_by_first_fb_rule():
    """On a non Mode 1 Green plan, FB longs are NOT gated by the
    ``_fb_long_taken_today`` flag — multiple FB longs per session are fine.
    """
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(mode="trending"))

    first = agg._qualify_signal(_fb_pattern(), SignalType.FAILED_BREAKDOWN)
    # Even if the flag were True from a prior session, non-Green plans
    # should still allow FB longs through.
    agg._fb_long_taken_today = True
    second = agg._qualify_signal(_fb_pattern(), SignalType.FAILED_BREAKDOWN)

    assert first is not None
    assert second is not None


# ---------------------------------------------------------------------------
# The rule only targets FB longs — other signal types are not impacted
# ---------------------------------------------------------------------------


def test_mode_1_green_allows_level_reclaim_long_even_after_first_fb():
    """Mancini's rule is specific to Failed Breakdowns. A LEVEL_RECLAIM
    long must pass on a Mode 1 Green day even when an FB long was already
    taken.
    """
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(mode="mode_1_green"))
    agg._fb_long_taken_today = True  # FB already taken this session

    signal = agg._qualify_signal(_lr_pattern(), SignalType.LEVEL_RECLAIM)

    assert signal is not None


def test_mode_1_green_allows_fb_short_even_after_first_fb_long():
    """Mancini's rule is for FB LONGS only. An FB pattern marked as a
    short (direction='short') must not be gated by the first-FB-long flag.
    The qualifier path here is long-only, so we test the gate function
    directly to confirm it does not reject FB shorts.
    """
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(mode="mode_1_green"))
    agg._fb_long_taken_today = True  # FB long already taken this session

    short_pattern = _fb_pattern(direction="short")
    reject = agg._check_mancini_llm_gates(
        short_pattern, SignalType.FAILED_BREAKDOWN,
    )

    # The Mode 1 Green gate must not reject a short — only longs.
    assert reject is None


# ---------------------------------------------------------------------------
# reset() clears the flag for the next session
# ---------------------------------------------------------------------------


def test_reset_clears_fb_long_taken_today():
    """``reset()`` must zero the flag so the new session can take its own
    triggering FB long.
    """
    agg = _agg()
    agg.set_mancini_llm_plan(ManciniPlan(mode="mode_1_green"))
    agg._fb_long_taken_today = True

    agg.reset()

    assert agg._fb_long_taken_today is False
