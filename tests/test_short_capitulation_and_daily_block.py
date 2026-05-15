"""Regression tests for the short-side capitulation guard and the
DAILY_FB_BULL hard-block.

Each "should reject" test below corresponds to a real losing live trade
identified in the 5/2026 short-side post-mortem. Each replays the same
session context (entry, session_high, session_low, daily_bias) and
asserts the qualification path now returns None where it previously
returned a Signal.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.patterns import ConfirmationType, PatternSignal
from core.patterns_short_v2 import VelocityBreakdownShort
from core.signals import SignalAggregator, SignalType


_TS = datetime(2026, 5, 12, 10, 47)


def _level(price: float, level_type: LevelType = LevelType.PRIOR_DAY_LOW,
           touches: int = 3) -> Level:
    return Level(
        price=price,
        level_type=level_type,
        created_at=_TS,
        confirmed_at=_TS,
        touch_count=touches,
    )


def _short_pattern(level_price: float, entry_price: float,
                   stop_price: float | None = None,
                   signal_type_for_pattern: str = "breakdown_short"
                   ) -> PatternSignal:
    """Build a short-side PatternSignal in the same shape the live
    BreakdownShort / VelocityBreakdownShort detectors emit.
    """
    return PatternSignal(
        pattern_type=signal_type_for_pattern,
        confirmation=ConfirmationType.ACCEPTANCE,
        level=_level(level_price),
        sweep_low=level_price - 3.0,
        entry_price=entry_price,
        stop_price=stop_price if stop_price is not None else level_price + 4.0,
        bar_idx=200,
        timestamp=_TS,
        sweep_depth_pts=3.0,
        direction="short",
        sweep_high=level_price,
    )


def _agg(session_high: float, session_low: float,
         daily_bias: str = "NEUTRAL",
         use_daily_structure: bool = True,
         block_capitulation_shorts: bool = True,
         **extra) -> SignalAggregator:
    """SignalAggregator with the gates we're testing enabled and other
    noise (confluence, sweep-depth-sizing, volume) turned off so the
    test isolates the gate behavior.
    """
    defaults = dict(
        allow_breakdown_short=True,
        bd_short_min_rr=0.1,           # don't reject for RR
        min_signal_rr=0.1,
        use_daily_structure=use_daily_structure,
        use_level_quality_scoring=False,
        use_confluence_scoring=False,
        use_sweep_depth_sizing=False,
        block_capitulation_shorts=block_capitulation_shorts,
        block_pdl_shorts=False,        # test the capitulation gate, not PDL
    )
    defaults.update(extra)
    params = StrategyParams(**defaults)
    agg = SignalAggregator(strategy_params=params, min_rr_ratio=0.1)
    agg.level_store = LevelStore()
    # Add supports below typical entry so target-finding succeeds
    agg.level_store.add(_level(7340.0, LevelType.PRIOR_DAY_LOW))
    agg.level_store.add(_level(7300.0, LevelType.HORIZONTAL_SR))
    agg.level_store.add(_level(7270.0, LevelType.HORIZONTAL_SR))
    agg._session_high = session_high
    agg._session_low = session_low
    agg._daily_bias = daily_bias
    return agg


# ---------------------------------------------------------------------------
# Capitulation-entry guard — replay each real loser
# ---------------------------------------------------------------------------


def test_capitulation_blocks_may_12_bd_short_loss():
    """5/12 BD short: entry=7387 session_low=7380 session_high=7420.75.
    Entry is 7pt above session_low, 33.75pt below session_high — pure
    capitulation. Should reject under default thresholds (10 / 25).
    """
    agg = _agg(session_high=7420.75, session_low=7380.0)
    pattern = _short_pattern(level_price=7385.0, entry_price=7387.0,
                             stop_price=7395.0)
    signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
    assert signal is None


def test_capitulation_blocks_velocity_short_5_07_loss():
    """5/07 velocity short: entry=7350.5 session_low=7348.5 session_high=7410.
    Entry 2pt off the floor, 59.5pt off the high. -72pt loss in production.
    """
    agg = _agg(session_high=7410.0, session_low=7348.5)
    pattern = _short_pattern(level_price=7365.5, entry_price=7350.5,
                             stop_price=7369.0,
                             signal_type_for_pattern="velocity_short")
    signal = agg._qualify_short_signal(pattern, SignalType.VELOCITY_SHORT)
    assert signal is None


def test_capitulation_blocks_velocity_short_5_04_loss():
    """5/04 velocity short: entry=7208.75 session_low=7208.25 session_high=7271.
    Entry literally 0.5pt above the floor, 62.25pt off the high. -19pt.
    """
    agg = _agg(session_high=7271.0, session_low=7208.25)
    pattern = _short_pattern(level_price=7240.75, entry_price=7208.75,
                             stop_price=7212.0,
                             signal_type_for_pattern="velocity_short")
    signal = agg._qualify_short_signal(pattern, SignalType.VELOCITY_SHORT)
    assert signal is None


def test_capitulation_does_not_block_mid_range_entry():
    """Entry well within the session range should pass the gate. Sanity
    check that we're not over-rejecting.
    """
    # Entry 20pt off low AND 20pt off high — fails BOTH conditions
    agg = _agg(session_high=7400.0, session_low=7360.0)
    pattern = _short_pattern(level_price=7385.0, entry_price=7380.0,
                             stop_price=7388.0)
    signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
    assert signal is not None


def test_capitulation_does_not_block_early_session_entry():
    """Entry near session high (early session, before flush): the bottom
    leg of the AND condition fails (session_high - entry < 25), so the
    gate stays silent. Sanity check for the AND-not-OR semantics.
    """
    # Entry IS within 10pt of session_low, but session_high is only 12pt above
    agg = _agg(session_high=7397.0, session_low=7380.0)
    pattern = _short_pattern(level_price=7390.0, entry_price=7385.0,
                             stop_price=7393.0)
    signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
    assert signal is not None


def test_capitulation_can_be_disabled():
    """Setting block_capitulation_shorts=False must let the trade through
    even when the thresholds are met. Knob exists for backtest sweeps.
    """
    agg = _agg(session_high=7420.75, session_low=7380.0,
               block_capitulation_shorts=False)
    pattern = _short_pattern(level_price=7385.0, entry_price=7387.0,
                             stop_price=7395.0)
    signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
    assert signal is not None


# ---------------------------------------------------------------------------
# DAILY_FB_BULL hard block — the smoking gun
# ---------------------------------------------------------------------------


def test_daily_fb_bull_now_blocks_low_lqs_short():
    """The 5/12 BD short fired with daily_bias=DAILY_FB_BULL and LQS=47
    (below daily_bd_short_min_lqs=70). Pre-fix: marked-but-not-blocked
    and traded. Post-fix: return None.
    """
    # Disable capitulation gate so we isolate the daily-bias block
    agg = _agg(session_high=7400.0, session_low=7360.0,
               daily_bias="DAILY_FB_BULL",
               block_capitulation_shorts=False,
               use_level_quality_scoring=True,
               lqs_shadow_threshold=0,
               lqs_min_trade_threshold=0,
               daily_bd_short_min_lqs=70)
    pattern = _short_pattern(level_price=7385.0, entry_price=7380.0,
                             stop_price=7388.0)
    signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
    assert signal is None


def test_daily_fb_bull_does_not_block_high_lqs_short():
    """When LQS exceeds daily_bd_short_min_lqs, the daily gate stays
    silent (a strong-quality short overrides the bias)."""
    agg = _agg(session_high=7400.0, session_low=7360.0,
               daily_bias="DAILY_FB_BULL",
               block_capitulation_shorts=False,
               use_level_quality_scoring=False,  # bypass post-gate threshold
               daily_bd_short_min_lqs=20)  # threshold low enough to be exceeded
    pattern = _short_pattern(level_price=7385.0, entry_price=7380.0,
                             stop_price=7388.0)

    # Force a high LQS by stubbing the scorer
    agg._level_scorer.compute_lqs = lambda *_a, **_k: 75
    signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
    assert signal is not None


def test_neutral_daily_bias_does_not_block():
    """Non-DAILY_FB_BULL bias should not engage the gate at all."""
    agg = _agg(session_high=7400.0, session_low=7360.0,
               daily_bias="NEUTRAL",
               block_capitulation_shorts=False,
               use_level_quality_scoring=False)
    agg._level_scorer.compute_lqs = lambda *_a, **_k: 30  # would fail if gate fired
    pattern = _short_pattern(level_price=7385.0, entry_price=7380.0,
                             stop_price=7388.0)
    signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
    assert signal is not None


# ---------------------------------------------------------------------------
# vbd_max_break_pts — pattern-specific cap on the velocity detector
# ---------------------------------------------------------------------------


def _vbd(max_break_pts: float = 20.0) -> VelocityBreakdownShort:
    return VelocityBreakdownShort(StrategyParams(
        allow_velocity_short=True,
        vbd_min_break_pts=8.0,
        vbd_max_break_pts=max_break_pts,
        vbd_min_volume_ratio=0.0,  # disable volume check for unit test
        vbd_require_close_below=True,
        vbd_only_major_levels=True,
    ))


def _vbd_store(level_price: float) -> LevelStore:
    s = LevelStore()
    s.add(_level(level_price, LevelType.PRIOR_DAY_LOW))
    return s


def test_vbd_max_break_caps_excess_sweep():
    """A 49pt one-bar print (T4 from the post-mortem) should NOT emit."""
    vbd = _vbd(max_break_pts=20.0)
    store = _vbd_store(7140.0)
    # Bar takes the level by 49.25pt — back-half-of-crash territory
    sig = vbd.update(
        bar_idx=0, timestamp=_TS,
        high=7141.0, low=7090.75, close=7110.5,
        volume=10000, avg_volume_20=1000, level_store=store,
    )
    assert sig is None


def test_vbd_allows_reasonable_break():
    """A clean ~15pt one-bar break should still emit. Sanity check."""
    vbd = _vbd(max_break_pts=20.0)
    store = _vbd_store(7140.0)
    sig = vbd.update(
        bar_idx=0, timestamp=_TS,
        high=7141.0, low=7125.0, close=7126.0,  # 15pt break, closes below
        volume=10000, avg_volume_20=1000, level_store=store,
    )
    assert sig is not None
    assert sig.sweep_depth_pts == pytest.approx(15.0)


def test_vbd_max_break_zero_disables_cap():
    """vbd_max_break_pts=0.0 should disable the cap (backtest knob)."""
    vbd = _vbd(max_break_pts=0.0)
    store = _vbd_store(7140.0)
    sig = vbd.update(
        bar_idx=0, timestamp=_TS,
        high=7141.0, low=7090.75, close=7110.5,
        volume=10000, avg_volume_20=1000, level_store=store,
    )
    assert sig is not None


# ---------------------------------------------------------------------------
# Mancini-aligned PDL short block (Phase 1 of short-engine rewrite)
# ---------------------------------------------------------------------------


def test_pdl_short_blocked_by_default():
    """PRIOR_DAY_LOW shorts should be rejected with the default
    block_pdl_shorts=True flag. Live data 2026-02-25 → 2026-05-12:
    5/5 PDL shorts lost ($-813). Mancini explicitly classifies PDL as
    a long-side Failed Breakdown level."""
    agg = _agg(session_high=7400.0, session_low=7370.0,
               block_pdl_shorts=True)  # _agg helper defaults this False
    # Use a non-capitulation entry so we know the PDL gate is what blocks
    pattern = _short_pattern(level_price=7390.0, entry_price=7388.0,
                             stop_price=7395.0)
    # Confirm pattern level is PDL (default in helper)
    assert pattern.level.level_type.name == "PRIOR_DAY_LOW"

    signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
    assert signal is None

    pdl_events = [e for e in agg.shadow_events
                  if e.get("feature") == "block_pdl_shorts"]
    assert len(pdl_events) == 1
    assert pdl_events[0]["signal_type"] == "BREAKDOWN_SHORT"
    assert pdl_events[0]["level_price"] == 7390.0


def test_pdl_short_blocks_velocity_short_too():
    """The gate applies to both BREAKDOWN_SHORT and VELOCITY_SHORT —
    they share the same _qualify_short_signal cascade."""
    agg = _agg(session_high=7400.0, session_low=7370.0,
               block_pdl_shorts=True)
    pattern = _short_pattern(level_price=7390.0, entry_price=7388.0,
                             stop_price=7395.0,
                             signal_type_for_pattern="velocity_short")
    signal = agg._qualify_short_signal(pattern, SignalType.VELOCITY_SHORT)
    assert signal is None


def test_pdl_short_block_allows_multi_hour_low():
    """MULTI_HOUR_LOW remains a valid short level — the gate only fires
    on PRIOR_DAY_LOW. Live data: BD shorts on MHL went 1W/1L (+$178).
    """
    agg = _agg(session_high=7400.0, session_low=7370.0,
               block_pdl_shorts=True)
    # Build a pattern with MULTI_HOUR_LOW level (not PDL)
    level = Level(
        price=7390.0,
        level_type=LevelType.MULTI_HOUR_LOW,
        created_at=_TS,
        confirmed_at=_TS,
        touch_count=3,
    )
    pattern = PatternSignal(
        pattern_type="breakdown_short",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=level,
        sweep_low=7387.0,
        entry_price=7388.0,
        stop_price=7395.0,
        bar_idx=200,
        timestamp=_TS,
        sweep_depth_pts=3.0,
        direction="short",
        sweep_high=7390.0,
    )
    signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
    assert signal is not None


def test_pdl_short_block_can_be_disabled():
    """block_pdl_shorts=False lets PDL shorts through (for backtest sweeps
    that want to measure baseline behavior or for diagnostic runs)."""
    agg = _agg(session_high=7400.0, session_low=7370.0,
               block_pdl_shorts=False)
    pattern = _short_pattern(level_price=7390.0, entry_price=7388.0,
                             stop_price=7395.0)
    assert pattern.level.level_type.name == "PRIOR_DAY_LOW"
    signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
    assert signal is not None
