"""ES contract specifications, risk parameters, and session times."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time


@dataclass(frozen=True)
class ESContractSpec:
    """E-mini S&P 500 futures contract specifications."""

    symbol: str = "ES"
    tick_size: float = 0.25
    tick_value: float = 12.50  # USD per tick
    point_value: float = 50.0  # USD per point (4 ticks)
    margin_initial: float = 12_650.0  # Approximate initial margin
    margin_maintenance: float = 11_500.0
    exchange: str = "CME"

    def ticks_to_points(self, ticks: int) -> float:
        return ticks * self.tick_size

    def points_to_dollars(self, points: float, contracts: int = 1) -> float:
        return points * self.point_value * contracts


@dataclass(frozen=True)
class SessionTimes:
    """RTH and key intraday windows (US/Eastern)."""

    rth_open: time = field(default_factory=lambda: time(9, 30))
    rth_close: time = field(default_factory=lambda: time(16, 0))
    # Globex session
    globex_open: time = field(default_factory=lambda: time(18, 0))  # prior day
    globex_close: time = field(default_factory=lambda: time(17, 0))
    # Preferred trading windows
    morning_window_start: time = field(default_factory=lambda: time(9, 30))
    morning_window_end: time = field(default_factory=lambda: time(11, 0))
    afternoon_window_start: time = field(default_factory=lambda: time(15, 0))
    afternoon_window_end: time = field(default_factory=lambda: time(15, 55))
    # EOD flatten
    eod_flatten_time: time = field(default_factory=lambda: time(15, 55))
    # Dead zone (avoid) - only trade morning (9:30-11) and close (15:00-15:55)
    chop_zone_start: time = field(default_factory=lambda: time(13, 0))
    chop_zone_end: time = field(default_factory=lambda: time(15, 0))

    def in_preferred_window(self, t: time) -> bool:
        """Check if time is within a preferred trading window."""
        in_morning = self.morning_window_start <= t <= self.morning_window_end
        in_afternoon = self.afternoon_window_start <= t <= self.afternoon_window_end
        return in_morning or in_afternoon

    def in_chop_zone(self, t: time) -> bool:
        """Check if time is in the 11AM-2PM chop zone."""
        return self.chop_zone_start <= t <= self.chop_zone_end

    def in_rth(self, t: time) -> bool:
        """Check if time is within trading hours.

        Handles wrap-around for full session (e.g., 18:00 -> 17:00 crosses midnight).
        """
        if self.rth_open <= self.rth_close:
            # Normal range (e.g., 9:30 -> 16:00)
            return self.rth_open <= t <= self.rth_close
        else:
            # Wrap-around range (e.g., 18:00 -> 17:00)
            return t >= self.rth_open or t <= self.rth_close

    def past_eod_flatten(self, t: time) -> bool:
        """Check if time is past the EOD flatten deadline.

        For full session with wrap-around, flatten zone is between
        eod_flatten_time and rth_close (e.g., 16:50 -> 17:00).
        """
        if self.rth_open <= self.rth_close:
            # Normal range
            return t >= self.eod_flatten_time
        else:
            # Wrap-around: flatten zone is eod_flatten -> rth_close (before break)
            return self.eod_flatten_time <= t <= self.rth_close


@dataclass(frozen=True)
class ElevatorParams:
    """Parameters for Elevator Down detection."""

    min_velocity_pts_per_min: float = 0.75  # catch broader selloffs
    velocity_window_bars: int = 5  # 5-bar rolling window (1-min bars = 5 min)
    min_levels_broken: int = 2  # require breaking 2 support levels
    completion_velocity_ratio: float = 0.5  # velocity must drop to this fraction
    higher_low_lookback: int = 4  # bars to confirm higher low


@dataclass(frozen=True)
class ExitParams:
    """Exit management parameters (75/15/10 split).

    Mancini's actual method (from 500 Substack posts):
      T1: Lock 75% at first resistance level up (~7-9 pts median)
      Stop: Move to several pts UNDER breakeven (not at BE)
      T2: Lock 15% at second resistance level up (~16 pts median)
      Runner: Trail 10% under the PRIOR DAY'S RTH low (updated at EOD)
      Runner carries overnight/multi-day until prior day low is lost.
    """

    default_contracts: int = 4
    # Marketable-limit entry: bound adverse entry slippage to this many points
    # (priced through the signal so it fills like a market order, but never
    # worse). 0 disables (falls back to a plain market order). On the 12-min
    # delayed feed this caps bad fills when price has already moved.
    entry_slippage_cap_pts: float = 5.0
    t1_exit_fraction: float = 0.75  # Mancini: 75% at first target
    t2_exit_fraction: float = 0.15  # Mancini: 15% at second target
    runner_fraction: float = 0.10   # Mancini: 10% runner
    initial_stop_buffer_pts: float = 4.5  # matches fb_stop_buffer_pts
    # After T1, stop goes several pts UNDER breakeven to give room.
    # Mancini: "it will usually go several points under break-even"
    breakeven_buffer_pts: float = -3.0  # negative = below breakeven (longs)
    # Short-specific breakeven buffer: wider than longs because short runners
    # need room to survive post-T1 bounces. Entry + 8 pts = 6821.75 on a
    # 6813.75 entry, vs entry + 3 = 6816.75 which gets clipped by normal bounces.
    short_breakeven_buffer_pts: float = -8.0  # negative = above breakeven for shorts
    # Legacy field kept for backward compat with tests
    breakeven_buffer_ticks: int = 1
    # Intraday trailing only used before EOD prior-day-low trail kicks in.
    # This is a fallback for the backtest engine which doesn't have EOD hooks.
    trailing_stop_pts: float = 12.0
    trailing_tighten_thresholds: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (10.0, 3.0),
            (15.0, 2.0),
        ]
    )
    # Runner trail: buffer below the prior day's low for the overnight trail.
    # Mancini: "trail under the prior days low" — we add 1 pt buffer.
    runner_prior_day_low_buffer_pts: float = 1.0

    # --- Structure-based runner trail (Mancini 2025-05-14) ---
    # After T2, instead of trailing a fixed `trailing_stop_pts` distance, find
    # the most recent significant swing low (for long runners) and place the
    # stop a few points below it. Mancini: "I typically move my stop up, often
    # to below wherever structure is." Per his 2025-08-05 quote, when T2 is hit
    # he leaves a 10% runner and "initiates the trailing stop methodology".
    structure_trail_enabled: bool = True
    structure_trail_buffer_pts: float = 3.0      # pts below swing low for stop
    structure_trail_lookback_bars: int = 30      # how far back to look for swings
    structure_trail_swing_order: int = 3         # bars on each side for a local extremum

    # Multi-session runner hold (Mancini 2025-10-12): "I am still holding my
    # 10% long runner from the Tuesday noon 6754 Failed Breakdown." Mancini
    # carries the 10% runner across multiple sessions to catch trend moves,
    # only exiting when the runner's structural trail stop is taken out.
    #
    # When True, AFTER_T2 runners (the 10% slice — already past T1 *and* T2)
    # are NOT flattened at EOD. They persist across the Globex rollover, the
    # exit_manager continues to update the structural trail at each new EOD,
    # and the runner only flattens when its stop is hit or the safety cap
    # (multi_session_runner_max_days) trips.
    #
    # IMPORTANT: this ONLY applies AFTER T2 has fired. AFTER_T1 positions
    # still hold the 25% (15% + 10%) tranche which is too much overnight
    # exposure for Mancini's method — they still flatten at EOD.
    multi_session_runner: bool = True
    # Safety cap: force-flatten if the runner has been alive this many
    # sessions. Prevents a forgotten runner from bleeding indefinitely.
    multi_session_runner_max_days: int = 5

    # Master switch for EOD flatten. Default OFF — all positions hold across
    # the session boundary via their existing stops (initial / BE-3 / structure
    # trail). The original intraday EOD flatten was throwing away upside on
    # trades that hit T1 then got force-closed before T2.
    #
    # When False (default):
    #   - INITIAL / AFTER_T1 / AFTER_T2 positions all hold across EOD.
    #   - update_prior_day_low() is still called so the structural trail
    #     ratchets correctly when a position has reached AFTER_T1/AFTER_T2.
    #   - The multi_session_runner_max_days cap STILL applies as a safety
    #     net — the _runner_sessions_held counter is bumped at session
    #     rollover for ANY held position (not just AFTER_T2), and once it
    #     hits the cap, the next EOD force-flattens regardless of phase.
    #
    # When True:
    #   - Legacy behavior: INITIAL / AFTER_T1 flatten at EOD, AFTER_T2
    #     honors the multi_session_runner flag.
    eod_flatten_enabled: bool = False


@dataclass(frozen=True)
class StrategyParams:
    """Core strategy parameters."""

    # Significant low detection
    swing_low_order: int = 30  # argrelextrema order (30 bars = 30 min on 1-min)
    cluster_proximity_pts: float = 2.0  # points within which lows form a cluster
    cluster_min_touches: int = 5  # minimum touches for a cluster
    # Mancini: a "significant low" requires a 20+ pt bounce/rally from it
    # (criterion #2 in his 3-tier definition). Previously 25.0 — tightened
    # to match the exact Mancini number from Apr 15 2026 Substack (6983 FB).
    multi_hour_rally_min_pts: float = 20.0  # min rally from low to qualify (only significant levels)

    # Failed breakdown
    # Mancini: "2-11 points ideally" for sweep depth. 1 pt = "not ideal."
    # 8 ticks = 2.0 pts minimum sweep below level.
    sweep_min_ticks: int = 8  # minimum ticks below level (8 ticks = 2.0 pts)

    # Acceptance confirmation
    acceptance_max_dip_pts: float = 4.0  # max dip below level during acceptance (relaxed from 3.0)
    acceptance_min_hold_seconds: int = 60  # hold above level
    acceptance_min_hold_bars: int = 7  # strong confirmation (7 min above level)

    # Non-acceptance (fast market) confirmation
    non_acceptance_min_recovery_pts: float = 5.0
    non_acceptance_min_hold_seconds: int = 120
    non_acceptance_min_hold_bars: int = 3

    # Sweep depth classification
    shallow_flush_threshold_pts: float = 20.0  # < 20 pts = shallow, >= 20 = deep

    # Deep flush uses longer hold/timeout.
    # 5/13/2026 — shallow timeout raised 15→20 per FB near-miss audit:
    # 27 resolved near-misses at this gate had 63% WR, +6pt avg; many were
    # rejected after achieving 7-10 hold bars. Loosening this from 15→20
    # captures that cohort while the downstream prior-day-low runner trail
    # still bounds total trade duration.
    acceptance_min_hold_bars_deep: int = 4
    acceptance_timeout_bars_shallow: int = 20
    acceptance_timeout_bars_deep: int = 60

    # FB level freshness gate (Mancini's "24-36 hours" rule).
    # Standard FBs require the swept low to be within `fb_max_level_age_hours`
    # of the signal timestamp. Older levels are "macro FBs" — Mancini says
    # they're rare and need elevated volatility to work. Setting to 0 disables
    # the gate (any age accepted).
    fb_max_level_age_hours: float = 36.0
    # Macro-FB VIX override: when current VIX is above this, allow older levels.
    # Mancini explicitly: "when volatility hits, I not only get more Failed
    # Breakdowns, but I get bigger Failed Breakdowns" — macro FBs of week-old
    # lows producing 100-170pt rallies happen in this regime. Set to 0 to
    # disable the override (always enforce the 36h gate).
    fb_macro_vix_threshold: float = 20.0
    # Type-based exemption: structural multi-day levels (prior-day low,
    # multi-hour low, intraday shelves, and Mancini plan-named CUSTOM levels)
    # are not bound by the same age cap as engine-derived intraday clusters.
    # Mancini routinely holds multi-day runners off PDL / MHL shelves, and
    # his own plan levels can remain valid for the entire week. Off by
    # default; enable in a follow-up after backtest validation.
    fb_age_cap_exempt_high_quality_levels: bool = False

    # True breakdown abort: consecutive bars closing below level
    true_breakdown_abort_bars: int = 20

    # Stop buffer: how far below the level to place the stop
    fb_stop_buffer_pts: float = 4.5  # FB stop at level - this value
    lr_stop_buffer_pts: float = 4.5  # LR stop at level - this value

    # Max sweep depth: reject FB signals where the sweep went too far below the level.
    # Live data (1,651 events): deep sweeps (50+ pts) produce 117.6 pt avg recovery —
    # the biggest bounces come from the deepest sweeps. Default raised from 10 to 100.
    max_fb_sweep_depth_pts: float = 100.0
    # Deep sweep dual stop: for sweeps deeper than this threshold, use
    # level_price - fb_stop_buffer_pts as the stop instead of sweep_low - buffer.
    # This avoids massive 50+ pt stops on deep sweeps while still capturing
    # the bounce. Set to 0 to disable (always use sweep_low stop).
    deep_sweep_level_stop_threshold_pts: float = 20.0

    # Mancini-style position sizing based on stop distance.
    # Max stop for full size; beyond this, size down proportionally.
    # 15 pts = full, 30 pts = half, 50 pts = quarter.
    max_full_stop_pts: float = 15.0

    # Max target distance cap (points).
    # Mancini: "average FB return is 30-50 points (~70%), 4-15 points (~15%),
    # 70-600 points (~15%)." When first resistance is very far, cap the T1
    # target so the R:R filter doesn't pass trades with unrealistic targets.
    # Diagnostic: trades with R:R > 5 win only 8%. Sweet spot is 2.0-2.5 R:R.
    max_target_distance_pts: float = 30.0
    min_target_distance_pts: float = 8.0  # min distance to target (filter too-close levels)
    min_signal_rr: float = 0.5  # absolute R:R floor (reject garbage signals)

    # Level sweep FB (no elevator required for high-quality levels)
    # When price sweeps below a prior day low, multi-hour low, or cluster,
    # the level quality alone justifies the FB — no fast selloff needed.
    # Requires price to CLOSE below the level for min_bars_below before
    # recovery, ensuring it was a real break attempt, not just a wick.
    allow_level_sweep_fb: bool = True  # enable non-elevator FB path
    level_sweep_min_depth_pts: float = 1.0  # min sweep depth below level
    level_sweep_min_bars_below: int = 3  # min bars closing below before recovery counts

    # Candle bias filter: skip entries when prior candles show bearish bias.
    # Tested Feb 2026: hurts more than helps (-200 to -300 pts) because
    # FB/LR entries naturally follow bearish candles. Disabled by default.
    candle_bias_filter: bool = False
    candle_bias_min_range_pts: float = 2.0  # min bar range to judge bias
    candle_bias_bearish_threshold: float = 0.3  # close in bottom 30% = bearish

    # Level-resume filter (2026-07-06, default OFF): before an FB long fires on
    # an ENGINE auto-detected level, require the level's resume to show a proven
    # rally of at least this many points ("proven launcher"). Mancini's levels
    # (CUSTOM / mancini_confirmed) are ALWAYS exempt — this only polices the
    # levels our engine invents, which trade 51% WR / +4avg vs his 77% / +28.
    # Validated on 59 real auto-level trades: proven-launcher levels made
    # +345pts, weak ones lost -111; his traded levels' median resume = 53pt.
    # 0.0 = off.
    fb_auto_level_min_rally_pts: float = 0.0
    # News-reaction entry blackout (2026-07-14, default OFF): calendar-free.
    # When a scheduled-release minute (8:30/10:00/14:00 ET) prints a bar with
    # range >= news_bar_range_pts, block NEW entries for news_blackout_minutes.
    # Exits/stops/runners unaffected (riding runners through data IS the
    # Mancini edge: "sit back, hold runners, and wait"). Study 2026-07-14:
    # all 10 entries taken on data mornings netted -234pts vs +3597 across
    # 110 normal-day entries; trade 732 chased the 55pt CPI bar and lost -65.
    # 0.0 = off.
    news_bar_range_pts: float = 0.0
    news_blackout_minutes: int = 30

    # Level reclaim
    level_reclaim_min_touches: int = 4  # S/R line touches required
    # Master switch: disable LR entirely (FB-only mode). LR has historically
    # had ~18% WR and may be the dominant bleed source; disabling lets us
    # isolate FB longs cleanly in backtest sweeps and live operation.
    allow_level_reclaim: bool = True

    # Legacy short-side parameters (Failed Rally + Level Rejection) — DEPRECATED
    # Use allow_breakdown_short / allow_backtest_short instead.
    allow_short_fr: bool = False
    allow_short_lj: bool = False
    fr_stop_buffer_pts: float = 4.5
    lj_stop_buffer_pts: float = 4.5
    short_acceptance_max_dip_pts: float = 4.0
    short_acceptance_min_hold_bars: int = 7
    short_acceptance_min_hold_bars_deep: int = 4
    short_true_rally_abort_bars: int = 20
    short_swing_high_order: int = 15
    short_max_fr_sweep_depth_pts: float = 10.0
    short_candle_bias_filter: bool = True
    short_candle_bias_min_range_pts: float = 2.0
    short_candle_bias_bullish_threshold: float = 0.7

    # --- Mancini-faithful short patterns (v2) ---
    # Breakdown Short (Mancini-faithful Phase 3 rewrite). Per his 2025-08-24
    # post: shorts the FAILURE of a Failed Breakdown long. Pattern triggers
    # only when (1) an obvious support shelf exists, (2) price loses the
    # shelf, recovers it, and rallies (an FB attempt), then (3) price comes
    # BACK DOWN and breaks the lowest low of that FB. Short trigger is a
    # few pts below the broken FB-low. See core.patterns_short_v2.BreakdownShort.
    allow_breakdown_short: bool = False
    bd_shelf_min_touches: int = 3           # CLUSTER_LOW/HORIZONTAL_SR shelf strength gate
    bd_shelf_expire_bars: int = 480         # forget tracked shelves after ~8h (1-min bars)
    bd_min_flush_depth_pts: float = 3.0     # min pts price must drop below shelf to count as "real" FB attempt
    bd_max_flush_bars: int = 30             # max bars below shelf before declaring "not an FB" (just a trend leg)
    bd_fb_fail_buffer_pts: float = 3.0      # pts below flush_low for the short trigger (Mancini: "few point buffer")
    bd_fb_success_rally_pts: float = 20.0   # rally pts after recovery that classify the FB as having "succeeded" → abandon
    bd_fb_success_timeout_bars: int = 60    # how long after recovery to wait before declaring success
    bd_recovery_watch_bars: int = 120       # max time after recovery to keep watching for an FB failure
    bd_stop_buffer_pts: float = 3.0         # pts above the recovery_high for the stop
    # Note: older bd_* params (bd_confirm_bars, bd_max_break_depth_pts,
    # bd_timeout_bars, bd_min_break_depth_pts, bd_require_major_level,
    # bd_conviction_*, bd_min_bars_floor) are no longer used — the new
    # state machine doesn't have the concept of "consecutive bars below"
    # or "conviction score". Kept as no-op placeholders below to avoid
    # crashing scripts that set them explicitly.
    bd_min_break_depth_pts: float = 1.0     # DEPRECATED — no-op
    bd_confirm_bars: int = 15               # DEPRECATED — no-op
    bd_timeout_bars: int = 40               # DEPRECATED — no-op
    bd_max_break_depth_pts: float = 15.0    # DEPRECATED — no-op
    bd_require_major_level: bool = True     # DEPRECATED — no-op (new BD uses _SHELF_TYPES)

    # Velocity Breakdown Short: single-bar news-driven breakdown
    # Catches moves where a major level breaks on one 4000+ volume bar
    # with 5x average volume — too fast for the multi-bar BD detector.
    allow_velocity_short: bool = False       # enable single-bar velocity breakdown
    vbd_min_break_pts: float = 8.0          # minimum break below level in one bar
    # Upper cap on a velocity break: a 30-49pt one-bar print isn't a velocity
    # breakdown, it's the back half of a crash. Empirically every BD/VBD trade
    # in the 5/2026 sample with sweep_depth > 20pts lost. Mirrors the existing
    # bd_max_break_depth_pts=15.0 for the multi-bar BD detector.
    vbd_max_break_pts: float = 20.0
    vbd_min_volume_ratio: float = 3.0       # bar volume >= 3x 20-bar average
    vbd_require_close_below: bool = True    # bar must CLOSE below the level
    vbd_stop_buffer_pts: float = 3.0        # stop above the broken level + buffer
    vbd_position_size_factor: float = 0.25  # 25% size (aggressive signal, small position)
    vbd_only_major_levels: bool = True      # only at PDL, not minor levels

    # Capitulation-entry guard for short signals (applies to BREAKDOWN_SHORT
    # and VELOCITY_SHORT). Rejects shorts that fire when price has already
    # crashed to the session low — the bot was fading the flush instead of
    # the breakdown. Per the 5/2026 short-side post-mortem: 5 of 9 production
    # BD losers and 5 of 7 velocity losers had entry within 10pts of the
    # session low while the session_high was 25+pts above entry. The existing
    # move_exhaustion gate (session_low <= T1) is too lenient because T1 sits
    # 20-30pts BELOW entry, so it stays silent on the exact entries that lose.
    block_capitulation_shorts: bool = True
    short_capitulation_floor_pts: float = 10.0   # max (entry - session_low)
    short_capitulation_off_high_pts: float = 25.0  # min (session_high - entry)

    # Mancini-aligned short size de-rating. Per Mancini 2026-02-04: shorts
    # have "substantially lower win rate and lower R/R" — risk-management
    # response is smaller size, not "tune harder until shorts WR matches
    # longs". Applied multiplicatively to size_factor in _qualify_short_signal
    # AFTER stop-distance and sweep-depth sizing. 1.0 (default) disables;
    # 0.5 is half-size; 0.25 is quarter-size. Default left at 1.0 because
    # the 5y backtest's max_daily_loss_pts=20 cap interacts with halved
    # short losses by allowing more same-day chop-zone longs to fire (net
    # -$6,844 in backtest, mostly artifact). Live PRODUCTION_RISK has
    # max_daily_loss_pts=9999 so that dynamic doesn't apply — flip to 0.5
    # in production once we have evidence the live impact is positive.
    short_size_factor: float = 1.0

    # Mancini-aligned PDL short block. Per Mancini 2026-02-17: "A Failed
    # Breakdown requires price to flush and recover a significant low.
    # A significant low can be 1 of 3 things: 1.) The prior days low
    # 2.) A multi-hour low/a low that goes 20+ points or 3.) A cluster
    # or shelf of lows." Shorting PRIOR_DAY_LOW means positioning against
    # his core long setup at the level where it triggers. Live data
    # 2026-02-25 → 2026-05-12: 5/5 PDL shorts lost ($-813). Block them.
    # MULTI_HOUR_LOW is also a Mancini-significant-low per his definition,
    # but is the only short bucket with positive live signal (1/2 W,
    # $+178) — kept open pending more samples.
    block_pdl_shorts: bool = True
    # Alert-only shorts: route every short to the Discord alert/shadow path and
    # place NO live short order. The P&L-at-targets report showed the existing
    # short detectors are net-negative (13/14 lose at targets, 0/14 hit T1).
    # Default off so backtests still simulate shorts; ON in PRODUCTION_STRATEGY.
    shorts_alert_only: bool = False

    # DEPRECATED — conviction-scoring scaffold for the old "consecutive close-below"
    # BD detector. The Phase 3 rewrite no longer uses these. Kept as no-op
    # placeholders so PRODUCTION_STRATEGY / external scripts setting them
    # don't crash. Safe to delete after one release cycle.
    bd_conviction_threshold: float = 21.0
    bd_min_bars_floor: int = 5
    bd_conviction_depth_norm_pts: float = 10.0
    bd_conviction_depth_weight: float = 0.0
    bd_conviction_velocity_norm: float = 2.0
    bd_conviction_velocity_weight: float = 0.0
    bd_conviction_candle_weight: float = 0.0
    bd_conviction_new_low_weight: float = 0.0

    # Back-Test Short: Mancini-faithful pattern. Per Mancini 2024-10-09:
    # "Price must have set a clearly defined support [shelf]. Price must
    # break down that support decisively — forceful, deep, lasting hours+.
    # Price must back-test the level from below. The FIRST retest from
    # below is typically actionable; odds drop with each successive test."
    # See core.patterns_short_v2.BacktestShort for the implementation.
    allow_backtest_short: bool = False
    bts_support_min_touches: int = 3        # min touches for HORIZONTAL_SR/CLUSTER_LOW shelf eligibility
    bts_breakdown_confirm_bars: int = 5     # consecutive close-below bars to register breakdown
    bts_min_flush_depth_pts: float = 15.0   # "deep" flush — pts price must drop below shelf
    bts_max_distance_from_level: float = 2.0  # max distance for the retest touch
    bts_confirm_bars: int = 3               # bars closing below after touch to confirm rejection
    bts_reclaim_abort_bars: int = 3         # close-above bars that abort (level reclaimed)
    bts_timeout_bars: int = 30              # give up waiting for rejection after this many bars
    bts_stop_buffer_pts: float = 3.0        # stop above the back-test wick high
    bts_first_touch_only: bool = True       # Mancini: first retest is best, odds drop
    bts_breakout_expire_bars: int = 200     # forget broken supports after this many bars

    # Legacy bt_* params (deprecated — old "broken resistance" semantics).
    # Kept temporarily so any external script setting them doesn't crash.
    # Remove after one release cycle if nothing else references them.
    bt_breakout_confirm_bars: int = 5
    bt_pullback_min_pts: float = 3.0
    bt_confirm_bars: int = 5
    bt_timeout_bars: int = 20
    bt_stop_buffer_pts: float = 3.0
    bt_reclaim_abort_bars: int = 3
    bt_max_distance_from_level: float = 2.0

    # Deep sell recovery: detect intraday levels during massive selloffs.
    # Mancini: "The bigger the sell, the bigger the squeeze." After deep sells,
    # he FBs at NEW levels formed during the selloff, not the original broken level.
    # When price is >30 pts below nearest support, use faster swing detection
    # (order=5 instead of 30) to catch crash bottoms and consolidation zones.
    allow_deep_sell_recovery: bool = False   # +148 pts on 5yr (+1 trade) but disabled pending more validation
    deep_sell_threshold_pts: float = 30.0   # below nearest support = deep sell mode
    deep_sell_swing_order: int = 5          # faster swing confirmation (5 bars vs 30)
    deep_sell_rally_confirm_pts: float = 20.0  # Mancini: significant low = 20+ pt bounce (V-shaped reversal)
    # Path 2 retroactive FB requires a real drop into the level — without this
    # gate, Path 2 fires on any INTRADAY_LOW where current close happens to be
    # above. Min drop = (session high since FB-detector init) - (level price).
    deep_sell_min_drop_pts: float = 15.0

    # Signal cooldown: suppress repeated signals of the same type within N bars.
    # Feb 26 analysis: 97 signals in one session = noise. Mancini takes 1-3/day.
    # Cooldown prevents taking signal #2 when signal #1 from the same zone just lost.
    signal_cooldown_bars: int = 15          # min bars between signals of same type (sweep: 15=optimal)

    # --- Data-driven gates (from live trade_lessons.md analysis, Mar 2026) ---

    # Level reuse: second trade at same level same session = 0W/4L, avg -18.6 pts.
    max_trades_per_level: int = 1           # max signals per level per session (0=disabled)

    # BD Short R:R floor: BD Shorts at R:R 1.0-1.5 had 14% WR. Need higher floor.
    bd_short_min_rr: float = 1.5            # pattern-specific minimum R:R for BD Short

    # Session range minimum: trades at <20 pt session range = negative expectancy.
    min_session_range_pts: float = 15.0     # min session H-L before allowing signals
    min_session_range_grace_bars: int = 30  # grace period at session start (range builds)

    # Cross-type cooldown: after ANY signal at a level, no other signal type can fire
    # at that level for N bars. Prevents BD Short → FB Long whipsaw on same level.
    cross_type_level_cooldown_bars: int = 30  # bars before same level can produce another signal

    # BD Short max entry distance: reject if entry is too far below broken level.
    # Mar 18: BD Short entered 10.5 pts below level, move was already exhausted.
    bd_max_entry_distance_pts: float = 10.0   # max distance from level at confirmation

    # --- Intraday Price Action Context (Mancini-faithful trend detection) ---
    # Reads intraday direction from swing structure, bounce quality, and session position.
    # POST_SELL_SETUP (after elevator) always overrides — never suppress FB Longs after a flush.
    use_intraday_context: bool = False      # master switch (default off)
    idc_swing_order: int = 5               # bars to confirm swing H/L (fast detection)
    idc_min_swing_pts: float = 3.0         # min swing size (noise filter)
    idc_weak_bounce_pts: float = 5.0       # avg bounce below this = weak recovery
    idc_bounce_lookback: int = 3           # how many recent bounces to average
    idc_elevator_recency_bars: int = 30    # POST_SELL_SETUP duration after elevator completes
    idc_session_pos_bearish: float = 0.2   # session position below this = bearish vote
    idc_session_pos_bullish: float = 0.8   # session position above this = bullish vote
    idc_bearish_threshold: int = 3         # votes needed for BEARISH_PRESSURE
    idc_bullish_threshold: int = 3         # votes needed for BULLISH_PRESSURE

    # Multi-day level memory: carry significant levels forward across sessions.
    # Mancini tracks levels that persist for days/weeks (e.g., Monday's session low
    # is still valid on Friday). Set to 0 to disable (current-day + prior-day only).
    level_memory_days: int = 5              # trading days to carry levels (5 = one week)
    max_persistent_levels: int = 10         # cap total persistent levels to prevent accumulation
    level_decay_rate: float = 0.85          # daily multiplicative decay for significance_score
    level_persist_min_score: float = 0.3    # drop persistent levels below this score
    level_persist_min_touches: int = 2      # min touch_count to qualify for persistence

    # Mancini exit scaling: toggle between current 50/50 split and Mancini 75/15/10.
    # When enabled, ExitManager overrides ExitParams fractions with these values,
    # and _qualify_signal uses level-based T1 instead of fixed distance.
    mancini_exit_scaling: bool = False       # use Mancini 75/15/10 scaling vs current 50/50
    mancini_t1_at_first_resistance: bool = False  # T1 at first resistance level vs fixed distance
    mancini_t1_exit_pct: float = 0.75       # 75% off at T1 (Mancini standard)
    mancini_t2_exit_pct: float = 0.15       # 15% off at T2
    mancini_runner_pct: float = 0.10        # 10% runner
    mancini_t1_min_distance_pts: float = 8.0  # min distance from entry to first resistance for T1

    # Sweep depth position sizing: "the bigger the sell, the bigger the squeeze"
    # Scale size proportionally to how far price swept below the level.
    # Deep sweeps (30+ pts) produce +264 pts on 5yr backtest; shallow sweeps are noise.
    use_sweep_depth_sizing: bool = False     # scale size by sweep depth (overrides stop-distance sizing)
    sweep_depth_min_pts: float = 2.0        # minimum sweep to qualify (below this = quarter size)
    sweep_depth_full_size_pts: float = 8.0  # sweep >= this = full size
    sweep_depth_quarter_size_pts: float = 2.0  # sweep this shallow = 25% size

    # Mode 1 trend day detection
    # Mancini: 90% of days are Mode 2 (range/chop, FBs work), 10% are Mode 1
    # (open-to-close trend). Mode 1 Red days destroy FB longs.
    use_mode1_detection: bool = False       # enable Mode 1 detector
    mode1_levels_broken_threshold: int = 3  # 3+ levels broken without recovery = Mode 1
    mode1_min_bars_below_pdl: int = 30      # 30+ bars below PDL = strong Mode 1 signal
    mode1_bearish_pressure_bars: int = 60   # 60+ bars of bearish pressure = Mode 1 signal
    mode1_level_broken_hold_bars: int = 20  # level must stay broken for 20+ bars to count
    mode1_size_reduction: float = 0.25      # reduce to 25% size on Mode 1 Red days
    mode1_disable_fb_longs: bool = False    # legacy: completely disable FB longs on Mode 1 Red

    # Mancini's Mode 1 Red rule (May 8 2025 post): "On Mode 1 red days,
    # there often won't be a failed breakdown until before the close or
    # in the early evening and one just has to wait patiently all day for
    # the sell to complete." Block FB longs while Mode 1 Red is active
    # AND we're earlier than this hour ET. After the cutoff, FBs are
    # allowed so the late-day exhaustion reversal can fire.
    mode1_red_fb_long_block_until_hour: int = 15  # 3pm ET
    mode1_red_fb_long_block_until_minute: int = 0

    # Mode 1 Green trend day detection (mirror of Mode 1 Red, UP direction)
    # Mancini Apr 15 2026: on trend-up days FBs can fire near significant lows
    # with relaxed gates — the trend is the edge, not level R:R.
    # Still requires acceptance OR non-acceptance (rules DON'T change on trend days).
    use_mode1_green_detection: bool = False  # master switch (default off / shadow-first)
    mode1_green_resistance_broken_threshold: int = 3  # 3+ resistance levels broken UP and held
    mode1_green_bars_above_pdh: int = 30     # bars continuously above PDH
    mode1_green_bullish_pressure_bars: int = 60  # sustained higher highs for N bars
    mode1_green_level_broken_hold_bars: int = 20  # bars price must hold above a broken resistance
    mode1_green_fb_min_rr: float = 1.0       # relaxed R:R on confirmed Mode 1 Green
    mode1_green_size_factor: float = 1.0     # full size on confirmed trend days
    # Data-backed tells (5y study 2026-06-10: 177 green days of 1292 sessions).
    # Shallow-fast dips: green median 6/day vs 1 on normal days.
    mode1_green_shallow_dip_max_pts: float = 8.0   # dip depth cap to count as shallow
    mode1_green_shallow_dip_max_bars: int = 20     # recovery speed cap (bars)
    mode1_green_shallow_dips_min: int = 5          # dips needed (5y replay: 5 → precision 0.29 / recall 0.77)
    # Breakdown squeeze (counter-trade fails): 17% of green days vs 7% normal.
    mode1_green_squeeze_min: int = 1               # squeezes needed for the condition

    # Danger zone enforcement (Mancini: 0-5 pts above level = danger zone)
    # Entries in the danger zone require clear acceptance (dip-back pattern).
    # If recovery >= danger_zone_pts, non-acceptance protocol applies normally.
    danger_zone_pts: float = 5.0
    danger_zone_require_dip_acceptance: bool = True  # require a dip-back touch when recovery < 5 pts
    danger_zone_dip_proximity_pts: float = 2.0       # dip within this pts of level counts as acceptance touch

    # Risky trend-day FB flag: FB fired within N pts of session high on an
    # up-trending session. Mancini: "FBs not far off major highs after big
    # rally are dangerous — tend to fakeout unless parabolic rally sustains."
    risky_trend_fb_distance_from_high_pts: float = 30.0

    # Level confluence scoring
    # When enabled, entries require a minimum confluence score based on level
    # quality (PDL=5, MHL=3, etc.), proximity to other levels, touch count,
    # and rally size. Live data: PDL=100% WR, MHL=67%, CLUSTER_LOW=0%.
    use_confluence_scoring: bool = False     # gate entries by confluence score
    confluence_min_score: int = 2           # minimum score to take a trade
    confluence_proximity_pts: float = 3.0   # how close levels must be to count as confluent

    # ATM level tracking: levels that produce profits every time they're tested.
    # Mancini: "ATM machine levels" — multi-day, multi-touch shelves where
    # every FB produces a 30+ pt rally. Track per-level profitability and
    # boost size when a level qualifies.
    use_atm_level_boost: bool = False       # boost size at ATM levels
    atm_min_winning_trades: int = 2         # min wins at this level to qualify
    atm_min_win_rate: float = 0.6           # 60% WR at this level
    atm_size_boost: float = 1.5             # 150% size at ATM levels (up from 100%)

    # Double Dip re-entry after stop-out
    allow_double_dip: bool = True
    dd_cooldown_bars: int = 120              # max bars after stop-out to allow re-entry
    dd_min_depth_below_stop_pts: float = 5.0 # new sweep must go this far below the original stop
    dd_bypass_level_gate: bool = True        # bypass max_trades_per_level for double dips
    dd_bypass_cooldown: bool = True          # bypass signal_cooldown_bars for double dips
    dd_position_size_factor: float = 0.5     # sizing for DD re-entries (half size)
    dd_fixed_stop_until_t1: bool = True      # no trailing until T1 hits (keep stop fixed at entry)
    dd_trail_pts_after_t1: float = 25.0      # wider trail for DD runners (25 vs 12 pts)

    # Level Quality Scoring (LQS): continuous 0-100 score per level that drives
    # position size, R:R requirements, and FB eligibility. Shadow first, then enable.
    use_level_quality_scoring: bool = False   # master switch (shadow first)
    lqs_min_trade_threshold: int = 25         # min LQS to trade (PDL=53, MHL=45, SWING=15)
    lqs_full_size_threshold: int = 55         # full size (PDL+recency=53+, Mancini-confirmed=60+)
    lqs_shadow_threshold: int = 10            # below = skip (CLUSTER=0-10, bare SWING=5-15)
    # Drop HORIZONTAL_SR as an ENTRY source: forces its LQS to 0 so both entry
    # paths skip it, while the level survives for trailing-stop structure.
    # Live + 5y backtest both show HORIZONTAL_SR (the bot's own touch-counted
    # flat lines) is a net loser vs Mancini's hand-called CUSTOM levels.
    block_horizontal_sr_entries: bool = False  # backtest-gated; see PR

    # Conviction sizing fix: APPLY sweep-depth sizing (deep flush = bigger
    # squeeze = bigger size) instead of letting stop distance shrink it. A deep
    # sweep produces a wide stop, and the stop-distance rule sizes those to the
    # minimum — exactly backwards for Mancini's best setups. When True, size is
    # max(stop_distance_factor, sweep_depth_factor): it only ever sizes UP on a
    # deep flush, never down. Independent of shadow_mode_features.
    apply_sweep_depth_sizing: bool = False

    # Evidence-based conviction sizing (replaces stop-width as the size driver).
    # Conviction study on 100 live trades: stop-width is an ANTI-tell (33% WR);
    # the real edge is a deep crash-bottom flush that reclaims a QUALITY level
    # (INTRADAY_LOW / Mancini CUSTOM) — together 85% WR. Score:
    #   +2 deep flush (sweep >= conviction_deep_flush_pts)
    #   +2 quality level (INTRADAY_LOW, CUSTOM, MANCINI_LEVEL)
    #   +1 Failed Breakdown
    # score>=4 (deep AND quality) -> full size; 2-3 -> half; else stop-distance.
    # Does NOT reward tight stops or high R:R (both anti-tells). Flag-gated.
    use_conviction_sizing: bool = False
    conviction_deep_flush_pts: float = 25.0   # sweep depth that counts as "deep"

    # Shadow mode: features log what they WOULD do but don't change trading decisions.
    # When True, sweep depth sizing, Mode 1 detection, and velocity short all run
    # but only produce shadow log entries — actual sizing/gating/signals are unchanged.
    shadow_mode_features: bool = False

    # 5-minute timeframe for level detection (Mancini uses 5-min charts)
    use_5min_levels: bool = False             # enable 5-min bar level detection
    swing_low_order_5min: int = 6             # Mancini's swing order on 5-min (6 bars = 30 min)
    level_detection_timeframe_min: int = 5    # aggregation period

    # Shelf of lows detection (Mancini's multi-touch horizontal base)
    detect_shelf_levels: bool = False          # enable shelf detection on 5-min
    shelf_min_touches: int = 8               # real Mancini shelves have 8+ touches on 5-min
    shelf_proximity_pts: float = 3.0         # max range of the shelf (tight base)
    shelf_min_bars: int = 12                 # minimum 5-min bars the shelf spans (1 hour)
    shelf_sweep_min_pts: float = 2.0         # need 2+ pts below shelf to qualify

    # Regime filter gating
    use_regime_filter: bool = False          # enable EMA regime direction gating
    regime_mode: str = "ema"                # "ema", "structure", "composite", "composite_strict"
    # Which patterns get regime-filtered. Empty tuple = all patterns gated.
    # Set to e.g. ("LEVEL_RECLAIM",) to only gate LR while FB/BD trade freely.
    regime_filter_patterns: tuple = ()      # empty = gate all; names from SignalType enum

    # --- Mancini Substack level overlay ---
    # Augments engine-derived levels with levels Mancini calls out in his post.
    # Ships OFF by default; defaults to shadow mode when enabled.
    use_mancini_levels: bool = False          # master switch (off by default)
    mancini_mode: str = "shadow"              # "shadow" | "confirmation" | "augmentation"
    mancini_levels_dir: str = "/app/data"
    mancini_confirm_tolerance_pts: float = 3.0
    mancini_min_conviction_for_trade: int = 2

    # --- Mancini LLM-extracted plan (Phase 3) ---
    # When True, the engine loads the structured daily plan written by
    # live/mancini_llm_extract.py to /app/data/mancini_plan_<date>.json
    # and applies its mode/danger_zones/no_trade_zones gates and the
    # planned_setups LQS boost. Plan extraction itself runs nightly via
    # cron regardless — this only controls whether the engine consumes
    # the output. Ships OFF; flip to True after several days of shadow
    # validation against the JSON output.
    use_mancini_llm_plan: bool = False
    mancini_llm_plan_dir: str = "/app/data"
    mancini_llm_setup_match_tolerance_pts: float = 2.0
    # Inject Mancini's published target ladder (plan.targets) into the level
    # store as MANCINI_LEVEL targets, and bump source_count where they coincide
    # with an engine level (confluence). Shadow-first: validate by backtest
    # before enabling live.
    use_mancini_targets: bool = False
    mancini_target_confluence_tol_pts: float = 3.0
    # Floor-trader pivots (PP/R1-3/S1-3) from prior-day H/L/C as a third
    # confluence source. Weak alone; a pivot near an engine level bumps that
    # level's source_count. Shadow-first: validate by backtest before live.
    use_pivot_levels: bool = False
    pivot_confluence_tol_pts: float = 3.0
    # Continuous confluence: recompute a level's source_count from the live
    # store at scoring time (catches intraday convergence the one-shot
    # injection-time match misses). Drives the LQS confluence bonus.
    use_source_confluence: bool = False
    source_confluence_tol_pts: float = 3.0
    mancini_llm_setup_lqs_bonus: int = 15
    # CLUSTER_LOW quality filter (5y leak audit, 2026-06-08):
    # 98% of acceptance-protocol FB longs fire on engine-derived CLUSTER_LOW
    # levels, producing -$94K losses over 5 years. Mancini explicitly warns
    # "mid-range entries are noisy" — CLUSTER_LOW IS the noisy mid-range
    # cluster. When True, an FB-long or LR-long whose underlying
    # pattern.level.level_type is CLUSTER_LOW may only fire if it also
    # matches a Mancini plan setup price within
    # mancini_llm_setup_match_tolerance_pts AND direction is long. No plan
    # loaded => treated as no match => reject. Other level types unaffected.
    # Off by default; enable in PRODUCTION_STRATEGY only after backtest.
    cluster_low_requires_plan_match: bool = False
    # Mancini's verbatim danger-zone rule:
    #   "5 pts above swept low is danger zone; use non-acceptance protocol
    #    or wait for clear acceptance."
    # When True, signals that the engine's pattern detector qualified via
    # the NON_ACCEPTANCE confirmation path are allowed through the danger
    # zone gate (matching Mancini's explicit carve-out). Acceptance-protocol
    # signals are still blocked by the gate when in a danger zone — the
    # 5y leak analysis shows acceptance-protocol FB longs lose money and
    # the gate is correct for that path.
    danger_zone_allow_non_acceptance: bool = False

    # --- Daily Structure Detector ---
    # Macro bias from daily chart: detects daily FB (bullish) or BD (bearish)
    # to suppress contra-trend trades and boost with-trend trades via LQS.
    use_daily_structure: bool = True
    daily_shelf_lookback_days: int = 20
    daily_shelf_cluster_pts: float = 30.0   # daily lows within 30 pts = shelf
    daily_shelf_min_touches: int = 3        # 3+ daily bars touching the shelf
    daily_fb_recovery_bars: int = 3         # 3+ daily closes above shelf = confirmed
    daily_fb_lqs_bonus: int = 10            # LQS boost for intraday longs during daily FB
    daily_bd_short_min_lqs: int = 70        # minimum LQS for shorts during daily FB bull


@dataclass(frozen=True)
class RiskParams:
    """Risk management parameters."""

    max_trades_per_day: int = 3
    max_daily_loss_pts: float = 20.0  # per-contract loss limit in points
    max_stop_distance_pts: float = 15.0  # live data shows BD Shorts need 10-15 pt stops to capture winners
    max_position_contracts: int = 4
    # Min prior-day range (high-low) to allow entries. Set to 0 to disable.
    # Testing showed prior-day range is a poor proxy for current conditions.
    min_prior_day_range_pts: float = 0.0
    # Skip Tuesdays: deprecated — use regime filter instead of day skipping.
    skip_tuesdays: bool = False
    never_let_green_go_red: bool = True
    # After first win: only risk profits from first trade
    risk_only_profits_after_first_win: bool = True
    # Minimum R:R to accept a trade (used by RiskManager as final gate).
    # Set low (e.g. 0.1) for data collection / bypass mode.
    min_rr_ratio: float = 1.0


# Default instances for easy import
DEFAULT_CONTRACT = ESContractSpec()
MES_CONTRACT = ESContractSpec(
    symbol="MES",
    tick_size=0.25,
    tick_value=1.25,
    point_value=5.0,
    margin_initial=1_265.0,
    margin_maintenance=1_150.0,
    exchange="CME",
)
DEFAULT_SESSION = SessionTimes()
DEFAULT_ELEVATOR = ElevatorParams()
DEFAULT_EXIT = ExitParams()
DEFAULT_STRATEGY = StrategyParams()
DEFAULT_RISK = RiskParams()
