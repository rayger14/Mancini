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
    # FB-specific max hold time (bars). FB returns happen in first 20 min
    # or not at all. Sweep: 20 bars = best FB PF (1.53), +234 pts, 53% WR.
    fb_max_hold_bars: int = 20


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
    # rejected after achieving 7-10 hold bars. The downstream `fb_max_hold_bars=20`
    # still bounds the actual trade life, so loosening this is safe.
    acceptance_min_hold_bars_deep: int = 4
    acceptance_timeout_bars_shallow: int = 20
    acceptance_timeout_bars_deep: int = 60

    # FB level freshness gate (Mancini's "24-36 hours" rule).
    # Standard FBs require the swept low to be within `fb_max_level_age_hours`
    # of the signal timestamp. Older levels are "macro FBs" — Mancini says
    # they're rare and need elevated volatility to work. Setting to 0 disables
    # the gate (any age accepted).
    fb_max_level_age_hours: float = 36.0

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

    # Level reclaim
    level_reclaim_min_touches: int = 4  # S/R line touches required

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
    # Breakdown Short: support breaks and HOLDS broken → short the confirmed breakdown
    allow_breakdown_short: bool = False
    bd_min_break_depth_pts: float = 1.0     # min break below level to detect
    bd_confirm_bars: int = 15               # bars closing below to confirm (< FB's 20-bar abort)
    bd_timeout_bars: int = 40               # max wait for confirmation
    bd_stop_buffer_pts: float = 3.0         # stop above broken level
    bd_max_break_depth_pts: float = 15.0    # reject if already too far below (late entry)
    # BD SHORT level quality gate: only trigger off high-quality levels.
    # CLUSTER_LOW forms rapidly in consolidation and produces 97 noisy signals.
    # Mancini only shorts breakdowns of major levels (prior day low, multi-hour low).
    bd_require_major_level: bool = True     # if True, CLUSTER_LOW excluded from BD SHORT

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

    # BD Short conviction-based confirmation (replaces flat bd_confirm_bars count).
    # Mancini reads conviction: velocity, depth, candle character, follow-through.
    # Each bar accumulates a score; confirmation when score >= threshold AND bars >= floor.
    # With all weights=0.0, collapses to flat bar count (backward compatible).
    bd_conviction_threshold: float = 21.0       # score to confirm (21 = backward compat with bd_confirm_bars)
    bd_min_bars_floor: int = 5                  # absolute minimum bars below before confirming
    bd_conviction_depth_norm_pts: float = 10.0  # depth normalizer (10 pts below = max bonus)
    bd_conviction_depth_weight: float = 0.0     # max depth bonus per bar (0=disabled for safe default)
    bd_conviction_velocity_norm: float = 2.0    # velocity normalizer (2 pts/bar selling = max bonus)
    bd_conviction_velocity_weight: float = 0.0  # max velocity bonus per bar (0=disabled)
    bd_conviction_candle_weight: float = 0.0    # max candle character bonus per bar (0=disabled)
    bd_conviction_new_low_weight: float = 0.0   # bonus for making new low (0=disabled)

    # Backtest Short: broken resistance retested from below and fails → short
    allow_backtest_short: bool = False
    bt_breakout_confirm_bars: int = 5       # bars above resistance = breakout confirmed
    bt_pullback_min_pts: float = 3.0        # min pullback from breakout high
    bt_confirm_bars: int = 5                # bars below to confirm failed backtest
    bt_timeout_bars: int = 20               # max time to confirm rejection
    bt_stop_buffer_pts: float = 3.0         # stop above backtest high
    bt_reclaim_abort_bars: int = 3          # abort if price reclaims level
    bt_max_distance_from_level: float = 2.0 # max distance for backtest touch

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
    mode1_disable_fb_longs: bool = False    # option to completely disable FB longs on Mode 1 Red

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
    mancini_llm_setup_lqs_bonus: int = 15

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
