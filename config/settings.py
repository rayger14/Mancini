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
    multi_hour_rally_min_pts: float = 25.0  # min rally from low to qualify (only significant levels)

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

    # Deep flush uses longer hold/timeout
    acceptance_min_hold_bars_deep: int = 4
    acceptance_timeout_bars_shallow: int = 15
    acceptance_timeout_bars_deep: int = 60

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
    deep_sell_rally_confirm_pts: float = 10.0  # lower rally threshold for crash lows

    # Signal cooldown: suppress repeated signals of the same type within N bars.
    # Feb 26 analysis: 97 signals in one session = noise. Mancini takes 1-3/day.
    # Cooldown prevents taking signal #2 when signal #1 from the same zone just lost.
    signal_cooldown_bars: int = 15          # min bars between signals of same type (sweep: 15=optimal)

    # Multi-day level memory: carry significant levels forward across sessions.
    # Mancini tracks levels that persist for days/weeks (e.g., Monday's session low
    # is still valid on Friday). Set to 0 to disable (current-day + prior-day only).
    level_memory_days: int = 5              # trading days to carry levels (5 = one week)
    max_persistent_levels: int = 10         # cap total persistent levels to prevent accumulation
    level_decay_rate: float = 0.85          # daily multiplicative decay for significance_score
    level_persist_min_score: float = 0.3    # drop persistent levels below this score
    level_persist_min_touches: int = 2      # min touch_count to qualify for persistence

    # Regime filter gating
    use_regime_filter: bool = False          # enable EMA regime direction gating
    regime_mode: str = "ema"                # "ema", "structure", "composite", "composite_strict"
    # Which patterns get regime-filtered. Empty tuple = all patterns gated.
    # Set to e.g. ("LEVEL_RECLAIM",) to only gate LR while FB/BD trade freely.
    regime_filter_patterns: tuple = ()      # empty = gate all; names from SignalType enum


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
