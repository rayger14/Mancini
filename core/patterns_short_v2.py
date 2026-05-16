"""Mancini-faithful short patterns: Breakdown Short & Backtest Short.

Breakdown Short:
  Support shelf breaks and HOLDS broken → short the confirmed breakdown.
  This is the INVERSE of Failed Breakdown (FB watches break then recover;
  BD watches break and STAY broken).

Backtest Short:
  Previously broken resistance retested from below and fails → short.
  Entry at failed retest, stop above backtest high.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams, DEFAULT_STRATEGY
from core.patterns import PatternSignal, ConfirmationType


class ShortState(Enum):
    """State machine states for short pattern detection."""

    IDLE = auto()
    BREAK_DETECTED = auto()       # Price broke below support
    HOLDING_BELOW = auto()        # Counting bars below
    BREAKOUT_TRACKED = auto()     # (legacy)
    PULLBACK_DETECTED = auto()    # (legacy)
    BACKTEST_WATCH = auto()       # Watching for failed backtest (broken support retested from below)
    CONFIRMED = auto()


class BreakdownShort:
    """Detect true breakdowns of support for short entries.

    When a significant support level (PRIOR_DAY_LOW, MULTI_HOUR_LOW, CLUSTER_LOW)
    breaks and price stays below for bd_confirm_bars consecutive bars, emit
    a short signal.

    This is the INVERSE of what FailedBreakdown detects. When FB's
    true_breakdown_abort triggers (price stays below too long), that IS
    a Breakdown Short confirmation.

    State machine: IDLE → BREAK_DETECTED → HOLDING_BELOW → CONFIRMED → Signal
    """

    _SUPPORT_TYPES = frozenset({
        LevelType.PRIOR_DAY_LOW,
        LevelType.MULTI_HOUR_LOW,
        LevelType.CLUSTER_LOW,
    })
    # Major levels only — cluster lows form rapidly in consolidation and produce noise.
    _MAJOR_SUPPORT_TYPES = frozenset({
        LevelType.PRIOR_DAY_LOW,
        LevelType.MULTI_HOUR_LOW,
    })

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params
        self.state = ShortState.IDLE
        self._target_level: Optional[Level] = None
        self._break_bar: int = -1
        self._bars_below: int = 0
        self._lowest_low: float = float("inf")
        self._break_close: float = 0.0
        # Conviction scoring state
        self._conviction_score: float = 0.0
        self._prev_low_below: float = float("inf")

    def reset(self) -> None:
        self.state = ShortState.IDLE
        self._target_level = None
        self._break_bar = -1
        self._bars_below = 0
        self._lowest_low = float("inf")
        self._break_close = 0.0
        self._conviction_score = 0.0
        self._prev_low_below = float("inf")

    def update(
        self,
        bar_idx: int,
        timestamp: datetime,
        open_: float = 0.0,
        high: float = 0.0,
        low: float = 0.0,
        close: float = 0.0,
        velocity: float = 0.0,
        level_store: LevelStore = None,
        # Legacy positional support: if called with old signature (high, low, close, level_store)
        **kwargs,
    ) -> Optional[PatternSignal]:
        """Process one bar. Returns PatternSignal if breakdown confirms."""

        if self.state == ShortState.IDLE:
            self._scan_for_break(
                open_, low, close, velocity, level_store, timestamp, bar_idx
            )
            return None

        elif self.state == ShortState.BREAK_DETECTED:
            assert self._target_level is not None
            level_price = self._target_level.price

            # Track lowest low during the breakdown
            if low < self._lowest_low:
                self._lowest_low = low

            # Check if price recovered above level — this is a failed breakdown (long), not our setup
            # Use >= so a close exactly AT the level counts as recovery (not breakdown)
            if close >= level_price:
                self.reset()
                return None

            # Price still below level — count confirmation bars
            self._bars_below += 1

            # Reject if price has already moved too far below (late entry)
            if level_price - close > self.params.bd_max_break_depth_pts:
                self.reset()
                return None

            # Timeout: if we've been watching too long without confirming
            if bar_idx - self._break_bar > self.params.bd_timeout_bars:
                self.reset()
                return None

            # Accumulate conviction score for this bar
            bar_score = self._compute_bar_conviction(
                level_price, open_, high, low, close, velocity
            )
            self._conviction_score += bar_score
            self._prev_low_below = low

            # Confirm when conviction threshold met AND minimum bars observed
            if (self._conviction_score >= self.params.bd_conviction_threshold
                    and self._bars_below >= self.params.bd_min_bars_floor):
                return self._emit_signal(bar_idx, timestamp, close)

            return None

        return None

    def _compute_bar_conviction(
        self,
        level_price: float,
        open_: float,
        high: float,
        low: float,
        close: float,
        velocity: float,
    ) -> float:
        """Compute conviction score for one bar below the broken level.

        Components:
          A. Base hold: 1.0 (every bar below)
          B. Depth bonus: how far below the level (normalized)
          C. Velocity bonus: selling speed (negative velocity)
          D. Candle character: bearish = close near low
          E. New low bonus: bar makes a new low vs prior bars
        """
        p = self.params
        score = 1.0  # A: base hold

        # B: Depth bonus
        depth_pts = level_price - close
        if p.bd_conviction_depth_norm_pts > 0 and depth_pts > 0 and p.bd_conviction_depth_weight > 0:
            depth_ratio = min(depth_pts / p.bd_conviction_depth_norm_pts, 1.0)
            score += depth_ratio * p.bd_conviction_depth_weight

        # C: Velocity bonus (negative velocity = selling pressure)
        if p.bd_conviction_velocity_norm > 0 and p.bd_conviction_velocity_weight > 0:
            abs_sell_velocity = max(-velocity, 0.0)
            velocity_ratio = min(abs_sell_velocity / p.bd_conviction_velocity_norm, 1.0)
            score += velocity_ratio * p.bd_conviction_velocity_weight

        # D: Candle character (close near low = bearish follow-through)
        bar_range = high - low
        if bar_range > 0 and p.bd_conviction_candle_weight > 0:
            close_position = (close - low) / bar_range
            bearish_score = 1.0 - close_position
            score += bearish_score * p.bd_conviction_candle_weight

        # E: New low bonus (progression, not stalling)
        if low < self._prev_low_below and p.bd_conviction_new_low_weight > 0:
            score += p.bd_conviction_new_low_weight

        return score

    def _scan_for_break(
        self,
        open_: float,
        low: float,
        close: float,
        velocity: float,
        level_store: LevelStore,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """Detect price breaking below a significant support level.

        Requires BOTH:
        - low penetrates level by at least bd_min_break_depth_pts
        - close is below the level (not just a wick)
        """
        min_depth = self.params.bd_min_break_depth_pts
        confirmed = level_store.get_confirmed(timestamp)

        allowed = self._MAJOR_SUPPORT_TYPES if self.params.bd_require_major_level else self._SUPPORT_TYPES
        for level in confirmed:
            if level.level_type in allowed:
                sweep_depth = level.price - low
                if sweep_depth >= min_depth and close < level.price:
                    self.state = ShortState.BREAK_DETECTED
                    self._target_level = level
                    self._break_bar = bar_idx
                    self._bars_below = 1  # this bar counts
                    self._lowest_low = low
                    self._break_close = close
                    # Initialize conviction with first bar's score
                    self._prev_low_below = float("inf")  # first bar always counts as new low
                    self._conviction_score = self._compute_bar_conviction(
                        level.price, open_, level.price + 1, low, close, velocity
                    )
                    self._prev_low_below = low
                    return

    def _emit_signal(
        self,
        bar_idx: int,
        timestamp: datetime,
        entry_price: float,
    ) -> PatternSignal:
        """Emit a breakdown_short PatternSignal."""
        assert self._target_level is not None

        # Stop above the broken level
        stop_price = self._target_level.price + self.params.bd_stop_buffer_pts

        signal = PatternSignal(
            pattern_type="breakdown_short",
            confirmation=ConfirmationType.ACCEPTANCE,
            level=self._target_level,
            sweep_low=self._lowest_low,
            sweep_depth_pts=self._target_level.price - self._lowest_low,
            entry_price=entry_price,
            stop_price=stop_price,
            bar_idx=bar_idx,
            timestamp=timestamp,
            direction="short",
            sweep_high=self._target_level.price,  # the broken level
        )
        self.reset()
        return signal


    def get_state_snapshot(self) -> dict:
        """Serialize active pattern state for persistence across restarts."""
        target_level = None
        if self._target_level is not None:
            target_level = {
                "price": self._target_level.price,
                "level_type": self._target_level.level_type.name,
                "created_at": self._target_level.created_at.isoformat(),
                "confirmed_at": self._target_level.confirmed_at.isoformat() if self._target_level.confirmed_at else None,
                "touch_count": self._target_level.touch_count,
                "rally_from_low_pts": self._target_level.rally_from_low_pts,
                "is_active": self._target_level.is_active,
                "label": self._target_level.label,
            }

        return {
            "state": self.state.name,
            "target_level": target_level,
            "break_bar": self._break_bar,
            "bars_below": self._bars_below,
            "lowest_low": self._lowest_low,
            "break_close": self._break_close,
            "conviction_score": self._conviction_score,
            "prev_low_below": self._prev_low_below,
        }

    def restore_state(self, snapshot: dict) -> None:
        """Restore pattern state from a saved snapshot."""
        from config.levels import Level, LevelType

        state_name = snapshot.get("state", "IDLE")
        try:
            self.state = ShortState[state_name]
        except KeyError:
            self.state = ShortState.IDLE
            return

        tl = snapshot.get("target_level")
        if tl is not None:
            self._target_level = Level(
                price=tl["price"],
                level_type=LevelType[tl["level_type"]],
                created_at=datetime.fromisoformat(tl["created_at"]),
                confirmed_at=datetime.fromisoformat(tl["confirmed_at"]) if tl.get("confirmed_at") else None,
                touch_count=tl.get("touch_count", 1),
                rally_from_low_pts=tl.get("rally_from_low_pts", 0.0),
                is_active=tl.get("is_active", True),
                label=tl.get("label", ""),
            )
        else:
            self._target_level = None

        self._break_bar = snapshot.get("break_bar", -1)
        self._bars_below = snapshot.get("bars_below", 0)
        self._lowest_low = snapshot.get("lowest_low", float("inf"))
        self._break_close = snapshot.get("break_close", 0.0)
        self._conviction_score = snapshot.get("conviction_score", 0.0)
        self._prev_low_below = snapshot.get("prev_low_below", float("inf"))


class VelocityBreakdownShort:
    """Single-bar velocity breakdown — catches news-driven breaks.

    When a major support level breaks on a single high-volume bar (e.g., 4000+
    volume, 5x average), the multi-bar BD detector can't catch it because
    it requires 15+ bars of confirmation. This detector fires on ONE bar
    if the break is deep enough and volume is high enough.

    Conservative by default: 25% position size, major levels only.
    """

    _MAJOR_SUPPORT_TYPES = frozenset({
        LevelType.PRIOR_DAY_LOW,
        LevelType.MULTI_HOUR_LOW,
    })

    _ALL_SUPPORT_TYPES = frozenset({
        LevelType.PRIOR_DAY_LOW,
        LevelType.MULTI_HOUR_LOW,
        LevelType.CLUSTER_LOW,
    })

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params

    def update(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        volume: float,
        avg_volume_20: float,
        level_store: LevelStore,
    ) -> Optional[PatternSignal]:
        """Process one bar. Returns PatternSignal if velocity breakdown detected.

        Parameters
        ----------
        volume : float
            Current bar's volume.
        avg_volume_20 : float
            20-bar rolling average volume. If <= 0, check is skipped.
        """
        if avg_volume_20 <= 0:
            return None

        # Volume must be >= ratio * average
        if volume < self.params.vbd_min_volume_ratio * avg_volume_20:
            return None

        allowed = (
            self._MAJOR_SUPPORT_TYPES
            if self.params.vbd_only_major_levels
            else self._ALL_SUPPORT_TYPES
        )
        confirmed = level_store.get_confirmed(timestamp)

        for level in confirmed:
            if level.level_type not in allowed:
                continue

            level_price = level.price
            break_depth = level_price - low

            # Bar must break through level by enough points
            if break_depth < self.params.vbd_min_break_pts:
                continue

            # Upper cap: a one-bar print taking the level by 30-49pts is the
            # back half of a crash, not a clean velocity breakdown. Reject.
            max_break = getattr(self.params, 'vbd_max_break_pts', 0.0)
            if max_break > 0 and break_depth > max_break:
                continue

            # Bar must close below level (if required)
            if self.params.vbd_require_close_below and close >= level_price:
                continue

            # All conditions met — emit signal
            stop_price = level_price + self.params.vbd_stop_buffer_pts

            return PatternSignal(
                pattern_type="velocity_short",
                confirmation=ConfirmationType.ACCEPTANCE,
                level=level,
                sweep_low=low,
                sweep_depth_pts=break_depth,
                entry_price=close,
                stop_price=stop_price,
                bar_idx=bar_idx,
                timestamp=timestamp,
                direction="short",
                sweep_high=level_price,  # the broken level
            )

        return None


class BacktestShort:
    """Mancini-faithful Back-Test Short: failed retest of broken SUPPORT.

    Per Mancini (2024-10-09 post, 3 explicit criteria):
      1. Price sets a clearly defined support — horizontal support with a few
         touches, a trendline, or a "significant low".
      2. Price breaks down that support decisively. "It must be a forceful,
         deep breakdown that ideally lasts hours, days, or weeks." Example:
         "we flushed 5805 and sold 80+ points in one flush."
      3. Price back-tests the broken level from below. The FIRST retest of
         the zone from below is typically an actionable short. "Sometimes the
         second/third tests are actionable, but the odds drop with each
         successive test. Eventually the level will be drained and price
         will rip through and squeeze."

    Also per Mancini (2025-01-16): "any level that was previously the trigger
    point for a Breakdown Short will by default be a location to engage a
    Back-Test Short when re-tested."

    PRIOR_DAY_LOW is excluded from the support-shelf set even though Mancini
    would qualify it — Phase 1 of the short-engine rewrite (block_pdl_shorts)
    rejects ANY short at PDL because PDL is the primary Mancini long-side FB
    level. Re-enable in a future PR if we get evidence it's profitable here.

    State machine:
      IDLE → (close below shelf for N bars) → tracking broken support
        → (flush depth ≥ X pts reached) → eligible for back-test
        → (high rallies back to shelf from below) → BACKTEST_WATCH
        → (closes below for M bars after touch) → emit signal
        → (closes above for K bars) → abort (reclaimed)
    """

    _SHELF_TYPES = frozenset({
        LevelType.HORIZONTAL_SR,
        LevelType.CLUSTER_LOW,
        LevelType.MULTI_HOUR_LOW,
        LevelType.SWING_LOW,
        # PRIOR_DAY_LOW intentionally excluded — see class docstring.
    })

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params
        self.state = ShortState.IDLE
        # Tracking broken support shelves: level_price -> dict of state
        # {"level": Level, "broken_bar": int, "lowest_low": float,
        #  "deep_flush": bool, "touch_count": int}
        self._broken_supports: dict[float, dict] = {}
        # Per-level consecutive-close-below counter (to confirm breakdown)
        self._bars_below: dict[float, int] = {}
        # Per-level: was price recently above (for breakdown detection)
        self._was_above: dict[float, bool] = {}
        # Current back-test state
        self._target_level: Optional[Level] = None
        self._target_lp: float = 0.0
        self._backtest_bar: int = -1
        self._backtest_high: float = float("-inf")
        self._bars_below_after_touch: int = 0
        self._bars_above_after_touch: int = 0

    def reset(self) -> None:
        """Reset the active back-test watch only (preserves broken-support memory)."""
        self.state = ShortState.IDLE
        self._target_level = None
        self._target_lp = 0.0
        self._backtest_bar = -1
        self._backtest_high = float("-inf")
        self._bars_below_after_touch = 0
        self._bars_above_after_touch = 0

    def full_reset(self) -> None:
        """Full reset for a new session — clears broken-support memory."""
        self.reset()
        self._broken_supports.clear()
        self._bars_below.clear()
        self._was_above.clear()

    def update(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        level_store: LevelStore,
    ) -> Optional[PatternSignal]:
        """Process one bar. Returns PatternSignal if back-test rejection confirms."""

        # 1. Track breakdowns of support shelves (always, even mid-watch)
        self._track_breakdowns(bar_idx, low, close, level_store, timestamp)

        # 2. Expire stale broken supports
        expire_bars = self.params.bts_breakout_expire_bars
        self._broken_supports = {
            lp: state for lp, state in self._broken_supports.items()
            if bar_idx - state["broken_bar"] <= expire_bars
        }

        # 3. State-machine processing
        if self.state == ShortState.IDLE:
            self._scan_for_backtest(bar_idx, high, low, close)
            return None

        elif self.state == ShortState.BACKTEST_WATCH:
            return self._check_rejection(bar_idx, timestamp, high, low, close)

        return None

    # --- Breakdown tracking ----------------------------------------------------

    def _track_breakdowns(
        self,
        bar_idx: int,
        low: float,
        close: float,
        level_store: LevelStore,
        timestamp: datetime,
    ) -> None:
        """Detect when price closes below a support shelf for N consecutive
        bars (Mancini's "decisive breakdown" condition). Track lowest-low
        afterward to confirm "forceful, deep" flush depth."""
        confirm_bars = self.params.bts_breakdown_confirm_bars
        min_touches = self.params.bts_support_min_touches
        confirmed = level_store.get_confirmed(timestamp)

        for level in confirmed:
            if level.level_type not in self._SHELF_TYPES:
                continue
            # Shelf strength gate: cluster/horizontal must have enough touches
            if level.level_type in (LevelType.HORIZONTAL_SR, LevelType.CLUSTER_LOW):
                if (level.touch_count or 0) < min_touches:
                    continue

            lp = round(level.price, 2)

            # Already-broken shelves: just update lowest_low & flush flag
            if lp in self._broken_supports:
                state = self._broken_supports[lp]
                if low < state["lowest_low"]:
                    state["lowest_low"] = low
                if not state["deep_flush"]:
                    depth = level.price - state["lowest_low"]
                    if depth >= self.params.bts_min_flush_depth_pts:
                        state["deep_flush"] = True
                continue

            # New shelf: count consecutive closes below — but only after
            # we've first seen price AT or ABOVE the level. Levels that
            # form while price is already below them aren't "breakdowns"
            # in Mancini's sense.
            if close >= level.price:
                self._was_above[lp] = True
                self._bars_below[lp] = 0
            elif self._was_above.get(lp, False):
                self._bars_below[lp] = self._bars_below.get(lp, 0) + 1
                if self._bars_below[lp] == confirm_bars:
                    # Decisive breakdown confirmed — register the broken shelf
                    self._broken_supports[lp] = {
                        "level": level,
                        "broken_bar": bar_idx,
                        "lowest_low": low,
                        "deep_flush": (level.price - low) >= self.params.bts_min_flush_depth_pts,
                        "touch_count": 0,
                    }

    # --- Back-test scanning ----------------------------------------------------

    def _scan_for_backtest(
        self, bar_idx: int, high: float, low: float, close: float
    ) -> None:
        """When price rallies back up to within max_distance of a broken support
        shelf (and that shelf had a deep flush), engage BACKTEST_WATCH."""
        max_dist = self.params.bts_max_distance_from_level

        for lp, state in self._broken_supports.items():
            if not state["deep_flush"]:
                continue  # Not yet a Mancini-grade breakdown
            level_price = state["level"].price

            # Skip if price is currently above the level — back-test is from
            # below approaching up, not from above.
            if close >= level_price:
                continue

            # High must reach the level from below (within max_dist)
            if high >= level_price - max_dist:
                # Touch counted. Apply first-touch-only rule.
                state["touch_count"] += 1
                if (self.params.bts_first_touch_only
                        and state["touch_count"] > 1):
                    continue
                self.state = ShortState.BACKTEST_WATCH
                self._target_level = state["level"]
                self._target_lp = level_price
                self._backtest_bar = bar_idx
                self._backtest_high = high
                self._bars_below_after_touch = 0
                self._bars_above_after_touch = 0
                return

    # --- Rejection confirmation -----------------------------------------------

    def _check_rejection(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ) -> Optional[PatternSignal]:
        """After the back-test touch: confirm price rejects (closes back below
        the broken shelf for N bars) before emitting a signal. If price
        reclaims the shelf, abort."""
        level_price = self._target_lp

        # Track post-touch highest
        if high > self._backtest_high:
            self._backtest_high = high

        if close < level_price:
            self._bars_below_after_touch += 1
            self._bars_above_after_touch = 0
        else:
            self._bars_above_after_touch += 1
            self._bars_below_after_touch = 0
            if self._bars_above_after_touch >= self.params.bts_reclaim_abort_bars:
                # Mancini's "level was drained, price will rip through and
                # squeeze" — abort, no short.
                self.reset()
                return None

        # Timeout — give up waiting for rejection
        if bar_idx - self._backtest_bar > self.params.bts_timeout_bars:
            self.reset()
            return None

        # Confirmed rejection
        if self._bars_below_after_touch >= self.params.bts_confirm_bars:
            return self._emit_signal(bar_idx, timestamp, close)

        return None

    def _emit_signal(
        self,
        bar_idx: int,
        timestamp: datetime,
        entry_price: float,
    ) -> PatternSignal:
        """Emit a backtest_short PatternSignal. Stop above the back-test high
        (Mancini: 'stop above the rejection wick')."""
        assert self._target_level is not None
        stop_price = self._backtest_high + self.params.bts_stop_buffer_pts

        signal = PatternSignal(
            pattern_type="backtest_short",
            confirmation=ConfirmationType.ACCEPTANCE,
            level=self._target_level,
            sweep_low=entry_price,
            sweep_depth_pts=0.0,
            entry_price=entry_price,
            stop_price=stop_price,
            bar_idx=bar_idx,
            timestamp=timestamp,
            direction="short",
            sweep_high=self._backtest_high,
        )
        self.reset()
        return signal
