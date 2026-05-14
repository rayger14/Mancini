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
    BREAKOUT_TRACKED = auto()     # Resistance was broken upward (for BacktestShort)
    PULLBACK_DETECTED = auto()    # Price pulled back after breakout
    BACKTEST_WATCH = auto()       # Watching for failed backtest
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
    """Detect failed backtests of previously broken resistance for short entries.

    When a resistance level was previously broken above (breakout), then price
    pulls back and retests it from below, and the retest fails — short.

    State machine:
    1. Track breakouts (price closes above resistance for N bars)
    2. Detect pullback (price drops back toward level)
    3. Watch for backtest (price touches level from below)
    4. Confirm rejection (price fails to hold above for N bars)

    IDLE → BREAKOUT_TRACKED → PULLBACK_DETECTED → BACKTEST_WATCH → Signal
    """

    _RESISTANCE_TYPES = frozenset({
        LevelType.PRIOR_DAY_HIGH,
        LevelType.MULTI_HOUR_HIGH,
        LevelType.CLUSTER_HIGH,
        LevelType.HORIZONTAL_SR,
        LevelType.SWING_HIGH,
    })

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params
        self.state = ShortState.IDLE
        # Track broken resistance levels (breakouts)
        self._broken_resistances: list[tuple[Level, int, float]] = []
        # (level, bar_idx_of_breakout, breakout_high)
        self._target_level: Optional[Level] = None
        self._backtest_bar: int = -1
        self._backtest_high: float = float("-inf")
        self._bars_below: int = 0
        self._breakout_expire_bars: int = 200  # breakout memory window
        # Track price relative to resistance levels for breakout detection
        self._bars_above: dict[float, int] = {}  # level_price -> consecutive bars above

    def reset(self) -> None:
        self.state = ShortState.IDLE
        self._target_level = None
        self._backtest_bar = -1
        self._backtest_high = float("-inf")
        self._bars_below = 0

    def full_reset(self) -> None:
        """Full reset including breakout memory (for new session)."""
        self.reset()
        self._broken_resistances.clear()
        self._bars_above.clear()

    def update(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        level_store: LevelStore,
    ) -> Optional[PatternSignal]:
        """Process one bar. Returns PatternSignal if backtest rejection confirms."""

        # Always track breakouts (even while in other states)
        self._track_breakouts(bar_idx, high, close, level_store, timestamp)

        # Expire old breakouts
        self._broken_resistances = [
            (lvl, bidx, bh)
            for lvl, bidx, bh in self._broken_resistances
            if bar_idx - bidx <= self._breakout_expire_bars
        ]

        if self.state == ShortState.IDLE:
            self._scan_for_backtest(high, low, close, timestamp, bar_idx)
            return None

        elif self.state == ShortState.BACKTEST_WATCH:
            return self._check_rejection(bar_idx, timestamp, high, low, close)

        return None

    def _track_breakouts(
        self,
        bar_idx: int,
        high: float,
        close: float,
        level_store: LevelStore,
        timestamp: datetime,
    ) -> None:
        """Track when price breaks above resistance levels (potential future backtest targets)."""
        confirm_bars = self.params.bt_breakout_confirm_bars
        confirmed = level_store.get_confirmed(timestamp)

        for level in confirmed:
            if level.level_type not in self._RESISTANCE_TYPES:
                continue

            lp = round(level.price, 2)

            if close > level.price:
                self._bars_above[lp] = self._bars_above.get(lp, 0) + 1
                if self._bars_above[lp] == confirm_bars:
                    # Breakout confirmed — record it
                    # Check not already recorded
                    already = any(
                        abs(lvl.price - level.price) < 1.0
                        for lvl, _, _ in self._broken_resistances
                    )
                    if not already:
                        self._broken_resistances.append((level, bar_idx, high))
            else:
                self._bars_above[lp] = 0

    def _scan_for_backtest(
        self,
        high: float,
        low: float,
        close: float,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """Check if price is retesting a previously broken resistance from below.

        Conditions:
        - A breakout was recorded at this level
        - Price pulled back (was below level)
        - Current bar's high touches or exceeds the level
        - Close is below the level (backtest failing)
        """
        max_dist = self.params.bt_max_distance_from_level

        for level, breakout_bar, breakout_high in self._broken_resistances:
            lp = level.price

            # Price must have pulled back below the level first
            pullback = breakout_high - close
            if pullback < self.params.bt_pullback_min_pts:
                continue

            # Current bar touches or approaches the level from below
            if high >= lp - max_dist and close < lp:
                self.state = ShortState.BACKTEST_WATCH
                self._target_level = level
                self._backtest_bar = bar_idx
                self._backtest_high = high
                self._bars_below = 1 if close < lp else 0
                return

    def _check_rejection(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ) -> Optional[PatternSignal]:
        """Confirm the backtest rejection.

        Count bars closing below the level after the backtest touch.
        If bt_confirm_bars reached, the backtest failed — emit short.
        If close goes back above level for bt_reclaim_abort_bars, abort.
        """
        assert self._target_level is not None
        level_price = self._target_level.price

        # Track the highest point during the backtest
        if high > self._backtest_high:
            self._backtest_high = high

        if close < level_price:
            self._bars_below += 1
        else:
            # Price reclaimed the level — check if we should abort
            self._bars_below = 0
            # If price closes above for too many bars, the backtest succeeded
            if bar_idx - self._backtest_bar > self.params.bt_reclaim_abort_bars:
                self.reset()
                return None

        # Timeout
        if bar_idx - self._backtest_bar > self.params.bt_timeout_bars:
            self.reset()
            return None

        # Confirmed! Backtest failed — price rejected from resistance
        if self._bars_below >= self.params.bt_confirm_bars:
            return self._emit_signal(bar_idx, timestamp, close)

        return None

    def _emit_signal(
        self,
        bar_idx: int,
        timestamp: datetime,
        entry_price: float,
    ) -> PatternSignal:
        """Emit a backtest_short PatternSignal."""
        assert self._target_level is not None

        # Stop above the highest point of the backtest attempt
        stop_price = self._backtest_high + self.params.bt_stop_buffer_pts

        signal = PatternSignal(
            pattern_type="backtest_short",
            confirmation=ConfirmationType.ACCEPTANCE,
            level=self._target_level,
            sweep_low=entry_price,  # entry is the "sweep" point for shorts
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
