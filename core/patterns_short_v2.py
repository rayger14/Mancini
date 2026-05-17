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
    """Mancini-faithful Breakdown Short: shorts the FAILURE of an FB long.

    Per Mancini's definitive 2025-08-24 post, the BD short setup has THREE
    sequential criteria:
      1. An obvious support shelf (multi-touch / well-defined / "gold
         standard" prior-day low or shelf of lows).
      2. A "successful" Failed Breakdown attempt at the shelf: price loses
         the shelf, recovers it, rallies.
      3. The FB then FAILS — price comes back DOWN and breaks the lowest
         low of that FB attempt. "Short trigger should go a few point
         buffer below" the broken FB low.

    Direct quote (2025-08-24): "criteria for Breakdown Shorts. We had 1) An
    obvious shelf at 6460. 2) A successful Failed Breakdown at the shelf
    where price lost the Monday daily low of 6456, recovered it, and
    rallied to 6474. Once these criteria are met, we can short when the
    lowest low of that Failed Breakdown (6454 1:30AM Tuesday low fails).
    Short trigger should go a few point buffer below; 6447 would suffice."

    Mancini personally SKIPS this setup ("to simplify my trading, I often
    just skip the Breakdown Shorts altogether and wait for the Failed
    Breakdown" — 2025-03-24) because flipping short→long is cognitively
    expensive for him. The engine has no such constraint — it takes both
    the BD short (this pattern) AND the next FB long when it triggers.

    Replaces the prior "close-below-shelf for N bars" detector, which fired
    on the INITIAL break — exactly when Mancini says NOT to short ("when ES
    is in 'elevator down' mode we don't bottom pick. We wait for a failed
    breakdown or planned level reclaim" — 2025-08-24). That older detector
    had 0/21 WR on live signals (production + phantom).

    State machine, per shelf being tracked:
      watching → (close < shelf) → flush
      flush    → (deep flush + recovery above shelf) → recovered
               → (max flush bars exceeded) → abandoned
               → (recovery shallow, no real FB attempt) → reset to watching
      recovered → (close <= flush_low - buffer) → emit BD short
                → (rally pts + timeout without failure) → abandoned (FB won)
                → (recovery_watch_bars timeout) → abandoned (no resolution)
    """

    # Shelf level types that can trigger BD. Mancini: "gold standard" PDL,
    # multi-hour low (20+pt), cluster shelf, horizontal multi-touch. PDL
    # is included here even though Phase 1 (block_pdl_shorts) blocks any
    # short with pattern.level=PDL: the BD signal emitted here carries a
    # synthetic INTRADAY_LOW level representing the broken flush_low, not
    # the shelf itself, so the Phase 1 gate does not apply.
    _SHELF_TYPES = frozenset({
        LevelType.PRIOR_DAY_LOW,
        LevelType.MULTI_HOUR_LOW,
        LevelType.CLUSTER_LOW,
        LevelType.HORIZONTAL_SR,
    })

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params
        # Compatibility surface: outer code reads self.state as a binary
        # IDLE / not-IDLE flag. Set non-IDLE when any shelf is in active
        # flush or recovered state.
        self.state = ShortState.IDLE
        # Per-shelf state: {round(price,2) -> dict}
        # Each entry:
        #   level: Level
        #   created_bar: int (when added to tracking, for expiration)
        #   state: "watching" | "flush" | "recovered" | "abandoned"
        #   flush_start_bar: int
        #   flush_low: float
        #   recovery_bar: int
        #   recovery_high: float
        self._shelves: dict[float, dict] = {}

    def reset(self) -> None:
        """Reset state flag only (preserves shelf tracking memory)."""
        self.state = ShortState.IDLE

    def full_reset(self) -> None:
        """Full reset for a new session — clears shelf memory."""
        self.reset()
        self._shelves.clear()

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
        **kwargs,
    ) -> Optional[PatternSignal]:
        """Process one bar. Returns PatternSignal when a tracked shelf's
        FB attempt fails (price re-breaks the flush_low)."""
        if level_store is None:
            return None

        # 1. Discover new support shelves
        self._discover_shelves(bar_idx, level_store, timestamp)

        # 2. Expire stale shelves
        expire_bars = getattr(self.params, "bd_shelf_expire_bars", 480)
        self._shelves = {
            lp: s for lp, s in self._shelves.items()
            if bar_idx - s["created_bar"] <= expire_bars
        }

        # 3. Advance each shelf's state machine. First signal wins.
        for lp in list(self._shelves.keys()):
            signal = self._advance_shelf(
                self._shelves[lp], bar_idx, timestamp, high, low, close
            )
            if signal is not None:
                # Remove the shelf — it has fired its signal
                del self._shelves[lp]
                self.reset()
                return signal

        # Compatibility: surface non-IDLE state when any shelf is mid-cycle
        any_active = any(
            s["state"] in ("flush", "recovered")
            for s in self._shelves.values()
        )
        self.state = ShortState.BREAK_DETECTED if any_active else ShortState.IDLE
        return None

    def _discover_shelves(
        self, bar_idx: int, level_store: LevelStore, timestamp: datetime,
    ) -> None:
        """Add confirmed support shelves to tracking. Strength gates:
        cluster/horizontal shelves require min touches."""
        confirmed = level_store.get_confirmed(timestamp)
        min_touches = getattr(self.params, "bd_shelf_min_touches", 3)

        for level in confirmed:
            if level.level_type not in self._SHELF_TYPES:
                continue
            if level.level_type in (
                LevelType.CLUSTER_LOW, LevelType.HORIZONTAL_SR,
            ):
                if (level.touch_count or 0) < min_touches:
                    continue
            lp = round(level.price, 2)
            if lp in self._shelves:
                continue
            self._shelves[lp] = {
                "level": level,
                "created_bar": bar_idx,
                "state": "watching",
                "flush_start_bar": -1,
                "flush_low": float("inf"),
                "recovery_bar": -1,
                "recovery_high": float("-inf"),
            }

    def _advance_shelf(
        self,
        shelf: dict,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ) -> Optional[PatternSignal]:
        """Per-shelf state machine. Returns a PatternSignal if the FB at
        this shelf fails (Mancini's criterion 3 met)."""
        level_price = shelf["level"].price
        state = shelf["state"]

        if state == "watching":
            # Looking for the start of an FB attempt
            if close < level_price:
                shelf["state"] = "flush"
                shelf["flush_start_bar"] = bar_idx
                shelf["flush_low"] = low
            return None

        if state == "flush":
            # Track flush_low while price is below the shelf
            if low < shelf["flush_low"]:
                shelf["flush_low"] = low
            flush_depth = level_price - shelf["flush_low"]
            time_in_flush = bar_idx - shelf["flush_start_bar"]

            # Recovery: close back above shelf
            if close >= level_price:
                min_depth = getattr(self.params, "bd_min_flush_depth_pts", 3.0)
                if flush_depth >= min_depth:
                    shelf["state"] = "recovered"
                    shelf["recovery_bar"] = bar_idx
                    shelf["recovery_high"] = high
                else:
                    # Shallow tag — not a real FB attempt; reset to watching
                    shelf["state"] = "watching"
                    shelf["flush_start_bar"] = -1
                    shelf["flush_low"] = float("inf")
                return None

            # Sat below too long without recovery — abandon (this shelf
            # broke and stayed broken; that's a trend leg, not an FB
            # failure setup)
            max_flush_bars = getattr(self.params, "bd_max_flush_bars", 30)
            if time_in_flush > max_flush_bars:
                shelf["state"] = "abandoned"
            return None

        if state == "recovered":
            # Track recovery rally extremes
            if high > shelf["recovery_high"]:
                shelf["recovery_high"] = high
            time_since_recovery = bar_idx - shelf["recovery_bar"]

            # Mancini's criterion 3: "we can short when the lowest low of
            # that Failed Breakdown fails". Short trigger goes a few pts
            # below.
            flush_low = shelf["flush_low"]
            buffer = getattr(self.params, "bd_fb_fail_buffer_pts", 3.0)
            if close <= flush_low - buffer:
                return self._emit_signal(shelf, bar_idx, timestamp, close)

            # FB succeeded — meaningful rally and no failure → abandon
            success_rally = getattr(self.params, "bd_fb_success_rally_pts", 20.0)
            success_timeout = getattr(self.params, "bd_fb_success_timeout_bars", 60)
            rally_pts = shelf["recovery_high"] - level_price
            if (rally_pts >= success_rally
                    and time_since_recovery > success_timeout):
                shelf["state"] = "abandoned"
                return None

            # General timeout — no resolution within window
            watch_bars = getattr(self.params, "bd_recovery_watch_bars", 120)
            if time_since_recovery > watch_bars:
                shelf["state"] = "abandoned"
            return None

        # "abandoned" — wait for expiry cleanup
        return None

    def _emit_signal(
        self,
        shelf: dict,
        bar_idx: int,
        timestamp: datetime,
        entry_price: float,
    ) -> PatternSignal:
        """Emit a BD short signal. The signal carries a SYNTHETIC level
        representing the broken flush_low (the actionable level), not the
        underlying shelf — so Phase 1's PDL block applies to the right
        kind of trade."""
        flush_low = shelf["flush_low"]
        recovery_high = shelf["recovery_high"]
        shelf_level = shelf["level"]

        flush_level = Level(
            price=flush_low,
            level_type=LevelType.INTRADAY_LOW,
            created_at=timestamp,
            confirmed_at=timestamp,
            touch_count=1,
            label=f"FB_FAIL_LOW@{flush_low:.2f}(shelf={shelf_level.label})",
        )

        stop_buffer = getattr(self.params, "bd_stop_buffer_pts", 3.0)
        signal = PatternSignal(
            pattern_type="breakdown_short",
            confirmation=ConfirmationType.ACCEPTANCE,
            level=flush_level,
            sweep_low=entry_price,
            sweep_depth_pts=flush_low - entry_price,
            entry_price=entry_price,
            stop_price=recovery_high + stop_buffer,
            bar_idx=bar_idx,
            timestamp=timestamp,
            direction="short",
            sweep_high=recovery_high,
        )
        return signal

    def get_state_snapshot(self) -> dict:
        """Serialize per-shelf state for restart persistence."""
        shelves_out = {}
        for lp, s in self._shelves.items():
            lvl = s["level"]
            shelves_out[str(lp)] = {
                "level": {
                    "price": lvl.price,
                    "level_type": lvl.level_type.name,
                    "created_at": lvl.created_at.isoformat(),
                    "confirmed_at": lvl.confirmed_at.isoformat() if lvl.confirmed_at else None,
                    "touch_count": lvl.touch_count,
                    "rally_from_low_pts": lvl.rally_from_low_pts,
                    "is_active": lvl.is_active,
                    "label": lvl.label,
                },
                "created_bar": s["created_bar"],
                "state": s["state"],
                "flush_start_bar": s["flush_start_bar"],
                "flush_low": s["flush_low"],
                "recovery_bar": s["recovery_bar"],
                "recovery_high": s["recovery_high"],
            }
        return {
            "state": self.state.name,
            "shelves": shelves_out,
        }

    def restore_state(self, snapshot: dict) -> None:
        """Restore per-shelf state from a saved snapshot."""
        state_name = snapshot.get("state", "IDLE")
        try:
            self.state = ShortState[state_name]
        except KeyError:
            self.state = ShortState.IDLE

        self._shelves = {}
        for lp_str, s in (snapshot.get("shelves") or {}).items():
            lvl_d = s.get("level") or {}
            level = Level(
                price=lvl_d["price"],
                level_type=LevelType[lvl_d["level_type"]],
                created_at=datetime.fromisoformat(lvl_d["created_at"]),
                confirmed_at=(datetime.fromisoformat(lvl_d["confirmed_at"])
                              if lvl_d.get("confirmed_at") else None),
                touch_count=lvl_d.get("touch_count", 1),
                rally_from_low_pts=lvl_d.get("rally_from_low_pts", 0.0),
                is_active=lvl_d.get("is_active", True),
                label=lvl_d.get("label", ""),
            )
            self._shelves[float(lp_str)] = {
                "level": level,
                "created_bar": s.get("created_bar", -1),
                "state": s.get("state", "watching"),
                "flush_start_bar": s.get("flush_start_bar", -1),
                "flush_low": s.get("flush_low", float("inf")),
                "recovery_bar": s.get("recovery_bar", -1),
                "recovery_high": s.get("recovery_high", float("-inf")),
            }


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
