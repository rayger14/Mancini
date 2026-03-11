"""Failed Breakdown & Level Reclaim state machines.

Failed Breakdown sequence:
  Elevator Down → Significant Low Swept → Recovery → Confirmation → Signal

Level Reclaim sequence:
  Horizontal S/R Reclaimed from Below → Confirmation → Signal
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams, DEFAULT_STRATEGY
from core.elevator_down import ElevatorEvent


class PatternState(Enum):
    """State machine states for pattern detection."""

    IDLE = auto()
    SWEEP_DETECTED = auto()     # Price broke below level
    RECOVERY_DETECTED = auto()  # Price recovered above level
    ACCEPTANCE_WATCH = auto()   # Waiting for acceptance confirmation
    NON_ACCEPTANCE_WATCH = auto()  # Waiting for non-acceptance confirmation
    CONFIRMED = auto()          # Pattern confirmed, ready for entry


class ConfirmationType(Enum):
    """How the pattern was confirmed."""

    ACCEPTANCE = auto()       # backtest + hold
    NON_ACCEPTANCE = auto()   # fast recovery + hold


@dataclass
class PatternSignal:
    """Output from a confirmed pattern detection."""

    pattern_type: str  # "failed_breakdown", "level_reclaim", "failed_rally", "level_rejection"
    confirmation: ConfirmationType
    level: Level
    sweep_low: float  # lowest price during the sweep (long-side)
    entry_price: float  # confirmation price (entry point)
    stop_price: float  # below sweep low (long) or above sweep high (short)
    bar_idx: int
    timestamp: datetime
    sweep_depth_pts: float = 0.0  # how far below/above the level price swept
    elevator_event: Optional[ElevatorEvent] = None
    direction: str = "long"  # "long" or "short"
    sweep_high: float = 0.0  # highest price during the sweep (short-side)

    @property
    def risk_pts(self) -> float:
        if self.direction == "short":
            return self.stop_price - self.entry_price
        return self.entry_price - self.stop_price


class FailedBreakdown:
    """State machine for Failed Breakdown detection.

    Three entry paths:
    1. Elevator FB — fast selloff sweeps a significant low, then recovers
    2. Level Sweep FB — price sweeps below a high-quality level (prior day low,
       multi-hour low, cluster) without needing a fast elevator. The level
       quality is the filter, not the selloff speed.
    3. Double-dip — re-entry without elevator at a level where we were
       recently stopped out

    All paths share the same confirmation logic (acceptance/non-acceptance).
    """

    # High-quality level types that don't need an elevator to justify a FB.
    # These are significant enough that a sweep + recovery IS the signal.
    _HIGH_QUALITY_LEVELS = frozenset({
        LevelType.PRIOR_DAY_LOW,
        LevelType.MULTI_HOUR_LOW,
        LevelType.INTRADAY_LOW,
    })

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params
        self.state = PatternState.IDLE
        self._target_level: Optional[Level] = None
        self._sweep_low: float = float("inf")
        self._recovery_bar: int = -1
        self._recovery_price: float = 0.0
        self._hold_bars: int = 0
        self._elevator_event: Optional[ElevatorEvent] = None
        self._bars_below_level: int = 0  # for true breakdown abort
        # Double-dip re-entry: levels where we were recently stopped out
        self._stopped_out_levels: list[tuple[float, int]] = []  # (price, bar_idx)
        self._double_dip_cooldown_bars: int = 60  # allow re-entry within 60 bars
        self._is_double_dip: bool = False
        self._is_level_sweep: bool = False  # Path 2: level sweep without elevator
        # Near-miss tracking — setups that almost triggered but missed a threshold
        self.near_misses: list[dict] = []
        # Deep sell recovery: track used intraday levels to avoid re-triggers
        self._used_intraday_levels: list[float] = []
        # Level sweep tracking: count bars below a high-quality level
        self._sweep_tracking_level: Optional[Level] = None
        self._sweep_tracking_bars_below: int = 0
        self._sweep_tracking_low: float = float("inf")

    def reset(self) -> None:
        self.state = PatternState.IDLE
        self._target_level = None
        self._sweep_low = float("inf")
        self._recovery_bar = -1
        self._recovery_price = 0.0
        self._hold_bars = 0
        self._elevator_event = None
        self._bars_below_level = 0
        self._is_double_dip = False
        self._is_level_sweep = False
        self._sweep_tracking_level = None
        self._sweep_tracking_bars_below = 0
        self._sweep_tracking_low = float("inf")

    def record_stop_out(self, level_price: float, bar_idx: int) -> None:
        """Record a stop-out at a level for double-dip tracking."""
        self._stopped_out_levels.append((level_price, bar_idx))

    def _is_double_dip_level(self, level_price: float, bar_idx: int) -> bool:
        """Check if this level had a recent stop-out (double-dip candidate)."""
        for price, stop_bar in self._stopped_out_levels:
            if abs(price - level_price) <= 1.0 and bar_idx - stop_bar <= self._double_dip_cooldown_bars:
                return True
        return False

    def update(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        level_store: LevelStore,
        elevator_event: Optional[ElevatorEvent] = None,
    ) -> Optional[PatternSignal]:
        """Process one bar. Returns PatternSignal if pattern confirms.

        Parameters
        ----------
        bar_idx : int
        timestamp : datetime
        high, low, close : float
        level_store : LevelStore
        elevator_event : ElevatorEvent, optional
            Most recent completed elevator down event.

        Returns
        -------
        PatternSignal or None
        """
        # Pre-emption: if we're tracking a non-INTRADAY_LOW level but a fresh
        # INTRADAY_LOW is available, abandon the current tracking and switch.
        # The crash bottom is the highest-priority FB opportunity.
        if (
            self.state != PatternState.IDLE
            and self.params.allow_deep_sell_recovery
            and (self._target_level is None or self._target_level.level_type != LevelType.INTRADAY_LOW)
        ):
            # Save state in case we don't preempt
            saved_state = self.state
            saved_target = self._target_level
            saved_sweep_low = self._sweep_low
            saved_recovery_bar = self._recovery_bar
            saved_recovery_price = self._recovery_price
            saved_hold_bars = self._hold_bars
            saved_elevator = self._elevator_event
            saved_bars_below = self._bars_below_level
            saved_is_dd = self._is_double_dip
            saved_is_ls = self._is_level_sweep

            self.state = PatternState.IDLE
            self._scan_for_deep_sell_recovery(low, close, level_store, timestamp, bar_idx)
            if self.state != PatternState.IDLE:
                # Deep sell recovery found — preempt the old tracking
                return self._check_confirmation(bar_idx, timestamp, high, low, close)
            else:
                # No INTRADAY_LOW available — restore previous state
                self.state = saved_state
                self._target_level = saved_target
                self._sweep_low = saved_sweep_low
                self._recovery_bar = saved_recovery_bar
                self._recovery_price = saved_recovery_price
                self._hold_bars = saved_hold_bars
                self._elevator_event = saved_elevator
                self._bars_below_level = saved_bars_below
                self._is_double_dip = saved_is_dd
                self._is_level_sweep = saved_is_ls

        if self.state == PatternState.IDLE:
            # Clean up expired stop-out records
            self._stopped_out_levels = [
                (p, b) for p, b in self._stopped_out_levels
                if bar_idx - b <= self._double_dip_cooldown_bars
            ]

            # Path 1: Normal FB — need a completed elevator event
            if elevator_event is not None and elevator_event.is_complete:
                self._elevator_event = elevator_event
                self._is_double_dip = False
                self._is_level_sweep = False
                # Elevator takes priority — clear any level sweep tracking
                self._sweep_tracking_level = None
                self._sweep_tracking_bars_below = 0
                self._sweep_tracking_low = float("inf")
                # Check if the elevator itself swept a significant low
                self._scan_for_sweep_with_elevator(
                    low, close, level_store, timestamp, bar_idx, elevator_event
                )
                # If sweep was detected and we already recovered, fast-track
                if self.state == PatternState.SWEEP_DETECTED:
                    level_price = self._target_level.price
                    if close > level_price:
                        self.state = PatternState.RECOVERY_DETECTED
                        self._recovery_bar = bar_idx
                        self._recovery_price = close
                        recovery_pts = close - level_price
                        if recovery_pts >= self.params.non_acceptance_min_recovery_pts:
                            self.state = PatternState.NON_ACCEPTANCE_WATCH
                            self._hold_bars = 1  # this bar counts
                        else:
                            self.state = PatternState.ACCEPTANCE_WATCH
                            self._hold_bars = 1
                        return self._check_confirmation(bar_idx, timestamp, high, low, close)

            # Path 2: Deep sell recovery — newly confirmed INTRADAY_LOW levels
            # Runs BEFORE level sweep to prioritize crash bottom FBs.
            # Mancini: "The bigger the sell, the bigger the squeeze."
            # After a crash, the bottom becomes a level. By the time it's
            # confirmed (rally proves significance), price is already above.
            # Retroactively treat the crash as the sweep and current price
            # as recovery. The stop goes below the crash low.
            if self.state == PatternState.IDLE and self.params.allow_deep_sell_recovery:
                self._scan_for_deep_sell_recovery(low, close, level_store, timestamp, bar_idx)
                if self.state != PatternState.IDLE:
                    return self._check_confirmation(bar_idx, timestamp, high, low, close)

            # Path 3: Level Sweep FB — no elevator needed for high-quality levels
            # Prior day low, multi-hour low, and cluster lows are significant
            # enough that a sweep + recovery defines the pattern.
            if self.state == PatternState.IDLE and self.params.allow_level_sweep_fb:
                self._scan_for_level_sweep(low, close, level_store, timestamp, bar_idx)
                if self.state == PatternState.SWEEP_DETECTED:
                    level_price = self._target_level.price
                    if close > level_price:
                        self.state = PatternState.RECOVERY_DETECTED
                        self._recovery_bar = bar_idx
                        self._recovery_price = close
                        recovery_pts = close - level_price
                        if recovery_pts >= self.params.non_acceptance_min_recovery_pts:
                            self.state = PatternState.NON_ACCEPTANCE_WATCH
                            self._hold_bars = 1
                        else:
                            self.state = PatternState.ACCEPTANCE_WATCH
                            self._hold_bars = 1
                        return self._check_confirmation(bar_idx, timestamp, high, low, close)

            # Path 4: Double-dip — no elevator needed if recently stopped at this level
            if self.state == PatternState.IDLE and self._stopped_out_levels:
                self._scan_for_double_dip(low, close, level_store, timestamp, bar_idx)
                if self.state == PatternState.SWEEP_DETECTED:
                    level_price = self._target_level.price
                    if close > level_price:
                        self.state = PatternState.RECOVERY_DETECTED
                        self._recovery_bar = bar_idx
                        self._recovery_price = close
                        recovery_pts = close - level_price
                        if recovery_pts >= self.params.non_acceptance_min_recovery_pts:
                            self.state = PatternState.NON_ACCEPTANCE_WATCH
                            self._hold_bars = 1
                        else:
                            self.state = PatternState.ACCEPTANCE_WATCH
                            self._hold_bars = 1
                        return self._check_confirmation(bar_idx, timestamp, high, low, close)

            return None

        elif self.state == PatternState.SWEEP_DETECTED:
            assert self._target_level is not None
            # Track the sweep low
            if low < self._sweep_low:
                self._sweep_low = low

            # True breakdown abort: if price stays below level for too many bars
            if close < self._target_level.price:
                self._bars_below_level += 1
                if self._bars_below_level >= self.params.true_breakdown_abort_bars:
                    self.reset()
                    return None
            else:
                self._bars_below_level = 0

            # Check for recovery: close back above the level
            level_price = self._target_level.price
            if close > level_price:
                self.state = PatternState.RECOVERY_DETECTED
                self._recovery_bar = bar_idx
                self._recovery_price = close

                # Decide confirmation path
                recovery_pts = close - level_price
                if recovery_pts >= self.params.non_acceptance_min_recovery_pts:
                    self.state = PatternState.NON_ACCEPTANCE_WATCH
                    self._hold_bars = 0
                else:
                    self.state = PatternState.ACCEPTANCE_WATCH
                    self._hold_bars = 0
            return None

        elif self.state == PatternState.RECOVERY_DETECTED:
            # Transition to watching for confirmation
            level_price = self._target_level.price
            recovery_pts = close - level_price
            if recovery_pts >= self.params.non_acceptance_min_recovery_pts:
                self.state = PatternState.NON_ACCEPTANCE_WATCH
                self._hold_bars = 0
            else:
                self.state = PatternState.ACCEPTANCE_WATCH
                self._hold_bars = 0
            return self._check_confirmation(bar_idx, timestamp, high, low, close)

        elif self.state == PatternState.ACCEPTANCE_WATCH:
            return self._check_acceptance(bar_idx, timestamp, high, low, close)

        elif self.state == PatternState.NON_ACCEPTANCE_WATCH:
            return self._check_non_acceptance(bar_idx, timestamp, high, low, close)

        return None

    # Mancini's "significant low" has a precise 3-tier definition:
    # 1. Prior day's low
    # 2. Multi-hour low (20+ pt rally from it)
    # 3. Cluster/shelf of lows
    # SWING_LOW is not a "significant low" — it's just a local minimum
    # that didn't produce a 20+ pt rally. Mancini explicitly says a 10-pt
    # bounce is "not ideal" and "comes shy of the 20 point low required."
    _SIGNIFICANT_LOW_TYPES = frozenset({
        LevelType.PRIOR_DAY_LOW,
        LevelType.MULTI_HOUR_LOW,
        LevelType.CLUSTER_LOW,
        LevelType.INTRADAY_LOW,
    })

    def _scan_for_sweep(
        self,
        low: float,
        close: float,
        level_store: LevelStore,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """Check if current bar sweeps below a significant low."""
        tick = self.params.sweep_min_ticks * 0.25  # convert ticks to points
        confirmed = level_store.get_confirmed(timestamp)

        # Only sweep significant lows (Mancini's 3-tier definition)
        for level in confirmed:
            if level.level_type in self._SIGNIFICANT_LOW_TYPES:
                if low <= level.price - tick:
                    self.state = PatternState.SWEEP_DETECTED
                    self._target_level = level
                    self._sweep_low = low
                    return

    def _scan_for_sweep_with_elevator(
        self,
        low: float,
        close: float,
        level_store: LevelStore,
        timestamp: datetime,
        bar_idx: int,
        elevator_event: ElevatorEvent,
    ) -> None:
        """Check if the elevator event swept a significant low.

        The elevator's low_price is the deepest point of the selloff.
        If that swept below a level, we use it as our sweep low.
        """
        tick = self.params.sweep_min_ticks * 0.25
        confirmed = level_store.get_confirmed(timestamp)
        sweep_low = min(low, elevator_event.low_price)

        for level in confirmed:
            if level.level_type in self._SIGNIFICANT_LOW_TYPES:
                if sweep_low <= level.price - tick:
                    self.state = PatternState.SWEEP_DETECTED
                    self._target_level = level
                    self._sweep_low = sweep_low
                    return

    def _scan_for_level_sweep(
        self,
        low: float,
        close: float,
        level_store: LevelStore,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """Track bars below a high-quality level, trigger sweep after enough time.

        A real failed breakdown means price actually BROKE the level and held
        below for multiple bars before recovering. A single wick below is noise.
        We require min_bars_below closes below the level before the sweep
        is considered real.
        """
        min_depth = self.params.level_sweep_min_depth_pts
        min_bars = self.params.level_sweep_min_bars_below
        confirmed = level_store.get_confirmed(timestamp)

        # If we're already tracking a level, update the count
        if self._sweep_tracking_level is not None:
            level = self._sweep_tracking_level
            if close < level.price:
                # Still below — count it
                self._sweep_tracking_bars_below += 1
                self._sweep_tracking_low = min(self._sweep_tracking_low, low)
            elif close >= level.price and self._sweep_tracking_bars_below >= min_bars:
                # Pre-check: if sweep got too deep during tracking, skip it.
                # Avoids entering SWEEP_DETECTED only to be rejected in _emit_signal,
                # which causes an infinite detect/reject/reset loop.
                final_depth = level.price - self._sweep_tracking_low
                if level.level_type != LevelType.INTRADAY_LOW and final_depth > self.params.max_fb_sweep_depth_pts:
                    self._sweep_tracking_level = None
                    self._sweep_tracking_bars_below = 0
                    self._sweep_tracking_low = float("inf")
                    return
                # Recovery! We've been below long enough — this is a real FB
                self.state = PatternState.SWEEP_DETECTED
                self._target_level = level
                self._sweep_low = self._sweep_tracking_low
                self._is_level_sweep = True
                self._elevator_event = None
                # Clear tracking
                self._sweep_tracking_level = None
                self._sweep_tracking_bars_below = 0
                self._sweep_tracking_low = float("inf")
                return
            else:
                # Recovered too quickly — not a real break, reset
                self._sweep_tracking_level = None
                self._sweep_tracking_bars_below = 0
                self._sweep_tracking_low = float("inf")

        # Look for a new level to track
        if self._sweep_tracking_level is None:
            for level in confirmed:
                if level.level_type in self._HIGH_QUALITY_LEVELS:
                    sweep_depth = level.price - low
                    if sweep_depth >= min_depth and close < level.price:
                        # Pre-check: skip non-INTRADAY_LOW levels that are already
                        # too deep — they'll just get rejected in _emit_signal anyway,
                        # causing an infinite detect/reject/reset loop.
                        if level.level_type != LevelType.INTRADAY_LOW:
                            if sweep_depth > self.params.max_fb_sweep_depth_pts:
                                continue
                        self._sweep_tracking_level = level
                        self._sweep_tracking_bars_below = 1
                        self._sweep_tracking_low = low
                        return

    def _scan_for_double_dip(
        self,
        low: float,
        close: float,
        level_store: LevelStore,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """Check if current bar sweeps a level where we were recently stopped out.

        Double-dip: no elevator required — the level was already proven.
        """
        tick = self.params.sweep_min_ticks * 0.25
        confirmed = level_store.get_confirmed(timestamp)

        for level in confirmed:
            if level.level_type in self._SIGNIFICANT_LOW_TYPES:
                if low <= level.price - tick:
                    if self._is_double_dip_level(level.price, bar_idx):
                        self.state = PatternState.SWEEP_DETECTED
                        self._target_level = level
                        self._sweep_low = low
                        self._is_double_dip = True
                        self._elevator_event = None
                        return

    def _scan_for_deep_sell_recovery(
        self,
        low: float,
        close: float,
        level_store: LevelStore,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """Check for freshly confirmed INTRADAY_LOW levels to FB retroactively.

        When a crash bottom is confirmed (rally proves significance), price is
        already above the level. The crash itself was the "sweep" and the current
        price action is the "recovery". Fast-track to acceptance/non-acceptance.

        Uses _used_intraday_levels to avoid re-triggering on the same level.
        """
        confirmed = level_store.get_confirmed(timestamp)

        for level in confirmed:
            if level.level_type != LevelType.INTRADAY_LOW:
                continue

            # Skip if already used this level (tracked by price proximity)
            if hasattr(self, '_used_intraday_levels'):
                if any(abs(p - level.price) < 1.0 for p in self._used_intraday_levels):
                    continue
            else:
                self._used_intraday_levels = []

            # Must be above the level (already recovered)
            if close <= level.price:
                continue

            # The crash that created this level IS the sweep
            self.state = PatternState.SWEEP_DETECTED
            self._target_level = level
            self._sweep_low = level.price  # crash bottom = sweep low
            self._is_level_sweep = True
            self._elevator_event = None

            # Already recovered — fast-track to confirmation watch
            self.state = PatternState.RECOVERY_DETECTED
            self._recovery_bar = bar_idx
            self._recovery_price = close
            recovery_pts = close - level.price
            if recovery_pts >= self.params.non_acceptance_min_recovery_pts:
                self.state = PatternState.NON_ACCEPTANCE_WATCH
                self._hold_bars = 1
            else:
                self.state = PatternState.ACCEPTANCE_WATCH
                self._hold_bars = 1

            # Mark as used so we don't re-trigger
            self._used_intraday_levels.append(level.price)
            return

    def _check_acceptance(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ) -> Optional[PatternSignal]:
        """Acceptance: price backtests level, dips allowed, returns, holds.

        Mancini Type 1: "price backtests the significant low from below,
        dips, then returns to it." The dip IS part of acceptance — it proves
        no supply. We do NOT reset _hold_bars to 0 on a dip; instead, we
        pause the counter. Only abort if dip exceeds acceptance_max_dip_pts.
        """
        assert self._target_level is not None
        level_price = self._target_level.price

        # Check if price dips too far below level — abort
        dip = level_price - low
        if dip > self.params.acceptance_max_dip_pts:
            self.near_misses.append({
                "timestamp": str(timestamp),
                "bar_idx": bar_idx,
                "level_price": level_price,
                "failure_reason": "dip_too_deep",
                "achieved": {"dip_pts": round(dip, 2)},
                "required": {"max_dip_pts": self.params.acceptance_max_dip_pts},
                "sweep_low": self._sweep_low,
                "close_at_failure": close,
            })
            self.reset()
            return None

        # Mancini: dips below level are EXPECTED during acceptance
        # (Type 1 = backtest-dip-return). Only count bars above level,
        # but do NOT reset count on dips — the dip is part of the process.
        if close >= level_price:
            self._hold_bars += 1

        # Use depth-aware hold requirement
        sweep_depth = self._target_level.price - self._sweep_low
        if sweep_depth >= self.params.shallow_flush_threshold_pts:
            required_hold = self.params.acceptance_min_hold_bars_deep
            timeout = self.params.acceptance_timeout_bars_deep
        else:
            required_hold = self.params.acceptance_min_hold_bars
            timeout = self.params.acceptance_timeout_bars_shallow

        if self._hold_bars >= required_hold:
            return self._emit_signal(
                bar_idx, timestamp, close, ConfirmationType.ACCEPTANCE
            )

        if bar_idx - self._recovery_bar > timeout:
            self.near_misses.append({
                "timestamp": str(timestamp),
                "bar_idx": bar_idx,
                "level_price": level_price,
                "failure_reason": "acceptance_timeout",
                "achieved": {"hold_bars": self._hold_bars},
                "required": {"hold_bars": required_hold},
                "sweep_low": self._sweep_low,
                "close_at_failure": close,
            })
            self.reset()

        return None

    def _check_non_acceptance(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ) -> Optional[PatternSignal]:
        """Non-acceptance: price recovers 5+ pts above level, holds 3+ bars."""
        assert self._target_level is not None
        level_price = self._target_level.price

        recovery = close - level_price
        if recovery >= self.params.non_acceptance_min_recovery_pts:
            self._hold_bars += 1
        else:
            self._hold_bars = 0

        if self._hold_bars >= self.params.non_acceptance_min_hold_bars:
            return self._emit_signal(
                bar_idx, timestamp, close, ConfirmationType.NON_ACCEPTANCE
            )

        # Use depth-aware timeout
        sweep_depth = self._target_level.price - self._sweep_low
        if sweep_depth >= self.params.shallow_flush_threshold_pts:
            timeout = self.params.acceptance_timeout_bars_deep
        else:
            timeout = self.params.acceptance_timeout_bars_shallow

        if bar_idx - self._recovery_bar > timeout:
            self.reset()

        return None

    def _check_confirmation(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ) -> Optional[PatternSignal]:
        """Delegate to the appropriate confirmation check."""
        if self.state == PatternState.ACCEPTANCE_WATCH:
            return self._check_acceptance(bar_idx, timestamp, high, low, close)
        elif self.state == PatternState.NON_ACCEPTANCE_WATCH:
            return self._check_non_acceptance(bar_idx, timestamp, high, low, close)
        return None

    def _emit_signal(
        self,
        bar_idx: int,
        timestamp: datetime,
        entry_price: float,
        confirmation: ConfirmationType,
    ) -> Optional[PatternSignal]:
        """Create and return the pattern signal, then reset.

        Returns None if sweep depth exceeds max_fb_sweep_depth_pts.
        """
        assert self._target_level is not None
        sweep_depth = self._target_level.price - self._sweep_low

        # Reject deep sweeps: likely true breakdowns, not failed breakdowns.
        # Exception: INTRADAY_LOW levels form during the crash itself, so a
        # deeper sweep is expected and normal. Skip the depth filter for these.
        is_intraday = self._target_level.level_type == LevelType.INTRADAY_LOW
        if not is_intraday and sweep_depth > self.params.max_fb_sweep_depth_pts:
            self.near_misses.append({
                "timestamp": str(timestamp),
                "bar_idx": bar_idx,
                "level_price": self._target_level.price,
                "failure_reason": "sweep_too_deep",
                "achieved": {"sweep_depth_pts": round(sweep_depth, 2)},
                "required": {"max_sweep_depth_pts": self.params.max_fb_sweep_depth_pts},
                "sweep_low": self._sweep_low,
                "close_at_failure": entry_price,
            })
            self.reset()
            return None

        # Mancini: "Stops for Failed Breakdowns ALWAYS go below (a few points)
        # the lowest low of the structure." The sweep_low IS the lowest low.
        # For deep sweeps (>threshold), use level-based stop to avoid massive
        # risk. E.g., 50pt sweep → 55pt stop is unreasonable; level - 4.5 = 8pt stop.
        threshold = self.params.deep_sweep_level_stop_threshold_pts
        if threshold > 0 and sweep_depth > threshold:
            stop_price = self._target_level.price - self.params.fb_stop_buffer_pts
        else:
            stop_price = self._sweep_low - self.params.fb_stop_buffer_pts
        signal = PatternSignal(
            pattern_type="failed_breakdown",
            confirmation=confirmation,
            level=self._target_level,
            sweep_low=self._sweep_low,
            sweep_depth_pts=sweep_depth,
            entry_price=entry_price,
            stop_price=stop_price,
            bar_idx=bar_idx,
            timestamp=timestamp,
            elevator_event=self._elevator_event,
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

        sweep_tracking_level = None
        if self._sweep_tracking_level is not None:
            sweep_tracking_level = {
                "price": self._sweep_tracking_level.price,
                "level_type": self._sweep_tracking_level.level_type.name,
                "created_at": self._sweep_tracking_level.created_at.isoformat(),
                "confirmed_at": self._sweep_tracking_level.confirmed_at.isoformat() if self._sweep_tracking_level.confirmed_at else None,
                "touch_count": self._sweep_tracking_level.touch_count,
                "rally_from_low_pts": self._sweep_tracking_level.rally_from_low_pts,
                "is_active": self._sweep_tracking_level.is_active,
                "label": self._sweep_tracking_level.label,
            }

        return {
            "state": self.state.name,
            "target_level": target_level,
            "sweep_low": self._sweep_low,
            "recovery_bar": self._recovery_bar,
            "recovery_price": self._recovery_price,
            "hold_bars": self._hold_bars,
            "bars_below_level": self._bars_below_level,
            "is_double_dip": self._is_double_dip,
            "is_level_sweep": self._is_level_sweep,
            "sweep_tracking_level": sweep_tracking_level,
            "sweep_tracking_bars_below": self._sweep_tracking_bars_below,
            "sweep_tracking_low": self._sweep_tracking_low,
            "stopped_out_levels": self._stopped_out_levels,
            "near_misses": self.near_misses[-10:],
        }

    def restore_state(self, snapshot: dict) -> None:
        """Restore pattern state from a saved snapshot."""
        state_name = snapshot.get("state", "IDLE")
        try:
            self.state = PatternState[state_name]
        except KeyError:
            self.state = PatternState.IDLE
            return

        # Restore target level
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

        self._sweep_low = snapshot.get("sweep_low", float("inf"))
        self._recovery_bar = snapshot.get("recovery_bar", -1)
        self._recovery_price = snapshot.get("recovery_price", 0.0)
        self._hold_bars = snapshot.get("hold_bars", 0)
        self._bars_below_level = snapshot.get("bars_below_level", 0)
        self._is_double_dip = snapshot.get("is_double_dip", False)
        self._is_level_sweep = snapshot.get("is_level_sweep", False)

        # Restore sweep tracking level
        stl = snapshot.get("sweep_tracking_level")
        if stl is not None:
            self._sweep_tracking_level = Level(
                price=stl["price"],
                level_type=LevelType[stl["level_type"]],
                created_at=datetime.fromisoformat(stl["created_at"]),
                confirmed_at=datetime.fromisoformat(stl["confirmed_at"]) if stl.get("confirmed_at") else None,
                touch_count=stl.get("touch_count", 1),
                rally_from_low_pts=stl.get("rally_from_low_pts", 0.0),
                is_active=stl.get("is_active", True),
                label=stl.get("label", ""),
            )
        else:
            self._sweep_tracking_level = None

        self._sweep_tracking_bars_below = snapshot.get("sweep_tracking_bars_below", 0)
        self._sweep_tracking_low = snapshot.get("sweep_tracking_low", float("inf"))
        self._stopped_out_levels = snapshot.get("stopped_out_levels", [])
        self.near_misses = snapshot.get("near_misses", [])


class LevelReclaim:
    """State machine for Level Reclaim detection.

    Sequence:
    1. Horizontal S/R level with multiple touches
    2. Price reclaims the level from below
    3. Confirmation via acceptance or non-acceptance
    """

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params
        self.state = PatternState.IDLE
        self._target_level: Optional[Level] = None
        self._was_below: bool = False
        self._reclaim_bar: int = -1
        self._hold_bars: int = 0

    def reset(self) -> None:
        self.state = PatternState.IDLE
        self._target_level = None
        self._was_below = False
        self._reclaim_bar = -1
        self._hold_bars = 0

    def update(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        level_store: LevelStore,
    ) -> Optional[PatternSignal]:
        """Process one bar. Returns PatternSignal if pattern confirms."""

        if self.state == PatternState.IDLE:
            self._scan_for_reclaim(low, close, level_store, timestamp, bar_idx)
            return None

        elif self.state == PatternState.RECOVERY_DETECTED:
            assert self._target_level is not None
            level_price = self._target_level.price
            recovery = close - level_price

            if recovery >= self.params.non_acceptance_min_recovery_pts:
                self.state = PatternState.NON_ACCEPTANCE_WATCH
                self._hold_bars = 0
            else:
                self.state = PatternState.ACCEPTANCE_WATCH
                self._hold_bars = 0
            return self._check_confirmation(bar_idx, timestamp, high, low, close)

        elif self.state == PatternState.ACCEPTANCE_WATCH:
            return self._check_acceptance(bar_idx, timestamp, high, low, close)

        elif self.state == PatternState.NON_ACCEPTANCE_WATCH:
            return self._check_non_acceptance(bar_idx, timestamp, high, low, close)

        return None

    def _scan_for_reclaim(
        self,
        low: float,
        close: float,
        level_store: LevelStore,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """Check for S/R level reclaimed from below."""
        confirmed = level_store.get_confirmed(timestamp)
        for level in confirmed:
            if level.level_type == LevelType.HORIZONTAL_SR:
                if level.touch_count >= self.params.level_reclaim_min_touches:
                    # Was below, now closing above
                    if low < level.price and close > level.price:
                        self.state = PatternState.RECOVERY_DETECTED
                        self._target_level = level
                        self._reclaim_bar = bar_idx
                        self._hold_bars = 0
                        return

    def _check_acceptance(
        self, bar_idx: int, timestamp: datetime,
        high: float, low: float, close: float,
    ) -> Optional[PatternSignal]:
        assert self._target_level is not None
        level_price = self._target_level.price

        dip = level_price - low
        if dip > self.params.acceptance_max_dip_pts:
            self.reset()
            return None

        if close >= level_price:
            self._hold_bars += 1
        else:
            self._hold_bars = 0

        if self._hold_bars >= self.params.acceptance_min_hold_bars:
            return self._emit_signal(
                bar_idx, timestamp, close, ConfirmationType.ACCEPTANCE
            )

        if bar_idx - self._reclaim_bar > self.params.acceptance_timeout_bars_shallow:
            self.reset()

        return None

    def _check_non_acceptance(
        self, bar_idx: int, timestamp: datetime,
        high: float, low: float, close: float,
    ) -> Optional[PatternSignal]:
        assert self._target_level is not None
        level_price = self._target_level.price

        recovery = close - level_price
        if recovery >= self.params.non_acceptance_min_recovery_pts:
            self._hold_bars += 1
        else:
            self._hold_bars = 0

        if self._hold_bars >= self.params.non_acceptance_min_hold_bars:
            return self._emit_signal(
                bar_idx, timestamp, close, ConfirmationType.NON_ACCEPTANCE
            )

        if bar_idx - self._reclaim_bar > self.params.acceptance_timeout_bars_shallow:
            self.reset()

        return None

    def _check_confirmation(
        self, bar_idx: int, timestamp: datetime,
        high: float, low: float, close: float,
    ) -> Optional[PatternSignal]:
        if self.state == PatternState.ACCEPTANCE_WATCH:
            return self._check_acceptance(bar_idx, timestamp, high, low, close)
        elif self.state == PatternState.NON_ACCEPTANCE_WATCH:
            return self._check_non_acceptance(bar_idx, timestamp, high, low, close)
        return None

    def _emit_signal(
        self, bar_idx: int, timestamp: datetime,
        entry_price: float, confirmation: ConfirmationType,
    ) -> PatternSignal:
        assert self._target_level is not None
        signal = PatternSignal(
            pattern_type="level_reclaim",
            confirmation=confirmation,
            level=self._target_level,
            sweep_low=self._target_level.price,
            sweep_depth_pts=0.0,
            entry_price=entry_price,
            stop_price=self._target_level.price - self.params.lr_stop_buffer_pts,
            bar_idx=bar_idx,
            timestamp=timestamp,
        )
        self.reset()
        return signal
