"""Failed Breakdown & Level Reclaim state machines.

Failed Breakdown sequence:
  Elevator Down → Significant Low Swept → Recovery → Confirmation → Signal

Level Reclaim sequence:
  Horizontal S/R Reclaimed from Below → Confirmation → Signal
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    pattern_type: str  # "failed_breakdown" or "level_reclaim"
    confirmation: ConfirmationType
    level: Level
    sweep_low: float  # lowest price during the sweep
    entry_price: float  # confirmation price (entry point)
    stop_price: float  # below sweep low
    bar_idx: int
    timestamp: datetime
    sweep_depth_pts: float = 0.0  # how far below the level price swept
    elevator_event: Optional[ElevatorEvent] = None

    @property
    def risk_pts(self) -> float:
        return self.entry_price - self.stop_price


class FailedBreakdown:
    """State machine for Failed Breakdown detection.

    Sequence:
    1. Elevator Down completes
    2. Price sweeps below a significant low (by at least 1 tick)
    3. Price recovers above the level
    4. Confirmation via acceptance or non-acceptance protocol
    """

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

    def reset(self) -> None:
        self.state = PatternState.IDLE
        self._target_level = None
        self._sweep_low = float("inf")
        self._recovery_bar = -1
        self._recovery_price = 0.0
        self._hold_bars = 0
        self._elevator_event = None
        self._bars_below_level = 0

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
        if self.state == PatternState.IDLE:
            # Need a completed elevator event to start looking
            if elevator_event is not None and elevator_event.is_complete:
                self._elevator_event = elevator_event
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

        # Look for levels that have been swept (low broke below them)
        for level in confirmed:
            if level.level_type in (
                LevelType.PRIOR_DAY_LOW,
                LevelType.MULTI_HOUR_LOW,
                LevelType.CLUSTER_LOW,
                LevelType.SWING_LOW,
            ):
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
            if level.level_type in (
                LevelType.PRIOR_DAY_LOW,
                LevelType.MULTI_HOUR_LOW,
                LevelType.CLUSTER_LOW,
                LevelType.SWING_LOW,
            ):
                if sweep_low <= level.price - tick:
                    self.state = PatternState.SWEEP_DETECTED
                    self._target_level = level
                    self._sweep_low = sweep_low
                    return

    def _check_acceptance(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ) -> Optional[PatternSignal]:
        """Acceptance: price backtests level, dips allowed, returns, holds."""
        assert self._target_level is not None
        level_price = self._target_level.price

        # Check if price dips too far below level
        dip = level_price - low
        if dip > self.params.acceptance_max_dip_pts:
            self.reset()
            return None

        # Price must stay above (or within tolerance of) the level
        if close >= level_price:
            self._hold_bars += 1
        else:
            self._hold_bars = 0

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
    ) -> PatternSignal:
        """Create and return the pattern signal, then reset."""
        assert self._target_level is not None
        sweep_depth = self._target_level.price - self._sweep_low
        stop_buffer = 5.0  # points below sweep low
        signal = PatternSignal(
            pattern_type="failed_breakdown",
            confirmation=confirmation,
            level=self._target_level,
            sweep_low=self._sweep_low,
            sweep_depth_pts=sweep_depth,
            entry_price=entry_price,
            stop_price=self._sweep_low - stop_buffer,
            bar_idx=bar_idx,
            timestamp=timestamp,
            elevator_event=self._elevator_event,
        )
        self.reset()
        return signal


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
        stop_buffer = 5.0
        signal = PatternSignal(
            pattern_type="level_reclaim",
            confirmation=confirmation,
            level=self._target_level,
            sweep_low=self._target_level.price,
            sweep_depth_pts=0.0,
            entry_price=entry_price,
            stop_price=self._target_level.price - stop_buffer,
            bar_idx=bar_idx,
            timestamp=timestamp,
        )
        self.reset()
        return signal
