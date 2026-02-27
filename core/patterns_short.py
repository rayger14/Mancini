"""Failed Rally & Level Rejection state machines (short-side mirrors).

Failed Rally sequence:
  Elevator Up -> Significant High Swept -> Recovery Below -> Confirmation -> Short Signal

Level Rejection sequence:
  Horizontal S/R Rejected from Above -> Confirmation -> Short Signal
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams, DEFAULT_STRATEGY
from core.elevator_up import ElevatorUpEvent
from core.patterns import PatternState, ConfirmationType, PatternSignal


# High-quality resistance level types (mirror of FB's _HIGH_QUALITY_LEVELS)
_HIGH_QUALITY_RESISTANCE = frozenset({
    LevelType.PRIOR_DAY_HIGH,
    LevelType.MULTI_HOUR_HIGH,
    LevelType.CLUSTER_HIGH,
})

# All resistance types that can trigger a Failed Rally
_RESISTANCE_TYPES = frozenset({
    LevelType.PRIOR_DAY_HIGH,
    LevelType.MULTI_HOUR_HIGH,
    LevelType.CLUSTER_HIGH,
    LevelType.SWING_HIGH,
})


class FailedRally:
    """State machine for Failed Rally detection (short-side mirror of FailedBreakdown).

    Three entry paths (mirror of FB):
    1. Elevator FR — fast rally sweeps a significant high, then recovers below
    2. Level Sweep FR — price sweeps above a high-quality resistance without
       needing a fast elevator
    3. Double-dip — re-entry without elevator at a level where we were
       recently stopped out
    """

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params
        self.state = PatternState.IDLE
        self._target_level: Optional[Level] = None
        self._sweep_high: float = float("-inf")
        self._recovery_bar: int = -1
        self._recovery_price: float = 0.0
        self._hold_bars: int = 0
        self._elevator_event: Optional[ElevatorUpEvent] = None
        self._bars_above_level: int = 0
        self._stopped_out_levels: list[tuple[float, int]] = []
        self._double_dip_cooldown_bars: int = 60
        self._is_double_dip: bool = False
        self._is_level_sweep: bool = False
        self._sweep_tracking_level: Optional[Level] = None
        self._sweep_tracking_bars_above: int = 0
        self._sweep_tracking_high: float = float("-inf")

    def reset(self) -> None:
        self.state = PatternState.IDLE
        self._target_level = None
        self._sweep_high = float("-inf")
        self._recovery_bar = -1
        self._recovery_price = 0.0
        self._hold_bars = 0
        self._elevator_event = None
        self._bars_above_level = 0
        self._is_double_dip = False
        self._is_level_sweep = False
        self._sweep_tracking_level = None
        self._sweep_tracking_bars_above = 0
        self._sweep_tracking_high = float("-inf")

    def record_stop_out(self, level_price: float, bar_idx: int) -> None:
        self._stopped_out_levels.append((level_price, bar_idx))

    def _is_double_dip_level(self, level_price: float, bar_idx: int) -> bool:
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
        elevator_event: Optional[ElevatorUpEvent] = None,
    ) -> Optional[PatternSignal]:
        if self.state == PatternState.IDLE:
            # Clean up expired stop-out records
            self._stopped_out_levels = [
                (p, b) for p, b in self._stopped_out_levels
                if bar_idx - b <= self._double_dip_cooldown_bars
            ]

            # Path 1: Normal FR — need a completed elevator up event
            if elevator_event is not None and elevator_event.is_complete:
                self._elevator_event = elevator_event
                self._is_double_dip = False
                self._is_level_sweep = False
                self._sweep_tracking_level = None
                self._sweep_tracking_bars_above = 0
                self._sweep_tracking_high = float("-inf")
                self._scan_for_sweep_with_elevator(
                    high, close, level_store, timestamp, bar_idx, elevator_event
                )
                if self.state == PatternState.SWEEP_DETECTED:
                    level_price = self._target_level.price
                    if close < level_price:
                        self.state = PatternState.RECOVERY_DETECTED
                        self._recovery_bar = bar_idx
                        self._recovery_price = close
                        drop_pts = level_price - close
                        if drop_pts >= self.params.non_acceptance_min_recovery_pts:
                            self.state = PatternState.NON_ACCEPTANCE_WATCH
                            self._hold_bars = 1
                        else:
                            self.state = PatternState.ACCEPTANCE_WATCH
                            self._hold_bars = 1
                        return self._check_confirmation(bar_idx, timestamp, high, low, close)

            # Path 2: Level Sweep FR
            if self.state == PatternState.IDLE and self.params.allow_level_sweep_fb:
                self._scan_for_level_sweep(high, close, level_store, timestamp, bar_idx)
                if self.state == PatternState.SWEEP_DETECTED:
                    level_price = self._target_level.price
                    if close < level_price:
                        self.state = PatternState.RECOVERY_DETECTED
                        self._recovery_bar = bar_idx
                        self._recovery_price = close
                        drop_pts = level_price - close
                        if drop_pts >= self.params.non_acceptance_min_recovery_pts:
                            self.state = PatternState.NON_ACCEPTANCE_WATCH
                            self._hold_bars = 1
                        else:
                            self.state = PatternState.ACCEPTANCE_WATCH
                            self._hold_bars = 1
                        return self._check_confirmation(bar_idx, timestamp, high, low, close)

            # Path 3: Double-dip
            if self.state == PatternState.IDLE and self._stopped_out_levels:
                self._scan_for_double_dip(high, close, level_store, timestamp, bar_idx)
                if self.state == PatternState.SWEEP_DETECTED:
                    level_price = self._target_level.price
                    if close < level_price:
                        self.state = PatternState.RECOVERY_DETECTED
                        self._recovery_bar = bar_idx
                        self._recovery_price = close
                        drop_pts = level_price - close
                        if drop_pts >= self.params.non_acceptance_min_recovery_pts:
                            self.state = PatternState.NON_ACCEPTANCE_WATCH
                            self._hold_bars = 1
                        else:
                            self.state = PatternState.ACCEPTANCE_WATCH
                            self._hold_bars = 1
                        return self._check_confirmation(bar_idx, timestamp, high, low, close)

            return None

        elif self.state == PatternState.SWEEP_DETECTED:
            assert self._target_level is not None
            if high > self._sweep_high:
                self._sweep_high = high

            # True rally abort: stays above level too long (short-specific param)
            if close > self._target_level.price:
                self._bars_above_level += 1
                if self._bars_above_level >= self.params.short_true_rally_abort_bars:
                    self.reset()
                    return None
            else:
                self._bars_above_level = 0

            # Check for recovery: close back below the level
            level_price = self._target_level.price
            if close < level_price:
                self.state = PatternState.RECOVERY_DETECTED
                self._recovery_bar = bar_idx
                self._recovery_price = close
                drop_pts = level_price - close
                if drop_pts >= self.params.non_acceptance_min_recovery_pts:
                    self.state = PatternState.NON_ACCEPTANCE_WATCH
                    self._hold_bars = 0
                else:
                    self.state = PatternState.ACCEPTANCE_WATCH
                    self._hold_bars = 0
            return None

        elif self.state == PatternState.RECOVERY_DETECTED:
            level_price = self._target_level.price
            drop_pts = level_price - close
            if drop_pts >= self.params.non_acceptance_min_recovery_pts:
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

    # ------------------------------------------------------------------
    # Sweep scanners (mirror of FB's scanners, flipped direction)
    # ------------------------------------------------------------------

    def _scan_for_sweep_with_elevator(
        self, high: float, close: float, level_store: LevelStore,
        timestamp: datetime, bar_idx: int, elevator_event: ElevatorUpEvent,
    ) -> None:
        tick = self.params.sweep_min_ticks * 0.25
        confirmed = level_store.get_confirmed(timestamp)
        sweep_high = max(high, elevator_event.high_price)

        for level in confirmed:
            if level.level_type in _RESISTANCE_TYPES:
                if sweep_high >= level.price + tick:
                    self.state = PatternState.SWEEP_DETECTED
                    self._target_level = level
                    self._sweep_high = sweep_high
                    return

    def _scan_for_level_sweep(
        self, high: float, close: float, level_store: LevelStore,
        timestamp: datetime, bar_idx: int,
    ) -> None:
        min_depth = self.params.level_sweep_min_depth_pts
        min_bars = self.params.level_sweep_min_bars_below
        confirmed = level_store.get_confirmed(timestamp)

        if self._sweep_tracking_level is not None:
            level = self._sweep_tracking_level
            if close > level.price:
                self._sweep_tracking_bars_above += 1
                self._sweep_tracking_high = max(self._sweep_tracking_high, high)
            elif close <= level.price and self._sweep_tracking_bars_above >= min_bars:
                # Recovery below! Real failed rally.
                self.state = PatternState.SWEEP_DETECTED
                self._target_level = level
                self._sweep_high = self._sweep_tracking_high
                self._is_level_sweep = True
                self._elevator_event = None
                self._sweep_tracking_level = None
                self._sweep_tracking_bars_above = 0
                self._sweep_tracking_high = float("-inf")
                return
            else:
                self._sweep_tracking_level = None
                self._sweep_tracking_bars_above = 0
                self._sweep_tracking_high = float("-inf")

        if self._sweep_tracking_level is None:
            for level in confirmed:
                if level.level_type in _HIGH_QUALITY_RESISTANCE:
                    sweep_depth = high - level.price
                    if sweep_depth >= min_depth and close > level.price:
                        self._sweep_tracking_level = level
                        self._sweep_tracking_bars_above = 1
                        self._sweep_tracking_high = high
                        return

    def _scan_for_double_dip(
        self, high: float, close: float, level_store: LevelStore,
        timestamp: datetime, bar_idx: int,
    ) -> None:
        tick = self.params.sweep_min_ticks * 0.25
        confirmed = level_store.get_confirmed(timestamp)

        for level in confirmed:
            if level.level_type in _RESISTANCE_TYPES:
                if high >= level.price + tick:
                    if self._is_double_dip_level(level.price, bar_idx):
                        self.state = PatternState.SWEEP_DETECTED
                        self._target_level = level
                        self._sweep_high = high
                        self._is_double_dip = True
                        self._elevator_event = None
                        return

    # ------------------------------------------------------------------
    # Confirmation (mirror: price must stay BELOW level)
    # ------------------------------------------------------------------

    def _check_acceptance(
        self, bar_idx: int, timestamp: datetime,
        high: float, low: float, close: float,
    ) -> Optional[PatternSignal]:
        assert self._target_level is not None
        level_price = self._target_level.price

        # Spike too far above level — abort (uses short-specific param)
        spike = high - level_price
        if spike > self.params.short_acceptance_max_dip_pts:
            self.reset()
            return None

        # Must stay below level
        if close <= level_price:
            self._hold_bars += 1
        else:
            self._hold_bars = 0

        sweep_depth = self._sweep_high - self._target_level.price
        if sweep_depth >= self.params.shallow_flush_threshold_pts:
            required_hold = self.params.short_acceptance_min_hold_bars_deep
            timeout = self.params.acceptance_timeout_bars_deep
        else:
            required_hold = self.params.short_acceptance_min_hold_bars
            timeout = self.params.acceptance_timeout_bars_shallow

        if self._hold_bars >= required_hold:
            return self._emit_signal(
                bar_idx, timestamp, close, ConfirmationType.ACCEPTANCE
            )

        if bar_idx - self._recovery_bar > timeout:
            self.reset()

        return None

    def _check_non_acceptance(
        self, bar_idx: int, timestamp: datetime,
        high: float, low: float, close: float,
    ) -> Optional[PatternSignal]:
        assert self._target_level is not None
        level_price = self._target_level.price

        drop = level_price - close
        if drop >= self.params.non_acceptance_min_recovery_pts:
            self._hold_bars += 1
        else:
            self._hold_bars = 0

        if self._hold_bars >= self.params.non_acceptance_min_hold_bars:
            return self._emit_signal(
                bar_idx, timestamp, close, ConfirmationType.NON_ACCEPTANCE
            )

        sweep_depth = self._sweep_high - self._target_level.price
        if sweep_depth >= self.params.shallow_flush_threshold_pts:
            timeout = self.params.acceptance_timeout_bars_deep
        else:
            timeout = self.params.acceptance_timeout_bars_shallow

        if bar_idx - self._recovery_bar > timeout:
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
        sweep_depth = self._sweep_high - self._target_level.price
        # Reject deep sweeps (mirrored from long-side max_fb_sweep_depth_pts)
        if sweep_depth > self.params.short_max_fr_sweep_depth_pts:
            self.reset()
            return None  # type: ignore[return-value]
        stop_price = self._target_level.price + self.params.fr_stop_buffer_pts
        signal = PatternSignal(
            pattern_type="failed_rally",
            confirmation=confirmation,
            level=self._target_level,
            sweep_low=0.0,
            sweep_high=self._sweep_high,
            sweep_depth_pts=sweep_depth,
            entry_price=entry_price,
            stop_price=stop_price,
            bar_idx=bar_idx,
            timestamp=timestamp,
            elevator_event=self._elevator_event,
            direction="short",
        )
        self.reset()
        return signal


class LevelRejection:
    """State machine for Level Rejection detection (short-side mirror of LevelReclaim).

    Sequence:
    1. Horizontal S/R level with multiple touches
    2. Price rejects the level from above (was above, closes below)
    3. Confirmation via acceptance or non-acceptance
    """

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params
        self.state = PatternState.IDLE
        self._target_level: Optional[Level] = None
        self._was_above: bool = False
        self._rejection_bar: int = -1
        self._hold_bars: int = 0

    def reset(self) -> None:
        self.state = PatternState.IDLE
        self._target_level = None
        self._was_above = False
        self._rejection_bar = -1
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
        if self.state == PatternState.IDLE:
            self._scan_for_rejection(high, close, level_store, timestamp, bar_idx)
            return None

        elif self.state == PatternState.RECOVERY_DETECTED:
            assert self._target_level is not None
            level_price = self._target_level.price
            drop = level_price - close
            if drop >= self.params.non_acceptance_min_recovery_pts:
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

    def _scan_for_rejection(
        self, high: float, close: float, level_store: LevelStore,
        timestamp: datetime, bar_idx: int,
    ) -> None:
        """Check for S/R level rejected from above (was above, now closing below)."""
        confirmed = level_store.get_confirmed(timestamp)
        for level in confirmed:
            if level.level_type == LevelType.HORIZONTAL_SR:
                if level.touch_count >= self.params.level_reclaim_min_touches:
                    # Was above, now closing below
                    if high > level.price and close < level.price:
                        self.state = PatternState.RECOVERY_DETECTED
                        self._target_level = level
                        self._rejection_bar = bar_idx
                        self._hold_bars = 0
                        return

    def _check_acceptance(
        self, bar_idx: int, timestamp: datetime,
        high: float, low: float, close: float,
    ) -> Optional[PatternSignal]:
        assert self._target_level is not None
        level_price = self._target_level.price

        # Use short-specific acceptance params
        spike = high - level_price
        if spike > self.params.short_acceptance_max_dip_pts:
            self.reset()
            return None

        if close <= level_price:
            self._hold_bars += 1
        else:
            self._hold_bars = 0

        if self._hold_bars >= self.params.short_acceptance_min_hold_bars:
            return self._emit_signal(
                bar_idx, timestamp, close, ConfirmationType.ACCEPTANCE
            )

        if bar_idx - self._rejection_bar > self.params.acceptance_timeout_bars_shallow:
            self.reset()

        return None

    def _check_non_acceptance(
        self, bar_idx: int, timestamp: datetime,
        high: float, low: float, close: float,
    ) -> Optional[PatternSignal]:
        assert self._target_level is not None
        level_price = self._target_level.price

        drop = level_price - close
        if drop >= self.params.non_acceptance_min_recovery_pts:
            self._hold_bars += 1
        else:
            self._hold_bars = 0

        if self._hold_bars >= self.params.non_acceptance_min_hold_bars:
            return self._emit_signal(
                bar_idx, timestamp, close, ConfirmationType.NON_ACCEPTANCE
            )

        if bar_idx - self._rejection_bar > self.params.acceptance_timeout_bars_shallow:
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
            pattern_type="level_rejection",
            confirmation=confirmation,
            level=self._target_level,
            sweep_low=0.0,
            sweep_high=self._target_level.price,
            sweep_depth_pts=0.0,
            entry_price=entry_price,
            stop_price=self._target_level.price + self.params.lj_stop_buffer_pts,
            bar_idx=bar_idx,
            timestamp=timestamp,
            direction="short",
        )
        self.reset()
        return signal
