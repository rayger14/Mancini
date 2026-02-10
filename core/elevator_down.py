"""Elevator Down (sharp selloff) detection.

Detects the characteristic rapid selloff that precedes a Failed Breakdown setup:
- Velocity >= 2 pts/min over 5-bar rolling window
- At least 2 support levels broken
- Completion: velocity drops and price makes a higher low
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd

from config.levels import LevelStore
from config.settings import ElevatorParams, DEFAULT_ELEVATOR


class ElevatorState(Enum):
    """State machine for elevator down detection."""

    IDLE = auto()       # No active selloff
    ACTIVE = auto()     # Selloff in progress
    COMPLETE = auto()   # Selloff ended, watching for setup


@dataclass
class ElevatorEvent:
    """Record of a detected elevator down event."""

    start_idx: int
    start_time: datetime
    start_price: float
    end_idx: Optional[int] = None
    end_time: Optional[datetime] = None
    end_price: Optional[float] = None
    low_price: float = float("inf")
    low_idx: Optional[int] = None
    peak_velocity: float = 0.0
    levels_broken: int = 0

    @property
    def total_drop_pts(self) -> float:
        return self.start_price - self.low_price

    @property
    def is_complete(self) -> bool:
        return self.end_idx is not None


class ElevatorDownDetector:
    """Detects sharp selloffs (elevator down) in price data."""

    def __init__(self, params: ElevatorParams = DEFAULT_ELEVATOR):
        self.params = params
        self.state = ElevatorState.IDLE
        self.current_event: Optional[ElevatorEvent] = None
        self.completed_events: list[ElevatorEvent] = []
        self._recent_low: float = float("inf")
        self._recent_low_idx: int = -1

    def reset(self) -> None:
        """Reset detector state for a new session."""
        self.state = ElevatorState.IDLE
        self.current_event = None
        self.completed_events.clear()
        self._recent_low = float("inf")
        self._recent_low_idx = -1

    def update(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        velocity: float,
        level_store: LevelStore,
    ) -> Optional[ElevatorEvent]:
        """Process one bar and return a completed ElevatorEvent if the elevator just ended.

        Parameters
        ----------
        bar_idx : int
            Current bar index.
        timestamp : datetime
            Bar timestamp.
        high, low, close : float
            Bar OHLC (open not needed).
        velocity : float
            Pre-computed price velocity (pts/bar, negative = selling).
        level_store : LevelStore
            Current support levels for counting breaks.

        Returns
        -------
        ElevatorEvent or None
            Returns the event only when it transitions from ACTIVE → COMPLETE.
        """
        abs_velocity = abs(velocity) if velocity < 0 else 0.0

        if self.state == ElevatorState.IDLE:
            # Check if a selloff is starting
            if abs_velocity >= self.params.min_velocity_pts_per_min:
                self.state = ElevatorState.ACTIVE
                self.current_event = ElevatorEvent(
                    start_idx=bar_idx,
                    start_time=timestamp,
                    start_price=close,
                    low_price=low,
                    low_idx=bar_idx,
                    peak_velocity=abs_velocity,
                )
                self._count_broken_levels(low, level_store, timestamp)
            return None

        elif self.state == ElevatorState.ACTIVE:
            assert self.current_event is not None

            # Update tracking
            if abs_velocity > self.current_event.peak_velocity:
                self.current_event.peak_velocity = abs_velocity

            if low < self.current_event.low_price:
                self.current_event.low_price = low
                self.current_event.low_idx = bar_idx
                self._count_broken_levels(low, level_store, timestamp)

            # Check for completion: velocity drops + higher low forming
            velocity_dropped = (
                abs_velocity
                < self.current_event.peak_velocity * self.params.completion_velocity_ratio
            )
            higher_low = low > self.current_event.low_price

            if velocity_dropped and higher_low:
                # Confirm higher low with lookback
                if self._confirm_higher_low(bar_idx):
                    return self._complete_event(bar_idx, timestamp, close)

            # Also complete if velocity has fully stalled
            if abs_velocity == 0 and bar_idx - self.current_event.low_idx >= 3:
                return self._complete_event(bar_idx, timestamp, close)

            return None

        else:  # COMPLETE
            # Stay in complete state until reset or new elevator starts
            if abs_velocity >= self.params.min_velocity_pts_per_min:
                # New elevator starting
                self.state = ElevatorState.ACTIVE
                self.current_event = ElevatorEvent(
                    start_idx=bar_idx,
                    start_time=timestamp,
                    start_price=close,
                    low_price=low,
                    low_idx=bar_idx,
                    peak_velocity=abs_velocity,
                )
                self._count_broken_levels(low, level_store, timestamp)
            return None

    def is_active(self) -> bool:
        """True if an elevator down is currently in progress."""
        return self.state == ElevatorState.ACTIVE

    def is_complete(self) -> bool:
        """True if an elevator recently completed (setup window open)."""
        return self.state == ElevatorState.COMPLETE

    def last_completed(self) -> Optional[ElevatorEvent]:
        """Return the most recently completed elevator event."""
        return self.completed_events[-1] if self.completed_events else None

    # ------------------------------------------------------------------
    # Batch detection (for backtest)
    # ------------------------------------------------------------------

    def detect_all(
        self,
        df: pd.DataFrame,
        velocity: pd.Series,
        level_store: LevelStore,
    ) -> list[ElevatorEvent]:
        """Run detection across entire DataFrame (not bar-by-bar).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV bars.
        velocity : pd.Series
            Pre-computed velocity series (aligned with df).
        level_store : LevelStore
            Levels for support-break counting.

        Returns
        -------
        list[ElevatorEvent]
        """
        self.reset()
        events: list[ElevatorEvent] = []

        for i in range(len(df)):
            result = self.update(
                bar_idx=i,
                timestamp=df.index[i],
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                velocity=float(velocity.iat[i]) if not np.isnan(velocity.iat[i]) else 0.0,
                level_store=level_store,
            )
            if result is not None:
                events.append(result)

        return events

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _complete_event(
        self, bar_idx: int, timestamp: datetime, close: float
    ) -> ElevatorEvent:
        """Mark current event as complete."""
        assert self.current_event is not None
        self.current_event.end_idx = bar_idx
        self.current_event.end_time = timestamp
        self.current_event.end_price = close
        self.state = ElevatorState.COMPLETE
        self.completed_events.append(self.current_event)
        event = self.current_event
        self.current_event = None
        return event

    def _count_broken_levels(
        self, price: float, level_store: LevelStore, as_of: datetime
    ) -> None:
        """Count how many support levels have been broken."""
        if self.current_event is None:
            return
        confirmed = level_store.get_confirmed(as_of)
        broken = sum(1 for l in confirmed if l.price > price)
        self.current_event.levels_broken = max(
            self.current_event.levels_broken, broken
        )

    def _confirm_higher_low(self, bar_idx: int) -> bool:
        """Check if we have a confirmed higher low."""
        if self.current_event is None or self.current_event.low_idx is None:
            return False
        bars_since_low = bar_idx - self.current_event.low_idx
        return bars_since_low >= self.params.higher_low_lookback
