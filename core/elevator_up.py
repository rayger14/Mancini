"""Elevator Up (sharp rally) detection — mirror of Elevator Down.

Detects the characteristic rapid rally that precedes a Failed Rally setup:
- Positive velocity >= min_velocity over 5-bar rolling window
- At least 2 resistance levels broken
- Completion: velocity drops and price makes a lower high
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


class ElevatorUpState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    COMPLETE = auto()


@dataclass
class ElevatorUpEvent:
    """Record of a detected elevator up event."""

    start_idx: int
    start_time: datetime
    start_price: float
    end_idx: Optional[int] = None
    end_time: Optional[datetime] = None
    end_price: Optional[float] = None
    high_price: float = float("-inf")
    high_idx: Optional[int] = None
    peak_velocity: float = 0.0
    levels_broken: int = 0

    @property
    def total_rally_pts(self) -> float:
        return self.high_price - self.start_price

    @property
    def is_complete(self) -> bool:
        return self.end_idx is not None


class ElevatorUpDetector:
    """Detects sharp rallies (elevator up) in price data."""

    def __init__(self, params: ElevatorParams = DEFAULT_ELEVATOR):
        self.params = params
        self.state = ElevatorUpState.IDLE
        self.current_event: Optional[ElevatorUpEvent] = None
        self.completed_events: list[ElevatorUpEvent] = []

    def reset(self) -> None:
        self.state = ElevatorUpState.IDLE
        self.current_event = None
        self.completed_events.clear()

    def update(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        velocity: float,
        level_store: LevelStore,
    ) -> Optional[ElevatorUpEvent]:
        """Process one bar. Returns completed event when rally ends.

        velocity > 0 means buying pressure (positive = rally).
        """
        pos_velocity = velocity if velocity > 0 else 0.0

        if self.state == ElevatorUpState.IDLE:
            if pos_velocity >= self.params.min_velocity_pts_per_min:
                self.state = ElevatorUpState.ACTIVE
                self.current_event = ElevatorUpEvent(
                    start_idx=bar_idx,
                    start_time=timestamp,
                    start_price=close,
                    high_price=high,
                    high_idx=bar_idx,
                    peak_velocity=pos_velocity,
                )
                self._count_broken_levels(high, level_store, timestamp)
            return None

        elif self.state == ElevatorUpState.ACTIVE:
            assert self.current_event is not None

            if pos_velocity > self.current_event.peak_velocity:
                self.current_event.peak_velocity = pos_velocity

            if high > self.current_event.high_price:
                self.current_event.high_price = high
                self.current_event.high_idx = bar_idx
                self._count_broken_levels(high, level_store, timestamp)

            # Completion: velocity drops + lower high forming
            velocity_dropped = (
                pos_velocity
                < self.current_event.peak_velocity * self.params.completion_velocity_ratio
            )
            lower_high = high < self.current_event.high_price

            if velocity_dropped and lower_high:
                if self._confirm_lower_high(bar_idx):
                    return self._complete_event(bar_idx, timestamp, close)

            if pos_velocity == 0 and bar_idx - self.current_event.high_idx >= 3:
                return self._complete_event(bar_idx, timestamp, close)

            return None

        else:  # COMPLETE
            if pos_velocity >= self.params.min_velocity_pts_per_min:
                self.state = ElevatorUpState.ACTIVE
                self.current_event = ElevatorUpEvent(
                    start_idx=bar_idx,
                    start_time=timestamp,
                    start_price=close,
                    high_price=high,
                    high_idx=bar_idx,
                    peak_velocity=pos_velocity,
                )
                self._count_broken_levels(high, level_store, timestamp)
            return None

    def is_active(self) -> bool:
        return self.state == ElevatorUpState.ACTIVE

    def is_complete(self) -> bool:
        return self.state == ElevatorUpState.COMPLETE

    def last_completed(self) -> Optional[ElevatorUpEvent]:
        return self.completed_events[-1] if self.completed_events else None

    def _complete_event(
        self, bar_idx: int, timestamp: datetime, close: float
    ) -> ElevatorUpEvent:
        assert self.current_event is not None
        self.current_event.end_idx = bar_idx
        self.current_event.end_time = timestamp
        self.current_event.end_price = close
        self.state = ElevatorUpState.COMPLETE
        self.completed_events.append(self.current_event)
        event = self.current_event
        self.current_event = None
        return event

    def _count_broken_levels(
        self, price: float, level_store: LevelStore, as_of: datetime
    ) -> None:
        """Count how many resistance levels have been broken (price above them)."""
        if self.current_event is None:
            return
        confirmed = level_store.get_confirmed(as_of)
        broken = sum(1 for l in confirmed if l.price < price)
        self.current_event.levels_broken = max(
            self.current_event.levels_broken, broken
        )

    def _confirm_lower_high(self, bar_idx: int) -> bool:
        if self.current_event is None or self.current_event.high_idx is None:
            return False
        bars_since_high = bar_idx - self.current_event.high_idx
        return bars_since_high >= self.params.higher_low_lookback
