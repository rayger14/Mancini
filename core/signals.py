"""Signal aggregation from all pattern detectors + R:R calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd

from config.levels import LevelStore
from config.settings import (
    StrategyParams,
    ElevatorParams,
    ExitParams,
    DEFAULT_STRATEGY,
    DEFAULT_ELEVATOR,
    DEFAULT_EXIT,
)
from core.elevator_down import ElevatorDownDetector, ElevatorEvent
from core.indicators import compute_velocity
from core.patterns import (
    FailedBreakdown,
    LevelReclaim,
    PatternSignal,
)
from core.price_levels import PriceLevelDetector


class SignalType(Enum):
    """Signal classification."""

    FAILED_BREAKDOWN = auto()
    LEVEL_RECLAIM = auto()


@dataclass
class Signal:
    """A fully qualified trade signal with targets and R:R."""

    signal_type: SignalType
    pattern: PatternSignal
    target_1: float  # R1 (first resistance above entry)
    target_2: float  # R2 (second resistance above entry)
    risk_pts: float
    reward_t1_pts: float
    reward_t2_pts: float
    rr_ratio_t1: float
    rr_ratio_t2: float
    bar_idx: int
    timestamp: datetime

    @property
    def entry_price(self) -> float:
        return self.pattern.entry_price

    @property
    def stop_price(self) -> float:
        return self.pattern.stop_price


class SignalAggregator:
    """Orchestrates pattern detection and produces qualified Signals.

    Runs all detectors bar-by-bar and filters for minimum R:R.
    """

    def __init__(
        self,
        strategy_params: StrategyParams = DEFAULT_STRATEGY,
        elevator_params: ElevatorParams = DEFAULT_ELEVATOR,
        exit_params: ExitParams = DEFAULT_EXIT,
        min_rr_ratio: float = 1.5,
    ):
        self.strategy_params = strategy_params
        self.exit_params = exit_params
        self.min_rr_ratio = min_rr_ratio

        # Sub-detectors
        self.price_level_detector = PriceLevelDetector(strategy_params)
        self.elevator_detector = ElevatorDownDetector(elevator_params)
        self.failed_breakdown = FailedBreakdown(strategy_params)
        self.level_reclaim = LevelReclaim(strategy_params)

        # State
        self.level_store = LevelStore()
        self.signals: list[Signal] = []
        self._last_elevator: Optional[ElevatorEvent] = None

    def reset(self) -> None:
        """Reset all state for a new session."""
        self.elevator_detector.reset()
        self.failed_breakdown.reset()
        self.level_reclaim.reset()
        self.level_store.clear()
        self.signals.clear()
        self._last_elevator = None

    def initialize_levels(
        self,
        df: pd.DataFrame,
        prior_day_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Pre-compute levels from historical data (called before bar-by-bar)."""
        if prior_day_df is not None:
            store = self.price_level_detector.detect_all(df, prior_day_df)
            self.level_store = store
        else:
            self.level_store = self.price_level_detector.detect_all(df)

    def update(
        self,
        bar_idx: int,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        velocity: float,
        df: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        """Process one bar through all detectors.

        Parameters
        ----------
        bar_idx : int
        timestamp : datetime
        open_, high, low, close, volume : float
        velocity : float
            Pre-computed velocity for this bar.
        df : pd.DataFrame, optional
            Full DataFrame (for incremental level detection).

        Returns
        -------
        Signal or None
        """
        # 1. Incremental level detection
        if df is not None:
            self.price_level_detector.detect_incremental(
                self.level_store, df, bar_idx
            )

        # 2. Elevator down detection
        elevator_event = self.elevator_detector.update(
            bar_idx=bar_idx,
            timestamp=timestamp,
            high=high,
            low=low,
            close=close,
            velocity=velocity,
            level_store=self.level_store,
        )
        if elevator_event is not None:
            # Gate: only accept elevators that broke enough support levels
            if elevator_event.levels_broken >= self.elevator_detector.params.min_levels_broken:
                self._last_elevator = elevator_event

        # 3. Failed Breakdown detection
        fb_signal = self.failed_breakdown.update(
            bar_idx=bar_idx,
            timestamp=timestamp,
            high=high,
            low=low,
            close=close,
            level_store=self.level_store,
            elevator_event=self._last_elevator,
        )
        if fb_signal is not None:
            signal = self._qualify_signal(fb_signal, SignalType.FAILED_BREAKDOWN)
            if signal is not None:
                self.signals.append(signal)
                return signal

        # 4. Level Reclaim detection
        lr_signal = self.level_reclaim.update(
            bar_idx=bar_idx,
            timestamp=timestamp,
            high=high,
            low=low,
            close=close,
            level_store=self.level_store,
        )
        if lr_signal is not None:
            signal = self._qualify_signal(lr_signal, SignalType.LEVEL_RECLAIM)
            if signal is not None:
                self.signals.append(signal)
                return signal

        return None

    def run_bars(
        self,
        df: pd.DataFrame,
        prior_day_df: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        """Run aggregation across all bars in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV bars.
        prior_day_df : pd.DataFrame, optional
            Previous session for prior-day levels.

        Returns
        -------
        list[Signal]
        """
        self.reset()
        self.initialize_levels(df, prior_day_df)

        velocity = compute_velocity(df, window=5)
        signals: list[Signal] = []

        for i in range(len(df)):
            vel = float(velocity.iat[i]) if not np.isnan(velocity.iat[i]) else 0.0
            signal = self.update(
                bar_idx=i,
                timestamp=df.index[i],
                open_=float(df["open"].iat[i]),
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                volume=float(df["volume"].iat[i]),
                velocity=vel,
                df=df,
            )
            if signal is not None:
                signals.append(signal)

        return signals

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _qualify_signal(
        self, pattern: PatternSignal, signal_type: SignalType
    ) -> Optional[Signal]:
        """Calculate targets, R:R, and filter."""
        entry = pattern.entry_price
        stop = pattern.stop_price
        risk = entry - stop

        if risk <= 0:
            return None

        # Find targets from level store
        above = self.level_store.resistances_above(entry, pattern.timestamp)
        if len(above) >= 2:
            t1 = above[0].price
            t2 = above[1].price
        elif len(above) == 1:
            t1 = above[0].price
            t2 = entry + risk * 3  # fallback: 3R target
        else:
            t1 = entry + risk * 2
            t2 = entry + risk * 3

        reward_t1 = t1 - entry
        reward_t2 = t2 - entry
        rr_t1 = reward_t1 / risk if risk > 0 else 0
        rr_t2 = reward_t2 / risk if risk > 0 else 0

        if rr_t1 < self.min_rr_ratio:
            return None

        return Signal(
            signal_type=signal_type,
            pattern=pattern,
            target_1=t1,
            target_2=t2,
            risk_pts=risk,
            reward_t1_pts=reward_t1,
            reward_t2_pts=reward_t2,
            rr_ratio_t1=rr_t1,
            rr_ratio_t2=rr_t2,
            bar_idx=pattern.bar_idx,
            timestamp=pattern.timestamp,
        )
