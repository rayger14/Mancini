"""Signal aggregation from all pattern detectors + R:R calculation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time
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
from core.elevator_up import ElevatorUpDetector, ElevatorUpEvent
from core.indicators import compute_velocity
from core.patterns import (
    FailedBreakdown,
    LevelReclaim,
    PatternSignal,
)
from core.patterns_short import FailedRally, LevelRejection
from core.patterns_short_v2 import BreakdownShort, BacktestShort
from core.price_levels import PriceLevelDetector


class SignalType(Enum):
    """Signal classification."""

    FAILED_BREAKDOWN = auto()
    LEVEL_RECLAIM = auto()
    FAILED_RALLY = auto()       # deprecated (mirrored FR)
    LEVEL_REJECTION = auto()    # deprecated (mirrored LJ)
    BREAKDOWN_SHORT = auto()    # Mancini: support breaks and holds broken
    BACKTEST_SHORT = auto()     # Mancini: failed retest of broken resistance


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
    def direction(self) -> str:
        """Infer trade direction from signal type."""
        _SHORT_TYPES = {SignalType.BREAKDOWN_SHORT, SignalType.BACKTEST_SHORT}
        return "short" if self.signal_type in _SHORT_TYPES else "long"

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
        rth_filter: Optional[tuple[dt_time, dt_time]] = None,
    ):
        self.strategy_params = strategy_params
        self.exit_params = exit_params
        self.min_rr_ratio = min_rr_ratio

        # Sub-detectors (long side)
        self.price_level_detector = PriceLevelDetector(
            strategy_params, rth_filter=rth_filter
        )
        self.elevator_detector = ElevatorDownDetector(elevator_params)
        self.failed_breakdown = FailedBreakdown(strategy_params)
        self.level_reclaim = LevelReclaim(strategy_params)

        # Sub-detectors (short side — legacy mirrored patterns)
        self.elevator_up_detector = ElevatorUpDetector(elevator_params)
        self.failed_rally = FailedRally(strategy_params)
        self.level_rejection = LevelRejection(strategy_params)

        # Sub-detectors (short side — Mancini-faithful v2)
        self.breakdown_short = BreakdownShort(strategy_params)
        self.backtest_short = BacktestShort(strategy_params)

        # State
        self.level_store = LevelStore()
        self.signals: list[Signal] = []
        self._last_elevator: Optional[ElevatorEvent] = None
        self._last_elevator_up: Optional[ElevatorUpEvent] = None
        # Elevator events expire after this many bars.
        # For full-session (with rth_filter), expire after 60 bars to prevent
        # stale overnight elevators. For RTH-only, use 999 (effectively no expiry).
        self._elevator_max_age_bars: int = 60 if rth_filter is not None else 999
        # Volume tracking for confirmation
        self._volume_history: list[float] = []
        self._volume_lookback: int = 20
        self._volume_spike_threshold: float = 1.5  # 1.5x avg = spike
        self.require_volume_confirmation: bool = False  # opt-in
        # Signal cooldown: track last signal bar per type to suppress rapid-fire signals
        self._last_signal_bar: dict[SignalType, int] = {}

    def reset(self) -> None:
        """Reset all state for a new session."""
        self.elevator_detector.reset()
        self.failed_breakdown.reset()
        self.level_reclaim.reset()
        self.elevator_up_detector.reset()
        self.failed_rally.reset()
        self.level_rejection.reset()
        self.breakdown_short.reset()
        self.backtest_short.full_reset()
        self.level_store.clear()
        self.signals.clear()
        self._last_elevator = None
        self._last_elevator_up = None
        self._volume_history.clear()
        self._last_signal_bar.clear()

    def initialize_levels(
        self,
        df: pd.DataFrame,
        prior_day_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize levels from prior-day data only (no current-day look-ahead).

        Current-day levels are discovered incrementally via detect_incremental
        during the bar-by-bar loop.
        """
        store = LevelStore()
        if prior_day_df is not None:
            self.price_level_detector._add_prior_day_levels(store, prior_day_df)
        self.level_store = store

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
        # Track volume for confirmation checks
        self._volume_history.append(volume)

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

        # Expire stale elevator events (Mancini uses them within ~1 hour)
        active_elevator = self._last_elevator
        if active_elevator is not None and active_elevator.end_idx is not None:
            age_bars = bar_idx - active_elevator.end_idx
            if age_bars > self._elevator_max_age_bars:
                active_elevator = None

        # 3. Failed Breakdown detection
        fb_signal = self.failed_breakdown.update(
            bar_idx=bar_idx,
            timestamp=timestamp,
            high=high,
            low=low,
            close=close,
            level_store=self.level_store,
            elevator_event=active_elevator,
        )
        if fb_signal is not None:
            if self._check_cooldown(SignalType.FAILED_BREAKDOWN, bar_idx):
                signal = self._qualify_signal(fb_signal, SignalType.FAILED_BREAKDOWN)
                if signal is not None:
                    self._record_signal(SignalType.FAILED_BREAKDOWN, bar_idx)
                    self.signals.append(signal)
                    return signal

        # 4. Level Reclaim detection (deferred if FB is actively tracking)
        # FB is the primary Mancini setup; LR should not steal position slots
        from core.patterns import PatternState
        fb_active = self.failed_breakdown.state != PatternState.IDLE

        if not fb_active:
            lr_signal = self.level_reclaim.update(
                bar_idx=bar_idx,
                timestamp=timestamp,
                high=high,
                low=low,
                close=close,
                level_store=self.level_store,
            )
            if lr_signal is not None:
                if self._check_cooldown(SignalType.LEVEL_RECLAIM, bar_idx):
                    signal = self._qualify_signal(lr_signal, SignalType.LEVEL_RECLAIM)
                    if signal is not None:
                        self._record_signal(SignalType.LEVEL_RECLAIM, bar_idx)
                        self.signals.append(signal)
                        return signal

        # --- Short-side pipeline: legacy mirrored patterns (gated by allow flags) ---
        if self.strategy_params.allow_short_fr or self.strategy_params.allow_short_lj:
            short_signal = self._run_short_pipeline(
                bar_idx, timestamp, high, low, close, velocity
            )
            if short_signal is not None:
                return short_signal

        # --- Short-side pipeline v2: Mancini-faithful patterns ---
        if self.strategy_params.allow_breakdown_short or self.strategy_params.allow_backtest_short:
            v2_signal = self._run_short_pipeline_v2(
                bar_idx, timestamp, high, low, close
            )
            if v2_signal is not None:
                return v2_signal

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

    def _check_cooldown(self, signal_type: SignalType, bar_idx: int) -> bool:
        """Return True if this signal type is allowed (not in cooldown)."""
        cooldown = self.strategy_params.signal_cooldown_bars
        if cooldown <= 0:
            return True
        last_bar = self._last_signal_bar.get(signal_type)
        if last_bar is not None and bar_idx - last_bar < cooldown:
            return False
        return True

    def _record_signal(self, signal_type: SignalType, bar_idx: int) -> None:
        """Record that a signal was emitted for cooldown tracking."""
        self._last_signal_bar[signal_type] = bar_idx

    def _has_volume_confirmation(self) -> bool:
        """Check if recent bars had above-average volume (Mancini volume spike)."""
        if len(self._volume_history) < self._volume_lookback:
            return True  # not enough history, allow signal
        recent = self._volume_history[-self._volume_lookback:]
        avg_vol = sum(recent) / len(recent)
        if avg_vol <= 0:
            return True
        # Check if any of the last 5 bars had a volume spike
        last_5 = self._volume_history[-5:]
        return any(v >= avg_vol * self._volume_spike_threshold for v in last_5)

    def _run_short_pipeline(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        velocity: float,
    ) -> Optional[Signal]:
        """Run short-side detectors (elevator up, failed rally, level rejection)."""
        # 5. Elevator up detection
        elevator_up_event = self.elevator_up_detector.update(
            bar_idx=bar_idx,
            timestamp=timestamp,
            high=high,
            low=low,
            close=close,
            velocity=velocity,
            level_store=self.level_store,
        )
        if elevator_up_event is not None:
            if elevator_up_event.levels_broken >= self.elevator_up_detector.params.min_levels_broken:
                self._last_elevator_up = elevator_up_event

        # Expire stale elevator up events
        active_elevator_up = self._last_elevator_up
        if active_elevator_up is not None and active_elevator_up.end_idx is not None:
            age_bars = bar_idx - active_elevator_up.end_idx
            if age_bars > self._elevator_max_age_bars:
                active_elevator_up = None

        # 6. Failed Rally detection
        if self.strategy_params.allow_short_fr:
            fr_signal = self.failed_rally.update(
                bar_idx=bar_idx,
                timestamp=timestamp,
                high=high,
                low=low,
                close=close,
                level_store=self.level_store,
                elevator_event=active_elevator_up,
            )
            if fr_signal is not None:
                signal = self._qualify_short_signal(fr_signal, SignalType.FAILED_RALLY)
                if signal is not None:
                    self.signals.append(signal)
                    return signal

        # 7. Level Rejection detection (deferred if FR is actively tracking)
        if self.strategy_params.allow_short_lj:
            from core.patterns import PatternState
            fr_active = self.failed_rally.state != PatternState.IDLE

            if not fr_active:
                lj_signal = self.level_rejection.update(
                    bar_idx=bar_idx,
                    timestamp=timestamp,
                    high=high,
                    low=low,
                    close=close,
                    level_store=self.level_store,
                )
                if lj_signal is not None:
                    signal = self._qualify_short_signal(lj_signal, SignalType.LEVEL_REJECTION)
                    if signal is not None:
                        self.signals.append(signal)
                        return signal

        return None

    def _run_short_pipeline_v2(
        self,
        bar_idx: int,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ) -> Optional[Signal]:
        """Run Mancini-faithful short detectors (breakdown + backtest)."""
        # 1. Breakdown Short: support breaks and holds broken
        if self.strategy_params.allow_breakdown_short:
            bd_signal = self.breakdown_short.update(
                bar_idx=bar_idx,
                timestamp=timestamp,
                high=high,
                low=low,
                close=close,
                level_store=self.level_store,
            )
            if bd_signal is not None:
                if self._check_cooldown(SignalType.BREAKDOWN_SHORT, bar_idx):
                    signal = self._qualify_short_signal(bd_signal, SignalType.BREAKDOWN_SHORT)
                    if signal is not None:
                        self._record_signal(SignalType.BREAKDOWN_SHORT, bar_idx)
                        self.signals.append(signal)
                        return signal

        # 2. Backtest Short: failed backtest of broken resistance
        if self.strategy_params.allow_backtest_short:
            bt_signal = self.backtest_short.update(
                bar_idx=bar_idx,
                timestamp=timestamp,
                high=high,
                low=low,
                close=close,
                level_store=self.level_store,
            )
            if bt_signal is not None:
                if self._check_cooldown(SignalType.BACKTEST_SHORT, bar_idx):
                    signal = self._qualify_short_signal(bt_signal, SignalType.BACKTEST_SHORT)
                    if signal is not None:
                        self._record_signal(SignalType.BACKTEST_SHORT, bar_idx)
                        self.signals.append(signal)
                        return signal

        return None

    def _qualify_short_signal(
        self, pattern: PatternSignal, signal_type: SignalType
    ) -> Optional[Signal]:
        """Calculate targets and R:R for short signals (supports below entry)."""
        if self.require_volume_confirmation and not self._has_volume_confirmation():
            return None

        entry = pattern.entry_price
        stop = pattern.stop_price
        risk = stop - entry  # short: stop is above entry

        if risk <= 0:
            return None

        # Find targets from level store (supports below entry)
        below = self.level_store.supports_below(entry, pattern.timestamp)
        below.reverse()  # nearest first
        if len(below) >= 2:
            t1 = below[0].price
            t2 = below[1].price
        elif len(below) == 1:
            t1 = below[0].price
            t2 = entry - risk * 3
        else:
            t1 = entry - risk * 2
            t2 = entry - risk * 3

        # Cap target distance (same as long side) to avoid unrealistic targets
        max_dist = self.strategy_params.max_target_distance_pts
        if entry - t1 > max_dist:
            t1 = entry - max_dist
        if entry - t2 > max_dist * 1.5:
            t2 = entry - max_dist * 1.5

        reward_t1 = entry - t1
        reward_t2 = entry - t2
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

    def _qualify_signal(
        self, pattern: PatternSignal, signal_type: SignalType
    ) -> Optional[Signal]:
        """Calculate targets, R:R, and filter."""
        # Volume confirmation check (opt-in)
        if self.require_volume_confirmation and not self._has_volume_confirmation():
            return None
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

        # Cap target distance: Mancini's avg FB return is 30-50 pts.
        # When first resistance is very far, cap T1 to avoid unrealistic R:R.
        # Diagnostic: R:R > 5 trades win only 8% of the time.
        max_dist = self.strategy_params.max_target_distance_pts
        if t1 - entry > max_dist:
            t1 = entry + max_dist
        if t2 - entry > max_dist * 1.5:
            t2 = entry + max_dist * 1.5

        reward_t1 = t1 - entry
        reward_t2 = t2 - entry
        rr_t1 = reward_t1 / risk if risk > 0 else 0
        rr_t2 = reward_t2 / risk if risk > 0 else 0

        if rr_t1 < self.min_rr_ratio:
            # Track as near-miss for self-improvement data
            detector = self.failed_breakdown if signal_type == SignalType.FAILED_BREAKDOWN else None
            if detector and hasattr(detector, 'near_misses'):
                detector.near_misses.append({
                    "timestamp": str(pattern.timestamp),
                    "bar_idx": pattern.bar_idx,
                    "level_price": pattern.level.price,
                    "failure_reason": "rr_too_low",
                    "achieved": {"rr_ratio": round(rr_t1, 2)},
                    "required": {"min_rr_ratio": self.min_rr_ratio},
                    "sweep_low": pattern.sweep_low,
                    "close_at_failure": entry,
                })
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
