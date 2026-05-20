"""Signal aggregation from all pattern detectors + R:R calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.levels import LevelStore, LevelType, compute_confluence_score
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
from core.patterns_short_v2 import BreakdownShort, BacktestShort, VelocityBreakdownShort
from core.bar_aggregator import BarAggregator
from core.daily_structure import DailyStructureDetector
from core.intraday_context import IntradayContextTracker, IntradayState
from core.level_scoring import LevelQualityScorer
from core.price_levels import PriceLevelDetector


class SignalType(Enum):
    """Signal classification."""

    FAILED_BREAKDOWN = auto()
    LEVEL_RECLAIM = auto()
    FAILED_RALLY = auto()       # deprecated (mirrored FR)
    LEVEL_REJECTION = auto()    # deprecated (mirrored LJ)
    BREAKDOWN_SHORT = auto()    # Mancini: support breaks and holds broken
    BACKTEST_SHORT = auto()     # Mancini: first retest of broken support shelf from below
    VELOCITY_SHORT = auto()     # Single-bar velocity breakdown (news-driven)


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
    position_size_factor: float = 1.0  # Mancini sizing: 1.0=full, 0.5=half, 0.25=quarter
    bar_idx: int = 0
    timestamp: datetime = None
    confluence_score: int = 0  # level confluence score (0 = not computed)
    lqs: int = 0  # Level Quality Score (0-100) — logged for retrospective analysis

    @property
    def direction(self) -> str:
        """Infer trade direction from signal type."""
        _SHORT_TYPES = {SignalType.BREAKDOWN_SHORT, SignalType.BACKTEST_SHORT, SignalType.VELOCITY_SHORT}
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

        # Level Quality Scorer (LQS)
        self._level_scorer = LevelQualityScorer(strategy_params)
        self._market_data: Optional[dict] = None
        self._session_context: Optional[dict] = None

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
        self.velocity_breakdown = VelocityBreakdownShort(strategy_params)

        # 5-min bar aggregator for level detection
        self._bar_aggregator = BarAggregator(
            period_minutes=strategy_params.level_detection_timeframe_min
        )

        # State
        self.level_store = LevelStore()
        self.signals: list[Signal] = []
        # All signals evaluated on the current bar (for diagnostics)
        self._bar_signals: list[dict] = []
        # Shadow mode events: features log what they WOULD do without acting
        self.shadow_events: list[dict] = []
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
        # ATM level tracking: per-level profitability for "ATM machine level" detection
        # key: rounded level price, value: {"wins": int, "losses": int, "total_pnl": float, "last_session": str}
        self._level_performance: dict[float, dict] = {}
        # Data-driven gates (Mar 2026 trade_lessons.md)
        self._traded_levels: dict[float, int] = {}  # level_price -> trade count this session
        self._last_signal_level_bar: dict[float, int] = {}  # level_price -> bar of last signal
        self._session_high: float = float('-inf')
        self._session_low: float = float('inf')
        # Elevator→FB cycle: when a big selloff completes and price recovers,
        # shorts are fighting the squeeze.  Mancini: "the flush IS the setup."
        self._elevator_fb_active: bool = False
        self._elevator_recovery_low: float = 0.0  # low of the elevator that triggered
        self._elevator_recovery_drop: float = 0.0  # total drop in pts
        # Intraday price action context
        sp = strategy_params
        self._intraday_tracker = IntradayContextTracker(
            swing_order=sp.idc_swing_order,
            min_swing_pts=sp.idc_min_swing_pts,
            weak_bounce_pts=sp.idc_weak_bounce_pts,
            bounce_lookback=sp.idc_bounce_lookback,
            elevator_recency_bars=sp.idc_elevator_recency_bars,
            session_pos_bearish=sp.idc_session_pos_bearish,
            session_pos_bullish=sp.idc_session_pos_bullish,
            bearish_threshold=sp.idc_bearish_threshold,
            bullish_threshold=sp.idc_bullish_threshold,
        )
        self._intraday_state = IntradayState.NEUTRAL
        # Set externally by ManciniLongStrategy each bar when Mode 1 Green
        # is active and live (not shadow). When True, ``_qualify_signal`` uses
        # ``mode1_green_fb_min_rr`` as the R:R floor for FB longs.
        self.mode1_green_active: bool = False

        # Daily structure detector — macro bias from daily chart
        self._daily_structure = DailyStructureDetector(strategy_params)
        self._daily_bias: str = "NEUTRAL"

        # Mancini LLM-extracted plan (Phase 3) — populated externally by
        # IBRunner._initialize_session when use_mancini_llm_plan is True.
        # When set, _qualify_signal applies mode / danger_zones / no_trade_*
        # gates and a planned_setups LQS boost. None disables all gating.
        self._mancini_llm_plan = None

    @property
    def intraday_state(self) -> IntradayState:
        """Current intraday price action context state."""
        return self._intraday_state

    @property
    def daily_bias(self) -> str:
        """Current daily structure bias (DAILY_FB_BULL, DAILY_BD_BEAR, or NEUTRAL)."""
        return self._daily_bias

    def set_mancini_llm_plan(self, plan) -> None:
        """Inject a ManciniPlan loaded from mancini_plan_<date>.json.

        Pass ``None`` to clear. The plan is consulted only when
        ``strategy_params.use_mancini_llm_plan`` is True. Caller is
        responsible for loading the plan via ``live.mancini_llm_extract.load_plan``.
        """
        self._mancini_llm_plan = plan

    def _check_mancini_llm_gates(self, pattern, signal_type) -> Optional[str]:
        """Return a non-empty rejection reason if the LLM plan vetoes
        this signal; ``None`` to let it through.

        Three gates:
        1. Mode 1 Green — Mancini's explicit rule is "no FB longs on
           open-to-close trend-up days". Block FAILED_BREAKDOWN longs.
        2. Danger zones — reject longs whose entry falls inside any
           ``DangerZone`` price band Mancini flagged.
        3. ``no_trade_above`` / ``no_trade_below`` — single-sided guards
           Mancini sometimes states ("don't be long below 7100").
        """
        plan = self._mancini_llm_plan
        if plan is None:
            return None
        if not getattr(self.strategy_params, 'use_mancini_llm_plan', False):
            return None

        entry = pattern.entry_price
        direction = getattr(pattern, 'direction', 'long')

        if (plan.mode == "mode_1_green"
                and signal_type == SignalType.FAILED_BREAKDOWN
                and direction == 'long'):
            return f"mancini_mode={plan.mode} blocks FB long"

        if direction == 'long':
            for zone in plan.danger_zones:
                lo = zone.price_low
                hi = zone.price_high if zone.price_high is not None else lo
                if lo <= entry <= hi:
                    rule = (zone.rule or "")[:60]
                    return f"entry {entry} in danger_zone [{lo},{hi}]: {rule}"
            if plan.no_trade_above is not None and entry >= plan.no_trade_above:
                return f"entry {entry} >= no_trade_above {plan.no_trade_above}"
            if plan.no_trade_below is not None and entry <= plan.no_trade_below:
                return f"entry {entry} <= no_trade_below {plan.no_trade_below}"

        return None

    def _mancini_llm_setup_bonus(self, pattern, signal_type) -> int:
        """If the signal matches one of ``plan.planned_setups`` (level
        price within tolerance + matching direction), return the LQS
        bonus to apply. ``0`` otherwise.

        Setup type matching is intentionally relaxed — any matching
        level + direction qualifies, even if Mancini's named
        ``setup_type`` is ``level_reclaim`` and the engine fired
        ``failed_breakdown``. Both express the same conviction in the
        same level; a stricter match would suppress legitimate boosts.
        """
        plan = self._mancini_llm_plan
        if plan is None or not plan.planned_setups:
            return 0
        if not getattr(self.strategy_params, 'use_mancini_llm_plan', False):
            return 0
        if pattern.level is None:
            return 0

        level_price = pattern.level.price
        direction = getattr(pattern, 'direction', 'long')
        tolerance = getattr(
            self.strategy_params, 'mancini_llm_setup_match_tolerance_pts', 2.0,
        )
        bonus = getattr(self.strategy_params, 'mancini_llm_setup_lqs_bonus', 15)

        for setup in plan.planned_setups:
            if abs(setup.level_price - level_price) > tolerance:
                continue
            if setup.direction != direction:
                continue
            return bonus
        return 0

    def set_daily_structure(self, daily_bars: pd.DataFrame) -> str:
        """Update daily structure detector with daily OHLC bars.

        Parameters
        ----------
        daily_bars : pd.DataFrame
            Daily OHLC bars (at least ``daily_shelf_lookback_days`` rows).

        Returns
        -------
        str
            The detected daily bias.
        """
        self._daily_bias = self._daily_structure.update(daily_bars)
        return self._daily_bias

    def get_daily_structure_snapshot(self) -> dict:
        """Return daily structure state for logging/dashboard."""
        return self._daily_structure.get_snapshot()

    @property
    def bar_signals(self) -> list[dict]:
        """All signals evaluated on the most recent bar (for diagnostics)."""
        return self._bar_signals

    def get_swing_snapshot(self) -> dict:
        """Return current swing structure snapshot for trade logging."""
        return self._intraday_tracker.get_swing_snapshot()

    def get_pattern_state(self) -> dict:
        """Serialize pattern state for persistence across restarts."""
        from datetime import datetime as dt
        return {
            "failed_breakdown": self.failed_breakdown.get_state_snapshot(),
            "breakdown_short": self.breakdown_short.get_state_snapshot(),
            "timestamp": dt.now().isoformat(),
        }

    def restore_pattern_state(self, state: dict) -> None:
        """Restore pattern state from a saved snapshot."""
        if "failed_breakdown" in state:
            self.failed_breakdown.restore_state(state["failed_breakdown"])
        if "breakdown_short" in state:
            self.breakdown_short.restore_state(state["breakdown_short"])

    def reset(self) -> None:
        """Reset all state for a new session."""
        self.elevator_detector.reset()
        self.failed_breakdown.reset()
        self.level_reclaim.reset()
        self.elevator_up_detector.reset()
        self.failed_rally.reset()
        self.level_rejection.reset()
        # full_reset clears the per-shelf tracking dict; reset() alone
        # only clears the state flag and leaks shelves across sessions.
        self.breakdown_short.full_reset()
        self.backtest_short.full_reset()
        self.level_store.clear()
        self.signals.clear()
        self._last_elevator = None
        self._last_elevator_up = None
        self._volume_history.clear()
        self._last_signal_bar.clear()
        self._traded_levels.clear()
        self._last_signal_level_bar.clear()
        self._session_high = float('-inf')
        self._session_low = float('inf')
        self._elevator_fb_active = False
        self._elevator_recovery_low = 0.0
        self._elevator_recovery_drop = 0.0
        self._intraday_tracker.reset()
        self._intraday_state = IntradayState.NEUTRAL

    def initialize_levels(
        self,
        df: pd.DataFrame,
        prior_day_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize levels from prior-day data only (no current-day look-ahead).

        Current-day levels are discovered incrementally via detect_incremental
        during the bar-by-bar loop.

        When ``use_5min_levels`` is enabled, prior-day data is resampled to
        5-min bars so that swing detection runs on the same timeframe.
        """
        store = LevelStore()
        if prior_day_df is not None:
            self.price_level_detector._add_prior_day_levels(store, prior_day_df)

            # Run 5-min swing detection on prior-day data for initial levels
            if self.strategy_params.use_5min_levels and len(prior_day_df) >= self.strategy_params.level_detection_timeframe_min:
                df_5min_prior = self._bar_aggregator.resample(prior_day_df)
                if len(df_5min_prior) > self.strategy_params.swing_low_order_5min * 2:
                    order_5 = self.strategy_params.swing_low_order_5min
                    for idx in range(order_5 * 2, len(df_5min_prior)):
                        self.price_level_detector._detect_swing_lows_on_df(
                            store, df_5min_prior, idx,
                            order=order_5,
                        )
                    # Shelf detection on prior day
                    if self.strategy_params.detect_shelf_levels:
                        for idx in range(self.strategy_params.shelf_min_bars, len(df_5min_prior)):
                            self.price_level_detector._detect_shelf_levels(
                                store, df_5min_prior, idx,
                            )
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
        market_data: Optional[dict] = None,
        session_context: Optional[dict] = None,
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
        market_data : dict, optional
            Market snapshot (vix, vix_term_structure, etc.) for LQS regime scoring.
        session_context : dict, optional
            Session context (session_date, current_price, session_high, session_low,
            bar_count) for LQS recency scoring.

        Returns
        -------
        Signal or None
        """
        # Store market data and session context for LQS scoring in _qualify_signal
        if market_data is not None:
            self._market_data = market_data
        if session_context is not None:
            self._session_context = session_context
        # Clear per-bar signal diagnostics
        self._bar_signals = []

        # Track volume for confirmation checks
        self._volume_history.append(volume)

        # Track session high/low for range gate
        if high > self._session_high:
            self._session_high = high
        if low < self._session_low:
            self._session_low = low

        # 1. Incremental level detection
        if df is not None:
            df_5min = None
            bar_idx_5min = None
            if self.strategy_params.use_5min_levels:
                df_5min = self._bar_aggregator.update_incremental(df)
                if len(df_5min) > 0:
                    bar_idx_5min = len(df_5min) - 1
            self.price_level_detector.detect_incremental(
                self.level_store, df, bar_idx,
                df_5min=df_5min,
                bar_idx_5min=bar_idx_5min,
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

        # Elevator→FB cycle detection: when a big selloff completes and price
        # recovers 50%+ of the drop, we're in the squeeze phase.  Any short
        # fired now is fighting institutional accumulation.
        # Mancini: "Elevator down → Failed Breakdown → Short Squeeze.  The
        # cycle repeats over and over."
        if active_elevator is not None and active_elevator.is_complete:
            drop = active_elevator.total_drop_pts
            recovery = close - active_elevator.low_price
            # Activate when: drop was 40+ pts AND price has recovered 50%+
            if drop >= 40 and recovery >= drop * 0.5:
                if not self._elevator_fb_active:
                    logger.info(
                        f"Elevator→FB ACTIVE: {drop:.0f}pt drop, "
                        f"{recovery:.0f}pt recovery ({recovery/drop*100:.0f}%) "
                        f"— suppressing shorts during squeeze"
                    )
                self._elevator_fb_active = True
                self._elevator_recovery_low = active_elevator.low_price
                self._elevator_recovery_drop = drop
            else:
                self._elevator_fb_active = False
        else:
            # No completed elevator — if we had an active cycle from a now-expired
            # elevator, keep it active as long as price stays above the recovery low
            if self._elevator_fb_active and close < self._elevator_recovery_low:
                logger.info(
                    f"Elevator→FB DEACTIVATED: price {close:.2f} fell below "
                    f"elevator low {self._elevator_recovery_low:.2f}"
                )
                self._elevator_fb_active = False

        # 2b. Intraday context update (swing structure, bounce quality, session position)
        if self.strategy_params.use_intraday_context:
            # Elevator is "active" if currently tracking a selloff (not yet completed)
            elevator_is_active = self.elevator_detector.is_active()
            self._intraday_state = self._intraday_tracker.update(
                bar_idx=bar_idx,
                high=high,
                low=low,
                close=close,
                elevator_event=active_elevator,
                elevator_active=elevator_is_active,
                session_high=self._session_high,
                session_low=self._session_low,
            )

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
            is_dd = getattr(fb_signal, 'is_double_dip', False)
            cooldown_ok = (is_dd and getattr(self.strategy_params, 'dd_bypass_cooldown', True)) or self._check_cooldown(SignalType.FAILED_BREAKDOWN, bar_idx)
            gate_ok = (is_dd and getattr(self.strategy_params, 'dd_bypass_level_gate', True)) or self._check_level_gates(fb_signal, bar_idx)
            if not cooldown_ok:
                self._log_bar_signal(SignalType.FAILED_BREAKDOWN, fb_signal, "cooldown_blocked")
            elif not gate_ok:
                self._log_bar_signal(SignalType.FAILED_BREAKDOWN, fb_signal, "level_gate_blocked")
            else:
                signal = self._qualify_signal(fb_signal, SignalType.FAILED_BREAKDOWN)
                if signal is not None:
                    # Apply DD position size factor
                    if is_dd:
                        dd_size = getattr(self.strategy_params, 'dd_position_size_factor', 0.5)
                        signal.position_size_factor = min(signal.position_size_factor, dd_size)
                    self._log_bar_signal(SignalType.FAILED_BREAKDOWN, fb_signal, "taken", signal.rr_ratio_t1)
                    self._record_signal(SignalType.FAILED_BREAKDOWN, bar_idx,
                                        fb_signal.level.price)
                    self.signals.append(signal)
                    return signal
                else:
                    self._log_bar_signal(SignalType.FAILED_BREAKDOWN, fb_signal, "rr_rejected")

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
                cooldown_ok = self._check_cooldown(SignalType.LEVEL_RECLAIM, bar_idx)
                gate_ok = self._check_level_gates(lr_signal, bar_idx)
                if not cooldown_ok:
                    self._log_bar_signal(SignalType.LEVEL_RECLAIM, lr_signal, "cooldown_blocked")
                elif not gate_ok:
                    self._log_bar_signal(SignalType.LEVEL_RECLAIM, lr_signal, "level_gate_blocked")
                else:
                    signal = self._qualify_signal(lr_signal, SignalType.LEVEL_RECLAIM)
                    if signal is not None:
                        self._log_bar_signal(SignalType.LEVEL_RECLAIM, lr_signal, "taken", signal.rr_ratio_t1)
                        self._record_signal(SignalType.LEVEL_RECLAIM, bar_idx,
                                            lr_signal.level.price)
                        self.signals.append(signal)
                        return signal
                    else:
                        self._log_bar_signal(SignalType.LEVEL_RECLAIM, lr_signal, "rr_rejected")

        # --- Short-side pipeline: legacy mirrored patterns (gated by allow flags) ---
        if self.strategy_params.allow_short_fr or self.strategy_params.allow_short_lj:
            short_signal = self._run_short_pipeline(
                bar_idx, timestamp, high, low, close, velocity
            )
            if short_signal is not None:
                return short_signal

        # --- Short-side pipeline v2: Mancini-faithful patterns ---
        if (self.strategy_params.allow_breakdown_short
                or self.strategy_params.allow_backtest_short
                or self.strategy_params.allow_velocity_short):
            v2_signal = self._run_short_pipeline_v2(
                bar_idx, timestamp, open_, high, low, close, velocity, volume
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

    def _compute_sweep_depth_size_factor(self, pattern: PatternSignal) -> float:
        """Compute position size factor based on sweep depth.

        Mancini: "the bigger the sell, the bigger the squeeze."
        Deeper sweeps below the level indicate stronger institutional interest
        and produce higher win rates. Scale position size accordingly:
          - sweep < quarter_size_pts (2): 0.25
          - sweep 2-5 pts: linear 0.25 -> 0.50
          - sweep 5-8 pts: linear 0.50 -> 1.00
          - sweep >= full_size_pts (8): 1.0
          - sweep >= 20 pts (crash bottom): 1.0 (wider stop already handled elsewhere)
        """
        params = self.strategy_params

        # Compute sweep depth: use the stored value, or derive from level price - sweep_low
        sweep_depth = pattern.sweep_depth_pts
        if sweep_depth <= 0 and pattern.level is not None:
            if pattern.direction == "short":
                sweep_depth = pattern.sweep_high - pattern.level.price if pattern.sweep_high > 0 else 0.0
            else:
                sweep_depth = pattern.level.price - pattern.sweep_low if pattern.sweep_low > 0 else 0.0

        quarter = params.sweep_depth_quarter_size_pts  # 2.0
        full = params.sweep_depth_full_size_pts        # 8.0
        mid_pt = (quarter + full) / 2.0                # 5.0 — boundary between half and three-quarter

        if sweep_depth < quarter:
            return 0.25
        elif sweep_depth < mid_pt:
            # Linear interpolation: quarter -> mid_pt maps to 0.25 -> 0.50
            frac = (sweep_depth - quarter) / (mid_pt - quarter) if mid_pt > quarter else 0.0
            return 0.25 + frac * 0.25
        elif sweep_depth < full:
            # Linear interpolation: mid_pt -> full maps to 0.50 -> 1.00
            frac = (sweep_depth - mid_pt) / (full - mid_pt) if full > mid_pt else 0.0
            return 0.50 + frac * 0.50
        else:
            return 1.0

    # ------------------------------------------------------------------
    # ATM level tracking
    # ------------------------------------------------------------------

    def record_level_outcome(self, level_price: float, pnl: float, session_date: str) -> None:
        """Record a trade outcome at a level for ATM tracking.

        Parameters
        ----------
        level_price : float
            The level price (rounded to nearest 1.0).
        pnl : float
            Trade PnL in points (positive = win).
        session_date : str
            Session date string (e.g. "2026-04-03").
        """
        key = round(level_price)
        if key not in self._level_performance:
            self._level_performance[key] = {
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "last_session": session_date,
            }
        record = self._level_performance[key]
        if pnl > 0:
            record["wins"] += 1
        else:
            record["losses"] += 1
        record["total_pnl"] += pnl
        record["last_session"] = session_date
        logger.debug(
            f"ATM tracking: level {key} now {record['wins']}W/{record['losses']}L "
            f"(total PnL: {record['total_pnl']:.1f} pts)"
        )

    def is_atm_level(self, level_price: float) -> bool:
        """Check if a level qualifies as an ATM machine level.

        A level qualifies when it has enough winning trades and
        a win rate above the configured threshold.

        Parameters
        ----------
        level_price : float
            The level price (rounded to nearest 1.0).

        Returns
        -------
        bool
            True if the level meets ATM criteria.
        """
        key = round(level_price)
        record = self._level_performance.get(key)
        if record is None:
            return False
        total = record["wins"] + record["losses"]
        if record["wins"] < self.strategy_params.atm_min_winning_trades:
            return False
        if total == 0:
            return False
        win_rate = record["wins"] / total
        return win_rate >= self.strategy_params.atm_min_win_rate

    def expire_atm_levels(self, current_date: str, memory_days: int) -> None:
        """Remove ATM level records older than memory_days.

        Parameters
        ----------
        current_date : str
            Current session date string (e.g. "2026-04-03").
        memory_days : int
            Max age in trading days before a level record expires.
        """
        from datetime import datetime as dt
        try:
            today = dt.strptime(current_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return
        # Calendar day cutoff (rough: memory_days trading days ~ memory_days * 7/5 + 2 calendar days)
        cutoff_calendar = memory_days * 7 // 5 + 2
        expired_keys = []
        for key, record in self._level_performance.items():
            try:
                last = dt.strptime(record["last_session"], "%Y-%m-%d").date()
                if (today - last).days > cutoff_calendar:
                    expired_keys.append(key)
            except (ValueError, TypeError):
                continue
        for key in expired_keys:
            logger.debug(f"ATM tracking: expiring level {key} (last session: {self._level_performance[key]['last_session']})")
            del self._level_performance[key]

    def _compute_confluence(self, pattern: PatternSignal) -> int:
        """Compute confluence score for a pattern's level."""
        return compute_confluence_score(
            level=pattern.level,
            all_levels=self.level_store.levels,
            proximity=self.strategy_params.confluence_proximity_pts,
        )

    def _get_avg_volume_20(self) -> float:
        """Compute 20-bar average volume from volume history."""
        if len(self._volume_history) < self._volume_lookback:
            if len(self._volume_history) == 0:
                return 0.0
            return sum(self._volume_history) / len(self._volume_history)
        recent = self._volume_history[-self._volume_lookback:]
        return sum(recent) / len(recent)

    def _check_cooldown(self, signal_type: SignalType, bar_idx: int) -> bool:
        """Return True if this signal type is allowed (not in cooldown)."""
        cooldown = self.strategy_params.signal_cooldown_bars
        if cooldown <= 0:
            return True
        last_bar = self._last_signal_bar.get(signal_type)
        if last_bar is not None and bar_idx - last_bar < cooldown:
            return False
        return True

    def _record_signal(self, signal_type: SignalType, bar_idx: int,
                        level_price: float = 0.0) -> None:
        """Record that a signal was emitted for cooldown and level tracking."""
        self._last_signal_bar[signal_type] = bar_idx
        if level_price > 0:
            rounded = round(level_price / 0.25) * 0.25
            self._traded_levels[rounded] = self._traded_levels.get(rounded, 0) + 1
            self._last_signal_level_bar[rounded] = bar_idx

    def _check_level_gates(self, pattern: PatternSignal, bar_idx: int) -> bool:
        """Check data-driven gates: level reuse, session range, cross-type cooldown.

        Returns True if signal passes all gates, False if blocked.
        """
        sp = self.strategy_params
        rounded = round(pattern.level.price / 0.25) * 0.25

        # Gate 1: Level reuse — one trade per level per session
        if sp.max_trades_per_level > 0:
            if self._traded_levels.get(rounded, 0) >= sp.max_trades_per_level:
                return False

        # Gate 2: Cross-type cooldown — no opposing signal at same level within N bars
        if sp.cross_type_level_cooldown_bars > 0:
            last_bar = self._last_signal_level_bar.get(rounded)
            if last_bar is not None and bar_idx - last_bar < sp.cross_type_level_cooldown_bars:
                return False

        # Gate 3: Session range minimum — market must have established range
        if sp.min_session_range_pts > 0 and bar_idx > sp.min_session_range_grace_bars:
            session_range = self._session_high - self._session_low
            if session_range < sp.min_session_range_pts:
                return False

        return True

    def _log_bar_signal(
        self,
        signal_type: SignalType,
        pattern: PatternSignal,
        status: str,
        rr: Optional[float] = None,
    ) -> None:
        """Record a signal evaluation for the current bar's diagnostics."""
        direction = "short" if signal_type in {
            SignalType.BREAKDOWN_SHORT, SignalType.BACKTEST_SHORT,
            SignalType.VELOCITY_SHORT, SignalType.FAILED_RALLY,
            SignalType.LEVEL_REJECTION,
        } else "long"
        self._bar_signals.append({
            "signal_type": signal_type.name,
            "direction": direction,
            "entry_price": pattern.entry_price,
            "level_price": pattern.level.price if pattern.level else None,
            "level_type": pattern.level.level_type.name if pattern.level else None,
            "status": status,
            "rr_ratio": round(rr, 2) if rr is not None else None,
        })

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
                    self._log_bar_signal(SignalType.FAILED_RALLY, fr_signal, "taken", signal.rr_ratio_t1)
                    self.signals.append(signal)
                    return signal
                else:
                    self._log_bar_signal(SignalType.FAILED_RALLY, fr_signal, "rr_rejected")

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
                        self._log_bar_signal(SignalType.LEVEL_REJECTION, lj_signal, "taken", signal.rr_ratio_t1)
                        self.signals.append(signal)
                        return signal
                    else:
                        self._log_bar_signal(SignalType.LEVEL_REJECTION, lj_signal, "rr_rejected")

        return None

    def _run_short_pipeline_v2(
        self,
        bar_idx: int,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        velocity: float,
        volume: float = 0.0,
    ) -> Optional[Signal]:
        """Run Mancini-faithful short detectors (breakdown + velocity + backtest)."""
        # 1. Breakdown Short: support breaks and holds broken
        if self.strategy_params.allow_breakdown_short:
            bd_signal = self.breakdown_short.update(
                bar_idx=bar_idx,
                timestamp=timestamp,
                open_=open_,
                high=high,
                low=low,
                close=close,
                velocity=velocity,
                level_store=self.level_store,
            )
            if bd_signal is not None:
                cooldown_ok = self._check_cooldown(SignalType.BREAKDOWN_SHORT, bar_idx)
                gate_ok = self._check_level_gates(bd_signal, bar_idx)
                if not cooldown_ok:
                    self._log_bar_signal(SignalType.BREAKDOWN_SHORT, bd_signal, "cooldown_blocked")
                elif not gate_ok:
                    self._log_bar_signal(SignalType.BREAKDOWN_SHORT, bd_signal, "level_gate_blocked")
                else:
                    signal = self._qualify_short_signal(bd_signal, SignalType.BREAKDOWN_SHORT)
                    if signal is not None:
                        self._log_bar_signal(SignalType.BREAKDOWN_SHORT, bd_signal, "taken", signal.rr_ratio_t1)
                        self._record_signal(SignalType.BREAKDOWN_SHORT, bar_idx,
                                            bd_signal.level.price)
                        self.signals.append(signal)
                        return signal
                    else:
                        self._log_bar_signal(SignalType.BREAKDOWN_SHORT, bd_signal, "rr_rejected")

        # 2. Velocity Breakdown Short: single-bar news-driven breakdown
        if self.strategy_params.allow_velocity_short:
            avg_vol_20 = self._get_avg_volume_20()
            vbd_signal = self.velocity_breakdown.update(
                bar_idx=bar_idx,
                timestamp=timestamp,
                high=high,
                low=low,
                close=close,
                volume=volume,
                avg_volume_20=avg_vol_20,
                level_store=self.level_store,
            )
            if vbd_signal is not None:
                if not self._check_cooldown(SignalType.VELOCITY_SHORT, bar_idx):
                    self._log_bar_signal(SignalType.VELOCITY_SHORT, vbd_signal, "cooldown_blocked")
                else:
                    signal = self._qualify_short_signal(vbd_signal, SignalType.VELOCITY_SHORT)
                    if signal is not None:
                        # Override position size with the conservative VBD factor
                        signal = Signal(
                            signal_type=signal.signal_type,
                            pattern=signal.pattern,
                            target_1=signal.target_1,
                            target_2=signal.target_2,
                            risk_pts=signal.risk_pts,
                            reward_t1_pts=signal.reward_t1_pts,
                            reward_t2_pts=signal.reward_t2_pts,
                            rr_ratio_t1=signal.rr_ratio_t1,
                            rr_ratio_t2=signal.rr_ratio_t2,
                            position_size_factor=self.strategy_params.vbd_position_size_factor,
                            bar_idx=signal.bar_idx,
                            timestamp=signal.timestamp,
                        )
                        # Always log to shadow for tracking
                        self.shadow_events.append({
                            "feature": "velocity_short",
                            "bar_idx": bar_idx,
                            "timestamp": str(timestamp),
                            "entry_price": signal.pattern.entry_price,
                            "stop_price": signal.pattern.stop_price,
                            "target_1": signal.target_1,
                            "rr_ratio_t1": signal.rr_ratio_t1,
                            "position_size_factor": signal.position_size_factor,
                            "level_price": signal.pattern.level.price if signal.pattern.level else None,
                            "level_type": signal.pattern.level.level_type.name if signal.pattern.level else None,
                            "volume": volume,
                            "avg_volume_20": avg_vol_20,
                            "would_trade": True,
                        })
                        # Velocity short is LIVE — trade it
                        self._log_bar_signal(SignalType.VELOCITY_SHORT, vbd_signal, "taken", signal.rr_ratio_t1)
                        self._record_signal(SignalType.VELOCITY_SHORT, bar_idx)
                        self.signals.append(signal)
                        return signal
                    else:
                        self._log_bar_signal(SignalType.VELOCITY_SHORT, vbd_signal, "rr_rejected")

        # 3. Backtest Short: failed backtest of broken resistance
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
                cooldown_ok = self._check_cooldown(SignalType.BACKTEST_SHORT, bar_idx)
                gate_ok = self._check_level_gates(bt_signal, bar_idx)
                if not cooldown_ok:
                    self._log_bar_signal(SignalType.BACKTEST_SHORT, bt_signal, "cooldown_blocked")
                elif not gate_ok:
                    self._log_bar_signal(SignalType.BACKTEST_SHORT, bt_signal, "level_gate_blocked")
                else:
                    signal = self._qualify_short_signal(bt_signal, SignalType.BACKTEST_SHORT)
                    if signal is not None:
                        self._log_bar_signal(SignalType.BACKTEST_SHORT, bt_signal, "taken", signal.rr_ratio_t1)
                        self._record_signal(SignalType.BACKTEST_SHORT, bar_idx,
                                            bt_signal.level.price)
                        self.signals.append(signal)
                        return signal
                    else:
                        self._log_bar_signal(SignalType.BACKTEST_SHORT, bt_signal, "rr_rejected")

        return None

    def _qualify_short_signal(
        self, pattern: PatternSignal, signal_type: SignalType
    ) -> Optional[Signal]:
        """Calculate targets and R:R for short signals (supports below entry)."""
        if self.require_volume_confirmation and not self._has_volume_confirmation():
            return None

        # Mancini-aligned PDL short block. PRIOR_DAY_LOW is defined in
        # Mancini's methodology as a "significant low" — the level his core
        # Failed Breakdown long setup looks to buy on flush+recover. Shorting
        # PDL positions the bot against his highest-conviction long pattern.
        # Live data 2026-02-25 → 2026-05-12: 5/5 PDL shorts lost ($-813).
        if getattr(self.strategy_params, 'block_pdl_shorts', True):
            level = pattern.level
            if level is not None and getattr(level.level_type, 'name', '') == 'PRIOR_DAY_LOW':
                logger.info(
                    f"PDL short blocked: {signal_type.name} at {pattern.entry_price:.2f} "
                    f"on PRIOR_DAY_LOW {level.price:.2f} — Mancini PDL is a long-side "
                    f"FB level, not a short trigger"
                )
                self.shadow_events.append({
                    "feature": "block_pdl_shorts",
                    "bar_idx": pattern.bar_idx,
                    "timestamp": str(pattern.timestamp),
                    "signal_type": signal_type.name,
                    "entry_price": pattern.entry_price,
                    "level_price": level.price,
                })
                return None

        # Confluence scoring gate (opt-in)
        confluence_score = 0
        if self.strategy_params.use_confluence_scoring:
            confluence_score = self._compute_confluence(pattern)
            if confluence_score < self.strategy_params.confluence_min_score:
                return None

        entry = pattern.entry_price
        stop = pattern.stop_price
        risk = stop - entry  # short: stop is above entry

        if risk <= 0:
            return None

        # Find targets from level store (supports below entry), filtering out
        # too-close levels and clusters.
        # When mancini_t1_at_first_resistance is enabled, use the Mancini min
        # distance for the first support target (mirrored from long side).
        if self.strategy_params.mancini_t1_at_first_resistance:
            min_dist = self.strategy_params.mancini_t1_min_distance_pts
        else:
            min_dist = self.strategy_params.min_target_distance_pts
        below = [
            l for l in self.level_store.supports_below(entry, pattern.timestamp)
            if entry - l.price >= min_dist
            and l.level_type.name not in ('CLUSTER_LOW', 'CLUSTER_HIGH')
        ]
        below.reverse()  # nearest first
        # T2 must be a DISTINCT level below T1 (different level types can share
        # an identical price — LevelStore.add only merges within-type, so a
        # collision would otherwise yield T1 == T2). Require >= min_dist
        # separation so the staged exit bracket has actual room.
        if len(below) >= 1:
            t1 = below[0].price
            t2 = None
            for level in below[1:]:
                if t1 - level.price >= min_dist:
                    t2 = level.price
                    break
            if t2 is None:
                # No second distinct level: fall back to risk-multiple, but
                # ensure T2 is at least min_dist below T1.
                t2 = min(entry - risk * 3, t1 - min_dist)
        else:
            t1 = entry - risk * 2
            t2 = entry - risk * 3

        # Cap target distance (same as long side) to avoid unrealistic targets
        max_dist = self.strategy_params.max_target_distance_pts
        if entry - t1 > max_dist:
            t1 = entry - max_dist
        if entry - t2 > max_dist * 1.5:
            t2 = entry - max_dist * 1.5

        # Final guard: capping (or upstream collisions surviving the loop)
        # can still leave T2 >= T1. Force T2 strictly below T1 by min_dist.
        if t1 - t2 < min_dist:
            t2 = t1 - min_dist

        reward_t1 = entry - t1
        reward_t2 = entry - t2
        rr_t1 = reward_t1 / risk if risk > 0 else 0
        rr_t2 = reward_t2 / risk if risk > 0 else 0

        # Mancini-style position sizing based on stop distance
        max_full_stop = self.strategy_params.max_full_stop_pts
        if risk <= max_full_stop:
            size_factor = 1.0
        elif risk <= max_full_stop * 2:
            size_factor = 0.5
        elif risk <= max_full_stop * (10.0 / 3.0):
            size_factor = 0.25
        else:
            size_factor = 0.25  # minimum size for very deep sweeps

        # Sweep depth sizing override: "the bigger the sell, the bigger the squeeze"
        if self.strategy_params.use_sweep_depth_sizing:
            sweep_depth_factor = self._compute_sweep_depth_size_factor(pattern)
            if self.strategy_params.shadow_mode_features:
                self.shadow_events.append({
                    "feature": "sweep_depth",
                    "bar_idx": pattern.bar_idx,
                    "timestamp": str(pattern.timestamp),
                    "signal_type": signal_type.name,
                    "current_size_factor": size_factor,
                    "shadow_size_factor": sweep_depth_factor,
                    "sweep_depth_pts": pattern.sweep_depth_pts,
                    "level_price": pattern.level.price if pattern.level else None,
                })
            else:
                size_factor = sweep_depth_factor

        # Mancini-aligned short size de-rating (Phase 4 of short-engine rewrite).
        # Per Mancini 2026-02-04: "I still give short entries here for those who
        # like them and don't mind having a substantially lower win rate and
        # lower R/R - SPX is a structurally bullish asset class". This isn't a
        # detector rule he states — it's the risk-management response to his
        # own framing: lower-edge trades deserve smaller size. Default 0.5x.
        short_factor = getattr(self.strategy_params, 'short_size_factor', 0.5)
        if short_factor != 1.0:
            size_factor *= short_factor

        # Move exhaustion: if session low already hit the target, the move is
        # done — don't short into a bounce.  Mancini: the flush IS the setup
        # for the long, not a short continuation.
        if self._session_low <= t1:
            logger.info(
                f"Move exhaustion: session low {self._session_low:.2f} already "
                f"reached short target {t1:.2f} — skipping {signal_type.name}"
            )
            self.shadow_events.append({
                "feature": "move_exhaustion",
                "bar_idx": pattern.bar_idx,
                "timestamp": str(pattern.timestamp),
                "signal_type": signal_type.name,
                "session_low": self._session_low,
                "short_target": t1,
                "entry_price": entry,
            })
            return None

        # Capitulation-entry guard: reject shorts that fire near the session
        # low when most of the day's drop has already happened. The existing
        # move_exhaustion check only fires when session_low <= T1 (T1 is 20-30pts
        # BELOW entry), so it stays silent at the worst moment — when price has
        # crashed to the entry but not yet to the target. Per the 5/2026
        # short-side post-mortem, this caught 5 of 9 BD production losers and
        # 5 of 7 velocity losers.
        if getattr(self.strategy_params, 'block_capitulation_shorts', True):
            floor = self.strategy_params.short_capitulation_floor_pts
            off_high = self.strategy_params.short_capitulation_off_high_pts
            entry_above_low = entry - self._session_low
            high_above_entry = self._session_high - entry
            if (entry_above_low <= floor
                    and high_above_entry >= off_high
                    and self._session_high > 0  # guard pre-session no-data case
                    and self._session_low < float('inf')):
                logger.info(
                    f"Capitulation entry: {signal_type.name} short at {entry:.2f} "
                    f"sits {entry_above_low:.1f}pt above session_low "
                    f"{self._session_low:.2f}, {high_above_entry:.1f}pt below "
                    f"session_high {self._session_high:.2f} — skipping (fading the flush)"
                )
                self.shadow_events.append({
                    "feature": "capitulation_entry",
                    "bar_idx": pattern.bar_idx,
                    "timestamp": str(pattern.timestamp),
                    "signal_type": signal_type.name,
                    "entry_price": entry,
                    "session_low": self._session_low,
                    "session_high": self._session_high,
                    "entry_above_low_pts": round(entry_above_low, 2),
                    "high_above_entry_pts": round(high_above_entry, 2),
                })
                return None

        # Elevator→FB cycle: suppress shorts during the squeeze phase.
        # Mancini: "Elevator down, Failed Breakdown, Short Squeeze — the cycle
        # repeats.  If you try to short during the squeeze, you will lose."
        if self._elevator_fb_active:
            logger.info(
                f"Elevator→FB squeeze: {self._elevator_recovery_drop:.0f}pt "
                f"elevator completed, price recovering — suppressing "
                f"{signal_type.name} short at {entry:.2f}"
            )
            self.shadow_events.append({
                "feature": "elevator_fb_squeeze_suppression",
                "bar_idx": pattern.bar_idx,
                "timestamp": str(pattern.timestamp),
                "signal_type": signal_type.name,
                "entry_price": entry,
                "elevator_drop_pts": self._elevator_recovery_drop,
                "elevator_low": self._elevator_recovery_low,
                "current_recovery_pct": round(
                    (entry - self._elevator_recovery_low)
                    / self._elevator_recovery_drop * 100, 1
                ) if self._elevator_recovery_drop > 0 else 0,
            })
            return None

        # BD Short R:R floor — BD Shorts at R:R 1.0-1.5 had 14% WR
        if (signal_type == SignalType.BREAKDOWN_SHORT
                and rr_t1 < self.strategy_params.bd_short_min_rr):
            return None

        # Absolute R:R floor — reject truly garbage signals
        if rr_t1 < self.strategy_params.min_signal_rr:
            detector = self.failed_breakdown if signal_type == SignalType.FAILED_BREAKDOWN else None
            if detector and hasattr(detector, 'near_misses'):
                detector.near_misses.append({
                    "timestamp": str(pattern.timestamp),
                    "bar_idx": pattern.bar_idx,
                    "level_price": pattern.level.price,
                    "failure_reason": "rr_below_floor",
                    "achieved": {"rr_ratio": round(rr_t1, 2)},
                    "required": {"min_signal_rr": self.strategy_params.min_signal_rr},
                    "sweep_low": pattern.sweep_low,
                    "close_at_failure": entry,
                })
            return None

        # Absolute R:R floor — reject truly garbage signals
        if rr_t1 < self.strategy_params.min_signal_rr:
            return None

        # ATM level boost: increase size at levels with proven profitability
        if self.strategy_params.use_atm_level_boost:
            level_price = pattern.level.price
            if self.is_atm_level(level_price):
                record = self._level_performance[round(level_price)]
                size_factor *= self.strategy_params.atm_size_boost
                logger.info(
                    f"ATM LEVEL: {round(level_price)} has "
                    f"{record['wins']}W/{record['losses']}L — boosting size"
                )

        # Mancini overlay is LONG-SIDE ONLY — skip CUSTOM (Mancini) levels for shorts
        # Mancini is a long-side trader. His levels help longs (+42 pts) but create
        # bad short setups (-101 pts) per 5yr backtest.
        if (pattern.level and pattern.level.level_type == LevelType.CUSTOM
                and self.strategy_params.use_mancini_levels):
            self.shadow_events.append({
                "feature": "mancini_short_skip",
                "bar_idx": bar_idx,
                "timestamp": str(timestamp),
                "signal_type": signal_type.name,
                "level_price": pattern.level.price,
                "entry_price": entry,
            })
            return None

        # Level Quality Score (LQS): always compute for logging (short side)
        lqs = self._level_scorer.compute_lqs(
            pattern.level, self._market_data, self._session_context
        )

        # Daily structure gate: suppress low-LQS shorts during DAILY_FB_BULL.
        # Per the 5/2026 short-side post-mortem, every losing production BD
        # short and 6 of 7 velocity short losses fired with daily_bias =
        # DAILY_FB_BULL; the prior behavior here logged a shadow event but
        # fell through to return the signal ("Production mode would return
        # None here"). Now it actually blocks.
        if (self._daily_bias == "DAILY_FB_BULL"
                and self.strategy_params.use_daily_structure):
            min_lqs = self.strategy_params.daily_bd_short_min_lqs
            if lqs < min_lqs:
                self.shadow_events.append({
                    "feature": "daily_structure_short_suppression",
                    "bar_idx": pattern.bar_idx,
                    "timestamp": str(pattern.timestamp),
                    "signal_type": signal_type.name,
                    "lqs": lqs,
                    "required_lqs": min_lqs,
                    "daily_bias": self._daily_bias,
                    "level_price": pattern.level.price if pattern.level else None,
                    "level_type": pattern.level.level_type.name if pattern.level else None,
                    "entry_price": entry,
                })
                logger.info(
                    f"Daily FB BULL: contra-trend {signal_type.name} short "
                    f"(LQS {lqs} < {min_lqs}) at {entry:.2f} — BLOCKED"
                )
                return None

        # When LQS gating is enabled, apply trade params (short side)
        if self.strategy_params.use_level_quality_scoring:
            if lqs < self.strategy_params.lqs_shadow_threshold:
                return None  # skip entirely
            elif lqs < self.strategy_params.lqs_min_trade_threshold:
                # Shadow only — log phantom but don't trade
                self.shadow_events.append({
                    "feature": "lqs_shadow",
                    "bar_idx": pattern.bar_idx,
                    "timestamp": str(pattern.timestamp),
                    "signal_type": signal_type.name,
                    "direction": "short",
                    "lqs": lqs,
                    "level_price": pattern.level.price if pattern.level else None,
                    "level_type": pattern.level.level_type.name if pattern.level else None,
                    "entry_price": entry,
                    "rr_ratio_t1": round(rr_t1, 2),
                })
                return None
            else:
                trade_params = self._level_scorer.get_trade_params(lqs)
                size_factor = min(size_factor, trade_params["size_factor"])

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
            position_size_factor=size_factor,
            bar_idx=pattern.bar_idx,
            timestamp=pattern.timestamp,
            confluence_score=confluence_score,
            lqs=lqs,
        )

    def _qualify_signal(
        self, pattern: PatternSignal, signal_type: SignalType
    ) -> Optional[Signal]:
        """Calculate targets, R:R, and filter."""
        # Mancini LLM plan early gates (mode_1_green, danger zones, no-trade
        # zones). No-op when use_mancini_llm_plan is False or no plan loaded.
        llm_reject = self._check_mancini_llm_gates(pattern, signal_type)
        if llm_reject is not None:
            logger.debug(f"Mancini LLM plan rejected signal: {llm_reject}")
            return None

        # FB level freshness gate (Mancini's 24-36 hour rule). Older levels
        # are "macro FBs" — rare, only work in elevated volatility per his
        # own commentary. Skip the gate when VIX is elevated; enforce when
        # calm. Set fb_max_level_age_hours=0 to disable entirely.
        max_age_hours = getattr(self.strategy_params, "fb_max_level_age_hours", 0.0)
        if (max_age_hours > 0
                and signal_type == SignalType.FAILED_BREAKDOWN
                and pattern.level is not None
                and pattern.level.created_at is not None):
            # Macro-FB VIX override: in elevated volatility, allow older
            # levels (Mancini explicitly: "when volatility hits, I get
            # bigger Failed Breakdowns"). 0 disables the override.
            vix_threshold = getattr(
                self.strategy_params, "fb_macro_vix_threshold", 0.0,
            )
            current_vix = None
            if vix_threshold > 0 and self._market_data:
                try:
                    current_vix = float(self._market_data.get("vix") or 0.0)
                except (TypeError, ValueError):
                    current_vix = None

            if current_vix is not None and current_vix > vix_threshold:
                # Elevated-vol regime — macro FB territory, skip the gate.
                logger.debug(
                    f"FB freshness gate skipped: VIX {current_vix:.1f} > "
                    f"{vix_threshold:.1f} threshold (macro-FB regime)"
                )
            else:
                try:
                    age_seconds = (pattern.timestamp - pattern.level.created_at).total_seconds()
                    # Force to float — guards against mocked/non-datetime types
                    # silently returning unrelated objects through the arithmetic
                    age_hours = float(age_seconds) / 3600.0
                except (AttributeError, TypeError, ValueError):
                    age_hours = 0.0  # can't compute → don't gate
                if age_hours > max_age_hours:
                    logger.info(
                        f"FB rejected: level age {age_hours:.1f}h > "
                        f"{max_age_hours:.0f}h cap (level @ {pattern.level.price:.2f}, "
                        f"VIX={current_vix})"
                    )
                    self.shadow_events.append({
                        "feature": "fb_level_too_old",
                        "bar_idx": pattern.bar_idx,
                        "timestamp": str(pattern.timestamp),
                        "signal_type": signal_type.name,
                        "level_price": pattern.level.price,
                        "level_age_hours": round(age_hours, 1),
                        "max_age_hours": max_age_hours,
                        "vix": current_vix,
                    })
                    return None

        # Volume confirmation check (opt-in)
        if self.require_volume_confirmation and not self._has_volume_confirmation():
            return None

        # Confluence scoring gate (opt-in)
        confluence_score = 0
        if self.strategy_params.use_confluence_scoring:
            confluence_score = self._compute_confluence(pattern)
            if confluence_score < self.strategy_params.confluence_min_score:
                return None

        entry = pattern.entry_price
        stop = pattern.stop_price
        risk = entry - stop

        if risk <= 0:
            return None

        # Find targets from level store, filtering out too-close levels and clusters.
        # When mancini_t1_at_first_resistance is enabled, T1 is set at the first
        # resistance level >= mancini_t1_min_distance_pts above entry (level-based,
        # not fixed distance). Otherwise use the existing min_target_distance_pts.
        if self.strategy_params.mancini_t1_at_first_resistance:
            min_dist = self.strategy_params.mancini_t1_min_distance_pts
        else:
            min_dist = self.strategy_params.min_target_distance_pts
        above = [
            l for l in self.level_store.resistances_above(entry, pattern.timestamp)
            if l.price - entry >= min_dist
            and l.level_type.name not in ('CLUSTER_LOW', 'CLUSTER_HIGH')
        ]
        # T2 must be a DISTINCT level above T1 (different level types can sit at
        # identical prices since LevelStore.add only merges within-type — that
        # collision would otherwise yield T1 == T2 and break the staged exit).
        # Require at least `min_dist` separation between T1 and T2 so the
        # bracket actually has room for the runner.
        if len(above) >= 1:
            t1 = above[0].price
            t2 = None
            for level in above[1:]:
                if level.price - t1 >= min_dist:
                    t2 = level.price
                    break
            if t2 is None:
                # No second distinct level: fall back to risk-multiple, but
                # ensure T2 is at least min_dist above T1.
                t2 = max(entry + risk * 3, t1 + min_dist)
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

        # Final guard: capping (or upstream collisions surviving the loop)
        # can still leave T2 <= T1. Force T2 strictly above T1 by min_dist.
        if t2 - t1 < min_dist:
            t2 = t1 + min_dist

        reward_t1 = t1 - entry
        reward_t2 = t2 - entry
        rr_t1 = reward_t1 / risk if risk > 0 else 0
        rr_t2 = reward_t2 / risk if risk > 0 else 0

        # Mode 1 Green relaxed R:R floor for FB longs on confirmed trend-up days.
        # The trend itself is a large part of the edge; Mancini accepts tighter
        # R:R on trend days (Apr 15 2026 post). Only applies when live (not shadow).
        effective_min_rr = self.min_rr_ratio
        effective_min_signal_rr = self.strategy_params.min_signal_rr
        if (signal_type == SignalType.FAILED_BREAKDOWN
                and self.mode1_green_active
                and not self.strategy_params.shadow_mode_features):
            green_min = self.strategy_params.mode1_green_fb_min_rr
            effective_min_rr = min(effective_min_rr, green_min)
            effective_min_signal_rr = min(effective_min_signal_rr, green_min)

        # Absolute R:R floor — reject truly garbage signals (same as short side)
        if rr_t1 < effective_min_signal_rr:
            return None

        # Track near-miss diagnostics when R:R is low (but don't reject)
        if rr_t1 < effective_min_rr:
            detector = self.failed_breakdown if signal_type == SignalType.FAILED_BREAKDOWN else None
            if detector and hasattr(detector, 'near_misses'):
                detector.near_misses.append({
                    "timestamp": str(pattern.timestamp),
                    "bar_idx": pattern.bar_idx,
                    "level_price": pattern.level.price,
                    "failure_reason": "rr_low_sized_down",
                    "achieved": {"rr_ratio": round(rr_t1, 2)},
                    "required": {"min_rr_ratio": self.min_rr_ratio},
                    "sweep_low": pattern.sweep_low,
                    "close_at_failure": entry,
                })

        # Mancini-style position sizing based on stop distance
        # Max 15 pts full size, size down proportionally beyond that
        max_full_stop = self.strategy_params.max_full_stop_pts
        if risk <= max_full_stop:
            size_factor = 1.0
        elif risk <= max_full_stop * 2:
            size_factor = 0.5
        elif risk <= max_full_stop * (10.0 / 3.0):
            size_factor = 0.25
        else:
            size_factor = 0.25  # minimum size for very deep sweeps

        # Sweep depth sizing override: "the bigger the sell, the bigger the squeeze"
        if self.strategy_params.use_sweep_depth_sizing:
            sweep_depth_factor = self._compute_sweep_depth_size_factor(pattern)
            if self.strategy_params.shadow_mode_features:
                self.shadow_events.append({
                    "feature": "sweep_depth",
                    "bar_idx": pattern.bar_idx,
                    "timestamp": str(pattern.timestamp),
                    "signal_type": "LONG_QUALIFY",
                    "current_size_factor": size_factor,
                    "shadow_size_factor": sweep_depth_factor,
                    "sweep_depth_pts": pattern.sweep_depth_pts,
                    "level_price": pattern.level.price if pattern.level else None,
                })
            else:
                size_factor = sweep_depth_factor

        # ATM level boost: increase size at levels with proven profitability
        if self.strategy_params.use_atm_level_boost:
            level_price = pattern.level.price
            if self.is_atm_level(level_price):
                record = self._level_performance[round(level_price)]
                size_factor *= self.strategy_params.atm_size_boost
                logger.info(
                    f"ATM LEVEL: {round(level_price)} has "
                    f"{record['wins']}W/{record['losses']}L — boosting size"
                )

        # Level Quality Score (LQS): always compute for logging
        lqs = self._level_scorer.compute_lqs(
            pattern.level, self._market_data, self._session_context
        )

        # Daily structure bonus: boost FB Long LQS during DAILY_FB_BULL
        if (self._daily_bias == "DAILY_FB_BULL"
                and signal_type == SignalType.FAILED_BREAKDOWN
                and self.strategy_params.use_daily_structure):
            bonus = self.strategy_params.daily_fb_lqs_bonus
            lqs = min(100, lqs + bonus)
            logger.debug(f"Daily FB BULL: +{bonus} LQS bonus for FB Long → LQS={lqs}")

        # Mancini LLM plan setup-match bonus: when the signal level is one
        # Mancini explicitly flagged as a planned setup, boost LQS so the
        # high-conviction setup clears the trade threshold even when other
        # confluence is lukewarm.
        llm_setup_bonus = self._mancini_llm_setup_bonus(pattern, signal_type)
        if llm_setup_bonus > 0:
            lqs = min(100, lqs + llm_setup_bonus)
            logger.debug(
                f"Mancini LLM planned setup match: +{llm_setup_bonus} LQS → {lqs}"
            )

        # When LQS gating is enabled, apply trade params
        if self.strategy_params.use_level_quality_scoring:
            if lqs < self.strategy_params.lqs_shadow_threshold:
                return None  # skip entirely
            elif lqs < self.strategy_params.lqs_min_trade_threshold:
                # Shadow only — log phantom but don't trade
                self.shadow_events.append({
                    "feature": "lqs_shadow",
                    "bar_idx": pattern.bar_idx,
                    "timestamp": str(pattern.timestamp),
                    "signal_type": signal_type.name,
                    "direction": "long",
                    "lqs": lqs,
                    "level_price": pattern.level.price if pattern.level else None,
                    "level_type": pattern.level.level_type.name if pattern.level else None,
                    "entry_price": entry,
                    "rr_ratio_t1": round(rr_t1, 2),
                })
                return None
            else:
                trade_params = self._level_scorer.get_trade_params(lqs)
                size_factor = min(size_factor, trade_params["size_factor"])

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
            position_size_factor=size_factor,
            bar_idx=pattern.bar_idx,
            timestamp=pattern.timestamp,
            confluence_score=confluence_score,
            lqs=lqs,
        )
