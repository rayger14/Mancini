"""Main orchestrator: bar-by-bar loop + VectorBT backtest array preparation.

Dual-mode execution:
1. Python objects for live trading (readable, debuggable)
2. Flat arrays for VectorBT Numba callbacks (fast backtesting)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    StrategyParams,
    ElevatorParams,
    ExitParams,
    RiskParams,
    SessionTimes,
    ESContractSpec,
    DEFAULT_STRATEGY,
    DEFAULT_ELEVATOR,
    DEFAULT_EXIT,
    DEFAULT_RISK,
    DEFAULT_SESSION,
    DEFAULT_CONTRACT,
)
from config.levels import Level, LevelType
from core.indicators import enrich_dataframe
from core.mode1_detector import Mode1Detector
from core.mode1_green_detector import Mode1GreenDetector
from core.regime_filter import compute_regime, RegimeState, RegimeParams, Direction, VolRegime
from core.signals import Signal, SignalAggregator, SignalType
from strategy.entry_manager import EntryManager, EntryDecision
from strategy.exit_manager import ExitManager, ExitAction, ExitPhase, TradePosition
from strategy.position_manager import PositionManager, TradeRecord
from strategy.risk_manager import RiskManager


@dataclass
class BarResult:
    """Result of processing a single bar."""

    bar_idx: int
    timestamp: datetime
    signal: Optional[Signal] = None
    entry_decision: Optional[EntryDecision] = None
    exit_action: Optional[ExitAction] = None
    trade_record: Optional[TradeRecord] = None


class ManciniLongStrategy:
    """Mancini Method long-only strategy orchestrator.

    Wires together signal detection, entry decisions, exit management,
    and risk controls.
    """

    def __init__(
        self,
        strategy_params: StrategyParams = DEFAULT_STRATEGY,
        elevator_params: ElevatorParams = DEFAULT_ELEVATOR,
        exit_params: ExitParams = DEFAULT_EXIT,
        risk_params: RiskParams = DEFAULT_RISK,
        session_times: SessionTimes = DEFAULT_SESSION,
        contract: ESContractSpec = DEFAULT_CONTRACT,
        min_rr_ratio: float = 1.5,
        rth_filter: Optional[tuple[dt_time, dt_time]] = None,
        regime_params: Optional[RegimeParams] = None,
        daily_history: Optional[pd.DataFrame] = None,
    ):
        self.strategy_params = strategy_params
        self.exit_params = exit_params
        self.risk_params = risk_params
        self.session_times = session_times
        self.contract = contract

        # Sub-components
        self.signal_aggregator = SignalAggregator(
            strategy_params=strategy_params,
            elevator_params=elevator_params,
            exit_params=exit_params,
            min_rr_ratio=min_rr_ratio,
            rth_filter=rth_filter,
        )
        self.entry_manager = EntryManager(
            session=session_times,
            exit_params=exit_params,
            risk_params=risk_params,
        )
        self.exit_manager = ExitManager(
            params=exit_params,
            contract=contract,
        )
        self.position_manager = PositionManager(risk_params=risk_params, point_value=contract.point_value)
        self.risk_manager = RiskManager(
            risk_params=risk_params,
            session=session_times,
            contract=contract,
        )

        # Regime filter
        self.regime_params = regime_params or RegimeParams()
        self._regime_state: Optional[RegimeState] = None
        self._daily_history = daily_history

        # State — support concurrent long + short positions
        self._long_position: Optional[TradePosition] = None
        self._long_pattern_type: str = ""
        self._long_signal: Optional[Signal] = None
        self._long_entry_bar_idx: int = 0
        self._short_position: Optional[TradePosition] = None
        self._short_pattern_type: str = ""
        self._short_signal: Optional[Signal] = None
        self._short_entry_bar_idx: int = 0
        # Legacy alias for backward compat (read-only callers)
        self._current_position: Optional[TradePosition] = None
        self._results: list[BarResult] = []
        # Recent bars for candle bias filter (rolling 5-bar window)
        self._recent_bars: list[tuple[float, float, float, float]] = []  # [(o, h, l, c), ...]

        # Mode 1 (trend day) detector
        self.mode1_detector = Mode1Detector(strategy_params)
        # Mode 1 Green (trend up day) detector — mirror
        self.mode1_green_detector = Mode1GreenDetector(strategy_params)

        # Multi-day level persistence — survives reset() across days
        self._persistent_levels: list[Level] = []

    def reset(self) -> None:
        """Reset all state for a new session.

        NOTE: _persistent_levels is intentionally NOT cleared here.
        It carries significant levels across days for multi-day memory.
        """
        self.signal_aggregator.reset()
        self.mode1_detector.reset()
        self.mode1_green_detector.reset()
        self._long_position = None
        self._long_pattern_type = ""
        self._long_signal = None
        self._long_entry_bar_idx = 0
        self._short_position = None
        self._short_pattern_type = ""
        self._short_signal = None
        self._short_entry_bar_idx = 0
        self._current_position = None
        self._results.clear()
        self._recent_bars.clear()

    # ------------------------------------------------------------------
    # Bar-by-bar execution (live mode)
    # ------------------------------------------------------------------

    def run_day(
        self,
        df: pd.DataFrame,
        prior_day_df: Optional[pd.DataFrame] = None,
        session_date: Optional[datetime] = None,
        runner_state: Optional[object] = None,
    ) -> list[BarResult]:
        """Run strategy bar-by-bar over one day of data.

        Parameters
        ----------
        df : pd.DataFrame
            Current day OHLCV bars (1-min).
        prior_day_df : pd.DataFrame, optional
            Previous day for level initialization.
        session_date : datetime, optional
            Session date (defaults to first bar date).
        runner_state : RunnerCarryState, optional
            Runner position carried from prior day (for multi-day backtests).

        Returns
        -------
        list[BarResult]
            Results for each bar.
        """
        self.reset()

        # Re-inject runner from prior day (after reset cleared signal state)
        if runner_state is not None:
            pos = runner_state.position
            if runner_state.direction == "long":
                self._long_position = pos
                self._long_pattern_type = runner_state.pattern_type
                self._long_signal = runner_state.signal
                self._long_entry_bar_idx = -runner_state.cumulative_bars
                self._current_position = pos
            else:
                self._short_position = pos
                self._short_pattern_type = runner_state.pattern_type
                self._short_signal = runner_state.signal
                self._short_entry_bar_idx = -runner_state.cumulative_bars

        if session_date is None:
            session_date = df.index[0].to_pydatetime()

        # Expire stale ATM level records at session rollover
        if self.strategy_params.level_memory_days > 0:
            session_str = session_date.strftime("%Y-%m-%d")
            self.signal_aggregator.expire_atm_levels(
                session_str, self.strategy_params.level_memory_days
            )

        # Skip Tuesdays if configured (legacy — use regime filter instead)
        if self.risk_params.skip_tuesdays and session_date.weekday() == 1:
            return []

        # Compute regime for today (no lookahead — uses prior daily bars)
        if (self.strategy_params.use_regime_filter
                and self._daily_history is not None
                and len(self._daily_history) >= 50):
            self._regime_state = compute_regime(
                self._daily_history, self.regime_params
            )
        else:
            # Not enough history or filter disabled — allow both directions
            self._regime_state = RegimeState(
                direction=Direction.NEUTRAL,
                vol_regime=VolRegime.NORMAL,
                longs_enabled=True,
                shorts_enabled=True,
            )

        # Initialize session
        self.position_manager.start_session(session_date)
        self.signal_aggregator.initialize_levels(df, prior_day_df)

        # Re-inject persistent multi-day levels (after reset cleared the store)
        if (self.strategy_params.level_memory_days > 0
                and self._persistent_levels):
            today = session_date.date() if isinstance(session_date, datetime) else session_date
            # Filter out levels older than level_memory_days trading days
            cutoff_days = self.strategy_params.level_memory_days
            fresh_levels = [
                lv for lv in self._persistent_levels
                if lv.origin_date is not None
                and (today - lv.origin_date).days <= cutoff_days * 7 // 5 + 2  # rough calendar→trading conversion
            ]
            if fresh_levels:
                self.signal_aggregator.level_store.inject_levels(fresh_levels)
                logger.debug(
                    f"Injecting {len(fresh_levels)} persistent levels "
                    f"(scores: {[round(l.significance_score, 2) for l in fresh_levels]})"
                )
            # Update persistent list to only keep non-expired
            self._persistent_levels = fresh_levels

        # Set prior day range for volatility filter
        if prior_day_df is not None and len(prior_day_df) > 0:
            prior_range = float(prior_day_df["high"].max() - prior_day_df["low"].min())
            self.risk_manager.set_prior_day_range(prior_range)
            # Set PDL for Mode 1 detector
            if self.strategy_params.use_mode1_detection:
                pdl = float(prior_day_df["low"].min())
                self.mode1_detector.set_pdl(pdl)
            # Set PDH for Mode 1 Green detector
            if self.strategy_params.use_mode1_green_detection:
                pdh = float(prior_day_df["high"].max())
                self.mode1_green_detector.set_pdh(pdh)
        else:
            self.risk_manager.set_prior_day_range(0.0)

        # Enrich data
        enriched = enrich_dataframe(df)
        velocity = enriched["velocity_5"]

        # Pre-extract arrays for fast per-bar access (avoid pandas iat overhead)
        timestamps = df.index.to_pydatetime()
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        volumes = df["volume"].values
        vels = velocity.values
        n = len(df)

        results: list[BarResult] = []

        # Overnight gap check: if runner carried from prior day gaps below stop
        if runner_state is not None and n > 0:
            first_open = float(opens[0])
            gap_closed = False
            if runner_state.direction == "long" and self._long_position is not None:
                if first_open <= self._long_position.stop_price:
                    # Gap below stop — close at open (realistic slippage)
                    pos = self._long_position
                    pnl = (first_open - pos.entry_price) * pos.remaining_contracts
                    pos.realized_pnl_pts += pnl
                    pos.remaining_contracts = 0
                    pos.phase = ExitPhase.CLOSED
                    record = self.position_manager.close_position(
                        exit_price=first_open, timestamp=timestamps[0],
                        exit_reason="Overnight gap below stop",
                        pattern_type=self._long_pattern_type,
                        signal=self._long_signal,
                        entry_bar_idx=self._long_entry_bar_idx, exit_bar_idx=0,
                    )
                    if record is not None:
                        record.direction = "long"
                        record.is_runner = True
                        record.entry_date = runner_state.entry_date
                        record.days_held = runner_state.cumulative_bars // 390 + 1
                    self._long_position = None
                    self._current_position = None
                    gap_closed = True
            elif runner_state.direction == "short" and self._short_position is not None:
                if first_open >= self._short_position.stop_price:
                    pos = self._short_position
                    pnl = (pos.entry_price - first_open) * pos.remaining_contracts
                    pos.realized_pnl_pts += pnl
                    pos.remaining_contracts = 0
                    pos.phase = ExitPhase.CLOSED
                    record = self.position_manager.close_position(
                        exit_price=first_open, timestamp=timestamps[0],
                        exit_reason="Overnight gap above stop",
                        pattern_type=self._short_pattern_type,
                        signal=self._short_signal,
                        entry_bar_idx=self._short_entry_bar_idx, exit_bar_idx=0,
                    )
                    if record is not None:
                        record.direction = "short"
                        record.is_runner = True
                        record.entry_date = runner_state.entry_date
                    self._short_position = None
                    gap_closed = True

        for i in range(n):
            vel = float(vels[i])
            if vel != vel:  # fast NaN check
                vel = 0.0
            result = self._process_bar(
                bar_idx=i,
                timestamp=timestamps[i],
                open_=float(opens[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                volume=float(volumes[i]),
                velocity=vel,
                df=df,
            )
            results.append(result)

        # EOD processing: flatten non-runners, leave runners for carry
        for pos, ptype, sig, entry_idx, direction in [
            (self._long_position, self._long_pattern_type, self._long_signal,
             self._long_entry_bar_idx, "long"),
            (self._short_position, self._short_pattern_type, self._short_signal,
             self._short_entry_bar_idx, "short"),
        ]:
            if pos is None or not pos.is_open:
                continue
            if pos.phase in (ExitPhase.AFTER_T1, ExitPhase.AFTER_T2):
                # Runner survives EOD — BacktestRunner will extract it
                pass
            else:
                # Non-runner: flatten at last close
                last_close = float(closes[-1])
                contracts = pos.remaining_contracts
                if direction == "long":
                    pnl = (last_close - pos.entry_price) * contracts
                else:
                    pnl = (pos.entry_price - last_close) * contracts
                pos.realized_pnl_pts += pnl
                pos.remaining_contracts = 0
                pos.phase = ExitPhase.CLOSED
                record = self.position_manager.close_position(
                    exit_price=last_close, timestamp=timestamps[-1],
                    exit_reason="EOD flatten",
                    pattern_type=ptype, signal=sig,
                    entry_bar_idx=entry_idx, exit_bar_idx=n - 1,
                )
                if record is not None:
                    record.direction = direction
                if direction == "long":
                    self._long_position = None
                    self._current_position = None
                else:
                    self._short_position = None

        # Score existing persistent levels based on today's price action, then snapshot
        if self.strategy_params.level_memory_days > 0:
            self._score_persistent_levels(df)
            self._snapshot_persistent_levels(session_date)

        self._results = results
        return results

    def _score_persistent_levels(self, bars_df: pd.DataFrame) -> None:
        """Score persistent levels based on today's price action.

        Applies daily decay and checks if levels were tested/held today.
        Must be called BEFORE _snapshot_persistent_levels().
        """
        if not self._persistent_levels:
            return

        decay = self.strategy_params.level_decay_rate

        for level in self._persistent_levels:
            # Apply daily decay
            level.significance_score *= decay

            # Check if tested today
            is_support = level.level_type.name in (
                'PRIOR_DAY_LOW', 'MULTI_HOUR_LOW', 'SWING_LOW',
            )
            if is_support:
                # Tested if price came within 3 pts from above
                tested = (bars_df['low'] <= level.price + 3.0).any()
                # Held if price then went 5+ pts above
                held = tested and (bars_df['high'] >= level.price + 5.0).any()
            else:
                # Resistance: tested if price came within 3 pts from below
                tested = (bars_df['high'] >= level.price - 3.0).any()
                # Held if price then went 5+ pts below
                held = tested and (bars_df['low'] <= level.price - 5.0).any()

            if tested:
                level.touch_count += 1
            if held:
                level.tested_and_held = True
                level.significance_score = min(level.significance_score * 1.5, 2.0)

    def _snapshot_persistent_levels(self, session_date) -> None:
        """Snapshot today's significant levels into _persistent_levels.

        Score-based persistence: only keeps levels that are proven significant
        by type, touch count, and significance score. Caps total to prevent
        accumulation of noise.

        Eligible types (NO clusters, NO horizontals):
        - PRIOR_DAY_LOW / PRIOR_DAY_HIGH
        - MULTI_HOUR_LOW / MULTI_HOUR_HIGH
        - SWING_LOW / SWING_HIGH

        Each level gets tagged with origin_date for aging.
        """
        today = session_date.date() if isinstance(session_date, datetime) else session_date
        store = self.signal_aggregator.level_store
        params = self.strategy_params

        # Only these types qualify for multi-day persistence
        _PERSISTABLE_TYPES = {
            LevelType.PRIOR_DAY_LOW,
            LevelType.PRIOR_DAY_HIGH,
            LevelType.MULTI_HOUR_LOW,
            LevelType.MULTI_HOUR_HIGH,
            LevelType.SWING_LOW,
            LevelType.SWING_HIGH,
        }

        new_persistent: list[Level] = []

        # Keep existing persistent levels that still meet thresholds
        for level in self._persistent_levels:
            if (level.significance_score >= params.level_persist_min_score
                    and level.touch_count >= params.level_persist_min_touches):
                new_persistent.append(level)

        # Collect already-persistent prices for dedup
        existing_prices = {lv.price for lv in new_persistent}

        # Consider new levels from today's session
        for level in store.levels:
            if not level.is_active:
                continue
            # Skip if already persistent (origin_date set means it was injected)
            if level.origin_date is not None:
                continue
            # Must be a persistable type
            if level.level_type not in _PERSISTABLE_TYPES:
                continue
            # Must have enough touches
            if level.touch_count < params.level_persist_min_touches:
                continue

            # Dedup: skip if within 1 pt of an existing persistent level
            too_close = any(abs(level.price - p) <= 1.0 for p in existing_prices)
            if too_close:
                continue

            # Set initial persistence metadata
            level.origin_date = today
            level.significance_score = 1.0
            new_persistent.append(level)
            existing_prices.add(level.price)

        # Hard calendar cutoff (backup safety)
        cutoff_days = params.level_memory_days
        new_persistent = [
            lv for lv in new_persistent
            if lv.origin_date is not None
            and (today - lv.origin_date).days <= cutoff_days * 7 // 5 + 2
        ]

        # Cap to top N by significance_score
        if len(new_persistent) > params.max_persistent_levels:
            new_persistent.sort(key=lambda l: l.significance_score, reverse=True)
            new_persistent = new_persistent[:params.max_persistent_levels]

        self._persistent_levels = new_persistent

    def get_runner_state(self) -> Optional[object]:
        """Extract surviving runner position for cross-day carry.

        Returns a dict with runner state if a position in AFTER_T1/AFTER_T2
        is still open, else None. The BacktestRunner wraps this into
        RunnerCarryState.
        """
        for pos, ptype, sig, entry_idx, direction in [
            (self._long_position, self._long_pattern_type, self._long_signal,
             self._long_entry_bar_idx, "long"),
            (self._short_position, self._short_pattern_type, self._short_signal,
             self._short_entry_bar_idx, "short"),
        ]:
            if pos is not None and pos.is_open:
                if pos.phase in (ExitPhase.AFTER_T1, ExitPhase.AFTER_T2):
                    return {
                        "position": pos,
                        "pattern_type": ptype,
                        "signal": sig,
                        "entry_bar_idx": entry_idx,
                        "direction": direction,
                    }
        return None

    def _has_bullish_bias(self) -> bool:
        """Check if recent price action shows bullish bias for a long entry.

        Uses a 5-bar aggregate candle (open of bar -5, high/low of range,
        close of bar -1) to judge bias on a ~5 minute timeframe.
        Single 1-min candles are too noisy — this smooths the signal.

        Returns False (skip entry) only on clear bearish structure:
        - Close in bottom 30% of 5-bar range with significant range
        """
        if len(self._recent_bars) < 5:
            return True

        bars = self._recent_bars[-5:]
        o = bars[0][0]  # open of first bar
        h = max(b[1] for b in bars)  # highest high
        l = min(b[2] for b in bars)  # lowest low
        c = bars[-1][3]  # close of last bar

        bar_range = h - l
        if bar_range < self.strategy_params.candle_bias_min_range_pts:
            return True  # Too small to judge

        close_position = (c - l) / bar_range
        threshold = self.strategy_params.candle_bias_bearish_threshold

        # Only reject on clear bearish structure
        if close_position <= threshold:
            return False

        return True

    def _has_bearish_bias(self) -> bool:
        """Check if recent price action shows bearish bias for a short entry.

        Mirror of _has_bullish_bias: for shorts, we want to see prior candles
        were BULLISH (price rallied up, now failing). If the 5-bar aggregate
        shows bullish structure, that CONFIRMS the short setup.

        Returns False (skip short entry) only when prior candles are bearish
        (meaning the selloff already happened — late to the party).
        """
        if len(self._recent_bars) < 5:
            return True

        bars = self._recent_bars[-5:]
        h = max(b[1] for b in bars)
        l = min(b[2] for b in bars)
        c = bars[-1][3]

        bar_range = h - l
        if bar_range < self.strategy_params.short_candle_bias_min_range_pts:
            return True

        close_position = (c - l) / bar_range
        threshold = self.strategy_params.short_candle_bias_bullish_threshold

        # Only take short when prior candles are bullish (close in top portion)
        if close_position >= threshold:
            return True  # Bullish prior candles = good for short (rally then fail)

        return False  # Bearish prior candles = bad for short (already sold off)

    def _process_bar(
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
    ) -> BarResult:
        """Process a single bar through the full strategy pipeline.

        Supports concurrent long + short positions.
        """
        result = BarResult(bar_idx=bar_idx, timestamp=timestamp)
        current_time = timestamp.time()

        # Step 1a: Check for exit on long position
        if self._long_position is not None and self._long_position.is_open:
            # FB time-based exit: force close after max hold bars
            fb_time_exit = False
            if (self._long_pattern_type == "failed_breakdown"
                    and self._long_position.phase == ExitPhase.INITIAL
                    and self.exit_params.fb_max_hold_bars > 0
                    and bar_idx - self._long_entry_bar_idx >= self.exit_params.fb_max_hold_bars):
                fb_time_exit = True

            exit_action = self.exit_manager.update(
                self._long_position, high, low, close
            )

            # If time exit triggered and no stop/target hit, force close at market
            if fb_time_exit and (exit_action is None or self._long_position.is_open):
                contracts = self._long_position.remaining_contracts
                pnl = (close - self._long_position.entry_price) * contracts
                self._long_position.realized_pnl_pts += pnl
                self._long_position.remaining_contracts = 0
                self._long_position.phase = ExitPhase.CLOSED
                exit_action = ExitAction(
                    contracts_to_close=contracts,
                    exit_price=close,
                    new_stop=0.0,
                    new_phase=ExitPhase.CLOSED,
                    reason=f"FB time exit ({self.exit_params.fb_max_hold_bars} bars)",
                )

            if exit_action is not None:
                result.exit_action = exit_action
                logger.debug(
                    f"Bar {bar_idx}: Long exit - {exit_action.reason} "
                    f"({exit_action.contracts_to_close} contracts @ {exit_action.exit_price:.2f})"
                )
                if not self._long_position.is_open:
                    record = self.position_manager.close_position(
                        exit_price=exit_action.exit_price,
                        timestamp=timestamp,
                        exit_reason=exit_action.reason,
                        pattern_type=self._long_pattern_type,
                        signal=self._long_signal,
                        entry_bar_idx=self._long_entry_bar_idx,
                        exit_bar_idx=bar_idx,
                    )
                    if record is not None:
                        record.direction = "long"
                        result.trade_record = record
                        logger.info(
                            f"Long closed: {record.pnl_pts:.1f} pts "
                            f"({record.exit_reason})"
                        )
                        if "Stop" in exit_action.reason and self._long_signal is not None:
                            level_price = self._long_signal.pattern.level.price
                            stop_price = self._long_signal.pattern.stop_price
                            entry_price_val = self._long_signal.pattern.entry_price
                            level_type = getattr(self._long_signal.pattern.level, 'level_type', None)
                            level_type_str = level_type.name if level_type else ""
                            self.signal_aggregator.failed_breakdown.record_stop_out(
                                level_price, bar_idx, stop_price=stop_price,
                                entry_price=entry_price_val, level_type=level_type_str
                            )
                        # ATM level tracking: record outcome at this level
                        if self._long_signal is not None:
                            session_str = timestamp.strftime("%Y-%m-%d")
                            self.signal_aggregator.record_level_outcome(
                                self._long_signal.pattern.level.price,
                                record.pnl_pts,
                                session_str,
                            )
                    self._long_position = None
                    self._long_signal = None
                    self._current_position = None

        # Step 1b: Check for exit on short position
        if self._short_position is not None and self._short_position.is_open:
            exit_action = self.exit_manager.update(
                self._short_position, high, low, close
            )
            if exit_action is not None:
                if result.exit_action is None:
                    result.exit_action = exit_action
                logger.debug(
                    f"Bar {bar_idx}: Short exit - {exit_action.reason} "
                    f"({exit_action.contracts_to_close} contracts @ {exit_action.exit_price:.2f})"
                )
                if not self._short_position.is_open:
                    record = self.position_manager.close_position(
                        exit_price=exit_action.exit_price,
                        timestamp=timestamp,
                        exit_reason=exit_action.reason,
                        pattern_type=self._short_pattern_type,
                        signal=self._short_signal,
                        entry_bar_idx=self._short_entry_bar_idx,
                        exit_bar_idx=bar_idx,
                    )
                    if record is not None:
                        record.direction = "short"
                        if result.trade_record is None:
                            result.trade_record = record
                        logger.info(
                            f"Short closed: {record.pnl_pts:.1f} pts "
                            f"({record.exit_reason})"
                        )
                        if "Stop" in exit_action.reason and self._short_signal is not None:
                            level_price = self._short_signal.pattern.level.price
                            self.signal_aggregator.failed_rally.record_stop_out(
                                level_price, bar_idx
                            )
                        # ATM level tracking: record outcome at this level
                        if self._short_signal is not None:
                            session_str = timestamp.strftime("%Y-%m-%d")
                            self.signal_aggregator.record_level_outcome(
                                self._short_signal.pattern.level.price,
                                record.pnl_pts,
                                session_str,
                            )
                    self._short_position = None
                    self._short_signal = None

        # Step 2: Check if session is done
        if self.position_manager.is_done_for_day:
            self._recent_bars.append((open_, high, low, close))
            if len(self._recent_bars) > 10:
                self._recent_bars = self._recent_bars[-10:]
            return result

        # Step 3: Detect signals (always run — may get long or short signals)
        signal = self.signal_aggregator.update(
            bar_idx=bar_idx,
            timestamp=timestamp,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            velocity=velocity,
            df=df,
        )

        # Step 3b: Update Mode 1 detector (runs every bar regardless of signal)
        if self.strategy_params.use_mode1_detection:
            was_mode1 = self.mode1_detector.state.is_mode1_red
            self.mode1_detector.update(
                bar_idx=bar_idx,
                close=close,
                low=low,
                level_store=self.signal_aggregator.level_store,
                timestamp=timestamp,
            )
            # Shadow log on transition to MODE_1_RED
            if self.mode1_detector.state.is_mode1_red and not was_mode1:
                state = self.mode1_detector.state
                self.signal_aggregator.shadow_events.append({
                    "feature": "mode1",
                    "bar_idx": bar_idx,
                    "timestamp": str(timestamp),
                    "state": "MODE_1_RED",
                    "event": "transition",
                    "levels_broken": state.levels_broken_sustained,
                    "bars_below_pdl": state.bars_below_pdl,
                    "bearish_pressure_bars": state.bearish_pressure_bars,
                    "conditions_met": state.conditions_met,
                })

        # Step 3b-green: Update Mode 1 Green detector (trend up day)
        if self.strategy_params.use_mode1_green_detection:
            was_green = self.mode1_green_detector.state.is_mode1_green
            self.mode1_green_detector.update(
                bar_idx=bar_idx,
                close=close,
                high=high,
                level_store=self.signal_aggregator.level_store,
                timestamp=timestamp,
            )
            # Expose current Mode 1 Green status so _qualify_signal can apply
            # the relaxed R:R floor. Only live mode (not shadow) takes effect.
            self.signal_aggregator.mode1_green_active = (
                self.mode1_green_detector.state.is_mode1_green
            )
            if self.mode1_green_detector.state.is_mode1_green and not was_green:
                g = self.mode1_green_detector.state
                self.signal_aggregator.shadow_events.append({
                    "feature": "mode1_green",
                    "bar_idx": bar_idx,
                    "timestamp": str(timestamp),
                    "state": "MODE_1_GREEN",
                    "event": "transition",
                    "resistances_broken": g.resistances_broken_sustained,
                    "bars_above_pdh": g.bars_above_pdh,
                    "bullish_pressure_bars": g.bullish_pressure_bars,
                    "conditions_met": g.conditions_met,
                })

        if signal is not None:
            result.signal = signal
            direction = signal.pattern.direction

            # Flag risky trend-day FBs: fired within N pts of session high.
            # Mancini Apr 15 2026: "FBs not far off major highs after big rally
            # are dangerous — tend to fakeout unless parabolic rally sustains."
            if (signal.signal_type == SignalType.FAILED_BREAKDOWN
                    and direction == "long"):
                session_high = self.signal_aggregator._session_high
                if session_high != float('-inf'):
                    dist = session_high - signal.entry_price
                    if 0 <= dist <= self.strategy_params.risky_trend_fb_distance_from_high_pts:
                        signal.pattern.is_risky_trend_fb = True
                        logger.info(
                            f"Bar {bar_idx}: Risky trend-day FB — {dist:.1f} pts "
                            f"below session_high={session_high:.2f}"
                        )

            logger.info(
                f"Bar {bar_idx}: Signal - {signal.signal_type.name} ({direction}) "
                f"entry={signal.entry_price:.2f} stop={signal.stop_price:.2f} "
                f"R:R={signal.rr_ratio_t1:.2f}"
            )

            # Step 3b-g: Mode 1 Green — relax R:R and apply size factor on
            # confirmed trend-up days (FB LONG only). Ship in shadow mode first.
            if (self.strategy_params.use_mode1_green_detection
                    and self.mode1_green_detector.state.is_mode1_green
                    and direction == "long"
                    and signal.signal_type == SignalType.FAILED_BREAKDOWN):
                g = self.mode1_green_detector.state
                green_min_rr = self.strategy_params.mode1_green_fb_min_rr
                green_size = self.strategy_params.mode1_green_size_factor
                self.signal_aggregator.shadow_events.append({
                    "feature": "mode1_green",
                    "bar_idx": bar_idx,
                    "timestamp": str(timestamp),
                    "state": "MODE_1_GREEN",
                    "signal_type": signal.signal_type.name,
                    "rr_ratio": round(signal.rr_ratio_t1, 2),
                    "would_relax_min_rr_to": green_min_rr,
                    "would_apply_size_factor": green_size,
                    "resistances_broken": g.resistances_broken_sustained,
                    "bars_above_pdh": g.bars_above_pdh,
                    "bullish_pressure_bars": g.bullish_pressure_bars,
                    "conditions_met": g.conditions_met,
                })
                if not self.strategy_params.shadow_mode_features:
                    # Live: apply size factor. The relaxed R:R floor
                    # (mode1_green_fb_min_rr) is consumed upstream by
                    # _qualify_signal via the Mode 1 Green hook — here we
                    # just size the confirmed trend-day signal.
                    signal.position_size_factor *= green_size
                    logger.info(
                        f"Bar {bar_idx}: MODE 1 GREEN — applying size_factor "
                        f"{green_size:.2f} (new={signal.position_size_factor:.2f})"
                    )
                else:
                    logger.info(
                        f"SHADOW mode1_green @ bar {bar_idx}: MODE_1_GREEN "
                        f"would apply size={green_size:.2f}, min_rr={green_min_rr:.2f} "
                        f"(not acting)"
                    )

            # Step 3c: Mode 1 Red gating — reduce size or reject FB longs
            if (self.strategy_params.use_mode1_detection
                    and self.mode1_detector.state.is_mode1_red
                    and direction == "long"):
                state = self.mode1_detector.state
                reduction = self.strategy_params.mode1_size_reduction
                would_reject = (
                    self.strategy_params.mode1_disable_fb_longs
                    and signal.signal_type == SignalType.FAILED_BREAKDOWN
                )
                # Shadow log for Mode 1 (always emitted when Mode 1 triggers)
                self.signal_aggregator.shadow_events.append({
                    "feature": "mode1",
                    "bar_idx": bar_idx,
                    "timestamp": str(timestamp),
                    "state": "MODE_1_RED",
                    "signal_type": signal.signal_type.name,
                    "would_reject": would_reject,
                    "would_reduce_to": reduction if not would_reject else 0.0,
                    "levels_broken": state.levels_broken_sustained,
                    "bars_below_pdl": state.bars_below_pdl,
                    "bearish_pressure_bars": state.bearish_pressure_bars,
                    "conditions_met": state.conditions_met,
                })
                if not self.strategy_params.shadow_mode_features:
                    # Live mode: actually reject or reduce
                    if would_reject:
                        logger.warning(
                            f"Bar {bar_idx}: MODE 1 RED — FB long rejected "
                            f"(mode1_disable_fb_longs=True)"
                        )
                        self._recent_bars.append((open_, high, low, close))
                        if len(self._recent_bars) > 10:
                            self._recent_bars = self._recent_bars[-10:]
                        return result
                    # Apply size reduction to the signal
                    signal.position_size_factor *= reduction
                    logger.warning(
                        f"Bar {bar_idx}: MODE 1 RED detected — reducing long size "
                        f"(factor={signal.position_size_factor:.2f})"
                    )
                else:
                    logger.info(
                        f"SHADOW mode1 @ bar {bar_idx}: MODE_1_RED "
                        f"{'would REJECT' if would_reject else f'would reduce to {reduction:.0%}'} "
                        f"{signal.signal_type.name} (not acting)"
                    )

            # Step 4a: Regime filter gating
            if self._regime_state is not None:
                # If regime_filter_patterns is set, only gate those specific patterns
                gated_patterns = self.strategy_params.regime_filter_patterns
                apply_regime = (
                    not gated_patterns  # empty = gate all
                    or signal.signal_type.name in gated_patterns
                )
                if apply_regime:
                    if direction == "long" and not self._regime_state.longs_enabled:
                        logger.debug(f"Bar {bar_idx}: Long {signal.signal_type.name} rejected by regime filter (BEAR)")
                        self._recent_bars.append((open_, high, low, close))
                        if len(self._recent_bars) > 10:
                            self._recent_bars = self._recent_bars[-10:]
                        return result
                    if direction == "short" and not self._regime_state.shorts_enabled:
                        logger.debug(f"Bar {bar_idx}: Short {signal.signal_type.name} rejected by regime filter (BULL)")
                        self._recent_bars.append((open_, high, low, close))
                        if len(self._recent_bars) > 10:
                            self._recent_bars = self._recent_bars[-10:]
                        return result

            # Step 4b: Intraday price action context gating
            if self.strategy_params.use_intraday_context:
                from core.intraday_context import IntradayState
                idc_state = self.signal_aggregator.intraday_state
                if direction == "long" and idc_state == IntradayState.BEARISH_PRESSURE:
                    logger.debug(
                        f"Bar {bar_idx}: Long {signal.signal_type.name} rejected by "
                        f"intraday context (BEARISH_PRESSURE — LH/LL or weak bounces)"
                    )
                    self._recent_bars.append((open_, high, low, close))
                    if len(self._recent_bars) > 10:
                        self._recent_bars = self._recent_bars[-10:]
                    return result
                if direction == "short" and idc_state == IntradayState.BULLISH_PRESSURE:
                    logger.debug(
                        f"Bar {bar_idx}: Short {signal.signal_type.name} rejected by "
                        f"intraday context (BULLISH_PRESSURE — HH/HL)"
                    )
                    self._recent_bars.append((open_, high, low, close))
                    if len(self._recent_bars) > 10:
                        self._recent_bars = self._recent_bars[-10:]
                    return result

            # Step 4c: Check if we already have a position in this direction
            if direction == "long" and self._long_position is not None and self._long_position.is_open:
                self._recent_bars.append((open_, high, low, close))
                if len(self._recent_bars) > 10:
                    self._recent_bars = self._recent_bars[-10:]
                return result
            if direction == "short" and self._short_position is not None and self._short_position.is_open:
                self._recent_bars.append((open_, high, low, close))
                if len(self._recent_bars) > 10:
                    self._recent_bars = self._recent_bars[-10:]
                return result

            # Step 5: Candle bias filter
            if direction == "long":
                if self.strategy_params.candle_bias_filter and not self._has_bullish_bias():
                    logger.debug(f"Bar {bar_idx}: Candle bias filter — bearish, skipping long")
                    self._recent_bars.append((open_, high, low, close))
                    if len(self._recent_bars) > 10:
                        self._recent_bars = self._recent_bars[-10:]
                    return result
            else:  # short
                if self.strategy_params.short_candle_bias_filter and not self._has_bearish_bias():
                    logger.debug(f"Bar {bar_idx}: Short candle bias — not bullish enough, skipping short")
                    self._recent_bars.append((open_, high, low, close))
                    if len(self._recent_bars) > 10:
                        self._recent_bars = self._recent_bars[-10:]
                    return result

            # Step 6: Risk check
            risk_check = self.risk_manager.validate_entry(
                signal, current_time, self.position_manager
            )
            if not risk_check.passed:
                logger.debug(f"Risk check failed: {risk_check.reason}")
                self._recent_bars.append((open_, high, low, close))
                if len(self._recent_bars) > 10:
                    self._recent_bars = self._recent_bars[-10:]
                return result

            # Step 7: Entry decision
            entry = self.entry_manager.evaluate(
                signal=signal,
                current_time=current_time,
                trades_today=self.position_manager.trades_today,
                is_in_profit_protection=self.position_manager.is_profit_protection,
                daily_pnl_pts=self.position_manager.daily_pnl_pts,
            )
            result.entry_decision = entry

            if entry.should_enter:
                # Step 8: Open position
                position = self.exit_manager.create_position(
                    entry_price=entry.entry_price,
                    stop_price=entry.stop_price,
                    target_1=signal.target_1,
                    target_2=signal.target_2,
                    contracts=entry.contracts,
                    direction=direction,
                    is_double_dip=getattr(signal.pattern, 'is_double_dip', False),
                )
                accepted = self.position_manager.open_position(
                    position, timestamp, signal.pattern.pattern_type
                )
                if accepted:
                    if direction == "long":
                        self._long_position = position
                        self._long_pattern_type = signal.pattern.pattern_type
                        self._long_signal = signal
                        self._long_entry_bar_idx = bar_idx
                        self._current_position = position
                    else:
                        self._short_position = position
                        self._short_pattern_type = signal.pattern.pattern_type
                        self._short_signal = signal
                        self._short_entry_bar_idx = bar_idx
                    logger.info(
                        f"ENTRY ({direction.upper()}): {entry.contracts} contracts "
                        f"@ {entry.entry_price:.2f} stop={entry.stop_price:.2f} "
                        f"T1={signal.target_1:.2f} T2={signal.target_2:.2f}"
                    )

        # Update recent bars for candle bias filter
        self._recent_bars.append((open_, high, low, close))
        if len(self._recent_bars) > 10:
            self._recent_bars = self._recent_bars[-10:]
        return result

    # ------------------------------------------------------------------
    # VectorBT backtest array preparation
    # ------------------------------------------------------------------

    def prepare_backtest_arrays(
        self,
        df: pd.DataFrame,
        prior_day_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, np.ndarray]:
        """Run the strategy and produce flat arrays for VectorBT order_func_nb.

        Returns
        -------
        dict with keys:
            'signal_bars': int array, bar indices where signals fire
            'entry_prices': float array, entry prices per bar (0 if no entry)
            'stop_prices': float array
            'target_1': float array
            'target_2': float array
            'contracts': int array
            'signal_types': int array (0=none, 1=failed_breakdown, 2=level_reclaim)
        """
        results = self.run_day(df, prior_day_df)
        n = len(df)

        arrays = {
            "entry_prices": np.zeros(n, dtype=np.float64),
            "stop_prices": np.zeros(n, dtype=np.float64),
            "target_1": np.zeros(n, dtype=np.float64),
            "target_2": np.zeros(n, dtype=np.float64),
            "contracts": np.zeros(n, dtype=np.int32),
            "signal_types": np.zeros(n, dtype=np.int32),
        }

        for r in results:
            if r.entry_decision is not None and r.entry_decision.should_enter:
                i = r.bar_idx
                arrays["entry_prices"][i] = r.entry_decision.entry_price
                arrays["stop_prices"][i] = r.entry_decision.stop_price
                arrays["contracts"][i] = r.entry_decision.contracts

                if r.signal is not None:
                    arrays["target_1"][i] = r.signal.target_1
                    arrays["target_2"][i] = r.signal.target_2
                    arrays["signal_types"][i] = r.signal.signal_type.value

        return arrays

    # ------------------------------------------------------------------
    # Results access
    # ------------------------------------------------------------------

    @property
    def trade_records(self) -> list[TradeRecord]:
        """All completed trade records from the last run."""
        if self.position_manager.session is None:
            return []
        return self.position_manager.session.trades

    @property
    def total_pnl_pts(self) -> float:
        return sum(t.pnl_pts for t in self.trade_records)

    @property
    def total_pnl_dollars(self) -> float:
        return sum(t.pnl_dollars for t in self.trade_records)
