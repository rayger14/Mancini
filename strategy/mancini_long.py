"""Main orchestrator: bar-by-bar loop + VectorBT backtest array preparation.

Dual-mode execution:
1. Python objects for live trading (readable, debuggable)
2. Flat arrays for VectorBT Numba callbacks (fast backtesting)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time
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
from core.indicators import enrich_dataframe
from core.regime_filter import compute_regime, RegimeState, RegimeParams, Direction, VolRegime
from core.signals import Signal, SignalAggregator
from strategy.entry_manager import EntryManager, EntryDecision
from strategy.exit_manager import ExitManager, ExitAction, TradePosition
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
        self.position_manager = PositionManager(risk_params=risk_params)
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

    def reset(self) -> None:
        """Reset all state for a new session."""
        self.signal_aggregator.reset()
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

        Returns
        -------
        list[BarResult]
            Results for each bar.
        """
        self.reset()

        if session_date is None:
            session_date = df.index[0].to_pydatetime()

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

        # Set prior day range for volatility filter
        if prior_day_df is not None and len(prior_day_df) > 0:
            prior_range = float(prior_day_df["high"].max() - prior_day_df["low"].min())
            self.risk_manager.set_prior_day_range(prior_range)
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

        self._results = results
        return results

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
                    and self.exit_params.fb_max_hold_bars > 0
                    and bar_idx - self._long_entry_bar_idx >= self.exit_params.fb_max_hold_bars):
                fb_time_exit = True

            exit_action = self.exit_manager.update(
                self._long_position, high, low, close
            )

            # If time exit triggered and no stop/target hit, force close at market
            if fb_time_exit and (exit_action is None or self._long_position.is_open):
                from strategy.exit_manager import ExitAction, ExitPhase
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
                            self.signal_aggregator.failed_breakdown.record_stop_out(
                                level_price, bar_idx
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

        if signal is not None:
            result.signal = signal
            direction = signal.pattern.direction
            logger.info(
                f"Bar {bar_idx}: Signal - {signal.signal_type.name} ({direction}) "
                f"entry={signal.entry_price:.2f} stop={signal.stop_price:.2f} "
                f"R:R={signal.rr_ratio_t1:.2f}"
            )

            # Step 4a: Regime filter gating
            if self._regime_state is not None:
                if direction == "long" and not self._regime_state.longs_enabled:
                    logger.debug(f"Bar {bar_idx}: Long signal rejected by regime filter (BEAR)")
                    self._recent_bars.append((open_, high, low, close))
                    if len(self._recent_bars) > 10:
                        self._recent_bars = self._recent_bars[-10:]
                    return result
                if direction == "short" and not self._regime_state.shorts_enabled:
                    logger.debug(f"Bar {bar_idx}: Short signal rejected by regime filter (BULL)")
                    self._recent_bars.append((open_, high, low, close))
                    if len(self._recent_bars) > 10:
                        self._recent_bars = self._recent_bars[-10:]
                    return result

            # Step 4b: Check if we already have a position in this direction
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
