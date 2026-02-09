"""Main orchestrator: bar-by-bar loop + VectorBT backtest array preparation.

Dual-mode execution:
1. Python objects for live trading (readable, debuggable)
2. Flat arrays for VectorBT Numba callbacks (fast backtesting)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
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
from core.indicators import compute_velocity, enrich_dataframe
from core.signals import Signal, SignalAggregator
from strategy.entry_manager import EntryManager, EntryDecision
from strategy.exit_manager import ExitManager, ExitAction, TradePosition, ExitPhase
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

        # State
        self._current_position: Optional[TradePosition] = None
        self._current_pattern_type: str = ""
        self._current_signal: Optional[Signal] = None
        self._entry_bar_idx: int = 0
        self._results: list[BarResult] = []

    def reset(self) -> None:
        """Reset all state for a new session."""
        self.signal_aggregator.reset()
        self._current_position = None
        self._current_pattern_type = ""
        self._current_signal = None
        self._entry_bar_idx = 0
        self._results.clear()

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

        # Initialize session
        self.position_manager.start_session(session_date)
        self.signal_aggregator.initialize_levels(df, prior_day_df)

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
        """Process a single bar through the full strategy pipeline."""
        result = BarResult(bar_idx=bar_idx, timestamp=timestamp)
        current_time = timestamp.time()

        # Step 1: Check for exit on existing position
        if self._current_position is not None and self._current_position.is_open:
            exit_action = self.exit_manager.update(
                self._current_position, high, low, close
            )
            if exit_action is not None:
                result.exit_action = exit_action
                logger.debug(
                    f"Bar {bar_idx}: Exit action - {exit_action.reason} "
                    f"({exit_action.contracts_to_close} contracts @ {exit_action.exit_price:.2f})"
                )

                # If position is fully closed, record the trade
                if not self._current_position.is_open:
                    record = self.position_manager.close_position(
                        exit_price=exit_action.exit_price,
                        timestamp=timestamp,
                        exit_reason=exit_action.reason,
                        pattern_type=self._current_pattern_type,
                        signal=self._current_signal,
                        entry_bar_idx=self._entry_bar_idx,
                        exit_bar_idx=bar_idx,
                    )
                    if record is not None:
                        result.trade_record = record
                        logger.info(
                            f"Trade closed: {record.pnl_pts:.1f} pts "
                            f"({record.exit_reason})"
                        )
                    self._current_position = None
                    self._current_signal = None

        # Step 2: Don't look for new signals if we already have a position
        if self._current_position is not None and self._current_position.is_open:
            return result

        # Step 3: Check if session is done
        if self.position_manager.is_done_for_day:
            return result

        # Step 4: Detect signals
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
            logger.info(
                f"Bar {bar_idx}: Signal - {signal.signal_type.name} "
                f"entry={signal.entry_price:.2f} stop={signal.stop_price:.2f} "
                f"R:R={signal.rr_ratio_t1:.2f}"
            )

            # Step 5: Risk check
            risk_check = self.risk_manager.validate_entry(
                signal, current_time, self.position_manager
            )
            if not risk_check.passed:
                logger.debug(f"Risk check failed: {risk_check.reason}")
                return result

            # Step 6: Entry decision
            entry = self.entry_manager.evaluate(
                signal=signal,
                current_time=current_time,
                trades_today=self.position_manager.trades_today,
                is_in_profit_protection=self.position_manager.is_profit_protection,
                daily_pnl_pts=self.position_manager.daily_pnl_pts,
            )
            result.entry_decision = entry

            if entry.should_enter:
                # Step 7: Open position
                position = self.exit_manager.create_position(
                    entry_price=entry.entry_price,
                    stop_price=entry.stop_price,
                    target_1=signal.target_1,
                    target_2=signal.target_2,
                    contracts=entry.contracts,
                )
                accepted = self.position_manager.open_position(
                    position, timestamp, signal.pattern.pattern_type
                )
                if accepted:
                    self._current_position = position
                    self._current_pattern_type = signal.pattern.pattern_type
                    self._current_signal = signal
                    self._entry_bar_idx = bar_idx
                    logger.info(
                        f"ENTRY: {entry.contracts} contracts @ {entry.entry_price:.2f} "
                        f"stop={entry.stop_price:.2f} T1={signal.target_1:.2f} T2={signal.target_2:.2f}"
                    )

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
