"""Fast backtest runner optimized for parameter optimization.

Key optimizations vs. the standard BacktestRunner:
  - Computes only velocity (not all 7 indicators from enrich_dataframe)
  - Skips BarResult object creation (just tracks trade records)
  - Suppresses all logging by default
  - Returns only the data needed for optimization scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

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
from core.indicators import compute_velocity
from core.signals import SignalAggregator
from strategy.entry_manager import EntryManager
from strategy.exit_manager import ExitManager, TradePosition
from strategy.position_manager import PositionManager, TradeRecord
from strategy.risk_manager import RiskManager


@dataclass
class FastDayResult:
    """Minimal day result for optimization."""

    date: date
    trade_records: list[TradeRecord]
    pnl_pts: float
    num_trades: int


@dataclass
class FastBacktestResult:
    """Minimal aggregated result for optimization scoring."""

    days: list[FastDayResult] = field(default_factory=list)
    all_trades: list[TradeRecord] = field(default_factory=list)

    @property
    def total_pnl_pts(self) -> float:
        return sum(d.pnl_pts for d in self.days)

    @property
    def total_trades(self) -> int:
        return len(self.all_trades)

    @property
    def win_rate(self) -> float:
        if not self.all_trades:
            return 0.0
        wins = sum(1 for t in self.all_trades if t.pnl_pts > 0)
        return wins / len(self.all_trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pts for t in self.all_trades if t.pnl_pts > 0)
        gross_loss = abs(sum(t.pnl_pts for t in self.all_trades if t.pnl_pts < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown_pts(self) -> float:
        if not self.days:
            return 0.0
        equity = np.cumsum([d.pnl_pts for d in self.days])
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        return float(dd.max()) if len(dd) > 0 else 0.0


class FastBacktestRunner:
    """Lean backtest runner for optimization loops.

    Uses the same signal/entry/exit/risk logic as BacktestRunner but
    avoids overhead that matters when running hundreds of trials.
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
        self.elevator_params = elevator_params
        self.exit_params = exit_params
        self.risk_params = risk_params
        self.session_times = session_times
        self.contract = contract
        self.min_rr_ratio = min_rr_ratio

    def _create_components(self):
        """Create fresh strategy components for a new day."""
        signal_agg = SignalAggregator(
            strategy_params=self.strategy_params,
            elevator_params=self.elevator_params,
            exit_params=self.exit_params,
            min_rr_ratio=self.min_rr_ratio,
        )
        entry_mgr = EntryManager(
            session=self.session_times,
            exit_params=self.exit_params,
            risk_params=self.risk_params,
        )
        exit_mgr = ExitManager(
            params=self.exit_params,
            contract=self.contract,
        )
        pos_mgr = PositionManager(risk_params=self.risk_params)
        risk_mgr = RiskManager(
            risk_params=self.risk_params,
            session=self.session_times,
            contract=self.contract,
        )
        return signal_agg, entry_mgr, exit_mgr, pos_mgr, risk_mgr

    def run_single_day(
        self,
        df: pd.DataFrame,
        prior_day_df: Optional[pd.DataFrame] = None,
        day: Optional[date] = None,
    ) -> FastDayResult:
        """Run strategy on one day, returning minimal result."""
        if day is None:
            day = df.index[0].date()

        signal_agg, entry_mgr, exit_mgr, pos_mgr, risk_mgr = self._create_components()

        # Initialize session
        pos_mgr.start_session(df.index[0].to_pydatetime())
        signal_agg.initialize_levels(df, prior_day_df)

        # Compute only velocity (skip atr, vwap, roc, etc.)
        velocity = compute_velocity(df, window=5)

        # Pre-extract numpy arrays for fast access
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        volumes = df["volume"].values
        vel_vals = velocity.values
        timestamps = df.index

        current_position: Optional[TradePosition] = None
        current_pattern_type = ""
        n = len(df)

        for i in range(n):
            ts = timestamps[i].to_pydatetime()
            o, h, lo, c, v = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i]), float(volumes[i])
            vel = float(vel_vals[i]) if not np.isnan(vel_vals[i]) else 0.0
            current_time = ts.time()

            # Exit check on existing position
            if current_position is not None and current_position.is_open:
                exit_action = exit_mgr.update(current_position, h, lo, c)
                if exit_action is not None and not current_position.is_open:
                    pos_mgr.close_position(
                        exit_price=exit_action.exit_price,
                        timestamp=ts,
                        exit_reason=exit_action.reason,
                        pattern_type=current_pattern_type,
                    )
                    current_position = None

            # Skip signal if position open
            if current_position is not None and current_position.is_open:
                continue

            if pos_mgr.is_done_for_day:
                continue

            # Signal detection
            signal = signal_agg.update(
                bar_idx=i, timestamp=ts,
                open_=o, high=h, low=lo, close=c,
                volume=v, velocity=vel, df=df,
            )

            if signal is None:
                continue

            # Risk check
            risk_check = risk_mgr.validate_entry(signal, current_time, pos_mgr)
            if not risk_check.passed:
                continue

            # Entry evaluation
            entry = entry_mgr.evaluate(
                signal=signal,
                current_time=current_time,
                trades_today=pos_mgr.trades_today,
                is_in_profit_protection=pos_mgr.is_profit_protection,
                daily_pnl_pts=pos_mgr.daily_pnl_pts,
            )

            if not entry.should_enter:
                continue

            # Open position
            position = exit_mgr.create_position(
                entry_price=entry.entry_price,
                stop_price=entry.stop_price,
                target_1=signal.target_1,
                target_2=signal.target_2,
                contracts=entry.contracts,
            )
            accepted = pos_mgr.open_position(position, ts, signal.pattern.pattern_type)
            if accepted:
                current_position = position
                current_pattern_type = signal.pattern.pattern_type

        # Gather results
        records = pos_mgr.session.trades if pos_mgr.session else []
        pnl_pts = sum(t.pnl_pts for t in records)

        return FastDayResult(
            date=day,
            trade_records=list(records),
            pnl_pts=pnl_pts,
            num_trades=len(records),
        )

    def run_multi_day(
        self,
        daily_dfs: dict[date, pd.DataFrame],
    ) -> FastBacktestResult:
        """Run strategy across multiple days."""
        result = FastBacktestResult()
        dates = sorted(daily_dfs.keys())
        prior_day_df: Optional[pd.DataFrame] = None

        for day in dates:
            df = daily_dfs[day]
            if len(df) < 10:
                continue

            day_result = self.run_single_day(df, prior_day_df, day)
            result.days.append(day_result)
            result.all_trades.extend(day_result.trade_records)
            prior_day_df = df

        return result
