"""Multi-day backtest engine.

Uses ManciniLongStrategy's bar-by-bar Python mode (strategy.run_day) for
each day, then aggregates results. VectorBT integration is available for
users with vectorbtpro installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
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
from core.market_data import MarketDataFetcher
from strategy.mancini_long import ManciniLongStrategy, BarResult
from strategy.position_manager import TradeRecord


@dataclass
class DayResult:
    """Results for a single trading day."""

    date: date
    bar_results: list[BarResult]
    trade_records: list[TradeRecord]
    pnl_pts: float
    pnl_dollars: float
    num_trades: int
    win_rate: float


@dataclass
class BacktestResult:
    """Aggregated results across multiple days."""

    days: list[DayResult] = field(default_factory=list)
    all_trades: list[TradeRecord] = field(default_factory=list)

    @property
    def total_pnl_pts(self) -> float:
        return sum(d.pnl_pts for d in self.days)

    @property
    def total_pnl_dollars(self) -> float:
        return sum(d.pnl_dollars for d in self.days)

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


class BacktestRunner:
    """Runs the Mancini strategy over multiple days."""

    def __init__(
        self,
        strategy_params: StrategyParams = DEFAULT_STRATEGY,
        elevator_params: ElevatorParams = DEFAULT_ELEVATOR,
        exit_params: ExitParams = DEFAULT_EXIT,
        risk_params: RiskParams = DEFAULT_RISK,
        session_times: SessionTimes = DEFAULT_SESSION,
        contract: ESContractSpec = DEFAULT_CONTRACT,
        data_fetcher: Optional[MarketDataFetcher] = None,
        min_rr_ratio: float = 1.5,
    ):
        self.strategy = ManciniLongStrategy(
            strategy_params=strategy_params,
            elevator_params=elevator_params,
            exit_params=exit_params,
            risk_params=risk_params,
            session_times=session_times,
            contract=contract,
            min_rr_ratio=min_rr_ratio,
        )
        self.data_fetcher = data_fetcher or MarketDataFetcher()

    def run_single_day(
        self,
        df: pd.DataFrame,
        prior_day_df: Optional[pd.DataFrame] = None,
        day: Optional[date] = None,
    ) -> DayResult:
        """Run the strategy on a single day of data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV bars for the day.
        prior_day_df : pd.DataFrame, optional
            Previous day's data for level initialization.
        day : date, optional
            Session date.

        Returns
        -------
        DayResult
        """
        if day is None:
            day = df.index[0].date()

        bar_results = self.strategy.run_day(df, prior_day_df)
        records = self.strategy.trade_records
        pnl_pts = self.strategy.total_pnl_pts
        pnl_dollars = self.strategy.total_pnl_dollars
        wins = sum(1 for r in records if r.pnl_pts > 0)
        wr = wins / len(records) if records else 0.0

        return DayResult(
            date=day,
            bar_results=bar_results,
            trade_records=list(records),
            pnl_pts=pnl_pts,
            pnl_dollars=pnl_dollars,
            num_trades=len(records),
            win_rate=wr,
        )

    def run_multi_day(
        self,
        symbol: str = "ES",
        start: Optional[date] = None,
        end: Optional[date] = None,
        daily_dfs: Optional[dict[date, pd.DataFrame]] = None,
    ) -> BacktestResult:
        """Run the strategy across multiple days.

        Parameters
        ----------
        symbol : str
            Futures symbol.
        start, end : date, optional
            Date range (used with data_fetcher).
        daily_dfs : dict, optional
            Pre-loaded DataFrames keyed by date (for testing without API).

        Returns
        -------
        BacktestResult
        """
        result = BacktestResult()

        if daily_dfs is not None:
            dates = sorted(daily_dfs.keys())
        elif start is not None and end is not None:
            dates = []
            d = start
            while d <= end:
                if d.weekday() < 5:  # skip weekends
                    dates.append(d)
                d += timedelta(days=1)
        else:
            raise ValueError("Provide either daily_dfs or start+end dates")

        prior_day_df: Optional[pd.DataFrame] = None

        for day in dates:
            try:
                if daily_dfs is not None:
                    df = daily_dfs[day]
                else:
                    df = self.data_fetcher.get_single_day(symbol, day)

                if len(df) < 10:
                    logger.warning(f"Skipping {day}: only {len(df)} bars")
                    continue

                day_result = self.run_single_day(df, prior_day_df, day)
                result.days.append(day_result)
                result.all_trades.extend(day_result.trade_records)

                logger.info(
                    f"{day}: {day_result.num_trades} trades, "
                    f"PnL={day_result.pnl_pts:+.1f} pts, "
                    f"WR={day_result.win_rate:.0%}"
                )

                prior_day_df = df

            except Exception as e:
                logger.error(f"Error on {day}: {e}")
                continue

        logger.info(
            f"\nBacktest complete: {len(result.days)} days, "
            f"{result.total_trades} trades, "
            f"PnL={result.total_pnl_pts:+.1f} pts (${result.total_pnl_dollars:+,.0f}), "
            f"WR={result.win_rate:.0%}, PF={result.profit_factor:.2f}, "
            f"MaxDD={result.max_drawdown_pts:.1f} pts"
        )

        return result
