"""Robustness validation: default params across all CPCV paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from loguru import logger

from backtest.runner import BacktestRunner, BacktestResult
from backtest.metrics import compute_metrics, StrategyMetrics
from backtest.cpcv.splitter import CPCVSplitter, CPCVSplit, CPCVConfig
from config.settings import (
    StrategyParams,
    ElevatorParams,
    ExitParams,
    RiskParams,
    DEFAULT_STRATEGY,
    DEFAULT_ELEVATOR,
    DEFAULT_EXIT,
    DEFAULT_RISK,
)


@dataclass
class PathResult:
    """Result from a single CPCV path."""

    split: CPCVSplit
    backtest_result: BacktestResult
    metrics: StrategyMetrics
    fold_type: str  # "test" or "train"


@dataclass
class RobustnessResult:
    """Aggregated results from the full robustness test."""

    config: CPCVConfig
    path_results: list[PathResult] = field(default_factory=list)

    @property
    def test_results(self) -> list[PathResult]:
        return [r for r in self.path_results if r.fold_type == "test"]

    @property
    def metrics_df(self) -> pd.DataFrame:
        rows = []
        for r in self.test_results:
            m = r.metrics
            rows.append({
                "split_id": r.split.split_id,
                "test_groups": r.split.test_group_indices,
                "n_test_days": len(r.split.test_dates),
                "n_trades": m.total_trades,
                "win_rate": m.win_rate,
                "profit_factor": m.profit_factor,
                "total_pnl_pts": m.total_pnl_pts,
                "sharpe_daily": m.sharpe_daily,
                "max_drawdown_pts": m.max_drawdown_pts,
                "expectancy_pts": m.expectancy_pts,
                "avg_trade_pts": m.avg_trade_pts,
            })
        return pd.DataFrame(rows)


class RobustnessTest:
    """Run strategy with fixed params across all CPCV test folds."""

    def __init__(
        self,
        daily_dfs: dict[date, pd.DataFrame],
        splitter: CPCVSplitter,
        strategy_params: StrategyParams = DEFAULT_STRATEGY,
        elevator_params: ElevatorParams = DEFAULT_ELEVATOR,
        exit_params: ExitParams = DEFAULT_EXIT,
        risk_params: RiskParams = DEFAULT_RISK,
        min_rr_ratio: float = 1.5,
    ):
        self.daily_dfs = daily_dfs
        self.splitter = splitter
        self.strategy_params = strategy_params
        self.elevator_params = elevator_params
        self.exit_params = exit_params
        self.risk_params = risk_params
        self.min_rr_ratio = min_rr_ratio

    def run(self, run_train_folds: bool = False) -> RobustnessResult:
        result = RobustnessResult(config=self.splitter.config)

        for split in self.splitter.splits():
            logger.info(
                f"Path {split.split_id + 1}/{self.splitter.num_paths}: "
                f"test groups {split.test_group_indices}, "
                f"{len(split.test_dates)} test days"
            )

            test_dfs = self._select(split.test_dates)
            if test_dfs:
                bt = self._run_backtest(test_dfs)
                result.path_results.append(PathResult(
                    split=split,
                    backtest_result=bt,
                    metrics=compute_metrics(bt),
                    fold_type="test",
                ))

            if run_train_folds:
                train_dfs = self._select(split.train_dates)
                if train_dfs:
                    bt = self._run_backtest(train_dfs)
                    result.path_results.append(PathResult(
                        split=split,
                        backtest_result=bt,
                        metrics=compute_metrics(bt),
                        fold_type="train",
                    ))

        return result

    def _select(self, dates: list[date]) -> dict[date, pd.DataFrame]:
        return {d: self.daily_dfs[d] for d in dates if d in self.daily_dfs}

    def _run_backtest(self, fold_dfs: dict[date, pd.DataFrame]) -> BacktestResult:
        runner = BacktestRunner(
            strategy_params=self.strategy_params,
            elevator_params=self.elevator_params,
            exit_params=self.exit_params,
            risk_params=self.risk_params,
            min_rr_ratio=self.min_rr_ratio,
        )
        return runner.run_multi_day(daily_dfs=fold_dfs)
