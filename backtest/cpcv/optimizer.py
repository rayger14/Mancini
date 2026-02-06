"""CPCV-based parameter optimization with overfitting detection."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from itertools import product

import numpy as np
import pandas as pd
from loguru import logger

from backtest.runner import BacktestRunner, BacktestResult
from backtest.metrics import compute_metrics, StrategyMetrics
from backtest.cpcv.splitter import CPCVSplitter
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


@dataclass(frozen=True)
class ParamSet:
    """A single point in the parameter grid."""

    min_rr_ratio: float = 1.5
    elevator_min_velocity: float = 2.0
    t1_exit_fraction: float = 0.75
    initial_stop_buffer_pts: float = 2.0

    def to_dict(self) -> dict[str, float]:
        return {
            "min_rr_ratio": self.min_rr_ratio,
            "elevator_min_velocity": self.elevator_min_velocity,
            "t1_exit_fraction": self.t1_exit_fraction,
            "initial_stop_buffer_pts": self.initial_stop_buffer_pts,
        }


DEFAULT_PARAM_GRID = {
    "min_rr_ratio": [1.0, 1.5, 2.0, 2.5],
    "elevator_min_velocity": [1.5, 2.0, 2.5, 3.0],
    "t1_exit_fraction": [0.50, 0.75, 1.0],
    "initial_stop_buffer_pts": [1.0, 2.0, 3.0],
}


@dataclass
class OptimizationTrialResult:
    split_id: int
    params: ParamSet
    train_metrics: StrategyMetrics
    test_metrics: StrategyMetrics
    objective_train: float
    objective_test: float


@dataclass
class OptimizationResult:
    """Full optimization results across all splits and param sets."""

    trials: list[OptimizationTrialResult] = field(default_factory=list)

    @property
    def results_df(self) -> pd.DataFrame:
        rows = []
        for t in self.trials:
            row = t.params.to_dict()
            row.update({
                "split_id": t.split_id,
                "train_sharpe": t.train_metrics.sharpe_daily,
                "test_sharpe": t.test_metrics.sharpe_daily,
                "train_pnl": t.train_metrics.total_pnl_pts,
                "test_pnl": t.test_metrics.total_pnl_pts,
                "train_pf": t.train_metrics.profit_factor,
                "test_pf": t.test_metrics.profit_factor,
                "train_wr": t.train_metrics.win_rate,
                "test_wr": t.test_metrics.win_rate,
                "obj_train": t.objective_train,
                "obj_test": t.objective_test,
            })
            rows.append(row)
        return pd.DataFrame(rows)

    def overfitting_probability(self) -> float:
        """Probability of Backtest Overfitting (PBO).

        For each split, the IS-best param set is checked OOS.
        PBO = fraction of splits where the IS-best has OOS objective <= 0.
        """
        if not self.trials:
            return 0.0
        df = self.results_df
        overfit_count = 0
        split_ids = df["split_id"].unique()
        for sid in split_ids:
            fold = df[df["split_id"] == sid]
            best_idx = fold["obj_train"].idxmax()
            if fold.loc[best_idx, "obj_test"] <= 0:
                overfit_count += 1
        return overfit_count / len(split_ids)

    def best_params_by_rank(self) -> ParamSet:
        """Find params with best average rank across test folds."""
        df = self.results_df.copy()
        param_cols = list(ParamSet().to_dict().keys())
        df["rank"] = df.groupby("split_id")["obj_test"].rank(
            ascending=False, method="average"
        )
        avg_ranks = df.groupby(param_cols)["rank"].mean()
        best_idx = avg_ranks.idxmin()
        best_dict = dict(zip(param_cols, best_idx))
        return ParamSet(**best_dict)


class CPCVOptimizer:
    """Grid search over parameter space with CPCV train/test validation."""

    def __init__(
        self,
        daily_dfs: dict[date, pd.DataFrame],
        splitter: CPCVSplitter,
        param_grid: dict[str, list[float]] | None = None,
        objective: str = "sharpe_daily",
        base_strategy_params: StrategyParams = DEFAULT_STRATEGY,
        base_elevator_params: ElevatorParams = DEFAULT_ELEVATOR,
        base_exit_params: ExitParams = DEFAULT_EXIT,
        base_risk_params: RiskParams = DEFAULT_RISK,
    ):
        self.daily_dfs = daily_dfs
        self.splitter = splitter
        self.param_grid = param_grid or DEFAULT_PARAM_GRID
        self.objective = objective
        self.base_strategy_params = base_strategy_params
        self.base_elevator_params = base_elevator_params
        self.base_exit_params = base_exit_params
        self.base_risk_params = base_risk_params

    def run(self, max_splits: int | None = None) -> OptimizationResult:
        result = OptimizationResult()
        param_sets = self._generate_param_sets()
        logger.info(
            f"Optimization: {len(param_sets)} param sets x "
            f"{self.splitter.num_paths} splits"
        )

        for i, split in enumerate(self.splitter.splits()):
            if max_splits is not None and i >= max_splits:
                break

            train_dfs = self._select(split.train_dates)
            test_dfs = self._select(split.test_dates)
            if not train_dfs or not test_dfs:
                continue

            logger.info(
                f"Split {split.split_id + 1}: "
                f"{len(train_dfs)} train, {len(test_dfs)} test days, "
                f"{len(param_sets)} params"
            )

            for params in param_sets:
                train_bt = self._run_with_params(train_dfs, params)
                test_bt = self._run_with_params(test_dfs, params)
                train_m = compute_metrics(train_bt)
                test_m = compute_metrics(test_bt)

                result.trials.append(OptimizationTrialResult(
                    split_id=split.split_id,
                    params=params,
                    train_metrics=train_m,
                    test_metrics=test_m,
                    objective_train=self._get_objective(train_m),
                    objective_test=self._get_objective(test_m),
                ))

        return result

    def _generate_param_sets(self) -> list[ParamSet]:
        keys = sorted(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        return [ParamSet(**dict(zip(keys, combo))) for combo in product(*values)]

    def _run_with_params(
        self, fold_dfs: dict[date, pd.DataFrame], params: ParamSet
    ) -> BacktestResult:
        sp = replace(self.base_strategy_params)
        ep = replace(
            self.base_elevator_params,
            min_velocity_pts_per_min=params.elevator_min_velocity,
        )
        xp = replace(
            self.base_exit_params,
            t1_exit_fraction=params.t1_exit_fraction,
            initial_stop_buffer_pts=params.initial_stop_buffer_pts,
        )
        runner = BacktestRunner(
            strategy_params=sp,
            elevator_params=ep,
            exit_params=xp,
            risk_params=self.base_risk_params,
            min_rr_ratio=params.min_rr_ratio,
        )
        return runner.run_multi_day(daily_dfs=fold_dfs)

    def _get_objective(self, m: StrategyMetrics) -> float:
        return getattr(m, self.objective, 0.0)

    def _select(self, dates: list[date]) -> dict[date, pd.DataFrame]:
        return {d: self.daily_dfs[d] for d in dates if d in self.daily_dfs}
