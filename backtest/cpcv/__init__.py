"""Combinatorial Purged Cross-Validation framework for Mancini strategy."""

from backtest.cpcv.splitter import CPCVSplitter, CPCVConfig, CPCVSplit
from backtest.cpcv.data_pipeline import fetch_es_hourly, split_into_daily_dfs
from backtest.cpcv.robustness import RobustnessTest, RobustnessResult
from backtest.cpcv.optimizer import CPCVOptimizer, OptimizationResult, ParamSet
from backtest.cpcv.determinism import verify_determinism
from backtest.cpcv.report import (
    robustness_report,
    optimization_report,
    plot_robustness_histograms,
)

__all__ = [
    "CPCVSplitter", "CPCVConfig", "CPCVSplit",
    "fetch_es_hourly", "split_into_daily_dfs",
    "RobustnessTest", "RobustnessResult",
    "CPCVOptimizer", "OptimizationResult", "ParamSet",
    "verify_determinism",
    "robustness_report", "optimization_report", "plot_robustness_histograms",
]
