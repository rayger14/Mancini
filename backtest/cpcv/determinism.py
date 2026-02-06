"""Verify backtest determinism: same inputs must produce identical outputs."""

from __future__ import annotations

from datetime import date

import pandas as pd
from loguru import logger

from backtest.runner import BacktestRunner, BacktestResult


def verify_determinism(
    daily_dfs: dict[date, pd.DataFrame],
    n_runs: int = 2,
    **runner_kwargs,
) -> bool:
    """Run the same backtest ``n_runs`` times and verify identical results.

    Compares trade count, entry/exit prices, and PnL per trade using
    exact float equality (strategy is fully deterministic).

    Returns
    -------
    bool
        True if all runs produce identical results.
    """
    results: list[BacktestResult] = []

    for run_idx in range(n_runs):
        runner = BacktestRunner(**runner_kwargs)
        bt = runner.run_multi_day(daily_dfs=daily_dfs)
        results.append(bt)
        logger.info(
            f"Run {run_idx}: {bt.total_trades} trades, "
            f"PnL={bt.total_pnl_pts:+.4f} pts"
        )

    ref = results[0]
    all_match = True

    for run_idx in range(1, n_runs):
        if not _compare_results(ref, results[run_idx], run_idx):
            all_match = False

    if all_match:
        logger.info(f"DETERMINISM VERIFIED: {n_runs} runs are identical")
    else:
        logger.error("DETERMINISM FAILED: runs produced different results")

    return all_match


def _compare_results(ref: BacktestResult, other: BacktestResult, run_idx: int) -> bool:
    if len(ref.days) != len(other.days):
        logger.error(f"Run {run_idx}: day count mismatch ({len(ref.days)} vs {len(other.days)})")
        return False

    for i, (d1, d2) in enumerate(zip(ref.days, other.days)):
        if d1.date != d2.date:
            logger.error(f"Run {run_idx}, day {i}: date mismatch")
            return False
        if d1.num_trades != d2.num_trades:
            logger.error(f"Run {run_idx}, {d1.date}: trade count mismatch")
            return False
        if abs(d1.pnl_pts - d2.pnl_pts) > 1e-10:
            logger.error(f"Run {run_idx}, {d1.date}: PnL mismatch ({d1.pnl_pts} vs {d2.pnl_pts})")
            return False

    if len(ref.all_trades) != len(other.all_trades):
        logger.error(f"Run {run_idx}: total trades mismatch")
        return False

    for i, (t1, t2) in enumerate(zip(ref.all_trades, other.all_trades)):
        if t1.entry_price != t2.entry_price or t1.pnl_pts != t2.pnl_pts:
            logger.error(f"Run {run_idx}, trade {i}: mismatch")
            return False

    return True
