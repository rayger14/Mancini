"""Strategy-specific performance analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtest.runner import BacktestResult
from strategy.position_manager import TradeRecord


@dataclass
class StrategyMetrics:
    """Comprehensive strategy performance metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win_pts: float = 0.0
    avg_loss_pts: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pts: float = 0.0
    total_pnl_dollars: float = 0.0
    max_drawdown_pts: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_pts: float = 0.0
    sharpe_daily: float = 0.0
    expectancy_pts: float = 0.0

    # By pattern type
    fb_trades: int = 0
    fb_win_rate: float = 0.0
    lr_trades: int = 0
    lr_win_rate: float = 0.0

    # By time window
    morning_trades: int = 0
    morning_win_rate: float = 0.0
    afternoon_trades: int = 0
    afternoon_win_rate: float = 0.0


def compute_metrics(result: BacktestResult) -> StrategyMetrics:
    """Compute comprehensive strategy metrics from backtest results.

    Parameters
    ----------
    result : BacktestResult
        Output from BacktestRunner.

    Returns
    -------
    StrategyMetrics
    """
    trades = result.all_trades
    m = StrategyMetrics()

    if not trades:
        return m

    m.total_trades = len(trades)
    m.winning_trades = sum(1 for t in trades if t.pnl_pts > 0)
    m.losing_trades = sum(1 for t in trades if t.pnl_pts <= 0)
    m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0.0

    wins = [t.pnl_pts for t in trades if t.pnl_pts > 0]
    losses = [t.pnl_pts for t in trades if t.pnl_pts <= 0]

    m.avg_win_pts = float(np.mean(wins)) if wins else 0.0
    m.avg_loss_pts = float(np.mean(losses)) if losses else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    m.total_pnl_pts = result.total_pnl_pts
    m.total_pnl_dollars = result.total_pnl_dollars
    m.max_drawdown_pts = result.max_drawdown_pts

    all_pnl = [t.pnl_pts for t in trades]
    m.avg_trade_pts = float(np.mean(all_pnl))

    # Expectancy
    m.expectancy_pts = (m.win_rate * m.avg_win_pts) + (
        (1 - m.win_rate) * m.avg_loss_pts
    )

    # Consecutive wins/losses
    m.max_consecutive_wins = _max_consecutive(trades, win=True)
    m.max_consecutive_losses = _max_consecutive(trades, win=False)

    # Daily Sharpe
    if result.days:
        daily_pnl = [d.pnl_pts for d in result.days if d.num_trades > 0]
        if len(daily_pnl) > 1:
            m.sharpe_daily = float(np.mean(daily_pnl) / np.std(daily_pnl)) * np.sqrt(
                252
            )

    # By pattern type
    fb = [t for t in trades if t.pattern_type == "failed_breakdown"]
    lr = [t for t in trades if t.pattern_type == "level_reclaim"]
    m.fb_trades = len(fb)
    m.fb_win_rate = sum(1 for t in fb if t.pnl_pts > 0) / len(fb) if fb else 0.0
    m.lr_trades = len(lr)
    m.lr_win_rate = sum(1 for t in lr if t.pnl_pts > 0) / len(lr) if lr else 0.0

    # By time window (approximate from entry_time hour)
    morning = [t for t in trades if 7 <= t.entry_time.hour < 9]
    afternoon = [t for t in trades if t.entry_time.hour >= 15]
    m.morning_trades = len(morning)
    m.morning_win_rate = (
        sum(1 for t in morning if t.pnl_pts > 0) / len(morning) if morning else 0.0
    )
    m.afternoon_trades = len(afternoon)
    m.afternoon_win_rate = (
        sum(1 for t in afternoon if t.pnl_pts > 0) / len(afternoon)
        if afternoon
        else 0.0
    )

    return m


def _max_consecutive(trades: list[TradeRecord], win: bool) -> int:
    """Count max consecutive wins or losses."""
    max_run = 0
    current_run = 0
    for t in trades:
        if (win and t.pnl_pts > 0) or (not win and t.pnl_pts <= 0):
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def monte_carlo_analysis(
    trades: list[TradeRecord],
    n_simulations: int = 10000,
    n_trades: int = 100,
) -> dict:
    """Run Monte Carlo simulation on trade distribution.

    Randomly samples (with replacement) from actual trade results
    to estimate distribution of outcomes.

    Parameters
    ----------
    trades : list[TradeRecord]
    n_simulations : int
    n_trades : int
        Number of trades per simulation run.

    Returns
    -------
    dict with keys: median_pnl, p5_pnl, p95_pnl, prob_profit, max_dd_median
    """
    if not trades:
        return {"median_pnl": 0, "p5_pnl": 0, "p95_pnl": 0, "prob_profit": 0}

    pnl_array = np.array([t.pnl_pts for t in trades])
    rng = np.random.default_rng(42)

    final_pnls = []
    max_dds = []

    for _ in range(n_simulations):
        sample = rng.choice(pnl_array, size=n_trades, replace=True)
        equity = np.cumsum(sample)
        final_pnls.append(equity[-1])

        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dds.append(dd.max())

    final_pnls = np.array(final_pnls)
    max_dds = np.array(max_dds)

    return {
        "median_pnl": float(np.median(final_pnls)),
        "p5_pnl": float(np.percentile(final_pnls, 5)),
        "p95_pnl": float(np.percentile(final_pnls, 95)),
        "prob_profit": float(np.mean(final_pnls > 0)),
        "max_dd_median": float(np.median(max_dds)),
        "max_dd_p95": float(np.percentile(max_dds, 95)),
    }


def format_metrics(metrics: StrategyMetrics) -> str:
    """Format metrics as a readable string report."""
    lines = [
        "=" * 50,
        "MANCINI STRATEGY PERFORMANCE REPORT",
        "=" * 50,
        f"Total Trades:         {metrics.total_trades}",
        f"Win Rate:             {metrics.win_rate:.1%}",
        f"Profit Factor:        {metrics.profit_factor:.2f}",
        f"Total PnL:            {metrics.total_pnl_pts:+.1f} pts (${metrics.total_pnl_dollars:+,.0f})",
        f"Avg Trade:            {metrics.avg_trade_pts:+.2f} pts",
        f"Avg Win:              {metrics.avg_win_pts:+.2f} pts",
        f"Avg Loss:             {metrics.avg_loss_pts:+.2f} pts",
        f"Expectancy:           {metrics.expectancy_pts:+.2f} pts/trade",
        f"Max Drawdown:         {metrics.max_drawdown_pts:.1f} pts",
        f"Daily Sharpe (ann.):  {metrics.sharpe_daily:.2f}",
        f"Max Consec. Wins:     {metrics.max_consecutive_wins}",
        f"Max Consec. Losses:   {metrics.max_consecutive_losses}",
        "",
        "BY PATTERN:",
        f"  Failed Breakdown:   {metrics.fb_trades} trades, {metrics.fb_win_rate:.0%} WR",
        f"  Level Reclaim:      {metrics.lr_trades} trades, {metrics.lr_win_rate:.0%} WR",
        "",
        "BY TIME WINDOW:",
        f"  Morning (7:30-8:30): {metrics.morning_trades} trades, {metrics.morning_win_rate:.0%} WR",
        f"  Afternoon (3:00+):   {metrics.afternoon_trades} trades, {metrics.afternoon_win_rate:.0%} WR",
        "=" * 50,
    ]
    return "\n".join(lines)
