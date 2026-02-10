"""Reporting: text summaries, statistical tests, histograms."""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from backtest.cpcv.robustness import RobustnessResult
from backtest.cpcv.optimizer import OptimizationResult


def robustness_report(result: RobustnessResult) -> str:
    """Generate a text report from the robustness test."""
    df = result.metrics_df
    if df.empty:
        return "No results to report."

    lines = [
        "=" * 60,
        "CPCV ROBUSTNESS REPORT",
        "=" * 60,
        f"Paths: {len(df)}",
        "",
    ]

    for col in [
        "sharpe_daily", "win_rate", "profit_factor",
        "total_pnl_pts", "expectancy_pts", "max_drawdown_pts",
    ]:
        if col not in df.columns:
            continue
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        lines.append(
            f"{col:25s}: "
            f"mean={vals.mean():.3f}  "
            f"std={vals.std():.3f}  "
            f"median={vals.median():.3f}  "
            f"p5={vals.quantile(0.05):.3f}  "
            f"p95={vals.quantile(0.95):.3f}"
        )

    pct_profitable = (df["total_pnl_pts"] > 0).mean()
    sharpes = df["sharpe_daily"].replace([np.inf, -np.inf], np.nan).dropna()
    pct_sharpe_pos = (sharpes > 0).mean() if len(sharpes) > 0 else 0

    lines.extend([
        "",
        "CONSISTENCY:",
        f"  % paths with PnL > 0:    {pct_profitable:.1%}",
        f"  % paths with Sharpe > 0: {pct_sharpe_pos:.1%}",
    ])

    if len(sharpes) > 1:
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(sharpes, 0)
        p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        lines.extend([
            "",
            "STATISTICAL SIGNIFICANCE (H0: mean Sharpe = 0):",
            f"  t-stat: {t_stat:.3f}",
            f"  p-value (one-sided): {p_one_sided:.4f}",
            f"  Significant at 5%: {'YES' if p_one_sided < 0.05 else 'NO'}",
        ])

    lines.append("=" * 60)
    return "\n".join(lines)


def optimization_report(result: OptimizationResult) -> str:
    """Generate a text report from the optimization."""
    df = result.results_df
    if df.empty:
        return "No optimization results."

    pbo = result.overfitting_probability()
    best = result.best_params_by_rank()

    lines = [
        "=" * 60,
        "CPCV OPTIMIZATION REPORT",
        "=" * 60,
        "",
        f"Probability of Backtest Overfitting (PBO): {pbo:.1%}",
        f"  Interpretation: "
        f"{'LOW RISK' if pbo < 0.25 else 'MODERATE' if pbo < 0.5 else 'HIGH RISK'}",
        "",
        "BEST PARAMETERS (by average rank across test folds):",
    ]
    for k, v in best.to_dict().items():
        lines.append(f"  {k}: {v}")

    lines.extend([
        "",
        f"{'Metric':20s} {'Train Mean':>12s} {'Test Mean':>12s} {'Degradation':>12s}",
    ])
    for metric in ["sharpe", "pnl", "pf", "wr"]:
        tc, oc = f"train_{metric}", f"test_{metric}"
        if tc in df.columns and oc in df.columns:
            tm = df[tc].replace([np.inf, -np.inf], np.nan).mean()
            om = df[oc].replace([np.inf, -np.inf], np.nan).mean()
            deg = (tm - om) / abs(tm) * 100 if tm != 0 else 0
            lines.append(f"  {metric:18s} {tm:>12.3f} {om:>12.3f} {deg:>11.1f}%")

    lines.append("=" * 60)
    return "\n".join(lines)


def plot_robustness_histograms(
    result: RobustnessResult,
    save_path: Optional[str] = None,
) -> None:
    """Plot 2x2 histograms of key metrics across CPCV paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib required for plots")
        return

    df = result.metrics_df
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CPCV Robustness: Metric Distributions Across Paths", fontsize=14)

    metrics = [
        ("sharpe_daily", "Daily Sharpe (Annualized)", axes[0, 0]),
        ("win_rate", "Win Rate", axes[0, 1]),
        ("profit_factor", "Profit Factor", axes[1, 0]),
        ("total_pnl_pts", "Total PnL (pts)", axes[1, 1]),
    ]

    for col, title, ax in metrics:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            ax.set_title(f"{title} (no data)")
            continue
        ax.hist(vals, bins=min(15, len(vals)), edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(vals.mean(), color="red", linestyle="--", label=f"Mean={vals.mean():.2f}")
        ax.axvline(vals.median(), color="orange", linestyle="-.", label=f"Med={vals.median():.2f}")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    else:
        fig.savefig("data/cpcv_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
