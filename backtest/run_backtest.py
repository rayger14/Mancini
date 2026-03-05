"""Run Mancini Failed-Breakdown backtest on ES 1-min data.

Loads cached parquet, splits into per-day frames, runs the strategy
via BacktestRunner, then prints metrics and saves artifacts.

Usage:
    python3 backtest/run_backtest.py              # single-core (default)
    python3 backtest/run_backtest.py --parallel    # multi-core (split by year)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import BacktestRunner
from backtest.metrics import compute_metrics, format_metrics, monte_carlo_analysis
from backtest.visualizer import plot_equity_curve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/ES_1m_2024-02-05_2026-02-05.parquet")
EQUITY_CURVE_PATH = Path("data/equity_curve.png")
RESULTS_JSON_PATH = Path("data/backtest_results.json")
RTH_START = "09:30"
RTH_END = "16:00"
EASTERN_TZ = "US/Eastern"


def load_data(path: Path) -> pd.DataFrame:
    """Load parquet and localize naive timestamps to Eastern."""
    logger.info(f"Loading data from {path}")
    df = pd.read_parquet(path)
    # The data has naive timestamps already in Eastern time
    df.index = df.index.tz_localize(EASTERN_TZ)
    logger.info(f"Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
    return df


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Regular Trading Hours bars (9:30 AM - 3:59 PM Eastern)."""
    return df.between_time(RTH_START, "15:59")


def split_by_day(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Split a DataFrame into per-day DataFrames keyed by date."""
    daily: dict[date, pd.DataFrame] = {}
    for dt, group in df.groupby(df.index.date):
        daily[dt] = group
    return daily


def monthly_pnl_table(result) -> str:
    """Build a monthly PnL breakdown string from BacktestResult."""
    monthly: dict[str, float] = defaultdict(float)
    for day in result.days:
        key = day.date.strftime("%Y-%m") if hasattr(day.date, "strftime") else str(day.date)[:7]
        monthly[key] += day.pnl_pts

    lines = [
        "",
        "=" * 50,
        "MONTHLY PnL BREAKDOWN (pts)",
        "=" * 50,
        f"{'Month':<12} {'PnL (pts)':>12} {'Cum PnL':>12}",
        "-" * 38,
    ]
    cum = 0.0
    for month in sorted(monthly.keys()):
        cum += monthly[month]
        lines.append(f"{month:<12} {monthly[month]:>+12.1f} {cum:>+12.1f}")
    lines.append("=" * 50)
    return "\n".join(lines)


def trades_per_day_stats(result) -> str:
    """Compute trades-per-day distribution."""
    counts = [d.num_trades for d in result.days]
    if not counts:
        return "No trading days."
    arr = np.array(counts)
    lines = [
        "",
        "=" * 50,
        "TRADES PER DAY DISTRIBUTION",
        "=" * 50,
        f"  Mean:    {arr.mean():.2f}",
        f"  Median:  {np.median(arr):.1f}",
        f"  Std:     {arr.std():.2f}",
        f"  Min:     {arr.min()}",
        f"  Max:     {arr.max()}",
        f"  Days w/ 0 trades: {(arr == 0).sum()}",
        f"  Days w/ 1+ trade: {(arr >= 1).sum()}",
        f"  Days w/ 2+ trades: {(arr >= 2).sum()}",
        "=" * 50,
    ]
    return "\n".join(lines)


def save_results_json(metrics, mc: dict, result, path: Path) -> None:
    """Serialize key results to JSON."""
    data = {
        "total_trades": metrics.total_trades,
        "win_rate": round(metrics.win_rate, 4),
        "profit_factor": round(metrics.profit_factor, 2),
        "total_pnl_pts": round(metrics.total_pnl_pts, 1),
        "total_pnl_dollars": round(metrics.total_pnl_dollars, 0),
        "avg_trade_pts": round(metrics.avg_trade_pts, 2),
        "expectancy_pts": round(metrics.expectancy_pts, 2),
        "max_drawdown_pts": round(metrics.max_drawdown_pts, 1),
        "sharpe_daily_annualized": round(metrics.sharpe_daily, 2),
        "max_consecutive_wins": metrics.max_consecutive_wins,
        "max_consecutive_losses": metrics.max_consecutive_losses,
        "trading_days": len(result.days),
        "monte_carlo": {k: round(v, 2) for k, v in mc.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved results JSON to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mancini backtest")
    parser.add_argument(
        "--parallel", action="store_true",
        help="Split by year and run on multiple cores",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: one per year chunk)",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to parquet data file (default: RTH 2-year dataset)",
    )
    parser.add_argument(
        "--full-session", action="store_true",
        help="Use full session (no RTH filter)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load and prepare data
    # -----------------------------------------------------------------------
    data_path = Path(args.data) if args.data else DATA_PATH
    df = load_data(data_path)
    if args.full_session:
        df_filtered = df
        logger.info(f"Full session bars: {len(df_filtered):,}")
    else:
        df_filtered = filter_rth(df)
        logger.info(f"RTH bars: {len(df_filtered):,}")

    daily_dfs = split_by_day(df_filtered)
    logger.info(f"Trading days: {len(daily_dfs)}")

    # -----------------------------------------------------------------------
    # 2. Run backtest
    # -----------------------------------------------------------------------
    if args.parallel:
        from backtest.parallel import run_parallel_backtest

        logger.info("Starting PARALLEL backtest...")
        parallel_result = run_parallel_backtest(
            daily_dfs, num_workers=args.workers, min_rr_ratio=1.5,
        )

        # Wrap into a BacktestResult-compatible object for metrics
        from backtest.runner import BacktestResult, DayResult
        result = BacktestResult()
        result.all_trades = parallel_result.all_trades
        for day_date, day_pnl in parallel_result.day_pnls:
            day_trades = [t for t in parallel_result.all_trades
                          if t.entry_time.date() == day_date]
            wins = sum(1 for t in day_trades if t.pnl_pts > 0)
            wr = wins / len(day_trades) if day_trades else 0.0
            result.days.append(DayResult(
                date=day_date,
                bar_results=[],
                trade_records=day_trades,
                pnl_pts=day_pnl,
                pnl_dollars=day_pnl * 50.0,
                num_trades=len(day_trades),
                win_rate=wr,
            ))
    else:
        logger.info("Starting backtest (single-core)...")
        logger.remove()
        run_id = logger.add(sys.stderr, level="WARNING")

        runner = BacktestRunner(min_rr_ratio=1.5)
        result = runner.run_multi_day(daily_dfs=daily_dfs)

        # Restore normal logging for summary output
        logger.remove(run_id)
        logger.add(sys.stderr, level="INFO")

    # -----------------------------------------------------------------------
    # 3. Compute and display metrics
    # -----------------------------------------------------------------------
    metrics = compute_metrics(result)
    print(format_metrics(metrics))

    # Monte Carlo
    mc = monte_carlo_analysis(result.all_trades)
    mc_lines = [
        "",
        "=" * 50,
        "MONTE CARLO SIMULATION (10,000 runs x 100 trades)",
        "=" * 50,
        f"  Median PnL:      {mc['median_pnl']:+.1f} pts",
        f"  5th Percentile:  {mc['p5_pnl']:+.1f} pts",
        f"  95th Percentile: {mc['p95_pnl']:+.1f} pts",
        f"  Prob of Profit:  {mc['prob_profit']:.1%}",
        f"  Median Max DD:   {mc.get('max_dd_median', 0):+.1f} pts",
        f"  95th %ile Max DD:{mc.get('max_dd_p95', 0):+.1f} pts",
        "=" * 50,
    ]
    print("\n".join(mc_lines))

    # Monthly PnL
    print(monthly_pnl_table(result))

    # Trades per day
    print(trades_per_day_stats(result))

    # -----------------------------------------------------------------------
    # 4. Save artifacts
    # -----------------------------------------------------------------------
    try:
        plot_equity_curve(result.all_trades, save_path=str(EQUITY_CURVE_PATH))
        logger.info(f"Saved equity curve to {EQUITY_CURVE_PATH}")
    except Exception as e:
        logger.warning(f"Could not save equity curve: {e}")

    save_results_json(metrics, mc, result, RESULTS_JSON_PATH)

    logger.info("Backtest complete.")


if __name__ == "__main__":
    main()
