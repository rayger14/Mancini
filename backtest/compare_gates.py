"""Quick A/B comparison: baseline vs new data-driven gates.

Usage:
    python3 backtest/compare_gates.py [--year 2025] [--full-session]
"""
from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import BacktestRunner
from config.settings import (
    StrategyParams, RiskParams, SessionTimes, DEFAULT_STRATEGY,
)
from datetime import time as dt_time

# Base: production Optuna v2 params
_BASE = replace(DEFAULT_STRATEGY,
    acceptance_max_dip_pts=15.0,
    acceptance_min_hold_bars=11,
    fb_stop_buffer_pts=6.0,
    lr_stop_buffer_pts=4.0,
    max_fb_sweep_depth_pts=20.0,
    max_target_distance_pts=30.0,
    allow_breakdown_short=True,
    allow_backtest_short=False,
    bd_confirm_bars=21,
    bd_stop_buffer_pts=6.0,
    bd_max_break_depth_pts=17.0,
    bd_timeout_bars=35,
    min_signal_rr=0.8,
    signal_cooldown_bars=15,
    use_regime_filter=True,
)

# Wide session times = bypass time gates (same as live bot)
WIDE_SESSION = SessionTimes(
    morning_window_start=dt_time(0, 0),
    morning_window_end=dt_time(23, 59),
    afternoon_window_start=dt_time(0, 0),
    afternoon_window_end=dt_time(23, 59),
    chop_zone_start=dt_time(23, 58),
    chop_zone_end=dt_time(23, 59),
)

# Baseline: new gates DISABLED
PRODUCTION = replace(_BASE,
    max_trades_per_level=0,
    bd_short_min_rr=0.0,
    min_session_range_pts=0.0,
    cross_type_level_cooldown_bars=0,
    bd_max_entry_distance_pts=0.0,
)

# New gates ON
GATED = replace(_BASE,
    max_trades_per_level=1,
    bd_short_min_rr=1.5,
    min_session_range_pts=15.0,
    min_session_range_grace_bars=30,
    cross_type_level_cooldown_bars=30,
    bd_max_entry_distance_pts=10.0,
)

RISK = RiskParams(
    max_trades_per_day=999,
    max_daily_loss_pts=9999.0,
    max_stop_distance_pts=20.0,
    skip_tuesdays=False,
    min_rr_ratio=0.8,
)


def load_and_filter(year: int, full_session: bool) -> dict[date, pd.DataFrame]:
    data_path = Path("data/ES_1m_full_session_2021-01-01_2026-02-05.parquet")
    if not data_path.exists():
        data_path = Path("data/ES_1m_2024-02-05_2026-02-05.parquet")

    df = pd.read_parquet(data_path)
    df.index = df.index.tz_localize("US/Eastern")

    # Filter to requested year
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    df = df[start:end]

    if not full_session:
        df = df.between_time("09:30", "15:59")

    daily: dict[date, pd.DataFrame] = {}
    for dt, group in df.groupby(df.index.date):
        daily[dt] = group
    return daily


def run_config(name: str, params: StrategyParams, daily_dfs: dict) -> None:
    logger.remove()
    run_id = logger.add(sys.stderr, level="WARNING")

    runner = BacktestRunner(
        strategy_params=params,
        risk_params=RISK,
        session_times=WIDE_SESSION,
        min_rr_ratio=0.8,
    )
    result = runner.run_multi_day(daily_dfs=daily_dfs, carry_runners=True)

    logger.remove(run_id)
    logger.add(sys.stderr, level="INFO")

    # Compute stats
    trades = result.all_trades
    total = len(trades)
    wins = sum(1 for t in trades if t.pnl_pts > 0)
    wr = wins / total if total > 0 else 0
    pnl = sum(t.pnl_pts for t in trades)
    gross_w = sum(t.pnl_pts for t in trades if t.pnl_pts > 0)
    gross_l = abs(sum(t.pnl_pts for t in trades if t.pnl_pts < 0))
    pf = gross_w / gross_l if gross_l > 0 else float('inf')

    # BD Short breakdown
    bd_trades = [t for t in trades if t.pattern_type == "breakdown_short"]
    bd_total = len(bd_trades)
    bd_wins = sum(1 for t in bd_trades if t.pnl_pts > 0)
    bd_wr = bd_wins / bd_total if bd_total > 0 else 0
    bd_pnl = sum(t.pnl_pts for t in bd_trades)

    # FB breakdown
    fb_trades = [t for t in trades if t.pattern_type == "failed_breakdown"]
    fb_total = len(fb_trades)
    fb_wins = sum(1 for t in fb_trades if t.pnl_pts > 0)
    fb_wr = fb_wins / fb_total if fb_total > 0 else 0
    fb_pnl = sum(t.pnl_pts for t in fb_trades)

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  {name}")
    print(f"{sep}")
    print(f"  Total:  {total} trades, WR={wr:.1%}, PF={pf:.2f}, PnL={pnl:+.1f} pts")
    print(f"  FB:     {fb_total} trades, WR={fb_wr:.1%}, PnL={fb_pnl:+.1f} pts")
    print(f"  BD:     {bd_total} trades, WR={bd_wr:.1%}, PnL={bd_pnl:+.1f} pts")
    print(f"  Avg W:  {(gross_w/wins if wins else 0):+.1f} pts")
    print(f"  Avg L:  {(-gross_l/(total-wins) if total-wins else 0):+.1f} pts")
    print(f"{sep}")
    sys.stdout.flush()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--full-session", action="store_true")
    args = parser.parse_args()

    print(f"\nLoading {args.year} data...")
    sys.stdout.flush()
    daily = load_and_filter(args.year, args.full_session)
    print(f"Loaded {len(daily)} trading days")
    sys.stdout.flush()

    print("\n>>> Running BASELINE (no new gates)...")
    sys.stdout.flush()
    run_config("BASELINE (production params, no new gates)", PRODUCTION, daily)

    # Individual gate tests
    configs = {
        "LEVEL_REUSE only": replace(PRODUCTION, max_trades_per_level=1),
        "BD_RR only": replace(PRODUCTION, bd_short_min_rr=1.5),
        "SESSION_RANGE only": replace(PRODUCTION, min_session_range_pts=15.0, min_session_range_grace_bars=30),
        "CROSS_COOLDOWN only": replace(PRODUCTION, cross_type_level_cooldown_bars=30),
        "ENTRY_CAP only": replace(PRODUCTION, bd_max_entry_distance_pts=10.0),
        "ALL GATES": GATED,
    }
    for name, params in configs.items():
        print(f"\n>>> Running {name}...")
        sys.stdout.flush()
        run_config(name, params, daily)


if __name__ == "__main__":
    main()
