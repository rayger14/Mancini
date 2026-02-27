"""Phase 1: RTH-only production params on 5-year data.

Tests the validated RTH-only config (59T, 55.9% WR, PF=2.83 on 2024-2026)
against the full 2021-2026 dataset. True OOS for 2021-2024.

Usage:
    python3 backtest/five_year_rth_baseline.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date, time as dt_time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

logger.remove()

from backtest.runner import BacktestRunner, BacktestResult
from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    RiskParams, SessionTimes, DEFAULT_SESSION,
)


# ── Production RTH config (from validation_final.json) ──────────────

STRATEGY = StrategyParams(
    swing_low_order=15,
    multi_hour_rally_min_pts=22.5,
    level_reclaim_min_touches=4,
    acceptance_min_hold_bars=7,
    acceptance_min_hold_bars_deep=8,
    acceptance_max_dip_pts=3.0,
    true_breakdown_abort_bars=12,
    fb_stop_buffer_pts=5.5,
    lr_stop_buffer_pts=5.0,
    non_acceptance_min_recovery_pts=5.0,
)
ELEVATOR = ElevatorParams(
    min_velocity_pts_per_min=0.75,
    min_levels_broken=2,
    higher_low_lookback=4,
)
EXIT = ExitParams(
    default_contracts=4,
    t1_exit_fraction=1.0,
    trailing_stop_pts=7.0,
)
RISK = RiskParams(max_trades_per_day=4)
SESSION = SessionTimes(
    chop_zone_start=dt_time(12, 0),
    chop_zone_end=dt_time(15, 0),
)


def load_rth_daily_dfs() -> dict[date, pd.DataFrame]:
    """Load 5-year data, extract RTH bars, group by date, skip Mondays."""
    path = Path(__file__).parent.parent / "data" / "ES_1m_full_session_2021-01-01_2026-02-05.parquet"
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")

    # Extract RTH bars only
    rth = df.between_time("09:30", "15:59")

    # Group by date, skip Mondays
    daily: dict[date, pd.DataFrame] = {}
    for dt, group in rth.groupby(rth.index.date):
        if dt.weekday() == 0:  # skip Monday
            continue
        if len(group) >= 10:
            daily[dt] = group

    return daily


def run_and_collect(daily_dfs: dict[date, pd.DataFrame]) -> list[dict]:
    """Run backtest and collect trade records with dates."""
    runner = BacktestRunner(
        strategy_params=STRATEGY,
        elevator_params=ELEVATOR,
        exit_params=EXIT,
        risk_params=RISK,
        session_times=SESSION,
        min_rr_ratio=1.0,
    )
    result = runner.run_multi_day(daily_dfs=daily_dfs)

    trades = []
    for day_result in result.days:
        for t in day_result.trade_records:
            trades.append({
                "date": day_result.date,
                "year": day_result.date.year,
                "pattern": t.pattern_type,
                "pnl_pts": t.pnl_pts,
                "risk_pts": t.risk_pts,
                "won": t.pnl_pts > 0,
            })
    return trades


def print_stats(trades, label):
    n = len(trades)
    if n == 0:
        print(f"  {label}: 0 trades")
        return

    wins = [t for t in trades if t["won"]]
    losses = [t for t in trades if not t["won"]]
    total_pnl = sum(t["pnl_pts"] for t in trades)
    wr = len(wins) / n * 100
    gross_p = sum(t["pnl_pts"] for t in wins)
    gross_l = abs(sum(t["pnl_pts"] for t in losses))
    pf = gross_p / gross_l if gross_l > 0 else float("inf")
    avg_w = np.mean([t["pnl_pts"] for t in wins]) if wins else 0
    avg_l = np.mean([t["pnl_pts"] for t in losses]) if losses else 0

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum += t["pnl_pts"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    print(f"\n  {label}")
    print(f"  {'─'*70}")
    print(f"  Trades: {n}  |  Wins: {len(wins)}  Losses: {len(losses)}  |  "
          f"WR: {wr:.1f}%  |  PF: {pf:.2f}")
    print(f"  PnL: {total_pnl:+,.1f} pts  |  Max DD: {max_dd:.1f} pts")
    print(f"  Avg Win: {avg_w:+.1f}  |  Avg Loss: {avg_l:+.1f}")

    # Per-pattern
    for p in ["failed_breakdown", "level_reclaim"]:
        pt = [t for t in trades if t["pattern"] == p]
        if not pt:
            continue
        pw = [t for t in pt if t["won"]]
        pl = [t for t in pt if not t["won"]]
        ppnl = sum(t["pnl_pts"] for t in pt)
        pwr = len(pw) / len(pt) * 100
        pgp = sum(t["pnl_pts"] for t in pw)
        pgl = abs(sum(t["pnl_pts"] for t in pl))
        ppf = pgp / pgl if pgl > 0 else float("inf")
        short = "FB" if p == "failed_breakdown" else "LR"
        print(f"    {short}: {len(pt)}T, {pwr:.1f}% WR, PF={ppf:.2f}, {ppnl:+.1f} pts")


def main():
    print("Loading 5-year RTH data...")
    daily_dfs = load_rth_daily_dfs()
    print(f"Loaded {len(daily_dfs)} trading days (Mon skipped)")
    print(f"Date range: {min(daily_dfs.keys())} to {max(daily_dfs.keys())}")

    print("\nRunning backtest with RTH production params...")
    trades = run_and_collect(daily_dfs)

    # ── Full period ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FULL PERIOD: 5 Years RTH-Only")
    print(f"{'='*80}")
    print_stats(trades, "ALL (Jan 2021 - Feb 2026)")

    # ── Yearly breakdown ────────────────────────────────────────────
    print(f"\n  {'Year':<6} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PF':>7} {'PnL':>10} {'MaxDD':>8}")
    print("  " + "-" * 55)

    for year in sorted(set(t["year"] for t in trades)):
        yt = [t for t in trades if t["year"] == year]
        yw = [t for t in yt if t["won"]]
        yl = [t for t in yt if not t["won"]]
        ypnl = sum(t["pnl_pts"] for t in yt)
        ywr = len(yw) / len(yt) * 100 if yt else 0
        ygp = sum(t["pnl_pts"] for t in yw)
        ygl = abs(sum(t["pnl_pts"] for t in yl))
        ypf = ygp / ygl if ygl > 0 else float("inf")

        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in yt:
            cum += t["pnl_pts"]
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        print(f"  {year:<6} {len(yt):>7} {len(yw):>5} {ywr:>5.1f}% {ypf:>7.2f} "
              f"{ypnl:>+10.1f} {max_dd:>8.1f}")

    # ── OOS vs IS ───────────────────────────────────────────────────
    oos = [t for t in trades if t["year"] < 2024 or (t["year"] == 2024 and t["date"].month < 2)]
    is_trades = [t for t in trades if t not in oos]

    print(f"\n{'='*80}")
    print("TRUE OOS: Jan 2021 - Jan 2024 (params NEVER tuned on this data)")
    print(f"{'='*80}")
    print_stats(oos, "OOS (2021-2024)")

    print(f"\n{'='*80}")
    print("IN-SAMPLE: Feb 2024 - Feb 2026 (params tuned on this period)")
    print(f"{'='*80}")
    print_stats(is_trades, "IS (2024-2026)")

    # ── Monthly breakdown ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print("MONTHLY PNL")
    print(f"{'='*80}")
    monthly = {}
    for t in trades:
        key = (t["date"].year, t["date"].month)
        if key not in monthly:
            monthly[key] = []
        monthly[key].append(t)

    cum_pnl = 0.0
    print(f"  {'Month':<10} {'Trades':>7} {'WR%':>6} {'PnL':>10} {'CumPnL':>10}")
    print("  " + "-" * 45)
    for (year, month) in sorted(monthly.keys()):
        mt = monthly[(year, month)]
        mw = [t for t in mt if t["won"]]
        mpnl = sum(t["pnl_pts"] for t in mt)
        cum_pnl += mpnl
        mwr = len(mw) / len(mt) * 100 if mt else 0
        print(f"  {year}-{month:02d}   {len(mt):>7} {mwr:>5.1f}% {mpnl:>+10.1f} {cum_pnl:>+10.1f}")

    # ── Dollar amounts ──────────────────────────────────────────────
    total_pts = sum(t["pnl_pts"] for t in trades)
    oos_pts = sum(t["pnl_pts"] for t in oos)
    is_pts = sum(t["pnl_pts"] for t in is_trades)
    print(f"\n  DOLLAR PnL (1 MES @ $5/pt):")
    print(f"    Full: {total_pts:+,.1f} pts = ${total_pts * 5:+,.0f}")
    print(f"    OOS:  {oos_pts:+,.1f} pts = ${oos_pts * 5:+,.0f}")
    print(f"    IS:   {is_pts:+,.1f} pts = ${is_pts * 5:+,.0f}")


if __name__ == "__main__":
    main()
