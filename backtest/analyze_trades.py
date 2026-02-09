"""Trade diagnostic analyzer — understand WHY trades win or lose.

Runs backtest with current params and prints detailed diagnostic report
showing exit reason distribution, win rate by category, and timing analysis.
"""
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from backtest.runner import BacktestRunner
from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    RiskParams, SessionTimes,
)

DATA_PATH = Path("data/ES_1m_2024-02-05_2026-02-05.parquet")
EASTERN_TZ = "US/Eastern"


def load_daily_dfs() -> dict[date, pd.DataFrame]:
    df = pd.read_parquet(DATA_PATH)
    if df.index.tz is None:
        df.index = df.index.tz_localize(EASTERN_TZ)
    df_rth = df.between_time("09:30", "15:59")
    daily = {}
    for d, grp in df_rth.groupby(df_rth.index.date):
        if len(grp) >= 10:
            daily[d] = grp
    return daily


def analyze(trades, label="DEFAULT PARAMS"):
    """Print comprehensive diagnostic report for a list of TradeRecords."""
    if not trades:
        print(f"\n{'='*60}\n{label}: NO TRADES\n{'='*60}")
        return

    print(f"\n{'='*70}")
    print(f"  TRADE DIAGNOSTIC REPORT — {label}")
    print(f"{'='*70}")

    n = len(trades)
    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts <= 0]

    print(f"\n  Total trades: {n}")
    print(f"  Wins: {len(wins)} ({len(wins)/n*100:.1f}%)")
    print(f"  Losses: {len(losses)} ({len(losses)/n*100:.1f}%)")
    total_pnl = sum(t.pnl_pts for t in trades)
    print(f"  Total PnL: {total_pnl:+.1f} pts")
    print(f"  Avg win: {np.mean([t.pnl_pts for t in wins]):+.1f} pts" if wins else "  Avg win: N/A")
    print(f"  Avg loss: {np.mean([t.pnl_pts for t in losses]):+.1f} pts" if losses else "  Avg loss: N/A")

    # --- Exit Reason Distribution ---
    print(f"\n  {'─'*50}")
    print(f"  EXIT REASON DISTRIBUTION")
    print(f"  {'─'*50}")
    reasons = defaultdict(list)
    for t in trades:
        reasons[t.exit_reason].append(t)
    for reason, group in sorted(reasons.items(), key=lambda x: -len(x[1])):
        w = sum(1 for t in group if t.pnl_pts > 0)
        avg = np.mean([t.pnl_pts for t in group])
        print(f"  {reason:<30s}  {len(group):>4d} trades  WR={w/len(group)*100:>5.1f}%  avg={avg:+.1f} pts")

    # --- Win Rate by Pattern Type ---
    print(f"\n  {'─'*50}")
    print(f"  WIN RATE BY PATTERN TYPE")
    print(f"  {'─'*50}")
    by_pattern = defaultdict(list)
    for t in trades:
        by_pattern[t.pattern_type].append(t)
    for pat, group in sorted(by_pattern.items()):
        w = sum(1 for t in group if t.pnl_pts > 0)
        avg = np.mean([t.pnl_pts for t in group])
        print(f"  {pat:<25s}  {len(group):>4d} trades  WR={w/len(group)*100:>5.1f}%  avg={avg:+.1f} pts")

    # --- Win Rate by Confirmation Type ---
    print(f"\n  {'─'*50}")
    print(f"  WIN RATE BY CONFIRMATION TYPE")
    print(f"  {'─'*50}")
    by_conf = defaultdict(list)
    for t in trades:
        key = t.confirmation_type or "unknown"
        by_conf[key].append(t)
    for conf, group in sorted(by_conf.items()):
        w = sum(1 for t in group if t.pnl_pts > 0)
        avg = np.mean([t.pnl_pts for t in group])
        print(f"  {conf:<25s}  {len(group):>4d} trades  WR={w/len(group)*100:>5.1f}%  avg={avg:+.1f} pts")

    # --- Win Rate by Level Type ---
    print(f"\n  {'─'*50}")
    print(f"  WIN RATE BY LEVEL TYPE")
    print(f"  {'─'*50}")
    by_level = defaultdict(list)
    for t in trades:
        key = t.level_type or "unknown"
        by_level[key].append(t)
    for ltype, group in sorted(by_level.items()):
        w = sum(1 for t in group if t.pnl_pts > 0)
        avg = np.mean([t.pnl_pts for t in group])
        print(f"  {ltype:<25s}  {len(group):>4d} trades  WR={w/len(group)*100:>5.1f}%  avg={avg:+.1f} pts")

    # --- Time of Day Analysis ---
    print(f"\n  {'─'*50}")
    print(f"  WIN RATE BY TIME OF DAY")
    print(f"  {'─'*50}")
    by_hour = defaultdict(list)
    for t in trades:
        h = t.entry_time.hour
        if h < 11:
            bucket = "09:30-11:00 (morning)"
        elif h < 15:
            bucket = "11:00-15:00 (chop zone)"
        else:
            bucket = "15:00-16:00 (afternoon)"
        by_hour[bucket].append(t)
    for bucket, group in sorted(by_hour.items()):
        w = sum(1 for t in group if t.pnl_pts > 0)
        avg = np.mean([t.pnl_pts for t in group])
        print(f"  {bucket:<30s}  {len(group):>4d} trades  WR={w/len(group)*100:>5.1f}%  avg={avg:+.1f} pts")

    # --- Time-to-Exit Analysis (bars) ---
    has_bars = [t for t in trades if t.exit_bar_idx > 0]
    if has_bars:
        print(f"\n  {'─'*50}")
        print(f"  TIME-TO-EXIT (bars from entry to exit)")
        print(f"  {'─'*50}")
        bars_to_exit = [t.exit_bar_idx - t.entry_bar_idx for t in has_bars]
        stopped = [t for t in has_bars if "Stop" in t.exit_reason or "stop" in t.exit_reason]
        if stopped:
            stop_bars = [t.exit_bar_idx - t.entry_bar_idx for t in stopped]
            print(f"  Stopped trades: mean={np.mean(stop_bars):.1f} bars, median={np.median(stop_bars):.0f}")
            quick_stops = sum(1 for b in stop_bars if b <= 5)
            print(f"  Quick stops (≤5 bars): {quick_stops}/{len(stopped)} ({quick_stops/len(stopped)*100:.0f}%)")
        winners_bars = [t.exit_bar_idx - t.entry_bar_idx for t in has_bars if t.pnl_pts > 0]
        if winners_bars:
            print(f"  Winning trades: mean={np.mean(winners_bars):.1f} bars, median={np.median(winners_bars):.0f}")

    # --- Sweep Depth Analysis ---
    has_depth = [t for t in trades if t.sweep_depth_pts > 0]
    if has_depth:
        print(f"\n  {'─'*50}")
        print(f"  SWEEP DEPTH vs OUTCOME")
        print(f"  {'─'*50}")
        depths = [t.sweep_depth_pts for t in has_depth]
        print(f"  Range: {min(depths):.1f} – {max(depths):.1f} pts, mean={np.mean(depths):.1f}")
        shallow = [t for t in has_depth if t.sweep_depth_pts < 5]
        medium = [t for t in has_depth if 5 <= t.sweep_depth_pts < 15]
        deep = [t for t in has_depth if t.sweep_depth_pts >= 15]
        for label_d, group in [("< 5 pts (shallow)", shallow), ("5-15 pts (medium)", medium), ("≥ 15 pts (deep)", deep)]:
            if group:
                w = sum(1 for t in group if t.pnl_pts > 0)
                avg = np.mean([t.pnl_pts for t in group])
                print(f"  {label_d:<25s}  {len(group):>4d} trades  WR={w/len(group)*100:>5.1f}%  avg={avg:+.1f} pts")

    # --- Elevator Velocity Analysis ---
    has_elev = [t for t in trades if t.elevator_peak_velocity > 0]
    if has_elev:
        print(f"\n  {'─'*50}")
        print(f"  ELEVATOR VELOCITY vs OUTCOME")
        print(f"  {'─'*50}")
        vels = [t.elevator_peak_velocity for t in has_elev]
        print(f"  Range: {min(vels):.2f} – {max(vels):.2f} pts/bar, mean={np.mean(vels):.2f}")
        slow = [t for t in has_elev if t.elevator_peak_velocity < 1.0]
        fast = [t for t in has_elev if t.elevator_peak_velocity >= 1.0]
        for label_v, group in [("< 1.0 pts/bar (slow)", slow), ("≥ 1.0 pts/bar (fast)", fast)]:
            if group:
                w = sum(1 for t in group if t.pnl_pts > 0)
                avg = np.mean([t.pnl_pts for t in group])
                print(f"  {label_v:<25s}  {len(group):>4d} trades  WR={w/len(group)*100:>5.1f}%  avg={avg:+.1f} pts")

    # --- Stop Distance Analysis ---
    has_risk = [t for t in trades if t.risk_pts > 0]
    if has_risk:
        print(f"\n  {'─'*50}")
        print(f"  STOP DISTANCE (risk_pts) vs OUTCOME")
        print(f"  {'─'*50}")
        risks = [t.risk_pts for t in has_risk]
        print(f"  Range: {min(risks):.1f} – {max(risks):.1f} pts, mean={np.mean(risks):.1f}")
        tight = [t for t in has_risk if t.risk_pts < 3]
        med = [t for t in has_risk if 3 <= t.risk_pts < 6]
        wide = [t for t in has_risk if t.risk_pts >= 6]
        for label_r, group in [("< 3 pts (tight)", tight), ("3-6 pts (medium)", med), ("≥ 6 pts (wide)", wide)]:
            if group:
                w = sum(1 for t in group if t.pnl_pts > 0)
                avg = np.mean([t.pnl_pts for t in group])
                print(f"  {label_r:<25s}  {len(group):>4d} trades  WR={w/len(group)*100:>5.1f}%  avg={avg:+.1f} pts")

    # --- R:R Distribution ---
    has_rr = [t for t in trades if t.rr_ratio_t1 > 0]
    if has_rr:
        print(f"\n  {'─'*50}")
        print(f"  R:R RATIO vs OUTCOME")
        print(f"  {'─'*50}")
        rrs = [t.rr_ratio_t1 for t in has_rr]
        print(f"  Range: {min(rrs):.2f} – {max(rrs):.2f}, mean={np.mean(rrs):.2f}")

    print(f"\n{'='*70}\n")


def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("Loading data...")
    daily_dfs = load_daily_dfs()
    print(f"Loaded {len(daily_dfs)} trading days")

    # Run with default params
    print("\nRunning backtest with DEFAULT params...")
    runner = BacktestRunner(min_rr_ratio=1.5)
    result = runner.run_multi_day(daily_dfs=daily_dfs)
    analyze(result.all_trades, "DEFAULT PARAMS")

    # Run with Optuna best params (from saved results)
    import json
    optuna_path = Path("data/optuna_results.json")
    if optuna_path.exists():
        with open(optuna_path) as f:
            optuna_data = json.load(f)
        params = optuna_data["best_params"]
        from datetime import time as dtime

        print("Running backtest with OPTUNA BEST params...")
        runner2 = BacktestRunner(
            strategy_params=StrategyParams(
                swing_low_order=params["swing_low_order"],
                multi_hour_rally_min_pts=params["multi_hour_rally_min_pts"],
                level_reclaim_min_touches=params["level_reclaim_min_touches"],
                acceptance_min_hold_bars=params["acceptance_min_hold_bars"],
                acceptance_min_hold_bars_deep=params["acceptance_min_hold_bars_deep"],
                acceptance_max_dip_pts=params["acceptance_max_dip_pts"],
                true_breakdown_abort_bars=params["true_breakdown_abort_bars"],
                fb_stop_buffer_pts=params["fb_stop_buffer"],
                lr_stop_buffer_pts=params["lr_stop_buffer"],
            ),
            elevator_params=ElevatorParams(
                min_velocity_pts_per_min=params["min_velocity"],
                min_levels_broken=params["min_levels_broken"],
                higher_low_lookback=params["higher_low_lookback"],
            ),
            exit_params=ExitParams(
                t1_exit_fraction=params["t1_exit_fraction"],
                trailing_stop_pts=params["trailing_stop_pts"],
            ),
            risk_params=RiskParams(max_trades_per_day=params["max_trades_per_day"]),
            session_times=SessionTimes(
                chop_zone_start=dtime(params["chop_start_hour"], 0),
                chop_zone_end=dtime(params["chop_end_hour"], 0),
            ),
            min_rr_ratio=params["min_rr_ratio"],
        )
        result2 = runner2.run_multi_day(daily_dfs=daily_dfs)
        analyze(result2.all_trades, "OPTUNA BEST PARAMS (full dataset)")


if __name__ == "__main__":
    main()
