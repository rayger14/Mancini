"""Validate live data hypotheses against 5-year historical backtest.

Runs the full backtest, then slices every trade by the emergent patterns
discovered in live data collection (data/trade_lessons.md) to check
whether they hold historically or are overfitting to 27 trades.

Usage:
    python3 backtest/validate_hypotheses.py
    python3 backtest/validate_hypotheses.py --data data/ES_1m_full_session_2021-01-01_2026-02-05.parquet --full-session
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import BacktestRunner, BacktestResult
from config.settings import StrategyParams, DEFAULT_STRATEGY

DATA_PATH = Path("data/ES_1m_2024-02-05_2026-02-05.parquet")
EASTERN_TZ = "US/Eastern"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize(EASTERN_TZ)
    return df


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def split_by_day(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    daily: dict[date, pd.DataFrame] = {}
    for dt, group in df.groupby(df.index.date):
        daily[dt] = group
    return daily


def run_backtest(daily_dfs: dict) -> BacktestResult:
    """Run backtest with production Optuna v2 params."""
    logger.remove()
    run_id = logger.add(sys.stderr, level="WARNING")

    # Use production params with BD Short enabled
    from config.settings import ElevatorParams
    strategy = StrategyParams(
        acceptance_max_dip_pts=15.0,
        acceptance_min_hold_bars=11,
        fb_stop_buffer_pts=6.0,
        lr_stop_buffer_pts=4.0,
        max_target_distance_pts=30.0,
        max_fb_sweep_depth_pts=999.0,
        bd_confirm_bars=21,
        bd_stop_buffer_pts=6.0,
        bd_max_break_depth_pts=17.0,
        bd_timeout_bars=35,
        allow_breakdown_short=True,
        allow_backtest_short=False,
        signal_cooldown_bars=15,
        use_regime_filter=False,
    )
    runner = BacktestRunner(strategy_params=strategy, min_rr_ratio=0.8)
    result = runner.run_multi_day(daily_dfs=daily_dfs, carry_runners=True)

    logger.remove(run_id)
    logger.add(sys.stderr, level="INFO")
    return result


# ---------------------------------------------------------------------------
# Hypothesis validators
# ---------------------------------------------------------------------------

def _bucket(value: float, edges: list[float]) -> str:
    """Place value into a bucket label like '5-10'."""
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            return f"{edges[i]}-{edges[i+1]}"
    return f"{edges[-1]}+"


def _stats(trades: list, label: str = "") -> dict:
    """Compute summary stats for a list of TradeRecord."""
    if not trades:
        return {"label": label, "n": 0, "wr": 0, "avg_pnl": 0, "total_pnl": 0,
                "avg_win": 0, "avg_loss": 0}
    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts <= 0]
    total = sum(t.pnl_pts for t in trades)
    return {
        "label": label,
        "n": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "avg_pnl": total / len(trades),
        "total_pnl": total,
        "avg_win": sum(t.pnl_pts for t in wins) / len(wins) if wins else 0,
        "avg_loss": sum(t.pnl_pts for t in losses) / len(losses) if losses else 0,
    }


def print_table(title: str, rows: list[dict], live_finding: str = ""):
    """Pretty-print a hypothesis validation table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    if live_finding:
        print(f"  Live finding: {live_finding}")
    print(f"{'='*80}")
    print(f"  {'Bucket':<25} {'Trades':>7} {'WR%':>7} {'Avg PnL':>9} {'Total PnL':>10} {'Avg Win':>9} {'Avg Loss':>9}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*9} {'-'*10} {'-'*9} {'-'*9}")
    for r in rows:
        if r["n"] == 0:
            continue
        print(f"  {r['label']:<25} {r['n']:>7} {r['wr']:>6.1f}% {r['avg_pnl']:>+9.2f} {r['total_pnl']:>+10.1f} {r['avg_win']:>+9.2f} {r['avg_loss']:>+9.2f}")


def h1_level_type(trades):
    """H1: PRIOR_DAY_LOW is best level type; MULTI_HOUR_LOW is toxic."""
    by_type = defaultdict(list)
    for t in trades:
        lt = t.level_type or "UNKNOWN"
        by_type[lt].append(t)

    rows = []
    for lt in sorted(by_type.keys(), key=lambda k: -sum(t.pnl_pts for t in by_type[k])):
        rows.append(_stats(by_type[lt], lt))

    print_table(
        "H1: Level Type Performance",
        rows,
        "PRIOR_DAY_LOW: 60% WR, +20 pts | MULTI_HOUR_LOW: 20% WR, -102 pts (27 live trades)"
    )
    return rows


def h2_stop_distance(trades):
    """H2: Hard stop cap at 15 pts. Wider stops = catastrophic losses."""
    edges = [0, 5, 10, 15, 20, 30]
    buckets = defaultdict(list)
    for t in trades:
        risk = t.risk_pts if t.risk_pts > 0 else abs(t.entry_price - t.stop_price)
        b = _bucket(risk, edges)
        buckets[b].append(t)

    rows = [_stats(buckets.get(f"{edges[i]}-{edges[i+1]}", []), f"{edges[i]}-{edges[i+1]} pts")
            for i in range(len(edges) - 1)]
    rows.append(_stats(buckets.get(f"{edges[-1]}+", []), f"{edges[-1]}+ pts"))

    print_table(
        "H2: Stop Distance Impact",
        rows,
        "10-15 pts = killing zone (17% WR); >15 pts catastrophic. Cap at 15 pts."
    )
    return rows


def h3_rr_buckets(trades):
    """H3: R:R 0.5-1.0 paradoxically optimal in live data."""
    edges = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
    buckets = defaultdict(list)
    for t in trades:
        rr = t.rr_ratio_t1
        b = _bucket(rr, edges)
        buckets[b].append(t)

    rows = [_stats(buckets.get(f"{edges[i]}-{edges[i+1]}", []), f"R:R {edges[i]}-{edges[i+1]}")
            for i in range(len(edges) - 1)]
    rows.append(_stats(buckets.get(f"{edges[-1]}+", []), f"R:R {edges[-1]}+"))

    print_table(
        "H3: Risk-Reward Ratio Buckets",
        rows,
        "Live: R:R 0.5-1.0 = 100% WR (5/5); R:R 1.0-1.5 = 14% WR (1/7)"
    )
    return rows


def h4_sequential_same_level(trades):
    """H4: Second trade at same level in same session loses."""
    # Group trades by (session_date, level_price)
    session_level = defaultdict(list)
    for t in trades:
        day = t.entry_time.date() if hasattr(t.entry_time, 'date') else t.entry_date
        key = (day, round(t.level_price, 0))  # round to nearest point
        session_level[key].append(t)

    first_trades = []
    second_plus_trades = []
    for key, group in session_level.items():
        if len(group) == 0:
            continue
        # Sort by entry time
        group.sort(key=lambda t: t.entry_time)
        first_trades.append(group[0])
        second_plus_trades.extend(group[1:])

    rows = [
        _stats(first_trades, "1st trade at level"),
        _stats(second_plus_trades, "2nd+ trade at level"),
    ]

    print_table(
        "H4: Sequential Trades at Same Level (Same Session)",
        rows,
        "Live: 1st = 50% WR; 2nd = 0% WR (0/4). One trade per level rule."
    )
    return rows


def h5_pattern_type(trades):
    """H5: Performance by pattern type (FB, LR, BD Short)."""
    by_type = defaultdict(list)
    for t in trades:
        by_type[t.pattern_type].append(t)

    rows = [_stats(by_type[pt], pt) for pt in sorted(by_type.keys(), key=lambda k: -sum(t.pnl_pts for t in by_type[k]))]

    print_table("H5: Pattern Type Performance", rows,
                "Live: FB Long best; BD Short volatile; LR inconsistent")
    return rows


def h6_time_of_day(trades):
    """H6: Time-of-day edge. Evening trades lose, afternoon FB best."""
    buckets = defaultdict(list)
    for t in trades:
        hour = t.entry_time.hour if hasattr(t.entry_time, 'hour') else 12
        if 18 <= hour <= 23:
            b = "Evening (6-11 PM ET)"
        elif 0 <= hour < 4:
            b = "Late Night (12-4 AM ET)"
        elif 4 <= hour < 9:
            b = "Pre-Market (4-9 AM ET)"
        elif 9 <= hour < 12:
            b = "Morning (9 AM-12 PM ET)"
        elif 12 <= hour < 14:
            b = "Midday (12-2 PM ET)"
        else:
            b = "Afternoon (2-4 PM ET)"
        buckets[b].append(t)

    order = ["Evening (6-11 PM ET)", "Late Night (12-4 AM ET)", "Pre-Market (4-9 AM ET)",
             "Morning (9 AM-12 PM ET)", "Midday (12-2 PM ET)", "Afternoon (2-4 PM ET)"]
    rows = [_stats(buckets.get(b, []), b) for b in order]

    print_table("H6: Time-of-Day Edge", rows,
                "Live: Evening -140 pts (13T); Afternoon +23 pts (6T, 83% WR)")
    return rows


def h7_bd_short_rr_threshold(trades):
    """H7: BD Short requires R:R > 1.5 to be profitable."""
    bd_shorts = [t for t in trades if "breakdown" in t.pattern_type.lower() and t.direction == "short"]
    edges = [0, 1.0, 1.5, 2.0, 3.0]
    buckets = defaultdict(list)
    for t in bd_shorts:
        b = _bucket(t.rr_ratio_t1, edges)
        buckets[b].append(t)

    rows = [_stats(buckets.get(f"{edges[i]}-{edges[i+1]}", []), f"BD Short R:R {edges[i]}-{edges[i+1]}")
            for i in range(len(edges) - 1)]
    rows.append(_stats(buckets.get(f"{edges[-1]}+", []), f"BD Short R:R {edges[-1]}+"))

    print_table("H7: BD Short R:R Threshold", rows,
                "Live: R:R 1.0-1.5 = 14% WR; R:R 1.5+ = 50% WR")
    return rows


def h8_fb_by_level_type(trades):
    """H8: FB Long works only at PRIOR_DAY_LOW."""
    fb_longs = [t for t in trades if "failed_breakdown" in t.pattern_type.lower() and t.direction == "long"]
    by_type = defaultdict(list)
    for t in fb_longs:
        lt = t.level_type or "UNKNOWN"
        by_type[lt].append(t)

    rows = [_stats(by_type[lt], f"FB @ {lt}")
            for lt in sorted(by_type.keys(), key=lambda k: -sum(t.pnl_pts for t in by_type[k]))]

    print_table("H8: FB Long by Level Type", rows,
                "Live: FB @ PRIOR_DAY_LOW = profitable; FB @ MULTI_HOUR_LOW = toxic")
    return rows


def h9_sweep_depth(trades):
    """H9: Deeper sweeps → bigger bounces (Mancini thesis)."""
    fb_trades = [t for t in trades if "failed_breakdown" in t.pattern_type.lower()]
    edges = [0, 2, 5, 10, 20, 50]
    buckets = defaultdict(list)
    for t in fb_trades:
        sd = t.sweep_depth_pts if t.sweep_depth_pts > 0 else 0
        b = _bucket(sd, edges)
        buckets[b].append(t)

    rows = [_stats(buckets.get(f"{edges[i]}-{edges[i+1]}", []), f"Sweep {edges[i]}-{edges[i+1]} pts")
            for i in range(len(edges) - 1)]
    rows.append(_stats(buckets.get(f"{edges[-1]}+", []), f"Sweep {edges[-1]}+ pts"))

    print_table("H9: FB Sweep Depth vs Outcome", rows,
                "Live data: 50+ pt sweep = 117 avg recovery. Bigger sell = bigger squeeze.")
    return rows


def h10_confirmation_type(trades):
    """H10: Acceptance vs non-acceptance confirmation."""
    by_conf = defaultdict(list)
    for t in trades:
        c = t.confirmation_type or "unknown"
        by_conf[c].append(t)

    rows = [_stats(by_conf[c], c) for c in sorted(by_conf.keys())]

    print_table("H10: Confirmation Type", rows,
                "Which confirmation method produces better trades?")
    return rows


def h11_direction(trades):
    """H11: Long vs Short overall performance."""
    by_dir = defaultdict(list)
    for t in trades:
        by_dir[t.direction].append(t)

    rows = [_stats(by_dir[d], d) for d in ["long", "short"]]

    print_table("H11: Direction Performance", rows,
                "Live: Longs +69 pts, Shorts -183 pts overall")
    return rows


def h12_pattern_x_level(trades):
    """H12: Pattern + Level Type cross-tabulation (top combos)."""
    combos = defaultdict(list)
    for t in trades:
        key = f"{t.pattern_type} @ {t.level_type or 'UNKNOWN'}"
        combos[key].append(t)

    rows = [_stats(combos[k], k)
            for k in sorted(combos.keys(), key=lambda k: -sum(t.pnl_pts for t in combos[k]))]

    # Only show combos with 5+ trades
    rows = [r for r in rows if r["n"] >= 5]

    print_table("H12: Pattern x Level Type (5+ trades)", rows,
                "Which pattern+level combos have real edge?")
    return rows


def h13_runner_performance(trades):
    """H13: Runner trades (multi-day holds) vs quick exits."""
    runners = [t for t in trades if t.is_runner and t.days_held > 1]
    non_runners = [t for t in trades if not t.is_runner or t.days_held <= 1]

    rows = [
        _stats(runners, f"Runners ({len(runners)}T, avg {np.mean([t.days_held for t in runners]):.1f}d)" if runners else "Runners"),
        _stats(non_runners, "Quick exits (≤1 day)"),
    ]

    print_table("H13: Runner vs Quick Exit Performance", rows,
                "Do multi-day runners add edge?")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate live data hypotheses on historical backtest")
    parser.add_argument("--data", type=str, default=None, help="Path to parquet data file")
    parser.add_argument("--full-session", action="store_true", help="Use full session (no RTH filter)")
    args = parser.parse_args()

    # Load data
    data_path = Path(args.data) if args.data else DATA_PATH
    logger.info(f"Loading {data_path}...")
    df = load_data(data_path)

    if not args.full_session:
        df = filter_rth(df)
        logger.info(f"RTH bars: {len(df):,}")
    else:
        logger.info(f"Full session bars: {len(df):,}")

    daily_dfs = split_by_day(df)
    logger.info(f"Trading days: {len(daily_dfs)}")

    # Run backtest
    logger.info("Running backtest with production params (min_rr=0.8, carry_runners=True)...")
    result = run_backtest(daily_dfs)

    trades = result.all_trades
    logger.info(f"Backtest complete: {len(trades)} trades, {result.total_pnl_pts:+.1f} pts")

    # Header
    print(f"\n{'#'*80}")
    print(f"#  HYPOTHESIS VALIDATION: Live Data Findings vs {len(trades)} Historical Trades")
    print(f"#  Data: {data_path.name}")
    print(f"#  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"#  Total PnL: {result.total_pnl_pts:+.1f} pts | WR: {result.win_rate:.1%} | PF: {result.profit_factor:.2f}")
    print(f"{'#'*80}")

    # Run all hypothesis validators
    all_results = {}
    all_results["h1_level_type"] = h1_level_type(trades)
    all_results["h2_stop_distance"] = h2_stop_distance(trades)
    all_results["h3_rr_buckets"] = h3_rr_buckets(trades)
    all_results["h4_sequential_level"] = h4_sequential_same_level(trades)
    all_results["h5_pattern_type"] = h5_pattern_type(trades)
    all_results["h6_time_of_day"] = h6_time_of_day(trades)
    all_results["h7_bd_short_rr"] = h7_bd_short_rr_threshold(trades)
    all_results["h8_fb_by_level"] = h8_fb_by_level_type(trades)
    all_results["h9_sweep_depth"] = h9_sweep_depth(trades)
    all_results["h10_confirmation"] = h10_confirmation_type(trades)
    all_results["h11_direction"] = h11_direction(trades)
    all_results["h12_pattern_x_level"] = h12_pattern_x_level(trades)
    all_results["h13_runners"] = h13_runner_performance(trades)

    # Summary verdict
    print(f"\n{'#'*80}")
    print(f"#  VERDICT SUMMARY")
    print(f"{'#'*80}")
    print("""
    For each hypothesis, compare the live finding (27 trades) vs historical (N trades).
    If the pattern holds on 500+ historical trades, it's likely REAL EDGE.
    If it reverses or flattens, it's likely OVERFITTING to the small live sample.

    Key questions answered:
    - Is MULTI_HOUR_LOW really toxic, or was that just 5 unlucky trades?
    - Does the 15-pt stop cap hold up on 5 years of data?
    - Is the R:R 0.5-1.0 "paradox" real or sampling noise?
    - Does "one trade per level" matter historically?
    - Are BD Shorts only profitable at R:R > 1.5?
    """)

    # Save results as JSON for further analysis
    output_path = Path("data/hypothesis_validation.json")
    serializable = {}
    for key, rows in all_results.items():
        serializable[key] = [{k: round(v, 4) if isinstance(v, float) else v
                              for k, v in r.items()} for r in rows]
    output_path.write_text(json.dumps(serializable, indent=2))
    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
