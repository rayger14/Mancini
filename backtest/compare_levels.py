#!/usr/bin/env python3
"""
Compare backtest results across three level modes:
  A) Engine-only — engine detects its own levels from price action
  B) Mancini-only — only Mancini's newsletter levels (no engine detection)
  C) Hybrid — engine detects levels + Mancini levels injected as extra context

The hybrid tests whether giving the engine Mancini's pre-market levels
(for targets, supports, R:R) while keeping its own real-time detection
improves trade outcomes.
"""

import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Suppress debug/info logs from strategy during backtest
logger.remove()
logger.add(sys.stderr, level="WARNING")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import DayResult, BacktestResult
from config.levels import Level, LevelType
from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    DEFAULT_STRATEGY, DEFAULT_ELEVATOR, DEFAULT_EXIT, DEFAULT_RISK,
)
from strategy.mancini_long import ManciniLongStrategy, enrich_dataframe


# ── Parse Mancini levels ──────────────────────────────────────


def parse_mancini_levels(text: str) -> dict:
    result = {"supports": [], "resistances": []}
    sup_match = re.search(
        r'[Ss]upports?\s+(?:are|is)[:\s]*(.+?)(?:\n\n|In terms|As readers|As usual|$)',
        text, re.DOTALL)
    if sup_match:
        result["supports"] = _parse_level_list(sup_match.group(1))
    res_match = re.search(
        r'[Rr]esistances?\s+(?:are|is)[:\s]*(.+?)(?:\n\n|In terms|As readers|As usual|$)',
        text, re.DOTALL)
    if res_match:
        result["resistances"] = _parse_level_list(res_match.group(1))
    return result


def _parse_level_list(text: str) -> list:
    levels = []
    for m in re.finditer(r'(\d{4,5})(?:\s*[-–]\s*(\d{4,5}))?\s*(\(major\))?', text):
        price1 = float(m.group(1))
        price2 = float(m.group(2)) if m.group(2) else None
        is_major = bool(m.group(3))
        price = (price1 + price2) / 2 if price2 else price1
        if 3000 < price < 9000:
            levels.append({"price": price, "is_major": is_major})
    return levels


def get_newsletter_date(post: dict) -> date:
    pub_date = pd.Timestamp(post["date"][:10])
    next_day = pub_date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)
    return next_day.date()


def inject_mancini_levels(strategy: ManciniLongStrategy, levels: dict, session_time: datetime,
                          supports=True, resistances=True):
    """Inject Mancini's newsletter levels into the strategy's level store."""
    store = strategy.signal_aggregator.level_store
    if supports:
        for sup in levels["supports"]:
            store.add(Level(
                price=sup["price"],
                level_type=LevelType.MULTI_HOUR_LOW if sup["is_major"] else LevelType.HORIZONTAL_SR,
                created_at=session_time,
                confirmed_at=session_time,
                touch_count=3 if sup["is_major"] else 1,
            ))
    if resistances:
        for res in levels["resistances"]:
            store.add(Level(
                price=res["price"],
                level_type=LevelType.HORIZONTAL_SR,
                created_at=session_time,
                confirmed_at=session_time,
                touch_count=3 if res["is_major"] else 1,
            ))


def run_strategy_with_levels(
    strategy: ManciniLongStrategy,
    df: pd.DataFrame,
    prior_day_df: Optional[pd.DataFrame],
    mancini_levels: Optional[dict],
    mode: str,  # "engine", "mancini", "hybrid", "targets_only"
) -> DayResult:
    """Run strategy for one day with specified level mode.

    engine:       normal — engine detects all levels
    mancini:      clear engine levels, inject only Mancini's
    hybrid:       engine levels + ALL Mancini levels on top
    targets_only: engine levels + Mancini RESISTANCES only (better targets)
    """
    day = df.index[0].date()
    session_dt = df.index[0].to_pydatetime()

    strategy.reset()
    strategy.position_manager.start_session(session_dt)

    if mode == "engine":
        strategy.signal_aggregator.initialize_levels(df, prior_day_df)

    elif mode == "mancini":
        strategy.signal_aggregator.initialize_levels(df, prior_day_df)
        strategy.signal_aggregator.level_store.clear()
        if mancini_levels:
            inject_mancini_levels(strategy, mancini_levels, session_dt)

    elif mode == "hybrid":
        strategy.signal_aggregator.initialize_levels(df, prior_day_df)
        if mancini_levels:
            inject_mancini_levels(strategy, mancini_levels, session_dt)

    elif mode == "targets_only":
        # Engine detects its own supports/entries, but Mancini resistances
        # give better T1/T2 targets
        strategy.signal_aggregator.initialize_levels(df, prior_day_df)
        if mancini_levels:
            inject_mancini_levels(strategy, mancini_levels, session_dt,
                                  supports=False, resistances=True)

    # Run bar-by-bar
    enriched = enrich_dataframe(df)
    velocity = enriched["velocity_5"]
    timestamps = df.index.to_pydatetime()
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values
    vels = velocity.values

    bar_results = []
    for i in range(len(df)):
        vel = float(vels[i])
        if vel != vel:
            vel = 0.0
        br = strategy._process_bar(
            bar_idx=i,
            timestamp=timestamps[i],
            open_=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
            volume=float(volumes[i]),
            velocity=vel,
            df=df,
        )
        bar_results.append(br)

    records = strategy.trade_records
    pnl = strategy.total_pnl_pts
    pnl_d = strategy.total_pnl_dollars
    wins = sum(1 for r in records if r.pnl_pts > 0)
    wr = wins / len(records) if records else 0.0

    return DayResult(
        date=day,
        bar_results=bar_results,
        trade_records=list(records),
        pnl_pts=pnl,
        pnl_dollars=pnl_d,
        num_trades=len(records),
        win_rate=wr,
    )


# ── Main comparison ───────────────────────────────────────────


def run_three_way_comparison(
    daily_dfs: dict,
    posts_by_date: dict,
    strategy_params=DEFAULT_STRATEGY,
    elevator_params=DEFAULT_ELEVATOR,
    exit_params=DEFAULT_EXIT,
    risk_params=DEFAULT_RISK,
    min_rr_ratio: float = 1.0,
):
    dates = sorted(daily_dfs.keys())
    overlapping = [d for d in dates if d in posts_by_date]

    print(f"Total trading days: {len(dates)}")
    print(f"Days with Mancini newsletter: {len(overlapping)}")
    print()

    # Create 3 independent strategy instances
    def make_strategy():
        return ManciniLongStrategy(
            strategy_params=strategy_params,
            elevator_params=elevator_params,
            exit_params=exit_params,
            risk_params=risk_params,
            min_rr_ratio=min_rr_ratio,
        )

    engine_strat = make_strategy()
    mancini_strat = make_strategy()
    hybrid_strat = make_strategy()
    targets_strat = make_strategy()

    engine_res = BacktestResult()
    mancini_res = BacktestResult()
    hybrid_res = BacktestResult()
    targets_res = BacktestResult()
    rows = []

    prior_day_df = None

    for i, day in enumerate(dates):
        df = daily_dfs[day]

        if day in posts_by_date:
            ml = posts_by_date[day]

            e_day = run_strategy_with_levels(engine_strat, df, prior_day_df, ml, "engine")
            m_day = run_strategy_with_levels(mancini_strat, df, prior_day_df, ml, "mancini")
            h_day = run_strategy_with_levels(hybrid_strat, df, prior_day_df, ml, "hybrid")
            t_day = run_strategy_with_levels(targets_strat, df, prior_day_df, ml, "targets_only")

            engine_res.days.append(e_day)
            engine_res.all_trades.extend(e_day.trade_records)
            mancini_res.days.append(m_day)
            mancini_res.all_trades.extend(m_day.trade_records)
            hybrid_res.days.append(h_day)
            hybrid_res.all_trades.extend(h_day.trade_records)
            targets_res.days.append(t_day)
            targets_res.all_trades.extend(t_day.trade_records)

            # Compare entries
            e_entries = {round(t.entry_price, 2) for t in e_day.trade_records}
            t_entries = {round(t.entry_price, 2) for t in t_day.trade_records}

            te_exact = len(t_entries & e_entries)
            te_close = sum(1 for tp in t_entries
                           if any(abs(tp - ep) <= 2.0 for ep in e_entries))

            rows.append({
                "date": day,
                "engine_trades": e_day.num_trades,
                "mancini_trades": m_day.num_trades,
                "hybrid_trades": h_day.num_trades,
                "targets_trades": t_day.num_trades,
                "engine_pnl": e_day.pnl_pts,
                "mancini_pnl": m_day.pnl_pts,
                "hybrid_pnl": h_day.pnl_pts,
                "targets_pnl": t_day.pnl_pts,
                "engine_wr": e_day.win_rate,
                "targets_wr": t_day.win_rate,
                "targets_engine_exact": te_exact,
                "targets_engine_close": te_close,
            })

        prior_day_df = df

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(dates)} days...")

    return pd.DataFrame(rows), engine_res, mancini_res, hybrid_res, targets_res


def print_report(df, engine, mancini, hybrid, targets):
    print("\n" + "=" * 100)
    print("FOUR-WAY COMPARISON: ENGINE vs MANCINI vs HYBRID vs TARGETS-ONLY")
    print("=" * 100)

    print(f"\nDays compared: {len(df)}")

    print("\n── OVERALL PERFORMANCE ──")
    print(f"  {'Metric':<25} {'Engine':>12} {'Targets':>12} {'Hybrid':>12} {'Mancini':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for label, e, t, h, m in [
        ("Total trades", engine.total_trades, targets.total_trades, hybrid.total_trades, mancini.total_trades),
        ("Win rate", f"{engine.win_rate:.1%}", f"{targets.win_rate:.1%}", f"{hybrid.win_rate:.1%}", f"{mancini.win_rate:.1%}"),
        ("Total PnL (pts)", f"{engine.total_pnl_pts:+.1f}", f"{targets.total_pnl_pts:+.1f}", f"{hybrid.total_pnl_pts:+.1f}", f"{mancini.total_pnl_pts:+.1f}"),
        ("Profit factor", f"{engine.profit_factor:.2f}", f"{targets.profit_factor:.2f}", f"{hybrid.profit_factor:.2f}", f"{mancini.profit_factor:.2f}"),
        ("Max drawdown (pts)", f"{engine.max_drawdown_pts:.1f}", f"{targets.max_drawdown_pts:.1f}", f"{hybrid.max_drawdown_pts:.1f}", f"{mancini.max_drawdown_pts:.1f}"),
    ]:
        print(f"  {label:<25} {e:>12} {t:>12} {h:>12} {m:>12}")

    # Per-trade
    print("\n── PER-TRADE METRICS ──")
    for label, res in [("Engine", engine), ("Targets", targets), ("Hybrid", hybrid), ("Mancini", mancini)]:
        trades = res.all_trades
        if trades:
            avg_win = np.mean([t.pnl_pts for t in trades if t.pnl_pts > 0]) if any(t.pnl_pts > 0 for t in trades) else 0
            avg_loss = np.mean([t.pnl_pts for t in trades if t.pnl_pts <= 0]) if any(t.pnl_pts <= 0 for t in trades) else 0
            print(f"  {label:<12} avg_win={avg_win:+.1f}  avg_loss={avg_loss:+.1f}  "
                  f"avg_pnl={np.mean([t.pnl_pts for t in trades]):+.1f}  trades={len(trades)}")

    # Targets vs Engine overlap
    print("\n── TARGETS-ONLY vs ENGINE TRADE OVERLAP ──")
    total_t = df["targets_trades"].sum()
    total_e = df["engine_trades"].sum()
    total_exact = df["targets_engine_exact"].sum()
    total_close = df["targets_engine_close"].sum()
    print(f"  Targets trades:          {total_t}")
    print(f"  Engine trades:           {total_e}")
    print(f"  Exact same entry:        {total_exact}")
    print(f"  Close match (±2 pts):    {total_close}")
    if total_t > 0:
        print(f"  % targets also in engine: {total_close / total_t:.1%}")
    print(f"  NEW targets-only trades:  {total_t - total_close}")

    # Targets vs Engine daily
    print("\n── TARGETS-ONLY vs ENGINE DAILY PnL ──")
    df["targets_vs_engine"] = df["targets_pnl"] - df["engine_pnl"]
    t_better = df[df["targets_vs_engine"] > 1]
    e_better = df[df["targets_vs_engine"] < -1]
    same = df[df["targets_vs_engine"].abs() <= 1]
    print(f"  Days targets better: {len(t_better)}  avg: {t_better['targets_vs_engine'].mean():+.1f} pts" if len(t_better) else "  Days targets better: 0")
    print(f"  Days engine better:  {len(e_better)}  avg: {e_better['targets_vs_engine'].mean():+.1f} pts" if len(e_better) else "  Days engine better:  0")
    print(f"  Days roughly same:   {len(same)}")

    # Monthly
    print("\n── MONTHLY PnL (Engine vs Targets-Only) ──")
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month").agg({
        "engine_pnl": "sum",
        "targets_pnl": "sum",
        "date": "count",
    }).rename(columns={"date": "days"})

    print(f"  {'Month':<10} {'Days':>5} {'Engine':>10} {'Targets':>10} {'T-E Diff':>10}")
    print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
    for month, row in monthly.iterrows():
        diff = row["targets_pnl"] - row["engine_pnl"]
        print(f"  {str(month):<10} {row['days']:>5.0f} {row['engine_pnl']:>+10.1f} "
              f"{row['targets_pnl']:>+10.1f} {diff:>+10.1f}")

    # Stat tests
    print("\n── STATISTICAL SIGNIFICANCE ──")
    from scipy import stats
    valid = df[(df["engine_trades"] > 0) | (df["targets_trades"] > 0)]
    if len(valid) > 10:
        t_stat, p = stats.ttest_rel(valid["targets_pnl"], valid["engine_pnl"])
        d = valid["targets_pnl"].mean() - valid["engine_pnl"].mean()
        print(f"  Targets vs Engine: mean diff={d:+.2f} pts/day  t={t_stat:.2f}  p={p:.4f}")
        if p < 0.05:
            w = "TARGETS" if d > 0 else "ENGINE"
            print(f"    → {w} significantly better at p<0.05")
        else:
            print(f"    → No significant difference")


if __name__ == "__main__":
    print("Loading data...")

    df = pd.read_parquet("data/ES_1m_2024-02-05_2026-02-05.parquet")
    df.index = df.index.tz_localize("US/Eastern")
    rth = df.between_time("09:30", "15:59")

    daily_dfs = {}
    for dt, group in rth.groupby(rth.index.date):
        daily_dfs[dt] = group
    print(f"  Trading days: {len(daily_dfs)}")

    with open("data/substack/all_posts.json") as f:
        posts = json.load(f)

    posts_by_date = {}
    for post in posts:
        levels = parse_mancini_levels(post["text"])
        if levels["supports"] or levels["resistances"]:
            posts_by_date[get_newsletter_date(post)] = levels
    print(f"  Posts with levels: {len(posts_by_date)}")
    print(f"  Overlapping days: {sum(1 for d in daily_dfs if d in posts_by_date)}")

    # Production params
    production_params = StrategyParams(
        acceptance_max_dip_pts=3.0,
        acceptance_min_hold_bars=7,
        acceptance_min_hold_bars_deep=8,
        fb_stop_buffer_pts=5.5,
        lr_stop_buffer_pts=5.0,
        swing_low_order=15,
    )
    production_elevator = ElevatorParams(
        min_levels_broken=2,
        min_velocity_pts_per_min=0.75,
    )
    production_exit = ExitParams(
        trailing_stop_pts=7.0,
        t1_exit_fraction=1.0,
    )

    print("\nRunning three-way comparison...")
    comp_df, e_res, m_res, h_res, t_res = run_three_way_comparison(
        daily_dfs, posts_by_date,
        strategy_params=production_params,
        elevator_params=production_elevator,
        exit_params=production_exit,
        min_rr_ratio=1.0,
    )

    comp_df.to_csv("data/level_comparison_4way.csv", index=False)
    print(f"\nResults saved to data/level_comparison_4way.csv")

    print_report(comp_df, e_res, m_res, h_res, t_res)
