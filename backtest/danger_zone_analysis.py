"""5-year historical analysis: how does the Mancini danger zone gate
   actually perform on FB long trades?

Runs the production engine on 2021-2026 via Nautilus and aggregates every
FB long trade by:
  * distance = entry_price - level_price (the synthetic danger zone)
  * confirmation_type = "acceptance" / "non_acceptance"

The question we're answering:
  If we REMOVED the Mancini LLM plan's danger-zone gate, how would
  historical PnL have changed? The engine's pattern detector already
  enforces both Mancini protocols before emitting a signal — so the
  gate's hard-reject is only sensible if these qualified entries lose
  money in aggregate.

Usage:
    python3 backtest/danger_zone_analysis.py [--year 2024] [--all-years]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from backtest.nautilus_runner import NautilusBacktestRunner, NautilusBacktestConfig
from backtest.nautilus_production_5y import load_data, build_daily_sessions
from live.ib_runner import (
    PRODUCTION_STRATEGY, PRODUCTION_ELEVATOR,
    PRODUCTION_EXIT, PRODUCTION_RISK,
)


DANGER_ZONE_THRESHOLD_PTS = 5.0  # Mancini: "5 points above swept low"


def _run_engine_on_years(years: list[int]) -> list:
    """Run Nautilus on the given years using PRODUCTION_* params,
    return every completed TradeRecord."""
    print(f"Loading 5y ES data…", flush=True)
    df = load_data()
    sessions = build_daily_sessions(df)
    target = {d: v for d, v in sessions.items() if d.year in years}
    sorted_days = sorted(target)
    print(f"Sessions in target years {years}: {len(sorted_days)} "
          f"({sorted_days[0]} → {sorted_days[-1]})", flush=True)

    cfg = NautilusBacktestConfig(
        strategy_params=PRODUCTION_STRATEGY,
        elevator_params=PRODUCTION_ELEVATOR,
        exit_params=PRODUCTION_EXIT,
        risk_params=PRODUCTION_RISK,
        min_rr_ratio=PRODUCTION_RISK.min_rr_ratio,
    )
    runner = NautilusBacktestRunner(cfg)
    all_trades = []
    prior = None
    for i, d in enumerate(sorted_days):
        df_sess = target[d]
        if len(df_sess) < 10:
            continue
        try:
            r = runner.run_single_day(df_sess, prior_day_df=prior, day=d)
            all_trades.extend(r.trade_records)
            prior = df_sess
            if (i + 1) % 100 == 0:
                tot = sum(t.pnl_pts for t in all_trades)
                print(f"  [{i+1:>4}/{len(sorted_days)}] {d}  "
                      f"cum_n={len(all_trades):>4}  cum_pnl={tot:+.1f}",
                      flush=True)
        except Exception as e:
            print(f"  ERROR on {d}: {e}", flush=True)
    print(f"\nTotal trades from engine: {len(all_trades)}", flush=True)
    return all_trades


def _summarize_bucket(label: str, trades: list) -> None:
    if not trades:
        print(f"{label}: empty")
        return
    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts <= 0]
    total = sum(t.pnl_pts for t in trades)
    avg = mean(t.pnl_pts for t in trades)
    med = median(t.pnl_pts for t in trades)
    wr = len(wins) / len(trades) * 100
    avg_w = mean(t.pnl_pts for t in wins) if wins else 0.0
    avg_l = mean(t.pnl_pts for t in losses) if losses else 0.0
    avg_d = mean(t.entry_price - t.level_price for t in trades)
    print(f"\n{label}")
    print(f"  N             {len(trades)}")
    print(f"  Win rate      {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Total PnL     {total:+.1f} pts  (${total*50:+.0f} on ES, 1ct)")
    print(f"  Avg PnL/trade {avg:+.2f} pts")
    print(f"  Median PnL    {med:+.2f} pts")
    print(f"  Avg win       {avg_w:+.2f} pts")
    print(f"  Avg loss      {avg_l:+.2f} pts")
    print(f"  Avg distance  {avg_d:.2f} pts above level")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, action="append", default=None,
                    help="Specific year(s) to run. Repeat to include multiple.")
    ap.add_argument("--all-years", action="store_true",
                    help="Run on all available years 2021-2026 (default).")
    args = ap.parse_args()

    years = args.year if args.year else list(range(2021, 2027))
    if args.all_years:
        years = list(range(2021, 2027))
    print(f"Running danger-zone analysis on years: {years}")

    all_trades = _run_engine_on_years(years)

    # Diagnostic: distribution of pattern_type / level_price
    from collections import Counter
    pt_counter = Counter((t.pattern_type or "?") for t in all_trades)
    print(f"\nDiagnostic — pattern_type distribution:")
    for pt, c in pt_counter.most_common():
        print(f"  {pt!r:>30}: {c}")
    if all_trades:
        has_level = sum(1 for t in all_trades if getattr(t, "level_price", 0.0) > 0)
        print(f"\nTrades with level_price > 0: {has_level} / {len(all_trades)}")
        sample = next((t for t in all_trades if (t.pattern_type or "").lower().startswith("fail")), all_trades[0])
        print(f"\nSample record fields:")
        for k in ("entry_time","entry_price","level_price","level_type","pattern_type",
                  "confirmation_type","stop_price","target_1","pnl_pts","exit_reason"):
            print(f"  {k:>20}: {getattr(sample, k, '<missing>')!r}")

    # Filter to FB long trades with valid level_price
    fb_long = [
        t for t in all_trades
        if (t.pattern_type or "").lower() == "failed_breakdown"
        and t.level_price > 0
    ]
    print(f"\nFB long trades (with valid level_price): {len(fb_long)}")

    # ------------------ Bucket by distance ------------------
    danger = [t for t in fb_long if 0 < (t.entry_price - t.level_price) <= DANGER_ZONE_THRESHOLD_PTS]
    safe = [t for t in fb_long if (t.entry_price - t.level_price) > DANGER_ZONE_THRESHOLD_PTS]
    at_or_below = [t for t in fb_long if (t.entry_price - t.level_price) <= 0]

    print(f"  ⚠️  DANGER ZONE (0 < dist <= {DANGER_ZONE_THRESHOLD_PTS}pt): {len(danger)}")
    print(f"  ✅ SAFE ZONE  (dist > {DANGER_ZONE_THRESHOLD_PTS}pt):       {len(safe)}")
    print(f"  ❓ At/below   (dist <= 0pt, data artifact):    {len(at_or_below)}")

    print("\n" + "=" * 70)
    print("BUCKET 1: Distance-based classification (Mancini's danger zone)")
    print("=" * 70)
    _summarize_bucket("⚠️  DANGER ZONE (entry within 5pt above level)", danger)
    _summarize_bucket("✅ SAFE ZONE  (entry > 5pt above level)", safe)
    _summarize_bucket("📊 ALL FB LONGS", fb_long)

    # ------------------ Bucket by confirmation_type ------------------
    print("\n" + "=" * 70)
    print("BUCKET 2: Confirmation type (acceptance vs non_acceptance)")
    print("=" * 70)
    by_conf = defaultdict(list)
    for t in fb_long:
        by_conf[t.confirmation_type or "unknown"].append(t)
    for ct, bucket in sorted(by_conf.items(), key=lambda x: -len(x[1])):
        _summarize_bucket(f"  confirmation_type={ct!r}", bucket)

    # ------------------ Interaction: danger zone × confirmation ------------------
    print("\n" + "=" * 70)
    print("BUCKET 3: Danger zone × confirmation_type interaction")
    print("=" * 70)
    for ct in sorted(set(t.confirmation_type or "unknown" for t in fb_long)):
        dz = [t for t in danger if (t.confirmation_type or "unknown") == ct]
        sz = [t for t in safe if (t.confirmation_type or "unknown") == ct]
        _summarize_bucket(f"⚠️ DANGER ZONE × confirmation={ct!r}", dz)
        _summarize_bucket(f"✅ SAFE ZONE   × confirmation={ct!r}", sz)

    # ------------------ Yearly breakdown of danger zone ------------------
    print("\n" + "=" * 70)
    print("DANGER ZONE per year — is the effect stable?")
    print("=" * 70)
    print(f"{'year':>6} {'N':>5} {'WR%':>6} {'totalPnL':>10} {'avgPnL':>8}")
    by_year = defaultdict(list)
    for t in danger:
        by_year[t.entry_time.year].append(t)
    for y in sorted(by_year):
        bucket = by_year[y]
        if not bucket:
            continue
        wins = sum(1 for t in bucket if t.pnl_pts > 0)
        wr = wins / len(bucket) * 100
        total = sum(t.pnl_pts for t in bucket)
        avg = total / len(bucket)
        print(f"{y:>6} {len(bucket):>5} {wr:>6.1f} {total:>+10.1f} {avg:>+8.2f}")

    # ------------------ Verdict ------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if not danger:
        print("No danger-zone trades found.")
        return
    d_total = sum(t.pnl_pts for t in danger)
    d_wr = sum(1 for t in danger if t.pnl_pts > 0) / len(danger) * 100
    s_total = sum(t.pnl_pts for t in safe) if safe else 0.0
    s_wr = sum(1 for t in safe if t.pnl_pts > 0) / len(safe) * 100 if safe else 0.0
    s_avg = (s_total / len(safe)) if safe else 0.0
    d_avg = d_total / len(danger)

    print(f"DANGER ZONE  : N={len(danger):>4}  WR={d_wr:>5.1f}%  totalPnL={d_total:+9.1f} pts  avgEV={d_avg:+6.2f}")
    print(f"SAFE ZONE    : N={len(safe):>4}  WR={s_wr:>5.1f}%  totalPnL={s_total:+9.1f} pts  avgEV={s_avg:+6.2f}")
    print()

    if d_total > 0:
        print(f"✅ Danger zone is net POSITIVE: +{d_total:.1f} pts over {len(danger)} trades")
        print(f"   Removing the gate would have added ~${d_total*50:.0f} (ES) "
              f"to total PnL over the test period.")
    else:
        print(f"❌ Danger zone is net NEGATIVE: {d_total:.1f} pts over {len(danger)} trades")
        print(f"   The gate would have saved ~${-d_total*50:.0f} (ES) by blocking these.")
    if d_avg < s_avg:
        print(f"   But EV/trade is {s_avg - d_avg:.2f} pts WORSE than safe zone — "
              f"the gate is directionally correct even if it's blocking some winners.")


if __name__ == "__main__":
    main()
