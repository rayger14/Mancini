"""Regime backfill simulation.

Re-derives the daily regime (EMA / structure / composite) for every
session_date that appears in `data/training/trades.jsonl`, then tags each
historical trade with `would_have_been_filtered: bool` to estimate the
counterfactual PnL impact of flipping `use_regime_filter=True`.

Per-trade lookahead is avoided by computing the regime from minute bars
strictly BEFORE the trade's session_date.

Usage:
    python3 backtest/regime_backfill_sim.py [--mode ema|structure|composite|composite_strict]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.regime_filter import (
    RegimeParams,
    build_daily_bars,
    compute_regime,
)

ROOT = Path(__file__).resolve().parent.parent
TRADES = ROOT / "data" / "backtest_5y_trades.jsonl"
BARS = ROOT / "data" / "ES_1m_full_session_2021-01-01_2026-02-05.parquet"


def load_trades() -> list[dict]:
    """Read the per-trade dump produced by five_year_long_short.py."""
    rows = []
    with open(TRADES) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sd = r.get("date")
            if not sd:
                continue
            # Normalize date string (may be "2024-01-05" or "2024-01-05T..")
            sd = str(sd)[:10]
            rows.append({
                "entry_time": str(r.get("entry_time", "")),
                "session_date": sd,
                "direction": (r.get("direction") or "").lower(),
                "pattern": r.get("pattern"),
                "pnl_pts": r.get("pnl_pts", 0.0) or 0.0,
                "pnl_dollars": (r.get("pnl_pts", 0.0) or 0.0) * 50.0,  # ES multiplier
                "exit_reason": r.get("exit_reason"),
                "production_would_take": True,
            })
    return rows


def build_session_regimes(minute_df: pd.DataFrame, session_dates: list[str],
                          params: RegimeParams) -> dict[str, dict]:
    """For each session_date, compute regime using ONLY daily bars strictly
    before that date. Returns {session_date: {direction, longs_enabled, ...}}."""
    daily = build_daily_bars(minute_df)
    daily.index = pd.to_datetime(daily.index).normalize()

    out: dict[str, dict] = {}
    # daily.index inherits tz from minute bars; strip to compare with naive dates
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    for sd in sorted(set(session_dates)):
        sd_dt = pd.Timestamp(sd).normalize()
        prior = daily[daily.index < sd_dt]
        if len(prior) < 126:
            out[sd] = {"direction": "NEUTRAL", "longs_enabled": True,
                       "shorts_enabled": True, "ema_slope": 0.0, "armed": False}
            continue
        state = compute_regime(prior, params)
        out[sd] = {
            "direction": state.direction.name,
            "longs_enabled": state.longs_enabled,
            "shorts_enabled": state.shorts_enabled,
            "ema_slope": state.ema_slope,
            "vol_regime": state.vol_regime.name,
            "structure": state.structure,
            "armed": True,
        }
    return out


def simulate(mode: str) -> None:
    print(f"\n=== Regime backfill simulation — mode='{mode}' ===\n")

    trades = load_trades()
    if not trades:
        print("No trades found.")
        return

    # Restrict to trades production_would_take=True to mirror live behavior
    real_trades = [t for t in trades if t.get("production_would_take")]
    print(f"Loaded {len(trades)} trades total ({len(real_trades)} production-realistic).")

    print(f"Loading bars: {BARS.name} …")
    bars = pd.read_parquet(BARS)
    if "timestamp" in bars.columns:
        bars = bars.set_index("timestamp")
    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern", nonexistent="shift_forward",
                                            ambiguous="NaT")
    else:
        bars.index = bars.index.tz_convert("US/Eastern")
    bars.columns = [c.lower() for c in bars.columns]
    print(f"  {len(bars):,} bars, {bars.index.min()} → {bars.index.max()}")

    params = RegimeParams(mode=mode)
    session_dates = [t["session_date"] for t in real_trades]
    regimes = build_session_regimes(bars, session_dates, params)

    # Tag each trade
    kept_pnl = filtered_pnl = 0.0
    kept_n = filtered_n = 0
    by_dir = defaultdict(lambda: {"kept_n": 0, "kept_pnl": 0.0,
                                  "filt_n": 0, "filt_pnl": 0.0})
    by_year = defaultdict(lambda: {"kept_n": 0, "kept_pnl": 0.0,
                                   "filt_n": 0, "filt_pnl": 0.0})
    unarmed = 0
    for t in real_trades:
        r = regimes.get(t["session_date"])
        if not r or not r["armed"]:
            unarmed += 1
            kept_pnl += t["pnl_dollars"]; kept_n += 1
            continue

        is_long = t["direction"] == "long"
        is_short = t["direction"] == "short"
        would_filter = (is_long and not r["longs_enabled"]) or \
                       (is_short and not r["shorts_enabled"])

        year = t["session_date"][:4]
        d = t["direction"]
        if would_filter:
            filtered_pnl += t["pnl_dollars"]; filtered_n += 1
            by_dir[d]["filt_n"] += 1; by_dir[d]["filt_pnl"] += t["pnl_dollars"]
            by_year[year]["filt_n"] += 1; by_year[year]["filt_pnl"] += t["pnl_dollars"]
        else:
            kept_pnl += t["pnl_dollars"]; kept_n += 1
            by_dir[d]["kept_n"] += 1; by_dir[d]["kept_pnl"] += t["pnl_dollars"]
            by_year[year]["kept_n"] += 1; by_year[year]["kept_pnl"] += t["pnl_dollars"]

    total_pnl = kept_pnl + filtered_pnl
    print(f"\nRegime not armed (insufficient history): {unarmed} trades")
    print(f"\n--- Net PnL impact ---")
    print(f"All trades:          ${total_pnl:>10.2f}  ({kept_n+filtered_n} trades)")
    print(f"With filter on:      ${kept_pnl:>10.2f}  ({kept_n} kept)")
    print(f"  ...blocked PnL:    ${filtered_pnl:>10.2f}  ({filtered_n} blocked — these would NOT have fired)")
    delta = kept_pnl - total_pnl  # positive => filter helps
    sign = "+" if delta >= 0 else ""
    print(f"  ΔPnL by flipping:  {sign}${delta:>10.2f}  ({'filter HELPS' if delta > 0 else 'filter HURTS'})")

    print(f"\n--- By direction ---")
    print(f"{'dir':6} {'kept_n':>7} {'kept_pnl':>10} {'filt_n':>7} {'filt_pnl':>10}")
    for d in sorted(by_dir):
        s = by_dir[d]
        print(f"{d:6} {s['kept_n']:>7} ${s['kept_pnl']:>9.2f} {s['filt_n']:>7} ${s['filt_pnl']:>9.2f}")

    print(f"\n--- By year ---")
    print(f"{'year':6} {'kept_n':>7} {'kept_pnl':>10} {'filt_n':>7} {'filt_pnl':>10}  delta")
    for y in sorted(by_year):
        s = by_year[y]
        d = -s["filt_pnl"]
        sign = "+" if d >= 0 else ""
        print(f"{y:6} {s['kept_n']:>7} ${s['kept_pnl']:>9.2f} {s['filt_n']:>7} ${s['filt_pnl']:>9.2f}  {sign}${d:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ema",
                    choices=["ema", "structure", "composite", "composite_strict"])
    args = ap.parse_args()
    simulate(args.mode)


if __name__ == "__main__":
    main()
