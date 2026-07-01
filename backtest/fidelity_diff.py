"""Backtest-fidelity diff: run the backtest engine WITH Mancini plan levels over
recent sessions and diff its LONG entries against what the bot actually traded
live. Answers "how faithfully does the backtest reproduce live?" on the ~50
plan-available sessions (the 5y dataset has no plans, so this is the only window
where plan-augmented fidelity can be measured).

Run on the VM (plans/parquets/live-trades live there):
  docker cp backtest/fidelity_diff.py mancini-mancini-bot-1:/app/fidelity_diff.py
  docker exec -e PLAN_DIR=/app/data -e SESSIONS_DIR=/app/data/sessions \
    -e TRADE_LOG=/app/logs/trades.jsonl -w /app mancini-mancini-bot-1 \
    python3 fidelity_diff.py
"""
from __future__ import annotations

import os
from datetime import datetime, date, time as dt_time
from pathlib import Path

from config.settings import ESContractSpec
from live.ib_runner import (
    PRODUCTION_STRATEGY, PRODUCTION_ELEVATOR, PRODUCTION_EXIT,
    PRODUCTION_RISK, PRODUCTION_SESSION, PRODUCTION_REGIME,
)
from strategy.mancini_long import ManciniLongStrategy

# MES spec (same as backtest/feature_comparison) — inlined so the harness has no
# dependency on the backtest package (it isn't shipped in the live container).
MES = ESContractSpec(symbol="MES", tick_size=0.25, tick_value=1.25,
                     point_value=5.0, margin_initial=1_265.0,
                     margin_maintenance=1_150.0, exchange="CME")
from core.mancini_plan_levels import build_plan_levels
from live.retrospective import load_session_bars, load_trades
from live.mancini_llm_extract import load_plan

PLAN_DIR = Path(os.environ.get("PLAN_DIR", "/app/data"))
SESSIONS_DIR = Path(os.environ.get("SESSIONS_DIR", "/app/data/sessions"))


def _rth(df):
    if df is None or len(df) == 0:
        return None
    mask = df.index.map(lambda t: dt_time(9, 30) <= t.time() < dt_time(16, 0))
    out = df[mask]
    return out if len(out) else None


def _match(live_prices, bt_prices, tol=5.0):
    """Greedy nearest-price match of live entries to backtest entries within tol.
    Returns (matched, live_only, bt_only)."""
    bt = sorted(bt_prices)
    used = [False] * len(bt)
    matched = 0
    for lp in live_prices:
        best, bd = -1, tol + 1
        for i, bp in enumerate(bt):
            if used[i]:
                continue
            d = abs(bp - lp)
            if d <= tol and d < bd:
                best, bd = i, d
        if best >= 0:
            used[best] = True
            matched += 1
    return matched, len(live_prices) - matched, used.count(False)


def discover_dates():
    """Sessions that have a plan + a session parquet + live trades."""
    out = []
    for p in sorted(PLAN_DIR.glob("mancini_plan_*.json")):
        d = p.stem.replace("mancini_plan_", "")
        if (SESSIONS_DIR / f"{d}_bars.parquet").exists():
            out.append(d)
    return out


PROD_ONLY = os.environ.get("PROD_ONLY", "1") == "1"


def _live_long_prices(d):
    """Live LONG entry prices. With PROD_ONLY (default), keep only
    production-window trades (production_would_take=True) — the ones the backtest
    can structurally take — so the fidelity number reflects true entry/detection
    fidelity, not the collection-mode trades the backtest can never reproduce."""
    prices = []
    for t in load_trades(d):
        if t.get("event") != "entry":
            continue
        if (t.get("direction") or "long") != "long":
            continue
        if PROD_ONLY and not t.get("production_would_take"):
            continue
        p = t.get("entry_price") or (t.get("signal") or {}).get("entry") or 0
        if p:
            prices.append(float(p))
    return prices


def run(dates):
    strat = ManciniLongStrategy(
        strategy_params=PRODUCTION_STRATEGY, elevator_params=PRODUCTION_ELEVATOR,
        exit_params=PRODUCTION_EXIT, risk_params=PRODUCTION_RISK,
        session_times=PRODUCTION_SESSION, contract=MES,
        min_rr_ratio=PRODUCTION_RISK.min_rr_ratio,
        rth_filter=(dt_time(9, 30), dt_time(16, 0)), regime_params=PRODUCTION_REGIME,
    )
    prev = None
    tot_m = tot_lo = tot_bo = 0
    print(f"{'date':<12} {'live':>4} {'bt':>4} {'match':>5} {'live_only':>9} {'bt_only':>7}  plan")
    for d in sorted(dates):
        sdf = load_session_bars(d)
        if sdf is None:
            continue
        dd = date.fromisoformat(d)
        # Plan-level timestamps must match the bars' tz (live uses datetime.now(_ET),
        # tz-aware) or run_day's level/timestamp comparisons fail.
        import pandas as _pd
        now = _pd.Timestamp(datetime.combine(dd, dt_time(9, 30)))
        if getattr(sdf.index, "tz", None) is not None:
            now = now.tz_localize(sdf.index.tz)
        now = now.to_pydatetime()
        plan = load_plan(dd, input_dir=PLAN_DIR)
        strat._extra_levels = build_plan_levels(plan, now) if plan else []
        before = len(strat.trade_records)
        try:
            strat.run_day(sdf, prior_day_df=_rth(prev),
                          session_date=datetime.combine(dd, dt_time(0, 0)))
        except Exception as e:
            print(f"{d:<12} run_day error: {e}")
            prev = sdf
            continue
        bt_prices = [t.entry_price for t in strat.trade_records[before:]
                     if getattr(t, "direction", "long") == "long"]
        live_prices = _live_long_prices(d)
        m, lo, bo = _match(live_prices, bt_prices)
        tot_m += m; tot_lo += lo; tot_bo += bo
        if live_prices or bt_prices:
            print(f"{d:<12} {len(live_prices):>4} {len(bt_prices):>4} {m:>5} {lo:>9} {bo:>7}"
                  f"  {'Y' if plan else '-'}")
        prev = sdf

    live_total = tot_m + tot_lo
    fidelity = 100.0 * tot_m / live_total if live_total else 0.0
    print("\n=== FIDELITY (long entries, plan-augmented backtest vs live) ===")
    print(f"  reproduced {tot_m}/{live_total} live entries = {fidelity:.0f}% fidelity")
    print(f"  live-only (backtest missed): {tot_lo}   backtest-only (extra): {tot_bo}")
    print("  live-only ⇒ likely missing level / daily_bias / collection-mode gate")
    print("  backtest-only ⇒ idealized fills / no live time-gate")


if __name__ == "__main__":
    run(discover_dates())
