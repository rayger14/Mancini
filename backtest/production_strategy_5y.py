"""5-year backtest of the LIVE production strategy.

This harness imports PRODUCTION_STRATEGY (and the rest of the production
config tuple) directly from live/ib_runner.py and runs them over the same
5y of ES 1-minute data as backtest/five_year_long_short.py.

Why this exists: the existing five_year_long_short.py harness uses
allow_short_fr=True / allow_short_lj=True (FR/LJ short patterns in
patterns_short.py). Those patterns are OFF in production
(allow_short_fr=False by default). Production runs allow_breakdown_short
/ allow_velocity_short / allow_backtest_short — the v2 patterns in
patterns_short_v2.py — which the legacy harness never exercises. So the
+$93K headline from that harness has no clean mapping to live behavior.

This harness fixes that: same engine path the live bot uses
(SignalAggregator + ManciniLongStrategy with PRODUCTION_STRATEGY),
same exits (PRODUCTION_EXIT, 4 contracts, 75/25 split), same risk
(PRODUCTION_RISK, max_daily_loss_pts=9999), same session windows
(FULL_SESSION).

Usage:
    python3 backtest/production_strategy_5y.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, time as dt_time, timedelta, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

# Silence engine logs during the long backtest run
logger.remove()

from strategy.mancini_long import ManciniLongStrategy
from live.ib_runner import (
    PRODUCTION_STRATEGY,
    PRODUCTION_ELEVATOR,
    PRODUCTION_EXIT,
    PRODUCTION_RISK,
    FULL_SESSION,
    MES_CONTRACT,
)


# --- Data loading & session bucketing (same as five_year_long_short.py) -----

def load_data() -> pd.DataFrame:
    path = (Path(__file__).parent.parent
            / "data" / "ES_1m_full_session_2021-01-01_2026-02-05.parquet")
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    return df


def build_sessions(df: pd.DataFrame) -> list[tuple[date, pd.DataFrame]]:
    """Bucket 1-min bars into trading sessions starting at 18:00 ET."""
    evening_mask = df.index.time == dt_time(18, 0)
    session_starts = df.index[evening_mask]
    sessions = []
    for start_ts in session_starts:
        next_day = start_ts.date() + timedelta(days=1)
        end_ts = pd.Timestamp(
            datetime.combine(next_day, dt_time(16, 59)),
            tz="US/Eastern",
        )
        s = df[(df.index >= start_ts) & (df.index <= end_ts)]
        # Skip the 17:00-18:00 globex break
        s = s[~((s.index.time >= dt_time(17, 0)) & (s.index.time < dt_time(18, 0)))]
        if len(s) > 0:
            sessions.append((next_day, s))
    return sessions


def get_window(t: dt_time) -> str:
    if dt_time(9, 30) <= t < dt_time(13, 0):
        return "Morning"
    if dt_time(15, 0) <= t <= dt_time(16, 50):
        return "Afternoon"
    if t >= dt_time(22, 0) or t < dt_time(2, 0):
        return "Late Night"
    if dt_time(6, 0) <= t < dt_time(9, 30):
        return "Pre-RTH"
    return "Blocked"


# Production pattern_type → trade direction
_LONG_PATTERNS = frozenset({"failed_breakdown", "level_reclaim"})
_SHORT_PATTERNS = frozenset({
    "breakdown_short", "velocity_short", "backtest_short",
    # FR/LJ — should never fire in production but mapped for safety
    "failed_rally", "level_rejection",
})


def classify_direction(pattern_type: str) -> str:
    if pattern_type in _SHORT_PATTERNS:
        return "SHORT"
    return "LONG"


# --- Backtest core loop -----------------------------------------------------

def run_backtest(sessions, skip_mondays: bool = True):
    strategy = ManciniLongStrategy(
        strategy_params=PRODUCTION_STRATEGY,
        elevator_params=PRODUCTION_ELEVATOR,
        exit_params=PRODUCTION_EXIT,
        risk_params=PRODUCTION_RISK,
        session_times=FULL_SESSION,
        contract=MES_CONTRACT,
        # PRODUCTION_RISK.min_rr_ratio is the authoritative floor; mirror here
        min_rr_ratio=PRODUCTION_RISK.min_rr_ratio,
        rth_filter=(dt_time(9, 30), dt_time(16, 0)),
    )

    all_trades = []
    prev_session_df = None
    sessions_total = 0
    sessions_skipped_monday = 0

    for session_date, session_df in sessions:
        if skip_mondays and session_date.weekday() == 0:
            sessions_skipped_monday += 1
            prev_session_df = session_df
            continue

        sessions_total += 1

        prior_rth = None
        if prev_session_df is not None:
            rth_mask = prev_session_df.index.map(
                lambda t: dt_time(9, 30) <= t.time() < dt_time(16, 0)
            )
            prior_rth = prev_session_df[rth_mask]
            if len(prior_rth) == 0:
                prior_rth = None

        strategy.run_day(
            session_df,
            prior_day_df=prior_rth,
            session_date=datetime.combine(session_date, dt_time(0, 0)),
        )

        for t in strategy.trade_records:
            if t.entry_bar_idx >= len(session_df):
                continue
            entry_ts = session_df.index[t.entry_bar_idx]
            window = get_window(entry_ts.time())
            if window == "Blocked":
                continue
            # Afternoon: only FB (long) and FR-equivalent shorts; skip
            # other patterns in afternoon — same rule as legacy harness
            if window == "Afternoon" and t.pattern_type not in (
                "failed_breakdown", "failed_rally", "breakdown_short",
            ):
                continue

            all_trades.append({
                "date": session_date,
                "entry_time": entry_ts,
                "window": window,
                "pattern": t.pattern_type,
                "direction": classify_direction(t.pattern_type),
                "level_type": t.level_type,
                "entry": t.entry_price,
                "stop": t.stop_price,
                "target_1": t.target_1,
                "pnl_pts": t.pnl_pts,
                "risk_pts": t.risk_pts,
                "rr_ratio": t.rr_ratio_t1,
                "exit_reason": t.exit_reason,
                "won": t.pnl_pts > 0,
                "lqs": getattr(t, "lqs", 0),
            })

        prev_session_df = session_df

    return all_trades, sessions_total, sessions_skipped_monday


# --- Reporting --------------------------------------------------------------

def _summarize(trades, label, contract_value=50.0):
    n = len(trades)
    if n == 0:
        print(f"  {label}: 0 trades"); return
    wins = [t for t in trades if t["won"]]
    losses = [t for t in trades if not t["won"]]
    total_pnl = sum(t["pnl_pts"] for t in trades)
    wr = len(wins) / n * 100
    gross_p = sum(t["pnl_pts"] for t in wins)
    gross_l = abs(sum(t["pnl_pts"] for t in losses))
    pf = gross_p / gross_l if gross_l > 0 else float("inf")
    avg_w = np.mean([t["pnl_pts"] for t in wins]) if wins else 0.0
    avg_l = np.mean([t["pnl_pts"] for t in losses]) if losses else 0.0

    cum = 0.0; peak = 0.0; max_dd = 0.0
    for t in trades:
        cum += t["pnl_pts"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    print(f"\n  {label}")
    print(f"  {'─'*75}")
    print(f"  Trades: {n}  |  Wins: {len(wins)}  Losses: {len(losses)}  |  "
          f"WR: {wr:.1f}%  |  PF: {pf:.2f}")
    print(f"  PnL: {total_pnl:+,.1f} pts (${total_pnl*contract_value:+,.0f} ES)  |  "
          f"Max DD: {max_dd:.1f} pts")
    print(f"  Avg Win: {avg_w:+.1f}  |  Avg Loss: {avg_l:+.1f}")


def _per_pattern(trades):
    print(f"\n  {'Pattern':22} {'Dir':>5}  {'N':>5}  {'WR':>6}  {'PF':>6}  {'PnL':>10}")
    print(f"  {'-'*22} {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*10}")
    grouped = {}
    for t in trades:
        key = (t["pattern"], t["direction"])
        grouped.setdefault(key, []).append(t)
    for (pat, direction), items in sorted(grouped.items()):
        n = len(items)
        wins = sum(1 for t in items if t["won"])
        wr = wins/n*100
        gp = sum(t["pnl_pts"] for t in items if t["won"])
        gl = abs(sum(t["pnl_pts"] for t in items if not t["won"]))
        pf = gp/gl if gl > 0 else float("inf")
        pnl = sum(t["pnl_pts"] for t in items)
        print(f"  {pat:22} {direction:>5}  {n:>5}  {wr:>5.1f}% {pf:>6.2f}  {pnl:>+9.1f}")


def _per_year(trades):
    print(f"\n  {'Year':>4}  {'N':>5}  {'L':>4}  {'S':>4}  {'WR':>6}  {'PF':>6}  {'PnL':>10}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*10}")
    years = sorted({t["date"].year for t in trades})
    for y in years:
        yr = [t for t in trades if t["date"].year == y]
        if not yr: continue
        n = len(yr)
        longs = sum(1 for t in yr if t["direction"] == "LONG")
        shorts = sum(1 for t in yr if t["direction"] == "SHORT")
        wins = sum(1 for t in yr if t["won"])
        gp = sum(t["pnl_pts"] for t in yr if t["won"])
        gl = abs(sum(t["pnl_pts"] for t in yr if not t["won"]))
        pf = gp/gl if gl > 0 else float("inf")
        pnl = sum(t["pnl_pts"] for t in yr)
        print(f"  {y:>4}  {n:>5}  {longs:>4}  {shorts:>4}  {wins/n*100:>5.1f}%  {pf:>6.2f}  {pnl:>+9.1f}")


def _per_window(trades):
    print(f"\n  {'Window':>14}  {'N':>5}  {'WR':>6}  {'PF':>6}  {'PnL':>10}  {'L/S':>10}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")
    groups = {}
    for t in trades:
        groups.setdefault(t["window"], []).append(t)
    for w in ("Morning", "Afternoon", "Late Night", "Pre-RTH"):
        ts = groups.get(w, [])
        if not ts: continue
        n = len(ts)
        wins = sum(1 for t in ts if t["won"])
        gp = sum(t["pnl_pts"] for t in ts if t["won"])
        gl = abs(sum(t["pnl_pts"] for t in ts if not t["won"]))
        pf = gp/gl if gl > 0 else float("inf")
        pnl = sum(t["pnl_pts"] for t in ts)
        longs = sum(1 for t in ts if t["direction"] == "LONG")
        shorts = sum(1 for t in ts if t["direction"] == "SHORT")
        print(f"  {w:>14}  {n:>5}  {wins/n*100:>5.1f}%  {pf:>6.2f}  {pnl:>+9.1f}  {longs:>4}/{shorts:>4}")


def _short_pattern_breakdown(trades):
    """Detail short side — these are the patterns we just rewrote."""
    print(f"\n  --- SHORT pattern detail (live production patterns only) ---")
    shorts = [t for t in trades if t["direction"] == "SHORT"]
    if not shorts:
        print(f"    No short trades fired in 5y backtest.")
        return

    by_pat_lvl = {}
    for t in shorts:
        key = (t["pattern"], str(t["level_type"]))
        by_pat_lvl.setdefault(key, []).append(t)
    print(f"  {'Pattern':22} {'Level':22}  {'N':>3}  {'WR':>6}  {'PnL':>9}")
    for (pat, lvl), items in sorted(
        by_pat_lvl.items(), key=lambda x: sum(t["pnl_pts"] for t in x[1])
    ):
        n = len(items)
        wins = sum(1 for t in items if t["won"])
        pnl = sum(t["pnl_pts"] for t in items)
        print(f"  {pat:22} {lvl:22}  {n:>3}  {wins/n*100:>5.1f}%  {pnl:>+8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-mondays", action="store_true",
                    help="Don't skip Mondays (default skips Mondays per Optuna v2)")
    ap.add_argument("--dump", action="store_true",
                    help="Dump trades to data/backtest_5y_production_trades.jsonl")
    args = ap.parse_args()

    print("=" * 80)
    print("5-YEAR PRODUCTION-STRATEGY BACKTEST")
    print("Engine: live PRODUCTION_STRATEGY (BD/VBD/BTS shorts, FB/LR longs)")
    print("=" * 80)

    print(f"\nProduction config flags:")
    print(f"  allow_breakdown_short  = {PRODUCTION_STRATEGY.allow_breakdown_short}")
    print(f"  allow_velocity_short   = {PRODUCTION_STRATEGY.allow_velocity_short}")
    print(f"  allow_backtest_short   = {PRODUCTION_STRATEGY.allow_backtest_short}")
    print(f"  allow_short_fr/_lj     = {PRODUCTION_STRATEGY.allow_short_fr}"
          f"/{PRODUCTION_STRATEGY.allow_short_lj}  (legacy patterns)")
    print(f"  block_pdl_shorts       = {PRODUCTION_STRATEGY.block_pdl_shorts}")
    print(f"  short_size_factor      = {PRODUCTION_STRATEGY.short_size_factor}")
    print(f"  use_mancini_llm_plan   = {PRODUCTION_STRATEGY.use_mancini_llm_plan}")
    print(f"  shadow_mode_features   = {PRODUCTION_STRATEGY.shadow_mode_features}")
    print(f"  default_contracts (exit) = {PRODUCTION_EXIT.default_contracts}")

    print(f"\nLoading 5-year full-session data...")
    df = load_data()
    sessions = build_sessions(df)
    print(f"Built {len(sessions)} sessions ({sessions[0][0]} → {sessions[-1][0]})")

    print(f"\nRunning backtest (skip Mondays={'no' if args.keep_mondays else 'yes'})...")
    trades, sessions_total, skipped_mon = run_backtest(
        sessions, skip_mondays=not args.keep_mondays,
    )
    print(f"  {sessions_total} sessions traded, {skipped_mon} skipped (Mondays)")
    print(f"  {len(trades)} trades emitted")

    print("\n" + "=" * 80)
    print("FULL PERIOD")
    print("=" * 80)
    _summarize(trades, "ALL TRADES")
    _per_pattern(trades)
    _per_year(trades)
    _per_window(trades)
    _short_pattern_breakdown(trades)

    # Direction split
    print("\n" + "=" * 80)
    print("DIRECTION SPLIT")
    print("=" * 80)
    longs = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]
    _summarize(longs, "LONG total")
    _summarize(shorts, "SHORT total (BD + VBD + BTS only)")

    # Quick dollar summary
    total_pts = sum(t["pnl_pts"] for t in trades)
    long_pts = sum(t["pnl_pts"] for t in longs)
    short_pts = sum(t["pnl_pts"] for t in shorts)
    print("\n" + "=" * 80)
    print("DOLLAR PnL")
    print("=" * 80)
    print(f"  Full period:  {total_pts:+,.1f} pts = ${total_pts*50:+,.0f} ES  "
          f"(${total_pts*5:+,.0f} MES)")
    print(f"  LONG only:    {long_pts:+,.1f} pts = ${long_pts*50:+,.0f} ES")
    print(f"  SHORT only:   {short_pts:+,.1f} pts = ${short_pts*50:+,.0f} ES")

    if args.dump:
        out = (Path(__file__).resolve().parent.parent
               / "data" / "backtest_5y_production_trades.jsonl")
        with open(out, "w") as f:
            for t in trades:
                row = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                       for k, v in t.items()}
                f.write(json.dumps(row, default=str) + "\n")
        print(f"\nDumped {len(trades)} trades → {out}")


if __name__ == "__main__":
    main()
