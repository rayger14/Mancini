"""5-year backtest: Jan 2021 - Feb 2026.

True OOS: 2021-2024 data was NEVER seen during parameter tuning.
In-sample: 2024-2026 (params were optimized on this period).

Usage:
    python3 backtest/five_year_backtest.py
"""

from __future__ import annotations

import sys
from datetime import datetime, time as dt_time, timedelta, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

logger.remove()

from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    RiskParams, SessionTimes, ESContractSpec,
)
from strategy.mancini_long import ManciniLongStrategy


# ── Production config ─────────────────────────────────────────────────

STRATEGY = StrategyParams(
    swing_low_order=15,
    multi_hour_rally_min_pts=22.5,
    level_reclaim_min_touches=4,
    acceptance_min_hold_bars=7,
    acceptance_min_hold_bars_deep=8,
    acceptance_max_dip_pts=4.0,
    true_breakdown_abort_bars=20,
    fb_stop_buffer_pts=5.5,
    lr_stop_buffer_pts=5.0,
    non_acceptance_min_recovery_pts=5.0,
    max_fb_sweep_depth_pts=10.0,
    level_sweep_min_bars_below=3,
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

FULL_SESSION = SessionTimes(
    rth_open=dt_time(18, 0),
    rth_close=dt_time(17, 0),
    morning_window_start=dt_time(9, 30),
    morning_window_end=dt_time(11, 0),
    afternoon_window_start=dt_time(15, 0),
    afternoon_window_end=dt_time(16, 50),
    eod_flatten_time=dt_time(16, 50),
    chop_zone_start=dt_time(13, 0),
    chop_zone_end=dt_time(15, 0),
)

MES = ESContractSpec(
    symbol="MES", tick_size=0.25, tick_value=1.25,
    point_value=5.0, margin_initial=1_265.0,
    margin_maintenance=1_150.0, exchange="CME",
)


def load_data() -> pd.DataFrame:
    path = Path(__file__).parent.parent / "data" / "ES_1m_full_session_2021-01-01_2026-02-05.parquet"
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    return df


def build_sessions(df: pd.DataFrame) -> list[tuple[date, pd.DataFrame]]:
    evening_mask = df.index.time == dt_time(18, 0)
    session_starts = df.index[evening_mask]
    sessions = []
    for start_ts in session_starts:
        next_day = start_ts.date() + timedelta(days=1)
        end_ts = pd.Timestamp(
            datetime.combine(next_day, dt_time(16, 59)),
            tz="US/Eastern",
        )
        session_df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        session_df = session_df[
            ~((session_df.index.time >= dt_time(17, 0)) & (session_df.index.time < dt_time(18, 0)))
        ]
        if len(session_df) > 0:
            sessions.append((next_day, session_df))
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


def run_backtest(sessions):
    strategy = ManciniLongStrategy(
        strategy_params=STRATEGY,
        elevator_params=ELEVATOR,
        exit_params=EXIT,
        risk_params=RISK,
        session_times=FULL_SESSION,
        contract=MES,
        min_rr_ratio=1.0,
        rth_filter=(dt_time(9, 30), dt_time(16, 0)),
    )

    all_trades = []
    prev_session_df = None

    for session_date, session_df in sessions:
        if session_date.weekday() == 0:  # skip Mondays
            prev_session_df = session_df
            continue

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
            et = entry_ts.time()
            window = get_window(et)

            if window == "Blocked":
                continue
            if window == "Afternoon" and t.pattern_type != "failed_breakdown":
                continue

            all_trades.append({
                "date": session_date,
                "entry_time": entry_ts,
                "time": et,
                "window": window,
                "pattern": t.pattern_type,
                "level_type": t.level_type,
                "entry": t.entry_price,
                "stop": t.stop_price,
                "target_1": t.target_1,
                "pnl_pts": t.pnl_pts,
                "risk_pts": t.risk_pts,
                "rr_ratio": t.rr_ratio_t1,
                "exit_reason": t.exit_reason,
                "won": t.pnl_pts > 0,
            })

        prev_session_df = session_df

    return all_trades


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

    avg_risk = np.mean([t["risk_pts"] for t in trades])

    # Trading days
    unique_dates = set(t["date"] for t in trades)
    first_date = min(t["date"] for t in trades)
    last_date = max(t["date"] for t in trades)
    calendar_days = (last_date - first_date).days
    trading_days = len(unique_dates)

    print(f"\n  {label}")
    print(f"  {'─'*75}")
    print(f"  Trades: {n}  |  Wins: {len(wins)}  Losses: {len(losses)}  |  "
          f"WR: {wr:.1f}%  |  PF: {pf:.2f}")
    print(f"  PnL: {total_pnl:+,.1f} pts  |  Max DD: {max_dd:.1f} pts  |  "
          f"Avg Risk: {avg_risk:.1f} pts")
    print(f"  Avg Win: {avg_w:+.1f}  |  Avg Loss: {avg_l:+.1f}")
    print(f"  Period: {first_date} to {last_date} ({calendar_days} days, {trading_days} active)")

    # Per-window
    print(f"\n  {'Window':<15} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PF':>7} {'PnL':>10} {'AvgWin':>8} {'AvgLoss':>8}")
    print("  " + "-" * 70)
    for w in ["Morning", "Afternoon", "Late Night", "Pre-RTH"]:
        wt = [t for t in trades if t["window"] == w]
        if not wt:
            print(f"  {w:<15} {0:>7}")
            continue
        ww = [t for t in wt if t["won"]]
        wl = [t for t in wt if not t["won"]]
        wpnl = sum(t["pnl_pts"] for t in wt)
        wwr = len(ww) / len(wt) * 100
        wgp = sum(t["pnl_pts"] for t in ww)
        wgl = abs(sum(t["pnl_pts"] for t in wl))
        wpf = wgp / wgl if wgl > 0 else float("inf")
        wavg_w = np.mean([t["pnl_pts"] for t in ww]) if ww else 0
        wavg_l = np.mean([t["pnl_pts"] for t in wl]) if wl else 0
        print(f"  {w:<15} {len(wt):>7} {len(ww):>5} {wwr:>5.1f}% {wpf:>7.2f} "
              f"{wpnl:>+10.1f} {wavg_w:>+8.1f} {wavg_l:>+8.1f}")

    # Per-pattern
    print(f"\n  {'Pattern':<20} {'Trades':>7} {'WR%':>6} {'PF':>7} {'PnL':>10}")
    print("  " + "-" * 55)
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
        print(f"  {short:<20} {len(pt):>7} {pwr:>5.1f}% {ppf:>7.2f} {ppnl:>+10.1f}")


def print_monthly(trades):
    """Monthly PnL breakdown."""
    if not trades:
        return

    print(f"\n  {'Month':<10} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PnL':>10} {'CumPnL':>10}")
    print("  " + "-" * 55)

    cum_pnl = 0.0
    # Group by year-month
    monthly = {}
    for t in trades:
        key = (t["date"].year, t["date"].month)
        if key not in monthly:
            monthly[key] = []
        monthly[key].append(t)

    for (year, month) in sorted(monthly.keys()):
        mt = monthly[(year, month)]
        mw = [t for t in mt if t["won"]]
        mpnl = sum(t["pnl_pts"] for t in mt)
        cum_pnl += mpnl
        mwr = len(mw) / len(mt) * 100 if mt else 0
        print(f"  {year}-{month:02d}   {len(mt):>7} {len(mw):>5} {mwr:>5.1f}% "
              f"{mpnl:>+10.1f} {cum_pnl:>+10.1f}")


def print_yearly(trades):
    """Yearly breakdown."""
    if not trades:
        return

    print(f"\n  {'Year':<6} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PF':>7} {'PnL':>10} {'MaxDD':>8}")
    print("  " + "-" * 55)

    for year in sorted(set(t["date"].year for t in trades)):
        yt = [t for t in trades if t["date"].year == year]
        yw = [t for t in yt if t["won"]]
        yl = [t for t in yt if not t["won"]]
        ypnl = sum(t["pnl_pts"] for t in yt)
        ywr = len(yw) / len(yt) * 100
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


def main():
    print("Loading 5-year full session data...")
    df = load_data()
    sessions = build_sessions(df)
    print(f"Built {len(sessions)} sessions ({df.index[0].date()} to {df.index[-1].date()})")

    print("\nRunning backtest...")
    all_trades = run_backtest(sessions)

    # ── Full period ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FULL PERIOD (5 years)")
    print(f"{'='*80}")
    print_stats(all_trades, "ALL TRADES")
    print_yearly(all_trades)
    print_monthly(all_trades)

    # ── True OOS: 2021-2024 (params NEVER tuned on this data) ──────
    oos_trades = [t for t in all_trades if t["date"].year < 2024 or
                  (t["date"].year == 2024 and t["date"].month < 2)]
    is_trades = [t for t in all_trades if t not in oos_trades]

    print(f"\n{'='*80}")
    print("TRUE OUT-OF-SAMPLE: Jan 2021 - Jan 2024")
    print("(Parameters were NEVER tuned on this data)")
    print(f"{'='*80}")
    print_stats(oos_trades, "OOS (2021-2024)")

    print(f"\n{'='*80}")
    print("IN-SAMPLE: Feb 2024 - Feb 2026")
    print("(Parameters were tuned on this period)")
    print(f"{'='*80}")
    print_stats(is_trades, "IS (2024-2026)")

    # ── Dollar PnL for MES ─────────────────────────────────────────
    total_pts = sum(t["pnl_pts"] for t in all_trades)
    oos_pts = sum(t["pnl_pts"] for t in oos_trades)
    is_pts = sum(t["pnl_pts"] for t in is_trades)

    print(f"\n{'='*80}")
    print("DOLLAR PnL (1 MES contract @ $5/pt)")
    print(f"{'='*80}")
    print(f"  Full period:  {total_pts:+,.1f} pts = ${total_pts * 5:+,.0f}")
    print(f"  OOS (21-24):  {oos_pts:+,.1f} pts = ${oos_pts * 5:+,.0f}")
    print(f"  IS  (24-26):  {is_pts:+,.1f} pts = ${is_pts * 5:+,.0f}")
    print(f"\n  Full period (1 ES @ $50/pt): {total_pts:+,.1f} pts = ${total_pts * 50:+,.0f}")


if __name__ == "__main__":
    main()
