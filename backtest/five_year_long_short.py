"""5-year long+short backtest: Jan 2021 - Feb 2026.

Enables Failed Rally (FR) and Level Rejection (LJ) short-side patterns
alongside existing Failed Breakdown (FB) and Level Reclaim (LR) longs.

Supports optional Mancini Substack level overlay via --mancini-levels-dir:
    python3 backtest/five_year_long_short.py --mancini-levels-dir data/mancini_levels

Usage:
    python3 backtest/five_year_long_short.py
    python3 backtest/five_year_long_short.py --config production
    python3 backtest/five_year_long_short.py --config production --mancini-levels-dir data/mancini_levels
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time as dt_time, timedelta, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

logger.remove()

from config.levels import Level, LevelType, LevelStore
from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    RiskParams, SessionTimes, ESContractSpec,
)
from core.level_scoring import LevelQualityScorer
from strategy.mancini_long import ManciniLongStrategy


# ── Config: production longs + enabled shorts ────────────────────────

STRATEGY = StrategyParams(
    # Long-side (production)
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
    # Short-side (ENABLED)
    allow_short_fr=True,
    allow_short_lj=True,
    fr_stop_buffer_pts=5.5,   # mirror FB stop buffer
    lj_stop_buffer_pts=5.0,   # mirror LR stop buffer
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


# ---------------------------------------------------------------------------
# Mancini level overlay helpers
# ---------------------------------------------------------------------------

def load_mancini_levels_for_date(
    levels_dir: Path, session_date: date
) -> dict | None:
    """Load parsed Mancini levels JSON for a given trading date.

    Returns the parsed dict or None if no file exists for that date.
    """
    path = levels_dir / f"mancini_levels_{session_date}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _build_mancini_level_objects(
    mancini_data: dict,
    session_date: date,
) -> list[Level]:
    """Build Level objects from parsed Mancini data for injection into the LevelStore.

    Mancini support levels are injected as CUSTOM type so the engine can
    sweep them just like engine-derived levels.
    """
    levels_out: list[Level] = []
    levels_list = mancini_data.get("levels", [])
    ts = pd.Timestamp(datetime.combine(session_date, dt_time(9, 30)), tz="US/Eastern")  # confirmed at RTH open

    for lv_data in levels_list:
        price = float(lv_data.get("price", 0))
        if price <= 0:
            continue
        role = lv_data.get("role", "level")

        level = Level(
            price=price,
            level_type=LevelType.CUSTOM,
            created_at=ts,
            confirmed_at=ts,
            touch_count=3,  # give Mancini levels moderate credibility
            rally_from_low_pts=25.0 if role in ("support", "level") else 0.0,
            is_active=True,
            label=f"MANCINI_{role.upper()}@{price:.2f}",
            origin_date=session_date,
            significance_score=1.0,
            tested_and_held=True,
        )
        levels_out.append(level)

    return levels_out


def classify_trade_mancini(
    trade: dict, mancini_data: dict | None, tolerance_pts: float = 3.0
) -> str:
    """Classify a trade's relationship to Mancini levels.

    Returns one of:
      "mancini_confirmed" - trade at a level that matches both engine AND Mancini
      "mancini_only"      - trade at a CUSTOM/MANCINI level (Mancini-sourced, not engine)
      "engine_only"       - no Mancini data for this session or level not in Mancini's list
    """
    if mancini_data is None:
        return "engine_only"

    entry_price = trade.get("entry", 0)
    level_type = trade.get("level_type", "")

    # Get all Mancini prices for this date
    mancini_prices = [float(lv.get("price", 0)) for lv in mancini_data.get("levels", [])]

    # Check if the trade's entry is near any Mancini level
    near_mancini = any(abs(entry_price - mp) <= tolerance_pts for mp in mancini_prices)

    if level_type == "CUSTOM" or "MANCINI" in level_type:
        return "mancini_only"
    elif near_mancini:
        return "mancini_confirmed"
    else:
        return "engine_only"


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


def run_backtest(sessions, mancini_levels_dir: Path | None = None):
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
    sessions_with_mancini = 0
    sessions_total = 0

    for session_date, session_df in sessions:
        if session_date.weekday() == 0:  # skip Mondays
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

        # Pre-load Mancini levels so they're available during bar processing
        mancini_data = None
        if mancini_levels_dir is not None:
            mancini_data = load_mancini_levels_for_date(mancini_levels_dir, session_date)
            if mancini_data is not None:
                # Build Level objects and set on strategy for injection during run_day
                extra_levels = _build_mancini_level_objects(mancini_data, session_date)
                if extra_levels:
                    strategy._extra_levels = extra_levels
                    sessions_with_mancini += 1

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
            # Afternoon: only FB (long) and FR (short) patterns
            if window == "Afternoon" and t.pattern_type not in ("failed_breakdown", "failed_rally"):
                continue

            direction = "SHORT" if t.pattern_type in ("failed_rally", "level_rejection") else "LONG"

            trade_dict = {
                "date": session_date,
                "entry_time": entry_ts,
                "time": et,
                "window": window,
                "pattern": t.pattern_type,
                "direction": direction,
                "level_type": t.level_type,
                "entry": t.entry_price,
                "stop": t.stop_price,
                "target_1": t.target_1,
                "pnl_pts": t.pnl_pts,
                "risk_pts": t.risk_pts,
                "rr_ratio": t.rr_ratio_t1,
                "exit_reason": t.exit_reason,
                "won": t.pnl_pts > 0,
                "is_double_dip": getattr(t, 'is_double_dip', False),
                "lqs": getattr(t, 'lqs', 0),
            }

            # Classify trade's relationship to Mancini levels
            trade_dict["mancini_class"] = classify_trade_mancini(
                trade_dict, mancini_data
            )

            all_trades.append(trade_dict)

        prev_session_df = session_df

    return all_trades, sessions_with_mancini, sessions_total


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

    unique_dates = set(t["date"] for t in trades)
    first_date = min(t["date"] for t in trades)
    last_date = max(t["date"] for t in trades)

    print(f"\n  {label}")
    print(f"  {'─'*75}")
    print(f"  Trades: {n}  |  Wins: {len(wins)}  Losses: {len(losses)}  |  "
          f"WR: {wr:.1f}%  |  PF: {pf:.2f}")
    print(f"  PnL: {total_pnl:+,.1f} pts  |  Max DD: {max_dd:.1f} pts  |  "
          f"Avg Risk: {avg_risk:.1f} pts")
    print(f"  Avg Win: {avg_w:+.1f}  |  Avg Loss: {avg_l:+.1f}")
    print(f"  Period: {first_date} to {last_date} ({len(unique_dates)} active days)")

    # Per direction
    for d in ["LONG", "SHORT"]:
        dt = [t for t in trades if t["direction"] == d]
        if not dt:
            continue
        dw = [t for t in dt if t["won"]]
        dl = [t for t in dt if not t["won"]]
        dpnl = sum(t["pnl_pts"] for t in dt)
        dwr = len(dw) / len(dt) * 100
        dgp = sum(t["pnl_pts"] for t in dw)
        dgl = abs(sum(t["pnl_pts"] for t in dl))
        dpf = dgp / dgl if dgl > 0 else float("inf")
        print(f"    {d}: {len(dt)}T, {dwr:.1f}% WR, PF={dpf:.2f}, {dpnl:+,.1f} pts")

    # Per pattern
    print(f"\n  {'Pattern':<20} {'Dir':<6} {'Trades':>7} {'WR%':>6} {'PF':>7} {'PnL':>10}")
    print("  " + "-" * 60)
    for p in ["failed_breakdown", "level_reclaim", "failed_rally", "level_rejection"]:
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
        short_name = {"failed_breakdown": "FB", "level_reclaim": "LR",
                      "failed_rally": "FR", "level_rejection": "LJ"}[p]
        d = "SHORT" if p in ("failed_rally", "level_rejection") else "LONG"
        print(f"  {short_name:<20} {d:<6} {len(pt):>7} {pwr:>5.1f}% {ppf:>7.2f} {ppnl:>+10.1f}")

    # Per window
    print(f"\n  {'Window':<15} {'Trades':>7} {'WR%':>6} {'PF':>7} {'PnL':>10} {'L/S':>10}")
    print("  " + "-" * 60)
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
        n_long = sum(1 for t in wt if t["direction"] == "LONG")
        n_short = sum(1 for t in wt if t["direction"] == "SHORT")
        print(f"  {w:<15} {len(wt):>7} {wwr:>5.1f}% {wpf:>7.2f} {wpnl:>+10.1f} {n_long}L/{n_short}S")

    # LQS breakdown (only if any trades have LQS > 0)
    lqs_trades = [t for t in trades if t.get("lqs", 0) > 0]
    if lqs_trades:
        print(f"\n  LEVEL QUALITY SCORE BREAKDOWN:")
        print(f"  {'LQS Range':<15} {'Trades':>7} {'WR%':>6} {'PF':>7} {'PnL':>10}")
        print("  " + "-" * 50)
        for lqs_label, lqs_lo, lqs_hi in [
            ("70-100", 70, 101),
            ("50-69", 50, 70),
            ("30-49", 30, 50),
            ("0-29", 0, 30),
        ]:
            lt = [t for t in trades if lqs_lo <= t.get("lqs", 0) < lqs_hi]
            if not lt:
                print(f"  {lqs_label:<15} {0:>7}")
                continue
            lw = [t for t in lt if t["won"]]
            ll = [t for t in lt if not t["won"]]
            lpnl = sum(t["pnl_pts"] for t in lt)
            lwr = len(lw) / len(lt) * 100
            lgp = sum(t["pnl_pts"] for t in lw)
            lgl = abs(sum(t["pnl_pts"] for t in ll))
            lpf = lgp / lgl if lgl > 0 else float("inf")
            print(f"  {lqs_label:<15} {len(lt):>7} {lwr:>5.1f}% {lpf:>7.2f} {lpnl:>+10.1f}")


def print_mancini_overlay_stats(
    trades: list[dict],
    sessions_with_mancini: int,
    sessions_total: int,
) -> None:
    """Print breakdown of trades by Mancini overlay classification."""
    print(f"\n{'='*80}")
    print("MANCINI OVERLAY IMPACT")
    print(f"{'='*80}")
    print(f"  Sessions with Mancini levels: {sessions_with_mancini} / {sessions_total} total")

    for cls_label, cls_key in [
        ("Mancini-confirmed levels", "mancini_confirmed"),
        ("Mancini-only levels", "mancini_only"),
        ("Engine-only levels", "engine_only"),
    ]:
        subset = [t for t in trades if t.get("mancini_class") == cls_key]
        n = len(subset)
        if n == 0:
            print(f"  Trades at {cls_label}: 0")
            continue
        wins = [t for t in subset if t["won"]]
        losses = [t for t in subset if not t["won"]]
        pnl = sum(t["pnl_pts"] for t in subset)
        wr = len(wins) / n * 100
        gp = sum(t["pnl_pts"] for t in wins)
        gl = abs(sum(t["pnl_pts"] for t in losses))
        pf = gp / gl if gl > 0 else float("inf")
        print(f"  Trades at {cls_label}: {n}  (WR: {wr:.1f}%, PF: {pf:.2f}, PnL: {pnl:+,.1f} pts)")


def main():
    parser = argparse.ArgumentParser(description="5-year long+short backtest")
    parser.add_argument(
        "--config", default="default",
        help="Config profile: 'default' or 'production'"
    )
    parser.add_argument(
        "--mancini-levels-dir", type=str, default=None,
        help="Path to directory of mancini_levels_YYYY-MM-DD.json files"
    )
    args = parser.parse_args()

    mancini_dir = Path(args.mancini_levels_dir) if args.mancini_levels_dir else None
    if mancini_dir and not mancini_dir.exists():
        print(f"WARNING: Mancini levels dir does not exist: {mancini_dir}")
        print("  Running without Mancini overlay.")
        mancini_dir = None

    print("Loading 5-year full session data...")
    df = load_data()
    sessions = build_sessions(df)
    print(f"Built {len(sessions)} sessions ({df.index[0].date()} to {df.index[-1].date()})")

    mode = "LONG + SHORT"
    if mancini_dir:
        # Count available level files
        n_files = len(list(mancini_dir.glob("mancini_levels_*.json")))
        mode += f" + MANCINI OVERLAY ({n_files} level files)"

    print(f"\nRunning {mode} backtest...")
    all_trades, sessions_with_mancini, sessions_total = run_backtest(
        sessions, mancini_levels_dir=mancini_dir
    )

    # ── Full period ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FULL PERIOD: 5 Years Long + Short")
    print(f"{'='*80}")
    print_stats(all_trades, "ALL TRADES (Jan 2021 - Feb 2026)")

    # ── Yearly breakdown ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("YEARLY BREAKDOWN")
    print(f"{'='*80}")
    print(f"\n  {'Year':<6} {'Total':>7} {'Long':>6} {'Short':>6} {'WR%':>6} {'PF':>7} {'PnL':>10} {'MaxDD':>8}")
    print("  " + "-" * 65)

    for year in sorted(set(t["date"].year for t in all_trades)):
        yt = [t for t in all_trades if t["date"].year == year]
        yw = [t for t in yt if t["won"]]
        yl = [t for t in yt if not t["won"]]
        ypnl = sum(t["pnl_pts"] for t in yt)
        ywr = len(yw) / len(yt) * 100
        ygp = sum(t["pnl_pts"] for t in yw)
        ygl = abs(sum(t["pnl_pts"] for t in yl))
        ypf = ygp / ygl if ygl > 0 else float("inf")
        n_long = sum(1 for t in yt if t["direction"] == "LONG")
        n_short = sum(1 for t in yt if t["direction"] == "SHORT")

        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in yt:
            cum += t["pnl_pts"]
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        print(f"  {year:<6} {len(yt):>7} {n_long:>6} {n_short:>6} {ywr:>5.1f}% {ypf:>7.2f} "
              f"{ypnl:>+10.1f} {max_dd:>8.1f}")

    # ── Per-year per-direction ──────────────────────────────────────
    print(f"\n{'='*80}")
    print("PER-YEAR DIRECTION BREAKDOWN")
    print(f"{'='*80}")
    print(f"\n  {'Year':<6} {'Dir':<6} {'Trades':>7} {'WR%':>6} {'PF':>7} {'PnL':>10}")
    print("  " + "-" * 50)

    for year in sorted(set(t["date"].year for t in all_trades)):
        for d in ["LONG", "SHORT"]:
            dt = [t for t in all_trades if t["date"].year == year and t["direction"] == d]
            if not dt:
                continue
            dw = [t for t in dt if t["won"]]
            dpnl = sum(t["pnl_pts"] for t in dt)
            dwr = len(dw) / len(dt) * 100
            dgp = sum(t["pnl_pts"] for t in dw)
            dgl = abs(sum(t["pnl_pts"] for t in dt if not t["won"]))
            dpf = dgp / dgl if dgl > 0 else float("inf")
            print(f"  {year:<6} {d:<6} {len(dt):>7} {dwr:>5.1f}% {dpf:>7.2f} {dpnl:>+10.1f}")

    # ── OOS vs IS ───────────────────────────────────────────────────
    oos_trades = [t for t in all_trades if t["date"].year < 2024 or
                  (t["date"].year == 2024 and t["date"].month < 2)]
    is_trades = [t for t in all_trades if t not in oos_trades]

    print(f"\n{'='*80}")
    print("TRUE OUT-OF-SAMPLE: Jan 2021 - Jan 2024")
    print(f"{'='*80}")
    print_stats(oos_trades, "OOS (2021-2024)")

    print(f"\n{'='*80}")
    print("IN-SAMPLE: Feb 2024 - Feb 2026")
    print(f"{'='*80}")
    print_stats(is_trades, "IS (2024-2026)")

    # ── Dollar PnL ─────────────────────────────────────────────────
    total_pts = sum(t["pnl_pts"] for t in all_trades)
    oos_pts = sum(t["pnl_pts"] for t in oos_trades)
    is_pts = sum(t["pnl_pts"] for t in is_trades)

    print(f"\n{'='*80}")
    print("DOLLAR PnL")
    print(f"{'='*80}")
    print(f"  Full period (MES): {total_pts:+,.1f} pts = ${total_pts * 5:+,.0f}")
    print(f"  OOS  (MES):        {oos_pts:+,.1f} pts = ${oos_pts * 5:+,.0f}")
    print(f"  IS   (MES):        {is_pts:+,.1f} pts = ${is_pts * 5:+,.0f}")
    print(f"  Full period (ES):  {total_pts:+,.1f} pts = ${total_pts * 50:+,.0f}")

    # ── Double Dip Breakdown ──────────────────────────────────────
    dd_trades = [t for t in all_trades if t["is_double_dip"]]
    non_dd_trades = [t for t in all_trades if not t["is_double_dip"]]

    print(f"\n{'='*80}")
    print("DOUBLE DIP BREAKDOWN")
    print(f"{'='*80}")

    for label, subset in [("DD Trades", dd_trades), ("Non-DD   ", non_dd_trades)]:
        n = len(subset)
        if n == 0:
            print(f"  {label}: 0 trades")
            continue
        wins = [t for t in subset if t["won"]]
        losses = [t for t in subset if not t["won"]]
        pnl = sum(t["pnl_pts"] for t in subset)
        wr = len(wins) / n * 100
        gp = sum(t["pnl_pts"] for t in wins)
        gl = abs(sum(t["pnl_pts"] for t in losses))
        pf = gp / gl if gl > 0 else float("inf")
        print(f"  {label}: {n:>4}  |  WR: {wr:>5.1f}%  |  PF: {pf:.2f}  |  PnL: {pnl:>+,.1f} pts")

    if dd_trades:
        print(f"\n  DD by level type:")
        level_types = sorted(set(t["level_type"] for t in dd_trades))
        for lt in level_types:
            lt_trades = [t for t in dd_trades if t["level_type"] == lt]
            lt_wins = [t for t in lt_trades if t["won"]]
            lt_losses = [t for t in lt_trades if not t["won"]]
            lt_pnl = sum(t["pnl_pts"] for t in lt_trades)
            lt_wr = len(lt_wins) / len(lt_trades) * 100 if lt_trades else 0
            print(f"    {lt:<25} {len(lt_wins)}W/{len(lt_losses)}L  {lt_wr:>5.1f}% WR  {lt_pnl:>+,.1f} pts")

        print(f"\n  DD by pattern:")
        dd_patterns = sorted(set(t["pattern"] for t in dd_trades))
        for p in dd_patterns:
            pt = [t for t in dd_trades if t["pattern"] == p]
            pw = [t for t in pt if t["won"]]
            pl = [t for t in pt if not t["won"]]
            ppnl = sum(t["pnl_pts"] for t in pt)
            pwr = len(pw) / len(pt) * 100 if pt else 0
            short_name = {"failed_breakdown": "FB", "level_reclaim": "LR",
                          "failed_rally": "FR", "level_rejection": "LJ"}.get(p, p)
            print(f"    {short_name:<25} {len(pw)}W/{len(pl)}L  {pwr:>5.1f}% WR  {ppnl:>+,.1f} pts")

    # ── Mancini Overlay Impact ────────────────────────────────────
    if mancini_dir is not None:
        print_mancini_overlay_stats(all_trades, sessions_with_mancini, sessions_total)

        # Also break down by year for the Mancini period
        mancini_trades = [t for t in all_trades if t.get("mancini_class") != "engine_only"]
        if mancini_trades:
            print(f"\n  Mancini-period yearly breakdown:")
            print(f"  {'Year':<6} {'Confirmed':>10} {'M-Only':>8} {'EngOnly':>8} {'Total':>7}")
            print("  " + "-" * 50)
            for year in sorted(set(t["date"].year for t in all_trades)):
                yt = [t for t in all_trades if t["date"].year == year]
                n_conf = sum(1 for t in yt if t.get("mancini_class") == "mancini_confirmed")
                n_monly = sum(1 for t in yt if t.get("mancini_class") == "mancini_only")
                n_eonly = sum(1 for t in yt if t.get("mancini_class") == "engine_only")
                print(f"  {year:<6} {n_conf:>10} {n_monly:>8} {n_eonly:>8} {len(yt):>7}")


if __name__ == "__main__":
    main()
