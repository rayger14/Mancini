"""Feature comparison backtest: run baseline + each feature individually.

Uses PRODUCTION_STRATEGY from ib_runner.py as the base, toggling one feature
at a time. Produces a comparison table.

Usage:
    python3 backtest/feature_comparison.py
"""

from __future__ import annotations

import copy
import sys
from dataclasses import replace
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
from live.ib_runner import (
    PRODUCTION_STRATEGY, PRODUCTION_ELEVATOR, PRODUCTION_EXIT,
    PRODUCTION_RISK, PRODUCTION_SESSION, PRODUCTION_REGIME,
)
from core.regime_filter import RegimeParams
from strategy.mancini_long import ManciniLongStrategy
from strategy.exit_manager import ExitManager


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


def run_backtest(sessions, strategy_params, exit_params=None):
    if exit_params is None:
        exit_params = PRODUCTION_EXIT

    strategy = ManciniLongStrategy(
        strategy_params=strategy_params,
        elevator_params=PRODUCTION_ELEVATOR,
        exit_params=exit_params,
        risk_params=PRODUCTION_RISK,
        session_times=PRODUCTION_SESSION,
        contract=MES,
        min_rr_ratio=PRODUCTION_RISK.min_rr_ratio,
        rth_filter=(dt_time(9, 30), dt_time(16, 0)),
        regime_params=PRODUCTION_REGIME,
    )

    all_trades = []
    prev_session_df = None

    for session_date, session_df in sessions:
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

            direction = getattr(t, 'direction', 'long')
            if direction == 'long':
                direction_label = "LONG"
            else:
                direction_label = "SHORT"

            all_trades.append({
                "date": session_date,
                "entry_time": entry_ts,
                "time": et,
                "window": window,
                "pattern": t.pattern_type,
                "direction": direction_label,
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


def compute_stats(trades):
    n = len(trades)
    if n == 0:
        return {"trades": 0, "wr": 0, "pf": 0, "pnl": 0, "maxdd": 0}

    wins = [t for t in trades if t["won"]]
    losses = [t for t in trades if not t["won"]]
    total_pnl = sum(t["pnl_pts"] for t in trades)
    wr = len(wins) / n * 100
    gross_p = sum(t["pnl_pts"] for t in wins)
    gross_l = abs(sum(t["pnl_pts"] for t in losses))
    pf = gross_p / gross_l if gross_l > 0 else float("inf")

    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum += t["pnl_pts"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    return {
        "trades": n,
        "wr": wr,
        "pf": pf,
        "pnl": total_pnl,
        "maxdd": max_dd,
    }


# Feature configurations: (name, dict of overrides to enable the feature)
FEATURES = [
    ("baseline", {}),
    ("exit_scaling", {
        "mancini_exit_scaling": True,
        "mancini_t1_at_first_resistance": True,
    }),
    ("sweep_depth", {
        "use_sweep_depth_sizing": True,
    }),
    ("mode1", {
        "use_mode1_detection": True,
    }),
    ("confluence", {
        "use_confluence_scoring": True,
        "confluence_min_score": 3,
    }),
    ("atm", {
        "use_atm_level_boost": True,
    }),
    ("velocity", {
        "allow_velocity_short": True,
    }),
]


def make_params(overrides: dict) -> StrategyParams:
    """Create a StrategyParams from PRODUCTION_STRATEGY with overrides applied."""
    # Get all field values from PRODUCTION_STRATEGY
    from dataclasses import fields, asdict
    base = {}
    for f in fields(PRODUCTION_STRATEGY):
        base[f.name] = getattr(PRODUCTION_STRATEGY, f.name)
    base.update(overrides)
    return StrategyParams(**base)


def main():
    print("Loading 5-year full session data...")
    df = load_data()
    sessions = build_sessions(df)
    print(f"Built {len(sessions)} sessions ({df.index[0].date()} to {df.index[-1].date()})")

    results = []

    for name, overrides in FEATURES:
        print(f"\nRunning backtest: {name}...")
        params = make_params(overrides)
        trades = run_backtest(sessions, params)
        stats = compute_stats(trades)
        results.append((name, stats))
        print(f"  -> {stats['trades']}T, {stats['wr']:.1f}% WR, PF={stats['pf']:.2f}, "
              f"PnL={stats['pnl']:+,.1f}, MaxDD={stats['maxdd']:.1f}")

    # Print comparison table
    print(f"\n{'='*75}")
    print("FEATURE COMPARISON TABLE")
    print(f"{'='*75}")
    print(f"{'Feature':<16} {'Trades':>7} {'WR':>7} {'PF':>7} {'PnL':>12} {'MaxDD':>8}")
    print("-" * 75)
    for name, stats in results:
        print(f"{name:<16} {stats['trades']:>7} {stats['wr']:>6.1f}% {stats['pf']:>7.2f} "
              f"{stats['pnl']:>+11.1f} {stats['maxdd']:>8.1f}")

    # Delta from baseline
    if results:
        base = results[0][1]
        print(f"\n{'Feature':<16} {'dTrades':>8} {'dWR':>7} {'dPF':>7} {'dPnL':>12} {'dMaxDD':>8}")
        print("-" * 75)
        for name, stats in results[1:]:
            dt = stats['trades'] - base['trades']
            dwr = stats['wr'] - base['wr']
            dpf = stats['pf'] - base['pf']
            dpnl = stats['pnl'] - base['pnl']
            ddd = stats['maxdd'] - base['maxdd']
            print(f"{name:<16} {dt:>+8} {dwr:>+6.1f}% {dpf:>+7.2f} "
                  f"{dpnl:>+11.1f} {ddd:>+8.1f}")


if __name__ == "__main__":
    main()
