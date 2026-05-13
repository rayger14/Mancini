"""Optuna optimization for data-driven gates (session range + BD entry cap).

Walk-forward validation:
  Train: 2021-01 to 2024-12 (~4 years)
  Test:  2025-01 to 2026-02 (~13 months, most recent OOS)

Optimizes only the 2 gate params while keeping all other params fixed
at production Optuna v2 values. This avoids overfitting a large param
space and focuses on tuning what the live data told us to add.

Objective: maximize Sharpe ratio on train set (balances returns vs consistency).
Validation: PF > 1.0 and positive PnL on OOS test set.

Usage:
    python3 -u backtest/optuna_gates.py [--trials 50] [--timeout 120]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loguru import logger
logger.remove()

import time
import json
import numpy as np
from datetime import datetime, date, time as dt_time, timedelta
from pathlib import Path
from dataclasses import replace

import optuna
import pandas as pd

from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    RiskParams, SessionTimes, ESContractSpec, DEFAULT_STRATEGY,
)
from core.regime_filter import RegimeParams, build_daily_bars
from strategy.mancini_long import ManciniLongStrategy

# ── Constants ──────────────────────────────────────────────────────────
DATA_PATH = Path("data/ES_1m_full_session_2021-01-01_2026-02-05.parquet")
RESULTS_PATH = Path("data/optuna_gates_results.json")

# Production Optuna v2 params (fixed — not re-optimized)
ELEVATOR = ElevatorParams(min_velocity_pts_per_min=0.75, min_levels_broken=2, higher_low_lookback=4)
FULL_SESSION = SessionTimes(
    rth_open=dt_time(18, 0), rth_close=dt_time(17, 0),
    morning_window_start=dt_time(9, 30), morning_window_end=dt_time(11, 0),
    afternoon_window_start=dt_time(15, 0), afternoon_window_end=dt_time(16, 50),
    eod_flatten_time=dt_time(16, 50),
    chop_zone_start=dt_time(13, 0), chop_zone_end=dt_time(15, 0),
)
MES = ESContractSpec(symbol="MES", tick_size=0.25, tick_value=1.25,
    point_value=5.0, margin_initial=1_265.0, margin_maintenance=1_150.0, exchange="CME")

# Fixed production params
BASE_STRATEGY = replace(DEFAULT_STRATEGY,
    acceptance_max_dip_pts=15.0,
    acceptance_min_hold_bars=11,
    fb_stop_buffer_pts=6.0,
    lr_stop_buffer_pts=4.0,
    max_fb_sweep_depth_pts=20.0,
    max_target_distance_pts=30.0,
    allow_breakdown_short=True,
    allow_backtest_short=False,
    bd_confirm_bars=21,
    bd_stop_buffer_pts=6.0,
    bd_max_break_depth_pts=17.0,
    bd_timeout_bars=35,
    min_signal_rr=0.8,
    signal_cooldown_bars=15,
    use_regime_filter=True,
)
BASE_EXIT = ExitParams(
    default_contracts=4, t1_exit_fraction=1.0,
    trailing_stop_pts=7.0, fb_max_hold_bars=14,
)
BASE_RISK = RiskParams(
    max_trades_per_day=4,
    max_stop_distance_pts=20.0,
    skip_tuesdays=False,
    min_rr_ratio=0.8,
)
BASE_REGIME = RegimeParams(
    mode="ema", ema_span=30, slope_lookback=10, slope_threshold_atr_mult=0.325,
)

# Walk-forward dates
TRAIN_START = date(2021, 1, 1)
TRAIN_END = date(2024, 12, 31)
TEST_START = date(2025, 1, 1)
TEST_END = date(2026, 2, 5)


# ── Pre-load data ──────────────────────────────────────────────────────
print("Loading data...", flush=True)
t0 = time.time()
DF = pd.read_parquet(DATA_PATH)
if DF.index.tz is None:
    DF.index = DF.index.tz_localize("US/Eastern")
DAILY_BARS = build_daily_bars(DF)

evening_mask = DF.index.time == dt_time(18, 0)
session_starts = DF.index[evening_mask]
ALL_SESSIONS = []
for start_ts in session_starts:
    next_day = start_ts.date() + timedelta(days=1)
    end_ts = pd.Timestamp(datetime.combine(next_day, dt_time(16, 59)), tz="US/Eastern")
    session_df = DF[(DF.index >= start_ts) & (DF.index <= end_ts)]
    session_df = session_df[~((session_df.index.time >= dt_time(17, 0)) & (session_df.index.time < dt_time(18, 0)))]
    if len(session_df) > 0:
        ALL_SESSIONS.append((next_day, session_df))

TRAIN_SESSIONS = [(d, df) for d, df in ALL_SESSIONS if TRAIN_START <= d <= TRAIN_END]
TEST_SESSIONS = [(d, df) for d, df in ALL_SESSIONS if TEST_START <= d <= TEST_END]

print(f"  {len(ALL_SESSIONS)} total sessions", flush=True)
print(f"  Train: {len(TRAIN_SESSIONS)} sessions ({TRAIN_START} to {TRAIN_END})", flush=True)
print(f"  Test:  {len(TEST_SESSIONS)} sessions ({TEST_START} to {TEST_END})", flush=True)
print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)


def run_sessions(sessions, strategy_params, min_rr=0.8):
    """Run backtest on a set of sessions. Returns list of trade dicts."""
    trades = []
    prev_rth = None

    for session_date, session_df in sessions:
        daily_history = DAILY_BARS[DAILY_BARS.index.date < session_date]

        strategy = ManciniLongStrategy(
            strategy_params=strategy_params, elevator_params=ELEVATOR,
            exit_params=BASE_EXIT, risk_params=BASE_RISK,
            session_times=FULL_SESSION, contract=MES,
            min_rr_ratio=min_rr, rth_filter=(dt_time(9, 30), dt_time(16, 0)),
            regime_params=BASE_REGIME, daily_history=daily_history,
        )
        prior = prev_rth if prev_rth is not None and len(prev_rth) > 0 else None
        strategy.run_day(session_df, prior_day_df=prior,
                        session_date=datetime.combine(session_date, dt_time(0, 0)))

        for t in strategy.trade_records:
            if t.entry_bar_idx >= len(session_df):
                continue
            entry_ts = session_df.index[t.entry_bar_idx]
            et = entry_ts.time()
            # Time window filter (same as production)
            if dt_time(9, 30) <= et < dt_time(13, 0):
                window = "Morning"
            elif dt_time(15, 0) <= et <= dt_time(16, 50):
                window = "Afternoon"
            elif et >= dt_time(22, 0) or et < dt_time(2, 0):
                window = "Late Night"
            elif dt_time(6, 0) <= et < dt_time(9, 30):
                window = "Pre-RTH"
            else:
                continue  # blocked window

            direction = getattr(t, 'direction', 'long')
            if window == "Pre-RTH" and direction == "short":
                continue
            if window == "Afternoon" and t.pattern_type not in (
                "failed_breakdown", "failed_rally", "breakdown_short"):
                continue

            trades.append({
                "date": session_date,
                "year": session_date.year,
                "pnl_pts": t.pnl_pts,
                "won": t.pnl_pts > 0,
                "direction": direction,
                "pattern": t.pattern_type,
            })

        prev_rth = session_df.between_time("09:30", "15:59")

    return trades


def compute_metrics(trades):
    """Compute summary metrics."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "total_pnl": -9999, "pf": 0, "wr": 0, "sharpe": -10}

    total_pnl = sum(t["pnl_pts"] for t in trades)
    wins = sum(1 for t in trades if t["won"])
    wr = wins / n * 100
    gw = sum(t["pnl_pts"] for t in trades if t["won"])
    gl = abs(sum(t["pnl_pts"] for t in trades if not t["won"]))
    pf = gw / gl if gl > 0 else 0

    daily_pnl = {}
    for t in trades:
        d = t["date"]
        daily_pnl[d] = daily_pnl.get(d, 0) + t["pnl_pts"]
    pnl_arr = np.array(list(daily_pnl.values()))
    if len(pnl_arr) > 1 and np.std(pnl_arr, ddof=1) > 0.01:
        sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    year_pnls = {}
    for year in sorted(set(t["year"] for t in trades)):
        year_pnls[year] = round(sum(t["pnl_pts"] for t in trades if t["year"] == year), 1)

    return {
        "n": n, "total_pnl": round(total_pnl, 1), "pf": round(pf, 2),
        "wr": round(wr, 1), "sharpe": round(sharpe, 2), "year_pnls": year_pnls,
    }


def objective(trial):
    """Optuna objective: maximize train Sharpe with gate params."""
    # Gate params to optimize
    min_session_range = trial.suggest_float("min_session_range_pts", 0.0, 30.0, step=2.5)
    grace_bars = trial.suggest_int("min_session_range_grace_bars", 10, 60, step=5)
    bd_entry_cap = trial.suggest_float("bd_max_entry_distance_pts", 0.0, 20.0, step=1.0)

    strategy_params = replace(BASE_STRATEGY,
        min_session_range_pts=min_session_range,
        min_session_range_grace_bars=grace_bars,
        bd_max_entry_distance_pts=bd_entry_cap,
        # Keep other new gates disabled for this optimization
        max_trades_per_level=0,
        bd_short_min_rr=0.0,
        cross_type_level_cooldown_bars=0,
    )

    trades = run_sessions(TRAIN_SESSIONS, strategy_params)
    m = compute_metrics(trades)

    # Prune clearly bad trials early
    if m["n"] < 50 or m["pf"] < 0.5:
        raise optuna.TrialPruned()

    trial.set_user_attr("train_n", m["n"])
    trial.set_user_attr("train_pf", m["pf"])
    trial.set_user_attr("train_pnl", m["total_pnl"])
    trial.set_user_attr("train_wr", m["wr"])

    return m["sharpe"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120, help="timeout in minutes")
    args = parser.parse_args()

    print(f"\n{'='*60}", flush=True)
    print(f"  OPTUNA GATE OPTIMIZATION", flush=True)
    print(f"  Params: min_session_range_pts, grace_bars, bd_max_entry_distance_pts", flush=True)
    print(f"  Objective: maximize Sharpe on train ({TRAIN_START} → {TRAIN_END})", flush=True)
    print(f"  Validation: OOS test ({TEST_START} → {TEST_END})", flush=True)
    print(f"  Trials: {args.trials}, Timeout: {args.timeout}m", flush=True)
    print(f"{'='*60}\n", flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Seed with baseline (no gates) and our initial guess
    study.enqueue_trial({
        "min_session_range_pts": 0.0,
        "min_session_range_grace_bars": 30,
        "bd_max_entry_distance_pts": 0.0,
    })
    study.enqueue_trial({
        "min_session_range_pts": 15.0,
        "min_session_range_grace_bars": 30,
        "bd_max_entry_distance_pts": 10.0,
    })

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout * 60,
        show_progress_bar=True,
    )

    # ── Results ────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}", flush=True)
    print(f"  BEST TRIAL: #{best.number}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Params:", flush=True)
    for k, v in best.params.items():
        print(f"    {k}: {v}", flush=True)
    print(f"  Train Sharpe: {best.value:.2f}", flush=True)
    print(f"  Train PF: {best.user_attrs.get('train_pf', '?')}", flush=True)
    print(f"  Train PnL: {best.user_attrs.get('train_pnl', '?')}", flush=True)
    print(f"  Train WR: {best.user_attrs.get('train_wr', '?')}%", flush=True)

    # ── OOS validation ─────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  OUT-OF-SAMPLE VALIDATION ({TEST_START} → {TEST_END})", flush=True)
    print(f"{'='*60}", flush=True)

    # Baseline (no gates)
    baseline_params = replace(BASE_STRATEGY,
        min_session_range_pts=0.0,
        bd_max_entry_distance_pts=0.0,
        max_trades_per_level=0,
        bd_short_min_rr=0.0,
        cross_type_level_cooldown_bars=0,
    )
    baseline_trades = run_sessions(TEST_SESSIONS, baseline_params)
    bm = compute_metrics(baseline_trades)
    print(f"  Baseline: {bm['n']}T, PF={bm['pf']}, PnL={bm['total_pnl']:+.1f}, "
          f"WR={bm['wr']}%, Sharpe={bm['sharpe']}", flush=True)
    if bm.get("year_pnls"):
        for y, p in bm["year_pnls"].items():
            print(f"    {y}: {p:+.1f} pts", flush=True)

    # Best trial params on OOS
    best_params = replace(BASE_STRATEGY,
        min_session_range_pts=best.params["min_session_range_pts"],
        min_session_range_grace_bars=best.params["min_session_range_grace_bars"],
        bd_max_entry_distance_pts=best.params["bd_max_entry_distance_pts"],
        max_trades_per_level=0,
        bd_short_min_rr=0.0,
        cross_type_level_cooldown_bars=0,
    )
    best_trades = run_sessions(TEST_SESSIONS, best_params)
    bst = compute_metrics(best_trades)
    print(f"  Best:     {bst['n']}T, PF={bst['pf']}, PnL={bst['total_pnl']:+.1f}, "
          f"WR={bst['wr']}%, Sharpe={bst['sharpe']}", flush=True)
    if bst.get("year_pnls"):
        for y, p in bst["year_pnls"].items():
            print(f"    {y}: {p:+.1f} pts", flush=True)

    delta = bst["total_pnl"] - bm["total_pnl"]
    print(f"\n  OOS Delta: {delta:+.1f} pts", flush=True)
    if bst["pf"] > 1.0 and bst["total_pnl"] > 0:
        print(f"  ✓ OOS VALIDATED (PF > 1.0, positive PnL)", flush=True)
    else:
        print(f"  ✗ OOS FAILED — gates may not generalize", flush=True)

    # ── Top 5 trials ───────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  TOP 5 TRIALS", flush=True)
    print(f"{'='*60}", flush=True)
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)
    for t in sorted_trials[:5]:
        if t.value is None:
            continue
        p = t.params
        print(f"  #{t.number}: Sharpe={t.value:.2f}, PF={t.user_attrs.get('train_pf','?')}, "
              f"PnL={t.user_attrs.get('train_pnl','?')}, "
              f"range={p['min_session_range_pts']}, grace={p['min_session_range_grace_bars']}, "
              f"entry_cap={p['bd_max_entry_distance_pts']}", flush=True)

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "best_params": best.params,
        "best_train_sharpe": best.value,
        "best_train_metrics": {k: best.user_attrs.get(f"train_{k}") for k in ["n", "pf", "pnl", "wr"]},
        "oos_baseline": bm,
        "oos_best": bst,
        "oos_delta_pts": round(delta, 1),
        "top_5": [{
            "number": t.number,
            "params": t.params,
            "sharpe": t.value,
            "pf": t.user_attrs.get("train_pf"),
            "pnl": t.user_attrs.get("train_pnl"),
        } for t in sorted_trials[:5] if t.value is not None],
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Saved results to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
