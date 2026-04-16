"""Optuna optimization for intraday price action context params.

Walk-forward validation:
  Train: 2021-01 to 2024-12 (~4 years)
  Test:  2025-01 to 2026-02 (~13 months OOS)

Optimizes the 10 intraday context params while keeping all other strategy
params fixed at production values. Session range gate (already validated)
is included as baseline.

Objective: maximize Sharpe ratio on train set.
Validation: PF > 1.0 and positive PnL on OOS test set.

Usage:
    python3 -u backtest/optuna_intraday.py [--trials 30] [--timeout 180]
"""
import sys
sys.path.insert(0, "/Users/raymondghandchi/Mancini/Mancini")
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
RESULTS_PATH = Path("data/optuna_intraday_results.json")

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

# Fixed production params (session range gate included)
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
    min_session_range_pts=15.0,
    min_session_range_grace_bars=30,
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
    """Run backtest on a set of sessions."""
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
            if dt_time(9, 30) <= et < dt_time(13, 0):
                window = "Morning"
            elif dt_time(15, 0) <= et <= dt_time(16, 50):
                window = "Afternoon"
            elif et >= dt_time(22, 0) or et < dt_time(2, 0):
                window = "Late Night"
            elif dt_time(6, 0) <= et < dt_time(9, 30):
                window = "Pre-RTH"
            else:
                continue

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
    """Optuna objective: maximize train Sharpe with intraday context params."""
    # Intraday context params
    swing_order = trial.suggest_int("idc_swing_order", 3, 15)
    min_swing_pts = trial.suggest_float("idc_min_swing_pts", 2.0, 12.0, step=1.0)
    weak_bounce_pts = trial.suggest_float("idc_weak_bounce_pts", 3.0, 15.0, step=1.0)
    bounce_lookback = trial.suggest_int("idc_bounce_lookback", 2, 5)
    elevator_recency = trial.suggest_int("idc_elevator_recency_bars", 15, 60, step=5)
    session_pos_bear = trial.suggest_float("idc_session_pos_bearish", 0.1, 0.35, step=0.05)
    session_pos_bull = trial.suggest_float("idc_session_pos_bullish", 0.65, 0.9, step=0.05)
    bearish_thresh = trial.suggest_int("idc_bearish_threshold", 2, 4)
    bullish_thresh = trial.suggest_int("idc_bullish_threshold", 2, 4)

    strategy_params = replace(BASE_STRATEGY,
        use_intraday_context=True,
        idc_swing_order=swing_order,
        idc_min_swing_pts=min_swing_pts,
        idc_weak_bounce_pts=weak_bounce_pts,
        idc_bounce_lookback=bounce_lookback,
        idc_elevator_recency_bars=elevator_recency,
        idc_session_pos_bearish=session_pos_bear,
        idc_session_pos_bullish=session_pos_bull,
        idc_bearish_threshold=bearish_thresh,
        idc_bullish_threshold=bullish_thresh,
    )

    trades = run_sessions(TRAIN_SESSIONS, strategy_params)
    m = compute_metrics(trades)

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
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=180, help="timeout in minutes")
    args = parser.parse_args()

    print(f"\n{'='*60}", flush=True)
    print(f"  OPTUNA INTRADAY CONTEXT OPTIMIZATION", flush=True)
    print(f"  Params: swing_order, min_swing_pts, weak_bounce, thresholds, etc.", flush=True)
    print(f"  Objective: maximize Sharpe on train ({TRAIN_START} → {TRAIN_END})", flush=True)
    print(f"  Validation: OOS test ({TEST_START} → {TEST_END})", flush=True)
    print(f"  Trials: {args.trials}, Timeout: {args.timeout}m", flush=True)
    print(f"{'='*60}\n", flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Seed with baseline (context OFF) and a conservative guess
    study.enqueue_trial({
        "idc_swing_order": 5, "idc_min_swing_pts": 3.0,
        "idc_weak_bounce_pts": 5.0, "idc_bounce_lookback": 3,
        "idc_elevator_recency_bars": 30,
        "idc_session_pos_bearish": 0.2, "idc_session_pos_bullish": 0.8,
        "idc_bearish_threshold": 3, "idc_bullish_threshold": 3,
    })
    # More conservative: larger swings, higher threshold
    study.enqueue_trial({
        "idc_swing_order": 10, "idc_min_swing_pts": 8.0,
        "idc_weak_bounce_pts": 8.0, "idc_bounce_lookback": 3,
        "idc_elevator_recency_bars": 30,
        "idc_session_pos_bearish": 0.15, "idc_session_pos_bullish": 0.85,
        "idc_bearish_threshold": 4, "idc_bullish_threshold": 4,
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

    # Baseline (context OFF, session range ON)
    baseline_trades = run_sessions(TEST_SESSIONS, BASE_STRATEGY)
    bm = compute_metrics(baseline_trades)
    print(f"  Baseline (context OFF): {bm['n']}T, PF={bm['pf']}, PnL={bm['total_pnl']:+.1f}, "
          f"WR={bm['wr']}%, Sharpe={bm['sharpe']}", flush=True)

    # Best trial on OOS
    best_params = replace(BASE_STRATEGY,
        use_intraday_context=True,
        **{k: v for k, v in best.params.items()},
    )
    best_trades = run_sessions(TEST_SESSIONS, best_params)
    bst = compute_metrics(best_trades)
    print(f"  Best (context ON):      {bst['n']}T, PF={bst['pf']}, PnL={bst['total_pnl']:+.1f}, "
          f"WR={bst['wr']}%, Sharpe={bst['sharpe']}", flush=True)

    delta = bst["total_pnl"] - bm["total_pnl"]
    print(f"\n  OOS Delta: {delta:+.1f} pts", flush=True)
    if bst["pf"] >= bm["pf"] and bst["total_pnl"] > bm["total_pnl"]:
        print(f"  ✓ OOS IMPROVED (PF and PnL better than baseline)", flush=True)
    elif bst["total_pnl"] > 0:
        print(f"  ~ OOS MIXED (positive PnL but not clearly better)", flush=True)
    else:
        print(f"  ✗ OOS FAILED — intraday context may not generalize", flush=True)

    # ── Top 5 ──────────────────────────────────────────────────────────
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
              f"swing={p['idc_swing_order']}, min_sw={p['idc_min_swing_pts']}, "
              f"bounce={p['idc_weak_bounce_pts']}, bear_th={p['idc_bearish_threshold']}", flush=True)

    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "best_params": best.params,
        "best_train_sharpe": best.value,
        "oos_baseline": bm,
        "oos_best": bst,
        "oos_delta_pts": round(delta, 1),
        "top_5": [{
            "number": t.number, "params": t.params,
            "sharpe": t.value, "pf": t.user_attrs.get("train_pf"),
        } for t in sorted_trials[:5] if t.value is not None],
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
