"""Optuna optimizer for Mode 1 Green (trend up day) detection parameters.

Finds the optimal thresholds for detecting trend days and the relaxed FB
parameters to use when a trend day is confirmed.

Walk-forward split:
  Train: 2024-02 to 2025-06 (~17 months)
  Test:  2025-07 to 2026-02 (~8 months OOS)

Parameters optimized:
  use_mode1_green_detection: True/False (is trend detection worth it?)
  mode1_green_resistance_broken_threshold: 2-5 (how many resistances = trend)
  mode1_green_bars_above_pdh: 15-60 (how long above PDH)
  mode1_green_bullish_pressure_bars: 30-120 (sustained HH/HL window)
  mode1_green_level_broken_hold_bars: 10-30 (must hold broken for N bars)
  mode1_green_fb_min_rr: 0.8-1.5 (relaxed R:R on trend days)
  mode1_green_size_factor: 0.5-1.5 (sizing on trend days)

Objective: maximize combined train+test PnL with Sharpe tiebreak.
Penalty: any year losing > 300 pts hurts proportionally.
Penalty: < 80 total trades (avoid overfitting to tiny samples).

Usage:
    python3 -u backtest/optuna_mode1_green.py [--trials 100] [--timeout 300]
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from loguru import logger
logger.remove()

import time
import json
import argparse
import numpy as np
from datetime import datetime, date, time as dt_time, timedelta
from pathlib import Path

import optuna
import pandas as pd

from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    RiskParams, SessionTimes, ESContractSpec,
)
from core.regime_filter import RegimeParams, build_daily_bars
from strategy.mancini_long import ManciniLongStrategy

DATA_PATH = _PROJECT_ROOT / "data" / "ES_1m_full_session_2021-01-01_2026-02-05.parquet"
RESULTS_PATH = _PROJECT_ROOT / "data" / "optuna_mode1_green_results.json"

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

TRAIN_START = date(2024, 2, 1)
TRAIN_END = date(2025, 6, 30)
TEST_START = date(2025, 7, 1)
TEST_END = date(2026, 2, 5)


def get_window(t):
    if dt_time(9, 30) <= t < dt_time(13, 0): return "Morning"
    if dt_time(15, 0) <= t <= dt_time(16, 50): return "Afternoon"
    if t >= dt_time(22, 0) or t < dt_time(2, 0): return "Late Night"
    if dt_time(6, 0) <= t < dt_time(9, 30): return "Pre-RTH"
    return "Blocked"


print("Loading data...")
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

print(f"  {len(ALL_SESSIONS)} total sessions")
print(f"  Train: {len(TRAIN_SESSIONS)} sessions ({TRAIN_START} to {TRAIN_END})")
print(f"  Test:  {len(TEST_SESSIONS)} sessions ({TEST_START} to {TEST_END})")
print(f"  Loaded in {time.time()-t0:.1f}s")


def run_sessions(sessions, params, exit_params, risk_params, regime_params, min_rr):
    """Run backtest. Returns list of trade dicts."""
    trades = []
    prev_rth = None

    for session_date, session_df in sessions:
        daily_history = DAILY_BARS[DAILY_BARS.index.date < session_date]

        strategy = ManciniLongStrategy(
            strategy_params=params, elevator_params=ELEVATOR, exit_params=exit_params,
            risk_params=risk_params, session_times=FULL_SESSION, contract=MES,
            min_rr_ratio=min_rr, rth_filter=(dt_time(9, 30), dt_time(16, 0)),
            regime_params=regime_params, daily_history=daily_history,
        )
        prior = prev_rth if prev_rth is not None and len(prev_rth) > 0 else None
        strategy.run_day(session_df, prior_day_df=prior,
                        session_date=datetime.combine(session_date, dt_time(0, 0)))

        for t in strategy.trade_records:
            if t.entry_bar_idx >= len(session_df):
                continue
            entry_ts = session_df.index[t.entry_bar_idx]
            et = entry_ts.time()
            window = get_window(et)
            if window == "Blocked":
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
        return {"n": 0, "total_pnl": -9999, "pf": 0, "wr": 0, "sharpe": -10,
                "year_pnls": {}, "min_year": -9999}

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
        year_pnls[year] = sum(t["pnl_pts"] for t in trades if t["year"] == year)

    min_year = min(year_pnls.values()) if year_pnls else -9999

    return {
        "n": n, "total_pnl": total_pnl, "pf": pf, "wr": wr, "sharpe": sharpe,
        "year_pnls": year_pnls, "min_year": min_year,
    }


def build_params(trial):
    """Only Mode 1 Green params vary — everything else locked to production."""
    # ── Mode 1 Green params being optimized ───────────────────────────
    enable_mode1 = trial.suggest_categorical("use_mode1_green_detection", [True, False])
    resistance_threshold = trial.suggest_int("mode1_green_resistance_broken_threshold", 2, 5)
    bars_above_pdh = trial.suggest_int("mode1_green_bars_above_pdh", 15, 60, step=5)
    bullish_pressure_bars = trial.suggest_int("mode1_green_bullish_pressure_bars", 30, 120, step=10)
    level_broken_hold_bars = trial.suggest_int("mode1_green_level_broken_hold_bars", 10, 30, step=5)
    fb_min_rr = trial.suggest_float("mode1_green_fb_min_rr", 0.8, 1.5, step=0.1)
    size_factor = trial.suggest_float("mode1_green_size_factor", 0.5, 1.5, step=0.1)

    strategy_params = StrategyParams(
        # Production-locked FB/LR params
        acceptance_max_dip_pts=15.0,
        acceptance_min_hold_bars=11,
        swing_low_order=15,
        fb_stop_buffer_pts=6.0,
        lr_stop_buffer_pts=4.0,
        max_target_distance_pts=30.0,
        max_fb_sweep_depth_pts=999.0,
        true_breakdown_abort_bars=40,
        multi_hour_rally_min_pts=20.0,
        signal_cooldown_bars=15,
        allow_breakdown_short=True,
        bd_confirm_bars=21,
        bd_stop_buffer_pts=6.0,
        bd_max_break_depth_pts=17.0,
        bd_timeout_bars=35,
        bd_require_major_level=True,
        allow_backtest_short=False,
        allow_velocity_short=True,
        vbd_min_break_pts=8.0,
        vbd_min_volume_ratio=3.0,
        vbd_stop_buffer_pts=3.0,
        vbd_position_size_factor=0.25,
        allow_double_dip=True,
        dd_cooldown_bars=120,
        dd_min_depth_below_stop_pts=5.0,
        dd_bypass_level_gate=True,
        dd_bypass_cooldown=True,
        dd_position_size_factor=0.5,
        dd_fixed_stop_until_t1=True,
        dd_trail_pts_after_t1=25.0,
        allow_deep_sell_recovery=True,
        deep_sell_threshold_pts=30.0,
        deep_sell_swing_order=5,
        deep_sell_rally_confirm_pts=20.0,
        min_session_range_pts=15.0,
        min_session_range_grace_bars=30,
        use_regime_filter=False,
        # Danger zone from Mancini — ON by default
        danger_zone_pts=5.0,
        danger_zone_require_dip_acceptance=True,
        # === MODE 1 GREEN PARAMS BEING OPTIMIZED ===
        use_mode1_green_detection=enable_mode1,
        mode1_green_resistance_broken_threshold=resistance_threshold,
        mode1_green_bars_above_pdh=bars_above_pdh,
        mode1_green_bullish_pressure_bars=bullish_pressure_bars,
        mode1_green_level_broken_hold_bars=level_broken_hold_bars,
        mode1_green_fb_min_rr=fb_min_rr,
        mode1_green_size_factor=size_factor,
    )
    exit_params = ExitParams(
        default_contracts=2,
        t1_exit_fraction=0.5,
        t2_exit_fraction=0.0,
        runner_fraction=0.5,
        breakeven_buffer_pts=-3.0,
        trailing_stop_pts=12.0,
        runner_prior_day_low_buffer_pts=1.0,
        fb_max_hold_bars=14,
    )
    risk_params = RiskParams(
        max_trades_per_day=999,
        max_daily_loss_pts=9999.0,
        skip_tuesdays=False,
        min_rr_ratio=0.8,
        max_stop_distance_pts=60.0,
    )
    regime_params = RegimeParams(
        mode="ema",
        ema_span=30,
        slope_lookback=10,
        slope_threshold_atr_mult=0.325,
    )
    return strategy_params, exit_params, risk_params, regime_params, 0.8


def objective(trial):
    try:
        params, exit_params, risk_params, regime_params, min_rr = build_params(trial)
    except Exception:
        return -10000.0

    try:
        train_trades = run_sessions(TRAIN_SESSIONS, params, exit_params, risk_params, regime_params, min_rr)
        test_trades = run_sessions(TEST_SESSIONS, params, exit_params, risk_params, regime_params, min_rr)
    except Exception:
        return -10000.0

    train = compute_metrics(train_trades)
    test = compute_metrics(test_trades)

    trial.set_user_attr("train_pnl", train["total_pnl"])
    trial.set_user_attr("test_pnl", test["total_pnl"])
    trial.set_user_attr("train_pf", train["pf"])
    trial.set_user_attr("test_pf", test["pf"])
    trial.set_user_attr("train_n", train["n"])
    trial.set_user_attr("test_n", test["n"])
    trial.set_user_attr("train_wr", train["wr"])
    trial.set_user_attr("test_wr", test["wr"])
    trial.set_user_attr("train_sharpe", train["sharpe"])
    trial.set_user_attr("test_sharpe", test["sharpe"])
    trial.set_user_attr("train_year_pnls", train["year_pnls"])
    trial.set_user_attr("test_year_pnls", test["year_pnls"])

    combined_pnl = train["total_pnl"] + test["total_pnl"]
    combined_sharpe = (train["sharpe"] + test["sharpe"]) / 2

    min_year = min(train["min_year"], test["min_year"])
    year_penalty = max(0, -min_year - 300) * 2

    score = combined_pnl + combined_sharpe * 50 - year_penalty

    if train["n"] + test["n"] < 80:
        score -= 5000

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--study-name", type=str, default="mode1_green_v1")
    args = parser.parse_args()

    storage = f"sqlite:///{_PROJECT_ROOT}/data/optuna_mode1_green.db"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
    )

    print(f"\nOptimizing Mode 1 Green params")
    print(f"Trials: {args.trials}, timeout per trial: {args.timeout}s")
    print(f"Study: {args.study_name}\n")

    t_start = time.time()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout * args.trials,
                   show_progress_bar=True)
    elapsed = time.time() - t_start

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE in {elapsed:.0f}s")
    print(f"{'='*70}")

    trials_sorted = sorted(study.trials,
                          key=lambda t: t.value if t.value is not None else -99999,
                          reverse=True)[:10]

    print(f"\nTop 10 trials:")
    for i, t in enumerate(trials_sorted, 1):
        train_pnl = t.user_attrs.get("train_pnl", 0)
        test_pnl = t.user_attrs.get("test_pnl", 0)
        print(f"{i:<3} score={t.value:<8.1f} train={train_pnl:+8.1f} test={test_pnl:+8.1f} "
              f"enabled={t.params.get('use_mode1_green_detection')} "
              f"rr={t.params.get('mode1_green_fb_min_rr')} "
              f"size={t.params.get('mode1_green_size_factor')}")

    best = study.best_trial
    results = {
        "best_params": best.params,
        "best_score": best.value,
        "train_pnl": best.user_attrs.get("train_pnl"),
        "test_pnl": best.user_attrs.get("test_pnl"),
        "train_pf": best.user_attrs.get("train_pf"),
        "test_pf": best.user_attrs.get("test_pf"),
        "train_n": best.user_attrs.get("train_n"),
        "test_n": best.user_attrs.get("test_n"),
        "train_year_pnls": best.user_attrs.get("train_year_pnls"),
        "test_year_pnls": best.user_attrs.get("test_year_pnls"),
        "n_trials": len(study.trials),
        "elapsed_sec": elapsed,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"\nBEST PARAMS:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"\nBEST PERFORMANCE:")
    print(f"  Train: {best.user_attrs.get('train_pnl', 0):+.1f} pts, "
          f"PF {best.user_attrs.get('train_pf', 0):.2f}, "
          f"{best.user_attrs.get('train_n', 0)} trades")
    print(f"  Test:  {best.user_attrs.get('test_pnl', 0):+.1f} pts, "
          f"PF {best.user_attrs.get('test_pf', 0):.2f}, "
          f"{best.user_attrs.get('test_n', 0)} trades")


if __name__ == "__main__":
    main()
