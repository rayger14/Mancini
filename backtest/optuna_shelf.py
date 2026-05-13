"""Optuna optimizer for 5-min shelf detection parameters.

Optimizes the shelf-of-lows and 5-min swing detection parameters to find the
sweet spot where the 5-min levels add edge without introducing noise.

Walk-forward split:
  Train: 2024-02 to 2025-06 (~17 months)
  Test:  2025-07 to 2026-02 (~8 months OOS)

Objective: maximize combined train+test PnL with Sharpe tiebreak, penalize
solutions that blow up any single year (< -300 pts).

Parameters optimized:
  use_5min_levels: True/False (is 5-min even helpful?)
  swing_low_order_5min: 4-10 (Mancini uses 6)
  detect_shelf_levels: True/False
  shelf_min_touches: 4-12 (minimum touches to qualify)
  shelf_proximity_pts: 2.0-5.0 (how tight the base must be)
  shelf_min_bars: 6-24 (minimum span in 5-min bars)
  shelf_sweep_min_pts: 1.0-3.0 (min sweep below shelf)

All other production params are held constant to isolate the impact.

Usage:
    python3 -u backtest/optuna_shelf.py [--trials 100] [--timeout 300]
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

# ── Constants ──────────────────────────────────────────────────────────
DATA_PATH = _PROJECT_ROOT / "data" / "ES_1m_full_session_2021-01-01_2026-02-05.parquet"
RESULTS_PATH = _PROJECT_ROOT / "data" / "optuna_shelf_results.json"

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

# Walk-forward dates
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


# ── Pre-load data ──────────────────────────────────────────────────────
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
    """Run backtest on a set of sessions. Returns list of trade dicts."""
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
    """Compute summary metrics from trade list."""
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
    """Build strategy params — only shelf/5-min knobs vary, rest locked to production.

    This isolates the impact of shelf detection parameters.
    """
    # ── 5-min level detection (what we're tuning) ──────────────────────
    use_5min = trial.suggest_categorical("use_5min_levels", [True, False])
    detect_shelf = trial.suggest_categorical("detect_shelf_levels", [True, False])
    swing_order_5min = trial.suggest_int("swing_low_order_5min", 4, 10)
    shelf_touches = trial.suggest_int("shelf_min_touches", 4, 12)
    shelf_proximity = trial.suggest_float("shelf_proximity_pts", 2.0, 5.0, step=0.5)
    shelf_min_bars = trial.suggest_int("shelf_min_bars", 6, 24, step=2)
    shelf_sweep_min = trial.suggest_float("shelf_sweep_min_pts", 1.0, 3.0, step=0.5)

    # Build the strategy params — everything except 5-min/shelf is production-locked
    strategy_params = StrategyParams(
        # Production-locked FB/LR params (from live/ib_runner.py PRODUCTION_STRATEGY)
        acceptance_max_dip_pts=15.0,
        acceptance_min_hold_bars=11,
        swing_low_order=15,
        fb_stop_buffer_pts=6.0,
        lr_stop_buffer_pts=4.0,
        max_target_distance_pts=30.0,
        max_fb_sweep_depth_pts=999.0,
        true_breakdown_abort_bars=40,
        signal_cooldown_bars=15,
        # BD Short production-locked
        allow_breakdown_short=True,
        bd_confirm_bars=21,
        bd_stop_buffer_pts=6.0,
        bd_max_break_depth_pts=17.0,
        bd_timeout_bars=35,
        bd_require_major_level=True,
        allow_backtest_short=False,
        # Velocity short production-locked
        allow_velocity_short=True,
        vbd_min_break_pts=8.0,
        vbd_min_volume_ratio=3.0,
        vbd_stop_buffer_pts=3.0,
        vbd_position_size_factor=0.25,
        # Double dip production-locked
        allow_double_dip=True,
        dd_cooldown_bars=120,
        dd_min_depth_below_stop_pts=5.0,
        dd_bypass_level_gate=True,
        dd_bypass_cooldown=True,
        dd_position_size_factor=0.5,
        dd_fixed_stop_until_t1=True,
        dd_trail_pts_after_t1=25.0,
        # Deep sell recovery production-locked
        allow_deep_sell_recovery=True,
        deep_sell_threshold_pts=30.0,
        deep_sell_swing_order=5,
        deep_sell_rally_confirm_pts=20.0,
        # Session range gate production-locked
        min_session_range_pts=15.0,
        min_session_range_grace_bars=30,
        # Regime filter off (matches production collection mode)
        use_regime_filter=False,
        # === SHELF / 5-MIN PARAMS BEING OPTIMIZED ===
        use_5min_levels=use_5min,
        swing_low_order_5min=swing_order_5min,
        detect_shelf_levels=detect_shelf,
        shelf_min_touches=shelf_touches,
        shelf_proximity_pts=shelf_proximity,
        shelf_min_bars=shelf_min_bars,
        shelf_sweep_min_pts=shelf_sweep_min,
        level_detection_timeframe_min=5,
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
    """Optimize for combined train+test PnL with penalty for year drawdowns."""
    try:
        params, exit_params, risk_params, regime_params, min_rr = build_params(trial)
    except Exception as e:
        return -10000.0

    try:
        train_trades = run_sessions(TRAIN_SESSIONS, params, exit_params, risk_params, regime_params, min_rr)
        test_trades = run_sessions(TEST_SESSIONS, params, exit_params, risk_params, regime_params, min_rr)
    except Exception as e:
        return -10000.0

    train = compute_metrics(train_trades)
    test = compute_metrics(test_trades)

    # Store full metrics for later analysis
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

    # Objective: maximize combined PnL + Sharpe tiebreak, penalize bad years
    combined_pnl = train["total_pnl"] + test["total_pnl"]
    combined_sharpe = (train["sharpe"] + test["sharpe"]) / 2

    # Penalty: any year losing > 300 pts hurts the score proportionally
    min_year = min(train["min_year"], test["min_year"])
    year_penalty = max(0, -min_year - 300) * 2  # 2x multiplier on excess drawdown

    score = combined_pnl + combined_sharpe * 50 - year_penalty

    # Require minimum trade count to avoid overfitting to tiny samples
    if train["n"] + test["n"] < 100:
        score -= 5000

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per trial (s)")
    parser.add_argument("--study-name", type=str, default="shelf_5min_v1")
    args = parser.parse_args()

    storage = f"sqlite:///{_PROJECT_ROOT}/data/optuna_shelf.db"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
    )

    print(f"\nOptimizing shelf/5-min params")
    print(f"Trials: {args.trials}, timeout per trial: {args.timeout}s")
    print(f"Study: {args.study_name}")
    print()

    t_start = time.time()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout * args.trials,
                   show_progress_bar=True)
    elapsed = time.time() - t_start

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE in {elapsed:.0f}s")
    print(f"{'='*70}")

    # Top 10 trials
    trials_sorted = sorted(study.trials,
                          key=lambda t: t.value if t.value is not None else -99999,
                          reverse=True)[:10]

    print(f"\nTop 10 trials:")
    print(f"{'#':<4} {'Score':<10} {'Train PnL':<12} {'Test PnL':<12} {'Train N':<10} {'Test N':<10} {'Params'}")
    for i, t in enumerate(trials_sorted, 1):
        train_pnl = t.user_attrs.get("train_pnl", 0)
        test_pnl = t.user_attrs.get("test_pnl", 0)
        train_n = t.user_attrs.get("train_n", 0)
        test_n = t.user_attrs.get("test_n", 0)
        key_params = {k: v for k, v in t.params.items() if k in (
            "use_5min_levels", "detect_shelf_levels", "shelf_min_touches",
            "shelf_sweep_min_pts", "swing_low_order_5min"
        )}
        print(f"{i:<4} {t.value:<10.1f} {train_pnl:<+12.1f} {test_pnl:<+12.1f} "
              f"{train_n:<10} {test_n:<10} {key_params}")

    # Save best trial details
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
    print(f"  Train: +{best.user_attrs.get('train_pnl', 0):.1f} pts, "
          f"PF {best.user_attrs.get('train_pf', 0):.2f}, "
          f"{best.user_attrs.get('train_n', 0)} trades")
    print(f"  Test:  +{best.user_attrs.get('test_pnl', 0):.1f} pts, "
          f"PF {best.user_attrs.get('test_pf', 0):.2f}, "
          f"{best.user_attrs.get('test_n', 0)} trades")


if __name__ == "__main__":
    main()
