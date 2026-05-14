"""Targeted Optuna optimizer informed by live data analysis (Mar 2026).

Walk-forward split:
  Train: 2024-02 to 2025-06 (~17 months, most representative of current regime)
  Test:  2025-07 to 2026-02 (~8 months, most recent OOS)

Key insights from 1,651 live events:
  - Deep sweeps (50+ pts) produce 117.6 pt avg recovery — biggest winners
  - BD Shorts with 10-15 pt stops are ALL winners when allowed through
  - Wider acceptance dips are profitable, not risky
  - Sweep depth gate was blocking the most profitable setups

Usage:
    python3 -u backtest/optuna_targeted.py [--trials 60] [--timeout 180]
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from loguru import logger
logger.remove()

import time
import json
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
RESULTS_PATH = _PROJECT_ROOT / "data" / "optuna_targeted_results.json"

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

# Walk-forward split
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

    # Daily PnL for Sharpe
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
    n_long = sum(1 for t in trades if t["direction"] == "long")
    n_short = sum(1 for t in trades if t["direction"] == "short")
    long_pnl = sum(t["pnl_pts"] for t in trades if t["direction"] == "long")
    short_pnl = sum(t["pnl_pts"] for t in trades if t["direction"] == "short")

    return {
        "n": n, "total_pnl": total_pnl, "pf": pf, "wr": wr, "sharpe": sharpe,
        "year_pnls": year_pnls, "min_year": min_year,
        "n_long": n_long, "n_short": n_short,
        "long_pnl": long_pnl, "short_pnl": short_pnl,
    }


def build_params(trial):
    """Build params from trial suggestions — data-informed ranges."""
    # ── Regime filter ──────────────────────────────────────────────────
    slope_mult = trial.suggest_float("slope_threshold_atr_mult", 0.05, 0.35, step=0.025)
    ema_span = trial.suggest_int("ema_span", 20, 80, step=10)
    slope_lookback = trial.suggest_int("slope_lookback", 3, 10)

    # ── Long-side params ───────────────────────────────────────────────
    fb_stop = trial.suggest_float("fb_stop_buffer_pts", 3.5, 8.0, step=0.5)
    lr_stop = trial.suggest_float("lr_stop_buffer_pts", 3.0, 7.0, step=0.5)
    hold_bars = trial.suggest_int("acceptance_min_hold_bars", 5, 12)
    # Live data: deep dips are profitable — widen range
    accept_dip = trial.suggest_float("acceptance_max_dip_pts", 3.0, 15.0, step=1.0)
    # Live data: deep sweeps (50+ pts) produce biggest winners (117.6 pt avg recovery)
    max_sweep_depth = trial.suggest_float("max_fb_sweep_depth_pts", 10.0, 100.0, step=10.0)
    max_target_dist = trial.suggest_float("max_target_distance_pts", 15.0, 45.0, step=5.0)

    # ── BD Short ───────────────────────────────────────────────────────
    bd_confirm = trial.suggest_int("bd_confirm_bars", 8, 25)
    bd_stop = trial.suggest_float("bd_stop_buffer_pts", 1.5, 8.0, step=0.5)
    bd_max_depth = trial.suggest_float("bd_max_break_depth_pts", 8.0, 25.0, step=1.0)
    bd_timeout = trial.suggest_int("bd_timeout_bars", 20, 60, step=5)

    # ── Risk ───────────────────────────────────────────────────────────
    min_rr = trial.suggest_float("min_rr_ratio", 0.5, 2.0, step=0.1)
    # Live data: BD Shorts need 10-15 pt stops — allow up to 20
    max_stop_dist = trial.suggest_float("max_stop_distance_pts", 10.0, 20.0, step=1.0)

    strategy_params = StrategyParams(
        swing_low_order=15, multi_hour_rally_min_pts=22.5,
        level_reclaim_min_touches=4,
        acceptance_min_hold_bars=hold_bars,
        acceptance_min_hold_bars_deep=8,
        acceptance_max_dip_pts=accept_dip,
        true_breakdown_abort_bars=20,
        fb_stop_buffer_pts=fb_stop,
        lr_stop_buffer_pts=lr_stop,
        non_acceptance_min_recovery_pts=5.0,
        max_fb_sweep_depth_pts=max_sweep_depth,
        level_sweep_min_bars_below=3,
        max_target_distance_pts=max_target_dist,
        allow_breakdown_short=True,
        bd_min_break_depth_pts=1.0,
        bd_confirm_bars=bd_confirm,
        bd_timeout_bars=bd_timeout,
        bd_stop_buffer_pts=bd_stop,
        bd_max_break_depth_pts=bd_max_depth,
        allow_backtest_short=False,
        use_regime_filter=True,
        regime_mode="ema",
        signal_cooldown_bars=15,
    )
    exit_params = ExitParams(
        default_contracts=4, t1_exit_fraction=1.0,
        trailing_stop_pts=7.0,
    )
    risk_params = RiskParams(
        max_trades_per_day=4,
        max_stop_distance_pts=max_stop_dist,
        skip_tuesdays=False,
        min_rr_ratio=min_rr,
    )
    regime_params = RegimeParams(
        mode="ema",
        ema_span=ema_span,
        slope_lookback=slope_lookback,
        slope_threshold_atr_mult=slope_mult,
    )
    return strategy_params, exit_params, risk_params, regime_params, min_rr


def objective(trial):
    """Maximize composite: PF-weighted Sharpe on train set."""
    strategy_params, exit_params, risk_params, regime_params, min_rr = build_params(trial)

    trades = run_sessions(TRAIN_SESSIONS, strategy_params, exit_params, risk_params,
                          regime_params, min_rr)
    m = compute_metrics(trades)

    trial.set_user_attr("train_pnl", m["total_pnl"])
    trial.set_user_attr("train_n", m["n"])
    trial.set_user_attr("train_pf", m["pf"])
    trial.set_user_attr("train_wr", m["wr"])
    trial.set_user_attr("train_sharpe", m["sharpe"])

    # Reject if too few trades (need statistical significance)
    if m["n"] < 20:
        return -10.0

    # Composite score: Sharpe * sqrt(PF) — rewards both consistency and edge
    # PF < 1.0 gets penalized, PF > 1.0 boosts Sharpe
    pf_factor = np.sqrt(max(m["pf"], 0.01))
    score = m["sharpe"] * pf_factor

    print(f"  T{trial.number:>3}: {m['n']}T {m['wr']:.0f}%WR PF={m['pf']:.2f} "
          f"Sharpe={m['sharpe']:.2f} PnL={m['total_pnl']:+,.0f} "
          f"L:{m['n_long']}/{m['long_pnl']:+,.0f} S:{m['n_short']}/{m['short_pnl']:+,.0f} "
          f"score={score:+.2f}")

    return score


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in minutes")
    args = parser.parse_args()

    print("=" * 80)
    print(f"TARGETED OPTUNA OPTIMIZATION (live-data-informed, {args.trials} trials)")
    print(f"  Train: {TRAIN_START} to {TRAIN_END} ({len(TRAIN_SESSIONS)} sessions)")
    print(f"  Test:  {TEST_START} to {TEST_END} ({len(TEST_SESSIONS)} sessions)")
    print(f"  Objective: Sharpe * sqrt(PF) on train")
    print(f"  Key: wider sweep depth (10-100), wider BD stops (up to 20 pts)")
    print("=" * 80)

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=15,
        seed=42,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="targeted_live_informed")

    # Seed with current production config
    study.enqueue_trial({
        "slope_threshold_atr_mult": 0.35, "ema_span": 80, "slope_lookback": 6,
        "fb_stop_buffer_pts": 7.0, "lr_stop_buffer_pts": 3.0,
        "acceptance_min_hold_bars": 7,
        "acceptance_max_dip_pts": 4.0, "max_fb_sweep_depth_pts": 50.0,
        "max_target_distance_pts": 15.0,
        "bd_confirm_bars": 8, "bd_stop_buffer_pts": 4.0,
        "bd_max_break_depth_pts": 14.0, "bd_timeout_bars": 55,
        "min_rr_ratio": 1.0, "max_stop_distance_pts": 15.0,
    })

    # Seed with data-informed config (wider params)
    study.enqueue_trial({
        "slope_threshold_atr_mult": 0.15, "ema_span": 50, "slope_lookback": 5,
        "fb_stop_buffer_pts": 5.5, "lr_stop_buffer_pts": 5.0,
        "acceptance_min_hold_bars": 7,
        "acceptance_max_dip_pts": 10.0, "max_fb_sweep_depth_pts": 100.0,
        "max_target_distance_pts": 30.0,
        "bd_confirm_bars": 15, "bd_stop_buffer_pts": 3.0,
        "bd_max_break_depth_pts": 15.0, "bd_timeout_bars": 40,
        "min_rr_ratio": 0.5, "max_stop_distance_pts": 20.0,
    })

    t0 = time.time()
    study.optimize(objective, n_trials=args.trials,
                   timeout=args.timeout * 60 if args.timeout else None)
    elapsed = time.time() - t0

    # ── Best on train ──────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*80}")
    print(f"BEST TRAIN Trial #{best.number} (score={best.value:+.3f})")
    print(f"{'='*80}")
    print(f"  Train: {best.user_attrs['train_n']}T, "
          f"WR={best.user_attrs['train_wr']:.0f}%, "
          f"PF={best.user_attrs['train_pf']:.2f}, "
          f"Sharpe={best.user_attrs['train_sharpe']:.2f}, "
          f"PnL={best.user_attrs['train_pnl']:+,.1f}")
    for k, v in sorted(best.params.items()):
        print(f"    {k}: {v}")

    # ── OOS Validation ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"OOS VALIDATION ({TEST_START} to {TEST_END})")
    print(f"{'='*80}")

    strategy_params, exit_params, risk_params, regime_params, min_rr = build_params(best)
    test_trades = run_sessions(TEST_SESSIONS, strategy_params, exit_params, risk_params,
                               regime_params, min_rr)
    tm = compute_metrics(test_trades)
    print(f"  OOS: {tm['n']}T, WR={tm['wr']:.0f}%, PF={tm['pf']:.2f}, "
          f"Sharpe={tm['sharpe']:.2f}, PnL={tm['total_pnl']:+,.1f}")
    print(f"  Long: {tm['n_long']}T {tm['long_pnl']:+,.1f} | "
          f"Short: {tm['n_short']}T {tm['short_pnl']:+,.1f}")

    oos_pass = tm['pf'] > 1.0 and tm['sharpe'] > 0
    print(f"  OOS PASS: {'YES' if oos_pass else 'NO'} (PF>1.0: {tm['pf']>1.0}, Sharpe>0: {tm['sharpe']>0})")

    # ── Full dataset validation ────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FULL DATASET VALIDATION (all sessions)")
    print(f"{'='*80}")
    all_sessions_filtered = [(d, df) for d, df in ALL_SESSIONS if d >= TRAIN_START]
    all_trades = run_sessions(all_sessions_filtered, strategy_params, exit_params,
                              risk_params, regime_params, min_rr)
    am = compute_metrics(all_trades)
    yr_str = " | ".join(f"{y}:{v:+.0f}" for y, v in sorted(am["year_pnls"].items()))
    print(f"  Full: {am['n']}T PF={am['pf']:.2f} Sharpe={am['sharpe']:.2f} "
          f"PnL={am['total_pnl']:+,.1f}")
    print(f"  Long: {am['n_long']}T {am['long_pnl']:+,.1f} | "
          f"Short: {am['n_short']}T {am['short_pnl']:+,.1f}")
    print(f"  Years: {yr_str}")

    # ── Top 5 trials ───────────────────────────────────────────────────
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print(f"\n  Top 5 trials:")
    print(f"  {'#':<5} {'Score':>7} {'N':>4} {'PF':>5} {'Sharpe':>7} {'PnL':>8}")
    print(f"  {'-'*45}")
    for i, t in enumerate(sorted_trials, 1):
        print(f"  {i:<5} {t.value:>+7.3f} {t.user_attrs.get('train_n',0):>4} "
              f"{t.user_attrs.get('train_pf',0):>5.2f} "
              f"{t.user_attrs.get('train_sharpe',0):>+7.2f} "
              f"{t.user_attrs.get('train_pnl',0):>+8.0f}")

    # ── Param importance ───────────────────────────────────────────────
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\n  Param importance:")
        for k, v in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v:.1%}")
    except Exception as e:
        print(f"  fANOVA failed: {e}")

    # ── Validate top 3 OOS ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("TOP 3 TRIALS — OOS VALIDATION")
    print(f"{'='*80}")
    for i, t in enumerate(sorted_trials[:3], 1):
        sp, ep, rp, regp, mrr = build_params(t)
        tt = run_sessions(TEST_SESSIONS, sp, ep, rp, regp, mrr)
        ttm = compute_metrics(tt)
        oos_ok = "PASS" if ttm['pf'] > 1.0 and ttm['sharpe'] > 0 else "FAIL"
        print(f"  #{i} T{t.number}: OOS {ttm['n']}T PF={ttm['pf']:.2f} "
              f"Sharpe={ttm['sharpe']:.2f} PnL={ttm['total_pnl']:+,.0f} [{oos_ok}]")

    print(f"\n  Elapsed: {elapsed/60:.1f} min ({elapsed/len(study.trials):.0f}s/trial)")

    # Save
    results = {
        "best_trial": best.number, "best_score": best.value,
        "best_params": best.params,
        "train_metrics": {
            "n": best.user_attrs["train_n"],
            "pf": best.user_attrs["train_pf"],
            "wr": best.user_attrs["train_wr"],
            "sharpe": best.user_attrs["train_sharpe"],
            "pnl": best.user_attrs["train_pnl"],
        },
        "oos_metrics": {
            "n": tm["n"], "pf": tm["pf"], "wr": tm["wr"],
            "sharpe": tm["sharpe"], "pnl": tm["total_pnl"],
            "pass": oos_pass,
        },
        "full_metrics": {
            "n": am["n"], "pf": am["pf"], "sharpe": am["sharpe"],
            "pnl": am["total_pnl"], "year_pnls": {str(k): v for k, v in am["year_pnls"].items()},
        },
        "n_trials": len(study.trials),
        "elapsed_min": elapsed / 60,
        "walk_forward": {
            "train": f"{TRAIN_START} to {TRAIN_END}",
            "test": f"{TEST_START} to {TEST_END}",
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
