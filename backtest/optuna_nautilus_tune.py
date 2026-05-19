"""Optuna-over-Nautilus parameter tuning harness.

Goal: find PRODUCTION_STRATEGY params that GENERALIZE across 5y of ES
data, validated via Nautilus realistic execution simulation. Avoid the
Optuna v1/v2 overfit failure mode that produced the −$184K production
config (tuned on idealized fills, didn't survive realistic execution).

Anti-overfit measures built in:
  1. Walk-forward split: train on 2021-2023, validate on 2024, hold out
     2025-2026. Tuning never sees the hold-out years.
  2. Multi-objective scoring: total PnL × min(yearly_PnL > 0) — refuses
     configs that have one great year and four red years.
  3. Per-trial walk-forward: each trial evaluates on training AND validation
     years and is penalized for overfitting (good train, bad val).
  4. Studies are SCOPED — vary only the params relevant to one hypothesis,
     leave the rest at production defaults. Avoids the joint-search-explosion
     problem that crashes Optuna into local minima.

Usage:
    # See available studies
    python3 backtest/optuna_nautilus_tune.py --list

    # Smoke test (3 trials) — verify pipeline end-to-end
    python3 backtest/optuna_nautilus_tune.py --study fb-acceptance --trials 3

    # Real sweep (overnight)
    python3 backtest/optuna_nautilus_tune.py --study fb-acceptance --trials 100

    # Resume an interrupted study
    python3 backtest/optuna_nautilus_tune.py --study fb-acceptance --trials 50 --resume

    # Report on a completed study without running more trials
    python3 backtest/optuna_nautilus_tune.py --study fb-acceptance --report

Available studies (defined in STUDIES below):
  fb-acceptance  : acceptance gate timing & dip tolerance (today's pain point)
  fb-exits       : 75/25 runner mechanics vs 100% T1 exit
  mode-1-green   : block FB longs vs unblock vs unblock+relax R:R
  fb-stops       : FB stop buffer placement
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

import optuna

from backtest.nautilus_runner import NautilusBacktestRunner, NautilusBacktestConfig
from backtest.nautilus_production_5y import load_data, build_daily_sessions
from live.ib_runner import (
    PRODUCTION_STRATEGY, PRODUCTION_ELEVATOR,
    PRODUCTION_EXIT, PRODUCTION_RISK,
)


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------
TRAINING_YEARS = [2021, 2022, 2023]
VALIDATION_YEAR = 2024
HOLDOUT_YEARS = [2025, 2026]


# ---------------------------------------------------------------------------
# Study definitions
# ---------------------------------------------------------------------------
# Each study scopes the search to a hypothesis. Each entry:
#   - "description": human-readable
#   - "strategy_params": dict of {param_name: (kind, *bounds)} for StrategyParams
#   - "exit_params":     dict of {param_name: (kind, *bounds)} for ExitParams
#   - "fixed":           dict of {param_name: value} to FIX outside production default
#   - "longs_only":      bool — disable shorts so we tune just longs
# kind is "int", "float" with (low, high) or "float" with (low, high, step),
# or "categorical" with a list of choices.

STUDIES: dict[str, dict[str, Any]] = {
    "fb-acceptance": {
        "description": (
            "FB acceptance gate timing & dip tolerance. Mancini's literal "
            "rule is 'hold above the level for 2-3 minutes with shallow dips'. "
            "Current production has 11 bars / 15pt dip — much looser. Find "
            "values that GENERALIZE across 5y, not just one window."
        ),
        "strategy_params": {
            "acceptance_min_hold_bars":     ("int",   2, 15),
            "acceptance_max_dip_pts":       ("float", 2.0, 20.0, 0.5),
            "acceptance_min_hold_bars_deep":("int",   2, 12),
            "non_acceptance_min_recovery_pts": ("float", 3.0, 10.0, 0.5),
            "true_breakdown_abort_bars":    ("int",   15, 50),
        },
        "exit_params": {},
        "longs_only": True,
    },
    "fb-exits": {
        "description": (
            "Exit mechanics. Production has t1_exit_fraction=0.75 + 25% runner "
            "BUT MISSING T2 STAGE. Mancini's literal rule (2025-08-05): "
            "'lock in 75% at first level, leave 25% runner, then lock in more "
            "at second level up, and let a 10% runner go'. Three stages, not two. "
            "This study adds t2_exit_fraction to the search so we can validate "
            "the missing stage. Note: structure-based trailing + multi-session "
            "runners are CODE changes Mancini also requires — not tunable here."
        ),
        "strategy_params": {},
        "exit_params": {
            "t1_exit_fraction":   ("float", 0.5, 1.0, 0.05),
            "t2_exit_fraction":   ("float", 0.0, 0.30, 0.05),  # Mancini: 0.15 typical (25% → 10%)
            "runner_fraction":    ("float", 0.05, 0.30, 0.05),  # Mancini: 0.10 final
            "trailing_stop_pts":  ("float", 5.0, 18.0, 0.5),
            "breakeven_buffer_pts": ("float", -10.0, 2.0, 0.5),
        },
        "longs_only": True,
    },
    "mode-1-green": {
        "description": (
            "Mode 1 Green response. Per Mancini quotes (2025-10-12, 2025-06-15), "
            "Mode 1 Green days are TRIGGERED BY a Failed Breakdown — "
            "blocking FB longs on these days misses the best trade of the day. "
            "Compare block vs unblock vs unblock+relax_R:R."
        ),
        "strategy_params": {
            # Categorical: which Mode 1 Green policy
            # Note: encoded via post-trial hook that monkey-patches the gate
            "mode1_green_policy": ("categorical", [
                "block",        # current production
                "unblock",      # treat like any other FB
                "unblock_relax",  # unblock + relax R:R to 0.5
                "unblock_size_up",  # unblock + 1.5x size
            ]),
        },
        "exit_params": {},
        "longs_only": True,
        "requires_hook": True,  # categorical needs custom strategy logic
    },
    "fb-stops": {
        "description": "FB stop buffer placement (sensitivity check).",
        "strategy_params": {
            "fb_stop_buffer_pts":  ("float", 3.0, 12.0, 0.5),
            "lr_stop_buffer_pts":  ("float", 3.0, 10.0, 0.5),
            "max_fb_sweep_depth_pts": ("float", 5.0, 60.0, 1.0),
        },
        "exit_params": {},
        "longs_only": True,
    },
}


def _suggest(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    """Suggest a value for one param based on its spec tuple."""
    kind = spec[0]
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "float":
        step = spec[3] if len(spec) > 3 else None
        return trial.suggest_float(name, spec[1], spec[2], step=step)
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"Unknown spec kind for {name}: {kind}")


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------


def _build_configs(trial: optuna.Trial, study: dict) -> tuple[Any, Any, dict]:
    """Build StrategyParams and ExitParams overrides from the trial's
    suggested values. Returns (strategy_params, exit_params, overrides_dict)."""
    overrides: dict[str, Any] = {}
    sp_overrides: dict[str, Any] = {}
    ep_overrides: dict[str, Any] = {}

    for name, spec in study["strategy_params"].items():
        val = _suggest(trial, name, spec)
        overrides[name] = val
        # Only apply to StrategyParams if it's actually a field there
        if hasattr(PRODUCTION_STRATEGY, name):
            sp_overrides[name] = val

    for name, spec in study["exit_params"].items():
        val = _suggest(trial, name, spec)
        overrides[name] = val
        if hasattr(PRODUCTION_EXIT, name):
            ep_overrides[name] = val

    strategy_params = PRODUCTION_STRATEGY
    if study.get("longs_only"):
        strategy_params = replace(
            strategy_params,
            allow_breakdown_short=False,
            allow_velocity_short=False,
            allow_backtest_short=False,
            allow_short_fr=False,
            allow_short_lj=False,
        )
    if sp_overrides:
        strategy_params = replace(strategy_params, **sp_overrides)

    exit_params = PRODUCTION_EXIT
    if ep_overrides:
        exit_params = replace(exit_params, **ep_overrides)

    return strategy_params, exit_params, overrides


def _run_nautilus_5y(strategy_params, exit_params,
                     daily_dfs: dict[date, pd.DataFrame]) -> list:
    """Run Nautilus over the given sessions, return list of TradeRecords."""
    cfg = NautilusBacktestConfig(
        strategy_params=strategy_params,
        elevator_params=PRODUCTION_ELEVATOR,
        exit_params=exit_params,
        risk_params=PRODUCTION_RISK,
        min_rr_ratio=PRODUCTION_RISK.min_rr_ratio,
    )
    runner = NautilusBacktestRunner(cfg)
    trades = []
    prior_day_df = None
    for d in sorted(daily_dfs.keys()):
        df = daily_dfs[d]
        if len(df) < 10:
            continue
        try:
            dr = runner.run_single_day(df, prior_day_df=prior_day_df, day=d)
            trades.extend(dr.trade_records)
            prior_day_df = df
        except Exception as e:
            # Don't kill the whole study on one session's bug
            continue
    return trades


def _compute_score(trades: list, years_to_score: list[int],
                   penalty_per_red_year_pts: float = 200.0) -> tuple[float, dict]:
    """Composite objective. Higher = better.

    score = total_pnl - penalty_per_red_year × num_red_years

    The penalty enforces the "consistently positive across years" property.
    A config that makes +2000pt in 2021 and -500pt in 2022/2023 scores
    LOWER than +1000pt every year. Also returns per-year breakdown for
    trial attrs.
    """
    year_pnls = {y: 0.0 for y in years_to_score}
    year_trades = {y: 0 for y in years_to_score}
    for t in trades:
        et = getattr(t, "entry_time", None)
        if et is None:
            continue
        y = et.year
        if y in year_pnls:
            year_pnls[y] += t.pnl_pts
            year_trades[y] += 1

    total = sum(year_pnls.values())
    red_years = sum(1 for v in year_pnls.values() if v < 0)
    score = total - red_years * penalty_per_red_year_pts
    return score, {
        "total_pnl": round(total, 1),
        "year_pnls": {str(y): round(v, 1) for y, v in year_pnls.items()},
        "year_trades": year_trades,
        "red_years": red_years,
    }


def _objective(trial: optuna.Trial,
               study_name: str,
               training_dfs: dict,
               validation_dfs: dict) -> float:
    """Walk-forward objective: train + validate.

    Returns: train_score + val_score - overfit_penalty.

    Where overfit_penalty triggers if train > 0 but val < 0 (we found
    a config that wins on training years but loses on the held-back year).
    """
    study = STUDIES[study_name]
    strategy_params, exit_params, overrides = _build_configs(trial, study)

    # Train phase
    train_trades = _run_nautilus_5y(strategy_params, exit_params, training_dfs)
    train_score, train_attrs = _compute_score(train_trades, TRAINING_YEARS)

    # Validation phase
    val_trades = _run_nautilus_5y(strategy_params, exit_params, validation_dfs)
    val_score, val_attrs = _compute_score(val_trades, [VALIDATION_YEAR])

    # Overfit penalty: if train is positive but val is negative, this config
    # likely memorized training noise. Penalize heavily.
    overfit_penalty = 0.0
    if train_score > 0 and val_score < 0:
        overfit_penalty = abs(val_score) * 2

    final_score = train_score + val_score - overfit_penalty

    # Store details for post-tune inspection
    trial.set_user_attr("train_score", round(train_score, 1))
    trial.set_user_attr("val_score", round(val_score, 1))
    trial.set_user_attr("train_total_pnl", train_attrs["total_pnl"])
    trial.set_user_attr("val_total_pnl", val_attrs["total_pnl"])
    trial.set_user_attr("train_year_pnls", train_attrs["year_pnls"])
    trial.set_user_attr("val_year_pnls", val_attrs["year_pnls"])
    trial.set_user_attr("train_red_years", train_attrs["red_years"])
    trial.set_user_attr("val_red_years", val_attrs["red_years"])
    trial.set_user_attr("train_trades", len(train_trades))
    trial.set_user_attr("val_trades", len(val_trades))
    trial.set_user_attr("overfit_penalty", round(overfit_penalty, 1))
    trial.set_user_attr("overrides", {k: float(v) if isinstance(v, (int, float)) else v
                                       for k, v in overrides.items()})

    return final_score


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report_study(study: optuna.Study) -> None:
    """Print a summary of the study results."""
    if not study.trials:
        print("No trials run yet.")
        return

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nTotal trials: {len(study.trials)}  (completed: {len(completed)})")

    if not completed:
        return

    best = study.best_trial
    print(f"\n{'='*72}")
    print(f"BEST TRIAL: #{best.number}  score={best.value:.1f}")
    print(f"{'='*72}")
    print(f"  Params: {best.params}")
    print(f"  Train score: {best.user_attrs.get('train_score'):.1f}  "
          f"(total {best.user_attrs.get('train_total_pnl')} pts, "
          f"{best.user_attrs.get('train_red_years')} red years, "
          f"{best.user_attrs.get('train_trades')} trades)")
    print(f"  Val score:   {best.user_attrs.get('val_score'):.1f}  "
          f"(total {best.user_attrs.get('val_total_pnl')} pts, "
          f"{best.user_attrs.get('val_red_years')} red years, "
          f"{best.user_attrs.get('val_trades')} trades)")
    print(f"  Overfit penalty: {best.user_attrs.get('overfit_penalty')}")
    print(f"  Train year PnLs: {best.user_attrs.get('train_year_pnls')}")
    print(f"  Val year PnLs:   {best.user_attrs.get('val_year_pnls')}")

    print(f"\nTop 5 trials by score:")
    top = sorted(completed, key=lambda t: -(t.value or float("-inf")))[:5]
    for t in top:
        print(f"  #{t.number:>3}  score={t.value:>+8.1f}  "
              f"train_pnl={t.user_attrs.get('train_total_pnl'):>+7.1f}  "
              f"val_pnl={t.user_attrs.get('val_total_pnl'):>+7.1f}  "
              f"{t.params}")


def _list_studies() -> None:
    print("\nAvailable studies:\n")
    for name, study in STUDIES.items():
        print(f"  {name:18} {study['description']}")
        n_sp = len(study.get('strategy_params', {}))
        n_ep = len(study.get('exit_params', {}))
        longs_only = study.get('longs_only', False)
        print(f"  {'':18} params: {n_sp} strategy + {n_ep} exit, "
              f"longs_only={longs_only}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", help="Study name (see --list)")
    ap.add_argument("--trials", type=int, default=10,
                    help="Optuna trials this invocation (default 10 for smoke testing)")
    ap.add_argument("--db", default="data/optuna_nautilus.db",
                    help="SQLite path for Optuna storage")
    ap.add_argument("--list", action="store_true",
                    help="List available studies and exit")
    ap.add_argument("--report", action="store_true",
                    help="Report on a study without running trials")
    ap.add_argument("--resume", action="store_true",
                    help="Resume an existing study (load_if_exists)")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel trials (uses joblib threading). 4 on a Mac M-series "
                         "gives ~3x speedup. 8+ may thrash on memory.")
    args = ap.parse_args()

    if args.list:
        _list_studies()
        return

    if not args.study:
        _list_studies()
        print("error: --study is required (or use --list)")
        return

    if args.study not in STUDIES:
        print(f"Unknown study: {args.study}")
        _list_studies()
        return

    study_def = STUDIES[args.study]

    # Reject studies that need custom hooks until we implement them
    if study_def.get("requires_hook"):
        print(f"\nStudy '{args.study}' requires custom hook support — not yet implemented.")
        print("Categorical Mode 1 Green policy needs runtime monkey-patching of")
        print("_check_mancini_llm_gates. Skipping for now; ship a hook in a follow-up.")
        return

    print(f"Study: {args.study}")
    print(f"  {study_def['description']}\n")
    print(f"  strategy params varied: {list(study_def['strategy_params'].keys())}")
    print(f"  exit params varied:     {list(study_def['exit_params'].keys())}")
    print(f"  longs_only: {study_def.get('longs_only', False)}")

    # Load data
    print("\nLoading 5y ES data...")
    df = load_data()
    all_sessions = build_daily_sessions(df)
    training_dfs = {d: v for d, v in all_sessions.items() if d.year in TRAINING_YEARS}
    validation_dfs = {d: v for d, v in all_sessions.items() if d.year == VALIDATION_YEAR}
    holdout_dfs = {d: v for d, v in all_sessions.items() if d.year in HOLDOUT_YEARS}

    print(f"  Train: {len(training_dfs)} sessions ({TRAINING_YEARS})")
    print(f"  Val:   {len(validation_dfs)} sessions ({VALIDATION_YEAR})")
    print(f"  Hold:  {len(holdout_dfs)} sessions ({HOLDOUT_YEARS})  [LOCKED — never accessed during tuning]")

    # Optuna study
    db_path = Path(args.db).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path}"
    study_full_name = f"nautilus-{args.study}"

    study = optuna.create_study(
        study_name=study_full_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        load_if_exists=args.resume or args.report,
    )

    if args.report:
        _report_study(study)
        return

    print(f"\nRunning {args.trials} trials (existing: {len(study.trials)})...")
    print(f"  Each trial ~15-20 min (Nautilus 5y backtest)")
    if args.n_jobs > 1:
        wall_min = (args.trials * 17) // args.n_jobs
        print(f"  n_jobs={args.n_jobs} parallelism → ETA ~{wall_min} min wall time "
              f"({args.trials * 17} CPU-min)")
    else:
        print(f"  ETA: ~{args.trials * 17} minutes")
    print()

    try:
        study.optimize(
            lambda t: _objective(t, args.study, training_dfs, validation_dfs),
            n_trials=args.trials,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — study state saved in DB. Resume with --resume.")

    _report_study(study)


if __name__ == "__main__":
    main()
