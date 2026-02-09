"""Reduced-parameter Optuna optimization with walk-forward validation.

Only optimizes ~10 high-impact parameters (vs 19+ in run_optuna.py) to reduce
overfitting risk. Remaining params are fixed at Mancini domain knowledge defaults.

After optimization, runs full_validation() from walk_forward.py on the best params.
"""
import sys
import json
import time
from datetime import date, time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import optuna
from loguru import logger

# Suppress all logging for speed
logger.remove()
optuna.logging.set_verbosity(optuna.logging.WARNING)

from backtest.runner import BacktestRunner
from backtest.walk_forward import full_validation, load_daily_dfs
from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    RiskParams, SessionTimes,
)

# Fixed params from Mancini domain knowledge (not optimized)
FIXED_PARAMS = {
    "acceptance_min_hold_bars": 3,
    "acceptance_min_hold_bars_deep": 8,
    "acceptance_max_dip_pts": 3.0,
    "non_acceptance_min_recovery_pts": 5.0,
    "true_breakdown_abort_bars": 10,
    "higher_low_lookback": 4,
    "level_reclaim_min_touches": 4,
    "multi_hour_rally_min_pts": 15.0,
    "min_velocity": 0.75,
    "require_volume": False,
    "volume_spike_threshold": 1.5,
    "contracts": 4,
}


def build_full_params(optimized: dict) -> dict:
    """Merge optimized params with fixed defaults."""
    params = dict(FIXED_PARAMS)
    params.update(optimized)
    return params


def run_backtest(daily_dfs, params):
    """Run backtest with given params, return metrics dict."""
    strategy = StrategyParams(
        swing_low_order=params["swing_low_order"],
        multi_hour_rally_min_pts=params["multi_hour_rally_min_pts"],
        level_reclaim_min_touches=params["level_reclaim_min_touches"],
        acceptance_min_hold_bars=params["acceptance_min_hold_bars"],
        acceptance_min_hold_bars_deep=params["acceptance_min_hold_bars_deep"],
        acceptance_max_dip_pts=params["acceptance_max_dip_pts"],
        true_breakdown_abort_bars=params["true_breakdown_abort_bars"],
        fb_stop_buffer_pts=params["fb_stop_buffer"],
        lr_stop_buffer_pts=params["lr_stop_buffer"],
        non_acceptance_min_recovery_pts=params["non_acceptance_min_recovery_pts"],
    )
    elevator = ElevatorParams(
        min_velocity_pts_per_min=params["min_velocity"],
        min_levels_broken=params["min_levels_broken"],
        higher_low_lookback=params["higher_low_lookback"],
    )
    exit_params = ExitParams(
        t1_exit_fraction=params["t1_exit_fraction"],
        trailing_stop_pts=params["trailing_stop_pts"],
        default_contracts=params.get("contracts", 4),
    )
    risk = RiskParams(max_trades_per_day=params["max_trades_per_day"])
    session = SessionTimes(
        chop_zone_start=dtime(params["chop_start_hour"], 0),
        chop_zone_end=dtime(params["chop_end_hour"], 0),
    )

    runner = BacktestRunner(
        strategy_params=strategy,
        elevator_params=elevator,
        exit_params=exit_params,
        risk_params=risk,
        session_times=session,
        min_rr_ratio=params["min_rr_ratio"],
    )
    if params.get("require_volume", False):
        runner.strategy.signal_aggregator.require_volume_confirmation = True
        runner.strategy.signal_aggregator._volume_spike_threshold = params.get(
            "volume_spike_threshold", 1.5
        )
    result = runner.run_multi_day(daily_dfs=daily_dfs)

    daily_pnls = [d.pnl_pts for d in result.days]
    mean_d = np.mean(daily_pnls) if daily_pnls else 0
    std_d = np.std(daily_pnls) if daily_pnls else 1
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else 0

    fb_trades = sum(1 for t in result.all_trades if t.pattern_type == "failed_breakdown")
    lr_trades = sum(1 for t in result.all_trades if t.pattern_type == "level_reclaim")

    wins = [t.pnl_pts for t in result.all_trades if t.pnl_pts > 0]
    losses = [t.pnl_pts for t in result.all_trades if t.pnl_pts <= 0]

    return {
        "sharpe": sharpe,
        "total_pnl": result.total_pnl_pts,
        "total_trades": result.total_trades,
        "fb_trades": fb_trades,
        "lr_trades": lr_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "max_drawdown": result.max_drawdown_pts,
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "expectancy": mean_d,
    }


def objective(trial, train_dfs):
    """Optuna objective: maximize risk-adjusted returns with 10 free params."""
    optimized = {
        # Stop buffers (high impact on win rate)
        "fb_stop_buffer": trial.suggest_float("fb_stop_buffer", 1.0, 6.0, step=0.5),
        "lr_stop_buffer": trial.suggest_float("lr_stop_buffer", 1.0, 5.0, step=0.5),
        # Level detection
        "swing_low_order": trial.suggest_int("swing_low_order", 10, 60, step=5),
        # Elevator
        "min_levels_broken": trial.suggest_int("min_levels_broken", 0, 2),
        # Exit management
        "t1_exit_fraction": trial.suggest_float("t1_exit_fraction", 0.5, 1.0, step=0.25),
        "min_rr_ratio": trial.suggest_float("min_rr_ratio", 0.25, 2.0, step=0.25),
        "trailing_stop_pts": trial.suggest_float("trailing_stop_pts", 2.0, 8.0, step=1.0),
        # Risk/session
        "max_trades_per_day": trial.suggest_int("max_trades_per_day", 1, 5),
        "chop_start_hour": trial.suggest_int("chop_start_hour", 11, 15),
        "chop_end_hour": trial.suggest_int("chop_end_hour", 13, 16),
    }

    # Validate chop zone
    if optimized["chop_end_hour"] <= optimized["chop_start_hour"]:
        return float("-inf")

    params = build_full_params(optimized)
    metrics = run_backtest(train_dfs, params)

    # Minimum trade count (need statistical evidence)
    if metrics["total_trades"] < 30:
        return float("-inf")

    if metrics["win_rate"] < 0.10:
        return float("-inf")

    # Primary: Sharpe ratio
    score = metrics["sharpe"]

    # Penalty for very few trades
    if metrics["total_trades"] < 50:
        score -= 1.0

    # Bonus for higher win rate
    if metrics["win_rate"] > 0.40:
        score += 1.0
    elif metrics["win_rate"] > 0.30:
        score += 0.5

    # Bonus for positive expectancy
    if metrics["expectancy"] > 0:
        score += 0.5

    # Log metrics
    trial.set_user_attr("trades", metrics["total_trades"])
    trial.set_user_attr("wr", round(metrics["win_rate"], 3))
    trial.set_user_attr("pf", round(metrics["profit_factor"], 2))
    trial.set_user_attr("pnl", round(metrics["total_pnl"], 1))
    trial.set_user_attr("fb", metrics["fb_trades"])
    trial.set_user_attr("lr", metrics["lr_trades"])

    return score


def main():
    data_path = Path(__file__).parent.parent / "data" / "ES_1m_2024-02-05_2026-02-05.parquet"
    print("Loading data...")
    daily_dfs = load_daily_dfs(str(data_path))

    # Walk-forward split: 70/30
    dates = sorted(daily_dfs.keys())
    split = int(len(dates) * 0.70)
    train_dfs = {d: daily_dfs[d] for d in dates[:split]}
    test_dfs = {d: daily_dfs[d] for d in dates[split:]}
    print(f"Total: {len(daily_dfs)} days | Train: {len(train_dfs)} | Test: {len(test_dfs)}")
    print(f"\nReduced parameter optimization: 10 free params (vs 19+ in full optimizer)")
    print(f"Fixed params: {list(FIXED_PARAMS.keys())}")

    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15),
    )

    best_so_far = float("-inf")
    start_time = time.time()

    def callback(study, trial):
        nonlocal best_so_far
        elapsed = time.time() - start_time
        n = trial.number + 1
        per_trial = elapsed / n
        remaining = per_trial * (n_trials - n)

        val = trial.value if trial.value is not None else float("-inf")
        if val > best_so_far:
            best_so_far = val
            wr = trial.user_attrs.get("wr", 0)
            pf = trial.user_attrs.get("pf", 0)
            trades = trial.user_attrs.get("trades", 0)
            pnl = trial.user_attrs.get("pnl", 0)
            print(
                f"  [{n:3d}/{n_trials}] NEW BEST score={val:+.2f} | "
                f"{trades}T WR={wr*100:.0f}% PF={pf:.2f} PnL={pnl:+.0f} | "
                f"{per_trial:.0f}s/trial, ~{remaining/60:.0f}min left",
                flush=True,
            )
        elif n % 10 == 0:
            print(
                f"  [{n:3d}/{n_trials}] best={best_so_far:+.2f} | "
                f"{per_trial:.0f}s/trial, ~{remaining/60:.0f}min left",
                flush=True,
            )

    print(f"\nRunning {n_trials} Optuna trials on {len(train_dfs)} training days...")
    study.optimize(
        lambda trial: objective(trial, train_dfs),
        n_trials=n_trials,
        callbacks=[callback],
    )

    # Results
    best = study.best_trial
    best_full_params = build_full_params(best.params)

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS (REDUCED PARAMS)")
    print("=" * 70)
    print(f"Best score: {best.value:+.3f}")
    print(f"\nOptimized params ({len(best.params)}):")
    for k, v in sorted(best.params.items()):
        print(f"  {k}: {v}")
    print(f"\nFixed params ({len(FIXED_PARAMS)}):")
    for k, v in sorted(FIXED_PARAMS.items()):
        print(f"  {k}: {v}")

    # Train metrics
    print("\n--- Train Performance ---")
    train_metrics = run_backtest(train_dfs, best_full_params)
    for k, v in sorted(train_metrics.items()):
        print(f"  {k}: {v:+.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # Test (out-of-sample) validation
    print("\n--- Test Performance (OUT-OF-SAMPLE) ---")
    test_metrics = run_backtest(test_dfs, best_full_params)
    for k, v in sorted(test_metrics.items()):
        print(f"  {k}: {v:+.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # Overfitting check
    train_s = train_metrics["sharpe"]
    test_s = test_metrics["sharpe"]
    deg = (train_s - test_s) / abs(train_s) * 100 if train_s != 0 else 0
    print(f"\n--- Overfitting Check ---")
    print(f"Train Sharpe: {train_s:+.2f}")
    print(f"Test Sharpe:  {test_s:+.2f}")
    print(f"Degradation:  {deg:.1f}%")
    if test_s > 0:
        print("PASS: Profitable out-of-sample!")
    elif deg < 50:
        print("MARGINAL: Some overfitting")
    else:
        print("FAIL: Significant overfitting")

    # Full statistical validation on best params
    print("\n\n" + "#" * 70)
    print("FULL STATISTICAL VALIDATION (all 508 days)")
    print("#" * 70)
    validation = full_validation(
        daily_dfs, best_full_params,
        n_trials_tested=n_trials,
        label=f"Reduced Optuna Best ({n_trials} trials, 10 params)",
    )

    # Save results
    results = {
        "optimized_params": best.params,
        "fixed_params": FIXED_PARAMS,
        "full_params": best_full_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "n_trials": n_trials,
        "n_free_params": len(best.params),
        "overfitting_degradation_pct": round(deg, 1),
        "validation": {
            k: v for k, v in validation.items()
            if k not in ("full_metrics",)
        },
        "validation_full_summary": {
            k: v for k, v in validation["full_metrics"].items()
            if k not in ("daily_pnls", "trade_pnls")
        },
    }
    out_path = Path(__file__).parent.parent / "data" / "optuna_reduced_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Top 5
    completed = [t for t in study.trials if t.value is not None and t.value > float("-inf")]
    completed.sort(key=lambda t: t.value, reverse=True)
    print(f"\nTop 5 trials:")
    for t in completed[:5]:
        print(
            f"  #{t.number}: score={t.value:+.2f} "
            f"trades={t.user_attrs.get('trades', '?')} "
            f"WR={t.user_attrs.get('wr', 0)*100:.0f}% "
            f"PF={t.user_attrs.get('pf', 0):.2f} "
            f"PnL={t.user_attrs.get('pnl', 0):+.0f}"
        )


if __name__ == "__main__":
    main()
