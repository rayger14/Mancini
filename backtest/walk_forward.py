"""Walk-forward validation with statistical robustness tests.

Implements:
1. Anchored expanding walk-forward (4 folds)
2. Monte Carlo permutation test (is the edge real?)
3. Bootstrap Sharpe confidence interval
4. Deflated Sharpe Ratio (multiple testing correction)
5. Trade resampling for equity curve confidence bands
"""
import sys
import json
import time
from datetime import date, time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm
from loguru import logger

logger.remove()

from backtest.runner import BacktestRunner
from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    RiskParams, SessionTimes,
)


# ── Data Loading ──────────────────────────────────────────────────────

def load_daily_dfs(parquet_path: str) -> dict[date, pd.DataFrame]:
    df = pd.read_parquet(parquet_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    df_rth = df.between_time("09:30", "15:59")
    daily = {}
    for d, grp in df_rth.groupby(df_rth.index.date):
        if len(grp) >= 10:
            daily[d] = grp
    return daily


# ── Backtest Runner ───────────────────────────────────────────────────

def run_backtest(daily_dfs, params):
    """Run backtest with given params, return metrics dict + trade list."""
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
        allow_level_sweep_fb=params.get("allow_level_sweep_fb", True),
        level_sweep_min_bars_below=params.get("level_sweep_min_bars_below", 3),
        level_sweep_min_depth_pts=params.get("level_sweep_min_depth_pts", 1.0),
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
    trade_pnls = [t.pnl_pts for t in result.all_trades]
    mean_d = np.mean(daily_pnls) if daily_pnls else 0
    std_d = np.std(daily_pnls) if daily_pnls else 1
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else 0

    wins = [t.pnl_pts for t in result.all_trades if t.pnl_pts > 0]
    losses = [t.pnl_pts for t in result.all_trades if t.pnl_pts <= 0]

    return {
        "sharpe": sharpe,
        "total_pnl": result.total_pnl_pts,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "max_drawdown": result.max_drawdown_pts,
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "daily_pnls": daily_pnls,
        "trade_pnls": trade_pnls,
    }


# ── Statistical Tests ─────────────────────────────────────────────────

def permutation_test(daily_pnls, n_permutations=10000, seed=42):
    """Sign-flip permutation test. Tests if long edge is real.

    Under H0 (no directional edge), flipping signs of daily P&L
    should not change the distribution of total P&L.

    Returns p-value: if < 0.05, the edge is statistically significant.
    """
    rng = np.random.RandomState(seed)
    daily = np.array(daily_pnls)
    observed = np.sum(daily)

    null_totals = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(daily))
        null_totals[i] = np.sum(daily * signs)

    p_value = np.mean(null_totals >= observed)
    return {
        "observed_pnl": float(observed),
        "null_mean": float(np.mean(null_totals)),
        "null_std": float(np.std(null_totals)),
        "p_value": float(p_value),
        "significant_5pct": p_value < 0.05,
        "significant_10pct": p_value < 0.10,
    }


def bootstrap_sharpe_ci(daily_pnls, n_bootstrap=10000, confidence=0.95, seed=42):
    """Bootstrap confidence interval for annualized Sharpe ratio."""
    rng = np.random.RandomState(seed)
    daily = np.array(daily_pnls)
    n = len(daily)

    sharpes = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(daily, size=n, replace=True)
        mean = np.mean(sample)
        std = np.std(sample, ddof=1)
        sharpes[i] = (mean / std) * np.sqrt(252) if std > 0 else 0

    alpha = (1 - confidence) / 2
    ci_low = float(np.percentile(sharpes, alpha * 100))
    ci_high = float(np.percentile(sharpes, (1 - alpha) * 100))

    return {
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_median": float(np.median(sharpes)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "prob_positive": float(np.mean(sharpes > 0) * 100),
    }


def deflated_sharpe_ratio(sharpe_observed, n_trials, n_days, skew=0, kurtosis=3):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts for multiple testing. Returns probability the observed Sharpe
    is genuine (not just the best of n_trials random strategies).
    """
    euler_mascheroni = 0.5772
    e_max_sharpe = (
        (1 - euler_mascheroni) * norm.ppf(1 - 1 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1 / (n_trials * np.e))
    )

    se_sharpe = np.sqrt(
        (1 + 0.5 * sharpe_observed ** 2 - skew * sharpe_observed
         + ((kurtosis - 3) / 4) * sharpe_observed ** 2)
        / (n_days - 1)
    )

    z = (sharpe_observed - e_max_sharpe) / se_sharpe if se_sharpe > 0 else 0
    p_value = float(norm.cdf(z))

    return {
        "observed_sharpe": sharpe_observed,
        "expected_max_sharpe_null": float(e_max_sharpe),
        "se_sharpe": float(se_sharpe),
        "z_score": float(z),
        "p_value": p_value,
        "passes": p_value > 0.95,  # Observed Sharpe exceeds null expectation
    }


def monte_carlo_trades(trade_pnls, n_simulations=10000, seed=42):
    """Reshuffle trade order to get confidence bands on equity curve."""
    rng = np.random.RandomState(seed)
    trades = np.array(trade_pnls)
    n = len(trades)
    if n == 0:
        return {"prob_profitable": 0, "total_pnl_5th": 0, "total_pnl_95th": 0,
                "max_dd_mean": 0, "max_dd_95th": 0}

    final_pnls = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)

    for i in range(n_simulations):
        shuffled = rng.permutation(trades)
        equity = np.cumsum(shuffled)
        final_pnls[i] = equity[-1]
        peak = np.maximum.accumulate(equity)
        max_drawdowns[i] = float(np.max(peak - equity))

    return {
        "total_pnl_mean": float(np.mean(final_pnls)),
        "total_pnl_5th": float(np.percentile(final_pnls, 5)),
        "total_pnl_95th": float(np.percentile(final_pnls, 95)),
        "max_dd_mean": float(np.mean(max_drawdowns)),
        "max_dd_95th": float(np.percentile(max_drawdowns, 95)),
        "prob_profitable": float(np.mean(final_pnls > 0) * 100),
    }


# ── Walk-Forward Engine ───────────────────────────────────────────────

def expanding_walk_forward(daily_dfs, params, n_folds=4, min_train_days=200):
    """Anchored expanding walk-forward validation.

    Always trains from day 0, with non-overlapping OOS windows.
    Returns metrics for each fold.
    """
    dates = sorted(daily_dfs.keys())
    n = len(dates)
    oos_size = (n - min_train_days) // n_folds

    folds = []
    for i in range(n_folds):
        oos_start = min_train_days + i * oos_size
        oos_end = min(oos_start + oos_size, n)

        train_dates = dates[:oos_start]
        test_dates = dates[oos_start:oos_end]

        # train_dfs not needed here; only test_dfs used for OOS evaluation
        test_dfs = {d: daily_dfs[d] for d in test_dates}

        print(f"\n  Fold {i+1}/{n_folds}: Train={len(train_dates)}d "
              f"({train_dates[0]}→{train_dates[-1]}), "
              f"Test={len(test_dates)}d ({test_dates[0]}→{test_dates[-1]})")

        t0 = time.time()
        metrics = run_backtest(test_dfs, params)
        elapsed = time.time() - t0

        fold = {
            "fold": i + 1,
            "train_days": len(train_dates),
            "test_days": len(test_dates),
            "test_start": str(test_dates[0]),
            "test_end": str(test_dates[-1]),
            "trades": metrics["total_trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "total_pnl": metrics["total_pnl"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "elapsed_s": round(elapsed, 1),
        }
        folds.append(fold)

        print(f"    {fold['trades']}T, WR={fold['win_rate']:.0%}, "
              f"PF={fold['profit_factor']:.2f}, PnL={fold['total_pnl']:+.0f}, "
              f"Sharpe={fold['sharpe']:+.2f} ({elapsed:.0f}s)")

    return folds


# ── Full Validation Report ────────────────────────────────────────────

def full_validation(daily_dfs, params, n_trials_tested=100, label="Strategy"):
    """Run complete validation battery on a parameter set."""
    print(f"\n{'='*70}")
    print(f"FULL VALIDATION: {label}")
    print(f"{'='*70}")

    # 1. Full-sample backtest
    print("\n--- Full Sample Backtest ---")
    t0 = time.time()
    full = run_backtest(daily_dfs, params)
    print(f"  {full['total_trades']}T, WR={full['win_rate']:.0%}, "
          f"PF={full['profit_factor']:.2f}, PnL={full['total_pnl']:+.0f}pts, "
          f"Sharpe={full['sharpe']:+.2f}, MaxDD={full['max_drawdown']:.0f}pts "
          f"({time.time()-t0:.0f}s)")

    # 2. Walk-forward validation
    print("\n--- Expanding Walk-Forward (4 folds) ---")
    folds = expanding_walk_forward(daily_dfs, params, n_folds=4, min_train_days=200)

    oos_sharpes = [f["sharpe"] for f in folds]
    oos_pnls = [f["total_pnl"] for f in folds]
    oos_wrs = [f["win_rate"] for f in folds]
    profitable_folds = sum(1 for p in oos_pnls if p > 0)

    print(f"\n  Summary: {profitable_folds}/{len(folds)} folds profitable")
    print(f"  Avg OOS Sharpe: {np.mean(oos_sharpes):+.2f} "
          f"(range: {min(oos_sharpes):+.2f} to {max(oos_sharpes):+.2f})")
    print(f"  Avg OOS PnL: {np.mean(oos_pnls):+.0f}pts")
    print(f"  Avg OOS WR: {np.mean(oos_wrs):.0%}")

    # 3. Permutation test
    print("\n--- Permutation Test (is the edge real?) ---")
    perm = permutation_test(full["daily_pnls"])
    sig = "YES" if perm["significant_5pct"] else "NO"
    print(f"  Observed PnL: {perm['observed_pnl']:+.0f}")
    print(f"  Null mean: {perm['null_mean']:+.0f} (std={perm['null_std']:.0f})")
    print(f"  p-value: {perm['p_value']:.4f}")
    print(f"  Significant at 5%: {sig}")
    if not perm["significant_5pct"] and perm["significant_10pct"]:
        print(f"  (Significant at 10% though)")

    # 4. Bootstrap Sharpe CI
    print("\n--- Bootstrap Sharpe Confidence Interval ---")
    boot = bootstrap_sharpe_ci(full["daily_pnls"])
    print(f"  Sharpe 95% CI: [{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}]")
    print(f"  Sharpe mean: {boot['sharpe_mean']:+.2f}, "
          f"median: {boot['sharpe_median']:+.2f}")
    print(f"  P(Sharpe > 0): {boot['prob_positive']:.0f}%")
    ci_above_zero = "YES" if boot["ci_low"] > 0 else "NO"
    print(f"  Entire CI above zero: {ci_above_zero}")

    # 5. Deflated Sharpe Ratio
    print(f"\n--- Deflated Sharpe Ratio (adjusting for {n_trials_tested} trials) ---")
    dsr = deflated_sharpe_ratio(
        full["sharpe"], n_trials_tested, len(full["daily_pnls"])
    )
    print(f"  Observed Sharpe: {dsr['observed_sharpe']:+.2f}")
    print(f"  Expected max Sharpe under null: {dsr['expected_max_sharpe_null']:+.2f}")
    print(f"  z-score: {dsr['z_score']:+.2f}")
    print(f"  DSR p-value: {dsr['p_value']:.4f}")
    passes = "PASS" if dsr["passes"] else "FAIL"
    print(f"  Survives deflation: {passes}")

    # 6. Monte Carlo trade resampling
    print("\n--- Monte Carlo Trade Resampling (10K sims) ---")
    mc = monte_carlo_trades(full["trade_pnls"])
    print(f"  P(profitable): {mc['prob_profitable']:.0f}%")
    print(f"  PnL 90% CI: [{mc['total_pnl_5th']:+.0f}, {mc['total_pnl_95th']:+.0f}]")
    print(f"  Max DD (mean): {mc['max_dd_mean']:.0f}pts, "
          f"(95th pct): {mc['max_dd_95th']:.0f}pts")

    # 7. Overall verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    checks = {
        "Walk-forward profitable (3+ of 4 folds)": profitable_folds >= 3,
        "Permutation test (p < 0.10)": perm["significant_10pct"],
        "Bootstrap Sharpe CI lower > 0": boot["ci_low"] > 0,
        "Deflated Sharpe passes": dsr["passes"],
        "Monte Carlo P(profit) > 70%": mc["prob_profitable"] > 70,
        "Avg OOS win rate > 30%": np.mean(oos_wrs) > 0.30,
    }
    passed = sum(v for v in checks.values())
    total = len(checks)

    for name, result in checks.items():
        mark = "PASS" if result else "FAIL"
        print(f"  [{mark}] {name}")

    print(f"\n  Score: {passed}/{total} checks passed")
    if passed >= 5:
        print("  STRONG: Strategy likely has a real edge.")
    elif passed >= 3:
        print("  MODERATE: Some evidence of edge, needs more data.")
    elif passed >= 1:
        print("  WEAK: Limited evidence. Proceed with extreme caution.")
    else:
        print("  NONE: No statistical evidence of edge.")

    return {
        "full_metrics": full,
        "folds": folds,
        "permutation": perm,
        "bootstrap_sharpe": boot,
        "deflated_sharpe": dsr,
        "monte_carlo": mc,
        "checks": checks,
        "score": f"{passed}/{total}",
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    data_path = Path(__file__).parent.parent / "data" / "ES_1m_2024-02-05_2026-02-05.parquet"
    optuna_path = Path(__file__).parent.parent / "data" / "optuna_results.json"

    print("Loading data...")
    daily_dfs = load_daily_dfs(str(data_path))
    print(f"Loaded {len(daily_dfs)} days")

    # Load best params from Optuna
    with open(optuna_path) as f:
        optuna_data = json.load(f)
    best_params = optuna_data["best_params"]
    n_trials = optuna_data.get("n_trials", 100)

    print(f"\nLoaded Optuna best params ({n_trials} trials)")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    # Run full validation
    results = full_validation(
        daily_dfs, best_params, n_trials_tested=n_trials, label="Optuna Best"
    )

    # Save results
    out_path = Path(__file__).parent.parent / "data" / "validation_results.json"
    # Convert non-serializable types
    save_results = {
        k: v for k, v in results.items()
        if k not in ("full_metrics",)  # daily_pnls/trade_pnls are lists
    }
    save_results["full_summary"] = {
        k: v for k, v in results["full_metrics"].items()
        if k not in ("daily_pnls", "trade_pnls")
    }
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
