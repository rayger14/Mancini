"""Validate previous best Optuna params with full statistical battery."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.walk_forward import load_daily_dfs, full_validation

PREV_BEST_PARAMS = {
    "acceptance_max_dip_pts": 3.0,
    "acceptance_min_hold_bars": 7,
    "acceptance_min_hold_bars_deep": 8,
    "chop_end_hour": 15,
    "chop_start_hour": 13,
    "fb_stop_buffer": 5.5,
    "higher_low_lookback": 4,
    "level_reclaim_min_touches": 4,
    "lr_stop_buffer": 5.0,
    "max_trades_per_day": 4,
    "min_levels_broken": 2,
    "min_rr_ratio": 1.0,
    "min_velocity": 0.75,
    "multi_hour_rally_min_pts": 22.5,
    "non_acceptance_min_recovery_pts": 5.0,
    "swing_low_order": 15,
    "t1_exit_fraction": 1.0,
    "trailing_stop_pts": 7.0,
    "true_breakdown_abort_bars": 12,
}

def main():
    data_path = Path(__file__).parent.parent / "data" / "ES_1m_2024-02-05_2026-02-05.parquet"

    print("Loading data...")
    daily_dfs = load_daily_dfs(str(data_path))
    print(f"Loaded {len(daily_dfs)} days")

    print("\nPrevious best Optuna params:")
    for k, v in sorted(PREV_BEST_PARAMS.items()):
        print(f"  {k}: {v}")

    results = full_validation(
        daily_dfs, PREV_BEST_PARAMS, n_trials_tested=100, label="Previous Optuna Best"
    )

    # Save results
    out_path = Path(__file__).parent.parent / "data" / "validation_prev_best.json"
    save_results = {
        k: v for k, v in results.items()
        if k not in ("full_metrics",)
    }
    save_results["full_summary"] = {
        k: v for k, v in results["full_metrics"].items()
        if k not in ("daily_pnls", "trade_pnls")
    }
    save_results["params"] = PREV_BEST_PARAMS
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
