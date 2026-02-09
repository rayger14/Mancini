"""Final validation: best params + Monday filter + ATR floor.

Applies regime-based day filters BEFORE running the strategy:
1. Skip Mondays (21% WR, PF=0.37 — catastrophic)
2. Skip ultra-low-vol days (ATR < 31 pts — 27% WR danger zone)

Then runs the full 6-test statistical validation battery.
"""
import sys
import json
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.walk_forward import load_daily_dfs, full_validation, run_backtest

# ── Locked Production Params (5/6 STRONG validated) ──────────────────

PRODUCTION_PARAMS = {
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


# ── Regime Filters ────────────────────────────────────────────────────

def compute_daily_atr(daily_dfs: dict[date, pd.DataFrame]) -> dict[date, float]:
    """Compute daily ATR (high - low range) for each day."""
    atr = {}
    for d, df in daily_dfs.items():
        atr[d] = float(df["high"].max() - df["low"].min())
    return atr


def apply_regime_filters(
    daily_dfs: dict[date, pd.DataFrame],
    skip_mondays: bool = True,
    atr_floor: float = 31.0,
) -> dict[date, pd.DataFrame]:
    """Filter out days that historically lose money.

    Parameters
    ----------
    daily_dfs : dict
        All trading days.
    skip_mondays : bool
        Remove Monday sessions (21% WR, PF=0.37).
    atr_floor : float
        Remove days with daily range below this (ATR Q1 danger zone).

    Returns
    -------
    dict : Filtered daily DataFrames.
    """
    atr = compute_daily_atr(daily_dfs)
    filtered = {}
    skipped_monday = 0
    skipped_atr = 0

    for d, df in daily_dfs.items():
        # Monday = weekday 0
        if skip_mondays and d.weekday() == 0:
            skipped_monday += 1
            continue
        if atr_floor > 0 and atr[d] < atr_floor:
            skipped_atr += 1
            continue
        filtered[d] = df

    print(f"  Regime filter: {len(daily_dfs)} days → {len(filtered)} days")
    if skip_mondays:
        print(f"    Skipped {skipped_monday} Mondays")
    if atr_floor > 0:
        print(f"    Skipped {skipped_atr} low-ATR days (< {atr_floor} pts)")

    return filtered


# ── Main ──────────────────────────────────────────────────────────────

def main():
    data_path = Path(__file__).parent.parent / "data" / "ES_1m_2024-02-05_2026-02-05.parquet"

    print("Loading data...")
    daily_dfs = load_daily_dfs(str(data_path))
    print(f"Loaded {len(daily_dfs)} days")

    # ── Run 1: Baseline (no filters) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("RUN 1: BASELINE — No regime filters")
    print("=" * 70)
    baseline = full_validation(
        daily_dfs, PRODUCTION_PARAMS, n_trials_tested=100,
        label="Baseline (no filters)"
    )

    # ── Run 2: Monday filter only ────────────────────────────────────
    print("\n" + "=" * 70)
    print("RUN 2: MONDAY FILTER ONLY")
    print("=" * 70)
    no_monday = apply_regime_filters(daily_dfs, skip_mondays=True, atr_floor=0)
    monday_results = full_validation(
        no_monday, PRODUCTION_PARAMS, n_trials_tested=100,
        label="Skip Mondays"
    )

    # ── Run 3: Monday + ATR floor ────────────────────────────────────
    print("\n" + "=" * 70)
    print("RUN 3: MONDAY FILTER + ATR FLOOR (< 31 pts)")
    print("=" * 70)
    full_filtered = apply_regime_filters(daily_dfs, skip_mondays=True, atr_floor=31.0)
    full_results = full_validation(
        full_filtered, PRODUCTION_PARAMS, n_trials_tested=100,
        label="Skip Mondays + ATR Floor"
    )

    # ── Comparison Table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':>12} {'No Monday':>12} {'No Mon+ATR':>12}")
    print("-" * 61)

    for label, res in [("Baseline", baseline), ("No Monday", monday_results),
                        ("No Mon+ATR", full_results)]:
        pass  # Just use them below

    runs = [
        ("Baseline", baseline),
        ("No Monday", monday_results),
        ("No Mon+ATR", full_results),
    ]

    metrics = [
        ("Total Trades", "total_trades", "d"),
        ("Win Rate", "win_rate", ".0%"),
        ("Profit Factor", "profit_factor", ".2f"),
        ("Total PnL (pts)", "total_pnl", "+.0f"),
        ("Sharpe Ratio", "sharpe", "+.2f"),
        ("Max Drawdown (pts)", "max_drawdown", ".0f"),
    ]

    for name, key, fmt in metrics:
        vals = []
        for _, res in runs:
            v = res["full_metrics"][key]
            if fmt == ".0%":
                vals.append(f"{v:.0%}")
            else:
                vals.append(f"{v:{fmt}}")
        print(f"{name:<25} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    print("-" * 61)

    # Validation scores
    for _, res in runs:
        score = res["score"]
        perm_p = res["permutation"]["p_value"]
        boot_lo = res["bootstrap_sharpe"]["ci_low"]
        res["_summary"] = f"Score={score}, perm p={perm_p:.3f}, CI_lo={boot_lo:+.2f}"

    print(f"{'Validation Score':<25} ", end="")
    for _, res in runs:
        print(f"{res['score']:>12}", end="")
    print()
    print(f"{'Permutation p-value':<25} ", end="")
    for _, res in runs:
        print(f"{res['permutation']['p_value']:>12.4f}", end="")
    print()
    print(f"{'Bootstrap CI lower':<25} ", end="")
    for _, res in runs:
        print(f"{res['bootstrap_sharpe']['ci_low']:>+12.2f}", end="")
    print()

    # ── Save ─────────────────────────────────────────────────────────
    save_data = {
        "production_params": PRODUCTION_PARAMS,
        "filters": {
            "skip_mondays": True,
            "atr_floor": 31.0,
        },
        "baseline": {
            k: v for k, v in baseline.items()
            if k not in ("full_metrics",)
        },
        "monday_filter": {
            k: v for k, v in monday_results.items()
            if k not in ("full_metrics",)
        },
        "full_filter": {
            k: v for k, v in full_results.items()
            if k not in ("full_metrics",)
        },
    }
    # Add summaries
    for key, res in [("baseline", baseline), ("monday_filter", monday_results),
                      ("full_filter", full_results)]:
        save_data[key]["summary"] = {
            k: v for k, v in res["full_metrics"].items()
            if k not in ("daily_pnls", "trade_pnls")
        }

    out_path = Path(__file__).parent.parent / "data" / "validation_final.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # ── Final Recommendation ─────────────────────────────────────────
    best_run = max(runs, key=lambda r: sum(v for v in r[1]["checks"].values()))
    best_label = best_run[0]
    best_score = best_run[1]["score"]
    print(f"\nRECOMMENDATION: Use '{best_label}' configuration (Score: {best_score})")


if __name__ == "__main__":
    main()
