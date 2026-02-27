---
description: Run joint regime + short-side Optuna optimization
---

# Joint Regime + Short-Side Optimization

Run the joint optimizer that tunes regime filter AND short-side params together.

## Parameters
- **Regime params** (3): ema_span, slope_lookback, slope_threshold_atr_mult
- **Short params** (8): fr_stop, lj_stop, acceptance, hold, abort, swing, sweep_depth, min_rr

## Usage

```bash
python3 -u backtest/optuna_joint.py --trials ${1:-100} --timeout ${2:-480}
```

## What to look for
- p10 OOS Sharpe > 0.5 (strong), > 0.3 (acceptable)
- Regime params should differ meaningfully from defaults (ema=50, slope=5, threshold=0.15)
- Per-year validation: no catastrophic single year (< -200 pts)
- Total PnL should beat both baselines: no-filter (-548) and EMA-default (+96)
- Fanova: check if regime or short params matter more

## Warm-starting
Automatically warm-starts from `data/optuna_short_results.json` if available.
Results saved to `data/optuna_joint_results.json`.
