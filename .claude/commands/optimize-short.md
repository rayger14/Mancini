---
description: Run Optuna short-side parameter optimization with CPCV
argument-hint: [trials] [timeout_min]
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Short-Side Parameter Optimization

Run Optuna optimization for Failed Rally (FR) and Level Rejection (LJ) short-side parameters using TPE + HyperbandPruner with CPCV cross-validation on 5-year ES data.

## Parameters to Optimize
- `fr_stop_buffer_pts`: FR stop distance below resistance level (range: 3.0-8.0)
- `lj_stop_buffer_pts`: LJ stop distance below resistance level (range: 3.0-8.0)
- `short_swing_high_order`: Swing high detection lookback (range: 5-25)
- `short_acceptance_max_dip_pts`: Max rally above level for acceptance (range: 2.0-6.0)
- `short_acceptance_min_hold_bars`: Min bars holding above level (range: 4-12)
- `short_abort_bars`: True rally abort threshold (range: 8-30)
- `min_rr_ratio_short`: Min risk:reward for short entries (range: 0.8-2.5)

## Execution
```bash
cd /Users/raymondghandchi/Mancini/Mancini
.venv311/bin/python3 -u backtest/optuna_short.py \
  --trials ${1:-100} \
  --timeout ${2:-120} \
  2>&1
```

## What to look for
- 10th percentile OOS Sharpe > 0 (robustness across regimes)
- PBO < 0.30 (probability of backtest overfitting)
- FR should be profitable in 2022 bear market (the key validation year)
- Check Fanova parameter importance — if one param dominates >60%, the others may be noise

## After optimization
1. Run `/backtest-5yr` with winning params to validate
2. Check per-year breakdown — no single year worse than -300 pts
3. Compare to long-only baseline to ensure shorts ADD value, not just noise
