---
description: Run Optuna regime filter parameter optimization
argument-hint: [trials] [timeout_min]
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Regime Filter Parameter Optimization

Optimize the EMA slope + ATR regime filter parameters jointly with strategy params using Optuna TPE + HyperbandPruner on 5-year ES data.

## Parameters to Optimize
- `ema_span`: EMA period for trend detection (range: 20-100)
- `slope_lookback`: Days to measure EMA slope (range: 3-15)
- `slope_threshold_atr_mult`: Slope threshold as ATR multiple (range: 0.05-0.40)
- `atr_period`: ATR calculation period (range: 7-21)
- `vol_percentile_window`: Rolling window for vol percentile (range: 63-252)
- `vol_high_threshold`: Percentile threshold for HIGH vol (range: 0.65-0.90)
- `vol_low_threshold`: Percentile threshold for LOW vol (range: 0.10-0.35)

## Execution
```bash
cd /Users/raymondghandchi/Mancini/Mancini
.venv311/bin/python3 -u backtest/optuna_regime.py \
  --trials ${1:-80} \
  --timeout ${2:-90} \
  2>&1
```

## Key metrics
- Filter accuracy: % of removed trades that were losers (target: >80%)
- Regime flip frequency: 10-25/year is ideal (too few = slow adaptation, too many = whipsaws)
- PnL improvement over no-filter baseline
- Must remain profitable on BOTH 2024-2026 (bull, where longs dominate) AND 2022 (bear, where shorts dominate)
