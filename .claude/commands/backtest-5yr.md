---
description: Run full 5-year long+short+regime backtest
argument-hint: [config_name]
allowed-tools: Bash, Read, Write, Glob, Grep
---

# 5-Year Full Validation Backtest

Run the complete Mancini strategy (long FB/LR + short FR/LJ + regime filter) on 5 years of ES data (Jan 2021 - Feb 2026) with all production filters.

## Configuration
Uses the config specified by $1 (default: "production") from `data/configs/`.
Falls back to current production params if no config file found.

## Execution
```bash
cd /Users/raymondghandchi/Mancini/Mancini
python3 -u backtest/five_year_long_short.py \
  --config ${1:-production} \
  --regime-filter ema \
  --skip-mondays \
  --chop-zone 13-15 \
  --evening-block 17-22 \
  2>&1
```

## Output includes
- Total PnL, PF, WR, MaxDD, Sharpe
- Per-direction breakdown (LONG vs SHORT)
- Per-pattern breakdown (FB, LR, FR, LJ)
- Per-year breakdown (2021-2026)
- Per-window breakdown (Morning, Afternoon, Late Night, Pre-RTH)
- Regime distribution and filter effectiveness
- OOS vs IS comparison (2021-2024 vs 2024-2026)

## Success criteria
- Total PnL positive across full 5 years
- No single year worse than -300 pts
- PF > 1.10
- Shorts contribute positively in 2022 bear market
- Regime filter removes >80% losers from filtered trades
- Longs still profitable in 2024-2026 (no regression from adding shorts/regime)
