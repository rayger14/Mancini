---
description: Generate daily regime classification report
argument-hint: [start_date] [end_date]
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Daily Regime Report

Generate a regime classification report showing the current market regime (Bull/Bear/Neutral) and what trade directions are enabled.

## Execution
```bash
cd /Users/raymondghandchi/Mancini/Mancini
python3 -c "
from core.regime_filter import compute_regime, build_daily_bars, RegimeParams
import pandas as pd

df = pd.read_parquet('data/ES_1m_full_session_2021-01-01_2026-02-05.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('US/Eastern')

daily = build_daily_bars(df)
start = '${1:-2025-01-01}'
end = '${2:-2026-02-05}'

print(f'Regime Report: {start} to {end}')
print('=' * 70)
print(f'{\"Date\":<12} {\"Dir\":<8} {\"Vol\":<8} {\"Longs\":<7} {\"Shorts\":<7} {\"Structure\":<12} {\"EMA Slope\":>10}')
print('-' * 70)

for i in range(len(daily)):
    d = daily.index[i]
    if str(d.date()) < start or str(d.date()) > end:
        continue
    window = daily.iloc[:i+1]
    if len(window) < 130:
        continue
    regime = compute_regime(window)
    print(f'{str(d.date()):<12} {regime.direction.name:<8} {regime.vol_regime.name:<8} '
          f'{\"YES\" if regime.longs_enabled else \"NO\":<7} '
          f'{\"YES\" if regime.shorts_enabled else \"NO\":<7} '
          f'{regime.structure:<12} {regime.ema_slope:>+10.2f}')
" 2>&1
```

## Reading the output
- **Dir**: BULL = longs only, BEAR = shorts only, NEUTRAL = both enabled
- **Vol**: LOW/NORMAL/HIGH based on ATR percentile
- **EMA Slope**: Positive = uptrend, negative = downtrend
- **Structure**: HH/HL = bull, LH/LL = bear, expanding/contracting/neutral = range
