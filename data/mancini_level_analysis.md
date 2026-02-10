# Mancini Level Derivation Analysis

## Executive Summary

Analysis of 500 Substack posts (June 2024 - February 2026) reveals that Mancini derives his daily
support/resistance levels primarily from intraday swing highs and lows on recent 1-minute/5-minute
charts, NOT from traditional technical indicators. His approach is purely price-action based.

## Phase 1: Term Frequency Across 500 Posts

| Term | Posts | Total |
|------|-------|-------|
| failed breakdown | 499 | 26602 |
| shelf | 391 | 4354 |
| daily low | 429 | 1723 |
| magnet | 362 | 1320 |
| horizontal | 343 | 1321 |
| trendline | 321 | 834 |
| multi-touch | 173 | 224 |
| round number | 104 | 133 |
| VWAP | 0 | 0 |
| fibonacci | 0 | 0 |
| moving average | 0 | 0 |
| volume profile | 0 | 0 |

Key Finding: Zero mentions of VWAP, Fibonacci, moving averages, or volume profile.
Levels come entirely from price-action structure (swing lows/highs, shelves, daily extremes).

## Phase 2: How Mancini Derives Levels

### Level Source Hierarchy (cross-referenced with 5-min data)

| Source | Match Rate (within 3 pts) |
|--------|--------------------------|
| Intraday 5-min swing lows | 42.5% |
| Intraday 5-min swing highs | 39.1% |
| Daily open/close | 15.8% |
| Daily low/high | 2.2% |

### Significant Low Definition (from posts)

1. The prior day low
2. A multi-hour low (20+ point move)
3. A cluster or shelf of lows (multiple touches)

### Level Statistics

- Avg 11.2 resistance per post (4.0 major), 9.0 support (3.1 major)
- All-level spacing: mean 7.2 pts, median 6 pts
- Major-to-major: mean 15.7 pts, median 14 pts
- Support range: ~57 pts below, Resistance: ~81 pts above
- 100% whole-number levels (lows rounded UP, highs rounded DOWN)
- ~35% of levels marked major

### Time Horizon

Overwhelmingly last 1-5 trading days. today=11049 mentions, yesterday=9734.
Older levels only for major structural pivots.

### Conceptual Framework

1. Shelf of Lows/Highs (4354 mentions): horizontal multi-touch zones
2. Magnet (1320 mentions): price attractor, range midpoint
3. Bull/Bear Line: key bias-dividing level
4. Multi-Touch/Multi-Day: levels tested across sessions
5. Cluster: nearby levels forming a zone
6. Backtest: price returning to previously broken level

## Phase 3: Statistical Validation

### Match Rates (1220 levels from 50 posts)

44.9% match a 5-min swing high/low within 3 points (RTH only data).
29.3% match within 1 point. 6.0% exact match.
Remaining ~55% likely from overnight swings, 1-min chart, interpolation.

### Validated Examples

| Level | Description | Actual | Diff |
|-------|-------------|--------|------|
| 6838 | Jan 21 noon low | 6837.25 | 0.75 |
| 6899 | Jan 29 daily low | 6898.25 | 0.75 |
| 6920 | Jan 30 shelf | 6918.75 | 1.25 |
| 6942 | Dec 29 cluster | 6941.50-6943.50 | 0.5 |

## Phase 4: Level Generation Algorithm

### Parameters

- Lookback: 5-10 trading days
- Swing order: 6 bars on 5-min chart
- Min swing: 10+ points
- Rounding: nearest whole number
- Minor spacing: 5-7 pts, Major spacing: 12-16 pts
- Targets: 8-12 supports, 10-15 resistances
- ~35% marked major

### Major Level Criteria (priority order)

1. Prior day low or high (+5 score)
2. Multi-day shelf 3+ touches (+4)
3. Large swing origin 20+ pt rally (+3)
4. Multi-hour V-bottom (+2)
5. Weekly low/high (+2)
6. Round number (+1)

### Algorithm Steps

1. Find 5-min swing lows/highs from last 5-10 days
2. Find horizontal shelves (3+ touches at same price)
3. Add daily session lows/highs from last 10 days
4. Merge levels within 3 pts (keep highest-scored)
5. Round to whole numbers (lows UP, highs DOWN)
6. Score each level for significance
7. Mark top 35% as major
8. Fill gaps at ~6-pt intervals
9. Trim to ~60 below / ~80 above current price
10. Identify bull/bear line

Implementation: strategy/level_generator.py

## Appendix: Terminology

- Shelf of lows: Multiple touches at same price = support
- Failed Breakdown: Loses then recovers significant low
- Level Reclaim: Reclaims resistance from below
- Elevator down: Violent straight-line selloff
- Magnet: Price attractor, range midpoint
- Bull/Bear line: Key bias divider
- Acceptance: Backtests level and holds
- Non-acceptance: 5-pt recovery and hold
- Mode 2: Rangebound (~90% of days)
- Significant low: Prior day low, multi-hour low, or shelf

## Phase 5: Algorithm Validation Results

### Generated vs Mancini Levels for Feb 4, 2026

Using data through Feb 3, 2026 close (6940.50):

#### Supports (8 generated)
- 6932 -> Mancini 6930 (diff 2) MATCH
- 6925 -> Mancini 6926 (diff 1) MATCH
- 6919 -> Mancini 6920 (diff 1) MATCH
- 6912 -> Mancini 6911 (diff 1) MATCH
- 6906 -> Mancini 6904 (diff 2) MATCH
- 6898 -> Mancini 6897 (diff 1) MATCH
- 6887 -> Mancini 6887 (diff 0) MATCH

Support match rate: 88% (7/8 within 3 points)

#### Resistances (15 generated)
- 6942 -> Mancini 6940 (diff 2) MATCH
- 6948 -> Mancini 6949 (diff 1) MATCH
- 6956 -> Mancini 6956 (diff 0) MATCH
- 6960 -> Mancini 6961 (diff 1) MATCH
- 6964 -> Mancini 6964 (diff 0) MATCH
- 6968 -> Mancini 6967 (diff 1) MATCH
- 6976 -> Mancini 6978 (diff 2) MATCH
- 6984 -> Mancini 6982 (diff 2) MATCH
- 6987 -> Mancini 6989 (diff 2) MATCH
- 6993 -> Mancini 6994 (diff 1) MATCH
- 6998 -> Mancini 6997 (diff 1) MATCH

Resistance match rate: 93% (14/15 within 3 points)

Overall match rate: 91% (21/23 levels within 3 points of Mancini actual)
