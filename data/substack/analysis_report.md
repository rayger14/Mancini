# Mancini Newsletter Analysis Report
## Compiled from 500 posts (June 2024 - February 2026)

---

## EXECUTIVE SUMMARY

Three parallel agents analyzed 500 paywalled posts from Adam Mancini's Trade Companion newsletter. The findings reveal extremely precise, codifiable trading rules that can directly improve our engine.

---

## 1. FAILED BREAKDOWN SETUP (90-95% of trades)

### Definition
Price sets a "significant low", loses it (sweeps below), then recovers above it. The recovery triggers a long entry.

### Significant Low Definition (3 precise criteria):
1. **Prior day's low** (highest quality)
2. **Any low producing a 20+ point bounce** (V-shaped reversal)
3. **Shelf/cluster of 2+ lows at similar prices**

### Flush Classification:
- **Shallow (<20 pts below level)**: Faster acceptance, quicker entry
- **Deep (>20 pts below level)**: Requires more patience, longer acceptance

### The "Acceptance" Requirement (CRITICAL - we're missing this):

You CANNOT just long the recovery. You must see "acceptance" first:

**Type 1 (deep flushes)**: Price back-tests the significant low from below, sells off, then returns to it. "The dog returns to its owner."

**Type 2 (shallow flushes)**: Price rips through the significant low, sells back below it, then recovers again. "Market getting the trap out of its system."

**Non-Acceptance Protocol (fast markets)**: Price clears level by 5+ points and holds above for 2-3 minutes.

### The "Danger Zone": 0-5 points above the significant low
> "5 points above the significant low is the danger zone for Failed Breakdowns and where most losses occur."

### Acceptance Timing:
- Shallow + high volatility: ~5-10 minutes
- Deep + low volatility: can take over an hour
- "Acceptance is a function of structure, not of time"

### Stop Placement:
- Below the LOWEST LOW of the entire sweep structure (not just the significant low)
- Plus a few points buffer (~5 pts)
- Max risk: 15 ES points at full size; size down if wider stop needed

### Return Distribution:
- Average: 30-60 points
- ~70% cluster in average range
- ~15% right tail: 70-600 points
- ~15% left tail: 4-15 points (goes 1 level and fails)

---

## 2. TRADE MANAGEMENT (Extremely Codifiable)

### Position Scaling:
1. **75% off at first level up** from entry (5-11 pts)
2. **Move stop to approximately break-even**
3. **More off at second level up** (11-20 pts), leave **10% runner**
4. **Trail runner** under prior day's low, updated daily

### Daily Risk Rules:
- Max 2 losing trades/day, then QUIT
- After 1st win: "Profit Protection Mode" — only risk profits on subsequent trades
- Never let a green day go red

### Time-of-Day Windows:
- **Best**: 7:30-8:30 AM or after 3 PM
- **Avoid**: 11 AM - 2 PM (midday chop)
- Max 1-2 trades/day; ~15 minutes actual trading time

---

## 3. MARKET CONTEXT / DAY TYPES

### Mode 1 vs Mode 2:
- **Mode 1** (10% of days): Open-to-close trend day. Mode 1 Red even rarer (~1-2/month)
- **Mode 2** (90% of days): Range, consolidation, traps both directions

### The "Two Siblings" (core market dynamic):
1. **Elevator Down**: Sharp, violent sell lasting minutes to hours
2. **Short Squeeze**: ALWAYS follows elevator down, but ONLY when a Failed Breakdown triggers

> "The larger the sell, the larger the squeeze."

### The "No Knife Catching" Rule:
> "When ES is in elevator down mode, we patiently wait for the Failed Breakdown."

---

## 4. NUMERICAL LEVEL DATA & CALIBRATION

### Level Grid Parameters (from 6 deeply-analyzed posts):
- **Minor level spacing**: 4-7 ES points (CONSTANT, does not scale with price)
- **Major level spacing**: 10-20 ES points
- **Total levels per day**: ~30-45 supports, ~25-40 resistances

### Failed Breakdown Parameters:
| Parameter | Value |
|-----------|-------|
| Ideal flush depth | 2-11 points below level (sweet spot: 3-8) |
| Danger zone | 0-5 points above significant low |
| Non-acceptance trigger | Level + 5 pts, hold 2-3 min |
| 1st profit target | 5-11 points above entry |
| 2nd profit target | 11-20 points above entry |
| Max stop size | 15 points full size |
| Average FB return | 30-60 points |

### Regime Parameters:
| Parameter | Value |
|-----------|-------|
| Normal daily range | 40-80 ES points |
| High-vol daily range | 100-140 ES points |
| Elevator down magnitude | 30-130+ ES points |
| Squeeze magnitude | 0.5x-1.0x of preceding sell |
| Breakdown trap rate | 80% (only 20% are real breakdowns) |

### Flush Depth vs Rally Result (from real trades):
| Significant Low | Flush Depth | Rally Result |
|----------------|-------------|-------------|
| 5517 | 1 pt | +62 pts |
| 5519 | 8 pts | +21 pts |
| 5585 | 2 pts | +77 pts |
| 5734 | 5 pts | +71 pts |
| 6920 | 32 pts | +74 pts |
| 6864 | 2 pts | +68 pts |
| 6862 | 5 pts | +9 pts |
| 6838 | 6 pts | +19 pts |
| 6832 | 10 pts | +28 pts |

---

## 5. KEY TERMINOLOGY

| Term | Meaning |
|------|---------|
| Elevator Down | Sharp, violent sell cutting every support |
| Failed Breakdown | Sweep of significant low + recovery = long entry |
| Level Reclaim | Recovery of horizontal S/R line = long entry |
| Acceptance | Price action confirming low wants to hold before entry |
| Non-Acceptance Protocol | 5pt clear + 2-3 min hold when no traditional acceptance |
| Danger Zone | 0-5 points above significant low |
| Significant Low | Prior day's low, 20+ pt bounce low, or shelf of lows |
| Two Siblings | Elevator Down + Short Squeeze (always paired) |
| Mode 1 | Open-to-close trend day (~10% of sessions) |
| Mode 2 | Default rangebound/trappy day (~90% of sessions) |
| Golden Rule | 90% of moves don't follow through |
| Double-Dip FB | Two consecutive FBs at same shelf |
| ATM Machine Level | A level tested so many times it produces profits every touch |

---

## 6. SETUP VARIANTS

### Double-Dip Failed Breakdown
Price puts in a FB, rallies, then takes out the LOWEST LOW of the original FB, recovers, rallies again. Re-entry opportunity.

### Level Reclaim (5-10% of trades)
Recovery of a clear horizontal S/R line with multiple touches. Same acceptance/entry rules as FB.

### Back-Test Short (advanced)
After a support shelf breaks, the re-test from below is shortable. Safest on 1st or 2nd test only.

---

## 7. ACTIONABLE ENGINE IMPROVEMENTS

Based on comparing Mancini's methodology to our current engine:

### Must-Add Features:
1. **Acceptance detection** — Three types, this is the biggest gap
2. **Danger zone filtering** — No entry within 5 pts without acceptance
3. **Non-acceptance protocol** — 5pt clear + 2-3 min hold as fast-entry variant
4. **Shallow vs deep flush classification** — 20pt divider affects acceptance timing
5. **Time-of-day filtering** — Avoid 11am-2pm, prefer 7:30-8:30am and 3pm+
6. **Profit protection mode** — After 1st win, restrict subsequent trades

### Should-Calibrate:
1. **Significant low threshold** — Currently may differ from 20pt bounce requirement
2. **Stop placement** — Below lowest low of structure + ~5pt buffer, max 15pt
3. **Profit targets** — 75% at first level (5-11pts), 10% runner with daily trail
4. **Daily trade limit** — Max 2 trades, quit after 2 losses

### Nice-to-Have:
1. **Mode 1 vs Mode 2 detection** — Reduce sizing on sustained trend-down days
2. **Double-Dip FB detection** — Re-entry after first FB fails
3. **Level Reclaim setup** — Separate entry type for horizontal S/R recovery
4. **OPEX/Holiday sizing** — Reduce size on OPEX/holiday weeks
