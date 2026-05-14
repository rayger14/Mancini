# Mancini Bot Trade Analysis: Feb 25 - Mar 12, 2026

**Dataset**: 32 entries, 27 completed trades, 5 unmatched/open entries, 1,581 near-misses, 22 phantom-resolved signals.
**Instrument**: MES (Micro E-mini S&P 500)
**Net P&L**: -114.4 pts ($-572 on MES, would be $-5,720 on ES)
**Overall Win Rate**: 48% (13W / 14L on 27 completed trades)

---

## Part 1: Individual Trade Dissections

### Trade 1 -- Feb 25, BD Short @ 6951.50 -> -9.04 pts (LOSS)
- **Setup**: Breakdown short below CLUSTER_LOW at 6956.54. Price had already sold off 36+ pts from session high of 6988. Late Night Session, bar 393.
- **Signal**: Entry 6951.50, stop 6960.54, target 6936.50. R:R = 1.66.
- **Why it lost**: Entered during Late Night Session (low liquidity). Price was already near session low (6941.50 -- only 10 pts below entry). The cluster low had been tested extensively (99 touches), meaning the breakdown was into exhausted selling. Price bounced off the session low and retraced to stop.
- **Context**: Session range 46.5 pts already established. NEUTRAL regime. 5 nearby levels within 7.5 pts above entry -- dense resistance overhead for a short, but the levels had already broken.
- **Actual vs planned R:R**: Planned 1.66, actual -1.0 (full stop hit).
- **Production**: Yes, production would have taken this. BAD SIGNAL for production.

### Trade 2 -- Feb 25, BD Short @ 6951.75 -> -8.79 pts (LOSS)
- **Setup**: Identical to Trade 1 -- re-entered same breakdown 8 minutes later at essentially the same price.
- **Why it lost**: Same structural problem. Doubling down on a Late Night breakdown that already failed once. The session low at 6941.50 was clearly holding.
- **Lesson**: **Never re-enter the same failed breakdown in the same session without a new catalyst.** The first stop-out IS the information.
- **Production**: Yes. Another production loss.

### Trade 3 -- Feb 26, FORCE_TEST Long @ 6963.25 -> +5.00 pts (WIN)
- **Setup**: Manual force test entry during Pre-Market. Session range only 18.25 pts.
- **Why it won**: Low-risk test in a narrow range. Target was small (5 pts) and achievable.
- **Context**: Only 1 nearby level (PRIOR_DAY_HIGH at 6988, 24.75 pts away). Clean air to run.
- **Production**: Yes.

### Trade 4 -- Feb 26, BD Short @ 6894.50 -> -8.29 pts (LOSS)
- **Setup**: Breakdown short below CLUSTER_LOW at 6898.79 during Evening session (gate bypassed). R:R = 1.03.
- **Why it lost**: Evening session bypass. Low R:R (barely 1:1). Price was already down 76 pts from session high of 6970.50 -- the move was exhausted.
- **Context**: Session range 44.75 pts. Two gates bypassed: Globex Evening + Evening block.
- **Lesson**: **Evening session BD Shorts after extended moves are low probability.** The selling pressure that created the breakdown is absent in thin liquidity.
- **Production**: No. Gates correctly rejected this.

### Trade 5 -- Feb 27, BD Short @ 6859.75 -> -15.00 pts (LOSS)
- **Setup**: Breakdown short below PRIOR_DAY_LOW at 6870.75 during Pre-Market. Bar 508. R:R = 1.0 (marginal).
- **Why it lost**: Only 1:1 R:R means even a minor bounce wipes out the trade. Price was already at session lows. The prior day low breakdown into pre-market thin liquidity failed.
- **Context**: Session range 64.75 pts. Price was 74 pts off highs already. Deep in the move.
- **Lesson**: **BD Shorts with R:R = 1.0 are coin flips. Below 1.5, skip them.** Also, PRIOR_DAY_LOW breakdowns work when selling accelerates; in pre-market, there is no acceleration.
- **Production**: Yes. Production took a -15 pt hit.

### Trade 6 -- Feb 27, FB Long @ 6875.75 -> +15.00 pts (WIN)
- **Setup**: Failed breakdown long off PRIOR_DAY_LOW at 6870.75. This is the SAME level Trade 5 shorted through. After the BD Short got stopped, price confirmed a failed breakdown pattern.
- **Why it won**: **BD Short stop = FB Long signal.** The stop-out confirmed that the breakdown was false. The sweep below 6870.75 followed by recovery is the textbook failed breakdown.
- **Context**: Morning Window (Prime) entry. Session range 74.25 pts. R:R stated as 0.73 (below threshold), but gate was bypassed and it hit target for +15.
- **Lesson**: **When a BD Short gets stopped at a PRIOR_DAY_LOW, immediately look for FB Long.** This is the highest-conviction pattern in the dataset: breakdown fails, shorts are trapped, bounce ensues.
- **Production**: No (R:R gate rejected it). **The gate was WRONG here.**

### Trade 7 -- Feb 27, FB Long @ 6872.75 -> +0.75 pts (WIN)
- **Setup**: FB Long off CLUSTER_LOW at 6859.03. Afternoon session. R:R = 0.03 (essentially zero upside to target).
- **Why it barely won**: Target was 6873.50 -- only 0.75 pts above entry. This was a micro-scalp that technically hit target but offered no real edge.
- **Lesson**: Sub-0.1 R:R trades are noise, not signals.
- **Production**: No (R:R gate). Gate was right -- this is meaningless.

### Trade 8 -- Feb 27, Level Reclaim Long @ 6888.25 -> +15.00 pts (WIN)
- **Setup**: Level reclaim of HORIZONTAL_SR at 6872.00. Afternoon session. R:R = 0.78 (below threshold).
- **Why it won**: Price had already recovered significantly from lows. The horizontal SR reclaim at 6872 signaled bullish momentum continuation. Target 6903.25 was hit cleanly.
- **Context**: Range 52.5 pts. Two gates bypassed (Late Day FB-Only + R:R). Despite "low" R:R, the directional bias was correct.
- **Lesson**: **Level reclaim longs in afternoon sessions after morning selloffs have strong momentum.** The R:R gate may be too strict for this pattern type.
- **Production**: No. Gate was WRONG.

### Trade 9 -- Feb 27, FB Long @ 6886.50 -> -2.50 pts (LOSS)
- **Setup**: FB Long off CLUSTER_LOW at 6872.45. Afternoon. R:R = 0.02.
- **Why it lost**: Essentially zero R:R. Stop was 23 pts away at 6863.25, target only 0.50 pts away. The setup parameters were absurd.
- **Production**: No. Gate was right.

### Trade 10 -- Feb 27, Level Reclaim Long @ 6884.00 -> +0.50 pts (WIN)
- **Setup**: Level reclaim of HORIZONTAL_SR at 6883.00. Stop at 6880, target 6884.50. R:R = 0.12.
- **Barely won**: Target only 0.50 pts above entry.
- **Production**: No. Meaningless scalp.

### Trade 11 -- Feb 28, BD Short @ 6834.75 -> +15.00 pts (WIN)
- **Setup**: Breakdown short below PRIOR_DAY_LOW at 6841.50. Evening session (gate bypassed). R:R = 1.4.
- **Why it won**: Fresh PRIOR_DAY_LOW that hadn't been traded yet. Price broke cleanly below and continued. Target 6819.75 hit.
- **Context**: Session range 70.5 pts. Bar 15 (very early in the session -- evening open). Two evening gates bypassed.
- **Lesson**: **First touch of PRIOR_DAY_LOW in a new session, even evening, can work if R:R > 1.3.** The level is fresh and meaningful.
- **Production**: No. The evening gate blocked a +15 winner.

### Trade 12 -- Mar 1, BD Short @ 6833.75 -> -23.50 pts (LOSS)
- **Setup**: BD Short below SAME PRIOR_DAY_LOW 6841.50 (second use). Evening session. R:R = 1.28.
- **Why it lost**: **Second use of the same level.** The first trade (Feb 28) won +15. When the same level is traded again 1 day later, the edge has decayed. The market "knows" this level.
- **Context**: Range only 31.25 pts (low volatility evening). Hit full stop for -23.5 pts.
- **Lesson**: **Level freshness matters enormously.** First trade at level 6841.5 = +15 (win). Second trade = -23.5 (loss). Third trade (Mar 3) = +25.75 (win, but 3 days later with a different market structure).
- **Production**: No.

### Trade 13 -- Mar 2, Level Reclaim Long @ 6905.25 -> -13.25 pts (LOSS)
- **Setup**: Level reclaim of HORIZONTAL_SR at 6895.00 in Chop Zone. R:R = 1.13.
- **Why it lost**: Chop zone entry (1-3 PM). Two gates bypassed. Session range was 105 pts -- huge intraday range suggesting violent, directionless price action.
- **Lesson**: **Level reclaims during chop zone in 100+ pt range sessions are mean-reversion traps.** The range is so wide that "reclaim" is just noise within the range.
- **Production**: No. Gate was RIGHT.

### Trade 14 -- Mar 3, BD Short @ 6837.50 -> +25.75 pts (WIN -- BEST TRADE)
- **Setup**: BD Short below PRIOR_DAY_LOW at 6841.50 (third use of this level). Late Night Session. R:R = 1.88.
- **Why it won**: This was the highest R:R BD Short in the dataset. Price broke below and ran all the way to 6826.75. Runner stopped after T1 for +25.75 pts.
- **Context**: Range 43.25 pts. The level at 6841.5 was being used for the third time, but 3 calendar days had passed since last use, and market had consolidated around it -- making it "re-fresh."
- **Lesson**: **R:R > 1.8 BD Shorts are the premium signals.** Also, runner strategy captured 25.75 pts vs the T1 of only 15 pts -- the runner added 10+ pts.
- **Production**: Yes. Production's best trade.

### Trade 15 -- Mar 4, BD Short @ 6807.00 -> -26.00 pts (LOSS)
- **Setup**: BD Short below MULTI_HOUR_LOW at 6816.00. Evening session, bar 8 (very start of session). R:R = 0.21 (extremely low). Range only 9 pts.
- **Why it lost**: Absurdly low R:R. Entry 6807, stop 6820, target 6804.25 -- only 2.75 pts of upside vs 13 pts of risk. The session had barely begun (range 9 pts). Two evening gates bypassed.
- **Lesson**: **Never trade R:R < 0.5. Period.** Also, entering at bar 8 in an evening session with a 9-pt range is information-free.
- **Production**: No.

### Trade 16 -- Mar 4, BD Short @ 6815.00 -> -10.00 pts (LOSS)
- **Setup**: Second BD Short at SAME MULTI_HOUR_LOW 6816.00. Evening. R:R = 2.15 (looks good on paper). Range 5.5 pts.
- **Why it lost**: Range was only 5.5 pts -- the market hadn't decided direction yet. Despite "good" R:R, the level at 6816 had ALREADY been tested (Trade 15 just stopped out here). Evening liquidity couldn't sustain the breakdown.
- **Lesson**: **R:R alone is insufficient. Session range < 10 pts = no trade.** The market needs to have established a direction before breakdown signals are meaningful.
- **Production**: No.

### Trade 17 -- Mar 4, FB Long @ 6831.00 -> -65.00 pts (WORST TRADE)
- **Setup**: FB Long off MULTI_HOUR_LOW 6816.00. Evening session. R:R = 0.46. Stop at 6798.50 (32.5 pts away!).
- **Why it lost catastrophically**: After two BD Short failures at 6816, the bot flipped to FB Long. But the stop was 32.5 pts wide -- the widest in the entire dataset. Price continued lower and hit the distant stop for -65 pts.
- **Context**: Range 27.25 pts. Evening gates bypassed. The level at 6816 had now been traded 3 times in one session -- completely exhausted.
- **Lesson**: **CRITICAL -- Stops > 20 pts are portfolio-destroying.** A 65-pt loss on MES is $325; on ES it's $3,250. Hard-cap stops at 15 pts maximum. Also, **do not flip from BD Short to FB Long at the same level in the same session** -- if the level is broken in both directions, it's not a level anymore.
- **Production**: No. Gates protected production from a -65 pt disaster.

### Trade 18 -- Mar 4, FB Long @ 6874.00 -> +0.19 pts (WIN -- barely)
- **Setup**: FB Long off CLUSTER_LOW at 6868.56. Afternoon. R:R = 0.01. Target 6874.19 (0.19 pts above entry).
- **Production**: Yes. Meaningless result.

### Trade 19 -- Mar 5, FB Long @ 6887.25 -> -27.00 pts (LOSS)
- **Setup**: FB Long off CLUSTER_LOW at 6883.00. Evening session. R:R = 1.11. Range only 7.5 pts.
- **Why it lost**: Session range 7.5 pts (barely moved). Evening entry. Stop 13.5 pts away. Price dropped to stop.
- **Lesson**: **Session range < 10 at entry is a strong AVOID signal.** Also, evening CLUSTER_LOW FB Longs have no structural support.
- **Production**: No.

### Trade 20 -- Mar 5, FB Long @ 6827.00 -> +15.00 pts (WIN)
- **Setup**: FB Long off PRIOR_DAY_LOW at 6818.50. Midday. R:R = 0.66. Range 69.5 pts.
- **Why it won**: PRIOR_DAY_LOW is a high-significance level. Session range was already 69.5 pts (adequate volatility). Price swept below 6818.50 and recovered.
- **Context**: Bar 694 (deep in session). The failed breakdown off PDL after a 69.5-pt range day is the classic Mancini setup.
- **Production**: Yes. Good production trade.

### Trade 21 -- Mar 6, BD Short @ 6813.75 -> +25.75 pts (WIN)
- **Setup**: BD Short below PRIOR_DAY_LOW at 6818.50. European Open (gate bypassed). R:R = 1.71. Range 41.5 pts.
- **Why it won**: Same PDL (6818.50) that Trade 20 used for an FB Long the previous day. Now price broke below it decisively in the European open with momentum. Runner captured +25.75.
- **Lesson**: **PRIOR_DAY_LOW works in both directions across sessions.** FB Long when it holds, BD Short when it breaks the next day. This is a "level narrative" -- the market respects PDL.
- **Production**: No (European gate). Gate blocked a +25.75 winner.

### Trade 22 -- Mar 9, BD Short @ 6706.50 -> -25.50 pts (LOSS)
- **Setup**: BD Short below PRIOR_DAY_LOW at 6715.25. Midday. R:R = 1.18. Range 70 pts.
- **Why it lost**: Midday entry (historically choppy). Despite adequate R:R and range, the PDL breakdown reversed.
- **Context**: 938 bars in (very deep in session). Price had been selling all day -- breakdown was into exhaustion.
- **Production**: Yes. -25.5 pt production loss.

### Trade 23 -- Mar 10, FB Long @ 6792.50 -> +9.50 pts (WIN)
- **Setup**: FB Long off MULTI_HOUR_LOW at 6791.25. Afternoon. R:R = 0.94. Range 87 pts.
- **Why it won**: Despite "low" R:R, the multi-hour low held. Price swept below 6791.25 and recovered. EOD flatten captured +9.50 (would have been +15 at T1 but flattened early).
- **Production**: Yes.

### Trade 24 -- Mar 11, BD Short @ 6790.00 -> -10.50 pts (LOSS)
- **Setup**: BD Short below MULTI_HOUR_LOW at 6791.25 (same level as Trade 23). Evening session. R:R = 2.86 (highest in dataset). Range 2.25 pts.
- **Why it lost**: **Session range 2.25 pts.** The session had barely started. Despite the best R:R in the dataset, there was zero information in the price action. Bar 9, evening session, 2.25 pt range = complete noise.
- **Lesson**: **R:R is meaningless when session range < 5 pts.** The "high R:R" was a mathematical artifact of thin evening pricing, not a real signal.
- **Production**: No.

### Trade 25 -- Mar 11, FB Long @ 6796.75 -> +15.00 pts (WIN)
- **Setup**: FB Long off CLUSTER_LOW at 6785.89. Evening. R:R = 0.71. Range 10.25 pts.
- **Why it won**: **BD Short stop -> FB Long chain.** Trade 24's BD Short got stopped, confirming the breakdown at 6791.25 was false. The FB Long captured the reversal.
- **Context**: This is the third instance of "BD Short stop -> FB Long wins" in the dataset (after Feb 27 and before).
- **Production**: No. Gate blocked another winner.

### Trade 26 -- Mar 12, BD Short @ 6750.00 -> +29.00 pts (WIN -- second best)
- **Setup**: BD Short below PRIOR_DAY_LOW at 6765.00. Evening session. R:R = 1.50. Range 22.5 pts.
- **Why it won**: PRIOR_DAY_LOW with adequate R:R (1.5) and enough session range (22.5 pts) to confirm direction. Target 6721 hit cleanly.
- **Context**: Evening gates bypassed, but this time the trade worked because PDL + range + R:R all aligned.
- **Production**: No. Gate blocked a +29 winner.

### Trade 27 -- Mar 12, BD Short @ 6735.00 -> -41.50 pts (LOSS)
- **Setup**: BD Short below PRIOR_DAY_LOW at 6749.75 (second level, lower). European Open. R:R = 1.58. Range 29 pts.
- **Why it lost**: **Sequential BD Short at lower level after a winner.** Trade 26 won at 6765, then Trade 27 tried the same thing at 6749.75 (15 pts lower). The selling was exhausted.
- **Lesson**: **BD Short #2 at a lower level in the same session is high risk.** The first breakdown captured the momentum. The second one is chasing.
- **Production**: No.

---

## Part 2: Pattern-Level Statistics

### By Pattern Type x Direction

| Pattern | N | Win Rate | Avg Winner | Avg Loser | Expectancy | Total P&L |
|---------|---|----------|------------|-----------|------------|-----------|
| BD Short (short) | 14 | 29% | +23.9 | -17.8 | -5.9 | -82.6 |
| FB Long (long) | 9 | 67% | +9.2 | -31.5 | -4.3 | -39.1 |
| Level Reclaim (long) | 3 | 67% | +7.8 | -13.2 | +0.8 | +2.2 |
| FORCE_TEST (long) | 1 | 100% | +5.0 | -- | +5.0 | +5.0 |

**Key insight**: BD Shorts have the largest winners (+23.9 avg) but terrible win rate (29%). FB Longs win more often (67%) but the losers are catastrophic (-31.5 avg -- driven by the -65 pt Mar 4 trade). **Both patterns are net negative**, suggesting filter improvements are critical.

### By Level Type

| Level Type | N | Win Rate | Expectancy | Total P&L |
|------------|---|----------|------------|-----------|
| PRIOR_DAY_LOW | 10 | 60% | +2.0 | +20.0 |
| HORIZONTAL_SR | 3 | 67% | +0.8 | +2.2 |
| NO_SIGNAL | 1 | 100% | +5.0 | +5.0 |
| CLUSTER_LOW | 8 | 38% | -5.0 | -39.7 |
| MULTI_HOUR_LOW | 5 | 20% | -20.4 | -102.0 |

**PRIOR_DAY_LOW is the only profitable level type.** MULTI_HOUR_LOW is disastrous (-102 pts on 5 trades, 20% WR). CLUSTER_LOW is also net negative. This suggests the bot should ONLY trade PRIOR_DAY_LOW and HORIZONTAL_SR levels, or apply much stricter filters to CLUSTER_LOW and MULTI_HOUR_LOW.

### By Session Window

| Window | N | Win Rate | Expectancy | Total P&L |
|--------|---|----------|------------|-----------|
| Morning (Prime) | 1 | 100% | +15.0 | +15.0 |
| Afternoon (FB Only) | 6 | 83% | +3.9 | +23.4 |
| Late Night | 3 | 33% | +2.6 | +7.9 |
| Pre-Market | 2 | 50% | -5.0 | -10.0 |
| Midday | 2 | 50% | -5.2 | -10.5 |
| **Evening (Blocked)** | **13** | **31%** | **-10.8** | **-140.3** |

**Evening (Blocked) accounts for 13 of 27 trades and -140.3 pts of losses.** The gates are correct to block these. Morning and Afternoon windows are the only reliably profitable windows.

### By R:R Ratio Bucket

| R:R Bucket | N | Win Rate | Expectancy | Total P&L |
|------------|---|----------|------------|-----------|
| < 0.5 | 6 | 50% | -15.3 | -92.1 |
| 0.5 - 1.0 | 5 | 100% | +13.9 | +69.5 |
| 1.0 - 1.5 | 7 | 14% | -13.9 | -97.5 |
| 1.5 - 2.0 | 6 | 50% | +3.5 | +21.2 |
| 2.0+ | 2 | 0% | -10.2 | -20.5 |

**Paradoxical finding: R:R 0.5-1.0 has 100% win rate (5/5).** R:R 1.0-1.5 has only 14% win rate. R:R 2.0+ has 0% win rate. This is counterintuitive. The explanation: high R:R trades in this dataset are mostly evening session entries with thin liquidity and tiny ranges -- the R:R is a mathematical artifact. The 0.5-1.0 bucket contains afternoon FB Longs that have genuine directional conviction even if the target-to-stop ratio is modest.

### By Session Range at Entry

| Range | N | Win Rate | Expectancy | Total P&L |
|-------|---|----------|------------|-----------|
| < 20 pts | 6 | 33% | -8.9 | -53.5 |
| 20-50 pts | 10 | 40% | -7.5 | -74.9 |
| 50-80 pts | 9 | 67% | +2.0 | +17.7 |
| 80+ pts | 2 | 50% | -1.9 | -3.8 |

**Session range 50-80 pts is the sweet spot.** Below 20 pts, the session hasn't established direction -- signals are noise. 80+ pts is exhaustion territory.

### By Bar Count at Entry

| Bar Count | N | Win Rate | Expectancy | Total P&L |
|-----------|---|----------|------------|-----------|
| < 100 (early) | 8 | 38% | -9.9 | -79.5 |
| 100-400 | 8 | 50% | -5.6 | -44.9 |
| 400-700 | 9 | 56% | +2.9 | +25.9 |
| 700+ | 2 | 50% | -8.0 | -16.0 |

**Entries at bar 400-700 are the best.** Early entries (< 100 bars, mostly evening session starts) are net losers.

### By Production Gate

| Gate | N | Win Rate | Expectancy | Total P&L |
|------|---|----------|------------|-----------|
| Production would take | 9 | 56% | -0.3 | -2.9 |
| Production would NOT take | 18 | 44% | -6.2 | -111.5 |

**Production gates are working.** Gate-bypassed trades lost -111.5 pts on 18 trades (-6.2 exp). Production-approved trades were roughly breakeven (-2.9 pts total). The gates saved the account from -111.5 pts of additional losses.

### By Stop Distance

| Stop Distance | N | Win Rate | Expectancy | Total P&L |
|---------------|---|----------|------------|-----------|
| < 5 pts | 1 | 100% | +0.5 | +0.5 |
| 5-10 pts | 7 | 29% | +0.7 | +4.9 |
| 10-15 pts | 6 | 17% | -16.7 | -100.2 |
| 15-20 pts | 4 | 75% | +2.4 | +9.7 |
| 20+ pts | 8 | 62% | -4.3 | -34.2 |

**10-15 pt stops are the killing zone: 17% WR, -100.2 total.** Tight stops (5-10) are modestly profitable. Wide stops (15-20) win more often but are a small sample. 20+ pt stops are net losers driven by the -65 pt Mar 4 catastrophe.

---

## Part 3: Near-Miss Analysis

### 1,581 near-misses broken down by failure reason:

| Failure Reason | Count | % of Total |
|----------------|-------|------------|
| rr_too_low | 1,196 | 75.6% |
| sweep_too_deep | 249 | 15.7% |
| dip_too_deep | 100 | 6.3% |
| acceptance_timeout | 21 | 1.3% |
| rr_low_sized_down | 15 | 0.9% |

**ALL 1,581 near-misses are FAILED_BREAKDOWN (long) patterns.** Zero BD Short or Level Reclaim near-misses. This means the FB Long detector is generating massive quantities of almost-triggered signals that are being filtered out.

### rr_too_low (1,196 signals)
- **100% (1,195/1,196) had close_at_failure ABOVE the level price** -- meaning the would-be FB Long trade direction was correct.
- R:R distribution of blocked signals:
  - < 0.1: 735 (61%) -- price was AT the level, target hadn't developed yet
  - 0.1-0.3: 95 (8%)
  - 0.3-0.5: 118 (10%)
  - 0.5-0.7: 123 (10%)
  - 0.7-0.9: 94 (8%)
  - 0.9-1.0: 31 (3%) -- these ALMOST qualified

**The 0.7-1.0 bucket (125 signals) is the most interesting.** These nearly qualified for the 1.0 R:R threshold. Given that 100% were directionally correct at the snapshot, lowering the R:R threshold from 1.0 to 0.7 for FAILED_BREAKDOWN on specific level types (PRIOR_DAY_LOW) could capture significant edge.

**HOWEVER**, close_at_failure being above the level does NOT mean the trade would have been profitable -- many of these are bars where price is just oscillating around the level. The 735 signals with R:R < 0.1 confirm that most are noise (price barely above level).

### sweep_too_deep (249 signals)
- **100% (248/249) had close_at_failure above the level price.**
- These are FB Long signals where the sweep went too far below the level before recovering.
- Sweep depth sample: 15.25, 11.0, 11.0 pts.
- **Current max dip parameter is 4.0 pts.** Sweeps of 11-15 pts are being rejected, but price DID recover above the level in virtually all cases.
- **Possible opportunity**: Sweeps that go 5-10 pts deep but recover cleanly may represent stronger failed breakdowns (more shorts trapped). Consider a "deep sweep" variant with a wider dip tolerance but tighter recovery requirements.

### dip_too_deep (100 signals)
- **0% had close_at_failure above the level.** These are correctly rejected -- when the dip goes too deep, the recovery is incomplete.
- The dip_too_deep gate is doing its job perfectly.

### acceptance_timeout (21 signals)
- Only 10% (2/21) had close above level. These are mostly cases where the recovery stalled and never confirmed. The timeout gate is correctly filtering weak recoveries.

### rr_low_sized_down (15 signals)
- All 15 are FAILED_BREAKDOWN longs from Mar 4 and Mar 11.
- Close_at_failure is above level in all cases (e.g., level 6868.6, close 6874.0 -- 5.4 pts above).
- These were signals where R:R was low but a reduced-size trade was considered. The achieved R:R ranged from 0.00 to 0.66.
- Mar 11 signals (3 of them) had achieved R:R of 0.30-0.66 -- these were closer to viable.

### Phantom Resolved (22 signals)
- **21 of 22 would have been stopped out.** These are rejected signals (by stop-too-wide or daily loss limit) where the bot tracked what would have happened. The rejections were overwhelmingly correct.
- Reject reasons: 9 daily loss limit, 7 stop too wide, 3 R:R too low, 1 window block, 2 other.
- **The daily loss limit saved the account on 9 additional losing trades.**

---

## Part 4: Emergent Insights

### 1. BD Short Stop -> FB Long Chaining

Three instances observed:

| Date | BD Short P&L | FB Long P&L | Net |
|------|-------------|-------------|-----|
| Feb 27 | -15.0 | +15.0 | 0.0 |
| Mar 4 | -10.0 (2nd BD) | -65.0 | -75.0 |
| Mar 11 | -10.5 | +15.0 | +4.5 |

**Result: 2 of 3 FB Longs won (+15 each). The outlier lost -65 because the stop was 32.5 pts wide.**

The pattern works IF:
- The FB Long stop is reasonable (< 15 pts)
- The level is a PRIOR_DAY_LOW or high-significance level
- It is NOT the third trade at the same level in the same session (Mar 4 disaster)

**Recommendation**: After a BD Short stop at a PRIOR_DAY_LOW, automatically generate an FB Long signal with a HARD CAP of 15 pts on the stop. Do NOT use this chain if already 2+ trades at the same level in the session.

### 2. Sequential BD Shorts at Lower Levels

Three instances:

| Date | BD #1 Price | BD #1 P&L | BD #2 Price | BD #2 P&L | Delta |
|------|-------------|-----------|-------------|-----------|-------|
| Feb 25 | 6951.50 | -9.0 | 6951.75 | -8.8 | Same level (both lost) |
| Mar 4 | 6807.00 | -26.0 | 6815.00 | -10.0 | Higher (both lost) |
| Mar 12 | 6750.00 | +29.0 | 6735.00 | -41.5 | Lower (2nd lost) |

**Result: BD Short #2 lost in ALL three cases.** When BD #1 was also a loss, #2 compounded the damage. When BD #1 was a win, #2 gave back the profit.

**Recommendation**: Block BD Short #2 in the same session entirely. One BD Short per session, per direction.

### 3. Session Range as Entry Filter

| Range at Entry | Win Rate | Expectancy |
|----------------|----------|------------|
| < 10 pts | 17% (1/6) | -13.8 |
| 10-20 pts | 50% (1/2) | -3.9 |
| 20-50 pts | 44% (4/9) | -5.2 |
| 50-80 pts | 67% (6/9) | +2.0 |
| 80+ pts | 50% (1/2) | -1.9 |

**Session range < 20 pts at entry is a strong negative signal** (only 2 wins out of 8 trades). The market hasn't established direction. Range 50-80 is the sweet spot where trends are established but not exhausted.

**Recommendation**: Add a hard gate: **Do not enter any trade when session range < 15 pts**, unless the session is < 30 minutes old (where low range is expected).

### 4. Level Freshness Decay

Levels traded multiple times:

| Level | Trade #1 | Trade #2 | Trade #3 |
|-------|----------|----------|----------|
| 6956.5 (CLUSTER_LOW) | -9.0 | -8.8 | -- |
| 6841.5 (PRIOR_DAY_LOW) | +15.0 | -23.5 | +25.8 (3 days later) |
| 6816.0 (MULTI_HOUR_LOW) | -26.0 | -10.0 | -65.0 |
| 6791.25 (MULTI_HOUR_LOW) | +9.5 | -10.5 | -- |
| 6818.5 (PRIOR_DAY_LOW) | +15.0 | +25.8 | -- |
| 6883.0 | +0.5 | -27.0 | -- |

**First trade at a level**: 3W / 3L (50%), avg P&L = +0.8
**Second trade at same level in same session**: 0W / 4L (0%), avg P&L = -18.6
**Second trade at same level, different session**: 1W / 1L, mixed

**Recommendation**: After the first trade at a level in a session, mark that level as "used" and do not re-enter. Cross-session re-use is acceptable for PRIOR_DAY_LOW only.

### 5. Stop Distance Optimization

| Stop Distance | Win Rate | Avg P&L When Loss |
|---------------|----------|-------------------|
| < 5 pts | 100% (1/1) | -- |
| 5-10 pts | 29% (2/7) | -7.2 |
| 10-15 pts | 17% (1/6) | -17.3 |
| 15-20 pts | 75% (3/4) | -15.0 |
| 20-32 pts | 63% (5/8) | -45.7 |

The 20-32 pt stops have a decent win rate but the losers are devastating. **The optimal stop range is 5-10 pts** based on risk-adjusted outcomes (modest losses when wrong). 10-15 pt stops are the worst zone -- too wide to limit damage, not wide enough to avoid being stopped by noise.

**Recommendation**: Hard cap stops at 10 pts for BD Short, 15 pts for FB Long. If the calculated stop exceeds these limits, the signal must be rejected.

### 6. MULTI_HOUR_LOW Is Toxic

MULTI_HOUR_LOW trades: 5 completed, 1 win (20%), total -102 pts.

Every MULTI_HOUR_LOW trade except one (Mar 10 FB Long +9.5) was a loss. The structural reason: multi-hour lows are noisy, frequently recalculated, and lack the institutional significance of PRIOR_DAY_LOW. They represent "where price happened to bottom during a consolidation" rather than "a level the market cares about."

**Recommendation**: Remove MULTI_HOUR_LOW from the eligible level types, or require MULTI_HOUR_LOW signals to have R:R > 2.0 and session range > 50 pts.

### 7. The Evening Session Trap

Evening (Blocked) trades: 13 completed, 4 wins (31%), total -140.3 pts.

The evening session (6-10 PM ET) is the single largest source of losses. Of the 4 evening wins, 3 were at PRIOR_DAY_LOW levels. The 9 losses were spread across all level types.

**The gates are correct to block evening trades.** When bypassed, they hemorrhage money. The rare evening winners happen at PRIOR_DAY_LOW -- the only level type with enough significance to cut through thin liquidity.

**Recommendation**: If evening trading is ever enabled, restrict to PRIOR_DAY_LOW only, R:R > 1.5, session range > 20 pts.

### 8. Recovery After Stop-Out

Of 14 stop-outs:
- 6 had the position direction eventually proven correct (shakeout) -- 43%
- 8 had price continue against the position (correct stop) -- 57%

**Stop placement is roughly appropriate** -- the stops are not being systematically shaken out. The issue is not stop placement but entry selection.

### 9. Regime Context

**ALL 27 trades occurred in NEUTRAL regime.** There is zero data on BULL or BEAR regime trades. This means the regime filter is either not varying (always reading NEUTRAL) or the dataset period was a ranging market. The EMA slope values are all identical (3.67), suggesting the regime filter hasn't been updating.

**Recommendation**: Investigate why the regime is always NEUTRAL. If the regime filter is broken, fix it. If the market genuinely was NEUTRAL for 16 days, this is still useful -- it means the current parameters were tested in a choppy, directionless environment.

### 10. Gate Bypass Performance

| Gate Type | Bypass Trades | Win Rate | Total P&L |
|-----------|--------------|----------|-----------|
| Evening block | 10 | 30% | -111.3 |
| R:R too low (various) | 5 | 60% | +28.8 |
| European dead zone | 2 | 50% | -15.8 |
| Chop zone | 1 | 0% | -13.2 |
| RTH Late Day FB-Only | 2 | 100% | +15.5 |
| Past EOD flatten time | (unmatched entries) | -- | -- |

**The R:R gate blocks too aggressively for FB Longs.** 3 of 5 R:R-bypassed trades won (the 2 losses were degenerate sub-0.1 R:R entries). The "RTH Late Day FB-Only" gate blocked 2 winners.

**The evening and chop zone gates are correct.** Do not weaken them.

---

## Part 5: Actionable Recommendations

### A. CRITICAL CHANGES (Implement Immediately)

1. **Hard Stop Cap**: Max stop distance = 15 pts for all patterns. Any signal with calculated stop > 15 pts is rejected. This alone would have prevented the -65 pt Mar 4 loss.

2. **Session Range Minimum**: Do not enter when session_range < 15 pts AND bar_count > 30. (Allow early-session entries where range hasn't developed yet.) This blocks the Mar 4 (range 5.5, 9.0), Mar 5 (range 7.5), and Mar 11 (range 2.25) losers.

3. **One Trade Per Level Per Session**: After entering at a level, mark it as "traded" for the current session. No re-entries at the same level. This blocks 4 second-attempt losses (-18.6 pts avg).

4. **One BD Short Per Session**: Do not take a second BD Short in the same session. The first one is the information event -- if it fails, the edge is gone. If it wins, do not chase.

### B. SIGNAL QUALITY IMPROVEMENTS

5. **Level Type Hierarchy**:
   - Tier 1 (full confidence): PRIOR_DAY_LOW, HORIZONTAL_SR
   - Tier 2 (reduced size): CLUSTER_LOW with R:R > 1.3 and session range > 40
   - Tier 3 (avoid): MULTI_HOUR_LOW unless R:R > 2.0 and session range > 50

6. **R:R Threshold Adjustment**: For FB Long at PRIOR_DAY_LOW only, lower R:R minimum from 1.0 to 0.7. The data shows FB Longs at PDL with R:R 0.5-1.0 won 100% (2/2) and the near-miss data shows massive directional correctness at 0.7+ R:R.

7. **BD Short R:R Minimum**: Raise from 1.0 to 1.5. BD Shorts with R:R 1.0-1.5 won only 14% (1/7). Only R:R > 1.5 BD Shorts showed positive expectancy.

### C. COMPOUND PATTERN SIGNALS

8. **BD Short Stop -> FB Long Chain**: When a BD Short is stopped at a PRIOR_DAY_LOW:
   - Wait for price to reclaim the level (close above level_price)
   - Enter FB Long with stop = min(sweep_low - 2 pts, entry - 15 pts)
   - Target = 15 pts above entry (T1)
   - This pattern won 2/3 times; the loss was due to a 32.5 pt stop. With the 15 pt hard cap, it would be 2/2.

9. **Win at Level -> Block Same-Direction Trade at Lower Level**: When BD Short #1 wins at level A, do NOT take BD Short #2 at level B (where B < A) in the same session. The momentum that made #1 work is exhausted by #2. This blocks the Mar 12 -41.5 pt loss.

### D. SESSION WINDOW OPTIMIZATION

10. **Afternoon FB-Only Window is Underutilized**: 83% WR, +3.9 exp, +23.4 total. The current RTH Late Day FB-Only gate blocked 2 winning level reclaim trades. Consider widening the Afternoon window to include level reclaims, not just FB.

11. **Evening Session**: Keep blocked for all patterns EXCEPT: PRIOR_DAY_LOW BD Short with R:R > 1.5 and session range > 20 pts. This specific combination won 2/3 times (the loss was a sequential #2 trade which would be blocked by recommendation #4).

### E. MONITORING & DIAGNOSTICS

12. **Fix Regime Filter**: All trades show NEUTRAL with identical EMA slope (3.67). Either the regime filter is not updating or the EMA parameters need recalibration. A working regime filter could meaningfully improve signal quality.

13. **Track Level Usage Counter**: Add a `times_traded_today` counter to each level. Log it with every signal. This enables real-time enforcement of the "one trade per level" rule and provides data for future analysis.

14. **Near-Miss Conversion Tracking**: For the 125 near-misses with R:R 0.7-1.0, add post-hoc P&L tracking (what would have happened if entered). This data is needed before implementing recommendation #6.

---

## Summary Scorecard

| Metric | Value |
|--------|-------|
| Total Completed Trades | 27 |
| Win Rate | 48% (13W / 14L) |
| Total P&L | -114.4 pts |
| Largest Winner | +29.0 pts (Mar 12 BD Short @ PDL) |
| Largest Loser | -65.0 pts (Mar 4 FB Long, 32.5 pt stop) |
| Best Pattern | Level Reclaim Long (+2.2 total, 3 trades) |
| Worst Pattern | BD Short (-82.6 total, 14 trades) |
| Best Level Type | PRIOR_DAY_LOW (+20.0 total, 10 trades) |
| Worst Level Type | MULTI_HOUR_LOW (-102.0 total, 5 trades) |
| Best Window | Afternoon FB-Only (+23.4, 6 trades) |
| Worst Window | Evening Blocked (-140.3, 13 trades) |
| Production P&L | -2.9 pts (9 trades, roughly breakeven) |
| Bypass P&L | -111.5 pts (18 trades, heavy losses) |

**Bottom line**: The gates are doing their job. Production-approved trades are near breakeven (-2.9 pts). The -111.5 pts of bypass losses confirm that the filtering logic is sound. The path to profitability is (a) fixing the stop cap issue, (b) avoiding MULTI_HOUR_LOW, (c) implementing the session range minimum, and (d) exploiting the BD Short -> FB Long chain at PRIOR_DAY_LOW levels.
