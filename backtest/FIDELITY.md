# Backtest Fidelity vs Live — what the backtest does and doesn't represent

> **CORRECTION (2026-06-30, later):** the 0% numbers below are largely a
> BROKEN-BAR-SOURCE artifact, not a proven divergence. The archived
> `data/sessions/{date}_bars.parquet` files are the bot's rolling **400-bar
> window snapshot** — capped, partial (e.g. 06-09 = 400 bars 10:21→16:58,
> missing the open + overnight), and overwritten each session/restart (06-29 =
> a SINGLE bar after tonight's restarts). So the fidelity harness fed the
> backtest truncated/near-empty data → ~0 trades. **True fidelity is UNMEASURED.**
> First fix: get proper full-session 1-min bars for the plan-sessions (re-fetch
> from IB; the 5y parquet ends 2026-02-05) and fix `_archive_session` to store
> the uncapped accumulator. The P&L-at-targets report reads the same parquets,
> so its reliability depends on this too.

Measured 2026-06-30 with `backtest/fidelity_diff.py` (replays the backtest engine
over the ~50 plan-available sessions, May–Jun 2026, and diffs its entries against
what the bot actually traded live).

## Result: ~0% of live entries reproduced
Over 39 sessions the plan-augmented backtest reproduced **0 of 39** live LONG
entries. Root cause is structural, not a bug:

- **Live runs in COLLECTION MODE** (`bypass_session_gates=True`, an `ib_runner`
  execution mode). It takes off-hours / evening "data-collection" trades. **33 of
  44 (75%)** live long entries in this window were collection-mode (gates
  bypassed); only 11 were production-window.
- **The backtest enforces session windows** (`rth_filter`, time gates in
  `mancini_long.run_day`). It has no `bypass_session_gates` equivalent, so it
  **cannot** take those 75%.

So the backtest models the *disciplined in-window strategy*; live runs a *much more
permissive collection mode*. They are not measuring the same thing.

**Even production-window only: 0/10.** Restricting the diff to the ~10 live trades
the backtest *could* structurally take (production_would_take=True, `PROD_ONLY=1`)
it STILL reproduced 0 — and fired only 1 long across 39 sessions total. So it's not
only collection mode: the backtest's entry detection on the archived per-session
parquets doesn't reproduce live at all. Likely a bar-source difference (archived
`{date}_bars.parquet` coverage vs the full-session bars `feature_comparison` uses)
and/or plan-level FBs needing flushes those bars don't capture — a follow-up
diagnostic. Headline is settled: **backtest ≈ 0% faithful to live.**

## What this means for past backtests
- The 5y `feature_comparison` results (e.g. shorts −445.8pt, the Mode-1-Green
  continuation −1183pt) measure the **in-window, engine-detected-level** behavior.
  They are directionally useful but **do not faithfully represent the live bot**,
  which (a) runs collection mode and (b) trades on Mancini plan levels.

## Known structural divergences (live → backtest)
1. **Collection mode** — live `bypass_session_gates=True` takes off-window trades;
   backtest can't. *(Dominant gap.)*
2. **Mancini plan levels** — now closable: `core/mancini_plan_levels.build_plan_levels`
   is shared; the backtest injects them via `mancini_long._extra_levels`.
3. **daily_bias** — backtest never calls `set_daily_structure` → runs NEUTRAL.
4. **Idealized fills** — backtest fills entries same-bar at signal price, no
   slippage; live is IB bracket OCO real fills.
5. **No cross-session runner carry** in `feature_comparison`.
6. **Conviction sizing** is size-blind / not backtestable.

## To get a trustworthy fidelity number next
Diff the backtest against **production-window live trades only** (the 11), since
those are the only ones the backtest can structurally take — that isolates the
real entry/detection fidelity from the collection-mode noise. Then decide whether
collection mode should be (a) modeled in the backtest, or (b) treated as a
non-backtestable data-gathering mode and excluded from "does the strategy work"
conclusions.
