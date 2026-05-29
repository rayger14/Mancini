"""Production-strategy 5y backtest through NautilusTrader.

Uses the existing nautilus_runner/nautilus_strategy scaffolding (which
delegates signal generation to our SignalAggregator) but injects
PRODUCTION_STRATEGY / PRODUCTION_EXIT / PRODUCTION_RISK directly from
live/ib_runner.py — so we backtest what's actually deployed.

Why this exists: the legacy backtest harness (five_year_long_short.py)
uses different params from live, and the custom production_strategy_5y.py
harness runs in idealised mode without realistic fill simulation. This
harness uses NautilusTrader for real OCO order execution, slippage, and
commission modelling.

Requires:
    - .venv-nautilus active (Python 3.12 + nautilus_trader installed)
    - data/ES_1m_full_session_2021-01-01_2026-02-05.parquet

Usage:
    source .venv-nautilus/bin/activate
    python3 backtest/nautilus_production_5y.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

from backtest.nautilus_runner import (
    NautilusBacktestRunner, NautilusBacktestConfig,
)
from live.ib_runner import (
    PRODUCTION_STRATEGY, PRODUCTION_ELEVATOR,
    PRODUCTION_EXIT, PRODUCTION_RISK,
)


def load_data() -> pd.DataFrame:
    path = (Path(__file__).parent.parent
            / "data" / "ES_1m_full_session_2021-01-01_2026-02-05.parquet")
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    return df


def build_daily_sessions(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Bucket 1-min bars into ET trading sessions (18:00 prev → 16:59 today)."""
    sessions: dict[date, pd.DataFrame] = {}
    evening_mask = df.index.time == dt_time(18, 0)
    starts = df.index[evening_mask]
    for start in starts:
        next_day = start.date() + timedelta(days=1)
        end = pd.Timestamp(
            datetime.combine(next_day, dt_time(16, 59)),
            tz="US/Eastern",
        )
        s = df[(df.index >= start) & (df.index <= end)]
        # Skip the 17:00-18:00 globex break
        s = s[~((s.index.time >= dt_time(17, 0)) & (s.index.time < dt_time(18, 0)))]
        if len(s) > 0:
            sessions[next_day] = s
    return sessions


def build_runner(longs_only: bool = False,
                 legacy_params: bool = False,
                 legacy_exit: bool = False) -> NautilusBacktestRunner:
    """Build the Nautilus runner.

    - longs_only=True disables all short pattern flags (the existing
      scaffolding doesn't yet handle short order submission).
    - legacy_params=True overlays the long-side Optuna v2 drift with the
      legacy tight tuning that produced +$93K in the old harness:
        acceptance_max_dip_pts:   15  → 4
        acceptance_min_hold_bars: 11  → 7
        max_fb_sweep_depth_pts:   999 → 10
        true_breakdown_abort_bars: 40 → 20
        fb_stop_buffer_pts:       6.0 → 5.5
        lr_stop_buffer_pts:       4.0 → 5.0
        mancini_t1_at_first_resistance: True → False
      Scientific control: if this reproduces ~+$93K through Nautilus,
      we've isolated the regression to PRODUCTION_STRATEGY's params.
    """
    from dataclasses import replace  # noqa: used below
    strategy_params = PRODUCTION_STRATEGY
    if longs_only:
        strategy_params = replace(
            strategy_params,
            allow_breakdown_short=False,
            allow_velocity_short=False,
            allow_backtest_short=False,
            allow_short_fr=False,
            allow_short_lj=False,
        )
    if legacy_params:
        strategy_params = replace(
            strategy_params,
            acceptance_max_dip_pts=4.0,
            acceptance_min_hold_bars=7,
            max_fb_sweep_depth_pts=10.0,
            true_breakdown_abort_bars=20,
            fb_stop_buffer_pts=5.5,
            lr_stop_buffer_pts=5.0,
            mancini_t1_at_first_resistance=False,
        )

    exit_params = PRODUCTION_EXIT
    if legacy_exit:
        # Legacy harness used: t1_exit_fraction=1.0, trailing_stop_pts=7.0,
        # default_contracts=4. NO runner. Mirror those defaults so we
        # isolate the exit-mechanics effect from the strategy-params effect.
        exit_params = replace(
            exit_params,
            t1_exit_fraction=1.0,
            t2_exit_fraction=0.0,
            runner_fraction=0.0,
            trailing_stop_pts=7.0,
        )

    cfg = NautilusBacktestConfig(
        strategy_params=strategy_params,
        elevator_params=PRODUCTION_ELEVATOR,
        exit_params=exit_params,
        risk_params=PRODUCTION_RISK,
        min_rr_ratio=PRODUCTION_RISK.min_rr_ratio,
    )
    return NautilusBacktestRunner(cfg)


def smoke_test(sessions: dict[date, pd.DataFrame], longs_only: bool = True) -> None:
    """Multi-session smoke test (5 sessions) — validates pipeline +
    confirms long signals actually fire and produce orders. Defaults to
    longs_only because the existing scaffolding doesn't handle short
    order submission."""
    runner = build_runner(longs_only=longs_only)

    # Try 5 sessions spaced across the data
    sorted_dates = sorted(sessions)
    pick_idxs = [50, 250, 500, 750, 1000]  # spread across 5y
    targets = [sorted_dates[i] for i in pick_idxs if i < len(sorted_dates)]

    print(f"\nSmoke test: {len(targets)} sessions, longs_only={longs_only}")
    print(f"{'date':>12}  {'bars':>5}  {'trades':>6}  {'pnl_pts':>9}")
    total_n = 0; total_pnl = 0.0
    for target in targets:
        df = sessions[target]
        prior_dates = [d for d in sorted_dates if d < target]
        prior_day_df = sessions[prior_dates[-1]] if prior_dates else None

        result = runner.run_single_day(df, prior_day_df=prior_day_df, day=target)
        total_n += result.num_trades
        total_pnl += result.pnl_pts
        print(f"  {target}  {len(df):>5}  {result.num_trades:>6}  "
              f"{result.pnl_pts:>+8.1f}")
        for t in result.trade_records[:2]:
            print(f"      {t.pattern_type} entry={t.entry_price:.2f} "
                  f"pnl={t.pnl_pts:+.2f}")
    print(f"\nTotal: {total_n} trades, {total_pnl:+.1f} pts")


def full_run(sessions: dict[date, pd.DataFrame],
             skip_mondays: bool = True,
             longs_only: bool = True,
             legacy_params: bool = False,
             legacy_exit: bool = False) -> None:
    runner = build_runner(
        longs_only=longs_only,
        legacy_params=legacy_params,
        legacy_exit=legacy_exit,
    )
    daily_dfs = {}
    for d in sorted(sessions):
        if skip_mondays and d.weekday() == 0:
            continue
        daily_dfs[d] = sessions[d]
    print(f"\nFull run: {len(daily_dfs)} sessions "
          f"({sorted(daily_dfs)[0]} → {sorted(daily_dfs)[-1]})  longs_only={longs_only}",
          flush=True)

    # Manual loop so we get progress output (run_multi_day uses loguru,
    # which is silenced for this harness).
    sorted_dates = sorted(daily_dfs)
    from backtest.runner import BacktestResult
    result = BacktestResult()
    prior_day_df = None
    for i, d in enumerate(sorted_dates):
        df = daily_dfs[d]
        if len(df) < 10:
            continue
        try:
            day_result = runner.run_single_day(df, prior_day_df=prior_day_df, day=d)
            result.days.append(day_result)
            result.all_trades.extend(day_result.trade_records)
            if (i + 1) % 25 == 0 or day_result.num_trades > 0:
                cum_pnl = sum(t.pnl_pts for t in result.all_trades)
                cum_n = len(result.all_trades)
                print(f"  [{i+1:>4}/{len(sorted_dates)}] {d}  "
                      f"day_trades={day_result.num_trades}  day_pnl={day_result.pnl_pts:+.1f}  "
                      f"cum_n={cum_n}  cum_pnl={cum_pnl:+.1f}",
                      flush=True)
            prior_day_df = df
        except Exception as e:
            print(f"  ERROR on {d}: {e}", flush=True)
            continue

    print("\n" + "=" * 80)
    print("NAUTILUS PRODUCTION-STRATEGY BACKTEST — 5 YEAR")
    print("=" * 80)
    print(f"Sessions:      {len(result.days)}")
    print(f"Trades:        {result.total_trades}")
    print(f"Win rate:      {result.win_rate:.1%}")
    print(f"PnL pts:       {result.total_pnl_pts:+,.1f}")
    print(f"PnL ES $:      ${result.total_pnl_pts*50:+,.0f}")
    print(f"PnL (with contracts/commission): ${result.total_pnl_dollars:+,.2f}")

    # Per-pattern
    by_pat: dict[str, list] = {}
    for t in result.all_trades:
        by_pat.setdefault(t.pattern_type, []).append(t)
    print("\n--- Per pattern ---")
    print(f"{'pattern':24} {'n':>5} {'WR':>6} {'PnL pts':>10}")
    for p, ts in sorted(by_pat.items(), key=lambda x: -sum(t.pnl_pts for t in x[1])):
        n = len(ts)
        wins = sum(1 for t in ts if t.pnl_pts > 0)
        pnl = sum(t.pnl_pts for t in ts)
        print(f"{p:24} {n:>5} {wins/n*100:>5.1f}% {pnl:>+9.1f}")

    # Per-year
    by_year: dict[int, list] = {}
    for t in result.all_trades:
        # t.entry_time is datetime; t may not have .session_date — extract year from entry
        et = getattr(t, "entry_time", None)
        if et is None:
            continue
        by_year.setdefault(et.year, []).append(t)
    print("\n--- Per year ---")
    print(f"{'year':>5} {'n':>5} {'WR':>6} {'PnL pts':>10}")
    for y in sorted(by_year):
        ts = by_year[y]
        n = len(ts)
        wins = sum(1 for t in ts if t.pnl_pts > 0)
        pnl = sum(t.pnl_pts for t in ts)
        print(f"{y:>5} {n:>5} {wins/n*100:>5.1f}% {pnl:>+9.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Run only a single-session smoke test")
    ap.add_argument("--keep-mondays", action="store_true",
                    help="Don't skip Mondays (default skips per Optuna v2 convention)")
    ap.add_argument("--legacy-params", action="store_true",
                    help="Overlay legacy tight params on PRODUCTION_STRATEGY")
    ap.add_argument("--legacy-exit", action="store_true",
                    help="Use legacy exit (t1_exit_fraction=1.0, no runner, "
                         "trailing_stop_pts=7.0). Isolate exit-mechanics effect.")
    args = ap.parse_args()

    print("Loading 5y ES data…")
    df = load_data()
    sessions = build_daily_sessions(df)
    print(f"Built {len(sessions)} sessions "
          f"({sorted(sessions)[0]} → {sorted(sessions)[-1]})")

    print(f"\nProduction config:")
    print(f"  acceptance_max_dip_pts:           {PRODUCTION_STRATEGY.acceptance_max_dip_pts}")
    print(f"  max_fb_sweep_depth_pts:           {PRODUCTION_STRATEGY.max_fb_sweep_depth_pts}")
    print(f"  mancini_t1_at_first_resistance:   {PRODUCTION_STRATEGY.mancini_t1_at_first_resistance}")
    print(f"  block_pdl_shorts:                 {PRODUCTION_STRATEGY.block_pdl_shorts}")
    print(f"  default_contracts:                {PRODUCTION_EXIT.default_contracts}")
    print(f"  t1/runner fractions:              {PRODUCTION_EXIT.t1_exit_fraction}/{PRODUCTION_EXIT.runner_fraction}")
    print(f"  max_daily_loss_pts:               {PRODUCTION_RISK.max_daily_loss_pts}")

    if args.legacy_params or args.legacy_exit:
        print(f"\n*** OVERLAYS ACTIVE — legacy_params={args.legacy_params}, "
              f"legacy_exit={args.legacy_exit} ***")

    if args.smoke:
        smoke_test(sessions)
    else:
        full_run(
            sessions,
            skip_mondays=not args.keep_mondays,
            legacy_params=args.legacy_params,
            legacy_exit=args.legacy_exit,
        )


if __name__ == "__main__":
    main()
