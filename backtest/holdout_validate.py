"""Hold-out validation runner.

Runs a SPECIFIC config through Nautilus on the 2025-2026 hold-out years
that were locked away during Optuna tuning. This is the standard quant
safety check: if the best-tuned config also works on data the tuner
never saw, we can ship with confidence. If it fails on hold-out, the
tuning result was overfit and must not be deployed.

Usage:
    python3 backtest/holdout_validate.py [--config trial-26]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from backtest.nautilus_runner import NautilusBacktestRunner, NautilusBacktestConfig
from backtest.nautilus_production_5y import load_data, build_daily_sessions
from live.ib_runner import (
    PRODUCTION_STRATEGY, PRODUCTION_ELEVATOR,
    PRODUCTION_EXIT, PRODUCTION_RISK,
)


# Hold-out is 2025-2026 — the tuner never saw these
HOLDOUT_YEARS = [2025, 2026]


CONFIGS = {
    "trial-26": {
        "description": "FB-only sweep best (Mancini-three-stage exits + LR disabled)",
        "strategy_overrides": {
            # Best from fb-only-acceptance sweep
            "acceptance_min_hold_bars": 15,
            "acceptance_max_dip_pts": 19.5,
            "acceptance_min_hold_bars_deep": 5,
            "non_acceptance_min_recovery_pts": 3.5,
            "true_breakdown_abort_bars": 41,
            # FB-only mode
            "allow_level_reclaim": False,
            # Shorts off (longs only — same as sweep)
            "allow_breakdown_short": False,
            "allow_velocity_short": False,
            "allow_backtest_short": False,
            "allow_short_fr": False,
            "allow_short_lj": False,
        },
        "exit_overrides": {
            # EOD flatten off (new default after agent F)
            "eod_flatten_enabled": False,
        },
    },
    "production": {
        "description": "Current PRODUCTION_STRATEGY exactly as deployed (control)",
        "strategy_overrides": {},
        "exit_overrides": {},
    },
    "production-eod-on": {
        "description": "Production params + EOD flatten ON (old behavior for A/B)",
        "strategy_overrides": {},
        "exit_overrides": {"eod_flatten_enabled": True},
    },
    "trial-26-eod-on": {
        "description": "Trial 26 + EOD flatten ON (A/B vs trial-26)",
        "strategy_overrides": {
            "acceptance_min_hold_bars": 15,
            "acceptance_max_dip_pts": 19.5,
            "acceptance_min_hold_bars_deep": 5,
            "non_acceptance_min_recovery_pts": 3.5,
            "true_breakdown_abort_bars": 41,
            "allow_level_reclaim": False,
            "allow_breakdown_short": False,
            "allow_velocity_short": False,
            "allow_backtest_short": False,
            "allow_short_fr": False,
            "allow_short_lj": False,
        },
        "exit_overrides": {"eod_flatten_enabled": True},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="trial-26", choices=list(CONFIGS.keys()))
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    print(f"Hold-out validation: {args.config}")
    print(f"  {cfg['description']}\n")

    # Build params
    strategy_params = PRODUCTION_STRATEGY
    if cfg["strategy_overrides"]:
        strategy_params = replace(strategy_params, **cfg["strategy_overrides"])
    exit_params = PRODUCTION_EXIT
    if cfg["exit_overrides"]:
        exit_params = replace(exit_params, **cfg["exit_overrides"])

    print("Key params:")
    print(f"  acceptance_min_hold_bars:    {strategy_params.acceptance_min_hold_bars}")
    print(f"  acceptance_max_dip_pts:      {strategy_params.acceptance_max_dip_pts}")
    print(f"  non_acceptance_min_recovery: {strategy_params.non_acceptance_min_recovery_pts}")
    print(f"  allow_level_reclaim:         {strategy_params.allow_level_reclaim}")
    print(f"  eod_flatten_enabled:         {exit_params.eod_flatten_enabled}")
    print(f"  multi_session_runner:        {exit_params.multi_session_runner}")
    print(f"  structure_trail_enabled:     {exit_params.structure_trail_enabled}")
    print()

    print("Loading 5y ES data...")
    df = load_data()
    all_sessions = build_daily_sessions(df)
    holdout_dfs = {d: v for d, v in all_sessions.items() if d.year in HOLDOUT_YEARS}
    print(f"Hold-out sessions: {len(holdout_dfs)} "
          f"({sorted(holdout_dfs)[0]} → {sorted(holdout_dfs)[-1]}, {HOLDOUT_YEARS})")
    print(f"  *** This data was LOCKED during Optuna tuning. ***\n")

    bt_cfg = NautilusBacktestConfig(
        strategy_params=strategy_params,
        elevator_params=PRODUCTION_ELEVATOR,
        exit_params=exit_params,
        risk_params=PRODUCTION_RISK,
        min_rr_ratio=PRODUCTION_RISK.min_rr_ratio,
    )
    runner = NautilusBacktestRunner(bt_cfg)

    # Run all hold-out sessions
    all_trades = []
    prior_day_df = None
    print("Running hold-out (each session ~1-2 sec)…")
    for i, d in enumerate(sorted(holdout_dfs)):
        df_sess = holdout_dfs[d]
        if len(df_sess) < 10:
            continue
        try:
            dr = runner.run_single_day(df_sess, prior_day_df=prior_day_df, day=d)
            all_trades.extend(dr.trade_records)
            prior_day_df = df_sess
            if (i + 1) % 50 == 0:
                cum_pnl = sum(t.pnl_pts for t in all_trades)
                print(f"  [{i+1:>3}/{len(holdout_dfs)}] {d}  "
                      f"cum_n={len(all_trades)}  cum_pnl={cum_pnl:+.1f}",
                      flush=True)
        except Exception as e:
            print(f"  ERROR on {d}: {e}")
            continue

    # Per-year breakdown
    by_year: dict[int, list] = {}
    for t in all_trades:
        et = getattr(t, "entry_time", None)
        if et is None:
            continue
        by_year.setdefault(et.year, []).append(t)

    print("\n" + "=" * 70)
    print(f"HOLD-OUT RESULT — config '{args.config}'")
    print("=" * 70)
    total_pts = sum(t.pnl_pts for t in all_trades)
    wins = sum(1 for t in all_trades if t.pnl_pts > 0)
    n = len(all_trades)
    print(f"Trades:    {n}")
    print(f"Wins:      {wins}  WR {wins/max(n,1)*100:.1f}%")
    print(f"PnL:       {total_pts:+,.1f} pts (${total_pts*50:+,.0f} ES, 1 contract basis)")
    if PRODUCTION_EXIT.default_contracts > 1:
        print(f"PnL 4 ct:  ${total_pts*50*4:+,.0f} ES "
              f"(approx — actual depends on size factors)")

    print(f"\nPer-year:")
    print(f"  {'year':>5}  {'n':>5}  {'WR':>6}  {'PnL pts':>10}")
    for y in sorted(by_year):
        ts = by_year[y]
        nn = len(ts)
        w = sum(1 for t in ts if t.pnl_pts > 0)
        p = sum(t.pnl_pts for t in ts)
        print(f"  {y:>5}  {nn:>5}  {w/nn*100:>5.1f}%  {p:>+9.1f}")

    # Verdict
    print("\n" + "=" * 70)
    if total_pts > 0:
        print("✅ HOLD-OUT POSITIVE — config generalizes. Safe to consider for production.")
    elif by_year.get(2025, []) and sum(t.pnl_pts for t in by_year[2025]) > 0:
        print("⚠️  Hold-out negative overall but 2025 (full year) is positive.")
        print("    2026 is partial (5 weeks). Worth deeper look.")
    else:
        print("❌ HOLD-OUT NEGATIVE — config did NOT generalize. Do not ship.")


if __name__ == "__main__":
    main()
