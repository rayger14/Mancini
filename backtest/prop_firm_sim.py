"""Prop firm funding simulation + compounding backtest over past year.

Models:
1. What prop firm accounts $10K can buy
2. Full-year backtest with compounding (scale up contracts as equity grows)
3. Prop firm profit split and drawdown rules

Usage:
    python3 backtest/prop_firm_sim.py
"""

from __future__ import annotations

import sys
from datetime import date, time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

logger.remove()

from backtest.runner import BacktestRunner
from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    RiskParams, SessionTimes, ESContractSpec,
)

PRODUCTION_STRATEGY = StrategyParams(
    swing_low_order=15, multi_hour_rally_min_pts=22.5,
    level_reclaim_min_touches=4, acceptance_min_hold_bars=7,
    acceptance_min_hold_bars_deep=8, acceptance_max_dip_pts=3.0,
    true_breakdown_abort_bars=12, fb_stop_buffer_pts=5.5,
    lr_stop_buffer_pts=5.0, non_acceptance_min_recovery_pts=5.0,
)
PRODUCTION_ELEVATOR = ElevatorParams(
    min_velocity_pts_per_min=0.75, min_levels_broken=2, higher_low_lookback=4,
)
PRODUCTION_EXIT = ExitParams(t1_exit_fraction=1.0, trailing_stop_pts=7.0)
PRODUCTION_RISK = RiskParams(max_trades_per_day=4)
PRODUCTION_SESSION = SessionTimes(
    chop_zone_start=dtime(13, 0), chop_zone_end=dtime(15, 0),
)

POINT_VALUE = 5.0
COMMISSION_RT = 1.24
SLIPPAGE_PTS = 0.50
MARGIN_PER_CONTRACT = 1_265.0


def load_daily_dfs(parquet_path):
    df = pd.read_parquet(parquet_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    df_rth = df.between_time("09:30", "15:59")
    daily = {}
    for d, grp in df_rth.groupby(df_rth.index.date):
        if len(grp) >= 10:
            daily[d] = grp
    return daily


def run_backtest(daily_dfs):
    contract = ESContractSpec(
        symbol="MES", tick_size=0.25, tick_value=1.25, point_value=5.0,
        margin_initial=1_265.0, margin_maintenance=1_150.0, exchange="CME",
    )
    runner = BacktestRunner(
        strategy_params=PRODUCTION_STRATEGY,
        elevator_params=PRODUCTION_ELEVATOR,
        exit_params=PRODUCTION_EXIT,
        risk_params=PRODUCTION_RISK,
        session_times=PRODUCTION_SESSION,
        contract=contract,
        min_rr_ratio=1.0,
    )
    return runner.run_multi_day(daily_dfs=daily_dfs)


def compound_sim(trades, starting_equity, max_contracts=None,
                 profit_split=1.0, max_dd_dollars=None, label=""):
    equity = starting_equity
    peak_equity = equity
    max_dd = 0
    blown = False
    log = []
    daily_pnl = {}

    for t in trades:
        if blown:
            break

        affordable = int(equity / (MARGIN_PER_CONTRACT * 1.4))
        contracts = max(1, affordable)
        if max_contracts:
            contracts = min(contracts, max_contracts)

        trade_date = t.entry_time.date()

        gross_pts = t.pnl_pts
        slippage = SLIPPAGE_PTS * POINT_VALUE * contracts
        commission = COMMISSION_RT * contracts
        gross_dollars = gross_pts * POINT_VALUE * contracts
        net_dollars = gross_dollars - slippage - commission

        # Apply profit split only on profits
        if net_dollars > 0:
            net_dollars *= profit_split

        equity += net_dollars
        peak_equity = max(peak_equity, equity)
        dd = peak_equity - equity
        max_dd = max(max_dd, dd)

        daily_pnl[trade_date] = daily_pnl.get(trade_date, 0) + net_dollars

        if max_dd_dollars and dd > max_dd_dollars:
            blown = True

        log.append({
            "date": trade_date,
            "time": t.entry_time.strftime("%H:%M"),
            "pattern": t.pattern_type[:4].upper(),
            "contracts": contracts,
            "gross_pts": gross_pts,
            "gross_$": gross_dollars,
            "costs": slippage + commission,
            "net_$": net_dollars,
            "equity": equity,
            "dd": dd,
            "peak": peak_equity,
        })

    return log, equity, max_dd, blown


def print_trade_log(log, max_rows=None):
    print(f"\n  {'#':>3s}  {'Date':>10s}  {'Time':>5s}  {'Type':>4s}  {'Ctrs':>4s}  "
          f"{'Gross':>8s}  {'Costs':>6s}  {'Net':>8s}  {'Equity':>10s}  {'DD':>6s}")
    print(f"  {'─'*80}")
    rows = log[:max_rows] if max_rows else log
    for i, r in enumerate(rows, 1):
        print(f"  {i:>3d}  {r['date']}  {r['time']:>5s}  {r['pattern']:>4s}  "
              f"{r['contracts']:>4d}  {r['gross_$']:>+8,.0f}  {r['costs']:>6,.0f}  "
              f"{r['net_$']:>+8,.0f}  ${r['equity']:>9,.0f}  ${r['dd']:>5,.0f}")
    if max_rows and len(log) > max_rows:
        print(f"  ... ({len(log) - max_rows} more trades)")


def main():
    DATA_PATH = "data/ES_1m_2024-02-05_2026-02-05.parquet"

    print("Loading data...")
    all_daily = load_daily_dfs(DATA_PATH)

    # Last 12 months, skip Mondays
    cutoff = date(2025, 2, 5)
    daily_dfs = {d: v for d, v in all_daily.items()
                 if d >= cutoff and d.weekday() != 0}
    dates = sorted(daily_dfs.keys())
    print(f"Past 12 months: {len(dates)} trading days ({dates[0]} to {dates[-1]})")

    result = run_backtest(daily_dfs)
    trades = result.all_trades
    print(f"Trades: {len(trades)}, WR: {result.win_rate:.0%}, "
          f"PF: {result.profit_factor:.2f}, Raw PnL: {result.total_pnl_pts:+.1f} pts\n")

    # ══════════════════════════════════════════════════════════════════
    print("=" * 75)
    print("  PROP FIRM OPTIONS WITH $10K BUDGET")
    print("=" * 75)
    print("""
  Popular futures prop firms and what $10K buys you:

  APEX TRADER FUNDING (most popular for futures)
  ───────────────────────────────────────────────
  Account    Eval Fee   Max Contracts   Trailing DD   Profit Target
  $50K       $167/mo    10 MES          $2,500        $3,000
  $100K      $207/mo    20 MES          $3,000        $6,000
  $150K      $297/mo    25 MES          $5,000        $9,000

  Profit split: 100% of first $25,000, then 90/10
  Payout: bi-weekly after 10 trading days

  With $10K you could run 2-3 eval accounts simultaneously
  and keep the rest as buffer for resets (~$80 each).

  TOPSTEP
  ───────
  $50K       $165/mo    5 ES (=50 MES)  $2,000        $3,000
  $100K      $325/mo    10 ES           $3,000        $6,000
  Profit split: 100% first $10K, then 90/10
""")

    # ══════════════════════════════════════════════════════════════════
    print("=" * 75)
    print("  SCENARIO 1: PERSONAL ACCOUNT — $10K COMPOUNDING")
    print("  Scale MES contracts as equity grows, reinvest everything")
    print("=" * 75)

    log1, end1, dd1, _ = compound_sim(trades, 10_000, max_contracts=20)
    print_trade_log(log1)

    profit1 = end1 - 10_000
    print(f"\n  $10,000 -> ${end1:,.0f} ({profit1/10_000*100:+,.1f}% return)")
    print(f"  Max drawdown: ${dd1:,.0f}")
    print(f"  Contracts grew from {log1[0]['contracts']} to {log1[-1]['contracts']}")

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*75}")
    print("  SCENARIO 2: APEX $50K FUNDED — 10 MES max, $2,500 trailing DD")
    print("=" * 75)

    log50, end50, dd50, blown50 = compound_sim(
        trades, 50_000, max_contracts=10, profit_split=1.0,
        max_dd_dollars=2_500,
    )
    print_trade_log(log50)

    if blown50:
        blown_at = len(log50)
        print(f"\n  ACCOUNT BLOWN after trade #{blown_at} — DD exceeded $2,500")
        print(f"  Cost: ~$167 eval + $80 reset = $247 to try again")
    else:
        profit50 = end50 - 50_000
        payout50 = min(profit50, 25_000) + max(0, (profit50 - 25_000) * 0.9)
        eval_cost = 167 * 12
        print(f"\n  Prop firm equity: $50,000 -> ${end50:,.0f}")
        print(f"  Gross profit:  ${profit50:,.0f}")
        print(f"  Your payout:   ${payout50:,.0f} (100% first $25K, 90% after)")
        print(f"  Eval cost:     ${eval_cost:,.0f}/year")
        print(f"  Net to you:    ${payout50 - eval_cost:,.0f}")
        print(f"  Max DD:        ${dd50:,.0f} / $2,500 limit ({dd50/2500*100:.0f}% used)")
        print(f"  ROI on $10K:   {(payout50 - eval_cost)/10_000*100:+,.0f}%")

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*75}")
    print("  SCENARIO 3: APEX $150K FUNDED — 25 MES max, $5,000 trailing DD")
    print("=" * 75)

    log150, end150, dd150, blown150 = compound_sim(
        trades, 150_000, max_contracts=25, profit_split=1.0,
        max_dd_dollars=5_000,
    )
    print_trade_log(log150)

    if blown150:
        blown_at = len(log150)
        print(f"\n  ACCOUNT BLOWN after trade #{blown_at} — DD exceeded $5,000")
    else:
        profit150 = end150 - 150_000
        payout150 = min(profit150, 25_000) + max(0, (profit150 - 25_000) * 0.9)
        eval_cost150 = 297 * 12
        print(f"\n  Prop firm equity: $150,000 -> ${end150:,.0f}")
        print(f"  Gross profit:  ${profit150:,.0f}")
        print(f"  Your payout:   ${payout150:,.0f}")
        print(f"  Eval cost:     ${eval_cost150:,.0f}/year")
        print(f"  Net to you:    ${payout150 - eval_cost150:,.0f}")
        print(f"  Max DD:        ${dd150:,.0f} / $5,000 limit ({dd150/5000*100:.0f}% used)")
        print(f"  ROI on $10K:   {(payout150 - eval_cost150)/10_000*100:+,.0f}%")

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*75}")
    print("  SIDE-BY-SIDE COMPARISON (PAST 12 MONTHS)")
    print("=" * 75)

    print(f"\n  {'Scenario':<35s}  {'Your $':>10s}  {'Net Profit':>10s}  "
          f"{'ROI':>8s}  {'Max DD':>8s}  {'Blown?':>6s}")
    print(f"  {'─'*85}")

    # Personal
    print(f"  {'Personal $10K (compound MES)':<35s}  ${'10,000':>9s}  ${profit1:>9,.0f}  "
          f"{profit1/10_000*100:>+7.0f}%  ${dd1:>7,.0f}  {'No':>6s}")

    # $50K
    if not blown50:
        net50 = payout50 - eval_cost
        print(f"  {'Apex $50K funded (10 MES)':<35s}  ${'10,000':>9s}  ${net50:>9,.0f}  "
              f"{net50/10_000*100:>+7.0f}%  ${dd50:>7,.0f}  {'No':>6s}")
    else:
        print(f"  {'Apex $50K funded (10 MES)':<35s}  ${'10,000':>9s}  {'N/A':>10s}  "
              f"{'N/A':>8s}  ${dd50:>7,.0f}  {'YES':>6s}")

    # $150K
    if not blown150:
        net150 = payout150 - eval_cost150
        print(f"  {'Apex $150K funded (25 MES)':<35s}  ${'10,000':>9s}  ${net150:>9,.0f}  "
              f"{net150/10_000*100:>+7.0f}%  ${dd150:>7,.0f}  {'No':>6s}")
    else:
        print(f"  {'Apex $150K funded (25 MES)':<35s}  ${'10,000':>9s}  {'N/A':>10s}  "
              f"{'N/A':>8s}  ${dd150:>7,.0f}  {'YES':>6s}")

    print(f"\n  Notes:")
    print(f"  - All scenarios use identical production params (no lookahead)")
    print(f"  - Costs include 2-tick slippage + $1.24/contract RT commissions")
    print(f"  - Personal account compounds (adds contracts as equity grows)")
    print(f"  - Prop accounts use fixed max contracts")
    print(f"  - Prop eval fees estimated at 12 months")
    print("=" * 75)


if __name__ == "__main__":
    main()
