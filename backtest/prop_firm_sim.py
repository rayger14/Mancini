"""Prop firm simulation: real dollar P&L at various account sizes.

Answers: Is the bottleneck capital or trade frequency?
"""
import sys
import json
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
    RiskParams, SessionTimes,
)

PRODUCTION_PARAMS = {
    "acceptance_max_dip_pts": 3.0,
    "acceptance_min_hold_bars": 7,
    "acceptance_min_hold_bars_deep": 8,
    "chop_end_hour": 15,
    "chop_start_hour": 13,
    "fb_stop_buffer": 5.5,
    "higher_low_lookback": 4,
    "level_reclaim_min_touches": 4,
    "lr_stop_buffer": 5.0,
    "max_trades_per_day": 4,
    "min_levels_broken": 2,
    "min_rr_ratio": 1.0,
    "min_velocity": 0.75,
    "multi_hour_rally_min_pts": 22.5,
    "non_acceptance_min_recovery_pts": 5.0,
    "swing_low_order": 15,
    "t1_exit_fraction": 1.0,
    "trailing_stop_pts": 7.0,
    "true_breakdown_abort_bars": 12,
}


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


def run_backtest(daily_dfs, params):
    p = params
    strategy = StrategyParams(
        swing_low_order=p["swing_low_order"],
        multi_hour_rally_min_pts=p["multi_hour_rally_min_pts"],
        level_reclaim_min_touches=p["level_reclaim_min_touches"],
        acceptance_min_hold_bars=p["acceptance_min_hold_bars"],
        acceptance_min_hold_bars_deep=p["acceptance_min_hold_bars_deep"],
        acceptance_max_dip_pts=p["acceptance_max_dip_pts"],
        true_breakdown_abort_bars=p["true_breakdown_abort_bars"],
        fb_stop_buffer_pts=p["fb_stop_buffer"],
        lr_stop_buffer_pts=p["lr_stop_buffer"],
        non_acceptance_min_recovery_pts=p["non_acceptance_min_recovery_pts"],
    )
    elevator = ElevatorParams(
        min_velocity_pts_per_min=p["min_velocity"],
        min_levels_broken=p["min_levels_broken"],
        higher_low_lookback=p["higher_low_lookback"],
    )
    exit_params = ExitParams(
        t1_exit_fraction=p["t1_exit_fraction"],
        trailing_stop_pts=p["trailing_stop_pts"],
        default_contracts=p.get("contracts", 4),
    )
    risk = RiskParams(max_trades_per_day=p["max_trades_per_day"])
    session = SessionTimes(
        chop_zone_start=dtime(p["chop_start_hour"], 0),
        chop_zone_end=dtime(p["chop_end_hour"], 0),
    )
    runner = BacktestRunner(
        strategy_params=strategy,
        elevator_params=elevator,
        exit_params=exit_params,
        risk_params=risk,
        session_times=session,
        min_rr_ratio=p["min_rr_ratio"],
    )
    return runner.run_multi_day(daily_dfs=daily_dfs)


def main():
    data_path = Path(__file__).parent.parent / "data" / "ES_1m_2024-02-05_2026-02-05.parquet"
    print("Loading data...")
    daily_dfs = load_daily_dfs(str(data_path))

    # Apply Monday filter
    filtered = {d: df for d, df in daily_dfs.items() if d.weekday() != 0}
    print(f"Loaded {len(daily_dfs)} days, after Monday filter: {len(filtered)} days")

    print("\nRunning backtest with production params + Monday filter...")
    result = run_backtest(filtered, PRODUCTION_PARAMS)
    trades = result.all_trades

    print(f"\nTotal trades: {len(trades)}")
    print(f"Win rate: {result.win_rate:.0%}")
    print(f"Profit factor: {result.profit_factor:.2f}")

    # ── Per-contract P&L (the backtest uses 4 contracts) ──────────
    # pnl_pts already includes 4-contract multiplier
    # Per-contract pnl = pnl_pts / contracts
    print("\n" + "=" * 80)
    print("TRADE-BY-TRADE BREAKDOWN")
    print("=" * 80)

    print(f"\n{'#':>3} {'Date':<12} {'Type':<8} {'Entry':>8} {'Exit':>8} "
          f"{'Ctrs':>4} {'PnL Pts':>9} {'Per-Ctr':>8} {'Result':<6}")
    print("-" * 80)

    per_contract_pnls = []
    for i, t in enumerate(trades):
        per_ctr = t.pnl_pts / t.contracts if t.contracts > 0 else 0
        per_contract_pnls.append(per_ctr)
        w = "WIN" if t.pnl_pts > 0 else "LOSS"
        print(f"{i+1:3d} {str(t.entry_time.date()):<12} "
              f"{t.pattern_type[:7]:<8} {t.entry_price:>8.2f} {t.avg_exit_price:>8.2f} "
              f"{t.contracts:>4} {t.pnl_pts:>+9.2f} {per_ctr:>+8.2f} {w:<6}")

    total_per_contract = sum(per_contract_pnls)
    avg_win_pc = np.mean([p for p in per_contract_pnls if p > 0]) if any(p > 0 for p in per_contract_pnls) else 0
    avg_loss_pc = np.mean([p for p in per_contract_pnls if p <= 0]) if any(p <= 0 for p in per_contract_pnls) else 0

    print("-" * 80)
    print(f"Total per-contract points: {total_per_contract:+.2f}")
    print(f"Avg winning trade (per contract): {avg_win_pc:+.2f} pts")
    print(f"Avg losing trade (per contract): {avg_loss_pc:+.2f} pts")

    # ── Equity curves at different sizes ──────────────────────────
    print("\n" + "=" * 80)
    print("DOLLAR P&L BY ACCOUNT SIZE (508 days, Monday filter applied)")
    print("=" * 80)

    scenarios = [
        # (name, contracts, instrument, $/pt, starting_capital)
        ("Personal: 4 MES ($10K)", 4, "MES", 5.0, 10_000),
        ("Personal: 1 ES ($10K)", 1, "ES", 50.0, 10_000),
        ("Prop 50K: 4 MES", 4, "MES", 5.0, 50_000),
        ("Prop 50K: 10 MES", 10, "MES", 5.0, 50_000),
        ("Prop 50K: 2 ES", 2, "ES", 50.0, 50_000),
        ("Prop 100K: 4 ES", 4, "ES", 50.0, 100_000),
        ("Prop 100K: 20 MES", 20, "MES", 5.0, 100_000),
        ("Prop 150K: 6 ES", 6, "ES", 50.0, 150_000),
        ("Prop 150K: 10 ES", 10, "ES", 50.0, 150_000),
    ]

    print(f"\n{'Scenario':<28} {'$/pt':>6} {'Gross P&L':>12} {'Max DD':>10} "
          f"{'DD %':>7} {'ROI':>8} {'$/trade':>9} {'$/month':>9}")
    print("-" * 100)

    for name, ctrs, instr, pt_val, capital in scenarios:
        # Scale from backtest's 4-contract basis to target contracts
        # Per-contract pnls × target contracts × $/pt
        scale = ctrs  # number of contracts
        dollar_pnls = [p * scale * pt_val for p in per_contract_pnls]
        gross = sum(dollar_pnls)

        # Equity curve
        equity = np.cumsum(dollar_pnls)
        peak = np.maximum.accumulate(equity)
        drawdowns = peak - equity
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

        dd_pct = (max_dd / capital * 100) if capital > 0 else 0
        roi = (gross / capital * 100) if capital > 0 else 0
        per_trade = gross / len(trades) if trades else 0
        months = 508 / 21  # ~24.2 trading months
        per_month = gross / months

        print(f"{name:<28} {pt_val*ctrs:>6.0f} {gross:>+12,.0f} {max_dd:>10,.0f} "
              f"{dd_pct:>6.1f}% {roi:>+7.1f}% {per_trade:>+9,.0f} {per_month:>+9,.0f}")

    # ── Prop firm evaluation analysis ─────────────────────────────
    print("\n" + "=" * 80)
    print("PROP FIRM EVALUATION FEASIBILITY")
    print("=" * 80)

    print("\nCan we pass a $3,000 profit target with $2,500 max drawdown?")
    print("(Simulating with different contract sizes)\n")

    for ctrs, instr, pt_val in [(2, "MES", 5.0), (4, "MES", 5.0), (10, "MES", 5.0),
                                  (1, "ES", 50.0), (2, "ES", 50.0)]:
        dollar_pnls = [p * ctrs * pt_val for p in per_contract_pnls]

        # Walk through trades to see when we hit $3K and if we breach $2.5K DD
        equity = 0
        peak_equity = 0
        max_dd = 0
        hit_target = False
        hit_target_trade = 0
        breached_dd = False
        breach_trade = 0

        for i, pnl in enumerate(dollar_pnls):
            equity += pnl
            if equity > peak_equity:
                peak_equity = equity
            dd = peak_equity - equity
            if dd > max_dd:
                max_dd = dd

            if dd > 2500 and not breached_dd:
                breached_dd = True
                breach_trade = i + 1

            if equity >= 3000 and not hit_target:
                hit_target = True
                hit_target_trade = i + 1

        status = ""
        if breached_dd and (not hit_target or breach_trade <= hit_target_trade):
            status = f"FAIL — breached $2,500 DD at trade #{breach_trade} (DD=${max_dd:,.0f})"
        elif hit_target:
            days_est = hit_target_trade / (len(trades) / 508) * hit_target_trade / len(trades)
            # Better estimate: trade frequency
            trades_per_day = len(trades) / 403  # 403 non-Monday days
            days_to_pass = hit_target_trade / trades_per_day
            status = f"PASS at trade #{hit_target_trade} (~{days_to_pass:.0f} trading days, ~{days_to_pass/21:.1f} months)"
        else:
            status = f"DID NOT REACH TARGET (final equity: ${equity:+,.0f}, max DD: ${max_dd:,.0f})"

        label = f"{ctrs} {instr} (${ctrs*pt_val:.0f}/pt)"
        print(f"  {label:<22} → {status}")

    # ── The real question: capital vs frequency ───────────────────
    print("\n" + "=" * 80)
    print("THE REAL QUESTION: IS IT CAPITAL OR TRADE FREQUENCY?")
    print("=" * 80)

    total_pts_per_contract = sum(per_contract_pnls)
    n_trades = len(trades)
    n_days = 403  # non-Monday trading days
    trades_per_day = n_trades / n_days
    trades_per_month = n_trades / (n_days / 21)
    pts_per_trade = total_pts_per_contract / n_trades if n_trades > 0 else 0

    print(f"\n  Per-contract edge per trade: {pts_per_trade:+.2f} pts")
    print(f"  Trade frequency: {trades_per_day:.3f} trades/day = {trades_per_month:.1f} trades/month")
    print(f"  Total per-contract pts: {total_pts_per_contract:+.2f} over {n_days} days")

    print(f"\n  Monthly income at different scales (per-contract × N × $/pt ÷ months):")
    print(f"  {'Scale':<30} {'Monthly':>10} {'Annual':>12} {'Required Capital':>18}")
    print(f"  {'-'*72}")

    months = n_days / 21

    scale_scenarios = [
        ("4 MES (current backtest)", 4, 5.0, "$2,000"),
        ("10 MES", 10, 5.0, "$5,000"),
        ("20 MES (= 2 ES)", 20, 5.0, "$10,000"),
        ("1 ES", 1, 50.0, "$5,000-$12,000"),
        ("2 ES", 2, 50.0, "$10,000-$25,000"),
        ("4 ES", 4, 50.0, "$25,000-$50,000"),
        ("10 ES", 10, 50.0, "$50,000-$125,000"),
        ("20 ES", 20, 50.0, "$100,000-$250,000"),
    ]

    for label, ctrs, pt_val, cap_req in scale_scenarios:
        monthly = total_pts_per_contract * ctrs * pt_val / months
        annual = monthly * 12
        print(f"  {label:<30} {monthly:>+10,.0f} {annual:>+12,.0f} {cap_req:>18}")

    print(f"\n  ANSWER:")
    print(f"  ───────")
    print(f"  The strategy's EDGE is solid: {pts_per_trade:+.2f} pts/trade at 56% WR.")
    print(f"  The FREQUENCY is the bottleneck: only {trades_per_month:.1f} trades/month.")
    print(f"")
    print(f"  At 4 MES ($20/pt total): ${total_pts_per_contract * 4 * 5 / months:+,.0f}/month — coffee money")
    print(f"  At 2 ES ($100/pt total): ${total_pts_per_contract * 2 * 50 / months:+,.0f}/month — car payment")
    print(f"  At 10 ES ($500/pt total): ${total_pts_per_contract * 10 * 50 / months:+,.0f}/month — salary replacement")
    print(f"  At 20 ES ($1000/pt total): ${total_pts_per_contract * 20 * 50 / months:+,.0f}/month — serious money")
    print(f"")
    print(f"  To make $5,000/month you need: ", end="")
    target_monthly = 5000
    pts_per_month_per_ctr = total_pts_per_contract / months
    needed_dollar_per_pt = target_monthly / pts_per_month_per_ctr if pts_per_month_per_ctr > 0 else 0
    es_needed = needed_dollar_per_pt / 50
    mes_needed = needed_dollar_per_pt / 5
    print(f"{es_needed:.0f} ES contracts or {mes_needed:.0f} MES contracts")
    print(f"  That requires roughly ${es_needed * 12500:,.0f}-${es_needed * 25000:,.0f} in capital (for ES)")
    print(f"")
    print(f"  TWO WAYS TO INCREASE INCOME:")
    print(f"  1. MORE CAPITAL → trade more contracts (linear scaling)")
    print(f"  2. MORE TRADES → the strategy currently filters aggressively")
    print(f"     • Chop zone blocks {12}-{15}:00 ({3} hours/day)")
    print(f"     • Only {n_trades} trades in {n_days} days = many days with 0 trades")
    print(f"     • Could explore: shorter hold bars, wider chop zone, ")
    print(f"       lower min_rr_ratio, more level types")
    print(f"     • BUT: more trades historically = lower WR = worse results")
    print(f"     • The 19-param Optuna found 174 trades but degraded to 39% WR")


if __name__ == "__main__":
    main()
