"""Regime analysis: break down backtest performance by volatility, weekday, and month.

Runs the backtest with the previous best Optuna params, then slices trades
by daily ATR regime, day of week, and calendar month to find clusters of
losses or wins.
"""
import sys
from datetime import date, time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

logger.remove()

from backtest.runner import BacktestRunner
from backtest.walk_forward import load_daily_dfs
from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    RiskParams, SessionTimes,
)

# Previous best Optuna params (49% WR set)
BEST_PARAMS = {
    "acceptance_max_dip_pts": 3.0,
    "acceptance_min_hold_bars": 7,
    "acceptance_min_hold_bars_deep": 8,
    "chop_end_hour": 15,
    "chop_start_hour": 12,
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
    "contracts": 4,
}


def compute_daily_atr(daily_dfs: dict) -> dict[date, float]:
    """Compute daily range (high - low) for each day."""
    atr = {}
    for d, df in daily_dfs.items():
        day_high = df["high"].max()
        day_low = df["low"].min()
        atr[d] = day_high - day_low
    return atr


def classify_regimes(daily_atr: dict[date, float]) -> dict[date, str]:
    """Classify each day into low_vol / normal / high_vol by ATR percentile."""
    values = np.array(list(daily_atr.values()))
    p33 = np.percentile(values, 33)
    p67 = np.percentile(values, 67)
    regimes = {}
    for d, v in daily_atr.items():
        if v < p33:
            regimes[d] = "low_vol"
        elif v <= p67:
            regimes[d] = "normal"
        else:
            regimes[d] = "high_vol"
    return regimes


def run_full_backtest(daily_dfs):
    """Run backtest and return the BacktestResult with trade-level data."""
    strategy = StrategyParams(
        swing_low_order=BEST_PARAMS["swing_low_order"],
        multi_hour_rally_min_pts=BEST_PARAMS["multi_hour_rally_min_pts"],
        level_reclaim_min_touches=BEST_PARAMS["level_reclaim_min_touches"],
        acceptance_min_hold_bars=BEST_PARAMS["acceptance_min_hold_bars"],
        acceptance_min_hold_bars_deep=BEST_PARAMS["acceptance_min_hold_bars_deep"],
        acceptance_max_dip_pts=BEST_PARAMS["acceptance_max_dip_pts"],
        true_breakdown_abort_bars=BEST_PARAMS["true_breakdown_abort_bars"],
        fb_stop_buffer_pts=BEST_PARAMS["fb_stop_buffer"],
        lr_stop_buffer_pts=BEST_PARAMS["lr_stop_buffer"],
        non_acceptance_min_recovery_pts=BEST_PARAMS["non_acceptance_min_recovery_pts"],
    )
    elevator = ElevatorParams(
        min_velocity_pts_per_min=BEST_PARAMS["min_velocity"],
        min_levels_broken=BEST_PARAMS["min_levels_broken"],
        higher_low_lookback=BEST_PARAMS["higher_low_lookback"],
    )
    exit_params = ExitParams(
        t1_exit_fraction=BEST_PARAMS["t1_exit_fraction"],
        trailing_stop_pts=BEST_PARAMS["trailing_stop_pts"],
        default_contracts=BEST_PARAMS["contracts"],
    )
    risk = RiskParams(max_trades_per_day=BEST_PARAMS["max_trades_per_day"])
    session = SessionTimes(
        chop_zone_start=dtime(BEST_PARAMS["chop_start_hour"], 0),
        chop_zone_end=dtime(BEST_PARAMS["chop_end_hour"], 0),
    )

    runner = BacktestRunner(
        strategy_params=strategy,
        elevator_params=elevator,
        exit_params=exit_params,
        risk_params=risk,
        session_times=session,
        min_rr_ratio=BEST_PARAMS["min_rr_ratio"],
    )
    return runner.run_multi_day(daily_dfs=daily_dfs)


def print_group_stats(label: str, trades: list, indent: str = "  "):
    """Print performance stats for a group of trades."""
    n = len(trades)
    if n == 0:
        print(f"{indent}{label:15s}  0 trades")
        return

    wins = [t.pnl_pts for t in trades if t.pnl_pts > 0]
    losses = [t.pnl_pts for t in trades if t.pnl_pts <= 0]
    total_pnl = sum(t.pnl_pts for t in trades)
    wr = len(wins) / n * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print(
        f"{indent}{label:15s}  {n:3d}T  WR={wr:5.1f}%  PF={pf:5.2f}  "
        f"PnL={total_pnl:+8.1f} pts  "
        f"AvgW={avg_win:+6.1f}  AvgL={avg_loss:+6.1f}"
    )


def main():
    data_path = Path(__file__).parent.parent / "data" / "ES_1m_2024-02-05_2026-02-05.parquet"
    print("Loading data...")
    daily_dfs = load_daily_dfs(str(data_path))
    print(f"Loaded {len(daily_dfs)} trading days")

    # Compute daily ATR and regimes
    daily_atr = compute_daily_atr(daily_dfs)
    regimes = classify_regimes(daily_atr)

    atr_values = list(daily_atr.values())
    print(f"\nDaily ATR stats: min={min(atr_values):.1f}  median={np.median(atr_values):.1f}  "
          f"max={max(atr_values):.1f}  p33={np.percentile(atr_values, 33):.1f}  "
          f"p67={np.percentile(atr_values, 67):.1f}")

    regime_counts = {}
    for r in regimes.values():
        regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"Regime days: {regime_counts}")

    # Run backtest
    print("\nRunning backtest with previous best params...")
    result = run_full_backtest(daily_dfs)
    print(f"Total: {result.total_trades} trades, WR={result.win_rate:.1%}, "
          f"PF={result.profit_factor:.2f}, PnL={result.total_pnl_pts:+.1f} pts")

    # Map each trade to its day's regime
    trade_by_regime = {"low_vol": [], "normal": [], "high_vol": []}
    for trade in result.all_trades:
        trade_date = trade.entry_time.date()
        regime = regimes.get(trade_date, "unknown")
        if regime in trade_by_regime:
            trade_by_regime[regime].append(trade)

    # === REGIME BREAKDOWN ===
    print("\n" + "=" * 80)
    print("PERFORMANCE BY VOLATILITY REGIME (daily ATR percentiles)")
    print("=" * 80)
    for regime in ["low_vol", "normal", "high_vol"]:
        print_group_stats(regime, trade_by_regime[regime])

    # Also break down by pattern type within each regime
    print("\n  By regime + pattern type:")
    for regime in ["low_vol", "normal", "high_vol"]:
        trades = trade_by_regime[regime]
        fb = [t for t in trades if t.pattern_type == "failed_breakdown"]
        lr = [t for t in trades if t.pattern_type == "level_reclaim"]
        if fb:
            print_group_stats(f"  {regime}/FB", fb, indent="    ")
        if lr:
            print_group_stats(f"  {regime}/LR", lr, indent="    ")

    # === DAY OF WEEK BREAKDOWN ===
    print("\n" + "=" * 80)
    print("PERFORMANCE BY DAY OF WEEK")
    print("=" * 80)
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    trade_by_dow = {i: [] for i in range(5)}
    for trade in result.all_trades:
        dow = trade.entry_time.weekday()
        if dow in trade_by_dow:
            trade_by_dow[dow].append(trade)

    for dow in range(5):
        print_group_stats(dow_names[dow], trade_by_dow[dow])

    # === MONTHLY BREAKDOWN ===
    print("\n" + "=" * 80)
    print("PERFORMANCE BY MONTH")
    print("=" * 80)
    trade_by_month = {}
    for trade in result.all_trades:
        key = trade.entry_time.strftime("%Y-%m")
        trade_by_month.setdefault(key, []).append(trade)

    for month in sorted(trade_by_month.keys()):
        print_group_stats(month, trade_by_month[month])

    # === MONTHLY SEASONAL (aggregated across years) ===
    print("\n" + "=" * 80)
    print("PERFORMANCE BY CALENDAR MONTH (aggregated)")
    print("=" * 80)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    trade_by_cal_month = {i: [] for i in range(1, 13)}
    for trade in result.all_trades:
        m = trade.entry_time.month
        trade_by_cal_month[m].append(trade)

    for m in range(1, 13):
        if trade_by_cal_month[m]:
            print_group_stats(month_names[m - 1], trade_by_cal_month[m])

    # === ATR CORRELATION ===
    print("\n" + "=" * 80)
    print("ATR vs TRADE PnL CORRELATION")
    print("=" * 80)
    trade_atrs = []
    trade_pnls = []
    for trade in result.all_trades:
        td = trade.entry_time.date()
        if td in daily_atr:
            trade_atrs.append(daily_atr[td])
            trade_pnls.append(trade.pnl_pts)

    if len(trade_atrs) > 5:
        corr = np.corrcoef(trade_atrs, trade_pnls)[0, 1]
        print(f"  Pearson correlation(ATR, PnL): {corr:+.3f}")

        # ATR quintiles for finer granularity
        atr_arr = np.array(trade_atrs)
        pnl_arr = np.array(trade_pnls)
        quintiles = np.percentile(atr_arr, [20, 40, 60, 80])
        labels = [
            f"Q1 (ATR<{quintiles[0]:.0f})",
            f"Q2 ({quintiles[0]:.0f}-{quintiles[1]:.0f})",
            f"Q3 ({quintiles[1]:.0f}-{quintiles[2]:.0f})",
            f"Q4 ({quintiles[2]:.0f}-{quintiles[3]:.0f})",
            f"Q5 (ATR>{quintiles[3]:.0f})",
        ]
        bins = [-np.inf] + list(quintiles) + [np.inf]
        for i in range(5):
            mask = (atr_arr >= bins[i]) & (atr_arr < bins[i + 1])
            q_trades = pnl_arr[mask]
            if len(q_trades) > 0:
                n_q = len(q_trades)
                wr_q = np.sum(q_trades > 0) / n_q * 100
                pnl_q = np.sum(q_trades)
                print(f"  {labels[i]:25s}  {n_q:3d}T  WR={wr_q:5.1f}%  PnL={pnl_q:+8.1f}")

    # === DAILY PnL BY REGIME ===
    print("\n" + "=" * 80)
    print("DAILY PnL DISTRIBUTION BY REGIME")
    print("=" * 80)
    for regime in ["low_vol", "normal", "high_vol"]:
        regime_days = [d for d in result.days if regimes.get(d.date) == regime]
        pnls = [d.pnl_pts for d in regime_days]
        if pnls:
            pos_days = sum(1 for p in pnls if p > 0)
            neg_days = sum(1 for p in pnls if p < 0)
            flat_days = sum(1 for p in pnls if p == 0)
            print(
                f"  {regime:10s}  {len(pnls):3d} days  "
                f"PnL={sum(pnls):+8.1f}  Avg={np.mean(pnls):+5.1f}  "
                f"Med={np.median(pnls):+5.1f}  "
                f"+days={pos_days} -days={neg_days} flat={flat_days}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
