"""FB Long Recovery Quality vs Trade Outcome Analysis.

Hypothesis: The quality of the bounce (recovery_ratio = recovery_pts / sweep_depth)
at the moment of FB entry predicts whether the trade wins or loses.

Runs the full 5-year backtest with production params, then for each FB Long trade
measures recovery quality metrics from the raw bar data at the entry bar.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import date, timedelta
from collections import defaultdict

from config.settings import (
    StrategyParams, ElevatorParams, ExitParams, RiskParams,
    SessionTimes, ESContractSpec,
)
from core.regime_filter import RegimeParams
from backtest.runner import BacktestRunner


def p(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def main():
    p("=" * 80)
    p("FB LONG RECOVERY QUALITY vs TRADE OUTCOME — 5yr Analysis")
    p("=" * 80)

    # Load data
    data_path = "data/ES_1m_full_session_2021-01-01_2026-02-05.parquet"
    p(f"\nLoading {data_path}...")
    df_all = pd.read_parquet(data_path)
    p(f"Loaded {len(df_all):,} bars, {df_all.index[0]} to {df_all.index[-1]}")

    # Production params (Optuna v2 — from MEMORY.md)
    strategy_params = StrategyParams(
        acceptance_max_dip_pts=15.0,
        acceptance_min_hold_bars=11,
        fb_stop_buffer_pts=6.0,
        lr_stop_buffer_pts=4.0,
        max_fb_sweep_depth_pts=20.0,
        max_target_distance_pts=30.0,
        signal_cooldown_bars=15,
        allow_breakdown_short=True,
        allow_backtest_short=False,
        use_regime_filter=True,
        allow_level_sweep_fb=True,
    )

    exit_params = ExitParams(
        fb_max_hold_bars=14,
    )

    risk_params = RiskParams(
        min_rr_ratio=0.8,
        max_stop_distance_pts=20.0,
        max_trades_per_day=999,
    )

    regime_params = RegimeParams(
        ema_span=30,
        slope_lookback=10,
        slope_threshold_atr_mult=0.325,
    )

    session = SessionTimes(
        rth_open=SessionTimes().globex_open,
        rth_close=SessionTimes().globex_close,
        eod_flatten_time=SessionTimes().globex_close,
    )

    contract = ESContractSpec()

    # Build daily DataFrames
    p("\nSplitting into daily DataFrames...")
    df_all.index = pd.to_datetime(df_all.index)
    dates = sorted(df_all.index.date)
    unique_dates = sorted(set(dates))
    p(f"Found {len(unique_dates)} unique dates")

    daily_dfs = {}
    for d in unique_dates:
        mask = df_all.index.date == d
        day_df = df_all[mask]
        if len(day_df) >= 10:
            daily_dfs[d] = day_df

    p(f"Valid trading days: {len(daily_dfs)}")

    # Build daily history for regime filter
    daily_rows = []
    for d in sorted(daily_dfs.keys()):
        day_df = daily_dfs[d]
        daily_rows.append({
            'date': pd.Timestamp(d),
            'open': day_df['open'].iloc[0],
            'high': day_df['high'].max(),
            'low': day_df['low'].min(),
            'close': day_df['close'].iloc[-1],
            'volume': day_df['volume'].sum(),
        })
    daily_history = pd.DataFrame(daily_rows).set_index('date')

    # Run backtest day by day, collecting trades + bar data
    p("\nRunning backtest with production params...")
    all_trades = []
    all_bar_data = {}  # date -> DataFrame for recovery analysis

    sorted_dates = sorted(daily_dfs.keys())
    prior_day_df = None

    for i, day in enumerate(sorted_dates):
        day_df = daily_dfs[day]

        # Get daily history up to (not including) today for regime
        day_ts = pd.Timestamp(day)
        dh = daily_history[daily_history.index < day_ts]

        runner = BacktestRunner(
            strategy_params=strategy_params,
            elevator_params=ElevatorParams(),
            exit_params=exit_params,
            risk_params=risk_params,
            session_times=session,
            contract=contract,
            min_rr_ratio=0.8,
        )
        runner.strategy.regime_params = regime_params
        runner.strategy._daily_history = dh

        day_result = runner.run_single_day(day_df, prior_day_df, day)

        for tr in day_result.trade_records:
            tr._day = day
            all_trades.append(tr)
            all_bar_data[day] = day_df

        prior_day_df = day_df

        if (i + 1) % 200 == 0:
            p(f"  Processed {i+1}/{len(sorted_dates)} days...")

    p(f"\nTotal trades: {len(all_trades)}")

    # Filter to FB Longs only
    fb_longs = [t for t in all_trades if t.pattern_type == "failed_breakdown" and t.direction == "long"]
    p(f"FB Long trades: {len(fb_longs)}")

    if not fb_longs:
        p("No FB Long trades found — aborting.")
        return

    # Overall FB Long stats
    wins = [t for t in fb_longs if t.pnl_pts > 0]
    losses = [t for t in fb_longs if t.pnl_pts <= 0]
    p(f"\nOverall FB Long: {len(wins)}W / {len(losses)}L = {len(wins)/len(fb_longs)*100:.1f}% WR")
    p(f"  Avg Win: {np.mean([t.pnl_pts for t in wins]):.1f} pts" if wins else "  No wins")
    p(f"  Avg Loss: {np.mean([t.pnl_pts for t in losses]):.1f} pts" if losses else "  No losses")
    total_pnl = sum(t.pnl_pts for t in fb_longs)
    p(f"  Total PnL: {total_pnl:+.1f} pts")

    # For each FB Long, compute recovery metrics from bar data
    p("\nComputing recovery quality metrics...")
    trade_metrics = []

    for t in fb_longs:
        day = t._day
        day_df = all_bar_data.get(day)
        if day_df is None:
            continue

        entry_bar = t.entry_bar_idx
        if entry_bar < 0 or entry_bar >= len(day_df):
            continue

        entry_price = t.entry_price
        level_price = t.level_price
        sweep_depth = t.sweep_depth_pts  # from PatternSignal

        # Compute session high/low up to entry bar
        bars_up_to_entry = day_df.iloc[:entry_bar + 1]
        session_high = bars_up_to_entry['high'].max()
        session_low = bars_up_to_entry['low'].min()
        session_range = session_high - session_low

        # Recovery: how far price bounced from sweep low back to entry
        sweep_low_price = level_price - sweep_depth if sweep_depth > 0 else level_price
        recovery_pts = entry_price - sweep_low_price

        # Recovery ratio: quality of bounce relative to sweep depth
        recovery_ratio = recovery_pts / sweep_depth if sweep_depth > 0.5 else 0.0

        # Position in session range
        position_in_range = (entry_price - session_low) / session_range if session_range > 1.0 else 0.5

        # Bars since sweep low: find the bar with the lowest low before entry
        if entry_bar > 0:
            lows_before_entry = bars_up_to_entry['low'].values
            sweep_low_bar = np.argmin(lows_before_entry)
            bars_since_sweep = entry_bar - sweep_low_bar
        else:
            bars_since_sweep = 0

        # Entry above/below level
        entry_vs_level = entry_price - level_price

        # Confirmation type
        conf_type = t.confirmation_type

        trade_metrics.append({
            'date': day,
            'entry_price': entry_price,
            'level_price': level_price,
            'sweep_depth_pts': sweep_depth,
            'recovery_pts': recovery_pts,
            'recovery_ratio': recovery_ratio,
            'session_range': session_range,
            'position_in_range': position_in_range,
            'bars_since_sweep': bars_since_sweep,
            'entry_vs_level': entry_vs_level,
            'pnl_pts': t.pnl_pts,
            'win': 1 if t.pnl_pts > 0 else 0,
            'exit_reason': t.exit_reason,
            'confirmation_type': conf_type,
            'level_type': t.level_type,
            'rr_ratio_t1': t.rr_ratio_t1,
            'risk_pts': t.risk_pts,
            'entry_bar_idx': entry_bar,
        })

    df_tm = pd.DataFrame(trade_metrics)
    p(f"Trades with computed metrics: {len(df_tm)}")

    # ===================================================================
    # ANALYSIS 1: Recovery Ratio Buckets
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 1: RECOVERY RATIO BUCKETS")
    p("  recovery_ratio = (entry_price - sweep_low) / sweep_depth")
    p("  Higher = stronger bounce relative to the sweep")
    p("=" * 80)

    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 100.0]
    labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-3.0', '3.0-5.0', '5.0+']
    df_tm['rr_bucket'] = pd.cut(df_tm['recovery_ratio'], bins=bins, labels=labels, right=False)

    p(f"\n{'Bucket':>10} | {'Trades':>6} | {'Win%':>6} | {'Avg PnL':>8} | {'Tot PnL':>8} | {'Avg Sweep':>9} | {'Avg Recovery':>12}")
    p("-" * 85)

    for bucket in labels:
        sub = df_tm[df_tm['rr_bucket'] == bucket]
        if len(sub) == 0:
            continue
        wr = sub['win'].mean() * 100
        avg_pnl = sub['pnl_pts'].mean()
        tot_pnl = sub['pnl_pts'].sum()
        avg_sweep = sub['sweep_depth_pts'].mean()
        avg_rec = sub['recovery_pts'].mean()
        p(f"{bucket:>10} | {len(sub):>6} | {wr:>5.1f}% | {avg_pnl:>+7.1f} | {tot_pnl:>+7.0f} | {avg_sweep:>8.1f} | {avg_rec:>11.1f}")

    # ===================================================================
    # ANALYSIS 2: Session Range Buckets
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 2: SESSION RANGE AT ENTRY")
    p("  session_range = session_high - session_low (up to entry bar)")
    p("=" * 80)

    range_bins = [0, 10, 20, 30, 40, 60, 80, 200]
    range_labels = ['0-10', '10-20', '20-30', '30-40', '40-60', '60-80', '80+']
    df_tm['range_bucket'] = pd.cut(df_tm['session_range'], bins=range_bins, labels=range_labels, right=False)

    p(f"\n{'Range':>10} | {'Trades':>6} | {'Win%':>6} | {'Avg PnL':>8} | {'Tot PnL':>8} | {'Avg Sweep':>9} | {'Avg RR':>7}")
    p("-" * 75)

    for bucket in range_labels:
        sub = df_tm[df_tm['range_bucket'] == bucket]
        if len(sub) == 0:
            continue
        wr = sub['win'].mean() * 100
        avg_pnl = sub['pnl_pts'].mean()
        tot_pnl = sub['pnl_pts'].sum()
        avg_sweep = sub['sweep_depth_pts'].mean()
        avg_rr = sub['recovery_ratio'].mean()
        p(f"{bucket:>10} | {len(sub):>6} | {wr:>5.1f}% | {avg_pnl:>+7.1f} | {tot_pnl:>+7.0f} | {avg_sweep:>8.1f} | {avg_rr:>6.1f}")

    # ===================================================================
    # ANALYSIS 3: Sweep Depth Buckets
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 3: SWEEP DEPTH BUCKETS")
    p("=" * 80)

    depth_bins = [0, 3, 5, 8, 12, 20, 100]
    depth_labels = ['0-3', '3-5', '5-8', '8-12', '12-20', '20+']
    df_tm['depth_bucket'] = pd.cut(df_tm['sweep_depth_pts'], bins=depth_bins, labels=depth_labels, right=False)

    p(f"\n{'Depth':>10} | {'Trades':>6} | {'Win%':>6} | {'Avg PnL':>8} | {'Tot PnL':>8} | {'Avg Recovery':>12}")
    p("-" * 72)

    for bucket in depth_labels:
        sub = df_tm[df_tm['depth_bucket'] == bucket]
        if len(sub) == 0:
            continue
        wr = sub['win'].mean() * 100
        avg_pnl = sub['pnl_pts'].mean()
        tot_pnl = sub['pnl_pts'].sum()
        avg_rec = sub['recovery_pts'].mean()
        p(f"{bucket:>10} | {len(sub):>6} | {wr:>5.1f}% | {avg_pnl:>+7.1f} | {tot_pnl:>+7.0f} | {avg_rec:>11.1f}")

    # ===================================================================
    # ANALYSIS 4: Position in Session Range
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 4: POSITION IN SESSION RANGE AT ENTRY")
    p("  0 = at session low, 1 = at session high")
    p("=" * 80)

    pos_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    pos_labels = ['0-0.2 (near low)', '0.2-0.4', '0.4-0.6 (mid)', '0.6-0.8', '0.8-1.0 (near high)']
    df_tm['pos_bucket'] = pd.cut(df_tm['position_in_range'], bins=pos_bins, labels=pos_labels, right=False)

    p(f"\n{'Position':>22} | {'Trades':>6} | {'Win%':>6} | {'Avg PnL':>8} | {'Tot PnL':>8}")
    p("-" * 62)

    for bucket in pos_labels:
        sub = df_tm[df_tm['pos_bucket'] == bucket]
        if len(sub) == 0:
            continue
        wr = sub['win'].mean() * 100
        avg_pnl = sub['pnl_pts'].mean()
        tot_pnl = sub['pnl_pts'].sum()
        p(f"{bucket:>22} | {len(sub):>6} | {wr:>5.1f}% | {avg_pnl:>+7.1f} | {tot_pnl:>+7.0f}")

    # ===================================================================
    # ANALYSIS 5: Bars Since Sweep Low
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 5: BARS SINCE SWEEP LOW (speed of recovery)")
    p("=" * 80)

    bar_bins = [0, 5, 10, 20, 40, 80, 500]
    bar_labels = ['0-5', '5-10', '10-20', '20-40', '40-80', '80+']
    df_tm['bars_bucket'] = pd.cut(df_tm['bars_since_sweep'], bins=bar_bins, labels=bar_labels, right=False)

    p(f"\n{'Bars':>10} | {'Trades':>6} | {'Win%':>6} | {'Avg PnL':>8} | {'Tot PnL':>8} | {'Avg RR':>7}")
    p("-" * 65)

    for bucket in bar_labels:
        sub = df_tm[df_tm['bars_bucket'] == bucket]
        if len(sub) == 0:
            continue
        wr = sub['win'].mean() * 100
        avg_pnl = sub['pnl_pts'].mean()
        tot_pnl = sub['pnl_pts'].sum()
        avg_rr = sub['recovery_ratio'].mean()
        p(f"{bucket:>10} | {len(sub):>6} | {wr:>5.1f}% | {avg_pnl:>+7.1f} | {tot_pnl:>+7.0f} | {avg_rr:>6.1f}")

    # ===================================================================
    # ANALYSIS 6: Confirmation Type
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 6: CONFIRMATION TYPE")
    p("=" * 80)

    for conf in df_tm['confirmation_type'].unique():
        sub = df_tm[df_tm['confirmation_type'] == conf]
        wr = sub['win'].mean() * 100
        avg_pnl = sub['pnl_pts'].mean()
        tot_pnl = sub['pnl_pts'].sum()
        avg_sweep = sub['sweep_depth_pts'].mean()
        avg_rr = sub['recovery_ratio'].mean()
        p(f"  {conf:20s}: {len(sub):>4} trades, {wr:>5.1f}% WR, avg {avg_pnl:>+6.1f}, tot {tot_pnl:>+7.0f}, avg_sweep={avg_sweep:.1f}, avg_rr={avg_rr:.1f}")

    # ===================================================================
    # ANALYSIS 7: Level Type
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 7: LEVEL TYPE")
    p("=" * 80)

    for lt in sorted(df_tm['level_type'].unique()):
        sub = df_tm[df_tm['level_type'] == lt]
        wr = sub['win'].mean() * 100
        avg_pnl = sub['pnl_pts'].mean()
        tot_pnl = sub['pnl_pts'].sum()
        p(f"  {lt:20s}: {len(sub):>4} trades, {wr:>5.1f}% WR, avg {avg_pnl:>+6.1f}, tot {tot_pnl:>+7.0f}")

    # ===================================================================
    # ANALYSIS 8: Cross-tabulation — Recovery Ratio x Session Range
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 8: CROSS-TAB — RECOVERY RATIO x SESSION RANGE")
    p("  Each cell: Trades / Win% / Total PnL")
    p("=" * 80)

    p(f"\n{'':>12}", end="")
    for rl in range_labels:
        p(f" | {rl:>18}", end="")
    p("")
    p("-" * (14 + 21 * len(range_labels)))

    for rr_b in labels:
        p(f"{rr_b:>12}", end="")
        for sr_b in range_labels:
            sub = df_tm[(df_tm['rr_bucket'] == rr_b) & (df_tm['range_bucket'] == sr_b)]
            if len(sub) == 0:
                p(f" | {'---':>18}", end="")
            else:
                wr = sub['win'].mean() * 100
                tot = sub['pnl_pts'].sum()
                p(f" | {len(sub):>2}T/{wr:>4.0f}%/{tot:>+5.0f}pt", end="")
        p("")

    # ===================================================================
    # ANALYSIS 9: Entry vs Level (how far above level at entry)
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 9: ENTRY vs LEVEL (entry_price - level_price)")
    p("  How far above the level is the entry?")
    p("=" * 80)

    evl_bins = [-100, 0, 2, 4, 6, 8, 12, 100]
    evl_labels = ['<0 (below)', '0-2', '2-4', '4-6', '6-8', '8-12', '12+']
    df_tm['evl_bucket'] = pd.cut(df_tm['entry_vs_level'], bins=evl_bins, labels=evl_labels, right=False)

    p(f"\n{'Entry-Level':>14} | {'Trades':>6} | {'Win%':>6} | {'Avg PnL':>8} | {'Tot PnL':>8}")
    p("-" * 55)

    for bucket in evl_labels:
        sub = df_tm[df_tm['evl_bucket'] == bucket]
        if len(sub) == 0:
            continue
        wr = sub['win'].mean() * 100
        avg_pnl = sub['pnl_pts'].mean()
        tot_pnl = sub['pnl_pts'].sum()
        p(f"{bucket:>14} | {len(sub):>6} | {wr:>5.1f}% | {avg_pnl:>+7.1f} | {tot_pnl:>+7.0f}")

    # ===================================================================
    # ANALYSIS 10: Composite — combining top features
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 10: COMPOSITE FILTERS — Finding the Losers")
    p("=" * 80)

    # Test various combinations
    filters = [
        ("All FB Longs", df_tm),
        ("recovery_ratio < 1.0", df_tm[df_tm['recovery_ratio'] < 1.0]),
        ("recovery_ratio >= 1.0", df_tm[df_tm['recovery_ratio'] >= 1.0]),
        ("recovery_ratio < 1.0 AND session_range < 20", df_tm[(df_tm['recovery_ratio'] < 1.0) & (df_tm['session_range'] < 20)]),
        ("recovery_ratio < 1.0 AND session_range >= 20", df_tm[(df_tm['recovery_ratio'] < 1.0) & (df_tm['session_range'] >= 20)]),
        ("recovery_ratio >= 1.5 AND session_range >= 20", df_tm[(df_tm['recovery_ratio'] >= 1.5) & (df_tm['session_range'] >= 20)]),
        ("sweep_depth < 3 (shallow)", df_tm[df_tm['sweep_depth_pts'] < 3]),
        ("sweep_depth >= 5 AND recovery_ratio >= 1.0", df_tm[(df_tm['sweep_depth_pts'] >= 5) & (df_tm['recovery_ratio'] >= 1.0)]),
        ("entry_vs_level > 8 (far from level)", df_tm[df_tm['entry_vs_level'] > 8]),
        ("entry_vs_level <= 4 (close to level)", df_tm[df_tm['entry_vs_level'] <= 4]),
        ("bars_since_sweep < 10 (fast)", df_tm[df_tm['bars_since_sweep'] < 10]),
        ("bars_since_sweep >= 40 (slow)", df_tm[df_tm['bars_since_sweep'] >= 40]),
        ("BEST: sweep>=3 AND rr>=1.0 AND range>=15", df_tm[(df_tm['sweep_depth_pts'] >= 3) & (df_tm['recovery_ratio'] >= 1.0) & (df_tm['session_range'] >= 15)]),
        ("WORST: sweep<3 OR (rr<0.5 AND range<15)", df_tm[(df_tm['sweep_depth_pts'] < 3) | ((df_tm['recovery_ratio'] < 0.5) & (df_tm['session_range'] < 15))]),
    ]

    p(f"\n{'Filter':>55} | {'Trades':>6} | {'Win%':>6} | {'Avg PnL':>8} | {'Tot PnL':>8} | {'PF':>6}")
    p("-" * 102)

    for label, sub in filters:
        if len(sub) == 0:
            p(f"{label:>55} | {0:>6} |    --- |      --- |      --- |    ---")
            continue
        wr = sub['win'].mean() * 100
        avg_pnl = sub['pnl_pts'].mean()
        tot_pnl = sub['pnl_pts'].sum()
        gross_w = sub[sub['pnl_pts'] > 0]['pnl_pts'].sum()
        gross_l = abs(sub[sub['pnl_pts'] <= 0]['pnl_pts'].sum())
        pf = gross_w / gross_l if gross_l > 0 else float('inf')
        p(f"{label:>55} | {len(sub):>6} | {wr:>5.1f}% | {avg_pnl:>+7.1f} | {tot_pnl:>+7.0f} | {pf:>5.2f}")

    # ===================================================================
    # ANALYSIS 11: Entry Bar Time of Day
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 11: ENTRY HOUR (ET)")
    p("=" * 80)

    df_tm['entry_hour'] = df_tm['date'].apply(lambda d: None)  # placeholder
    # Reconstruct from bar data
    hour_metrics = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    for _, row in df_tm.iterrows():
        day = row['date']
        bar_idx = row['entry_bar_idx']
        day_df = all_bar_data.get(day)
        if day_df is not None and bar_idx < len(day_df):
            ts = day_df.index[bar_idx]
            hour = ts.hour
            hour_metrics[hour]['trades'] += 1
            hour_metrics[hour]['wins'] += row['win']
            hour_metrics[hour]['pnl'] += row['pnl_pts']

    p(f"\n{'Hour':>6} | {'Trades':>6} | {'Win%':>6} | {'Tot PnL':>8}")
    p("-" * 38)
    for h in sorted(hour_metrics.keys()):
        m = hour_metrics[h]
        wr = m['wins'] / m['trades'] * 100 if m['trades'] > 0 else 0
        p(f"{h:>6} | {m['trades']:>6} | {wr:>5.1f}% | {m['pnl']:>+7.0f}")

    # ===================================================================
    # ANALYSIS 12: Win/Loss Distribution Details
    # ===================================================================
    p("\n" + "=" * 80)
    p("ANALYSIS 12: WIN/LOSS DISTRIBUTION")
    p("=" * 80)

    winners = df_tm[df_tm['pnl_pts'] > 0]
    losers = df_tm[df_tm['pnl_pts'] <= 0]

    p(f"\nWinners ({len(winners)}):")
    if len(winners) > 0:
        p(f"  PnL:    mean={winners['pnl_pts'].mean():+.1f}, median={winners['pnl_pts'].median():+.1f}, "
          f"min={winners['pnl_pts'].min():+.1f}, max={winners['pnl_pts'].max():+.1f}")
        p(f"  Sweep:  mean={winners['sweep_depth_pts'].mean():.1f}, median={winners['sweep_depth_pts'].median():.1f}")
        p(f"  RecRat: mean={winners['recovery_ratio'].mean():.2f}, median={winners['recovery_ratio'].median():.2f}")
        p(f"  Range:  mean={winners['session_range'].mean():.1f}, median={winners['session_range'].median():.1f}")
        p(f"  Bars:   mean={winners['bars_since_sweep'].mean():.0f}, median={winners['bars_since_sweep'].median():.0f}")

    p(f"\nLosers ({len(losers)}):")
    if len(losers) > 0:
        p(f"  PnL:    mean={losers['pnl_pts'].mean():+.1f}, median={losers['pnl_pts'].median():+.1f}, "
          f"min={losers['pnl_pts'].min():+.1f}, max={losers['pnl_pts'].max():+.1f}")
        p(f"  Sweep:  mean={losers['sweep_depth_pts'].mean():.1f}, median={losers['sweep_depth_pts'].median():.1f}")
        p(f"  RecRat: mean={losers['recovery_ratio'].mean():.2f}, median={losers['recovery_ratio'].median():.2f}")
        p(f"  Range:  mean={losers['session_range'].mean():.1f}, median={losers['session_range'].median():.1f}")
        p(f"  Bars:   mean={losers['bars_since_sweep'].mean():.0f}, median={losers['bars_since_sweep'].median():.0f}")

    # Statistical test: recovery_ratio difference between wins and losses
    if len(winners) > 5 and len(losers) > 5:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(winners['recovery_ratio'], losers['recovery_ratio'])
        p(f"\nt-test recovery_ratio (winners vs losers): t={t_stat:.2f}, p={p_val:.4f}")

        t_stat2, p_val2 = stats.ttest_ind(winners['sweep_depth_pts'], losers['sweep_depth_pts'])
        p(f"t-test sweep_depth (winners vs losers): t={t_stat2:.2f}, p={p_val2:.4f}")

        t_stat3, p_val3 = stats.ttest_ind(winners['session_range'], losers['session_range'])
        p(f"t-test session_range (winners vs losers): t={t_stat3:.2f}, p={p_val3:.4f}")

        t_stat4, p_val4 = stats.ttest_ind(winners['bars_since_sweep'], losers['bars_since_sweep'])
        p(f"t-test bars_since_sweep (winners vs losers): t={t_stat4:.2f}, p={p_val4:.4f}")

        t_stat5, p_val5 = stats.ttest_ind(winners['entry_vs_level'], losers['entry_vs_level'])
        p(f"t-test entry_vs_level (winners vs losers): t={t_stat5:.2f}, p={p_val5:.4f}")

    p("\n" + "=" * 80)
    p("DONE")
    p("=" * 80)


if __name__ == "__main__":
    main()
