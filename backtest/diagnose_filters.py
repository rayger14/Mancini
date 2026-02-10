"""Diagnose where trades are being filtered out.

For each filter/gate in the pipeline, count how many signals it blocks
and what quality those blocked signals would have been.
"""
import sys
from datetime import time as dtime
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger
logger.remove()

from backtest.runner import BacktestRunner
from config.settings import (
    StrategyParams, ElevatorParams, ExitParams,
    RiskParams, SessionTimes,
)
from core.signals import SignalAggregator
from core.indicators import enrich_dataframe

PRODUCTION_PARAMS = {
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


def build_components(params):
    p = params
    strategy_params = StrategyParams(
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
    elevator_params = ElevatorParams(
        min_velocity_pts_per_min=p["min_velocity"],
        min_levels_broken=p["min_levels_broken"],
        higher_low_lookback=p["higher_low_lookback"],
    )
    exit_params = ExitParams(
        t1_exit_fraction=p["t1_exit_fraction"],
        trailing_stop_pts=p["trailing_stop_pts"],
        default_contracts=p.get("contracts", 4),
    )
    risk_params = RiskParams(max_trades_per_day=p["max_trades_per_day"])
    session = SessionTimes(
        chop_zone_start=dtime(p["chop_start_hour"], 0),
        chop_zone_end=dtime(p["chop_end_hour"], 0),
    )
    return strategy_params, elevator_params, exit_params, risk_params, session


def run_signal_only_backtest(daily_dfs, params):
    """Run JUST the signal aggregator (no risk gates) to see raw signal production."""
    strategy_params, elevator_params, exit_params, risk_params, session = build_components(params)

    all_signals = []
    dates = sorted(daily_dfs.keys())
    prior_day_df = None

    for day in dates:
        df = daily_dfs[day]
        agg = SignalAggregator(
            strategy_params=strategy_params,
            elevator_params=elevator_params,
            exit_params=exit_params,
            min_rr_ratio=params["min_rr_ratio"],
        )
        agg.reset()
        agg.initialize_levels(df, prior_day_df)

        enriched = enrich_dataframe(df)
        velocity = enriched["velocity_5"]

        for i in range(len(df)):
            vel = float(velocity.iat[i])
            if vel != vel:
                vel = 0.0
            signal = agg.update(
                bar_idx=i,
                timestamp=df.index[i],
                open_=float(df["open"].iat[i]),
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                volume=float(df["volume"].iat[i]),
                velocity=vel,
                df=df,
            )
            if signal is not None:
                all_signals.append({
                    "signal": signal,
                    "day": day,
                    "bar_idx": i,
                    "time": df.index[i].time(),
                    "df": df,
                })
        prior_day_df = df

    return all_signals


def simulate_signal_outcome(signal_info, df):
    """Given a signal, simulate what would happen if we took the trade.
    Simple: track forward from entry, check if T1 or stop hit first."""
    sig = signal_info["signal"]
    entry = sig.entry_price
    stop = sig.stop_price
    t1 = sig.target_1
    bar_start = signal_info["bar_idx"]

    for i in range(bar_start + 1, len(df)):
        high = float(df["high"].iat[i])
        low = float(df["low"].iat[i])

        # Stop hit
        if low <= stop:
            pnl = stop - entry
            return {"result": "stop", "pnl_per_contract": pnl, "bars": i - bar_start}
        # Target hit
        if high >= t1:
            pnl = t1 - entry
            return {"result": "target", "pnl_per_contract": pnl, "bars": i - bar_start}

    # EOD — exit at last close
    last_close = float(df["close"].iat[-1])
    pnl = last_close - entry
    return {"result": "eod", "pnl_per_contract": pnl, "bars": len(df) - bar_start}


def main():
    data_path = Path(__file__).parent.parent / "data" / "ES_1m_2024-02-05_2026-02-05.parquet"
    print("Loading data...")
    daily_dfs = load_daily_dfs(str(data_path))

    # Skip Mondays
    filtered = {d: df for d, df in daily_dfs.items() if d.weekday() != 0}
    print(f"Total days: {len(daily_dfs)}, after Monday filter: {len(filtered)}")

    # ── Step 1: Raw signal production (no risk gates) ─────────────
    print("\n" + "=" * 80)
    print("STEP 1: RAW SIGNAL PRODUCTION (before any risk/entry filters)")
    print("=" * 80)

    print("\nRunning signal-only scan...")
    all_signals = run_signal_only_backtest(filtered, PRODUCTION_PARAMS)
    print(f"\nTotal raw signals generated: {len(all_signals)}")

    # Break down by type
    fb_signals = [s for s in all_signals if s["signal"].signal_type.name == "FAILED_BREAKDOWN"]
    lr_signals = [s for s in all_signals if s["signal"].signal_type.name == "LEVEL_RECLAIM"]
    print(f"  Failed Breakdown: {len(fb_signals)}")
    print(f"  Level Reclaim: {len(lr_signals)}")

    # Time distribution
    print(f"\nSignals by hour:")
    hour_counts = Counter()
    for s in all_signals:
        hour_counts[s["time"].hour] += 1
    for h in sorted(hour_counts.keys()):
        bar = "█" * hour_counts[h]
        print(f"  {h:02d}:00  {hour_counts[h]:3d}  {bar}")

    # ── Step 2: Filter analysis ───────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 2: WHAT KILLS EACH SIGNAL?")
    print("=" * 80)

    strategy_params, elevator_params, exit_params, risk_params, session = build_components(PRODUCTION_PARAMS)

    filter_kills = Counter()
    filter_would_win = Counter()  # signals killed by filter that would have won
    filter_would_lose = Counter()

    passed_signals = []

    for sig_info in all_signals:
        sig = sig_info["signal"]
        t = sig_info["time"]

        # Simulate outcome regardless of filters
        outcome = simulate_signal_outcome(sig_info, sig_info["df"])

        # Check each filter
        killed_by = None

        # Chop zone
        if session.in_chop_zone(t):
            killed_by = "chop_zone"

        # EOD flatten
        elif session.past_eod_flatten(t):
            killed_by = "eod_flatten"

        # R:R check (already applied in signal aggregator via min_rr_ratio)
        # Signals that make it here already passed R:R

        # Stop too wide
        elif sig.risk_pts > 10.0:
            killed_by = "stop_too_wide"

        # R:R below 1.0 (risk manager hard check)
        elif sig.rr_ratio_t1 < 1.0:
            killed_by = "rr_below_1"

        if killed_by:
            filter_kills[killed_by] += 1
            if outcome["result"] == "target":
                filter_would_win[killed_by] += 1
            else:
                filter_would_lose[killed_by] += 1
        else:
            passed_signals.append((sig_info, outcome))

    print(f"\n{'Filter':<20} {'Blocked':>8} {'Would Win':>10} {'Would Lose':>11} {'WR if Taken':>12}")
    print("-" * 65)
    for filt in sorted(filter_kills.keys(), key=lambda k: filter_kills[k], reverse=True):
        blocked = filter_kills[filt]
        wins = filter_would_win[filt]
        losses = filter_would_lose[filt]
        wr = wins / blocked * 100 if blocked > 0 else 0
        print(f"  {filt:<18} {blocked:>8} {wins:>10} {losses:>11} {wr:>11.0f}%")

    total_blocked = sum(filter_kills.values())
    total_would_win = sum(filter_would_win.values())
    total_would_lose = sum(filter_would_lose.values())
    print("-" * 65)
    print(f"  {'TOTAL BLOCKED':<18} {total_blocked:>8} {total_would_win:>10} {total_would_lose:>11} "
          f"{total_would_win/total_blocked*100 if total_blocked > 0 else 0:>11.0f}%")
    print(f"  {'PASSED':<18} {len(passed_signals):>8}", end="")
    passed_wins = sum(1 for _, o in passed_signals if o["result"] == "target")
    passed_losses = len(passed_signals) - passed_wins
    print(f" {passed_wins:>10} {passed_losses:>11} "
          f"{passed_wins/len(passed_signals)*100 if passed_signals else 0:>11.0f}%")

    # ── Step 3: What if we relaxed each filter? ───────────────────
    print("\n" + "=" * 80)
    print("STEP 3: IMPACT OF RELAXING EACH FILTER")
    print("=" * 80)

    # Current baseline
    baseline_wins = passed_wins
    baseline_total = len(passed_signals)
    baseline_pnl = sum(o["pnl_per_contract"] for _, o in passed_signals)

    print(f"\n  Current baseline: {baseline_total} trades, "
          f"{baseline_wins}/{baseline_total} wins ({baseline_wins/baseline_total*100:.0f}% WR), "
          f"{baseline_pnl:+.1f} pts/contract")

    for filt in sorted(filter_kills.keys(), key=lambda k: filter_kills[k], reverse=True):
        extra_trades = filter_kills[filt]
        extra_wins = filter_would_win[filt]

        # Reconstruct P&L for the blocked signals
        extra_pnl = 0
        for sig_info in all_signals:
            sig = sig_info["signal"]
            t = sig_info["time"]

            # Check if this specific signal was killed by this filter
            is_this_filter = False
            if filt == "chop_zone" and session.in_chop_zone(t):
                is_this_filter = True
            elif filt == "eod_flatten" and session.past_eod_flatten(t):
                is_this_filter = True
            elif filt == "stop_too_wide" and sig.risk_pts > 10.0:
                is_this_filter = True
            elif filt == "rr_below_1" and sig.rr_ratio_t1 < 1.0:
                is_this_filter = True

            if is_this_filter:
                outcome = simulate_signal_outcome(sig_info, sig_info["df"])
                extra_pnl += outcome["pnl_per_contract"]

        new_total = baseline_total + extra_trades
        new_wins = baseline_wins + extra_wins
        new_pnl = baseline_pnl + extra_pnl
        new_wr = new_wins / new_total * 100 if new_total > 0 else 0
        delta_pnl = extra_pnl

        print(f"\n  Remove '{filt}': +{extra_trades} trades")
        print(f"    New total: {new_total} trades, WR: {new_wr:.0f}%, PnL/ctr: {new_pnl:+.1f} pts (delta: {delta_pnl:+.1f})")
        if extra_trades > 0:
            blocked_wr = extra_wins / extra_trades * 100
            print(f"    Those blocked trades: {blocked_wr:.0f}% WR, avg P&L: {extra_pnl/extra_trades:+.1f} pts/ctr")

    # ── Step 4: Why aren't more signals generated? ────────────────
    print("\n" + "=" * 80)
    print("STEP 4: WHY SO FEW RAW SIGNALS?")
    print("=" * 80)

    # Count days with 0 signals
    signal_days = set(s["day"] for s in all_signals)
    no_signal_days = len(filtered) - len(signal_days)
    print(f"\n  Trading days: {len(filtered)}")
    print(f"  Days with signals: {len(signal_days)} ({len(signal_days)/len(filtered)*100:.0f}%)")
    print(f"  Days with NO signals: {no_signal_days} ({no_signal_days/len(filtered)*100:.0f}%)")

    # Signals per day distribution
    signals_per_day = Counter(s["day"] for s in all_signals)
    dist = Counter(signals_per_day.values())
    print(f"\n  Signals per day distribution:")
    for n in sorted(dist.keys()):
        print(f"    {n} signals: {dist[n]} days")

    # ── Step 5: Pattern-level diagnostics ─────────────────────────
    print("\n" + "=" * 80)
    print("STEP 5: PATTERN-LEVEL BOTTLENECKS")
    print("=" * 80)

    # For FB: elevator events vs signals
    # We need to count elevator events separately
    from core.elevator_down import ElevatorDownDetector
    from core.price_levels import PriceLevelDetector
    from config.levels import LevelStore

    total_elevators = 0
    elevators_with_enough_levels = 0
    _elevators_with_sweep = 0  # noqa: F841

    dates = sorted(filtered.keys())
    prior_day_df = None
    sp = build_components(PRODUCTION_PARAMS)[0]
    ep = ElevatorParams(
        min_velocity_pts_per_min=PRODUCTION_PARAMS["min_velocity"],
        min_levels_broken=PRODUCTION_PARAMS["min_levels_broken"],
        higher_low_lookback=PRODUCTION_PARAMS["higher_low_lookback"],
    )

    for day in dates:
        df = filtered[day]
        detector = ElevatorDownDetector(ep)
        level_detector = PriceLevelDetector(sp)
        store = LevelStore()
        if prior_day_df is not None:
            level_detector._add_prior_day_levels(store, prior_day_df)

        enriched = enrich_dataframe(df)
        velocity = enriched["velocity_5"]

        for i in range(len(df)):
            vel = float(velocity.iat[i])
            if vel != vel:
                vel = 0.0
            level_detector.detect_incremental(store, df, i)
            event = detector.update(
                bar_idx=i,
                timestamp=df.index[i],
                high=float(df["high"].iat[i]),
                low=float(df["low"].iat[i]),
                close=float(df["close"].iat[i]),
                velocity=vel,
                level_store=store,
            )
            if event is not None and event.is_complete:
                total_elevators += 1
                if event.levels_broken >= PRODUCTION_PARAMS["min_levels_broken"]:
                    elevators_with_enough_levels += 1

        prior_day_df = df

    print(f"\n  FAILED BREAKDOWN pipeline:")
    print(f"    Total elevator events detected: {total_elevators}")
    print(f"    With >= {PRODUCTION_PARAMS['min_levels_broken']} levels broken: {elevators_with_enough_levels}")
    print(f"    Produced FB signals: {len(fb_signals)}")
    print(f"    Conversion rate: elevator → FB signal = {len(fb_signals)/total_elevators*100:.1f}%" if total_elevators > 0 else "    No elevators")

    print(f"\n  LEVEL RECLAIM pipeline:")
    print(f"    Produced LR signals: {len(lr_signals)}")
    print(f"    Requires: {PRODUCTION_PARAMS['level_reclaim_min_touches']}+ touches on horizontal S/R")
    print(f"    Requires: acceptance hold of {PRODUCTION_PARAMS['acceptance_min_hold_bars']} bars")

    # ── Step 6: What-if scenarios for more trades ─────────────────
    print("\n" + "=" * 80)
    print("STEP 6: WHAT-IF SCENARIOS FOR MORE TRADES")
    print("=" * 80)

    scenarios = [
        ("Current production params", PRODUCTION_PARAMS),
        ("Lower min_levels_broken to 1", {**PRODUCTION_PARAMS, "min_levels_broken": 1}),
        ("Lower min_levels_broken to 0", {**PRODUCTION_PARAMS, "min_levels_broken": 0}),
        ("Lower acceptance_min_hold_bars to 4", {**PRODUCTION_PARAMS, "acceptance_min_hold_bars": 4}),
        ("Lower acceptance_min_hold_bars to 3", {**PRODUCTION_PARAMS, "acceptance_min_hold_bars": 3}),
        ("Lower level_reclaim_min_touches to 3", {**PRODUCTION_PARAMS, "level_reclaim_min_touches": 3}),
        ("Lower level_reclaim_min_touches to 2", {**PRODUCTION_PARAMS, "level_reclaim_min_touches": 2}),
        ("Lower swing_low_order to 10", {**PRODUCTION_PARAMS, "swing_low_order": 10}),
        ("Lower min_rr_ratio to 0.75", {**PRODUCTION_PARAMS, "min_rr_ratio": 0.75}),
        ("Lower min_rr_ratio to 0.5", {**PRODUCTION_PARAMS, "min_rr_ratio": 0.5}),
        ("Widen chop zone to 13-15", {**PRODUCTION_PARAMS, "chop_start_hour": 13, "chop_end_hour": 15}),
        ("Remove chop zone entirely", {**PRODUCTION_PARAMS, "chop_start_hour": 15, "chop_end_hour": 16}),
        # Combos
        ("COMBO: hold=4, touches=3, chop=13-15", {
            **PRODUCTION_PARAMS,
            "acceptance_min_hold_bars": 4,
            "level_reclaim_min_touches": 3,
            "chop_start_hour": 13, "chop_end_hour": 15,
        }),
        ("COMBO: hold=3, touches=3, no chop", {
            **PRODUCTION_PARAMS,
            "acceptance_min_hold_bars": 3,
            "level_reclaim_min_touches": 3,
            "chop_start_hour": 15, "chop_end_hour": 16,
        }),
    ]

    print(f"\n{'Scenario':<45} {'Trades':>7} {'WR':>6} {'PF':>6} {'PnL':>8} {'PnL/T':>7}")
    print("-" * 85)

    for name, params in scenarios:
        result = run_full_backtest(filtered, params)
        trades = result.total_trades
        wr = result.win_rate
        pf = result.profit_factor
        pnl = result.total_pnl_pts
        per_trade = pnl / trades if trades > 0 else 0

        marker = ""
        if trades > 80:
            marker = " ←"
        print(f"  {name:<43} {trades:>7} {wr:>5.0%} {pf:>6.2f} {pnl:>+8.0f} {per_trade:>+7.1f}{marker}")


def run_full_backtest(daily_dfs, params):
    """Full backtest through the strategy engine."""
    p = params
    strategy_params = StrategyParams(
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
        strategy_params=strategy_params,
        elevator_params=elevator,
        exit_params=exit_params,
        risk_params=risk,
        session_times=session,
        min_rr_ratio=p["min_rr_ratio"],
    )
    return runner.run_multi_day(daily_dfs=daily_dfs)


if __name__ == "__main__":
    main()
