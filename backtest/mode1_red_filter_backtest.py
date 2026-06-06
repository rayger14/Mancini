"""A/B backtest of the Mode 1 Red FB-block filter.

Replays the engine on 2024-2026 sessions TWICE — once with
``use_mode1_detection=False`` (baseline, what we ran in production
until tonight) and once with ``True`` (the just-shipped filter).

For each emitted FB long signal, simulates the trade outcome from the
actual bar data — stop hit or T1 hit, EOD if neither. Aggregates PnL
and reports the delta between the two runs.

Specifically calls out:
  * sessions where Mode 1 Red was detected
  * the FB long entries that fired on those sessions in the baseline
  * the outcomes of those entries (if the filter blocked, what was lost
    or saved)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

import pandas as pd

from backtest.nautilus_production_5y import load_data, build_daily_sessions
from core.indicators import enrich_dataframe
from core.signals import SignalAggregator, SignalType
from live.ib_runner import (
    PRODUCTION_STRATEGY, PRODUCTION_ELEVATOR, PRODUCTION_EXIT, PRODUCTION_RISK,
)


@dataclass
class TradeOutcome:
    """One simulated trade."""
    session_date: date
    signal_type: str
    entry_bar: int
    entry_price: float
    stop_price: float
    target_price: float
    direction: str
    result: str       # "target" | "stop" | "eod"
    exit_bar: int
    exit_price: float
    pnl_pts: float
    mode1_red_at_entry: bool


def simulate_outcome(
    df: pd.DataFrame, entry_bar: int,
    entry: float, stop: float, target: float, direction: str,
) -> tuple[str, int, float]:
    """Walk forward from entry_bar+1; return (result, exit_bar, exit_price)."""
    n = len(df)
    for i in range(entry_bar + 1, n):
        high = float(df["high"].iat[i])
        low = float(df["low"].iat[i])
        if direction == "long":
            # Stop hit?
            if low <= stop:
                return "stop", i, stop
            # Target hit?
            if high >= target:
                return "target", i, target
        else:
            if high >= stop:
                return "stop", i, stop
            if low <= target:
                return "target", i, target
    # EOD without resolution
    return "eod", n - 1, float(df["close"].iat[-1])


def replay_session(
    df: pd.DataFrame, prior_df: pd.DataFrame | None,
    *,
    strategy_params, elevator_params, exit_params, min_rr_ratio: float,
) -> list[TradeOutcome]:
    """Run SignalAggregator over one session, simulate every emitted signal."""
    agg = SignalAggregator(
        strategy_params=strategy_params,
        elevator_params=elevator_params,
        exit_params=exit_params,
        min_rr_ratio=min_rr_ratio,
    )
    agg.reset()
    agg.initialize_levels(df, prior_df)

    enriched = enrich_dataframe(df)
    velocity = enriched["velocity_5"]

    outcomes: list[TradeOutcome] = []
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
        if signal is None:
            continue
        # We only care about FB longs for the Mode 1 Red gate analysis,
        # but record everything for context.
        is_long = signal.direction == "long"
        entry = signal.entry_price
        stop = signal.stop_price
        target = signal.target_1
        result, exit_bar, exit_px = simulate_outcome(
            df, i, entry, stop, target, signal.direction,
        )
        if is_long:
            pnl = exit_px - entry
        else:
            pnl = entry - exit_px

        outcomes.append(TradeOutcome(
            session_date=df.index[i].date(),
            signal_type=signal.signal_type.name,
            entry_bar=i,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            direction=signal.direction,
            result=result,
            exit_bar=exit_bar,
            exit_price=exit_px,
            pnl_pts=pnl,
            mode1_red_at_entry=bool(agg.mode1_red_active),
        ))
    return outcomes


def summarize(label: str, trades: list[TradeOutcome]) -> dict:
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl_pts > 0)
    total = sum(t.pnl_pts for t in trades)
    print(f"\n{label}:")
    print(f"  Trades:    {n}")
    if n:
        print(f"  Win rate:  {wins}/{n} = {wins/n*100:.1f}%")
        print(f"  Total PnL: {total:+.1f} pts (${total*50:+,.0f} on ES per 1ct)")
        print(f"  Avg PnL:   {total/n:+.2f} pts/trade")
    by_type = defaultdict(list)
    for t in trades:
        by_type[t.signal_type].append(t)
    if len(by_type) > 1:
        print(f"  Per signal type:")
        for st, ts in sorted(by_type.items(), key=lambda x: -len(x[1])):
            tp = sum(t.pnl_pts for t in ts)
            wr = sum(1 for t in ts if t.pnl_pts > 0) / len(ts) * 100
            print(f"    {st:>22}: n={len(ts):>4}  WR={wr:>5.1f}%  PnL={tp:+8.1f}")
    return {"n": n, "wins": wins, "total_pnl": total}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-06-27",
                    help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2026-02-05",
                    help="End date (YYYY-MM-DD)")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print("Loading 5y ES data…")
    df_all = load_data()
    sessions = build_daily_sessions(df_all)
    target_days = sorted(d for d in sessions if start <= d <= end)
    print(f"Sessions {start} → {end}: {len(target_days)}")

    # Build both param sets
    baseline_params = replace(PRODUCTION_STRATEGY, use_mode1_detection=False)
    with_filter_params = replace(PRODUCTION_STRATEGY, use_mode1_detection=True)

    baseline_trades: list[TradeOutcome] = []
    filter_trades: list[TradeOutcome] = []

    prior_d = None
    for i, d in enumerate(target_days):
        df_sess = sessions[d]
        if len(df_sess) < 30:
            prior_d = d
            continue
        prior_df = sessions[prior_d] if prior_d is not None else None
        try:
            baseline_trades.extend(replay_session(
                df_sess, prior_df,
                strategy_params=baseline_params,
                elevator_params=PRODUCTION_ELEVATOR,
                exit_params=PRODUCTION_EXIT,
                min_rr_ratio=PRODUCTION_RISK.min_rr_ratio,
            ))
            filter_trades.extend(replay_session(
                df_sess, prior_df,
                strategy_params=with_filter_params,
                elevator_params=PRODUCTION_ELEVATOR,
                exit_params=PRODUCTION_EXIT,
                min_rr_ratio=PRODUCTION_RISK.min_rr_ratio,
            ))
        except Exception as e:
            print(f"  {d}: error {e!r}")
        prior_d = d
        if (i + 1) % 50 == 0:
            print(f"  [{i+1:>3}/{len(target_days)}] "
                  f"baseline n={len(baseline_trades)} "
                  f"filter n={len(filter_trades)}",
                  flush=True)

    # ---------- Top-line A/B ----------
    print("\n" + "=" * 70)
    print(f"A/B RESULT  {start} → {end}")
    print("=" * 70)
    base = summarize("BASELINE  (use_mode1_detection=False)", baseline_trades)
    filt = summarize("WITH MODE 1 RED FILTER  (use_mode1_detection=True)", filter_trades)

    delta_n = filt["n"] - base["n"]
    delta_pnl = filt["total_pnl"] - base["total_pnl"]
    print("\n" + "-" * 70)
    print(f"DELTA  Trades: {delta_n:+d}    PnL: {delta_pnl:+.1f} pts "
          f"(${delta_pnl*50:+,.0f} ES/1ct)")

    # ---------- Mode 1 Red session zoom ----------
    print("\n" + "=" * 70)
    print("MODE 1 RED SESSIONS — what the filter blocked + how it would have played")
    print("=" * 70)
    # Find sessions where the baseline had at least one FB long emitted
    # AND where the mode1 detector was active at some bar in baseline.
    sessions_with_red = defaultdict(list)
    for t in baseline_trades:
        if (t.signal_type == "FAILED_BREAKDOWN"
                and t.direction == "long"
                and t.mode1_red_at_entry):
            sessions_with_red[t.session_date].append(t)

    if not sessions_with_red:
        print("No FB long entries fired during a Mode 1 Red bar in baseline.")
    else:
        print(f"Sessions with FB long entries during Mode 1 Red: "
              f"{len(sessions_with_red)}")
        print(f"\n  {'date':<12} {'n':>3} {'wins':>5} {'total_pnl':>10}")
        all_red_pnl = 0.0
        n_total = 0
        n_wins = 0
        for d in sorted(sessions_with_red):
            ts = sessions_with_red[d]
            wr = sum(1 for t in ts if t.pnl_pts > 0)
            tp = sum(t.pnl_pts for t in ts)
            all_red_pnl += tp
            n_total += len(ts)
            n_wins += wr
            print(f"  {d}  {len(ts):>3} {wr:>5} {tp:>+9.1f}")
        print(f"\n  TOTAL during Mode 1 Red bars: "
              f"n={n_total}  WR={n_wins/max(n_total,1)*100:.1f}%  "
              f"PnL={all_red_pnl:+.1f} pts (${all_red_pnl*50:+,.0f} ES)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
