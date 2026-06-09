"""A/B backtest of the C1' danger zone carve-out.

Hypothesis (Mancini's verbatim rule):
  "5 pts above swept low is danger zone; use non-acceptance protocol or
   wait for clear acceptance."

Current behavior: hard-block ANY FB long whose entry falls in a Mancini
danger zone, including signals that already qualified via non-acceptance.

C1' carve-out: allow non-acceptance-protocol signals through the danger
zone gate. Keep acceptance-protocol blocking (5y analysis shows
acceptance trades lose money).

Method:
  For each historical date where we have a Haiku-extracted plan and
  bar data, replay SignalAggregator twice with same params except
  danger_zone_allow_non_acceptance flipped. Capture every emitted FB
  signal and simulate the trade outcome from real bars (stop hit /
  target hit / EOD). Report PnL delta.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, time as dt_time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

import pandas as pd

from backtest.nautilus_production_5y import load_data, build_daily_sessions
from core.indicators import enrich_dataframe
from core.signals import SignalAggregator
from live.mancini_llm_extract import load_plan
from live.ib_runner import (
    PRODUCTION_STRATEGY, PRODUCTION_ELEVATOR, PRODUCTION_EXIT, PRODUCTION_RISK,
)


PLAN_DIR = Path("data/training/llm_plans__claude_haiku_4_5")


@dataclass
class SimTrade:
    session_date: date
    bar_idx: int
    entry_time: datetime
    signal_type: str
    direction: str
    entry: float
    stop: float
    target: float
    result: str   # "target" | "stop" | "eod"
    exit_bar: int
    exit_price: float
    pnl_pts: float
    confirmation_type: str


def simulate(df: pd.DataFrame, bar_idx: int, entry: float, stop: float,
             target: float, direction: str
             ) -> tuple[str, int, float]:
    n = len(df)
    for i in range(bar_idx + 1, n):
        h = float(df["high"].iat[i])
        lo = float(df["low"].iat[i])
        if direction == "long":
            if lo <= stop:
                return "stop", i, stop
            if h >= target:
                return "target", i, target
        else:
            if h >= stop:
                return "stop", i, stop
            if lo <= target:
                return "target", i, target
    return "eod", n - 1, float(df["close"].iat[-1])


def replay_session(df: pd.DataFrame, prior_df, *, strategy_params,
                   plan) -> list[SimTrade]:
    """Run engine on one session, simulate every emitted signal."""
    agg = SignalAggregator(
        strategy_params=strategy_params,
        elevator_params=PRODUCTION_ELEVATOR,
        exit_params=PRODUCTION_EXIT,
        min_rr_ratio=PRODUCTION_RISK.min_rr_ratio,
    )
    agg.reset()
    agg.initialize_levels(df, prior_df)
    if plan is not None:
        agg.set_mancini_llm_plan(plan)

    enriched = enrich_dataframe(df)
    velocity = enriched["velocity_5"]
    out: list[SimTrade] = []
    for i in range(len(df)):
        vel = float(velocity.iat[i])
        if vel != vel:
            vel = 0.0
        signal = agg.update(
            bar_idx=i, timestamp=df.index[i],
            open_=float(df["open"].iat[i]),
            high=float(df["high"].iat[i]),
            low=float(df["low"].iat[i]),
            close=float(df["close"].iat[i]),
            volume=float(df["volume"].iat[i]),
            velocity=vel, df=df,
        )
        if signal is None:
            continue
        conf = getattr(signal.pattern, "confirmation", None)
        conf_name = conf.name.lower() if hasattr(conf, "name") else ""
        is_long = signal.direction == "long"
        result, exit_bar, exit_px = simulate(
            df, i, signal.entry_price, signal.stop_price,
            signal.target_1, signal.direction,
        )
        pnl = (exit_px - signal.entry_price) if is_long else (signal.entry_price - exit_px)
        out.append(SimTrade(
            session_date=df.index[i].date(),
            bar_idx=i,
            entry_time=df.index[i].to_pydatetime(),
            signal_type=signal.signal_type.name,
            direction=signal.direction,
            entry=signal.entry_price,
            stop=signal.stop_price,
            target=signal.target_1,
            result=result,
            exit_bar=exit_bar,
            exit_price=exit_px,
            pnl_pts=pnl,
            confirmation_type=conf_name,
        ))
    return out


def summarize(label: str, trades: list[SimTrade]) -> dict:
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl_pts > 0)
    total = sum(t.pnl_pts for t in trades)
    print(f"\n{label}:")
    print(f"  Trades:    {n}")
    if n:
        print(f"  Win rate:  {wins}/{n} = {wins/n*100:.1f}%")
        print(f"  Total PnL: {total:+.1f} pts (${total*50:+,.0f} ES, 1ct)")
        print(f"  Avg PnL:   {total/n:+.2f} pts/trade")
    # Per signal_type
    by_sig = defaultdict(list)
    for t in trades:
        by_sig[t.signal_type].append(t)
    if len(by_sig) > 1:
        print("  Per signal type:")
        for st in sorted(by_sig.keys(), key=lambda k: -len(by_sig[k])):
            ts = by_sig[st]
            tp = sum(t.pnl_pts for t in ts)
            wr = sum(1 for t in ts if t.pnl_pts > 0) / len(ts) * 100
            print(f"    {st:>22}: n={len(ts):>4}  WR={wr:>5.1f}%  PnL={tp:+8.1f}")
    return {"n": n, "wins": wins, "total": total}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    plan_dates_iso = sorted(p.stem for p in PLAN_DIR.glob("*.json"))
    plan_dates = [date.fromisoformat(d) for d in plan_dates_iso]
    print(f"Plan dates available: {len(plan_dates)} "
          f"({plan_dates[0]} → {plan_dates[-1]})")

    print("Loading bar data…")
    sessions = build_daily_sessions(load_data())
    print(f"Bar sessions: {len(sessions)} "
          f"({sorted(sessions)[0]} → {sorted(sessions)[-1]})")

    # Intersection of plan dates and bar data
    target = [d for d in plan_dates if d in sessions and len(sessions[d]) >= 30]
    print(f"Intersection (plan + bars): {len(target)}")
    if args.limit:
        target = target[: args.limit]
        print(f"Limited to first {args.limit}")

    # Param sets — both have use_mancini_llm_plan ON, only flag differs.
    base_params = replace(
        PRODUCTION_STRATEGY,
        use_mancini_llm_plan=True,
        danger_zone_allow_non_acceptance=False,
    )
    c1prime_params = replace(
        PRODUCTION_STRATEGY,
        use_mancini_llm_plan=True,
        danger_zone_allow_non_acceptance=True,
    )

    base_trades: list[SimTrade] = []
    c1prime_trades: list[SimTrade] = []

    prior_d = None
    for i, d in enumerate(target):
        df_sess = sessions[d]
        prior_df = sessions[prior_d] if prior_d is not None else None
        try:
            plan = load_plan(d, input_dir=PLAN_DIR)
        except Exception:
            plan = None
        try:
            base_trades.extend(replay_session(
                df_sess, prior_df, strategy_params=base_params, plan=plan,
            ))
            c1prime_trades.extend(replay_session(
                df_sess, prior_df, strategy_params=c1prime_params, plan=plan,
            ))
        except Exception as e:
            print(f"  {d}: {e!r}")
        prior_d = d
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(target)}] base n={len(base_trades)} "
                  f"c1' n={len(c1prime_trades)}", flush=True)

    print("\n" + "=" * 70)
    print(f"A/B RESULT  ({len(target)} sessions with Haiku plans)")
    print("=" * 70)
    b = summarize("BASELINE  (danger_zone_allow_non_acceptance=False)",
                  base_trades)
    c = summarize("C1'  (danger_zone_allow_non_acceptance=True)",
                  c1prime_trades)
    print("\n" + "-" * 70)
    dn = c["n"] - b["n"]
    dp = c["total"] - b["total"]
    print(f"DELTA  Trades: {dn:+d}    PnL: {dp:+.1f} pts "
          f"(${dp*50:+,.0f} ES/1ct, ${dp*50*4:+,.0f} 4ct prod)")

    # Zoom into non-acceptance trades only (the carve-out)
    print("\n" + "=" * 70)
    print("NON-ACCEPTANCE FB LONG TRADES — what the carve-out admits")
    print("=" * 70)
    base_na = [t for t in base_trades
               if t.signal_type == "FAILED_BREAKDOWN"
               and t.direction == "long"
               and t.confirmation_type == "non_acceptance"]
    c1_na = [t for t in c1prime_trades
             if t.signal_type == "FAILED_BREAKDOWN"
             and t.direction == "long"
             and t.confirmation_type == "non_acceptance"]
    summarize("BASELINE — non-acceptance FB longs", base_na)
    summarize("C1' — non-acceptance FB longs", c1_na)
    delta_na = len(c1_na) - len(base_na)
    delta_pnl_na = sum(t.pnl_pts for t in c1_na) - sum(t.pnl_pts for t in base_na)
    print(f"\nDELTA non-acceptance FB longs: "
          f"+{delta_na} trades, {delta_pnl_na:+.1f} pts "
          f"(${delta_pnl_na*50:+,.0f} ES/1ct)")

    # Acceptance trades — should be identical (carve-out doesn't admit these)
    base_acc = [t for t in base_trades
                if t.signal_type == "FAILED_BREAKDOWN"
                and t.direction == "long"
                and t.confirmation_type == "acceptance"]
    c1_acc = [t for t in c1prime_trades
              if t.signal_type == "FAILED_BREAKDOWN"
              and t.direction == "long"
              and t.confirmation_type == "acceptance"]
    delta_acc = len(c1_acc) - len(base_acc)
    print(f"\nAcceptance FB longs delta: +{delta_acc} trades (should be 0)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
