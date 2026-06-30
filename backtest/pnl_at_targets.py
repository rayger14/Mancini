"""P&L at planned exits — what each trade WOULD have made at its 75/15/10
scale-out, replayed over the real post-entry price path using the LIVE
ExitManager.

Why: recorded P&L in trades.jsonl is muddied by phantom exits, collection-mode
estimates, and single-price fills, so it doesn't reflect what the strategy's
real T1/T2/runner exits would have produced. This reconstructs the faithful
outcome and is the measurement tool for judging the short-gating change.

Usage (on the VM, where session parquets live):
    python3 -m backtest.pnl_at_targets                 # all sessions
    python3 -m backtest.pnl_at_targets 2026-06-30      # one session
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from strategy.exit_manager import ExitManager


# ---------------------------------------------------------------------------
# Core replay — the testable unit
# ---------------------------------------------------------------------------

def replay_trade(*, entry: float, stop: float, target_1: float, target_2: float,
                 contracts: int, direction: str,
                 bars: Iterable[tuple], prior_day_low: float | None = None,
                 exit_manager: ExitManager | None = None) -> dict:
    """Replay a trade against its post-entry price path through the live
    ExitManager and return the 75/15/10 scale-out outcome.

    ``bars`` is an iterable of (high, low, close) for each 1-min bar AFTER entry,
    in order. A runner still open when the bars run out is flattened at the last
    close (EOD), matching how the live bot would end the session.
    """
    em = exit_manager or ExitManager()
    pos = em.create_position(entry, stop, target_1, target_2, int(contracts), direction)
    if prior_day_low:
        pos.prior_day_low = float(prior_day_low)

    last_close = entry
    for high, low, close in bars:
        last_close = close
        if not pos.is_open:
            break
        em.update(pos, float(high), float(low), float(close))

    truncated = pos.is_open and pos.remaining_contracts > 0
    if truncated:
        # Flatten the runner at the last close (EOD), like the live bot.
        if direction == "short":
            pos.realized_pnl_pts += (pos.entry_price - last_close) * pos.remaining_contracts
        else:
            pos.realized_pnl_pts += (last_close - pos.entry_price) * pos.remaining_contracts
        pos.remaining_contracts = 0

    return {
        "pnl_pts": round(pos.realized_pnl_pts, 2),
        "t1_hit": bool(pos.t1_hit),
        "t2_hit": bool(pos.t2_hit),
        "runner_truncated_at_eod": bool(truncated),
    }


# ---------------------------------------------------------------------------
# Report orchestration
# ---------------------------------------------------------------------------

TRADE_LOG = os.environ.get("TRADE_LOG", "/app/data/live/trades.jsonl")
SESSIONS_DIR = os.environ.get("SESSIONS_DIR", "/app/data/sessions")


def _load_trade_pairs(path: str) -> list[dict]:
    """Read trades.jsonl, join entry↔exit by trade_id, keep entries that have
    the planned signal{} block."""
    entries: dict = {}
    exits: dict = {}
    p = Path(path)
    if not p.exists():
        return []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = r.get("trade_id")
        if r.get("event") == "entry" and r.get("signal"):
            entries[tid] = r
        elif r.get("event") == "exit":
            exits[tid] = r
    out = []
    for tid, e in entries.items():
        sig = e.get("signal") or {}
        if not all(k in sig for k in ("stop", "target_1", "target_2")):
            continue
        out.append({"entry": e, "exit": exits.get(tid)})
    return out


def _post_entry_bars(session_df, entry_time: str):
    """Slice (high, low, close) bars strictly after entry_time from a session
    parquet. Handles a DatetimeIndex or a time/timestamp column."""
    import pandas as pd
    df = session_df
    ts = pd.Timestamp(entry_time)
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("time", "timestamp"):
            if col in df.columns:
                df = df.set_index(pd.to_datetime(df[col]))
                break
    idx = df.index
    if getattr(idx, "tz", None) is not None and ts.tzinfo is None:
        ts = ts.tz_localize(idx.tz)
    elif getattr(idx, "tz", None) is None and ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    after = df[idx > ts]
    return list(zip(after["high"], after["low"], after["close"]))


def run_report(session_date: str | None = None) -> dict:
    import pandas as pd  # noqa: F401 (parquet engine)
    from live.retrospective import load_session_bars
    try:
        from live.ib_runner import PRODUCTION_EXIT, PRODUCTION_STRATEGY
        em = ExitManager(params=PRODUCTION_EXIT, strategy_params=PRODUCTION_STRATEGY)
    except Exception:
        em = ExitManager()

    pairs = _load_trade_pairs(TRADE_LOG)
    _bars_cache: dict = {}
    rows = []
    for pr in pairs:
        e = pr["entry"]
        sd = e.get("session_date")
        if session_date and sd != session_date:
            continue
        sig = e["signal"]
        if sd not in _bars_cache:
            _bars_cache[sd] = load_session_bars(sd)
        sdf = _bars_cache[sd]
        if sdf is None:
            continue
        bars = _post_entry_bars(sdf, e.get("timestamp", ""))
        rep = replay_trade(
            entry=float(e.get("entry_price", sig.get("entry", 0.0))),
            stop=float(sig["stop"]), target_1=float(sig["target_1"]),
            target_2=float(sig["target_2"]),
            contracts=int(e.get("contracts", 1) or 1),
            direction=(e.get("direction") or "long"),
            bars=bars, exit_manager=em,
        )
        recorded = (pr["exit"] or {}).get("pnl_pts")
        rows.append({
            "trade_id": e.get("trade_id"), "session": sd,
            "direction": e.get("direction"), "pattern": e.get("pattern_type"),
            "recorded": recorded, "at_targets": rep["pnl_pts"],
            "t1": rep["t1_hit"], "t2": rep["t2_hit"],
            "trunc": rep["runner_truncated_at_eod"],
            "prod": e.get("production_would_take"),
        })

    _print_report(rows)
    return {"rows": rows}


def _print_report(rows: list[dict]) -> None:
    if not rows:
        print("No replayable trades found (need entry signal{} + session parquet).")
        return
    print(f"\n{'tid':>7} {'session':<11} {'dir':<5} {'recorded':>9} {'at_targets':>11} "
          f"{'T1':>3} {'T2':>3} {'prod':>5}  pattern")
    for r in rows:
        rec = f"{r['recorded']:+.1f}" if r["recorded"] is not None else "   —"
        print(f"{str(r['trade_id']):>7} {str(r['session']):<11} {str(r['direction']):<5} "
              f"{rec:>9} {r['at_targets']:+11.1f} {('Y' if r['t1'] else '·'):>3} "
              f"{('Y' if r['t2'] else '·'):>3} {str(r['prod']):>5}  {r['pattern']}")

    def _agg(sel):
        sub = [r for r in rows if sel(r)]
        rec = sum(r["recorded"] for r in sub if r["recorded"] is not None)
        att = sum(r["at_targets"] for r in sub)
        return len(sub), rec, att
    print("\n=== totals (recorded vs at-planned-exits) ===")
    for label, sel in [
        ("ALL", lambda r: True),
        ("longs", lambda r: r["direction"] == "long"),
        ("shorts", lambda r: r["direction"] == "short"),
        ("production", lambda r: r["prod"] is True),
        ("collection", lambda r: r["prod"] is False),
    ]:
        n, rec, att = _agg(sel)
        if n:
            print(f"  {label:<11} n={n:3d}  recorded={rec:+9.1f}pt  at_targets={att:+9.1f}pt  "
                  f"delta={att - rec:+9.1f}pt")


if __name__ == "__main__":
    run_report(sys.argv[1] if len(sys.argv) > 1 else None)
