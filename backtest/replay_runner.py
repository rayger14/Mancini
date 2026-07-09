"""ReplayRunner — run the ACTUAL live IBRunner over a recorded tape.

The backtester that IS the live engine: same run() loop, same gates,
collection/bypass mode, single-position blocking and runner reconcile — with
SimBridge standing in for IB. See live/sim_bridge.py for the fill model.

Usage (in the bot container, where the archives live):
  docker cp backtest/replay_runner.py <c>:/app/replay_runner.py
  docker exec -w /app <c> python3 replay_runner.py --date 2026-07-02 \
      --data-dir /app/data --out-dir /tmp/replay

  --dates 2026-05-06..2026-07-01   batch mode; pattern state is chained
                                   across sessions like live carries it.

Every artifact (trade log, shadow log, status, pattern state) is confined to
--out-dir; webhooks/alerts are disabled; the live session archives are never
touched (_archive_session neutralized).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _set_replay_env(out_dir: Path, date: str, chain_pattern_state: bool) -> None:
    """Confine every runner artifact to out_dir + silence external effects.
    MUST run before live modules are imported (paths are read in __init__)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRADE_LOG"] = str(out_dir / f"replay_{date}.jsonl")
    os.environ["SHADOW_LOG"] = str(out_dir / f"shadow_{date}.jsonl")
    os.environ["STATUS_FILE"] = str(out_dir / "status.json")
    os.environ["LOG_FILE"] = str(out_dir / "replay.log")
    pattern = "pattern_state.json" if chain_pattern_state else f"pattern_{date}.json"
    os.environ["PATTERN_STATE_FILE"] = str(out_dir / pattern)
    os.environ["FORCE_TRADE_FILE"] = str(out_dir / "force_trade_absent.json")
    os.environ["MANCINI_TRADE_WEBHOOK"] = ""
    os.environ["WATCHDOG_WEBHOOK"] = ""
    os.environ["SHORT_ALERTS"] = "0"
    os.environ["BLOCKED_ALERTS"] = "0"
    os.environ["FREEZE_TIMEOUT_SEC"] = "0"


def build_replay(date: str, data_dir, out_dir, tape=None):
    """Construct the live runner wired to a SimBridge for `date`.

    Env must already be set (main() does it; tests set their own). Returns
    (runner, bridge) ready for runner.run().
    """
    import live.ib_runner as ibr
    # Yahoo snapshot: replay has no live market data; None is the exact value
    # the live code produces on fetch failure (an exercised production path).
    ibr.fetch_market_snapshot = lambda: None

    from live.ib_bridge import IBConfig
    from live.sim_bridge import SimBridge

    bridge = SimBridge(session_date=date, tape=tape,
                       data_dir=str(data_dir) if data_dir else None,
                       config=IBConfig())
    runner = ibr.build_live_runner(IBConfig(), full_session=True)
    runner.bridge = bridge

    # Clocks follow the tape: session date/rollover/plan-level timestamps from
    # the last popped bar; monotonic advances 60s per poll (incl. drain) so the
    # 45s entry-grace and the 3x-None close confirmation work bar-paced.
    runner._now_fn = bridge.current_time
    runner._mono_fn = lambda: 60.0 * bridge.poll_count
    runner._session_date = runner._compute_globex_trading_date(bridge.current_time())

    # Never touch the live session archives from a replay.
    runner._archive_session = lambda: None

    # Terminate when the tape is done (backstop; a 17:00 break bar in the tape
    # ends the session through the live _check_eod path first).
    bridge.on_tape_exhausted = lambda: setattr(runner, "_running", False)
    return runner, bridge


def _print_session(trade_log: Path, date: str) -> dict:
    entries = exits = 0
    pnl = 0.0
    lines = []
    if trade_log.exists():
        for line in trade_log.read_text().splitlines():
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if t.get("event") == "entry":
                entries += 1
                s = t.get("signal") or {}
                lines.append(f"  ENTRY {str(t.get('timestamp'))[:16]} "
                             f"{t.get('direction')} @ {t.get('entry_price')} "
                             f"[{s.get('type')} lvl={s.get('level_price')}]")
            elif t.get("event") == "exit":
                exits += 1
                pnl += float(t.get("pnl_pts") or 0.0)
                lines.append(f"  EXIT  {str(t.get('timestamp'))[:16]} "
                             f"@ {t.get('exit_price')} pnl={t.get('pnl_pts')} "
                             f"({str(t.get('exit_reason'))[:40]})")
    print(f"{date}: {entries} entries, {exits} exits, pnl={pnl:+.1f}")
    for l in lines:
        print(l)
    return {"date": date, "entries": entries, "exits": exits, "pnl": round(pnl, 1)}


def _date_range(spec: str) -> list:
    from datetime import date as _d, timedelta
    a, b = spec.split("..")
    d0, d1 = _d.fromisoformat(a), _d.fromisoformat(b)
    out = []
    while d0 <= d1:
        if d0.weekday() < 5:
            out.append(d0.isoformat())
        d0 += timedelta(days=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None)
    ap.add_argument("--dates", default=None, help="A..B inclusive batch")
    ap.add_argument("--data-dir", default="/app/data")
    ap.add_argument("--out-dir", default="/tmp/replay")
    args = ap.parse_args()

    dates = _date_range(args.dates) if args.dates else [args.date]
    if not dates or dates[0] is None:
        raise SystemExit("pass --date or --dates")
    out_dir = Path(args.out_dir)
    batch = len(dates) > 1

    results = []
    for d in dates:
        _set_replay_env(out_dir, d, chain_pattern_state=batch)
        try:
            runner, bridge = build_replay(d, args.data_dir, out_dir)
        except FileNotFoundError:
            print(f"{d}: no tape — skipped")
            continue
        runner.run()
        results.append(_print_session(Path(os.environ["TRADE_LOG"]), d))

    if batch:
        n = sum(r["entries"] for r in results)
        pnl = sum(r["pnl"] for r in results)
        print(f"\nBATCH: {len(results)} sessions, {n} entries, pnl={pnl:+.1f}")


if __name__ == "__main__":
    main()
