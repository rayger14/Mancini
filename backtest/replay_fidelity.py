"""Replay self-fidelity: diff a ReplayRunner session against the REAL trade log.

Same code + same inputs should reproduce the live session. This harness
quantifies exactly how true that is, per day: matched entries, price deltas,
exit-class agreement, and an annotated miss list (live outages, restarts,
market-snapshot LQS deltas are expected divergence sources — see the plan).

  python3 replay_fidelity.py --date 2026-07-02 \
      --replay-log /tmp/replay/replay_2026-07-02.jsonl \
      --live-log /app/logs/trades.jsonl
  python3 replay_fidelity.py --dates 2026-05-06..2026-07-01 --replay-dir /tmp/replay
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LEVEL_TOL = 1.0
ENTRY_TOL = 2.0
EXIT_TOL = 2.0


def _load(path: Path, session_date: str, source: str) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        if t.get("event") != "entry":
            continue
        if source == "live" and t.get("session_date") != session_date:
            continue
        sig = t.get("signal") or {}
        out.append({
            "direction": (t.get("direction") or "long").lower(),
            "entry": float(t.get("entry_price") or 0),
            "level": float(sig.get("level_price") or 0),
            "type": str(sig.get("type") or t.get("pattern_type") or ""),
            "raw": t,
        })
    return out


def _exits(path: Path, session_date: str, source: str) -> dict:
    """trade_id -> exit record."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        if t.get("event") != "exit":
            continue
        if source == "live" and t.get("session_date") not in (None, session_date):
            continue
        out[t.get("trade_id")] = t
    return out


def _exit_class(reason: str) -> str:
    r = (reason or "").lower()
    for key, label in [("tp", "target"), ("target", "target"), ("stop", "stop"),
                       ("trail", "trail"), ("eod", "eod"), ("flatten", "eod")]:
        if key in r:
            return label
    return "other"


def diff_day(session_date: str, replay_log: Path, live_log: Path) -> dict:
    live = _load(live_log, session_date, "live")
    replay = _load(replay_log, session_date, "replay")
    live_exits = _exits(live_log, session_date, "live")
    replay_exits = _exits(replay_log, session_date, "replay")

    matched, misses = [], []
    rpool = list(replay)
    for lv in live:
        hit = None
        for rp in rpool:
            if (rp["direction"] == lv["direction"]
                    and abs(rp["level"] - lv["level"]) <= LEVEL_TOL
                    and abs(rp["entry"] - lv["entry"]) <= ENTRY_TOL):
                hit = rp
                break
        if hit is None:
            misses.append(lv)
            continue
        rpool.remove(hit)
        le = live_exits.get(lv["raw"].get("trade_id")) or {}
        re_ = replay_exits.get(hit["raw"].get("trade_id")) or {}
        matched.append({
            "level": lv["level"],
            "entry_delta": round(hit["entry"] - lv["entry"], 2),
            "exit_class_live": _exit_class(str(le.get("exit_reason"))),
            "exit_class_replay": _exit_class(str(re_.get("exit_reason"))),
            "exit_delta": (round(float(re_.get("exit_price") or 0)
                                 - float(le.get("exit_price") or 0), 2)
                           if le.get("exit_price") and re_.get("exit_price") else None),
            "pnl_live": le.get("pnl_pts"),
            "pnl_replay": re_.get("pnl_pts"),
        })

    n_live = len(live)
    fidelity = 100.0 * len(matched) / n_live if n_live else (100.0 if not replay else 0.0)
    return {"date": session_date, "live": n_live, "replay": len(replay),
            "matched": len(matched), "fidelity": round(fidelity, 1),
            "extra_in_replay": len(rpool), "matches": matched,
            "misses": [{"level": m["level"], "entry": m["entry"],
                        "type": m["type"],
                        "window": m["raw"].get("session_window"),
                        "prod": m["raw"].get("production_would_take")}
                       for m in misses],
            "extras": [{"level": r["level"], "entry": r["entry"], "type": r["type"]}
                       for r in rpool]}


def _print_day(r: dict) -> None:
    print(f"\n=== {r['date']}: fidelity {r['fidelity']}% "
          f"(live={r['live']} replay={r['replay']} matched={r['matched']} "
          f"extra={r['extra_in_replay']}) ===")
    for m in r["matches"]:
        agree = "==" if m["exit_class_live"] == m["exit_class_replay"] else "!="
        print(f"  MATCH lvl {m['level']}: entryΔ {m['entry_delta']:+.2f} | "
              f"exit {m['exit_class_live']}{agree}{m['exit_class_replay']} | "
              f"pnl live={m['pnl_live']} replay={m['pnl_replay']}")
    for m in r["misses"]:
        print(f"  MISS  lvl {m['level']} entry {m['entry']} [{m['type']}] "
              f"window={m['window']} prod={m['prod']}  <- annotate cause")
    for e in r["extras"]:
        print(f"  EXTRA lvl {e['level']} entry {e['entry']} [{e['type']}] "
              f"(replay-only; live outage/restart?)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None)
    ap.add_argument("--dates", default=None, help="A..B batch")
    ap.add_argument("--replay-log", default=None)
    ap.add_argument("--replay-dir", default="/tmp/replay")
    ap.add_argument("--live-log", default="/app/logs/trades.jsonl")
    args = ap.parse_args()

    if args.dates:
        from datetime import date as _d, timedelta
        a, b = args.dates.split("..")
        d0, d1 = _d.fromisoformat(a), _d.fromisoformat(b)
        dates = []
        while d0 <= d1:
            if d0.weekday() < 5:
                dates.append(d0.isoformat())
            d0 += timedelta(days=1)
    else:
        dates = [args.date]

    rows = []
    for d in dates:
        rlog = Path(args.replay_log) if args.replay_log \
            else Path(args.replay_dir) / f"replay_{d}.jsonl"
        r = diff_day(d, rlog, Path(args.live_log))
        if r["live"] == 0 and r["replay"] == 0:
            continue
        _print_day(r)
        rows.append(r)

    if len(rows) > 1:
        tl = sum(r["live"] for r in rows)
        tm = sum(r["matched"] for r in rows)
        print("\n" + "=" * 60)
        print(f"AGGREGATE: {tm}/{tl} live entries reproduced "
              f"({100.0*tm/tl if tl else 0:.0f}%) across {len(rows)} sessions; "
              f"extras={sum(r['extra_in_replay'] for r in rows)}")


if __name__ == "__main__":
    main()
