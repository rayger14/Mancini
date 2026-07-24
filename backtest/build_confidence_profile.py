"""Build data/confidence_profile.json from replay corpora + the live log.

The artifact behind the engine-confidence badge (Phase A) and, after its
gates, confidence sizing (Phase B). See core/confidence.py for the key
derivation — this script MUST use the same functions.

Cumulative-P&L rule: a trade's net = its FINAL 'exit' row's pnl_pts (the
live/replay exit writer records the CUMULATIVE trade total there). Summing
partial_exit rows on top double-counts T1 winners (~25% inflation — the
bug that skewed earlier harvests).

Usage:
  python3 backtest/build_confidence_profile.py \
      --corpus-dirs ~/mancini_research/corpus_base \
      --live-log /path/trades.jsonl \
      [--exclude-dates 2026-05-06..2026-07-18] [--corpus-plan-dim off] \
      --out data/confidence_profile.json
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.confidence import key_from_record, _cell_id  # noqa: E402


def net_pnl_for_trade(rows) -> float:
    """Final 'exit' row is CUMULATIVE; partials alone when no final exists."""
    finals = [r for r in rows if r.get("event") == "exit"]
    if finals:
        return float(finals[-1].get("pnl_pts") or 0.0)
    return sum(float(r.get("pnl_pts") or 0.0)
               for r in rows if r.get("event") == "partial_exit")


def iter_trades(paths):
    """Yield (entry_record, all_rows) per closed long trade in the given
    jsonl files. Groups by trade_id within a file; replay logs are
    single-position so ordering suffices."""
    for path in paths:
        ent, rows_by = {}, defaultdict(list)
        try:
            lines = open(path).read().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = r.get("trade_id")
            if tid is None:
                continue
            if r.get("event") == "entry":
                ent[tid] = r
            elif r.get("event") in ("exit", "partial_exit"):
                rows_by[tid].append(r)
        for tid, e in ent.items():
            if str(e.get("direction", "long")).lower() != "long":
                continue
            if tid not in rows_by:
                continue
            yield e, rows_by[tid]


def in_range(sd: str, spec) -> bool:
    if not spec:
        return False
    a, b = spec.split("..")
    return a <= sd <= b


def build(corpus_dirs, live_log, exclude, corpus_plan_dim, out_path):
    cells = defaultdict(lambda: {"n": 0, "wins": 0, "pnl_sum": 0.0})
    parents = defaultdict(lambda: {"n": 0, "wins": 0, "pnl_sum": 0.0})
    glob_stats = {"n": 0, "wins": 0, "pnl_sum": 0.0}
    src_counts = {"corpus": 0, "live": 0}

    def add(key, pnl, collapse_plan):
        win = 1 if pnl > 0 else 0
        for bucket, cid in (
            (cells, _cell_id(key.confirmation, key.plan_match,
                             key.session_window)),
            (parents, _cell_id(key.confirmation, key.plan_match)),
            (parents, key.confirmation),
        ):
            if collapse_plan and "|" in cid:
                continue   # corpus rows skip plan-split cells when collapsed
            bucket[cid]["n"] += 1
            bucket[cid]["wins"] += win
            bucket[cid]["pnl_sum"] += pnl
        glob_stats["n"] += 1
        glob_stats["wins"] += win
        glob_stats["pnl_sum"] += pnl

    sources = []
    for d in corpus_dirs:
        files = sorted(glob.glob(str(Path(d).expanduser() / "replay_*.jsonl")))
        sources.append(f"corpus:{d}({len(files)} files)")
        for e, rows in iter_trades(files):
            sd = str(e.get("session_date", ""))
            if in_range(sd, exclude):
                continue
            add(key_from_record(e), net_pnl_for_trade(rows),
                collapse_plan=(corpus_plan_dim == "off"))
            src_counts["corpus"] += 1
    if live_log:
        sources.append(f"live:{live_log}")
        for e, rows in iter_trades([live_log]):
            sd = str(e.get("session_date", ""))
            if in_range(sd, exclude):
                continue
            add(key_from_record(e), net_pnl_for_trade(rows),
                collapse_plan=False)
            src_counts["live"] += 1

    def finalize(d):
        return {cid: {"n": v["n"], "wins": v["wins"],
                      "avg_pnl": round(v["pnl_sum"] / v["n"], 2)}
                for cid, v in d.items() if v["n"] > 0}

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        sha = "?"

    out = {
        "meta": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "excluded_dates": exclude or "",
            "corpus_plan_dim": corpus_plan_dim,
            "script_git_sha": sha,
            "n_corpus": src_counts["corpus"],
            "n_live": src_counts["live"],
        },
        "cells": finalize(cells),
        "parents": finalize(parents),
        "global": {"n": glob_stats["n"], "wins": glob_stats["wins"],
                   "avg_pnl": round(glob_stats["pnl_sum"]
                                    / max(glob_stats["n"], 1), 2)},
    }
    Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True))

    # ---- sanity gate: must reproduce the known ground-truth split --------
    na = out["parents"].get("non_acceptance")
    ac = out["parents"].get("acceptance")
    print(f"built {out_path}: {glob_stats['n']} trades "
          f"({src_counts['corpus']} corpus / {src_counts['live']} live)")
    if na:
        print(f"  NON_ACCEPTANCE: n={na['n']} WR={100*na['wins']/na['n']:.0f}% "
              f"avg={na['avg_pnl']:+.1f}")
    if ac:
        print(f"  ACCEPTANCE:     n={ac['n']} WR={100*ac['wins']/ac['n']:.0f}% "
              f"avg={ac['avg_pnl']:+.1f}")
    if na and ac:
        na_wr = na["wins"] / na["n"]
        ac_wr = ac["wins"] / ac["n"]
        if not (na_wr > ac_wr):
            raise SystemExit(
                "SANITY GATE FAILED: NON_ACCEPTANCE should out-WR ACCEPTANCE "
                "(ground truth 62% vs 48%). Harvest grouping is wrong — STOP.")
    print("sanity gate: OK (non-acceptance > acceptance)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dirs", nargs="*", default=[])
    ap.add_argument("--live-log", default=None)
    ap.add_argument("--exclude-dates", default=None, help="A..B inclusive")
    ap.add_argument("--corpus-plan-dim", choices=["on", "off"], default="on")
    ap.add_argument("--out", default="config/confidence_profile.json")
    a = ap.parse_args()
    build(a.corpus_dirs, a.live_log, a.exclude_dates, a.corpus_plan_dim, a.out)


if __name__ == "__main__":
    main()
