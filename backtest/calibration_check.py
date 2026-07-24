"""Gate (a): does the confidence table PREDICT out-of-sample?

Build the table from the corpus ONLY, score every live trade through it,
and compare predicted P(win) to what actually happened. Pass criteria
(pre-stated in the plan — no post-hoc tuning):
  - every cell with live n >= 10: |predicted WR - realized WR| <= 10pts
  - Brier score <= the base-rate constant predictor's Brier

Usage:
  python3 backtest/calibration_check.py \
      --table /tmp/table_corpus_only.json \
      --live-log ~/mancini_research/trades_live.jsonl
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.build_confidence_profile import iter_trades, net_pnl_for_trade  # noqa: E402
from core.confidence import ConfidenceTable, key_from_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True)
    ap.add_argument("--live-log", required=True)
    a = ap.parse_args()

    table = ConfidenceTable.load(a.table)
    rows = []
    for e, exit_rows in iter_trades([a.live_log]):
        pnl = net_pnl_for_trade(exit_rows)
        pred = table.lookup(key_from_record(e))
        if pred.p_win is None:
            continue
        rows.append((pred, 1 if pnl > 0 else 0))

    if not rows:
        raise SystemExit("no scored trades — table or log unusable")

    n = len(rows)
    base_rate = sum(w for _, w in rows) / n
    brier = sum((p.p_win - w) ** 2 for p, w in rows) / n
    brier_base = sum((base_rate - w) ** 2 for p, w in rows) / n

    by_cell = defaultdict(list)
    for p, w in rows:
        by_cell[p.cell_label].append((p.p_win, w))

    print(f"scored live trades: {n} | realized WR {100*base_rate:.0f}%")
    print(f"Brier(table) = {brier:.4f}  vs  Brier(base-rate) = {brier_base:.4f}"
          f"  -> {'PASS' if brier <= brier_base else 'FAIL'}")
    print(f"\n{'cell':38s}{'n':>4s}{'pred WR':>9s}{'real WR':>9s}{'|err|':>7s}")
    worst_fail = False
    for cell, items in sorted(by_cell.items()):
        cn = len(items)
        pw = sum(p for p, _ in items) / cn
        rw = sum(w for _, w in items) / cn
        err = abs(pw - rw)
        flag = ""
        if cn >= 10 and err > 0.10:
            flag = "  << FAIL(>10pts)"
            worst_fail = True
        print(f"{cell:38s}{cn:4d}{100*pw:8.0f}%{100*rw:8.0f}%{100*err:6.0f}%{flag}")

    ok = (brier <= brier_base) and not worst_fail
    print(f"\nGATE (a): {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
