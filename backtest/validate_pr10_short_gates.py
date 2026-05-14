"""A/B validation of PR #10's short-side gates over 5 years of ES data.

Runs the standard 5y long+short backtest TWICE with identical configs
except for two PR #10 levers:
  - block_capitulation_shorts (capitulation-entry guard)
  - daily_bd_short_min_lqs (DAILY_FB_BULL contra-trend hard block)

Both runs enable the BD_SHORT + VELOCITY_SHORT detectors PR #10
actually targets (the default `production` config in
five_year_long_short.py only enables FR/LJ legacy shorts).

Diff reports short-side PnL, win rate, and trade count delta —
the actual signal of PR #10's impact.

Usage:
    python3 -u backtest/validate_pr10_short_gates.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import backtest.five_year_long_short as bt  # noqa: E402


def build_strategy(*, gates_on: bool):
    """Build a StrategyParams that's identical to the production config
    in five_year_long_short.py except (a) BD_SHORT and VELOCITY_SHORT
    detectors are enabled, (b) PR #10's two gates are toggled per arg.
    """
    base = bt.STRATEGY
    overrides = dict(
        # Enable the detectors PR #10 actually fixes
        allow_breakdown_short=True,
        allow_velocity_short=True,
        # PR #10's gates — A/B variable
        block_capitulation_shorts=gates_on,
        # daily_bd_short_min_lqs=0 effectively disables the DAILY_FB_BULL
        # block (any LQS >= 0 passes); 70 (default) is the production gate.
        daily_bd_short_min_lqs=70 if gates_on else 0,
    )
    return replace(base, **overrides)


_SHORT_PATTERNS = {
    "failed_rally", "level_rejection",
    "breakdown_short", "backtest_short", "velocity_short",
}


def short_side_summary(trades) -> dict:
    """Compute short-only stats from a list of completed trade dicts.

    The upstream script labels `direction` as LONG/SHORT but the
    classifier only sees failed_rally/level_rejection. We re-classify
    by the explicit pattern_type string so BD_SHORT and VELOCITY_SHORT
    actually count as shorts here.
    """
    shorts = [t for t in trades if t.get("pattern") in _SHORT_PATTERNS]
    n = len(shorts)
    if n == 0:
        return {"n": 0, "wins": 0, "wr": 0.0, "pnl": 0.0,
                "by_pattern": {}, "trades": []}
    wins = sum(1 for t in shorts if t.get("pnl_pts", 0) > 0)
    pnl = sum(t.get("pnl_pts", 0) for t in shorts)

    by_pattern: dict[str, dict] = {}
    for t in shorts:
        pat = t.get("pattern", "?")
        d = by_pattern.setdefault(
            pat, {"n": 0, "wins": 0, "pnl": 0.0}
        )
        d["n"] += 1
        d["wins"] += 1 if t.get("pnl_pts", 0) > 0 else 0
        d["pnl"] += t.get("pnl_pts", 0)

    return {
        "n": n,
        "wins": wins,
        "wr": wins / n * 100,
        "pnl": pnl,
        "by_pattern": by_pattern,
        "trades": shorts,
    }


def main():
    print("=" * 80)
    print("PR #10 short-side gate A/B validation (5y, 2021-01 → 2026-02)")
    print("=" * 80)

    df = bt.load_data()
    sessions = bt.build_sessions(df)
    print(f"\nLoaded {len(sessions)} sessions\n")

    results = {}
    for label, gates_on in [("GATES_ON", True), ("GATES_OFF", False)]:
        print(f"\n──── Running {label} ────")
        bt.STRATEGY = build_strategy(gates_on=gates_on)
        all_trades, _, _ = bt.run_backtest(sessions, mancini_levels_dir=None)
        results[label] = short_side_summary(all_trades)
        # Total all-trade pnl for context
        total_pnl = sum(t.get("pnl_pts", 0) for t in all_trades)
        results[label]["total_pnl"] = total_pnl
        results[label]["total_n"] = len(all_trades)

    # ── Report ────────────────────────────────────────────────────────
    on, off = results["GATES_ON"], results["GATES_OFF"]
    print("\n" + "=" * 80)
    print("SHORT-SIDE A/B RESULT")
    print("=" * 80)
    print(f"  {'':12s}  {'trades':>8s}  {'WR%':>6s}  {'PnL pts':>10s}")
    print(f"  {'GATES_OFF':12s}  {off['n']:>8d}  {off['wr']:>5.1f}  "
          f"{off['pnl']:>+10.1f}")
    print(f"  {'GATES_ON':12s}  {on['n']:>8d}  {on['wr']:>5.1f}  "
          f"{on['pnl']:>+10.1f}")
    print(f"  {'DELTA':12s}  {on['n'] - off['n']:>+8d}  "
          f"{on['wr'] - off['wr']:>+5.1f}  "
          f"{on['pnl'] - off['pnl']:>+10.1f}")

    print("\n  Per-pattern breakdown (GATES_ON):")
    for pat, d in sorted(on["by_pattern"].items()):
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print(f"    {pat:24s}  {d['n']:>4d} trades  "
              f"{wr:>5.1f}% WR  {d['pnl']:>+8.1f} pts")

    print("\n  Per-pattern breakdown (GATES_OFF):")
    for pat, d in sorted(off["by_pattern"].items()):
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print(f"    {pat:24s}  {d['n']:>4d} trades  "
              f"{wr:>5.1f}% WR  {d['pnl']:>+8.1f} pts")

    print("\n  Whole-strategy total (longs + shorts) for context:")
    print(f"    GATES_OFF: {off['total_n']:>5d} trades  {off['total_pnl']:>+8.1f} pts")
    print(f"    GATES_ON : {on['total_n']:>5d} trades  {on['total_pnl']:>+8.1f} pts")
    print(f"    DELTA    : {on['total_n']-off['total_n']:>+5d} trades  "
          f"{on['total_pnl']-off['total_pnl']:>+8.1f} pts")


if __name__ == "__main__":
    main()
