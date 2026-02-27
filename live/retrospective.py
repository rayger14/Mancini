"""Next-day retrospective analysis: how did each level actually play out?

Runs the night after a session (9:05 PM ET / 2:05 AM UTC via cron).
Loads archived session bars, engine levels, Mancini levels, and trades,
then scores every level and identifies missed opportunities.

Usage:
    python3 live/retrospective.py                    # analyze yesterday
    python3 live/retrospective.py --date 2026-02-25  # analyze specific date

Cron:
    5 2 * * * docker exec mancini-bot python3 live/retrospective.py >> /app/logs/retrospective_cron.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

SESSIONS_DIR = os.environ.get("SESSIONS_DIR", "/app/data/sessions")
TRADE_LOG = os.environ.get("TRADE_LOG", "/app/logs/trades.jsonl")
SUBSTACK_FILE = os.environ.get("SUBSTACK_FILE", "/app/logs/substack_comparison.json")
OUTPUT_DIR = os.environ.get("RETRO_OUTPUT", "/app/logs")

# Scoring params
TOUCH_TOLERANCE = 2.0   # pts — how close price must get to count as a "touch"
BOUNCE_MIN_MOVE = 3.0   # pts — move away from level to count as a "bounce"
BOUNCE_WINDOW = 5       # bars after touch to check for bounce
BREAK_CLOSE_DIST = 2.0  # pts — close beyond level to count as a "break"
MISSED_BOUNCE_MIN = 5.0 # pts — minimum bounce to flag as missed opportunity


def load_session_bars(session_date: str) -> pd.DataFrame | None:
    """Load archived 1-min bars for a session."""
    path = Path(SESSIONS_DIR) / f"{session_date}_bars.parquet"
    if path.exists():
        return pd.read_parquet(path)
    print(f"No session bars found at {path}")
    return None


def load_engine_levels(session_date: str) -> list[dict]:
    """Load archived engine level snapshot."""
    path = Path(SESSIONS_DIR) / f"{session_date}_levels.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def load_substack_levels() -> dict:
    """Load the substack comparison JSON (written by nightly cron)."""
    path = Path(SUBSTACK_FILE)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def load_trades(session_date: str) -> list[dict]:
    """Load trades for a specific session from the JSONL trade log."""
    trades = []
    path = Path(TRADE_LOG)
    if not path.exists():
        return trades
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line.strip())
            if rec.get("session_date") == session_date:
                trades.append(rec)
        except (json.JSONDecodeError, KeyError):
            continue
    return trades


def build_level_list(engine_levels: list[dict], substack_data: dict) -> list[dict]:
    """Merge engine and Mancini levels into a unified list with source tags."""
    levels = []
    seen = set()

    # Engine levels
    for lv in engine_levels:
        price = round(lv["price"], 0)
        levels.append({"price": price, "source": "engine", "type": lv.get("type", "?")})
        seen.add(price)

    # Mancini levels from substack comparison
    comp = substack_data.get("comparison", {})

    # Matched levels (both sources)
    for m in comp.get("matched", []):
        price = round(m.get("mancini_price", m.get("engine_price", 0)), 0)
        # Upgrade existing engine level to "both"
        upgraded = False
        for lv in levels:
            if abs(lv["price"] - price) <= 3:
                lv["source"] = "both"
                upgraded = True
                break
        if not upgraded:
            levels.append({"price": price, "source": "both", "type": "matched"})
            seen.add(price)

    # Mancini-only levels
    for m in comp.get("mancini_only", []):
        price = round(m.get("price", 0), 0)
        if price > 0 and not any(abs(lv["price"] - price) <= 3 for lv in levels):
            levels.append({
                "price": price,
                "source": "mancini",
                "type": m.get("role", "?"),
            })

    levels.sort(key=lambda x: x["price"])
    return levels


def score_level(price: float, df: pd.DataFrame) -> dict:
    """Score a single level against session bars."""
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    touches = 0
    bounces = 0
    breaks = 0
    max_bounce = 0.0

    for i in range(n):
        # Touch: bar high/low comes within tolerance of level
        near_from_below = abs(highs[i] - price) <= TOUCH_TOLERANCE
        near_from_above = abs(lows[i] - price) <= TOUCH_TOLERANCE
        is_touch = near_from_below or near_from_above

        if is_touch:
            touches += 1

            # Check for bounce in next BOUNCE_WINDOW bars
            best_bounce = 0.0
            for j in range(i + 1, min(i + 1 + BOUNCE_WINDOW, n)):
                if near_from_below:
                    # Price approaching from below → bounce = drop away
                    move = price - lows[j]
                elif near_from_above:
                    # Price approaching from above → bounce = rally away
                    move = highs[j] - price
                else:
                    move = 0
                best_bounce = max(best_bounce, move)

            if best_bounce >= BOUNCE_MIN_MOVE:
                bounces += 1
            max_bounce = max(max_bounce, best_bounce)

        # Break: close beyond level by BREAK_CLOSE_DIST
        if closes[i] > price + BREAK_CLOSE_DIST and (i == 0 or closes[i - 1] <= price + BREAK_CLOSE_DIST):
            breaks += 1
        elif closes[i] < price - BREAK_CLOSE_DIST and (i == 0 or closes[i - 1] >= price - BREAK_CLOSE_DIST):
            breaks += 1

    # Verdict
    if touches == 0:
        verdict = "UNTESTED"
    elif bounces > 0 and breaks <= 1:
        verdict = "HELD"
    elif breaks >= 2:
        verdict = "BROKEN"
    elif bounces > 0:
        verdict = "HELD"
    else:
        verdict = "BROKEN"

    return {
        "touches": touches,
        "bounces": bounces,
        "breaks": breaks,
        "max_bounce": round(max_bounce, 1),
        "verdict": verdict,
    }


def match_trades_to_levels(trades: list[dict], scored_levels: list[dict]) -> list[dict]:
    """Match each trade to the level it was based on and assess quality."""
    results = []
    for t in trades:
        if t.get("event") != "exit":
            continue
        entry = t.get("entry_price", 0)
        pnl = t.get("pnl_pts", 0)
        pattern = t.get("pattern_type", t.get("signal", {}).get("type", "?"))
        level_price = t.get("signal", {}).get("level_price", 0)

        # Find matching scored level
        level_held = None
        mancini_mentioned = False
        for lv in scored_levels:
            if level_price > 0 and abs(lv["price"] - level_price) <= 3:
                level_held = lv["verdict"] == "HELD"
                mancini_mentioned = lv["source"] in ("mancini", "both")
                break

        results.append({
            "pattern": pattern,
            "entry": entry,
            "pnl": round(pnl, 1) if pnl else 0,
            "level": level_price,
            "level_held": level_held,
            "mancini_mentioned": mancini_mentioned,
        })
    return results


def find_missed_opportunities(
    scored_levels: list[dict],
    trades: list[dict],
) -> list[dict]:
    """Find levels that held with big bounces but we didn't trade."""
    # Collect levels we actually traded at
    traded_levels = set()
    for t in trades:
        if t.get("event") == "entry":
            lp = t.get("signal", {}).get("level_price", 0)
            if lp > 0:
                traded_levels.add(round(lp, 0))

    missed = []
    for lv in scored_levels:
        if lv["verdict"] != "HELD":
            continue
        if lv["max_bounce"] < MISSED_BOUNCE_MIN:
            continue
        # Check if we traded near this level
        was_traded = any(abs(lv["price"] - tp) <= 3 for tp in traded_levels)
        if not was_traded:
            missed.append({
                "level": lv["price"],
                "source": lv["source"],
                "verdict": lv["verdict"],
                "max_bounce": lv["max_bounce"],
                "reason": "No signal generated",
            })
    return missed


def run_retrospective(target_date: str) -> dict:
    """Run the full retrospective analysis for a session date."""
    print(f"Running retrospective for {target_date}...")

    # Load data
    df = load_session_bars(target_date)
    if df is None or len(df) == 0:
        return {"error": f"No session bars for {target_date}"}

    engine_levels = load_engine_levels(target_date)
    substack_data = load_substack_levels()
    trades = load_trades(target_date)

    print(f"  Bars: {len(df)}, Engine levels: {len(engine_levels)}, Trades: {len(trades)}")

    # Session summary
    session_info = {
        "bars": len(df),
        "high": round(float(df["high"].max()), 2),
        "low": round(float(df["low"].min()), 2),
        "range": round(float(df["high"].max() - df["low"].min()), 2),
        "open": round(float(df["open"].iat[0]), 2),
        "close": round(float(df["close"].iat[-1]), 2),
    }

    # Build unified level list
    all_levels = build_level_list(engine_levels, substack_data)
    print(f"  Total levels to score: {len(all_levels)}")

    # Score each level
    scored_levels = []
    for lv in all_levels:
        score = score_level(lv["price"], df)
        scored_levels.append({
            "price": lv["price"],
            "source": lv["source"],
            **score,
        })

    # Match trades to levels
    trades_analysis = match_trades_to_levels(trades, scored_levels)

    # Find missed opportunities
    missed = find_missed_opportunities(scored_levels, trades)

    # Compute summary stats
    tested = [lv for lv in scored_levels if lv["verdict"] != "UNTESTED"]
    held = [lv for lv in scored_levels if lv["verdict"] == "HELD"]
    broken = [lv for lv in scored_levels if lv["verdict"] == "BROKEN"]

    # Source-specific accuracy (held / tested)
    def accuracy(source_filter):
        src_tested = [lv for lv in tested if lv["source"] in source_filter]
        src_held = [lv for lv in held if lv["source"] in source_filter]
        return round(100 * len(src_held) / len(src_tested), 1) if src_tested else 0

    trades_at_held = sum(1 for t in trades_analysis if t.get("level_held") is True)
    trades_at_broken = sum(1 for t in trades_analysis if t.get("level_held") is False)

    summary = {
        "total_levels": len(scored_levels),
        "tested": len(tested),
        "held": len(held),
        "broken": len(broken),
        "mancini_accuracy_pct": accuracy(("mancini", "both")),
        "engine_accuracy_pct": accuracy(("engine", "both")),
        "matched_accuracy_pct": accuracy(("both",)),
        "trades_at_held_levels": trades_at_held,
        "trades_at_broken_levels": trades_at_broken,
        "missed_with_5pt_bounce": len(missed),
    }

    result = {
        "session_date": target_date,
        "generated_at": datetime.now().isoformat(),
        "session": session_info,
        "level_scores": scored_levels,
        "trades_analysis": trades_analysis,
        "missed_opportunities": missed,
        "summary": summary,
    }

    # Write output
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"retrospective_{target_date}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Written to {out_path}")

    # Print summary
    print(f"\n  SUMMARY for {target_date}:")
    print(f"  Session: {session_info['low']:.0f} - {session_info['high']:.0f} ({session_info['range']:.0f} pts)")
    print(f"  Levels: {summary['total_levels']} total, {summary['tested']} tested, {summary['held']} held, {summary['broken']} broken")
    print(f"  Accuracy: Mancini {summary['mancini_accuracy_pct']}%, Engine {summary['engine_accuracy_pct']}%, Matched {summary['matched_accuracy_pct']}%")
    print(f"  Trades: {trades_at_held} at held levels, {trades_at_broken} at broken levels")
    print(f"  Missed: {len(missed)} levels with {MISSED_BOUNCE_MIN}+ pt bounce we didn't trade")

    return result


def main():
    parser = argparse.ArgumentParser(description="Retrospective level analysis")
    parser.add_argument("--date", type=str, default=None,
                        help="Session date to analyze (YYYY-MM-DD). Defaults to yesterday.")
    args = parser.parse_args()

    if args.date:
        target = args.date
    else:
        target = str(date.today() - timedelta(days=1))

    run_retrospective(target)


if __name__ == "__main__":
    main()
