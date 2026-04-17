"""Validate Mancini Substack directional lean against engine trades.

For each Substack post, extracts:
  1. Directional lean (bullish / bearish / neutral)
  2. Key levels mentioned
  3. Trade setups described

Then runs the engine on the NEXT trading day's bars and checks:
  - Do trades align with Mancini's lean?
  - Are aligned trades more profitable than misaligned?
  - Would filtering by lean have improved results?
  - Do Mancini-flagged levels produce better FB signals?

Usage:
    python3 backtest/validate_substack.py
    python3 backtest/validate_substack.py --data data/ES_1m_full_session_2021-01-01_2026-02-05.parquet --full-session
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import BacktestRunner, BacktestResult
from config.settings import StrategyParams
from live.substack_compare import (
    extract_body_html,
    extract_highlights,
)

DATA_PATH = Path("data/ES_1m_2024-02-05_2026-02-05.parquet")
POSTS_PATH = Path("data/substack/all_posts.json")
EASTERN_TZ = "US/Eastern"


# ── Lean classification ──────────────────────────────────────────────

def extract_lean_statement(body_text: str) -> str | None:
    """Extract Mancini's explicit lean statement from the post.

    His lean is always in a sentence containing "my (general) lean is/was..."
    This is the ONLY reliable signal — "bull case" and "bear case" are
    scenario headers he uses in EVERY post (98-99% of posts have both).
    """
    # Find all "my lean is..." statements — take the LAST one (his conclusion)
    pattern = r'(?:my\s+(?:general\s+)?lean\s+(?:is|was)\s+(?:that\s+)?)(.*?)(?:\.|;|$)'
    matches = list(re.finditer(pattern, body_text, re.I))
    if matches:
        return matches[-1].group(1).strip()

    # Fallback: look for concluding lean
    pattern2 = r'(?:overall\s+(?:lean|bias)\s+(?:is|remains?)\s+)(.*?)(?:\.|;|$)'
    m = re.search(pattern2, body_text, re.I)
    if m:
        return m.group(1).strip()

    return None


def extract_trade_plan(body_text: str) -> dict:
    """Extract Mancini's specific trade setups from the post.

    Returns dict with:
      - fb_levels: prices where he expects failed breakdowns (long entries)
      - short_levels: prices where he'd short on breakdown
      - key_levels: all mentioned levels with context
      - has_fb_setup: bool - does he call out an explicit FB setup?
    """
    result = {
        "fb_levels": [],
        "short_levels": [],
        "key_levels": [],
        "has_fb_setup": False,
    }

    low = body_text.lower()

    # FB setups: "failed breakdown of XXXX" or "FB at XXXX"
    for m in re.finditer(r'failed\s+breakdown\s+(?:of\s+)?(\d{4,5})', low):
        result["fb_levels"].append(float(m.group(1)))
        result["has_fb_setup"] = True

    # "long at XXXX" or "entry at XXXX"
    for m in re.finditer(r'(?:long|entry|buy)\s+(?:at|near|around)\s+(\d{4,5})', low):
        result["fb_levels"].append(float(m.group(1)))

    # Short levels: "short below XXXX" or "breakdown of XXXX"
    for m in re.finditer(r'(?:short|sell)\s+(?:below|under|at)\s+(\d{4,5})', low):
        result["short_levels"].append(float(m.group(1)))

    return result


def classify_lean(highlights: list[dict], body_text: str) -> str:
    """Classify directional lean as 'bullish', 'bearish', or 'neutral'.

    Mancini's posts have a VERY consistent structure:
    1. Summary of what happened today (narrative)
    2. "My general lean is [X]" — his actual directional call
    3. "Bull case tomorrow: [scenario]" — NOT his lean, just a scenario
    4. "Bear case tomorrow: [scenario]" — NOT his lean, just a scenario
    5. Trade setups with specific levels

    The old classifier counted "bear case tomorrow" as bearish lean,
    which is why 329/505 posts were classified bearish. In reality,
    Mancini says "defer to the trend" (= bullish) in 64% of posts.

    New approach:
    - Extract his explicit lean statement (the text after "my lean is")
    - Classify THAT statement, not the full post
    - "defer to the trend" = bullish (ES has been in uptrend 2024-2026)
    - Specific setups: "can backtest X" = bullish, "can sell to X" = bearish
    - Ignore "bull case" / "bear case" section headers entirely
    """
    # Step 1: Extract his explicit lean statement
    lean_stmt = extract_lean_statement(body_text)

    if lean_stmt:
        low = lean_stmt.lower()

        # "defer to the trend" = bullish (ES uptrend throughout data period)
        if 'defer to the trend' in low:
            return "bullish"

        # Explicit bullish signals in lean statement
        bull_patterns = [
            r'(?:bounce|rally|squeeze|pop|higher|recover|backtest|back-test)',
            r'(?:can\s+try|can\s+see|can\s+work)\s+(?:a\s+)?(?:relief|bounce|rally|pop|up|higher)',
            r'bulls?\s+(?:have|still|more|energy|motivated|in control)',
            r'(?:hold|defend|support)',
        ]
        for pat in bull_patterns:
            if re.search(pat, low):
                return "bullish"

        # Explicit bearish signals in lean statement
        bear_patterns = [
            r'(?:sell|drop|lower|decline|flush|weakness|cautious)',
            r'(?:can\s+try|can\s+see|can\s+work)\s+(?:a\s+)?(?:sell|drop|lower|down)',
            r'bears?\s+(?:have|still|more|energy|motivated|in control)',
            r'lose|fail|break(?:down)?|vulnerable',
        ]
        for pat in bear_patterns:
            if re.search(pat, low):
                return "bearish"

        # Lean statement exists but unclear direction
        return "neutral"

    # Step 2: No explicit lean — look at his conclusion paragraph
    # Find the last section of the post (his trade plan)
    body_lower = body_text.lower()

    # Check if the conclusion is bullish or bearish
    # Look at the LAST 500 chars (his wrap-up)
    conclusion = body_lower[-500:]

    bull_conclusion = len(re.findall(
        r'\b(?:bounce|rally|squeeze|higher|long|support held|buy the dip|failed breakdown)\b',
        conclusion
    ))
    bear_conclusion = len(re.findall(
        r'\b(?:sell|drop|lower|decline|short|resistance held|breakdown)\b',
        conclusion
    ))

    if bull_conclusion > bear_conclusion + 1:
        return "bullish"
    elif bear_conclusion > bull_conclusion + 1:
        return "bearish"

    return "neutral"


def get_next_trading_day(post_date: date, available_dates: set[date]) -> date | None:
    """Get the next trading day after a post is published.

    Posts are published evening before the trading day they describe.
    E.g., "Feb 24 Plan" published Feb 23 evening → trades on Feb 24.
    """
    # Try next day first (most common: evening post → next day trading)
    for offset in range(1, 5):
        candidate = post_date + timedelta(days=offset)
        if candidate in available_dates:
            return candidate
    return None


def extract_actionable_levels(body_text: str, trading_day_df: pd.DataFrame) -> list[float]:
    """Extract ONLY actionable trade levels from post — not narrative prices.

    Mancini mentions 30+ prices per post in his narrative ("we rallied to 5533,
    then sold to 5518"). Those are NOT trade setups. His actionable levels are:

    1. "XXXX is key [support/resistance/level/zone]"
    2. "failed breakdown of XXXX" / "FB at XXXX"
    3. "XXXX must hold" / "must defend XXXX"
    4. "long at/above XXXX" / "short below XXXX"
    5. "XXXX [support/resistance]" in his level list section
    6. "watch XXXX" / "important level XXXX"
    7. "backtest XXXX" / "back-test XXXX"
    """
    if trading_day_df is None or len(trading_day_df) == 0:
        return []

    day_low = float(trading_day_df["low"].min())
    day_high = float(trading_day_df["high"].max())
    price_range = (day_low - 100, day_high + 100)

    levels = set()
    low = body_text.lower()

    # Actionable patterns — price must appear in a SETUP context
    actionable_patterns = [
        # Key levels he explicitly calls out
        r'(\d{4,5})\s+(?:is\s+)?(?:key|important|critical|major|big|huge)',
        r'(?:key|important|critical|major)\s+(?:level|support|resistance|zone).*?(\d{4,5})',
        # Must hold / must defend — his strongest conviction calls
        r'(\d{4,5})\s+(?:must|needs? to|has to)\s+(?:hold|defend)',
        r'(?:must|needs? to)\s+(?:hold|defend)\s+(\d{4,5})',
        # Failed breakdown setups
        r'(?:failed\s+breakdown|fb)\s+(?:of\s+|at\s+)?(\d{4,5})',
        r'(\d{4,5})\s+(?:failed\s+breakdown|fb)',
        # Explicit entry setups
        r'(?:long|entry|buy)\s+(?:at|above|near|around)\s+(\d{4,5})',
        r'(?:short|sell)\s+(?:at|below|under|near)\s+(\d{4,5})',
        # Backtest setups (his term for price returning to a level)
        r'(?:backtest|back-test)\s+(?:of\s+|at\s+)?(\d{4,5})',
        r'(\d{4,5})\s+(?:backtest|back-test)',
        # Watch / look for
        r'(?:watch|watching|look for|looking for|eyes on)\s+(\d{4,5})',
        # Support/resistance in level list context (not narrative)
        r'(\d{4,5})\s*(?::|=)\s*(?:support|resistance|key|major|bull|bear)',
        # "as long as XXXX holds" — conditional support
        r'as\s+long\s+as\s+(\d{4,5})',
        # Daily low/high callouts
        r'(?:daily|prior day|yesterday.s?)\s+(?:low|high)\s+(?:at|of|is)\s+(\d{4,5})',
    ]

    for pat in actionable_patterns:
        for m in re.finditer(pat, low):
            price = float(m.group(1))
            if price_range[0] <= price <= price_range[1]:
                levels.add(price)

    return sorted(levels)


# ── Data loading ─────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize(EASTERN_TZ)
    return df


def split_by_day(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    daily: dict[date, pd.DataFrame] = {}
    for dt, group in df.groupby(df.index.date):
        daily[dt] = group
    return daily


def load_posts(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ── Main analysis ────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--full-session", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("SUBSTACK DIRECTIONAL LEAN VALIDATION")
    print("=" * 80)

    # Load bar data
    print("\nLoading bar data...")
    df = load_data(Path(args.data))
    daily_dfs = split_by_day(df)
    available_dates = set(daily_dfs.keys())
    print(f"  {len(daily_dfs)} trading days, {len(df)} bars")
    print(f"  Range: {min(available_dates)} → {max(available_dates)}")

    # Load Substack posts
    print("\nLoading Substack posts...")
    posts = load_posts(POSTS_PATH)
    print(f"  {len(posts)} posts loaded")

    # Run full backtest to get all trades
    print("\nRunning backtest (production params, BD Short enabled)...")
    logger.remove()
    run_id = logger.add(sys.stderr, level="WARNING")

    strategy = StrategyParams(
        acceptance_max_dip_pts=15.0,
        acceptance_min_hold_bars=11,
        fb_stop_buffer_pts=6.0,
        lr_stop_buffer_pts=4.0,
        max_target_distance_pts=30.0,
        max_fb_sweep_depth_pts=999.0,
        bd_confirm_bars=21,
        bd_stop_buffer_pts=6.0,
        bd_max_break_depth_pts=17.0,
        bd_timeout_bars=35,
        allow_breakdown_short=True,
        allow_backtest_short=False,
        signal_cooldown_bars=15,
        use_regime_filter=False,
    )
    runner = BacktestRunner(strategy_params=strategy, min_rr_ratio=0.8)
    result = runner.run_multi_day(daily_dfs=daily_dfs, carry_runners=True)

    logger.remove(run_id)
    logger.add(sys.stderr, level="INFO")

    print(f"  {len(result.all_trades)} trades across {len(result.days)} days")
    print(f"  Total PnL: {result.total_pnl_pts:+.1f} pts, WR: {result.win_rate:.0%}")

    # Index trades by date
    trades_by_date: dict[date, list] = defaultdict(list)
    for t in result.all_trades:
        trade_date = t.entry_time.date() if hasattr(t.entry_time, 'date') else t.entry_date
        if trade_date:
            trades_by_date[trade_date].append(t)

    # ── Process each post ────────────────────────────────────────────
    print("\nProcessing posts...")

    post_results = []
    lean_counts = defaultdict(int)
    skipped = 0

    for post in posts:
        # Parse post date
        post_date_str = post.get("date", "")[:10]
        if not post_date_str:
            skipped += 1
            continue
        try:
            post_date = date.fromisoformat(post_date_str)
        except ValueError:
            skipped += 1
            continue

        # Get the trading day this post is about
        trading_day = get_next_trading_day(post_date, available_dates)
        if trading_day is None:
            skipped += 1
            continue

        # Extract body text
        raw_text = post.get("body_html_clean", "") or post.get("text", "")
        if not raw_text or len(raw_text) < 100:
            skipped += 1
            continue

        # If raw_text is a page dump, extract body
        if "<html" in raw_text.lower() or "body_html" in raw_text:
            body_text = extract_body_html(raw_text)
        else:
            body_text = raw_text

        if len(body_text) < 100:
            skipped += 1
            continue

        # Extract highlights and classify lean
        highlights = extract_highlights(body_text)
        lean = classify_lean(highlights, body_text)
        lean_counts[lean] += 1

        # Extract levels from post
        day_df = daily_dfs.get(trading_day)
        mancini_levels = extract_actionable_levels(body_text, day_df)

        # Get trades for that day
        day_trades = trades_by_date.get(trading_day, [])

        # Score each trade against the lean
        for trade in day_trades:
            direction = getattr(trade, 'direction', 'long')
            pattern = getattr(trade, 'pattern_type', '')
            pnl = trade.pnl_pts
            level_price = getattr(trade, 'level_price', 0.0)
            level_type = getattr(trade, 'level_type', '')

            # Does trade align with lean?
            if lean == "bullish":
                aligned = direction == "long"
            elif lean == "bearish":
                aligned = direction == "short"
            else:
                aligned = True  # Neutral = no lean filter

            # Was the trade at a Mancini-mentioned level?
            at_mancini_level = False
            if mancini_levels and level_price > 0:
                for mlvl in mancini_levels:
                    if abs(level_price - mlvl) <= 3.0:  # within 3 pts
                        at_mancini_level = True
                        break

            post_results.append({
                "post_date": post_date_str,
                "trading_day": str(trading_day),
                "lean": lean,
                "direction": direction,
                "pattern": pattern,
                "pnl_pts": pnl,
                "aligned": aligned,
                "at_mancini_level": at_mancini_level,
                "level_type": level_type,
                "level_price": level_price,
                "n_mancini_levels": len(mancini_levels),
                "n_highlights": len(highlights),
                "title": post.get("title", "?")[:60],
            })

    print(f"  Processed {len(posts) - skipped} posts, skipped {skipped}")
    print(f"  Lean distribution: {dict(lean_counts)}")
    print(f"  {len(post_results)} trade-post pairs")

    if not post_results:
        print("\nNo trade-post pairs found. Check data overlap.")
        return

    # ── Analysis ─────────────────────────────────────────────────────
    pdf = pd.DataFrame(post_results)

    print("\n" + "=" * 80)
    print("ANALYSIS 1: DIRECTIONAL LEAN — DOES IT PREDICT TRADE OUTCOMES?")
    print("=" * 80)

    for lean_val in ["bullish", "bearish", "neutral"]:
        subset = pdf[pdf["lean"] == lean_val]
        if len(subset) == 0:
            continue
        aligned = subset[subset["aligned"]]
        misaligned = subset[~subset["aligned"]]

        print(f"\n  Lean: {lean_val.upper()} ({len(subset)} trades on {subset['trading_day'].nunique()} days)")

        if len(aligned) > 0:
            a_wr = (aligned["pnl_pts"] > 0).mean()
            a_pnl = aligned["pnl_pts"].sum()
            a_avg = aligned["pnl_pts"].mean()
            print(f"    ALIGNED trades:    {len(aligned):>4} trades | WR: {a_wr:>5.0%} | Total: {a_pnl:>+8.1f} pts | Avg: {a_avg:>+6.1f} pts")

        if len(misaligned) > 0:
            m_wr = (misaligned["pnl_pts"] > 0).mean()
            m_pnl = misaligned["pnl_pts"].sum()
            m_avg = misaligned["pnl_pts"].mean()
            print(f"    MISALIGNED trades: {len(misaligned):>4} trades | WR: {m_wr:>5.0%} | Total: {m_pnl:>+8.1f} pts | Avg: {m_avg:>+6.1f} pts")

    # Overall aligned vs misaligned
    aligned_all = pdf[pdf["aligned"]]
    misaligned_all = pdf[~pdf["aligned"]]
    print(f"\n  OVERALL:")
    if len(aligned_all) > 0:
        print(f"    Aligned:    {len(aligned_all)} trades | WR: {(aligned_all['pnl_pts'] > 0).mean():.0%} | PnL: {aligned_all['pnl_pts'].sum():+.1f} pts | Avg: {aligned_all['pnl_pts'].mean():+.1f}")
    if len(misaligned_all) > 0:
        print(f"    Misaligned: {len(misaligned_all)} trades | WR: {(misaligned_all['pnl_pts'] > 0).mean():.0%} | PnL: {misaligned_all['pnl_pts'].sum():+.1f} pts | Avg: {misaligned_all['pnl_pts'].mean():+.1f}")

    print("\n" + "=" * 80)
    print("ANALYSIS 2: MANCINI-MENTIONED LEVELS — DO THEY PRODUCE BETTER SIGNALS?")
    print("=" * 80)

    at_level = pdf[pdf["at_mancini_level"]]
    not_at_level = pdf[~pdf["at_mancini_level"]]

    if len(at_level) > 0:
        l_wr = (at_level["pnl_pts"] > 0).mean()
        l_pnl = at_level["pnl_pts"].sum()
        l_avg = at_level["pnl_pts"].mean()
        print(f"\n  At Mancini level:     {len(at_level):>4} trades | WR: {l_wr:>5.0%} | Total: {l_pnl:>+8.1f} pts | Avg: {l_avg:>+6.1f} pts")

    if len(not_at_level) > 0:
        n_wr = (not_at_level["pnl_pts"] > 0).mean()
        n_pnl = not_at_level["pnl_pts"].sum()
        n_avg = not_at_level["pnl_pts"].mean()
        print(f"  Not at Mancini level: {len(not_at_level):>4} trades | WR: {n_wr:>5.0%} | Total: {n_pnl:>+8.1f} pts | Avg: {n_avg:>+6.1f} pts")

    # Mancini levels by pattern type
    if len(at_level) > 5:
        print(f"\n  Mancini-level trades by pattern:")
        for pat, group in at_level.groupby("pattern"):
            if len(group) >= 3:
                wr = (group["pnl_pts"] > 0).mean()
                pnl = group["pnl_pts"].sum()
                print(f"    {pat:<25} {len(group):>3} trades | WR: {wr:>5.0%} | PnL: {pnl:>+8.1f} pts")

    print("\n" + "=" * 80)
    print("ANALYSIS 3: LEAN + PATTERN TYPE INTERACTION")
    print("=" * 80)

    for pat in pdf["pattern"].unique():
        pat_df = pdf[pdf["pattern"] == pat]
        if len(pat_df) < 10:
            continue

        aligned_pat = pat_df[pat_df["aligned"]]
        misaligned_pat = pat_df[~pat_df["aligned"]]

        print(f"\n  {pat} ({len(pat_df)} trades)")
        if len(aligned_pat) > 0:
            wr = (aligned_pat["pnl_pts"] > 0).mean()
            pnl = aligned_pat["pnl_pts"].sum()
            print(f"    Aligned:    {len(aligned_pat):>4} | WR: {wr:>5.0%} | PnL: {pnl:>+8.1f} pts")
        if len(misaligned_pat) > 0:
            wr = (misaligned_pat["pnl_pts"] > 0).mean()
            pnl = misaligned_pat["pnl_pts"].sum()
            print(f"    Misaligned: {len(misaligned_pat):>4} | WR: {wr:>5.0%} | PnL: {pnl:>+8.1f} pts")

    print("\n" + "=" * 80)
    print("ANALYSIS 4: LEAN ACCURACY — DID THE LEAN PREDICT MARKET DIRECTION?")
    print("=" * 80)

    # For each post+trading day, check if the market moved in the direction of lean
    day_results = []
    seen_days = set()
    for _, row in pdf.iterrows():
        td = row["trading_day"]
        if td in seen_days:
            continue
        seen_days.add(td)
        trading_date = date.fromisoformat(td)
        day_df = daily_dfs.get(trading_date)
        if day_df is None or len(day_df) == 0:
            continue

        # RTH only for direction
        rth = day_df.between_time("09:30", "15:59")
        if len(rth) < 10:
            continue

        rth_open = float(rth.iloc[0]["open"])
        rth_close = float(rth.iloc[-1]["close"])
        day_move = rth_close - rth_open

        lean = row["lean"]
        if lean == "bullish":
            correct = day_move > 0
        elif lean == "bearish":
            correct = day_move < 0
        else:
            correct = None  # neutral = no prediction

        day_results.append({
            "date": td,
            "lean": lean,
            "rth_move": day_move,
            "correct": correct,
        })

    day_df_results = pd.DataFrame(day_results)

    for lean_val in ["bullish", "bearish", "neutral"]:
        subset = day_df_results[day_df_results["lean"] == lean_val]
        if len(subset) == 0:
            continue
        if lean_val == "neutral":
            avg_move = subset["rth_move"].mean()
            print(f"\n  {lean_val.upper()}: {len(subset)} days | Avg RTH move: {avg_move:+.1f} pts")
        else:
            correct = subset[subset["correct"] == True]
            accuracy = len(correct) / len(subset) if len(subset) > 0 else 0
            avg_correct_move = correct["rth_move"].abs().mean() if len(correct) > 0 else 0
            avg_wrong_move = subset[subset["correct"] == False]["rth_move"].abs().mean() if len(subset) > len(correct) else 0
            print(f"\n  {lean_val.upper()}: {len(subset)} days | Accuracy: {accuracy:.0%} ({len(correct)}/{len(subset)})")
            print(f"    Avg move when correct: {avg_correct_move:+.1f} pts")
            print(f"    Avg move when wrong:   {avg_wrong_move:+.1f} pts")

    print("\n" + "=" * 80)
    print("ANALYSIS 5: HYPOTHETICAL — WHAT IF WE ONLY TRADED WITH THE LEAN?")
    print("=" * 80)

    # Scenario 1: Only take trades aligned with lean (skip neutral days)
    non_neutral = pdf[pdf["lean"] != "neutral"]
    aligned_only = non_neutral[non_neutral["aligned"]]

    if len(non_neutral) > 0 and len(aligned_only) > 0:
        baseline_pnl = pdf["pnl_pts"].sum()
        aligned_pnl = aligned_only["pnl_pts"].sum()
        baseline_wr = (pdf["pnl_pts"] > 0).mean()
        aligned_wr = (aligned_only["pnl_pts"] > 0).mean()

        print(f"\n  Baseline (all trades):        {len(pdf):>4} trades | WR: {baseline_wr:.0%} | PnL: {baseline_pnl:>+8.1f} pts")
        print(f"  Lean-aligned only:            {len(aligned_only):>4} trades | WR: {aligned_wr:.0%} | PnL: {aligned_pnl:>+8.1f} pts")
        print(f"  Trades removed:               {len(pdf) - len(aligned_only):>4}")
        print(f"  PnL impact:                   {aligned_pnl - baseline_pnl:>+8.1f} pts")

    # Scenario 2: Lean-aligned + at Mancini level
    best_combo = pdf[(pdf["aligned"]) & (pdf["at_mancini_level"])]
    if len(best_combo) > 5:
        combo_wr = (best_combo["pnl_pts"] > 0).mean()
        combo_pnl = best_combo["pnl_pts"].sum()
        print(f"\n  Lean-aligned + Mancini level: {len(best_combo):>4} trades | WR: {combo_wr:.0%} | PnL: {combo_pnl:>+8.1f} pts")

    # ── Save results ─────────────────────────────────────────────────
    output_path = Path("data/substack_validation.json")
    output_path.write_text(json.dumps(post_results, indent=2, default=str))
    print(f"\n  Results saved to {output_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
  This analysis answers: "Does Mancini's directional lean add edge?"

  Key metrics to evaluate:
  1. Lean accuracy > 50% → He predicts direction better than coin flip
  2. Aligned WR > Misaligned WR → His lean filters losers
  3. Mancini-level WR > non-Mancini WR → His called levels are special
  4. Lean-only PnL > Baseline PnL → Filtering by lean improves results

  If all 4 are true → integrate lean as a signal filter
  If 1-2 are true → lean is a weak edge, maybe boost confidence only
  If none → lean adds no value, keep engine pure price action
""")


if __name__ == "__main__":
    main()
