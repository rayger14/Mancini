#!/usr/bin/env python3
"""
Blind validation of generated levels vs Mancini's actual newsletter levels.

For each trading day where we have BOTH:
  1. A Mancini newsletter with levels
  2. Price data available BEFORE that day's open

We generate levels using ONLY prior data, then compare against his published levels.
This ensures zero look-ahead bias.

Also measures "level quality" — did price actually react at the generated levels?
"""

import json
import re
import sys
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strategy.level_generator import generate_levels, PriceLevel


# ── Parse Mancini Levels from Newsletter ──────────────────────


def parse_mancini_levels(text: str) -> Dict[str, List[dict]]:
    """Extract support and resistance levels from a Mancini post.

    Returns dict with 'supports' and 'resistances', each a list of
    {'price': float, 'is_major': bool}.
    """
    result = {"supports": [], "resistances": []}

    sup_match = re.search(r'[Ss]upports?\s+(?:are|is)[:\s]*(.+?)(?:\n\n|In terms|As readers|As usual|$)',
                          text, re.DOTALL)
    if sup_match:
        result["supports"] = _parse_level_list(sup_match.group(1))

    res_match = re.search(r'[Rr]esistances?\s+(?:are|is)[:\s]*(.+?)(?:\n\n|In terms|As readers|As usual|$)',
                          text, re.DOTALL)
    if res_match:
        result["resistances"] = _parse_level_list(res_match.group(1))

    return result


def _parse_level_list(text: str) -> List[dict]:
    """Parse a comma-separated list of levels like '6838, 6833 (major), 6828'."""
    levels = []
    pattern = r'(\d{4,5})(?:\s*[-–]\s*(\d{4,5}))?\s*(\(major\))?'
    for m in re.finditer(pattern, text):
        price1 = float(m.group(1))
        price2 = float(m.group(2)) if m.group(2) else None
        is_major = bool(m.group(3))

        if price2:
            price = (price1 + price2) / 2
        else:
            price = price1

        if 3000 < price < 9000:
            levels.append({"price": price, "is_major": is_major})

    return levels


def get_newsletter_date(post: dict) -> pd.Timestamp:
    """Determine which trading day a newsletter's levels are FOR.

    Mancini publishes the evening before → levels are for the NEXT trading day.
    """
    pub_date = pd.Timestamp(post["date"][:10])
    next_day = pub_date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)
    return next_day


# ── Level Matching ────────────────────────────────────────────


def match_levels(
    generated: List[PriceLevel],
    actual: List[dict],
    tolerance: float = 3.0,
) -> Dict:
    """Match generated levels against actual Mancini levels."""
    if not generated or not actual:
        return {"matches": 0, "total_gen": len(generated), "total_actual": len(actual),
                "match_rate_gen": 0, "match_rate_actual": 0, "avg_error": float("inf"),
                "matched_pairs": []}

    gen_prices = [l.price for l in generated]
    act_prices = [a["price"] for a in actual]

    matched_pairs = []
    used_actual = set()

    for g_price in gen_prices:
        best_dist = float("inf")
        best_idx = -1
        for i, a_price in enumerate(act_prices):
            if i in used_actual:
                continue
            dist = abs(g_price - a_price)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0 and best_dist <= tolerance:
            matched_pairs.append({
                "generated": g_price,
                "actual": act_prices[best_idx],
                "diff": best_dist,
                "actual_major": actual[best_idx]["is_major"],
            })
            used_actual.add(best_idx)

    n_matches = len(matched_pairs)
    avg_error = np.mean([p["diff"] for p in matched_pairs]) if matched_pairs else float("inf")

    return {
        "matches": n_matches,
        "total_gen": len(generated),
        "total_actual": len(actual),
        "match_rate_gen": n_matches / len(generated) if generated else 0,
        "match_rate_actual": n_matches / len(actual) if actual else 0,
        "avg_error": avg_error,
        "matched_pairs": matched_pairs,
    }


# ── Level Quality: Did Price React? ──────────────────────────


def measure_level_reaction(
    levels: List[PriceLevel],
    day_bars: pd.DataFrame,
    reaction_threshold: float = 3.0,
    bounce_threshold: float = 5.0,
) -> Dict:
    """Measure how price reacted at generated levels during the trading day.

    For support: did price touch and bounce?
    For resistance: did price touch and reject?
    """
    if levels is None or len(levels) == 0 or len(day_bars) == 0:
        return {"tested": 0, "reacted": 0, "reaction_rate": 0}

    tested = 0
    reacted = 0

    lows = day_bars["low"].values
    highs = day_bars["high"].values
    closes = day_bars["close"].values

    for lv in levels:
        price = lv.price
        touched = False

        for i in range(len(lows)):
            if lows[i] <= price + reaction_threshold and highs[i] >= price - reaction_threshold:
                touched = True
                end_idx = min(i + 30, len(closes))
                if lv.price <= day_bars["close"].iloc[0]:  # support
                    future_high = highs[i:end_idx].max() if end_idx > i else 0
                    if future_high - price >= bounce_threshold:
                        reacted += 1
                        break
                else:  # resistance
                    future_low = lows[i:end_idx].min() if end_idx > i else float("inf")
                    if price - future_low >= bounce_threshold:
                        reacted += 1
                        break

        if touched:
            tested += 1

    return {
        "tested": tested,
        "reacted": reacted,
        "reaction_rate": reacted / tested if tested > 0 else 0,
    }


# ── Main Validation Loop ─────────────────────────────────────


def run_validation(
    bars_1min: pd.DataFrame,
    posts: List[dict],
    tolerance: float = 3.0,
) -> pd.DataFrame:
    """Run blind validation across all available dates."""
    results = []

    bars_1min = bars_1min.sort_index()
    available_dates = sorted(bars_1min.index.normalize().unique())

    print(f"Price data: {available_dates[0].date()} to {available_dates[-1].date()}")
    print(f"Posts to validate: {len(posts)}")
    print()

    skipped_no_data = 0
    skipped_no_levels = 0

    for post_idx, post in enumerate(posts):
        mancini = parse_mancini_levels(post["text"])
        if not mancini["supports"] and not mancini["resistances"]:
            skipped_no_levels += 1
            continue

        target_date = get_newsletter_date(post)
        cutoff = pd.Timestamp(target_date.date())
        prior_data = bars_1min[bars_1min.index < cutoff]
        day_data = bars_1min[
            (bars_1min.index >= cutoff) &
            (bars_1min.index < cutoff + pd.Timedelta(days=1))
        ]

        prior_dates = prior_data.index.normalize().unique()
        if len(prior_dates) < 10:
            skipped_no_data += 1
            continue

        if len(day_data) == 0:
            skipped_no_data += 1
            continue

        prior_close = prior_data["close"].iloc[-1]

        try:
            our_levels = generate_levels(prior_data, current_price=prior_close)
        except Exception as e:
            print(f"  ERROR generating levels for {target_date.date()}: {e}")
            continue

        sup_stats = match_levels(our_levels.supports, mancini["supports"], tolerance)
        res_stats = match_levels(our_levels.resistances, mancini["resistances"], tolerance)

        all_our_levels = our_levels.supports + our_levels.resistances
        quality = measure_level_reaction(all_our_levels, day_data)

        mancini_as_pricelevels = [
            PriceLevel(price=l["price"], is_major=l["is_major"])
            for l in mancini["supports"] + mancini["resistances"]
        ]
        mancini_quality = measure_level_reaction(mancini_as_pricelevels, day_data)

        major_correct = 0
        major_total = 0
        for pair in sup_stats["matched_pairs"] + res_stats["matched_pairs"]:
            gen_price = pair["generated"]
            gen_level = None
            for lv in all_our_levels:
                if abs(lv.price - gen_price) < 0.5:
                    gen_level = lv
                    break
            if gen_level:
                major_total += 1
                if gen_level.is_major == pair["actual_major"]:
                    major_correct += 1

        row = {
            "date": target_date.date(),
            "post_title": post["title"][:60],
            "prior_close": prior_close,
            "sup_gen": sup_stats["total_gen"],
            "sup_actual": sup_stats["total_actual"],
            "sup_matches": sup_stats["matches"],
            "sup_match_rate": sup_stats["match_rate_actual"],
            "sup_avg_err": sup_stats["avg_error"],
            "res_gen": res_stats["total_gen"],
            "res_actual": res_stats["total_actual"],
            "res_matches": res_stats["matches"],
            "res_match_rate": res_stats["match_rate_actual"],
            "res_avg_err": res_stats["avg_error"],
            "total_matches": sup_stats["matches"] + res_stats["matches"],
            "total_actual": sup_stats["total_actual"] + res_stats["total_actual"],
            "overall_match_rate": (
                (sup_stats["matches"] + res_stats["matches"]) /
                max(1, sup_stats["total_actual"] + res_stats["total_actual"])
            ),
            "overall_avg_err": np.mean(
                [p["diff"] for p in sup_stats["matched_pairs"] + res_stats["matched_pairs"]]
            ) if (sup_stats["matched_pairs"] or res_stats["matched_pairs"]) else float("inf"),
            "levels_tested": quality["tested"],
            "levels_reacted": quality["reacted"],
            "our_reaction_rate": quality["reaction_rate"],
            "mancini_tested": mancini_quality["tested"],
            "mancini_reacted": mancini_quality["reacted"],
            "mancini_reaction_rate": mancini_quality["reaction_rate"],
            "major_correct": major_correct,
            "major_total": major_total,
            "major_accuracy": major_correct / major_total if major_total > 0 else 0,
        }

        results.append(row)

        if (post_idx + 1) % 25 == 0:
            print(f"  Processed {post_idx + 1}/{len(posts)} posts...")

    print(f"\nSkipped: {skipped_no_data} no data, {skipped_no_levels} no levels")

    return pd.DataFrame(results)


def print_report(df: pd.DataFrame):
    """Print comprehensive validation report."""
    print("\n" + "=" * 80)
    print("MANCINI LEVEL GENERATOR — BLIND VALIDATION REPORT")
    print("=" * 80)

    print(f"\nDates validated: {df['date'].min()} to {df['date'].max()}")
    print(f"Total days: {len(df)}")

    print("\n── LEVEL MATCHING (within 3 pts) ──")
    print(f"  Overall match rate:     {df['overall_match_rate'].mean():.1%}  (std: {df['overall_match_rate'].std():.1%})")
    print(f"  Support match rate:     {df['sup_match_rate'].mean():.1%}  (std: {df['sup_match_rate'].std():.1%})")
    print(f"  Resistance match rate:  {df['res_match_rate'].mean():.1%}  (std: {df['res_match_rate'].std():.1%})")
    print(f"  Avg error (matched):    {df['overall_avg_err'].replace([np.inf], np.nan).mean():.2f} pts")

    print("\n── LEVEL COUNTS (avg per day) ──")
    print(f"  Generated supports:     {df['sup_gen'].mean():.1f}  (Mancini: {df['sup_actual'].mean():.1f})")
    print(f"  Generated resistances:  {df['res_gen'].mean():.1f}  (Mancini: {df['res_actual'].mean():.1f})")

    print("\n── MAJOR/MINOR CLASSIFICATION ──")
    valid_major = df[df["major_total"] > 0]
    if len(valid_major) > 0:
        print(f"  Major accuracy:         {valid_major['major_accuracy'].mean():.1%}")

    print("\n── LEVEL QUALITY (price reaction at levels) ──")
    valid_q = df[df["levels_tested"] > 0]
    valid_mq = df[df["mancini_tested"] > 0]
    if len(valid_q) > 0:
        print(f"  Our levels tested:      {valid_q['levels_tested'].mean():.1f}/day")
        print(f"  Our reaction rate:      {valid_q['our_reaction_rate'].mean():.1%}")
    if len(valid_mq) > 0:
        print(f"  Mancini levels tested:  {valid_mq['mancini_tested'].mean():.1f}/day")
        print(f"  Mancini reaction rate:  {valid_mq['mancini_reaction_rate'].mean():.1%}")

    print("\n── MONTHLY BREAKDOWN ──")
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month").agg({
        "overall_match_rate": "mean",
        "our_reaction_rate": "mean",
        "mancini_reaction_rate": "mean",
        "date": "count",
    }).rename(columns={"date": "days"})

    print(f"  {'Month':<10} {'Days':>5} {'Match%':>8} {'Our React%':>11} {'Mancini React%':>15}")
    print(f"  {'-'*10} {'-'*5} {'-'*8} {'-'*11} {'-'*15}")
    for month, row in monthly.iterrows():
        print(f"  {str(month):<10} {row['days']:>5.0f} {row['overall_match_rate']:>7.1%} "
              f"{row['our_reaction_rate']:>10.1%} {row['mancini_reaction_rate']:>14.1%}")

    print("\n── BEST 5 DAYS ──")
    best = df.nlargest(5, "overall_match_rate")
    for _, row in best.iterrows():
        print(f"  {row['date']}  match={row['overall_match_rate']:.0%}  "
              f"err={row['overall_avg_err']:.1f}pts  ({row['total_matches']}/{row['total_actual']})")

    print("\n── WORST 5 DAYS ──")
    worst = df.nsmallest(5, "overall_match_rate")
    for _, row in worst.iterrows():
        print(f"  {row['date']}  match={row['overall_match_rate']:.0%}  "
              f"err={row['overall_avg_err']:.1f}pts  ({row['total_matches']}/{row['total_actual']})")

    print("\n── STATISTICAL SIGNIFICANCE ──")
    from scipy import stats
    both = df[(df["levels_tested"] > 0) & (df["mancini_tested"] > 0)]
    if len(both) > 10:
        t_stat, p_val = stats.ttest_rel(both["our_reaction_rate"], both["mancini_reaction_rate"])
        diff = both["our_reaction_rate"].mean() - both["mancini_reaction_rate"].mean()
        print(f"  Paired t-test (our vs Mancini reaction rate):")
        print(f"    Difference: {diff:+.1%}  t={t_stat:.2f}  p={p_val:.4f}")
        if p_val < 0.05:
            winner = "OURS" if diff > 0 else "MANCINI"
            print(f"    → {winner} significantly better at p<0.05")
        else:
            print(f"    → No significant difference")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-session", action="store_true",
                        help="Use full-session (globex+RTH) data if available")
    args = parser.parse_args()

    print("Loading data...")

    # Try full-session data first if requested
    full_session_path = Path("data").glob("ES_1m_full_session_*.parquet")
    full_session_files = sorted(full_session_path)

    if args.full_session and full_session_files:
        data_path = full_session_files[-1]
        print(f"  Using FULL SESSION data: {data_path}")
    else:
        data_path = Path("data/ES_1m_2024-02-05_2026-02-05.parquet")
        if args.full_session:
            print("  WARNING: Full session data not found, using RTH only")
            print("  Run: python3 backtest/fetch_globex_data.py  to download it")
        print(f"  Using RTH-only data: {data_path}")

    bars = pd.read_parquet(data_path)
    print(f"  Price bars: {len(bars)} ({bars.index[0]} to {bars.index[-1]})")

    # Show session coverage
    rth_bars = len(bars.between_time("09:30", "15:59"))
    overnight_bars = len(bars) - rth_bars
    print(f"  RTH bars: {rth_bars}, Overnight bars: {overnight_bars}")

    with open("data/substack/all_posts.json") as f:
        posts = json.load(f)
    print(f"  Posts: {len(posts)}")

    posts.sort(key=lambda p: p["date"])

    print("\nRunning blind validation (this takes a few minutes)...\n")
    results_df = run_validation(bars, posts)

    suffix = "_full_session" if overnight_bars > 0 else "_rth"
    out_path = f"data/level_validation_results{suffix}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nRaw results saved to {out_path}")

    print_report(results_df)
