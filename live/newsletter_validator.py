#!/usr/bin/env python3
"""Daily newsletter validation engine.

Compares our engine's detected levels against Mancini's published newsletter levels.
Tracks accuracy over time and identifies systematic misses for engine improvement.

Usage:
    # Validate latest post
    python3 live/newsletter_validator.py

    # Validate specific date
    python3 live/newsletter_validator.py --date 2025-01-15

    # Validate all posts (full report)
    python3 live/newsletter_validator.py --all

    # Fetch latest post from Substack first
    python3 live/newsletter_validator.py --fetch
"""

import json
import re
import sys
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strategy.level_generator import generate_levels, PriceLevel


# ── Config ─────────────────────────────────────────────────────────────
POSTS_PATH = Path(__file__).resolve().parent.parent / "data" / "substack" / "all_posts.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORT_PATH = DATA_DIR / "newsletter_validation_report.json"
MATCH_TOLERANCE = 3.0  # points


# ── Newsletter Parsing ─────────────────────────────────────────────────

def parse_mancini_levels(text: str) -> Dict[str, List[dict]]:
    """Extract support/resistance levels from a Mancini newsletter post."""
    result = {"supports": [], "resistances": []}

    sup_match = re.search(
        r'[Ss]upports?\s+(?:are|is)[:\s]*(.+?)(?:\n\n|In terms|As readers|As usual|$)',
        text, re.DOTALL)
    if sup_match:
        result["supports"] = _parse_level_list(sup_match.group(1))

    res_match = re.search(
        r'[Rr]esistances?\s+(?:are|is)[:\s]*(.+?)(?:\n\n|In terms|As readers|As usual|$)',
        text, re.DOTALL)
    if res_match:
        result["resistances"] = _parse_level_list(res_match.group(1))

    return result


def _parse_level_list(text: str) -> List[dict]:
    """Parse comma-separated price levels like '6838, 6833 (major), 6828'."""
    levels = []
    pattern = r'(\d{4,5})(?:\s*[-–]\s*(\d{4,5}))?\s*(\(major\))?'
    for m in re.finditer(pattern, text):
        price1 = float(m.group(1))
        price2 = float(m.group(2)) if m.group(2) else None
        is_major = bool(m.group(3))
        price = (price1 + price2) / 2 if price2 else price1
        if 3000 < price < 9000:
            levels.append({"price": price, "is_major": is_major})
    return levels


def get_newsletter_date(post: dict) -> pd.Timestamp:
    """Determine which trading day a newsletter is for (next business day)."""
    pub_date = pd.Timestamp(post["date"][:10])
    next_day = pub_date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)
    return next_day


# ── Level Matching ─────────────────────────────────────────────────────

def match_levels(
    engine_levels: List[PriceLevel],
    newsletter_levels: List[dict],
    tolerance: float = MATCH_TOLERANCE,
) -> Dict:
    """Match engine levels against newsletter levels."""
    if not engine_levels or not newsletter_levels:
        return {
            "matches": 0, "total_engine": len(engine_levels or []),
            "total_newsletter": len(newsletter_levels or []),
            "match_rate": 0.0, "avg_error": float("inf"),
            "matched": [], "missed_newsletter": [], "extra_engine": [],
        }

    eng_prices = [l.price for l in engine_levels]
    nl_prices = [a["price"] for a in newsletter_levels]

    matched = []
    used_nl = set()
    used_eng = set()

    for i, ep in enumerate(eng_prices):
        best_dist, best_j = float("inf"), -1
        for j, np_ in enumerate(nl_prices):
            if j in used_nl:
                continue
            dist = abs(ep - np_)
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j >= 0 and best_dist <= tolerance:
            matched.append({
                "engine": ep, "newsletter": nl_prices[best_j],
                "diff": round(best_dist, 2),
                "is_major": newsletter_levels[best_j]["is_major"],
            })
            used_nl.add(best_j)
            used_eng.add(i)

    missed_nl = [newsletter_levels[j] for j in range(len(nl_prices)) if j not in used_nl]
    extra_eng = [engine_levels[i] for i in range(len(eng_prices)) if i not in used_eng]

    n = len(matched)
    return {
        "matches": n,
        "total_engine": len(eng_prices),
        "total_newsletter": len(nl_prices),
        "match_rate": n / len(nl_prices) if nl_prices else 0,
        "avg_error": round(np.mean([m["diff"] for m in matched]), 2) if matched else float("inf"),
        "matched": matched,
        "missed_newsletter": missed_nl,
        "extra_engine": [{"price": l.price, "source": l.source} for l in extra_eng],
    }


# ── Level Quality ──────────────────────────────────────────────────────

def measure_reactions(
    levels: List[dict],
    day_bars: pd.DataFrame,
    is_support: bool = True,
    touch_threshold: float = 2.0,
    bounce_threshold: float = 5.0,
) -> Dict:
    """Measure how many levels got a price reaction during the trading day."""
    if not levels or len(day_bars) == 0:
        return {"tested": 0, "reacted": 0, "reaction_rate": 0.0, "details": []}

    lows = day_bars["low"].values
    highs = day_bars["high"].values
    details = []

    for lvl in levels:
        price = lvl["price"]
        touched = False
        bounced = False

        if is_support:
            touch_mask = lows <= price + touch_threshold
            if touch_mask.any():
                touched = True
                first_touch = np.argmax(touch_mask)
                post_touch_highs = highs[first_touch:]
                if len(post_touch_highs) > 0 and (post_touch_highs.max() - price) >= bounce_threshold:
                    bounced = True
        else:
            touch_mask = highs >= price - touch_threshold
            if touch_mask.any():
                touched = True
                first_touch = np.argmax(touch_mask)
                post_touch_lows = lows[first_touch:]
                if len(post_touch_lows) > 0 and (price - post_touch_lows.min()) >= bounce_threshold:
                    bounced = True

        details.append({
            "price": price, "is_major": lvl.get("is_major", False),
            "touched": touched, "bounced": bounced,
        })

    reacted = sum(1 for d in details if d["bounced"])
    tested = sum(1 for d in details if d["touched"])
    return {
        "tested": tested, "reacted": reacted,
        "reaction_rate": reacted / tested if tested > 0 else 0.0,
        "details": details,
    }


# ── Daily Validation ───────────────────────────────────────────────────

def validate_day(
    post: dict,
    price_data: pd.DataFrame,
    verbose: bool = True,
) -> Optional[Dict]:
    """Validate engine levels vs newsletter for one trading day.

    Returns validation result dict, or None if data unavailable.
    """
    target_date = get_newsletter_date(post)
    nl_levels = parse_mancini_levels(post["text"])
    n_sup = len(nl_levels["supports"])
    n_res = len(nl_levels["resistances"])

    if n_sup == 0 and n_res == 0:
        if verbose:
            print(f"  {target_date.date()}: No levels found in newsletter")
        return None

    # Get prior data for level generation (no lookahead)
    prior_data = price_data[price_data.index.date < target_date.date()]
    if len(prior_data) < 100:
        if verbose:
            print(f"  {target_date.date()}: Insufficient prior data ({len(prior_data)} bars)")
        return None

    # Generate engine levels (use last prior close as current_price)
    current_price = prior_data["close"].iloc[-1]
    engine_result = generate_levels(prior_data, current_price)
    if engine_result is None:
        if verbose:
            print(f"  {target_date.date()}: Engine failed to generate levels")
        return None

    # Separate engine supports and resistances
    engine_supports = [l for l in engine_result.supports]
    engine_resistances = [l for l in engine_result.resistances]

    # Match
    sup_match = match_levels(engine_supports, nl_levels["supports"])
    res_match = match_levels(engine_resistances, nl_levels["resistances"])

    # Combined match rate
    total_nl = n_sup + n_res
    total_matches = sup_match["matches"] + res_match["matches"]
    combined_rate = total_matches / total_nl if total_nl > 0 else 0

    # Get day's price data for reaction measurement
    day_bars = price_data[
        (price_data.index.date == target_date.date()) &
        (price_data.index.time >= dt_time(9, 30)) &
        (price_data.index.time < dt_time(16, 0))
    ]

    # Measure reactions for newsletter levels
    nl_sup_reactions = measure_reactions(nl_levels["supports"], day_bars, is_support=True)
    nl_res_reactions = measure_reactions(nl_levels["resistances"], day_bars, is_support=False)

    result = {
        "date": str(target_date.date()),
        "post_title": post["title"],
        "newsletter_supports": n_sup,
        "newsletter_resistances": n_res,
        "engine_supports": len(engine_supports),
        "engine_resistances": len(engine_resistances),
        "support_match": sup_match,
        "resistance_match": res_match,
        "combined_match_rate": round(combined_rate, 3),
        "combined_avg_error": round(
            np.mean([sup_match["avg_error"], res_match["avg_error"]])
            if sup_match["avg_error"] < 100 and res_match["avg_error"] < 100
            else max(sup_match["avg_error"], res_match["avg_error"]) if min(sup_match["avg_error"], res_match["avg_error"]) > 100
            else min(sup_match["avg_error"], res_match["avg_error"]), 2),
        "nl_support_reactions": nl_sup_reactions,
        "nl_resistance_reactions": nl_res_reactions,
        # Identify systematic misses for engine improvement
        "missed_supports": sup_match["missed_newsletter"],
        "missed_resistances": res_match["missed_newsletter"],
    }

    if verbose:
        _print_day_result(result)

    return result


def _print_day_result(r: Dict) -> None:
    """Pretty print one day's validation result."""
    date = r["date"]
    sm = r["support_match"]
    rm = r["resistance_match"]
    cr = r["combined_match_rate"]

    status = "ALL MATCH" if cr >= 0.8 else "GOOD" if cr >= 0.6 else "PARTIAL" if cr >= 0.4 else "WEAK"
    print(f"\n  {date} [{status}] — {r['post_title'][:50]}")
    print(f"    Supports:    {sm['matches']}/{sm['total_newsletter']} matched "
          f"({sm['match_rate']*100:.0f}%), engine had {sm['total_engine']}")
    print(f"    Resistances: {rm['matches']}/{rm['total_newsletter']} matched "
          f"({rm['match_rate']*100:.0f}%), engine had {rm['total_engine']}")
    print(f"    Combined:    {cr*100:.1f}% match rate, avg error {r['combined_avg_error']:.1f} pts")

    # Show missed levels (these are what the engine should learn)
    if r["missed_supports"]:
        missed_str = ", ".join(f"{l['price']:.0f}" + (" (MAJOR)" if l["is_major"] else "")
                              for l in r["missed_supports"][:5])
        print(f"    Missed sup:  {missed_str}")
    if r["missed_resistances"]:
        missed_str = ", ".join(f"{l['price']:.0f}" + (" (MAJOR)" if l["is_major"] else "")
                              for l in r["missed_resistances"][:5])
        print(f"    Missed res:  {missed_str}")

    # Newsletter level reaction quality
    sr = r["nl_support_reactions"]
    rr = r["nl_resistance_reactions"]
    if sr["tested"] > 0 or rr["tested"] > 0:
        print(f"    NL quality:  sup {sr['reacted']}/{sr['tested']} reacted, "
              f"res {rr['reacted']}/{rr['tested']} reacted")


# ── Aggregated Report ──────────────────────────────────────────────────

def generate_report(results: List[Dict]) -> Dict:
    """Aggregate validation results into an improvement report."""
    if not results:
        return {"error": "No results"}

    match_rates = [r["combined_match_rate"] for r in results]
    n = len(results)

    # Collect all missed levels to find systematic gaps
    all_missed_sup = []
    all_missed_res = []
    for r in results:
        for m in r.get("missed_supports", []):
            all_missed_sup.append({"price": m["price"], "is_major": m["is_major"], "date": r["date"]})
        for m in r.get("missed_resistances", []):
            all_missed_res.append({"price": m["price"], "is_major": m["is_major"], "date": r["date"]})

    # Find recurring missed price zones (levels that appear across multiple days)
    missed_major_sup = [m for m in all_missed_sup if m["is_major"]]
    missed_major_res = [m for m in all_missed_res if m["is_major"]]

    # Accuracy trend (monthly)
    monthly = {}
    for r in results:
        month = r["date"][:7]
        if month not in monthly:
            monthly[month] = []
        monthly[month].append(r["combined_match_rate"])

    monthly_avg = {k: round(np.mean(v), 3) for k, v in sorted(monthly.items())}

    # Reaction quality
    nl_reactions = []
    for r in results:
        sr = r.get("nl_support_reactions", {})
        rr = r.get("nl_resistance_reactions", {})
        tested = sr.get("tested", 0) + rr.get("tested", 0)
        reacted = sr.get("reacted", 0) + rr.get("reacted", 0)
        if tested > 0:
            nl_reactions.append(reacted / tested)

    report = {
        "summary": {
            "days_validated": n,
            "avg_match_rate": round(np.mean(match_rates), 3),
            "median_match_rate": round(np.median(match_rates), 3),
            "min_match_rate": round(min(match_rates), 3),
            "max_match_rate": round(max(match_rates), 3),
            "days_above_80pct": sum(1 for r in match_rates if r >= 0.8),
            "days_above_60pct": sum(1 for r in match_rates if r >= 0.6),
            "days_below_40pct": sum(1 for r in match_rates if r < 0.4),
        },
        "monthly_trend": monthly_avg,
        "missed_analysis": {
            "total_missed_supports": len(all_missed_sup),
            "total_missed_resistances": len(all_missed_res),
            "missed_major_supports": len(missed_major_sup),
            "missed_major_resistances": len(missed_major_res),
        },
        "newsletter_quality": {
            "avg_reaction_rate": round(np.mean(nl_reactions), 3) if nl_reactions else 0,
        },
        "improvement_suggestions": _suggest_improvements(results, all_missed_sup, all_missed_res),
    }

    return report


def _suggest_improvements(results, missed_sup, missed_res) -> List[str]:
    """Analyze systematic misses and suggest engine improvements."""
    suggestions = []

    # Check if we consistently miss major levels
    n_major_missed = sum(1 for m in missed_sup + missed_res if m["is_major"])
    n_total_missed = len(missed_sup) + len(missed_res)
    if n_total_missed > 0:
        major_pct = n_major_missed / n_total_missed
        if major_pct > 0.3:
            suggestions.append(
                f"Engine misses {major_pct:.0%} major levels — consider wider "
                f"lookback or multi-day shelf detection"
            )

    # Check match rate trend
    rates = [r["combined_match_rate"] for r in results]
    if len(rates) > 20:
        first_half = np.mean(rates[:len(rates)//2])
        second_half = np.mean(rates[len(rates)//2:])
        if second_half > first_half + 0.05:
            suggestions.append(
                f"Match rate improving ({first_half:.0%} -> {second_half:.0%})"
            )
        elif second_half < first_half - 0.05:
            suggestions.append(
                f"Match rate declining ({first_half:.0%} -> {second_half:.0%}) — "
                f"check if Mancini changed methodology"
            )

    # Check support vs resistance imbalance
    sup_rates = [r["support_match"]["match_rate"] for r in results if r["support_match"]["total_newsletter"] > 0]
    res_rates = [r["resistance_match"]["match_rate"] for r in results if r["resistance_match"]["total_newsletter"] > 0]
    if sup_rates and res_rates:
        avg_sup = np.mean(sup_rates)
        avg_res = np.mean(res_rates)
        if avg_sup < avg_res - 0.1:
            suggestions.append(
                f"Support detection weaker ({avg_sup:.0%}) than resistance ({avg_res:.0%}) "
                f"— improve swing low detection"
            )
        elif avg_res < avg_sup - 0.1:
            suggestions.append(
                f"Resistance detection weaker ({avg_res:.0%}) than support ({avg_sup:.0%}) "
                f"— improve swing high detection"
            )

    if not suggestions:
        suggestions.append("No systematic issues detected — engine aligned with newsletter")

    return suggestions


# ── Substack Fetcher ───────────────────────────────────────────────────

MANCINI_PUB_ID = 492935  # Mancini's Substack publication ID
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_substack_cookie() -> Optional[str]:
    """Load SUBSTACK_COOKIE from .env file or environment."""
    import os
    cookie = os.environ.get("SUBSTACK_COOKIE")
    if cookie:
        return cookie
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("SUBSTACK_COOKIE="):
                return line.split("=", 1)[1].strip()
    return None


def _html_to_text(html: str) -> str:
    """Simple HTML to plain text conversion."""
    text = re.sub(r'<br\s*/?>|</p>|</div>|</li>', '\n', html)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_post_text(html: str) -> str:
    """Extract post body text from a full Substack post page HTML."""
    # Extract all paragraph content
    paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL)

    # Filter to paragraphs with trading content (levels, analysis)
    trading_keywords = [
        'support', 'resistance', 'spx', 'bulls', 'bears', 'level',
        'failed breakdown', 'long', 'short', 'target', 'stop',
        'bounce', 'reclaim', 'major', 'elevator', 'chop',
    ]
    trading_ps = [
        p for p in paragraphs
        if any(kw in p.lower() for kw in trading_keywords)
    ]

    if not trading_ps:
        trading_ps = paragraphs

    full_text = '\n'.join(trading_ps)
    return _html_to_text(full_text)


def fetch_latest_posts(n: int = 5) -> List[dict]:
    """Fetch the latest N Mancini posts from Substack using cookie auth.

    Uses the substack.com reader API to list posts, then fetches each
    individual post page for full content.

    Requires SUBSTACK_COOKIE in .env or environment.
    To get your cookie:
      1. Log in to substack.com in your browser
      2. Open DevTools (F12) → Application → Cookies → substack.com
      3. Find 'substack.sid' and copy its Value
      4. Add to .env: SUBSTACK_COOKIE=substack.sid=<value>

    Returns list of post dicts or empty list if fetch fails.
    """
    try:
        import requests
    except ImportError:
        print("  requests not installed. Run: python3 -m pip install requests")
        return []

    cookie = _load_substack_cookie()
    if not cookie:
        print("  No SUBSTACK_COOKIE found.")
        print("  To set up daily fetching:")
        print("    1. Log in to substack.com in your browser")
        print("    2. Open DevTools (F12) → Application → Cookies")
        print("    3. Find 'substack.sid' and copy its Value")
        print(f"    4. Add to {ENV_PATH}: SUBSTACK_COOKIE=substack.sid=<value>")
        return []

    headers = {
        "Cookie": cookie,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }

    try:
        # Step 1: List recent posts from reader feed
        resp = requests.get(
            f"https://substack.com/api/v1/reader/posts?limit={n * 3}",
            headers=headers, timeout=15,
        )
        if resp.status_code != 200:
            print(f"  Substack reader API returned {resp.status_code}")
            if resp.status_code == 403:
                print("  Cookie may be expired — re-copy from browser.")
            return []

        data = resp.json()
        all_posts = data.get("posts", [])

        # Filter to Mancini's publication
        mancini_posts = [p for p in all_posts if p.get("publication_id") == MANCINI_PUB_ID]
        print(f"  Found {len(mancini_posts)} Mancini posts in feed (of {len(all_posts)} total)")

        if not mancini_posts:
            print("  No Mancini posts found. Check publication ID.")
            return []

        # Step 2: Fetch full content for each post
        fetched = []
        for p in mancini_posts[:n]:
            post_id = p["id"]
            title = p.get("title", "")
            print(f"  Fetching: {title[:60]}...")

            page_resp = requests.get(
                f"https://substack.com/home/post/p-{post_id}",
                headers=headers, timeout=20,
            )
            if page_resp.status_code != 200:
                print(f"    Failed ({page_resp.status_code})")
                continue

            text = _extract_post_text(page_resp.text)
            if len(text) < 200:
                print(f"    Content too short ({len(text)} chars), skipping")
                continue

            # Verify it has trading levels
            has_levels = "support" in text.lower() and "resistance" in text.lower()
            if not has_levels:
                print(f"    No support/resistance levels found, skipping")
                continue

            fetched.append({
                "title": title,
                "date": p.get("post_date", ""),
                "slug": p.get("slug", ""),
                "wordcount": p.get("wordcount", 0),
                "text": text,
                "has_content": True,
            })
            print(f"    OK ({len(text)} chars)")

        print(f"  Fetched {len(fetched)} complete Mancini posts")
        return fetched

    except Exception as e:
        print(f"  Substack fetch failed: {e}")
        return []


def fetch_latest_post() -> Optional[dict]:
    """Fetch the single most recent post. Wrapper for backwards compat."""
    posts = fetch_latest_posts(n=1)
    return posts[0] if posts else None


def append_post(post: dict) -> bool:
    """Append a new post to the posts JSON file (deduplicates by title)."""
    posts = []
    if POSTS_PATH.exists():
        with open(POSTS_PATH) as f:
            posts = json.load(f)

    # Dedup by title
    existing_titles = {p["title"] for p in posts}
    if post["title"] in existing_titles:
        print(f"  Post already exists: {post['title']}")
        return False

    posts.append(post)
    posts.sort(key=lambda p: p["date"], reverse=True)

    with open(POSTS_PATH, "w") as f:
        json.dump(posts, f, indent=2)

    print(f"  Added post: {post['title']}")
    return True


# ── Main ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Daily newsletter level validation")
    parser.add_argument("--date", help="Validate specific date (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Validate all posts")
    parser.add_argument("--last", type=int, default=5, help="Validate last N posts (default 5)")
    parser.add_argument("--fetch", action="store_true", help="Fetch latest post first")
    parser.add_argument("--report", action="store_true", help="Generate improvement report")
    args = parser.parse_args()

    # Load posts
    if not POSTS_PATH.exists():
        print(f"No posts file at {POSTS_PATH}")
        return

    with open(POSTS_PATH) as f:
        posts = json.load(f)
    print(f"Loaded {len(posts)} newsletter posts")

    # Fetch latest
    if args.fetch:
        new_posts = fetch_latest_posts(n=5)
        added = 0
        for new_post in new_posts:
            if append_post(new_post):
                added += 1
        if added > 0:
            print(f"  Added {added} new posts")
            with open(POSTS_PATH) as f:
                posts = json.load(f)

    # Load price data
    data_files = sorted(DATA_DIR.glob("ES_1m_full_session_*.parquet"))
    if not data_files:
        data_files = sorted(DATA_DIR.glob("ES_1m_*.parquet"))
    if not data_files:
        print("No price data found in data/")
        return

    print(f"Loading price data from {data_files[-1].name}...")
    df = pd.read_parquet(data_files[-1])
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")

    # Filter posts by date if specified
    if args.date:
        target = pd.Timestamp(args.date)
        selected = [p for p in posts if get_newsletter_date(p).date() == target.date()]
        if not selected:
            print(f"No newsletter found for {args.date}")
            return
    elif args.all:
        selected = posts
    else:
        selected = posts[:args.last]

    # Run validation
    print(f"\n{'='*70}")
    print(f"NEWSLETTER LEVEL VALIDATION ({len(selected)} posts)")
    print(f"{'='*70}")

    results = []
    for post in selected:
        r = validate_day(post, df, verbose=True)
        if r:
            results.append(r)

    if not results:
        print("\nNo valid results")
        return

    # Summary
    rates = [r["combined_match_rate"] for r in results]
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(results)} days validated")
    print(f"{'='*70}")
    print(f"  Avg match rate:    {np.mean(rates)*100:.1f}%")
    print(f"  Median:            {np.median(rates)*100:.1f}%")
    print(f"  Range:             {min(rates)*100:.1f}% — {max(rates)*100:.1f}%")
    print(f"  Days >= 80%:       {sum(1 for r in rates if r >= 0.8)}/{len(results)}")
    print(f"  Days >= 60%:       {sum(1 for r in rates if r >= 0.6)}/{len(results)}")

    # Generate improvement report
    if args.report or args.all:
        report = generate_report(results)
        print(f"\n{'='*70}")
        print("IMPROVEMENT REPORT")
        print(f"{'='*70}")
        for k, v in report["summary"].items():
            print(f"  {k}: {v}")
        print(f"\n  Monthly trend:")
        for month, rate in report["monthly_trend"].items():
            print(f"    {month}: {rate*100:.1f}%")
        print(f"\n  Suggestions:")
        for s in report["improvement_suggestions"]:
            print(f"    - {s}")

        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
