"""Batch-parse all Mancini Substack posts into structured level files.

Fetches the archive, parses each post, writes mancini_levels_YYYY-MM-DD.json
for each trading date. These files are used by the backtest to replay with
Mancini's actual level calls.

Usage:
    SUBSTACK_COOKIE=... python3 backtest/parse_all_mancini_posts.py [--limit 500] [--output-dir data/mancini_levels]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time as time_mod
import urllib.request
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from live.substack_compare import (
    extract_body_html,
    extract_levels_from_text,
    extract_highlights,
)

SUBSTACK_BASE = "https://tradecompanion.substack.com"
SUBSTACK_COOKIE = os.environ.get("SUBSTACK_COOKIE", "")
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "mancini_levels"


# ---------------------------------------------------------------------------
# Fetching helpers
# ---------------------------------------------------------------------------

def fetch_archive_page(offset: int, limit: int = 12) -> list[dict]:
    """Fetch one page from the Substack archive API.

    Returns list of post metadata dicts (slug, title, post_date, etc).
    """
    url = f"{SUBSTACK_BASE}/api/v1/archive?sort=new&limit={limit}&offset={offset}"
    req = urllib.request.Request(url)
    req.add_header("Cookie", SUBSTACK_COOKIE)
    req.add_header("User-Agent", "Mozilla/5.0")

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch_post_body(slug: str) -> str | None:
    """Fetch the full body_html for a single post by slug.

    Returns cleaned plain text, or None on failure.
    """
    url = f"{SUBSTACK_BASE}/api/v1/posts/{slug}"
    req = urllib.request.Request(url)
    req.add_header("Cookie", SUBSTACK_COOKIE)
    req.add_header("User-Agent", "Mozilla/5.0")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            detail = json.loads(resp.read().decode())
    except Exception as e:
        print(f"    Failed to fetch post {slug}: {e}")
        return None

    body_html = detail.get("body_html", "")
    if body_html:
        # Clean HTML to plain text (same logic as substack_compare.py)
        clean = re.sub(r"<[^>]+>", " ", body_html)
        clean = re.sub(r"&amp;", "&", clean)
        clean = re.sub(r"&nbsp;", " ", clean)
        clean = re.sub(r"&#x200B;", "", clean)
        clean = re.sub(r"&lt;", "<", clean)
        clean = re.sub(r"&gt;", ">", clean)
        clean = re.sub(r"&[a-z]+;", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    # Fallback: fetch the page and extract body_html from embedded JSON
    page_url = f"{SUBSTACK_BASE}/p/{slug}"
    req2 = urllib.request.Request(page_url)
    req2.add_header("Cookie", SUBSTACK_COOKIE)
    req2.add_header("User-Agent", "Mozilla/5.0")

    try:
        with urllib.request.urlopen(req2, timeout=30) as resp2:
            page_text = resp2.read().decode(errors="replace")
        return extract_body_html(page_text)
    except Exception as e:
        print(f"    Fallback page fetch failed for {slug}: {e}")
        return None


def fetch_all_posts(max_posts: int = 500) -> list[dict]:
    """Paginate through the Substack archive, returning all post metadata."""
    all_posts = []
    offset = 0
    page_size = 12

    while offset < max_posts:
        print(f"  Fetching archive offset={offset}...")
        try:
            page = fetch_archive_page(offset, limit=page_size)
        except Exception as e:
            print(f"  Archive fetch failed at offset={offset}: {e}")
            break

        if not page:
            print(f"  No more posts at offset={offset}")
            break

        all_posts.extend(page)
        offset += len(page)

        # If the page returned fewer than requested, we've hit the end
        if len(page) < page_size:
            break

        time_mod.sleep(0.5)  # light rate limit between archive pages

    print(f"  Total post metadata fetched: {len(all_posts)}")
    return all_posts[:max_posts]


# ---------------------------------------------------------------------------
# Parsing logic
# ---------------------------------------------------------------------------

def determine_trading_date(post_date_str: str) -> date | None:
    """Determine the trading date for a post.

    Mancini publishes around 4 PM ET for the NEXT trading day.
    - Posts published Mon-Thu -> next calendar day
    - Posts published Fri -> next Monday (skip weekend)
    - Posts published Sat -> next Monday
    - Posts published Sun -> next Monday
    """
    try:
        post_date = datetime.strptime(post_date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None

    weekday = post_date.weekday()  # Mon=0, Sun=6
    if weekday == 4:  # Friday -> Monday
        return post_date + timedelta(days=3)
    elif weekday == 5:  # Saturday -> Monday
        return post_date + timedelta(days=2)
    elif weekday == 6:  # Sunday -> Monday
        return post_date + timedelta(days=1)
    else:  # Mon-Thu -> next day
        return post_date + timedelta(days=1)


def determine_price_range_for_date(trading_date: date) -> tuple[float, float]:
    """Estimate the ES price range for a given trading date.

    Used to filter extracted levels to plausible values.
    ES has ranged roughly:
      2024-06: ~5400-5600
      2024-09: ~5400-5800
      2024-12: ~5800-6100
      2025-03: ~5600-6000
      2025-06: ~5900-6300
      2025-09: ~6000-6500
      2025-12: ~6200-6800
      2026-02: ~6400-7000
    Use a generous window: min - 500 to max + 500
    """
    year = trading_date.year
    if year <= 2024:
        return (4800.0, 6200.0)
    elif year == 2025:
        month = trading_date.month
        if month <= 3:
            return (5200.0, 6200.0)
        elif month <= 6:
            return (5400.0, 6500.0)
        elif month <= 9:
            return (5600.0, 6800.0)
        else:
            return (5800.0, 7200.0)
    else:  # 2026+
        return (6000.0, 7500.0)


def parse_post_to_levels(body_text: str, trading_date: date) -> dict | None:
    """Parse a post body into structured Mancini levels data.

    Returns a dict suitable for JSON serialization, or None if no levels found.
    """
    price_range = determine_price_range_for_date(trading_date)
    levels = extract_levels_from_text(body_text, current_price_range=price_range)
    highlights = extract_highlights(body_text)

    if not levels:
        return None

    # Classify levels into support/resistance for the overlay
    supports = [lv for lv in levels if lv["role"] in ("support", "level", "range", "mentioned")]
    resistances = [lv for lv in levels if lv["role"] in ("resistance", "target")]

    return {
        "trading_date": str(trading_date),
        "levels": levels,
        "supports": supports,
        "resistances": resistances,
        "highlights": highlights,
        "level_count": len(levels),
        "support_count": len(supports),
        "resistance_count": len(resistances),
    }


# ---------------------------------------------------------------------------
# Main batch processing
# ---------------------------------------------------------------------------

def batch_parse(
    max_posts: int = 500,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sleep_between: float = 1.0,
) -> dict:
    """Fetch and parse all Mancini posts, writing level files.

    Returns summary stats dict.
    """
    if not SUBSTACK_COOKIE:
        print("ERROR: SUBSTACK_COOKIE environment variable not set.")
        print("Set it with: export SUBSTACK_COOKIE='substack.sid=...'")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch all post metadata
    print("=" * 60)
    print("STEP 1: Fetching post archive metadata")
    print("=" * 60)
    posts = fetch_all_posts(max_posts)

    if not posts:
        print("No posts fetched. Check SUBSTACK_COOKIE.")
        return {"total": 0, "parsed": 0, "failed": 0, "skipped": 0}

    # Step 2: Parse each post
    print()
    print("=" * 60)
    print(f"STEP 2: Parsing {len(posts)} posts (est. {len(posts)} seconds)")
    print("=" * 60)

    parsed = 0
    failed = 0
    skipped = 0
    files_written = 0
    dates_covered: list[str] = []

    for i, meta in enumerate(posts):
        slug = meta.get("slug", "")
        title = meta.get("title", "?")
        post_date_str = meta.get("post_date", "")[:10]

        # Skip non-trading posts (e.g., "Welcome", "About", etc.)
        if not re.search(r'\d{4}', title) and not re.search(r'(?:plan|levels|ES|NQ|market)', title, re.I):
            # Check if it looks like a trading post at all
            if not re.search(r'(?:support|resistance|bull|bear|long|short)', title, re.I):
                print(f"  [{i+1}/{len(posts)}] SKIP (non-trading): {title}")
                skipped += 1
                continue

        # Determine trading date
        trading_date = determine_trading_date(post_date_str)
        if trading_date is None:
            print(f"  [{i+1}/{len(posts)}] SKIP (no date): {title}")
            skipped += 1
            continue

        # Check if file already exists
        out_file = output_dir / f"mancini_levels_{trading_date}.json"
        if out_file.exists():
            print(f"  [{i+1}/{len(posts)}] EXISTS: {trading_date} ({title[:40]}...)")
            parsed += 1
            dates_covered.append(str(trading_date))
            continue

        # Fetch full post body
        print(f"  [{i+1}/{len(posts)}] Fetching: {title[:50]}... -> {trading_date}")
        body_text = fetch_post_body(slug)

        if body_text is None or len(body_text) < 100:
            print(f"    FAILED: empty or too short body ({len(body_text or '')} chars)")
            failed += 1
            time_mod.sleep(sleep_between)
            continue

        # Parse levels
        try:
            level_data = parse_post_to_levels(body_text, trading_date)
        except Exception as e:
            print(f"    FAILED: parse error: {e}")
            failed += 1
            time_mod.sleep(sleep_between)
            continue

        if level_data is None or level_data["level_count"] == 0:
            print(f"    SKIP: no levels extracted from {len(body_text)} chars")
            skipped += 1
            time_mod.sleep(sleep_between)
            continue

        # Add metadata
        level_data["post_title"] = title
        level_data["post_date"] = post_date_str
        level_data["post_slug"] = slug
        level_data["parsed_at"] = datetime.now().isoformat()
        level_data["body_length"] = len(body_text)

        # Write file
        with open(out_file, "w") as f:
            json.dump(level_data, f, indent=2, default=str)

        print(f"    OK: {level_data['level_count']} levels "
              f"({level_data['support_count']}S/{level_data['resistance_count']}R)")
        parsed += 1
        files_written += 1
        dates_covered.append(str(trading_date))

        time_mod.sleep(sleep_between)

    # Step 3: Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Posts fetched:      {len(posts)}")
    print(f"  Posts parsed:       {parsed}")
    print(f"  Files written:      {files_written}")
    print(f"  Failures:           {failed}")
    print(f"  Skipped:            {skipped}")

    if dates_covered:
        dates_sorted = sorted(dates_covered)
        print(f"  Date range:         {dates_sorted[0]} to {dates_sorted[-1]}")
        print(f"  Trading days:       {len(dates_covered)}")

    print(f"\n  Output directory:   {output_dir}")

    return {
        "total": len(posts),
        "parsed": parsed,
        "files_written": files_written,
        "failed": failed,
        "skipped": skipped,
        "dates_covered": len(dates_covered),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch-parse all Mancini Substack posts into structured level files"
    )
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Maximum number of posts to fetch (default: 500)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for level JSON files"
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="Seconds to sleep between post fetches (default: 1.0)"
    )
    args = parser.parse_args()

    batch_parse(
        max_posts=args.limit,
        output_dir=Path(args.output_dir),
        sleep_between=args.sleep,
    )


if __name__ == "__main__":
    main()
