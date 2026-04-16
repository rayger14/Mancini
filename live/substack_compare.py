"""Daily Substack vs Engine level comparison.

Extracts price levels and key highlights from Mancini's latest Substack post,
then compares his mentioned levels to the engine's detected levels.

Usage:
    python3 live/substack_compare.py                    # compare latest post
    python3 live/substack_compare.py --date 2026-02-24  # compare specific date

Cron (9 PM ET daily = 2 AM UTC):
    0 2 * * * docker exec mancini-bot python3 live/substack_compare.py >> /app/logs/substack_cron.log 2>&1
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

POSTS_FILE = os.environ.get("POSTS_FILE", "/app/data/substack/all_posts.json")
STATUS_FILE = os.environ.get("STATUS_FILE", "/app/logs/status.json")
OUTPUT_DIR = os.environ.get("COMPARE_OUTPUT", "/app/logs")
SUBSTACK_COOKIE = os.environ.get("SUBSTACK_COOKIE", "")
SUBSTACK_BASE = "https://tradecompanion.substack.com"


def fetch_latest_post() -> dict | None:
    """Fetch the latest post from Substack API using session cookie.

    Returns a dict with keys: title, date, slug, text, body_html, has_content.
    Falls back to None if fetch fails.
    """
    import urllib.request

    cookie = SUBSTACK_COOKIE
    if not cookie:
        print("No SUBSTACK_COOKIE set, skipping live fetch")
        return None

    try:
        # Step 1: Get latest post metadata
        url = f"{SUBSTACK_BASE}/api/v1/posts?limit=1"
        req = urllib.request.Request(url)
        req.add_header("Cookie", cookie)
        req.add_header("User-Agent", "Mozilla/5.0")

        with urllib.request.urlopen(req, timeout=15) as resp:
            posts = json.loads(resp.read().decode())

        if not posts:
            print("API returned no posts")
            return None

        meta = posts[0]
        slug = meta.get("slug", "")
        title = meta.get("title", "?")
        post_date = meta.get("post_date", "")[:10]
        print(f"Latest post: {title} ({post_date})")

        # Step 2: Fetch full post content (body_html)
        detail_url = f"{SUBSTACK_BASE}/api/v1/posts/{slug}"
        req2 = urllib.request.Request(detail_url)
        req2.add_header("Cookie", cookie)
        req2.add_header("User-Agent", "Mozilla/5.0")

        with urllib.request.urlopen(req2, timeout=15) as resp2:
            detail = json.loads(resp2.read().decode())

        body_html = detail.get("body_html", "")
        if not body_html:
            print("  body_html is empty (paywall?), trying page scrape...")
            # Fallback: fetch the page directly and extract body_html from embedded JSON
            page_url = f"{SUBSTACK_BASE}/p/{slug}"
            req3 = urllib.request.Request(page_url)
            req3.add_header("Cookie", cookie)
            req3.add_header("User-Agent", "Mozilla/5.0")

            with urllib.request.urlopen(req3, timeout=15) as resp3:
                page_text = resp3.read().decode(errors="replace")

            return {
                "title": title,
                "date": post_date,
                "slug": slug,
                "text": page_text,
                "has_content": True,
                "source": "live_page",
            }

        # Clean HTML body to plain text for extraction
        clean_text = re.sub(r"<[^>]+>", " ", body_html)
        clean_text = re.sub(r"&amp;", "&", clean_text)
        clean_text = re.sub(r"&nbsp;", " ", clean_text)
        clean_text = re.sub(r"&#x200B;", "", clean_text)
        clean_text = re.sub(r"&[a-z]+;", " ", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        return {
            "title": title,
            "date": post_date,
            "slug": slug,
            "text": "",
            "body_html_clean": clean_text,
            "has_content": True,
            "source": "live_api",
        }

    except Exception as e:
        print(f"Live fetch failed: {e}")
        return None


def load_posts(path: str = POSTS_FILE) -> list[dict]:
    """Load all cached Substack posts."""
    p = Path(path)
    if not p.exists():
        p = Path(__file__).parent.parent / path
    if not p.exists():
        print(f"Posts file not found: {path}")
        return []
    return json.loads(p.read_text())


def extract_body_html(raw_text: str) -> str:
    """Extract the actual post body from the raw Substack page dump.

    The page embeds the post content in a JSON field: body_html\\":\\"...content...\\"
    which is double-escaped because the page JSON is inside another JSON string.
    """
    # Find body_html field in the double-escaped JSON
    idx = raw_text.find('body_html')
    if idx < 0:
        # Fallback: just strip HTML from entire text
        return re.sub(r"<[^>]+>", " ", raw_text)

    # Content starts right after body_html\\":\\"
    content_markers = ['body_html\\":\\"', "body_html\\\":\\\""]
    start = -1
    for marker in content_markers:
        pos = raw_text.find(marker, idx - 5)
        if pos >= 0:
            start = pos + len(marker)
            break

    if start < 0:
        # Try: content starts with first sentence-like text after body_html
        # Look for capital letter starting a sentence
        m = re.search(r'body_html.*?([A-Z][a-z]{2,})', raw_text[idx:idx+200])
        if m:
            start = idx + m.start(1)
        else:
            return re.sub(r"<[^>]+>", " ", raw_text)

    # Find end: look for the closing pattern (next JSON key after body_html value)
    # The value ends with \\",\\" (next key) or \\"}
    end = len(raw_text)
    for end_marker in ['\\"\\n\\",\\"', '\\",\\"', '\\"}\\"', '",\\"truncated']:
        pos = raw_text.find(end_marker, start + 100)
        if 0 < pos < end:
            end = pos

    body = raw_text[start:end]

    # Unescape: \\u2019 -> unicode, \\n -> newline, \\" -> "
    body = re.sub(r'\\\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), body)
    body = body.replace('\\\\n', '\n').replace('\\\\"', '"')
    body = body.replace('\\u2019', '\u2019').replace('\\u201C', '\u201C').replace('\\u201D', '\u201D')
    body = body.replace('\\n', '\n').replace('\\"', '"')

    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", body)
    clean = re.sub(r"&amp;", "&", clean)
    clean = re.sub(r"&nbsp;", " ", clean)
    clean = re.sub(r"&#x200B;", "", clean)
    clean = re.sub(r"&lt;", "<", clean)
    clean = re.sub(r"&gt;", ">", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    return clean


def extract_highlights(body_text: str) -> list[dict]:
    """Extract key highlights from Mancini's post.

    Looks for:
    - Directional bias (bullish/bearish lean)
    - Key trade setups he calls out
    - Important levels with context
    - His explicit targets and invalidation levels
    """
    highlights = []
    sentences = re.split(r'(?<=[.!?])\s+', body_text)

    # 1. Directional lean — sentences with bullish/bearish bias
    lean_patterns = [
        r'(?:my|general)\s+lean',
        r'(?:bull|bear)(?:ish)?\s+(?:case|scenario|target|bias)',
        r'(?:if|when)\s+(?:bulls?|bears?)\s+(?:win|lose|hold|fail|defend)',
        r'defer\s+to\s+the\s+trend',
        r'overall\s+(?:lean|bias|direction)',
    ]
    for sent in sentences:
        for pat in lean_patterns:
            if re.search(pat, sent, re.I) and len(sent) > 30:
                highlights.append({"type": "DIRECTIONAL LEAN", "text": sent.strip()})
                break

    # 2. Key trade setups — Failed Breakdown, level reclaim, etc.
    setup_patterns = [
        r'(?:obvious|clear|key|ideal|best)\s+(?:trade|entry|setup|play)',
        r'failed\s+breakdown',
        r'(?:entry|enter|long|short)\s+(?:is|at|on|if)\s+',
        r'(?:buy|sell)\s+(?:the|a)\s+(?:dip|rip|flush|sweep)',
        r'elevator\s+(?:down|up)',
    ]
    for sent in sentences:
        for pat in setup_patterns:
            if re.search(pat, sent, re.I) and len(sent) > 30:
                highlights.append({"type": "TRADE SETUP", "text": sent.strip()})
                break

    # 3. Explicit targets
    target_patterns = [
        r'target[s]?\s+(?:are|is|would\s+be)\s+',
        r'(?:upside|downside)\s+target',
        r'(?:first|next|initial)\s+target',
        r'(?:heading|head)\s+(?:to|toward)',
    ]
    for sent in sentences:
        for pat in target_patterns:
            if re.search(pat, sent, re.I) and len(sent) > 20:
                highlights.append({"type": "TARGETS", "text": sent.strip()})
                break

    # 4. Key level calls — "PRICE is key", "watch PRICE", "important level"
    level_patterns = [
        r'\d{4,5}\s+(?:is\s+)?(?:key|important|critical|major|big|huge|massive)',
        r'(?:key|important|critical|watch)\s+(?:level|support|resistance).*?\d{4,5}',
        r'(?:daily\s+(?:low|high))\s+(?:at|of)\s+\d{4,5}',
        r'(?:magnet|pivot)\b',
        r'rangebound.*?\d{4,5}.*?to.*?\d{4,5}',
    ]
    for sent in sentences:
        for pat in level_patterns:
            if re.search(pat, sent, re.I) and len(sent) > 20:
                highlights.append({"type": "KEY LEVEL", "text": sent.strip()})
                break

    # 5. Invalidation / risk
    risk_patterns = [
        r'(?:if|when)\s+(?:we\s+)?(?:lose|break|fail)',
        r'invalidat',
        r'stop\s+(?:would\s+be|at|is)',
        r'(?:risk|danger|careful|caution)',
    ]
    for sent in sentences:
        for pat in risk_patterns:
            if re.search(pat, sent, re.I) and len(sent) > 30 and re.search(r'\d{4}', sent):
                highlights.append({"type": "RISK / INVALIDATION", "text": sent.strip()})
                break

    # Deduplicate (same sentence can match multiple patterns, keep first)
    seen = set()
    deduped = []
    for h in highlights:
        if h["text"] not in seen:
            seen.add(h["text"])
            deduped.append(h)

    return deduped


def extract_levels_from_text(text: str, current_price_range: tuple[float, float] = (5500, 7500)) -> list[dict]:
    """Extract price levels mentioned in cleaned post body text."""
    lo, hi = current_price_range
    levels = {}

    # Pattern 1: "support/resistance at PRICE"
    for m in re.finditer(r"(support|resistance|level)\s+(?:at|near|around)\s+(\d{4,5}(?:\.\d+)?)", text, re.I):
        price = float(m.group(2))
        if lo <= price <= hi:
            role = "support" if "support" in m.group(1).lower() else "resistance"
            levels[price] = {"price": price, "role": role, "context": m.group(0).strip()}

    # Pattern 2: "PRICE area/zone/level/support/resistance"
    for m in re.finditer(r"(\d{4,5}(?:\.\d+)?)\s+(area|zone|level|support|resistance|region)", text, re.I):
        price = float(m.group(1))
        if lo <= price <= hi and price not in levels:
            role_word = m.group(2).lower()
            role = "support" if role_word == "support" else "resistance" if role_word == "resistance" else "level"
            levels[price] = {"price": price, "role": role, "context": m.group(0).strip()}

    # Pattern 3: "above/below/hold/lose/reclaim/break PRICE"
    for m in re.finditer(r"(above|below|hold|holds|lose|loses|reclaim|reclaims|break|breaks|defend|defends|recover|recovers|swept|flush|target)\s+(\d{4,5}(?:\.\d+)?)", text, re.I):
        price = float(m.group(2))
        if lo <= price <= hi and price not in levels:
            verb = m.group(1).lower()
            if verb in ("hold", "holds", "above", "reclaim", "reclaims", "defend", "defends", "recover", "recovers"):
                role = "support"
            elif verb in ("below", "lose", "loses", "break", "breaks"):
                role = "resistance"
            elif verb in ("target",):
                role = "target"
            else:
                role = "level"
            levels[price] = {"price": price, "role": role, "context": m.group(0).strip()}

    # Pattern 4: "daily low/high at PRICE" or "PRICE daily low/high"
    for m in re.finditer(r"(?:daily\s+(?:low|high))\s+(?:at|of)?\s*(\d{4,5}(?:\.\d+)?)", text, re.I):
        price = float(m.group(1))
        if lo <= price <= hi and price not in levels:
            role = "support" if "low" in m.group(0).lower() else "resistance"
            levels[price] = {"price": price, "role": role, "context": m.group(0).strip()}

    for m in re.finditer(r"(\d{4,5}(?:\.\d+)?)\s+daily\s+(low|high)", text, re.I):
        price = float(m.group(1))
        if lo <= price <= hi and price not in levels:
            role = "support" if m.group(2).lower() == "low" else "resistance"
            levels[price] = {"price": price, "role": role, "context": m.group(0).strip()}

    # Pattern 5: "PRICE-PRICE" range
    for m in re.finditer(r"(\d{4,5})\s*[-–]\s*(\d{2,5})", text):
        p1 = float(m.group(1))
        p2_raw = m.group(2)
        # Handle shorthand like "6925-30" meaning 6925-6930
        if len(p2_raw) <= 2:
            p2 = float(m.group(1)[:-len(p2_raw)] + p2_raw)
        else:
            p2 = float(p2_raw)
        for price in (p1, p2):
            if lo <= price <= hi and price not in levels:
                levels[price] = {"price": price, "role": "range", "context": m.group(0).strip()}

    # Pattern 6: Bare prices in context (only if near other level mentions)
    for m in re.finditer(r"\b(\d{4,5})\b", text):
        price = float(m.group(1))
        if lo <= price <= hi and price not in levels:
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            ctx = text[start:end].strip()
            # Only include if surrounding context is trading-related
            if re.search(r'(dip|rip|bounce|flush|sweep|target|magnet|pivot|backtest|key|important|watch)', ctx, re.I):
                levels[price] = {"price": price, "role": "mentioned", "context": ctx}

    return sorted(levels.values(), key=lambda x: x["price"])


def load_engine_levels(status_path: str = STATUS_FILE) -> list[dict]:
    """Load current engine-detected levels from status.json."""
    p = Path(status_path)
    if not p.exists():
        return []
    try:
        status = json.loads(p.read_text())
        return status.get("levels", [])
    except (json.JSONDecodeError, OSError):
        return []


def compare_levels(
    mancini_levels: list[dict],
    engine_levels: list[dict],
    tolerance_pts: float = 3.0,
) -> dict:
    """Compare Mancini's mentioned levels to engine-detected levels."""
    engine_prices = [(lv.get("price", 0), lv) for lv in engine_levels]

    matched = []
    mancini_only = []

    for ml in mancini_levels:
        mp = ml["price"]
        best_match = None
        best_dist = tolerance_pts + 1

        for ep, elv in engine_prices:
            dist = abs(mp - ep)
            if dist <= tolerance_pts and dist < best_dist:
                best_dist = dist
                best_match = elv

        if best_match:
            matched.append({
                "mancini_price": mp,
                "engine_price": best_match["price"],
                "distance": round(mp - best_match["price"], 2),
                "engine_type": best_match.get("type", "?"),
                "engine_touches": best_match.get("touches", 0),
                "mancini_role": ml.get("role", "?"),
                "mancini_context": ml.get("context", ""),
            })
        else:
            mancini_only.append(ml)

    matched_engine_prices = {m["engine_price"] for m in matched}
    engine_only = [
        lv for lv in engine_levels
        if lv.get("price", 0) not in matched_engine_prices
    ]

    return {
        "matched": matched,
        "mancini_only": mancini_only,
        "engine_only": engine_only[:20],
    }


def find_post_for_date(posts: list[dict], target_date: date) -> dict | None:
    """Find the Mancini post that covers a given trading date.

    Posts are published the evening before (e.g. "Feb 24 Plan" published Feb 23).
    """
    for p in sorted(posts, key=lambda x: x.get("date", ""), reverse=True):
        post_date_str = p.get("date", "")[:10]
        try:
            post_date = datetime.strptime(post_date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        if post_date == target_date or post_date == target_date - timedelta(days=1):
            return p

    # Fallback: return most recent post
    if posts:
        return sorted(posts, key=lambda x: x.get("date", ""), reverse=True)[0]
    return None


def run_comparison(target_date: date | None = None) -> dict:
    """Run the full comparison and return results."""
    if target_date is None:
        target_date = date.today()

    # Try live fetch first (gets latest post with cookie auth)
    post = fetch_latest_post()
    source = "live"

    if post is None:
        # Fallback to cached posts file
        source = "cached"
        posts = load_posts()
        if not posts:
            return {"error": "No posts loaded and live fetch failed"}
        post = find_post_for_date(posts, target_date)
        if not post:
            return {"error": "No matching post found"}

    # Extract body text — different paths for live API vs page scrape vs cached
    if post.get("body_html_clean"):
        # Live API returned clean body_html directly
        body_text = post["body_html_clean"]
        print(f"  Using clean body_html from API ({len(body_text)} chars)")
    else:
        # Page scrape or cached — extract from raw page dump
        body_text = extract_body_html(post.get("text", ""))
        print(f"  Extracted body from page dump ({len(body_text)} chars)")

    # Extract highlights (directional lean, trade setups, key levels, targets, risk)
    highlights = extract_highlights(body_text)

    # Extract price levels from cleaned body
    mancini_levels = extract_levels_from_text(body_text)

    # Load engine levels
    engine_levels = load_engine_levels()

    # Compare
    comparison = compare_levels(mancini_levels, engine_levels)

    result = {
        "timestamp": datetime.now().isoformat(),
        "target_date": str(target_date),
        "post_title": post.get("title", "?"),
        "post_date": post.get("date", "?")[:10],
        "highlights": highlights,
        "mancini_levels_found": len(mancini_levels),
        "engine_levels_active": len(engine_levels),
        "matched_count": len(comparison["matched"]),
        "mancini_only_count": len(comparison["mancini_only"]),
        "engine_only_count": len(comparison["engine_only"]),
        "match_rate": round(
            len(comparison["matched"]) / max(len(mancini_levels), 1) * 100, 1
        ),
        "comparison": comparison,
        "all_mancini_levels": mancini_levels,
    }

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare Mancini Substack levels to engine")
    parser.add_argument("--date", default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    target = None
    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d").date()

    result = run_comparison(target)

    # Write output
    output_path = args.output or os.path.join(OUTPUT_DIR, "substack_comparison.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Print summary
    print("=" * 60)
    print("MANCINI SUBSTACK vs ENGINE COMPARISON")
    print("=" * 60)
    print(f"Post: {result.get('post_title', '?')}")
    print(f"Date: {result.get('post_date', '?')}")
    print()

    # Highlights
    highlights = result.get("highlights", [])
    if highlights:
        print("KEY HIGHLIGHTS FROM MANCINI:")
        print("-" * 40)
        for h in highlights:
            print(f"  [{h['type']}] {h['text'][:200]}")
        print()

    # Level comparison
    print(f"Mancini levels found: {result.get('mancini_levels_found', 0)}")
    print(f"Engine levels active:  {result.get('engine_levels_active', 0)}")
    print(f"Matched:               {result.get('matched_count', 0)}")
    print(f"Mancini-only:          {result.get('mancini_only_count', 0)}")
    print(f"Engine-only:           {result.get('engine_only_count', 0)}")
    print(f"Match rate:            {result.get('match_rate', 0)}%")
    print()

    comp = result.get("comparison", {})
    if comp.get("matched"):
        print("MATCHED LEVELS:")
        for m in comp["matched"]:
            print(f"  {m['mancini_price']:.0f} <-> {m['engine_price']:.2f} "
                  f"({m['engine_type']}, {m['engine_touches']} touches, "
                  f"dist={m['distance']:+.1f})")

    if comp.get("mancini_only"):
        print("\nMANCINI MENTIONED (engine missed):")
        for m in comp["mancini_only"][:15]:
            print(f"  {m['price']:.0f} ({m['role']}: {m.get('context', '')[:80]})")

    print(f"\nOutput written to: {output_path}")

    # Also dump structured levels for tomorrow's trading session so the
    # live runner can load an overlay at session start.
    try:
        from live.mancini_levels import dump_for_trading_date
        tomorrow = date.today() + timedelta(days=1)
        out = dump_for_trading_date(tomorrow, Path("/app/data"))
        if out is not None:
            print(f"Mancini levels overlay written to: {out}")
    except Exception as e:
        print(f"Failed to dump mancini levels: {e}")


if __name__ == "__main__":
    main()
