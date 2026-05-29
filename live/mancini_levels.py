"""Parse Mancini's Substack post into structured levels for engine overlay.

Builds on top of the existing parsing in live/substack_compare.py.

The parser runs from a cron job the evening before a trading session. Output
is a JSON file at {output_dir}/mancini_levels_{trading_date}.json that the
IB runner loads at session start to augment engine-detected levels.

Usage:
    python3 live/mancini_levels.py                       # dump for tomorrow
    python3 live/mancini_levels.py --date 2026-04-16     # specific date
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from loguru import logger

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse existing parsing primitives
from live.substack_compare import (  # noqa: E402
    fetch_latest_post,
    extract_levels_from_text,
    extract_highlights,
    extract_body_html,
    SUBSTACK_BASE,
)


def _get_body_text(post: dict) -> str:
    """Extract clean body text from a post dict returned by fetch_latest_post."""
    # Live API: body_html_clean is already stripped
    clean = post.get("body_html_clean")
    if clean:
        return clean
    # Page-scrape fallback: raw HTML page stored in 'text'
    raw = post.get("text") or post.get("body_html") or post.get("body") or ""
    if not raw:
        return ""
    # If this is a raw page scrape, try the dedicated extractor
    try:
        return extract_body_html(raw)
    except Exception:
        # Last-resort: strip HTML tags and collapse whitespace
        stripped = re.sub(r"<[^>]+>", " ", raw)
        return re.sub(r"\s+", " ", stripped).strip()


def _classify_level(raw: dict, lean: str) -> dict | None:
    """Enrich a raw level dict with tags, conviction, side, and biases.

    Returns None if the level fails sanity checks.
    """
    price = raw.get("price")
    if not isinstance(price, (int, float)):
        return None
    if price < 1000 or price > 20000:
        return None

    context = (raw.get("context") or "").lower()
    role = raw.get("role") or "level"

    tags: list[str] = []
    if any(w in context for w in ("trap", "watch", "risky", "dangerous", "careful", "caution")):
        tags.append("caution")
    if any(w in context for w in ("magnet", "live direct", "pivot")):
        tags.append("magnet")
    if any(w in context for w in ("key", "major", "big ", "huge", "massive", "critical", "important")):
        tags.append("key")
    if "bear" in context:
        tags.append("bear_case")
    if "bull" in context:
        tags.append("bull_case")

    # Conviction scoring (1-3)
    conviction = 1
    if "key" in tags or "magnet" in tags:
        conviction += 1
    if role in ("support", "resistance"):
        conviction += 1
    conviction = min(3, conviction)

    # Side determination
    if role == "support":
        side = "support"
    elif role == "resistance":
        side = "resistance"
    elif "bear_case" in tags:
        side = "resistance"
    elif "bull_case" in tags:
        side = "support"
    else:
        side = "either"

    long_bias = side in ("support", "either") and "bear_case" not in tags
    short_bias = side in ("resistance", "either") and "bull_case" not in tags

    return {
        "price": float(price),
        "side": side,
        "role": role,
        "tags": tags,
        "conviction": conviction,
        "long_bias": long_bias,
        "short_bias": short_bias,
        "caution": "caution" in tags,
        "context_snippet": (raw.get("context") or "")[:200],
    }


def _extract_lean(highlights: list[dict]) -> str:
    """Pull directional lean out of the highlights list."""
    for h in highlights:
        if h.get("type") == "DIRECTIONAL LEAN":
            txt = (h.get("text") or "").lower()
            if "bull" in txt:
                return "bullish"
            if "bear" in txt:
                return "bearish"
    return "neutral"


def dump_for_trading_date(trading_date: date, output_dir: Path | None = None) -> Path | None:
    """Fetch Mancini's latest post, parse into levels, write JSON.

    Returns the path to the written file, or None on failure.
    Never raises — safe to call from cron.
    """
    try:
        output_dir = output_dir or Path("/app/data")
        output_dir.mkdir(parents=True, exist_ok=True)

        post = fetch_latest_post()
        if not post:
            logger.warning("Mancini levels: no post fetched")
            # Still write a degraded stub so downstream load() can reason about it
            stub = {
                "schema_version": 1,
                "trading_date": trading_date.isoformat(),
                "post_date": "",
                "post_title": "",
                "fetched_at": datetime.now().isoformat(),
                "lean": "neutral",
                "parse_status": "failed",
                "levels": [],
                "highlights": [],
            }
            stub_path = output_dir / f"mancini_levels_{trading_date.isoformat()}.json"
            stub_path.write_text(json.dumps(stub, indent=2, default=str))
            return None

        body_text = _get_body_text(post)
        if not body_text:
            logger.warning("Mancini levels: empty post body")

        raw_levels = extract_levels_from_text(body_text) if body_text else []
        highlights = extract_highlights(body_text) if body_text else []
        lean = _extract_lean(highlights)

        enriched: list[dict] = []
        for raw in raw_levels:
            lv = _classify_level(raw, lean)
            if lv is not None:
                enriched.append(lv)

        # Deduplicate by price (keep highest conviction)
        by_price: dict[float, dict] = {}
        for lv in enriched:
            key = round(lv["price"], 2)
            existing = by_price.get(key)
            if existing is None or lv["conviction"] > existing["conviction"]:
                by_price[key] = lv
        levels = sorted(by_price.values(), key=lambda x: x["price"])

        parse_status = "ok" if levels else "degraded"

        result = {
            "schema_version": 1,
            "trading_date": trading_date.isoformat(),
            "post_date": str(post.get("post_date", post.get("date", "")))[:10],
            "post_title": post.get("title", ""),
            "fetched_at": datetime.now().isoformat(),
            "lean": lean,
            "parse_status": parse_status,
            "levels": levels,
            "highlights": highlights[:20],
        }

        output_path = output_dir / f"mancini_levels_{trading_date.isoformat()}.json"
        output_path.write_text(json.dumps(result, indent=2, default=str))
        logger.info(
            f"Mancini levels: wrote {len(levels)} levels for {trading_date} "
            f"(lean={lean}, status={parse_status}) to {output_path}"
        )
        return output_path
    except Exception as e:
        logger.error(f"Mancini levels dump failed: {e}")
        return None


def load(trading_date: date, input_dir: Path | None = None) -> dict | None:
    """Load parsed Mancini levels for a trading date.

    Returns None if the file is missing, unreadable, has an unexpected schema,
    or represents a failed parse.
    """
    try:
        input_dir = input_dir or Path("/app/data")
        path = input_dir / f"mancini_levels_{trading_date.isoformat()}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Mancini levels: corrupt JSON at {path}: {e}")
            return None
        if not isinstance(data, dict):
            logger.warning(f"Mancini levels: unexpected top-level type at {path}")
            return None
        if data.get("schema_version") != 1:
            logger.warning(f"Mancini levels: unexpected schema_version in {path}")
            return None
        if data.get("parse_status") == "failed":
            return None
        return data
    except Exception as e:
        logger.warning(f"Mancini levels load failed: {e}")
        return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dump Mancini Substack levels to JSON")
    parser.add_argument("--date", default="tomorrow",
                        help="trading date YYYY-MM-DD, or 'tomorrow' (default)")
    parser.add_argument("--output-dir", default="/app/data",
                        help="Output directory for the JSON file")
    args = parser.parse_args()

    if args.date == "tomorrow":
        target = date.today() + timedelta(days=1)
    else:
        target = date.fromisoformat(args.date)

    dump_for_trading_date(target, Path(args.output_dir))


if __name__ == "__main__":
    main()
