"""LLM-structured extraction of Mancini Substack post bodies.

Goes beyond regex price extraction (live/mancini_levels.py) to capture
the daily trade plan: directional lean, mode classification, planned
setups with conviction, danger zones, targets, no-trade zones, and
risk warnings. Output is consumed by the engine to inform signal
qualification, level conviction, and trade gates.

Two surfaces:
  - extract_plan(post_body, ...) — pure function, takes text, returns
    a ManciniPlan. Useful for tests and direct calls.
  - dump_plan_for_trading_date(trading_date, ...) — cron entry point.
    Fetches the latest Substack post, extracts a plan, writes JSON.
  - load_plan(trading_date, ...) — engine-side reader.

Cron usage on the VM:
    python3 live/mancini_llm_extract.py                     # plan for tomorrow
    python3 live/mancini_llm_extract.py --date 2026-05-06   # specific date
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))


_SYSTEM_PROMPT = """You extract structured trading plans from daily Substack posts by Adam Mancini, a futures trader who trades the ES (S&P 500 e-mini) using Failed Breakdown setups.

His vocabulary you must recognize:
- "Failed Breakdown" (FB): a long setup where price flushes below a significant low (PDL, multi-hour low, cluster), traps shorts, then recovers above. His core edge.
- "Level Reclaim": price reclaims a horizontal support/resistance level from below.
- "Elevator down": a sharp, sustained sell-off that flushes through multiple supports. Mancini says FBs are valid only after elevator-down sells.
- "Short squeeze": the bounce that follows a FB. The bigger the sell, the bigger the squeeze.
- "Mode 1 Green": an open-to-close trend day up — the explicit exception to "always wait for FB". On these days, ride the trend instead.
- "Danger zone": entries within 5 points above a swept low, where most FB losses occur. He warns against these.
- "Live direct" / "magnet": his highest-conviction levels.
- "Bull case" / "Bear case": conditional scenarios with explicit invalidation prices.

Extract STRICTLY what is in the post — do not infer levels from generic market commentary. Conviction guidance:
- "high" if Mancini explicitly says he plans to trade it ("the obvious trade is", "I'll be looking for", "ideal setup")
- "medium" for setups he discusses as plausible
- "low" for setups mentioned only in passing or as edge cases

For danger_zones, capture explicit no-trade or caution zones (e.g. "below 6838 = bear case begins", "danger zone is 5pts above the level"). For risk_warnings, capture explicit cautionary statements (e.g. "FBs near major highs are dangerous").

Set lean to "neutral" if no clear bias is stated. Set mode to null if not classifiable from the post.

NARRATIVE FIELDS — these are written for human subscribers to understand the market read at a glance. Keep them concise, in Mancini's voice, and grounded in the post:

- thesis_summary: 2-3 sentences capturing Mancini's CURRENT READ of the tape — what just happened, what he thinks the setup is, the central question of the day. Avoid level numbers here; this is the macro narrative. Empty string if the post has no clear thesis.
- bull_case: 1-2 sentences describing the conditional bullish scenario with the invalidation/trigger price. Format like: "Hold above 7528, sweep the FB lows and recover. Path up: 7600 → 7620 → 7649." Empty string if not present.
- bear_case: 1-2 sentences for the bearish scenario with trigger price. Format like: "Lose 7528 with conviction. First short: break of bounce ~7522 (low WR setup)." Empty string if not present.
- key_observations: 2-5 bulleted insights from the post that aren't already captured by levels/setups — things like "multiple breakouts have been traps this week", "6:40am low has held since Monday", "VIX still subdued despite the move". Each is one short sentence. Empty list if nothing notable beyond the setups themselves."""


class PlannedSetup(BaseModel):
    """A specific setup Mancini plans to take or named in his post."""

    setup_type: str = Field(
        description='One of: "failed_breakdown", "level_reclaim", "breakdown_short", "trend_continuation", "other"'
    )
    level_price: float = Field(description="The key level price for this setup")
    direction: str = Field(description='"long" or "short"')
    context: str = Field(
        description="Short human description, e.g. 'FB of yesterday's 6847 daily low'"
    )
    conviction: str = Field(description='"high", "medium", or "low"')


class DangerZone(BaseModel):
    """A price zone Mancini flags as dangerous or no-trade."""

    price_low: float
    price_high: Optional[float] = Field(
        default=None,
        description="Upper bound if zone is a range; null if single-sided",
    )
    rule: str = Field(description="The rule that defines the zone")


class ManciniPlan(BaseModel):
    """Structured trading plan extracted from a Mancini Substack post."""

    post_title: str = ""
    post_date: str = ""
    lean: str = Field(
        default="neutral",
        description='Directional bias: "bullish", "bearish", or "neutral"',
    )
    mode: Optional[str] = Field(
        default=None,
        description='Day mode: "mode_1_green", "range", "trending", or null',
    )
    planned_setups: list[PlannedSetup] = Field(default_factory=list)
    danger_zones: list[DangerZone] = Field(default_factory=list)
    targets: list[float] = Field(
        default_factory=list,
        description="Numeric upside or downside targets named for the session",
    )
    no_trade_above: Optional[float] = None
    no_trade_below: Optional[float] = None
    risk_warnings: list[str] = Field(default_factory=list)
    thesis_summary: str = Field(
        default="",
        description="2-3 sentences capturing Mancini's macro read for the day",
    )
    bull_case: str = Field(
        default="",
        description="Conditional bullish scenario with trigger/invalidation price",
    )
    bear_case: str = Field(
        default="",
        description="Conditional bearish scenario with trigger/invalidation price",
    )
    key_observations: list[str] = Field(
        default_factory=list,
        description="2-5 short insights from the post beyond what levels/setups capture",
    )
    raw_extraction_metadata: dict = Field(default_factory=dict)


class ManciniExtractionError(Exception):
    """Raised when the LLM extraction fails or returns unparseable output."""


def extract_plan(
    post_body: str,
    post_title: str = "",
    post_date: str = "",
    model: str = "claude-opus-4-7",
    api_key: Optional[str] = None,
) -> ManciniPlan:
    """Extract a structured ManciniPlan from a Substack post body.

    The Anthropic SDK is lazy-imported so unit tests can monkeypatch the
    client without the package installed at import time. The system
    prompt is cached (ephemeral) — the post format is stable across all
    daily posts, so the prefix is a strong cache candidate.

    Parameters
    ----------
    post_body : str
        Plain-text body of the post.
    post_title : str
        Optional title to thread through to the output.
    post_date : str
        Optional ISO date for the post.
    model : str
        Anthropic model id. Defaults to Opus 4.7 — the daily extraction
        is a leveraged decision input, so prefer accuracy. Swap to
        ``claude-haiku-4-5`` if cost dominates and accuracy holds up.
    api_key : str, optional
        Override ``ANTHROPIC_API_KEY`` env var.

    Raises
    ------
    ManciniExtractionError
        If the SDK is unavailable, the API call fails, or the model
        returns output that doesn't match the schema.
    """
    try:
        import anthropic
    except ImportError as e:
        raise ManciniExtractionError(f"anthropic SDK not installed: {e}") from e

    client = (
        anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    )

    user_message = (
        f"Post title: {post_title}\n"
        f"Post date: {post_date}\n\n"
        f"--- POST BODY ---\n{post_body}"
    )

    # Note: we use messages.create() (not messages.parse) because the
    # full Pydantic schema for ManciniPlan with its 4 nested classes,
    # 13 fields, multiple lists, and optional fields exceeds the
    # Anthropic grammar compiler limit and returns
    # "Grammar compilation timed out". Plain JSON instruction +
    # model_validate sidesteps the grammar engine entirely.
    json_schema_hint = json.dumps(
        ManciniPlan.model_json_schema(), separators=(",", ":")
    )
    json_instruction = (
        "\n\nReturn ONLY a single JSON object that validates against this "
        f"schema (no prose, no markdown fences):\n{json_schema_hint}\n"
        "Use empty strings for missing string fields and empty lists for "
        "missing list fields."
    )

    t0 = time.monotonic()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT + json_instruction,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:
        raise ManciniExtractionError(f"API call failed: {e}") from e

    latency_ms = round((time.monotonic() - t0) * 1000, 1)

    # Pull text out of the response and parse as JSON. The model
    # occasionally wraps in a ```json fence — strip if present.
    text = ""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", "") == "text":
            text += getattr(block, "text", "")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        plan_dict = json.loads(text)
    except json.JSONDecodeError as e:
        raise ManciniExtractionError(
            f"Model response was not valid JSON: {e}; first 300 chars: {text[:300]}"
        ) from e

    try:
        plan = ManciniPlan.model_validate(plan_dict)
    except Exception as e:
        raise ManciniExtractionError(
            f"Model response failed schema validation: {e}"
        ) from e

    if not plan.post_title:
        plan.post_title = post_title
    if not plan.post_date:
        plan.post_date = post_date

    usage = getattr(response, "usage", None)
    plan.raw_extraction_metadata = {
        "model": model,
        "latency_ms": latency_ms,
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "cache_creation_input_tokens": (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )
        if usage
        else 0,
        "cache_read_input_tokens": (
            getattr(usage, "cache_read_input_tokens", 0) or 0
        )
        if usage
        else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
    }

    logger.info(
        f"Mancini plan extracted: lean={plan.lean} mode={plan.mode} "
        f"setups={len(plan.planned_setups)} danger={len(plan.danger_zones)} "
        f"targets={len(plan.targets)} warnings={len(plan.risk_warnings)} "
        f"obs={len(plan.key_observations)} thesis={'y' if plan.thesis_summary else 'n'} "
        f"bull={'y' if plan.bull_case else 'n'} bear={'y' if plan.bear_case else 'n'} "
        f"latency_ms={latency_ms}"
    )

    return plan


# ---------------------------------------------------------------------------
# Cron entry point + engine-side reader
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}

# "<Month> <day>[/| and <day2>] [Trade ]Plan" — e.g. "June 12th Plan",
# "July 3rd/4th Plan", "Feb 17th and 18th Plan", "Dec 24/26 Trade Plan".
_TITLE_PLAN_DATE_RE = re.compile(
    r"\b([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:(?:\s*/\s*|\s+and\s+)(\d{1,2})(?:st|nd|rd|th)?)?"
    r"\s+(?:Trade\s+)?Plan\b",
    re.IGNORECASE,
)

# Reversed form: "Plan for November 6th."
_TITLE_PLAN_FOR_RE = re.compile(
    r"\bPlan\s+for\s+([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)


def parse_plan_date_from_title(title: str, reference: date) -> Optional[date]:
    """Parse the trading date out of a Mancini post title.

    Titles end in "<Month> <day>[st|nd|rd|th] Plan" (e.g. "June 12th
    Plan"), with holiday variants covering two sessions ("July 3rd/4th
    Plan", "Dec 24/26 Trade Plan") and an occasional reversed "Plan for
    November 6th" form. The year is inferred as whichever candidate lands
    closest to `reference` (handles the December/January wrap); among the
    days of a multi-session title, the one closest to `reference` wins.
    Returns None when no such pattern exists.
    """
    if not title:
        return None
    m = _TITLE_PLAN_DATE_RE.search(title)
    days: list[int] = []
    month = None
    if m:
        month = _MONTHS.get(m.group(1).lower())
        days = [int(m.group(2))]
        if m.group(3):
            days.append(int(m.group(3)))
    if month is None:
        m = _TITLE_PLAN_FOR_RE.search(title)
        if not m:
            return None
        month = _MONTHS.get(m.group(1).lower())
        if month is None:
            return None
        days = [int(m.group(2))]
    candidates = []
    for day in days:
        for year in (reference.year - 1, reference.year, reference.year + 1):
            try:
                candidates.append(date(year, month, day))
            except ValueError:
                continue
    if not candidates:
        return None
    return min(candidates, key=lambda d: abs((d - reference).days))


def next_trading_date(today: date) -> date:
    """The next ES trading date after `today`, skipping Sat/Sun.

    Friday evening's post is the "Monday Plan" (covers the weekend), so a
    cron run on Friday or Saturday must target Monday, not the calendar
    tomorrow.
    """
    d = today + timedelta(days=1)
    while d.weekday() > 4:  # 5=Sat, 6=Sun
        d += timedelta(days=1)
    return d


def post_matches_trading_date(
    post_title: str,
    post_date_str: str,
    trading_date: date,
) -> tuple[bool, str]:
    """Does this post actually describe `trading_date`'s session?

    The title's "<Month> <day> Plan" date is authoritative. Without one,
    fall back to the publish date: a valid post is published the prior
    calendar day, or the prior Fri/Sat/Sun for a Monday session. With no
    parsable metadata at all, accept — validation only blocks
    provably-wrong posts; it must not brick the pipeline on a metadata
    regression.
    """
    title_date = parse_plan_date_from_title(post_title, reference=trading_date)
    if title_date is not None:
        if title_date == trading_date:
            return True, f"title plan date {title_date} matches"
        return False, (
            f"title plan date {title_date.isoformat()} != trading_date "
            f"{trading_date.isoformat()} (title={post_title!r})"
        )
    try:
        post_d = date.fromisoformat(str(post_date_str)[:10])
    except ValueError:
        return True, "no parsable title or post date — accepting"
    delta = (trading_date - post_d).days
    if delta == 1 or (trading_date.weekday() == 0 and 1 <= delta <= 3):
        return True, f"post_date {post_d} is the session-eve of {trading_date}"
    return False, (
        f"post_date {post_d.isoformat()} is not the session-eve of "
        f"trading_date {trading_date.isoformat()}"
    )


def _existing_ok_plan(output_path: Path) -> bool:
    """True when output_path already holds a healthy extracted plan."""
    try:
        if not output_path.exists():
            return False
        data = json.loads(output_path.read_text())
        return (
            isinstance(data, dict)
            and data.get("schema_version") == 1
            and data.get("extract_status") == "ok"
            and bool(data.get("plan"))
        )
    except Exception:
        return False


def _write_stub_preserving_ok(output_path: Path, payload: dict) -> None:
    """Write a degraded stub unless a good plan is already on disk.

    The 17:00 primary can succeed and the 20:00 backup fail (expired
    cookie, late re-fetch of yesterday's post) — the failure must never
    clobber the good plan.
    """
    if _existing_ok_plan(output_path):
        logger.info(
            f"Mancini plan: keeping existing OK plan at {output_path.name}; "
            f"not overwriting with {payload.get('extract_status')} stub"
        )
        return
    output_path.write_text(json.dumps(payload, indent=2, default=str))


def _stub_payload(trading_date: date, status: str,
                  post_title: str = "", post_date: str = "",
                  error: str = "") -> dict:
    """Schema-versioned stub written when extraction fails — keeps load_plan
    able to reason about the failure without the engine crashing.
    """
    return {
        "schema_version": 1,
        "trading_date": trading_date.isoformat(),
        "post_date": post_date,
        "post_title": post_title,
        "fetched_at": datetime.now().isoformat(),
        "extract_status": status,
        "plan": None,
        "error": error,
    }


def dump_plan_for_trading_date(
    trading_date: date,
    output_dir: Path | None = None,
    model: str = "claude-opus-4-7",
) -> Path | None:
    """Cron entry point: fetch the latest Substack post, run LLM extraction,
    write `mancini_plan_<trading_date>.json`.

    Never raises — any failure writes a degraded-stub JSON so the engine's
    load_plan can return None cleanly. Returns the path to the written
    file, or None if writing itself failed.

    Outputs are written under the same directory as `mancini_levels_*.json`,
    so a single mount is enough on the VM.
    """
    output_dir = output_dir or Path("/app/data")
    output_path = output_dir / f"mancini_plan_{trading_date.isoformat()}.json"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Mancini plan: cannot create output dir: {e}")
        return None

    # Fetch the post via the existing scraper. Reuse its body extractor.
    try:
        from live.substack_compare import fetch_latest_post
        from live.mancini_levels import _get_body_text
    except Exception as e:
        logger.error(f"Mancini plan: failed to import scraper helpers: {e}")
        _write_stub_preserving_ok(
            output_path,
            _stub_payload(trading_date, "import_failed", error=str(e)),
        )
        return output_path

    try:
        post = fetch_latest_post()
    except Exception as e:
        logger.error(f"Mancini plan: fetch_latest_post raised: {e}")
        _write_stub_preserving_ok(
            output_path,
            _stub_payload(trading_date, "fetch_failed", error=str(e)),
        )
        return output_path

    if not post:
        logger.warning("Mancini plan: no post fetched (auth or upstream issue)")
        _write_stub_preserving_ok(
            output_path,
            _stub_payload(trading_date, "no_post"),
        )
        return output_path

    body_text = _get_body_text(post)
    post_title = str(post.get("title", ""))
    post_date_str = str(post.get("post_date", post.get("date", "")))[:10]

    matches, reason = post_matches_trading_date(
        post_title, post_date_str, trading_date)
    if not matches:
        logger.warning(
            f"Mancini plan: latest post does not describe "
            f"{trading_date.isoformat()} — {reason}. Skipping extraction; "
            f"a later cron run will retry once the post is published."
        )
        _write_stub_preserving_ok(
            output_path,
            _stub_payload(trading_date, "stale_post",
                          post_title=post_title, post_date=post_date_str,
                          error=reason),
        )
        return output_path

    if not body_text:
        logger.warning("Mancini plan: post body empty after extraction")
        _write_stub_preserving_ok(
            output_path,
            _stub_payload(trading_date, "empty_body",
                          post_title=post_title, post_date=post_date_str),
        )
        return output_path

    try:
        plan = extract_plan(
            post_body=body_text,
            post_title=post_title,
            post_date=post_date_str,
            model=model,
        )
    except ManciniExtractionError as e:
        logger.error(f"Mancini plan: extraction failed: {e}")
        _write_stub_preserving_ok(
            output_path,
            _stub_payload(trading_date, "extract_failed",
                          post_title=post_title, post_date=post_date_str,
                          error=str(e)),
        )
        return output_path

    payload = {
        "schema_version": 1,
        "trading_date": trading_date.isoformat(),
        "post_date": post_date_str,
        "post_title": post_title,
        "fetched_at": datetime.now().isoformat(),
        "extract_status": "ok",
        "plan": plan.model_dump(),
        "error": "",
    }

    output_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(
        f"Mancini plan: wrote {output_path.name} "
        f"(lean={plan.lean} mode={plan.mode} setups={len(plan.planned_setups)} "
        f"danger={len(plan.danger_zones)} targets={len(plan.targets)})"
    )
    return output_path


def load_plan(
    trading_date: date,
    input_dir: Path | None = None,
) -> ManciniPlan | None:
    """Engine-side reader. Returns the validated ManciniPlan for a trading
    date, or None when the file is missing / corrupt / a degraded stub.
    Never raises.
    """
    try:
        input_dir = input_dir or Path("/app/data")
        path = input_dir / f"mancini_plan_{trading_date.isoformat()}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Mancini plan: corrupt JSON at {path}: {e}")
            return None
        if not isinstance(data, dict):
            logger.warning(f"Mancini plan: unexpected top-level type at {path}")
            return None
        if data.get("schema_version") != 1:
            logger.warning(f"Mancini plan: unexpected schema_version in {path}")
            return None
        if data.get("extract_status") != "ok" or not data.get("plan"):
            return None
        try:
            return ManciniPlan.model_validate(data["plan"])
        except Exception as e:
            logger.warning(f"Mancini plan: schema validation failed at {path}: {e}")
            return None
    except Exception as e:
        logger.warning(f"Mancini plan load failed: {e}")
        return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Dump structured Mancini Substack trade plan to JSON",
    )
    parser.add_argument(
        "--date",
        default="tomorrow",
        help="Trading date (YYYY-MM-DD) or 'tomorrow' (default: the next "
             "trading day in ET, skipping weekends — Friday targets Monday)",
    )
    parser.add_argument(
        "--output-dir",
        default="/app/data",
        help="Output directory (default: /app/data)",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="Anthropic model id (default: claude-opus-4-7)",
    )
    args = parser.parse_args()

    if args.date == "tomorrow":
        try:
            from zoneinfo import ZoneInfo
            today_et = datetime.now(ZoneInfo("US/Eastern")).date()
        except Exception:
            today_et = date.today()  # container clock is ET
        target = next_trading_date(today_et)
    else:
        target = date.fromisoformat(args.date)

    dump_plan_for_trading_date(target, Path(args.output_dir), model=args.model)


if __name__ == "__main__":
    main()
