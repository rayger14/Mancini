"""LLM-structured extraction of Mancini Substack post bodies.

Goes beyond regex price extraction (live/mancini_levels.py) to capture
the daily trade plan: directional lean, mode classification, planned
setups with conviction, danger zones, targets, no-trade zones, and
risk warnings. Output is consumed by the engine to inform signal
qualification, level conviction, and trade gates.

Pure module — not yet wired into any cron/runner.

Usage:
    from live.mancini_llm_extract import extract_plan
    plan = extract_plan(post_body="...", post_title="...", post_date="2026-05-04")
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field


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

Set lean to "neutral" if no clear bias is stated. Set mode to null if not classifiable from the post."""


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

    t0 = time.monotonic()
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
            output_format=ManciniPlan,
        )
    except ManciniExtractionError:
        raise
    except Exception as e:
        raise ManciniExtractionError(f"API call failed: {e}") from e

    latency_ms = round((time.monotonic() - t0) * 1000, 1)

    plan = getattr(response, "parsed_output", None)
    if plan is None:
        raise ManciniExtractionError(
            "Model response had no parsed_output — schema validation failed"
        )

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
        f"latency_ms={latency_ms}"
    )

    return plan
