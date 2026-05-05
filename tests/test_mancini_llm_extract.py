"""Tests for live/mancini_llm_extract.py.

Default test path makes ZERO live API calls — the Anthropic client
is monkeypatched in every test that exercises extract_plan().

To run a real-API smoke test against today's Mancini post (paid call,
needs ANTHROPIC_API_KEY):

    MANCINI_LLM_LIVE=1 python3 -m pytest tests/test_mancini_llm_extract.py::test_live_smoke -v
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from live.mancini_llm_extract import (
    DangerZone,
    ManciniExtractionError,
    ManciniPlan,
    PlannedSetup,
    extract_plan,
)


def _mock_response(plan: ManciniPlan,
                   input_tokens: int = 100,
                   cache_read: int = 0,
                   cache_creation: int = 50,
                   output_tokens: int = 30) -> SimpleNamespace:
    """Build a stub object shaped like an Anthropic messages.parse() response."""
    return SimpleNamespace(
        parsed_output=plan,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            output_tokens=output_tokens,
        ),
    )


def _install_fake_anthropic(monkeypatch, parse_callable):
    """Replace anthropic.Anthropic with a fake client whose
    messages.parse delegates to parse_callable(**kwargs).
    """
    import anthropic

    class _FakeMessages:
        def __init__(self):
            self.parse = parse_callable

    class _FakeClient:
        def __init__(self, *_, **__):
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_schema_round_trip():
    """ManciniPlan dump/load is identity for a representative payload."""
    plan = ManciniPlan(
        post_title="May 5th Plan",
        post_date="2026-05-04",
        lean="bullish",
        mode="mode_1_green",
        planned_setups=[
            PlannedSetup(
                setup_type="failed_breakdown",
                level_price=7137.0,
                direction="long",
                context="FB of yesterday's 7137 daily low",
                conviction="high",
            )
        ],
        danger_zones=[DangerZone(price_low=7240.0, rule="below 7240 = bear case")],
        targets=[7253.0, 7267.0, 7297.0],
        no_trade_below=7100.0,
        risk_warnings=["FBs near major highs are dangerous"],
    )
    dumped = plan.model_dump()
    rebuilt = ManciniPlan.model_validate(dumped)
    assert rebuilt.model_dump() == dumped


def test_schema_defaults_are_safe():
    """Empty construction yields neutral, no-op plan."""
    plan = ManciniPlan()
    assert plan.lean == "neutral"
    assert plan.mode is None
    assert plan.planned_setups == []
    assert plan.danger_zones == []
    assert plan.targets == []
    assert plan.no_trade_above is None
    assert plan.no_trade_below is None
    assert plan.risk_warnings == []


# ---------------------------------------------------------------------------
# Mocked happy-path tests
# ---------------------------------------------------------------------------


def test_extract_plan_returns_parsed_output(monkeypatch):
    """A normal API call returns the parsed ManciniPlan."""
    expected = ManciniPlan(
        lean="bullish",
        mode="mode_1_green",
        planned_setups=[
            PlannedSetup(
                setup_type="failed_breakdown",
                level_price=7137.0,
                direction="long",
                context="FB of 7137",
                conviction="high",
            )
        ],
        targets=[7253.0, 7267.0],
    )

    def fake_parse(**kwargs):
        return _mock_response(expected, input_tokens=2000, cache_read=1500,
                              cache_creation=0, output_tokens=120)

    _install_fake_anthropic(monkeypatch, fake_parse)

    plan = extract_plan(
        post_body="Today we held 7198 at 12:10PM...",
        post_title="May 5th Plan",
        post_date="2026-05-04",
    )

    assert plan.lean == "bullish"
    assert plan.mode == "mode_1_green"
    assert len(plan.planned_setups) == 1
    assert plan.planned_setups[0].level_price == 7137.0
    assert plan.targets == [7253.0, 7267.0]


def test_extract_plan_threads_through_post_metadata(monkeypatch):
    """If the model leaves post_title/date empty, caller-supplied values fill in."""
    bare_plan = ManciniPlan(lean="neutral")  # title and date not set by model

    _install_fake_anthropic(monkeypatch, lambda **kw: _mock_response(bare_plan))

    plan = extract_plan(
        post_body="...",
        post_title="May 5th Plan",
        post_date="2026-05-04",
    )

    assert plan.post_title == "May 5th Plan"
    assert plan.post_date == "2026-05-04"


def test_extract_plan_records_metadata(monkeypatch):
    """raw_extraction_metadata captures model, tokens, and latency."""
    plan_obj = ManciniPlan(lean="bearish")

    _install_fake_anthropic(
        monkeypatch,
        lambda **kw: _mock_response(plan_obj, input_tokens=1000, cache_read=900,
                                    cache_creation=0, output_tokens=200),
    )

    plan = extract_plan(post_body="...", model="claude-haiku-4-5")
    md = plan.raw_extraction_metadata

    assert md["model"] == "claude-haiku-4-5"
    assert md["input_tokens"] == 1000
    assert md["cache_read_input_tokens"] == 900
    assert md["output_tokens"] == 200
    assert md["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# Failure-mode tests
# ---------------------------------------------------------------------------


def test_extract_plan_raises_when_no_parsed_output(monkeypatch):
    """If parsed_output is None (validation failed), raise ManciniExtractionError."""
    failed_response = SimpleNamespace(
        parsed_output=None,
        usage=SimpleNamespace(
            input_tokens=10, cache_creation_input_tokens=0,
            cache_read_input_tokens=0, output_tokens=0,
        ),
    )
    _install_fake_anthropic(monkeypatch, lambda **kw: failed_response)

    with pytest.raises(ManciniExtractionError, match="parsed_output"):
        extract_plan(post_body="...")


def test_extract_plan_raises_when_api_call_fails(monkeypatch):
    """Any exception from the API call is wrapped in ManciniExtractionError."""

    def raising_parse(**kwargs):
        raise RuntimeError("connection reset")

    _install_fake_anthropic(monkeypatch, raising_parse)

    with pytest.raises(ManciniExtractionError, match="API call failed"):
        extract_plan(post_body="...")


# ---------------------------------------------------------------------------
# Cache-control assertion
# ---------------------------------------------------------------------------


def test_system_prompt_has_cache_control(monkeypatch):
    """The system block must carry cache_control: ephemeral so the prefix
    is cacheable across daily posts."""
    captured = {}

    def fake_parse(**kwargs):
        captured.update(kwargs)
        return _mock_response(ManciniPlan(lean="neutral"))

    _install_fake_anthropic(monkeypatch, fake_parse)
    extract_plan(post_body="anything", post_title="t", post_date="2026-05-04")

    system = captured.get("system")
    assert isinstance(system, list) and system, (
        "system must be a non-empty list of content blocks"
    )
    assert any(
        isinstance(b, dict) and b.get("cache_control") == {"type": "ephemeral"}
        for b in system
    ), f"no ephemeral cache_control on system blocks: {system!r}"


def test_user_message_includes_post_body(monkeypatch):
    """The post body is forwarded into the user message."""
    captured = {}

    def fake_parse(**kwargs):
        captured.update(kwargs)
        return _mock_response(ManciniPlan(lean="neutral"))

    _install_fake_anthropic(monkeypatch, fake_parse)
    extract_plan(post_body="DIAGNOSTIC_TOKEN_XYZ", post_title="t", post_date="d")

    messages = captured["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "DIAGNOSTIC_TOKEN_XYZ" in messages[0]["content"]


# ---------------------------------------------------------------------------
# Live smoke test (skipped by default — requires real API key)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MANCINI_LLM_LIVE"),
    reason="set MANCINI_LLM_LIVE=1 (and ANTHROPIC_API_KEY) to hit the real API",
)
def test_live_smoke():
    """One real API call against a tiny stub post — verifies wiring end-to-end."""
    plan = extract_plan(
        post_body=(
            "I wrote on Friday at 4pm: 'My general lean is to defer to the trend.' "
            "Bull case Monday: ES is now rangebound 6925-6849 with 6894 and 6870 "
            "big pivots inside it. Bear case begins below 6838. The obvious trade "
            "tomorrow is the Failed Breakdown of yesterday's 6849 daily low. "
            "Targets are 6884, 6898, 6913. Remember 5 points above the significant "
            "low is the danger zone for Failed Breakdowns and where most losses occur."
        ),
        post_title="Smoke Test Stub",
        post_date="2026-05-04",
    )

    assert plan.lean in {"bullish", "bearish", "neutral"}
    assert isinstance(plan.planned_setups, list)
    # Don't assert exact contents — model output varies; just verify wiring.
    assert plan.raw_extraction_metadata.get("model")
