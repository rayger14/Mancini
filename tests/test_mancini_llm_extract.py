"""Tests for live/mancini_llm_extract.py.

Default test path makes ZERO live API calls — the Anthropic client
is monkeypatched in every test that exercises extract_plan().

To run a real-API smoke test against today's Mancini post (paid call,
needs ANTHROPIC_API_KEY):

    MANCINI_LLM_LIVE=1 python3 -m pytest tests/test_mancini_llm_extract.py::test_live_smoke -v
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import live.mancini_llm_extract as mle
from live.mancini_llm_extract import (
    DangerZone,
    ManciniExtractionError,
    ManciniPlan,
    PlannedSetup,
    dump_plan_for_trading_date,
    extract_plan,
    load_plan,
)


def _mock_response(plan: ManciniPlan | None,
                   input_tokens: int = 100,
                   cache_read: int = 0,
                   cache_creation: int = 50,
                   output_tokens: int = 30,
                   raw_text: str | None = None) -> SimpleNamespace:
    """Build a stub shaped like an Anthropic messages.create() response.

    The response carries a single text block whose payload is the JSON
    serialisation of `plan`. Pass `raw_text` to inject an arbitrary
    string (used for the JSON-parse-failure test).
    """
    if raw_text is None:
        text = plan.model_dump_json() if plan is not None else "{}"
    else:
        text = raw_text
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            output_tokens=output_tokens,
        ),
    )


def _install_fake_anthropic(monkeypatch, create_callable):
    """Replace anthropic.Anthropic with a fake client whose
    messages.create delegates to create_callable(**kwargs).
    """
    import anthropic

    class _FakeMessages:
        def __init__(self):
            self.create = create_callable

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


def test_extract_plan_raises_when_response_is_not_json(monkeypatch):
    """If the model returns prose instead of JSON, raise ManciniExtractionError."""
    bad_text_response = _mock_response(None, raw_text="sorry, can't help with that")
    _install_fake_anthropic(monkeypatch, lambda **kw: bad_text_response)

    with pytest.raises(ManciniExtractionError, match="not valid JSON"):
        extract_plan(post_body="...")


def test_extract_plan_raises_when_json_fails_schema_validation(monkeypatch):
    """Valid JSON but missing required structure → schema validation error."""
    # `lean` must be a string per the schema — int fails Pydantic.
    bad_schema_response = _mock_response(None, raw_text='{"lean": 12345}')
    _install_fake_anthropic(monkeypatch, lambda **kw: bad_schema_response)

    with pytest.raises(ManciniExtractionError, match="failed schema validation"):
        extract_plan(post_body="...")


def test_extract_plan_strips_markdown_fences(monkeypatch):
    """Models sometimes wrap JSON in ```json fences — we strip them."""
    fenced = '```json\n{"lean": "bullish"}\n```'
    _install_fake_anthropic(
        monkeypatch, lambda **kw: _mock_response(None, raw_text=fenced)
    )
    plan = extract_plan(post_body="...")
    assert plan.lean == "bullish"


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


# ---------------------------------------------------------------------------
# dump_plan_for_trading_date — cron entry point
# ---------------------------------------------------------------------------


def _patch_fetch_and_extract(monkeypatch, post: dict | None,
                             extracted: ManciniPlan | None,
                             extract_raises: Exception | None = None):
    """Stub fetch_latest_post + extract_plan inside live.mancini_llm_extract
    so tests run without network or API.

    The dump_plan_for_trading_date function imports both lazily inside its
    body via `from live.substack_compare import fetch_latest_post` and
    `from live.mancini_levels import _get_body_text`. We patch those source
    modules directly so the import statements pick up our stubs.
    """
    import live.substack_compare as sc
    import live.mancini_levels as ml

    monkeypatch.setattr(sc, "fetch_latest_post", lambda: post)
    monkeypatch.setattr(ml, "_get_body_text",
                        lambda p: (p or {}).get("body_html_clean", "") or (p or {}).get("text", ""))

    def _stub_extract(post_body, post_title="", post_date="", model="claude-opus-4-7", api_key=None):
        if extract_raises is not None:
            raise extract_raises
        if extracted is None:
            raise ManciniExtractionError("test forced failure")
        plan = extracted.model_copy()
        if not plan.post_title:
            plan.post_title = post_title
        if not plan.post_date:
            plan.post_date = post_date
        return plan

    monkeypatch.setattr(mle, "extract_plan", _stub_extract)


def test_dump_writes_ok_payload_on_success(monkeypatch, tmp_path):
    """Happy path: post fetched, extracted plan written with extract_status=ok."""
    post = {
        "title": "May 5th Plan",
        "post_date": "2026-05-04",
        "body_html_clean": "Today we held 7198 at 12:10PM. ...",
        "source": "live_api",
    }
    plan = ManciniPlan(
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
    _patch_fetch_and_extract(monkeypatch, post, plan)

    out = dump_plan_for_trading_date(date(2026, 5, 6), output_dir=tmp_path)
    assert out is not None
    payload = json.loads(out.read_text())

    assert payload["schema_version"] == 1
    assert payload["trading_date"] == "2026-05-06"
    assert payload["extract_status"] == "ok"
    assert payload["post_title"] == "May 5th Plan"
    assert payload["post_date"] == "2026-05-04"
    assert payload["plan"]["lean"] == "bullish"
    assert payload["plan"]["mode"] == "mode_1_green"
    assert len(payload["plan"]["planned_setups"]) == 1
    assert payload["error"] == ""


def test_dump_writes_no_post_stub(monkeypatch, tmp_path):
    """Auth failure / cookie expired: write a degraded stub, don't crash."""
    _patch_fetch_and_extract(monkeypatch, post=None, extracted=None)

    out = dump_plan_for_trading_date(date(2026, 5, 6), output_dir=tmp_path)
    assert out is not None
    payload = json.loads(out.read_text())

    assert payload["extract_status"] == "no_post"
    assert payload["plan"] is None
    assert payload["schema_version"] == 1


def test_dump_writes_extract_failed_stub(monkeypatch, tmp_path):
    """Post fetched, but LLM extraction raised: write extract_failed stub."""
    post = {
        "title": "Some Post",
        "post_date": "2026-05-04",
        "body_html_clean": "non-empty body",
    }
    _patch_fetch_and_extract(
        monkeypatch, post, extracted=None,
        extract_raises=ManciniExtractionError("API timeout"),
    )

    out = dump_plan_for_trading_date(date(2026, 5, 6), output_dir=tmp_path)
    assert out is not None
    payload = json.loads(out.read_text())

    assert payload["extract_status"] == "extract_failed"
    assert "API timeout" in payload["error"]
    assert payload["post_title"] == "Some Post"
    assert payload["plan"] is None


def test_dump_writes_empty_body_stub(monkeypatch, tmp_path):
    """Post fetched but body is empty (paywall, parser regression)."""
    post = {"title": "Empty", "post_date": "2026-05-04", "body_html_clean": ""}
    _patch_fetch_and_extract(monkeypatch, post, extracted=ManciniPlan())

    out = dump_plan_for_trading_date(date(2026, 5, 6), output_dir=tmp_path)
    payload = json.loads(out.read_text())

    assert payload["extract_status"] == "empty_body"
    assert payload["plan"] is None


def test_dump_swallows_fetch_exceptions(monkeypatch, tmp_path):
    """fetch_latest_post raising shouldn't crash the cron — write a stub."""
    import live.substack_compare as sc
    import live.mancini_levels as ml

    def _raise():
        raise RuntimeError("network down")

    monkeypatch.setattr(sc, "fetch_latest_post", _raise)
    monkeypatch.setattr(ml, "_get_body_text", lambda p: "")

    out = dump_plan_for_trading_date(date(2026, 5, 6), output_dir=tmp_path)
    assert out is not None
    payload = json.loads(out.read_text())

    assert payload["extract_status"] == "fetch_failed"
    assert "network down" in payload["error"]


# ---------------------------------------------------------------------------
# load_plan — engine-side reader
# ---------------------------------------------------------------------------


def _write_plan_file(tmp_path: Path, trading_date: str, payload: dict) -> Path:
    p = tmp_path / f"mancini_plan_{trading_date}.json"
    p.write_text(json.dumps(payload, default=str))
    return p


def test_load_plan_returns_validated_model(tmp_path):
    plan = ManciniPlan(lean="bullish", mode="trending", targets=[7300.0])
    _write_plan_file(tmp_path, "2026-05-06", {
        "schema_version": 1,
        "trading_date": "2026-05-06",
        "post_date": "2026-05-04",
        "post_title": "T",
        "fetched_at": "2026-05-05T02:30:00",
        "extract_status": "ok",
        "plan": plan.model_dump(),
        "error": "",
    })
    loaded = load_plan(date(2026, 5, 6), input_dir=tmp_path)
    assert isinstance(loaded, ManciniPlan)
    assert loaded.lean == "bullish"
    assert loaded.mode == "trending"
    assert loaded.targets == [7300.0]


def test_load_plan_returns_none_for_missing_file(tmp_path):
    assert load_plan(date(2026, 5, 6), input_dir=tmp_path) is None


def test_load_plan_returns_none_for_failed_stub(tmp_path):
    _write_plan_file(tmp_path, "2026-05-06", {
        "schema_version": 1,
        "trading_date": "2026-05-06",
        "post_date": "",
        "post_title": "",
        "fetched_at": "2026-05-05T02:30:00",
        "extract_status": "no_post",
        "plan": None,
        "error": "",
    })
    assert load_plan(date(2026, 5, 6), input_dir=tmp_path) is None


def test_load_plan_returns_none_for_corrupt_json(tmp_path):
    p = tmp_path / "mancini_plan_2026-05-06.json"
    p.write_text("not json {{{")
    assert load_plan(date(2026, 5, 6), input_dir=tmp_path) is None


def test_load_plan_returns_none_for_unexpected_schema_version(tmp_path):
    _write_plan_file(tmp_path, "2026-05-06", {
        "schema_version": 99,
        "extract_status": "ok",
        "plan": ManciniPlan().model_dump(),
    })
    assert load_plan(date(2026, 5, 6), input_dir=tmp_path) is None
