"""Tests for the focused FB-levels Discord card (live/post_fb_levels.py)."""
import json

from live.post_fb_levels import build_content, _already_posted, _state_path


def _plan():
    return {
        "post_title": "July 1st Plan",
        "lean": "bullish",
        "targets": [7550.0, 7560.0],
        "planned_setups": [
            {"setup_type": "failed_breakdown", "direction": "long",
             "level_price": 7491.0, "conviction": "high",
             "context": "FB of the 9am low at 7491 from which ES rallied 70+ "
                        "points; actionable on flush and recover"},
            {"setup_type": "failed_breakdown", "direction": "long",
             "level_price": 7368.0, "conviction": "low",
             "context": "FB of 7368; bonus if we hit 7354 on this one"},
            # non-FB and short setups must be excluded
            {"setup_type": "level_reclaim", "direction": "long",
             "level_price": 7542.0, "conviction": "low", "context": "LR"},
            {"setup_type": "failed_breakdown", "direction": "short",
             "level_price": 7459.0, "conviction": "low", "context": "short FB"},
        ],
    }


def test_only_long_failed_breakdowns_included():
    desc = build_content(_plan(), "2026-07-01")["embeds"][0]["description"]
    assert "7491" in desc and "7368" in desc
    assert "7542" not in desc  # level_reclaim excluded
    assert "short FB" not in desc  # short excluded


def test_levels_sorted_high_to_low():
    desc = build_content(_plan(), "2026-07-01")["embeds"][0]["description"]
    assert desc.index("7491") < desc.index("7368")


def test_full_sentence_not_truncated():
    """The whole context line must survive — no ellipsis truncation."""
    desc = build_content(_plan(), "2026-07-01")["embeds"][0]["description"]
    assert "actionable on flush and recover" in desc
    assert "…" not in desc


def test_target_ladder_and_lean_present():
    embed = build_content(_plan(), "2026-07-01")["embeds"][0]
    assert "Trim ladder:" in embed["description"]
    assert "7550" in embed["description"] and "7560" in embed["description"]
    assert "Bullish" in embed["description"]
    assert embed["color"] == 0x2ecc71  # green for bullish


def test_idempotency_matches_on_title(tmp_path):
    state = _state_path(tmp_path, "2026-07-01")
    assert _already_posted(state, "July 1st Plan") is False
    state.write_text(json.dumps({"post_title": "July 1st Plan"}))
    assert _already_posted(state, "July 1st Plan") is True
    # a corrected/newer post (different title) should NOT be considered posted
    assert _already_posted(state, "July 1st Plan (revised)") is False
