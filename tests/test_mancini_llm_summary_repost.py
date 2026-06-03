"""Tests for content-aware idempotency in the Mancini brief poster.

The original idempotency just checked "state file exists". That meant
when the primary cron grabbed an old/stale post and then the backup
cron caught the actual fresh post, the Discord brief was NEVER updated.

Fix: state file records ``post_title``. If the new extraction shows a
different title, the brief re-publishes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from live.mancini_llm_summary import should_post


@pytest.fixture
def state_file(tmp_path) -> Path:
    return tmp_path / ".mancini_brief_posted_2026-06-04"


def test_no_prior_state_means_post(state_file):
    post, reason = should_post(state_file, "Any Title")
    assert post is True
    assert "no prior" in reason


def test_same_title_means_skip(state_file):
    state_file.write_text(json.dumps({
        "post_title": "June 4 Plan",
        "posted_at": "2026-06-03T17:30:00",
    }))
    post, reason = should_post(state_file, "June 4 Plan")
    assert post is False
    assert "unchanged" in reason


def test_different_title_means_repost(state_file):
    """The actual production bug: primary cron grabbed yesterday's plan,
    backup caught the real one — old idempotency blocked repost."""
    state_file.write_text(json.dumps({
        "post_title": "June 3rd Plan",  # stale (yesterday)
        "posted_at": "2026-06-03T13:30:00",
    }))
    post, reason = should_post(state_file, "June 4 Plan")  # fresh
    assert post is True
    assert "title changed" in reason


def test_legacy_state_file_treated_as_posted(state_file):
    """Pre-v2 state was a raw ISO timestamp string — must not trigger
    a flood of re-posts on upgrade."""
    state_file.write_text("2026-06-03T17:30:00")  # legacy format
    post, reason = should_post(state_file, "June 4 Plan")
    assert post is False
    assert "legacy" in reason


def test_whitespace_in_title_is_normalized(state_file):
    """Substack occasionally returns the same title with trailing
    whitespace — must not trigger a spurious re-post."""
    state_file.write_text(json.dumps({
        "post_title": "June 4 Plan",
        "posted_at": "2026-06-03T17:30:00",
    }))
    post, reason = should_post(state_file, "  June 4 Plan  ")
    assert post is False


def test_corrupt_state_file_treated_as_legacy(state_file):
    """Half-written JSON, bad encoding etc — fall back to legacy
    (treat as posted), do not spam."""
    state_file.write_text("{ not valid json")
    post, reason = should_post(state_file, "June 4 Plan")
    assert post is False
