"""Tests for Discord brief setup pagination.

2026-06-23: the Mancini brief packed all high/medium (and all low) conviction
setups into a single Discord embed field. Discord caps field values at 1024
chars, so a plan with many setups got truncated mid-list with a "…" — the
brief "cut off" talking about the conviction setups. Long blocks must split
across multiple fields, losing nothing.
"""
from __future__ import annotations

from live.mancini_llm_summary import (
    _paginate_rows_to_blocks,
    _setup_rows,
    build_payload,
)

DISCORD_FIELD_LIMIT = 1024


def _setup(price, conviction, ctx, setup_type="failed_breakdown", direction="long"):
    return {
        "level_price": price,
        "conviction": conviction,
        "context": ctx,
        "setup_type": setup_type,
        "direction": direction,
    }


class TestPaginateRowsToBlocks:
    def test_empty_returns_no_blocks(self):
        assert _paginate_rows_to_blocks([]) == []

    def test_few_rows_one_block(self):
        rows = ["row one", "row two", "row three"]
        blocks = _paginate_rows_to_blocks(rows)
        assert len(blocks) == 1
        assert blocks[0].startswith("```\n") and blocks[0].endswith("\n```")
        for r in rows:
            assert r in blocks[0]

    def test_many_rows_split_and_each_within_limit(self):
        rows = [f"{i:02d} " + "x" * 80 for i in range(40)]  # ~40 long rows
        blocks = _paginate_rows_to_blocks(rows, budget=DISCORD_FIELD_LIMIT)
        assert len(blocks) > 1
        for b in blocks:
            assert len(b) <= DISCORD_FIELD_LIMIT
            assert b.startswith("```\n") and b.endswith("\n```")

    def test_no_rows_are_lost_across_blocks(self):
        rows = [f"ROW-{i}" for i in range(60)]
        blocks = _paginate_rows_to_blocks(rows, budget=120)
        joined = "\n".join(blocks)
        for r in rows:
            assert r in joined


class TestSetupRows:
    def test_filters_by_conviction(self):
        setups = [
            _setup(7000, "high", "a"),
            _setup(6990, "low", "b"),
            _setup(6980, "medium", "c"),
        ]
        hi_med = _setup_rows(setups, {"high", "medium"})
        lo = _setup_rows(setups, {"low"})
        assert len(hi_med) == 2
        assert len(lo) == 1


class TestBuildPayloadNoTruncation:
    def test_many_setups_produce_multiple_fields_none_truncated(self):
        # 20 high-conviction setups with long context — would overflow one
        # 1024-char field. Expect multiple setup fields, none ending in "…".
        setups = [
            _setup(7000 + i, "high",
                   "long descriptive context number " + str(i) + " " * 20)
            for i in range(20)
        ]
        plan_json = {
            "extract_status": "ok",
            "trading_date": "2026-06-24",
            "post_title": "June 24th Plan",
            "plan": {"lean": "bullish", "planned_setups": setups},
        }
        payload = build_payload(plan_json)
        embed = payload["embeds"][0]
        setup_fields = [f for f in embed["fields"]
                        if "Conviction" in f["name"] or "Setups" in f["name"]]
        assert len(setup_fields) >= 2, "long setup list should span >1 field"
        for f in setup_fields:
            assert len(f["value"]) <= DISCORD_FIELD_LIMIT
            assert not f["value"].rstrip().endswith("…"), (
                "setups must paginate, not truncate"
            )
        # All 20 setups represented across the fields.
        all_setup_text = "\n".join(f["value"] for f in setup_fields)
        for i in range(20):
            assert f"{7000 + i:.2f}" in all_setup_text
