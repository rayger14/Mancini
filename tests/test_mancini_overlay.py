"""Tests for the Mancini Substack level overlay system.

Covers:
  - LevelType.MANCINI_LEVEL enum + base score wired into compute_confluence_score
  - apply_mancini_overlay modes: confirmation / shadow / augmentation
  - Safety: corrupt JSON, missing files, empty data
  - mancini_levels.load() error handling
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from config.levels import Level, LevelStore, LevelType, compute_confluence_score
from core.mancini_overlay import apply_mancini_overlay, MancinitOverlayResult
from live.mancini_levels import load as load_mancini_levels


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ts() -> datetime:
    return datetime(2026, 4, 16, 9, 30)


def _make_engine_level(price: float, lt: LevelType = LevelType.PRIOR_DAY_LOW) -> Level:
    ts = _ts()
    return Level(price=price, level_type=lt, created_at=ts, confirmed_at=ts)


def _make_store(prices: list[tuple[float, LevelType]]) -> LevelStore:
    store = LevelStore()
    for p, lt in prices:
        store.add(_make_engine_level(p, lt))
    return store


def _sample_mancini(levels: list[dict], *, lean: str = "neutral",
                    status: str = "ok") -> dict:
    return {
        "schema_version": 1,
        "trading_date": "2026-04-16",
        "post_date": "2026-04-15",
        "post_title": "Apr 16 Plan",
        "fetched_at": _ts().isoformat(),
        "lean": lean,
        "parse_status": status,
        "levels": levels,
        "highlights": [],
    }


# ---------------------------------------------------------------------------
# LevelType / confluence score wiring
# ---------------------------------------------------------------------------


class TestLevelTypeWiring:
    """LevelType.MANCINI_LEVEL exists and is wired into confluence scoring."""

    def test_mancini_level_enum_exists(self):
        assert hasattr(LevelType, "MANCINI_LEVEL")

    def test_mancini_level_base_score_is_3(self):
        lv = _make_engine_level(7048.0, LevelType.MANCINI_LEVEL)
        assert compute_confluence_score(lv, [lv]) == 3

    def test_mancini_confirmed_adds_score_bonus(self):
        lv = _make_engine_level(7048.0, LevelType.CLUSTER_LOW)
        base = compute_confluence_score(lv, [lv])
        lv.mancini_confirmed = True
        boosted = compute_confluence_score(lv, [lv])
        assert boosted == base + 2


# ---------------------------------------------------------------------------
# apply_mancini_overlay: confirmation mode
# ---------------------------------------------------------------------------


class TestConfirmationMode:
    """Confirmation mode boosts engine levels within tolerance, never injects."""

    def test_engine_level_within_tolerance_is_confirmed(self):
        store = _make_store([(7050.0, LevelType.PRIOR_DAY_LOW)])
        data = _sample_mancini([
            {"price": 7048.0, "side": "support", "conviction": 3,
             "tags": ["key"], "role": "support"},
        ])
        res = apply_mancini_overlay(
            store=store,
            mancini_data=data,
            mode="confirmation",
            confirm_tolerance_pts=3.0,
            current_price=7075.0,
        )
        assert res.confirmed_count == 1
        assert res.injected_count == 0
        assert res.shadow_count == 0
        # The engine level must actually be marked
        pdl = store.get_active(LevelType.PRIOR_DAY_LOW)[0]
        assert pdl.mancini_confirmed is True
        assert pdl.mancini_side == "support"
        assert pdl.mancini_conviction == 3
        assert "key" in pdl.mancini_tags

    def test_no_duplicate_when_level_within_tolerance(self):
        """A Mancini call near an engine level should not be re-injected."""
        store = _make_store([(7050.0, LevelType.PRIOR_DAY_LOW)])
        data = _sample_mancini([
            {"price": 7049.0, "side": "support", "conviction": 2,
             "tags": [], "role": "support"},
        ])
        apply_mancini_overlay(store, data, mode="confirmation",
                              confirm_tolerance_pts=3.0)
        mancini_levels = store.get_active(LevelType.MANCINI_LEVEL)
        assert len(mancini_levels) == 0
        assert len(store.levels) == 1  # still just the engine PDL

    def test_far_mancini_level_is_blind_spot(self):
        """In confirmation mode, Mancini calls with no engine coverage are reported."""
        store = _make_store([(7200.0, LevelType.PRIOR_DAY_HIGH)])
        data = _sample_mancini([
            {"price": 7048.0, "side": "support", "conviction": 3,
             "tags": ["key"], "role": "support"},
        ])
        res = apply_mancini_overlay(
            store=store,
            mancini_data=data,
            mode="confirmation",
            confirm_tolerance_pts=3.0,
            current_price=7075.0,
        )
        assert res.confirmed_count == 0
        assert res.injected_count == 0
        assert len(res.blind_spots) == 1
        spot = res.blind_spots[0]
        assert spot["price"] == 7048.0
        assert spot["distance_pts"] == 27.0  # abs(7048 - 7075)
        # No MANCINI_LEVEL injected
        assert store.get_active(LevelType.MANCINI_LEVEL) == []


# ---------------------------------------------------------------------------
# apply_mancini_overlay: shadow mode
# ---------------------------------------------------------------------------


class TestShadowMode:
    """Shadow mode injects MANCINI_LEVEL with shadow_only=True."""

    def test_shadow_injects_level_with_shadow_only(self):
        store = _make_store([(7200.0, LevelType.PRIOR_DAY_HIGH)])
        data = _sample_mancini([
            {"price": 7048.0, "side": "support", "conviction": 3,
             "tags": ["key"], "role": "support"},
        ])
        res = apply_mancini_overlay(store, data, mode="shadow",
                                    confirm_tolerance_pts=3.0,
                                    current_price=7075.0)
        assert res.shadow_count == 1
        assert res.injected_count == 0
        # Level is in the store and flagged shadow
        injected = store.get_active(LevelType.MANCINI_LEVEL)
        assert len(injected) == 1
        assert injected[0].price == 7048.0
        assert injected[0].shadow_only is True
        assert injected[0].mancini_confirmed is True
        assert injected[0].mancini_conviction == 3

    def test_shadow_confirms_overlapping_engine_level(self):
        """Engine overlap should still be confirmed, not injected, in shadow mode."""
        store = _make_store([(7048.5, LevelType.PRIOR_DAY_LOW)])
        data = _sample_mancini([
            {"price": 7048.0, "side": "support", "conviction": 3,
             "tags": ["key"], "role": "support"},
        ])
        res = apply_mancini_overlay(store, data, mode="shadow",
                                    confirm_tolerance_pts=3.0)
        assert res.confirmed_count == 1
        assert res.shadow_count == 0
        assert store.get_active(LevelType.MANCINI_LEVEL) == []


# ---------------------------------------------------------------------------
# apply_mancini_overlay: augmentation mode
# ---------------------------------------------------------------------------


class TestAugmentationMode:
    """Augmentation mode injects non-shadow MANCINI_LEVEL entries."""

    def test_augmentation_injects_tradeable_level(self):
        store = _make_store([(7200.0, LevelType.PRIOR_DAY_HIGH)])
        data = _sample_mancini([
            {"price": 7048.0, "side": "support", "conviction": 3,
             "tags": ["key"], "role": "support"},
        ])
        res = apply_mancini_overlay(store, data, mode="augmentation",
                                    confirm_tolerance_pts=3.0)
        assert res.injected_count == 1
        assert res.shadow_count == 0
        injected = store.get_active(LevelType.MANCINI_LEVEL)
        assert len(injected) == 1
        assert injected[0].shadow_only is False
        assert injected[0].mancini_confirmed is True


# ---------------------------------------------------------------------------
# apply_mancini_overlay: safety
# ---------------------------------------------------------------------------


class TestOverlaySafety:
    """Overlay must never raise even on garbage inputs."""

    def test_none_input_returns_missing(self):
        store = _make_store([(7200.0, LevelType.PRIOR_DAY_HIGH)])
        res = apply_mancini_overlay(store, None, mode="shadow")
        assert isinstance(res, MancinitOverlayResult)
        assert res.parse_status == "missing"
        assert res.confirmed_count == 0
        assert res.injected_count == 0
        assert res.shadow_count == 0

    def test_failed_parse_returns_missing(self):
        store = _make_store([(7200.0, LevelType.PRIOR_DAY_HIGH)])
        data = _sample_mancini([], status="failed")
        res = apply_mancini_overlay(store, data, mode="shadow")
        assert res.parse_status == "missing"
        assert len(store.get_active(LevelType.MANCINI_LEVEL)) == 0

    def test_empty_levels_returns_ok_no_changes(self):
        store = _make_store([(7200.0, LevelType.PRIOR_DAY_HIGH)])
        data = _sample_mancini([])
        res = apply_mancini_overlay(store, data, mode="shadow")
        assert res.confirmed_count == 0
        assert res.shadow_count == 0
        assert res.injected_count == 0

    def test_invalid_price_skipped(self):
        store = _make_store([(7200.0, LevelType.PRIOR_DAY_HIGH)])
        data = _sample_mancini([
            {"price": "not-a-number", "side": "support", "conviction": 1, "tags": []},
            {"price": -5, "side": "support", "conviction": 1, "tags": []},
            {"price": 7048.0, "side": "support", "conviction": 2, "tags": []},
        ])
        res = apply_mancini_overlay(store, data, mode="shadow")
        assert res.shadow_count == 1  # only the valid one


# ---------------------------------------------------------------------------
# mancini_levels.load(): safety
# ---------------------------------------------------------------------------


class TestLoadSafety:
    """load() returns None on every failure mode; never raises."""

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert load_mancini_levels(date(2026, 4, 16), input_dir=tmp_path) is None

    def test_corrupt_json_returns_none(self, tmp_path: Path):
        path = tmp_path / "mancini_levels_2026-04-16.json"
        path.write_text("{not valid json")
        assert load_mancini_levels(date(2026, 4, 16), input_dir=tmp_path) is None

    def test_wrong_schema_returns_none(self, tmp_path: Path):
        path = tmp_path / "mancini_levels_2026-04-16.json"
        path.write_text(json.dumps({"schema_version": 99, "levels": []}))
        assert load_mancini_levels(date(2026, 4, 16), input_dir=tmp_path) is None

    def test_failed_parse_returns_none(self, tmp_path: Path):
        path = tmp_path / "mancini_levels_2026-04-16.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "parse_status": "failed",
            "levels": [],
        }))
        assert load_mancini_levels(date(2026, 4, 16), input_dir=tmp_path) is None

    def test_valid_file_returns_dict(self, tmp_path: Path):
        data = _sample_mancini([
            {"price": 7048.0, "side": "support", "conviction": 3,
             "tags": ["key"], "role": "support"},
        ])
        path = tmp_path / "mancini_levels_2026-04-16.json"
        path.write_text(json.dumps(data))
        loaded = load_mancini_levels(date(2026, 4, 16), input_dir=tmp_path)
        assert loaded is not None
        assert loaded["schema_version"] == 1
        assert len(loaded["levels"]) == 1


# ---------------------------------------------------------------------------
# end-to-end wiring: overlay result wired to confluence scoring
# ---------------------------------------------------------------------------


class TestConfluenceIntegration:
    """Confirmed engine levels carry the +2 score bonus after overlay."""

    def test_confirmed_pdl_gets_bonus(self):
        store = _make_store([(7050.0, LevelType.PRIOR_DAY_LOW)])
        pdl = store.levels[0]
        base_score = compute_confluence_score(pdl, list(store.levels))
        data = _sample_mancini([
            {"price": 7048.0, "side": "support", "conviction": 3,
             "tags": ["key"], "role": "support"},
        ])
        apply_mancini_overlay(store, data, mode="confirmation",
                              confirm_tolerance_pts=3.0)
        boosted_score = compute_confluence_score(pdl, list(store.levels))
        assert boosted_score == base_score + 2
