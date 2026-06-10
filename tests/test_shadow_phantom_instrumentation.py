"""Tests for phantom-outcome instrumentation of blocker shadow events.

Only velocity_short historically wrote shadow_outcome records, because
_flush_shadow_events creates a phantom tracker only when an event carries
entry_price AND stop_price — and the blocker features (block_pdl_shorts,
move_exhaustion, capitulation_entry, daily_structure_short_suppression,
fb_level_too_old) never included stop/target. Without outcomes these gates
can never be promoted or killed on evidence.

Each blocker event must now carry entry_price, stop_price and an explicit
direction (capitulation_entry blocks SHORTS but its feature name contains
no "short", so the name heuristic in _flush_shadow_events guessed long).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.patterns import ConfirmationType, PatternSignal
from core.signals import SignalAggregator, SignalType
from live.ib_runner import IBRunner


_TS = datetime(2024, 6, 15, 10, 30)


def _make_level(price: float, level_type=LevelType.CLUSTER_LOW,
                created_at=_TS) -> Level:
    return Level(
        price=price,
        level_type=level_type,
        created_at=created_at,
        touch_count=5,
    )


def _short_pattern(level_type=LevelType.CLUSTER_LOW) -> PatternSignal:
    return PatternSignal(
        pattern_type="breakdown_short",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=_make_level(5800.0, level_type),
        sweep_low=5790.0,
        entry_price=5799.0,
        stop_price=5805.0,
        bar_idx=100,
        timestamp=_TS,
        sweep_depth_pts=5.0,
        direction="short",
        sweep_high=5805.0,
    )


def _agg(**flags) -> SignalAggregator:
    params = StrategyParams(min_session_range_pts=0.0, **flags)
    agg = SignalAggregator(strategy_params=params)
    agg.level_store = LevelStore()
    agg.level_store.add(_make_level(5785.0))
    agg.level_store.add(_make_level(5770.0))
    return agg


def _event(agg: SignalAggregator, feature: str) -> dict:
    events = [e for e in agg.shadow_events if e["feature"] == feature]
    assert len(events) == 1, f"expected 1 {feature} event, got {len(events)}"
    return events[0]


class TestShortBlockerEventsCarryPhantomFields:
    def test_block_pdl_shorts_event(self):
        agg = _agg(block_pdl_shorts=True)
        result = agg._qualify_short_signal(
            _short_pattern(LevelType.PRIOR_DAY_LOW), SignalType.BREAKDOWN_SHORT
        )
        assert result is None
        ev = _event(agg, "block_pdl_shorts")
        assert ev["entry_price"] == 5799.0
        assert ev["stop_price"] == 5805.0
        assert ev["direction"] == "short"

    def test_move_exhaustion_event(self):
        agg = _agg(block_pdl_shorts=False, block_capitulation_shorts=False)
        agg._session_low = 5750.0  # below any short target → exhausted
        agg._session_high = 5810.0
        result = agg._qualify_short_signal(
            _short_pattern(), SignalType.BREAKDOWN_SHORT
        )
        assert result is None
        ev = _event(agg, "move_exhaustion")
        assert ev["entry_price"] == 5799.0
        assert ev["stop_price"] == 5805.0
        assert ev["direction"] == "short"

    def test_capitulation_entry_event(self):
        agg = _agg(block_pdl_shorts=False, block_capitulation_shorts=True)
        # entry 4 pts above session low, session high 41 pts above entry
        agg._session_low = 5795.0
        agg._session_high = 5840.0
        result = agg._qualify_short_signal(
            _short_pattern(), SignalType.BREAKDOWN_SHORT
        )
        assert result is None
        ev = _event(agg, "capitulation_entry")
        assert ev["entry_price"] == 5799.0
        assert ev["stop_price"] == 5805.0
        assert ev["direction"] == "short"


class TestFBAgeBlockerEventCarriesPhantomFields:
    def test_fb_level_too_old_event(self):
        agg = _agg()
        old_level = _make_level(
            5800.0, LevelType.CLUSTER_LOW, created_at=_TS - timedelta(hours=48)
        )
        pattern = PatternSignal(
            pattern_type="failed_breakdown",
            confirmation=ConfirmationType.ACCEPTANCE,
            level=old_level,
            sweep_low=5795.0,
            entry_price=5801.0,
            stop_price=5790.0,
            bar_idx=100,
            timestamp=_TS,
            sweep_depth_pts=5.0,
            direction="long",
        )
        result = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
        assert result is None
        ev = _event(agg, "fb_level_too_old")
        assert ev["entry_price"] == 5801.0
        assert ev["stop_price"] == 5790.0
        assert ev["direction"] == "long"


class TestFlushRespectsEventDirection:
    def test_phantom_direction_from_event_not_name(self, tmp_path):
        runner = SimpleNamespace(
            signal_aggregator=SimpleNamespace(shadow_events=[{
                "feature": "capitulation_entry",  # no "short" in the name
                "timestamp": str(_TS),
                "entry_price": 5799.0,
                "stop_price": 5805.0,
                "target_1": 5785.0,
                "direction": "short",
            }]),
            _shadow_log_path=tmp_path / "shadow.jsonl",
            _shadow_phantoms=[],
            _bar_count=100,
        )
        IBRunner._flush_shadow_events(runner)
        assert len(runner._shadow_phantoms) == 1
        assert runner._shadow_phantoms[0]["direction"] == "short"

    def test_name_heuristic_still_works_without_direction(self, tmp_path):
        runner = SimpleNamespace(
            signal_aggregator=SimpleNamespace(shadow_events=[{
                "feature": "velocity_short",
                "timestamp": str(_TS),
                "entry_price": 5799.0,
                "stop_price": 5805.0,
                "target_1": 5785.0,
            }]),
            _shadow_log_path=tmp_path / "shadow.jsonl",
            _shadow_phantoms=[],
            _bar_count=100,
        )
        IBRunner._flush_shadow_events(runner)
        assert len(runner._shadow_phantoms) == 1
        assert runner._shadow_phantoms[0]["direction"] == "short"
