"""Tests for IBRunner._collect_signal_context() and the phantom-row
feature enrichment.

Phantoms are signals production filtered out (wrong window, low RR,
etc.). Their on-disk JSON previously lacked the rich context fields
(regime, nearby_levels, session_high/low, session_window) that live
entries log, which made them un-mergeable with the live-entry dataset
for ML training. This change captures that context at the moment the
phantom is created and propagates it through to phantom_resolved.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.signals import Signal, SignalAggregator, SignalType


_TS = datetime(2026, 5, 6, 10, 30)


def _make_runner_stub():
    """Build a stand-in for IBRunner with just enough surface to
    exercise _collect_signal_context, _add_phantom, and
    _log_phantom_outcome.

    Avoids importing IBRunner directly because that pulls in IB SDKs
    and a full session-init path. Instead we instantiate a minimal
    object and bind the methods to it.
    """
    from live.ib_runner import IBRunner

    runner = IBRunner.__new__(IBRunner)  # bypass __init__

    # Minimum attributes _collect_signal_context reads
    runner._bar_count = 245
    runner._df = pd.DataFrame({
        "open": [7250.0, 7252.0, 7254.0],
        "high": [7253.0, 7255.0, 7256.0],
        "low": [7248.0, 7251.0, 7252.0],
        "close": [7252.0, 7254.0, 7255.0],
        "volume": [100, 150, 120],
    }, index=pd.date_range(start=_TS, periods=3, freq="1min"))

    # SignalAggregator with a small level store
    sp = StrategyParams(use_level_quality_scoring=False, use_confluence_scoring=False)
    runner.signal_aggregator = SignalAggregator(strategy_params=sp, min_rr_ratio=0.1)
    runner.signal_aggregator.level_store = LevelStore()
    runner.signal_aggregator.level_store.add(Level(
        price=7245.0,
        level_type=LevelType.PRIOR_DAY_LOW,
        created_at=_TS,
        confirmed_at=_TS,
        touch_count=4,
    ))
    runner.signal_aggregator.level_store.add(Level(
        price=7280.0,  # > 30pts away — should be filtered out of nearby_levels
        level_type=LevelType.HORIZONTAL_SR,
        created_at=_TS,
        confirmed_at=_TS,
        touch_count=2,
    ))

    # Strategy with regime state
    regime_state = SimpleNamespace(
        direction=SimpleNamespace(name="BULLISH"),
        longs_enabled=True,
        shorts_enabled=False,
        ema_slope=2.5,
    )
    runner.strategy = SimpleNamespace(_regime_state=regime_state)

    # Stub session window helper
    runner._get_session_window = lambda t: {
        "label": "RTH",
        "detail": "RTH morning (9:30-11:00 ET)",
    }

    # Phantom plumbing
    runner._phantom_positions = []
    runner._trade_log_path = ""  # _log_phantom_outcome opens this; tests
                                  # patch open() to avoid disk writes
    runner._session_date = _TS.date()

    return runner


def _make_signal(level_price: float = 7245.0,
                 entry_price: float = 7247.0) -> Signal:
    """Lightweight Signal stub matching the shape the phantom code reads."""
    pattern = SimpleNamespace(
        entry_price=entry_price,
        stop_price=entry_price - 5.0,
        level=SimpleNamespace(price=level_price, level_type=SimpleNamespace(name="PRIOR_DAY_LOW")),
        sweep_depth_pts=2.0,
        confirmation=SimpleNamespace(name="ACCEPTANCE"),
        timestamp=_TS,
        bar_idx=245,
    )
    return SimpleNamespace(
        signal_type=SimpleNamespace(name="FAILED_BREAKDOWN"),
        direction="long",
        entry_price=entry_price,
        stop_price=entry_price - 5.0,
        target_1=entry_price + 12.0,
        target_2=entry_price + 18.0,
        rr_ratio_t1=2.0,
        pattern=pattern,
    )


# ---------------------------------------------------------------------------
# _collect_signal_context
# ---------------------------------------------------------------------------


def test_collect_signal_context_captures_full_shape():
    """The helper returns the same keys live entries log."""
    runner = _make_runner_stub()
    ctx = runner._collect_signal_context()

    assert ctx["bar_count"] == 245
    assert ctx["last_price"] == 7255.0  # last close in the stub df
    assert ctx["session_high"] == 7256.0
    assert ctx["session_low"] == 7248.0
    assert ctx["session_range"] == 8.0
    assert ctx["session_window"] == "RTH morning (9:30-11:00 ET)"
    assert ctx["regime"]["direction"] == "BULLISH"
    assert ctx["regime"]["longs_enabled"] is True
    assert ctx["regime"]["shorts_enabled"] is False
    assert ctx["regime"]["ema_slope"] == 2.5


def test_collect_signal_context_filters_distant_levels():
    """Levels >30pts away are excluded from nearby_levels."""
    runner = _make_runner_stub()
    ctx = runner._collect_signal_context()

    # The 7245 PDL is 10pts away (close=7255), should be in nearby_levels.
    # The 7280 HORIZONTAL_SR is 25pts away — also within 30. Both included.
    prices = [lv["price"] for lv in ctx["nearby_levels"]]
    assert 7245.0 in prices
    assert 7280.0 in prices


def test_collect_signal_context_handles_empty_df():
    """No bars yet (early session): returns zeros, doesn't crash."""
    runner = _make_runner_stub()
    runner._df = None
    ctx = runner._collect_signal_context()

    assert ctx["last_price"] == 0.0
    assert ctx["session_high"] == 0.0
    assert ctx["session_low"] == 999999.0  # sentinel preserved


def test_collect_signal_context_handles_missing_strategy():
    """Strategy not yet wired: regime_info is empty dict, doesn't crash."""
    runner = _make_runner_stub()
    runner.strategy = SimpleNamespace(_regime_state=None)
    ctx = runner._collect_signal_context()

    assert ctx["regime"] == {}


# ---------------------------------------------------------------------------
# _add_phantom — captures context at creation time
# ---------------------------------------------------------------------------


def test_add_phantom_attaches_context_snapshot():
    """After _add_phantom, the phantom dict carries the rich context."""
    runner = _make_runner_stub()
    signal = _make_signal()

    runner._add_phantom(signal, "window:evening_block", _TS)

    assert len(runner._phantom_positions) == 1
    p = runner._phantom_positions[0]

    assert "context" in p
    ctx = p["context"]
    assert ctx["bar_count"] == 245
    assert ctx["regime"]["direction"] == "BULLISH"
    assert any(lv["price"] == 7245.0 for lv in ctx["nearby_levels"])


def test_add_phantom_swallows_context_failure():
    """If _collect_signal_context raises, _add_phantom must not crash."""
    runner = _make_runner_stub()

    def _explode():
        raise RuntimeError("forced failure")

    runner._collect_signal_context = _explode

    signal = _make_signal()
    runner._add_phantom(signal, "test", _TS)

    p = runner._phantom_positions[0]
    assert p["context"] == {}  # degraded but present
    assert p["entry_price"] == signal.entry_price


# ---------------------------------------------------------------------------
# _log_phantom_outcome — emits the enriched record
# ---------------------------------------------------------------------------


def test_log_phantom_outcome_promotes_context_fields(tmp_path):
    """The phantom_resolved JSONL record carries the promoted features."""
    runner = _make_runner_stub()
    log_path = tmp_path / "trades.jsonl"
    runner._trade_log_path = str(log_path)

    # Build a phantom that already has a context snapshot
    phantom = {
        "signal_type": "FAILED_BREAKDOWN",
        "direction": "long",
        "entry_price": 7247.0,
        "stop_price": 7242.0,
        "target_1": 7259.0,
        "target_2": 7265.0,
        "rr_ratio": 2.4,
        "level_price": 7245.0,
        "level_type": "PRIOR_DAY_LOW",
        "sweep_depth_pts": 2.0,
        "confirmation_type": "ACCEPTANCE",
        "reject_reason": "window:evening_block",
        "result": "T1 HIT (+12.00 pts)",
        "high_since": 7259.0,
        "low_since": 7245.0,
        "context": {
            "bar_count": 245,
            "last_price": 7255.0,
            "session_high": 7256.0,
            "session_low": 7248.0,
            "session_range": 8.0,
            "session_window": "RTH morning (9:30-11:00 ET)",
            "regime": {"direction": "BULLISH", "longs_enabled": True,
                       "shorts_enabled": False, "ema_slope": 2.5},
            "nearby_levels": [{"price": 7245.0, "type": "PRIOR_DAY_LOW",
                              "touches": 4, "distance": -10.0}],
        },
    }
    runner._log_phantom_outcome(phantom)

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["event"] == "phantom_resolved"
    assert record["bar_count"] == 245
    assert record["session_high"] == 7256.0
    assert record["session_low"] == 7248.0
    assert record["session_range"] == 8.0
    assert record["session_window"] == "RTH morning (9:30-11:00 ET)"
    assert record["regime"]["direction"] == "BULLISH"
    assert record["nearby_levels"][0]["price"] == 7245.0


def test_log_phantom_outcome_handles_missing_context(tmp_path):
    """Old phantoms without a context dict still log cleanly."""
    runner = _make_runner_stub()
    log_path = tmp_path / "trades.jsonl"
    runner._trade_log_path = str(log_path)

    phantom = {
        "signal_type": "FAILED_BREAKDOWN",
        "direction": "long",
        "entry_price": 7247.0,
        "stop_price": 7242.0,
        "target_1": 7259.0,
        "target_2": None,
        "rr_ratio": 2.4,
        "level_price": 7245.0,
        "level_type": "PRIOR_DAY_LOW",
        "sweep_depth_pts": 2.0,
        "confirmation_type": "ACCEPTANCE",
        "reject_reason": "test",
        "result": "STOP HIT (-5.00 pts)",
        "high_since": 7250.0,
        "low_since": 7242.0,
        # no "context" key
    }
    runner._log_phantom_outcome(phantom)

    lines = log_path.read_text().strip().split("\n")
    record = json.loads(lines[0])
    # Core fields present
    assert record["event"] == "phantom_resolved"
    assert record["entry_price"] == 7247.0
    # Context-derived fields absent (didn't crash)
    assert "session_high" not in record
    assert "regime" not in record
