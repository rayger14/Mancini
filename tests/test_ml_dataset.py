"""Smoke tests for ml.dataset — verifies the JSON-to-DataFrame plumbing
without depending on a real trade log being present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.dataset import build_dataset, _flatten_entry, _flatten_phantom, _parse_result_label


def test_parse_result_label():
    assert _parse_result_label("T1 HIT (+12.5 pts)") == 1
    assert _parse_result_label("STOP HIT (+5.0 pts)") == 0
    assert _parse_result_label("STOP HIT (-10.0 pts)") == 0
    assert _parse_result_label("") is None
    assert _parse_result_label(None) is None
    assert _parse_result_label("VOIDED") is None


def test_flatten_phantom_extracts_label():
    phantom = {
        "event": "phantom_resolved",
        "timestamp": "2026-04-30T15:30:00",
        "session_date": "2026-04-30",
        "signal_type": "FAILED_BREAKDOWN",
        "entry_price": 7250.0,
        "stop_price": 7240.0,
        "target_1": 7270.0,
        "result": "T1 HIT (+20.0 pts)",
    }
    row = _flatten_phantom(phantom)
    assert row["label"] == 1
    assert row["entry_price"] == 7250.0
    assert row["ts_hour"] == 15
    assert row["source"] == "phantom_resolved"


def test_flatten_entry_pnl_label():
    entry = {
        "event": "entry",
        "timestamp": "2026-04-30T10:00:00",
        "session_date": "2026-04-30",
        "session_high": 7280.0,
        "session_low": 7240.0,
        "session_range": 40.0,
        "last_price": 7260.0,
        "direction": "long",
        "pattern_type": "failed_breakdown",
        "signal": {
            "type": "FAILED_BREAKDOWN",
            "entry": 7260.0,
            "stop": 7250.0,
            "target_1": 7280.0,
            "rr_ratio": 2.0,
            "level_price": 7255.0,
            "level_type": "PRIOR_DAY_LOW",
        },
        "regime": {"direction": "BULLISH", "longs_enabled": True,
                   "shorts_enabled": False, "ema_slope": 1.5},
        "nearby_levels": [
            {"price": 7255.0, "type": "PRIOR_DAY_LOW", "touches": 3, "distance": 5.0}
        ],
    }
    exit_event = {
        "event": "exit",
        "session_date": "2026-04-30",
        "pnl_pts": 18.0,  # close to T1 width of 20
    }
    row = _flatten_entry(entry, exit_event)
    assert row["label"] == 1  # 18 / 20 >= 0.8
    assert row["pos_in_range"] == 0.5
    assert row["nearest_level_distance"] == 5.0


def test_flatten_entry_loss():
    entry = {
        "event": "entry",
        "timestamp": "2026-04-30T10:00:00",
        "session_date": "2026-04-30",
        "direction": "long",
        "signal": {
            "type": "FAILED_BREAKDOWN",
            "entry": 7260.0,
            "stop": 7250.0,
            "target_1": 7280.0,
        },
    }
    exit_event = {"pnl_pts": -10.0, "session_date": "2026-04-30"}
    row = _flatten_entry(entry, exit_event)
    assert row["label"] == 0


def test_build_dataset_round_trip(tmp_path: Path):
    """End-to-end: write a small JSONL, build_dataset reads it, returns
    a DataFrame with expected labels.
    """
    trades = tmp_path / "trades.jsonl"
    records = [
        # Live entry + exit pair (T1 hit)
        {"event": "entry", "timestamp": "2026-04-30T10:00:00",
         "session_date": "2026-04-30", "direction": "long",
         "signal": {"entry": 7260.0, "stop": 7250.0, "target_1": 7280.0}},
        {"event": "exit", "timestamp": "2026-04-30T10:30:00",
         "session_date": "2026-04-30", "direction": "long",
         "pnl_pts": 19.0},
        # Phantom loss
        {"event": "phantom_resolved", "timestamp": "2026-04-30T11:00:00",
         "session_date": "2026-04-30", "signal_type": "FAILED_BREAKDOWN",
         "entry_price": 7270.0, "result": "STOP HIT (-10.0 pts)"},
    ]
    with trades.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    df = build_dataset(trades)
    assert len(df) == 2  # entry+exit -> 1 row, phantom -> 1 row
    assert set(df["label"].unique()) == {0, 1}
    assert "live_entry" in df["source"].values
    assert "phantom_resolved" in df["source"].values
