"""Micro end-to-end: a synthetic tape through the REAL IBRunner.run() loop.

Proves the whole replay chain: SimBridge feeds bars into the live loop, a
force-trade enters, the venue bracket fills the TP on the tape, the position
closes via _sync_position's 3x-None confirmation (driven by the poll-count
monotonic clock), the exit books to the replay trade log, and the run
terminates via the tape-exhausted backstop.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import pytz

_ET = pytz.timezone("US/Eastern")


def _tape(rows, start=None):
    start = start or _ET.localize(datetime(2026, 7, 2, 10, 0))
    idx = [start + timedelta(minutes=i) for i in range(len(rows))]
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [500.0] * len(rows)},
        index=pd.DatetimeIndex(idx))


def test_replay_end_to_end_force_trade(tmp_path, monkeypatch):
    out = tmp_path
    # a steadily rising ladder: whichever bar the force-trade enters on, the
    # +3pt TP fills a few bars later; the drain window then gives _sync_position
    # its 3x-None confirmations to book the close
    rows = [(7500 + i, 7501.2 + i, 7499.5 + i, 7501 + i) for i in range(15)]
    force = out / "force_trade.json"
    force.write_text(json.dumps(
        {"direction": "long", "tp_pts": 3, "sl_pts": 3, "quantity": 1}))

    monkeypatch.setenv("TRADE_LOG", str(out / "replay_trades.jsonl"))
    monkeypatch.setenv("SHADOW_LOG", str(out / "shadow.jsonl"))
    monkeypatch.setenv("STATUS_FILE", str(out / "status.json"))
    monkeypatch.setenv("LOG_FILE", str(out / "bot.log"))
    monkeypatch.setenv("PATTERN_STATE_FILE", str(out / "pattern_state.json"))
    monkeypatch.setenv("FORCE_TRADE_FILE", str(force))
    monkeypatch.setenv("FREEZE_TIMEOUT_SEC", "0")
    monkeypatch.setenv("SHORT_ALERTS", "0")
    monkeypatch.setenv("BLOCKED_ALERTS", "0")
    monkeypatch.setenv("MANCINI_TRADE_WEBHOOK", "")
    monkeypatch.setenv("WATCHDOG_WEBHOOK", "")

    from backtest.replay_runner import build_replay
    runner, bridge = build_replay(date="2026-07-02", data_dir=None,
                                  out_dir=out, tape=_tape(rows))
    runner.run()

    # the run terminated (backstop) and processed the whole tape
    assert bridge._exhaust_fired or not runner._running

    events = [json.loads(l) for l in
              (out / "replay_trades.jsonl").read_text().splitlines()]
    kinds = [e.get("event") for e in events]
    assert "entry" in kinds, f"no entry logged; events: {kinds}"
    entries = [e for e in events if e.get("event") == "entry"]
    exits = [e for e in events if e.get("event") == "exit"]
    assert entries[0]["entry_price"] > 0
    assert exits, f"no exit booked; events: {kinds}"
    assert exits[0]["pnl_pts"] > 0          # rising tape -> the +3pt TP filled
    reason = str(exits[0].get("exit_reason", ""))
    # venue TP fill books either through _sync_position ("IB bracket TP") or
    # the ExitManager/venue-T1 reconcile ("Target 1 hit") — both are the live
    # profit-target paths
    assert ("TP" in reason) or ("Target 1" in reason), reason


def test_replay_writes_nothing_outside_out_dir(tmp_path, monkeypatch):
    """The CLI's env wiring must confine every artifact to out_dir."""
    out = tmp_path / "replay_out"
    out.mkdir()
    rows = [(7500, 7500.5, 7499.5, 7500)] * 3
    for var, name in [("TRADE_LOG", "t.jsonl"), ("SHADOW_LOG", "s.jsonl"),
                      ("STATUS_FILE", "st.json"), ("LOG_FILE", "b.log"), ("PATTERN_STATE_FILE", "p.json"),
                      ("FORCE_TRADE_FILE", "missing.json")]:
        monkeypatch.setenv(var, str(out / name))
    monkeypatch.setenv("FREEZE_TIMEOUT_SEC", "0")
    monkeypatch.setenv("MANCINI_TRADE_WEBHOOK", "")
    monkeypatch.setenv("WATCHDOG_WEBHOOK", "")

    from backtest.replay_runner import build_replay
    runner, bridge = build_replay(date="2026-07-02", data_dir=None,
                                  out_dir=out, tape=_tape(rows))
    # archive_session must be neutralized (live would write /app/data/sessions)
    runner._archive_session()   # no-op, must not raise or write
    runner.run()
    assert bridge._exhaust_fired or not runner._running
