"""Tests for missed-bar backfill (continuous-engine Layer 1).

The bot used to process only bars[-2] per poll and advance a single
timestamp watermark, so any 1-min bars that closed during a stall
(reconnect burst, IB pacing, farm blip) were silently dropped and never
evaluated for signals — the missed-trade source. get_new_bars() must
replay EVERY closed bar newer than the watermark, in order.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from live.ib_bridge import IBBridge


def _raw_bar(dt_utc: datetime, close: float = 7000.0, vol: float = 100.0):
    """Mimic an ib_async historical bar object (formatDate=2 → datetime date)."""
    return SimpleNamespace(date=dt_utc, open=close - 1, high=close + 1,
                           low=close - 2, close=close, volume=vol)


def _bridge(last_bar_time=None):
    b = IBBridge.__new__(IBBridge)
    b._last_bar_time = last_bar_time
    return b


def _series(n: int, start: datetime | None = None):
    """n consecutive 1-min bars starting at `start` (UTC)."""
    if start is None:
        start = datetime(2026, 6, 29, 14, 30, tzinfo=timezone.utc)
    return [_raw_bar(start + timedelta(minutes=i), close=7000.0 + i) for i in range(n)]


class TestNewBarsFromRaw:
    def test_cold_start_seeds_to_latest_only(self):
        # No watermark (fresh restart): must NOT replay the whole fetch window
        # as a gap — just seed to the latest closed bar.
        b = _bridge(last_bar_time=None)
        out = b._new_bars_from_raw(_series(3))
        assert [d["close"] for d in out] == [7002.0]
        assert b._last_bar_time is not None

    def test_gap_replays_all_missed_bars(self):
        bars = _series(5)
        b = _bridge()
        # Seed the watermark at bar 0 (as if we'd processed only it, then stalled)
        b._extract_bar(bars[0])
        out = b._new_bars_from_raw(bars)
        # bars 1..4 must ALL be replayed, in order — not just the latest
        assert [d["close"] for d in out] == [7001.0, 7002.0, 7003.0, 7004.0]

    def test_no_gap_returns_single_new_bar(self):
        bars = _series(4)
        b = _bridge()
        for x in bars[:3]:
            b._extract_bar(x)
        out = b._new_bars_from_raw(bars)
        assert len(out) == 1 and out[0]["close"] == 7003.0

    def test_all_already_seen_returns_empty(self):
        bars = _series(4)
        b = _bridge()
        for x in bars:
            b._extract_bar(x)
        assert b._new_bars_from_raw(bars) == []

    def test_duplicates_are_deduped(self):
        bars = _series(3)
        b = _bridge(last_bar_time=pd.Timestamp("2026-06-29 10:29", tz="US/Eastern"))
        out = b._new_bars_from_raw(bars + bars)  # same bars twice
        assert len(out) == 3


class TestGetNewBarsExcludesFormingBar:
    """get_new_bars() must drop the still-forming last element (bars[-1])."""

    def test_polling_excludes_forming_bar(self):
        bars = _series(4)  # bars[-1] is the in-progress minute
        ib = SimpleNamespace(reqHistoricalData=lambda *a, **k: bars,
                             isConnected=lambda: True)
        b = IBBridge.__new__(IBBridge)
        b._ib = ib
        b._connected = True
        b._streaming_active = True
        b._use_polling = True
        b._contract = object()
        # Seed a watermark before the series so the multi-bar backfill path runs
        # (cold-start would otherwise return just the latest bar).
        b._last_bar_time = pd.Timestamp("2026-06-29 10:29", tz="US/Eastern")
        b._last_poll_time = 0.0
        b._poll_interval = 0
        b._zero_volume_count = 0
        b.config = SimpleNamespace(use_rth_only=False)
        out = b.get_new_bars()
        # 3 closed bars returned (forming bars[-1] excluded)
        assert [d["close"] for d in out] == [7000.0, 7001.0, 7002.0]
