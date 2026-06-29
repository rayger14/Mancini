"""Tests for real-time streaming verification (continuous-engine Layer 2).

The old start_streaming() verified real-time data with a single 2-second
reqMktData NaN check; that false-negatived even with a live CME subscription
and dropped the bot to 60s polling. _verify_realtime() polls the ticker until
a real tick arrives or it times out.
"""
from __future__ import annotations

from types import SimpleNamespace

from live.ib_bridge import IBBridge


def _bridge(ticker, sleep_effect=None):
    b = IBBridge.__new__(IBBridge)
    b._contract = object()
    calls = {"n": 0}

    def _sleep(secs):
        calls["n"] += 1
        if sleep_effect:
            sleep_effect(calls["n"], ticker)

    b._ib = SimpleNamespace(
        reqMktData=lambda *a, **k: ticker,
        sleep=_sleep,
        cancelMktData=lambda *a, **k: None,
    )
    return b


class TestVerifyRealtime:
    def test_returns_true_once_a_real_tick_arrives(self):
        ticker = SimpleNamespace(last=float("nan"))
        # tick becomes a real number on the 2nd sleep (data farm pushed a price)
        def effect(n, t):
            if n >= 2:
                t.last = 7000.25
        b = _bridge(ticker, sleep_effect=effect)
        assert b._verify_realtime(timeout_sec=5.0) is True

    def test_returns_false_when_no_tick_within_timeout(self):
        ticker = SimpleNamespace(last=float("nan"))  # never ticks
        b = _bridge(ticker)
        assert b._verify_realtime(timeout_sec=0.2) is False

    def test_cancels_market_data_subscription(self):
        ticker = SimpleNamespace(last=7000.0)
        cancelled = {"yes": False}
        b = _bridge(ticker)
        b._ib.cancelMktData = lambda *a, **k: cancelled.__setitem__("yes", True)
        b._verify_realtime(timeout_sec=1.0)
        assert cancelled["yes"] is True
