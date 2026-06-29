"""Tests for heartbeat-gated lighter reconnect (continuous-engine Layer 3).

When bars go stale, the bot now pings the socket (reqCurrentTime). If the
socket answers, only the data subscription is dead — re-subscribe (light,
keeps the position cache) instead of a full reconnect that wipes it (the
phantom-exit window). If the ping fails, fall back to the full reconnect.
"""
from __future__ import annotations

from types import SimpleNamespace

from live.ib_bridge import IBBridge
from live.ib_runner import _should_resubscribe


class TestShouldResubscribe:
    def _base(self, **over):
        kw = dict(minutes_since_bar=4.0, market_closed=False, connected=True,
                  socket_alive=True, seconds_since_last=600.0,
                  threshold_min=3.0, throttle_sec=300.0)
        kw.update(over)
        return _should_resubscribe(kw.pop("minutes_since_bar"), **kw)

    def test_fires_when_socket_alive_and_stale(self):
        assert self._base() is True

    def test_not_when_socket_dead(self):
        # ping failed → full reconnect path handles it, not resubscribe
        assert self._base(socket_alive=False) is False

    def test_not_when_market_closed(self):
        assert self._base(market_closed=True) is False

    def test_not_under_threshold(self):
        assert self._base(minutes_since_bar=2.0) is False

    def test_not_when_throttled(self):
        assert self._base(seconds_since_last=10.0) is False

    def test_not_when_disconnected(self):
        assert self._base(connected=False) is False


class TestPing:
    def test_true_when_ib_answers(self):
        b = IBBridge.__new__(IBBridge)
        b._connected = True
        b._ib = SimpleNamespace(reqCurrentTime=lambda: object())
        assert b.ping() is True

    def test_false_on_exception(self):
        b = IBBridge.__new__(IBBridge)
        b._connected = True
        def boom():
            raise ConnectionError("dead")
        b._ib = SimpleNamespace(reqCurrentTime=boom)
        assert b.ping() is False

    def test_false_when_not_connected(self):
        b = IBBridge.__new__(IBBridge)
        b._connected = False
        b._ib = SimpleNamespace(reqCurrentTime=lambda: object())
        assert b.ping() is False


class TestResubscribe:
    def test_restarts_stream_and_repopulates_positions(self):
        calls = []
        b = IBBridge.__new__(IBBridge)
        b.stop_streaming = lambda: calls.append("stop")
        b.start_streaming = lambda: calls.append("start")
        b._ib = SimpleNamespace(reqPositions=lambda: calls.append("reqPositions"))
        b.resubscribe("test")
        assert calls == ["stop", "start", "reqPositions"]
