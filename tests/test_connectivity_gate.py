"""Tests for connectivity-aware recovery (freeze prevention).

Both freezes on 2026-06-29 were triggered by our OWN recovery code firing a
blocking IB call DURING connection churn (a resubscribe during a login Error
162; a reconnect/re-qualify during an IBKR data-farm blip Error 1100/1102).
The bot now tracks connectivity via an errorEvent handler and defers recovery
while IBKR is mid-blip or just restored — so it doesn't hang a request in the
first place (the freeze watchdog still backstops a true hang).
"""
from __future__ import annotations

import time as _time
from types import SimpleNamespace

from live.ib_bridge import IBBridge
from live.ib_runner import recovery_blocked_by_connectivity


class TestRecoveryBlockedByConnectivity:
    def test_blocked_while_connectivity_down(self):
        assert recovery_blocked_by_connectivity(True, 9999.0, grace_sec=15.0) is True

    def test_blocked_within_grace_after_restore(self):
        assert recovery_blocked_by_connectivity(False, 5.0, grace_sec=15.0) is True

    def test_allowed_past_grace(self):
        assert recovery_blocked_by_connectivity(False, 30.0, grace_sec=15.0) is False

    def test_allowed_when_never_blipped(self):
        # restored timestamp 0 → huge seconds_since → not blocked
        assert recovery_blocked_by_connectivity(False, 1e9, grace_sec=15.0) is False


class TestOnErrorConnectivityTracking:
    def _bridge(self):
        b = IBBridge.__new__(IBBridge)
        b._connectivity_down = False
        b._connectivity_restored_monotonic = 0.0
        return b

    def test_1100_marks_connectivity_down(self):
        b = self._bridge()
        b._on_error(-1, 1100, "Connectivity lost")
        assert b._connectivity_down is True

    def test_1102_marks_restored_and_stamps_time(self):
        b = self._bridge()
        b._on_error(-1, 1100, "lost")
        b._on_error(-1, 1102, "restored")
        assert b._connectivity_down is False
        assert b._connectivity_restored_monotonic > 0.0

    def test_unrelated_error_does_not_change_connectivity(self):
        b = self._bridge()
        b._on_error(588, 10182, "Failed to request live updates")
        assert b._connectivity_down is False
        assert b._connectivity_restored_monotonic == 0.0


class TestPingBounded:
    def test_uses_async_when_available(self):
        # production path: reqCurrentTimeAsync exists → bounded run
        b = IBBridge.__new__(IBBridge)
        b._connected = True
        calls = {"ran": False}

        async def _areq():
            return object()

        def _run(coro, *a, **k):
            calls["ran"] = True
            import asyncio
            asyncio.new_event_loop().run_until_complete(coro)  # consume fully

        b._ib = SimpleNamespace(reqCurrentTimeAsync=_areq, run=_run)
        assert b.ping() is True
        assert calls["ran"] is True

    def test_falls_back_to_sync(self):
        # test/older path: no async variant → synchronous reqCurrentTime
        b = IBBridge.__new__(IBBridge)
        b._connected = True
        b._ib = SimpleNamespace(reqCurrentTime=lambda: object())
        assert b.ping() is True

    def test_false_when_not_connected(self):
        b = IBBridge.__new__(IBBridge)
        b._connected = False
        b._ib = SimpleNamespace(reqCurrentTime=lambda: object())
        assert b.ping() is False
