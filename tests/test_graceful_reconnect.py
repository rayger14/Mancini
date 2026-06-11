"""Tests for graceful IB reconnect in the main loop.

The nightly IB Gateway re-auth at ~19:45 ET drops the socket; the
underlying ib_async layer raises ConnectionError out of `bridge.sleep`.
Previously this bubbled to main() and killed the process — Docker's
restart-unless-stopped saved us but with 2-5 min of bar loss (and
tonight a real FB setup at 7538 was missed because the flush happened
inside the restart gap).

This test verifies:
  - bridge.sleep flags the bridge for reconnect on ConnectionError
  - bridge.sleep re-raises so the caller knows to back off
  - the main-loop catch path triggers reconnect via the existing
    _needs_reconnect machinery
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from live.ib_bridge import IBBridge


class _FakeIB:
    """Minimal IB stub that lets us pre-program the sleep behavior."""

    def __init__(self):
        self.sleep_side_effects: list = []
        self.sleep_calls: int = 0

    def sleep(self, secs):
        self.sleep_calls += 1
        if self.sleep_side_effects:
            effect = self.sleep_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect


def _make_bridge() -> tuple[IBBridge, _FakeIB]:
    bridge = IBBridge.__new__(IBBridge)
    bridge._ib = _FakeIB()
    bridge._connected = True
    bridge._needs_reconnect = False
    return bridge, bridge._ib


class TestBridgeSleepReconnectFlag:
    def test_clean_sleep_does_nothing_special(self):
        bridge, ib = _make_bridge()
        bridge.sleep(1.0)
        assert ib.sleep_calls == 1
        assert bridge._connected is True
        assert bridge._needs_reconnect is False

    def test_connection_error_flags_reconnect(self):
        bridge, ib = _make_bridge()
        ib.sleep_side_effects = [ConnectionError("Socket disconnect")]
        with pytest.raises(ConnectionError):
            bridge.sleep(1.0)
        # After the error, the bridge has flagged itself for reconnect
        assert bridge._connected is False
        assert bridge._needs_reconnect is True

    def test_connection_error_when_already_disconnected_does_not_double_flag(self):
        """If we're already in the disconnected state, the flag stays True
        and we don't crash."""
        bridge, ib = _make_bridge()
        bridge._connected = False
        bridge._needs_reconnect = True
        ib.sleep_side_effects = [ConnectionError("Socket disconnect")]
        with pytest.raises(ConnectionError):
            bridge.sleep(1.0)
        assert bridge._connected is False
        assert bridge._needs_reconnect is True

    def test_unrelated_error_does_not_flag_reconnect(self):
        """A non-ConnectionError exception must NOT toggle the reconnect
        flag — that would mask the real problem."""
        bridge, ib = _make_bridge()
        ib.sleep_side_effects = [ValueError("bad arg")]
        with pytest.raises(ValueError):
            bridge.sleep(1.0)
        # Connection state is untouched
        assert bridge._connected is True
        assert bridge._needs_reconnect is False


class TestReconnectNeverGivesUp:
    """The IB Gateway's daily restart (19:45 ET) takes minutes; the bot's
    5-attempt burst lasts ~2.5 min and then cleared _needs_reconnect —
    permanently blind until a manual container restart (two nights in a
    row, 2026-06-09 and 06-10). Exhaustion must keep the flag set and
    retry in backed-off bursts."""

    def _bridge(self):
        from types import SimpleNamespace
        bridge = IBBridge.__new__(IBBridge)
        fake = MagicMock()
        fake.connect.side_effect = ConnectionRefusedError("gateway restarting")
        fake.isConnected.return_value = False
        bridge._ib = fake
        bridge._connected = False
        bridge._needs_reconnect = True
        bridge._streaming_active = False
        bridge._reconnect_backoff_until = 0.0
        bridge._reconnect_exhausted_logged = False
        bridge.config = SimpleNamespace(
            max_reconnect_attempts=2, reconnect_delay_sec=0.0,
            host="x", port=1, client_id=9,
        )
        return bridge, fake

    def test_exhausted_burst_keeps_reconnect_flag(self):
        bridge, fake = self._bridge()
        assert bridge.check_reconnect() is False
        assert bridge._needs_reconnect is True, (
            "clearing the flag after one failed burst bricks the bot until "
            "manual restart — it must keep retrying"
        )

    def test_backoff_prevents_hot_loop_then_retry_succeeds(self):
        bridge, fake = self._bridge()
        bridge.check_reconnect()  # burst 1 fails, backoff armed
        n = fake.connect.call_count
        assert bridge.check_reconnect() is False
        assert fake.connect.call_count == n, "no attempts during backoff"
        # backoff expires; gateway is back
        bridge._reconnect_backoff_until = 0.0
        fake.connect.side_effect = None
        fake.isConnected.return_value = True
        bridge._qualify_contract = lambda: object()
        assert bridge.check_reconnect() is True
        assert bridge._needs_reconnect is False
