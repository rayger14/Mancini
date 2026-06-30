"""Tests for the in-process freeze watchdog.

On 2026-06-29 a resubscribe() that collided with a competing-login Error 162
hung a synchronous IB call and froze the single-threaded run loop — the bot
went totally silent for minutes and only a manual restart recovered it. A
daemon thread now force-exits the process when the main loop stops iterating,
letting Docker's restart-unless-stopped policy bring up a fresh, reconnected
bot. This catches ANY hang, not just that path.
"""
from __future__ import annotations

import time as _time

from live.ib_runner import IBRunner, _should_force_exit_frozen


class TestShouldForceExitFrozen:
    def test_fires_past_threshold(self):
        assert _should_force_exit_frozen(300.0, 240.0) is True

    def test_not_under_threshold(self):
        assert _should_force_exit_frozen(60.0, 240.0) is False

    def test_at_threshold(self):
        assert _should_force_exit_frozen(240.0, 240.0) is True

    def test_disabled_when_threshold_zero(self):
        # FREEZE_TIMEOUT_SEC=0 disables the watchdog; never exit.
        assert _should_force_exit_frozen(99999.0, 0.0) is False


class TestCheckFreeze:
    def _runner(self, idle, threshold=240.0):
        r = IBRunner.__new__(IBRunner)
        r._last_loop_progress = _time.monotonic() - idle
        r._freeze_timeout_sec = threshold
        state = {"exits": 0}
        r._freeze_exit = lambda: state.__setitem__("exits", state["exits"] + 1)
        return r, state

    def test_exits_when_loop_frozen(self):
        r, state = self._runner(idle=300.0)
        assert IBRunner._check_freeze(r) is True
        assert state["exits"] == 1

    def test_no_exit_while_progressing(self):
        r, state = self._runner(idle=5.0)
        assert IBRunner._check_freeze(r) is False
        assert state["exits"] == 0

    def test_disabled_threshold_never_exits(self):
        r, state = self._runner(idle=99999.0, threshold=0.0)
        assert IBRunner._check_freeze(r) is False
        assert state["exits"] == 0
