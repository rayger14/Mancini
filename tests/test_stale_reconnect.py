"""Tests for stale-bar auto-recovery (force-reconnect when bars go stale while
the socket still looks connected)."""

from __future__ import annotations

from live.ib_runner import _should_force_reconnect


def test_stale_open_connected_triggers():
    # bars stale past threshold, market open, socket up, not throttled -> force
    assert _should_force_reconnect(
        7.0, market_closed=False, connected=True,
        seconds_since_last_force=999.0) is True


def test_market_closed_does_not_trigger():
    # the daily break / weekend: no bars is expected
    assert _should_force_reconnect(
        30.0, market_closed=True, connected=True,
        seconds_since_last_force=999.0) is False


def test_not_connected_does_not_trigger():
    # genuine socket disconnect -> normal reconnect path handles it, not this
    assert _should_force_reconnect(
        30.0, market_closed=False, connected=False,
        seconds_since_last_force=999.0) is False


def test_under_threshold_does_not_trigger():
    assert _should_force_reconnect(
        4.0, market_closed=False, connected=True,
        seconds_since_last_force=999.0) is False


def test_throttled_does_not_retrigger():
    # forced recently -> wait out the throttle before forcing again
    assert _should_force_reconnect(
        20.0, market_closed=False, connected=True,
        seconds_since_last_force=10.0) is False
