"""Tests for the BAR_GAP threshold + IB Gateway reset window suppression.

The watchdog was alerting Discord every night at ~19:45 ET because of two
related issues:
  1. The 3-min bar gap threshold was tripped by the IB Gateway's normal
     nightly re-auth cycle (~2-4 min gap).
  2. There was no suppression window around the known gateway reset.

After this change:
  - MAX_BAR_GAP_SEC bumped to 5 min (300s)
  - Bar gap AND error spike alerts are suppressed between 19:40-19:55 ET
    on weekdays (the known IB Gateway re-auth window).
"""
from __future__ import annotations

from datetime import datetime, timedelta, time, timezone
from pathlib import Path

import pytest

from live.watchdog import (
    Watchdog,
    CRITICAL,
    HIGH,
)


_ET = timezone(timedelta(hours=-4))  # EDT — matches sane defaults


def _watchdog(tmp_path: Path) -> Watchdog:
    return Watchdog(
        log_path=str(tmp_path / "bot.log"),
        status_path=str(tmp_path / "status.json"),
        alerts_path=str(tmp_path / "watchdog_alerts.json"),
        poll_interval=1.0,
        webhook_url=None,  # no Discord during tests
    )


class TestBarGapThreshold:
    def test_default_threshold_is_5_minutes(self, tmp_path):
        wd = _watchdog(tmp_path)
        assert wd.MAX_BAR_GAP_SEC == 300, (
            "default raised from 180 → 300 so the IB Gateway nightly "
            "re-auth (~2-4 min gap) stops false-triggering CRITICAL alerts"
        )

    def test_gap_under_threshold_does_not_alert(self, tmp_path):
        wd = _watchdog(tmp_path)
        now = datetime(2026, 6, 3, 12, 0, 0, tzinfo=_ET)  # Wed noon
        wd._last_bar_time = now - timedelta(seconds=240)  # 4 min ago
        wd._last_bar_number = 100
        wd._check_bar_flow(now)
        assert "BAR_GAP" not in wd._active_alerts

    def test_gap_over_threshold_alerts(self, tmp_path):
        wd = _watchdog(tmp_path)
        now = datetime(2026, 6, 3, 12, 0, 0, tzinfo=_ET)  # Wed noon
        wd._last_bar_time = now - timedelta(seconds=360)  # 6 min ago
        wd._last_bar_number = 100
        wd._check_bar_flow(now)
        assert "BAR_GAP" in wd._active_alerts
        assert wd._active_alerts["BAR_GAP"].severity == CRITICAL


class TestGatewayResetWindow:
    """The 19:40-19:55 ET weekday window when IB Gateway re-authenticates."""

    def test_window_constants_are_set(self, tmp_path):
        wd = _watchdog(tmp_path)
        assert wd.GATEWAY_RESET_START == time(19, 40)
        assert wd.GATEWAY_RESET_END == time(19, 55)

    @pytest.mark.parametrize("h,m,expected", [
        (19, 39, False),  # just before
        (19, 40, True),   # start of window
        (19, 45, True),   # middle (actual crash time)
        (19, 55, True),   # end of window
        (19, 56, False),  # just after
        (12, 0, False),   # noon, definitely not
        (3, 30, False),   # 3:30am, definitely not
    ])
    def test_window_detection_on_weekday(self, tmp_path, h, m, expected):
        wd = _watchdog(tmp_path)
        now = datetime(2026, 6, 3, h, m, 0, tzinfo=_ET)  # Wednesday
        assert wd._is_in_gateway_reset_window(now) is expected

    def test_window_not_active_on_weekend(self, tmp_path):
        wd = _watchdog(tmp_path)
        # Saturday at 19:45 ET — within time window but NOT a weekday
        sat = datetime(2026, 6, 6, 19, 45, 0, tzinfo=_ET)
        assert wd._is_in_gateway_reset_window(sat) is False

    def test_bar_gap_suppressed_during_gateway_reset(self, tmp_path):
        """Wed 19:45 ET, 6-min gap — bot would normally CRITICAL but we suppress."""
        wd = _watchdog(tmp_path)
        now = datetime(2026, 6, 3, 19, 45, 0, tzinfo=_ET)
        wd._last_bar_time = now - timedelta(seconds=360)
        wd._last_bar_number = 100
        wd._check_bar_flow(now)
        assert "BAR_GAP" not in wd._active_alerts, (
            "gap during the 19:40-19:55 IB Gateway reset window must not alert"
        )

    def test_bar_gap_fires_outside_gateway_reset_window(self, tmp_path):
        """Wed 12:00 ET, 6-min gap — must still alert (real outage)."""
        wd = _watchdog(tmp_path)
        now = datetime(2026, 6, 3, 12, 0, 0, tzinfo=_ET)
        wd._last_bar_time = now - timedelta(seconds=360)
        wd._last_bar_number = 100
        wd._check_bar_flow(now)
        assert "BAR_GAP" in wd._active_alerts, (
            "gap outside the reset window must still produce a CRITICAL alert"
        )

    def test_error_spike_suppressed_during_gateway_reset(self, tmp_path):
        """Same window suppresses ERROR_SPIKE alerts."""
        wd = _watchdog(tmp_path)
        now = datetime(2026, 6, 3, 19, 45, 0, tzinfo=_ET)
        # 50 errors in the recent window — would normally trip MAX_ERRORS_PER_5MIN=10
        import time as _time
        wd._error_window.extend([_time.monotonic()] * 50)
        wd._check_error_rate(now)
        assert "ERROR_SPIKE" not in wd._active_alerts, (
            "error spike during IB Gateway re-auth is expected, must not alert"
        )

    def test_error_spike_fires_outside_gateway_reset(self, tmp_path):
        """50 errors at noon DOES trigger HIGH alert."""
        wd = _watchdog(tmp_path)
        now = datetime(2026, 6, 3, 12, 0, 0, tzinfo=_ET)
        import time as _time
        wd._error_window.extend([_time.monotonic()] * 50)
        wd._check_error_rate(now)
        assert "ERROR_SPIKE" in wd._active_alerts
        assert wd._active_alerts["ERROR_SPIKE"].severity == HIGH


class TestNoSignalsRthSpam:
    """NO_SIGNALS_RTH is a *persistent* condition (a quiet session can hold it
    for hours). The 5-min cooldown still let it re-fire every cooldown window,
    flooding Discord with dozens of identical warnings. It must be
    edge-triggered: alert once when it crosses, resolve when signals resume.
    """

    def test_fires_once_while_condition_persists(self, tmp_path):
        wd = _watchdog(tmp_path)
        wd._rth_bars_without_signal = wd.RTH_SIGNAL_CHECK_BARS + 10
        wd._signal_count_rth = 12

        wd._check_signal_pipeline()
        assert "NO_SIGNALS_RTH" in wd._active_alerts
        first = wd._active_alerts["NO_SIGNALS_RTH"]
        assert not first.resolved

        # Expire the cooldown so we prove the EDGE-TRIGGER (not the time
        # cooldown) is what suppresses the repeat.
        wd._last_alert_times["NO_SIGNALS_RTH"] -= 10_000
        wd._rth_bars_without_signal += 50  # condition still true, worse
        wd._check_signal_pipeline()

        # Same alert object — no re-emit while the dry spell persists.
        assert wd._active_alerts["NO_SIGNALS_RTH"] is first

    def test_resolves_when_signals_resume(self, tmp_path):
        wd = _watchdog(tmp_path)
        wd._rth_bars_without_signal = wd.RTH_SIGNAL_CHECK_BARS + 10
        wd._check_signal_pipeline()
        assert not wd._active_alerts["NO_SIGNALS_RTH"].resolved

        wd._rth_bars_without_signal = 0  # a signal arrived
        wd._check_signal_pipeline()
        assert wd._active_alerts["NO_SIGNALS_RTH"].resolved

    def test_refires_on_a_new_dry_spell(self, tmp_path):
        wd = _watchdog(tmp_path)
        wd._rth_bars_without_signal = wd.RTH_SIGNAL_CHECK_BARS + 10
        wd._check_signal_pipeline()
        first = wd._active_alerts["NO_SIGNALS_RTH"]

        wd._rth_bars_without_signal = 0
        wd._check_signal_pipeline()  # resolves
        wd._last_alert_times["NO_SIGNALS_RTH"] -= 10_000  # clear cooldown
        wd._rth_bars_without_signal = wd.RTH_SIGNAL_CHECK_BARS + 10
        wd._check_signal_pipeline()  # new dry spell -> new alert

        assert wd._active_alerts["NO_SIGNALS_RTH"] is not first
        assert not wd._active_alerts["NO_SIGNALS_RTH"].resolved
