"""Tests for shorts-alert-only execution gating.

The P&L-at-targets report proved the existing short detectors are net-negative
(13/14 lose at targets, 0/14 ever hit T1, -445.8pt). So in production the bot
no longer places LIVE short orders — every short routes to the Discord
alert/shadow path instead. A genuinely new local-top short would be a separate,
re-validated feature.
"""
from __future__ import annotations

from datetime import datetime, time
from types import SimpleNamespace

import pytest

from live.ib_runner import IBRunner, route_short_to_alert_only


class TestRouteShortToAlertOnly:
    def test_short_with_flag_on_is_alert_only(self):
        assert route_short_to_alert_only("short", True) is True

    def test_short_with_flag_off_still_trades(self):
        assert route_short_to_alert_only("short", False) is False

    def test_long_is_never_alert_only(self):
        assert route_short_to_alert_only("long", True) is False

    def test_case_insensitive(self):
        assert route_short_to_alert_only("SHORT", True) is True


class TestEvaluateAndEnterShortGate:
    def _runner(self, *, alert_only: bool):
        phantoms = []
        runner = SimpleNamespace(
            strategy=SimpleNamespace(
                strategy_params=SimpleNamespace(shorts_alert_only=alert_only)),
            _add_phantom=lambda sig, reason, ts: phantoms.append(reason),
            bridge=SimpleNamespace(
                send_entry=lambda **k: pytest.fail("short must NOT place a live order")),
        )
        return runner, phantoms

    def _short_signal(self):
        return SimpleNamespace(
            direction="short",
            signal_type=SimpleNamespace(name="BREAKDOWN_SHORT"),
            entry_price=7485.0,
        )

    def test_short_alert_only_places_no_order(self):
        runner, phantoms = self._runner(alert_only=True)
        # Returns at the top gate before any execution path is touched.
        IBRunner._evaluate_and_enter(
            runner, self._short_signal(), time(21, 3), datetime(2026, 6, 30, 21, 3))
        assert "shorts_alert_only" in phantoms
