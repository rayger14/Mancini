"""Tests for the rich Discord embed builders in live/trade_notifications.

Builders are pure functions: feed mock objects, assert the embed shape +
key text. Network is never touched in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from live.trade_notifications import (
    build_entry_embed,
    build_exit_embed,
    is_short_alert_event,
    short_alert_key,
    build_short_alert_embed,
    plan_short_match,
)


_ET = timezone(timedelta(hours=-4))


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _SigType:
    name: str


@dataclass
class _Conf:
    name: str


@dataclass
class _LvlType:
    name: str


@dataclass
class _Lvl:
    price: float = 7517.0
    level_type: _LvlType = field(default_factory=lambda: _LvlType(name="MULTI_HOUR_LOW"))


@dataclass
class _Pattern:
    level: _Lvl = field(default_factory=_Lvl)
    confirmation: _Conf = field(default_factory=lambda: _Conf(name="ACCEPTANCE"))
    sweep_depth_pts: float = 8.0
    pattern_type: str = "failed_breakdown"


@dataclass
class _Signal:
    signal_type: _SigType = field(default_factory=lambda: _SigType(name="FAILED_BREAKDOWN"))
    pattern: _Pattern = field(default_factory=_Pattern)
    target_1: float = 7530.0
    target_2: float = 7541.0
    rr_ratio_t1: float = 2.8
    direction: str = "long"


@dataclass
class _Position:
    direction: str = "long"
    entry_price: float = 7517.5
    stop_price: float = 7513.0
    target_1: float = 7530.0
    target_2: float = 7541.0
    remaining_contracts: int = 4
    realized_pnl_pts: float = 0.0


@dataclass
class _PlanSetup:
    setup_type: str = "failed_breakdown"
    direction: str = "long"
    level_price: float = 7517.0
    conviction: str = "high"
    context: str = "FB of massive multi-touch shelf since last Tuesday"


@dataclass
class _Plan:
    planned_setups: list = field(default_factory=list)


@dataclass
class _Contract:
    point_value: float = 5.0
    symbol: str = "MES"


# ---------------------------------------------------------------------------
# Entry embed
# ---------------------------------------------------------------------------


class TestEntryEmbed:
    def test_long_with_plan_match_20_contracts(self):
        """20 contracts gives clean splits: T1=15, T2=3, runner=2. All
        three fields should appear."""
        plan = _Plan(planned_setups=[_PlanSetup()])
        payload = build_entry_embed(
            position=_Position(remaining_contracts=20),
            signal=_Signal(),
            fill_price=7517.5,
            contracts_ordered=20,
            contract_spec=_Contract(),
            exit_params=SimpleNamespace(
                t1_exit_fraction=0.75,
                t2_exit_fraction=0.15,
                runner_fraction=0.10,
            ),
            plan=plan,
            session_date="2026-05-29",
            entry_time=datetime(2026, 5, 29, 12, 34, tzinfo=_ET),
        )
        emb = payload["embeds"][0]
        # Title shows side + entry price + size + conviction badge
        assert "🟢" in emb["title"]
        assert "LONG" in emb["title"]
        assert "7517" in emb["title"]
        assert "20 MES" in emb["title"]
        assert "HIGH conviction" in emb["title"]
        # Description includes plan match line
        assert "Mancini plan" in emb["description"]
        assert "7517" in emb["description"]
        assert "MULTI_HOUR_LOW" in emb["description"]
        # Fields include stop + T1 + T2 + runner + R:R + time + plan-date
        names = {f["name"] for f in emb["fields"]}
        assert "⛔ Stop" in names
        assert "🎯 T1 (75%)" in names
        assert "🎯 T2 (15%)" in names
        assert "🏃 Runner (10%)" in names
        assert "📊 R:R" in names
        assert "⏰ Entry" in names

    def test_long_4_contracts_omits_t2_field(self):
        """With 4 contracts, floor(4*0.15)=0 — T2 closes 0 contracts. The
        embed should NOT show a T2 field because nothing will fire."""
        payload = build_entry_embed(
            position=_Position(),
            signal=_Signal(),
            fill_price=7517.5,
            contracts_ordered=4,
            contract_spec=_Contract(),
            exit_params=SimpleNamespace(
                t1_exit_fraction=0.75, t2_exit_fraction=0.15,
                runner_fraction=0.10,
            ),
            plan=_Plan(),
            session_date="2026-05-29",
        )
        emb = payload["embeds"][0]
        names = {f["name"] for f in emb["fields"]}
        assert "🎯 T1 (75%)" in names
        assert "🏃 Runner (10%)" in names
        # T2 omitted because qty == 0
        assert "🎯 T2 (15%)" not in names

    def test_long_no_plan_match(self):
        payload = build_entry_embed(
            position=_Position(),
            signal=_Signal(),
            fill_price=7517.5,
            contracts_ordered=4,
            contract_spec=_Contract(),
            exit_params=SimpleNamespace(
                t1_exit_fraction=0.75, t2_exit_fraction=0.15,
                runner_fraction=0.10,
            ),
            plan=_Plan(planned_setups=[]),  # empty plan
            session_date="2026-05-29",
        )
        emb = payload["embeds"][0]
        # Title has no conviction badge
        assert "conviction" not in emb["title"].lower()
        # Description does not have plan-match block
        assert "Mancini plan" not in emb["description"]

    def test_two_contracts_only_t1_and_runner(self):
        """With 2 contracts, math.floor(2 * 0.75) = 1 (T1), runner=1.
        T2 quantity = floor(2 * 0.15) = 0 — should not appear as field."""
        payload = build_entry_embed(
            position=_Position(),
            signal=_Signal(),
            fill_price=7517.5,
            contracts_ordered=2,
            contract_spec=_Contract(),
            exit_params=SimpleNamespace(
                t1_exit_fraction=0.75, t2_exit_fraction=0.15,
                runner_fraction=0.10,
            ),
            plan=_Plan(planned_setups=[]),
            session_date="2026-05-29",
        )
        emb = payload["embeds"][0]
        names = {f["name"] for f in emb["fields"]}
        assert "🎯 T1 (75%)" in names
        assert "🏃 Runner (10%)" in names
        # T2 quantity is zero so the field is omitted
        assert "🎯 T2 (15%)" not in names

    def test_short_uses_red_color(self):
        pos = _Position(direction="short", stop_price=7530.0, target_1=7510.0)
        payload = build_entry_embed(
            position=pos,
            signal=_Signal(
                signal_type=_SigType(name="BREAKDOWN_SHORT"),
                pattern=_Pattern(),
                target_1=7510.0,
                target_2=7500.0,
                rr_ratio_t1=2.0,
                direction="short",
            ),
            fill_price=7520.0,
            contracts_ordered=4,
            contract_spec=_Contract(),
            exit_params=SimpleNamespace(
                t1_exit_fraction=0.75, t2_exit_fraction=0.15,
                runner_fraction=0.10,
            ),
            plan=_Plan(),
            session_date="2026-05-29",
        )
        emb = payload["embeds"][0]
        assert "🔴" in emb["title"]
        assert "SHORT" in emb["title"]


# ---------------------------------------------------------------------------
# Exit embed
# ---------------------------------------------------------------------------


class TestExitEmbed:
    def test_t1_fill_shows_locked_profit_and_remaining(self):
        payload = build_exit_embed(
            phase="t1",
            fill_price=7530.0,
            contracts_closed=3,
            entry_price=7517.5,
            direction="long",
            contract_spec=_Contract(),
            remaining_contracts=1,
            realized_pnl_pts_so_far=37.5,
            new_stop=7515.0,
            next_target=7541.0,
            reason="Target 1 hit (7530.00)",
            fill_time=datetime(2026, 5, 29, 12, 47, tzinfo=_ET),
        )
        emb = payload["embeds"][0]
        assert "T1 FILLED" in emb["title"]
        assert "3 of 4" in emb["title"]
        assert "7530" in emb["title"]
        assert "Locked +12.5 pt × 3" in emb["description"]
        assert "1 contract(s) remaining" in emb["description"]
        assert "7515" in emb["description"]
        assert "7541" in emb["description"]
        assert "Trade P&L so far" in emb["description"]

    def test_stop_hit_shows_red(self):
        payload = build_exit_embed(
            phase="stop",
            fill_price=7513.0,
            contracts_closed=4,
            entry_price=7517.5,
            direction="long",
            contract_spec=_Contract(),
            remaining_contracts=0,
            realized_pnl_pts_so_far=-18.0,
            reason="Stop loss hit",
        )
        emb = payload["embeds"][0]
        assert "STOP HIT" in emb["title"]
        assert emb["color"] == 0xE74C3C
        assert "fully closed" in emb["description"].lower()

    def test_runner_stopped_shows_final_summary(self):
        payload = build_exit_embed(
            phase="runner_trail",
            fill_price=7548.0,
            contracts_closed=1,
            entry_price=7517.5,
            direction="long",
            contract_spec=_Contract(),
            remaining_contracts=0,
            realized_pnl_pts_so_far=91.5,
            reason="Trailing stop hit",
        )
        emb = payload["embeds"][0]
        assert "RUNNER STOPPED" in emb["title"]
        assert "fully closed" in emb["description"].lower()


# ---------------------------------------------------------------------------
# Short heads-up alerts (shadow shorts the bot detects but does NOT trade)
# ---------------------------------------------------------------------------


def _short_entry_event(**over):
    """A live shadow-short entry event (the actionable kind)."""
    ev = {
        "feature": "capitulation_entry",
        "bar_idx": 159,
        "timestamp": "2026-06-24 20:41:00-04:00",
        "signal_type": "BREAKDOWN_SHORT",
        "entry_price": 7455.75,
        "stop_price": 7468.25,
        "target_1": 7439.0,
        "direction": "short",
        "level_price": 7459.0,
    }
    ev.update(over)
    return ev


class TestIsShortAlertEvent:
    def test_short_entry_event_qualifies(self):
        assert is_short_alert_event(_short_entry_event()) is True

    def test_sizing_diagnostic_does_not_qualify(self):
        # sweep_depth has no entry/stop and no direction — pure sizing telemetry
        ev = {"feature": "sweep_depth", "signal_type": "BREAKDOWN_SHORT",
              "level_price": 7464.0, "sweep_depth_pts": 3.25}
        assert is_short_alert_event(ev) is False

    def test_shadow_outcome_does_not_qualify(self):
        # An outcome record (target/stop resolved) is a result, not a new setup
        ev = _short_entry_event(event="shadow_outcome", outcome="timeout")
        assert is_short_alert_event(ev) is False

    def test_long_entry_does_not_qualify(self):
        assert is_short_alert_event(_short_entry_event(direction="long")) is False

    def test_missing_bracket_does_not_qualify(self):
        ev = _short_entry_event()
        ev.pop("stop_price")
        assert is_short_alert_event(ev) is False


class TestShortAlertKey:
    def test_same_setup_dedupes(self):
        # Consecutive bars nudge entry by <1pt — same setup, one alert
        a = short_alert_key(_short_entry_event(entry_price=7455.75))
        b = short_alert_key(_short_entry_event(entry_price=7455.50))
        assert a == b

    def test_different_level_is_distinct(self):
        a = short_alert_key(_short_entry_event(entry_price=7455.75))
        b = short_alert_key(_short_entry_event(entry_price=7399.0))
        assert a != b

    def test_different_signal_type_is_distinct(self):
        a = short_alert_key(_short_entry_event(signal_type="BREAKDOWN_SHORT"))
        b = short_alert_key(_short_entry_event(signal_type="BACKTEST_SHORT"))
        assert a != b


class TestPlanShortMatch:
    """Only alert on shorts that line up with a level Mancini actually called
    as a short setup — so alerts feel real and don't fire on every shadow flush."""

    def _plan(self):
        return SimpleNamespace(planned_setups=[
            SimpleNamespace(level_price=7399.0, setup_type="breakdown_short",
                            direction="short", conviction="low",
                            context="Bear case begins below 7399."),
            SimpleNamespace(level_price=7408.0, setup_type="failed_breakdown",
                            direction="long", conviction="medium", context="FB long."),
        ])

    def test_matches_short_setup_within_tolerance(self):
        # 7393 trigger is ~6pt below his 7399 short — should match
        m = plan_short_match(self._plan(), 7393.0, tol=8.0)
        assert m is not None and m.level_price == 7399.0

    def test_ignores_long_setups(self):
        # price right at his 7408 LONG level — must NOT match (it's a long)
        assert plan_short_match(self._plan(), 7408.0, tol=8.0) is None

    def test_no_match_when_far(self):
        assert plan_short_match(self._plan(), 7460.0, tol=8.0) is None

    def test_none_plan_is_safe(self):
        assert plan_short_match(None, 7399.0) is None


class TestBuildShortAlertEmbed:
    def test_embed_shape_and_disclaimer(self):
        emb = build_short_alert_embed(_short_entry_event(), symbol="MES")
        assert emb["color"] == 0xE74C3C  # red
        assert "SHORT SETUP" in emb["title"]
        assert "MES" in emb["title"]
        desc = emb["description"]
        # the bracket is shown
        assert "7468" in desc  # stop
        assert "7439" in desc  # target
        # unambiguous it is NOT a bot order
        assert "not" in desc.lower() and "order" in desc.lower()

    def test_plan_context_quoted_when_matched(self):
        plan = SimpleNamespace(planned_setups=[SimpleNamespace(
            level_price=7459.0, setup_type="breakdown_short",
            direction="short", conviction="low",
            context="Bear case begins below 7399.")])
        emb = build_short_alert_embed(_short_entry_event(), symbol="MES", plan=plan)
        assert "Bear case" in emb["description"]
