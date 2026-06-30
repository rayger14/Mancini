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


class TestEntryLevelSource:
    """The embed must say WHERE the level came from: Mancini's posted plan
    vs the engine's own price-action detection."""

    def _build(self, *, level_name, plan_setups, gate_bypass=None):
        sig = _Signal(pattern=_Pattern(level=_Lvl(price=7430.0,
                                                  level_type=_LvlType(name=level_name))))
        return build_entry_embed(
            position=_Position(entry_price=7430.0, stop_price=7389.75,
                               target_1=7459.25, target_2=7470.0),
            signal=sig,
            fill_price=7430.0,
            contracts_ordered=2,
            contract_spec=_Contract(),
            exit_params=SimpleNamespace(
                t1_exit_fraction=0.75, t2_exit_fraction=0.15,
                runner_fraction=0.10,
            ),
            plan=_Plan(planned_setups=plan_setups),
            session_date="2026-06-29",
            gate_bypass=gate_bypass,
        )["embeds"][0]

    def test_engine_detected_level_is_labeled(self):
        """INTRADAY_LOW with no plan match → clearly tagged engine-detected,
        with the human mechanism and a note it's not on his plan."""
        emb = self._build(level_name="INTRADAY_LOW", plan_setups=[])
        desc = emb["description"]
        assert "Source:" in desc
        assert "engine-detected" in desc.lower()
        assert "intraday flush low" in desc.lower()
        assert "not on" in desc.lower() and "plan" in desc.lower()

    def test_mancini_plan_match_is_labeled(self):
        """A level that matches one of his posted setups → tagged as on his
        plan, not engine noise."""
        setup = _PlanSetup(level_price=7430.0, conviction="high",
                           context="FB at 7430 — the obvious trade")
        emb = self._build(level_name="INTRADAY_LOW", plan_setups=[setup])
        desc = emb["description"]
        assert "Source:" in desc
        assert "Mancini" in desc and "plan" in desc.lower()
        # Should NOT call a plan-matched level engine noise
        assert "not on" not in desc.lower()


class TestEntryFBLogic:
    """The embed must say what KIND of failed breakdown fired, keyed off the
    reliable sweep_depth signature (fb_entry_path is broken — it tags every
    live FB 'elevator_fb' even on 30pt+ sweeps). A 0-sweep momentum entry must
    never read like a deep flush-and-reclaim."""

    def _build(self, *, sweep, conf_name="NON_ACCEPTANCE",
               sig_name="FAILED_BREAKDOWN"):
        pat = _Pattern(
            level=_Lvl(price=7395.75, level_type=_LvlType(name="INTRADAY_LOW")),
            confirmation=_Conf(name=conf_name),
            sweep_depth_pts=sweep,
        )
        sig = _Signal(signal_type=_SigType(name=sig_name), pattern=pat)
        return build_entry_embed(
            position=_Position(entry_price=7430.0, stop_price=7389.75),
            signal=sig, fill_price=7432.75, contracts_ordered=2,
            contract_spec=_Contract(),
            exit_params=SimpleNamespace(t1_exit_fraction=0.75,
                                        t2_exit_fraction=0.15, runner_fraction=0.10),
            plan=_Plan(planned_setups=[]), session_date="2026-06-29",
        )["embeds"][0]

    def test_zero_sweep_flagged_as_momentum_elevator(self):
        desc = self._build(sweep=0.0)["description"]
        assert "FB type:" in desc
        assert "elevator" in desc.lower() or "momentum" in desc.lower()
        assert "no breakdown" in desc.lower()
        assert "non-acceptance" in desc.lower()

    def test_midsweep_is_sweep_reclaim_with_depth(self):
        desc = self._build(sweep=8.5)["description"]
        assert "FB type:" in desc
        assert "sweep" in desc.lower() and "reclaim" in desc.lower()
        assert "8.5" in desc

    def test_deep_flush_flagged_as_high_quality(self):
        desc = self._build(sweep=36.0)["description"]
        assert "deep flush" in desc.lower()
        assert "36" in desc

    def test_shallow_sweep_labeled_shallow(self):
        desc = self._build(sweep=3.0)["description"]
        assert "shallow" in desc.lower()

    def test_non_fb_signal_has_no_fb_type_line(self):
        desc = self._build(sweep=0.0, sig_name="LEVEL_RECLAIM")["description"]
        assert "FB type:" not in desc


class TestEntryCollectionMode:
    """Collection-mode fills (production would skip them — wrong time window)
    must be visually unmistakable so they don't read as real signals."""

    def _build(self, gate_bypass):
        return build_entry_embed(
            position=_Position(),
            signal=_Signal(),
            fill_price=7430.0,
            contracts_ordered=2,
            contract_spec=_Contract(),
            exit_params=SimpleNamespace(
                t1_exit_fraction=0.75, t2_exit_fraction=0.15,
                runner_fraction=0.10,
            ),
            plan=_Plan(planned_setups=[]),
            session_date="2026-06-29",
            gate_bypass=gate_bypass,
        )["embeds"][0]

    def test_collection_mode_banner_and_gates(self):
        emb = self._build(["Evening block (17:00-22:00 ET)"])
        assert "COLLECTION MODE" in emb["description"]
        assert "Evening block (17:00-22:00 ET)" in emb["description"]
        # Title carries a marker too so it's obvious in the channel list
        assert "🧪" in emb["title"]

    def test_production_trade_has_no_collection_banner(self):
        emb = self._build(None)
        assert "COLLECTION MODE" not in emb["description"]
        assert "🧪" not in emb["title"]


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

    def test_collection_mode_exit_is_tagged(self):
        """An exit on a collection-mode trade must stay visually tagged so it
        doesn't read as a real T1/stop."""
        payload = build_exit_embed(
            phase="t1",
            fill_price=7459.25,
            contracts_closed=1,
            entry_price=7432.75,
            direction="long",
            contract_spec=_Contract(),
            remaining_contracts=1,
            realized_pnl_pts_so_far=26.5,
            reason="Target 1 hit",
            gate_bypass=["Evening block (17:00-22:00 ET)"],
        )
        emb = payload["embeds"][0]
        assert "🧪" in emb["title"]
        assert emb["color"] == 0x607D8B
        assert "COLLECTION" in emb["description"]

    def test_production_exit_not_tagged(self):
        payload = build_exit_embed(
            phase="t1", fill_price=7530.0, contracts_closed=3,
            entry_price=7517.5, direction="long", contract_spec=_Contract(),
            remaining_contracts=1, realized_pnl_pts_so_far=37.5,
        )
        emb = payload["embeds"][0]
        assert "🧪" not in emb["title"]
        assert "COLLECTION" not in emb["description"]

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
    """A GENUINE short trigger — survived all guards (the failed-bounce setup)."""
    ev = {
        "feature": "short_triggered",
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
    def test_short_triggered_event_qualifies(self):
        assert is_short_alert_event(_short_entry_event()) is True

    def test_capitulation_rejection_does_not_qualify(self):
        # capitulation_entry is a REJECTION log (bot faded the flush) — must
        # NOT alert even though it carries a short bracket. This is the bug fix.
        ev = _short_entry_event(feature="capitulation_entry")
        assert is_short_alert_event(ev) is False

    def test_move_exhaustion_does_not_qualify(self):
        ev = _short_entry_event(feature="move_exhaustion")
        assert is_short_alert_event(ev) is False

    def test_sizing_diagnostic_does_not_qualify(self):
        ev = {"feature": "sweep_depth", "signal_type": "BREAKDOWN_SHORT",
              "level_price": 7464.0, "sweep_depth_pts": 3.25}
        assert is_short_alert_event(ev) is False

    def test_shadow_outcome_does_not_qualify(self):
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
