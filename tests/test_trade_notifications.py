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
