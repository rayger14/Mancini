"""Tests for volume-aware front-month contract selection.

2026-06-16 incident: the bot was still trading the June micro (MESM6,
expiry 2026-06-18) while the market had rolled to September (MESU6). The
old auto-detect picked the *nearest non-expired* expiry and only rerolled
when the contract went zero-volume (dead) — so it clung to the expiring
contract through roll week, trading ~68 pts off the liquid September price.

_select_front_contract rolls on volume (and, when volume is unavailable,
on a calendar safety margin) so the bot tracks the liquid front month.
"""
from __future__ import annotations

from datetime import date

from live.ib_bridge import IBBridge


def _cand(symbol, expiry, volume):
    # The "contract" only needs to be an identity sentinel the selector
    # returns; a string stands in fine for these pure-logic tests.
    return {"contract": symbol, "expiry": expiry, "volume": volume}


class TestSelectFrontContract:
    def test_empty_returns_none(self):
        assert IBBridge._select_front_contract([]) is None

    def test_single_candidate_returned(self):
        c = [_cand("MESU6", "20260918", 1_000_000)]
        assert IBBridge._select_front_contract(c) == "MESU6"

    def test_picks_higher_volume_back_month_during_roll(self):
        # Roll week: September volume has overtaken June. Pick September.
        c = [
            _cand("MESM6", "20260618", 120_000),
            _cand("MESU6", "20260918", 980_000),
        ]
        assert IBBridge._select_front_contract(c) == "MESU6"

    def test_keeps_front_when_it_still_leads_volume(self):
        # Mid-quarter: front month still dominant. Don't roll early.
        c = [
            _cand("MESU6", "20260918", 1_400_000),
            _cand("MESZ6", "20261218", 60_000),
        ]
        assert IBBridge._select_front_contract(c) == "MESU6"

    def test_calendar_safety_roll_when_volume_unavailable(self):
        # Volume fetch failed (None). Front expires in 2 days -> roll anyway
        # rather than trade a dying contract. This is the live 2026-06-16 case.
        c = [
            _cand("MESM6", "20260618", None),
            _cand("MESU6", "20260918", None),
        ]
        assert IBBridge._select_front_contract(
            c, today=date(2026, 6, 16)) == "MESU6"

    def test_no_premature_calendar_roll_when_front_has_runway(self):
        # Volume unavailable but front has weeks left -> stay on front.
        c = [
            _cand("MESU6", "20260918", None),
            _cand("MESZ6", "20261218", None),
        ]
        assert IBBridge._select_front_contract(
            c, today=date(2026, 8, 1)) == "MESU6"

    def test_partial_volume_info_falls_back_to_calendar(self):
        # Only one side has volume -> can't compare -> calendar rule applies.
        c = [
            _cand("MESM6", "20260618", 120_000),
            _cand("MESU6", "20260918", None),
        ]
        assert IBBridge._select_front_contract(
            c, today=date(2026, 6, 16)) == "MESU6"
