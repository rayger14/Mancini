"""Confidence profile (Phase 0): the evidence table behind the engine-
confidence badge and (later, gated) sizing.

Invariant under test everywhere: harvest-side and live-side derive the SAME
cell key from the same trade — key drift silently destroys calibration.
"""
from datetime import time

import pytest

from core.confidence import (
    ConfidenceTable,
    ProfileKey,
    confirmation_class,
    key_from_record,
    plan_match_class,
    session_window_class,
)


class TestClassifiers:
    def test_confirmation_class(self):
        assert confirmation_class("NON_ACCEPTANCE") == "non_acceptance"
        assert confirmation_class("ACCEPTANCE") == "acceptance"
        assert confirmation_class(None) == "other"
        assert confirmation_class("WEIRD") == "other"

    def test_session_window_class_bins(self):
        assert session_window_class(time(9, 45)) == "morning"
        assert session_window_class(time(10, 59)) == "morning"
        assert session_window_class(time(11, 0)) == "day"
        assert session_window_class(time(14, 30)) == "day"
        assert session_window_class(time(16, 55)) == "day"
        assert session_window_class(time(7, 15)) == "premarket"
        assert session_window_class(time(19, 30)) == "evening"
        assert session_window_class(time(17, 30)) == "evening"
        assert session_window_class(time(23, 30)) == "overnight"
        assert session_window_class(time(3, 0)) == "overnight"

    def test_session_window_matches_runner_details(self):
        # the harvest reads logged session_window DETAIL strings — the text
        # produced by IBRunner._get_session_window. Mapping parity check.
        from core.confidence import window_class_from_detail
        pairs = {
            "Morning Window (Prime)": "morning",
            "Midday": "day",
            "Chop Zone (Blocked) [BYPASS]": "day",
            "Afternoon (FB Only)": "day",
            "Pre-Market": "premarket",
            "Evening (Blocked 6-10PM ET) [BYPASS]": "evening",
            "Late Night Session": "overnight",
            "European Open (Blocked) [BYPASS]": "overnight",
        }
        for detail, expect in pairs.items():
            assert window_class_from_detail(detail) == expect, detail

    def test_plan_match_class(self):
        from types import SimpleNamespace
        plan = SimpleNamespace(planned_setups=[
            SimpleNamespace(level_price=7506.0, direction="long")])
        assert plan_match_class("CUSTOM", 7480.0, None) == "plan"
        assert plan_match_class("MANCINI_PLAN", 7480.0, None) == "plan"
        assert plan_match_class("INTRADAY_LOW", 7505.0, plan) == "plan"   # within 2.0
        assert plan_match_class("INTRADAY_LOW", 7490.0, plan) == "engine"
        assert plan_match_class("MULTI_HOUR_LOW", 7490.0, None) == "engine"


class TestKeyParity:
    def test_record_and_context_agree(self):
        rec = {
            "confirmation_type": "NON_ACCEPTANCE",
            "session_window": "Late Night Session",
            "level_type": "CUSTOM",
            "level_price": 7473.75,
        }
        k1 = key_from_record(rec, plan=None)
        assert k1 == ProfileKey("non_acceptance", "plan", "overnight")


class TestTable:
    def _table(self):
        cells = {
            "non_acceptance|plan|overnight": {"n": 40, "wins": 26, "avg_pnl": 22.0},
            "non_acceptance|plan": {"n": 60, "wins": 38, "avg_pnl": 21.0},
            "non_acceptance": {"n": 82, "wins": 51, "avg_pnl": 23.8},
        }
        glob = {"n": 284, "wins": 148, "avg_pnl": 16.9}
        return ConfidenceTable(cells=cells, global_stats=glob)

    def test_lookup_finest_cell(self):
        p = self._table().lookup(ProfileKey("non_acceptance", "plan", "overnight"))
        assert p.n == 40 and p.backoff_level == 0
        assert 0.55 < p.p_win < 0.70

    def test_backoff_to_parent(self):
        p = self._table().lookup(ProfileKey("non_acceptance", "plan", "day"))
        assert p.backoff_level == 1
        assert p.n == 60

    def test_backoff_to_global(self):
        p = self._table().lookup(ProfileKey("acceptance", "engine", "day"))
        assert p.backoff_level == 3
        assert p.n == 284

    def test_shrinkage_pulls_small_cells_toward_parent(self):
        cells = {
            "non_acceptance|plan|overnight": {"n": 5, "wins": 5, "avg_pnl": 40.0},
            "non_acceptance": {"n": 100, "wins": 50, "avg_pnl": 15.0},
        }
        t = ConfidenceTable(cells=cells,
                            global_stats={"n": 300, "wins": 150, "avg_pnl": 15.0})
        p = t.lookup(ProfileKey("non_acceptance", "plan", "overnight"))
        # raw 100% on n=5 must shrink well below 1.0 toward the 50% parent
        assert p.p_win < 0.85

    def test_null_table_returns_none(self):
        t = ConfidenceTable.load("/nonexistent/path.json")
        p = t.lookup(ProfileKey("non_acceptance", "plan", "overnight"))
        assert p.p_win is None

    def test_size_factor_tiers(self):
        from types import SimpleNamespace
        t = self._table()
        params = SimpleNamespace(confidence_full_size_pwin=0.60,
                                 confidence_half_size_pwin=0.50)
        assert t.size_factor(0.65, params) == 1.0
        assert t.size_factor(0.55, params) == 0.5
        assert t.size_factor(0.45, params) == 0.25
        assert t.size_factor(None, params) is None


class TestCumulativePnlFix:
    def test_final_exit_is_cumulative(self):
        from backtest.build_confidence_profile import net_pnl_for_trade
        rows = [
            {"event": "entry", "pnl_pts": None},
            {"event": "partial_exit", "pnl_pts": 10.0},
            {"event": "exit", "pnl_pts": 14.0},   # CUMULATIVE
        ]
        assert net_pnl_for_trade(rows) == 14.0    # never 24

    def test_single_exit_trade(self):
        from backtest.build_confidence_profile import net_pnl_for_trade
        rows = [{"event": "entry", "pnl_pts": None},
                {"event": "exit", "pnl_pts": -33.5}]
        assert net_pnl_for_trade(rows) == -33.5
