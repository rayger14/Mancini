"""Tests for floor-trader pivot points (Phase 2 — third confluence source).

Pivots are computed from the prior session's High/Low/Close and act as a
third independent source: a pivot that coincides with an engine or Mancini
level boosts that level's confluence; standalone pivots stay weak.
"""
from __future__ import annotations

from datetime import datetime

from config.levels import Level, LevelStore, LevelType
from core.pivots import compute_pivots, build_pivot_levels


class TestComputePivots:
    # H=110, L=90, C=100 → PP=100, clean round numbers.
    def test_pivot_point(self):
        p = compute_pivots(110.0, 90.0, 100.0)
        assert p["pp"] == 100.0

    def test_r1_s1(self):
        p = compute_pivots(110.0, 90.0, 100.0)
        assert p["r1"] == 110.0   # 2*PP - L
        assert p["s1"] == 90.0    # 2*PP - H

    def test_r2_s2(self):
        p = compute_pivots(110.0, 90.0, 100.0)
        assert p["r2"] == 120.0   # PP + (H-L)
        assert p["s2"] == 80.0    # PP - (H-L)

    def test_r3_s3(self):
        p = compute_pivots(110.0, 90.0, 100.0)
        assert p["r3"] == 130.0   # H + 2*(PP-L)
        assert p["s3"] == 70.0    # L - 2*(H-PP)

    def test_all_seven_levels_present(self):
        p = compute_pivots(110.0, 90.0, 100.0)
        assert set(p) == {"pp", "r1", "r2", "r3", "s1", "s2", "s3"}

    def test_realistic_es_values(self):
        # Sanity on ES-scale numbers: pivots ordered S3<S2<S1<PP<R1<R2<R3.
        p = compute_pivots(7470.0, 7410.0, 7450.0)
        vals = [p[k] for k in ("s3", "s2", "s1", "pp", "r1", "r2", "r3")]
        assert vals == sorted(vals)


class TestBuildPivotLevels:
    def test_builds_seven_levels(self):
        ts = datetime(2026, 6, 24, 18, 0)
        levels = build_pivot_levels(110.0, 90.0, 100.0, created_at=ts)
        assert len(levels) == 7
        assert all(isinstance(l, Level) for l in levels)
        assert all(l.level_type == LevelType.PIVOT for l in levels)
        assert all(l.confirmed_at == ts for l in levels)

    def test_levels_labeled_by_name(self):
        ts = datetime(2026, 6, 24, 18, 0)
        levels = build_pivot_levels(110.0, 90.0, 100.0, created_at=ts)
        labels = {l.label for l in levels}
        assert any("R1" in lb for lb in labels)
        assert any("PP" in lb for lb in labels)
        assert any("S2" in lb for lb in labels)

    def test_pivot_base_score_is_low(self):
        # A standalone pivot should be weak (it only matters via confluence).
        from config.levels import _LEVEL_BASE_SCORES
        assert _LEVEL_BASE_SCORES.get(LevelType.PIVOT, 99) <= 1


# ── Injection wiring into initialize_levels ──────────────────────────

import pandas as pd  # noqa: E402

from config.settings import StrategyParams  # noqa: E402
from core.signals import SignalAggregator  # noqa: E402


def _prior_df():
    idx = pd.date_range("2026-06-23 09:30", periods=3, freq="1min", tz="US/Eastern")
    return pd.DataFrame({
        "open": [7450, 7460, 7455], "high": [7460, 7470, 7465],
        "low": [7440, 7450, 7445], "close": [7455, 7465, 7460],
        "volume": [100, 100, 100],
    }, index=idx)  # high.max=7470 low.min=7440 close.iat[-1]=7460


class TestPivotInjectionWiring:
    def test_pivots_injected_when_flag_on(self):
        agg = SignalAggregator(strategy_params=StrategyParams(use_pivot_levels=True))
        pdf = _prior_df()
        agg.initialize_levels(pdf, prior_day_df=pdf)
        pivots = agg.level_store.get_active(LevelType.PIVOT)
        assert len(pivots) == 7

    def test_no_pivots_when_flag_off(self):
        agg = SignalAggregator(strategy_params=StrategyParams(use_pivot_levels=False))
        pdf = _prior_df()
        agg.initialize_levels(pdf, prior_day_df=pdf)
        assert agg.level_store.get_active(LevelType.PIVOT) == []
