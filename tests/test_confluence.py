"""Tests for level confluence scoring."""

from __future__ import annotations

from datetime import datetime

import pytest

from config.levels import Level, LevelStore, LevelType, compute_confluence_score
from config.settings import StrategyParams


# ---------------------------------------------------------------------------
# compute_confluence_score unit tests
# ---------------------------------------------------------------------------


def _make_level(
    price: float,
    level_type: LevelType,
    touch_count: int = 1,
    rally_from_low_pts: float = 0.0,
    tested_and_held: bool = False,
) -> Level:
    """Helper to build a Level with sensible defaults."""
    ts = datetime(2024, 1, 15, 9, 30)
    return Level(
        price=price,
        level_type=level_type,
        created_at=ts,
        confirmed_at=ts,
        touch_count=touch_count,
        rally_from_low_pts=rally_from_low_pts,
        tested_and_held=tested_and_held,
    )


class TestConfluenceScoreBase:
    """Base score from level type alone (no nearby levels, no bonuses)."""

    def test_pdl_base_score(self):
        level = _make_level(5000.0, LevelType.PRIOR_DAY_LOW)
        assert compute_confluence_score(level, [level]) == 5

    def test_mhl_base_score(self):
        level = _make_level(5000.0, LevelType.MULTI_HOUR_LOW)
        assert compute_confluence_score(level, [level]) == 3

    def test_swing_low_base_score(self):
        level = _make_level(5000.0, LevelType.SWING_LOW)
        assert compute_confluence_score(level, [level]) == 2

    def test_cluster_low_base_score(self):
        level = _make_level(5000.0, LevelType.CLUSTER_LOW)
        assert compute_confluence_score(level, [level]) == 1

    def test_intraday_low_base_score(self):
        level = _make_level(5000.0, LevelType.INTRADAY_LOW)
        assert compute_confluence_score(level, [level]) == 2

    def test_horizontal_sr_base_score(self):
        level = _make_level(5000.0, LevelType.HORIZONTAL_SR)
        assert compute_confluence_score(level, [level]) == 1


class TestConfluenceProximity:
    """Nearby levels of different type add +2 each."""

    def test_one_nearby_different_type(self):
        pdl = _make_level(5000.0, LevelType.PRIOR_DAY_LOW)
        cluster = _make_level(5002.0, LevelType.CLUSTER_LOW)
        # PDL base=5, +2 for nearby CLUSTER_LOW within 3 pts
        assert compute_confluence_score(pdl, [pdl, cluster], proximity=3.0) == 7

    def test_two_nearby_different_types(self):
        pdl = _make_level(5000.0, LevelType.PRIOR_DAY_LOW)
        cluster = _make_level(5001.0, LevelType.CLUSTER_LOW)
        swing = _make_level(5002.5, LevelType.SWING_LOW)
        # PDL base=5, +2 for cluster, +2 for swing = 9
        assert compute_confluence_score(pdl, [pdl, cluster, swing], proximity=3.0) == 9

    def test_nearby_same_type_no_bonus(self):
        pdl1 = _make_level(5000.0, LevelType.PRIOR_DAY_LOW)
        pdl2 = _make_level(5001.0, LevelType.PRIOR_DAY_LOW)
        # Same type -> no confluence bonus
        assert compute_confluence_score(pdl1, [pdl1, pdl2], proximity=3.0) == 5

    def test_far_level_no_bonus(self):
        pdl = _make_level(5000.0, LevelType.PRIOR_DAY_LOW)
        cluster = _make_level(5010.0, LevelType.CLUSTER_LOW)
        # 10 pts apart > 3 pts proximity -> no bonus
        assert compute_confluence_score(pdl, [pdl, cluster], proximity=3.0) == 5

    def test_inactive_level_ignored(self):
        pdl = _make_level(5000.0, LevelType.PRIOR_DAY_LOW)
        cluster = _make_level(5001.0, LevelType.CLUSTER_LOW)
        cluster.is_active = False
        # Inactive levels should not count
        assert compute_confluence_score(pdl, [pdl, cluster], proximity=3.0) == 5


class TestConfluenceBonuses:
    """Touch count, rally, and tested_and_held bonuses."""

    def test_touch_count_bonus(self):
        level = _make_level(5000.0, LevelType.CLUSTER_LOW, touch_count=3)
        # base=1, +1 for touch_count>=3
        assert compute_confluence_score(level, [level]) == 2

    def test_touch_count_below_threshold(self):
        level = _make_level(5000.0, LevelType.CLUSTER_LOW, touch_count=2)
        # base=1, no bonus
        assert compute_confluence_score(level, [level]) == 1

    def test_rally_bonus(self):
        level = _make_level(5000.0, LevelType.MULTI_HOUR_LOW, rally_from_low_pts=25.0)
        # base=3, +1 for rally>=20
        assert compute_confluence_score(level, [level]) == 4

    def test_rally_below_threshold(self):
        level = _make_level(5000.0, LevelType.MULTI_HOUR_LOW, rally_from_low_pts=15.0)
        # base=3, no rally bonus
        assert compute_confluence_score(level, [level]) == 3

    def test_tested_and_held_bonus(self):
        level = _make_level(5000.0, LevelType.SWING_LOW, tested_and_held=True)
        # base=2, +1 for tested_and_held
        assert compute_confluence_score(level, [level]) == 3

    def test_all_bonuses_combined(self):
        level = _make_level(
            5000.0,
            LevelType.PRIOR_DAY_LOW,
            touch_count=5,
            rally_from_low_pts=30.0,
            tested_and_held=True,
        )
        # base=5, +1 touch, +1 rally, +1 tested = 8
        assert compute_confluence_score(level, [level]) == 8


class TestMonsterLevel:
    """The 'monster level' scenario: PDL + shelf + cluster within 3 pts."""

    def test_monster_level_score(self):
        pdl = _make_level(5000.0, LevelType.PRIOR_DAY_LOW, touch_count=4,
                          rally_from_low_pts=25.0, tested_and_held=True)
        cluster = _make_level(5001.5, LevelType.CLUSTER_LOW, touch_count=5)
        shelf = _make_level(5002.0, LevelType.SWING_LOW, touch_count=3)
        all_levels = [pdl, cluster, shelf]
        score = compute_confluence_score(pdl, all_levels, proximity=3.0)
        # base=5, +2 cluster, +2 swing, +1 touch, +1 rally, +1 tested = 12
        assert score == 12

    def test_noise_cluster_low_score(self):
        """Single CLUSTER_LOW with 1 touch = noise (score 1)."""
        cluster = _make_level(5000.0, LevelType.CLUSTER_LOW, touch_count=1)
        assert compute_confluence_score(cluster, [cluster]) == 1


# ---------------------------------------------------------------------------
# Integration: confluence gating in SignalAggregator._qualify_signal
# ---------------------------------------------------------------------------


class TestConfluenceGating:
    """Verify the config flag gates signals through _qualify_signal."""

    def _make_pattern_signal(self, level: Level):
        """Build a minimal PatternSignal for testing."""
        from core.patterns import PatternSignal, ConfirmationType
        return PatternSignal(
            pattern_type="failed_breakdown",
            confirmation=ConfirmationType.ACCEPTANCE,
            level=level,
            sweep_low=level.price - 3.0,
            entry_price=level.price + 1.0,
            stop_price=level.price - 5.0,
            bar_idx=50,
            timestamp=datetime(2024, 1, 15, 10, 0),
        )

    def _make_aggregator(self, use_confluence: bool, min_score: int = 2):
        from core.signals import SignalAggregator, SignalType
        params = StrategyParams(
            use_confluence_scoring=use_confluence,
            confluence_min_score=min_score,
            confluence_proximity_pts=3.0,
        )
        agg = SignalAggregator(strategy_params=params, min_rr_ratio=0.1)
        return agg

    def test_disabled_by_default_passes(self):
        """When use_confluence_scoring=False, any level passes."""
        from core.signals import SignalType
        agg = self._make_aggregator(use_confluence=False)
        level = _make_level(5000.0, LevelType.CLUSTER_LOW, touch_count=1)
        # Add targets above entry so R:R works
        ts = datetime(2024, 1, 15, 9, 0)
        agg.level_store.add(Level(price=5020.0, level_type=LevelType.HORIZONTAL_SR,
                                  created_at=ts, confirmed_at=ts))
        agg.level_store.add(Level(price=5040.0, level_type=LevelType.HORIZONTAL_SR,
                                  created_at=ts, confirmed_at=ts))
        agg.level_store.add(level)
        pattern = self._make_pattern_signal(level)
        signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
        assert signal is not None
        assert signal.confluence_score == 0  # not computed

    def test_enabled_rejects_low_score(self):
        """When enabled with min_score=5, a lone CLUSTER_LOW (score=1) is rejected."""
        from core.signals import SignalType
        agg = self._make_aggregator(use_confluence=True, min_score=5)
        level = _make_level(5000.0, LevelType.CLUSTER_LOW, touch_count=1)
        ts = datetime(2024, 1, 15, 9, 0)
        agg.level_store.add(Level(price=5020.0, level_type=LevelType.HORIZONTAL_SR,
                                  created_at=ts, confirmed_at=ts))
        agg.level_store.add(level)
        pattern = self._make_pattern_signal(level)
        signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
        assert signal is None

    def test_enabled_passes_high_score(self):
        """When enabled, a PDL with bonuses passes min_score=5."""
        from core.signals import SignalType
        agg = self._make_aggregator(use_confluence=True, min_score=5)
        level = _make_level(5000.0, LevelType.PRIOR_DAY_LOW, touch_count=4,
                            rally_from_low_pts=25.0)
        ts = datetime(2024, 1, 15, 9, 0)
        agg.level_store.add(Level(price=5020.0, level_type=LevelType.HORIZONTAL_SR,
                                  created_at=ts, confirmed_at=ts))
        agg.level_store.add(Level(price=5040.0, level_type=LevelType.HORIZONTAL_SR,
                                  created_at=ts, confirmed_at=ts))
        agg.level_store.add(level)
        pattern = self._make_pattern_signal(level)
        signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
        assert signal is not None
        # PDL base=5, +1 touch, +1 rally = 7
        assert signal.confluence_score == 7

    def test_short_signal_confluence_gating(self):
        """Confluence gating also applies to _qualify_short_signal."""
        from core.signals import SignalType
        from core.patterns import PatternSignal, ConfirmationType
        agg = self._make_aggregator(use_confluence=True, min_score=5)
        level = _make_level(5050.0, LevelType.CLUSTER_LOW, touch_count=1)
        ts = datetime(2024, 1, 15, 9, 0)
        # Add support below for targets
        agg.level_store.add(Level(price=5020.0, level_type=LevelType.HORIZONTAL_SR,
                                  created_at=ts, confirmed_at=ts))
        agg.level_store.add(Level(price=5000.0, level_type=LevelType.HORIZONTAL_SR,
                                  created_at=ts, confirmed_at=ts))
        agg.level_store.add(level)
        pattern = PatternSignal(
            pattern_type="breakdown_short",
            confirmation=ConfirmationType.ACCEPTANCE,
            level=level,
            sweep_low=5050.0,
            sweep_high=5053.0,
            entry_price=5048.0,
            stop_price=5055.0,  # above entry for shorts
            bar_idx=50,
            timestamp=ts,
            direction="short",
        )
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
        assert signal is None  # CLUSTER_LOW score=1 < min_score=5
