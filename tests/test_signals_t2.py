"""Regression tests for T2 distinctness in _qualify_signal / _qualify_short_signal.

Live bug (5/2026): a real trade logged T1 == T2 == 7435.50 because two
levels of *different* types sat at the same price. LevelStore.add() only
merges nearby same-type levels, so e.g. a PRIOR_DAY_HIGH and a CUSTOM
(Mancini overlay) both at 7435.50 produce two entries with identical
prices. The old `above[0]/above[1]` indexing then assigned them to T1
and T2, collapsing the staged exit.

These tests pin down the fix: T1 and T2 must always be distinct and
separated by at least `mancini_t1_min_distance_pts` (or the configured
`min_target_distance_pts`), and the fallback when only one level exists
must also produce T2 != T1.
"""

from __future__ import annotations

from datetime import datetime

from config.levels import Level, LevelType
from config.settings import StrategyParams
from core.patterns import PatternSignal, ConfirmationType
from core.signals import SignalAggregator, SignalType


TS = datetime(2024, 1, 15, 10, 0)
CREATED = datetime(2024, 1, 15, 9, 0)


def _make_level(
    price: float,
    level_type: LevelType,
    touch_count: int = 1,
) -> Level:
    """Build a confirmed Level with sensible defaults."""
    return Level(
        price=price,
        level_type=level_type,
        created_at=CREATED,
        confirmed_at=CREATED,
        touch_count=touch_count,
    )


def _make_long_pattern(level: Level) -> PatternSignal:
    """Build a FailedBreakdown long PatternSignal pointing at `level`."""
    return PatternSignal(
        pattern_type="failed_breakdown",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=level,
        sweep_low=level.price - 3.0,
        entry_price=level.price + 1.0,
        stop_price=level.price - 5.0,
        bar_idx=50,
        timestamp=TS,
        direction="long",
    )


def _make_short_pattern(level: Level) -> PatternSignal:
    """Build a BreakdownShort PatternSignal pointing at `level`."""
    return PatternSignal(
        pattern_type="breakdown_short",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=level,
        sweep_low=level.price,
        sweep_high=level.price + 3.0,
        entry_price=level.price - 2.0,
        stop_price=level.price + 5.0,
        bar_idx=50,
        timestamp=TS,
        direction="short",
    )


def _make_aggregator(min_dist: float = 8.0) -> SignalAggregator:
    """Aggregator with permissive R:R floor; no confluence/LQS gating."""
    params = StrategyParams(
        use_confluence_scoring=False,
        use_level_quality_scoring=False,
        mancini_t1_at_first_resistance=True,
        mancini_t1_min_distance_pts=min_dist,
        min_target_distance_pts=min_dist,
        min_signal_rr=0.1,
        bd_short_min_rr=0.1,
        block_pdl_shorts=False,
        block_capitulation_shorts=False,
        use_daily_structure=False,
        short_size_factor=1.0,
    )
    return SignalAggregator(strategy_params=params, min_rr_ratio=0.1)


# ---------------------------------------------------------------------------
# Long side
# ---------------------------------------------------------------------------


class TestLongT2Distinct:
    """_qualify_signal must keep T2 distinct from T1."""

    def test_two_distinct_levels_become_t1_t2(self):
        """When two well-separated resistance levels exist, use them directly."""
        agg = _make_aggregator(min_dist=8.0)
        # Entry pattern at 6020 (entry = 6020 + 1 = 6021).
        trigger = _make_level(6020.0, LevelType.PRIOR_DAY_LOW)
        agg.level_store.add(trigger)
        # Two distinct resistances above entry.
        agg.level_store.add(_make_level(6040.0, LevelType.HORIZONTAL_SR))
        agg.level_store.add(_make_level(6060.0, LevelType.HORIZONTAL_SR))

        pattern = _make_long_pattern(trigger)
        signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)

        assert signal is not None
        # T1/T2 may be capped by max_target_distance_pts (default 30), so
        # we don't compare to raw level prices — only to each other.
        assert signal.target_1 != signal.target_2
        assert signal.target_2 > signal.target_1

    def test_collision_two_levels_same_price_skips_to_next_distinct(self):
        """The live bug: PDH + CUSTOM at the same price. T2 must skip past."""
        agg = _make_aggregator(min_dist=8.0)
        trigger = _make_level(7400.0, LevelType.PRIOR_DAY_LOW)
        agg.level_store.add(trigger)
        # Two different-type levels at the IDENTICAL price (the live bug).
        # LevelStore.add merges only within-type, so both survive.
        agg.level_store.add(_make_level(7435.50, LevelType.PRIOR_DAY_HIGH))
        agg.level_store.add(_make_level(7435.50, LevelType.CUSTOM))
        # A third, properly separated level the fix should pick up for T2.
        agg.level_store.add(_make_level(7450.0, LevelType.HORIZONTAL_SR))

        pattern = _make_long_pattern(trigger)
        signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)

        assert signal is not None
        assert signal.target_1 != signal.target_2, (
            f"Bug regression: T1==T2=={signal.target_1} on duplicate-price levels"
        )
        assert signal.target_2 - signal.target_1 >= 8.0

    def test_single_level_fallback_distinct(self):
        """Only one resistance level → fallback T2 must differ from T1."""
        agg = _make_aggregator(min_dist=8.0)
        trigger = _make_level(6020.0, LevelType.PRIOR_DAY_LOW)
        agg.level_store.add(trigger)
        # Only ONE resistance above entry.
        agg.level_store.add(_make_level(6035.0, LevelType.HORIZONTAL_SR))

        pattern = _make_long_pattern(trigger)
        signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)

        assert signal is not None
        assert signal.target_1 != signal.target_2
        assert signal.target_2 > signal.target_1
        assert signal.target_2 - signal.target_1 >= 8.0

    def test_collision_with_no_further_level_falls_back(self):
        """Two duplicate-price levels and no third level → fallback T2."""
        agg = _make_aggregator(min_dist=8.0)
        trigger = _make_level(7400.0, LevelType.PRIOR_DAY_LOW)
        agg.level_store.add(trigger)
        # Two same-price different-type levels and nothing further above.
        agg.level_store.add(_make_level(7435.50, LevelType.PRIOR_DAY_HIGH))
        agg.level_store.add(_make_level(7435.50, LevelType.CUSTOM))

        pattern = _make_long_pattern(trigger)
        signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)

        assert signal is not None
        assert signal.target_1 != signal.target_2
        assert signal.target_2 > signal.target_1


# ---------------------------------------------------------------------------
# Short side
# ---------------------------------------------------------------------------


class TestShortT2Distinct:
    """_qualify_short_signal must keep T2 distinct from T1."""

    def test_two_distinct_levels_become_t1_t2(self):
        """Two well-separated support levels below entry produce distinct T1/T2."""
        agg = _make_aggregator(min_dist=8.0)
        trigger = _make_level(6050.0, LevelType.CLUSTER_HIGH)
        agg.level_store.add(trigger)
        # Two distinct supports below entry (entry = 6050 - 2 = 6048).
        agg.level_store.add(_make_level(6030.0, LevelType.HORIZONTAL_SR))
        agg.level_store.add(_make_level(6010.0, LevelType.HORIZONTAL_SR))

        pattern = _make_short_pattern(trigger)
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)

        assert signal is not None
        assert signal.target_1 != signal.target_2
        assert signal.target_2 < signal.target_1

    def test_collision_two_levels_same_price_skips_to_next_distinct(self):
        """Mirror of the long-side live bug, short side."""
        agg = _make_aggregator(min_dist=8.0)
        trigger = _make_level(7500.0, LevelType.CLUSTER_HIGH)
        agg.level_store.add(trigger)
        # Two different-type supports at the IDENTICAL price.
        agg.level_store.add(_make_level(7465.0, LevelType.PRIOR_DAY_LOW))
        agg.level_store.add(_make_level(7465.0, LevelType.CUSTOM))
        # A third, properly separated support for T2 to pick up.
        agg.level_store.add(_make_level(7450.0, LevelType.HORIZONTAL_SR))

        pattern = _make_short_pattern(trigger)
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)

        assert signal is not None
        assert signal.target_1 != signal.target_2, (
            f"Bug regression: T1==T2=={signal.target_1} on duplicate-price levels"
        )
        assert signal.target_1 - signal.target_2 >= 8.0

    def test_single_level_fallback_distinct(self):
        """Only one support level → fallback T2 must differ from T1."""
        agg = _make_aggregator(min_dist=8.0)
        trigger = _make_level(6050.0, LevelType.CLUSTER_HIGH)
        agg.level_store.add(trigger)
        # Only ONE support below entry.
        agg.level_store.add(_make_level(6035.0, LevelType.HORIZONTAL_SR))

        pattern = _make_short_pattern(trigger)
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)

        assert signal is not None
        assert signal.target_1 != signal.target_2
        assert signal.target_2 < signal.target_1
        assert signal.target_1 - signal.target_2 >= 8.0

    def test_collision_with_no_further_level_falls_back(self):
        """Two duplicate-price supports and no third → fallback T2."""
        agg = _make_aggregator(min_dist=8.0)
        trigger = _make_level(7500.0, LevelType.CLUSTER_HIGH)
        agg.level_store.add(trigger)
        # Two same-price different-type supports, nothing further below.
        agg.level_store.add(_make_level(7465.0, LevelType.PRIOR_DAY_LOW))
        agg.level_store.add(_make_level(7465.0, LevelType.CUSTOM))

        pattern = _make_short_pattern(trigger)
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)

        assert signal is not None
        assert signal.target_1 != signal.target_2
        assert signal.target_2 < signal.target_1
