"""Tests for sweep depth position sizing."""

from __future__ import annotations

from datetime import datetime

import pytest

from config.levels import Level, LevelType
from config.settings import StrategyParams
from core.patterns import PatternSignal, ConfirmationType
from core.signals import SignalAggregator, SignalType

_LEVEL_TS = datetime(2024, 6, 14, 16, 0)


def _make_level(price: float = 5800.0) -> Level:
    return Level(
        price=price,
        level_type=LevelType.PRIOR_DAY_LOW,
        created_at=_LEVEL_TS,
        touch_count=3,
    )


def _make_pattern(
    level_price: float = 5800.0,
    sweep_low: float = 5795.0,
    entry_price: float = 5801.0,
    stop_price: float = 5790.0,
    sweep_depth_pts: float = 5.0,
    direction: str = "long",
) -> PatternSignal:
    """Create a PatternSignal for testing."""
    return PatternSignal(
        pattern_type="failed_breakdown",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=_make_level(level_price),
        sweep_low=sweep_low,
        entry_price=entry_price,
        stop_price=stop_price,
        bar_idx=100,
        timestamp=datetime(2024, 6, 15, 10, 30),
        sweep_depth_pts=sweep_depth_pts,
        direction=direction,
    )


def _make_short_pattern(
    level_price: float = 5800.0,
    sweep_high: float = 5805.0,
    entry_price: float = 5799.0,
    stop_price: float = 5810.0,
    sweep_depth_pts: float = 5.0,
) -> PatternSignal:
    """Create a short-side PatternSignal for testing."""
    return PatternSignal(
        pattern_type="breakdown_short",
        confirmation=ConfirmationType.ACCEPTANCE,
        level=_make_level(level_price),
        sweep_low=5790.0,
        entry_price=entry_price,
        stop_price=stop_price,
        bar_idx=100,
        timestamp=datetime(2024, 6, 15, 10, 30),
        sweep_depth_pts=sweep_depth_pts,
        direction="short",
        sweep_high=sweep_high,
    )


def _make_agg_with_levels(use_sweep_depth: bool = True, **extra_params):
    """Create a SignalAggregator with resistance/support levels pre-loaded."""
    from config.levels import LevelStore
    params = StrategyParams(use_sweep_depth_sizing=use_sweep_depth, **extra_params)
    agg = SignalAggregator(strategy_params=params)
    agg.level_store = LevelStore()
    # Resistance targets above ~5801 entry (for long signals)
    agg.level_store.add(_make_level(5815.0))
    agg.level_store.add(_make_level(5830.0))
    # Support targets below ~5799 entry (for short signals)
    agg.level_store.add(_make_level(5785.0))
    agg.level_store.add(_make_level(5770.0))
    return agg


class TestSweepDepthSizingConfig:
    """Test that sweep depth config params have correct defaults."""

    def test_defaults_off(self):
        params = StrategyParams()
        assert params.use_sweep_depth_sizing is False
        assert params.sweep_depth_min_pts == 2.0
        assert params.sweep_depth_full_size_pts == 8.0
        assert params.sweep_depth_quarter_size_pts == 2.0

    def test_enable_via_constructor(self):
        params = StrategyParams(use_sweep_depth_sizing=True)
        assert params.use_sweep_depth_sizing is True


class TestComputeSweepDepthSizeFactor:
    """Test _compute_sweep_depth_size_factor directly."""

    @pytest.fixture
    def agg(self):
        params = StrategyParams(use_sweep_depth_sizing=True)
        return SignalAggregator(strategy_params=params)

    def test_very_shallow_sweep_quarter_size(self, agg):
        """Sweep < 2 pts should return 0.25 (quarter size)."""
        pattern = _make_pattern(sweep_depth_pts=1.0)
        assert agg._compute_sweep_depth_size_factor(pattern) == 0.25

    def test_zero_sweep_quarter_size(self, agg):
        """Zero sweep depth should return 0.25."""
        pattern = _make_pattern(sweep_depth_pts=0.0, sweep_low=5800.0, level_price=5800.0)
        assert agg._compute_sweep_depth_size_factor(pattern) == 0.25

    def test_sweep_at_quarter_boundary(self, agg):
        """Sweep exactly at quarter_size_pts (2.0) should be 0.25."""
        pattern = _make_pattern(sweep_depth_pts=2.0)
        assert agg._compute_sweep_depth_size_factor(pattern) == pytest.approx(0.25, abs=0.01)

    def test_sweep_3pt_half_way_lower_tier(self, agg):
        """Sweep 3.5 pts: midpoint of 2-5 range -> ~0.375."""
        pattern = _make_pattern(sweep_depth_pts=3.5)
        factor = agg._compute_sweep_depth_size_factor(pattern)
        assert 0.25 < factor < 0.50

    def test_sweep_5pt_half_size(self, agg):
        """Sweep at mid_pt (5.0) should be 0.50."""
        pattern = _make_pattern(sweep_depth_pts=5.0)
        assert agg._compute_sweep_depth_size_factor(pattern) == pytest.approx(0.50, abs=0.01)

    def test_sweep_6pt_between_half_and_full(self, agg):
        """Sweep 6.5 pts: between 5 and 8 -> between 0.50 and 1.0."""
        pattern = _make_pattern(sweep_depth_pts=6.5)
        factor = agg._compute_sweep_depth_size_factor(pattern)
        assert 0.50 < factor < 1.0

    def test_sweep_8pt_full_size(self, agg):
        """Sweep >= full_size_pts (8.0) should be 1.0."""
        pattern = _make_pattern(sweep_depth_pts=8.0)
        assert agg._compute_sweep_depth_size_factor(pattern) == 1.0

    def test_deep_sweep_30pt_full_size(self, agg):
        """Deep sweep (30+ pts crash bottom) should still be 1.0."""
        pattern = _make_pattern(sweep_depth_pts=30.0)
        assert agg._compute_sweep_depth_size_factor(pattern) == 1.0

    def test_linear_interpolation_lower_tier(self, agg):
        """Verify linear interpolation in 2-5 pt range."""
        # At 2.0: 0.25, at 5.0: 0.50
        # At 3.0: 0.25 + (1/3)*0.25 = 0.3333
        pattern = _make_pattern(sweep_depth_pts=3.0)
        assert agg._compute_sweep_depth_size_factor(pattern) == pytest.approx(0.3333, abs=0.01)

    def test_linear_interpolation_upper_tier(self, agg):
        """Verify linear interpolation in 5-8 pt range."""
        # At 5.0: 0.50, at 8.0: 1.0
        # At 6.5: 0.50 + (1.5/3.0)*0.50 = 0.75
        pattern = _make_pattern(sweep_depth_pts=6.5)
        assert agg._compute_sweep_depth_size_factor(pattern) == pytest.approx(0.75, abs=0.01)

    def test_fallback_to_level_minus_sweep_low(self, agg):
        """When sweep_depth_pts is 0, derive from level.price - sweep_low."""
        pattern = _make_pattern(
            sweep_depth_pts=0.0,
            level_price=5800.0,
            sweep_low=5794.0,  # 6 pts below level
        )
        # 6.0 pts is in the 5-8 range: 0.50 + (1.0/3.0)*0.50 = 0.6667
        assert agg._compute_sweep_depth_size_factor(pattern) == pytest.approx(0.6667, abs=0.01)


class TestSweepDepthSizingDisabled:
    """When use_sweep_depth_sizing=False, stop-distance sizing should be used."""

    def test_stop_distance_sizing_when_disabled(self):
        agg = _make_agg_with_levels(use_sweep_depth=False)
        # Large sweep depth but feature is off: should use stop-distance sizing
        pattern = _make_pattern(
            sweep_depth_pts=30.0,
            entry_price=5801.0,
            stop_price=5790.0,  # 11 pt risk -> within max_full_stop (15)
        )
        signal = agg._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
        assert signal is not None
        # risk is 11 pts, max_full_stop is 15 -> full size from stop-distance
        assert signal.position_size_factor == 1.0


class TestSweepDepthSizingIntegration:
    """Integration: _qualify_signal uses sweep depth sizing when enabled."""

    @pytest.fixture
    def agg_with_levels(self):
        return _make_agg_with_levels(use_sweep_depth=True)

    def test_qualify_signal_shallow_sweep(self, agg_with_levels):
        """Shallow sweep (1 pt) should produce quarter size signal."""
        pattern = _make_pattern(sweep_depth_pts=1.0)
        signal = agg_with_levels._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
        assert signal is not None
        assert signal.position_size_factor == 0.25

    def test_qualify_signal_deep_sweep(self, agg_with_levels):
        """Deep sweep (10 pts) should produce full size signal."""
        pattern = _make_pattern(sweep_depth_pts=10.0)
        signal = agg_with_levels._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
        assert signal is not None
        assert signal.position_size_factor == 1.0

    def test_qualify_signal_medium_sweep(self, agg_with_levels):
        """Medium sweep (5 pts) should produce half size signal."""
        pattern = _make_pattern(sweep_depth_pts=5.0)
        signal = agg_with_levels._qualify_signal(pattern, SignalType.FAILED_BREAKDOWN)
        assert signal is not None
        assert signal.position_size_factor == pytest.approx(0.50, abs=0.01)


class TestSweepDepthShortSide:
    """Sweep depth sizing also works for short signals."""

    def test_short_signal_uses_sweep_depth(self):
        agg = _make_agg_with_levels(use_sweep_depth=True, allow_breakdown_short=True)
        pattern = _make_short_pattern(sweep_depth_pts=10.0)
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
        assert signal is not None
        assert signal.position_size_factor == 1.0

    def test_short_signal_shallow_sweep(self):
        agg = _make_agg_with_levels(use_sweep_depth=True, allow_breakdown_short=True)
        pattern = _make_short_pattern(sweep_depth_pts=1.5)
        signal = agg._qualify_short_signal(pattern, SignalType.BREAKDOWN_SHORT)
        assert signal is not None
        assert signal.position_size_factor == 0.25
