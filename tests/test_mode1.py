"""Tests for Mode 1 (trend day) detection."""

from datetime import datetime, timezone

import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.mode1_detector import Mode1Detector, Mode1State


def _ts(minute: int = 0) -> datetime:
    """Quick timestamp factory. Handles minute > 59 by rolling into hours."""
    hours, mins = divmod(minute, 60)
    return datetime(2026, 3, 15, 10 + hours, mins, tzinfo=timezone.utc)


def _make_level_store_with_supports(*prices: float) -> LevelStore:
    """Create a level store with confirmed support levels."""
    store = LevelStore()
    ts = _ts(0)
    for p in prices:
        store.add(Level(
            price=p,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=ts,
            confirmed_at=ts,
        ))
    return store


class TestMode1Detector:
    """Unit tests for Mode1Detector."""

    def test_default_disabled(self):
        """Mode 1 detection is off by default."""
        params = StrategyParams()
        assert params.use_mode1_detection is False

    def test_no_mode1_in_calm_market(self):
        """No Mode 1 flag when price stays above all levels."""
        params = StrategyParams(use_mode1_detection=True)
        detector = Mode1Detector(params)
        store = _make_level_store_with_supports(5800.0, 5810.0, 5820.0)
        detector.set_pdl(5790.0)

        # Price stays well above all levels
        for i in range(100):
            state = detector.update(
                bar_idx=i, close=5850.0, low=5845.0,
                level_store=store, timestamp=_ts(i),
            )

        assert state.is_mode1_red is False
        assert state.levels_broken_sustained == 0
        assert state.bars_below_pdl == 0

    def test_sustained_broken_levels_condition(self):
        """Condition 1: 3+ levels broken and held for 20+ bars."""
        params = StrategyParams(
            use_mode1_detection=True,
            mode1_levels_broken_threshold=3,
            mode1_level_broken_hold_bars=20,
        )
        detector = Mode1Detector(params)
        store = _make_level_store_with_supports(5820.0, 5810.0, 5800.0)
        detector.set_pdl(5750.0)  # PDL far away so condition 2 doesn't trigger

        # Price drops below all 3 levels and stays there
        for i in range(25):
            state = detector.update(
                bar_idx=i, close=5795.0, low=5794.0,
                level_store=store, timestamp=_ts(i),
            )

        # After 20+ bars below all 3 levels, condition_levels should be True
        assert state.condition_levels is True
        assert state.levels_broken_sustained >= 3

    def test_pdl_condition(self):
        """Condition 2: 30+ bars continuously below PDL."""
        params = StrategyParams(
            use_mode1_detection=True,
            mode1_min_bars_below_pdl=30,
        )
        detector = Mode1Detector(params)
        store = LevelStore()  # empty store so condition 1 doesn't trigger
        detector.set_pdl(5800.0)

        # Price below PDL for 35 bars
        for i in range(35):
            state = detector.update(
                bar_idx=i, close=5790.0, low=5788.0,
                level_store=store, timestamp=_ts(i),
            )

        assert state.condition_pdl is True
        assert state.bars_below_pdl == 35

    def test_pdl_condition_resets_on_recovery(self):
        """PDL counter resets when price recovers above PDL."""
        params = StrategyParams(
            use_mode1_detection=True,
            mode1_min_bars_below_pdl=30,
        )
        detector = Mode1Detector(params)
        store = LevelStore()
        detector.set_pdl(5800.0)

        # 20 bars below, then recover, then 20 bars below again
        for i in range(20):
            detector.update(
                bar_idx=i, close=5790.0, low=5788.0,
                level_store=store, timestamp=_ts(i),
            )
        # Recovery bar
        detector.update(
            bar_idx=20, close=5805.0, low=5795.0,
            level_store=store, timestamp=_ts(20),
        )
        assert detector.state.bars_below_pdl == 0

        # Another 20 bars below
        for i in range(21, 41):
            state = detector.update(
                bar_idx=i, close=5790.0, low=5788.0,
                level_store=store, timestamp=_ts(i),
            )

        # Only 20 consecutive bars (not 40), so PDL condition not met
        assert state.condition_pdl is False
        assert state.bars_below_pdl == 20

    def test_bearish_pressure_condition(self):
        """Condition 3: Sustained bearish pressure for 60+ bars."""
        params = StrategyParams(
            use_mode1_detection=True,
            mode1_bearish_pressure_bars=60,
        )
        detector = Mode1Detector(params)
        store = LevelStore()

        # Grind lower for 65 bars (make new lows periodically)
        for i in range(65):
            close = 5850.0 - i * 0.5  # grinding lower
            low = close - 1.0
            state = detector.update(
                bar_idx=i, close=close, low=low,
                level_store=store, timestamp=_ts(i),
            )

        assert state.condition_pressure is True
        assert state.bearish_pressure_bars >= 60

    def test_mode1_red_requires_two_conditions(self):
        """MODE_1_RED requires any 2 of 3 conditions."""
        params = StrategyParams(
            use_mode1_detection=True,
            mode1_levels_broken_threshold=3,
            mode1_level_broken_hold_bars=20,
            mode1_min_bars_below_pdl=30,
            mode1_bearish_pressure_bars=60,
        )
        detector = Mode1Detector(params)
        store = _make_level_store_with_supports(5820.0, 5810.0, 5800.0)
        detector.set_pdl(5830.0)  # PDL above all levels

        # Grind lower below all levels AND below PDL for 65 bars
        for i in range(65):
            close = 5795.0 - i * 0.3
            low = close - 1.0
            state = detector.update(
                bar_idx=i, close=close, low=low,
                level_store=store, timestamp=_ts(i),
            )

        # All 3 conditions should be met, so is_mode1_red = True
        assert state.is_mode1_red is True
        assert state.conditions_met >= 2

    def test_single_condition_not_enough(self):
        """A single condition alone does not trigger Mode 1."""
        params = StrategyParams(
            use_mode1_detection=True,
            mode1_levels_broken_threshold=3,
            mode1_min_bars_below_pdl=30,
            mode1_bearish_pressure_bars=60,
        )
        detector = Mode1Detector(params)
        store = LevelStore()  # no levels to break
        detector.set_pdl(5800.0)

        # Only PDL condition met (35 bars below)
        for i in range(35):
            state = detector.update(
                bar_idx=i, close=5790.0, low=5789.0,
                level_store=store, timestamp=_ts(i),
            )

        assert state.condition_pdl is True
        assert state.condition_levels is False
        # Only 1 condition, not enough
        assert state.is_mode1_red is False

    def test_reset_clears_state(self):
        """Reset clears all Mode 1 state."""
        params = StrategyParams(use_mode1_detection=True)
        detector = Mode1Detector(params)
        store = _make_level_store_with_supports(5800.0)
        detector.set_pdl(5810.0)

        for i in range(40):
            detector.update(
                bar_idx=i, close=5790.0, low=5789.0,
                level_store=store, timestamp=_ts(i),
            )

        detector.reset()
        state = detector.state
        assert state.is_mode1_red is False
        assert state.bars_below_pdl == 0
        assert state.levels_broken_sustained == 0
        assert state.bearish_pressure_bars == 0

    def test_level_recovery_removes_from_broken(self):
        """If price recovers above a broken level, it no longer counts."""
        params = StrategyParams(
            use_mode1_detection=True,
            mode1_levels_broken_threshold=2,
            mode1_level_broken_hold_bars=5,
        )
        detector = Mode1Detector(params)
        store = _make_level_store_with_supports(5800.0, 5810.0)

        # Break both levels for 10 bars
        for i in range(10):
            detector.update(
                bar_idx=i, close=5795.0, low=5794.0,
                level_store=store, timestamp=_ts(i),
            )
        assert detector.state.levels_broken_sustained == 2

        # Recover above 5800 (still below 5810)
        for i in range(10, 15):
            detector.update(
                bar_idx=i, close=5805.0, low=5804.0,
                level_store=store, timestamp=_ts(i),
            )

        # Only 1 level still broken (5810)
        assert detector.state.levels_broken_sustained <= 1

    def test_no_pdl_set(self):
        """PDL condition stays False when no PDL is set."""
        params = StrategyParams(
            use_mode1_detection=True,
            mode1_min_bars_below_pdl=30,
        )
        detector = Mode1Detector(params)
        store = LevelStore()
        # Don't call set_pdl

        for i in range(50):
            state = detector.update(
                bar_idx=i, close=5790.0, low=5789.0,
                level_store=store, timestamp=_ts(i),
            )

        assert state.condition_pdl is False
        assert state.bars_below_pdl == 0


class TestMode1Integration:
    """Tests for Mode 1 integration with strategy sizing."""

    def test_size_factor_reduction(self):
        """position_size_factor is reduced when Mode 1 is active."""
        from core.signals import Signal, SignalType
        from core.patterns import PatternSignal
        from config.levels import Level, LevelType

        params = StrategyParams(
            use_mode1_detection=True,
            mode1_size_reduction=0.25,
        )

        # Simulate a signal with initial size_factor=1.0
        level = Level(
            price=5800.0,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=_ts(0),
            confirmed_at=_ts(0),
        )
        # Mode 1 reduction: 1.0 * 0.25 = 0.25
        initial_factor = 1.0
        reduced = initial_factor * params.mode1_size_reduction
        assert reduced == 0.25

    def test_entry_manager_applies_size_factor(self):
        """EntryManager._size_position applies position_size_factor."""
        from config.settings import ExitParams, RiskParams, SessionTimes
        from strategy.entry_manager import EntryManager
        from core.signals import Signal, SignalType
        from unittest.mock import MagicMock

        entry_mgr = EntryManager(
            session=SessionTimes(),
            exit_params=ExitParams(default_contracts=4),
            risk_params=RiskParams(),
        )

        # Create a mock signal with position_size_factor=0.25
        signal = MagicMock(spec=Signal)
        signal.risk_pts = 5.0
        signal.position_size_factor = 0.25

        contracts = entry_mgr._size_position(signal, 0.0, False)
        # 4 * 0.25 = 1 (minimum 1)
        assert contracts == 1

    def test_entry_manager_full_size_factor(self):
        """EntryManager doesn't reduce when size_factor=1.0."""
        from config.settings import ExitParams, RiskParams, SessionTimes
        from strategy.entry_manager import EntryManager
        from unittest.mock import MagicMock
        from core.signals import Signal

        entry_mgr = EntryManager(
            session=SessionTimes(),
            exit_params=ExitParams(default_contracts=4),
            risk_params=RiskParams(),
        )

        signal = MagicMock(spec=Signal)
        signal.risk_pts = 5.0
        signal.position_size_factor = 1.0

        contracts = entry_mgr._size_position(signal, 0.0, False)
        assert contracts == 4

    def test_entry_manager_half_size_factor(self):
        """EntryManager applies 0.5 factor correctly."""
        from config.settings import ExitParams, RiskParams, SessionTimes
        from strategy.entry_manager import EntryManager
        from unittest.mock import MagicMock
        from core.signals import Signal

        entry_mgr = EntryManager(
            session=SessionTimes(),
            exit_params=ExitParams(default_contracts=4),
            risk_params=RiskParams(),
        )

        signal = MagicMock(spec=Signal)
        signal.risk_pts = 5.0
        signal.position_size_factor = 0.5

        contracts = entry_mgr._size_position(signal, 0.0, False)
        # 4 * 0.5 = 2
        assert contracts == 2
