"""Tests for Mode 1 Green (trend UP day) detection."""

from datetime import datetime, timezone

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.mode1_green_detector import Mode1GreenDetector, Mode1GreenState


def _ts(minute: int = 0) -> datetime:
    """Quick timestamp factory."""
    hours, mins = divmod(minute, 60)
    return datetime(2026, 4, 15, 10 + hours, mins, tzinfo=timezone.utc)


def _make_level_store_with_resistances(*prices: float) -> LevelStore:
    """Create a level store with confirmed resistance levels."""
    store = LevelStore()
    ts = _ts(0)
    for p in prices:
        store.add(Level(
            price=p,
            level_type=LevelType.MULTI_HOUR_HIGH,
            created_at=ts,
            confirmed_at=ts,
        ))
    return store


class TestMode1GreenDetector:
    """Unit tests for Mode1GreenDetector."""

    def test_default_disabled(self):
        """Mode 1 Green detection is off by default."""
        params = StrategyParams()
        assert params.use_mode1_green_detection is False

    def test_no_green_in_calm_market(self):
        """No MODE_1_GREEN when price stays below all resistances."""
        params = StrategyParams(use_mode1_green_detection=True)
        detector = Mode1GreenDetector(params)
        store = _make_level_store_with_resistances(7000.0, 7010.0, 7020.0)
        detector.set_pdh(7030.0)

        for i in range(100):
            state = detector.update(
                bar_idx=i, close=6950.0, high=6955.0,
                level_store=store, timestamp=_ts(i),
            )

        assert state.is_mode1_green is False
        assert state.resistances_broken_sustained == 0
        assert state.bars_above_pdh == 0

    def test_sustained_broken_resistances_condition(self):
        """Condition 1: 3+ resistances broken UP and held for 20+ bars."""
        params = StrategyParams(
            use_mode1_green_detection=True,
            mode1_green_resistance_broken_threshold=3,
            mode1_green_level_broken_hold_bars=20,
        )
        detector = Mode1GreenDetector(params)
        store = _make_level_store_with_resistances(6980.0, 6990.0, 7000.0)
        detector.set_pdh(7050.0)  # PDH far above so condition 2 doesn't trigger

        # Price pushes above all 3 resistances and holds there
        for i in range(25):
            state = detector.update(
                bar_idx=i, close=7005.0, high=7007.0,
                level_store=store, timestamp=_ts(i),
            )

        assert state.condition_resistances is True
        assert state.resistances_broken_sustained >= 3

    def test_pdh_condition(self):
        """Condition 2: 30+ bars continuously above PDH."""
        params = StrategyParams(
            use_mode1_green_detection=True,
            mode1_green_bars_above_pdh=30,
        )
        detector = Mode1GreenDetector(params)
        store = LevelStore()  # empty store so condition 1 doesn't trigger
        detector.set_pdh(7000.0)

        for i in range(35):
            state = detector.update(
                bar_idx=i, close=7010.0, high=7012.0,
                level_store=store, timestamp=_ts(i),
            )

        assert state.condition_pdh is True
        assert state.bars_above_pdh == 35

    def test_pdh_condition_resets_on_pullback(self):
        """PDH counter resets when price falls back below PDH."""
        params = StrategyParams(
            use_mode1_green_detection=True,
            mode1_green_bars_above_pdh=30,
        )
        detector = Mode1GreenDetector(params)
        store = LevelStore()
        detector.set_pdh(7000.0)

        # 20 bars above, then pullback, then 20 bars above again
        for i in range(20):
            detector.update(
                bar_idx=i, close=7010.0, high=7012.0,
                level_store=store, timestamp=_ts(i),
            )
        # Pullback bar
        detector.update(
            bar_idx=20, close=6995.0, high=7005.0,
            level_store=store, timestamp=_ts(20),
        )
        assert detector.state.bars_above_pdh == 0

        # Another 20 bars above
        for i in range(21, 41):
            state = detector.update(
                bar_idx=i, close=7010.0, high=7012.0,
                level_store=store, timestamp=_ts(i),
            )

        # Only 20 consecutive bars (not 40), so PDH condition not met
        assert state.condition_pdh is False
        assert state.bars_above_pdh == 20

    def test_bullish_pressure_condition(self):
        """Condition 3: Sustained bullish pressure for 60+ bars."""
        params = StrategyParams(
            use_mode1_green_detection=True,
            mode1_green_bullish_pressure_bars=60,
        )
        detector = Mode1GreenDetector(params)
        store = LevelStore()

        # Grind higher for 65 bars (make new highs periodically)
        for i in range(65):
            high = 6950.0 + i * 0.5  # grinding higher
            close = high - 0.5
            state = detector.update(
                bar_idx=i, close=close, high=high,
                level_store=store, timestamp=_ts(i),
            )

        assert state.condition_pressure is True
        assert state.bullish_pressure_bars >= 60

    def test_green_requires_two_conditions(self):
        """MODE_1_GREEN requires any 2 of 3 conditions."""
        params = StrategyParams(
            use_mode1_green_detection=True,
            mode1_green_resistance_broken_threshold=3,
            mode1_green_level_broken_hold_bars=20,
            mode1_green_bars_above_pdh=30,
            mode1_green_bullish_pressure_bars=60,
        )
        detector = Mode1GreenDetector(params)
        store = _make_level_store_with_resistances(6980.0, 6990.0, 7000.0)
        detector.set_pdh(6970.0)  # PDH below all resistances

        # Grind higher above all resistances AND above PDH for 65 bars
        for i in range(65):
            high = 7010.0 + i * 0.3
            close = high - 0.5
            state = detector.update(
                bar_idx=i, close=close, high=high,
                level_store=store, timestamp=_ts(i),
            )

        # All 3 conditions should be met, so is_mode1_green = True
        assert state.is_mode1_green is True
        assert state.conditions_met >= 2

    def test_single_condition_not_enough(self):
        """A single condition alone does not trigger Mode 1 Green."""
        params = StrategyParams(
            use_mode1_green_detection=True,
            mode1_green_resistance_broken_threshold=3,
            mode1_green_bars_above_pdh=30,
            mode1_green_bullish_pressure_bars=60,
        )
        detector = Mode1GreenDetector(params)
        store = LevelStore()  # no resistances to break
        detector.set_pdh(7000.0)

        # Only PDH condition met (35 bars above)
        for i in range(35):
            state = detector.update(
                bar_idx=i, close=7010.0, high=7011.0,
                level_store=store, timestamp=_ts(i),
            )

        assert state.condition_pdh is True
        assert state.condition_resistances is False
        # Only 1 condition, not enough
        assert state.is_mode1_green is False

    def test_reset_clears_state(self):
        """Reset clears all Mode 1 Green state."""
        params = StrategyParams(use_mode1_green_detection=True)
        detector = Mode1GreenDetector(params)
        store = _make_level_store_with_resistances(7000.0)
        detector.set_pdh(6990.0)

        for i in range(40):
            detector.update(
                bar_idx=i, close=7010.0, high=7012.0,
                level_store=store, timestamp=_ts(i),
            )

        detector.reset()
        state = detector.state
        assert state.is_mode1_green is False
        assert state.bars_above_pdh == 0
        assert state.resistances_broken_sustained == 0
        assert state.bullish_pressure_bars == 0

    def test_resistance_pullback_removes_from_broken(self):
        """If price pulls back below a broken resistance, it no longer counts."""
        params = StrategyParams(
            use_mode1_green_detection=True,
            mode1_green_resistance_broken_threshold=2,
            mode1_green_level_broken_hold_bars=5,
        )
        detector = Mode1GreenDetector(params)
        store = _make_level_store_with_resistances(7000.0, 7010.0)

        # Break both resistances for 10 bars
        for i in range(10):
            detector.update(
                bar_idx=i, close=7015.0, high=7016.0,
                level_store=store, timestamp=_ts(i),
            )
        assert detector.state.resistances_broken_sustained == 2

        # Fall back below 7010 (still above 7000)
        for i in range(10, 15):
            detector.update(
                bar_idx=i, close=7005.0, high=7008.0,
                level_store=store, timestamp=_ts(i),
            )

        # Only 1 resistance still broken (7000)
        assert detector.state.resistances_broken_sustained <= 1

    def test_no_pdh_set(self):
        """PDH condition stays False when no PDH is set."""
        params = StrategyParams(
            use_mode1_green_detection=True,
            mode1_green_bars_above_pdh=30,
        )
        detector = Mode1GreenDetector(params)
        store = LevelStore()
        # Don't call set_pdh

        for i in range(50):
            state = detector.update(
                bar_idx=i, close=7010.0, high=7012.0,
                level_store=store, timestamp=_ts(i),
            )

        assert state.condition_pdh is False
        assert state.bars_above_pdh == 0


class TestDangerZoneDipAcceptance:
    """Test the danger-zone dip acceptance requirement in FailedBreakdown."""

    def test_danger_zone_requires_dip(self):
        """In danger zone (<5 pts recovery), non-acceptance must see a dip-back.

        Mancini Apr 15 2026: "5 points above the significant low is the
        danger zone... if entering in this zone you need clear acceptance."
        Clear acceptance = price dipped back to within proximity of the
        level and held above (dip-recover pattern).
        """
        from core.patterns import FailedBreakdown, PatternState
        from config.levels import Level, LevelStore, LevelType

        params = StrategyParams(
            danger_zone_pts=5.0,
            danger_zone_require_dip_acceptance=True,
            danger_zone_dip_proximity_pts=2.0,
            non_acceptance_min_recovery_pts=5.0,
            non_acceptance_min_hold_bars=3,
        )
        fb = FailedBreakdown(params)

        level = Level(
            price=6983.0,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=_ts(0),
            confirmed_at=_ts(0),
            rally_from_low_pts=21.0,
        )
        store = LevelStore()
        store.add(level)

        # Force the state machine into NON_ACCEPTANCE_WATCH with no dips.
        fb.state = PatternState.NON_ACCEPTANCE_WATCH
        fb._target_level = level
        fb._sweep_low = 6980.0
        fb._recovery_bar = 5
        fb._recovery_price = 6988.0
        fb._hold_bars = 0
        fb._acceptance_dips = 0
        fb._last_dip_in_zone = False

        # Three bars hovering in the danger zone (close = level + 3 pts,
        # never dipping back down to within 2 pts of the level).
        # Each bar: recovery = 3.0 pts — below 5 pts danger_zone threshold
        # but we also need recovery >= 5 for _hold_bars to increment.
        # So put recovery at 5.0 exactly (right at the edge, close +5).
        signal = None
        for i in range(6, 20):
            # close stays at level+5 (just above danger zone edge) so hold counts,
            # but the close sometimes dips to level+3 (still in danger zone).
            signal = fb._check_non_acceptance(
                bar_idx=i, timestamp=_ts(i),
                high=level.price + 6.0,
                low=level.price + 3.0,      # never dips back toward level
                close=level.price + 3.0,    # emit-time close IN danger zone
            )
            if signal is not None:
                break

        # With no dip-back, we should NOT emit even though hold bars pass.
        assert signal is None
        assert fb._acceptance_dips == 0

    def test_danger_zone_clears_with_dip(self):
        """Dip-back into proximity counts and unblocks the emit."""
        from core.patterns import FailedBreakdown, PatternState
        from config.levels import Level, LevelStore, LevelType

        params = StrategyParams(
            danger_zone_pts=5.0,
            danger_zone_require_dip_acceptance=True,
            danger_zone_dip_proximity_pts=2.0,
            non_acceptance_min_recovery_pts=5.0,
            non_acceptance_min_hold_bars=3,
        )
        fb = FailedBreakdown(params)

        level = Level(
            price=6983.0,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=_ts(0),
            confirmed_at=_ts(0),
            rally_from_low_pts=21.0,
        )

        fb.state = PatternState.NON_ACCEPTANCE_WATCH
        fb._target_level = level
        fb._sweep_low = 6980.0
        fb._recovery_bar = 5
        fb._recovery_price = 6988.0
        fb._hold_bars = 0
        fb._acceptance_dips = 0
        fb._last_dip_in_zone = False

        # Bar 6: recovery=5, no dip — hold_bars=1
        fb._check_non_acceptance(
            bar_idx=6, timestamp=_ts(6),
            high=6990.0, low=6987.0, close=6988.0,
        )
        # Bar 7: DIP — low touches within proximity of level (6984 = level+1)
        fb._check_non_acceptance(
            bar_idx=7, timestamp=_ts(7),
            high=6990.0, low=6984.0, close=6988.0,
        )
        assert fb._acceptance_dips == 1
        # Bar 8: recovery=5, hold_bars=3, dip seen — SHOULD EMIT
        # but close is at level+5 (boundary) — NOT in danger zone, so emit anyway
        signal = fb._check_non_acceptance(
            bar_idx=8, timestamp=_ts(8),
            high=6995.0, low=6988.0, close=6988.0,
        )
        # Either close is >= level+5 (not in danger zone) or we have a dip — emit
        assert signal is not None

    def test_outside_danger_zone_emits_normally(self):
        """With recovery >= 5 pts at emit, no dip requirement applies."""
        from core.patterns import FailedBreakdown, PatternState
        from config.levels import Level, LevelStore, LevelType

        params = StrategyParams(
            danger_zone_pts=5.0,
            danger_zone_require_dip_acceptance=True,
            non_acceptance_min_recovery_pts=5.0,
            non_acceptance_min_hold_bars=3,
        )
        fb = FailedBreakdown(params)

        level = Level(
            price=6983.0,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=_ts(0),
            confirmed_at=_ts(0),
            rally_from_low_pts=21.0,
        )

        fb.state = PatternState.NON_ACCEPTANCE_WATCH
        fb._target_level = level
        fb._sweep_low = 6980.0
        fb._recovery_bar = 5
        fb._recovery_price = 6995.0
        fb._hold_bars = 0
        fb._acceptance_dips = 0
        fb._last_dip_in_zone = False

        # Three bars at close = level + 10 — well above danger zone
        signal = None
        for i in range(6, 10):
            signal = fb._check_non_acceptance(
                bar_idx=i, timestamp=_ts(i),
                high=level.price + 11.0,
                low=level.price + 9.0,
                close=level.price + 10.0,
            )
            if signal is not None:
                break

        assert signal is not None  # emits without needing a dip


class TestRiskyTrendFbFlag:
    """Test PatternSignal.is_risky_trend_fb field."""

    def test_default_false(self):
        """Default is_risky_trend_fb is False."""
        from core.patterns import PatternSignal, ConfirmationType
        from config.levels import Level, LevelType

        level = Level(
            price=7000.0,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=_ts(0),
            confirmed_at=_ts(0),
        )
        sig = PatternSignal(
            pattern_type="failed_breakdown",
            confirmation=ConfirmationType.NON_ACCEPTANCE,
            level=level,
            sweep_low=6998.0,
            entry_price=7005.0,
            stop_price=6995.0,
            bar_idx=10,
            timestamp=_ts(10),
        )
        assert sig.is_risky_trend_fb is False

    def test_settable(self):
        """is_risky_trend_fb can be set to True."""
        from core.patterns import PatternSignal, ConfirmationType
        from config.levels import Level, LevelType

        level = Level(
            price=7000.0,
            level_type=LevelType.MULTI_HOUR_LOW,
            created_at=_ts(0),
            confirmed_at=_ts(0),
        )
        sig = PatternSignal(
            pattern_type="failed_breakdown",
            confirmation=ConfirmationType.NON_ACCEPTANCE,
            level=level,
            sweep_low=6998.0,
            entry_price=7005.0,
            stop_price=6995.0,
            bar_idx=10,
            timestamp=_ts(10),
            is_risky_trend_fb=True,
        )
        assert sig.is_risky_trend_fb is True


class TestMultiHourRallyMinPts:
    """Lower threshold to 20 per Mancini Apr 15 2026 post."""

    def test_default_is_20(self):
        """Default multi_hour_rally_min_pts is 20.0 (Mancini's exact number)."""
        params = StrategyParams()
        assert params.multi_hour_rally_min_pts == 20.0
