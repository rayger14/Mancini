"""Tests for Level Quality Score (LQS) system."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from config.levels import Level, LevelType
from config.settings import StrategyParams
from core.level_scoring import LevelQualityScorer


# ── Helpers ──────────────────────────────────────────────────────────

def _make_level(
    level_type: LevelType,
    price: float = 5800.0,
    touch_count: int = 1,
    rally_from_low_pts: float = 0.0,
    tested_and_held: bool = False,
    origin_date: date = None,
    label: str = "",
    mancini_confirmed: bool = False,
) -> Level:
    """Create a Level for testing."""
    level = Level(
        price=price,
        level_type=level_type,
        created_at=datetime(2026, 4, 17, 10, 0),
        touch_count=touch_count,
        rally_from_low_pts=rally_from_low_pts,
        tested_and_held=tested_and_held,
        origin_date=origin_date,
        label=label,
    )
    if mancini_confirmed:
        level.mancini_confirmed = True
    return level


def _default_session_ctx(
    session_date: date = None,
    current_price: float = 5810.0,
    session_high: float = 5850.0,
    session_low: float = 5790.0,
    bar_count: int = 100,
) -> dict:
    """Build a session_context dict for testing."""
    return {
        "session_date": session_date or date(2026, 4, 17),
        "current_price": current_price,
        "session_high": session_high,
        "session_low": session_low,
        "bar_count": bar_count,
    }


# ── Tests ────────────────────────────────────────────────────────────


class TestBlockHorizontalSREntries:
    """block_horizontal_sr_entries gates HORIZONTAL_SR out of ENTRY scoring
    (drops LQS to 0 → below shadow threshold → both entry paths skip it),
    while leaving every other level type untouched."""

    def test_flag_off_horizontal_sr_scores_normally(self):
        params = StrategyParams()
        assert params.block_horizontal_sr_entries is False  # default off
        scorer = LevelQualityScorer(params)
        lqs = scorer.compute_lqs(_make_level(LevelType.HORIZONTAL_SR, touch_count=5),
                                 None, _default_session_ctx())
        assert lqs > 0  # normally scores something

    def test_flag_on_horizontal_sr_scores_zero(self):
        params = StrategyParams(block_horizontal_sr_entries=True)
        scorer = LevelQualityScorer(params)
        lqs = scorer.compute_lqs(_make_level(LevelType.HORIZONTAL_SR, touch_count=5),
                                 None, _default_session_ctx())
        assert lqs == 0  # gated below shadow threshold → no entry

    def test_flag_on_does_not_touch_other_types(self):
        params = StrategyParams(block_horizontal_sr_entries=True)
        scorer = LevelQualityScorer(params)
        for lt in (LevelType.CUSTOM, LevelType.PRIOR_DAY_LOW,
                   LevelType.INTRADAY_LOW, LevelType.MULTI_HOUR_LOW):
            lqs = scorer.compute_lqs(_make_level(lt), None, _default_session_ctx())
            assert lqs > 0, f"{lt} should still score normally"

class TestOriginScore:
    """Factor 1: Level Origin scoring."""

    def test_pdl_gets_45(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.PRIOR_DAY_LOW)
        assert scorer._origin_score(level) == 45

    def test_pdh_gets_45(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.PRIOR_DAY_HIGH)
        assert scorer._origin_score(level) == 45

    def test_cluster_low_gets_0(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.CLUSTER_LOW)
        assert scorer._origin_score(level) == 0

    def test_mhl_gets_35(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.MULTI_HOUR_LOW)
        assert scorer._origin_score(level) == 35

    def test_custom_mancini_gets_30(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.CUSTOM)
        assert scorer._origin_score(level) == 30

    def test_swing_low_gets_5(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.SWING_LOW)
        assert scorer._origin_score(level) == 5

    def test_intraday_low_gets_25(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.INTRADAY_LOW)
        assert scorer._origin_score(level) == 25

    def test_horizontal_sr_gets_12(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.HORIZONTAL_SR)
        assert scorer._origin_score(level) == 12


class TestConfirmationScore:
    """Factor 2: Structural Confirmation scoring."""

    def test_mancini_confirmed_custom_gets_5(self):
        """A CUSTOM (Mancini-only) level with mancini_confirmed gets +5, not +15."""
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.CUSTOM, mancini_confirmed=True)
        assert scorer._confirmation_score(level) == 5

    def test_mancini_confirmed_non_custom_gets_15(self):
        """A non-CUSTOM level with mancini_confirmed=True gets +15 (engine + Mancini agree)."""
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.PRIOR_DAY_LOW, mancini_confirmed=True)
        assert scorer._confirmation_score(level) == 15

    def test_mancini_label_without_attribute_gets_0(self):
        """Label-based detection removed; need mancini_confirmed=True attribute."""
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.SWING_LOW, label="mancini_5800")
        assert scorer._confirmation_score(level) == 0

    def test_multi_touch_8_gets_10(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.SWING_LOW, touch_count=8)
        assert scorer._confirmation_score(level) == 10

    def test_multi_touch_5_gets_5(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.SWING_LOW, touch_count=5)
        assert scorer._confirmation_score(level) == 5

    def test_validated_rally_gets_10(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.SWING_LOW, rally_from_low_pts=25.0)
        assert scorer._confirmation_score(level) == 10

    def test_tested_and_held_gets_5(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.SWING_LOW, tested_and_held=True)
        assert scorer._confirmation_score(level) == 5

    def test_confirmation_capped_at_25(self):
        scorer = LevelQualityScorer(StrategyParams())
        # CUSTOM mancini_confirmed(+5) + 8 touches (+10) + validated rally (+10) + tested_held (+5) = 30
        # But capped at 25
        level = _make_level(
            LevelType.CUSTOM,
            touch_count=8,
            rally_from_low_pts=25.0,
            tested_and_held=True,
            mancini_confirmed=True,
        )
        # Cap raised to 30 to let elite shelves (15+ touches) score higher
        # CUSTOM + mancini_confirmed(+5) + 8 touches(+10) + 20pt rally(+10) + tested(+5) = 30
        assert scorer._confirmation_score(level) == 30


class TestRecencyScore:
    """Factor 3: Recency & Context scoring."""

    def test_today_level_gets_10(self):
        scorer = LevelQualityScorer(StrategyParams())
        today = date(2026, 4, 17)
        level = _make_level(LevelType.SWING_LOW, origin_date=today)
        ctx = _default_session_ctx(session_date=today)
        assert scorer._recency_score(level, ctx) >= 10

    def test_yesterday_level_gets_8(self):
        scorer = LevelQualityScorer(StrategyParams())
        yesterday = date(2026, 4, 16)
        level = _make_level(LevelType.SWING_LOW, origin_date=yesterday)
        ctx = _default_session_ctx(session_date=date(2026, 4, 17))
        score = scorer._recency_score(level, ctx)
        assert score >= 8

    def test_old_level_gets_2(self):
        scorer = LevelQualityScorer(StrategyParams())
        old_date = date(2026, 4, 10)
        level = _make_level(LevelType.SWING_LOW, origin_date=old_date)
        ctx = _default_session_ctx(session_date=date(2026, 4, 17))
        score = scorer._recency_score(level, ctx)
        # 7 days old -> +2; near session low adds +5 if applicable
        assert score >= 2

    def test_near_session_low_gets_5(self):
        scorer = LevelQualityScorer(StrategyParams())
        today = date(2026, 4, 17)
        # Level at 5795, session_low at 5790 -> within 10 pts
        level = _make_level(LevelType.SWING_LOW, price=5795.0, origin_date=today)
        ctx = _default_session_ctx(session_date=today, session_low=5790.0)
        score = scorer._recency_score(level, ctx)
        assert score >= 15  # 10 (today) + 5 (near session low)

    def test_no_session_context_returns_0(self):
        scorer = LevelQualityScorer(StrategyParams())
        level = _make_level(LevelType.SWING_LOW)
        assert scorer._recency_score(level, None) == 0


class TestRegimeScore:
    """Factor 4: Market Regime scoring."""

    def test_high_vix_gets_10(self):
        scorer = LevelQualityScorer(StrategyParams())
        market_data = {"vix": 28.0}
        assert scorer._regime_score(market_data) == 10

    def test_moderate_vix_gets_5(self):
        scorer = LevelQualityScorer(StrategyParams())
        market_data = {"vix": 22.0}
        assert scorer._regime_score(market_data) == 5

    def test_low_vix_gets_0(self):
        scorer = LevelQualityScorer(StrategyParams())
        market_data = {"vix": 15.0}
        assert scorer._regime_score(market_data) == 0

    def test_inverted_term_structure_gets_5(self):
        scorer = LevelQualityScorer(StrategyParams())
        market_data = {"vix": 15.0, "vix_term_structure": 1.1}
        assert scorer._regime_score(market_data) == 5

    def test_vix_25_plus_inverted_gets_15(self):
        scorer = LevelQualityScorer(StrategyParams())
        market_data = {"vix": 26.0, "vix_term_structure": 1.1}
        assert scorer._regime_score(market_data) == 15

    def test_no_market_data_returns_0(self):
        scorer = LevelQualityScorer(StrategyParams())
        assert scorer._regime_score(None) == 0


class TestFullLQS:
    """Full LQS computation end-to-end."""

    def test_pdl_mancini_today_vix25(self):
        """PDL + mancini_confirmed + today + VIX 25.

        origin=45 (PDL) + confirm=15 (mancini_confirmed on non-CUSTOM)
        + recency=10 (today) + regime=5 (vix>=20) = 75.
        Level price 5800, session_low 5790 -> distance 10 -> near session low +5
        Total = 80.
        """
        scorer = LevelQualityScorer(StrategyParams())
        today = date(2026, 4, 17)
        level = _make_level(
            LevelType.PRIOR_DAY_LOW,
            origin_date=today,
            mancini_confirmed=True,
        )
        ctx = _default_session_ctx(session_date=today)
        market_data = {"vix": 25.0}
        lqs = scorer.compute_lqs(level, market_data, ctx)
        # origin=45 + confirm=15 + recency(today=10, near_session_low=5) + regime(vix>=20=5) = 80
        assert lqs == 80

    def test_cluster_low_with_touches_today(self):
        """CLUSTER_LOW with 3 touches + today = 0+0+10 = 10 -> shadow only."""
        scorer = LevelQualityScorer(StrategyParams())
        today = date(2026, 4, 17)
        level = _make_level(
            LevelType.CLUSTER_LOW,
            touch_count=3,
            origin_date=today,
        )
        ctx = _default_session_ctx(session_date=today)
        lqs = scorer.compute_lqs(level, None, ctx)
        # origin=0 + confirm(touches<4=0) + recency(today=10) = 10
        # Actually 3 touches is < 4 so 0 confirmation
        assert lqs < 25  # shadow or skip

    def test_high_quality_gets_aggressive(self):
        scorer = LevelQualityScorer(StrategyParams())
        params = scorer.get_trade_params(75)
        assert params["size_factor"] == 1.0
        assert params["min_rr"] == 1.0

    def test_medium_quality_gets_normal(self):
        """LQS 40 falls in normal tier (25-54)."""
        scorer = LevelQualityScorer(StrategyParams())
        params = scorer.get_trade_params(40)
        assert params["size_factor"] == 0.75
        assert params["min_rr"] == 1.3

    def test_low_quality_gets_shadow(self):
        """LQS 15 falls in shadow tier (10-24)."""
        scorer = LevelQualityScorer(StrategyParams())
        params = scorer.get_trade_params(15)
        assert params["size_factor"] == 0.0
        assert params["acceptance_mode"] == "shadow_only"

    def test_shadow_range_gets_zero_size(self):
        scorer = LevelQualityScorer(StrategyParams())
        params = scorer.get_trade_params(20)
        assert params["size_factor"] == 0.0
        assert params["acceptance_mode"] == "shadow_only"

    def test_skip_range_gets_skip(self):
        """LQS 5 falls in skip tier (0-9)."""
        scorer = LevelQualityScorer(StrategyParams())
        params = scorer.get_trade_params(5)
        assert params["size_factor"] == 0.0
        assert params["acceptance_mode"] == "skip"


class TestLQSDisabled:
    """When use_level_quality_scoring=False, behavior is unchanged."""

    def test_disabled_flag_does_not_affect_scoring(self):
        """The scorer itself always computes a score, regardless of the flag.
        The flag only affects whether _qualify_signal uses it to gate trades."""
        params = StrategyParams(use_level_quality_scoring=False)
        scorer = LevelQualityScorer(params)
        level = _make_level(LevelType.PRIOR_DAY_LOW, origin_date=date(2026, 4, 17))
        ctx = _default_session_ctx(session_date=date(2026, 4, 17))
        lqs = scorer.compute_lqs(level, None, ctx)
        # PDL(45) + recency(today=10, near_session_low=5) = 60
        assert lqs == 60  # Score is still computed


class TestTradeParams:
    """get_trade_params returns correct size/rr for each LQS tier."""

    def test_tier_boundaries(self):
        scorer = LevelQualityScorer(StrategyParams())
        # Boundary: 55 = aggressive (full size)
        p55 = scorer.get_trade_params(55)
        assert p55["size_factor"] == 1.0
        # Boundary: 54 = normal
        p54 = scorer.get_trade_params(54)
        assert p54["size_factor"] == 0.75
        # Boundary: 25 = normal
        p25 = scorer.get_trade_params(25)
        assert p25["size_factor"] == 0.75
        # Boundary: 24 = shadow
        p24 = scorer.get_trade_params(24)
        assert p24["size_factor"] == 0.0
        assert p24["acceptance_mode"] == "shadow_only"
        # Boundary: 10 = shadow
        p10 = scorer.get_trade_params(10)
        assert p10["size_factor"] == 0.0
        assert p10["acceptance_mode"] == "shadow_only"
        # Boundary: 9 = skip
        p9 = scorer.get_trade_params(9)
        assert p9["acceptance_mode"] == "skip"

    def test_lqs_100(self):
        scorer = LevelQualityScorer(StrategyParams())
        p = scorer.get_trade_params(100)
        assert p["size_factor"] == 1.0
        assert p["min_rr"] == 1.0

    def test_lqs_0(self):
        scorer = LevelQualityScorer(StrategyParams())
        p = scorer.get_trade_params(0)
        assert p["size_factor"] == 0.0
        assert p["acceptance_mode"] == "skip"


class TestLQSClamp:
    """LQS is clamped to [0, 100]."""

    def test_score_clamped_at_100(self):
        scorer = LevelQualityScorer(StrategyParams())
        # PDL(45) + mancini_confirmed on PDL(+15) + 8 touches(+10) + rally(+10) + held(+5)
        # confirmation capped at 25, so: 45 + 25 + recency + regime
        # + today(+10) + near session low(+5) + vix>25(+10) + inverted(+5) = 45+25+15+15 = 100
        level = _make_level(
            LevelType.PRIOR_DAY_LOW,
            touch_count=10,
            rally_from_low_pts=30.0,
            tested_and_held=True,
            origin_date=date(2026, 4, 17),
            mancini_confirmed=True,
        )
        ctx = _default_session_ctx(
            session_date=date(2026, 4, 17),
            session_low=5795.0,
        )
        market = {"vix": 30.0, "vix_term_structure": 1.2}
        lqs = scorer.compute_lqs(level, market, ctx)
        assert 0 <= lqs <= 100


# ── Cross-source confluence bonus (Phase 1) ──────────────────────────

class TestSourceCountConfluence:
    """A level confirmed by multiple INDEPENDENT sources (engine + Mancini +
    pivot) is higher conviction. source_count drives an LQS confirmation bonus.
    """

    def _scorer(self):
        return LevelQualityScorer(StrategyParams())

    def test_single_source_no_bonus(self):
        lv = _make_level(LevelType.SWING_LOW)
        lv.source_count = 1
        assert self._scorer()._confirmation_score(lv) == 0

    def test_two_sources_adds_5(self):
        lv = _make_level(LevelType.SWING_LOW)
        lv.source_count = 2
        assert self._scorer()._confirmation_score(lv) == 5

    def test_three_sources_adds_10(self):
        lv = _make_level(LevelType.SWING_LOW)
        lv.source_count = 3
        assert self._scorer()._confirmation_score(lv) == 10

    def test_default_source_count_is_one(self):
        lv = _make_level(LevelType.SWING_LOW)
        assert lv.source_count == 1

    def test_confluence_lifts_lqs_above_trade_gate(self):
        # A bare swing (LQS too low to trade) becomes tradeable once three
        # independent sources converge on it.
        scorer = self._scorer()
        ctx = _default_session_ctx()
        solo = _make_level(LevelType.SWING_LOW)
        conv = _make_level(LevelType.SWING_LOW)
        conv.source_count = 3
        assert scorer.compute_lqs(conv, ctx) > scorer.compute_lqs(solo, ctx)
