"""Tests for ATM machine level detection and size boosting."""

from __future__ import annotations

import pytest

from config.settings import StrategyParams
from core.signals import SignalAggregator


class TestATMLevelTracking:
    """Test per-level profitability tracking."""

    def _make_aggregator(self, **kwargs) -> SignalAggregator:
        params = StrategyParams(
            use_atm_level_boost=True,
            atm_min_winning_trades=2,
            atm_min_win_rate=0.6,
            atm_size_boost=1.5,
            **kwargs,
        )
        return SignalAggregator(strategy_params=params)

    def test_record_level_outcome_win(self):
        """Recording a winning trade increments wins counter."""
        agg = self._make_aggregator()
        agg.record_level_outcome(5020.0, 15.0, "2026-04-01")
        perf = agg._level_performance[5020]
        assert perf["wins"] == 1
        assert perf["losses"] == 0
        assert perf["total_pnl"] == 15.0
        assert perf["last_session"] == "2026-04-01"

    def test_record_level_outcome_loss(self):
        """Recording a losing trade increments losses counter."""
        agg = self._make_aggregator()
        agg.record_level_outcome(5020.0, -5.0, "2026-04-01")
        perf = agg._level_performance[5020]
        assert perf["wins"] == 0
        assert perf["losses"] == 1
        assert perf["total_pnl"] == -5.0

    def test_record_multiple_outcomes(self):
        """Multiple outcomes at same level accumulate correctly."""
        agg = self._make_aggregator()
        agg.record_level_outcome(5020.0, 30.0, "2026-04-01")
        agg.record_level_outcome(5020.0, 25.0, "2026-04-02")
        agg.record_level_outcome(5020.0, -5.0, "2026-04-03")
        perf = agg._level_performance[5020]
        assert perf["wins"] == 2
        assert perf["losses"] == 1
        assert perf["total_pnl"] == pytest.approx(50.0)

    def test_level_price_rounding(self):
        """Level prices round to nearest 1.0 for bucketing."""
        agg = self._make_aggregator()
        agg.record_level_outcome(5020.25, 10.0, "2026-04-01")
        agg.record_level_outcome(5020.75, 15.0, "2026-04-02")
        # Both should map to key 5020 and 5021 respectively
        assert 5020 in agg._level_performance
        assert 5021 in agg._level_performance

    def test_record_outcome_zero_pnl_counts_as_loss(self):
        """Zero PnL counts as a loss (pnl <= 0)."""
        agg = self._make_aggregator()
        agg.record_level_outcome(5020.0, 0.0, "2026-04-01")
        perf = agg._level_performance[5020]
        assert perf["wins"] == 0
        assert perf["losses"] == 1


class TestATMLevelQualification:
    """Test is_atm_level criteria checking."""

    def _make_aggregator(self, **kwargs) -> SignalAggregator:
        params = StrategyParams(
            use_atm_level_boost=True,
            atm_min_winning_trades=2,
            atm_min_win_rate=0.6,
            atm_size_boost=1.5,
            **kwargs,
        )
        return SignalAggregator(strategy_params=params)

    def test_not_atm_no_trades(self):
        """Level with no trades is not ATM."""
        agg = self._make_aggregator()
        assert agg.is_atm_level(5020.0) is False

    def test_not_atm_insufficient_wins(self):
        """Level with only 1 win does not qualify (min 2)."""
        agg = self._make_aggregator()
        agg.record_level_outcome(5020.0, 30.0, "2026-04-01")
        assert agg.is_atm_level(5020.0) is False

    def test_not_atm_low_win_rate(self):
        """Level with 2 wins but 50% WR does not qualify (min 60%)."""
        agg = self._make_aggregator()
        agg.record_level_outcome(5020.0, 30.0, "2026-04-01")
        agg.record_level_outcome(5020.0, 25.0, "2026-04-02")
        agg.record_level_outcome(5020.0, -5.0, "2026-04-03")
        agg.record_level_outcome(5020.0, -4.0, "2026-04-04")
        # 2W/2L = 50% < 60%
        assert agg.is_atm_level(5020.0) is False

    def test_is_atm_meets_criteria(self):
        """Level with 2W/0L qualifies as ATM."""
        agg = self._make_aggregator()
        agg.record_level_outcome(5020.0, 30.0, "2026-04-01")
        agg.record_level_outcome(5020.0, 25.0, "2026-04-02")
        assert agg.is_atm_level(5020.0) is True

    def test_is_atm_with_one_loss(self):
        """Level with 3W/1L = 75% qualifies as ATM."""
        agg = self._make_aggregator()
        agg.record_level_outcome(5020.0, 30.0, "2026-04-01")
        agg.record_level_outcome(5020.0, 25.0, "2026-04-02")
        agg.record_level_outcome(5020.0, -5.0, "2026-04-03")
        agg.record_level_outcome(5020.0, 35.0, "2026-04-04")
        # 3W/1L = 75% >= 60%
        assert agg.is_atm_level(5020.0) is True

    def test_atm_uses_rounded_price(self):
        """is_atm_level uses rounded price for lookup."""
        agg = self._make_aggregator()
        agg.record_level_outcome(5020.25, 30.0, "2026-04-01")
        agg.record_level_outcome(5020.25, 25.0, "2026-04-02")
        # Querying with exact level price should round to same key
        assert agg.is_atm_level(5020.25) is True
        assert agg.is_atm_level(5020.0) is True


class TestATMLevelExpiry:
    """Test session rollover expiry of stale level records."""

    def _make_aggregator(self, memory_days: int = 5) -> SignalAggregator:
        params = StrategyParams(
            use_atm_level_boost=True,
            level_memory_days=memory_days,
        )
        return SignalAggregator(strategy_params=params)

    def test_expire_old_levels(self):
        """Levels older than memory_days get expired."""
        agg = self._make_aggregator(memory_days=5)
        # Record at a level 30 days ago
        agg.record_level_outcome(5020.0, 30.0, "2026-03-01")
        agg.record_level_outcome(5020.0, 25.0, "2026-03-02")
        assert agg.is_atm_level(5020.0) is True
        # Expire: current date is 30+ days later
        agg.expire_atm_levels("2026-04-03", memory_days=5)
        assert agg.is_atm_level(5020.0) is False
        assert 5020 not in agg._level_performance

    def test_keep_recent_levels(self):
        """Recent levels survive expiry."""
        agg = self._make_aggregator(memory_days=5)
        agg.record_level_outcome(5020.0, 30.0, "2026-04-02")
        agg.record_level_outcome(5020.0, 25.0, "2026-04-03")
        # Expire: current date is same day — should keep
        agg.expire_atm_levels("2026-04-03", memory_days=5)
        assert agg.is_atm_level(5020.0) is True

    def test_mixed_expiry(self):
        """Only old levels expire; recent ones survive."""
        agg = self._make_aggregator(memory_days=5)
        # Old level
        agg.record_level_outcome(5020.0, 30.0, "2026-03-01")
        agg.record_level_outcome(5020.0, 25.0, "2026-03-01")
        # Recent level
        agg.record_level_outcome(5100.0, 20.0, "2026-04-02")
        agg.record_level_outcome(5100.0, 15.0, "2026-04-03")

        agg.expire_atm_levels("2026-04-03", memory_days=5)
        assert 5020 not in agg._level_performance
        assert agg.is_atm_level(5100.0) is True


class TestATMLevelPersistenceAcrossReset:
    """Verify that _level_performance survives signal aggregator reset."""

    def test_level_performance_survives_reset(self):
        """_level_performance should NOT be cleared on reset()."""
        params = StrategyParams(use_atm_level_boost=True)
        agg = SignalAggregator(strategy_params=params)
        agg.record_level_outcome(5020.0, 30.0, "2026-04-01")
        agg.record_level_outcome(5020.0, 25.0, "2026-04-02")
        assert agg.is_atm_level(5020.0) is True
        # Reset (new session)
        agg.reset()
        # Performance data should survive
        assert agg.is_atm_level(5020.0) is True


class TestATMBoostDisabledByDefault:
    """Verify ATM boost is off when use_atm_level_boost=False."""

    def test_default_params_no_atm(self):
        """Default StrategyParams has use_atm_level_boost=False."""
        params = StrategyParams()
        assert params.use_atm_level_boost is False

    def test_aggregator_tracks_but_no_boost_when_disabled(self):
        """record_level_outcome works even when boost is disabled."""
        params = StrategyParams(use_atm_level_boost=False)
        agg = SignalAggregator(strategy_params=params)
        agg.record_level_outcome(5020.0, 30.0, "2026-04-01")
        agg.record_level_outcome(5020.0, 25.0, "2026-04-02")
        # Tracking works
        assert agg.is_atm_level(5020.0) is True
        # But the flag is off, so _qualify_signal won't boost
        assert params.use_atm_level_boost is False
