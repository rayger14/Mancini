"""Tests for Mancini-faithful short patterns (BreakdownShort + BacktestShort + VelocityBreakdownShort)."""

from datetime import datetime, timedelta

import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.patterns_short_v2 import BreakdownShort, BacktestShort, VelocityBreakdownShort, ShortState


def _ts(minute: int) -> datetime:
    """Helper to create timestamps at minute offsets from 9:30."""
    base = datetime(2024, 6, 15, 9, 30)
    return base + timedelta(minutes=minute)


def _make_level_store(levels: list[tuple[float, LevelType]]) -> LevelStore:
    """Create a LevelStore with pre-confirmed levels."""
    store = LevelStore()
    ts = datetime(2024, 6, 15, 9, 0)
    for price, lt in levels:
        store.add(Level(price=price, level_type=lt, created_at=ts, confirmed_at=ts))
    return store


class TestBreakdownShort:
    """Tests for BreakdownShort pattern."""

    def test_confirmed_breakdown(self):
        """Price breaks below support and stays below for confirm_bars → signal."""
        params = StrategyParams(
            allow_breakdown_short=True,
            bd_min_break_depth_pts=1.0,
            bd_confirm_bars=5,
            bd_timeout_bars=20,
            bd_stop_buffer_pts=3.0,
            bd_max_break_depth_pts=15.0,
            bd_conviction_threshold=5.0,  # match bd_confirm_bars (weights=0 → 1.0/bar)
            bd_min_bars_floor=3,
        )
        bd = BreakdownShort(params)
        store = _make_level_store([(5000.0, LevelType.PRIOR_DAY_LOW)])

        # Bar 0: Break below level
        sig = bd.update(0, _ts(0), high=5001.0, low=4998.0, close=4998.5, level_store=store)
        assert sig is None
        assert bd.state == ShortState.BREAK_DETECTED

        # Bars 1-4: Stay below
        for i in range(1, 5):
            sig = bd.update(i, _ts(i), high=4999.5, low=4997.0, close=4998.0, level_store=store)
            if i < 4:
                assert sig is None

        # Bar 4 is the 5th bar below (0-indexed: bars 0,1,2,3,4 = 5 bars)
        assert sig is not None
        assert sig.pattern_type == "breakdown_short"
        assert sig.direction == "short"
        assert sig.stop_price == 5000.0 + 3.0  # level + buffer

    def test_recovery_aborts(self):
        """Price breaks below then recovers above → no signal (this is a FB)."""
        params = StrategyParams(
            allow_breakdown_short=True,
            bd_min_break_depth_pts=1.0,
            bd_confirm_bars=10,
            bd_require_major_level=False,
            bd_conviction_threshold=10.0,
            bd_min_bars_floor=3,
        )
        bd = BreakdownShort(params)
        store = _make_level_store([(5000.0, LevelType.CLUSTER_LOW)])

        # Break below
        bd.update(0, _ts(0), high=5001.0, low=4998.0, close=4998.5, level_store=store)
        assert bd.state == ShortState.BREAK_DETECTED

        # Recover above level
        sig = bd.update(1, _ts(1), high=5002.0, low=4999.0, close=5001.0, level_store=store)
        assert sig is None
        assert bd.state == ShortState.IDLE  # Reset — this was a failed breakdown

    def test_max_depth_rejects(self):
        """Price breaks too far below level → rejected (late entry)."""
        params = StrategyParams(
            allow_breakdown_short=True,
            bd_min_break_depth_pts=1.0,
            bd_confirm_bars=3,
            bd_max_break_depth_pts=10.0,
            bd_conviction_threshold=3.0,
            bd_min_bars_floor=2,
        )
        bd = BreakdownShort(params)
        store = _make_level_store([(5000.0, LevelType.MULTI_HOUR_LOW)])

        # Break with close already 12 pts below
        bd.update(0, _ts(0), high=4995.0, low=4987.0, close=4988.0, level_store=store)
        assert bd.state == ShortState.BREAK_DETECTED

        # Close moves even further — exceeds max depth
        sig = bd.update(1, _ts(1), high=4990.0, low=4985.0, close=4988.0, level_store=store)
        assert sig is None
        assert bd.state == ShortState.IDLE  # Rejected: too far below

    def test_timeout_resets(self):
        """If breakdown doesn't confirm within timeout, reset."""
        params = StrategyParams(
            allow_breakdown_short=True,
            bd_min_break_depth_pts=1.0,
            bd_confirm_bars=50,  # Very high — will timeout first
            bd_timeout_bars=10,
            bd_conviction_threshold=50.0,
            bd_min_bars_floor=5,
        )
        bd = BreakdownShort(params)
        store = _make_level_store([(5000.0, LevelType.PRIOR_DAY_LOW)])

        bd.update(0, _ts(0), high=5001.0, low=4998.0, close=4998.5, level_store=store)
        assert bd.state == ShortState.BREAK_DETECTED

        # 11 bars later → timeout
        sig = bd.update(11, _ts(11), high=4999.0, low=4997.0, close=4998.0, level_store=store)
        assert sig is None
        assert bd.state == ShortState.IDLE

    def test_only_significant_levels(self):
        """SWING_LOW should NOT trigger a breakdown short."""
        params = StrategyParams(
            allow_breakdown_short=True,
            bd_min_break_depth_pts=1.0,
            bd_confirm_bars=3,
        )
        bd = BreakdownShort(params)
        store = _make_level_store([(5000.0, LevelType.SWING_LOW)])

        bd.update(0, _ts(0), high=5001.0, low=4998.0, close=4998.5, level_store=store)
        assert bd.state == ShortState.IDLE  # SWING_LOW not in _SUPPORT_TYPES

    def test_min_depth_filter(self):
        """Tiny break below level (< min_depth) should not trigger."""
        params = StrategyParams(
            allow_breakdown_short=True,
            bd_min_break_depth_pts=2.0,
            bd_confirm_bars=3,
        )
        bd = BreakdownShort(params)
        store = _make_level_store([(5000.0, LevelType.PRIOR_DAY_LOW)])

        # Only 0.5 pts below — less than min_depth
        bd.update(0, _ts(0), high=5001.0, low=4999.5, close=4999.7, level_store=store)
        assert bd.state == ShortState.IDLE


class TestBacktestShort:
    """Tests for BacktestShort pattern."""

    def test_full_sequence(self):
        """Breakout → pullback → failed backtest → signal."""
        params = StrategyParams(
            allow_backtest_short=True,
            bt_breakout_confirm_bars=3,
            bt_pullback_min_pts=3.0,
            bt_confirm_bars=3,
            bt_timeout_bars=20,
            bt_stop_buffer_pts=3.0,
            bt_reclaim_abort_bars=3,
            bt_max_distance_from_level=2.0,
        )
        bt = BacktestShort(params)
        store = _make_level_store([(5100.0, LevelType.PRIOR_DAY_HIGH)])

        # Phase 1: Breakout — 3 bars closing above resistance
        for i in range(3):
            sig = bt.update(i, _ts(i), high=5105.0, low=5101.0, close=5103.0, level_store=store)
            assert sig is None

        # Verify breakout was recorded
        assert len(bt._broken_resistances) == 1

        # Phase 2: Pullback — price drops well below resistance (out of backtest range)
        for i in range(3, 6):
            sig = bt.update(i, _ts(i), high=5092.0, low=5088.0, close=5090.0, level_store=store)
            assert sig is None

        # Phase 3: Backtest — price approaches resistance from below, closes below
        sig = bt.update(6, _ts(6), high=5099.5, low=5096.0, close=5097.0, level_store=store)
        assert sig is None
        assert bt.state == ShortState.BACKTEST_WATCH

        # Phase 4: Rejection confirmed — 2 more bars below
        sig = bt.update(7, _ts(7), high=5098.0, low=5094.0, close=5095.0, level_store=store)
        assert sig is None
        sig = bt.update(8, _ts(8), high=5097.0, low=5093.0, close=5094.0, level_store=store)
        assert sig is not None
        assert sig.pattern_type == "backtest_short"
        assert sig.direction == "short"

    def test_no_breakout_no_signal(self):
        """Without prior breakout, backtest cannot trigger."""
        params = StrategyParams(
            allow_backtest_short=True,
            bt_breakout_confirm_bars=5,
            bt_confirm_bars=3,
        )
        bt = BacktestShort(params)
        store = _make_level_store([(5100.0, LevelType.PRIOR_DAY_HIGH)])

        # Only 2 bars above (need 5 for breakout)
        bt.update(0, _ts(0), high=5105.0, low=5101.0, close=5103.0, level_store=store)
        bt.update(1, _ts(1), high=5105.0, low=5101.0, close=5103.0, level_store=store)

        # Pullback
        bt.update(2, _ts(2), high=5098.0, low=5094.0, close=5095.0, level_store=store)

        # Touch from below
        bt.update(3, _ts(3), high=5099.0, low=5096.0, close=5097.0, level_store=store)
        assert bt.state == ShortState.IDLE  # No breakout recorded

    def test_successful_backtest_no_signal(self):
        """If price reclaims the level (backtest succeeds), no short signal."""
        params = StrategyParams(
            allow_backtest_short=True,
            bt_breakout_confirm_bars=2,
            bt_pullback_min_pts=3.0,
            bt_confirm_bars=5,
            bt_reclaim_abort_bars=3,
            bt_max_distance_from_level=2.0,
        )
        bt = BacktestShort(params)
        store = _make_level_store([(5100.0, LevelType.CLUSTER_HIGH)])

        # Breakout
        bt.update(0, _ts(0), high=5105.0, low=5101.0, close=5103.0, level_store=store)
        bt.update(1, _ts(1), high=5106.0, low=5102.0, close=5104.0, level_store=store)

        # Pullback — well below level so not detected as backtest yet
        for i in range(2, 6):
            bt.update(i, _ts(i), high=5092.0, low=5088.0, close=5090.0, level_store=store)

        # Backtest touch — approaches level from below
        bt.update(6, _ts(6), high=5099.0, low=5096.0, close=5097.0, level_store=store)
        assert bt.state == ShortState.BACKTEST_WATCH

        # Price reclaims — backtest succeeds (close above level)
        bt.update(7, _ts(7), high=5103.0, low=5099.0, close=5101.0, level_store=store)
        # After reclaim, wait for reclaim_abort_bars
        bt.update(8, _ts(8), high=5104.0, low=5100.0, close=5102.0, level_store=store)
        bt.update(9, _ts(9), high=5105.0, low=5101.0, close=5103.0, level_store=store)
        bt.update(10, _ts(10), high=5106.0, low=5102.0, close=5104.0, level_store=store)
        assert bt.state == ShortState.IDLE  # Aborted — backtest succeeded

    def test_expired_breakout(self):
        """Breakout too old should not trigger backtest."""
        params = StrategyParams(
            allow_backtest_short=True,
            bt_breakout_confirm_bars=2,
            bt_pullback_min_pts=3.0,
            bt_confirm_bars=3,
            bt_max_distance_from_level=2.0,
        )
        bt = BacktestShort(params)
        bt._breakout_expire_bars = 10  # Short expiry for testing
        store = _make_level_store([(5100.0, LevelType.PRIOR_DAY_HIGH)])

        # Breakout at bar 0-1
        bt.update(0, _ts(0), high=5105.0, low=5101.0, close=5103.0, level_store=store)
        bt.update(1, _ts(1), high=5106.0, low=5102.0, close=5104.0, level_store=store)
        assert len(bt._broken_resistances) == 1

        # 12 bars later — breakout expired
        bt.update(12, _ts(12), high=5098.0, low=5094.0, close=5095.0, level_store=store)
        assert len(bt._broken_resistances) == 0  # Expired


class TestVelocityBreakdownShort:
    """Tests for VelocityBreakdownShort pattern."""

    def _default_params(self, **overrides) -> StrategyParams:
        defaults = dict(
            allow_velocity_short=True,
            vbd_min_break_pts=8.0,
            vbd_min_volume_ratio=3.0,
            vbd_require_close_below=True,
            vbd_stop_buffer_pts=3.0,
            vbd_only_major_levels=True,
        )
        defaults.update(overrides)
        return StrategyParams(**defaults)

    def test_velocity_breakdown_fires(self):
        """High-volume bar breaking PDL by 10+ pts with close below -> signal."""
        params = self._default_params()
        vbd = VelocityBreakdownShort(params)
        store = _make_level_store([(5400.0, LevelType.PRIOR_DAY_LOW)])

        avg_vol = 1000.0  # 20-bar average
        sig = vbd.update(
            bar_idx=50,
            timestamp=_ts(50),
            high=5402.0,
            low=5390.0,   # 10 pts below level
            close=5391.0, # closed below level
            volume=5000.0, # 5x average
            avg_volume_20=avg_vol,
            level_store=store,
        )
        assert sig is not None
        assert sig.pattern_type == "velocity_short"
        assert sig.direction == "short"
        assert sig.entry_price == 5391.0
        assert sig.stop_price == 5400.0 + 3.0  # level + buffer
        assert sig.sweep_depth_pts == 10.0

    def test_insufficient_volume_no_signal(self):
        """Bar breaks level but volume is only 2x avg (need 3x) -> no signal."""
        params = self._default_params()
        vbd = VelocityBreakdownShort(params)
        store = _make_level_store([(5400.0, LevelType.PRIOR_DAY_LOW)])

        sig = vbd.update(
            bar_idx=50,
            timestamp=_ts(50),
            high=5402.0,
            low=5390.0,
            close=5391.0,
            volume=2000.0,  # only 2x avg
            avg_volume_20=1000.0,
            level_store=store,
        )
        assert sig is None

    def test_insufficient_break_depth_no_signal(self):
        """Volume is high but break is only 5 pts (need 8) -> no signal."""
        params = self._default_params()
        vbd = VelocityBreakdownShort(params)
        store = _make_level_store([(5400.0, LevelType.PRIOR_DAY_LOW)])

        sig = vbd.update(
            bar_idx=50,
            timestamp=_ts(50),
            high=5402.0,
            low=5395.0,   # only 5 pts below
            close=5396.0,
            volume=5000.0,
            avg_volume_20=1000.0,
            level_store=store,
        )
        assert sig is None

    def test_close_above_level_no_signal(self):
        """Bar wicks below but closes above level -> no signal (require_close_below)."""
        params = self._default_params()
        vbd = VelocityBreakdownShort(params)
        store = _make_level_store([(5400.0, LevelType.PRIOR_DAY_LOW)])

        sig = vbd.update(
            bar_idx=50,
            timestamp=_ts(50),
            high=5405.0,
            low=5390.0,   # 10 pts below
            close=5401.0, # closed ABOVE level
            volume=5000.0,
            avg_volume_20=1000.0,
            level_store=store,
        )
        assert sig is None

    def test_close_below_not_required(self):
        """With require_close_below=False, wick break is enough."""
        params = self._default_params(vbd_require_close_below=False)
        vbd = VelocityBreakdownShort(params)
        store = _make_level_store([(5400.0, LevelType.PRIOR_DAY_LOW)])

        sig = vbd.update(
            bar_idx=50,
            timestamp=_ts(50),
            high=5405.0,
            low=5390.0,
            close=5401.0,  # above level but not required
            volume=5000.0,
            avg_volume_20=1000.0,
            level_store=store,
        )
        assert sig is not None

    def test_only_major_levels(self):
        """With vbd_only_major_levels=True, CLUSTER_LOW should not trigger."""
        params = self._default_params(vbd_only_major_levels=True)
        vbd = VelocityBreakdownShort(params)
        store = _make_level_store([(5400.0, LevelType.CLUSTER_LOW)])

        sig = vbd.update(
            bar_idx=50,
            timestamp=_ts(50),
            high=5402.0,
            low=5390.0,
            close=5391.0,
            volume=5000.0,
            avg_volume_20=1000.0,
            level_store=store,
        )
        assert sig is None

    def test_all_levels_allowed(self):
        """With vbd_only_major_levels=False, CLUSTER_LOW should trigger."""
        params = self._default_params(vbd_only_major_levels=False)
        vbd = VelocityBreakdownShort(params)
        store = _make_level_store([(5400.0, LevelType.CLUSTER_LOW)])

        sig = vbd.update(
            bar_idx=50,
            timestamp=_ts(50),
            high=5402.0,
            low=5390.0,
            close=5391.0,
            volume=5000.0,
            avg_volume_20=1000.0,
            level_store=store,
        )
        assert sig is not None

    def test_multi_hour_low_triggers(self):
        """Multi-hour low is a major level and should trigger."""
        params = self._default_params()
        vbd = VelocityBreakdownShort(params)
        store = _make_level_store([(5400.0, LevelType.MULTI_HOUR_LOW)])

        sig = vbd.update(
            bar_idx=50,
            timestamp=_ts(50),
            high=5402.0,
            low=5390.0,
            close=5391.0,
            volume=5000.0,
            avg_volume_20=1000.0,
            level_store=store,
        )
        assert sig is not None

    def test_zero_avg_volume_no_signal(self):
        """If avg_volume_20 is 0, should not crash or fire."""
        params = self._default_params()
        vbd = VelocityBreakdownShort(params)
        store = _make_level_store([(5400.0, LevelType.PRIOR_DAY_LOW)])

        sig = vbd.update(
            bar_idx=50,
            timestamp=_ts(50),
            high=5402.0,
            low=5390.0,
            close=5391.0,
            volume=5000.0,
            avg_volume_20=0.0,
            level_store=store,
        )
        assert sig is None

    def test_swing_low_never_triggers(self):
        """SWING_LOW is not in any support set for velocity breakdown."""
        params = self._default_params(vbd_only_major_levels=False)
        vbd = VelocityBreakdownShort(params)
        store = _make_level_store([(5400.0, LevelType.SWING_LOW)])

        sig = vbd.update(
            bar_idx=50,
            timestamp=_ts(50),
            high=5402.0,
            low=5390.0,
            close=5391.0,
            volume=5000.0,
            avg_volume_20=1000.0,
            level_store=store,
        )
        assert sig is None
