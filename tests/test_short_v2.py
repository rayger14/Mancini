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


def _make_level_store(
    levels: list[tuple[float, LevelType]], touches: int = 5
) -> LevelStore:
    """Create a LevelStore with pre-confirmed levels. `touches` sets
    Level.touch_count so shelf-strength gates (BacktestShort) can be
    exercised."""
    store = LevelStore()
    ts = datetime(2024, 6, 15, 9, 0)
    for price, lt in levels:
        store.add(Level(price=price, level_type=lt,
                        created_at=ts, confirmed_at=ts,
                        touch_count=touches))
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
    """Tests for the Mancini-faithful BacktestShort pattern.

    Per Mancini (2024-10-09): "Back-Test Shorts have a few criteria.
      1. Price must have set a clearly defined support [shelf].
      2. Price must break down that support decisively. Forceful, deep
         breakdown that ideally lasts hours, days, or weeks.
      3. Price must back-test the level from below. The first retest is
         typically actionable; odds drop with each successive test."
    """

    def _params(self, **overrides):
        defaults = dict(
            allow_backtest_short=True,
            bts_support_min_touches=3,
            bts_breakdown_confirm_bars=3,
            bts_min_flush_depth_pts=10.0,
            bts_max_distance_from_level=2.0,
            bts_confirm_bars=2,
            bts_reclaim_abort_bars=3,
            bts_timeout_bars=20,
            bts_stop_buffer_pts=3.0,
            bts_first_touch_only=True,
            bts_breakout_expire_bars=200,
        )
        defaults.update(overrides)
        return StrategyParams(**defaults)

    def _shelf(self, price=5100.0, level_type=LevelType.CLUSTER_LOW, touches=5):
        return _make_level_store([(price, level_type)], touches=touches)

    def test_full_sequence_emits_signal(self):
        """Mancini 3-criteria happy path: shelf set, decisive breakdown with
        deep flush, retest from below, rejection → SHORT signal."""
        params = self._params()
        bt = BacktestShort(params)
        store = self._shelf(price=5100.0, level_type=LevelType.CLUSTER_LOW, touches=5)

        # Phase 1: price is above the shelf
        bt.update(0, _ts(0), high=5108.0, low=5102.0, close=5105.0, level_store=store)

        # Phase 2: decisive breakdown — 3 consecutive closes below
        bt.update(1, _ts(1), high=5101.0, low=5097.0, close=5098.0, level_store=store)
        bt.update(2, _ts(2), high=5098.0, low=5094.0, close=5095.0, level_store=store)
        bt.update(3, _ts(3), high=5095.0, low=5089.0, close=5090.0, level_store=store)

        # Verify shelf is now registered as broken
        assert 5100.0 in bt._broken_supports

        # Phase 3: deep flush — drop another 10pt to satisfy min_flush_depth
        for i in range(4, 8):
            bt.update(i, _ts(i), high=5089.0, low=5085.0, close=5087.0, level_store=store)
        # Confirm deep_flush flag now set
        assert bt._broken_supports[5100.0]["deep_flush"] is True

        # Phase 4: retest from below — price rallies up to within 2pt of shelf,
        # but close stays below
        sig = bt.update(8, _ts(8), high=5099.0, low=5095.0, close=5097.0, level_store=store)
        assert sig is None
        assert bt.state == ShortState.BACKTEST_WATCH

        # Phase 5: rejection — 2 bars of close-below confirm
        sig = bt.update(9, _ts(9), high=5097.0, low=5094.0, close=5095.0, level_store=store)
        assert sig is None  # 1 of 2 bars below
        sig = bt.update(10, _ts(10), high=5096.0, low=5093.0, close=5094.0, level_store=store)
        assert sig is not None
        assert sig.pattern_type == "backtest_short"
        assert sig.direction == "short"
        # Stop above the rejection wick high (5099.0 + 3.0 buffer)
        assert sig.stop_price == 5099.0 + 3.0

    def test_no_signal_without_decisive_breakdown(self):
        """Mancini criterion 2 fails: price only briefly tags below the shelf
        but doesn't flush deeply. No short setup."""
        params = self._params(bts_min_flush_depth_pts=20.0)  # need 20pt flush
        bt = BacktestShort(params)
        store = self._shelf(price=5100.0)

        # Decisive close below (3 bars)
        bt.update(0, _ts(0), high=5105.0, low=5101.0, close=5103.0, level_store=store)
        bt.update(1, _ts(1), high=5099.0, low=5095.0, close=5097.0, level_store=store)
        bt.update(2, _ts(2), high=5096.0, low=5094.0, close=5095.0, level_store=store)
        bt.update(3, _ts(3), high=5095.0, low=5093.0, close=5094.0, level_store=store)
        assert 5100.0 in bt._broken_supports
        # No deep flush — only 6pt below the shelf
        assert bt._broken_supports[5100.0]["deep_flush"] is False

        # Retest from below — should NOT engage BACKTEST_WATCH because no deep flush
        sig = bt.update(4, _ts(4), high=5099.0, low=5097.0, close=5098.0, level_store=store)
        assert sig is None
        assert bt.state == ShortState.IDLE

    def test_reclaim_aborts(self):
        """If price reclaims the broken shelf after the touch (Mancini's
        'level was drained and price will rip through and squeeze'),
        abort — no short signal."""
        params = self._params()
        bt = BacktestShort(params)
        store = self._shelf(price=5100.0)

        # Setup: breakdown + deep flush
        bt.update(0, _ts(0), high=5108.0, low=5102.0, close=5105.0, level_store=store)
        for i in range(1, 4):
            bt.update(i, _ts(i), high=5099.0 - i, low=5095.0 - i, close=5097.0 - i, level_store=store)
        for i in range(4, 8):
            bt.update(i, _ts(i), high=5089.0, low=5085.0, close=5087.0, level_store=store)
        assert bt._broken_supports[5100.0]["deep_flush"] is True

        # Touch — engages BACKTEST_WATCH
        bt.update(8, _ts(8), high=5099.0, low=5095.0, close=5097.0, level_store=store)
        assert bt.state == ShortState.BACKTEST_WATCH

        # Price reclaims — 3 consecutive closes above the shelf
        bt.update(9, _ts(9), high=5103.0, low=5099.0, close=5101.0, level_store=store)
        bt.update(10, _ts(10), high=5105.0, low=5101.0, close=5103.0, level_store=store)
        sig = bt.update(11, _ts(11), high=5106.0, low=5102.0, close=5104.0, level_store=store)
        assert sig is None
        assert bt.state == ShortState.IDLE

    def test_first_touch_only(self):
        """Mancini emphasizes that first retest is best — subsequent touches
        drop in odds. With bts_first_touch_only=True, second touch must NOT
        engage BACKTEST_WATCH."""
        params = self._params(bts_first_touch_only=True)
        bt = BacktestShort(params)
        store = self._shelf(price=5100.0)

        # Setup: breakdown + deep flush
        bt.update(0, _ts(0), high=5108.0, low=5102.0, close=5105.0, level_store=store)
        for i in range(1, 4):
            bt.update(i, _ts(i), high=5099.0 - i, low=5095.0 - i, close=5097.0 - i, level_store=store)
        for i in range(4, 8):
            bt.update(i, _ts(i), high=5089.0, low=5085.0, close=5087.0, level_store=store)

        # First touch — engages
        bt.update(8, _ts(8), high=5099.0, low=5095.0, close=5097.0, level_store=store)
        assert bt.state == ShortState.BACKTEST_WATCH
        assert bt._broken_supports[5100.0]["touch_count"] == 1

        # Price reclaims, then back below (resets state to IDLE)
        for i in range(9, 13):
            bt.update(i, _ts(i), high=5103.0, low=5099.0, close=5101.0, level_store=store)
        # After reclaim_abort, state is IDLE but broken_supports memory persists
        assert bt.state == ShortState.IDLE

        # Price dips back below and we attempt a SECOND retest
        for i in range(13, 17):
            bt.update(i, _ts(i), high=5095.0, low=5092.0, close=5094.0, level_store=store)
        bt.update(17, _ts(17), high=5099.0, low=5095.0, close=5097.0, level_store=store)
        # With first_touch_only, second touch is rejected — touch_count
        # increments but state remains IDLE
        assert bt._broken_supports[5100.0]["touch_count"] == 2
        assert bt.state == ShortState.IDLE

    def test_pdl_shelf_excluded(self):
        """PRIOR_DAY_LOW is intentionally excluded from _SHELF_TYPES — Phase 1
        block_pdl_shorts would reject any short there anyway, and we don't
        want this detector firing on the bot's primary long-side FB level."""
        params = self._params()
        bt = BacktestShort(params)
        store = _make_level_store([(5100.0, LevelType.PRIOR_DAY_LOW)])

        # Drive the full breakdown sequence — none of it should register
        for i in range(20):
            bt.update(i, _ts(i),
                      high=5099.0 if i > 0 else 5108.0,
                      low=5085.0, close=5090.0,
                      level_store=store)
        assert len(bt._broken_supports) == 0
        assert bt.state == ShortState.IDLE

    def test_insufficient_touches_excludes_cluster(self):
        """CLUSTER_LOW with too-few touches doesn't count as a 'clearly
        defined support' per Mancini. Should not register as a shelf."""
        params = self._params(bts_support_min_touches=5)
        bt = BacktestShort(params)
        # touch_count=2 < required 5 → ineligible
        store = _make_level_store([(5100.0, LevelType.CLUSTER_LOW)], touches=2)

        for i in range(8):
            bt.update(i, _ts(i),
                      high=5099.0 if i > 0 else 5108.0,
                      low=5085.0, close=5090.0,
                      level_store=store)
        assert len(bt._broken_supports) == 0

    def test_no_breakdown_if_price_never_above_level(self):
        """If price is already below the level and never trades above it,
        we never register a 'breakdown' — the level just exists below where
        we are. Lock in the close-vs-was_above guard."""
        params = self._params(bts_breakdown_confirm_bars=3)
        bt = BacktestShort(params)
        store = self._shelf(price=5100.0)

        # 10 bars all closing below 5100, never above
        for i in range(10):
            bt.update(i, _ts(i),
                      high=5095.0, low=5088.0, close=5090.0,
                      level_store=store)
        assert len(bt._broken_supports) == 0

    def test_expired_broken_support_forgotten(self):
        """After bts_breakout_expire_bars, the broken-support memory is
        cleared and a fresh setup is required."""
        params = self._params(bts_breakout_expire_bars=10)
        bt = BacktestShort(params)
        store = self._shelf(price=5100.0)

        # Setup breakdown
        bt.update(0, _ts(0), high=5108.0, low=5102.0, close=5105.0, level_store=store)
        for i in range(1, 4):
            bt.update(i, _ts(i), high=5099.0 - i, low=5095.0 - i, close=5097.0 - i, level_store=store)
        assert 5100.0 in bt._broken_supports

        # Jump way past the expiry window
        bt.update(20, _ts(20), high=5099.0, low=5095.0, close=5097.0, level_store=store)
        assert 5100.0 not in bt._broken_supports


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
