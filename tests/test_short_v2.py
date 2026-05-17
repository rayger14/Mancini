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
    """Tests for the Mancini-faithful BreakdownShort.

    Per Mancini's 2025-08-24 post, BD short fires on the FAILURE of a
    Failed Breakdown long attempt:
      1. Obvious support shelf
      2. Price loses the shelf, recovers it, rallies (FB attempt)
      3. Price comes back down and breaks the FB's lowest low → short
         trigger a few pts below.
    """

    def _params(self, **overrides):
        defaults = dict(
            allow_breakdown_short=True,
            bd_shelf_min_touches=3,
            bd_shelf_expire_bars=480,
            bd_min_flush_depth_pts=3.0,
            bd_max_flush_bars=30,
            bd_fb_fail_buffer_pts=3.0,
            bd_fb_success_rally_pts=20.0,
            bd_fb_success_timeout_bars=60,
            bd_recovery_watch_bars=120,
            bd_stop_buffer_pts=3.0,
        )
        defaults.update(overrides)
        return StrategyParams(**defaults)

    def _shelf_store(self, price=5000.0, level_type=LevelType.PRIOR_DAY_LOW,
                     touches=5):
        return _make_level_store([(price, level_type)], touches=touches)

    def test_failed_fb_emits_signal(self):
        """Happy path: shelf → flush → recovery → re-break of flush_low →
        BD short signal, with stop above the recovery_high."""
        bd = BreakdownShort(self._params())
        store = self._shelf_store(price=5000.0)

        # Bar 0: price above shelf — shelf is discovered, state = watching
        bd.update(0, _ts(0), high=5008.0, low=5002.0, close=5005.0, level_store=store)
        assert 5000.0 in bd._shelves
        assert bd._shelves[5000.0]["state"] == "watching"

        # Bar 1: close below shelf — flush begins
        bd.update(1, _ts(1), high=5001.0, low=4995.0, close=4996.0, level_store=store)
        assert bd._shelves[5000.0]["state"] == "flush"

        # Bars 2-3: deeper flush
        bd.update(2, _ts(2), high=4997.0, low=4990.0, close=4991.0, level_store=store)
        bd.update(3, _ts(3), high=4993.0, low=4988.0, close=4989.0, level_store=store)
        assert bd._shelves[5000.0]["flush_low"] == 4988.0

        # Bar 4: recovery — close back above shelf
        bd.update(4, _ts(4), high=5002.0, low=4990.0, close=5001.0, level_store=store)
        assert bd._shelves[5000.0]["state"] == "recovered"
        assert bd._shelves[5000.0]["recovery_high"] == 5002.0

        # Bar 5: rally a bit higher (the squeeze after FB)
        bd.update(5, _ts(5), high=5010.0, low=5000.0, close=5008.0, level_store=store)
        assert bd._shelves[5000.0]["recovery_high"] == 5010.0

        # Bar 6: NO signal yet — price stays above flush_low - buffer
        sig = bd.update(6, _ts(6), high=5009.0, low=4990.0, close=4992.0, level_store=store)
        assert sig is None

        # Bar 7: close breaks flush_low - buffer (4988 - 3 = 4985)
        sig = bd.update(7, _ts(7), high=4990.0, low=4980.0, close=4984.0, level_store=store)
        assert sig is not None
        assert sig.pattern_type == "breakdown_short"
        assert sig.direction == "short"
        # Stop is recovery_high (5010) + bd_stop_buffer_pts (3)
        assert sig.stop_price == 5013.0
        # Synthetic level carries the broken flush_low, NOT the shelf —
        # so Phase 1's PDL block doesn't apply even though the underlying
        # shelf is PRIOR_DAY_LOW
        assert sig.level.price == 4988.0
        assert sig.level.level_type == LevelType.INTRADAY_LOW

    def test_no_signal_if_fb_succeeds(self):
        """If the FB rally extends past bd_fb_success_rally_pts AND time
        passes without a failure, abandon the shelf — Mancini's FB long
        worked, no BD short available."""
        bd = BreakdownShort(self._params(
            bd_fb_success_rally_pts=10.0,
            bd_fb_success_timeout_bars=5,
        ))
        store = self._shelf_store(price=5000.0)

        # Setup: above, flush, recovery
        bd.update(0, _ts(0), high=5008.0, low=5002.0, close=5005.0, level_store=store)
        bd.update(1, _ts(1), high=5001.0, low=4995.0, close=4996.0, level_store=store)
        bd.update(2, _ts(2), high=4997.0, low=4988.0, close=4990.0, level_store=store)
        bd.update(3, _ts(3), high=5002.0, low=4990.0, close=5001.0, level_store=store)
        assert bd._shelves[5000.0]["state"] == "recovered"

        # Rally well above shelf for 6 bars — FB succeeded
        for i in range(4, 10):
            bd.update(i, _ts(i), high=5015.0 + i, low=5008.0,
                      close=5012.0 + i, level_store=store)
        assert bd._shelves[5000.0]["state"] == "abandoned"

    def test_shallow_tag_resets_to_watching(self):
        """If price dips below shelf by less than bd_min_flush_depth_pts
        and recovers immediately, it's a tag, not an FB attempt. Shelf
        should reset to watching."""
        bd = BreakdownShort(self._params(bd_min_flush_depth_pts=10.0))
        store = self._shelf_store(price=5000.0)

        bd.update(0, _ts(0), high=5008.0, low=5002.0, close=5005.0, level_store=store)
        # Shallow tag — only 2pt below shelf
        bd.update(1, _ts(1), high=5001.0, low=4998.0, close=4999.0, level_store=store)
        assert bd._shelves[5000.0]["state"] == "flush"
        # Immediate recovery — flush_depth only 2pt < 10pt min
        bd.update(2, _ts(2), high=5003.0, low=4999.0, close=5002.0, level_store=store)
        assert bd._shelves[5000.0]["state"] == "watching"

    def test_long_flush_without_recovery_abandons(self):
        """If price stays below shelf for bd_max_flush_bars without
        recovering, that's a trend leg (not an FB failure setup)."""
        bd = BreakdownShort(self._params(bd_max_flush_bars=5))
        store = self._shelf_store(price=5000.0)

        bd.update(0, _ts(0), high=5008.0, low=5002.0, close=5005.0, level_store=store)
        # Flush begins
        bd.update(1, _ts(1), high=5001.0, low=4995.0, close=4996.0, level_store=store)
        # Stay below for 7 more bars without recovery
        for i in range(2, 9):
            bd.update(i, _ts(i), high=4998.0, low=4990.0 - i, close=4992.0, level_store=store)
        assert bd._shelves[5000.0]["state"] == "abandoned"

    def test_recovery_watch_timeout_abandons(self):
        """After recovery, if neither failure nor success criteria fire
        within bd_recovery_watch_bars, abandon."""
        bd = BreakdownShort(self._params(
            bd_recovery_watch_bars=5,
            bd_fb_success_rally_pts=999,  # never satisfied
        ))
        store = self._shelf_store(price=5000.0)

        bd.update(0, _ts(0), high=5008.0, low=5002.0, close=5005.0, level_store=store)
        bd.update(1, _ts(1), high=5001.0, low=4995.0, close=4996.0, level_store=store)
        bd.update(2, _ts(2), high=4997.0, low=4988.0, close=4990.0, level_store=store)
        bd.update(3, _ts(3), high=5002.0, low=4990.0, close=5001.0, level_store=store)
        assert bd._shelves[5000.0]["state"] == "recovered"

        # 7 bars pass with price oscillating but never failing/succeeding
        for i in range(4, 11):
            bd.update(i, _ts(i), high=5002.0, low=4990.0, close=4995.0, level_store=store)
        assert bd._shelves[5000.0]["state"] == "abandoned"

    def test_swing_low_not_a_shelf(self):
        """SWING_LOW is not in _SHELF_TYPES — should not be tracked."""
        bd = BreakdownShort(self._params())
        store = _make_level_store([(5000.0, LevelType.SWING_LOW)])
        bd.update(0, _ts(0), high=5008.0, low=5002.0, close=5005.0, level_store=store)
        assert len(bd._shelves) == 0

    def test_cluster_with_few_touches_excluded(self):
        """CLUSTER_LOW with touch_count < bd_shelf_min_touches is filtered."""
        bd = BreakdownShort(self._params(bd_shelf_min_touches=5))
        store = _make_level_store([(5000.0, LevelType.CLUSTER_LOW)], touches=2)
        bd.update(0, _ts(0), high=5008.0, low=5002.0, close=5005.0, level_store=store)
        assert len(bd._shelves) == 0

    def test_pdl_shelf_tracked_but_signal_uses_synthetic_level(self):
        """PRIOR_DAY_LOW IS a valid shelf for BD — Mancini explicitly cites
        it ('Monday daily low of 6456' in the 2025-08-24 example). The
        Phase 1 block_pdl_shorts gate would block any short whose
        pattern.level is PDL, so the emitted signal must carry the
        synthetic INTRADAY_LOW representing the broken flush_low."""
        bd = BreakdownShort(self._params())
        store = self._shelf_store(price=5000.0, level_type=LevelType.PRIOR_DAY_LOW)

        # Full sequence to emit signal
        bd.update(0, _ts(0), high=5008.0, low=5002.0, close=5005.0, level_store=store)
        bd.update(1, _ts(1), high=5001.0, low=4995.0, close=4996.0, level_store=store)
        bd.update(2, _ts(2), high=4997.0, low=4988.0, close=4990.0, level_store=store)
        bd.update(3, _ts(3), high=5002.0, low=4990.0, close=5001.0, level_store=store)
        bd.update(4, _ts(4), high=5010.0, low=5000.0, close=5008.0, level_store=store)
        sig = bd.update(5, _ts(5), high=4990.0, low=4980.0, close=4984.0, level_store=store)

        assert sig is not None
        # Critical: pattern.level is the synthetic flush_low, NOT the PDL shelf
        assert sig.level.level_type == LevelType.INTRADAY_LOW
        assert sig.level.price == 4988.0
        # Phase 1 would have blocked if level_type were PRIOR_DAY_LOW
        assert sig.level.level_type != LevelType.PRIOR_DAY_LOW


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
