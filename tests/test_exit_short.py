"""Tests for short-side exit management."""

from strategy.exit_manager import ExitManager, TradePosition, ExitPhase
from config.settings import ExitParams


class TestShortExitManager:

    def _make_short_position(self) -> TradePosition:
        """Short entry at 6000, stop at 6004.5, T1 at 5990, T2 at 5980."""
        return TradePosition(
            entry_price=6000.0,
            stop_price=6004.5,
            target_1=5990.0,
            target_2=5980.0,
            total_contracts=4,
            remaining_contracts=4,
            direction="short",
        )

    def test_short_stop_loss(self):
        """Short stop hit when high >= stop."""
        params = ExitParams(t1_exit_fraction=1.0)
        mgr = ExitManager(params=params)
        pos = self._make_short_position()

        # High hits stop
        action = mgr.update(pos, high=6005.0, low=5998.0, close=6004.0)
        assert action is not None
        assert action.reason == "Stop loss hit"
        assert pos.remaining_contracts == 0
        # PnL should be negative (entry 6000 - stop 6004.5 = -4.5 per contract)
        assert pos.realized_pnl_pts < 0

    def test_short_t1_hit(self):
        """Short T1 hit when low <= target_1."""
        params = ExitParams(t1_exit_fraction=1.0)
        mgr = ExitManager(params=params)
        pos = self._make_short_position()

        # Low hits T1
        action = mgr.update(pos, high=5995.0, low=5989.0, close=5991.0)
        assert action is not None
        assert "Target 1" in action.reason
        assert pos.remaining_contracts == 0
        # PnL: (6000 - 5990) * 4 = +40
        assert pos.realized_pnl_pts == 40.0

    def test_short_trailing_ratchets_down(self):
        """Short trailing stop moves down as price drops."""
        params = ExitParams(
            t1_exit_fraction=0.75,
            trailing_stop_pts=7.0,
        )
        mgr = ExitManager(params=params)
        pos = self._make_short_position()

        # Hit T1
        mgr.update(pos, high=5995.0, low=5989.0, close=5991.0)
        assert pos.phase == ExitPhase.AFTER_T1
        initial_stop = pos.stop_price

        # Price drops further — trail should ratchet down
        mgr.update(pos, high=5985.0, low=5978.0, close=5980.0)
        assert pos.stop_price <= initial_stop

    def test_short_no_action_above_targets(self):
        """No exit when price stays between entry and stop."""
        params = ExitParams(t1_exit_fraction=1.0)
        mgr = ExitManager(params=params)
        pos = self._make_short_position()

        # Price dips but doesn't reach T1
        action = mgr.update(pos, high=6002.0, low=5995.0, close=5997.0)
        assert action is None
        assert pos.is_open

    def test_short_pnl_sign(self):
        """Verify short PnL is positive when price drops, negative when price rises."""
        params = ExitParams(t1_exit_fraction=1.0)
        mgr = ExitManager(params=params)

        # Winning short: price drops to target
        pos_win = self._make_short_position()
        mgr.update(pos_win, high=5995.0, low=5989.0, close=5991.0)
        assert pos_win.realized_pnl_pts > 0

        # Losing short: price rises to stop
        pos_lose = self._make_short_position()
        mgr.update(pos_lose, high=6005.0, low=5998.0, close=6004.0)
        assert pos_lose.realized_pnl_pts < 0
