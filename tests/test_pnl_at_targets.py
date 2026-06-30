"""Tests for the P&L-at-planned-exits replay (the measurement tool).

replay_trade() reconstructs what a trade WOULD have made at its planned
75/15/10 scale-out by feeding the post-entry price path through the LIVE
ExitManager — so the number is faithful to how the bot really exits, not a
separate simulation. This is the instrument we use to judge whether the
short-gating change is actually better.
"""
from __future__ import annotations

import math

from strategy.exit_manager import ExitManager
from backtest.pnl_at_targets import replay_trade


def _em():
    return ExitManager()


class TestReplayTrade:
    def test_long_stop_out_returns_the_loss(self):
        r = replay_trade(
            entry=7000, stop=6990, target_1=7050, target_2=7100,
            contracts=20, direction="long",
            bars=[(7000, 6990, 6990)],  # first bar tags the stop
            exit_manager=_em(),
        )
        assert r["pnl_pts"] == 20 * (6990 - 7000)  # -200
        assert r["t1_hit"] is False

    def test_long_t1_then_eod_flatten(self):
        em = _em()
        t1_frac = em._t1_exit_fraction
        t1q = math.floor(20 * t1_frac)
        # hit T1 @ 7010, then drift flat above the post-T1 stop; runner flattens at EOD
        bars = [(7010, 7000, 7010), (7011, 7009, 7010), (7012, 7009, 7010)]
        r = replay_trade(
            entry=7000, stop=6980, target_1=7010, target_2=7060,
            contracts=20, direction="long", bars=bars, exit_manager=em,
        )
        expected = t1q * (7010 - 7000) + (20 - t1q) * (7010 - 7000)  # T1 tranche + runner@EOD
        assert r["pnl_pts"] == expected
        assert r["t1_hit"] is True
        assert r["runner_truncated_at_eod"] is True

    def test_short_stop_out_returns_the_loss(self):
        r = replay_trade(
            entry=7000, stop=7010, target_1=6950, target_2=6900,
            contracts=20, direction="short",
            bars=[(7010, 7000, 7010)],  # high tags the short stop
            exit_manager=_em(),
        )
        assert r["pnl_pts"] == 20 * (7000 - 7010)  # -200

    def test_no_bars_flattens_at_entry_zero_pnl(self):
        r = replay_trade(
            entry=7000, stop=6990, target_1=7050, target_2=7100,
            contracts=2, direction="long", bars=[], exit_manager=_em(),
        )
        assert r["pnl_pts"] == 0.0
