"""Tests for broker-confirmed exits and exit-record integrity.

Trade #16872 (2026-06-08): the bot logged "Stop loss hit, FLATTEN closed
4 MES" at 22:38, but IB still held the long all night — the flatten market
order died silently (GTC market orders are rejected on some IB routes) and
flatten() never verified the position actually went flat. The bracket TP
filled at 7451 (+101 pts) at 06:01 and was double-logged as a second -37
exit because close_position() returned None (the manager had already
closed the trade) while the runner unconditionally re-logged
session.trades[-1].

Three behaviors under test:
1. ib_bridge.flatten() confirms the position is flat (and uses a DAY
   market order), returning False otherwise.
2. _handle_exit_action() on flatten failure reverts the ExitManager's
   position mutation and returns False, so the position stays live and
   the exit retries next bar.
3. _sync_position() only logs the exit record that close_position()
   actually created — never trades[-1] as a fallback.
"""
from __future__ import annotations

import time as _time_mod
from datetime import datetime
from types import SimpleNamespace

import pytest

from live.ib_bridge import IBBridge
from live.ib_runner import IBRunner
from strategy.exit_manager import ExitAction, ExitPhase, TradePosition


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _ib_position(qty: int):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol="MES", secType="FUT"),
        position=qty,
    )


class _FakeIB:
    """Scriptable stand-in for the ib_insync IB client."""

    def __init__(self, position_script):
        # Each positions() call pops the next snapshot; last repeats forever.
        self._script = list(position_script)
        self.placed_orders = []
        self.cancelled = []

    def positions(self):
        if len(self._script) > 1:
            return self._script.pop(0)
        return self._script[0]

    def openOrders(self):
        return []

    def cancelOrder(self, order):
        self.cancelled.append(order)

    def placeOrder(self, contract, order):
        self.placed_orders.append(order)
        return SimpleNamespace(orderStatus=SimpleNamespace(status="Submitted"))

    def sleep(self, seconds):
        pass


def _fake_bridge(ib):
    return SimpleNamespace(
        is_connected=True,
        _contract=SimpleNamespace(symbol="MES"),
        config=SimpleNamespace(symbol="MES"),
        _ib=ib,
        _active_orders={},
    )


def _stopped_out_position() -> tuple[TradePosition, ExitAction]:
    """A position exactly as ExitManager._stop_out leaves it."""
    pos = TradePosition(
        entry_price=7425.75,
        stop_price=7416.5,
        target_1=7451.0,
        target_2=7472.0,
        total_contracts=4,
        remaining_contracts=4,
    )
    # simulate _stop_out mutation
    pos.realized_pnl_pts += (pos.stop_price - pos.entry_price) * 4  # -37
    pos.remaining_contracts = 0
    pos.phase = ExitPhase.CLOSED
    action = ExitAction(
        contracts_to_close=4,
        exit_price=7416.5,
        new_stop=0.0,
        new_phase=ExitPhase.CLOSED,
        reason="Stop loss hit",
    )
    return pos, action


# ---------------------------------------------------------------------------
# 1. flatten() verification
# ---------------------------------------------------------------------------


class TestFlattenVerification:
    def test_returns_false_when_position_survives(self):
        ib = _FakeIB([[_ib_position(4)]])  # position never goes away
        ok = IBBridge.flatten(_fake_bridge(ib), reason="Stop loss hit")
        assert ok is False
        assert len(ib.placed_orders) >= 1, "close order must still be attempted"

    def test_returns_true_when_position_goes_flat(self):
        ib = _FakeIB([[_ib_position(4)], []])  # flat on first verify check
        ok = IBBridge.flatten(_fake_bridge(ib), reason="Stop loss hit")
        assert ok is True

    def test_market_order_uses_day_tif(self):
        ib = _FakeIB([[_ib_position(4)], []])
        IBBridge.flatten(_fake_bridge(ib), reason="Stop loss hit")
        assert ib.placed_orders[0].tif == "DAY", (
            "GTC market orders are rejected on some IB routes — "
            "the silent rejection is how #16872 survived its flatten"
        )


# ---------------------------------------------------------------------------
# 2. _handle_exit_action failure path
# ---------------------------------------------------------------------------


class TestRevertPositionClose:
    def test_revert_restores_open_state(self):
        pos, action = _stopped_out_position()
        IBRunner._revert_position_close(SimpleNamespace(), pos, action)
        assert pos.remaining_contracts == 4
        assert pos.phase == ExitPhase.INITIAL
        assert pos.realized_pnl_pts == pytest.approx(0.0)
        assert pos.is_open

    def test_revert_infers_after_t1_phase(self):
        pos, action = _stopped_out_position()
        pos.t1_hit = True
        IBRunner._revert_position_close(SimpleNamespace(), pos, action)
        assert pos.phase == ExitPhase.AFTER_T1


class TestHandleExitActionFlattenFailure:
    def _runner(self, flatten_ok: bool, pos: TradePosition):
        embeds = []
        runner = SimpleNamespace(
            _trade_id=16872,
            _position=pos,
            bridge=SimpleNamespace(flatten=lambda reason="": flatten_ok),
            _post_trade_exit_embed=lambda **kw: embeds.append(kw),
            _revert_position_close=lambda p, a: IBRunner._revert_position_close(
                None, p, a
            ),
        )
        return runner, embeds

    def test_failure_returns_false_reverts_and_skips_embed(self):
        pos, action = _stopped_out_position()
        runner, embeds = self._runner(flatten_ok=False, pos=pos)
        result = IBRunner._handle_exit_action(runner, action, datetime(2026, 6, 8, 22, 38))
        assert result is False
        assert pos.is_open, "position must stay live for retry next bar"
        assert embeds == [], "no exit embed for an unconfirmed exit"

    def test_success_proceeds_and_posts_embed(self):
        pos, action = _stopped_out_position()
        runner, embeds = self._runner(flatten_ok=True, pos=pos)
        result = IBRunner._handle_exit_action(runner, action, datetime(2026, 6, 8, 22, 38))
        assert result is not False
        assert not pos.is_open
        assert len(embeds) == 1


# ---------------------------------------------------------------------------
# 3. _sync_position duplicate-exit guard
# ---------------------------------------------------------------------------


class TestSyncPositionDisconnectGuard:
    """Trade #25196 (2026-06-09 19:47): the IB connection was dead (5 failed
    reconnects), get_position() returned None for "no connection", and the
    3x-None confirmation booked a fictional -12.5 exit while the bracket was
    still working on IB's servers. Sync must not interpret anything while
    the bridge is disconnected."""

    def test_no_close_confirmation_while_disconnected(self):
        pos = TradePosition(
            entry_price=7375.5,
            stop_price=7357.5,
            target_1=7390.0,
            target_2=7400.0,
            total_contracts=2,
            remaining_contracts=2,
        )
        logged = []
        runner = SimpleNamespace(
            _position=pos,
            _trade_id=25196,
            _last_entry_monotonic=_time_mod.monotonic() - 300.0,
            _mono_fn=_time_mod.monotonic,  # IBRunner clock seam (ReplayRunner)
            _now_fn=__import__("live.ib_runner", fromlist=["IBRunner"]).IBRunner._now_fn,
            _sync_none_count=2,  # one more None would have confirmed closure
            bridge=SimpleNamespace(
                is_connected=False,
                get_position=lambda: None,
                get_bracket_fill_price=lambda tid: (0.0, "unknown"),
            ),
            _post_trade_exit_embed=lambda **kw: None,
            position_manager=SimpleNamespace(
                close_position=lambda **kw: None,
                session=SimpleNamespace(trades=[]),
            ),
            _log_trade=lambda rec, sig, ev: logged.append(rec),
        )
        IBRunner._sync_position(runner)
        assert runner._position is pos, "position must survive a blind sync"
        assert pos.is_open
        assert logged == []
        assert runner._sync_none_count == 0, (
            "stale None-counts from before the disconnect must not carry "
            "over and instantly confirm closure on reconnect"
        )


class TestSyncPositionDupGuard:
    def _runner(self, close_returns):
        pos = TradePosition(
            entry_price=7425.75,
            stop_price=7416.5,
            target_1=7451.0,
            target_2=7472.0,
            total_contracts=4,
            remaining_contracts=4,
        )
        logged = []
        old_trade = object()  # sentinel: the PREVIOUS session trade
        runner = SimpleNamespace(
            _position=pos,
            _trade_id=0,
            _last_entry_monotonic=_time_mod.monotonic() - 300.0,
            _mono_fn=_time_mod.monotonic,  # IBRunner clock seam (ReplayRunner)
            _now_fn=__import__("live.ib_runner", fromlist=["IBRunner"]).IBRunner._now_fn,
            _sync_none_count=2,  # this call is the 3rd None → confirm
            _last_price=7450.75,
            _pattern_type="failed_breakdown",
            _entry_timestamp=datetime(2026, 6, 8, 22, 0),
            _current_signal=None,
            bridge=SimpleNamespace(
                is_connected=True,
                get_position=lambda: None,
                get_bracket_fill_price=lambda tid: (7451.0, "TP"),
                seconds_since_reconnect=lambda: 1e9,  # long past any reconnect
                get_bracket_orders=lambda: {},        # OCA resolved → truly flat
            ),
            _post_trade_exit_embed=lambda **kw: None,
            position_manager=SimpleNamespace(
                close_position=lambda **kw: close_returns,
                session=SimpleNamespace(trades=[old_trade]),
            ),
            _log_trade=lambda rec, sig, ev: logged.append(rec),
        )
        return runner, logged, old_trade

    def test_no_log_when_close_position_returns_none(self):
        runner, logged, old_trade = self._runner(close_returns=None)
        IBRunner._sync_position(runner)
        assert logged == [], (
            "close_position() returned None (nothing recorded) — re-logging "
            "trades[-1] duplicates the previous trade's exit (#16872 dup)"
        )
        assert runner._position is None

    def test_logs_the_record_close_position_returned(self):
        record = object()
        runner, logged, old_trade = self._runner(close_returns=record)
        IBRunner._sync_position(runner)
        assert logged == [record]


# ---------------------------------------------------------------------------
# 4. _sync_position phantom-close guard (reconnect / bracket-still-live)
# ---------------------------------------------------------------------------


class TestPhantomCloseGuard:
    """Pure decision: when get_position() returns None, is it a real closure
    or a post-reconnect / cache-lag desync that would book a fictional exit?"""

    def test_within_reconnect_grace_ignores(self):
        from live.ib_runner import phantom_close_guard
        assert phantom_close_guard(8.0, bracket_live=False) == "ignore_reconnect"

    def test_bracket_still_live_ignores(self):
        from live.ib_runner import phantom_close_guard
        assert phantom_close_guard(1e9, bracket_live=True) == "ignore_bracket_live"

    def test_past_grace_no_bracket_counts(self):
        from live.ib_runner import phantom_close_guard
        assert phantom_close_guard(1e9, bracket_live=False) == "count"


class TestSyncPositionPhantomGuard:
    """Trades #567 and #579 (2026-06-28/29): get_position() returned None right
    after a reconnect / shortly after entry while the bracket was still live on
    IB, and the 3x-None confirmation booked fictional exits at estimated prices
    before T1. Sync must not book a close in either window."""

    def _runner(self, *, secs_since_reconnect, bracket_orders):
        pos = TradePosition(
            entry_price=7453.5, stop_price=7438.5,
            target_1=7486.0, target_2=7500.0,
            total_contracts=2, remaining_contracts=2,
        )
        logged = []
        runner = SimpleNamespace(
            _position=pos,
            _trade_id=579,
            _last_entry_monotonic=_time_mod.monotonic() - 300.0,
            _mono_fn=_time_mod.monotonic,  # IBRunner clock seam (ReplayRunner)
            _now_fn=__import__("live.ib_runner", fromlist=["IBRunner"]).IBRunner._now_fn,  # past 45s grace
            _sync_none_count=2,  # the next None would confirm closure
            _last_price=7456.0,
            _pattern_type="failed_breakdown",
            _entry_timestamp=datetime(2026, 6, 29, 10, 31),
            _current_signal=None,
            bridge=SimpleNamespace(
                is_connected=True,
                get_position=lambda: None,
                seconds_since_reconnect=lambda: secs_since_reconnect,
                get_bracket_orders=lambda: bracket_orders,
                get_bracket_fill_price=lambda tid: (0.0, "unknown"),
            ),
            _post_trade_exit_embed=lambda **kw: None,
            position_manager=SimpleNamespace(
                close_position=lambda **kw: object(),
                session=SimpleNamespace(trades=[]),
            ),
            _log_trade=lambda rec, sig, ev: logged.append(rec),
        )
        return runner, logged

    def test_no_close_right_after_reconnect(self):
        # #567: reconnected 8s ago, position cache not yet repushed
        runner, logged = self._runner(secs_since_reconnect=8.0, bracket_orders={})
        IBRunner._sync_position(runner)
        assert runner._position is not None and runner._position.is_open
        assert logged == []
        assert runner._sync_none_count == 0

    def test_no_close_while_bracket_still_live(self):
        # #579: 53s after entry, but SL/TP still working on IB → not closed
        runner, logged = self._runner(
            secs_since_reconnect=1e9, bracket_orders={"sl": 7438.5, "tp": 7486.0})
        IBRunner._sync_position(runner)
        assert runner._position is not None and runner._position.is_open
        assert logged == []
        assert runner._sync_none_count == 0

    def test_genuine_close_still_books(self):
        # control: past grace AND bracket gone (OCA resolved) → real closure
        runner, logged = self._runner(secs_since_reconnect=1e9, bracket_orders={})
        IBRunner._sync_position(runner)
        assert runner._position is None
        assert len(logged) == 1
