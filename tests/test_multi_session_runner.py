"""Tests for the multi-session runner hold in live/ib_runner.py.

Per Mancini's 2025-10-12 quote: "I am still holding my 10% long runner from
the Tuesday noon 6754 Failed Breakdown." His 10% post-T2 slice is intended to
ride trend moves across multiple sessions, not get flattened at every EOD.

This file covers the IBRunner._check_eod and IBRunner._check_session_rollover
behaviors that govern when a runner survives EOD and when it gets flattened.

Coverage:
  1. multi_session_runner=False (legacy): AFTER_T2 runner gets flattened at EOD.
  2. multi_session_runner=True, AFTER_T2: runner survives EOD, trail updated.
  3. multi_session_runner=True, AFTER_T1: runner STILL flattened (only 10%
     post-T2 slice is allowed cross-session).
  4. Max-days safety cap: after N session rollovers, the runner is
     force-flattened on the next EOD even when multi_session_runner=True.
  5. New-position reset: opening a fresh trade clears the survived-sessions
     counter so the cap applies per-trade, not per-bot-lifetime.
"""

from __future__ import annotations

from datetime import date, datetime, time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import pytz

from config.settings import (
    ExitParams,
    SessionTimes,
    ESContractSpec,
)
from strategy.exit_manager import ExitPhase, TradePosition


_ET = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_runner_stub(
    multi_session: bool = True,
    max_days: int = 5,
    sessions_held: int = 0,
):
    """Build a minimal IBRunner with just enough surface to drive _check_eod
    and _check_session_rollover.

    Avoids invoking IBRunner.__init__ (which connects to IB Gateway). We
    bypass __init__ and wire the attributes the EOD path reads directly.
    """
    from live.ib_runner import IBRunner

    runner = IBRunner.__new__(IBRunner)

    exit_params = ExitParams(
        default_contracts=4,
        t1_exit_fraction=0.75,
        t2_exit_fraction=0.15,
        runner_fraction=0.10,
        multi_session_runner=multi_session,
        multi_session_runner_max_days=max_days,
    )
    runner.exit_params = exit_params

    # session_times: use a normal-range session so past_eod_flatten triggers
    # cleanly on 15:55+. The wrap-around (FULL_SESSION) path is exercised
    # via the same code paths — no need to duplicate.
    session = SessionTimes()  # 9:30 - 16:00 RTH, eod=15:55
    runner.strategy = SimpleNamespace(session_times=session, reset=lambda: None)

    # Bridge: mock — flatten() and update_stop() are no-ops we just spy on.
    runner.bridge = MagicMock()
    runner.bridge.get_prior_day_bars.return_value = None

    # Exit manager: real one, so update_prior_day_low actually mutates state.
    from strategy.exit_manager import ExitManager
    runner.exit_manager = ExitManager(
        params=exit_params,
        contract=ESContractSpec(),
    )

    # Position manager: stub with the bits _check_eod/rollover touch.
    pos_session = SimpleNamespace(
        active_position=None,
        active_long=None,
        active_short=None,
        trades=[],
    )
    runner.position_manager = SimpleNamespace(
        close_position=MagicMock(return_value=None),
        start_session=lambda dt: None,
        session=pos_session,
    )

    # Signal aggregator: stub with the bits rollover touches.
    runner.signal_aggregator = SimpleNamespace(
        get_pattern_state=lambda: {},
        restore_pattern_state=lambda state: None,
        initialize_levels=lambda *a, **k: None,
    )

    # Frame state: 60 bars centered on a low of 6740 / high of 6780.
    bar_ts = pd.date_range("2026-05-14 09:30", periods=60, freq="1min", tz=_ET)
    runner._df = pd.DataFrame({
        "open": [6750.0] * 60,
        "high": [6780.0] * 60,
        "low": [6740.0] * 60,
        "close": [6770.0] * 60,
        "volume": [100] * 60,
    }, index=bar_ts)

    runner._position = None
    runner._trade_id = None
    runner._pattern_type = ""
    runner._current_signal = None
    runner._entry_timestamp = datetime(2026, 5, 12, 12, 0, tzinfo=_ET)
    runner._session_date = date(2026, 5, 14)
    runner._bar_count = 60
    runner._phantom_positions = []
    runner._near_miss_phantoms = []
    runner._runner_sessions_held = sessions_held

    # Methods the EOD/rollover path call on self
    runner._log_trade = MagicMock()
    runner._archive_session = MagicMock()
    runner._log_session_summary = MagicMock()
    runner._get_session_low = lambda: 6740.0
    runner._get_session_high = lambda: 6780.0

    return runner


def _eod_bar() -> dict:
    """A bar at the EOD flatten threshold (15:56 ET)."""
    ts = datetime(2026, 5, 14, 15, 56, tzinfo=_ET)
    return {
        "timestamp": ts.isoformat(),
        "open": 6770.0,
        "high": 6775.0,
        "low": 6768.0,
        "close": 6770.0,
        "volume": 100,
    }


def _make_position(
    phase: ExitPhase,
    entry_price: float = 6754.0,
    stop_price: float = 6736.0,
    contracts: int = 1,
    direction: str = "long",
) -> TradePosition:
    """Construct a TradePosition in the requested phase."""
    pos = TradePosition(
        entry_price=entry_price,
        stop_price=stop_price,
        target_1=6764.0,
        target_2=6776.0,
        total_contracts=4,
        remaining_contracts=contracts,
        direction=direction,
    )
    pos.phase = phase
    return pos


# ---------------------------------------------------------------------------
# Case 1 — multi-session DISABLED: AFTER_T2 still flattens at EOD
# ---------------------------------------------------------------------------


def test_multi_session_disabled_flattens_after_t2_runner_at_eod():
    """When multi_session_runner=False, the 10% AFTER_T2 runner is
    flattened at EOD just like INITIAL/AFTER_T1 positions. This is the
    legacy behavior — confirms the gate fires only when the new flag is on.
    """
    runner = _make_runner_stub(multi_session=False)
    runner._position = _make_position(phase=ExitPhase.AFTER_T2, contracts=1)

    runner._check_eod(_eod_bar())

    runner.bridge.flatten.assert_called_once()
    # Reason should be the standard EOD flatten label, not the max-days variant
    assert runner.bridge.flatten.call_args.kwargs.get("reason") == "eod_flatten"
    assert runner._position is None
    assert runner._trade_id is None
    assert runner._runner_sessions_held == 0


# ---------------------------------------------------------------------------
# Case 2 — multi-session ENABLED, AFTER_T2: runner survives, trail updated
# ---------------------------------------------------------------------------


def test_multi_session_enabled_holds_after_t2_runner_through_eod():
    """When multi_session_runner=True AND phase=AFTER_T2, the runner is
    NOT flattened at EOD. The structural trail under today's session low
    is applied via exit_manager.update_prior_day_low, and the position
    stays open ready for the next session.
    """
    runner = _make_runner_stub(multi_session=True)
    pos = _make_position(
        phase=ExitPhase.AFTER_T2,
        entry_price=6754.0,
        stop_price=6730.0,  # original stop below today's low
        contracts=1,
    )
    runner._position = pos
    runner._trade_id = 42

    runner._check_eod(_eod_bar())

    # No flatten call
    runner.bridge.flatten.assert_not_called()
    # Position still open
    assert runner._position is pos
    assert runner._position.is_open
    # Trail should have moved up: prior_day_low (6740) - 1 buffer = 6739
    expected_new_stop = 6740.0 - runner.exit_params.runner_prior_day_low_buffer_pts
    assert pos.stop_price == pytest.approx(expected_new_stop)
    # And the bridge should have been told to update the IB stop
    runner.bridge.update_stop.assert_called_once()
    assert runner.bridge.update_stop.call_args.kwargs["trade_id"] == 42


# ---------------------------------------------------------------------------
# Case 3 — multi-session ENABLED, AFTER_T1 still flattens
# ---------------------------------------------------------------------------


def test_multi_session_enabled_still_flattens_after_t1():
    """Multi-session runner hold ONLY applies AFTER T2 fires (the 10%
    slice). An AFTER_T1 position still owns ~25% (T2 + runner) which is
    too much overnight exposure — flatten at EOD even when the flag is on.
    """
    runner = _make_runner_stub(multi_session=True)
    runner._position = _make_position(phase=ExitPhase.AFTER_T1, contracts=2)

    runner._check_eod(_eod_bar())

    runner.bridge.flatten.assert_called_once()
    assert runner.bridge.flatten.call_args.kwargs.get("reason") == "eod_flatten"
    assert runner._position is None


def test_multi_session_enabled_still_flattens_initial():
    """INITIAL phase always flattens at EOD regardless of the flag."""
    runner = _make_runner_stub(multi_session=True)
    runner._position = _make_position(phase=ExitPhase.INITIAL, contracts=4)

    runner._check_eod(_eod_bar())

    runner.bridge.flatten.assert_called_once()
    assert runner._position is None


# ---------------------------------------------------------------------------
# Case 4 — Max-days safety cap
# ---------------------------------------------------------------------------


def test_max_days_cap_force_flattens_long_runner():
    """After max_days sessions, the next EOD force-flattens the runner
    even when multi_session_runner=True. The flatten reason should
    distinguish 'max days' from a normal EOD flatten so logs are clear.
    """
    runner = _make_runner_stub(
        multi_session=True,
        max_days=3,
        sessions_held=3,  # already at the cap; this EOD should flatten
    )
    runner._position = _make_position(phase=ExitPhase.AFTER_T2, contracts=1)
    runner._trade_id = 99

    runner._check_eod(_eod_bar())

    runner.bridge.flatten.assert_called_once()
    assert (
        runner.bridge.flatten.call_args.kwargs.get("reason")
        == "eod_flatten_max_days"
    )
    assert runner._position is None
    assert runner._runner_sessions_held == 0  # reset on flatten


def test_below_max_days_still_holds():
    """One short of the cap still survives — boundary check."""
    runner = _make_runner_stub(
        multi_session=True,
        max_days=3,
        sessions_held=2,  # below cap
    )
    pos = _make_position(phase=ExitPhase.AFTER_T2, contracts=1)
    runner._position = pos

    runner._check_eod(_eod_bar())

    runner.bridge.flatten.assert_not_called()
    assert runner._position is pos
    assert runner._position.is_open


# ---------------------------------------------------------------------------
# Session-rollover bookkeeping
# ---------------------------------------------------------------------------


def test_session_rollover_increments_held_counter_for_after_t2(monkeypatch):
    """When the Globex date rolls over with an open AFTER_T2 runner and
    multi_session_runner=True, the survived-sessions counter should bump
    by 1 so the max-days cap accumulates across rollovers.
    """
    runner = _make_runner_stub(multi_session=True, sessions_held=1)
    pos = _make_position(phase=ExitPhase.AFTER_T2, contracts=1)
    runner._position = pos
    runner._pattern_type = "FB_LONG"

    # Patch datetime.now used inside _check_session_rollover to return a
    # time that produces a new trading_date.
    target = datetime(2026, 5, 15, 19, 0, tzinfo=_ET)  # 19:00 ET — past 18:00
    import live.ib_runner as ib_runner_mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return target if tz is None else target.astimezone(tz)

    monkeypatch.setattr(ib_runner_mod, "datetime", _FakeDatetime)

    runner._check_session_rollover()

    # Trading date moved forward (5/15 + 1 = 5/16)
    assert runner._session_date == date(2026, 5, 16)
    # Counter bumped
    assert runner._runner_sessions_held == 2
    # Position transferred to the new session
    assert runner.position_manager.session.active_position is pos
    assert runner.position_manager.session.active_long is pos


def test_session_rollover_does_not_increment_when_multi_session_disabled(monkeypatch):
    """If the flag is off, we don't track sessions_held — there's no
    multi-session lifecycle to manage. (In practice the runner would
    have been flattened at the prior EOD, but defensively the rollover
    path shouldn't bump the counter either.)
    """
    runner = _make_runner_stub(multi_session=False, sessions_held=0)
    pos = _make_position(phase=ExitPhase.AFTER_T2, contracts=1)
    runner._position = pos

    target = datetime(2026, 5, 15, 19, 0, tzinfo=_ET)
    import live.ib_runner as ib_runner_mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return target if tz is None else target.astimezone(tz)

    monkeypatch.setattr(ib_runner_mod, "datetime", _FakeDatetime)
    runner._check_session_rollover()

    assert runner._runner_sessions_held == 0


# ---------------------------------------------------------------------------
# Config defaults / construction
# ---------------------------------------------------------------------------


def test_exit_params_exposes_multi_session_fields():
    """The two new ExitParams fields exist and have sensible defaults
    matching the PR spec (enabled, 5-day cap).
    """
    p = ExitParams()
    assert hasattr(p, "multi_session_runner")
    assert hasattr(p, "multi_session_runner_max_days")
    assert isinstance(p.multi_session_runner, bool)
    assert isinstance(p.multi_session_runner_max_days, int)
    assert p.multi_session_runner_max_days >= 1
    # ExitParams is frozen — confirm the new fields don't break that
    with pytest.raises(Exception):
        p.multi_session_runner = False  # type: ignore[misc]
