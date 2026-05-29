"""Tests for the opt-in EOD flatten (exit_params.eod_flatten_enabled).

The user observed live trades being killed by the intraday EOD flatten:
positions hit T1 then got force-closed before they could reach T2, leaving
money on the table. The flag `eod_flatten_enabled` (default False) makes
the EOD flatten opt-in. With the flag OFF (the new default), all phases
hold across EOD via their existing stops; the multi_session_runner_max_days
cap still applies as a safety net.

This file covers BOTH execution paths that drive EOD behavior:
  * live/ib_runner.py::_check_eod
  * strategy/mancini_long.py::run_day (the EOD loop at end of run_day)

Coverage:
  1. eod_flatten_enabled=False (new default):
       a. INITIAL position holds across EOD; stop unchanged.
       b. AFTER_T1 position holds across EOD; structure trail ratchets.
       c. AFTER_T2 position holds across EOD; structure trail ratchets.
       d. Max-days cap force-flattens any phase after N sessions held.
       e. Session rollover bumps counter for ANY held phase (not just AFTER_T2).
  2. eod_flatten_enabled=True (legacy):
       a. INITIAL/AFTER_T1 still flatten regardless of multi_session_runner.
       b. AFTER_T2 honors multi_session_runner.
"""

from __future__ import annotations

from datetime import date, datetime, time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import pytz

from config.settings import (
    DEFAULT_EXIT,
    ESContractSpec,
    ExitParams,
    SessionTimes,
    StrategyParams,
)
from strategy.exit_manager import ExitManager, ExitPhase, TradePosition
from strategy.mancini_long import ManciniLongStrategy


_ET = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# Live (ib_runner) helpers
# ---------------------------------------------------------------------------


def _make_runner_stub(
    eod_flatten_enabled: bool = False,
    multi_session: bool = True,
    max_days: int = 5,
    sessions_held: int = 0,
):
    """Build a minimal IBRunner exercising _check_eod / _check_session_rollover.

    Mirrors tests/test_multi_session_runner.py::_make_runner_stub but
    parameterises the new eod_flatten_enabled flag (default False to
    match production).
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
        eod_flatten_enabled=eod_flatten_enabled,
    )
    runner.exit_params = exit_params

    session = SessionTimes()  # 9:30-16:00 RTH, eod_flatten=15:55
    runner.strategy = SimpleNamespace(session_times=session, reset=lambda: None)

    runner.bridge = MagicMock()
    runner.bridge.get_prior_day_bars.return_value = None

    runner.exit_manager = ExitManager(
        params=exit_params,
        contract=ESContractSpec(),
    )

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

    runner.signal_aggregator = SimpleNamespace(
        get_pattern_state=lambda: {},
        restore_pattern_state=lambda state: None,
        initialize_levels=lambda *a, **k: None,
    )

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


def _make_live_position(
    phase: ExitPhase,
    entry_price: float = 6754.0,
    stop_price: float = 6736.0,
    contracts: int = 1,
    direction: str = "long",
) -> TradePosition:
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
# Backtest (mancini_long) helpers
# ---------------------------------------------------------------------------


def _make_strategy(
    eod_flatten_enabled: bool = False,
    multi_session: bool = True,
    max_days: int = 5,
) -> ManciniLongStrategy:
    strategy_params = StrategyParams(
        use_regime_filter=False,
        use_mode1_detection=False,
        use_mode1_green_detection=False,
        use_intraday_context=False,
        candle_bias_filter=False,
        short_candle_bias_filter=False,
    )
    exit_params = ExitParams(
        default_contracts=4,
        t1_exit_fraction=0.75,
        t2_exit_fraction=0.15,
        runner_fraction=0.10,
        multi_session_runner=multi_session,
        multi_session_runner_max_days=max_days,
        eod_flatten_enabled=eod_flatten_enabled,
    )
    return ManciniLongStrategy(
        strategy_params=strategy_params,
        exit_params=exit_params,
    )


def _make_flat_df(
    bars: int = 30,
    high: float = 6780.0,
    low: float = 6770.0,
    start: datetime | None = None,
) -> pd.DataFrame:
    """Build a quiet OHLCV frame that produces no signals."""
    if start is None:
        start = datetime(2026, 5, 14, 9, 30)
    mid = (high + low) / 2
    idx = pd.date_range(start, periods=bars, freq="1min")
    return pd.DataFrame(
        {
            "open": [mid] * bars,
            "high": [high] * bars,
            "low": [low] * bars,
            "close": [mid] * bars,
            "volume": [100] * bars,
        },
        index=idx,
    )


def _carry(
    pos: TradePosition,
    direction: str = "long",
    pattern_type: str = "FB_LONG",
) -> SimpleNamespace:
    return SimpleNamespace(
        position=pos,
        pattern_type=pattern_type,
        signal=None,
        entry_date=datetime(2026, 5, 12).date(),
        direction=direction,
        cumulative_bars=390,
    )


def _make_bt_position(
    phase: ExitPhase,
    entry_price: float = 6754.0,
    stop_price: float = 6730.0,
    target_1: float = 6764.0,
    target_2: float = 6790.0,
    contracts: int = 4,
    direction: str = "long",
    sessions_held: int = 0,
) -> TradePosition:
    """Build a TradePosition with the survived-sessions counter pre-set."""
    pos = TradePosition(
        entry_price=entry_price,
        stop_price=stop_price,
        target_1=target_1,
        target_2=target_2,
        total_contracts=contracts,
        remaining_contracts=contracts,
        direction=direction,
    )
    pos.phase = phase
    pos._runner_sessions_held = sessions_held
    return pos


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_exit_params_eod_flatten_defaults_off():
    """The new field defaults to False — EOD flatten is OPT-IN, not default."""
    p = ExitParams()
    assert hasattr(p, "eod_flatten_enabled")
    assert p.eod_flatten_enabled is False
    # Field is part of frozen dataclass
    with pytest.raises(Exception):
        p.eod_flatten_enabled = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Default (eod_flatten_enabled=False): live path holds every phase
# ---------------------------------------------------------------------------


def test_default_initial_position_holds_across_eod_live():
    """With the flag OFF, an INITIAL position is NOT flattened at EOD.
    The initial stop continues to protect the trade overnight.
    """
    runner = _make_runner_stub(eod_flatten_enabled=False)
    pos = _make_live_position(
        phase=ExitPhase.INITIAL,
        stop_price=6736.0,
        contracts=4,
    )
    runner._position = pos
    runner._trade_id = 7

    runner._check_eod(_eod_bar())

    # No flatten — position survives
    runner.bridge.flatten.assert_not_called()
    assert runner._position is pos
    assert runner._position.is_open
    # INITIAL stop unchanged (update_prior_day_low is a no-op for pre-T1)
    assert pos.stop_price == 6736.0
    assert pos.phase == ExitPhase.INITIAL


def test_default_after_t1_position_holds_across_eod_live():
    """AFTER_T1 (25% tranche) holds across EOD with the flag off. The
    structural trail ratchets to (session_low - buffer).
    """
    runner = _make_runner_stub(eod_flatten_enabled=False)
    pos = _make_live_position(
        phase=ExitPhase.AFTER_T1,
        stop_price=6730.0,  # below today's low so ratchet has room
        contracts=2,
    )
    runner._position = pos
    runner._trade_id = 11

    runner._check_eod(_eod_bar())

    runner.bridge.flatten.assert_not_called()
    assert runner._position is pos
    assert runner._position.is_open
    # Trail moved up to session_low(6740) - buffer(1) = 6739
    expected_stop = 6740.0 - runner.exit_params.runner_prior_day_low_buffer_pts
    assert pos.stop_price == pytest.approx(expected_stop)
    runner.bridge.update_stop.assert_called_once()


def test_default_after_t2_position_holds_across_eod_live():
    """AFTER_T2 still holds (matches prior behavior when
    multi_session_runner=True), trail ratchets to today's session low.
    """
    runner = _make_runner_stub(eod_flatten_enabled=False)
    pos = _make_live_position(
        phase=ExitPhase.AFTER_T2,
        stop_price=6730.0,
        contracts=1,
    )
    runner._position = pos
    runner._trade_id = 19

    runner._check_eod(_eod_bar())

    runner.bridge.flatten.assert_not_called()
    assert runner._position is pos
    expected_stop = 6740.0 - runner.exit_params.runner_prior_day_low_buffer_pts
    assert pos.stop_price == pytest.approx(expected_stop)


# ---------------------------------------------------------------------------
# Default — max-days cap STILL applies and now covers ANY phase
# ---------------------------------------------------------------------------


def test_default_max_days_cap_force_flattens_initial_position_live():
    """Even with EOD flatten off, the safety cap force-flattens once the
    counter hits max_days — regardless of phase. INITIAL gets the same
    safety net as runners.
    """
    runner = _make_runner_stub(
        eod_flatten_enabled=False,
        max_days=3,
        sessions_held=3,  # at cap — next EOD must flatten
    )
    pos = _make_live_position(
        phase=ExitPhase.INITIAL,
        stop_price=6736.0,
        contracts=4,
    )
    runner._position = pos
    runner._trade_id = 31

    runner._check_eod(_eod_bar())

    runner.bridge.flatten.assert_called_once()
    assert (
        runner.bridge.flatten.call_args.kwargs.get("reason")
        == "eod_flatten_max_days"
    )
    assert runner._position is None
    assert runner._runner_sessions_held == 0  # reset on flatten


def test_default_max_days_cap_force_flattens_after_t1_position_live():
    """Same cap applies to AFTER_T1 with the flag off."""
    runner = _make_runner_stub(
        eod_flatten_enabled=False,
        max_days=4,
        sessions_held=4,
    )
    runner._position = _make_live_position(
        phase=ExitPhase.AFTER_T1,
        contracts=2,
    )
    runner._trade_id = 33

    runner._check_eod(_eod_bar())

    runner.bridge.flatten.assert_called_once()
    assert (
        runner.bridge.flatten.call_args.kwargs.get("reason")
        == "eod_flatten_max_days"
    )
    assert runner._position is None


# ---------------------------------------------------------------------------
# Default — session rollover bumps counter for ANY held phase
# ---------------------------------------------------------------------------


def test_default_session_rollover_bumps_counter_for_initial(monkeypatch):
    """When the flag is off, a session rollover must bump the
    sessions_held counter for ANY held position, including INITIAL.
    This lets the max-days cap accumulate correctly across rolls.
    """
    runner = _make_runner_stub(
        eod_flatten_enabled=False,
        sessions_held=0,
    )
    pos = _make_live_position(phase=ExitPhase.INITIAL, contracts=4)
    runner._position = pos
    runner._pattern_type = "FB_LONG"

    target = datetime(2026, 5, 15, 19, 0, tzinfo=_ET)
    import live.ib_runner as ib_runner_mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return target if tz is None else target.astimezone(tz)

    monkeypatch.setattr(ib_runner_mod, "datetime", _FakeDatetime)
    runner._check_session_rollover()

    assert runner._runner_sessions_held == 1
    # Position transferred forward
    assert runner.position_manager.session.active_position is pos


def test_default_session_rollover_bumps_counter_for_after_t1(monkeypatch):
    runner = _make_runner_stub(eod_flatten_enabled=False, sessions_held=2)
    pos = _make_live_position(phase=ExitPhase.AFTER_T1, contracts=2)
    runner._position = pos

    target = datetime(2026, 5, 15, 19, 0, tzinfo=_ET)
    import live.ib_runner as ib_runner_mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return target if tz is None else target.astimezone(tz)

    monkeypatch.setattr(ib_runner_mod, "datetime", _FakeDatetime)
    runner._check_session_rollover()

    assert runner._runner_sessions_held == 3


# ---------------------------------------------------------------------------
# Legacy (eod_flatten_enabled=True): INITIAL / AFTER_T1 still flatten
# ---------------------------------------------------------------------------


def test_legacy_initial_position_still_flattens_at_eod_live():
    """With the flag explicitly ON, the legacy behavior survives:
    INITIAL flattens at EOD just like before."""
    runner = _make_runner_stub(
        eod_flatten_enabled=True, multi_session=True
    )
    runner._position = _make_live_position(
        phase=ExitPhase.INITIAL, contracts=4
    )

    runner._check_eod(_eod_bar())

    runner.bridge.flatten.assert_called_once()
    assert runner.bridge.flatten.call_args.kwargs.get("reason") == "eod_flatten"
    assert runner._position is None


def test_legacy_after_t1_position_still_flattens_at_eod_live():
    """AFTER_T1 flatten in legacy mode (only AFTER_T2 + multi_session
    flag enables the hold)."""
    runner = _make_runner_stub(
        eod_flatten_enabled=True, multi_session=True
    )
    runner._position = _make_live_position(
        phase=ExitPhase.AFTER_T1, contracts=2
    )

    runner._check_eod(_eod_bar())

    runner.bridge.flatten.assert_called_once()


def test_legacy_after_t2_honours_multi_session_runner_flag_live():
    """AFTER_T2 with multi_session_runner=True holds; with =False flattens."""
    # Holds
    runner_hold = _make_runner_stub(
        eod_flatten_enabled=True, multi_session=True
    )
    pos_hold = _make_live_position(
        phase=ExitPhase.AFTER_T2, stop_price=6730.0, contracts=1
    )
    runner_hold._position = pos_hold
    runner_hold._trade_id = 1
    runner_hold._check_eod(_eod_bar())
    runner_hold.bridge.flatten.assert_not_called()
    assert runner_hold._position is pos_hold

    # Flattens
    runner_flat = _make_runner_stub(
        eod_flatten_enabled=True, multi_session=False
    )
    runner_flat._position = _make_live_position(
        phase=ExitPhase.AFTER_T2, contracts=1
    )
    runner_flat._check_eod(_eod_bar())
    runner_flat.bridge.flatten.assert_called_once()


# ---------------------------------------------------------------------------
# Backtest path (strategy/mancini_long.py): same semantics
# ---------------------------------------------------------------------------


def test_default_initial_position_holds_across_eod_backtest():
    """Backtest: INITIAL position with eod_flatten_enabled=False is NOT
    closed at the end of run_day. The position survives for the next
    session's run_day to pick up via runner_state.
    """
    strategy = _make_strategy(eod_flatten_enabled=False)
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    pos = _make_bt_position(
        phase=ExitPhase.INITIAL,
        stop_price=6700.0,  # well below band so no in-bar trigger
        target_1=6900.0,    # well above band
        target_2=6950.0,
        contracts=4,
    )
    strategy.run_day(df, runner_state=_carry(pos, "long"))

    # Position still open and held by the strategy
    assert strategy._long_position is pos
    assert pos.is_open
    assert pos.phase == ExitPhase.INITIAL
    # Counter bumped — this session counts toward the safety cap
    assert pos._runner_sessions_held == 1


def test_default_after_t1_position_holds_across_eod_backtest():
    """Backtest: AFTER_T1 (25% tranche) holds with the flag off. The
    structure trail ratchets to (session_low - buffer)."""
    strategy = _make_strategy(eod_flatten_enabled=False)
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    pos = _make_bt_position(
        phase=ExitPhase.AFTER_T1,
        stop_price=6730.0,
        target_2=6900.0,  # avoid in-bar T2 trigger
        contracts=2,
    )
    strategy.run_day(df, runner_state=_carry(pos, "long"))

    assert strategy._long_position is pos
    assert pos.is_open
    assert pos.phase == ExitPhase.AFTER_T1
    # Trail moved up to session_low (6770) - buffer (1) = 6769
    expected_stop = 6770.0 - DEFAULT_EXIT.runner_prior_day_low_buffer_pts
    assert pos.stop_price == pytest.approx(expected_stop)
    assert pos._runner_sessions_held == 1


def test_default_max_days_cap_force_flattens_initial_backtest():
    """Backtest: even with the flag off, max_days cap force-flattens
    INITIAL positions once the per-position counter hits the cap.
    """
    strategy = _make_strategy(eod_flatten_enabled=False, max_days=3)
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    pos = _make_bt_position(
        phase=ExitPhase.INITIAL,
        stop_price=6700.0,
        target_1=6900.0,
        target_2=6950.0,
        contracts=4,
        sessions_held=3,  # at cap
    )
    strategy.run_day(df, runner_state=_carry(pos, "long"))

    # Flattened due to max-days cap
    assert strategy._long_position is None
    assert pos.phase == ExitPhase.CLOSED
    assert pos.remaining_contracts == 0


def test_legacy_after_t1_flattens_at_eod_backtest():
    """Backtest legacy mode: AFTER_T1 still flattens at EOD."""
    strategy = _make_strategy(
        eod_flatten_enabled=True, multi_session=True
    )
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    pos = _make_bt_position(
        phase=ExitPhase.AFTER_T1,
        stop_price=6730.0,
        target_2=6900.0,
        contracts=2,
    )
    strategy.run_day(df, runner_state=_carry(pos, "long"))

    assert strategy._long_position is None
    assert pos.phase == ExitPhase.CLOSED


def test_legacy_initial_flattens_at_eod_backtest():
    """Backtest legacy mode: INITIAL still flattens at EOD."""
    strategy = _make_strategy(
        eod_flatten_enabled=True, multi_session=True
    )
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    pos = _make_bt_position(
        phase=ExitPhase.INITIAL,
        stop_price=6700.0,
        target_1=6900.0,
        target_2=6950.0,
        contracts=4,
    )
    strategy.run_day(df, runner_state=_carry(pos, "long"))

    assert strategy._long_position is None
    assert pos.phase == ExitPhase.CLOSED


def test_legacy_after_t2_holds_with_multi_session_backtest():
    """Backtest legacy mode: AFTER_T2 with multi_session_runner=True
    still holds as before — this PR doesn't change legacy semantics."""
    strategy = _make_strategy(
        eod_flatten_enabled=True, multi_session=True
    )
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    pos = _make_bt_position(
        phase=ExitPhase.AFTER_T2,
        stop_price=6730.0,
        target_2=6900.0,
        contracts=1,
    )
    strategy.run_day(df, runner_state=_carry(pos, "long"))

    assert strategy._long_position is pos
    assert pos.is_open
    assert pos._runner_sessions_held == 1
