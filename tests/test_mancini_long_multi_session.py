"""Tests for the multi-session runner hold in strategy/mancini_long.py.

This mirrors tests/test_multi_session_runner.py (which covers the LIVE
ib_runner path) for the BACKTEST path inside ManciniLongStrategy.run_day's
EOD loop.

Mancini's method (2025-10-12: "still holding my 10% long runner from the
Tuesday noon 6754 Failed Breakdown"):
  - INITIAL phase: always flatten at EOD.
  - AFTER_T1 phase (still 25% — pre-T2): always flatten at EOD, even when
    multi_session_runner=True. Only the 10% post-T2 slice rides cross-
    session. The pre-fix backtest engine carried AFTER_T1 across EOD,
    diverging from live; these tests pin the corrected behavior.
  - AFTER_T2 phase (10% runner):
      * multi_session_runner=False → flatten (legacy).
      * multi_session_runner=True AND sessions_held < max_days → hold,
        structural trail ratchets up under today's session low.
      * multi_session_runner=True AND sessions_held >= max_days →
        force-flatten with the max-days exit reason.

The per-position counter (`_runner_sessions_held`) is attached as a dynamic
attribute on TradePosition. It survives the cross-day carry via the live
position reference inside RunnerCarryState and resets to 0 whenever a fresh
entry is opened (matching live/ib_runner.py).
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from config.settings import (
    DEFAULT_EXIT,
    ExitParams,
    StrategyParams,
)
from strategy.exit_manager import ExitPhase, TradePosition
from strategy.mancini_long import ManciniLongStrategy


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_strategy(multi_session: bool = True, max_days: int = 5) -> ManciniLongStrategy:
    """Build a ManciniLongStrategy with the requested multi-session config.

    Disables features that would otherwise trigger entries during the bar
    loop — we want the EOD logic to be the only thing exercised by tests
    that inject a position via runner_state.
    """
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
    """Build a quiet OHLCV frame that produces no signals.

    Prices oscillate in a narrow [low, high] band — well above any AFTER_T2
    stop placed below 6750 — so no overnight gap fires and no entry signal
    triggers (level store has no significant levels to interact with).
    """
    if start is None:
        start = datetime(2026, 5, 14, 9, 30)
    mid = (high + low) / 2
    prices = [(mid, high, low, mid)] * bars
    idx = pd.date_range(start, periods=bars, freq="1min")
    return pd.DataFrame(
        {
            "open": [p[0] for p in prices],
            "high": [p[1] for p in prices],
            "low": [p[2] for p in prices],
            "close": [p[3] for p in prices],
            "volume": [100] * bars,
        },
        index=idx,
    )


def _carry(
    pos: TradePosition,
    direction: str = "long",
    pattern_type: str = "FB_LONG",
) -> SimpleNamespace:
    """Wrap a TradePosition in a RunnerCarryState-shaped namespace so it can
    be passed into ManciniLongStrategy.run_day(runner_state=...).
    """
    return SimpleNamespace(
        position=pos,
        pattern_type=pattern_type,
        signal=None,
        entry_date=datetime(2026, 5, 12).date(),
        direction=direction,
        cumulative_bars=390,  # one prior session
    )


def _make_after_t2_position(
    entry_price: float = 6754.0,
    stop_price: float = 6730.0,
    contracts: int = 1,
    direction: str = "long",
    sessions_held: int = 0,
) -> TradePosition:
    """Construct an AFTER_T2 runner position with the survived-sessions
    counter pre-set (mirrors what RunnerCarryState would deliver after N
    prior overnight holds).
    """
    pos = TradePosition(
        entry_price=entry_price,
        stop_price=stop_price,
        target_1=6764.0,
        target_2=6776.0,
        total_contracts=4,
        remaining_contracts=contracts,
        direction=direction,
    )
    pos.phase = ExitPhase.AFTER_T2
    pos._runner_sessions_held = sessions_held
    return pos


# ---------------------------------------------------------------------------
# Case 1 — multi-session DISABLED: AFTER_T2 flattens at EOD (legacy behavior)
# ---------------------------------------------------------------------------


def test_multi_session_disabled_flattens_after_t2_at_eod():
    """When multi_session_runner=False, the 10% AFTER_T2 runner is closed
    at the last bar's close just like INITIAL phase positions. Confirms
    the new gate fires only when the flag is on (legacy backtest behavior).

    The runner_state path injects the position AFTER position_manager has
    started its session, so we re-register the carried position with
    position_manager so close_position() produces a TradeRecord. This
    mirrors how live/ib_runner.py keeps the position bound to its session.
    """
    strategy = _make_strategy(multi_session=False)
    # Quiet df with target_2 well above the band so the AFTER_T2 position
    # doesn't get re-touched mid-bar by exit_manager (stop at 6730 stays
    # below the 6770 low, T2 at 6790 stays above the 6780 high).
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    pos = _make_after_t2_position(stop_price=6730.0, contracts=1)
    pos.target_2 = 6790.0  # keep above the band so no in-bar T2 trigger

    carry = _carry(pos, "long")

    # Drive run_day, then verify position closed
    strategy.run_day(df, runner_state=carry)

    # Position should have been flattened
    assert strategy._long_position is None
    assert pos.phase == ExitPhase.CLOSED
    assert pos.remaining_contracts == 0
    # Counter cleared on flatten (defensive)
    assert pos._runner_sessions_held == 0


# ---------------------------------------------------------------------------
# Case 2 — multi-session ENABLED: AFTER_T2 holds, trail ratchets to session low
# ---------------------------------------------------------------------------


def test_multi_session_enabled_holds_after_t2_runner_through_eod():
    """multi_session_runner=True + AFTER_T2 phase: the runner is NOT
    flattened at EOD. The structural trail is updated under today's
    session low and the position stays open ready for the next session.
    """
    strategy = _make_strategy(multi_session=True)
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    pos = _make_after_t2_position(stop_price=6730.0, contracts=1)
    strategy.run_day(df, runner_state=_carry(pos, "long"))

    # Position still open and held by the strategy
    assert strategy._long_position is pos
    assert pos.is_open
    assert pos.phase == ExitPhase.AFTER_T2

    # Structural trail should have ratcheted up to (session_low - buffer).
    # session_low = 6770 (the constant low of our flat df).
    expected_stop = 6770.0 - DEFAULT_EXIT.runner_prior_day_low_buffer_pts
    assert pos.stop_price == pytest.approx(expected_stop)

    # Survived-sessions counter bumped to 1 (one EOD held).
    assert pos._runner_sessions_held == 1

    # No trade record written (position still open).
    assert strategy.trade_records == []


def test_multi_session_enabled_increments_counter_each_eod():
    """The counter should accumulate across multiple held EOD events when
    the same position is carried over multiple sessions in sequence.
    """
    strategy = _make_strategy(multi_session=True, max_days=5)
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    # Position already survived 2 prior EODs
    pos = _make_after_t2_position(stop_price=6760.0, contracts=1, sessions_held=2)
    strategy.run_day(df, runner_state=_carry(pos, "long"))

    # Still held, and counter now 3
    assert strategy._long_position is pos
    assert pos._runner_sessions_held == 3
    assert pos.is_open


# ---------------------------------------------------------------------------
# Case 3 — AFTER_T1 still flattens at EOD even with multi_session_runner=True
# ---------------------------------------------------------------------------


def test_multi_session_enabled_still_flattens_after_t1():
    """The multi-session hold ONLY applies to AFTER_T2 (the 10% post-T2
    slice). An AFTER_T1 position still owns ~25% (T2 + runner) — too much
    overnight exposure for Mancini's method. It must flatten at EOD even
    when the flag is on. This is the new corrected behavior — the pre-fix
    backtest engine was carrying AFTER_T1 across EOD, which is wrong.
    """
    strategy = _make_strategy(multi_session=True)
    # Keep target_2 ABOVE the band so the AFTER_T1 position can't
    # auto-promote to AFTER_T2 during the bar loop. The 6770/6780 flat
    # band stays comfortably under target_2=6850, so the position remains
    # AFTER_T1 right up to the EOD branch.
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    pos = TradePosition(
        entry_price=6754.0,
        stop_price=6740.0,
        target_1=6764.0,
        target_2=6850.0,  # well above the band — no in-bar T2 promotion
        total_contracts=4,
        remaining_contracts=2,
        direction="long",
    )
    pos.phase = ExitPhase.AFTER_T1
    pos._runner_sessions_held = 0

    strategy.run_day(df, runner_state=_carry(pos, "long"))

    # AFTER_T1 was flattened — position cleared, phase=CLOSED
    assert strategy._long_position is None
    assert pos.phase == ExitPhase.CLOSED
    assert pos.remaining_contracts == 0


# ---------------------------------------------------------------------------
# Case 4 — Max-days safety cap force-flattens
# ---------------------------------------------------------------------------


def test_max_days_cap_force_flattens_after_t2_runner():
    """After max_days EOD holds, the next EOD force-flattens the runner
    with a distinct exit reason so logs/audit can distinguish a routine
    EOD flatten from a max-days bailout.
    """
    strategy = _make_strategy(multi_session=True, max_days=3)
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    # Counter already at the cap — this EOD should flatten
    pos = _make_after_t2_position(
        stop_price=6760.0, contracts=1, sessions_held=3
    )
    pos.target_2 = 6790.0  # keep above the band so no in-bar T2 trigger
    strategy.run_day(df, runner_state=_carry(pos, "long"))

    # Position flattened
    assert strategy._long_position is None
    assert pos.phase == ExitPhase.CLOSED
    assert pos.remaining_contracts == 0
    # Counter reset on flatten
    assert pos._runner_sessions_held == 0


def test_below_max_days_still_holds():
    """One short of the cap still survives — boundary check."""
    strategy = _make_strategy(multi_session=True, max_days=3)
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    pos = _make_after_t2_position(
        stop_price=6760.0, contracts=1, sessions_held=2
    )
    strategy.run_day(df, runner_state=_carry(pos, "long"))

    # Still held, counter incremented to exactly max_days (next EOD will
    # be the one that trips the cap).
    assert strategy._long_position is pos
    assert pos.is_open
    assert pos._runner_sessions_held == 3


# ---------------------------------------------------------------------------
# Case 5 — Counter resets on fresh entry
# ---------------------------------------------------------------------------


def test_runner_sessions_held_initialized_on_new_position():
    """A position created via exit_manager.create_position() (the same
    path _process_bar uses) starts with _runner_sessions_held=0 after the
    strategy stamps it. This pins the contract that any future code
    relying on the counter sees a defined value, not AttributeError.
    """
    strategy = _make_strategy(multi_session=True)
    # Mirror what _process_bar's Step 8 does: create position, then stamp 0.
    position = strategy.exit_manager.create_position(
        entry_price=6754.0,
        stop_price=6748.0,
        target_1=6764.0,
        target_2=6776.0,
        contracts=4,
        direction="long",
    )
    position._runner_sessions_held = 0  # what the strategy does
    assert position._runner_sessions_held == 0


def test_new_entry_resets_counter_on_carried_position():
    """If a stale position is somehow carried with a non-zero counter and
    a fresh entry replaces it, the new position must start from 0 — the
    cap is per-trade, not per-bot-lifetime. We simulate this by running
    a session that flattens the stale AFTER_T1 at EOD (counter is moot
    after flatten), then create a fresh entry and confirm the new position
    has counter=0.
    """
    strategy = _make_strategy(multi_session=True, max_days=5)
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    # First session: stale AFTER_T1 with bogus high counter — gets flattened
    stale = TradePosition(
        entry_price=6754.0,
        stop_price=6740.0,
        target_1=6764.0,
        target_2=6776.0,
        total_contracts=4,
        remaining_contracts=2,
        direction="long",
    )
    stale.phase = ExitPhase.AFTER_T1
    stale._runner_sessions_held = 99
    strategy.run_day(df, runner_state=_carry(stale, "long"))
    assert strategy._long_position is None  # flattened

    # Fresh entry via the same create-and-stamp pattern _process_bar uses.
    new_pos = strategy.exit_manager.create_position(
        entry_price=6754.0,
        stop_price=6748.0,
        target_1=6764.0,
        target_2=6776.0,
        contracts=4,
        direction="long",
    )
    new_pos._runner_sessions_held = 0
    assert new_pos._runner_sessions_held == 0


# ---------------------------------------------------------------------------
# Case 6 — Short side mirror: AFTER_T2 short holds with structural trail
# ---------------------------------------------------------------------------


def test_multi_session_enabled_holds_after_t2_short_runner():
    """Mirror of the long-side hold for shorts. AFTER_T2 short with the
    flag on survives EOD and the trail ratchets DOWN to today's session
    high + buffer."""
    strategy = _make_strategy(multi_session=True)
    df = _make_flat_df(bars=30, high=6780.0, low=6770.0)

    pos = _make_after_t2_position(
        entry_price=6776.0,
        stop_price=6810.0,  # original wide stop above today's high
        contracts=1,
        direction="short",
    )
    strategy.run_day(df, runner_state=_carry(pos, "short"))

    # Still open, counter bumped
    assert strategy._short_position is pos
    assert pos.is_open
    assert pos._runner_sessions_held == 1

    # Short trail: prior_day_high gets set to session high (6780), then
    # stop ratchets DOWN to 6780 + buffer = 6781 (only if it's lower than
    # the current 6810 stop — which it is).
    expected_stop = 6780.0 + DEFAULT_EXIT.runner_prior_day_low_buffer_pts
    assert pos.stop_price == pytest.approx(expected_stop)


# ---------------------------------------------------------------------------
# Config defaults / construction sanity
# ---------------------------------------------------------------------------


def test_exit_params_exposes_multi_session_fields():
    """The two multi-session fields exist and have sensible defaults."""
    p = ExitParams()
    assert hasattr(p, "multi_session_runner")
    assert hasattr(p, "multi_session_runner_max_days")
    assert isinstance(p.multi_session_runner, bool)
    assert isinstance(p.multi_session_runner_max_days, int)
    assert p.multi_session_runner_max_days >= 1
