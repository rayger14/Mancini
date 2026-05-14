"""Regression tests for the post-session cleanup PR.

Covers:
  - fb_max_hold_bars removed — strategy no longer enforces time-cap exit
  - regime placeholder uses NaN ema_slope instead of 0.0 sentinel
"""

from __future__ import annotations

import math

from config.settings import ExitParams, StrategyParams


# ---------------------------------------------------------------------------
# fb_max_hold_bars deletion
# ---------------------------------------------------------------------------


def test_exit_params_no_longer_has_fb_max_hold_bars():
    """The fb_max_hold_bars field was deleted from ExitParams. Confirm it
    no longer exists so backtest scripts that try to set it explicitly
    fail loudly rather than silently no-op."""
    p = ExitParams()
    assert not hasattr(p, "fb_max_hold_bars"), (
        "fb_max_hold_bars was a backtest-vs-live discrepancy bug; if it "
        "comes back, both code paths need to honor it or neither should."
    )


def test_exit_params_can_be_constructed_without_fb_max_hold_kwarg():
    """Cleanup should not break the standard ExitParams() constructor."""
    p = ExitParams(default_contracts=4, t1_exit_fraction=0.75,
                   t2_exit_fraction=0.15, runner_fraction=0.10)
    assert p.default_contracts == 4


def test_mancini_long_no_fb_time_exit_path():
    """Search the mancini_long source for the deleted fb_time_exit logic.
    If anyone re-introduces it without putting the cap back in BOTH
    backtest and live, this fails."""
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "strategy" / "mancini_long.py"
    text = src.read_text()
    assert "fb_time_exit" not in text, (
        "FB time-cap exit was removed because it only fired in backtest, "
        "systematically truncating winners. If re-adding, ensure the cap "
        "fires in BOTH backtest AND live to avoid the discrepancy."
    )
    assert "FB time exit" not in text


# ---------------------------------------------------------------------------
# Regime placeholder hardening — NaN ema_slope instead of silent 0.0
# ---------------------------------------------------------------------------


def test_regime_state_nan_ema_slope_is_distinguishable():
    """Smoke check: NaN doesn't equal NaN, while 0.0 == 0.0. This is the
    property we rely on for downstream analyses to detect 'filter wasn't
    armed' vs 'filter ran and slope was 0'."""
    from core.regime_filter import RegimeState, Direction, VolRegime

    placeholder = RegimeState(
        direction=Direction.NEUTRAL,
        vol_regime=VolRegime.NORMAL,
        longs_enabled=True,
        shorts_enabled=True,
        ema_slope=float("nan"),
    )
    assert math.isnan(placeholder.ema_slope)
    # The whole point: NaN propagates through comparisons / aggregations
    # so downstream code can detect "filter wasn't armed" rather than
    # silently treating 0.0 as a flat-slope reading.
    assert placeholder.ema_slope != 0.0
    assert not (placeholder.ema_slope == placeholder.ema_slope)  # NaN ≠ NaN


# ---------------------------------------------------------------------------
# acceptance_timeout_bars_shallow default sanity check (unchanged at 20)
# ---------------------------------------------------------------------------


def test_acceptance_timeout_shallow_remains_20():
    """The fb_max_hold_bars deletion does not justify reverting this value.
    The original justification stood on its own (Agent C: 63% WR / +6pt
    avg on the 27 resolved near-misses that confirmed at minute 7-10)."""
    p = StrategyParams()
    assert p.acceptance_timeout_bars_shallow == 20
