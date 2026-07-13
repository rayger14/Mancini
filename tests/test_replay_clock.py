"""Clock seams for the ReplayRunner (step 1 of the live-replica backtester).

IBRunner gets injectable clocks — `_now_fn` (wall clock) and `_mono_fn`
(monotonic) — with class-level defaults that are byte-identical to the old
hardcoded calls. A replay overrides them to follow the tape. These tests are
the no-behavior-change gate: defaults must equal the real clocks, and the
decision sites must actually consume the seams.
"""
import os
import time
from datetime import datetime, date
from types import SimpleNamespace

import pytest
import pytz

_ET = pytz.timezone("US/Eastern")


@pytest.fixture()
def runner_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setenv("SHADOW_LOG", str(tmp_path / "shadow.jsonl"))
    monkeypatch.setenv("FREEZE_TIMEOUT_SEC", "0")
    monkeypatch.setenv("SHORT_ALERTS", "0")
    monkeypatch.setenv("BLOCKED_ALERTS", "0")
    return tmp_path


def _make_runner(**kw):
    from live.ib_runner import IBRunner
    from live.ib_bridge import IBConfig
    return IBRunner(ib_config=IBConfig(), **kw)


def test_now_fn_defaults_to_wall_clock(runner_env):
    r = _make_runner()
    now = r._now_fn()
    assert abs((now - datetime.now(_ET)).total_seconds()) < 5
    assert now.tzinfo is not None


def test_mono_fn_defaults_to_monotonic(runner_env):
    r = _make_runner()
    assert abs(r._mono_fn() - time.monotonic()) < 1.0


def test_session_date_follows_injected_clock(runner_env):
    """__init__'s session date must come from the seam: freeze the clock at a
    Thursday 20:00 ET and the globex trading date must be Friday."""
    from live.ib_runner import IBRunner

    frozen = _ET.localize(datetime(2026, 7, 2, 20, 0))  # Thu 8pm ET

    class Frozen(IBRunner):
        _now_fn = staticmethod(lambda: frozen)

    from live.ib_bridge import IBConfig
    r = Frozen(ib_config=IBConfig())
    assert r._session_date == date(2026, 7, 3)


def test_entry_grace_uses_mono_fn(runner_env):
    """_sync_position's 45s entry grace must read the injected monotonic clock
    (in replay the wall clock barely moves, so this seam is what lets bracket
    closures ever get booked)."""
    r = _make_runner()
    clock = {"t": 1000.0}
    r._mono_fn = lambda: clock["t"]
    r._position = SimpleNamespace(is_open=True, remaining_contracts=1)
    r._last_entry_monotonic = r._mono_fn()

    class Tripwire:
        @property
        def is_connected(self):
            raise RuntimeError("past the grace gate")

    r.bridge = Tripwire()
    # inside the grace window -> returns before touching the bridge
    r._sync_position()
    # advance the injected clock past 45s -> the gate opens, bridge is touched
    clock["t"] += 60.0
    with pytest.raises(RuntimeError, match="past the grace gate"):
        r._sync_position()


def test_shadow_log_env_override_and_local_construction(runner_env, tmp_path):
    # constructing locally (no /app) must not raise, and SHADOW_LOG must win
    r = _make_runner()
    assert str(r._shadow_log_path) == str(tmp_path / "shadow.jsonl")


def test_build_live_runner_matches_main_construction(runner_env):
    """The factory must reproduce main()'s --full-session construction (the
    config the paper bot actually runs) so replay can never drift from live."""
    from live.ib_runner import build_live_runner, PRODUCTION_EXIT, FULL_SESSION
    from live.ib_bridge import IBConfig

    r = build_live_runner(IBConfig(), full_session=True)
    assert r._bypass_session_gates is True
    assert r._fb_only_pm is True
    assert r.exit_params is PRODUCTION_EXIT
    assert r.strategy.session_times is FULL_SESSION
    rp = r.strategy.risk_manager.risk_params
    assert rp.max_trades_per_day == 999
    assert rp.min_rr_ratio == 0.8
    assert rp.max_stop_distance_pts == 60.0
