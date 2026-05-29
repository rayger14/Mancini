"""Plan loading honors the Globex 18:00 ET session boundary.

Background
----------
CME Globex ES sessions run 18:00 ET (previous calendar day) -> 17:00 ET. The
nightly `mancini_plan_<trading_date>.json` is keyed by the day the session
CLOSES — so a session starting at 18:00 on May 20 must load
``mancini_plan_2026-05-21.json``, NOT ``mancini_plan_2026-05-20.json``.

Live regression (2026-05-20 19:51 ET): the bot restarted past 18:00 ET and
loaded yesterday's plan because ``_session_date`` was set from
``date.today()`` rather than the Globex-aware trading date.

These tests pin:
  1. The static helper ``IBRunner._compute_globex_trading_date`` — before
     18:00 keeps today's date; at/after 18:00 returns the next calendar day.
  2. ``_load_mancini_llm_plan`` reads the file keyed by ``self._session_date``
     (the input we'd be wrong about if step 1 were wrong).
  3. ``_check_session_rollover`` calls the same plan loader after rollover,
     so a long-running bot picks up the new session's plan without restart.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytz

from config.settings import ExitParams, ESContractSpec


_ET = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_plan(tmp_path: Path, trading_date: str, mode: str = "range") -> Path:
    """Materialise a minimal but schema-valid plan JSON keyed by trading_date."""
    payload = {
        "schema_version": 1,
        "trading_date": trading_date,
        "source_url": "https://example.test/post",
        "source_published_at": f"{trading_date}T07:00:00-04:00",
        "extract_status": "ok",
        "plan": {
            "lean": "bullish",
            "mode": mode,
            "planned_setups": [],
            "danger_zones": [],
            "no_trade_above": None,
            "no_trade_below": None,
            "targets": [],
            "notes": "",
        },
    }
    p = tmp_path / f"mancini_plan_{trading_date}.json"
    p.write_text(json.dumps(payload))
    return p


def _make_runner_stub(plan_dir: Path, session_date: date):
    """Construct a bare IBRunner with just enough surface to drive
    ``_load_mancini_llm_plan`` and ``_check_session_rollover``.

    Mirrors the technique in tests/test_multi_session_runner.py: bypass
    __init__ (which would connect to IB Gateway) and stitch the attributes
    used by the methods under test.
    """
    from live.ib_runner import IBRunner

    runner = IBRunner.__new__(IBRunner)

    # Strategy carries the StrategyParams the loader reads
    sp = SimpleNamespace(
        use_mancini_llm_plan=True,
        mancini_llm_plan_dir=str(plan_dir),
    )
    runner.strategy = SimpleNamespace(
        strategy_params=sp,
        reset=lambda: None,
    )

    # Signal aggregator: spy on set_mancini_llm_plan so we can assert it
    # received the right plan after each load.
    runner.signal_aggregator = SimpleNamespace(
        set_mancini_llm_plan=MagicMock(),
        get_pattern_state=lambda: {},
        restore_pattern_state=lambda state: None,
        initialize_levels=lambda *a, **k: None,
    )

    # Bare attrs touched by rollover
    runner._session_date = session_date
    runner._mancini_llm_plan = None
    runner._position = None
    runner._trade_id = None
    runner._pattern_type = ""
    runner._df = None
    runner._bar_count = 0
    runner._phantom_positions = []
    runner._near_miss_phantoms = []
    runner._runner_sessions_held = 0

    runner.exit_params = ExitParams(multi_session_runner=False)
    runner.bridge = MagicMock()
    runner.bridge.get_prior_day_bars.return_value = None

    runner.position_manager = SimpleNamespace(
        close_position=MagicMock(return_value=None),
        start_session=lambda dt: None,
        session=SimpleNamespace(
            active_position=None,
            active_long=None,
            active_short=None,
            trades=[],
        ),
    )

    runner._archive_session = MagicMock()
    runner._log_session_summary = MagicMock()

    return runner


# ---------------------------------------------------------------------------
# 1. _compute_globex_trading_date pure-function behaviour
# ---------------------------------------------------------------------------


class TestComputeGlobexTradingDate:
    """The static helper that decides ``_session_date``."""

    def _call(self, dt_naive: datetime) -> date:
        from live.ib_runner import IBRunner
        return IBRunner._compute_globex_trading_date(_ET.localize(dt_naive))

    def test_pre_session_open_returns_calendar_day(self):
        # 09:30 ET on May 20 — well before 18:00. trading_date = May 20.
        assert self._call(datetime(2026, 5, 20, 9, 30)) == date(2026, 5, 20)

    def test_just_before_eighteen_oclock_still_current_day(self):
        # 17:59 ET on May 20 — boundary is exclusive below. May 20.
        assert self._call(datetime(2026, 5, 20, 17, 59, 59)) == date(2026, 5, 20)

    def test_exactly_eighteen_oclock_rolls_to_next_day(self):
        # 18:00 ET on May 20 — Globex open, this is the May 21 session.
        assert self._call(datetime(2026, 5, 20, 18, 0)) == date(2026, 5, 21)

    def test_after_eighteen_oclock_rolls_to_next_day(self):
        # The literal bug scenario: 19:51 ET on 2026-05-20 must yield
        # 2026-05-21 so the right plan file is loaded.
        assert self._call(datetime(2026, 5, 20, 19, 51, 28)) == date(2026, 5, 21)

    def test_late_night_after_midnight_still_session_day(self):
        # 02:00 ET on May 21 is INSIDE the same session that opened 18:00
        # May 20. trading_date is May 21 (the day the session closes).
        assert self._call(datetime(2026, 5, 21, 2, 0)) == date(2026, 5, 21)

    def test_morning_pre_close_keeps_current_day(self):
        # 11:00 ET on May 21 — between Globex open (18:00 prior) and the
        # 17:00 close. trading_date = May 21.
        assert self._call(datetime(2026, 5, 21, 11, 0)) == date(2026, 5, 21)

    def test_month_boundary_rolls_correctly(self):
        # 20:00 ET on May 31 must yield June 1, not May 32.
        assert self._call(datetime(2026, 5, 31, 20, 0)) == date(2026, 6, 1)


# ---------------------------------------------------------------------------
# 2. _load_mancini_llm_plan honors self._session_date
# ---------------------------------------------------------------------------


class TestLoadManciniLLMPlan:
    """The loader pulls the file keyed by ``_session_date`` — verifying that
    Step 1's date is what actually drives the disk read."""

    def test_loads_plan_keyed_by_session_date(self, tmp_path):
        # Materialise both plans; loader must pick the May 21 file when
        # session_date=May 21.
        _write_plan(tmp_path, "2026-05-20", mode="range")
        _write_plan(tmp_path, "2026-05-21", mode="trend")

        runner = _make_runner_stub(tmp_path, session_date=date(2026, 5, 21))
        runner._load_mancini_llm_plan()

        assert runner._mancini_llm_plan is not None
        assert runner._mancini_llm_plan.mode == "trend"
        runner.signal_aggregator.set_mancini_llm_plan.assert_called_once_with(
            runner._mancini_llm_plan
        )

    def test_loads_yesterday_when_session_date_is_yesterday(self, tmp_path):
        # Symmetric check — proves the loader is genuinely keyed off
        # _session_date, not hard-coded to "today".
        _write_plan(tmp_path, "2026-05-20", mode="range")
        _write_plan(tmp_path, "2026-05-21", mode="trend")

        runner = _make_runner_stub(tmp_path, session_date=date(2026, 5, 20))
        runner._load_mancini_llm_plan()

        assert runner._mancini_llm_plan.mode == "range"

    def test_no_plan_file_sets_none(self, tmp_path):
        runner = _make_runner_stub(tmp_path, session_date=date(2026, 5, 21))
        runner._load_mancini_llm_plan()

        assert runner._mancini_llm_plan is None
        runner.signal_aggregator.set_mancini_llm_plan.assert_not_called()

    def test_feature_disabled_skips_load(self, tmp_path):
        _write_plan(tmp_path, "2026-05-21", mode="trend")
        runner = _make_runner_stub(tmp_path, session_date=date(2026, 5, 21))
        runner.strategy.strategy_params.use_mancini_llm_plan = False

        runner._load_mancini_llm_plan()

        assert runner._mancini_llm_plan is None
        runner.signal_aggregator.set_mancini_llm_plan.assert_not_called()


# ---------------------------------------------------------------------------
# 3. _check_session_rollover reloads the plan for the new trading_date
# ---------------------------------------------------------------------------


def test_session_rollover_reloads_plan_for_new_trading_date(monkeypatch, tmp_path):
    """When the Globex clock crosses 18:00 ET, rollover must reload the plan
    file keyed by the NEW session date. Without this, a long-running bot
    keeps qualifying signals against yesterday's plan after rollover.
    """
    # Plans for both sides of the rollover; new session = May 22.
    _write_plan(tmp_path, "2026-05-21", mode="range")
    _write_plan(tmp_path, "2026-05-22", mode="trend")

    runner = _make_runner_stub(tmp_path, session_date=date(2026, 5, 21))

    # Pre-load the May 21 plan so we can detect that it gets REPLACED.
    runner._load_mancini_llm_plan()
    assert runner._mancini_llm_plan.mode == "range"
    runner.signal_aggregator.set_mancini_llm_plan.reset_mock()

    # Freeze the clock past 18:00 ET on May 21 — the same trick used in
    # tests/test_multi_session_runner.py.
    target = datetime(2026, 5, 21, 19, 0, tzinfo=_ET)
    import live.ib_runner as ib_runner_mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return target if tz is None else target.astimezone(tz)

    monkeypatch.setattr(ib_runner_mod, "datetime", _FakeDatetime)

    runner._check_session_rollover()

    # Date advanced and the May 22 plan is now active.
    assert runner._session_date == date(2026, 5, 22)
    assert runner._mancini_llm_plan is not None
    assert runner._mancini_llm_plan.mode == "trend"
    # Aggregator received the new plan exactly once during rollover.
    runner.signal_aggregator.set_mancini_llm_plan.assert_called_once_with(
        runner._mancini_llm_plan
    )


def test_session_rollover_clears_plan_when_new_plan_missing(monkeypatch, tmp_path):
    """If the next session's plan hasn't been generated yet, rollover should
    drop the stale plan rather than silently keeping yesterday's. The bot
    can run without a plan (gates are no-ops); it MUST NOT keep using a
    plan that no longer applies."""
    _write_plan(tmp_path, "2026-05-21", mode="range")
    # Note: no file for 2026-05-22.

    runner = _make_runner_stub(tmp_path, session_date=date(2026, 5, 21))
    runner._load_mancini_llm_plan()
    assert runner._mancini_llm_plan is not None  # May 21 plan loaded

    target = datetime(2026, 5, 21, 19, 0, tzinfo=_ET)
    import live.ib_runner as ib_runner_mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return target if tz is None else target.astimezone(tz)

    monkeypatch.setattr(ib_runner_mod, "datetime", _FakeDatetime)
    runner._check_session_rollover()

    assert runner._session_date == date(2026, 5, 22)
    # Stale plan cleared.
    assert runner._mancini_llm_plan is None


def test_session_rollover_during_break_window_does_not_change_date(monkeypatch, tmp_path):
    """17:00-18:00 ET is the daily Globex break — rollover early-exits so
    _session_date and the loaded plan must not change."""
    _write_plan(tmp_path, "2026-05-21", mode="range")

    runner = _make_runner_stub(tmp_path, session_date=date(2026, 5, 21))
    runner._load_mancini_llm_plan()
    original_plan = runner._mancini_llm_plan
    assert original_plan is not None

    target = datetime(2026, 5, 21, 17, 30, tzinfo=_ET)  # in the break window
    import live.ib_runner as ib_runner_mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return target if tz is None else target.astimezone(tz)

    monkeypatch.setattr(ib_runner_mod, "datetime", _FakeDatetime)
    runner._check_session_rollover()

    # No change.
    assert runner._session_date == date(2026, 5, 21)
    assert runner._mancini_llm_plan is original_plan
