"""News-reaction entry blackout — calendar-free, price-reactive.

Data mornings are hostile to our entries (study 2026-07-14: all 10 entries on
data mornings netted -234pts vs +3597 across 110 normal-day entries; trade 732
chased the CPI spike 26pt extended and lost -65). Mancini's data-day rule:
"sit back, hold runners, and wait."

No calendar needed: the release bar announces itself (CPI's 8:30 bar was 55pt
vs ~2pt normal). When a scheduled-release minute (8:30 / 10:00 / 14:00 ET)
prints a bar range >= threshold, NEW entries are blocked for a cooldown.
Exits/stops/runners are never affected.
"""
from datetime import datetime, time as dt_time

import pytz

from core.signals import NewsBlackout

_ET = pytz.timezone("US/Eastern")


def _ts(hh, mm):
    return _ET.localize(datetime(2026, 7, 14, hh, mm))


def _bo(**kw):
    p = {"news_bar_range_pts": 8.0, "news_blackout_minutes": 30}
    p.update(kw)
    from types import SimpleNamespace
    return NewsBlackout(SimpleNamespace(**p))


def test_violent_830_bar_triggers_blackout():
    b = _bo()
    b.observe_bar(_ts(8, 30), high=7611.5, low=7556.5)   # 55pt CPI bar
    assert b.blocked(_ts(8, 32)) is True                  # 732 would be blocked
    assert b.blocked(_ts(8, 55)) is True
    assert b.blocked(_ts(9, 1)) is False                  # 31 min later: clear


def test_quiet_830_bar_does_not_trigger():
    b = _bo()
    b.observe_bar(_ts(8, 30), high=7560.0, low=7558.0)    # normal 2pt bar
    assert b.blocked(_ts(8, 32)) is False


def test_violent_bar_at_non_release_minute_ignored():
    b = _bo()
    b.observe_bar(_ts(11, 17), high=7600.0, low=7580.0)   # random 20pt bar
    assert b.blocked(_ts(11, 20)) is False


def test_1000_and_1400_release_minutes_covered():
    b = _bo()
    b.observe_bar(_ts(10, 0), high=7600.0, low=7590.0)    # 10pt @ 10:00
    assert b.blocked(_ts(10, 10)) is True
    b2 = _bo()
    b2.observe_bar(_ts(14, 0), high=7600.0, low=7588.0)   # FOMC 2pm
    assert b2.blocked(_ts(14, 25)) is True


def test_threshold_zero_disables():
    b = _bo(news_bar_range_pts=0.0)
    b.observe_bar(_ts(8, 30), high=7611.5, low=7556.5)
    assert b.blocked(_ts(8, 32)) is False


def test_naive_timestamps_handled():
    b = _bo()
    b.observe_bar(datetime(2026, 7, 14, 8, 30), high=7611.5, low=7556.5)
    assert b.blocked(datetime(2026, 7, 14, 8, 40)) is True


# --- Forecast layer: events known the night before (from Mancini's post) ---

def _bo_sched(**kw):
    p = {"news_bar_range_pts": 8.0, "news_blackout_minutes": 30,
         "news_pre_blackout_minutes": 15}
    p.update(kw)
    from types import SimpleNamespace
    return NewsBlackout(SimpleNamespace(**p))


def test_scheduled_event_blocks_before_and_after():
    b = _bo_sched()
    b.set_scheduled_events(["08:30 CPI"])
    assert b.blocked(_ts(8, 14)) is False    # before the pre-window
    assert b.blocked(_ts(8, 16)) is True     # 15 min before: paused
    assert b.blocked(_ts(8, 30)) is True     # the release itself
    assert b.blocked(_ts(8, 59)) is True     # inside the 30-min post window
    assert b.blocked(_ts(9, 1)) is False     # clear


def test_scheduled_layer_off_when_pre_minutes_zero():
    b = _bo_sched(news_pre_blackout_minutes=0)
    b.set_scheduled_events(["08:30 CPI"])
    assert b.blocked(_ts(8, 20)) is False    # forecast layer disabled
    # reactive layer still works independently
    b.observe_bar(_ts(8, 30), high=7611.5, low=7556.5)
    assert b.blocked(_ts(8, 40)) is True


def test_malformed_events_ignored():
    b = _bo_sched()
    b.set_scheduled_events(["CPI sometime", "", "25:99 X", "14:00 FOMC"])
    assert b.blocked(_ts(13, 50)) is True    # the one valid event works
    assert b.blocked(_ts(12, 0)) is False


def test_events_reset_each_session():
    b = _bo_sched()
    b.set_scheduled_events(["08:30 CPI"])
    b.set_scheduled_events([])               # next day: no data mentioned
    assert b.blocked(_ts(8, 30)) is False
