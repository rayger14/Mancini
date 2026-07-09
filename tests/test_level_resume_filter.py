"""Tests for the level-resume filter: before the FB fires on an ENGINE
auto-detected level, require the level's resume to show a proven rally
("proven launcher"). Mancini's CUSTOM plan levels are always exempt — this
filter only polices the levels our engine invents.

Validated on real history before building: auto-level FBs at proven-launcher
levels made +345pts; at weak levels lost -111 (59 real trades). His traded
levels' median rally resume: 53pt.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from core.signals import level_rally_resume, auto_level_resume_ok
from config.levels import Level, LevelType

_T0 = datetime(2026, 7, 6, 9, 30)


def _df(prices):
    """1-min bars from a list of (high, low) tuples; close = mid."""
    idx = [_T0 + timedelta(minutes=i) for i in range(len(prices))]
    return pd.DataFrame(
        {"high": [h for h, _ in prices], "low": [l for _, l in prices],
         "close": [(h + l) / 2 for h, l in prices]},
        index=pd.DatetimeIndex(idx))


def _lvl(price, level_type=LevelType.MULTI_HOUR_LOW, rally=0.0, mancini=False):
    return Level(price=price, level_type=level_type, created_at=_T0,
                 confirmed_at=_T0, rally_from_low_pts=rally,
                 mancini_confirmed=mancini)


# --- level_rally_resume: max run-up launched from the level in the window ---
def test_resume_measures_rally_from_level():
    # touch 6000, rally to 6062 before coming back → resume ≈ 62
    df = _df([(6010, 6000), (6030, 6008), (6062, 6025), (6040, 6020)])
    assert level_rally_resume(df, 6000.0) >= 60


def test_resume_small_when_level_never_launched():
    # touches 6000 twice, never gets more than ~8 above
    df = _df([(6006, 6000), (6008, 6002), (6005, 5999.5), (6007, 6001)])
    assert level_rally_resume(df, 6000.0) < 10


def test_resume_zero_when_level_untouched():
    df = _df([(6050, 6040), (6060, 6045)])
    assert level_rally_resume(df, 6000.0) == 0.0


def test_resume_handles_none_df():
    assert level_rally_resume(None, 6000.0) == 0.0


# --- auto_level_resume_ok: the gate ---
_P_ON = SimpleNamespace(fb_auto_level_min_rally_pts=50.0)
_P_OFF = SimpleNamespace(fb_auto_level_min_rally_pts=0.0)


def test_gate_off_passes_everything():
    assert auto_level_resume_ok(_lvl(6000), resume_pts=0.0, params=_P_OFF) is True


def test_weak_auto_level_blocked():
    assert auto_level_resume_ok(_lvl(6000), resume_pts=12.0, params=_P_ON) is False


def test_proven_auto_level_passes():
    assert auto_level_resume_ok(_lvl(6000), resume_pts=65.0, params=_P_ON) is True


def test_custom_mancini_level_always_exempt():
    lvl = _lvl(6000, level_type=LevelType.CUSTOM, mancini=True)
    assert auto_level_resume_ok(lvl, resume_pts=0.0, params=_P_ON) is True


def test_mancini_confirmed_engine_level_exempt():
    # an engine level Mancini's overlay confirmed is his call — exempt
    lvl = _lvl(6000, level_type=LevelType.MULTI_HOUR_LOW, mancini=True)
    assert auto_level_resume_ok(lvl, resume_pts=0.0, params=_P_ON) is True


# --- ResumeCache: don't rescan the tape on every FB re-emission ---
from core.signals import ResumeCache


def test_cache_computes_once_within_ttl():
    calls = []
    c = ResumeCache(ttl_bars=30)
    fn = lambda: (calls.append(1), 42.0)[1]
    assert c.get_or_compute(6000.0, bar_idx=100, compute_fn=fn) == 42.0
    assert c.get_or_compute(6000.0, bar_idx=120, compute_fn=fn) == 42.0  # cached
    assert len(calls) == 1


def test_cache_recomputes_after_ttl():
    calls = []
    c = ResumeCache(ttl_bars=30)
    fn = lambda: (calls.append(1), 42.0)[1]
    c.get_or_compute(6000.0, bar_idx=100, compute_fn=fn)
    c.get_or_compute(6000.0, bar_idx=131, compute_fn=fn)  # ttl expired
    assert len(calls) == 2


def test_cache_keys_by_level_price():
    calls = []
    c = ResumeCache(ttl_bars=30)
    fn = lambda: (calls.append(1), 1.0)[1]
    c.get_or_compute(6000.0, bar_idx=100, compute_fn=fn)
    c.get_or_compute(6010.0, bar_idx=100, compute_fn=fn)  # different level
    assert len(calls) == 2


def test_stored_rally_counts_via_max():
    # gate reads the FUSED resume: max(stored rally_from_low_pts, computed).
    # caller fuses; here stored=55 should pass through as resume_pts
    lvl = _lvl(6000, rally=55.0)
    assert auto_level_resume_ok(lvl, resume_pts=max(lvl.rally_from_low_pts, 3.0),
                                params=_P_ON) is True
