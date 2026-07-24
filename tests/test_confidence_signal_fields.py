"""Phase A no-behavior-change guarantee: the confidence profile ANNOTATES
signals (predictions attached, logged, displayed) but must not alter any
trading decision — size factor identical with the flag on vs off."""
import json
from datetime import datetime

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.patterns import ConfirmationType, PatternSignal
from core.signals import SignalAggregator, SignalType

_TS = datetime(2026, 7, 20, 23, 19)


def _pattern():
    lvl = Level(price=7473.75, level_type=LevelType.CUSTOM,
                created_at=_TS, confirmed_at=_TS, touch_count=1)
    return PatternSignal(
        pattern_type="failed_breakdown",
        confirmation=ConfirmationType.NON_ACCEPTANCE,
        level=lvl, sweep_low=7470.0, entry_price=7505.75,
        stop_price=7467.75, bar_idx=100, timestamp=_TS,
        sweep_depth_pts=3.75, direction="long")


def _agg(**extra):
    params = StrategyParams(use_level_quality_scoring=False,
                            use_confluence_scoring=False, **extra)
    agg = SignalAggregator(strategy_params=params, min_rr_ratio=0.1)
    agg.level_store = LevelStore()
    agg.level_store.add(Level(price=7535.0, level_type=LevelType.HORIZONTAL_SR,
                              created_at=_TS, confirmed_at=_TS, touch_count=3))
    agg.level_store.add(Level(price=7560.0, level_type=LevelType.HORIZONTAL_SR,
                              created_at=_TS, confirmed_at=_TS, touch_count=3))
    return agg


def _table(tmp_path):
    art = {
        "meta": {}, "cells": {}, 
        "parents": {"non_acceptance": {"n": 100, "wins": 62, "avg_pnl": 20.0}},
        "global": {"n": 300, "wins": 155, "avg_pnl": 12.0},
    }
    p = tmp_path / "table.json"
    p.write_text(json.dumps(art))
    return str(p)


def test_flag_off_no_predictions_same_sizing():
    off = _agg()._qualify_signal(_pattern(), SignalType.FAILED_BREAKDOWN)
    assert off is not None
    assert off.predicted_p_win is None


def test_flag_on_annotates_without_changing_size(tmp_path):
    path = _table(tmp_path)
    on = _agg(use_confidence_profile=True,
              confidence_profile_path=path)._qualify_signal(
        _pattern(), SignalType.FAILED_BREAKDOWN)
    off = _agg()._qualify_signal(_pattern(), SignalType.FAILED_BREAKDOWN)
    assert on is not None and off is not None
    # THE guarantee: identical trading decision
    assert on.position_size_factor == off.position_size_factor
    assert on.target_1 == off.target_1 and on.risk_pts == off.risk_pts
    # and the annotation exists
    assert on.predicted_p_win is not None
    assert 0.55 < on.predicted_p_win < 0.68
    assert on.confidence_n == 100


def test_missing_table_never_breaks_qualification(tmp_path):
    on = _agg(use_confidence_profile=True,
              confidence_profile_path=str(tmp_path / "nope.json"))
    sig = on._qualify_signal(_pattern(), SignalType.FAILED_BREAKDOWN)
    assert sig is not None
    assert sig.predicted_p_win is None
