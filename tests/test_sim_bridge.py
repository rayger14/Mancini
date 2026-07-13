"""SimBridge: the simulated broker behind the ReplayRunner. Duck-types
IBBridge exactly as IBRunner consumes it; fills brackets against tape bars.

Fill model under test (documented in sim_bridge.py):
- entry fills at round_tick(signal bar close); marketable-limit respected
- bracket evaluates per popped bar, arming AFTER the entry bar
- both SL+TP inside one bar -> SL first (conservative); gap-through at open
- TP fill closes tp_qty (runner_split), OCA-reduces SL to the runner
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz

from live.sim_bridge import SimBridge

_ET = pytz.timezone("US/Eastern")
_T0 = _ET.localize(datetime(2026, 7, 2, 9, 30))


def _tape(rows):
    """rows: list of (open, high, low, close) -> 1-min tape DataFrame."""
    idx = [_T0 + timedelta(minutes=i) for i in range(len(rows))]
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [100.0] * len(rows)},
        index=pd.DatetimeIndex(idx))


def _bridge(rows, **kw):
    return SimBridge(session_date="2026-07-02", tape=_tape(rows), **kw)


def _pop(b):
    bars = b.get_new_bars()
    return bars[0] if bars else None


# ---------- lifecycle / health ----------

def test_health_surface():
    b = _bridge([(7500, 7501, 7499, 7500)])
    assert b.connect() is True
    assert b.is_connected is True
    assert b.ping() is True
    assert b.seconds_since_reconnect() >= 60.0
    assert b.connectivity_down() is False
    assert b.get_account_info() is None
    b.sleep(99)  # no-op, returns instantly


def test_bar_shape_and_clock():
    b = _bridge([(7500, 7502, 7499, 7501), (7501, 7503, 7500, 7502)])
    bar = _pop(b)
    assert set(bar) == {"timestamp", "open", "high", "low", "close", "volume"}
    assert bar["close"] == 7501
    assert b.poll_count == 1
    assert b.current_time().minute == 30
    _pop(b)
    assert b.poll_count == 2
    assert b.current_time().minute == 31


def test_drain_and_exhaustion_callback():
    fired = []
    b = _bridge([(7500, 7501, 7499, 7500)], drain_iterations=2,
                on_tape_exhausted=lambda: fired.append(1))
    _pop(b)                    # the only bar
    assert b.get_new_bars() == []   # drain 1
    assert b.get_new_bars() == []   # drain 2
    assert fired == []
    assert b.get_new_bars() == []   # exhausted -> callback fires once
    assert fired == [1]
    b.get_new_bars()
    assert fired == [1]
    # poll_count kept advancing through the drain (feeds the replay mono clock)
    assert b.poll_count >= 5


# ---------- entry fills ----------

def test_entry_fills_at_signal_close():
    b = _bridge([(7500, 7502, 7499, 7501.13)])
    _pop(b)
    oid, fill = b.send_entry(quantity=4, sl=7490.0, tp=7520.0, direction="long",
                             entry_price=7501.0, slippage_cap_pts=5.0)
    assert oid is not None
    assert fill == 7501.25  # round_tick of 7501.13
    pos = b.get_position()
    assert pos is not None and pos["volume"] == 4


def test_marketable_limit_rejects_chase():
    # signal price 7500 with 2pt cap, but tape already at 7510 -> no fill
    b = _bridge([(7510, 7511, 7509, 7510)])
    _pop(b)
    oid, fill = b.send_entry(quantity=2, sl=7490.0, tp=7520.0, direction="long",
                             entry_price=7500.0, slippage_cap_pts=2.0)
    assert oid is None and fill == 0.0
    assert b.get_position() is None


# ---------- bracket fills ----------

def _entered(rows, qty=4, sl=7490.0, tp=7520.0, tp_fraction=0.75):
    b = _bridge(rows)
    _pop(b)  # entry signal bar
    oid, fill = b.send_entry(quantity=qty, sl=sl, tp=tp, direction="long",
                             tp_fraction=tp_fraction)
    assert oid is not None
    return b, oid


def test_tp_fill_reduces_to_runner():
    rows = [(7500, 7501, 7499, 7500),      # entry bar
            (7500, 7521, 7500, 7519)]      # TP 7520 tagged
    b, oid = _entered(rows)
    _pop(b)                                 # venue acts on this bar
    pos = b.get_position()
    assert pos is not None and pos["volume"] == 1          # runner held
    price, typ = b.get_bracket_fill_price(oid)
    assert typ == "TP" and price == 7520.0
    orders = b.get_bracket_orders()
    assert orders and orders["sl"] == 7490.0               # SL reduced, working


def test_full_close_when_no_runner():
    rows = [(7500, 7501, 7499, 7500), (7500, 7521, 7500, 7519)]
    b, oid = _entered(rows, qty=1)          # 1 lot -> no runner
    _pop(b)
    assert b.get_position() is None
    assert b.get_bracket_fill_price(oid) == (7520.0, "TP")
    assert b.get_bracket_orders() == {}


def test_sl_first_when_both_hit_same_bar():
    rows = [(7500, 7501, 7499, 7500), (7500, 7525, 7488, 7510)]  # both inside
    b, oid = _entered(rows)
    _pop(b)
    assert b.get_position() is None
    price, typ = b.get_bracket_fill_price(oid)
    assert typ == "SL" and price == 7490.0


def test_gap_through_fills_at_open():
    rows = [(7500, 7501, 7499, 7500), (7484, 7486, 7482, 7485)]  # gaps under SL
    b, oid = _entered(rows)
    _pop(b)
    price, typ = b.get_bracket_fill_price(oid)
    assert typ == "SL" and price == 7484  # open, not the SL price


def test_bracket_does_not_fire_on_entry_bar():
    # entry bar itself tags the TP; bracket must arm only from the NEXT bar
    rows = [(7500, 7522, 7499, 7500), (7500, 7501, 7499, 7500)]
    b, oid = _entered(rows)
    assert b.get_position()["volume"] == 4
    _pop(b)                                 # quiet bar -> still open
    assert b.get_position()["volume"] == 4


# ---------- bot-initiated orders ----------

def test_update_stop_moves_sl_and_fails_after_close():
    rows = [(7500, 7501, 7499, 7500), (7500, 7501, 7499, 7500),
            (7500, 7501, 7493, 7494)]
    b, oid = _entered(rows)
    assert b.update_stop(trade_id=oid, new_sl=7495.0) is True
    _pop(b); _pop(b)                        # bar 3 low 7493 <= new stop 7495
    assert b.get_position() is None
    assert b.get_bracket_fill_price(oid) == (7495.0, "SL")
    assert b.update_stop(trade_id=oid, new_sl=7480.0) is False


def test_partial_exit_cancels_tp_installs_sl_only():
    rows = [(7500, 7501, 7499, 7500), (7500, 7505, 7499, 7504)]
    b, oid = _entered(rows)
    _pop(b)
    assert b.partial_exit(trade_id=oid, quantity=3, new_sl=7497.0) is True
    pos = b.get_position()
    assert pos["volume"] == 1
    orders = b.get_bracket_orders()
    assert orders["sl"] == 7497.0 and orders.get("tp") is None


def test_flatten_true_when_flat_and_closes_at_last_close():
    rows = [(7500, 7501, 7499, 7500), (7500, 7509, 7499, 7508)]
    b, oid = _entered(rows)
    assert b.flatten("test") is True        # closes at last close 7500
    assert b.get_position() is None
    b2 = _bridge(rows)
    assert b2.flatten("already flat") is True


# ---------- history for _initialize_session ----------

def _history_frames():
    prior = _tape([(7480 + i, 7482 + i, 7478 + i, 7481 + i) for i in range(30)])
    prior.index = prior.index - pd.Timedelta(days=1)
    return {"2026-07-01": prior}


def test_prior_day_bars_rth_slice():
    b = SimBridge(session_date="2026-07-02",
                  tape=_tape([(7500, 7501, 7499, 7500)]),
                  history=_history_frames())
    pd_bars = b.get_prior_day_bars()
    assert pd_bars is not None and len(pd_bars) > 0
    times = pd_bars.index.time
    import datetime as _dt
    assert min(times) >= _dt.time(9, 30) and max(times) < _dt.time(16, 0)


def test_utc_indexed_archive_normalized_to_et():
    """Some archive parquets are UTC-indexed (partial refetches); every frame
    must be normalized to ET or session gates / RTH slices read wrong times."""
    from live.sim_bridge import _normalize_et
    utc = pytz.UTC
    idx = [utc.localize(datetime(2026, 7, 1, 14, 30)),  # = 10:30 ET
           utc.localize(datetime(2026, 7, 1, 15, 30))]
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2],
                       "close": [1, 2], "volume": [0, 0]},
                      index=pd.DatetimeIndex(idx))
    out = _normalize_et(df)
    assert str(out.index.tz) in ("US/Eastern", "America/New_York")
    assert out.index[0].hour == 10 and out.index[0].minute == 30


def test_daily_bars_synthesized():
    b = SimBridge(session_date="2026-07-02",
                  tape=_tape([(7500, 7501, 7499, 7500)]),
                  history=_history_frames())
    daily = b.get_daily_bars(days=365)
    assert daily is not None and len(daily) >= 1
    assert {"open", "high", "low", "close"} <= set(daily.columns)
