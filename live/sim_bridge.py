"""SimBridge — the simulated broker behind the ReplayRunner.

Duck-types IBBridge exactly as IBRunner consumes it, so the REAL live runner
(gates, collection mode, single-position, runner reconcile) can run over a
recorded tape with zero strategy reimplementation.

FILL MODEL (one rule, no intrabar path modeling — documented divergences are
annotated by the fidelity harness):
- Entry fills at ``round_tick(signal_bar.close)``. The marketable-limit cap is
  respected: if the fill would be beyond ``entry_price ± slippage_cap_pts`` the
  entry does NOT fill and ``(None, 0.0)`` is returned (this feeds the live
  ``ib:entry_rejected_or_timeout`` phantom path).
- The bracket arms on the bar AFTER the entry bar (the venue acts after the
  signal bar closed).
- Per popped bar (long; shorts mirrored): ``sl_hit = low <= sl``;
  ``tp_hit = high >= tp`` (while the TP works). Both inside one bar -> SL
  fills first (conservative). Gap-through fills at the bar OPEN.
- A TP fill closes ``tp_qty`` (``runner_split``) and OCA-reduces the SL to the
  runner quantity; with no runner it is a full "TP" close.
- Bot-initiated exits (``flatten`` / ``partial_exit``) fill at the close of the
  last popped bar — the price the ExitManager decided on.

The bracket is evaluated inside ``get_new_bars()`` BEFORE the bar is handed to
the runner: the venue trades during the bar, before the bot sees it — matching
the live ordering that produced the venue-T1 reconcile path.

Clock: ``current_time()`` is the last popped bar's timestamp and ``poll_count``
increments on every ``get_new_bars()`` call (including empty drain calls) —
the ReplayRunner wires ``runner._now_fn``/``runner._mono_fn`` to these so
session dates, rollover, entry-grace and the 3x close-confirmation all follow
the tape.
"""
from __future__ import annotations

from datetime import time as dt_time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from live.ib_bridge import IBConfig, _round_tick, marketable_limit_price, runner_split

RTH_START = dt_time(9, 30)
RTH_END = dt_time(16, 0)


def _load_session(data_dir: Path, date_str: str) -> Optional[pd.DataFrame]:
    for sub in ("sessions_full", "sessions"):
        p = Path(data_dir) / sub / f"{date_str}_bars.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                return df
            except Exception:
                continue
    return None


class SimBridge:
    """Simulated IBBridge over a recorded tape."""

    def __init__(self, session_date: str, tape: Optional[pd.DataFrame] = None,
                 data_dir: Optional[str] = None, config: Optional[IBConfig] = None,
                 history: Optional[dict] = None, bars_per_call: int = 1,
                 drain_iterations: int = 5,
                 on_tape_exhausted: Optional[Callable] = None,
                 history_lookback: int = 40):
        self.session_date = str(session_date)
        self.config = config or IBConfig()
        self.bars_per_call = bars_per_call
        self.drain_iterations = drain_iterations
        self.on_tape_exhausted = on_tape_exhausted
        self._exhaust_fired = False
        self._drains_done = 0

        if tape is None:
            if data_dir is None:
                raise ValueError("SimBridge needs a tape or a data_dir")
            tape = _load_session(Path(data_dir), self.session_date)
            if tape is None:
                raise FileNotFoundError(f"no tape for {self.session_date}")
        self.tape = tape
        self._cursor = 0

        # history: {date_str: DataFrame} of PRIOR sessions for
        # get_prior_day_bars / get_bars / get_daily_bars
        if history is not None:
            self.history = dict(history)
        elif data_dir is not None:
            self.history = self._load_history(Path(data_dir), history_lookback)
        else:
            self.history = {}

        # live-compat attributes
        self._connected = True
        self._needs_reconnect = False
        self._active_orders: dict = {}

        # broker state
        self.poll_count = 0
        self._last_bar: Optional[pd.Series] = None
        self._last_ts = tape.index[0] if len(tape) else None
        self._next_order_id = 1000
        self._position: Optional[dict] = None
        self._bracket: Optional[dict] = None
        self._last_fill: dict = {}     # trade_id -> (price, "TP"/"SL"/"unknown")

    # ------------------------------------------------------------------
    # lifecycle / health (all trivially healthy in replay)
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def start_streaming(self) -> bool:
        return True

    def stop_streaming(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return True

    def ping(self, timeout: float = 3.0) -> bool:
        return True

    def sleep(self, seconds: float) -> None:  # no real time passes in replay
        pass

    def check_reconnect(self) -> bool:
        return False

    def resubscribe(self, reason: str = "") -> None:
        pass

    def force_reconnect(self, reason: str = "") -> None:
        pass

    def seconds_since_reconnect(self) -> float:
        return 1e9  # far past every post-reconnect grace window

    def connectivity_down(self) -> bool:
        return False

    def seconds_since_connectivity_restored(self) -> float:
        return 1e9

    def get_account_info(self) -> Optional[dict]:
        return None

    # ------------------------------------------------------------------
    # clock (consumed via IBRunner._now_fn / _mono_fn)
    # ------------------------------------------------------------------
    def current_time(self):
        ts = self._last_ts
        return ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

    # ------------------------------------------------------------------
    # bars
    # ------------------------------------------------------------------
    def get_new_bars(self) -> list:
        self.poll_count += 1
        if self._cursor >= len(self.tape):
            if self._drains_done < self.drain_iterations:
                self._drains_done += 1
            elif not self._exhaust_fired:
                self._exhaust_fired = True
                if self.on_tape_exhausted:
                    self.on_tape_exhausted()
            return []
        out = []
        for _ in range(self.bars_per_call):
            if self._cursor >= len(self.tape):
                break
            ts = self.tape.index[self._cursor]
            row = self.tape.iloc[self._cursor]
            self._cursor += 1
            self._last_bar = row
            self._last_ts = ts
            # venue acts on the bar before the (delayed) bot sees it
            self._evaluate_bracket(row)
            out.append({
                "timestamp": ts.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            })
        return out

    def get_latest_bar(self) -> Optional[dict]:
        bars = self.get_new_bars()
        return bars[-1] if bars else None

    # ------------------------------------------------------------------
    # orders / position
    # ------------------------------------------------------------------
    def send_entry(self, quantity: int, sl: float, tp: float,
                   direction: str = "long", comment: str = "",
                   fill_timeout_sec: float = 30.0, tp_fraction: float = 0.75,
                   entry_price: float = 0.0, slippage_cap_pts: float = 0.0):
        if quantity <= 0 or self._last_bar is None:
            return None, 0.0
        fill = _round_tick(float(self._last_bar["close"]))
        if entry_price and slippage_cap_pts and slippage_cap_pts > 0:
            limit = marketable_limit_price(direction, float(entry_price),
                                           float(slippage_cap_pts))
            if (direction == "long" and fill > limit) or \
               (direction == "short" and fill < limit):
                return None, 0.0  # would be chasing — no fill (live no-chase)
        oid = self._next_order_id
        self._next_order_id += 1
        tp_qty, runner_qty = runner_split(int(quantity), float(tp_fraction))
        self._position = {
            "ticket": oid, "volume": int(quantity), "price_open": fill,
            "sl": float(sl), "tp": float(tp), "profit": 0.0,
            "time": self.current_time(),
            "market_position": 1 if direction == "long" else -1,
            "direction": direction,
        }
        self._bracket = {
            "trade_id": oid, "direction": direction,
            "sl": float(sl), "sl_qty": int(quantity),
            "tp": float(tp), "tp_qty": int(tp_qty),
            "runner_qty": int(runner_qty),
            "sl_order_id": oid + 1, "tp_order_id": oid + 2,
            "armed": False,  # arms on the NEXT popped bar
        }
        self._active_orders[oid] = dict(self._bracket)
        return oid, fill

    def _evaluate_bracket(self, bar: pd.Series) -> None:
        br = self._bracket
        if br is None or self._position is None:
            return
        if not br["armed"]:
            br["armed"] = True   # first bar after entry: arm, then evaluate
        o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])
        long = br["direction"] == "long"
        sl, tp = br["sl"], br.get("tp")
        sl_hit = (l <= sl) if long else (h >= sl)
        tp_hit = tp is not None and ((h >= tp) if long else (l <= tp))
        if sl_hit:  # SL first when both hit inside one bar (conservative)
            price = o if ((o < sl) if long else (o > sl)) else sl
            self._close_all(price, "SL")
            return
        if tp_hit:
            price = o if ((o > tp) if long else (o < tp)) else tp
            if br["runner_qty"] > 0:
                # OCA reduce: TP fraction closes, SL shrinks to the runner
                self._position["volume"] = br["runner_qty"]
                br["sl_qty"] = br["runner_qty"]
                br["tp"] = None
                self._last_fill[br["trade_id"]] = (price, "TP")
            else:
                self._close_all(price, "TP")

    def _close_all(self, price: float, typ: str) -> None:
        if self._bracket is not None:
            self._last_fill[self._bracket["trade_id"]] = (price, typ)
        self._position = None
        self._bracket = None

    def update_stop(self, trade_id: int, new_sl: float, reason: str = "") -> bool:
        if self._position is None or self._bracket is None:
            return False
        self._bracket["sl"] = float(new_sl)
        self._position["sl"] = float(new_sl)
        return True

    def partial_exit(self, trade_id: int, quantity: int, new_sl: float,
                     reason: str = "") -> bool:
        if self._position is None:
            return False
        qty = min(int(quantity), int(self._position["volume"]))
        self._position["volume"] -= qty
        if self._position["volume"] <= 0:
            price = _round_tick(float(self._last_bar["close"]))
            self._close_all(price, "unknown")
            return True
        # live behavior: cancel BOTH children, install SL-only for the remainder
        self._bracket["tp"] = None
        self._bracket["sl"] = float(new_sl)
        self._bracket["sl_qty"] = int(self._position["volume"])
        self._position["sl"] = float(new_sl)
        return True

    def flatten(self, reason: str = "") -> bool:
        if self._position is None:
            return True  # already flat — matches live
        price = _round_tick(float(self._last_bar["close"]))
        self._close_all(price, "unknown")
        return True

    def get_position(self) -> Optional[dict]:
        return dict(self._position) if self._position is not None else None

    def get_bracket_fill_price(self, trade_id: int):
        return self._last_fill.get(trade_id, (0.0, "unknown"))

    def get_bracket_orders(self) -> dict:
        if self._bracket is None:
            return {}
        return {"sl": self._bracket["sl"], "tp": self._bracket.get("tp"),
                "sl_order_id": self._bracket["sl_order_id"],
                "tp_order_id": self._bracket["tp_order_id"]}

    # ------------------------------------------------------------------
    # history served to _initialize_session
    # ------------------------------------------------------------------
    def _load_history(self, data_dir: Path, lookback: int) -> dict:
        dates = set()
        for sub in ("sessions_full", "sessions"):
            base = data_dir / sub
            if base.exists():
                dates.update(p.name.replace("_bars.parquet", "")
                             for p in base.glob("*_bars.parquet"))
        prior = sorted(d for d in dates if d < self.session_date)[-lookback:]
        out = {}
        for d in prior:
            df = _load_session(data_dir, d)
            if df is not None and len(df):
                out[d] = df
        return out

    def _rth(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = [(RTH_START <= t.time() < RTH_END) for t in df.index]
        return df[mask]

    def get_prior_day_bars(self) -> Optional[pd.DataFrame]:
        prior = sorted(d for d in self.history if d < self.session_date)
        if not prior:
            return None
        df = self._rth(self.history[prior[-1]])
        return df if len(df) else None

    def get_bars(self, count: int = 400) -> Optional[pd.DataFrame]:
        frames = [self.history[d] for d in sorted(self.history)
                  if d < self.session_date]
        if not frames:
            return None
        hist = pd.concat(frames)
        start = self.tape.index[0] if len(self.tape) else None
        if start is not None:
            hist = hist[hist.index < start]
        return hist.tail(count) if len(hist) else None

    def get_daily_bars(self, days: int = 365) -> Optional[pd.DataFrame]:
        rows, idx = [], []
        for d in sorted(self.history):
            rth = self._rth(self.history[d])
            if not len(rth):
                continue
            rows.append({"open": float(rth["open"].iloc[0]),
                         "high": float(rth["high"].max()),
                         "low": float(rth["low"].min()),
                         "close": float(rth["close"].iloc[-1]),
                         "volume": float(rth.get("volume", pd.Series([0])).sum())})
            idx.append(pd.Timestamp(d))
        if not rows:
            return None
        return pd.DataFrame(rows, index=pd.DatetimeIndex(idx)).tail(days)
