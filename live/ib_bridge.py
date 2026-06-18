"""Interactive Brokers bridge for the Mancini strategy engine.

Wraps ib_insync (Python 3.9) or ib_async (Python 3.11+) to provide
bar data, bracket order execution, and position management for MES futures.

Key design:
- Bracket orders (parent + SL + TP) are OCO at the exchange, so the position
  is protected even if Python crashes.
- All times are converted to US/Eastern for strategy compatibility.
- Reconnection logic handles TWS disconnects gracefully.

Usage:
    bridge = IBBridge(IBConfig(port=7497))  # paper trading
    bridge.connect()
    bars = bridge.get_bars(count=400)
    order_id, fill_price = bridge.send_entry(quantity=4, sl=6041.50, tp=6052.00)
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional


def _round_tick(price: float, tick: float = 0.25) -> float:
    """Round a price to the nearest valid tick increment (MES = 0.25)."""
    return round(price / tick) * tick

import pandas as pd
from loguru import logger

# ── ib_insync / ib_async import (version-aware) ────────────────────────
# ib_insync works on Python 3.9 (system python), ib_async needs 3.11+.
# Both expose the same API surface — try ib_async first, fall back to ib_insync.

try:
    from ib_async import (
        IB, Future, MarketOrder, LimitOrder, StopOrder,
        Contract, util,
    )
    _IB_PACKAGE = "ib_async"
except ImportError:
    from ib_insync import (  # type: ignore[no-redef]
        IB, Future, MarketOrder, LimitOrder, StopOrder,
        Contract, util,
    )
    _IB_PACKAGE = "ib_insync"


def marketable_limit_price(direction: str, entry_price: float,
                           cap_pts: float) -> float:
    """Limit price for a *marketable* entry.

    Prices the order ``cap_pts`` through the signal in the adverse direction so
    it crosses the spread and fills immediately at the live market (or better),
    but never worse than ``cap_pts`` of adverse slippage:
    - long  → willing to pay up to entry + cap
    - short → willing to sell down to entry - cap

    On the 12-min delayed feed this fills like a market order whenever the real
    price is within ``cap_pts`` of the signal, and skips (no fill) only when
    price has already run further than that — i.e. when we'd be chasing.
    """
    if direction == "short":
        return entry_price - cap_pts
    return entry_price + cap_pts


def runner_split(quantity: int, tp_fraction: float) -> tuple[int, int]:
    """Split a position into (tp_quantity, runner_quantity).

    floor (not round) mirrors ExitManager._check_t1 and avoids banker's
    rounding turning 2*0.75=1.5 into 2. 1 contract leaves no runner.
    """
    import math as _math
    if quantity <= 1:
        return quantity, 0
    tp_quantity = min(quantity - 1, max(1, _math.floor(quantity * tp_fraction)))
    return tp_quantity, quantity - tp_quantity


def build_bracket_orders(*, parent_id: int, tp_id: int, sl_id: int,
                         quantity: int, tp_fraction: float,
                         entry_limit_price: float, sl: float, tp: float,
                         direction: str, comment: str):
    """Construct (parent, take_profit, stop_loss) for a marketable-limit entry
    with an OCA reduce-on-fill exit pair.

    The parent is a marketable LIMIT (bounded slippage) when ``entry_limit_price``
    is given, else a MARKET order (legacy). The TP (a fraction of the position)
    and the SL (full) share one OCA group with ``ocaType=2`` (reduce-with-block),
    so when the venue fills the TP fraction in real time, IB auto-reduces the SL
    to the runner quantity — the runner rests at the exchange, protected, immune
    to the bot's delayed feed.
    """
    entry_action, exit_action = (
        ("SELL", "BUY") if direction == "short" else ("BUY", "SELL")
    )

    tp_quantity, _runner = runner_split(quantity, tp_fraction)
    oca_group = f"oca_{parent_id}"

    if entry_limit_price is None:
        parent = MarketOrder(
            action=entry_action,
            totalQuantity=quantity,
            orderId=parent_id,
            transmit=False,
            orderRef=comment,
            tif="GTC",
        )
    else:
        parent = LimitOrder(
            action=entry_action,
            totalQuantity=quantity,
            lmtPrice=_round_tick(entry_limit_price),
            orderId=parent_id,
            transmit=False,
            orderRef=comment,
            tif="GTC",
        )
    take_profit = LimitOrder(
        action=exit_action,
        totalQuantity=tp_quantity,
        lmtPrice=_round_tick(tp),
        orderId=tp_id,
        parentId=parent_id,
        ocaGroup=oca_group,
        ocaType=2,  # reduce remaining group size on (partial) fill
        transmit=False,
        orderRef=f"{comment}:TP",
        tif="GTC",
    )
    stop_loss = StopOrder(
        action=exit_action,
        totalQuantity=quantity,
        stopPrice=_round_tick(sl),
        orderId=sl_id,
        parentId=parent_id,
        ocaGroup=oca_group,
        ocaType=2,
        transmit=True,  # transmits the whole bracket
        orderRef=f"{comment}:SL",
        tif="GTC",
    )
    return parent, take_profit, stop_loss


@dataclass
class IBConfig:
    """Configuration for the IB bridge."""

    host: str = "127.0.0.1"
    port: int = 7497           # 7497 = paper, 7496 = live
    client_id: int = 1
    # MES contract
    symbol: str = "MES"
    exchange: str = "CME"
    currency: str = "USD"
    sec_type: str = "FUT"
    # Auto-detect front month if empty; otherwise YYYYMM or YYYYMMDD
    contract_month: str = ""
    # Polling
    poll_interval_sec: float = 0.5
    # Reconnection
    max_reconnect_attempts: int = 5
    reconnect_delay_sec: float = 5.0
    # Timeout for historical data requests
    hist_timeout_sec: float = 60.0
    # Use extended hours (globex) data
    use_rth_only: bool = True


class IBBridge:
    """Communication layer with Interactive Brokers via TWS / IB Gateway.

    Provides bar data, bracket order execution, and position tracking.
    All entry orders are sent as bracket orders (parent + SL + TP) so
    the exchange protects the position even if Python dies.
    """

    def __init__(self, config: IBConfig = IBConfig()):
        self.config = config
        self._ib = IB()
        self._contract: Optional[Contract] = None
        self._connected: bool = False
        self._last_bar_time: Optional[pd.Timestamp] = None
        self._stale_count: int = 0
        self._zero_volume_count: int = 0  # consecutive zero-volume bars → expired contract
        # Track our bracket orders: parent_order_id -> {parent, tp, sl}
        self._active_orders: dict[int, dict] = {}
        # Streaming bars state
        self._streaming_bars: list = []
        self._bar_callback = None
        self._streaming_active: bool = False
        self._use_polling: bool = False
        self._poll_interval: float = 60.0
        self._last_poll_time: float = 0.0
        self._needs_reconnect: bool = False
        self._reconnect_backoff_until: float = 0.0
        self._reconnect_exhausted_logged: bool = False

    # ── Connection ────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to TWS / IB Gateway and qualify the MES contract.

        Returns True if connected and contract qualified successfully.
        """
        try:
            self._ib.connect(
                self.config.host,
                self.config.port,
                clientId=self.config.client_id,
                timeout=20,
            )
        except Exception as e:
            logger.error(f"IB connect failed: {e}")
            return False

        # Register disconnect handler for auto-reconnect
        self._ib.disconnectedEvent += self._on_disconnect

        # Qualify the MES contract
        self._contract = self._qualify_contract()
        if self._contract is None:
            logger.error("Failed to qualify MES contract")
            self._ib.disconnect()
            return False

        self._connected = True
        accts = self._ib.managedAccounts()
        logger.info(f"IB connected via {_IB_PACKAGE}: "
                     f"accounts={accts}, contract={self._contract.localSymbol}")
        return True

    def disconnect(self) -> None:
        """Disconnect from IB."""
        self._connected = False  # Set BEFORE disconnect to prevent reconnect handler
        self.stop_streaming()
        if self._ib.isConnected():
            self._ib.disconnect()
        logger.info("IB disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ib.isConnected()

    @staticmethod
    def _select_front_contract(candidates, today=None, roll_within_days: int = 5):
        """Choose the *liquid* front-month from expiry-sorted candidates.

        Parameters
        ----------
        candidates : list[dict]
            Each ``{"contract": <Contract>, "expiry": "YYYYMMDD",
            "volume": float | None}``, pre-filtered to non-expired and
            sorted by expiry ascending.
        today : date, optional
            Reference date for the calendar fallback (defaults to today).
        roll_within_days : int
            When volume is unavailable, roll off a front contract that
            expires within this many days.

        Selection order:
        1. **Roll on volume** (preferred): if the next contract's recent
           volume exceeds the front's, the market has rolled — pick next.
           This tracks the liquidity migration ~a week before expiry
           instead of clinging to the expiring contract until it dies.
        2. **Calendar safety** (when volume is unavailable): if the front
           contract expires within ``roll_within_days``, roll to the next
           contract rather than trade a dying contract.
        3. Otherwise: nearest expiry.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]["contract"]
        front, nxt = candidates[0], candidates[1]
        fv, nv = front.get("volume"), nxt.get("volume")
        if fv is not None and nv is not None:
            return (nxt if nv > fv else front)["contract"]
        # Volume unavailable on at least one leg — calendar-based safety roll.
        if today is None:
            today = date.today()
        try:
            fexp = datetime.strptime(str(front["expiry"])[:8], "%Y%m%d").date()
            if (fexp - today).days <= roll_within_days:
                return nxt["contract"]
        except (ValueError, KeyError, TypeError):
            pass
        return front["contract"]

    def _recent_volume(self, contract) -> Optional[float]:
        """Best-effort recent daily volume for a contract; None on failure.

        Used only to compare liquidity between adjacent expiries at
        qualify time. Any IB error returns None so the caller falls back
        to the calendar rule — never blocks contract qualification.
        """
        try:
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )
            if bars:
                vol = getattr(bars[-1], "volume", None)
                return float(vol) if vol is not None and vol >= 0 else None
        except Exception as e:
            logger.debug(
                f"Volume fetch failed for "
                f"{getattr(contract, 'localSymbol', '?')}: {e}"
            )
        return None

    def _qualify_contract(self) -> Optional[Contract]:
        """Build and qualify the MES futures contract.

        If contract_month is empty, auto-detects the front month.
        """
        if self.config.contract_month:
            contract = Future(
                symbol=self.config.symbol,
                lastTradeDateOrContractMonth=self.config.contract_month,
                exchange=self.config.exchange,
                currency=self.config.currency,
            )
            try:
                qualified = self._ib.qualifyContracts(contract)
                if qualified:
                    c = qualified[0]
                    logger.info(f"Qualified contract: {c.localSymbol} "
                                 f"(conId={c.conId}, expiry={c.lastTradeDateOrContractMonth})")
                    return c
            except Exception as e:
                logger.error(f"Contract qualification failed: {e}")
                return None
        else:
            # Auto-detect front month: request all MES contracts, pick nearest
            # ACTIVE expiry (skip expired contracts to avoid dead-contract blindness).
            contract = Future(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                currency=self.config.currency,
            )
            try:
                details = self._ib.reqContractDetails(contract)
                if details:
                    # Sort by expiry ascending
                    details.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
                    today_str = date.today().strftime("%Y%m%d")
                    # Filter to contracts expiring today or later
                    active = [d for d in details
                              if d.contract.lastTradeDateOrContractMonth >= today_str]
                    if not active:
                        # All contracts appear expired — fall back to latest expiry
                        logger.warning("All MES contracts appear expired, using latest expiry")
                        active = details[-1:]
                    # Roll on volume: compare the nearest two expiries by
                    # recent liquidity so we track the liquid front month
                    # through roll week instead of clinging to the expiring
                    # contract until it goes dead.
                    candidates = [
                        {
                            "contract": d.contract,
                            "expiry": d.contract.lastTradeDateOrContractMonth,
                            "volume": self._recent_volume(d.contract),
                        }
                        for d in active[:2]
                    ]
                    front = self._select_front_contract(candidates)
                    if front is None:
                        front = active[0].contract
                    if len(candidates) > 1:
                        logger.info(
                            "Front-month liquidity: "
                            + ", ".join(
                                f"{c['contract'].localSymbol}="
                                f"{int(c['volume']) if c['volume'] is not None else 'n/a'}"
                                for c in candidates
                            )
                        )
                    qualified = self._ib.qualifyContracts(front)
                    if qualified:
                        c = qualified[0]
                        logger.info(f"Qualified front-month: {c.localSymbol} "
                                     f"(conId={c.conId}, expiry={c.lastTradeDateOrContractMonth})")
                        return c
            except Exception as e:
                logger.error(f"Front-month detection failed: {e}")
                return None

        try:
            qualified = self._ib.qualifyContracts(contract)
            if qualified:
                c = qualified[0]
                logger.info(f"Qualified contract: {c.localSymbol} "
                             f"(conId={c.conId}, expiry={c.lastTradeDateOrContractMonth})")
                return c
            else:
                logger.error("qualifyContracts returned empty list")
                return None
        except Exception as e:
            logger.error(f"Contract qualification failed: {e}")
            return None

    def _attempt_contract_reroll(self, reason: str) -> None:
        """Re-qualify the front-month contract after detecting stale/expired data.

        This handles the scenario where a contract expires mid-session (e.g.,
        quarterly rollover) and the bot keeps polling the dead contract.
        """
        old_symbol = getattr(self._contract, "localSymbol", "unknown")
        logger.warning(
            f"CONTRACT REROLL triggered: {reason}. "
            f"Current contract: {old_symbol}. Re-qualifying..."
        )
        # Force auto-detect by temporarily clearing contract_month
        saved_month = self.config.contract_month
        self.config.contract_month = ""
        new_contract = self._qualify_contract()
        self.config.contract_month = saved_month

        if new_contract and new_contract.conId != (self._contract.conId if self._contract else 0):
            logger.warning(
                f"CONTRACT ROLLED: {old_symbol} → {new_contract.localSymbol} "
                f"(conId {new_contract.conId})"
            )
            self._contract = new_contract
            self._last_bar_time = None  # Reset dedup so new bars come through
            self._zero_volume_count = 0
            self._stale_count = 0
        else:
            logger.error(
                f"Contract reroll failed — still on {old_symbol}. "
                f"May need manual intervention."
            )
            # Reset counter to avoid spamming reroll attempts every poll
            self._zero_volume_count = 0

    def _on_disconnect(self) -> None:
        """Handle unexpected disconnection — flag for reconnect.

        This is a callback from the ib_async event loop. We must NOT call
        self._ib.sleep() or self._ib.connect() here because the event loop
        is already running, which would raise "event loop already running".
        Instead, set a flag and let the main polling loop call check_reconnect().
        """
        if not self._connected:
            return  # Intentional disconnect

        logger.error("IB CONNECTION LOST — flagging for reconnect")
        self._connected = False
        self._needs_reconnect = True

    def check_reconnect(self) -> bool:
        """Attempt reconnection if flagged by _on_disconnect.

        Called from the main polling loop in ib_runner.py, outside the
        ib_async event loop callback context, so it is safe to call
        self._ib.sleep() and self._ib.connect() here.

        Returns True if reconnected successfully, False otherwise.
        """
        if not self._needs_reconnect:
            return True  # Nothing to do

        # Between failed bursts, wait out the backoff instead of hammering
        # a gateway that is mid-restart (its daily 19:45 ET restart takes
        # minutes; full auth can take longer).
        if _time.monotonic() < getattr(self, "_reconnect_backoff_until", 0.0):
            return False

        logger.info("Attempting IB reconnect...")

        for attempt in range(1, self.config.max_reconnect_attempts + 1):
            try:
                _time.sleep(self.config.reconnect_delay_sec)
                self._ib.connect(
                    self.config.host,
                    self.config.port,
                    clientId=self.config.client_id,
                    timeout=20,
                )
                if self._ib.isConnected():
                    # Re-qualify contract
                    self._contract = self._qualify_contract()
                    self._connected = True
                    self._needs_reconnect = False
                    self._reconnect_exhausted_logged = False
                    logger.info(f"Reconnected on attempt {attempt}")
                    # Restart streaming if it was active before disconnect
                    if self._streaming_active:
                        self.stop_streaming()
                        self.start_streaming()
                    return True
            except Exception as e:
                logger.warning(f"Reconnect attempt {attempt} failed: {e}")

        # Burst exhausted — NEVER clear the flag. The bot stayed blind for
        # hours on 2026-06-09 and 06-10 because exhaustion stopped all
        # retries while the gateway's nightly restart was still completing.
        # Keep retrying in 60s-backed-off bursts until the gateway returns.
        self._reconnect_backoff_until = _time.monotonic() + 60.0
        if not getattr(self, "_reconnect_exhausted_logged", False):
            logger.error(
                "ALL RECONNECT ATTEMPTS EXHAUSTED — bot is blind; will keep "
                "retrying every 60s (gateway daily restart can take minutes)"
            )
            self._reconnect_exhausted_logged = True
        else:
            logger.warning("Reconnect burst failed — next burst in 60s")
        return False

    # ── Bar Data ──────────────────────────────────────────────────────

    def get_bars(self, count: int = 400) -> Optional[pd.DataFrame]:
        """Get the last `count` 1-minute bars as a DataFrame.

        Returns DataFrame with columns: open, high, low, close, volume
        and a DatetimeIndex in US/Eastern.
        """
        if not self.is_connected or self._contract is None:
            return None

        # IB limits 1-min bars to ~1-2 days per request; 400 bars is < 7 hours
        duration = f"{count * 60} S"  # seconds
        try:
            bars = self._ib.reqHistoricalData(
                self._contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=self.config.use_rth_only,
                formatDate=2,  # UTC datetime objects
                timeout=self.config.hist_timeout_sec,
            )
        except Exception as e:
            logger.error(f"reqHistoricalData failed: {e}")
            return None

        if not bars:
            return None

        df = util.df(bars)
        if df is None or df.empty:
            return None

        # Rename and index
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df.index = df.index.tz_convert("US/Eastern")

        # Keep only OHLCV
        cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]
        df = df[available]
        return df

    def start_streaming(self) -> bool:
        """Start streaming 1-minute bars.

        Tries keepUpToDate first (real-time). If that fails (no market data
        subscription), falls back to polling historical data every 60 seconds.
        """
        if not self.is_connected or self._contract is None:
            return False

        if self._streaming_active:
            return True

        # Try real-time streaming first
        try:
            self._streaming_bars = self._ib.reqHistoricalData(
                self._contract,
                endDateTime="",
                durationStr="900 S",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=self.config.use_rth_only,
                formatDate=2,
                keepUpToDate=True,
                timeout=self.config.hist_timeout_sec,
            )
            n = len(self._streaming_bars) if self._streaming_bars else 0
            if n > 0:
                # Verify streaming works: check if real-time data is available
                try:
                    ticker = self._ib.reqMktData(self._contract, "", False, False)
                    self._ib.sleep(2)
                    has_realtime = not (ticker.last != ticker.last)  # NaN check
                    self._ib.cancelMktData(self._contract)
                except Exception:
                    has_realtime = False

                if has_realtime:
                    self._streaming_active = True
                    self._use_polling = False
                    logger.info(f"Streaming 1-min bars (real-time keepUpToDate), seed={n} bars")
                    return True
                else:
                    # Cancel the keepUpToDate request — it won't update
                    try:
                        self._ib.cancelHistoricalData(self._streaming_bars)
                    except Exception:
                        pass
                    logger.warning("No real-time market data subscription — falling back to polling")
        except Exception as e:
            logger.warning(f"keepUpToDate failed: {e}")

        # Fall back to polling mode (works with historical data, no subscription needed)
        self._streaming_active = True
        self._use_polling = True
        self._poll_interval = 60  # 1-min bars close every 60s; polling faster just returns the same bar
        # Delay first poll by 5s to avoid pacing violations after the keepUpToDate cancel
        self._last_poll_time = _time.monotonic() - self._poll_interval + 5.0
        logger.info("Streaming 1-min bars started (polling every 60s). "
                     "Subscribe to CME market data for real-time streaming.")
        return True

    def stop_streaming(self) -> None:
        """Stop streaming bars."""
        if self._streaming_active:
            if not self._use_polling and self._streaming_bars:
                try:
                    self._ib.cancelHistoricalData(self._streaming_bars)
                except Exception:
                    pass
            self._streaming_active = False
            logger.info("Streaming/polling stopped")

    def get_latest_bar(self) -> Optional[dict]:
        """Get the most recent closed 1-minute bar.

        Returns None if no new bar since last call.
        Works in both streaming mode (keepUpToDate) and polling mode.
        """
        if not self.is_connected or self._contract is None:
            return None

        if not self._streaming_active:
            return None

        if self._use_polling:
            return self._poll_latest_bar()
        else:
            return self._stream_latest_bar()

    def _stream_latest_bar(self) -> Optional[dict]:
        """Get latest bar from keepUpToDate stream."""
        if not self._streaming_bars or len(self._streaming_bars) < 2:
            return None

        bar = self._streaming_bars[-2]
        return self._extract_bar(bar)

    def _poll_latest_bar(self) -> Optional[dict]:
        """Get latest bar by polling historical data.

        Rate-limited to one request per _poll_interval seconds.
        """
        now = _time.monotonic()
        if now - self._last_poll_time < self._poll_interval:
            return None  # Too soon, skip this poll
        self._last_poll_time = now

        try:
            bars = self._ib.reqHistoricalData(
                self._contract,
                endDateTime="",
                durationStr="3600 S",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=self.config.use_rth_only,
                formatDate=2,
                timeout=30,
            )
        except Exception as e:
            logger.error(f"Poll failed: {e}")
            return None

        if not bars or len(bars) < 2:
            logger.warning(f"Poll returned {len(bars) if bars else 0} bars")
            self._zero_volume_count += 1
            if self._zero_volume_count >= 10:
                self._attempt_contract_reroll("Poll returning no bars")
            return None

        # bars[-1] may be incomplete, bars[-2] is last closed bar
        bar = bars[-2]

        # Zero-volume detection: expired contracts produce bars with V=0
        if getattr(bar, "volume", -1) == 0:
            self._zero_volume_count += 1
            if self._zero_volume_count >= 5:
                self._attempt_contract_reroll(
                    f"Zero-volume bars for {self._zero_volume_count} polls "
                    f"(contract may be expired)"
                )
        else:
            self._zero_volume_count = 0

        result = self._extract_bar(bar)
        if result:
            logger.info(f"Poll OK: {len(bars)} bars, latest closed={bar.date}")
            self._stale_count = 0
        else:
            # Check staleness: if bar is old during market hours, escalate
            # First check if market is closed (weekends + daily break)
            from datetime import datetime as _dt, time as _t
            _now = _dt.now()
            _wd = _now.weekday()
            _nt = _now.time()
            _market_closed = (
                _wd == 5  # Saturday
                or (_wd == 6 and _nt < _t(18, 0))  # Sunday before 6 PM
                or (_wd == 4 and _nt >= _t(17, 0))  # Friday after 5 PM
                or (_t(17, 0) <= _nt < _t(18, 0))   # Daily break
            )

            if _market_closed:
                # During daily break / market closure, reset stale counter
                self._stale_count = 0
                logger.debug(f"Poll OK but no new bar (market closed): {len(bars)} bars, latest={bar.date}")
            else:
                self._stale_count = getattr(self, "_stale_count", 0) + 1
                bar_time = pd.Timestamp(bar.date)
                if bar_time.tzinfo is None:
                    bar_time = bar_time.tz_localize("UTC")
                age_minutes = (pd.Timestamp.now(tz="UTC") - bar_time).total_seconds() / 60

                if age_minutes > 5 and self._stale_count >= 3:
                    logger.error(
                        f"STALE DATA: latest bar is {age_minutes:.0f} min old "
                        f"({bar.date}), stale for {self._stale_count} polls. "
                        f"IB may be disconnected or not returning new session bars."
                    )
                elif self._stale_count <= 1:
                    logger.debug(f"Poll OK but no new bar (dedup): {len(bars)} bars, latest={bar.date}, last_seen={self._last_bar_time}")
        return result

    def _extract_bar(self, bar) -> Optional[dict]:
        """Extract bar dict and deduplicate by timestamp."""
        bar_time = pd.Timestamp(bar.date)
        if bar_time.tzinfo is None:
            bar_time = bar_time.tz_localize("UTC")
        bar_time_et = bar_time.tz_convert("US/Eastern")

        if self._last_bar_time is not None and bar_time_et <= self._last_bar_time:
            return None  # Already processed

        self._last_bar_time = bar_time_et

        return {
            "timestamp": bar_time_et.isoformat(),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }

    def get_prior_day_bars(self) -> Optional[pd.DataFrame]:
        """Get all 1-min bars from the prior trading day.

        Used for level initialization at session start.
        """
        if not self.is_connected or self._contract is None:
            return None

        # Request 2 days of RTH data to capture prior day
        try:
            bars = self._ib.reqHistoricalData(
                self._contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,
                timeout=self.config.hist_timeout_sec,
            )
        except Exception as e:
            logger.error(f"get_prior_day_bars failed: {e}")
            return None

        if not bars:
            return None

        df = util.df(bars)
        if df is None or df.empty:
            return None

        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df.index = df.index.tz_convert("US/Eastern")

        cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]
        df = df[available]

        today = date.today()
        prior = df[df.index.date < today]
        if prior.empty:
            return None

        last_date = prior.index.date[-1]
        return prior[prior.index.date == last_date]

    def get_daily_bars(self, days: int = 365) -> Optional[pd.DataFrame]:
        """Get daily OHLCV bars for regime filter computation.

        Requests `days` of daily bars from IB. Needed for the 80-day EMA
        regime filter which requires long-term daily history.
        """
        if not self.is_connected or self._contract is None:
            return None

        duration = f"{days} D"
        try:
            bars = self._ib.reqHistoricalData(
                self._contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,
                timeout=self.config.hist_timeout_sec,
            )
        except Exception as e:
            logger.error(f"get_daily_bars failed: {e}")
            return None

        if not bars:
            return None

        df = util.df(bars)
        if df is None or df.empty:
            return None

        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df.index = df.index.tz_convert("US/Eastern")

        cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]
        df = df[available]
        logger.info(f"Daily bars: {len(df)} days loaded for regime filter")
        return df

    # ── Order Execution ───────────────────────────────────────────────

    def send_entry(
        self,
        quantity: int,
        sl: float,
        tp: float,
        direction: str = "long",
        comment: str = "ManciniEntry",
        fill_timeout_sec: float = 30.0,
        tp_fraction: float = 0.75,
        entry_price: float = 0.0,
        slippage_cap_pts: float = 0.0,
    ) -> tuple[Optional[int], float]:
        """Send a bracket order: market entry + SL + TP as OCO.

        Waits for the parent (market) order to fill before returning.
        If the parent doesn't fill within fill_timeout_sec, the entire
        bracket is cancelled and (None, 0.0) is returned.

        Uses IB's bracket order mechanism: parent (market) + take-profit
        (limit) + stop-loss (stop). The child orders have
        parentId = parent.orderId so IB treats them as OCO.

        Parameters
        ----------
        quantity : int
            Number of MES contracts.
        sl : float
            Stop loss price.
        tp : float
            Take profit price.
        direction : str
            "long" for BUY entry or "short" for SELL entry.
        comment : str
            Order reference for identification.
        fill_timeout_sec : float
            Max seconds to wait for parent fill confirmation.
        tp_fraction : float
            Fraction of contracts to put on the TP order (rest become runner).
            Default 0.75 = Mancini's 75/25 exit scaling.

        Returns
        -------
        (order_id, fill_price) : tuple[int | None, float]
            order_id: Parent order ID if filled, None if failed/timeout.
            fill_price: Actual fill price from IB, or 0.0 if unknown.
        """
        if not self.is_connected or self._contract is None:
            return None, 0.0

        if quantity <= 0:
            logger.error(f"send_entry() called with quantity={quantity} — rejecting ghost order")
            return None, 0.0

        parent_id = self._ib.client.getReqId()
        tp_id = self._ib.client.getReqId()
        sl_id = self._ib.client.getReqId()

        # Marketable-limit entry: bound adverse slippage to slippage_cap_pts.
        # Falls back to a market order when no cap/price is supplied.
        entry_limit_price = None
        if entry_price > 0 and slippage_cap_pts > 0:
            entry_limit_price = marketable_limit_price(
                direction, entry_price, slippage_cap_pts)

        # Bracket: marketable-limit (or market) parent + OCA reduce-on-fill
        # TP/SL, so the runner rests at the venue with a stop that auto-shrinks
        # when the TP fraction fills.
        parent, take_profit, stop_loss = build_bracket_orders(
            parent_id=parent_id, tp_id=tp_id, sl_id=sl_id,
            quantity=quantity, tp_fraction=tp_fraction,
            entry_limit_price=entry_limit_price, sl=sl, tp=tp,
            direction=direction, comment=comment,
        )

        try:
            parent_trade = self._ib.placeOrder(self._contract, parent)
            tp_trade = self._ib.placeOrder(self._contract, take_profit)
            sl_trade = self._ib.placeOrder(self._contract, stop_loss)

            # Wait for parent fill confirmation with polling loop
            fill_price = 0.0
            filled = False
            deadline = _time.monotonic() + fill_timeout_sec
            poll_interval = 0.5  # check every 500ms

            while _time.monotonic() < deadline:
                self._ib.sleep(poll_interval)

                # Check parent trade for fills
                if parent_trade.fills:
                    fill = parent_trade.fills[-1]
                    fill_price = getattr(fill, "avgPrice", 0.0) or getattr(fill, "price", 0.0)
                    if fill_price > 0:
                        filled = True
                        break

                # Also check order status
                status = parent_trade.orderStatus.status if parent_trade.orderStatus else ""
                if status == "Filled":
                    # Fills list may not be populated yet, try avgFillPrice
                    fill_price = getattr(parent_trade.orderStatus, "avgFillPrice", 0.0)
                    filled = True
                    break
                elif status in ("Cancelled", "Inactive", "ApiCancelled"):
                    logger.error(
                        f"Parent order {parent_id} was {status} by IB — bracket rejected"
                    )
                    return None, 0.0

            if not filled:
                # Timeout: cancel the entire bracket
                logger.error(
                    f"ENTRY TIMEOUT: parent order {parent_id} not filled after "
                    f"{fill_timeout_sec:.0f}s — cancelling bracket"
                )
                try:
                    self._ib.cancelOrder(parent)
                    self._ib.cancelOrder(take_profit)
                    self._ib.cancelOrder(stop_loss)
                    self._ib.sleep(1.0)
                except Exception:
                    pass
                return None, 0.0

            # Store for tracking (including direction for partial exits)
            self._active_orders[parent_id] = {
                "parent": parent_trade,
                "tp": tp_trade,
                "sl": sl_trade,
                "tp_order_id": tp_id,
                "sl_order_id": sl_id,
                "quantity": quantity,
                "direction": direction,
            }

            logger.info(
                f"BRACKET ENTRY FILLED ({direction.upper()}): parentId={parent_id}, "
                f"fill={fill_price:.2f}, {quantity} MES "
                f"SL={_round_tick(sl):.2f} TP={_round_tick(tp):.2f} [{comment}]"
            )
            return parent_id, fill_price

        except Exception as e:
            logger.error(f"Bracket order failed: {e}")
            return None, 0.0

    def update_stop(self, trade_id: int, new_sl: float, reason: str = "") -> bool:
        """Modify the stop loss order in an active bracket.

        Parameters
        ----------
        trade_id : int
            Parent order ID from send_entry().
        new_sl : float
            New stop loss price.

        Returns True if modification succeeded.
        """
        if not self.is_connected:
            return False

        bracket = self._active_orders.get(trade_id)
        if bracket is None:
            logger.warning(f"No bracket found for trade {trade_id}")
            return False

        sl_trade = bracket["sl"]
        try:
            # Modify the stop order
            sl_trade.order.auxPrice = _round_tick(new_sl)
            self._ib.placeOrder(self._contract, sl_trade.order)
            self._ib.sleep(0.5)

            logger.info(f"STOP UPDATED: trade={trade_id}, "
                         f"new_sl={new_sl:.2f} [{reason}]")
            return True
        except Exception as e:
            logger.error(f"Stop update failed: {e}")
            return False

    def flatten(self, reason: str = "") -> bool:
        """Close all open positions for our MES contract.

        Cancels any open bracket orders and sends a market sell to close.
        Returns True if successful.
        """
        if not self.is_connected or self._contract is None:
            return False

        # Cancel all pending orders for this contract
        open_orders = self._ib.openOrders()
        for order in open_orders:
            try:
                self._ib.cancelOrder(order)
            except Exception:
                pass

        # Check current position
        positions = self._ib.positions()
        for pos in positions:
            if (pos.contract.symbol == self.config.symbol and
                    pos.contract.secType == "FUT" and
                    pos.position != 0):
                qty = abs(int(pos.position))
                action = "SELL" if pos.position > 0 else "BUY"
                direction = "long" if pos.position > 0 else "short"
                # DAY, not GTC: market orders with GTC are rejected on some
                # IB routes and the rejection arrives asynchronously — a
                # silently dead close order is how trade #16872 survived
                # its "Stop loss hit" flatten overnight.
                close_order = MarketOrder(
                    action=action,
                    totalQuantity=qty,
                    orderRef=f"flatten:{reason}",
                    tif="DAY",
                )
                try:
                    trade = self._ib.placeOrder(self._contract, close_order)
                except Exception as e:
                    logger.error(f"Flatten failed: {e}")
                    return False

                # Verify the position actually went flat before reporting
                # success — callers log the exit and drop position tracking
                # when this returns True.
                confirmed = False
                for _ in range(8):
                    self._ib.sleep(1.0)
                    status = getattr(
                        getattr(trade, "orderStatus", None), "status", ""
                    )
                    if status in ("Cancelled", "ApiCancelled", "Inactive"):
                        break
                    still_open = any(
                        p.contract.symbol == self.config.symbol
                        and p.contract.secType == "FUT"
                        and p.position != 0
                        for p in self._ib.positions()
                    )
                    if not still_open:
                        confirmed = True
                        break
                if not confirmed:
                    logger.error(
                        f"FLATTEN NOT CONFIRMED: {qty} MES {direction} may "
                        f"still be open at IB (close order status: "
                        f"{getattr(getattr(trade, 'orderStatus', None), 'status', '?')}) "
                        f"[{reason}]"
                    )
                    return False
                logger.info(f"FLATTEN: closed {qty} MES {direction} [{reason}]")

        self._active_orders.clear()
        return True

    def partial_exit(
        self,
        trade_id: int,
        quantity: int,
        new_sl: float,
        reason: str = "",
    ) -> bool:
        """Close partial position and update stop on remainder.

        Parameters
        ----------
        trade_id : int
            Parent order ID from send_entry().
        quantity : int
            Number of contracts to close.
        new_sl : float
            New stop for remaining contracts.
        """
        if not self.is_connected or self._contract is None:
            return False

        if quantity <= 0:
            logger.error(f"partial_exit() called with quantity={quantity} — skipping")
            return False

        bracket = self._active_orders.get(trade_id)

        # Determine direction from stored bracket
        direction = bracket.get("direction", "long") if bracket else "long"
        exit_action = "BUY" if direction == "short" else "SELL"

        # Save old bracket children references before overwriting
        old_tp_trade = bracket.get("tp") if bracket else None
        old_sl_trade = bracket.get("sl") if bracket else None

        # Place new SL for remaining quantity FIRST to keep position protected
        if bracket:
            remaining = bracket["quantity"] - quantity
            if remaining > 0:
                new_sl_order = StopOrder(
                    action=exit_action,
                    totalQuantity=remaining,
                    stopPrice=_round_tick(new_sl),
                    orderRef=f"sl_after_partial:{reason}",
                    tif="GTC",
                )
                try:
                    sl_trade = self._ib.placeOrder(self._contract, new_sl_order)
                    self._ib.sleep(0.5)
                    bracket["sl"] = sl_trade
                    bracket["quantity"] = remaining
                except Exception as e:
                    logger.error(f"New SL placement failed: {e}")

        # Cancel old bracket children (new SL is already in place)
        if old_tp_trade or old_sl_trade:
            try:
                if old_tp_trade:
                    self._ib.cancelOrder(old_tp_trade.order)
                if old_sl_trade:
                    self._ib.cancelOrder(old_sl_trade.order)
                self._ib.sleep(0.5)
            except Exception:
                pass

        # Market exit the partial quantity
        partial_order = MarketOrder(
            action=exit_action,
            totalQuantity=quantity,
            orderRef=f"partial:{reason}",
            tif="GTC",
        )
        try:
            self._ib.placeOrder(self._contract, partial_order)
            self._ib.sleep(1.0)
            logger.info(f"PARTIAL EXIT: {quantity} MES [{reason}]")
        except Exception as e:
            logger.error(f"Partial exit failed: {e}")
            return False

        return True

    # ── Fill Price Retrieval ──────────────────────────────────────────

    def get_bracket_fill_price(self, trade_id: int) -> tuple[float, str]:
        """Retrieve the actual fill price from a bracket order's child trades.

        Checks the TP and SL Trade objects for fills to determine which
        child order was executed and at what price.

        Three strategies attempted in order:
        1. Check Trade.fills list (populated by IB event loop)
        2. Check Trade.orderStatus (avgFillPrice, status=="Filled")
        3. Query IB executions API as final fallback

        Returns
        -------
        (fill_price, exit_type) : tuple[float, str]
            fill_price: actual execution price, or 0.0 if unknown
            exit_type: "TP", "SL", or "unknown"
        """
        bracket = self._active_orders.get(trade_id)
        if not bracket:
            logger.warning(f"get_bracket_fill_price: no bracket found for trade_id={trade_id}")
            return 0.0, "unknown"

        # Flush IB event loop to receive any pending fill updates
        try:
            self._ib.sleep(1.0)
        except Exception:
            pass

        for label, trade_obj in [("tp", bracket.get("tp")), ("sl", bracket.get("sl"))]:
            if not trade_obj:
                continue

            # Strategy 1: check fills list
            if hasattr(trade_obj, "fills") and trade_obj.fills:
                fill = trade_obj.fills[-1]
                price = getattr(fill, "avgPrice", 0.0) or getattr(fill, "price", 0.0)
                if price > 0:
                    exit_type = "TP" if label == "tp" else "SL"
                    logger.info(f"Bracket {exit_type} filled (fills list): price={price:.2f} (trade_id={trade_id})")
                    return price, exit_type

            # Strategy 2: check orderStatus
            if hasattr(trade_obj, "orderStatus") and trade_obj.orderStatus:
                status = trade_obj.orderStatus.status
                avg_price = getattr(trade_obj.orderStatus, "avgFillPrice", 0.0)
                if status == "Filled" and avg_price > 0:
                    exit_type = "TP" if label == "tp" else "SL"
                    logger.info(f"Bracket {exit_type} filled (orderStatus): price={avg_price:.2f} (trade_id={trade_id})")
                    return avg_price, exit_type

        # Strategy 3: query IB executions as fallback
        try:
            tp_order_id = bracket.get("tp_order_id")
            sl_order_id = bracket.get("sl_order_id")
            all_fills = self._ib.fills()
            for f in all_fills:
                if f.contract.symbol == self.config.symbol:
                    oid = f.execution.orderId
                    price = f.execution.avgPrice or f.execution.price
                    if price > 0:
                        if oid == tp_order_id:
                            logger.info(f"Bracket TP filled (executions API): price={price:.2f} (trade_id={trade_id})")
                            return price, "TP"
                        elif oid == sl_order_id:
                            logger.info(f"Bracket SL filled (executions API): price={price:.2f} (trade_id={trade_id})")
                            return price, "SL"
        except Exception as e:
            logger.debug(f"Executions API fallback failed: {e}")

        # Last resort: check if we know the stop/target prices from the bracket
        # and infer from last known market price
        logger.warning(f"Bracket fill price unavailable after 3 strategies (trade_id={trade_id})")
        return 0.0, "unknown"

    def get_bracket_orders(self) -> dict:
        """Read actual SL/TP prices from open bracket orders on IB.

        Scans openOrders() for stop and limit orders on the current contract
        to recover the real bracket prices after a restart.

        Returns
        -------
        dict with keys: sl, tp, sl_order_id, tp_order_id
            Empty dict if no matching bracket orders found.
        """
        if not self.is_connected:
            return {}

        try:
            open_orders = self._ib.openOrders()
        except Exception as e:
            logger.warning(f"get_bracket_orders: openOrders() failed: {e}")
            return {}

        sl_price = 0.0
        tp_price = 0.0
        sl_order_id = None
        tp_order_id = None

        for order in open_orders:
            # Filter for orders on our futures contract
            contract = getattr(order, "contract", None)
            if contract is None:
                # openOrders() returns Order objects; need to match via trades
                continue

        # openOrders() returns Order objects without contract info attached.
        # Use openTrades() instead which pairs orders with contracts.
        try:
            open_trades = self._ib.openTrades()
        except Exception as e:
            logger.warning(f"get_bracket_orders: openTrades() failed: {e}")
            return {}

        for trade in open_trades:
            contract = trade.contract
            order = trade.order
            if (contract.symbol != self.config.symbol or
                    contract.secType != "FUT"):
                continue

            if order.orderType == "STP":
                sl_price = order.auxPrice
                sl_order_id = order.orderId
            elif order.orderType == "LMT":
                tp_price = order.lmtPrice
                tp_order_id = order.orderId

        if sl_price > 0 or tp_price > 0:
            result = {
                "sl": sl_price,
                "tp": tp_price,
                "sl_order_id": sl_order_id,
                "tp_order_id": tp_order_id,
            }
            logger.info(f"Found bracket orders on IB: SL={sl_price:.2f}, TP={tp_price:.2f}")
            return result

        return {}

    # ── Position Tracking ─────────────────────────────────────────────

    def get_position(self) -> Optional[dict]:
        """Get current open MES position.

        Returns dict with: position, avg_cost, contract
        Returns None if no position.
        """
        if not self.is_connected:
            return None

        positions = self._ib.positions()
        for pos in positions:
            if (pos.contract.symbol == self.config.symbol and
                    pos.contract.secType == "FUT" and
                    pos.position != 0):
                return {
                    "ticket": pos.contract.conId,
                    "volume": abs(int(pos.position)),
                    "price_open": pos.avgCost / 5.0,  # MES multiplier is 5
                    "sl": 0.0,  # IB doesn't track SL on position level
                    "tp": 0.0,
                    "profit": 0.0,
                    "time": datetime.now(),
                    "market_position": "long" if pos.position > 0 else "short",
                }
        return None

    def get_account_info(self) -> Optional[dict]:
        """Get account balance, equity, margin info."""
        if not self.is_connected:
            return None

        try:
            summary = self._ib.accountSummary()
            info = {}
            for item in summary:
                if item.tag == "NetLiquidation":
                    info["equity"] = float(item.value)
                elif item.tag == "TotalCashValue":
                    info["balance"] = float(item.value)
                elif item.tag == "BuyingPower":
                    info["free_margin"] = float(item.value)
                elif item.tag == "GrossPositionValue":
                    info["margin"] = float(item.value)

            info.setdefault("balance", 0.0)
            info.setdefault("equity", 0.0)
            info.setdefault("free_margin", 0.0)
            info.setdefault("margin", 0.0)
            info["profit"] = 0.0
            info["server"] = f"IB:{self.config.host}:{self.config.port}"
            info["name"] = ",".join(self._ib.managedAccounts())
            return info
        except Exception as e:
            logger.error(f"Account info failed: {e}")
            return None

    # ── Utility ───────────────────────────────────────────────────────

    def sleep(self, seconds: float) -> None:
        """IB-compatible sleep that keeps the event loop running.

        Catches ConnectionError from the underlying ib_async layer so the
        disconnect flag is reliably set even if the caller forgets to
        catch — the main loop's reconnect path then handles recovery.
        """
        try:
            self._ib.sleep(seconds)
        except ConnectionError:
            # Flag for reconnect (idempotent — _on_disconnect's own check
            # short-circuits if we've already disconnected) and re-raise
            # so the caller's loop knows to back off.
            if self._connected:
                self._connected = False
                self._needs_reconnect = True
            raise

    @property
    def ib(self) -> IB:
        """Direct access to IB instance for advanced usage."""
        return self._ib
