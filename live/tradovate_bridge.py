"""Tradovate futures bridge for the Mancini strategy engine.

Wraps the Tradovate REST + WebSocket API to provide bar data, bracket order
execution (placeOSO), and position management for MES futures.

Key design:
- Bracket orders (entry + SL + TP) are sent as OSO orders, so the position
  is protected even if Python crashes.
- All times are converted to US/Eastern for strategy compatibility.
- Token auto-refresh before expiration.
- Credentials loaded from environment variables or .env file.

Usage:
    bridge = TradovateBridge(TradovateConfig())
    bridge.connect()
    bars = bridge.get_bars(count=400)
    trade_id = bridge.send_entry(quantity=4, sl=6041.50, tp=6052.00)
"""

from __future__ import annotations

import json
import os
import threading
import time as _time
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

try:
    from dotenv import load_dotenv
    # Load .env from live/ directory or project root
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        _root_env = Path(__file__).parent.parent / ".env"
        if _root_env.exists():
            load_dotenv(_root_env)
except ImportError:
    pass  # python-dotenv not installed, rely on env vars

try:
    import websocket  # websocket-client
except ImportError:
    websocket = None  # type: ignore[assignment]


# ── Tradovate API base URLs ─────────────────────────────────────────

_BASE_URLS = {
    "demo": "https://demo.tradovateapi.com/v1",
    "live": "https://live.tradovateapi.com/v1",
}
_MD_WS_URLS = {
    "demo": "wss://md-demo.tradovateapi.com/v1/websocket",
    "live": "wss://md-live.tradovateapi.com/v1/websocket",
}


@dataclass
class TradovateConfig:
    """Configuration for the Tradovate bridge.

    Credentials are loaded from environment variables by default.
    """

    username: str = ""
    password: str = ""
    app_id: str = ""
    cid: str = ""        # client ID
    sec: str = ""        # client secret
    env: str = "demo"    # "demo" or "live"
    device_id: str = "mancini_strategy_001"
    # MES contract
    symbol: str = "MES"
    # Polling
    poll_interval_sec: float = 1.0
    # Reconnection
    max_reconnect_attempts: int = 5
    reconnect_delay_sec: float = 5.0
    # Request timeout
    request_timeout_sec: float = 15.0
    # Token refresh margin (seconds before expiration)
    token_refresh_margin_sec: float = 300.0

    def __post_init__(self):
        """Load credentials from environment variables if not explicitly set."""
        if not self.username:
            self.username = os.environ.get("TRADOVATE_USERNAME", "")
        if not self.password:
            self.password = os.environ.get("TRADOVATE_PASSWORD", "")
        if not self.app_id:
            self.app_id = os.environ.get("TRADOVATE_APP_ID", "")
        if not self.cid:
            self.cid = os.environ.get("TRADOVATE_CID", "")
        if not self.sec:
            self.sec = os.environ.get("TRADOVATE_SEC", "")
        env_val = os.environ.get("TRADOVATE_ENV", "")
        if env_val and not self.env:
            self.env = env_val

    @property
    def base_url(self) -> str:
        return _BASE_URLS.get(self.env, _BASE_URLS["demo"])

    @property
    def md_ws_url(self) -> str:
        return _MD_WS_URLS.get(self.env, _MD_WS_URLS["demo"])


class TradovateBridge:
    """Communication layer with Tradovate REST + WebSocket API.

    Provides bar data, bracket order execution (placeOSO), and position
    tracking for MES futures.
    """

    def __init__(self, config: TradovateConfig = TradovateConfig()):
        self.config = config
        self._connected: bool = False

        # Auth tokens
        self._access_token: str = ""
        self._md_access_token: str = ""
        self._expiration_time: Optional[datetime] = None
        self._user_id: Optional[int] = None

        # Account info
        self._account_id: Optional[int] = None
        self._account_spec: str = ""

        # Contract
        self._contract_id: Optional[int] = None
        self._contract_name: str = ""  # e.g. "MESH6"

        # Bar tracking
        self._last_bar_time: Optional[pd.Timestamp] = None
        self._bar_buffer: list[dict] = []

        # Active bracket orders: parent_order_id -> order info
        self._active_orders: dict[int, dict] = {}

        # WebSocket for market data
        self._ws: Optional[object] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_connected: bool = False
        self._ws_request_id: int = 0

        # Session for HTTP keep-alive
        self._session = requests.Session()

    # ── Connection ────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Authenticate, get account info, and find front-month MES contract.

        Returns True if all steps succeeded.
        """
        # Validate credentials
        if not all([self.config.username, self.config.password,
                     self.config.app_id, self.config.cid, self.config.sec]):
            logger.error(
                "Missing Tradovate credentials. Set environment variables: "
                "TRADOVATE_USERNAME, TRADOVATE_PASSWORD, TRADOVATE_APP_ID, "
                "TRADOVATE_CID, TRADOVATE_SEC"
            )
            return False

        # Step 1: Authenticate
        if not self._authenticate():
            return False

        # Step 2: Get account info
        if not self._fetch_account_info():
            return False

        # Step 3: Find front-month MES contract
        if not self._find_contract():
            return False

        self._connected = True
        logger.info(
            f"Tradovate connected: env={self.config.env}, "
            f"account={self._account_spec} (id={self._account_id}), "
            f"contract={self._contract_name} (id={self._contract_id})"
        )
        return True

    def disconnect(self) -> None:
        """Cleanup: close WebSocket and session."""
        self._connected = False
        if self._ws is not None:
            try:
                self._ws.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._ws = None
        self._session.close()
        logger.info("Tradovate disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and bool(self._access_token)

    # ── Authentication ────────────────────────────────────────────────

    def _authenticate(self) -> bool:
        """POST /auth/accesstokenrequest to get access tokens."""
        url = f"{self.config.base_url}/auth/accesstokenrequest"
        payload = {
            "name": self.config.username,
            "password": self.config.password,
            "appId": self.config.app_id,
            "appVersion": "1.0",
            "deviceId": self.config.device_id,
            "cid": self.config.cid,
            "sec": self.config.sec,
        }

        try:
            resp = self._session.post(
                url, json=payload, timeout=self.config.request_timeout_sec
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Tradovate auth failed: {e}")
            return False

        self._access_token = data.get("accessToken", "")
        self._md_access_token = data.get("mdAccessToken", "")
        self._user_id = data.get("userId")

        exp_str = data.get("expirationTime", "")
        if exp_str:
            try:
                # Tradovate returns ISO format with timezone
                self._expiration_time = pd.Timestamp(exp_str).to_pydatetime()
            except Exception:
                # Fallback: assume 24h from now
                self._expiration_time = datetime.now(timezone.utc) + timedelta(hours=24)

        if not self._access_token:
            logger.error(f"No access token in auth response: {data}")
            return False

        # Set default Authorization header
        self._session.headers.update({
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        })

        logger.info(f"Tradovate authenticated: userId={self._user_id}, "
                     f"expires={self._expiration_time}")
        return True

    def _ensure_token_valid(self) -> bool:
        """Refresh token if close to expiration."""
        if self._expiration_time is None:
            return True

        now = datetime.now(timezone.utc)
        exp = self._expiration_time
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)

        remaining = (exp - now).total_seconds()
        if remaining < self.config.token_refresh_margin_sec:
            logger.info(f"Token expires in {remaining:.0f}s, refreshing...")
            return self._authenticate()
        return True

    # ── Account & Contract Discovery ──────────────────────────────────

    def _fetch_account_info(self) -> bool:
        """GET /account/list to find account ID and spec."""
        try:
            resp = self._session.get(
                f"{self.config.base_url}/account/list",
                timeout=self.config.request_timeout_sec,
            )
            resp.raise_for_status()
            accounts = resp.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch accounts: {e}")
            return False

        if not accounts:
            logger.error("No accounts found")
            return False

        # Use first account
        acct = accounts[0]
        self._account_id = acct.get("id")
        self._account_spec = acct.get("name", "")
        logger.info(f"Account: {self._account_spec} (id={self._account_id})")
        return True

    def _find_contract(self) -> bool:
        """GET /contract/suggest to find front-month MES contract."""
        try:
            resp = self._session.get(
                f"{self.config.base_url}/contract/suggest",
                params={"t": self.config.symbol, "l": 5},
                timeout=self.config.request_timeout_sec,
            )
            resp.raise_for_status()
            contracts = resp.json()
        except requests.RequestException as e:
            logger.error(f"Contract suggest failed: {e}")
            return False

        if not contracts:
            logger.error(f"No contracts found for {self.config.symbol}")
            return False

        # Pick the first result (front month)
        contract = contracts[0]
        self._contract_id = contract.get("id")
        self._contract_name = contract.get("name", self.config.symbol)
        logger.info(f"Contract: {self._contract_name} (id={self._contract_id})")
        return True

    # ── Bar Data ──────────────────────────────────────────────────────

    def get_bars(self, count: int = 400) -> Optional[pd.DataFrame]:
        """Get the last `count` 1-minute RTH bars as a DataFrame.

        Uses the Tradovate REST chart endpoint to fetch historical bars.
        Returns DataFrame with columns: open, high, low, close, volume
        and a DatetimeIndex in US/Eastern.
        """
        if not self.is_connected or self._contract_id is None:
            return None

        self._ensure_token_valid()

        # Tradovate REST chart endpoint: md/getchart
        # We use the replay history endpoint for historical bars
        try:
            # Use the /md/getchart endpoint via REST as a workaround:
            # Request tick chart data and aggregate, OR use the history endpoint
            # Tradovate provides /history/bars endpoint for historical data
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=count + 120)  # buffer

            resp = self._session.get(
                f"{self.config.base_url}/md/getchart",
                params={
                    "symbol": self._contract_name,
                    "chartDescription": json.dumps({
                        "underlyingType": "MinuteBar",
                        "elementSize": 1,
                        "elementSizeUnit": "UnderlyingUnits",
                        "withHistogram": False,
                    }),
                    "timeRange": json.dumps({
                        "asFarAsTimestamp": start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                        "closestTimestamp": end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    }),
                },
                timeout=self.config.request_timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"get_bars failed: {e}")
            return None

        bars = data.get("bars", data) if isinstance(data, dict) else data
        if not bars or not isinstance(bars, list):
            logger.warning("No bar data returned from Tradovate")
            return None

        return self._bars_to_dataframe(bars, count)

    def _bars_to_dataframe(
        self, bars: list[dict], limit: int = 400
    ) -> Optional[pd.DataFrame]:
        """Convert Tradovate bar data to a pandas DataFrame."""
        records = []
        for bar in bars:
            ts = bar.get("timestamp", bar.get("t", ""))
            try:
                if isinstance(ts, (int, float)):
                    timestamp = pd.Timestamp(ts, unit="s", tz="UTC")
                else:
                    timestamp = pd.Timestamp(ts)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.tz_localize("UTC")
            except Exception:
                continue

            records.append({
                "timestamp": timestamp,
                "open": float(bar.get("open", bar.get("o", 0))),
                "high": float(bar.get("high", bar.get("h", 0))),
                "low": float(bar.get("low", bar.get("l", 0))),
                "close": float(bar.get("close", bar.get("c", 0))),
                "volume": float(bar.get("volume", bar.get("v", 0))),
            })

        if not records:
            return None

        df = pd.DataFrame(records)
        df = df.set_index("timestamp")
        df.index = df.index.tz_convert("US/Eastern")

        # Filter RTH only (9:30-16:00 ET)
        from datetime import time as dt_time
        rth_start = dt_time(9, 30)
        rth_end = dt_time(16, 0)
        rth_mask = [(rth_start <= t.time() <= rth_end) for t in df.index]
        df = df[rth_mask]

        # Keep last N bars
        if len(df) > limit:
            df = df.iloc[-limit:]

        return df if not df.empty else None

    def get_latest_bar(self) -> Optional[dict]:
        """Get the most recent closed 1-minute bar.

        Returns None if no new bar since last call.
        """
        if not self.is_connected or self._contract_id is None:
            return None

        self._ensure_token_valid()

        # Fetch last 3 bars (to get the last completed one)
        df = self.get_bars(count=5)
        if df is None or len(df) < 2:
            return None

        # Last closed bar is df.iloc[-2] (df.iloc[-1] may still be forming)
        bar_time = df.index[-2]

        if self._last_bar_time is not None and bar_time <= self._last_bar_time:
            return None  # Already processed

        self._last_bar_time = bar_time

        return {
            "timestamp": bar_time.isoformat(),
            "open": float(df["open"].iloc[-2]),
            "high": float(df["high"].iloc[-2]),
            "low": float(df["low"].iloc[-2]),
            "close": float(df["close"].iloc[-2]),
            "volume": float(df["volume"].iloc[-2]),
        }

    def get_prior_day_bars(self) -> Optional[pd.DataFrame]:
        """Get all 1-min bars from the prior trading day.

        Used for level initialization at session start.
        """
        if not self.is_connected:
            return None

        # Request enough bars to cover 2 trading days (~780 bars per day)
        df = self.get_bars(count=800)
        if df is None:
            return None

        today = date.today()
        prior = df[df.index.date < today]
        if prior.empty:
            return None

        last_date = prior.index.date[-1]
        return prior[prior.index.date == last_date]

    # ── Order Execution ───────────────────────────────────────────────

    def send_entry(
        self,
        quantity: int,
        sl: float,
        tp: float,
        comment: str = "ManciniEntry",
    ) -> Optional[int]:
        """Send a market buy with bracket SL/TP via placeOSO.

        Parameters
        ----------
        quantity : int
            Number of MES contracts.
        sl : float
            Stop loss price.
        tp : float
            Take profit price.
        comment : str
            Order reference for identification.

        Returns
        -------
        int or None
            Order ID if submitted, None if failed.
        """
        if not self.is_connected or self._contract_id is None:
            return None

        self._ensure_token_valid()

        # Round prices to tick (0.25)
        sl = round(sl * 4) / 4
        tp = round(tp * 4) / 4

        payload = {
            "accountSpec": self._account_spec,
            "accountId": self._account_id,
            "symbol": self._contract_name,
            "action": "Buy",
            "orderType": "Market",
            "orderQty": quantity,
            "isAutomated": True,
            "text": comment,
            "bracket1": {
                "action": "Sell",
                "orderType": "Stop",
                "stopPrice": sl,
            },
            "bracket2": {
                "action": "Sell",
                "orderType": "Limit",
                "price": tp,
            },
        }

        try:
            resp = self._session.post(
                f"{self.config.base_url}/order/placeOSO",
                json=payload,
                timeout=self.config.request_timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"placeOSO failed: {e}")
            return None

        order_id = data.get("orderId", data.get("id"))
        if order_id is None:
            logger.error(f"No order ID in response: {data}")
            return None

        # Store bracket info for tracking
        self._active_orders[order_id] = {
            "order_id": order_id,
            "sl_price": sl,
            "tp_price": tp,
            "quantity": quantity,
            "bracket1_id": data.get("bracket1OrdId"),
            "bracket2_id": data.get("bracket2OrdId"),
            "data": data,
        }

        logger.info(
            f"BRACKET ENTRY: orderId={order_id}, {quantity} {self._contract_name} "
            f"SL={sl:.2f} TP={tp:.2f} [{comment}]"
        )
        return order_id

    def update_stop(self, trade_id: int, new_sl: float, reason: str = "") -> bool:
        """Modify the stop loss order in an active bracket.

        Parameters
        ----------
        trade_id : int
            Order ID from send_entry().
        new_sl : float
            New stop loss price.

        Returns True if modification succeeded.
        """
        if not self.is_connected:
            return False

        self._ensure_token_valid()

        bracket = self._active_orders.get(trade_id)
        if bracket is None:
            logger.warning(f"No bracket found for trade {trade_id}")
            return False

        sl_order_id = bracket.get("bracket1_id")
        if sl_order_id is None:
            logger.warning(f"No SL order ID for trade {trade_id}")
            return False

        new_sl = round(new_sl * 4) / 4  # round to tick

        payload = {
            "orderId": sl_order_id,
            "stopPrice": new_sl,
        }

        try:
            resp = self._session.post(
                f"{self.config.base_url}/order/modifyorder",
                json=payload,
                timeout=self.config.request_timeout_sec,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Stop update failed: {e}")
            return False

        bracket["sl_price"] = new_sl
        logger.info(f"STOP UPDATED: trade={trade_id}, "
                     f"new_sl={new_sl:.2f} [{reason}]")
        return True

    def flatten(self, reason: str = "") -> bool:
        """Close all open positions for our MES contract.

        Cancels any open bracket orders, then liquidates position.
        Returns True if successful.
        """
        if not self.is_connected:
            return False

        self._ensure_token_valid()

        # Cancel all tracked bracket orders
        for order_id, bracket in list(self._active_orders.items()):
            for key in ["bracket1_id", "bracket2_id", "order_id"]:
                oid = bracket.get(key)
                if oid is not None:
                    try:
                        self._session.post(
                            f"{self.config.base_url}/order/cancelorder",
                            json={"orderId": oid},
                            timeout=self.config.request_timeout_sec,
                        )
                    except Exception:
                        pass

        # Liquidate position
        pos = self.get_position()
        if pos and pos.get("volume", 0) > 0:
            try:
                resp = self._session.post(
                    f"{self.config.base_url}/order/placeorder",
                    json={
                        "accountSpec": self._account_spec,
                        "accountId": self._account_id,
                        "symbol": self._contract_name,
                        "action": "Sell",
                        "orderType": "Market",
                        "orderQty": pos["volume"],
                        "isAutomated": True,
                        "text": f"flatten:{reason}",
                    },
                    timeout=self.config.request_timeout_sec,
                )
                resp.raise_for_status()
                logger.info(f"FLATTEN: sold {pos['volume']} {self._contract_name} [{reason}]")
            except requests.RequestException as e:
                logger.error(f"Flatten market sell failed: {e}")
                return False

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
            Order ID from send_entry().
        quantity : int
            Number of contracts to close.
        new_sl : float
            New stop for remaining contracts.
        """
        if not self.is_connected:
            return False

        self._ensure_token_valid()

        bracket = self._active_orders.get(trade_id)

        # Cancel existing bracket children first
        if bracket:
            for key in ["bracket1_id", "bracket2_id"]:
                oid = bracket.get(key)
                if oid is not None:
                    try:
                        self._session.post(
                            f"{self.config.base_url}/order/cancelorder",
                            json={"orderId": oid},
                            timeout=self.config.request_timeout_sec,
                        )
                    except Exception:
                        pass

        # Market sell the partial quantity
        try:
            resp = self._session.post(
                f"{self.config.base_url}/order/placeorder",
                json={
                    "accountSpec": self._account_spec,
                    "accountId": self._account_id,
                    "symbol": self._contract_name,
                    "action": "Sell",
                    "orderType": "Market",
                    "orderQty": quantity,
                    "isAutomated": True,
                    "text": f"partial:{reason}",
                },
                timeout=self.config.request_timeout_sec,
            )
            resp.raise_for_status()
            logger.info(f"PARTIAL EXIT: {quantity} {self._contract_name} [{reason}]")
        except requests.RequestException as e:
            logger.error(f"Partial exit failed: {e}")
            return False

        # Place new SL for remaining quantity
        if bracket:
            remaining = bracket["quantity"] - quantity
            if remaining > 0:
                new_sl = round(new_sl * 4) / 4
                try:
                    resp = self._session.post(
                        f"{self.config.base_url}/order/placeorder",
                        json={
                            "accountSpec": self._account_spec,
                            "accountId": self._account_id,
                            "symbol": self._contract_name,
                            "action": "Sell",
                            "orderType": "Stop",
                            "stopPrice": new_sl,
                            "orderQty": remaining,
                            "isAutomated": True,
                            "text": f"sl_after_partial:{reason}",
                        },
                        timeout=self.config.request_timeout_sec,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    bracket["bracket1_id"] = data.get("orderId", data.get("id"))
                    bracket["bracket2_id"] = None
                    bracket["sl_price"] = new_sl
                    bracket["quantity"] = remaining
                except requests.RequestException as e:
                    logger.error(f"New SL placement failed: {e}")

        return True

    # ── Position Tracking ─────────────────────────────────────────────

    def get_position(self) -> Optional[dict]:
        """Get current open MES position.

        Returns dict with: ticket, volume, price_open, sl, tp, market_position
        Returns None if no position.
        """
        if not self.is_connected:
            return None

        self._ensure_token_valid()

        try:
            resp = self._session.get(
                f"{self.config.base_url}/position/list",
                timeout=self.config.request_timeout_sec,
            )
            resp.raise_for_status()
            positions = resp.json()
        except requests.RequestException as e:
            logger.error(f"get_position failed: {e}")
            return None

        for pos in positions:
            # Match by account and contract
            if pos.get("accountId") != self._account_id:
                continue

            net_pos = pos.get("netPos", 0)
            if net_pos > 0:
                # Check if this is our contract
                contract_id = pos.get("contractId")
                if contract_id != self._contract_id:
                    continue

                avg_price = pos.get("netPrice", 0)
                return {
                    "ticket": pos.get("id", 0),
                    "volume": int(net_pos),
                    "price_open": float(avg_price),
                    "sl": 0.0,  # Tradovate doesn't track SL on position level
                    "tp": 0.0,
                    "profit": float(pos.get("ohlcv", {}).get("pnl", 0)),
                    "time": datetime.now(),
                    "market_position": "long",
                }

        return None

    def get_account_info(self) -> Optional[dict]:
        """Get account balance, equity, margin info."""
        if not self.is_connected:
            return None

        self._ensure_token_valid()

        try:
            resp = self._session.get(
                f"{self.config.base_url}/account/list",
                timeout=self.config.request_timeout_sec,
            )
            resp.raise_for_status()
            accounts = resp.json()
        except requests.RequestException as e:
            logger.error(f"get_account_info failed: {e}")
            return None

        if not accounts:
            return None

        acct = accounts[0]
        # Tradovate account fields vary; extract what we can
        info = {
            "balance": float(acct.get("cashBalance", 0)),
            "equity": float(acct.get("netLiq", acct.get("cashBalance", 0))),
            "free_margin": float(acct.get("availableMargin", 0)),
            "margin": float(acct.get("initialMargin", 0)),
            "profit": 0.0,
            "server": f"Tradovate:{self.config.env}",
            "name": self._account_spec,
        }

        # Try to get more detailed cash balance info
        try:
            resp2 = self._session.get(
                f"{self.config.base_url}/cashBalance/getcashbalancesnapshot",
                params={"accountId": self._account_id},
                timeout=self.config.request_timeout_sec,
            )
            if resp2.status_code == 200:
                cb = resp2.json()
                if isinstance(cb, dict):
                    info["balance"] = float(cb.get("totalCashValue", info["balance"]))
                    info["equity"] = float(cb.get("netLiq", info["equity"]))
                    info["profit"] = float(cb.get("realizedPnl", 0))
        except Exception:
            pass

        return info

    # ── Utility ───────────────────────────────────────────────────────

    def _api_get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """Convenience GET request with error handling and token refresh."""
        self._ensure_token_valid()
        try:
            resp = self._session.get(
                f"{self.config.base_url}{endpoint}",
                params=params,
                timeout=self.config.request_timeout_sec,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"GET {endpoint} failed: {e}")
            return None

    def _api_post(self, endpoint: str, payload: dict) -> Optional[dict]:
        """Convenience POST request with error handling and token refresh."""
        self._ensure_token_valid()
        try:
            resp = self._session.post(
                f"{self.config.base_url}{endpoint}",
                json=payload,
                timeout=self.config.request_timeout_sec,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"POST {endpoint} failed: {e}")
            return None

    def sleep(self, seconds: float) -> None:
        """Simple sleep (compatible with runner interface)."""
        _time.sleep(seconds)
