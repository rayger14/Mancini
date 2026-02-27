"""MetaTrader 5 bridge for the Mancini strategy engine.

Wraps the MetaTrader5 Python API to provide bar data, order execution,
and position management.

On Windows:  import MetaTrader5 as mt5  (direct)
On Mac/Linux: connects via RPyC to mt5_server.py running inside Wine

Usage:
    bridge = MT5Bridge(MT5Config(symbol="MES-MICRO", magic=20260209))
    bridge.connect()
    bars = bridge.get_bars(count=400)
    bridge.send_entry(volume=4, sl=6041.50, tp=6052.00)

Mac setup:
    1. Start MT5 in Wine
    2. Start server: wine python.exe live/mt5_server.py
    3. Run strategy: python3 live/mt5_runner.py
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── MT5 RPyC client for Mac/Linux ────────────────────────────────────

class _MT5RPyCClient:
    """Thin client that connects to mt5_server.py via RPyC.

    Mimics the MetaTrader5 module API so MT5Bridge works identically
    on Mac/Linux as on Windows.
    """

    def __init__(self, host: str = "localhost", port: int = 18812):
        self._host = host
        self._port = port
        self._conn = None
        self._constants: dict = {}

    def initialize(self, **kwargs) -> bool:
        import rpyc
        try:
            self._conn = rpyc.connect(
                self._host, self._port,
                config={"allow_public_attrs": True, "allow_pickle": True},
            )
            result = self._conn.root.initialize(**kwargs)
            # Cache constants
            self._constants = self._conn.root.get_constants()
            return result
        except Exception as e:
            logger.error(f"RPyC connection failed: {e}")
            return False

    def shutdown(self):
        if self._conn:
            try:
                self._conn.root.shutdown()
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def last_error(self):
        if self._conn:
            return self._conn.root.last_error()
        return (-1, "Not connected")

    def terminal_info(self):
        if not self._conn:
            return None
        d = self._conn.root.terminal_info()
        return _DictProxy(d) if d else None

    def account_info(self):
        if not self._conn:
            return None
        d = self._conn.root.account_info()
        return _DictProxy(d) if d else None

    def symbol_info(self, symbol):
        if not self._conn:
            return None
        d = self._conn.root.symbol_info(symbol)
        return _DictProxy(d) if d else None

    def symbol_info_tick(self, symbol):
        if not self._conn:
            return None
        d = self._conn.root.symbol_info_tick(symbol)
        return _DictProxy(d) if d else None

    def symbol_select(self, symbol, enable=True):
        if not self._conn:
            return False
        return self._conn.root.symbol_select(symbol, enable)

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        if not self._conn:
            return None
        result = self._conn.root.copy_rates_from_pos(symbol, timeframe, start_pos, count)
        if result is None:
            return None
        # Convert list of tuples back to numpy structured array
        return np.array(result, dtype=[
            ('time', '<i8'), ('open', '<f8'), ('high', '<f8'),
            ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<i8'),
            ('spread', '<i4'), ('real_volume', '<i8'),
        ])

    def order_send(self, request):
        if not self._conn:
            return None
        d = self._conn.root.order_send(request)
        return _DictProxy(d) if d else None

    def positions_get(self, **kwargs):
        if not self._conn:
            return None
        result = self._conn.root.positions_get(**kwargs)
        if result is None:
            return None
        return [_DictProxy(p) for p in result]

    def __getattr__(self, name):
        """Look up MT5 constants like TRADE_ACTION_DEAL, TIMEFRAME_M1, etc."""
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._constants:
            return self._constants[name]
        # Try fetching from server
        if self._conn:
            val = self._conn.root.get_constant(name)
            if val is not None:
                self._constants[name] = val
                return val
        raise AttributeError(f"MT5 has no constant '{name}'")


class _DictProxy:
    """Makes a dict accessible via attribute access (like MT5 named tuples)."""
    def __init__(self, d):
        self._d = dict(d) if d else {}
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattribute__(name)
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(f"No attribute '{name}'")
    def _asdict(self):
        return self._d
    def __repr__(self):
        return f"_DictProxy({self._d})"


def _import_mt5(host: str = "localhost", port: int = 18812):
    """Get MT5 interface — direct on Windows, RPyC client on Mac/Linux."""
    if platform.system() == "Windows":
        try:
            import MetaTrader5 as mt5
            return mt5
        except ImportError:
            raise ImportError("MetaTrader5 not found. Install: pip install MetaTrader5")
    else:
        return _MT5RPyCClient(host=host, port=port)


@dataclass
class MT5Config:
    """Configuration for the MT5 bridge."""

    symbol: str = "MES-MICRO"  # Broker-specific symbol name for MES
    magic: int = 20260209  # EA magic number for order identification
    deviation: int = 20  # Max slippage in points
    timeframe: str = "M1"  # 1-minute bars
    # mt5linux connection (Mac/Linux only)
    host: str = "localhost"
    port: int = 18812
    # Session params
    poll_interval_sec: float = 0.5
    heartbeat_interval_sec: float = 5.0


class MT5Bridge:
    """Communication layer with MetaTrader 5 terminal.

    Provides bar data, order execution, and position tracking.
    All order sends include SL/TP as bracket orders so the position
    is protected even if Python crashes.
    """

    def __init__(self, config: MT5Config = MT5Config()):
        self.config = config
        self._mt5 = None
        self._connected = False
        self._last_bar_time: int = 0  # Unix timestamp of last processed bar
        self._pending_orders: dict[int, datetime] = {}  # ticket -> sent time
        self._symbol_info = None

    # ── Connection ────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialize connection to MT5 terminal.

        Returns True if connected successfully.
        """
        self._mt5 = _import_mt5(host=self.config.host, port=self.config.port)
        mt5 = self._mt5

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        # Verify symbol exists
        self._symbol_info = mt5.symbol_info(self.config.symbol)
        if self._symbol_info is None:
            logger.error(f"Symbol {self.config.symbol} not found in MT5")
            mt5.shutdown()
            return False

        # Enable symbol in market watch
        if not self._symbol_info.visible:
            mt5.symbol_select(self.config.symbol, True)

        self._connected = True
        info = mt5.terminal_info()
        logger.info(f"MT5 connected: {info.name if info else 'unknown'}")
        logger.info(f"Symbol: {self.config.symbol}, "
                     f"point={self._symbol_info.point}, "
                     f"digits={self._symbol_info.digits}")
        return True

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if self._mt5 is not None and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._mt5 is not None

    # ── Bar Data ──────────────────────────────────────────────────────

    def get_bars(self, count: int = 400) -> Optional[pd.DataFrame]:
        """Get the last `count` 1-minute bars as a DataFrame.

        Returns DataFrame with columns: open, high, low, close, volume
        and a DatetimeIndex in US/Eastern.
        """
        if not self.is_connected:
            return None

        mt5 = self._mt5
        # TIMEFRAME_M1 = 1-minute
        tf = mt5.TIMEFRAME_M1

        rates = mt5.copy_rates_from_pos(self.config.symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df.index = df.index.tz_convert("US/Eastern")
        df = df.rename(columns={
            "tick_volume": "volume",
        })
        # Keep only OHLCV
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    def get_latest_bar(self) -> Optional[dict]:
        """Get the most recent closed 1-minute bar.

        Returns None if no new bar since last call.
        """
        if not self.is_connected:
            return None

        mt5 = self._mt5
        rates = mt5.copy_rates_from_pos(self.config.symbol, mt5.TIMEFRAME_M1, 0, 2)
        if rates is None or len(rates) < 2:
            return None

        # rates[-1] is the current (incomplete) bar, rates[-2] is the last closed bar
        bar = rates[-2]  # Use the closed bar
        bar_time = int(bar[0])  # Unix timestamp

        if bar_time <= self._last_bar_time:
            return None  # Already processed

        self._last_bar_time = bar_time

        return {
            "timestamp": pd.Timestamp(bar_time, unit="s", tz="UTC")
                          .tz_convert("US/Eastern").isoformat(),
            "open": float(bar[1]),
            "high": float(bar[2]),
            "low": float(bar[3]),
            "close": float(bar[4]),
            "volume": float(bar[5]),  # tick_volume
        }

    def get_prior_day_bars(self) -> Optional[pd.DataFrame]:
        """Get all 1-min bars from the prior trading day.

        Used for level initialization at session start.
        """
        if not self.is_connected:
            return None

        # Get enough bars to cover today + prior day (2 * 390 RTH minutes + buffer)
        df = self.get_bars(count=900)
        if df is None:
            return None

        today = date.today()
        # Filter to bars before today
        prior = df[df.index.date < today]
        if prior.empty:
            return None

        # Get only the most recent prior trading day
        last_date = prior.index.date[-1]
        return prior[prior.index.date == last_date]

    # ── Order Execution ───────────────────────────────────────────────

    def send_entry(
        self,
        volume: int,
        sl: float,
        tp: float,
        comment: str = "ManciniEntry",
    ) -> Optional[int]:
        """Send a market buy order with stop loss and take profit.

        Parameters
        ----------
        volume : int
            Number of contracts/lots
        sl : float
            Stop loss price
        tp : float
            Take profit (target) price
        comment : str
            Order comment for identification

        Returns
        -------
        int or None : Order ticket if filled, None if rejected.
        """
        if not self.is_connected:
            return None

        mt5 = self._mt5
        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            logger.error("Cannot get current tick")
            return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "deviation": self.config.deviation,
            "magic": self.config.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"order_send returned None: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order rejected: retcode={result.retcode}, "
                         f"comment={result.comment}")
            return None

        ticket = result.order
        self._pending_orders[ticket] = datetime.now()
        logger.info(f"ENTRY FILLED: ticket={ticket}, {volume} @ {result.price:.2f}, "
                     f"SL={sl:.2f}, TP={tp:.2f}")
        return ticket

    def update_stop(self, position_ticket: int, new_sl: float, reason: str = "") -> bool:
        """Modify stop loss on an existing position.

        Parameters
        ----------
        position_ticket : int
            Position ticket from get_position()
        new_sl : float
            New stop loss price (must be tighter than current)

        Returns True if modification succeeded.
        """
        if not self.is_connected:
            return False

        mt5 = self._mt5

        # Get current position to find its TP
        positions = mt5.positions_get(ticket=position_ticket)
        if not positions:
            logger.warning(f"Position {position_ticket} not found for stop update")
            return False

        pos = positions[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.config.symbol,
            "position": position_ticket,
            "sl": round(new_sl, 2),
            "tp": pos.tp,  # Keep existing TP
            "magic": self.config.magic,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else mt5.last_error()
            logger.error(f"Stop update failed: {err}")
            return False

        logger.info(f"STOP UPDATED: position={position_ticket}, "
                     f"new_sl={new_sl:.2f} [{reason}]")
        return True

    def flatten(self, reason: str = "") -> bool:
        """Close all open positions for our symbol.

        Returns True if all positions closed successfully.
        """
        if not self.is_connected:
            return False

        mt5 = self._mt5
        positions = mt5.positions_get(symbol=self.config.symbol)
        if not positions:
            return True  # Nothing to close

        success = True
        for pos in positions:
            if pos.magic != self.config.magic:
                continue  # Not our position

            tick = mt5.symbol_info_tick(self.config.symbol)
            if tick is None:
                continue

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL,  # Close long with sell
                "price": tick.bid,
                "deviation": self.config.deviation,
                "magic": self.config.magic,
                "comment": f"flatten:{reason}",
                "position": pos.ticket,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                err = result.comment if result else mt5.last_error()
                logger.error(f"Flatten failed for ticket {pos.ticket}: {err}")
                success = False
            else:
                logger.info(f"FLATTEN: closed {pos.volume} @ {result.price:.2f} [{reason}]")

        return success

    def partial_exit(
        self,
        position_ticket: int,
        volume: int,
        new_sl: float,
        reason: str = "",
    ) -> bool:
        """Close partial position and update stop on remainder.

        Parameters
        ----------
        position_ticket : int
            Position to partially close
        volume : int
            Number of contracts to close
        new_sl : float
            New stop for remaining contracts
        """
        if not self.is_connected:
            return False

        mt5 = self._mt5
        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            return False

        # Partial close
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_SELL,
            "price": tick.bid,
            "deviation": self.config.deviation,
            "magic": self.config.magic,
            "comment": f"partial:{reason}",
            "position": position_ticket,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else mt5.last_error()
            logger.error(f"Partial exit failed: {err}")
            return False

        logger.info(f"PARTIAL EXIT: {volume} @ {result.price:.2f} [{reason}]")

        # Update stop on remaining position
        self.update_stop(position_ticket, new_sl, reason=f"after_partial:{reason}")
        return True

    # ── Position Tracking ─────────────────────────────────────────────

    def get_position(self) -> Optional[dict]:
        """Get current open position for our symbol and magic number.

        Returns dict with: ticket, volume, price_open, sl, tp, profit, time
        Returns None if no position.
        """
        if not self.is_connected:
            return None

        mt5 = self._mt5
        positions = mt5.positions_get(symbol=self.config.symbol)
        if not positions:
            return None

        for pos in positions:
            if pos.magic == self.config.magic:
                return {
                    "ticket": pos.ticket,
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "time": datetime.fromtimestamp(pos.time),
                    "market_position": "long" if pos.type == 0 else "short",
                }
        return None

    def get_account_info(self) -> Optional[dict]:
        """Get account balance, equity, margin info."""
        if not self.is_connected:
            return None

        mt5 = self._mt5
        info = mt5.account_info()
        if info is None:
            return None

        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "leverage": info.leverage,
            "server": info.server,
            "name": info.name,
        }

    # ── Utility ───────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Check if the market is currently tradeable."""
        if not self.is_connected:
            return False

        mt5 = self._mt5
        tick = mt5.symbol_info_tick(self.config.symbol)
        return tick is not None and tick.ask > 0

    @property
    def mt5(self):
        """Direct access to MT5 module for advanced usage."""
        return self._mt5
