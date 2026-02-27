"""RPyC server that exposes MetaTrader5 API from Wine Python to Mac Python.

This replaces the mt5linux package's server component. Run this inside Wine:

    wine python.exe live/mt5_server.py

Then connect from Mac Python using the mt5linux client or our mt5_bridge.py.

The server exposes all MetaTrader5 functions via RPyC so that Mac Python
can call them as if they were local.
"""

import rpyc
from rpyc.utils.server import ThreadedServer
import MetaTrader5 as mt5
import sys


class MT5Service(rpyc.Service):
    """RPyC service that proxies all MetaTrader5 API calls."""

    ALIASES = ["MT5"]

    def on_connect(self, conn):
        print(f"[MT5 Server] Client connected")

    def on_disconnect(self, conn):
        print(f"[MT5 Server] Client disconnected")

    # ── Core ──────────────────────────────────────────────────────

    def exposed_initialize(self, **kwargs):
        return mt5.initialize(**kwargs)

    def exposed_shutdown(self):
        return mt5.shutdown()

    def exposed_last_error(self):
        return mt5.last_error()

    def exposed_version(self):
        return mt5.version()

    # ── Terminal / Account ────────────────────────────────────────

    def exposed_terminal_info(self):
        info = mt5.terminal_info()
        if info is None:
            return None
        return info._asdict()

    def exposed_account_info(self):
        info = mt5.account_info()
        if info is None:
            return None
        return info._asdict()

    # ── Symbol ────────────────────────────────────────────────────

    def exposed_symbol_info(self, symbol):
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return info._asdict()

    def exposed_symbol_info_tick(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return tick._asdict()

    def exposed_symbol_select(self, symbol, enable=True):
        return mt5.symbol_select(symbol, enable)

    def exposed_symbols_get(self, group=""):
        symbols = mt5.symbols_get(group=group) if group else mt5.symbols_get()
        if symbols is None:
            return None
        return [s._asdict() for s in symbols]

    # ── Market Data ───────────────────────────────────────────────

    def exposed_copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
        if rates is None:
            return None
        # Convert numpy structured array to list of tuples for serialization
        return [tuple(r) for r in rates]

    def exposed_copy_rates_from(self, symbol, timeframe, date_from, count):
        rates = mt5.copy_rates_from(symbol, timeframe, date_from, count)
        if rates is None:
            return None
        return [tuple(r) for r in rates]

    def exposed_copy_rates_range(self, symbol, timeframe, date_from, date_to):
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
        if rates is None:
            return None
        return [tuple(r) for r in rates]

    def exposed_copy_ticks_from(self, symbol, date_from, count, flags):
        ticks = mt5.copy_ticks_from(symbol, date_from, count, flags)
        if ticks is None:
            return None
        return [tuple(t) for t in ticks]

    # ── Orders / Positions ────────────────────────────────────────

    def exposed_order_send(self, request):
        """Send trading order. request is a dict."""
        # Convert dict to MqlTradeRequest
        result = mt5.order_send(request)
        if result is None:
            return None
        return result._asdict()

    def exposed_positions_get(self, **kwargs):
        positions = mt5.positions_get(**kwargs)
        if positions is None:
            return None
        return [p._asdict() for p in positions]

    def exposed_positions_total(self):
        return mt5.positions_total()

    def exposed_orders_get(self, **kwargs):
        orders = mt5.orders_get(**kwargs)
        if orders is None:
            return None
        return [o._asdict() for o in orders]

    def exposed_orders_total(self):
        return mt5.orders_total()

    def exposed_history_orders_get(self, date_from, date_to, **kwargs):
        orders = mt5.history_orders_get(date_from, date_to, **kwargs)
        if orders is None:
            return None
        return [o._asdict() for o in orders]

    def exposed_history_deals_get(self, date_from, date_to, **kwargs):
        deals = mt5.history_deals_get(date_from, date_to, **kwargs)
        if deals is None:
            return None
        return [d._asdict() for d in deals]

    # ── Constants (expose MT5 enums) ──────────────────────────────

    def exposed_get_constant(self, name):
        """Get any MT5 constant by name (e.g., 'TRADE_ACTION_DEAL')."""
        return getattr(mt5, name, None)

    def exposed_get_constants(self):
        """Return dict of all commonly used MT5 constants."""
        constants = {}
        for name in dir(mt5):
            if name.startswith(('TRADE_', 'ORDER_', 'POSITION_', 'TIMEFRAME_',
                              'TICK_FLAG_', 'COPY_TICKS_', 'SYMBOL_',
                              'ACCOUNT_')):
                val = getattr(mt5, name, None)
                if isinstance(val, (int, float)):
                    constants[name] = val
        return constants


def main():
    port = 18812
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    print(f"[MT5 Server] Starting RPyC server on port {port}...")
    print(f"[MT5 Server] MetaTrader5 version: {mt5.__version__}")
    print(f"[MT5 Server] Initializing MT5 connection...")

    if not mt5.initialize():
        print(f"[MT5 Server] WARNING: MT5 not initialized yet. "
              f"Make sure MetaTrader 5 terminal is running.")
        print(f"[MT5 Server] Clients can call initialize() after connecting.")

    print(f"[MT5 Server] Ready. Waiting for connections on localhost:{port}")
    print(f"[MT5 Server] Press Ctrl+C to stop.")

    server = ThreadedServer(
        MT5Service,
        port=port,
        protocol_config={
            "allow_public_attrs": True,
            "allow_pickle": True,
        },
    )
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[MT5 Server] Shutting down...")
        mt5.shutdown()


if __name__ == "__main__":
    main()
