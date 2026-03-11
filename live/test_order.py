"""Quick test: connect to IB paper trading and place a 1-contract MES bracket order.

Usage:
    python3 live/test_order.py              # default: paper port 7497
    python3 live/test_order.py --port 7497  # explicit paper
    python3 live/test_order.py --dry-run    # connect + qualify only, no order

Requires TWS or IB Gateway running with API enabled on the specified port.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from live.ib_bridge import IBBridge, IBConfig


def main():
    parser = argparse.ArgumentParser(description="IB Test Order")
    parser.add_argument("--port", type=int, default=7497,
                        help="TWS/Gateway port (default: 7497 paper)")
    parser.add_argument("--symbol", default="MES", help="Symbol (default: MES)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Connect and get quote only, don't place order")
    args = parser.parse_args()

    if args.port == 7496:
        print("WARNING: Port 7496 is LIVE trading. Use 7497 for paper.")
        confirm = input("Type 'YES' to continue with LIVE: ")
        if confirm.strip() != "YES":
            print("Aborted.")
            return

    config = IBConfig(
        port=args.port,
        symbol=args.symbol,
        client_id=99,  # Use different client_id to avoid conflicts
    )
    bridge = IBBridge(config)

    print(f"Connecting to IB on port {args.port}...")
    if not bridge.connect():
        print("FAILED to connect. Is TWS/IB Gateway running with API enabled?")
        print("  - TWS: File > Global Configuration > API > Settings")
        print("  - Enable 'ActiveX and Socket Clients'")
        print(f"  - Socket port: {args.port}")
        print("  - Trusted IPs: 127.0.0.1")
        return

    print(f"Connected! Account: {bridge._ib.managedAccounts()}")

    # Get account info
    acct = bridge.get_account_info()
    if acct:
        print(f"Balance: ${acct.get('balance', 0):,.2f}")
        print(f"Equity:  ${acct.get('equity', 0):,.2f}")

    # Get current price
    bars = bridge.get_bars(count=5)
    if bars is None or bars.empty:
        print("Could not get bar data. Market may be closed.")
        bridge.disconnect()
        return

    last_close = float(bars["close"].iloc[-1])
    print(f"\nCurrent {args.symbol} price: {last_close:.2f}")

    # Check existing positions
    pos = bridge.get_position()
    if pos:
        print(f"Existing position: {pos['volume']} contracts @ {pos['price_open']:.2f}")

    if args.dry_run:
        print("\n--dry-run: skipping order placement")
        bridge.disconnect()
        return

    # Place test bracket order: 1 contract, 5pt stop, 5pt target
    sl = round(last_close - 5.0, 2)
    tp = round(last_close + 5.0, 2)
    qty = 1

    print(f"\nPlacing TEST bracket order:")
    print(f"  BUY {qty} {args.symbol} @ market")
    print(f"  Stop Loss:   {sl:.2f} (-5 pts)")
    print(f"  Take Profit: {tp:.2f} (+5 pts)")

    confirm = input("\nProceed? (y/n): ")
    if confirm.strip().lower() != "y":
        print("Cancelled.")
        bridge.disconnect()
        return

    order_id, fill_price = bridge.send_entry(
        quantity=qty,
        sl=sl,
        tp=tp,
        comment="TEST_ORDER",
    )

    if order_id is not None:
        print(f"\nOrder filled! Parent ID: {order_id}, fill price: {fill_price:.2f}")
        print("Check TWS for order status.")

        # Wait and check fill
        bridge.sleep(3)
        pos = bridge.get_position()
        if pos:
            print(f"Position confirmed: {pos['volume']} @ {pos['price_open']:.2f}")
        else:
            print("No position detected yet (may still be filling).")
    else:
        print("Order FAILED -- check TWS logs")

    bridge.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
