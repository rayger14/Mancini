"""Quick test trade: places a 1-lot MES bracket order to verify the full order flow.

Usage (inside container):
    python3 live/test_trade.py               # buy 1 MES, TP +2, SL -2
    python3 live/test_trade.py --direction short  # sell 1 MES
    python3 live/test_trade.py --dry-run     # just check connection, don't trade
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ib_async import IB, MarketOrder, LimitOrder, StopOrder, ContFuture
except ImportError:
    from ib_insync import IB, MarketOrder, LimitOrder, StopOrder, ContFuture


def round_tick(price: float, tick: float = 0.25) -> float:
    return round(round(price / tick) * tick, 2)


def main():
    parser = argparse.ArgumentParser(description="Test trade on MES")
    parser.add_argument("--host", default="ib-gateway", help="IB Gateway host")
    parser.add_argument("--port", type=int, default=4004, help="IB Gateway port")
    parser.add_argument("--direction", default="long", choices=["long", "short"])
    parser.add_argument("--tp-pts", type=float, default=2.0, help="Take profit distance in pts")
    parser.add_argument("--sl-pts", type=float, default=2.0, help="Stop loss distance in pts")
    parser.add_argument("--dry-run", action="store_true", help="Just connect, don't trade")
    args = parser.parse_args()

    ib = IB()
    print(f"Connecting to {args.host}:{args.port}...")
    ib.connect(args.host, args.port, clientId=99, timeout=15)
    print(f"Connected! Account: {ib.managedAccounts()}")

    # Qualify MES contract
    contract = ContFuture("MES", "CME")
    ib.qualifyContracts(contract)
    print(f"Contract: {contract}")

    # Get current price — try live data, then delayed, then status file
    ib.reqMarketDataType(3)  # 3 = delayed data as fallback
    ib.reqMktData(contract, "", False, False)
    mid = float("nan")
    for _ in range(20):
        ib.sleep(0.5)
        ticker = ib.ticker(contract)
        mid = ticker.midpoint()
        if mid != mid or mid <= 0:
            mid = ticker.last if (ticker.last == ticker.last and ticker.last > 0) else float("nan")
        if mid != mid or mid <= 0:
            # Try delayed fields
            mid = getattr(ticker, "close", float("nan"))
        if mid == mid and mid > 0:
            break
    if mid != mid or mid <= 0:
        # Last resort: read from bot's status file
        try:
            import json
            status = json.load(open("/app/logs/status.json"))
            mid = status.get("last_price", 0)
            if mid > 0:
                print(f"Using price from status.json: {mid}")
        except Exception:
            pass
    if mid != mid or mid <= 0:
        print("ERROR: Could not get a valid price. Aborting.")
        ib.disconnect()
        return
    print(f"Current price: {mid}")

    if args.dry_run:
        print("DRY RUN — not placing any orders.")
        ib.disconnect()
        return

    # Calculate bracket prices
    if args.direction == "long":
        tp = round_tick(mid + args.tp_pts)
        sl = round_tick(mid - args.sl_pts)
        entry_action, exit_action = "BUY", "SELL"
    else:
        tp = round_tick(mid - args.tp_pts)
        sl = round_tick(mid + args.sl_pts)
        entry_action, exit_action = "SELL", "BUY"

    print(f"\nPlacing {args.direction.upper()} bracket order:")
    print(f"  Entry: MARKET ({entry_action} 1 MES)")
    print(f"  TP:    {tp} ({exit_action})")
    print(f"  SL:    {sl} ({exit_action})")

    # Create bracket
    parent_id = ib.client.getReqId()
    tp_id = ib.client.getReqId()
    sl_id = ib.client.getReqId()

    parent = MarketOrder(
        action=entry_action,
        totalQuantity=1,
        orderId=parent_id,
        transmit=False,
        orderRef="TestTrade",
        tif="GTC",
    )

    take_profit = LimitOrder(
        action=exit_action,
        totalQuantity=1,
        lmtPrice=tp,
        orderId=tp_id,
        parentId=parent_id,
        transmit=False,
        orderRef="TestTrade:TP",
        tif="GTC",
    )

    stop_loss = StopOrder(
        action=exit_action,
        totalQuantity=1,
        stopPrice=sl,
        orderId=sl_id,
        parentId=parent_id,
        transmit=True,
        orderRef="TestTrade:SL",
        tif="GTC",
    )

    # Place orders
    trades = []
    for order in [parent, take_profit, stop_loss]:
        trade = ib.placeOrder(contract, order)
        trades.append(trade)
        print(f"  Placed: {order.orderType} {order.action} qty={order.totalQuantity} id={order.orderId}")

    # Wait for fill
    print("\nWaiting for parent fill...")
    for i in range(30):
        ib.sleep(1)
        parent_trade = trades[0]
        if parent_trade.orderStatus.status == "Filled":
            fill_price = parent_trade.orderStatus.avgFillPrice
            print(f"  FILLED at {fill_price}!")
            print(f"  TP waiting at {tp}")
            print(f"  SL waiting at {sl}")
            break
        print(f"  Status: {parent_trade.orderStatus.status}...")
    else:
        print("  Timed out waiting for fill")

    print("\nMonitoring bracket for 60s...")
    for i in range(60):
        ib.sleep(1)
        tp_status = trades[1].orderStatus.status
        sl_status = trades[2].orderStatus.status
        if tp_status == "Filled":
            print(f"  TP HIT at {trades[1].orderStatus.avgFillPrice} — trade complete!")
            break
        if sl_status == "Filled":
            print(f"  SL HIT at {trades[2].orderStatus.avgFillPrice} — trade complete!")
            break
        if i % 10 == 0:
            t = ib.ticker(contract)
            print(f"  [{i}s] Price: {t.midpoint():.2f}  TP: {tp_status}  SL: {sl_status}")
    else:
        print("  Monitoring timeout — bracket still active. Check TWS/Gateway.")

    ib.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
