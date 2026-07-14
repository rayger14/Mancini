"""Two-bracket (per-tranche) exit structure — the fix for the venue runner
collapse (issue #74).

Diagnosed at the venue 2026-07-13: exits attached to the parent via parentId
get their quantity RESIZED by the gateway to the parent's full quantity (we
send TP=1 of 2, IB rests TP=2), so the TP fill closes everything and the
runner never survives. Fix: place NO parent-attached exits. After the entry
fills, place independent tranches:
  Tranche A (tp_qty): TP + SL as their own OCA pair (cancel-all within pair)
  Tranche B (runner): a standalone SL only — the trail later moves this order
"""
from types import SimpleNamespace

import pytest

from live.ib_bridge import build_exit_tranches, IBBridge, IBConfig


# ---------------------------------------------------------------------------
# Pure construction helper
# ---------------------------------------------------------------------------

def test_tranches_two_lots():
    tpA, slA, slB = build_exit_tranches(
        quantity=2, tp_fraction=0.75, sl=7510.25, tp=7560.75,
        direction="long", oca_group="oca_x", tp_id=2, sl_id=3, runner_sl_id=4)
    # tranche A: 1-lot TP + 1-lot SL, OCA'd together, NOT parent-attached
    assert tpA.totalQuantity == 1 and slA.totalQuantity == 1
    assert tpA.ocaGroup == slA.ocaGroup == "oca_x"
    assert tpA.parentId == 0 and slA.parentId == 0
    # tranche B: 1-lot standalone SL, no OCA with tranche A, no parent
    assert slB is not None
    assert slB.totalQuantity == 1
    assert slB.parentId == 0
    assert slB.ocaGroup != "oca_x"
    # all sells for a long, all transmit, GTC
    for o in (tpA, slA, slB):
        assert o.action == "SELL"
        assert o.transmit is True
        assert o.tif == "GTC"
    assert tpA.lmtPrice == 7560.75
    assert slA.auxPrice == 7510.25 and slB.auxPrice == 7510.25


def test_tranches_four_lots():
    tpA, slA, slB = build_exit_tranches(
        quantity=4, tp_fraction=0.75, sl=7500.0, tp=7550.0,
        direction="long", oca_group="g", tp_id=2, sl_id=3, runner_sl_id=4)
    assert tpA.totalQuantity == 3 and slA.totalQuantity == 3
    assert slB.totalQuantity == 1


def test_tranches_single_lot_no_runner():
    tpA, slA, slB = build_exit_tranches(
        quantity=1, tp_fraction=0.75, sl=7500.0, tp=7550.0,
        direction="long", oca_group="g", tp_id=2, sl_id=3, runner_sl_id=4)
    assert tpA.totalQuantity == 1 and slA.totalQuantity == 1
    assert slB is None


def test_tranches_short_direction():
    tpA, slA, slB = build_exit_tranches(
        quantity=2, tp_fraction=0.75, sl=7550.0, tp=7500.0,
        direction="short", oca_group="g", tp_id=2, sl_id=3, runner_sl_id=4)
    for o in (tpA, slA, slB):
        assert o.action == "BUY"


# ---------------------------------------------------------------------------
# send_entry places the tranches (fake IB records everything)
# ---------------------------------------------------------------------------

class _FakeIB:
    def __init__(self):
        self.placed = []          # (order, trade) in placement order
        self._next_id = 100

    class _Client:
        def __init__(self, outer):
            self._o = outer

        def getReqId(self):
            self._o._next_id += 1
            return self._o._next_id

    @property
    def client(self):
        return _FakeIB._Client(self)

    def placeOrder(self, contract, order):
        # parent fills instantly; exits rest
        is_parent = order.orderType in ("MKT", "LMT") and not getattr(order, "ocaGroup", "") \
            and getattr(order, "orderRef", "").startswith(("ManciniEntry", "ForceTest", "test"))
        status = SimpleNamespace(status="Filled" if is_parent else "Submitted",
                                 avgFillPrice=7530.25 if is_parent else 0.0,
                                 filled=order.totalQuantity if is_parent else 0,
                                 remaining=0 if is_parent else order.totalQuantity)
        trade = SimpleNamespace(order=order, orderStatus=status,
                                fills=[SimpleNamespace(avgPrice=7530.25)] if is_parent else [])
        self.placed.append((order, trade))
        return trade

    def sleep(self, s):
        pass

    def cancelOrder(self, order):
        pass

    def isConnected(self):
        return True

    def openOrders(self):
        return []

    def positions(self):
        return []


def _bridge_with_fake():
    b = IBBridge.__new__(IBBridge)
    b.config = IBConfig()
    b._ib = _FakeIB()
    b._connected = True
    b._contract = SimpleNamespace(symbol="MES")
    b._active_orders = {}
    return b


def test_send_entry_places_parent_then_tranches():
    b = _bridge_with_fake()
    oid, fill = b.send_entry(quantity=2, sl=7510.25, tp=7560.75,
                             direction="long", comment="ManciniEntry")
    assert oid is not None and fill == 7530.25
    orders = [o for o, _ in b._ib.placed]
    # parent + SL-A + runner-SL + TP-A = 4 orders, STOPS PLACED FIRST
    # (protection before profit-taking)
    assert len(orders) == 4
    parent, slA, slB, tpA = orders
    assert parent.totalQuantity == 2
    assert slA.orderType == "STP" and slB.orderType == "STP"
    assert tpA.orderType == "LMT"
    # THE FIX: no exit is parent-attached (parentId resizing is the bug)
    assert tpA.parentId == 0 and slA.parentId == 0 and slB.parentId == 0
    assert tpA.totalQuantity == 1 and slA.totalQuantity == 1 and slB.totalQuantity == 1
    assert tpA.ocaGroup == slA.ocaGroup != slB.ocaGroup
    # bookkeeping carries the runner SL
    br = b._active_orders[oid]
    assert br["runner_sl"] is not None
    assert br["tp_qty"] == 1 and br["runner_qty"] == 1


def test_send_entry_single_lot_no_runner_order():
    b = _bridge_with_fake()
    oid, _ = b.send_entry(quantity=1, sl=7510.0, tp=7560.0,
                          direction="long", comment="ManciniEntry")
    orders = [o for o, _ in b._ib.placed]
    assert len(orders) == 3          # parent + TP + SL only
    assert b._active_orders[oid]["runner_sl"] is None


def test_update_stop_moves_all_working_stops():
    b = _bridge_with_fake()
    oid, _ = b.send_entry(quantity=2, sl=7510.0, tp=7560.0,
                          direction="long", comment="ManciniEntry")
    assert b.update_stop(oid, 7520.0) is True
    br = b._active_orders[oid]
    assert br["sl"].order.auxPrice == 7520.0
    assert br["runner_sl"].order.auxPrice == 7520.0


def test_update_stop_runner_only_after_tranche_a_done():
    b = _bridge_with_fake()
    oid, _ = b.send_entry(quantity=2, sl=7510.0, tp=7560.0,
                          direction="long", comment="ManciniEntry")
    br = b._active_orders[oid]
    br["sl"].orderStatus.status = "Cancelled"     # A's SL died with A's TP fill
    assert b.update_stop(oid, 7525.0) is True
    assert br["runner_sl"].order.auxPrice == 7525.0
    assert br["sl"].order.auxPrice != 7525.0      # dead order untouched


def test_partial_exit_cancels_runner_sl_too():
    b = _bridge_with_fake()
    oid, _ = b.send_entry(quantity=2, sl=7510.0, tp=7560.0,
                          direction="long", comment="ManciniEntry")
    cancelled = []
    b._ib.cancelOrder = lambda o: cancelled.append(o.orderId)
    assert b.partial_exit(oid, quantity=1, new_sl=7515.0) is True
    br = b._active_orders[oid]
    ids = {br["tp_order_id"], br["sl_order_id"], br["runner_sl_order_id"]}
    assert ids <= set(cancelled) | {None}
