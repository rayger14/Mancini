"""Broker integration: paper and live order execution."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from loguru import logger

from config.settings import ESContractSpec, DEFAULT_CONTRACT


class OrderSide(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


@dataclass
class Order:
    """An order to be submitted to the broker."""

    side: OrderSide
    contracts: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tag: str = ""  # for tracking (e.g., "T1_exit", "stop_loss")


@dataclass
class Fill:
    """Confirmation of an executed order."""

    order: Order
    fill_price: float
    fill_time: datetime
    commission: float = 0.0

    @property
    def side(self) -> OrderSide:
        return self.order.side

    @property
    def contracts(self) -> int:
        return self.order.contracts


class OrderExecutor(abc.ABC):
    """Abstract base class for order execution."""

    @abc.abstractmethod
    def submit_order(self, order: Order) -> Optional[Fill]:
        """Submit an order and return fill (or None if rejected)."""
        ...

    @abc.abstractmethod
    def cancel_all(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        ...

    @abc.abstractmethod
    def get_position(self) -> int:
        """Get current net position (positive = long)."""
        ...


class PaperOrderExecutor(OrderExecutor):
    """Simulated order execution for paper trading.

    Fills immediately at last known price (market orders)
    or at limit/stop price when triggered.
    """

    def __init__(
        self,
        contract: ESContractSpec = DEFAULT_CONTRACT,
        commission_per_contract: float = 2.25,
    ):
        self.contract = contract
        self.commission_per_contract = commission_per_contract
        self._position: int = 0
        self._last_price: float = 0.0
        self._fills: list[Fill] = []
        self._pending_orders: list[Order] = []
        self._pnl: float = 0.0
        self._avg_entry: float = 0.0

    def set_price(self, price: float) -> list[Fill]:
        """Update last known price and check for triggered stop/limit orders.

        Returns any fills from triggered orders.
        """
        self._last_price = price
        triggered: list[Fill] = []

        remaining = []
        for order in self._pending_orders:
            fill = self._try_fill(order, price)
            if fill is not None:
                triggered.append(fill)
            else:
                remaining.append(order)
        self._pending_orders = remaining

        return triggered

    def submit_order(self, order: Order) -> Optional[Fill]:
        """Submit an order. Market orders fill immediately."""
        if order.order_type == OrderType.MARKET:
            return self._execute_fill(order, self._last_price)

        # Pending order (stop/limit)
        self._pending_orders.append(order)
        logger.debug(
            f"Paper: Pending {order.order_type.name} {order.side.name} "
            f"{order.contracts} @ {order.stop_price or order.limit_price}"
        )
        return None

    def cancel_all(self) -> int:
        count = len(self._pending_orders)
        self._pending_orders.clear()
        return count

    def get_position(self) -> int:
        return self._position

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def realized_pnl(self) -> float:
        return self._pnl

    def _try_fill(self, order: Order, price: float) -> Optional[Fill]:
        """Check if a pending order should be triggered at the given price."""
        if order.order_type == OrderType.STOP:
            if order.side == OrderSide.SELL and price <= order.stop_price:
                return self._execute_fill(order, order.stop_price)
            if order.side == OrderSide.BUY and price >= order.stop_price:
                return self._execute_fill(order, order.stop_price)

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and price <= order.limit_price:
                return self._execute_fill(order, order.limit_price)
            if order.side == OrderSide.SELL and price >= order.limit_price:
                return self._execute_fill(order, order.limit_price)

        return None

    def _execute_fill(self, order: Order, fill_price: float) -> Fill:
        """Execute a fill and update position/PnL."""
        commission = self.commission_per_contract * order.contracts

        # Update position and track P&L
        if order.side == OrderSide.BUY:
            if self._position <= 0 and order.contracts > abs(self._position):
                # Opening or adding to long
                self._avg_entry = fill_price
            self._position += order.contracts
        else:  # SELL
            if self._position > 0:
                pnl = (fill_price - self._avg_entry) * min(
                    order.contracts, self._position
                )
                self._pnl += pnl * self.contract.point_value - commission
            self._position -= order.contracts

        fill = Fill(
            order=order,
            fill_price=fill_price,
            fill_time=datetime.utcnow(),
            commission=commission,
        )
        self._fills.append(fill)

        logger.info(
            f"Paper FILL: {order.side.name} {order.contracts} @ {fill_price:.2f} "
            f"[{order.tag}] pos={self._position}"
        )
        return fill


class LiveOrderExecutor(OrderExecutor):
    """Placeholder for live broker integration.

    To be implemented with NautilusTrader or broker-specific API.
    """

    def __init__(self):
        logger.warning("LiveOrderExecutor is a placeholder - not connected to a broker")
        self._position = 0

    def submit_order(self, order: Order) -> Optional[Fill]:
        raise NotImplementedError(
            "LiveOrderExecutor requires broker integration. "
            "Use PaperOrderExecutor for testing."
        )

    def cancel_all(self) -> int:
        raise NotImplementedError("LiveOrderExecutor not implemented")

    def get_position(self) -> int:
        return self._position
