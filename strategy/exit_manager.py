"""Level-to-level exit management (75/15/10 split with trailing stop).

Exit phases:
  Entry → full position, stop below sweep low
  Target 1 (R1) → exit 75%, move stop to breakeven + 1 tick
  Target 2 (R2) → exit ~15%, trail runner
  Runner → trail remaining 10% with dynamic tightening
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from config.settings import ExitParams, ESContractSpec, DEFAULT_EXIT, DEFAULT_CONTRACT


class ExitPhase(Enum):
    """Current phase of the exit management."""

    INITIAL = auto()      # Full position, initial stop
    AFTER_T1 = auto()     # 75% exited, stop at breakeven
    AFTER_T2 = auto()     # Runner only, trailing stop
    CLOSED = auto()       # All positions closed


@dataclass
class ExitAction:
    """Describes an exit action to take."""

    contracts_to_close: int
    exit_price: float
    new_stop: float
    new_phase: ExitPhase
    reason: str


@dataclass
class TradePosition:
    """Tracks an open trade's position and exit state."""

    entry_price: float
    stop_price: float
    target_1: float
    target_2: float
    total_contracts: int
    remaining_contracts: int
    phase: ExitPhase = ExitPhase.INITIAL
    highest_price_since_entry: float = 0.0
    realized_pnl_pts: float = 0.0

    def __post_init__(self):
        self.highest_price_since_entry = self.entry_price

    @property
    def is_open(self) -> bool:
        return self.remaining_contracts > 0 and self.phase != ExitPhase.CLOSED


class ExitManager:
    """Manages the 75/15/10 exit strategy for a single trade."""

    def __init__(
        self,
        params: ExitParams = DEFAULT_EXIT,
        contract: ESContractSpec = DEFAULT_CONTRACT,
    ):
        self.params = params
        self.contract = contract

    def create_position(
        self,
        entry_price: float,
        stop_price: float,
        target_1: float,
        target_2: float,
        contracts: int,
    ) -> TradePosition:
        """Create a new trade position."""
        return TradePosition(
            entry_price=entry_price,
            stop_price=stop_price,
            target_1=target_1,
            target_2=target_2,
            total_contracts=contracts,
            remaining_contracts=contracts,
        )

    def update(
        self, position: TradePosition, high: float, low: float, close: float
    ) -> Optional[ExitAction]:
        """Evaluate current bar against position and return an exit action if triggered.

        Note: only one exit action per bar (VectorBT constraint).
        Priority: stop loss > T1 > T2 > trail.

        Parameters
        ----------
        position : TradePosition
            Current position state.
        high, low, close : float
            Current bar OHLC.

        Returns
        -------
        ExitAction or None
        """
        if not position.is_open:
            return None

        # Update tracking
        if high > position.highest_price_since_entry:
            position.highest_price_since_entry = high

        # 1. Check stop loss (highest priority)
        if low <= position.stop_price:
            return self._stop_out(position)

        # 2. Phase-specific logic
        if position.phase == ExitPhase.INITIAL:
            return self._check_t1(position, high, close)
        elif position.phase == ExitPhase.AFTER_T1:
            return self._check_t2(position, high, close)
        elif position.phase == ExitPhase.AFTER_T2:
            return self._check_trail(position, high, low, close)

        return None

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _stop_out(self, position: TradePosition) -> ExitAction:
        """Close entire position at stop."""
        contracts = position.remaining_contracts
        pnl = (position.stop_price - position.entry_price) * contracts
        position.realized_pnl_pts += pnl
        position.remaining_contracts = 0
        position.phase = ExitPhase.CLOSED
        return ExitAction(
            contracts_to_close=contracts,
            exit_price=position.stop_price,
            new_stop=0.0,
            new_phase=ExitPhase.CLOSED,
            reason="Stop loss hit",
        )

    def _check_t1(
        self, position: TradePosition, high: float, close: float
    ) -> Optional[ExitAction]:
        """Check if Target 1 is reached → exit 75%."""
        if high >= position.target_1:
            contracts_to_exit = round(
                position.total_contracts * self.params.t1_exit_fraction
            )
            contracts_to_exit = min(contracts_to_exit, position.remaining_contracts)

            if contracts_to_exit <= 0:
                return None

            pnl = (position.target_1 - position.entry_price) * contracts_to_exit
            position.realized_pnl_pts += pnl
            position.remaining_contracts -= contracts_to_exit

            # Move stop to breakeven + 1 tick
            be_stop = position.entry_price + (
                self.params.breakeven_buffer_ticks * self.contract.tick_size
            )
            position.stop_price = be_stop
            position.phase = ExitPhase.AFTER_T1

            return ExitAction(
                contracts_to_close=contracts_to_exit,
                exit_price=position.target_1,
                new_stop=be_stop,
                new_phase=ExitPhase.AFTER_T1,
                reason=f"Target 1 hit ({position.target_1:.2f})",
            )
        return None

    def _check_t2(
        self, position: TradePosition, high: float, close: float
    ) -> Optional[ExitAction]:
        """Check if Target 2 is reached → exit all but runner."""
        if high >= position.target_2:
            # Keep 1 contract as runner
            runner_contracts = max(
                1, round(position.total_contracts * self.params.runner_fraction)
            )
            contracts_to_exit = position.remaining_contracts - runner_contracts

            if contracts_to_exit <= 0:
                # Already at runner size, switch to trailing
                position.phase = ExitPhase.AFTER_T2
                trail_stop = self._compute_trail_stop(position, high)
                position.stop_price = trail_stop
                return None

            pnl = (position.target_2 - position.entry_price) * contracts_to_exit
            position.realized_pnl_pts += pnl
            position.remaining_contracts -= contracts_to_exit

            trail_stop = self._compute_trail_stop(position, high)
            position.stop_price = trail_stop
            position.phase = ExitPhase.AFTER_T2

            return ExitAction(
                contracts_to_close=contracts_to_exit,
                exit_price=position.target_2,
                new_stop=trail_stop,
                new_phase=ExitPhase.AFTER_T2,
                reason=f"Target 2 hit ({position.target_2:.2f})",
            )
        return None

    def _check_trail(
        self, position: TradePosition, high: float, low: float, close: float
    ) -> Optional[ExitAction]:
        """Update trailing stop for the runner."""
        new_trail = self._compute_trail_stop(position, high)
        if new_trail > position.stop_price:
            position.stop_price = new_trail
        return None  # trail update, no exit action (stop checked separately)

    def _compute_trail_stop(self, position: TradePosition, high: float) -> float:
        """Compute dynamic trailing stop distance based on profit."""
        profit = high - position.entry_price
        trail_pts = self.params.trailing_stop_pts  # default 4.0

        # Tighten trail as profit grows
        for threshold, tighter_trail in self.params.trailing_tighten_thresholds:
            if profit >= threshold:
                trail_pts = tighter_trail

        return high - trail_pts
