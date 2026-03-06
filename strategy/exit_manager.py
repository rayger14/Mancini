"""Level-to-level exit management (75/15/10 split with prior-day-low runner trail).

Mancini's actual method (from 500 Substack posts):
  Entry → full position, stop below sweep low
  Target 1 (R1) → exit 75%, move stop to several pts under breakeven
  Target 2 (R2) → exit 15%, leave 10% runner
  Runner → trail under PRIOR DAY'S RTH low (updated once at EOD)
  Runner carries overnight/multi-day until prior day low is lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from config.settings import ExitParams, ESContractSpec, DEFAULT_EXIT, DEFAULT_CONTRACT


class ExitPhase(Enum):
    """Current phase of the exit management."""

    INITIAL = auto()      # Full position, initial stop
    AFTER_T1 = auto()     # 75% exited, stop near breakeven
    AFTER_T2 = auto()     # Runner only, trailing under prior day's low
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
    lowest_price_since_entry: float = float("inf")
    realized_pnl_pts: float = 0.0
    direction: str = "long"  # "long" or "short"
    # Prior day's RTH low — set by ib_runner at EOD for runner trailing.
    # When set, the runner stop trails under this level instead of fixed pts.
    prior_day_low: float = 0.0
    # Prior day's RTH high — for short runner trailing
    prior_day_high: float = 0.0
    # Track whether T1/T2 were hit (for accurate exit_reason on final close)
    t1_hit: bool = False
    t2_hit: bool = False

    def __post_init__(self):
        self.highest_price_since_entry = self.entry_price
        self.lowest_price_since_entry = self.entry_price

    @property
    def is_open(self) -> bool:
        return self.remaining_contracts > 0 and self.phase != ExitPhase.CLOSED


class ExitManager:
    """Manages the 75/15/10 exit strategy for a single trade.

    After T1 hit, stop moves to several pts under breakeven.
    After T2 hit (or at EOD), runner trails under prior day's low.
    """

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
        direction: str = "long",
    ) -> TradePosition:
        """Create a new trade position."""
        return TradePosition(
            entry_price=entry_price,
            stop_price=stop_price,
            target_1=target_1,
            target_2=target_2,
            total_contracts=contracts,
            remaining_contracts=contracts,
            direction=direction,
        )

    def update_prior_day_low(self, position: TradePosition, prior_day_low: float) -> Optional[ExitAction]:
        """Update the runner's trailing stop to under the prior day's low.

        Called by ib_runner at EOD. Mancini: "Move stop under daily low at
        end of day." This only applies to runners (AFTER_T1 or AFTER_T2).
        """
        if not position.is_open:
            return None
        if position.phase not in (ExitPhase.AFTER_T1, ExitPhase.AFTER_T2):
            return None

        position.prior_day_low = prior_day_low
        buffer = self.params.runner_prior_day_low_buffer_pts

        if position.direction == "long":
            new_stop = prior_day_low - buffer
            # Only ratchet up, never down
            if new_stop > position.stop_price:
                position.stop_price = new_stop
                return ExitAction(
                    contracts_to_close=0,
                    exit_price=0.0,
                    new_stop=new_stop,
                    new_phase=position.phase,
                    reason=f"Runner trail to prior day low {prior_day_low:.2f}",
                )
        else:
            new_stop = prior_day_low + buffer  # For shorts, trail above prior day's HIGH
            # For short runners, we'd use prior_day_high — but use prior_day_low
            # as fallback if high not set
            if position.prior_day_high > 0:
                new_stop = position.prior_day_high + buffer
            if new_stop < position.stop_price:
                position.stop_price = new_stop
                return ExitAction(
                    contracts_to_close=0,
                    exit_price=0.0,
                    new_stop=new_stop,
                    new_phase=position.phase,
                    reason=f"Runner trail to prior day high {new_stop:.2f}",
                )
        return None

    def update(
        self, position: TradePosition, high: float, low: float, close: float
    ) -> Optional[ExitAction]:
        """Evaluate current bar against position and return an exit action if triggered.

        Priority: stop loss > T1 > T2 > trail.
        Direction-aware: long checks low<=stop, short checks high>=stop.
        """
        if not position.is_open:
            return None

        # Update tracking
        if high > position.highest_price_since_entry:
            position.highest_price_since_entry = high
        if low < position.lowest_price_since_entry:
            position.lowest_price_since_entry = low

        # 1. Check stop loss (highest priority)
        if position.direction == "short":
            if high >= position.stop_price:
                return self._stop_out(position)
        else:
            if low <= position.stop_price:
                return self._stop_out(position)

        # 2. Phase-specific logic
        if position.direction == "short":
            if position.phase == ExitPhase.INITIAL:
                return self._check_t1_short(position, low, close)
            elif position.phase == ExitPhase.AFTER_T1:
                return self._check_t2_short(position, low, close)
            elif position.phase == ExitPhase.AFTER_T2:
                return self._check_trail_short(position, high, low, close)
        else:
            if position.phase == ExitPhase.INITIAL:
                return self._check_t1(position, high, close)
            elif position.phase == ExitPhase.AFTER_T1:
                return self._check_t2(position, high, close)
            elif position.phase == ExitPhase.AFTER_T2:
                return self._check_trail(position, high, low, close)

        return None

    # ------------------------------------------------------------------
    # Phase handlers (long)
    # ------------------------------------------------------------------

    def _stop_out(self, position: TradePosition) -> ExitAction:
        """Close entire position at stop."""
        contracts = position.remaining_contracts
        if position.direction == "short":
            pnl = (position.entry_price - position.stop_price) * contracts
        else:
            pnl = (position.stop_price - position.entry_price) * contracts
        position.realized_pnl_pts += pnl
        position.remaining_contracts = 0

        if position.phase in (ExitPhase.AFTER_T1, ExitPhase.AFTER_T2):
            if position.t2_hit:
                reason = "Runner stopped after T1+T2"
            elif position.t1_hit:
                reason = "Runner stopped after T1"
            else:
                reason = "Trailing stop hit"
        else:
            reason = "Stop loss hit"

        position.phase = ExitPhase.CLOSED
        return ExitAction(
            contracts_to_close=contracts,
            exit_price=position.stop_price,
            new_stop=0.0,
            new_phase=ExitPhase.CLOSED,
            reason=reason,
        )

    def _check_t1(
        self, position: TradePosition, high: float, close: float
    ) -> Optional[ExitAction]:
        """Check if Target 1 is reached → exit 75%, stop to under breakeven.

        Mancini: "Lock in 75% profits at first level up. Move stop several
        points under break-even. Will not let the entire trade go back red."
        """
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

            # Mancini: "several points under break-even"
            # breakeven_buffer_pts is negative (e.g., -3.0) = below breakeven
            new_stop = position.entry_price + self.params.breakeven_buffer_pts
            # If prior_day_low is set and is lower, use that (wider stop for runner)
            if position.prior_day_low > 0:
                pdl_stop = position.prior_day_low - self.params.runner_prior_day_low_buffer_pts
                new_stop = min(new_stop, pdl_stop)  # use the wider (lower) stop

            position.stop_price = new_stop
            position.phase = ExitPhase.AFTER_T1
            position.t1_hit = True

            return ExitAction(
                contracts_to_close=contracts_to_exit,
                exit_price=position.target_1,
                new_stop=new_stop,
                new_phase=ExitPhase.AFTER_T1,
                reason=f"Target 1 hit ({position.target_1:.2f})",
            )
        return None

    def _check_t2(
        self, position: TradePosition, high: float, close: float
    ) -> Optional[ExitAction]:
        """Check if Target 2 is reached → exit down to runner.

        Mancini: "lock in more at next level up, leave 10% to go."
        After T2, runner trails under prior day's low (set at EOD).
        During the day, use fallback trailing if prior_day_low not yet set.
        """
        # Fallback intraday trail (ratchet up) if no prior_day_low set
        # For runners (AFTER_T1), use base trailing_stop_pts without aggressive tightening
        if position.prior_day_low <= 0:
            new_trail = position.highest_price_since_entry - self.params.trailing_stop_pts
            if new_trail > position.stop_price:
                position.stop_price = new_trail

        if high >= position.target_2:
            runner_contracts = max(
                1, round(position.total_contracts * self.params.runner_fraction)
            )
            contracts_to_exit = position.remaining_contracts - runner_contracts

            if contracts_to_exit <= 0:
                position.phase = ExitPhase.AFTER_T2
                return None

            pnl = (position.target_2 - position.entry_price) * contracts_to_exit
            position.realized_pnl_pts += pnl
            position.remaining_contracts -= contracts_to_exit

            # Runner stop: use prior_day_low if available, else fallback trail
            if position.prior_day_low > 0:
                new_stop = position.prior_day_low - self.params.runner_prior_day_low_buffer_pts
                new_stop = max(new_stop, position.stop_price)  # ratchet up only
            else:
                new_stop = self._compute_trail_stop(position, high)
                new_stop = max(new_stop, position.stop_price)

            position.stop_price = new_stop
            position.phase = ExitPhase.AFTER_T2
            position.t2_hit = True

            return ExitAction(
                contracts_to_close=contracts_to_exit,
                exit_price=position.target_2,
                new_stop=new_stop,
                new_phase=ExitPhase.AFTER_T2,
                reason=f"Target 2 hit ({position.target_2:.2f})",
            )
        return None

    def _check_trail(
        self, position: TradePosition, high: float, low: float, close: float
    ) -> Optional[ExitAction]:
        """Update trailing stop for the runner.

        If prior_day_low is set (by EOD hook), runner trails under it.
        Otherwise falls back to fixed-distance trailing.
        """
        if position.prior_day_low > 0:
            # Mancini: trail under prior day's low — only updated at EOD
            # During the session, stop stays fixed. No per-bar trail.
            pass
        else:
            # Fallback: intraday trailing with base trailing_stop_pts (no tightening for runners)
            new_trail = position.highest_price_since_entry - self.params.trailing_stop_pts
            if new_trail > position.stop_price:
                position.stop_price = new_trail
        return None

    def _compute_trail_stop(self, position: TradePosition, high: float) -> float:
        """Compute fallback trailing stop distance based on profit (long)."""
        profit = high - position.entry_price
        trail_pts = self.params.trailing_stop_pts

        for threshold, tighter_trail in self.params.trailing_tighten_thresholds:
            if profit >= threshold:
                trail_pts = tighter_trail

        return high - trail_pts

    # ------------------------------------------------------------------
    # Short-side phase handlers
    # ------------------------------------------------------------------

    def _check_t1_short(
        self, position: TradePosition, low: float, close: float
    ) -> Optional[ExitAction]:
        """Check if Target 1 is reached for short (price drops to target)."""
        if low <= position.target_1:
            contracts_to_exit = round(
                position.total_contracts * self.params.t1_exit_fraction
            )
            contracts_to_exit = min(contracts_to_exit, position.remaining_contracts)

            if contracts_to_exit <= 0:
                return None

            pnl = (position.entry_price - position.target_1) * contracts_to_exit
            position.realized_pnl_pts += pnl
            position.remaining_contracts -= contracts_to_exit

            # Stop to several pts above breakeven (for shorts, above = tighter)
            new_stop = position.entry_price - self.params.breakeven_buffer_pts
            if position.prior_day_high > 0:
                pdh_stop = position.prior_day_high + self.params.runner_prior_day_low_buffer_pts
                new_stop = max(new_stop, pdh_stop)  # use the wider (higher) stop for short

            position.stop_price = new_stop
            position.phase = ExitPhase.AFTER_T1
            position.t1_hit = True

            return ExitAction(
                contracts_to_close=contracts_to_exit,
                exit_price=position.target_1,
                new_stop=new_stop,
                new_phase=ExitPhase.AFTER_T1,
                reason=f"Target 1 hit ({position.target_1:.2f})",
            )
        return None

    def _check_t2_short(
        self, position: TradePosition, low: float, close: float
    ) -> Optional[ExitAction]:
        """Check if Target 2 is reached for short."""
        # Fallback intraday trail (ratchet down) if no prior_day_high set
        # For runners (AFTER_T1), use base trailing_stop_pts without aggressive tightening
        if position.prior_day_high <= 0:
            new_trail = position.lowest_price_since_entry + self.params.trailing_stop_pts
            if new_trail < position.stop_price:
                position.stop_price = new_trail

        if low <= position.target_2:
            runner_contracts = max(
                1, round(position.total_contracts * self.params.runner_fraction)
            )
            contracts_to_exit = position.remaining_contracts - runner_contracts

            if contracts_to_exit <= 0:
                position.phase = ExitPhase.AFTER_T2
                return None

            pnl = (position.entry_price - position.target_2) * contracts_to_exit
            position.realized_pnl_pts += pnl
            position.remaining_contracts -= contracts_to_exit

            if position.prior_day_high > 0:
                new_stop = position.prior_day_high + self.params.runner_prior_day_low_buffer_pts
                new_stop = min(new_stop, position.stop_price)
            else:
                new_stop = self._compute_trail_stop_short(position, low)
                new_stop = min(new_stop, position.stop_price)

            position.stop_price = new_stop
            position.phase = ExitPhase.AFTER_T2
            position.t2_hit = True

            return ExitAction(
                contracts_to_close=contracts_to_exit,
                exit_price=position.target_2,
                new_stop=new_stop,
                new_phase=ExitPhase.AFTER_T2,
                reason=f"Target 2 hit ({position.target_2:.2f})",
            )
        return None

    def _check_trail_short(
        self, position: TradePosition, high: float, low: float, close: float
    ) -> Optional[ExitAction]:
        """Update trailing stop for short runner."""
        if position.prior_day_high > 0:
            pass  # Trail updated at EOD only
        else:
            # Fallback: intraday trailing with base trailing_stop_pts (no tightening for runners)
            new_trail = position.lowest_price_since_entry + self.params.trailing_stop_pts
            if new_trail < position.stop_price:
                position.stop_price = new_trail
        return None

    def _compute_trail_stop_short(self, position: TradePosition, low: float) -> float:
        """Compute fallback trailing stop for short (above lowest low)."""
        profit = position.entry_price - low
        trail_pts = self.params.trailing_stop_pts

        for threshold, tighter_trail in self.params.trailing_tighten_thresholds:
            if profit >= threshold:
                trail_pts = tighter_trail

        return low + trail_pts
