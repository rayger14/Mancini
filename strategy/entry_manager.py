"""Entry decisions: session rules, time windows, position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

from config.settings import (
    SessionTimes,
    ExitParams,
    RiskParams,
    DEFAULT_SESSION,
    DEFAULT_EXIT,
    DEFAULT_RISK,
)
from core.signals import Signal


@dataclass
class EntryDecision:
    """The result of evaluating whether to enter a trade."""

    should_enter: bool
    signal: Optional[Signal] = None
    contracts: int = 0
    reason: str = ""
    entry_price: float = 0.0
    stop_price: float = 0.0


class EntryManager:
    """Evaluates signals against session rules and decides entries."""

    def __init__(
        self,
        session: SessionTimes = DEFAULT_SESSION,
        exit_params: ExitParams = DEFAULT_EXIT,
        risk_params: RiskParams = DEFAULT_RISK,
    ):
        self.session = session
        self.exit_params = exit_params
        self.risk_params = risk_params

    def evaluate(
        self,
        signal: Signal,
        current_time: time,
        trades_today: int,
        is_in_profit_protection: bool,
        daily_pnl_pts: float,
    ) -> EntryDecision:
        """Evaluate a signal and decide whether to enter.

        Parameters
        ----------
        signal : Signal
            The trade signal to evaluate.
        current_time : time
            Current time of day (Eastern).
        trades_today : int
            Number of trades already taken today.
        is_in_profit_protection : bool
            Whether we're in profit protection mode (had a winner).
        daily_pnl_pts : float
            Current day's P&L in points.

        Returns
        -------
        EntryDecision
        """
        # Check max trades per day
        if trades_today >= self.risk_params.max_trades_per_day:
            return EntryDecision(
                should_enter=False,
                reason=f"Max trades reached ({trades_today}/{self.risk_params.max_trades_per_day})",
            )

        # Check if past EOD flatten time
        if self.session.past_eod_flatten(current_time):
            return EntryDecision(
                should_enter=False,
                reason="Past EOD flatten time (3:55 PM)",
            )

        # Check if in chop zone
        if self.session.in_chop_zone(current_time):
            return EntryDecision(
                should_enter=False,
                reason="In chop zone (11AM-2PM)",
            )

        # Check profit protection: only risk first trade's profits
        if is_in_profit_protection and daily_pnl_pts <= 0:
            return EntryDecision(
                should_enter=False,
                reason="Profit protection: no profits to risk",
            )

        # Check daily loss limit
        if daily_pnl_pts <= -self.risk_params.max_daily_loss_pts:
            return EntryDecision(
                should_enter=False,
                reason=f"Daily loss limit reached ({daily_pnl_pts:.1f} pts)",
            )

        # Calculate position size
        contracts = self._size_position(signal, daily_pnl_pts, is_in_profit_protection)

        if contracts <= 0:
            return EntryDecision(
                should_enter=False,
                reason="Position size is zero",
            )

        # Prefer preferred windows but allow trades outside
        in_preferred = self.session.in_preferred_window(current_time)
        reason = "In preferred window" if in_preferred else "Outside preferred window"

        return EntryDecision(
            should_enter=True,
            signal=signal,
            contracts=contracts,
            reason=reason,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
        )

    def _size_position(
        self,
        signal: Signal,
        daily_pnl_pts: float,
        is_in_profit_protection: bool,
    ) -> int:
        """Determine number of contracts.

        Applies position_size_factor from the signal (e.g., Mancini stop-based
        sizing, Mode 1 reduction). In profit protection mode, size down to
        risk only current profits.
        """
        base_contracts = self.exit_params.default_contracts

        if is_in_profit_protection and daily_pnl_pts > 0:
            # Risk only profits from first trade
            risk_per_contract = signal.risk_pts
            if risk_per_contract <= 0:
                return 0
            max_contracts = int(daily_pnl_pts / risk_per_contract)
            contracts = min(max(max_contracts, 1), base_contracts)
        else:
            contracts = min(base_contracts, self.risk_params.max_position_contracts)

        # Apply signal-level size factor (stop-based sizing, Mode 1 reduction, etc.)
        size_factor = getattr(signal, "position_size_factor", 1.0)
        if size_factor < 1.0:
            contracts = max(1, int(contracts * size_factor))

        return contracts
