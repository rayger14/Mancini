"""Safety net: daily loss limits, position validation, entry screening."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

from config.settings import (
    RiskParams,
    SessionTimes,
    ESContractSpec,
    DEFAULT_RISK,
    DEFAULT_SESSION,
    DEFAULT_CONTRACT,
)
from core.signals import Signal
from strategy.position_manager import PositionManager


@dataclass
class RiskCheck:
    """Result of a risk check."""

    passed: bool
    reason: str


class RiskManager:
    """Final safety net before entries. Validates everything."""

    def __init__(
        self,
        risk_params: RiskParams = DEFAULT_RISK,
        session: SessionTimes = DEFAULT_SESSION,
        contract: ESContractSpec = DEFAULT_CONTRACT,
    ):
        self.risk_params = risk_params
        self.session = session
        self.contract = contract

    def validate_entry(
        self,
        signal: Signal,
        current_time: time,
        position_manager: PositionManager,
    ) -> RiskCheck:
        """Run all risk checks before allowing an entry.

        Parameters
        ----------
        signal : Signal
            Proposed trade signal.
        current_time : time
            Current time of day.
        position_manager : PositionManager
            Current session state.

        Returns
        -------
        RiskCheck
        """
        checks = [
            self._check_session_active(position_manager),
            self._check_daily_loss(position_manager),
            self._check_no_open_position(position_manager),
            self._check_trade_count(position_manager),
            self._check_risk_reward(signal),
            self._check_stop_distance(signal),
            self._check_not_chop_zone(current_time),
        ]

        for check in checks:
            if not check.passed:
                return check

        return RiskCheck(passed=True, reason="All checks passed")

    def validate_position_size(
        self, contracts: int, signal: Signal
    ) -> RiskCheck:
        """Validate the position size."""
        if contracts <= 0:
            return RiskCheck(False, "Contracts must be > 0")

        if contracts > self.risk_params.max_position_contracts:
            return RiskCheck(
                False,
                f"Contracts {contracts} exceeds max {self.risk_params.max_position_contracts}",
            )

        max_risk_dollars = (
            signal.risk_pts
            * self.contract.point_value
            * contracts
        )

        max_daily_loss_dollars = (
            self.risk_params.max_daily_loss_pts * self.contract.point_value
        )

        if max_risk_dollars > max_daily_loss_dollars:
            return RiskCheck(
                False,
                f"Risk ${max_risk_dollars:.0f} exceeds daily loss limit ${max_daily_loss_dollars:.0f}",
            )

        return RiskCheck(True, "Position size OK")

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_session_active(self, pm: PositionManager) -> RiskCheck:
        if pm.is_done_for_day:
            return RiskCheck(False, "Session is done for the day")
        return RiskCheck(True, "Session active")

    def _check_daily_loss(self, pm: PositionManager) -> RiskCheck:
        if pm.daily_pnl_pts <= -self.risk_params.max_daily_loss_pts:
            return RiskCheck(
                False,
                f"Daily loss limit reached: {pm.daily_pnl_pts:.1f} pts",
            )
        return RiskCheck(True, "Within daily loss limit")

    def _check_no_open_position(self, pm: PositionManager) -> RiskCheck:
        if pm.session and pm.session.has_active_position:
            return RiskCheck(False, "Already have an open position")
        return RiskCheck(True, "No open position")

    def _check_trade_count(self, pm: PositionManager) -> RiskCheck:
        if pm.trades_today >= self.risk_params.max_trades_per_day:
            return RiskCheck(
                False,
                f"Max trades reached: {pm.trades_today}",
            )
        return RiskCheck(True, "Trade count OK")

    def _check_risk_reward(self, signal: Signal) -> RiskCheck:
        if signal.rr_ratio_t1 < 1.0:
            return RiskCheck(
                False,
                f"R:R too low: {signal.rr_ratio_t1:.2f}",
            )
        return RiskCheck(True, f"R:R = {signal.rr_ratio_t1:.2f}")

    def _check_stop_distance(self, signal: Signal) -> RiskCheck:
        if signal.risk_pts <= 0:
            return RiskCheck(False, "Invalid stop distance")
        if signal.risk_pts > 10.0:
            return RiskCheck(
                False,
                f"Stop too wide: {signal.risk_pts:.1f} pts",
            )
        return RiskCheck(True, f"Stop distance: {signal.risk_pts:.1f} pts")

    def _check_not_chop_zone(self, current_time: time) -> RiskCheck:
        if self.session.in_chop_zone(current_time):
            return RiskCheck(False, "In chop zone (11AM-2PM)")
        return RiskCheck(True, "Outside chop zone")
