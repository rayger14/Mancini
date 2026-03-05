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
from core.signals import Signal, SignalType
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
        self._prior_day_range: float = 0.0  # set by strategy at session start

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
            self._check_not_chop_zone(current_time, signal),
            self._check_not_euro_dead_zone(current_time),
            self._check_not_evening_block(current_time),
            self._check_fb_blocked_hours(signal, current_time),
            self._check_min_volatility(),
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
        # Mancini never skips a structurally valid trade due to R:R.
        # Position sizing handles risk (size_factor on the signal).
        size_info = f", size_factor={signal.position_size_factor:.2f}" if hasattr(signal, 'position_size_factor') else ""
        return RiskCheck(True, f"R:R = {signal.rr_ratio_t1:.2f}{size_info}")

    def _check_stop_distance(self, signal: Signal) -> RiskCheck:
        if signal.risk_pts <= 0:
            return RiskCheck(False, "Invalid stop distance")
        if signal.risk_pts > self.risk_params.max_stop_distance_pts:
            return RiskCheck(
                False,
                f"Stop too wide: {signal.risk_pts:.1f} pts "
                f"(max {self.risk_params.max_stop_distance_pts:.1f})",
            )
        return RiskCheck(True, f"Stop distance: {signal.risk_pts:.1f} pts")

    def _check_not_chop_zone(self, current_time: time, signal: Signal | None = None) -> RiskCheck:
        """Block entries during chop zone, but exempt BD SHORT.

        BD SHORT breakdowns can trigger during afternoon consolidation
        and the 2:12 PM ET winner on the Feb 5 sim was in the chop zone.
        """
        if self.session.in_chop_zone(current_time):
            if signal is not None and signal.signal_type == SignalType.BREAKDOWN_SHORT:
                return RiskCheck(True, "BD SHORT exempt from chop zone")
            return RiskCheck(False, "In chop zone (1PM-3PM)")
        return RiskCheck(True, "Outside chop zone")

    def _check_not_euro_dead_zone(self, current_time: time) -> RiskCheck:
        """Block entries during European open (02:00-06:00 ET).

        Backtest: PF=0.40, -291 pts over 19 trades. Avoid.
        """
        if time(2, 0) <= current_time <= time(6, 0):
            return RiskCheck(False, "European dead zone (2AM-6AM ET)")
        return RiskCheck(True, "Outside European dead zone")

    def _check_not_evening_block(self, current_time: time) -> RiskCheck:
        """Block entries during evening session (17:00-22:00 ET).

        Backtest audit: 21 phantom trades at -298 pts total. These also
        consume position slots and max_trades_per_day, blocking later
        late night entries that are profitable.
        """
        if time(17, 0) <= current_time < time(22, 0):
            return RiskCheck(False, "Evening block (17:00-22:00 ET)")
        return RiskCheck(True, "Outside evening block")

    # FB-specific blocked hours: {12, 23} = noon and 11pm.
    # Autopsy: 0% WR, -565 pts combined on 16 trades.
    _FB_BLOCKED_HOURS = frozenset({12, 23})

    def _check_fb_blocked_hours(self, signal: Signal, current_time: time) -> RiskCheck:
        """Block FB entries during poison hours (12, 23).

        These hours have 0% win rate for Failed Breakdown trades.
        Other pattern types (LR, FR, LJ) are not affected.
        """
        if signal.signal_type == SignalType.FAILED_BREAKDOWN:
            if current_time.hour in self._FB_BLOCKED_HOURS:
                return RiskCheck(False, f"FB blocked hour ({current_time.hour}:00)")
        return RiskCheck(True, "FB hour OK")

    def set_prior_day_range(self, range_pts: float) -> None:
        """Set the prior day's high-low range for volatility filtering."""
        self._prior_day_range = range_pts

    def _check_min_volatility(self) -> RiskCheck:
        """Block entries on low-volatility days (prior day range too small).

        Low-range days produce choppy price action that whipsaws both FB
        and LR entries. 2021 Q1(low vol, avg 18 pts) = 28% WR, -175 pts.
        """
        min_range = self.risk_params.min_prior_day_range_pts
        if min_range <= 0:
            return RiskCheck(True, "Volatility filter disabled")
        if self._prior_day_range <= 0:
            return RiskCheck(True, "No prior day range available")
        if self._prior_day_range < min_range:
            return RiskCheck(
                False,
                f"Low volatility: prior day range {self._prior_day_range:.1f} pts "
                f"(min {min_range:.1f})",
            )
        return RiskCheck(True, f"Prior day range: {self._prior_day_range:.1f} pts")
