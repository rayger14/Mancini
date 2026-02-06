"""Session state, daily trade limits, profit protection mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from config.settings import RiskParams, DEFAULT_RISK
from strategy.exit_manager import TradePosition, ExitPhase


class SessionState(Enum):
    """Current session trading state."""

    ACTIVE = auto()          # Actively looking for trades
    PROFIT_PROTECTION = auto()  # Had a winner, protecting profits
    DONE_FOR_DAY = auto()    # No more trades today


@dataclass
class TradeRecord:
    """Record of a completed trade."""

    entry_time: datetime
    exit_time: datetime
    entry_price: float
    avg_exit_price: float
    contracts: int
    pnl_pts: float
    pnl_dollars: float
    pattern_type: str
    exit_reason: str


@dataclass
class DaySession:
    """Tracks the state of a single trading day."""

    date: datetime
    state: SessionState = SessionState.ACTIVE
    trades: list[TradeRecord] = field(default_factory=list)
    active_position: Optional[TradePosition] = None
    daily_pnl_pts: float = 0.0
    daily_pnl_dollars: float = 0.0
    peak_pnl_pts: float = 0.0

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pts > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pts <= 0)

    @property
    def has_active_position(self) -> bool:
        return self.active_position is not None and self.active_position.is_open


class PositionManager:
    """Manages session state, trade counting, and profit protection.

    Rules:
    - Max 2 trades/day
    - First win → profit protection mode (hold runner, stop trading base)
    - First loss → one more trade allowed (risk only first trade's profits)
    - Two losses → done for the day
    - Never let a green day go red
    """

    def __init__(self, risk_params: RiskParams = DEFAULT_RISK):
        self.risk_params = risk_params
        self.session: Optional[DaySession] = None

    def start_session(self, date: datetime) -> DaySession:
        """Initialize a new trading day session."""
        self.session = DaySession(date=date)
        return self.session

    def open_position(
        self,
        position: TradePosition,
        timestamp: datetime,
        pattern_type: str,
    ) -> bool:
        """Register a new position.

        Returns
        -------
        bool
            True if the position was accepted.
        """
        if self.session is None:
            return False

        if self.session.state == SessionState.DONE_FOR_DAY:
            return False

        if self.session.has_active_position:
            return False

        if self.session.trade_count >= self.risk_params.max_trades_per_day:
            self.session.state = SessionState.DONE_FOR_DAY
            return False

        self.session.active_position = position
        return True

    def close_position(
        self,
        exit_price: float,
        timestamp: datetime,
        exit_reason: str,
        pattern_type: str = "",
    ) -> Optional[TradeRecord]:
        """Record a closed trade and update session state.

        Should be called when ALL contracts in a position are closed
        (either by stop or after runner exits).
        """
        if self.session is None or self.session.active_position is None:
            return None

        pos = self.session.active_position
        if pos.is_open:
            return None  # still has open contracts

        pnl_pts = pos.realized_pnl_pts
        pnl_dollars = pnl_pts * 50.0  # point_value

        record = TradeRecord(
            entry_time=timestamp,  # approximate
            exit_time=timestamp,
            entry_price=pos.entry_price,
            avg_exit_price=exit_price,
            contracts=pos.total_contracts,
            pnl_pts=pnl_pts,
            pnl_dollars=pnl_dollars,
            pattern_type=pattern_type,
            exit_reason=exit_reason,
        )

        self.session.trades.append(record)
        self.session.daily_pnl_pts += pnl_pts
        self.session.daily_pnl_dollars += pnl_dollars
        self.session.peak_pnl_pts = max(
            self.session.peak_pnl_pts, self.session.daily_pnl_pts
        )
        self.session.active_position = None

        # Update session state
        self._update_session_state(pnl_pts)

        return record

    def should_flatten_to_protect(self) -> bool:
        """Check if we should flatten the position to protect a green day.

        'Never let a green day go red' rule.
        """
        if not self.risk_params.never_let_green_go_red:
            return False
        if self.session is None:
            return False

        # If we've been green and now approaching breakeven
        if self.session.peak_pnl_pts > 5.0 and self.session.daily_pnl_pts <= 1.0:
            return True

        return False

    @property
    def is_profit_protection(self) -> bool:
        if self.session is None:
            return False
        return self.session.state == SessionState.PROFIT_PROTECTION

    @property
    def is_done_for_day(self) -> bool:
        if self.session is None:
            return True
        return self.session.state == SessionState.DONE_FOR_DAY

    @property
    def trades_today(self) -> int:
        if self.session is None:
            return 0
        return self.session.trade_count

    @property
    def daily_pnl_pts(self) -> float:
        if self.session is None:
            return 0.0
        return self.session.daily_pnl_pts

    def check_eod_flatten(self, current_time: 'time') -> bool:
        """Check if position should be flattened for end of day.

        Returns True if there's an active position and we should flatten.
        """
        from config.settings import DEFAULT_SESSION
        if self.session is None:
            return False
        if not self.session.has_active_position:
            return False
        return DEFAULT_SESSION.past_eod_flatten(current_time)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_session_state(self, trade_pnl_pts: float) -> None:
        """Update session state after a trade closes."""
        assert self.session is not None

        if trade_pnl_pts > 0:
            # Winner
            if self.session.state == SessionState.ACTIVE:
                self.session.state = SessionState.PROFIT_PROTECTION
        else:
            # Loser
            if self.session.losses >= 2:
                self.session.state = SessionState.DONE_FOR_DAY
            elif self.session.state == SessionState.PROFIT_PROTECTION:
                # Lost after a win — done if two trades total
                if self.session.trade_count >= 2:
                    self.session.state = SessionState.DONE_FOR_DAY
