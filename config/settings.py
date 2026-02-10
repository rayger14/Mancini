"""ES contract specifications, risk parameters, and session times."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time


@dataclass(frozen=True)
class ESContractSpec:
    """E-mini S&P 500 futures contract specifications."""

    symbol: str = "ES"
    tick_size: float = 0.25
    tick_value: float = 12.50  # USD per tick
    point_value: float = 50.0  # USD per point (4 ticks)
    margin_initial: float = 12_650.0  # Approximate initial margin
    margin_maintenance: float = 11_500.0
    exchange: str = "CME"

    def ticks_to_points(self, ticks: int) -> float:
        return ticks * self.tick_size

    def points_to_dollars(self, points: float, contracts: int = 1) -> float:
        return points * self.point_value * contracts


@dataclass(frozen=True)
class SessionTimes:
    """RTH and key intraday windows (US/Eastern)."""

    rth_open: time = field(default_factory=lambda: time(9, 30))
    rth_close: time = field(default_factory=lambda: time(16, 0))
    # Globex session
    globex_open: time = field(default_factory=lambda: time(18, 0))  # prior day
    globex_close: time = field(default_factory=lambda: time(17, 0))
    # Preferred trading windows
    morning_window_start: time = field(default_factory=lambda: time(9, 30))
    morning_window_end: time = field(default_factory=lambda: time(11, 0))
    afternoon_window_start: time = field(default_factory=lambda: time(15, 0))
    afternoon_window_end: time = field(default_factory=lambda: time(15, 55))
    # EOD flatten
    eod_flatten_time: time = field(default_factory=lambda: time(15, 55))
    # Dead zone (avoid) - only trade morning (9:30-11) and close (15:00-15:55)
    chop_zone_start: time = field(default_factory=lambda: time(13, 0))
    chop_zone_end: time = field(default_factory=lambda: time(15, 0))

    def in_preferred_window(self, t: time) -> bool:
        """Check if time is within a preferred trading window."""
        in_morning = self.morning_window_start <= t <= self.morning_window_end
        in_afternoon = self.afternoon_window_start <= t <= self.afternoon_window_end
        return in_morning or in_afternoon

    def in_chop_zone(self, t: time) -> bool:
        """Check if time is in the 11AM-2PM chop zone."""
        return self.chop_zone_start <= t <= self.chop_zone_end

    def in_rth(self, t: time) -> bool:
        """Check if time is within Regular Trading Hours."""
        return self.rth_open <= t <= self.rth_close

    def past_eod_flatten(self, t: time) -> bool:
        """Check if time is past the EOD flatten deadline."""
        return t >= self.eod_flatten_time


@dataclass(frozen=True)
class ElevatorParams:
    """Parameters for Elevator Down detection."""

    min_velocity_pts_per_min: float = 0.75  # catch broader selloffs
    velocity_window_bars: int = 5  # 5-bar rolling window (1-min bars = 5 min)
    min_levels_broken: int = 2  # require breaking 2 support levels
    completion_velocity_ratio: float = 0.5  # velocity must drop to this fraction
    higher_low_lookback: int = 4  # bars to confirm higher low


@dataclass(frozen=True)
class ExitParams:
    """Exit management parameters (75/15/10 split)."""

    default_contracts: int = 4
    t1_exit_fraction: float = 1.0  # 100% at first target (no runner)
    t2_exit_fraction: float = 0.0  # not used with 100% T1 exit
    runner_fraction: float = 0.0  # no runner
    initial_stop_buffer_pts: float = 4.5  # matches fb_stop_buffer_pts
    breakeven_buffer_ticks: int = 1  # 1 tick above breakeven after T1
    trailing_stop_pts: float = 7.0  # wider trailing for any remaining contracts
    trailing_tighten_thresholds: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (10.0, 3.0),  # after 10 pts profit, trail to 3 pts
            (15.0, 2.0),  # after 15 pts profit, trail to 2 pts
        ]
    )


@dataclass(frozen=True)
class StrategyParams:
    """Core strategy parameters."""

    # Significant low detection
    swing_low_order: int = 30  # argrelextrema order (30 bars = 30 min on 1-min)
    cluster_proximity_pts: float = 1.0  # points within which lows form a cluster
    cluster_min_touches: int = 3  # minimum touches for a cluster
    multi_hour_rally_min_pts: float = 25.0  # min rally from low to qualify (only significant levels)

    # Failed breakdown
    sweep_min_ticks: int = 1  # minimum ticks below level (1 tick = 0.25 pts)

    # Acceptance confirmation
    acceptance_max_dip_pts: float = 3.0  # max dip below level during backtest
    acceptance_min_hold_seconds: int = 60  # hold above level
    acceptance_min_hold_bars: int = 7  # strong confirmation (7 min above level)

    # Non-acceptance (fast market) confirmation
    non_acceptance_min_recovery_pts: float = 5.0
    non_acceptance_min_hold_seconds: int = 120
    non_acceptance_min_hold_bars: int = 3

    # Sweep depth classification
    shallow_flush_threshold_pts: float = 20.0  # < 20 pts = shallow, >= 20 = deep

    # Deep flush uses longer hold/timeout
    acceptance_min_hold_bars_deep: int = 4
    acceptance_timeout_bars_shallow: int = 15
    acceptance_timeout_bars_deep: int = 60

    # True breakdown abort: consecutive bars closing below level
    true_breakdown_abort_bars: int = 12

    # Stop buffer: how far below the level to place the stop
    fb_stop_buffer_pts: float = 4.5  # FB stop at level - this value
    lr_stop_buffer_pts: float = 4.5  # LR stop at level - this value

    # Level reclaim
    level_reclaim_min_touches: int = 4  # S/R line touches required


@dataclass(frozen=True)
class RiskParams:
    """Risk management parameters."""

    max_trades_per_day: int = 3
    max_daily_loss_pts: float = 20.0  # per-contract loss limit in points
    max_position_contracts: int = 4
    never_let_green_go_red: bool = True
    # After first win: only risk profits from first trade
    risk_only_profits_after_first_win: bool = True


# Default instances for easy import
DEFAULT_CONTRACT = ESContractSpec()
MES_CONTRACT = ESContractSpec(
    symbol="MES",
    tick_size=0.25,
    tick_value=1.25,
    point_value=5.0,
    margin_initial=1_265.0,
    margin_maintenance=1_150.0,
    exchange="CME",
)
DEFAULT_SESSION = SessionTimes()
DEFAULT_ELEVATOR = ElevatorParams()
DEFAULT_EXIT = ExitParams()
DEFAULT_STRATEGY = StrategyParams()
DEFAULT_RISK = RiskParams()
