"""Mode 1 (trend day) detection.

Mancini says 90% of days are Mode 2 (range/chop where FBs work great) and
10% are Mode 1 (open-to-close trend). Mode 1 Red days destroy FB longs --
price breaks support and never comes back.

Detection logic (any 2 of 3 conditions = MODE_1_RED):
1. 3+ support levels broken AND stayed broken (price below for 20+ bars each)
2. Price below PDL (prior day low) for 30+ continuous bars
3. Sustained bearish pressure: price making lower lows for 60+ bars

When MODE_1_RED is detected, the strategy reduces position size and can
optionally disable FB longs entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from config.levels import LevelStore, LevelType
from config.settings import StrategyParams, DEFAULT_STRATEGY


@dataclass
class Mode1State:
    """Current Mode 1 detection state."""

    is_mode1_red: bool = False
    levels_broken_sustained: int = 0
    bars_below_pdl: int = 0
    bearish_pressure_bars: int = 0
    conditions_met: int = 0
    # Which conditions triggered
    condition_levels: bool = False
    condition_pdl: bool = False
    condition_pressure: bool = False


class Mode1Detector:
    """Detects Mode 1 (trend day) conditions bar-by-bar.

    Tracks three independent signals and flags MODE_1_RED when any two
    are simultaneously true. All state resets per session.
    """

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params
        self._state = Mode1State()

        # Track broken levels: {level_price: first_bar_below}
        self._broken_levels: dict[float, int] = {}
        # Track PDL
        self._pdl_price: Optional[float] = None
        self._bars_below_pdl: int = 0
        # Track bearish pressure via rolling low
        self._session_low: float = float("inf")
        self._bars_since_new_low: int = 0
        self._bearish_pressure_bars: int = 0
        # Track which levels we've already seen as confirmed-broken
        self._confirmed_broken: set[float] = set()

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._state = Mode1State()
        self._broken_levels.clear()
        self._pdl_price = None
        self._bars_below_pdl = 0
        self._session_low = float("inf")
        self._bars_since_new_low = 0
        self._bearish_pressure_bars = 0
        self._confirmed_broken.clear()

    def set_pdl(self, pdl_price: float) -> None:
        """Set prior day low price for PDL tracking."""
        self._pdl_price = pdl_price

    @property
    def state(self) -> Mode1State:
        """Current detection state."""
        return self._state

    def update(
        self,
        bar_idx: int,
        close: float,
        low: float,
        level_store: LevelStore,
        timestamp: datetime,
    ) -> Mode1State:
        """Process one bar and return updated Mode 1 state.

        Parameters
        ----------
        bar_idx : int
            Current bar index.
        close : float
            Bar close price.
        low : float
            Bar low price.
        level_store : LevelStore
            Current support levels for tracking breaks.
        timestamp : datetime
            Bar timestamp.

        Returns
        -------
        Mode1State
            Updated detection state.
        """
        hold_bars = self.params.mode1_level_broken_hold_bars

        # --- Condition 1: Sustained broken levels ---
        # Check all confirmed support levels; track how long price stays below each
        confirmed = level_store.get_confirmed(timestamp)
        support_levels = [
            lv for lv in confirmed
            if lv.level_type in (
                LevelType.PRIOR_DAY_LOW,
                LevelType.MULTI_HOUR_LOW,
                LevelType.SWING_LOW,
                LevelType.CLUSTER_LOW,
            )
        ]

        for lv in support_levels:
            lv_price = round(lv.price, 2)
            if close < lv.price:
                # Price is below this level
                if lv_price not in self._broken_levels:
                    self._broken_levels[lv_price] = bar_idx
            else:
                # Price recovered above this level -- remove from broken tracking
                self._broken_levels.pop(lv_price, None)
                self._confirmed_broken.discard(lv_price)

        # Count levels that have been broken for hold_bars+ bars
        sustained_count = 0
        for lv_price, first_broken_bar in self._broken_levels.items():
            if bar_idx - first_broken_bar >= hold_bars:
                sustained_count += 1
                if lv_price not in self._confirmed_broken:
                    self._confirmed_broken.add(lv_price)
        self._state.levels_broken_sustained = sustained_count
        condition_levels = sustained_count >= self.params.mode1_levels_broken_threshold

        # --- Condition 2: Sustained below PDL ---
        if self._pdl_price is not None:
            if close < self._pdl_price:
                self._bars_below_pdl += 1
            else:
                self._bars_below_pdl = 0
        self._state.bars_below_pdl = self._bars_below_pdl
        condition_pdl = self._bars_below_pdl >= self.params.mode1_min_bars_below_pdl

        # --- Condition 3: Bearish pressure (sustained lower lows) ---
        # Track whether price keeps making new session lows. Each new low
        # resets the counter; if no new low for 60+ bars the pressure is off.
        # We count bearish pressure as: consecutive bars where session low
        # is still being extended (new lows keep coming within a window).
        if low < self._session_low:
            self._session_low = low
            self._bars_since_new_low = 0
            self._bearish_pressure_bars += 1
        else:
            self._bars_since_new_low += 1
            # If we haven't made a new low in a long time, pressure dissipates
            if self._bars_since_new_low <= self.params.mode1_bearish_pressure_bars:
                self._bearish_pressure_bars += 1
            # else: stop counting, but don't reset (the damage is done)

        self._state.bearish_pressure_bars = self._bearish_pressure_bars
        condition_pressure = (
            self._bearish_pressure_bars >= self.params.mode1_bearish_pressure_bars
        )

        # --- Evaluate: any 2 of 3 conditions = MODE_1_RED ---
        conditions_met = sum([condition_levels, condition_pdl, condition_pressure])
        self._state.condition_levels = condition_levels
        self._state.condition_pdl = condition_pdl
        self._state.condition_pressure = condition_pressure
        self._state.conditions_met = conditions_met

        was_mode1 = self._state.is_mode1_red
        self._state.is_mode1_red = conditions_met >= 2

        if self._state.is_mode1_red and not was_mode1:
            logger.warning(
                f"MODE 1 RED detected at bar {bar_idx} | "
                f"levels_broken={sustained_count}, "
                f"bars_below_pdl={self._bars_below_pdl}, "
                f"bearish_pressure={self._bearish_pressure_bars}"
            )

        return self._state
