"""Mode 1 Green (trend UP day) detection — mirror of Mode 1 Red.

Mancini (Apr 15 2026 Substack on the 6983 FB): trend days don't change the
rules for FB longs — you still need a significant low and acceptance (or
non-acceptance) above it. But on a confirmed trend-up day the R:R gate can
be relaxed because the trend itself is a large part of the edge.

Detection logic (any 2 of 5 conditions = MODE_1_GREEN):
1. 3+ resistance levels broken UP and stayed broken (price above for 20+ bars)
2. Price above PDH (prior day high) for 30+ continuous bars
3. Sustained bullish pressure: price making higher highs for 60+ bars
4. Shallow dips bought fast: 4+ pullbacks <= 8 pts recovering <= 20 bars
   (5y study: green-day median 6/day vs 1 on normal days)
5. Breakdown squeeze: break below the 60-bar low reclaimed within 20 bars
   then a new session high (green days 17% vs normal 7%)

When MODE_1_GREEN is active:
- FB longs may fire with ``mode1_green_fb_min_rr`` (relaxed R:R floor)
- ``mode1_green_size_factor`` applies to position sizing
- Non-acceptance protocol is preferred (trend moves fast)

Default is OFF — ship behind a config flag with shadow-mode logging first.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from loguru import logger

from config.levels import LevelStore, LevelType
from config.settings import StrategyParams, DEFAULT_STRATEGY


@dataclass
class Mode1GreenState:
    """Current Mode 1 Green detection state."""

    is_mode1_green: bool = False
    resistances_broken_sustained: int = 0
    bars_above_pdh: int = 0
    bullish_pressure_bars: int = 0
    shallow_fast_dips: int = 0
    squeezes: int = 0
    conditions_met: int = 0
    # Which conditions triggered
    condition_resistances: bool = False
    condition_pdh: bool = False
    condition_pressure: bool = False
    condition_shallow_dips: bool = False
    condition_squeeze: bool = False


class Mode1GreenDetector:
    """Detects Mode 1 Green (trend UP day) conditions bar-by-bar.

    Mirror of ``Mode1Detector``. Tracks three independent signals and flags
    MODE_1_GREEN when any two are simultaneously true. All state resets per
    session.
    """

    _RESISTANCE_TYPES = (
        LevelType.PRIOR_DAY_HIGH,
        LevelType.MULTI_HOUR_HIGH,
        LevelType.SWING_HIGH,
        LevelType.CLUSTER_HIGH,
        LevelType.HORIZONTAL_SR,
    )

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params
        self._state = Mode1GreenState()

        # Track broken resistances: {level_price: first_bar_above}
        self._broken_resistances: dict[float, int] = {}
        # Track PDH
        self._pdh_price: Optional[float] = None
        self._bars_above_pdh: int = 0
        # Bullish pressure via rolling high
        self._session_high: float = float("-inf")
        self._bars_since_new_high: int = 0
        self._bullish_pressure_bars: int = 0
        # Levels we've already counted as confirmed-broken-up
        self._confirmed_broken: set[float] = set()
        # Shallow-dip tracker ("dips get bought instantly")
        self._in_dip: bool = False
        self._dip_start_high: float = 0.0
        self._dip_max_depth: float = 0.0
        self._dip_bars: int = 0
        self._shallow_fast_dips: int = 0
        # Breakdown-squeeze tracker (counter-trade fails)
        self._recent_lows: deque[float] = deque(maxlen=60)
        self._sq_phase: str = "idle"  # idle / awaiting_reclaim / awaiting_high
        self._sq_level: float = 0.0
        self._sq_high_at_break: float = 0.0
        self._sq_bars: int = 0
        self._squeezes: int = 0

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._state = Mode1GreenState()
        self._broken_resistances.clear()
        self._pdh_price = None
        self._bars_above_pdh = 0
        self._session_high = float("-inf")
        self._bars_since_new_high = 0
        self._bullish_pressure_bars = 0
        self._confirmed_broken.clear()
        self._in_dip = False
        self._dip_start_high = 0.0
        self._dip_max_depth = 0.0
        self._dip_bars = 0
        self._shallow_fast_dips = 0
        self._recent_lows.clear()
        self._sq_phase = "idle"
        self._sq_level = 0.0
        self._sq_high_at_break = 0.0
        self._sq_bars = 0
        self._squeezes = 0

    def set_pdh(self, pdh_price: float) -> None:
        """Set prior day high price for PDH tracking."""
        self._pdh_price = pdh_price

    @property
    def state(self) -> Mode1GreenState:
        """Current detection state."""
        return self._state

    def update(
        self,
        bar_idx: int,
        close: float,
        high: float,
        level_store: LevelStore,
        timestamp: datetime,
        low: Optional[float] = None,
    ) -> Mode1GreenState:
        """Process one bar and return updated Mode 1 Green state.

        Parameters
        ----------
        bar_idx : int
            Current bar index.
        close : float
            Bar close price.
        high : float
            Bar high price (used for bullish pressure tracking).
        level_store : LevelStore
            Current levels (resistances are filtered internally).
        timestamp : datetime
            Bar timestamp.
        low : float, optional
            Bar low. Required for the shallow-dip and breakdown-squeeze
            tells; when omitted those conditions stay False.

        Returns
        -------
        Mode1GreenState
            Updated detection state.
        """
        hold_bars = self.params.mode1_green_level_broken_hold_bars

        # --- Condition 1: Sustained broken resistances (price above for hold_bars+) ---
        confirmed = level_store.get_confirmed(timestamp)
        resistance_levels = [
            lv for lv in confirmed if lv.level_type in self._RESISTANCE_TYPES
        ]

        for lv in resistance_levels:
            lv_price = round(lv.price, 2)
            if close > lv.price:
                # Price is above this resistance
                if lv_price not in self._broken_resistances:
                    self._broken_resistances[lv_price] = bar_idx
            else:
                # Price fell back below — remove from broken tracking
                self._broken_resistances.pop(lv_price, None)
                self._confirmed_broken.discard(lv_price)

        sustained_count = 0
        for lv_price, first_broken_bar in self._broken_resistances.items():
            if bar_idx - first_broken_bar >= hold_bars:
                sustained_count += 1
                if lv_price not in self._confirmed_broken:
                    self._confirmed_broken.add(lv_price)
        self._state.resistances_broken_sustained = sustained_count
        condition_resistances = (
            sustained_count >= self.params.mode1_green_resistance_broken_threshold
        )

        # --- Condition 2: Sustained above PDH ---
        if self._pdh_price is not None:
            if close > self._pdh_price:
                self._bars_above_pdh += 1
            else:
                self._bars_above_pdh = 0
        self._state.bars_above_pdh = self._bars_above_pdh
        condition_pdh = self._bars_above_pdh >= self.params.mode1_green_bars_above_pdh

        # --- Condition 3: Bullish pressure (sustained higher highs) ---
        if high > self._session_high:
            self._session_high = high
            self._bars_since_new_high = 0
            self._bullish_pressure_bars += 1
        else:
            self._bars_since_new_high += 1
            if self._bars_since_new_high <= self.params.mode1_green_bullish_pressure_bars:
                self._bullish_pressure_bars += 1
            # else: pressure dissipates — stop counting but don't reset

        self._state.bullish_pressure_bars = self._bullish_pressure_bars
        condition_pressure = (
            self._bullish_pressure_bars >= self.params.mode1_green_bullish_pressure_bars
        )

        # --- Condition 4: Shallow dips bought fast ---
        # 5y study: green days median 6 shallow-fast dips vs 1 on normal days.
        # An episode starts when price pulls > 2 pts off the session high and
        # ends when the high is reclaimed; it counts if it stayed shallow and
        # recovered quickly.
        if low is not None and self._session_high > float("-inf"):
            depth = self._session_high - low
            if not self._in_dip:
                if depth > 2.0:
                    self._in_dip = True
                    self._dip_start_high = self._session_high
                    self._dip_max_depth = depth
                    self._dip_bars = 0
            else:
                self._dip_bars += 1
                self._dip_max_depth = max(self._dip_max_depth, depth)
                if high >= self._dip_start_high:
                    if (self._dip_max_depth <= self.params.mode1_green_shallow_dip_max_pts
                            and self._dip_bars <= self.params.mode1_green_shallow_dip_max_bars):
                        self._shallow_fast_dips += 1
                    self._in_dip = False
        self._state.shallow_fast_dips = self._shallow_fast_dips
        condition_shallow = (
            self._shallow_fast_dips >= self.params.mode1_green_shallow_dips_min
        )

        # --- Condition 5: Breakdown squeeze (the counter-trade fails) ---
        # Break below the trailing 60-bar low, reclaim within 20 bars, then a
        # new session high within 60 bars — a failed breakdown that squeezed.
        if low is not None:
            if self._sq_phase == "idle":
                if (len(self._recent_lows) == self._recent_lows.maxlen
                        and low < min(self._recent_lows) - 2.0):
                    self._sq_phase = "awaiting_reclaim"
                    self._sq_level = min(self._recent_lows)
                    self._sq_high_at_break = self._session_high
                    self._sq_bars = 0
            elif self._sq_phase == "awaiting_reclaim":
                self._sq_bars += 1
                if close > self._sq_level:
                    self._sq_phase = "awaiting_high"
                    self._sq_bars = 0
                elif self._sq_bars > 20:
                    self._sq_phase = "idle"
            elif self._sq_phase == "awaiting_high":
                self._sq_bars += 1
                if high > self._sq_high_at_break:
                    self._squeezes += 1
                    self._sq_phase = "idle"
                elif self._sq_bars > 60:
                    self._sq_phase = "idle"
            self._recent_lows.append(low)
        self._state.squeezes = self._squeezes
        condition_squeeze = self._squeezes >= self.params.mode1_green_squeeze_min

        # --- Evaluate: any 2 of 5 conditions = MODE_1_GREEN ---
        conditions_met = sum([
            condition_resistances, condition_pdh, condition_pressure,
            condition_shallow, condition_squeeze,
        ])
        self._state.condition_resistances = condition_resistances
        self._state.condition_pdh = condition_pdh
        self._state.condition_pressure = condition_pressure
        self._state.condition_shallow_dips = condition_shallow
        self._state.condition_squeeze = condition_squeeze
        self._state.conditions_met = conditions_met

        was_green = self._state.is_mode1_green
        self._state.is_mode1_green = conditions_met >= 2

        if self._state.is_mode1_green and not was_green:
            logger.warning(
                f"MODE 1 GREEN detected at bar {bar_idx} | "
                f"resistances_broken={sustained_count}, "
                f"bars_above_pdh={self._bars_above_pdh}, "
                f"bullish_pressure={self._bullish_pressure_bars}, "
                f"shallow_fast_dips={self._shallow_fast_dips}, "
                f"squeezes={self._squeezes}"
            )

        return self._state
