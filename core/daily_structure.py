"""Daily Structure Detector — Macro bias from the daily chart.

Reads daily OHLC bars and determines:
- Is a daily FB active? (sweep of major support shelf + recovery)
- Is a daily BD active? (break below support shelf, holding below)
- How extended is the current move from the shelf?

This macro bias is used by SignalAggregator to:
- Boost FB Long LQS during DAILY_FB_BULL
- Suppress low-quality shorts during DAILY_FB_BULL
- (Future: suppress weak longs during DAILY_BD_BEAR)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import StrategyParams


class DailyStructureDetector:
    """Detects macro bias from daily chart structure.

    Analyzes daily OHLC bars to find:
    1. Daily support shelves — clusters of daily lows within a proximity window
    2. Daily FB — price swept below shelf and recovered (bullish)
    3. Daily BD — price broke below shelf and holds (bearish)
    4. Move position — how extended the current move is from the shelf
    """

    def __init__(self, params: StrategyParams):
        self.params = params
        self._state: str = "NEUTRAL"
        self._shelf_price: float = 0.0
        self._sweep_low: float = 0.0
        self._recovery_confirmed: bool = False
        self._move_position: float = 0.0  # 0 = at shelf, 1 = at sweep distance above shelf

    def update(self, daily_bars: pd.DataFrame) -> str:
        """Analyze daily bars and return macro bias.

        Parameters
        ----------
        daily_bars : pd.DataFrame
            Daily OHLC bars with columns: open, high, low, close.
            Must have at least ``daily_shelf_lookback_days`` rows.

        Returns
        -------
        str
            One of: "DAILY_FB_BULL", "DAILY_BD_BEAR", or "NEUTRAL"
        """
        if not self.params.use_daily_structure:
            self._state = "NEUTRAL"
            return self._state

        lookback = self.params.daily_shelf_lookback_days
        cluster_pts = self.params.daily_shelf_cluster_pts
        min_touches = self.params.daily_shelf_min_touches
        recovery_bars = self.params.daily_fb_recovery_bars

        if daily_bars is None or len(daily_bars) < lookback:
            self._state = "NEUTRAL"
            return self._state

        # Ensure we have the required columns
        for col in ("low", "close"):
            if col not in daily_bars.columns:
                self._state = "NEUTRAL"
                return self._state

        # Step 1: Find the daily support shelf (cluster of daily lows)
        lows = daily_bars["low"].values[-lookback:]
        best_shelf = 0.0
        best_count = 0

        for i, low in enumerate(lows):
            nearby = sum(1 for other in lows if abs(other - low) <= cluster_pts)
            if nearby >= min_touches and nearby > best_count:
                # Use the median of nearby lows as the shelf price for stability
                nearby_lows = [other for other in lows if abs(other - low) <= cluster_pts]
                best_shelf = float(np.median(nearby_lows))
                best_count = nearby

        if best_shelf <= 0 or best_count < min_touches:
            self._state = "NEUTRAL"
            self._shelf_price = 0.0
            self._sweep_low = 0.0
            self._move_position = 0.0
            return self._state

        self._shelf_price = best_shelf

        # Step 2: Check if price swept below the shelf
        sweep_low = float(np.min(lows))
        self._sweep_low = sweep_low

        # Sweep must be meaningfully below the shelf (beyond cluster range)
        swept_below = sweep_low < best_shelf - cluster_pts

        if not swept_below:
            # No sweep — check for BD (price breaking down through shelf)
            recent_closes = daily_bars["close"].values[-recovery_bars:]
            if all(c < best_shelf for c in recent_closes):
                self._state = "DAILY_BD_BEAR"
                self._recovery_confirmed = False
                self._move_position = 0.0
                return self._state

            self._state = "NEUTRAL"
            self._move_position = 0.0
            return self._state

        # Step 3: Check recovery — are last N daily closes above the shelf?
        recent_closes = daily_bars["close"].values[-recovery_bars:]
        recovery = all(c > best_shelf for c in recent_closes)

        if recovery:
            self._state = "DAILY_FB_BULL"
            self._recovery_confirmed = True

            # Step 4: Compute move position
            # How far has price traveled from the shelf relative to the sweep depth?
            current_price = float(daily_bars["close"].values[-1])
            sweep_depth = abs(sweep_low - best_shelf)
            if sweep_depth > 0:
                self._move_position = (current_price - best_shelf) / sweep_depth
            else:
                self._move_position = 0.0
        else:
            # Price swept below but hasn't recovered — still bearish
            recent_closes_arr = daily_bars["close"].values[-recovery_bars:]
            if all(c < best_shelf for c in recent_closes_arr):
                self._state = "DAILY_BD_BEAR"
                self._recovery_confirmed = False
            else:
                self._state = "NEUTRAL"
                self._recovery_confirmed = False
            self._move_position = 0.0

        logger.debug(
            f"Daily structure: {self._state} | shelf={self._shelf_price:.1f} "
            f"sweep_low={self._sweep_low:.1f} move_pos={self._move_position:.2f}"
        )
        return self._state

    def get_snapshot(self) -> dict:
        """Return current state for logging/dashboard."""
        return {
            "state": self._state,
            "shelf_price": self._shelf_price,
            "sweep_low": self._sweep_low,
            "move_position": round(self._move_position, 3),
        }
