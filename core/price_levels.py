"""Significant low identification and S/R level detection.

Three algorithms for significant lows:
1. Prior day's low
2. Multi-hour low (produced a 20+ point rally) via argrelextrema
3. Cluster/shelf of lows (3+ touches within 1 point)

Plus horizontal S/R detection from multiple touches.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams, DEFAULT_STRATEGY


class PriceLevelDetector:
    """Detects significant price levels from OHLCV data."""

    def __init__(self, params: StrategyParams = DEFAULT_STRATEGY):
        self.params = params

    def detect_all(
        self,
        df: pd.DataFrame,
        prior_day_df: Optional[pd.DataFrame] = None,
    ) -> LevelStore:
        """Run all detection algorithms and return a merged LevelStore.

        Parameters
        ----------
        df : pd.DataFrame
            Current session OHLCV bars (1-min).
        prior_day_df : pd.DataFrame, optional
            Previous session bars for prior-day levels.

        Returns
        -------
        LevelStore
        """
        store = LevelStore()

        # 1. Prior day's low/high
        if prior_day_df is not None and len(prior_day_df) > 0:
            self._add_prior_day_levels(store, prior_day_df)

        # 2. Multi-hour swing lows (argrelextrema)
        if len(df) > self.params.swing_low_order * 2:
            self._add_swing_lows(store, df)

        # 3. Cluster / shelf lows
        self._add_cluster_lows(store, df)

        # 4. Horizontal S/R levels
        self._add_horizontal_sr(store, df)

        return store

    def detect_incremental(
        self, store: LevelStore, df: pd.DataFrame, bar_idx: int,
        hr_sr_interval: int = 10,
    ) -> list[Level]:
        """Detect new levels based on bars up to bar_idx (lookahead-safe).

        Parameters
        ----------
        hr_sr_interval : int
            Only run horizontal S/R and cluster detection every N bars
            (they change slowly, and the O(n) scan per call makes the
            overall day O(n²) if run every bar).

        Returns newly added levels.
        """
        new_levels: list[Level] = []

        # Check for newly confirmed swing lows (cheap: single candidate check)
        order = self.params.swing_low_order
        if bar_idx >= order * 2:
            candidate_idx = bar_idx - order  # the potential low was `order` bars ago
            lows = df["low"].values
            # Check if candidate_idx is a local minimum within [candidate_idx - order, candidate_idx + order]
            window_start = max(0, candidate_idx - order)
            window_end = min(len(lows), candidate_idx + order + 1)
            window = lows[window_start:window_end]
            local_min = lows[candidate_idx]

            if local_min == window.min() and local_min < window.mean():
                low_price = float(local_min)
                low_time = df.index[candidate_idx]
                confirmed_time = df.index[bar_idx]

                # Check rally from this low
                rally = float(df["high"].values[candidate_idx:bar_idx + 1].max()) - low_price

                if rally >= self.params.multi_hour_rally_min_pts:
                    level = Level(
                        price=low_price,
                        level_type=LevelType.MULTI_HOUR_LOW,
                        created_at=low_time,
                        confirmed_at=confirmed_time,
                        rally_from_low_pts=rally,
                    )
                else:
                    level = Level(
                        price=low_price,
                        level_type=LevelType.SWING_LOW,
                        created_at=low_time,
                        confirmed_at=confirmed_time,
                    )

                store.add(level)
                new_levels.append(level)

        # Run cluster and H/SR detection at reduced frequency (performance).
        # These scans are O(bar_idx) so running every bar makes the day O(n²).
        if bar_idx % hr_sr_interval != 0:
            return new_levels

        # Check for cluster formation in recent bars
        lookback = min(bar_idx + 1, 120)  # last 2 hours
        recent_lows = df["low"].values[max(0, bar_idx + 1 - lookback) : bar_idx + 1]
        clusters = self._find_clusters(recent_lows, self.params.cluster_proximity_pts)
        for cluster_price, count in clusters:
            if count >= self.params.cluster_min_touches:
                level = Level(
                    price=cluster_price,
                    level_type=LevelType.CLUSTER_LOW,
                    created_at=df.index[bar_idx],
                    confirmed_at=df.index[bar_idx],
                    touch_count=count,
                )
                store.add(level)
                new_levels.append(level)

        # Check for horizontal S/R levels in bars seen so far
        if bar_idx >= self.params.level_reclaim_min_touches - 1:
            highs = df["high"].values[: bar_idx + 1]
            lows = df["low"].values[: bar_idx + 1]
            all_levels = np.concatenate([highs, lows])
            rounded = np.round(all_levels * 2) / 2
            unique, counts = np.unique(rounded, return_counts=True)

            for price, count in zip(unique, counts):
                if count >= self.params.level_reclaim_min_touches:
                    high_touches = np.abs(highs - price) <= 0.5
                    low_touches = np.abs(lows - price) <= 0.5
                    all_touches = high_touches | low_touches
                    if all_touches.any():
                        last_idx = np.where(all_touches)[0][-1]
                        level = Level(
                            price=float(price),
                            level_type=LevelType.HORIZONTAL_SR,
                            created_at=df.index[last_idx],
                            confirmed_at=df.index[last_idx],
                            touch_count=int(count),
                        )
                        store.add(level)
                        new_levels.append(level)

        return new_levels

    # ------------------------------------------------------------------
    # Private detection methods
    # ------------------------------------------------------------------

    def _add_prior_day_levels(
        self, store: LevelStore, prior_df: pd.DataFrame
    ) -> None:
        """Add prior day's high and low."""
        prior_low = float(prior_df["low"].min())
        prior_high = float(prior_df["high"].max())
        low_time = prior_df["low"].idxmin()
        high_time = prior_df["high"].idxmax()

        store.add(
            Level(
                price=prior_low,
                level_type=LevelType.PRIOR_DAY_LOW,
                created_at=low_time,
                confirmed_at=low_time,  # immediately available
            )
        )
        store.add(
            Level(
                price=prior_high,
                level_type=LevelType.PRIOR_DAY_HIGH,
                created_at=high_time,
                confirmed_at=high_time,
            )
        )

    def _add_swing_lows(self, store: LevelStore, df: pd.DataFrame) -> None:
        """Detect swing lows via argrelextrema with proper confirmation delay."""
        order = self.params.swing_low_order
        lows = df["low"].values
        indices = argrelextrema(lows, np.less_equal, order=order)[0]

        for idx in indices:
            low_price = float(lows[idx])
            low_time = df.index[idx]
            # confirmed_at = order bars later (lookahead prevention)
            confirm_idx = min(idx + order, len(df) - 1)
            confirmed_time = df.index[confirm_idx]

            # Check if this low produced a significant rally
            future_high = float(df["high"].values[idx : confirm_idx + 1].max())
            rally = future_high - low_price

            if rally >= self.params.multi_hour_rally_min_pts:
                level_type = LevelType.MULTI_HOUR_LOW
            else:
                level_type = LevelType.SWING_LOW

            store.add(
                Level(
                    price=low_price,
                    level_type=level_type,
                    created_at=low_time,
                    confirmed_at=confirmed_time,
                    rally_from_low_pts=rally,
                )
            )

    def _add_cluster_lows(self, store: LevelStore, df: pd.DataFrame) -> None:
        """Detect clusters/shelves of lows (3+ touches within proximity)."""
        lows = df["low"].values
        clusters = self._find_clusters(lows, self.params.cluster_proximity_pts)

        for cluster_price, count in clusters:
            if count >= self.params.cluster_min_touches:
                # Find the time of the last touch in this cluster
                mask = np.abs(lows - cluster_price) <= self.params.cluster_proximity_pts
                last_touch_idx = np.where(mask)[0][-1]

                store.add(
                    Level(
                        price=cluster_price,
                        level_type=LevelType.CLUSTER_LOW,
                        created_at=df.index[last_touch_idx],
                        confirmed_at=df.index[last_touch_idx],
                        touch_count=count,
                    )
                )

    def _add_horizontal_sr(self, store: LevelStore, df: pd.DataFrame) -> None:
        """Detect horizontal S/R levels from high/low touches."""
        all_levels = np.concatenate([df["high"].values, df["low"].values])
        # Round to nearest 0.5 for bucketing
        rounded = np.round(all_levels * 2) / 2
        unique, counts = np.unique(rounded, return_counts=True)

        for price, count in zip(unique, counts):
            if count >= self.params.level_reclaim_min_touches:
                # Find when this level was last touched
                high_touches = np.abs(df["high"].values - price) <= 0.5
                low_touches = np.abs(df["low"].values - price) <= 0.5
                all_touches = high_touches | low_touches
                if all_touches.any():
                    last_idx = np.where(all_touches)[0][-1]
                    store.add(
                        Level(
                            price=float(price),
                            level_type=LevelType.HORIZONTAL_SR,
                            created_at=df.index[last_idx],
                            confirmed_at=df.index[last_idx],
                            touch_count=int(count),
                        )
                    )

    @staticmethod
    def _find_clusters(
        values: np.ndarray, proximity: float
    ) -> list[tuple[float, int]]:
        """Group values into clusters within `proximity` and return (center, count)."""
        if len(values) == 0:
            return []

        sorted_vals = np.sort(values)
        clusters: list[tuple[float, int]] = []
        cluster_start = 0

        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] - sorted_vals[cluster_start] > proximity:
                count = i - cluster_start
                center = float(np.mean(sorted_vals[cluster_start:i]))
                clusters.append((center, count))
                cluster_start = i

        # Last cluster
        count = len(sorted_vals) - cluster_start
        center = float(np.mean(sorted_vals[cluster_start:]))
        clusters.append((center, count))

        return clusters
