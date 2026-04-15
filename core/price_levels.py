"""Significant low identification and S/R level detection.

Three algorithms for significant lows:
1. Prior day's low
2. Multi-hour low (produced a 20+ point rally) via argrelextrema
3. Cluster/shelf of lows (3+ touches within 1 point)

Plus horizontal S/R detection from multiple touches.
"""

from __future__ import annotations

from datetime import time as dt_time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams, DEFAULT_STRATEGY


class PriceLevelDetector:
    """Detects significant price levels from OHLCV data."""

    def __init__(
        self,
        params: StrategyParams = DEFAULT_STRATEGY,
        rth_filter: Optional[tuple[dt_time, dt_time]] = None,
    ):
        self.params = params
        # When set, only bars within (start, end) create new levels.
        # Pattern detection still runs on all bars against existing levels.
        self.rth_filter = rth_filter

    def _is_rth_time(self, t: dt_time) -> bool:
        """Check if a time falls within RTH hours for level creation."""
        if self.rth_filter is None:
            return True
        start, end = self.rth_filter
        return start <= t < end

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
            if self.params.allow_short_fr or self.params.allow_short_lj or self.params.allow_breakdown_short or self.params.allow_backtest_short:
                self._add_swing_highs(store, df)

        # 3. Cluster / shelf lows and highs
        self._add_cluster_lows(store, df)
        if self.params.allow_short_fr or self.params.allow_short_lj or self.params.allow_breakdown_short or self.params.allow_backtest_short:
            self._add_cluster_highs(store, df)

        # 4. Horizontal S/R levels
        self._add_horizontal_sr(store, df)

        return store

    def detect_incremental(
        self, store: LevelStore, df: pd.DataFrame, bar_idx: int,
        hr_sr_interval: int = 10,
        df_5min: Optional[pd.DataFrame] = None,
        bar_idx_5min: Optional[int] = None,
    ) -> list[Level]:
        """Detect new levels based on bars up to bar_idx (lookahead-safe).

        Parameters
        ----------
        hr_sr_interval : int
            Only run horizontal S/R and cluster detection every N bars
            (they change slowly, and the O(n) scan per call makes the
            overall day O(n²) if run every bar).
        df_5min : pd.DataFrame, optional
            5-minute resampled OHLCV bars. When provided with
            ``use_5min_levels=True``, swing detection runs on 5-min data.
        bar_idx_5min : int, optional
            Current bar index into df_5min.

        Returns newly added levels.
        """
        new_levels: list[Level] = []

        # Decide whether to use 5-min data for swing detection
        use_5min = (
            self.params.use_5min_levels
            and df_5min is not None
            and bar_idx_5min is not None
            and len(df_5min) > 0
        )

        if use_5min:
            # --- 5-min swing low detection ---
            new_levels.extend(
                self._detect_swing_lows_on_df(
                    store, df_5min, bar_idx_5min,
                    order=self.params.swing_low_order_5min,
                    current_close_1min=float(df["close"].values[bar_idx]),
                    timestamp_1min=df.index[bar_idx],
                )
            )

            # --- 5-min swing high detection (when short-side enabled) ---
            if (self.params.allow_short_fr or self.params.allow_short_lj
                    or self.params.allow_breakdown_short or self.params.allow_backtest_short):
                new_levels.extend(
                    self._detect_swing_highs_on_df(
                        store, df_5min, bar_idx_5min,
                        order=self.params.swing_low_order_5min,
                    )
                )

            # --- Shelf-of-lows detection on 5-min ---
            if self.params.detect_shelf_levels:
                new_levels.extend(
                    self._detect_shelf_levels(store, df_5min, bar_idx_5min)
                )
        else:
            # --- Original 1-min swing low detection ---
            new_levels.extend(
                self._detect_swing_lows_on_df(
                    store, df, bar_idx,
                    order=self.params.swing_low_order,
                    current_close_1min=float(df["close"].values[bar_idx]),
                    timestamp_1min=df.index[bar_idx],
                )
            )

        # Check for newly confirmed swing highs (mirror of swing low detection)
        # Only when short-side trading is enabled and NOT using 5-min (handled above)
        if not use_5min:
            order_1min = self.params.swing_low_order
            if (self.params.allow_short_fr or self.params.allow_short_lj
                    or self.params.allow_breakdown_short or self.params.allow_backtest_short) and bar_idx >= order_1min * 2:
                new_levels.extend(
                    self._detect_swing_highs_on_df(store, df, bar_idx, order=order_1min)
                )

        # Deep sell recovery: detect intraday levels with faster confirmation
        # when price is far below known support (Mancini: FB new levels during selloff)
        if self.params.allow_deep_sell_recovery and bar_idx >= self.params.deep_sell_swing_order * 2:
            deep_sell_levels = self._detect_deep_sell_levels(store, df, bar_idx)
            new_levels.extend(deep_sell_levels)

        # Run cluster and H/SR detection at reduced frequency (performance).
        # These scans are O(bar_idx) so running every bar makes the day O(n²).
        if bar_idx % hr_sr_interval != 0:
            return new_levels

        # Check for cluster formation in recent bars (RTH bars only)
        lookback = min(bar_idx + 1, 120)  # last 2 hours
        slice_start = max(0, bar_idx + 1 - lookback)
        slice_end = bar_idx + 1
        if self.rth_filter is not None:
            # Only use RTH bars for cluster detection
            times = df.index[slice_start:slice_end]
            rth_mask = np.array([self._is_rth_time(t.time()) for t in times])
            recent_lows = df["low"].values[slice_start:slice_end][rth_mask]
        else:
            recent_lows = df["low"].values[slice_start:slice_end]
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

        # Check for cluster highs (resistance) — only when short-side enabled
        if self.params.allow_short_fr or self.params.allow_short_lj or self.params.allow_breakdown_short or self.params.allow_backtest_short:
            if self.rth_filter is not None:
                recent_highs = df["high"].values[slice_start:slice_end][rth_mask]
            else:
                recent_highs = df["high"].values[slice_start:slice_end]
            clusters_high = self._find_clusters(recent_highs, self.params.cluster_proximity_pts)
            for cluster_price, count in clusters_high:
                if count >= self.params.cluster_min_touches:
                    level = Level(
                        price=cluster_price,
                        level_type=LevelType.CLUSTER_HIGH,
                        created_at=df.index[bar_idx],
                        confirmed_at=df.index[bar_idx],
                        touch_count=count,
                    )
                    store.add(level)
                    new_levels.append(level)

        # Check for horizontal S/R levels in bars seen so far (RTH bars only)
        if bar_idx >= self.params.level_reclaim_min_touches - 1:
            if self.rth_filter is not None:
                # Only use RTH bars for H/SR detection
                times = df.index[: bar_idx + 1]
                rth_mask = np.array([self._is_rth_time(t.time()) for t in times])
                highs = df["high"].values[: bar_idx + 1][rth_mask]
                lows = df["low"].values[: bar_idx + 1][rth_mask]
                rth_indices = np.where(rth_mask)[0]  # original indices of RTH bars
            else:
                highs = df["high"].values[: bar_idx + 1]
                lows = df["low"].values[: bar_idx + 1]
                rth_indices = None

            all_levels = np.concatenate([highs, lows])
            rounded = np.round(all_levels * 2) / 2
            unique, counts = np.unique(rounded, return_counts=True)

            for price, count in zip(unique, counts):
                if count >= self.params.level_reclaim_min_touches:
                    high_touches = np.abs(highs - price) <= 0.5
                    low_touches = np.abs(lows - price) <= 0.5
                    all_touches = high_touches | low_touches
                    if all_touches.any():
                        last_rth_pos = np.where(all_touches)[0][-1]
                        # Map back to original DataFrame index
                        if rth_indices is not None:
                            last_idx = rth_indices[last_rth_pos]
                        else:
                            last_idx = last_rth_pos
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

    def _add_swing_highs(self, store: LevelStore, df: pd.DataFrame) -> None:
        """Detect swing highs via argrelextrema (mirror of _add_swing_lows)."""
        order = self.params.short_swing_high_order
        highs = df["high"].values
        indices = argrelextrema(highs, np.greater_equal, order=order)[0]

        for idx in indices:
            high_price = float(highs[idx])
            high_time = df.index[idx]
            confirm_idx = min(idx + order, len(df) - 1)
            confirmed_time = df.index[confirm_idx]

            # Check if this high produced a significant selloff
            future_low = float(df["low"].values[idx : confirm_idx + 1].min())
            selloff = high_price - future_low

            if selloff >= self.params.multi_hour_rally_min_pts:
                level_type = LevelType.MULTI_HOUR_HIGH
            else:
                level_type = LevelType.SWING_HIGH

            store.add(
                Level(
                    price=high_price,
                    level_type=level_type,
                    created_at=high_time,
                    confirmed_at=confirmed_time,
                    rally_from_low_pts=selloff,
                )
            )

    def _add_cluster_highs(self, store: LevelStore, df: pd.DataFrame) -> None:
        """Detect clusters/shelves of highs (3+ touches within proximity)."""
        highs = df["high"].values
        clusters = self._find_clusters(highs, self.params.cluster_proximity_pts)

        for cluster_price, count in clusters:
            if count >= self.params.cluster_min_touches:
                mask = np.abs(highs - cluster_price) <= self.params.cluster_proximity_pts
                last_touch_idx = np.where(mask)[0][-1]

                store.add(
                    Level(
                        price=cluster_price,
                        level_type=LevelType.CLUSTER_HIGH,
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

    def _detect_swing_lows_on_df(
        self,
        store: LevelStore,
        df: pd.DataFrame,
        bar_idx: int,
        order: int,
        current_close_1min: Optional[float] = None,
        timestamp_1min: Optional[pd.Timestamp] = None,
    ) -> list[Level]:
        """Detect a swing low on the given DataFrame at ``bar_idx``.

        Works identically on 1-min or 5-min data — the caller chooses
        the order and DataFrame.  RTH and level-desert bypass logic use
        the 1-min close when available (5-min path) or fall back to the
        df close.
        """
        new_levels: list[Level] = []
        if bar_idx < order * 2:
            return new_levels

        candidate_idx = bar_idx - order

        # Determine current close for level-desert bypass
        if current_close_1min is not None:
            current_close = current_close_1min
        else:
            current_close = float(df["close"].values[bar_idx])

        # Determine timestamp for confirmation
        if timestamp_1min is not None:
            confirm_ts = timestamp_1min
        else:
            confirm_ts = df.index[bar_idx]

        confirmed = store.get_confirmed(confirm_ts)
        has_nearby_level = any(
            abs(l.price - current_close) <= self.params.deep_sell_threshold_pts
            for l in confirmed
        )
        is_rth = self._is_rth_time(df.index[candidate_idx].time())
        if not is_rth and has_nearby_level:
            return new_levels

        lows = df["low"].values
        window_start = max(0, candidate_idx - order)
        window_end = min(len(lows), candidate_idx + order + 1)
        window = lows[window_start:window_end]
        local_min = lows[candidate_idx]

        if local_min == window.min() and local_min < window.mean():
            low_price = float(local_min)
            low_time = df.index[candidate_idx]
            confirmed_time = confirm_ts

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

        return new_levels

    def _detect_swing_highs_on_df(
        self,
        store: LevelStore,
        df: pd.DataFrame,
        bar_idx: int,
        order: int,
    ) -> list[Level]:
        """Detect a swing high on the given DataFrame at ``bar_idx``."""
        new_levels: list[Level] = []
        if bar_idx < order * 2:
            return new_levels

        candidate_idx = bar_idx - order
        if not self._is_rth_time(df.index[candidate_idx].time()):
            return new_levels

        highs = df["high"].values
        window_start = max(0, candidate_idx - order)
        window_end = min(len(highs), candidate_idx + order + 1)
        window = highs[window_start:window_end]
        local_max = highs[candidate_idx]

        if local_max == window.max() and local_max > window.mean():
            high_price = float(local_max)
            high_time = df.index[candidate_idx]
            confirmed_time = df.index[bar_idx]

            selloff = high_price - float(df["low"].values[candidate_idx:bar_idx + 1].min())

            if selloff >= self.params.multi_hour_rally_min_pts:
                level = Level(
                    price=high_price,
                    level_type=LevelType.MULTI_HOUR_HIGH,
                    created_at=high_time,
                    confirmed_at=confirmed_time,
                    rally_from_low_pts=selloff,
                )
            else:
                level = Level(
                    price=high_price,
                    level_type=LevelType.SWING_HIGH,
                    created_at=high_time,
                    confirmed_at=confirmed_time,
                )

            store.add(level)
            new_levels.append(level)

        return new_levels

    def _detect_shelf_levels(
        self,
        store: LevelStore,
        df_5min: pd.DataFrame,
        bar_idx_5min: int,
    ) -> list[Level]:
        """Detect shelf-of-lows: tight horizontal zones with 4+ touches on 5-min bars.

        A shelf is a price range of <= shelf_proximity_pts where the low has been
        tested shelf_min_touches times over shelf_min_bars bars.
        """
        new_levels: list[Level] = []
        if bar_idx_5min < self.params.shelf_min_bars:
            return new_levels

        # Look at the last shelf_min_bars * 2 5-min bars
        lookback = min(bar_idx_5min + 1, self.params.shelf_min_bars * 2)
        start = max(0, bar_idx_5min + 1 - lookback)
        recent_lows = df_5min["low"].values[start:bar_idx_5min + 1]

        # Find the floor of the range
        floor_price = float(recent_lows.min())
        ceiling = floor_price + self.params.shelf_proximity_pts

        # Count how many bars touched the shelf zone (low within proximity of floor)
        touches = sum(1 for low in recent_lows if low <= ceiling)

        if touches >= self.params.shelf_min_touches:
            # Check it's not already tracked
            too_close = any(
                abs(l.price - floor_price) <= 1.0
                for l in store.levels
                if l.level_type in (
                    LevelType.HORIZONTAL_SR,
                    LevelType.SWING_LOW,
                    LevelType.CLUSTER_LOW,
                    LevelType.MULTI_HOUR_LOW,
                )
            )
            if not too_close:
                # Check rally from shelf (has price moved away from it?)
                current_close = float(df_5min["close"].values[bar_idx_5min])
                rally = current_close - floor_price

                level = Level(
                    price=floor_price,
                    level_type=LevelType.HORIZONTAL_SR,
                    created_at=df_5min.index[start],
                    confirmed_at=df_5min.index[bar_idx_5min],
                    touch_count=touches,
                    rally_from_low_pts=rally if rally > 0 else 0,
                )
                store.add(level)
                new_levels.append(level)

        return new_levels

    def _detect_deep_sell_levels(
        self, store: LevelStore, df: pd.DataFrame, bar_idx: int,
    ) -> list[Level]:
        """Detect intraday levels when price is in a deep sell.

        During a massive selloff (>30 pts below nearest support), use faster
        swing detection (order=5) to catch consolidation zones and crash bottoms.
        These become INTRADAY_LOW levels eligible for FB detection.

        Also detects the absolute crash bottom when rally confirms it.
        """
        new_levels: list[Level] = []
        close = float(df["close"].values[bar_idx])
        timestamp = df.index[bar_idx]

        # Check if we're in a deep sell: close is far below nearest confirmed support
        confirmed = store.get_confirmed(timestamp)
        supports = [l for l in confirmed if l.price > close]
        if not supports:
            return new_levels

        nearest_support = min(supports, key=lambda l: l.price)
        depth_below = nearest_support.price - close

        if depth_below < self.params.deep_sell_threshold_pts:
            return new_levels

        # --- Deep sell mode: faster swing low detection ---
        order = self.params.deep_sell_swing_order
        if bar_idx < order * 2:
            return new_levels

        lows = df["low"].values
        candidate_idx = bar_idx - order

        # Check if candidate_idx is a local minimum with the shorter order
        window_start = max(0, candidate_idx - order)
        window_end = min(len(lows), candidate_idx + order + 1)
        window = lows[window_start:window_end]
        local_min = float(lows[candidate_idx])

        if local_min == window.min() and local_min < window.mean():
            # Check rally from this low
            rally = float(df["high"].values[candidate_idx:bar_idx + 1].max()) - local_min

            # Use lower rally threshold for deep sell intraday levels
            if rally >= self.params.deep_sell_rally_confirm_pts:
                # Check it's not too close to an existing level
                too_close = any(
                    abs(l.price - local_min) <= 1.0
                    for l in confirmed
                    if l.level_type in (
                        LevelType.INTRADAY_LOW,
                        LevelType.MULTI_HOUR_LOW,
                        LevelType.PRIOR_DAY_LOW,
                        LevelType.SWING_LOW,
                    )
                )
                if not too_close:
                    level = Level(
                        price=local_min,
                        level_type=LevelType.INTRADAY_LOW,
                        created_at=df.index[candidate_idx],
                        confirmed_at=timestamp,  # confirmed now
                        rally_from_low_pts=rally,
                    )
                    store.add(level)
                    new_levels.append(level)

        # --- Crash bottom detection: absolute session low ---
        # When the session low has produced a significant rally, it's confirmed
        session_lows = lows[:bar_idx + 1]
        session_low_val = float(session_lows.min())
        session_low_idx = int(session_lows.argmin())
        rally_from_bottom = close - session_low_val

        if rally_from_bottom >= self.params.deep_sell_rally_confirm_pts:
            # Check crash bottom isn't already tracked
            too_close = any(
                abs(l.price - session_low_val) <= 1.0
                for l in store.levels
                if l.level_type == LevelType.INTRADAY_LOW
            )
            if not too_close:
                level = Level(
                    price=session_low_val,
                    level_type=LevelType.INTRADAY_LOW,
                    created_at=df.index[session_low_idx],
                    confirmed_at=timestamp,
                    rally_from_low_pts=rally_from_bottom,
                )
                store.add(level)
                new_levels.append(level)

        return new_levels

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
