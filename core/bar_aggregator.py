"""5-minute bar aggregation for level detection.

Mancini reads 5-min charts for level identification. This module resamples
1-min OHLCV bars into 5-min bars so that swing detection and shelf-of-lows
detection operate on the same timeframe Mancini uses.
"""

from __future__ import annotations

import pandas as pd


class BarAggregator:
    """Aggregates 1-min bars into 5-min OHLCV bars for level detection."""

    def __init__(self, period_minutes: int = 5):
        self.period = period_minutes

    def resample(self, df_1min: pd.DataFrame) -> pd.DataFrame:
        """Resample full 1-min DF to 5-min. Used for batch/initial.

        Parameters
        ----------
        df_1min : pd.DataFrame
            1-minute OHLCV bars with a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            5-minute OHLCV bars (incomplete trailing bar included).
        """
        if df_1min is None or len(df_1min) == 0:
            return pd.DataFrame()
        return df_1min.resample(f"{self.period}min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    def update_incremental(self, df_1min: pd.DataFrame) -> pd.DataFrame:
        """Resample and return only completed 5-min bars.

        The last 5-min bar is dropped if it is still forming (fewer than
        ``period`` 1-min bars have contributed to it).

        Parameters
        ----------
        df_1min : pd.DataFrame
            1-minute OHLCV bars with a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Completed 5-minute OHLCV bars.
        """
        if df_1min is None or len(df_1min) < self.period:
            return pd.DataFrame()
        resampled = self.resample(df_1min)
        # Drop the last row if it's incomplete (current 5-min bar still forming)
        if len(resampled) > 0 and len(df_1min) % self.period != 0:
            resampled = resampled.iloc[:-1]
        return resampled
