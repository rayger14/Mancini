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

        Completeness is decided by timestamps, not row count: live DFs are
        rolling windows trimmed mid-bucket and can have data gaps, so
        ``len % period`` says nothing about whether the last bucket is done.

        - The trailing bucket is kept only once data covers its final minute
          (bars are labeled by start time).
        - The leading bucket is dropped if the window starts mid-bucket,
          since its OHLC would be computed from a truncated slice.

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
        if len(resampled) == 0:
            return resampled
        period_td = pd.Timedelta(minutes=self.period)
        covered_until = df_1min.index[-1] + pd.Timedelta(minutes=1)
        resampled = resampled[resampled.index + period_td <= covered_until]
        first_ts = df_1min.index[0]
        bucket_of_first = first_ts.floor(f"{self.period}min")
        if (
            len(resampled) > 0
            and first_ts != bucket_of_first
            and resampled.index[0] == bucket_of_first
        ):
            resampled = resampled.iloc[1:]
        return resampled
