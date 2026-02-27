"""Technical indicators: ATR, VWAP, velocity, volume spike helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(
    df: pd.DataFrame, period: int = 14, column_prefix: str = ""
) -> pd.Series:
    """Average True Range.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'high', 'low', 'close' columns.
    period : int
        Lookback period.

    Returns
    -------
    pd.Series
        ATR values aligned with df index.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=period, min_periods=1).mean()
    atr.name = f"{column_prefix}atr_{period}"
    return atr


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (intraday, resets daily).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'high', 'low', 'close', 'volume' with DatetimeIndex.

    Returns
    -------
    pd.Series
        VWAP values.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cumulative_tp_vol = (typical_price * df["volume"]).groupby(df.index.date).cumsum()
    cumulative_vol = df["volume"].groupby(df.index.date).cumsum()
    vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    vwap.name = "vwap"
    return vwap


def compute_velocity(
    df: pd.DataFrame, window: int = 5, price_col: str = "close"
) -> pd.Series:
    """Price velocity in points per bar (proxy for pts/min on 1-min bars).

    Negative velocity = selling pressure (elevator down).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `price_col`.
    window : int
        Rolling window size in bars.

    Returns
    -------
    pd.Series
        Velocity (pts/bar). Negative = selloff.
    """
    price = df[price_col]
    velocity = (price - price.shift(window)) / window
    velocity.name = f"velocity_{window}"
    return velocity


def compute_rate_of_change(
    df: pd.DataFrame, period: int = 5, price_col: str = "close"
) -> pd.Series:
    """Rate of Change (percentage)."""
    price = df[price_col]
    roc = (price - price.shift(period)) / price.shift(period) * 100.0
    roc.name = f"roc_{period}"
    return roc


def detect_volume_spike(
    df: pd.DataFrame, lookback: int = 20, threshold: float = 2.0
) -> pd.Series:
    """Detect volume spikes relative to recent average.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'volume'.
    lookback : int
        Bars for rolling average.
    threshold : float
        Multiple of average volume to qualify as spike.

    Returns
    -------
    pd.Series[bool]
        True where volume >= threshold * rolling_mean.
    """
    vol = df["volume"]
    avg_vol = vol.rolling(window=lookback, min_periods=1).mean()
    spike = vol >= (threshold * avg_vol)
    spike.name = "volume_spike"
    return spike


def compute_bar_range(df: pd.DataFrame) -> pd.Series:
    """High - Low range for each bar."""
    r = df["high"] - df["low"]
    r.name = "bar_range"
    return r


def compute_body_ratio(df: pd.DataFrame) -> pd.Series:
    """Ratio of candle body to full range. 1.0 = marubozu, 0.0 = doji."""
    body = (df["close"] - df["open"]).abs()
    full_range = (df["high"] - df["low"]).replace(0, np.nan)
    ratio = body / full_range
    ratio.name = "body_ratio"
    return ratio.fillna(0.0)


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add all standard indicators as columns to a copy of df."""
    out = df.copy()
    out["atr_14"] = compute_atr(df, period=14)
    out["vwap"] = compute_vwap(df)
    out["velocity_5"] = compute_velocity(df, window=5)
    out["roc_5"] = compute_rate_of_change(df, period=5)
    out["volume_spike"] = detect_volume_spike(df)
    out["bar_range"] = compute_bar_range(df)
    out["body_ratio"] = compute_body_ratio(df)
    return out
