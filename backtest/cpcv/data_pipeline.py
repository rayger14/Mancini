"""Fetch ES futures data (1-min from Databento, hourly from yfinance) and split into daily DataFrames."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger


_CACHE_DIR = Path("data")


def fetch_es_1min_databento(
    start: date | None = None,
    end: date | None = None,
    cache_dir: Path = _CACHE_DIR,
) -> pd.DataFrame:
    """Fetch 1-minute ES continuous front-month bars from Databento.

    Converts UTC timestamps to US/Eastern, filters to RTH (9:30-16:00),
    and returns tz-naive DatetimeIndex in Eastern time.

    Requires DATABENTO_API_KEY environment variable.

    Parameters
    ----------
    start, end : date, optional
        Date range. Defaults to 2024-02-05 to today.
    cache_dir : Path
        Directory for parquet cache.

    Returns
    -------
    pd.DataFrame
        OHLCV with tz-naive DatetimeIndex (US/Eastern).
    """
    if start is None:
        start = date(2024, 2, 5)
    if end is None:
        end = date.today()

    cache_path = cache_dir / f"ES_1m_{start}_{end}.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached: {cache_path}")
        return pd.read_parquet(cache_path)

    import databento as db

    client = db.Historical()

    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3",
        symbols=["ES.c.0"],
        schema="ohlcv-1m",
        stype_in="continuous",
        start=start.isoformat(),
        end=end.isoformat(),
    )
    logger.info(f"Databento cost estimate: ${cost:.2f}")

    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["ES.c.0"],
        schema="ohlcv-1m",
        stype_in="continuous",
        start=start.isoformat(),
        end=end.isoformat(),
    )

    df = data.to_df()
    logger.info(f"Downloaded {len(df)} raw 1-min bars")

    # Convert UTC index to US/Eastern
    if df.index.tz is not None:
        df.index = df.index.tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")

    # Strip timezone for compatibility
    df.index = df.index.tz_localize(None)

    # Keep only OHLCV columns
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("open", "high", "low", "close", "volume"):
            col_map[c] = cl
    df = df.rename(columns=col_map)[["open", "high", "low", "close", "volume"]].copy()

    # Filter to RTH: 9:30 - 15:59 (bars at 16:00 are after-close)
    df = df.between_time("09:30", "15:59")
    logger.info(f"After RTH filter: {len(df)} bars")

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info(f"Cached {len(df)} bars -> {cache_path}")
    return df


def fetch_es_hourly(
    start: date | None = None,
    end: date | None = None,
    symbol: str = "ES=F",
    cache_dir: Path = _CACHE_DIR,
) -> pd.DataFrame:
    """Fetch hourly OHLCV bars for ES futures from yfinance.

    Parameters
    ----------
    start, end : date, optional
        Date range. Defaults to max available (~730 days).
    symbol : str
        yfinance ticker symbol.
    cache_dir : Path
        Directory for parquet cache.

    Returns
    -------
    pd.DataFrame
        OHLCV with tz-naive DatetimeIndex.
    """
    label = f"{symbol}_1h"
    if start and end:
        label += f"_{start}_{end}"
    cache_path = cache_dir / f"{label}.parquet"

    if cache_path.exists():
        logger.info(f"Loading cached: {cache_path}")
        return pd.read_parquet(cache_path)

    import yfinance as yf

    ticker = yf.Ticker(symbol)

    if start and end:
        df = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1h",
            auto_adjust=True,
        )
    else:
        df = ticker.history(period="max", interval="1h", auto_adjust=True)

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Strip timezone for compatibility with the rest of the codebase
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info(f"Cached {len(df)} bars -> {cache_path}")
    return df


def split_into_daily_dfs(
    df: pd.DataFrame,
    min_bars: int = 4,
) -> dict[date, pd.DataFrame]:
    """Split a multi-day DataFrame into per-day DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Multi-day OHLCV data with DatetimeIndex.
    min_bars : int
        Skip days with fewer bars than this.

    Returns
    -------
    dict[date, pd.DataFrame]
    """
    daily: dict[date, pd.DataFrame] = {}
    for d, group in df.groupby(df.index.date):
        if len(group) >= min_bars:
            daily[d] = group
    return daily
