#!/usr/bin/env python3
"""
Fetch full-session (23-hour) ES 1-min bars from Databento.

Includes both RTH (9:30-16:00 ET) and overnight/globex (18:00-9:29 ET).
Saves to data/ES_1m_full_session_YYYY-MM-DD_YYYY-MM-DD.parquet

Usage:
    export DATABENTO_API_KEY="your-key-here"
    python3 backtest/fetch_globex_data.py
"""

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

# Load .env if present
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def fetch_es_full_session(
    start: date = date(2024, 2, 5),
    end: date = date(2026, 2, 5),
    cache_dir: Path = Path("data"),
) -> pd.DataFrame:
    cache_path = cache_dir / f"ES_1m_full_session_{start}_{end}.parquet"
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

    if cost > 20.0:
        proceed = input(f"Download will cost ~${cost:.2f}. Proceed? [y/N] ")
        if proceed.lower() != "y":
            logger.info("Aborted.")
            sys.exit(0)
    else:
        logger.info(f"Cost is ${cost:.2f}, proceeding automatically.")

    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["ES.c.0"],
        schema="ohlcv-1m",
        stype_in="continuous",
        start=start.isoformat(),
        end=end.isoformat(),
    )

    df = data.to_df()
    logger.info(f"Downloaded {len(df)} raw 1-min bars (full session)")

    # Convert UTC to Eastern
    if df.index.tz is not None:
        df.index = df.index.tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    df.index = df.index.tz_localize(None)

    # Keep only OHLCV
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("open", "high", "low", "close", "volume"):
            col_map[c] = cl
    df = df.rename(columns=col_map)[["open", "high", "low", "close", "volume"]].copy()

    # NO RTH filter — keep everything
    # Remove weekend bars and zero-volume bars
    df = df[df.index.weekday < 5]
    df = df[df["volume"] > 0]

    logger.info(f"Full session bars: {len(df)}")

    # Show hour distribution
    hours = df.index.hour.value_counts().sort_index()
    logger.info("Bars by hour:")
    for h, c in hours.items():
        logger.info(f"  {h:02d}:00  {c:>7} bars")

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info(f"Saved to {cache_path}")

    return df


if __name__ == "__main__":
    df = fetch_es_full_session()
    print(f"\nTotal bars: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nRTH bars: {len(df.between_time('09:30', '15:59'))}")
    print(f"Overnight bars: {len(df) - len(df.between_time('09:30', '15:59'))}")
