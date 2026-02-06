"""Data fetching (Databento/Polygon) + local Parquet caching."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


_CACHE_DIR = Path("data")


class MarketDataFetcher:
    """Fetches ES futures OHLCV data with local Parquet caching."""

    def __init__(
        self,
        cache_dir: Path | str = _CACHE_DIR,
        api_key: Optional[str] = None,
        dataset: str = "GLBX.MDP3",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("DATABENTO_API_KEY")
        self.dataset = dataset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        bar_size: str = "1min",
    ) -> pd.DataFrame:
        """Return OHLCV bars, loading from cache or fetching from Databento.

        Parameters
        ----------
        symbol : str
            Root symbol, e.g. "ES".
        start, end : date
            Inclusive date range.
        bar_size : str
            Bar frequency: "1min", "5min", "1h", etc.

        Returns
        -------
        pd.DataFrame
            Columns: open, high, low, close, volume, with DatetimeIndex (UTC).
        """
        cache_path = self._cache_path(symbol, start, end, bar_size)
        if cache_path.exists():
            logger.info(f"Loading cached data: {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Fetching {symbol} {bar_size} bars {start}→{end} from Databento")
        df = self._fetch_databento(symbol, start, end, bar_size)
        df.to_parquet(cache_path)
        logger.info(f"Cached {len(df)} bars → {cache_path}")
        return df

    def get_single_day(
        self, symbol: str, day: date, bar_size: str = "1min"
    ) -> pd.DataFrame:
        """Convenience: fetch a single day of data."""
        return self.get_bars(symbol, day, day, bar_size)

    # ------------------------------------------------------------------
    # Data Sources
    # ------------------------------------------------------------------

    def _fetch_databento(
        self, symbol: str, start: date, end: date, bar_size: str
    ) -> pd.DataFrame:
        """Fetch data from Databento API."""
        try:
            import databento as db
        except ImportError:
            raise ImportError(
                "databento package required. Install with: pip install databento"
            )

        if not self.api_key:
            raise ValueError(
                "Databento API key required. Set DATABENTO_API_KEY env var "
                "or pass api_key to MarketDataFetcher."
            )

        client = db.Historical(self.api_key)

        schema_map = {
            "1min": "ohlcv-1m",
            "5min": "ohlcv-5m",
            "1h": "ohlcv-1h",
            "1d": "ohlcv-1d",
        }
        schema = schema_map.get(bar_size)
        if schema is None:
            raise ValueError(f"Unsupported bar_size: {bar_size}")

        # Databento uses continuous front-month symbol
        stype = "continuous"
        db_symbol = f"{symbol}.FUT"

        data = client.timeseries.get_range(
            dataset=self.dataset,
            symbols=[db_symbol],
            schema=schema,
            start=datetime.combine(start, datetime.min.time()).isoformat(),
            end=datetime.combine(end + timedelta(days=1), datetime.min.time()).isoformat(),
            stype_in=stype,
        )

        df = data.to_df()
        return self._normalize_df(df)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a DataFrame to standard OHLCV format."""
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ("open", "high", "low", "close", "volume"):
                col_map[col] = lower

        if col_map:
            df = df.rename(columns=col_map)

        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df[required].copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts_event" in df.columns:
                df.index = pd.to_datetime(df["ts_event"])
            elif "timestamp" in df.columns:
                df.index = pd.to_datetime(df["timestamp"])
        df.index.name = "timestamp"
        df = df.sort_index()
        return df

    def _cache_path(
        self, symbol: str, start: date, end: date, bar_size: str
    ) -> Path:
        """Build a deterministic cache file path."""
        fname = f"{symbol}_{bar_size}_{start.isoformat()}_{end.isoformat()}.parquet"
        return self.cache_dir / fname

    @staticmethod
    def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
        """Load bars from a CSV file (useful for testing/custom data)."""
        df = pd.read_csv(path, parse_dates=True, index_col=0, **kwargs)
        df.columns = [c.lower().strip() for c in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        df.index.name = "timestamp"
        return df[required]
