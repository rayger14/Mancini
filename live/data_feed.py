"""Real-time data subscription via Databento."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
from loguru import logger


@dataclass
class DataFeedConfig:
    """Configuration for real-time data feed."""

    api_key: Optional[str] = None
    dataset: str = "GLBX.MDP3"
    symbol: str = "ES.FUT"
    schema: str = "ohlcv-1m"

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("DATABENTO_API_KEY")


@dataclass
class BarData:
    """A single real-time bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class DataFeed:
    """Real-time data subscription using Databento Live API.

    Subscribes to 1-minute OHLCV bars and invokes a callback on each new bar.
    """

    def __init__(self, config: DataFeedConfig = DataFeedConfig()):
        self.config = config
        self._client = None
        self._running = False
        self._bars: list[BarData] = []
        self._callbacks: list[Callable[[BarData], None]] = []

    def add_callback(self, callback: Callable[[BarData], None]) -> None:
        """Register a callback to be invoked on each new bar."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start the real-time data subscription."""
        try:
            import databento as db
        except ImportError:
            raise ImportError("databento required: pip install databento")

        if not self.config.api_key:
            raise ValueError("Databento API key required")

        logger.info(f"Starting data feed: {self.config.symbol} {self.config.schema}")
        self._client = db.Live(self.config.api_key)
        self._client.subscribe(
            dataset=self.config.dataset,
            schema=self.config.schema,
            symbols=[self.config.symbol],
            stype_in="continuous",
        )
        self._running = True

        for record in self._client:
            if not self._running:
                break

            bar = self._record_to_bar(record)
            if bar is not None:
                self._bars.append(bar)
                for cb in self._callbacks:
                    try:
                        cb(bar)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

    def stop(self) -> None:
        """Stop the data subscription."""
        self._running = False
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        logger.info("Data feed stopped")

    def get_history_df(self) -> pd.DataFrame:
        """Get all received bars as a DataFrame."""
        if not self._bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        data = {
            "open": [b.open for b in self._bars],
            "high": [b.high for b in self._bars],
            "low": [b.low for b in self._bars],
            "close": [b.close for b in self._bars],
            "volume": [b.volume for b in self._bars],
        }
        index = pd.DatetimeIndex([b.timestamp for b in self._bars], name="timestamp")
        return pd.DataFrame(data, index=index)

    @property
    def bar_count(self) -> int:
        return len(self._bars)

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def _record_to_bar(record) -> Optional[BarData]:
        """Convert a Databento record to BarData."""
        try:
            return BarData(
                timestamp=record.ts_event if hasattr(record, "ts_event") else datetime.utcnow(),
                open=float(record.open) / 1e9 if record.open > 1e6 else float(record.open),
                high=float(record.high) / 1e9 if record.high > 1e6 else float(record.high),
                low=float(record.low) / 1e9 if record.low > 1e6 else float(record.low),
                close=float(record.close) / 1e9 if record.close > 1e6 else float(record.close),
                volume=int(record.volume),
            )
        except (AttributeError, TypeError):
            return None
