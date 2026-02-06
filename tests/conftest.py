"""Shared fixtures: synthetic OHLCV data and sample levels for testing."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams, ElevatorParams


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def make_bars(
    prices: list[tuple[float, float, float, float]],
    start: datetime | None = None,
    freq: str = "1min",
    volume: int = 1000,
) -> pd.DataFrame:
    """Create a DataFrame of OHLCV bars from a list of (O, H, L, C) tuples.

    Parameters
    ----------
    prices : list of (open, high, low, close)
    start : datetime, optional
        Start timestamp (default: 2024-01-15 09:30 ET).
    freq : str
        Bar frequency.
    volume : int
        Default volume per bar.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns [open, high, low, close, volume].
    """
    if start is None:
        start = datetime(2024, 1, 15, 9, 30)

    index = pd.date_range(start=start, periods=len(prices), freq=freq)
    data = {
        "open": [p[0] for p in prices],
        "high": [p[1] for p in prices],
        "low": [p[2] for p in prices],
        "close": [p[3] for p in prices],
        "volume": [volume] * len(prices),
    }
    return pd.DataFrame(data, index=index)


def make_selloff_then_recovery(
    start_price: float = 5000.0,
    selloff_bars: int = 10,
    selloff_rate: float = 3.0,
    low_price: float | None = None,
    sweep_below: float = 1.0,
    recovery_bars: int = 5,
    hold_bars: int = 3,
    start: datetime | None = None,
) -> pd.DataFrame:
    """Generate synthetic data: selloff → sweep below a level → recovery → hold.

    The 'significant low' will be at `start_price - selloff_bars * selloff_rate`.
    The sweep dips `sweep_below` points further, then recovers.

    Returns
    -------
    pd.DataFrame
    """
    if start is None:
        start = datetime(2024, 1, 15, 9, 30)

    prices: list[tuple[float, float, float, float]] = []
    price = start_price

    # Phase 1: Selloff (elevator down)
    for i in range(selloff_bars):
        o = price
        c = price - selloff_rate
        h = o + 0.5
        l = c - 0.5
        prices.append((o, h, l, c))
        price = c

    significant_low = price
    if low_price is not None:
        significant_low = low_price

    # Phase 2: Sweep below significant low
    sweep_price = significant_low - sweep_below
    o = price
    c = sweep_price + 0.25  # close just below significant low
    h = o + 0.25
    l = sweep_price
    prices.append((o, h, l, c))
    price = c

    # Phase 3: Recovery back above level
    recovery_step = (significant_low + 5.0 - price) / max(recovery_bars, 1)
    for i in range(recovery_bars):
        o = price
        c = price + recovery_step
        h = c + 0.5
        l = o - 0.25
        prices.append((o, h, l, c))
        price = c

    # Phase 4: Hold above level
    for i in range(hold_bars):
        o = price
        c = price + 0.5
        h = c + 0.5
        l = o - 0.25
        prices.append((o, h, l, c))
        price = c

    return make_bars(prices, start=start)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_bars() -> pd.DataFrame:
    """10 bars of simple uptrending data."""
    prices = []
    p = 5000.0
    for _ in range(10):
        o = p
        h = p + 2
        l = p - 1
        c = p + 1
        prices.append((o, h, l, c))
        p = c
    return make_bars(prices)


@pytest.fixture
def selloff_recovery_bars() -> pd.DataFrame:
    """Synthetic elevator down → sweep → recovery → hold sequence."""
    return make_selloff_then_recovery(
        start_price=5050.0,
        selloff_bars=10,
        selloff_rate=3.0,
        sweep_below=1.5,
        recovery_bars=5,
        hold_bars=5,
    )


@pytest.fixture
def sample_level_store() -> LevelStore:
    """A LevelStore with a few significant levels."""
    store = LevelStore()
    base_time = datetime(2024, 1, 15, 9, 0)
    store.add(
        Level(
            price=5020.0,
            level_type=LevelType.PRIOR_DAY_LOW,
            created_at=base_time,
            confirmed_at=base_time,
        )
    )
    store.add(
        Level(
            price=5000.0,
            level_type=LevelType.CLUSTER_LOW,
            created_at=base_time,
            confirmed_at=base_time,
            touch_count=4,
        )
    )
    store.add(
        Level(
            price=5040.0,
            level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time,
            confirmed_at=base_time,
            touch_count=5,
        )
    )
    store.add(
        Level(
            price=5060.0,
            level_type=LevelType.HORIZONTAL_SR,
            created_at=base_time,
            confirmed_at=base_time,
            touch_count=3,
        )
    )
    return store


@pytest.fixture
def strategy_params() -> StrategyParams:
    """Default strategy parameters for testing."""
    return StrategyParams()


@pytest.fixture
def elevator_params() -> ElevatorParams:
    """Default elevator parameters for testing."""
    return ElevatorParams()
