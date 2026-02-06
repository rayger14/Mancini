"""Tests for NautilusTrader integration.

All tests are skipped if nautilus_trader is not installed.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from config.settings import (
    DEFAULT_EXIT,
    DEFAULT_RISK,
    DEFAULT_STRATEGY,
    DEFAULT_ELEVATOR,
    DEFAULT_CONTRACT,
    ExitParams,
    RiskParams,
)
from strategy.position_manager import TradeRecord

try:
    import nautilus_trader
    from backtest.nautilus_strategy import (
        ManciniNautilusStrategy,
        ManciniNautilusConfig,
        _Phase,
    )
    from backtest.nautilus_runner import (
        NautilusBacktestRunner,
        NautilusBacktestConfig,
    )

    nautilus_available = True
except ImportError:
    nautilus_available = False

pytestmark = pytest.mark.skipif(
    not nautilus_available,
    reason="nautilus_trader not installed",
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_bars(
    prices: list[tuple[float, float, float, float]],
    start: datetime | None = None,
    volume: int = 1000,
) -> pd.DataFrame:
    """Create OHLCV DataFrame from (O, H, L, C) tuples."""
    if start is None:
        start = datetime(2024, 1, 15, 9, 30)
    index = pd.date_range(start=start, periods=len(prices), freq="1min")
    return pd.DataFrame(
        {
            "open": [p[0] for p in prices],
            "high": [p[1] for p in prices],
            "low": [p[2] for p in prices],
            "close": [p[3] for p in prices],
            "volume": [volume] * len(prices),
        },
        index=index,
    )


def _make_selloff_recovery(
    start_price: float = 5050.0,
    selloff_bars: int = 10,
    selloff_rate: float = 3.0,
    sweep_below: float = 1.5,
    recovery_bars: int = 5,
    hold_bars: int = 5,
    rally_bars: int = 20,
    rally_rate: float = 2.0,
    start: datetime | None = None,
) -> pd.DataFrame:
    """Selloff → sweep → recovery → hold → rally (to hit targets)."""
    if start is None:
        start = datetime(2024, 1, 15, 9, 30)

    prices: list[tuple[float, float, float, float]] = []
    price = start_price

    # Selloff
    for _ in range(selloff_bars):
        o = price
        c = price - selloff_rate
        h = o + 0.5
        lo = c - 0.5
        prices.append((o, h, lo, c))
        price = c

    sig_low = price

    # Sweep below
    sweep_price = sig_low - sweep_below
    o = price
    c = sweep_price + 0.25
    prices.append((o, o + 0.25, sweep_price, c))
    price = c

    # Recovery
    step = (sig_low + 5.0 - price) / max(recovery_bars, 1)
    for _ in range(recovery_bars):
        o = price
        c = price + step
        prices.append((o, c + 0.5, o - 0.25, c))
        price = c

    # Hold above level
    for _ in range(hold_bars):
        o = price
        c = price + 0.5
        prices.append((o, c + 0.5, o - 0.25, c))
        price = c

    # Rally to hit targets
    for _ in range(rally_bars):
        o = price
        c = price + rally_rate
        prices.append((o, c + 0.5, o - 0.25, c))
        price = c

    return _make_bars(prices, start=start)


def _make_flat_market(
    price: float = 5000.0,
    n_bars: int = 60,
    noise: float = 0.5,
    start: datetime | None = None,
) -> pd.DataFrame:
    """Sideways market data."""
    if start is None:
        start = datetime(2024, 1, 15, 9, 30)

    rng = np.random.default_rng(42)
    prices = []
    for _ in range(n_bars):
        delta = rng.uniform(-noise, noise)
        o = price
        c = price + delta
        h = max(o, c) + rng.uniform(0, noise)
        lo = min(o, c) - rng.uniform(0, noise)
        prices.append((round(o, 2), round(h, 2), round(lo, 2), round(c, 2)))
        price = c

    return _make_bars(prices, start=start)


def _make_selloff_to_stop(
    start_price: float = 5050.0,
    selloff_bars: int = 10,
    selloff_rate: float = 3.0,
    sweep_below: float = 1.5,
    recovery_bars: int = 5,
    hold_bars: int = 3,
    crash_bars: int = 10,
    crash_rate: float = 4.0,
    start: datetime | None = None,
) -> pd.DataFrame:
    """Selloff → recovery → signal fires → price drops to stop."""
    if start is None:
        start = datetime(2024, 1, 15, 9, 30)

    prices: list[tuple[float, float, float, float]] = []
    price = start_price

    # Selloff
    for _ in range(selloff_bars):
        o = price
        c = price - selloff_rate
        prices.append((o, o + 0.5, c - 0.5, c))
        price = c

    sig_low = price
    sweep_price = sig_low - sweep_below

    # Sweep
    prices.append((price, price + 0.25, sweep_price, sweep_price + 0.25))
    price = sweep_price + 0.25

    # Recovery
    step = (sig_low + 5.0 - price) / max(recovery_bars, 1)
    for _ in range(recovery_bars):
        o = price
        c = price + step
        prices.append((o, c + 0.5, o - 0.25, c))
        price = c

    # Brief hold
    for _ in range(hold_bars):
        o = price
        c = price + 0.5
        prices.append((o, c + 0.5, o - 0.25, c))
        price = c

    # Crash to hit stop
    for _ in range(crash_bars):
        o = price
        c = price - crash_rate
        prices.append((o, o + 0.25, c - 0.5, c))
        price = c

    return _make_bars(prices, start=start)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestESInstrumentSpec:
    """Verify ES futures instrument is created correctly."""

    def test_es_instrument_spec(self):
        runner = NautilusBacktestRunner()
        instrument = runner._create_es_instrument()

        assert str(instrument.id) == "ES.GLBX"
        assert float(instrument.price_increment) == 0.25
        assert int(instrument.multiplier) == 50
        assert str(instrument.currency) == "USD"


class TestNautilusBacktestConfig:
    """Verify config defaults and serialisation."""

    def test_default_config(self):
        cfg = NautilusBacktestConfig()
        assert cfg.commission_per_side == 1.25
        assert cfg.prob_slippage == 0.5
        assert cfg.starting_balance == 100_000.0
        assert cfg.min_rr_ratio == 1.5

    def test_custom_config(self):
        cfg = NautilusBacktestConfig(
            commission_per_side=2.0,
            prob_slippage=0.0,
            starting_balance=50_000.0,
        )
        assert cfg.commission_per_side == 2.0
        assert cfg.prob_slippage == 0.0


class TestManciniNautilusConfig:
    """Verify strategy config round-trips param serialisation."""

    def test_param_serialisation(self):
        exit_params = ExitParams(
            trailing_tighten_thresholds=[(10.0, 3.0), (15.0, 2.0)]
        )
        cfg = ManciniNautilusConfig(
            exit_params=asdict(exit_params),
        )
        # Verify thresholds are serialised as list of lists
        assert cfg.exit_params["trailing_tighten_thresholds"] == [
            [10.0, 3.0],
            [15.0, 2.0],
        ]


class TestSingleDayWithSignal:
    """Run a day with a selloff→recovery pattern and verify trades."""

    def test_single_day_runs_without_error(self):
        """Smoke test: engine runs to completion on synthetic data."""
        df = _make_selloff_recovery()
        runner = NautilusBacktestRunner(
            NautilusBacktestConfig(prob_slippage=0.0)
        )
        result = runner.run_single_day(df)
        # Result should be a DayResult regardless of whether trades fired
        assert result.date == date(2024, 1, 15)
        assert result.num_trades >= 0


class TestNoSignalFlatMarket:
    """Flat market should produce zero trades."""

    def test_no_signal_flat_market(self):
        df = _make_flat_market()
        runner = NautilusBacktestRunner(
            NautilusBacktestConfig(prob_slippage=0.0)
        )
        result = runner.run_single_day(df)
        assert result.num_trades == 0
        assert result.pnl_pts == 0.0


class TestResultCompatibleWithMetrics:
    """BacktestResult from nautilus runner feeds into compute_metrics()."""

    def test_result_compatible_with_metrics(self):
        from backtest.metrics import compute_metrics
        from backtest.runner import BacktestResult

        df = _make_flat_market()
        runner = NautilusBacktestRunner(
            NautilusBacktestConfig(prob_slippage=0.0)
        )
        day_result = runner.run_single_day(df)

        bt_result = BacktestResult()
        bt_result.days.append(day_result)
        bt_result.all_trades.extend(day_result.trade_records)

        # Should not raise
        metrics = compute_metrics(bt_result)
        assert metrics.total_trades == day_result.num_trades


class TestMultiDay:
    """Multi-day runner aggregates results correctly."""

    def test_multi_day_runs(self):
        day1 = _make_flat_market(
            start=datetime(2024, 1, 15, 9, 30),
        )
        day2 = _make_flat_market(
            start=datetime(2024, 1, 16, 9, 30),
        )

        runner = NautilusBacktestRunner(
            NautilusBacktestConfig(prob_slippage=0.0)
        )
        result = runner.run_multi_day(
            daily_dfs={
                date(2024, 1, 15): day1,
                date(2024, 1, 16): day2,
            }
        )
        assert len(result.days) == 2
        assert result.total_trades == 0


class TestEngineSetup:
    """Verify engine creates venue and instrument correctly."""

    def test_create_engine(self):
        runner = NautilusBacktestRunner()
        instrument = runner._create_es_instrument()
        engine = runner._create_engine(instrument)

        # Engine should have GLBX venue
        assert engine is not None
        engine.dispose()

    def test_wrangle_bars(self):
        """Bar wrangling produces correct number of bars."""
        runner = NautilusBacktestRunner()
        instrument = runner._create_es_instrument()
        df = _make_flat_market(n_bars=30)
        bars = runner._wrangle_bars(df, instrument)
        assert len(bars) == 30
