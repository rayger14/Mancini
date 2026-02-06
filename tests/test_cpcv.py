"""Tests for the CPCV framework: splitter, data pipeline, determinism."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.cpcv.splitter import CPCVSplitter, CPCVConfig, CPCVSplit
from backtest.cpcv.data_pipeline import split_into_daily_dfs
from backtest.cpcv.determinism import verify_determinism


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trading_days(n: int, start: date = date(2024, 1, 2)) -> list[date]:
    """Generate n weekday dates."""
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _make_synthetic_daily_dfs(
    n_days: int = 20,
    bars_per_day: int = 7,
    start_date: date = date(2024, 1, 2),
    base_price: float = 5000.0,
) -> dict[date, pd.DataFrame]:
    """Build synthetic hourly-like daily DataFrames for testing."""
    days = _make_trading_days(n_days, start_date)
    daily_dfs = {}
    price = base_price

    for d in days:
        start_dt = datetime(d.year, d.month, d.day, 9, 30)
        bars = []
        for j in range(bars_per_day):
            o = price
            c = price + np.random.uniform(-2, 3)
            h = max(o, c) + np.random.uniform(0, 1)
            lo = min(o, c) - np.random.uniform(0, 1)
            bars.append((o, h, lo, c, 1000))
            price = c

        idx = pd.date_range(start=start_dt, periods=bars_per_day, freq="1h")
        daily_dfs[d] = pd.DataFrame(
            bars, index=idx, columns=["open", "high", "low", "close", "volume"]
        )

    return daily_dfs


# ---------------------------------------------------------------------------
# Splitter tests
# ---------------------------------------------------------------------------


class TestCPCVSplitter:

    def test_correct_number_of_splits(self):
        days = _make_trading_days(100)
        splitter = CPCVSplitter(days, CPCVConfig(n_groups=10, n_test_groups=2))
        assert splitter.num_paths == 45  # C(10, 2)

    def test_groups_cover_all_days(self):
        days = _make_trading_days(103)  # non-divisible by 10
        splitter = CPCVSplitter(days, CPCVConfig(n_groups=10, n_test_groups=2))
        all_grouped = [d for g in splitter.groups for d in g]
        assert all_grouped == days

    def test_groups_are_contiguous(self):
        days = _make_trading_days(100)
        splitter = CPCVSplitter(days, CPCVConfig(n_groups=10, n_test_groups=2))
        for group in splitter.groups:
            indices = [days.index(d) for d in group]
            assert indices == list(range(indices[0], indices[-1] + 1))

    def test_no_overlap_train_test(self):
        days = _make_trading_days(100)
        config = CPCVConfig(n_groups=10, n_test_groups=2, purge_days=2, embargo_days=1)
        splitter = CPCVSplitter(days, config)
        for split in splitter.splits():
            train_set = set(split.train_dates)
            test_set = set(split.test_dates)
            assert train_set.isdisjoint(test_set), "Train and test overlap"
            assert train_set.isdisjoint(set(split.purged_dates)), "Train and purged overlap"

    def test_purge_removes_boundary_days(self):
        days = _make_trading_days(50)
        config = CPCVConfig(n_groups=5, n_test_groups=1, purge_days=2, embargo_days=0)
        splitter = CPCVSplitter(days, config)

        for split in splitter.splits():
            train_set = set(split.train_dates)
            first_test = split.test_dates[0]
            idx = days.index(first_test)
            # 2 days before first test day should NOT be in train
            for offset in range(1, 3):
                if idx - offset >= 0:
                    assert days[idx - offset] not in train_set, (
                        f"Day {days[idx - offset]} should be purged (before test)"
                    )

    def test_all_days_appear_in_test_correct_times(self):
        """Without purge/embargo, each day appears in test C(9,1)=9 times."""
        days = _make_trading_days(100)
        config = CPCVConfig(n_groups=10, n_test_groups=2, purge_days=0, embargo_days=0)
        splitter = CPCVSplitter(days, config)

        counts: Counter = Counter()
        for split in splitter.splits():
            for d in split.test_dates:
                counts[d] += 1

        # Each group appears in C(9,1) = 9 test combos
        assert all(c == 9 for c in counts.values())

    def test_summary_string(self):
        days = _make_trading_days(50)
        splitter = CPCVSplitter(days, CPCVConfig(n_groups=5, n_test_groups=2))
        summary = splitter.summary()
        assert "50 days" in summary
        assert "5 groups" in summary
        assert "10" in summary  # C(5,2) = 10


# ---------------------------------------------------------------------------
# Data pipeline tests
# ---------------------------------------------------------------------------


class TestDataPipeline:

    def test_split_daily_dfs_skips_short_days(self):
        """Days with fewer than min_bars should be dropped."""
        idx1 = pd.date_range("2024-01-02 09:30", periods=7, freq="1h")
        df1 = pd.DataFrame(
            {"open": 1, "high": 2, "low": 0, "close": 1, "volume": 100},
            index=idx1,
        )
        # Only 2 bars — should be skipped
        idx2 = pd.date_range("2024-01-03 09:30", periods=2, freq="1h")
        df2 = pd.DataFrame(
            {"open": 1, "high": 2, "low": 0, "close": 1, "volume": 100},
            index=idx2,
        )
        combined = pd.concat([df1, df2])
        result = split_into_daily_dfs(combined, min_bars=4)
        assert date(2024, 1, 2) in result
        assert date(2024, 1, 3) not in result

    def test_split_preserves_all_bars(self):
        """Total bars across all days should equal input bars (for days >= min_bars)."""
        idx = pd.date_range("2024-01-02 09:30", periods=21, freq="1h")
        df = pd.DataFrame(
            {"open": 1, "high": 2, "low": 0, "close": 1, "volume": 100},
            index=idx,
        )
        result = split_into_daily_dfs(df, min_bars=1)
        total_bars = sum(len(v) for v in result.values())
        assert total_bars == len(df)


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_determinism_with_synthetic_data(self):
        """Two runs on identical data should produce identical results."""
        np.random.seed(42)
        daily_dfs = _make_synthetic_daily_dfs(n_days=10, bars_per_day=7)

        # Use hourly-adapted params
        from config.settings import StrategyParams, ElevatorParams, RiskParams

        sp = StrategyParams(swing_low_order=3, non_acceptance_min_hold_bars=1)
        ep = ElevatorParams(velocity_window_bars=2, higher_low_lookback=2)
        rp = RiskParams(max_trades_per_day=99)

        assert verify_determinism(
            daily_dfs,
            n_runs=2,
            strategy_params=sp,
            elevator_params=ep,
            risk_params=rp,
        )


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


class TestRobustnessSmoke:

    def test_robustness_runs_end_to_end(self):
        """Smoke test: run robustness on synthetic data with minimal config."""
        np.random.seed(42)
        daily_dfs = _make_synthetic_daily_dfs(n_days=25, bars_per_day=7)
        trading_days = sorted(daily_dfs.keys())

        config = CPCVConfig(n_groups=5, n_test_groups=2, purge_days=1, embargo_days=0)
        splitter = CPCVSplitter(trading_days, config)
        assert splitter.num_paths == 10  # C(5, 2)

        from backtest.cpcv.robustness import RobustnessTest
        from backtest.cpcv.report import robustness_report
        from config.settings import StrategyParams, ElevatorParams, RiskParams

        sp = StrategyParams(swing_low_order=3, non_acceptance_min_hold_bars=1)
        ep = ElevatorParams(velocity_window_bars=2, higher_low_lookback=2)
        rp = RiskParams(max_trades_per_day=99)

        test = RobustnessTest(
            daily_dfs, splitter,
            strategy_params=sp,
            elevator_params=ep,
            risk_params=rp,
        )
        result = test.run()
        assert len(result.test_results) == 10

        report = robustness_report(result)
        assert "CPCV ROBUSTNESS REPORT" in report
