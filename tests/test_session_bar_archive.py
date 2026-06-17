"""Tests for full-session bar archival.

2026-06-16 audit: the session parquet only held the last 400 bars of the
live rolling window (~10:17-16:58 ET), so overnight/pre-market bars — where
the BEST trades live (Pre-Market FB longs, +388 pts / 90% win) — were never
persisted and couldn't be audited. build_session_bars_df assembles the
*entire* session from an uncapped accumulator.
"""
from __future__ import annotations

import pandas as pd

from live.ib_runner import build_session_bars_df


def _ts(hhmm):
    return pd.Timestamp(f"2026-06-16 {hhmm}", tz="US/Eastern")


class TestBuildSessionBarsDf:
    def test_empty_returns_empty_frame(self):
        df = build_session_bars_df({})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_columns_and_values(self):
        acc = {_ts("09:30"): (7000.0, 7005.0, 6999.0, 7004.0, 120.0)}
        df = build_session_bars_df(acc)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.iloc[0]["high"] == 7005.0
        assert df.iloc[0]["volume"] == 120.0

    def test_sorted_by_timestamp_regardless_of_insertion_order(self):
        acc = {}
        acc[_ts("15:00")] = (1, 1, 1, 1, 1)
        acc[_ts("09:30")] = (2, 2, 2, 2, 2)
        acc[_ts("12:00")] = (3, 3, 3, 3, 3)
        df = build_session_bars_df(acc)
        assert list(df.index) == [_ts("09:30"), _ts("12:00"), _ts("15:00")]

    def test_dedup_keeps_last_write_per_timestamp(self):
        acc = {}
        acc[_ts("09:30")] = (1, 1, 1, 1, 1)
        acc[_ts("09:30")] = (9, 9, 9, 9, 9)  # corrected bar overwrites
        df = build_session_bars_df(acc)
        assert len(df) == 1
        assert df.iloc[0]["close"] == 9

    def test_not_capped_at_400(self):
        # The whole point: a full Globex session (~1380 1-min bars) must be
        # kept in full, not trimmed to the live 400-bar window.
        acc = {
            pd.Timestamp("2026-06-16 00:00", tz="US/Eastern") + pd.Timedelta(minutes=i):
            (i, i, i, i, i)
            for i in range(1000)
        }
        df = build_session_bars_df(acc)
        assert len(df) == 1000

    def test_index_is_datetimeindex(self):
        acc = {_ts("09:30"): (1, 1, 1, 1, 1)}
        df = build_session_bars_df(acc)
        assert isinstance(df.index, pd.DatetimeIndex)
