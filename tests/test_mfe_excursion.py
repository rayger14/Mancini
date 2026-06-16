"""Tests for MFE/MAE excursion computation at exit.

Background (2026-06-16 audit): logged MFE was systematically understated.
On the 12-min delayed feed the venue bracket closes the position at the
target BEFORE the bars that printed the true extreme arrive, so the
in-trade high/low watermark freezes near the exit price. Worse, the
recorded MFE sometimes came in *below* the realized favorable excursion
(you can't peak-favorably less than where you actually exited in profit).

`compute_excursion_pts` guarantees MFE/MAE never undershoot the realized
exit fill, and folds in the most recent bar's high/low.
"""
from __future__ import annotations

from live.ib_runner import compute_excursion_pts


class TestComputeExcursion:
    def test_long_basic_watermark(self):
        # Long: MFE from highest, MAE from lowest.
        mfe, mae = compute_excursion_pts(
            direction="long", entry_price=7000.0,
            highest=7030.0, lowest=6990.0,
            exit_price=7025.0, recent_high=None, recent_low=None)
        assert mfe == 30.0
        assert mae == 10.0

    def test_short_basic_watermark(self):
        # Short: MFE from lowest, MAE from highest.
        mfe, mae = compute_excursion_pts(
            direction="short", entry_price=7553.25,
            highest=7556.25, lowest=7516.5,
            exit_price=7518.5, recent_high=None, recent_low=None)
        assert mfe == 36.75  # 7553.25 - 7516.5
        assert mae == 3.0    # 7556.25 - 7553.25

    def test_short_mfe_floored_at_exit_fill(self):
        # The live 28474 bug: watermark lowest (7523.75) is ABOVE the exit
        # fill (7518.5) because the venue closed us before the low bar
        # arrived. MFE must not undershoot the realized exit excursion.
        mfe, mae = compute_excursion_pts(
            direction="short", entry_price=7553.25,
            highest=7556.25, lowest=7523.75,
            exit_price=7518.5, recent_high=None, recent_low=None)
        assert mfe == 34.75  # floored at entry-exit, not the stale 29.5

    def test_long_mfe_floored_at_exit_fill(self):
        mfe, _ = compute_excursion_pts(
            direction="long", entry_price=7399.0,
            highest=7421.75, lowest=7399.0,
            exit_price=7424.75, recent_high=None, recent_low=None)
        assert mfe == 25.75  # exit above watermark high -> use exit

    def test_recent_bar_extends_excursion(self):
        # A late-arriving bar print beyond the watermark is folded in.
        mfe, mae = compute_excursion_pts(
            direction="long", entry_price=7000.0,
            highest=7020.0, lowest=6995.0,
            exit_price=7010.0, recent_high=7035.0, recent_low=6985.0)
        assert mfe == 35.0
        assert mae == 15.0

    def test_mae_never_negative_when_price_stayed_favorable(self):
        # 26954: price never traded below the long entry -> MAE is 0, not a
        # spurious positive. lowest == entry, exit above.
        _, mae = compute_excursion_pts(
            direction="long", entry_price=7399.0,
            highest=7428.0, lowest=7399.0,
            exit_price=7424.75, recent_high=None, recent_low=None)
        assert mae == 0.0

    def test_missing_watermark_falls_back_to_exit(self):
        mfe, mae = compute_excursion_pts(
            direction="long", entry_price=7000.0,
            highest=None, lowest=None,
            exit_price=7012.0, recent_high=None, recent_low=None)
        assert mfe == 12.0
        assert mae == 0.0

    def test_no_data_at_all_returns_none(self):
        mfe, mae = compute_excursion_pts(
            direction="long", entry_price=7000.0,
            highest=None, lowest=None,
            exit_price=None, recent_high=None, recent_low=None)
        assert mfe is None
        assert mae is None
