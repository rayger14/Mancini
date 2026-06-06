"""Empirically tune the Mode 1 Red detector against bar-data ground truth.

Mancini's definition (verbatim):
  "Mode 1 red day = open to close trend day to the downside"
  "Open to close sell"
  "Wait patiently all day for the sell to complete"

Structural ground truth (RTH 9:30-16:00 ET, applied per session):
  ALL of the following must hold:
    1. RTH close < RTH open
    2. Day's range (high - low) > 0.5% of open
    3. Close is within bottom 20% of the day's range (close to low)
    4. Open is within top 30% of the day's range (open near high)
    5. No mid-day bounce > 50% retrace of the open-to-low decline
       (true open-to-close trend, not V-shape)

Then run the existing Mode1Detector on every session and report:
  * precision (when detector fires, how often is it a true Mode 1 Red?)
  * recall    (of all true Mode 1 Red days, how many does it catch?)
  * F1

Grid-search threshold combinations to maximise F1 — output the best params.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, time as dt_time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

import pandas as pd

from backtest.nautilus_production_5y import load_data, build_daily_sessions
from core.mode1_detector import Mode1Detector
from core.signals import SignalAggregator
from core.indicators import enrich_dataframe
from config.settings import StrategyParams
from live.ib_runner import (
    PRODUCTION_STRATEGY, PRODUCTION_ELEVATOR, PRODUCTION_EXIT, PRODUCTION_RISK,
)


@dataclass
class RTHStats:
    rth_open: float
    rth_close: float
    rth_high: float
    rth_low: float
    range_pts: float
    range_pct: float  # range / open
    close_position: float  # 0.0 = at low, 1.0 = at high
    open_position: float   # 0.0 = at low, 1.0 = at high
    deepest_bounce_pct: float  # post-low bounce / open-to-low decline
    low_bar_position: float    # 0.0 = low at open, 1.0 = low at close


def compute_rth_stats(df: pd.DataFrame) -> RTHStats | None:
    """Compute Mode-1-Red structural stats over RTH hours.

    Bars are 1-min, timestamps tz-aware ET. RTH = 09:30-16:00 ET inclusive.
    """
    rth_mask = df.index.map(
        lambda ts: dt_time(9, 30) <= ts.time() <= dt_time(15, 59)
    )
    rth_df = df[rth_mask]
    if len(rth_df) < 100:
        return None

    rth_open = float(rth_df["open"].iat[0])
    rth_close = float(rth_df["close"].iat[-1])
    rth_high = float(rth_df["high"].max())
    rth_low = float(rth_df["low"].min())
    rng = rth_high - rth_low
    if rng <= 0:
        return None
    range_pct = rng / rth_open

    close_pos = (rth_close - rth_low) / rng
    open_pos = (rth_open - rth_low) / rng

    # "Bounce" = the largest counter-trend rally AFTER the low has been
    # made. A true Mode 1 Red day has the low at/near the close, so there
    # is little-to-no post-low recovery. A V-shape day has a big post-low
    # rally. We measure: max(high) after the low-bar minus the low,
    # normalised by the open→low decline.
    low_at_bar_no = int(rth_df["low"].argmin())
    after_low = rth_df.iloc[low_at_bar_no + 1:]
    decline = max(0.0, rth_open - rth_low)
    if decline <= 0 or len(after_low) == 0:
        deepest_bounce_pct = 0.0
    else:
        max_high_after_low = float(after_low["high"].max())
        bounce_pts = max(0.0, max_high_after_low - rth_low)
        deepest_bounce_pct = bounce_pts / decline
    # Also: WHEN did the low occur? Mancini's Mode 1 Red has the low
    # at or very near the close (last 30 min of RTH = bottom 8% of session).
    low_bar_position = low_at_bar_no / max(len(rth_df) - 1, 1)

    return RTHStats(
        rth_open=rth_open,
        rth_close=rth_close,
        rth_high=rth_high,
        rth_low=rth_low,
        range_pts=rng,
        range_pct=range_pct,
        close_position=close_pos,
        open_position=open_pos,
        deepest_bounce_pct=deepest_bounce_pct,
        low_bar_position=low_bar_position,
    )


def is_mode1_red_structural(s: RTHStats,
                            *,
                            min_close_minus_open_pct: float = 0.003,
                            min_range_pct: float = 0.005,
                            max_close_position: float = 0.25,
                            min_open_position: float = 0.65,
                            max_bounce_pct: float = 0.50,
                            min_low_bar_position: float = 0.50) -> bool:
    """Structural classifier for Mode 1 Red day (no detector involved).

    All conditions must hold:
      * RTH close at least 0.3% below RTH open (real down day)
      * Day's range at least 0.5% of open (real volatility)
      * Close within bottom 25% of the day's range
      * Open within top 35% of the day's range
      * Post-low bounce < 50% of the open-to-low decline
      * Low happened in the second half of the session
    """
    if s is None:
        return False
    if s.rth_close >= s.rth_open:
        return False
    move_pct = (s.rth_open - s.rth_close) / s.rth_open
    if move_pct < min_close_minus_open_pct:
        return False
    if s.range_pct < min_range_pct:
        return False
    if s.close_position > max_close_position:
        return False
    if s.open_position < min_open_position:
        return False
    if s.deepest_bounce_pct > max_bounce_pct:
        return False
    if s.low_bar_position < min_low_bar_position:
        return False
    return True


def run_detector(df: pd.DataFrame, prior_df: pd.DataFrame | None,
                 strategy_params: StrategyParams) -> dict:
    """Run SignalAggregator on the session; report whether Mode 1 Red
    fired and when. We use the aggregator (not the bare detector) because
    it sets PDL and updates the level store the detector reads from."""
    agg = SignalAggregator(
        strategy_params=strategy_params,
        elevator_params=PRODUCTION_ELEVATOR,
        exit_params=PRODUCTION_EXIT,
        min_rr_ratio=PRODUCTION_RISK.min_rr_ratio,
    )
    agg.reset()
    agg.initialize_levels(df, prior_df)

    enriched = enrich_dataframe(df)
    velocity = enriched["velocity_5"]

    first_fire_bar = None
    first_fire_time = None
    for i in range(len(df)):
        vel = float(velocity.iat[i])
        if vel != vel:
            vel = 0.0
        agg.update(
            bar_idx=i,
            timestamp=df.index[i],
            open_=float(df["open"].iat[i]),
            high=float(df["high"].iat[i]),
            low=float(df["low"].iat[i]),
            close=float(df["close"].iat[i]),
            volume=float(df["volume"].iat[i]),
            velocity=vel,
            df=df,
        )
        if agg.mode1_red_active and first_fire_bar is None:
            first_fire_bar = i
            first_fire_time = df.index[i].time()
    return {
        "fired": first_fire_bar is not None,
        "first_fire_bar": first_fire_bar,
        "first_fire_time": str(first_fire_time) if first_fire_time else None,
    }


@dataclass
class Eval:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        if self.tp == 0:
            return 0.0
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-06-27")
    ap.add_argument("--end", default="2026-02-05")
    ap.add_argument(
        "--mancini-dates",
        default="data/training/mancini_mode1_red_days.json",
    )
    ap.add_argument("--grid-search", action="store_true",
                    help="Grid-search threshold combinations and print top 10")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print("Loading 5y data…")
    sessions = build_daily_sessions(load_data())
    target_days = sorted(d for d in sessions if start <= d <= end)
    print(f"Sessions in window: {len(target_days)}")

    mancini_dates = set()
    try:
        for x in json.loads(Path(args.mancini_dates).read_text()):
            mancini_dates.add(x["date"])
    except (json.JSONDecodeError, OSError):
        pass
    print(f"Mancini-labeled Mode 1 Red days (raw, may be noisy): "
          f"{len(mancini_dates)}")

    # --- 1. Structural ground truth ---
    structural_red: set[str] = set()
    prior = None
    rth_cache: dict[str, RTHStats] = {}
    for d in target_days:
        df_sess = sessions[d]
        if len(df_sess) < 30:
            prior = d
            continue
        s = compute_rth_stats(df_sess)
        if s is None:
            prior = d
            continue
        rth_cache[d.isoformat()] = s
        if is_mode1_red_structural(s):
            structural_red.add(d.isoformat())
        prior = d
    print(f"\nStructural Mode 1 Red days: {len(structural_red)} "
          f"({len(structural_red)/len(rth_cache)*100:.1f}% of sessions)")
    # Sort + print the days with their RTH stats
    print(f"\n{'date':<12} {'open':>7} {'close':>7} {'low':>7} "
          f"{'%down':>6} {'cls%':>5} {'op%':>5} {'lowAt%':>6} {'bnc%':>5}")
    for d_iso in sorted(structural_red):
        s = rth_cache[d_iso]
        pct_down = (s.rth_open - s.rth_close) / s.rth_open * 100
        print(f"{d_iso:<12} {s.rth_open:>7.2f} {s.rth_close:>7.2f} "
              f"{s.rth_low:>7.2f} {pct_down:>5.2f}% "
              f"{s.close_position*100:>4.0f}% {s.open_position*100:>4.0f}% "
              f"{s.low_bar_position*100:>5.0f}% "
              f"{s.deepest_bounce_pct*100:>4.0f}%")
    # Compare with Mancini-text labels (informational)
    overlap = structural_red & mancini_dates
    print(f"\nOverlap with Mancini-text labels: {len(overlap)} / "
          f"{len(mancini_dates)} text labels (text labels are noisy)")

    # --- 2. Evaluate the current production detector ---
    print("\nRunning detector (default thresholds) over all sessions…")
    detector_fires: dict[str, dict] = {}
    prior = None
    for i, d in enumerate(target_days):
        df_sess = sessions[d]
        if len(df_sess) < 30:
            prior = d
            continue
        prior_df = sessions[prior] if prior is not None else None
        prod = replace(PRODUCTION_STRATEGY, use_mode1_detection=True)
        info = run_detector(df_sess, prior_df, prod)
        detector_fires[d.isoformat()] = info
        prior = d
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(target_days)}]", flush=True)

    # --- 3. Confusion matrix ---
    e = Eval()
    for d_iso in detector_fires:
        truth = d_iso in structural_red
        pred = detector_fires[d_iso]["fired"]
        if truth and pred:
            e.tp += 1
        elif pred and not truth:
            e.fp += 1
        elif truth and not pred:
            e.fn += 1
        else:
            e.tn += 1

    print(f"\nDETECTOR vs STRUCTURAL GROUND TRUTH (default thresholds):")
    print(f"  Sessions evaluated: {sum([e.tp, e.fp, e.fn, e.tn])}")
    print(f"  True positives (correct fire):   {e.tp}")
    print(f"  False positives (over-fire):     {e.fp}")
    print(f"  False negatives (missed):        {e.fn}")
    print(f"  True negatives (correct quiet):  {e.tn}")
    print()
    print(f"  Precision:  {e.precision*100:.1f}%")
    print(f"  Recall:     {e.recall*100:.1f}%")
    print(f"  F1:         {e.f1*100:.1f}%")
    print(f"  Structural Mode 1 Red days: {len(structural_red)} "
          f"({len(structural_red)/sum([e.tp,e.fp,e.fn,e.tn])*100:.1f}% of sessions)")

    # --- 4. Optional grid search ---
    if args.grid_search:
        print("\n" + "=" * 70)
        print("GRID SEARCH — tighter threshold combinations")
        print("=" * 70)
        results = []
        # Re-evaluate the detector with each combo. To avoid running the
        # detector hundreds of times, we apply post-hoc filters: require
        # that BOTH a bars_below_pdl threshold AND a bearish_pressure
        # threshold are simultaneously crossed (instead of "any 2 of 3").
        # The detector already exposes state; we compute the joint
        # condition from the bar-level data of each session.
        for min_below_pdl, min_pressure, levels_thr, hold_bars in itertools.product(
            (30, 45, 60, 90),
            (60, 90, 120),
            (3, 4, 5),
            (20, 30, 45),
        ):
            params = replace(
                PRODUCTION_STRATEGY,
                use_mode1_detection=True,
                mode1_min_bars_below_pdl=min_below_pdl,
                mode1_bearish_pressure_bars=min_pressure,
                mode1_levels_broken_threshold=levels_thr,
                mode1_level_broken_hold_bars=hold_bars,
            )
            ee = Eval()
            prior = None
            for d in target_days:
                df_sess = sessions[d]
                if len(df_sess) < 30:
                    prior = d
                    continue
                prior_df = sessions[prior] if prior is not None else None
                info = run_detector(df_sess, prior_df, params)
                truth = d.isoformat() in structural_red
                pred = info["fired"]
                if truth and pred:
                    ee.tp += 1
                elif pred and not truth:
                    ee.fp += 1
                elif truth and not pred:
                    ee.fn += 1
                else:
                    ee.tn += 1
                prior = d
            results.append((ee.f1, ee.precision, ee.recall, ee.tp, ee.fp,
                            ee.fn, min_below_pdl, min_pressure,
                            levels_thr, hold_bars))
        # Top by F1
        results.sort(key=lambda r: -r[0])
        print(f"\n{'F1':>5} {'Prec':>6} {'Rec':>6} {'TP':>3} {'FP':>4} {'FN':>3} "
              f"{'belowPDL':>9} {'press':>6} {'lvls':>5} {'hold':>5}")
        for r in results[:10]:
            f1, p, rec, tp, fp, fn, bp, pr, lv, h = r
            print(f"{f1*100:>4.1f}% {p*100:>5.1f}% {rec*100:>5.1f}% "
                  f"{tp:>3} {fp:>4} {fn:>3} {bp:>9} {pr:>6} {lv:>5} {h:>5}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
