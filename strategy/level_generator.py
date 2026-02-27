#!/usr/bin/env python3
"""
Mancini-style level generator for ES futures.

Reverse-engineered from analysis of 500 Substack posts.
Generates daily support/resistance levels using price-action
structure (swing highs/lows, shelves, daily extremes).

No traditional indicators (VWAP, fibonacci, moving averages,
volume profile) are used -- consistent with Mancini's approach.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.signal import argrelextrema
from loguru import logger


@dataclass
class PriceLevel:
    """A support/resistance level with metadata."""
    price: float
    is_major: bool = False
    score: float = 0.0
    source: str = ""  # e.g. "daily_low", "swing_low", "shelf", "fill"
    description: str = ""
    touch_count: int = 0
    age_days: int = 0  # how many trading days old

    def __repr__(self):
        tag = "(major)" if self.is_major else ""
        return f"Level({self.price:.0f} {tag} src={self.source} score={self.score:.1f})"


@dataclass
class LevelSet:
    """Complete set of levels for a trading day."""
    supports: List[PriceLevel] = field(default_factory=list)
    resistances: List[PriceLevel] = field(default_factory=list)
    bull_bear_line: Optional[float] = None
    current_price: float = 0.0
    date: Optional[pd.Timestamp] = None


# ── Configuration ──────────────────────────────────────────────

LOOKBACK_DAYS = 10          # Days of daily data to consider
SWING_LOOKBACK_DAYS = 5     # Days of intraday data for swing detection
SWING_ORDER = 3             # Bars on each side for swing detection (5-min) — lower = more swings
MIN_SWING_MOVE = 3.0        # Minimum swing significance in points — very low = catch all
MERGE_TOLERANCE = 2.0       # Merge levels within this many points — tighter clustering
TARGET_SPACING = 5.0        # Target spacing between minor levels — Mancini ~5-7 pt gaps
MAX_GAP = 6.0               # Fill gaps larger than this — aggressive fill like Mancini
SUPPORT_RANGE = 150.0       # Points below current — Mancini covers wide range
RESISTANCE_RANGE = 200.0    # Points above current — Mancini covers wide range
TARGET_SUPPORTS = 50        # Max support levels — Mancini avg ~43
TARGET_RESISTANCES = 50     # Max resistance levels — Mancini avg ~42
MAJOR_PCT = 0.18            # Top N% by score are marked major
SHELF_TOLERANCE = 2.0       # Points tolerance for shelf detection
MIN_SHELF_TOUCHES = 8       # Minimum touches to qualify as shelf — lower = more shelves


def find_swing_points(
    bars_5min: pd.DataFrame,
    order: int = SWING_ORDER,
) -> List[Tuple[float, str, pd.Timestamp]]:
    """Find swing highs and lows from 5-minute bars.
    
    Returns list of (price, type, timestamp) tuples.
    type is 'swing_low' or 'swing_high'.
    """
    swings = []
    
    if len(bars_5min) < order * 2 + 1:
        return swings
    
    lows = bars_5min["low"].values
    highs = bars_5min["high"].values
    
    # Find swing lows
    low_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    for idx in low_idx:
        price = lows[idx]
        ts = bars_5min.index[idx]
        swings.append((price, "swing_low", ts))
    
    # Find swing highs
    high_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    for idx in high_idx:
        price = highs[idx]
        ts = bars_5min.index[idx]
        swings.append((price, "swing_high", ts))
    
    return swings


def find_shelves(
    bars_1min: pd.DataFrame,
    tolerance: float = SHELF_TOLERANCE,
    min_touches: int = MIN_SHELF_TOUCHES,
) -> List[Tuple[float, int, str]]:
    """Find horizontal shelves of lows and highs.
    
    A shelf is a price zone that has been touched multiple times.
    Returns list of (price, touch_count, type) tuples.
    """
    shelves = []
    
    # Round lows and highs to nearest integer
    low_prices = np.round(bars_1min["low"].values).astype(int)
    high_prices = np.round(bars_1min["high"].values).astype(int)
    
    # Count touches at each price level (lows)
    from collections import Counter
    low_counts = Counter(low_prices)
    for price, count in low_counts.items():
        if count >= min_touches:
            shelves.append((float(price), count, "shelf_low"))
    
    # Count touches at each price level (highs)
    high_counts = Counter(high_prices)
    for price, count in high_counts.items():
        if count >= min_touches:
            shelves.append((float(price), count, "shelf_high"))
    
    return shelves


def compute_level_score(
    price: float,
    source: str,
    daily_bars: pd.DataFrame,
    touch_count: int = 0,
    age_days: int = 0,
    move_from_low: float = 0.0,
) -> float:
    """Score a level by importance (higher = more significant).
    
    Scoring based on Mancini's hierarchy:
    1. Prior day low/high: +5
    2. Multi-day shelf (3+ touches): +4
    3. Large swing origin (20+ pt rally): +3
    4. Multi-hour swing: +2
    5. Weekly extreme: +2
    6. Round number: +1
    """
    score = 0.0
    
    # Prior day low or high
    if len(daily_bars) > 0:
        prior_day = daily_bars.iloc[-1]
        if abs(price - prior_day["low"]) <= MERGE_TOLERANCE:
            score += 5.0
            source = "prior_day_low"
        elif abs(price - prior_day["high"]) <= MERGE_TOLERANCE:
            score += 5.0
            source = "prior_day_high"
    
    # Multi-day shelf
    if touch_count >= MIN_SHELF_TOUCHES:
        score += min(4.0, touch_count * 0.8)
    
    # Large move origin
    if move_from_low >= 20:
        score += 3.0
    elif move_from_low >= 10:
        score += 1.5
    
    # Swing significance
    if "swing" in source:
        score += 2.0
    
    # Weekly extreme
    if len(daily_bars) >= 5:
        week_low = daily_bars.iloc[-5:]["low"].min()
        week_high = daily_bars.iloc[-5:]["high"].max()
        if abs(price - week_low) <= MERGE_TOLERANCE:
            score += 2.0
        elif abs(price - week_high) <= MERGE_TOLERANCE:
            score += 2.0
    
    # Round number bonus (minor)
    if price % 50 <= 2 or price % 50 >= 48:
        score += 1.0
    elif price % 25 <= 2 or price % 25 >= 23:
        score += 0.5
    
    # Recency bonus (more recent = more important)
    if age_days <= 1:
        score += 1.0
    elif age_days <= 3:
        score += 0.5
    
    return score


def merge_levels(
    levels: List[PriceLevel],
    tolerance: float = MERGE_TOLERANCE,
) -> List[PriceLevel]:
    """Merge levels that are within tolerance of each other.
    
    Keeps the highest-scored level from each cluster.
    """
    if not levels:
        return []
    
    sorted_levels = sorted(levels, key=lambda l: l.price)
    merged = []
    current_cluster = [sorted_levels[0]]
    
    for lv in sorted_levels[1:]:
        if lv.price - current_cluster[0].price <= tolerance:
            current_cluster.append(lv)
        else:
            # Keep best from cluster
            best = max(current_cluster, key=lambda l: l.score)
            # Accumulate touch counts
            best.touch_count = sum(l.touch_count for l in current_cluster)
            merged.append(best)
            current_cluster = [lv]
    
    # Don't forget last cluster
    if current_cluster:
        best = max(current_cluster, key=lambda l: l.score)
        best.touch_count = sum(l.touch_count for l in current_cluster)
        merged.append(best)
    
    return merged


def fill_gaps(
    levels: List[PriceLevel],
    max_gap: float = MAX_GAP,
    target_spacing: float = TARGET_SPACING,
) -> List[PriceLevel]:
    """Fill gaps between levels with minor interpolated levels."""
    if len(levels) < 2:
        return levels
    
    filled = [levels[0]]
    
    for i in range(1, len(levels)):
        gap = abs(levels[i].price - levels[i-1].price)
        if gap > max_gap:
            # Fill with intermediate levels
            n_fill = int(gap / target_spacing) - 1
            if n_fill > 0:
                step = gap / (n_fill + 1)
                for j in range(1, n_fill + 1):
                    fill_price = round(min(levels[i-1].price, levels[i].price) + j * step)
                    filled.append(PriceLevel(
                        price=fill_price,
                        is_major=False,
                        score=0.0,
                        source="fill",
                        description="interpolated",
                    ))
        filled.append(levels[i])
    
    return sorted(filled, key=lambda l: l.price)


def round_level(price: float, direction: str = "nearest") -> float:
    """Round a price to the nearest whole number.
    
    Mancini rounds lows UP and highs DOWN (toward current price).
    """
    if direction == "up":
        return float(np.ceil(price))
    elif direction == "down":
        return float(np.floor(price))
    else:
        return float(round(price))


def generate_levels(
    bars_1min: pd.DataFrame,
    current_price: Optional[float] = None,
    lookback_days: int = LOOKBACK_DAYS,
    swing_lookback_days: int = SWING_LOOKBACK_DAYS,
) -> LevelSet:
    """Generate Mancini-style support/resistance levels.

    Args:
        bars_1min: DataFrame with 1-min OHLCV data, indexed by datetime.
                   Can include overnight/globex bars — swing detection uses
                   RTH only while daily OHLC uses the full session.
        current_price: Current price (defaults to last close)
        lookback_days: Days of daily data to consider
        swing_lookback_days: Days of intraday data for swing detection

    Returns:
        LevelSet with supports, resistances, and bull/bear line
    """
    if current_price is None:
        current_price = bars_1min["close"].iloc[-1]

    # Build daily OHLC
    daily = bars_1min.groupby(bars_1min.index.date).agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    })
    daily.index = pd.to_datetime(daily.index)

    if len(daily) < 2:
        logger.warning("Not enough daily data for level generation")
        return LevelSet(current_price=current_price)

    # Limit to lookback
    daily = daily.iloc[-lookback_days:]
    today = daily.index[-1]

    # ── Step 1: Find intraday swing points (5-min) ──

    swing_start = daily.index[-min(swing_lookback_days, len(daily))]
    swing_ts = pd.Timestamp(swing_start)
    if bars_1min.index.tz is not None and swing_ts.tz is None:
        swing_ts = swing_ts.tz_localize(bars_1min.index.tz)
    intra_mask = bars_1min.index >= swing_ts
    intra_bars = bars_1min[intra_mask]

    # Resample to 5-min
    bars_5min = intra_bars.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()

    swings = find_swing_points(bars_5min)

    # Also find swings on 1-min chart (finer levels)
    swings_1min = find_swing_points(intra_bars, order=10)
    swings.extend(swings_1min)

    # ── Step 2: Find shelves ──

    shelves = find_shelves(intra_bars)
    
    # ── Step 3: Collect all candidate levels ──
    
    candidates = []
    
    # From swings
    for price, stype, ts in swings:
        age = (today - pd.Timestamp(ts.date())).days
        direction = "up" if "low" in stype else "down"
        rounded = round_level(price, direction)
        
        # Calculate move from this swing
        if "low" in stype:
            future_mask = bars_1min.index > ts
            if future_mask.any():
                future_high = bars_1min.loc[future_mask, "high"].max()
                move = future_high - price
            else:
                move = 0
        else:
            move = 0
        
        score = compute_level_score(
            rounded, stype, daily,
            age_days=age, move_from_low=move
        )
        
        candidates.append(PriceLevel(
            price=rounded,
            score=score,
            source=stype,
            age_days=age,
            description=f"{stype} from {ts.strftime('%m/%d %H:%M')}",
        ))
    
    # From shelves
    for price, touches, stype in shelves:
        score = compute_level_score(
            price, stype, daily, touch_count=touches
        )
        candidates.append(PriceLevel(
            price=price,
            score=score,
            source=stype,
            touch_count=touches,
            description=f"{stype} with {touches} touches",
        ))
    
    # From daily extremes
    for i, (dt, row) in enumerate(daily.iterrows()):
        age = (today - dt).days
        
        # Daily low
        low_rounded = round_level(row["low"], "up")
        score = compute_level_score(
            low_rounded, "daily_low", daily, age_days=age
        )
        candidates.append(PriceLevel(
            price=low_rounded,
            score=score,
            source="daily_low",
            age_days=age,
            description=f"daily low {dt.strftime('%m/%d')}",
        ))
        
        # Daily high
        high_rounded = round_level(row["high"], "down")
        score = compute_level_score(
            high_rounded, "daily_high", daily, age_days=age
        )
        candidates.append(PriceLevel(
            price=high_rounded,
            score=score,
            source="daily_high",
            age_days=age,
            description=f"daily high {dt.strftime('%m/%d')}",
        ))
    
    # ── Step 4: Merge nearby levels ──
    
    merged = merge_levels(candidates, tolerance=MERGE_TOLERANCE)
    
    # ── Step 5: Mark majors (top 35% by score) ──
    
    if merged:
        scores = sorted([l.score for l in merged], reverse=True)
        threshold_idx = max(1, int(len(scores) * MAJOR_PCT))
        threshold = scores[min(threshold_idx, len(scores) - 1)]
        
        for lv in merged:
            lv.is_major = lv.score >= threshold and lv.score > 0
    
    # ── Step 6: Split into supports and resistances ──
    
    supports = sorted(
        [l for l in merged if l.price <= current_price],
        key=lambda l: l.price, reverse=True
    )
    resistances = sorted(
        [l for l in merged if l.price > current_price],
        key=lambda l: l.price
    )
    
    # ── Step 7: Trim to range ──
    
    supports = [l for l in supports if current_price - l.price <= SUPPORT_RANGE]
    resistances = [l for l in resistances if l.price - current_price <= RESISTANCE_RANGE]
    
    # Limit count
    supports = supports[:TARGET_SUPPORTS]
    resistances = resistances[:TARGET_RESISTANCES]
    
    # ── Step 8: Fill gaps + ensure full range coverage ──

    supports = fill_gaps(sorted(supports, key=lambda l: l.price))
    resistances = fill_gaps(sorted(resistances, key=lambda l: l.price))

    # Ensure we cover the full range — Mancini always fills ~5-7 pt intervals
    # across his entire support/resistance range
    def ensure_range_coverage(levels, range_start, range_end, spacing=TARGET_SPACING):
        """Fill the full range with minor levels at regular spacing."""
        existing = {round(l.price) for l in levels}
        fills = []
        price = round(range_start)
        while price <= round(range_end):
            # Only add if no existing level within spacing/2
            too_close = any(abs(price - e) < spacing * 0.6 for e in existing)
            if not too_close:
                fills.append(PriceLevel(
                    price=float(price),
                    is_major=False,
                    score=0.0,
                    source="fill",
                    description="range fill",
                ))
                existing.add(price)
            price += int(spacing)
        return fills

    sup_fills = ensure_range_coverage(
        supports,
        current_price - SUPPORT_RANGE,
        current_price - 3,  # don't fill right at current price
    )
    res_fills = ensure_range_coverage(
        resistances,
        current_price + 3,
        current_price + RESISTANCE_RANGE,
    )

    supports = sorted(supports + sup_fills, key=lambda l: l.price)
    resistances = sorted(resistances + res_fills, key=lambda l: l.price)

    # Re-trim to target count (keep highest-scored when trimming)
    if len(supports) > TARGET_SUPPORTS:
        # Keep all structural levels, trim fills first
        structural = [l for l in supports if l.source != "fill"]
        fills = [l for l in supports if l.source == "fill"]
        if len(structural) <= TARGET_SUPPORTS:
            # Evenly sample fills to reach target
            n_fills_needed = TARGET_SUPPORTS - len(structural)
            step = max(1, len(fills) // max(1, n_fills_needed))
            sampled_fills = fills[::step][:n_fills_needed]
            supports = sorted(structural + sampled_fills, key=lambda l: l.price)
        else:
            supports = sorted(structural, key=lambda l: -l.score)[:TARGET_SUPPORTS]
            supports = sorted(supports, key=lambda l: l.price)

    if len(resistances) > TARGET_RESISTANCES:
        structural = [l for l in resistances if l.source != "fill"]
        fills = [l for l in resistances if l.source == "fill"]
        if len(structural) <= TARGET_RESISTANCES:
            n_fills_needed = TARGET_RESISTANCES - len(structural)
            step = max(1, len(fills) // max(1, n_fills_needed))
            sampled_fills = fills[::step][:n_fills_needed]
            resistances = sorted(structural + sampled_fills, key=lambda l: l.price)
        else:
            resistances = sorted(structural, key=lambda l: -l.score)[:TARGET_RESISTANCES]
            resistances = sorted(resistances, key=lambda l: l.price)
    
    # ── Step 9: Identify bull/bear line ──
    
    bull_bear_line = None
    # Bull/bear line is typically the most recent significant
    # daily low or shelf that was broken or defended
    prior_day_low = round_level(daily.iloc[-1]["low"], "up")
    bull_bear_line = prior_day_low
    
    # Check if there is a more significant shelf nearby
    for lv in sorted(supports, key=lambda l: -l.score):
        if lv.source in ("shelf_low", "prior_day_low") and lv.score > 4:
            bull_bear_line = lv.price
            break
    
    result = LevelSet(
        supports=sorted(supports, key=lambda l: l.price, reverse=True),
        resistances=sorted(resistances, key=lambda l: l.price),
        bull_bear_line=bull_bear_line,
        current_price=current_price,
        date=today,
    )
    
    return result


def format_levels(level_set: LevelSet) -> str:
    """Format levels in Mancini newsletter style."""
    lines = []
    lines.append(f"Levels for {level_set.date} (current: {level_set.current_price:.0f})")
    lines.append(f"Bull/Bear Line: {level_set.bull_bear_line:.0f}")
    lines.append("")
    
    # Supports
    sup_parts = []
    for lv in level_set.supports:
        if lv.is_major:
            sup_parts.append(f"{lv.price:.0f} (major)")
        else:
            sup_parts.append(f"{lv.price:.0f}")
    lines.append("Supports are: " + ", ".join(sup_parts))
    lines.append("")
    
    # Resistances
    res_parts = []
    for lv in level_set.resistances:
        if lv.is_major:
            res_parts.append(f"{lv.price:.0f} (major)")
        else:
            res_parts.append(f"{lv.price:.0f}")
    lines.append("Resistances are: " + ", ".join(res_parts))
    
    return chr(10).join(lines)


if __name__ == "__main__":
    # Load data
    data_path = "data/ES_1m_2024-02-05_2026-02-05.parquet"
    df = pd.read_parquet(data_path)
    
    # Generate levels for the last available date
    levels = generate_levels(df)
    
    print(format_levels(levels))
    print()
    print("--- Detail ---")
    print("Supports:")
    for lv in levels.supports:
        print(f"  {lv}")
    print("Resistances:")
    for lv in levels.resistances:
        print(f"  {lv}")
