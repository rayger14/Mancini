"""Daily regime filter for enabling/disabling long vs short signals.

Three approaches:
  1. EMA Slope + ATR: 50-EMA slope normalized by ATR for direction,
     ATR percentile for volatility regime.
  2. Market Structure: Daily swing highs/lows to detect HH/HL (bull)
     vs LH/LL (bear) structure.
  3. Composite: Both must agree to disable a direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


class Direction(Enum):
    BULL = auto()
    BEAR = auto()
    NEUTRAL = auto()


class VolRegime(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()


@dataclass(frozen=True)
class RegimeState:
    """Daily regime classification."""
    direction: Direction
    vol_regime: VolRegime
    longs_enabled: bool
    shorts_enabled: bool
    ema_slope: float = 0.0
    vol_percentile: float = 0.5
    structure: str = "neutral"


@dataclass(frozen=True)
class RegimeParams:
    """Tunable parameters for regime detection."""
    # EMA slope
    ema_span: int = 50
    slope_lookback: int = 5
    slope_threshold_atr_mult: float = 0.15

    # ATR percentile
    atr_period: int = 14
    vol_percentile_window: int = 126
    vol_high_threshold: float = 0.75
    vol_low_threshold: float = 0.25

    # Market structure
    swing_order: int = 10
    min_swing_pts: float = 15.0

    # Filter mode
    mode: str = "ema"  # "ema", "structure", "composite", "composite_strict"


def compute_ema_regime(
    daily_closes: np.ndarray,
    daily_highs: np.ndarray,
    daily_lows: np.ndarray,
    params: RegimeParams,
) -> tuple[Direction, VolRegime, float, float]:
    """Compute direction from EMA slope and vol from ATR percentile.

    All inputs use PRIOR data only (no lookahead).
    """
    n = len(daily_closes)
    if n < max(params.ema_span + params.slope_lookback, params.vol_percentile_window):
        return Direction.NEUTRAL, VolRegime.NORMAL, 0.0, 0.5

    # EMA
    closes = pd.Series(daily_closes)
    ema = closes.ewm(span=params.ema_span, adjust=False).mean().values

    # Slope: change over slope_lookback days
    slope = ema[-1] - ema[-1 - params.slope_lookback]

    # ATR
    tr = np.maximum(
        daily_highs[1:] - daily_lows[1:],
        np.maximum(
            np.abs(daily_highs[1:] - daily_closes[:-1]),
            np.abs(daily_lows[1:] - daily_closes[:-1]),
        ),
    )
    # Prepend NaN for alignment
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(params.atr_period).mean().values

    current_atr = atr[-1]
    if np.isnan(current_atr) or current_atr <= 0:
        return Direction.NEUTRAL, VolRegime.NORMAL, slope, 0.5

    # Slope threshold adaptive to ATR
    threshold = current_atr * params.slope_threshold_atr_mult

    if slope > threshold:
        direction = Direction.BULL
    elif slope < -threshold:
        direction = Direction.BEAR
    else:
        direction = Direction.NEUTRAL

    # Vol percentile
    natr = atr / daily_closes * 100
    window_start = max(0, n - params.vol_percentile_window)
    natr_window = natr[window_start:]
    valid = natr_window[~np.isnan(natr_window)]
    if len(valid) < 10:
        vol_pct = 0.5
    else:
        vol_pct = float(np.sum(valid < natr[-1]) / len(valid))

    if vol_pct > params.vol_high_threshold:
        vol = VolRegime.HIGH
    elif vol_pct < params.vol_low_threshold:
        vol = VolRegime.LOW
    else:
        vol = VolRegime.NORMAL

    return direction, vol, float(slope), float(vol_pct)


def compute_structure_regime(
    daily_highs: np.ndarray,
    daily_lows: np.ndarray,
    params: RegimeParams,
) -> tuple[Direction, str]:
    """Detect market structure from swing highs/lows.

    Only uses confirmed swings (swing_order bars after the extreme).
    """
    n = len(daily_highs)
    if n < params.swing_order * 3:
        return Direction.NEUTRAL, "neutral"

    # Find swing highs and lows
    high_idx = argrelextrema(daily_highs, np.greater_equal, order=params.swing_order)[0]
    low_idx = argrelextrema(daily_lows, np.less_equal, order=params.swing_order)[0]

    # Only keep confirmed swings (at least swing_order bars after)
    max_confirmed = n - params.swing_order
    high_idx = high_idx[high_idx <= max_confirmed]
    low_idx = low_idx[low_idx <= max_confirmed]

    if len(high_idx) < 2 or len(low_idx) < 2:
        return Direction.NEUTRAL, "neutral"

    # Last 2 swing highs and lows
    sh1, sh2 = daily_highs[high_idx[-2]], daily_highs[high_idx[-1]]
    sl1, sl2 = daily_lows[low_idx[-2]], daily_lows[low_idx[-1]]

    # Filter by minimum swing size
    high_swing_size = abs(sh2 - sh1)
    low_swing_size = abs(sl2 - sl1)

    hh = sh2 > sh1 + params.min_swing_pts  # Higher high (with threshold)
    hl = sl2 > sl1 + params.min_swing_pts  # Higher low
    lh = sh2 < sh1 - params.min_swing_pts  # Lower high
    ll = sl2 < sl1 - params.min_swing_pts  # Lower low

    if hh and hl:
        return Direction.BULL, "HH/HL"
    elif lh and ll:
        return Direction.BEAR, "LH/LL"
    elif hh and ll:
        return Direction.NEUTRAL, "expanding"
    elif lh and hl:
        return Direction.NEUTRAL, "contracting"
    else:
        return Direction.NEUTRAL, "neutral"


def compute_regime(
    daily_df: pd.DataFrame,
    params: RegimeParams = RegimeParams(),
) -> RegimeState:
    """Compute regime state from daily OHLC data.

    daily_df must have 'open', 'high', 'low', 'close' columns
    and cover sufficient history (at least 126 days).
    """
    closes = daily_df["close"].values
    highs = daily_df["high"].values
    lows = daily_df["low"].values

    # EMA + ATR regime
    ema_dir, vol, slope, vol_pct = compute_ema_regime(closes, highs, lows, params)

    # Market structure regime
    struct_dir, structure = compute_structure_regime(highs, lows, params)

    # Determine enablement based on mode
    mode = params.mode

    if mode == "ema":
        direction = ema_dir
        longs_enabled = direction != Direction.BEAR
        shorts_enabled = direction != Direction.BULL

    elif mode == "structure":
        direction = struct_dir
        longs_enabled = direction != Direction.BEAR
        shorts_enabled = direction != Direction.BULL

    elif mode == "composite":
        # Conservative: disable only if BOTH agree
        longs_enabled = not (ema_dir == Direction.BEAR and struct_dir == Direction.BEAR)
        shorts_enabled = not (ema_dir == Direction.BULL and struct_dir == Direction.BULL)
        # Combined direction
        if ema_dir == struct_dir:
            direction = ema_dir
        else:
            direction = Direction.NEUTRAL

    elif mode == "composite_strict":
        # Strict: disable if EITHER says so
        longs_enabled = ema_dir != Direction.BEAR and struct_dir != Direction.BEAR
        shorts_enabled = ema_dir != Direction.BULL and struct_dir != Direction.BULL
        if ema_dir == struct_dir:
            direction = ema_dir
        else:
            direction = Direction.NEUTRAL

    else:
        direction = Direction.NEUTRAL
        longs_enabled = True
        shorts_enabled = True

    return RegimeState(
        direction=direction,
        vol_regime=vol,
        longs_enabled=longs_enabled,
        shorts_enabled=shorts_enabled,
        ema_slope=slope,
        vol_percentile=vol_pct,
        structure=structure,
    )


def build_daily_bars(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute bars into daily OHLC for regime computation.

    Uses RTH bars only (9:30-16:00) to avoid overnight noise.
    """
    rth = minute_df.between_time("09:30", "15:59")
    daily = rth.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()
    return daily
