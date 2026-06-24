"""Floor-trader pivot points — a third confluence source.

Classic pivots derived from the prior session's High/Low/Close. On their own
they're weak (formulaic, not structural), so they carry a low base score and
matter only when one lands near an engine or Mancini level — there it bumps
that level's ``source_count`` (cross-source confluence).

    PP = (H + L + C) / 3
    R1 = 2*PP - L      S1 = 2*PP - H
    R2 = PP + (H - L)  S2 = PP - (H - L)
    R3 = H + 2*(PP-L)  S3 = L - 2*(H - PP)
"""
from __future__ import annotations

from datetime import datetime

from config.levels import Level, LevelType


def compute_pivots(high: float, low: float, close: float) -> dict[str, float]:
    """Return the 7 floor-trader pivot levels from prior H/L/C, tick-rounded."""
    pp = (high + low + close) / 3.0
    rng = high - low
    raw = {
        "pp": pp,
        "r1": 2 * pp - low,
        "s1": 2 * pp - high,
        "r2": pp + rng,
        "s2": pp - rng,
        "r3": high + 2 * (pp - low),
        "s3": low - 2 * (high - pp),
    }
    return {k: round(v * 4) / 4 for k, v in raw.items()}  # round to 0.25 tick


def build_pivot_levels(high: float, low: float, close: float, *,
                       created_at: datetime) -> list[Level]:
    """Build PIVOT Level objects (confirmed immediately) from prior H/L/C."""
    pivots = compute_pivots(high, low, close)
    levels: list[Level] = []
    for name, price in pivots.items():
        if price <= 0:
            continue
        levels.append(Level(
            price=price,
            level_type=LevelType.PIVOT,
            created_at=created_at,
            confirmed_at=created_at,
            label=f"PIVOT:{name.upper()}@{price:.2f}",
        ))
    return levels
