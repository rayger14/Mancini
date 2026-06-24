"""Support/resistance level data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
from typing import Optional


class LevelType(Enum):
    """Types of significant price levels."""

    PRIOR_DAY_LOW = auto()
    PRIOR_DAY_HIGH = auto()
    MULTI_HOUR_LOW = auto()  # produced a 20+ pt rally
    MULTI_HOUR_HIGH = auto()  # produced a 20+ pt selloff
    CLUSTER_LOW = auto()  # 3+ touches within 1 pt
    CLUSTER_HIGH = auto()  # 3+ touches within 1 pt (resistance)
    SWING_LOW = auto()  # argrelextrema-detected
    SWING_HIGH = auto()  # argrelextrema-detected (resistance)
    HORIZONTAL_SR = auto()  # horizontal S/R with multiple touches
    INTRADAY_LOW = auto()   # fast-confirmed low during deep sell (crash bottom / consolidation)
    VWAP = auto()
    CUSTOM = auto()
    MANCINI_LEVEL = auto()  # level called out in Mancini's Substack (overlay/augmentation)
    PIVOT = auto()  # floor-trader pivot (PP/R1-3/S1-3) — weak alone, matters via confluence


@dataclass
class Level:
    """A single price level with metadata."""

    price: float
    level_type: LevelType
    created_at: datetime
    confirmed_at: Optional[datetime] = None  # when lookahead-safe to use
    touch_count: int = 1
    rally_from_low_pts: float = 0.0  # for MULTI_HOUR_LOW: size of ensuing rally
    is_active: bool = True
    label: str = ""
    origin_date: Optional[date] = None  # date level was first detected (for multi-day aging)
    significance_score: float = 1.0    # starts at 1.0, decays over time
    tested_and_held: bool = False       # True if price tested and bounced (not broke through)
    # Mancini Substack overlay fields (augmentation, not replacement)
    mancini_confirmed: bool = False     # True if this level was called out in Mancini's post
    mancini_side: str = ""              # "support" | "resistance" | "either"
    mancini_conviction: int = 0         # 1-3 from Mancini's context (key/magnet/etc)
    mancini_tags: list = field(default_factory=list)  # tags parsed from context (key, magnet, caution, ...)
    shadow_only: bool = False           # True = do not influence trading (log-only overlay)
    # Cross-source confluence: how many INDEPENDENT sources (engine / Mancini /
    # pivot) agree on this price. 1 = engine-only; bumped when another source
    # lands within tolerance. Drives an LQS confirmation bonus.
    source_count: int = 1

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.level_type.name}@{self.price:.2f}"

    @property
    def is_confirmed(self) -> bool:
        return self.confirmed_at is not None

    def distance_to(self, price: float) -> float:
        """Signed distance: positive means price is above this level."""
        return price - self.price


@dataclass
class LevelStore:
    """Container for managing active price levels during a session."""

    levels: list[Level] = field(default_factory=list)

    def add(self, level: Level) -> None:
        """Add a level, merging with nearby existing levels of the same type."""
        for existing in self.levels:
            if (
                existing.level_type == level.level_type
                and existing.is_active
                and abs(existing.price - level.price) <= 1.0
            ):
                # Merge: increment touch count, keep lower price for lows
                existing.touch_count += level.touch_count
                if "LOW" in level.level_type.name:
                    existing.price = min(existing.price, level.price)
                else:
                    existing.price = max(existing.price, level.price)
                return
        self.levels.append(level)

    def get_active(self, level_type: Optional[LevelType] = None) -> list[Level]:
        """Get all active levels, optionally filtered by type."""
        result = [l for l in self.levels if l.is_active]
        if level_type is not None:
            result = [l for l in result if l.level_type == level_type]
        return sorted(result, key=lambda l: l.price)

    def get_confirmed(
        self, as_of: datetime, level_type: Optional[LevelType] = None
    ) -> list[Level]:
        """Get levels confirmed by the given time (lookahead-safe)."""
        result = [
            l
            for l in self.levels
            if l.is_active and l.is_confirmed and l.confirmed_at <= as_of
        ]
        if level_type is not None:
            result = [l for l in result if l.level_type == level_type]
        return sorted(result, key=lambda l: l.price)

    def supports_below(self, price: float, as_of: datetime) -> list[Level]:
        """Get confirmed support levels below current price."""
        return [l for l in self.get_confirmed(as_of) if l.price < price]

    def resistances_above(self, price: float, as_of: datetime) -> list[Level]:
        """Get confirmed resistance levels above current price."""
        return [l for l in self.get_confirmed(as_of) if l.price > price]

    def nearest_below(self, price: float, as_of: datetime) -> Optional[Level]:
        """Get the nearest confirmed level below price."""
        below = self.supports_below(price, as_of)
        return below[-1] if below else None

    def nearest_above(self, price: float, as_of: datetime) -> Optional[Level]:
        """Get the nearest confirmed level above price."""
        above = self.resistances_above(price, as_of)
        return above[0] if above else None

    def deactivate_below(self, price: float) -> int:
        """Deactivate all levels below a price (e.g., after a breakdown)."""
        count = 0
        for level in self.levels:
            if level.is_active and level.price < price:
                level.is_active = False
                count += 1
        return count

    def inject_levels(self, levels: list[Level]) -> None:
        """Inject external levels into the store (e.g., persistent multi-day levels).

        Uses the standard add() method so nearby same-type levels are merged.
        """
        for level in levels:
            self.add(level)

    def clear(self) -> None:
        """Clear all levels."""
        self.levels.clear()


# Base scores by level type — reflects empirical edge from live data.
# PDL=100% WR (highest conviction), MHL=67%, CLUSTER_LOW=0% (noise).
_LEVEL_BASE_SCORES: dict[LevelType, int] = {
    LevelType.PRIOR_DAY_LOW: 5,
    LevelType.PRIOR_DAY_HIGH: 5,
    LevelType.MULTI_HOUR_LOW: 3,
    LevelType.MULTI_HOUR_HIGH: 3,
    LevelType.SWING_LOW: 2,
    LevelType.SWING_HIGH: 2,
    LevelType.INTRADAY_LOW: 2,
    LevelType.CLUSTER_LOW: 1,
    LevelType.CLUSTER_HIGH: 1,
    LevelType.HORIZONTAL_SR: 1,
    LevelType.VWAP: 1,
    LevelType.CUSTOM: 1,
    LevelType.MANCINI_LEVEL: 3,
    LevelType.PIVOT: 1,  # weak alone — only valuable when it confirms another source
}


def compute_confluence_score(
    level: Level, all_levels: list[Level], proximity: float = 3.0
) -> int:
    """Compute confluence score for a level based on type, nearby levels, and metadata.

    Scoring rules:
    - Base score from level type (PDL=5, MHL=3, SWING=2, CLUSTER=1, etc.)
    - +2 for each additional level of *different* type within ``proximity`` pts
    - +1 if touch_count >= 3 (shelf of lows / multiple tests)
    - +1 if rally_from_low_pts >= 20 (significant bounce proves the level)
    - +1 if level.tested_and_held (previously defended)

    Parameters
    ----------
    level : Level
        The level being scored.
    all_levels : list[Level]
        All active levels in the store (used for proximity check).
    proximity : float
        Points within which another level counts as confluent.

    Returns
    -------
    int
        Confluence score (higher = more conviction).
    """
    score = _LEVEL_BASE_SCORES.get(level.level_type, 1)

    # Confluence: nearby levels of different type reinforce the zone
    seen_types = {level.level_type}
    for other in all_levels:
        if other is level or not other.is_active:
            continue
        if other.level_type in seen_types:
            continue
        if abs(other.price - level.price) <= proximity:
            score += 2
            seen_types.add(other.level_type)

    # Shelf of lows: multiple touches prove the level is real
    if level.touch_count >= 3:
        score += 1

    # Significant bounce: a 20+ pt rally from the level proves demand
    if level.rally_from_low_pts >= 20.0:
        score += 1

    # Previously tested and held: market already proved the level
    if level.tested_and_held:
        score += 1

    # Mancini confirmation bonus: level explicitly called out in Substack post
    if level.mancini_confirmed:
        score += 2

    return score
