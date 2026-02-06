"""Support/resistance level data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional


class LevelType(Enum):
    """Types of significant price levels."""

    PRIOR_DAY_LOW = auto()
    PRIOR_DAY_HIGH = auto()
    MULTI_HOUR_LOW = auto()  # produced a 20+ pt rally
    CLUSTER_LOW = auto()  # 3+ touches within 1 pt
    SWING_LOW = auto()  # argrelextrema-detected
    HORIZONTAL_SR = auto()  # horizontal S/R with multiple touches
    VWAP = auto()
    CUSTOM = auto()


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

    def clear(self) -> None:
        """Clear all levels."""
        self.levels.clear()
