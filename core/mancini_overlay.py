"""Apply Mancini Substack levels to the engine's LevelStore as an overlay.

Three modes:
  - "shadow":       inject Mancini levels as MANCINI_LEVEL(shadow_only=True).
                    Trading decisions are unaffected; used to collect data.
  - "confirmation": only boost engine levels that already exist within
                    tolerance. No new levels injected. Mancini levels with no
                    engine coverage are reported as blind_spots.
  - "augmentation": confirm overlapping engine levels AND inject fresh
                    MANCINI_LEVEL entries for Mancini-only calls.

Never raises — the overlay is a quality-of-life layer that must fail safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from config.levels import Level, LevelStore, LevelType


@dataclass
class MancinitOverlayResult:
    """Summary of what the overlay did for a given session."""

    mode: str = ""
    parse_status: str = "missing"
    confirmed_count: int = 0        # engine levels boosted by a nearby Mancini call
    injected_count: int = 0         # new MANCINI_LEVEL levels added (augmentation)
    shadow_count: int = 0           # new MANCINI_LEVEL levels added with shadow_only=True
    blind_spots: list = field(default_factory=list)  # Mancini levels with no engine coverage
    lean: str = "neutral"
    levels_applied: list = field(default_factory=list)


def apply_mancini_overlay(
    store: LevelStore,
    mancini_data: dict | None,
    mode: str = "shadow",
    confirm_tolerance_pts: float = 3.0,
    current_price: float = 0.0,
    timestamp: Optional[datetime] = None,
) -> MancinitOverlayResult:
    """Apply Mancini levels to ``store`` according to ``mode``.

    Safe — never raises. Returns a summary describing what was done.
    """
    result = MancinitOverlayResult(mode=mode)
    timestamp = timestamp or datetime.now()

    try:
        if not mancini_data or mancini_data.get("parse_status") == "failed":
            result.parse_status = "missing"
            return result

        result.parse_status = mancini_data.get("parse_status", "ok")
        result.lean = mancini_data.get("lean", "neutral")

        mancini_levels = mancini_data.get("levels") or []
        if not mancini_levels:
            return result

        # Snapshot engine levels up-front so confirmations match against the
        # original set rather than levels we inject during this pass.
        engine_levels = list(store.levels)

        for m_level in mancini_levels:
            m_price = m_level.get("price", 0)
            try:
                m_price = float(m_price)
            except (TypeError, ValueError):
                continue
            if m_price <= 0:
                continue

            # Find nearest engine level (by absolute distance)
            nearest: Optional[Level] = None
            min_dist = float("inf")
            for e_lv in engine_levels:
                dist = abs(e_lv.price - m_price)
                if dist < min_dist:
                    min_dist = dist
                    nearest = e_lv

            if nearest is not None and min_dist <= confirm_tolerance_pts:
                # CONFIRMATION: boost the engine level
                nearest.mancini_confirmed = True
                nearest.mancini_side = m_level.get("side", "") or ""
                nearest.mancini_conviction = int(m_level.get("conviction", 1) or 1)
                nearest.mancini_tags = list(m_level.get("tags", []) or [])
                result.confirmed_count += 1
                result.levels_applied.append({
                    "action": "confirmed",
                    "engine_price": nearest.price,
                    "mancini_price": m_price,
                    "type": nearest.level_type.name,
                })
                continue

            if mode in ("augmentation", "shadow"):
                # INJECT a new MANCINI_LEVEL
                new_level = Level(
                    price=m_price,
                    level_type=LevelType.MANCINI_LEVEL,
                    created_at=timestamp,
                    confirmed_at=timestamp,
                    touch_count=1,
                    mancini_confirmed=True,
                    mancini_side=m_level.get("side", "") or "",
                    mancini_conviction=int(m_level.get("conviction", 1) or 1),
                    mancini_tags=list(m_level.get("tags", []) or []),
                    shadow_only=(mode == "shadow"),
                )
                store.add(new_level)
                if mode == "shadow":
                    result.shadow_count += 1
                    action = "injected_shadow"
                else:
                    result.injected_count += 1
                    action = "injected"
                result.levels_applied.append({
                    "action": action,
                    "mancini_price": m_price,
                    "side": m_level.get("side"),
                    "conviction": m_level.get("conviction"),
                })
            else:
                # CONFIRMATION mode: this is a blind spot (no engine coverage)
                distance_pts = None
                if current_price > 0:
                    distance_pts = round(abs(m_price - current_price), 2)
                result.blind_spots.append({
                    "price": m_price,
                    "side": m_level.get("side"),
                    "distance_pts": distance_pts,
                    "conviction": m_level.get("conviction"),
                })

        logger.info(
            f"Mancini overlay ({mode}): confirmed={result.confirmed_count} "
            f"injected={result.injected_count} shadow={result.shadow_count} "
            f"blind_spots={len(result.blind_spots)} lean={result.lean}"
        )
        return result
    except Exception as e:
        logger.error(f"Mancini overlay failed: {e}")
        result.parse_status = "failed"
        return result
