"""Engine-confidence profile: evidence-based P(win) per trade profile.

Three datasets agree the deployed sizing signals were inverted or
non-predictive (Mancini's rating badge; stop-distance tiers de-sizing
deep-flush winners). This module owns the replacement's data layer: a
committed profile table mapping (confirmation x plan_match x
session_window) cells to historical n / win-rate / avg P&L, with
hierarchical backoff + m-estimate shrinkage.

CRITICAL INVARIANT: harvest-side (`key_from_record`, reading logged trade
dicts) and live-side (`key_from_signal_context`, reading live objects)
must derive identical keys for the same trade — the shared normalization
functions below are the single source of truth, and
tests/test_confidence_profile.py asserts parity.

The table is optional infrastructure: a missing/corrupt artifact loads as
a null table whose predictions are None, and every consumer no-ops.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Optional

from loguru import logger

_PLAN_LEVEL_TYPES = {"CUSTOM", "MANCINI_LEVEL", "MANCINI_PLAN"}
# Same rule as live/ib_runner._collection_mode_is_quality_setup path 2.
_PLAN_MATCH_TOLERANCE_PTS = 2.0
# m-estimate shrinkage weight: cells are pulled toward their parent by the
# equivalent of m pseudo-observations.
_SHRINKAGE_M = 10.0


@dataclass(frozen=True)
class ProfileKey:
    confirmation: str    # "non_acceptance" | "acceptance" | "other"
    plan_match: str      # "plan" | "engine"
    session_window: str  # "morning" | "day" | "premarket" | "evening" | "overnight"


@dataclass
class Prediction:
    p_win: Optional[float]
    avg_pts: Optional[float]
    n: int
    cell_label: str
    backoff_level: int   # 0=finest cell .. 3=global; -1 = null table


def confirmation_class(conf_name) -> str:
    c = (str(conf_name) if conf_name else "").upper()
    if c == "NON_ACCEPTANCE":
        return "non_acceptance"
    if c == "ACCEPTANCE":
        return "acceptance"
    return "other"


def session_window_class(t: time) -> str:
    """Five bins, a coarsening of IBRunner._get_session_window's ten:
    morning 09:30-11 | day 11-17 | premarket 06-09:30 |
    evening 17-22 | overnight 22-06."""
    if time(9, 30) <= t < time(11, 0):
        return "morning"
    if time(11, 0) <= t < time(17, 0):
        return "day"
    if time(6, 0) <= t < time(9, 30):
        return "premarket"
    if time(17, 0) <= t < time(22, 0):
        return "evening"
    return "overnight"


def window_class_from_detail(detail: str) -> str:
    """Map IBRunner._get_session_window detail strings (as logged on trade
    records) to the same five bins as session_window_class."""
    d = (detail or "").lower()
    if "morning" in d:
        return "morning"
    if "midday" in d or "chop" in d or "afternoon" in d or "eod" in d:
        return "day"
    if "pre-market" in d:
        return "premarket"
    if "evening" in d or "daily break" in d:
        return "evening"
    return "overnight"   # Late Night, European Open, unknown


def plan_match_class(level_type_name, level_price, plan) -> str:
    """"plan" iff the level is plan-injected OR sits within tolerance of a
    planned long setup — mirrors the collection-mode quality rule."""
    if (level_type_name or "").upper() in _PLAN_LEVEL_TYPES:
        return "plan"
    try:
        lp = float(level_price)
    except (TypeError, ValueError):
        return "engine"
    for st in (getattr(plan, "planned_setups", None) or []):
        try:
            if (str(getattr(st, "direction", "")).lower() == "long"
                    and abs(float(st.level_price) - lp)
                    <= _PLAN_MATCH_TOLERANCE_PTS):
                return "plan"
        except (TypeError, ValueError):
            continue
    return "engine"


def key_from_record(record: dict, plan=None) -> ProfileKey:
    """Harvest side: derive the key from a logged entry record.
    level_type/price live in the nested "signal" sub-dict on live records
    (the top-level fields are null) — read both."""
    sig = record.get("signal") or {}
    lt = record.get("level_type") or sig.get("level_type")
    lp = record.get("level_price") or sig.get("level_price")
    return ProfileKey(
        confirmation=confirmation_class(record.get("confirmation_type")),
        plan_match=plan_match_class(lt, lp, plan),
        session_window=window_class_from_detail(
            str(record.get("session_window") or "")),
    )


def key_from_signal_context(pattern, plan, et_time: time) -> ProfileKey:
    """Live side: derive the key from the pattern/plan at signal time."""
    conf = getattr(pattern, "confirmation", None)
    conf_name = getattr(conf, "name", conf)
    level = getattr(pattern, "level", None)
    lt = getattr(getattr(level, "level_type", None), "name", None)
    lp = getattr(level, "price", None)
    return ProfileKey(
        confirmation=confirmation_class(conf_name),
        plan_match=plan_match_class(lt, lp, plan),
        session_window=session_window_class(et_time),
    )


def _cell_id(*parts) -> str:
    return "|".join(parts)


class ConfidenceTable:
    """Cells keyed 'conf|plan|window' with parents 'conf|plan' and 'conf';
    lookup backs off finest->global, blending each level toward its parent
    with m-estimate shrinkage so tiny cells cannot scream."""

    def __init__(self, cells: dict, global_stats: dict, meta: dict | None = None):
        self.cells = cells or {}
        self.global_stats = global_stats or {}
        self.meta = meta or {}

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path) -> "ConfidenceTable":
        try:
            data = json.loads(Path(path).read_text())
            cells = dict(data.get("cells") or {})
            cells.update(data.get("parents") or {})
            return cls(cells=cells,
                       global_stats=data.get("global") or {},
                       meta=data.get("meta") or {})
        except Exception as e:
            logger.warning(f"Confidence table unavailable ({path}): {e} — "
                           f"predictions disabled (null table)")
            return _NullTable()

    # -- lookup ------------------------------------------------------------
    def _stats_for(self, cid: str):
        c = self.cells.get(cid)
        if c and int(c.get("n", 0)) > 0:
            return c
        return None

    def lookup(self, key: ProfileKey) -> Prediction:
        chain = [
            (_cell_id(key.confirmation, key.plan_match, key.session_window), 0),
            (_cell_id(key.confirmation, key.plan_match), 1),
            (key.confirmation, 2),
        ]
        g = self.global_stats
        g_n = int(g.get("n", 0) or 0)
        if g_n <= 0:
            return Prediction(None, None, 0, "no-data", -1)
        g_p = float(g.get("wins", 0)) / g_n
        g_avg = float(g.get("avg_pnl", 0.0))

        # walk from coarse to fine, shrinking each level toward its parent
        p_parent, avg_parent = g_p, g_avg
        found = None
        for cid, lvl in reversed(chain):
            c = self._stats_for(cid)
            if c is None:
                continue
            n = int(c["n"])
            p = (float(c["wins"]) + _SHRINKAGE_M * p_parent) / (n + _SHRINKAGE_M)
            avg = ((float(c.get("avg_pnl", 0.0)) * n + _SHRINKAGE_M * avg_parent)
                   / (n + _SHRINKAGE_M))
            p_parent, avg_parent = p, avg
            found = Prediction(round(p, 4), round(avg, 2), n, cid, lvl)
        if found is not None:
            return found
        return Prediction(round(g_p, 4), round(g_avg, 2), g_n, "global", 3)

    # -- sizing map (Phase B) ---------------------------------------------
    @staticmethod
    def size_factor(p_win, params):
        """Confidence tier -> size factor. None passes through so callers
        fall back to the legacy sizing path untouched."""
        if p_win is None:
            return None
        full = float(getattr(params, "confidence_full_size_pwin", 0.60))
        half = float(getattr(params, "confidence_half_size_pwin", 0.50))
        if p_win >= full:
            return 1.0
        if p_win >= half:
            return 0.5
        return 0.25


class _NullTable(ConfidenceTable):
    def __init__(self):
        super().__init__(cells={}, global_stats={}, meta={"null": True})

    def lookup(self, key: ProfileKey) -> Prediction:
        return Prediction(None, None, 0, "null", -1)
