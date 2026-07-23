"""Static economic calendar — the primary forecast source for NewsBlackout.

BLS/BEA/Fed publish their entire year's release schedule in advance (OMB
"Principal Federal Economic Indicators" CY schedule), so instead of guessing
from bar ranges or hoping Mancini's post mentions the data, the engine simply
KNOWS: `config/econ_calendar_<year>.json` holds the dated releases (CPI, PPI,
NFP, Retail Sales, GDP+PCE, ECI, FOMC statement/minutes, ISM) plus a weekly
rule for Thursday jobless claims.

`events_for(date)` returns 'HH:MM NAME' strings — the exact format
NewsBlackout consumes. A missing calendar file (e.g. the year rolled over
before the new schedule was added) degrades gracefully to []: the
Mancini-post layer and the reactive bar-range layer still cover data days.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

from loguru import logger

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_warned_years: set = set()


def load_calendar(year: int, config_dir: Optional[Path] = None) -> Optional[dict]:
    """Load the calendar JSON for ``year``; None (with a one-time warning)
    if no file exists for that year."""
    path = Path(config_dir or _CONFIG_DIR) / f"econ_calendar_{year}.json"
    if not path.exists():
        if year not in _warned_years:
            _warned_years.add(year)
            logger.warning(
                f"Econ calendar missing for {year} ({path}) — calendar "
                f"forecast layer inactive; refresh from the OMB PFEI schedule"
            )
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Econ calendar unreadable ({path}): {e}")
        return None


def events_for(trading_date: date, config_dir: Optional[Path] = None,
               tier: str = "all") -> list:
    """Scheduled releases hitting ``trading_date``, as 'HH:MM NAME' strings
    (sorted). Dated entries plus weekly rules (weekday: Mon=0).

    ``tier="hard"`` keeps only the events named in the calendar's ``hard``
    list (substring match on the name) — the ones that reliably print a
    violent release bar and warrant a pre-emptive entry block. Everything
    else is advisory: the reactive bar-range layer covers its rare spikes."""
    cal = load_calendar(trading_date.year, config_dir)
    if not cal:
        return []
    out = list(cal.get("events", {}).get(trading_date.isoformat(), []))
    for rule in cal.get("weekly", []):
        try:
            if trading_date.weekday() == int(rule["weekday"]):
                out.append(f"{rule['time']} {rule['name']}")
        except (KeyError, TypeError, ValueError):
            continue
    if tier == "hard":
        hard = cal.get("hard", [])
        out = [e for e in out
               if any(h in e.split(" ", 1)[1] for h in hard)]
    return sorted(out)


# ---------------------------------------------------------------------------
# Quarterly contract-roll guard (June 16-18 2026 discovery: the bot traded
# the June contract at ~7600 while Mancini's plan quoted September prices
# ~60pts lower — his levels were useless for ~3 sessions and both June-17
# trades lost at engine-only levels).
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 15)
    while d.weekday() != 4:
        d = d.replace(day=d.day + 1)
    return d


def in_roll_window(trading_date: date) -> bool:
    """True during quarterly roll week: the Monday before the 3rd-Friday
    expiry (Mancini announces his roll for that Monday 6pm) through the
    expiry Friday, in Mar/Jun/Sep/Dec."""
    if trading_date.month not in (3, 6, 9, 12):
        return False
    from datetime import timedelta
    expiry = _third_friday(trading_date.year, trading_date.month)
    monday = expiry - timedelta(days=4)
    return monday <= trading_date <= expiry


def median_signed_offset(levels, ref_price: float) -> float:
    """Median of (level - ref_price) across the plan's setup levels.
    Normal plans are roughly balanced around price (|median| < ~40);
    a contract mismatch shifts EVERY level the same direction (~55-75pts),
    so the signed median jumps."""
    offs = sorted(float(l) - float(ref_price) for l in levels if l)
    if not offs:
        return 0.0
    n = len(offs)
    mid = n // 2
    return offs[mid] if n % 2 else (offs[mid - 1] + offs[mid]) / 2.0


def plan_usable(median_offset: float, in_roll: bool, threshold: float) -> bool:
    """Should the plan's levels be consumed? False ONLY when the guard is
    on (threshold > 0), we're inside a roll window, and the plan's levels
    sit systematically far from the market — the contract-mismatch
    signature. Outside roll windows a big offset is warn-only (his deep
    ladders can legitimately skew)."""
    if threshold <= 0 or not in_roll:
        return True
    return abs(median_offset) <= threshold
