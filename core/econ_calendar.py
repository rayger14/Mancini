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
