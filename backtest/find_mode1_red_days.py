"""Build a ground-truth set of Mancini-labeled Mode 1 Red days from his
585-post archive.

Method:
  For every post that mentions a "Mode 1 red day" (or close variants),
  parse the surrounding context to identify WHICH calendar date the
  post is referring to. Output: a JSON list of {date, evidence_quote,
  post_date, post_title} entries.

These dates are the "yes" examples the detector should fire on.
Everything else (in a similar time period) is implicitly "no".

Phrases handled:
  * "yesterday was a Mode 1 red day"           → post_date - 1 business day
  * "today was a Mode 1 red day"               → post_date
  * "Friday was a Mode 1 red day"              → previous Friday
  * "Monday was a Mode 1 red day" (etc.)       → previous DOW
  * "the last Mode 1 red day was November 20th" → explicit date
  * "we had a Mode 1 red day yesterday"        → post_date - 1
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

POSTS = json.loads(
    Path("/Users/rayghandchi/Mancini bot/Mancini/data/substack/all_posts.json").read_text()
)


# Recognise "Mode 1 red" specifically (not Green, not generic Mode 1)
RED_RE = re.compile(r"mode\s*1\s*red", re.I)

# Date-phrase patterns within ±200 chars of a Mode 1 Red mention
PHRASE_YESTERDAY = re.compile(r"\byesterday\b", re.I)
PHRASE_TODAY     = re.compile(r"\btoday\b", re.I)
PHRASE_DOW       = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday)\b", re.I
)
PHRASE_EXPLICIT  = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|"
    r"july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?"
    r")\s+(\d{1,2})(?:st|nd|rd|th)?",
    re.I,
)

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS_ABBR = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
_MONTHS_FULL = {**_MONTHS, **_MONTHS_ABBR}
_DOW = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}


def _prev_business_day(d: date) -> date:
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _previous_dow(reference: date, target_dow: int) -> date:
    """The most recent past occurrence of target_dow (0=Mon) at or before
    reference - 1."""
    offset = (reference.weekday() - target_dow) % 7
    if offset == 0:
        offset = 7
    return reference - timedelta(days=offset)


def _resolve_date_phrase(text_window: str, post_date: date) -> tuple[date | None, str]:
    """Find the first date phrase in the window relative to post_date.
    Returns (resolved_date, phrase_kind) or (None, '')."""
    if PHRASE_YESTERDAY.search(text_window):
        return _prev_business_day(post_date), "yesterday"
    if PHRASE_TODAY.search(text_window):
        return post_date, "today"

    m = PHRASE_EXPLICIT.search(text_window)
    if m:
        mon = _MONTHS_FULL.get(m.group(1).lower())
        day = int(m.group(2))
        if mon:
            # Guess year — same year as post if month <= post.month + 1,
            # else previous year. Simple heuristic.
            year = post_date.year
            try:
                cand = date(year, mon, day)
                if (cand - post_date).days > 60:
                    cand = date(year - 1, mon, day)
            except ValueError:
                cand = None
            if cand is not None:
                return cand, "explicit_date"

    m = PHRASE_DOW.search(text_window)
    if m:
        dow = _DOW.get(m.group(1).lower())
        if dow is not None:
            return _previous_dow(post_date, dow), f"dow:{m.group(1).lower()}"

    return None, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/training/mancini_mode1_red_days.json")
    ap.add_argument("--window", type=int, default=200,
                    help="chars on either side of 'Mode 1 red' to search "
                         "for date phrase")
    args = ap.parse_args()

    found: dict[str, list[dict]] = {}
    skipped = 0

    for p in POSTS:
        text = p.get("text") or ""
        if not text:
            continue
        ds = (p.get("date") or "")[:10]
        if not ds:
            continue
        try:
            post_d = datetime.fromisoformat(ds).date()
        except ValueError:
            continue

        for m in RED_RE.finditer(text):
            start = max(0, m.start() - args.window)
            end = min(len(text), m.end() + args.window)
            window = text[start:end]
            resolved, kind = _resolve_date_phrase(window, post_d)
            if resolved is None:
                skipped += 1
                continue
            evidence = re.sub(r"\s+", " ", window).strip()
            entry = {
                "post_date": post_d.isoformat(),
                "post_title": p.get("title") or "",
                "phrase_kind": kind,
                "evidence_window": evidence,
            }
            found.setdefault(resolved.isoformat(), []).append(entry)

    # De-duplicate by date, keep first 3 evidence entries per date
    output = []
    for d in sorted(found):
        entries = found[d][:3]
        output.append({
            "date": d,
            "evidence_count": len(found[d]),
            "evidence": entries,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")
    print(f"Unique Mode 1 Red days identified: {len(output)}")
    print(f"Mentions skipped (no date phrase nearby): {skipped}")
    print()
    print(f"Date range: {output[0]['date']} → {output[-1]['date']}")
    print()
    print("Per-year counts:")
    by_year: dict[int, int] = {}
    for o in output:
        y = int(o["date"][:4])
        by_year[y] = by_year.get(y, 0) + 1
    for y in sorted(by_year):
        print(f"  {y}: {by_year[y]:>3} Mode 1 Red days "
              f"(Mancini says ~24/year = 1-2/month)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
