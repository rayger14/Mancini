"""Static economic calendar — the primary forecast source for NewsBlackout.

The reactive 8pt bar and the Mancini-post extraction both guessed at what a
real calendar simply KNOWS: BLS/BEA/Fed publish their entire year's release
schedule in advance (OMB "Principal Federal Economic Indicators CY2026").
`config/econ_calendar_2026.json` holds those dates; `core/econ_calendar`
answers "which events hit this trading date?" as 'HH:MM NAME' strings —
the exact format NewsBlackout already consumes.
"""
import json
from datetime import date

from core.econ_calendar import events_for, load_calendar


def test_cpi_day_found():
    # CPI July 14 2026 — the 55pt bar that cost trade 732
    evs = events_for(date(2026, 7, 14))
    assert any("CPI" in e and e.startswith("08:30") for e in evs)


def test_retail_sales_day_found():
    evs = events_for(date(2026, 7, 16))
    assert any("Retail" in e and e.startswith("08:30") for e in evs)


def test_fomc_day_found():
    evs = events_for(date(2026, 7, 29))
    assert any("FOMC" in e and e.startswith("14:00") for e in evs)


def test_quiet_day_has_no_dated_events():
    # 2026-07-21 (Tue): nothing scheduled in the calendar
    evs = events_for(date(2026, 7, 21))
    assert evs == []


def test_thursday_jobless_claims_rule():
    # Every Thursday 08:30 — weekly rule, not a dated entry
    evs = events_for(date(2026, 7, 23))
    assert any("Claims" in e and e.startswith("08:30") for e in evs)
    # ...and NOT on a Wednesday
    evs_wed = events_for(date(2026, 7, 22))
    assert not any("Claims" in e for e in evs_wed)


def test_nfp_and_claims_can_coexist():
    # 2026-09-04 is NFP Friday; the day before is claims Thursday
    evs = events_for(date(2026, 9, 4))
    assert any("Employment" in e or "NFP" in e for e in evs)


def test_missing_year_returns_empty_not_raises():
    assert events_for(date(2031, 1, 15)) == []
    assert load_calendar(2031) is None


def test_calendar_file_is_valid_and_covers_h2():
    cal = load_calendar(2026)
    assert cal is not None
    assert cal["year"] == 2026
    # every remaining month of 2026 must have at least CPI + NFP + PPI + retail
    for month in range(7, 13):
        month_events = [
            (d, names) for d, names in cal["events"].items()
            if int(d.split("-")[1]) == month
        ]
        joined = " ".join(n for _, names in month_events for n in names)
        for needle in ("CPI", "Employment", "PPI", "Retail"):
            assert needle in joined, f"month {month} missing {needle}"


def test_events_are_wellformed_hhmm_name():
    cal = load_calendar(2026)
    for d, names in cal["events"].items():
        date.fromisoformat(d)  # keys are real dates
        for n in names:
            hh, mm = n.split(" ")[0].split(":")
            assert 0 <= int(hh) <= 23 and 0 <= int(mm) <= 59
            assert len(n.split(" ", 1)[1]) > 0  # has a name
