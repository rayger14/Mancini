"""Which releases deserve a hard entry pre-block? 5y ES 1-min study.

For each event class, the distribution of the release-minute bar range
(high-low, the minute of the release) vs. the same-minute baseline on
non-event days. Wrong/mismatched date lists self-expose: they regress to
the ~3pt baseline instead of showing a fat tail.
"""
import pandas as pd
from datetime import date, timedelta

PARQ = "data/ES_1m_full_session_2021-01-01_2026-02-05.parquet"
df = pd.read_parquet(PARQ)

# --- timezone detection: naive index; ET vs UTC hypothesis on a KNOWN
# CPI day (2026-01-13, verified). ET -> big 08:30 bar; UTC -> big 13:30.
d = df[df.index.date == date(2026, 1, 13)]
def rng_at(frame, hh, mm):
    b = frame[(frame.index.hour == hh) & (frame.index.minute == mm)]
    return float(b["high"].max() - b["low"].min()) if len(b) else None
print(f"tz probe 2026-01-13 CPI: bar@08:30={rng_at(d,8,30)} bar@13:30={rng_at(d,13,30)}")

# ---- event date lists ----
FOMC = """2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28""".split()

CPI = """2021-01-13 2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10 2021-07-13 2021-08-11 2021-09-14 2021-10-13 2021-11-10 2021-12-10
2022-01-12 2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10 2022-07-13 2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13
2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18
2026-01-13""".split()

PPI = """2021-01-15 2021-02-17 2021-03-12 2021-04-09 2021-05-13 2021-06-15 2021-07-14 2021-08-12 2021-09-10 2021-10-14 2021-11-09 2021-12-14
2022-01-13 2022-02-15 2022-03-15 2022-04-13 2022-05-12 2022-06-14 2022-07-14 2022-08-11 2022-09-14 2022-10-12 2022-11-15 2022-12-09
2023-01-18 2023-02-16 2023-03-15 2023-04-13 2023-05-11 2023-06-14 2023-07-13 2023-08-11 2023-09-14 2023-10-11 2023-11-15 2023-12-13
2024-01-12 2024-02-16 2024-03-14 2024-04-11 2024-05-14 2024-06-13 2024-07-12 2024-08-13 2024-09-12 2024-10-11 2024-11-14 2024-12-12
2025-01-14 2025-02-13 2025-03-13 2025-04-11 2025-05-15 2025-06-12 2025-07-16 2025-08-14 2025-09-10
2026-01-14""".split()

RETAIL = """2021-01-15 2021-02-17 2021-03-16 2021-04-15 2021-05-14 2021-06-15 2021-07-16 2021-08-17 2021-09-16 2021-10-15 2021-11-16 2021-12-15
2022-01-14 2022-02-16 2022-03-16 2022-04-14 2022-05-17 2022-06-15 2022-07-15 2022-08-17 2022-09-15 2022-10-14 2022-11-16 2022-12-15
2023-01-18 2023-02-15 2023-03-15 2023-04-14 2023-05-16 2023-06-15 2023-07-18 2023-08-15 2023-09-14 2023-10-17 2023-11-15 2023-12-14
2024-01-17 2024-02-15 2024-03-14 2024-04-15 2024-05-15 2024-06-18 2024-07-16 2024-08-15 2024-09-17 2024-10-17 2024-11-15 2024-12-17
2025-01-16 2025-02-14 2025-03-17 2025-04-16 2025-05-15 2025-06-17 2025-07-17 2025-09-16
2026-01-15""".split()

all_days = sorted(set(df.index.date))
day_set = set(all_days)

def parse(dl):
    return [date.fromisoformat(s) for s in dl if date.fromisoformat(s) in day_set]

fomc = parse(FOMC)
cpi = parse(CPI)
ppi = parse(PPI)
retail = parse(RETAIL)

# deterministic rules
def first_friday(y, m):
    d0 = date(y, m, 1)
    return d0 + timedelta(days=(4 - d0.weekday()) % 7)
nfp = [d for d in (first_friday(y, m) for y in range(2021, 2027) for m in range(1, 13)) if d in day_set]
minutes = [d for d in (f + timedelta(days=21) for f in parse(FOMC)) if d in day_set]

def nth_bd(y, m, n):
    d0, c = date(y, m, 1), 0
    while True:
        if d0.weekday() < 5:
            c += 1
            if c == n:
                return d0
        d0 += timedelta(days=1)
ism_m = [d for d in (nth_bd(y, m, 1) for y in range(2021, 2027) for m in range(1, 13)) if d in day_set]
ism_s = [d for d in (nth_bd(y, m, 3) for y in range(2021, 2027) for m in range(1, 13)) if d in day_set]

dated_830 = set(cpi) | set(ppi) | set(retail) | set(nfp)
claims = [d for d in all_days if d.weekday() == 3 and d not in dated_830]
# GDP/PCE proxy window: last 5 business days of month, 08:30, minus other events
def in_gdp_window(d):
    eom = (pd.Timestamp(d) + pd.offsets.BMonthEnd(0))
    start = (eom - pd.offsets.BDay(4)).date()
    return start <= d <= eom.date()
gdp_win = [d for d in all_days
           if d.weekday() < 5 and d not in dated_830 and in_gdp_window(d)]

# baseline: same minute, all days not in ANY event list for that minute
groups_830 = {"CPI": cpi, "PPI": ppi, "NFP": nfp, "Retail Sales": retail,
              "Claims-only Thu": claims, "GDP/PCE window(proxy)": gdp_win}
groups_1000 = {"ISM Mfg": ism_m, "ISM Svc": ism_s}
groups_1400 = {"FOMC Statement": fomc, "FOMC Minutes": minutes}

by_date = dict(tuple(df.groupby(df.index.date)))
def minute_range(d, hh, mm):
    fr = by_date.get(d)
    if fr is None:
        return None
    return rng_at(fr, hh, mm)

def report(groups, hh, mm):
    used = set().union(*groups.values())
    base = [minute_range(d, hh, mm) for d in all_days if d not in used]
    base = pd.Series([x for x in base if x is not None])
    print(f"\n--- {hh:02d}:{mm:02d} ET (baseline n={len(base)} med={base.median():.1f} "
          f"p90={base.quantile(.9):.1f} >=8pt {100*(base>=8).mean():.0f}%)")
    print(f"{'event':24s}{'n':>4s}{'med':>7s}{'p90':>7s}{'max':>7s}{'>=8pt':>7s}{'>=12pt':>7s}")
    for name, days in groups.items():
        v = pd.Series([x for x in (minute_range(d, hh, mm) for d in days) if x is not None])
        if not len(v):
            print(f"{name:24s}   0")
            continue
        print(f"{name:24s}{len(v):4d}{v.median():7.1f}{v.quantile(.9):7.1f}"
              f"{v.max():7.1f}{100*(v>=8).mean():6.0f}%{100*(v>=12).mean():6.0f}%")

report(groups_830, 8, 30)
report(groups_1000, 10, 0)
report(groups_1400, 14, 0)
