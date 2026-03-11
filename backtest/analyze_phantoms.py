#!/usr/bin/env python3
"""Comprehensive analysis of live bot trades, phantoms, and near-misses.

Reads data/live/trades.jsonl and produces a detailed report covering:
  A. Summary stats
  B. Actual trade performance
  C. Phantom P&L by rejection reason
  D. Sweep depth vs outcome
  E. Time-of-day analysis
  F. Level type performance
  G. Gate bypass analysis
  H. Deep sweep analysis (Mancini thesis)

Usage:
    python3 backtest/analyze_phantoms.py
    python3 backtest/analyze_phantoms.py --data path/to/trades.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime


def load_events(path):
    """Load all events from a JSONL file."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def parse_hour(timestamp_str):
    """Extract hour from various timestamp formats. Returns None on failure."""
    if not timestamp_str:
        return None
    try:
        # Try ISO format first (entry/exit events)
        dt = datetime.fromisoformat(timestamp_str)
        return dt.hour
    except (ValueError, TypeError):
        pass
    try:
        # Try "YYYY-MM-DD HH:MM:SS-05:00" format (near_miss events)
        return int(timestamp_str[11:13])
    except (ValueError, IndexError, TypeError):
        return None


def parse_date(timestamp_str):
    """Extract date string from timestamp."""
    if not timestamp_str:
        return "?"
    return timestamp_str[:10]


def simplify_reject_reason(reason):
    """Simplify rejection reason for grouping."""
    if not reason:
        return "unknown"
    if "Stop too wide" in reason:
        return "Stop too wide"
    if "window:" in reason:
        return reason.split("window:")[-1].strip()
    if "risk:" in reason:
        return reason.split("risk:")[-1].strip()
    return reason


def compute_phantom_data(phantoms):
    """Parse phantom events into structured data with would-be PnL."""
    results = []
    for p in phantoms:
        entry = p.get("entry_price", 0)
        stop = p.get("stop_price", 0)
        target = p.get("target_1", 0)
        signal_type = p.get("signal_type", "")
        reject = p.get("reject_reason", "")
        high = p.get("high_since", 0)
        low = p.get("low_since", 0)
        result_str = p.get("result", "")

        is_short = "SHORT" in signal_type.upper()

        if is_short:
            would_be_pnl = entry - low
            hit_target = low <= target if target else False
            hit_stop = high >= stop if stop else False
        else:
            would_be_pnl = high - entry
            hit_target = high >= target if target else False
            hit_stop = low <= stop if stop else False

        results.append({
            "date": p.get("session_date", "?"),
            "signal_type": signal_type,
            "reject_reason": reject,
            "entry": entry,
            "stop": stop,
            "target": target,
            "high": high,
            "low": low,
            "is_short": is_short,
            "would_be_pnl": would_be_pnl,
            "hit_target": hit_target,
            "hit_stop": hit_stop,
            "result_str": result_str,
            "stop_distance": abs(stop - entry) if stop and entry else 0,
        })
    return results


def print_header(title, char="="):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def section_a_summary(events):
    """A. Summary Stats."""
    print_header("A. SUMMARY STATS")

    by_type = Counter(e["event"] for e in events)
    print(f"\n  Total events: {len(events)}")
    for k, v in by_type.most_common():
        print(f"    {k:25s} {v:>5d}")

    # Date range
    dates = set()
    for e in events:
        d = e.get("session_date")
        if d:
            dates.add(d)
    if dates:
        print(f"\n  Date range: {min(dates)} to {max(dates)} ({len(dates)} unique sessions)")

    # IB bracket fill tracking
    exits = [e for e in events if e["event"] == "exit"]
    bracket_fills = [e for e in exits if e.get("exit_reason") == "IB bracket fill"]
    print(f"\n  IB bracket fills: {len(bracket_fills)} / {len(exits)} exits "
          f"({len(bracket_fills)/len(exits)*100:.0f}%)" if exits else "")


def section_b_trades(events):
    """B. Actual Trade Performance."""
    print_header("B. ACTUAL TRADE PERFORMANCE")

    entries = [e for e in events if e["event"] == "entry"]
    exits = [e for e in events if e["event"] == "exit"]

    real_exits = [e for e in exits
                  if e.get("exit_reason") != "IB bracket fill"
                  and e.get("pattern_type") != "FORCE_TEST"]
    bracket_fills = [e for e in exits if e.get("exit_reason") == "IB bracket fill"]
    force_tests = [e for e in entries if e.get("pattern_type") == "FORCE_TEST"]

    print(f"\n  Entries: {len(entries)}, Exits: {len(exits)}")
    print(f"  Real exits: {len(real_exits)}, IB bracket fills: {len(bracket_fills)}, "
          f"FORCE_TEST: {len(force_tests)}")

    # Real trade P&L
    real_trade_pnl = []
    by_pattern = defaultdict(list)

    print(f"\n  --- Individual Trades ---")
    for e in real_exits:
        pnl = e.get("pnl_pts", 0) or 0
        pattern = e.get("pattern_type", "?")
        direction = e.get("direction", "?")
        date = e.get("session_date", "?")
        level_type = e.get("signal", {}).get("level_type", "?")
        print(f"    {date} {pattern:25s} {direction:5s} "
              f"entry={e.get('entry_price', '?'):>8} exit={e.get('exit_price', '?'):>8} "
              f"pnl={pnl:+7.1f} pts  {e.get('exit_reason', '?'):20s} [{level_type}]")
        real_trade_pnl.append(pnl)
        by_pattern[pattern].append(pnl)

    if real_trade_pnl:
        wins = [p for p in real_trade_pnl if p > 0]
        losses = [p for p in real_trade_pnl if p <= 0]
        print(f"\n  --- Aggregate ---")
        print(f"    Total trades: {len(real_trade_pnl)}")
        print(f"    Wins: {len(wins)}, Losses: {len(losses)}")
        print(f"    Win rate: {len(wins)/len(real_trade_pnl)*100:.1f}%")
        print(f"    Total PnL: {sum(real_trade_pnl):+.1f} pts")
        if wins:
            print(f"    Avg winner: {sum(wins)/len(wins):+.1f} pts")
        if losses:
            print(f"    Avg loser: {sum(losses)/len(losses):+.1f} pts")

        # By pattern type
        print(f"\n  --- By Pattern Type ---")
        for pattern, pnls in sorted(by_pattern.items()):
            w = sum(1 for p in pnls if p > 0)
            print(f"    {pattern:25s} {len(pnls):>3d}T  WR={w/len(pnls)*100:5.1f}%  "
                  f"PnL={sum(pnls):+7.1f} pts")
    else:
        print("\n  No real trades found.")

    # FORCE_TEST trades
    force_exits = [e for e in exits if e.get("pattern_type") == "FORCE_TEST"]
    if force_exits:
        print(f"\n  --- FORCE_TEST Trades ---")
        for e in force_exits:
            pnl = e.get("pnl_pts", 0) or 0
            print(f"    {e.get('session_date','?')} entry={e.get('entry_price','?')} "
                  f"exit={e.get('exit_price','?')} pnl={pnl:+.1f} reason={e.get('exit_reason','?')}")
        total = sum(e.get("pnl_pts", 0) or 0 for e in force_exits)
        print(f"    Total FORCE_TEST PnL: {total:+.1f} pts ({len(force_exits)} trades)")


def section_c_phantoms(events):
    """C. Phantom P&L by Rejection Reason."""
    print_header("C. PHANTOM P&L BY REJECTION REASON")

    phantoms = [e for e in events if e["event"] == "phantom_resolved"]
    print(f"\n  Total phantom signals: {len(phantoms)}")

    if not phantoms:
        return

    phantom_data = compute_phantom_data(phantoms)

    by_reason = defaultdict(list)
    for p in phantom_data:
        reason = simplify_reject_reason(p["reject_reason"])
        by_reason[reason].append(p)

    for reason, trades in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        n = len(trades)
        targets_hit = sum(1 for t in trades if t["hit_target"])
        stops_hit = sum(1 for t in trades if t["hit_stop"])
        total_pnl = sum(t["would_be_pnl"] for t in trades)
        avg_stop = sum(t["stop_distance"] for t in trades) / n
        print(f"\n    {reason}")
        print(f"      Count: {n}, Targets hit: {targets_hit}, Stops hit: {stops_hit}")
        print(f"      Would-be total PnL: {total_pnl:+.1f} pts (avg {total_pnl/n:+.1f})")
        print(f"      Avg stop distance: {avg_stop:.1f} pts")
        # Show up to 3 examples
        for t in trades[:3]:
            print(f"        {t['date']} {t['signal_type']:20s} entry={t['entry']:.1f} "
                  f"stop={t['stop']:.1f} target={t['target']:.1f} => {t['result_str']}")

    # By signal type
    print(f"\n  --- By Signal Type ---")
    by_signal = defaultdict(list)
    for p in phantom_data:
        by_signal[p["signal_type"]].append(p)

    for sig, trades in sorted(by_signal.items()):
        n = len(trades)
        targets_hit = sum(1 for t in trades if t["hit_target"])
        stops_hit = sum(1 for t in trades if t["hit_stop"])
        total_pnl = sum(t["would_be_pnl"] for t in trades)
        print(f"    {sig:25s} {n:>3d}T  targets={targets_hit:>3d}  stops={stops_hit:>3d}  "
              f"would-be PnL={total_pnl:+.1f}")


def section_d_sweep_depth(events):
    """D. Sweep Depth vs Outcome (Mancini's thesis)."""
    print_header("D. SWEEP DEPTH VS OUTCOME")

    near_misses = [e for e in events if e["event"] == "near_miss"]
    fb_near_misses = [nm for nm in near_misses if nm.get("pattern") == "FAILED_BREAKDOWN"]
    print(f"\n  Total near misses: {len(near_misses)}")
    print(f"  FB near misses: {len(fb_near_misses)}")

    # Deduplicate by date + level + sweep_low
    unique_setups = {}
    for nm in fb_near_misses:
        ts = nm.get("timestamp", "")
        date = parse_date(ts)
        level = round(nm.get("level_price", 0), 1)
        sweep = nm.get("sweep_low", 0)
        key = (date, level, sweep)
        close = nm.get("close_at_failure", 0)
        if key not in unique_setups or close > unique_setups[key].get("close_at_failure", 0):
            unique_setups[key] = nm

    print(f"  Unique FB setups (deduped by date+level+sweep): {len(unique_setups)}")

    # Bucket by sweep depth
    depth_buckets = defaultdict(list)
    for key, nm in unique_setups.items():
        level = nm.get("level_price", 0)
        sweep_low = nm.get("sweep_low", 0)
        close = nm.get("close_at_failure", 0)
        sweep_depth = level - sweep_low
        recovery = close - sweep_low
        rr = nm.get("achieved", {}).get("rr_ratio", 0) or 0

        if sweep_depth <= 5:
            bucket = "0-5"
        elif sweep_depth <= 10:
            bucket = "5-10"
        elif sweep_depth <= 20:
            bucket = "10-20"
        elif sweep_depth <= 50:
            bucket = "20-50"
        else:
            bucket = "50+"

        depth_buckets[bucket].append({
            "date": key[0],
            "level": level,
            "sweep_low": sweep_low,
            "close": close,
            "depth": sweep_depth,
            "recovery": recovery,
            "rr": rr,
            "reason": nm.get("failure_reason", "?"),
        })

    for bucket in ["0-5", "5-10", "10-20", "20-50", "50+"]:
        setups = depth_buckets.get(bucket, [])
        if not setups:
            continue
        avg_depth = sum(s["depth"] for s in setups) / len(setups)
        avg_recovery = sum(s["recovery"] for s in setups) / len(setups)
        avg_rr = sum(s["rr"] for s in setups) / len(setups)
        print(f"\n    Sweep Depth {bucket} pts: {len(setups)} setups")
        print(f"      Avg depth: {avg_depth:.1f}, Avg recovery: {avg_recovery:.1f}, "
              f"Avg R:R: {avg_rr:.2f}")
        for s in setups[:5]:
            print(f"        {s['date']} level={s['level']:.1f} sweep={s['sweep_low']:.1f} "
                  f"depth={s['depth']:.1f} recovery={s['recovery']:.1f} "
                  f"rr={s['rr']:.2f} reason={s['reason']}")
        if len(setups) > 5:
            print(f"        ... and {len(setups) - 5} more")


def section_e_time_of_day(events):
    """E. Time-of-Day Analysis."""
    print_header("E. TIME-OF-DAY ANALYSIS")

    entries = [e for e in events if e["event"] == "entry"]
    near_misses = [e for e in events if e["event"] == "near_miss"]
    fb_near_misses = [nm for nm in near_misses if nm.get("pattern") == "FAILED_BREAKDOWN"]

    # Entries by hour
    entry_hours = defaultdict(list)
    for entry in entries:
        hour = parse_hour(entry.get("timestamp", ""))
        if hour is not None:
            entry_hours[hour].append(entry)

    print(f"\n  --- Entries by Hour ---")
    print(f"    {'Hour':>4s}  {'Count':>5s}  {'Bypass':>6s}  Patterns")
    for hour in sorted(entry_hours.keys()):
        ents = entry_hours[hour]
        patterns = Counter(e.get("pattern_type", "?") for e in ents)
        bypassed = sum(1 for e in ents if e.get("gate_bypassed"))
        pat_str = ", ".join(f"{k}={v}" for k, v in patterns.most_common())
        print(f"    {hour:4d}  {len(ents):5d}  {bypassed:6d}  {pat_str}")

    # Near-miss FB by hour
    print(f"\n  --- Near Miss FB by Hour ---")
    nm_hours = defaultdict(int)
    for nm in fb_near_misses:
        hour = parse_hour(nm.get("timestamp", ""))
        if hour is not None:
            nm_hours[hour] += 1

    for hour in sorted(nm_hours.keys()):
        print(f"    Hour {hour:02d}: {nm_hours[hour]} near misses")

    # Gate bypass frequency by hour
    print(f"\n  --- Gate Bypass Frequency by Hour ---")
    bypass_hours = defaultdict(int)
    total_hours = defaultdict(int)
    for entry in entries:
        hour = parse_hour(entry.get("timestamp", ""))
        if hour is not None:
            total_hours[hour] += 1
            if entry.get("gate_bypassed"):
                bypass_hours[hour] += 1

    for hour in sorted(total_hours.keys()):
        bp = bypass_hours.get(hour, 0)
        tot = total_hours[hour]
        pct = bp / tot * 100 if tot else 0
        print(f"    Hour {hour:02d}: {bp}/{tot} bypassed ({pct:.0f}%)")


def section_f_level_types(events):
    """F. Level Type Performance."""
    print_header("F. LEVEL TYPE PERFORMANCE")

    entries = [e for e in events if e["event"] == "entry"]
    exits = [e for e in events if e["event"] == "exit"]
    near_misses = [e for e in events if e["event"] == "near_miss"]

    # Which level types generate entries
    print(f"\n  --- Entry Signal Types ---")
    signal_types = Counter()
    level_types = Counter()
    for entry in entries:
        signal = entry.get("signal", {})
        sig_type = signal.get("type", entry.get("pattern_type", "?"))
        lev_type = signal.get("level_type", "?")
        signal_types[sig_type] += 1
        level_types[lev_type] += 1

    print(f"    Signal types:")
    for st, n in signal_types.most_common():
        print(f"      {st:25s} {n:>3d}")

    print(f"\n    Level types:")
    for lt, n in level_types.most_common():
        print(f"      {lt:25s} {n:>3d}")

    # Level type P&L from exits
    real_exits = [e for e in exits
                  if e.get("exit_reason") != "IB bracket fill"
                  and e.get("pattern_type") != "FORCE_TEST"]

    if real_exits:
        print(f"\n  --- Level Type P&L ---")
        lt_pnl = defaultdict(list)
        for e in real_exits:
            lev_type = e.get("signal", {}).get("level_type", "?")
            pnl = e.get("pnl_pts", 0) or 0
            lt_pnl[lev_type].append(pnl)

        for lt, pnls in sorted(lt_pnl.items(), key=lambda x: -sum(x[1])):
            w = sum(1 for p in pnls if p > 0)
            print(f"      {lt:25s} {len(pnls):>3d}T  WR={w/len(pnls)*100:5.1f}%  "
                  f"PnL={sum(pnls):+7.1f} pts")

    # Near-miss failure reasons by level
    fb_near_misses = [nm for nm in near_misses if nm.get("pattern") == "FAILED_BREAKDOWN"]

    # Deduplicate
    unique_setups = {}
    for nm in fb_near_misses:
        ts = nm.get("timestamp", "")
        date = parse_date(ts)
        level = round(nm.get("level_price", 0), 1)
        sweep = nm.get("sweep_low", 0)
        key = (date, level, sweep)
        close = nm.get("close_at_failure", 0)
        if key not in unique_setups or close > unique_setups[key].get("close_at_failure", 0):
            unique_setups[key] = nm

    print(f"\n  --- Near-Miss FB Failure Reasons (unique setups) ---")
    reason_stats = defaultdict(lambda: {"count": 0, "total_rr": 0, "total_recovery": 0})
    for key, nm in unique_setups.items():
        sweep_low = nm.get("sweep_low", 0)
        close = nm.get("close_at_failure", 0)
        recovery = close - sweep_low if close and sweep_low else 0
        rr = nm.get("achieved", {}).get("rr_ratio", 0) or 0
        reason = nm.get("failure_reason", "?")
        reason_stats[reason]["count"] += 1
        reason_stats[reason]["total_rr"] += rr
        reason_stats[reason]["total_recovery"] += recovery

    for reason, data in sorted(reason_stats.items(), key=lambda x: -x[1]["count"]):
        n = data["count"]
        avg_rr = data["total_rr"] / n if n else 0
        avg_rec = data["total_recovery"] / n if n else 0
        print(f"      {reason:25s} {n:>3d} setups  avg_rr={avg_rr:.2f}  "
              f"avg_recovery={avg_rec:.1f}")


def section_g_gate_bypass(events):
    """G. Gate Bypass Analysis."""
    print_header("G. GATE BYPASS ANALYSIS")

    entries = [e for e in events if e["event"] == "entry"]
    exits = [e for e in events if e["event"] == "exit"]

    # Which gates are most bypassed
    bypass_counts = Counter()
    bypassed_entries = []
    for entry in entries:
        bypassed = entry.get("gate_bypassed", [])
        if bypassed:
            bypassed_entries.append(entry)
            for gate in bypassed:
                bypass_counts[gate] += 1

    print(f"\n  --- Gate Bypass Frequency ---")
    if bypass_counts:
        for gate, n in bypass_counts.most_common():
            print(f"    {gate:45s} {n:>3d} times")
    else:
        print("    No gate bypass data found in entries")

    # P&L of bypassed-gate trades vs normal
    print(f"\n  --- Bypassed-Gate Trades vs Normal ---")

    real_exits = [e for e in exits
                  if e.get("exit_reason") != "IB bracket fill"
                  and e.get("pattern_type") != "FORCE_TEST"]

    bypassed_pnl = []
    normal_pnl = []
    for ex in real_exits:
        pnl = ex.get("pnl_pts", 0) or 0
        if ex.get("gate_bypassed"):
            bypassed_pnl.append(pnl)
        else:
            normal_pnl.append(pnl)

    if bypassed_pnl:
        bw = sum(1 for p in bypassed_pnl if p > 0)
        print(f"    Bypassed: {len(bypassed_pnl)}T  WR={bw/len(bypassed_pnl)*100:.1f}%  "
              f"PnL={sum(bypassed_pnl):+.1f} pts  avg={sum(bypassed_pnl)/len(bypassed_pnl):+.1f}")
    else:
        print("    Bypassed: 0 trades")

    if normal_pnl:
        nw = sum(1 for p in normal_pnl if p > 0)
        print(f"    Normal:   {len(normal_pnl)}T  WR={nw/len(normal_pnl)*100:.1f}%  "
              f"PnL={sum(normal_pnl):+.1f} pts  avg={sum(normal_pnl)/len(normal_pnl):+.1f}")
    else:
        print("    Normal: 0 trades")

    # Individual bypassed-gate trade details
    if bypassed_entries:
        print(f"\n  --- Individual Bypassed-Gate Trades ---")
        for entry in bypassed_entries:
            date = entry.get("session_date", "?")
            pattern = entry.get("pattern_type", "?")
            price = entry.get("entry_price", 0) or entry.get("last_price", 0)
            direction = entry.get("direction", "?")
            bypassed = entry.get("gate_bypassed", [])
            prod = entry.get("production_would_take", "?")
            gates_str = ", ".join(bypassed)
            print(f"    {date} {pattern:20s} {direction:5s} at {price:>8}  "
                  f"prod={prod}  gates=[{gates_str}]")
            # Find matching exit
            for ex in exits:
                if (ex.get("session_date") == date
                        and ex.get("pattern_type") == pattern
                        and ex.get("exit_reason") != "IB bracket fill"):
                    pnl = ex.get("pnl_pts", 0) or 0
                    reason = ex.get("exit_reason", "?")
                    print(f"      => pnl={pnl:+.1f} pts  reason={reason}")
                    break


def section_h_deep_sweeps(events):
    """H. Deep Sweep Analysis (Mancini 'bigger sell = bigger squeeze')."""
    print_header("H. DEEP SWEEP ANALYSIS (sweep > 20 pts)")

    near_misses = [e for e in events if e["event"] == "near_miss"]
    entries = [e for e in events if e["event"] == "entry"]
    fb_near_misses = [nm for nm in near_misses if nm.get("pattern") == "FAILED_BREAKDOWN"]

    # High-range session entries
    print(f"\n  --- High-Range Session Entries (range >= 50 pts) ---")
    high_range_found = False
    for entry in entries:
        sr = entry.get("session_range", 0) or 0
        if sr >= 50:
            high_range_found = True
            date = entry.get("session_date", "?")
            high = entry.get("session_high", 0)
            low = entry.get("session_low", 0)
            pattern = entry.get("pattern_type", "?")
            direction = entry.get("direction", "?")
            price = entry.get("entry_price", 0) or entry.get("last_price", 0)
            level_type = entry.get("signal", {}).get("level_type", "?")
            print(f"    {date} range={sr:.0f} pts (H={high}, L={low})  "
                  f"{pattern} {direction} at {price}  [{level_type}]")
    if not high_range_found:
        print("    None found")

    # Deep sweep near-misses (>20 pts below level)
    print(f"\n  --- Deep Sweep Near-Misses (depth > 20 pts) ---")
    deep_unique = {}
    for nm in fb_near_misses:
        level = nm.get("level_price", 0)
        sweep = nm.get("sweep_low", 0)
        depth = level - sweep
        if depth > 20:
            ts = nm.get("timestamp", "")
            date = parse_date(ts)
            key = (date, round(level, 0))
            close = nm.get("close_at_failure", 0)
            if key not in deep_unique or close > deep_unique[key].get("close_at_failure", 0):
                deep_unique[key] = nm

    print(f"    Unique deep sweeps (>20 pts): {len(deep_unique)}")
    if deep_unique:
        print(f"\n    {'Date':12s} {'Level':>8s} {'SweepLow':>8s} {'Depth':>6s} "
              f"{'Close':>8s} {'Recovery':>8s} {'Reason':20s} {'R:R':>5s}")
        print(f"    {'-'*80}")
        for key, nm in sorted(deep_unique.items()):
            level = nm.get("level_price", 0)
            sweep = nm.get("sweep_low", 0)
            close = nm.get("close_at_failure", 0)
            depth = level - sweep
            recovery = close - sweep
            reason = nm.get("failure_reason", "?")
            rr = nm.get("achieved", {}).get("rr_ratio", "?")
            rr_str = f"{rr:.2f}" if isinstance(rr, (int, float)) else str(rr)
            print(f"    {key[0]:12s} {level:8.1f} {sweep:8.1f} {depth:6.1f} "
                  f"{close:8.1f} {recovery:8.1f} {reason:20s} {rr_str:>5s}")

    # Check if any deep sweep entries were actually taken
    print(f"\n  --- Were Any Deep Sweep Setups Actually Traded? ---")
    deep_dates = set(k[0] for k in deep_unique.keys())
    for entry in entries:
        date = entry.get("session_date", "?")
        if date in deep_dates:
            pattern = entry.get("pattern_type", "?")
            direction = entry.get("direction", "?")
            price = entry.get("entry_price", 0) or entry.get("last_price", 0)
            sr = entry.get("session_range", 0) or 0
            print(f"    {date} {pattern} {direction} at {price} (range={sr:.0f})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze live bot trades, phantoms, and near-misses from trades.jsonl")
    parser.add_argument("--data", default=None,
                        help="Path to trades.jsonl (default: data/live/trades.jsonl)")
    args = parser.parse_args()

    # Resolve data path
    if args.data:
        data_path = args.data
    else:
        # Try relative to script location first, then CWD
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_path = os.path.join(project_root, "data", "live", "trades.jsonl")
        if not os.path.exists(data_path):
            data_path = os.path.join(os.getcwd(), "data", "live", "trades.jsonl")

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        print("Use --data to specify the path to trades.jsonl")
        sys.exit(1)

    print(f"Loading data from: {data_path}")
    events = load_events(data_path)
    print(f"Loaded {len(events)} events")

    section_a_summary(events)
    section_b_trades(events)
    section_c_phantoms(events)
    section_d_sweep_depth(events)
    section_e_time_of_day(events)
    section_f_level_types(events)
    section_g_gate_bypass(events)
    section_h_deep_sweeps(events)

    print(f"\n{'=' * 70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
