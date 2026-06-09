"""Build a labeled training dataset from live trade logs.

Sources combined:
  1. entry+exit pairs in trades.jsonl (real live outcomes)
  2. phantom_resolved events in trades.jsonl (rejected signals with sim outcome)
  3. near_miss_resolved events in trades.jsonl
  4. shadow_trades.jsonl (signals fired in shadow mode)

Output: a tidy DataFrame with one row per setup and a binary `label` column
(1 = T1 hit before stop, 0 = stop hit).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


_RESULT_TARGET_RE = re.compile(r"^T1 HIT", re.IGNORECASE)
_RESULT_STOP_RE = re.compile(r"^STOP HIT", re.IGNORECASE)


def _parse_result_label(result: str | None) -> int | None:
    """Map a 'T1 HIT (...)' / 'STOP HIT (...)' string to 0/1.

    None for ambiguous / unresolved.
    """
    if not result:
        return None
    if _RESULT_TARGET_RE.match(result):
        return 1
    if _RESULT_STOP_RE.match(result):
        return 0
    return None


def _to_dt(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _flatten_entry(entry: dict, exit_event: dict | None) -> dict:
    """Convert an entry event (optionally paired with exit) into a flat row."""
    sig = entry.get("signal") or {}
    regime = entry.get("regime") or {}
    nearby = entry.get("nearby_levels") or []

    ts = _to_dt(entry.get("timestamp"))
    ts_hour = ts.hour if ts else None
    ts_minute = ts.minute if ts else None
    ts_dow = ts.weekday() if ts else None

    session_high = entry.get("session_high")
    session_low = entry.get("session_low")
    last = entry.get("last_price")
    sess_range = entry.get("session_range")
    pos_in_range = None
    if session_high and session_low and last and (session_high - session_low) > 0:
        pos_in_range = (last - session_low) / (session_high - session_low)

    # Resolve label: prefer exit pnl_pts when available, else phantom-style
    label = None
    pnl_pts = None
    if exit_event is not None:
        pnl_pts = exit_event.get("pnl_pts")
        if pnl_pts is not None:
            # T1 width approximation
            t1 = sig.get("target_1")
            entry_px = sig.get("entry") or entry.get("entry_price")
            stop = sig.get("stop") or entry.get("stop_price")
            if t1 and entry_px and stop:
                t1_width = abs(t1 - entry_px)
                stop_width = abs(entry_px - stop)
                # Win if pnl reached at least 80% of T1 distance
                if pnl_pts >= 0.8 * t1_width:
                    label = 1
                elif pnl_pts <= -0.8 * stop_width:
                    label = 0
                else:
                    label = 1 if pnl_pts > 0 else 0
            else:
                label = 1 if pnl_pts > 0 else 0

    nearest_level = nearby[0] if nearby else {}

    return {
        "source": "live_entry",
        "session_date": entry.get("session_date"),
        "timestamp": entry.get("timestamp"),
        "ts_hour": ts_hour,
        "ts_minute": ts_minute,
        "ts_dow": ts_dow,
        "session_window": entry.get("session_window"),
        "bar_count": entry.get("bar_count"),
        "session_high": session_high,
        "session_low": session_low,
        "session_range": sess_range,
        "last_price": last,
        "pos_in_range": pos_in_range,
        "direction": entry.get("direction"),
        "pattern_type": entry.get("pattern_type"),
        "production_would_take": entry.get("production_would_take"),
        "signal_type": sig.get("type"),
        "entry_price": sig.get("entry") or entry.get("entry_price"),
        "stop_price": sig.get("stop") or entry.get("stop_price"),
        "target_1": sig.get("target_1"),
        "rr_ratio": sig.get("rr_ratio"),
        "level_price": sig.get("level_price"),
        "level_type": sig.get("level_type"),
        "regime_direction": regime.get("direction"),
        "longs_enabled": regime.get("longs_enabled"),
        "shorts_enabled": regime.get("shorts_enabled"),
        "ema_slope": regime.get("ema_slope"),
        "nearby_count": len(nearby),
        "nearest_level_distance": nearest_level.get("distance"),
        "nearest_level_type": nearest_level.get("type"),
        "nearest_level_touches": nearest_level.get("touches"),
        "pnl_pts": pnl_pts,
        "label": label,
    }


def _flatten_phantom(p: dict) -> dict:
    """Phantom_resolved / near_miss_resolved events have a slimmer schema."""
    label = _parse_result_label(p.get("result"))
    ts = _to_dt(p.get("timestamp"))
    return {
        "source": p.get("event", "phantom"),
        "session_date": p.get("session_date"),
        "timestamp": p.get("timestamp"),
        "ts_hour": ts.hour if ts else None,
        "ts_minute": ts.minute if ts else None,
        "ts_dow": ts.weekday() if ts else None,
        "signal_type": p.get("signal_type"),
        "entry_price": p.get("entry_price"),
        "stop_price": p.get("stop_price"),
        "target_1": p.get("target_1"),
        "high_since": p.get("high_since"),
        "low_since": p.get("low_since"),
        "reject_reason": p.get("reject_reason"),
        "label": label,
    }


def build_dataset(trades_path: Path, shadow_path: Path | None = None) -> pd.DataFrame:
    """Build the unified training DataFrame.

    Joins live entry events with the next matching exit event (same
    session_date, same direction). Adds phantom_resolved and
    near_miss_resolved as standalone rows.

    Returns
    -------
    DataFrame with one row per labeled setup. Rows missing a label are
    dropped (caller can override).
    """
    entries: list[dict] = []
    exits: list[dict] = []
    phantoms: list[dict] = []

    for r in _read_jsonl(trades_path):
        ev = r.get("event")
        if ev == "entry":
            entries.append(r)
        elif ev == "exit":
            exits.append(r)
        elif ev in ("phantom_resolved", "near_miss_resolved"):
            phantoms.append(r)

    # Pair entries with their exits (greedy, by session_date + first available
    # exit after the entry timestamp with matching direction)
    rows: list[dict] = []
    used_exits: set[int] = set()
    for entry in entries:
        ts = entry.get("timestamp", "")
        sess = entry.get("session_date")
        direction = entry.get("direction")
        match = None
        for i, ex in enumerate(exits):
            if i in used_exits:
                continue
            if ex.get("session_date") != sess:
                continue
            if direction and ex.get("direction") and ex["direction"] != direction:
                continue
            if ex.get("timestamp", "") < ts:
                continue
            match = (i, ex)
            break
        if match is not None:
            used_exits.add(match[0])
        rows.append(_flatten_entry(entry, match[1] if match else None))

    for p in phantoms:
        rows.append(_flatten_phantom(p))

    if shadow_path is not None:
        for s in _read_jsonl(shadow_path):
            label = _parse_result_label(s.get("result")) or _parse_result_label(s.get("outcome"))
            if label is None:
                continue
            ts = _to_dt(s.get("timestamp"))
            rows.append({
                "source": "shadow",
                "session_date": s.get("session_date"),
                "timestamp": s.get("timestamp"),
                "ts_hour": ts.hour if ts else None,
                "ts_minute": ts.minute if ts else None,
                "ts_dow": ts.weekday() if ts else None,
                "signal_type": s.get("signal_type") or s.get("type"),
                "entry_price": s.get("entry_price") or s.get("entry"),
                "stop_price": s.get("stop_price") or s.get("stop"),
                "target_1": s.get("target_1"),
                "label": label,
            })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="data/training/trades.jsonl")
    ap.add_argument("--shadow", default="data/training/shadow_trades.jsonl")
    ap.add_argument("--out", default="data/training/dataset.parquet")
    args = ap.parse_args()

    df = build_dataset(Path(args.trades), Path(args.shadow))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)

    print(f"Wrote {len(df)} rows -> {args.out}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"Source breakdown:   {df['source'].value_counts().to_dict()}")
    print(f"Date range:         {df['session_date'].min()} .. {df['session_date'].max()}")


if __name__ == "__main__":
    main()
