"""Post a focused FB-long levels card for a given trading date to Discord.

Reads the webhook from $WATCHDOG_WEBHOOK in its own environment (never printed),
loads data/mancini_plan_<date>.json, and posts a compact monospace table of the
long Failed-Breakdown levels + the target ladder. A lightweight companion to the
full brief in mancini_llm_summary.py — "just the levels" for at-a-glance use.

  python3 live/post_fb_levels.py [--date YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

_CONV_TAG = {"high": "🟢 HIGH", "medium": "🟡 MED", "low": "⚪ LOW"}
_CONV_PAD = {"high": "HIGH", "medium": "MED ", "low": "LOW "}


def build_content(plan: dict, date: str) -> dict:
    lean = (plan.get("lean") or "n/a").capitalize()
    title = plan.get("post_title") or f"{date} Plan"

    fbs = []
    for s in plan.get("planned_setups", []):
        if (s.get("setup_type") or "").lower() != "failed_breakdown":
            continue
        if (s.get("direction") or "long").lower() != "long":
            continue
        fbs.append(s)
    fbs.sort(key=lambda s: float(s.get("level_price") or 0), reverse=True)

    # One line per level, bold header + full sentence. Regular markdown wraps in
    # Discord so nothing is cut off (a monospace code block would truncate).
    lines = []
    for s in fbs:
        price = float(s.get("level_price") or 0)
        conv = (s.get("conviction") or "low").lower()
        note = (s.get("context") or "").strip()
        lines.append(f"**`{price:.0f}`** {_CONV_TAG.get(conv, '⚪ LOW')} — {note}")

    table = "\n".join(lines)

    targets = plan.get("targets") or []
    tgt_line = ""
    if targets:
        tgt_line = "\n**Trim ladder:** " + " · ".join(f"{t:.0f}" for t in targets)

    events = plan.get("economic_events") or []
    ev_line = ""
    if events:
        ev_line = ("⚠️ **Data:** " + " · ".join(str(e) for e in events)
                   + "  ·  entries auto-pause around these\n")

    desc = (
        f"{ev_line}"
        f"**Lean:** {lean}  ·  Every FB = **flush *below* the level, then reclaim** "
        f"back above (no reclaim = no trade).\n\n"
        f"{table}\n{tgt_line}"
    )

    return {
        "username": "Mancini Brief",
        "embeds": [{
            "title": f"📋 {date} — FB Long Levels",
            "description": desc,
            "color": 0x2ecc71 if (plan.get("lean") or "").lower() == "bullish" else 0x95a5a6,
            "footer": {"text": title},
        }],
    }


def _state_path(data_dir: Path, date: str) -> Path:
    return data_dir / f".fb_levels_posted_{date}.json"


def _already_posted(state: Path, title: str) -> bool:
    """True if we already posted this exact plan (matched by post_title, so a
    corrected/newer post for the same date still re-posts). Mirrors the brief
    poster's content-aware idempotency."""
    if not state.exists():
        return False
    try:
        return (json.loads(state.read_text()).get("post_title") or "") == title
    except (json.JSONDecodeError, OSError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None, help="Trading date YYYY-MM-DD")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Post even if already posted for this date/title.")
    args = ap.parse_args()

    if not args.date:
        print("❌ pass --date YYYY-MM-DD", file=sys.stderr)
        return 2

    pf = args.data_dir / f"mancini_plan_{args.date}.json"
    if not pf.exists():
        print(f"❌ Plan file not found: {pf}", file=sys.stderr)
        return 2
    doc = json.loads(pf.read_text())

    if doc.get("extract_status") == "stale_post":
        # Post for this trading date isn't published yet — nothing to show.
        print(f"ℹ️  {pf.stem}: post not published yet (stale_post) — skipping.")
        return 0

    plan = doc.get("plan", {})
    title = (plan.get("post_title") or "").strip()

    payload = build_content(plan, args.date)

    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    state = _state_path(args.data_dir, args.date)
    if not args.force and _already_posted(state, title):
        print(f"ℹ️  FB card already posted for {args.date} — use --force to repost.")
        return 0

    webhook = os.environ.get("WATCHDOG_WEBHOOK") or ""
    if not webhook:
        print("❌ No $WATCHDOG_WEBHOOK set", file=sys.stderr)
        return 2
    if requests is None:
        print("❌ requests not installed", file=sys.stderr)
        return 2

    r = requests.post(webhook, json=payload, timeout=10)
    if 200 <= r.status_code < 300:
        try:
            state.write_text(json.dumps({"post_title": title}))
        except OSError:
            pass
        print(f"✅ FB levels card posted ({r.status_code})")
        return 0
    print(f"❌ HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
