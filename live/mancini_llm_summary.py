"""Build and post a human-readable Discord brief from a Mancini plan JSON.

The cron runs the LLM extractor at 17:30 ET, this script runs at 17:35 ET
and posts a single rich embed to the Discord channel so subscribers see
Mancini's read for the next session at a glance.

Usage:
    python3 live/mancini_llm_summary.py [--plan-file PATH] [--webhook URL]
                                        [--dry-run] [--force]

Behavior:
    - Default plan file: data/mancini_plan_<next_trading_date>.json where
      next_trading_date is "tomorrow" in ET (matches the cron convention).
    - Default webhook: $WATCHDOG_WEBHOOK env var.
    - Idempotency: writes data/.mancini_brief_posted_<date> after a
      successful POST and refuses to repost unless --force is given.
      This means the cron can re-fire (e.g., 17:30 and 20:30 backup) and
      subscribers only see the brief once.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


_ET = timezone(timedelta(hours=-5))  # EST baseline; DST handled by stdlib elsewhere

# Discord embed colors (decimal int)
_COLOR_BULL = 0x2ECC71
_COLOR_BEAR = 0xE74C3C
_COLOR_NEUTRAL = 0x95A5A6
_COLOR_ERROR = 0xF39C12


def _trading_date_default(today: date | None = None) -> date:
    """The trading_date the *next* session ends on, matching the cron.

    Skips weekends: a Friday-evening run targets Monday's plan file
    (Mancini's Friday post is the "Monday Plan").
    """
    if today is None:
        today = datetime.now(_ET).date()
    d = today + timedelta(days=1)
    while d.weekday() > 4:  # 5=Sat, 6=Sun
        d += timedelta(days=1)
    return d


def _state_path(plan_file: Path) -> Path:
    """Lockfile so we don't double-post for the same plan."""
    trading_date = plan_file.stem.replace("mancini_plan_", "")
    return plan_file.parent / f".mancini_brief_posted_{trading_date}"


def should_post(state_file: Path, current_post_title: str) -> tuple[bool, str]:
    """Decide whether to (re-)publish the brief.

    Returns (post, reason). Content-aware: compares the recorded
    ``post_title`` in the state file to the current extraction's title.
    A legacy state file (raw ISO timestamp, pre-v2 format) is treated as
    "already posted" to preserve prior behavior across the upgrade.
    """
    if not state_file.exists():
        return True, "no prior brief"
    try:
        recorded = json.loads(state_file.read_text())
        recorded_title = (recorded.get("post_title") or "").strip()
    except (json.JSONDecodeError, OSError):
        # Pre-v2 state file (raw timestamp) — preserve prior idempotency.
        return False, "legacy state file (pre-v2), treating as posted"
    cur = (current_post_title or "").strip()
    if recorded_title == cur:
        return False, "title unchanged"
    return True, f"title changed ({recorded_title!r} -> {cur!r})"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _conviction_badge(c: str) -> str:
    return {"high": "⭐", "medium": "▪", "low": "·"}.get((c or "").lower(), "·")


def _setup_type_label(t: str) -> str:
    return {
        "failed_breakdown": "FB",
        "level_reclaim": "LR",
        "breakdown_short": "BD",
        "trend_continuation": "TC",
        "other": "—",
    }.get((t or "").lower(), (t or "?").upper())


def _format_setups_block(setups: list[dict], conviction_set: set[str]) -> str:
    """Format a subset of setups (by conviction) as a code block."""
    rows: list[str] = []
    for s in setups:
        if (s.get("conviction") or "").lower() not in conviction_set:
            continue
        price = s.get("level_price")
        kind = _setup_type_label(s.get("setup_type") or "")
        dir_arrow = "↑" if (s.get("direction") or "") == "long" else "↓"
        badge = _conviction_badge(s.get("conviction") or "")
        ctx = _truncate((s.get("context") or "").replace("\n", " "), 70)
        rows.append(f"{badge} {price:>7.2f}  {kind} {dir_arrow}  {ctx}")
    if not rows:
        return ""
    return "```\n" + "\n".join(rows) + "\n```"


def _format_danger_zones(zones: list[dict]) -> str:
    parts: list[str] = []
    for z in zones:
        lo = z.get("price_low")
        hi = z.get("price_high")
        if hi and hi != lo:
            zone_str = f"{lo:.2f}–{hi:.2f}"
        else:
            zone_str = f"{lo:.2f}" if lo is not None else "?"
        rule = _truncate(z.get("rule") or "", 200)
        parts.append(f"• **{zone_str}** — {rule}")
    return "\n".join(parts)


def _lean_color(lean: str) -> int:
    lean = (lean or "").lower()
    if lean == "bullish":
        return _COLOR_BULL
    if lean == "bearish":
        return _COLOR_BEAR
    return _COLOR_NEUTRAL


def build_embed(plan_json: dict) -> dict:
    """Render a single Discord embed from a parsed plan JSON dict."""
    status = plan_json.get("extract_status")
    trading_date = plan_json.get("trading_date", "?")
    post_title = (plan_json.get("post_title") or "").strip()

    if status != "ok" or not plan_json.get("plan"):
        return {
            "title": f"⚠️  Mancini Plan extraction issue — {trading_date}",
            "description": f"status=`{status}` error=`{(plan_json.get('error') or '')[:200]}`",
            "color": _COLOR_ERROR,
        }

    p = plan_json["plan"]
    lean = (p.get("lean") or "neutral").lower()
    mode = (p.get("mode") or "—") or "—"

    # Header: trading day + post title
    try:
        d = datetime.strptime(trading_date, "%Y-%m-%d").date()
        weekday = d.strftime("%A")
        header = f"📋  MANCINI'S READ — {weekday} {d.strftime('%b %-d')}"
    except (ValueError, TypeError):
        header = f"📋  MANCINI'S READ — {trading_date}"

    # Description: post title + lean/mode row + thesis
    desc_parts = []
    if post_title:
        desc_parts.append(f"_{_truncate(post_title, 240)}_")
    badge_lean = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(lean, "⚪")
    desc_parts.append(
        f"{badge_lean} **Lean:** {lean.upper()}   •   **Mode:** {mode.upper()}"
    )
    thesis = (p.get("thesis_summary") or "").strip()
    if thesis:
        desc_parts.append(f"\n📊 **THESIS**\n{_truncate(thesis, 800)}")
    description = "\n".join(desc_parts)
    description = _truncate(description, 4000)

    fields: list[dict[str, Any]] = []

    # Key observations
    obs = [o.strip() for o in (p.get("key_observations") or []) if o and o.strip()]
    if obs:
        obs_text = "\n".join(f"• {_truncate(o, 200)}" for o in obs[:6])
        fields.append({
            "name": "👀 Key Observations",
            "value": _truncate(obs_text, 1024),
            "inline": False,
        })

    # Scenarios
    bull = (p.get("bull_case") or "").strip()
    if bull:
        fields.append({
            "name": "🎯 Bullish Case",
            "value": _truncate(bull, 1024),
            "inline": False,
        })
    bear = (p.get("bear_case") or "").strip()
    if bear:
        fields.append({
            "name": "🚨 Bearish Case",
            "value": _truncate(bear, 1024),
            "inline": False,
        })

    # No-trade lines (compact, inline pair)
    nta = p.get("no_trade_above")
    ntb = p.get("no_trade_below")
    if nta is not None:
        fields.append({"name": "🚫 No-Trade Above", "value": f"**{nta}**", "inline": True})
    if ntb is not None:
        fields.append({"name": "🚫 No-Trade Below", "value": f"**{ntb}**", "inline": True})

    # Targets up
    targets = p.get("targets") or []
    if targets:
        tgt_str = " → ".join(f"{t:g}" for t in targets[:6])
        fields.append({"name": "🎯 Targets", "value": tgt_str, "inline": False})

    # Setups — high/medium conviction (what the bot will care about)
    setups = p.get("planned_setups") or []
    hi_med_block = _format_setups_block(setups, {"high", "medium"})
    if hi_med_block:
        fields.append({
            "name": "⭐ High / Medium Conviction Setups",
            "value": _truncate(hi_med_block, 1024),
            "inline": False,
        })
    lo_block = _format_setups_block(setups, {"low"})
    if lo_block:
        fields.append({
            "name": "· Lower Conviction (FYI)",
            "value": _truncate(lo_block, 1024),
            "inline": False,
        })

    # Danger zones
    zones = p.get("danger_zones") or []
    if zones:
        fields.append({
            "name": "⚠️ Danger Zones",
            "value": _truncate(_format_danger_zones(zones), 1024),
            "inline": False,
        })

    # Risk warnings (Mancini's "don't do this")
    warnings = [w.strip() for w in (p.get("risk_warnings") or []) if w and w.strip()]
    if warnings:
        warn_text = "\n".join(f"• {_truncate(w, 180)}" for w in warnings[:6])
        fields.append({
            "name": "⛔ Don't Do This",
            "value": _truncate(warn_text, 1024),
            "inline": False,
        })

    # Footer with extraction metadata (small, low-signal)
    md = p.get("raw_extraction_metadata") or {}
    footer_bits = [f"model: {md.get('model', '?')}"]
    if md.get("latency_ms"):
        footer_bits.append(f"latency: {md['latency_ms']/1000:.1f}s")
    footer_bits.append(f"trading_date: {trading_date}")

    return {
        "title": header,
        "description": description,
        "color": _lean_color(lean),
        "fields": fields[:25],  # Discord cap
        "footer": {"text": " • ".join(footer_bits)},
    }


def build_payload(plan_json: dict) -> dict:
    return {
        "username": "Mancini Brief",
        "embeds": [build_embed(plan_json)],
    }


def post_to_discord(payload: dict, webhook: str, timeout: float = 8.0) -> tuple[bool, str]:
    if requests is None:
        return False, "requests library not installed"
    try:
        r = requests.post(webhook, json=payload, timeout=timeout)
        if 200 <= r.status_code < 300:
            return True, f"HTTP {r.status_code}"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"exception: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-file", type=Path, default=None,
                    help="Path to mancini_plan_<date>.json. Default: next trading day's file.")
    ap.add_argument("--webhook", default=None,
                    help="Discord webhook URL. Default: $WATCHDOG_WEBHOOK env var.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the payload JSON instead of posting.")
    ap.add_argument("--force", action="store_true",
                    help="Post even if a brief was already posted for this date.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="Directory containing mancini_plan_*.json files.")
    args = ap.parse_args()

    # Resolve plan file
    plan_file: Path
    if args.plan_file:
        plan_file = args.plan_file
    else:
        plan_file = args.data_dir / f"mancini_plan_{_trading_date_default()}.json"

    if not plan_file.exists():
        print(f"❌ Plan file not found: {plan_file}", file=sys.stderr)
        return 2

    try:
        plan_json = json.loads(plan_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"❌ Could not parse plan JSON {plan_file}: {e}", file=sys.stderr)
        return 2

    if plan_json.get("extract_status") == "stale_post":
        # Expected pre-publication state: the cron fired before Mancini's
        # post for this trading date went up. A later run will retry —
        # nothing to announce.
        print(f"ℹ️  {plan_file.stem}: post not published yet (stale_post) — "
              "skipping brief.")
        return 0

    # Idempotency check — content-aware so a backup cron that catches a
    # NEW post (after the primary grabbed yesterday's stale one) still
    # re-posts.
    state = _state_path(plan_file)
    current_title = (plan_json.get("post_title") or "").strip()
    if not args.force and not args.dry_run:
        post, reason = should_post(state, current_title)
        if not post:
            print(f"ℹ️  Brief already posted for {plan_file.stem} "
                  f"({reason}). Use --force to repost.")
            return 0
        if state.exists():
            print(f"ℹ️  Re-posting brief for {plan_file.stem} ({reason})")

    payload = build_payload(plan_json)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    webhook = args.webhook or os.environ.get("WATCHDOG_WEBHOOK") or ""
    if not webhook:
        print("❌ No webhook (pass --webhook or set $WATCHDOG_WEBHOOK)", file=sys.stderr)
        return 2

    ok, info = post_to_discord(payload, webhook)
    if ok:
        try:
            state.write_text(json.dumps({
                "post_title": current_title,
                "posted_at": datetime.now().isoformat(),
            }))
        except OSError:
            pass
        print(f"✅ Brief posted to Discord ({info})")
        return 0
    print(f"❌ Discord post failed: {info}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
