"""Rich Discord embeds for live trade events.

Replaces the watchdog's plain ``TRADE ENTRY — 4 MES`` notification with
context-rich embeds that include:

  * Mancini plan match (level, conviction, context quote)
  * Engine level type + confirmation protocol + sweep depth
  * The three-stage exit schedule (T1 / T2 / runner) per Mancini
  * Per-contract risk in points and dollars
  * R:R and time of day
  * For exits: realized fill, remaining contracts, stop migration,
    cumulative trade P&L

The bot posts these DIRECTLY to the webhook (no log-tail parsing) so
nothing is lost between fill and notification.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


_ET = timezone(timedelta(hours=-4))

_COLOR_LONG = 0x2ECC71
_COLOR_SHORT = 0xE74C3C
_COLOR_T1 = 0x2ECC71
_COLOR_T2 = 0x27AE60
_COLOR_STOP = 0xE74C3C
_COLOR_TRAIL = 0xF39C12
_COLOR_EOD = 0x95A5A6
_COLOR_RUNNER = 0xF39C12


def _conviction_badge(conv: str) -> str:
    return {
        "high": "⭐ HIGH conviction",
        "medium": "▪ MEDIUM conviction",
        "low": "· low conviction",
    }.get((conv or "").lower(), "")


def _find_plan_match(plan: Any, level_price: float,
                     tolerance_pts: float = 2.0) -> dict | None:
    """Return the closest matching Mancini plan setup, or None."""
    if plan is None or not getattr(plan, "planned_setups", None):
        return None
    best = None
    best_d = tolerance_pts + 1
    for s in plan.planned_setups:
        try:
            d = abs(float(s.level_price) - float(level_price))
        except (TypeError, ValueError):
            continue
        if d <= tolerance_pts and d < best_d:
            best = s
            best_d = d
    return best


def plan_short_match(plan: Any, price: float, tol: float = 8.0) -> Any:
    """Return the nearest Mancini planned SHORT setup within ``tol`` of price.

    Used to gate live short heads-ups: only alert when the engine's shadow
    short lines up with a level Mancini actually called as a short (e.g. his
    7399 / 7530 breakdowns), so alerts feel real instead of firing on every
    mechanical flush. Longs are ignored. Returns the setup object or None.
    """
    if plan is None or not getattr(plan, "planned_setups", None):
        return None
    best, best_d = None, tol + 1
    for s in plan.planned_setups:
        if (getattr(s, "direction", "") or "").lower() != "short":
            continue
        try:
            d = abs(float(s.level_price) - float(price))
        except (TypeError, ValueError):
            continue
        if d <= tol and d < best_d:
            best, best_d = s, d
    return best


def build_entry_embed(*,
                      position,
                      signal,
                      fill_price: float,
                      contracts_ordered: int,
                      contract_spec,
                      exit_params,
                      plan,
                      session_date,
                      entry_time: datetime | None = None,
                      trade_id=None) -> dict:
    """Build a Discord embed payload describing a fresh trade entry.

    Plain function so it's trivially unit-testable without the bot.
    """
    direction = (getattr(position, "direction", None)
                 or getattr(signal, "direction", "long")).lower()
    is_long = direction == "long"
    color = _COLOR_LONG if is_long else _COLOR_SHORT
    side_icon = "🟢" if is_long else "🔴"
    side_word = "LONG" if is_long else "SHORT"
    sig_name = signal.signal_type.name
    sig_label = sig_name.replace("_", " ").title()
    point_value = float(getattr(contract_spec, "point_value", 5.0))
    symbol = getattr(contract_spec, "symbol", "MES")

    pat = getattr(signal, "pattern", None)
    lvl = getattr(pat, "level", None) if pat else None
    level_price = float(getattr(lvl, "price", 0.0)) if lvl else 0.0
    level_type = getattr(getattr(lvl, "level_type", None), "name", "?") if lvl else "?"
    conf = getattr(pat, "confirmation", None) if pat else None
    conf_name = conf.name.lower() if hasattr(conf, "name") else (str(conf).lower() if conf else "?")
    sweep = float(getattr(pat, "sweep_depth_pts", 0.0)) if pat else 0.0

    # Mancini plan match
    plan_match = _find_plan_match(plan, level_price)
    plan_match_str = ""
    if plan_match is not None:
        ctx = (getattr(plan_match, "context", "") or "")[:100]
        conv = (getattr(plan_match, "conviction", "") or "")[:8]
        plan_match_str = (
            f"_Mancini plan: **{plan_match.level_price:.0f}** "
            f"{plan_match.setup_type} {plan_match.direction} ({conv}) "
            f"— \"{ctx}\"_"
        )

    risk_pts = abs(fill_price - getattr(position, "stop_price", fill_price))
    risk_per_ct_dollars = risk_pts * point_value
    target_1 = float(getattr(position, "target_1", 0.0))
    target_2 = float(getattr(position, "target_2", 0.0))
    reward_t1_pts = abs(target_1 - fill_price) if target_1 else 0.0
    reward_t2_pts = abs(target_2 - fill_price) if target_2 else 0.0
    rr = float(getattr(signal, "rr_ratio_t1", 0.0))

    t1_fr = float(getattr(exit_params, "t1_exit_fraction", 0.75))
    t2_fr = float(getattr(exit_params, "t2_exit_fraction", 0.15))
    runner_fr = float(getattr(exit_params, "runner_fraction", 0.10))
    import math as _math
    t1_qty = _math.floor(contracts_ordered * t1_fr) if contracts_ordered > 1 else contracts_ordered
    t1_qty = max(t1_qty, 1 if contracts_ordered >= 1 else 0)
    t2_qty = _math.floor(contracts_ordered * t2_fr)
    runner_qty = max(0, contracts_ordered - t1_qty - t2_qty)

    title_extra = f"  •  {_conviction_badge((plan_match.conviction if plan_match else '') or '')}".rstrip(" •")
    title = (f"{side_icon} {sig_label.upper()} {side_word} "
             f"@ {fill_price:.2f}  •  {contracts_ordered} {symbol}{title_extra}")

    description_parts: list[str] = []
    if plan_match_str:
        description_parts.append(plan_match_str)
    description_parts.append(
        f"Pattern: **{sig_name}**  •  Level: **{level_type}** @ {level_price:.2f}"
    )
    description_parts.append(
        f"Confirmation: **{conf_name}** protocol  •  "
        f"Sweep depth: {sweep:.1f} pts"
    )
    description = "\n".join(description_parts)

    fields: list[dict[str, Any]] = []
    fields.append({
        "name": "⛔ Stop",
        "value": (f"${getattr(position, 'stop_price', 0.0):.2f}\n"
                  f"({risk_pts:.1f} pt / ${risk_per_ct_dollars:.0f} per ct)"),
        "inline": True,
    })
    if target_1:
        sign = "+" if is_long else "-"
        fields.append({
            "name": f"🎯 T1 ({int(t1_fr*100)}%)",
            "value": (f"${target_1:.2f}\n"
                      f"{sign}{reward_t1_pts:.1f} pt × {t1_qty} = "
                      f"${reward_t1_pts*t1_qty*point_value:.0f}"),
            "inline": True,
        })
    if target_2 and t2_qty > 0:
        sign = "+" if is_long else "-"
        fields.append({
            "name": f"🎯 T2 ({int(t2_fr*100)}%)",
            "value": (f"${target_2:.2f}\n"
                      f"{sign}{reward_t2_pts:.1f} pt × {t2_qty} = "
                      f"${reward_t2_pts*t2_qty*point_value:.0f}"),
            "inline": True,
        })
    if runner_qty > 0:
        fields.append({
            "name": f"🏃 Runner ({int(runner_fr*100)}%)",
            "value": f"structure trail\n{runner_qty} ct, multi-session",
            "inline": True,
        })

    et = (entry_time or datetime.now(_ET)).astimezone(_ET)
    rr_str = f"{rr:.1f}:1" if rr else "—"
    fields.append({
        "name": "📊 R:R",
        "value": rr_str,
        "inline": True,
    })
    fields.append({
        "name": "⏰ Entry",
        "value": et.strftime("%I:%M %p ET"),
        "inline": True,
    })
    fields.append({
        "name": "📋 Plan",
        "value": str(session_date),
        "inline": True,
    })

    return {
        "username": "Mancini Bot",
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "fields": fields[:25],
            "footer": {"text": (f"Mancini Bot • {symbol}"
                                + (f" • trade #{trade_id}" if trade_id else ""))},
        }],
    }


def build_exit_embed(*,
                     phase: str,                # "t1", "t2", "stop", "runner_trail", "eod"
                     fill_price: float,
                     contracts_closed: int,
                     entry_price: float,
                     direction: str,
                     contract_spec,
                     remaining_contracts: int,
                     realized_pnl_pts_so_far: float,
                     new_stop: float | None = None,
                     next_target: float | None = None,
                     reason: str = "",
                     fill_time: datetime | None = None,
                     trade_id=None) -> dict:
    """Build a Discord embed for an exit event (T1 / T2 / stop / runner)."""
    is_long = (direction or "long").lower() == "long"
    point_value = float(getattr(contract_spec, "point_value", 5.0))
    symbol = getattr(contract_spec, "symbol", "MES")

    # Per-contract gain so the embed reads "+12.5 pt × 3 = $187" instead
    # of the less-intuitive "+37.5 pt × 3".
    if is_long:
        per_contract_pts = (fill_price - entry_price)
    else:
        per_contract_pts = (entry_price - fill_price)
    slice_pnl_pts = per_contract_pts * contracts_closed
    slice_pnl_dollars = slice_pnl_pts * point_value

    phase_meta = {
        "t1": ("🎯 T1 FILLED", _COLOR_T1, "✅"),
        "t2": ("🎯 T2 FILLED", _COLOR_T2, "✅"),
        "stop": ("🛑 STOP HIT", _COLOR_STOP, "❌"),
        "runner_trail": ("📏 RUNNER STOPPED", _COLOR_TRAIL, "✅"),
        "eod": ("🕐 EOD FLATTEN", _COLOR_EOD, "🕐"),
    }.get(phase, (f"📤 EXIT — {phase}", _COLOR_EOD, ""))
    title_label, color, status_icon = phase_meta

    title = (f"{title_label}  •  {contracts_closed} of "
             f"{contracts_closed + remaining_contracts} closed @ {fill_price:.2f}")

    sign = "+" if per_contract_pts >= 0 else ""
    desc_parts = [
        f"{status_icon} Locked {sign}{per_contract_pts:.1f} pt × "
        f"{contracts_closed} = ${slice_pnl_dollars:+.0f}",
    ]
    if remaining_contracts > 0:
        desc_parts.append(
            f"🟡 {remaining_contracts} contract(s) remaining"
        )
        if new_stop is not None and new_stop > 0:
            desc_parts.append(
                f"📍 Stop moved to **{new_stop:.2f}**"
            )
        if next_target is not None and next_target > 0:
            sign2 = "+" if is_long else "-"
            dist = abs(next_target - fill_price)
            desc_parts.append(
                f"🎯 Next target: **{next_target:.2f}** "
                f"({sign2}{dist:.1f} pt away)"
            )
    else:
        desc_parts.append("🏁 Position fully closed")

    cum_sign = "+" if realized_pnl_pts_so_far >= 0 else ""
    cum_dollars = realized_pnl_pts_so_far * point_value
    ts = (fill_time or datetime.now(_ET)).astimezone(_ET)
    desc_parts.append(
        f"\nTrade P&L so far: **{cum_sign}{realized_pnl_pts_so_far:.1f} pt** "
        f"(${cum_dollars:+,.0f})   •   {ts.strftime('%I:%M %p ET')}"
    )

    description = "\n".join(desc_parts)

    return {
        "username": "Mancini Bot",
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "footer": {
                "text": (f"Mancini Bot • {symbol}"
                         + (f" • trade #{trade_id}" if trade_id else "")
                         + (f" • {reason[:60]}" if reason else "")),
            },
        }],
    }


# ---------------------------------------------------------------------------
# Short heads-up alerts
#
# The engine runs Mancini's short detectors live but in shadow mode
# (``shadow_mode_features=True``): it finds breakdown / velocity / backtest
# shorts and logs them as ``shadow_events`` but never places an order (the bot
# is long-only). These helpers turn an actionable shadow-short event into a
# Discord heads-up so the user can take it manually — it is NEVER a bot order.
# ---------------------------------------------------------------------------

# Map detector signal_type → human label for the alert.
_SHORT_LABELS = {
    "BREAKDOWN_SHORT": "Breakdown short (BD)",
    "VELOCITY_SHORT": "Velocity breakdown short",
    "BACKTEST_SHORT": "Backtest short (failed reclaim)",
    "DEEP_SELL_SHORT": "Deep-sell continuation short",
}


def is_short_alert_event(event: Any) -> bool:
    """True only for a GENUINE short trigger worth alerting.

    Requires ``feature == "short_triggered"`` — the event the aggregator emits
    only when a short survives EVERY guard (the Mancini failed-bounce sequence
    completed: price lost the level, bounced, and failed beneath it). This
    deliberately excludes the rejection logs that also carry a short bracket —
    ``capitulation_entry`` (faded the flush), ``move_exhaustion``,
    ``block_pdl_shorts``, ``daily_structure_short_suppression``, ``lqs_shadow``
    — which are shorts the engine THREW AWAY, not setups to act on.
    """
    if not isinstance(event, dict):
        return False
    if event.get("event") == "shadow_outcome":   # a resolved result, not a new trigger
        return False
    if event.get("feature") != "short_triggered":
        return False
    return ((event.get("direction") or "").lower() == "short"
            and event.get("entry_price") is not None
            and event.get("stop_price") is not None)


def short_alert_key(event: dict) -> str:
    """Stable dedup key so one setup alerts once, not every bar it re-fires.

    Buckets the entry to the nearest point — consecutive bars nudge the entry
    by sub-point amounts (same setup); a move to a different level is distinct.
    """
    sig = event.get("signal_type") or "SHORT"
    try:
        bucket = round(float(event["entry_price"]))
    except (KeyError, TypeError, ValueError):
        bucket = event.get("entry_price")
    return f"{sig}:{bucket}"


def build_short_alert_embed(event: dict, symbol: str = "MES",
                            plan: Any = None) -> dict:
    """Build a red Discord embed announcing a (heads-up only) short setup."""
    entry = float(event["entry_price"])
    stop = float(event["stop_price"])
    target = event.get("target_1")
    sig = event.get("signal_type") or "BREAKDOWN_SHORT"
    label = _SHORT_LABELS.get(sig, sig.replace("_", " ").title())
    risk = abs(stop - entry)

    lines = [
        f"**Type:** {label}",
        f"**Short near:** {entry:.2f}   **Stop:** {stop:.2f}  _({risk:.1f} pt risk)_",
    ]
    if target is not None:
        try:
            tgt = float(target)
            reward = abs(entry - tgt)
            rr = (reward / risk) if risk else 0.0
            lines.append(f"**First target:** {tgt:.2f}  _({reward:.1f} pt, {rr:.1f}R)_")
        except (TypeError, ValueError):
            pass

    # Quote Mancini's plan context if this lines up with a published setup.
    lvl = event.get("level_price")
    match = _find_plan_match(plan, float(lvl if lvl is not None else entry))
    if match is not None:
        ctx = (getattr(match, "context", "") or "")[:140]
        conv = (getattr(match, "conviction", "") or "")
        if ctx:
            lines.append(f"\n_Mancini ({conv}): “{ctx}”_")

    lines.append(
        "\n⚠️ _Heads-up only — the bot is long-only and will **not** "
        "place this order._")

    return {
        "title": f"\U0001f534 SHORT SETUP — {symbol} {entry:.0f}",
        "description": "\n".join(lines),
        "color": _COLOR_SHORT,
        "footer": {"text": f"shadow short • {event.get('feature', '')}"},
    }


def post_payload(payload: dict, webhook_url: str,
                 timeout: float = 5.0) -> tuple[bool, str]:
    """POST a Discord webhook payload. Returns (ok, info)."""
    if requests is None:
        return False, "requests not installed"
    if not webhook_url:
        return False, "no webhook url"
    try:
        r = requests.post(webhook_url, json=payload, timeout=timeout)
        return (200 <= r.status_code < 300), f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def get_webhook_url() -> str:
    """Resolve the trade-notification webhook from env. Reuses the
    watchdog webhook by default."""
    return (os.environ.get("MANCINI_TRADE_WEBHOOK")
            or os.environ.get("WATCHDOG_WEBHOOK")
            or "")
