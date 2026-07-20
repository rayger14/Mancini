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
_COLOR_COLLECTION = 0x607D8B  # muted slate — collection-mode (non-production) fills


def _conviction_badge(conv: str) -> str:
    return {
        "high": "⭐ HIGH conviction",
        "medium": "▪ MEDIUM conviction",
        "low": "· low conviction",
    }.get((conv or "").lower(), "")


# Level types Mancini injects from his Substack plan; everything else is the
# engine's own price-action detection. Mirrors config.levels.level_source but
# keyed by the type NAME so the embed builders stay enum-free (and trivially
# testable with string stubs).
_MANCINI_LEVEL_TYPES = {"CUSTOM", "MANCINI_LEVEL", "MANCINI_PLAN"}

# Human-readable mechanism for each engine-detected level type, so a reader
# sees "intraday flush low" instead of the raw enum name.
_ENGINE_LEVEL_DESC = {
    "INTRADAY_LOW": "intraday flush low (deep sell → V-recovery)",
    "INTRADAY_HIGH": "intraday spike high",
    "MULTI_HOUR_LOW": "multi-hour reaction low",
    "MULTI_HOUR_HIGH": "multi-hour reaction high",
    "PRIOR_DAY_LOW": "prior-day low",
    "PRIOR_DAY_HIGH": "prior-day high",
    "OVERNIGHT_LOW": "overnight low",
    "OVERNIGHT_HIGH": "overnight high",
    "SWING_LOW": "swing low",
    "SWING_HIGH": "swing high",
    "HORIZONTAL_SR": "horizontal support/resistance",
    "SESSION_OPEN": "session open",
    "VWAP": "VWAP",
    "PIVOT": "pivot",
}


def _fb_logic_line(sweep_pts: float, conf_name: str) -> str:
    """Plain-language description of WHAT KIND of failed breakdown fired, keyed
    off the actual sweep depth (the reliable signature — the detector's
    fb_entry_path field is broken, tagging every live FB 'elevator_fb' even on
    30pt+ sweeps). A 0-sweep momentum entry must never read like a deep flush.

    Live edge read (n=41): 0-sweep momentum ~65% WR but small; deep flush
    (>15pt) ~80% WR and by far the biggest winners.
    """
    sw = sweep_pts or 0.0
    if sw <= 0.0:
        label, mech = ("Momentum elevator",
                       "no breakdown swept — price held the level and ran")
    elif sw <= 5.0:
        label, mech = ("Shallow sweep-reclaim",
                       f"swept only {sw:.1f} pt below & reclaimed (shallow)")
    elif sw <= 15.0:
        label, mech = ("Sweep-reclaim",
                       f"swept {sw:.1f} pt below the level & reclaimed")
    else:
        label, mech = ("Deep flush-reclaim",
                       f"swept {sw:.1f} pt below & reclaimed (deep flush — highest-quality)")
    acc = ""
    cn = (conf_name or "").lower()
    if "non_acceptance" in cn or "non-acceptance" in cn:
        acc = "; entered on non-acceptance (price refused lower)"
    elif "acceptance" in cn:
        acc = "; entered on acceptance (based above the level)"
    return f"🔍 **FB type: {label}** — {mech}{acc}"


def _level_origin_line(level_type: str, plan_match: Any) -> str:
    """One line stating WHERE the level came from: Mancini's posted plan vs
    the engine's own detection. ``plan_match`` (closest posted setup) wins —
    a level he actually called is on-plan even if the engine tagged it."""
    if plan_match is not None:
        price = float(getattr(plan_match, "level_price", 0.0) or 0.0)
        conv = (getattr(plan_match, "conviction", "") or "").lower()
        conv_tag = f", {conv} conviction" if conv else ""
        return (f"📍 **Source: Mancini's plan** — matches his called "
                f"{price:.0f} level{conv_tag}")
    if (level_type or "").upper() in _MANCINI_LEVEL_TYPES:
        return "📍 **Source: Mancini's plan** (injected level)"
    desc = _ENGINE_LEVEL_DESC.get(
        (level_type or "").upper(),
        (level_type or "?").replace("_", " ").lower(),
    )
    return (f"📍 **Source: engine-detected** ({desc}) — "
            f"not on Mancini's posted plan")


def _find_plan_match(plan: Any, level_price: float,
                     tolerance_pts: float = 2.0,
                     reclaim_zone_below_pts: float = 8.0,
                     direction: str = "long") -> dict | None:
    """Return the closest matching Mancini plan setup, or None.

    Zone-aware (trade 765): his reclaim setups describe a defend->recover
    band ("at 7483, wait for it to defend and recover 7490"), so a
    level_reclaim setup matches engine levels within [level - zone, level],
    not just point-tolerance. Direction-filtered so a nearby SHORT setup
    never labels a long entry."""
    if plan is None or not getattr(plan, "planned_setups", None):
        return None
    from core.signals import plan_setup_matches_level
    best = None
    best_d = None
    for s in plan.planned_setups:
        if (getattr(s, "direction", "") or "").lower() != direction:
            continue
        if not plan_setup_matches_level(s, level_price, tolerance_pts,
                                        reclaim_zone_below_pts):
            continue
        try:
            d = abs(float(s.level_price) - float(level_price))
        except (TypeError, ValueError):
            continue
        if best_d is None or d < best_d:
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
                      trade_id=None,
                      gate_bypass=None) -> dict:
    """Build a Discord embed payload describing a fresh trade entry.

    ``gate_bypass`` is the list of production gates this fill skipped (set
    only in collection mode — e.g. ["Evening block (17:00-22:00 ET)"]). When
    present the embed is clearly marked as a non-production, data-only fill.

    Plain function so it's trivially unit-testable without the bot.
    """
    is_collection = bool(gate_bypass)
    direction = (getattr(position, "direction", None)
                 or getattr(signal, "direction", "long")).lower()
    is_long = direction == "long"
    color = _COLOR_COLLECTION if is_collection else (
        _COLOR_LONG if is_long else _COLOR_SHORT)
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
    collection_prefix = "🧪 COLLECTION  •  " if is_collection else ""
    title = (f"{collection_prefix}{side_icon} {sig_label.upper()} {side_word} "
             f"@ {fill_price:.2f}  •  {contracts_ordered} {symbol}{title_extra}")

    description_parts: list[str] = []
    if is_collection:
        description_parts.append(
            "🧪 **COLLECTION MODE — data only, production would SKIP this trade**\n"
            f"_Bypassed time gate(s): {', '.join(gate_bypass)}_"
        )
    if plan_match_str:
        description_parts.append(plan_match_str)
    # Where did this level come from — Mancini's posted plan or the engine?
    description_parts.append(_level_origin_line(level_type, plan_match))
    description_parts.append(
        f"Pattern: **{sig_name}**  •  Level: **{level_type}** @ {level_price:.2f}"
    )
    # For failed breakdowns, spell out WHAT KIND fired (momentum elevator vs
    # deep flush-reclaim) from the sweep depth, so the reader knows the entry's
    # real character — not just the generic FAILED_BREAKDOWN label.
    if sig_name == "FAILED_BREAKDOWN":
        description_parts.append(_fb_logic_line(sweep, conf_name))
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
                     trade_id=None,
                     gate_bypass=None) -> dict:
    """Build a Discord embed for an exit event (T1 / T2 / stop / runner).

    ``gate_bypass`` carries through from the entry: when set, this trade was a
    collection-mode (non-production) fill, so the exit stays tagged 🧪 instead
    of reading like a real T1/stop.
    """
    is_collection = bool(gate_bypass)
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
    if is_collection:
        color = _COLOR_COLLECTION

    collection_prefix = "🧪 COLLECTION  •  " if is_collection else ""
    title = (f"{collection_prefix}{title_label}  •  {contracts_closed} of "
             f"{contracts_closed + remaining_contracts} closed @ {fill_price:.2f}")

    def _usd(x: float) -> str:
        # "+$187" / "-$168" — never the malformed "$-168" / "$+187"
        return f"{'-' if x < 0 else '+'}${abs(x):,.0f}"

    desc_parts = []
    if is_collection:
        desc_parts.append(
            "🧪 **COLLECTION MODE exit** — non-production trade (data only)"
        )
    # "Locked" is winner language — a stop-out must read as a loss
    # (trade 746's card said "Locked -16.8 pt ... $-168").
    if per_contract_pts >= 0:
        desc_parts.append(
            f"{status_icon} Locked +{per_contract_pts:.1f} pt × "
            f"{contracts_closed} = {_usd(slice_pnl_dollars)}"
        )
    else:
        desc_parts.append(
            f"{status_icon} Lost {abs(per_contract_pts):.1f} pt × "
            f"{contracts_closed} = {_usd(slice_pnl_dollars)}"
        )
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

    # Dollars lead (unambiguous); the pt figure is per-contract, since the
    # bare contract-summed total ("-33.5 pt" on a 16.5-pt stop) reads as if
    # the market moved twice as far as it did.
    cum_dollars = realized_pnl_pts_so_far * point_value
    total_ct = max(1, contracts_closed + remaining_contracts)
    per_ct_cum = realized_pnl_pts_so_far / total_ct
    cum_sign = "+" if per_ct_cum >= 0 else ""
    ts = (fill_time or datetime.now(_ET)).astimezone(_ET)
    desc_parts.append(
        f"\nTrade P&L so far: **{_usd(cum_dollars)}** "
        f"({cum_sign}{per_ct_cum:.1f} pt/contract)   •   "
        f"{ts.strftime('%I:%M %p ET')}"
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
    match = _find_plan_match(plan, float(lvl if lvl is not None else entry),
                             reclaim_zone_below_pts=0.0, direction="short")
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
