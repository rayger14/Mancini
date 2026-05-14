"""Nightly analysis report — reads today's data and produces recommendations.

Runs after Substack cron at 2:10 AM ET. Reads trades.jsonl, analyzes trades
taken, near-misses, and blind spots, then sends a Discord summary.

Usage:
    python3 live/nightly_report.py

Cron (add to VM crontab):
    10 2 * * * docker exec mancini_mancini-bot_1 python3 live/nightly_report.py >> /home/ubuntu/mancini/logs/nightly_report.log 2>&1

Environment:
    WATCHDOG_WEBHOOK — Discord webhook URL for the report
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


TRADE_LOG = Path(os.environ.get("TRADE_LOG", "/app/logs/trades.jsonl"))
SHADOW_LOG = Path(os.environ.get("SHADOW_LOG", "/app/logs/shadow_trades.jsonl"))
MANCINI_DIR = Path(os.environ.get("MANCINI_DIR", "/app/data"))
WEBHOOK_URL = os.environ.get("WATCHDOG_WEBHOOK", "")


def load_today_records(log_path: Path, session_date: str) -> list[dict]:
    """Load all JSONL records matching today's session_date."""
    records = []
    if not log_path.exists():
        return records
    try:
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                sd = r.get("session_date", "")
                ts = r.get("timestamp", "")
                if sd == session_date or ts.startswith(session_date):
                    records.append(r)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    return records


def analyze_trades(records: list[dict]) -> dict:
    """Analyze today's trade entries and exits."""
    entries = [r for r in records if r.get("event") == "entry"]
    exits = [r for r in records if r.get("event") == "exit"]

    trades = []
    for ex in exits:
        pnl = ex.get("pnl_pts", 0)
        lqs = ex.get("lqs", 0)
        pat = ex.get("pattern_type", "?")
        entry_price = ex.get("entry_price", 0)
        exit_price = ex.get("exit_price", 0)
        reason = ex.get("exit_reason", "")
        won = pnl > 0 if pnl is not None else False
        trades.append({
            "won": won,
            "pnl": pnl or 0,
            "lqs": lqs or 0,
            "pattern": pat,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "reason": reason,
        })

    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["won"]]
    losses = [t for t in trades if not t["won"]]

    # LQS accuracy
    high_lqs = [t for t in trades if t["lqs"] >= 55]
    mid_lqs = [t for t in trades if 25 <= t["lqs"] < 55]
    low_lqs = [t for t in trades if t["lqs"] < 25]

    return {
        "count": len(trades),
        "trades": trades,
        "total_pnl": total_pnl,
        "wins": len(wins),
        "losses": len(losses),
        "high_lqs": {"count": len(high_lqs), "wins": len([t for t in high_lqs if t["won"]])},
        "mid_lqs": {"count": len(mid_lqs), "wins": len([t for t in mid_lqs if t["won"]])},
        "low_lqs": {"count": len(low_lqs), "wins": len([t for t in low_lqs if t["won"]])},
    }


def analyze_winners(records: list[dict]) -> dict:
    """Analyze what made winning trades work — extract the common factors."""
    entries = {r.get("trade_id"): r for r in records if r.get("event") == "entry"}
    exits = [r for r in records if r.get("event") == "exit" and r.get("pnl_pts", 0) > 0]

    patterns = []
    for ex in exits:
        tid = ex.get("trade_id")
        entry = entries.get(tid, {})
        sig = entry.get("signal", {})
        mc = entry.get("market_correlation", {})

        patterns.append({
            "pnl": ex.get("pnl_pts", 0),
            "pattern": entry.get("pattern_type", "?"),
            "level_type": sig.get("level_type", entry.get("level_type", "?")),
            "level_price": sig.get("level_price", 0),
            "lqs": ex.get("lqs", entry.get("lqs", 0)),
            "sweep_depth": entry.get("sweep_depth_pts", 0),
            "confirmation": entry.get("confirmation_type", "?"),
            "mfe": ex.get("mfe_pts", 0),
            "mae": ex.get("mae_pts", 0),
            "bars_held": ex.get("bars_held", 0),
            "session_window": entry.get("session_window", ""),
            "vix": mc.get("vix", 0),
            "volume_trend": entry.get("volume_trend", 0),
            "session_range": entry.get("session_range", 0),
        })

    if not patterns:
        return {"count": 0, "patterns": [], "common_factors": []}

    # Find common factors among winners
    common = []
    level_types = [p["level_type"] for p in patterns if p["level_type"] != "?"]
    if level_types:
        from collections import Counter
        most_common_level = Counter(level_types).most_common(1)[0]
        common.append(f"Best level type: {most_common_level[0]} ({most_common_level[1]}/{len(patterns)} wins)")

    avg_sweep = sum(p["sweep_depth"] or 0 for p in patterns) / len(patterns)
    if avg_sweep > 0:
        common.append(f"Avg sweep depth: {avg_sweep:.1f} pts")

    avg_mfe = sum(p["mfe"] or 0 for p in patterns) / len(patterns)
    avg_mae = sum(p["mae"] or 0 for p in patterns) / len(patterns)
    if avg_mfe > 0:
        common.append(f"Avg MFE: +{avg_mfe:.1f} pts, Avg MAE: -{avg_mae:.1f} pts")

    confirmations = [p["confirmation"] for p in patterns if p["confirmation"] != "?"]
    if confirmations:
        from collections import Counter
        conf_counts = Counter(confirmations).most_common()
        common.append(f"Confirmation: {', '.join(f'{c}({n})' for c, n in conf_counts)}")

    avg_lqs = sum(p["lqs"] or 0 for p in patterns) / len(patterns)
    common.append(f"Avg LQS: {avg_lqs:.0f}")

    vix_vals = [p["vix"] for p in patterns if p["vix"]]
    if vix_vals:
        common.append(f"Avg VIX at entry: {sum(vix_vals)/len(vix_vals):.1f}")

    return {
        "count": len(patterns),
        "patterns": patterns[:5],
        "common_factors": common,
    }


def analyze_losers(records: list[dict]) -> dict:
    """Analyze what made losing trades fail — extract warning signs."""
    entries = {r.get("trade_id"): r for r in records if r.get("event") == "entry"}
    exits = [r for r in records if r.get("event") == "exit" and (r.get("pnl_pts") or 0) < 0]

    patterns = []
    for ex in exits:
        tid = ex.get("trade_id")
        entry = entries.get(tid, {})
        sig = entry.get("signal", {})
        mc = entry.get("market_correlation", {})

        patterns.append({
            "pnl": ex.get("pnl_pts", 0),
            "pattern": entry.get("pattern_type", "?"),
            "level_type": sig.get("level_type", entry.get("level_type", "?")),
            "level_price": sig.get("level_price", 0),
            "lqs": ex.get("lqs", entry.get("lqs", 0)),
            "sweep_depth": entry.get("sweep_depth_pts", 0),
            "confirmation": entry.get("confirmation_type", "?"),
            "mfe": ex.get("mfe_pts", 0),
            "mae": ex.get("mae_pts", 0),
            "bars_held": ex.get("bars_held", 0),
            "exit_reason": ex.get("exit_reason", ""),
            "session_window": entry.get("session_window", ""),
            "vix": mc.get("vix", 0),
            "session_range": entry.get("session_range", 0),
        })

    if not patterns:
        return {"count": 0, "patterns": [], "warning_signs": []}

    # Find common factors among losers — what went wrong?
    warnings = []

    # Check if losers had low sweep depth (shallow = fake sweep)
    avg_sweep = sum(p["sweep_depth"] or 0 for p in patterns) / len(patterns)
    if avg_sweep < 3:
        warnings.append(f"Avg sweep depth only {avg_sweep:.1f} pts — shallow sweeps = weak setups")

    # Check MFE — did it ever go in our favor?
    avg_mfe = sum(p["mfe"] or 0 for p in patterns) / len(patterns)
    avg_mae = sum(p["mae"] or 0 for p in patterns) / len(patterns)
    if avg_mfe < 5:
        warnings.append(f"Avg MFE only +{avg_mfe:.1f} pts — trades never got momentum")
    if avg_mae > 15:
        warnings.append(f"Avg MAE -{avg_mae:.1f} pts — too much drawdown before stop")

    # Check level types
    from collections import Counter
    level_types = [p["level_type"] for p in patterns if p["level_type"] != "?"]
    if level_types:
        worst = Counter(level_types).most_common(1)[0]
        warnings.append(f"Most losing level type: {worst[0]} ({worst[1]}/{len(patterns)} losses)")

    # Check LQS
    avg_lqs = sum(p["lqs"] or 0 for p in patterns) / len(patterns)
    warnings.append(f"Avg loser LQS: {avg_lqs:.0f} (vs winners — are losers lower-scored?)")

    # Check session window
    windows = [p["session_window"] for p in patterns if p["session_window"]]
    if windows:
        worst_window = Counter(windows).most_common(1)[0]
        if "Evening" in worst_window[0] or "Late Night" in worst_window[0]:
            warnings.append(f"Most losses in: {worst_window[0][:30]} — consider tightening off-hours")

    # Check bars held — were we too patient or too quick?
    avg_bars = sum(p["bars_held"] or 0 for p in patterns) / len(patterns)
    if avg_bars > 200:
        warnings.append(f"Avg bars held: {avg_bars:.0f} — holding too long, tighten max hold")
    elif avg_bars < 20:
        warnings.append(f"Avg bars held: {avg_bars:.0f} — stopped out quickly, stops may be too tight")

    return {
        "count": len(patterns),
        "patterns": patterns[:5],
        "warning_signs": warnings,
        "total_loss": sum(p["pnl"] for p in patterns),
    }


def analyze_near_misses(records: list[dict]) -> dict:
    """Analyze near-misses and their outcomes."""
    misses = [r for r in records if r.get("event") == "near_miss"]
    resolved = [r for r in records if r.get("event") == "near_miss_resolved"]

    # Match resolved to misses by timestamp
    resolved_by_ts = {}
    for res in resolved:
        ts = res.get("near_miss_timestamp", res.get("timestamp", ""))
        resolved_by_ts[ts] = res

    analyzed = []
    would_have_won = 0
    would_have_lost = 0
    gates_saved = 0.0
    gates_cost = 0.0

    for nm in misses:
        ts = nm.get("timestamp", "")
        reason = nm.get("failure_reason", "?")
        lqs = nm.get("lqs", 0)
        entry = nm.get("entry_price", 0)
        stop = nm.get("stop_price", 0)
        target = nm.get("target_1", 0)
        level_type = nm.get("level_type", "?")
        achieved = nm.get("achieved", {})
        required = nm.get("required", {})

        # Check if resolved
        outcome = resolved_by_ts.get(ts, {})
        outcome_result = outcome.get("outcome", "unresolved")
        outcome_pnl = outcome.get("pnl_pts", 0)

        if outcome_result == "target_hit":
            would_have_won += 1
            gates_cost += abs(outcome_pnl) if outcome_pnl else 0
        elif outcome_result == "stop_hit":
            would_have_lost += 1
            gates_saved += abs(outcome_pnl) if outcome_pnl else 0

        analyzed.append({
            "reason": reason,
            "lqs": lqs,
            "entry": entry,
            "level_type": level_type,
            "outcome": outcome_result,
            "outcome_pnl": outcome_pnl,
            "achieved": achieved,
            "required": required,
        })

    return {
        "count": len(misses),
        "analyzed": analyzed[:10],  # top 10
        "would_have_won": would_have_won,
        "would_have_lost": would_have_lost,
        "gates_saved": gates_saved,
        "gates_cost": gates_cost,
        "net_gate_value": gates_saved - gates_cost,
    }


def analyze_mancini_blind_spots(session_date: str, records: list[dict]) -> list[dict]:
    """Find Mancini levels that were swept but had no engine coverage."""
    mancini_path = MANCINI_DIR / f"mancini_levels_{session_date}.json"
    if not mancini_path.exists():
        return []

    try:
        mancini_data = json.loads(mancini_path.read_text())
        mancini_levels = mancini_data.get("levels", [])
    except Exception:
        return []

    # Get all levels the engine had (from entry records)
    engine_levels = set()
    for r in records:
        if r.get("event") == "entry":
            sig = r.get("signal", {})
            if sig.get("level_price"):
                engine_levels.add(round(sig["level_price"], 1))
        nearby = r.get("nearby_levels", [])
        for nl in nearby:
            if nl.get("price"):
                engine_levels.add(round(nl["price"], 1))

    blind_spots = []
    for ml in mancini_levels:
        mp = ml.get("price", 0)
        if mp <= 0:
            continue
        # Check if engine had a level within 3 pts
        covered = any(abs(ep - mp) <= 3.0 for ep in engine_levels)
        if not covered:
            blind_spots.append({
                "price": mp,
                "side": ml.get("side", "?"),
                "conviction": ml.get("conviction", 0),
            })

    return blind_spots[:10]


def generate_recommendations(trade_analysis: dict, miss_analysis: dict, blind_spots: list) -> list[str]:
    """Generate actionable recommendations from today's data."""
    recs = []

    # Check if high-LQS trades are outperforming
    high = trade_analysis["high_lqs"]
    mid = trade_analysis["mid_lqs"]
    if high["count"] > 0 and high["wins"] == high["count"]:
        recs.append("LQS gating validated: all high-LQS trades won today")
    elif high["count"] > 0 and high["wins"] < high["count"] // 2:
        recs.append("WARNING: High-LQS trades underperforming — review scoring weights")

    # Check near-miss gate value
    if miss_analysis["net_gate_value"] > 0:
        recs.append(f"Gates are net positive today: saved {miss_analysis['gates_saved']:.0f} pts, cost {miss_analysis['gates_cost']:.0f} pts")
    elif miss_analysis["net_gate_value"] < -20:
        recs.append(f"Gates are net negative today: cost {miss_analysis['gates_cost']:.0f} pts of missed winners")

    # Check acceptance timeout pattern
    acceptance_misses = [m for m in miss_analysis["analyzed"] if m["reason"] == "acceptance_timeout"]
    winning_acceptance = [m for m in acceptance_misses if m["outcome"] == "target_hit"]
    if len(winning_acceptance) >= 2:
        achieved_bars = [m["achieved"].get("hold_bars", 0) for m in winning_acceptance if m.get("achieved")]
        if achieved_bars:
            avg_achieved = sum(achieved_bars) / len(achieved_bars)
            recs.append(f"acceptance_timeout blocked {len(winning_acceptance)} winners (avg held {avg_achieved:.0f} bars). Consider lowering acceptance_min_hold_bars.")

    # Blind spots
    if blind_spots:
        recs.append(f"{len(blind_spots)} Mancini levels had no engine coverage — enable Mancini overlay to fill gaps")

    # No trades warning
    if trade_analysis["count"] == 0:
        recs.append("No trades today. Check if market was in chop (near-miss outcomes) or if gates are too strict.")

    return recs if recs else ["No specific recommendations today. System performing as expected."]


def format_discord_report(
    session_date: str,
    trade_analysis: dict,
    miss_analysis: dict,
    blind_spots: list,
    recommendations: list[str],
    winner_analysis: dict = None,
    loser_analysis: dict = None,
) -> dict:
    """Format the analysis as a Discord embed."""
    # Section 1: Trades
    if trade_analysis["count"] > 0:
        trade_lines = []
        for t in trade_analysis["trades"][:5]:
            icon = "✅" if t["won"] else "❌"
            pat = t["pattern"].replace("failed_breakdown", "FB").replace("breakdown_short", "BD").replace("velocity_short", "VBD")
            trade_lines.append(f"{icon} {pat} @ {t['entry_price']:.0f} (LQS {t['lqs']}) → {t['pnl']:+.1f} pts")
        trade_text = "\n".join(trade_lines)
        trade_text += f"\n**Net: {trade_analysis['total_pnl']:+.1f} pts** ({trade_analysis['wins']}W/{trade_analysis['losses']}L)"
    else:
        trade_text = "No trades taken today"

    # Section 2: Winner analysis
    if winner_analysis and winner_analysis["count"] > 0:
        winner_lines = []
        for p in winner_analysis["patterns"][:3]:
            pat = p["pattern"].replace("failed_breakdown", "FB").replace("breakdown_short", "BD").replace("velocity_short", "VBD")
            winner_lines.append(f"🏆 {pat} @ {p['level_type']} (LQS {p['lqs']}) → +{p['pnl']:.1f} pts, MFE +{p['mfe']:.1f}")
        winner_text = "\n".join(winner_lines)
        if winner_analysis["common_factors"]:
            winner_text += "\n**What worked:**\n" + "\n".join(f"• {f}" for f in winner_analysis["common_factors"])
    else:
        winner_text = "No winners today"

    # Section 3: Loser analysis
    if loser_analysis and loser_analysis["count"] > 0:
        loser_lines = []
        for p in loser_analysis["patterns"][:3]:
            pat = p["pattern"].replace("failed_breakdown", "FB").replace("breakdown_short", "BD").replace("velocity_short", "VBD")
            loser_lines.append(f"💀 {pat} @ {p['level_type']} (LQS {p['lqs']}) → {p['pnl']:+.1f} pts, MFE +{p['mfe']:.1f}")
        loser_text = "\n".join(loser_lines)
        loser_text += f"\n**Total lost: {loser_analysis['total_loss']:+.1f} pts**"
        if loser_analysis["warning_signs"]:
            loser_text += "\n**Warning signs:**\n" + "\n".join(f"⚠️ {w}" for w in loser_analysis["warning_signs"])
    else:
        loser_text = "No losers today 🎉"

    # Section 4: Near-misses
    if miss_analysis["count"] > 0:
        miss_text = f"**{miss_analysis['count']} near-misses** "
        miss_text += f"({miss_analysis['would_have_won']} would-have-won, {miss_analysis['would_have_lost']} would-have-lost)\n"
        if miss_analysis["net_gate_value"] >= 0:
            miss_text += f"Gates saved **{miss_analysis['gates_saved']:.0f} pts** (net positive ✅)"
        else:
            miss_text += f"Gates cost **{miss_analysis['gates_cost']:.0f} pts** of missed winners ⚠️"
    else:
        miss_text = "No near-misses today"

    # Section 3: Blind spots
    if blind_spots:
        spot_lines = [f"• {bs['price']:.0f} ({bs['side']}, conviction {bs['conviction']})" for bs in blind_spots[:5]]
        blind_text = "\n".join(spot_lines)
    else:
        blind_text = "No blind spots — engine covered all Mancini levels ✅"

    # Section 4: Recommendations
    rec_text = "\n".join(f"• {r}" for r in recommendations[:5])

    embed = {
        "title": f"📊 Nightly Report — {session_date}",
        "color": 0x3498DB if trade_analysis["total_pnl"] >= 0 else 0xE74C3C,
        "fields": [
            {"name": "Trades", "value": trade_text[:1000], "inline": False},
            {"name": "Winner Patterns", "value": winner_text[:1000], "inline": False},
            {"name": "Loser Analysis", "value": loser_text[:1000], "inline": False},
            {"name": "Near-Misses", "value": miss_text[:1000], "inline": False},
            {"name": "Mancini Blind Spots", "value": blind_text[:500], "inline": False},
            {"name": "Recommendations", "value": rec_text[:1000], "inline": False},
        ],
        "footer": {"text": "Mancini Bot • Nightly Analysis"},
    }

    return {"username": "Mancini Quant", "embeds": [embed]}


def main():
    # Determine session date (today, or yesterday if before 6 PM ET)
    now = datetime.now()
    session_date = date.today().isoformat()

    print(f"[NIGHTLY REPORT] Generating report for session {session_date}")

    # Load data
    records = load_today_records(TRADE_LOG, session_date)
    shadow_records = load_today_records(SHADOW_LOG, session_date)

    print(f"  Trade records: {len(records)}")
    print(f"  Shadow records: {len(shadow_records)}")

    # Analyze
    trade_analysis = analyze_trades(records)
    winner_analysis = analyze_winners(records)
    loser_analysis = analyze_losers(records)
    miss_analysis = analyze_near_misses(records)
    blind_spots = analyze_mancini_blind_spots(session_date, records)
    recommendations = generate_recommendations(trade_analysis, miss_analysis, blind_spots)

    print(f"  Trades: {trade_analysis['count']} ({trade_analysis['wins']}W/{trade_analysis['losses']}L)")
    print(f"  Winners analyzed: {winner_analysis['count']}")
    print(f"  Losers analyzed: {loser_analysis['count']}")
    print(f"  Near-misses: {miss_analysis['count']}")
    print(f"  Blind spots: {len(blind_spots)}")
    print(f"  Recommendations: {len(recommendations)}")

    # Format and send
    payload = format_discord_report(session_date, trade_analysis, miss_analysis, blind_spots, recommendations, winner_analysis, loser_analysis)

    # Print to stdout
    print(f"\n{json.dumps(payload, indent=2, default=str)[:2000]}")

    # Archive structured report for future model learning
    archive_path = Path(os.environ.get("REPORT_ARCHIVE", "/app/logs/nightly_reports.jsonl"))
    try:
        report_record = {
            "session_date": session_date,
            "generated_at": datetime.now().isoformat(),
            "trades": trade_analysis,
            "winners": winner_analysis,
            "losers": loser_analysis,
            "near_misses": {
                "count": miss_analysis["count"],
                "would_have_won": miss_analysis["would_have_won"],
                "would_have_lost": miss_analysis["would_have_lost"],
                "gates_saved": miss_analysis["gates_saved"],
                "gates_cost": miss_analysis["gates_cost"],
                "net_gate_value": miss_analysis["net_gate_value"],
            },
            "blind_spots": blind_spots,
            "recommendations": recommendations,
        }
        with open(archive_path, "a") as f:
            f.write(json.dumps(report_record, default=str) + "\n")
        print(f"  Archived to {archive_path}")
    except Exception as e:
        print(f"  Archive failed: {e}")

    # Send to Discord
    if WEBHOOK_URL and _HAS_REQUESTS:
        try:
            resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
            print(f"\n  Discord: sent ({resp.status_code})")
        except Exception as e:
            print(f"\n  Discord: failed ({e})")
    else:
        print("\n  Discord: no webhook configured")

    print("[NIGHTLY REPORT] Done")


if __name__ == "__main__":
    main()
