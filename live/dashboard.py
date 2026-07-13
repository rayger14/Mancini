"""Lightweight web dashboard for monitoring the Mancini trading bot.

Shows real-time status, detected levels, signals, and trades without
needing to log into IB (which would kick the bot's session).

Usage:
    python3 live/dashboard.py              # serves on port 8080
    python3 live/dashboard.py --port 9090  # custom port
"""

from __future__ import annotations

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from string import Template

STATUS_FILE = os.environ.get("STATUS_FILE", "/app/logs/status.json")
LOG_FILE = os.environ.get("LOG_FILE", "/app/logs/bot.log")
TRADES_FILE = os.environ.get("TRADE_LOG", "/app/logs/trades.jsonl")
PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))

HTML_TEMPLATE = Template(r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mancini Bot</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0d1117; color: #c9d1d9;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
    font-size: 14px; padding: 0;
  }

  /* Header bar */
  .header {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border-bottom: 1px solid #30363d;
    padding: 16px 24px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .header-left { display: flex; align-items: center; gap: 16px; }
  .logo { font-size: 18px; font-weight: 700; color: #58a6ff; letter-spacing: 0.5px; }
  .connection-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .dot-connected { background: #3fb950; box-shadow: 0 0 8px #3fb95080; }
  .dot-disconnected { background: #f85149; box-shadow: 0 0 8px #f8514980; }
  .header-right { display: flex; align-items: center; gap: 20px; color: #8b949e; font-size: 12px; }

  /* Session badge */
  .session-badge {
    padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .session-rth { background: #238636; color: #fff; }
  .session-globex { background: #1f6feb; color: #fff; }
  .session-blocked { background: #da3633; color: #fff; }
  .session-closed { background: #484f58; color: #c9d1d9; }

  /* Tab navigation */
  .tab-bar { display: flex; gap: 0; border-bottom: 1px solid #30363d; background: #161b22; padding: 0 24px; }
  .tab-btn {
    padding: 12px 24px; cursor: pointer; border: none; background: none;
    color: #8b949e; font-size: 13px; font-weight: 600; border-bottom: 2px solid transparent;
    transition: all 0.2s;
  }
  .tab-btn:hover { color: #c9d1d9; }
  .tab-btn.active { color: #58a6ff; border-bottom-color: #58a6ff; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .main { padding: 20px 24px; max-width: 1400px; margin: 0 auto; }

  /* Ticker strip */
  .ticker-strip {
    display: flex; align-items: center; gap: 32px; padding: 12px 20px;
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    margin-bottom: 16px; flex-wrap: wrap;
  }
  .ticker-price { font-size: 32px; font-weight: 700; color: #f0f6fc; font-family: 'SF Mono', monospace; }
  .ticker-symbol { font-size: 13px; color: #8b949e; margin-right: -20px; }
  .ticker-stat { text-align: center; }
  .ticker-stat-label { font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
  .ticker-stat-value { font-size: 15px; font-weight: 600; color: #c9d1d9; font-family: 'SF Mono', monospace; }

  /* Grids */
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }

  /* Cards */
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; overflow: hidden; }
  .card-header {
    font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #21262d;
  }

  .pnl-big { font-size: 36px; font-weight: 700; font-family: 'SF Mono', monospace; margin: 8px 0; }
  .pnl-sub { font-size: 12px; color: #8b949e; }

  .green { color: #3fb950; } .red { color: #f85149; } .yellow { color: #d29922; }
  .blue { color: #58a6ff; } .muted { color: #484f58; } .dim { color: #8b949e; }

  .stat-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; }
  .stat-row + .stat-row { border-top: 1px solid #21262d; }
  .stat-label { color: #8b949e; font-size: 13px; }
  .stat-value { color: #f0f6fc; font-weight: 600; font-size: 13px; font-family: 'SF Mono', monospace; }

  .position-banner {
    padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .pos-long { background: #238636; color: #fff; }
  .pos-short { background: #da3633; color: #fff; }
  .pos-flat { background: #21262d; color: #8b949e; }
  .pos-direction { font-size: 16px; font-weight: 700; text-transform: uppercase; }
  .pos-pattern { font-size: 12px; opacity: 0.8; }

  table { width: 100%; border-collapse: collapse; }
  th {
    text-align: left; font-size: 10px; color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.5px; padding: 8px; border-bottom: 1px solid #30363d; font-weight: 600;
  }
  td { padding: 8px; border-bottom: 1px solid #21262d; font-size: 13px; font-family: 'SF Mono', monospace; }
  tr:hover { background: #1c2128; }

  /* Level tags with tooltips */
  .level-tag {
    font-size: 10px; padding: 2px 6px; border-radius: 4px; background: #21262d;
    color: #8b949e; display: inline-block; position: relative; cursor: help;
  }
  .level-tag .level-tip {
    display: none; position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px;
    font-size: 11px; color: #c9d1d9; white-space: nowrap; z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4); min-width: 180px; font-family: -apple-system, sans-serif;
  }
  .level-tag .level-tip::after {
    content: ''; position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
    border: 5px solid transparent; border-top-color: #30363d;
  }
  .level-tag:hover .level-tip { display: block; }

  /* Touches header tooltip */
  .th-tip { position: relative; cursor: help; text-decoration: underline dotted; }
  .th-tip .th-tip-text {
    display: none; position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px;
    font-size: 11px; color: #c9d1d9; white-space: normal; z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4); width: 220px; font-weight: 400;
    text-transform: none; letter-spacing: 0; font-family: -apple-system, sans-serif;
  }
  .th-tip:hover .th-tip-text { display: block; }

  /* Bypass gate badges */
  .bypass-tag {
    font-size: 9px; padding: 1px 6px; border-radius: 3px; font-weight: 600;
    display: inline-block; margin-left: 4px;
  }
  .bypass-collection { background: #d2992230; color: #d29922; border: 1px solid #d2992240; }
  .bypass-production { background: #23863620; color: #3fb950; border: 1px solid #23863640; }

  /* Near-miss section */
  .near-miss-toggle {
    cursor: pointer; user-select: none; display: flex; align-items: center;
    gap: 6px; padding: 4px 0;
  }
  .near-miss-toggle .arrow { transition: transform 0.2s; display: inline-block; font-size: 10px; }
  .near-miss-toggle .arrow.open { transform: rotate(90deg); }
  .near-miss-item {
    padding: 8px 12px; background: #0d1117; border-radius: 6px; margin-bottom: 6px;
    border-left: 3px solid #d29922; font-size: 12px; line-height: 1.5;
  }
  .near-miss-reason { color: #d29922; font-weight: 600; }
  .near-miss-detail { color: #8b949e; font-size: 11px; }

  .log-viewer { max-height: 300px; overflow-y: auto; font-family: 'SF Mono', monospace; font-size: 11px; line-height: 1.8; }
  .log-line { padding: 1px 0; white-space: pre-wrap; word-break: break-all; }
  .log-entry { color: #3fb950; font-weight: 600; }
  .log-exit { color: #f85149; font-weight: 600; }
  .log-signal { color: #d29922; }
  .log-bar { color: #484f58; }

  .phantom-row { opacity: 0.7; }
  .phantom-tag { font-size: 9px; padding: 1px 5px; border-radius: 3px; background: #d2992220; color: #d29922; font-weight: 600; }

  /* Interactive session timeline */
  .timeline { display: flex; height: 28px; border-radius: 6px; overflow: visible; background: #21262d; margin: 8px 0; position: relative; }
  .tl-segment {
    height: 100%; position: relative; cursor: pointer; transition: filter 0.2s;
    display: flex; align-items: center; justify-content: center;
    font-size: 9px; font-weight: 600; color: rgba(255,255,255,0.7); letter-spacing: 0.3px;
    overflow: visible;
  }
  .tl-segment:first-child { border-radius: 6px 0 0 6px; }
  .tl-segment:last-child { border-radius: 0 6px 6px 0; }
  .tl-segment:hover { filter: brightness(1.3); z-index: 1; }
  .tl-tooltip {
    display: none; position: absolute; bottom: 36px; left: 50%; transform: translateX(-50%);
    background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px;
    font-size: 11px; color: #c9d1d9; white-space: nowrap; z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4); min-width: 240px;
  }
  .tl-tooltip::after {
    content: ''; position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
    border: 6px solid transparent; border-top-color: #30363d;
  }
  .tl-segment:hover .tl-tooltip { display: block; }
  .tl-tooltip-title { font-weight: 700; color: #f0f6fc; margin-bottom: 4px; }
  .tl-tooltip-time { color: #8b949e; margin-bottom: 4px; }
  .tl-tooltip-desc { color: #8b949e; font-size: 10px; line-height: 1.5; white-space: normal; }
  .tl-tooltip-status { margin-top: 4px; font-weight: 700; }
  .tl-tooltip-mode { margin-top: 4px; font-size: 10px; color: #d29922; font-style: italic; }

  .tl-maintenance { background: #484f58; flex: 1; }
  .tl-evening { background: #1f6feb; flex: 4; opacity: 0.5; }
  .tl-overnight { background: #1f6feb; flex: 4; }
  .tl-euro { background: #1f6feb; flex: 4; opacity: 0.5; }
  .tl-premarket { background: #1f6feb; flex: 3.5; opacity: 0.7; }
  .tl-morning { background: #238636; flex: 1.5; }
  .tl-midday { background: #238636; flex: 2; opacity: 0.7; }
  .tl-chop { background: #238636; flex: 2; opacity: 0.5; }
  .tl-afternoon { background: #238636; flex: 1.83; opacity: 0.8; }
  .tl-settle { background: #484f58; flex: 1; }
  .tl-labels { display: flex; justify-content: space-between; font-size: 9px; color: #484f58; margin-top: 2px; }

  .done-badge {
    background: #da363320; color: #f85149; border: 1px solid #da3633;
    padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;
  }
  .active-badge {
    background: #23863620; color: #3fb950; border: 1px solid #238636;
    padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;
  }

  /* Info blurb */
  .info-blurb {
    font-size: 11px; color: #8b949e; line-height: 1.5; padding: 8px 12px;
    background: #0d1117; border-radius: 6px; margin-bottom: 12px;
  }

  #chart-container {
    width: 100%; height: 400px; margin-bottom: 16px;
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    overflow: hidden; position: relative;
  }

  /* Strategy tab */
  .strategy-hero {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d; border-radius: 8px; padding: 24px; margin-bottom: 16px;
  }
  .strategy-hero h2 { color: #58a6ff; font-size: 20px; margin-bottom: 8px; }
  .strategy-hero p { color: #8b949e; font-size: 13px; line-height: 1.6; }
  .pattern-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; position: relative; overflow: hidden; }
  .pattern-card.active { border-color: #238636; }
  .pattern-card.disabled { border-color: #da363380; opacity: 0.7; }
  .pattern-name { font-size: 16px; font-weight: 700; color: #f0f6fc; margin-bottom: 4px; }
  .pattern-type { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
  .pattern-desc { color: #8b949e; font-size: 13px; line-height: 1.6; margin-bottom: 12px; }
  .pattern-stats { display: flex; gap: 16px; flex-wrap: wrap; }
  .pattern-stat { text-align: center; padding: 8px 12px; background: #0d1117; border-radius: 6px; }
  .pattern-stat-val { font-size: 16px; font-weight: 700; font-family: 'SF Mono', monospace; }
  .pattern-stat-lbl { font-size: 9px; color: #8b949e; text-transform: uppercase; }
  .pattern-badge {
    position: absolute; top: 16px; right: 16px;
    padding: 4px 10px; border-radius: 12px; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .badge-active { background: #238636; color: #fff; }
  .badge-disabled { background: #484f58; color: #8b949e; }
  .badge-blocked { background: #da3633; color: #fff; }
  .flow-steps { display: flex; gap: 8px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
  .flow-step { background: #0d1117; padding: 6px 12px; border-radius: 6px; font-size: 11px; color: #c9d1d9; border: 1px solid #30363d; }
  .flow-arrow { color: #484f58; font-size: 16px; }
  .backtest-note { font-size: 10px; color: #484f58; margin-top: 8px; font-style: italic; }

  /* Regime modal */
  .regime-clickable { cursor: pointer; text-decoration: underline; text-decoration-style: dotted; }
  .regime-clickable:hover { color: #58a6ff; }
  .modal-overlay {
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.7); z-index: 1000; justify-content: center; align-items: center;
  }
  .modal-overlay.show { display: flex; }
  .modal-content {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 24px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto;
  }
  .modal-close { float: right; cursor: pointer; color: #8b949e; font-size: 20px; border: none; background: none; }
  .modal-close:hover { color: #f0f6fc; }

  .not-mancini {
    background: #d2992215; border: 1px solid #d2992240; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 16px;
  }
  .not-mancini-title { color: #d29922; font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
  .not-mancini-text { color: #8b949e; font-size: 12px; line-height: 1.6; }

  /* Trade History tab */
  .history-summary { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
  .history-stat { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; text-align: center; flex: 1; min-width: 100px; }
  .history-stat-val { font-size: 18px; font-weight: 700; font-family: 'SF Mono', monospace; }
  .history-stat-lbl { font-size: 9px; color: #8b949e; text-transform: uppercase; margin-top: 4px; letter-spacing: 0.5px; }
  .history-filters { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
  .filter-pill { padding: 4px 10px; border-radius: 12px; font-size: 11px; cursor: pointer; border: 1px solid #30363d; background: #21262d; color: #8b949e; transition: all 0.15s; }
  .filter-pill:hover { border-color: #58a6ff; color: #c9d1d9; }
  .filter-pill.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
  .filter-toggle { display: flex; align-items: center; gap: 6px; font-size: 11px; color: #8b949e; cursor: pointer; margin-left: 12px; }
  .filter-toggle input { accent-color: #1f6feb; }
  .history-wrap { max-height: 600px; overflow-y: auto; border: 1px solid #30363d; border-radius: 8px; }
  .history-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .history-table thead th {
    position: sticky; top: 0; background: #161b22; cursor: pointer; user-select: none;
    padding: 8px 6px; font-size: 10px; color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.5px; border-bottom: 1px solid #30363d; font-weight: 600; white-space: nowrap;
  }
  .history-table thead th:hover { color: #58a6ff; }
  .history-table thead th .sort-arrow { font-size: 8px; margin-left: 2px; opacity: 0.5; }
  .history-table thead th .sort-arrow.active { opacity: 1; color: #58a6ff; }
  .history-table tbody td { padding: 6px; border-bottom: 1px solid #21262d; font-family: 'SF Mono', monospace; white-space: nowrap; }
  .history-table tbody tr { cursor: pointer; transition: background 0.1s; }
  .history-table tbody tr:hover { background: #1c2128; }
  .history-table tbody tr.expanded { background: #1c2128; }
  .trade-detail-row td { padding: 0 !important; border-bottom: 1px solid #30363d; }
  .trade-detail { padding: 12px 16px; border-left: 3px solid #58a6ff; background: #0d1117; }
  .trade-detail-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; }
  .trade-detail-item { font-size: 11px; }
  .trade-detail-label { color: #8b949e; font-size: 10px; }
  .trade-detail-value { color: #f0f6fc; font-family: 'SF Mono', monospace; font-size: 12px; }
  .pnl-chart-wrap { margin-top: 16px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .pnl-chart-title { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; font-weight: 600; }

  /* Shadow Mode tab */
  .shadow-banner { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px 24px; margin-bottom: 20px; }
  .shadow-banner-title { font-size: 16px; font-weight: 700; color: #58a6ff; margin-bottom: 8px; letter-spacing: 0.5px; }
  .shadow-banner-text { font-size: 13px; color: #8b949e; line-height: 1.6; }
  .shadow-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }
  @media (max-width: 900px) { .shadow-cards { grid-template-columns: 1fr; } }
  .shadow-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; display: flex; flex-direction: column; }
  .shadow-card-sweep { border-top: 3px solid #58a6ff; }
  .shadow-card-mode1 { border-top: 3px solid #d29922; }
  .shadow-card-velocity { border-top: 3px solid #f85149; }
  .shadow-card-icon { font-size: 20px; margin-bottom: 4px; }
  .shadow-card-name { font-size: 14px; font-weight: 700; color: #f0f6fc; margin-bottom: 2px; }
  .shadow-card-tagline { font-size: 11px; color: #8b949e; font-style: italic; margin-bottom: 12px; }
  .shadow-card-desc { font-size: 12px; color: #c9d1d9; line-height: 1.5; margin-bottom: 14px; }
  .shadow-card-status { font-size: 12px; color: #8b949e; margin-bottom: 6px; }
  .shadow-card-status strong { color: #f0f6fc; font-family: 'SF Mono', monospace; }
  .shadow-card-latest { font-size: 11px; color: #c9d1d9; line-height: 1.5; background: #0d1117; border-radius: 6px; padding: 10px 12px; margin-top: auto; }
  .shadow-card-latest .lbl { font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
  .shadow-card-impact { font-size: 11px; color: #3fb950; margin-top: 8px; font-weight: 600; }
  .shadow-card-impact.neg { color: #f85149; }
  .shadow-timeline { margin-bottom: 20px; }
  .shadow-timeline-title { font-size: 13px; font-weight: 700; color: #f0f6fc; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
  .shadow-timeline-wrap { max-height: 500px; overflow-y: auto; border: 1px solid #30363d; border-radius: 8px; background: #161b22; }
  .shadow-tl-day { font-size: 11px; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 16px 6px; background: #0d1117; border-bottom: 1px solid #21262d; position: sticky; top: 0; z-index: 1; }
  .shadow-tl-row { display: flex; align-items: flex-start; padding: 10px 16px; border-bottom: 1px solid #21262d; transition: background 0.1s; }
  .shadow-tl-row:hover { background: #1c2128; }
  .shadow-tl-time { flex: 0 0 90px; font-size: 12px; font-family: 'SF Mono', monospace; color: #8b949e; padding-top: 1px; }
  .shadow-tl-icon { flex: 0 0 28px; font-size: 16px; }
  .shadow-tl-body { flex: 1; font-size: 12px; color: #c9d1d9; line-height: 1.5; }
  .shadow-tl-headline { color: #f0f6fc; font-weight: 600; }
  .shadow-tl-detail { color: #8b949e; font-size: 11px; margin-top: 2px; }
  .shadow-tl-result { font-weight: 600; }
  .shadow-tl-result.win { color: #3fb950; }
  .shadow-tl-result.loss { color: #f85149; }
  .shadow-statsbar { display: flex; gap: 16px; flex-wrap: wrap; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 20px; align-items: center; }
  .shadow-statsbar-item { font-size: 12px; color: #8b949e; }
  .shadow-statsbar-item strong { color: #f0f6fc; font-family: 'SF Mono', monospace; }
  .shadow-statsbar-sep { color: #30363d; }
  .shadow-empty { text-align: center; padding: 60px 20px; color: #8b949e; }
  .shadow-empty-icon { font-size: 48px; margin-bottom: 16px; opacity: 0.3; }
  .shadow-empty-text { font-size: 14px; }

  /* Bypass mode banner */
  .bypass-banner {
    background: #d2992215; border: 1px solid #d2992240; border-radius: 6px;
    padding: 8px 16px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
    font-size: 12px;
  }
  .bypass-banner-icon { font-size: 16px; }
  .bypass-banner-text { color: #d29922; font-weight: 600; }
  .bypass-banner-desc { color: #8b949e; }

  /* Market hours bar */
  .market-hours-bar {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 8px 24px; display: flex; align-items: center; gap: 16px;
    font-size: 12px;
  }
  .mh-status {
    font-weight: 700; font-size: 13px; padding: 3px 10px;
    border-radius: 4px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .mh-open { background: #238636; color: #fff; }
  .mh-closed { background: #484f58; color: #c9d1d9; }
  .mh-weekend { background: #6e40c9; color: #fff; }
  .mh-break { background: #d29922; color: #000; }
  .mh-detail { color: #8b949e; }
  .mh-next { color: #58a6ff; font-weight: 600; }
  .mh-schedule {
    position: relative; cursor: help; color: #58a6ff;
    text-decoration: underline dotted; margin-left: auto;
  }
  .mh-schedule-tip {
    display: none; position: absolute; top: 28px; right: 0;
    background: #1c2128; border: 1px solid #30363d; border-radius: 8px;
    padding: 14px 18px; font-size: 11px; color: #c9d1d9; z-index: 200;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5); min-width: 340px; line-height: 1.8;
    white-space: nowrap;
  }
  .mh-schedule:hover .mh-schedule-tip { display: block; }
  .mh-schedule-tip .mh-row { display: flex; justify-content: space-between; gap: 24px; }
  .mh-schedule-tip .mh-window { color: #f0f6fc; font-weight: 600; }
  .mh-schedule-tip .mh-time { color: #8b949e; }
  .mh-schedule-tip .mh-gate { font-size: 10px; padding: 1px 5px; border-radius: 3px; }
  .mh-gate-active { background: #23863630; color: #3fb950; }
  .mh-gate-blocked { background: #da363320; color: #f85149; }
  .mh-gate-data { background: #d2992220; color: #d29922; }

  /* Last updated indicator */
  .refresh-note { color: #484f58; font-size: 10px; text-align: center; padding: 16px 0; }
  #last-updated { color: #484f58; font-size: 10px; }

  @media (max-width: 900px) {
    .grid-3 { grid-template-columns: 1fr; }
    .grid-2 { grid-template-columns: 1fr; }
    .ticker-strip { gap: 16px; }
  }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <span class="logo">MANCINI BOT</span>
    <span class="connection-dot $dot_class" id="hdr-dot"></span>
    <span class="$connection_css" style="font-size:12px; font-weight:600" id="hdr-conn">$connection_status</span>
    <span class="session-badge $session_css" id="hdr-session">$session_label</span>
  </div>
  <div class="header-right">
    <span id="hdr-update">$last_update_pst</span>
    <span id="last-updated"></span>
    <span>$account_name</span>
  </div>
</div>

<!-- Market Hours Bar -->
<div class="market-hours-bar" id="market-hours-bar">
  <span class="mh-status $market_status_css" id="mh-status">$market_status_label</span>
  <span class="mh-detail" id="mh-detail">$market_status_detail</span>
  <span class="mh-next" id="mh-next">$market_next</span>
  <span class="mh-schedule">
    Full Schedule
    <div class="mh-schedule-tip">
      <div style="font-weight:700; color:#f0f6fc; margin-bottom:8px; font-size:12px;">ES/MES Futures &mdash; CME Globex</div>
      <div style="color:#8b949e; margin-bottom:6px; font-size:10px;">All times shown in ET (Eastern) &amp; PT (Pacific)</div>
      <div class="mh-row"><span class="mh-window">Daily Break</span><span class="mh-time">5:00-6:00 PM ET (2-3 PM PT)</span><span class="mh-gate mh-gate-blocked">CLOSED</span></div>
      <div class="mh-row"><span class="mh-window">Globex Evening</span><span class="mh-time">6:00-10:00 PM ET (3-7 PM PT)</span><span class="mh-gate mh-gate-data">BYPASS</span></div>
      <div class="mh-row"><span class="mh-window">Overnight (Asia)</span><span class="mh-time">10:00 PM-2:00 AM ET (7-11 PM PT)</span><span class="mh-gate mh-gate-active">ACTIVE</span></div>
      <div class="mh-row"><span class="mh-window">European Open</span><span class="mh-time">2:00-6:00 AM ET (11 PM-3 AM PT)</span><span class="mh-gate mh-gate-data">BYPASS</span></div>
      <div class="mh-row"><span class="mh-window">US Pre-Market</span><span class="mh-time">6:00-9:30 AM ET (3-6:30 AM PT)</span><span class="mh-gate mh-gate-active">ACTIVE</span></div>
      <div class="mh-row"><span class="mh-window">RTH Morning</span><span class="mh-time">9:30-11:00 AM ET (6:30-8 AM PT)</span><span class="mh-gate mh-gate-active">PRIME</span></div>
      <div class="mh-row"><span class="mh-window">RTH Midday</span><span class="mh-time">11:00 AM-1:00 PM ET (8-10 AM PT)</span><span class="mh-gate mh-gate-active">ACTIVE</span></div>
      <div class="mh-row"><span class="mh-window">Chop Zone</span><span class="mh-time">1:00-3:00 PM ET (10 AM-12 PM PT)</span><span class="mh-gate mh-gate-data">BYPASS</span></div>
      <div class="mh-row"><span class="mh-window">RTH Afternoon</span><span class="mh-time">3:00-4:50 PM ET (12-1:50 PM PT)</span><span class="mh-gate mh-gate-active">FB ONLY</span></div>
      <div class="mh-row"><span class="mh-window">EOD Settle</span><span class="mh-time">4:50-5:00 PM ET (1:50-2 PM PT)</span><span class="mh-gate mh-gate-blocked">FLATTEN</span></div>
      <div style="margin-top:8px; color:#484f58; font-size:10px; white-space:normal;">
        Weekend: Closed Friday 5 PM ET &rarr; Sunday 6 PM ET.<br>
        BYPASS = time gate bypassed, quality gates still enforced (R:R &ge; 1.0).
      </div>
    </div>
  </span>
</div>

<!-- Tabs -->
<div class="tab-bar">
  <button class="tab-btn active" data-tab="overview" onclick="switchTab('overview')">Overview</button>
  <button class="tab-btn" data-tab="strategy" onclick="switchTab('strategy')">Strategy</button>
  <button class="tab-btn" data-tab="chart" onclick="switchTab('chart')">Chart</button>
  <button class="tab-btn" data-tab="history" onclick="switchTab('history')">Trade History</button>
  <button class="tab-btn" data-tab="shadow" onclick="switchTab('shadow')">Shadow Mode</button>
  <button class="tab-btn" data-tab="report" onclick="switchTab('report')">Daily Report</button>
</div>

<!-- ==================== OVERVIEW TAB ==================== -->
<div id="tab-overview" class="tab-content active">
<div class="main">

$bypass_banner_html

<div class="ticker-strip">
  <div>
    <div class="ticker-symbol" id="tk-symbol">$symbol</div>
    <div class="ticker-price" id="tk-price">$last_price</div>
  </div>
  <div class="ticker-stat">
    <div class="ticker-stat-label">Session High</div>
    <div class="ticker-stat-value green" id="tk-high">$session_high</div>
  </div>
  <div class="ticker-stat">
    <div class="ticker-stat-label">Session Low</div>
    <div class="ticker-stat-value red" id="tk-low">$session_low</div>
  </div>
  <div class="ticker-stat">
    <div class="ticker-stat-label">Bars</div>
    <div class="ticker-stat-value" id="tk-bars">$bar_count</div>
  </div>
  <div class="ticker-stat">
    <div class="ticker-stat-label">Last Bar (PT)</div>
    <div class="ticker-stat-value" id="tk-bar-pst">$last_bar_pst</div>
  </div>
  <div class="ticker-stat">
    <div class="ticker-stat-label">Last Bar (ET)</div>
    <div class="ticker-stat-value dim" id="tk-bar-et">$last_bar_et</div>
  </div>
</div>

<!-- Market correlation data (VIX, SPY, 10Y) -->
<div id="market-data-strip" class="ticker-strip" style="margin-bottom:12px; display:none;">
  <div class="ticker-stat">
    <div class="ticker-stat-label">VIX</div>
    <div class="ticker-stat-value" id="mk-vix">--</div>
  </div>
  <div class="ticker-stat">
    <div class="ticker-stat-label">SPY</div>
    <div class="ticker-stat-value" id="mk-spy">--</div>
  </div>
  <div class="ticker-stat">
    <div class="ticker-stat-label">10Y Yield</div>
    <div class="ticker-stat-value" id="mk-yield">--</div>
  </div>
</div>

<!-- Interactive session timeline -->
<div class="card" style="margin-bottom:16px; padding:12px 16px;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
    <span class="card-header" style="margin:0; padding:0; border:none;">CME E-mini S&amp;P 500 Session Timeline</span>
    <span class="dim" style="font-size:12px;">$session_detail &mdash; hover for details</span>
  </div>
  <div class="timeline">
    <div class="tl-segment tl-maintenance">
      <div class="tl-tooltip">
        <div class="tl-tooltip-title">Daily Maintenance Break</div>
        <div class="tl-tooltip-time">5:00 PM - 6:00 PM ET (2-3 PM PT)</div>
        <div class="tl-tooltip-desc">
          CME daily settlement period. All futures markets closed for clearing and settlement.
          No trading possible. The engine resets daily state during this window.
        </div>
        <div class="tl-tooltip-status dim">CLOSED &mdash; CME settlement</div>
      </div>
    </div>
    <div class="tl-segment tl-evening">
      <span>GLBX EVE</span>
      <div class="tl-tooltip">
        <div class="tl-tooltip-title">Globex Evening Session</div>
        <div class="tl-tooltip-time">6:00 PM - 10:00 PM ET (3-7 PM PT)</div>
        <div class="tl-tooltip-desc">
          Post-settlement Globex session. Low volume, erratic moves.
          Historically unprofitable for the engine's patterns. Data is collected
          but production would skip entries.
        </div>
        <div class="tl-tooltip-status yellow">DATA COLLECTION &mdash; signals logged, not production-valid</div>
        <div class="tl-tooltip-mode">Collection mode: engine takes trades here for data</div>
      </div>
    </div>
    <div class="tl-segment tl-overnight">
      <span>OVERNIGHT</span>
      <div class="tl-tooltip">
        <div class="tl-tooltip-title">Globex Overnight (Asia Session)</div>
        <div class="tl-tooltip-time">10:00 PM - 2:00 AM ET (7-11 PM PT)</div>
        <div class="tl-tooltip-desc">
          Asian market overlap. Moderate liquidity. Level reclaim signals
          can fire here if levels were established during RTH. Often sets
          the tone for the next day's direction.
        </div>
        <div class="tl-tooltip-status green">ACTIVE &mdash; All patterns</div>
      </div>
    </div>
    <div class="tl-segment tl-euro">
      <span>EURO</span>
      <div class="tl-tooltip">
        <div class="tl-tooltip-title">European Session (London Open)</div>
        <div class="tl-tooltip-time">2:00 AM - 6:00 AM ET (11 PM - 3 AM PT)</div>
        <div class="tl-tooltip-desc">
          European institutional flow creates unpredictable moves that frequently
          sweep US levels. Data collected but historically poor for ES setups.
        </div>
        <div class="tl-tooltip-status yellow">DATA COLLECTION &mdash; signals logged, not production-valid</div>
        <div class="tl-tooltip-mode">Collection mode: engine takes trades here for data</div>
      </div>
    </div>
    <div class="tl-segment tl-premarket">
      <span>US PRE-MKT</span>
      <div class="tl-tooltip">
        <div class="tl-tooltip-title">US Pre-Market / Pre-Open</div>
        <div class="tl-tooltip-time">6:00 AM - 9:30 AM ET (3-6:30 AM PT)</div>
        <div class="tl-tooltip-desc">
          US economic data releases, increasing volume as US traders come online.
          Levels from prior RTH start being tested.
          FB and LR patterns can fire with moderate conviction.
        </div>
        <div class="tl-tooltip-status green">ACTIVE &mdash; All patterns</div>
      </div>
    </div>
    <div class="tl-segment tl-morning">
      <span>RTH AM</span>
      <div class="tl-tooltip">
        <div class="tl-tooltip-title">RTH Morning Session (Prime)</div>
        <div class="tl-tooltip-time">9:30 AM - 11:00 AM ET (6:30-8 AM PT)</div>
        <div class="tl-tooltip-desc">
          <strong>Highest conviction window.</strong> Regular Trading Hours open brings maximum
          volume and institutional participation. Failed breakdowns here have the
          strongest follow-through. Most profitable time for the Mancini method.
        </div>
        <div class="tl-tooltip-status green">PRIME &mdash; Best signals</div>
      </div>
    </div>
    <div class="tl-segment tl-midday">
      <span>RTH MID</span>
      <div class="tl-tooltip">
        <div class="tl-tooltip-title">RTH Midday Session</div>
        <div class="tl-tooltip-time">11:00 AM - 1:00 PM ET (8-10 AM PT)</div>
        <div class="tl-tooltip-desc">
          Transitional period. Volume tapers off from the morning rush.
          Level reclaims still work well. Watch for failed breakdowns
          at levels established during the morning session.
        </div>
        <div class="tl-tooltip-status green">ACTIVE &mdash; All patterns</div>
      </div>
    </div>
    <div class="tl-segment tl-chop">
      <span>RTH CHOP</span>
      <div class="tl-tooltip">
        <div class="tl-tooltip-title">RTH Afternoon (Chop Zone)</div>
        <div class="tl-tooltip-time">1:00 PM - 3:00 PM ET (10 AM - 12 PM PT)</div>
        <div class="tl-tooltip-desc">
          Low-volume choppy price action. Signals frequently whipsaw.
          Mancini himself warns against trading this window. Data collected
          but production would skip entries.
        </div>
        <div class="tl-tooltip-status yellow">DATA COLLECTION &mdash; signals logged, not production-valid</div>
        <div class="tl-tooltip-mode">Collection mode: engine takes trades here for data</div>
      </div>
    </div>
    <div class="tl-segment tl-afternoon">
      <span>RTH PM</span>
      <div class="tl-tooltip">
        <div class="tl-tooltip-title">RTH Late Day / Power Hour</div>
        <div class="tl-tooltip-time">3:00 PM - 4:00 PM ET (12-1 PM PT)</div>
        <div class="tl-tooltip-desc">
          "Power hour" &mdash; volume picks up for the RTH close. Only Failed Breakdown
          signals are allowed in production because they have the strongest edge
          in afternoon reversals. Level Reclaims filtered out (lower PM win rate).
        </div>
        <div class="tl-tooltip-status yellow">FB ONLY &mdash; Highest conviction pattern only</div>
      </div>
    </div>
    <div class="tl-segment tl-settle">
      <div class="tl-tooltip">
        <div class="tl-tooltip-title">RTH Close / Settlement</div>
        <div class="tl-tooltip-time">4:00 PM - 5:00 PM ET (1-2 PM PT)</div>
        <div class="tl-tooltip-desc">
          All open positions are flattened before the daily maintenance break.
          No new entries. Protects against overnight gap risk.
        </div>
        <div class="tl-tooltip-status red">FLATTEN &mdash; Close all positions</div>
      </div>
    </div>
  </div>
  <div class="tl-labels">
    <span>5PM</span>
    <span>6PM ET</span>
    <span>10PM</span>
    <span>2AM</span>
    <span>6AM</span>
    <span>9:30</span>
    <span>1PM</span>
    <span>3PM</span>
    <span>5PM</span>
  </div>
</div>

<!-- Top row: PnL + Position + Regime -->
<div class="grid-3">
  <div class="card">
    <div class="card-header">Today's P&amp;L</div>
    <div class="pnl-big $pnl_class" id="pnl-value">$daily_pnl</div>
    <div class="pnl-sub" id="pnl-sub">$trades_today trades today $status_badge</div>
    <div style="margin-top:12px;">
      <div class="stat-row"><span class="stat-label">Winners</span><span class="stat-value green" id="stat-winners">$winners</span></div>
      <div class="stat-row"><span class="stat-label">Losers</span><span class="stat-value red" id="stat-losers">$losers</span></div>
      <div class="stat-row"><span class="stat-label">Balance</span><span class="stat-value" id="stat-balance">$account_balance</span></div>
      <div class="stat-row"><span class="stat-label">Equity</span><span class="stat-value" id="stat-equity">$account_equity</span></div>
      <div class="stat-row"><span class="stat-label">All-Time Trades</span><span class="stat-value blue" id="stat-total">$total_logged</span></div>
    </div>
  </div>

  <div class="card">
    <div class="card-header">Current Position</div>
    <div id="position-content">$position_html</div>
  </div>

  <div class="card">
    <div class="card-header">Regime &amp; Config</div>
    <div class="stat-row">
      <span class="stat-label">Market Regime</span>
      <span class="stat-value $regime_css regime-clickable" onclick="showRegimeModal()" id="stat-regime">$regime</span>
    </div>
    <div class="stat-row"><span class="stat-label">Daily Bars Loaded</span><span class="stat-value blue" id="stat-dbars">$regime_daily_bars</span></div>
    <div class="stat-row"><span class="stat-label">Longs Enabled</span><span class="stat-value $longs_css" id="stat-longs">$longs_enabled</span></div>
    <div class="stat-row"><span class="stat-label">Shorts Enabled</span><span class="stat-value $shorts_css" id="stat-shorts">$shorts_enabled</span></div>
    <div class="stat-row"><span class="stat-label">Session Date</span><span class="stat-value" id="stat-date">$session_date</span></div>
    <div class="stat-row"><span class="stat-label">Trading Active</span><span class="stat-value $trading_css" id="stat-trading">$trading_active</span></div>
  </div>
</div>

<!-- Levels + Trades -->
<div class="grid-2">
  <div class="card">
    <div class="card-header">Active Levels (nearest first)</div>
    <div class="info-blurb">
      Price levels where the engine expects support or resistance. Detected from prior day
      highs/lows, swing points, and price clusters. Levels within 1 pt are merged.
    </div>
    <div id="levels-content">$levels_html</div>
  </div>
  <div class="card">
    <div class="card-header">Trade History (Today)</div>
    <div id="trades-content">$trades_html</div>
  </div>
</div>

<!-- Near Misses + Phantoms + Log -->
<div class="grid-2">
  <div class="card">
    <div class="card-header">
      <span class="near-miss-toggle" onclick="toggleNearMisses()">
        <span class="arrow" id="nm-arrow">&#9654;</span>
        Near Misses &mdash; setups that almost triggered
      </span>
    </div>
    <div id="near-misses-content" style="display:none;">$near_misses_html</div>
  </div>
  <div class="card">
    <div class="card-header">Phantom Signals (rejected)</div>
    <div id="phantoms-content">$phantoms_html</div>
  </div>
</div>

<!-- Mancini Substack Comparison -->
<div class="card" style="margin-top:16px;">
  <div class="card-header">
    <span class="near-miss-toggle" onclick="toggleSubstack()">
      <span class="arrow" id="ss-arrow">&#9654;</span>
      Mancini Substack &mdash; latest post highlights &amp; level comparison
    </span>
  </div>
  <div id="substack-content" style="display:none;">$substack_html</div>
</div>

<!-- Retrospective Analysis -->
<div class="card" style="margin-top:16px;">
  <div class="card-header">
    <span class="near-miss-toggle" onclick="toggleRetro()">
      <span class="arrow" id="retro-arrow">&#9654;</span>
      Retrospective &mdash; how did yesterday&rsquo;s levels actually play out?
    </span>
  </div>
  <div id="retro-content" style="display:none;">$retrospective_html</div>
</div>

<div class="card" style="margin-top:16px;">
  <div class="card-header">Recent Log</div>
  <div class="log-viewer" id="log-content">$log_html</div>
</div>

</div>
</div>

<!-- ==================== STRATEGY TAB ==================== -->
<div id="tab-strategy" class="tab-content">
<div class="main">

<div class="strategy-hero">
  <h2>Mancini Method Engine</h2>
  <p>
    Automated ES/MES futures day trading engine based on the Mancini Method
    (David Mancini, Substack). The strategy uses <strong>pure price action</strong> &mdash;
    zero indicators &mdash; to identify key support/resistance levels and trade
    bounces (FB Long) and confirmed breakdowns (BD Short) at those levels.
    Levels are detected in real time using swing point analysis. Entries use
    bracket orders (stop + target) through Interactive Brokers.
    <strong>Mancini thesis: &ldquo;The bigger the sell, the bigger the squeeze.&rdquo;</strong>
  </p>
  <div style="margin-top:12px; display:flex; gap:16px; flex-wrap:wrap;">
    <div class="pattern-stat"><div class="pattern-stat-val blue">2,279</div><div class="pattern-stat-lbl">Trades (5yr)</div></div>
    <div class="pattern-stat"><div class="pattern-stat-val">0.91</div><div class="pattern-stat-lbl">Profit Factor</div></div>
    <div class="pattern-stat"><div class="pattern-stat-val red">-2,232</div><div class="pattern-stat-lbl">Total Pts (all hrs)</div></div>
    <div class="pattern-stat"><div class="pattern-stat-val">41.3%</div><div class="pattern-stat-lbl">Win Rate</div></div>
  </div>
  <div style="margin-top:8px; display:flex; gap:16px; flex-wrap:wrap;">
    <div class="pattern-stat"><div class="pattern-stat-val green">+4,432</div><div class="pattern-stat-lbl">RTH Only (9:30-4)</div></div>
    <div class="pattern-stat"><div class="pattern-stat-val red">-6,689</div><div class="pattern-stat-lbl">Late Night 12-4AM</div></div>
    <div class="pattern-stat"><div class="pattern-stat-val green">55%</div><div class="pattern-stat-lbl">Morning WR</div></div>
    <div class="pattern-stat"><div class="pattern-stat-val green">63%</div><div class="pattern-stat-lbl">Afternoon WR</div></div>
  </div>
  <div class="backtest-note">
    5-year full-session backtest (Jan 2021 - Feb 2026), Optuna v2 params, 1.69M bars.
    Currently running in <strong>data collection mode</strong> (all time gates bypassed, quality gates enforced).
    Key validated finding: Late night LR trades (12-4 AM) account for nearly all losses.
    RTH hours are profitable. Deep sweeps get 40-bar recovery window (Mancini: bigger sell = bigger squeeze).
  </div>
</div>

<h3 style="color:#8b949e; font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Long Patterns</h3>
<div class="grid-2" style="margin-bottom:24px;">
  <div class="pattern-card $fb_card_class">
    <span class="pattern-badge $fb_badge_class">$fb_status</span>
    <div class="pattern-name">Failed Breakdown (FB)</div>
    <div class="pattern-type green">LONG</div>
    <div class="pattern-desc">
      Price sweeps below a key support level (PDL, MHL, Cluster), then recovers above.
      Four entry paths: <strong>Elevator FB</strong> (fast selloff + sweep + bounce),
      <strong>Level Sweep FB</strong> (3+ bars below, then recovery &mdash; no elevator needed),
      <strong>Double Dip</strong> (re-sweep after stop-out), and <strong>Deep Sell Recovery</strong> (crash bottoms).
      Confirmation: acceptance (7 bars above level) or non-acceptance (fast 5+ pt recovery held 3 bars).
      Deep sweeps get 40-bar recovery window before abort. Sweep depth unlimited.
    </div>
    <div class="flow-steps">
      <span class="flow-step">Support level identified</span><span class="flow-arrow">&rarr;</span>
      <span class="flow-step">Sweep &ge;2 pts below</span><span class="flow-arrow">&rarr;</span>
      <span class="flow-step">Recovery + acceptance/non-acceptance</span><span class="flow-arrow">&rarr;</span>
      <span class="flow-step">LONG entry</span>
    </div>
    <div class="pattern-stats">
      <div class="pattern-stat"><div class="pattern-stat-val green">49.4%</div><div class="pattern-stat-lbl">Win Rate</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val">1,315</div><div class="pattern-stat-lbl">Trades (5yr)</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val red">-691</div><div class="pattern-stat-lbl">Total Pts (all hrs)</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val green">+2,855</div><div class="pattern-stat-lbl">Morning RTH</div></div>
    </div>
    <div class="backtest-note">
      Stop: sweep_low - 6.0 pts (deep sweeps &gt;20 pts use level - 6.0 to cap risk).
      Max hold: 14 bars. True breakdown abort: 40 bars (was 20). Sweep 5-10 pts = 63% WR (sweet spot).
      Targets: first resistance &ge;8 pts away, capped at 30 pts.
    </div>
  </div>

  <div class="pattern-card $lr_card_class">
    <span class="pattern-badge $lr_badge_class">$lr_status</span>
    <div class="pattern-name">Level Reclaim (LR)</div>
    <div class="pattern-type green">LONG</div>
    <div class="pattern-desc">
      Price dips below a HORIZONTAL_SR level (4+ touches), then closes back above on the same bar.
      Confirmation via acceptance (7 bars above) or non-acceptance (3 bars &ge;5 pts above).
      Uses horizontal S/R levels only (not PDL/MHL/Cluster). <strong>Weakest pattern historically</strong>
      &mdash; 29% WR, largest source of losses. Late night LR at HORIZONTAL_SR = 7% WR, -8,713 pts (black hole).
      LR only fires when FB is idle (FB has priority).
    </div>
    <div class="flow-steps">
      <span class="flow-step">HORIZONTAL_SR (4+ touches)</span><span class="flow-arrow">&rarr;</span>
      <span class="flow-step">Dip below + close above</span><span class="flow-arrow">&rarr;</span>
      <span class="flow-step">Hold 7 bars above</span><span class="flow-arrow">&rarr;</span>
      <span class="flow-step">LONG entry</span>
    </div>
    <div class="pattern-stats">
      <div class="pattern-stat"><div class="pattern-stat-val red">29.2%</div><div class="pattern-stat-lbl">Win Rate</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val">870</div><div class="pattern-stat-lbl">Trades (5yr)</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val red">-1,549</div><div class="pattern-stat-lbl">Total Pts</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val green">+928</div><div class="pattern-stat-lbl">RTH Midday+PM</div></div>
    </div>
    <div class="backtest-note">
      Stop: level - 4.0 pts. Avg win +49 pts, avg loss -23 pts (wins big but rarely).
      Late night 12-4AM LR = 7% WR, -8,713 pts. Consider disabling LR outside RTH or entirely.
    </div>
  </div>
</div>

<h3 style="color:#8b949e; font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Short Patterns</h3>
<div class="grid-2" style="margin-bottom:24px;">
  <div class="pattern-card $bd_card_class">
    <span class="pattern-badge $bd_badge_class">$bd_status</span>
    <div class="pattern-name">Breakdown Short (BD)</div>
    <div class="pattern-type red">SHORT</div>
    <div class="pattern-desc">
      Price breaks below a <strong>major</strong> support level (PRIOR_DAY_LOW or MULTI_HOUR_LOW only &mdash;
      CLUSTER_LOW excluded as noise) and <em>holds below</em> for 21 consecutive bars.
      Unlike FB where the break fails, here it succeeds &mdash; the level flips to resistance.
      Entry short at confirmation. Fires at ALL hours (exempt from chop zone).
      <strong>BD @ PRIOR_DAY_LOW = validated edge (+199 pts, 44% WR, 75 trades).</strong>
      BD @ MULTI_HOUR_LOW = net loser (-191 pts, 19 trades).
    </div>
    <div class="flow-steps">
      <span class="flow-step">Major support (PDL/MHL)</span><span class="flow-arrow">&rarr;</span>
      <span class="flow-step">Break &ge;1 pt + close below</span><span class="flow-arrow">&rarr;</span>
      <span class="flow-step">Hold below 21 bars</span><span class="flow-arrow">&rarr;</span>
      <span class="flow-step">SHORT entry (if depth &le;17 pts)</span>
    </div>
    <div class="pattern-stats">
      <div class="pattern-stat"><div class="pattern-stat-val">41.5%</div><div class="pattern-stat-lbl">Win Rate</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val">94</div><div class="pattern-stat-lbl">Trades (5yr)</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val green">+8</div><div class="pattern-stat-lbl">Total Pts</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val green">+199</div><div class="pattern-stat-lbl">@ PDL only</div></div>
    </div>
    <div class="backtest-note">
      Stop: level + 6.0 pts (above broken level). Timeout: 35 bars. Max depth: 17 pts (reject late entries).
      Regime filter OFF (data collection). R:R 1.0-1.5 = best bucket (71% WR, 14 trades).
      Avg win +49 pts, avg loss -34 pts. Live winner: +29 pts BD Short @ PDL 6750 on Mar 11.
    </div>
  </div>

  <div class="pattern-card disabled">
    <span class="pattern-badge badge-disabled">DISABLED</span>
    <div class="pattern-name">Backtest Short (BT)</div>
    <div class="pattern-type red">SHORT</div>
    <div class="pattern-desc">
      After a support level is broken, price retests the now-resistance level
      from below and gets rejected. Entry short on the rejection. Currently
      <strong>disabled</strong> due to catastrophic backtest results (3.4% win rate,
      -627 pts). The pattern logic exists in code but is gated off.
    </div>
    <div class="pattern-stats">
      <div class="pattern-stat"><div class="pattern-stat-val red">3.4%</div><div class="pattern-stat-lbl">Win Rate</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val red">0.07</div><div class="pattern-stat-lbl">Profit Factor</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val red">-627</div><div class="pattern-stat-lbl">Total Pts</div></div>
      <div class="pattern-stat"><div class="pattern-stat-val">58</div><div class="pattern-stat-lbl">Trades</div></div>
    </div>
    <div class="backtest-note">Permanently disabled. allow_backtest_short=False in config.</div>
  </div>
</div>

<h3 style="color:#8b949e; font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Exit Strategy (Mancini 50/50 on 2 MES)</h3>
<div class="grid-2" style="margin-bottom:24px;">
  <div class="card">
    <div class="card-header">Position Management</div>
    <div class="stat-row"><span class="stat-label">Contracts</span><span class="stat-value">2 MES</span></div>
    <div class="stat-row"><span class="stat-label">T1 Exit</span><span class="stat-value">1 contract (50%)</span></div>
    <div class="stat-row"><span class="stat-label">Runner</span><span class="stat-value">1 contract (50%)</span></div>
    <div class="stat-row"><span class="stat-label">After T1 Stop</span><span class="stat-value">Breakeven - 3 pts</span></div>
    <div class="stat-row"><span class="stat-label">Runner Trail</span><span class="stat-value">Prior day low - 1 pt</span></div>
    <div class="stat-row"><span class="stat-label">Runner Duration</span><span class="stat-value green">Multi-day (never flatten on restart)</span></div>
    <div class="stat-row"><span class="stat-label">FB Time Exit</span><span class="stat-value">14 bars max hold</span></div>
    <div class="stat-row"><span class="stat-label">EOD Flatten</span><span class="stat-value">3:58 PM ET (runners exempt)</span></div>
  </div>
  <div class="card">
    <div class="card-header">Exit Priority (each bar)</div>
    <div style="padding:8px 0; color:#8b949e; font-size:12px; line-height:1.8;">
      <div><span style="color:#f85149;font-weight:600;">1.</span> <strong>Stop loss</strong> &mdash; highest priority. Long: low &le; stop. Short: high &ge; stop.</div>
      <div><span style="color:#d29922;font-weight:600;">2.</span> <strong>T1 target</strong> &mdash; exit 1 of 2 contracts. Move stop to breakeven - 3 pts.</div>
      <div><span style="color:#3fb950;font-weight:600;">3.</span> <strong>T2 target</strong> &mdash; if reached, runner trails under prior day low.</div>
      <div><span style="color:#58a6ff;font-weight:600;">4.</span> <strong>Runner trail</strong> &mdash; updated at EOD. Carries overnight/multi-day until prior day low lost.</div>
      <div style="margin-top:8px;border-top:1px solid #21262d;padding-top:8px;">
        <strong>Mancini:</strong> &ldquo;75% at T1, 15% at T2, 10% runner.&rdquo;
        With 2 MES: 50/50 split (closest approximation). Runners documented +125 to +280 pts.
      </div>
    </div>
  </div>
</div>

<h3 style="color:#8b949e; font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Validated Findings (5yr Hypothesis Testing)</h3>
<div class="grid-2" style="margin-bottom:24px;">
  <div class="card">
    <div class="card-header" style="color:#3fb950;">Confirmed Edge (large sample)</div>
    <div style="padding:8px 0; color:#8b949e; font-size:12px; line-height:1.8;">
      <div><strong>Time of day is everything</strong> &mdash; Morning RTH +4,866 pts, Afternoon +927 pts. Late night LR = -8,713 pts.</div>
      <div><strong>FB &gt; LR</strong> &mdash; FB 49% WR vs LR 29% WR on 2,185 trades.</div>
      <div><strong>R:R 1.0-1.5 sweet spot</strong> &mdash; +232 pts, 53% WR (246 trades).</div>
      <div><strong>BD Short @ PDL = real edge</strong> &mdash; +199 pts, 44% WR (75 trades).</div>
      <div><strong>Sweep 5-10 pts = strongest FB</strong> &mdash; 63% WR (40 trades).</div>
      <div><strong>Non-acceptance slightly better</strong> &mdash; 48% vs 41% WR.</div>
    </div>
  </div>
  <div class="card">
    <div class="card-header" style="color:#f85149;">Overfit (reversed historically)</div>
    <div style="padding:8px 0; color:#8b949e; font-size:12px; line-height:1.8;">
      <div><strong>PRIOR_DAY_LOW is NOT best level</strong> &mdash; CLUSTER_LOW is the workhorse (1,285 trades, 50% WR).</div>
      <div><strong>&ldquo;One trade per level&rdquo; rule wrong</strong> &mdash; 2nd trade at level = 46% WR vs 41% for 1st.</div>
      <div><strong>R:R 0.5-1.0 &ldquo;paradox&rdquo; was noise</strong> &mdash; 5 live trades, not replicable on 816 historical.</div>
      <div><strong>15-pt stop cap wrong</strong> &mdash; 15-20 pt stops = 51% WR (best bucket).</div>
      <div><strong>BD Short R:R &gt; 1.5 wrong</strong> &mdash; R:R 1.0-1.5 = 71% WR (best, 14 trades).</div>
    </div>
  </div>
</div>

<h3 style="color:#8b949e; font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Risk Management &amp; Configuration</h3>
<div class="grid-3">
  <div class="card">
    <div class="card-header">Risk &amp; Quality Gates (Optuna v2)</div>
    <div class="stat-row"><span class="stat-label">Max Trades/Day</span><span class="stat-value">999 (data collection)</span></div>
    <div class="stat-row"><span class="stat-label">Min R:R Ratio</span><span class="stat-value">0.8</span></div>
    <div class="stat-row"><span class="stat-label">Max Stop Distance</span><span class="stat-value">20.0 pts</span></div>
    <div class="stat-row"><span class="stat-label">FB Stop Buffer</span><span class="stat-value">6.0 pts</span></div>
    <div class="stat-row"><span class="stat-label">LR Stop Buffer</span><span class="stat-value">4.0 pts</span></div>
    <div class="stat-row"><span class="stat-label">BD Stop Buffer</span><span class="stat-value">6.0 pts</span></div>
    <div class="stat-row"><span class="stat-label">Max Target Dist</span><span class="stat-value">30.0 pts</span></div>
    <div class="stat-row"><span class="stat-label">Max Sweep Depth</span><span class="stat-value">999 (unlimited)</span></div>
    <div class="stat-row"><span class="stat-label">Deep Sweep Level Stop</span><span class="stat-value">&gt;20 pts: use level stop</span></div>
    <div class="stat-row"><span class="stat-label">True BD Abort</span><span class="stat-value">40 bars (was 20)</span></div>
    <div class="stat-row"><span class="stat-label">Signal Cooldown</span><span class="stat-value">15 bars</span></div>
  </div>
  <div class="card">
    <div class="card-header">Session Windows (Bypass Mode)</div>
    <div class="stat-row"><span class="stat-label">Mode</span><span class="stat-value yellow">DATA COLLECTION</span></div>
    <div class="stat-row"><span class="stat-label">All Time Gates</span><span class="stat-value yellow">Bypassed (logged)</span></div>
    <div class="stat-row"><span class="stat-label">Quality Gates</span><span class="stat-value green">Enforced</span></div>
    <div class="stat-row"><span class="stat-label">Loss Limits</span><span class="stat-value yellow">Bypassed</span></div>
    <div class="stat-row" style="margin-top:8px;border-top:1px solid #30363d;padding-top:8px;"><span class="stat-label" style="color:#58a6ff;">5yr Validated Edge</span><span class="stat-value"></span></div>
    <div class="stat-row"><span class="stat-label">Morning 9-12 ET</span><span class="stat-value green">+2,855 pts (55% WR)</span></div>
    <div class="stat-row"><span class="stat-label">Afternoon 2-4 ET</span><span class="stat-value green">+927 pts (58% WR)</span></div>
    <div class="stat-row"><span class="stat-label">Midday 12-2 ET</span><span class="stat-value green">+650 pts (63% WR)</span></div>
    <div class="stat-row"><span class="stat-label">Late Night 12-4 AM</span><span class="stat-value red">-6,689 pts (22% WR)</span></div>
    <div class="stat-row"><span class="stat-label">Evening 6-11 PM</span><span class="stat-value green">+452 pts (77% WR)</span></div>
  </div>
  <div class="card">
    <div class="card-header">Regime Filter <span class="regime-clickable" onclick="showRegimeModal()" style="font-size:10px">[explain]</span></div>
    <div class="stat-row"><span class="stat-label">Status</span><span class="stat-value yellow">DISABLED (data collection)</span></div>
    <div class="stat-row"><span class="stat-label">Mode</span><span class="stat-value">EMA Slope</span></div>
    <div class="stat-row"><span class="stat-label">EMA Span</span><span class="stat-value">30 days</span></div>
    <div class="stat-row"><span class="stat-label">Slope Lookback</span><span class="stat-value">10 days</span></div>
    <div class="stat-row"><span class="stat-label">Threshold</span><span class="stat-value">ATR &times; 0.325</span></div>
    <div class="stat-row"><span class="stat-label">Daily Bars</span><span class="stat-value blue">$regime_daily_bars</span></div>
    <div class="stat-row"><span class="stat-label">Current Regime</span><span class="stat-value $regime_css regime-clickable" onclick="showRegimeModal()">$regime</span></div>
    <div class="stat-row"><span class="stat-label">Longs</span><span class="stat-value $longs_css">$longs_enabled</span></div>
    <div class="stat-row"><span class="stat-label">Shorts</span><span class="stat-value $shorts_css">$shorts_enabled</span></div>
  </div>
</div>

</div>
</div>

<!-- ==================== CHART TAB ==================== -->
<div id="tab-chart" class="tab-content">
<div class="main">
  <div id="chart-container"></div>
  <div class="card">
    <div class="card-header">Active Levels (shown on chart)</div>
    $levels_html
  </div>
</div>
</div>

<!-- ==================== TRADE HISTORY TAB ==================== -->
<div id="tab-history" class="tab-content">
<div class="main">

<div id="history-summary" class="history-summary">
  <div class="history-stat"><div class="history-stat-val dim" id="hs-total">--</div><div class="history-stat-lbl">Total Trades</div></div>
  <div class="history-stat"><div class="history-stat-val dim" id="hs-winrate">--</div><div class="history-stat-lbl">Win Rate</div></div>
  <div class="history-stat"><div class="history-stat-val dim" id="hs-pnl">--</div><div class="history-stat-lbl">Total PnL (pts)</div></div>
  <div class="history-stat"><div class="history-stat-val dim" id="hs-pf">--</div><div class="history-stat-lbl">Profit Factor</div></div>
  <div class="history-stat"><div class="history-stat-val dim" id="hs-best">--</div><div class="history-stat-lbl">Best Trade</div></div>
  <div class="history-stat"><div class="history-stat-val dim" id="hs-worst">--</div><div class="history-stat-lbl">Worst Trade</div></div>
  <div class="history-stat"><div class="history-stat-val dim" id="hs-avg">--</div><div class="history-stat-lbl">Avg PnL</div></div>
</div>

<div class="history-filters">
  <div id="date-pills"></div>
  <label class="filter-toggle">
    <input type="checkbox" id="prod-filter" onchange="toggleProdFilter()">
    Production only
  </label>
</div>

<div class="history-wrap">
  <table class="history-table">
    <thead>
      <tr>
        <th onclick="sortHistory('date')">Date <span class="sort-arrow" id="sa-date"></span></th>
        <th onclick="sortHistory('entry_time')">Entry <span class="sort-arrow" id="sa-entry_time"></span></th>
        <th onclick="sortHistory('exit_time')">Exit <span class="sort-arrow" id="sa-exit_time"></span></th>
        <th onclick="sortHistory('direction')">Dir <span class="sort-arrow" id="sa-direction"></span></th>
        <th onclick="sortHistory('pattern')">Pattern <span class="sort-arrow" id="sa-pattern"></span></th>
        <th onclick="sortHistory('entry_price')">Entry <span class="sort-arrow" id="sa-entry_price"></span></th>
        <th onclick="sortHistory('stop')">Stop <span class="sort-arrow" id="sa-stop"></span></th>
        <th onclick="sortHistory('target_1')">T1 <span class="sort-arrow" id="sa-target_1"></span></th>
        <th onclick="sortHistory('exit_price')">Exit <span class="sort-arrow" id="sa-exit_price"></span></th>
        <th onclick="sortHistory('pnl_pts')">PnL <span class="sort-arrow" id="sa-pnl_pts"></span></th>
        <th onclick="sortHistory('rr_ratio')">R:R <span class="sort-arrow" id="sa-rr_ratio"></span></th>
        <th onclick="sortHistory('exit_reason')">Reason <span class="sort-arrow" id="sa-exit_reason"></span></th>
        <th>Mode</th>
      </tr>
    </thead>
    <tbody id="history-tbody"></tbody>
  </table>
</div>

<div class="pnl-chart-wrap">
  <div class="pnl-chart-title">Cumulative PnL (pts)</div>
  <svg id="pnl-chart" width="100%" height="120" preserveAspectRatio="none"></svg>
</div>

</div>
</div>

<!-- ==================== SHADOW MODE TAB ==================== -->
<div id="tab-shadow" class="tab-content">
<div class="main">

<!-- Header Banner -->
<div class="shadow-banner">
  <div class="shadow-banner-title">SHADOW MODE</div>
  <div class="shadow-banner-text">
    Testing 3 new features without risking real trades.<br>
    These features run silently and log what they <strong style="color:#f0f6fc;">would have done</strong> differently.
    When we're confident they improve performance, we'll activate them for real.
  </div>
</div>

<!-- Three Feature Cards -->
<div class="shadow-cards" id="shadow-cards">
  <div class="shadow-card shadow-card-sweep">
    <div class="shadow-card-icon">&#x1F4CF;</div>
    <div class="shadow-card-name">Sweep Depth Sizing</div>
    <div class="shadow-card-tagline">"Bigger sweep = bigger position"</div>
    <div class="shadow-card-desc">Sizes your position based on how deep price sweeps below the level. Shallow sweep = small size, deep sweep = full size.</div>
    <div class="shadow-card-status">Status: <strong id="sc-sweep-status">--</strong></div>
    <div class="shadow-card-latest" id="sc-sweep-latest"><div class="lbl">Latest</div>Waiting for events...</div>
    <div class="shadow-card-impact" id="sc-sweep-impact"></div>
  </div>
  <div class="shadow-card shadow-card-mode1">
    <div class="shadow-card-icon">&#x1F534;</div>
    <div class="shadow-card-name">Mode 1 Red Detection</div>
    <div class="shadow-card-tagline">"Detect trend days that destroy FB Longs"</div>
    <div class="shadow-card-desc">When 3+ support levels break and stay broken, it's a trend day &mdash; not a bounce day. Cuts size to 25%.</div>
    <div class="shadow-card-status">Status: <strong id="sc-mode1-status">--</strong></div>
    <div class="shadow-card-latest" id="sc-mode1-latest"><div class="lbl">Latest</div>Waiting for events...</div>
    <div class="shadow-card-impact" id="sc-mode1-impact"></div>
  </div>
  <div class="shadow-card shadow-card-velocity">
    <div class="shadow-card-icon">&#x26A1;</div>
    <div class="shadow-card-name">Velocity Breakdown Short</div>
    <div class="shadow-card-tagline">"Catch single-bar news-driven breaks"</div>
    <div class="shadow-card-desc">When a major level breaks on one bar with 3x+ volume, enters a short at 25% size. Catches moves the normal BD Short detector is too slow for.</div>
    <div class="shadow-card-status">Status: <strong id="sc-vshort-status">--</strong></div>
    <div class="shadow-card-latest" id="sc-vshort-latest"><div class="lbl">Latest</div>Waiting for events...</div>
    <div class="shadow-card-impact" id="sc-vshort-impact"></div>
  </div>
</div>

<!-- Event Timeline -->
<div class="shadow-timeline">
  <div class="shadow-timeline-title">Event Timeline</div>
  <div class="shadow-timeline-wrap" id="shadow-timeline">
    <div class="shadow-empty"><div class="shadow-empty-text">Loading shadow events...</div></div>
  </div>
</div>

<!-- Summary Stats Bar -->
<div class="shadow-statsbar" id="shadow-statsbar">
  <div class="shadow-statsbar-item">Shadow Features Active: <strong>3/3</strong></div>
  <div class="shadow-statsbar-sep">|</div>
  <div class="shadow-statsbar-item">Events Today: <strong id="ss-total">--</strong></div>
  <div class="shadow-statsbar-sep">|</div>
  <div class="shadow-statsbar-item">Potential Savings: <strong id="ss-savings">--</strong></div>
  <div class="shadow-statsbar-sep">|</div>
  <div class="shadow-statsbar-item">Velocity Signals: <strong id="ss-vsignals">--</strong></div>
</div>

</div>
</div>

<!-- ==================== DAILY REPORT TAB ==================== -->
<div id="tab-report" class="tab-content">
<div class="main">
  <div id="report-content" style="color:#8b949e;text-align:center;padding:40px;">Loading daily reports...</div>
</div>
</div>

<!-- ==================== REGIME MODAL ==================== -->
<div id="regime-modal" class="modal-overlay" onclick="if(event.target===this)closeRegimeModal()">
  <div class="modal-content">
    <button class="modal-close" onclick="closeRegimeModal()">&times;</button>
    <h2 style="color:#58a6ff; margin-bottom:16px;">Market Regime Filter</h2>

    <div class="not-mancini">
      <div class="not-mancini-title">Not from Mancini</div>
      <div class="not-mancini-text">
        This is an engineering enhancement &mdash; Mancini never teaches directional filtering.
        Across 505 Substack posts, there are zero mentions of EMA-based regime filtering.
        The Mancini method works in all market conditions. This uses an 80-day EMA slope
        on daily bars to classify Bullish/Bearish/Neutral, then gates entries: longs in
        bull/neutral, shorts in bear/neutral. Improved backtest profit factor, but consider
        this an optional safety layer, not part of the core methodology.
      </div>
    </div>

    <div style="margin-bottom:16px;">
      <div style="font-size:13px; color:#8b949e; line-height:1.6;">
        The regime filter determines the overall market direction using the <strong>80-day
        Exponential Moving Average (EMA)</strong> slope on daily bars. It gates which patterns
        are allowed to trade:
      </div>
    </div>
    <div style="margin-bottom:16px;">
      <div class="stat-row">
        <span class="stat-label" style="font-size:14px">Current Reading</span>
        <span class="stat-value $regime_css" style="font-size:18px">$regime</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Daily Bars Loaded</span>
        <span class="stat-value">$regime_daily_bars</span>
      </div>
    </div>
    <div style="background:#0d1117; border-radius:8px; padding:16px; margin-bottom:16px;">
      <div style="font-weight:700; color:#3fb950; margin-bottom:8px;">BULLISH Regime</div>
      <div style="color:#8b949e; font-size:12px; line-height:1.6;">
        80-day EMA slope is positive (price trending up over ~4 months).
        <strong>Longs enabled, shorts disabled.</strong>
        Failed Breakdowns and Level Reclaims fire normally. Breakdown Shorts are blocked
        because shorting against the trend has negative expectancy.
      </div>
    </div>
    <div style="background:#0d1117; border-radius:8px; padding:16px; margin-bottom:16px;">
      <div style="font-weight:700; color:#f85149; margin-bottom:8px;">BEARISH Regime</div>
      <div style="color:#8b949e; font-size:12px; line-height:1.6;">
        80-day EMA slope is negative (price trending down).
        <strong>Both longs and shorts enabled.</strong>
        Breakdown Shorts activate, catching confirmed support breaks.
        Longs still fire because failed breakdowns work in all regimes
        (they catch the exhaustion of sell-offs).
      </div>
    </div>
    <div style="background:#0d1117; border-radius:8px; padding:16px; margin-bottom:16px;">
      <div style="font-weight:700; color:#d29922; margin-bottom:8px;">NEUTRAL Regime</div>
      <div style="color:#8b949e; font-size:12px; line-height:1.6;">
        80-day EMA slope is flat (within the ATR-based threshold).
        <strong>Longs enabled, shorts disabled.</strong>
        Same as bullish &mdash; the engine defaults to long-only unless there's
        a clear bearish trend to justify shorting.
      </div>
    </div>
    <div style="background:#0d1117; border-radius:8px; padding:16px;">
      <div style="font-weight:700; color:#484f58; margin-bottom:8px;">&mdash; (No Data)</div>
      <div style="color:#8b949e; font-size:12px; line-height:1.6;">
        Regime cannot be computed &mdash; not enough daily bars loaded from IB.
        The filter needs at least 80 daily bars for the EMA.
        <strong>Longs default to enabled, shorts default to disabled.</strong>
        The engine now requests 1 year of daily bars from IB at startup to fix this.
      </div>
    </div>
  </div>
</div>

<div class="refresh-note">Auto-refreshes via AJAX every 15s &mdash; no page reload</div>

<!-- Lightweight Charts -->
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
// Tab switching with hash persistence (survives refresh)
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(function(el) { el.classList.remove('active'); });
  document.querySelectorAll('.tab-btn').forEach(function(el) { el.classList.remove('active'); });
  document.getElementById('tab-' + name).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(function(el) {
    if (el.getAttribute('data-tab') === name) el.classList.add('active');
  });
  window.location.hash = name;
  if (name === 'chart') initChart();
  if (name === 'history') loadTradeHistory();
  if (name === 'shadow') loadShadowEvents();
  if (name === 'report') loadDailyReports();
}

// Restore tab from URL hash on load
(function() {
  var hash = window.location.hash.replace('#', '');
  if (hash && document.getElementById('tab-' + hash)) {
    switchTab(hash);
  }
})();

// Regime modal
function showRegimeModal() {
  document.getElementById('regime-modal').classList.add('show');
}
function closeRegimeModal() {
  document.getElementById('regime-modal').classList.remove('show');
}

// Near-miss toggle
function toggleNearMisses() {
  var content = document.getElementById('near-misses-content');
  var arrow = document.getElementById('nm-arrow');
  if (content.style.display === 'none') {
    content.style.display = 'block';
    arrow.classList.add('open');
  } else {
    content.style.display = 'none';
    arrow.classList.remove('open');
  }
}

// Substack toggle
function toggleSubstack() {
  var content = document.getElementById('substack-content');
  var arrow = document.getElementById('ss-arrow');
  if (content.style.display === 'none') {
    content.style.display = 'block';
    arrow.classList.add('open');
  } else {
    content.style.display = 'none';
    arrow.classList.remove('open');
  }
}

// Retrospective toggle
function toggleRetro() {
  var content = document.getElementById('retro-content');
  var arrow = document.getElementById('retro-arrow');
  if (content.style.display === 'none') {
    content.style.display = 'block';
    arrow.classList.add('open');
  } else {
    content.style.display = 'none';
    arrow.classList.remove('open');
  }
}

// Candlestick chart
var _chartInitialized = false;
var _candleSeries = null;   // module-level so the AJAX refresh can push live bars
var _lastBarTime = null;    // dedupe: only redraw when a new/changed bar arrives
function initChart() {
  if (_chartInitialized) return;
  var container = document.getElementById('chart-container');
  if (!container || container.offsetWidth === 0) {
    setTimeout(initChart, 100);
    return;
  }

  // Check if library loaded
  if (typeof LightweightCharts === 'undefined') {
    container.innerHTML = '<div style="padding:40px;text-align:center;color:#8b949e;">Loading chart library...</div>';
    setTimeout(initChart, 500);
    return;
  }

  var chart = LightweightCharts.createChart(container, {
    width: container.offsetWidth,
    height: 400,
    layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
    grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
    rightPriceScale: { borderColor: '#30363d' },
  });

  var candleSeries = chart.addCandlestickSeries({
    upColor: '#3fb950', downColor: '#f85149',
    borderUpColor: '#3fb950', borderDownColor: '#f85149',
    wickUpColor: '#3fb950', wickDownColor: '#f85149',
  });

  var bars = $bars_json;
  if (bars && bars.length > 0) {
    candleSeries.setData(bars);
    _candleSeries = candleSeries;
    _lastBarTime = bars[bars.length - 1].time;
  } else {
    container.innerHTML = '<div style="padding:40px;text-align:center;color:#8b949e;">No bar data available yet. Waiting for bars to accumulate...</div>';
    return;
  }

  var levels = $levels_json;
  if (levels && levels.length > 0) {
    var nearest = levels.slice(0, 10);
    for (var i = 0; i < nearest.length; i++) {
      var lv = nearest[i];
      var isSupport = lv.type.indexOf('LOW') >= 0 || lv.type.indexOf('SUPPORT') >= 0;
      var isResistance = lv.type.indexOf('HIGH') >= 0 || lv.type.indexOf('RESISTANCE') >= 0;
      var color = isSupport ? '#3fb95080' : (isResistance ? '#f8514980' : '#58a6ff80');
      candleSeries.createPriceLine({
        price: lv.price, color: color, lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: lv.type.replace(/_/g, ' '),
      });
    }
  }

  chart.timeScale().fitContent();
  _chartInitialized = true;

  new ResizeObserver(function() {
    chart.applyOptions({ width: container.offsetWidth });
  }).observe(container);
}

// AJAX refresh — no full page reload
var _lastRefresh = Date.now();

function updateDashboard() {
  fetch('/api/fragments')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      _lastRefresh = Date.now();
      var s = data.status;
      var h = data.html;

      // Update simple ticker values
      var price = s.last_price || 0;
      var el;

      el = document.getElementById('tk-price');
      if (el) el.textContent = price > 0 ? price.toFixed(2) : '\u2014';

      el = document.getElementById('tk-high');
      if (el) el.textContent = (s.session_high || 0) > 0 ? s.session_high.toFixed(2) : '\u2014';

      el = document.getElementById('tk-low');
      if (el) el.textContent = (s.session_low || 999999) < 999999 ? s.session_low.toFixed(2) : '\u2014';

      el = document.getElementById('tk-bars');
      if (el) el.textContent = s.bar_count || 0;

      el = document.getElementById('tk-bar-pst');
      if (el) el.textContent = s.last_bar_pst || '\u2014';

      el = document.getElementById('tk-bar-et');
      if (el) el.textContent = s.last_bar_et || '\u2014';

      // Live chart: push fresh bars into the candle series (the initial page
      // render drew once and never updated). setData redraws in place and
      // preserves the user's zoom/scroll; skipped when no new bar arrived.
      try {
        if (_candleSeries && s.bars && s.bars.length > 0) {
          var newest = s.bars[s.bars.length - 1];
          if (newest.time !== _lastBarTime ||
              (newest.close !== undefined && newest.time === _lastBarTime)) {
            _candleSeries.setData(s.bars);
            _lastBarTime = newest.time;
          }
        }
      } catch (e) { /* chart refresh is best-effort */ }

      // Market correlation data (VIX, SPY, 10Y)
      var md = s.market_data;
      var mdStrip = document.getElementById('market-data-strip');
      if (md && mdStrip) {
        mdStrip.style.display = '';
        el = document.getElementById('mk-vix');
        if (el && md.vix != null) el.textContent = md.vix.toFixed(1);
        el = document.getElementById('mk-spy');
        if (el && md.spy != null) el.textContent = md.spy.toFixed(2);
        el = document.getElementById('mk-yield');
        if (el && md.yield_10y != null) el.textContent = md.yield_10y.toFixed(2) + '%';
      }

      el = document.getElementById('hdr-update');
      if (el) el.textContent = s.last_update_pst || '\u2014';

      // PnL
      var pnl = s.daily_pnl_pts || 0;
      el = document.getElementById('pnl-value');
      if (el) {
        el.textContent = pnl !== 0 ? (pnl > 0 ? '+' : '') + pnl.toFixed(1) + ' pts' : '0.0 pts';
        el.className = 'pnl-big ' + (pnl > 0 ? 'green' : pnl < 0 ? 'red' : 'dim');
      }

      // Stats
      el = document.getElementById('stat-winners'); if (el) el.textContent = s.winners || 0;
      el = document.getElementById('stat-losers'); if (el) el.textContent = s.losers || 0;
      el = document.getElementById('stat-balance'); if (el) el.textContent = s.account_balance || '\u2014';
      el = document.getElementById('stat-equity'); if (el) el.textContent = s.account_equity || '\u2014';
      el = document.getElementById('stat-total'); if (el) el.textContent = s.total_logged_trades || 0;
      el = document.getElementById('stat-regime'); if (el) el.textContent = s.regime || '\u2014';
      el = document.getElementById('stat-dbars'); if (el) el.textContent = s.regime_daily_bars || 0;

      // Complex sections via server-rendered HTML
      el = document.getElementById('levels-content'); if (el && h.levels) el.innerHTML = h.levels;
      el = document.getElementById('trades-content'); if (el && h.trades) el.innerHTML = h.trades;
      el = document.getElementById('phantoms-content'); if (el && h.phantoms) el.innerHTML = h.phantoms;
      el = document.getElementById('position-content'); if (el && h.position) el.innerHTML = h.position;
      el = document.getElementById('log-content'); if (el && h.log) el.innerHTML = h.log;
      el = document.getElementById('near-misses-content');
      if (el && h.near_misses) {
        var wasOpen = el.style.display !== 'none';
        el.innerHTML = h.near_misses;
        if (!wasOpen) el.style.display = 'none';
      }
      el = document.getElementById('substack-content');
      if (el && h.substack) {
        var wasOpen = el.style.display !== 'none';
        el.innerHTML = h.substack;
        if (!wasOpen) el.style.display = 'none';
      }
      el = document.getElementById('retro-content');
      if (el && h.retrospective) {
        var wasOpen = el.style.display !== 'none';
        el.innerHTML = h.retrospective;
        if (!wasOpen) el.style.display = 'none';
      }
    })
    .catch(function(err) {
      console.log('AJAX refresh error:', err);
    });
}

// Update "last updated" counter
function updateLastUpdated() {
  var el = document.getElementById('last-updated');
  if (el) {
    var secs = Math.round((Date.now() - _lastRefresh) / 1000);
    el.textContent = secs > 0 ? '(' + secs + 's ago)' : '';
  }
}

// Client-side market hours update (deterministic from ET clock)
function updateMarketHours() {
  // Compute ET time from UTC
  var now = new Date();
  var etStr = now.toLocaleString('en-US', {timeZone: 'America/New_York'});
  var et = new Date(etStr);
  var ptStr = now.toLocaleString('en-US', {timeZone: 'America/Los_Angeles'});
  var pt = new Date(ptStr);
  var h = et.getHours(), m = et.getMinutes(), wd = et.getDay(); // 0=Sun, 5=Fri, 6=Sat
  var t = h * 60 + m; // minutes since midnight ET
  var ptTime = pt.toLocaleTimeString('en-US', {hour:'numeric', minute:'2-digit', timeZone:'America/Los_Angeles'});
  var ptDay = pt.toLocaleDateString('en-US', {weekday:'long', timeZone:'America/Los_Angeles'});

  var label, css, detail, next;

  // Weekend checks
  if (wd === 6) { // Saturday
    label = 'MARKET CLOSED'; css = 'mh-weekend';
    detail = 'Weekend \u2014 ' + ptDay + ' ' + ptTime + ' PT';
    next = 'Opens Sunday 3:00 PM PT (6:00 PM ET)';
  } else if (wd === 0 && t < 18*60) { // Sunday before 6 PM ET
    label = 'MARKET CLOSED'; css = 'mh-weekend';
    detail = 'Weekend \u2014 ' + ptDay + ' ' + ptTime + ' PT';
    next = 'Opens today at 3:00 PM PT (6:00 PM ET)';
  } else if (wd === 5 && t >= 17*60) { // Friday after 5 PM ET
    label = 'MARKET CLOSED'; css = 'mh-weekend';
    detail = 'Weekend \u2014 ' + ptDay + ' ' + ptTime + ' PT';
    next = 'Opens Sunday 3:00 PM PT (6:00 PM ET)';
  } else if (t >= 17*60 && t < 18*60) { // Daily break
    label = 'DAILY BREAK'; css = 'mh-break';
    detail = 'CME maintenance \u2014 ' + ptTime + ' PT';
    next = 'Reopens at 3:00 PM PT (6:00 PM ET)';
  } else { // Market open
    label = 'MARKET OPEN'; css = 'mh-open';
    next = 'Daily break at 2:00 PM PT (5:00 PM ET)';
    if (t >= 18*60 && t < 22*60) detail = 'Globex Evening (time gate bypassed)';
    else if (t >= 22*60 || t < 2*60) detail = 'Globex Overnight';
    else if (t < 6*60) detail = 'European Session (time gate bypassed)';
    else if (t < 9*60+30) detail = 'US Pre-Market';
    else if (t < 11*60) detail = 'RTH Morning (Prime Window)';
    else if (t < 13*60) detail = 'RTH Midday';
    else if (t < 15*60) detail = 'Chop Zone (time gate bypassed)';
    else if (t < 16*60+50) detail = 'RTH Afternoon (FB Only)';
    else detail = 'EOD Settle / Flatten';
    detail += ' \u2014 ' + ptTime + ' PT';
  }

  var el = document.getElementById('mh-status');
  if (el) { el.textContent = label; el.className = 'mh-status ' + css; }
  el = document.getElementById('mh-detail');
  if (el) el.innerHTML = detail;
  el = document.getElementById('mh-next');
  if (el) el.textContent = next;
}
updateMarketHours();
setInterval(updateMarketHours, 10000);

setInterval(updateDashboard, 15000);
setInterval(updateLastUpdated, 1000);

// ==================== TRADE HISTORY ====================
var _allTrades = [];
var _sortCol = 'exit_time';
var _sortAsc = false;
var _filterDate = 'all';
var _filterProd = false;
var _expandedIdx = -1;
var _historyLoaded = false;

function loadTradeHistory() {
  if (_historyLoaded && _allTrades.length > 0) { renderTradeHistory(); return; }
  var tbody = document.getElementById('history-tbody');
  if (tbody) tbody.innerHTML = '<tr><td colspan="13" style="text-align:center;padding:40px;color:#8b949e;">Loading trade history...</td></tr>';
  fetch('/api/trades')
    .then(function(r) { return r.json(); })
    .then(function(trades) {
      _allTrades = trades;
      _historyLoaded = true;
      renderTradeHistory();
    })
    .catch(function(err) {
      if (tbody) tbody.innerHTML = '<tr><td colspan="13" style="text-align:center;padding:40px;color:#f85149;">Failed to load trades: ' + err + '</td></tr>';
    });
}

function getFiltered() {
  var filtered = _allTrades.slice();
  if (_filterDate !== 'all') filtered = filtered.filter(function(t) { return getDatePT(t) === _filterDate; });
  if (_filterProd) filtered = filtered.filter(function(t) { return t.production_would_take && !(t.gate_bypassed && t.gate_bypassed.length); });
  filtered.sort(function(a, b) {
    var va = a[_sortCol], vb = b[_sortCol];
    if (va == null) va = '';
    if (vb == null) vb = '';
    if (typeof va === 'number' && typeof vb === 'number') return _sortAsc ? va - vb : vb - va;
    return _sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
  });
  return filtered;
}

function renderTradeHistory() {
  var filtered = getFiltered();
  updateHistorySummary(filtered);
  updateDatePills();
  renderHistoryTable(filtered);
  renderPnlChart(filtered);
}

function updateHistorySummary(trades) {
  var total = trades.length;
  var wins = 0, totalPnl = 0, grossWin = 0, grossLoss = 0, best = -Infinity, worst = Infinity;
  for (var i = 0; i < trades.length; i++) {
    var p = trades[i].pnl_pts || 0;
    totalPnl += p;
    if (p > 0) { wins++; grossWin += p; }
    if (p < 0) { grossLoss += Math.abs(p); }
    if (p > best) best = p;
    if (p < worst) worst = p;
  }
  var wr = total > 0 ? (wins / total * 100).toFixed(1) + '%' : '--';
  var pf = grossLoss > 0 ? (grossWin / grossLoss).toFixed(2) : grossWin > 0 ? '99.0' : '--';
  var avg = total > 0 ? (totalPnl / total).toFixed(1) : '--';

  var el;
  el = document.getElementById('hs-total'); if (el) { el.textContent = total; el.className = 'history-stat-val blue'; }
  el = document.getElementById('hs-winrate'); if (el) { el.textContent = wr; el.className = 'history-stat-val ' + (wins/total >= 0.5 ? 'green' : total > 0 ? 'yellow' : 'dim'); }
  el = document.getElementById('hs-pnl'); if (el) { el.textContent = total > 0 ? (totalPnl > 0 ? '+' : '') + totalPnl.toFixed(1) : '--'; el.className = 'history-stat-val ' + (totalPnl > 0 ? 'green' : totalPnl < 0 ? 'red' : 'dim'); }
  el = document.getElementById('hs-pf'); if (el) { el.textContent = pf; el.className = 'history-stat-val ' + (parseFloat(pf) >= 1.0 ? 'green' : 'red'); }
  el = document.getElementById('hs-best'); if (el) { el.textContent = total > 0 ? '+' + best.toFixed(1) : '--'; el.className = 'history-stat-val green'; }
  el = document.getElementById('hs-worst'); if (el) { el.textContent = total > 0 ? worst.toFixed(1) : '--'; el.className = 'history-stat-val red'; }
  el = document.getElementById('hs-avg'); if (el) { el.textContent = avg !== '--' ? (parseFloat(avg) > 0 ? '+' : '') + avg : '--'; el.className = 'history-stat-val ' + (parseFloat(avg) > 0 ? 'green' : parseFloat(avg) < 0 ? 'red' : 'dim'); }
}

function getDatePT(t) {
  return fmtDatePT(t.entry_time);
}

function updateDatePills() {
  var dates = {};
  for (var i = 0; i < _allTrades.length; i++) {
    var d = getDatePT(_allTrades[i]);
    if (d && d !== '--') dates[d] = (dates[d] || 0) + 1;
  }
  var sorted = Object.keys(dates).sort().reverse();
  var html = '<span class="filter-pill ' + (_filterDate === 'all' ? 'active' : '') + '" onclick="setDateFilter(\'all\')">All (' + _allTrades.length + ')</span> ';
  for (var i = 0; i < Math.min(sorted.length, 14); i++) {
    var d = sorted[i];
    var label = d.slice(5);
    html += '<span class="filter-pill ' + (_filterDate === d ? 'active' : '') + '" onclick="setDateFilter(\'' + d + '\')">' + label + ' (' + dates[d] + ')</span> ';
  }
  var el = document.getElementById('date-pills');
  if (el) el.innerHTML = html;
}

function setDateFilter(d) {
  _filterDate = d;
  _expandedIdx = -1;
  renderTradeHistory();
}

function toggleProdFilter() {
  _filterProd = document.getElementById('prod-filter').checked;
  _expandedIdx = -1;
  renderTradeHistory();
}

function sortHistory(col) {
  if (_sortCol === col) { _sortAsc = !_sortAsc; } else { _sortCol = col; _sortAsc = false; }
  _expandedIdx = -1;
  renderTradeHistory();
  // Update sort arrows
  document.querySelectorAll('.sort-arrow').forEach(function(el) { el.className = 'sort-arrow'; el.textContent = ''; });
  var arrow = document.getElementById('sa-' + col);
  if (arrow) { arrow.className = 'sort-arrow active'; arrow.textContent = _sortAsc ? ' \\u25B2' : ' \\u25BC'; }
}

function fmtTime(ts) {
  if (!ts) return '--';
  try {
    var s = String(ts);
    // New records have timezone offset (e.g. -05:00). Old records are naive ET (container TZ=America/New_York).
    var hasOffset = /[+-]\d{2}:\d{2}$$/.test(s) || s.endsWith('Z');
    if (!hasOffset && s.length >= 19) {
      s += '-05:00';  // naive timestamps are ET (container TZ=America/New_York)
    }
    var d = new Date(s);
    return d.toLocaleTimeString('en-US', {hour:'numeric', minute:'2-digit', timeZone:'America/Los_Angeles'}) + ' PT';
  } catch(e) { return ts.slice(11, 16); }
}

function fmtDatePT(ts) {
  if (!ts) return '--';
  try {
    var s = String(ts);
    var hasOffset = /[+-]\d{2}:\d{2}$$/.test(s) || s.endsWith('Z');
    if (!hasOffset && s.length >= 19) { s += '-05:00'; }
    var d = new Date(s);
    return d.toLocaleDateString('en-US', {month:'2-digit', day:'2-digit', timeZone:'America/Los_Angeles'});
  } catch(e) { return ts.slice(0, 10); }
}

function fmtDuration(entry, exit) {
  if (!entry || !exit) return '--';
  try {
    var ms = new Date(exit) - new Date(entry);
    var mins = Math.round(ms / 60000);
    if (mins < 60) return mins + 'm';
    return Math.floor(mins/60) + 'h ' + (mins%60) + 'm';
  } catch(e) { return '--'; }
}

function renderHistoryTable(trades) {
  var tbody = document.getElementById('history-tbody');
  if (!tbody) return;
  if (trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="13" style="text-align:center;padding:40px;color:#8b949e;">No trades match the current filters</td></tr>';
    return;
  }
  var html = '';
  for (var i = 0; i < trades.length; i++) {
    var t = trades[i];
    var dirCss = t.direction === 'long' ? 'green' : 'red';
    var pnl = t.pnl_pts || 0;
    var pnlCss = pnl > 0 ? 'green' : pnl < 0 ? 'red' : 'dim';
    var bypassed = t.gate_bypassed && t.gate_bypassed.length > 0;
    var prodValid = t.production_would_take && !bypassed;
    var modeBadge = prodValid
      ? '<span class="bypass-tag bypass-production">PROD</span>'
      : '<span class="bypass-tag bypass-collection">COLL</span>';
    var expanded = i === _expandedIdx;

    html += '<tr class="' + (expanded ? 'expanded' : '') + '" onclick="toggleTradeDetail(' + i + ')">'
      + '<td>' + fmtDatePT(t.entry_time) + '</td>'
      + '<td>' + fmtTime(t.entry_time) + '</td>'
      + '<td>' + fmtTime(t.exit_time) + '</td>'
      + '<td class="' + dirCss + '">' + (t.direction === 'long' ? 'L' : 'S') + '</td>'
      + '<td><span class="level-tag">' + (t.pattern || '?').replace('failed_breakdown','FB').replace('level_reclaim','LR').replace('FAILED_BREAKDOWN','FB').replace('LEVEL_RECLAIM','LR').replace('breakdown_short','BD') + '</span></td>'
      + '<td>' + (t.entry_price ? t.entry_price.toFixed(2) : '--') + '</td>'
      + '<td>' + (t.stop ? t.stop.toFixed(2) : '--') + '</td>'
      + '<td>' + (t.target_1 ? t.target_1.toFixed(2) : '--') + '</td>'
      + '<td>' + (t.exit_price ? t.exit_price.toFixed(2) : '--') + '</td>'
      + '<td class="' + pnlCss + '" style="font-weight:700">' + (pnl > 0 ? '+' : '') + pnl.toFixed(1) + '</td>'
      + '<td>' + (t.rr_ratio ? t.rr_ratio.toFixed(2) : '--') + '</td>'
      + '<td class="dim" style="font-size:11px">' + (t.exit_reason || '?').replace(/_/g,' ') + '</td>'
      + '<td>' + modeBadge + '</td>'
      + '</tr>';

    if (expanded) {
      var exitsHtml = '';
      if (t.exits && t.exits.length > 1) {
        exitsHtml = '<div style="margin-top:10px; border-top:1px solid #30363d; padding-top:8px;">'
          + '<div style="font-size:10px; color:#8b949e; text-transform:uppercase; margin-bottom:6px; letter-spacing:0.5px;">Exit Breakdown</div>'
          + '<table style="width:100%; font-size:11px; border-collapse:collapse;">'
          + '<tr style="color:#8b949e;"><th style="text-align:left; padding:2px 8px;">Time</th><th style="text-align:right; padding:2px 8px;">Qty</th><th style="text-align:right; padding:2px 8px;">Price</th><th style="text-align:right; padding:2px 8px;">PnL</th><th style="text-align:left; padding:2px 8px;">Reason</th></tr>';
        for (var ei = 0; ei < t.exits.length; ei++) {
          var ex = t.exits[ei];
          var ePnl = ex.pnl_pts || 0;
          var ePnlCss = ePnl > 0 ? 'green' : ePnl < 0 ? 'red' : 'dim';
          exitsHtml += '<tr><td style="padding:2px 8px;">' + fmtTime(ex.time) + '</td>'
            + '<td style="text-align:right; padding:2px 8px;">' + (ex.contracts || '?') + '</td>'
            + '<td style="text-align:right; padding:2px 8px; font-family:SF Mono,monospace;">' + (ex.price ? ex.price.toFixed(2) : '--') + '</td>'
            + '<td style="text-align:right; padding:2px 8px; font-family:SF Mono,monospace;" class="' + ePnlCss + '">' + (ePnl > 0 ? '+' : '') + ePnl.toFixed(1) + '</td>'
            + '<td style="padding:2px 8px; color:#8b949e;">' + (ex.reason || '').replace(/_/g,' ') + '</td></tr>';
        }
        exitsHtml += '</table></div>';
      }
      html += '<tr class="trade-detail-row"><td colspan="13"><div class="trade-detail"><div class="trade-detail-grid">'
        + detailItem('Duration', fmtDuration(t.entry_time, t.exit_time))
        + detailItem('Contracts', t.contracts || '--')
        + detailItem('PnL ($$)', t.pnl_dollars ? '$$' + t.pnl_dollars.toFixed(2) : '--')
        + detailItem('Level Type', (t.level_type || '--').replace(/_/g,' '))
        + detailItem('Level Price', t.level_price ? t.level_price.toFixed(2) : '--')
        + detailItem('Target 2', t.target_2 ? t.target_2.toFixed(2) : '--')
        + detailItem('Regime', t.regime || '--')
        + detailItem('Session', (t.session_window || '--').replace(/_/g,' '))
        + (bypassed ? detailItem('Bypassed', t.gate_bypassed.join(', ')) : '')
        + '</div>' + exitsHtml + '</div></td></tr>';
    }
  }
  tbody.innerHTML = html;
}

function detailItem(label, value) {
  return '<div class="trade-detail-item"><div class="trade-detail-label">' + label + '</div><div class="trade-detail-value">' + value + '</div></div>';
}

function toggleTradeDetail(idx) {
  _expandedIdx = _expandedIdx === idx ? -1 : idx;
  renderHistoryTable(getFiltered());
}

function renderPnlChart(trades) {
  var svg = document.getElementById('pnl-chart');
  if (!svg || trades.length === 0) {
    if (svg) svg.innerHTML = '<text x="50%" y="60" text-anchor="middle" fill="#8b949e" font-size="12">No trade data</text>';
    return;
  }
  // Reverse to chronological order for cumulative calc
  var chrono = trades.slice().reverse();
  var cumPnl = [];
  var running = 0;
  for (var i = 0; i < chrono.length; i++) {
    running += (chrono[i].pnl_pts || 0);
    cumPnl.push(running);
  }

  var w = svg.getBoundingClientRect().width || 800;
  var h = 120;
  var minY = Math.min(0, Math.min.apply(null, cumPnl));
  var maxY = Math.max(0, Math.max.apply(null, cumPnl));
  var range = maxY - minY || 1;
  var pad = 10;

  function x(i) { return pad + (i / Math.max(cumPnl.length - 1, 1)) * (w - 2 * pad); }
  function y(v) { return h - pad - ((v - minY) / range) * (h - 2 * pad); }

  var points = '';
  var areaPoints = x(0) + ',' + y(0) + ' ';
  for (var i = 0; i < cumPnl.length; i++) {
    points += x(i) + ',' + y(cumPnl[i]) + ' ';
    areaPoints += x(i) + ',' + y(cumPnl[i]) + ' ';
  }
  areaPoints += x(cumPnl.length - 1) + ',' + y(0);

  var lineColor = running >= 0 ? '#3fb950' : '#f85149';
  var fillColor = running >= 0 ? '#3fb95015' : '#f8514915';
  var zeroY = y(0);

  var svgHtml = '<line x1="' + pad + '" y1="' + zeroY + '" x2="' + (w - pad) + '" y2="' + zeroY + '" stroke="#30363d" stroke-dasharray="4,4"/>'
    + '<polygon points="' + areaPoints + '" fill="' + fillColor + '"/>'
    + '<polyline points="' + points + '" fill="none" stroke="' + lineColor + '" stroke-width="2"/>'
    + '<text x="' + (w - pad) + '" y="' + (y(running) - 6) + '" text-anchor="end" fill="' + lineColor + '" font-size="11" font-weight="700" font-family="SF Mono, monospace">' + (running > 0 ? '+' : '') + running.toFixed(1) + '</text>'
    + '<text x="' + pad + '" y="' + (h - 2) + '" fill="#484f58" font-size="9">' + (chrono[0].date || '') + '</text>'
    + '<text x="' + (w - pad) + '" y="' + (h - 2) + '" text-anchor="end" fill="#484f58" font-size="9">' + (chrono[chrono.length-1].date || '') + '</text>';

  svg.innerHTML = svgHtml;
}

// ==================== DAILY REPORT ====================
var _reportLoaded = false;
function loadDailyReports() {
  var container = document.getElementById('report-content');
  if (!container) return;
  if (!_reportLoaded) container.innerHTML = '<div style="color:#8b949e;text-align:center;padding:40px;">Loading reports...</div>';
  fetch('/api/reports')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      _reportLoaded = true;
      renderDailyReports(data.reports || []);
    })
    .catch(function(err) {
      container.innerHTML = '<div style="color:#f85149;text-align:center;padding:40px;">Failed to load reports: ' + err + '</div>';
    });
}

function renderDailyReports(reports) {
  var container = document.getElementById('report-content');
  if (!container) return;
  if (reports.length === 0) {
    container.innerHTML = '<div style="color:#8b949e;text-align:center;padding:40px;">No reports yet. First report generates at 2:10 AM ET.</div>';
    return;
  }
  var latest = reports[reports.length - 1];
  var html = '';

  // Latest report header
  html += '<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px;">';
  html += '<h3 style="color:#58a6ff;margin:0 0 12px 0;">Latest Report: ' + (latest.session_date || '?') + '</h3>';

  // Trades summary
  var tr = latest.trades || {};
  var pnlColor = (tr.total_pnl || 0) >= 0 ? '#3fb950' : '#f85149';
  html += '<div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:12px;">';
  html += '<div><span style="color:#8b949e;">Trades</span><br><span style="font-size:20px;font-weight:700;color:#f0f6fc;">' + (tr.count || 0) + '</span></div>';
  html += '<div><span style="color:#8b949e;">W/L</span><br><span style="font-size:20px;font-weight:700;color:#f0f6fc;">' + (tr.wins || 0) + '/' + (tr.losses || 0) + '</span></div>';
  html += '<div><span style="color:#8b949e;">PnL</span><br><span style="font-size:20px;font-weight:700;color:' + pnlColor + ';">' + ((tr.total_pnl || 0) >= 0 ? '+' : '') + (tr.total_pnl || 0).toFixed(1) + ' pts</span></div>';
  html += '</div>';

  // Winners
  var win = latest.winners || {};
  if (win.count > 0) {
    html += '<div style="border-top:1px solid #21262d;padding-top:8px;margin-top:8px;">';
    html += '<strong style="color:#3fb950;">Winner Patterns (' + win.count + ')</strong><br>';
    var factors = win.common_factors || [];
    for (var i = 0; i < factors.length; i++) {
      html += '<span style="color:#c9d1d9;font-size:13px;">&#x1F3C6; ' + factors[i] + '</span><br>';
    }
    html += '</div>';
  }

  // Losers
  var los = latest.losers || {};
  if (los.count > 0) {
    html += '<div style="border-top:1px solid #21262d;padding-top:8px;margin-top:8px;">';
    html += '<strong style="color:#f85149;">Loser Warning Signs (' + los.count + ')</strong><br>';
    var warnings = los.warning_signs || [];
    for (var i = 0; i < warnings.length; i++) {
      html += '<span style="color:#c9d1d9;font-size:13px;">&#x26A0;&#xFE0F; ' + warnings[i] + '</span><br>';
    }
    html += '</div>';
  }

  // Near misses
  var nm = latest.near_misses || {};
  if (nm.count > 0) {
    html += '<div style="border-top:1px solid #21262d;padding-top:8px;margin-top:8px;">';
    html += '<strong style="color:#d29922;">Near-Misses (' + nm.count + ')</strong><br>';
    html += '<span style="color:#c9d1d9;font-size:13px;">Would-have-won: ' + (nm.would_have_won || 0) + ' | Would-have-lost: ' + (nm.would_have_lost || 0) + '</span><br>';
    var netGate = (nm.net_gate_value || 0);
    var gateColor = netGate >= 0 ? '#3fb950' : '#f85149';
    html += '<span style="color:' + gateColor + ';font-size:13px;">Net gate value: ' + (netGate >= 0 ? '+' : '') + netGate.toFixed(0) + ' pts</span>';
    html += '</div>';
  }

  // Blind spots
  var bs = latest.blind_spots || [];
  if (bs.length > 0) {
    html += '<div style="border-top:1px solid #21262d;padding-top:8px;margin-top:8px;">';
    html += '<strong style="color:#f85149;">Mancini Blind Spots (' + bs.length + ')</strong><br>';
    for (var i = 0; i < Math.min(bs.length, 5); i++) {
      html += '<span style="color:#c9d1d9;font-size:13px;">' + (bs[i].price || '?') + ' (' + (bs[i].side || '?') + ')</span><br>';
    }
    html += '</div>';
  }

  // Recommendations
  var recs = latest.recommendations || [];
  if (recs.length > 0) {
    html += '<div style="border-top:1px solid #21262d;padding-top:8px;margin-top:8px;">';
    html += '<strong style="color:#58a6ff;">Recommendations</strong><br>';
    for (var i = 0; i < recs.length; i++) {
      html += '<span style="color:#c9d1d9;font-size:13px;">&#x1F4A1; ' + recs[i] + '</span><br>';
    }
    html += '</div>';
  }

  html += '</div>';

  // History table
  if (reports.length > 1) {
    html += '<h3 style="color:#8b949e;margin:16px 0 8px;">Report History</h3>';
    html += '<table style="width:100%;border-collapse:collapse;">';
    html += '<tr style="border-bottom:1px solid #30363d;">';
    html += '<th style="text-align:left;padding:6px;color:#8b949e;font-size:11px;">Date</th>';
    html += '<th style="text-align:center;padding:6px;color:#8b949e;font-size:11px;">Trades</th>';
    html += '<th style="text-align:center;padding:6px;color:#8b949e;font-size:11px;">W/L</th>';
    html += '<th style="text-align:right;padding:6px;color:#8b949e;font-size:11px;">PnL</th>';
    html += '<th style="text-align:right;padding:6px;color:#8b949e;font-size:11px;">Gate Value</th>';
    html += '<th style="text-align:center;padding:6px;color:#8b949e;font-size:11px;">Blind Spots</th>';
    html += '</tr>';
    for (var i = reports.length - 1; i >= 0; i--) {
      var r = reports[i];
      var rt = r.trades || {};
      var rnm = r.near_misses || {};
      var rpnl = rt.total_pnl || 0;
      var rpnlColor = rpnl >= 0 ? '#3fb950' : '#f85149';
      var rgv = rnm.net_gate_value || 0;
      var rgvColor = rgv >= 0 ? '#3fb950' : '#f85149';
      html += '<tr style="border-bottom:1px solid #21262d;">';
      html += '<td style="padding:6px;color:#c9d1d9;font-size:12px;">' + (r.session_date || '?') + '</td>';
      html += '<td style="text-align:center;padding:6px;color:#c9d1d9;font-size:12px;">' + (rt.count || 0) + '</td>';
      html += '<td style="text-align:center;padding:6px;color:#c9d1d9;font-size:12px;">' + (rt.wins || 0) + '/' + (rt.losses || 0) + '</td>';
      html += '<td style="text-align:right;padding:6px;color:' + rpnlColor + ';font-size:12px;">' + (rpnl >= 0 ? '+' : '') + rpnl.toFixed(1) + '</td>';
      html += '<td style="text-align:right;padding:6px;color:' + rgvColor + ';font-size:12px;">' + (rgv >= 0 ? '+' : '') + rgv.toFixed(0) + '</td>';
      html += '<td style="text-align:center;padding:6px;color:#c9d1d9;font-size:12px;">' + (r.blind_spots || []).length + '</td>';
      html += '</tr>';
    }
    html += '</table>';
  }

  container.innerHTML = html;
}

// ==================== SHADOW MODE ====================
var _shadowEvents = [];
var _shadowLoaded = false;
var _shadowInterval = null;

function loadShadowEvents() {
  var timeline = document.getElementById('shadow-timeline');
  if (!_shadowLoaded && timeline) {
    timeline.innerHTML = '<div class="shadow-empty"><div class="shadow-empty-text">Loading shadow events...</div></div>';
  }
  fetch('/api/shadow')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      _shadowEvents = data.events || [];
      _shadowLoaded = true;
      renderShadowCards(data);
      renderShadowTimeline(_shadowEvents);
      renderShadowStatsBar(data);
    })
    .catch(function(err) {
      if (timeline) timeline.innerHTML = '<div class="shadow-empty"><div class="shadow-empty-text" style="color:#f85149;">Failed to load shadow events: ' + err + '</div></div>';
    });

  // Set up auto-refresh every 30s
  if (!_shadowInterval) {
    _shadowInterval = setInterval(function() {
      var activeTab = document.querySelector('.tab-btn.active');
      if (activeTab && activeTab.getAttribute('data-tab') === 'shadow') {
        loadShadowEvents();
      }
    }, 30000);
  }
}

function formatShadowTimePT(ts) {
  if (!ts) return '--';
  try {
    var d = new Date(ts);
    return d.toLocaleTimeString('en-US', {hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/Los_Angeles'}) + ' PT';
  } catch(e) {
    var m = String(ts).match(/(\d{2}):(\d{2})/);
    return m ? m[0] : ts;
  }
}

function formatShadowDatePT(ts) {
  if (!ts) return '';
  try {
    var d = new Date(ts);
    return d.toLocaleDateString('en-US', {timeZone: 'America/Los_Angeles', year: 'numeric', month: '2-digit', day: '2-digit'});
  } catch(e) {
    return String(ts).substring(0, 10);
  }
}

function describeSweepEvent(ev) {
  var level = ev.level_price ? ev.level_price.toFixed(1) : '?';
  var sweep = (ev.sweep_depth_pts != null) ? ev.sweep_depth_pts.toFixed(1) : '0.0';
  var curPct = (ev.current_size_factor != null) ? Math.round(ev.current_size_factor * 100) : '?';
  var sugPct = (ev.shadow_size_factor != null) ? Math.round(ev.shadow_size_factor * 100) : '?';
  var sigType = ev.signal_type || 'FB Long';
  var headline = sigType.replace('LONG_QUALIFY','FB Long').replace('BREAKDOWN_SHORT','BD Short').replace('VELOCITY_SHORT','Velocity Short') + ' at ' + level + ': sweep was ' + sweep + ' pts';
  var detail = 'Current size: ' + curPct + '% &rarr; Shadow recommends: ' + sugPct + '%';
  if (curPct !== '?' && sugPct !== '?' && curPct > sugPct) {
    detail += ' (would reduce risk by ' + (curPct - sugPct) + '%)';
  } else if (curPct !== '?' && sugPct !== '?' && sugPct > curPct) {
    detail += ' (would increase size by ' + (sugPct - curPct) + '%)';
  }
  return {headline: headline, detail: detail};
}

function describeMode1Event(ev) {
  var broken = ev.levels_broken || ev.bars_in_mode1 || '?';
  var headline = ev.details || ('Mode 1 detected &mdash; ' + broken + ' support levels broken');
  var detail = 'Would cut FB Long size to 25%';
  if (ev.would_have_done) detail = ev.would_have_done;
  return {headline: headline, detail: detail};
}

function describeVelocityEvent(ev) {
  var levelType = ev.level_type || '?';
  var levelPrice = ev.level_price ? ev.level_price.toFixed(0) : '?';
  var price = ev.entry_price ? ev.entry_price.toFixed(2) : '?';
  var vol = ev.volume ? Math.round(ev.volume) : '?';
  var avgVol = ev.avg_volume_20 ? Math.round(ev.avg_volume_20) : '?';
  var volRatio = (ev.volume && ev.avg_volume_20) ? (ev.volume / ev.avg_volume_20).toFixed(1) + 'x' : '?';
  var headline = levelType + ' ' + levelPrice + ' broke with ' + volRatio + ' volume (' + vol + ' vs avg ' + avgVol + ')';
  var detail = 'Would SHORT at ' + price;
  if (ev.stop_price) detail += ', stop ' + ev.stop_price.toFixed(2);
  if (ev.rr_ratio_t1) detail += ', R:R ' + ev.rr_ratio_t1.toFixed(1);
  return {headline: headline, detail: detail};
}

function renderShadowCards(data) {
  var stats = data.summary || {};
  var sweep = stats.sweep_depth || {};
  var mode1 = stats.mode1 || {};
  var vshort = stats.velocity_short || {};
  var events = data.events || [];

  // Sweep card
  var el = document.getElementById('sc-sweep-status');
  if (el) el.textContent = (sweep.count || 0) + ' events today';
  var sweepLatest = document.getElementById('sc-sweep-latest');
  var lastSweep = null;
  for (var i = events.length - 1; i >= 0; i--) { if (events[i].feature === 'sweep_depth') { lastSweep = events[i]; break; } }
  if (sweepLatest) {
    if (lastSweep) {
      var sd = describeSweepEvent(lastSweep);
      sweepLatest.innerHTML = '<div class="lbl">Latest</div>' + sd.headline + '<br>' + sd.detail;
    } else {
      sweepLatest.innerHTML = '<div class="lbl">Latest</div>No sweep events yet today';
    }
  }
  var sweepImpact = document.getElementById('sc-sweep-impact');
  if (sweepImpact) {
    if (sweep.count > 0 && sweep.avg_size != null) {
      sweepImpact.textContent = 'Avg recommended size: ' + Math.round(sweep.avg_size * 100) + '%';
      sweepImpact.className = 'shadow-card-impact';
    } else { sweepImpact.textContent = ''; }
  }

  // Mode1 card
  el = document.getElementById('sc-mode1-status');
  if (el) {
    if ((mode1.count || 0) > 0) {
      el.innerHTML = (mode1.count || 0) + ' detections today';
    } else {
      el.innerHTML = 'No trend day detected &#x2705;';
    }
  }
  var mode1Latest = document.getElementById('sc-mode1-latest');
  var lastMode1 = null;
  for (var i = events.length - 1; i >= 0; i--) { if (events[i].feature === 'mode1') { lastMode1 = events[i]; break; } }
  if (mode1Latest) {
    if (lastMode1) {
      var md = describeMode1Event(lastMode1);
      mode1Latest.innerHTML = '<div class="lbl">Latest</div>' + md.headline + '<br>' + md.detail;
    } else {
      mode1Latest.innerHTML = '<div class="lbl">Latest</div>Market is in normal Mode 2 \\u2014 bounces working as expected';
    }
  }
  var mode1Impact = document.getElementById('sc-mode1-impact');
  if (mode1Impact) {
    if (mode1.total_bars != null && mode1.total_bars > 0) {
      mode1Impact.textContent = mode1.total_bars + ' bars spent in Mode 1 today';
      mode1Impact.className = 'shadow-card-impact neg';
    } else { mode1Impact.textContent = ''; }
  }

  // Velocity card
  el = document.getElementById('sc-vshort-status');
  if (el) el.textContent = (vshort.count || 0) + ' signals today';
  var vshortLatest = document.getElementById('sc-vshort-latest');
  var lastVshort = null;
  for (var i = events.length - 1; i >= 0; i--) { if (events[i].feature === 'velocity_short') { lastVshort = events[i]; break; } }
  if (vshortLatest) {
    if (lastVshort) {
      var vd = describeVelocityEvent(lastVshort);
      vshortLatest.innerHTML = '<div class="lbl">Latest</div>' + vd.headline + '<br>' + vd.detail;
    } else {
      vshortLatest.innerHTML = '<div class="lbl">Latest</div>No velocity signals yet today';
    }
  }
  var vshortImpact = document.getElementById('sc-vshort-impact');
  if (vshortImpact) {
    if (vshort.total_pnl != null) {
      var pnlStr = (vshort.total_pnl >= 0 ? '+' : '') + vshort.total_pnl.toFixed(1) + ' pts total';
      vshortImpact.textContent = 'Would-have PnL: ' + pnlStr;
      vshortImpact.className = 'shadow-card-impact' + (vshort.total_pnl < 0 ? ' neg' : '');
    } else { vshortImpact.textContent = ''; }
  }
}

function renderShadowTimeline(events) {
  var container = document.getElementById('shadow-timeline');
  if (!container) return;
  if (!events || events.length === 0) {
    container.innerHTML = '<div class="shadow-empty">'
      + '<div class="shadow-empty-icon">&#x1f441;</div>'
      + '<div class="shadow-empty-text">No shadow events yet.<br>Events will appear here as shadow features fire during market hours.</div>'
      + '</div>';
    return;
  }

  // Group events by date (PT), newest first
  var sorted = events.slice().reverse();
  var groups = {};
  var groupOrder = [];
  for (var i = 0; i < sorted.length; i++) {
    var ev = sorted[i];
    var dateStr = formatShadowDatePT(ev.timestamp);
    if (!groups[dateStr]) { groups[dateStr] = []; groupOrder.push(dateStr); }
    groups[dateStr].push(ev);
  }

  // Determine today/yesterday labels
  var now = new Date();
  var todayPT = now.toLocaleDateString('en-US', {timeZone: 'America/Los_Angeles', year: 'numeric', month: '2-digit', day: '2-digit'});
  var yesterday = new Date(now.getTime() - 86400000);
  var yesterdayPT = yesterday.toLocaleDateString('en-US', {timeZone: 'America/Los_Angeles', year: 'numeric', month: '2-digit', day: '2-digit'});

  var html = '';
  for (var g = 0; g < groupOrder.length; g++) {
    var dateKey = groupOrder[g];
    var label = dateKey === todayPT ? 'TODAY' : dateKey === yesterdayPT ? 'YESTERDAY' : dateKey;
    html += '<div class="shadow-tl-day">' + label + '</div>';
    var evs = groups[dateKey];
    for (var j = 0; j < evs.length; j++) {
      var ev = evs[j];
      var feature = ev.feature || 'unknown';
      var icon = feature === 'sweep_depth' ? '&#x1F4CF;' : feature === 'mode1' ? '&#x1F534;' : feature === 'velocity_short' ? '&#x26A1;' : '&#x1F50D;';
      var desc;
      if (feature === 'sweep_depth') {
        desc = describeSweepEvent(ev);
      } else if (feature === 'mode1') {
        desc = describeMode1Event(ev);
      } else if (feature === 'velocity_short') {
        desc = describeVelocityEvent(ev);
      } else {
        desc = {headline: ev.details || feature, detail: ev.would_have_done || ''};
      }

      var resultClass = '';
      if (feature === 'velocity_short' && ev.would_have_pnl != null) {
        resultClass = ev.would_have_pnl >= 0 ? 'win' : 'loss';
      }

      html += '<div class="shadow-tl-row">';
      html += '<div class="shadow-tl-time">' + formatShadowTimePT(ev.timestamp) + '</div>';
      html += '<div class="shadow-tl-icon">' + icon + '</div>';
      html += '<div class="shadow-tl-body">';
      html += '<div class="shadow-tl-headline">' + desc.headline + '</div>';
      html += '<div class="shadow-tl-detail">' + desc.detail + '</div>';

      // Show outcome if available
      if (ev.outcome) {
        var pnl = ev.outcome_pnl || 0;
        var outcomeIcon, outcomeText, outcomeColor;
        if (ev.outcome === 'target_hit') {
          outcomeIcon = '&#x1F3AF;';
          outcomeText = 'TARGET HIT';
          outcomeColor = '#3fb950';
        } else if (ev.outcome === 'stop_hit') {
          outcomeIcon = '&#x1F6D1;';
          outcomeText = 'STOP HIT';
          outcomeColor = '#f85149';
        } else {
          outcomeIcon = '&#x23F0;';
          outcomeText = 'TIMED OUT';
          outcomeColor = '#8b949e';
        }
        var pnlStr = (pnl >= 0 ? '+' : '') + pnl.toFixed(1) + ' pts';
        var pnlColor = pnl >= 0 ? '#3fb950' : '#f85149';
        html += '<div style="margin-top:4px;padding:4px 8px;border-radius:4px;background:#21262d;display:inline-block;">'
          + '<span>' + outcomeIcon + ' ' + outcomeText + '</span>'
          + '<span style="color:' + pnlColor + ';font-weight:700;margin-left:8px;">' + pnlStr + '</span>'
          + (ev.outcome_bars ? '<span style="color:#8b949e;margin-left:8px;">(' + ev.outcome_bars + ' bars)</span>' : '')
          + '</div>';
      } else if (ev.entry_price && ev.stop_price) {
        html += '<div style="margin-top:4px;color:#8b949e;font-size:11px;">&#x23F3; Tracking outcome...</div>';
      }

      html += '</div></div>';
    }
  }
  container.innerHTML = html;
}

function renderShadowStatsBar(data) {
  var stats = data.summary || {};
  var total = stats.total_today || 0;
  var vshort = stats.velocity_short || {};
  var sweep = stats.sweep_depth || {};

  var el = document.getElementById('ss-total');
  if (el) el.textContent = total;

  var vsEl = document.getElementById('ss-vsignals');
  if (vsEl) vsEl.textContent = (vshort.count || 0) + (vshort.total_pnl != null ? ' (' + (vshort.total_pnl >= 0 ? '+' : '') + vshort.total_pnl.toFixed(1) + ' pts)' : '');

  var savEl = document.getElementById('ss-savings');
  if (savEl) {
    // Estimate savings from sweep depth sizing reductions
    if (sweep.count > 0 && sweep.avg_size != null) {
      var avgReduction = Math.round((1 - sweep.avg_size) * 100);
      savEl.textContent = '~' + avgReduction + '% size reduction avg';
    } else {
      savEl.textContent = '--';
    }
  }
}
</script>
</body>
</html>""")


# Level type tooltip descriptions
LEVEL_TOOLTIPS = {
    "Prior Day Low": "Yesterday's low — key Mancini support level",
    "Prior Day High": "Yesterday's high — key Mancini resistance level",
    "Cluster Low": "3+ touches within 1 pt — strong support shelf",
    "Cluster High": "3+ touches within 1 pt — strong resistance shelf",
    "Swing Low": "Local low from swing point detection",
    "Swing High": "Local high from swing point detection",
    "Multi Hour Low": "Low that produced a 20+ pt rally — major support",
    "Multi Hour High": "High that produced a 20+ pt selloff — major resistance",
    "Horizontal Sr": "Price tested multiple times — horizontal S/R",
}


def read_status() -> dict:
    """Read the status JSON file written by the bot."""
    try:
        path = Path(STATUS_FILE)
        if path.exists():
            data = json.loads(path.read_text())
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def read_recent_logs(n=50) -> list[str]:
    """Read the last N lines from the bot log file."""
    try:
        path = Path(LOG_FILE)
        if path.exists():
            lines = path.read_text().splitlines()
            return lines[-n:]
    except OSError:
        pass
    return []


def classify_log_line(line: str) -> str:
    """Return CSS class for a log line."""
    if "ENTRY:" in line or "TRADE ALERT | ENTRY" in line:
        return "log-entry"
    if "EXIT:" in line or "TRADE ALERT | EXIT" in line:
        return "log-exit"
    if "SIGNAL" in line or "PHANTOM" in line:
        return "log-signal"
    if "BAR #" in line:
        return "log-bar"
    return ""


def render_position(status: dict) -> str:
    """Render current position HTML."""
    pos = status.get("position")
    if not pos or not pos.get("is_open"):
        return """
        <div class="position-banner pos-flat">
          <span class="pos-direction">FLAT</span>
          <span class="pos-pattern">No open position</span>
        </div>"""

    direction = pos.get("direction", "long")
    css = "pos-long" if direction == "long" else "pos-short"
    entry = pos.get("entry_price", 0)
    stop = pos.get("stop_price", 0)
    target = pos.get("target_price", 0)
    pattern = pos.get("pattern", "?")
    unrealized = pos.get("unrealized_pnl", 0)
    pnl_css = "green" if unrealized >= 0 else "red"
    contracts = pos.get("contracts", 1)
    risk = pos.get("risk_pts", 0)
    reward = pos.get("reward_pts", 0)
    rr = reward / risk if risk > 0 else 0
    phase = pos.get("phase", "INITIAL")
    t1_hit = pos.get("t1_hit", False)
    mfe = pos.get("mfe_pts", 0)
    mae = pos.get("mae_pts", 0)
    trail_stop = pos.get("trail_stop")
    trail_dist = pos.get("trail_distance_pts")

    # Phase badge
    if phase == "AFTER_T1":
        phase_badge = '<span style="background:#238636;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">RUNNER (T1 HIT)</span>'
    elif phase == "AFTER_T2":
        phase_badge = '<span style="background:#1f6feb;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">RUNNER (T2 HIT)</span>'
    elif phase == "RUNNER":
        phase_badge = '<span style="background:#d29922;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">RUNNER</span>'
    else:
        phase_badge = ""

    # Trail stop display
    trail_html = ""
    if trail_stop is not None and trail_dist is not None:
        trail_html = f"""
    <div class="stat-row"><span class="stat-label">Trail Stop</span><span class="stat-value yellow" style="font-size:16px;font-weight:700;">📏 {trail_stop:.2f}</span></div>
    <div class="stat-row"><span class="stat-label">Trail Distance</span><span class="stat-value">{trail_dist:.1f} pts from price</span></div>"""

    # MFE/MAE display
    mfe_mae_html = ""
    if mfe > 0 or mae > 0:
        mfe_mae_html = f"""
    <div class="stat-row"><span class="stat-label">Best (MFE)</span><span class="stat-value green">+{mfe:.1f} pts</span></div>
    <div class="stat-row"><span class="stat-label">Worst (MAE)</span><span class="stat-value red">-{mae:.1f} pts</span></div>"""

    return f"""
    <div class="position-banner {css}">
      <span class="pos-direction">{direction.upper()} x{contracts}</span>
      <span class="pos-pattern">{pattern} {phase_badge}</span>
    </div>
    <div class="stat-row"><span class="stat-label">Entry</span><span class="stat-value">{entry:.2f}</span></div>
    <div class="stat-row"><span class="stat-label">{"Trail Stop" if trail_stop else "Stop"}</span><span class="stat-value {"yellow" if trail_stop else "red"}" style="font-size:16px">{stop:.2f} ({risk:.1f} pts)</span></div>
    <div class="stat-row"><span class="stat-label">Target</span><span class="stat-value green">{target:.2f} {"✅ HIT" if t1_hit else f"({reward:.1f} pts)"}</span></div>
    <div class="stat-row"><span class="stat-label">R:R</span><span class="stat-value">{rr:.1f}</span></div>
    <div class="stat-row"><span class="stat-label">Unrealized</span><span class="stat-value {pnl_css}" style="font-size:18px">{unrealized:+.1f} pts</span></div>{trail_html}{mfe_mae_html}
    """


def render_levels(status: dict) -> str:
    """Render active levels table sorted by proximity with type tooltips."""
    levels = status.get("levels", [])
    if not levels:
        return '<div class="muted" style="padding:20px; text-align:center;">No levels detected yet</div>'

    nearest = levels[:15]
    rows = ""
    for lv in nearest:
        price = lv.get("price", 0)
        ltype = lv.get("type", "?")
        touches = lv.get("touches", 0)
        dist = lv.get("distance", 0)

        is_support = "LOW" in ltype or "SUPPORT" in ltype
        is_resistance = "HIGH" in ltype or "RESISTANCE" in ltype
        price_css = "green" if is_support else ("red" if is_resistance else "blue")
        clean_type = ltype.replace("_", " ").title()
        dist_str = f"{dist:+.2f}" if dist != 0 else "AT"
        dist_css = "green" if dist < 0 else "red" if dist > 0 else "yellow"

        # Look up tooltip for this level type
        tooltip = LEVEL_TOOLTIPS.get(clean_type, "")
        tip_html = f'<span class="level-tip">{tooltip}</span>' if tooltip else ""

        rows += (
            f"<tr>"
            f"<td class='{price_css}'>{price:.2f}</td>"
            f"<td><span class='level-tag'>{clean_type}{tip_html}</span></td>"
            f"<td style='text-align:center'>{touches}</td>"
            f"<td class='{dist_css}' style='text-align:right'>{dist_str}</td>"
            f"</tr>\n"
        )

    return f"""
    <div style="max-height:350px; overflow-y:auto;">
    <table>
      <tr>
        <th>Price</th>
        <th>Type</th>
        <th style="text-align:center" class="th-tip">Touches
          <span class="th-tip-text">Times price tested this level. More touches = stronger. Levels within 1 pt merge their touch counts.</span>
        </th>
        <th style="text-align:right">Dist</th>
      </tr>
      {rows}
    </table>
    </div>"""


def render_trades(status: dict) -> str:
    """Render today's trade history with bypass gate badges."""
    trades = status.get("trades", [])
    if not trades:
        return '<div class="muted" style="padding:20px; text-align:center;">No trades today</div>'

    rows = ""
    for t in trades:
        direction = t.get("direction", "long")
        css = "green" if direction == "long" else "red"
        pnl = t.get("pnl_pts", 0)
        pnl_css = "green" if pnl > 0 else "red" if pnl < 0 else "dim"

        # Bypass gate badge
        gate_bypassed = t.get("gate_bypassed", [])
        production_valid = t.get("production_would_take", True)
        if gate_bypassed:
            gate_names = ", ".join(gate_bypassed) if isinstance(gate_bypassed, list) else str(gate_bypassed)
            badge = f'<span class="bypass-tag bypass-collection" title="{gate_names}">COLLECTION</span>'
        elif not production_valid:
            badge = '<span class="bypass-tag bypass-collection">COLLECTION</span>'
        else:
            badge = '<span class="bypass-tag bypass-production">PROD</span>'

        rows += (
            f"<tr>"
            f"<td>{t.get('time', '?')}</td>"
            f"<td class='{css}'>{direction[0].upper()}</td>"
            f"<td><span class='level-tag'>{t.get('pattern', '?')}</span> {badge}</td>"
            f"<td>{t.get('entry_price', 0):.2f}</td>"
            f"<td>{t.get('exit_price', 0):.2f}</td>"
            f"<td class='{pnl_css}' style='font-weight:700'>{pnl:+.1f}</td>"
            f"<td class='dim'>{t.get('exit_reason', '?')}</td>"
            f"</tr>\n"
        )

    return f"""
    <table>
      <tr><th>Time</th><th>Dir</th><th>Pattern</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Reason</th></tr>
      {rows}
    </table>"""


def render_near_misses(status: dict) -> str:
    """Render near-miss setups that almost triggered."""
    near_misses = status.get("near_misses", [])
    if not near_misses:
        return '<div class="muted" style="padding:12px; text-align:center; font-size:12px;">No near misses detected yet</div>'

    html = ""
    for nm in near_misses:
        reason = nm.get("failure_reason", "unknown")
        level = nm.get("level_price", 0)
        achieved = nm.get("achieved", {})
        required = nm.get("required", {})
        timestamp = nm.get("timestamp", "?")

        # Build human-readable description
        if reason == "acceptance_timeout":
            held = achieved.get("hold_bars", "?")
            needed = required.get("hold_bars", "?")
            desc = f"Held {held}/{needed} bars needed (acceptance timeout)"
        elif reason == "dip_too_deep":
            dip = achieved.get("dip_pts", "?")
            mx = required.get("max_dip_pts", "?")
            desc = f"Dip {dip} pts, max allowed {mx} pts (too deep)"
        elif reason == "sweep_too_deep":
            depth = achieved.get("sweep_depth", "?")
            mx = required.get("max_depth", "?")
            desc = f"Sweep {depth} pts deep, max {mx} pts (too deep)"
        elif reason == "rr_too_low":
            rr = achieved.get("rr_ratio", "?")
            mn = required.get("min_rr_ratio", "?")
            desc = f"R:R ratio {rr}, minimum {mn} (too low)"
        else:
            desc = reason

        # Outcome tracking
        outcome = nm.get("outcome")
        if outcome:
            if outcome.get("resolved"):
                result = outcome["result"]
                if "T1 HIT" in result:
                    outcome_html = f'<span class="green" style="font-weight:700">{result}</span>'
                    border_color = "#3fb950"
                else:
                    outcome_html = f'<span class="red" style="font-weight:700">{result}</span>'
                    border_color = "#f85149"
            else:
                hi = outcome.get("high_since", 0)
                lo = outcome.get("low_since", 0)
                entry = outcome.get("entry_price", 0)
                outcome_html = f'<span class="yellow" style="font-weight:700">TRACKING</span> (hi {hi:.2f}, lo {lo:.2f}, from {entry:.2f})'
                border_color = "#d29922"
        else:
            outcome_html = '<span class="dim">Awaiting outcome data</span>'
            border_color = "#d29922"

        html += f"""
        <div class="near-miss-item" style="border-left-color:{border_color}">
          <span class="near-miss-reason">NEAR MISS</span> &mdash; FB at {level:.2f}: {desc}
          <div style="margin-top:4px; font-size:12px;">Outcome: {outcome_html}</div>
          <div class="near-miss-detail">{timestamp}</div>
        </div>
        """

    return html


def render_phantoms(status: dict) -> str:
    """Render phantom (rejected) signals."""
    phantoms = status.get("phantoms", [])
    if not phantoms:
        return '<div class="muted" style="padding:20px; text-align:center;">No rejected signals yet</div>'

    rows = ""
    for p in phantoms:
        result = p.get("result", "OPEN")
        result_css = "green" if "T1 HIT" in result else "red" if "STOP HIT" in result else "yellow"
        reason = p.get("reject_reason", "?")
        if ":" in reason:
            reason = reason.split(":", 1)[1]

        tracking_span = '<span class="phantom-tag">TRACKING</span>'
        outcome = result if p.get("resolved") else tracking_span
        rows += (
            f"<tr class='phantom-row'>"
            f"<td><span class='level-tag'>{p.get('signal_type', '?')}</span></td>"
            f"<td>{p.get('entry_price', 0):.2f}</td>"
            f"<td class='dim'>{reason}</td>"
            f"<td class='{result_css}'>{outcome}</td>"
            f"</tr>\n"
        )

    return f"""
    <div style="max-height:250px; overflow-y:auto;">
    <table>
      <tr><th>Signal</th><th>Price</th><th>Rejected</th><th>Outcome</th></tr>
      {rows}
    </table>
    </div>"""


SUBSTACK_FILE = os.environ.get("SUBSTACK_FILE", "/app/logs/substack_comparison.json")


def read_substack() -> dict:
    """Read the substack comparison JSON written by the cron job."""
    try:
        path = Path(SUBSTACK_FILE)
        if path.exists():
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def render_substack(data: dict) -> str:
    """Render Mancini Substack highlights and level comparison."""
    if not data or data.get("error"):
        return '<div class="muted" style="padding:20px; text-align:center;">No Substack comparison available yet. Runs daily at 9 PM ET.</div>'

    title = data.get("post_title", "?")
    post_date = data.get("post_date", "?")
    match_rate = data.get("match_rate", 0)
    matched = data.get("matched_count", 0)
    mancini_count = data.get("mancini_levels_found", 0)
    engine_count = data.get("engine_levels_active", 0)
    timestamp = data.get("timestamp", "?")[:19]

    html = f"""
    <div style="margin-bottom:12px;">
      <div style="font-size:14px; font-weight:700; color:#f0f6fc; margin-bottom:4px;">{title}</div>
      <div class="dim" style="font-size:11px;">Published {post_date} &mdash; compared {timestamp}</div>
      <div style="margin-top:8px; display:flex; gap:16px; flex-wrap:wrap;">
        <div class="pattern-stat"><div class="pattern-stat-val blue">{mancini_count}</div><div class="pattern-stat-lbl">Mancini Levels</div></div>
        <div class="pattern-stat"><div class="pattern-stat-val blue">{engine_count}</div><div class="pattern-stat-lbl">Engine Levels</div></div>
        <div class="pattern-stat"><div class="pattern-stat-val green">{matched}</div><div class="pattern-stat-lbl">Matched</div></div>
        <div class="pattern-stat"><div class="pattern-stat-val {'green' if match_rate >= 50 else 'yellow' if match_rate >= 30 else 'red'}">{match_rate}%</div><div class="pattern-stat-lbl">Match Rate</div></div>
      </div>
    </div>
    """

    # Info blurb
    html += """
    <div style="margin-bottom:12px; padding:10px 12px; background:rgba(56,139,253,0.08); border:1px solid rgba(56,139,253,0.2); border-radius:6px;">
      <div style="font-size:11px; font-weight:700; color:#58a6ff; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">How the Engine Uses This</div>
      <div style="font-size:11px; color:#c9d1d9; line-height:1.6;">
        <b>Matched levels</b> &mdash; where both Mancini and the engine agree &mdash; carry the highest
        conviction for entries. <b>Mancini-only levels</b> highlight blind spots in the
        engine&rsquo;s level detector. His directional lean and trade setups provide context
        the engine can&rsquo;t derive from price alone. The <b>retrospective analysis</b> (runs next day
        at 9 PM ET) scores how each level actually performed &mdash; held, broken, or untested &mdash;
        to continuously calibrate both sources.
      </div>
    </div>
    """

    # Highlights
    highlights = data.get("highlights", [])
    if highlights:
        html += '<div style="margin-bottom:12px;">'
        html += '<div style="font-size:11px; font-weight:600; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Key Highlights</div>'
        for h in highlights[:15]:
            htype = h.get("type", "")
            text = h.get("text", "")[:300]
            if htype == "DIRECTIONAL LEAN":
                color = "#58a6ff"
                icon = "&#8593;&#8595;"
            elif htype == "TRADE SETUP":
                color = "#3fb950"
                icon = "&#9654;"
            elif htype == "TARGETS":
                color = "#d29922"
                icon = "&#9673;"
            elif htype == "KEY LEVEL":
                color = "#bc8cff"
                icon = "&#9644;"
            elif htype == "RISK / INVALIDATION":
                color = "#f85149"
                icon = "&#9888;"
            else:
                color = "#8b949e"
                icon = "&#8226;"

            html += f"""
            <div style="padding:6px 10px; background:#0d1117; border-radius:4px; margin-bottom:4px; border-left:3px solid {color}; font-size:12px; line-height:1.5;">
              <span style="color:{color}; font-weight:700; font-size:10px;">{icon} {htype}</span>
              <div style="color:#c9d1d9; margin-top:2px;">{text}</div>
            </div>
            """
        html += '</div>'

    # Matched levels
    comp = data.get("comparison", {})
    matched_list = comp.get("matched", [])
    if matched_list:
        html += '<div style="margin-bottom:12px;">'
        html += '<div style="font-size:11px; font-weight:600; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Matched Levels (Mancini &harr; Engine)</div>'
        html += '<table><tr><th>Mancini</th><th>Engine</th><th>Type</th><th>Touches</th><th>Dist</th></tr>'
        for m in matched_list:
            html += (
                f"<tr>"
                f"<td class='green'>{m['mancini_price']:.0f}</td>"
                f"<td class='blue'>{m['engine_price']:.2f}</td>"
                f"<td><span class='level-tag'>{m.get('engine_type', '?').replace('_', ' ').title()}</span></td>"
                f"<td style='text-align:center'>{m.get('engine_touches', 0)}</td>"
                f"<td class='dim'>{m.get('distance', 0):+.1f}</td>"
                f"</tr>"
            )
        html += '</table></div>'

    # Mancini-only levels (engine missed)
    mancini_only = comp.get("mancini_only", [])
    if mancini_only:
        html += '<div>'
        html += '<div style="font-size:11px; font-weight:600; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Mancini Mentioned (Engine Missed)</div>'
        items = ""
        for m in mancini_only[:12]:
            role = m.get("role", "?")
            role_css = "green" if role == "support" else "red" if role == "resistance" else "yellow" if role == "target" else "dim"
            ctx = m.get("context", "")[:80]
            items += f'<span class="level-tag" style="margin:2px; cursor:default;"><span class="{role_css}" style="font-weight:700">{m["price"]:.0f}</span> {role}<span class="level-tip">{ctx}</span></span> '
        html += f'<div style="line-height:2;">{items}</div>'
        html += '</div>'

    return html


RETRO_DIR = os.environ.get("RETRO_DIR", "/app/logs")


def read_retrospective() -> dict:
    """Read the most recent retrospective JSON."""
    try:
        retro_dir = Path(RETRO_DIR)
        files = sorted(retro_dir.glob("retrospective_*.json"), reverse=True)
        if files:
            return json.loads(files[0].read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def render_retrospective(data: dict) -> str:
    """Render the retrospective analysis section."""
    if not data:
        return '<div class="muted" style="padding:20px; text-align:center;">No retrospective data yet &mdash; runs nightly at 9:05 PM ET</div>'

    session_date = data.get("session_date", "?")
    summary = data.get("summary", {})
    level_scores = data.get("level_scores", [])
    trades_analysis = data.get("trades_analysis", [])
    missed = data.get("missed_opportunities", [])
    session = data.get("session", {})

    # Summary bar
    total = summary.get("total_levels", 0)
    tested = summary.get("tested", 0)
    held = summary.get("held", 0)
    broken = summary.get("broken", 0)
    manc_acc = summary.get("mancini_accuracy_pct", 0)
    eng_acc = summary.get("engine_accuracy_pct", 0)
    match_acc = summary.get("matched_accuracy_pct", 0)
    trades_held = summary.get("trades_at_held_levels", 0)
    trades_broken = summary.get("trades_at_broken_levels", 0)
    missed_count = summary.get("missed_with_5pt_bounce", 0)

    html = f"""
    <div style="margin-bottom:12px;">
      <div style="font-size:14px; font-weight:700; color:#f0f6fc; margin-bottom:4px;">Session {session_date}</div>
      <div class="dim" style="font-size:11px;">
        Range: {session.get('low', 0):.0f} &ndash; {session.get('high', 0):.0f}
        ({session.get('range', 0):.0f} pts) &bull;
        {session.get('bars', 0)} bars
      </div>
      <div style="margin-top:8px; display:flex; gap:12px; flex-wrap:wrap;">
        <div class="pattern-stat"><div class="pattern-stat-val blue">{total}</div><div class="pattern-stat-lbl">Levels</div></div>
        <div class="pattern-stat"><div class="pattern-stat-val green">{tested}</div><div class="pattern-stat-lbl">Tested</div></div>
        <div class="pattern-stat"><div class="pattern-stat-val green">{held}</div><div class="pattern-stat-lbl">Held</div></div>
        <div class="pattern-stat"><div class="pattern-stat-val red">{broken}</div><div class="pattern-stat-lbl">Broken</div></div>
        <div class="pattern-stat"><div class="pattern-stat-val {'green' if match_acc >= 70 else 'yellow'}">{match_acc:.0f}%</div><div class="pattern-stat-lbl">Matched Acc</div></div>
      </div>
    </div>
    """

    # Accuracy comparison
    html += f"""
    <div style="margin-bottom:12px; display:flex; gap:16px; flex-wrap:wrap;">
      <div style="padding:8px 12px; background:#0d1117; border-radius:6px; flex:1; min-width:120px;">
        <div class="dim" style="font-size:10px; text-transform:uppercase;">Mancini Accuracy</div>
        <div style="font-size:18px; font-weight:700;" class="{'green' if manc_acc >= 60 else 'yellow' if manc_acc >= 40 else 'red'}">{manc_acc:.0f}%</div>
      </div>
      <div style="padding:8px 12px; background:#0d1117; border-radius:6px; flex:1; min-width:120px;">
        <div class="dim" style="font-size:10px; text-transform:uppercase;">Engine Accuracy</div>
        <div style="font-size:18px; font-weight:700;" class="{'green' if eng_acc >= 60 else 'yellow' if eng_acc >= 40 else 'red'}">{eng_acc:.0f}%</div>
      </div>
      <div style="padding:8px 12px; background:#0d1117; border-radius:6px; flex:1; min-width:120px;">
        <div class="dim" style="font-size:10px; text-transform:uppercase;">Trades @ Held</div>
        <div style="font-size:18px; font-weight:700; color:#3fb950;">{trades_held}</div>
      </div>
      <div style="padding:8px 12px; background:#0d1117; border-radius:6px; flex:1; min-width:120px;">
        <div class="dim" style="font-size:10px; text-transform:uppercase;">Trades @ Broken</div>
        <div style="font-size:18px; font-weight:700; color:#f85149;">{trades_broken}</div>
      </div>
    </div>
    """

    # Level scorecard
    if level_scores:
        html += '<div style="margin-bottom:12px;">'
        html += '<div style="font-size:11px; font-weight:600; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Level Scorecard</div>'
        html += '<table><tr><th>Price</th><th>Source</th><th>Touches</th><th>Bounces</th><th>Breaks</th><th>Max Bounce</th><th>Verdict</th></tr>'
        for lv in level_scores[:20]:
            verdict = lv.get("verdict", "?")
            if verdict == "HELD":
                v_css = "color:#3fb950; font-weight:700;"
            elif verdict == "BROKEN":
                v_css = "color:#f85149; font-weight:700;"
            else:
                v_css = "color:#8b949e;"
            source = lv.get("source", "?")
            src_css = "color:#bc8cff;" if source == "both" else "color:#58a6ff;" if source == "mancini" else ""
            html += (
                f"<tr>"
                f"<td style='font-weight:700;'>{lv.get('price', 0):.0f}</td>"
                f"<td style='{src_css}'>{source}</td>"
                f"<td style='text-align:center;'>{lv.get('touches', 0)}</td>"
                f"<td style='text-align:center;'>{lv.get('bounces', 0)}</td>"
                f"<td style='text-align:center;'>{lv.get('breaks', 0)}</td>"
                f"<td style='text-align:center;'>{lv.get('max_bounce', 0):.1f}</td>"
                f"<td style='{v_css}'>{verdict}</td>"
                f"</tr>"
            )
        html += '</table></div>'

    # Trade-level matches
    if trades_analysis:
        html += '<div style="margin-bottom:12px;">'
        html += '<div style="font-size:11px; font-weight:600; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Trades &harr; Levels</div>'
        for t in trades_analysis:
            pnl = t.get("pnl", 0)
            held = t.get("level_held", False)
            manc = t.get("mancini_mentioned", False)
            pnl_css = "green" if pnl > 0 else "red"
            html += f"""
            <div style="padding:6px 10px; background:#0d1117; border-radius:4px; margin-bottom:4px; border-left:3px solid {'#3fb950' if pnl > 0 else '#f85149'}; font-size:12px;">
              <span style="font-weight:700;">{t.get('pattern', '?')}</span>
              @ {t.get('entry', 0):.0f} (level {t.get('level', 0):.0f})
              &rarr; <span class="{pnl_css}">{pnl:+.1f} pts</span>
              {'<span style="color:#3fb950;"> &#10003; Level held</span>' if held else '<span style="color:#f85149;"> &#10007; Level broke</span>'}
              {'<span style="color:#bc8cff;"> &#9733; Mancini level</span>' if manc else ''}
            </div>"""
        html += '</div>'

    # Missed opportunities
    if missed:
        html += '<div>'
        html += '<div style="font-size:11px; font-weight:600; color:#d29922; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Missed Opportunities</div>'
        for m in missed[:8]:
            html += f"""
            <div style="padding:6px 10px; background:#0d1117; border-radius:4px; margin-bottom:4px; border-left:3px solid #d29922; font-size:12px;">
              <span style="font-weight:700;">{m.get('level', 0):.0f}</span>
              ({m.get('source', '?')}) &mdash;
              {m.get('verdict', '?')}, bounce {m.get('max_bounce', 0):.1f} pts
              <span class="dim"> &mdash; {m.get('reason', '?')}</span>
            </div>"""
        html += '</div>'

    return html


def render_logs(lines: list[str]) -> str:
    """Render recent log lines with syntax highlighting."""
    if not lines:
        return '<div class="muted" style="padding:20px; text-align:center;">No logs available</div>'

    filtered = [l for l in lines if "DEBUG" not in l]
    filtered = filtered[-40:]

    html = ""
    for line in filtered:
        css = classify_log_line(line)
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html += f'<div class="log-line {css}">{safe}</div>\n'
    return html


def _compute_market_status() -> dict:
    """Compute current ES/MES market status from real time (ET)."""
    from datetime import datetime, time as dt_time
    import pytz

    et = pytz.timezone("US/Eastern")
    now = datetime.now(et)
    t = now.time()
    wd = now.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    pt = pytz.timezone("US/Pacific")
    now_pt = now.astimezone(pt)

    # Weekend check
    if wd == 5:  # Saturday
        return {
            "label": "MARKET CLOSED", "css": "mh-weekend",
            "detail": f"Weekend &mdash; {now_pt.strftime('%A %I:%M %p PT')}",
            "next": "Opens Sunday 3:00 PM PT (6:00 PM ET)",
        }
    if wd == 6 and t < dt_time(18, 0):  # Sunday before 6 PM ET
        open_pt = "3:00 PM PT"
        return {
            "label": "MARKET CLOSED", "css": "mh-weekend",
            "detail": f"Weekend &mdash; {now_pt.strftime('%A %I:%M %p PT')}",
            "next": f"Opens today at {open_pt} (6:00 PM ET)",
        }
    if wd == 4 and t >= dt_time(17, 0):  # Friday after 5 PM ET
        return {
            "label": "MARKET CLOSED", "css": "mh-weekend",
            "detail": f"Weekend &mdash; {now_pt.strftime('%A %I:%M %p PT')}",
            "next": "Opens Sunday 3:00 PM PT (6:00 PM ET)",
        }

    # Daily break
    if dt_time(17, 0) <= t < dt_time(18, 0):
        return {
            "label": "DAILY BREAK", "css": "mh-break",
            "detail": f"CME maintenance &mdash; {now_pt.strftime('%I:%M %p PT')}",
            "next": "Reopens at 3:00 PM PT (6:00 PM ET)",
        }

    # Market is open — determine which window
    if dt_time(18, 0) <= t <= dt_time(23, 59) or dt_time(0, 0) <= t < dt_time(2, 0):
        if dt_time(18, 0) <= t < dt_time(22, 0):
            window = "Globex Evening (time gate bypassed)"
        else:
            window = "Globex Overnight"
    elif dt_time(2, 0) <= t < dt_time(6, 0):
        window = "European Session (time gate bypassed)"
    elif dt_time(6, 0) <= t < dt_time(9, 30):
        window = "US Pre-Market"
    elif dt_time(9, 30) <= t < dt_time(11, 0):
        window = "RTH Morning (Prime Window)"
    elif dt_time(11, 0) <= t < dt_time(13, 0):
        window = "RTH Midday"
    elif dt_time(13, 0) <= t < dt_time(15, 0):
        window = "Chop Zone (time gate bypassed)"
    elif dt_time(15, 0) <= t < dt_time(16, 50):
        window = "RTH Afternoon (FB Only)"
    elif dt_time(16, 50) <= t < dt_time(17, 0):
        window = "EOD Settle / Flatten"
    else:
        window = "Globex"

    return {
        "label": "MARKET OPEN", "css": "mh-open",
        "detail": f"{window} &mdash; {now_pt.strftime('%I:%M %p PT')}",
        "next": "Daily break at 2:00 PM PT (5:00 PM ET)",
    }


def read_all_trades() -> list:
    """Read all trades from trades.jsonl, pair entries with exits."""
    path = Path(TRADES_FILE)
    if not path.exists():
        return []
    entries = {}  # key: (session_date, entry_price, pattern_type) -> entry record
    partials = {}  # key -> list of partial_exit records
    trades = []
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt = rec.get("event")
            if evt == "entry":
                key = (rec.get("session_date"), rec.get("entry_price"), rec.get("pattern_type"))
                entries[key] = rec
            elif evt == "partial_exit":
                key = (rec.get("session_date"), rec.get("entry_price"), rec.get("pattern_type"))
                partials.setdefault(key, []).append(rec)
            elif evt == "exit":
                key = (rec.get("session_date"), rec.get("entry_price"), rec.get("pattern_type"))
                entry_rec = entries.pop(key, {})
                partial_list = partials.pop(key, [])
                signal = entry_rec.get("signal", {})
                # Build partial exits summary for display
                exits = []
                for p in partial_list:
                    exits.append({
                        "time": p.get("timestamp", ""),
                        "price": p.get("exit_price", 0),
                        "contracts": p.get("contracts", 0),
                        "pnl_pts": p.get("pnl_pts", 0),
                        "reason": p.get("exit_reason", ""),
                        "new_stop": p.get("new_stop", 0),
                    })
                # Final exit (runner stop or full close)
                final_contracts = rec.get("contracts", 1)
                # If we have partials, the final exit contracts = total - sum(partial contracts)
                if exits:
                    partial_qty = sum(e.get("contracts", 0) for e in exits)
                    final_contracts = max(1, (entry_rec.get("contracts", None) or entry_rec.get("total_contracts", 2) or 2) - partial_qty)
                exits.append({
                    "time": rec.get("timestamp", ""),
                    "price": rec.get("exit_price", 0),
                    "contracts": final_contracts,
                    "pnl_pts": round((rec.get("pnl_pts", 0) or 0) - sum(e.get("pnl_pts", 0) for e in exits[:-1]), 2),
                    "reason": rec.get("exit_reason", "?"),
                    "new_stop": 0,
                })
                total_contracts = entry_rec.get("contracts", None) or entry_rec.get("total_contracts", None) or rec.get("contracts", 1)
                # Use actual entry timestamp for the date column (not session_date)
                # so evening Globex trades show the correct calendar date
                _entry_ts = entry_rec.get("timestamp", "")
                _trade_date = _entry_ts[:10] if len(_entry_ts) >= 10 else rec.get("session_date", "")
                trades.append({
                    "date": _trade_date,
                    "session_date": rec.get("session_date", ""),
                    "entry_time": _entry_ts,
                    "exit_time": rec.get("timestamp", ""),
                    "direction": entry_rec.get("direction", rec.get("direction", "long")),
                    "pattern": entry_rec.get("pattern_type", rec.get("pattern_type", "?")),
                    "entry_price": rec.get("entry_price", 0),
                    "exit_price": rec.get("exit_price", 0),
                    "stop": signal.get("stop", 0),
                    "target_1": signal.get("target_1", 0),
                    "target_2": signal.get("target_2", 0),
                    "rr_ratio": signal.get("rr_ratio", 0),
                    "pnl_pts": rec.get("pnl_pts", 0),
                    "pnl_dollars": rec.get("pnl_dollars", 0),
                    "contracts": total_contracts,
                    "exit_reason": rec.get("exit_reason", "?"),
                    "level_type": signal.get("level_type", ""),
                    "level_price": signal.get("level_price", 0),
                    "regime": entry_rec.get("regime", {}).get("direction", "") if isinstance(entry_rec.get("regime"), dict) else str(entry_rec.get("regime", "")),
                    "session_window": entry_rec.get("session_window", ""),
                    "gate_bypassed": entry_rec.get("gate_bypassed", []),
                    "production_would_take": entry_rec.get("production_would_take", True),
                    "exits": exits,
                })
    except OSError:
        return []
    return sorted(trades, key=lambda t: t.get("exit_time", ""), reverse=True)


def read_nightly_reports() -> dict:
    """Read nightly_reports.jsonl and return last 14 reports."""
    log_dir = os.environ.get("LOG_PATH", os.environ.get("LOG_FILE", "/app/logs/bot.log"))
    log_dir_path = Path(log_dir)
    if log_dir_path.suffix:
        log_dir_path = log_dir_path.parent
    report_path = log_dir_path / "nightly_reports.jsonl"

    reports = []
    if report_path.exists():
        try:
            for line in report_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    reports.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass

    return {"reports": reports[-14:]}


def read_shadow_events() -> dict:
    """Read shadow_trades.jsonl and return last 100 events with summary stats."""
    log_dir = os.environ.get("LOG_PATH", os.environ.get("LOG_FILE", "/app/logs/bot.log"))
    # LOG_PATH or LOG_FILE points to a file; get the parent directory
    log_dir_path = Path(log_dir)
    if log_dir_path.suffix:  # it's a file path, get parent
        log_dir_path = log_dir_path.parent
    shadow_path = log_dir_path / "shadow_trades.jsonl"

    events = []
    if shadow_path.exists():
        try:
            for line in shadow_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass

    # Separate outcomes from signal events and attach outcomes to their signals
    outcomes = [e for e in events if e.get("event") == "shadow_outcome"]
    signal_events = [e for e in events if e.get("event") != "shadow_outcome"]

    # Match outcomes to signals by feature + timestamp
    for sig in signal_events:
        sig_ts = sig.get("timestamp", "")
        sig_feat = sig.get("feature", "")
        for out in outcomes:
            if out.get("feature") == sig_feat and out.get("timestamp") == sig_ts:
                sig["outcome"] = out.get("outcome")
                sig["outcome_pnl"] = out.get("pnl_pts")
                sig["outcome_price"] = out.get("outcome_price")
                sig["outcome_bars"] = out.get("bars_tracked")
                sig["outcome_mfe"] = out.get("mfe_pts")
                break

    # Only return last 100 signal events (not raw outcomes)
    events = signal_events[-100:]

    # Compute today's summary
    from datetime import datetime, timezone
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        et = pytz.timezone("America/New_York")
    today_str = datetime.now(tz=et).strftime("%Y-%m-%d")

    today_events = []
    for ev in events:
        ts = ev.get("timestamp", "")
        if ts.startswith(today_str):
            today_events.append(ev)

    # Aggregate by feature
    sweep_events = [e for e in today_events if e.get("feature") == "sweep_depth"]
    mode1_events = [e for e in today_events if e.get("feature") == "mode1"]
    vshort_events = [e for e in today_events if e.get("feature") == "velocity_short"]

    sweep_sizes = [e.get("suggested_size", 0) for e in sweep_events if e.get("suggested_size")]
    mode1_bars = [e.get("bars_in_mode1", 0) for e in mode1_events if e.get("bars_in_mode1")]
    vshort_pnls = [e.get("would_have_pnl", 0) for e in vshort_events if e.get("would_have_pnl") is not None]

    summary = {
        "total_today": len(today_events),
        "sweep_depth": {
            "count": len(sweep_events),
            "avg_size": sum(sweep_sizes) / len(sweep_sizes) if sweep_sizes else None,
        },
        "mode1": {
            "count": len(mode1_events),
            "total_bars": sum(mode1_bars) if mode1_bars else None,
        },
        "velocity_short": {
            "count": len(vshort_events),
            "total_pnl": sum(vshort_pnls) if vshort_pnls else None,
        },
    }

    return {"events": events, "summary": summary}


def build_page() -> str:
    """Build the full dashboard HTML page."""
    status = read_status()
    logs = read_recent_logs(100)

    connected = status.get("connected", False)
    pnl = status.get("daily_pnl_pts", 0)
    session_window = status.get("session_window", {})
    regime = status.get("regime", "\u2014")
    last_price = status.get("last_price", 0)
    is_done = status.get("is_done_for_day", False)
    regime_longs = status.get("regime_longs", True)
    regime_shorts = status.get("regime_shorts", False)

    if regime in ("BULLISH", "BULL"):
        regime_css = "green"
    elif regime in ("BEARISH", "BEAR"):
        regime_css = "red"
    else:
        regime_css = "yellow"

    fb_active = regime_longs
    lr_active = regime_longs
    bd_active = regime_shorts

    bars_json = json.dumps(status.get("bars", []))
    levels_json = json.dumps(status.get("levels", []))

    regime_daily_bars = status.get("regime_daily_bars", 0)

    # Bypass mode banner
    bypass_mode = status.get("bypass_mode", False)
    if bypass_mode:
        bypass_banner_html = """
        <div class="bypass-banner">
          <span class="bypass-banner-icon">&#9888;</span>
          <span class="bypass-banner-text">COLLECTION MODE ACTIVE</span>
          <span class="bypass-banner-desc">&mdash; Engine takes ALL setups regardless of session windows. Trades marked COLLECTION would be skipped in production.</span>
        </div>"""
    else:
        bypass_banner_html = ""

    # Market hours status (computed from real time, not from bot status)
    mkt = _compute_market_status()

    return HTML_TEMPLATE.substitute(
        market_status_label=mkt["label"],
        market_status_css=mkt["css"],
        market_status_detail=mkt["detail"],
        market_next=mkt["next"],
        connection_status="CONNECTED" if connected else "DISCONNECTED",
        connection_css="green" if connected else "red",
        dot_class="dot-connected" if connected else "dot-disconnected",
        session_label=session_window.get("label", "\u2014"),
        session_css=session_window.get("css", "session-closed"),
        session_detail=session_window.get("detail", ""),
        last_update_pst=status.get("last_update_pst", "\u2014"),
        account_name=status.get("account_name", ""),
        symbol=status.get("symbol", "MES"),
        last_price=f"{last_price:.2f}" if last_price > 0 else "\u2014",
        session_high=f"{status.get('session_high', 0):.2f}" if status.get("session_high", 0) > 0 else "\u2014",
        session_low=f"{status.get('session_low', 0):.2f}" if status.get("session_low", 999999) < 999999 else "\u2014",
        bar_count=status.get("bar_count", 0),
        last_bar_pst=status.get("last_bar_pst", "\u2014"),
        last_bar_et=status.get("last_bar_et", "\u2014"),
        daily_pnl=f"{pnl:+.1f} pts" if pnl != 0 else "0.0 pts",
        pnl_class="green" if pnl > 0 else ("red" if pnl < 0 else "dim"),
        trades_today=status.get("trades_today", 0),
        max_trades=status.get("max_trades", 4),
        winners=status.get("winners", 0),
        losers=status.get("losers", 0),
        account_balance=status.get("account_balance", "\u2014"),
        account_equity=status.get("account_equity", "\u2014"),
        status_badge='<span class="done-badge">DONE FOR DAY</span>' if is_done else '<span class="active-badge">ACTIVE</span>',
        total_logged=status.get("total_logged_trades", 0),
        regime=regime,
        regime_css=regime_css,
        regime_daily_bars=regime_daily_bars,
        longs_enabled="YES" if regime_longs else "NO",
        longs_css="green" if regime_longs else "red",
        shorts_enabled="YES" if regime_shorts else "NO",
        shorts_css="green" if regime_shorts else "red",
        session_date=status.get("session_date", "\u2014"),
        trading_active="YES" if session_window.get("trading", False) else "NO",
        trading_css="green" if session_window.get("trading", False) else "red",
        fb_card_class="active" if fb_active else "disabled",
        fb_badge_class="badge-active" if fb_active else "badge-blocked",
        fb_status="ACTIVE" if fb_active else "BLOCKED",
        lr_card_class="active" if lr_active else "disabled",
        lr_badge_class="badge-active" if lr_active else "badge-blocked",
        lr_status="ACTIVE" if lr_active else "BLOCKED",
        bd_card_class="active" if bd_active else "disabled",
        bd_badge_class="badge-active" if bd_active else "badge-blocked",
        bd_status="ACTIVE" if bd_active else "BLOCKED BY REGIME",
        bars_json=bars_json,
        levels_json=levels_json,
        bypass_banner_html=bypass_banner_html,
        position_html=render_position(status),
        levels_html=render_levels(status),
        trades_html=render_trades(status),
        near_misses_html=render_near_misses(status),
        phantoms_html=render_phantoms(status),
        substack_html=render_substack(read_substack()),
        retrospective_html=render_retrospective(read_retrospective()),
        log_html=render_logs(logs),
    )


def build_fragments() -> dict:
    """Build AJAX-refreshable fragments (status + pre-rendered HTML)."""
    status = read_status()
    logs = read_recent_logs(100)

    return {
        "status": status,
        "html": {
            "levels": render_levels(status),
            "trades": render_trades(status),
            "phantoms": render_phantoms(status),
            "position": render_position(status),
            "log": render_logs(logs),
            "near_misses": render_near_misses(status),
            "substack": render_substack(read_substack()),
            "retrospective": render_retrospective(read_retrospective()),
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the dashboard."""

    def do_GET(self):
        try:
            if self.path == "/" or self.path == "/dashboard":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(build_page().encode())
            elif self.path == "/api/status":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(read_status(), indent=2).encode())
            elif self.path == "/api/fragments":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(build_fragments()).encode())
            elif self.path == "/api/trades":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(read_all_trades(), default=str).encode())
            elif self.path == "/api/shadow":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(read_shadow_events(), default=str).encode())
            elif self.path == "/api/reports":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(read_nightly_reports(), default=str).encode())
            elif self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            # Browser closed connection before we finished writing — harmless
            pass
        except Exception as e:
            # Catch-all so the server thread never dies on unexpected errors
            try:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Internal error: {e}".encode())
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

    def log_message(self, format, *args):
        pass


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread so concurrent browser requests
    (HTML page + API calls) don't block each other and cause
    ConnectionResetError crashes."""
    daemon_threads = True


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), DashboardHandler)
    print(f"Dashboard running on http://0.0.0.0:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
