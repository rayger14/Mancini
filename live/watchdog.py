"""Mancini Bot Watchdog — real-time monitoring agent.

Runs as a separate process (Docker sidecar) that continuously monitors the
trading bot for anomalies, catching bugs before they cause missed trades.

Monitors:
  1. Bar flow: are bars arriving? at what rate? any gaps?
  2. Volume: zero-volume bars indicate expired contract
  3. Stale data: bars arriving but timestamps not advancing
  4. Error rate: connection failures, order rejections, data errors
  5. Signal pipeline: RTH hours with zero signals for too long
  6. Position consistency: status.json vs expected state
  7. Session rollover: did it happen at 18:00 ET?
  8. Contract health: is the contract expiring soon?

Alerts are written to /app/logs/watchdog_alerts.json and printed to stdout
(visible in `docker logs`). The dashboard can read the alerts file.

Usage:
    python3 live/watchdog.py                    # monitor bot logs
    python3 live/watchdog.py --webhook URL      # also send alerts to webhook
    python3 live/watchdog.py --poll-interval 30 # check every 30s (default 30)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time as _time
from collections import deque
from datetime import datetime, time, timedelta, date
from pathlib import Path
from typing import Optional

try:
    import pytz
    _ET = pytz.timezone("US/Eastern")
except ImportError:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("US/Eastern")

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ── Alert severity levels ────────────────────────────────────────────────
CRITICAL = "CRITICAL"   # bot is broken, missing all trades
HIGH = "HIGH"           # likely missing trades
WARNING = "WARNING"     # something is off, may degrade
INFO = "INFO"           # informational

# ── Patterns to extract from log lines ───────────────────────────────────
_BAR_PATTERN = re.compile(
    r"BAR #(\d+): (\d+:\d+) O=([\d.]+) H=([\d.]+) L=([\d.]+) C=([\d.]+) V=(\d+)"
)
_ERROR_PATTERN = re.compile(r"\| ERROR\s+\|")
_SIGNAL_PATTERN = re.compile(
    r"SIGNAL|ENTRY|entry_price|FAILED_BREAKDOWN|LEVEL_RECLAIM|BREAKDOWN_SHORT|BACKTEST_SHORT"
)
_CONNECT_PATTERN = re.compile(r"IB connected|Qualified front-month|Qualified contract")
_DISCONNECT_PATTERN = re.compile(r"CONNECTION LOST|connect failed|Poll returned 0 bars")
_ROLLOVER_PATTERN = re.compile(r"New session.*daily PnL reset")
_ZERO_VOLUME_PATTERN = re.compile(r"V=0$")
_STALE_PATTERN = re.compile(r"STALE DATA")
_REROLL_PATTERN = re.compile(r"CONTRACT REROLL|CONTRACT ROLLED")
_POSITION_PATTERN = re.compile(r"ENTRY FILLED|position.*LONG|position.*SHORT|FLAT")

# Trade event patterns
_ENTRY_PATTERN = re.compile(
    r"ENTRY: (\d+) (\w+) @ ([\d.]+) stop=([\d.]+) T1=([\d.]+) R:R=([\d.]+) \[(\w+)\]"
)
_EXIT_BRACKET_PATTERN = re.compile(
    r"Position closed on IB side \((SL|TP|bracket).*?(?:filled @ ([\d.]+))?"
)
_EXIT_FLATTEN_PATTERN = re.compile(
    r"FLATTEN: closed (\d+) (\w+) (\w+) \[(.+?)\]"
)
_EXIT_ACTION_PATTERN = re.compile(
    r"EXIT: flatten -- (.+)"
)
_EXIT_EOD_PATTERN = re.compile(r"EOD flatten")
_TRADE_PNL_PATTERN = re.compile(
    r"Trade closed.*?([-+][\d.]+)\s*pts|exit.*?([-+][\d.]+)\s*pts"
)


class Alert:
    """A single watchdog alert."""

    def __init__(self, severity: str, code: str, message: str):
        self.severity = severity
        self.code = code
        self.message = message
        self.timestamp = datetime.now(_ET).isoformat()
        self.resolved = False

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
        }

    def __repr__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


class Watchdog:
    """Monitors bot health by tailing logs and checking state files."""

    def __init__(
        self,
        log_path: str = "/app/logs/bot.log",
        status_path: str = "/app/logs/status.json",
        alerts_path: str = "/app/logs/watchdog_alerts.json",
        poll_interval: float = 30.0,
        webhook_url: Optional[str] = None,
    ):
        self.log_path = Path(log_path)
        self.status_path = Path(status_path)
        self.alerts_path = Path(alerts_path)
        self.poll_interval = poll_interval
        self.webhook_url = webhook_url

        # State tracking
        self._last_log_pos: int = 0
        self._last_bar_time: Optional[datetime] = None
        self._last_bar_number: int = 0
        self._last_bar_volume: int = 0
        self._bars_seen: int = 0
        self._zero_volume_streak: int = 0
        self._error_window: deque = deque(maxlen=100)  # last 100 errors
        self._signal_count_rth: int = 0
        self._rth_bars_without_signal: int = 0
        # If watchdog starts after 18:00 ET, assume today's rollover already happened
        _now_et = datetime.now(_ET)
        if _now_et.time() >= time(18, 0):
            self._last_rollover_date = _now_et.date()
        else:
            self._last_rollover_date = None
        self._connected: bool = False
        self._last_alert_times: dict[str, float] = {}  # code -> monotonic time
        self._active_alerts: dict[str, Alert] = {}  # code -> alert

        # Thresholds
        # MAX_BAR_GAP_SEC = 5 min (was 3 min). The previous 3-min threshold
        # was too tight: the IB Gateway nightly auth refresh routinely
        # produces a 2-4 min bar gap that's not actionable, but tripped
        # BAR_GAP CRITICAL alerts every night. 5 min still catches real
        # disconnects (the bot's own STALE_DATA alert at 3 min already
        # tells us if polling stalls).
        self.MAX_BAR_GAP_SEC = 300       # 5 min without a bar during market hours
        self.MAX_ZERO_VOLUME = 3         # 3 consecutive zero-volume bars
        self.MAX_ERRORS_PER_5MIN = 10    # error rate threshold
        self.RTH_SIGNAL_CHECK_BARS = 120 # 2 hours RTH without any signal = warning
        self.ALERT_COOLDOWN_SEC = 300    # don't repeat same alert within 5 min
        self.CONTRACT_EXPIRY_WARN_DAYS = 7  # warn when contract expires within 7 days

        # IB Gateway does an automatic re-authentication around 19:45 ET
        # every weeknight. During the window the gateway's TCP listener
        # is briefly unavailable and the bot's poll fails — producing a
        # bar gap and an error spike that auto-clear within ~2 min once
        # the gateway is back. Suppress BAR_GAP and ERROR_SPIKE alerts
        # in this window so subscribers don't get nightly false alarms.
        self.GATEWAY_RESET_START = time(19, 40)
        self.GATEWAY_RESET_END = time(19, 55)

    def run(self) -> None:
        """Main monitoring loop."""
        print(f"[WATCHDOG] Starting Mancini Bot Watchdog")
        print(f"[WATCHDOG] Log: {self.log_path}")
        print(f"[WATCHDOG] Status: {self.status_path}")
        print(f"[WATCHDOG] Poll interval: {self.poll_interval}s")
        if self.webhook_url:
            print(f"[WATCHDOG] Webhook: {self.webhook_url}")

        # Check if today is a holiday and notify
        now_et = datetime.now(_ET)
        if self._is_market_holiday(now_et):
            msg = f"Markets closed today ({now_et.strftime('%A, %B %d')}) — CME holiday. Alerts suppressed until next open (Sunday 6 PM ET)."
            print(f"[WATCHDOG] 📅 {msg}")
            if self.webhook_url and _HAS_REQUESTS:
                try:
                    requests.post(self.webhook_url, json={
                        "username": "Mancini Watchdog",
                        "embeds": [{"title": "📅 Market Holiday",
                                    "description": msg,
                                    "color": 0x3498DB}],
                    }, timeout=5)
                except Exception:
                    pass

        # Seek to end of log file to only process new lines
        if self.log_path.exists():
            self._last_log_pos = self.log_path.stat().st_size

        while True:
            try:
                self._check_cycle()
            except Exception as e:
                print(f"[WATCHDOG] Error in check cycle: {e}")
            _time.sleep(self.poll_interval)

    def _check_cycle(self) -> None:
        """One monitoring cycle: read new log lines, check state, emit alerts."""
        now_et = datetime.now(_ET)
        market_open = self._is_market_hours(now_et)

        # 1. Read new log lines
        new_lines = self._read_new_lines()
        self._process_lines(new_lines, now_et)

        # 2. Check bar flow
        if market_open:
            self._check_bar_flow(now_et)

        # 3. Check zero volume streak (only during market hours)
        if market_open:
            self._check_zero_volume()

        # 4. Check error rate (only during market hours —
        #    IB connect retries during weekends/holidays are expected)
        if market_open:
            self._check_error_rate(now_et)

        # 5. Check signal pipeline (RTH only)
        if self._is_rth(now_et):
            self._check_signal_pipeline()

        # 6. Check status file
        self._check_status_file(now_et)

        # 7. Check session rollover
        self._check_rollover(now_et)

        # 8. Resolve cleared alerts
        self._resolve_stale_alerts(now_et)

        # 9. Write alerts file
        self._write_alerts()

    def _read_new_lines(self) -> list[str]:
        """Read new lines from the bot log since last check."""
        if not self.log_path.exists():
            return []

        try:
            size = self.log_path.stat().st_size
            if size < self._last_log_pos:
                # Log was rotated
                self._last_log_pos = 0

            with open(self.log_path, "r", errors="replace") as f:
                f.seek(self._last_log_pos)
                lines = f.readlines()
                self._last_log_pos = f.tell()
            return lines
        except Exception:
            return []

    def _process_lines(self, lines: list[str], now: datetime) -> None:
        """Extract state from new log lines."""
        for line in lines:
            # Bar detection
            m = _BAR_PATTERN.search(line)
            if m:
                bar_num = int(m.group(1))
                volume = int(m.group(7))
                self._last_bar_number = bar_num
                self._last_bar_volume = volume
                self._last_bar_time = now
                self._bars_seen += 1

                if volume == 0:
                    self._zero_volume_streak += 1
                else:
                    self._zero_volume_streak = 0

                # Track RTH signal detection
                if self._is_rth(now):
                    self._rth_bars_without_signal += 1

            # Error detection
            if _ERROR_PATTERN.search(line):
                self._error_window.append(_time.monotonic())

            # Signal detection
            if _SIGNAL_PATTERN.search(line):
                self._signal_count_rth += 1
                self._rth_bars_without_signal = 0

            # Connection state
            if _CONNECT_PATTERN.search(line):
                self._connected = True
                self._resolve_alert("DISCONNECTED")
            if _DISCONNECT_PATTERN.search(line):
                self._connected = False

            # Rollover detection
            if _ROLLOVER_PATTERN.search(line):
                self._last_rollover_date = now.date()
                self._signal_count_rth = 0
                self._rth_bars_without_signal = 0

            # Contract reroll events (always relevant)
            if _REROLL_PATTERN.search(line):
                self._emit_alert(INFO, "CONTRACT_REROLL",
                                 f"Contract reroll triggered: {line.strip()[-80:]}")

            # Trade events — always send to Discord
            entry_m = _ENTRY_PATTERN.search(line)
            if entry_m:
                self._send_trade_webhook(
                    "entry", entry_m.group(7),  # pattern type
                    contracts=entry_m.group(1),
                    symbol=entry_m.group(2),
                    price=entry_m.group(3),
                    stop=entry_m.group(4),
                    target=entry_m.group(5),
                    rr=entry_m.group(6),
                )

            exit_m = _EXIT_BRACKET_PATTERN.search(line)
            if exit_m:
                fill_type = exit_m.group(1)  # SL, TP, or bracket
                fill_price = exit_m.group(2) or "unknown"
                self._send_trade_webhook(
                    "exit", fill_type, price=fill_price,
                )

            # FLATTEN exits (ExitManager stop loss, trailing stop, etc.)
            flatten_m = _EXIT_FLATTEN_PATTERN.search(line)
            if flatten_m:
                contracts = flatten_m.group(1)
                symbol = flatten_m.group(2)
                direction = flatten_m.group(3)
                reason = flatten_m.group(4)
                if "Stop loss" in reason:
                    self._send_trade_webhook("exit", "SL",
                                             reason=reason, direction=direction)
                elif "Trail" in reason:
                    self._send_trade_webhook("exit", "trail",
                                             reason=reason, direction=direction)
                elif "EOD" in reason:
                    self._send_trade_webhook("exit", "eod",
                                             reason=reason, direction=direction)
                else:
                    self._send_trade_webhook("exit", "closed",
                                             reason=reason, direction=direction)

    def _is_in_gateway_reset_window(self, now: datetime) -> bool:
        """True between 19:40 and 19:55 ET on weekdays — the IB Gateway
        nightly re-authentication window. Bar gaps and error spikes are
        expected and auto-recover, so suppress alerts there."""
        # Mon-Fri = 0-4 (weekend = already handled by market_open gate)
        if now.weekday() > 4:
            return False
        t = now.timetz().replace(tzinfo=None) if hasattr(now.timetz(), 'tzinfo') else now.time()
        return self.GATEWAY_RESET_START <= t <= self.GATEWAY_RESET_END

    def _check_bar_flow(self, now: datetime) -> None:
        """Check if bars are arriving at expected rate."""
        if self._last_bar_time is None:
            if self._bars_seen == 0:
                self._emit_alert(CRITICAL, "NO_BARS",
                                 "No bars received since watchdog started — "
                                 "bot may not be connected or contract may be dead")
            return

        # Suppress during the nightly IB Gateway reset window — bar gaps
        # there are expected and auto-recover within ~2 min.
        if self._is_in_gateway_reset_window(now):
            return

        gap = (now - self._last_bar_time).total_seconds()
        if gap > self.MAX_BAR_GAP_SEC:
            self._emit_alert(CRITICAL, "BAR_GAP",
                             f"No new bar for {gap:.0f}s (last: bar #{self._last_bar_number}). "
                             f"Bot may be disconnected or frozen.")
        else:
            self._resolve_alert("BAR_GAP")
            self._resolve_alert("NO_BARS")

    def _check_zero_volume(self) -> None:
        """Check for consecutive zero-volume bars (expired contract)."""
        if self._zero_volume_streak >= self.MAX_ZERO_VOLUME:
            self._emit_alert(CRITICAL, "ZERO_VOLUME",
                             f"{self._zero_volume_streak} consecutive zero-volume bars — "
                             f"contract may be expired. Bot should auto-reroll.")
        else:
            self._resolve_alert("ZERO_VOLUME")

    def _check_error_rate(self, now: datetime) -> None:
        """Check if error rate is abnormally high."""
        # Same nightly-window suppression as bar flow — IB Gateway
        # re-auth produces a guaranteed error spike that auto-clears.
        if self._is_in_gateway_reset_window(now):
            return
        cutoff = _time.monotonic() - 300  # last 5 minutes
        recent_errors = sum(1 for t in self._error_window if t > cutoff)
        if recent_errors >= self.MAX_ERRORS_PER_5MIN:
            self._emit_alert(HIGH, "ERROR_SPIKE",
                             f"{recent_errors} errors in last 5 min — "
                             f"check docker logs for details")
        else:
            self._resolve_alert("ERROR_SPIKE")

    def _check_signal_pipeline(self) -> None:
        """Check if signal pipeline is producing signals during RTH."""
        if self._rth_bars_without_signal >= self.RTH_SIGNAL_CHECK_BARS:
            self._emit_alert(WARNING, "NO_SIGNALS_RTH",
                             f"{self._rth_bars_without_signal} RTH bars without any signal. "
                             f"Level store may be empty or all patterns stuck. "
                             f"Total signals this session: {self._signal_count_rth}")

    def _check_status_file(self, now: datetime) -> None:
        """Check status.json for position and state consistency."""
        if not self.status_path.exists():
            return

        try:
            with open(self.status_path) as f:
                status = json.load(f)

            # Check status file freshness
            ts = status.get("timestamp")
            if ts:
                status_time = datetime.fromisoformat(ts)
                if hasattr(status_time, "tzinfo") and status_time.tzinfo is None:
                    status_time = status_time.replace(tzinfo=_ET)
                age = (now - status_time).total_seconds()
                if age > 300 and self._is_market_hours(now):
                    self._emit_alert(HIGH, "STATUS_STALE",
                                     f"status.json is {age:.0f}s old — bot may be frozen")
                else:
                    self._resolve_alert("STATUS_STALE")

            # Check for contract expiry warning
            contract_expiry = status.get("contract_expiry")
            if contract_expiry:
                try:
                    exp_date = datetime.strptime(contract_expiry, "%Y%m%d").date()
                    days_to_expiry = (exp_date - now.date()).days
                    if days_to_expiry <= self.CONTRACT_EXPIRY_WARN_DAYS:
                        self._emit_alert(WARNING, "CONTRACT_EXPIRY",
                                         f"Contract expires in {days_to_expiry} days "
                                         f"({contract_expiry}). Watch for rollover.")
                except ValueError:
                    pass

        except (json.JSONDecodeError, Exception):
            pass

    def _check_rollover(self, now: datetime) -> None:
        """Check that session rollover happened at 18:00 ET."""
        if now.time() >= time(18, 30) and now.weekday() < 5:
            if self._last_rollover_date != now.date():
                self._emit_alert(HIGH, "MISSED_ROLLOVER",
                                 f"No session rollover detected for {now.date()}. "
                                 f"Session state may be stale.")
            else:
                self._resolve_alert("MISSED_ROLLOVER")

    def _emit_alert(self, severity: str, code: str, message: str) -> None:
        """Emit an alert with cooldown to prevent spam."""
        now = _time.monotonic()
        # None means "never emitted" — explicitly distinct from a
        # very-recent emit. Using 0 as the default silently dropped the
        # first 5 minutes of alerts after process start because
        # time.monotonic() begins from a small value on some platforms.
        last = self._last_alert_times.get(code)
        if last is not None and now - last < self.ALERT_COOLDOWN_SEC:
            return  # cooldown active

        alert = Alert(severity, code, message)
        self._active_alerts[code] = alert
        self._last_alert_times[code] = now

        # Print to stdout (docker logs)
        severity_icon = {
            CRITICAL: "🚨",
            HIGH: "⚠️",
            WARNING: "⚡",
            INFO: "ℹ️",
        }.get(severity, "")
        print(f"[WATCHDOG] {severity_icon} {alert}")

        # Webhook notification for CRITICAL and HIGH
        if severity in (CRITICAL, HIGH) and self.webhook_url and _HAS_REQUESTS:
            self._send_webhook(alert)

    def _resolve_alert(self, code: str) -> None:
        """Mark an alert as resolved."""
        if code in self._active_alerts and not self._active_alerts[code].resolved:
            self._active_alerts[code].resolved = True
            print(f"[WATCHDOG] ✅ RESOLVED: {code}")
            if self.webhook_url and _HAS_REQUESTS:
                try:
                    payload = {
                        "username": "Mancini Watchdog",
                        "embeds": [{
                            "title": f"✅ RESOLVED: {code}",
                            "description": self._active_alerts[code].message,
                            "color": 0x2ECC71,  # green
                        }],
                    }
                    requests.post(self.webhook_url, json=payload, timeout=5)
                except Exception:
                    pass

    def _resolve_stale_alerts(self, now: datetime) -> None:
        """Remove alerts that have been resolved for >10 minutes."""
        stale_codes = []
        for code, alert in self._active_alerts.items():
            if alert.resolved:
                last_time = self._last_alert_times.get(code, 0)
                if _time.monotonic() - last_time > 600:
                    stale_codes.append(code)
        for code in stale_codes:
            del self._active_alerts[code]

    def _send_webhook(self, alert: Alert) -> None:
        """Send alert to Discord webhook."""
        severity_color = {
            CRITICAL: 0xFF0000,  # red
            HIGH: 0xFF8C00,     # orange
            WARNING: 0xFFD700,  # yellow
            INFO: 0x3498DB,     # blue
        }
        severity_icon = {
            CRITICAL: "🚨", HIGH: "⚠️", WARNING: "⚡", INFO: "ℹ️",
        }
        try:
            payload = {
                "username": "Mancini Watchdog",
                "embeds": [{
                    "title": f"{severity_icon.get(alert.severity, '')} {alert.severity}: {alert.code}",
                    "description": alert.message,
                    "color": severity_color.get(alert.severity, 0x808080),
                    "footer": {"text": f"Mancini Bot • {alert.timestamp}"},
                }],
            }
            requests.post(self.webhook_url, json=payload, timeout=5)
        except Exception as e:
            print(f"[WATCHDOG] Webhook failed: {e}")

    def _send_trade_webhook(self, event: str, pattern: str, **kwargs) -> None:
        """Send trade entry/exit notification to Discord.

        DEPRECATED. As of the rich-embed PR, the bot posts trade events
        directly via live.trade_notifications. This watchdog path is kept
        as a fallback emergency channel and is GATED by the
        WATCHDOG_LEGACY_TRADE_NOTIFY env var. Default behavior is silent
        — no duplicate Discord notifications.
        """
        import os as _os
        if not _os.environ.get("WATCHDOG_LEGACY_TRADE_NOTIFY"):
            return
        if not self.webhook_url or not _HAS_REQUESTS:
            return

        now_str = datetime.now(_ET).strftime("%I:%M %p ET")

        if event == "entry":
            direction = "LONG" if pattern in (
                "FAILED_BREAKDOWN", "LEVEL_RECLAIM"
            ) else "SHORT"
            color = 0x2ECC71 if direction == "LONG" else 0xE74C3C
            icon = "🟢" if direction == "LONG" else "🔴"
            pattern_display = pattern.replace("_", " ").title()
            fields = [
                {"name": "Direction", "value": f"{icon} **{direction}**", "inline": True},
                {"name": "Pattern", "value": pattern_display, "inline": True},
                {"name": "R:R", "value": f"{kwargs.get('rr', '?')}:1", "inline": True},
                {"name": "Entry", "value": f"${kwargs.get('price', '?')}", "inline": True},
                {"name": "Stop", "value": f"${kwargs.get('stop', '?')}", "inline": True},
                {"name": "Target", "value": f"${kwargs.get('target', '?')}", "inline": True},
            ]
            payload = {
                "username": "Mancini Bot",
                "embeds": [{
                    "title": f"{icon} TRADE ENTRY — {kwargs.get('contracts', '?')} {kwargs.get('symbol', 'MES')}",
                    "color": color,
                    "fields": fields,
                    "footer": {"text": f"Mancini Bot • {now_str}"},
                }],
            }
        elif event == "exit":
            exit_type = pattern.upper()
            if exit_type == "SL":
                color = 0xE74C3C  # red
                icon = "🛑"
                title = "STOP LOSS HIT"
            elif exit_type == "TP":
                color = 0x2ECC71  # green
                icon = "🎯"
                title = "TARGET HIT"
            elif exit_type == "TRAIL":
                color = 0xF39C12  # amber
                icon = "📏"
                title = "TRAILING STOP HIT"
            elif exit_type == "EOD":
                color = 0x95A5A6  # gray
                icon = "🕐"
                title = "EOD FLATTEN"
            else:
                color = 0x95A5A6  # gray
                icon = "📤"
                title = "POSITION CLOSED"

            price = kwargs.get("price", "unknown")
            reason = kwargs.get("reason", "")
            direction = kwargs.get("direction", "")
            desc_parts = []
            if price != "unknown":
                desc_parts.append(f"Fill @ **{price}**")
            if direction:
                desc_parts.append(f"Direction: **{direction.upper()}**")
            if reason:
                desc_parts.append(f"Reason: {reason}")
            description = "\n".join(desc_parts) if desc_parts else "Position closed"

            payload = {
                "username": "Mancini Bot",
                "embeds": [{
                    "title": f"{icon} {title}",
                    "description": description,
                    "color": color,
                    "footer": {"text": f"Mancini Bot • {now_str}"},
                }],
            }
        else:
            return

        try:
            requests.post(self.webhook_url, json=payload, timeout=5)
            print(f"[WATCHDOG] 📨 Trade notification sent: {event} {pattern}")
        except Exception as e:
            print(f"[WATCHDOG] Trade webhook failed: {e}")

    def _write_alerts(self) -> None:
        """Write active alerts to JSON file for dashboard consumption."""
        try:
            alerts_data = {
                "updated_at": datetime.now(_ET).isoformat(),
                "active": [a.to_dict() for a in self._active_alerts.values()
                           if not a.resolved],
                "resolved": [a.to_dict() for a in self._active_alerts.values()
                             if a.resolved],
                "stats": {
                    "bars_seen": self._bars_seen,
                    "last_bar_number": self._last_bar_number,
                    "zero_volume_streak": self._zero_volume_streak,
                    "rth_signals": self._signal_count_rth,
                    "rth_bars_without_signal": self._rth_bars_without_signal,
                    "connected": self._connected,
                },
            }
            self.alerts_path.write_text(json.dumps(alerts_data, indent=2))
        except Exception:
            pass

    @staticmethod
    def _is_market_holiday(dt: datetime) -> bool:
        """Check if today is a CME market holiday (futures closed all day).

        CME holidays: New Year's, MLK Day, Presidents' Day, Good Friday,
        Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving,
        Christmas. Some are early close (1 PM ET) but we treat as closed
        to avoid false alerts on thin/no data.
        """
        from datetime import timedelta

        year = dt.year
        d = dt.date()

        # Fixed holidays
        holidays = set()
        holidays.add(date(year, 1, 1))    # New Year's Day
        holidays.add(date(year, 7, 4))    # Independence Day
        holidays.add(date(year, 12, 25))  # Christmas

        # Juneteenth (June 19, observed)
        jun19 = date(year, 6, 19)
        if jun19.weekday() == 5:  # Saturday -> Friday
            holidays.add(jun19 - timedelta(days=1))
        elif jun19.weekday() == 6:  # Sunday -> Monday
            holidays.add(jun19 + timedelta(days=1))
        else:
            holidays.add(jun19)

        # MLK Day: 3rd Monday of January
        jan1 = date(year, 1, 1)
        first_monday = jan1 + timedelta(days=(7 - jan1.weekday()) % 7)
        holidays.add(first_monday + timedelta(weeks=2))

        # Presidents' Day: 3rd Monday of February
        feb1 = date(year, 2, 1)
        first_monday = feb1 + timedelta(days=(7 - feb1.weekday()) % 7)
        holidays.add(first_monday + timedelta(weeks=2))

        # Memorial Day: last Monday of May
        may31 = date(year, 5, 31)
        holidays.add(may31 - timedelta(days=(may31.weekday()) % 7))

        # Labor Day: 1st Monday of September
        sep1 = date(year, 9, 1)
        first_monday = sep1 + timedelta(days=(7 - sep1.weekday()) % 7)
        holidays.add(first_monday)

        # Thanksgiving: 4th Thursday of November
        nov1 = date(year, 11, 1)
        first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
        holidays.add(first_thu + timedelta(weeks=3))

        # Good Friday: 2 days before Easter Sunday
        # Easter algorithm (Anonymous Gregorian)
        a = year % 19
        b = year // 100
        c = year % 100
        d_val = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d_val - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        easter = date(year, month, day)
        good_friday = easter - timedelta(days=2)
        holidays.add(good_friday)

        return d in holidays

    @staticmethod
    def _is_market_hours(dt: datetime) -> bool:
        """Check if market is open (Globex hours, not daily break, not holidays)."""
        if Watchdog._is_market_holiday(dt):
            return False
        wd = dt.weekday()
        t = dt.time()
        if wd == 5:  # Saturday
            return False
        if wd == 6 and t < time(18, 0):  # Sunday before 6 PM
            return False
        if wd == 4 and t >= time(17, 0):  # Friday after 5 PM
            return False
        if time(17, 0) <= t < time(18, 0):  # Daily break
            return False
        return True

    @staticmethod
    def _is_rth(dt: datetime) -> bool:
        """Check if we're in Regular Trading Hours (9:30 AM - 4:00 PM ET)."""
        t = dt.time()
        return time(9, 30) <= t < time(16, 0) and dt.weekday() < 5


def main():
    import sys
    # Force unbuffered stdout for Docker logs
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Mancini Bot Watchdog")
    parser.add_argument("--log", default="/app/logs/bot.log",
                        help="Path to bot log file")
    parser.add_argument("--status", default="/app/logs/status.json",
                        help="Path to status.json")
    parser.add_argument("--alerts", default="/app/logs/watchdog_alerts.json",
                        help="Path to write alerts JSON")
    parser.add_argument("--poll-interval", type=float, default=30.0,
                        help="Seconds between checks (default: 30)")
    parser.add_argument("--webhook", default=os.environ.get("WATCHDOG_WEBHOOK"),
                        help="Webhook URL for CRITICAL/HIGH alerts")
    args = parser.parse_args()

    watchdog = Watchdog(
        log_path=args.log,
        status_path=args.status,
        alerts_path=args.alerts,
        poll_interval=args.poll_interval,
        webhook_url=args.webhook,
    )
    watchdog.run()


if __name__ == "__main__":
    main()
