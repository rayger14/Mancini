"""Health check and trade alert module for cloud deployment.

Runs as a background thread inside the IB runner to:
- Monitor IB Gateway connection
- Write heartbeat file for external monitoring
- Log trade alerts to stdout (visible in docker logs)
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


class HealthMonitor:
    """Background health monitor for the trading bot."""

    def __init__(
        self,
        heartbeat_path: str = "/app/logs/heartbeat.txt",
        check_interval_sec: float = 30.0,
    ):
        self._heartbeat_path = Path(heartbeat_path)
        self._check_interval = check_interval_sec
        self._connected: bool = False
        self._last_bar_time: Optional[datetime] = None
        self._trades_today: int = 0
        self._daily_pnl_pts: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

    def start(self) -> None:
        """Start the health monitor background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Health monitor started")

    def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def update_connection(self, connected: bool) -> None:
        self._connected = connected

    def update_bar(self, timestamp: datetime) -> None:
        self._last_bar_time = timestamp

    def update_trade(self, trades_today: int, daily_pnl_pts: float) -> None:
        self._trades_today = trades_today
        self._daily_pnl_pts = daily_pnl_pts

    def _run(self) -> None:
        """Background loop that writes heartbeat and checks health."""
        while self._running:
            try:
                self._write_heartbeat()
                self._check_data_staleness()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            time.sleep(self._check_interval)

    def _write_heartbeat(self) -> None:
        """Write heartbeat file with current status."""
        try:
            self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            now = datetime.now().isoformat()
            last_bar = self._last_bar_time.isoformat() if self._last_bar_time else "none"
            status = "connected" if self._connected else "disconnected"
            self._heartbeat_path.write_text(
                f"timestamp={now}\n"
                f"status={status}\n"
                f"last_bar={last_bar}\n"
                f"trades_today={self._trades_today}\n"
                f"daily_pnl_pts={self._daily_pnl_pts:.1f}\n"
            )
        except Exception as e:
            logger.warning(f"Failed to write heartbeat: {e}")

    def _check_data_staleness(self) -> None:
        """Warn if no bar received in >5 minutes during market hours."""
        if self._last_bar_time is None:
            return
        elapsed = (datetime.now() - self._last_bar_time).total_seconds()
        if elapsed > 300:
            logger.warning(
                f"STALE DATA: No bar received in {elapsed:.0f}s "
                f"(last bar: {self._last_bar_time})"
            )


def log_trade_alert(
    action: str,
    symbol: str,
    direction: str,
    pattern: str,
    entry_price: float,
    stop_price: float = 0,
    target_price: float = 0,
    pnl_pts: float = 0,
    exit_reason: str = "",
) -> None:
    """Log a trade alert visible in docker logs."""
    if action == "ENTRY":
        logger.info(
            f"TRADE ALERT | {action} {direction.upper()} {symbol} "
            f"@ {entry_price:.2f} | Stop: {stop_price:.2f} | "
            f"Target: {target_price:.2f} | Pattern: {pattern}"
        )
    elif action == "EXIT":
        logger.info(
            f"TRADE ALERT | {action} {direction.upper()} {symbol} "
            f"@ {entry_price:.2f} | PnL: {pnl_pts:+.1f} pts | "
            f"Reason: {exit_reason} | Pattern: {pattern}"
        )
