"""File-based IPC bridge between Python strategy engine and NinjaTrader 8.

NinjaTrader writes bar data and fill confirmations.
Python writes trade signals (enter, update_stop, flatten).
Communication happens through JSON files in a shared directory.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from datetime import datetime, date, time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class NTBridgeConfig:
    """Configuration for the NinjaTrader file bridge."""

    shared_dir: str = r"C:\ManciniShared"
    instrument: str = "MES 03-26"
    poll_interval_sec: float = 0.5
    heartbeat_interval_sec: float = 5.0
    stale_heartbeat_sec: float = 30.0
    signal_timeout_bars: int = 2  # warn if no fill after N bars


class NTBridge:
    """File-based communication layer with NinjaTrader 8.

    All writes use atomic pattern (write .tmp, rename to final) to prevent
    partial reads. All timestamps are ISO 8601 US/Eastern.
    """

    def __init__(self, config: NTBridgeConfig):
        self.config = config
        self._base = Path(config.shared_dir)
        self._bars_dir = self._base / "bars"
        self._signals_dir = self._base / "signals"
        self._fills_dir = self._base / "fills"
        self._state_dir = self._base / "state"

        self._last_bar_number: int = -1
        self._signal_counter: int = 0
        self._processed_fill_ids: set[str] = set()
        self._pending_signals: dict[str, datetime] = {}

    def ensure_directories(self) -> None:
        """Create shared directory structure if it doesn't exist."""
        for d in [self._bars_dir, self._signals_dir, self._fills_dir, self._state_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Bridge directories ready at {self._base}")

    # ── Reading from NinjaTrader ─────────────────────────────────────

    def wait_for_history(self, session_date: date, timeout_sec: float = 300) -> bool:
        """Block until NT writes the history file for this session date.

        Returns True if history file appeared, False on timeout.
        """
        filename = f"history_{session_date.strftime('%Y%m%d')}.json"
        path = self._bars_dir / filename
        start = _time.time()
        while _time.time() - start < timeout_sec:
            if path.exists():
                logger.info(f"History file found: {filename}")
                return True
            _time.sleep(1.0)
        logger.error(f"Timeout waiting for history file: {filename}")
        return False

    def read_history(self, session_date: date) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Read history file written by NT at session start.

        Returns (prior_day_df, current_day_df). Either may be None.
        """
        filename = f"history_{session_date.strftime('%Y%m%d')}.json"
        path = self._bars_dir / filename
        if not path.exists():
            return None, None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to read history: {e}")
            return None, None

        prior_df = self._bars_list_to_df(data.get("prior_day_bars", []))
        current_df = self._bars_list_to_df(data.get("current_day_bars", []))
        return prior_df, current_df

    def poll_new_bar(self) -> Optional[dict]:
        """Check for new bar files. Returns the next unprocessed bar or None.

        Scans bars/ for files with bar_number > _last_bar_number.
        """
        if not self._bars_dir.exists():
            return None

        # Find bar files matching pattern bar_YYYYMMDD_HHMM.json
        bar_files = sorted(self._bars_dir.glob("bar_*.json"))
        for path in bar_files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                bar_num = data.get("bar_number", -1)
                if bar_num > self._last_bar_number:
                    self._last_bar_number = bar_num
                    return data
            except (json.JSONDecodeError, OSError):
                continue
        return None

    def read_fills(self) -> list[dict]:
        """Read all unprocessed fill files from fills/ directory."""
        fills = []
        if not self._fills_dir.exists():
            return fills

        for path in sorted(self._fills_dir.glob("fill_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                fill_id = data.get("fill_id", "")
                if fill_id and fill_id not in self._processed_fill_ids:
                    self._processed_fill_ids.add(fill_id)
                    fills.append(data)
                    # Remove from pending signals
                    sig_id = data.get("signal_id", "")
                    self._pending_signals.pop(sig_id, None)
            except (json.JSONDecodeError, OSError):
                continue
        return fills

    def read_position(self) -> Optional[dict]:
        """Read current position state from state/position.json."""
        path = self._state_dir / "position.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def check_nt_heartbeat(self) -> bool:
        """Return True if NT heartbeat is fresh (< stale_heartbeat_sec old)."""
        path = self._state_dir / "nt_heartbeat.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(data["timestamp"])
            age = (datetime.now(ts.tzinfo) - ts).total_seconds()
            return age < self.config.stale_heartbeat_sec
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            return False

    # ── Writing to NinjaTrader ───────────────────────────────────────

    def write_signal(self, action: str, **kwargs) -> str:
        """Write a signal file for NT to execute.

        Parameters
        ----------
        action : str
            One of: 'enter_long', 'update_stop', 'flatten', 'partial_exit'
        **kwargs
            Signal-specific fields (entry_price, stop_price, target_price, etc.)

        Returns
        -------
        str : signal_id
        """
        self._signal_counter += 1
        now = datetime.now()
        signal_id = f"sig_{now.strftime('%Y%m%d_%H%M%S')}_{self._signal_counter:03d}"

        data = {
            "signal_id": signal_id,
            "timestamp": now.isoformat(),
            "action": action,
            "instrument": self.config.instrument,
            "status": "UNREAD",
            **kwargs,
        }

        filename = f"signal_{now.strftime('%Y%m%d_%H%M%S')}_{self._signal_counter:03d}.json"
        self._atomic_write(self._signals_dir / filename, data)

        self._pending_signals[signal_id] = now
        logger.info(f"Signal written: {action} [{signal_id}]")
        return signal_id

    def write_entry_signal(
        self,
        quantity: int,
        entry_price: float,
        stop_price: float,
        target_price: float,
        signal_type: str = "",
        rr_ratio: float = 0.0,
    ) -> str:
        """Write an entry signal with full bracket order details."""
        return self.write_signal(
            action="enter_long",
            quantity=quantity,
            entry_price=round(entry_price, 2),
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            signal_type=signal_type,
            rr_ratio=round(rr_ratio, 2),
        )

    def write_stop_update(self, new_stop_price: float, reason: str = "") -> str:
        """Write a stop price modification signal."""
        return self.write_signal(
            action="update_stop",
            new_stop_price=round(new_stop_price, 2),
            reason=reason,
        )

    def write_flatten(self, reason: str = "") -> str:
        """Write a flatten (exit all) signal."""
        return self.write_signal(action="flatten", reason=reason)

    def write_partial_exit(self, quantity: int, new_stop_price: float, reason: str = "") -> str:
        """Write a partial exit signal."""
        return self.write_signal(
            action="partial_exit",
            quantity=quantity,
            new_stop_price=round(new_stop_price, 2),
            reason=reason,
        )

    def write_heartbeat(self, bars_processed: int = 0, session_date: Optional[date] = None) -> None:
        """Write Python heartbeat file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "status": "running",
            "bars_processed": bars_processed,
            "session_date": str(session_date or date.today()),
        }
        self._atomic_write(self._state_dir / "py_heartbeat.json", data)

    # ── Pending signal tracking ──────────────────────────────────────

    def get_pending_signals(self) -> dict[str, datetime]:
        """Return signals that have been sent but not yet filled."""
        return dict(self._pending_signals)

    def check_stale_signals(self, current_bar_count: int) -> list[str]:
        """Return signal IDs that are older than signal_timeout_bars."""
        stale = []
        cutoff = datetime.now()
        for sig_id, sent_time in self._pending_signals.items():
            age_sec = (cutoff - sent_time).total_seconds()
            # Each bar is ~60 seconds
            if age_sec > self.config.signal_timeout_bars * 60:
                stale.append(sig_id)
        return stale

    # ── Housekeeping ─────────────────────────────────────────────────

    def cleanup_old_files(self, days: int = 7) -> int:
        """Remove files older than N days. Returns count of files removed."""
        import time as _t
        cutoff = _t.time() - days * 86400
        removed = 0
        for subdir in [self._bars_dir, self._signals_dir, self._fills_dir]:
            if not subdir.exists():
                continue
            for path in subdir.glob("*.json"):
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
        if removed:
            logger.info(f"Cleaned up {removed} files older than {days} days")
        return removed

    # ── Internal helpers ─────────────────────────────────────────────

    def _atomic_write(self, path: Path, data: dict) -> None:
        """Write JSON atomically: write to .tmp, then rename."""
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(path)

    def _bars_list_to_df(self, bars: list[dict]) -> Optional[pd.DataFrame]:
        """Convert a list of bar dicts to a pandas DataFrame with DatetimeIndex."""
        if not bars:
            return None
        df = pd.DataFrame(bars)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            if df.index.tz is None:
                df.index = df.index.tz_localize("US/Eastern")
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
