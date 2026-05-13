"""Interactive Brokers live runner: bridges IB bar data to Python strategy engine.

Mirrors the bar-by-bar loop from the backtest engine and mt5_runner.py but
reads bars from IB TWS/Gateway via ib_insync/ib_async.

Usage (local):
    python3 live/ib_runner.py --port 7497 --contracts 4 --symbol MES  # TWS paper
    python3 live/ib_runner.py --port 7496 --contracts 4               # TWS live

Usage (Docker / cloud):
    python3 live/ib_runner.py --host ib-gateway --port 4002 --symbol MES --full-session
"""

from __future__ import annotations

import json
import os
import signal as os_signal
import sys
import time as _time
from datetime import datetime, date, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
from loguru import logger

_ET = pytz.timezone("US/Eastern")

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    StrategyParams,
    ElevatorParams,
    ExitParams,
    RiskParams,
    SessionTimes,
    ESContractSpec,
)
from core.indicators import compute_velocity
from core.regime_filter import RegimeParams, build_daily_bars
from core.signals import Signal
from live.ib_bridge import IBBridge, IBConfig
from strategy.entry_manager import EntryDecision, EntryManager
from strategy.exit_manager import ExitManager, ExitAction, ExitPhase, TradePosition
from strategy.mancini_long import ManciniLongStrategy
from strategy.position_manager import PositionManager, TradeRecord
from strategy.risk_manager import RiskManager
from live.market_data import fetch_market_snapshot


# ── Production parameters (Optuna v2 Trial 16 — data-informed, Mar 2026) ────
# Walk-forward validated: Train PF=1.70/+2,589 pts, OOS PF=1.14/+289 pts
# Full: 737T, PF=1.50, Sharpe=2.90, +2,878 pts (2024: +1,466 | 2025: +1,616)
# Key changes from live data analysis of 1,651 events:
#   - wider dips (15 pts): captures more FB opportunities
#   - wider stops (20 pts): BD Shorts with 10-15 pt stops are winners
#   - shorter hold (14 bars): quicker exits improve PF
#   - sweep depth UNCAPPED: live data confirms 50+ pt sweeps = 117.6 avg recovery
#   - slower BD confirmation (21 bars): fewer false breakdowns

PRODUCTION_STRATEGY = StrategyParams(
    acceptance_max_dip_pts=15.0,          # Optuna v2: wider dips capture more winners (was 50/4)
    acceptance_min_hold_bars=11,           # Optuna v2: stricter confirmation (was 7)
    swing_low_order=15,
    allow_breakdown_short=True,
    allow_backtest_short=False,
    fb_stop_buffer_pts=6.0,               # Optuna v2 (was 7.0)
    lr_stop_buffer_pts=4.0,               # Optuna v2 (was 3.0)
    max_target_distance_pts=30.0,         # Optuna v2 (was 15.0)
    max_fb_sweep_depth_pts=999.0,         # No cap — live data: 50+ pt sweeps = 117.6 avg recovery (Mancini: bigger sell = bigger squeeze)
    true_breakdown_abort_bars=40,         # Was 20 — too aggressive for deep sweeps. 30+ pt sweeps need time to recover.
                                          # Mancini: "bigger sell = bigger squeeze" but recovery takes 25-40 bars, not 20.
                                          # 5yr H9: 5-10 pt sweeps = 63% WR (best bucket). Let deep sweeps confirm.
    bd_confirm_bars=21,                   # Legacy fallback (conviction system overrides this)
    bd_stop_buffer_pts=6.0,               # Optuna v2: wider BD stops (was 4.0)
    bd_max_break_depth_pts=17.0,          # Optuna v2 (was 14.0)
    bd_timeout_bars=35,                   # Optuna v2 (was 55)
    signal_cooldown_bars=15,
    # Mancini exit scaling: T1 at first resistance level, not fixed distance
    # "Lock in 75% profits at first level up" — with 2 contracts, best we can do is 50/50
    # but T1 should be at the ACTUAL next level, not a fixed point target
    mancini_t1_at_first_resistance=True,
    # Shadow mode features: run detectors but only log, don't trade
    shadow_mode_features=True,
    use_sweep_depth_sizing=True,          # Shadow: log sweep-depth-adjusted sizing
    use_mode1_detection=True,             # Shadow: log Mode 1 trend day detection
    allow_velocity_short=True,            # Shadow: log velocity breakdown shorts
    use_regime_filter=False,              # Data collection: don't block longs/shorts based on regime
    # BD Short validated: PDL = +199 pts/75T/44% WR (edge); MHL = -191 pts/19T/32% WR (loser)
    # bd_require_major_level=True keeps CLUSTER_LOW out. MHL exclusion needs future code change.
    # --- Session range gate (Optuna-validated: +125 pts OOS, Sharpe 2.05) ---
    min_session_range_pts=15.0,           # Don't trade until market moves 15+ pts
    min_session_range_grace_bars=30,      # Grace period at session start
    # --- BD Short conviction scoring (replaces flat 21-bar count) ---
    # Confirms faster on strong breaks (vel+depth+candles), slower on weak ones.
    # Backtest: PF 1.10→1.15, BD 10→19 trades, +203 pts improvement on 2025.
    bd_conviction_threshold=21.0,         # Score to confirm (same as old bar count when weights=0)
    bd_min_bars_floor=5,                  # Safety minimum — at least 5 bars no matter what
    bd_conviction_depth_weight=1.0,       # Deeper breaks score higher
    bd_conviction_velocity_weight=1.0,    # Fast selling scores higher
    bd_conviction_candle_weight=0.5,      # Bearish candles (close near low) score higher
    bd_conviction_new_low_weight=0.5,     # Making new lows = progression
    # --- Deep sell recovery: catch crash bottoms when no levels exist nearby ---
    # Mar 24 2026: missed 6573→6610 FB because nearest level was 6616 (43 pts above).
    # Deep sell mode uses 5-bar swing order (vs 30), 10-pt rally confirm, bypasses RTH filter.
    # 5yr backtest: +148 pts, creates INTRADAY_LOW levels eligible for FB.
    allow_deep_sell_recovery=True,
    deep_sell_threshold_pts=30.0,         # 30+ pts below nearest support = deep sell
    deep_sell_swing_order=5,              # fast swing confirmation (5 bars vs 30)
    deep_sell_rally_confirm_pts=20.0,     # Mancini: significant low = 20+ pt bounce (V-shaped reversal)
    # --- 5-min level detection (Mancini reads 5-min charts for level ID) ---
    use_5min_levels=False,                # Off — needs DatetimeIndex fix for live DF before enabling
    swing_low_order_5min=6,               # 6 bars on 5-min = 30 min confirmation (ready when enabled)
    detect_shelf_levels=False,            # Shelf detection ready but 5-min must be fixed first
    shelf_min_touches=8,                  # Real Mancini shelves have 8+ touches on 5-min
    shelf_sweep_min_pts=2.0,              # Need 2+ pts below shelf to qualify
    # --- Mancini LLM plan consumption (PR #8) ---
    # Engine loads the nightly LLM-extracted plan (mancini_plan_<date>.json)
    # and applies its gates to signal qualification. Boosts FB-trade quality
    # by recognizing Mancini's explicit planned setups (LQS bonus), blocks FB
    # longs on Mode 1 Green days, rejects entries in named danger zones. Plan
    # extraction runs nightly via cron regardless; this enables the engine to
    # actually consume the output. Inert when plan file is missing or
    # extract_status != "ok".
    use_mancini_llm_plan=True,
)
PRODUCTION_ELEVATOR = ElevatorParams(
    min_velocity_pts_per_min=0.75,
    min_levels_broken=2,
    higher_low_lookback=4,
)
PRODUCTION_EXIT = ExitParams(
    default_contracts=4,
    t1_exit_fraction=0.75,      # Mancini: "Lock in 75% profits at first level up" = 3 of 4
    t2_exit_fraction=0.0,
    runner_fraction=0.25,       # 1 contract runner (25%) — catches the rare trend days
    breakeven_buffer_pts=-3.0,  # Mancini: "several pts under breakeven"
    trailing_stop_pts=12.0,
    runner_prior_day_low_buffer_pts=1.0,
    fb_max_hold_bars=14,        # Optuna v2: shorter hold = quicker exits (was 0)
)
PRODUCTION_RISK = RiskParams(max_trades_per_day=999, max_daily_loss_pts=9999.0, skip_tuesdays=False, min_rr_ratio=0.8)  # Optuna v2: 0.8 R:R filter (was 0.1)
PRODUCTION_REGIME = RegimeParams(
    mode="ema",
    ema_span=30,                          # Optuna v2: faster regime detection (was 80)
    slope_lookback=10,                    # Optuna v2 (was 6)
    slope_threshold_atr_mult=0.325,       # Optuna v2 (was 0.35)
)
PRODUCTION_SESSION = SessionTimes(
    chop_zone_start=time(13, 0),
    chop_zone_end=time(15, 0),
    afternoon_window_end=time(15, 58),
    eod_flatten_time=time(15, 58),
)

# Full session (23-hour): globex open 18:00 -> RTH close 16:59, break 17:00-17:59
# Trade all hours EXCEPT European open (02:00-06:00) and chop zone (13:00-15:00)
FULL_SESSION = SessionTimes(
    rth_open=time(18, 0),       # Globex open (start of session)
    rth_close=time(17, 0),      # Globex close (end of session, next day)
    morning_window_start=time(9, 30),
    morning_window_end=time(11, 0),
    afternoon_window_start=time(15, 0),
    afternoon_window_end=time(16, 50),
    eod_flatten_time=time(16, 50),  # Flatten 10 min before break
    chop_zone_start=time(13, 0),
    chop_zone_end=time(15, 0),
)
MES_CONTRACT = ESContractSpec(
    symbol="MES",
    tick_size=0.25,
    tick_value=1.25,
    point_value=5.0,
    margin_initial=1_265.0,
    margin_maintenance=1_150.0,
    exchange="CME",
)


def _is_market_closed(now: datetime = None) -> bool:
    """Check if ES/MES futures market is closed right now.

    Schedule (all times US/Eastern):
      - Daily break: 17:00-18:00 ET (Mon-Thu)
      - Weekend: Friday 17:00 ET through Sunday 18:00 ET
    """
    if now is None:
        now = datetime.now()
    t = now.time()
    wd = now.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun

    # Saturday: always closed
    if wd == 5:
        return True
    # Sunday: closed until 18:00 ET
    if wd == 6:
        return t < time(18, 0)
    # Friday: closed after 17:00 ET
    if wd == 4 and t >= time(17, 0):
        return True
    # Mon-Thu: daily break 17:00-18:00
    if time(17, 0) <= t < time(18, 0):
        return True

    return False


class IBRunner:
    """Main live runner: bridges Interactive Brokers data to Python strategy engine.

    Polls IB for new 1-minute bars, runs the full strategy pipeline,
    and executes bracket orders via the IB API.

    Config 2 + FB-only PM:
    - Active windows: 22:00-02:00, 06:00-09:30, 09:30-13:00, 15:00-16:50 (FB only)
    - Blocked: 02:00-06:00 (Euro), 13:00-15:00 (Chop), 18:00-22:00 (Evening)
    - Skip Mondays
    - Half Kelly: 17 MES on $25K
    """

    def __init__(
        self,
        ib_config: IBConfig = IBConfig(),
        strategy_params: StrategyParams = PRODUCTION_STRATEGY,
        elevator_params: ElevatorParams = PRODUCTION_ELEVATOR,
        exit_params: ExitParams = PRODUCTION_EXIT,
        risk_params: RiskParams = PRODUCTION_RISK,
        session_times: SessionTimes = PRODUCTION_SESSION,
        contract: ESContractSpec = MES_CONTRACT,
        min_rr_ratio: float = 1.0,
        rth_filter: tuple = None,
        fb_only_pm: bool = False,
        regime_params: Optional[RegimeParams] = PRODUCTION_REGIME,
        bypass_session_gates: bool = True,
    ):
        self.bridge = IBBridge(ib_config)
        self.contract = contract
        self.exit_params = exit_params
        self._fb_only_pm = fb_only_pm
        # Shadow mode log: features log what they WOULD do without trading
        self._shadow_log_path = Path("/app/logs/shadow_trades.jsonl")
        self._shadow_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._shadow_phantoms: list[dict] = []  # track shadow signal outcomes
        self._regime_params = regime_params
        self._bypass_session_gates = bypass_session_gates
        self._pending_gate_bypass: list[str] | None = None

        # Reuse ManciniLongStrategy's sub-components
        self.strategy = ManciniLongStrategy(
            strategy_params=strategy_params,
            elevator_params=elevator_params,
            exit_params=exit_params,
            risk_params=risk_params,
            session_times=session_times,
            contract=contract,
            min_rr_ratio=min_rr_ratio,
            rth_filter=rth_filter,
            regime_params=regime_params,
        )
        self.signal_aggregator = self.strategy.signal_aggregator
        self.entry_manager = EntryManager(
            session=session_times,
            exit_params=exit_params,
            risk_params=risk_params,
        )
        self.exit_manager = ExitManager(params=exit_params, contract=contract)
        self.position_manager = PositionManager(
            risk_params=risk_params,
            point_value=contract.point_value,
            bypass_loss_limits=bypass_session_gates,
        )
        self.risk_manager = RiskManager(
            risk_params=risk_params,
            session=session_times,
            contract=contract,
        )

        # State
        self._df: Optional[pd.DataFrame] = None
        self._position: Optional[TradePosition] = None
        self._trade_id: Optional[int] = None  # IB parent order ID
        self._entry_timestamp: Optional[datetime] = None  # ET timestamp of entry fill
        self._pattern_type: str = ""
        self._current_signal: Optional[Signal] = None
        self._last_entry_monotonic: float = 0.0  # monotonic time of last entry
        self._current_gate_bypass: list[str] | None = None
        self._bar_count: int = 0
        self._entry_bar_count: int = 0  # bar_count at entry, for bars_held calc
        self._last_trade_bar: int = 0  # bar_count at last entry/exit, for time_since_last_trade
        self._running: bool = False
        self._session_date: date = date.today()

        # Phantom trade tracking — signals that fired but were filtered out
        self._phantom_positions: list[dict] = []

        # Near-miss outcome tracking — setups that almost triggered, track what would have happened
        self._near_miss_phantoms: list[dict] = []

        # Post-exit continuation tracking — what happens after we exit a trade
        self._post_exit_trackers: list[dict] = []

        # Persistent trade log — survives restarts, accumulates data for self-improvement
        self._trade_log_path = Path(os.environ.get("TRADE_LOG", "/app/logs/trades.jsonl"))
        self._trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._all_trades: list[dict] = self._load_trade_log()

        # Mancini Substack overlay result (populated in _initialize_session when enabled)
        self._mancini_overlay_result = None

    def run(self) -> None:
        """Main event loop. Blocks until session ends or shutdown signal."""
        # Add file log sink for dashboard
        log_path = os.environ.get("LOG_FILE", "/app/logs/bot.log")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, rotation="10 MB", retention="7 days",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

        logger.info("=" * 60)
        logger.info("MANCINI IB RUNNER STARTING")
        logger.info(f"  Symbol: {self.bridge.config.symbol}")
        is_paper = self.bridge.config.port in (7497, 4002, 4003)
        logger.info(f"  Port:   {self.bridge.config.port} "
                     f"({'PAPER' if is_paper else 'LIVE'})")
        logger.info("=" * 60)

        # Setup
        self._running = True
        os_signal.signal(os_signal.SIGINT, self._handle_shutdown)
        os_signal.signal(os_signal.SIGTERM, self._handle_shutdown)

        self._session_date = date.today()

        # Connect to IB
        if not self.bridge.connect():
            logger.error("Failed to connect to IB")
            return

        # Initialize session
        if not self._initialize_session():
            logger.error("Session initialization failed")
            self.bridge.disconnect()
            return

        # Start streaming bars (keepUpToDate — IB pushes new bars to us)
        if not self.bridge.start_streaming():
            logger.error("Failed to start bar streaming")
            self.bridge.disconnect()
            return

        logger.info("Listening for streaming bars from IB...")
        self._last_bar_received = _time.monotonic()
        try:
            while self._running:
                # Check if IB disconnected and needs reconnection
                if self.bridge._needs_reconnect:
                    self.bridge.check_reconnect()

                bar = self.bridge.get_latest_bar()
                if bar is not None:
                    self._last_bar_received = _time.monotonic()
                    self._process_bar(bar)
                    self._check_eod(bar)
                else:
                    # Check if we haven't received a bar in too long
                    minutes_since_bar = (_time.monotonic() - self._last_bar_received) / 60
                    if minutes_since_bar > 5 and not _is_market_closed():
                        # Throttle: only log once per 5-minute interval
                        last_alert = getattr(self, "_last_stale_alert", 0.0)
                        if _time.monotonic() - last_alert >= 300:  # 5 min
                            self._last_stale_alert = _time.monotonic()
                            logger.error(
                                f"NO NEW BARS for {minutes_since_bar:.0f} minutes. "
                                f"IB connection may be stale. Last bar: {self._bar_count}"
                            )

                # IB-aware sleep keeps the event loop alive for streaming callbacks
                self.bridge.sleep(self.bridge.config.poll_interval_sec)

                # Session rollover: detect new Globex session (date change)
                # Globex sessions start at 18:00 ET, so the "trading date" rolls
                # over then. Without this, daily loss limits from yesterday block
                # today's signals.
                self._check_session_rollover()

                # Sync position with IB periodically
                self._sync_position()

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        self._log_session_summary()
        self.bridge.disconnect()
        logger.info("IB Runner stopped")

    # ── Session initialization ───────────────────────────────────────

    def _initialize_session(self) -> bool:
        """Initialize strategy from IB historical bars.

        1. Get prior day bars for level initialization
        2. Get current day bars for catchup
        3. Initialize levels
        4. Check for existing position (crash recovery)
        """
        logger.info("Initializing session from IB data...")

        # Get prior day data for level calculation
        prior_day_df = self.bridge.get_prior_day_bars()
        if prior_day_df is not None:
            logger.info(f"Prior day: {len(prior_day_df)} bars loaded")
        else:
            logger.warning("No prior day data available")

        # Get current day bars (for catchup if restarting mid-session)
        all_bars = self.bridge.get_bars(count=400)
        current_day_df = None
        if all_bars is not None:
            today_mask = all_bars.index.date == self._session_date
            current_day_df = all_bars[today_mask]
            if current_day_df.empty:
                current_day_df = None

        # Build daily history for regime filter (needs ~80+ days for EMA)
        # Request daily bars BEFORE intraday to avoid IB pacing violations
        if self._regime_params is not None:
            daily_history = None
            try:
                logger.info("Requesting 1 year of daily bars for regime filter...")
                import time as _t
                _t.sleep(2)  # Avoid IB pacing after prior_day request
                daily_df = self.bridge.get_daily_bars(days=365)
                if daily_df is not None and len(daily_df) > 0:
                    daily_history = daily_df[daily_df.index.date < self._session_date]
                    logger.info(f"Got {len(daily_df)} daily bars, {len(daily_history)} before today")
                else:
                    logger.warning("get_daily_bars returned empty")
            except Exception as e:
                logger.warning(f"IB daily bars failed: {e}")

            # Fallback: build daily bars from intraday (limited ~2 days)
            if (daily_history is None or len(daily_history) == 0) and all_bars is not None and len(all_bars) > 0:
                daily_history = build_daily_bars(all_bars)
                daily_history = daily_history[daily_history.index.date < self._session_date]

            if daily_history is not None and len(daily_history) > 0:
                self.strategy._daily_history = daily_history
                logger.info(f"Regime filter: {len(daily_history)} daily bars for EMA computation")
            else:
                logger.warning("No daily history available for regime filter")

        # Reset strategy state
        self.strategy.reset()
        self.position_manager.start_session(datetime.now())
        self._restore_trade_history()

        # Compute regime state (normally done in process_day, but runner
        # processes bars individually so we trigger it manually)
        if (self.strategy.strategy_params.use_regime_filter
                and self.strategy._daily_history is not None
                and len(self.strategy._daily_history) >= 50):
            from core.regime_filter import compute_regime, RegimeState, Direction, VolRegime
            self.strategy._regime_state = compute_regime(
                self.strategy._daily_history, self.strategy.regime_params
            )
            logger.info(f"Regime: {self.strategy._regime_state.direction.name} "
                        f"(longs={'ON' if self.strategy._regime_state.longs_enabled else 'OFF'}, "
                        f"shorts={'ON' if self.strategy._regime_state.shorts_enabled else 'OFF'})")
        else:
            from core.regime_filter import RegimeState, Direction, VolRegime
            self.strategy._regime_state = RegimeState(
                direction=Direction.NEUTRAL,
                vol_regime=VolRegime.NORMAL,
                longs_enabled=True,
                shorts_enabled=True,
            )
            logger.info("Regime: NEUTRAL (filter disabled or insufficient history)")

        # Daily structure detector — macro bias from daily chart
        if self.strategy.strategy_params.use_daily_structure:
            # Use the daily history already fetched for regime filter
            dh = getattr(self.strategy, "_daily_history", None)
            if dh is not None and len(dh) >= self.strategy.strategy_params.daily_shelf_lookback_days:
                bias = self.signal_aggregator.set_daily_structure(dh)
                snap = self.signal_aggregator.get_daily_structure_snapshot()
                logger.info(
                    f"Daily structure: {bias} | shelf={snap['shelf_price']:.1f} "
                    f"sweep_low={snap['sweep_low']:.1f} move_pos={snap['move_position']:.2f}"
                )
            else:
                logger.info("Daily structure: NEUTRAL (insufficient daily history)")

        if current_day_df is not None:
            self._df = current_day_df
            self._bar_count = len(current_day_df)
        else:
            self._df = pd.DataFrame()
            self._bar_count = 0

        # Initialize levels from prior day
        self.signal_aggregator.initialize_levels(
            self._df if self._df is not None and len(self._df) > 0 else pd.DataFrame(),
            prior_day_df,
        )

        # Mancini Substack level overlay — augments engine-detected levels.
        # Failure modes (missing file, corrupt JSON, parse failure) are all
        # swallowed so the bot runs normally when this isn't available.
        self._mancini_overlay_result = None
        sp = self.strategy.strategy_params
        if getattr(sp, "use_mancini_levels", False):
            try:
                from live.mancini_levels import load as load_mancini_levels
                from core.mancini_overlay import apply_mancini_overlay

                current_price = 0.0
                if self._df is not None and len(self._df) > 0:
                    current_price = float(self._df["close"].iat[-1])

                mancini_data = load_mancini_levels(
                    self._session_date,
                    input_dir=Path(sp.mancini_levels_dir),
                )
                if mancini_data:
                    self._mancini_overlay_result = apply_mancini_overlay(
                        store=self.signal_aggregator.level_store,
                        mancini_data=mancini_data,
                        mode=sp.mancini_mode,
                        confirm_tolerance_pts=sp.mancini_confirm_tolerance_pts,
                        current_price=current_price,
                        timestamp=datetime.now(_ET),
                    )
                    logger.info(
                        f"Mancini overlay applied: mode={sp.mancini_mode} "
                        f"confirmed={self._mancini_overlay_result.confirmed_count} "
                        f"injected={self._mancini_overlay_result.injected_count} "
                        f"shadow={self._mancini_overlay_result.shadow_count} "
                        f"blind_spots={len(self._mancini_overlay_result.blind_spots)}"
                    )
                else:
                    logger.info("Mancini overlay: no parsed levels for session (skipping)")
            except Exception as e:
                logger.warning(f"Mancini overlay failed (non-fatal): {e}")

        # Mancini LLM-extracted plan (Phase 3) — gates signal qualification on
        # mode/danger_zones/no_trade_zones and boosts planned-setup matches.
        # Plan extraction runs nightly via cron regardless; this only loads
        # the JSON when use_mancini_llm_plan is on.
        self._mancini_llm_plan = None
        if getattr(sp, "use_mancini_llm_plan", False):
            try:
                from live.mancini_llm_extract import load_plan as _load_llm_plan

                plan = _load_llm_plan(
                    self._session_date,
                    input_dir=Path(getattr(sp, "mancini_llm_plan_dir", "/app/data")),
                )
                if plan is not None:
                    self._mancini_llm_plan = plan
                    self.signal_aggregator.set_mancini_llm_plan(plan)
                    logger.info(
                        f"Mancini LLM plan loaded for {self._session_date}: "
                        f"lean={plan.lean} mode={plan.mode} "
                        f"setups={len(plan.planned_setups)} "
                        f"danger={len(plan.danger_zones)} "
                        f"no_trade_above={plan.no_trade_above} "
                        f"no_trade_below={plan.no_trade_below}"
                    )
                else:
                    logger.info(
                        f"Mancini LLM plan: no plan available for {self._session_date}"
                    )
            except Exception as e:
                logger.warning(f"Mancini LLM plan load failed (non-fatal): {e}")

        # Catch up on current-day bars (update state, don't trade)
        if self._df is not None and len(self._df) > 0:
            logger.info(f"Catching up on {len(self._df)} bars from session start")
            velocity = compute_velocity(self._df, window=5)
            for i in range(len(self._df)):
                vel = float(velocity.iat[i]) if not np.isnan(velocity.iat[i]) else 0.0
                self.signal_aggregator.update(
                    bar_idx=i,
                    timestamp=self._df.index[i],
                    open_=float(self._df["open"].iat[i]),
                    high=float(self._df["high"].iat[i]),
                    low=float(self._df["low"].iat[i]),
                    close=float(self._df["close"].iat[i]),
                    volume=float(self._df["volume"].iat[i]),
                    velocity=vel,
                    df=self._df,
                )

        # Restore pattern state from prior session (if saved)
        self._load_pattern_state()

        # Check for existing position (crash recovery)
        pos_data = self.bridge.get_position()
        if pos_data and pos_data.get("market_position") in ("long", "short"):
            logger.warning(f"Existing {pos_data['market_position']} position detected -- recovering state")
            self._recover_position(pos_data)

        # Log account info
        acct = self.bridge.get_account_info()
        if acct:
            logger.info(f"Account: balance=${acct.get('balance', 0):,.0f}, "
                         f"equity=${acct.get('equity', 0):,.0f}, "
                         f"server={acct.get('server', 'unknown')}")

        logger.info(f"Session initialized: {self._bar_count} bars, "
                     f"levels initialized from prior day")
        return True

    # ── Bar processing ───────────────────────────────────────────────

    def _process_bar(self, bar: dict) -> None:
        """Process a new bar from IB.

        Same logic as the backtest engine:
        1. Append bar to DataFrame
        2. Compute velocity
        3. Check exits on open position
        4. Check for new signals
        """
        self._bar_count += 1
        ts_str = bar.get("timestamp", "")
        try:
            timestamp = pd.Timestamp(ts_str)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("US/Eastern")
        except (ValueError, TypeError):
            timestamp = pd.Timestamp.now(tz="US/Eastern")

        current_time = timestamp.time()
        open_ = float(bar.get("open", 0))
        high = float(bar.get("high", 0))
        low = float(bar.get("low", 0))
        close = float(bar.get("close", 0))
        volume = float(bar.get("volume", 0))
        self._last_price = close  # Track for exit fill fallback

        logger.info(f"BAR #{self._bar_count}: {timestamp.strftime('%H:%M')} "
                     f"O={open_:.2f} H={high:.2f} L={low:.2f} C={close:.2f} V={volume:.0f}")

        # Append to DataFrame
        new_row = pd.DataFrame(
            {"open": [open_], "high": [high], "low": [low],
             "close": [close], "volume": [volume]},
            index=pd.DatetimeIndex([timestamp]),
        )
        if self._df is None or len(self._df) == 0:
            self._df = new_row
        else:
            self._df = pd.concat([self._df, new_row])
            # Keep last 400 bars
            if len(self._df) > 400:
                self._df = self._df.iloc[-400:]

        if len(self._df) < 6:
            return

        # Compute velocity
        velocity = compute_velocity(self._df, window=5)
        vel = float(velocity.iat[-1]) if not np.isnan(velocity.iat[-1]) else 0.0

        # Step 0: Update phantom trades (rejected signals we're still tracking)
        if self._phantom_positions:
            self._update_phantoms(high, low, close, timestamp)

        # Step 1: Check exits on existing position
        if self._position is not None and self._position.is_open:
            exit_action = self.exit_manager.update(self._position, high, low, close)
            if exit_action is not None:
                self._handle_exit_action(exit_action, timestamp)
                if self._position is not None and not self._position.is_open:
                    self.position_manager.close_position(
                        exit_price=exit_action.exit_price,
                        timestamp=timestamp,
                        exit_reason=exit_action.reason,
                        pattern_type=self._pattern_type,
                        entry_time=self._entry_timestamp,
                    )
                    # Log rich exit to persistent trade log
                    if self.position_manager.session and self.position_manager.session.trades:
                        self._log_trade(
                            self.position_manager.session.trades[-1],
                            self._current_signal,
                            "exit",
                        )
                    # Start post-exit continuation tracker
                    self._start_post_exit_tracker(
                        trade_id=self._trade_id,
                        exit_price=exit_action.exit_price,
                        direction=getattr(self._position, "direction", "long"),
                        exit_reason=exit_action.reason,
                    )
                    self._position = None
                    self._trade_id = None
                self._write_status()

        # Step 2: Check for new signals (only if no position and not done)
        if self._position is None or not self._position.is_open:
            if self.position_manager.is_done_for_day:
                self._write_status()
                return

            bar_idx = len(self._df) - 1  # Index into current DataFrame, not cumulative count

            # Build market data and session context for LQS scoring
            try:
                _lqs_market_data = fetch_market_snapshot()
            except Exception:
                _lqs_market_data = None
            _lqs_session_ctx = {
                "session_date": self._session_date,
                "current_price": close,
                "session_high": float(self._df["high"].max()),
                "session_low": float(self._df["low"].min()),
                "bar_count": self._bar_count,
            }

            signal = self.signal_aggregator.update(
                bar_idx=bar_idx,
                timestamp=timestamp,
                open_=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                velocity=vel,
                df=self._df,
                market_data=_lqs_market_data,
                session_context=_lqs_session_ctx,
            )

            if signal is not None:
                self._evaluate_and_enter(signal, current_time, timestamp)

        # Step 0b: Update near-miss phantoms
        if self._near_miss_phantoms:
            self._update_near_miss_phantoms(high, low, close, timestamp)

        # Step 3: Update post-exit continuation trackers
        if self._post_exit_trackers:
            self._update_post_exit_trackers(high, low, timestamp)

        # Log any new near-misses to persistent trade log
        fb = self.signal_aggregator.failed_breakdown
        if hasattr(fb, "near_misses") and fb.near_misses:
            for nm in fb.near_misses:
                if not nm.get("_logged"):
                    try:
                        # Build enriched near-miss record with full signal details
                        volume_at_signal = None
                        try:
                            if self._df is not None and len(self._df) > 0:
                                volume_at_signal = float(self._df["volume"].iat[-1])
                        except Exception:
                            pass
                        now_et = datetime.now(_ET)
                        level_price = nm.get("level_price", 0)
                        stop_buffer = self.strategy.strategy_params.fb_stop_buffer_pts
                        entry_price = close  # would have entered at current close
                        stop_price = level_price - stop_buffer
                        risk = abs(entry_price - stop_price)
                        target_price = entry_price + risk * 1.0  # 1:1 R:R

                        nm_record = {
                            "event": "near_miss",
                            "timestamp": str(timestamp),
                            "session_date": str(self._session_date),
                            "pattern": "FAILED_BREAKDOWN",
                            "signal_type": "FAILED_BREAKDOWN",
                            "direction": "long",
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "target_1": target_price,
                            "target_2": None,
                            "rr_ratio": round(risk / risk, 2) if risk > 0 else None,
                            "level_price": level_price,
                            "level_type": nm.get("level_type"),
                            "sweep_depth_pts": nm.get("sweep_depth_pts"),
                            "confirmation_type": nm.get("confirmation_type"),
                            "failure_reason": nm.get("failure_reason", ""),
                            "session_window": self._get_session_window(now_et.time()).get("detail", ""),
                            "bar_count": self._bar_count,
                            "volume_at_signal": volume_at_signal,
                        }
                        # Merge any extra fields from the detector's nm dict
                        for k, v in nm.items():
                            if k not in nm_record and k != "_logged":
                                nm_record[k] = v
                        with open(self._trade_log_path, "a") as f:
                            f.write(json.dumps(nm_record, default=str) + "\n")
                        nm["_logged"] = True

                        # Stable key: (timestamp, level_price, failure_reason)
                        nm_key = f"{nm.get('timestamp')}_{level_price:.2f}_{nm.get('failure_reason','')}"
                        if risk > 0 and not any(
                            p.get("near_miss_key") == nm_key
                            for p in self._near_miss_phantoms
                        ):
                            self._near_miss_phantoms.append({
                                "near_miss_key": nm_key,
                                "direction": "long",  # FB near-misses are always long
                                "entry_price": entry_price,
                                "stop_price": stop_price,
                                "target_price": target_price,
                                "level_price": level_price,
                                "risk_pts": risk,
                                "entry_time": str(timestamp),
                                "failure_reason": nm.get("failure_reason", ""),
                                "high_since": entry_price,
                                "low_since": entry_price,
                                "resolved": False,
                                "result": "",
                            })
                    except Exception:
                        pass

        # Check for manual force-trade trigger file
        self._check_force_trade(close, timestamp)

        # Flush shadow mode events to disk and track outcomes
        self._flush_shadow_events()
        self._update_shadow_phantoms(high, low)

        # Write status for dashboard after every bar
        self._write_status()

    def _evaluate_and_enter(self, signal: Signal, current_time: time, timestamp: datetime) -> None:
        """Evaluate signal through risk/entry gates, execute if approved.

        In bypass_session_gates mode, time-based gates are recorded but not
        enforced — the trade is taken anyway with a 'gate_bypassed' marker.
        Non-time gates (max trades, done for day) still block.
        """
        gates_that_would_fire: list[str] = []

        # Gate: Evening session (18:00-22:00)
        if time(18, 0) <= current_time < time(22, 0):
            if self._bypass_session_gates:
                gates_that_would_fire.append("Globex Evening (6-10PM)")
            else:
                logger.debug(f"Evening block (18:00-22:00): skipping signal")
                self._add_phantom(signal, "window:evening_block", timestamp)
                return

        # Gate: FB-only filter for afternoon (15:00-16:50)
        if self._fb_only_pm and time(15, 0) <= current_time <= time(16, 50):
            if signal.signal_type.name != "FAILED_BREAKDOWN":
                if self._bypass_session_gates:
                    gates_that_would_fire.append("RTH Late Day FB-Only (3-5PM)")
                else:
                    logger.info(
                        f"PM FB-only filter: rejecting {signal.signal_type.name} "
                        f"@ {signal.entry_price:.2f} (only FBs allowed after 3PM)"
                    )
                    self._add_phantom(signal, "window:pm_fb_only_filter", timestamp)
                    return

        # Risk check
        risk_check = self.risk_manager.validate_entry(
            signal, current_time, self.position_manager
        )
        if not risk_check.passed:
            reason = risk_check.reason
            # Bypass time gates so all strategies can fire any time.
            # Quality gates (R:R, stop width) are ALWAYS enforced — trades must be real.
            TIME_GATES = ["chop zone", "european dead zone", "evening block", "fb blocked hour"]
            is_bypassable = any(g in reason.lower() for g in TIME_GATES)
            if is_bypassable and self._bypass_session_gates:
                gates_that_would_fire.append(f"{reason}")
            else:
                logger.warning(
                    f"PHANTOM SIGNAL (rejected by risk: {reason}): "
                    f"{signal.signal_type.name} @ {signal.entry_price:.2f} "
                    f"stop={signal.stop_price:.2f} T1={signal.target_1:.2f} "
                    f"R:R={signal.rr_ratio_t1:.1f} level={signal.pattern.level.price:.2f}"
                )
                self._add_phantom(signal, f"risk:{reason}", timestamp)
                return

        # Entry decision
        entry = self.entry_manager.evaluate(
            signal=signal,
            current_time=current_time,
            trades_today=self.position_manager.trades_today,
            is_in_profit_protection=self.position_manager.is_profit_protection,
            daily_pnl_pts=self.position_manager.daily_pnl_pts,
        )
        if not entry.should_enter:
            reason = entry.reason
            # Chop zone / EOD flatten in entry manager are also time gates
            ENTRY_TIME_GATES = ["chop zone", "eod flatten"]
            is_time_gate = any(g in reason.lower() for g in ENTRY_TIME_GATES)
            if is_time_gate and self._bypass_session_gates:
                gates_that_would_fire.append(f"{reason}")
            else:
                logger.warning(
                    f"PHANTOM SIGNAL (entry declined: {reason}): "
                    f"{signal.signal_type.name} @ {signal.entry_price:.2f} "
                    f"stop={signal.stop_price:.2f} T1={signal.target_1:.2f} "
                    f"R:R={signal.rr_ratio_t1:.1f} level={signal.pattern.level.price:.2f}"
                )
                self._add_phantom(signal, f"entry:{reason}", timestamp)
                return

        # Record gate bypass info for logging
        self._pending_gate_bypass = gates_that_would_fire if gates_that_would_fire else None
        if gates_that_would_fire:
            logger.info(
                f"COLLECTION MODE: Taking {signal.signal_type.name} @ {signal.entry_price:.2f} "
                f"(production would skip: {', '.join(gates_that_would_fire)})"
            )
            # Entry was rejected by time gate — use signal values directly
            if entry.contracts <= 0:
                entry = EntryDecision(
                    should_enter=True,
                    signal=signal,
                    contracts=self.exit_manager.params.default_contracts or 1,
                    reason=f"Bypass: {', '.join(gates_that_would_fire)}",
                    entry_price=signal.entry_price,
                    stop_price=signal.stop_price,
                )

        # Final contracts sanity check before sending to IB
        if entry.contracts <= 0:
            logger.error(
                f"REJECTED: entry.contracts={entry.contracts} for {signal.signal_type.name} — "
                f"would create ghost order"
            )
            return

        # Execute via IB — bracket order (market + SL + TP as OCO)
        # send_entry waits for fill confirmation before returning
        trade_id, fill_price = self.bridge.send_entry(
            quantity=entry.contracts,
            sl=entry.stop_price,
            tp=signal.target_1,
            direction=signal.direction,
            comment=f"Mancini:{signal.signal_type.name}",
            tp_fraction=self.exit_params.t1_exit_fraction,
        )

        if trade_id is None:
            logger.error("Entry order rejected or timed out — no position created")
            self._add_phantom(signal, "ib:entry_rejected_or_timeout", timestamp)
            return

        # Use IB's actual fill price if available, fall back to signal entry
        actual_entry_price = fill_price if fill_price > 0 else entry.entry_price
        if fill_price > 0 and abs(fill_price - entry.entry_price) > 1.0:
            logger.warning(
                f"Fill price slippage: expected={entry.entry_price:.2f}, "
                f"actual={fill_price:.2f} ({fill_price - entry.entry_price:+.2f} pts)"
            )

        # Create local position tracking (only after confirmed fill)
        self._position = self.exit_manager.create_position(
            entry_price=actual_entry_price,
            stop_price=entry.stop_price,
            target_1=signal.target_1,
            target_2=signal.target_2,
            contracts=entry.contracts,
            direction=signal.direction.lower(),
        )
        self._trade_id = trade_id
        self._pattern_type = signal.pattern.pattern_type
        self._current_signal = signal
        self._entry_timestamp = datetime.now(_ET)
        self._entry_bar_count = self._bar_count
        self._last_entry_monotonic = _time.monotonic()
        self._current_gate_bypass = self._pending_gate_bypass
        self._pending_gate_bypass = None
        self.position_manager.open_position(self._position, timestamp, self._pattern_type)

        logger.info(
            f"ENTRY: {entry.contracts} {self.contract.symbol} @ {actual_entry_price:.2f} "
            f"stop={entry.stop_price:.2f} T1={signal.target_1:.2f} "
            f"R:R={signal.rr_ratio_t1:.1f} [{signal.signal_type.name}]"
        )
        # Log rich entry to persistent trade log
        self._log_trade(self._position, signal, "entry")

    def _handle_exit_action(self, action: ExitAction, timestamp: datetime) -> None:
        """Translate ExitAction from ExitManager into IB orders."""
        if self._trade_id is None:
            return

        if action.new_phase == ExitPhase.CLOSED:
            self.bridge.flatten(reason=action.reason)
            logger.info(f"EXIT: flatten -- {action.reason}")

        elif action.reason.startswith("Target 1"):
            if action.contracts_to_close > 0 and self._position and self._position.remaining_contracts > 0:
                self.bridge.partial_exit(
                    trade_id=self._trade_id,
                    quantity=action.contracts_to_close,
                    new_sl=action.new_stop,
                    reason=action.reason,
                )
            else:
                self.bridge.flatten(reason=action.reason)
            logger.info(f"EXIT: partial {action.contracts_to_close} @ T1, "
                         f"new stop={action.new_stop:.2f}")
            self._log_partial_exit(action, timestamp)

        elif action.reason.startswith("Target 2"):
            self.bridge.partial_exit(
                trade_id=self._trade_id,
                quantity=action.contracts_to_close,
                new_sl=action.new_stop,
                reason=action.reason,
            )
            self._log_partial_exit(action, timestamp)

        else:
            # Stop update (trailing)
            self.bridge.update_stop(
                trade_id=self._trade_id,
                new_sl=action.new_stop,
                reason=action.reason,
            )

    def _log_partial_exit(self, action: ExitAction, timestamp: datetime) -> None:
        """Log a partial exit (T1/T2) to trades.jsonl so the dashboard can show it."""
        try:
            if not self._position:
                return
            pos = self._position
            direction = getattr(pos, "direction", "long")
            if direction == "short":
                pnl_pts = (pos.entry_price - action.exit_price) * action.contracts_to_close
            else:
                pnl_pts = (action.exit_price - pos.entry_price) * action.contracts_to_close
            pnl_dollars = pnl_pts * self.contract.point_value

            now_et = datetime.now(_ET)
            record = {
                "event": "partial_exit",
                "trade_id": self._trade_id,
                "timestamp": now_et.isoformat(),
                "session_date": str(self._session_date),
                "symbol": self.bridge.config.symbol,
                "entry_price": pos.entry_price,
                "exit_price": action.exit_price,
                "pnl_pts": round(pnl_pts, 2),
                "pnl_dollars": round(pnl_dollars, 2),
                "contracts": action.contracts_to_close,
                "total_contracts": pos.total_contracts,
                "remaining_contracts": pos.remaining_contracts,
                "direction": direction,
                "pattern_type": self._pattern_type,
                "exit_reason": action.reason,
                "new_stop": action.new_stop,
            }
            with open(self._trade_log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    def _sync_position(self) -> None:
        """Sync local position state with IB actual position.

        Handles cases where IB's bracket order (stop/target) fires
        independently of Python's exit logic.

        Requires 3 consecutive None reads before confirming position closed,
        to avoid false closures from transient IB connection issues.
        """
        if self._position is None or not self._position.is_open:
            return

        # Grace period: IB needs time to register the position after entry
        # or after recovery. Without this, get_position() returns None
        # immediately and we falsely conclude the bracket was filled.
        # 45s accounts for the fill confirmation wait (up to 30s) plus buffer.
        elapsed = _time.monotonic() - self._last_entry_monotonic
        if elapsed < 45.0:
            return

        ib_pos = self.bridge.get_position()
        if ib_pos is None:
            # Require 3 consecutive None reads to confirm position truly closed
            self._sync_none_count = getattr(self, "_sync_none_count", 0) + 1
            if self._sync_none_count < 3:
                logger.debug(f"IB position returned None ({self._sync_none_count}/3), waiting for confirmation...")
                return

            # Position confirmed closed on IB side — retrieve actual fill price
            # Try up to 3 times with 2s delay to let IB propagate fill data
            fill_price, exit_type = 0.0, "unknown"
            for attempt in range(3):
                fill_price, exit_type = self.bridge.get_bracket_fill_price(self._trade_id)
                if fill_price > 0:
                    logger.info(
                        f"Position closed on IB side ({exit_type} filled @ {fill_price:.2f}, "
                        f"confirmed 3x, attempt {attempt + 1})"
                    )
                    break
                if attempt < 2:
                    logger.debug(f"Fill price not yet available (attempt {attempt + 1}/3), retrying in 2s...")
                    _time.sleep(2)

            if fill_price <= 0:
                # Final fallback: use last known market price as estimate
                last_price = getattr(self, "_last_price", 0.0)
                logger.warning(
                    f"Position closed on IB side (bracket filled, confirmed 3x) "
                    f"but fill price unavailable after 3 attempts — "
                    f"using last market price {last_price:.2f} as estimate"
                )
                fill_price = last_price
            self._sync_none_count = 0
            # Calculate PnL before closing — IB bracket exits bypass ExitManager
            contracts = self._position.remaining_contracts
            if self._position.direction == "long":
                pnl = (fill_price - self._position.entry_price) * contracts
            else:
                pnl = (self._position.entry_price - fill_price) * contracts
            self._position.realized_pnl_pts += pnl
            self._position.remaining_contracts = 0
            self._position.phase = ExitPhase.CLOSED
            now = datetime.now()
            self.position_manager.close_position(
                exit_price=fill_price,
                timestamp=now,
                exit_reason=f"IB bracket {exit_type}" if exit_type != "unknown" else "IB bracket fill",
                pattern_type=self._pattern_type,
                entry_time=self._entry_timestamp,
            )
            # Log exit to persistent trade log
            if self.position_manager.session and self.position_manager.session.trades:
                self._log_trade(
                    self.position_manager.session.trades[-1],
                    self._current_signal,
                    "exit",
                )
            self._position = None
            self._trade_id = None
        else:
            # Position still open — reset the None counter
            self._sync_none_count = 0

    def _check_force_trade(self, close: float, timestamp) -> None:
        """Check for /app/logs/force_trade.json trigger file.

        Allows manual test trades without modifying the running engine.
        File format: {"direction": "long", "tp_pts": 3, "sl_pts": 3}
        """
        trigger_path = Path(os.environ.get("FORCE_TRADE_FILE", "/app/logs/force_trade.json"))
        if not trigger_path.exists():
            return
        if self._position is not None and self._position.is_open:
            logger.warning("Force trade ignored: already in position")
            trigger_path.unlink()
            return

        try:
            params = json.loads(trigger_path.read_text())
            trigger_path.unlink()

            direction = params.get("direction", "long")
            tp_pts = params.get("tp_pts", 3.0)
            sl_pts = params.get("sl_pts", 3.0)
            qty = int(params.get("quantity", 1))

            if direction == "long":
                tp = close + tp_pts
                sl = close - sl_pts
            else:
                tp = close - tp_pts
                sl = close + sl_pts

            logger.warning(f"FORCE TRADE: {direction.upper()} {qty} @ ~{close:.2f}, "
                           f"TP={tp:.2f}, SL={sl:.2f}")

            trade_id, fill_price = self.bridge.send_entry(
                quantity=qty, sl=sl, tp=tp,
                direction=direction, comment="ForceTest",
            )
            if trade_id:
                self._position = self.exit_manager.create_position(
                    entry_price=close,
                    stop_price=sl,
                    target_1=tp,
                    target_2=tp + tp_pts,
                    contracts=qty,
                    direction=direction,
                )
                self._trade_id = trade_id
                self._pattern_type = "FORCE_TEST"
                self.position_manager.open_position(self._position, timestamp, "FORCE_TEST")
                self._log_trade(self._position, None, "entry")
                logger.warning(f"Force trade placed: orderId={trade_id}")
            else:
                logger.error("Force trade FAILED: send_entry returned None")

        except Exception as e:
            logger.error(f"Force trade error: {e}")
            if trigger_path.exists():
                trigger_path.unlink()

    def _recover_position(self, pos_data: dict) -> None:
        """Reconstruct TradePosition from IB position on restart.

        IB doesn't track SL/TP at the position level (always returns 0.0),
        so we look up the original entry in trades.jsonl to get the signal's
        stop/target. If no match, compute sensible defaults from pattern type.
        """
        entry = pos_data.get("price_open", 0)
        qty = int(pos_data.get("volume", 1))
        direction = pos_data.get("market_position", "long")

        # Priority 1: Read actual bracket orders from IB (most reliable source)
        stop = 0.0
        target = 0.0
        pattern_type = "recovered"
        signal_data = None
        bracket_info = self.bridge.get_bracket_orders()

        if bracket_info.get("sl", 0) > 0 and bracket_info.get("tp", 0) > 0:
            stop = bracket_info["sl"]
            target = bracket_info["tp"]
            logger.info(f"Recovered bracket orders from IB: SL={stop:.2f}, TP={target:.2f}")
            # Populate _active_orders so _sync_position works
            self.bridge._active_orders[0] = {
                "parent": None,
                "tp": None,
                "sl": None,
                "tp_order_id": bracket_info.get("tp_order_id"),
                "sl_order_id": bracket_info.get("sl_order_id"),
                "quantity": qty,
                "direction": direction,
            }

        # Priority 2: Look up the last entry in trade log matching this position
        if stop == 0 or target == 0:
            for record in reversed(self._all_trades):
                if record.get("event") != "entry":
                    continue
                rec_entry = record.get("entry_price", 0)
                # Match by entry price (within 1 pt) and same session date
                if abs(rec_entry - entry) < 1.0 and record.get("session_date") == str(self._session_date):
                    sig = record.get("signal", {})
                    if sig:
                        if stop == 0:
                            stop = sig.get("stop", 0)
                        if target == 0:
                            target = sig.get("target_1", 0)
                        signal_data = sig
                    pattern_type = record.get("pattern_type", "recovered")
                    logger.info(f"Position recovery: matched trade log entry "
                                f"(pattern={pattern_type}, stop={stop:.2f}, target={target:.2f})")
                    break

        # Priority 3: If no IB bracket or trade log match, compute sensible defaults
        if stop == 0 or target == 0:
            sp = self.strategy.strategy_params
            if direction == "long":
                if stop == 0:
                    stop = entry - sp.fb_stop_buffer_pts
                if target == 0:
                    risk = abs(entry - stop)
                    target = entry + max(risk, 8.0)  # at least 8 pts target
            else:  # short
                if stop == 0:
                    stop = entry + sp.bd_stop_buffer_pts
                if target == 0:
                    risk = abs(stop - entry)
                    target = entry - max(risk, 8.0)
            logger.warning(f"Position recovery: no trade log match, using defaults "
                           f"(stop={stop:.2f}, target={target:.2f})")

        target_2 = target + (10 if direction == "long" else -10)

        self._position = TradePosition(
            entry_price=entry,
            stop_price=stop,
            target_1=target,
            target_2=target_2,
            total_contracts=qty,
            remaining_contracts=qty,
            direction=direction,
        )
        self._trade_id = 0  # Unknown, but mark as having a position
        self._pattern_type = pattern_type
        self._last_entry_monotonic = _time.monotonic()  # prevent immediate sync close
        # Recover entry timestamp from trade log if available
        if signal_data:
            for record in reversed(self._all_trades):
                if record.get("event") == "entry" and abs(record.get("entry_price", 0) - entry) < 1.0:
                    try:
                        self._entry_timestamp = datetime.fromisoformat(record.get("timestamp", ""))
                    except (ValueError, TypeError):
                        self._entry_timestamp = datetime.now(_ET)
                    break
        if not getattr(self, "_entry_timestamp", None):
            self._entry_timestamp = datetime.now(_ET)
        logger.warning(f"Recovered position: {direction.upper()} {qty} @ {entry:.2f}, "
                        f"SL={stop:.2f}, TP={target:.2f} [{pattern_type}]")

    def _restore_trade_history(self) -> None:
        """Rebuild today's trade history from trades.jsonl so restarts don't lose it.

        Scans the persistent log for completed trades (entry+exit pairs) from
        today's session date and reconstructs TradeRecord objects in the
        position manager's session.
        """
        today = str(self._session_date)
        entries = {}  # entry_price -> entry record
        exits = {}    # entry_price -> exit record

        for record in self._all_trades:
            if record.get("session_date") != today:
                continue
            evt = record.get("event")
            ep = record.get("entry_price", 0)
            if evt == "entry" and ep > 0:
                entries[round(ep, 2)] = record
            elif evt == "exit" and ep > 0:
                exits[round(ep, 2)] = record

        restored = 0
        for ep_key, entry_rec in entries.items():
            exit_rec = exits.get(ep_key)
            if exit_rec is None:
                continue  # still open or no exit logged

            sig = entry_rec.get("signal", {})
            try:
                entry_time = datetime.fromisoformat(entry_rec.get("timestamp", ""))
            except (ValueError, TypeError):
                entry_time = datetime.now()
            try:
                exit_time = datetime.fromisoformat(exit_rec.get("timestamp", ""))
            except (ValueError, TypeError):
                exit_time = datetime.now()

            pnl_pts = exit_rec.get("pnl_pts", 0) or 0
            pnl_dollars = exit_rec.get("pnl_dollars", 0) or 0

            trade = TradeRecord(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_rec.get("entry_price", 0),
                avg_exit_price=exit_rec.get("exit_price", 0) or 0,
                contracts=entry_rec.get("contracts", 1) or 1,
                pnl_pts=pnl_pts,
                pnl_dollars=pnl_dollars,
                pattern_type=entry_rec.get("pattern_type", "unknown"),
                exit_reason=exit_rec.get("exit_reason", "unknown"),
                stop_price=sig.get("stop", 0),
                target_1=sig.get("target_1", 0),
                direction=sig.get("type", "long").lower() if "SHORT" not in sig.get("type", "") else "short",
            )
            self.position_manager.session.trades.append(trade)
            self.position_manager.session.daily_pnl_pts += pnl_pts
            self.position_manager.session.daily_pnl_dollars += pnl_dollars
            restored += 1

        if restored > 0:
            logger.info(f"Restored {restored} trade(s) from today's log "
                        f"(PnL: {self.position_manager.session.daily_pnl_pts:+.1f} pts)")

    # ── Persistent trade log ────────────────────────────────────────

    def _load_trade_log(self) -> list[dict]:
        """Load all historical trades from the persistent JSONL log."""
        trades = []
        try:
            if self._trade_log_path.exists():
                for line in self._trade_log_path.read_text().splitlines():
                    if line.strip():
                        trades.append(json.loads(line))
                logger.info(f"Loaded {len(trades)} historical trades from {self._trade_log_path}")
        except Exception as e:
            logger.warning(f"Failed to load trade log: {e}")
        return trades

    def _start_post_exit_tracker(
        self, trade_id: Optional[int], exit_price: float, direction: str, exit_reason: str
    ) -> None:
        """Begin tracking price action after a trade exit.

        Tracks high/low for 60 bars (1 hour) to measure whether our exit
        was premature (price continued) or well-timed (price reversed).
        """
        tracker = {
            "trade_id": trade_id,
            "exit_bar": self._bar_count,
            "exit_price": exit_price,
            "direction": direction,
            "exit_reason": exit_reason,
            "high_after": exit_price,
            "low_after": exit_price,
            "bars_tracked": 0,
            "max_bars": 60,
        }
        self._post_exit_trackers.append(tracker)
        # Cap at 10 trackers to avoid memory growth
        if len(self._post_exit_trackers) > 10:
            self._post_exit_trackers = self._post_exit_trackers[-10:]
        logger.info(
            f"POST-EXIT TRACKER started: trade_id={trade_id} "
            f"exit_price={exit_price:.2f} direction={direction} reason={exit_reason}"
        )

    def _update_post_exit_trackers(self, high: float, low: float, timestamp) -> None:
        """Update all active post-exit trackers with latest bar data.

        When a tracker reaches max_bars, compute continuation/reversal metrics
        and write a post_exit_analysis record to trades.jsonl.
        """
        completed = []
        for tracker in self._post_exit_trackers:
            tracker["high_after"] = max(tracker["high_after"], high)
            tracker["low_after"] = min(tracker["low_after"], low)
            tracker["bars_tracked"] += 1

            if tracker["bars_tracked"] >= tracker["max_bars"]:
                # Compute continuation and reversal
                exit_price = tracker["exit_price"]
                direction = tracker["direction"]
                if direction == "short":
                    continuation_pts = round(exit_price - tracker["low_after"], 2)
                    reversal_pts = round(tracker["high_after"] - exit_price, 2)
                else:
                    continuation_pts = round(tracker["high_after"] - exit_price, 2)
                    reversal_pts = round(exit_price - tracker["low_after"], 2)

                record = {
                    "event": "post_exit_analysis",
                    "trade_id": tracker["trade_id"],
                    "timestamp": str(timestamp),
                    "session_date": str(self._session_date),
                    "exit_price": exit_price,
                    "direction": direction,
                    "exit_reason": tracker["exit_reason"],
                    "exit_bar": tracker["exit_bar"],
                    "bars_tracked": tracker["bars_tracked"],
                    "high_after": tracker["high_after"],
                    "low_after": tracker["low_after"],
                    "post_exit_continuation_pts": continuation_pts,
                    "post_exit_reversal_pts": reversal_pts,
                }
                try:
                    with open(self._trade_log_path, "a") as f:
                        f.write(json.dumps(record, default=str) + "\n")
                    logger.info(
                        f"POST-EXIT ANALYSIS: trade_id={tracker['trade_id']} "
                        f"direction={direction} reason={tracker['exit_reason']} "
                        f"continuation={continuation_pts}pts reversal={reversal_pts}pts"
                    )
                except Exception as e:
                    logger.warning(f"Failed to write post-exit analysis: {e}")
                completed.append(tracker)

        for t in completed:
            self._post_exit_trackers.remove(t)

    def _collect_signal_context(self) -> dict:
        """Snapshot the engine's market context at the current bar.

        Returns a dict with the same shape used by ``_log_trade`` for
        live entries, so phantom rows can carry identical features for
        downstream ML training. Captured at the moment of the call —
        downstream consumers should call this when the signal is
        identified, not later when its outcome resolves.
        """
        last_price = 0.0
        session_high = 0.0
        session_low = 999999.0
        if self._df is not None and len(self._df) > 0:
            last_price = float(self._df["close"].iat[-1])
            session_high = float(self._df["high"].max())
            session_low = float(self._df["low"].min())

        nearby_levels: list[dict] = []
        if hasattr(self.signal_aggregator, "level_store"):
            for lv in self.signal_aggregator.level_store.get_active():
                dist = abs(lv.price - last_price)
                if dist < 30:
                    nearby_levels.append({
                        "price": lv.price,
                        "type": lv.level_type.name,
                        "touches": lv.touch_count,
                        "distance": round(lv.price - last_price, 2),
                    })
            nearby_levels.sort(key=lambda x: abs(x["distance"]))

        regime_info: dict = {}
        if getattr(self, "strategy", None) is not None and self.strategy._regime_state:
            rs = self.strategy._regime_state
            regime_info = {
                "direction": rs.direction.name,
                "longs_enabled": rs.longs_enabled,
                "shorts_enabled": rs.shorts_enabled,
                "ema_slope": getattr(rs, "ema_slope", None),
            }

        try:
            session_window = self._get_session_window(datetime.now(_ET).time())
        except Exception:
            session_window = {"detail": "", "label": ""}

        return {
            "bar_count": self._bar_count,
            "last_price": last_price,
            "session_high": session_high,
            "session_low": session_low,
            "session_range": round(session_high - session_low, 2),
            "session_window": session_window.get("detail", ""),
            "regime": regime_info,
            "nearby_levels": nearby_levels[:5],
        }

    def _log_trade(self, trade_record, signal: Optional[Signal], context: str) -> None:
        """Append a rich trade record to the persistent JSONL log.

        Captures everything needed for future analysis and self-improvement:
        market context, regime, session window, nearby levels, pattern details,
        entry path classification, exit excursion stats, and sizing info.
        """
        try:
            # Gather market context at trade time
            last_price = 0.0
            session_high = 0.0
            session_low = 999999.0
            bar_count = self._bar_count
            if self._df is not None and len(self._df) > 0:
                last_price = float(self._df["close"].iat[-1])
                session_high = float(self._df["high"].max())
                session_low = float(self._df["low"].min())

            # Nearby levels for context
            nearby_levels = []
            if hasattr(self.signal_aggregator, "level_store"):
                for lv in self.signal_aggregator.level_store.get_active():
                    dist = abs(lv.price - last_price)
                    if dist < 30:  # within 30 pts
                        nearby_levels.append({
                            "price": lv.price,
                            "type": lv.level_type.name,
                            "touches": lv.touch_count,
                            "distance": round(lv.price - last_price, 2),
                        })
                nearby_levels.sort(key=lambda x: abs(x["distance"]))

            # Full level store snapshot for ML analysis
            all_levels = []
            if hasattr(self.signal_aggregator, "level_store"):
                for lv in self.signal_aggregator.level_store.get_active():
                    all_levels.append({
                        "price": round(lv.price, 2),
                        "type": lv.level_type.name,
                        "touches": lv.touch_count,
                        "rally_pts": round(lv.rally_from_low_pts, 1) if lv.rally_from_low_pts else 0,
                        "age_bars": bar_count - getattr(lv, '_created_bar', 0) if hasattr(lv, '_created_bar') else None,
                    })
                all_levels.sort(key=lambda x: x["price"])

            # Regime state
            regime_info = {}
            if self.strategy._regime_state:
                rs = self.strategy._regime_state
                regime_info = {
                    "direction": rs.direction.name,
                    "longs_enabled": rs.longs_enabled,
                    "shorts_enabled": rs.shorts_enabled,
                    "ema_slope": getattr(rs, "ema_slope", None),
                }

            # Session window
            now_et = datetime.now(_ET)
            session_window = self._get_session_window(now_et.time())

            # Build the rich record
            record = {
                "event": context,  # "entry" or "exit"
                "trade_id": self._trade_id,
                "timestamp": now_et.isoformat(),
                "session_date": str(self._session_date),
                "symbol": self.bridge.config.symbol,
                "bar_count": bar_count,
                "last_price": last_price,
                "session_high": session_high,
                "session_low": session_low,
                "session_range": round(session_high - session_low, 2),
                "session_window": session_window.get("detail", ""),
                "regime": regime_info,
                "nearby_levels": nearby_levels[:5],
                "all_levels": all_levels,
                "bars_since_last_trade": self._bar_count - self._last_trade_bar,
            }

            # Trade-specific data
            if hasattr(trade_record, "entry_price"):
                record.update({
                    "entry_price": trade_record.entry_price,
                    "exit_price": getattr(trade_record, "avg_exit_price", None),
                    "pnl_pts": getattr(trade_record, "pnl_pts", None),
                    "pnl_dollars": getattr(trade_record, "pnl_dollars", None),
                    "contracts": getattr(trade_record, "contracts", None) or getattr(trade_record, "total_contracts", 1),
                    "direction": getattr(trade_record, "direction", "long"),
                    "pattern_type": getattr(trade_record, "pattern_type", self._pattern_type),
                    "exit_reason": getattr(trade_record, "exit_reason", None),
                    "entry_time": str(getattr(trade_record, "entry_time", "")),
                    "exit_time": str(getattr(trade_record, "exit_time", "")),
                })

            # Signal details (for entries)
            if signal is not None:
                record["signal"] = {
                    "type": signal.signal_type.name,
                    "entry": signal.entry_price,
                    "stop": signal.stop_price,
                    "target_1": signal.target_1,
                    "target_2": getattr(signal, "target_2", None),
                    "rr_ratio": round(signal.rr_ratio_t1, 2),
                    "level_price": signal.pattern.level.price,
                    "level_type": signal.pattern.level.level_type.name,
                    "lqs": getattr(signal, "lqs", None),
                }
                # Human-readable trade explanation
                record["reason"] = self._build_trade_reason(signal, regime_info, session_window)
                record["daily_bias"] = self.signal_aggregator.daily_bias

            # All signals evaluated on the entry bar (not just the winner)
            if context == "entry":
                try:
                    record["signals_evaluated"] = self.signal_aggregator.bar_signals
                except Exception:
                    record["signals_evaluated"] = []

            # ── ENRICHED ENTRY FIELDS ──
            if context == "entry" and signal is not None:
                pattern = signal.pattern

                # Sweep depth
                record["sweep_depth_pts"] = getattr(pattern, "sweep_depth_pts", None)

                # Confirmation type (ACCEPTANCE / NON_ACCEPTANCE)
                try:
                    record["confirmation_type"] = pattern.confirmation.name
                except Exception:
                    record["confirmation_type"] = None

                # FB entry path classification
                fb_detector = getattr(self.signal_aggregator, "failed_breakdown", None)
                if fb_detector is not None:
                    is_dd = getattr(fb_detector, "_is_double_dip", False)
                    is_ls = getattr(fb_detector, "_is_level_sweep", False)
                else:
                    is_dd = False
                    is_ls = False
                if is_dd:
                    record["fb_entry_path"] = "double_dip"
                elif is_ls:
                    record["fb_entry_path"] = "level_sweep"
                else:
                    record["fb_entry_path"] = "elevator_fb"

                # BD Short conviction score
                try:
                    bd_detector = self.signal_aggregator.breakdown_short
                    record["conviction_score"] = getattr(bd_detector, "_conviction_score", None)
                except Exception:
                    record["conviction_score"] = None

                # Market context at entry
                try:
                    vel_series = compute_velocity(self._df, window=5)
                    record["velocity"] = float(vel_series.iat[-1]) if not np.isnan(vel_series.iat[-1]) else None
                except Exception:
                    record["velocity"] = None

                try:
                    record["volume_at_entry"] = float(self._df["volume"].iat[-1]) if self._df is not None and len(self._df) > 0 else None
                except Exception:
                    record["volume_at_entry"] = None

                try:
                    record["intraday_state"] = self.signal_aggregator.intraday_state.name
                except Exception:
                    record["intraday_state"] = None

                try:
                    record["swing_structure"] = self.signal_aggregator.get_swing_snapshot()
                except Exception:
                    record["swing_structure"] = None

                # Session position pct: (close - session_low) / (session_high - session_low)
                session_range = session_high - session_low
                if session_range > 0:
                    record["session_position_pct"] = round((last_price - session_low) / session_range, 4)
                else:
                    record["session_position_pct"] = None

                # Level quality
                try:
                    record["level_touch_count"] = getattr(pattern.level, "touch_count", None)
                except Exception:
                    record["level_touch_count"] = None

                try:
                    record["level_rally_from_low_pts"] = getattr(pattern.level, "rally_from_low_pts", None)
                except Exception:
                    record["level_rally_from_low_pts"] = None

                try:
                    created_at = getattr(pattern.level, "created_at", None)
                    if created_at is not None and self._df is not None and len(self._df) > 0:
                        delta = self._df.index[-1] - created_at
                        # Approximate bars: assume 5-min bars
                        record["level_age_bars"] = int(delta.total_seconds() / 300) if hasattr(delta, "total_seconds") else None
                    else:
                        record["level_age_bars"] = None
                except Exception:
                    record["level_age_bars"] = None

                # Sizing
                record["position_size_factor"] = getattr(signal, "position_size_factor", None)
                record["risk_pts"] = getattr(signal, "risk_pts", None)
                record["reward_t1_pts"] = getattr(signal, "reward_t1_pts", None)

            # ── BAR WINDOW & VOLUME TREND (entries only) ──
            if context == "entry" and self._df is not None and len(self._df) > 0:
                window_size = min(20, len(self._df))
                window_df = self._df.iloc[-window_size:]
                bar_window = []
                for idx, row in window_df.iterrows():
                    bar_window.append({
                        "time": str(idx),
                        "open": round(float(row["open"]), 2),
                        "high": round(float(row["high"]), 2),
                        "low": round(float(row["low"]), 2),
                        "close": round(float(row["close"]), 2),
                        "volume": int(row["volume"]),
                    })
                record["bar_window"] = bar_window

                if len(self._df) >= 20:
                    vol = self._df["volume"].values
                    vol_5 = float(vol[-5:].mean())
                    vol_20 = float(vol[-20:].mean())
                    record["volume_5bar_avg"] = round(vol_5, 1)
                    record["volume_20bar_avg"] = round(vol_20, 1)
                    record["volume_trend"] = round(vol_5 / vol_20, 2) if vol_20 > 0 else 1.0

            # ── MARKET CONTEXT (entries only) ──
            if context == "entry" and self._df is not None and len(self._df) >= 20:
                try:
                    closes = self._df["close"].values
                    highs = self._df["high"].values
                    lows = self._df["low"].values
                    # Average True Range (simplified: avg bar range over 20 bars)
                    ranges = highs[-20:] - lows[-20:]
                    atr_20 = float(ranges.mean())
                    record["atr_20"] = round(atr_20, 2)
                    # Multi-bar trend: is price above or below 20-bar average?
                    sma_20 = float(closes[-20:].mean())
                    record["sma_20"] = round(sma_20, 2)
                    record["price_vs_sma20"] = round(float(closes[-1]) - sma_20, 2)
                except Exception:
                    pass

            # ── ENRICHED EXIT FIELDS ──
            if context == "exit":
                # Bars held
                record["bars_held"] = self._bar_count - self._entry_bar_count

                # MFE/MAE from TradePosition
                pos = self._position  # may already be None if cleared; use trade_record fallback
                direction = getattr(trade_record, "direction", "long")
                entry_price = getattr(trade_record, "entry_price", 0.0)

                # Try to get excursion data from the position or trade_record
                highest = getattr(pos, "highest_price_since_entry", None) or getattr(trade_record, "highest_price_since_entry", None)
                lowest = getattr(pos, "lowest_price_since_entry", None) or getattr(trade_record, "lowest_price_since_entry", None)

                if direction == "long":
                    record["mfe_pts"] = round(highest - entry_price, 2) if highest is not None else None
                    record["mae_pts"] = round(entry_price - lowest, 2) if lowest is not None else None
                else:
                    record["mfe_pts"] = round(entry_price - lowest, 2) if lowest is not None else None
                    record["mae_pts"] = round(highest - entry_price, 2) if highest is not None else None

                # T1/T2 hit flags
                record["t1_hit"] = getattr(pos, "t1_hit", None) or getattr(trade_record, "t1_hit", None)
                record["t2_hit"] = getattr(pos, "t2_hit", None) or getattr(trade_record, "t2_hit", None)

                # Exit phase
                try:
                    phase = getattr(pos, "phase", None) or getattr(trade_record, "phase", None)
                    record["exit_phase"] = phase.name if phase is not None else None
                except Exception:
                    record["exit_phase"] = None

                # Slippage: actual fill vs signal entry price
                if signal is not None:
                    record["slippage_pts"] = round(entry_price - signal.entry_price, 2)
                else:
                    record["slippage_pts"] = None

            # External market correlation data (VIX, SPY, 10Y yield)
            try:
                market_snapshot = fetch_market_snapshot()
                if market_snapshot:
                    record["market_correlation"] = market_snapshot
            except Exception:
                pass

            # Gate bypass info (collection mode)
            if self._current_gate_bypass:
                record["gate_bypassed"] = self._current_gate_bypass
                record["production_would_take"] = False
            else:
                record["production_would_take"] = True

            # Update last trade bar for time_since_last_trade tracking
            self._last_trade_bar = self._bar_count

            # Append to file (one JSON per line — append-only, crash-safe)
            with open(self._trade_log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

            self._all_trades.append(record)
            logger.info(f"Trade logged ({context}): {json.dumps(record, default=str)[:200]}...")

        except Exception as e:
            logger.warning(f"Failed to log trade: {e}")

    def _build_trade_reason(self, signal: Signal, regime_info: dict, session_window: dict) -> str:
        """Build a plain English explanation of why this trade was taken."""
        p = signal.pattern
        level_type = p.level.level_type.name.replace("_", " ").title()
        level_price = p.level.price
        direction = getattr(p, "direction", "long")
        sig_type = signal.signal_type.name

        if sig_type == "FAILED_BREAKDOWN":
            sweep_depth = getattr(p, "sweep_depth_pts", 0)
            conf = getattr(p, "confirmation", None)
            conf_str = conf.name.replace("_", "-").lower() if conf else "confirmed"
            reason = (
                f"FB at {level_type} {level_price:.2f}: "
                f"price swept {sweep_depth:.1f} pts below, recovered and {conf_str}. "
            )
        elif sig_type == "LEVEL_RECLAIM":
            reason = (
                f"Level Reclaim at {level_type} {level_price:.2f}: "
                f"price reclaimed level from below and held. "
            )
        elif sig_type == "BREAKDOWN_SHORT":
            reason = (
                f"BD Short at {level_type} {level_price:.2f}: "
                f"support broke and held below. "
            )
        elif sig_type == "BACKTEST_SHORT":
            reason = (
                f"Backtest Short at {level_type} {level_price:.2f}: "
                f"broken resistance retest failed. "
            )
        else:
            reason = f"{sig_type} at {level_type} {level_price:.2f}. "

        # Add risk/reward
        reason += f"Risk: {signal.risk_pts:.1f} pts, T1: {signal.target_1:.2f} (R:R {signal.rr_ratio_t1:.1f}). "

        # Add regime
        regime_dir = regime_info.get("direction", "?")
        reason += f"Regime: {regime_dir}. "

        # Add session window
        window_name = session_window.get("detail", "") if isinstance(session_window, dict) else str(session_window)
        if window_name:
            reason += f"Window: {window_name}."

        return reason

    def _log_phantom(self, signal: Signal, reject_reason: str, result: str = "") -> None:
        """Log a phantom (rejected) signal to the persistent log."""
        try:
            last_price = 0.0
            volume = None
            if self._df is not None and len(self._df) > 0:
                last_price = float(self._df["close"].iat[-1])
                try:
                    volume = float(self._df["volume"].iat[-1])
                except Exception:
                    pass

            now_et = datetime.now(_ET)
            record = {
                "event": "phantom",
                "timestamp": now_et.isoformat(),
                "session_date": str(self._session_date),
                "symbol": self.bridge.config.symbol,
                "signal_type": signal.signal_type.name,
                "direction": signal.direction,
                "entry_price": signal.entry_price,
                "stop_price": signal.stop_price,
                "target_1": signal.target_1,
                "target_2": getattr(signal, "target_2", None),
                "rr_ratio": round(signal.rr_ratio_t1, 2),
                "level_price": signal.pattern.level.price if signal.pattern.level else None,
                "level_type": signal.pattern.level.level_type.name if signal.pattern.level else None,
                "sweep_depth_pts": getattr(signal.pattern, "sweep_depth_pts", None),
                "confirmation_type": signal.pattern.confirmation.name if hasattr(signal.pattern, "confirmation") and signal.pattern.confirmation else None,
                "reject_reason": reject_reason,
                "result": result,
                "last_price": last_price,
                "session_window": self._get_session_window(now_et.time()).get("detail", ""),
                "bar_count": self._bar_count,
                "volume_at_signal": volume,
                "risk_pts": getattr(signal, "risk_pts", None),
                "reward_t1_pts": getattr(signal, "reward_t1_pts", None),
                "position_size_factor": getattr(signal, "position_size_factor", None),
            }
            with open(self._trade_log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    def _log_phantom_outcome(self, p: dict) -> None:
        """Log a resolved phantom trade outcome to the persistent log."""
        try:
            record = {
                "event": "phantom_resolved",
                "timestamp": datetime.now(_ET).isoformat(),
                "session_date": str(self._session_date),
                "signal_type": p["signal_type"],
                "direction": p.get("direction", "long"),
                "entry_price": p["entry_price"],
                "stop_price": p["stop_price"],
                "target_1": p["target_1"],
                "target_2": p.get("target_2"),
                "rr_ratio": p.get("rr_ratio"),
                "level_price": p.get("level_price"),
                "level_type": p.get("level_type"),
                "sweep_depth_pts": p.get("sweep_depth_pts"),
                "confirmation_type": p.get("confirmation_type"),
                "reject_reason": p["reject_reason"],
                "result": p["result"],
                "high_since": p["high_since"],
                "low_since": p["low_since"],
            }
            # Promote feature-richness fields captured at signal-time so
            # phantoms train alongside live entries with a unified schema.
            ctx = p.get("context") or {}
            for k in ("bar_count", "last_price", "session_high", "session_low",
                      "session_range", "session_window", "regime", "nearby_levels"):
                if k in ctx:
                    record[k] = ctx[k]
            with open(self._trade_log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    # ── Phantom trade tracking ──────────────────────────────────────

    def _add_phantom(self, signal: Signal, reject_reason: str, timestamp: datetime) -> None:
        """Record a rejected signal as a phantom trade to track what would have happened."""
        # Snapshot the bot's full context at signal-time so the phantom
        # carries the same feature set as live entries (regime, nearby
        # levels, session_high/low, session_window, etc.). Downstream ML
        # treats phantoms and live entries as a unified labeled set.
        try:
            context_snapshot = self._collect_signal_context()
        except Exception as e:
            logger.debug(f"Phantom context snapshot failed (non-fatal): {e}")
            context_snapshot = {}

        phantom = {
            "signal_type": signal.signal_type.name,
            "direction": signal.direction,
            "entry_price": signal.entry_price,
            "stop_price": signal.stop_price,
            "target_1": signal.target_1,
            "target_2": getattr(signal, "target_2", None),
            "rr_ratio": round(signal.rr_ratio_t1, 2),
            "level_price": signal.pattern.level.price if signal.pattern.level else None,
            "level_type": signal.pattern.level.level_type.name if signal.pattern.level else None,
            "sweep_depth_pts": getattr(signal.pattern, "sweep_depth_pts", None),
            "confirmation_type": signal.pattern.confirmation.name if hasattr(signal.pattern, "confirmation") and signal.pattern.confirmation else None,
            "reject_reason": reject_reason,
            "entry_time": timestamp,
            "high_since": signal.entry_price,
            "low_since": signal.entry_price,
            "resolved": False,
            "result": "",
            "context": context_snapshot,
        }
        self._phantom_positions.append(phantom)
        logger.warning(
            f"  >> Tracking phantom: entry={signal.entry_price:.2f} "
            f"SL={signal.stop_price:.2f} T1={signal.target_1:.2f} — will report outcome"
        )

    def _update_phantoms(self, high: float, low: float, close: float, timestamp: datetime) -> None:
        """Update all active phantom trades with current bar data."""
        for p in self._phantom_positions:
            if p["resolved"]:
                continue

            # Track extremes
            if high > p["high_since"]:
                p["high_since"] = high
            if low < p["low_since"]:
                p["low_since"] = low

            is_short = p.get("direction", "long") == "short"

            # Check stop hit (short: high >= stop, long: low <= stop)
            stop_hit = (high >= p["stop_price"]) if is_short else (low <= p["stop_price"])
            if stop_hit:
                if is_short:
                    pnl = p["entry_price"] - p["stop_price"]
                else:
                    pnl = p["stop_price"] - p["entry_price"]
                p["resolved"] = True
                p["result"] = f"STOP HIT ({pnl:+.2f} pts)"
                logger.warning(
                    f"PHANTOM RESULT: {p['signal_type']} @ {p['entry_price']:.2f} "
                    f"-> STOP HIT @ {p['stop_price']:.2f} = {pnl:+.2f} pts "
                    f"[rejected: {p['reject_reason']}] "
                    f"(high reached {p['high_since']:.2f})"
                )
                # Log phantom outcome for future analysis
                self._log_phantom_outcome(p)
                continue

            # Check target hit (short: low <= target, long: high >= target)
            target_hit = (low <= p["target_1"]) if is_short else (high >= p["target_1"])
            if target_hit:
                if is_short:
                    pnl = p["entry_price"] - p["target_1"]
                else:
                    pnl = p["target_1"] - p["entry_price"]
                p["resolved"] = True
                p["result"] = f"T1 HIT ({pnl:+.2f} pts)"
                logger.warning(
                    f"PHANTOM RESULT: {p['signal_type']} @ {p['entry_price']:.2f} "
                    f"-> T1 HIT @ {p['target_1']:.2f} = {pnl:+.2f} pts "
                    f"[rejected: {p['reject_reason']}] "
                    f"(low reached {p['low_since']:.2f})"
                )
                self._log_phantom_outcome(p)
                continue

    def _enrich_near_misses(self) -> list[dict]:
        """Merge near-miss detections with their phantom outcome tracking."""
        raw = getattr(self.signal_aggregator.failed_breakdown, "near_misses", [])[-10:]
        enriched = []
        for nm in raw:
            item = dict(nm)
            # Build stable key to match phantom
            level_price = nm.get("level_price", 0)
            nm_key = f"{nm.get('timestamp')}_{level_price:.2f}_{nm.get('failure_reason','')}"
            # Find matching phantom
            for p in self._near_miss_phantoms:
                if p.get("near_miss_key") == nm_key:
                    item["outcome"] = {
                        "entry_price": p["entry_price"],
                        "stop_price": p["stop_price"],
                        "target_price": p["target_price"],
                        "resolved": p["resolved"],
                        "result": p["result"],
                        "high_since": p["high_since"],
                        "low_since": p["low_since"],
                    }
                    break
            enriched.append(item)
        return enriched

    def _log_near_miss_outcome(self, p: dict) -> None:
        """Log a resolved near-miss phantom outcome to the persistent log."""
        try:
            record = {
                "event": "near_miss_resolved",
                "timestamp": datetime.now(_ET).isoformat(),
                "session_date": str(self._session_date),
                "direction": p.get("direction", "long"),
                "entry_price": p["entry_price"],
                "stop_price": p["stop_price"],
                "target_price": p["target_price"],
                "level_price": p.get("level_price"),
                "risk_pts": p.get("risk_pts"),
                "failure_reason": p.get("failure_reason", ""),
                "result": p["result"],
                "high_since": p["high_since"],
                "low_since": p["low_since"],
            }
            with open(self._trade_log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    def _update_near_miss_phantoms(self, high: float, low: float, close: float, timestamp: datetime) -> None:
        """Update near-miss phantom trades — track what would have happened."""
        for p in self._near_miss_phantoms:
            if p["resolved"]:
                continue

            if high > p["high_since"]:
                p["high_since"] = high
            if low < p["low_since"]:
                p["low_since"] = low

            is_short = p.get("direction", "long") == "short"

            # Check stop hit (short: high >= stop, long: low <= stop)
            stop_hit = (high >= p["stop_price"]) if is_short else (low <= p["stop_price"])
            if stop_hit:
                if is_short:
                    pnl = p["entry_price"] - p["stop_price"]
                else:
                    pnl = p["stop_price"] - p["entry_price"]
                p["resolved"] = True
                p["result"] = f"STOP HIT ({pnl:+.2f} pts)"
                logger.info(
                    f"NEAR-MISS OUTCOME: FB @ {p['level_price']:.2f} "
                    f"-> STOP HIT {pnl:+.2f} pts (entry {p['entry_price']:.2f}, "
                    f"stop {p['stop_price']:.2f})"
                )
                self._log_near_miss_outcome(p)
                continue

            # Check target hit (short: low <= target, long: high >= target)
            target_hit = (low <= p["target_price"]) if is_short else (high >= p["target_price"])
            if target_hit:
                if is_short:
                    pnl = p["entry_price"] - p["target_price"]
                else:
                    pnl = p["target_price"] - p["entry_price"]
                p["resolved"] = True
                p["result"] = f"T1 HIT ({pnl:+.2f} pts)"
                logger.info(
                    f"NEAR-MISS OUTCOME: FB @ {p['level_price']:.2f} "
                    f"-> T1 HIT {pnl:+.2f} pts (entry {p['entry_price']:.2f}, "
                    f"target {p['target_price']:.2f})"
                )
                self._log_near_miss_outcome(p)
                continue

    # ── Session archival ───────────────────────────────────────────

    def _archive_session(self) -> None:
        """Archive session bars and level snapshot for retrospective analysis."""
        try:
            sessions_dir = Path("/app/data/sessions")
            sessions_dir.mkdir(parents=True, exist_ok=True)

            # Archive full session bars
            if self._df is not None and len(self._df) > 0:
                bars_path = sessions_dir / f"{self._session_date}_bars.parquet"
                self._df.to_parquet(bars_path)
                logger.info(f"Session bars archived: {len(self._df)} bars -> {bars_path}")

            # Snapshot active levels
            levels_snapshot = []
            if hasattr(self.signal_aggregator, "level_store"):
                for lv in self.signal_aggregator.level_store.get_active():
                    levels_snapshot.append({
                        "price": lv.price,
                        "type": lv.level_type.name,
                        "touches": lv.touch_count,
                    })
            levels_path = sessions_dir / f"{self._session_date}_levels.json"
            with open(levels_path, "w") as f:
                json.dump(levels_snapshot, f, indent=2)
            logger.info(f"Level snapshot archived: {len(levels_snapshot)} levels -> {levels_path}")

        except Exception as e:
            logger.warning(f"Failed to archive session: {e}")

    def _check_session_rollover(self) -> None:
        """Detect new Globex session and reset daily state.

        CME Globex sessions run 18:00 ET -> 17:00 ET next day.
        The "trading date" is the NEXT calendar day after 18:00.
        E.g., Sunday 18:00 ET = Monday's session.

        Without this, daily loss limits and trade counts from the
        previous session carry over and block new signals.
        """
        now = datetime.now(_ET)
        t = now.time()

        # Compute the trading date: after 18:00 ET, it's "tomorrow's" session
        if t >= time(18, 0):
            trading_date = now.date() + __import__('datetime').timedelta(days=1)
        elif t < time(17, 0):
            trading_date = now.date()
        else:
            # 17:00-18:00 = break, keep current date
            return

        if trading_date != self._session_date:
            logger.info(f"SESSION ROLLOVER: {self._session_date} -> {trading_date}")
            # Save pattern state before reset (carries across sessions)
            pattern_snapshot = self.signal_aggregator.get_pattern_state()
            # Archive old session
            self._archive_session()
            self._log_session_summary()

            # Reset for new session
            old_date = self._session_date
            self._session_date = trading_date
            # Preserve open position reference before reset
            open_pos = self._position if (self._position is not None and self._position.is_open) else None
            self.strategy.reset()
            self.position_manager.start_session(now)
            # Transfer open position to new session so it isn't lost
            if open_pos is not None:
                self.position_manager.session.active_position = open_pos
                if open_pos.direction == "long":
                    self.position_manager.session.active_long = open_pos
                else:
                    self.position_manager.session.active_short = open_pos
            self._phantom_positions.clear()
            self._near_miss_phantoms.clear()
            self._bar_count = 0
            self._df = pd.DataFrame()

            # Re-initialize levels from IB (prior day data)
            try:
                prior_day_df = self.bridge.get_prior_day_bars()
                if prior_day_df is not None:
                    self.signal_aggregator.initialize_levels(pd.DataFrame(), prior_day_df)
                    logger.info(f"Rollover: re-initialized levels from {len(prior_day_df)} prior day bars")
            except Exception as e:
                logger.warning(f"Rollover: failed to reload prior day bars: {e}")

            # Restore pattern state from prior session (sweeps carry across)
            self.signal_aggregator.restore_pattern_state(pattern_snapshot)
            logger.info(f"New session {trading_date}: daily PnL reset, "
                        f"trade count reset, levels re-initialized, pattern state restored")

    # ── EOD and session management ───────────────────────────────────

    def _check_eod(self, bar: dict) -> None:
        """Check for EOD: update runner trail or flatten non-runners.

        Mancini's method: runners carry overnight with stop under today's
        daily low. Non-runners (INITIAL phase) get flattened at EOD.
        At session break (17:00-18:00), archive and stop.
        """
        ts_str = bar.get("timestamp", "")
        try:
            timestamp = pd.Timestamp(ts_str)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("US/Eastern")

            t = timestamp.time()
            session = self.strategy.session_times

            if session.past_eod_flatten(t):
                if self._position is not None and self._position.is_open:
                    # Runners (AFTER_T1, AFTER_T2) carry overnight — update trail
                    if self._position.phase in (ExitPhase.AFTER_T1, ExitPhase.AFTER_T2):
                        daily_low = self._get_session_low()
                        daily_high = self._get_session_high()
                        if daily_low > 0:
                            self._position.prior_day_low = daily_low
                            self._position.prior_day_high = daily_high
                            action = self.exit_manager.update_prior_day_low(
                                self._position, daily_low
                            )
                            if action and action.new_stop > 0:
                                self.bridge.update_stop(
                                    trade_id=self._trade_id,
                                    new_sl=action.new_stop,
                                    reason=action.reason,
                                )
                            logger.info(
                                f"RUNNER EOD: trailing stop → {self._position.stop_price:.2f} "
                                f"(daily low={daily_low:.2f}, {self._position.remaining_contracts} contracts)"
                            )
                    else:
                        # Non-runner positions (still in INITIAL): flatten
                        self.bridge.flatten(reason="eod_flatten")
                        flatten_time = session.eod_flatten_time.strftime("%H:%M")
                        logger.info(f"EOD flatten sent ({flatten_time} ET)")
                        exit_price = float(bar.get("close", 0))
                        # Compute PnL before closing
                        if self._position.direction == "short":
                            self._position.realized_pnl_pts += (self._position.entry_price - exit_price) * self._position.remaining_contracts
                        else:
                            self._position.realized_pnl_pts += (exit_price - self._position.entry_price) * self._position.remaining_contracts
                        self._position.remaining_contracts = 0
                        self._position.phase = ExitPhase.CLOSED
                        trade_rec = self.position_manager.close_position(
                            exit_price=exit_price,
                            timestamp=timestamp,
                            exit_reason="EOD flatten",
                            pattern_type=self._pattern_type,
                            entry_time=self._entry_timestamp,
                        )
                        if trade_rec:
                            self._log_trade(trade_rec, self._current_signal, "exit")
                        self._position = None
                        self._trade_id = None

            # Check if session break (17:00-18:00) — archive and stop
            if time(17, 0) <= t < time(18, 0):
                if not session.in_rth(t):
                    logger.info("Session break (17:00-18:00 ET)")
                    self._archive_session()
                    self._running = False
        except (ValueError, TypeError):
            pass

    def _get_session_low(self) -> float:
        """Get today's session low from bar data."""
        if self._df is not None and len(self._df) > 0:
            return float(self._df["low"].min())
        return 0.0

    def _get_session_high(self) -> float:
        """Get today's session high from bar data."""
        if self._df is not None and len(self._df) > 0:
            return float(self._df["high"].max())
        return 0.0

    # ── Shutdown ─────────────────────────────────────────────────────

    def _save_pattern_state(self) -> None:
        """Save pattern state to disk for restoration after restart."""
        try:
            state = self.signal_aggregator.get_pattern_state()
            state_path = Path(os.environ.get("PATTERN_STATE_FILE", "/app/data/pattern_state.json"))
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, default=str))
            logger.info(f"Pattern state saved to {state_path}")
        except Exception as e:
            logger.warning(f"Failed to save pattern state: {e}")

    def _load_pattern_state(self) -> None:
        """Load and restore pattern state from disk (one-shot)."""
        state_path = Path(os.environ.get("PATTERN_STATE_FILE", "/app/data/pattern_state.json"))
        if not state_path.exists():
            return
        try:
            state = json.loads(state_path.read_text())
            self.signal_aggregator.restore_pattern_state(state)
            saved_at = state.get("timestamp", "unknown")
            logger.info(f"Restored pattern state from {state_path} (saved at {saved_at})")
            state_path.unlink()  # One-shot: clear after loading
        except Exception as e:
            logger.warning(f"Failed to load pattern state: {e}")

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle SIGINT/SIGTERM.

        NEVER flatten positions on shutdown — they must survive restarts.
        IB bracket orders (SL/TP) remain active on the exchange regardless.
        """
        logger.info("Shutdown signal received")
        if self._position is not None and self._position.is_open:
            logger.info(f"Position stays open on IB: {self._pattern_type} "
                        f"@ {self._position.entry_price:.2f}, "
                        f"SL={self._position.stop_price:.2f}, "
                        f"TP={self._position.target_1:.2f}")
        self._save_pattern_state()
        self._archive_session()
        self._running = False

    def _log_session_summary(self) -> None:
        """Log end-of-session statistics and write summary to JSONL."""
        if self.position_manager.session is None:
            return
        s = self.position_manager.session
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info(f"  Date:    {self._session_date}")
        logger.info(f"  Bars:    {self._bar_count}")
        logger.info(f"  Trades:  {s.trade_count}")
        logger.info(f"  Wins:    {s.wins}")
        logger.info(f"  Losses:  {s.losses}")
        logger.info(f"  PnL:     {s.daily_pnl_pts:+.1f} pts (${s.daily_pnl_dollars:+,.0f})")
        logger.info(f"  State:   {s.state.name}")

        # Phantom trade summary
        phantom_count = len(self._phantom_positions)
        if self._phantom_positions:
            logger.info("-" * 60)
            logger.info("PHANTOM TRADES (signals rejected by filters):")
            for i, p in enumerate(self._phantom_positions, 1):
                status = p["result"] if p["resolved"] else f"OPEN (last high={p['high_since']:.2f}, low={p['low_since']:.2f})"
                logger.info(
                    f"  #{i} {p['signal_type']} @ {p['entry_price']:.2f} "
                    f"SL={p['stop_price']:.2f} T1={p['target_1']:.2f} "
                    f"-> {status} [{p['reject_reason']}]"
                )
            # Tally
            resolved = [p for p in self._phantom_positions if p["resolved"]]
            wins = sum(1 for p in resolved if "T1 HIT" in p["result"])
            losses = sum(1 for p in resolved if "STOP HIT" in p["result"])
            logger.info(f"  Phantom tally: {wins}W / {losses}L / {len(self._phantom_positions)-len(resolved)} unresolved")

        logger.info("=" * 60)

        # Write session summary JSONL record
        try:
            signals_generated = len(getattr(self.signal_aggregator, "signals", []))
            summary_record = {
                "event": "session_summary",
                "timestamp": datetime.now(_ET).isoformat(),
                "session_date": str(self._session_date),
                "symbol": self.bridge.config.symbol,
                "bar_count": self._bar_count,
                "trades": s.trade_count,
                "wins": s.wins,
                "losses": s.losses,
                "pnl_pts": round(s.daily_pnl_pts, 2),
                "phantom_count": phantom_count,
                "signals_generated": signals_generated,
            }
            with open(self._trade_log_path, "a") as f:
                f.write(json.dumps(summary_record, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log session summary: {e}")


    @staticmethod
    def _et_to_pt(dt_obj: datetime) -> datetime:
        """Convert an ET datetime to PT (subtract 3 hours)."""
        from datetime import timedelta
        return dt_obj - timedelta(hours=3)

    def _get_session_window(self, et_time: time) -> dict:
        """Determine current market session window from ET time."""
        t = et_time
        if time(17, 0) <= t < time(18, 0):
            window = {"label": "CLOSED", "detail": "Daily Break", "trading": False, "css": "session-closed"}
        elif time(18, 0) <= t < time(22, 0):
            window = {"label": "GLOBEX", "detail": "Evening (Blocked 6-10PM ET)", "trading": False, "css": "session-blocked"}
        elif time(22, 0) <= t <= time(23, 59) or time(0, 0) <= t < time(2, 0):
            window = {"label": "GLOBEX", "detail": "Late Night Session", "trading": True, "css": "session-globex"}
        elif time(2, 0) <= t < time(6, 0):
            window = {"label": "EURO", "detail": "European Open (Blocked)", "trading": False, "css": "session-blocked"}
        elif time(6, 0) <= t < time(9, 30):
            window = {"label": "PRE-RTH", "detail": "Pre-Market", "trading": True, "css": "session-globex"}
        elif time(9, 30) <= t < time(11, 0):
            window = {"label": "RTH", "detail": "Morning Window (Prime)", "trading": True, "css": "session-rth"}
        elif time(11, 0) <= t < time(13, 0):
            window = {"label": "RTH", "detail": "Midday", "trading": True, "css": "session-rth"}
        elif time(13, 0) <= t < time(15, 0):
            window = {"label": "CHOP", "detail": "Chop Zone (Blocked)", "trading": False, "css": "session-blocked"}
        elif time(15, 0) <= t < time(16, 50):
            window = {"label": "RTH", "detail": "Afternoon (FB Only)", "trading": True, "css": "session-rth"}
        elif time(16, 50) <= t < time(17, 0):
            window = {"label": "RTH", "detail": "EOD Flatten Zone", "trading": False, "css": "session-blocked"}
        else:
            window = {"label": "UNKNOWN", "detail": "", "trading": False, "css": "session-closed"}

        # Bypass mode: override time gates so dashboard shows trading active
        if self._bypass_session_gates and not window["trading"]:
            window["trading"] = True
            window["detail"] += " [BYPASS]"
            window["css"] = "session-bypass"

        return window

    def _flush_shadow_events(self) -> None:
        """Write any pending shadow mode events to the shadow log file.

        Also creates phantom trackers for shadow signals that have entry/stop/target
        so we can track what would have happened.
        """
        events = self.signal_aggregator.shadow_events
        if not events:
            return
        try:
            with open(self._shadow_log_path, "a") as f:
                for event in events:
                    f.write(json.dumps(event, default=str) + "\n")
                    # Create phantom tracker for signals with entry/stop/target
                    if event.get("entry_price") and event.get("stop_price"):
                        direction = "short" if "short" in event.get("feature", "").lower() else "long"
                        self._shadow_phantoms.append({
                            "feature": event.get("feature"),
                            "entry_price": event["entry_price"],
                            "stop_price": event["stop_price"],
                            "target_price": event.get("target_1", 0),
                            "direction": direction,
                            "bar_start": self._bar_count,
                            "bars_tracked": 0,
                            "max_bars": 60,
                            "high_since": event["entry_price"],
                            "low_since": event["entry_price"],
                            "outcome": None,  # "target_hit", "stop_hit", "timeout"
                            "outcome_price": 0,
                            "timestamp": event.get("timestamp", ""),
                        })
            logger.debug(f"Flushed {len(events)} shadow event(s) to {self._shadow_log_path}")
        except Exception as e:
            logger.error(f"Failed to write shadow log: {e}")
        events.clear()

    def _update_shadow_phantoms(self, high: float, low: float) -> None:
        """Track shadow signal outcomes — did they hit target or stop?"""
        completed = []
        for p in self._shadow_phantoms:
            if p["outcome"]:
                continue
            p["high_since"] = max(p["high_since"], high)
            p["low_since"] = min(p["low_since"], low)
            p["bars_tracked"] += 1

            # Check target/stop hit
            if p["direction"] == "long":
                if p["target_price"] > 0 and high >= p["target_price"]:
                    p["outcome"] = "target_hit"
                    p["outcome_price"] = p["target_price"]
                elif low <= p["stop_price"]:
                    p["outcome"] = "stop_hit"
                    p["outcome_price"] = p["stop_price"]
            else:  # short
                if p["target_price"] > 0 and low <= p["target_price"]:
                    p["outcome"] = "target_hit"
                    p["outcome_price"] = p["target_price"]
                elif high >= p["stop_price"]:
                    p["outcome"] = "stop_hit"
                    p["outcome_price"] = p["stop_price"]

            # Timeout
            if p["bars_tracked"] >= p["max_bars"] and not p["outcome"]:
                p["outcome"] = "timeout"
                if p["direction"] == "long":
                    p["outcome_price"] = p["low_since"]
                else:
                    p["outcome_price"] = p["high_since"]

            if p["outcome"]:
                # Compute PnL
                if p["direction"] == "short":
                    pnl = p["entry_price"] - p["outcome_price"]
                else:
                    pnl = p["outcome_price"] - p["entry_price"]
                p["pnl_pts"] = round(pnl, 2)
                # Write outcome to shadow log
                try:
                    record = {
                        "event": "shadow_outcome",
                        "feature": p["feature"],
                        "timestamp": p["timestamp"],
                        "direction": p["direction"],
                        "entry_price": p["entry_price"],
                        "stop_price": p["stop_price"],
                        "target_price": p["target_price"],
                        "outcome": p["outcome"],
                        "outcome_price": p["outcome_price"],
                        "pnl_pts": p["pnl_pts"],
                        "bars_tracked": p["bars_tracked"],
                        "mfe_pts": round(p["high_since"] - p["entry_price"] if p["direction"] == "long" else p["entry_price"] - p["low_since"], 2),
                    }
                    with open(self._shadow_log_path, "a") as f:
                        f.write(json.dumps(record, default=str) + "\n")
                except Exception:
                    pass
                completed.append(p)

        for p in completed:
            self._shadow_phantoms.remove(p)
        # Cap
        if len(self._shadow_phantoms) > 20:
            self._shadow_phantoms = self._shadow_phantoms[-20:]

    def _write_status(self) -> None:
        """Write current state to JSON file for the dashboard."""
        status_path = os.environ.get("STATUS_FILE", "/app/logs/status.json")
        try:
            Path(status_path).parent.mkdir(parents=True, exist_ok=True)

            # Current price and time info
            last_price = 0.0
            last_bar_et = ""
            last_bar_pst = ""
            bar_high = 0.0
            bar_low = 0.0
            session_high = 0.0
            session_low = 999999.0
            session_window = {"label": "—", "detail": "", "trading": False, "css": "session-closed"}

            if self._df is not None and len(self._df) > 0:
                last_price = float(self._df["close"].iat[-1])
                bar_high = float(self._df["high"].iat[-1])
                bar_low = float(self._df["low"].iat[-1])
                session_high = float(self._df["high"].max())
                session_low = float(self._df["low"].min())
                ts = self._df.index[-1]
                last_bar_et = ts.strftime("%I:%M:%S %p ET")
                pst_ts = self._et_to_pt(ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts)
                last_bar_pst = pst_ts.strftime("%I:%M:%S %p PT")
                session_window = self._get_session_window(ts.time())

            # Current position info
            pos_data = None
            if self._position is not None and self._position.is_open:
                unrealized = 0.0
                bars_held = 0
                if self._df is not None and len(self._df) > 0:
                    if self._current_signal and self._current_signal.direction == "short":
                        unrealized = self._position.entry_price - last_price
                    else:
                        unrealized = last_price - self._position.entry_price
                phase = getattr(self._position, "phase", None)
                phase_name = phase.name if phase else "INITIAL"
                highest = getattr(self._position, "highest_price_since_entry", 0)
                lowest = getattr(self._position, "lowest_price_since_entry", 0)
                t1_hit = getattr(self._position, "t1_hit", False)
                direction = getattr(self._position, "direction", None) or (getattr(self._current_signal, "direction", "long") if self._current_signal else "long")
                if direction == "long":
                    mfe = highest - self._position.entry_price if highest else 0
                    mae = self._position.entry_price - lowest if lowest else 0
                else:
                    mfe = self._position.entry_price - lowest if lowest else 0
                    mae = highest - self._position.entry_price if highest else 0
                pos_data = {
                    "is_open": True,
                    "direction": direction,
                    "pattern": self._pattern_type,
                    "entry_price": self._position.entry_price,
                    "stop_price": self._position.stop_price,
                    "target_price": self._position.target_1,
                    "unrealized_pnl": unrealized,
                    "contracts": self._position.remaining_contracts,
                    "risk_pts": abs(self._position.entry_price - self._position.stop_price),
                    "reward_pts": abs(self._position.target_1 - self._position.entry_price),
                    "phase": phase_name,
                    "t1_hit": t1_hit,
                    "mfe_pts": round(mfe, 2),
                    "mae_pts": round(mae, 2),
                    "trail_stop": self._position.stop_price if phase_name in ("AFTER_T1", "AFTER_T2", "RUNNER") else None,
                    "trail_distance_pts": round(abs(last_price - self._position.stop_price), 2) if phase_name in ("AFTER_T1", "AFTER_T2", "RUNNER") else None,
                }

            # Active levels — sorted by proximity to current price
            levels = []
            if hasattr(self.signal_aggregator, "level_store"):
                for lv in self.signal_aggregator.level_store.get_active():
                    dist = lv.price - last_price if last_price > 0 else 0
                    levels.append({
                        "price": lv.price,
                        "type": lv.level_type.name,
                        "touches": lv.touch_count,
                        "distance": dist,
                    })
                levels.sort(key=lambda x: abs(x["distance"]))

            # Trade history from position manager
            trades = []
            if self.position_manager.session is not None:
                for t in self.position_manager.session.trades:
                    trade_dict = {
                        "time": self._et_to_pt(t.entry_time).strftime("%I:%M %p PT") if hasattr(t.entry_time, "strftime") else str(t.entry_time),
                        "direction": getattr(t, "direction", "long"),
                        "pattern": t.pattern_type,
                        "entry_price": t.entry_price,
                        "exit_price": t.avg_exit_price,
                        "pnl_pts": t.pnl_pts,
                        "exit_reason": t.exit_reason,
                    }
                    # Check persistent log for gate bypass info
                    for logged in reversed(self._all_trades):
                        if (logged.get("event") == "entry"
                                and abs(logged.get("entry_price", 0) - t.entry_price) < 0.01):
                            if logged.get("gate_bypassed"):
                                trade_dict["gate_bypassed"] = logged["gate_bypassed"]
                                trade_dict["production_would_take"] = False
                            else:
                                trade_dict["production_would_take"] = True
                            break
                    trades.append(trade_dict)

            # Phantom trades (filtered signals)
            phantoms = []
            for p in self._phantom_positions:
                phantoms.append({
                    "signal_type": p["signal_type"],
                    "entry_price": p["entry_price"],
                    "stop_price": p["stop_price"],
                    "target_1": p["target_1"],
                    "reject_reason": p["reject_reason"],
                    "resolved": p["resolved"],
                    "result": p["result"],
                })

            # Account info (cached, don't call IB every bar)
            acct = {}
            if self._bar_count % 30 == 0:  # refresh every 30 bars (~30 min)
                acct = self.bridge.get_account_info() or {}
                self._cached_account = acct
            else:
                acct = getattr(self, "_cached_account", {})

            session = self.position_manager.session
            now_et = datetime.now()
            now_pst = self._et_to_pt(now_et)

            # Recent OHLCV bars for chart (last 100)
            bars_data = []
            if self._df is not None and len(self._df) > 0:
                chart_df = self._df.tail(100)
                for ts, row in chart_df.iterrows():
                    bars_data.append({
                        "time": int(ts.timestamp()) if hasattr(ts, "timestamp") else 0,
                        "open": round(float(row["open"]), 2),
                        "high": round(float(row["high"]), 2),
                        "low": round(float(row["low"]), 2),
                        "close": round(float(row["close"]), 2),
                    })

            status = {
                "connected": getattr(self.bridge, "is_connected", True),
                "symbol": self.bridge.config.symbol,
                "session_date": str(self._session_date),
                "bar_count": self._bar_count,
                "last_bar_et": last_bar_et,
                "last_bar_pst": last_bar_pst,
                "last_price": last_price,
                "bar_high": bar_high,
                "bar_low": bar_low,
                "session_high": session_high,
                "session_low": session_low,
                "session_window": session_window,
                "regime": self.strategy._regime_state.direction.name if self.strategy._regime_state else "—",
                "regime_longs": self.strategy._regime_state.longs_enabled if self.strategy._regime_state else True,
                "regime_shorts": self.strategy._regime_state.shorts_enabled if self.strategy._regime_state else False,
                "last_update_et": now_et.strftime("%Y-%m-%d %I:%M:%S %p ET"),
                "last_update_pst": now_pst.strftime("%Y-%m-%d %I:%M:%S %p PT"),
                "daily_pnl_pts": session.daily_pnl_pts if session else 0,
                "trades_today": session.trade_count if session else 0,
                "winners": session.wins if session else 0,
                "losers": session.losses if session else 0,
                "max_trades": self.strategy.risk_params.max_trades_per_day,
                "is_done_for_day": self.position_manager.is_done_for_day,
                "account_balance": f"${acct.get('balance', 0):,.0f}" if acct.get("balance") else "—",
                "account_equity": f"${acct.get('equity', 0):,.0f}" if acct.get("equity") else "—",
                "account_name": acct.get("name", "—"),
                "position": pos_data,
                "levels": levels,
                "trades": trades,
                "phantoms": phantoms,
                "bars": bars_data,
                "total_logged_trades": len(self._all_trades),
                "regime_daily_bars": len(dh) if (dh := getattr(self.strategy, "_daily_history", None)) is not None and hasattr(dh, "__len__") else 0,
                "daily_bias": self.signal_aggregator.daily_bias,
                "daily_structure": self.signal_aggregator.get_daily_structure_snapshot(),
                "near_misses": self._enrich_near_misses(),
                "bypass_mode": self._bypass_session_gates,
            }

            # Mancini Substack overlay summary (if enabled for this session)
            if self._mancini_overlay_result is not None:
                status["mancini"] = {
                    "mode": self._mancini_overlay_result.mode,
                    "lean": self._mancini_overlay_result.lean,
                    "parse_status": self._mancini_overlay_result.parse_status,
                    "confirmed_count": self._mancini_overlay_result.confirmed_count,
                    "injected_count": self._mancini_overlay_result.injected_count,
                    "shadow_count": self._mancini_overlay_result.shadow_count,
                    "blind_spots": self._mancini_overlay_result.blind_spots[:20],
                }

            # External market correlation data for dashboard
            try:
                market_snapshot = fetch_market_snapshot()
                if market_snapshot:
                    status["market_data"] = market_snapshot
            except Exception:
                pass

            # Atomic write (write to temp then rename)
            tmp_path = status_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(status, f, indent=2, default=str)
            os.replace(tmp_path, status_path)

        except Exception as e:
            logger.error(f"Failed to write status: {e}")


def main():
    """Run the IB bridge with production params."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Mancini IB Runner")
    parser.add_argument("--symbol", default="MES",
                        help="IB symbol (default: MES)")
    parser.add_argument("--contracts", type=int, default=1,
                        help="Number of contracts to trade (default: 1)")
    parser.add_argument("--host", default=os.environ.get("IB_HOST", "127.0.0.1"),
                        help="TWS/Gateway host (env: IB_HOST)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("IB_PORT", "7497")),
                        help="TWS/Gateway port (env: IB_PORT, 7497=TWS paper, 4002=Gateway paper)")
    parser.add_argument("--client-id", type=int, default=1,
                        help="IB client ID (must be unique per connection)")
    parser.add_argument("--contract-month", default="",
                        help="Contract month (YYYYMM), empty for front-month auto-detect")
    parser.add_argument("--full-session", action="store_true",
                        help="Trade full 23-hour session (globex + RTH), not just RTH")
    args = parser.parse_args()

    # Safety warning for live trading
    if args.port in (7496, 4001):
        logger.warning("=" * 60)
        logger.warning("  LIVE TRADING MODE")
        logger.warning("  Real money is at risk!")
        logger.warning("=" * 60)
        response = input("Type 'YES' to confirm live trading: ")
        if response.strip() != "YES":
            logger.info("Live trading aborted by user")
            return

    config = IBConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        symbol=args.symbol,
        contract_month=args.contract_month,
        use_rth_only=not args.full_session,
    )

    # Mancini-faithful exit: 75/25 split with 4 contracts.
    # 3 contracts exit at T1 (75%), 1 runner (25%) trails under prior day low.
    # Runners carry overnight/multi-day until prior day low is lost.
    exit_params = PRODUCTION_EXIT

    session_times = FULL_SESSION if args.full_session else PRODUCTION_SESSION

    # Full session: use RTH filter for level detection + FB-only PM filter
    rth_filter = None
    fb_only_pm = False
    if args.full_session:
        from datetime import time as dt_time
        rth_filter = (dt_time(9, 30), dt_time(16, 0))
        fb_only_pm = True
        logger.info("Full session active: rth_filter for levels, FB-only PM, evening block")

    # Live mode: bypass time gates (trade all hours) but enforce quality gates.
    # Optuna v2 optimized params (Mar 2026): OOS validated PF=1.14, Sharpe=1.00
    # Quality gates tightened based on 1,651 live events analysis.
    live_risk = RiskParams(
        max_trades_per_day=999,
        max_daily_loss_pts=9999.0,  # no daily loss limit
        max_stop_distance_pts=60.0,  # Data collection: allow deep sweep FBs (30-40pt stops = 61% WR, +264 pts on 5yr)
        skip_tuesdays=False,
        min_rr_ratio=0.8,  # Optuna v2: moderate filter (was 0.1 data collection)
    )

    runner = IBRunner(
        ib_config=config,
        exit_params=exit_params,
        risk_params=live_risk,
        session_times=session_times,
        min_rr_ratio=0.8,  # Optuna v2 optimized (was 0.1 data collection)
        rth_filter=rth_filter,
        fb_only_pm=fb_only_pm,
        regime_params=PRODUCTION_REGIME,
        bypass_session_gates=True,  # bypass time gates only, quality gates enforced
    )
    runner.run()


if __name__ == "__main__":
    main()
