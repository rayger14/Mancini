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
from types import SimpleNamespace
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


# A None read from get_position() is NOT proof the position closed. Right
# after a reconnect IB hasn't re-pushed its position/order caches yet, so
# positions() (and openTrades()) come back empty for a still-open position.
POST_RECONNECT_GRACE_SEC = 60.0


def phantom_close_guard(seconds_since_reconnect: float, bracket_live: bool,
                        grace: float = POST_RECONNECT_GRACE_SEC) -> str:
    """Decide whether a None position read can be trusted as a real closure.

    Returns:
    - "ignore_reconnect"   — within the post-reconnect grace; IB hasn't
      re-pushed its caches yet (trade #567 booked a fictional exit this way
      after a 00:29 reconnect).
    - "ignore_bracket_live" — the OCA bracket children are still working on IB.
      A real fill cancels the sibling, so a live bracket means the position is
      open and get_position() merely lagged (trade #579, 53s after entry).
    - "count" — neither guard applies; trust the None and count it toward the
      3x-consecutive closure confirmation.
    """
    if seconds_since_reconnect < grace:
        return "ignore_reconnect"
    if bracket_live:
        return "ignore_bracket_live"
    return "count"


def classify_position_sync(local_remaining: int, ib_volume, t1_booked: bool) -> str:
    """Classify what a venue-side position change means during _sync_position.

    On the delayed feed the exchange can fill the TP fraction before the bot's
    bars show T1, leaving a runner. This decides how to reconcile:
    - "full_close"       — IB shows flat/zero: the whole position closed.
    - "venue_t1_partial" — IB shows fewer contracts than we hold and T1 isn't
      booked yet: the venue filled the TP fraction; book the partial, keep the
      runner.
    - "no_change"        — same size, an expected post-T1 state, or a defensive
      unexpected increase.
    """
    if ib_volume is None or ib_volume == 0:
        return "full_close"
    if ib_volume < local_remaining and not t1_booked:
        return "venue_t1_partial"
    return "no_change"


def plan_full_close_legs(*, exit_type: str, t1_booked: bool,
                         remaining_contracts: int, total_contracts: int,
                         t1_fraction: float) -> "list[tuple[str, int]]":
    """Decide how a venue-side FULL close should be booked into P&L legs.

    Normally the whole remaining size books as one ``("full", n)`` leg. But when
    the venue reports the position flat via a TP fill while T1 was never booked
    — the exchange swept the TP fraction AND the runner before ``_sync_position``
    caught a clean runner-sized read (the 2026-07-01 trade 622 race) — booking
    all N at the TP price silently erases the scale-out and the runner ride.
    Split that case into the T1 fraction + runner so the collapsed runner is
    visible in the record. Attribution only: the leg quantities always sum to
    ``remaining_contracts`` (no P&L created or destroyed).
    """
    from live.ib_bridge import runner_split
    if (exit_type == "TP" and not t1_booked and total_contracts > 1
            and remaining_contracts == total_contracts):
        tp_qty, runner_qty = runner_split(total_contracts, t1_fraction)
        if runner_qty > 0:
            return [("t1", tp_qty), ("runner", runner_qty)]
    return [("full", remaining_contracts)]


def venue_t1_booking_plan(*, tp_confirmed: bool, t1_booked: bool,
                          total_contracts: int,
                          t1_fraction: float) -> "tuple[bool, int, int]":
    """Decide whether/what to book for a venue-side T1, driven by the TP-order
    fill confirmation rather than the (laggy, partial) position-volume read.

    The 2026-07-01 trade 622 race: the old reconcile only booked when
    ``ib_volume == expected_runner`` exactly, so a lagged/partial read (3 when
    the runner is 1) made it refuse — T1 stayed unbooked and the runner was
    swallowed by the later full-close. Booking off ``tp_confirmed`` (the TP
    order reached status Filled) decouples "T1 happened" from the volume read,
    and we book the INTENDED split so a stale read can't distort the quantities.

    Returns ``(should_book, filled, runner)`` — ``filled`` at T1, ``runner``
    held. ``(False, 0, 0)`` when there's nothing to do (already booked, TP not
    confirmed, or no runner to preserve).
    """
    from live.ib_bridge import runner_split
    if t1_booked or not tp_confirmed:
        return (False, 0, 0)
    tp_qty, runner_qty = runner_split(total_contracts, t1_fraction)
    if runner_qty <= 0:
        return (False, 0, 0)
    return (True, tp_qty, runner_qty)


def venue_t1_pnl_and_stop(*, direction: str, entry_price: float,
                          fill_price: float, filled: int,
                          breakeven_buffer_pts: float,
                          prior_day_low: float, pdl_buffer_pts: float):
    """Realized P&L for a venue-filled T1 fraction + the runner's new stop.

    Mirrors ExitManager._check_t1: the post-T1 stop moves several points under
    breakeven (``breakeven_buffer_pts`` is negative); if the prior-day low gives
    a lower stop, use that wider one so the runner has room.
    """
    if direction == "short":
        pnl = (entry_price - fill_price) * filled
    else:
        pnl = (fill_price - entry_price) * filled
    new_stop = entry_price + breakeven_buffer_pts
    if prior_day_low and prior_day_low > 0:
        new_stop = min(new_stop, prior_day_low - pdl_buffer_pts)
    return round(pnl, 2), new_stop


def build_session_bars_df(session_bars: dict) -> "pd.DataFrame":
    """Assemble the full session's OHLCV bars from an uncapped accumulator.

    ``session_bars`` maps a tz-aware Timestamp -> (open, high, low, close,
    volume). The live ``self._df`` is trimmed to the last 400 bars for
    processing speed, so it can't be the archive source — this keeps the
    entire session (overnight + pre-market included) for retrospective
    analysis. Deduplicates by timestamp (last write wins) and sorts.
    """
    cols = ["open", "high", "low", "close", "volume"]
    if not session_bars:
        return pd.DataFrame(columns=cols)
    index = sorted(session_bars.keys())
    data = [session_bars[ts] for ts in index]
    return pd.DataFrame(data, index=pd.DatetimeIndex(index), columns=cols)


def compute_excursion_pts(
    *,
    direction: str,
    entry_price: float,
    highest,
    lowest,
    exit_price,
    recent_high=None,
    recent_low=None,
):
    """Return ``(mfe_pts, mae_pts)`` for a trade, robust to delayed data.

    MFE = max favorable excursion, MAE = max adverse, both in points.

    On the ~12-min delayed feed the venue bracket can close the position
    at the target before the bars printing the true extreme arrive, so the
    in-trade high/low watermark undershoots — and the recorded MFE could
    even land *below* the realized favorable exit. We therefore fold the
    exit fill and the most recent bar's high/low into the watermark:
    the favorable side floors at the actual exit, never below it.

    Returns ``(None, None)`` when there is no price information at all.
    """
    highs = [h for h in (highest, recent_high, exit_price) if h is not None]
    lows = [lo for lo in (lowest, recent_low, exit_price) if lo is not None]
    if not highs and not lows:
        return None, None
    hi = max(highs) if highs else None
    lo = min(lows) if lows else None

    if direction == "long":
        mfe = round(hi - entry_price, 2) if hi is not None else None
        mae = round(entry_price - lo, 2) if lo is not None else None
    else:
        mfe = round(entry_price - lo, 2) if lo is not None else None
        mae = round(hi - entry_price, 2) if hi is not None else None

    # Excursions are magnitudes — a price path that never went adverse
    # yields MAE 0, not a negative number (and vice versa for MFE).
    if mfe is not None:
        mfe = max(mfe, 0.0)
    if mae is not None:
        mae = max(mae, 0.0)
    return mfe, mae


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
    # Drop HORIZONTAL_SR (the bot's own touch-counted flat lines) as an ENTRY
    # source — net loser confirmed 3 ways: live trades (33% WR, negative),
    # bootstrap, and 5y A/B (HSR = 86% of trades, −6,977 pt; dropping it flips
    # the book −6,156→+961, PF 0.79→1.21, without touching the good trades).
    # Level still kept for trailing-stop structure; just not entered on.
    block_horizontal_sr_entries=True,
    # Conviction sizing (LIVE forward-test on paper): size UP the high-conviction
    # setup — a deep crash-bottom flush reclaiming a quality level (INTRADAY_LOW
    # / Mancini CUSTOM), the 85%-WR combo from the conviction study. Size-up-only
    # (never shrinks ordinary trades). Not backtestable (harness is size-blind +
    # has no deep flushes / CUSTOM levels), so validated forward on the paper
    # account. Today's +71 deep-flush winner would size full instead of 1ct.
    use_conviction_sizing=True,
    # Shorts alert-only: no live short orders (P&L-at-targets report: the short
    # detectors are net -445.8pt at targets, 0/14 hit T1). Discord heads-up
    # still fires for plan-aligned shorts so they can be taken manually.
    shorts_alert_only=True,
    # Mancini exit scaling: T1 at first resistance level, not fixed distance
    # "Lock in 75% profits at first level up" — with 2 contracts, best we can do is 50/50
    # but T1 should be at the ACTUAL next level, not a fixed point target
    mancini_t1_at_first_resistance=True,
    # Shadow mode features: run detectors but only log, don't trade
    shadow_mode_features=True,
    use_sweep_depth_sizing=True,          # Shadow: log sweep-depth-adjusted sizing
    use_mode1_detection=True,             # LIVE: blocks FB longs while Mode 1 Red active, before mode1_red_fb_long_block_until_hour ET
    # Shadow: log Mode 1 Green confirmations (data-backed tells, PR #41).
    # 5y replay: precision 0.29 / recall 0.77, median confirm 11:40 ET.
    # shadow_mode_features=True keeps sizing/R:R actions OFF — log-only.
    use_mode1_green_detection=True,
    allow_velocity_short=True,            # Shadow: log velocity breakdown shorts
    # Back-Test Short (Mancini-faithful): broken support shelf retested from
    # below after a deep flush. See core.patterns_short_v2.BacktestShort for
    # the 3 Mancini criteria. PRIOR_DAY_LOW shelves excluded — Phase 1's
    # block_pdl_shorts rejects ANY short at PDL.
    allow_backtest_short=True,
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
    # Enabled 2026-06-11 after sign-off: four independent backtests agree
    # (fully-gated Haiku-plan harness: +638 pts, PF +0.20, WR +4.0%,
    # MaxDD -30% on 315 plan sessions). Live aggregation fixed in PR #35.
    use_5min_levels=True,
    swing_low_order_5min=6,               # 6 bars on 5-min = 30 min confirmation
    detect_shelf_levels=True,             # Mancini shelves on 5-min (the bigger half of the edge)
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
    # Mancini's verbatim danger-zone carve-out (C1'): allow non-acceptance
    # protocol FB longs through the gate. June 8 2026 we missed a $785
    # winner because the gate hard-blocked a non-acceptance signal — the
    # exact case Mancini's rule says IS the way to enter inside the zone.
    danger_zone_allow_non_acceptance=True,
    # CLUSTER_LOW quality filter: require a Mancini plan match for FB/LR
    # longs anchored to engine-derived CLUSTER_LOW levels. Backtest (332
    # sessions) showed CLUSTER_LOW was 98% of the acceptance-protocol
    # leak (-$97K). Gating these by plan match cuts the leak and routes
    # the engine to Mancini's actual structural levels.
    cluster_low_requires_plan_match=True,
    # FB level freshness gate: exempt structural-quality level types
    # (PRIOR_DAY_LOW, MULTI_HOUR_LOW, INTRADAY_LOW, CUSTOM Mancini-plan
    # levels) from the 24-36h age cap. Mancini routinely holds runners
    # against multi-day shelves (e.g. his 7517 PDL flagged "since last
    # Tuesday"). Engine-derived intraday clusters still expire normally.
    # Backtest (332 sessions): +$4.5K/1ct.
    fb_age_cap_exempt_high_quality_levels=True,
)
PRODUCTION_ELEVATOR = ElevatorParams(
    min_velocity_pts_per_min=0.75,
    min_levels_broken=2,
    higher_low_lookback=4,
)
PRODUCTION_EXIT = ExitParams(
    # 8 full-conviction contracts so the 75/15/10 actually splits into three
    # rungs (T1=6 / T2=1 / runner=1) — at 4, floor(4*0.15)=0 collapsed it to
    # 75% + runner. Lets the bot run Mancini's full level-to-level system.
    # (Paper account; conviction derating still scales down to 4/2 contracts.)
    default_contracts=8,
    # Mancini three-stage exits (2025-08-05 rule):
    #   T1: 75% off at first resistance       (3 of 4 contracts)
    #   T2: 15% off at second resistance      (rounded to ~1 of 4 — sub-contract granularity smoothed by ExitManager)
    #   Runner: 10% trails structure          (~1 of 4)
    t1_exit_fraction=0.75,
    t2_exit_fraction=0.15,
    runner_fraction=0.10,
    breakeven_buffer_pts=-3.0,  # Mancini: "several pts under breakeven"
    trailing_stop_pts=12.0,
    runner_prior_day_low_buffer_pts=1.0,
)
PRODUCTION_RISK = RiskParams(max_trades_per_day=999, max_daily_loss_pts=9999.0, skip_tuesdays=False, min_rr_ratio=0.8, max_position_contracts=8)  # Optuna v2: 0.8 R:R filter; cap raised to 8 so default_contracts=8 isn't clipped (full 75/15/10)
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


def _should_force_reconnect(minutes_since_bar: float, *, market_closed: bool,
                            connected: bool, seconds_since_last_force: float,
                            threshold_min: float = 6.0,
                            throttle_sec: float = 300.0) -> bool:
    """Whether to force a reconnect because bars went stale while the socket
    still looks connected.

    Fires ONLY for the connected-but-stale bug (IBKR farm blip / dropped data
    subscription where ``_on_disconnect`` never fired): market open, the bridge
    still thinks it's connected, past ``threshold_min``, and not throttled. A
    genuine socket disconnect (``connected=False``) is left to the normal
    reconnect path; the daily break / weekend (``market_closed``) is expected.
    """
    return (not market_closed
            and connected
            and minutes_since_bar >= threshold_min
            and seconds_since_last_force >= throttle_sec)


def _should_resubscribe(minutes_since_bar: float, *, market_closed: bool,
                        connected: bool, socket_alive: bool,
                        seconds_since_last: float,
                        threshold_min: float = 3.0,
                        throttle_sec: float = 300.0) -> bool:
    """Whether to do a LIGHT data re-subscribe instead of a full reconnect.

    Fires when bars are stale but the socket still answers a ping
    (``socket_alive``): the data farm dropped the subscription while the
    connection itself is fine, so re-requesting market data fixes it without a
    full reconnect — which would wipe the position cache (the phantom-exit
    window). If the ping fails (``socket_alive=False``), this returns False and
    the caller falls back to ``_should_force_reconnect`` / ``force_reconnect``.
    """
    return (not market_closed
            and connected
            and socket_alive
            and minutes_since_bar >= threshold_min
            and seconds_since_last >= throttle_sec)


def blocked_alert_key(signal) -> str:
    """Dedup key for a 'missed setup while in a position' alert — one per
    setup-type + rounded level per session, so it fires once (on the first bar
    the blocked setup appears), not on every bar we remain in the position."""
    st = getattr(getattr(signal, "signal_type", None), "name", "SIG")
    lvl = getattr(getattr(signal, "pattern", None), "level", None)
    price = getattr(lvl, "price", None)
    if not price:
        price = getattr(signal, "entry_price", 0) or 0
    return f"{st}@{round(float(price))}"


def route_short_to_alert_only(direction: str, shorts_alert_only: bool) -> bool:
    """Whether a signal should be alert-only (no live order) because it's a
    short and live shorts are disabled. The P&L-at-targets report showed the
    existing short detectors are net-negative (13/14 lose at targets, 0/14 hit
    T1), so production routes all shorts to the Discord alert/shadow path."""
    return shorts_alert_only and (direction or "").lower() == "short"


def recovery_blocked_by_connectivity(connectivity_down: bool,
                                     seconds_since_restored: float,
                                     grace_sec: float = 15.0) -> bool:
    """Whether to DEFER recovery (resubscribe/reconnect/ping) because IBKR
    connectivity is mid-blip or only just restored. Firing a blocking IB call
    during the churn is what hung the loop on 2026-06-29 (both freezes). Wait
    for connectivity to settle (the freeze watchdog still backstops a true
    hang). True = defer this cycle."""
    if connectivity_down:
        return True
    return seconds_since_restored < grace_sec


def _should_force_exit_frozen(idle_seconds: float, threshold_sec: float = 240.0) -> bool:
    """Whether the main loop has been idle long enough to be considered FROZEN
    (a hung synchronous IB call wedged the single-threaded loop, as on
    2026-06-29 when a resubscribe collided with a competing-login Error 162).
    A threshold of 0 disables the watchdog. The threshold must exceed the
    longest legitimate blocking call (a ~2-min reconnect burst), so the 240s
    default leaves margin."""
    if threshold_sec <= 0:
        return False
    return idle_seconds >= threshold_sec


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
        # Short heads-up alerts: the bot is long-only and shadow-detects shorts.
        # Post a Discord heads-up the first time each distinct short setup fires
        # (deduped by level), never an order. Toggle off with SHORT_ALERTS=0.
        self._short_alert_keys: set[str] = set()
        self._short_alerts_enabled = os.environ.get("SHORT_ALERTS", "1") != "0"
        # Missed-setup alerts: when already in a position (single-position; IB
        # nets) a new signal can't be taken, but we alert + record it so it
        # isn't silently lost (it may be a better setup than the trade we hold).
        # Deduped per setup per session. Toggle off with BLOCKED_ALERTS=0.
        self._blocked_alert_keys: set[str] = set()
        self._blocked_alerts_enabled = os.environ.get("BLOCKED_ALERTS", "1") != "0"
        # Minutes of no bars (while connected + market open) before forcing a
        # reconnect to re-subscribe — self-heal the connected-but-stale outage.
        self._stale_reconnect_min = float(os.environ.get("STALE_RECONNECT_MIN", "6.0"))
        self._last_force_reconnect = 0.0
        # Freeze watchdog: if the main loop stops iterating (a hung IB call
        # wedged the single-threaded loop), a daemon thread force-exits so
        # Docker's restart-unless-stopped policy brings up a fresh process.
        # 0 disables. Must exceed the longest legit blocking call (~2m reconnect).
        self._freeze_timeout_sec = float(os.environ.get("FREEZE_TIMEOUT_SEC", "240"))
        self._freeze_check_interval = 30.0
        self._last_loop_progress = _time.monotonic()
        self._freeze_exit = lambda: os._exit(1)  # injectable for tests
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
        # Full-session accumulator (timestamp -> OHLCV) for archival. Unlike
        # self._df (trimmed to 400 bars for processing), this keeps the whole
        # session so overnight/pre-market bars survive for retrospective audit.
        self._session_bars: dict = {}
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
        # Globex-aware session date (the day the session CLOSES). After
        # 18:00 ET we are already in the next calendar day's session.
        self._session_date: date = self._compute_globex_trading_date(datetime.now(_ET))

        # Multi-session runner state: tracks how many EOD rolls a runner has
        # survived. Reset to 0 every time a new position opens; incremented
        # once per session-rollover when AFTER_T2 runner is kept alive.
        # Compared against multi_session_runner_max_days to enforce safety cap.
        self._runner_sessions_held: int = 0

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

    def _check_freeze(self) -> bool:
        """One freeze check: if the main loop has been idle past the threshold
        (hung IB call), force-exit so Docker restarts a fresh process. Returns
        True if it triggered the exit hook."""
        idle = _time.monotonic() - self._last_loop_progress
        if _should_force_exit_frozen(idle, self._freeze_timeout_sec):
            logger.critical(
                f"MAIN LOOP FROZEN for {idle:.0f}s "
                f"(>{self._freeze_timeout_sec:.0f}s) — likely a hung IB call "
                f"(e.g. a resubscribe during a competing-login Error 162). "
                f"Force-exiting for a clean Docker restart."
            )
            self._freeze_exit()
            return True
        return False

    def _start_freeze_watchdog(self) -> None:
        """Daemon thread that force-exits the process if the main run loop stops
        iterating — the single-threaded loop can't unstick a hung synchronous IB
        call, so we exit and let Docker reconnect us fresh."""
        import threading

        def _watch() -> None:
            while self._running:
                _time.sleep(self._freeze_check_interval)
                try:
                    if self._check_freeze():
                        return
                except Exception:
                    pass

        threading.Thread(target=_watch, daemon=True, name="freeze-watchdog").start()
        logger.info(
            f"Freeze watchdog armed (force-exit after "
            f"{self._freeze_timeout_sec:.0f}s of a stuck loop)"
            if self._freeze_timeout_sec > 0 else "Freeze watchdog disabled"
        )

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

        # Globex session boundary (18:00 ET): if we restart after the new
        # session has already opened, the trading_date is the NEXT calendar
        # day, not today's calendar date. Plan files, level snapshots and
        # archives all key off this — using date.today() here would load
        # yesterday's plan after a 18:00–midnight restart.
        self._session_date = self._compute_globex_trading_date(datetime.now(_ET))

        # Connect to IB with retry. IB Gateway takes ~120s to fully accept
        # API connections after cold start (Java boot + login + session
        # config), and the bot can race that on a coordinated docker
        # compose up. Without retry, the bot exits and docker restarts it
        # in a tight loop — observed 25 restarts in 23 min on 2026-06-06
        # while Gateway was rebuilding. PR #21 added graceful mid-session
        # reconnect but didn't cover this initial-connect window.
        connect_max_attempts = 12  # ~6 min budget — Gateway boot ≤ 120s
        connect_attempt = 0
        while self._running:
            if self.bridge.connect():
                break
            connect_attempt += 1
            if connect_attempt >= connect_max_attempts:
                logger.error(
                    f"Failed to connect to IB after {connect_max_attempts} "
                    "attempts — giving up"
                )
                return
            delay = min(30, 5 * connect_attempt)
            logger.warning(
                f"IB connect attempt {connect_attempt}/{connect_max_attempts} "
                f"failed — retrying in {delay}s"
            )
            _time.sleep(delay)
        if not self._running:
            return  # shutdown signal during retry

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
        # Counter for consecutive in-loop errors. Resets to 0 on a clean
        # iteration. Crashes loud only after many consecutive failures —
        # otherwise we keep trying and let the existing reconnect machinery
        # bring the bridge back.
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 60  # ~5 minutes at 5s backoff
        # Arm the freeze watchdog now that the loop is about to start.
        self._start_freeze_watchdog()
        try:
            while self._running:
                # Heartbeat for the freeze watchdog: proves the loop is still
                # iterating. If a synchronous IB call below hangs, this stops
                # advancing and the watchdog force-exits for a clean restart.
                self._last_loop_progress = _time.monotonic()
                try:
                    # Check if IB disconnected and needs reconnection
                    if self.bridge._needs_reconnect:
                        self.bridge.check_reconnect()

                    # Process EVERY bar that closed since the last cycle, not
                    # just the latest — if the loop stalled (reconnect, pacing,
                    # farm blip) the intermediate bars are replayed here instead
                    # of being dropped, so a gap never silently skips a setup.
                    new_bars = self.bridge.get_new_bars()
                    if new_bars:
                        if len(new_bars) > 1:
                            logger.warning(
                                f"BACKFILL: replaying {len(new_bars)} bars that closed "
                                f"during a gap ({new_bars[0]['timestamp']} → "
                                f"{new_bars[-1]['timestamp']}) — catching up so no "
                                f"setup/exit is missed"
                            )
                        self._last_bar_received = _time.monotonic()
                        for bar in new_bars:
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
                            # Auto-recover. The socket can stay connected while the
                            # data subscription is dead (Error 1100/1102 farm blip /
                            # 10141), so _on_disconnect never fires (the 2026-06-26
                            # ~12h outage). Ping the socket to choose the cheapest fix:
                            #   ping OK   → only the data sub is stale → re-subscribe
                            #               (light; keeps the position cache, so it
                            #               can't trigger a phantom exit).
                            #   ping fail → socket truly dead → full reconnect.
                            last_force = getattr(self, "_last_force_reconnect", 0.0)
                            connected = getattr(self.bridge, "_connected", False)
                            # PREVENTION: if IBKR is mid-blip or just restored, DEFER
                            # recovery — firing a blocking resubscribe/reconnect/ping
                            # during the churn is what hung the loop (2026-06-29). Wait
                            # for connectivity to settle; the freeze watchdog backstops
                            # a true hang.
                            if recovery_blocked_by_connectivity(
                                    self.bridge.connectivity_down(),
                                    self.bridge.seconds_since_connectivity_restored()):
                                logger.info(
                                    "IBKR connectivity unstable (mid-blip / just "
                                    "restored) — deferring recovery to avoid hanging "
                                    "a request")
                                self.bridge.sleep(self.bridge.config.poll_interval_sec)
                                continue
                            socket_alive = self.bridge.ping()
                            resub_min = getattr(self.bridge.config,
                                                "resubscribe_stale_min", 3.0)
                            if _should_resubscribe(
                                    minutes_since_bar,
                                    market_closed=_is_market_closed(),
                                    connected=connected,
                                    socket_alive=socket_alive,
                                    seconds_since_last=_time.monotonic() - last_force,
                                    threshold_min=resub_min):
                                self._last_force_reconnect = _time.monotonic()
                                self.bridge.resubscribe(
                                    f"no new bars for {minutes_since_bar:.0f} min")
                            elif _should_force_reconnect(
                                    minutes_since_bar,
                                    market_closed=_is_market_closed(),
                                    connected=connected,
                                    seconds_since_last_force=_time.monotonic() - last_force,
                                    threshold_min=self._stale_reconnect_min):
                                self._last_force_reconnect = _time.monotonic()
                                self.bridge.force_reconnect(
                                    f"no new bars for {minutes_since_bar:.0f} min")

                    # IB-aware sleep keeps the event loop alive for streaming callbacks
                    self.bridge.sleep(self.bridge.config.poll_interval_sec)

                    # Session rollover: detect new Globex session (date change)
                    # Globex sessions start at 18:00 ET, so the "trading date" rolls
                    # over then. Without this, daily loss limits from yesterday block
                    # today's signals.
                    self._check_session_rollover()

                    # Sync position with IB periodically
                    self._sync_position()

                    # Clean iteration — reset the error counter.
                    consecutive_errors = 0

                except ConnectionError as e:
                    # IB Gateway nightly re-auth (~19:45 ET) drops the socket.
                    # Previously this exception killed main() and relied on
                    # Docker's restart-policy to bring the bot back, losing
                    # 2-5 minutes of bars per outage AND any setup that
                    # flushed during the gap. Now we flag the bridge for
                    # reconnect and continue the loop — check_reconnect() at
                    # the top will handle it on the next iteration.
                    consecutive_errors += 1
                    if self.bridge._connected:
                        self.bridge._connected = False
                    self.bridge._needs_reconnect = True
                    logger.error(
                        f"IB ConnectionError (consecutive #{consecutive_errors}): {e} "
                        f"— flagging for reconnect, continuing main loop"
                    )
                    _time.sleep(5.0)
                except Exception as e:
                    # Catch-all so a downstream pattern/exit bug doesn't kill
                    # the entire bot. Log loud, brief backoff, keep going.
                    consecutive_errors += 1
                    logger.exception(
                        f"Unexpected error in main loop (consecutive #{consecutive_errors}): {e}"
                    )
                    _time.sleep(5.0)

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        f"Main loop has hit {consecutive_errors} consecutive errors — "
                        f"bailing out so Docker can restart us cleanly."
                    )
                    break

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

    @staticmethod
    def _compute_globex_trading_date(now_et: datetime) -> date:
        """Return the Globex trading_date for the given ET timestamp.

        CME Globex ES sessions run 18:00 ET (previous day) -> 17:00 ET. The
        "trading_date" labels the day the session CLOSES — so anything at or
        after 18:00 ET belongs to the NEXT calendar day's session.

        17:00–18:00 ET is the daily break. We attribute it to the next
        session here (engine-side bookkeeping), matching how plan files and
        levels for the upcoming session are usually staged during this
        window. _check_session_rollover() has its own break-window handling
        which short-circuits rather than rolling during the gap; that
        difference is intentional — initialization just needs a date, while
        rollover decides when to RESET daily state.

        Args:
            now_et: timezone-aware datetime in US/Eastern.

        Returns:
            date object representing the active trading session.
        """
        if now_et.time() >= time(18, 0):
            from datetime import timedelta
            return (now_et + timedelta(days=1)).date()
        return now_et.date()

    def _load_mancini_llm_plan(self) -> None:
        """Load the nightly LLM-extracted Mancini plan for self._session_date.

        Idempotent: safe to call from _initialize_session() and again from
        _check_session_rollover(). Failures (missing file, corrupt JSON,
        validation error) are swallowed — the bot runs without the plan.
        """
        self._mancini_llm_plan = None
        sp = getattr(self.strategy, "strategy_params", None)
        if sp is None or not getattr(sp, "use_mancini_llm_plan", False):
            return
        try:
            from live.mancini_llm_extract import load_plan as _load_llm_plan

            plan = _load_llm_plan(
                self._session_date,
                input_dir=Path(getattr(sp, "mancini_llm_plan_dir", "/app/data")),
            )
            if plan is not None:
                self._mancini_llm_plan = plan
                self.signal_aggregator.set_mancini_llm_plan(plan)
                # Inject Mancini's planned long setups as CUSTOM levels so the
                # FB pattern detector can fire on them. Without this, the plan
                # only gates / boosts engine-discovered levels — a Mancini
                # high-conviction FB level that the engine hasn't classified
                # as a swing/cluster/PDL low is invisible. See _HIGH_QUALITY_LEVELS
                # in core/patterns.py — CUSTOM is whitelisted there.
                injected = self._inject_plan_levels(plan)
                logger.info(
                    f"Mancini LLM plan loaded for {self._session_date}: "
                    f"lean={plan.lean} mode={plan.mode} "
                    f"setups={len(plan.planned_setups)} "
                    f"danger={len(plan.danger_zones)} "
                    f"no_trade_above={plan.no_trade_above} "
                    f"no_trade_below={plan.no_trade_below} "
                    f"levels_injected={injected}"
                )
            else:
                logger.info(
                    f"Mancini LLM plan: no plan available for {self._session_date}"
                )
        except Exception as e:
            logger.warning(f"Mancini LLM plan load failed (non-fatal): {e}")

    # Levels the engine considers structurally high quality. Mirrors
    # core/patterns.py _HIGH_QUALITY_LEVELS — these are the only level
    # types the FB pattern detector will already accept as the basis
    # for a sweep+reclaim. Used for the collection-mode quality gate.
    _HIGH_QUALITY_LEVEL_TYPES = frozenset({
        "PRIOR_DAY_LOW",
        "PRIOR_DAY_HIGH",
        "MULTI_HOUR_LOW",
        "MULTI_HOUR_HIGH",
        "INTRADAY_LOW",
        "CUSTOM",  # Mancini-injected and other overlay-sourced levels
    })

    _MANCINI_PLAN_MATCH_TOLERANCE_PTS = 2.0

    def _build_bypass_entry(self, signal, gates_bypassed: list) -> EntryDecision:
        """Build the EntryDecision for a collection-mode time-gate bypass.

        Bypass entries skip EntryManager sizing entirely, so they must not
        inherit full default_contracts: trade #16229 (2026-06-05) carried
        position_size_factor 0.25 for a 31-pt stop yet went out 4 contracts
        and lost 130 pts. Collection data is worth the same at minimum size
        — risk-floor every bypass entry at 1 contract.
        """
        return EntryDecision(
            should_enter=True,
            signal=signal,
            contracts=1,
            reason=f"Bypass: {', '.join(gates_bypassed)}",
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
        )

    def _collection_mode_is_quality_setup(self, signal) -> bool:
        """Return True if a signal that was about to be taken via the time-
        gate bypass is also a *quality* setup worth collecting.

        Pure noise — engine-derived mid-range swing levels with no plan
        match — should not be taken even in collection mode. Tonight's
        LR @ 7607 (no plan level, SWING_LOW type, lost $425) is the
        canonical bad case.

        Two ways to qualify:
          1. The signal's level is on Mancini's LLM plan within 2 pt
             (regardless of conviction — plan-listed = Mancini blessed)
          2. The level type itself is structurally high-quality
             (PDL, MULTI_HOUR_LOW, INTRADAY_LOW, CUSTOM)
        """
        pat = getattr(signal, "pattern", None)
        if pat is None or getattr(pat, "level", None) is None:
            return False
        lvl = pat.level
        lvl_type_name = getattr(lvl.level_type, "name", "")

        # Path 1: high-quality structural level type. Access via the class
        # so test doubles can pass `self` as a SimpleNamespace.
        if lvl_type_name in IBRunner._HIGH_QUALITY_LEVEL_TYPES:
            return True

        # Path 2: matches a Mancini-listed plan level within tolerance.
        # Same matching tolerance used by _mancini_llm_setup_bonus.
        plan = getattr(self, "_mancini_llm_plan", None)
        if plan is None:
            return False
        tol = IBRunner._MANCINI_PLAN_MATCH_TOLERANCE_PTS
        for setup in (plan.planned_setups or []):
            try:
                if abs(setup.level_price - lvl.price) <= tol:
                    return True
            except (TypeError, AttributeError):
                continue
        return False

    def _inject_plan_levels(self, plan) -> int:
        """Push Mancini's planned LONG setups into the engine's level store
        as CUSTOM levels so the FB pattern detector can fire on them.

        Only LONG-direction FB / level-reclaim setups are injected. Short
        setups and trend-continuation entries are skipped — those run via
        their own detectors (or are not auto-tradeable). Confirmed at the
        moment of load so the levels are immediately eligible.

        Returns the number of levels injected.
        """
        try:
            from config.levels import Level, LevelType
        except Exception:
            return 0
        store = getattr(self.signal_aggregator, "level_store", None)
        if store is None:
            return 0
        now = datetime.now(_ET)
        conv_score = {"high": 3, "medium": 2, "low": 1}
        accept_types = {"failed_breakdown", "level_reclaim"}
        injected = 0
        for setup in (plan.planned_setups or []):
            if (getattr(setup, "direction", "") or "").lower() != "long":
                continue
            if (getattr(setup, "setup_type", "") or "").lower() not in accept_types:
                continue
            price = float(getattr(setup, "level_price", 0.0) or 0.0)
            if price <= 0:
                continue
            ctx = (getattr(setup, "context", "") or "")[:120]
            lvl = Level(
                price=price,
                level_type=LevelType.CUSTOM,
                created_at=now,
                confirmed_at=now,
                touch_count=1,
                label=f"MANCINI_PLAN:{setup.setup_type}@{price:.2f}",
                mancini_confirmed=True,
                mancini_side="support",
                mancini_conviction=conv_score.get(
                    (getattr(setup, "conviction", "") or "").lower(), 1
                ),
                mancini_tags=[
                    "llm_plan",
                    f"conv:{(setup.conviction or '').lower()}",
                    f"type:{setup.setup_type}",
                ],
            )
            store.add(lvl)
            injected += 1
            logger.debug(
                f"Mancini plan level injected: {lvl.label} "
                f"({setup.conviction} conviction) — {ctx}"
            )

        # Mancini's published TARGET ladder (e.g. 7424→7452→7472→...). Currently
        # the engine derives T1/T2 from its own levels and ignores these; inject
        # them so target selection can use where HE targets. Where a target
        # coincides with an engine level, bump source_count (confluence).
        sp = getattr(self.signal_aggregator, "strategy_params", None)
        if getattr(sp, "use_mancini_targets", False):
            tol = getattr(sp, "mancini_target_confluence_tol_pts", 3.0)
            for tprice in (getattr(plan, "targets", None) or []):
                try:
                    tprice = float(tprice)
                except (TypeError, ValueError):
                    continue
                if tprice <= 0:
                    continue
                tlvl = Level(
                    price=tprice,
                    level_type=LevelType.MANCINI_LEVEL,
                    created_at=now,
                    confirmed_at=now,
                    label=f"MANCINI_TARGET@{tprice:.2f}",
                    mancini_confirmed=True,
                    mancini_tags=["llm_plan", "target"],
                )
                # Confluence: an engine level near this target = 2 sources.
                for ex in store.get_active():
                    if ex.level_type in (LevelType.CUSTOM, LevelType.MANCINI_LEVEL):
                        continue
                    if abs(ex.price - tprice) <= tol:
                        ex.source_count = max(getattr(ex, "source_count", 1), 2)
                        tlvl.source_count = 2
                        break
                store.add(tlvl)
                injected += 1
        return injected

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
            # Filter disabled (or insufficient history) — install a placeholder
            # RegimeState. Use NaN ema_slope so downstream JSONL records don't
            # silently treat 0.0 as a real reading. The gates default to ON
            # so trades aren't blocked when the filter is off intentionally.
            from core.regime_filter import RegimeState, Direction, VolRegime
            self.strategy._regime_state = RegimeState(
                direction=Direction.NEUTRAL,
                vol_regime=VolRegime.NORMAL,
                longs_enabled=True,
                shorts_enabled=True,
                ema_slope=float("nan"),  # explicit sentinel — not "0 slope"
            )
            why = ("disabled"
                   if not self.strategy.strategy_params.use_regime_filter
                   else f"insufficient history ({len(self.strategy._daily_history) if self.strategy._daily_history is not None else 0} < 50 bars)")
            logger.warning(f"Regime: NEUTRAL placeholder — {why}; ema_slope=NaN")

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
        self._load_mancini_llm_plan()

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

        # Full-session archive accumulator (uncapped, deduped by timestamp).
        self._session_bars[timestamp] = (open_, high, low, close, volume)

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
        else:
            # Step 2b: already in a position (often a collection-mode experiment).
            # We can't take a second live order (single-position; IB nets same-
            # direction fills into one), but DETECT anyway and alert + record any
            # blocked setup — it may be a more legit setup than the trade we hold.
            # No IB order is ever placed here.
            self._detect_blocked_signal(open_, high, low, close, volume, vel,
                                        current_time, timestamp)

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

    def _detect_blocked_signal(self, open_, high, low, close, volume, vel,
                               current_time, timestamp) -> None:
        """Run signal detection while already in a position and alert + record
        any setup we can't take (single-position). Mirrors the flat-path
        detection inputs but NEVER places an order. Best-effort — a failure here
        must never disrupt the bar loop or the open position's management."""
        if not self._blocked_alerts_enabled:
            return
        try:
            bar_idx = len(self._df) - 1
            try:
                _md = fetch_market_snapshot()
            except Exception:
                _md = None
            _ctx = {
                "session_date": self._session_date,
                "current_price": close,
                "session_high": float(self._df["high"].max()),
                "session_low": float(self._df["low"].min()),
                "bar_count": self._bar_count,
            }
            signal = self.signal_aggregator.update(
                bar_idx=bar_idx, timestamp=timestamp, open_=open_, high=high,
                low=low, close=close, volume=volume, velocity=vel, df=self._df,
                market_data=_md, session_context=_ctx,
            )
            if signal is not None:
                self._alert_blocked_signal(signal, timestamp)
        except Exception as e:
            logger.warning(f"blocked-signal detection error: {e!r}")

    def _alert_blocked_signal(self, signal, timestamp) -> None:
        """Record + Discord-alert a setup that fired while already in a position
        (no live order). Deduped per setup per session. Surfaces the setup's
        quality (type / LQS / R:R) so it can be judged against the (often
        experimental collection-mode) trade we're holding."""
        # Record for later analysis / phantom outcome tracking.
        try:
            self._add_phantom(signal, "blocked:in_position", timestamp)
        except Exception:
            pass
        key = blocked_alert_key(signal)
        if key in self._blocked_alert_keys:
            return
        self._blocked_alert_keys.add(key)
        st = getattr(getattr(signal, "signal_type", None), "name", "SIGNAL")
        direction = (getattr(signal, "direction", "long") or "long")
        entry = float(getattr(signal, "entry_price", 0.0) or 0.0)
        lqs = getattr(signal, "lqs", None)
        rr = float(getattr(signal, "rr_ratio_t1", 0.0) or 0.0)
        logger.warning(
            f"BLOCKED SETUP (in position — recorded, NO order): {st} {direction} "
            f"@ {entry:.2f}" + (f" LQS={lqs}" if lqs is not None else "")
            + (f" R:R={rr:.1f}" if rr else "")
        )
        try:
            from live.trade_notifications import post_payload, get_webhook_url
            webhook = get_webhook_url()
            if not webhook:
                return
            held = self._position
            held_desc = (
                f"{getattr(held, 'direction', '?')} "
                f"{getattr(held, 'remaining_contracts', '?')} @ "
                f"{getattr(held, 'entry_price', 0.0):.2f}"
            ) if held is not None else "?"
            desc = (
                "A setup fired but the bot is already in a trade — **no order placed** "
                "(single-position).\n"
                f"**Missed:** {st} {direction} @ {entry:.2f}"
                + (f"  ·  LQS {lqs}" if lqs is not None else "")
                + (f"  ·  R:R {rr:.1f}" if rr else "")
                + f"\n**Currently holding:** {held_desc}\n"
                "_Recorded as a phantom for outcome tracking — compare vs the held trade._"
            )
            embed = {
                "title": f"⚠️ MISSED SETUP (in a position): {st} {direction.upper()} @ {entry:.2f}",
                "description": desc,
                "color": 0xF1C40F,
            }
            post_payload({"username": "Mancini Bot", "embeds": [embed]}, webhook)
        except Exception as e:
            logger.warning(f"blocked-signal alert error: {e!r}")

    def _evaluate_and_enter(self, signal: Signal, current_time: time, timestamp: datetime) -> None:
        """Evaluate signal through risk/entry gates, execute if approved.

        In bypass_session_gates mode, time-based gates are recorded but not
        enforced — the trade is taken anyway with a 'gate_bypassed' marker.
        Non-time gates (max trades, done for day) still block.
        """
        # Shorts are alert-only in production: the P&L-at-targets report proved
        # the existing short detectors are net-negative (13/14 lose at targets,
        # 0/14 hit T1). Route every short to the Discord alert/shadow path (the
        # heads-up still fires from the shadow events) — never a live order.
        if route_short_to_alert_only(
                getattr(signal, "direction", ""),
                getattr(self.strategy.strategy_params, "shorts_alert_only", False)):
            logger.info(
                f"SHORT ALERT-ONLY: {signal.signal_type.name} @ "
                f"{signal.entry_price:.2f} — no live order (shorts are net-negative; "
                f"Discord heads-up still fires)")
            self._add_phantom(signal, "shorts_alert_only", timestamp)
            return

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
            # COLLECTION MODE QUALITY FILTER: bypass time windows ONLY for
            # high-quality setups. The whole point of collection mode is to
            # gather data on "would Mancini take this if not for the clock?"
            # — which is interesting only when the setup IS Mancini-quality.
            # A low-quality mid-range LR at a SWING_LOW with no plan match
            # is just noise. Today's trade #15161 (LR @ 7607, lost $425)
            # was the wake-up call.
            #
            # Required: either the level is on Mancini's plan within 2pt,
            # or the engine classified it as a high-quality structural
            # level (PRIOR_DAY_LOW, MULTI_HOUR_LOW, INTRADAY_LOW, CUSTOM).
            if not self._collection_mode_is_quality_setup(signal):
                lvl = signal.pattern.level
                lvl_type = getattr(lvl.level_type, "name", "?") if lvl else "?"
                lvl_price = lvl.price if lvl else 0.0
                logger.info(
                    f"COLLECTION MODE QUALITY REJECT: {signal.signal_type.name} "
                    f"@ {signal.entry_price:.2f} on {lvl_type}@{lvl_price:.2f} — "
                    f"low quality + no plan match. "
                    f"Would have bypassed time gates: {', '.join(gates_that_would_fire)}"
                )
                self._add_phantom(
                    signal,
                    f"collection_mode_low_quality:{lvl_type}",
                    timestamp,
                )
                return

            logger.info(
                f"COLLECTION MODE: Taking {signal.signal_type.name} @ {signal.entry_price:.2f} "
                f"(production would skip: {', '.join(gates_that_would_fire)})"
            )
            # Entry was rejected by time gate — use signal values directly
            if entry.contracts <= 0:
                entry = self._build_bypass_entry(signal, gates_that_would_fire)

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
            entry_price=entry.entry_price,
            slippage_cap_pts=getattr(
                self.exit_params, "entry_slippage_cap_pts", 0.0),
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
        # Reset multi-session runner counter — new position, new clock.
        self._runner_sessions_held = 0
        self.position_manager.open_position(self._position, timestamp, self._pattern_type)

        logger.info(
            f"ENTRY: {entry.contracts} {self.contract.symbol} @ {actual_entry_price:.2f} "
            f"stop={entry.stop_price:.2f} T1={signal.target_1:.2f} "
            f"R:R={signal.rr_ratio_t1:.1f} [{signal.signal_type.name}]"
        )
        # Log rich entry to persistent trade log
        self._log_trade(self._position, signal, "entry")
        # Post rich Discord embed for this entry (best-effort, never blocks).
        self._post_trade_entry_embed(
            position=self._position,
            signal=signal,
            fill_price=actual_entry_price,
            contracts_ordered=entry.contracts,
        )

    def _revert_position_close(self, position, action: ExitAction) -> None:
        """Undo ExitManager's close mutation after an unconfirmed broker close.

        _stop_out (and friends) zero remaining_contracts, set phase=CLOSED
        and book the realized P&L BEFORE the broker order goes out. If the
        flatten could not be confirmed, the position is still live at IB —
        restore the open state so exit checks fire again next bar.
        """
        if position.direction == "short":
            pnl = (position.entry_price - action.exit_price) * action.contracts_to_close
        else:
            pnl = (action.exit_price - position.entry_price) * action.contracts_to_close
        position.realized_pnl_pts -= pnl
        position.remaining_contracts = action.contracts_to_close
        if position.t2_hit:
            position.phase = ExitPhase.AFTER_T2
        elif position.t1_hit:
            position.phase = ExitPhase.AFTER_T1
        else:
            position.phase = ExitPhase.INITIAL

    def _handle_exit_action(self, action: ExitAction, timestamp: datetime) -> bool:
        """Translate ExitAction from ExitManager into IB orders.

        Returns False when a full close could not be confirmed at the
        broker — position state is reverted to open so the exit retries
        next bar, and the caller's exit bookkeeping is skipped (the
        position reads as still open).
        """
        if self._trade_id is None:
            return True

        # Capture pre-action position state so the embed can show
        # accurate "remaining contracts" + cumulative P&L.
        pre_position = self._position

        if action.new_phase == ExitPhase.CLOSED:
            if not self.bridge.flatten(reason=action.reason):
                logger.critical(
                    f"FLATTEN UNCONFIRMED for trade {self._trade_id} "
                    f"[{action.reason}] — position may still be open at IB; "
                    f"reverting close state, will retry next bar"
                )
                if self._position is not None:
                    self._revert_position_close(self._position, action)
                return False
            logger.info(f"EXIT: flatten -- {action.reason}")
            # Classify phase for the embed
            r = (action.reason or "").lower()
            if "stop loss" in r or "stop" in r:
                phase = "stop"
            elif "eod" in r:
                phase = "eod"
            elif "trail" in r or "structure" in r:
                phase = "runner_trail"
            else:
                phase = "stop"
            self._post_trade_exit_embed(
                phase=phase,
                action=action,
                pre_position=pre_position,
                timestamp=timestamp,
            )

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
            self._post_trade_exit_embed(
                phase="t1",
                action=action,
                pre_position=pre_position,
                timestamp=timestamp,
            )

        elif action.reason.startswith("Target 2"):
            self.bridge.partial_exit(
                trade_id=self._trade_id,
                quantity=action.contracts_to_close,
                new_sl=action.new_stop,
                reason=action.reason,
            )
            self._log_partial_exit(action, timestamp)
            self._post_trade_exit_embed(
                phase="t2",
                action=action,
                pre_position=pre_position,
                timestamp=timestamp,
            )

        else:
            # Stop update (trailing) — no notification, just a stop modify.
            self.bridge.update_stop(
                trade_id=self._trade_id,
                new_sl=action.new_stop,
                reason=action.reason,
            )

        return True

    # ------------------------------------------------------------------
    # Discord rich-embed notifications for entries and exits
    # ------------------------------------------------------------------

    def _classify_fb_entry_path(self) -> str:
        """Which failed-breakdown logic fired, read from the FB detector state:
        ``double_dip`` (retested twice), ``level_sweep`` (classic swept-below
        then reclaimed), or ``elevator_fb`` (momentum recovery, no real sweep).
        Used by both the trade log and the Discord entry embed."""
        fb = getattr(self.signal_aggregator, "failed_breakdown", None)
        if fb is not None and getattr(fb, "_is_double_dip", False):
            return "double_dip"
        if fb is not None and getattr(fb, "_is_level_sweep", False):
            return "level_sweep"
        return "elevator_fb"

    def _post_trade_entry_embed(self, *,
                                position,
                                signal,
                                fill_price: float,
                                contracts_ordered: int) -> None:
        """Best-effort: build + POST the entry embed to Discord. Any
        failure is logged and swallowed — we never block trading on a
        notification."""
        try:
            from live.trade_notifications import (
                build_entry_embed, post_payload, get_webhook_url,
            )
            webhook = get_webhook_url()
            if not webhook:
                return
            payload = build_entry_embed(
                position=position,
                signal=signal,
                fill_price=fill_price,
                contracts_ordered=contracts_ordered,
                contract_spec=self.contract,
                exit_params=self.exit_params,
                plan=getattr(self, "_mancini_llm_plan", None),
                session_date=str(self._session_date),
                entry_time=getattr(self, "_entry_timestamp", None),
                trade_id=getattr(self, "_trade_id", None) or None,
                gate_bypass=getattr(self, "_current_gate_bypass", None),
            )
            ok, info = post_payload(payload, webhook)
            if not ok:
                logger.warning(f"Trade entry embed post failed: {info}")
        except Exception as e:
            logger.warning(f"Trade entry embed build failed: {e!r}")

    def _post_trade_exit_embed(self, *,
                               phase: str,
                               action,
                               pre_position,
                               timestamp: datetime) -> None:
        """Best-effort: build + POST the exit embed (T1 / T2 / stop /
        runner trail / EOD). Reads pre_position to compute remaining
        contracts AFTER this fill correctly."""
        try:
            from live.trade_notifications import (
                build_exit_embed, post_payload, get_webhook_url,
            )
            webhook = get_webhook_url()
            if not webhook:
                return
            if pre_position is None:
                return
            entry_price = float(getattr(pre_position, "entry_price", 0.0))
            direction = (getattr(pre_position, "direction", None) or "long").lower()
            fill_price = float(getattr(action, "exit_price", 0.0))
            contracts_closed = int(getattr(action, "contracts_to_close", 0) or 0)
            # Remaining AFTER this fill
            remaining_before = int(getattr(pre_position, "remaining_contracts", 0))
            remaining_after = max(0, remaining_before - contracts_closed)
            if phase in ("stop", "runner_trail", "eod"):
                # These close everything that's left
                contracts_closed = remaining_before
                remaining_after = 0
            # Cumulative realized PnL across the trade so far
            realized_so_far = float(getattr(pre_position, "realized_pnl_pts", 0.0))
            if direction == "long":
                slice_pnl = (fill_price - entry_price) * contracts_closed
            else:
                slice_pnl = (entry_price - fill_price) * contracts_closed
            cum_pnl_pts = realized_so_far + slice_pnl
            new_stop = getattr(action, "new_stop", None)
            target_2 = float(getattr(pre_position, "target_2", 0.0))
            next_target = (target_2 if phase == "t1" and target_2 > 0
                           and remaining_after > 0 else None)
            payload = build_exit_embed(
                phase=phase,
                fill_price=fill_price,
                contracts_closed=contracts_closed,
                entry_price=entry_price,
                direction=direction,
                contract_spec=self.contract,
                remaining_contracts=remaining_after,
                realized_pnl_pts_so_far=cum_pnl_pts,
                new_stop=new_stop,
                next_target=next_target,
                reason=getattr(action, "reason", ""),
                fill_time=timestamp,
                trade_id=getattr(self, "_trade_id", None) or None,
                gate_bypass=getattr(self, "_current_gate_bypass", None),
            )
            ok, info = post_payload(payload, webhook)
            if not ok:
                logger.warning(f"Trade exit embed post failed: {info}")
        except Exception as e:
            logger.warning(f"Trade exit embed build failed: {e!r}")

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

        # Never interpret position reads while the bridge is down — a dead
        # connection returns None exactly like "no position" does, and the
        # 3x confirmation can't tell them apart (trade #25196 booked a
        # fictional exit during a 42-min outage this way). Reset the None
        # streak so a pre-disconnect count can't instantly confirm closure
        # on reconnect.
        if not self.bridge.is_connected:
            self._sync_none_count = 0
            return

        ib_pos = self.bridge.get_position()
        if ib_pos is None:
            # A None read is NOT proof of closure. Right after a reconnect IB
            # hasn't re-pushed its position cache (trade #567 booked a fictional
            # exit at a 00:29 reconnect); and even without a reconnect the
            # cache can briefly lag a live position (trade #579 phantom-closed
            # 53s after entry). Guard both before counting toward closure — but
            # only query the (costlier) bracket orders once past the reconnect
            # grace, to avoid an extra IB round-trip in the noisy post-reconnect
            # window where that cache is stale too.
            secs = self.bridge.seconds_since_reconnect()
            bracket_live = (bool(self.bridge.get_bracket_orders())
                            if secs >= POST_RECONNECT_GRACE_SEC else False)
            guard = phantom_close_guard(secs, bracket_live)
            if guard != "count":
                if getattr(self, "_sync_none_count", 0):
                    logger.info(
                        f"IB position read None but {guard} "
                        f"(secs_since_reconnect={secs:.0f}, bracket_live={bracket_live}) "
                        f"— treating as desync, not a close; resetting None streak"
                    )
                self._sync_none_count = 0
                return

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
            # Detect the venue T1 + full-close race (trade 622, 2026-07-01): a
            # TP fill closed the whole position while T1 was never booked, so the
            # runner ride was silently collapsed into this single T1-priced
            # close. Surface it — attribution only, P&L is unchanged.
            legs = plan_full_close_legs(
                exit_type=exit_type,
                t1_booked=bool(getattr(self._position, "t1_hit", False)),
                remaining_contracts=contracts,
                total_contracts=int(getattr(self._position, "total_contracts",
                                            contracts)),
                t1_fraction=getattr(getattr(self, "exit_params", None),
                                    "t1_exit_fraction", 0.75),
            )
            if len(legs) > 1:
                _t1_qty = next((q for name, q in legs if name == "t1"), 0)
                _run_qty = next((q for name, q in legs if name == "runner"), 0)
                logger.warning(
                    f"RUNNER COLLAPSED: venue TP fill closed all {contracts} "
                    f"contracts at {fill_price:.2f} before T1 was booked — the "
                    f"{_run_qty}-lot runner never got to ride (booked as "
                    f"{_t1_qty} T1 + {_run_qty} runner at the same price). "
                    f"trade_id={self._trade_id}"
                )
                try:
                    self._position.runner_collapsed = True
                except Exception:
                    pass
            if self._position.direction == "long":
                pnl = (fill_price - self._position.entry_price) * contracts
            else:
                pnl = (self._position.entry_price - fill_price) * contracts
            # Post the Discord exit embed BEFORE mutating position state —
            # the embed reads pre-fill remaining_contracts / realized_pnl_pts
            # from the position. Broker-side bracket fills are the main
            # production exit path and previously never posted an embed
            # (only _handle_exit_action did).
            embed_phase = {"TP": "t1", "SL": "stop"}.get(exit_type, "bracket")
            self._post_trade_exit_embed(
                phase=embed_phase,
                action=SimpleNamespace(
                    exit_price=fill_price,
                    contracts_to_close=contracts,
                    new_stop=None,
                    reason=f"IB bracket {exit_type} fill",
                ),
                pre_position=self._position,
                timestamp=datetime.now(_ET),
            )
            self._position.realized_pnl_pts += pnl
            self._position.remaining_contracts = 0
            self._position.phase = ExitPhase.CLOSED
            now = datetime.now()
            closed_record = self.position_manager.close_position(
                exit_price=fill_price,
                timestamp=now,
                exit_reason=f"IB bracket {exit_type}" if exit_type != "unknown" else "IB bracket fill",
                pattern_type=self._pattern_type,
                entry_time=self._entry_timestamp,
            )
            # Log ONLY the record close_position actually created.
            # Falling back to session.trades[-1] re-logs the PREVIOUS
            # trade when close_position no-ops (no open position in the
            # manager) — that produced the duplicate -37 exit for trade
            # #16872 while the real outcome was the +101 TP fill.
            if closed_record is not None:
                self._log_trade(closed_record, self._current_signal, "exit")
            else:
                logger.warning(
                    f"IB bracket {exit_type} fill @ {fill_price:.2f}: "
                    f"position_manager recorded nothing (no matching open "
                    f"position) — skipping exit log to avoid duplicating "
                    f"the previous trade's record"
                )
            self._position = None
            self._trade_id = None
        else:
            # Position still open at IB — reset the None counter.
            self._sync_none_count = 0
            # On the delayed feed the venue can fill the TP fraction in real
            # time before our bars show T1, leaving a runner. Detect that
            # reduced position and book the partial, keeping the runner.
            ib_volume = ib_pos.get("volume") if isinstance(ib_pos, dict) else None
            if ib_volume is not None and self._position is not None:
                decision = classify_position_sync(
                    local_remaining=self._position.remaining_contracts,
                    ib_volume=ib_volume,
                    t1_booked=bool(getattr(self._position, "t1_hit", False)),
                )
                if decision == "venue_t1_partial":
                    self._reconcile_venue_t1(ib_volume, datetime.now(_ET))

    def _reconcile_venue_t1(self, ib_volume: int, timestamp: datetime) -> None:
        """Book a venue-side T1 fill the delayed feed missed, keeping the runner.

        The OCA bracket lets the exchange fill the TP fraction and auto-reduce
        the stop to the runner. Mirror ExitManager._check_t1 locally: book the
        closed fraction at the real fill, keep the runner, move its stop to
        breakeven, and let the per-bar logic trail it from here.
        """
        pos = self._position
        if pos is None:
            return

        # Book T1 off the TP-order fill confirmation, NOT the (laggy, partial)
        # position-volume read. The old `ib_volume == expected_runner` guard
        # refused to book on an intermediate read (3 when the runner is 1), so
        # T1 stayed unbooked and the runner was swallowed by the later
        # full-close (trade 622, 2026-07-01). Booking off the confirmed TP fill
        # + the INTENDED split preserves the runner regardless of read lag.
        fill_price, exit_type = self.bridge.get_bracket_fill_price(self._trade_id)
        tp_confirmed = (exit_type == "TP" and fill_price > 0)
        should_book, filled, runner_qty = venue_t1_booking_plan(
            tp_confirmed=tp_confirmed,
            t1_booked=bool(getattr(pos, "t1_hit", False)),
            total_contracts=pos.total_contracts,
            t1_fraction=self.exit_params.t1_exit_fraction,
        )
        if not should_book:
            logger.debug(
                f"_sync_position: IB shows {ib_volume} contract(s) but the TP "
                f"fill isn't confirmed yet — waiting to book T1 (will retry)"
            )
            return

        # Mirror the AFTER_T1 transition from ExitManager._check_t1.
        be_buffer = (
            self.exit_params.short_breakeven_buffer_pts
            if pos.direction == "short"
            else self.exit_params.breakeven_buffer_pts
        )
        pnl, new_stop = venue_t1_pnl_and_stop(
            direction=pos.direction,
            entry_price=pos.entry_price,
            fill_price=fill_price,
            filled=filled,
            breakeven_buffer_pts=be_buffer,
            prior_day_low=getattr(pos, "prior_day_low", 0) or 0,
            pdl_buffer_pts=self.exit_params.runner_prior_day_low_buffer_pts,
        )

        pos.realized_pnl_pts += pnl
        pos.remaining_contracts = runner_qty
        pos.t1_hit = True
        pos.phase = ExitPhase.AFTER_T1
        pos.stop_price = new_stop

        # Trail the runner's venue stop up to breakeven.
        try:
            self.bridge.update_stop(
                trade_id=self._trade_id, new_sl=new_stop,
                reason="venue T1 reconcile")
        except Exception as e:
            logger.warning(f"Runner stop update after venue T1 failed: {e!r}")

        action = SimpleNamespace(
            exit_price=fill_price,
            contracts_to_close=filled,
            new_stop=new_stop,
            reason=f"Target 1 hit ({pos.target_1:.2f}) [venue]",
        )
        self._log_partial_exit(action, timestamp)
        self._post_trade_exit_embed(
            phase="t1", action=action, pre_position=pos, timestamp=timestamp)
        logger.info(
            f"VENUE T1 reconciled: {filled} closed @ {fill_price:.2f}, "
            f"runner={runner_qty} held (IB read {ib_volume}), "
            f"new stop={new_stop:.2f}"
        )
        # Confirm the runner actually has a resting protective stop at the
        # venue. If the OCA reduce cancelled the sibling SL instead of reducing
        # it, the runner would be left naked and could be flushed before its
        # own logic exits — surface that so paper validation catches it.
        try:
            if not self.bridge.get_bracket_orders():
                logger.warning(
                    f"RUNNER UNPROTECTED: T1 booked but no live bracket order "
                    f"rests for the {runner_qty}-lot runner (trade_id="
                    f"{self._trade_id}) — venue stop may have been cancelled by "
                    f"the OCA reduce; runner is exposed until re-armed"
                )
        except Exception:
            pass

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

        # Mirror to position_manager.session so subsequent close_position
        # calls credit the trade to daily_pnl / trade_count / winners /
        # losers. Without this, _sync_position's bracket auto-fill path
        # calls close_position() which short-circuits on the
        # `session.active_position is None` guard, and the win/loss is
        # logged to trades.jsonl but never reaches the dashboard.
        if self.position_manager.session is not None:
            self.position_manager.session.active_position = self._position
            if direction == "long":
                self.position_manager.session.active_long = self._position
            else:
                self.position_manager.session.active_short = self._position

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

                # FB entry path classification (shared with the Discord embed
                # via _classify_fb_entry_path so the two never disagree)
                record["fb_entry_path"] = self._classify_fb_entry_path()

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

                # Augment with the most recent bar and the exit fill price.
                # The IB bracket OCO can fill mid-bar BEFORE the strategy's
                # bar-update loop absorbs that bar's high/low into the
                # position state — so the recorded MFE undershoots. Fold the
                # latest bar's high/low and the actual exit price in, and
                # floor the favorable side at the realized exit.
                exit_price = getattr(trade_record, "exit_price", None)
                recent_high = None
                recent_low = None
                if self._df is not None and len(self._df) > 0:
                    try:
                        recent_high = float(self._df["high"].iat[-1])
                        recent_low = float(self._df["low"].iat[-1])
                    except Exception:
                        pass
                record["mfe_pts"], record["mae_pts"] = compute_excursion_pts(
                    direction=direction,
                    entry_price=entry_price,
                    highest=highest,
                    lowest=lowest,
                    exit_price=exit_price,
                    recent_high=recent_high,
                    recent_low=recent_low,
                )

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

            # Archive full session bars — from the uncapped accumulator, not
            # the 400-bar live window. Fall back to self._df only if the
            # accumulator is empty (e.g. a recovery path that bypassed it).
            session_df = build_session_bars_df(self._session_bars)
            if len(session_df) == 0 and self._df is not None and len(self._df) > 0:
                session_df = self._df
            if len(session_df) > 0:
                bars_path = sessions_dir / f"{self._session_date}_bars.parquet"
                session_df.to_parquet(bars_path)
                logger.info(f"Session bars archived: {len(session_df)} bars -> {bars_path}")

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
            # Start a fresh full-session accumulator for the new session.
            self._session_bars = {}

            # Reset for new session
            old_date = self._session_date
            self._session_date = trading_date
            # Preserve open position reference before reset
            open_pos = self._position if (self._position is not None and self._position.is_open) else None
            # Survived-sessions counter accounting.
            #
            # New default (eod_flatten_enabled=False): EOD flatten is off, so
            # ANY phase that's still open at rollover counts toward the
            # multi_session_runner_max_days safety cap. Without this bump,
            # INITIAL/AFTER_T1 positions could ride forever once the EOD
            # flatten was disabled.
            #
            # Legacy (eod_flatten_enabled=True): only AFTER_T2 runners with
            # multi_session_runner=True survive EOD, so only those need
            # counter bumps.
            eod_flatten_enabled = getattr(
                self.exit_params, "eod_flatten_enabled", False
            )
            multi_session_enabled = getattr(
                self.exit_params, "multi_session_runner", False
            )
            should_bump_counter = open_pos is not None and (
                (not eod_flatten_enabled) or
                (open_pos.phase == ExitPhase.AFTER_T2 and multi_session_enabled)
            )
            if should_bump_counter:
                self._runner_sessions_held += 1
                logger.info(
                    f"EOD HOLD ROLLOVER: now on session "
                    f"{self._runner_sessions_held}/"
                    f"{getattr(self.exit_params, 'multi_session_runner_max_days', 5)} "
                    f"(phase={open_pos.phase.name}, "
                    f"entry={open_pos.entry_price:.2f}, stop={open_pos.stop_price:.2f}, "
                    f"contracts={open_pos.remaining_contracts}, "
                    f"pattern={self._pattern_type})"
                )
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

            # Reload the LLM-extracted Mancini plan for the new trading_date.
            # Without this, a long-running bot that never restarts would keep
            # using the prior session's plan after Globex rollover at 18:00 ET.
            self._load_mancini_llm_plan()

            logger.info(f"New session {trading_date}: daily PnL reset, "
                        f"trade count reset, levels re-initialized, "
                        f"pattern state restored, LLM plan reloaded")

    # ── EOD and session management ───────────────────────────────────

    def _check_eod(self, bar: dict) -> None:
        """Check for EOD: update runner trail or flatten non-runners.

        Mancini's method (2025-10-12: "still holding my 10% long runner from
        the Tuesday noon 6754 Failed Breakdown"): the 10% AFTER_T2 runner
        carries across sessions to catch trend moves.

        Behavior selector — controlled by exit_params.eod_flatten_enabled
        (master switch) and exit_params.multi_session_runner (legacy mode).

        Default (eod_flatten_enabled=False):
          - INITIAL / AFTER_T1 / AFTER_T2 all HOLD across EOD via their
            existing stops (initial stop / BE-3 / structure trail).
          - update_prior_day_low() is invoked so the runner trail ratchets
            for any AFTER_T1/AFTER_T2 position (no-op for INITIAL).
          - multi_session_runner_max_days safety cap still applies to ALL
            phases — once sessions_held >= max_days, the next EOD flattens.

        Legacy (eod_flatten_enabled=True), driven by multi_session_runner:
          - INITIAL phase: always flatten at EOD.
          - AFTER_T1 phase (still 25% — pre-T2): always flatten at EOD,
            even with multi_session_runner=True. Only the 10% post-T2
            slice is allowed to hold cross-session.
          - AFTER_T2 phase (10% runner):
              * multi_session_runner=False: flatten at EOD.
              * multi_session_runner=True: update structural trail, leave
                position open. The runner persists until its trailing stop
                is hit OR until multi_session_runner_max_days sessions have
                elapsed (safety cap).

        At session break (17:00-18:00), archive and stop the bot.
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
                    phase = self._position.phase
                    eod_flatten_enabled = getattr(
                        self.exit_params, "eod_flatten_enabled", False
                    )
                    multi_session_enabled = getattr(
                        self.exit_params, "multi_session_runner", False
                    )
                    max_days = getattr(
                        self.exit_params, "multi_session_runner_max_days", 5
                    )

                    # Determine whether to flatten or hold across EOD.
                    #
                    # New default (eod_flatten_enabled=False): hold ALL phases
                    # across EOD, capped only by multi_session_runner_max_days.
                    #
                    # Legacy (eod_flatten_enabled=True): only AFTER_T2 +
                    # multi_session_runner=True holds; everything else flattens.
                    if not eod_flatten_enabled:
                        hold_across_eod = self._runner_sessions_held < max_days
                    else:
                        hold_across_eod = (
                            multi_session_enabled
                            and phase == ExitPhase.AFTER_T2
                            and self._runner_sessions_held < max_days
                        )

                    if hold_across_eod:
                        # Update structural trail under today's session low
                        # (no-op for INITIAL — update_prior_day_low ignores
                        # pre-T1 positions and they ride their initial stop).
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
                                f"EOD HOLD (day {self._runner_sessions_held + 1}/"
                                f"{max_days}, phase={phase.name}): "
                                f"stop={self._position.stop_price:.2f} "
                                f"(daily low={daily_low:.2f}, "
                                f"{self._position.remaining_contracts} contracts, "
                                f"pattern={self._pattern_type})"
                            )
                    else:
                        # Force-flatten path. Two sub-cases:
                        #   1. eod_flatten_enabled=False with max-days cap hit
                        #      (applies to any phase)
                        #   2. eod_flatten_enabled=True legacy semantics
                        #      (INITIAL/AFTER_T1 always; AFTER_T2 when cap or
                        #       multi_session_runner=False)
                        if self._runner_sessions_held >= max_days:
                            flatten_reason = "eod_flatten_max_days"
                            log_label = (
                                f"EOD MAX-DAYS CAP HIT "
                                f"({self._runner_sessions_held}/{max_days}, "
                                f"phase={phase.name}): force-flattening"
                            )
                        else:
                            flatten_reason = "eod_flatten"
                            log_label = (
                                f"EOD flatten sent ({session.eod_flatten_time.strftime('%H:%M')} ET, "
                                f"phase={phase.name})"
                            )
                        self.bridge.flatten(reason=flatten_reason)
                        logger.info(log_label)
                        exit_price = float(bar.get("close", 0))
                        # Compute PnL before closing
                        if self._position.direction == "short":
                            self._position.realized_pnl_pts += (self._position.entry_price - exit_price) * self._position.remaining_contracts
                        else:
                            self._position.realized_pnl_pts += (exit_price - self._position.entry_price) * self._position.remaining_contracts
                        self._position.remaining_contracts = 0
                        self._position.phase = ExitPhase.CLOSED
                        exit_reason_label = (
                            "EOD flatten (max-days cap)"
                            if flatten_reason == "eod_flatten_max_days"
                            else "EOD flatten"
                        )
                        trade_rec = self.position_manager.close_position(
                            exit_price=exit_price,
                            timestamp=timestamp,
                            exit_reason=exit_reason_label,
                            pattern_type=self._pattern_type,
                            entry_time=self._entry_timestamp,
                        )
                        if trade_rec:
                            self._log_trade(trade_rec, self._current_signal, "exit")
                        self._position = None
                        self._trade_id = None
                        self._runner_sessions_held = 0

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
                    # Heads-up Discord alert for actionable shadow shorts —
                    # once per distinct setup, never an order. Isolated so an
                    # alert failure never disrupts logging / phantom tracking.
                    try:
                        self._maybe_alert_short(event)
                    except Exception:
                        pass
                    # Create phantom tracker for signals with entry/stop/target
                    if event.get("entry_price") and event.get("stop_price"):
                        direction = event.get("direction") or (
                            "short" if "short" in event.get("feature", "").lower() else "long"
                        )
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

    def _maybe_alert_short(self, event: dict) -> None:
        """Post a Discord heads-up when a short fires AT A LEVEL MANCINI CALLED.

        Heads-up only — the bot is long-only and places no short order. Gated to
        shorts that line up with one of Mancini's planned short setups (e.g. his
        7399 / 7530 breakdowns), then deduped by that called level so each fires
        ONCE per session — not on every mechanical flush. Without a plan-short
        match the shadow short is logged but never posted. Best-effort: any
        failure is logged and swallowed so it never disrupts the bar loop.
        """
        if not self._short_alerts_enabled:
            return
        try:
            from live.trade_notifications import (
                is_short_alert_event, plan_short_match,
                build_short_alert_embed, post_payload, get_webhook_url,
            )
            if not is_short_alert_event(event):
                return
            plan = getattr(self, "_mancini_llm_plan", None)
            # Match against a price Mancini actually called as a short. Try the
            # structural level first, then the (chasing) entry.
            price = event.get("level_price") or event.get("entry_price")
            match = plan_short_match(plan, price)
            if match is None and event.get("entry_price") is not None:
                match = plan_short_match(plan, event["entry_price"])
            if match is None:
                return  # not a Mancini-called short — don't post (still logged)
            # Dedup by the CALLED level, so 7399 alerts once even as price flushes.
            key = f"short@{round(float(match.level_price))}"
            if key in self._short_alert_keys:
                return
            self._short_alert_keys.add(key)
            webhook = get_webhook_url()
            if not webhook:
                return
            symbol = getattr(self.contract, "symbol", "MES")
            embed = build_short_alert_embed(event, symbol=symbol, plan=plan)
            ok, info = post_payload(
                {"username": "Mancini Bot", "embeds": [embed]}, webhook)
            if ok:
                logger.info(f"Short heads-up posted: {event.get('signal_type')} "
                            f"@ {event.get('entry_price')} "
                            f"(Mancini short {match.level_price:g}) ({info})")
            else:
                logger.warning(f"Short heads-up post failed: {info}")
        except Exception as e:
            logger.warning(f"Short heads-up alert error: {e}")

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
