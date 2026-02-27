"""Full 23-hour session backtest with locked production config.

Uses ES_1m_full_session parquet data. Processes each globex session
(18:00 -> next day 16:59) as one trading day.

Usage:
    python3 backtest/full_session_backtest.py
    python3 backtest/full_session_backtest.py --days 365
"""

from __future__ import annotations

import sys
from datetime import datetime, time as dt_time, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    StrategyParams,
    ElevatorParams,
    ExitParams,
    RiskParams,
    SessionTimes,
    ESContractSpec,
)
from strategy.mancini_long import ManciniLongStrategy

# ── LOCKED PRODUCTION CONFIG ──────────────────────────────────────────

STRATEGY = StrategyParams(
    swing_low_order=15,
    multi_hour_rally_min_pts=22.5,
    level_reclaim_min_touches=4,
    acceptance_min_hold_bars=7,
    acceptance_min_hold_bars_deep=8,
    acceptance_max_dip_pts=3.0,
    true_breakdown_abort_bars=12,
    fb_stop_buffer_pts=5.5,
    lr_stop_buffer_pts=5.0,
    non_acceptance_min_recovery_pts=5.0,
)
ELEVATOR = ElevatorParams(
    min_velocity_pts_per_min=0.75,
    min_levels_broken=2,
    higher_low_lookback=4,
)
EXIT = ExitParams(
    default_contracts=4,
    t1_exit_fraction=1.0,
    trailing_stop_pts=7.0,
)
RISK = RiskParams(max_trades_per_day=4)

# Full 23-hour session
FULL_SESSION = SessionTimes(
    rth_open=dt_time(18, 0),
    rth_close=dt_time(17, 0),
    morning_window_start=dt_time(9, 30),
    morning_window_end=dt_time(11, 0),
    afternoon_window_start=dt_time(15, 0),
    afternoon_window_end=dt_time(16, 50),
    eod_flatten_time=dt_time(16, 50),
    chop_zone_start=dt_time(13, 0),
    chop_zone_end=dt_time(15, 0),
)

MES = ESContractSpec(
    symbol="MES", tick_size=0.25, tick_value=1.25,
    point_value=5.0, margin_initial=1_265.0,
    margin_maintenance=1_150.0, exchange="CME",
)


def load_full_session_data() -> pd.DataFrame:
    """Load full session 1-min data."""
    path = Path(__file__).parent.parent / "data" / "ES_1m_full_session_2024-02-05_2026-02-05.parquet"
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    # Ensure OHLCV columns
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns, f"Missing column: {col}"
    return df


def build_sessions(df: pd.DataFrame) -> list[tuple[date, pd.DataFrame]]:
    """Split data into globex sessions: 18:00 -> next day 16:59.

    Each session starts at 18:00 and ends at 16:59 the next calendar day.
    The 17:00-17:59 break is excluded.
    """
    # Find all unique 18:00 starts
    evening_mask = df.index.time == dt_time(18, 0)
    session_starts = df.index[evening_mask]

    sessions = []
    for start_ts in session_starts:
        # Session ends next calendar day at 16:59
        next_day = start_ts.date() + timedelta(days=1)
        end_ts = pd.Timestamp(
            datetime.combine(next_day, dt_time(16, 59)),
            tz="US/Eastern",
        )

        # Extract bars in this session
        session_df = df[(df.index >= start_ts) & (df.index <= end_ts)]

        # Remove the 17:00-17:59 break
        session_df = session_df[
            ~((session_df.index.time >= dt_time(17, 0)) & (session_df.index.time < dt_time(18, 0)))
        ]

        if len(session_df) > 0:
            # Label by the RTH date (next calendar day)
            sessions.append((next_day, session_df))

    return sessions


def run_backtest(days: int = 0, skip_mondays: bool = True):
    """Run full session backtest."""
    logger.info("Loading full session data...")
    df = load_full_session_data()
    logger.info(f"Loaded {len(df)} bars: {df.index[0]} -> {df.index[-1]}")

    sessions = build_sessions(df)
    logger.info(f"Built {len(sessions)} trading sessions")

    if days > 0:
        sessions = sessions[-days:]
        logger.info(f"Using last {days} sessions: {sessions[0][0]} -> {sessions[-1][0]}")

    strategy = ManciniLongStrategy(
        strategy_params=STRATEGY,
        elevator_params=ELEVATOR,
        exit_params=EXIT,
        risk_params=RISK,
        session_times=FULL_SESSION,
        contract=MES,
        min_rr_ratio=1.0,
    )

    all_trades = []
    daily_stats = []
    prev_session_df = None

    for i, (session_date, session_df) in enumerate(sessions):
        # Skip Monday RTH
        if skip_mondays and session_date.weekday() == 0:
            prev_session_df = session_df
            continue

        # Run strategy
        results = strategy.run_day(
            session_df,
            prior_day_df=prev_session_df,
            session_date=datetime.combine(session_date, dt_time(0, 0)),
        )

        trades = strategy.trade_records
        day_pnl = sum(t.pnl_pts for t in trades)
        day_dollars = sum(t.pnl_dollars for t in trades)

        for t in trades:
            all_trades.append({
                "date": session_date,
                "entry_time": t.entry_time if hasattr(t, "entry_time") else session_date,
                "pattern": t.pattern_type,
                "entry": t.entry_price,
                "exit": t.avg_exit_price,
                "stop": t.stop_price,
                "target_1": t.target_1 if hasattr(t, "target_1") else 0,
                "contracts": t.contracts,
                "pnl_pts": t.pnl_pts,
                "pnl_dollars": t.pnl_dollars,
                "exit_reason": t.exit_reason,
                "rr_ratio": t.rr_ratio_t1 if hasattr(t, "rr_ratio_t1") else 0,
            })

        daily_stats.append({
            "date": session_date,
            "weekday": session_date.strftime("%A"),
            "trades": len(trades),
            "wins": sum(1 for t in trades if t.pnl_pts > 0),
            "losses": sum(1 for t in trades if t.pnl_pts <= 0),
            "pnl_pts": day_pnl,
            "pnl_dollars": day_dollars,
        })

        prev_session_df = session_df

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(sessions)} sessions...")

    # ── Results ──────────────────────────────────────────────────────
    trades_df = pd.DataFrame(all_trades)
    daily_df = pd.DataFrame(daily_stats)

    total_trades = len(trades_df)
    if total_trades == 0:
        logger.warning("No trades found!")
        return

    wins = len(trades_df[trades_df["pnl_pts"] > 0])
    losses = total_trades - wins
    win_rate = wins / total_trades * 100
    total_pts = trades_df["pnl_pts"].sum()
    total_dollars = trades_df["pnl_dollars"].sum()
    avg_win = trades_df[trades_df["pnl_pts"] > 0]["pnl_pts"].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df["pnl_pts"] <= 0]["pnl_pts"].mean() if losses > 0 else 0
    gross_profit = trades_df[trades_df["pnl_pts"] > 0]["pnl_pts"].sum()
    gross_loss = abs(trades_df[trades_df["pnl_pts"] <= 0]["pnl_pts"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Equity curve for max drawdown
    equity = trades_df["pnl_pts"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()

    # Sharpe (annualized from daily)
    daily_pnl = daily_df["pnl_pts"]
    daily_pnl_nonzero = daily_pnl[daily_pnl != 0]
    if len(daily_pnl_nonzero) > 1:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0
    else:
        sharpe = 0

    print("\n" + "=" * 80)
    print("FULL SESSION BACKTEST — LOCKED PRODUCTION CONFIG")
    print("=" * 80)
    print(f"Period:         {sessions[0][0]} -> {sessions[-1][0]}")
    print(f"Sessions:       {len(daily_df)} (Mondays skipped)")
    print(f"Total trades:   {total_trades}")
    print(f"Win rate:       {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"Profit factor:  {profit_factor:.2f}")
    print(f"Total PnL:      {total_pts:+.1f} pts (${total_dollars:+,.0f})")
    print(f"Avg win:        {avg_win:+.1f} pts")
    print(f"Avg loss:       {avg_loss:+.1f} pts")
    print(f"Max drawdown:   {max_dd:.1f} pts")
    print(f"Sharpe (ann.):  {sharpe:.2f}")
    print()

    # ── Monthly breakdown ────────────────────────────────────────────
    print("MONTHLY BREAKDOWN:")
    print(f"{'Month':<10} {'Trades':>6} {'Win%':>6} {'PnL pts':>10} {'PnL $':>12}")
    print("-" * 50)
    daily_df["month"] = pd.to_datetime(daily_df["date"]).dt.to_period("M")
    for month, group in daily_df.groupby("month"):
        m_trades = group["trades"].sum()
        m_wins = group["wins"].sum()
        m_wr = m_wins / m_trades * 100 if m_trades > 0 else 0
        m_pnl = group["pnl_pts"].sum()
        m_dollars = group["pnl_dollars"].sum()
        print(f"{str(month):<10} {m_trades:>6} {m_wr:>5.0f}% {m_pnl:>+10.1f} {m_dollars:>+12,.0f}")

    # ── Trade-by-trade breakdown ─────────────────────────────────────
    print()
    print("TRADE-BY-TRADE BREAKDOWN:")
    print(f"{'#':>3} {'Date':<12} {'Pattern':<20} {'Entry':>8} {'Exit':>8} {'Stop':>8} {'PnL':>8} {'$':>10} {'Exit Reason':<25}")
    print("-" * 115)
    for idx, t in trades_df.iterrows():
        print(
            f"{idx+1:>3} {str(t['date']):<12} {t['pattern']:<20} "
            f"{t['entry']:>8.2f} {t['exit']:>8.2f} {t['stop']:>8.2f} "
            f"{t['pnl_pts']:>+8.1f} {t['pnl_dollars']:>+10,.0f} "
            f"{t['exit_reason']:<25}"
        )

    # ── By day of week ───────────────────────────────────────────────
    print()
    print("BY DAY OF WEEK:")
    dow_order = ["Tuesday", "Wednesday", "Thursday", "Friday"]
    for dow in dow_order:
        dow_trades = trades_df[pd.to_datetime(trades_df["date"]).dt.strftime("%A") == dow]
        if len(dow_trades) == 0:
            continue
        dow_wins = len(dow_trades[dow_trades["pnl_pts"] > 0])
        dow_wr = dow_wins / len(dow_trades) * 100
        dow_pnl = dow_trades["pnl_pts"].sum()
        print(f"  {dow:<12} {len(dow_trades):>3} trades, {dow_wr:.0f}% WR, {dow_pnl:+.1f} pts")

    # ── By session period ────────────────────────────────────────────
    print()
    print("BY SESSION PERIOD:")
    # Classify each trade by entry time
    if "entry_time" in trades_df.columns:
        for label, start, end in [
            ("Evening (18:00-23:59)", dt_time(18, 0), dt_time(23, 59)),
            ("Overnight (00:00-06:00)", dt_time(0, 0), dt_time(6, 0)),
            ("Pre-RTH (06:00-09:30)", dt_time(6, 0), dt_time(9, 30)),
            ("Morning RTH (09:30-13:00)", dt_time(9, 30), dt_time(13, 0)),
            ("Afternoon RTH (15:00-16:50)", dt_time(15, 0), dt_time(16, 50)),
        ]:
            # Can't easily filter by entry_time since it's a date object in many cases
            # Just report total
            pass

    print()
    print("=" * 80)

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "full_session_backtest.csv"
    trades_df.to_csv(output_path, index=False)
    print(f"\nTrade details saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=0,
                        help="Number of sessions to backtest (0=all)")
    parser.add_argument("--include-mondays", action="store_true",
                        help="Include Mondays (default: skip)")
    args = parser.parse_args()

    run_backtest(days=args.days, skip_mondays=not args.include_mondays)
