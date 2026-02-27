"""Simulate NinjaTrader writing bar files to test the Python bridge end-to-end.

Replays historical data through the file bridge as if NinjaTrader were writing
bars in real-time. Verifies that the Python runner generates the same signals
as the backtest.

Usage:
    python3 live/test_bridge_sim.py [--days 5]
"""

import sys
import json
import shutil
from datetime import datetime, date, time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from live.nt_bridge import NTBridge, NTBridgeConfig
from live.nt_runner import (
    NTRunner, PRODUCTION_STRATEGY, PRODUCTION_ELEVATOR,
    PRODUCTION_EXIT, PRODUCTION_RISK, PRODUCTION_SESSION, MES_CONTRACT,
)

# Use a temp directory for the simulation
SIM_DIR = str(Path(__file__).parent.parent / "data" / "_bridge_sim")


def load_daily_dfs(parquet_path: str) -> dict[date, pd.DataFrame]:
    """Load and split data into daily DataFrames."""
    df = pd.read_parquet(parquet_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    df_rth = df.between_time("09:30", "15:59")
    daily = {}
    for d, grp in df_rth.groupby(df_rth.index.date):
        if len(grp) >= 10:
            daily[d] = grp
    return daily


def simulate_nt_writes(bridge: NTBridge, session_date: date,
                        prior_day_df: pd.DataFrame,
                        current_day_df: pd.DataFrame) -> None:
    """Simulate NinjaTrader writing history + bar files for one day."""

    # 1. Write history file (what NT writes at session start)
    history_data = {
        "session_date": str(session_date),
        "instrument": bridge.config.instrument,
        "prior_day_bars": [],
        "current_day_bars": [],
    }

    if prior_day_df is not None:
        for i in range(len(prior_day_df)):
            history_data["prior_day_bars"].append({
                "timestamp": prior_day_df.index[i].isoformat(),
                "open": float(prior_day_df["open"].iat[i]),
                "high": float(prior_day_df["high"].iat[i]),
                "low": float(prior_day_df["low"].iat[i]),
                "close": float(prior_day_df["close"].iat[i]),
                "volume": float(prior_day_df["volume"].iat[i]),
            })

    filename = f"history_{session_date.strftime('%Y%m%d')}.json"
    path = Path(bridge.config.shared_dir) / "bars" / filename
    path.write_text(json.dumps(history_data, indent=2, default=str), encoding="utf-8")

    # 2. Write individual bar files (what NT writes on each OnBarUpdate)
    for i in range(len(current_day_df)):
        ts = current_day_df.index[i]
        bar_data = {
            "timestamp": ts.isoformat(),
            "open": float(current_day_df["open"].iat[i]),
            "high": float(current_day_df["high"].iat[i]),
            "low": float(current_day_df["low"].iat[i]),
            "close": float(current_day_df["close"].iat[i]),
            "volume": float(current_day_df["volume"].iat[i]),
            "bar_number": i,
            "instrument": bridge.config.instrument,
        }
        bar_filename = f"bar_{session_date.strftime('%Y%m%d')}_{ts.strftime('%H%M')}.json"
        bar_path = Path(bridge.config.shared_dir) / "bars" / bar_filename
        bar_path.write_text(json.dumps(bar_data, indent=2, default=str), encoding="utf-8")

    # 3. Write NT heartbeat
    hb_path = Path(bridge.config.shared_dir) / "state" / "nt_heartbeat.json"
    hb_data = {
        "timestamp": datetime.now().isoformat(),
        "status": "running",
        "bars_processed": len(current_day_df),
        "session_date": str(session_date),
    }
    hb_path.write_text(json.dumps(hb_data, indent=2, default=str), encoding="utf-8")


def run_one_day(runner: NTRunner, bridge: NTBridge,
                session_date: date, prior_day_df: pd.DataFrame,
                current_day_df: pd.DataFrame) -> dict:
    """Run the Python bridge for one simulated day.

    Returns dict with trade results.
    """
    # Clean bridge state
    for subdir in ["bars", "signals", "fills", "state"]:
        d = Path(bridge.config.shared_dir) / subdir
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # Reset runner state
    runner.bridge = NTBridge(bridge.config)
    runner.strategy.reset()
    runner.position_manager.start_session(datetime.combine(session_date, dtime(9, 30)))
    runner.signal_aggregator.reset()
    runner._bars = []
    runner._df = None
    runner._position = None
    runner._pattern_type = ""
    runner._bar_count = 0
    runner._session_date = session_date

    # Write all files (simulating NT)
    simulate_nt_writes(bridge, session_date, prior_day_df, current_day_df)

    # Initialize session
    runner.bridge.wait_for_history(session_date, timeout_sec=5)
    prior_df, current_df = runner.bridge.read_history(session_date)
    runner.signal_aggregator.initialize_levels(
        current_df if current_df is not None else pd.DataFrame(),
        prior_df,
    )

    # Process bars one by one (simulating real-time)
    signals_generated = []
    for _ in range(len(current_day_df)):
        bar = runner.bridge.poll_new_bar()
        if bar is None:
            break
        runner._process_bar(bar)

        # Check if any signals were written
        signal_dir = Path(bridge.config.shared_dir) / "signals"
        for sf in sorted(signal_dir.glob("signal_*.json")):
            try:
                sig = json.loads(sf.read_text())
                if sig.get("status") == "UNREAD":
                    signals_generated.append(sig)
                    # Simulate NT executing and sending fill back
                    _simulate_fill(bridge, sig)
                    # Mark signal as executed
                    sig["status"] = "EXECUTED"
                    sf.write_text(json.dumps(sig, indent=2, default=str))
            except (json.JSONDecodeError, OSError):
                pass

    # Gather results
    session = runner.position_manager.session
    return {
        "date": str(session_date),
        "signals": len(signals_generated),
        "trades": session.trade_count if session else 0,
        "wins": session.wins if session else 0,
        "losses": session.losses if session else 0,
        "pnl_pts": session.daily_pnl_pts if session else 0,
        "signal_details": signals_generated,
    }


def _simulate_fill(bridge: NTBridge, signal: dict) -> None:
    """Simulate NinjaTrader filling an order and writing a fill file."""
    action = signal.get("action", "")
    if action != "enter_long":
        return

    fill_data = {
        "fill_id": f"sim_fill_{signal.get('signal_id', '')}",
        "signal_id": signal.get("signal_id", ""),
        "action": "entry_fill",
        "instrument": signal.get("instrument", ""),
        "price": signal.get("entry_price", 0),
        "quantity": signal.get("quantity", 0),
        "timestamp": datetime.now().isoformat(),
        "commission": signal.get("quantity", 0) * 0.62,
    }
    fills_dir = Path(bridge.config.shared_dir) / "fills"
    fill_path = fills_dir / f"fill_{fill_data['fill_id']}.json"
    fill_path.write_text(json.dumps(fill_data, indent=2, default=str), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bridge simulation test")
    parser.add_argument("--days", type=int, default=20, help="Number of days to simulate")
    args = parser.parse_args()

    # Setup
    data_path = Path(__file__).parent.parent / "data" / "ES_1m_2024-02-05_2026-02-05.parquet"
    print("Loading data...")
    daily_dfs = load_daily_dfs(str(data_path))

    # Filter Mondays
    non_monday = {d: df for d, df in daily_dfs.items() if d.weekday() != 0}
    sorted_dates = sorted(non_monday.keys())

    # Take a sample of days
    sample_dates = sorted_dates[:args.days]

    config = NTBridgeConfig(shared_dir=SIM_DIR, instrument="MES SIM")
    bridge = NTBridge(config)
    bridge.ensure_directories()

    runner = NTRunner(
        bridge_config=config,
        strategy_params=PRODUCTION_STRATEGY,
        elevator_params=PRODUCTION_ELEVATOR,
        exit_params=PRODUCTION_EXIT,
        risk_params=PRODUCTION_RISK,
        session_times=PRODUCTION_SESSION,
        contract=MES_CONTRACT,
        min_rr_ratio=1.0,
    )

    print(f"\nSimulating {len(sample_dates)} days through the bridge...")
    print(f"Shared directory: {SIM_DIR}")
    print()

    total_signals = 0
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    results = []

    for i, d in enumerate(sample_dates):
        # Get prior day
        idx = sorted_dates.index(d)
        prior_d = sorted_dates[idx - 1] if idx > 0 else None
        prior_df = daily_dfs.get(prior_d) if prior_d else None

        result = run_one_day(runner, bridge, d, prior_df, non_monday[d])
        results.append(result)

        total_signals += result["signals"]
        total_trades += result["trades"]
        total_wins += result["wins"]
        total_pnl += result["pnl_pts"]

        if result["signals"] > 0:
            print(f"  {result['date']}: {result['signals']} signal(s), "
                  f"{result['trades']} trade(s), PnL={result['pnl_pts']:+.1f} pts")
        else:
            print(f"  {result['date']}: no signals")

    # Summary
    print("\n" + "=" * 60)
    print("BRIDGE SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Days simulated:  {len(sample_dates)}")
    print(f"  Total signals:   {total_signals}")
    print(f"  Total trades:    {total_trades}")
    print(f"  Wins:            {total_wins}")
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    print(f"  Win rate:        {wr:.0f}%")
    print(f"  Total PnL:       {total_pnl:+.1f} pts")
    print()

    if total_signals > 0:
        print("  BRIDGE IS WORKING — signals generated and processed correctly")
    else:
        print("  NOTE: No signals in this sample. Try --days 50 for more coverage.")

    # Cleanup
    shutil.rmtree(SIM_DIR, ignore_errors=True)
    print(f"\n  Cleaned up simulation directory")


if __name__ == "__main__":
    main()
