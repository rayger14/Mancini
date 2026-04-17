"""Run a single feature backtest and write results to a file.

Usage:
    python3 backtest/run_single_feature.py <feature_name> <output_file>
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.feature_comparison import (
    load_data, build_sessions, run_backtest, compute_stats, make_params, FEATURES
)

feature_name = sys.argv[1]
output_file = sys.argv[2]

# Find feature config
feat_config = None
for name, overrides in FEATURES:
    if name == feature_name:
        feat_config = overrides
        break

if feat_config is None:
    print(f"Unknown feature: {feature_name}", file=sys.stderr)
    sys.exit(1)

print(f"Loading data for {feature_name}...", flush=True)
df = load_data()
sessions = build_sessions(df)
print(f"Running {feature_name} ({len(sessions)} sessions)...", flush=True)

params = make_params(feat_config)
trades = run_backtest(sessions, params)
stats = compute_stats(trades)
stats["name"] = feature_name

print(f"{feature_name}: {stats['trades']}T, {stats['wr']:.1f}% WR, PF={stats['pf']:.2f}, "
      f"PnL={stats['pnl']:+,.1f}, MaxDD={stats['maxdd']:.1f}", flush=True)

with open(output_file, "w") as f:
    json.dump(stats, f)
