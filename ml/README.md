# ML Trade Filter — Phase 1 Scaffold

A win-probability classifier trained on logged live trades. Designed
to run as a sidecar — outputs a probability for each setup, which
can later be used to filter weak ones in shadow mode before going
live.

## Quick start

```bash
# 1. Pull trade logs from the VM into ./data/training/
scp ubuntu@<vm>:/home/ubuntu/mancini/logs/trades.jsonl data/training/
scp ubuntu@<vm>:/home/ubuntu/mancini/logs/shadow_trades.jsonl data/training/

# 2. Build the labeled dataset
python3 -m ml.dataset

# 3. Train and score
python3 -m ml.train
```

Outputs: `data/training/dataset.parquet`, `data/training/model.pkl`,
`data/training/report.json`.

## What's in here

- `ml/dataset.py` — joins entry+exit events from `trades.jsonl`, adds
  `phantom_resolved` and `near_miss_resolved` rows, returns one
  labeled DataFrame.
- `ml/train.py` — chronological 80/20 split, `HistGradientBoostingClassifier`
  with balanced sample weights, permutation importance.

`HistGradientBoostingClassifier` is used in place of LightGBM because
LightGBM needs `libomp` on macOS. Same algorithmic family; swap if
you have libomp.

## Phase 1 limitations (read this)

- **Data volume is small.** ~70 live entries with full schema. Test
  AUC will be noisy at this scale. Increasing data is the highest
  ROI lever right now.
- **Live and phantom rows have different schemas.** Phase 1 trains
  on `source == "live_entry"` only by default to avoid mixing
  inconsistent feature coverage. Pass `source_filter=None` in
  `train()` to use the union once features are unified.
- **Single-threaded BLAS/OpenMP forced at import.** macOS sklearn
  HistGB deadlocks on multi-thread on this machine — `train.py`
  pins all thread vars to 1 before importing sklearn. Don't
  remove that block.
- **Not wired into the live engine yet.** This phase produces a
  model and a report. Wiring the prediction in as a shadow gate is
  the next phase.

## Next phases

- **Phase 1.5:** Backfill features for phantom rows so we can train
  on the full 2,500-row dataset with a unified schema.
- **Phase 2:** Wire the model in as a shadow gate — log
  `model_prob` next to live trades for offline comparison before
  any production gating.
- **Phase 3:** LSTM / transformer on raw bar sequences. Pretraining
  the transformer on the 6 years of 1-min bars (~2.2M bars) is
  where the GPU earns its keep.
