"""Train a gradient-boosting win-probability classifier on the labeled
dataset.

Usage:
    python3 -m ml.train                                    # default paths
    python3 -m ml.train --dataset path/to/dataset.parquet  # custom

Splits chronologically (oldest 80% train, newest 20% test) to mimic
walk-forward evaluation. Prints AUC, accuracy, confusion matrix, and
permutation importance to stdout. Writes the fitted model + a JSON report.

We use scikit-learn's HistGradientBoostingClassifier as the default model
because it has no system-level deps (LightGBM needs libomp on macOS).
The algorithmic family is the same — swap to lightgbm.LGBMClassifier
once libomp is available if you want.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

# macOS sklearn HistGB deadlocks badly without proper libomp. Force
# single-threaded BLAS/OpenMP before importing sklearn — must happen at
# import time, not in main(), or the threadpool is already initialized.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Numeric features fed directly to the model.
NUMERIC_FEATURES = [
    "ts_hour",
    "ts_minute",
    "ts_dow",
    "bar_count",
    "session_high",
    "session_low",
    "session_range",
    "last_price",
    "pos_in_range",
    "entry_price",
    "stop_price",
    "target_1",
    "rr_ratio",
    "level_price",
    "ema_slope",
    "nearby_count",
    "nearest_level_distance",
    "nearest_level_touches",
    "high_since",
    "low_since",
]

# Low-cardinality strings — ordinal-encoded, then HistGB treats them as
# categorical features (handles NaN as a separate level).
CATEGORICAL_FEATURES = [
    "session_window",
    "direction",
    "pattern_type",
    "signal_type",
    "level_type",
    "regime_direction",
    "nearest_level_type",
    "source",
    "reject_reason",
]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(df.get(col), errors="coerce") if col in df.columns else np.nan
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            X[col] = df[col].astype("string").fillna("__missing__")
        else:
            X[col] = "__missing__"
    return X, df["label"].astype(int)


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8
                       ) -> tuple[pd.Index, pd.Index]:
    df = df.copy()
    df["_sd"] = pd.to_datetime(df["session_date"], errors="coerce")
    df = df.sort_values(["_sd", "timestamp"], kind="stable")
    n = len(df)
    cut = int(round(train_frac * n))
    return df.iloc[:cut].index, df.iloc[cut:].index


def build_pipeline(n_train: int) -> Pipeline:
    """HistGB pipeline with ordinal-encoded categoricals."""
    cat_cols_idx = list(range(len(NUMERIC_FEATURES),
                             len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))

    encoder = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=max(4, n_train // 30),
        l2_regularization=1.0,
        categorical_features=cat_cols_idx,
        early_stopping=False,  # too few rows for a stable validation split
        random_state=42,
    )

    return Pipeline([("encode", encoder), ("clf", clf)])


def _sample_weight(y: pd.Series) -> np.ndarray:
    """Balanced class weights — equivalent to class_weight='balanced'."""
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    total = pos + neg
    w_pos = total / (2 * max(pos, 1))
    w_neg = total / (2 * max(neg, 1))
    return np.where(y == 1, w_pos, w_neg)


def train(df: pd.DataFrame, source_filter: str | None = "live_entry") -> dict:
    """Phase 1 trains on a single source schema by default — phantom and
    live rows have very different feature coverage so mixing them doubles
    the missingness and weakens the signal. Pass source_filter=None to
    train on the union.
    """
    if source_filter is not None and "source" in df.columns:
        df = df[df["source"] == source_filter].reset_index(drop=True)
    if len(df) < 30:
        raise ValueError(f"Dataset too small for training: {len(df)} rows")

    X, y = prepare_xy(df)
    train_idx, test_idx = chronological_split(df)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    pipe = build_pipeline(len(X_train))
    pipe.fit(X_train, y_train, clf__sample_weight=_sample_weight(y_train))

    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    metrics = {
        "n_total": len(df),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate": float(y_test.mean()),
        "test_auc": float(roc_auc_score(y_test, proba_test))
                    if y_test.nunique() > 1 else None,
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "test_log_loss": float(log_loss(y_test, proba_test, labels=[0, 1]))
                         if y_test.nunique() > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, pred_test).tolist(),
    }

    perm = permutation_importance(pipe, X_test, y_test, n_repeats=4,
                                  random_state=42, n_jobs=1,
                                  scoring="roc_auc" if y_test.nunique() > 1
                                  else "accuracy")
    importances = pd.DataFrame({
        "feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return {"pipeline": pipe, "metrics": metrics, "importances": importances,
            "test_proba": proba_test, "test_label": y_test.values}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/training/dataset.parquet")
    ap.add_argument("--model-out", default="data/training/model.pkl")
    ap.add_argument("--report-out", default="data/training/report.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    out = train(df)

    print("=" * 60)
    print("Training summary")
    print("=" * 60)
    for k, v in out["metrics"].items():
        print(f"  {k:24s} {v}")
    print()
    print("Top 12 features by permutation importance:")
    print(out["importances"].head(12).to_string(index=False))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump(out["pipeline"], f)
    with open(args.report_out, "w") as f:
        json.dump({
            "metrics": out["metrics"],
            "top_features": out["importances"].head(20).to_dict(orient="records"),
        }, f, indent=2, default=str)

    print(f"\nModel  -> {args.model_out}")
    print(f"Report -> {args.report_out}")


if __name__ == "__main__":
    main()
