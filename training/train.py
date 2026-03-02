from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .common import (
    FEATURE_COLUMNS,
    ORDERING_FEATURE_COLUMNS,
    SCHEMA_VERSION,
    read_table,
)


def split_train_validation(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_ids = sorted(frame["run_id"].dropna().unique().tolist())
    if len(run_ids) <= 1:
        return frame, frame.iloc[0:0]
    cut = max(1, int(len(run_ids) * 0.8))
    train_ids = set(run_ids[:cut])
    train = frame[frame["run_id"].isin(train_ids)].copy()
    val = frame[~frame["run_id"].isin(train_ids)].copy()
    return train, val


def ridge_fit(features: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    ones = np.ones((features.shape[0], 1), dtype=float)
    x = np.hstack([ones, features])
    xtx = x.T @ x
    reg = np.eye(xtx.shape[0], dtype=float) * alpha
    reg[0, 0] = 0.0
    w = np.linalg.solve(xtx + reg, x.T @ targets)
    return w


def predict(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    ones = np.ones((features.shape[0], 1), dtype=float)
    x = np.hstack([ones, features])
    return x @ weights


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    err = y_true - y_pred
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    ss_res = float((err**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a per-mode linear scorer.")
    parser.add_argument("--mode", required=True, choices=["easy", "medium", "hard", "expert", "custom"])
    parser.add_argument("--data", default="data/runs_features.parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-rows", type=int, default=50)
    args = parser.parse_args()

    frame = read_table(Path(args.data))
    frame = frame[frame["mode"] == args.mode].copy()
    if frame.empty or len(frame) < args.min_rows:
        raise SystemExit(f"not enough rows for mode={args.mode}: {len(frame)} (min={args.min_rows})")

    if "n_step_return" not in frame.columns:
        frame["n_step_return"] = frame["reward_proxy"]
    if "ordering_target" not in frame.columns:
        frame["ordering_target"] = 0.0

    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0
    for col in ORDERING_FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train, val = split_train_validation(frame)
    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["n_step_return"].to_numpy(dtype=float)
    weights = ridge_fit(x_train, y_train, alpha=args.alpha)

    x_val = val[FEATURE_COLUMNS].to_numpy(dtype=float) if not val.empty else np.zeros((0, len(FEATURE_COLUMNS)))
    y_val = val["n_step_return"].to_numpy(dtype=float) if not val.empty else np.zeros((0,))
    y_hat = predict(weights, x_val) if x_val.size else np.zeros((0,))

    named_weights = {"bias": float(weights[0])}
    for idx, name in enumerate(FEATURE_COLUMNS, start=1):
        named_weights[name] = float(weights[idx])

    x_order = train[ORDERING_FEATURE_COLUMNS].to_numpy(dtype=float)
    y_order = train["ordering_target"].to_numpy(dtype=float)
    ordering_weights_raw = ridge_fit(x_order, y_order, alpha=args.alpha)
    named_ordering_weights = {"bias": float(ordering_weights_raw[0])}
    for idx, name in enumerate(ORDERING_FEATURE_COLUMNS, start=1):
        named_ordering_weights[name] = float(ordering_weights_raw[idx])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "weights": named_weights,
        "ordering_weights": named_ordering_weights,
        "metrics": {
            "rows_train": int(len(train)),
            "rows_validation": int(len(val)),
            **metrics(y_val, y_hat),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote model to {out_path}")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
