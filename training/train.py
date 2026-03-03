from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .common import (
    CONVERSION_LABEL_COLUMNS,
    FEATURE_COLUMNS,
    ORDERING_FEATURE_COLUMNS,
    SCHEMA_VERSION,
    ensure_columns,
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


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -24.0, 24.0)))


def prob_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    p = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    loss = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(loss.mean())


def auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    positives = int((y_true > 0.5).sum())
    negatives = int((y_true <= 0.5).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, y_true.size + 1, dtype=float)
    pos_ranks = ranks[y_true > 0.5].sum()
    auc = (pos_ranks - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(np.clip(auc, 0.0, 1.0))


def fit_normalization(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std <= 1e-9, 1.0, std)
    return mean, std


def apply_normalization(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    if logits.size == 0 or y_true.size == 0:
        return 1.0
    best_temp = 1.0
    best_loss = float("inf")
    for temp in np.linspace(0.5, 3.0, 26):
        probs = sigmoid(logits / temp)
        loss = prob_logloss(y_true, probs)
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temp)
    return best_temp


def fit_binary_prob_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float,
    clip: float = 8.0,
) -> Tuple[Dict, Dict[str, float]]:
    if x_train.size == 0 or y_train.size == 0:
        weights = np.zeros((x_train.shape[1] + 1 if x_train.ndim == 2 else 1,), dtype=float)
        logits_val = np.zeros_like(y_val, dtype=float)
    else:
        weights = ridge_fit(x_train, y_train, alpha=alpha)
        logits_val = predict(weights, x_val) if x_val.size else np.zeros((0,), dtype=float)
    temp = fit_temperature(logits_val, y_val)
    probs_val = sigmoid(np.clip(logits_val, -clip, clip) / temp) if logits_val.size else np.zeros((0,))
    head = {
        "type": "ridge_logit",
        "bias": float(weights[0]),
        "weights": [float(v) for v in weights[1:]],
        "clip": float(clip),
        "temperature": float(temp),
    }
    head_metrics = {
        "rows_train": int(y_train.size),
        "rows_validation": int(y_val.size),
        "logloss": prob_logloss(y_val, probs_val),
        "auc": auc_roc(y_val, probs_val),
    }
    return head, head_metrics


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
    frame = ensure_columns(frame, CONVERSION_LABEL_COLUMNS, default=0.0)

    frame = ensure_columns(frame, FEATURE_COLUMNS, default=0.0)
    frame = ensure_columns(frame, ORDERING_FEATURE_COLUMNS, default=0.0)
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

    x_train_v2_raw = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    x_val_v2_raw = (
        val[FEATURE_COLUMNS].to_numpy(dtype=float)
        if not val.empty
        else np.zeros((0, len(FEATURE_COLUMNS)))
    )
    mean, std = fit_normalization(x_train_v2_raw)
    x_train_v2 = apply_normalization(x_train_v2_raw, mean, std)
    x_val_v2 = apply_normalization(x_val_v2_raw, mean, std) if x_val_v2_raw.size else x_val_v2_raw

    pickup_train_mask = train["pickup_attempt"].to_numpy(dtype=float) > 0.5
    pickup_val_mask = val["pickup_attempt"].to_numpy(dtype=float) > 0.5 if not val.empty else np.zeros((0,), dtype=bool)
    dropoff_train_mask = train["dropoff_attempt"].to_numpy(dtype=float) > 0.5
    dropoff_val_mask = val["dropoff_attempt"].to_numpy(dtype=float) > 0.5 if not val.empty else np.zeros((0,), dtype=bool)

    pickup_head, pickup_metrics = fit_binary_prob_head(
        x_train_v2[pickup_train_mask],
        train["pickup_success"].to_numpy(dtype=float)[pickup_train_mask],
        x_val_v2[pickup_val_mask] if x_val_v2.size else np.zeros((0, len(FEATURE_COLUMNS))),
        val["pickup_success"].to_numpy(dtype=float)[pickup_val_mask] if not val.empty else np.zeros((0,)),
        alpha=args.alpha,
    )
    dropoff_head, dropoff_metrics = fit_binary_prob_head(
        x_train_v2[dropoff_train_mask],
        train["dropoff_success"].to_numpy(dtype=float)[dropoff_train_mask],
        x_val_v2[dropoff_val_mask] if x_val_v2.size else np.zeros((0, len(FEATURE_COLUMNS))),
        val["dropoff_success"].to_numpy(dtype=float)[dropoff_val_mask] if not val.empty else np.zeros((0,)),
        alpha=args.alpha,
    )
    ordering_head_raw = ridge_fit(
        x_train_v2,
        train["ordering_target"].to_numpy(dtype=float),
        alpha=args.alpha,
    )
    ordering_val_pred = (
        predict(ordering_head_raw, x_val_v2) if x_val_v2.size else np.zeros((0,), dtype=float)
    )
    ordering_head = {
        "type": "ridge",
        "bias": float(ordering_head_raw[0]),
        "weights": [float(v) for v in ordering_head_raw[1:]],
        "clip": 12.0,
    }
    ordering_metrics = metrics(
        val["ordering_target"].to_numpy(dtype=float) if not val.empty else np.zeros((0,)),
        ordering_val_pred,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "feature_columns": FEATURE_COLUMNS,
        "normalization": {
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
        },
        "heads": {
            "pickup": pickup_head,
            "dropoff": dropoff_head,
            "ordering": ordering_head,
        },
        "calibration": {
            "pickup_temp": float(pickup_head["temperature"]),
            "dropoff_temp": float(dropoff_head["temperature"]),
        },
        "weights": named_weights,
        "ordering_weights": named_ordering_weights,
        "metrics": {
            "rows_train": int(len(train)),
            "rows_validation": int(len(val)),
            **metrics(y_val, y_hat),
        },
        "metrics_v2": {
            "pickup": pickup_metrics,
            "dropoff": dropoff_metrics,
            "ordering": ordering_metrics,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote model to {out_path}")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
