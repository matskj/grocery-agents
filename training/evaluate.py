from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .common import CONVERSION_LABEL_COLUMNS, FEATURE_COLUMNS, ensure_columns, read_table


def load_model(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score(frame: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = np.ones((x.shape[0],), dtype=float) * float(weights.get("bias", 0.0))
    for idx, name in enumerate(FEATURE_COLUMNS):
        y += x[:, idx] * float(weights.get(name, 0.0))
    return y


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


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std <= 1e-9 or b_std <= 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def v2_head_predict(
    x_norm: np.ndarray,
    head: Dict,
    default_clip: float = 8.0,
) -> np.ndarray:
    weights = np.array(head.get("weights", []), dtype=float)
    if x_norm.shape[1] != weights.size:
        return np.zeros((x_norm.shape[0],), dtype=float)
    bias = float(head.get("bias", 0.0))
    clip = float(head.get("clip", default_clip))
    temp = float(head.get("temperature", 1.0))
    logits = bias + x_norm @ weights
    return sigmoid(np.clip(logits, -clip, clip) / max(0.25, temp))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
    parser = argparse.ArgumentParser(description="Evaluate a trained mode model.")
    parser.add_argument("--data", default="data/runs_features.parquet")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default=None)
    args = parser.parse_args()

    frame = read_table(Path(args.data))
    model = load_model(Path(args.model))
    mode = args.mode or model.get("mode")
    if mode:
        frame = frame[frame["mode"] == mode].copy()
    if frame.empty:
        raise SystemExit("no rows to evaluate")

    if "n_step_return" not in frame.columns:
        frame["n_step_return"] = frame["reward_proxy"]
    frame = ensure_columns(frame, FEATURE_COLUMNS, default=0.0)
    frame = ensure_columns(frame, CONVERSION_LABEL_COLUMNS, default=0.0)
    if "ordering_target" not in frame.columns:
        frame["ordering_target"] = 0.0

    y_true = frame["n_step_return"].to_numpy(dtype=float)
    y_pred = score(frame, model.get("weights", {}))
    result: Dict[str, object] = {
        "mode": mode,
        "rows": int(len(frame)),
        **compute_metrics(y_true, y_pred),
    }
    feature_columns = model.get("feature_columns", [])
    heads = model.get("heads", {})
    normalization = model.get("normalization", {})
    if isinstance(feature_columns, list) and feature_columns and isinstance(heads, dict):
        eval_cols = [str(c) for c in feature_columns]
        frame = ensure_columns(frame, eval_cols, default=0.0)
        x = frame[eval_cols].to_numpy(dtype=float)
        mean = np.array(normalization.get("mean", [0.0] * len(eval_cols)), dtype=float)
        std = np.array(normalization.get("std", [1.0] * len(eval_cols)), dtype=float)
        if mean.size != len(eval_cols):
            mean = np.zeros((len(eval_cols),), dtype=float)
        if std.size != len(eval_cols):
            std = np.ones((len(eval_cols),), dtype=float)
        std = np.where(std <= 1e-9, 1.0, std)
        x_norm = (x - mean) / std

        pickup_probs = v2_head_predict(x_norm, heads.get("pickup", {}))
        dropoff_probs = v2_head_predict(x_norm, heads.get("dropoff", {}))
        ordering_head = heads.get("ordering", {})
        ordering_weights = np.array(ordering_head.get("weights", []), dtype=float)
        if ordering_weights.size == x_norm.shape[1]:
            ordering_score = float(ordering_head.get("bias", 0.0)) + x_norm @ ordering_weights
        else:
            ordering_score = frame.get("ordering_score", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
        eta = frame.get("dist_to_dropoff", pd.Series(np.ones(len(frame)))).to_numpy(dtype=float)
        eta = np.clip(eta, 0.0, 99.0)
        expected_score = ordering_score * pickup_probs * dropoff_probs / (eta + 1.0)
        realized = frame.get("items_delivered_delta", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)

        pickup_mask = frame["pickup_attempt"].to_numpy(dtype=float) > 0.5
        dropoff_mask = frame["dropoff_attempt"].to_numpy(dtype=float) > 0.5
        pickup_true = frame["pickup_success"].to_numpy(dtype=float)[pickup_mask]
        dropoff_true = frame["dropoff_success"].to_numpy(dtype=float)[dropoff_mask]
        pickup_eval = pickup_probs[pickup_mask] if pickup_probs.size else np.zeros((0,))
        dropoff_eval = dropoff_probs[dropoff_mask] if dropoff_probs.size else np.zeros((0,))
        result["metrics_v2"] = {
            "pickup": {
                "rows": int(pickup_true.size),
                "logloss": prob_logloss(pickup_true, pickup_eval),
                "auc": auc_roc(pickup_true, pickup_eval),
            },
            "dropoff": {
                "rows": int(dropoff_true.size),
                "logloss": prob_logloss(dropoff_true, dropoff_eval),
                "auc": auc_roc(dropoff_true, dropoff_eval),
            },
            "expected_score_delivery_corr": pearson_corr(expected_score, realized),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
