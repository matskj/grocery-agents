from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .common import (
    ACTION_SIGNATURE_COLUMNS,
    CONVERSION_LABEL_COLUMNS,
    FEATURE_COLUMNS,
    ORDERING_FEATURE_COLUMNS,
    SCHEMA_VERSION,
    STATE_ACTION_SIGNATURE_EXTRA_COLUMNS,
    build_run_signature_map,
    ensure_columns,
    read_table,
    signature_cluster_sizes,
)


def split_train_validation(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "run_signature" in frame.columns:
        signatures = sorted(frame["run_signature"].dropna().astype(str).unique().tolist())
        if len(signatures) > 1:
            cut = max(1, int(len(signatures) * 0.8))
            train_sigs = set(signatures[:cut])
            train = frame[frame["run_signature"].isin(train_sigs)].copy()
            val = frame[~frame["run_signature"].isin(train_sigs)].copy()
            return train, val
    run_ids = sorted(frame["run_id"].dropna().unique().tolist())
    if len(run_ids) <= 1:
        return frame, frame.iloc[0:0]
    cut = max(1, int(len(run_ids) * 0.8))
    train_ids = set(run_ids[:cut])
    train = frame[frame["run_id"].isin(train_ids)].copy()
    val = frame[~frame["run_id"].isin(train_ids)].copy()
    return train, val


def ridge_fit(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    ones = np.ones((features.shape[0], 1), dtype=float)
    x = np.hstack([ones, features])
    y = targets
    if sample_weight is not None and sample_weight.size == features.shape[0]:
        w = np.sqrt(np.clip(sample_weight.astype(float), 1e-6, None))
        x = x * w[:, None]
        y = y * w
    xtx = x.T @ x
    reg = np.eye(xtx.shape[0], dtype=float) * alpha
    reg[0, 0] = 0.0
    w = np.linalg.solve(xtx + reg, x.T @ y)
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
    sample_weight_train: np.ndarray | None = None,
    clip: float = 8.0,
) -> Tuple[Dict, Dict[str, float]]:
    if x_train.size == 0 or y_train.size == 0:
        weights = np.zeros((x_train.shape[1] + 1 if x_train.ndim == 2 else 1,), dtype=float)
        logits_val = np.zeros_like(y_val, dtype=float)
    else:
        weights = ridge_fit(
            x_train,
            y_train,
            alpha=alpha,
            sample_weight=sample_weight_train,
        )
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


def apply_run_dedup(
    frame: pd.DataFrame,
    strategy: str,
    signature_kind: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if frame.empty or "run_id" not in frame.columns:
        out = frame.copy()
        out["run_signature"] = ""
        out["sample_weight"] = 1.0
        return out, {
            "strategy": strategy,
            "signature_kind": signature_kind,
            "runs_total": 0,
            "runs_kept": 0,
            "signature_clusters": 0,
            "duplicated_clusters": 0,
            "duplicate_runs": 0,
            "max_cluster_size": 0,
            "signature_columns": ACTION_SIGNATURE_COLUMNS,
        }

    signature_by_run = build_run_signature_map(frame, signature_kind=signature_kind)
    cluster_sizes = signature_cluster_sizes(signature_by_run)
    cluster_counts = np.array(list(cluster_sizes.values()), dtype=int)
    duplicated_clusters = int((cluster_counts > 1).sum()) if cluster_counts.size else 0
    duplicate_runs = int(cluster_counts[cluster_counts > 1].sum() - duplicated_clusters) if cluster_counts.size else 0

    run_ids = sorted(signature_by_run.keys())
    kept_run_ids = run_ids
    if strategy == "drop":
        seen = set()
        kept: list[str] = []
        for run_id in run_ids:
            signature = signature_by_run[run_id]
            if signature in seen:
                continue
            seen.add(signature)
            kept.append(run_id)
        kept_run_ids = kept

    out = frame[frame["run_id"].astype(str).isin(set(kept_run_ids))].copy()
    out["run_signature"] = out["run_id"].astype(str).map(signature_by_run).fillna("")

    if strategy == "downweight":
        run_weight = {
            run_id: 1.0 / float(max(1, cluster_sizes.get(signature_by_run[run_id], 1)))
            for run_id in kept_run_ids
        }
    else:
        run_weight = {run_id: 1.0 for run_id in kept_run_ids}
    out["sample_weight"] = (
        out["run_id"].astype(str).map(run_weight).fillna(1.0).astype(float).clip(lower=1e-6)
    )

    signature_columns = list(ACTION_SIGNATURE_COLUMNS)
    if signature_kind == "state_action":
        signature_columns.extend(STATE_ACTION_SIGNATURE_EXTRA_COLUMNS)
    dedup = {
        "strategy": strategy,
        "signature_kind": signature_kind,
        "runs_total": int(len(run_ids)),
        "runs_kept": int(len(kept_run_ids)),
        "signature_clusters": int(len(cluster_sizes)),
        "duplicated_clusters": duplicated_clusters,
        "duplicate_runs": duplicate_runs,
        "max_cluster_size": int(cluster_counts.max()) if cluster_counts.size else 0,
        "top_cluster_sizes": [int(v) for v in sorted(cluster_sizes.values(), reverse=True)[:5]],
        "effective_run_weight_sum": float(sum(run_weight.values())),
        "signature_columns": signature_columns,
    }
    return out, dedup


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a per-mode linear scorer.")
    parser.add_argument("--mode", required=True, choices=["easy", "medium", "hard", "expert", "custom"])
    parser.add_argument("--data", default="data/runs_features.parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument(
        "--dedup-strategy",
        choices=["none", "downweight", "drop"],
        default="downweight",
        help="Handle near-identical runs via action/state-action signatures.",
    )
    parser.add_argument(
        "--signature-kind",
        choices=["action", "state_action"],
        default="action",
        help="Signature basis for run deduplication.",
    )
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
    frame, dedup = apply_run_dedup(frame, args.dedup_strategy, args.signature_kind)
    if frame.empty or len(frame) < args.min_rows:
        raise SystemExit(
            f"not enough rows after dedup for mode={args.mode}: {len(frame)} (min={args.min_rows})"
        )

    train, val = split_train_validation(frame)
    train_weights = train.get("sample_weight", pd.Series(np.ones(len(train)))).to_numpy(dtype=float)
    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["n_step_return"].to_numpy(dtype=float)
    weights = ridge_fit(x_train, y_train, alpha=args.alpha, sample_weight=train_weights)

    x_val = val[FEATURE_COLUMNS].to_numpy(dtype=float) if not val.empty else np.zeros((0, len(FEATURE_COLUMNS)))
    y_val = val["n_step_return"].to_numpy(dtype=float) if not val.empty else np.zeros((0,))
    y_hat = predict(weights, x_val) if x_val.size else np.zeros((0,))

    named_weights = {"bias": float(weights[0])}
    for idx, name in enumerate(FEATURE_COLUMNS, start=1):
        named_weights[name] = float(weights[idx])

    x_order = train[ORDERING_FEATURE_COLUMNS].to_numpy(dtype=float)
    y_order = train["ordering_target"].to_numpy(dtype=float)
    ordering_weights_raw = ridge_fit(
        x_order,
        y_order,
        alpha=args.alpha,
        sample_weight=train_weights,
    )
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
        sample_weight_train=train_weights[pickup_train_mask],
    )
    dropoff_head, dropoff_metrics = fit_binary_prob_head(
        x_train_v2[dropoff_train_mask],
        train["dropoff_success"].to_numpy(dtype=float)[dropoff_train_mask],
        x_val_v2[dropoff_val_mask] if x_val_v2.size else np.zeros((0, len(FEATURE_COLUMNS))),
        val["dropoff_success"].to_numpy(dtype=float)[dropoff_val_mask] if not val.empty else np.zeros((0,)),
        alpha=args.alpha,
        sample_weight_train=train_weights[dropoff_train_mask],
    )
    ordering_head_raw = ridge_fit(
        x_train_v2,
        train["ordering_target"].to_numpy(dtype=float),
        alpha=args.alpha,
        sample_weight=train_weights,
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
        "dedup": dedup,
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
