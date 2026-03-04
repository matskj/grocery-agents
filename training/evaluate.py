from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .common import (
    CONVERSION_LABEL_COLUMNS,
    FEATURE_COLUMNS,
    build_run_signature_map,
    ensure_columns,
    read_table,
    signature_cluster_sizes,
)


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


def extract_ordering_sequence_matrix(
    frame: pd.DataFrame, feature_columns: list[str]
) -> np.ndarray:
    cols = []
    for name in feature_columns:
        if name == "dist_to_goal":
            series = frame.get("dist_to_goal_proxy", pd.Series(np.zeros(len(frame))))
        elif name == "choke_occupancy":
            series = frame.get("choke_occupancy_proxy", pd.Series(np.zeros(len(frame))))
        else:
            series = frame.get(name, pd.Series(np.zeros(len(frame))))
        cols.append(series.astype(float).to_numpy(dtype=float))
    if not cols:
        return np.zeros((len(frame), 0), dtype=float)
    return np.stack(cols, axis=1)


def ordering_preference_target(frame: pd.DataFrame) -> np.ndarray:
    pickup = frame.get("pickup_success", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
    dropoff = frame.get("dropoff_success", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
    conflict_reduction = frame.get(
        "conflict_reduction", pd.Series(np.zeros(len(frame)))
    ).to_numpy(dtype=float)
    queue_eta_improve = frame.get(
        "queue_eta_improve", pd.Series(np.zeros(len(frame)))
    ).to_numpy(dtype=float)
    delta_item = np.maximum(
        frame.get("delta_dist_to_active_item", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float),
        0.0,
    )
    delta_drop = np.maximum(
        frame.get("delta_dist_to_dropoff", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float),
        0.0,
    )
    move_failed = frame.get("move_failed", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
    noop_move = frame.get("noop_move", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
    local_conflict = frame.get(
        "local_conflict_count", pd.Series(np.zeros(len(frame)))
    ).to_numpy(dtype=float)
    return (
        2.0 * (pickup + dropoff)
        + 1.0 * conflict_reduction
        + 0.8 * queue_eta_improve
        + 0.5 * delta_drop
        + 0.4 * delta_item
        - 1.0 * move_failed
        - 0.4 * noop_move
        - 0.3 * local_conflict
    )


def build_pairwise_eval(
    frame: pd.DataFrame, x_norm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame.empty or x_norm.size == 0:
        return (
            np.zeros((0, x_norm.shape[1] if x_norm.ndim == 2 else 0), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=bool),
        )
    pref = ordering_preference_target(frame)
    blocked = (
        frame.get("blocked_ticks", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float) > 0.0
    ) | (
        frame.get("local_conflict_count", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
        > 0.0
    )
    work = frame[["run_id", "tick"]].copy()
    work["row_idx"] = np.arange(len(frame), dtype=int)
    pair_x: list[np.ndarray] = []
    pair_y: list[float] = []
    pair_blocked: list[bool] = []
    for (_, _), group in work.groupby(["run_id", "tick"], sort=False):
        idxs = group["row_idx"].to_numpy(dtype=int)
        if idxs.size < 2:
            continue
        pref_local = pref[idxs]
        blocked_local = blocked[idxs]
        for i in range(idxs.size - 1):
            for j in range(i + 1, idxs.size):
                left = idxs[i]
                right = idxs[j]
                p_left = pref_local[i]
                p_right = pref_local[j]
                if p_left == p_right:
                    continue
                pair_x.append(x_norm[left] - x_norm[right])
                pair_y.append(1.0 if p_left > p_right else 0.0)
                pair_blocked.append(bool(blocked_local[i] or blocked_local[j]))
    if not pair_x:
        return (
            np.zeros((0, x_norm.shape[1]), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=bool),
        )
    return (
        np.vstack(pair_x).astype(float),
        np.array(pair_y, dtype=float),
        np.array(pair_blocked, dtype=bool),
    )


def build_team_tick_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keep = frame.sort_values(["run_id", "tick", "bot_id"], kind="mergesort")
    return keep.drop_duplicates(["run_id", "tick"], keep="first").copy()


def summarize_runs(team_ticks: pd.DataFrame) -> pd.DataFrame:
    if team_ticks.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "final_score",
                "items_delivered",
                "orders_completed",
                "ticks",
                "delivered_per_100_ticks",
            ]
        )
    grouped = team_ticks.groupby("run_id", sort=False)
    out = grouped.agg(
        final_score=("score", "max"),
        items_delivered=("items_delivered_delta", "sum"),
        orders_completed=("order_completed_delta", "sum"),
        ticks=("tick", "max"),
    ).reset_index()
    out["ticks"] = out["ticks"].astype(int) + 1
    out["delivered_per_100_ticks"] = (
        out["items_delivered"].astype(float) * 100.0 / out["ticks"].clip(lower=1).astype(float)
    )
    return out


def _expanded_event_ticks(group: pd.DataFrame, delta_col: str) -> list[int]:
    ticks: list[int] = []
    for tick, delta in zip(group["tick"].astype(int), group[delta_col].astype(int)):
        for _ in range(max(0, delta)):
            ticks.append(int(tick))
    return ticks


def run_completion_profile(team_ticks: pd.DataFrame, run_id: str) -> Dict[str, object]:
    group = team_ticks[team_ticks["run_id"].astype(str) == str(run_id)].copy()
    if group.empty:
        return {}
    group = group.sort_values("tick", kind="mergesort")
    order_ticks = _expanded_event_ticks(group, "order_completed_delta")
    item_ticks = _expanded_event_ticks(group, "items_delivered_delta")

    order_gaps = np.diff(np.array(order_ticks, dtype=int)) if len(order_ticks) >= 2 else np.array([], dtype=int)
    item_gaps = np.diff(np.array(item_ticks, dtype=int)) if len(item_ticks) >= 2 else np.array([], dtype=int)
    return {
        "run_id": str(run_id),
        "final_score": int(group["score"].max()),
        "items_delivered": int(group["items_delivered_delta"].sum()),
        "orders_completed": int(group["order_completed_delta"].sum()),
        "order_completion_ticks": [int(t) for t in order_ticks[:12]],
        "item_delivery_ticks": [int(t) for t in item_ticks[:16]],
        "order_completion_gap_mean": float(order_gaps.mean()) if order_gaps.size else 0.0,
        "order_completion_gap_max": int(order_gaps.max()) if order_gaps.size else 0,
        "item_delivery_gap_mean": float(item_gaps.mean()) if item_gaps.size else 0.0,
        "item_delivery_gap_max": int(item_gaps.max()) if item_gaps.size else 0,
    }


def choose_run_id(
    run_summary: pd.DataFrame,
    preferred_run_id: Optional[str],
    preferred_score: Optional[int],
) -> Optional[str]:
    if preferred_run_id:
        run_id = str(preferred_run_id)
        if run_id in set(run_summary["run_id"].astype(str)):
            return run_id
        return None
    if preferred_score is None:
        return None
    score_matches = run_summary[run_summary["final_score"].astype(int) == int(preferred_score)].copy()
    if score_matches.empty:
        return None
    score_matches = score_matches.sort_values(
        ["orders_completed", "items_delivered", "run_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return str(score_matches.iloc[0]["run_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained mode model.")
    parser.add_argument("--data", default="data/runs_features.parquet")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--reference-run-id", default=None)
    parser.add_argument("--candidate-run-id", default=None)
    parser.add_argument("--reference-score", type=int, default=None)
    parser.add_argument("--candidate-score", type=int, default=None)
    parser.add_argument(
        "--signature-kind",
        choices=["action", "state_action"],
        default="action",
        help="Run-signature basis for duplicate-cluster diagnostics.",
    )
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
    team_ticks = build_team_tick_frame(frame)
    run_summary = summarize_runs(team_ticks)
    signature_by_run = build_run_signature_map(frame, signature_kind=args.signature_kind)
    cluster_sizes = signature_cluster_sizes(signature_by_run)
    cluster_counts = np.array(list(cluster_sizes.values()), dtype=int)
    duplicated_clusters = int((cluster_counts > 1).sum()) if cluster_counts.size else 0
    duplicate_runs = int(cluster_counts[cluster_counts > 1].sum() - duplicated_clusters) if cluster_counts.size else 0
    best_run = (
        run_summary.sort_values(
            ["final_score", "orders_completed", "items_delivered", "run_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).iloc[0]
        if not run_summary.empty
        else None
    )
    result["run_analysis"] = {
        "runs": int(len(run_summary)),
        "best_run_id": str(best_run["run_id"]) if best_run is not None else None,
        "best_score": int(best_run["final_score"]) if best_run is not None else None,
        "duplicate_signature_clusters": duplicated_clusters,
        "duplicate_runs": duplicate_runs,
        "top_cluster_sizes": [int(v) for v in sorted(cluster_sizes.values(), reverse=True)[:5]],
    }

    reference_run_id = choose_run_id(run_summary, args.reference_run_id, args.reference_score)
    candidate_run_id = choose_run_id(run_summary, args.candidate_run_id, args.candidate_score)
    if not reference_run_id and not candidate_run_id and not run_summary.empty:
        auto_ref = choose_run_id(run_summary, None, 101)
        auto_cand = choose_run_id(run_summary, None, 85)
        if auto_ref and auto_cand:
            reference_run_id = auto_ref
            candidate_run_id = auto_cand
    if reference_run_id and candidate_run_id:
        ref_profile = run_completion_profile(team_ticks, reference_run_id)
        cand_profile = run_completion_profile(team_ticks, candidate_run_id)
        if ref_profile and cand_profile:
            result["trajectory_diff"] = {
                "reference": ref_profile,
                "candidate": cand_profile,
                "delta_items_delivered": int(ref_profile["items_delivered"]) - int(cand_profile["items_delivered"]),
                "delta_orders_completed": int(ref_profile["orders_completed"]) - int(cand_profile["orders_completed"]),
                "delta_score": int(ref_profile["final_score"]) - int(cand_profile["final_score"]),
                "same_signature": bool(
                    signature_by_run.get(reference_run_id, "") == signature_by_run.get(candidate_run_id, "")
                    and signature_by_run.get(reference_run_id, "") != ""
                ),
            }

    feature_columns = model.get("feature_columns", [])
    runtime_feature_columns = model.get("runtime_feature_columns", [])
    heads = model.get("heads", {})
    normalization = model.get("normalization", {})
    eval_cols: list[str] = []
    if isinstance(heads, dict):
        if isinstance(runtime_feature_columns, list) and runtime_feature_columns:
            eval_cols = [str(c) for c in runtime_feature_columns]
        elif isinstance(feature_columns, list) and feature_columns:
            eval_cols = [str(c) for c in feature_columns]
    if eval_cols and isinstance(heads, dict):
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
        value_proxy = np.maximum(ordering_score, 0.0)
        eta_item = frame.get("dist_to_nearest_active_item", pd.Series(np.ones(len(frame)))).to_numpy(dtype=float)
        eta_drop = frame.get("dist_to_dropoff", pd.Series(np.ones(len(frame)))).to_numpy(dtype=float)
        eta_proxy = np.clip(eta_item + 0.7 * eta_drop, 0.0, 160.0)
        expected_score = value_proxy * pickup_probs * dropoff_probs / (eta_proxy + 1.0)
        realized = frame.get("items_delivered_delta", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
        idle_far = frame.get("idle_far_from_dropoff", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
        carrying_only_inactive = frame.get(
            "carrying_only_inactive", pd.Series(np.zeros(len(frame)))
        ).to_numpy(dtype=float)
        preferred_area_match = frame.get(
            "preferred_area_match", pd.Series(np.zeros(len(frame)))
        ).to_numpy(dtype=float)
        expansion_mode_active = frame.get(
            "expansion_mode_active", pd.Series(np.zeros(len(frame)))
        ).to_numpy(dtype=float)
        out_of_area_target = frame.get(
            "out_of_area_target", pd.Series(np.zeros(len(frame)))
        ).to_numpy(dtype=float)
        out_of_radius_target = frame.get(
            "out_of_radius_target", pd.Series(np.zeros(len(frame)))
        ).to_numpy(dtype=float)
        goal_valid = frame.get("goal_cell_valid", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
        idle_far_inactive = (
            ((idle_far > 0.5) & (carrying_only_inactive > 0.5)).sum()
            / max(1, int((carrying_only_inactive > 0.5).sum()))
        )
        local_first_violation = (
            ((expansion_mode_active <= 0.5) & ((out_of_area_target > 0.5) | (out_of_radius_target > 0.5))).sum()
            / max(1, int((expansion_mode_active <= 0.5).sum()))
        )
        preferred_area_match_rate = (
            ((preferred_area_match > 0.5) & (goal_valid > 0.5)).sum()
            / max(1, int((goal_valid > 0.5).sum()))
        )

        pickup_mask = frame["pickup_attempt"].to_numpy(dtype=float) > 0.5
        dropoff_mask = frame["dropoff_attempt"].to_numpy(dtype=float) > 0.5
        pickup_true = frame["pickup_success"].to_numpy(dtype=float)[pickup_mask]
        dropoff_true = frame["dropoff_success"].to_numpy(dtype=float)[dropoff_mask]
        pickup_eval = pickup_probs[pickup_mask] if pickup_probs.size else np.zeros((0,))
        dropoff_eval = dropoff_probs[dropoff_mask] if dropoff_probs.size else np.zeros((0,))
        ordering_sequence_metrics: Dict[str, object] = {}
        ordering_sequence_head = model.get("ordering_sequence_head", {})
        if isinstance(ordering_sequence_head, dict):
            seq_cols = [str(v) for v in ordering_sequence_head.get("feature_columns", [])]
            seq_weights = np.array(ordering_sequence_head.get("weights", []), dtype=float)
            if seq_cols and seq_weights.size == len(seq_cols):
                x_seq = extract_ordering_sequence_matrix(frame, seq_cols)
                seq_norm = ordering_sequence_head.get("normalization", {})
                seq_mean = np.array(seq_norm.get("mean", [0.0] * len(seq_cols)), dtype=float)
                seq_std = np.array(seq_norm.get("std", [1.0] * len(seq_cols)), dtype=float)
                if seq_mean.size != len(seq_cols):
                    seq_mean = np.zeros((len(seq_cols),), dtype=float)
                if seq_std.size != len(seq_cols):
                    seq_std = np.ones((len(seq_cols),), dtype=float)
                seq_std = np.where(seq_std <= 1e-9, 1.0, seq_std)
                x_seq_norm = (x_seq - seq_mean) / seq_std
                pair_x, pair_y, pair_blocked = build_pairwise_eval(frame, x_seq_norm)
                seq_bias = float(ordering_sequence_head.get("bias", 0.0))
                seq_temp = float(ordering_sequence_head.get("temperature", 1.0))
                if pair_x.size:
                    pair_logits = seq_bias + pair_x @ seq_weights
                    pair_probs = sigmoid(pair_logits / max(0.25, seq_temp))
                    seq_pair_auc = auc_roc(pair_y, pair_probs)
                    seq_pair_logloss = prob_logloss(pair_y, pair_probs)
                    pair_pred = (pair_probs >= 0.5).astype(float)
                    pair_acc = float((pair_pred == pair_y).mean())
                    blocked_acc = (
                        float((pair_pred[pair_blocked] == pair_y[pair_blocked]).mean())
                        if pair_blocked.any()
                        else 0.0
                    )
                    non_blocked_mask = ~pair_blocked
                    non_blocked_acc = (
                        float((pair_pred[non_blocked_mask] == pair_y[non_blocked_mask]).mean())
                        if non_blocked_mask.any()
                        else 0.0
                    )
                else:
                    seq_pair_auc = 0.5
                    seq_pair_logloss = 0.0
                    pair_acc = 0.0
                    blocked_acc = 0.0
                    non_blocked_acc = 0.0
                seq_utility = seq_bias + x_seq_norm @ seq_weights
                conflict_proxy = (
                    frame.get("conflict_reduction", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
                    - 0.5
                    * frame.get("local_conflict_count", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
                )
                ordering_sequence_metrics = {
                    "pair_auc": float(seq_pair_auc),
                    "pair_logloss": float(seq_pair_logloss),
                    "pair_accuracy": float(pair_acc),
                    "blocked_pair_accuracy": float(blocked_acc),
                    "reorder_gain_blocked_stuck": float(blocked_acc - non_blocked_acc),
                    "conflict_avoidance_proxy_corr": pearson_corr(seq_utility, conflict_proxy),
                    "rows_pairs": int(pair_y.size),
                }
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
            "spatial_idle": {
                "idle_far_from_dropoff_rate": float((idle_far > 0.5).mean()) if idle_far.size else 0.0,
                "carrying_only_inactive_rate": float((carrying_only_inactive > 0.5).mean())
                if carrying_only_inactive.size
                else 0.0,
                "idle_far_given_carrying_only_inactive_rate": float(idle_far_inactive),
            },
            "locality": {
                "preferred_area_match_rate": float(preferred_area_match_rate),
                "out_of_area_target_rate": float((out_of_area_target > 0.5).mean())
                if out_of_area_target.size
                else 0.0,
                "out_of_radius_target_rate": float((out_of_radius_target > 0.5).mean())
                if out_of_radius_target.size
                else 0.0,
                "expansion_mode_tick_ratio": float((expansion_mode_active > 0.5).mean())
                if expansion_mode_active.size
                else 0.0,
                "local_first_violation_ratio": float(local_first_violation),
            },
            "ordering_sequence": ordering_sequence_metrics,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
