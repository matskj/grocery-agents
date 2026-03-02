from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_table, write_table


def col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name].astype(float)
    return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)


def reward_proxy(frame: pd.DataFrame) -> pd.Series:
    return (
        col(frame, "delta_score")
        + 0.75 * col(frame, "items_delivered_delta")
        + 3.0 * col(frame, "order_completed_delta")
        - 0.50 * col(frame, "invalid_action_count")
        - 0.05 * col(frame, "is_wait")
        - 0.60 * col(frame, "noop_move")
        - 0.20 * col(frame, "move_target_blocked")
        - 0.45 * col(frame, "is_queue_violation")
        - 0.30 * col(frame, "near_dropoff_blocking")
        - 0.10 * col(frame, "repeated_failed_move_count")
        - 0.08 * col(frame, "conflict_degree")
        - 0.55 * col(frame, "intent_move_but_wait")
        - 0.35 * col(frame, "wait_reason_no_path_with_constraints")
        - 0.30 * col(frame, "wait_reason_forbidden_queue_zone")
        - 0.20 * col(frame, "wait_reason_blocked_by_vertex_reservation")
        - 0.15 * col(frame, "wait_reason_blocked_by_edge_reservation")
        - 0.20 * col(frame, "wait_reason_prohibited_repeat_move")
        - 0.35 * col(frame, "cbs_timeout")
        - 0.20 * col(frame, "in_corner")
        - 0.10 * col(frame, "dead_end_depth")
        - 0.90 * col(frame, "dropoff_target_pending")
        - 0.45 * col(frame, "dropoff_attempt_same_order_streak")
        - 0.35 * col(frame, "dropoff_watchdog_triggered")
        - 0.20 * col(frame, "loop_two_cycle_count")
        + 0.30 * col(frame, "yield_applied")
        + 0.20 * col(frame, "queue_advance")
        + 0.35 * col(frame, "corner_exit_success")
        + 0.15 * col(frame, "conflict_reduction")
        + 0.10 * col(frame, "queue_eta_improve")
        + 0.25 * col(frame, "coverage_gain")
        + 0.20 * col(frame, "serviceable_dropoff")
        + 0.50 * col(frame, "dropoff_target_in_progress")
    )


def n_step_return(values: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for idx in range(len(values)):
        out[idx] = values[idx : idx + n].sum()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Add training features and n-step targets.")
    parser.add_argument("--data", default="data/runs.parquet")
    parser.add_argument("--out", default="data/runs_features.parquet")
    parser.add_argument("--n-step", type=int, default=5)
    args = parser.parse_args()

    frame = read_table(Path(args.data))
    if frame.empty:
        written = write_table(frame, Path(args.out))
        print(f"input dataset is empty; wrote {written}")
        return

    frame = frame.sort_values(["run_id", "bot_id", "tick"]).reset_index(drop=True)
    for required in [
        "in_corner",
        "local_conflict_count",
        "queue_eta_rank",
        "queue_role_courier",
        "queue_role_lead",
        "cbs_timeout",
        "dead_end_depth",
        "dropoff_target_status",
        "dropoff_attempt_same_order_streak",
        "dropoff_watchdog_triggered",
        "loop_two_cycle_count",
        "coverage_gain",
        "serviceable_dropoff",
        "ordering_rank",
        "ordering_score",
    ]:
        if required not in frame.columns:
            frame[required] = 0
    frame["next_bot_x"] = frame.groupby(["run_id", "bot_id"], sort=False)["bot_x"].shift(-1)
    frame["next_bot_y"] = frame.groupby(["run_id", "bot_id"], sort=False)["bot_y"].shift(-1)
    frame["noop_move"] = (
        (frame["action_kind"] == "move")
        & (frame["next_bot_x"] == frame["bot_x"])
        & (frame["next_bot_y"] == frame["bot_y"])
    ).astype(int)
    frame["move_failed"] = (
        (frame["action_kind"] == "move") & (frame["noop_move"] == 1)
    ).astype(int)
    frame["queue_distance_next"] = frame.groupby(["run_id", "bot_id"], sort=False)[
        "queue_distance"
    ].shift(-1)
    frame["queue_advance"] = (
        frame["queue_distance"] - frame["queue_distance_next"]
    ).clip(lower=0).fillna(0.0)
    frame["in_corner_next"] = frame.groupby(["run_id", "bot_id"], sort=False)["in_corner"].shift(-1)
    frame["corner_exit_success"] = (
        (frame["in_corner"] == 1) & (frame["in_corner_next"] == 0)
    ).astype(int)
    frame["local_conflict_next"] = frame.groupby(["run_id", "bot_id"], sort=False)[
        "local_conflict_count"
    ].shift(-1)
    frame["local_conflict_next2"] = frame.groupby(["run_id", "bot_id"], sort=False)[
        "local_conflict_count"
    ].shift(-2)
    frame["conflict_reduction"] = (
        frame["local_conflict_count"]
        - frame[["local_conflict_next", "local_conflict_next2"]].min(axis=1)
    ).clip(lower=0).fillna(0.0)
    frame["queue_eta_rank_next"] = frame.groupby(["run_id", "bot_id"], sort=False)[
        "queue_eta_rank"
    ].shift(-1)
    frame["queue_eta_improve"] = (
        frame["queue_eta_rank"] - frame["queue_eta_rank_next"]
    ).clip(lower=0).fillna(0.0)
    courier_mask = (
        (frame.get("queue_role_courier", 0) == 1)
        | (frame.get("queue_role_lead", 0) == 1)
    )
    frame["queue_eta_improve"] = frame["queue_eta_improve"] * courier_mask.astype(int)
    frame["dropoff_target_status"] = frame.get("dropoff_target_status", "none").astype(str)
    frame["dropoff_target_pending"] = (
        frame["dropoff_target_status"] == "pending"
    ).astype(int)
    frame["dropoff_target_in_progress"] = (
        frame["dropoff_target_status"] == "in_progress"
    ).astype(int)
    frame["carrying_active"] = frame.get("serviceable_dropoff", 0).astype(float)
    frame["dist_to_goal_proxy"] = frame.get("queue_distance", frame.get("dist_to_dropoff", 0)).astype(float)
    frame["dropoff_watchdog_pressure"] = (
        frame.get("dropoff_attempt_same_order_streak", 0).astype(float)
        + 2.0 * frame.get("dropoff_watchdog_triggered", 0).astype(float)
    )
    frame["choke_occupancy_proxy"] = (
        frame.get("local_congestion", 0).astype(float)
        + 0.5 * frame.get("dropoff_congestion", 0).astype(float)
    )
    frame["ordering_target"] = (
        1.2 * frame.get("conflict_reduction", 0).astype(float)
        + 1.0 * frame.get("queue_eta_improve", 0).astype(float)
        - 1.5 * frame.get("move_failed", 0).astype(float)
        - 0.2 * frame.get("loop_two_cycle_count", 0).astype(float)
        + 0.3 * frame.get("coverage_gain", 0).astype(float)
    )
    frame["wait_reason"] = frame.get("wait_reason", "intent_wait").astype(str)
    for reason in [
        "blocked_by_vertex_reservation",
        "blocked_by_edge_reservation",
        "forbidden_queue_zone",
        "prohibited_repeat_move",
        "no_path_with_constraints",
        "timeout_fallback",
    ]:
        frame[f"wait_reason_{reason}"] = (frame["wait_reason"] == reason).astype(int)
    frame["reward_proxy"] = reward_proxy(frame)

    result = []
    for (_, _, _), group in frame.groupby(["run_id", "bot_id", "mode"], sort=False):
        group = group.copy()
        group["n_step_return"] = n_step_return(group["reward_proxy"].to_numpy(dtype=float), args.n_step)
        result.append(group)
    merged = pd.concat(result, ignore_index=True)

    written = write_table(merged, Path(args.out))
    print(f"wrote {len(merged)} rows to {written}")


if __name__ == "__main__":
    main()
