from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, DefaultDict

import numpy as np
import pandas as pd

from .common import (
    CONVERSION_LABEL_COLUMNS,
    LOCALITY_FEATURE_COLUMNS,
    RELIABILITY_FEATURE_COLUMNS,
    ensure_columns,
    read_table,
    write_table,
)

RELIABILITY_WINDOW_TICKS = 40
STAND_COOLDOWN_TICKS = 12
MAX_COUNT_CLIP = 20.0
MAX_STREAK_CLIP = 20.0
MAX_CONTENTION_CLIP = 8.0
MAX_TIME_SINCE_CONVERSION = 200.0
MAX_DELTA_DIST_CLIP = 24.0
MAX_LOCAL_COUNT_CLIP = 32.0
MAX_LOCAL_RADIUS_CLIP = 32.0
FAR_DIST_THRESHOLD = 8.0
W_ITEM_PROGRESS = 0.18
W_DROP_PROGRESS = 0.24
W_IDLE_FAR = 0.35


def col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name].astype(float)
    return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)


def reward_proxy(frame: pd.DataFrame) -> pd.Series:
    capacity_available = (col(frame, "carrying_count") < col(frame, "capacity")).astype(float)
    drop_progress_gate = (
        (col(frame, "serviceable_dropoff") > 0.5) | (col(frame, "carrying_only_inactive") > 0.5)
    ).astype(float)
    item_progress = np.maximum(col(frame, "delta_dist_to_active_item"), 0.0)
    drop_progress = np.maximum(col(frame, "delta_dist_to_dropoff"), 0.0)
    return (
        col(frame, "delta_score")
        + 0.75 * col(frame, "items_delivered_delta")
        + 3.0 * col(frame, "order_completed_delta")
        + W_ITEM_PROGRESS * item_progress * capacity_available
        + W_DROP_PROGRESS * drop_progress * drop_progress_gate
        - 0.50 * col(frame, "invalid_action_count")
        - 0.05 * col(frame, "is_wait")
        - W_IDLE_FAR * col(frame, "idle_far_from_dropoff")
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
        + 0.08 * col(frame, "preferred_area_match")
        - 0.12 * col(frame, "out_of_area_target")
        - 0.18 * col(frame, "out_of_radius_target")
    )


def n_step_return(values: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for idx in range(len(values)):
        out[idx] = values[idx : idx + n].sum()
    return out


def clip_series(values: pd.Series, max_value: float) -> pd.Series:
    return values.clip(lower=0.0, upper=max_value).astype(float)


def _trim_window(events: Deque[int], tick: int) -> None:
    while events and tick - events[0] > RELIABILITY_WINDOW_TICKS:
        events.popleft()


def compute_contention_proxy(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(np.zeros(0, dtype=float))
    goal_counts = (
        frame[frame["goal_cell_valid"] == 1]
        .groupby(["run_id", "tick", "goal_cell_x", "goal_cell_y"], sort=False)
        .size()
        .rename("goal_claim_count")
    )
    merged = frame.join(
        goal_counts,
        on=["run_id", "tick", "goal_cell_x", "goal_cell_y"],
    )
    goal_proxy = (merged["goal_claim_count"].fillna(1.0).astype(float) - 1.0).clip(lower=0.0)

    pickup_rows = frame[frame["pickup_attempt"] == 1]
    pickup_counts = (
        pickup_rows.groupby(["run_id", "tick", "stand_key"], sort=False)
        .size()
        .rename("pickup_stand_claim_count")
    )
    merged = merged.join(
        pickup_counts,
        on=["run_id", "tick", "stand_key"],
    )
    pickup_proxy = (
        merged["pickup_stand_claim_count"].fillna(1.0).astype(float) - 1.0
    ).clip(lower=0.0)
    return np.maximum(goal_proxy.to_numpy(dtype=float), pickup_proxy.to_numpy(dtype=float))


def compute_reliability_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["stand_failure_count_recent"] = 0.0
    out["stand_success_count_recent"] = 0.0
    out["stand_cooldown_ticks_remaining"] = 0.0
    out["kind_failure_count_recent"] = 0.0
    out["repeated_same_stand_no_delta_streak"] = 0.0
    out["time_since_last_conversion_tick"] = MAX_TIME_SINCE_CONVERSION
    out["last_conversion_was_pickup"] = 0.0
    out["last_conversion_was_dropoff"] = 0.0

    for run_id, run_group in out.groupby("run_id", sort=False):
        stand_fail_events: DefaultDict[str, Deque[int]] = defaultdict(deque)
        stand_success_events: DefaultDict[str, Deque[int]] = defaultdict(deque)
        kind_fail_events: DefaultDict[str, Deque[int]] = defaultdict(deque)
        stand_last_fail_tick: dict[str, int] = {}
        per_bot_last_failed_stand: dict[str, str] = {}
        per_bot_same_stand_fail_streak: dict[str, int] = defaultdict(int)
        last_conversion_tick_by_bot: dict[str, int] = {}
        last_conversion_type_by_bot: dict[str, str] = {}

        for idx in run_group.index:
            tick = int(out.at[idx, "tick"])
            bot_id = str(out.at[idx, "bot_id"])
            stand_key = str(out.at[idx, "stand_key"])
            kind_key = str(out.at[idx, "action_item_kind"])
            pickup_attempt = int(out.at[idx, "pickup_attempt"]) == 1
            pickup_success = int(out.at[idx, "pickup_success"]) == 1
            dropoff_success = int(out.at[idx, "dropoff_success"]) == 1

            stand_fail = stand_fail_events[stand_key]
            stand_succ = stand_success_events[stand_key]
            _trim_window(stand_fail, tick)
            _trim_window(stand_succ, tick)
            out.at[idx, "stand_failure_count_recent"] = float(len(stand_fail))
            out.at[idx, "stand_success_count_recent"] = float(len(stand_succ))

            if stand_key in stand_last_fail_tick:
                age = tick - stand_last_fail_tick[stand_key]
                cooldown = max(0, STAND_COOLDOWN_TICKS - age)
            else:
                cooldown = 0
            out.at[idx, "stand_cooldown_ticks_remaining"] = float(cooldown)

            if kind_key:
                kind_fail = kind_fail_events[kind_key]
                _trim_window(kind_fail, tick)
                out.at[idx, "kind_failure_count_recent"] = float(len(kind_fail))
            else:
                out.at[idx, "kind_failure_count_recent"] = 0.0

            last_tick = last_conversion_tick_by_bot.get(bot_id)
            if last_tick is None:
                out.at[idx, "time_since_last_conversion_tick"] = MAX_TIME_SINCE_CONVERSION
            else:
                out.at[idx, "time_since_last_conversion_tick"] = float(
                    max(0, tick - last_tick)
                )
            last_type = last_conversion_type_by_bot.get(bot_id)
            out.at[idx, "last_conversion_was_pickup"] = 1.0 if last_type == "pickup" else 0.0
            out.at[idx, "last_conversion_was_dropoff"] = 1.0 if last_type == "dropoff" else 0.0
            out.at[idx, "repeated_same_stand_no_delta_streak"] = float(
                per_bot_same_stand_fail_streak.get(bot_id, 0)
            )

            if pickup_attempt:
                if pickup_success:
                    stand_success_events[stand_key].append(tick)
                    per_bot_same_stand_fail_streak[bot_id] = 0
                    per_bot_last_failed_stand.pop(bot_id, None)
                else:
                    stand_fail_events[stand_key].append(tick)
                    stand_last_fail_tick[stand_key] = tick
                    if kind_key:
                        kind_fail_events[kind_key].append(tick)
                    if per_bot_last_failed_stand.get(bot_id) == stand_key:
                        per_bot_same_stand_fail_streak[bot_id] = (
                            per_bot_same_stand_fail_streak.get(bot_id, 0) + 1
                        )
                    else:
                        per_bot_same_stand_fail_streak[bot_id] = 1
                    per_bot_last_failed_stand[bot_id] = stand_key

            if pickup_success:
                last_conversion_tick_by_bot[bot_id] = tick
                last_conversion_type_by_bot[bot_id] = "pickup"
            elif dropoff_success:
                last_conversion_tick_by_bot[bot_id] = tick
                last_conversion_type_by_bot[bot_id] = "dropoff"

    out["stand_failure_count_recent"] = clip_series(out["stand_failure_count_recent"], MAX_COUNT_CLIP)
    out["stand_success_count_recent"] = clip_series(out["stand_success_count_recent"], MAX_COUNT_CLIP)
    out["stand_cooldown_ticks_remaining"] = clip_series(
        out["stand_cooldown_ticks_remaining"], float(STAND_COOLDOWN_TICKS)
    )
    out["kind_failure_count_recent"] = clip_series(out["kind_failure_count_recent"], MAX_COUNT_CLIP)
    out["repeated_same_stand_no_delta_streak"] = clip_series(
        out["repeated_same_stand_no_delta_streak"], MAX_STREAK_CLIP
    )
    out["time_since_last_conversion_tick"] = clip_series(
        out["time_since_last_conversion_tick"], MAX_TIME_SINCE_CONVERSION
    )
    out["last_conversion_was_pickup"] = out["last_conversion_was_pickup"].clip(lower=0.0, upper=1.0)
    out["last_conversion_was_dropoff"] = out["last_conversion_was_dropoff"].clip(
        lower=0.0, upper=1.0
    )
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
    frame = ensure_columns(
        frame,
        [
            "goal_cell_x",
            "goal_cell_y",
            "goal_cell_valid",
            "action_item_kind",
            "carrying_count",
        ],
        default=0.0,
    )
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
        "preferred_area_id",
        "expansion_mode_active",
        "local_active_candidate_count",
        "local_radius",
        "goal_area_id",
    ]:
        if required not in frame.columns:
            frame[required] = 0
    frame["action_item_kind"] = frame["action_item_kind"].fillna("").astype(str)
    frame["goal_cell_x"] = frame["goal_cell_x"].astype(float).fillna(-1).astype(int)
    frame["goal_cell_y"] = frame["goal_cell_y"].astype(float).fillna(-1).astype(int)
    frame["goal_cell_valid"] = frame["goal_cell_valid"].astype(float).fillna(0).astype(int)
    frame["stand_key"] = np.where(
        frame["goal_cell_valid"] == 1,
        frame["goal_cell_x"].astype(str) + ":" + frame["goal_cell_y"].astype(str),
        frame["bot_x"].astype(int).astype(str) + ":" + frame["bot_y"].astype(int).astype(str),
    )
    frame["pickup_attempt"] = (frame["action_kind"] == "pick_up").astype(int)
    frame["dropoff_attempt"] = (frame["action_kind"] == "drop_off").astype(int)
    frame["next_carrying_count"] = frame.groupby(["run_id", "bot_id"], sort=False)[
        "carrying_count"
    ].shift(-1)
    frame["pickup_success"] = (
        (frame["pickup_attempt"] == 1)
        & (frame["next_carrying_count"] > frame["carrying_count"])
    ).astype(int)
    frame["dropoff_success"] = (
        (frame["dropoff_attempt"] == 1)
        & (frame["next_carrying_count"] < frame["carrying_count"])
    ).astype(int)
    frame["prev_dist_to_nearest_active_item"] = frame.groupby(["run_id", "bot_id"], sort=False)[
        "dist_to_nearest_active_item"
    ].shift(1)
    frame["prev_dist_to_dropoff"] = frame.groupby(["run_id", "bot_id"], sort=False)[
        "dist_to_dropoff"
    ].shift(1)
    frame["delta_dist_to_active_item"] = (
        frame["prev_dist_to_nearest_active_item"] - frame["dist_to_nearest_active_item"]
    ).fillna(0.0)
    frame["delta_dist_to_dropoff"] = (
        frame["prev_dist_to_dropoff"] - frame["dist_to_dropoff"]
    ).fillna(0.0)
    frame["delta_dist_to_active_item"] = frame["delta_dist_to_active_item"].clip(
        lower=-MAX_DELTA_DIST_CLIP, upper=MAX_DELTA_DIST_CLIP
    )
    frame["delta_dist_to_dropoff"] = frame["delta_dist_to_dropoff"].clip(
        lower=-MAX_DELTA_DIST_CLIP, upper=MAX_DELTA_DIST_CLIP
    )
    frame["contention_at_stand_proxy"] = compute_contention_proxy(frame)
    frame["contention_at_stand_proxy"] = clip_series(
        frame["contention_at_stand_proxy"], MAX_CONTENTION_CLIP
    )
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
    frame["carrying_only_inactive"] = (
        (frame["carrying_count"] > 0)
        & (frame.get("serviceable_dropoff", 0).astype(float) <= 0.5)
    ).astype(int)
    frame["idle_far_from_dropoff"] = (
        ((frame["is_wait"] == 1) | (frame["noop_move"] == 1))
        & (frame["dist_to_dropoff"] >= FAR_DIST_THRESHOLD)
    ).astype(int)
    frame["preferred_area_id"] = (
        frame.get("preferred_area_id", -1).astype(float).fillna(-1).astype(int)
    )
    frame["goal_area_id"] = frame.get("goal_area_id", -1).astype(float).fillna(-1).astype(int)
    frame.loc[frame["preferred_area_id"] >= 65535, "preferred_area_id"] = -1
    frame.loc[frame["goal_area_id"] >= 65535, "goal_area_id"] = -1
    frame["expansion_mode_active"] = frame.get("expansion_mode_active", 0).astype(float).clip(
        lower=0.0, upper=1.0
    )
    frame["local_active_candidate_count"] = frame.get(
        "local_active_candidate_count", 0
    ).astype(float)
    frame["local_active_candidate_count"] = clip_series(
        frame["local_active_candidate_count"], MAX_LOCAL_COUNT_CLIP
    )
    frame["local_radius"] = frame.get("local_radius", 0).astype(float)
    frame["local_radius"] = clip_series(frame["local_radius"], MAX_LOCAL_RADIUS_CLIP)
    frame["dist_to_goal"] = np.where(
        frame["goal_cell_valid"] == 1,
        np.abs(frame["goal_cell_x"] - frame["bot_x"].astype(int))
        + np.abs(frame["goal_cell_y"] - frame["bot_y"].astype(int)),
        0,
    ).astype(float)
    area_known = (
        (frame["preferred_area_id"] >= 0)
        & (frame["goal_area_id"] >= 0)
        & (frame["goal_cell_valid"] == 1)
    )
    frame["preferred_area_match"] = (
        area_known & (frame["preferred_area_id"] == frame["goal_area_id"])
    ).astype(int)
    frame["out_of_area_target"] = (
        area_known & (frame["preferred_area_id"] != frame["goal_area_id"])
    ).astype(int)
    frame["out_of_radius_target"] = (
        (frame["goal_cell_valid"] == 1)
        & (frame["local_radius"] > 0.0)
        & (frame["dist_to_goal"] > frame["local_radius"])
    ).astype(int)
    # Defragment after the large block of per-column inserts to keep downstream writes fast.
    frame = frame.copy()
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
    wait_reason_values = [
        "blocked_by_vertex_reservation",
        "blocked_by_edge_reservation",
        "forbidden_queue_zone",
        "prohibited_repeat_move",
        "no_path_with_constraints",
        "timeout_fallback",
    ]
    wait_reason_cols = {
        f"wait_reason_{reason}": (frame["wait_reason"] == reason).astype(np.int8)
        for reason in wait_reason_values
    }
    frame = pd.concat(
        [frame, pd.DataFrame(wait_reason_cols, index=frame.index)],
        axis=1,
    )
    frame["reward_proxy"] = reward_proxy(frame)
    frame = compute_reliability_features(frame)
    frame = ensure_columns(frame, CONVERSION_LABEL_COLUMNS, default=0.0)
    frame = ensure_columns(frame, RELIABILITY_FEATURE_COLUMNS, default=0.0)
    frame = ensure_columns(frame, LOCALITY_FEATURE_COLUMNS, default=0.0)

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
