from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence

import pandas as pd


SCHEMA_VERSION = "1.3.0"
CONVERSION_LABEL_COLUMNS = [
    "pickup_attempt",
    "pickup_success",
    "dropoff_attempt",
    "dropoff_success",
]
RELIABILITY_FEATURE_COLUMNS = [
    "stand_failure_count_recent",
    "stand_success_count_recent",
    "stand_cooldown_ticks_remaining",
    "kind_failure_count_recent",
    "repeated_same_stand_no_delta_streak",
    "contention_at_stand_proxy",
    "time_since_last_conversion_tick",
    "last_conversion_was_pickup",
    "last_conversion_was_dropoff",
]
LOCALITY_FEATURE_COLUMNS = [
    "preferred_area_match",
    "expansion_mode_active",
    "local_active_candidate_count",
    "local_radius",
    "out_of_area_target",
    "out_of_radius_target",
]
FEATURE_COLUMNS = [
    "dist_to_nearest_active_item",
    "dist_to_dropoff",
    "delta_dist_to_active_item",
    "delta_dist_to_dropoff",
    "inventory_util",
    "carrying_only_inactive",
    "idle_far_from_dropoff",
    "local_congestion",
    "teammate_proximity",
    "order_urgency",
    "blocked_ticks",
    "queue_distance",
    "queue_slot_index",
    "is_queue_violation",
    "near_dropoff_blocking",
    "repeated_failed_move_count",
    "conflict_degree",
    "yield_applied",
    "queue_role_lead",
    "queue_role_courier",
    "queue_role_collector",
    "queue_role_yield",
    "queue_advance",
    "move_target_blocked",
    "noop_move",
    "intent_move_but_wait",
    "queue_relaxation_active",
    "in_corner",
    "dead_end_depth",
    "escape_macro_active",
    "escape_macro_ticks_remaining",
    "queue_eta_rank",
    "local_conflict_count",
    "cbs_timeout",
    "cbs_expanded_nodes",
    "dropoff_attempt_same_order_streak",
    "dropoff_watchdog_triggered",
    "loop_two_cycle_count",
    "coverage_gain",
    "serviceable_dropoff",
    "ordering_rank",
    "ordering_score",
    "dropoff_target_pending",
    "dropoff_target_in_progress",
    "wait_reason_blocked_by_vertex_reservation",
    "wait_reason_blocked_by_edge_reservation",
    "wait_reason_forbidden_queue_zone",
    "wait_reason_prohibited_repeat_move",
    "wait_reason_no_path_with_constraints",
    "wait_reason_timeout_fallback",
    *RELIABILITY_FEATURE_COLUMNS,
    *LOCALITY_FEATURE_COLUMNS,
]

RUNTIME_FEATURE_COLUMNS = [
    "dist_to_nearest_active_item",
    "dist_to_dropoff",
    "inventory_util",
    "local_congestion",
    "teammate_proximity",
    "order_urgency",
    "blocked_ticks",
    "queue_distance",
    "local_conflict_count",
    "queue_role_lead",
    "queue_role_courier",
    "queue_role_collector",
    "queue_role_yield",
    "serviceable_dropoff",
    "stand_cooldown_ticks_remaining",
    "contention_at_stand_proxy",
    "time_since_last_conversion_tick",
    "last_conversion_was_pickup",
    "last_conversion_was_dropoff",
    "preferred_area_match",
    "expansion_mode_active",
    "local_active_candidate_count",
    "local_radius",
    "out_of_area_target",
    "out_of_radius_target",
]

ORDERING_FEATURE_COLUMNS = [
    "carrying_active",
    "queue_role_lead",
    "queue_role_courier",
    "blocked_ticks",
    "local_conflict_count",
    "dist_to_goal_proxy",
    "dropoff_watchdog_pressure",
    "choke_occupancy_proxy",
]

ORDERING_SEQUENCE_FEATURE_COLUMNS = [
    "carrying_active",
    "queue_role_lead",
    "queue_role_courier",
    "blocked_ticks",
    "local_conflict_count",
    "dist_to_goal",
    "dropoff_watchdog_pressure",
    "choke_occupancy",
]

ACTION_SIGNATURE_COLUMNS = [
    "tick",
    "bot_id",
    "action_kind",
    "action_dx",
    "action_dy",
    "action_item_kind",
    "action_order_id",
    "goal_cell_x",
    "goal_cell_y",
]

STATE_ACTION_SIGNATURE_EXTRA_COLUMNS = [
    "bot_x",
    "bot_y",
    "carrying_count",
]


def ensure_columns(frame: pd.DataFrame, columns: Sequence[str], default: float = 0.0) -> pd.DataFrame:
    for col in columns:
        if col not in frame.columns:
            frame[col] = default
    return frame


def build_run_signature_map(frame: pd.DataFrame, signature_kind: str = "action") -> Dict[str, str]:
    if frame.empty or "run_id" not in frame.columns:
        return {}

    normalized_kind = signature_kind.strip().lower()
    if normalized_kind not in {"action", "state_action"}:
        normalized_kind = "action"

    signature_columns = list(ACTION_SIGNATURE_COLUMNS)
    if normalized_kind == "state_action":
        signature_columns.extend(STATE_ACTION_SIGNATURE_EXTRA_COLUMNS)
    context_columns = ["mode", "map_id", "grid_width", "grid_height", "bot_count"]
    all_columns = context_columns + signature_columns

    work = ensure_columns(frame.copy(), [*all_columns, "tick", "bot_id"], default=0.0)
    work = work.sort_values(["run_id", "tick", "bot_id"], kind="mergesort")

    numeric_columns = {
        "tick",
        "action_dx",
        "action_dy",
        "goal_cell_x",
        "goal_cell_y",
        "bot_x",
        "bot_y",
        "carrying_count",
        "grid_width",
        "grid_height",
        "bot_count",
    }
    for col in all_columns:
        if col in numeric_columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)
        else:
            work[col] = work[col].fillna("").astype(str)

    out: Dict[str, str] = {}
    for run_id, group in work.groupby("run_id", sort=False):
        hasher = hashlib.blake2b(digest_size=16)
        for row in group[all_columns].itertuples(index=False, name=None):
            hasher.update("|".join(str(v) for v in row).encode("utf-8"))
            hasher.update(b"\n")
        out[str(run_id)] = hasher.hexdigest()
    return out


def signature_cluster_sizes(signature_by_run: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for signature in signature_by_run.values():
        counts[signature] = counts.get(signature, 0) + 1
    return counts


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def detect_mode_from_state(state: Dict) -> str:
    bots = state.get("bots", [])
    grid = state.get("grid", {})
    width = int(grid.get("width", 0) or 0)
    height = int(grid.get("height", 0) or 0)
    count = len(bots)
    if (count, width, height) == (1, 12, 10):
        return "easy"
    if (count, width, height) == (3, 16, 12):
        return "medium"
    if (count, width, height) == (5, 22, 14):
        return "hard"
    if (count, width, height) == (10, 28, 18):
        return "expert"
    return "custom"


def as_int(value: Optional[object], default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def avg(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    if not values:
        return default
    return sum(values) / len(values)


def write_table(frame: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        frame.to_csv(out_path, index=False)
        return out_path
    try:
        frame.to_parquet(out_path, index=False)
        return out_path
    except (ImportError, ModuleNotFoundError):
        fallback = out_path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        return fallback


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except (ImportError, ModuleNotFoundError):
            fallback = path.with_suffix(".csv")
            if fallback.exists():
                return pd.read_csv(fallback, low_memory=False)
            raise
    fallback = path.with_suffix(".csv")
    if fallback.exists():
        return pd.read_csv(fallback, low_memory=False)
    raise FileNotFoundError(path)
