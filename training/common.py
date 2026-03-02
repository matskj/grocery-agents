from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import pandas as pd


SCHEMA_VERSION = "1.1.0"
FEATURE_COLUMNS = [
    "dist_to_nearest_active_item",
    "dist_to_dropoff",
    "inventory_util",
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
