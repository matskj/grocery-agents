from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .common import SCHEMA_VERSION, as_int, detect_mode_from_state, iter_jsonl, write_table


def action_index(actions: List[Dict]) -> Dict[str, Dict]:
    by_bot: Dict[str, Dict] = {}
    for action in actions:
        bot_id = str(action.get("bot_id", ""))
        by_bot[bot_id] = action
    return by_bot


def parse_action(action: Dict) -> Tuple[str, int, int, str, str]:
    kind = str(action.get("kind", "wait"))
    if kind == "move":
        return kind, as_int(action.get("dx"), 0), as_int(action.get("dy"), 0), "", ""
    if kind == "pick_up":
        return kind, 0, 0, str(action.get("item_id", "")), ""
    if kind == "drop_off":
        return kind, 0, 0, "", str(action.get("order_id", ""))
    return "wait", 0, 0, "", ""


def bot_metric_map(summary: Dict, key: str) -> Dict:
    raw = summary.get(key, {})
    if isinstance(raw, dict):
        return raw
    return {}


def bool_flag(value: object) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0
    if isinstance(value, str):
        return 1 if value.lower() in {"1", "true", "yes"} else 0
    return 0


def nearest_item_distance(bot: Dict, items: List[Dict], needed: set[str]) -> float:
    bx = as_int(bot.get("x"), 0)
    by = as_int(bot.get("y"), 0)
    candidates = []
    for item in items:
        item_id = str(item.get("id", ""))
        kind = str(item.get("kind", ""))
        if needed and item_id not in needed and kind not in needed:
            continue
        ix = as_int(item.get("x"), 0)
        iy = as_int(item.get("y"), 0)
        candidates.append(abs(ix - bx) + abs(iy - by))
    if not candidates:
        return 99.0
    return float(min(candidates))


def nearest_dropoff_distance(bot: Dict, dropoffs: List[List[int]]) -> float:
    if not dropoffs:
        return 99.0
    bx = as_int(bot.get("x"), 0)
    by = as_int(bot.get("y"), 0)
    return float(min(abs(as_int(tile[0], 0) - bx) + abs(as_int(tile[1], 0) - by) for tile in dropoffs))


def build_rows(logs_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    run_meta: Dict[str, Dict] = {}
    prev_positions: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)
    blocked_ticks: Dict[str, Dict[str, int]] = defaultdict(dict)

    for path in sorted(logs_dir.glob("run-*.jsonl")):
        fallback_run_id = path.stem
        for record in iter_jsonl(path):
            event = str(record.get("event", ""))
            data = record.get("data", {})
            if not isinstance(data, dict):
                continue
            run_id = str(record.get("run_id") or data.get("run_id") or fallback_run_id)

            if event == "session_start":
                run_meta[run_id] = {
                    "map_id": data.get("map_id"),
                    "difficulty": data.get("difficulty"),
                    "team_id": data.get("team_id"),
                    "build_version": data.get("build_version"),
                    "schema_version": data.get("schema_version", record.get("schema_version", SCHEMA_VERSION)),
                }
                continue

            if event != "tick":
                continue

            state = data.get("game_state", {})
            if not isinstance(state, dict):
                continue
            actions = data.get("actions", [])
            actions = actions if isinstance(actions, list) else []
            team_summary = data.get("team_summary", {})
            team_summary = team_summary if isinstance(team_summary, dict) else {}
            tick_outcome = data.get("tick_outcome", {})
            tick_outcome = tick_outcome if isinstance(tick_outcome, dict) else {}

            bots = state.get("bots", [])
            bots = bots if isinstance(bots, list) else []
            items = state.get("items", [])
            items = items if isinstance(items, list) else []
            orders = state.get("orders", [])
            orders = orders if isinstance(orders, list) else []
            grid = state.get("grid", {})
            grid = grid if isinstance(grid, dict) else {}
            dropoffs = grid.get("drop_off_tiles", [])
            dropoffs = dropoffs if isinstance(dropoffs, list) else []
            item_cells = {
                (as_int(item.get("x"), 0), as_int(item.get("y"), 0))
                for item in items
                if isinstance(item, dict)
            }
            wall_cells = {
                (as_int(wall[0], 0), as_int(wall[1], 0))
                for wall in (grid.get("walls", []) or [])
                if isinstance(wall, list) and len(wall) >= 2
            }

            mode = str(data.get("mode") or detect_mode_from_state(state))
            tick = as_int(data.get("tick", state.get("tick")), 0)
            score = as_int(state.get("score"), 0)
            active_order_index = as_int(state.get("active_order_index"), 0)
            action_map = action_index(actions)

            active_needed = {
                str(order.get("item_id", ""))
                for order in orders
                if str(order.get("status", "")) == "in_progress"
            }
            order_urgency = float(len(active_needed))
            queue_roles = bot_metric_map(team_summary, "queue_roles")
            queue_slot_index_by_bot = bot_metric_map(team_summary, "queue_slot_index_by_bot")
            queue_distance_by_bot = bot_metric_map(team_summary, "queue_distance_by_bot")
            queue_violation_by_bot = bot_metric_map(team_summary, "is_queue_violation_by_bot")
            near_dropoff_blocking_by_bot = bot_metric_map(team_summary, "near_dropoff_blocking_by_bot")
            repeated_failed_move_count_by_bot = bot_metric_map(
                team_summary, "repeated_failed_move_count_by_bot"
            )
            conflict_degree_by_bot = bot_metric_map(team_summary, "conflict_degree_by_bot")
            yield_applied_by_bot = bot_metric_map(team_summary, "yield_applied_by_bot")
            wait_reason_by_bot = bot_metric_map(team_summary, "wait_reason_by_bot")
            intent_move_but_wait_by_bot = bot_metric_map(team_summary, "intent_move_but_wait_by_bot")
            queue_relaxation_active_by_bot = bot_metric_map(
                team_summary, "queue_relaxation_active_by_bot"
            )
            planner_fallback_stage_by_bot = bot_metric_map(
                team_summary, "planner_fallback_stage_by_bot"
            )
            in_corner_by_bot = bot_metric_map(team_summary, "in_corner_by_bot")
            dead_end_depth_by_bot = bot_metric_map(team_summary, "dead_end_depth_by_bot")
            escape_macro_active_by_bot = bot_metric_map(
                team_summary, "escape_macro_active_by_bot"
            )
            escape_macro_ticks_remaining_by_bot = bot_metric_map(
                team_summary, "escape_macro_ticks_remaining_by_bot"
            )
            queue_eta_rank_by_bot = bot_metric_map(team_summary, "queue_eta_rank_by_bot")
            local_conflict_count_by_bot = bot_metric_map(
                team_summary, "local_conflict_count_by_bot"
            )
            cbs_timeout = bool_flag(team_summary.get("cbs_timeout", False))
            cbs_expanded_nodes = as_int(team_summary.get("cbs_expanded_nodes"), 0)
            dropoff_target_status_by_bot = bot_metric_map(
                team_summary, "dropoff_target_status_by_bot"
            )
            dropoff_attempt_streak_by_bot = bot_metric_map(
                team_summary, "dropoff_attempt_same_order_streak_by_bot"
            )
            dropoff_watchdog_triggered_by_bot = bot_metric_map(
                team_summary, "dropoff_watchdog_triggered_by_bot"
            )
            loop_two_cycle_count_by_bot = bot_metric_map(
                team_summary, "loop_two_cycle_count_by_bot"
            )
            coverage_gain_by_bot = bot_metric_map(team_summary, "coverage_gain_by_bot")
            serviceable_dropoff_by_bot = bot_metric_map(
                team_summary, "serviceable_dropoff_by_bot"
            )
            ordering_rank_by_bot = bot_metric_map(team_summary, "ordering_rank_by_bot")
            ordering_score_by_bot = bot_metric_map(team_summary, "ordering_score_by_bot")
            ordering_stage_by_bot = bot_metric_map(team_summary, "ordering_stage_by_bot")

            for bot in bots:
                bot_id = str(bot.get("id", ""))
                bx = as_int(bot.get("x"), 0)
                by = as_int(bot.get("y"), 0)
                carrying = bot.get("carrying", [])
                carrying = carrying if isinstance(carrying, list) else []
                capacity = as_int(bot.get("capacity"), 3)
                inventory_util = float(len(carrying)) / float(max(capacity, 1))

                prev = prev_positions[run_id].get(bot_id)
                if prev == (bx, by):
                    blocked = blocked_ticks[run_id].get(bot_id, 0) + 1
                else:
                    blocked = 0
                blocked_ticks[run_id][bot_id] = blocked
                prev_positions[run_id][bot_id] = (bx, by)

                local_congestion = 0
                teammate_distances = []
                for other in bots:
                    other_id = str(other.get("id", ""))
                    if other_id == bot_id:
                        continue
                    ox = as_int(other.get("x"), 0)
                    oy = as_int(other.get("y"), 0)
                    d = abs(ox - bx) + abs(oy - by)
                    if d <= 1:
                        local_congestion += 1
                    teammate_distances.append(float(d))
                teammate_proximity = (
                    sum(teammate_distances) / len(teammate_distances) if teammate_distances else 0.0
                )

                action = action_map.get(bot_id, {"kind": "wait", "bot_id": bot_id})
                action_kind, action_dx, action_dy, action_item_id, action_order_id = parse_action(action)
                move_target_x = bx + action_dx if action_kind == "move" else bx
                move_target_y = by + action_dy if action_kind == "move" else by
                move_target_in_bounds = (
                    1
                    if 0 <= move_target_x < as_int(grid.get("width"), 0)
                    and 0 <= move_target_y < as_int(grid.get("height"), 0)
                    else 0
                )
                move_target_is_item = (
                    1 if action_kind == "move" and (move_target_x, move_target_y) in item_cells else 0
                )
                move_target_is_wall = (
                    1 if action_kind == "move" and (move_target_x, move_target_y) in wall_cells else 0
                )
                queue_role = str(queue_roles.get(bot_id, "idle"))
                queue_slot_index = as_int(queue_slot_index_by_bot.get(bot_id), -1)
                queue_distance = as_int(queue_distance_by_bot.get(bot_id), -1)
                is_queue_violation = bool_flag(queue_violation_by_bot.get(bot_id, False))
                near_dropoff_blocking = bool_flag(
                    near_dropoff_blocking_by_bot.get(bot_id, False)
                )
                repeated_failed_move_count = as_int(
                    repeated_failed_move_count_by_bot.get(bot_id), 0
                )
                conflict_degree = as_int(conflict_degree_by_bot.get(bot_id), 0)
                yield_applied = bool_flag(yield_applied_by_bot.get(bot_id, False))
                wait_reason = str(wait_reason_by_bot.get(bot_id, "intent_wait"))
                intent_move_but_wait = bool_flag(
                    intent_move_but_wait_by_bot.get(bot_id, False)
                )
                queue_relaxation_active = bool_flag(
                    queue_relaxation_active_by_bot.get(bot_id, False)
                )
                planner_fallback_stage = str(
                    planner_fallback_stage_by_bot.get(bot_id, "none")
                )
                in_corner = bool_flag(in_corner_by_bot.get(bot_id, False))
                dead_end_depth = as_int(dead_end_depth_by_bot.get(bot_id), 0)
                escape_macro_active = bool_flag(
                    escape_macro_active_by_bot.get(bot_id, False)
                )
                escape_macro_ticks_remaining = as_int(
                    escape_macro_ticks_remaining_by_bot.get(bot_id), 0
                )
                queue_eta_rank = as_int(queue_eta_rank_by_bot.get(bot_id), 99)
                local_conflict_count = as_int(local_conflict_count_by_bot.get(bot_id), 0)
                dropoff_target_status = str(
                    dropoff_target_status_by_bot.get(bot_id, "none")
                )
                dropoff_attempt_same_order_streak = as_int(
                    dropoff_attempt_streak_by_bot.get(bot_id), 0
                )
                dropoff_watchdog_triggered = bool_flag(
                    dropoff_watchdog_triggered_by_bot.get(bot_id, False)
                )
                loop_two_cycle_count = as_int(
                    loop_two_cycle_count_by_bot.get(bot_id), 0
                )
                coverage_gain = as_int(coverage_gain_by_bot.get(bot_id), 0)
                serviceable_dropoff = bool_flag(
                    serviceable_dropoff_by_bot.get(bot_id, False)
                )
                ordering_rank = as_int(ordering_rank_by_bot.get(bot_id), -1)
                ordering_score = float(ordering_score_by_bot.get(bot_id, 0.0) or 0.0)
                ordering_stage = str(ordering_stage_by_bot.get(bot_id, "none"))

                meta = run_meta.get(run_id, {})
                row = {
                    "schema_version": record.get("schema_version", meta.get("schema_version", SCHEMA_VERSION)),
                    "run_id": run_id,
                    "mode": mode,
                    "map_id": meta.get("map_id"),
                    "difficulty": meta.get("difficulty"),
                    "team_id": meta.get("team_id"),
                    "build_version": meta.get("build_version"),
                    "tick": tick,
                    "bot_id": bot_id,
                    "grid_width": as_int(grid.get("width"), 0),
                    "grid_height": as_int(grid.get("height"), 0),
                    "bot_count": len(bots),
                    "bot_x": bx,
                    "bot_y": by,
                    "carrying_count": len(carrying),
                    "capacity": capacity,
                    "inventory_util": inventory_util,
                    "dist_to_nearest_active_item": nearest_item_distance(bot, items, active_needed),
                    "dist_to_dropoff": nearest_dropoff_distance(bot, dropoffs),
                    "local_congestion": float(local_congestion),
                    "teammate_proximity": teammate_proximity,
                    "order_urgency": order_urgency,
                    "blocked_ticks": float(blocked),
                    "queue_role": queue_role,
                    "queue_slot_index": float(queue_slot_index),
                    "queue_distance": float(queue_distance if queue_distance >= 0 else 99),
                    "is_queue_violation": is_queue_violation,
                    "near_dropoff_blocking": near_dropoff_blocking,
                    "repeated_failed_move_count": float(repeated_failed_move_count),
                    "conflict_degree": float(conflict_degree),
                    "yield_applied": yield_applied,
                    "queue_role_lead": 1 if queue_role == "lead_courier" else 0,
                    "queue_role_courier": 1 if queue_role == "queue_courier" else 0,
                    "queue_role_collector": 1 if queue_role == "collector" else 0,
                    "queue_role_yield": 1 if queue_role == "yield" else 0,
                    "wait_reason": wait_reason,
                    "intent_move_but_wait": intent_move_but_wait,
                    "queue_relaxation_active": queue_relaxation_active,
                    "planner_fallback_stage": planner_fallback_stage,
                    "in_corner": in_corner,
                    "dead_end_depth": float(dead_end_depth),
                    "escape_macro_active": escape_macro_active,
                    "escape_macro_ticks_remaining": float(
                        max(0, escape_macro_ticks_remaining)
                    ),
                    "queue_eta_rank": float(max(0, queue_eta_rank)),
                    "local_conflict_count": float(local_conflict_count),
                    "cbs_timeout": cbs_timeout,
                    "cbs_expanded_nodes": float(max(0, cbs_expanded_nodes)),
                    "dropoff_target_status": dropoff_target_status,
                    "dropoff_attempt_same_order_streak": float(
                        max(0, dropoff_attempt_same_order_streak)
                    ),
                    "dropoff_watchdog_triggered": dropoff_watchdog_triggered,
                    "loop_two_cycle_count": float(max(0, loop_two_cycle_count)),
                    "coverage_gain": float(max(0, coverage_gain)),
                    "serviceable_dropoff": serviceable_dropoff,
                    "ordering_rank": float(ordering_rank),
                    "ordering_score": ordering_score,
                    "ordering_stage": ordering_stage,
                    "action_kind": action_kind,
                    "action_dx": action_dx,
                    "action_dy": action_dy,
                    "action_item_id": action_item_id,
                    "action_order_id": action_order_id,
                    "attempted_move": 1 if action_kind == "move" else 0,
                    "move_target_x": move_target_x,
                    "move_target_y": move_target_y,
                    "move_target_in_bounds": move_target_in_bounds,
                    "move_target_is_item": move_target_is_item,
                    "move_target_is_wall": move_target_is_wall,
                    "move_target_blocked": 1 if (move_target_is_item or move_target_is_wall) else 0,
                    "is_wait": 1 if action_kind == "wait" else 0,
                    "score": score,
                    "active_order_index": active_order_index,
                    "delta_score": as_int(tick_outcome.get("delta_score"), 0),
                    "items_delivered_delta": as_int(tick_outcome.get("items_delivered_delta"), 0),
                    "order_completed_delta": as_int(tick_outcome.get("order_completed_delta"), 0),
                    "invalid_action_count": as_int(tick_outcome.get("invalid_action_count"), 0),
                    "dropoff_congestion": as_int(team_summary.get("dropoff_congestion"), 0),
                    "blocked_bot_count": as_int(team_summary.get("blocked_bot_count"), 0),
                    "stuck_bot_count": as_int(team_summary.get("stuck_bot_count"), 0),
                    "wait_action_count": as_int(team_summary.get("wait_action_count"), 0),
                    "non_wait_action_count": as_int(team_summary.get("non_wait_action_count"), 0),
                    "state_json": json.dumps(state, separators=(",", ":"), sort_keys=True),
                }
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract training rows from JSONL game logs.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--out", default="data/runs.parquet")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(logs_dir)
    frame = pd.DataFrame(rows)
    written = write_table(frame, out_path)
    print(f"wrote {len(frame)} rows to {written}")


if __name__ == "__main__":
    main()
