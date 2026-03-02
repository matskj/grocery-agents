use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::OnceLock,
};

use tracing::info;

use crate::{
    dispatcher::{BotIntent, Dispatcher, Intent},
    dist::DistanceMap,
    model::{Action, GameState},
    motion::{MotionPlanner, PlannedAction},
    scoring::{detect_mode_label, maybe_score_ordering, OrderingFeatures},
    team_context::{BlockedMove, BotRole, StickyQueueRole, TeamContext, TeamContextConfig},
    world::World,
};

#[derive(Debug, Default)]
struct BotMemory {
    prev_cell: Option<u16>,
    blocked_ticks: u8,
    repeated_failed_moves: u8,
    failed_move_history: HashMap<(u16, i32, i32), u8>,
    last_move: Option<(u16, i32, i32)>,
    constraint_relax_ticks_remaining: u8,
    escape_macro_goal: Option<u16>,
    escape_macro_ticks_remaining: u8,
    recent_cells: VecDeque<u16>,
    loop_two_cycle_count: u16,
    coverage_gain_last: u16,
    last_dropoff_order_id: Option<String>,
    same_dropoff_order_streak: u8,
    last_successful_drop_tick: Option<u64>,
    dropoff_ban_ticks_by_order: HashMap<String, u8>,
    prev_carrying: Vec<String>,
    dropoff_watchdog_triggered: bool,
}

#[derive(Debug)]
pub struct Policy {
    dispatcher: Dispatcher,
    planner: MotionPlanner,
    memory: HashMap<String, BotMemory>,
    sticky_roles: HashMap<String, StickyQueueRole>,
    last_team_telemetry: serde_json::Value,
}

impl Policy {
    pub fn new() -> Self {
        Self {
            dispatcher: Dispatcher::new(),
            planner: MotionPlanner::new(16),
            memory: HashMap::new(),
            sticky_roles: HashMap::new(),
            last_team_telemetry: serde_json::json!({}),
        }
    }

    pub fn decide_round(&mut self, state: &GameState) -> Vec<Action> {
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let active_items = active_item_set(state);

        self.update_memory(state, map);
        let blocked_snapshot = self
            .memory
            .iter()
            .map(|(bot_id, mem)| (bot_id.clone(), mem.blocked_ticks))
            .collect::<HashMap<_, _>>();
        let repeated_snapshot = self
            .memory
            .iter()
            .map(|(bot_id, mem)| (bot_id.clone(), mem.repeated_failed_moves))
            .collect::<HashMap<_, _>>();
        let mut relax_snapshot = self
            .memory
            .iter()
            .map(|(bot_id, mem)| (bot_id.clone(), mem.constraint_relax_ticks_remaining))
            .collect::<HashMap<_, _>>();
        let mut escape_snapshot = self
            .memory
            .iter()
            .map(|(bot_id, mem)| (bot_id.clone(), mem.escape_macro_ticks_remaining))
            .collect::<HashMap<_, _>>();
        let recent_cell_visits_team = self.recent_cell_visits_team();
        let mut dropoff_attempt_streak_snapshot = self
            .memory
            .iter()
            .map(|(bot_id, mem)| (bot_id.clone(), mem.same_dropoff_order_streak))
            .collect::<HashMap<_, _>>();
        let loop_two_cycle_snapshot = self
            .memory
            .iter()
            .map(|(bot_id, mem)| (bot_id.clone(), mem.loop_two_cycle_count))
            .collect::<HashMap<_, _>>();
        let coverage_gain_snapshot = self
            .memory
            .iter()
            .map(|(bot_id, mem)| (bot_id.clone(), mem.coverage_gain_last))
            .collect::<HashMap<_, _>>();
        let mut dropoff_watchdog_snapshot = self
            .memory
            .iter()
            .map(|(bot_id, mem)| (bot_id.clone(), mem.dropoff_watchdog_triggered))
            .collect::<HashMap<_, _>>();
        let prohibited_moves = build_prohibited_move_map(&self.memory);

        let mode = detect_mode_label(state);
        let strict_queue = queue_strict_mode(mode);
        let cfg = TeamContextConfig {
            strict_queue,
            max_ring_entrants: queue_max_ring_entrants(),
            hold_ticks: 3,
            control_zone_radius: 4,
        };
        let mut team_ctx = TeamContext::build(
            state,
            map,
            &dist,
            &blocked_snapshot,
            &repeated_snapshot,
            &relax_snapshot,
            &escape_snapshot,
            &recent_cell_visits_team,
            &dropoff_attempt_streak_snapshot,
            &loop_two_cycle_snapshot,
            &coverage_gain_snapshot,
            &dropoff_watchdog_snapshot,
            &prohibited_moves,
            &self.sticky_roles,
            cfg,
        );

        let mut relax_changed = false;
        let mut escape_changed = false;
        let escape_ticks = deadlock_escape_ticks();
        for bot in &state.bots {
            let role = team_ctx.role_for(&bot.id);
            let blocked = blocked_snapshot.get(&bot.id).copied().unwrap_or(0);
            let repeated = repeated_snapshot.get(&bot.id).copied().unwrap_or(0);
            if !matches!(role, BotRole::LeadCourier | BotRole::QueueCourier)
                && (blocked >= escape_ticks || repeated >= 2)
            {
                let mem = self.memory.entry(bot.id.clone()).or_default();
                if mem.constraint_relax_ticks_remaining == 0 {
                    mem.constraint_relax_ticks_remaining = 2;
                    relax_snapshot.insert(bot.id.clone(), 2);
                    relax_changed = true;
                }
            }

            let mem = self.memory.entry(bot.id.clone()).or_default();
            if mem.escape_macro_ticks_remaining == 0 {
                let Some(cell) = map.idx(bot.x, bot.y) else {
                    continue;
                };
                let dead_end = team_ctx
                    .dead_end_depth_by_idx
                    .get(cell as usize)
                    .copied()
                    .unwrap_or(0);
                let in_corner = team_ctx.corner_cells.contains(&cell);
                if blocked >= 2 && (in_corner || dead_end >= 1) {
                    if let Some(junction) = nearest_junction(cell, map, &team_ctx.junction_cells) {
                        mem.escape_macro_goal = Some(junction);
                        mem.escape_macro_ticks_remaining = 4;
                        escape_snapshot.insert(bot.id.clone(), 4);
                        escape_changed = true;
                    }
                }
            }
        }

        if relax_changed || escape_changed {
            team_ctx = TeamContext::build(
                state,
                map,
                &dist,
                &blocked_snapshot,
                &repeated_snapshot,
                &relax_snapshot,
                &escape_snapshot,
                &recent_cell_visits_team,
                &dropoff_attempt_streak_snapshot,
                &loop_two_cycle_snapshot,
                &coverage_gain_snapshot,
                &dropoff_watchdog_snapshot,
                &prohibited_moves,
                &self.sticky_roles,
                cfg,
            );
        }
        self.sticky_roles = team_ctx.queue.next_sticky.clone();

        let intents = self
            .dispatcher
            .build_intents(state, map, &dist, &blocked_snapshot, &team_ctx);
        let mut goals = HashMap::new();
        let mut immediate = HashMap::new();
        let mut intent_labels = serde_json::Map::new();
        let mut intent_move_to_by_bot: HashMap<String, bool> = HashMap::new();
        let order_status_by_id = state
            .orders
            .iter()
            .map(|order| (order.id.clone(), order.status))
            .collect::<HashMap<_, _>>();
        let order_item_by_id = state
            .orders
            .iter()
            .map(|order| (order.id.clone(), order.item_id.clone()))
            .collect::<HashMap<_, _>>();
        let mut dropoff_target_status_by_bot = serde_json::Map::new();
        let mut serviceable_dropoff_by_bot = serde_json::Map::new();

        for BotIntent { bot_id, intent } in intents {
            let bot = match state.bots.iter().find(|b| b.id == bot_id) {
                Some(b) => b,
                None => continue,
            };
            let role = team_ctx.role_for(&bot.id);
            let cell = map.idx(bot.x, bot.y).unwrap_or(0);
            let mem = self.memory.entry(bot.id.clone()).or_default();
            mem.dropoff_watchdog_triggered = false;
            let queue_goal = team_ctx.queue_goal_for(&bot.id);
            serviceable_dropoff_by_bot.insert(
                bot.id.clone(),
                serde_json::Value::Bool(
                    team_ctx
                        .dropoff_serviceable_by_bot
                        .get(&bot.id)
                        .copied()
                        .unwrap_or(false),
                ),
            );
            if bot_debug_enabled() {
                info!(
                    tick = state.tick,
                    bot_id = %bot.id,
                    role = role.as_str(),
                    intent = ?intent,
                    blocked_ticks = mem.blocked_ticks,
                    "assignment selected"
                );
            }

            intent_labels.insert(
                bot.id.clone(),
                serde_json::Value::String(format!("{intent:?}")),
            );
            intent_move_to_by_bot.insert(bot.id.clone(), matches!(intent, Intent::MoveTo { .. }));

            if mem.escape_macro_ticks_remaining > 0 {
                let dead_end = team_ctx
                    .dead_end_depth_by_idx
                    .get(cell as usize)
                    .copied()
                    .unwrap_or(0);
                let in_corner = team_ctx.corner_cells.contains(&cell);
                let exit_reached = !in_corner && dead_end == 0;
                if exit_reached {
                    mem.escape_macro_ticks_remaining = 0;
                    mem.escape_macro_goal = None;
                } else if let Some(goal) = mem.escape_macro_goal {
                    goals.insert(bot.id.clone(), goal);
                    intent_move_to_by_bot.insert(bot.id.clone(), true);
                    intent_labels.insert(
                        bot.id.clone(),
                        serde_json::Value::String("EscapeMacro".to_owned()),
                    );
                    continue;
                }
            }

            match intent {
                Intent::DropOff { order_id } => {
                    let status_label = match order_status_by_id.get(&order_id).copied() {
                        Some(crate::model::OrderStatus::InProgress) => "in_progress",
                        Some(crate::model::OrderStatus::Pending) => "pending",
                        Some(crate::model::OrderStatus::Delivered) => "delivered",
                        Some(crate::model::OrderStatus::Cancelled) => "cancelled",
                        None => "missing",
                    };
                    dropoff_target_status_by_bot.insert(
                        bot.id.clone(),
                        serde_json::Value::String(status_label.to_owned()),
                    );
                    let order_item = order_item_by_id.get(&order_id).cloned();
                    let carrying_required = order_item
                        .as_ref()
                        .map(|item| bot.carrying.iter().any(|c| c == item))
                        .unwrap_or(false);
                    let in_progress = matches!(
                        order_status_by_id.get(&order_id).copied(),
                        Some(crate::model::OrderStatus::InProgress)
                    );
                    let banned = mem
                        .dropoff_ban_ticks_by_order
                        .get(&order_id)
                        .copied()
                        .unwrap_or(0)
                        > 0;
                    if mem.last_dropoff_order_id.as_deref() == Some(order_id.as_str()) {
                        mem.same_dropoff_order_streak = mem.same_dropoff_order_streak.saturating_add(1);
                    } else {
                        mem.last_dropoff_order_id = Some(order_id.clone());
                        mem.same_dropoff_order_streak = 1;
                    }
                    dropoff_attempt_streak_snapshot
                        .insert(bot.id.clone(), mem.same_dropoff_order_streak);
                    let watchdog_trigger = banned
                        || should_trigger_dropoff_watchdog(
                            in_progress,
                            carrying_required,
                            mem.same_dropoff_order_streak,
                        );
                    if watchdog_trigger {
                        mem.dropoff_watchdog_triggered = true;
                        dropoff_watchdog_snapshot.insert(bot.id.clone(), true);
                        mem.dropoff_ban_ticks_by_order.insert(
                            order_id.clone(),
                            dropoff_watchdog_ban_ticks(),
                        );
                        if let Some(escape) =
                            pick_deadlock_escape(bot.id.as_str(), state, map, &dist, &team_ctx)
                        {
                            goals.insert(bot.id.clone(), escape);
                        } else {
                            immediate.insert(
                                bot.id.clone(),
                                PlannedAction {
                                    action: Action::wait(bot.id.clone()),
                                    wait_reason: "timeout_fallback",
                                    fallback_stage: "dropoff_watchdog",
                                    ordering_stage: "immediate",
                                    path_preview: Vec::new(),
                                },
                            );
                        }
                        continue;
                    }
                    immediate.insert(
                        bot.id.clone(),
                        PlannedAction {
                            action: Action::DropOff {
                                bot_id: bot.id.clone(),
                                order_id,
                            },
                            wait_reason: "intent_wait",
                            fallback_stage: "immediate",
                            ordering_stage: "immediate",
                            path_preview: Vec::new(),
                        },
                    );
                }
                Intent::PickUp { item_id } => {
                    dropoff_target_status_by_bot.insert(
                        bot.id.clone(),
                        serde_json::Value::String("none".to_owned()),
                    );
                    if bot
                        .carrying
                        .iter()
                        .any(|item| active_items.contains(item.as_str()))
                        && queue_goal.is_some()
                    {
                        goals.insert(bot.id.clone(), queue_goal.unwrap_or(cell));
                    } else if matches!(role, BotRole::QueueCourier | BotRole::LeadCourier) {
                        if let Some(goal) = queue_goal {
                            goals.insert(bot.id.clone(), goal);
                        } else {
                            immediate.insert(
                                bot.id.clone(),
                                PlannedAction {
                                    action: Action::wait(bot.id.clone()),
                                    wait_reason: "intent_wait",
                                    fallback_stage: "immediate",
                                    ordering_stage: "immediate",
                                    path_preview: Vec::new(),
                                },
                            );
                        }
                    } else {
                        immediate.insert(
                            bot.id.clone(),
                            PlannedAction {
                                action: Action::PickUp {
                                    bot_id: bot.id.clone(),
                                    item_id,
                                },
                                wait_reason: "intent_wait",
                                fallback_stage: "immediate",
                                ordering_stage: "immediate",
                                path_preview: Vec::new(),
                            },
                        );
                    }
                }
                Intent::MoveTo { cell: mut goal } => {
                    dropoff_target_status_by_bot.insert(
                        bot.id.clone(),
                        serde_json::Value::String("none".to_owned()),
                    );
                    if let Some(queue_target) = queue_goal {
                        if matches!(role, BotRole::LeadCourier | BotRole::QueueCourier) {
                            goal = queue_target;
                        }
                    }
                    if mem.blocked_ticks >= escape_ticks {
                        if let Some(escape) =
                            pick_deadlock_escape(bot.id.as_str(), state, map, &dist, &team_ctx)
                        {
                            goal = escape;
                        }
                    }
                    goals.insert(bot.id.clone(), goal);
                }
                Intent::Wait => {
                    dropoff_target_status_by_bot.insert(
                        bot.id.clone(),
                        serde_json::Value::String("none".to_owned()),
                    );
                    if let Some(goal) = queue_goal {
                        if goal != cell {
                            goals.insert(bot.id.clone(), goal);
                        } else {
                            immediate.insert(
                                bot.id.clone(),
                                PlannedAction {
                                    action: Action::wait(bot.id.clone()),
                                    wait_reason: "intent_wait",
                                    fallback_stage: "immediate",
                                    ordering_stage: "immediate",
                                    path_preview: Vec::new(),
                                },
                            );
                        }
                    } else if mem.blocked_ticks >= escape_ticks
                        && matches!(role, BotRole::Yield | BotRole::Collector)
                    {
                        if let Some(escape) =
                            pick_deadlock_escape(bot.id.as_str(), state, map, &dist, &team_ctx)
                        {
                            goals.insert(bot.id.clone(), escape);
                        } else {
                            immediate.insert(
                                bot.id.clone(),
                                PlannedAction {
                                    action: Action::wait(bot.id.clone()),
                                    wait_reason: "intent_wait",
                                    fallback_stage: "immediate",
                                    ordering_stage: "immediate",
                                    path_preview: Vec::new(),
                                },
                            );
                        }
                    } else {
                        immediate.insert(
                            bot.id.clone(),
                            PlannedAction {
                                action: Action::wait(bot.id.clone()),
                                wait_reason: "intent_wait",
                                fallback_stage: "immediate",
                                ordering_stage: "immediate",
                                path_preview: Vec::new(),
                            },
                        );
                    }
                }
            }
        }

        let (ordering_sequence, ordering_scores, ordering_ranks) = compute_ordering_sequence(
            state,
            map,
            &dist,
            &goals,
            &team_ctx,
            &self.memory,
            mode,
        );

        let plan_result = self
            .planner
            .plan(state, map, &dist, &goals, &team_ctx.movement, &ordering_sequence);
        let mut planned = plan_result.actions;
        for (bot_id, action) in immediate {
            planned.insert(bot_id, action);
        }
        for bot in &state.bots {
            let Some(mem) = self.memory.get(&bot.id) else {
                continue;
            };
            if mem.escape_macro_ticks_remaining == 0 {
                continue;
            }
            if let Some(entry) = planned.get_mut(&bot.id) {
                if matches!(entry.action, Action::Move { .. }) {
                    entry.fallback_stage = if entry.fallback_stage == "local_sidestep" {
                        "escape_sidestep"
                    } else {
                        "escape_macro"
                    };
                    entry.ordering_stage = "pmat_escape_override";
                }
            }
        }
        apply_swap_loop_breaker(state, map, &team_ctx, &mut planned);
        apply_yield_actions(state, map, &dist, &team_ctx, &self.memory, &mut planned);
        self.capture_last_moves(state, map, &planned);
        for bot in &state.bots {
            let state_entry = team_ctx
                .bot_plan_state_by_bot
                .entry(bot.id.clone())
                .or_default();
            state_entry.goal_cell = goals.get(&bot.id).copied();
            state_entry.path_preview = planned
                .get(&bot.id)
                .map(|p| p.path_preview.clone())
                .unwrap_or_default();
            let mem = self.memory.get(&bot.id);
            state_entry.escape_macro_ticks_remaining =
                mem.map(|m| m.escape_macro_ticks_remaining).unwrap_or(0);
            state_entry.escape_macro_active = state_entry.escape_macro_ticks_remaining > 0;
            state_entry.constraint_relax_ticks_remaining = mem
                .map(|m| m.constraint_relax_ticks_remaining)
                .unwrap_or(0);
        }

        let wait_reason_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let reason = planned
                    .get(&bot.id)
                    .map(|p| p.wait_reason.to_owned())
                    .unwrap_or_else(|| "intent_wait".to_owned());
                (bot.id.clone(), serde_json::Value::String(reason))
            })
            .collect::<serde_json::Map<_, _>>();
        let planner_fallback_stage_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let stage = planned
                    .get(&bot.id)
                    .map(|p| p.fallback_stage.to_owned())
                    .unwrap_or_else(|| "none".to_owned());
                (bot.id.clone(), serde_json::Value::String(stage))
            })
            .collect::<serde_json::Map<_, _>>();
        let ordering_stage_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let stage = planned
                    .get(&bot.id)
                    .map(|p| p.ordering_stage.to_owned())
                    .unwrap_or_else(|| "none".to_owned());
                (bot.id.clone(), serde_json::Value::String(stage))
            })
            .collect::<serde_json::Map<_, _>>();
        let intent_move_but_wait_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let intent_move = intent_move_to_by_bot.get(&bot.id).copied().unwrap_or(false);
                let wait = planned
                    .get(&bot.id)
                    .map(|p| matches!(p.action, Action::Wait { .. }))
                    .unwrap_or(true);
                (bot.id.clone(), serde_json::Value::Bool(intent_move && wait))
            })
            .collect::<serde_json::Map<_, _>>();
        let dropoff_watchdog_triggered_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let triggered = self
                    .memory
                    .get(&bot.id)
                    .map(|m| m.dropoff_watchdog_triggered)
                    .unwrap_or(false);
                (bot.id.clone(), serde_json::Value::Bool(triggered))
            })
            .collect::<serde_json::Map<_, _>>();
        let dropoff_streak_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let streak = self
                    .memory
                    .get(&bot.id)
                    .map(|m| m.same_dropoff_order_streak)
                    .unwrap_or(0);
                (
                    bot.id.clone(),
                    serde_json::Value::Number(serde_json::Number::from(streak as i64)),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let loop_two_cycle_count_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let count = self
                    .memory
                    .get(&bot.id)
                    .map(|m| m.loop_two_cycle_count)
                    .unwrap_or(0);
                (
                    bot.id.clone(),
                    serde_json::Value::Number(serde_json::Number::from(count as i64)),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let coverage_gain_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let gain = self
                    .memory
                    .get(&bot.id)
                    .map(|m| m.coverage_gain_last)
                    .unwrap_or(0);
                (
                    bot.id.clone(),
                    serde_json::Value::Number(serde_json::Number::from(gain as i64)),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let ordering_rank_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let rank = ordering_ranks.get(&bot.id).copied().unwrap_or(u16::MAX);
                (
                    bot.id.clone(),
                    serde_json::Value::Number(serde_json::Number::from(rank as i64)),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let ordering_score_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let score = ordering_scores.get(&bot.id).copied().unwrap_or(0.0);
                (
                    bot.id.clone(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(score)
                            .unwrap_or_else(|| serde_json::Number::from(0)),
                    ),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        for bot in &state.bots {
            dropoff_target_status_by_bot
                .entry(bot.id.clone())
                .or_insert_with(|| serde_json::Value::String("none".to_owned()));
        }

        let mut telemetry = team_ctx.telemetry(map);
        if let Some(obj) = telemetry.as_object_mut() {
            obj.insert("selected_intents".to_owned(), serde_json::Value::Object(intent_labels));
            obj.insert(
                "wait_reason_by_bot".to_owned(),
                serde_json::Value::Object(wait_reason_by_bot),
            );
            obj.insert(
                "intent_move_but_wait_by_bot".to_owned(),
                serde_json::Value::Object(intent_move_but_wait_by_bot),
            );
            obj.insert(
                "planner_fallback_stage_by_bot".to_owned(),
                serde_json::Value::Object(planner_fallback_stage_by_bot),
            );
            obj.insert(
                "ordering_stage_by_bot".to_owned(),
                serde_json::Value::Object(ordering_stage_by_bot),
            );
            obj.insert(
                "cbs_timeout".to_owned(),
                serde_json::Value::Bool(plan_result.diagnostics.cbs_timeout),
            );
            obj.insert(
                "cbs_expanded_nodes".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    plan_result.diagnostics.cbs_expanded_nodes as i64,
                )),
            );
            let local_conflict_count = plan_result
                .diagnostics
                .local_conflict_count_by_bot
                .iter()
                .map(|(bot_id, count)| {
                    (
                        bot_id.clone(),
                        serde_json::Value::Number(serde_json::Number::from(*count as i64)),
                    )
                })
                .collect::<serde_json::Map<_, _>>();
            obj.insert(
                "local_conflict_count_by_bot".to_owned(),
                serde_json::Value::Object(local_conflict_count),
            );
            obj.insert(
                "dropoff_target_status_by_bot".to_owned(),
                serde_json::Value::Object(dropoff_target_status_by_bot),
            );
            obj.insert(
                "dropoff_attempt_same_order_streak_by_bot".to_owned(),
                serde_json::Value::Object(dropoff_streak_by_bot),
            );
            obj.insert(
                "dropoff_watchdog_triggered_by_bot".to_owned(),
                serde_json::Value::Object(dropoff_watchdog_triggered_by_bot),
            );
            obj.insert(
                "loop_two_cycle_count_by_bot".to_owned(),
                serde_json::Value::Object(loop_two_cycle_count_by_bot),
            );
            obj.insert(
                "coverage_gain_by_bot".to_owned(),
                serde_json::Value::Object(coverage_gain_by_bot),
            );
            obj.insert(
                "serviceable_dropoff_by_bot".to_owned(),
                serde_json::Value::Object(serviceable_dropoff_by_bot),
            );
            obj.insert(
                "ordering_rank_by_bot".to_owned(),
                serde_json::Value::Object(ordering_rank_by_bot),
            );
            obj.insert(
                "ordering_score_by_bot".to_owned(),
                serde_json::Value::Object(ordering_score_by_bot),
            );
        }
        self.last_team_telemetry = telemetry;

        state
            .bots
            .iter()
            .map(|bot| {
                planned
                    .remove(&bot.id)
                    .map(|planned| planned.action)
                    .unwrap_or_else(|| Action::wait(bot.id.clone()))
            })
            .collect()
    }

    pub fn last_team_telemetry(&self) -> serde_json::Value {
        self.last_team_telemetry.clone()
    }

    fn recent_cell_visits_team(&self) -> HashMap<u16, u16> {
        let mut heat = HashMap::new();
        for mem in self.memory.values() {
            for &cell in &mem.recent_cells {
                *heat.entry(cell).or_insert(0) += 1;
            }
        }
        heat
    }

    fn update_memory(&mut self, state: &GameState, map: &crate::world::MapCache) {
        for bot in &state.bots {
            let cell = map.idx(bot.x, bot.y).unwrap_or(0);
            let mem = self.memory.entry(bot.id.clone()).or_default();
            mem.constraint_relax_ticks_remaining = mem.constraint_relax_ticks_remaining.saturating_sub(1);
            mem.escape_macro_ticks_remaining = mem.escape_macro_ticks_remaining.saturating_sub(1);
            mem.dropoff_watchdog_triggered = false;
            if mem.escape_macro_ticks_remaining == 0 {
                mem.escape_macro_goal = None;
            }
            for value in mem.dropoff_ban_ticks_by_order.values_mut() {
                *value = value.saturating_sub(1);
            }
            mem.dropoff_ban_ticks_by_order.retain(|_, ticks| *ticks > 0);
            for value in mem.failed_move_history.values_mut() {
                *value = value.saturating_sub(1);
            }
            if mem.prev_cell == Some(cell) {
                mem.blocked_ticks = mem.blocked_ticks.saturating_add(1);
                if let Some((from, dx, dy)) = mem.last_move {
                    let count = mem.failed_move_history.entry((from, dx, dy)).or_insert(0);
                    *count = count.saturating_add(1).min(4);
                }
            } else {
                mem.blocked_ticks = 0;
            }
            let seen_recently = mem.recent_cells.iter().any(|v| *v == cell);
            mem.coverage_gain_last = if seen_recently { 0 } else { 1 };
            mem.recent_cells.push_back(cell);
            while mem.recent_cells.len() > 16 {
                mem.recent_cells.pop_front();
            }
            if mem.recent_cells.len() >= 3 {
                let n = mem.recent_cells.len();
                let a = mem.recent_cells.get(n - 1).copied().unwrap_or(cell);
                let b = mem.recent_cells.get(n - 2).copied().unwrap_or(cell);
                let c = mem.recent_cells.get(n - 3).copied().unwrap_or(cell);
                if a == c && a != b {
                    mem.loop_two_cycle_count = mem.loop_two_cycle_count.saturating_add(1);
                }
            }

            if !mem.prev_carrying.is_empty() && bot.carrying.len() < mem.prev_carrying.len() {
                mem.same_dropoff_order_streak = 0;
                mem.last_dropoff_order_id = None;
                mem.last_successful_drop_tick = Some(state.tick);
                mem.dropoff_ban_ticks_by_order.clear();
            }

            mem.failed_move_history.retain(|_, count| *count > 0);
            mem.repeated_failed_moves = mem.failed_move_history.values().copied().max().unwrap_or(0);
            mem.prev_cell = Some(cell);
            mem.prev_carrying = bot.carrying.clone();
        }
    }

    fn capture_last_moves(
        &mut self,
        state: &GameState,
        map: &crate::world::MapCache,
        planned: &HashMap<String, PlannedAction>,
    ) {
        for bot in &state.bots {
            let mem = self.memory.entry(bot.id.clone()).or_default();
            let start = map.idx(bot.x, bot.y).unwrap_or(0);
            mem.last_move = match planned.get(&bot.id).map(|p| &p.action) {
                Some(Action::Move { dx, dy, .. }) => Some((start, *dx, *dy)),
                _ => None,
            };
        }
    }
}

fn build_prohibited_move_map(memory: &HashMap<String, BotMemory>) -> HashMap<String, Vec<BlockedMove>> {
    let mut out = HashMap::new();
    for (bot_id, mem) in memory {
        let mut entries = mem
            .failed_move_history
            .iter()
            .map(|((from, dx, dy), count)| BlockedMove {
                from: *from,
                dx: *dx,
                dy: *dy,
                count: *count,
            })
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| b.count.cmp(&a.count));
        entries.truncate(4);
        if !entries.is_empty() {
            out.insert(bot_id.clone(), entries);
        }
    }
    out
}

fn active_item_set(state: &GameState) -> HashSet<&str> {
    state
        .orders
        .iter()
        .filter(|order| matches!(order.status, crate::model::OrderStatus::InProgress))
        .map(|order| order.item_id.as_str())
        .collect()
}

fn nearest_junction(
    start: u16,
    map: &crate::world::MapCache,
    junction_cells: &HashSet<u16>,
) -> Option<u16> {
    if junction_cells.contains(&start) {
        return Some(start);
    }
    let mut seen = HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    seen.insert(start);
    queue.push_back(start);
    while let Some(cell) = queue.pop_front() {
        for &next in &map.neighbors[cell as usize] {
            if !seen.insert(next) {
                continue;
            }
            if junction_cells.contains(&next) {
                return Some(next);
            }
            queue.push_back(next);
        }
    }
    None
}

fn pick_deadlock_escape(
    bot_id: &str,
    state: &GameState,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
    team_ctx: &TeamContext,
) -> Option<u16> {
    let bot = state.bots.iter().find(|b| b.id == bot_id)?;
    let current = map.idx(bot.x, bot.y)?;
    let occupied: HashSet<u16> = state
        .bots
        .iter()
        .filter(|b| b.id != bot_id)
        .filter_map(|b| map.idx(b.x, b.y))
        .collect();
    let ring_set: HashSet<u16> = team_ctx.queue.ring_cells.iter().copied().collect();
    let lane_set: HashSet<u16> = team_ctx.queue.lane_cells.iter().take(6).copied().collect();

    map.neighbors[current as usize]
        .iter()
        .copied()
        .filter(|candidate| !occupied.contains(candidate))
        .max_by_key(|candidate| {
            let near_drop = map
                .dropoff_cells
                .iter()
                .map(|&drop| dist.dist(*candidate, drop))
                .min()
                .unwrap_or(u16::MAX);
            let ring_penalty = if ring_set.contains(candidate) { 20 } else { 0 };
            let lane_penalty = if lane_set.contains(candidate) { 10 } else { 0 };
            let congestion = team_ctx
                .traffic
                .conflict_hotspots
                .iter()
                .find(|(cell, _)| cell == candidate)
                .map(|(_, count)| *count)
                .unwrap_or(0);
            (i32::from(near_drop.min(64)) - ring_penalty - lane_penalty - congestion as i32) as i16
        })
}

fn apply_yield_actions(
    state: &GameState,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
    team: &TeamContext,
    memory: &HashMap<String, BotMemory>,
    planned: &mut HashMap<String, PlannedAction>,
) {
    if map.dropoff_cells.is_empty() {
        return;
    }
    let mut reserved = HashSet::new();
    for bot in &state.bots {
        let Some(current) = map.idx(bot.x, bot.y) else {
            continue;
        };
        let next = match planned.get(&bot.id) {
            Some(PlannedAction {
                action: Action::Move { dx, dy, .. },
                ..
            }) => map.idx(bot.x + *dx, bot.y + *dy).unwrap_or(current),
            _ => current,
        };
        reserved.insert(next);
    }

    for bot in &state.bots {
        let role = team.role_for(&bot.id);
        if !matches!(role, BotRole::Yield) {
            continue;
        }
        let Some(current) = map.idx(bot.x, bot.y) else {
            continue;
        };
        let blocked = memory.get(&bot.id).map(|m| m.blocked_ticks).unwrap_or(0);
        if blocked < 2
            || !matches!(
                planned.get(&bot.id).map(|p| &p.action),
                Some(Action::Wait { .. })
            )
        {
            continue;
        }
        let mut best = None::<(u16, u16)>;
        for &candidate in &map.neighbors[current as usize] {
            if reserved.contains(&candidate) {
                continue;
            }
            let d = map
                .dropoff_cells
                .iter()
                .map(|&drop| dist.dist(candidate, drop))
                .min()
                .unwrap_or(u16::MAX);
            if best.map(|(best_d, _)| d > best_d).unwrap_or(true) {
                best = Some((d, candidate));
            }
        }
        let Some((_, next)) = best else {
            continue;
        };
        let (x0, y0) = map.xy(current);
        let (x1, y1) = map.xy(next);
        planned.insert(
            bot.id.clone(),
            PlannedAction {
                action: Action::Move {
                    bot_id: bot.id.clone(),
                    dx: x1 - x0,
                    dy: y1 - y0,
                },
                wait_reason: "intent_wait",
                fallback_stage: "yield_escape",
                ordering_stage: "pmat_postprocess",
                path_preview: vec![next],
            },
        );
        reserved.insert(next);
    }
}

fn apply_swap_loop_breaker(
    state: &GameState,
    map: &crate::world::MapCache,
    team: &TeamContext,
    planned: &mut HashMap<String, PlannedAction>,
) {
    let mut current: HashMap<String, u16> = HashMap::new();
    for bot in &state.bots {
        if let Some(cell) = map.idx(bot.x, bot.y) {
            current.insert(bot.id.clone(), cell);
        }
    }
    let mut seen_pairs = HashSet::new();
    for a in &state.bots {
        let Some(a_curr) = current.get(&a.id).copied() else {
            continue;
        };
        let Some(a_next) = planned_next_cell(a, map, planned) else {
            continue;
        };
        for b in &state.bots {
            if a.id >= b.id {
                continue;
            }
            let Some(b_curr) = current.get(&b.id).copied() else {
                continue;
            };
            let Some(b_next) = planned_next_cell(b, map, planned) else {
                continue;
            };
            if a_next == b_curr && b_next == a_curr {
                let key = (a.id.clone(), b.id.clone());
                if seen_pairs.contains(&key) {
                    continue;
                }
                seen_pairs.insert(key);
                let a_prio = team
                    .movement
                    .priorities
                    .get(&a.id)
                    .copied()
                    .unwrap_or(u8::MAX);
                let b_prio = team
                    .movement
                    .priorities
                    .get(&b.id)
                    .copied()
                    .unwrap_or(u8::MAX);
                let (higher, lower) = if a_prio <= b_prio {
                    (a.id.as_str(), b.id.as_str())
                } else {
                    (b.id.as_str(), a.id.as_str())
                };

                planned.insert(
                    lower.to_owned(),
                    PlannedAction {
                        action: Action::wait(lower.to_owned()),
                        wait_reason: "blocked_by_edge_reservation",
                        fallback_stage: "swap_loop_breaker_hold",
                        ordering_stage: "pmat_postprocess",
                        path_preview: Vec::new(),
                    },
                );
                if let Some(reroute) = pick_swap_reroute(higher, lower, state, map, planned) {
                    let Some(start) = current.get(higher).copied() else {
                        continue;
                    };
                    let (x0, y0) = map.xy(start);
                    let (x1, y1) = map.xy(reroute);
                    planned.insert(
                        higher.to_owned(),
                        PlannedAction {
                            action: Action::Move {
                                bot_id: higher.to_owned(),
                                dx: x1 - x0,
                                dy: y1 - y0,
                            },
                            wait_reason: "intent_wait",
                            fallback_stage: "swap_loop_breaker_reroute",
                            ordering_stage: "pmat_postprocess",
                            path_preview: vec![reroute],
                        },
                    );
                }
            }
        }
    }
}

fn planned_next_cell(
    bot: &crate::model::BotState,
    map: &crate::world::MapCache,
    planned: &HashMap<String, PlannedAction>,
) -> Option<u16> {
    let start = map.idx(bot.x, bot.y)?;
    let next = match planned.get(&bot.id).map(|p| &p.action) {
        Some(Action::Move { dx, dy, .. }) => map.idx(bot.x + *dx, bot.y + *dy).unwrap_or(start),
        _ => start,
    };
    Some(next)
}

fn pick_swap_reroute(
    higher: &str,
    lower: &str,
    state: &GameState,
    map: &crate::world::MapCache,
    planned: &HashMap<String, PlannedAction>,
) -> Option<u16> {
    let higher_bot = state.bots.iter().find(|b| b.id == higher)?;
    let lower_bot = state.bots.iter().find(|b| b.id == lower)?;
    let start = map.idx(higher_bot.x, higher_bot.y)?;
    let lower_cell = map.idx(lower_bot.x, lower_bot.y)?;
    let mut reserved: HashSet<u16> = state
        .bots
        .iter()
        .filter(|b| b.id != higher)
        .filter_map(|b| planned_next_cell(b, map, planned))
        .collect();
    reserved.insert(lower_cell);
    map.neighbors[start as usize]
        .iter()
        .copied()
        .filter(|candidate| !reserved.contains(candidate))
        .max_by_key(|candidate| map.neighbors[*candidate as usize].len())
}

fn compute_ordering_sequence(
    state: &GameState,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
    goals: &HashMap<String, u16>,
    team: &TeamContext,
    memory: &HashMap<String, BotMemory>,
    mode: &str,
) -> (Vec<String>, HashMap<String, f64>, HashMap<String, u16>) {
    let mut occupancy: HashMap<u16, u16> = HashMap::new();
    for bot in &state.bots {
        if let Some(cell) = map.idx(bot.x, bot.y) {
            *occupancy.entry(cell).or_insert(0) += 1;
        }
    }
    let mut scored = Vec::<(String, f64)>::new();
    let mut scores = HashMap::<String, f64>::new();
    for bot in &state.bots {
        let Some(start) = map.idx(bot.x, bot.y) else {
            continue;
        };
        let goal = goals.get(&bot.id).copied().unwrap_or(start);
        let dist_to_goal = f64::from(dist.dist(start, goal).min(64));
        let role = team.role_for(&bot.id);
        let blocked_ticks = memory
            .get(&bot.id)
            .map(|m| f64::from(m.blocked_ticks))
            .unwrap_or(0.0);
        let local_conflict = team
            .traffic
            .local_conflict_count_by_bot
            .get(&bot.id)
            .copied()
            .map(f64::from)
            .unwrap_or(0.0);
        let carrying_active = team
            .dropoff_serviceable_by_bot
            .get(&bot.id)
            .copied()
            .unwrap_or(false);
        let watchdog_pressure = memory
            .get(&bot.id)
            .map(|m| {
                f64::from(m.same_dropoff_order_streak)
                    + f64::from(m.dropoff_ban_ticks_by_order.values().copied().sum::<u8>()) * 0.25
            })
            .unwrap_or(0.0);
        let choke_occupancy = map
            .neighbors
            .get(start as usize)
            .map(|nbrs| {
                nbrs.iter()
                    .map(|cell| occupancy.get(cell).copied().unwrap_or(0))
                    .sum::<u16>() as f64
            })
            .unwrap_or(0.0);
        let mut score = 0.0;
        if carrying_active {
            score += 50.0;
        }
        if matches!(role, BotRole::LeadCourier) {
            score += 35.0;
        } else if matches!(role, BotRole::QueueCourier) {
            score += 25.0;
        }
        score += blocked_ticks * 2.0;
        score += local_conflict * 1.5;
        score -= dist_to_goal * 0.75;
        score += watchdog_pressure * 10.0;
        score += choke_occupancy * 3.0;

        let model_score = maybe_score_ordering(
            mode,
            OrderingFeatures {
                carrying_active: if carrying_active { 1.0 } else { 0.0 },
                queue_role_lead: if matches!(role, BotRole::LeadCourier) {
                    1.0
                } else {
                    0.0
                },
                queue_role_courier: if matches!(role, BotRole::QueueCourier) {
                    1.0
                } else {
                    0.0
                },
                blocked_ticks,
                local_conflict_count: local_conflict,
                dist_to_goal,
                dropoff_watchdog_pressure: watchdog_pressure,
                choke_occupancy,
            },
        )
        .unwrap_or(0.0);
        score += model_score;
        scores.insert(bot.id.clone(), score);
        scored.push((bot.id.clone(), score));
    }
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                if state.tick % 2 == 0 {
                    a.0.cmp(&b.0)
                } else {
                    b.0.cmp(&a.0)
                }
            })
    });
    let ordering = scored.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
    let ranks = ordering
        .iter()
        .enumerate()
        .map(|(idx, bot_id)| (bot_id.clone(), idx as u16))
        .collect::<HashMap<_, _>>();
    (ordering, scores, ranks)
}

fn queue_strict_mode(mode: &str) -> bool {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    if let Some(value) = *OVERRIDE.get_or_init(|| {
        std::env::var("QUEUE_STRICT_MODE")
            .ok()
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    }) {
        return value;
    }
    matches!(mode, "medium" | "hard" | "expert")
}

fn dropoff_watchdog_streak() -> u8 {
    4
}

fn dropoff_watchdog_ban_ticks() -> u8 {
    8
}

fn should_trigger_dropoff_watchdog(
    in_progress: bool,
    carrying_required: bool,
    streak: u8,
) -> bool {
    !in_progress || !carrying_required || streak >= dropoff_watchdog_streak()
}

#[cfg(test)]
mod tests {
    use super::should_trigger_dropoff_watchdog;

    #[test]
    fn watchdog_triggers_for_pending_or_streak() {
        assert!(should_trigger_dropoff_watchdog(false, true, 1));
        assert!(should_trigger_dropoff_watchdog(true, false, 1));
        assert!(should_trigger_dropoff_watchdog(true, true, 4));
        assert!(!should_trigger_dropoff_watchdog(true, true, 2));
    }
}

fn queue_max_ring_entrants() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("QUEUE_MAX_RING_ENTRANTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1)
    })
}

fn deadlock_escape_ticks() -> u8 {
    static VALUE: OnceLock<u8> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("DEADLOCK_ESCAPE_TICKS")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(3)
            .max(1)
    })
}

fn bot_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("BOT_DEBUG")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    })
}
