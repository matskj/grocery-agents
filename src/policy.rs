use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use tracing::info;

use crate::{
    assign::AssignmentEngine,
    config::{AssignmentMode, Config},
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
    config: Arc<Config>,
    assigner: AssignmentEngine,
    dispatcher: Dispatcher,
    planner: MotionPlanner,
    memory: HashMap<String, BotMemory>,
    sticky_roles: HashMap<String, StickyQueueRole>,
    last_team_telemetry: serde_json::Value,
    ticks_since_pickup: u16,
    ticks_since_dropoff: u16,
    recent_goal_cells: VecDeque<u16>,
    global_no_progress_streak: u8,
    forced_legacy_ticks_remaining: u8,
}

impl Policy {
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            planner: MotionPlanner::new(config.horizon),
            config,
            assigner: AssignmentEngine::new(),
            dispatcher: Dispatcher::new(),
            memory: HashMap::new(),
            sticky_roles: HashMap::new(),
            last_team_telemetry: serde_json::json!({}),
            ticks_since_pickup: 0,
            ticks_since_dropoff: 0,
            recent_goal_cells: VecDeque::new(),
            global_no_progress_streak: 0,
            forced_legacy_ticks_remaining: 0,
        }
    }

    pub fn decide_round(&mut self, state: &GameState, soft_budget: Duration) -> Vec<Action> {
        let tick_started = Instant::now();
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let active_items = active_item_set(state);
        let (pickup_events, dropoff_events) = self.detect_inventory_events(state);
        self.update_progress_watchdog(pickup_events, dropoff_events);
        let forced_legacy_active = self.forced_legacy_ticks_remaining > 0;
        if forced_legacy_active {
            self.forced_legacy_ticks_remaining =
                self.forced_legacy_ticks_remaining.saturating_sub(1);
        }

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

        let assign_started = Instant::now();
        let mut assignment_source = "legacy_dispatcher";
        let mut assignment_task_count = 0usize;
        let mut assignment_edge_count = 0usize;
        let mut assignment_active_task_count = 0usize;
        let mut assignment_preview_task_count = 0usize;
        let mut assignment_stand_task_count = 0usize;
        let mut assignment_dropoff_task_count = 0usize;
        let mut assignment_active_gap_total = 0usize;
        let mut assignment_preview_enabled = true;
        let mut assignment_guard_trigger_reason: Option<&'static str> = None;
        let assignment_mode = if self.config.assignment_enabled {
            self.config.assignment_mode
        } else {
            AssignmentMode::LegacyOnly
        };
        let global_result = if matches!(assignment_mode, AssignmentMode::LegacyOnly)
            || forced_legacy_active
        {
            None
        } else {
            let remaining = soft_budget.saturating_sub(tick_started.elapsed());
            self.assigner.build_intents(
                state,
                map,
                &dist,
                &team_ctx,
                self.config.candidate_k,
                self.config.lambda_density,
                self.config.lambda_choke,
                remaining,
            )
        };
        let legacy_intents =
            self.dispatcher
                .build_intents(state, map, &dist, &blocked_snapshot, &team_ctx);

        if let Some(result) = &global_result {
            assignment_task_count = result.task_count;
            assignment_edge_count = result.edge_count;
            assignment_active_task_count = result.active_task_count;
            assignment_preview_task_count = result.preview_task_count;
            assignment_stand_task_count = result.stand_task_count;
            assignment_dropoff_task_count = result.dropoff_task_count;
            assignment_active_gap_total = result.active_gap_total;
            assignment_preview_enabled = result.preview_enabled;
        }

        let mut intents = if forced_legacy_active {
            assignment_source = "legacy_forced_watchdog";
            legacy_intents.clone()
        } else {
            match assignment_mode {
                AssignmentMode::LegacyOnly => {
                    assignment_source = "legacy_dispatcher";
                    legacy_intents.clone()
                }
                AssignmentMode::GlobalOnly => {
                    if let Some(result) = global_result {
                        if let Some(reason) = assignment_guard_reason(
                            state,
                            map,
                            &team_ctx,
                            &result.intents,
                            self.ticks_since_pickup,
                            self.ticks_since_dropoff,
                            self.unique_goal_cells_recent(),
                        ) {
                            assignment_guard_trigger_reason = Some(reason);
                            assignment_source = "legacy_dispatcher_guard";
                            legacy_intents.clone()
                        } else {
                            assignment_source = "global_assignment";
                            result.intents
                        }
                    } else {
                        assignment_source = "legacy_dispatcher_timeout_fallback";
                        assignment_guard_trigger_reason = Some("global_timeout_fallback");
                        legacy_intents.clone()
                    }
                }
                AssignmentMode::Hybrid => {
                    if let Some(result) = global_result {
                        if let Some(reason) = assignment_guard_reason(
                            state,
                            map,
                            &team_ctx,
                            &result.intents,
                            self.ticks_since_pickup,
                            self.ticks_since_dropoff,
                            self.unique_goal_cells_recent(),
                        ) {
                            assignment_guard_trigger_reason = Some(reason);
                            assignment_source = "legacy_dispatcher_guard";
                            legacy_intents.clone()
                        } else {
                            assignment_source = "hybrid_assignment";
                            merge_hybrid_intents(state, &team_ctx, result.intents, &legacy_intents)
                        }
                    } else {
                        assignment_source = "legacy_dispatcher_timeout_fallback";
                        assignment_guard_trigger_reason = Some("global_timeout_fallback");
                        legacy_intents.clone()
                    }
                }
            }
        };
        if assignment_source == "global_assignment" || assignment_source == "hybrid_assignment" {
            if should_trigger_assignment_move_loop_watchdog(
                &intents,
                self.ticks_since_pickup,
                self.ticks_since_dropoff,
            ) {
                self.global_no_progress_streak = self.global_no_progress_streak.saturating_add(1);
            } else {
                self.global_no_progress_streak = 0;
            }
            if self.global_no_progress_streak >= assignment_no_progress_trigger_ticks() {
                self.global_no_progress_streak = 0;
                self.forced_legacy_ticks_remaining = assignment_forced_legacy_ticks();
                intents = legacy_intents.clone();
                assignment_source = "legacy_forced_watchdog_trigger";
                assignment_guard_trigger_reason = Some("global_move_loop_watchdog");
            }
        } else {
            self.global_no_progress_streak = 0;
        };
        rebalance_pickup_goal_crowding(state, map, &dist, &team_ctx, &active_items, &mut intents);
        let assign_ms = assign_started.elapsed().as_millis() as u64;
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
        let mut dropoff_schedule_status_by_bot = serde_json::Map::new();
        let mut dropoff_slot_reserved_by_bot = serde_json::Map::new();
        let mut dropoff_staging_cell_by_bot = serde_json::Map::new();
        let mut dropoff_slot_usage: HashMap<(u8, u16), u8> = HashMap::new();
        let mut reserved_staging_cells = HashSet::<u16>::new();
        let dropoff_staging_cells = map
            .dropoff_cells
            .iter()
            .copied()
            .map(|drop| {
                (
                    drop,
                    build_dropoff_staging_cells(drop, map, &dist, &team_ctx.traffic.conflict_hotspots),
                )
            })
            .collect::<HashMap<_, _>>();

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
                        mem.same_dropoff_order_streak =
                            mem.same_dropoff_order_streak.saturating_add(1);
                    } else {
                        mem.last_dropoff_order_id = Some(order_id.clone());
                        mem.same_dropoff_order_streak = 1;
                    }
                    dropoff_attempt_streak_snapshot
                        .insert(bot.id.clone(), mem.same_dropoff_order_streak);
                    let watchdog_trigger = banned
                        || if self.config.dropoff_scheduling_enabled {
                            !in_progress || !carrying_required
                        } else {
                            should_trigger_dropoff_watchdog(
                                in_progress,
                                carrying_required,
                                mem.same_dropoff_order_streak,
                            )
                        };
                    if watchdog_trigger {
                        mem.dropoff_watchdog_triggered = true;
                        dropoff_watchdog_snapshot.insert(bot.id.clone(), true);
                        mem.dropoff_ban_ticks_by_order
                            .insert(order_id.clone(), dropoff_watchdog_ban_ticks());
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
                    dropoff_target_status_by_bot
                        .insert(bot.id.clone(), serde_json::Value::String("none".to_owned()));
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
                    dropoff_target_status_by_bot
                        .insert(bot.id.clone(), serde_json::Value::String("none".to_owned()));
                    let carrying_active = bot
                        .carrying
                        .iter()
                        .any(|item| active_items.contains(item.as_str()));
                    if let Some(queue_target) = queue_goal {
                        if carrying_active
                            && matches!(role, BotRole::LeadCourier | BotRole::QueueCourier)
                        {
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
                    if self.config.dropoff_scheduling_enabled
                        && carrying_active
                        && map.dropoff_cells.contains(&goal)
                    {
                        let eta = dist.dist(cell, goal).min(u16::from(u8::MAX)) as u8;
                        let window = self.config.dropoff_window.max(2);
                        let mut reserved_slot = None::<u8>;
                        for t in eta..=eta.saturating_add(window) {
                            let used = dropoff_slot_usage.get(&(t, goal)).copied().unwrap_or(0);
                            if used < self.config.dropoff_capacity {
                                dropoff_slot_usage.insert((t, goal), used.saturating_add(1));
                                reserved_slot = Some(t);
                                break;
                            }
                        }
                        if let Some(slot_t) = reserved_slot {
                            dropoff_slot_reserved_by_bot.insert(
                                bot.id.clone(),
                                serde_json::Value::Number(serde_json::Number::from(slot_t as i64)),
                            );
                            if slot_t > eta.saturating_add(1) {
                                if let Some(stage) = pick_staging_cell(
                                    bot.id.as_str(),
                                    cell,
                                    goal,
                                    &dropoff_staging_cells,
                                    &reserved_staging_cells,
                                    map,
                                    &dist,
                                ) {
                                    reserved_staging_cells.insert(stage);
                                    goal = stage;
                                    let (sx, sy) = map.xy(stage);
                                    dropoff_staging_cell_by_bot.insert(
                                        bot.id.clone(),
                                        serde_json::json!([sx, sy]),
                                    );
                                    dropoff_schedule_status_by_bot.insert(
                                        bot.id.clone(),
                                        serde_json::Value::String("staging_wait_slot".to_owned()),
                                    );
                                } else {
                                    dropoff_schedule_status_by_bot.insert(
                                        bot.id.clone(),
                                        serde_json::Value::String("slot_reserved_direct".to_owned()),
                                    );
                                }
                            } else {
                                dropoff_schedule_status_by_bot.insert(
                                    bot.id.clone(),
                                    serde_json::Value::String("slot_reserved_direct".to_owned()),
                                );
                            }
                        } else if let Some(stage) = pick_staging_cell(
                            bot.id.as_str(),
                            cell,
                            goal,
                            &dropoff_staging_cells,
                            &reserved_staging_cells,
                            map,
                            &dist,
                        ) {
                            reserved_staging_cells.insert(stage);
                            goal = stage;
                            let (sx, sy) = map.xy(stage);
                            dropoff_staging_cell_by_bot
                                .insert(bot.id.clone(), serde_json::json!([sx, sy]));
                            dropoff_schedule_status_by_bot.insert(
                                bot.id.clone(),
                                serde_json::Value::String("staging_no_slot".to_owned()),
                            );
                        } else {
                            dropoff_schedule_status_by_bot.insert(
                                bot.id.clone(),
                                serde_json::Value::String("no_slot_wait".to_owned()),
                            );
                        }
                    } else {
                        dropoff_schedule_status_by_bot.insert(
                            bot.id.clone(),
                            serde_json::Value::String("not_scheduled".to_owned()),
                        );
                    }
                    goals.insert(bot.id.clone(), goal);
                }
                Intent::Wait => {
                    dropoff_target_status_by_bot
                        .insert(bot.id.clone(), serde_json::Value::String("none".to_owned()));
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
        self.update_recent_goal_cells(&goals);

        let (ordering_sequence, ordering_scores, ordering_ranks) =
            compute_ordering_sequence(state, map, &dist, &goals, &team_ctx, &self.memory, mode);

        let plan_result = self.planner.plan(
            state,
            map,
            &dist,
            &goals,
            &team_ctx.movement,
            &ordering_sequence,
            Some(Instant::now() + soft_budget.saturating_sub(tick_started.elapsed())),
            self.config.dropoff_capacity,
        );
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
            state_entry.constraint_relax_ticks_remaining =
                mem.map(|m| m.constraint_relax_ticks_remaining).unwrap_or(0);
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
        let blocked_ticks_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let blocked = self
                    .memory
                    .get(&bot.id)
                    .map(|m| m.blocked_ticks)
                    .unwrap_or(0);
                (
                    bot.id.clone(),
                    serde_json::Value::Number(serde_json::Number::from(blocked as i64)),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        for bot in &state.bots {
            dropoff_target_status_by_bot
                .entry(bot.id.clone())
                .or_insert_with(|| serde_json::Value::String("none".to_owned()));
            dropoff_schedule_status_by_bot
                .entry(bot.id.clone())
                .or_insert_with(|| serde_json::Value::String("not_scheduled".to_owned()));
            dropoff_slot_reserved_by_bot
                .entry(bot.id.clone())
                .or_insert(serde_json::Value::Null);
            dropoff_staging_cell_by_bot
                .entry(bot.id.clone())
                .or_insert(serde_json::Value::Null);
        }

        let mut telemetry = team_ctx.telemetry(map);
        if let Some(obj) = telemetry.as_object_mut() {
            obj.insert(
                "selected_intents".to_owned(),
                serde_json::Value::Object(intent_labels),
            );
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
            let reserved_cells_by_t = plan_result
                .diagnostics
                .reserved_cells_by_t
                .iter()
                .map(|(t, cells)| {
                    (
                        t.to_string(),
                        serde_json::Value::Array(
                            cells.iter()
                                .map(|idx| {
                                    let (x, y) = map.xy(*idx);
                                    serde_json::json!([x, y])
                                })
                                .collect::<Vec<_>>(),
                        ),
                    )
                })
                .collect::<serde_json::Map<_, _>>();
            obj.insert(
                "reserved_cells_by_t".to_owned(),
                serde_json::Value::Object(reserved_cells_by_t),
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
            obj.insert(
                "blocked_ticks_by_bot".to_owned(),
                serde_json::Value::Object(blocked_ticks_by_bot),
            );
            obj.insert(
                "assign_ms".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(assign_ms)),
            );
            obj.insert(
                "assignment_source".to_owned(),
                serde_json::Value::String(assignment_source.to_owned()),
            );
            obj.insert(
                "assignment_mode".to_owned(),
                serde_json::Value::String(
                    match assignment_mode {
                        AssignmentMode::Hybrid => "hybrid",
                        AssignmentMode::GlobalOnly => "global_only",
                        AssignmentMode::LegacyOnly => "legacy_only",
                    }
                    .to_owned(),
                ),
            );
            obj.insert(
                "assignment_enabled".to_owned(),
                serde_json::Value::Bool(self.config.assignment_enabled),
            );
            obj.insert(
                "assignment_task_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(assignment_task_count as i64)),
            );
            obj.insert(
                "assignment_edge_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(assignment_edge_count as i64)),
            );
            obj.insert(
                "assignment_active_task_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment_active_task_count as i64
                )),
            );
            obj.insert(
                "assignment_preview_task_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment_preview_task_count as i64
                )),
            );
            obj.insert(
                "assignment_stand_task_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment_stand_task_count as i64
                )),
            );
            obj.insert(
                "assignment_dropoff_task_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment_dropoff_task_count as i64
                )),
            );
            obj.insert(
                "assignment_active_gap_total".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment_active_gap_total as i64
                )),
            );
            obj.insert(
                "assignment_preview_enabled".to_owned(),
                serde_json::Value::Bool(assignment_preview_enabled),
            );
            obj.insert(
                "assignment_guard_triggered".to_owned(),
                serde_json::Value::Bool(assignment_guard_trigger_reason.is_some()),
            );
            obj.insert(
                "assignment_guard_reason".to_owned(),
                serde_json::Value::String(
                    assignment_guard_trigger_reason.unwrap_or("none").to_owned(),
                ),
            );
            obj.insert(
                "dropoff_schedule_status_by_bot".to_owned(),
                serde_json::Value::Object(dropoff_schedule_status_by_bot),
            );
            obj.insert(
                "dropoff_slot_reserved_by_bot".to_owned(),
                serde_json::Value::Object(dropoff_slot_reserved_by_bot),
            );
            obj.insert(
                "dropoff_staging_cell_by_bot".to_owned(),
                serde_json::Value::Object(dropoff_staging_cell_by_bot),
            );
            obj.insert(
                "ticks_since_pickup".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(self.ticks_since_pickup as i64)),
            );
            obj.insert(
                "ticks_since_dropoff".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(self.ticks_since_dropoff as i64)),
            );
            obj.insert(
                "unique_goal_cells_last_n".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.unique_goal_cells_recent() as i64
                )),
            );
            obj.insert(
                "forced_legacy_ticks_remaining".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.forced_legacy_ticks_remaining as i64
                )),
            );
            obj.insert(
                "global_no_progress_streak".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.global_no_progress_streak as i64
                )),
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

    fn detect_inventory_events(&self, state: &GameState) -> (u16, u16) {
        let mut pickups = 0u16;
        let mut dropoffs = 0u16;
        for bot in &state.bots {
            let current = bot.carrying.len() as i32;
            let previous = self
                .memory
                .get(&bot.id)
                .map(|m| m.prev_carrying.len() as i32)
                .unwrap_or(current);
            if current > previous {
                pickups = pickups.saturating_add((current - previous) as u16);
            } else if previous > current {
                dropoffs = dropoffs.saturating_add((previous - current) as u16);
            }
        }
        (pickups, dropoffs)
    }

    fn update_progress_watchdog(&mut self, pickup_events: u16, dropoff_events: u16) {
        if pickup_events > 0 {
            self.ticks_since_pickup = 0;
        } else {
            self.ticks_since_pickup = self.ticks_since_pickup.saturating_add(1);
        }
        if dropoff_events > 0 {
            self.ticks_since_dropoff = 0;
        } else {
            self.ticks_since_dropoff = self.ticks_since_dropoff.saturating_add(1);
        }
    }

    fn update_recent_goal_cells(&mut self, goals: &HashMap<String, u16>) {
        let mut cells = goals.values().copied().collect::<Vec<_>>();
        cells.sort_unstable();
        cells.dedup();
        for cell in cells {
            self.recent_goal_cells.push_back(cell);
            while self.recent_goal_cells.len() > assignment_goal_history_window() {
                self.recent_goal_cells.pop_front();
            }
        }
    }

    fn unique_goal_cells_recent(&self) -> usize {
        self.recent_goal_cells
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len()
    }

    fn update_memory(&mut self, state: &GameState, map: &crate::world::MapCache) {
        for bot in &state.bots {
            let cell = map.idx(bot.x, bot.y).unwrap_or(0);
            let mem = self.memory.entry(bot.id.clone()).or_default();
            mem.constraint_relax_ticks_remaining =
                mem.constraint_relax_ticks_remaining.saturating_sub(1);
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
            mem.repeated_failed_moves =
                mem.failed_move_history.values().copied().max().unwrap_or(0);
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

fn build_prohibited_move_map(
    memory: &HashMap<String, BotMemory>,
) -> HashMap<String, Vec<BlockedMove>> {
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

fn merge_hybrid_intents(
    state: &GameState,
    team_ctx: &TeamContext,
    global: Vec<BotIntent>,
    legacy: &[BotIntent],
) -> Vec<BotIntent> {
    let global_by_bot = global
        .into_iter()
        .map(|intent| (intent.bot_id.clone(), intent.intent))
        .collect::<HashMap<_, _>>();
    let legacy_by_bot = legacy
        .iter()
        .map(|intent| (intent.bot_id.clone(), intent.intent.clone()))
        .collect::<HashMap<_, _>>();
    let mut bots = state.bots.iter().collect::<Vec<_>>();
    bots.sort_by(|a, b| a.id.cmp(&b.id));
    bots.into_iter()
        .map(|bot| {
            let carrying_active = bot
                .carrying
                .iter()
                .any(|item| team_ctx.active_order_items_set.contains(item));
            let intent = if carrying_active {
                global_by_bot
                    .get(&bot.id)
                    .cloned()
                    .or_else(|| legacy_by_bot.get(&bot.id).cloned())
                    .unwrap_or(Intent::Wait)
            } else {
                legacy_by_bot
                    .get(&bot.id)
                    .cloned()
                    .or_else(|| global_by_bot.get(&bot.id).cloned())
                    .unwrap_or(Intent::Wait)
            };
            BotIntent {
                bot_id: bot.id.clone(),
                intent,
            }
        })
        .collect()
}

fn rebalance_pickup_goal_crowding(
    state: &GameState,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
    team_ctx: &TeamContext,
    active_items: &HashSet<&str>,
    intents: &mut [BotIntent],
) {
    if intents.is_empty() || active_items.is_empty() {
        return;
    }
    let mut active_stands = HashSet::<u16>::new();
    for item in &state.items {
        if !active_items.contains(item.kind.as_str()) {
            continue;
        }
        for &stand in map.stand_cells_for_item(&item.id) {
            active_stands.insert(stand);
        }
    }
    if active_stands.len() < 2 {
        return;
    }
    let mut active_stand_list = active_stands.into_iter().collect::<Vec<_>>();
    active_stand_list.sort_unstable();

    let bot_by_id = state
        .bots
        .iter()
        .map(|bot| (bot.id.as_str(), bot))
        .collect::<HashMap<_, _>>();
    let mut idx_by_bot = HashMap::<String, usize>::new();
    for (idx, entry) in intents.iter().enumerate() {
        idx_by_bot.insert(entry.bot_id.clone(), idx);
    }

    let mut crowd_by_goal = HashMap::<u16, Vec<String>>::new();
    for intent in intents.iter() {
        let Intent::MoveTo { cell } = intent.intent else {
            continue;
        };
        if !active_stand_list.contains(&cell) {
            continue;
        }
        let Some(bot) = bot_by_id.get(intent.bot_id.as_str()).copied() else {
            continue;
        };
        let carrying_active = bot
            .carrying
            .iter()
            .any(|item| active_items.contains(item.as_str()));
        if carrying_active {
            continue;
        }
        if matches!(
            team_ctx.role_for(&bot.id),
            BotRole::LeadCourier | BotRole::QueueCourier
        ) {
            continue;
        }
        crowd_by_goal
            .entry(cell)
            .or_default()
            .push(bot.id.clone());
    }

    let mut crowd_load = crowd_by_goal
        .iter()
        .map(|(cell, bots)| (*cell, bots.len() as u16))
        .collect::<HashMap<_, _>>();
    let mut reserved = HashSet::<u16>::new();
    for (goal, bots) in crowd_by_goal {
        if bots.len() <= 1 {
            reserved.insert(goal);
            continue;
        }
        let mut ordered = bots;
        ordered.sort_by(|a, b| {
            let da = bot_by_id
                .get(a.as_str())
                .and_then(|bot| map.idx(bot.x, bot.y))
                .map(|start| dist.dist(start, goal))
                .unwrap_or(u16::MAX);
            let db = bot_by_id
                .get(b.as_str())
                .and_then(|bot| map.idx(bot.x, bot.y))
                .map(|start| dist.dist(start, goal))
                .unwrap_or(u16::MAX);
            da.cmp(&db).then_with(|| a.cmp(b))
        });
        let keep = ordered.first().cloned();
        if let Some(keep_bot) = keep {
            reserved.insert(goal);
            for bot_id in ordered.into_iter().skip(1) {
                let Some(bot) = bot_by_id.get(bot_id.as_str()).copied() else {
                    continue;
                };
                let Some(start) = map.idx(bot.x, bot.y) else {
                    continue;
                };
                let mut candidates = active_stand_list.clone();
                candidates.sort_by(|a, b| {
                    let load_a = crowd_load.get(a).copied().unwrap_or(0);
                    let load_b = crowd_load.get(b).copied().unwrap_or(0);
                    load_a
                        .cmp(&load_b)
                        .then_with(|| dist.dist(start, *a).cmp(&dist.dist(start, *b)))
                        .then_with(|| a.cmp(b))
                });
                let mut chosen = None::<u16>;
                for candidate in &candidates {
                    if *candidate == goal || reserved.contains(candidate) {
                        continue;
                    }
                    chosen = Some(*candidate);
                    break;
                }
                if chosen.is_none() {
                    for candidate in &candidates {
                        if *candidate == goal {
                            continue;
                        }
                        chosen = Some(*candidate);
                        break;
                    }
                }
                let Some(new_goal) = chosen else {
                    continue;
                };
                if let Some(&intent_idx) = idx_by_bot.get(&bot_id) {
                    intents[intent_idx].intent = Intent::MoveTo { cell: new_goal };
                    let load = crowd_load.get(&new_goal).copied().unwrap_or(0);
                    crowd_load.insert(new_goal, load.saturating_add(1));
                    reserved.insert(new_goal);
                }
            }
            let keep_load = crowd_load.get(&goal).copied().unwrap_or(1);
            crowd_load.insert(goal, keep_load.min(1));
            let _ = keep_bot;
        }
    }
}

fn should_trigger_assignment_move_loop_watchdog(
    intents: &[BotIntent],
    ticks_since_pickup: u16,
    ticks_since_dropoff: u16,
) -> bool {
    let considered = intents
        .iter()
        .filter(|entry| !matches!(entry.intent, Intent::Wait))
        .count();
    if considered == 0 {
        return false;
    }
    let move_count = intents
        .iter()
        .filter(|entry| matches!(entry.intent, Intent::MoveTo { .. }))
        .count();
    let progress_count = intents
        .iter()
        .filter(|entry| matches!(entry.intent, Intent::PickUp { .. } | Intent::DropOff { .. }))
        .count();
    move_count * 100 >= considered * 80
        && progress_count == 0
        && ticks_since_pickup >= 10
        && ticks_since_dropoff >= 14
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

fn build_dropoff_staging_cells(
    dropoff: u16,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
    conflict_hotspots: &[(u16, u16)],
) -> Vec<u16> {
    let mut out = Vec::<u16>::new();
    let hotspot_penalty = conflict_hotspots
        .iter()
        .copied()
        .collect::<HashMap<u16, u16>>();
    for idx in 0..map.neighbors.len() as u16 {
        if map.wall_mask[idx as usize] || idx == dropoff {
            continue;
        }
        let d = dist.dist(idx, dropoff);
        if d == u16::MAX || !(1..=2).contains(&d) {
            continue;
        }
        out.push(idx);
    }
    out.sort_by(|a, b| {
        dist.dist(*a, dropoff)
            .cmp(&dist.dist(*b, dropoff))
            .then_with(|| {
                let ah = hotspot_penalty.get(a).copied().unwrap_or(0);
                let bh = hotspot_penalty.get(b).copied().unwrap_or(0);
                ah.cmp(&bh)
            })
            .then_with(|| {
                map.neighbors[*b as usize]
                    .len()
                    .cmp(&map.neighbors[*a as usize].len())
            })
            .then_with(|| a.cmp(b))
    });
    out
}

fn pick_staging_cell(
    bot_id: &str,
    from_cell: u16,
    dropoff: u16,
    staging_by_dropoff: &HashMap<u16, Vec<u16>>,
    reserved_staging: &HashSet<u16>,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
) -> Option<u16> {
    let mut candidates = staging_by_dropoff.get(&dropoff)?.clone();
    candidates.sort_by(|a, b| {
        dist.dist(from_cell, *a)
            .cmp(&dist.dist(from_cell, *b))
            .then_with(|| {
                map.neighbors[*b as usize]
                    .len()
                    .cmp(&map.neighbors[*a as usize].len())
            })
            .then_with(|| a.cmp(b))
    });
    let _ = bot_id;
    candidates
        .into_iter()
        .find(|cell| !reserved_staging.contains(cell))
}

fn assignment_guard_reason(
    state: &GameState,
    map: &crate::world::MapCache,
    team: &TeamContext,
    intents: &[BotIntent],
    ticks_since_pickup: u16,
    ticks_since_dropoff: u16,
    unique_goal_cells_last_n: usize,
) -> Option<&'static str> {
    let active_orders_exist = state
        .orders
        .iter()
        .any(|order| matches!(order.status, crate::model::OrderStatus::InProgress));
    if !active_orders_exist {
        return None;
    }

    let mut stand_cells = HashSet::<u16>::new();
    for item in &state.items {
        for &stand in map.stand_cells_for_item(&item.id) {
            stand_cells.insert(stand);
        }
    }

    let bot_by_id = state
        .bots
        .iter()
        .map(|bot| (bot.id.as_str(), bot))
        .collect::<HashMap<_, _>>();

    let mut pickup_progress_intents = 0usize;
    let mut dropoff_progress_intents = 0usize;
    let mut stand_move_intents = 0usize;
    let mut empty_dropoff_seek_count = 0usize;
    let mut carrying_dropoff_seek_count = 0usize;
    for BotIntent { bot_id, intent } in intents {
        let Some(bot) = bot_by_id.get(bot_id.as_str()).copied() else {
            continue;
        };
        let carrying_active = bot
            .carrying
            .iter()
            .any(|item| team.active_order_items_set.contains(item));
        match intent {
            Intent::PickUp { .. } => {
                pickup_progress_intents += 1;
            }
            Intent::DropOff { .. } => {
                dropoff_progress_intents += 1;
            }
            Intent::MoveTo { cell } => {
                if stand_cells.contains(cell) {
                    stand_move_intents += 1;
                }
                if map.dropoff_cells.contains(cell) {
                    if carrying_active {
                        carrying_dropoff_seek_count += 1;
                    } else if bot.carrying.is_empty() {
                        empty_dropoff_seek_count += 1;
                    }
                }
            }
            Intent::Wait => {}
        }
    }

    let empty_capacity_bots = state
        .bots
        .iter()
        .filter(|bot| bot.carrying.len() < bot.capacity)
        .count();
    let crowd_threshold = state.bots.len().saturating_sub(1).max(2);

    if carrying_dropoff_seek_count == 0 && empty_dropoff_seek_count >= crowd_threshold {
        return Some("empty_dropoff_cluster");
    }
    if empty_capacity_bots > 0 && pickup_progress_intents == 0 && ticks_since_pickup >= 8 {
        return Some("no_pickup_progress");
    }
    if pickup_progress_intents == 0
        && dropoff_progress_intents == 0
        && stand_move_intents > 0
        && ticks_since_pickup >= 12
        && ticks_since_dropoff >= 16
        && unique_goal_cells_last_n <= 4
    {
        return Some("move_loop_no_conversion");
    }
    None
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

fn should_trigger_dropoff_watchdog(in_progress: bool, carrying_required: bool, streak: u8) -> bool {
    !in_progress || !carrying_required || streak >= dropoff_watchdog_streak()
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use crate::{
        dispatcher::{BotIntent, Intent},
        dist::DistanceMap,
        model::{BotState, GameState, Grid, Item, Order, OrderStatus},
        team_context::{TeamContext, TeamContextConfig},
        world::World,
    };

    use super::{
        assignment_guard_reason, build_dropoff_staging_cells, merge_hybrid_intents,
        pick_staging_cell, rebalance_pickup_goal_crowding,
        should_trigger_assignment_move_loop_watchdog,
        should_trigger_dropoff_watchdog,
    };

    #[test]
    fn watchdog_triggers_for_pending_or_streak() {
        assert!(should_trigger_dropoff_watchdog(false, true, 1));
        assert!(should_trigger_dropoff_watchdog(true, false, 1));
        assert!(should_trigger_dropoff_watchdog(true, true, 4));
        assert!(!should_trigger_dropoff_watchdog(true, true, 2));
    }

    #[test]
    fn staging_cells_stay_within_ring() {
        let state = GameState {
            grid: Grid {
                width: 7,
                height: 7,
                drop_off_tiles: vec![[3, 3]],
                ..Grid::default()
            },
            ..GameState::default()
        };
        let world = World::new(state);
        let map = world.map();
        let dist = DistanceMap::build(map);
        let drop = map.idx(3, 3).expect("dropoff");
        let cells = build_dropoff_staging_cells(drop, map, &dist, &[]);
        assert!(!cells.is_empty(), "staging cells should be available");
        for cell in &cells {
            let d = dist.dist(*cell, drop);
            assert!((1..=2).contains(&d), "staging cell distance must be 1..2");
        }
    }

    #[test]
    fn pick_staging_respects_reserved_cells() {
        let state = GameState {
            grid: Grid {
                width: 7,
                height: 7,
                drop_off_tiles: vec![[3, 3]],
                ..Grid::default()
            },
            ..GameState::default()
        };
        let world = World::new(state);
        let map = world.map();
        let dist = DistanceMap::build(map);
        let drop = map.idx(3, 3).expect("dropoff");
        let staging = build_dropoff_staging_cells(drop, map, &dist, &[]);
        let mut by_drop = HashMap::new();
        by_drop.insert(drop, staging.clone());
        let mut reserved = HashSet::new();
        reserved.insert(staging[0]);
        let chosen = pick_staging_cell("0", drop, drop, &by_drop, &reserved, map, &dist)
            .expect("staging cell");
        assert_ne!(chosen, staging[0], "reserved staging cell must be skipped");
    }

    #[test]
    fn assignment_guard_detects_empty_dropoff_cluster() {
        let state = GameState {
            grid: Grid {
                width: 8,
                height: 6,
                drop_off_tiles: vec![[1, 4]],
                ..Grid::default()
            },
            bots: vec![
                BotState {
                    id: "0".to_owned(),
                    x: 6,
                    y: 4,
                    carrying: vec![],
                    capacity: 3,
                },
                BotState {
                    id: "1".to_owned(),
                    x: 6,
                    y: 4,
                    carrying: vec![],
                    capacity: 3,
                },
                BotState {
                    id: "2".to_owned(),
                    x: 6,
                    y: 4,
                    carrying: vec![],
                    capacity: 3,
                },
            ],
            items: vec![Item {
                id: "item_1".to_owned(),
                kind: "milk".to_owned(),
                x: 5,
                y: 3,
            }],
            orders: vec![Order {
                id: "o1".to_owned(),
                item_id: "milk".to_owned(),
                status: OrderStatus::InProgress,
            }],
            ..GameState::default()
        };
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let ctx = TeamContext::build(
            &state,
            map,
            &dist,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            TeamContextConfig::default(),
        );
        let drop = map.idx(1, 4).expect("drop");
        let intents = vec![
            BotIntent {
                bot_id: "0".to_owned(),
                intent: Intent::MoveTo { cell: drop },
            },
            BotIntent {
                bot_id: "1".to_owned(),
                intent: Intent::MoveTo { cell: drop },
            },
            BotIntent {
                bot_id: "2".to_owned(),
                intent: Intent::MoveTo { cell: drop },
            },
        ];
        let reason = assignment_guard_reason(&state, map, &ctx, &intents, 0, 0, 8);
        assert_eq!(reason, Some("empty_dropoff_cluster"));
    }

    #[test]
    fn hybrid_merge_prefers_legacy_for_empty_and_global_for_carrier() {
        let state = GameState {
            grid: Grid {
                width: 8,
                height: 6,
                drop_off_tiles: vec![[1, 4]],
                ..Grid::default()
            },
            bots: vec![
                BotState {
                    id: "0".to_owned(),
                    x: 5,
                    y: 4,
                    carrying: vec!["milk".to_owned()],
                    capacity: 3,
                },
                BotState {
                    id: "1".to_owned(),
                    x: 6,
                    y: 4,
                    carrying: vec![],
                    capacity: 3,
                },
            ],
            items: vec![Item {
                id: "item_1".to_owned(),
                kind: "milk".to_owned(),
                x: 4,
                y: 3,
            }],
            orders: vec![Order {
                id: "o1".to_owned(),
                item_id: "milk".to_owned(),
                status: OrderStatus::InProgress,
            }],
            ..GameState::default()
        };
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let ctx = TeamContext::build(
            &state,
            map,
            &dist,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            TeamContextConfig::default(),
        );
        let global = vec![
            BotIntent {
                bot_id: "0".to_owned(),
                intent: Intent::MoveTo {
                    cell: map.idx(1, 4).expect("dropoff"),
                },
            },
            BotIntent {
                bot_id: "1".to_owned(),
                intent: Intent::MoveTo {
                    cell: map.idx(1, 4).expect("dropoff"),
                },
            },
        ];
        let legacy = vec![
            BotIntent {
                bot_id: "0".to_owned(),
                intent: Intent::Wait,
            },
            BotIntent {
                bot_id: "1".to_owned(),
                intent: Intent::MoveTo {
                    cell: map.idx(5, 3).expect("stand"),
                },
            },
        ];
        let merged = merge_hybrid_intents(&state, &ctx, global, &legacy);
        let by_bot = merged
            .into_iter()
            .map(|entry| (entry.bot_id, entry.intent))
            .collect::<HashMap<_, _>>();
        assert!(matches!(
            by_bot.get("0"),
            Some(Intent::MoveTo { cell }) if map.dropoff_cells.contains(cell)
        ));
        assert!(matches!(by_bot.get("1"), Some(Intent::MoveTo { .. })));
        assert!(!matches!(
            by_bot.get("1"),
            Some(Intent::MoveTo { cell }) if map.dropoff_cells.contains(cell)
        ));
    }

    #[test]
    fn move_loop_watchdog_requires_conversion_stall() {
        let intents = vec![
            BotIntent {
                bot_id: "0".to_owned(),
                intent: Intent::MoveTo { cell: 10 },
            },
            BotIntent {
                bot_id: "1".to_owned(),
                intent: Intent::MoveTo { cell: 11 },
            },
            BotIntent {
                bot_id: "2".to_owned(),
                intent: Intent::Wait,
            },
        ];
        assert!(should_trigger_assignment_move_loop_watchdog(&intents, 12, 20));
        assert!(!should_trigger_assignment_move_loop_watchdog(&intents, 2, 20));
    }

    #[test]
    fn crowd_rebalance_spreads_collectors_across_active_stands() {
        let state = GameState {
            grid: Grid {
                width: 7,
                height: 5,
                drop_off_tiles: vec![[0, 4]],
                ..Grid::default()
            },
            bots: vec![
                BotState {
                    id: "0".to_owned(),
                    x: 5,
                    y: 4,
                    carrying: vec![],
                    capacity: 3,
                },
                BotState {
                    id: "1".to_owned(),
                    x: 5,
                    y: 4,
                    carrying: vec![],
                    capacity: 3,
                },
                BotState {
                    id: "2".to_owned(),
                    x: 5,
                    y: 4,
                    carrying: vec![],
                    capacity: 3,
                },
            ],
            items: vec![
                Item {
                    id: "item_1".to_owned(),
                    kind: "milk".to_owned(),
                    x: 4,
                    y: 2,
                },
                Item {
                    id: "item_2".to_owned(),
                    kind: "milk".to_owned(),
                    x: 2,
                    y: 2,
                },
            ],
            orders: vec![Order {
                id: "o1".to_owned(),
                item_id: "milk".to_owned(),
                status: OrderStatus::InProgress,
            }],
            ..GameState::default()
        };
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let team = TeamContext::build(
            &state,
            map,
            &dist,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            TeamContextConfig::default(),
        );
        let target = map.idx(5, 2).expect("stand");
        let mut intents = vec![
            BotIntent {
                bot_id: "0".to_owned(),
                intent: Intent::MoveTo { cell: target },
            },
            BotIntent {
                bot_id: "1".to_owned(),
                intent: Intent::MoveTo { cell: target },
            },
            BotIntent {
                bot_id: "2".to_owned(),
                intent: Intent::MoveTo { cell: target },
            },
        ];
        let active_items = HashSet::from(["milk"]);
        rebalance_pickup_goal_crowding(
            &state,
            map,
            &dist,
            &team,
            &active_items,
            &mut intents,
        );
        let goals = intents
            .iter()
            .filter_map(|entry| match entry.intent {
                Intent::MoveTo { cell } => Some(cell),
                _ => None,
            })
            .collect::<Vec<_>>();
        let distinct = goals.iter().copied().collect::<HashSet<_>>();
        assert!(distinct.len() >= 2, "collectors should be diversified");
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

fn assignment_no_progress_trigger_ticks() -> u8 {
    6
}

fn assignment_forced_legacy_ticks() -> u8 {
    12
}

fn assignment_goal_history_window() -> usize {
    96
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
