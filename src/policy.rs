use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use tracing::info;

use crate::{
    assign::{AssignmentEngine, AssignmentRuntimeHints},
    config::{AssignmentMode, Config},
    dispatcher::{BotIntent, Dispatcher, Intent},
    dist::DistanceMap,
    model::{Action, GameState},
    motion::{MotionPlanner, PlannedAction},
    scoring::{
        detect_mode_label, maybe_score_ordering, maybe_score_ordering_sequence, OrderingFeatures,
    },
    team_context::{
        BlockedMove, BotCommitment, BotRole, DemandTier, StandClaim, StickyQueueRole, TeamContext,
        TeamContextConfig,
    },
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
    post_dropoff_retask_ticks_remaining: u8,
    egress_ticks_remaining: u8,
    last_pickup_attempt: Option<PickupAttempt>,
    pickup_fail_streak: u8,
}

#[derive(Debug, Clone)]
struct PickupAttempt {
    item_id: String,
    stand_cell: u16,
    tick: u64,
    carrying_len: usize,
}

#[derive(Debug, Clone, Copy)]
struct AreaAssignment {
    area_id: u16,
    expires_tick: u64,
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
    stand_claims: HashMap<u16, StandClaim>,
    bot_commitments: HashMap<String, BotCommitment>,
    depleted_item_ids: HashSet<String>,
    pickup_failures_by_item: HashMap<String, u8>,
    preferred_area_by_bot: HashMap<String, AreaAssignment>,
    expansion_mode_by_bot: HashMap<String, bool>,
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
            stand_claims: HashMap::new(),
            bot_commitments: HashMap::new(),
            depleted_item_ids: HashSet::new(),
            pickup_failures_by_item: HashMap::new(),
            preferred_area_by_bot: HashMap::new(),
            expansion_mode_by_bot: HashMap::new(),
        }
    }

    pub fn decide_round(&mut self, state: &GameState, soft_budget: Duration) -> Vec<Action> {
        let tick_started = Instant::now();
        if state.tick == 0 {
            self.depleted_item_ids.clear();
            self.pickup_failures_by_item.clear();
            self.preferred_area_by_bot.clear();
            self.expansion_mode_by_bot.clear();
        }
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

        let newly_depleted = self.update_memory(state, map);
        for item_id in newly_depleted {
            self.depleted_item_ids.insert(item_id);
        }
        if !self.depleted_item_ids.is_empty() {
            self.prune_depleted_targets(state, map);
        }
        self.prune_coordination_state(state, map);
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
        )
        .with_claims(&self.stand_claims, &self.bot_commitments, state.tick);

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
            )
            .with_claims(&self.stand_claims, &self.bot_commitments, state.tick);
        }
        let (
            preferred_area_by_bot,
            expansion_mode_by_bot,
            local_active_candidate_count_by_bot,
            local_radius_by_bot,
        ) = self.compute_locality_hints(state, map, &dist, &team_ctx);
        team_ctx = team_ctx.with_locality(
            &preferred_area_by_bot,
            &expansion_mode_by_bot,
            &local_active_candidate_count_by_bot,
            &local_radius_by_bot,
        );
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
        let mut assignment_goal_concentration_top3 = 0.0f64;
        let mut assignment_late_phase_delivery_streak = 0u16;
        let mut assignment_commitment_reassign_count = 0u16;
        let mut assignment_guard_trigger_reason: Option<&'static str> = None;
        let assignment_mode = if self.config.assignment_enabled {
            self.config.assignment_mode
        } else {
            AssignmentMode::LegacyOnly
        };
        let global_result =
            if matches!(assignment_mode, AssignmentMode::LegacyOnly) || forced_legacy_active {
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
                    self.config.coord_area_balance_weight,
                    AssignmentRuntimeHints {
                        late_phase_delivery_streak: self.ticks_since_dropoff,
                        commitment_reassign_count: 0,
                        ticks_since_pickup: self.ticks_since_pickup,
                        ticks_since_dropoff: self.ticks_since_dropoff,
                        preferred_area_by_bot: preferred_area_by_bot.clone(),
                        expansion_mode_by_bot: expansion_mode_by_bot.clone(),
                        local_active_candidate_count_by_bot: local_active_candidate_count_by_bot
                            .clone(),
                        local_radius_by_bot: local_radius_by_bot.clone(),
                        out_of_area_penalty: self.config.coord_out_of_area_penalty,
                        out_of_radius_penalty: self.config.coord_out_of_radius_penalty,
                    },
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
                            self.config.coord_goal_collapse_threshold,
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
                            self.config.coord_goal_collapse_threshold,
                        ) {
                            assignment_guard_trigger_reason = Some(reason);
                            assignment_source = "legacy_dispatcher_guard";
                            legacy_intents.clone()
                        } else {
                            assignment_source = "hybrid_assignment";
                            merge_hybrid_intents(
                                state,
                                &team_ctx,
                                map,
                                &dist,
                                result.intents,
                                &legacy_intents,
                            )
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
                self.forced_legacy_ticks_remaining = 0;
                self.stand_claims.clear();
                self.bot_commitments.clear();
                assignment_source = if assignment_source == "global_assignment" {
                    "global_watchdog_reassign"
                } else {
                    "hybrid_watchdog_reassign"
                };
                assignment_guard_trigger_reason = Some("global_move_loop_watchdog");
            }
        } else {
            self.global_no_progress_streak = 0;
        };
        rebalance_pickup_goal_crowding(state, map, &dist, &team_ctx, &active_items, &mut intents);
        let post_dropoff_retasked_bots = self.apply_post_dropoff_retask(
            state,
            map,
            &dist,
            &team_ctx,
            &active_items,
            &mut intents,
        );
        let claim_conflicts_resolved =
            self.apply_stand_commitments(state, map, &dist, &team_ctx, &active_items, &mut intents);
        let egress_forced_bots =
            self.apply_recent_dropoff_egress(state, map, &dist, &team_ctx, &mut intents);
        assignment_commitment_reassign_count =
            self.invalidate_stale_commitments(state, &team_ctx, assignment_source);
        assignment_goal_concentration_top3 = goal_concentration_top3(&intents);
        assignment_late_phase_delivery_streak = self.ticks_since_dropoff;
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
        let item_kind_by_id = state
            .items
            .iter()
            .map(|item| (item.id.as_str(), item.kind.as_str()))
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
                    build_dropoff_staging_cells(
                        drop,
                        map,
                        &dist,
                        &team_ctx.traffic.conflict_hotspots,
                    ),
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
                    mem.last_pickup_attempt = None;
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
                    mem.last_pickup_attempt = None;
                    if self.depleted_item_ids.contains(&item_id) {
                        if let Some(kind) = item_kind_by_id.get(item_id.as_str()) {
                            if let Some(goal) = pick_alternate_stand_for_kind(
                                state,
                                map,
                                &dist,
                                cell,
                                kind,
                                &self.depleted_item_ids,
                                Some(cell),
                            ) {
                                goals.insert(bot.id.clone(), goal);
                            } else {
                                immediate.insert(
                                    bot.id.clone(),
                                    PlannedAction {
                                        action: Action::wait(bot.id.clone()),
                                        wait_reason: "pickup_item_depleted",
                                        fallback_stage: "depleted_pickup",
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
                                    wait_reason: "pickup_item_unknown",
                                    fallback_stage: "depleted_pickup",
                                    ordering_stage: "immediate",
                                    path_preview: Vec::new(),
                                },
                            );
                        }
                        self.bot_commitments.remove(&bot.id);
                        self.stand_claims.retain(|_, claim| claim.bot_id != bot.id);
                        continue;
                    }
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
                        mem.last_pickup_attempt = Some(PickupAttempt {
                            item_id: item_id.clone(),
                            stand_cell: cell,
                            tick: state.tick,
                            carrying_len: bot.carrying.len(),
                        });
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
                    mem.last_pickup_attempt = None;
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
                                    dropoff_staging_cell_by_bot
                                        .insert(bot.id.clone(), serde_json::json!([sx, sy]));
                                    dropoff_schedule_status_by_bot.insert(
                                        bot.id.clone(),
                                        serde_json::Value::String("staging_wait_slot".to_owned()),
                                    );
                                } else {
                                    dropoff_schedule_status_by_bot.insert(
                                        bot.id.clone(),
                                        serde_json::Value::String(
                                            "slot_reserved_direct".to_owned(),
                                        ),
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
                    mem.last_pickup_attempt = None;
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
        let pickup_fail_streak_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let streak = self
                    .memory
                    .get(&bot.id)
                    .map(|m| m.pickup_fail_streak)
                    .unwrap_or(0);
                (
                    bot.id.clone(),
                    serde_json::Value::Number(serde_json::Number::from(streak as i64)),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let last_successful_drop_tick_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let tick = self
                    .memory
                    .get(&bot.id)
                    .and_then(|m| m.last_successful_drop_tick)
                    .map(|v| v as i64)
                    .unwrap_or(-1);
                (
                    bot.id.clone(),
                    serde_json::Value::Number(serde_json::Number::from(tick)),
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
                            cells
                                .iter()
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
                "pickup_fail_streak_by_bot".to_owned(),
                serde_json::Value::Object(pickup_fail_streak_by_bot),
            );
            obj.insert(
                "last_successful_drop_tick_by_bot".to_owned(),
                serde_json::Value::Object(last_successful_drop_tick_by_bot),
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
                    assignment_active_task_count as i64,
                )),
            );
            obj.insert(
                "assignment_preview_task_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment_preview_task_count as i64,
                )),
            );
            obj.insert(
                "assignment_stand_task_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment_stand_task_count as i64,
                )),
            );
            obj.insert(
                "assignment_dropoff_task_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment_dropoff_task_count as i64,
                )),
            );
            obj.insert(
                "assignment_active_gap_total".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment_active_gap_total as i64,
                )),
            );
            obj.insert(
                "assignment_preview_enabled".to_owned(),
                serde_json::Value::Bool(assignment_preview_enabled),
            );
            obj.insert(
                "assignment_goal_concentration_top3".to_owned(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(assignment_goal_concentration_top3)
                        .unwrap_or_else(|| serde_json::Number::from(0)),
                ),
            );
            obj.insert(
                "assignment_late_phase_delivery_streak".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment_late_phase_delivery_streak as i64,
                )),
            );
            obj.insert(
                "assignment_commitment_reassign_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment_commitment_reassign_count as i64,
                )),
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
                "depleted_item_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.depleted_item_ids.len() as i64
                )),
            );
            obj.insert(
                "pickup_failure_item_count".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.pickup_failures_by_item.len() as i64,
                )),
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
                serde_json::Value::Number(serde_json::Number::from(
                    self.ticks_since_dropoff as i64,
                )),
            );
            obj.insert(
                "unique_goal_cells_last_n".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.unique_goal_cells_recent() as i64,
                )),
            );
            obj.insert(
                "forced_legacy_ticks_remaining".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.forced_legacy_ticks_remaining as i64,
                )),
            );
            obj.insert(
                "global_no_progress_streak".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.global_no_progress_streak as i64,
                )),
            );
            obj.insert(
                "post_dropoff_retasked_bots".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    post_dropoff_retasked_bots as i64,
                )),
            );
            obj.insert(
                "claim_conflicts_resolved".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(
                    claim_conflicts_resolved as i64,
                )),
            );
            obj.insert(
                "egress_forced_bots".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(egress_forced_bots as i64)),
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

    fn compute_locality_hints(
        &mut self,
        state: &GameState,
        map: &crate::world::MapCache,
        dist: &DistanceMap,
        team: &TeamContext,
    ) -> (
        HashMap<String, u16>,
        HashMap<String, bool>,
        HashMap<String, u16>,
        HashMap<String, u16>,
    ) {
        let mut active_stands_by_area = HashMap::<u16, Vec<u16>>::new();
        for (kind, gap) in &team.knowledge.active_gap_by_kind {
            if *gap == 0 {
                continue;
            }
            let Some(stands) = team.knowledge.kind_to_stands.get(kind) else {
                continue;
            };
            for &stand in stands {
                let area = team
                    .knowledge
                    .area_id_by_cell
                    .get(stand as usize)
                    .copied()
                    .unwrap_or(u16::MAX);
                active_stands_by_area.entry(area).or_default().push(stand);
            }
        }
        for stands in active_stands_by_area.values_mut() {
            stands.sort_unstable();
            stands.dedup();
        }

        let mut min_dropoff_dist_by_area = HashMap::<u16, u16>::new();
        for (area, stands) in &active_stands_by_area {
            let mut best = u16::MAX;
            for &stand in stands {
                for &drop in &map.dropoff_cells {
                    best = best.min(dist.dist(stand, drop));
                }
            }
            min_dropoff_dist_by_area.insert(*area, best);
        }

        let mut preferred_area_by_bot = HashMap::<String, u16>::new();
        let mut expansion_mode_by_bot = HashMap::<String, bool>::new();
        let mut local_active_candidate_count_by_bot = HashMap::<String, u16>::new();
        let mut local_radius_by_bot = HashMap::<String, u16>::new();

        let mode_adjust = mode_radius_adjustment(team.mode.as_str());
        let base_radius = i32::from(self.config.coord_local_radius_base) + mode_adjust;
        let base_radius = base_radius.clamp(4, i32::from(self.config.coord_local_radius_max)) as u16;
        let ttl = u64::from(self.config.coord_preferred_area_ttl_ticks);

        let mut known_bot_ids = HashSet::<String>::new();
        let mut bots = state.bots.iter().collect::<Vec<_>>();
        bots.sort_by(|a, b| a.id.cmp(&b.id));
        for bot in bots {
            known_bot_ids.insert(bot.id.clone());
            let role = team.role_for(&bot.id);
            let is_courier = matches!(role, BotRole::LeadCourier | BotRole::QueueCourier);
            let blocked = team
                .bot_snapshot
                .get(&bot.id)
                .map(|snap| snap.blocked_ticks)
                .unwrap_or(0);
            let local_radius = base_radius.min(u16::from(self.config.coord_local_radius_max));
            let local_radius_f = f64::from(local_radius.max(1));
            let Some(start) = map.idx(bot.x, bot.y) else {
                preferred_area_by_bot.insert(bot.id.clone(), u16::MAX);
                expansion_mode_by_bot.insert(bot.id.clone(), true);
                local_active_candidate_count_by_bot.insert(bot.id.clone(), 0);
                local_radius_by_bot.insert(bot.id.clone(), local_radius);
                continue;
            };

            let mut preferred = self
                .preferred_area_by_bot
                .get(&bot.id)
                .copied()
                .filter(|entry| {
                    entry.expires_tick >= state.tick
                        && active_stands_by_area
                            .get(&entry.area_id)
                            .map(|stands| !stands.is_empty())
                            .unwrap_or(false)
                })
                .map(|entry| entry.area_id);

            if preferred.is_none() {
                let mut candidates = active_stands_by_area.keys().copied().collect::<Vec<_>>();
                candidates.sort_unstable();
                let mut best = None::<(i32, u16)>;
                for area_id in candidates {
                    let Some(stands) = active_stands_by_area.get(&area_id) else {
                        continue;
                    };
                    if stands.is_empty() {
                        continue;
                    }
                    let min_dist = stands
                        .iter()
                        .copied()
                        .map(|stand| dist.dist(start, stand))
                        .min()
                        .unwrap_or(u16::MAX);
                    let min_drop = min_dropoff_dist_by_area
                        .get(&area_id)
                        .copied()
                        .unwrap_or(u16::MAX);
                    let area_load = team
                        .knowledge
                        .area_load_by_id
                        .get(&area_id)
                        .copied()
                        .unwrap_or(0);
                    let score = i32::from(min_dist.min(200)) * 2
                        + i32::from(min_drop.min(200))
                        + i32::from(area_load) * 3;
                    match best {
                        Some((best_score, best_area))
                            if score > best_score
                                || (score == best_score && area_id > best_area) => {}
                        _ => best = Some((score, area_id)),
                    }
                }
                preferred = best.map(|(_, area)| area);
            }

            let preferred_area = preferred.unwrap_or(u16::MAX);
            self.preferred_area_by_bot.insert(
                bot.id.clone(),
                AreaAssignment {
                    area_id: preferred_area,
                    expires_tick: state.tick.saturating_add(ttl),
                },
            );
            let local_count = active_stands_by_area
                .get(&preferred_area)
                .map(|stands| {
                    stands
                        .iter()
                        .copied()
                        .filter(|stand| f64::from(dist.dist(start, *stand)) <= local_radius_f)
                        .count() as u16
                })
                .unwrap_or(0);
            let expansion = is_courier
                || local_count == 0
                || blocked >= 3
                || self.ticks_since_pickup >= u16::from(self.config.coord_expansion_stall_ticks);
            self.expansion_mode_by_bot.insert(bot.id.clone(), expansion);
            preferred_area_by_bot.insert(bot.id.clone(), preferred_area);
            expansion_mode_by_bot.insert(bot.id.clone(), expansion);
            local_active_candidate_count_by_bot.insert(bot.id.clone(), local_count);
            local_radius_by_bot.insert(bot.id.clone(), local_radius);
        }

        self.preferred_area_by_bot
            .retain(|bot_id, _| known_bot_ids.contains(bot_id));
        self.expansion_mode_by_bot
            .retain(|bot_id, _| known_bot_ids.contains(bot_id));

        (
            preferred_area_by_bot,
            expansion_mode_by_bot,
            local_active_candidate_count_by_bot,
            local_radius_by_bot,
        )
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

    fn prune_coordination_state(&mut self, state: &GameState, map: &crate::world::MapCache) {
        let now = state.tick;
        let mut valid_stands = HashSet::<u16>::new();
        for item in &state.items {
            for &stand in map.stand_cells_for_item(&item.id) {
                valid_stands.insert(stand);
            }
        }
        self.stand_claims
            .retain(|cell, claim| claim.expires_tick >= now && valid_stands.contains(cell));
        let known_bots = state
            .bots
            .iter()
            .map(|bot| bot.id.as_str())
            .collect::<HashSet<_>>();
        self.bot_commitments
            .retain(|bot_id, _| known_bots.contains(bot_id.as_str()));
        let valid_goal_cells = self.stand_claims.keys().copied().collect::<HashSet<_>>();
        self.bot_commitments
            .retain(|_, c| valid_goal_cells.contains(&c.goal_cell));
    }

    fn prune_depleted_targets(&mut self, state: &GameState, map: &crate::world::MapCache) {
        if self.depleted_item_ids.is_empty() {
            return;
        }
        let mut viable_stands = HashSet::<u16>::new();
        for item in &state.items {
            if self.depleted_item_ids.contains(&item.id) {
                continue;
            }
            for &stand in map.stand_cells_for_item(&item.id) {
                viable_stands.insert(stand);
            }
        }
        self.stand_claims
            .retain(|cell, _| viable_stands.contains(cell));
        self.bot_commitments
            .retain(|_, commitment| viable_stands.contains(&commitment.goal_cell));
    }

    fn invalidate_stale_commitments(
        &mut self,
        state: &GameState,
        team: &TeamContext,
        assignment_source: &str,
    ) -> u16 {
        if !matches!(assignment_source, "hybrid_assignment" | "global_assignment") {
            return 0;
        }
        let mut removed = Vec::<String>::new();
        let mut removed_count = 0u16;
        for bot in &state.bots {
            let Some(commitment) = self.bot_commitments.get(&bot.id) else {
                continue;
            };
            let carrying_active = bot
                .carrying
                .iter()
                .any(|item| team.active_order_items_set.contains(item));
            if carrying_active || bot.carrying.len() >= bot.capacity {
                continue;
            }
            if matches!(
                team.role_for(&bot.id),
                BotRole::LeadCourier | BotRole::QueueCourier
            ) {
                continue;
            }
            let since = state.tick.saturating_sub(commitment.last_progress_tick);
            if since < u64::from(self.config.coord_reassign_no_progress_ticks) {
                continue;
            }
            removed.push(bot.id.clone());
        }
        for bot_id in removed {
            if let Some(c) = self.bot_commitments.remove(&bot_id) {
                self.stand_claims
                    .retain(|cell, claim| *cell != c.goal_cell && claim.bot_id != bot_id);
                removed_count = removed_count.saturating_add(1);
            }
        }
        removed_count
    }

    fn apply_post_dropoff_retask(
        &mut self,
        state: &GameState,
        map: &crate::world::MapCache,
        dist: &DistanceMap,
        team: &TeamContext,
        active_items: &HashSet<&str>,
        intents: &mut [BotIntent],
    ) -> u16 {
        if active_items.is_empty() {
            return 0;
        }
        let active_stand_count = team
            .knowledge
            .active_gap_by_kind
            .iter()
            .filter(|(_, gap)| **gap > 0)
            .filter_map(|(kind, _)| team.knowledge.kind_to_stands.get(kind))
            .flat_map(|stands| stands.iter().copied())
            .collect::<HashSet<_>>()
            .len();
        let sparse_stands = active_stand_count <= state.bots.len().saturating_add(1) / 2;
        let max_per_stand = if sparse_stands {
            self.config.coord_max_bots_per_stand.max(2)
        } else {
            self.config.coord_max_bots_per_stand
        };
        let mut idx_by_bot = HashMap::<String, usize>::new();
        for (idx, entry) in intents.iter().enumerate() {
            idx_by_bot.insert(entry.bot_id.clone(), idx);
        }
        let mut retasked = 0u16;
        let mut bot_ids = state.bots.iter().map(|b| b.id.clone()).collect::<Vec<_>>();
        bot_ids.sort();
        for bot_id in bot_ids {
            let Some(mem) = self.memory.get_mut(&bot_id) else {
                continue;
            };
            if mem.post_dropoff_retask_ticks_remaining == 0 {
                continue;
            }
            let Some(bot) = state.bots.iter().find(|b| b.id == bot_id) else {
                continue;
            };
            let carrying_active = bot
                .carrying
                .iter()
                .any(|item| team.active_order_items_set.contains(item));
            if carrying_active || bot.carrying.len() >= bot.capacity {
                continue;
            }
            if matches!(
                team.role_for(&bot.id),
                BotRole::LeadCourier | BotRole::QueueCourier
            ) {
                continue;
            }
            let Some(start) = map.idx(bot.x, bot.y) else {
                continue;
            };
            let Some((goal_cell, item_kind)) = choose_best_active_stand(
                start,
                team,
                map,
                dist,
                state,
                &self.stand_claims,
                &bot_id,
                self.config.coord_area_balance_weight,
                max_per_stand,
                state.tick,
                &self.depleted_item_ids,
            ) else {
                continue;
            };
            let Some(&idx) = idx_by_bot.get(&bot.id) else {
                continue;
            };
            intents[idx].intent = build_pick_or_move_intent(
                state,
                map,
                dist,
                start,
                goal_cell,
                &item_kind,
                &self.depleted_item_ids,
            );
            self.bot_commitments.insert(
                bot.id.clone(),
                BotCommitment {
                    goal_cell,
                    item_kind: Some(item_kind),
                    created_tick: state.tick,
                    last_progress_tick: state.tick,
                },
            );
            self.stand_claims.insert(
                goal_cell,
                StandClaim {
                    bot_id: bot.id.clone(),
                    expires_tick: state
                        .tick
                        .saturating_add(u64::from(self.config.coord_claim_ttl_ticks)),
                    demand_tier: DemandTier::Active,
                },
            );
            retasked = retasked.saturating_add(1);
        }
        retasked
    }

    fn apply_stand_commitments(
        &mut self,
        state: &GameState,
        map: &crate::world::MapCache,
        dist: &DistanceMap,
        team: &TeamContext,
        active_items: &HashSet<&str>,
        intents: &mut [BotIntent],
    ) -> u16 {
        if active_items.is_empty() {
            return 0;
        }
        let mut idx_by_bot = HashMap::<String, usize>::new();
        for (idx, entry) in intents.iter().enumerate() {
            idx_by_bot.insert(entry.bot_id.clone(), idx);
        }
        let mut active_stands = HashSet::<u16>::new();
        for item in &state.items {
            if !active_items.contains(item.kind.as_str()) {
                continue;
            }
            if self.depleted_item_ids.contains(&item.id) {
                continue;
            }
            for &stand in map.stand_cells_for_item(&item.id) {
                active_stands.insert(stand);
            }
        }
        if active_stands.is_empty() {
            self.stand_claims.clear();
            self.bot_commitments.clear();
            return 0;
        }
        let sparse_stands = active_stands.len() <= state.bots.len().saturating_add(1) / 2;
        let max_per_stand = if sparse_stands {
            self.config.coord_max_bots_per_stand.max(2)
        } else {
            self.config.coord_max_bots_per_stand
        };

        let mut conflicts_resolved = 0u16;
        let mut bots = state.bots.iter().collect::<Vec<_>>();
        bots.sort_by(|a, b| a.id.cmp(&b.id));
        for bot in bots {
            if matches!(
                team.role_for(&bot.id),
                BotRole::LeadCourier | BotRole::QueueCourier
            ) {
                continue;
            }
            let Some(&intent_idx) = idx_by_bot.get(&bot.id) else {
                continue;
            };
            let carries_active = bot
                .carrying
                .iter()
                .any(|item| team.active_order_items_set.contains(item));
            if carries_active || bot.carrying.len() >= bot.capacity {
                continue;
            }
            let Some(start) = map.idx(bot.x, bot.y) else {
                continue;
            };
            let chosen = match intents[intent_idx].intent {
                Intent::MoveTo { cell } if active_stands.contains(&cell) => {
                    let claimed_by_other = self
                        .stand_claims
                        .get(&cell)
                        .map(|claim| claim.bot_id != bot.id && claim.expires_tick >= state.tick)
                        .unwrap_or(false);
                    if claimed_by_other {
                        conflicts_resolved = conflicts_resolved.saturating_add(1);
                        choose_best_active_stand(
                            start,
                            team,
                            map,
                            dist,
                            state,
                            &self.stand_claims,
                            &bot.id,
                            self.config.coord_area_balance_weight,
                            max_per_stand,
                            state.tick,
                            &self.depleted_item_ids,
                        )
                        .map(|(cell, kind)| (cell, kind, true))
                    } else {
                        team.knowledge
                            .stand_to_kind
                            .get(&cell)
                            .cloned()
                            .map(|kind| (cell, kind, false))
                    }
                }
                _ => {
                    if let Some(commitment) = self.bot_commitments.get(&bot.id) {
                        if active_stands.contains(&commitment.goal_cell) {
                            commitment
                                .item_kind
                                .clone()
                                .or_else(|| {
                                    team.knowledge
                                        .stand_to_kind
                                        .get(&commitment.goal_cell)
                                        .cloned()
                                })
                                .map(|kind| (commitment.goal_cell, kind, false))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
            };
            let Some((mut goal, mut kind, rerouted)) = chosen else {
                continue;
            };
            let current_load = self
                .stand_claims
                .iter()
                .filter(|(cell, claim)| {
                    **cell == goal && claim.expires_tick >= state.tick && claim.bot_id != bot.id
                })
                .count() as u8;
            if current_load >= max_per_stand {
                if !rerouted {
                    conflicts_resolved = conflicts_resolved.saturating_add(1);
                }
                if let Some((alt_goal, alt_kind)) = choose_best_active_stand(
                    start,
                    team,
                    map,
                    dist,
                    state,
                    &self.stand_claims,
                    &bot.id,
                    self.config.coord_area_balance_weight,
                    max_per_stand,
                    state.tick,
                    &self.depleted_item_ids,
                ) {
                    goal = alt_goal;
                    kind = alt_kind;
                } else {
                    continue;
                }
            }
            intents[intent_idx].intent = build_pick_or_move_intent(
                state,
                map,
                dist,
                start,
                goal,
                &kind,
                &self.depleted_item_ids,
            );
            self.stand_claims.insert(
                goal,
                StandClaim {
                    bot_id: bot.id.clone(),
                    expires_tick: state
                        .tick
                        .saturating_add(u64::from(self.config.coord_claim_ttl_ticks)),
                    demand_tier: DemandTier::Active,
                },
            );
            self.bot_commitments.insert(
                bot.id.clone(),
                BotCommitment {
                    goal_cell: goal,
                    item_kind: Some(kind),
                    created_tick: state.tick,
                    last_progress_tick: state.tick,
                },
            );
        }
        conflicts_resolved
    }

    fn apply_recent_dropoff_egress(
        &mut self,
        state: &GameState,
        map: &crate::world::MapCache,
        dist: &DistanceMap,
        team: &TeamContext,
        intents: &mut [BotIntent],
    ) -> u16 {
        let mut idx_by_bot = HashMap::<String, usize>::new();
        for (idx, entry) in intents.iter().enumerate() {
            idx_by_bot.insert(entry.bot_id.clone(), idx);
        }
        let ring = team
            .queue
            .ring_cells
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let lane = team
            .queue
            .lane_cells
            .iter()
            .take(6)
            .copied()
            .collect::<HashSet<_>>();
        let drop_targets = map.dropoff_cells.iter().copied().collect::<HashSet<_>>();
        let active_targets = team
            .knowledge
            .active_gap_by_kind
            .iter()
            .filter(|(_, gap)| **gap > 0)
            .filter_map(|(kind, _)| team.knowledge.kind_to_stands.get(kind))
            .flat_map(|stands| stands.iter().copied())
            .collect::<HashSet<_>>();
        let item_kind_by_id = state
            .items
            .iter()
            .map(|item| (item.id.as_str(), item.kind.as_str()))
            .collect::<HashMap<_, _>>();
        if ring.is_empty() && lane.is_empty() {
            return 0;
        }
        let mut forced = 0u16;
        for bot in &state.bots {
            let Some(mem) = self.memory.get(&bot.id) else {
                continue;
            };
            if mem.egress_ticks_remaining == 0 || !bot.carrying.is_empty() {
                continue;
            }
            if matches!(
                team.role_for(&bot.id),
                BotRole::LeadCourier | BotRole::QueueCourier
            ) {
                continue;
            }
            let Some(cell) = map.idx(bot.x, bot.y) else {
                continue;
            };
            if !(ring.contains(&cell) || lane.contains(&cell)) {
                continue;
            }
            let Some(&idx) = idx_by_bot.get(&bot.id) else {
                continue;
            };
            if intent_is_active_aligned(bot, &intents[idx].intent, team, &item_kind_by_id) {
                continue;
            }
            let candidate = map.neighbors[cell as usize]
                .iter()
                .copied()
                .filter(|next| !ring.contains(next) && !lane.contains(next))
                .min_by(|a, b| {
                    let active_a = nearest_dist_to_targets(*a, &active_targets, dist);
                    let active_b = nearest_dist_to_targets(*b, &active_targets, dist);
                    let drop_a = nearest_dist_to_targets(*a, &drop_targets, dist);
                    let drop_b = nearest_dist_to_targets(*b, &drop_targets, dist);
                    active_a
                        .cmp(&active_b)
                        .then_with(|| drop_a.cmp(&drop_b))
                        .then_with(|| {
                            map.neighbors[*b as usize]
                                .len()
                                .cmp(&map.neighbors[*a as usize].len())
                        })
                        .then_with(|| a.cmp(b))
                });
            if let Some(next) = candidate {
                intents[idx].intent = Intent::MoveTo { cell: next };
                forced = forced.saturating_add(1);
            }
        }
        forced
    }

    fn update_memory(&mut self, state: &GameState, map: &crate::world::MapCache) -> Vec<String> {
        let mut newly_depleted = Vec::<String>::new();
        for bot in &state.bots {
            let cell = map.idx(bot.x, bot.y).unwrap_or(0);
            let mem = self.memory.entry(bot.id.clone()).or_default();
            mem.constraint_relax_ticks_remaining =
                mem.constraint_relax_ticks_remaining.saturating_sub(1);
            mem.escape_macro_ticks_remaining = mem.escape_macro_ticks_remaining.saturating_sub(1);
            mem.post_dropoff_retask_ticks_remaining =
                mem.post_dropoff_retask_ticks_remaining.saturating_sub(1);
            mem.egress_ticks_remaining = mem.egress_ticks_remaining.saturating_sub(1);
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

            if let Some(attempt) = mem.last_pickup_attempt.take() {
                if state.tick > attempt.tick {
                    let succeeded = bot.carrying.len() > attempt.carrying_len;
                    if succeeded {
                        mem.pickup_fail_streak = 0;
                        self.pickup_failures_by_item.remove(&attempt.item_id);
                    } else if bot.carrying.len() <= attempt.carrying_len
                        && attempt.stand_cell == cell
                    {
                        mem.pickup_fail_streak = mem.pickup_fail_streak.saturating_add(1);
                        let fail = self
                            .pickup_failures_by_item
                            .entry(attempt.item_id.clone())
                            .or_insert(0);
                        *fail = fail.saturating_add(1);
                        if *fail >= pickup_deplete_fail_threshold() {
                            newly_depleted.push(attempt.item_id);
                        }
                    } else {
                        mem.pickup_fail_streak = 0;
                    }
                } else {
                    mem.last_pickup_attempt = Some(attempt);
                }
            }

            if !mem.prev_carrying.is_empty() && bot.carrying.len() < mem.prev_carrying.len() {
                mem.same_dropoff_order_streak = 0;
                mem.last_dropoff_order_id = None;
                mem.last_successful_drop_tick = Some(state.tick);
                mem.dropoff_ban_ticks_by_order.clear();
                mem.post_dropoff_retask_ticks_remaining =
                    self.config.coord_post_dropoff_retask_ticks;
                mem.egress_ticks_remaining = 1;
                if let Some(commitment) = self.bot_commitments.get_mut(&bot.id) {
                    commitment.last_progress_tick = state.tick;
                }
                self.bot_commitments.remove(&bot.id);
                self.stand_claims.retain(|_, claim| claim.bot_id != bot.id);
            } else if bot.carrying.len() > mem.prev_carrying.len() {
                if let Some(commitment) = self.bot_commitments.get_mut(&bot.id) {
                    commitment.last_progress_tick = state.tick;
                }
            }

            mem.failed_move_history.retain(|_, count| *count > 0);
            mem.repeated_failed_moves =
                mem.failed_move_history.values().copied().max().unwrap_or(0);
            mem.prev_cell = Some(cell);
            mem.prev_carrying = bot.carrying.clone();
        }
        newly_depleted.sort();
        newly_depleted.dedup();
        newly_depleted
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
    map: &crate::world::MapCache,
    dist: &DistanceMap,
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
    let item_kind_by_id = state
        .items
        .iter()
        .map(|item| (item.id.as_str(), item.kind.as_str()))
        .collect::<HashMap<_, _>>();
    let active_gap_total = team_ctx
        .knowledge
        .active_gap_by_kind
        .values()
        .copied()
        .map(usize::from)
        .sum::<usize>();
    let active_phase = active_gap_total > 0;
    let mut bots = state.bots.iter().collect::<Vec<_>>();
    bots.sort_by(|a, b| a.id.cmp(&b.id));
    bots.into_iter()
        .map(|bot| {
            let carrying_active = bot
                .carrying
                .iter()
                .any(|item| team_ctx.active_order_items_set.contains(item));
            let global_intent = global_by_bot.get(&bot.id).cloned().unwrap_or(Intent::Wait);
            let legacy_intent = legacy_by_bot.get(&bot.id).cloned().unwrap_or(Intent::Wait);
            let intent = if carrying_active {
                prefer_non_wait(global_intent, legacy_intent)
            } else if active_phase {
                merge_hybrid_active_phase(
                    bot,
                    &global_intent,
                    &legacy_intent,
                    team_ctx,
                    &item_kind_by_id,
                    map,
                    dist,
                )
            } else {
                prefer_non_wait(legacy_intent, global_intent)
            };
            BotIntent {
                bot_id: bot.id.clone(),
                intent: intent.clone(),
            }
        })
        .collect()
}

fn prefer_non_wait(primary: Intent, fallback: Intent) -> Intent {
    if !matches!(primary, Intent::Wait) {
        primary
    } else {
        fallback
    }
}

fn merge_hybrid_active_phase(
    bot: &crate::model::BotState,
    global: &Intent,
    legacy: &Intent,
    team_ctx: &TeamContext,
    item_kind_by_id: &HashMap<&str, &str>,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
) -> Intent {
    let global_active = intent_is_active_aligned(bot, global, team_ctx, item_kind_by_id);
    let legacy_active = intent_is_active_aligned(bot, legacy, team_ctx, item_kind_by_id);
    let legacy_preview = intent_targets_preview(legacy, team_ctx, item_kind_by_id);
    let bot_carrying_only_inactive = !bot.carrying.is_empty()
        && !bot
            .carrying
            .iter()
            .any(|item| team_ctx.active_order_items_set.contains(item));

    if bot_carrying_only_inactive && bot.carrying.len() >= bot.capacity {
        if global_active {
            return global.clone();
        }
        let Some(bot_cell) = map.idx(bot.x, bot.y) else {
            return Intent::Wait;
        };
        let nearest_drop_dist = map
            .dropoff_cells
            .iter()
            .copied()
            .map(|drop| dist.dist(bot_cell, drop))
            .min()
            .unwrap_or(u16::MAX);
        let in_dropoff_zone = team_ctx.dropoff_control_zone.contains(&bot_cell)
            || map.dropoff_cells.contains(&bot_cell);
        if nearest_drop_dist <= full_inactive_stage_trigger_dist() && in_dropoff_zone {
            return Intent::Wait;
        }
        if let Some(stage) = nearest_dropoff_stage_cell(bot_cell, map, dist, team_ctx) {
            if stage != bot_cell {
                return Intent::MoveTo { cell: stage };
            }
        }
        return Intent::Wait;
    }

    match (global_active, legacy_active) {
        (true, false) => return global.clone(),
        (false, true) => return legacy.clone(),
        (true, true) => return prefer_non_wait(global.clone(), legacy.clone()),
        (false, false) => {}
    }

    if legacy_preview {
        return if !matches!(global, Intent::Wait) {
            global.clone()
        } else {
            Intent::Wait
        };
    }

    if !matches!(global, Intent::Wait) && matches!(legacy, Intent::Wait) {
        return global.clone();
    }

    legacy.clone()
}

fn nearest_dropoff_stage_cell(
    bot_cell: u16,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
    team_ctx: &TeamContext,
) -> Option<u16> {
    let mut candidates = Vec::<u16>::new();
    candidates.extend(team_ctx.queue.lane_cells.iter().copied());
    candidates.extend(team_ctx.queue.ring_cells.iter().copied());
    candidates.extend(map.dropoff_cells.iter().copied());
    candidates.sort_unstable();
    candidates.dedup();
    candidates
        .into_iter()
        .filter_map(|cell| {
            let d_bot = dist.dist(bot_cell, cell);
            if d_bot == u16::MAX {
                return None;
            }
            let d_drop = map
                .dropoff_cells
                .iter()
                .copied()
                .map(|drop| dist.dist(cell, drop))
                .filter(|&d| d != u16::MAX)
                .min()
                .unwrap_or(u16::MAX);
            Some((d_bot, d_drop, cell))
        })
        .min_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        })
        .map(|(_, _, cell)| cell)
}

fn intent_targets_preview(
    intent: &Intent,
    team_ctx: &TeamContext,
    item_kind_by_id: &HashMap<&str, &str>,
) -> bool {
    match intent {
        Intent::PickUp { item_id } => item_kind_by_id
            .get(item_id.as_str())
            .map(|kind| !team_ctx.active_order_items_set.contains(*kind))
            .unwrap_or(false),
        Intent::MoveTo { cell } => team_ctx
            .knowledge
            .stand_to_kind
            .get(cell)
            .map(|kind| !team_ctx.active_order_items_set.contains(kind))
            .unwrap_or(false),
        _ => false,
    }
}

fn full_inactive_stage_trigger_dist() -> u16 {
    6
}

fn intent_is_active_aligned(
    bot: &crate::model::BotState,
    intent: &Intent,
    team_ctx: &TeamContext,
    item_kind_by_id: &HashMap<&str, &str>,
) -> bool {
    match intent {
        Intent::DropOff { .. } => true,
        Intent::PickUp { item_id } => item_kind_by_id
            .get(item_id.as_str())
            .map(|kind| team_ctx.active_order_items_set.contains(*kind))
            .unwrap_or(false),
        Intent::MoveTo { cell } => {
            if let Some(kind) = team_ctx.knowledge.stand_to_kind.get(cell) {
                team_ctx.active_order_items_set.contains(kind)
            } else if team_ctx.queue.ring_cells.contains(cell)
                || team_ctx.queue.lane_cells.contains(cell)
            {
                bot.carrying
                    .iter()
                    .any(|item| team_ctx.active_order_items_set.contains(item))
            } else {
                false
            }
        }
        Intent::Wait => false,
    }
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
        crowd_by_goal.entry(cell).or_default().push(bot.id.clone());
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

fn build_pick_or_move_intent(
    state: &GameState,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
    bot_cell: u16,
    goal_cell: u16,
    item_kind: &str,
    depleted_item_ids: &HashSet<String>,
) -> Intent {
    if bot_cell != goal_cell {
        return Intent::MoveTo { cell: goal_cell };
    }
    let mut item_ids = state
        .items
        .iter()
        .filter(|item| item.kind == item_kind)
        .filter(|item| !depleted_item_ids.contains(&item.id))
        .filter_map(|item| {
            let stands = map.stand_cells_for_item(&item.id);
            if stands.contains(&goal_cell) {
                Some(item.id.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    item_ids.sort();
    if let Some(item_id) = item_ids.first() {
        Intent::PickUp {
            item_id: item_id.clone(),
        }
    } else {
        let next = pick_alternate_stand_for_kind(
            state,
            map,
            dist,
            bot_cell,
            item_kind,
            depleted_item_ids,
            Some(goal_cell),
        );
        next.map(|cell| Intent::MoveTo { cell })
            .unwrap_or(Intent::Wait)
    }
}

fn choose_best_active_stand(
    start: u16,
    team: &TeamContext,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
    state: &GameState,
    stand_claims: &HashMap<u16, StandClaim>,
    bot_id: &str,
    area_balance_weight: f64,
    max_per_stand: u8,
    now_tick: u64,
    depleted_item_ids: &HashSet<String>,
) -> Option<(u16, String)> {
    let mut kinds = team
        .knowledge
        .active_gap_by_kind
        .iter()
        .filter(|(_, gap)| **gap > 0)
        .map(|(kind, _)| kind.clone())
        .collect::<Vec<_>>();
    kinds.sort();
    let mut best = None::<(i32, u16, String)>;
    for kind in kinds {
        let Some(stands) = team.knowledge.kind_to_stands.get(&kind) else {
            continue;
        };
        for &stand in stands {
            if !stand_has_viable_item(state, map, stand, &kind, depleted_item_ids) {
                continue;
            }
            let claim_load = stand_claims
                .iter()
                .filter(|(cell, claim)| {
                    **cell == stand && claim.expires_tick >= now_tick && claim.bot_id != bot_id
                })
                .count() as u8;
            if claim_load >= max_per_stand {
                continue;
            }
            let d = i32::from(dist.dist(start, stand).min(255));
            let area = team
                .knowledge
                .area_id_by_cell
                .get(stand as usize)
                .copied()
                .unwrap_or(u16::MAX);
            let area_load = team
                .knowledge
                .area_load_by_id
                .get(&area)
                .copied()
                .unwrap_or(0);
            let choke = 4u16.saturating_sub(map.neighbors[stand as usize].len().min(4) as u16);
            let score = d
                + (area_balance_weight * f64::from(area_load)).round() as i32
                + i32::from(claim_load) * 2
                + i32::from(choke);
            match best {
                Some((best_score, best_cell, ref best_kind))
                    if score > best_score
                        || (score == best_score
                            && (stand > best_cell
                                || (stand == best_cell && kind > *best_kind))) => {}
                _ => {
                    best = Some((score, stand, kind.clone()));
                }
            }
        }
    }
    best.map(|(_, cell, kind)| (cell, kind))
}

fn stand_has_viable_item(
    state: &GameState,
    map: &crate::world::MapCache,
    stand: u16,
    item_kind: &str,
    depleted_item_ids: &HashSet<String>,
) -> bool {
    state
        .items
        .iter()
        .filter(|item| item.kind == item_kind)
        .filter(|item| !depleted_item_ids.contains(&item.id))
        .any(|item| map.stand_cells_for_item(&item.id).contains(&stand))
}

fn pick_alternate_stand_for_kind(
    state: &GameState,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
    start: u16,
    item_kind: &str,
    depleted_item_ids: &HashSet<String>,
    exclude: Option<u16>,
) -> Option<u16> {
    let mut stands = state
        .items
        .iter()
        .filter(|item| item.kind == item_kind)
        .filter(|item| !depleted_item_ids.contains(&item.id))
        .flat_map(|item| map.stand_cells_for_item(&item.id).iter().copied())
        .collect::<Vec<_>>();
    stands.sort_unstable();
    stands.dedup();
    if let Some(ex) = exclude {
        stands.retain(|cell| *cell != ex);
    }
    stands.sort_by(|a, b| {
        dist.dist(start, *a)
            .cmp(&dist.dist(start, *b))
            .then_with(|| a.cmp(b))
    });
    stands.into_iter().next()
}

fn nearest_dist_to_targets(cell: u16, targets: &HashSet<u16>, dist: &DistanceMap) -> u16 {
    if targets.is_empty() {
        return u16::MAX;
    }
    targets
        .iter()
        .map(|target| dist.dist(cell, *target))
        .min()
        .unwrap_or(u16::MAX)
}

fn goal_concentration_top3(intents: &[BotIntent]) -> f64 {
    let mut counts = HashMap::<u16, u64>::new();
    let mut total = 0u64;
    for intent in intents {
        let Intent::MoveTo { cell } = intent.intent else {
            continue;
        };
        *counts.entry(cell).or_insert(0) += 1;
        total = total.saturating_add(1);
    }
    if total == 0 {
        return 0.0;
    }
    let mut top = counts.values().copied().collect::<Vec<_>>();
    top.sort_unstable_by(|a, b| b.cmp(a));
    let top3 = top.into_iter().take(3).sum::<u64>();
    top3 as f64 / total as f64
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
    goal_collapse_threshold: usize,
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
    if empty_capacity_bots > 0
        && pickup_progress_intents == 0
        && stand_move_intents == 0
        && ticks_since_pickup >= 8
    {
        return Some("no_pickup_progress");
    }
    if pickup_progress_intents == 0
        && dropoff_progress_intents == 0
        && stand_move_intents > 0
        && ticks_since_pickup >= 12
        && ticks_since_dropoff >= 16
        && unique_goal_cells_last_n <= goal_collapse_threshold.max(2)
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
        let carrying_preview = bot
            .carrying
            .iter()
            .any(|item| team.pending_order_items_set.contains(item));
        let active_gap_total = team
            .knowledge
            .active_gap_by_kind
            .values()
            .copied()
            .map(u32::from)
            .sum::<u32>();
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
        if carrying_preview && active_gap_total <= 2 {
            score += 12.0;
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
        if !carrying_active
            && active_gap_total == 0
            && matches!(role, BotRole::Collector | BotRole::Yield | BotRole::Idle)
        {
            let mut nearest_preview = u16::MAX;
            let mut pending = team.pending_order_items_set.iter().collect::<Vec<_>>();
            pending.sort();
            for kind in pending {
                if let Some(stands) = team.knowledge.kind_to_stands.get(kind.as_str()) {
                    for &stand in stands {
                        nearest_preview = nearest_preview.min(dist.dist(start, stand));
                    }
                }
            }
            if nearest_preview != u16::MAX {
                score += (24.0 - f64::from(nearest_preview.min(24))) * 0.8;
            }
        }

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
        let sequence_score = maybe_score_ordering_sequence(
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
        score += sequence_score;
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
        assignment_guard_reason, build_dropoff_staging_cells, build_pick_or_move_intent,
        merge_hybrid_intents, pick_alternate_stand_for_kind, pick_staging_cell,
        rebalance_pickup_goal_crowding, should_trigger_assignment_move_loop_watchdog,
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
        let reason = assignment_guard_reason(&state, map, &ctx, &intents, 0, 0, 8, 4);
        assert_eq!(reason, Some("empty_dropoff_cluster"));
    }

    #[test]
    fn assignment_guard_does_not_trigger_when_bots_are_stand_seeking() {
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
        let stand = map.idx(5, 2).expect("stand");
        let intents = vec![
            BotIntent {
                bot_id: "0".to_owned(),
                intent: Intent::MoveTo { cell: stand },
            },
            BotIntent {
                bot_id: "1".to_owned(),
                intent: Intent::MoveTo { cell: stand },
            },
        ];
        let reason = assignment_guard_reason(&state, map, &ctx, &intents, 20, 20, 3, 4);
        assert_ne!(reason, Some("no_pickup_progress"));
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
        let merged = merge_hybrid_intents(&state, &ctx, map, &dist, global, &legacy);
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
    fn hybrid_merge_blocks_preview_pickups_when_active_gap_exists() {
        let state = GameState {
            grid: Grid {
                width: 9,
                height: 7,
                drop_off_tiles: vec![[1, 5]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 5,
                y: 5,
                carrying: vec![],
                capacity: 3,
            }],
            items: vec![
                Item {
                    id: "item_milk".to_owned(),
                    kind: "milk".to_owned(),
                    x: 4,
                    y: 3,
                },
                Item {
                    id: "item_bread".to_owned(),
                    kind: "bread".to_owned(),
                    x: 7,
                    y: 3,
                },
            ],
            orders: vec![
                Order {
                    id: "o_active".to_owned(),
                    item_id: "milk".to_owned(),
                    status: OrderStatus::InProgress,
                },
                Order {
                    id: "o_preview".to_owned(),
                    item_id: "bread".to_owned(),
                    status: OrderStatus::Pending,
                },
            ],
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
        let milk_stand = map.stand_cells_for_item("item_milk")[0];
        let global = vec![BotIntent {
            bot_id: "0".to_owned(),
            intent: Intent::MoveTo { cell: milk_stand },
        }];
        let legacy = vec![BotIntent {
            bot_id: "0".to_owned(),
            intent: Intent::PickUp {
                item_id: "item_bread".to_owned(),
            },
        }];
        let merged = merge_hybrid_intents(&state, &team, map, &dist, global, &legacy);
        assert_eq!(merged.len(), 1);
        assert!(matches!(
            merged[0].intent,
            Intent::MoveTo { cell } if cell == milk_stand
        ));
    }

    #[test]
    fn hybrid_merge_stages_full_inactive_carrier_during_active_gap() {
        let state = GameState {
            grid: Grid {
                width: 9,
                height: 7,
                drop_off_tiles: vec![[1, 5]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 8,
                y: 1,
                carrying: vec!["bread".to_owned()],
                capacity: 1,
            }],
            items: vec![
                Item {
                    id: "item_milk".to_owned(),
                    kind: "milk".to_owned(),
                    x: 4,
                    y: 3,
                },
                Item {
                    id: "item_bread".to_owned(),
                    kind: "bread".to_owned(),
                    x: 7,
                    y: 3,
                },
            ],
            orders: vec![
                Order {
                    id: "o_active".to_owned(),
                    item_id: "milk".to_owned(),
                    status: OrderStatus::InProgress,
                },
                Order {
                    id: "o_preview".to_owned(),
                    item_id: "bread".to_owned(),
                    status: OrderStatus::Pending,
                },
            ],
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
        let preview_stand = map.stand_cells_for_item("item_bread")[0];
        let global = vec![BotIntent {
            bot_id: "0".to_owned(),
            intent: Intent::Wait,
        }];
        let legacy = vec![BotIntent {
            bot_id: "0".to_owned(),
            intent: Intent::MoveTo {
                cell: preview_stand,
            },
        }];
        let merged = merge_hybrid_intents(&state, &team, map, &dist, global, &legacy);
        assert_eq!(merged.len(), 1);
        assert!(matches!(merged[0].intent, Intent::MoveTo { .. }));
        if let Intent::MoveTo { cell } = merged[0].intent {
            assert!(
                team.queue.ring_cells.contains(&cell)
                    || team.queue.lane_cells.contains(&cell)
                    || map.dropoff_cells.contains(&cell)
            );
        }
    }

    #[test]
    fn hybrid_merge_allows_wait_for_full_inactive_inside_dropoff_zone() {
        let state = GameState {
            grid: Grid {
                width: 9,
                height: 7,
                drop_off_tiles: vec![[1, 5]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 1,
                y: 5,
                carrying: vec!["bread".to_owned()],
                capacity: 1,
            }],
            items: vec![
                Item {
                    id: "item_milk".to_owned(),
                    kind: "milk".to_owned(),
                    x: 4,
                    y: 3,
                },
                Item {
                    id: "item_bread".to_owned(),
                    kind: "bread".to_owned(),
                    x: 7,
                    y: 3,
                },
            ],
            orders: vec![
                Order {
                    id: "o_active".to_owned(),
                    item_id: "milk".to_owned(),
                    status: OrderStatus::InProgress,
                },
                Order {
                    id: "o_preview".to_owned(),
                    item_id: "bread".to_owned(),
                    status: OrderStatus::Pending,
                },
            ],
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
        let preview_stand = map.stand_cells_for_item("item_bread")[0];
        let global = vec![BotIntent {
            bot_id: "0".to_owned(),
            intent: Intent::Wait,
        }];
        let legacy = vec![BotIntent {
            bot_id: "0".to_owned(),
            intent: Intent::MoveTo {
                cell: preview_stand,
            },
        }];
        let merged = merge_hybrid_intents(&state, &team, map, &dist, global, &legacy);
        assert_eq!(merged.len(), 1);
        assert!(matches!(merged[0].intent, Intent::Wait));
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
        assert!(should_trigger_assignment_move_loop_watchdog(
            &intents, 12, 20
        ));
        assert!(!should_trigger_assignment_move_loop_watchdog(
            &intents, 2, 20
        ));
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
        rebalance_pickup_goal_crowding(&state, map, &dist, &team, &active_items, &mut intents);
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

    #[test]
    fn alternate_stand_selection_skips_depleted_items() {
        let state = GameState {
            grid: Grid {
                width: 7,
                height: 5,
                drop_off_tiles: vec![[0, 4]],
                ..Grid::default()
            },
            items: vec![
                Item {
                    id: "item_1".to_owned(),
                    kind: "milk".to_owned(),
                    x: 2,
                    y: 2,
                },
                Item {
                    id: "item_2".to_owned(),
                    kind: "milk".to_owned(),
                    x: 4,
                    y: 2,
                },
            ],
            ..GameState::default()
        };
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let start = map.stand_cells_for_item("item_1")[0];
        let depleted = HashSet::from(["item_1".to_owned()]);
        let alt = pick_alternate_stand_for_kind(
            &state,
            map,
            &dist,
            start,
            "milk",
            &depleted,
            Some(start),
        )
        .expect("alternate stand");
        assert_ne!(alt, start);
        assert!(
            map.stand_cells_for_item("item_2").contains(&alt),
            "alternate stand must come from non-depleted item"
        );
    }

    #[test]
    fn build_pick_or_move_reroutes_when_goal_item_is_depleted() {
        let state = GameState {
            grid: Grid {
                width: 7,
                height: 5,
                drop_off_tiles: vec![[0, 4]],
                ..Grid::default()
            },
            items: vec![
                Item {
                    id: "item_1".to_owned(),
                    kind: "milk".to_owned(),
                    x: 2,
                    y: 2,
                },
                Item {
                    id: "item_2".to_owned(),
                    kind: "milk".to_owned(),
                    x: 4,
                    y: 2,
                },
            ],
            ..GameState::default()
        };
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let goal = map.stand_cells_for_item("item_1")[0];
        let depleted = HashSet::from(["item_1".to_owned()]);
        let intent = build_pick_or_move_intent(&state, map, &dist, goal, goal, "milk", &depleted);
        assert!(matches!(intent, Intent::MoveTo { .. }));
        let Intent::MoveTo { cell } = intent else {
            panic!("expected move intent");
        };
        assert_ne!(cell, goal);
        assert!(
            map.stand_cells_for_item("item_2").contains(&cell),
            "reroute should target non-depleted stand"
        );
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

fn mode_radius_adjustment(mode: &str) -> i32 {
    match mode {
        "easy" => -1,
        "medium" => 0,
        "hard" => 1,
        "expert" => 2,
        _ => 0,
    }
}

fn assignment_no_progress_trigger_ticks() -> u8 {
    10
}

fn assignment_goal_history_window() -> usize {
    96
}

fn pickup_deplete_fail_threshold() -> u8 {
    2
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
