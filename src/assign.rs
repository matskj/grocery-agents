use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::dispatcher::{BotIntent, Intent};
use crate::dist::DistanceMap;
use crate::model::{BotState, GameState, OrderStatus};
use crate::scoring::{maybe_score_pick, CandidateFeatures};
use crate::team_context::{BotRole, DemandTier as TeamDemandTier, TeamContext};
use crate::world::MapCache;

#[derive(Debug, Clone)]
pub struct AssignmentResult {
    pub intents: Vec<BotIntent>,
    pub task_count: usize,
    pub edge_count: usize,
    pub active_task_count: usize,
    pub preview_task_count: usize,
    pub stand_task_count: usize,
    pub dropoff_task_count: usize,
    pub active_gap_total: usize,
    pub preview_enabled: bool,
    pub goal_concentration_top3: f64,
    pub late_phase_delivery_streak: u16,
    pub commitment_reassign_count: u16,
}

#[derive(Debug, Clone, Default)]
pub struct AssignmentRuntimeHints {
    pub late_phase_delivery_streak: u16,
    pub commitment_reassign_count: u16,
    pub ticks_since_pickup: u16,
    pub ticks_since_dropoff: u16,
    pub preferred_area_by_bot: HashMap<String, u16>,
    pub expansion_mode_by_bot: HashMap<String, bool>,
    pub local_active_candidate_count_by_bot: HashMap<String, u16>,
    pub local_radius_by_bot: HashMap<String, u16>,
    pub out_of_area_penalty: f64,
    pub out_of_radius_penalty: f64,
}

#[derive(Debug, Clone)]
struct Task {
    task_id: usize,
    sort_key: (u8, String, u16),
    target_cell: u16,
    shareable: bool,
    kind: TaskKind,
    demand_tier: DemandTier,
    item_kind: Option<String>,
    nearest_drop_dist: u16,
    intent: Intent,
}

impl Task {
    fn target_share_cap(&self, sparse_active_stands: bool) -> u8 {
        if !self.shareable
            && sparse_active_stands
            && matches!(self.demand_tier, DemandTier::Active)
            && matches!(self.kind, TaskKind::PickupStand | TaskKind::ImmediatePickup)
        {
            2
        } else {
            1
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskKind {
    ImmediateDropOff,
    CarryToDropoff,
    PickupStand,
    ImmediatePickup,
    QueuePosition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DemandTier {
    None,
    Active,
    Preview,
}

#[derive(Debug, Clone)]
struct Edge {
    bot_id: String,
    task_id: usize,
    cost: i32,
}

#[derive(Debug, Clone)]
struct StandCandidate {
    stand: u16,
    item_id: String,
    nearest_drop_dist: u16,
}

#[derive(Debug, Clone)]
struct TaskBuildResult {
    tasks: Vec<Task>,
    active_gap_by_kind: HashMap<String, u16>,
    active_gap_total: usize,
    preview_enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct AssignmentEngine;

impl AssignmentEngine {
    pub fn new() -> Self {
        Self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_intents(
        &self,
        state: &GameState,
        map: &MapCache,
        dist: &DistanceMap,
        team: &TeamContext,
        candidate_k: usize,
        lambda_density: f64,
        lambda_choke: f64,
        area_balance_weight: f64,
        runtime_hints: AssignmentRuntimeHints,
        soft_budget: Duration,
    ) -> Option<AssignmentResult> {
        let started = Instant::now();
        let built = build_tasks(state, map, dist, team)?;
        let tasks = built.tasks;
        if tasks.is_empty() {
            return Some(AssignmentResult {
                intents: state
                    .bots
                    .iter()
                    .map(|bot| BotIntent {
                        bot_id: bot.id.clone(),
                        intent: Intent::Wait,
                    })
                    .collect(),
                task_count: 0,
                edge_count: 0,
                active_task_count: 0,
                preview_task_count: 0,
                stand_task_count: 0,
                dropoff_task_count: 0,
                active_gap_total: built.active_gap_total,
                preview_enabled: built.preview_enabled,
                goal_concentration_top3: 0.0,
                late_phase_delivery_streak: runtime_hints.late_phase_delivery_streak,
                commitment_reassign_count: runtime_hints.commitment_reassign_count,
            });
        }

        let active_task_count = tasks
            .iter()
            .filter(|task| matches!(task.demand_tier, DemandTier::Active))
            .count();
        let preview_task_count = tasks
            .iter()
            .filter(|task| matches!(task.demand_tier, DemandTier::Preview))
            .count();
        let stand_task_count = tasks
            .iter()
            .filter(|task| matches!(task.kind, TaskKind::PickupStand | TaskKind::ImmediatePickup))
            .count();
        let dropoff_task_count = tasks
            .iter()
            .filter(|task| {
                matches!(
                    task.kind,
                    TaskKind::ImmediateDropOff | TaskKind::CarryToDropoff
                )
            })
            .count();

        let mut bots = state.bots.iter().collect::<Vec<_>>();
        bots.sort_by(|a, b| a.id.cmp(&b.id));

        let order_urgency = team
            .order_snapshot
            .active_remaining_by_item
            .values()
            .copied()
            .sum::<u64>() as f64;

        let mut all_edges = Vec::<Edge>::new();
        for bot in &bots {
            if started.elapsed() + Duration::from_millis(4) >= soft_budget {
                return None;
            }
            let Some(bot_cell) = map.idx(bot.x, bot.y) else {
                continue;
            };
            let carrying_active = bot
                .carrying
                .iter()
                .any(|item| team.active_order_items_set.contains(item));
            let bot_has_capacity = bot.carrying.len() < bot.capacity;
            let role = team.role_for(&bot.id);
            let queue_goal = team.queue_goal_for(&bot.id);
            let queue_distance = team
                .queue
                .assignments
                .get(&bot.id)
                .map(|assignment| f64::from(assignment.queue_distance))
                .unwrap_or(99.0);
            let inventory_util = if bot.capacity == 0 {
                0.0
            } else {
                bot.carrying.len() as f64 / bot.capacity as f64
            };
            let teammate_proximity = avg_teammate_distance(bot, &state.bots);
            let local_congestion = team
                .traffic
                .local_conflict_count_by_bot
                .get(&bot.id)
                .copied()
                .map(f64::from)
                .unwrap_or(0.0);
            let local_conflict_count = local_congestion;
            let blocked_ticks = team
                .bot_snapshot
                .get(&bot.id)
                .map(|snap| f64::from(snap.blocked_ticks))
                .unwrap_or(0.0);
            let serviceable_dropoff = if team
                .dropoff_serviceable_by_bot
                .get(&bot.id)
                .copied()
                .unwrap_or(false)
            {
                1.0
            } else {
                0.0
            };
            let on_dropoff = map
                .idx(bot.x, bot.y)
                .map(|cell| map.dropoff_cells.contains(&cell))
                .unwrap_or(false);
            let preferred_area = runtime_hints.preferred_area_by_bot.get(&bot.id).copied();
            let expansion_mode = runtime_hints
                .expansion_mode_by_bot
                .get(&bot.id)
                .copied()
                .unwrap_or(false);
            let local_active_candidate_count = f64::from(
                runtime_hints
                    .local_active_candidate_count_by_bot
                    .get(&bot.id)
                    .copied()
                    .unwrap_or(0),
            );
            let local_radius = f64::from(
                runtime_hints
                    .local_radius_by_bot
                    .get(&bot.id)
                    .copied()
                    .unwrap_or(1)
                    .max(1),
            );
            let enforce_local_first = !matches!(role, BotRole::LeadCourier | BotRole::QueueCourier);

            let mut ranked = tasks
                .iter()
                .filter(|task| {
                    if matches!(task.kind, TaskKind::CarryToDropoff) && !carrying_active {
                        return false;
                    }
                    if matches!(task.kind, TaskKind::PickupStand | TaskKind::ImmediatePickup)
                        && !bot_has_capacity
                    {
                        return false;
                    }
                    if matches!(task.kind, TaskKind::PickupStand | TaskKind::ImmediatePickup)
                        && enforce_local_first
                    {
                        let target_area = team
                            .knowledge
                            .area_id_by_cell
                            .get(task.target_cell as usize)
                            .copied()
                            .unwrap_or(u16::MAX);
                        let stand_claimed_by_other = team
                            .knowledge
                            .stand_claims
                            .get(&task.target_cell)
                            .map(|claim| claim.bot_id != bot.id && claim.expires_tick >= state.tick)
                            .unwrap_or(false);
                        if stand_claimed_by_other && !expansion_mode {
                            return false;
                        }
                        let dist_to_stand = f64::from(dist_to(bot_cell, task.target_cell, dist));
                        let in_area = preferred_area
                            .map(|area| area == target_area)
                            .unwrap_or(true);
                        let in_radius = dist_to_stand <= local_radius;
                        if !expansion_mode && (!in_area || !in_radius) {
                            return false;
                        }
                    }
                    if matches!(task.kind, TaskKind::QueuePosition)
                        && !matches!(role, BotRole::LeadCourier | BotRole::QueueCourier)
                    {
                        return false;
                    }
                    if matches!(task.kind, TaskKind::ImmediateDropOff) {
                        let Intent::DropOff { order_id } = &task.intent else {
                            return false;
                        };
                        let required_item = state
                            .orders
                            .iter()
                            .find(|order| order.id == *order_id)
                            .map(|order| order.item_id.clone());
                        if !on_dropoff {
                            return false;
                        }
                        let Some(required_item) = required_item else {
                            return false;
                        };
                        if !bot.carrying.iter().any(|item| item == &required_item) {
                            return false;
                        }
                    }
                    true
                })
                .map(|task| {
                    let base = task_cost(
                        bot_cell,
                        task,
                        map,
                        dist,
                        state,
                        lambda_density,
                        lambda_choke,
                    );
                    let mut adjusted = base;
                    if matches!(task.kind, TaskKind::PickupStand | TaskKind::ImmediatePickup) {
                        let target_area = team
                            .knowledge
                            .area_id_by_cell
                            .get(task.target_cell as usize)
                            .copied()
                            .unwrap_or(u16::MAX);
                        let area_load = team
                            .knowledge
                            .area_load_by_id
                            .get(&target_area)
                            .copied()
                            .unwrap_or(0);
                        adjusted += (area_balance_weight * f64::from(area_load)).round() as i32;
                        if let Some(claim) = team.knowledge.stand_claims.get(&task.target_cell) {
                            if claim.bot_id != bot.id && claim.expires_tick >= state.tick {
                                adjusted += match claim.demand_tier {
                                    TeamDemandTier::Active => 120,
                                    TeamDemandTier::Preview => 80,
                                    TeamDemandTier::None => 40,
                                };
                            }
                        }
                        if let Some(commitment) = team.knowledge.bot_commitments.get(&bot.id) {
                            if commitment.goal_cell == task.target_cell {
                                adjusted -= 22;
                            } else if commitment.item_kind.as_deref() == task.item_kind.as_deref() {
                                adjusted -= 8;
                            } else {
                                adjusted += 10;
                            }
                        }
                    }
                    if carrying_active && matches!(task.kind, TaskKind::CarryToDropoff) {
                        adjusted -= 30;
                    }
                    if matches!(task.kind, TaskKind::ImmediateDropOff) {
                        adjusted -= 80;
                    }
                    if !carrying_active
                        && matches!(task.kind, TaskKind::PickupStand)
                        && matches!(task.demand_tier, DemandTier::Active)
                    {
                        adjusted -= 25;
                    }
                    if matches!(task.kind, TaskKind::ImmediatePickup) {
                        adjusted -= if matches!(task.demand_tier, DemandTier::Active) {
                            110
                        } else {
                            55
                        };
                    }
                    if matches!(role, BotRole::LeadCourier | BotRole::QueueCourier)
                        && matches!(task.intent, Intent::MoveTo { .. })
                        && queue_goal == Some(task.target_cell)
                    {
                        adjusted -= 15;
                    }
                    if matches!(task.kind, TaskKind::PickupStand | TaskKind::ImmediatePickup) {
                        let dist_to_stand = f64::from(dist_to(bot_cell, task.target_cell, dist));
                        let bot_dropoff_dist =
                            f64::from(nearest_drop_dist(bot_cell, map, dist).min(64));
                        let roundtrip_eta =
                            dist_to_stand + 0.7 * f64::from(task.nearest_drop_dist.min(64));
                        let target_area = team
                            .knowledge
                            .area_id_by_cell
                            .get(task.target_cell as usize)
                            .copied()
                            .unwrap_or(u16::MAX);
                        let preferred_area_match = preferred_area
                            .map(|area| area == target_area)
                            .unwrap_or(false);
                        let out_of_area_target = preferred_area
                            .map(|area| area != target_area)
                            .unwrap_or(false);
                        let out_of_radius_target = dist_to_stand > local_radius;
                        let stand_claimed_by_other = team
                            .knowledge
                            .stand_claims
                            .get(&task.target_cell)
                            .map(|claim| claim.bot_id != bot.id)
                            .unwrap_or(false);
                        let commitment_same_stand = team
                            .knowledge
                            .bot_commitments
                            .values()
                            .filter(|commitment| commitment.goal_cell == task.target_cell)
                            .count() as f64;
                        let commitment_same_kind = task
                            .item_kind
                            .as_ref()
                            .map(|kind| {
                                team.knowledge
                                    .bot_commitments
                                    .values()
                                    .filter(|commitment| {
                                        commitment.item_kind.as_ref() == Some(kind)
                                    })
                                    .count() as f64
                            })
                            .unwrap_or(0.0);
                        let stand_cooldown = if stand_claimed_by_other { 6.0 } else { 0.0 };
                        let contention = commitment_same_stand + 0.5 * commitment_same_kind;
                        let time_since_last_conversion = f64::from(
                            runtime_hints
                                .ticks_since_pickup
                                .min(runtime_hints.ticks_since_dropoff),
                        );
                        let last_conversion_was_pickup = if runtime_hints.ticks_since_pickup
                            <= runtime_hints.ticks_since_dropoff
                        {
                            1.0
                        } else {
                            0.0
                        };
                        let features = CandidateFeatures {
                            dist_to_nearest_active_item: dist_to_stand,
                            dist_to_dropoff: f64::from(task.nearest_drop_dist.min(64)),
                            inventory_util,
                            queue_distance,
                            local_congestion,
                            local_conflict_count,
                            teammate_proximity,
                            order_urgency,
                            blocked_ticks,
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
                            queue_role_collector: if matches!(role, BotRole::Collector) {
                                1.0
                            } else {
                                0.0
                            },
                            queue_role_yield: if matches!(role, BotRole::Yield) {
                                1.0
                            } else {
                                0.0
                            },
                            serviceable_dropoff,
                            stand_failure_count_recent: 0.0,
                            stand_success_count_recent: 0.0,
                            stand_cooldown_ticks_remaining: stand_cooldown,
                            kind_failure_count_recent: 0.0,
                            repeated_same_stand_no_delta_streak: 0.0,
                            contention_at_stand_proxy: contention,
                            time_since_last_conversion_tick: time_since_last_conversion,
                            last_conversion_was_pickup,
                            last_conversion_was_dropoff: 1.0 - last_conversion_was_pickup,
                            preferred_area_match: if preferred_area_match { 1.0 } else { 0.0 },
                            expansion_mode_active: if expansion_mode { 1.0 } else { 0.0 },
                            local_active_candidate_count,
                            local_radius,
                            out_of_area_target: if out_of_area_target { 1.0 } else { 0.0 },
                            out_of_radius_target: if out_of_radius_target { 1.0 } else { 0.0 },
                        };
                        let mode = team.mode.as_str();
                        let model_score = maybe_score_pick(mode, features);
                        if let Some(score) = model_score {
                            adjusted -= (score.combined_expected_score * 10.0).round() as i32;
                            if score.pickup_prob < 0.35 {
                                adjusted += ((0.35 - score.pickup_prob) * 120.0).round() as i32;
                            }
                            if stand_cooldown > 0.0 {
                                adjusted += (stand_cooldown * 4.0).round() as i32;
                            }
                            if contention > 0.0 {
                                adjusted += (contention * 10.0).round() as i32;
                            }
                        }
                        if expansion_mode && out_of_area_target {
                            adjusted += runtime_hints.out_of_area_penalty.round() as i32;
                        }
                        if expansion_mode && out_of_radius_target {
                            let overrun =
                                ((dist_to_stand - local_radius) / local_radius).clamp(0.0, 3.0);
                            adjusted +=
                                (runtime_hints.out_of_radius_penalty * overrun).round() as i32;
                        }
                        let pickup_stall_ticks = runtime_hints.ticks_since_pickup.saturating_sub(8);
                        if pickup_stall_ticks > 0 {
                            adjusted += roundtrip_idle_pressure_penalty(
                                roundtrip_eta,
                                bot_dropoff_dist,
                                f64::from(pickup_stall_ticks),
                            );
                        }
                    }
                    (adjusted, task.task_id)
                })
                .collect::<Vec<_>>();
            ranked.sort_by(|a, b| a.cmp(b));
            ranked.truncate(candidate_k.max(1));
            for (cost, task_id) in ranked {
                all_edges.push(Edge {
                    bot_id: bot.id.clone(),
                    task_id,
                    cost,
                });
            }
        }

        if all_edges.len() > 4_096 {
            return None;
        }
        all_edges.sort_by(|a, b| {
            a.cost
                .cmp(&b.cost)
                .then_with(|| a.bot_id.cmp(&b.bot_id))
                .then_with(|| a.task_id.cmp(&b.task_id))
        });

        let active_stands = tasks
            .iter()
            .filter(|task| {
                matches!(task.demand_tier, DemandTier::Active)
                    && matches!(task.kind, TaskKind::PickupStand | TaskKind::ImmediatePickup)
            })
            .map(|task| task.target_cell)
            .collect::<HashSet<_>>();
        let sparse_active_stands = !active_stands.is_empty()
            && active_stands.len() <= state.bots.len().saturating_add(1) / 2;
        let active_kind_cap = built
            .active_gap_by_kind
            .iter()
            .map(|(kind, gap)| (kind.clone(), (*gap).max(1)))
            .collect::<HashMap<_, _>>();

        let task_by_id = tasks
            .iter()
            .map(|task| (task.task_id, task.clone()))
            .collect::<HashMap<_, _>>();
        let mut bot_taken = HashSet::<String>::new();
        let mut task_taken = HashSet::<usize>::new();
        let mut target_cell_taken = HashMap::<u16, u8>::new();
        let mut active_kind_taken = HashMap::<String, u16>::new();
        let mut assigned = HashMap::<String, Intent>::new();
        for edge in &all_edges {
            if bot_taken.contains(&edge.bot_id) {
                continue;
            }
            let Some(task) = task_by_id.get(&edge.task_id) else {
                continue;
            };
            if !task.shareable && task_taken.contains(&edge.task_id) {
                continue;
            }
            if !task.shareable {
                let used = target_cell_taken
                    .get(&task.target_cell)
                    .copied()
                    .unwrap_or(0);
                let cap = task.target_share_cap(sparse_active_stands);
                if used >= cap {
                    continue;
                }
            }
            if matches!(task.demand_tier, DemandTier::Active)
                && matches!(task.kind, TaskKind::PickupStand | TaskKind::ImmediatePickup)
            {
                if let Some(kind) = task.item_kind.as_ref() {
                    let used = active_kind_taken.get(kind).copied().unwrap_or(0);
                    let cap = active_kind_cap.get(kind).copied().unwrap_or(1);
                    if used >= cap {
                        continue;
                    }
                }
            }
            bot_taken.insert(edge.bot_id.clone());
            if !task.shareable {
                task_taken.insert(edge.task_id);
                let used = target_cell_taken
                    .get(&task.target_cell)
                    .copied()
                    .unwrap_or(0);
                target_cell_taken.insert(task.target_cell, used.saturating_add(1));
            }
            if matches!(task.demand_tier, DemandTier::Active)
                && matches!(task.kind, TaskKind::PickupStand | TaskKind::ImmediatePickup)
            {
                if let Some(kind) = task.item_kind.as_ref() {
                    let used = active_kind_taken.get(kind).copied().unwrap_or(0);
                    active_kind_taken.insert(kind.clone(), used.saturating_add(1));
                }
            }
            assigned.insert(edge.bot_id.clone(), task.intent.clone());
        }

        let intents = state
            .bots
            .iter()
            .map(|bot| BotIntent {
                bot_id: bot.id.clone(),
                intent: assigned.remove(&bot.id).unwrap_or(Intent::Wait),
            })
            .collect::<Vec<_>>();
        let goal_concentration_top3 = goal_concentration_top3(&intents);

        Some(AssignmentResult {
            intents,
            task_count: tasks.len(),
            edge_count: all_edges.len(),
            active_task_count,
            preview_task_count,
            stand_task_count,
            dropoff_task_count,
            active_gap_total: built.active_gap_total,
            preview_enabled: built.preview_enabled,
            goal_concentration_top3,
            late_phase_delivery_streak: runtime_hints.late_phase_delivery_streak,
            commitment_reassign_count: runtime_hints.commitment_reassign_count,
        })
    }
}

fn build_tasks(
    state: &GameState,
    map: &MapCache,
    dist: &DistanceMap,
    team: &TeamContext,
) -> Option<TaskBuildResult> {
    if state.bots.is_empty() {
        return Some(TaskBuildResult {
            tasks: Vec::new(),
            active_gap_by_kind: HashMap::new(),
            active_gap_total: 0,
            preview_enabled: true,
        });
    }
    let mut tasks = Vec::<Task>::new();
    let mut next_id = 0usize;
    let active_carrier_exists = state.bots.iter().any(|bot| {
        bot.carrying
            .iter()
            .any(|item| team.active_order_items_set.contains(item))
    });

    let mut active_missing = HashMap::<String, u16>::new();
    let mut preview_missing = HashMap::<String, u16>::new();
    for order in &state.orders {
        match order.status {
            OrderStatus::InProgress => {
                *active_missing.entry(order.item_id.clone()).or_insert(0) += 1;
            }
            OrderStatus::Pending => {
                *preview_missing.entry(order.item_id.clone()).or_insert(0) += 1;
            }
            OrderStatus::Delivered | OrderStatus::Cancelled => {}
        }
    }
    let mut active_gap_by_kind = if team.knowledge.active_gap_by_kind.is_empty() {
        let effective_supply = build_effective_active_supply(state, map, dist, team);
        let mut computed = HashMap::<String, u16>::new();
        for (kind, missing) in &active_missing {
            let covered = effective_supply.get(kind).copied().unwrap_or(0);
            computed.insert(kind.clone(), missing.saturating_sub(covered));
        }
        computed
    } else {
        team.knowledge.active_gap_by_kind.clone()
    };
    for kind in active_missing.keys() {
        active_gap_by_kind.entry(kind.clone()).or_insert(0);
    }
    let active_gap_total = active_gap_by_kind
        .values()
        .copied()
        .map(usize::from)
        .sum::<usize>();
    let preview_quota = preview_prefetch_quota(active_gap_total);
    let preview_enabled = preview_quota > 0 && !preview_missing.is_empty();

    let mut active_orders = state
        .orders
        .iter()
        .filter(|order| matches!(order.status, OrderStatus::InProgress))
        .collect::<Vec<_>>();
    active_orders.sort_by(|a, b| a.id.cmp(&b.id));
    for order in active_orders {
        let order_item = order.item_id.as_str();
        for bot in &state.bots {
            let on_dropoff = map
                .idx(bot.x, bot.y)
                .map(|cell| map.dropoff_cells.contains(&cell))
                .unwrap_or(false);
            if on_dropoff && bot.carrying.iter().any(|item| item == order_item) {
                tasks.push(Task {
                    task_id: next_id,
                    sort_key: (0, order.item_id.clone(), 0),
                    target_cell: map.idx(bot.x, bot.y).unwrap_or(0),
                    shareable: true,
                    kind: TaskKind::ImmediateDropOff,
                    demand_tier: DemandTier::None,
                    item_kind: Some(order.item_id.clone()),
                    nearest_drop_dist: 0,
                    intent: Intent::DropOff {
                        order_id: order.id.clone(),
                    },
                });
                next_id += 1;
            }
        }
    }

    if active_carrier_exists {
        let mut dropoffs = map.dropoff_cells.clone();
        dropoffs.sort_unstable();
        for drop in dropoffs {
            let (x, y) = map.xy(drop);
            let order_id = state
                .orders
                .iter()
                .filter(|order| matches!(order.status, OrderStatus::InProgress))
                .map(|order| order.id.clone())
                .min()
                .unwrap_or_else(|| format!("dropoff@{x}:{y}"));
            tasks.push(Task {
                task_id: next_id,
                sort_key: (1, order_id.clone(), drop),
                target_cell: drop,
                shareable: true,
                kind: TaskKind::CarryToDropoff,
                demand_tier: DemandTier::None,
                item_kind: None,
                nearest_drop_dist: 0,
                intent: Intent::MoveTo { cell: drop },
            });
            next_id += 1;
        }
    }

    let stand_pool = build_stand_pool(state, map, dist);

    let mut active_kinds = active_missing.keys().cloned().collect::<Vec<_>>();
    active_kinds.sort();
    for kind in active_kinds {
        let demand = active_gap_by_kind.get(&kind).copied().unwrap_or(0);
        if demand == 0 {
            continue;
        }
        let Some(pool) = stand_pool.get(&kind) else {
            continue;
        };
        for candidate in pool {
            tasks.push(Task {
                task_id: next_id,
                sort_key: (2, kind.clone(), candidate.stand),
                target_cell: candidate.stand,
                shareable: false,
                kind: TaskKind::PickupStand,
                demand_tier: DemandTier::Active,
                item_kind: Some(kind.clone()),
                nearest_drop_dist: candidate.nearest_drop_dist,
                intent: Intent::MoveTo {
                    cell: candidate.stand,
                },
            });
            next_id += 1;
            tasks.push(Task {
                task_id: next_id,
                sort_key: (2, kind.clone(), candidate.stand),
                target_cell: candidate.stand,
                shareable: false,
                kind: TaskKind::ImmediatePickup,
                demand_tier: DemandTier::Active,
                item_kind: Some(kind.clone()),
                nearest_drop_dist: candidate.nearest_drop_dist,
                intent: Intent::PickUp {
                    item_id: candidate.item_id.clone(),
                },
            });
            next_id += 1;
        }
    }

    let mut preview_kinds = preview_missing.keys().cloned().collect::<Vec<_>>();
    preview_kinds.sort();
    let mut preview_slots_used = 0u16;
    for kind in preview_kinds {
        if !preview_enabled || preview_slots_used >= preview_quota {
            break;
        }
        let demand = preview_missing.get(&kind).copied().unwrap_or(0);
        if demand == 0 || active_missing.get(&kind).copied().unwrap_or(0) > 0 {
            continue;
        }
        let Some(pool) = stand_pool.get(&kind) else {
            continue;
        };
        for candidate in pool.iter().take(preview_stand_limit()) {
            if preview_slots_used >= preview_quota {
                break;
            }
            tasks.push(Task {
                task_id: next_id,
                sort_key: (3, kind.clone(), candidate.stand),
                target_cell: candidate.stand,
                shareable: false,
                kind: TaskKind::PickupStand,
                demand_tier: DemandTier::Preview,
                item_kind: Some(kind.clone()),
                nearest_drop_dist: candidate.nearest_drop_dist,
                intent: Intent::MoveTo {
                    cell: candidate.stand,
                },
            });
            next_id += 1;
            tasks.push(Task {
                task_id: next_id,
                sort_key: (3, kind.clone(), candidate.stand),
                target_cell: candidate.stand,
                shareable: false,
                kind: TaskKind::ImmediatePickup,
                demand_tier: DemandTier::Preview,
                item_kind: Some(kind.clone()),
                nearest_drop_dist: candidate.nearest_drop_dist,
                intent: Intent::PickUp {
                    item_id: candidate.item_id.clone(),
                },
            });
            next_id += 1;
            preview_slots_used = preview_slots_used.saturating_add(1);
        }
    }

    if active_carrier_exists {
        let Some(drop) = map.dropoff_cells.first().copied() else {
            return Some(TaskBuildResult {
                tasks,
                active_gap_by_kind,
                active_gap_total,
                preview_enabled,
            });
        };
        for bot in &state.bots {
            let role = team.role_for(&bot.id);
            if !matches!(role, BotRole::LeadCourier | BotRole::QueueCourier) {
                continue;
            }
            if let Some(goal) = team.queue_goal_for(&bot.id) {
                tasks.push(Task {
                    task_id: next_id,
                    sort_key: (4, bot.id.clone(), goal),
                    target_cell: goal,
                    shareable: true,
                    kind: TaskKind::QueuePosition,
                    demand_tier: DemandTier::None,
                    item_kind: None,
                    nearest_drop_dist: nearest_drop_dist(goal, map, dist),
                    intent: Intent::MoveTo { cell: goal },
                });
                next_id += 1;
            } else {
                tasks.push(Task {
                    task_id: next_id,
                    sort_key: (4, bot.id.clone(), drop),
                    target_cell: drop,
                    shareable: true,
                    kind: TaskKind::QueuePosition,
                    demand_tier: DemandTier::None,
                    item_kind: None,
                    nearest_drop_dist: 0,
                    intent: Intent::MoveTo { cell: drop },
                });
                next_id += 1;
            }
        }
    }

    tasks.sort_by(|a, b| {
        a.sort_key
            .cmp(&b.sort_key)
            .then_with(|| a.task_id.cmp(&b.task_id))
    });
    Some(TaskBuildResult {
        tasks,
        active_gap_by_kind,
        active_gap_total,
        preview_enabled,
    })
}

fn build_stand_pool(
    state: &GameState,
    map: &MapCache,
    dist: &DistanceMap,
) -> HashMap<String, Vec<StandCandidate>> {
    let mut items = state.items.iter().collect::<Vec<_>>();
    items.sort_by(|a, b| compare_item_ids(&a.id, &b.id).then_with(|| a.kind.cmp(&b.kind)));

    let mut by_kind: HashMap<String, HashMap<u16, StandCandidate>> = HashMap::new();
    for item in items {
        let mut stands = map.stand_cells_for_item(&item.id).to_vec();
        stands.sort_unstable();
        for stand in stands {
            by_kind
                .entry(item.kind.clone())
                .or_default()
                .entry(stand)
                .or_insert_with(|| StandCandidate {
                    stand,
                    item_id: item.id.clone(),
                    nearest_drop_dist: nearest_drop_dist(stand, map, dist),
                });
        }
    }

    let mut out = HashMap::<String, Vec<StandCandidate>>::new();
    for (kind, stands) in by_kind {
        let mut pool = stands.into_values().collect::<Vec<_>>();
        pool.sort_by(|a, b| {
            a.nearest_drop_dist
                .cmp(&b.nearest_drop_dist)
                .then_with(|| a.stand.cmp(&b.stand))
                .then_with(|| compare_item_ids(&a.item_id, &b.item_id))
        });
        out.insert(kind, pool);
    }
    out
}

fn build_effective_active_supply(
    state: &GameState,
    map: &MapCache,
    dist: &DistanceMap,
    team: &TeamContext,
) -> HashMap<String, u16> {
    const ACTIVE_SUPPLY_NEAR_DROP_MAX: u16 = 10;
    const ACTIVE_SUPPLY_COURIER_DROP_MAX: u16 = 16;
    let mut out = HashMap::<String, u16>::new();
    for bot in &state.bots {
        let Some(cell) = map.idx(bot.x, bot.y) else {
            continue;
        };
        let near_drop = map
            .dropoff_cells
            .iter()
            .copied()
            .map(|drop| dist_to(cell, drop, dist))
            .min()
            .unwrap_or(u16::MAX);
        let role = team.role_for(&bot.id);
        let courier_role = matches!(role, BotRole::LeadCourier | BotRole::QueueCourier);
        let serviceable = near_drop <= ACTIVE_SUPPLY_NEAR_DROP_MAX
            || (courier_role && near_drop <= ACTIVE_SUPPLY_COURIER_DROP_MAX);
        if !serviceable {
            continue;
        }
        for item_kind in &bot.carrying {
            if !team.active_order_items_set.contains(item_kind) {
                continue;
            }
            *out.entry(item_kind.clone()).or_insert(0) += 1;
        }
    }
    out
}

fn task_cost(
    bot_cell: u16,
    task: &Task,
    map: &MapCache,
    dist: &DistanceMap,
    state: &GameState,
    lambda_density: f64,
    lambda_choke: f64,
) -> i32 {
    match task.kind {
        TaskKind::ImmediateDropOff => i32::from(dist_to(bot_cell, task.target_cell, dist)),
        TaskKind::CarryToDropoff => {
            let d = f64::from(dist_to(bot_cell, task.target_cell, dist));
            let density = local_density(task.target_cell, state) as f64;
            let choke = choke_penalty(task.target_cell, map) as f64;
            (d + 0.4 * lambda_density * density + lambda_choke * choke).round() as i32
        }
        TaskKind::PickupStand => {
            let d = f64::from(dist_to(bot_cell, task.target_cell, dist));
            let density = local_density(task.target_cell, state) as f64;
            let choke = choke_penalty(task.target_cell, map) as f64;
            let demand_bias = match task.demand_tier {
                DemandTier::Active => -25.0,
                DemandTier::Preview => 30.0,
                DemandTier::None => 0.0,
            };
            (d + 0.45 * f64::from(task.nearest_drop_dist.min(64))
                + lambda_density * density
                + lambda_choke * choke
                + demand_bias)
                .round() as i32
        }
        TaskKind::ImmediatePickup => {
            if bot_cell != task.target_cell {
                return 9_000;
            }
            match task.demand_tier {
                DemandTier::Active => -60,
                DemandTier::Preview => -20,
                DemandTier::None => 0,
            }
        }
        TaskKind::QueuePosition => {
            let d = f64::from(dist_to(bot_cell, task.target_cell, dist));
            let choke = choke_penalty(task.target_cell, map) as f64;
            (d + lambda_choke * choke).round() as i32
        }
    }
}

fn local_density(cell: u16, state: &GameState) -> u16 {
    let x = (cell as i32) % state.grid.width.max(1);
    let y = (cell as i32) / state.grid.width.max(1);
    state
        .bots
        .iter()
        .filter(|bot| (bot.x - x).abs() + (bot.y - y).abs() <= 2)
        .count() as u16
}

fn choke_penalty(cell: u16, map: &MapCache) -> u16 {
    let degree = map.neighbors[cell as usize].len() as u16;
    4u16.saturating_sub(degree.min(4))
}

fn dist_to(a: u16, b: u16, dist: &DistanceMap) -> u16 {
    let d = dist.dist(a, b);
    if d == u16::MAX {
        256
    } else {
        d
    }
}

fn nearest_drop_dist(cell: u16, map: &MapCache, dist: &DistanceMap) -> u16 {
    map.dropoff_cells
        .iter()
        .copied()
        .map(|drop| dist_to(cell, drop, dist))
        .min()
        .unwrap_or(256)
}

fn roundtrip_idle_pressure_penalty(
    roundtrip_eta: f64,
    bot_dropoff_dist: f64,
    stall_ticks: f64,
) -> i32 {
    let eta_pressure = if roundtrip_eta > 18.0 {
        (roundtrip_eta - 18.0) * 0.7
    } else {
        0.0
    };
    let far_pressure = if bot_dropoff_dist > 12.0 {
        (bot_dropoff_dist - 12.0) * 1.2
    } else {
        0.0
    };
    ((eta_pressure + far_pressure) * (1.0 + stall_ticks / 16.0)).round() as i32
}

fn avg_teammate_distance(bot: &BotState, bots: &[BotState]) -> f64 {
    let mut total = 0.0;
    let mut count = 0.0;
    for other in bots {
        if other.id == bot.id {
            continue;
        }
        total += f64::from((other.x - bot.x).abs() + (other.y - bot.y).abs());
        count += 1.0;
    }
    if count == 0.0 {
        0.0
    } else {
        total / count
    }
}

fn preview_prefetch_quota(active_gap_total: usize) -> u16 {
    if active_gap_total == 0 {
        return 8;
    }
    if active_gap_total <= 2 {
        return 2;
    }
    if active_gap_total <= 4 {
        return 1;
    }
    0
}

fn preview_stand_limit() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("GROCERY_PREVIEW_STAND_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3)
            .clamp(1, 12)
    })
}

fn parse_item_numeric_suffix(id: &str) -> Option<u32> {
    id.strip_prefix("item_")?.parse::<u32>().ok()
}

fn compare_item_ids(a: &str, b: &str) -> Ordering {
    let a_num = parse_item_numeric_suffix(a);
    let b_num = parse_item_numeric_suffix(b);
    match (a_num, b_num) {
        (Some(aa), Some(bb)) => aa.cmp(&bb).then_with(|| a.cmp(b)),
        _ => a.cmp(b),
    }
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use crate::{
        dist::DistanceMap,
        model::{BotState, GameState, Grid, Item, Order, OrderStatus},
        team_context::{TeamContext, TeamContextConfig},
        world::World,
    };

    use super::{
        compare_item_ids, roundtrip_idle_pressure_penalty, AssignmentEngine, AssignmentRuntimeHints,
    };

    #[test]
    fn deterministic_assignment_ordering() {
        let state = GameState {
            grid: Grid {
                width: 6,
                height: 4,
                drop_off_tiles: vec![[0, 0]],
                ..Grid::default()
            },
            bots: vec![
                BotState {
                    id: "b".to_owned(),
                    x: 1,
                    y: 1,
                    carrying: vec![],
                    capacity: 3,
                },
                BotState {
                    id: "a".to_owned(),
                    x: 2,
                    y: 1,
                    carrying: vec![],
                    capacity: 3,
                },
            ],
            items: vec![Item {
                id: "item_1".to_owned(),
                kind: "milk".to_owned(),
                x: 3,
                y: 1,
            }],
            orders: vec![Order {
                id: "o1".to_owned(),
                item_id: "milk".to_owned(),
                status: OrderStatus::InProgress,
            }],
            ..GameState::default()
        };
        let world = World::new(&state);
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
        let engine = AssignmentEngine::new();
        let first = engine
            .build_intents(
                &state,
                map,
                &dist,
                &team,
                6,
                1.0,
                1.5,
                1.0,
                AssignmentRuntimeHints::default(),
                Duration::from_millis(200),
            )
            .expect("assignment");
        let second = engine
            .build_intents(
                &state,
                map,
                &dist,
                &team,
                6,
                1.0,
                1.5,
                1.0,
                AssignmentRuntimeHints::default(),
                Duration::from_millis(200),
            )
            .expect("assignment");
        assert_eq!(
            format!("{:?}", first.intents),
            format!("{:?}", second.intents)
        );
    }

    #[test]
    fn item_id_numeric_sorting_is_not_lexicographic() {
        let mut ids = vec!["item_102", "item_12", "item_2"];
        ids.sort_by(|a, b| compare_item_ids(a, b));
        assert_eq!(ids, vec!["item_2", "item_12", "item_102"]);
    }

    #[test]
    fn active_pickup_beats_preview_pickup_when_geometry_equal() {
        let state = GameState {
            grid: Grid {
                width: 7,
                height: 5,
                drop_off_tiles: vec![[0, 0]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 4,
                y: 2,
                carrying: vec![],
                capacity: 3,
            }],
            items: vec![
                Item {
                    id: "item_12".to_owned(),
                    kind: "milk".to_owned(),
                    x: 5,
                    y: 2,
                },
                Item {
                    id: "item_2".to_owned(),
                    kind: "bread".to_owned(),
                    x: 3,
                    y: 2,
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
        let world = World::new(&state);
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
        let engine = AssignmentEngine::new();
        let result = engine
            .build_intents(
                &state,
                map,
                &dist,
                &team,
                8,
                1.0,
                1.5,
                1.0,
                AssignmentRuntimeHints::default(),
                Duration::from_millis(200),
            )
            .expect("assignment");
        assert!(result.active_task_count > 0);
        assert!(result.preview_task_count <= result.task_count);
        assert!(result.active_gap_total > 0);
        let first = result
            .intents
            .into_iter()
            .find(|intent| intent.bot_id == "0")
            .expect("bot intent");
        assert!(matches!(
            first.intent,
            crate::dispatcher::Intent::PickUp { .. }
        ));
        if let crate::dispatcher::Intent::PickUp { item_id } = first.intent {
            assert_eq!(item_id, "item_12");
        }
    }

    #[test]
    fn far_carrier_does_not_clear_active_gap() {
        let state = GameState {
            grid: Grid {
                width: 20,
                height: 10,
                drop_off_tiles: vec![[0, 9]],
                ..Grid::default()
            },
            bots: vec![
                BotState {
                    id: "0".to_owned(),
                    x: 19,
                    y: 0,
                    carrying: vec!["milk".to_owned()],
                    capacity: 3,
                },
                BotState {
                    id: "1".to_owned(),
                    x: 18,
                    y: 0,
                    carrying: vec![],
                    capacity: 3,
                },
            ],
            items: vec![Item {
                id: "item_0".to_owned(),
                kind: "milk".to_owned(),
                x: 10,
                y: 4,
            }],
            orders: vec![Order {
                id: "o_active".to_owned(),
                item_id: "milk".to_owned(),
                status: OrderStatus::InProgress,
            }],
            ..GameState::default()
        };
        let world = World::new(&state);
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
        let engine = AssignmentEngine::new();
        let result = engine
            .build_intents(
                &state,
                map,
                &dist,
                &team,
                8,
                1.0,
                1.5,
                1.0,
                AssignmentRuntimeHints::default(),
                Duration::from_millis(200),
            )
            .expect("assignment");
        assert!(result.active_task_count > 0);
        assert!(result.active_gap_total > 0);
        assert!(result.preview_task_count <= result.task_count);
    }

    #[test]
    fn roundtrip_idle_pressure_penalty_increases_for_far_and_slow_paths() {
        let low = roundtrip_idle_pressure_penalty(12.0, 6.0, 4.0);
        let high = roundtrip_idle_pressure_penalty(28.0, 20.0, 24.0);
        assert!(high > low);
    }

    #[test]
    fn local_first_filters_far_pickups_when_local_exists() {
        let state = GameState {
            grid: Grid {
                width: 14,
                height: 8,
                drop_off_tiles: vec![[0, 7]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 2,
                y: 6,
                carrying: vec![],
                capacity: 3,
            }],
            items: vec![
                Item {
                    id: "item_local".to_owned(),
                    kind: "milk".to_owned(),
                    x: 3,
                    y: 3,
                },
                Item {
                    id: "item_far".to_owned(),
                    kind: "bread".to_owned(),
                    x: 12,
                    y: 1,
                },
            ],
            orders: vec![
                Order {
                    id: "o_milk".to_owned(),
                    item_id: "milk".to_owned(),
                    status: OrderStatus::InProgress,
                },
                Order {
                    id: "o_bread".to_owned(),
                    item_id: "bread".to_owned(),
                    status: OrderStatus::InProgress,
                },
            ],
            ..GameState::default()
        };
        let world = World::new(&state);
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
        let local_stand = map.stand_cells_for_item("item_local")[0];
        let local_area = team
            .knowledge
            .area_id_by_cell
            .get(local_stand as usize)
            .copied()
            .unwrap_or(u16::MAX);
        let mut hints = AssignmentRuntimeHints::default();
        hints
            .preferred_area_by_bot
            .insert("0".to_owned(), local_area);
        hints.expansion_mode_by_bot.insert("0".to_owned(), false);
        hints
            .local_active_candidate_count_by_bot
            .insert("0".to_owned(), 1);
        hints.local_radius_by_bot.insert("0".to_owned(), 5);
        hints.out_of_area_penalty = 28.0;
        hints.out_of_radius_penalty = 45.0;

        let engine = AssignmentEngine::new();
        let result = engine
            .build_intents(
                &state,
                map,
                &dist,
                &team,
                8,
                1.0,
                1.5,
                1.0,
                hints,
                Duration::from_millis(200),
            )
            .expect("assignment");
        let intent = result
            .intents
            .into_iter()
            .find(|entry| entry.bot_id == "0")
            .expect("bot intent");
        match intent.intent {
            crate::dispatcher::Intent::PickUp { item_id } => assert_eq!(item_id, "item_local"),
            crate::dispatcher::Intent::MoveTo { cell } => {
                assert_eq!(cell, local_stand);
            }
            _ => panic!("expected local pickup/move intent"),
        }
    }

    #[test]
    fn expansion_mode_allows_far_pickup_after_stall() {
        let state = GameState {
            grid: Grid {
                width: 14,
                height: 8,
                drop_off_tiles: vec![[0, 7]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 1,
                y: 6,
                carrying: vec![],
                capacity: 3,
            }],
            items: vec![Item {
                id: "item_far".to_owned(),
                kind: "milk".to_owned(),
                x: 12,
                y: 1,
            }],
            orders: vec![Order {
                id: "o_milk".to_owned(),
                item_id: "milk".to_owned(),
                status: OrderStatus::InProgress,
            }],
            ..GameState::default()
        };
        let world = World::new(&state);
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
        let mut hints = AssignmentRuntimeHints::default();
        hints.preferred_area_by_bot.insert("0".to_owned(), u16::MAX);
        hints.expansion_mode_by_bot.insert("0".to_owned(), true);
        hints
            .local_active_candidate_count_by_bot
            .insert("0".to_owned(), 0);
        hints.local_radius_by_bot.insert("0".to_owned(), 2);
        hints.out_of_area_penalty = 28.0;
        hints.out_of_radius_penalty = 45.0;

        let engine = AssignmentEngine::new();
        let result = engine
            .build_intents(
                &state,
                map,
                &dist,
                &team,
                8,
                1.0,
                1.5,
                1.0,
                hints,
                Duration::from_millis(200),
            )
            .expect("assignment");
        let intent = result
            .intents
            .into_iter()
            .find(|entry| entry.bot_id == "0")
            .expect("bot intent");
        assert!(
            matches!(
                intent.intent,
                crate::dispatcher::Intent::PickUp { .. } | crate::dispatcher::Intent::MoveTo { .. }
            ),
            "expansion mode should still produce an active pickup path"
        );
    }
}
