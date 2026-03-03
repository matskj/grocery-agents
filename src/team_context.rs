use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    dist::DistanceMap,
    model::{GameState, OrderStatus},
    scoring::detect_mode_label,
    world::MapCache,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DemandTier {
    None,
    Active,
    Preview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BotRole {
    LeadCourier,
    QueueCourier,
    Collector,
    Yield,
    Idle,
}

impl BotRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LeadCourier => "lead_courier",
            Self::QueueCourier => "queue_courier",
            Self::Collector => "collector",
            Self::Yield => "yield",
            Self::Idle => "idle",
        }
    }

    pub fn priority(self) -> u8 {
        match self {
            Self::LeadCourier => 0,
            Self::QueueCourier => 1,
            Self::Collector => 2,
            Self::Yield => 3,
            Self::Idle => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockedMove {
    pub from: u16,
    pub dx: i32,
    pub dy: i32,
    pub count: u8,
}

#[derive(Debug, Clone)]
pub struct StickyQueueRole {
    pub role: BotRole,
    pub dropoff: u16,
    pub goal_cell: u16,
    pub slot_index: Option<usize>,
    pub remaining: u8,
}

#[derive(Debug, Clone)]
pub struct QueueAssignment {
    pub role: BotRole,
    pub goal_cell: Option<u16>,
    pub slot_index: Option<usize>,
    pub dropoff: Option<u16>,
    pub queue_distance: u16,
}

#[derive(Debug, Clone, Default)]
pub struct QueuePlan {
    pub strict_mode: bool,
    pub max_ring_entrants: usize,
    pub lane_cells: Vec<u16>,
    pub ring_cells: Vec<u16>,
    pub assignments: HashMap<String, QueueAssignment>,
    pub queue_order: Vec<String>,
    pub queue_eta_rank_by_bot: HashMap<String, u16>,
    pub violations: HashSet<String>,
    pub near_dropoff_blocking: HashSet<String>,
    pub next_sticky: HashMap<String, StickyQueueRole>,
}

#[derive(Debug, Clone, Default)]
pub struct BotSnapshot {
    pub cell: u16,
    pub blocked_ticks: u8,
    pub repeated_failed_moves: u8,
    pub carrying_active: bool,
    pub constraint_relax_ticks: u8,
    pub escape_macro_ticks: u8,
}

#[derive(Debug, Clone, Default)]
pub struct OrderSnapshot {
    pub active_remaining_by_item: HashMap<String, u64>,
    pub preview_remaining_by_item: HashMap<String, u64>,
    pub active_order_items_set: HashSet<String>,
    pub pending_order_items_set: HashSet<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TrafficSnapshot {
    pub conflict_degree_by_bot: HashMap<String, u16>,
    pub local_conflict_count_by_bot: HashMap<String, u16>,
    pub local_conflict_candidates: HashMap<String, Vec<String>>,
    pub conflict_hotspots: Vec<(u16, u16)>,
    pub lane_congestion: u16,
    pub blocked_bot_count: u16,
    pub stuck_bot_count: u16,
}

#[derive(Debug, Clone, Default)]
pub struct MovementReservation {
    pub priorities: HashMap<String, u8>,
    pub reserve_horizon: HashMap<String, u8>,
    pub prohibited_moves: HashMap<String, Vec<BlockedMove>>,
    pub forbidden_cells: HashMap<String, HashSet<u16>>,
    pub queue_relaxation_active: HashMap<String, bool>,
    pub role_by_bot: HashMap<String, BotRole>,
    pub dropoff_control_zone: HashSet<u16>,
}

#[derive(Debug, Clone)]
pub struct StandClaim {
    pub bot_id: String,
    pub expires_tick: u64,
    pub demand_tier: DemandTier,
}

#[derive(Debug, Clone)]
pub struct BotCommitment {
    pub goal_cell: u16,
    pub item_kind: Option<String>,
    pub created_tick: u64,
    pub last_progress_tick: u64,
}

#[derive(Debug, Clone, Default)]
pub struct TeamKnowledge {
    pub kind_to_stands: HashMap<String, Vec<u16>>,
    pub stand_to_kind: HashMap<u16, String>,
    pub area_id_by_cell: Vec<u16>,
    pub area_load_by_id: HashMap<u16, u16>,
    pub active_gap_by_kind: HashMap<String, u16>,
    pub effective_supply_by_kind: HashMap<String, u16>,
    pub stand_claims: HashMap<u16, StandClaim>,
    pub bot_commitments: HashMap<String, BotCommitment>,
}

#[derive(Debug, Clone, Copy)]
pub struct TeamContextConfig {
    pub strict_queue: bool,
    pub max_ring_entrants: usize,
    pub hold_ticks: u8,
    pub control_zone_radius: u8,
}

impl Default for TeamContextConfig {
    fn default() -> Self {
        Self {
            strict_queue: true,
            max_ring_entrants: 1,
            hold_ticks: 3,
            control_zone_radius: 4,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct BotPlanState {
    pub goal_cell: Option<u16>,
    pub path_preview: Vec<u16>,
    pub escape_macro_active: bool,
    pub escape_macro_ticks_remaining: u8,
    pub constraint_relax_ticks_remaining: u8,
}

#[derive(Debug, Clone, Default)]
pub struct TeamContext {
    pub mode: String,
    pub active_items: HashSet<String>,
    pub active_order_items_set: HashSet<String>,
    pub pending_order_items_set: HashSet<String>,
    pub cell_degree_by_idx: Vec<u8>,
    pub dead_end_depth_by_idx: Vec<u8>,
    pub junction_cells: HashSet<u16>,
    pub corner_cells: HashSet<u16>,
    pub dropoff_control_zone: HashSet<u16>,
    pub bot_snapshot: HashMap<String, BotSnapshot>,
    pub dropoff_serviceable_by_bot: HashMap<String, bool>,
    pub recent_cell_visits_team: HashMap<u16, u16>,
    pub dropoff_attempt_streak_by_bot: HashMap<String, u8>,
    pub loop_two_cycle_count_by_bot: HashMap<String, u16>,
    pub coverage_gain_by_bot: HashMap<String, u16>,
    pub dropoff_watchdog_triggered_by_bot: HashMap<String, bool>,
    pub bot_plan_state_by_bot: HashMap<String, BotPlanState>,
    pub order_snapshot: OrderSnapshot,
    pub traffic: TrafficSnapshot,
    pub queue: QueuePlan,
    pub movement: MovementReservation,
    pub knowledge: TeamKnowledge,
}

impl TeamContext {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        state: &GameState,
        map: &MapCache,
        dist: &DistanceMap,
        blocked_ticks: &HashMap<String, u8>,
        repeated_failed: &HashMap<String, u8>,
        constraint_relax_ticks: &HashMap<String, u8>,
        escape_macro_ticks: &HashMap<String, u8>,
        recent_cell_visits_team: &HashMap<u16, u16>,
        dropoff_attempt_streak_by_bot: &HashMap<String, u8>,
        loop_two_cycle_count_by_bot: &HashMap<String, u16>,
        coverage_gain_by_bot: &HashMap<String, u16>,
        dropoff_watchdog_triggered_by_bot: &HashMap<String, bool>,
        prohibited_moves: &HashMap<String, Vec<BlockedMove>>,
        sticky: &HashMap<String, StickyQueueRole>,
        cfg: TeamContextConfig,
    ) -> Self {
        let (cell_degree_by_idx, dead_end_depth_by_idx, junction_cells, corner_cells) =
            build_topology(map);
        let active_items = active_item_set(state);
        let order_snapshot = order_snapshot(state);
        let bot_snapshot = build_bot_snapshot(
            state,
            map,
            blocked_ticks,
            repeated_failed,
            constraint_relax_ticks,
            escape_macro_ticks,
            &active_items,
        );
        let queue = build_queue_plan(state, map, dist, &bot_snapshot, sticky, cfg);
        let dropoff_control_zone =
            build_dropoff_control_zone(map, &queue, cfg.control_zone_radius);
        let traffic = build_traffic(state, map, &queue, &bot_snapshot);
        let movement = build_movement(
            state,
            map,
            dist,
            &bot_snapshot,
            &queue,
            &dropoff_control_zone,
            prohibited_moves,
            cfg,
        );
        let knowledge = build_team_knowledge(state, map, dist, &order_snapshot, &bot_snapshot);
        let bot_plan_state_by_bot = state
            .bots
            .iter()
            .map(|bot| {
                let escape_ticks = bot_snapshot
                    .get(&bot.id)
                    .map(|s| s.escape_macro_ticks)
                    .unwrap_or(0);
                let relax_ticks = bot_snapshot
                    .get(&bot.id)
                    .map(|s| s.constraint_relax_ticks)
                    .unwrap_or(0);
                (
                    bot.id.clone(),
                    BotPlanState {
                        goal_cell: None,
                        path_preview: Vec::new(),
                        escape_macro_active: escape_ticks > 0,
                        escape_macro_ticks_remaining: escape_ticks,
                        constraint_relax_ticks_remaining: relax_ticks,
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        let dropoff_serviceable_by_bot =
            build_dropoff_serviceable_by_bot(state, &order_snapshot.active_order_items_set);
        Self {
            mode: detect_mode_label(state).to_owned(),
            active_items,
            active_order_items_set: order_snapshot.active_order_items_set.clone(),
            pending_order_items_set: order_snapshot.pending_order_items_set.clone(),
            cell_degree_by_idx,
            dead_end_depth_by_idx,
            junction_cells,
            corner_cells,
            dropoff_control_zone,
            bot_snapshot,
            dropoff_serviceable_by_bot,
            recent_cell_visits_team: recent_cell_visits_team.clone(),
            dropoff_attempt_streak_by_bot: dropoff_attempt_streak_by_bot.clone(),
            loop_two_cycle_count_by_bot: loop_two_cycle_count_by_bot.clone(),
            coverage_gain_by_bot: coverage_gain_by_bot.clone(),
            dropoff_watchdog_triggered_by_bot: dropoff_watchdog_triggered_by_bot.clone(),
            bot_plan_state_by_bot,
            order_snapshot,
            traffic,
            queue,
            movement,
            knowledge,
        }
    }

    pub fn with_claims(
        mut self,
        stand_claims: &HashMap<u16, StandClaim>,
        bot_commitments: &HashMap<String, BotCommitment>,
        now_tick: u64,
    ) -> Self {
        self.knowledge.stand_claims = stand_claims
            .iter()
            .filter(|(_, claim)| claim.expires_tick >= now_tick)
            .map(|(cell, claim)| (*cell, claim.clone()))
            .collect();
        self.knowledge.bot_commitments = bot_commitments
            .iter()
            .map(|(bot_id, commitment)| (bot_id.clone(), commitment.clone()))
            .collect();
        self
    }

    pub fn role_for(&self, bot_id: &str) -> BotRole {
        self.queue
            .assignments
            .get(bot_id)
            .map(|a| a.role)
            .unwrap_or(BotRole::Idle)
    }

    pub fn queue_goal_for(&self, bot_id: &str) -> Option<u16> {
        self.queue.assignments.get(bot_id).and_then(|a| a.goal_cell)
    }

    pub fn telemetry(&self, map: &MapCache) -> serde_json::Value {
        let mut queue_roles = serde_json::Map::new();
        let mut queue_slot_index = serde_json::Map::new();
        let mut queue_distance = serde_json::Map::new();
        let mut queue_violation = serde_json::Map::new();
        let mut near_dropoff_blocking = serde_json::Map::new();
        let mut repeated_failed_move_count = serde_json::Map::new();
        let mut yield_applied = serde_json::Map::new();
        let mut queue_relaxation_active = serde_json::Map::new();
        let mut queue_eta_rank = serde_json::Map::new();
        let mut in_corner = serde_json::Map::new();
        let mut dead_end_depth = serde_json::Map::new();
        let mut escape_macro_active = serde_json::Map::new();
        let mut escape_macro_ticks_remaining = serde_json::Map::new();
        let mut goal_cell_by_bot = serde_json::Map::new();
        let mut path_preview_by_bot = serde_json::Map::new();
        let mut serviceable_dropoff_by_bot = serde_json::Map::new();
        let mut dropoff_attempt_streak_by_bot = serde_json::Map::new();
        let mut loop_two_cycle_count_by_bot = serde_json::Map::new();
        let mut coverage_gain_by_bot = serde_json::Map::new();
        let mut dropoff_watchdog_triggered_by_bot = serde_json::Map::new();

        for (bot_id, assignment) in &self.queue.assignments {
            queue_roles.insert(
                bot_id.clone(),
                serde_json::Value::String(assignment.role.as_str().to_owned()),
            );
            queue_slot_index.insert(
                bot_id.clone(),
                serde_json::Value::Number(serde_json::Number::from(
                    assignment.slot_index.map(|v| v as i64).unwrap_or(-1),
                )),
            );
            queue_distance.insert(
                bot_id.clone(),
                serde_json::Value::Number(serde_json::Number::from(assignment.queue_distance as i64)),
            );
            queue_violation.insert(
                bot_id.clone(),
                serde_json::Value::Bool(self.queue.violations.contains(bot_id)),
            );
            near_dropoff_blocking.insert(
                bot_id.clone(),
                serde_json::Value::Bool(self.queue.near_dropoff_blocking.contains(bot_id)),
            );
            yield_applied.insert(
                bot_id.clone(),
                serde_json::Value::Bool(matches!(assignment.role, BotRole::Yield)),
            );
            queue_relaxation_active.insert(
                bot_id.clone(),
                serde_json::Value::Bool(
                    self.movement
                        .queue_relaxation_active
                        .get(bot_id)
                        .copied()
                        .unwrap_or(false),
                ),
            );
            queue_eta_rank.insert(
                bot_id.clone(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.queue
                        .queue_eta_rank_by_bot
                        .get(bot_id)
                        .copied()
                        .unwrap_or(9_999) as i64,
                )),
            );
            serviceable_dropoff_by_bot.insert(
                bot_id.clone(),
                serde_json::Value::Bool(
                    self.dropoff_serviceable_by_bot
                        .get(bot_id)
                        .copied()
                        .unwrap_or(false),
                ),
            );
            dropoff_attempt_streak_by_bot.insert(
                bot_id.clone(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.dropoff_attempt_streak_by_bot
                        .get(bot_id)
                        .copied()
                        .unwrap_or(0) as i64,
                )),
            );
            loop_two_cycle_count_by_bot.insert(
                bot_id.clone(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.loop_two_cycle_count_by_bot
                        .get(bot_id)
                        .copied()
                        .unwrap_or(0) as i64,
                )),
            );
            coverage_gain_by_bot.insert(
                bot_id.clone(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.coverage_gain_by_bot
                        .get(bot_id)
                        .copied()
                        .unwrap_or(0) as i64,
                )),
            );
            dropoff_watchdog_triggered_by_bot.insert(
                bot_id.clone(),
                serde_json::Value::Bool(
                    self.dropoff_watchdog_triggered_by_bot
                        .get(bot_id)
                        .copied()
                        .unwrap_or(false),
                ),
            );
        }

        for (bot_id, snap) in &self.bot_snapshot {
            repeated_failed_move_count.insert(
                bot_id.clone(),
                serde_json::Value::Number(serde_json::Number::from(snap.repeated_failed_moves as i64)),
            );
            in_corner.insert(
                bot_id.clone(),
                serde_json::Value::Bool(self.corner_cells.contains(&snap.cell)),
            );
            dead_end_depth.insert(
                bot_id.clone(),
                serde_json::Value::Number(serde_json::Number::from(
                    self.dead_end_depth_by_idx
                        .get(snap.cell as usize)
                        .copied()
                        .unwrap_or(0) as i64,
                )),
            );
            let plan_state = self.bot_plan_state_by_bot.get(bot_id);
            escape_macro_active.insert(
                bot_id.clone(),
                serde_json::Value::Bool(
                    plan_state
                        .map(|s| s.escape_macro_active)
                        .unwrap_or(false),
                ),
            );
            escape_macro_ticks_remaining.insert(
                bot_id.clone(),
                serde_json::Value::Number(serde_json::Number::from(
                    plan_state
                        .map(|s| s.escape_macro_ticks_remaining)
                        .unwrap_or(0) as i64,
                )),
            );
            goal_cell_by_bot.insert(
                bot_id.clone(),
                match plan_state.and_then(|s| s.goal_cell) {
                    Some(goal) => {
                        let (x, y) = map.xy(goal);
                        serde_json::json!([x, y])
                    }
                    None => serde_json::Value::Null,
                },
            );
            path_preview_by_bot.insert(
                bot_id.clone(),
                serde_json::Value::Array(
                    plan_state
                        .map(|s| {
                            s.path_preview
                                .iter()
                                .map(|idx| {
                                    let (x, y) = map.xy(*idx);
                                    serde_json::json!([x, y])
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default(),
                ),
            );
        }

        let conflict_degree_by_bot = self
            .traffic
            .conflict_degree_by_bot
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    serde_json::Value::Number(serde_json::Number::from(*v as i64)),
                )
            })
            .collect::<serde_json::Map<_, _>>();

        let lane_xy = self
            .queue
            .lane_cells
            .iter()
            .map(|&idx| {
                let (x, y) = map.xy(idx);
                serde_json::json!([x, y])
            })
            .collect::<Vec<_>>();
        let ring_xy = self
            .queue
            .ring_cells
            .iter()
            .map(|&idx| {
                let (x, y) = map.xy(idx);
                serde_json::json!([x, y])
            })
            .collect::<Vec<_>>();
        let conflict_hotspots = self
            .traffic
            .conflict_hotspots
            .iter()
            .map(|(idx, count)| {
                let (x, y) = map.xy(*idx);
                serde_json::json!({
                    "x": x,
                    "y": y,
                    "count": count,
                })
            })
            .collect::<Vec<_>>();
        let control_zone_xy = self
            .dropoff_control_zone
            .iter()
            .copied()
            .map(|idx| {
                let (x, y) = map.xy(idx);
                serde_json::json!([x, y])
            })
            .collect::<Vec<_>>();
        let recent_cell_visit_heat = self
            .recent_cell_visits_team
            .iter()
            .map(|(idx, count)| {
                let (x, y) = map.xy(*idx);
                serde_json::json!({
                    "x": x,
                    "y": y,
                    "count": count,
                })
            })
            .collect::<Vec<_>>();
        let local_conflict_count_by_bot = self
            .traffic
            .local_conflict_count_by_bot
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    serde_json::Value::Number(serde_json::Number::from(*v as i64)),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let local_conflict_candidates = self
            .traffic
            .local_conflict_candidates
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    serde_json::Value::Array(
                        v.iter()
                            .map(|id| serde_json::Value::String(id.clone()))
                            .collect::<Vec<_>>(),
                    ),
                )
            })
            .collect::<serde_json::Map<_, _>>();

        let mut failed_move_attempts = Vec::new();
        for (bot_id, moves) in &self.movement.prohibited_moves {
            for blocked in moves {
                let (x, y) = map.xy(blocked.from);
                failed_move_attempts.push(serde_json::json!({
                    "bot_id": bot_id,
                    "from": [x, y],
                    "dx": blocked.dx,
                    "dy": blocked.dy,
                    "count": blocked.count,
                }));
            }
        }

        serde_json::json!({
            "mode": &self.mode,
            "active_order_items_set": &self.active_order_items_set,
            "pending_order_items_set": &self.pending_order_items_set,
            "active_order_remaining_by_item": &self.order_snapshot.active_remaining_by_item,
            "preview_order_remaining_by_item": &self.order_snapshot.preview_remaining_by_item,
            "queue_roles": queue_roles,
            "queue_slot_index_by_bot": queue_slot_index,
            "queue_distance_by_bot": queue_distance,
            "is_queue_violation_by_bot": queue_violation,
            "near_dropoff_blocking_by_bot": near_dropoff_blocking,
            "repeated_failed_move_count_by_bot": repeated_failed_move_count,
            "conflict_degree_by_bot": conflict_degree_by_bot,
            "yield_applied_by_bot": yield_applied,
            "queue_relaxation_active_by_bot": queue_relaxation_active,
            "queue_eta_rank_by_bot": queue_eta_rank,
            "serviceable_dropoff_by_bot": serviceable_dropoff_by_bot,
            "dropoff_attempt_same_order_streak_by_bot": dropoff_attempt_streak_by_bot,
            "loop_two_cycle_count_by_bot": loop_two_cycle_count_by_bot,
            "coverage_gain_by_bot": coverage_gain_by_bot,
            "dropoff_watchdog_triggered_by_bot": dropoff_watchdog_triggered_by_bot,
            "queue_lane_cells": lane_xy,
            "dropoff_ring_cells": ring_xy,
            "dropoff_control_zone_cells": control_zone_xy,
            "recent_cell_visit_heat": recent_cell_visit_heat,
            "conflict_hotspots": conflict_hotspots,
            "local_conflict_count_by_bot": local_conflict_count_by_bot,
            "local_conflict_candidates_by_bot": local_conflict_candidates,
            "failed_move_attempts": failed_move_attempts,
            "in_corner_by_bot": in_corner,
            "dead_end_depth_by_bot": dead_end_depth,
            "escape_macro_active_by_bot": escape_macro_active,
            "escape_macro_ticks_remaining_by_bot": escape_macro_ticks_remaining,
            "goal_cell_by_bot": goal_cell_by_bot,
            "path_preview_by_bot": path_preview_by_bot,
            "queue_violation_count": self.queue.violations.len(),
            "lane_congestion": self.traffic.lane_congestion,
            "blocked_bot_count": self.traffic.blocked_bot_count,
            "stuck_bot_count": self.traffic.stuck_bot_count,
            "knowledge_active_gap_by_kind": self.knowledge.active_gap_by_kind,
            "knowledge_effective_supply_by_kind": self.knowledge.effective_supply_by_kind,
            "knowledge_stand_claim_count": self.knowledge.stand_claims.len(),
            "knowledge_commitment_count": self.knowledge.bot_commitments.len(),
        })
    }
}

fn active_item_set(state: &GameState) -> HashSet<String> {
    state
        .orders
        .iter()
        .filter(|order| matches!(order.status, OrderStatus::InProgress))
        .map(|order| order.item_id.clone())
        .collect()
}

fn order_snapshot(state: &GameState) -> OrderSnapshot {
    let mut active_remaining_by_item: HashMap<String, u64> = HashMap::new();
    let mut preview_remaining_by_item: HashMap<String, u64> = HashMap::new();
    let mut active_order_items_set: HashSet<String> = HashSet::new();
    let mut pending_order_items_set: HashSet<String> = HashSet::new();
    for order in &state.orders {
        match order.status {
            OrderStatus::InProgress => {
                *active_remaining_by_item
                    .entry(order.item_id.clone())
                    .or_insert(0) += 1;
                active_order_items_set.insert(order.item_id.clone());
            }
            OrderStatus::Pending => {
                *preview_remaining_by_item
                    .entry(order.item_id.clone())
                    .or_insert(0) += 1;
                pending_order_items_set.insert(order.item_id.clone());
            }
            OrderStatus::Delivered | OrderStatus::Cancelled => {}
        }
    }
    OrderSnapshot {
        active_remaining_by_item,
        preview_remaining_by_item,
        active_order_items_set,
        pending_order_items_set,
    }
}

fn build_team_knowledge(
    state: &GameState,
    map: &MapCache,
    dist: &DistanceMap,
    order_snapshot: &OrderSnapshot,
    bot_snapshot: &HashMap<String, BotSnapshot>,
) -> TeamKnowledge {
    let mut items = state.items.iter().collect::<Vec<_>>();
    items.sort_by(|a, b| a.id.cmp(&b.id));

    let mut kind_to_stands_set = HashMap::<String, HashSet<u16>>::new();
    let mut stand_to_kind = HashMap::<u16, String>::new();
    for item in items {
        let mut stands = map.stand_cells_for_item(&item.id).to_vec();
        stands.sort_unstable();
        for stand in stands {
            kind_to_stands_set
                .entry(item.kind.clone())
                .or_default()
                .insert(stand);
            match stand_to_kind.get(&stand) {
                Some(existing) if existing <= &item.kind => {}
                _ => {
                    stand_to_kind.insert(stand, item.kind.clone());
                }
            }
        }
    }
    let mut kind_to_stands = HashMap::<String, Vec<u16>>::new();
    for (kind, set) in kind_to_stands_set {
        let mut stands = set.into_iter().collect::<Vec<_>>();
        stands.sort_unstable();
        kind_to_stands.insert(kind, stands);
    }

    let area_id_by_cell = build_area_id_by_cell(map);
    let mut area_load_by_id = HashMap::<u16, u16>::new();
    for bot in &state.bots {
        let Some(cell) = map.idx(bot.x, bot.y) else {
            continue;
        };
        let area = area_id_by_cell
            .get(cell as usize)
            .copied()
            .unwrap_or(u16::MAX);
        *area_load_by_id.entry(area).or_insert(0) += 1;
    }

    let effective_supply_by_kind = build_effective_supply_by_kind(
        state,
        map,
        dist,
        bot_snapshot,
        &order_snapshot.active_order_items_set,
    );
    let mut active_gap_by_kind = HashMap::<String, u16>::new();
    for (kind, missing) in &order_snapshot.active_remaining_by_item {
        let covered = effective_supply_by_kind.get(kind).copied().unwrap_or(0);
        active_gap_by_kind.insert(kind.clone(), (*missing as u16).saturating_sub(covered));
    }

    TeamKnowledge {
        kind_to_stands,
        stand_to_kind,
        area_id_by_cell,
        area_load_by_id,
        active_gap_by_kind,
        effective_supply_by_kind,
        stand_claims: HashMap::new(),
        bot_commitments: HashMap::new(),
    }
}

fn build_area_id_by_cell(map: &MapCache) -> Vec<u16> {
    let mut out = vec![u16::MAX; map.neighbors.len()];
    let mut next_area = 0u16;
    for start in 0..map.neighbors.len() {
        if map.wall_mask[start] || out[start] != u16::MAX {
            continue;
        }
        let mut queue = VecDeque::new();
        out[start] = next_area;
        queue.push_back(start as u16);
        while let Some(cell) = queue.pop_front() {
            for &next in &map.neighbors[cell as usize] {
                if out[next as usize] != u16::MAX {
                    continue;
                }
                out[next as usize] = next_area;
                queue.push_back(next);
            }
        }
        next_area = next_area.saturating_add(1);
    }
    out
}

fn build_effective_supply_by_kind(
    state: &GameState,
    map: &MapCache,
    dist: &DistanceMap,
    bot_snapshot: &HashMap<String, BotSnapshot>,
    active_items: &HashSet<String>,
) -> HashMap<String, u16> {
    const ACTIVE_SUPPLY_NEAR_DROP_MAX: u16 = 10;
    const ACTIVE_SUPPLY_BLOCKED_NEAR_DROP_MAX: u16 = 16;
    let mut out = HashMap::<String, u16>::new();
    for bot in &state.bots {
        if bot.carrying.is_empty() {
            continue;
        }
        let Some(cell) = map.idx(bot.x, bot.y) else {
            continue;
        };
        let near_drop = map
            .dropoff_cells
            .iter()
            .copied()
            .map(|drop| dist.dist(cell, drop))
            .min()
            .unwrap_or(u16::MAX);
        let blocked = bot_snapshot
            .get(&bot.id)
            .map(|snap| snap.blocked_ticks)
            .unwrap_or(0);
        let serviceable = near_drop <= ACTIVE_SUPPLY_NEAR_DROP_MAX
            || (blocked <= 1 && near_drop <= ACTIVE_SUPPLY_BLOCKED_NEAR_DROP_MAX);
        if !serviceable {
            continue;
        }
        for item_kind in &bot.carrying {
            if !active_items.contains(item_kind) {
                continue;
            }
            *out.entry(item_kind.clone()).or_insert(0) += 1;
        }
    }
    out
}

fn build_dropoff_serviceable_by_bot(
    state: &GameState,
    active_order_items_set: &HashSet<String>,
) -> HashMap<String, bool> {
    let mut out = HashMap::new();
    for bot in &state.bots {
        let serviceable = bot
            .carrying
            .iter()
            .any(|item| active_order_items_set.contains(item));
        out.insert(bot.id.clone(), serviceable);
    }
    out
}

fn build_bot_snapshot(
    state: &GameState,
    map: &MapCache,
    blocked_ticks: &HashMap<String, u8>,
    repeated_failed: &HashMap<String, u8>,
    constraint_relax_ticks: &HashMap<String, u8>,
    escape_macro_ticks: &HashMap<String, u8>,
    active_items: &HashSet<String>,
) -> HashMap<String, BotSnapshot> {
    let mut out = HashMap::with_capacity(state.bots.len());
    for bot in &state.bots {
        let cell = map.idx(bot.x, bot.y).unwrap_or(0);
        let carries_active = bot
            .carrying
            .iter()
            .any(|item| active_items.contains(item.as_str()));
        out.insert(
            bot.id.clone(),
            BotSnapshot {
                cell,
                blocked_ticks: blocked_ticks.get(&bot.id).copied().unwrap_or(0),
                repeated_failed_moves: repeated_failed.get(&bot.id).copied().unwrap_or(0),
                carrying_active: carries_active,
                constraint_relax_ticks: constraint_relax_ticks.get(&bot.id).copied().unwrap_or(0),
                escape_macro_ticks: escape_macro_ticks.get(&bot.id).copied().unwrap_or(0),
            },
        );
    }
    out
}

fn build_queue_plan(
    state: &GameState,
    map: &MapCache,
    dist: &DistanceMap,
    bot_snapshot: &HashMap<String, BotSnapshot>,
    sticky: &HashMap<String, StickyQueueRole>,
    cfg: TeamContextConfig,
) -> QueuePlan {
    let mut plan = QueuePlan {
        strict_mode: cfg.strict_queue,
        max_ring_entrants: cfg.max_ring_entrants.max(1),
        ..QueuePlan::default()
    };
    if map.dropoff_cells.is_empty() || state.bots.is_empty() {
        return plan;
    }

    let mut couriers = Vec::<(String, u16, u16, u8, u8, bool)>::new();
    for bot in &state.bots {
        let Some(snap) = bot_snapshot.get(&bot.id) else {
            continue;
        };
        if !snap.carrying_active {
            continue;
        }
        let Some(&dropoff) = map
            .dropoff_cells
            .iter()
            .min_by_key(|&&drop| dist.dist(snap.cell, drop))
        else {
            continue;
        };
        let sticky_valid = sticky
            .get(&bot.id)
            .map(|r| r.dropoff == dropoff && r.remaining > 0)
            .unwrap_or(false);
        couriers.push((
            bot.id.clone(),
            dropoff,
            dist.dist(snap.cell, dropoff),
            snap.blocked_ticks,
            snap.repeated_failed_moves,
            sticky_valid,
        ));
    }

    let primary_dropoff = couriers
        .iter()
        .map(|(_, dropoff, _, _, _, _)| *dropoff)
        .next()
        .or_else(|| map.dropoff_cells.first().copied());
    let Some(dropoff) = primary_dropoff else {
        return plan;
    };

    let mut ring_cells = vec![dropoff];
    ring_cells.extend(map.neighbors[dropoff as usize].iter().copied());
    ring_cells.sort_unstable();
    ring_cells.dedup();
    plan.ring_cells = ring_cells.clone();

    let lane_cells = build_queue_lane_cells(map, dist, dropoff, state.bots.len().saturating_add(3), &ring_cells);
    plan.lane_cells = lane_cells.clone();

    couriers.sort_by(|a, b| {
        b.5.cmp(&a.5)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| b.3.cmp(&a.3))
            .then_with(|| b.4.cmp(&a.4))
            .then_with(|| a.0.cmp(&b.0))
    });
    for (rank, (bot_id, ..)) in couriers.iter().enumerate() {
        plan.queue_eta_rank_by_bot
            .insert(bot_id.clone(), rank as u16);
    }

    let mut ring_entrants = if cfg.strict_queue {
        cfg.max_ring_entrants.max(1).min(1)
    } else {
        cfg.max_ring_entrants.max(1).min(2)
    };
    let mut max_queue_couriers = usize::MAX;

    let mut occupancy: HashMap<u16, u16> = HashMap::new();
    for bot in &state.bots {
        if let Some(cell) = map.idx(bot.x, bot.y) {
            *occupancy.entry(cell).or_insert(0) += 1;
        }
    }
    let lane_set: HashSet<u16> = lane_cells.iter().take(4).copied().collect();
    let ring_set: HashSet<u16> = ring_cells.iter().copied().collect();
    let lane_congestion_now: u16 = occupancy
        .iter()
        .filter(|(idx, _)| lane_set.contains(idx) || ring_set.contains(idx))
        .map(|(_, count)| *count)
        .sum();
    if lane_congestion_now >= 3 {
        ring_entrants = 1;
        max_queue_couriers = 1;
    }

    for (idx, (bot_id, drop, _, _, _, _)) in couriers.into_iter().enumerate() {
        let (role, goal_cell, slot_index) = if idx < ring_entrants {
            (BotRole::LeadCourier, Some(drop), None)
        } else if idx - ring_entrants < max_queue_couriers {
            let q_idx = idx - ring_entrants;
            let goal = lane_cells.get(q_idx).copied().or(Some(drop));
            (BotRole::QueueCourier, goal, Some(q_idx))
        } else {
            (BotRole::Collector, None, None)
        };
        let queue_distance = goal_cell
            .zip(bot_snapshot.get(&bot_id).map(|s| s.cell))
            .map(|(goal, cell)| dist.dist(cell, goal))
            .unwrap_or(0);
        let next_remaining = match sticky.get(&bot_id) {
            Some(prev) if prev.goal_cell == goal_cell.unwrap_or(drop) && prev.role == role => {
                prev.remaining.saturating_sub(1)
            }
            _ => cfg.hold_ticks,
        };
        plan.assignments.insert(
            bot_id.clone(),
            QueueAssignment {
                role,
                goal_cell,
                slot_index,
                dropoff: Some(drop),
                queue_distance,
            },
        );
        plan.queue_order.push(bot_id.clone());
        plan.next_sticky.insert(
            bot_id,
            StickyQueueRole {
                role,
                dropoff: drop,
                goal_cell: goal_cell.unwrap_or(drop),
                slot_index,
                remaining: next_remaining,
            },
        );
    }

    let ring_set: HashSet<u16> = plan.ring_cells.iter().copied().collect();
    let lane_front: HashSet<u16> = plan.lane_cells.iter().take(4).copied().collect();
    for bot in &state.bots {
        if plan.assignments.contains_key(&bot.id) {
            continue;
        }
        let cell = map.idx(bot.x, bot.y).unwrap_or(0);
        let role = if cfg.strict_queue && (ring_set.contains(&cell) || lane_front.contains(&cell)) {
            BotRole::Yield
        } else if !state.orders.is_empty() {
            BotRole::Collector
        } else {
            BotRole::Idle
        };
        if matches!(role, BotRole::Yield) {
            plan.near_dropoff_blocking.insert(bot.id.clone());
        }
        plan.assignments.insert(
            bot.id.clone(),
            QueueAssignment {
                role,
                goal_cell: None,
                slot_index: None,
                dropoff: Some(dropoff),
                queue_distance: 0,
            },
        );
    }

    for bot in &state.bots {
        let cell = map.idx(bot.x, bot.y).unwrap_or(0);
        let is_ring = ring_set.contains(&cell);
        let Some(assignment) = plan.assignments.get(&bot.id) else {
            continue;
        };
        if is_ring && !matches!(assignment.role, BotRole::LeadCourier) {
            plan.violations.insert(bot.id.clone());
        }
    }

    plan
}

fn build_queue_lane_cells(
    map: &MapCache,
    dist: &DistanceMap,
    dropoff: u16,
    count: usize,
    ring_cells: &[u16],
) -> Vec<u16> {
    let ring_set: HashSet<u16> = ring_cells.iter().copied().collect();
    let mut candidates = Vec::<(u16, i32, i32, u16)>::new();
    for idx in 0..map.neighbors.len() as u16 {
        if map.wall_mask[idx as usize] || ring_set.contains(&idx) {
            continue;
        }
        let d = dist.dist(idx, dropoff);
        if d == u16::MAX || d < 2 {
            continue;
        }
        let (x, y) = map.xy(idx);
        candidates.push((d, y, x, idx));
    }
    candidates.sort_by(|a, b| a.cmp(b));
    candidates.into_iter().take(count).map(|(_, _, _, idx)| idx).collect()
}

fn build_traffic(
    state: &GameState,
    map: &MapCache,
    queue: &QueuePlan,
    bot_snapshot: &HashMap<String, BotSnapshot>,
) -> TrafficSnapshot {
    let mut occupancy: HashMap<u16, u16> = HashMap::new();
    for bot in &state.bots {
        if let Some(cell) = map.idx(bot.x, bot.y) {
            *occupancy.entry(cell).or_insert(0) += 1;
        }
    }

    let mut conflict_degree_by_bot = HashMap::new();
    let mut local_conflict_count_by_bot = HashMap::new();
    let mut local_conflict_candidates = HashMap::<String, Vec<String>>::new();
    for bot in &state.bots {
        let mut candidates = Vec::new();
        let degree = state
            .bots
            .iter()
            .filter(|other| other.id != bot.id)
            .filter(|other| {
                let close = (other.x - bot.x).abs() + (other.y - bot.y).abs() <= 1;
                if close {
                    candidates.push(other.id.clone());
                }
                close
            })
            .count() as u16;
        conflict_degree_by_bot.insert(bot.id.clone(), degree);
        local_conflict_count_by_bot.insert(bot.id.clone(), degree);
        local_conflict_candidates.insert(bot.id.clone(), candidates);
    }

    let conflict_hotspots = occupancy
        .iter()
        .filter_map(|(idx, count)| {
            if *count > 1 {
                Some((*idx, *count))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let lane_set: HashSet<u16> = queue.lane_cells.iter().copied().collect();
    let ring_set: HashSet<u16> = queue.ring_cells.iter().copied().collect();
    let lane_congestion = occupancy
        .iter()
        .filter(|(idx, _)| lane_set.contains(idx) || ring_set.contains(idx))
        .map(|(_, count)| *count)
        .sum();
    let blocked_bot_count = bot_snapshot.values().filter(|s| s.blocked_ticks >= 1).count() as u16;
    let stuck_bot_count = bot_snapshot.values().filter(|s| s.blocked_ticks >= 2).count() as u16;

    TrafficSnapshot {
        conflict_degree_by_bot,
        local_conflict_count_by_bot,
        local_conflict_candidates,
        conflict_hotspots,
        lane_congestion,
        blocked_bot_count,
        stuck_bot_count,
    }
}

fn build_movement(
    state: &GameState,
    map: &MapCache,
    dist: &DistanceMap,
    bot_snapshot: &HashMap<String, BotSnapshot>,
    queue: &QueuePlan,
    dropoff_control_zone: &HashSet<u16>,
    prohibited_moves: &HashMap<String, Vec<BlockedMove>>,
    cfg: TeamContextConfig,
) -> MovementReservation {
    let mut priorities = HashMap::new();
    let mut reserve_horizon = HashMap::new();
    let mut forbidden_cells: HashMap<String, HashSet<u16>> = HashMap::new();
    let mut queue_relaxation_active: HashMap<String, bool> = HashMap::new();
    let mut role_by_bot: HashMap<String, BotRole> = HashMap::new();
    let ring_set: HashSet<u16> = queue.ring_cells.iter().copied().collect();
    let lane_head: HashSet<u16> = queue.lane_cells.iter().take(4).copied().collect();
    let strict_forbidden = cfg.strict_queue && !queue.queue_order.is_empty();
    let active_carrying_count = bot_snapshot
        .values()
        .filter(|snap| snap.carrying_active)
        .count();

    for bot in &state.bots {
        let assignment = queue.assignments.get(&bot.id);
        let role = assignment.map(|a| a.role).unwrap_or(BotRole::Idle);
        role_by_bot.insert(bot.id.clone(), role);
        let blocked = bot_snapshot
            .get(&bot.id)
            .map(|s| s.blocked_ticks)
            .unwrap_or_default();
        let escape_active = bot_snapshot
            .get(&bot.id)
            .map(|s| s.escape_macro_ticks > 0)
            .unwrap_or(false);
        let mut priority = role.priority().saturating_sub(blocked.min(1));
        if escape_active && !matches!(role, BotRole::LeadCourier) {
            priority = priority.min(1);
        }
        priorities.insert(bot.id.clone(), priority);

        let cell = map.idx(bot.x, bot.y).unwrap_or(0);
        let near_drop = map
            .dropoff_cells
            .iter()
            .map(|&drop| dist.dist(cell, drop))
            .min()
            .unwrap_or(u16::MAX);
        let horizon = if matches!(role, BotRole::LeadCourier) {
            3
        } else if escape_active {
            3
        } else if matches!(role, BotRole::QueueCourier) {
            2
        } else if near_drop <= 3 {
            2
        } else {
            1
        };
        reserve_horizon.insert(bot.id.clone(), horizon);

        let relax_ticks = bot_snapshot
            .get(&bot.id)
            .map(|s| s.constraint_relax_ticks)
            .unwrap_or(0);
        let relaxation_active = relax_ticks > 0;
        queue_relaxation_active.insert(bot.id.clone(), relaxation_active);

        if strict_forbidden
            && !matches!(role, BotRole::LeadCourier | BotRole::QueueCourier)
            && !relaxation_active
        {
            let mut blocked_cells = HashSet::new();
            blocked_cells.extend(ring_set.iter().copied());
            blocked_cells.extend(lane_head.iter().copied());
            if active_carrying_count > 0 {
                blocked_cells.extend(dropoff_control_zone.iter().copied());
            }
            blocked_cells.remove(&cell);
            forbidden_cells.insert(bot.id.clone(), blocked_cells);
        }
    }

    MovementReservation {
        priorities,
        reserve_horizon,
        prohibited_moves: prohibited_moves.clone(),
        forbidden_cells,
        queue_relaxation_active,
        role_by_bot,
        dropoff_control_zone: dropoff_control_zone.clone(),
    }
}

fn build_topology(map: &MapCache) -> (Vec<u8>, Vec<u8>, HashSet<u16>, HashSet<u16>) {
    let n = map.neighbors.len();
    let mut degree = vec![0u8; n];
    for idx in 0..n {
        if map.wall_mask[idx] {
            continue;
        }
        degree[idx] = map.neighbors[idx].len().min(4) as u8;
    }
    let junction_cells = degree
        .iter()
        .enumerate()
        .filter_map(|(idx, deg)| if *deg >= 3 { Some(idx as u16) } else { None })
        .collect::<HashSet<_>>();
    let corner_cells = degree
        .iter()
        .enumerate()
        .filter_map(|(idx, deg)| {
            if !map.wall_mask[idx] && *deg <= 1 {
                Some(idx as u16)
            } else {
                None
            }
        })
        .collect::<HashSet<_>>();

    let mut working = degree.clone();
    let mut dead_end_depth = vec![0u8; n];
    let mut queue = VecDeque::new();
    for idx in 0..n {
        if map.wall_mask[idx] {
            continue;
        }
        if working[idx] <= 1 {
            queue.push_back((idx as u16, 1u8));
        }
    }
    while let Some((cell, depth)) = queue.pop_front() {
        let cell_usize = cell as usize;
        if depth <= dead_end_depth[cell_usize] {
            continue;
        }
        dead_end_depth[cell_usize] = depth;
        if working[cell_usize] == 0 {
            continue;
        }
        working[cell_usize] = 0;
        for &nbr in &map.neighbors[cell_usize] {
            let nidx = nbr as usize;
            if working[nidx] == 0 {
                continue;
            }
            working[nidx] = working[nidx].saturating_sub(1);
            if working[nidx] == 1 {
                queue.push_back((nbr, depth.saturating_add(1)));
            }
        }
    }
    (degree, dead_end_depth, junction_cells, corner_cells)
}

fn build_dropoff_control_zone(
    map: &MapCache,
    queue: &QueuePlan,
    radius: u8,
) -> HashSet<u16> {
    let mut zone = HashSet::new();
    if map.dropoff_cells.is_empty() {
        return zone;
    }
    let r = i32::from(radius);
    for idx in 0..map.neighbors.len() as u16 {
        if map.wall_mask[idx as usize] {
            continue;
        }
        let (x, y) = map.xy(idx);
        let near_drop = map.dropoff_cells.iter().any(|&drop| {
            let (dx, dy) = map.xy(drop);
            (dx - x).abs() + (dy - y).abs() <= r
        });
        if near_drop {
            zone.insert(idx);
        }
    }
    zone.extend(queue.ring_cells.iter().copied());
    zone.extend(queue.lane_cells.iter().take(8).copied());
    zone
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        dist::DistanceMap,
        model::{BotState, GameState, Grid, Item, Order, OrderStatus},
        world::World,
    };

    use super::{BotRole, TeamContext, TeamContextConfig};

    #[test]
    fn queue_assignment_keeps_single_lead() {
        let state = GameState {
            grid: Grid {
                width: 12,
                height: 10,
                drop_off_tiles: vec![[1, 1]],
                ..Grid::default()
            },
            bots: vec![
                BotState {
                    id: "0".to_owned(),
                    x: 6,
                    y: 6,
                    carrying: vec!["milk".to_owned()],
                    capacity: 3,
                },
                BotState {
                    id: "1".to_owned(),
                    x: 7,
                    y: 6,
                    carrying: vec!["milk".to_owned()],
                    capacity: 3,
                },
                BotState {
                    id: "2".to_owned(),
                    x: 8,
                    y: 6,
                    carrying: vec!["milk".to_owned()],
                    capacity: 3,
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

        let lead_count = ctx
            .queue
            .assignments
            .values()
            .filter(|assignment| matches!(assignment.role, BotRole::LeadCourier))
            .count();
        assert_eq!(lead_count, 1);
    }

    #[test]
    fn kind_to_stands_is_deterministic() {
        let state = GameState {
            grid: Grid {
                width: 8,
                height: 6,
                drop_off_tiles: vec![[1, 4]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 6,
                y: 4,
                carrying: vec![],
                capacity: 3,
            }],
            items: vec![
                Item {
                    id: "item_9".to_owned(),
                    kind: "milk".to_owned(),
                    x: 3,
                    y: 2,
                },
                Item {
                    id: "item_2".to_owned(),
                    kind: "milk".to_owned(),
                    x: 5,
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
        let a = TeamContext::build(
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
        let b = TeamContext::build(
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
        assert_eq!(a.knowledge.kind_to_stands, b.knowledge.kind_to_stands);
        let milk = a
            .knowledge
            .kind_to_stands
            .get("milk")
            .cloned()
            .unwrap_or_default();
        assert!(!milk.is_empty());
    }

    #[test]
    fn area_id_assignment_is_stable() {
        let state = GameState {
            grid: Grid {
                width: 10,
                height: 6,
                drop_off_tiles: vec![[1, 4]],
                walls: vec![[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 2,
                y: 4,
                carrying: vec![],
                capacity: 3,
            }],
            items: vec![],
            orders: vec![],
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
        let left = map.idx(2, 2).expect("left area");
        let right = map.idx(7, 2).expect("right area");
        assert_ne!(
            ctx.knowledge.area_id_by_cell[left as usize],
            ctx.knowledge.area_id_by_cell[right as usize]
        );
    }
}
