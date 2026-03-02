use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::dispatcher::{BotIntent, Intent};
use crate::dist::DistanceMap;
use crate::model::{BotState, GameState, OrderStatus};
use crate::scoring::{maybe_score_pick, CandidateFeatures};
use crate::team_context::{BotRole, TeamContext};
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
}

#[derive(Debug, Clone)]
struct Task {
    task_id: usize,
    sort_key: (u8, String, u16),
    target_cell: u16,
    shareable: bool,
    kind: TaskKind,
    demand_tier: DemandTier,
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
        soft_budget: Duration,
    ) -> Option<AssignmentResult> {
        let started = Instant::now();
        let tasks = build_tasks(state, map, dist, team)?;
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
            .filter(|task| {
                matches!(task.kind, TaskKind::PickupStand | TaskKind::ImmediatePickup)
            })
            .count();
        let dropoff_task_count = tasks
            .iter()
            .filter(|task| {
                matches!(task.kind, TaskKind::ImmediateDropOff | TaskKind::CarryToDropoff)
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
            let blocked_ticks = team
                .bot_snapshot
                .get(&bot.id)
                .map(|snap| f64::from(snap.blocked_ticks))
                .unwrap_or(0.0);
            let on_dropoff = map
                .idx(bot.x, bot.y)
                .map(|cell| map.dropoff_cells.contains(&cell))
                .unwrap_or(false);

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
                        let features = CandidateFeatures {
                            dist_to_nearest_active_item: dist_to_stand,
                            dist_to_dropoff: f64::from(task.nearest_drop_dist.min(64)),
                            inventory_util,
                            local_congestion,
                            teammate_proximity,
                            order_urgency,
                            blocked_ticks,
                        };
                        let mode = team.mode.as_str();
                        let model_score = maybe_score_pick(mode, features).unwrap_or(0.0);
                        adjusted -= (model_score * 6.0).round() as i32;
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

        let task_by_id = tasks
            .iter()
            .map(|task| (task.task_id, task.clone()))
            .collect::<HashMap<_, _>>();
        let mut bot_taken = HashSet::<String>::new();
        let mut task_taken = HashSet::<usize>::new();
        let mut target_cell_taken = HashMap::<u16, u8>::new();
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
                let used = target_cell_taken.get(&task.target_cell).copied().unwrap_or(0);
                let cap = task.target_share_cap(sparse_active_stands);
                if used >= cap {
                    continue;
                }
            }
            bot_taken.insert(edge.bot_id.clone());
            if !task.shareable {
                task_taken.insert(edge.task_id);
                let used = target_cell_taken.get(&task.target_cell).copied().unwrap_or(0);
                target_cell_taken.insert(task.target_cell, used.saturating_add(1));
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

        Some(AssignmentResult {
            intents,
            task_count: tasks.len(),
            edge_count: all_edges.len(),
            active_task_count,
            preview_task_count,
            stand_task_count,
            dropoff_task_count,
        })
    }
}

fn build_tasks(
    state: &GameState,
    map: &MapCache,
    dist: &DistanceMap,
    team: &TeamContext,
) -> Option<Vec<Task>> {
    if state.bots.is_empty() {
        return Some(Vec::new());
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
        let demand = active_missing.get(&kind).copied().unwrap_or(0);
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
    for kind in preview_kinds {
        let demand = preview_missing.get(&kind).copied().unwrap_or(0);
        if demand == 0 || active_missing.get(&kind).copied().unwrap_or(0) > 0 {
            continue;
        }
        let Some(pool) = stand_pool.get(&kind) else {
            continue;
        };
        for candidate in pool {
            tasks.push(Task {
                task_id: next_id,
                sort_key: (3, kind.clone(), candidate.stand),
                target_cell: candidate.stand,
                shareable: false,
                kind: TaskKind::PickupStand,
                demand_tier: DemandTier::Preview,
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
                nearest_drop_dist: candidate.nearest_drop_dist,
                intent: Intent::PickUp {
                    item_id: candidate.item_id.clone(),
                },
            });
            next_id += 1;
        }
    }

    if active_carrier_exists {
        let Some(drop) = map.dropoff_cells.first().copied() else {
            return Some(tasks);
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
                    nearest_drop_dist: 0,
                    intent: Intent::MoveTo { cell: drop },
                });
                next_id += 1;
            }
        }
    }

    tasks.sort_by(|a, b| a.sort_key.cmp(&b.sort_key).then_with(|| a.task_id.cmp(&b.task_id)));
    Some(tasks)
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
            (d
                + 0.45 * f64::from(task.nearest_drop_dist.min(64))
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

    use super::{compare_item_ids, AssignmentEngine};

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
                Duration::from_millis(200),
            )
            .expect("assignment");
        assert_eq!(format!("{:?}", first.intents), format!("{:?}", second.intents));
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
                Duration::from_millis(200),
            )
            .expect("assignment");
        assert!(result.active_task_count > 0);
        assert!(result.preview_task_count > 0);
        let first = result
            .intents
            .into_iter()
            .find(|intent| intent.bot_id == "0")
            .expect("bot intent");
        assert!(matches!(first.intent, crate::dispatcher::Intent::PickUp { .. }));
        if let crate::dispatcher::Intent::PickUp { item_id } = first.intent {
            assert_eq!(item_id, "item_12");
        }
    }
}
