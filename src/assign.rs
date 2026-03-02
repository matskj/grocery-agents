use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::dispatcher::{BotIntent, Intent};
use crate::dist::DistanceMap;
use crate::model::{GameState, OrderStatus};
use crate::team_context::{BotRole, TeamContext};
use crate::world::MapCache;

#[derive(Debug, Clone)]
pub struct AssignmentResult {
    pub intents: Vec<BotIntent>,
    pub task_count: usize,
    pub edge_count: usize,
}

#[derive(Debug, Clone)]
struct Task {
    task_id: usize,
    sort_key: (u8, String, u16),
    target_cell: u16,
    shareable: bool,
    intent: Intent,
}

#[derive(Debug, Clone)]
struct Edge {
    bot_id: String,
    task_id: usize,
    cost: i32,
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
            });
        }

        let mut bots = state.bots.iter().collect::<Vec<_>>();
        bots.sort_by(|a, b| a.id.cmp(&b.id));

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
            let role = team.role_for(&bot.id);
            let queue_goal = team.queue_goal_for(&bot.id);
            let mut ranked = tasks
                .iter()
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
                    if carrying_active && matches!(task.intent, Intent::MoveTo { .. }) {
                        adjusted -= 20;
                    }
                    if matches!(task.intent, Intent::DropOff { .. }) {
                        adjusted -= 50;
                    }
                    if matches!(role, BotRole::LeadCourier | BotRole::QueueCourier)
                        && matches!(task.intent, Intent::MoveTo { .. })
                        && queue_goal == Some(task.target_cell)
                    {
                        adjusted -= 15;
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

        let task_by_id = tasks
            .iter()
            .map(|task| (task.task_id, task.clone()))
            .collect::<HashMap<_, _>>();
        let mut bot_taken = HashSet::<String>::new();
        let mut task_taken = HashSet::<usize>::new();
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
            bot_taken.insert(edge.bot_id.clone());
            if !task.shareable {
                task_taken.insert(edge.task_id);
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
                    intent: Intent::DropOff {
                        order_id: order.id.clone(),
                    },
                });
                next_id += 1;
            }
        }
    }

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
            intent: Intent::MoveTo { cell: drop },
        });
        next_id += 1;
    }

    let mut item_order = state.items.iter().collect::<Vec<_>>();
    item_order.sort_by(|a, b| a.id.cmp(&b.id));
    let mut active_slots_left = active_missing.clone();
    let mut preview_slots_left = preview_missing.clone();
    for item in item_order {
        let active_left = active_slots_left.get(&item.kind).copied().unwrap_or(0);
        let preview_left = preview_slots_left.get(&item.kind).copied().unwrap_or(0);
        if active_left == 0 && preview_left == 0 {
            continue;
        }
        let tier = if active_left > 0 { 2 } else { 3 };
        if active_left > 0 {
            active_slots_left.insert(item.kind.clone(), active_left.saturating_sub(1));
        } else {
            preview_slots_left.insert(item.kind.clone(), preview_left.saturating_sub(1));
        }
        let mut stands = map.stand_cells_for_item(&item.id).to_vec();
        stands.sort_unstable();
        for stand in stands {
            let nearest_drop = map
                .dropoff_cells
                .iter()
                .copied()
                .min_by_key(|drop| dist.dist(stand, *drop))
                .unwrap_or(stand);
            let intent = Intent::MoveTo { cell: stand };
            tasks.push(Task {
                task_id: next_id,
                sort_key: (tier, item.kind.clone(), stand),
                target_cell: stand,
                shareable: false,
                intent,
            });
            next_id += 1;
            let pickup_intent = Intent::PickUp {
                item_id: item.id.clone(),
            };
            tasks.push(Task {
                task_id: next_id,
                sort_key: (tier, item.kind.clone(), nearest_drop),
                target_cell: stand,
                shareable: false,
                intent: pickup_intent,
            });
            next_id += 1;
        }
    }

    if let Some(drop) = map.dropoff_cells.first().copied() {
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
                    intent: Intent::MoveTo { cell: goal },
                });
                next_id += 1;
            } else {
                tasks.push(Task {
                    task_id: next_id,
                    sort_key: (4, bot.id.clone(), drop),
                    target_cell: drop,
                    shareable: true,
                    intent: Intent::MoveTo { cell: drop },
                });
                next_id += 1;
            }
        }
    }

    tasks.sort_by(|a, b| a.sort_key.cmp(&b.sort_key).then_with(|| a.task_id.cmp(&b.task_id)));
    Some(tasks)
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
    match &task.intent {
        Intent::DropOff { .. } => i32::from(dist_to(bot_cell, task.target_cell, dist)),
        Intent::MoveTo { cell } => {
            let d0 = f64::from(dist_to(bot_cell, *cell, dist));
            let nearest_drop = map
                .dropoff_cells
                .iter()
                .copied()
                .map(|drop| f64::from(dist_to(*cell, drop, dist)))
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let density = local_density(*cell, state) as f64;
            let choke = choke_penalty(*cell, map) as f64;
            (d0 + nearest_drop + lambda_density * density + lambda_choke * choke).round() as i32
        }
        Intent::PickUp { .. } => {
            if bot_cell != task.target_cell {
                return 9_000;
            }
            let d = f64::from(dist_to(bot_cell, task.target_cell, dist));
            let density = local_density(task.target_cell, state) as f64;
            (d + lambda_density * density).round() as i32
        }
        Intent::Wait => 10_000,
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
    if d == u16::MAX { 256 } else { d }
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

    use super::AssignmentEngine;

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
}
