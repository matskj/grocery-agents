use std::collections::{HashMap, HashSet};

use crate::{
    dist::DistanceMap,
    model::{Action, BotState, GameState, OrderStatus},
    world::MapCache,
};

use super::Intent;

#[derive(Debug, Clone, Copy)]
pub struct ItemTarget<'a> {
    pub item_id: &'a str,
    pub stand_cell: u16,
    pub item_x: i32,
    pub item_y: i32,
}

pub fn active_kind_counts(state: &GameState) -> HashMap<String, u16> {
    let mut counts = HashMap::<String, u16>::new();
    for order in &state.orders {
        if matches!(order.status, OrderStatus::InProgress) {
            *counts.entry(order.item_id.clone()).or_insert(0) += 1;
        }
    }
    counts
}

pub fn preview_kind_counts(state: &GameState) -> HashMap<String, u16> {
    let mut counts = HashMap::<String, u16>::new();
    for order in &state.orders {
        if matches!(order.status, OrderStatus::Pending) {
            *counts.entry(order.item_id.clone()).or_insert(0) += 1;
        }
    }
    counts
}

pub fn active_missing_total(state: &GameState) -> usize {
    state
        .orders
        .iter()
        .filter(|order| matches!(order.status, OrderStatus::InProgress))
        .count()
}

pub fn bot_cell(map: &MapCache, bot: &BotState) -> Option<u16> {
    map.idx(bot.x, bot.y)
}

pub fn on_dropoff(map: &MapCache, bot: &BotState) -> bool {
    map.idx(bot.x, bot.y)
        .map(|cell| map.dropoff_cells.contains(&cell))
        .unwrap_or(false)
}

pub fn nearest_dropoff_cell(map: &MapCache, dist: &DistanceMap, from: u16) -> Option<u16> {
    map.dropoff_cells
        .iter()
        .copied()
        .min_by_key(|&drop| dist.dist(from, drop))
}

pub fn first_active_order_for_kind<'a>(state: &'a GameState, kind: &str) -> Option<&'a str> {
    state
        .orders
        .iter()
        .find(|order| matches!(order.status, OrderStatus::InProgress) && order.item_id == kind)
        .map(|order| order.id.as_str())
}

pub fn first_preview_kind(state: &GameState) -> Option<&str> {
    state
        .orders
        .iter()
        .find(|order| matches!(order.status, OrderStatus::Pending))
        .map(|order| order.item_id.as_str())
}

pub fn carrying_kind_counts(bot: &BotState) -> HashMap<String, u16> {
    let mut counts = HashMap::<String, u16>::new();
    for kind in &bot.carrying {
        *counts.entry(kind.clone()).or_insert(0) += 1;
    }
    counts
}

pub fn count_active_carried_team(
    state: &GameState,
    active_counts: &HashMap<String, u16>,
) -> HashMap<String, u16> {
    let mut out = HashMap::<String, u16>::new();
    for bot in &state.bots {
        for kind in &bot.carrying {
            if active_counts.contains_key(kind) {
                *out.entry(kind.clone()).or_insert(0) += 1;
            }
        }
    }
    out
}

pub fn try_dropoff_active(state: &GameState, map: &MapCache, bot: &BotState) -> Option<Intent> {
    if !on_dropoff(map, bot) {
        return None;
    }
    for kind in &bot.carrying {
        if let Some(order_id) = first_active_order_for_kind(state, kind) {
            return Some(Intent::DropOff {
                order_id: order_id.to_owned(),
            });
        }
    }
    None
}

pub fn is_adjacent_to_item(bot: &BotState, item_x: i32, item_y: i32) -> bool {
    (item_x - bot.x).unsigned_abs() + (item_y - bot.y).unsigned_abs() == 1
}

pub fn find_adjacent_item_for_kind<'a>(
    state: &'a GameState,
    bot: &BotState,
    kind: &str,
) -> Option<&'a str> {
    if bot.carrying.len() >= bot.capacity {
        return None;
    }
    state
        .items
        .iter()
        .find(|item| item.kind == kind && is_adjacent_to_item(bot, item.x, item.y))
        .map(|item| item.id.as_str())
}

pub fn item_targets_for_kind<'a>(
    state: &'a GameState,
    map: &'a MapCache,
    kind: &str,
) -> Vec<ItemTarget<'a>> {
    let mut out = Vec::new();
    for item in &state.items {
        if item.kind != kind {
            continue;
        }
        for &stand in map.stand_cells_for_item(&item.id) {
            out.push(ItemTarget {
                item_id: item.id.as_str(),
                stand_cell: stand,
                item_x: item.x,
                item_y: item.y,
            });
        }
    }
    out
}

pub fn pick_best_item_target<'a, F>(
    state: &'a GameState,
    map: &'a MapCache,
    dist: &DistanceMap,
    from: u16,
    kind: &str,
    mut extra_penalty: F,
) -> Option<ItemTarget<'a>>
where
    F: FnMut(ItemTarget<'a>) -> f64,
{
    let mut best = None::<(ItemTarget<'a>, f64)>;
    for target in item_targets_for_kind(state, map, kind) {
        let d_bot = dist.dist(from, target.stand_cell);
        if d_bot == u16::MAX {
            continue;
        }
        let d_drop = dist.dist_to_dropoff(target.stand_cell);
        if d_drop == u16::MAX {
            continue;
        }
        let score = f64::from(d_bot) + f64::from(d_drop) + extra_penalty(target);
        match best {
            Some((_, s)) if score >= s => {}
            _ => best = Some((target, score)),
        }
    }
    best.map(|(t, _)| t)
}

pub fn center_corridor_penalty(map: &MapCache, cell: u16) -> f64 {
    let (x, _) = map.xy(cell);
    let center = map.width / 2;
    if (x - center).unsigned_abs() <= 1 {
        3.0
    } else {
        0.0
    }
}

pub fn region_for_cell(map: &MapCache, cell: u16, region_count: u16) -> u16 {
    if region_count <= 1 || map.width <= 0 {
        return 0;
    }
    let (x, _) = map.xy(cell);
    let width = map.width.max(1);
    let scaled = (x.max(0) * i32::from(region_count)) / width;
    scaled.clamp(0, i32::from(region_count - 1)) as u16
}

pub fn detour_cost(dist: &DistanceMap, a: u16, via: u16, b: u16) -> Option<i32> {
    let direct = dist.dist(a, b);
    let left = dist.dist(a, via);
    let right = dist.dist(via, b);
    if direct == u16::MAX || left == u16::MAX || right == u16::MAX {
        return None;
    }
    Some(i32::from(left) + i32::from(right) - i32::from(direct))
}

pub fn intent_goal_cell(intent: &Intent) -> Option<u16> {
    match intent {
        Intent::MoveTo { cell } => Some(*cell),
        Intent::DropOff { .. } | Intent::PickUp { .. } | Intent::Wait => None,
    }
}

pub fn intent_to_action(intent: &Intent, bot_id: &str, dxdy: Option<(i32, i32)>) -> Action {
    match intent {
        Intent::DropOff { order_id } => Action::DropOff {
            bot_id: bot_id.to_owned(),
            order_id: order_id.clone(),
        },
        Intent::PickUp { item_id } => Action::PickUp {
            bot_id: bot_id.to_owned(),
            item_id: item_id.clone(),
        },
        Intent::MoveTo { .. } => {
            if let Some((dx, dy)) = dxdy {
                Action::Move {
                    bot_id: bot_id.to_owned(),
                    dx,
                    dy,
                }
            } else {
                Action::Wait {
                    bot_id: bot_id.to_owned(),
                }
            }
        }
        Intent::Wait => Action::Wait {
            bot_id: bot_id.to_owned(),
        },
    }
}

pub fn queue_role_from_label(label: &str) -> String {
    if label.contains("runner") || label.contains("stager") {
        "courier".to_owned()
    } else if label.contains("buffer") {
        "yield".to_owned()
    } else if label.contains("picker") || label.contains("worker") || label.contains("solo") {
        "collector".to_owned()
    } else {
        "collector".to_owned()
    }
}

pub fn best_step_towards(
    map: &MapCache,
    dist: &DistanceMap,
    start: u16,
    target: u16,
    blocked_cells: &HashSet<u16>,
) -> Option<u16> {
    let mut best = None::<(u16, u16)>;
    for &nb in &map.neighbors[start as usize] {
        if blocked_cells.contains(&nb) {
            continue;
        }
        let d = dist.dist(nb, target);
        match best {
            Some((_, bd)) if d >= bd => {}
            _ => best = Some((nb, d)),
        }
    }
    best.map(|(cell, _)| cell)
}

pub fn move_delta(map: &MapCache, from: u16, to: u16) -> (i32, i32) {
    let (x0, y0) = map.xy(from);
    let (x1, y1) = map.xy(to);
    (x1 - x0, y1 - y0)
}
