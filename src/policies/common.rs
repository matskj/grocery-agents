use std::collections::HashMap;

use crate::{
    dist::DistanceMap,
    model::{BotState, GameState, OrderStatus},
    team_context::TeamContext,
    world::MapCache,
};

#[derive(Debug, Clone)]
pub struct ItemCandidate<'a> {
    pub item_id: &'a str,
    pub kind: &'a str,
    pub x: i32,
    pub y: i32,
}

pub fn active_missing_kinds(state: &GameState) -> Vec<String> {
    state
        .orders
        .iter()
        .filter(|order| matches!(order.status, OrderStatus::InProgress))
        .map(|order| order.item_id.clone())
        .collect()
}

pub fn preview_missing_kinds(state: &GameState) -> Vec<String> {
    state
        .orders
        .iter()
        .filter(|order| matches!(order.status, OrderStatus::Pending))
        .map(|order| order.item_id.clone())
        .collect()
}

pub fn active_kind_counts(state: &GameState) -> HashMap<String, u16> {
    let mut out = HashMap::new();
    for kind in active_missing_kinds(state) {
        *out.entry(kind).or_insert(0) += 1;
    }
    out
}

pub fn preview_kind_counts(state: &GameState) -> HashMap<String, u16> {
    let mut out = HashMap::new();
    for kind in preview_missing_kinds(state) {
        *out.entry(kind).or_insert(0) += 1;
    }
    out
}

pub fn item_candidates_by_kind<'a>(
    state: &'a GameState,
) -> HashMap<&'a str, Vec<ItemCandidate<'a>>> {
    let mut out: HashMap<&str, Vec<ItemCandidate<'a>>> = HashMap::new();
    for item in &state.items {
        out.entry(item.kind.as_str())
            .or_default()
            .push(ItemCandidate {
                item_id: item.id.as_str(),
                kind: item.kind.as_str(),
                x: item.x,
                y: item.y,
            });
    }
    out
}

pub fn nearest_dropoff_cell(map: &MapCache, from: u16, dist: &DistanceMap) -> Option<u16> {
    map.dropoff_cells
        .iter()
        .copied()
        .min_by_key(|&cell| dist.dist(from, cell))
}

pub fn on_dropoff(bot: &BotState, map: &MapCache) -> bool {
    map.idx(bot.x, bot.y)
        .map(|cell| map.dropoff_cells.contains(&cell))
        .unwrap_or(false)
}

pub fn bot_has_active_item(bot: &BotState, active_counts: &HashMap<String, u16>) -> bool {
    bot.carrying
        .iter()
        .any(|kind| active_counts.get(kind).copied().unwrap_or(0) > 0)
}

pub fn first_active_order_for_kind<'a>(state: &'a GameState, kind: &str) -> Option<&'a str> {
    state
        .orders
        .iter()
        .find(|order| matches!(order.status, OrderStatus::InProgress) && order.item_id == kind)
        .map(|order| order.id.as_str())
}

pub fn bot_cell(map: &MapCache, bot: &BotState) -> Option<u16> {
    map.idx(bot.x, bot.y)
}

pub fn choose_nearest_item_of_kind<'a>(
    state: &'a GameState,
    map: &MapCache,
    dist: &DistanceMap,
    from: u16,
    kind: &str,
) -> Option<(&'a str, u16)> {
    let mut best: Option<(&str, u16, u16)> = None;
    for item in &state.items {
        if item.kind != kind {
            continue;
        }
        for &stand in map.stand_cells_for_item(&item.id) {
            let d = dist.dist(from, stand);
            if d == u16::MAX {
                continue;
            }
            match best {
                Some((_, _, bd)) if d >= bd => {}
                _ => best = Some((item.id.as_str(), stand, d)),
            }
        }
    }
    best.map(|(id, stand, _)| (id, stand))
}

#[allow(clippy::too_many_arguments)]
pub fn choose_scored_item_of_kind<'a>(
    state: &'a GameState,
    map: &MapCache,
    dist: &DistanceMap,
    team: &TeamContext,
    bot_id: &str,
    from: u16,
    kind: &str,
    carrying_len: usize,
    capacity: usize,
) -> Option<(&'a str, u16)> {
    let local_conflict = team
        .traffic
        .local_conflict_count_by_bot
        .get(bot_id)
        .copied()
        .unwrap_or(0) as f64;
    let lane_congestion = f64::from(team.traffic.lane_congestion);
    let congestion_penalty = lane_congestion + local_conflict;
    let inventory_penalty = if carrying_len.saturating_add(1) >= capacity {
        8.0
    } else if carrying_len.saturating_add(2) >= capacity {
        4.0
    } else {
        0.0
    };

    let mut best: Option<(&str, u16, f64)> = None;
    for item in &state.items {
        if item.kind != kind {
            continue;
        }
        for &stand in map.stand_cells_for_item(&item.id) {
            let d_bot = dist.dist(from, stand);
            if d_bot == u16::MAX {
                continue;
            }
            let d_drop = map
                .dropoff_bfs
                .get(stand as usize)
                .copied()
                .unwrap_or(u16::MAX);
            if d_drop == u16::MAX {
                continue;
            }
            let choke_penalty = if map.is_choke_point(stand) { 6.0 } else { 0.0 };
            let route_penalty = 1.4 * congestion_penalty + inventory_penalty + choke_penalty;
            let score = f64::from(d_bot) + f64::from(d_drop) + route_penalty;
            match best {
                Some((_, _, cur)) if score >= cur => {}
                _ => best = Some((item.id.as_str(), stand, score)),
            }
        }
    }
    best.map(|(id, stand, _)| (id, stand))
}

pub fn is_adjacent_to_item(bot: &BotState, item_x: i32, item_y: i32) -> bool {
    (item_x - bot.x).abs() + (item_y - bot.y).abs() == 1
}
