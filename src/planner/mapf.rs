use std::collections::{HashMap, HashSet};

use crate::{
    dist::DistanceMap,
    model::{Action, GameState},
    world::MapCache,
};

use super::{
    common::{best_step_towards, bot_cell, intent_to_action, move_delta},
    Intent,
};

#[derive(Debug, Clone)]
pub struct MapfOutcome {
    pub actions_by_bot: HashMap<String, Action>,
    pub conflict_count_by_bot: HashMap<String, u64>,
    pub reserved_cells_by_t: Vec<Vec<u16>>,
}

pub fn resolve_one_step(
    state: &GameState,
    map: &MapCache,
    dist: &DistanceMap,
    intents: &HashMap<String, Intent>,
    priority: &[String],
    lookahead: usize,
) -> MapfOutcome {
    let mut ordered = priority.to_vec();
    for bot in &state.bots {
        if !ordered.iter().any(|id| id == &bot.id) {
            ordered.push(bot.id.clone());
        }
    }

    let mut actions_by_bot = HashMap::<String, Action>::new();
    let mut conflict_count_by_bot = HashMap::<String, u64>::new();
    let mut reserved_cells: Vec<HashSet<u16>> = vec![HashSet::new(); lookahead.max(1) + 1];
    let mut reserved_edges = HashSet::<(u16, u16)>::new();

    for bot_id in ordered {
        let Some(bot) = state.bots.iter().find(|b| b.id == bot_id) else {
            continue;
        };
        let Some(start) = bot_cell(map, bot) else {
            actions_by_bot.insert(bot.id.clone(), Action::wait(bot.id.clone()));
            continue;
        };
        let intent = intents.get(&bot.id).cloned().unwrap_or(Intent::Wait);

        match intent {
            Intent::DropOff { .. } | Intent::PickUp { .. } | Intent::Wait => {
                reserved_cells[1].insert(start);
                actions_by_bot.insert(bot.id.clone(), intent_to_action(&intent, &bot.id, None));
            }
            Intent::MoveTo { cell: target } => {
                let mut blocked = reserved_cells[1].clone();
                blocked.remove(&start);

                let mut chosen = best_step_towards(map, dist, start, target, &blocked);
                let mut conflicts = 0u64;
                if let Some(next) = chosen {
                    if reserved_edges.contains(&(next, start)) {
                        conflicts = conflicts.saturating_add(1);
                        chosen = None;
                    }
                }

                if chosen.is_none() {
                    let mut local_blocked = blocked;
                    local_blocked.insert(start);
                    chosen = best_step_towards(map, dist, start, target, &local_blocked);
                    if chosen.is_none() {
                        conflicts = conflicts.saturating_add(1);
                    }
                }

                let next = chosen.unwrap_or(start);
                conflict_count_by_bot.insert(bot.id.clone(), conflicts);
                reserved_cells[1].insert(next);
                reserved_edges.insert((start, next));

                for t in 2..=lookahead.max(1) {
                    reserved_cells[t].insert(next);
                }

                if next == start {
                    actions_by_bot.insert(bot.id.clone(), Action::wait(bot.id.clone()));
                } else {
                    let (dx, dy) = move_delta(map, start, next);
                    actions_by_bot.insert(
                        bot.id.clone(),
                        Action::Move {
                            bot_id: bot.id.clone(),
                            dx,
                            dy,
                        },
                    );
                }
            }
        }
    }

    let mut reserved_cells_by_t = Vec::new();
    for cells in reserved_cells.into_iter().skip(1) {
        let mut row = cells.into_iter().collect::<Vec<_>>();
        row.sort_unstable();
        reserved_cells_by_t.push(row);
    }

    MapfOutcome {
        actions_by_bot,
        conflict_count_by_bot,
        reserved_cells_by_t,
    }
}
