use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{
    model::{Action, GameState, OrderStatus},
    world::World,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuleOutcome {
    pub invalid_actions: u64,
    pub blocked_moves: u64,
    pub items_delivered_delta: u64,
    pub orders_completed_delta: u64,
}

pub fn apply_actions(state: &mut GameState, actions: &[Action]) -> RuleOutcome {
    let world = World::new(state);
    let map = world.map();

    let mut outcome = RuleOutcome::default();

    let mut action_by_bot = HashMap::<String, Action>::new();
    for action in actions {
        action_by_bot.insert(action.bot_id().to_owned(), action.clone());
    }

    // Resolve movements first with deterministic tie-break by bot id.
    let mut proposals = Vec::<(String, u16, u16, bool)>::new();
    for bot in &state.bots {
        let Some(from) = map.idx(bot.x, bot.y) else {
            outcome.invalid_actions = outcome.invalid_actions.saturating_add(1);
            continue;
        };
        let action = action_by_bot
            .get(&bot.id)
            .cloned()
            .unwrap_or_else(|| Action::wait(bot.id.clone()));
        if let Action::Move { dx, dy, .. } = action {
            let nx = bot.x + dx;
            let ny = bot.y + dy;
            let Some(to) = map.idx(nx, ny) else {
                proposals.push((bot.id.clone(), from, from, false));
                outcome.invalid_actions = outcome.invalid_actions.saturating_add(1);
                continue;
            };
            let is_walkable = !state
                .grid
                .walls
                .iter()
                .any(|wall| wall[0] == nx && wall[1] == ny)
                && !state.items.iter().any(|item| item.x == nx && item.y == ny);
            if !is_walkable {
                proposals.push((bot.id.clone(), from, from, false));
                outcome.invalid_actions = outcome.invalid_actions.saturating_add(1);
                continue;
            }
            proposals.push((bot.id.clone(), from, to, true));
        } else {
            proposals.push((bot.id.clone(), from, from, false));
        }
    }

    proposals.sort_by(|a, b| a.0.cmp(&b.0));

    let mut winners = HashMap::<u16, String>::new();
    for (bot_id, _from, to, wants_move) in &proposals {
        if !wants_move {
            continue;
        }
        winners.entry(*to).or_insert_with(|| bot_id.clone());
    }

    let mut swap_edges = HashSet::<(String, String)>::new();
    let move_by_bot = proposals
        .iter()
        .map(|(bot_id, from, to, wants)| (bot_id.clone(), (*from, *to, *wants)))
        .collect::<HashMap<_, _>>();
    for (a_id, (a_from, a_to, a_move)) in &move_by_bot {
        if !a_move || a_from == a_to {
            continue;
        }
        for (b_id, (b_from, b_to, b_move)) in &move_by_bot {
            if a_id >= b_id || !b_move {
                continue;
            }
            if *a_from == *b_to && *a_to == *b_from {
                swap_edges.insert((a_id.clone(), b_id.clone()));
            }
        }
    }

    for bot in &mut state.bots {
        let Some((from, to, wants_move)) = move_by_bot.get(&bot.id).copied() else {
            continue;
        };
        if !wants_move || from == to {
            continue;
        }
        let winner = winners.get(&to);
        let won_cell = winner.map(|id| id == &bot.id).unwrap_or(false);
        let swap_blocked = swap_edges.iter().any(|(a, b)| a == &bot.id || b == &bot.id);
        if won_cell && !swap_blocked {
            let (x, y) = map.xy(to);
            bot.x = x;
            bot.y = y;
        } else {
            outcome.blocked_moves = outcome.blocked_moves.saturating_add(1);
        }
    }

    // Apply pickups/dropoffs in bot-id order for deterministic resource conflicts.
    let mut bot_ids = state
        .bots
        .iter()
        .map(|bot| bot.id.clone())
        .collect::<Vec<_>>();
    bot_ids.sort();

    let mut consumed_items = HashSet::<String>::new();

    for bot_id in bot_ids {
        let Some(bot_index) = state.bots.iter().position(|bot| bot.id == bot_id) else {
            continue;
        };
        let action = action_by_bot
            .get(&bot_id)
            .cloned()
            .unwrap_or_else(|| Action::wait(bot_id.clone()));
        match action {
            Action::PickUp { item_id, .. } => {
                let (bot_x, bot_y, bot_capacity, bot_carrying_len) = {
                    let bot = &state.bots[bot_index];
                    (bot.x, bot.y, bot.capacity, bot.carrying.len())
                };
                if bot_carrying_len >= bot_capacity {
                    outcome.invalid_actions = outcome.invalid_actions.saturating_add(1);
                    continue;
                }
                if consumed_items.contains(&item_id) {
                    outcome.invalid_actions = outcome.invalid_actions.saturating_add(1);
                    continue;
                }
                let Some(item_index) = state.items.iter().position(|item| item.id == item_id)
                else {
                    outcome.invalid_actions = outcome.invalid_actions.saturating_add(1);
                    continue;
                };
                let adjacent = {
                    let item = &state.items[item_index];
                    (item.x - bot_x).unsigned_abs() + (item.y - bot_y).unsigned_abs() == 1
                };
                if !adjacent {
                    outcome.invalid_actions = outcome.invalid_actions.saturating_add(1);
                    continue;
                }
                let item_kind = state.items[item_index].kind.clone();
                consumed_items.insert(item_id.clone());
                state.items.remove(item_index);
                state.bots[bot_index].carrying.push(item_kind);
            }
            Action::DropOff { order_id, .. } => {
                let on_dropoff = {
                    let bot = &state.bots[bot_index];
                    state
                        .grid
                        .drop_off_tiles
                        .iter()
                        .any(|tile| tile[0] == bot.x && tile[1] == bot.y)
                };
                if !on_dropoff {
                    outcome.invalid_actions = outcome.invalid_actions.saturating_add(1);
                    continue;
                }
                let Some(order_idx) = state.orders.iter().position(|order| {
                    order.id == order_id && matches!(order.status, OrderStatus::InProgress)
                }) else {
                    outcome.invalid_actions = outcome.invalid_actions.saturating_add(1);
                    continue;
                };
                let kind = state.orders[order_idx].item_id.clone();
                let Some(carry_idx) = state.bots[bot_index]
                    .carrying
                    .iter()
                    .position(|held| held == &kind)
                else {
                    outcome.invalid_actions = outcome.invalid_actions.saturating_add(1);
                    continue;
                };

                state.bots[bot_index].carrying.remove(carry_idx);
                state.orders[order_idx].status = OrderStatus::Delivered;
                outcome.items_delivered_delta = outcome.items_delivered_delta.saturating_add(1);
            }
            Action::Move { .. } | Action::Wait { .. } => {}
        }
    }

    let active_before = state
        .orders
        .iter()
        .any(|order| matches!(order.status, OrderStatus::InProgress));
    if active_before {
        let active_after = state
            .orders
            .iter()
            .any(|order| matches!(order.status, OrderStatus::InProgress));
        if !active_after {
            outcome.orders_completed_delta = 1;
        }
    }

    outcome
}

pub fn dropoff_order_for_kind(state: &GameState, kind: &str) -> Option<String> {
    state
        .orders
        .iter()
        .find(|order| matches!(order.status, OrderStatus::InProgress) && order.item_id == kind)
        .map(|order| order.id.clone())
}

#[cfg(test)]
mod tests {
    use crate::model::{Action, BotState, GameState, Grid, Item, Order, OrderStatus};

    use super::apply_actions;

    #[test]
    fn pickup_requires_adjacency_and_capacity() {
        let mut state = GameState {
            grid: Grid {
                width: 5,
                height: 5,
                drop_off_tiles: vec![[0, 0]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 1,
                y: 1,
                carrying: vec![],
                capacity: 1,
            }],
            items: vec![Item {
                id: "i1".to_owned(),
                kind: "milk".to_owned(),
                x: 2,
                y: 1,
            }],
            orders: vec![],
            ..GameState::default()
        };
        let outcome = apply_actions(
            &mut state,
            &[Action::PickUp {
                bot_id: "0".to_owned(),
                item_id: "i1".to_owned(),
            }],
        );
        assert_eq!(outcome.invalid_actions, 0);
        assert_eq!(state.bots[0].carrying, vec!["milk".to_owned()]);
        assert!(state.items.is_empty());
    }

    #[test]
    fn dropoff_requires_in_progress_order_and_matching_item() {
        let mut state = GameState {
            grid: Grid {
                width: 5,
                height: 5,
                drop_off_tiles: vec![[0, 0]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 0,
                y: 0,
                carrying: vec!["milk".to_owned()],
                capacity: 3,
            }],
            orders: vec![Order {
                id: "o1".to_owned(),
                item_id: "milk".to_owned(),
                status: OrderStatus::InProgress,
            }],
            ..GameState::default()
        };
        let outcome = apply_actions(
            &mut state,
            &[Action::DropOff {
                bot_id: "0".to_owned(),
                order_id: "o1".to_owned(),
            }],
        );
        assert_eq!(outcome.items_delivered_delta, 1);
        assert!(matches!(state.orders[0].status, OrderStatus::Delivered));
    }

    #[test]
    fn collision_resolution_is_deterministic_by_bot_id() {
        let mut state = GameState {
            grid: Grid {
                width: 5,
                height: 5,
                drop_off_tiles: vec![[0, 0]],
                ..Grid::default()
            },
            bots: vec![
                BotState {
                    id: "0".to_owned(),
                    x: 1,
                    y: 1,
                    carrying: vec![],
                    capacity: 3,
                },
                BotState {
                    id: "1".to_owned(),
                    x: 3,
                    y: 1,
                    carrying: vec![],
                    capacity: 3,
                },
            ],
            ..GameState::default()
        };
        let actions = vec![
            Action::Move {
                bot_id: "0".to_owned(),
                dx: 1,
                dy: 0,
            },
            Action::Move {
                bot_id: "1".to_owned(),
                dx: -1,
                dy: 0,
            },
        ];

        let outcome = apply_actions(&mut state, &actions);
        let bot0 = state.bots.iter().find(|b| b.id == "0").expect("bot0");
        let bot1 = state.bots.iter().find(|b| b.id == "1").expect("bot1");
        assert_eq!((bot0.x, bot0.y), (2, 1), "lower id wins target cell");
        assert_eq!((bot1.x, bot1.y), (3, 1), "other bot remains in place");
        assert_eq!(outcome.blocked_moves, 1);
    }
}
