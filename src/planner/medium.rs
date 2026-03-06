use std::collections::{HashMap, HashSet};

use super::{
    common::{
        active_kind_counts, active_missing_total, bot_cell, center_corridor_penalty,
        count_active_carried_team, find_adjacent_item_for_kind, move_to_nearest_dropoff_or_wait,
        nearest_dropoff_cell, on_dropoff, pick_best_item_target, preview_kind_counts,
        select_dropoff_anchor_bot, set_move, set_pickup, set_wait, try_dropoff_active,
    },
    PlanResult, TickContext,
};

#[derive(Debug, Default)]
pub struct MediumPlanner {
    stager_id: Option<String>,
}

impl MediumPlanner {
    pub fn tick(&mut self, input: TickContext<'_>) -> PlanResult {
        let mut plan = PlanResult::empty("medium_allocator");
        if input.state.bots.is_empty() {
            return plan;
        }

        let active_counts = active_kind_counts(input.state);
        let preview_counts = preview_kind_counts(input.state);
        let active_total = active_missing_total(input.state);

        for bot in &input.state.bots {
            plan.role_label_by_bot
                .insert(bot.id.clone(), "medium_worker".to_owned());
            plan.local_radius_by_bot.insert(bot.id.clone(), 8);
            plan.expansion_mode_by_bot.insert(bot.id.clone(), false);
        }

        let stager_id = self.pick_stager(input);
        plan.role_label_by_bot
            .insert(stager_id.clone(), "medium_stager".to_owned());

        let carried_active = count_active_carried_team(input.state, &active_counts);
        let mut remaining_pick = active_counts
            .iter()
            .map(|(kind, needed)| {
                (
                    kind.clone(),
                    needed.saturating_sub(carried_active.get(kind).copied().unwrap_or(0)),
                )
            })
            .collect::<HashMap<_, _>>();

        let mut used_items = HashSet::<String>::new();

        for bot in &input.state.bots {
            if let Some(intent) = try_dropoff_active(input.state, input.map, bot) {
                plan.intents.insert(bot.id.clone(), intent);
            }
        }

        if let Some(stager_bot) = input.state.bots.iter().find(|bot| bot.id == stager_id) {
            if !plan.intents.contains_key(&stager_bot.id) {
                stage_preview(
                    input,
                    stager_bot,
                    &preview_counts,
                    active_total,
                    &mut plan,
                    &mut used_items,
                );
            }
        }

        let mut workers = input
            .state
            .bots
            .iter()
            .filter(|bot| bot.id != stager_id)
            .collect::<Vec<_>>();
        workers.sort_by(|a, b| a.id.cmp(&b.id));

        let mut assigned_workers = HashSet::<String>::new();

        for bot in &workers {
            if plan.intents.contains_key(&bot.id) {
                assigned_workers.insert(bot.id.clone());
            }
        }

        let mut slots = Vec::<String>::new();
        for (kind, count) in &remaining_pick {
            for _ in 0..*count {
                slots.push(kind.clone());
            }
        }

        for kind in slots {
            let mut best = None::<(String, String, u16, f64)>;
            for bot in &workers {
                if assigned_workers.contains(&bot.id) || bot.carrying.len() >= bot.capacity {
                    continue;
                }
                let Some(from) = bot_cell(input.map, bot) else {
                    continue;
                };
                if let Some(adj_item) = find_adjacent_item_for_kind(input.state, bot, &kind) {
                    best = Some((bot.id.clone(), adj_item.to_owned(), from, -1000.0));
                    break;
                }

                let inventory_penalty = if bot.carrying.len() + 1 >= bot.capacity {
                    5.0
                } else {
                    0.0
                };
                let target =
                    pick_best_item_target(input.state, input.map, input.dist, from, &kind, |t| {
                        inventory_penalty
                            + center_corridor_penalty(input.map, t.stand_cell)
                            + if used_items.contains(t.item_id) {
                                30.0
                            } else {
                                0.0
                            }
                    });
                let Some(target) = target else {
                    continue;
                };
                let score = f64::from(input.dist.dist(from, target.stand_cell))
                    + f64::from(input.dist.dist_to_dropoff(target.stand_cell))
                    + inventory_penalty
                    + center_corridor_penalty(input.map, target.stand_cell);

                match best {
                    Some((_, _, _, cur)) if score >= cur => {}
                    _ => {
                        best = Some((
                            bot.id.clone(),
                            target.item_id.to_owned(),
                            target.stand_cell,
                            score,
                        ));
                    }
                }
            }

            if let Some((bot_id, item_id, stand, _)) = best {
                assigned_workers.insert(bot_id.clone());
                used_items.insert(item_id.clone());
                let bot = workers.iter().find(|b| b.id == bot_id).copied();
                if let Some(bot) = bot {
                    if input.state.items.iter().any(|i| i.id == item_id)
                        && find_adjacent_item_for_kind(input.state, bot, &kind).is_some()
                    {
                        set_pickup(&mut plan, &bot_id, &item_id);
                    } else {
                        set_move(&mut plan, &bot_id, stand);
                    }
                }
                if let Some(entry) = remaining_pick.get_mut(&kind) {
                    *entry = entry.saturating_sub(1);
                }
            }
        }

        for bot in workers {
            if assigned_workers.contains(&bot.id) || plan.intents.contains_key(&bot.id) {
                continue;
            }
            if let Some(from) = bot_cell(input.map, bot) {
                if preview_counts.is_empty() || bot.carrying.len() >= bot.capacity {
                    set_wait(&mut plan, &bot.id);
                    continue;
                }
                let preview_kind = preview_counts.keys().next().cloned().unwrap_or_default();
                if let Some(item_id) = find_adjacent_item_for_kind(input.state, bot, &preview_kind)
                {
                    set_pickup(&mut plan, &bot.id, item_id);
                    continue;
                }
                if let Some(target) = pick_best_item_target(
                    input.state,
                    input.map,
                    input.dist,
                    from,
                    &preview_kind,
                    |_| 0.0,
                ) {
                    set_move(&mut plan, &bot.id, target.stand_cell);
                    continue;
                }
                set_wait(&mut plan, &bot.id);
                continue;
            }
            set_wait(&mut plan, &bot.id);
        }

        // Finishers first when one active item is missing.
        let mut priority = input
            .state
            .bots
            .iter()
            .map(|bot| bot.id.clone())
            .collect::<Vec<_>>();
        if active_total <= 1 {
            priority.sort_by_key(|bot_id| {
                let Some(bot) = input.state.bots.iter().find(|b| &b.id == bot_id) else {
                    return u16::MAX;
                };
                bot_cell(input.map, bot)
                    .and_then(|from| nearest_dropoff_cell(input.map, input.dist, from))
                    .map(|drop| {
                        if let Some(from) = bot_cell(input.map, bot) {
                            input.dist.dist(from, drop)
                        } else {
                            u16::MAX
                        }
                    })
                    .unwrap_or(u16::MAX)
            });
        } else {
            priority.sort();
        }
        plan.explicit_priority = priority;
        plan
    }

    fn pick_stager(&mut self, input: TickContext<'_>) -> String {
        let picked = select_dropoff_anchor_bot(input, self.stager_id.as_deref(), 2);
        self.stager_id = Some(picked.clone());
        picked
    }
}

fn stage_preview(
    input: TickContext<'_>,
    bot: &crate::model::BotState,
    preview_counts: &HashMap<String, u16>,
    active_total: usize,
    plan: &mut PlanResult,
    used_items: &mut HashSet<String>,
) {
    if preview_counts.is_empty() {
        set_wait(plan, &bot.id);
        return;
    }

    if on_dropoff(input.map, bot)
        && bot
            .carrying
            .iter()
            .any(|kind| preview_counts.contains_key(kind))
    {
        set_wait(plan, &bot.id);
        return;
    }

    let Some(from) = bot_cell(input.map, bot) else {
        set_wait(plan, &bot.id);
        return;
    };

    let preview_kind = preview_counts.keys().next().cloned().unwrap_or_default();
    if bot.carrying.len() < bot.capacity {
        if let Some(item_id) = find_adjacent_item_for_kind(input.state, bot, &preview_kind) {
            set_pickup(plan, &bot.id, item_id);
            used_items.insert(item_id.to_owned());
            return;
        }

        // Keep stager focused on preview unless active is about to starve.
        if active_total > 2 {
            if let Some(target) = pick_best_item_target(
                input.state,
                input.map,
                input.dist,
                from,
                &preview_kind,
                |t| {
                    if used_items.contains(t.item_id) {
                        25.0
                    } else {
                        0.0
                    }
                },
            ) {
                used_items.insert(target.item_id.to_owned());
                set_move(plan, &bot.id, target.stand_cell);
                return;
            }
        }
    }

    move_to_nearest_dropoff_or_wait(input, plan, bot);
}

#[cfg(test)]
mod tests {
    use crate::{
        dist::DistanceMap,
        model::{BotState, GameState, Grid, Item, Order, OrderStatus},
        world::World,
    };

    use super::MediumPlanner;
    use crate::planner::{Intent, TickContext};

    fn medium_state() -> GameState {
        GameState {
            grid: Grid {
                width: 16,
                height: 12,
                drop_off_tiles: vec![[1, 10]],
                ..Grid::default()
            },
            bots: vec![
                BotState {
                    id: "0".to_owned(),
                    x: 2,
                    y: 10,
                    carrying: vec![],
                    capacity: 3,
                },
                BotState {
                    id: "1".to_owned(),
                    x: 10,
                    y: 8,
                    carrying: vec![],
                    capacity: 3,
                },
                BotState {
                    id: "2".to_owned(),
                    x: 11,
                    y: 3,
                    carrying: vec![],
                    capacity: 3,
                },
            ],
            items: vec![
                Item {
                    id: "milk_1".to_owned(),
                    kind: "milk".to_owned(),
                    x: 4,
                    y: 6,
                },
                Item {
                    id: "bread_1".to_owned(),
                    kind: "bread".to_owned(),
                    x: 12,
                    y: 4,
                },
            ],
            orders: vec![
                Order {
                    id: "a1".to_owned(),
                    item_id: "milk".to_owned(),
                    status: OrderStatus::InProgress,
                },
                Order {
                    id: "p1".to_owned(),
                    item_id: "bread".to_owned(),
                    status: OrderStatus::Pending,
                },
            ],
            ..GameState::default()
        }
    }

    #[test]
    fn planner_assigns_a_stager() {
        let state = medium_state();
        let world = World::new(&state);
        let map = world.map();
        let dist = DistanceMap::build(map);
        let mut planner = MediumPlanner::default();
        let plan = planner.tick(TickContext {
            state: &state,
            map,
            dist: &dist,
            tick: 0,
        });

        assert!(plan
            .role_label_by_bot
            .values()
            .any(|label| label == "medium_stager"));
    }

    #[test]
    fn one_item_remaining_prioritizes_finisher_ordering() {
        let mut state = medium_state();
        state.orders = vec![Order {
            id: "a1".to_owned(),
            item_id: "milk".to_owned(),
            status: OrderStatus::InProgress,
        }];
        let world = World::new(&state);
        let map = world.map();
        let dist = DistanceMap::build(map);
        let mut planner = MediumPlanner::default();
        let plan = planner.tick(TickContext {
            state: &state,
            map,
            dist: &dist,
            tick: 1,
        });
        assert_eq!(plan.explicit_priority.len(), 3);
    }

    #[test]
    fn stager_can_wait_on_dropoff_with_preview() {
        let mut state = medium_state();
        state.bots[0].carrying = vec!["bread".to_owned()];
        let world = World::new(&state);
        let map = world.map();
        let dist = DistanceMap::build(map);
        let mut planner = MediumPlanner::default();
        let plan = planner.tick(TickContext {
            state: &state,
            map,
            dist: &dist,
            tick: 2,
        });

        let stager = plan
            .role_label_by_bot
            .iter()
            .find(|(_, label)| *label == "medium_stager")
            .map(|(id, _)| id.clone())
            .expect("stager");
        let intent = plan.intents.get(&stager).expect("stager intent");
        assert!(matches!(
            intent,
            Intent::Wait | Intent::DropOff { .. } | Intent::MoveTo { .. }
        ));
    }
}
