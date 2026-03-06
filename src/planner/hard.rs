use std::collections::{HashMap, HashSet};

use super::{
    common::{
        active_kind_counts, active_missing_total, bot_cell, center_corridor_penalty,
        find_adjacent_item_for_kind, nearest_dropoff_cell, on_dropoff, pick_best_item_target,
        preview_kind_counts, region_for_cell, try_dropoff_active,
    },
    Intent, PlanResult, TickContext,
};

#[derive(Debug, Clone)]
struct Claim {
    bot_id: String,
    tick: u64,
}

#[derive(Debug, Default)]
pub struct HardPlanner {
    runner_id: Option<String>,
    item_claims: HashMap<String, Claim>,
    prev_cell_by_bot: HashMap<String, u16>,
    blocked_by_bot: HashMap<String, u8>,
}

impl HardPlanner {
    pub fn tick(&mut self, input: TickContext<'_>) -> PlanResult {
        let mut plan = PlanResult::empty("hard_hierarchical");
        if input.state.bots.is_empty() {
            return plan;
        }

        self.update_blocked_state(input);
        self.prune_claims(input);

        let active_counts = active_kind_counts(input.state);
        let preview_counts = preview_kind_counts(input.state);
        let active_total = active_missing_total(input.state);
        let runner_id = self.pick_runner(input);

        let mut picker_bots = input
            .state
            .bots
            .iter()
            .filter(|bot| bot.id != runner_id)
            .collect::<Vec<_>>();
        picker_bots.sort_by(|a, b| a.id.cmp(&b.id));

        for (idx, bot) in picker_bots.iter().enumerate() {
            let region_id = (idx as u16) % 4;
            plan.role_label_by_bot
                .insert(bot.id.clone(), format!("hard_picker_r{region_id}"));
            plan.preferred_area_by_bot.insert(bot.id.clone(), region_id);
            plan.local_radius_by_bot.insert(bot.id.clone(), 10);
            plan.expansion_mode_by_bot.insert(bot.id.clone(), false);
        }

        plan.role_label_by_bot
            .insert(runner_id.clone(), "hard_runner".to_owned());
        plan.local_radius_by_bot.insert(runner_id.clone(), 8);

        for bot in &input.state.bots {
            if let Some(intent) = try_dropoff_active(input.state, input.map, bot) {
                plan.intents.insert(bot.id.clone(), intent);
            }
        }

        let mut used_items = HashSet::<String>::new();

        // Runner/stager focuses on preview near dropoff.
        if let Some(runner) = input.state.bots.iter().find(|bot| bot.id == runner_id) {
            if !plan.intents.contains_key(&runner.id) {
                assign_runner(
                    input,
                    runner,
                    &preview_counts,
                    active_total,
                    &mut plan,
                    &mut used_items,
                );
            }
        }

        // Regional pickers handle active kinds with claim table.
        for (idx, bot) in picker_bots.into_iter().enumerate() {
            if plan.intents.contains_key(&bot.id) {
                continue;
            }
            let region_id = (idx as u16) % 4;
            let Some(from) = bot_cell(input.map, bot) else {
                plan.intents.insert(bot.id.clone(), Intent::Wait);
                continue;
            };

            let mut chosen = None::<(String, String, u16, f64)>;
            for kind in active_counts.keys() {
                if let Some(item_id) = find_adjacent_item_for_kind(input.state, bot, kind) {
                    chosen = Some((kind.clone(), item_id.to_owned(), from, -1000.0));
                    break;
                }

                let target = pick_best_item_target(
                    input.state,
                    input.map,
                    input.dist,
                    from,
                    kind,
                    |target| {
                        if self.is_claimed_by_other(target.item_id, &bot.id, input.tick)
                            || used_items.contains(target.item_id)
                        {
                            return 1000.0;
                        }
                        let region = region_for_cell(input.map, target.stand_cell, 4);
                        let region_penalty = if region == region_id { 0.0 } else { 12.0 };
                        let congestion = if input.map.is_choke_point(target.stand_cell) {
                            4.0
                        } else {
                            0.0
                        };
                        center_corridor_penalty(input.map, target.stand_cell)
                            + region_penalty
                            + congestion
                    },
                );
                let Some(target) = target else {
                    continue;
                };
                let score = f64::from(input.dist.dist(from, target.stand_cell))
                    + f64::from(input.dist.dist_to_dropoff(target.stand_cell));
                match chosen {
                    Some((_, _, _, cur)) if score >= cur => {}
                    _ => {
                        chosen = Some((
                            kind.clone(),
                            target.item_id.to_owned(),
                            target.stand_cell,
                            score,
                        ));
                    }
                }
            }

            if chosen.is_none() && (active_total <= 2 || bot.carrying.len() + 1 < bot.capacity) {
                if let Some(kind) = preview_counts.keys().next() {
                    if let Some(item_id) = find_adjacent_item_for_kind(input.state, bot, kind) {
                        chosen = Some((kind.clone(), item_id.to_owned(), from, -500.0));
                    } else if let Some(target) = pick_best_item_target(
                        input.state,
                        input.map,
                        input.dist,
                        from,
                        kind,
                        |_| 0.0,
                    ) {
                        chosen = Some((
                            kind.clone(),
                            target.item_id.to_owned(),
                            target.stand_cell,
                            100.0,
                        ));
                    }
                }
            }

            if let Some((_, item_id, stand, _)) = chosen {
                used_items.insert(item_id.clone());
                self.item_claims.insert(
                    item_id.clone(),
                    Claim {
                        bot_id: bot.id.clone(),
                        tick: input.tick,
                    },
                );
                if input
                    .state
                    .items
                    .iter()
                    .find(|item| item.id == item_id)
                    .map(|item| find_adjacent_item_for_kind(input.state, bot, &item.kind).is_some())
                    .unwrap_or(false)
                {
                    plan.intents
                        .insert(bot.id.clone(), Intent::PickUp { item_id });
                } else {
                    plan.goal_cell_by_bot.insert(bot.id.clone(), stand);
                    plan.intents
                        .insert(bot.id.clone(), Intent::MoveTo { cell: stand });
                }
            } else {
                plan.intents.insert(bot.id.clone(), Intent::Wait);
            }
        }

        // Priority by slack: lowest slack first.
        let mut urgency = input
            .state
            .bots
            .iter()
            .map(|bot| {
                let eta = estimate_eta(bot, &plan, input);
                let deadline_proxy = (2 * active_total + 6) as i32;
                let slack = deadline_proxy - eta;
                (bot.id.clone(), slack)
            })
            .collect::<Vec<_>>();
        urgency.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        plan.explicit_priority = urgency.into_iter().map(|(id, _)| id).collect();

        plan
    }

    fn pick_runner(&mut self, input: TickContext<'_>) -> String {
        let mut choices = input
            .state
            .bots
            .iter()
            .filter_map(|bot| {
                let from = bot_cell(input.map, bot)?;
                let drop = nearest_dropoff_cell(input.map, input.dist, from)?;
                Some((bot.id.clone(), input.dist.dist(from, drop)))
            })
            .collect::<Vec<_>>();
        choices.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        if let Some(existing) = &self.runner_id {
            if let Some((_, best)) = choices.first() {
                if let Some((_, current)) = choices.iter().find(|(id, _)| id == existing) {
                    if *current <= best.saturating_add(2) {
                        return existing.clone();
                    }
                }
            }
        }

        let picked = choices
            .first()
            .map(|(id, _)| id.clone())
            .unwrap_or_else(|| input.state.bots[0].id.clone());
        self.runner_id = Some(picked.clone());
        picked
    }

    fn is_claimed_by_other(&self, item_id: &str, bot_id: &str, tick: u64) -> bool {
        self.item_claims
            .get(item_id)
            .map(|claim| claim.bot_id != bot_id && tick.saturating_sub(claim.tick) <= 5)
            .unwrap_or(false)
    }

    fn update_blocked_state(&mut self, input: TickContext<'_>) {
        for bot in &input.state.bots {
            if let Some(cell) = bot_cell(input.map, bot) {
                let prev = self.prev_cell_by_bot.insert(bot.id.clone(), cell);
                let blocked_entry = self.blocked_by_bot.entry(bot.id.clone()).or_insert(0);
                if prev == Some(cell) {
                    *blocked_entry = blocked_entry.saturating_add(1);
                } else {
                    *blocked_entry = 0;
                }
            }
        }
    }

    fn prune_claims(&mut self, input: TickContext<'_>) {
        let mut release_bots = HashSet::<String>::new();
        for (bot_id, blocked) in &self.blocked_by_bot {
            if *blocked >= 3 {
                release_bots.insert(bot_id.clone());
            }
        }
        let live_items = input
            .state
            .items
            .iter()
            .map(|item| item.id.clone())
            .collect::<HashSet<_>>();

        self.item_claims.retain(|item_id, claim| {
            live_items.contains(item_id)
                && input.tick.saturating_sub(claim.tick) <= 10
                && !release_bots.contains(&claim.bot_id)
        });
    }
}

fn assign_runner(
    input: TickContext<'_>,
    runner: &crate::model::BotState,
    preview_counts: &HashMap<String, u16>,
    active_total: usize,
    plan: &mut PlanResult,
    used_items: &mut HashSet<String>,
) {
    if preview_counts.is_empty() {
        plan.intents.insert(runner.id.clone(), Intent::Wait);
        return;
    }

    if on_dropoff(input.map, runner)
        && runner
            .carrying
            .iter()
            .any(|kind| preview_counts.contains_key(kind))
    {
        plan.intents.insert(runner.id.clone(), Intent::Wait);
        return;
    }

    let Some(from) = bot_cell(input.map, runner) else {
        plan.intents.insert(runner.id.clone(), Intent::Wait);
        return;
    };

    let preview_kind = preview_counts.keys().next().cloned().unwrap_or_default();
    if runner.carrying.len() < runner.capacity {
        if let Some(item_id) = find_adjacent_item_for_kind(input.state, runner, &preview_kind) {
            used_items.insert(item_id.to_owned());
            plan.intents.insert(
                runner.id.clone(),
                Intent::PickUp {
                    item_id: item_id.to_owned(),
                },
            );
            return;
        }
    }

    if let Some(target) = pick_best_item_target(
        input.state,
        input.map,
        input.dist,
        from,
        &preview_kind,
        |target| {
            let near_drop = input.dist.dist_to_dropoff(target.stand_cell);
            f64::from(near_drop)
                + if used_items.contains(target.item_id) {
                    30.0
                } else {
                    0.0
                }
        },
    ) {
        if active_total <= 2 || input.dist.dist_to_dropoff(from) <= 4 {
            used_items.insert(target.item_id.to_owned());
            plan.goal_cell_by_bot
                .insert(runner.id.clone(), target.stand_cell);
            plan.intents.insert(
                runner.id.clone(),
                Intent::MoveTo {
                    cell: target.stand_cell,
                },
            );
            return;
        }
    }

    if let Some(drop) = nearest_dropoff_cell(input.map, input.dist, from) {
        plan.goal_cell_by_bot.insert(runner.id.clone(), drop);
        plan.intents
            .insert(runner.id.clone(), Intent::MoveTo { cell: drop });
    } else {
        plan.intents.insert(runner.id.clone(), Intent::Wait);
    }
}

fn estimate_eta(bot: &crate::model::BotState, plan: &PlanResult, input: TickContext<'_>) -> i32 {
    let Some(from) = bot_cell(input.map, bot) else {
        return 999;
    };
    match plan.intents.get(&bot.id) {
        Some(Intent::DropOff { .. }) => 0,
        Some(Intent::PickUp { .. }) => i32::from(input.dist.dist_to_dropoff(from)),
        Some(Intent::MoveTo { cell }) => {
            let a = input.dist.dist(from, *cell);
            let b = input.dist.dist_to_dropoff(*cell);
            if a == u16::MAX || b == u16::MAX {
                999
            } else {
                i32::from(a) + i32::from(b)
            }
        }
        _ => i32::from(input.dist.dist_to_dropoff(from)),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        dist::DistanceMap,
        model::{BotState, GameState, Grid, Item, Order, OrderStatus},
        world::World,
    };

    use super::HardPlanner;
    use crate::planner::{Intent, TickContext};

    fn hard_state() -> GameState {
        GameState {
            grid: Grid {
                width: 22,
                height: 14,
                drop_off_tiles: vec![[1, 12]],
                ..Grid::default()
            },
            bots: (0..5)
                .map(|idx| BotState {
                    id: idx.to_string(),
                    x: 2 + idx,
                    y: 10,
                    carrying: vec![],
                    capacity: 3,
                })
                .collect(),
            items: vec![
                Item {
                    id: "milk_1".to_owned(),
                    kind: "milk".to_owned(),
                    x: 4,
                    y: 7,
                },
                Item {
                    id: "milk_2".to_owned(),
                    kind: "milk".to_owned(),
                    x: 17,
                    y: 6,
                },
                Item {
                    id: "bread_1".to_owned(),
                    kind: "bread".to_owned(),
                    x: 8,
                    y: 3,
                },
                Item {
                    id: "egg_1".to_owned(),
                    kind: "egg".to_owned(),
                    x: 19,
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
                    id: "a2".to_owned(),
                    item_id: "bread".to_owned(),
                    status: OrderStatus::InProgress,
                },
                Order {
                    id: "p1".to_owned(),
                    item_id: "egg".to_owned(),
                    status: OrderStatus::Pending,
                },
            ],
            ..GameState::default()
        }
    }

    #[test]
    fn assigns_runner_and_regions() {
        let state = hard_state();
        let world = World::new(&state);
        let map = world.map();
        let dist = DistanceMap::build(map);

        let mut planner = HardPlanner::default();
        let plan = planner.tick(TickContext {
            state: &state,
            map,
            dist: &dist,
            tick: 0,
        });

        assert!(plan
            .role_label_by_bot
            .values()
            .any(|label| label == "hard_runner"));
        assert!(plan
            .role_label_by_bot
            .values()
            .any(|label| label.starts_with("hard_picker_r")));
    }

    #[test]
    fn duplicate_claim_control_keeps_unique_targets() {
        let state = hard_state();
        let world = World::new(&state);
        let map = world.map();
        let dist = DistanceMap::build(map);

        let mut planner = HardPlanner::default();
        let plan = planner.tick(TickContext {
            state: &state,
            map,
            dist: &dist,
            tick: 1,
        });

        let picked_items = plan
            .intents
            .values()
            .filter_map(|intent| match intent {
                Intent::PickUp { item_id } => Some(item_id.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let unique = picked_items
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(picked_items.len(), unique.len());
    }
}
