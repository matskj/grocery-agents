use std::collections::{HashMap, HashSet};

use super::{
    common::{
        active_kind_counts, active_missing_total, bot_cell, center_corridor_penalty,
        find_adjacent_item_for_kind, move_to_nearest_dropoff_or_wait, on_dropoff,
        pick_best_item_target, preview_kind_counts, region_for_cell, set_move, set_pickup,
        set_wait, try_dropoff_active,
    },
    Intent, PlanResult, TickContext,
};

#[derive(Debug, Default)]
pub struct ExpertPlanner {
    history_by_kind: HashMap<String, u32>,
    prev_active_order_index: Option<i64>,
    prev_active_kinds: Vec<String>,
}

impl ExpertPlanner {
    pub fn tick(&mut self, input: TickContext<'_>) -> PlanResult {
        self.update_history(input);

        let mut plan = PlanResult::empty("expert_swarm");
        if input.state.bots.is_empty() {
            return plan;
        }

        let active_counts = active_kind_counts(input.state);
        let preview_counts = preview_kind_counts(input.state);
        let active_total = active_missing_total(input.state);

        let mut bots = input.state.bots.iter().collect::<Vec<_>>();
        bots.sort_by(|a, b| a.id.cmp(&b.id));

        let picker_n = 5usize.min(bots.len());
        let runner_n = 3usize.min(bots.len().saturating_sub(picker_n));
        let buffer_n = 2usize.min(bots.len().saturating_sub(picker_n + runner_n));

        let picker_ids = bots
            .iter()
            .take(picker_n)
            .map(|bot| bot.id.clone())
            .collect::<Vec<_>>();
        let runner_ids = bots
            .iter()
            .skip(picker_n)
            .take(runner_n)
            .map(|bot| bot.id.clone())
            .collect::<Vec<_>>();
        let buffer_ids = bots
            .iter()
            .skip(picker_n + runner_n)
            .take(buffer_n)
            .map(|bot| bot.id.clone())
            .collect::<Vec<_>>();

        for (idx, bot_id) in picker_ids.iter().enumerate() {
            let area = (idx as u16) % 5;
            plan.role_label_by_bot
                .insert(bot_id.clone(), format!("expert_picker_r{area}"));
            plan.preferred_area_by_bot.insert(bot_id.clone(), area);
            plan.local_radius_by_bot.insert(bot_id.clone(), 11);
            plan.expansion_mode_by_bot.insert(bot_id.clone(), false);
        }
        for bot_id in &runner_ids {
            plan.role_label_by_bot
                .insert(bot_id.clone(), "expert_runner".to_owned());
            plan.local_radius_by_bot.insert(bot_id.clone(), 9);
            plan.expansion_mode_by_bot.insert(bot_id.clone(), false);
        }
        for bot_id in &buffer_ids {
            plan.role_label_by_bot
                .insert(bot_id.clone(), "expert_buffer".to_owned());
            plan.local_radius_by_bot.insert(bot_id.clone(), 7);
            plan.expansion_mode_by_bot.insert(bot_id.clone(), false);
        }

        for bot in &bots {
            if let Some(intent) = try_dropoff_active(input.state, input.map, bot) {
                plan.intents.insert(bot.id.clone(), intent);
            }
        }

        let mut remaining_active = active_counts.clone();
        let mut used_items = HashSet::<String>::new();

        // Buffers keep preview inventory staged near dropoff.
        let staged_preview = bots
            .iter()
            .filter(|bot| on_dropoff(input.map, bot))
            .map(|bot| {
                bot.carrying
                    .iter()
                    .filter(|kind| preview_counts.contains_key(kind.as_str()))
                    .count() as u32
            })
            .sum::<u32>();
        let mut preview_target = 2u32.saturating_sub(staged_preview);

        for bot in &bots {
            if plan.intents.contains_key(&bot.id) {
                continue;
            }

            let is_runner = runner_ids.contains(&bot.id);
            let is_buffer = buffer_ids.contains(&bot.id);
            let is_picker = picker_ids.contains(&bot.id);

            let Some(from) = bot_cell(input.map, bot) else {
                set_wait(&mut plan, &bot.id);
                continue;
            };

            let mut candidates = Vec::<(f64, String, String, u16)>::new();

            for (kind, count) in &remaining_active {
                if *count == 0 {
                    continue;
                }
                if let Some(item_id) = find_adjacent_item_for_kind(input.state, bot, kind) {
                    candidates.push((500.0, kind.clone(), item_id.to_owned(), from));
                    continue;
                }
                let role_penalty = if is_picker {
                    // Keep pickers in their regions when possible.
                    let area = plan
                        .preferred_area_by_bot
                        .get(&bot.id)
                        .copied()
                        .unwrap_or(0);
                    area as f64
                } else {
                    0.0
                };
                if let Some(target) = pick_best_item_target(
                    input.state,
                    input.map,
                    input.dist,
                    from,
                    kind,
                    |target| {
                        let mut penalty = center_corridor_penalty(input.map, target.stand_cell);
                        penalty += if used_items.contains(target.item_id) {
                            20.0
                        } else {
                            0.0
                        };
                        if is_picker {
                            let expected = plan
                                .preferred_area_by_bot
                                .get(&bot.id)
                                .copied()
                                .unwrap_or(0);
                            let region = region_for_cell(input.map, target.stand_cell, 5);
                            if region != expected {
                                penalty += 8.0;
                            }
                        }
                        penalty + role_penalty
                    },
                ) {
                    let eta = f64::from(input.dist.dist(from, target.stand_cell))
                        + f64::from(input.dist.dist_to_dropoff(target.stand_cell));
                    let gain = if is_runner { 180.0 } else { 140.0 };
                    candidates.push((
                        gain - eta,
                        kind.clone(),
                        target.item_id.to_owned(),
                        target.stand_cell,
                    ));
                }
            }

            if !preview_counts.is_empty() && (is_buffer || active_total <= runner_n.max(1)) {
                let preview_kind = preview_counts.keys().next().cloned().unwrap_or_default();
                if preview_target > 0 || is_buffer {
                    if let Some(item_id) =
                        find_adjacent_item_for_kind(input.state, bot, &preview_kind)
                    {
                        candidates.push((120.0, preview_kind.clone(), item_id.to_owned(), from));
                    } else if let Some(target) = pick_best_item_target(
                        input.state,
                        input.map,
                        input.dist,
                        from,
                        &preview_kind,
                        |_| 0.0,
                    ) {
                        let eta = f64::from(input.dist.dist(from, target.stand_cell));
                        candidates.push((
                            110.0 - eta,
                            preview_kind.clone(),
                            target.item_id.to_owned(),
                            target.stand_cell,
                        ));
                    }
                }
            }

            if candidates.is_empty() && is_buffer {
                for (kind, freq) in self.top_history_kinds(2) {
                    if let Some(item_id) = find_adjacent_item_for_kind(input.state, bot, &kind) {
                        candidates.push((
                            40.0 + f64::from(freq),
                            kind.clone(),
                            item_id.to_owned(),
                            from,
                        ));
                    } else if let Some(target) = pick_best_item_target(
                        input.state,
                        input.map,
                        input.dist,
                        from,
                        &kind,
                        |_| 0.0,
                    ) {
                        candidates.push((
                            35.0 + f64::from(freq),
                            kind.clone(),
                            target.item_id.to_owned(),
                            target.stand_cell,
                        ));
                    }
                }
            }

            candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

            if let Some((_, kind, item_id, stand)) = candidates
                .into_iter()
                .find(|(_, _, item_id, _)| !used_items.contains(item_id))
            {
                used_items.insert(item_id.clone());
                if stand == from {
                    set_pickup(&mut plan, &bot.id, &item_id);
                } else {
                    set_move(&mut plan, &bot.id, stand);
                }
                if let Some(entry) = remaining_active.get_mut(&kind) {
                    *entry = entry.saturating_sub(1);
                }
                if preview_counts.contains_key(&kind) && preview_target > 0 {
                    preview_target = preview_target.saturating_sub(1);
                }
            } else if is_buffer && on_dropoff(input.map, bot) {
                set_wait(&mut plan, &bot.id);
            } else {
                move_to_nearest_dropoff_or_wait(input, &mut plan, bot);
            }
        }

        // Urgent ordering: runners and finishers first.
        let mut priority = Vec::<(String, u16, u8)>::new();
        for bot in &bots {
            let Some(from) = bot_cell(input.map, bot) else {
                continue;
            };
            let eta = match plan.intents.get(&bot.id) {
                Some(Intent::MoveTo { cell }) => input.dist.dist(from, *cell),
                Some(Intent::PickUp { .. }) => 1,
                Some(Intent::DropOff { .. }) => 0,
                _ => input.dist.dist_to_dropoff(from),
            };
            let role_rank = if runner_ids.contains(&bot.id) {
                0
            } else if buffer_ids.contains(&bot.id) {
                2
            } else {
                1
            };
            priority.push((bot.id.clone(), eta, role_rank));
        }
        priority.sort_by(|a, b| {
            a.2.cmp(&b.2)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.0.cmp(&b.0))
        });
        plan.explicit_priority = priority.into_iter().map(|(id, _, _)| id).collect();

        plan
    }

    fn update_history(&mut self, input: TickContext<'_>) {
        if let Some(prev_idx) = self.prev_active_order_index {
            if input.state.active_order_index > prev_idx {
                for kind in &self.prev_active_kinds {
                    *self.history_by_kind.entry(kind.clone()).or_insert(0) += 1;
                }
            }
        }
        self.prev_active_order_index = Some(input.state.active_order_index);
        self.prev_active_kinds = active_kind_counts(input.state)
            .into_iter()
            .flat_map(|(kind, count)| std::iter::repeat(kind).take(count as usize))
            .collect();
    }

    fn top_history_kinds(&self, n: usize) -> Vec<(String, u32)> {
        let mut entries = self
            .history_by_kind
            .iter()
            .map(|(kind, freq)| (kind.clone(), *freq))
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        entries.truncate(n);
        entries
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        dist::DistanceMap,
        model::{BotState, GameState, Grid, Item, Order, OrderStatus},
        world::World,
    };

    use super::ExpertPlanner;
    use crate::planner::TickContext;

    fn expert_state() -> GameState {
        GameState {
            grid: Grid {
                width: 28,
                height: 18,
                drop_off_tiles: vec![[1, 16]],
                ..Grid::default()
            },
            bots: (0..10)
                .map(|idx| BotState {
                    id: idx.to_string(),
                    x: 3 + (idx % 5),
                    y: 14 - (idx / 5),
                    carrying: vec![],
                    capacity: 3,
                })
                .collect(),
            items: vec![
                Item {
                    id: "milk_1".to_owned(),
                    kind: "milk".to_owned(),
                    x: 6,
                    y: 8,
                },
                Item {
                    id: "bread_1".to_owned(),
                    kind: "bread".to_owned(),
                    x: 14,
                    y: 6,
                },
                Item {
                    id: "egg_1".to_owned(),
                    kind: "egg".to_owned(),
                    x: 20,
                    y: 5,
                },
                Item {
                    id: "juice_1".to_owned(),
                    kind: "juice".to_owned(),
                    x: 24,
                    y: 8,
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
    fn role_counts_match_expected_partition() {
        let state = expert_state();
        let world = World::new(&state);
        let map = world.map();
        let dist = DistanceMap::build(map);
        let mut planner = ExpertPlanner::default();
        let plan = planner.tick(TickContext {
            state: &state,
            map,
            dist: &dist,
            tick: 0,
        });

        let pickers = plan
            .role_label_by_bot
            .values()
            .filter(|label| label.starts_with("expert_picker_"))
            .count();
        let runners = plan
            .role_label_by_bot
            .values()
            .filter(|label| *label == "expert_runner")
            .count();
        let buffers = plan
            .role_label_by_bot
            .values()
            .filter(|label| *label == "expert_buffer")
            .count();

        assert_eq!(pickers, 5);
        assert_eq!(runners, 3);
        assert_eq!(buffers, 2);
    }
}
