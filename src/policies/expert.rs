use std::collections::HashMap;

use crate::dispatcher::Intent;

use super::{
    common::{
        active_kind_counts, bot_cell, choose_scored_item_of_kind, first_active_order_for_kind,
        is_adjacent_to_item, on_dropoff, preview_kind_counts,
    },
    Strategy, StrategyPlan, TickInput,
};

#[derive(Debug, Default)]
pub struct ExpertStrategy {
    imbalance_streak: u16,
    rotate_offset: usize,
}

impl Strategy for ExpertStrategy {
    fn tick(&mut self, input: TickInput<'_>) -> StrategyPlan {
        let mut plan = StrategyPlan {
            policy_name: "expert_roles_buffers",
            strategy_stage: "role_pipeline",
            ..StrategyPlan::default()
        };
        if input.state.bots.is_empty() {
            return plan;
        }

        let active_counts = active_kind_counts(input.state);
        let preview_counts = preview_kind_counts(input.state);
        let active_gap = active_counts.values().copied().map(u32::from).sum::<u32>();

        let mut bots = input.state.bots.iter().collect::<Vec<_>>();
        bots.sort_by(|a, b| a.id.cmp(&b.id));
        if !bots.is_empty() {
            let shift = self.rotate_offset % bots.len();
            bots.rotate_left(shift);
        }

        let pickers = 5usize.min(bots.len());
        let runners = 3usize.min(bots.len().saturating_sub(pickers));
        let buffers = 2usize.min(bots.len().saturating_sub(pickers + runners));
        let mut role_counts = HashMap::<&str, usize>::new();

        for (idx, bot) in bots.iter().enumerate() {
            if idx < pickers {
                let area_id = (idx as u16) % 5;
                plan.role_label_by_bot
                    .insert(bot.id.clone(), format!("expert_picker_r{area_id}"));
                *role_counts.entry("picker").or_insert(0) += 1;
                plan.preferred_area_by_bot.insert(bot.id.clone(), area_id);
                plan.expansion_mode_by_bot.insert(bot.id.clone(), false);
            } else if idx < pickers + runners {
                plan.role_label_by_bot
                    .insert(bot.id.clone(), "expert_runner".to_owned());
                *role_counts.entry("runner").or_insert(0) += 1;
                plan.explicit_order.push(bot.id.clone());
            } else if idx < pickers + runners + buffers {
                plan.role_label_by_bot
                    .insert(bot.id.clone(), "expert_buffer".to_owned());
                *role_counts.entry("buffer").or_insert(0) += 1;
                plan.explicit_order.push(bot.id.clone());
            } else {
                plan.role_label_by_bot
                    .insert(bot.id.clone(), "expert_flex".to_owned());
                *role_counts.entry("flex").or_insert(0) += 1;
            }
        }

        let expected = [
            ("picker", pickers),
            ("runner", runners),
            ("buffer", buffers),
        ];
        let imbalanced = expected
            .iter()
            .any(|(name, cnt)| role_counts.get(name).copied().unwrap_or(0) != *cnt);
        if imbalanced {
            self.imbalance_streak = self.imbalance_streak.saturating_add(1);
            if self.imbalance_streak >= 20 && !input.state.bots.is_empty() {
                self.rotate_offset = (self.rotate_offset + 1) % input.state.bots.len();
                self.imbalance_streak = 0;
            }
        } else {
            self.imbalance_streak = 0;
        }

        // Buffers prefetch preview kinds and stage near dropoff, but never at the expense of active completion.
        let mut staged_preview_items = 0u32;
        for bot in &bots {
            if !on_dropoff(bot, input.map) {
                continue;
            }
            staged_preview_items = staged_preview_items.saturating_add(
                bot.carrying
                    .iter()
                    .filter(|kind| preview_counts.contains_key(kind.as_str()))
                    .count() as u32,
            );
        }
        let buffer_target = 2u32;
        let mut preview_need = buffer_target.saturating_sub(staged_preview_items);
        if !preview_counts.is_empty() && active_gap <= runners as u32 {
            for bot in &bots[pickers + runners..pickers + runners + buffers] {
                if on_dropoff(bot, input.map) {
                    let has_preview_only = bot.carrying.iter().any(|kind| {
                        preview_counts.contains_key(kind.as_str())
                            && !active_counts.contains_key(kind.as_str())
                    });
                    for kind in &bot.carrying {
                        if let Some(order_id) = first_active_order_for_kind(input.state, kind) {
                            plan.forced_intents.insert(
                                bot.id.clone(),
                                Intent::DropOff {
                                    order_id: order_id.to_owned(),
                                },
                            );
                            continue;
                        }
                    }
                    if has_preview_only {
                        plan.forced_intents.insert(bot.id.clone(), Intent::Wait);
                        continue;
                    }
                }
                if preview_need == 0 {
                    break;
                }
                let Some(from) = bot_cell(input.map, bot) else {
                    continue;
                };
                if let Some(kind) = preview_counts.keys().next() {
                    if let Some((item_id, stand)) = choose_scored_item_of_kind(
                        input.state,
                        input.map,
                        input.dist,
                        input.team,
                        bot.id.as_str(),
                        from,
                        kind,
                        bot.carrying.len(),
                        bot.capacity,
                    ) {
                        if let Some(item) = input.state.items.iter().find(|i| i.id == item_id) {
                            if bot.carrying.len() < bot.capacity
                                && is_adjacent_to_item(bot, item.x, item.y)
                            {
                                plan.forced_intents.insert(
                                    bot.id.clone(),
                                    Intent::PickUp {
                                        item_id: item_id.to_owned(),
                                    },
                                );
                                preview_need = preview_need.saturating_sub(1);
                            } else {
                                plan.forced_intents
                                    .insert(bot.id.clone(), Intent::MoveTo { cell: stand });
                            }
                        }
                    }
                }
            }
        }

        // Runners are finishers when active order is nearly complete.
        if active_gap <= runners as u32 {
            for bot in &bots[pickers..pickers + runners] {
                let Some(from) = bot_cell(input.map, bot) else {
                    continue;
                };
                if let Some(kind) = active_counts.keys().next() {
                    if let Some((item_id, stand)) = choose_scored_item_of_kind(
                        input.state,
                        input.map,
                        input.dist,
                        input.team,
                        bot.id.as_str(),
                        from,
                        kind,
                        bot.carrying.len(),
                        bot.capacity,
                    ) {
                        if let Some(item) = input.state.items.iter().find(|i| i.id == item_id) {
                            if bot.carrying.len() < bot.capacity
                                && is_adjacent_to_item(bot, item.x, item.y)
                            {
                                plan.forced_intents.insert(
                                    bot.id.clone(),
                                    Intent::PickUp {
                                        item_id: item_id.to_owned(),
                                    },
                                );
                            } else {
                                plan.forced_intents
                                    .insert(bot.id.clone(), Intent::MoveTo { cell: stand });
                            }
                        }
                    }
                }
            }
        }

        plan
    }
}
