use crate::dispatcher::Intent;

use super::{
    common::{
        active_kind_counts, bot_cell, choose_nearest_item_of_kind, first_active_order_for_kind,
        is_adjacent_to_item, on_dropoff, preview_kind_counts,
    },
    Strategy, StrategyPlan, TickInput,
};

#[derive(Debug, Default)]
pub struct ExpertStrategy;

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

        let pickers = 5usize.min(bots.len());
        let runners = 3usize.min(bots.len().saturating_sub(pickers));
        let buffers = 2usize.min(bots.len().saturating_sub(pickers + runners));

        for (idx, bot) in bots.iter().enumerate() {
            if idx < pickers {
                let area_id = (idx as u16) % 5;
                plan.role_label_by_bot
                    .insert(bot.id.clone(), format!("expert_picker_r{area_id}"));
                plan.preferred_area_by_bot.insert(bot.id.clone(), area_id);
                plan.expansion_mode_by_bot.insert(bot.id.clone(), false);
            } else if idx < pickers + runners {
                plan.role_label_by_bot
                    .insert(bot.id.clone(), "expert_runner".to_owned());
                plan.explicit_order.push(bot.id.clone());
            } else if idx < pickers + runners + buffers {
                plan.role_label_by_bot
                    .insert(bot.id.clone(), "expert_buffer".to_owned());
                plan.explicit_order.push(bot.id.clone());
            } else {
                plan.role_label_by_bot
                    .insert(bot.id.clone(), "expert_flex".to_owned());
            }
        }

        // Buffers prefetch preview kinds and stage near dropoff.
        if !preview_counts.is_empty() {
            for bot in &bots[pickers + runners..pickers + runners + buffers] {
                if on_dropoff(bot, input.map) {
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
                }
                let Some(from) = bot_cell(input.map, bot) else {
                    continue;
                };
                if let Some(kind) = preview_counts.keys().next() {
                    if let Some((item_id, stand)) =
                        choose_nearest_item_of_kind(input.state, input.map, input.dist, from, kind)
                    {
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

        // Runners are finishers when active order is nearly complete.
        if active_gap <= runners as u32 {
            for bot in &bots[pickers..pickers + runners] {
                let Some(from) = bot_cell(input.map, bot) else {
                    continue;
                };
                if let Some(kind) = active_counts.keys().next() {
                    if let Some((item_id, stand)) =
                        choose_nearest_item_of_kind(input.state, input.map, input.dist, from, kind)
                    {
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
