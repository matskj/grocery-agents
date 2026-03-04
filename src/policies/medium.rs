use crate::dispatcher::Intent;

use super::{
    common::{
        active_kind_counts, bot_cell, choose_nearest_item_of_kind, first_active_order_for_kind,
        is_adjacent_to_item, nearest_dropoff_cell, on_dropoff, preview_kind_counts,
    },
    Strategy, StrategyPlan, TickInput,
};

#[derive(Debug, Default)]
pub struct MediumStrategy;

impl Strategy for MediumStrategy {
    fn tick(&mut self, input: TickInput<'_>) -> StrategyPlan {
        let mut plan = StrategyPlan {
            policy_name: "medium_stager",
            strategy_stage: "active_pipeline",
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
        for bot in &bots {
            plan.role_label_by_bot
                .insert(bot.id.clone(), "medium_worker".to_owned());
        }
        let stager = bots
            .iter()
            .filter_map(|bot| {
                let cell = bot_cell(input.map, bot)?;
                let drop = nearest_dropoff_cell(input.map, cell, input.dist)?;
                Some((bot.id.as_str(), input.dist.dist(cell, drop)))
            })
            .min_by_key(|(_, d)| *d)
            .map(|(id, _)| id.to_owned());

        if let Some(stager_id) = stager {
            plan.role_label_by_bot
                .insert(stager_id.clone(), "medium_stager".to_owned());
            plan.explicit_order.push(stager_id.clone());

            if active_gap <= 2 && !preview_counts.is_empty() {
                if let Some(bot) = input.state.bots.iter().find(|b| b.id == stager_id) {
                    if on_dropoff(bot, input.map) {
                        for kind in &bot.carrying {
                            if let Some(order_id) = first_active_order_for_kind(input.state, kind) {
                                plan.forced_intents.insert(
                                    bot.id.clone(),
                                    Intent::DropOff {
                                        order_id: order_id.to_owned(),
                                    },
                                );
                                return plan;
                            }
                        }
                    }
                    if let Some(from) = bot_cell(input.map, bot) {
                        let kind = preview_counts.keys().next().cloned();
                        if let Some(kind) = kind {
                            if let Some((item_id, stand)) = choose_nearest_item_of_kind(
                                input.state,
                                input.map,
                                input.dist,
                                from,
                                &kind,
                            ) {
                                if let Some(item) =
                                    input.state.items.iter().find(|item| item.id == item_id)
                                {
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
            }
        }

        plan
    }
}
