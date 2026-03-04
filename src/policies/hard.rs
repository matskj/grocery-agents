use crate::dispatcher::Intent;

use super::{
    common::{
        active_kind_counts, bot_cell, choose_scored_item_of_kind, first_active_order_for_kind,
        is_adjacent_to_item, nearest_dropoff_cell, on_dropoff, preview_kind_counts,
    },
    Strategy, StrategyPlan, TickInput,
};

#[derive(Debug, Default)]
pub struct HardStrategy;

impl Strategy for HardStrategy {
    fn tick(&mut self, input: TickInput<'_>) -> StrategyPlan {
        let mut plan = StrategyPlan {
            policy_name: "hard_regional_runner",
            strategy_stage: "regional_claims",
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
        let runner_id = bots
            .iter()
            .filter_map(|bot| {
                let cell = bot_cell(input.map, bot)?;
                let drop = nearest_dropoff_cell(input.map, cell, input.dist)?;
                Some((bot.id.clone(), input.dist.dist(cell, drop)))
            })
            .min_by_key(|(_, d)| *d)
            .map(|(id, _)| id);

        let mut picker_rank = 0u16;
        for bot in bots {
            if Some(bot.id.clone()) == runner_id {
                plan.role_label_by_bot
                    .insert(bot.id.clone(), "hard_runner".to_owned());
                plan.explicit_order.push(bot.id.clone());
                continue;
            }
            let area_id = picker_rank % 4;
            picker_rank = picker_rank.saturating_add(1);
            plan.role_label_by_bot
                .insert(bot.id.clone(), format!("hard_picker_r{area_id}"));
            plan.preferred_area_by_bot.insert(bot.id.clone(), area_id);
            plan.expansion_mode_by_bot.insert(bot.id.clone(), false);
        }

        if let Some(runner_id) = runner_id {
            if let Some(bot) = input.state.bots.iter().find(|b| b.id == runner_id) {
                if on_dropoff(bot, input.map) {
                    let has_preview = bot
                        .carrying
                        .iter()
                        .any(|kind| preview_counts.contains_key(kind.as_str()));
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
                    if has_preview {
                        plan.forced_intents.insert(bot.id.clone(), Intent::Wait);
                        return plan;
                    }
                }
                if let Some(from) = bot_cell(input.map, bot) {
                    let target_kind = if active_gap <= 2 {
                        active_counts.keys().next().cloned()
                    } else {
                        preview_counts.keys().next().cloned()
                    };
                    if let Some(kind) = target_kind {
                        if let Some((item_id, stand)) = choose_scored_item_of_kind(
                            input.state,
                            input.map,
                            input.dist,
                            input.team,
                            bot.id.as_str(),
                            from,
                            &kind,
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
        }

        plan
    }
}
