use std::collections::HashSet;

use crate::dispatcher::Intent;

use super::{
    common::{
        active_missing_kinds, bot_cell, choose_nearest_item_of_kind, first_active_order_for_kind,
        is_adjacent_to_item, nearest_dropoff_cell, on_dropoff,
    },
    Strategy, StrategyPlan, TickInput,
};

#[derive(Debug, Default)]
pub struct EasyStrategy;

impl Strategy for EasyStrategy {
    fn tick(&mut self, input: TickInput<'_>) -> StrategyPlan {
        let mut plan = StrategyPlan {
            policy_name: "easy_exact",
            strategy_stage: "easy_exact_solver",
            ..StrategyPlan::default()
        };
        let Some(bot) = input.state.bots.first() else {
            return plan;
        };
        let Some(bot_cell) = bot_cell(input.map, bot) else {
            return plan;
        };

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

        let needed = active_missing_kinds(input.state);
        if needed.is_empty() {
            return plan;
        }

        let target_kind = if needed.len() <= 3 {
            best_kind_from_exact_subset(&input, &needed).unwrap_or_else(|| needed[0].clone())
        } else if needed.len() == 4 {
            best_kind_from_split(&input, &needed).unwrap_or_else(|| needed[0].clone())
        } else {
            needed[0].clone()
        };

        let Some((item_id, stand)) =
            choose_nearest_item_of_kind(input.state, input.map, input.dist, bot_cell, &target_kind)
        else {
            return plan;
        };
        if let Some(item) = input.state.items.iter().find(|item| item.id == item_id) {
            if bot.carrying.len() < bot.capacity && is_adjacent_to_item(bot, item.x, item.y) {
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

        if let Some(drop) = nearest_dropoff_cell(input.map, bot_cell, input.dist) {
            plan.role_label_by_bot
                .insert(bot.id.clone(), format!("easy_solo_drop={drop}"));
        }
        plan.explicit_order.push(bot.id.clone());
        plan
    }
}

fn best_kind_from_exact_subset(input: &TickInput<'_>, needed: &[String]) -> Option<String> {
    let slots = needed.iter().take(3).cloned().collect::<Vec<_>>();
    if slots.is_empty() {
        return None;
    }
    let perms = permutations(slots.len());
    let mut best_cost = u32::MAX;
    let mut best_first_kind = None::<String>;
    for perm in perms {
        let seq = perm
            .iter()
            .map(|&idx| slots[idx].as_str())
            .collect::<Vec<_>>();
        if let Some((cost, first_kind)) = evaluate_kind_sequence(input, &seq) {
            if cost < best_cost {
                best_cost = cost;
                best_first_kind = Some(first_kind.to_owned());
            }
        }
    }
    best_first_kind
}

fn best_kind_from_split(input: &TickInput<'_>, needed: &[String]) -> Option<String> {
    if needed.len() != 4 {
        return None;
    }
    let mut best_cost = u32::MAX;
    let mut best_first_kind = None::<String>;
    for split_last in 0..4 {
        let mut primary = Vec::new();
        let mut tail = None::<&str>;
        for (idx, kind) in needed.iter().enumerate() {
            if idx == split_last {
                tail = Some(kind.as_str());
            } else {
                primary.push(kind.as_str());
            }
        }
        let Some(tail_kind) = tail else {
            continue;
        };
        let perms = permutations(primary.len());
        for perm in perms {
            let seq = perm.iter().map(|&idx| primary[idx]).collect::<Vec<_>>();
            let Some((first_cost, first_kind)) = evaluate_kind_sequence(input, &seq) else {
                continue;
            };
            let Some(bot) = input.state.bots.first() else {
                continue;
            };
            let Some(bot_cell) = input.map.idx(bot.x, bot.y) else {
                continue;
            };
            let Some(drop) = nearest_dropoff_cell(input.map, bot_cell, input.dist) else {
                continue;
            };
            let Some((_, stand_tail)) =
                choose_nearest_item_of_kind(input.state, input.map, input.dist, drop, tail_kind)
            else {
                continue;
            };
            let second = u32::from(input.dist.dist(drop, stand_tail))
                + u32::from(
                    nearest_dropoff_cell(input.map, stand_tail, input.dist)
                        .map(|d| input.dist.dist(stand_tail, d))
                        .unwrap_or(u16::MAX),
                );
            let total = first_cost.saturating_add(second);
            if total < best_cost {
                best_cost = total;
                best_first_kind = Some(first_kind.to_owned());
            }
        }
    }
    best_first_kind
}

fn evaluate_kind_sequence<'a>(
    input: &TickInput<'_>,
    sequence: &[&'a str],
) -> Option<(u32, &'a str)> {
    let bot = input.state.bots.first()?;
    let start = input.map.idx(bot.x, bot.y)?;
    let mut states = vec![(start, 0u32)];
    let mut used_items = HashSet::<String>::new();
    for kind in sequence {
        let mut next = Vec::<(u16, u32, String)>::new();
        for (cell, base_cost) in &states {
            for item in &input.state.items {
                if item.kind != *kind || used_items.contains(&item.id) {
                    continue;
                }
                for &stand in input.map.stand_cells_for_item(&item.id) {
                    let d = input.dist.dist(*cell, stand);
                    if d == u16::MAX {
                        continue;
                    }
                    next.push((
                        stand,
                        base_cost.saturating_add(u32::from(d)),
                        item.id.clone(),
                    ));
                }
            }
        }
        if next.is_empty() {
            return None;
        }
        next.sort_by_key(|entry| entry.1);
        let mut dedup = Vec::<(u16, u32)>::new();
        let mut local_used = HashSet::<String>::new();
        for (stand, cost, item_id) in next {
            if local_used.contains(&item_id) {
                continue;
            }
            local_used.insert(item_id);
            if let Some((_, best)) = dedup.iter().find(|(s, _)| *s == stand) {
                if cost >= *best {
                    continue;
                }
            }
            dedup.retain(|(s, _)| *s != stand);
            dedup.push((stand, cost));
            if dedup.len() >= 8 {
                break;
            }
        }
        states = dedup;
        if let Some(item) = input.state.items.iter().find(|i| i.kind == *kind) {
            used_items.insert(item.id.clone());
        }
    }
    let mut best = u32::MAX;
    for (cell, cost) in states {
        if let Some(drop) = nearest_dropoff_cell(input.map, cell, input.dist) {
            let tail = input.dist.dist(cell, drop);
            if tail != u16::MAX {
                best = best.min(cost.saturating_add(u32::from(tail)));
            }
        }
    }
    if best == u32::MAX {
        None
    } else {
        Some((best, sequence[0]))
    }
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn go(cur: &mut Vec<usize>, used: &mut [bool], out: &mut Vec<Vec<usize>>, n: usize) {
        if cur.len() == n {
            out.push(cur.clone());
            return;
        }
        for i in 0..n {
            if used[i] {
                continue;
            }
            used[i] = true;
            cur.push(i);
            go(cur, used, out, n);
            cur.pop();
            used[i] = false;
        }
    }
    let mut out = Vec::new();
    let mut cur = Vec::new();
    let mut used = vec![false; n];
    go(&mut cur, &mut used, &mut out, n);
    out
}
