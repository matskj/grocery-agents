use std::collections::{HashMap, HashSet};

use super::{
    common::{
        active_kind_counts, bot_cell, carrying_kind_counts, detour_cost,
        find_adjacent_item_for_kind, first_preview_kind, item_targets_for_kind,
        nearest_dropoff_cell, on_dropoff, pick_best_item_target, try_dropoff_active,
    },
    Intent, PlanResult, TickContext,
};

#[derive(Debug, Default)]
pub struct EasyPlanner;

impl EasyPlanner {
    pub fn tick(&mut self, input: TickContext<'_>) -> PlanResult {
        let mut plan = PlanResult::empty("easy_exact");
        let Some(bot) = input.state.bots.first() else {
            return plan;
        };

        plan.role_label_by_bot
            .insert(bot.id.clone(), "easy_solo".to_owned());
        plan.local_radius_by_bot.insert(bot.id.clone(), 6);
        plan.explicit_priority.push(bot.id.clone());

        if let Some(intent) = try_dropoff_active(input.state, input.map, bot) {
            plan.intents.insert(bot.id.clone(), intent);
            return plan;
        }

        let Some(start) = bot_cell(input.map, bot) else {
            plan.intents.insert(bot.id.clone(), Intent::Wait);
            return plan;
        };

        let active_counts = active_kind_counts(input.state);
        if active_counts.is_empty() {
            plan_preview_only(input, &mut plan, start);
            return plan;
        }

        let carrying_counts = carrying_kind_counts(bot);
        let mut required_slots = expand_required_slots(&active_counts, &carrying_counts);
        if required_slots.is_empty() {
            if !on_dropoff(input.map, bot) {
                if let Some(drop) = nearest_dropoff_cell(input.map, input.dist, start) {
                    plan.goal_cell_by_bot.insert(bot.id.clone(), drop);
                    plan.intents
                        .insert(bot.id.clone(), Intent::MoveTo { cell: drop });
                } else {
                    plan.intents.insert(bot.id.clone(), Intent::Wait);
                }
                return plan;
            }
            plan_preview_only(input, &mut plan, start);
            return plan;
        }

        required_slots.sort();
        let capacity_left = bot.capacity.saturating_sub(bot.carrying.len());
        let active_needed = required_slots.len();

        let best_primary = if active_needed <= 3 {
            solve_single_trip(input, start, &required_slots)
        } else {
            solve_two_trip_split(input, start, &required_slots)
        };

        if let Some(primary) = best_primary {
            plan.goal_cell_by_bot
                .insert(bot.id.clone(), primary.first_stand);

            if bot.carrying.len() < bot.capacity
                && is_adjacent_to_target(bot.x, bot.y, primary.first_item_x, primary.first_item_y)
            {
                plan.intents.insert(
                    bot.id.clone(),
                    Intent::PickUp {
                        item_id: primary.first_item_id,
                    },
                );
            } else {
                plan.intents.insert(
                    bot.id.clone(),
                    Intent::MoveTo {
                        cell: primary.first_stand,
                    },
                );
            }

            let can_fit_preview = active_needed < capacity_left;
            if can_fit_preview {
                if let Some((preview_item, preview_stand)) = find_preview_detour_candidate(
                    input,
                    start,
                    primary.first_stand,
                    primary.dropoff,
                ) {
                    plan.goal_cell_by_bot.insert(bot.id.clone(), preview_stand);
                    if bot.carrying.len() < bot.capacity
                        && is_adjacent_to_target(bot.x, bot.y, preview_item.0, preview_item.1)
                    {
                        plan.intents.insert(
                            bot.id.clone(),
                            Intent::PickUp {
                                item_id: preview_item.2,
                            },
                        );
                    } else {
                        plan.intents.insert(
                            bot.id.clone(),
                            Intent::MoveTo {
                                cell: preview_stand,
                            },
                        );
                    }
                }
            }
            return plan;
        }

        plan.intents.insert(bot.id.clone(), Intent::Wait);
        plan
    }
}

#[derive(Debug, Clone)]
struct TripPlan {
    total_cost: u32,
    first_item_id: String,
    first_stand: u16,
    first_item_x: i32,
    first_item_y: i32,
    dropoff: u16,
}

fn plan_preview_only(input: TickContext<'_>, plan: &mut PlanResult, start: u16) {
    let Some(bot) = input.state.bots.first() else {
        return;
    };
    let Some(kind) = first_preview_kind(input.state) else {
        plan.intents.insert(bot.id.clone(), Intent::Wait);
        return;
    };
    if let Some(item_id) = find_adjacent_item_for_kind(input.state, bot, kind) {
        plan.intents.insert(
            bot.id.clone(),
            Intent::PickUp {
                item_id: item_id.to_owned(),
            },
        );
        return;
    }

    if let Some(target) =
        pick_best_item_target(input.state, input.map, input.dist, start, kind, |_| 0.0)
    {
        plan.goal_cell_by_bot
            .insert(bot.id.clone(), target.stand_cell);
        plan.intents.insert(
            bot.id.clone(),
            Intent::MoveTo {
                cell: target.stand_cell,
            },
        );
    } else {
        plan.intents.insert(bot.id.clone(), Intent::Wait);
    }
}

fn expand_required_slots(
    active_counts: &HashMap<String, u16>,
    carrying_counts: &HashMap<String, u16>,
) -> Vec<String> {
    let mut slots = Vec::<String>::new();
    for (kind, needed) in active_counts {
        let carrying = carrying_counts.get(kind).copied().unwrap_or(0);
        let missing = needed.saturating_sub(carrying);
        for _ in 0..missing {
            slots.push(kind.clone());
        }
    }
    slots
}

fn solve_single_trip(input: TickContext<'_>, start: u16, slots: &[String]) -> Option<TripPlan> {
    let perms = permutations(slots.len());
    let mut best = None::<TripPlan>;
    for perm in perms {
        let mut ordered = Vec::with_capacity(slots.len());
        for idx in perm {
            ordered.push(slots[idx].as_str());
        }
        if let Some(plan) = evaluate_sequence(input, start, &ordered) {
            match &best {
                Some(current) if plan.total_cost >= current.total_cost => {}
                _ => best = Some(plan),
            }
        }
    }
    best
}

fn solve_two_trip_split(input: TickContext<'_>, start: u16, slots: &[String]) -> Option<TripPlan> {
    if slots.len() != 4 {
        return solve_single_trip(input, start, slots);
    }

    let mut best = None::<TripPlan>;
    for tail_idx in 0..slots.len() {
        let mut first_trip = Vec::<String>::new();
        let mut tail = None::<String>;
        for (idx, kind) in slots.iter().enumerate() {
            if idx == tail_idx {
                tail = Some(kind.clone());
            } else {
                first_trip.push(kind.clone());
            }
        }
        let Some(tail_kind) = tail else {
            continue;
        };
        let Some(primary) = solve_single_trip(input, start, &first_trip) else {
            continue;
        };

        let second =
            best_single_kind_roundtrip(input, primary.dropoff, &tail_kind).unwrap_or(u32::MAX);
        if second == u32::MAX {
            continue;
        }

        let total = primary.total_cost.saturating_add(second);
        if best.as_ref().map(|b| total < b.total_cost).unwrap_or(true) {
            best = Some(TripPlan {
                total_cost: total,
                ..primary
            });
        }
    }
    best
}

fn best_single_kind_roundtrip(input: TickContext<'_>, from: u16, kind: &str) -> Option<u32> {
    let drop = nearest_dropoff_cell(input.map, input.dist, from)?;
    let mut best = u32::MAX;
    for target in item_targets_for_kind(input.state, input.map, kind) {
        let a = input.dist.dist(from, target.stand_cell);
        let b = input.dist.dist(target.stand_cell, drop);
        if a == u16::MAX || b == u16::MAX {
            continue;
        }
        best = best.min(u32::from(a) + u32::from(b));
    }
    if best == u32::MAX {
        None
    } else {
        Some(best)
    }
}

fn evaluate_sequence(input: TickContext<'_>, start: u16, sequence: &[&str]) -> Option<TripPlan> {
    let mut used = HashSet::<String>::new();
    let mut best = None::<TripPlan>;

    dfs_sequence(input, sequence, 0, start, 0, &mut used, None, &mut best);

    best
}

fn dfs_sequence(
    input: TickContext<'_>,
    sequence: &[&str],
    idx: usize,
    current_cell: u16,
    current_cost: u32,
    used_items: &mut HashSet<String>,
    first_choice: Option<(String, u16, i32, i32)>,
    best: &mut Option<TripPlan>,
) {
    if idx == sequence.len() {
        let Some(drop) = nearest_dropoff_cell(input.map, input.dist, current_cell) else {
            return;
        };
        let tail = input.dist.dist(current_cell, drop);
        if tail == u16::MAX {
            return;
        }
        let total = current_cost.saturating_add(u32::from(tail));
        let Some((item_id, stand, item_x, item_y)) = first_choice else {
            return;
        };
        let candidate = TripPlan {
            total_cost: total,
            first_item_id: item_id,
            first_stand: stand,
            first_item_x: item_x,
            first_item_y: item_y,
            dropoff: drop,
        };
        if best
            .as_ref()
            .map(|current| total < current.total_cost)
            .unwrap_or(true)
        {
            *best = Some(candidate);
        }
        return;
    }

    let kind = sequence[idx];
    for target in item_targets_for_kind(input.state, input.map, kind) {
        if used_items.contains(target.item_id) {
            continue;
        }
        let step = input.dist.dist(current_cell, target.stand_cell);
        if step == u16::MAX {
            continue;
        }
        let next_cost = current_cost.saturating_add(u32::from(step));
        if let Some(current_best) = best {
            if next_cost >= current_best.total_cost {
                continue;
            }
        }

        used_items.insert(target.item_id.to_owned());
        let next_first = first_choice.clone().or_else(|| {
            Some((
                target.item_id.to_owned(),
                target.stand_cell,
                target.item_x,
                target.item_y,
            ))
        });
        dfs_sequence(
            input,
            sequence,
            idx + 1,
            target.stand_cell,
            next_cost,
            used_items,
            next_first,
            best,
        );
        used_items.remove(target.item_id);
    }
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn rec(cur: &mut Vec<usize>, used: &mut [bool], out: &mut Vec<Vec<usize>>, n: usize) {
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
            rec(cur, used, out, n);
            cur.pop();
            used[i] = false;
        }
    }

    let mut out = Vec::new();
    let mut cur = Vec::new();
    let mut used = vec![false; n];
    rec(&mut cur, &mut used, &mut out, n);
    out
}

fn find_preview_detour_candidate(
    input: TickContext<'_>,
    start: u16,
    active_stand: u16,
    dropoff: u16,
) -> Option<((i32, i32, String), u16)> {
    let preview_kind = first_preview_kind(input.state)?;
    let mut best = None::<((i32, i32, String), u16, i32)>;

    for target in item_targets_for_kind(input.state, input.map, preview_kind) {
        let detour_to_active = detour_cost(input.dist, start, target.stand_cell, active_stand);
        let detour_to_drop = detour_cost(input.dist, active_stand, target.stand_cell, dropoff);
        let detour = match (detour_to_active, detour_to_drop) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => continue,
        };
        if detour > 2 {
            continue;
        }
        match best {
            Some((_, _, cur)) if detour >= cur => {}
            _ => {
                best = Some((
                    (target.item_x, target.item_y, target.item_id.to_owned()),
                    target.stand_cell,
                    detour,
                ));
            }
        }
    }

    best.map(|(item, stand, _)| (item, stand))
}

fn is_adjacent_to_target(bot_x: i32, bot_y: i32, item_x: i32, item_y: i32) -> bool {
    (item_x - bot_x).unsigned_abs() + (item_y - bot_y).unsigned_abs() == 1
}

#[cfg(test)]
mod tests {
    use crate::{
        dist::DistanceMap,
        model::{BotState, GameState, Grid, Item, Order, OrderStatus},
        world::World,
    };

    use super::{expand_required_slots, solve_single_trip, solve_two_trip_split, EasyPlanner};
    use crate::planner::TickContext;

    fn base_easy_state() -> GameState {
        GameState {
            grid: Grid {
                width: 12,
                height: 10,
                drop_off_tiles: vec![[0, 9]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 1,
                y: 8,
                carrying: vec![],
                capacity: 3,
            }],
            items: vec![
                Item {
                    id: "milk_a".to_owned(),
                    kind: "milk".to_owned(),
                    x: 3,
                    y: 5,
                },
                Item {
                    id: "milk_b".to_owned(),
                    kind: "milk".to_owned(),
                    x: 8,
                    y: 2,
                },
                Item {
                    id: "bread_a".to_owned(),
                    kind: "bread".to_owned(),
                    x: 5,
                    y: 5,
                },
                Item {
                    id: "egg_a".to_owned(),
                    kind: "egg".to_owned(),
                    x: 9,
                    y: 5,
                },
                Item {
                    id: "juice_a".to_owned(),
                    kind: "juice".to_owned(),
                    x: 6,
                    y: 2,
                },
            ],
            orders: vec![],
            ..GameState::default()
        }
    }

    #[test]
    fn expand_required_slots_subtracts_carrying() {
        let mut active = std::collections::HashMap::<String, u16>::new();
        active.insert("milk".to_owned(), 2);
        active.insert("bread".to_owned(), 1);
        let mut carrying = std::collections::HashMap::<String, u16>::new();
        carrying.insert("milk".to_owned(), 1);

        let mut slots = expand_required_slots(&active, &carrying);
        slots.sort();
        assert_eq!(slots, vec!["bread".to_owned(), "milk".to_owned()]);
    }

    #[test]
    fn single_trip_solver_returns_plan_for_three_items() {
        let mut state = base_easy_state();
        state.orders = vec![
            Order {
                id: "o1".to_owned(),
                item_id: "milk".to_owned(),
                status: OrderStatus::InProgress,
            },
            Order {
                id: "o2".to_owned(),
                item_id: "bread".to_owned(),
                status: OrderStatus::InProgress,
            },
            Order {
                id: "o3".to_owned(),
                item_id: "egg".to_owned(),
                status: OrderStatus::InProgress,
            },
        ];
        let world = World::new(&state);
        let map = world.map();
        let dist = DistanceMap::build(map);
        let start = map.idx(1, 8).expect("start");
        let slots = vec!["milk".to_owned(), "bread".to_owned(), "egg".to_owned()];
        let ctx = TickContext {
            state: &state,
            map,
            dist: &dist,
            tick: 0,
        };

        let plan = solve_single_trip(ctx, start, &slots).expect("trip plan");
        assert!(plan.total_cost > 0);
        assert!(
            plan.first_item_id.starts_with("milk")
                || plan.first_item_id.starts_with("bread")
                || plan.first_item_id.starts_with("egg")
        );
    }

    #[test]
    fn two_trip_split_solver_handles_four_items() {
        let mut state = base_easy_state();
        state.orders = vec![
            Order {
                id: "o1".to_owned(),
                item_id: "milk".to_owned(),
                status: OrderStatus::InProgress,
            },
            Order {
                id: "o2".to_owned(),
                item_id: "bread".to_owned(),
                status: OrderStatus::InProgress,
            },
            Order {
                id: "o3".to_owned(),
                item_id: "egg".to_owned(),
                status: OrderStatus::InProgress,
            },
            Order {
                id: "o4".to_owned(),
                item_id: "juice".to_owned(),
                status: OrderStatus::InProgress,
            },
        ];
        let world = World::new(&state);
        let map = world.map();
        let dist = DistanceMap::build(map);
        let start = map.idx(1, 8).expect("start");
        let slots = vec![
            "milk".to_owned(),
            "bread".to_owned(),
            "egg".to_owned(),
            "juice".to_owned(),
        ];
        let ctx = TickContext {
            state: &state,
            map,
            dist: &dist,
            tick: 0,
        };

        let plan = solve_two_trip_split(ctx, start, &slots).expect("two trip plan");
        assert!(plan.total_cost > 0);
    }

    #[test]
    fn easy_planner_prefers_dropoff_when_carrying_active() {
        let mut state = base_easy_state();
        state.bots[0].x = 0;
        state.bots[0].y = 9;
        state.bots[0].carrying = vec!["milk".to_owned()];
        state.orders = vec![Order {
            id: "o1".to_owned(),
            item_id: "milk".to_owned(),
            status: OrderStatus::InProgress,
        }];
        let world = World::new(&state);
        let map = world.map();
        let dist = DistanceMap::build(map);

        let mut planner = EasyPlanner::default();
        let plan = planner.tick(TickContext {
            state: &state,
            map,
            dist: &dist,
            tick: 0,
        });
        let intent = plan.intents.get("0").expect("intent");
        assert!(matches!(intent, crate::planner::Intent::DropOff { .. }));
    }
}
