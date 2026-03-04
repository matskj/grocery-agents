use std::collections::{HashMap, HashSet};

use crate::{
    dist::DistanceMap,
    model::{BotState, GameState, OrderStatus},
    scoring::{detect_mode_label, maybe_score_pick, CandidateFeatures},
    team_context::{BotRole, TeamContext},
    world::MapCache,
};

#[derive(Debug, Clone)]
pub enum Intent {
    DropOff { order_id: String },
    PickUp { item_id: String },
    MoveTo { cell: u16 },
    Wait,
}

#[derive(Debug, Clone)]
pub struct BotIntent {
    pub bot_id: String,
    pub intent: Intent,
}

#[derive(Debug, Default)]
pub struct Dispatcher {
    item_intern: HashMap<String, u16>,
    item_rev: Vec<String>,
    needed_buf: Vec<u16>,
}

#[derive(Debug, Clone)]
struct PickupCandidate {
    item_id: String,
    cell: u16,
    score: f64,
}

#[derive(Debug, Clone)]
struct IntentCandidate {
    bot_id: String,
    target_cell: Option<u16>,
    score: f64,
    intent: Intent,
}

#[derive(Debug, Clone)]
struct BotCandidateSet {
    bot_id: String,
    blocked_ticks: u8,
    candidates: Vec<IntentCandidate>,
}

impl Dispatcher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dispatch(&self, action: crate::model::Action) -> crate::model::Action {
        action
    }

    pub fn build_intents(
        &mut self,
        state: &GameState,
        map: &MapCache,
        dist: &DistanceMap,
        blocked_ticks: &HashMap<String, u8>,
        team: &TeamContext,
    ) -> Vec<BotIntent> {
        if let Some(unstack) = self.spawn_unstack_intents(state, map, blocked_ticks) {
            return unstack;
        }

        let (active_missing, preview_missing) = self.compute_missing(state);
        let mode = detect_mode_label(state);
        let order_urgency = active_missing.values().copied().map(f64::from).sum::<f64>();
        let carried_supply = self.build_effective_carried_supply(state, map, dist, team);
        let active_gap_total = active_missing
            .iter()
            .map(|(item, needed)| {
                let covered = carried_supply.get(item).copied().unwrap_or(0);
                needed.saturating_sub(covered)
            })
            .sum::<u16>();
        let preview_enabled = active_gap_total == 0;

        let mut bot_order = state.bots.iter().collect::<Vec<_>>();
        bot_order.sort_by(|a, b| {
            let ab = blocked_ticks.get(&a.id).copied().unwrap_or(0);
            let bb = blocked_ticks.get(&b.id).copied().unwrap_or(0);
            bb.cmp(&ab).then_with(|| a.id.cmp(&b.id))
        });
        if !bot_order.is_empty() {
            let rot = (state.tick as usize) % bot_order.len();
            bot_order.rotate_left(rot);
        }

        let mut all_sets = Vec::with_capacity(bot_order.len());
        for bot in bot_order {
            let bot_idx = match map.idx(bot.x, bot.y) {
                Some(v) => v,
                None => {
                    all_sets.push(BotCandidateSet {
                        bot_id: bot.id.clone(),
                        blocked_ticks: 0,
                        candidates: vec![IntentCandidate {
                            bot_id: bot.id.clone(),
                            target_cell: None,
                            score: -1_000.0,
                            intent: Intent::Wait,
                        }],
                    });
                    continue;
                }
            };
            let blocked = blocked_ticks.get(&bot.id).copied().unwrap_or(0);
            let blocked_f = blocked as f64;
            let role = team.role_for(&bot.id);
            let queue_distance = team
                .queue
                .assignments
                .get(&bot.id)
                .map(|assignment| f64::from(assignment.queue_distance))
                .unwrap_or(99.0);
            let local_congestion = state
                .bots
                .iter()
                .filter(|other| other.id != bot.id)
                .filter(|other| (other.x - bot.x).abs() + (other.y - bot.y).abs() <= 1)
                .count() as f64;
            let local_conflict_count = team
                .traffic
                .local_conflict_count_by_bot
                .get(&bot.id)
                .copied()
                .map(f64::from)
                .unwrap_or(0.0);
            let conflict_degree = team
                .traffic
                .conflict_degree_by_bot
                .get(&bot.id)
                .copied()
                .unwrap_or(0) as f64;
            let teammate_proximity = avg_teammate_distance(bot, &state.bots);
            let inventory_util = if bot.capacity == 0 {
                0.0
            } else {
                bot.carrying.len() as f64 / bot.capacity as f64
            };
            let nearest_drop_dist = map
                .dropoff_cells
                .iter()
                .map(|&d| f64::from(dist.dist(bot_idx, d)))
                .filter(|v| *v < u16::MAX as f64)
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(99.0);
            let serviceable_dropoff = if team
                .dropoff_serviceable_by_bot
                .get(&bot.id)
                .copied()
                .unwrap_or(false)
            {
                1.0
            } else {
                0.0
            };

            let mut candidates = Vec::new();

            if let Some(queue_goal) = team.queue_goal_for(&bot.id) {
                if matches!(role, BotRole::LeadCourier | BotRole::QueueCourier) {
                    let queue_score = if queue_goal == bot_idx {
                        1_700.0
                    } else {
                        1_500.0 - f64::from(dist.dist(bot_idx, queue_goal)) * 5.0
                    };
                    candidates.push(IntentCandidate {
                        bot_id: bot.id.clone(),
                        target_cell: Some(queue_goal),
                        score: queue_score + blocked_f + conflict_degree,
                        intent: Intent::MoveTo { cell: queue_goal },
                    });
                }
            }

            if !bot.carrying.is_empty() && map.dropoff_cells.contains(&bot_idx) {
                if let Some(order) = state.orders.iter().find(|o| {
                    bot.carrying.iter().any(|c| c == &o.item_id)
                        && matches!(o.status, OrderStatus::InProgress)
                }) {
                    candidates.push(IntentCandidate {
                        bot_id: bot.id.clone(),
                        target_cell: Some(bot_idx),
                        score: 2_000.0 + f64::from(blocked),
                        intent: Intent::DropOff {
                            order_id: order.id.clone(),
                        },
                    });
                }
            }

            let useful_carried = bot.carrying.iter().any(|c| {
                self.item_intern
                    .get(c)
                    .map(|id| active_missing.contains_key(id))
                    .unwrap_or(false)
            });
            let queue_active = !team.queue.queue_order.is_empty();

            if useful_carried
                && (!queue_active || matches!(role, BotRole::LeadCourier | BotRole::QueueCourier))
            {
                if let Some(&nearest_drop) = map
                    .dropoff_cells
                    .iter()
                    .min_by_key(|&&d| dist.dist(bot_idx, d))
                {
                    let d = f64::from(dist.dist(bot_idx, nearest_drop));
                    candidates.push(IntentCandidate {
                        bot_id: bot.id.clone(),
                        target_cell: Some(nearest_drop),
                        score: 1_200.0 - d * 4.0 + blocked_f + conflict_degree * 0.2,
                        intent: Intent::MoveTo { cell: nearest_drop },
                    });
                }
            }

            if bot.carrying.len() < bot.capacity {
                if matches!(role, BotRole::QueueCourier) {
                    // Queue couriers should avoid over-picking while lining up for dropoff.
                } else {
                    let mut active_pickups = self.choose_pickup_candidates(
                        bot,
                        state,
                        map,
                        dist,
                        &active_missing,
                        &carried_supply,
                        5,
                        mode,
                        CandidateFeatures {
                            dist_to_nearest_active_item: 0.0,
                            dist_to_dropoff: nearest_drop_dist,
                            inventory_util,
                            queue_distance,
                            local_congestion,
                            local_conflict_count,
                            teammate_proximity,
                            order_urgency,
                            blocked_ticks: blocked_f,
                            queue_role_lead: if matches!(role, BotRole::LeadCourier) {
                                1.0
                            } else {
                                0.0
                            },
                            queue_role_courier: if matches!(role, BotRole::QueueCourier) {
                                1.0
                            } else {
                                0.0
                            },
                            queue_role_collector: if matches!(role, BotRole::Collector) {
                                1.0
                            } else {
                                0.0
                            },
                            queue_role_yield: if matches!(role, BotRole::Yield) {
                                1.0
                            } else {
                                0.0
                            },
                            serviceable_dropoff,
                            stand_failure_count_recent: 0.0,
                            stand_success_count_recent: 0.0,
                            stand_cooldown_ticks_remaining: 0.0,
                            kind_failure_count_recent: 0.0,
                            repeated_same_stand_no_delta_streak: 0.0,
                            contention_at_stand_proxy: 0.0,
                            time_since_last_conversion_tick: 0.0,
                            last_conversion_was_pickup: 0.0,
                            last_conversion_was_dropoff: 0.0,
                            preferred_area_match: 0.0,
                            expansion_mode_active: 0.0,
                            local_active_candidate_count: 0.0,
                            local_radius: 0.0,
                            out_of_area_target: 0.0,
                            out_of_radius_target: 0.0,
                        },
                    );
                    if blocked >= 2 && active_pickups.len() > 1 {
                        active_pickups.rotate_left(1);
                    }
                    for pick in active_pickups {
                        let intent = if pick.cell == bot_idx {
                            Intent::PickUp {
                                item_id: pick.item_id,
                            }
                        } else {
                            Intent::MoveTo { cell: pick.cell }
                        };
                        candidates.push(IntentCandidate {
                            bot_id: bot.id.clone(),
                            target_cell: Some(pick.cell),
                            score: 600.0 + pick.score + blocked_f * 0.5,
                            intent,
                        });
                    }

                    if preview_enabled {
                        let mut preview_pickups = self.choose_pickup_candidates(
                            bot,
                            state,
                            map,
                            dist,
                            &preview_missing,
                            &carried_supply,
                            3,
                            mode,
                            CandidateFeatures {
                                dist_to_nearest_active_item: 0.0,
                                dist_to_dropoff: nearest_drop_dist,
                                inventory_util,
                                queue_distance,
                                local_congestion,
                                local_conflict_count,
                                teammate_proximity,
                                order_urgency: order_urgency * 0.5,
                                blocked_ticks: blocked_f,
                                queue_role_lead: if matches!(role, BotRole::LeadCourier) {
                                    1.0
                                } else {
                                    0.0
                                },
                                queue_role_courier: if matches!(role, BotRole::QueueCourier) {
                                    1.0
                                } else {
                                    0.0
                                },
                                queue_role_collector: if matches!(role, BotRole::Collector) {
                                    1.0
                                } else {
                                    0.0
                                },
                                queue_role_yield: if matches!(role, BotRole::Yield) {
                                    1.0
                                } else {
                                    0.0
                                },
                                serviceable_dropoff,
                                stand_failure_count_recent: 0.0,
                                stand_success_count_recent: 0.0,
                                stand_cooldown_ticks_remaining: 0.0,
                                kind_failure_count_recent: 0.0,
                                repeated_same_stand_no_delta_streak: 0.0,
                                contention_at_stand_proxy: 0.0,
                                time_since_last_conversion_tick: 0.0,
                                last_conversion_was_pickup: 0.0,
                                last_conversion_was_dropoff: 0.0,
                                preferred_area_match: 0.0,
                                expansion_mode_active: 0.0,
                                local_active_candidate_count: 0.0,
                                local_radius: 0.0,
                                out_of_area_target: 0.0,
                                out_of_radius_target: 0.0,
                            },
                        );
                        if blocked >= 2 && preview_pickups.len() > 1 {
                            preview_pickups.rotate_left(1);
                        }
                        for pick in preview_pickups {
                            let intent = if pick.cell == bot_idx {
                                Intent::PickUp {
                                    item_id: pick.item_id,
                                }
                            } else {
                                Intent::MoveTo { cell: pick.cell }
                            };
                            candidates.push(IntentCandidate {
                                bot_id: bot.id.clone(),
                                target_cell: Some(pick.cell),
                                score: 300.0 + pick.score + blocked_f * 0.25,
                                intent,
                            });
                        }
                    }
                }
            }

            for &nb in &map.neighbors[bot_idx as usize] {
                let congestion = state
                    .bots
                    .iter()
                    .filter(|other| map.idx(other.x, other.y) == Some(nb))
                    .count() as f64;
                let visit_heat = team.recent_cell_visits_team.get(&nb).copied().unwrap_or(0) as f64;
                let loop_pressure = team
                    .loop_two_cycle_count_by_bot
                    .get(&bot.id)
                    .copied()
                    .unwrap_or(0) as f64;
                let mut move_score = 50.0 - congestion * 8.0 + blocked_f * 3.0
                    - conflict_degree * 0.5
                    - visit_heat * 1.5
                    - loop_pressure.min(24.0) * 0.2;
                if visit_heat <= 1.0 {
                    // Frontier bonus improves spatial coverage and breaks local loops.
                    move_score += 5.0;
                }
                if matches!(role, BotRole::Yield) {
                    let away = map
                        .dropoff_cells
                        .iter()
                        .map(|&drop| dist.dist(nb, drop))
                        .min()
                        .unwrap_or(u16::MAX);
                    move_score += f64::from(away.min(20));
                }
                candidates.push(IntentCandidate {
                    bot_id: bot.id.clone(),
                    target_cell: Some(nb),
                    score: move_score,
                    intent: Intent::MoveTo { cell: nb },
                });
            }

            candidates.push(IntentCandidate {
                bot_id: bot.id.clone(),
                target_cell: None,
                score: if matches!(role, BotRole::Yield) {
                    -220.0
                } else {
                    -100.0 + blocked_f * 2.0
                },
                intent: Intent::Wait,
            });
            candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.bot_id.cmp(&b.bot_id))
            });

            all_sets.push(BotCandidateSet {
                bot_id: bot.id.clone(),
                blocked_ticks: blocked,
                candidates,
            });
        }

        self.assign_team_intents(all_sets, map)
    }

    fn assign_team_intents(&self, sets: Vec<BotCandidateSet>, map: &MapCache) -> Vec<BotIntent> {
        let mut assigned: HashMap<String, BotIntent> = HashMap::new();
        let mut reserved: HashSet<u16> = HashSet::new();
        let mut flattened = Vec::new();
        for set in &sets {
            for candidate in &set.candidates {
                if matches!(candidate.intent, Intent::Wait) {
                    continue;
                }
                flattened.push((set.blocked_ticks, candidate.clone()));
            }
        }
        flattened.sort_by(|(blocked_a, a), (blocked_b, b)| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| blocked_b.cmp(blocked_a))
                .then_with(|| a.bot_id.cmp(&b.bot_id))
        });

        for (_, candidate) in &flattened {
            if assigned.contains_key(&candidate.bot_id) {
                continue;
            }
            let can_take = match candidate.target_cell {
                Some(cell) => !reserved.contains(&cell) || map.dropoff_cells.contains(&cell),
                None => true,
            };
            if !can_take {
                continue;
            }
            if let Some(cell) = candidate.target_cell {
                if !map.dropoff_cells.contains(&cell) {
                    reserved.insert(cell);
                }
            }
            assigned.insert(
                candidate.bot_id.clone(),
                BotIntent {
                    bot_id: candidate.bot_id.clone(),
                    intent: candidate.intent.clone(),
                },
            );
        }

        // Second pass: if conflicts prevent assignment, allow shared destination
        // before collapsing to wait.
        for set in &sets {
            if assigned.contains_key(&set.bot_id) {
                continue;
            }
            if let Some(best) = set.candidates.first() {
                assigned.insert(
                    set.bot_id.clone(),
                    BotIntent {
                        bot_id: set.bot_id.clone(),
                        intent: best.intent.clone(),
                    },
                );
            }
        }

        sets.into_iter()
            .map(|set| {
                assigned.remove(&set.bot_id).unwrap_or(BotIntent {
                    bot_id: set.bot_id,
                    intent: Intent::Wait,
                })
            })
            .collect()
    }

    fn spawn_unstack_intents(
        &self,
        state: &GameState,
        map: &MapCache,
        blocked_ticks: &HashMap<String, u8>,
    ) -> Option<Vec<BotIntent>> {
        const SPAWN_UNSTACK_TICKS: u64 = 8;
        if state.tick > SPAWN_UNSTACK_TICKS || state.bots.len() <= 1 {
            return None;
        }
        let avg_dist = average_pairwise_distance(&state.bots);
        if avg_dist > 1.2 {
            return None;
        }

        let mut bots = state.bots.iter().collect::<Vec<_>>();
        bots.sort_by(|a, b| {
            let ab = blocked_ticks.get(&a.id).copied().unwrap_or(0);
            let bb = blocked_ticks.get(&b.id).copied().unwrap_or(0);
            bb.cmp(&ab).then_with(|| a.id.cmp(&b.id))
        });

        let mut reserved = HashSet::new();
        let mut out = Vec::with_capacity(bots.len());
        for (idx, bot) in bots.into_iter().enumerate() {
            let lane_x = (((idx + 1) as i32) * map.width) / ((state.bots.len() + 1) as i32);
            let lane_y = if idx % 2 == 0 {
                map.height / 3
            } else {
                (2 * map.height) / 3
            };
            let anchor =
                nearest_open_cell(map, lane_x, lane_y, &reserved).or_else(|| map.idx(bot.x, bot.y));
            let intent = match anchor {
                Some(cell) => {
                    reserved.insert(cell);
                    Intent::MoveTo { cell }
                }
                None => Intent::Wait,
            };
            out.push(BotIntent {
                bot_id: bot.id.clone(),
                intent,
            });
        }
        Some(out)
    }

    fn compute_missing(&mut self, state: &GameState) -> (HashMap<u16, u16>, HashMap<u16, u16>) {
        self.needed_buf.clear();
        let mut active = HashMap::new();
        let mut preview = HashMap::new();
        for order in &state.orders {
            if matches!(
                order.status,
                OrderStatus::Delivered | OrderStatus::Cancelled
            ) {
                continue;
            }
            let item = self.intern(&order.item_id);
            match order.status {
                OrderStatus::InProgress => *active.entry(item).or_insert(0) += 1,
                OrderStatus::Pending => *preview.entry(item).or_insert(0) += 1,
                _ => {}
            }
        }
        (active, preview)
    }

    #[allow(clippy::too_many_arguments)]
    fn choose_pickup_candidates(
        &self,
        bot: &BotState,
        state: &GameState,
        map: &MapCache,
        dist: &DistanceMap,
        need: &HashMap<u16, u16>,
        carried_supply: &HashMap<u16, u16>,
        top_k: usize,
        mode: &str,
        base_features: CandidateFeatures,
    ) -> Vec<PickupCandidate> {
        let bot_idx = match map.idx(bot.x, bot.y) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let mut candidates: Vec<PickupCandidate> = Vec::new();
        for item in &state.items {
            let interned = match self.item_intern.get(&item.kind) {
                Some(v) => *v,
                None => continue,
            };
            let needed = need.get(&interned).copied().unwrap_or(0);
            let carried = carried_supply.get(&interned).copied().unwrap_or(0);
            if needed == 0 || carried >= needed {
                continue;
            }
            for &cell in map.stand_cells_for_item(&item.id) {
                let d = dist.dist(bot_idx, cell);
                if d != u16::MAX {
                    let mut features = base_features;
                    features.dist_to_nearest_active_item = f64::from(d);
                    let model_score = maybe_score_pick(mode, features)
                        .map(|score| score.combined_expected_score)
                        .unwrap_or(0.0);
                    let heuristic = -(d as f64);
                    candidates.push(PickupCandidate {
                        item_id: item.id.clone(),
                        cell,
                        score: heuristic + model_score,
                    });
                }
            }
        }
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cell.cmp(&b.cell))
        });
        candidates.truncate(top_k);
        candidates
    }

    fn intern(&mut self, item_id: &str) -> u16 {
        if let Some(v) = self.item_intern.get(item_id) {
            return *v;
        }
        let id = self.item_rev.len() as u16;
        self.item_rev.push(item_id.to_owned());
        self.item_intern.insert(item_id.to_owned(), id);
        id
    }

    fn build_effective_carried_supply(
        &mut self,
        state: &GameState,
        map: &MapCache,
        dist: &DistanceMap,
        team: &TeamContext,
    ) -> HashMap<u16, u16> {
        const ACTIVE_SUPPLY_NEAR_DROP_MAX: u16 = 10;
        const ACTIVE_SUPPLY_COURIER_DROP_MAX: u16 = 16;
        let mut out: HashMap<u16, u16> = HashMap::new();
        for bot in &state.bots {
            if bot.carrying.is_empty() {
                continue;
            }
            let Some(cell) = map.idx(bot.x, bot.y) else {
                continue;
            };
            let near_drop = map
                .dropoff_cells
                .iter()
                .copied()
                .map(|drop| dist.dist(cell, drop))
                .min()
                .unwrap_or(u16::MAX);
            let role = team.role_for(&bot.id);
            let courier_role = matches!(role, BotRole::LeadCourier | BotRole::QueueCourier);
            let serviceable = near_drop <= ACTIVE_SUPPLY_NEAR_DROP_MAX
                || (courier_role && near_drop <= ACTIVE_SUPPLY_COURIER_DROP_MAX);
            if !serviceable {
                continue;
            }
            for item in &bot.carrying {
                let item_id = self.intern(item);
                *out.entry(item_id).or_insert(0) += 1;
            }
        }
        out
    }
}

fn avg_teammate_distance(bot: &BotState, bots: &[BotState]) -> f64 {
    let mut total = 0.0;
    let mut count = 0.0;
    for other in bots {
        if other.id == bot.id {
            continue;
        }
        total += f64::from((other.x - bot.x).abs() + (other.y - bot.y).abs());
        count += 1.0;
    }
    if count == 0.0 {
        0.0
    } else {
        total / count
    }
}

fn average_pairwise_distance(bots: &[BotState]) -> f64 {
    if bots.len() <= 1 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0.0;
    for i in 0..bots.len() {
        for j in (i + 1)..bots.len() {
            total += f64::from((bots[i].x - bots[j].x).abs() + (bots[i].y - bots[j].y).abs());
            count += 1.0;
        }
    }
    if count == 0.0 {
        0.0
    } else {
        total / count
    }
}

fn nearest_open_cell(
    map: &MapCache,
    target_x: i32,
    target_y: i32,
    reserved: &HashSet<u16>,
) -> Option<u16> {
    let max_radius = map.width.max(map.height).max(1);
    for radius in 0..=max_radius {
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                if dx.abs() + dy.abs() != radius {
                    continue;
                }
                let x = target_x + dx;
                let y = target_y + dy;
                let Some(cell) = map.idx(x, y) else {
                    continue;
                };
                if map.wall_mask[cell as usize] || reserved.contains(&cell) {
                    continue;
                }
                return Some(cell);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        dist::DistanceMap,
        model::{BotState, GameState, Grid, Item, Order, OrderStatus},
        team_context::{TeamContext, TeamContextConfig},
        world::World,
    };

    use super::{Dispatcher, Intent};

    fn medium_like_state() -> GameState {
        GameState {
            tick: 0,
            grid: Grid {
                width: 16,
                height: 12,
                walls: Vec::new(),
                drop_off_tiles: vec![[0, 0]],
            },
            bots: vec![
                BotState {
                    id: "0".to_owned(),
                    x: 1,
                    y: 1,
                    carrying: Vec::new(),
                    capacity: 3,
                },
                BotState {
                    id: "1".to_owned(),
                    x: 1,
                    y: 1,
                    carrying: Vec::new(),
                    capacity: 3,
                },
                BotState {
                    id: "2".to_owned(),
                    x: 1,
                    y: 1,
                    carrying: Vec::new(),
                    capacity: 3,
                },
            ],
            items: vec![
                Item {
                    id: "item_a".to_owned(),
                    kind: "apple".to_owned(),
                    x: 5,
                    y: 2,
                },
                Item {
                    id: "item_b".to_owned(),
                    kind: "apple".to_owned(),
                    x: 7,
                    y: 3,
                },
                Item {
                    id: "item_c".to_owned(),
                    kind: "apple".to_owned(),
                    x: 9,
                    y: 4,
                },
            ],
            orders: vec![
                Order {
                    id: "o1".to_owned(),
                    item_id: "apple".to_owned(),
                    status: OrderStatus::InProgress,
                },
                Order {
                    id: "o2".to_owned(),
                    item_id: "apple".to_owned(),
                    status: OrderStatus::InProgress,
                },
                Order {
                    id: "o3".to_owned(),
                    item_id: "apple".to_owned(),
                    status: OrderStatus::Pending,
                },
            ],
            ..GameState::default()
        }
    }

    #[test]
    fn spawn_stack_produces_non_wait_distribution() {
        let state = medium_like_state();
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let mut dispatcher = Dispatcher::new();
        let team = TeamContext::build(
            &state,
            map,
            &dist,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            TeamContextConfig::default(),
        );

        let intents = dispatcher.build_intents(&state, map, &dist, &HashMap::new(), &team);
        let non_wait = intents
            .iter()
            .filter(|intent| !matches!(intent.intent, Intent::Wait))
            .count();
        assert!(non_wait >= 2, "expected at least two non-wait intents");
    }

    #[test]
    fn never_emits_dropoff_for_pending_order() {
        let state = GameState {
            tick: 25,
            grid: Grid {
                width: 12,
                height: 10,
                drop_off_tiles: vec![[1, 1]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 1,
                y: 1,
                carrying: vec!["milk".to_owned()],
                capacity: 3,
            }],
            items: vec![Item {
                id: "item_1".to_owned(),
                kind: "milk".to_owned(),
                x: 3,
                y: 1,
            }],
            orders: vec![Order {
                id: "o_pending".to_owned(),
                item_id: "milk".to_owned(),
                status: OrderStatus::Pending,
            }],
            ..GameState::default()
        };
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let mut dispatcher = Dispatcher::new();
        let team = TeamContext::build(
            &state,
            map,
            &dist,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            TeamContextConfig::default(),
        );
        let intents = dispatcher.build_intents(&state, map, &dist, &HashMap::new(), &team);
        assert!(
            intents
                .iter()
                .all(|it| !matches!(it.intent, Intent::DropOff { .. })),
            "dispatcher emitted dropoff for pending order"
        );
    }

    #[test]
    fn preview_pickups_disabled_while_active_gap_exists() {
        let state = GameState {
            tick: 7,
            grid: Grid {
                width: 7,
                height: 7,
                drop_off_tiles: vec![[1, 1]],
                walls: vec![[2, 3], [4, 3], [3, 2], [3, 4]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 5,
                y: 4,
                carrying: vec![],
                capacity: 3,
            }],
            items: vec![
                Item {
                    id: "item_active".to_owned(),
                    kind: "milk".to_owned(),
                    x: 3,
                    y: 3,
                },
                Item {
                    id: "item_preview".to_owned(),
                    kind: "bread".to_owned(),
                    x: 5,
                    y: 5,
                },
            ],
            orders: vec![
                Order {
                    id: "o_active".to_owned(),
                    item_id: "milk".to_owned(),
                    status: OrderStatus::InProgress,
                },
                Order {
                    id: "o_preview".to_owned(),
                    item_id: "bread".to_owned(),
                    status: OrderStatus::Pending,
                },
            ],
            ..GameState::default()
        };
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let mut dispatcher = Dispatcher::new();
        let team = TeamContext::build(
            &state,
            map,
            &dist,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            TeamContextConfig::default(),
        );
        let intents = dispatcher.build_intents(&state, map, &dist, &HashMap::new(), &team);
        assert!(
            intents.iter().all(|it| {
                !matches!(
                    &it.intent,
                    Intent::PickUp { item_id } if item_id == "item_preview"
                )
            }),
            "preview pickup should be disabled while active gap exists"
        );
    }
}
