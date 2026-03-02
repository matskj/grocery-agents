use std::collections::HashMap;

use crate::{
    dist::DistanceMap,
    model::{BotState, GameState, OrderStatus},
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
    ) -> Vec<BotIntent> {
        let (active_missing, preview_missing) = self.compute_missing(state);
        let mut intents = Vec::with_capacity(state.bots.len());
        let mut used_pick_targets: Vec<u16> = Vec::new();

        for bot in &state.bots {
            let bot_idx = map.idx(bot.x, bot.y).unwrap_or(0);

            if !bot.carrying.is_empty() && map.dropoff_cells.contains(&bot_idx) {
                if let Some(order) = state.orders.iter().find(|o| {
                    bot.carrying.iter().any(|c| c == &o.item_id)
                        && !matches!(o.status, OrderStatus::Delivered)
                }) {
                    intents.push(BotIntent {
                        bot_id: bot.id.clone(),
                        intent: Intent::DropOff {
                            order_id: order.id.clone(),
                        },
                    });
                    continue;
                }
            }

            let useful_carried = bot.carrying.iter().any(|c| {
                self.item_intern
                    .get(c)
                    .map(|id| active_missing.contains_key(id))
                    .unwrap_or(false)
            });

            if useful_carried {
                if let Some(&nearest_drop) = map
                    .dropoff_cells
                    .iter()
                    .min_by_key(|&&d| dist.dist(bot_idx, d))
                {
                    intents.push(BotIntent {
                        bot_id: bot.id.clone(),
                        intent: Intent::MoveTo { cell: nearest_drop },
                    });
                    continue;
                }
            }

            let mut picked = None;
            if bot.carrying.len() < bot.capacity {
                picked = self
                    .choose_pickup_target(bot, map, dist, &active_missing, 3)
                    .or_else(|| self.choose_pickup_target(bot, map, dist, &preview_missing, 2));
            }

            if let Some((item_id, cell)) = picked {
                if !used_pick_targets.contains(&cell) {
                    used_pick_targets.push(cell);
                    if cell == bot_idx {
                        intents.push(BotIntent {
                            bot_id: bot.id.clone(),
                            intent: Intent::PickUp { item_id },
                        });
                    } else {
                        intents.push(BotIntent {
                            bot_id: bot.id.clone(),
                            intent: Intent::MoveTo { cell },
                        });
                    }
                    continue;
                }
            }

            intents.push(BotIntent {
                bot_id: bot.id.clone(),
                intent: Intent::Wait,
            });
        }

        intents
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

    fn choose_pickup_target(
        &self,
        bot: &BotState,
        map: &MapCache,
        dist: &DistanceMap,
        need: &HashMap<u16, u16>,
        batch_cap: usize,
    ) -> Option<(String, u16)> {
        let bot_idx = map.idx(bot.x, bot.y)?;
        let mut candidates: Vec<(String, u16, u16)> = Vec::new();
        for (item_id, item_ix) in &map.item_by_id {
            let interned = match self.item_intern.get(item_id) {
                Some(v) => *v,
                None => continue,
            };
            if !need.contains_key(&interned) {
                continue;
            }
            for &cell in &map.item_stand_cells[*item_ix] {
                let d = dist.dist(bot_idx, cell);
                if d != u16::MAX {
                    candidates.push((item_id.clone(), cell, d));
                }
            }
        }
        candidates.sort_by_key(|c| (c.2, c.1));
        if candidates.is_empty() {
            return None;
        }

        // tiny batching heuristic over top candidates.
        let max_take = candidates.len().min(batch_cap);
        let mut best = (u32::MAX, 0usize);
        for i in 0..max_take {
            let mut score = candidates[i].2 as u32;
            let mut prev = candidates[i].1;
            for j in 0..max_take {
                if i == j {
                    continue;
                }
                score = score.saturating_add(dist.dist(prev, candidates[j].1) as u32);
                prev = candidates[j].1;
            }
            if score < best.0 {
                best = (score, i);
            }
        }
        let choice = &candidates[best.1];
        Some((choice.0.clone(), choice.1))
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
}
