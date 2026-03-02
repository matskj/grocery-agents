use std::{collections::HashMap, sync::OnceLock};

use tracing::info;

use crate::{
    dispatcher::{BotIntent, Dispatcher, Intent},
    dist::DistanceMap,
    model::{Action, GameState},
    motion::MotionPlanner,
    world::World,
};

#[derive(Debug, Default)]
struct BotMemory {
    prev_cell: Option<u16>,
    blocked_ticks: u8,
}

#[derive(Debug)]
pub struct Policy {
    dispatcher: Dispatcher,
    planner: MotionPlanner,
    memory: HashMap<String, BotMemory>,
}

impl Policy {
    pub fn new() -> Self {
        Self {
            dispatcher: Dispatcher::new(),
            planner: MotionPlanner::new(16),
            memory: HashMap::new(),
        }
    }

    pub fn decide_round(&mut self, state: &GameState) -> Vec<Action> {
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);

        let intents = self.dispatcher.build_intents(state, map, &dist);
        let mut goals = HashMap::new();
        let mut immediate = HashMap::new();

        for BotIntent { bot_id, intent } in intents {
            let bot = match state.bots.iter().find(|b| b.id == bot_id) {
                Some(b) => b,
                None => continue,
            };
            if bot_debug_enabled() {
                info!(tick = state.tick, bot_id = %bot.id, intent = ?intent, "assignment selected");
            }
            let cell = map.idx(bot.x, bot.y).unwrap_or(0);
            let mem = self.memory.entry(bot.id.clone()).or_default();
            if mem.prev_cell == Some(cell) {
                mem.blocked_ticks = mem.blocked_ticks.saturating_add(1);
            } else {
                mem.blocked_ticks = 0;
            }
            if bot_debug_enabled() && mem.blocked_ticks >= 2 {
                info!(
                    tick = state.tick,
                    bot_id = %bot.id,
                    blocked_ticks = mem.blocked_ticks,
                    cell,
                    "congestion/jam detected"
                );
            }
            mem.prev_cell = Some(cell);

            match intent {
                Intent::DropOff { order_id } => {
                    immediate.insert(
                        bot.id.clone(),
                        Action::DropOff {
                            bot_id: bot.id.clone(),
                            order_id,
                        },
                    );
                }
                Intent::PickUp { item_id } => {
                    immediate.insert(
                        bot.id.clone(),
                        Action::PickUp {
                            bot_id: bot.id.clone(),
                            item_id,
                        },
                    );
                }
                Intent::MoveTo { mut cell } => {
                    // drop-off ring shaping: cap direct entrants near drop-off congestion.
                    if map.dropoff_cells.contains(&cell) {
                        let crowded = state
                            .bots
                            .iter()
                            .filter(|b| {
                                map.idx(b.x, b.y)
                                    .map(|bi| dist.dist(bi, cell) <= 1)
                                    .unwrap_or(false)
                            })
                            .count();
                        if crowded > 2 {
                            if bot_debug_enabled() {
                                info!(
                                    tick = state.tick,
                                    bot_id = %bot.id,
                                    target_cell = cell,
                                    crowded,
                                    "drop-off congestion detected; rerouting toward ring"
                                );
                            }
                            if let Some(&ring) = map.neighbors[cell as usize].first() {
                                cell = ring;
                            }
                        }
                    }
                    if mem.blocked_ticks >= 2 {
                        if let Some(&evac) = map.neighbors[cell as usize].first() {
                            if bot_debug_enabled() {
                                info!(
                                    tick = state.tick,
                                    bot_id = %bot.id,
                                    from_cell = cell,
                                    evac_cell = evac,
                                    "evacuation trigger fired"
                                );
                            }
                            goals.insert(bot.id.clone(), evac);
                        } else {
                            goals.insert(bot.id.clone(), cell);
                        }
                    } else {
                        goals.insert(bot.id.clone(), cell);
                    }
                }
                Intent::Wait => {
                    immediate.insert(bot.id.clone(), Action::wait(bot.id.clone()));
                }
            }
        }

        let mut planned = self.planner.plan(state, map, &dist, &goals);
        for (bot_id, action) in immediate {
            planned.insert(bot_id, action);
        }

        state
            .bots
            .iter()
            .map(|b| {
                planned
                    .remove(&b.id)
                    .unwrap_or_else(|| Action::wait(b.id.clone()))
            })
            .collect()
    }
}

fn bot_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("BOT_DEBUG")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    })
}
