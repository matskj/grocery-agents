pub mod common;
pub mod easy;
pub mod expert;
pub mod hard;
pub mod medium;

use std::collections::HashMap;

use crate::{
    difficulty::Difficulty, dispatcher::Intent, dist::DistanceMap, model::GameState,
    team_context::TeamContext, world::MapCache,
};

#[derive(Debug, Clone, Default)]
pub struct StrategyPlan {
    pub policy_name: &'static str,
    pub strategy_stage: &'static str,
    pub forced_intents: HashMap<String, Intent>,
    pub explicit_order: Vec<String>,
    pub role_label_by_bot: HashMap<String, String>,
    pub preferred_area_by_bot: HashMap<String, u16>,
    pub expansion_mode_by_bot: HashMap<String, bool>,
}

pub struct TickInput<'a> {
    pub difficulty: Difficulty,
    pub state: &'a GameState,
    pub map: &'a MapCache,
    pub dist: &'a DistanceMap,
    pub team: &'a TeamContext,
    pub ticks_since_pickup: u16,
    pub ticks_since_dropoff: u16,
}

pub trait Strategy {
    fn tick(&mut self, input: TickInput<'_>) -> StrategyPlan;
}
