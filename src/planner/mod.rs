use std::collections::HashMap;

use crate::model::GameState;

pub mod common;
pub mod easy;
pub mod expert;
pub mod hard;
pub mod mapf;
pub mod medium;

#[derive(Debug, Clone)]
pub enum Intent {
    DropOff { order_id: String },
    PickUp { item_id: String },
    MoveTo { cell: u16 },
    Wait,
}

#[derive(Debug, Clone)]
pub struct PlanResult {
    pub intents: HashMap<String, Intent>,
    pub explicit_priority: Vec<String>,
    pub role_label_by_bot: HashMap<String, String>,
    pub preferred_area_by_bot: HashMap<String, u16>,
    pub expansion_mode_by_bot: HashMap<String, bool>,
    pub local_radius_by_bot: HashMap<String, u16>,
    pub goal_cell_by_bot: HashMap<String, u16>,
    pub strategy_stage: &'static str,
    pub assignment_source: &'static str,
    pub assignment_guard_reason: &'static str,
}

impl PlanResult {
    pub fn empty(stage: &'static str) -> Self {
        Self {
            intents: HashMap::new(),
            explicit_priority: Vec::new(),
            role_label_by_bot: HashMap::new(),
            preferred_area_by_bot: HashMap::new(),
            expansion_mode_by_bot: HashMap::new(),
            local_radius_by_bot: HashMap::new(),
            goal_cell_by_bot: HashMap::new(),
            strategy_stage: stage,
            assignment_source: "difficulty_planner",
            assignment_guard_reason: "none",
        }
    }
}

#[derive(Clone, Copy)]
pub struct TickContext<'a> {
    pub state: &'a GameState,
    pub map: &'a crate::world::MapCache,
    pub dist: &'a crate::dist::DistanceMap,
    pub tick: u64,
}
