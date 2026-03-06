use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::Duration,
};

use crate::{
    config::{Config, PolicyMode},
    difficulty::{infer_difficulty, Difficulty},
    dist::DistanceMap,
    model::{Action, GameState},
    planner::{
        common::{
            active_kind_counts, bot_cell, intent_goal_cell, queue_role_from_label, region_for_cell,
        },
        easy::EasyPlanner,
        expert::ExpertPlanner,
        hard::HardPlanner,
        mapf::resolve_one_step,
        medium::MediumPlanner,
        Intent, PlanResult, TickContext,
    },
    world::World,
};

#[derive(Debug, Clone, Default)]
struct BotMemory {
    prev_cell: Option<u16>,
    blocked_ticks: u8,
    recent_cells: VecDeque<u16>,
}

#[derive(Debug)]
pub struct Policy {
    config: Arc<Config>,
    last_team_telemetry: serde_json::Value,
    bot_memory: HashMap<String, BotMemory>,
    easy: EasyPlanner,
    medium: MediumPlanner,
    hard: HardPlanner,
    expert: ExpertPlanner,
}

impl Policy {
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            config,
            last_team_telemetry: serde_json::json!({}),
            bot_memory: HashMap::new(),
            easy: EasyPlanner::default(),
            medium: MediumPlanner::default(),
            hard: HardPlanner::default(),
            expert: ExpertPlanner::default(),
        }
    }

    pub fn decide_round(&mut self, state: &GameState, _soft_budget: Duration) -> Vec<Action> {
        let world = World::new(state);
        let map = world.map();
        let dist = DistanceMap::shared_for(map);

        self.update_bot_memory(state, map);
        let difficulty = self.selected_difficulty(state);
        let ctx = TickContext {
            state,
            map,
            dist: dist.as_ref(),
            tick: state.tick,
        };
        let plan = self.run_planner(difficulty, ctx);

        let mut priority = plan.explicit_priority.clone();
        for bot in &state.bots {
            if !priority.iter().any(|id| id == &bot.id) {
                priority.push(bot.id.clone());
            }
        }

        let lookahead = match difficulty {
            Difficulty::Easy => 2,
            Difficulty::Medium => 4,
            Difficulty::Hard => 6,
            Difficulty::Expert => 5,
            Difficulty::Custom => 4,
        };

        let mapf = resolve_one_step(
            state,
            map,
            dist.as_ref(),
            &plan.intents,
            &priority,
            lookahead,
        );

        let mut actions = Vec::with_capacity(state.bots.len());
        let mut wait_reason_by_bot = serde_json::Map::<String, serde_json::Value>::new();
        let mut ordering_stage_by_bot = serde_json::Map::<String, serde_json::Value>::new();
        let mut planner_stage_by_bot = serde_json::Map::<String, serde_json::Value>::new();
        let mut intent_move_but_wait_by_bot = serde_json::Map::<String, serde_json::Value>::new();
        let mut queue_roles = serde_json::Map::<String, serde_json::Value>::new();

        for bot in &state.bots {
            let action = mapf
                .actions_by_bot
                .get(&bot.id)
                .cloned()
                .unwrap_or_else(|| Action::wait(bot.id.clone()));
            let was_wait = matches!(action, Action::Wait { .. });
            let intended_move = matches!(plan.intents.get(&bot.id), Some(Intent::MoveTo { .. }));
            intent_move_but_wait_by_bot.insert(
                bot.id.clone(),
                serde_json::Value::Bool(was_wait && intended_move),
            );
            let wait_reason = if was_wait {
                if intended_move {
                    "no_path_with_constraints"
                } else {
                    "idle"
                }
            } else {
                "none"
            };
            wait_reason_by_bot.insert(
                bot.id.clone(),
                serde_json::Value::String(wait_reason.to_owned()),
            );
            ordering_stage_by_bot.insert(
                bot.id.clone(),
                serde_json::Value::String(plan.strategy_stage.to_owned()),
            );
            planner_stage_by_bot.insert(
                bot.id.clone(),
                serde_json::Value::String(plan.strategy_stage.to_owned()),
            );
            let role = plan
                .role_label_by_bot
                .get(&bot.id)
                .map(String::as_str)
                .unwrap_or("collector");
            queue_roles.insert(
                bot.id.clone(),
                serde_json::Value::String(queue_role_from_label(role)),
            );
            actions.push(action);
        }

        let blocked_bot_count = self
            .bot_memory
            .values()
            .filter(|memory| memory.blocked_ticks > 0)
            .count() as u64;
        let stuck_bot_count = self
            .bot_memory
            .values()
            .filter(|memory| loop_two_cycle(memory))
            .count() as u64;

        let local_conflicts = to_u64_map(&mapf.conflict_count_by_bot);
        let preferred_area = to_u16_map(&plan.preferred_area_by_bot);
        let expansion_map = to_bool_map(&plan.expansion_mode_by_bot);
        let local_radius = to_u16_map(&plan.local_radius_by_bot);
        let goal_cells = to_u16_map(&plan.goal_cell_by_bot);
        let goal_areas = plan
            .goal_cell_by_bot
            .iter()
            .map(|(bot_id, cell)| {
                let regions = if matches!(difficulty, Difficulty::Expert) {
                    5
                } else if matches!(difficulty, Difficulty::Hard) {
                    4
                } else {
                    1
                };
                (
                    bot_id.clone(),
                    serde_json::Value::Number(serde_json::Number::from(region_for_cell(
                        map, *cell, regions,
                    ))),
                )
            })
            .collect::<serde_json::Map<_, _>>();

        let active_counts = active_kind_counts(state);
        let mut claimed_item_type_counts = HashMap::<String, u16>::new();
        for intent in plan.intents.values() {
            if let Intent::PickUp { item_id } = intent {
                if let Some(item) = state.items.iter().find(|item| item.id == *item_id) {
                    *claimed_item_type_counts
                        .entry(item.kind.clone())
                        .or_insert(0) += 1;
                }
            }
        }

        let goal_concentration_top3 = goal_concentration_top3(&plan.intents);
        let unique_goal_cells = plan
            .intents
            .values()
            .filter_map(intent_goal_cell)
            .collect::<HashSet<_>>()
            .len() as u64;

        self.last_team_telemetry = serde_json::json!({
            "assignment_source": plan.assignment_source,
            "assignment_guard_reason": plan.assignment_guard_reason,
            "assignment_goal_concentration_top3": goal_concentration_top3,
            "assignment_active_missing_total": active_counts.values().copied().map(u64::from).sum::<u64>(),
            "blocked_bot_count": blocked_bot_count,
            "stuck_bot_count": stuck_bot_count,
            "assign_ms": 0,
            "queue_roles": queue_roles,
            "wait_reason_by_bot": wait_reason_by_bot,
            "ordering_stage_by_bot": ordering_stage_by_bot,
            "planner_fallback_stage_by_bot": planner_stage_by_bot,
            "intent_move_but_wait_by_bot": intent_move_but_wait_by_bot,
            "queue_relaxation_active_by_bot": empty_bool_map(state),
            "local_conflict_count_by_bot": serde_json::Value::Object(local_conflicts),
            "preferred_area_id_by_bot": serde_json::Value::Object(preferred_area),
            "expansion_mode_by_bot": serde_json::Value::Object(expansion_map),
            "local_radius_by_bot": serde_json::Value::Object(local_radius),
            "goal_area_id_by_bot": serde_json::Value::Object(goal_areas),
            "goal_cell_by_bot": serde_json::Value::Object(goal_cells),
            "dropoff_target_status_by_bot": dropoff_target_status(state, &plan.intents),
            "strategy_role_counts": strategy_role_counts(&plan.role_label_by_bot),
            "claimed_item_type_counts": claimed_item_type_counts,
            "reserved_cells_by_t": mapf.reserved_cells_by_t,
            "unique_goal_cells_last_n": unique_goal_cells,
            "fallback_level": "none"
        });

        actions
    }

    pub fn last_team_telemetry(&self) -> serde_json::Value {
        self.last_team_telemetry.clone()
    }

    fn selected_difficulty(&self, state: &GameState) -> Difficulty {
        match self.config.policy_mode {
            PolicyMode::Auto => infer_difficulty(state),
            PolicyMode::Easy => Difficulty::Easy,
            PolicyMode::Medium => Difficulty::Medium,
            PolicyMode::Hard => Difficulty::Hard,
            PolicyMode::Expert => Difficulty::Expert,
        }
    }

    fn run_planner(&mut self, difficulty: Difficulty, input: TickContext<'_>) -> PlanResult {
        match difficulty {
            Difficulty::Easy => self.easy.tick(input),
            Difficulty::Medium => self.medium.tick(input),
            Difficulty::Hard => self.hard.tick(input),
            Difficulty::Expert => self.expert.tick(input),
            Difficulty::Custom => self.medium.tick(input),
        }
    }

    fn update_bot_memory(&mut self, state: &GameState, map: &crate::world::MapCache) {
        for bot in &state.bots {
            let memory = self.bot_memory.entry(bot.id.clone()).or_default();
            let current = bot_cell(map, bot);
            if let Some(cell) = current {
                if memory.prev_cell == Some(cell) {
                    memory.blocked_ticks = memory.blocked_ticks.saturating_add(1);
                } else {
                    memory.blocked_ticks = 0;
                }
                memory.recent_cells.push_back(cell);
                while memory.recent_cells.len() > 6 {
                    memory.recent_cells.pop_front();
                }
                memory.prev_cell = Some(cell);
            }
        }
        let live = state
            .bots
            .iter()
            .map(|bot| bot.id.clone())
            .collect::<HashSet<_>>();
        self.bot_memory.retain(|bot_id, _| live.contains(bot_id));
    }
}

fn to_u64_map(values: &HashMap<String, u64>) -> serde_json::Map<String, serde_json::Value> {
    values
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                serde_json::Value::Number(serde_json::Number::from(*v)),
            )
        })
        .collect::<serde_json::Map<_, _>>()
}

fn to_u16_map(values: &HashMap<String, u16>) -> serde_json::Map<String, serde_json::Value> {
    values
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                serde_json::Value::Number(serde_json::Number::from(*v)),
            )
        })
        .collect::<serde_json::Map<_, _>>()
}

fn to_bool_map(values: &HashMap<String, bool>) -> serde_json::Map<String, serde_json::Value> {
    values
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::Bool(*v)))
        .collect::<serde_json::Map<_, _>>()
}

fn empty_bool_map(state: &GameState) -> serde_json::Map<String, serde_json::Value> {
    state
        .bots
        .iter()
        .map(|bot| (bot.id.clone(), serde_json::Value::Bool(false)))
        .collect::<serde_json::Map<_, _>>()
}

fn dropoff_target_status(
    state: &GameState,
    intents: &HashMap<String, Intent>,
) -> serde_json::Map<String, serde_json::Value> {
    state
        .bots
        .iter()
        .map(|bot| {
            let value = match intents.get(&bot.id) {
                Some(Intent::DropOff { .. }) => "active_dropoff",
                _ => "none",
            };
            (bot.id.clone(), serde_json::Value::String(value.to_owned()))
        })
        .collect::<serde_json::Map<_, _>>()
}

fn strategy_role_counts(role_label_by_bot: &HashMap<String, String>) -> serde_json::Value {
    let mut counts = HashMap::<String, u64>::new();
    for label in role_label_by_bot.values() {
        *counts.entry(label.clone()).or_insert(0) += 1;
    }
    serde_json::json!(counts)
}

fn goal_concentration_top3(intents: &HashMap<String, Intent>) -> f64 {
    let mut counts = HashMap::<u16, u64>::new();
    for intent in intents.values() {
        if let Some(cell) = intent_goal_cell(intent) {
            *counts.entry(cell).or_insert(0) += 1;
        }
    }
    let mut values = counts.values().copied().collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| b.cmp(a));
    let top3 = values.into_iter().take(3).sum::<u64>();
    let total = counts.values().copied().sum::<u64>().max(1);
    top3 as f64 / total as f64
}

fn loop_two_cycle(memory: &BotMemory) -> bool {
    if memory.recent_cells.len() < 4 {
        return false;
    }
    let cells = memory.recent_cells.iter().copied().collect::<Vec<_>>();
    let n = cells.len();
    cells[n - 1] == cells[n - 3] && cells[n - 2] == cells[n - 4]
}
