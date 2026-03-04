use std::path::PathBuf;

use clap::{Args, ValueEnum};

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
pub enum AssignmentMode {
    Hybrid,
    GlobalOnly,
    LegacyOnly,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
pub enum PlannerBudgetMode {
    Fixed,
    Adaptive,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
pub enum PolicyMode {
    Auto,
    Easy,
    Medium,
    Hard,
    Expert,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub horizon: u8,
    pub candidate_k: usize,
    pub policy_mode: PolicyMode,
    pub assignment_enabled: bool,
    pub assignment_mode: AssignmentMode,
    pub dropoff_scheduling_enabled: bool,
    pub dropoff_window: u8,
    pub dropoff_capacity: u8,
    pub lambda_density: f64,
    pub lambda_choke: f64,
    pub planner_budget_mode: PlannerBudgetMode,
    pub planner_soft_budget_ms: u64,
    pub planner_soft_budget_min_ms: u64,
    pub planner_soft_budget_max_ms: u64,
    pub planner_hard_budget_ms: u64,
    pub planner_deadline_slack_ms: u64,
    pub tick_soft_budget_ms: u64,
    pub tick_hard_budget_ms: u64,
    pub tick_greedy_fallback_ms: u64,
    pub log_level: String,
    pub structured_bot_log: bool,
    pub ascii_render: bool,
    pub debug: bool,
    pub replay_dump_path: Option<PathBuf>,
    pub coord_claim_ttl_ticks: u8,
    pub coord_reassign_no_progress_ticks: u8,
    pub coord_goal_collapse_threshold: usize,
    pub coord_max_bots_per_stand: u8,
    pub coord_post_dropoff_retask_ticks: u8,
    pub coord_area_balance_weight: f64,
    pub coord_local_radius_base: u8,
    pub coord_local_radius_max: u8,
    pub coord_expansion_stall_ticks: u8,
    pub coord_preferred_area_ttl_ticks: u8,
    pub coord_out_of_area_penalty: f64,
    pub coord_out_of_radius_penalty: f64,
}

#[derive(Debug, Clone, Args)]
pub struct ConfigArgs {
    #[arg(long, env = "GROCERY_HORIZON", default_value_t = 16)]
    pub horizon: u8,

    #[arg(long, env = "GROCERY_CANDIDATE_K", default_value_t = 8)]
    pub candidate_k: usize,

    #[arg(
        long = "policy",
        alias = "policy-mode",
        env = "GROCERY_POLICY",
        value_enum,
        default_value_t = PolicyMode::Auto
    )]
    pub policy_mode: PolicyMode,

    #[arg(long, env = "GROCERY_ASSIGNMENT_ENABLED", default_value_t = true)]
    pub assignment_enabled: bool,

    #[arg(
        long,
        env = "GROCERY_ASSIGNMENT_MODE",
        value_enum,
        default_value_t = AssignmentMode::Hybrid
    )]
    pub assignment_mode: AssignmentMode,

    #[arg(
        long,
        env = "GROCERY_DROPOFF_SCHEDULING_ENABLED",
        default_value_t = true
    )]
    pub dropoff_scheduling_enabled: bool,

    #[arg(long, env = "GROCERY_DROPOFF_WINDOW", default_value_t = 12)]
    pub dropoff_window: u8,

    #[arg(long, env = "GROCERY_DROPOFF_CAPACITY", default_value_t = 1)]
    pub dropoff_capacity: u8,

    #[arg(long, env = "GROCERY_LAMBDA_DENSITY", default_value_t = 1.0)]
    pub lambda_density: f64,

    #[arg(long, env = "GROCERY_LAMBDA_CHOKE", default_value_t = 1.5)]
    pub lambda_choke: f64,

    #[arg(
        long,
        env = "GROCERY_PLANNER_BUDGET_MODE",
        value_enum,
        default_value_t = PlannerBudgetMode::Adaptive
    )]
    pub planner_budget_mode: PlannerBudgetMode,

    #[arg(long, env = "GROCERY_PLANNER_SOFT_BUDGET_MS", default_value_t = 1_450)]
    pub planner_soft_budget_ms: u64,

    #[arg(
        long,
        env = "GROCERY_PLANNER_SOFT_BUDGET_MIN_MS",
        default_value_t = 1_450
    )]
    pub planner_soft_budget_min_ms: u64,

    #[arg(
        long,
        env = "GROCERY_PLANNER_SOFT_BUDGET_MAX_MS",
        default_value_t = 1_930
    )]
    pub planner_soft_budget_max_ms: u64,

    #[arg(long, env = "GROCERY_PLANNER_HARD_BUDGET_MS", default_value_t = 1_980)]
    pub planner_hard_budget_ms: u64,

    #[arg(long, env = "GROCERY_PLANNER_DEADLINE_SLACK_MS", default_value_t = 40)]
    pub planner_deadline_slack_ms: u64,

    #[arg(long, env = "GROCERY_TICK_SOFT_BUDGET_MS", default_value_t = 50)]
    pub tick_soft_budget_ms: u64,

    #[arg(long, env = "GROCERY_TICK_HARD_BUDGET_MS", default_value_t = 150)]
    pub tick_hard_budget_ms: u64,

    #[arg(long, env = "GROCERY_TICK_GREEDY_FALLBACK_MS", default_value_t = 8)]
    pub tick_greedy_fallback_ms: u64,

    #[arg(long, env = "GROCERY_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    #[arg(long, env = "GROCERY_STRUCTURED_BOT_LOG", default_value_t = false)]
    pub structured_bot_log: bool,

    #[arg(long, env = "GROCERY_ASCII_RENDER", default_value_t = false)]
    pub ascii_render: bool,

    #[arg(long, env = "GROCERY_DEBUG", default_value_t = false)]
    pub debug: bool,

    #[arg(long, env = "GROCERY_REPLAY_DUMP_PATH")]
    pub replay_dump_path: Option<PathBuf>,

    #[arg(long = "record", alias = "record-path", env = "GROCERY_RECORD_PATH")]
    pub record_path: Option<PathBuf>,

    #[arg(long, env = "GROCERY_COORD_CLAIM_TTL_TICKS", default_value_t = 10)]
    pub coord_claim_ttl_ticks: u8,

    #[arg(
        long,
        env = "GROCERY_COORD_REASSIGN_NO_PROGRESS_TICKS",
        default_value_t = 8
    )]
    pub coord_reassign_no_progress_ticks: u8,

    #[arg(
        long,
        env = "GROCERY_COORD_GOAL_COLLAPSE_THRESHOLD",
        default_value_t = 4
    )]
    pub coord_goal_collapse_threshold: usize,

    #[arg(long, env = "GROCERY_COORD_MAX_BOTS_PER_STAND", default_value_t = 1)]
    pub coord_max_bots_per_stand: u8,

    #[arg(
        long,
        env = "GROCERY_COORD_POST_DROPOFF_RETASK_TICKS",
        default_value_t = 6
    )]
    pub coord_post_dropoff_retask_ticks: u8,

    #[arg(long, env = "GROCERY_COORD_AREA_BALANCE_WEIGHT", default_value_t = 1.0)]
    pub coord_area_balance_weight: f64,

    #[arg(long, env = "GROCERY_COORD_LOCAL_RADIUS_BASE", default_value_t = 8)]
    pub coord_local_radius_base: u8,

    #[arg(long, env = "GROCERY_COORD_LOCAL_RADIUS_MAX", default_value_t = 14)]
    pub coord_local_radius_max: u8,

    #[arg(
        long,
        env = "GROCERY_COORD_EXPANSION_STALL_TICKS",
        default_value_t = 10
    )]
    pub coord_expansion_stall_ticks: u8,

    #[arg(
        long,
        env = "GROCERY_COORD_PREFERRED_AREA_TTL_TICKS",
        default_value_t = 10
    )]
    pub coord_preferred_area_ttl_ticks: u8,

    #[arg(
        long,
        env = "GROCERY_COORD_OUT_OF_AREA_PENALTY",
        default_value_t = 28.0
    )]
    pub coord_out_of_area_penalty: f64,

    #[arg(
        long,
        env = "GROCERY_COORD_OUT_OF_RADIUS_PENALTY",
        default_value_t = 45.0
    )]
    pub coord_out_of_radius_penalty: f64,
}

impl ConfigArgs {
    pub fn build(self) -> Config {
        let hard_budget = self.planner_hard_budget_ms.clamp(200, 1_980);
        let slack = self.planner_deadline_slack_ms.clamp(20, 250);
        let mut soft_min = self.planner_soft_budget_min_ms.clamp(100, 1_900);
        let mut soft_max = self.planner_soft_budget_max_ms.clamp(100, 1_950);
        if soft_min > soft_max {
            std::mem::swap(&mut soft_min, &mut soft_max);
        }
        let max_soft_allowed = hard_budget.saturating_sub(slack).max(100);
        soft_min = soft_min.min(max_soft_allowed);
        soft_max = soft_max.min(max_soft_allowed).max(soft_min);
        let tick_hard = self.tick_hard_budget_ms.clamp(20, 500);
        let tick_soft = self.tick_soft_budget_ms.clamp(10, tick_hard);
        let tick_greedy = self.tick_greedy_fallback_ms.clamp(1, tick_soft);
        let replay_dump_path = self.replay_dump_path.or(self.record_path);
        let local_radius_base = self.coord_local_radius_base.clamp(4, 24);
        let local_radius_max = self.coord_local_radius_max.clamp(local_radius_base, 32);

        Config {
            horizon: self.horizon.clamp(8, 32),
            candidate_k: self.candidate_k.clamp(1, 32),
            policy_mode: self.policy_mode,
            assignment_enabled: self.assignment_enabled,
            assignment_mode: if self.assignment_enabled {
                self.assignment_mode
            } else {
                AssignmentMode::LegacyOnly
            },
            dropoff_scheduling_enabled: self.dropoff_scheduling_enabled,
            dropoff_window: self.dropoff_window.clamp(4, 32),
            dropoff_capacity: self.dropoff_capacity.clamp(1, 4),
            lambda_density: self.lambda_density.clamp(0.0, 10.0),
            lambda_choke: self.lambda_choke.clamp(0.0, 10.0),
            planner_budget_mode: self.planner_budget_mode,
            planner_soft_budget_ms: self.planner_soft_budget_ms.clamp(100, max_soft_allowed),
            planner_soft_budget_min_ms: soft_min,
            planner_soft_budget_max_ms: soft_max,
            planner_hard_budget_ms: hard_budget,
            planner_deadline_slack_ms: slack,
            tick_soft_budget_ms: tick_soft,
            tick_hard_budget_ms: tick_hard,
            tick_greedy_fallback_ms: tick_greedy,
            log_level: self.log_level,
            structured_bot_log: self.structured_bot_log,
            ascii_render: self.ascii_render,
            debug: self.debug,
            replay_dump_path,
            coord_claim_ttl_ticks: self.coord_claim_ttl_ticks.clamp(2, 60),
            coord_reassign_no_progress_ticks: self.coord_reassign_no_progress_ticks.clamp(2, 64),
            coord_goal_collapse_threshold: self.coord_goal_collapse_threshold.clamp(2, 32),
            coord_max_bots_per_stand: self.coord_max_bots_per_stand.clamp(1, 3),
            coord_post_dropoff_retask_ticks: self.coord_post_dropoff_retask_ticks.clamp(1, 24),
            coord_area_balance_weight: self.coord_area_balance_weight.clamp(0.0, 10.0),
            coord_local_radius_base: local_radius_base,
            coord_local_radius_max: local_radius_max,
            coord_expansion_stall_ticks: self.coord_expansion_stall_ticks.clamp(2, 60),
            coord_preferred_area_ttl_ticks: self.coord_preferred_area_ttl_ticks.clamp(2, 60),
            coord_out_of_area_penalty: self.coord_out_of_area_penalty.clamp(0.0, 200.0),
            coord_out_of_radius_penalty: self.coord_out_of_radius_penalty.clamp(0.0, 200.0),
        }
    }
}
