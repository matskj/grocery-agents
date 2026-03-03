use std::path::PathBuf;

use clap::{Args, ValueEnum};

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
pub enum AssignmentMode {
    Hybrid,
    GlobalOnly,
    LegacyOnly,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub horizon: u8,
    pub candidate_k: usize,
    pub assignment_enabled: bool,
    pub assignment_mode: AssignmentMode,
    pub dropoff_scheduling_enabled: bool,
    pub dropoff_window: u8,
    pub dropoff_capacity: u8,
    pub lambda_density: f64,
    pub lambda_choke: f64,
    pub planner_soft_budget_ms: u64,
    pub log_level: String,
    pub structured_bot_log: bool,
    pub ascii_render: bool,
    pub replay_dump_path: Option<PathBuf>,
    pub coord_claim_ttl_ticks: u8,
    pub coord_reassign_no_progress_ticks: u8,
    pub coord_goal_collapse_threshold: usize,
    pub coord_max_bots_per_stand: u8,
    pub coord_post_dropoff_retask_ticks: u8,
    pub coord_area_balance_weight: f64,
}

#[derive(Debug, Clone, Args)]
pub struct ConfigArgs {
    #[arg(long, env = "GROCERY_HORIZON", default_value_t = 16)]
    pub horizon: u8,

    #[arg(long, env = "GROCERY_CANDIDATE_K", default_value_t = 8)]
    pub candidate_k: usize,

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

    #[arg(long, env = "GROCERY_PLANNER_SOFT_BUDGET_MS", default_value_t = 1_200)]
    pub planner_soft_budget_ms: u64,

    #[arg(long, env = "GROCERY_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    #[arg(long, env = "GROCERY_STRUCTURED_BOT_LOG", default_value_t = false)]
    pub structured_bot_log: bool,

    #[arg(long, env = "GROCERY_ASCII_RENDER", default_value_t = false)]
    pub ascii_render: bool,

    #[arg(long, env = "GROCERY_REPLAY_DUMP_PATH")]
    pub replay_dump_path: Option<PathBuf>,

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

    #[arg(
        long,
        env = "GROCERY_COORD_MAX_BOTS_PER_STAND",
        default_value_t = 1
    )]
    pub coord_max_bots_per_stand: u8,

    #[arg(
        long,
        env = "GROCERY_COORD_POST_DROPOFF_RETASK_TICKS",
        default_value_t = 6
    )]
    pub coord_post_dropoff_retask_ticks: u8,

    #[arg(
        long,
        env = "GROCERY_COORD_AREA_BALANCE_WEIGHT",
        default_value_t = 1.0
    )]
    pub coord_area_balance_weight: f64,
}

impl ConfigArgs {
    pub fn build(self) -> Config {
        Config {
            horizon: self.horizon.clamp(8, 32),
            candidate_k: self.candidate_k.clamp(1, 32),
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
            planner_soft_budget_ms: self.planner_soft_budget_ms.clamp(100, 1_390),
            log_level: self.log_level,
            structured_bot_log: self.structured_bot_log,
            ascii_render: self.ascii_render,
            replay_dump_path: self.replay_dump_path,
            coord_claim_ttl_ticks: self.coord_claim_ttl_ticks.clamp(2, 60),
            coord_reassign_no_progress_ticks: self
                .coord_reassign_no_progress_ticks
                .clamp(2, 64),
            coord_goal_collapse_threshold: self.coord_goal_collapse_threshold.clamp(2, 32),
            coord_max_bots_per_stand: self.coord_max_bots_per_stand.clamp(1, 3),
            coord_post_dropoff_retask_ticks: self.coord_post_dropoff_retask_ticks.clamp(1, 24),
            coord_area_balance_weight: self.coord_area_balance_weight.clamp(0.0, 10.0),
        }
    }
}
