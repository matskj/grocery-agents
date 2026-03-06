use std::path::PathBuf;

use clap::{Args, ValueEnum};

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
    pub policy_mode: PolicyMode,
    pub planner_budget_mode: PlannerBudgetMode,
    pub planner_soft_budget_ms: u64,
    pub planner_soft_budget_min_ms: u64,
    pub planner_soft_budget_max_ms: u64,
    pub planner_hard_budget_ms: u64,
    pub planner_deadline_slack_ms: u64,
    pub tick_soft_budget_ms: u64,
    pub tick_hard_budget_ms: u64,
    pub tick_greedy_fallback_ms: u64,
    pub cache_reuse_max_age_ticks: u64,
    pub cache_require_progress: bool,
    pub log_level: String,
    pub structured_bot_log: bool,
    pub ascii_render: bool,
    pub debug: bool,
    pub replay_dump_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct ConfigArgs {
    #[arg(
        long = "policy",
        alias = "policy-mode",
        env = "GROCERY_POLICY",
        value_enum,
        default_value_t = PolicyMode::Auto
    )]
    pub policy_mode: PolicyMode,

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

    #[arg(long, env = "GROCERY_CACHE_REUSE_MAX_AGE_TICKS", default_value_t = 1)]
    pub cache_reuse_max_age_ticks: u64,

    #[arg(long, env = "GROCERY_CACHE_REQUIRE_PROGRESS", default_value_t = true)]
    pub cache_require_progress: bool,

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

        Config {
            policy_mode: self.policy_mode,
            planner_budget_mode: self.planner_budget_mode,
            planner_soft_budget_ms: self.planner_soft_budget_ms.clamp(100, max_soft_allowed),
            planner_soft_budget_min_ms: soft_min,
            planner_soft_budget_max_ms: soft_max,
            planner_hard_budget_ms: hard_budget,
            planner_deadline_slack_ms: slack,
            tick_soft_budget_ms: tick_soft,
            tick_hard_budget_ms: tick_hard,
            tick_greedy_fallback_ms: tick_greedy,
            cache_reuse_max_age_ticks: self.cache_reuse_max_age_ticks.clamp(0, 8),
            cache_require_progress: self.cache_require_progress,
            log_level: self.log_level,
            structured_bot_log: self.structured_bot_log,
            ascii_render: self.ascii_render,
            debug: self.debug,
            replay_dump_path: self.replay_dump_path.or(self.record_path),
        }
    }
}
