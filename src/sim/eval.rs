use std::{path::Path, sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};

use crate::{
    config::{Config, PlannerBudgetMode, PolicyMode},
    policy::Policy,
    replay::{list_runs, load_run},
    sim::{scenario::sim_from_state, step::step_many},
};

#[derive(Debug, Clone)]
pub struct EvalRequest<'a> {
    pub logs_dir: &'a Path,
    pub episodes: usize,
    pub policy: PolicyMode,
    pub difficulty: Option<&'a str>,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalEpisode {
    pub run_id: String,
    pub mode: Option<String>,
    pub score: i64,
    pub items_delivered: u64,
    pub orders_completed: u64,
    pub ticks: u64,
    pub delivered_per_300: f64,
    pub avg_completion_latency: f64,
    pub invalid_actions: u64,
    pub blocked_moves: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSummary {
    pub episodes: usize,
    pub mean_score: f64,
    pub p50_score: f64,
    pub p90_score: f64,
    pub mean_items_per_300: f64,
    pub mean_orders_completed: f64,
    pub mean_completion_latency: f64,
    pub mean_invalid_actions: f64,
    pub mean_blocked_moves: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub policy: String,
    pub requested_episodes: usize,
    pub summary: EvalSummary,
    pub episodes: Vec<EvalEpisode>,
}

pub fn run_batch_eval(req: EvalRequest<'_>) -> Result<EvalReport, Box<dyn std::error::Error>> {
    let mut metas = list_runs(req.logs_dir)?;
    if let Some(mode) = req.difficulty {
        metas.retain(|m| m.mode.as_deref() == Some(mode));
    }
    metas.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    if metas.is_empty() {
        return Ok(EvalReport {
            policy: format!("{:?}", req.policy).to_lowercase(),
            requested_episodes: req.episodes,
            summary: EvalSummary {
                episodes: 0,
                mean_score: 0.0,
                p50_score: 0.0,
                p90_score: 0.0,
                mean_items_per_300: 0.0,
                mean_orders_completed: 0.0,
                mean_completion_latency: 0.0,
                mean_invalid_actions: 0.0,
                mean_blocked_moves: 0.0,
            },
            episodes: Vec::new(),
        });
    }
    metas.truncate(req.episodes.max(1));

    let mut episodes = Vec::with_capacity(metas.len());
    for (ix, meta) in metas.into_iter().enumerate() {
        let path = req.logs_dir.join(&meta.file_name);
        let run = load_run(&path)?;
        let Some(first) = run.frames.first() else {
            continue;
        };

        let seed = req.seed ^ first.tick ^ ix as u64;
        let mut sim = sim_from_state(first.game_state.clone(), seed);
        let mut policy = Policy::new(Arc::new(default_config(req.policy)));
        step_many(
            &mut sim,
            |s| policy.decide_round(&s.game_state, Duration::from_millis(20)),
            300,
        );

        episodes.push(EvalEpisode {
            run_id: run.run_id,
            mode: run.mode,
            score: sim.game_state.score,
            items_delivered: sim.metrics.items_delivered,
            orders_completed: sim.metrics.orders_completed,
            ticks: sim.metrics.ticks,
            delivered_per_300: sim.metrics.delivered_per_300_rounds,
            avg_completion_latency: sim.metrics.avg_order_completion_latency,
            invalid_actions: sim.metrics.invalid_actions,
            blocked_moves: sim.metrics.blocked_moves,
        });
    }

    let summary = summarize(&episodes);
    Ok(EvalReport {
        policy: format!("{:?}", req.policy).to_lowercase(),
        requested_episodes: req.episodes,
        summary,
        episodes,
    })
}

fn summarize(episodes: &[EvalEpisode]) -> EvalSummary {
    if episodes.is_empty() {
        return EvalSummary {
            episodes: 0,
            mean_score: 0.0,
            p50_score: 0.0,
            p90_score: 0.0,
            mean_items_per_300: 0.0,
            mean_orders_completed: 0.0,
            mean_completion_latency: 0.0,
            mean_invalid_actions: 0.0,
            mean_blocked_moves: 0.0,
        };
    }

    let n = episodes.len() as f64;
    let mean_score = episodes.iter().map(|e| e.score as f64).sum::<f64>() / n;
    let mean_items_per_300 = episodes.iter().map(|e| e.delivered_per_300).sum::<f64>() / n;
    let mean_orders_completed = episodes
        .iter()
        .map(|e| e.orders_completed as f64)
        .sum::<f64>()
        / n;
    let mean_completion_latency = episodes
        .iter()
        .map(|e| e.avg_completion_latency)
        .sum::<f64>()
        / n;
    let mean_invalid_actions = episodes
        .iter()
        .map(|e| e.invalid_actions as f64)
        .sum::<f64>()
        / n;
    let mean_blocked_moves = episodes.iter().map(|e| e.blocked_moves as f64).sum::<f64>() / n;

    let mut scores = episodes.iter().map(|e| e.score as f64).collect::<Vec<_>>();
    scores.sort_by(|a, b| a.total_cmp(b));

    EvalSummary {
        episodes: episodes.len(),
        mean_score,
        p50_score: percentile(&scores, 0.50),
        p90_score: percentile(&scores, 0.90),
        mean_items_per_300,
        mean_orders_completed,
        mean_completion_latency,
        mean_invalid_actions,
        mean_blocked_moves,
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

pub fn default_config(mode: PolicyMode) -> Config {
    Config {
        policy_mode: mode,
        planner_budget_mode: PlannerBudgetMode::Fixed,
        planner_soft_budget_ms: 300,
        planner_soft_budget_min_ms: 200,
        planner_soft_budget_max_ms: 600,
        planner_hard_budget_ms: 1000,
        planner_deadline_slack_ms: 20,
        tick_soft_budget_ms: 50,
        tick_hard_budget_ms: 120,
        tick_greedy_fallback_ms: 8,
        cache_reuse_max_age_ticks: 0,
        cache_require_progress: false,
        log_level: "warn".to_owned(),
        structured_bot_log: false,
        ascii_render: false,
        debug: false,
        replay_dump_path: None,
    }
}
