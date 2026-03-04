use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use serde_json::Value;

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum PolicyArg {
    #[default]
    Auto,
    Easy,
    Medium,
    Hard,
    Expert,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum BudgetProfile {
    Fast,
    #[default]
    Balanced,
    Safe,
}

#[derive(Debug, Parser)]
#[command(name = "replay_bench")]
struct Cli {
    #[arg(long)]
    replay: PathBuf,

    #[arg(long, value_enum, default_value_t = PolicyArg::Auto)]
    policy: PolicyArg,

    #[arg(long, value_enum, default_value_t = BudgetProfile::Balanced)]
    budget_profile: BudgetProfile,

    #[arg(long, default_value_t = false)]
    determinism_check: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let reader = BufReader::new(File::open(&cli.replay)?);
    let mut plan_ms = Vec::<u64>::new();
    let mut waits = 0u64;
    let mut actions = 0u64;
    let mut invalid_subs = 0u64;
    let mut fallback_cached = 0u64;
    let mut fallback_greedy = 0u64;
    let mut fallback_wait = 0u64;
    let mut action_sig = DefaultHasher::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(record) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let event = record.get("event").and_then(Value::as_str).unwrap_or("");
        if event != "tick" {
            continue;
        }
        let data = record.get("data").unwrap_or(&Value::Null);
        let tick = data.get("tick").and_then(Value::as_u64).unwrap_or(0);
        let team = data.get("team_summary").unwrap_or(&Value::Null);
        if let Some(v) = team.get("plan_ms").and_then(Value::as_u64) {
            plan_ms.push(v);
        } else if let Some(v) = team.get("assign_ms").and_then(Value::as_u64) {
            plan_ms.push(v);
        }
        invalid_subs = invalid_subs.saturating_add(
            data.get("tick_outcome")
                .and_then(|v| v.get("invalid_action_count"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );

        match team
            .get("fallback_level")
            .and_then(Value::as_str)
            .unwrap_or("none")
        {
            "cached" => fallback_cached = fallback_cached.saturating_add(1),
            "greedy" | "greedy_or_guard" => {
                fallback_greedy = fallback_greedy.saturating_add(1);
            }
            "wait" | "timeout_fallback" => fallback_wait = fallback_wait.saturating_add(1),
            _ => {}
        }

        if let Some(arr) = data.get("actions").and_then(Value::as_array) {
            for action in arr {
                actions = actions.saturating_add(1);
                let kind = action
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or("wait")
                    .to_owned();
                if kind == "wait" {
                    waits = waits.saturating_add(1);
                }
                let bot = action
                    .get("bot_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                (tick, bot, kind).hash(&mut action_sig);
            }
        }
    }

    if plan_ms.is_empty() {
        plan_ms.push(0);
    }
    plan_ms.sort_unstable();
    let avg = plan_ms.iter().sum::<u64>() as f64 / plan_ms.len() as f64;
    let p50 = percentile(&plan_ms, 0.50);
    let p95 = percentile(&plan_ms, 0.95);
    let p99 = percentile(&plan_ms, 0.99);
    let wait_ratio = if actions == 0 {
        0.0
    } else {
        waits as f64 / actions as f64
    };
    let projected_rounds = if avg > 0.0 {
        ((120_000.0 / avg).floor() as u64).min(300)
    } else {
        0
    };

    println!(
        "replay_bench policy={:?} budget_profile={:?} ticks={} avg_ms={:.2} p50={} p95={} p99={}",
        cli.policy,
        cli.budget_profile,
        plan_ms.len(),
        avg,
        p50,
        p95,
        p99
    );
    println!(
        "fallback_counts cached={} greedy={} wait={} invalid_substitutions={}",
        fallback_cached, fallback_greedy, fallback_wait, invalid_subs
    );
    println!(
        "throughput wait_ratio={:.3} projected_rounds_in_120s={}",
        wait_ratio, projected_rounds
    );
    if cli.determinism_check {
        let sig = action_sig.finish();
        println!("determinism signature={sig}");
    }
    Ok(())
}

fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}
