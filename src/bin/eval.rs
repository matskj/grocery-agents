use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use clap::{Parser, ValueEnum};
use serde_json::Value;

#[derive(Debug, Clone, Parser)]
#[command(name = "eval")]
struct Cli {
    #[arg(long, default_value_t = 10)]
    episodes: usize,

    #[arg(long, default_value = "logs")]
    logs_dir: PathBuf,

    #[arg(long, default_value = "x86_64-pc-windows-msvc")]
    target: String,

    #[arg(long)]
    token: Option<String>,

    #[arg(long)]
    ws_url: Option<String>,

    #[arg(long, default_value_t = EvalProfile::Default, value_enum)]
    profile: EvalProfile,

    #[arg(long, default_value_t = false)]
    from_logs: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum EvalProfile {
    Safe,
    Aggressive,
    #[default]
    Default,
}

#[derive(Debug, Clone, Default)]
struct EpisodeMetrics {
    run_file: String,
    final_score: i64,
    waits: u64,
    moves: u64,
    pickups: u64,
    dropoffs: u64,
    blocked_events: u64,
    near_dropoff_congestion_events: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.logs_dir)?;

    let metrics = if cli.from_logs {
        collect_latest_metrics(&cli.logs_dir, cli.episodes)?
    } else {
        run_episodes(&cli)?
    };
    if metrics.is_empty() {
        println!("No episodes were evaluated.");
        return Ok(());
    }
    print_summary(&metrics);
    Ok(())
}

fn run_episodes(cli: &Cli) -> Result<Vec<EpisodeMetrics>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for episode in 0..cli.episodes {
        let before = known_run_mtimes(&cli.logs_dir)?;
        let mut cmd = Command::new("cmd");
        cmd.arg("/c")
            .arg("cargo-x64.cmd")
            .arg("run")
            .arg("--target")
            .arg(&cli.target)
            .arg("--bin")
            .arg("grocery-agents")
            .arg("--");

        if let Some(token) = &cli.token {
            cmd.arg("--token").arg(token);
        }
        if let Some(ws_url) = &cli.ws_url {
            cmd.arg("--ws-url").arg(ws_url);
        }
        apply_profile_env(&mut cmd, cli.profile);

        let status = cmd.status()?;
        if !status.success() {
            eprintln!("episode {} failed with status {}", episode + 1, status);
            continue;
        }
        if let Some(path) = find_newest_run_since(&cli.logs_dir, &before)? {
            let parsed = parse_metrics(&path)?;
            println!(
                "episode {:>2}: score={} waits={} dropoffs={} file={}",
                episode + 1,
                parsed.final_score,
                parsed.waits,
                parsed.dropoffs,
                parsed.run_file
            );
            out.push(parsed);
        } else {
            eprintln!("episode {} completed but no new run log found", episode + 1);
        }
    }
    Ok(out)
}

fn apply_profile_env(cmd: &mut Command, profile: EvalProfile) {
    match profile {
        EvalProfile::Safe => {
            cmd.env("GROCERY_HORIZON", "12");
            cmd.env("GROCERY_CANDIDATE_K", "6");
            cmd.env("GROCERY_PLANNER_SOFT_BUDGET_MS", "900");
            cmd.env("GROCERY_DROPOFF_WINDOW", "10");
            cmd.env("GROCERY_ASSIGNMENT_ENABLED", "true");
            cmd.env("GROCERY_DROPOFF_SCHEDULING_ENABLED", "true");
        }
        EvalProfile::Aggressive => {
            cmd.env("GROCERY_HORIZON", "20");
            cmd.env("GROCERY_CANDIDATE_K", "10");
            cmd.env("GROCERY_PLANNER_SOFT_BUDGET_MS", "1300");
            cmd.env("GROCERY_DROPOFF_WINDOW", "16");
            cmd.env("GROCERY_ASSIGNMENT_ENABLED", "true");
            cmd.env("GROCERY_DROPOFF_SCHEDULING_ENABLED", "true");
        }
        EvalProfile::Default => {}
    }
}

fn collect_latest_metrics(
    logs_dir: &Path,
    episodes: usize,
) -> Result<Vec<EpisodeMetrics>, Box<dyn std::error::Error>> {
    let mut entries = fs::read_dir(logs_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("run-") && name.ends_with(".jsonl"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|path| {
        fs::metadata(path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    });
    entries.reverse();
    let mut out = Vec::new();
    for path in entries.into_iter().take(episodes) {
        out.push(parse_metrics(&path)?);
    }
    Ok(out)
}

fn known_run_mtimes(
    logs_dir: &Path,
) -> Result<HashMap<PathBuf, SystemTime>, Box<dyn std::error::Error>> {
    let mut out = HashMap::new();
    for entry in fs::read_dir(logs_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !is_run_log(&path) {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        out.insert(path, modified);
    }
    Ok(out)
}

fn find_newest_run_since(
    logs_dir: &Path,
    before: &HashMap<PathBuf, SystemTime>,
) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let mut newest = None::<(SystemTime, PathBuf)>;
    for entry in fs::read_dir(logs_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !is_run_log(&path) {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let is_new = before.get(&path).map(|old| modified > *old).unwrap_or(true);
        if !is_new {
            continue;
        }
        if newest
            .as_ref()
            .map(|(best_time, _)| modified > *best_time)
            .unwrap_or(true)
        {
            newest = Some((modified, path));
        }
    }
    Ok(newest.map(|(_, path)| path))
}

fn is_run_log(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("run-") && name.ends_with(".jsonl"))
        .unwrap_or(false)
}

fn parse_metrics(path: &Path) -> Result<EpisodeMetrics, Box<dyn std::error::Error>> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_owned();
    let content = fs::read_to_string(path)?;
    let mut metrics = EpisodeMetrics {
        run_file: file_name,
        ..EpisodeMetrics::default()
    };
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let event = value.get("event").and_then(Value::as_str).unwrap_or("");
        let data = value.get("data").cloned().unwrap_or(Value::Null);
        if event == "game_over" {
            metrics.final_score = data
                .get("final_score")
                .and_then(Value::as_i64)
                .unwrap_or_default();
            if let Some(counters) = data.get("episode_counters") {
                metrics.moves = counters.get("moves").and_then(Value::as_u64).unwrap_or(0);
                metrics.waits = counters.get("waits").and_then(Value::as_u64).unwrap_or(0);
                metrics.pickups = counters.get("pickups").and_then(Value::as_u64).unwrap_or(0);
                metrics.dropoffs = counters.get("dropoffs").and_then(Value::as_u64).unwrap_or(0);
                metrics.blocked_events = counters
                    .get("blocked_events")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                metrics.near_dropoff_congestion_events = counters
                    .get("near_dropoff_congestion_events")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
            }
        } else if event == "tick" {
            if let Some(actions) = data.get("actions").and_then(Value::as_array) {
                for action in actions {
                    match action.get("kind").and_then(Value::as_str).unwrap_or("") {
                        "move" => metrics.moves = metrics.moves.saturating_add(1),
                        "wait" => metrics.waits = metrics.waits.saturating_add(1),
                        "pick_up" => metrics.pickups = metrics.pickups.saturating_add(1),
                        "drop_off" => metrics.dropoffs = metrics.dropoffs.saturating_add(1),
                        _ => {}
                    }
                }
            }
            let blocked = data
                .get("team_summary")
                .and_then(|s| s.get("blocked_bot_count"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            metrics.blocked_events = metrics.blocked_events.saturating_add(blocked);
            let congestion = data
                .get("team_summary")
                .and_then(|s| s.get("dropoff_congestion"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if congestion >= 2 {
                metrics.near_dropoff_congestion_events =
                    metrics.near_dropoff_congestion_events.saturating_add(1);
            }
        }
    }
    Ok(metrics)
}

fn print_summary(metrics: &[EpisodeMetrics]) {
    let mut scores = metrics.iter().map(|m| m.final_score).collect::<Vec<_>>();
    scores.sort_unstable();
    let mean_score = metrics.iter().map(|m| m.final_score as f64).sum::<f64>() / metrics.len() as f64;
    let p50 = percentile(&scores, 50.0);
    let p90 = percentile(&scores, 90.0);

    let totals = metrics.iter().fold(EpisodeMetrics::default(), |mut acc, item| {
        acc.waits += item.waits;
        acc.moves += item.moves;
        acc.pickups += item.pickups;
        acc.dropoffs += item.dropoffs;
        acc.blocked_events += item.blocked_events;
        acc.near_dropoff_congestion_events += item.near_dropoff_congestion_events;
        acc
    });
    let total_actions = totals.waits + totals.moves + totals.pickups + totals.dropoffs;
    let wait_ratio = if total_actions == 0 {
        0.0
    } else {
        totals.waits as f64 / total_actions as f64
    };

    println!("episodes: {}", metrics.len());
    println!("score mean={:.2} p50={} p90={}", mean_score, p50, p90);
    println!(
        "wait_ratio={:.3} moves={} waits={} pickups={} dropoffs={}",
        wait_ratio, totals.moves, totals.waits, totals.pickups, totals.dropoffs
    );
    println!(
        "blocked_events={} near_dropoff_congestion_events={}",
        totals.blocked_events, totals.near_dropoff_congestion_events
    );
}

fn percentile(sorted: &[i64], p: f64) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((p / 100.0) * (sorted.len().saturating_sub(1) as f64)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}
