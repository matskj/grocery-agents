use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use clap::{Parser, ValueEnum};
use serde_json::Value;

const PHASE_COUNT: usize = 3;

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

    #[arg(long)]
    mode_filter: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum EvalProfile {
    Safe,
    Aggressive,
    #[default]
    Default,
}

#[derive(Debug, Clone, Copy, Default)]
struct PhaseMetrics {
    score_gain: i64,
    items_delivered: u64,
    order_completions: u64,
    blocked_sum: u64,
    stuck_sum: u64,
    unique_goal_sum: f64,
    unique_goal_samples: u64,
    goal_concentration_sum: f64,
    tick_count: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct TickSnapshot {
    score_delta: i64,
    items_delivered_delta: u64,
    order_completed_delta: u64,
    blocked_bot_count: u64,
    stuck_bot_count: u64,
    unique_goal_cells_last_n: Option<f64>,
    goal_concentration_top3: f64,
    guard_fallback: bool,
}

#[derive(Debug, Clone, Default)]
struct EpisodeMetrics {
    run_file: String,
    mode: String,
    final_score: i64,
    waits: u64,
    moves: u64,
    pickups: u64,
    dropoffs: u64,
    blocked_events: u64,
    near_dropoff_congestion_events: u64,
    assignment_ms_sum: u64,
    assignment_ticks: u64,
    repeated_goal_concentration: f64,
    phase: [PhaseMetrics; PHASE_COUNT],
    tick_count: u64,
    late_no_delivery_streak: u64,
    goal_collapse_ticks: u64,
    guard_fallback_ticks: u64,
    guard_fallback_ratio: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.logs_dir)?;

    let metrics = if cli.from_logs {
        collect_latest_metrics(&cli.logs_dir, cli.episodes, cli.mode_filter.as_deref())?
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
            if !mode_matches(parsed.mode.as_str(), cli.mode_filter.as_deref()) {
                println!(
                    "episode {:>2}: skipped mode={} (filter={}) file={}",
                    episode + 1,
                    parsed.mode,
                    cli.mode_filter.as_deref().unwrap_or("any"),
                    parsed.run_file
                );
                continue;
            }
            println!(
                "episode {:>2}: mode={} score={} waits={} dropoffs={} file={}",
                episode + 1,
                parsed.mode,
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
            cmd.env("GROCERY_ASSIGNMENT_MODE", "legacy-only");
            cmd.env("GROCERY_DROPOFF_SCHEDULING_ENABLED", "true");
        }
        EvalProfile::Aggressive => {
            cmd.env("GROCERY_HORIZON", "20");
            cmd.env("GROCERY_CANDIDATE_K", "10");
            cmd.env("GROCERY_PLANNER_SOFT_BUDGET_MS", "1300");
            cmd.env("GROCERY_DROPOFF_WINDOW", "16");
            cmd.env("GROCERY_ASSIGNMENT_ENABLED", "true");
            cmd.env("GROCERY_ASSIGNMENT_MODE", "global-only");
            cmd.env("GROCERY_DROPOFF_SCHEDULING_ENABLED", "true");
        }
        EvalProfile::Default => {
            cmd.env("GROCERY_ASSIGNMENT_ENABLED", "true");
            cmd.env("GROCERY_ASSIGNMENT_MODE", "hybrid");
        }
    }
}

fn collect_latest_metrics(
    logs_dir: &Path,
    episodes: usize,
    mode_filter: Option<&str>,
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
    for path in entries {
        let metrics = parse_metrics(&path)?;
        if !mode_matches(metrics.mode.as_str(), mode_filter) {
            continue;
        }
        out.push(metrics);
        if out.len() >= episodes {
            break;
        }
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
    let mut goal_counts = HashMap::<(i64, i64), u64>::new();
    let mut total_goal_observations = 0u64;
    let mut tick_snapshots = Vec::<TickSnapshot>::new();
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
        } else if event == "game_mode" {
            if let Some(mode) = data.get("mode").and_then(Value::as_str) {
                metrics.mode = mode.to_owned();
            }
        } else if event == "tick" {
            if metrics.mode.is_empty() {
                if let Some(mode) = data.get("mode").and_then(Value::as_str) {
                    metrics.mode = mode.to_owned();
                } else if let Some(mode) = data.get("game_mode").and_then(Value::as_str) {
                    metrics.mode = mode.to_owned();
                }
            }
            let team_summary = data.get("team_summary").cloned().unwrap_or(Value::Null);
            let tick_outcome = data.get("tick_outcome").cloned().unwrap_or(Value::Null);
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
            let stuck = data
                .get("team_summary")
                .and_then(|s| s.get("stuck_bot_count"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            metrics.blocked_events = metrics.blocked_events.saturating_add(blocked);
            if let Some(assign_ms) = data
                .get("team_summary")
                .and_then(|s| s.get("assign_ms"))
                .and_then(Value::as_u64)
            {
                metrics.assignment_ms_sum = metrics.assignment_ms_sum.saturating_add(assign_ms);
                metrics.assignment_ticks = metrics.assignment_ticks.saturating_add(1);
            }
            if let Some(goal_map) = data
                .get("team_summary")
                .and_then(|s| s.get("goal_cell_by_bot"))
                .and_then(Value::as_object)
            {
                for value in goal_map.values() {
                    let Some(arr) = value.as_array() else {
                        continue;
                    };
                    if arr.len() != 2 {
                        continue;
                    }
                    let Some(x) = arr[0].as_i64() else {
                        continue;
                    };
                    let Some(y) = arr[1].as_i64() else {
                        continue;
                    };
                    *goal_counts.entry((x, y)).or_insert(0) += 1;
                    total_goal_observations = total_goal_observations.saturating_add(1);
                }
            }
            let congestion = data
                .get("team_summary")
                .and_then(|s| s.get("dropoff_congestion"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if congestion >= 2 {
                metrics.near_dropoff_congestion_events =
                    metrics.near_dropoff_congestion_events.saturating_add(1);
            }
            let unique_goal_cells_last_n = team_summary
                .get("unique_goal_cells_last_n")
                .and_then(Value::as_f64)
                .or_else(|| {
                    team_summary
                        .get("unique_goal_cells_last_n")
                        .and_then(Value::as_u64)
                        .map(|v| v as f64)
                });
            let goal_concentration_top3 = team_summary
                .get("assignment_goal_concentration_top3")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let assignment_source = team_summary
                .get("assignment_source")
                .and_then(Value::as_str)
                .unwrap_or("");
            let assignment_guard_reason = team_summary
                .get("assignment_guard_reason")
                .and_then(Value::as_str)
                .unwrap_or("none");
            let guard_fallback = is_guard_fallback_tick(assignment_source, assignment_guard_reason);
            let score_delta = tick_outcome
                .get("delta_score")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let items_delivered_delta = tick_outcome
                .get("items_delivered_delta")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let order_completed_delta = tick_outcome
                .get("order_completed_delta")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            tick_snapshots.push(TickSnapshot {
                score_delta,
                items_delivered_delta,
                order_completed_delta,
                blocked_bot_count: blocked,
                stuck_bot_count: stuck,
                unique_goal_cells_last_n,
                goal_concentration_top3,
                guard_fallback,
            });
        }
    }
    if metrics.mode.is_empty() {
        metrics.mode = "unknown".to_owned();
    }
    if total_goal_observations > 0 {
        let max_goal = goal_counts.values().copied().max().unwrap_or(0);
        metrics.repeated_goal_concentration = max_goal as f64 / total_goal_observations as f64;
    }
    metrics.tick_count = tick_snapshots.len() as u64;
    if !tick_snapshots.is_empty() {
        let mut phase_metrics = [PhaseMetrics::default(); PHASE_COUNT];
        let mut goal_collapse_ticks = 0u64;
        let mut guard_fallback_ticks = 0u64;
        for (idx, snapshot) in tick_snapshots.iter().enumerate() {
            let phase_idx = (idx * PHASE_COUNT / tick_snapshots.len()).min(PHASE_COUNT - 1);
            let phase = &mut phase_metrics[phase_idx];
            phase.score_gain += snapshot.score_delta;
            phase.items_delivered = phase
                .items_delivered
                .saturating_add(snapshot.items_delivered_delta);
            phase.order_completions = phase
                .order_completions
                .saturating_add(snapshot.order_completed_delta);
            phase.blocked_sum = phase.blocked_sum.saturating_add(snapshot.blocked_bot_count);
            phase.stuck_sum = phase.stuck_sum.saturating_add(snapshot.stuck_bot_count);
            if let Some(unique_goal) = snapshot.unique_goal_cells_last_n {
                phase.unique_goal_sum += unique_goal;
                phase.unique_goal_samples = phase.unique_goal_samples.saturating_add(1);
                if unique_goal <= 1.0 {
                    goal_collapse_ticks = goal_collapse_ticks.saturating_add(1);
                }
            }
            phase.goal_concentration_sum += snapshot.goal_concentration_top3;
            phase.tick_count = phase.tick_count.saturating_add(1);
            if snapshot.guard_fallback {
                guard_fallback_ticks = guard_fallback_ticks.saturating_add(1);
            }
        }

        let late_start = tick_snapshots.len() * 2 / PHASE_COUNT;
        let mut streak = 0u64;
        let mut max_streak = 0u64;
        for snapshot in tick_snapshots.iter().skip(late_start) {
            if snapshot.items_delivered_delta == 0 {
                streak = streak.saturating_add(1);
            } else {
                max_streak = max_streak.max(streak);
                streak = 0;
            }
        }
        max_streak = max_streak.max(streak);
        metrics.phase = phase_metrics;
        metrics.late_no_delivery_streak = max_streak;
        metrics.goal_collapse_ticks = goal_collapse_ticks;
        metrics.guard_fallback_ticks = guard_fallback_ticks;
        metrics.guard_fallback_ratio = guard_fallback_ticks as f64 / tick_snapshots.len() as f64;
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
        acc.assignment_ms_sum += item.assignment_ms_sum;
        acc.assignment_ticks += item.assignment_ticks;
        acc.repeated_goal_concentration += item.repeated_goal_concentration;
        acc.tick_count += item.tick_count;
        acc.late_no_delivery_streak += item.late_no_delivery_streak;
        acc.goal_collapse_ticks += item.goal_collapse_ticks;
        acc.guard_fallback_ticks += item.guard_fallback_ticks;
        acc.guard_fallback_ratio += item.guard_fallback_ratio;
        for phase_idx in 0..PHASE_COUNT {
            let src = item.phase[phase_idx];
            let dst = &mut acc.phase[phase_idx];
            dst.score_gain += src.score_gain;
            dst.items_delivered = dst.items_delivered.saturating_add(src.items_delivered);
            dst.order_completions = dst.order_completions.saturating_add(src.order_completions);
            dst.blocked_sum = dst.blocked_sum.saturating_add(src.blocked_sum);
            dst.stuck_sum = dst.stuck_sum.saturating_add(src.stuck_sum);
            dst.unique_goal_sum += src.unique_goal_sum;
            dst.unique_goal_samples = dst
                .unique_goal_samples
                .saturating_add(src.unique_goal_samples);
            dst.goal_concentration_sum += src.goal_concentration_sum;
            dst.tick_count = dst.tick_count.saturating_add(src.tick_count);
        }
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
    let avg_assign_ms = if totals.assignment_ticks == 0 {
        0.0
    } else {
        totals.assignment_ms_sum as f64 / totals.assignment_ticks as f64
    };
    let avg_goal_concentration = totals.repeated_goal_concentration / metrics.len() as f64;
    println!(
        "avg_assign_ms={:.2} repeated_goal_concentration={:.3}",
        avg_assign_ms, avg_goal_concentration
    );
    println!(
        "collapse_alarms late_no_delivery_streak_avg={:.2} goal_collapse_ratio={:.3} guard_fallback_ratio={:.3}",
        totals.late_no_delivery_streak as f64 / metrics.len() as f64,
        if totals.tick_count == 0 {
            0.0
        } else {
            totals.goal_collapse_ticks as f64 / totals.tick_count as f64
        },
        if totals.tick_count == 0 {
            0.0
        } else {
            totals.guard_fallback_ticks as f64 / totals.tick_count as f64
        },
    );
    for phase_idx in 0..PHASE_COUNT {
        let phase = totals.phase[phase_idx];
        let avg_blocked = if phase.tick_count == 0 {
            0.0
        } else {
            phase.blocked_sum as f64 / phase.tick_count as f64
        };
        let avg_stuck = if phase.tick_count == 0 {
            0.0
        } else {
            phase.stuck_sum as f64 / phase.tick_count as f64
        };
        let avg_unique_goal = if phase.unique_goal_samples == 0 {
            0.0
        } else {
            phase.unique_goal_sum / phase.unique_goal_samples as f64
        };
        let avg_goal_concentration_top3 = if phase.tick_count == 0 {
            0.0
        } else {
            phase.goal_concentration_sum / phase.tick_count as f64
        };
        println!(
            "phase={} score_gain={} delivered={} completed={} avg_blocked={:.2} avg_stuck={:.2} avg_unique_goals={:.2} avg_goal_concentration_top3={:.3}",
            phase_label(phase_idx),
            phase.score_gain,
            phase.items_delivered,
            phase.order_completions,
            avg_blocked,
            avg_stuck,
            avg_unique_goal,
            avg_goal_concentration_top3,
        );
    }
}

fn percentile(sorted: &[i64], p: f64) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((p / 100.0) * (sorted.len().saturating_sub(1) as f64)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn mode_matches(mode: &str, filter: Option<&str>) -> bool {
    filter
        .map(|want| mode.eq_ignore_ascii_case(want))
        .unwrap_or(true)
}

fn phase_label(phase_idx: usize) -> &'static str {
    match phase_idx {
        0 => "early",
        1 => "mid",
        _ => "late",
    }
}

fn is_guard_fallback_tick(source: &str, guard_reason: &str) -> bool {
    if guard_reason != "none" {
        return true;
    }
    matches!(
        source,
        "legacy_dispatcher_guard"
            | "legacy_dispatcher_timeout_fallback"
            | "legacy_forced_watchdog"
            | "legacy_forced_watchdog_trigger"
    )
}
