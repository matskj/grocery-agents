use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
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

    #[arg(long, default_value_t = false)]
    enforce_gates: bool,

    #[arg(long, default_value_t = GateProfile::Default, value_enum)]
    gate_profile: GateProfile,

    #[arg(long)]
    coord_baseline: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    write_coord_baseline: bool,

    #[arg(long, default_value_t = false)]
    strict_all_modes: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum EvalProfile {
    Safe,
    Aggressive,
    #[default]
    Default,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum GateProfile {
    #[default]
    Default,
    Strict,
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
    items_delivered: u64,
    pickup_attempts: u64,
    pickup_successes: u64,
    dropoff_attempts: u64,
    dropoff_successes: u64,
    far_no_conversion_ticks: u64,
    collector_waits: u64,
    collector_far_waits: u64,
    full_inactive_waits: u64,
    full_inactive_far_waits: u64,
    local_first_checks: u64,
    local_first_violations: u64,
    expansion_mode_true: u64,
    expansion_mode_samples: u64,
    preferred_area_matches: u64,
    preferred_area_samples: u64,
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
    let failed = evaluate_gates(&metrics, cli.gate_profile);
    let baseline_path = cli
        .coord_baseline
        .clone()
        .unwrap_or_else(|| PathBuf::from("models/coord_baseline.json"));
    if cli.write_coord_baseline {
        let baseline = aggregate_mode_metrics(&metrics);
        let payload = serde_json::to_string_pretty(&baseline)?;
        if let Some(parent) = baseline_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&baseline_path, payload)?;
        println!(
            "coord-baseline status=WRITE path={} modes={}",
            baseline_path.display(),
            baseline.len()
        );
    }
    let strict_failed = if cli.strict_all_modes {
        let baseline_payload = fs::read_to_string(&baseline_path)?;
        let baseline: HashMap<String, CoordBaselineMetrics> =
            serde_json::from_str(&baseline_payload)?;
        evaluate_strict_all_modes(&metrics, &baseline)
    } else {
        false
    };
    if (failed || strict_failed) && cli.enforce_gates {
        return Err("regression gate failure".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CoordBaselineMetrics {
    delivered_per_100_ticks: f64,
    pickup_success_ratio: f64,
    repeated_goal_concentration: f64,
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
    let mut prev_carrying_by_bot = HashMap::<String, i64>::new();
    let mut prev_action_by_bot = HashMap::<String, String>::new();
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
                metrics.dropoffs = counters
                    .get("dropoffs")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
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
            let queue_roles = team_summary.get("queue_roles").and_then(Value::as_object);
            let active_order_items_set = data
                .get("game_state")
                .and_then(|s| s.get("orders"))
                .and_then(Value::as_array)
                .map(|orders| {
                    orders
                        .iter()
                        .filter_map(|order| {
                            let in_progress = order
                                .get("status")
                                .and_then(Value::as_str)
                                .map(|status| status.eq_ignore_ascii_case("in_progress"))
                                .unwrap_or(false);
                            if !in_progress {
                                return None;
                            }
                            order
                                .get("item_id")
                                .and_then(Value::as_str)
                                .map(|item| item.to_owned())
                        })
                        .collect::<HashSet<_>>()
                })
                .unwrap_or_default();
            let dropoff_tiles = data
                .get("game_state")
                .and_then(|s| s.get("grid"))
                .and_then(|g| g.get("drop_off_tiles"))
                .and_then(Value::as_array)
                .map(|tiles| {
                    tiles
                        .iter()
                        .filter_map(|tile| {
                            let arr = tile.as_array()?;
                            if arr.len() != 2 {
                                return None;
                            }
                            let x = arr[0].as_i64()?;
                            let y = arr[1].as_i64()?;
                            Some((x, y))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let mut action_by_bot = HashMap::<String, String>::new();
            if let Some(actions) = data.get("actions").and_then(Value::as_array) {
                for action in actions {
                    let bot_id = action
                        .get("bot_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned();
                    let kind = action
                        .get("kind")
                        .and_then(Value::as_str)
                        .unwrap_or("wait")
                        .to_owned();
                    match kind.as_str() {
                        "move" => metrics.moves = metrics.moves.saturating_add(1),
                        "wait" => metrics.waits = metrics.waits.saturating_add(1),
                        "pick_up" => metrics.pickups = metrics.pickups.saturating_add(1),
                        "drop_off" => metrics.dropoffs = metrics.dropoffs.saturating_add(1),
                        _ => {}
                    }
                    if !bot_id.is_empty() {
                        action_by_bot.insert(bot_id.clone(), kind.clone());
                    }
                }
            }
            let mut carrying_by_bot = HashMap::<String, i64>::new();
            let mut carrying_items_by_bot = HashMap::<String, Vec<String>>::new();
            let mut capacity_by_bot = HashMap::<String, usize>::new();
            let mut bot_dropoff_dist_by_bot = HashMap::<String, f64>::new();
            let mut bot_pos_by_bot = HashMap::<String, (i64, i64)>::new();
            if let Some(bots) = data
                .get("game_state")
                .and_then(|s| s.get("bots"))
                .and_then(Value::as_array)
            {
                for bot in bots {
                    let bot_id = bot
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned();
                    if bot_id.is_empty() {
                        continue;
                    }
                    let carrying_items = bot
                        .get("carrying")
                        .and_then(Value::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(Value::as_str)
                                .map(|item| item.to_owned())
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let carrying = carrying_items.len() as i64;
                    let capacity = bot
                        .get("capacity")
                        .and_then(Value::as_u64)
                        .map(|v| v as usize)
                        .unwrap_or(3);
                    let x = bot.get("x").and_then(Value::as_i64).unwrap_or(0);
                    let y = bot.get("y").and_then(Value::as_i64).unwrap_or(0);
                    let bot_dropoff_dist = nearest_dropoff_distance(x, y, &dropoff_tiles);
                    bot_dropoff_dist_by_bot.insert(bot_id.clone(), bot_dropoff_dist);
                    bot_pos_by_bot.insert(bot_id.clone(), (x, y));
                    carrying_items_by_bot.insert(bot_id.clone(), carrying_items);
                    capacity_by_bot.insert(bot_id.clone(), capacity);
                    carrying_by_bot.insert(bot_id, carrying);
                }
            }
            let preferred_area_by_bot = team_summary
                .get("preferred_area_id_by_bot")
                .and_then(Value::as_object);
            let expansion_mode_by_bot = team_summary
                .get("expansion_mode_by_bot")
                .and_then(Value::as_object);
            let local_radius_by_bot = team_summary
                .get("local_radius_by_bot")
                .and_then(Value::as_object);
            let goal_area_by_bot = team_summary
                .get("goal_area_id_by_bot")
                .and_then(Value::as_object);
            let goal_cell_by_bot = team_summary
                .get("goal_cell_by_bot")
                .and_then(Value::as_object);
            for (bot_id, (bx, by)) in &bot_pos_by_bot {
                let expansion = expansion_mode_by_bot
                    .and_then(|m| m.get(bot_id))
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                metrics.expansion_mode_samples = metrics.expansion_mode_samples.saturating_add(1);
                if expansion {
                    metrics.expansion_mode_true = metrics.expansion_mode_true.saturating_add(1);
                }
                let pref_area = preferred_area_by_bot
                    .and_then(|m| m.get(bot_id))
                    .and_then(Value::as_i64)
                    .unwrap_or(-1);
                let goal_area = goal_area_by_bot
                    .and_then(|m| m.get(bot_id))
                    .and_then(Value::as_i64)
                    .unwrap_or(-1);
                let local_radius = local_radius_by_bot
                    .and_then(|m| m.get(bot_id))
                    .and_then(Value::as_f64)
                    .or_else(|| {
                        local_radius_by_bot
                            .and_then(|m| m.get(bot_id))
                            .and_then(Value::as_i64)
                            .map(|v| v as f64)
                    })
                    .unwrap_or(0.0);
                let goal_xy = goal_cell_by_bot
                    .and_then(|m| m.get(bot_id))
                    .and_then(Value::as_array)
                    .and_then(|arr| {
                        if arr.len() != 2 {
                            return None;
                        }
                        Some((arr[0].as_i64()?, arr[1].as_i64()?))
                    });

                if pref_area >= 0 && goal_area >= 0 {
                    metrics.preferred_area_samples =
                        metrics.preferred_area_samples.saturating_add(1);
                    if pref_area == goal_area {
                        metrics.preferred_area_matches =
                            metrics.preferred_area_matches.saturating_add(1);
                    }
                }
                if !expansion {
                    if let Some((gx, gy)) = goal_xy {
                        metrics.local_first_checks = metrics.local_first_checks.saturating_add(1);
                        let out_of_area =
                            pref_area >= 0 && goal_area >= 0 && pref_area != goal_area;
                        let dist_to_goal = (bx - gx).abs() as f64 + (by - gy).abs() as f64;
                        let out_of_radius = local_radius > 0.0 && dist_to_goal > local_radius;
                        if out_of_area || out_of_radius {
                            metrics.local_first_violations =
                                metrics.local_first_violations.saturating_add(1);
                        }
                    }
                }
            }
            if let Some(actions) = data.get("actions").and_then(Value::as_array) {
                for action in actions {
                    let bot_id = action
                        .get("bot_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned();
                    let kind = action
                        .get("kind")
                        .and_then(Value::as_str)
                        .unwrap_or("wait")
                        .to_owned();
                    if kind != "wait" || bot_id.is_empty() {
                        continue;
                    }
                    let role = queue_roles
                        .and_then(|roles| roles.get(bot_id.as_str()))
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    let bot_dropoff_dist = bot_dropoff_dist_by_bot
                        .get(&bot_id)
                        .copied()
                        .unwrap_or(f64::INFINITY);
                    if role == "collector" {
                        metrics.collector_waits = metrics.collector_waits.saturating_add(1);
                        if bot_dropoff_dist >= far_distance_threshold() {
                            metrics.collector_far_waits =
                                metrics.collector_far_waits.saturating_add(1);
                        }
                    }
                    let carrying_items = carrying_items_by_bot
                        .get(&bot_id)
                        .cloned()
                        .unwrap_or_default();
                    let carrying_count = carrying_items.len();
                    let capacity = capacity_by_bot.get(&bot_id).copied().unwrap_or(3);
                    let full = capacity > 0 && carrying_count >= capacity;
                    let carrying_active = carrying_items
                        .iter()
                        .any(|item| active_order_items_set.contains(item));
                    let full_inactive = full && !carrying_active;
                    if full_inactive {
                        metrics.full_inactive_waits = metrics.full_inactive_waits.saturating_add(1);
                        if bot_dropoff_dist >= far_distance_threshold() {
                            metrics.full_inactive_far_waits =
                                metrics.full_inactive_far_waits.saturating_add(1);
                        }
                    }
                }
            }
            let mut tick_pickup_successes = 0u64;
            let mut tick_dropoff_successes = 0u64;
            for (bot_id, prev_action) in &prev_action_by_bot {
                let Some(&prev_carrying) = prev_carrying_by_bot.get(bot_id) else {
                    continue;
                };
                let Some(&current_carrying) = carrying_by_bot.get(bot_id) else {
                    continue;
                };
                if prev_action == "pick_up" {
                    metrics.pickup_attempts = metrics.pickup_attempts.saturating_add(1);
                    if current_carrying > prev_carrying {
                        metrics.pickup_successes = metrics.pickup_successes.saturating_add(1);
                        tick_pickup_successes = tick_pickup_successes.saturating_add(1);
                    }
                } else if prev_action == "drop_off" {
                    metrics.dropoff_attempts = metrics.dropoff_attempts.saturating_add(1);
                    if current_carrying < prev_carrying {
                        metrics.dropoff_successes = metrics.dropoff_successes.saturating_add(1);
                        tick_dropoff_successes = tick_dropoff_successes.saturating_add(1);
                    }
                }
            }
            let mean_dropoff_dist = if bot_dropoff_dist_by_bot.is_empty() {
                0.0
            } else {
                bot_dropoff_dist_by_bot.values().copied().sum::<f64>()
                    / bot_dropoff_dist_by_bot.len() as f64
            };
            if tick_pickup_successes + tick_dropoff_successes == 0
                && mean_dropoff_dist >= far_distance_threshold()
            {
                metrics.far_no_conversion_ticks = metrics.far_no_conversion_ticks.saturating_add(1);
            }
            prev_action_by_bot = action_by_bot;
            prev_carrying_by_bot = carrying_by_bot;
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
            metrics.items_delivered = metrics
                .items_delivered
                .saturating_add(items_delivered_delta);
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
    let mean_score =
        metrics.iter().map(|m| m.final_score as f64).sum::<f64>() / metrics.len() as f64;
    let p50 = percentile(&scores, 50.0);
    let p90 = percentile(&scores, 90.0);

    let totals = metrics
        .iter()
        .fold(EpisodeMetrics::default(), |mut acc, item| {
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
            acc.items_delivered += item.items_delivered;
            acc.pickup_attempts += item.pickup_attempts;
            acc.pickup_successes += item.pickup_successes;
            acc.dropoff_attempts += item.dropoff_attempts;
            acc.dropoff_successes += item.dropoff_successes;
            acc.far_no_conversion_ticks += item.far_no_conversion_ticks;
            acc.collector_waits += item.collector_waits;
            acc.collector_far_waits += item.collector_far_waits;
            acc.full_inactive_waits += item.full_inactive_waits;
            acc.full_inactive_far_waits += item.full_inactive_far_waits;
            acc.local_first_checks += item.local_first_checks;
            acc.local_first_violations += item.local_first_violations;
            acc.expansion_mode_true += item.expansion_mode_true;
            acc.expansion_mode_samples += item.expansion_mode_samples;
            acc.preferred_area_matches += item.preferred_area_matches;
            acc.preferred_area_samples += item.preferred_area_samples;
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
    let delivered_per_100_ticks = if totals.tick_count == 0 {
        0.0
    } else {
        totals.items_delivered as f64 * 100.0 / totals.tick_count as f64
    };
    let pickup_success_ratio = if totals.pickup_attempts == 0 {
        0.0
    } else {
        totals.pickup_successes as f64 / totals.pickup_attempts as f64
    };
    let dropoff_success_ratio = if totals.dropoff_attempts == 0 {
        0.0
    } else {
        totals.dropoff_successes as f64 / totals.dropoff_attempts as f64
    };
    let far_no_conversion_tick_ratio = if totals.tick_count == 0 {
        0.0
    } else {
        totals.far_no_conversion_ticks as f64 / totals.tick_count as f64
    };
    let collector_far_wait_ratio = if totals.collector_waits == 0 {
        0.0
    } else {
        totals.collector_far_waits as f64 / totals.collector_waits as f64
    };
    let full_inactive_far_wait_ratio = if totals.full_inactive_waits == 0 {
        0.0
    } else {
        totals.full_inactive_far_waits as f64 / totals.full_inactive_waits as f64
    };
    let local_first_violation_ratio = if totals.local_first_checks == 0 {
        0.0
    } else {
        totals.local_first_violations as f64 / totals.local_first_checks as f64
    };
    let expansion_mode_tick_ratio = if totals.expansion_mode_samples == 0 {
        0.0
    } else {
        totals.expansion_mode_true as f64 / totals.expansion_mode_samples as f64
    };
    let preferred_area_match_ratio = if totals.preferred_area_samples == 0 {
        0.0
    } else {
        totals.preferred_area_matches as f64 / totals.preferred_area_samples as f64
    };
    println!(
        "conversion delivered_per_100_ticks={:.3} pickup_success_ratio={:.3} ({}/{}) dropoff_success_ratio={:.3} ({}/{})",
        delivered_per_100_ticks,
        pickup_success_ratio,
        totals.pickup_successes,
        totals.pickup_attempts,
        dropoff_success_ratio,
        totals.dropoff_successes,
        totals.dropoff_attempts
    );
    println!(
        "spatial far_no_conversion_tick_ratio={:.3} collector_far_wait_ratio={:.3} ({}/{}) full_inactive_far_wait_ratio={:.3} ({}/{})",
        far_no_conversion_tick_ratio,
        collector_far_wait_ratio,
        totals.collector_far_waits,
        totals.collector_waits,
        full_inactive_far_wait_ratio,
        totals.full_inactive_far_waits,
        totals.full_inactive_waits
    );
    println!(
        "locality local_first_violation_ratio={:.3} expansion_mode_tick_ratio={:.3} preferred_area_match_ratio={:.3}",
        local_first_violation_ratio,
        expansion_mode_tick_ratio,
        preferred_area_match_ratio,
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

#[derive(Debug, Clone, Copy)]
struct GateThreshold {
    min_delivered_per_100_ticks: f64,
    min_pickup_success_ratio: f64,
    min_dropoff_success_ratio: f64,
    max_guard_fallback_ratio: f64,
    max_repeated_goal_concentration: f64,
    max_late_no_delivery_streak: f64,
}

fn threshold_for_mode(mode: &str) -> Option<GateThreshold> {
    match mode {
        "easy" => Some(GateThreshold {
            min_delivered_per_100_ticks: 1.8,
            min_pickup_success_ratio: 0.45,
            min_dropoff_success_ratio: 0.45,
            max_guard_fallback_ratio: 0.25,
            max_repeated_goal_concentration: 0.25,
            max_late_no_delivery_streak: 220.0,
        }),
        "medium" => Some(GateThreshold {
            min_delivered_per_100_ticks: 3.0,
            min_pickup_success_ratio: 0.55,
            min_dropoff_success_ratio: 0.22,
            max_guard_fallback_ratio: 0.15,
            max_repeated_goal_concentration: 0.25,
            max_late_no_delivery_streak: 180.0,
        }),
        "hard" => Some(GateThreshold {
            min_delivered_per_100_ticks: 3.5,
            min_pickup_success_ratio: 0.20,
            min_dropoff_success_ratio: 0.20,
            max_guard_fallback_ratio: 0.10,
            max_repeated_goal_concentration: 0.30,
            max_late_no_delivery_streak: 160.0,
        }),
        "expert" => Some(GateThreshold {
            min_delivered_per_100_ticks: 1.6,
            min_pickup_success_ratio: 0.25,
            min_dropoff_success_ratio: 0.07,
            max_guard_fallback_ratio: 0.12,
            max_repeated_goal_concentration: 0.22,
            max_late_no_delivery_streak: 230.0,
        }),
        _ => None,
    }
}

fn apply_gate_profile(threshold: GateThreshold, profile: GateProfile) -> GateThreshold {
    match profile {
        GateProfile::Default => threshold,
        GateProfile::Strict => GateThreshold {
            min_delivered_per_100_ticks: threshold.min_delivered_per_100_ticks * 1.1,
            min_pickup_success_ratio: threshold.min_pickup_success_ratio * 1.1,
            min_dropoff_success_ratio: threshold.min_dropoff_success_ratio * 1.1,
            max_guard_fallback_ratio: threshold.max_guard_fallback_ratio * 0.9,
            max_repeated_goal_concentration: threshold.max_repeated_goal_concentration * 0.9,
            max_late_no_delivery_streak: threshold.max_late_no_delivery_streak * 0.9,
        },
    }
}

fn evaluate_gates(metrics: &[EpisodeMetrics], profile: GateProfile) -> bool {
    let mut by_mode = HashMap::<String, Vec<&EpisodeMetrics>>::new();
    for item in metrics {
        by_mode.entry(item.mode.clone()).or_default().push(item);
    }
    let mut any_failed = false;
    for (mode, rows) in by_mode {
        let Some(base_threshold) = threshold_for_mode(mode.as_str()) else {
            continue;
        };
        let t = apply_gate_profile(base_threshold, profile);
        let ticks = rows.iter().map(|r| r.tick_count).sum::<u64>();
        let delivered = rows.iter().map(|r| r.items_delivered).sum::<u64>();
        let pickup_attempts = rows.iter().map(|r| r.pickup_attempts).sum::<u64>();
        let pickup_successes = rows.iter().map(|r| r.pickup_successes).sum::<u64>();
        let dropoff_attempts = rows.iter().map(|r| r.dropoff_attempts).sum::<u64>();
        let dropoff_successes = rows.iter().map(|r| r.dropoff_successes).sum::<u64>();
        let guard_ticks = rows.iter().map(|r| r.guard_fallback_ticks).sum::<u64>();
        let repeated_goal_concentration = rows
            .iter()
            .map(|r| r.repeated_goal_concentration)
            .sum::<f64>()
            / rows.len() as f64;
        let late_no_delivery_streak = rows
            .iter()
            .map(|r| r.late_no_delivery_streak as f64)
            .sum::<f64>()
            / rows.len() as f64;

        let delivered_per_100_ticks = if ticks == 0 {
            0.0
        } else {
            delivered as f64 * 100.0 / ticks as f64
        };
        let pickup_success_ratio = if pickup_attempts == 0 {
            0.0
        } else {
            pickup_successes as f64 / pickup_attempts as f64
        };
        let dropoff_success_ratio = if dropoff_attempts == 0 {
            0.0
        } else {
            dropoff_successes as f64 / dropoff_attempts as f64
        };
        let guard_fallback_ratio = if ticks == 0 {
            0.0
        } else {
            guard_ticks as f64 / ticks as f64
        };
        let mut failed_checks = Vec::<&str>::new();
        if delivered_per_100_ticks < t.min_delivered_per_100_ticks {
            failed_checks.push("delivered_per_100_ticks");
        }
        if pickup_success_ratio < t.min_pickup_success_ratio {
            failed_checks.push("pickup_success_ratio");
        }
        if dropoff_success_ratio < t.min_dropoff_success_ratio {
            failed_checks.push("dropoff_success_ratio");
        }
        if guard_fallback_ratio > t.max_guard_fallback_ratio {
            failed_checks.push("guard_fallback_ratio");
        }
        if repeated_goal_concentration > t.max_repeated_goal_concentration {
            failed_checks.push("repeated_goal_concentration");
        }
        if late_no_delivery_streak > t.max_late_no_delivery_streak {
            failed_checks.push("late_no_delivery_streak");
        }
        if failed_checks.is_empty() {
            println!(
                "gate mode={} status=PASS delivered_per_100_ticks={:.3} pickup_success_ratio={:.3} dropoff_success_ratio={:.3} guard_fallback_ratio={:.3} repeated_goal_concentration={:.3} late_no_delivery_streak={:.2}",
                mode,
                delivered_per_100_ticks,
                pickup_success_ratio,
                dropoff_success_ratio,
                guard_fallback_ratio,
                repeated_goal_concentration,
                late_no_delivery_streak,
            );
        } else {
            any_failed = true;
            println!(
                "gate mode={} status=FAIL failed={} delivered_per_100_ticks={:.3} pickup_success_ratio={:.3} dropoff_success_ratio={:.3} guard_fallback_ratio={:.3} repeated_goal_concentration={:.3} late_no_delivery_streak={:.2}",
                mode,
                failed_checks.join(","),
                delivered_per_100_ticks,
                pickup_success_ratio,
                dropoff_success_ratio,
                guard_fallback_ratio,
                repeated_goal_concentration,
                late_no_delivery_streak,
            );
        }
    }
    any_failed
}

fn aggregate_mode_metrics(metrics: &[EpisodeMetrics]) -> HashMap<String, CoordBaselineMetrics> {
    let mut by_mode = HashMap::<String, Vec<&EpisodeMetrics>>::new();
    for item in metrics {
        by_mode.entry(item.mode.clone()).or_default().push(item);
    }
    let mut out = HashMap::<String, CoordBaselineMetrics>::new();
    for (mode, rows) in by_mode {
        let ticks = rows.iter().map(|r| r.tick_count).sum::<u64>();
        let delivered = rows.iter().map(|r| r.items_delivered).sum::<u64>();
        let pickup_attempts = rows.iter().map(|r| r.pickup_attempts).sum::<u64>();
        let pickup_successes = rows.iter().map(|r| r.pickup_successes).sum::<u64>();
        let repeated_goal_concentration = rows
            .iter()
            .map(|r| r.repeated_goal_concentration)
            .sum::<f64>()
            / rows.len() as f64;
        let delivered_per_100_ticks = if ticks == 0 {
            0.0
        } else {
            delivered as f64 * 100.0 / ticks as f64
        };
        let pickup_success_ratio = if pickup_attempts == 0 {
            0.0
        } else {
            pickup_successes as f64 / pickup_attempts as f64
        };
        out.insert(
            mode,
            CoordBaselineMetrics {
                delivered_per_100_ticks,
                pickup_success_ratio,
                repeated_goal_concentration,
            },
        );
    }
    out
}

fn evaluate_strict_all_modes(
    metrics: &[EpisodeMetrics],
    baseline: &HashMap<String, CoordBaselineMetrics>,
) -> bool {
    let current = aggregate_mode_metrics(metrics);
    let required_modes = ["medium", "hard", "expert"];
    let mut any_failed = false;
    for mode in required_modes {
        let Some(cur) = current.get(mode) else {
            println!(
                "coord-strict mode={} status=FAIL failed=missing_current_mode",
                mode
            );
            any_failed = true;
            continue;
        };
        let Some(base) = baseline.get(mode) else {
            println!(
                "coord-strict mode={} status=FAIL failed=missing_baseline_mode",
                mode
            );
            any_failed = true;
            continue;
        };
        let repeated_ok =
            cur.repeated_goal_concentration <= base.repeated_goal_concentration - 0.01;
        let pickup_ok = cur.pickup_success_ratio >= base.pickup_success_ratio - 0.03;
        let delivered_ok = cur.delivered_per_100_ticks >= base.delivered_per_100_ticks - 0.20;
        let mut failed = Vec::<&str>::new();
        if !repeated_ok {
            failed.push("repeated_goal_concentration");
        }
        if !pickup_ok {
            failed.push("pickup_success_ratio_regression");
        }
        if !delivered_ok {
            failed.push("delivered_per_100_ticks_regression");
        }
        if failed.is_empty() {
            println!(
                "coord-strict mode={} status=PASS repeated_goal_concentration={:.3} baseline={:.3} pickup_success_ratio={:.3} baseline={:.3} delivered_per_100_ticks={:.3} baseline={:.3}",
                mode,
                cur.repeated_goal_concentration,
                base.repeated_goal_concentration,
                cur.pickup_success_ratio,
                base.pickup_success_ratio,
                cur.delivered_per_100_ticks,
                base.delivered_per_100_ticks,
            );
        } else {
            any_failed = true;
            println!(
                "coord-strict mode={} status=FAIL failed={} repeated_goal_concentration={:.3} baseline={:.3} pickup_success_ratio={:.3} baseline={:.3} delivered_per_100_ticks={:.3} baseline={:.3}",
                mode,
                failed.join(","),
                cur.repeated_goal_concentration,
                base.repeated_goal_concentration,
                cur.pickup_success_ratio,
                base.pickup_success_ratio,
                cur.delivered_per_100_ticks,
                base.delivered_per_100_ticks,
            );
        }
    }
    any_failed
}

fn percentile(sorted: &[i64], p: f64) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((p / 100.0) * (sorted.len().saturating_sub(1) as f64)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn far_distance_threshold() -> f64 {
    8.0
}

fn nearest_dropoff_distance(bot_x: i64, bot_y: i64, dropoffs: &[(i64, i64)]) -> f64 {
    if dropoffs.is_empty() {
        return f64::INFINITY;
    }
    dropoffs
        .iter()
        .map(|(x, y)| (bot_x - *x).abs() as f64 + (bot_y - *y).abs() as f64)
        .fold(f64::INFINITY, f64::min)
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::parse_metrics;

    fn tick_line(
        tick: u64,
        action_kind: &str,
        carrying: &[&str],
        items_delivered_delta: u64,
    ) -> String {
        let carrying_json = carrying
            .iter()
            .map(|v| serde_json::Value::String((*v).to_owned()))
            .collect::<Vec<_>>();
        serde_json::json!({
            "event": "tick",
            "data": {
                "mode": "easy",
                "tick": tick,
                "actions": [
                    {"bot_id":"0", "kind": action_kind}
                ],
                "game_state": {
                    "bots": [
                        {"id":"0","x":1,"y":1,"carrying": carrying_json}
                    ]
                },
                "team_summary": {
                    "assignment_source": "hybrid_assignment",
                    "assignment_guard_reason": "none",
                    "assignment_goal_concentration_top3": 0.1,
                    "dropoff_congestion": 0,
                    "blocked_bot_count": 0,
                    "stuck_bot_count": 0
                },
                "tick_outcome": {
                    "delta_score": 0,
                    "items_delivered_delta": items_delivered_delta,
                    "order_completed_delta": 0
                }
            }
        })
        .to_string()
    }

    #[test]
    fn parse_metrics_infers_conversion_attempts_and_successes() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("run-eval-test-{suffix}.jsonl"));
        let content = vec![
            tick_line(0, "pick_up", &[], 0),
            tick_line(1, "drop_off", &["milk"], 0),
            tick_line(2, "wait", &[], 1),
            serde_json::json!({
                "event": "game_over",
                "data": {
                    "final_score": 10
                }
            })
            .to_string(),
        ]
        .join("\n");
        fs::write(&path, content).expect("write fixture");
        let parsed = parse_metrics(&path).expect("parsed metrics");
        assert_eq!(parsed.pickup_attempts, 1);
        assert_eq!(parsed.pickup_successes, 1);
        assert_eq!(parsed.dropoff_attempts, 1);
        assert_eq!(parsed.dropoff_successes, 1);
        assert!(parsed.items_delivered >= 1);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn parse_metrics_reports_spatial_wait_diagnostics() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("run-eval-spatial-test-{suffix}.jsonl"));
        let content = vec![
            serde_json::json!({
                "event": "tick",
                "data": {
                    "mode": "easy",
                    "tick": 0,
                    "actions": [{"bot_id":"0", "kind":"wait"}],
                    "game_state": {
                        "grid": {"drop_off_tiles": [[0,0]]},
                        "bots": [{"id":"0","x":9,"y":9,"capacity":1,"carrying":["bread"]}],
                        "orders": [{"id":"o_active","item_id":"milk","status":"in_progress"}]
                    },
                    "team_summary": {
                        "queue_roles": {"0":"collector"},
                        "assignment_source": "hybrid_assignment",
                        "assignment_guard_reason": "none",
                        "assignment_goal_concentration_top3": 0.1,
                        "dropoff_congestion": 0,
                        "blocked_bot_count": 0,
                        "stuck_bot_count": 0
                    },
                    "tick_outcome": {"delta_score":0,"items_delivered_delta":0,"order_completed_delta":0}
                }
            })
            .to_string(),
            serde_json::json!({
                "event": "tick",
                "data": {
                    "mode": "easy",
                    "tick": 1,
                    "actions": [{"bot_id":"0", "kind":"wait"}],
                    "game_state": {
                        "grid": {"drop_off_tiles": [[0,0]]},
                        "bots": [{"id":"0","x":9,"y":9,"capacity":1,"carrying":["bread"]}],
                        "orders": [{"id":"o_active","item_id":"milk","status":"in_progress"}]
                    },
                    "team_summary": {
                        "queue_roles": {"0":"collector"},
                        "assignment_source": "hybrid_assignment",
                        "assignment_guard_reason": "none",
                        "assignment_goal_concentration_top3": 0.1,
                        "dropoff_congestion": 0,
                        "blocked_bot_count": 0,
                        "stuck_bot_count": 0
                    },
                    "tick_outcome": {"delta_score":0,"items_delivered_delta":0,"order_completed_delta":0}
                }
            })
            .to_string(),
            serde_json::json!({
                "event": "game_over",
                "data": {"final_score": 0}
            })
            .to_string(),
        ]
        .join("\n");
        fs::write(&path, content).expect("write fixture");
        let parsed = parse_metrics(&path).expect("parsed metrics");
        assert!(parsed.collector_waits >= 2);
        assert!(parsed.collector_far_waits >= 2);
        assert!(parsed.full_inactive_waits >= 2);
        assert!(parsed.full_inactive_far_waits >= 2);
        assert!(parsed.far_no_conversion_ticks >= 1);
        let _ = fs::remove_file(path);
    }
}
