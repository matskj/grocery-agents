use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use base64::Engine;
use clap::Parser;
use grocery_agents::{
    config::{ConfigArgs, PolicyMode},
    model::{RuntimeContext, SessionMetadata},
    net, policy,
};
use serde::Deserialize;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Parser)]
#[command(name = "grocery-agents")]
struct Cli {
    /// Full websocket URL or plain token as a positional argument.
    ///
    /// Examples:
    /// - wss://game.ainm.no/ws?token=...
    /// - eyJ...
    #[arg(value_name = "WS_URL_OR_TOKEN")]
    connect: Option<String>,

    #[arg(long, env = "GROCERY_TOKEN")]
    token: Option<String>,

    #[arg(long, env = "GROCERY_WS_URL")]
    ws_url: Option<String>,

    #[arg(long)]
    replay: Option<PathBuf>,

    #[command(flatten)]
    config: ConfigArgs,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw_args = std::env::args().collect::<Vec<_>>();
    if raw_args.get(1).map(String::as_str) == Some("benchmark") {
        return run_benchmark_command(&raw_args[2..]);
    }

    let cli = Cli::parse();
    let config = Arc::new(cli.config.clone().build());
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(config.log_level.clone()));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();

    if let Some(replay) = cli.replay {
        let policy = policy::Policy::new(Arc::clone(&config));
        let summary = net::run_replay_file(&replay, policy, Arc::clone(&config))?;
        println!(
            "replay_summary ticks={} score={} orders={} items={} avg_order_rounds={:.2} avg_plan_ms={:.2} p95_plan_ms={} waits={} invalid_fixed={} est_rounds_120s={}",
            summary.ticks,
            summary.total_score,
            summary.orders_completed,
            summary.items_delivered,
            summary.avg_order_rounds_to_complete,
            summary.avg_planning_time_ms,
            summary.p95_planning_time_ms,
            summary.wait_actions,
            summary.invalid_actions_corrected,
            summary.estimated_rounds_in_120s
        );
        if summary.p95_planning_time_ms > 100 {
            eprintln!(
                "warning: replay p95 planning time {}ms exceeds 100ms target",
                summary.p95_planning_time_ms
            );
        }
        return Ok(());
    }

    let (token, ws_url) = resolve_connection(cli.connect, cli.token, cli.ws_url)?;
    let session = parse_session_metadata(&token).unwrap_or_default();
    let ctx = RuntimeContext {
        token,
        ws_url,
        session,
    };
    let policy = policy::Policy::new(Arc::clone(&config));

    net::run_game_loop(ctx, policy, config).await
}

fn run_benchmark_command(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        return Err("benchmark expects one or more replay paths/patterns".into());
    }
    let mut files = Vec::<PathBuf>::new();
    for arg in args {
        files.extend(expand_replay_pattern(arg));
    }
    files.sort();
    files.dedup();
    if files.is_empty() {
        return Err("no replay files matched benchmark arguments".into());
    }

    let policy_modes = [
        PolicyMode::Auto,
        PolicyMode::Easy,
        PolicyMode::Medium,
        PolicyMode::Hard,
        PolicyMode::Expert,
    ];

    #[derive(Default)]
    struct Agg {
        n: u64,
        score: f64,
        orders: f64,
        avg_plan: f64,
        p95: f64,
    }
    let mut agg = policy_modes
        .iter()
        .copied()
        .map(|mode| (mode, Agg::default()))
        .collect::<Vec<_>>();

    for file in &files {
        for mode in policy_modes {
            let mut cfg = Cli::parse_from(["grocery-agents"]).config.build();
            cfg.policy_mode = mode;
            let cfg = Arc::new(cfg);
            let policy = policy::Policy::new(Arc::clone(&cfg));
            let summary = net::run_replay_file(file, policy, Arc::clone(&cfg))?;
            let Some((_, entry)) = agg.iter_mut().find(|(m, _)| *m == mode) else {
                continue;
            };
            entry.n = entry.n.saturating_add(1);
            entry.score += summary.total_score as f64;
            entry.orders += summary.orders_completed as f64;
            entry.avg_plan += summary.avg_planning_time_ms;
            entry.p95 += summary.p95_planning_time_ms as f64;
        }
    }

    println!("policy\truns\tavg_score\tavg_orders\tavg_plan_ms\tavg_p95_ms");
    for mode in policy_modes {
        if let Some((_, a)) = agg.iter().find(|(m, _)| *m == mode) {
            let n = a.n.max(1) as f64;
            println!(
                "{:?}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
                mode,
                a.n,
                a.score / n,
                a.orders / n,
                a.avg_plan / n,
                a.p95 / n
            );
        }
    }
    Ok(())
}

fn expand_replay_pattern(pattern: &str) -> Vec<PathBuf> {
    let path = Path::new(pattern);
    if !pattern.contains('*') {
        if path.exists() {
            return vec![path.to_path_buf()];
        }
        return Vec::new();
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let needle = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_owned();
    let (prefix, suffix) = needle
        .split_once('*')
        .map(|(a, b)| (a.to_owned(), b.to_owned()))
        .unwrap_or_else(|| (needle.clone(), String::new()));
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if name.starts_with(&prefix) && name.ends_with(&suffix) {
                out.push(entry.path());
            }
        }
    }
    out
}

fn resolve_connection(
    connect: Option<String>,
    mut token: Option<String>,
    mut ws_url: Option<String>,
) -> Result<(String, Option<String>), Box<dyn std::error::Error>> {
    if let Some(connect) = connect {
        if looks_like_ws_url(&connect) {
            let (base, url_token) = split_ws_url_and_token(&connect)?;
            ws_url = Some(base);
            if token.is_none() {
                token = Some(url_token);
            }
        } else if token.is_none() {
            token = Some(connect);
        }
    }

    if token.is_none() {
        if let Some(url) = ws_url.clone() {
            let (base, url_token) = split_ws_url_and_token(&url)?;
            ws_url = Some(base);
            token = Some(url_token);
        }
    }

    let token = token.ok_or_else(|| {
        "missing token: provide --token, GROCERY_TOKEN, a full ws URL, or positional token"
            .to_owned()
    })?;

    Ok((token, ws_url))
}

fn looks_like_ws_url(value: &str) -> bool {
    value.starts_with("ws://") || value.starts_with("wss://")
}

fn split_ws_url_and_token(input: &str) -> Result<(String, String), Box<dyn std::error::Error>> {
    let (left, fragment) = match input.split_once('#') {
        Some((head, tail)) => (head, Some(tail)),
        None => (input, None),
    };

    let (base, query) = match left.split_once('?') {
        Some((base, query)) => (base.to_owned(), Some(query)),
        None => (left.to_owned(), None),
    };

    let mut token = None::<String>;
    let mut keep_pairs = Vec::new();
    if let Some(query) = query {
        for pair in query.split('&') {
            if pair.is_empty() {
                continue;
            }
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            if key == "token" && !value.is_empty() {
                token = Some(value.to_owned());
            } else {
                keep_pairs.push(pair.to_owned());
            }
        }
    }

    let token =
        token.ok_or_else(|| "ws URL is missing token query parameter (?token=...)".to_owned())?;

    let mut rebuilt = base;
    if !keep_pairs.is_empty() {
        rebuilt.push('?');
        rebuilt.push_str(&keep_pairs.join("&"));
    }
    if let Some(frag) = fragment {
        rebuilt.push('#');
        rebuilt.push_str(frag);
    }

    Ok((rebuilt, token))
}

#[derive(Debug, Deserialize, Default)]
struct JwtClaims {
    team_id: Option<String>,
    map_id: Option<String>,
    difficulty: Option<String>,
    map_seed: Option<i64>,
}

fn parse_session_metadata(token: &str) -> Option<SessionMetadata> {
    let payload = token.split('.').nth(1)?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.as_bytes())
        .ok()?;
    let claims: JwtClaims = serde_json::from_slice(&decoded).ok()?;
    Some(SessionMetadata {
        team_id: claims.team_id,
        map_id: claims.map_id,
        difficulty: claims.difficulty,
        map_seed: claims.map_seed,
    })
}
