#![recursion_limit = "512"]

mod assign;
mod config;
mod dispatcher;
mod dist;
mod model;
mod motion;
mod net;
mod policy;
mod scoring;
mod team_context;
mod world;

use std::sync::Arc;

use base64::Engine;
use clap::Parser;
use config::ConfigArgs;
use model::{RuntimeContext, SessionMetadata};
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

    #[command(flatten)]
    config: ConfigArgs,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let config = Arc::new(cli.config.clone().build());
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(config.log_level.clone()));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();

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
