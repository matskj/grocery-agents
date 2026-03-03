use std::{
    collections::HashMap,
    fs::{create_dir_all, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, OnceLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, Message},
    MaybeTlsStream, WebSocketStream,
};
use tracing::{info, warn};

use crate::{
    config::Config,
    dispatcher::Dispatcher,
    model::{
        to_wire_action_envelope, Action, ActionEnvelope, BotState, GameOver, GameState,
        OrderStatus, RuntimeContext, WireServerMessage,
    },
    policy::Policy,
    scoring::detect_mode_label,
};

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;
const ROUND_PLANNING_BUDGET: Duration = Duration::from_millis(1_400);
const LOG_SCHEMA_VERSION: &str = "1.2.0";

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum ServerMessage {
    Wire(WireServerMessage),
    LegacyGameStateEnvelope { game_state: GameState },
    LegacyGameState(GameState),
    LegacyGameOverEnvelope { game_over: GameOver },
    LegacyGameOver(GameOver),
}

#[derive(Debug, Clone)]
struct TickAnalytics {
    team_summary: serde_json::Value,
    tick_outcome: serde_json::Value,
}

#[derive(Debug, Clone)]
struct PlanRoundResult {
    actions: Vec<Action>,
    invalid_action_count: u64,
    team_telemetry: serde_json::Value,
    action_validated_by_bot: HashMap<String, bool>,
    plan_ms: u64,
    assign_ms: u64,
}

async fn connect_with_base_url(
    base_url: &str,
    token: &str,
) -> Result<WsStream, tungstenite::Error> {
    let separator = if base_url.contains('?') { '&' } else { '?' };
    let url = format!("{base_url}{separator}token={token}");
    let (stream, _) = connect_async(url).await?;
    Ok(stream)
}

pub async fn run_game_loop(
    ctx: RuntimeContext,
    mut policy: Policy,
    config: Arc<Config>,
) -> Result<(), Box<dyn std::error::Error>> {
    let base_url = ctx.ws_url.clone().unwrap_or_else(|| {
        std::env::var("GROCERY_WS_URL").unwrap_or_else(|_| "ws://localhost:8765/ws".to_owned())
    });
    let mut run_logger = RunLogger::new(&base_url, config.replay_dump_path.clone());
    let mut socket = match connect_with_base_url(&base_url, &ctx.token).await {
        Ok(socket) => socket,
        Err(err) => {
            run_logger.log(
                "connect_error",
                serde_json::json!({
                    "error": err.to_string(),
                }),
            );
            return Err(Box::new(err));
        }
    };
    let dispatcher = Dispatcher::new();

    run_logger.log(
        "session_start",
        serde_json::json!({
            "schema_version": LOG_SCHEMA_VERSION,
            "run_id": run_logger.run_id(),
            "ws_url": base_url,
            "token_preview": token_preview(&ctx.token),
            "planning_budget_ms": ROUND_PLANNING_BUDGET.as_millis(),
            "mode": ctx.session.difficulty.as_deref().unwrap_or("unknown"),
            "map_id": ctx.session.map_id,
            "difficulty": ctx.session.difficulty,
            "team_id": ctx.session.team_id,
            "map_seed": ctx.session.map_seed,
            "build_version": build_version(),
        }),
    );

    info!("connected websocket, entering receive loop");
    let mut mode_logged = false;

    while let Some(frame) = socket.next().await {
        match frame {
            Ok(Message::Text(text)) => {
                let msg = match serde_json::from_str::<ServerMessage>(&text) {
                    Ok(m) => m,
                    Err(err) => {
                        warn!(error = %err, payload = %text, "failed to parse server message");
                        continue;
                    }
                };

                match msg {
                    ServerMessage::Wire(WireServerMessage::GameState(wire_state)) => {
                        let game_state = GameState::from_wire(wire_state);
                        handle_game_state(
                            &mut socket,
                            &mut policy,
                            &dispatcher,
                            &mut run_logger,
                            &mut mode_logged,
                            config.as_ref(),
                            game_state,
                        )
                        .await?;
                    }
                    ServerMessage::LegacyGameStateEnvelope { game_state }
                    | ServerMessage::LegacyGameState(game_state) => {
                        handle_game_state(
                            &mut socket,
                            &mut policy,
                            &dispatcher,
                            &mut run_logger,
                            &mut mode_logged,
                            config.as_ref(),
                            game_state,
                        )
                        .await?;
                    }
                    ServerMessage::Wire(WireServerMessage::GameOver(wire_game_over)) => {
                        let game_over = GameOver::from_wire(wire_game_over);
                        let reason = game_over.reason.clone();
                        run_logger.log(
                            "game_over",
                            serde_json::json!({
                                "mode": run_logger.last_mode.as_deref().unwrap_or("unknown"),
                                "final_score": game_over.final_score,
                                "reason": reason,
                                "episode_counters": run_logger.episode_counters_json(),
                            }),
                        );
                        info!(
                            final_score = game_over.final_score,
                            reason = ?game_over.reason,
                            "game over received"
                        );
                        break;
                    }
                    ServerMessage::LegacyGameOverEnvelope { game_over }
                    | ServerMessage::LegacyGameOver(game_over) => {
                        let reason = game_over.reason.clone();
                        run_logger.log(
                            "game_over",
                            serde_json::json!({
                                "mode": run_logger.last_mode.as_deref().unwrap_or("unknown"),
                                "final_score": game_over.final_score,
                                "reason": reason,
                                "episode_counters": run_logger.episode_counters_json(),
                            }),
                        );
                        info!(
                            final_score = game_over.final_score,
                            reason = ?game_over.reason,
                            "game over received"
                        );
                        break;
                    }
                }
            }
            Ok(Message::Ping(payload)) => {
                socket.send(Message::Pong(payload)).await?;
            }
            Ok(Message::Close(frame)) => {
                run_logger.log(
                    "socket_close",
                    serde_json::json!({
                        "frame": format!("{frame:?}"),
                    }),
                );
                info!(?frame, "server closed websocket");
                break;
            }
            Ok(Message::Binary(_)) | Ok(Message::Pong(_)) => {}
            Ok(Message::Frame(_)) => {}
            Err(err) => {
                run_logger.log(
                    "socket_error",
                    serde_json::json!({
                        "error": err.to_string(),
                    }),
                );
                warn!(error = %err, "websocket receive error, terminating loop");
                break;
            }
        }
    }

    Ok(())
}

async fn handle_game_state(
    socket: &mut WsStream,
    policy: &mut Policy,
    dispatcher: &Dispatcher,
    run_logger: &mut RunLogger,
    mode_logged: &mut bool,
    config: &Config,
    game_state: GameState,
) -> Result<(), Box<dyn std::error::Error>> {
    if !*mode_logged {
        run_logger.log("game_mode", game_mode_payload(&game_state));
        *mode_logged = true;
    }

    let planned = timeout(
        ROUND_PLANNING_BUDGET,
        plan_round_actions(policy, dispatcher, &game_state, config),
    )
    .await
    .unwrap_or_else(|_| {
        warn!(
            tick = game_state.tick,
            "planning timeout, falling back to wait actions"
        );
        if bot_debug_enabled() {
            warn!(
                tick = game_state.tick,
                budget_ms = ROUND_PLANNING_BUDGET.as_millis(),
                bot_count = game_state.bots.len(),
                "time-budget event: fallback wait envelope emitted"
            );
        }
        fallback_wait_actions(&game_state)
    });

    let envelope = ActionEnvelope {
        actions: planned.actions.clone(),
    };
    let analytics = run_logger.observe_tick(
        &game_state,
        &envelope.actions,
        planned.invalid_action_count,
        &planned.team_telemetry,
    );
    if config.structured_bot_log {
        log_structured_bot_ticks(run_logger, &game_state, &envelope.actions, &planned);
    }
    if config.ascii_render {
        let frame = render_ascii_frame(&game_state, &envelope.actions, &planned.team_telemetry);
        run_logger.log(
            "ascii_render",
            serde_json::json!({
                "tick": game_state.tick,
                "frame": frame,
            }),
        );
    }
    let payload = serde_json::to_string(&to_wire_action_envelope(&envelope.actions))?;
    let tick = game_state.tick;
    info!(
        tick,
        bots = game_state.bots.len(),
        items = game_state.items.len(),
        orders = game_state.orders.len(),
        bot_state = %bot_state_summary(&game_state),
        order_state = %order_summary(&game_state),
        actions = %action_summary(&envelope.actions),
        "tick summary"
    );
    run_logger.log(
        "tick",
        serde_json::json!({
            "run_id": run_logger.run_id(),
            "mode": detect_mode_label(&game_state),
            "game_mode": detect_mode_label(&game_state),
            "tick": tick,
            "game_state": &game_state,
            "actions": &envelope.actions,
            "team_summary": analytics.team_summary,
            "tick_outcome": analytics.tick_outcome,
        }),
    );
    run_logger.log(
        "tick_outcome",
        serde_json::json!({
            "run_id": run_logger.run_id(),
            "mode": detect_mode_label(&game_state),
            "game_mode": detect_mode_label(&game_state),
            "tick": tick,
            "tick_outcome": analytics.tick_outcome,
        }),
    );
    run_logger.dump_replay(&game_state, &envelope.actions, &planned.team_telemetry);
    socket.send(Message::Text(payload.into())).await?;
    info!(tick, "sent round action envelope");
    Ok(())
}

struct RunLogger {
    file: Option<File>,
    path: Option<PathBuf>,
    replay_dump_file: Option<File>,
    run_id: String,
    last_score: Option<i64>,
    last_items_remaining: Option<i64>,
    last_active_order_index: Option<i64>,
    prev_positions: HashMap<String, (i32, i32)>,
    blocked_ticks: HashMap<String, u8>,
    last_mode: Option<String>,
    episode_moves: u64,
    episode_waits: u64,
    episode_pickups: u64,
    episode_dropoffs: u64,
    episode_invalids_prevented: u64,
    episode_blocked_events: u64,
    episode_near_dropoff_congestion_events: u64,
}

impl RunLogger {
    fn new(base_url: &str, replay_dump_path: Option<PathBuf>) -> Self {
        let dir = std::env::var("GAME_LOG_DIR").unwrap_or_else(|_| "logs".to_owned());
        let dir_path = Path::new(&dir);
        if let Err(err) = create_dir_all(dir_path) {
            warn!(error = %err, dir = %dir, "failed to create game log directory");
            return Self {
                file: None,
                path: None,
                replay_dump_file: None,
                run_id: format!("run-{}", now_unix_millis()),
                last_score: None,
                last_items_remaining: None,
                last_active_order_index: None,
                prev_positions: HashMap::new(),
                blocked_ticks: HashMap::new(),
                last_mode: None,
                episode_moves: 0,
                episode_waits: 0,
                episode_pickups: 0,
                episode_dropoffs: 0,
                episode_invalids_prevented: 0,
                episode_blocked_events: 0,
                episode_near_dropoff_congestion_events: 0,
            };
        }

        let stamp = now_unix_millis();
        let run_id = format!("run-{stamp}");
        let path = dir_path.join(format!("{run_id}.jsonl"));
        let mut file = match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => file,
            Err(err) => {
                warn!(error = %err, path = %path.display(), "failed to open game log file");
                return Self {
                    file: None,
                    path: None,
                    replay_dump_file: None,
                    run_id,
                    last_score: None,
                    last_items_remaining: None,
                    last_active_order_index: None,
                    prev_positions: HashMap::new(),
                    blocked_ticks: HashMap::new(),
                    last_mode: None,
                    episode_moves: 0,
                    episode_waits: 0,
                    episode_pickups: 0,
                    episode_dropoffs: 0,
                    episode_invalids_prevented: 0,
                    episode_blocked_events: 0,
                    episode_near_dropoff_congestion_events: 0,
                };
            }
        };

        let header = serde_json::json!({
            "event": "log_opened",
            "run_id": run_id,
            "schema_version": LOG_SCHEMA_VERSION,
            "ts_ms": now_unix_millis(),
            "ws_url": base_url,
        });
        let line = serde_json::to_string(&header)
            .unwrap_or_else(|_| "{\"event\":\"log_opened\"}".to_owned());
        let _ = writeln!(file, "{line}");

        let replay_dump_file = replay_dump_path.and_then(|path| {
            if let Some(parent) = path.parent() {
                if let Err(err) = create_dir_all(parent) {
                    warn!(
                        error = %err,
                        path = %path.display(),
                        "failed creating replay dump parent"
                    );
                    return None;
                }
            }
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(mut replay_file) => {
                    let header = serde_json::json!({
                        "event": "replay_dump_opened",
                        "run_id": run_id,
                        "schema_version": LOG_SCHEMA_VERSION,
                        "ts_ms": now_unix_millis(),
                    });
                    if let Ok(line) = serde_json::to_string(&header) {
                        let _ = writeln!(replay_file, "{line}");
                    }
                    Some(replay_file)
                }
                Err(err) => {
                    warn!(
                        error = %err,
                        path = %path.display(),
                        "failed opening replay dump file"
                    );
                    None
                }
            }
        });

        info!(path = %path.display(), run_id = %run_id, "game log enabled");
        Self {
            file: Some(file),
            path: Some(path),
            replay_dump_file,
            run_id,
            last_score: None,
            last_items_remaining: None,
            last_active_order_index: None,
            prev_positions: HashMap::new(),
            blocked_ticks: HashMap::new(),
            last_mode: None,
            episode_moves: 0,
            episode_waits: 0,
            episode_pickups: 0,
            episode_dropoffs: 0,
            episode_invalids_prevented: 0,
            episode_blocked_events: 0,
            episode_near_dropoff_congestion_events: 0,
        }
    }

    fn run_id(&self) -> &str {
        &self.run_id
    }

    fn log(&mut self, event: &str, payload: serde_json::Value) {
        let Some(file) = self.file.as_mut() else {
            return;
        };

        let mut record = serde_json::Map::new();
        record.insert(
            "event".to_owned(),
            serde_json::Value::String(event.to_owned()),
        );
        record.insert(
            "run_id".to_owned(),
            serde_json::Value::String(self.run_id.clone()),
        );
        record.insert(
            "schema_version".to_owned(),
            serde_json::Value::String(LOG_SCHEMA_VERSION.to_owned()),
        );
        record.insert(
            "ts_ms".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(now_unix_millis())),
        );
        record.insert("data".to_owned(), payload);

        match serde_json::to_string(&serde_json::Value::Object(record)) {
            Ok(line) => {
                if writeln!(file, "{line}").is_err() {
                    self.file = None;
                    if let Some(path) = &self.path {
                        warn!(path = %path.display(), "failed writing game log; disabling logger");
                    }
                }
            }
            Err(err) => {
                warn!(error = %err, "failed to serialize game log record");
            }
        }
    }

    fn dump_replay(
        &mut self,
        state: &GameState,
        actions: &[Action],
        team_telemetry: &serde_json::Value,
    ) {
        let Some(file) = self.replay_dump_file.as_mut() else {
            return;
        };
        let record = serde_json::json!({
            "event": "tick_replay",
            "run_id": self.run_id,
            "schema_version": LOG_SCHEMA_VERSION,
            "ts_ms": now_unix_millis(),
            "tick": state.tick,
            "game_state": state,
            "actions": actions,
            "team_telemetry": team_telemetry,
        });
        match serde_json::to_string(&record) {
            Ok(line) => {
                if writeln!(file, "{line}").is_err() {
                    self.replay_dump_file = None;
                }
            }
            Err(err) => {
                warn!(error = %err, "failed to serialize replay dump tick");
            }
        }
    }

    fn episode_counters_json(&self) -> serde_json::Value {
        serde_json::json!({
            "moves": self.episode_moves,
            "waits": self.episode_waits,
            "pickups": self.episode_pickups,
            "dropoffs": self.episode_dropoffs,
            "invalids_prevented": self.episode_invalids_prevented,
            "blocked_events": self.episode_blocked_events,
            "near_dropoff_congestion_events": self.episode_near_dropoff_congestion_events,
        })
    }

    fn observe_tick(
        &mut self,
        state: &GameState,
        actions: &[Action],
        invalid_action_count: u64,
        team_telemetry: &serde_json::Value,
    ) -> TickAnalytics {
        let mode = detect_mode_label(state).to_owned();
        self.last_mode = Some(mode.clone());

        let mut active_remaining: HashMap<String, u64> = HashMap::new();
        let mut preview_remaining: HashMap<String, u64> = HashMap::new();
        for order in &state.orders {
            match order.status {
                OrderStatus::InProgress => {
                    *active_remaining.entry(order.item_id.clone()).or_insert(0) += 1;
                }
                OrderStatus::Pending => {
                    *preview_remaining.entry(order.item_id.clone()).or_insert(0) += 1;
                }
                OrderStatus::Delivered | OrderStatus::Cancelled => {}
            }
        }

        let mut blocked_bot_count = 0u64;
        let mut stuck_bot_count = 0u64;
        let mut now_positions: HashMap<String, (i32, i32)> = HashMap::new();
        for bot in &state.bots {
            now_positions.insert(bot.id.clone(), (bot.x, bot.y));
            let prev = self.prev_positions.get(&bot.id).copied();
            let next = if prev == Some((bot.x, bot.y)) {
                self.blocked_ticks
                    .get(&bot.id)
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(1)
            } else {
                0
            };
            self.blocked_ticks.insert(bot.id.clone(), next);
            if next >= 1 {
                blocked_bot_count += 1;
            }
            if next >= 2 {
                stuck_bot_count += 1;
            }
        }
        self.prev_positions = now_positions;

        let score = state.score;
        let delta_score = self.last_score.map(|v| score - v).unwrap_or_default();
        self.last_score = Some(score);

        let items_remaining = state.orders.len() as i64;
        let items_delivered_delta = self
            .last_items_remaining
            .map(|v| (v - items_remaining).max(0))
            .unwrap_or_default();
        self.last_items_remaining = Some(items_remaining);

        let order_completed_delta = self
            .last_active_order_index
            .map(|v| (state.active_order_index - v).max(0))
            .unwrap_or_default();
        self.last_active_order_index = Some(state.active_order_index);

        let dropoff_congestion = compute_dropoff_congestion(state);
        let wait_actions = actions
            .iter()
            .filter(|action| matches!(action, Action::Wait { .. }))
            .count() as u64;
        let non_wait_actions = actions.len() as u64 - wait_actions;
        let move_actions = actions
            .iter()
            .filter(|action| matches!(action, Action::Move { .. }))
            .count() as u64;
        let pickup_actions = actions
            .iter()
            .filter(|action| matches!(action, Action::PickUp { .. }))
            .count() as u64;
        let dropoff_actions = actions
            .iter()
            .filter(|action| matches!(action, Action::DropOff { .. }))
            .count() as u64;
        self.episode_moves = self.episode_moves.saturating_add(move_actions);
        self.episode_waits = self.episode_waits.saturating_add(wait_actions);
        self.episode_pickups = self.episode_pickups.saturating_add(pickup_actions);
        self.episode_dropoffs = self.episode_dropoffs.saturating_add(dropoff_actions);
        self.episode_invalids_prevented = self
            .episode_invalids_prevented
            .saturating_add(invalid_action_count);
        self.episode_blocked_events = self
            .episode_blocked_events
            .saturating_add(blocked_bot_count);
        if dropoff_congestion >= 2 {
            self.episode_near_dropoff_congestion_events = self
                .episode_near_dropoff_congestion_events
                .saturating_add(1);
        }

        let mut team_summary = serde_json::Map::new();
        team_summary.insert("mode".to_owned(), serde_json::Value::String(mode));
        team_summary.insert(
            "active_order_remaining_by_item".to_owned(),
            serde_json::to_value(active_remaining).unwrap_or_else(|_| serde_json::json!({})),
        );
        team_summary.insert(
            "preview_order_remaining_by_item".to_owned(),
            serde_json::to_value(preview_remaining).unwrap_or_else(|_| serde_json::json!({})),
        );
        team_summary.insert(
            "dropoff_congestion".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(dropoff_congestion)),
        );
        team_summary.insert(
            "blocked_bot_count".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(blocked_bot_count)),
        );
        team_summary.insert(
            "stuck_bot_count".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(stuck_bot_count)),
        );
        team_summary.insert(
            "wait_action_count".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(wait_actions)),
        );
        team_summary.insert(
            "non_wait_action_count".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(non_wait_actions)),
        );
        team_summary.insert("episode_counters".to_owned(), self.episode_counters_json());
        if let Some(extra) = team_telemetry.as_object() {
            for (key, value) in extra {
                team_summary.insert(key.clone(), value.clone());
            }
        }

        TickAnalytics {
            team_summary: serde_json::Value::Object(team_summary),
            tick_outcome: serde_json::json!({
                "delta_score": delta_score,
                "items_delivered_delta": items_delivered_delta,
                "order_completed_delta": order_completed_delta,
                "invalid_action_count": invalid_action_count,
            }),
        }
    }
}

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn token_preview(token: &str) -> String {
    let keep = token.chars().rev().take(8).collect::<String>();
    let suffix = keep.chars().rev().collect::<String>();
    format!("...{suffix}")
}

fn bot_state_summary(state: &GameState) -> String {
    state
        .bots
        .iter()
        .map(|bot| {
            let carrying = if bot.carrying.is_empty() {
                "-".to_owned()
            } else {
                bot.carrying.join("|")
            };
            format!(
                "{}@({},{}):{}/{}[{}]",
                bot.id,
                bot.x,
                bot.y,
                bot.carrying.len(),
                bot.capacity,
                carrying
            )
        })
        .collect::<Vec<_>>()
        .join(" ; ")
}

fn order_summary(state: &GameState) -> String {
    let mut parts = state
        .orders
        .iter()
        .map(|order| format!("{}:{}:{:?}", order.id, order.item_id, order.status))
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return "none".to_owned();
    }
    if parts.len() > 6 {
        parts.truncate(6);
        parts.push("...".to_owned());
    }
    parts.join(" ; ")
}

fn action_summary(actions: &[Action]) -> String {
    actions
        .iter()
        .map(action_label)
        .collect::<Vec<_>>()
        .join(" ; ")
}

fn action_label(action: &Action) -> String {
    match action {
        Action::Move { bot_id, dx, dy } => format!("{bot_id}:move({dx},{dy})"),
        Action::PickUp { bot_id, item_id } => format!("{bot_id}:pick({item_id})"),
        Action::DropOff { bot_id, order_id } => format!("{bot_id}:drop({order_id})"),
        Action::Wait { bot_id } => format!("{bot_id}:wait"),
    }
}

fn log_structured_bot_ticks(
    run_logger: &mut RunLogger,
    state: &GameState,
    actions: &[Action],
    planned: &PlanRoundResult,
) {
    let mut action_by_bot: HashMap<&str, &Action> = HashMap::new();
    for action in actions {
        action_by_bot.insert(action.bot_id(), action);
    }
    let mut bots = state.bots.iter().collect::<Vec<_>>();
    bots.sort_by(|a, b| a.id.cmp(&b.id));

    let intents = planned
        .team_telemetry
        .get("selected_intents")
        .and_then(serde_json::Value::as_object);
    let goals = planned
        .team_telemetry
        .get("goal_cell_by_bot")
        .and_then(serde_json::Value::as_object);
    let blocked = planned
        .team_telemetry
        .get("blocked_ticks_by_bot")
        .and_then(serde_json::Value::as_object);
    let fallback_stage = planned
        .team_telemetry
        .get("planner_fallback_stage_by_bot")
        .and_then(serde_json::Value::as_object);
    let wait_reason = planned
        .team_telemetry
        .get("wait_reason_by_bot")
        .and_then(serde_json::Value::as_object);

    for bot in bots {
        let action = action_by_bot
            .get(bot.id.as_str())
            .copied()
            .map(action_label)
            .unwrap_or_else(|| format!("{}:wait", bot.id));
        let intent = intents
            .and_then(|m| m.get(&bot.id))
            .cloned()
            .unwrap_or_else(|| serde_json::Value::String("unknown".to_owned()));
        let goal_cell = goals
            .and_then(|m| m.get(&bot.id))
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let blocked_ticks = blocked
            .and_then(|m| m.get(&bot.id))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let replan_reason = fallback_stage
            .and_then(|m| m.get(&bot.id))
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                wait_reason
                    .and_then(|m| m.get(&bot.id))
                    .and_then(serde_json::Value::as_str)
            })
            .unwrap_or("none")
            .to_owned();

        run_logger.log(
            "bot_tick",
            serde_json::json!({
                "tick": state.tick,
                "bot_id": bot.id,
                "pos": [bot.x, bot.y],
                "intent": intent,
                "goal_cell": goal_cell,
                "chosen_action": action,
                "action_validated": planned
                    .action_validated_by_bot
                    .get(&bot.id)
                    .copied()
                    .unwrap_or(false),
                "blocked_ticks": blocked_ticks,
                "replan_reason": replan_reason,
                "plan_ms": planned.plan_ms,
                "assign_ms": planned.assign_ms,
            }),
        );
    }
}

fn render_ascii_frame(
    state: &GameState,
    actions: &[Action],
    team_telemetry: &serde_json::Value,
) -> String {
    let width = state.grid.width.max(0) as usize;
    let height = state.grid.height.max(0) as usize;
    if width == 0 || height == 0 {
        return "empty-grid".to_owned();
    }

    let mut cells = vec![vec!['.'; width]; height];
    for [x, y] in &state.grid.walls {
        if *x >= 0 && *y >= 0 && (*x as usize) < width && (*y as usize) < height {
            cells[*y as usize][*x as usize] = '#';
        }
    }
    for [x, y] in &state.grid.drop_off_tiles {
        if *x >= 0 && *y >= 0 && (*x as usize) < width && (*y as usize) < height {
            cells[*y as usize][*x as usize] = 'D';
        }
    }
    for bot in &state.bots {
        if bot.x >= 0 && bot.y >= 0 && (bot.x as usize) < width && (bot.y as usize) < height {
            let ch = bot.id.chars().next().unwrap_or('B');
            cells[bot.y as usize][bot.x as usize] = ch;
        }
    }

    let mut lines = cells
        .into_iter()
        .map(|row| row.into_iter().collect::<String>())
        .collect::<Vec<_>>();

    let actions_line = format!(
        "actions: {}",
        actions
            .iter()
            .map(action_label)
            .collect::<Vec<_>>()
            .join(" ; ")
    );
    lines.push(actions_line);
    if let Some(reserved) = team_telemetry.get("reserved_cells_by_t") {
        lines.push(format!("reserved_cells_by_t: {reserved}"));
    }
    lines.join("\n")
}

fn game_mode_payload(state: &GameState) -> serde_json::Value {
    serde_json::json!({
        "mode": detect_mode_label(state),
        "bots": state.bots.len(),
        "grid": {
            "width": state.grid.width,
            "height": state.grid.height,
        },
    })
}

async fn plan_round_actions(
    policy: &mut Policy,
    dispatcher: &Dispatcher,
    state: &GameState,
    config: &Config,
) -> PlanRoundResult {
    let plan_started = Instant::now();
    let proposed = policy.decide_round(
        state,
        Duration::from_millis(config.planner_soft_budget_ms),
    );
    let team_telemetry = policy.last_team_telemetry();
    let assign_ms = team_telemetry
        .get("assign_ms")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let mut invalid = 0u64;
    let mut action_validated_by_bot = HashMap::new();
    let actions = state
        .bots
        .iter()
        .zip(proposed.into_iter())
        .map(|(bot, action)| {
            let dispatched = dispatcher.dispatch(action);
            let (validated, was_invalid) = validate_action(dispatched, state, bot);
            if was_invalid {
                invalid += 1;
            }
            action_validated_by_bot.insert(bot.id.clone(), !was_invalid);
            validated
        })
        .collect::<Vec<_>>();

    PlanRoundResult {
        actions,
        invalid_action_count: invalid,
        team_telemetry,
        action_validated_by_bot,
        plan_ms: plan_started.elapsed().as_millis() as u64,
        assign_ms,
    }
}

fn fallback_wait_actions(state: &GameState) -> PlanRoundResult {
    if bot_debug_enabled() {
        warn!(
            tick = state.tick,
            bot_count = state.bots.len(),
            "safety fallback activated: forcing wait actions"
        );
    }
    let wait_reason_by_bot = state
        .bots
        .iter()
        .map(|bot| {
            (
                bot.id.clone(),
                serde_json::Value::String("timeout_fallback".to_owned()),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let fallback_stage_by_bot = state
        .bots
        .iter()
        .map(|bot| {
            (
                bot.id.clone(),
                serde_json::Value::String("timeout_fallback".to_owned()),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let ordering_stage_by_bot = state
        .bots
        .iter()
        .map(|bot| {
            (
                bot.id.clone(),
                serde_json::Value::String("timeout_fallback".to_owned()),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let intent_move_but_wait_by_bot = state
        .bots
        .iter()
        .map(|bot| (bot.id.clone(), serde_json::Value::Bool(false)))
        .collect::<serde_json::Map<_, _>>();
    let queue_relaxation_active_by_bot = state
        .bots
        .iter()
        .map(|bot| (bot.id.clone(), serde_json::Value::Bool(false)))
        .collect::<serde_json::Map<_, _>>();
    let empty_count_map = state
        .bots
        .iter()
        .map(|bot| {
            (
                bot.id.clone(),
                serde_json::Value::Number(serde_json::Number::from(0)),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let none_string_map = state
        .bots
        .iter()
        .map(|bot| (bot.id.clone(), serde_json::Value::String("none".to_owned())))
        .collect::<serde_json::Map<_, _>>();

    PlanRoundResult {
        actions: state
            .bots
            .iter()
            .map(|bot| Action::wait(bot.id.clone()))
            .collect(),
        invalid_action_count: state.bots.len() as u64,
        team_telemetry: serde_json::json!({
            "wait_reason_by_bot": wait_reason_by_bot,
            "planner_fallback_stage_by_bot": fallback_stage_by_bot,
            "ordering_stage_by_bot": ordering_stage_by_bot,
            "intent_move_but_wait_by_bot": intent_move_but_wait_by_bot,
            "queue_relaxation_active_by_bot": queue_relaxation_active_by_bot,
            "local_conflict_count_by_bot": empty_count_map,
            "cbs_timeout": false,
            "cbs_expanded_nodes": 0,
            "dropoff_target_status_by_bot": none_string_map,
            "assign_ms": 0,
        }),
        action_validated_by_bot: state
            .bots
            .iter()
            .map(|bot| (bot.id.clone(), false))
            .collect(),
        plan_ms: 0,
        assign_ms: 0,
    }
}

fn validate_action(action: Action, state: &GameState, bot: &BotState) -> (Action, bool) {
    if action.bot_id() != bot.id {
        if bot_debug_enabled() {
            warn!(tick = state.tick, bot_id = %bot.id, "fallback event: action bot_id mismatch; waiting");
        }
        return (Action::wait(bot.id.clone()), true);
    }

    match action {
        Action::Move { bot_id, dx, dy } => {
            let nx = bot.x + dx;
            let ny = bot.y + dy;
            let in_bounds = nx >= 0 && ny >= 0 && nx < state.grid.width && ny < state.grid.height;
            let blocked_by_wall = state
                .grid
                .walls
                .iter()
                .any(|wall| wall[0] == nx && wall[1] == ny);
            let blocked_by_item = state.items.iter().any(|item| item.x == nx && item.y == ny);
            let blocked = blocked_by_wall || blocked_by_item;
            if in_bounds && !blocked {
                (Action::Move { bot_id, dx, dy }, false)
            } else {
                if bot_debug_enabled() {
                    warn!(tick = state.tick, bot_id = %bot.id, nx, ny, in_bounds, blocked_by_wall, blocked_by_item, "safety fallback: invalid move converted to wait");
                }
                (Action::wait(bot.id.clone()), true)
            }
        }
        Action::PickUp { bot_id, item_id } => {
            let has_capacity = bot.carrying.len() < bot.capacity;
            let maybe_item = state.items.iter().find(|item| item.id == item_id);
            let adjacent = maybe_item
                .map(|item| (item.x - bot.x).abs() + (item.y - bot.y).abs() == 1)
                .unwrap_or(false);
            if has_capacity && adjacent {
                (Action::PickUp { bot_id, item_id }, false)
            } else {
                if bot_debug_enabled() {
                    warn!(tick = state.tick, bot_id = %bot.id, has_capacity, adjacent, "safety fallback: invalid pickup converted to wait");
                }
                (Action::wait(bot.id.clone()), true)
            }
        }
        Action::DropOff { bot_id, order_id } => {
            let on_drop_off = state
                .grid
                .drop_off_tiles
                .iter()
                .any(|tile| tile[0] == bot.x && tile[1] == bot.y);
            let valid_order = state
                .orders
                .iter()
                .find(|order| order.id == order_id)
                .map(|order| {
                    matches!(order.status, OrderStatus::InProgress)
                        && bot.carrying.iter().any(|item| item == &order.item_id)
                })
                .unwrap_or(false);
            if on_drop_off && valid_order {
                (Action::DropOff { bot_id, order_id }, false)
            } else {
                if bot_debug_enabled() {
                    warn!(
                        tick = state.tick,
                        bot_id = %bot.id,
                        on_drop_off,
                        valid_order,
                        "safety fallback: invalid dropoff converted to wait"
                    );
                }
                (Action::wait(bot.id.clone()), true)
            }
        }
        Action::Wait { bot_id } => (Action::Wait { bot_id }, false),
    }
}

fn compute_dropoff_congestion(state: &GameState) -> u64 {
    if state.grid.drop_off_tiles.is_empty() {
        return 0;
    }
    state
        .grid
        .drop_off_tiles
        .iter()
        .map(|tile| {
            state
                .bots
                .iter()
                .filter(|bot| (bot.x - tile[0]).abs() + (bot.y - tile[1]).abs() <= 1)
                .count() as u64
        })
        .max()
        .unwrap_or(0)
}

fn build_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION
        .get_or_init(|| {
            if let Ok(v) = std::env::var("GIT_COMMIT_HASH") {
                let trimmed = v.trim().to_owned();
                if !trimmed.is_empty() {
                    return trimmed;
                }
            }
            if let Some(v) = option_env!("GIT_COMMIT_HASH") {
                if !v.trim().is_empty() {
                    return v.to_owned();
                }
            }
            Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .output()
                .ok()
                .and_then(|out| String::from_utf8(out.stdout).ok())
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "unknown".to_owned())
        })
        .as_str()
}

fn bot_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("BOT_DEBUG")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use crate::model::{Action, BotState, GameState, Grid, Order, OrderStatus};

    use super::validate_action;

    fn base_state(order_status: OrderStatus) -> (GameState, BotState) {
        let bot = BotState {
            id: "0".to_owned(),
            x: 1,
            y: 1,
            carrying: vec!["milk".to_owned()],
            capacity: 3,
        };
        let state = GameState {
            grid: Grid {
                width: 4,
                height: 4,
                drop_off_tiles: vec![[1, 1]],
                ..Grid::default()
            },
            bots: vec![bot.clone()],
            orders: vec![Order {
                id: "o1".to_owned(),
                item_id: "milk".to_owned(),
                status: order_status,
            }],
            ..GameState::default()
        };
        (state, bot)
    }

    #[test]
    fn dropoff_rejected_for_pending_order() {
        let (state, bot) = base_state(OrderStatus::Pending);
        let (validated, invalid) = validate_action(
            Action::DropOff {
                bot_id: bot.id.clone(),
                order_id: "o1".to_owned(),
            },
            &state,
            &bot,
        );
        assert!(invalid);
        assert!(matches!(validated, Action::Wait { .. }));
    }

    #[test]
    fn dropoff_allowed_for_active_order_with_item() {
        let (state, bot) = base_state(OrderStatus::InProgress);
        let (validated, invalid) = validate_action(
            Action::DropOff {
                bot_id: bot.id.clone(),
                order_id: "o1".to_owned(),
            },
            &state,
            &bot,
        );
        assert!(!invalid);
        assert!(matches!(validated, Action::DropOff { .. }));
    }
}
