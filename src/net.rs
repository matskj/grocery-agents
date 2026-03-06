use std::{
    collections::HashMap,
    fs::{create_dir_all, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::{mpsc, Arc, OnceLock},
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, Message},
    MaybeTlsStream, WebSocketStream,
};
use tracing::{info, warn};

use crate::{
    config::{Config, PlannerBudgetMode},
    difficulty::detect_mode_label,
    dist::DistanceMap,
    metrics::MetricsTracker,
    model::{
        to_wire_action_envelope, Action, ActionEnvelope, BotState, GameOver, GameState,
        OrderStatus, RuntimeContext, WireServerMessage,
    },
    policy::Policy,
    world::World,
};

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;
const LOG_SCHEMA_VERSION: &str = "1.3.0";

#[derive(Debug, Clone, Default)]
pub struct ReplaySummary {
    pub ticks: u64,
    pub total_score: i64,
    pub orders_completed: u64,
    pub items_delivered: u64,
    pub avg_planning_time_ms: f64,
    pub p95_planning_time_ms: u64,
    pub wait_actions: u64,
    pub invalid_actions_corrected: u64,
    pub estimated_rounds_in_120s: u64,
    pub avg_order_rounds_to_complete: f64,
}

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

#[derive(Debug, Clone)]
struct CachedPlanMeta {
    tick: u64,
    active_missing_total: usize,
    dropoff_ready_bot_count: u16,
    bot_pos_by_id: HashMap<String, (i32, i32)>,
}

#[derive(Debug, Clone, Copy, Default)]
struct BudgetPressure {
    blocked_bots_prev: u64,
    stuck_bots_prev: u64,
}

#[derive(Debug, Clone, Copy)]
struct BudgetDecision {
    soft_ms: u64,
    hard_ms: u64,
    fair_share_ms: u64,
    projected_rounds: u64,
}

#[derive(Debug)]
struct PlanRequest {
    seq: u64,
    tick: u64,
    soft_budget_ms: u64,
    state: GameState,
}

#[derive(Debug)]
struct PlanResponse {
    seq: u64,
    tick: u64,
    result: PlanRoundResult,
    worker_plan_ms: u64,
}

#[derive(Debug)]
enum PlannerWorkerMsg {
    Plan(PlanRequest),
    Shutdown,
}

struct PlannerManager {
    next_seq: u64,
    request_tx: mpsc::Sender<PlannerWorkerMsg>,
    response_rx: mpsc::Receiver<PlanResponse>,
    pending_by_seq: HashMap<u64, PlanResponse>,
    submitted_meta_by_seq: HashMap<u64, CachedPlanMeta>,
    last_result: Option<PlanRoundResult>,
    last_meta: Option<CachedPlanMeta>,
    worker: Option<JoinHandle<()>>,
}

impl PlannerManager {
    fn new(policy: Policy) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<PlannerWorkerMsg>();
        let (response_tx, response_rx) = mpsc::channel::<PlanResponse>();
        let worker = thread::Builder::new()
            .name("planner-worker".to_owned())
            .spawn(move || {
                let mut worker_policy = policy;
                while let Ok(msg) = request_rx.recv() {
                    match msg {
                        PlannerWorkerMsg::Plan(req) => {
                            let started = Instant::now();
                            let result = plan_round_actions(
                                &mut worker_policy,
                                &req.state,
                                req.soft_budget_ms,
                            );
                            let worker_plan_ms = started.elapsed().as_millis() as u64;
                            let response = PlanResponse {
                                seq: req.seq,
                                tick: req.tick,
                                result,
                                worker_plan_ms,
                            };
                            if response_tx.send(response).is_err() {
                                break;
                            }
                        }
                        PlannerWorkerMsg::Shutdown => break,
                    }
                }
            })
            .ok();
        Self {
            next_seq: 1,
            request_tx,
            response_rx,
            pending_by_seq: HashMap::new(),
            submitted_meta_by_seq: HashMap::new(),
            last_result: None,
            last_meta: None,
            worker,
        }
    }

    fn submit(&mut self, state: &GameState, soft_budget_ms: u64) -> Option<u64> {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);
        let request = PlanRequest {
            seq,
            tick: state.tick,
            soft_budget_ms,
            state: state.clone(),
        };
        self.submitted_meta_by_seq.insert(seq, cache_meta(state));
        if self
            .request_tx
            .send(PlannerWorkerMsg::Plan(request))
            .is_err()
        {
            self.submitted_meta_by_seq.remove(&seq);
            return None;
        }
        Some(seq)
    }

    fn poll(&mut self) {
        while let Ok(resp) = self.response_rx.try_recv() {
            self.pending_by_seq.insert(resp.seq, resp);
        }
    }

    fn take(&mut self, seq: u64) -> Option<PlanResponse> {
        self.poll();
        let resp = self.pending_by_seq.remove(&seq)?;
        self.last_result = Some(resp.result.clone());
        self.last_meta = self.submitted_meta_by_seq.remove(&seq);
        Some(resp)
    }

    fn cached_for_state(
        &self,
        state: &GameState,
        max_age_ticks: u64,
        require_progress: bool,
    ) -> Option<PlanRoundResult> {
        let last = self.last_result.as_ref()?;
        let meta = self.last_meta.as_ref()?;
        if last.actions.is_empty() || state.bots.is_empty() {
            return None;
        }
        if state.tick < meta.tick {
            return None;
        }
        let age_ticks = state.tick.saturating_sub(meta.tick);
        if age_ticks > max_age_ticks {
            return None;
        }
        if require_progress {
            if has_immediate_conversion_opportunity(state) {
                return None;
            }
            if active_missing_total(state) > meta.active_missing_total {
                return None;
            }
            if dropoff_ready_bot_count(state) != meta.dropoff_ready_bot_count {
                return None;
            }
            let mut total_delta = 0u64;
            let mut count = 0u64;
            for bot in &state.bots {
                let Some((px, py)) = meta.bot_pos_by_id.get(&bot.id).copied() else {
                    return None;
                };
                let delta = (bot.x - px).unsigned_abs() + (bot.y - py).unsigned_abs();
                if delta > 2 {
                    return None;
                }
                total_delta = total_delta.saturating_add(u64::from(delta));
                count = count.saturating_add(1);
            }
            if count > 0 && total_delta > count.saturating_mul(2) {
                return None;
            }
        }
        let mut action_by_bot = HashMap::<&str, Action>::new();
        for action in &last.actions {
            action_by_bot.insert(action.bot_id(), action.clone());
        }
        let mut out = Vec::with_capacity(state.bots.len());
        for bot in &state.bots {
            let candidate = action_by_bot
                .get(bot.id.as_str())
                .cloned()
                .unwrap_or_else(|| Action::wait(bot.id.clone()));
            let (validated, invalid) = validate_action(candidate, state, bot);
            if invalid {
                return None;
            }
            out.push(validated);
        }
        let mut telemetry = last.team_telemetry.clone();
        if let Some(obj) = telemetry.as_object_mut() {
            obj.insert(
                "fallback_level".to_owned(),
                serde_json::Value::String("cached".to_owned()),
            );
            obj.insert(
                "cached_age_ticks".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(age_ticks)),
            );
        }
        Some(PlanRoundResult {
            actions: out,
            invalid_action_count: 0,
            team_telemetry: telemetry,
            action_validated_by_bot: state
                .bots
                .iter()
                .map(|bot| (bot.id.clone(), true))
                .collect(),
            plan_ms: 0,
            assign_ms: last.assign_ms,
        })
    }
}

impl Drop for PlannerManager {
    fn drop(&mut self) {
        let _ = self.request_tx.send(PlannerWorkerMsg::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
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
    policy: Policy,
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
    let mut planner_manager = PlannerManager::new(policy);

    run_logger.log(
        "session_start",
        serde_json::json!({
            "schema_version": LOG_SCHEMA_VERSION,
            "run_id": run_logger.run_id(),
            "ws_url": base_url,
            "token_preview": token_preview(&ctx.token),
            "planner_budget_mode": format!("{:?}", config.planner_budget_mode).to_lowercase(),
            "planner_soft_budget_ms": config.planner_soft_budget_ms,
            "planner_soft_budget_min_ms": config.planner_soft_budget_min_ms,
            "planner_soft_budget_max_ms": config.planner_soft_budget_max_ms,
            "planner_hard_budget_ms": config.planner_hard_budget_ms,
            "planner_deadline_slack_ms": config.planner_deadline_slack_ms,
            "tick_soft_budget_ms": config.tick_soft_budget_ms,
            "tick_hard_budget_ms": config.tick_hard_budget_ms,
            "tick_greedy_fallback_ms": config.tick_greedy_fallback_ms,
            "cache_reuse_max_age_ticks": config.cache_reuse_max_age_ticks,
            "cache_require_progress": config.cache_require_progress,
            "policy_mode": format!("{:?}", config.policy_mode).to_lowercase(),
            "mode": ctx.session.difficulty.as_deref().unwrap_or("unknown"),
            "map_id": ctx.session.map_id,
            "difficulty": ctx.session.difficulty,
            "team_id": ctx.session.team_id,
            "map_seed": ctx.session.map_seed,
            "build_version": build_version(),
            "artifact_path": "",
            "artifact_loaded": false,
            "artifact_schema_version": "",
            "artifact_mode_count": 0,
        }),
    );

    info!("connected websocket, entering receive loop");
    let mut mode_logged = false;
    let mut budget_pressure = BudgetPressure::default();

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
                        budget_pressure = handle_game_state(
                            &mut socket,
                            &mut planner_manager,
                            &mut run_logger,
                            &mut mode_logged,
                            config.as_ref(),
                            budget_pressure,
                            game_state,
                        )
                        .await?;
                    }
                    ServerMessage::LegacyGameStateEnvelope { game_state }
                    | ServerMessage::LegacyGameState(game_state) => {
                        budget_pressure = handle_game_state(
                            &mut socket,
                            &mut planner_manager,
                            &mut run_logger,
                            &mut mode_logged,
                            config.as_ref(),
                            budget_pressure,
                            game_state,
                        )
                        .await?;
                    }
                    ServerMessage::Wire(WireServerMessage::GameOver(wire_game_over)) => {
                        let game_over = GameOver::from_wire(wire_game_over);
                        let reason = game_over.reason.clone();
                        let metrics_summary = run_logger.metrics_summary_json();
                        let metrics_summary_for_log = metrics_summary.clone();
                        let p95_ms = metrics_summary
                            .get("p95_planning_time_ms")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        let avg_order_rounds = metrics_summary
                            .get("order_metrics")
                            .and_then(serde_json::Value::as_array)
                            .map(|rows| {
                                if rows.is_empty() {
                                    0.0
                                } else {
                                    rows.iter()
                                        .filter_map(|r| r.get("rounds_to_complete"))
                                        .filter_map(serde_json::Value::as_u64)
                                        .map(|v| v as f64)
                                        .sum::<f64>()
                                        / rows.len() as f64
                                }
                            })
                            .unwrap_or(0.0);
                        run_logger.log(
                            "game_over",
                            serde_json::json!({
                                "mode": run_logger.last_mode.as_deref().unwrap_or("unknown"),
                                "final_score": game_over.final_score,
                                "reason": reason,
                                "episode_counters": run_logger.episode_counters_json(),
                                "game_metrics": metrics_summary_for_log,
                            }),
                        );
                        if p95_ms > 100 {
                            warn!(
                                p95_planning_time_ms = p95_ms,
                                "planning p95 exceeded 100ms target"
                            );
                        }
                        println!(
                            "game_end score={} orders={} items={} avg_order_rounds={:.2} avg_plan_ms={:.2} p95_plan_ms={} waits={} invalid_fixed={}",
                            metrics_summary.get("total_score").and_then(serde_json::Value::as_i64).unwrap_or(game_over.final_score),
                            metrics_summary.get("orders_completed").and_then(serde_json::Value::as_u64).unwrap_or(0),
                            metrics_summary.get("items_delivered").and_then(serde_json::Value::as_u64).unwrap_or(0),
                            avg_order_rounds,
                            metrics_summary.get("avg_planning_time_ms").and_then(serde_json::Value::as_f64).unwrap_or(0.0),
                            p95_ms,
                            metrics_summary.get("wait_actions").and_then(serde_json::Value::as_u64).unwrap_or(0),
                            metrics_summary.get("corrected_invalid_actions").and_then(serde_json::Value::as_u64).unwrap_or(0),
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
                        let metrics_summary = run_logger.metrics_summary_json();
                        let metrics_summary_for_log = metrics_summary.clone();
                        let p95_ms = metrics_summary
                            .get("p95_planning_time_ms")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        let avg_order_rounds = metrics_summary
                            .get("order_metrics")
                            .and_then(serde_json::Value::as_array)
                            .map(|rows| {
                                if rows.is_empty() {
                                    0.0
                                } else {
                                    rows.iter()
                                        .filter_map(|r| r.get("rounds_to_complete"))
                                        .filter_map(serde_json::Value::as_u64)
                                        .map(|v| v as f64)
                                        .sum::<f64>()
                                        / rows.len() as f64
                                }
                            })
                            .unwrap_or(0.0);
                        run_logger.log(
                            "game_over",
                            serde_json::json!({
                                "mode": run_logger.last_mode.as_deref().unwrap_or("unknown"),
                                "final_score": game_over.final_score,
                                "reason": reason,
                                "episode_counters": run_logger.episode_counters_json(),
                                "game_metrics": metrics_summary_for_log,
                            }),
                        );
                        if p95_ms > 100 {
                            warn!(
                                p95_planning_time_ms = p95_ms,
                                "planning p95 exceeded 100ms target"
                            );
                        }
                        println!(
                            "game_end score={} orders={} items={} avg_order_rounds={:.2} avg_plan_ms={:.2} p95_plan_ms={} waits={} invalid_fixed={}",
                            metrics_summary.get("total_score").and_then(serde_json::Value::as_i64).unwrap_or(game_over.final_score),
                            metrics_summary.get("orders_completed").and_then(serde_json::Value::as_u64).unwrap_or(0),
                            metrics_summary.get("items_delivered").and_then(serde_json::Value::as_u64).unwrap_or(0),
                            avg_order_rounds,
                            metrics_summary.get("avg_planning_time_ms").and_then(serde_json::Value::as_f64).unwrap_or(0.0),
                            p95_ms,
                            metrics_summary.get("wait_actions").and_then(serde_json::Value::as_u64).unwrap_or(0),
                            metrics_summary.get("corrected_invalid_actions").and_then(serde_json::Value::as_u64).unwrap_or(0),
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

pub fn run_replay_file(
    replay_path: &Path,
    mut policy: Policy,
    config: Arc<Config>,
) -> Result<ReplaySummary, Box<dyn std::error::Error>> {
    let file = File::open(replay_path)?;
    let reader = BufReader::new(file);
    let mut metrics = MetricsTracker::default();
    let mut prev_score = None::<i64>;
    let mut prev_active_index = None::<i64>;
    let mut ticks = 0u64;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(record) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let Some(state) = replay_state_from_record(&record) else {
            continue;
        };
        ticks = ticks.saturating_add(1);

        let started = Instant::now();
        let mut planned = plan_round_actions(&mut policy, &state, config.tick_soft_budget_ms);
        if started.elapsed() > Duration::from_millis(config.tick_hard_budget_ms) {
            planned = fallback_safe_actions(&state, config.tick_soft_budget_ms);
            if let Some(obj) = planned.team_telemetry.as_object_mut() {
                obj.insert(
                    "fallback_level".to_owned(),
                    serde_json::Value::String("wait".to_owned()),
                );
            }
        }

        let delta_score = prev_score.map(|v| state.score - v).unwrap_or(0);
        prev_score = Some(state.score);
        let order_completed_delta = prev_active_index
            .map(|v| (state.active_order_index - v).max(0))
            .unwrap_or(0);
        prev_active_index = Some(state.active_order_index);
        let items_delivered_delta = (delta_score - 5 * order_completed_delta).max(0);
        let tick_outcome = serde_json::json!({
            "delta_score": delta_score,
            "items_delivered_delta": items_delivered_delta,
            "order_completed_delta": order_completed_delta,
            "invalid_action_count": planned.invalid_action_count,
        });
        metrics.on_tick(
            &state,
            &planned.actions,
            &planned.team_telemetry,
            &tick_outcome,
            planned.plan_ms,
        );
    }

    let summary = metrics.summary();
    let estimated_rounds = if summary.avg_planning_time_ms > 0.0 {
        ((120_000.0 / summary.avg_planning_time_ms).floor() as u64).min(300)
    } else {
        0
    };
    let avg_order_rounds_to_complete = if summary.order_metrics.is_empty() {
        0.0
    } else {
        summary
            .order_metrics
            .iter()
            .map(|m| m.rounds_to_complete as f64)
            .sum::<f64>()
            / summary.order_metrics.len() as f64
    };
    Ok(ReplaySummary {
        ticks,
        total_score: summary.total_score,
        orders_completed: summary.orders_completed,
        items_delivered: summary.items_delivered,
        avg_planning_time_ms: summary.avg_planning_time_ms,
        p95_planning_time_ms: summary.p95_planning_time_ms,
        wait_actions: summary.wait_actions,
        invalid_actions_corrected: summary.corrected_invalid_actions,
        estimated_rounds_in_120s: estimated_rounds,
        avg_order_rounds_to_complete,
    })
}

fn replay_state_from_record(record: &serde_json::Value) -> Option<GameState> {
    let event = record.get("event").and_then(serde_json::Value::as_str)?;
    match event {
        "tick_replay" => record
            .get("game_state")
            .cloned()
            .and_then(|v| serde_json::from_value(v).ok()),
        "tick" => record
            .get("data")
            .and_then(|d| d.get("game_state"))
            .cloned()
            .and_then(|v| serde_json::from_value(v).ok()),
        _ => None,
    }
}

async fn handle_game_state(
    socket: &mut WsStream,
    planner: &mut PlannerManager,
    run_logger: &mut RunLogger,
    mode_logged: &mut bool,
    config: &Config,
    budget_pressure: BudgetPressure,
    game_state: GameState,
) -> Result<BudgetPressure, Box<dyn std::error::Error>> {
    if !*mode_logged {
        run_logger.log("game_mode", game_mode_payload(&game_state));
        *mode_logged = true;
    }

    run_logger.mark_episode_started();
    let elapsed_episode_ms = run_logger.elapsed_episode_ms();
    let budget = compute_tick_budget(
        config,
        budget_pressure,
        game_state.bots.len(),
        game_state.tick,
        elapsed_episode_ms,
    );
    let soft_budget_ms = budget.soft_ms;
    let hard_budget = Duration::from_millis(budget.hard_ms);
    let plan_started = Instant::now();
    let allow_cached_plan = !has_immediate_conversion_opportunity(&game_state);
    let mut planned = if allow_cached_plan {
        planner.cached_for_state(
            &game_state,
            config.cache_reuse_max_age_ticks,
            config.cache_require_progress,
        )
    } else {
        None
    };
    let mut worker_plan_ms = 0u64;
    if planned.is_none() {
        if let Some(req_seq) = planner.submit(&game_state, soft_budget_ms) {
            let soft_deadline = plan_started
                + Duration::from_millis(soft_budget_ms.max(1).min(budget.hard_ms.max(1)));
            while Instant::now() < soft_deadline {
                if let Some(resp) = planner.take(req_seq) {
                    if resp.tick == game_state.tick {
                        worker_plan_ms = resp.worker_plan_ms;
                        planned = Some(resp.result);
                    }
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    }

    let mut planned = planned.unwrap_or_else(|| {
        warn!(
            tick = game_state.tick,
            "planning timeout, falling back to greedy-safe actions"
        );
        if bot_debug_enabled() {
            warn!(
                tick = game_state.tick,
                budget_ms = hard_budget.as_millis(),
                soft_budget_ms,
                bot_count = game_state.bots.len(),
                "time-budget event: fallback greedy-safe envelope emitted"
            );
        }
        fallback_safe_actions(&game_state, config.tick_greedy_fallback_ms)
    });
    if let Some(obj) = planned.team_telemetry.as_object_mut() {
        obj.insert(
            "planner_hard_budget_ms".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(budget.hard_ms)),
        );
        obj.insert(
            "budget_fair_share_ms".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(budget.fair_share_ms)),
        );
        obj.insert(
            "projected_rounds_in_120s".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(budget.projected_rounds)),
        );
        obj.insert(
            "worker_plan_ms".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(worker_plan_ms)),
        );
    }

    let envelope = ActionEnvelope {
        actions: planned.actions.clone(),
    };
    let analytics = run_logger.observe_tick(
        &game_state,
        &envelope.actions,
        planned.invalid_action_count,
        &planned.team_telemetry,
        planned.plan_ms,
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
    if config.debug {
        let frame = render_debug_overlay(&game_state, &planned.team_telemetry);
        run_logger.log(
            "debug_overlay",
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
    Ok(budget_pressure_from_team_telemetry(&planned.team_telemetry))
}

fn has_immediate_conversion_opportunity(state: &GameState) -> bool {
    let active_kinds = state
        .orders
        .iter()
        .filter(|o| matches!(o.status, OrderStatus::InProgress))
        .map(|o| o.item_id.as_str())
        .collect::<std::collections::HashSet<_>>();
    if active_kinds.is_empty() {
        return false;
    }
    state.bots.iter().any(|bot| {
        let drop_dist = state
            .grid
            .drop_off_tiles
            .iter()
            .map(|tile| (tile[0] - bot.x).unsigned_abs() + (tile[1] - bot.y).unsigned_abs())
            .min()
            .unwrap_or(u32::MAX);
        let near_drop = drop_dist <= 1;
        let carries_active = bot
            .carrying
            .iter()
            .any(|kind| active_kinds.contains(kind.as_str()));
        near_drop && carries_active
    })
}

fn active_missing_total(state: &GameState) -> usize {
    state
        .orders
        .iter()
        .filter(|order| matches!(order.status, OrderStatus::InProgress))
        .count()
}

fn cache_meta(state: &GameState) -> CachedPlanMeta {
    let mut bot_pos_by_id = HashMap::with_capacity(state.bots.len());
    for bot in &state.bots {
        bot_pos_by_id.insert(bot.id.clone(), (bot.x, bot.y));
    }
    CachedPlanMeta {
        tick: state.tick,
        active_missing_total: active_missing_total(state),
        dropoff_ready_bot_count: dropoff_ready_bot_count(state),
        bot_pos_by_id,
    }
}

fn dropoff_ready_bot_count(state: &GameState) -> u16 {
    let active_kinds = state
        .orders
        .iter()
        .filter(|o| matches!(o.status, OrderStatus::InProgress))
        .map(|o| o.item_id.as_str())
        .collect::<std::collections::HashSet<_>>();
    if active_kinds.is_empty() {
        return 0;
    }
    state
        .bots
        .iter()
        .filter(|bot| {
            let on_drop = state
                .grid
                .drop_off_tiles
                .iter()
                .any(|tile| tile[0] == bot.x && tile[1] == bot.y);
            let carries_active = bot
                .carrying
                .iter()
                .any(|kind| active_kinds.contains(kind.as_str()));
            on_drop && carries_active
        })
        .count() as u16
}

fn compute_tick_budget(
    config: &Config,
    pressure: BudgetPressure,
    bot_count: usize,
    current_round: u64,
    elapsed_episode_ms: u64,
) -> BudgetDecision {
    const TOTAL_WALL_MS: u64 = 120_000;
    const MAX_ROUNDS: u64 = 300;
    const SAFETY_MARGIN_MS: u64 = 1_500;

    let remaining_rounds = MAX_ROUNDS.saturating_sub(current_round).max(1);
    let remaining_wall_ms = TOTAL_WALL_MS
        .saturating_sub(elapsed_episode_ms)
        .saturating_sub(SAFETY_MARGIN_MS)
        .max(1_000);
    let fair_share_ms = (remaining_wall_ms / remaining_rounds).max(10);

    let hard_cap = config.tick_hard_budget_ms.clamp(20, 500);
    let mut hard_ms = fair_share_ms.min(hard_cap).max(20);
    let soft_default = config.tick_soft_budget_ms.clamp(10, hard_ms);
    let mut soft_ms = ((fair_share_ms as f64) * 0.60).round() as u64;
    soft_ms = soft_ms.min(soft_default).max(10);

    match config.planner_budget_mode {
        PlannerBudgetMode::Fixed => {
            soft_ms = config.tick_soft_budget_ms.clamp(10, hard_ms);
        }
        PlannerBudgetMode::Adaptive => {
            let pressure_bonus = 4u64
                .saturating_mul(pressure.blocked_bots_prev)
                .saturating_add(3u64.saturating_mul(pressure.stuck_bots_prev))
                .saturating_add(2u64.saturating_mul(bot_count as u64));
            let adaptive_cap = (((fair_share_ms as f64) * 0.85).round() as u64).min(hard_ms);
            soft_ms = soft_ms
                .saturating_add(pressure_bonus)
                .clamp(10, adaptive_cap.max(10));
        }
    }
    if soft_ms >= hard_ms {
        hard_ms = soft_ms.saturating_add(5).min(hard_cap);
        soft_ms = soft_ms.min(hard_ms.saturating_sub(1).max(10));
    }

    let projected_rounds = if elapsed_episode_ms > 0 && current_round > 0 {
        let avg = elapsed_episode_ms as f64 / current_round as f64;
        if avg > 0.0 {
            (TOTAL_WALL_MS as f64 / avg).round() as u64
        } else {
            MAX_ROUNDS
        }
    } else {
        MAX_ROUNDS
    };

    BudgetDecision {
        soft_ms,
        hard_ms,
        fair_share_ms,
        projected_rounds,
    }
}

fn budget_pressure_from_team_telemetry(team_telemetry: &serde_json::Value) -> BudgetPressure {
    let blocked = team_telemetry
        .get("blocked_bot_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let stuck = team_telemetry
        .get("stuck_bot_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    BudgetPressure {
        blocked_bots_prev: blocked.min(64),
        stuck_bots_prev: stuck.min(64),
    }
}

struct RunLogger {
    file: Option<File>,
    path: Option<PathBuf>,
    replay_dump_file: Option<File>,
    run_id: String,
    last_score: Option<i64>,
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
    episode_started_at: Option<Instant>,
    metrics: MetricsTracker,
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
                episode_started_at: None,
                metrics: MetricsTracker::default(),
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
                    episode_started_at: None,
                    metrics: MetricsTracker::default(),
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
            episode_started_at: None,
            metrics: MetricsTracker::default(),
        }
    }

    fn mark_episode_started(&mut self) {
        if self.episode_started_at.is_none() {
            self.episode_started_at = Some(Instant::now());
        }
    }

    fn elapsed_episode_ms(&self) -> u64 {
        self.episode_started_at
            .map(|started| started.elapsed().as_millis() as u64)
            .unwrap_or(0)
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
        let planning_time_ms = team_telemetry
            .get("plan_ms")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let record = serde_json::json!({
            "event": "tick_replay",
            "run_id": self.run_id,
            "schema_version": LOG_SCHEMA_VERSION,
            "ts_ms": now_unix_millis(),
            "tick": state.tick,
            "game_state": state,
            "actions": actions,
            "planning_time_ms": planning_time_ms,
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

    fn metrics_summary_json(&self) -> serde_json::Value {
        let summary = self.metrics.summary();
        serde_json::json!({
            "total_score": summary.total_score,
            "orders_completed": summary.orders_completed,
            "items_delivered": summary.items_delivered,
            "avg_path_length_per_item": summary.avg_path_length_per_item,
            "avg_planning_time_ms": summary.avg_planning_time_ms,
            "p95_planning_time_ms": summary.p95_planning_time_ms,
            "corrected_invalid_actions": summary.corrected_invalid_actions,
            "wait_actions": summary.wait_actions,
            "collisions_prevented": summary.collisions_prevented,
            "bots_idle": summary.bots_idle,
            "order_metrics": summary.order_metrics.iter().map(|m| {
                serde_json::json!({
                    "order_index": m.order_index,
                    "rounds_to_complete": m.rounds_to_complete,
                    "items_delivered": m.items_delivered,
                    "items_pre_staged": m.items_pre_staged,
                })
            }).collect::<Vec<_>>(),
        })
    }

    fn observe_tick(
        &mut self,
        state: &GameState,
        actions: &[Action],
        invalid_action_count: u64,
        team_telemetry: &serde_json::Value,
        planning_time_ms: u64,
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

        let order_completed_delta = self
            .last_active_order_index
            .map(|v| (state.active_order_index - v).max(0))
            .unwrap_or_default();
        self.last_active_order_index = Some(state.active_order_index);
        let items_delivered_delta = (delta_score - 5 * order_completed_delta).max(0);

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
        let collisions_prevented_tick = team_telemetry
            .get("local_conflict_count_by_bot")
            .and_then(serde_json::Value::as_object)
            .map(|m| {
                m.values()
                    .filter_map(serde_json::Value::as_u64)
                    .fold(0u64, |acc, v| acc.saturating_add(v))
            })
            .unwrap_or(0);
        team_summary.insert(
            "planning_time_ms".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(planning_time_ms)),
        );
        team_summary.insert(
            "corrected_invalid_actions_tick".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(invalid_action_count)),
        );
        team_summary.insert(
            "collisions_prevented_tick".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(collisions_prevented_tick)),
        );
        team_summary.insert(
            "bots_idle_tick".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(wait_actions)),
        );
        team_summary.insert("episode_counters".to_owned(), self.episode_counters_json());
        if let Some(extra) = team_telemetry.as_object() {
            for (key, value) in extra {
                team_summary.insert(key.clone(), value.clone());
            }
        }

        let analytics = TickAnalytics {
            team_summary: serde_json::Value::Object(team_summary),
            tick_outcome: serde_json::json!({
                "delta_score": delta_score,
                "items_delivered_delta": items_delivered_delta,
                "order_completed_delta": order_completed_delta,
                "invalid_action_count": invalid_action_count,
            }),
        };
        self.metrics.on_tick(
            state,
            actions,
            &analytics.team_summary,
            &analytics.tick_outcome,
            planning_time_ms,
        );
        analytics
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

fn render_debug_overlay(state: &GameState, team_telemetry: &serde_json::Value) -> String {
    let world = World::new(state);
    let map = world.map();
    let width = map.width.max(0) as usize;
    let height = map.height.max(0) as usize;
    if width == 0 || height == 0 {
        return "empty-grid".to_owned();
    }
    let mut cells = vec![vec!['.'; width]; height];
    for idx in 0..(width * height) {
        if map.wall_mask[idx] {
            let x = idx % width;
            let y = idx / width;
            cells[y][x] = '#';
            continue;
        }
        if map.choke_points.get(idx).copied().unwrap_or(false) {
            let x = idx % width;
            let y = idx / width;
            cells[y][x] = '!';
        }
    }
    if let Some(goal_map) = team_telemetry
        .get("goal_cell_by_bot")
        .and_then(serde_json::Value::as_object)
    {
        for cell in goal_map.values().filter_map(serde_json::Value::as_u64) {
            let c = cell as usize;
            if c >= width * height {
                continue;
            }
            let x = c % width;
            let y = c / width;
            if cells[y][x] == '.' {
                cells[y][x] = 'T';
            }
        }
    }
    if let Some(res) = team_telemetry
        .get("reserved_cells_by_t")
        .and_then(serde_json::Value::as_object)
    {
        for cells_t in res.values().filter_map(serde_json::Value::as_array) {
            for cell in cells_t.iter().filter_map(serde_json::Value::as_u64) {
                let c = cell as usize;
                if c >= width * height {
                    continue;
                }
                let x = c % width;
                let y = c / width;
                if matches!(cells[y][x], '.' | 'T') {
                    cells[y][x] = 'R';
                }
            }
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
    let claimed = team_telemetry
        .get("claimed_item_type_counts")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let role_counts = team_telemetry
        .get("strategy_role_counts")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    lines.push(format!("claimed_item_type_counts={claimed}"));
    lines.push(format!("strategy_role_counts={role_counts}"));
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

fn plan_round_actions(
    policy: &mut Policy,
    state: &GameState,
    soft_budget_ms: u64,
) -> PlanRoundResult {
    let plan_started = Instant::now();
    let proposed = policy.decide_round(state, Duration::from_millis(soft_budget_ms));
    let mut team_telemetry = policy.last_team_telemetry();
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
            let (validated, was_invalid) = validate_action(action, state, bot);
            if was_invalid {
                invalid += 1;
            }
            action_validated_by_bot.insert(bot.id.clone(), !was_invalid);
            validated
        })
        .collect::<Vec<_>>();

    let plan_ms = plan_started.elapsed().as_millis() as u64;
    if let Some(obj) = team_telemetry.as_object_mut() {
        obj.insert(
            "plan_ms".to_owned(),
            serde_json::Value::Number(serde_json::Number::from(plan_ms)),
        );
        let fallback_level = if invalid > 0 {
            "greedy_or_guard"
        } else {
            "none"
        };
        obj.insert(
            "fallback_level".to_owned(),
            serde_json::Value::String(fallback_level.to_owned()),
        );
    }

    PlanRoundResult {
        actions,
        invalid_action_count: invalid,
        team_telemetry,
        action_validated_by_bot,
        plan_ms,
        assign_ms,
    }
}

fn fallback_safe_actions(state: &GameState, soft_budget_ms: u64) -> PlanRoundResult {
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

    let world = World::new(state);
    let map = world.map();
    let dist = DistanceMap::shared_for(map);
    let active_kinds = state
        .orders
        .iter()
        .filter(|o| matches!(o.status, OrderStatus::InProgress))
        .map(|o| o.item_id.as_str())
        .collect::<Vec<_>>();
    let mut actions = Vec::with_capacity(state.bots.len());
    let mut greedy_count = 0u64;
    for bot in &state.bots {
        let action = fallback_bot_action(bot, state, map, dist.as_ref(), &active_kinds);
        if !matches!(action, Action::Wait { .. }) {
            greedy_count = greedy_count.saturating_add(1);
        }
        actions.push(action);
    }

    PlanRoundResult {
        actions,
        invalid_action_count: 0,
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
            "blocked_bot_count": 0,
            "stuck_bot_count": 0,
            "planner_soft_budget_ms": soft_budget_ms,
            "fallback_level": "greedy",
            "greedy_fallback_non_wait_actions": greedy_count,
            "assign_ms": 0,
        }),
        action_validated_by_bot: state
            .bots
            .iter()
            .map(|bot| (bot.id.clone(), true))
            .collect(),
        plan_ms: 0,
        assign_ms: 0,
    }
}

fn fallback_bot_action(
    bot: &BotState,
    state: &GameState,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
    active_kinds: &[&str],
) -> Action {
    let Some(cell) = map.idx(bot.x, bot.y) else {
        return Action::wait(bot.id.clone());
    };
    let on_dropoff = map.dropoff_cells.contains(&cell);
    if on_dropoff {
        for kind in &bot.carrying {
            if active_kinds.iter().any(|k| *k == kind) {
                if let Some(order) = state
                    .orders
                    .iter()
                    .find(|o| matches!(o.status, OrderStatus::InProgress) && o.item_id == *kind)
                {
                    return Action::DropOff {
                        bot_id: bot.id.clone(),
                        order_id: order.id.clone(),
                    };
                }
            }
        }
    }

    if bot
        .carrying
        .iter()
        .any(|kind| active_kinds.iter().any(|k| *k == kind))
    {
        if let Some(drop) = map
            .dropoff_cells
            .iter()
            .copied()
            .min_by_key(|&d| dist.dist(cell, d))
        {
            if let Some((dx, dy)) = next_step_towards(cell, drop, map, dist) {
                return Action::Move {
                    bot_id: bot.id.clone(),
                    dx,
                    dy,
                };
            }
        }
    }

    if bot.carrying.len() < bot.capacity {
        for item in &state.items {
            if !active_kinds.iter().any(|kind| *kind == item.kind) {
                continue;
            }
            if (item.x - bot.x).abs() + (item.y - bot.y).abs() == 1 {
                return Action::PickUp {
                    bot_id: bot.id.clone(),
                    item_id: item.id.clone(),
                };
            }
        }
    }

    let mut best_target = None::<(u16, u16)>;
    for item in &state.items {
        if !active_kinds.iter().any(|kind| *kind == item.kind) {
            continue;
        }
        for &stand in map.stand_cells_for_item(&item.id) {
            let d = dist.dist(cell, stand);
            if d == u16::MAX {
                continue;
            }
            match best_target {
                Some((_, bd)) if d >= bd => {}
                _ => best_target = Some((stand, d)),
            }
        }
    }
    if let Some((target, _)) = best_target {
        if let Some((dx, dy)) = next_step_towards(cell, target, map, dist) {
            return Action::Move {
                bot_id: bot.id.clone(),
                dx,
                dy,
            };
        }
    }

    Action::wait(bot.id.clone())
}

fn next_step_towards(
    start: u16,
    target: u16,
    map: &crate::world::MapCache,
    dist: &DistanceMap,
) -> Option<(i32, i32)> {
    let mut best = None::<u16>;
    let mut best_d = u16::MAX;
    for &nb in &map.neighbors[start as usize] {
        let d = dist.dist(nb, target);
        if d < best_d {
            best_d = d;
            best = Some(nb);
        }
    }
    let next = best?;
    let (x0, y0) = map.xy(start);
    let (x1, y1) = map.xy(next);
    Some((x1 - x0, y1 - y0))
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
