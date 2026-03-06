use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    extract::{Path as AxPath, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tower_http::services::{ServeDir, ServeFile};
use uuid::Uuid;

use grocery_agents::{
    config::PolicyMode,
    policy::Policy,
    replay::{list_runs, load_run, ReplayRun, ReplayRunMeta},
    sim::{
        eval::{default_config, run_batch_eval, EvalRequest},
        scenario::sim_from_replay_fork,
        state::{SimSnapshot, SimState},
        step::{step_many, step_once},
    },
};

#[derive(Debug, Parser)]
#[command(name = "lab")]
struct Cli {
    #[arg(long, default_value = "logs")]
    logs_dir: PathBuf,
    #[arg(long, default_value_t = 8085)]
    port: u16,
    #[arg(long, default_value_t = 900)]
    session_ttl: u64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Clone)]
struct AppState {
    logs_dir: PathBuf,
    base_seed: u64,
    session_ttl: Duration,
    runs_cache: Arc<Mutex<HashMap<String, ReplayRun>>>,
    sessions: Arc<Mutex<HashMap<Uuid, SimSession>>>,
}

struct SimSession {
    id: Uuid,
    run_id: Option<String>,
    fork_tick: Option<u64>,
    created_at: Instant,
    sim: SimState,
}

#[derive(Debug, Deserialize)]
struct ForkBody {
    run_id: String,
    tick: u64,
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct StepBody {
    actions: Option<Vec<grocery_agents::model::Action>>,
    steps: Option<u32>,
    policy_mode: Option<PolicyModeArg>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PolicyModeArg {
    Auto,
    Easy,
    Medium,
    Hard,
    Expert,
}

impl From<PolicyModeArg> for PolicyMode {
    fn from(value: PolicyModeArg) -> Self {
        match value {
            PolicyModeArg::Auto => PolicyMode::Auto,
            PolicyModeArg::Easy => PolicyMode::Easy,
            PolicyModeArg::Medium => PolicyMode::Medium,
            PolicyModeArg::Hard => PolicyMode::Hard,
            PolicyModeArg::Expert => PolicyMode::Expert,
        }
    }
}

#[derive(Debug, Serialize)]
struct RunsResponse {
    runs: Vec<ReplayRunMeta>,
}

#[derive(Debug, Serialize)]
struct RunMetaResponse {
    run_id: String,
    file_name: String,
    mode: Option<String>,
    ticks: usize,
    first_tick: Option<u64>,
    last_tick: Option<u64>,
    final_score: i64,
}

#[derive(Debug, Serialize)]
struct SessionResponse {
    session_id: Uuid,
    snapshot: SimSnapshot,
}

#[derive(Debug, Deserialize)]
struct TickQuery {
    tick: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct EvalBatchRequest {
    episodes: Option<usize>,
    policy: Option<PolicyModeArg>,
    difficulty: Option<String>,
    seed: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let state = AppState {
        logs_dir: cli.logs_dir.clone(),
        base_seed: cli.seed,
        session_ttl: Duration::from_secs(cli.session_ttl.max(30)),
        runs_cache: Arc::new(Mutex::new(HashMap::new())),
        sessions: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = build_app(state);
    let addr = format!("127.0.0.1:{}", cli.port);
    println!("planner-lab listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/api/runs", get(api_runs))
        .route("/api/run/:run_id/meta", get(api_run_meta))
        .route("/api/run/:run_id/tick/:tick", get(api_run_tick))
        .route("/api/sim/fork", post(api_sim_fork))
        .route("/api/sim/:session_id/step", post(api_sim_step))
        .route("/api/sim/:session_id/state", get(api_sim_state))
        .route("/api/sim/:session_id/metrics", get(api_sim_metrics))
        .route("/api/eval/batch", post(api_eval_batch))
        .route("/health", get(|| async { "ok" }))
        .nest_service(
            "/",
            ServeDir::new("lab_ui").not_found_service(ServeFile::new("lab_ui/index.html")),
        )
        .with_state(state)
}

async fn api_runs(State(state): State<AppState>) -> impl IntoResponse {
    match list_runs(&state.logs_dir) {
        Ok(runs) => Json(RunsResponse { runs }).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

async fn api_run_meta(
    State(state): State<AppState>,
    AxPath(run_id): AxPath<String>,
) -> impl IntoResponse {
    match fetch_run(&state, &run_id) {
        Ok(run) => {
            let final_score = run.frames.last().map(|f| f.game_state.score).unwrap_or(0);
            Json(RunMetaResponse {
                run_id: run.run_id.clone(),
                file_name: run
                    .path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_owned(),
                mode: run.mode.clone(),
                ticks: run.frames.len(),
                first_tick: run.frames.first().map(|f| f.tick),
                last_tick: run.frames.last().map(|f| f.tick),
                final_score,
            })
            .into_response()
        }
        Err(err) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": err})),
        )
            .into_response(),
    }
}

async fn api_run_tick(
    State(state): State<AppState>,
    AxPath((run_id, tick)): AxPath<(String, u64)>,
) -> impl IntoResponse {
    match fetch_run(&state, &run_id) {
        Ok(run) => {
            if let Some(frame) = run.frames.iter().find(|f| f.tick == tick) {
                Json(frame).into_response()
            } else {
                (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({"error":"tick not found"})),
                )
                    .into_response()
            }
        }
        Err(err) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": err})),
        )
            .into_response(),
    }
}

async fn api_sim_fork(
    State(state): State<AppState>,
    Json(body): Json<ForkBody>,
) -> impl IntoResponse {
    let run = match fetch_run(&state, &body.run_id) {
        Ok(run) => run,
        Err(err) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": err})),
            )
                .into_response();
        }
    };

    let seed = body.seed.unwrap_or(state.base_seed);
    let Some(sim) = sim_from_replay_fork(&run, body.tick, seed) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"fork tick not available"})),
        )
            .into_response();
    };

    let id = Uuid::new_v4();
    let session = SimSession {
        id,
        run_id: Some(run.run_id.clone()),
        fork_tick: Some(body.tick),
        created_at: Instant::now(),
        sim,
    };
    if let Ok(mut sessions) = state.sessions.lock() {
        sessions.insert(id, session);
    }

    let snapshot = if let Ok(sessions) = state.sessions.lock() {
        sessions
            .get(&id)
            .map(|s| s.sim.snapshot())
            .unwrap_or_else(|| unreachable!("session created"))
    } else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error":"failed to acquire session lock"})),
        )
            .into_response();
    };

    Json(SessionResponse {
        session_id: id,
        snapshot,
    })
    .into_response()
}

async fn api_sim_step(
    State(state): State<AppState>,
    AxPath(session_id): AxPath<Uuid>,
    Json(body): Json<StepBody>,
) -> impl IntoResponse {
    let mut sessions = match state.sessions.lock() {
        Ok(guard) => guard,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error":"failed to acquire session lock"})),
            )
                .into_response();
        }
    };
    expire_sessions(&mut sessions, state.session_ttl);

    let Some(session) = sessions.get_mut(&session_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"session not found"})),
        )
            .into_response();
    };

    let steps = body.steps.unwrap_or(1).clamp(1, 300);
    let deltas = if let Some(actions) = body.actions {
        let mut out = Vec::with_capacity(steps as usize);
        for _ in 0..steps {
            out.push(step_once(&mut session.sim, &actions));
        }
        out
    } else if let Some(mode) = body.policy_mode {
        let mut policy = Policy::new(Arc::new(default_config(mode.into())));
        step_many(
            &mut session.sim,
            |sim| policy.decide_round(&sim.game_state, Duration::from_millis(20)),
            steps,
        )
    } else {
        step_many(&mut session.sim, |_| Vec::new(), steps)
    };

    Json(serde_json::json!({
        "session_id": session_id,
        "deltas": deltas,
        "snapshot": session.sim.snapshot()
    }))
    .into_response()
}

async fn api_sim_state(
    State(state): State<AppState>,
    AxPath(session_id): AxPath<Uuid>,
    Query(query): Query<TickQuery>,
) -> impl IntoResponse {
    let sessions = match state.sessions.lock() {
        Ok(guard) => guard,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error":"failed to acquire session lock"})),
            )
                .into_response();
        }
    };
    let Some(session) = sessions.get(&session_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"session not found"})),
        )
            .into_response();
    };

    if let Some(tick) = query.tick {
        if tick != session.sim.game_state.tick {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error":"random-access tick unavailable in mutable session"})),
            )
                .into_response();
        }
    }

    Json(serde_json::json!({
        "session_id": session.id,
        "run_id": session.run_id,
        "fork_tick": session.fork_tick,
        "snapshot": session.sim.snapshot()
    }))
    .into_response()
}

async fn api_sim_metrics(
    State(state): State<AppState>,
    AxPath(session_id): AxPath<Uuid>,
) -> impl IntoResponse {
    let sessions = match state.sessions.lock() {
        Ok(guard) => guard,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error":"failed to acquire session lock"})),
            )
                .into_response();
        }
    };
    let Some(session) = sessions.get(&session_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"session not found"})),
        )
            .into_response();
    };

    Json(serde_json::json!({
        "session_id": session.id,
        "metrics": session.sim.metrics
    }))
    .into_response()
}

async fn api_eval_batch(
    State(state): State<AppState>,
    Json(req): Json<EvalBatchRequest>,
) -> impl IntoResponse {
    let eval_req = EvalRequest {
        logs_dir: &state.logs_dir,
        episodes: req.episodes.unwrap_or(10).clamp(1, 500),
        policy: req.policy.unwrap_or(PolicyModeArg::Auto).into(),
        difficulty: req.difficulty.as_deref(),
        seed: req.seed.unwrap_or(state.base_seed),
    };

    match run_batch_eval(eval_req) {
        Ok(report) => Json(report).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

fn fetch_run(state: &AppState, run_id: &str) -> Result<ReplayRun, String> {
    if let Ok(cache) = state.runs_cache.lock() {
        if let Some(run) = cache.get(run_id) {
            return Ok(run.clone());
        }
    }

    let files = std::fs::read_dir(&state.logs_dir).map_err(|e| e.to_string())?;
    for entry in files.flatten() {
        let path = entry.path();
        if !is_run_file(&path) {
            continue;
        }
        let run = load_run(&path).map_err(|e| e.to_string())?;
        if let Ok(mut cache) = state.runs_cache.lock() {
            cache.insert(run.run_id.clone(), run.clone());
        }
        if run.run_id == run_id {
            return Ok(run);
        }
    }
    Err(format!("run not found: {run_id}"))
}

fn is_run_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|s| s.to_str())
        .map(|name| name.starts_with("run-") && name.ends_with(".jsonl"))
        .unwrap_or(false)
}

fn expire_sessions(sessions: &mut HashMap<Uuid, SimSession>, ttl: Duration) {
    let now = Instant::now();
    sessions.retain(|_, session| now.duration_since(session.created_at) <= ttl);
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::{build_app, AppState};

    fn unique_temp_dir() -> std::path::PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("lab-tests-{suffix}"));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn write_run(logs_dir: &std::path::Path) {
        let content = vec![
            serde_json::json!({"event":"log_opened","run_id":"run-test"}).to_string(),
            serde_json::json!({"event":"game_mode","data":{"mode":"easy"}}).to_string(),
            serde_json::json!({
                "event":"tick",
                "data":{
                    "tick":0,
                    "game_state":{"tick":0,"score":0,"active_order_index":0,"grid":{"width":2,"height":2,"walls":[],"drop_off_tiles":[[0,0]]},"bots":[],"items":[],"orders":[]},
                    "actions":[],
                    "team_summary":{},
                    "tick_outcome":{"delta_score":0}
                }
            }).to_string(),
        ]
        .join("\n");
        fs::write(logs_dir.join("run-test.jsonl"), content).expect("write log");
    }

    fn app_for(logs_dir: std::path::PathBuf) -> axum::Router {
        let state = AppState {
            logs_dir,
            base_seed: 7,
            session_ttl: std::time::Duration::from_secs(60),
            runs_cache: std::sync::Arc::new(
                std::sync::Mutex::new(std::collections::HashMap::new()),
            ),
            sessions: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        };
        build_app(state)
    }

    #[tokio::test]
    async fn api_runs_lists_logs() {
        let dir = unique_temp_dir();
        write_run(&dir);
        let app = app_for(dir.clone());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/runs")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);

        let body = response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(json["runs"][0]["run_id"], "run-test");

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn api_eval_batch_returns_report() {
        let dir = unique_temp_dir();
        write_run(&dir);
        let app = app_for(dir.clone());

        let request = Request::builder()
            .method("POST")
            .uri("/api/eval/batch")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"episodes":1,"policy":"auto"}"#))
            .expect("request");
        let response = app.oneshot(request).await.expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(json["summary"]["episodes"], 1);

        let _ = fs::remove_dir_all(dir);
    }
}
