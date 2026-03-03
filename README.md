# grocery-agents

A Rust bot client for the Norwegian AI Championship warm-up grocery game.

## Prerequisites

- Rust toolchain (install with rustup): https://rustup.rs/
- `cargo` available on your `PATH`
- Python 3.10+ (for offline training pipeline)

Quick verify:

```bash
cargo --version
rustc --version
```

### Windows setup notes (PowerShell)

If `cargo`/`rustc` are not found after install, close and reopen PowerShell.

Install required tooling:

```powershell
winget install -e --id Rustlang.Rustup --source winget --accept-source-agreements --accept-package-agreements
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --source winget --accept-source-agreements --accept-package-agreements --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

On Windows ARM64, use the x64 Rust toolchain for this repo:

```powershell
rustup toolchain install stable-x86_64-pc-windows-msvc --force-non-host
rustup target add x86_64-pc-windows-msvc
```

## Build and run

```bash
RUST_LOG=info cargo run --bin grocery-agents -- --token <jwt>
```

You can also paste the full websocket URL directly (token auto-extracted):

```bash
cargo run --bin grocery-agents -- 'wss://game.ainm.no/ws?token=eyJ...'
```

Windows ARM64 helper (uses VS `vcvars64` + x64 Rust toolchain):

```powershell
.\cargo-x64.cmd check --target x86_64-pc-windows-msvc
.\cargo-x64.cmd run --target x86_64-pc-windows-msvc -- --token eyJ...
.\cargo-x64.cmd run --target x86_64-pc-windows-msvc -- "wss://game.ainm.no/ws?token=eyJ..."
```

## Optional Python training setup

Install offline training dependencies:

```bash
python -m pip install -r requirements.txt
```

## Local evaluation runner (Rust)

Run quick metric summaries from existing logs:

```powershell
.\cargo-x64.cmd run --bin eval -- --from-logs --episodes 10
```

Expert-only regression gate from logs:

```powershell
.\cargo-x64.cmd run --bin eval -- --from-logs --episodes 20 --mode-filter expert
```

Run fresh episodes against local websocket simulator:

```powershell
.\cargo-x64.cmd run --bin eval -- --episodes 10 --ws-url ws://localhost:8765/ws --token <jwt>
```

Profiles:

```powershell
# conservative planning budget
.\cargo-x64.cmd run --bin eval -- --episodes 10 --profile safe --ws-url ws://localhost:8765/ws --token <jwt>

# wider search / longer horizon
.\cargo-x64.cmd run --bin eval -- --episodes 10 --profile aggressive --ws-url ws://localhost:8765/ws --token <jwt>
```

Profile behavior:

- `safe`: `legacy_only` assignment mode
- `default`: `hybrid` assignment mode
- `aggressive`: `global_only` assignment mode

The eval summary prints score mean/p50/p90, wait ratio, move/pickup/dropoff counts,
blocked events, and near-dropoff congestion events.

It also prints collapse-focused metrics:

- `collapse_alarms`: late no-delivery streak, goal-collapse ratio, guard-fallback ratio
- phase slices (`early`, `mid`, `late`): score gain, delivered/completed, avg blocked/stuck,
  avg unique-goals, avg goal concentration top-3

## Local Replay UI

You can inspect run behavior visually with a local browser replay tool.

Start server:

```bash
python tools/replay_server.py --port 8085
```

Open:

```text
http://127.0.0.1:8085
```

Features:

- run selector for `logs/run-*.jsonl`
- board rendering with walls, shelf/item cells, dropoff tile, and bot positions
- queue/ring overlays, conflict hotspots, failed-move arrows, and role badges (`L/Q/C/Y`)
- tick playback controls (play/pause, prev/next, speed, slider)
- live refresh while a run file is still being written
- side panel with active/preview order counts, queue congestion metrics, and per-tick actions

### Training-run wrapper (auto-starts UI)

For training games, use the wrapper below so replay UI is always available:

```powershell
.\run-training.cmd -- "wss://game.ainm.no/ws?token=eyJ..."
```

Equivalent Python entrypoint:

```powershell
python tools/run_training_run.py -- "wss://game.ainm.no/ws?token=eyJ..."
```

This does two things each run:

1. Ensures replay server is running on `http://127.0.0.1:8085`
2. Starts bot run via `cargo-x64.cmd run --target x86_64-pc-windows-msvc`
3. Triggers batch retraining whenever at least 10 new completed runs are available

Optional flags:

```powershell
# disable post-run batch retraining for a single run
python tools/run_training_run.py --no-batch-train -- "wss://game.ainm.no/ws?token=eyJ..."

# custom batch size / mode subset
python tools/run_training_run.py --batch-size 10 --train-modes "easy,expert" -- "wss://game.ainm.no/ws?token=eyJ..."
```

### Token input behavior (CLI + env fallback)

The token is configured with Clap as `--token` plus an environment fallback:

- `--token <jwt>` has highest precedence.
- If `--token` is omitted, `GROCERY_TOKEN` is used.
- If still missing, token can be read from a full websocket URL passed as positional argument or `--ws-url`.
- If no token source is provided, startup fails with a CLI argument error.

Examples:

```bash
# Explicit CLI token
RUST_LOG=info cargo run -- --token eyJ...

# Environment fallback
export GROCERY_TOKEN=eyJ...
RUST_LOG=info cargo run
```

PowerShell equivalents:

```powershell
# Explicit CLI token
$env:RUST_LOG="info"
cargo run -- --token eyJ...

# Environment fallback
$env:GROCERY_TOKEN="eyJ..."
$env:RUST_LOG="info"
cargo run
```

## Runtime message flow and timeout strategy

1. Connect to websocket (`GROCERY_WS_URL`, default `ws://localhost:8765/ws`) with `?token=...` query.
2. Receive `game_state` or `game_over` messages.
3. For each `game_state` tick:
   - Build intents via dispatcher.
   - Build movement plan via motion planner.
   - Validate actions before send.
   - Send one `ActionEnvelope` back to server.
4. On `game_over`, log final score/reason and exit loop.

### Per-run game logs (JSONL)

The client writes a run log automatically to `logs/run-<timestamp>.jsonl`.

Each line is a JSON event with top-level `schema_version` and `run_id`, including:

- `session_start` (ws url preview + metadata)
- `game_mode` (detected mode label + bots/grid dimensions)
- `tick` (full `game_state`, chosen `actions`, team summary, and `tick_outcome`)
- `tick_outcome` (`delta_score`, `items_delivered_delta`, `order_completed_delta`, `invalid_action_count`)
- `game_over` (final score + reason)

`team_summary` now also includes coordination telemetry (queue role/slot/distance by bot, queue violations,
near-dropoff blocking flags, repeated failed-move counts, conflict degrees/hotspots, and failed-move arrows).

You can override the output directory with `GAME_LOG_DIR`:

```powershell
$env:GAME_LOG_DIR="my-logs"
.\cargo-x64.cmd run --target x86_64-pc-windows-msvc -- --token eyJ...
```

### Planning timeout strategy

Per tick, round planning runs under a fixed budget (`1400ms`). If planning times out:

- The client logs a warning.
- It emits a full fallback envelope where every bot executes `wait`.
- Normal planning resumes on the next received tick.

## Training pipeline (per-mode specialists)

The repository includes a Python pipeline under `training/` for log-based training:

```bash
python -m training.extract --logs-dir logs --out data/runs.parquet
python -m training.featurize --data data/runs.parquet --out data/runs_features.parquet --n-step 5
python -m training.train --mode medium --data data/runs_features.parquet --out models/medium.json
python -m training.evaluate --data data/runs_features.parquet --model models/medium.json
python -m training.export --models-dir models --out models/policy_artifacts.json
python -m training.batch_train --logs-dir logs --models-dir models --batch-size 10
```

Train one model per mode (`easy`, `medium`, `hard`, `expert`) and export a combined artifact.

At runtime, the Rust bot can consume that artifact by setting:

```bash
POLICY_ARTIFACT_PATH=models/policy_artifacts.json
```

## Architecture by module

- `model`: Shared protocol/domain types (`GameState`, `Action`, envelopes, runtime context).
- `world`: Board/map cache generation (indices, neighbors, item/drop-off lookup tables).
- `dist`: All-pairs grid distance precomputation used by heuristics/planning.
- `dispatcher`: Converts state into per-bot intents (pickup/dropoff/move/wait).
- `motion`: Time-expanded path planning with reservation tables to avoid conflicts.
- `policy`: High-level round decision logic that combines dispatcher + motion and short-term bot memory.
- `team_context`: Shared per-tick blackboard (roles, strict queue lanes, conflict/deadlock context, movement reservations).
- `net`: Websocket networking loop, message parsing, timeout enforcement, action validation.

## Debug logging toggle

Set `BOT_DEBUG=1` (or `true`/`yes`) to enable additional operational logs.

```bash
BOT_DEBUG=1 RUST_LOG=info cargo run --bin grocery-agents -- --token <jwt>
```

When enabled, logs include:

- assignment selection per bot intent
- congestion/jam detection (blocked bot memory and drop-off crowding)
- evacuation trigger events
- fallback/time-budget events (timeout fallback, validation fallback-to-wait)

Coordination knobs:

- `QUEUE_STRICT_MODE=1` (default on in medium/hard/expert)
- `QUEUE_MAX_RING_ENTRANTS=1` (strict queue default)
- `DEADLOCK_ESCAPE_TICKS=3`
- `GROCERY_HORIZON=16`
- `GROCERY_CANDIDATE_K=8`
- `GROCERY_ASSIGNMENT_ENABLED=true`
- `GROCERY_ASSIGNMENT_MODE=hybrid` (`hybrid`, `global-only`, `legacy-only`)
- `GROCERY_DROPOFF_SCHEDULING_ENABLED=true`
- `GROCERY_DROPOFF_WINDOW=12`
- `GROCERY_DROPOFF_CAPACITY=1`
- `GROCERY_LAMBDA_DENSITY=1.0`
- `GROCERY_LAMBDA_CHOKE=1.5`
- `GROCERY_PLANNER_BUDGET_MODE=adaptive` (`adaptive`, `fixed`)
- `GROCERY_PLANNER_SOFT_BUDGET_MS=1200`
- `GROCERY_PLANNER_SOFT_BUDGET_MIN_MS=1350`
- `GROCERY_PLANNER_SOFT_BUDGET_MAX_MS=1900`
- `GROCERY_PLANNER_HARD_BUDGET_MS=1950`
- `GROCERY_PLANNER_DEADLINE_SLACK_MS=80`
- `GROCERY_STRUCTURED_BOT_LOG=1`
- `GROCERY_ASCII_RENDER=1`
- `GROCERY_REPLAY_DUMP_PATH=logs/replay_dump.jsonl`

## Safety fallback behavior (`wait`)

Bots intentionally `wait` in safety-first scenarios:

- round planning exceeded time budget
- proposed action is invalid during client-side validation (bad target bot, blocked/out-of-bounds move, invalid pickup/drop-off)
- no safe or useful intent/path is available for that bot this tick

This prevents sending risky/invalid actions and keeps the client responsive under pressure.
