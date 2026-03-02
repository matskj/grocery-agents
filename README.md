# grocery-agents

A Rust bot client for the Norwegian AI Championship warm-up grocery game.

## Build and run

```bash
RUST_LOG=info cargo run -- --token <jwt>
```

### Token input behavior (CLI + env fallback)

The token is configured with Clap as `--token` plus an environment fallback:

- `--token <jwt>` has highest precedence.
- If `--token` is omitted, `GROCERY_TOKEN` is used.
- If neither is provided, startup fails with a CLI argument error.

Examples:

```bash
# Explicit CLI token
RUST_LOG=info cargo run -- --token eyJ...

# Environment fallback
export GROCERY_TOKEN=eyJ...
RUST_LOG=info cargo run
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

### Planning timeout strategy

Per tick, round planning runs under a fixed budget (`1400ms`). If planning times out:

- The client logs a warning.
- It emits a full fallback envelope where every bot executes `wait`.
- Normal planning resumes on the next received tick.

## Architecture by module

- `model`: Shared protocol/domain types (`GameState`, `Action`, envelopes, runtime context).
- `world`: Board/map cache generation (indices, neighbors, item/drop-off lookup tables).
- `dist`: All-pairs grid distance precomputation used by heuristics/planning.
- `dispatcher`: Converts state into per-bot intents (pickup/dropoff/move/wait).
- `motion`: Time-expanded path planning with reservation tables to avoid conflicts.
- `policy`: High-level round decision logic that combines dispatcher + motion and short-term bot memory.
- `net`: Websocket networking loop, message parsing, timeout enforcement, action validation.

## Debug logging toggle

Set `BOT_DEBUG=1` (or `true`/`yes`) to enable additional operational logs.

```bash
BOT_DEBUG=1 RUST_LOG=info cargo run -- --token <jwt>
```

When enabled, logs include:

- assignment selection per bot intent
- congestion/jam detection (blocked bot memory and drop-off crowding)
- evacuation trigger events
- fallback/time-budget events (timeout fallback, validation fallback-to-wait)

## Safety fallback behavior (`wait`)

Bots intentionally `wait` in safety-first scenarios:

- round planning exceeded time budget
- proposed action is invalid during client-side validation (bad target bot, blocked/out-of-bounds move, invalid pickup/drop-off)
- no safe or useful intent/path is available for that bot this tick

This prevents sending risky/invalid actions and keeps the client responsive under pressure.
