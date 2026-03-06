# grocery-agents

Deterministic Rust bot for the grocery challenge, optimized for throughput:

- deliver as many active-order items as possible in 300 rounds
- collect +5 order-completion bonuses by minimizing order transition latency
- pipeline preview-order items without delaying active completion

## Runtime Focus

This repository is runtime-only.

Implemented planners by difficulty:

- `easy` (1 bot, 12x10): exact small-horizon routing via brute-force permutations
- `medium` (3 bots, 16x12): centralized assignment + short-horizon reservations + dedicated stager
- `hard` (5 bots, 22x14): regional pickers + runner/stager + claim-based duplication control
- `expert` (10 bots, 28x18): role-specialized swarm (pickers/runners/buffers) + marginal-gain scheduler

## Prerequisites

- Rust toolchain: https://rustup.rs/
- On Windows: Visual Studio C++ Build Tools (for `link.exe`)

## Build

```powershell
cargo check
cargo test
```

## Run

Use token directly:

```powershell
cargo run --bin grocery-agents -- --token <jwt>
```

Or full websocket URL:

```powershell
cargo run --bin grocery-agents -- "wss://game.ainm.no/ws?token=<jwt>"
```

Policy override:

```powershell
cargo run --bin grocery-agents -- --policy expert --token <jwt>
```

Replay mode:

```powershell
cargo run --bin grocery-agents -- --replay logs/run-<timestamp>.jsonl
```

## Offline Planner Lab

Local deterministic replay + what-if UI:

```powershell
cargo run --bin lab -- --logs-dir logs --port 8085 --seed 42
```

Then open `http://127.0.0.1:8085`.

Offline batch scoring:

```powershell
cargo run --bin sim_eval -- --logs-dir logs --episodes 20 --policy auto --seed 42 --out reports/eval.json
```

## Key Environment Variables

- `GROCERY_POLICY=auto|easy|medium|hard|expert`
- `GROCERY_PLANNER_BUDGET_MODE=adaptive|fixed`
- `GROCERY_TICK_SOFT_BUDGET_MS`
- `GROCERY_TICK_HARD_BUDGET_MS`
- `GROCERY_TICK_GREEDY_FALLBACK_MS`
- `GROCERY_CACHE_REUSE_MAX_AGE_TICKS`
- `GROCERY_CACHE_REQUIRE_PROGRESS=true|false`
- `GROCERY_LOG_LEVEL=info|debug|...`
- `GROCERY_STRUCTURED_BOT_LOG=1`
- `GROCERY_ASCII_RENDER=1`
- `GROCERY_DEBUG=1`

## Module Layout

- `src/model.rs`: protocol/domain types and wire conversion
- `src/world.rs`: static map cache and walkability
- `src/dist.rs`: distance precomputation
- `src/difficulty.rs`: difficulty inference
- `src/planner/*`: per-difficulty planning + shared helpers + MAPF
- `src/policy.rs`: planner orchestration + telemetry
- `src/net.rs`: websocket loop, timeout handling, safe validation, replay playback

## Testing Scope

Primary tests target planner correctness and safety-critical behavior:

- exact easy-trip construction
- medium stager/finisher behavior
- hard role + claim behavior
- expert role partitioning
- map/path invariants from existing map/model tests
