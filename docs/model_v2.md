# Model v2: Conversion-First Multi-Head Scoring

## Overview
Model v2 extends the per-mode policy artifacts with conversion-aware heads while preserving v1 compatibility.

- Schema version: `1.2.0`
- Artifact file: `models/policy_artifacts.json`
- Layout: per-mode map under `modes`

## Heads
Each mode can include:

- `pickup` head: ridge-logit + sigmoid + temperature
- `dropoff` head: ridge-logit + sigmoid + temperature
- `ordering` head: ridge linear score (value proxy)

The runtime combined score is:

`combined_expected_score = value_proxy * pickup_prob * dropoff_prob / (eta + 1.0)`

Where:
- `value_proxy` = `ordering_score` when available, otherwise urgency fallback
- `eta` = `dist_to_nearest_active_item` (clamped)

## New Features and Labels
### Conversion labels
- `pickup_attempt`
- `pickup_success`
- `dropoff_attempt`
- `dropoff_success`

Labels are inferred with one-tick alignment via carrying deltas.

### Reliability features
- `stand_failure_count_recent`
- `stand_success_count_recent`
- `stand_cooldown_ticks_remaining`
- `kind_failure_count_recent`
- `repeated_same_stand_no_delta_streak`
- `contention_at_stand_proxy`
- `time_since_last_conversion_tick`
- `last_conversion_was_pickup`
- `last_conversion_was_dropoff`

## Artifact Schema (per mode)
```json
{
  "weights": {"bias": 0.0, "...": 0.0},
  "ordering_weights": {"bias": 0.0, "...": 0.0},
  "feature_columns": ["..."],
  "normalization": {
    "mean": [0.0],
    "std": [1.0]
  },
  "heads": {
    "pickup": {
      "type": "ridge_logit",
      "bias": 0.0,
      "weights": [0.0],
      "clip": 8.0,
      "temperature": 1.0
    },
    "dropoff": {
      "type": "ridge_logit",
      "bias": 0.0,
      "weights": [0.0],
      "clip": 8.0,
      "temperature": 1.0
    },
    "ordering": {
      "type": "ridge",
      "bias": 0.0,
      "weights": [0.0],
      "clip": 12.0
    }
  },
  "calibration": {
    "pickup_temp": 1.0,
    "dropoff_temp": 1.0
  }
}
```

## Backward Compatibility
If v2 fields are missing, runtime falls back to v1 behavior:
- legacy linear pick score (`weights`)
- unchanged ordering scorer (`ordering_weights`)

## Training + Eval Commands
```powershell
python -m training.extract --logs-dir logs --out data/runs.parquet
python -m training.featurize --data data/runs.parquet --out data/runs_features.parquet --n-step 5
python -m training.train --mode expert --data data/runs_features.parquet --out models/expert.json --dedup-strategy downweight --signature-kind action
python -m training.evaluate --data data/runs_features.parquet --model models/expert.json --mode expert --reference-score 101 --candidate-score 85
python -m training.export --models-dir models --out models/policy_artifacts.json
```

## Dedup Strategy
`training.train` can control duplicate trajectories before fitting:

- `--dedup-strategy none`: keep all runs with weight `1.0`
- `--dedup-strategy downweight` (default): keep all runs, but each run gets weight `1 / cluster_size`
- `--dedup-strategy drop`: keep one deterministic representative run per signature cluster

Run signatures are deterministic and use:
- `--signature-kind action`: action/goal trace only
- `--signature-kind state_action`: action trace plus bot position/carrying state

The trained model JSON now includes a `dedup` section with cluster stats and effective run weight.

## Eval Gates
Use the Rust eval binary to print conversion KPIs and optional hard-fail gates:

```powershell
.\cargo-x64.cmd run --bin eval -- --from-logs --episodes 20 --mode-filter expert
.\cargo-x64.cmd run --bin eval -- --from-logs --episodes 20 --mode-filter expert --enforce-gates
```
