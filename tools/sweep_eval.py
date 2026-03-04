from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]

METRIC_PATTERN = {
    "delivered_per_100_ticks": re.compile(r"delivered_per_100_ticks=([0-9.]+)"),
    "pickup_success_ratio": re.compile(r"pickup_success_ratio=([0-9.]+)"),
    "dropoff_success_ratio": re.compile(r"dropoff_success_ratio=([0-9.]+)"),
    "far_no_conversion_tick_ratio": re.compile(r"far_no_conversion_tick_ratio=([0-9.]+)"),
    "collector_far_wait_ratio": re.compile(r"collector_far_wait_ratio=([0-9.]+)"),
    "local_first_violation_ratio": re.compile(r"local_first_violation_ratio=([0-9.]+)"),
    "preferred_area_match_ratio": re.compile(r"preferred_area_match_ratio=([0-9.]+)"),
}


def parse_metrics(stdout: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, pattern in METRIC_PATTERN.items():
        match = pattern.search(stdout)
        if match:
            out[key] = float(match.group(1))
    return out


def objective(metrics: Dict[str, float]) -> float:
    return (
        0.45 * metrics.get("delivered_per_100_ticks", 0.0)
        + 0.20 * metrics.get("pickup_success_ratio", 0.0)
        + 0.20 * metrics.get("dropoff_success_ratio", 0.0)
        - 0.10 * metrics.get("far_no_conversion_tick_ratio", 1.0)
        - 0.05 * metrics.get("collector_far_wait_ratio", 1.0)
        - 0.05 * metrics.get("local_first_violation_ratio", 1.0)
        + 0.05 * metrics.get("preferred_area_match_ratio", 0.0)
    )


def run_eval(args: List[str], env: Dict[str, str]) -> Dict[str, float]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"eval command failed ({proc.returncode}): {' '.join(args)}\n{proc.stdout}\n{proc.stderr}"
        )
    return parse_metrics(proc.stdout)


def cargo_eval_base() -> List[str]:
    if os.name == "nt":
        return ["cmd", "/c", "cargo-x64.cmd", "run", "--bin", "eval", "--"]
    return ["cargo", "run", "--bin", "eval", "--"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic parameter sweep for log replay / local eval."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--mode-filter", default=None)
    parser.add_argument("--out", default="models/sweep_results.json")
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--ws-url", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--coord-baseline", default="models/coord_baseline.json")
    parser.add_argument("--strict-all-modes", action="store_true")
    parser.add_argument("--write-coord-baseline", action="store_true")
    args = parser.parse_args()

    horizons = [16, 18, 20]
    candidate_ks = [8, 10, 12]
    soft_ranges = [(1350, 1650), (1450, 1800), (1550, 1900)]
    lambda_density = [0.8, 1.0, 1.2]
    lambda_choke = [1.2, 1.5, 1.8]

    base_cmd = cargo_eval_base()
    replay_cmd = base_cmd + [
        "--from-logs",
        "--episodes",
        str(args.episodes),
        "--logs-dir",
        args.logs_dir,
    ]
    if args.mode_filter:
        replay_cmd += ["--mode-filter", args.mode_filter]
    if args.strict_all_modes and not args.mode_filter:
        replay_cmd += [
            "--strict-all-modes",
            "--coord-baseline",
            args.coord_baseline,
            "--enforce-gates",
        ]
    if args.write_coord_baseline:
        replay_cmd += ["--write-coord-baseline", "--coord-baseline", args.coord_baseline]

    if args.run_live:
        if not args.ws_url or not args.token:
            raise SystemExit("--run-live requires --ws-url and --token")
        live_cmd = base_cmd + [
            "--episodes",
            str(args.episodes),
            "--ws-url",
            args.ws_url,
            "--token",
            args.token,
        ]
        if args.mode_filter:
            live_cmd += ["--mode-filter", args.mode_filter]
    else:
        live_cmd = []

    all_results = []
    for horizon, k, (soft_min, soft_max), ld, lc in itertools.product(
        horizons, candidate_ks, soft_ranges, lambda_density, lambda_choke
    ):
        env = dict(os.environ)
        env.update(
            {
                "GROCERY_HORIZON": str(horizon),
                "GROCERY_CANDIDATE_K": str(k),
                "GROCERY_PLANNER_BUDGET_MODE": "adaptive",
                "GROCERY_PLANNER_SOFT_BUDGET_MIN_MS": str(soft_min),
                "GROCERY_PLANNER_SOFT_BUDGET_MAX_MS": str(soft_max),
                "GROCERY_LAMBDA_DENSITY": str(ld),
                "GROCERY_LAMBDA_CHOKE": str(lc),
            }
        )
        replay_metrics = run_eval(replay_cmd, env)
        live_metrics = run_eval(live_cmd, env) if live_cmd else {}
        merged = dict(replay_metrics)
        if live_metrics:
            for key, value in live_metrics.items():
                merged[f"live_{key}"] = value
                merged[key] = (merged.get(key, value) + value) / 2.0
        score = objective(merged)
        row = {
            "params": {
                "horizon": horizon,
                "candidate_k": k,
                "soft_min": soft_min,
                "soft_max": soft_max,
                "lambda_density": ld,
                "lambda_choke": lc,
            },
            "metrics": merged,
            "objective": score,
        }
        all_results.append(row)
        print(
            f"sweep horizon={horizon} k={k} soft=({soft_min},{soft_max}) "
            f"lambda=({ld},{lc}) objective={score:.4f}"
        )

    all_results.sort(key=lambda row: row.get("objective", -1e9), reverse=True)
    payload = {
        "ts_ms": int(time.time() * 1000),
        "episodes": args.episodes,
        "mode_filter": args.mode_filter,
        "results": all_results,
        "best": all_results[0] if all_results else None,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote sweep results to {out_path}")


if __name__ == "__main__":
    main()
