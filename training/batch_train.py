from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

from .common import read_table

DEFAULT_MODES = ["easy", "medium", "hard", "expert"]


def load_state(path: Path) -> Dict:
    if not path.exists():
        return {"processed_run_ids": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"processed_run_ids": []}
    if not isinstance(payload, dict):
        return {"processed_run_ids": []}
    payload.setdefault("processed_run_ids", [])
    return payload


def save_state(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def completed_run_ids(logs_dir: Path) -> Set[str]:
    out: Set[str] = set()
    for path in sorted(logs_dir.glob("run-*.jsonl")):
        run_id = path.stem
        saw_game_over = False
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    data = record.get("data", {})
                    if isinstance(data, dict):
                        run_id = str(record.get("run_id") or data.get("run_id") or run_id)
                    if record.get("event") == "game_over":
                        saw_game_over = True
        except OSError:
            continue
        if saw_game_over:
            out.add(run_id)
    return out


def run_cmd(args: List[str]) -> None:
    print(">", " ".join(args))
    rc = subprocess.call(args)
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {' '.join(args)}")


def train_modes(
    modes: List[str],
    features_path: Path,
    models_dir: Path,
    min_rows: int,
) -> Dict[str, Dict]:
    frame = read_table(features_path)
    metrics = {}
    for mode in modes:
        mode_rows = int((frame["mode"] == mode).sum()) if "mode" in frame.columns else 0
        if mode_rows < min_rows:
            continue
        out_model = models_dir / f"{mode}.json"
        run_cmd(
            [
                sys.executable,
                "-m",
                "training.train",
                "--mode",
                mode,
                "--data",
                str(features_path),
                "--out",
                str(out_model),
                "--min-rows",
                str(min_rows),
            ]
        )
        payload = json.loads(out_model.read_text(encoding="utf-8"))
        metrics[mode] = payload.get("metrics", {})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch retraining after every N completed runs.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--modes", default="easy,medium,hard,expert")
    parser.add_argument("--state-path", default="models/batch_state.json")
    parser.add_argument("--data-out", default="data/runs.parquet")
    parser.add_argument("--features-out", default="data/runs_features.parquet")
    parser.add_argument("--min-rows", type=int, default=50)
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    models_dir = Path(args.models_dir)
    state_path = Path(args.state_path)
    data_out = Path(args.data_out)
    features_out = Path(args.features_out)
    models_dir.mkdir(parents=True, exist_ok=True)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not modes:
        modes = list(DEFAULT_MODES)

    state = load_state(state_path)
    processed = set(str(v) for v in state.get("processed_run_ids", []))
    completed = completed_run_ids(logs_dir)
    pending = sorted(completed - processed)
    if len(pending) < args.batch_size:
        print(
            f"batch_train: pending completed runs={len(pending)} (< {args.batch_size}), skipping."
        )
        return

    run_cmd(
        [
            sys.executable,
            "-m",
            "training.extract",
            "--logs-dir",
            str(logs_dir),
            "--out",
            str(data_out),
        ]
    )
    run_cmd(
        [
            sys.executable,
            "-m",
            "training.featurize",
            "--data",
            str(data_out),
            "--out",
            str(features_out),
            "--n-step",
            "5",
        ]
    )

    metrics = train_modes(modes, features_out, models_dir, args.min_rows)
    run_cmd(
        [
            sys.executable,
            "-m",
            "training.export",
            "--models-dir",
            str(models_dir),
            "--out",
            str(models_dir / "policy_artifacts.json"),
        ]
    )

    now_ms = int(time.time() * 1000)
    snapshot = {
        "ts_ms": now_ms,
        "batch_size": args.batch_size,
        "runs_seen": len(completed),
        "runs_processed_before": len(processed),
        "runs_processed_after": len(completed),
        "pending_runs_trained": pending,
        "modes": modes,
        "metrics": metrics,
    }
    metrics_path = models_dir / f"metrics-{now_ms}.json"
    metrics_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    state["processed_run_ids"] = sorted(completed)
    state["last_batch_ts_ms"] = now_ms
    state["last_metrics_path"] = str(metrics_path)
    save_state(state_path, state)
    print(
        f"batch_train: trained modes={sorted(metrics.keys())} from {len(pending)} pending runs. "
        f"metrics={metrics_path}"
    )


if __name__ == "__main__":
    main()
