from __future__ import annotations

import argparse
import json
import shutil
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


def run_json_cmd(args: List[str]) -> Dict:
    print(">", " ".join(args))
    proc = subprocess.run(args, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr}"
        )
    payload = proc.stdout.strip()
    if not payload:
        return {}
    return json.loads(payload)


def train_modes(
    modes: List[str],
    features_path: Path,
    out_dir: Path,
    min_rows: int,
    trainer_backend: str,
    runtime_feature_set: str,
    dedup_strategy: str,
    signature_kind: str,
) -> Dict[str, Path]:
    frame = read_table(features_path)
    model_paths: Dict[str, Path] = {}
    for mode in modes:
        mode_rows = int((frame["mode"] == mode).sum()) if "mode" in frame.columns else 0
        if mode_rows < min_rows:
            continue
        out_model = out_dir / f"{mode}.json"
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
                "--runtime-feature-set",
                runtime_feature_set,
                "--dedup-strategy",
                dedup_strategy,
                "--signature-kind",
                signature_kind,
                "--trainer-backend",
                trainer_backend,
            ]
        )
        model_paths[mode] = out_model
    return model_paths


def evaluate_model(features_path: Path, model_path: Path, mode: str) -> Dict:
    if not model_path.exists():
        return {}
    return run_json_cmd(
        [
            sys.executable,
            "-m",
            "training.evaluate",
            "--data",
            str(features_path),
            "--model",
            str(model_path),
            "--mode",
            mode,
        ]
    )


def _get(d: Dict, *path: str, default: float = 0.0) -> float:
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    try:
        return float(cur)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def candidate_beats_champion(mode: str, champion: Dict, candidate: Dict) -> tuple[bool, Dict[str, float]]:
    # No-regression constraints.
    cand_pick_auc = _get(candidate, "metrics_v2", "pickup", "auc", default=0.5)
    champ_pick_auc = _get(champion, "metrics_v2", "pickup", "auc", default=0.5)
    cand_drop_auc = _get(candidate, "metrics_v2", "dropoff", "auc", default=0.5)
    champ_drop_auc = _get(champion, "metrics_v2", "dropoff", "auc", default=0.5)
    cand_pick_loss = _get(candidate, "metrics_v2", "pickup", "logloss", default=1.0)
    champ_pick_loss = _get(champion, "metrics_v2", "pickup", "logloss", default=1.0)
    cand_drop_loss = _get(candidate, "metrics_v2", "dropoff", "logloss", default=1.0)
    champ_drop_loss = _get(champion, "metrics_v2", "dropoff", "logloss", default=1.0)
    cand_mae = _get(candidate, "mae", default=1e9)
    champ_mae = _get(champion, "mae", default=1e9)

    no_regression = (
        cand_pick_auc + 0.01 >= champ_pick_auc
        and cand_drop_auc + 0.01 >= champ_drop_auc
        and cand_pick_loss <= champ_pick_loss + 0.02
        and cand_drop_loss <= champ_drop_loss + 0.02
        and cand_mae <= champ_mae + 0.05
    )

    # Improvement criteria (mode-specific plus generic).
    cand_corr = _get(candidate, "metrics_v2", "expected_score_delivery_corr", default=0.0)
    champ_corr = _get(champion, "metrics_v2", "expected_score_delivery_corr", default=0.0)
    corr_gain_target = (
        champ_corr * 1.05 if champ_corr > 0 else champ_corr + 0.01
    )
    corr_improve = cand_corr >= corr_gain_target

    cand_seq_auc = _get(candidate, "metrics_v2", "ordering_sequence", "pair_auc", default=0.5)
    champ_seq_auc = _get(champion, "metrics_v2", "ordering_sequence", "pair_auc", default=0.5)
    seq_improve = cand_seq_auc >= champ_seq_auc + 0.01

    # Spatial idle improvements are mode-sensitive.
    cand_idle_far = _get(
        candidate, "metrics_v2", "spatial_idle", "idle_far_from_dropoff_rate", default=1.0
    )
    champ_idle_far = _get(
        champion, "metrics_v2", "spatial_idle", "idle_far_from_dropoff_rate", default=1.0
    )
    idle_improve = cand_idle_far <= champ_idle_far * 0.95

    if mode == "expert":
        improve_any = corr_improve or seq_improve or idle_improve
    elif mode == "medium":
        improve_any = corr_improve or seq_improve
    else:
        improve_any = corr_improve or seq_improve

    return no_regression and improve_any, {
        "cand_pick_auc": cand_pick_auc,
        "champ_pick_auc": champ_pick_auc,
        "cand_drop_auc": cand_drop_auc,
        "champ_drop_auc": champ_drop_auc,
        "cand_pick_loss": cand_pick_loss,
        "champ_pick_loss": champ_pick_loss,
        "cand_drop_loss": cand_drop_loss,
        "champ_drop_loss": champ_drop_loss,
        "cand_mae": cand_mae,
        "champ_mae": champ_mae,
        "cand_corr": cand_corr,
        "champ_corr": champ_corr,
        "cand_seq_auc": cand_seq_auc,
        "champ_seq_auc": champ_seq_auc,
        "cand_idle_far": cand_idle_far,
        "champ_idle_far": champ_idle_far,
    }


def promote_candidates(
    models_dir: Path,
    candidate_dir: Path,
    features_path: Path,
    modes: List[str],
) -> Dict[str, Dict]:
    decisions: Dict[str, Dict] = {}
    promoted_modes: List[str] = []
    for mode in modes:
        candidate_path = candidate_dir / f"{mode}.json"
        if not candidate_path.exists():
            decisions[mode] = {"status": "skipped", "reason": "no_candidate_model"}
            continue
        champion_path = models_dir / f"{mode}.json"
        candidate_eval = evaluate_model(features_path, candidate_path, mode)
        if not champion_path.exists():
            shutil.copy2(candidate_path, champion_path)
            promoted_modes.append(mode)
            decisions[mode] = {
                "status": "promoted",
                "reason": "no_existing_champion",
                "candidate_eval": candidate_eval,
            }
            continue
        champion_eval = evaluate_model(features_path, champion_path, mode)
        promote, diagnostics = candidate_beats_champion(mode, champion_eval, candidate_eval)
        if promote:
            shutil.copy2(candidate_path, champion_path)
            promoted_modes.append(mode)
            decisions[mode] = {
                "status": "promoted",
                "reason": "candidate_beats_champion",
                "diagnostics": diagnostics,
                "candidate_eval": candidate_eval,
                "champion_eval": champion_eval,
            }
        else:
            decisions[mode] = {
                "status": "rejected",
                "reason": "gate_not_met",
                "diagnostics": diagnostics,
                "candidate_eval": candidate_eval,
                "champion_eval": champion_eval,
            }

    # Export only after promotion decisions are applied.
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
    decisions["_promoted_modes"] = {"modes": promoted_modes}
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch retraining with challenger/champion promotion gates."
    )
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--modes", default="easy,medium,hard,expert")
    parser.add_argument("--state-path", default="models/batch_state.json")
    parser.add_argument("--champion-state-path", default="models/champion_state.json")
    parser.add_argument("--data-out", default="data/runs.parquet")
    parser.add_argument("--features-out", default="data/runs_features.parquet")
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--trainer-backend", choices=["auto", "torch", "ridge"], default="auto")
    parser.add_argument("--runtime-feature-set", choices=["strict", "extended"], default="strict")
    parser.add_argument("--dedup-strategy", choices=["none", "downweight", "drop"], default="downweight")
    parser.add_argument("--signature-kind", choices=["action", "state_action"], default="action")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    models_dir = Path(args.models_dir)
    candidate_dir = models_dir / "candidate"
    state_path = Path(args.state_path)
    champion_state_path = Path(args.champion_state_path)
    data_out = Path(args.data_out)
    features_out = Path(args.features_out)
    models_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

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

    for old in candidate_dir.glob("*.json"):
        try:
            old.unlink()
        except OSError:
            pass
    candidate_models = train_modes(
        modes,
        features_out,
        candidate_dir,
        args.min_rows,
        trainer_backend=args.trainer_backend,
        runtime_feature_set=args.runtime_feature_set,
        dedup_strategy=args.dedup_strategy,
        signature_kind=args.signature_kind,
    )
    decisions = promote_candidates(
        models_dir=models_dir,
        candidate_dir=candidate_dir,
        features_path=features_out,
        modes=sorted(candidate_models.keys()),
    )

    now_ms = int(time.time() * 1000)
    snapshot = {
        "ts_ms": now_ms,
        "batch_size": args.batch_size,
        "runs_seen": len(completed),
        "runs_processed_before": len(processed),
        "runs_processed_after": len(completed),
        "pending_runs_trained": pending,
        "modes_requested": modes,
        "modes_candidate_trained": sorted(candidate_models.keys()),
        "promotion_decisions": decisions,
    }
    metrics_path = models_dir / f"metrics-{now_ms}.json"
    metrics_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    champion_state = {
        "last_promotion_ts_ms": now_ms,
        "last_metrics_path": str(metrics_path),
        "promotion_decisions": decisions,
    }
    champion_state_path.write_text(json.dumps(champion_state, indent=2), encoding="utf-8")

    state["processed_run_ids"] = sorted(completed)
    state["last_batch_ts_ms"] = now_ms
    state["last_metrics_path"] = str(metrics_path)
    save_state(state_path, state)
    print(
        f"batch_train: candidate_modes={sorted(candidate_models.keys())} "
        f"promoted={decisions.get('_promoted_modes', {}).get('modes', [])} "
        f"metrics={metrics_path}"
    )


if __name__ == "__main__":
    main()
