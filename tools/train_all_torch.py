from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODES = ("easy", "medium", "hard", "expert")


def detect_python(preferred: str | None) -> str:
    if preferred:
        return preferred
    torch_venv = ROOT_DIR / ".venv311-torch" / "Scripts" / "python.exe"
    if torch_venv.exists():
        return str(torch_venv)
    return sys.executable


def run_cmd(args: list[str], env: dict[str, str]) -> None:
    print(">", " ".join(args))
    rc = subprocess.call(args, cwd=str(ROOT_DIR), env=env)
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {' '.join(args)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full all-mode retrain with torch-backed heads and artifact export."
    )
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--data-out", default="data/runs.parquet")
    parser.add_argument("--features-out", default="data/runs_features.parquet")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--modes", default="easy,medium,hard,expert")
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--runtime-feature-set", choices=["strict", "extended"], default="strict")
    parser.add_argument("--dedup-strategy", choices=["none", "downweight", "drop"], default="downweight")
    parser.add_argument("--signature-kind", choices=["action", "state_action"], default="action")
    parser.add_argument("--trainer-backend", choices=["auto", "torch", "ridge"], default="auto")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    py = detect_python(args.python_bin)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        modes = list(DEFAULT_MODES)

    env = dict(os.environ)
    models_dir = ROOT_DIR / args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            py,
            "-m",
            "training.extract",
            "--logs-dir",
            args.logs_dir,
            "--out",
            args.data_out,
        ],
        env,
    )
    run_cmd(
        [
            py,
            "-m",
            "training.featurize",
            "--data",
            args.data_out,
            "--out",
            args.features_out,
            "--n-step",
            str(args.n_step),
        ],
        env,
    )

    for mode in modes:
        out_model = models_dir / f"{mode}.json"
        run_cmd(
            [
                py,
                "-m",
                "training.train",
                "--mode",
                mode,
                "--data",
                args.features_out,
                "--out",
                str(out_model),
                "--min-rows",
                str(args.min_rows),
                "--runtime-feature-set",
                args.runtime_feature_set,
                "--dedup-strategy",
                args.dedup_strategy,
                "--signature-kind",
                args.signature_kind,
                "--trainer-backend",
                args.trainer_backend,
            ],
            env,
        )
        if not args.skip_eval:
            run_cmd(
                [
                    py,
                    "-m",
                    "training.evaluate",
                    "--data",
                    args.features_out,
                    "--model",
                    str(out_model),
                    "--mode",
                    mode,
                ],
                env,
            )

    run_cmd(
        [
            py,
            "-m",
            "training.export",
            "--models-dir",
            str(models_dir),
            "--out",
            str(models_dir / "policy_artifacts.json"),
        ],
        env,
    )
    print(f"Full retrain complete. Artifact: {models_dir / 'policy_artifacts.json'}")


if __name__ == "__main__":
    main()
