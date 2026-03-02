from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ensure_replay_server import DEFAULT_HOST, DEFAULT_PORT, ROOT_DIR, ensure_server


def normalize_cargo_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def build_cmd(target: str, cargo_args: list[str]) -> list[str]:
    cargo_script = ROOT_DIR / "cargo-x64.cmd"
    if not cargo_script.exists():
        raise FileNotFoundError(f"missing cargo wrapper: {cargo_script}")
    return [
        "cmd",
        "/c",
        str(cargo_script),
        "run",
        "--target",
        target,
        "--bin",
        "grocery-agents",
        "--",
        *cargo_args,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one training game and ensure replay UI is running."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--logs-dir", default=str(ROOT_DIR / "logs"))
    parser.add_argument("--target", default="x86_64-pc-windows-msvc")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--no-batch-train", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--train-modes", default="easy,medium,hard,expert")
    parser.add_argument(
        "cargo_args",
        nargs=argparse.REMAINDER,
        help="Bot args after `--`, e.g. -- \"wss://game.ainm.no/ws?token=...\"",
    )
    args = parser.parse_args()

    if not args.no_ui:
        ok = ensure_server(args.host, args.port, Path(args.logs_dir))
        if not ok:
            print(
                f"Failed to ensure replay UI at http://{args.host}:{args.port}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"Replay UI: http://{args.host}:{args.port}")

    cargo_args = normalize_cargo_args(args.cargo_args)
    if not cargo_args:
        print(
            "Missing bot run arguments. Example:\n"
            "  python tools/run_training_run.py -- \"wss://game.ainm.no/ws?token=...\"",
            file=sys.stderr,
        )
        raise SystemExit(2)

    cmd = build_cmd(args.target, cargo_args)
    rc = subprocess.call(cmd, cwd=str(ROOT_DIR))
    if rc == 0 and not args.no_batch_train:
        batch_cmd = [
            sys.executable,
            "-m",
            "training.batch_train",
            "--logs-dir",
            args.logs_dir,
            "--models-dir",
            str(ROOT_DIR / "models"),
            "--batch-size",
            str(args.batch_size),
            "--modes",
            args.train_modes,
        ]
        batch_rc = subprocess.call(batch_cmd, cwd=str(ROOT_DIR))
        if batch_rc != 0:
            print(f"Batch training failed with exit code {batch_rc}", file=sys.stderr)
            rc = batch_rc
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
