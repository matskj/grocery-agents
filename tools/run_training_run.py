from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ensure_replay_server import DEFAULT_HOST, DEFAULT_PORT, ROOT_DIR, ensure_server

WS_URL_PATTERN = re.compile(r"wss://game\.ainm\.no/ws\?token=[A-Za-z0-9._~-]+")


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


def extract_ws_url(text: str) -> str | None:
    match = WS_URL_PATTERN.search(text)
    if match:
        return match.group(0)
    return None


def fetch_ws_url_playwright(
    python_bin: str,
    difficulty: str,
    state_path: str,
    app_url: str,
    timeout_ms: int,
    headed: bool,
) -> str:
    cmd = [
        python_bin,
        str(ROOT_DIR / "tools" / "fetch_ws_url_playwright.py"),
        "--difficulty",
        difficulty,
        "--state-path",
        state_path,
        "--app-url",
        app_url,
        "--timeout-ms",
        str(timeout_ms),
    ]
    if headed:
        cmd.append("--headed")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(
            "Failed to fetch ws url via Playwright.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )
    ws_url = extract_ws_url(proc.stdout) or extract_ws_url(proc.stderr)
    if not ws_url:
        raise RuntimeError(
            "Playwright fetch completed without returning a ws URL. "
            "Check login state and selector compatibility."
        )
    return ws_url


def fetch_ws_url_http(
    python_bin: str,
    difficulty: str,
    endpoint: str | None,
    map_id: str | None,
    method: str,
    origin: str,
    referer: str,
    bearer_env: str,
    cookie_env: str,
    timeout_seconds: float,
    extra_headers: list[str],
) -> str:
    cmd = [
        python_bin,
        str(ROOT_DIR / "tools" / "fetch_ws_url_http.py"),
        "--difficulty",
        difficulty,
        "--method",
        method,
        "--origin",
        origin,
        "--referer",
        referer,
        "--bearer-env",
        bearer_env,
        "--cookie-env",
        cookie_env,
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    if endpoint:
        cmd.extend(["--endpoint", endpoint])
    if map_id:
        cmd.extend(["--map-id", map_id])
    for header in extra_headers:
        cmd.extend(["--header", header])
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(
            "Failed to fetch ws url via HTTP play endpoint.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )
    ws_url = extract_ws_url(proc.stdout) or extract_ws_url(proc.stderr)
    if not ws_url:
        raise RuntimeError(
            "HTTP fetch completed without returning a ws URL. "
            "Check endpoint/auth headers/cookies."
        )
    return ws_url


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
    parser.add_argument("--full-retrain", action="store_true")
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cooldown-seconds", type=float, default=62.0)
    parser.add_argument(
        "--auto-token-provider",
        choices=["playwright", "http"],
        default="playwright",
        help="Token refresh provider used with --auto-token-difficulty.",
    )
    parser.add_argument(
        "--auto-token-difficulty",
        choices=["easy", "medium", "hard", "expert"],
        default=None,
        help="Auto-fetch ws URL for this difficulty before each run.",
    )
    parser.add_argument(
        "--auto-token-state-path",
        default=str(ROOT_DIR / ".secrets" / "ainm_storage_state.json"),
        help="Persistent Playwright storage state file (login cookies/session).",
    )
    parser.add_argument(
        "--auto-token-app-url",
        default="https://app.ainm.no/challenge",
        help="Challenge page URL used by Playwright token fetcher.",
    )
    parser.add_argument("--auto-token-timeout-ms", type=int, default=30_000)
    parser.add_argument(
        "--auto-token-headed",
        action="store_true",
        help="Run Playwright in headed mode (useful for first login/session bootstrap).",
    )
    parser.add_argument(
        "--auto-token-endpoint",
        default=os.getenv("AINM_PLAY_ENDPOINT"),
        help="HTTP play endpoint (used by --auto-token-provider http).",
    )
    parser.add_argument(
        "--auto-token-map-id",
        default=None,
        help="Optional map_id override for HTTP token provider.",
    )
    parser.add_argument(
        "--auto-token-http-method",
        choices=["POST", "GET"],
        default="POST",
        help="HTTP method for token endpoint.",
    )
    parser.add_argument(
        "--auto-token-origin",
        default="https://app.ainm.no",
        help="Origin header for HTTP token endpoint.",
    )
    parser.add_argument(
        "--auto-token-referer",
        default="https://app.ainm.no/challenge",
        help="Referer header for HTTP token endpoint.",
    )
    parser.add_argument(
        "--auto-token-bearer-env",
        default="AINM_ACCESS_TOKEN",
        help="Env var containing bearer token for HTTP provider.",
    )
    parser.add_argument(
        "--auto-token-cookie-env",
        default="AINM_COOKIE",
        help="Env var containing cookie header for HTTP provider.",
    )
    parser.add_argument(
        "--auto-token-header",
        action="append",
        default=[],
        help="Extra HTTP header for token endpoint, format 'Key: Value'. Repeatable.",
    )
    parser.add_argument(
        "--auto-token-timeout-seconds",
        type=float,
        default=15.0,
        help="HTTP token fetch timeout in seconds.",
    )
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
    if not cargo_args and not args.auto_token_difficulty:
        print(
            "Missing bot run arguments or auto token mode. Example:\n"
            "  python tools/run_training_run.py -- \"wss://game.ainm.no/ws?token=...\"",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    if cargo_args and args.auto_token_difficulty:
        print(
            "Both cargo args and --auto-token-difficulty provided; "
            "auto-token mode will be used.",
            file=sys.stderr,
        )

    env = dict(os.environ)
    env.setdefault(
        "POLICY_ARTIFACT_PATH", str((ROOT_DIR / "models" / "policy_artifacts.json").resolve())
    )
    py = args.python_bin or sys.executable
    rc = 0
    successful_runs = 0
    for run_index in range(args.repeat):
        run_args = cargo_args
        if args.auto_token_difficulty:
            if args.auto_token_provider == "playwright":
                try:
                    ws_url = fetch_ws_url_playwright(
                        python_bin=py,
                        difficulty=args.auto_token_difficulty,
                        state_path=args.auto_token_state_path,
                        app_url=args.auto_token_app_url,
                        timeout_ms=args.auto_token_timeout_ms,
                        headed=args.auto_token_headed,
                    )
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
                    rc = 3
                    break
            else:
                try:
                    ws_url = fetch_ws_url_http(
                        python_bin=py,
                        difficulty=args.auto_token_difficulty,
                        endpoint=args.auto_token_endpoint,
                        map_id=args.auto_token_map_id,
                        method=args.auto_token_http_method,
                        origin=args.auto_token_origin,
                        referer=args.auto_token_referer,
                        bearer_env=args.auto_token_bearer_env,
                        cookie_env=args.auto_token_cookie_env,
                        timeout_seconds=args.auto_token_timeout_seconds,
                        extra_headers=args.auto_token_header,
                    )
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
                    rc = 4
                    break
            run_args = [ws_url]
            print(
                f"[run {run_index + 1}/{args.repeat}] using auto-fetched ws URL "
                f"(provider={args.auto_token_provider})"
            )
        cmd = build_cmd(args.target, run_args)
        rc = subprocess.call(cmd, cwd=str(ROOT_DIR), env=env)
        if rc != 0:
            break
        successful_runs += 1
        if run_index + 1 < args.repeat:
            time.sleep(max(0.0, args.cooldown_seconds))

    if rc == 0 and successful_runs > 0 and not args.no_batch_train:
        if args.full_retrain:
            py_train = args.python_bin
            if not py_train:
                torch_venv = ROOT_DIR / ".venv311-torch" / "Scripts" / "python.exe"
                py_train = str(torch_venv) if torch_venv.exists() else sys.executable
            retrain_cmd = [
                py_train,
                str(ROOT_DIR / "tools" / "train_all_torch.py"),
                "--logs-dir",
                args.logs_dir,
                "--models-dir",
                str(ROOT_DIR / "models"),
                "--modes",
                args.train_modes,
                "--trainer-backend",
                "auto",
            ]
            retrain_rc = subprocess.call(retrain_cmd, cwd=str(ROOT_DIR), env=env)
            if retrain_rc != 0:
                print(f"Full retrain failed with exit code {retrain_rc}", file=sys.stderr)
                rc = retrain_rc
        else:
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
            batch_rc = subprocess.call(batch_cmd, cwd=str(ROOT_DIR), env=env)
            if batch_rc != 0:
                print(f"Batch training failed with exit code {batch_rc}", file=sys.stderr)
                rc = batch_rc
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
