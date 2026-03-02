from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse


ROOT_DIR = Path(__file__).resolve().parent.parent
VIEWER_DIR = ROOT_DIR / "viewer"
DEFAULT_LOG_DIR = ROOT_DIR / "logs"


def parse_jsonl(path: Path) -> Dict:
    run_id = path.stem
    session_start = None
    game_mode = None
    game_over = None
    ticks: List[Dict] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = record.get("event")
            data = record.get("data") or {}
            if isinstance(data, dict):
                run_id = str(record.get("run_id") or data.get("run_id") or run_id)

            if event == "session_start" and isinstance(data, dict):
                session_start = data
            elif event == "game_mode" and isinstance(data, dict):
                game_mode = data
            elif event == "game_over" and isinstance(data, dict):
                game_over = data
            elif event == "tick" and isinstance(data, dict):
                tick_value = data.get("tick")
                if tick_value is None:
                    tick_value = (data.get("game_state") or {}).get("tick")
                ticks.append(
                    {
                        "tick": tick_value,
                        "mode": data.get("mode") or data.get("game_mode"),
                        "game_state": data.get("game_state"),
                        "actions": data.get("actions") or [],
                        "team_summary": data.get("team_summary") or {},
                        "tick_outcome": data.get("tick_outcome") or {},
                    }
                )

    return {
        "run_id": run_id,
        "file": path.name,
        "session_start": session_start,
        "game_mode": game_mode,
        "game_over": game_over,
        "tick_count": len(ticks),
        "ticks": ticks,
    }


def summarize_run(path: Path) -> Dict:
    result = {
        "file": path.name,
        "size": path.stat().st_size,
        "mtime_ms": int(path.stat().st_mtime * 1000),
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "mode": None,
        "final_score": None,
        "run_id": path.stem,
    }
    session_mode = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            data = record.get("data") or {}
            if not isinstance(data, dict):
                continue
            result["run_id"] = str(record.get("run_id") or data.get("run_id") or result["run_id"])
            if event == "session_start":
                session_mode = data.get("mode")
            elif event == "game_mode":
                result["mode"] = data.get("mode")
            elif event == "game_over":
                result["final_score"] = data.get("final_score")
    if not result["mode"]:
        result["mode"] = session_mode
    return result


class ReplayHandler(SimpleHTTPRequestHandler):
    log_dir: Path

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/runs":
            self.handle_runs()
            return
        if path == "/api/latest":
            self.handle_latest()
            return
        if path.startswith("/api/run/"):
            run_name = unquote(path[len("/api/run/") :])
            self.handle_run(run_name)
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def handle_runs(self) -> None:
        runs = []
        for candidate in sorted(
            self.log_dir.glob("run-*.jsonl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            runs.append(summarize_run(candidate))
        payload = {"runs": runs, "count": len(runs)}
        self.send_json(payload)

    def handle_latest(self) -> None:
        candidates = sorted(
            self.log_dir.glob("run-*.jsonl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            self.send_json({"run": None})
            return
        self.send_json({"run": summarize_run(candidates[0])})

    def handle_run(self, run_name: str) -> None:
        if "/" in run_name or "\\" in run_name:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid run name")
            return
        path = self.log_dir / run_name
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "run file not found")
            return
        payload = parse_jsonl(path)
        self.send_json(payload)

    def send_json(self, payload: Dict) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def run_server(host: str, port: int, log_dir: Path) -> None:
    if not VIEWER_DIR.exists():
        raise SystemExit(f"viewer directory not found: {VIEWER_DIR}")
    if not log_dir.exists():
        raise SystemExit(f"log directory not found: {log_dir}")

    class Handler(ReplayHandler):
        pass

    Handler.log_dir = log_dir.resolve()
    server = ThreadingHTTPServer((host, port), lambda *a, **k: Handler(*a, directory=str(VIEWER_DIR), **k))

    print(f"Replay server listening on http://{host}:{port}")
    print(f"Serving logs from: {log_dir.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local run-replay UI for grocery-agents logs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOG_DIR))
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    run_server(args.host, args.port, logs_dir)


if __name__ == "__main__":
    main()

