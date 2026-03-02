from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT_DIR = Path(__file__).resolve().parent.parent
REPLAY_SERVER = ROOT_DIR / "tools" / "replay_server.py"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8085


def server_alive(host: str, port: int, timeout: float = 3.0) -> bool:
    # /api/runs can be slow when many logs exist; /api/latest is much cheaper.
    url = f"http://{host}:{port}/api/latest"
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except URLError:
        return False
    except Exception:
        return False


def spawn_server(host: str, port: int, logs_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(REPLAY_SERVER),
        "--host",
        host,
        "--port",
        str(port),
        "--logs-dir",
        str(logs_dir),
    ]
    if sys.platform.startswith("win"):
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
    else:
        subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )


def ensure_server(host: str, port: int, logs_dir: Path) -> bool:
    if server_alive(host, port):
        return True

    spawn_server(host, port, logs_dir)
    for _ in range(20):
        time.sleep(0.15)
        if server_alive(host, port):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure local replay server is running.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--logs-dir", default=str(ROOT_DIR / "logs"))
    args = parser.parse_args()

    ok = ensure_server(args.host, args.port, Path(args.logs_dir))
    if ok:
        print(f"Replay UI ready at http://{args.host}:{args.port}")
        raise SystemExit(0)
    print(f"Failed to start replay UI at http://{args.host}:{args.port}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
