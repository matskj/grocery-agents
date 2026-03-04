from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


WS_URL_RE = re.compile(r"wss://game\.ainm\.no/ws\?token=[A-Za-z0-9._~-]+")
DEFAULT_MAP_ID_BY_DIFFICULTY = {
    "easy": "c89da2ec-3ca7-40c9-a3b1-8036fca3d0b7",
    "medium": "3c523f5e-160b-452c-9ffc-171ef1e845f5",
    "hard": "05ddc283-9097-4314-824c-90b3269a3d95",
    "expert": "c7c7f564-2496-4ab1-9179-7532979adcb4",
}


def extract_ws_url(text: str) -> str | None:
    match = WS_URL_RE.search(text)
    if match:
        return match.group(0)
    return None


def find_ws_url_in_obj(obj: Any) -> str | None:
    if isinstance(obj, str):
        return extract_ws_url(obj)
    if isinstance(obj, dict):
        if "ws_url" in obj and isinstance(obj["ws_url"], str):
            candidate = extract_ws_url(obj["ws_url"])
            if candidate:
                return candidate
        for value in obj.values():
            found = find_ws_url_in_obj(value)
            if found:
                return found
        return None
    if isinstance(obj, list):
        for value in obj:
            found = find_ws_url_in_obj(value)
            if found:
                return found
        return None
    return None


def parse_header_line(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"invalid header (expected 'Key: Value'): {value}")
    key, raw = value.split(":", 1)
    key = key.strip()
    val = raw.strip()
    if not key:
        raise ValueError(f"invalid header key: {value}")
    return key, val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a fresh ws URL by calling the HTTP play endpoint."
    )
    parser.add_argument("--endpoint", default=os.getenv("AINM_PLAY_ENDPOINT"))
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "expert"], required=True)
    parser.add_argument("--map-id", default=None)
    parser.add_argument("--method", choices=["POST", "GET"], default="POST")
    parser.add_argument("--origin", default="https://app.ainm.no")
    parser.add_argument("--referer", default="https://app.ainm.no/challenge")
    parser.add_argument("--bearer-token", default=None)
    parser.add_argument("--bearer-env", default="AINM_ACCESS_TOKEN")
    parser.add_argument("--cookie", default=None)
    parser.add_argument("--cookie-env", default="AINM_COOKIE")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    endpoint = (args.endpoint or "").strip()
    if not endpoint:
        raise SystemExit(
            "Missing --endpoint (or AINM_PLAY_ENDPOINT env). "
            "Use DevTools request URL for Generate/Play."
        )

    map_id = args.map_id or DEFAULT_MAP_ID_BY_DIFFICULTY.get(args.difficulty)
    if not map_id:
        raise SystemExit("Missing map id. Provide --map-id explicitly.")

    bearer = args.bearer_token or os.getenv(args.bearer_env)
    cookie = args.cookie or os.getenv(args.cookie_env)

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": args.origin,
        "Referer": args.referer,
    }
    if bearer:
        auth_value = bearer if bearer.lower().startswith("bearer ") else f"Bearer {bearer}"
        headers["Authorization"] = auth_value
    if cookie:
        headers["Cookie"] = cookie
    for raw in args.header:
        key, value = parse_header_line(raw)
        headers[key] = value

    payload = {"map_id": map_id}
    data = None if args.method == "GET" else json.dumps(payload).encode("utf-8")
    url = endpoint
    if args.method == "GET":
        joiner = "&" if "?" in endpoint else "?"
        url = f"{endpoint}{joiner}map_id={map_id}"

    req = Request(url=url, data=data, method=args.method, headers=headers)
    try:
        with urlopen(req, timeout=max(1.0, args.timeout_seconds)) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        raise SystemExit(
            f"HTTP token fetch failed: {exc.code} {exc.reason}\n"
            f"endpoint: {endpoint}\n"
            f"response: {detail}"
        )
    except URLError as exc:
        raise SystemExit(f"HTTP token fetch failed: {exc}")

    ws_url = extract_ws_url(body)
    if not ws_url:
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = None
        if parsed is not None:
            ws_url = find_ws_url_in_obj(parsed)

    if not ws_url:
        if args.verbose:
            print(body, file=sys.stderr)
        raise SystemExit("No ws_url found in play endpoint response.")

    print(ws_url)


if __name__ == "__main__":
    main()
