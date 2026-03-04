from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_APP_URL = "https://app.ainm.no/challenge"
WS_URL_RE = re.compile(r"wss://game\.ainm\.no/ws\?token=[A-Za-z0-9._~-]+")
PLAY_RE = re.compile(r"play", re.IGNORECASE)


def extract_ws_url(text: str) -> str | None:
    match = WS_URL_RE.search(text)
    if match:
        return match.group(0)
    return None


def find_ws_url_in_obj(obj: Any) -> str | None:
    if isinstance(obj, str):
        return extract_ws_url(obj)
    if isinstance(obj, dict):
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


def click_play_button(page: Any, difficulty: str) -> None:
    difficulty_re = re.compile(rf"\b{re.escape(difficulty)}\b", re.IGNORECASE)
    container_selectors = ["article", "section", "div"]
    for selector in container_selectors:
        scoped = (
            page.locator(selector)
            .filter(has_text=difficulty_re)
            .get_by_role("button", name=PLAY_RE)
        )
        if scoped.count() > 0:
            scoped.first.click()
            return

    play_buttons = page.get_by_role("button", name=PLAY_RE)
    count = play_buttons.count()
    if count <= 0:
        raise RuntimeError(
            "No Play button found. Login may be required. "
            "Run once with --headed and complete login in the opened browser."
        )
    fallback_order = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
    index = min(fallback_order[difficulty], count - 1)
    play_buttons.nth(index).click()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a fresh game websocket URL by clicking Play in a browser session."
    )
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "expert"], required=True)
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument(
        "--state-path",
        default=str(ROOT_DIR / ".secrets" / "ainm_storage_state.json"),
        help="Playwright storage-state JSON path for persistent login session.",
    )
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - dependency gate
        print(
            "Playwright is required. Install with:\n"
            "  python -m pip install playwright\n"
            "  python -m playwright install chromium",
            file=sys.stderr,
        )
        raise SystemExit(f"missing playwright dependency: {exc}")

    state_path = Path(args.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    captured: dict[str, str] = {}

    def remember(source: str, candidate: str | None) -> None:
        if candidate:
            captured["source"] = source
            captured["ws_url"] = candidate
            if args.verbose:
                print(f"[fetch] captured from {source}: {candidate}", file=sys.stderr)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context_kwargs = {}
        if state_path.exists():
            context_kwargs["storage_state"] = str(state_path)
        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        def on_request(req: Any) -> None:
            remember("request_url", extract_ws_url(req.url))

        def on_response(resp: Any) -> None:
            if "ws?token=" in resp.url:
                remember("response_url", extract_ws_url(resp.url))
                return
            if "json" not in (resp.headers.get("content-type") or "").lower():
                return
            try:
                text = resp.text()
            except Exception:
                return
            remember("response_text", extract_ws_url(text))
            try:
                payload = json.loads(text)
            except Exception:
                return
            remember("response_json", find_ws_url_in_obj(payload))

        page.on("request", on_request)
        page.on("response", on_response)

        page.goto(args.app_url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=min(args.timeout_ms, 12_000))
        except PlaywrightTimeoutError:
            pass

        click_play_button(page, args.difficulty)

        deadline = time.time() + max(2.0, args.timeout_ms / 1000.0)
        while time.time() < deadline:
            remember("page_url", extract_ws_url(page.url))
            if "ws_url" in captured:
                break
            try:
                body_text = page.content()
                remember("page_content", extract_ws_url(body_text))
            except Exception:
                pass
            page.wait_for_timeout(150)

        context.storage_state(path=str(state_path))
        browser.close()

    ws_url = captured.get("ws_url")
    if not ws_url:
        raise SystemExit(
            "Failed to capture ws URL. Run with --headed once and make sure your browser session is logged in."
        )
    print(ws_url)


if __name__ == "__main__":
    main()
