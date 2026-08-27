"""Download public ARC-AGI-3 replay JSONL files.

The public replay UI resolves a session with:
  /api/sessions/<session_id>
then downloads the raw JSONL with:
  /api/recordings/<game_id>/<session_id>
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


REPLAY_RE = re.compile(r"https?://arcprize\.org/replay/([0-9a-fA-F-]{36})|/replay/([0-9a-fA-F-]{36})")


def _fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "latent-space-reasoning-arc3-replay-downloader/1.0"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - public ARC endpoint.
        return response.read()


def _fetch_json(url: str) -> dict[str, Any]:
    return json.loads(_fetch_bytes(url).decode("utf-8-sig"))


def _session_ids_from_text(text: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for match in REPLAY_RE.finditer(text):
        session_id = match.group(1) or match.group(2)
        if session_id and session_id not in seen:
            ids.append(session_id)
            seen.add(session_id)
    return ids


def _game_id_from_session(session: dict[str, Any]) -> str:
    environments = session.get("environments")
    if isinstance(environments, list):
        for env in environments:
            if isinstance(env, dict) and isinstance(env.get("id"), str) and env["id"].strip():
                return env["id"].strip()
    game_id = session.get("game_id")
    if isinstance(game_id, str) and game_id.strip():
        return game_id.strip()
    raise ValueError("Could not resolve game_id from session metadata")


def _download_one(base_url: str, output_root: Path, session_id: str) -> dict[str, Any]:
    session_url = f"{base_url}/api/sessions/{quote(session_id)}"
    session = _fetch_json(session_url)
    game_id = _game_id_from_session(session)
    game_slug = game_id.split("-", 1)[0]
    replay_url = f"{base_url}/api/recordings/{quote(game_id)}/{quote(session_id)}"
    recording = _fetch_bytes(replay_url)

    replay_dir = output_root / "replays" / game_slug
    replay_dir.mkdir(parents=True, exist_ok=True)
    recording_path = replay_dir / f"{session_id}.jsonl"
    session_path = replay_dir / f"{session_id}.session.json"
    recording_path.write_bytes(recording)
    session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    return {
        "session_id": session_id,
        "game_id": game_id,
        "game_slug": game_slug,
        "session_url": session_url,
        "recording_url": replay_url,
        "recording_path": str(recording_path),
        "session_path": str(session_path),
        "bytes": len(recording),
        "score": session.get("score"),
        "total_levels_completed": session.get("total_levels_completed"),
        "total_actions": session.get("total_actions"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument("--replay-url", action="append", default=[])
    parser.add_argument("--source-html", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("external/arc-agi-3-human-baseline"))
    parser.add_argument("--base-url", default="https://arcprize.org")
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_ids: list[str] = []
    for value in args.session_id:
        session_ids.append(value.strip())
    for value in args.replay_url:
        session_ids.extend(_session_ids_from_text(value))
    if args.source_html:
        session_ids.extend(_session_ids_from_text(args.source_html.read_text(encoding="utf-8-sig")))

    unique_ids: list[str] = []
    seen: set[str] = set()
    for session_id in session_ids:
        if session_id and session_id not in seen:
            unique_ids.append(session_id)
            seen.add(session_id)
    if args.limit >= 0:
        unique_ids = unique_ids[: args.limit]
    if not unique_ids:
        raise SystemExit("No replay session ids provided or discovered")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    base_url = args.base_url.rstrip("/")
    for session_id in unique_ids:
        try:
            rows.append(_download_one(base_url, args.output_root, session_id))
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            failures.append({"session_id": session_id, "error": str(exc)})

    index = {
        "source": "ARC Prize public replay API",
        "base_url": base_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "requested_sessions": unique_ids,
        "downloaded": rows,
        "failures": failures,
    }
    index_path = args.output_root / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"ARC-3 human replay download index: {index_path}")
    print(json.dumps({"downloaded": len(rows), "failures": len(failures), "games": sorted({r['game_slug'] for r in rows})}, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
