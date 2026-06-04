"""Run the official ARC-AGI-3 interactive benchmark harness.

ARC-AGI-3 is an interactive agent benchmark, not the static grid format used by
ARC-AGI-1/2. This wrapper keeps official-harness runs reproducible from this
repo while the latent-reasoning agent integration is developed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List


OFFICIAL_REPO = "https://github.com/arcprize/arc-agi-3-benchmarking.git"
OFFICIAL_DOCS = "https://docs.arcprize.org/benchmarking-agent"
LOCAL_LATENT_CONFIG_ID = "local-latent-reasoning"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_repo_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _run_command(
    command: List[str],
    cwd: Path | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    if dry_run:
        return None
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


def _ensure_harness(args: argparse.Namespace) -> None:
    harness_dir = Path(args.harness_dir)
    if (harness_dir / "pyproject.toml").exists():
        return
    if not args.clone_if_missing:
        raise FileNotFoundError(
            f"ARC-AGI-3 harness not found at {harness_dir}. "
            "Use --clone-if-missing or pass --harness-dir."
        )

    harness_dir.parent.mkdir(parents=True, exist_ok=True)
    result = _run_command(
        ["git", "clone", "--depth", "1", OFFICIAL_REPO, str(harness_dir)],
        dry_run=args.dry_run,
    )
    if result is not None and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def _build_arcagi3_command(args: argparse.Namespace) -> List[str]:
    command = ["uv", "run", "main.py"]

    if args.list_games:
        command.append("--list-games")
        return command
    if args.list_configs:
        command.append("--list-configs")
        return command

    if args.game_id:
        command.extend(["--game", args.game_id])
    if args.config:
        command.extend(["--config", args.config])
    if args.tags:
        command.extend(["--tags", args.tags])
    return command


def _run_preflight(args: argparse.Namespace) -> None:
    commands = {
        "games": ["uv", "run", "main.py", "--list-games"],
        "configs": ["uv", "run", "main.py", "--list-configs"],
    }
    results: dict[str, Any] = {}
    for name, command in commands.items():
        result = _run_command(command, cwd=Path(args.harness_dir), dry_run=args.dry_run)
        results[name] = {
            "command": command,
            "returncode": result.returncode if result is not None else None,
            "stdout": result.stdout if result is not None else "",
            "stderr": result.stderr if result is not None else "",
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_repo": OFFICIAL_REPO,
        "official_docs": OFFICIAL_DOCS,
        "harness_dir": str(Path(args.harness_dir).resolve()),
        "dry_run": args.dry_run,
        "arc_api_key_present": bool(os.environ.get("ARC_API_KEY")),
        "local_latent_config_id": LOCAL_LATENT_CONFIG_ID,
        "local_latent_base_url": args.local_latent_base_url,
        "checks": results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Preflight manifest:", args.output)
    failed = [
        name
        for name, result in results.items()
        if result["returncode"] not in (0, None)
    ]
    if failed:
        raise SystemExit(1)


def _install_local_latent_config(args: argparse.Namespace) -> None:
    config_path = Path(args.harness_dir) / "benchmarking" / "model_configs.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")

    entry = f"""

- id: "{LOCAL_LATENT_CONFIG_ID}"
  agent:
    MAX_ACTIONS_BASELINE_MULTIPLIER: 3.0
    MAX_CONTEXT_LENGTH: 64_000
    analysis_mode: true
  runtime:
    sdk: "openai-python"
    api: "chat_completions"
    state: "manual_rolling"
  client:
    base_url: "{args.local_latent_base_url}"
    api_key_env: "LOCAL_LATENT_API_KEY"
  request:
    model: "{LOCAL_LATENT_CONFIG_ID}"
    max_completion_tokens: 512
    temperature: 0
  pricing: {{}}
"""
    marker = f'- id: "{LOCAL_LATENT_CONFIG_ID}"'
    if marker not in text:
        config_path.write_text(text.rstrip() + entry, encoding="utf-8")
        return

    start = text.index(marker)
    next_start = text.find("\n- id: ", start + len(marker))
    replacement = entry.lstrip("\n")
    if next_start == -1:
        updated = text[:start].rstrip() + "\n\n" + replacement
    else:
        updated = text[:start].rstrip() + "\n\n" + replacement.rstrip() + text[next_start:]
    config_path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def _write_manifest(
    args: argparse.Namespace,
    command: List[str],
    result: subprocess.CompletedProcess[str] | None,
) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_repo": OFFICIAL_REPO,
        "official_docs": OFFICIAL_DOCS,
        "harness_dir": str(Path(args.harness_dir).resolve()),
        "command": command,
        "dry_run": args.dry_run,
        "arc_api_key_present": bool(os.environ.get("ARC_API_KEY")),
        "local_latent_config_id": LOCAL_LATENT_CONFIG_ID,
        "local_latent_base_url": args.local_latent_base_url,
    }
    if result is not None:
        payload.update(
            {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness-dir", default="external/arc-agi-3-benchmarking")
    parser.add_argument("--clone-if-missing", action="store_true")
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument("--config", default="")
    parser.add_argument("--tags", default="")
    parser.add_argument("--list-games", action="store_true")
    parser.add_argument("--list-configs", "--list-models", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--install-local-latent-config", action="store_true")
    parser.add_argument("--local-latent-base-url", default="http://127.0.0.1:8013/v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="eval_results/arc3_official_harness_run.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_repo_env()
    _ensure_harness(args)
    if args.install_local_latent_config and not args.dry_run:
        _install_local_latent_config(args)
    if args.preflight:
        _run_preflight(args)
        return
    command = _build_arcagi3_command(args)
    result = _run_command(command, cwd=Path(args.harness_dir), dry_run=args.dry_run)
    _write_manifest(args, command, result)

    print("Command:", " ".join(command))
    print("Manifest:", args.output)
    if result is not None:
        print("Return code:", result.returncode)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
