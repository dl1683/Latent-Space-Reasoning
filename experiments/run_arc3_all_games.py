"""Run the local ARC-AGI-3 smoke harness across multiple games.

The script is intentionally backend-agnostic: use it for cheap baselines,
learned controllers, or latent runs. It writes one smoke manifest per game and
an aggregate summary for quick comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GAMES = [
    "ar25",
    "bp35",
    "cd82",
    "cn04",
    "dc22",
    "ft09",
    "g50t",
    "ka59",
    "lf52",
    "lp85",
    "ls20",
    "m0r0",
    "r11l",
    "re86",
    "s5i5",
    "sb26",
    "sc25",
    "sk48",
    "sp80",
    "su15",
    "tn36",
    "tr87",
    "tu93",
    "vc33",
    "wa30",
]


def _load_root_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _extract_scorecard(stdout: str) -> dict[str, Any]:
    marker = "--- FINAL SCORECARD REPORT ---"
    marker_index = stdout.rfind(marker)
    if marker_index < 0:
        return {}
    start = stdout.find("{", marker_index)
    if start < 0:
        return {}
    text = stdout[start:].strip()
    text = "\n".join(
        line.split("|", 2)[-1].strip() if " | " in line else line
        for line in text.splitlines()
    )
    try:
        scorecard, _end = json.JSONDecoder().raw_decode(text)
        return scorecard if isinstance(scorecard, dict) else {}
    except json.JSONDecodeError:
        return {}


def _run_game(args: argparse.Namespace, game: str, output_dir: Path) -> dict[str, Any]:
    prefix = output_dir / f"{game}_{args.server_backend}"
    command = [
        sys.executable,
        "experiments/run_arc3_local_latent_smoke.py",
        "--game-id",
        game,
        "--server-backend",
        args.server_backend,
        "--tags",
        f"{args.tags},{game}",
        "--harness-output",
        str(prefix.with_suffix(".harness.json")),
        "--server-log",
        str(prefix.with_suffix(".server.log")),
        "--trace-jsonl",
        str(prefix.with_suffix(".trace.jsonl")),
        "--output",
        str(prefix.with_suffix(".smoke.json")),
        "--server-ready-timeout-s",
        str(args.server_ready_timeout_s),
        "--state-probe-repeat-cap",
        str(args.state_probe_repeat_cap),
        "--max-latent-calls",
        str(args.max_latent_calls),
        "--chains",
        str(args.chains),
        "--generations",
        str(args.generations),
        "--max-tokens",
        str(args.max_tokens),
        "--ollama-model",
        args.ollama_model,
        "--ollama-url",
        args.ollama_url,
        "--ollama-timeout-s",
        str(args.ollama_timeout_s),
        "--fallback-policy",
        args.fallback_policy,
        "--max-gpu-utilization",
        str(args.max_gpu_utilization),
        "--max-gpu-memory-used-mb",
        str(args.max_gpu_memory_used_mb),
    ]
    if args.executable_search_plan:
        command.append("--executable-search-plan")
    command.extend(["--executable-search-max-levels", str(args.executable_search_max_levels)])
    if args.learned_trace:
        command.extend(["--learned-trace", args.learned_trace])
    command.extend(["--learned-policy-k", str(args.learned_policy_k)])
    command.extend(["--learned-max-train-level", str(args.learned_max_train_level)])
    if args.learned_sequence_backoff:
        command.append("--learned-sequence-backoff")
    if args.learned_phase_switch:
        command.append("--learned-phase-switch")
    if args.learned_goal_seek:
        command.append("--learned-goal-seek")
    if args.wait_for_gpu:
        command.append("--wait-for-gpu")

    env = None
    root_env = _load_root_env()
    if root_env:
        env = {**__import__("os").environ, **root_env}
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        env=env,
        timeout=args.per_game_timeout_s,
    )
    summary: dict[str, Any] = {
        "game": game,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "smoke_manifest": str(prefix.with_suffix(".smoke.json")),
    }
    smoke_path = prefix.with_suffix(".smoke.json")
    if smoke_path.exists():
        try:
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            harness = smoke.get("harness") if isinstance(smoke.get("harness"), dict) else {}
            harness_path = Path(str(smoke.get("harness_output", "")))
            harness_manifest = {}
            if harness_path.exists():
                harness_manifest = json.loads(harness_path.read_text(encoding="utf-8"))
            scorecard = _extract_scorecard(str(harness.get("stdout", "")))
            if not scorecard:
                scorecard = _extract_scorecard(str(harness_manifest.get("stdout", "")))
            envs = scorecard.get("environments") if isinstance(scorecard.get("environments"), list) else []
            env = envs[0] if envs and isinstance(envs[0], dict) else {}
            summary["score"] = scorecard.get("score", env.get("score"))
            summary["levels_completed"] = scorecard.get("total_levels_completed", env.get("levels_completed"))
            summary["total_levels"] = scorecard.get("total_levels", env.get("level_count"))
            summary["total_actions"] = scorecard.get("total_actions", env.get("actions"))
            summary["completed"] = env.get("completed")
            summary["harness_returncode"] = harness.get("returncode")
        except json.JSONDecodeError as exc:
            summary["smoke_error"] = str(exc)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=DEFAULT_GAMES)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/arc3_all_games"))
    parser.add_argument("--server-backend", default="state_probe")
    parser.add_argument("--tags", default="all-games")
    parser.add_argument("--learned-trace", default="")
    parser.add_argument("--learned-policy-k", type=int, default=1)
    parser.add_argument("--learned-max-train-level", type=int, default=-1)
    parser.add_argument("--learned-sequence-backoff", action="store_true")
    parser.add_argument("--learned-phase-switch", action="store_true")
    parser.add_argument("--learned-goal-seek", action="store_true")
    parser.add_argument("--state-probe-repeat-cap", type=int, default=8)
    parser.add_argument("--max-latent-calls", type=int, default=-1)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--ollama-model", default="mistral:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout-s", type=float, default=12.0)
    parser.add_argument("--fallback-policy", default="state_probe")
    parser.add_argument("--executable-search-plan", action="store_true")
    parser.add_argument("--executable-search-max-levels", type=int, default=2)
    parser.add_argument("--server-ready-timeout-s", type=float, default=120.0)
    parser.add_argument("--per-game-timeout-s", type=float, default=600.0)
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=float, default=35.0)
    parser.add_argument("--max-gpu-memory-used-mb", type=float, default=12000.0)
    parser.add_argument("--output", type=Path, default=Path("eval_results/arc3_all_games_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for game in args.games:
        rows.append(_run_game(args, game, args.output_dir))
        args.output.write_text(
            json.dumps(
                {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "server_backend": args.server_backend,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(rows[-1], indent=2))


if __name__ == "__main__":
    main()
