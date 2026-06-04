"""Build an ordered execution packet for the gated-attention null-probe.

The freeze file says what must be tested. This file turns that freeze into a
runner-facing packet: commands, expected artifacts, local cache state, and
preflight gates. It intentionally does not execute the expensive Qwen3-Next
runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_FREEZE = Path("eval_results/gated_attention/gated_attention_null_probe_freeze.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/gated_attention/gated_attention_probe_execution_plan.json")
DEFAULT_REPORT_OUTPUT = Path("GATED_ATTENTION_PROBE_EXECUTION_PLAN.md")
DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

PRIMARY_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
BASELINE_MODEL = "Qwen/Qwen3-4B"
MECHANICS_MODEL = "Qwen/Qwen3-0.6B"

PRIMARY_RESULT_PATHS = {
    "position_shift": Path("eval_results/gated_attention/qwen3_next_position_shift_control_result.json"),
    "zero_prefix": Path("eval_results/gated_attention/qwen3_next_zero_prefix_result.json"),
    "random_prefix_n10": Path("eval_results/gated_attention/qwen3_next_random_prefix_n10_result.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--allow-existing-primary-results", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_execution_plan(
        freeze_path=args.freeze,
        hf_cache=args.hf_cache,
        allow_existing_primary_results=args.allow_existing_primary_results,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(plan), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "ready_for_primary_gpu_run": plan["ready_for_primary_gpu_run"],
                "blocking_reasons": plan["blocking_reasons"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_execution_plan(
    *,
    freeze_path: Path,
    hf_cache: Path,
    allow_existing_primary_results: bool = False,
) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    commands = dict(freeze["commands"])
    commands.setdefault("gated_zero_prefix", _zero_prefix_command())

    existing_primary_results = [
        str(path) for path in PRIMARY_RESULT_PATHS.values() if path.exists()
    ]
    model_cache = {
        "baseline": _model_cache_state(BASELINE_MODEL, hf_cache),
        "mechanics": _model_cache_state(MECHANICS_MODEL, hf_cache),
        "primary_gated": _model_cache_state(PRIMARY_MODEL, hf_cache),
    }

    blocking_reasons = []
    if existing_primary_results and not allow_existing_primary_results:
        blocking_reasons.append(
            "primary Qwen3-Next result artifacts already exist; pass "
            "--allow-existing-primary-results only for post-run audit rebuilds"
        )
    if not model_cache["primary_gated"]["cached"]:
        blocking_reasons.append(
            "primary Qwen3-Next model is not cached locally; select and record a "
            "current quantized artifact before starting the expensive run"
        )

    ordered_runs = [
        {
            "id": "mechanics_position_shift_smoke",
            "claim_status": "mechanics_only",
            "required_before_primary": False,
            "command": (
                "python -u experiments/run_latent_sensitivity.py --model Qwen/Qwen3-0.6B "
                "--task-type nested --difficulty sweet_spot --n-tasks 2 "
                "--control-mode position_shift --num-soft-tokens 2 --quantization 4bit "
                "--max-new-tokens 64 "
                "--output eval_results/gated_attention/qwen3_06b_position_shift_mechanics_smoke.json"
            ),
            "expected_artifact": "eval_results/gated_attention/qwen3_06b_position_shift_mechanics_smoke.json",
        },
        {
            "id": "primary_position_shift_control",
            "claim_status": "required_primary_control",
            "required_before_primary": True,
            "command": commands["gated_position_shift"],
            "expected_artifact": str(PRIMARY_RESULT_PATHS["position_shift"]),
        },
        {
            "id": "primary_zero_prefix_control",
            "claim_status": "required_primary_control",
            "required_before_primary": True,
            "command": commands["gated_zero_prefix"],
            "expected_artifact": str(PRIMARY_RESULT_PATHS["zero_prefix"]),
        },
        {
            "id": "primary_random_prefix_n10",
            "claim_status": "primary_test",
            "required_before_primary": True,
            "command": commands["gated_primary_random_prefix"],
            "expected_artifact": str(PRIMARY_RESULT_PATHS["random_prefix_n10"]),
        },
    ]

    return {
        "schema": "gated_attention_probe_execution_plan.v1",
        "generated_by": "experiments/build_gated_attention_probe_execution_plan.py",
        "freeze_path": str(freeze_path),
        "freeze_probe_id": freeze["probe_id"],
        "task_preset": freeze["task_preset"],
        "primary_model": PRIMARY_MODEL,
        "baseline_model": BASELINE_MODEL,
        "mechanics_model": MECHANICS_MODEL,
        "hf_cache": str(hf_cache),
        "model_cache": model_cache,
        "existing_primary_results": existing_primary_results,
        "allow_existing_primary_results": allow_existing_primary_results,
        "ready_for_primary_gpu_run": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "ordered_runs": ordered_runs,
        "reporting_order": [
            "mean last-integer accuracy",
            "EOS/completion rate",
            "true generated-token count",
            "strict final-answer accuracy",
            "answer-anywhere accuracy",
            "position-shift delta vs zero-prefix delta",
            "oracle coverage after mean metrics",
            "pairwise error overlap/correlation after oracle",
        ],
        "claim_boundaries": [
            "Do not treat mechanics smoke as gated-attention evidence.",
            "Do not interpret random-prefix lift without the position-shift and zero-prefix controls.",
            "Do not update article or README claim strength before mean-first report exists.",
            "Do not rebuild the pre-result freeze after primary results exist unless explicitly auditing.",
        ],
    }


def render_markdown(plan: dict[str, object]) -> str:
    lines = [
        "# Gated Attention Probe Execution Plan",
        "",
        "This file is generated by `experiments/build_gated_attention_probe_execution_plan.py`.",
        "",
        "## Status",
        "",
        f"- Freeze: `{plan['freeze_path']}`",
        f"- Probe ID: `{plan['freeze_probe_id']}`",
        f"- Primary model: `{plan['primary_model']}`",
        f"- Ready for primary GPU run: `{plan['ready_for_primary_gpu_run']}`",
        "",
    ]
    if plan["blocking_reasons"]:
        lines.extend(["Blocking reasons:", ""])
        lines.extend(f"- {reason}" for reason in plan["blocking_reasons"])
        lines.append("")

    lines.extend(["## Model Cache", "", "| Role | Model | Cached | Cache path |", "|---|---|---:|---|"])
    for role, state in plan["model_cache"].items():
        lines.append(
            f"| `{role}` | `{state['model']}` | `{state['cached']}` | `{state['path']}` |"
        )

    lines.extend(["", "## Ordered Runs", ""])
    for idx, run in enumerate(plan["ordered_runs"], start=1):
        lines.extend(
            [
                f"### {idx}. `{run['id']}`",
                "",
                f"- Claim status: `{run['claim_status']}`",
                f"- Expected artifact: `{run['expected_artifact']}`",
                "",
                "```powershell",
                run["command"],
                "```",
                "",
            ]
        )

    lines.extend(["## Reporting Order", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(plan["reporting_order"], start=1))
    lines.extend(["", "## Claim Boundaries", ""])
    lines.extend(f"- {item}" for item in plan["claim_boundaries"])
    lines.append("")
    return "\n".join(lines)


def _zero_prefix_command() -> str:
    return (
        "python -u experiments/run_latent_sensitivity.py "
        "--model Qwen/Qwen3-Next-80B-A3B-Instruct --task-type nested "
        "--difficulty sweet_spot --n-tasks 25 --n-latents 1 "
        "--control-mode zero_embedding --num-soft-tokens 2 --quantization 4bit "
        "--output eval_results/gated_attention/qwen3_next_zero_prefix_result.json"
    )


def _model_cache_state(model: str, hf_cache: Path) -> dict[str, object]:
    cache_name = "models--" + model.replace("/", "--")
    path = hf_cache / cache_name
    return {
        "model": model,
        "cached": path.exists(),
        "path": str(path),
        "snapshot_count": _snapshot_count(path),
    }


def _snapshot_count(path: Path) -> int:
    snapshots = path / "snapshots"
    if not snapshots.exists():
        return 0
    return sum(1 for child in snapshots.iterdir() if child.is_dir())


if __name__ == "__main__":
    raise SystemExit(main())
