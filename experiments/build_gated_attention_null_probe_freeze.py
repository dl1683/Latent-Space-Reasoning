"""Build the frozen gated-attention null-probe proof obligation.

This is a pre-result artifact. It freezes the architecture-transfer test before
any Qwen3-Next labels or outputs are generated, so later claims cannot drift
toward whichever result is more convenient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_JSON_OUTPUT = Path("eval_results/gated_attention/gated_attention_null_probe_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("GATED_ATTENTION_NULL_PROBE_FREEZE.md")
DEFAULT_PRIOR_RESULTS = (
    Path("experiments/sensitivity_sweet_spot_random_noise_t2_results.json"),
    Path("experiments/sensitivity_sweet_spot_random_noise_t2_qwen38b_8bit_n10_results.json"),
    Path("experiments/planning_bp_2048_results.json"),
)
DEFAULT_RESULT_GLOB = "eval_results/gated_attention/*result*.json"

FROZEN_PROBE_ID = "gated_attention_null_probe_v1"
FROZEN_TASK_PRESET = "lsr_25_arithmetic_plus_cache_debug"
FROZEN_RANDOM_PREFIX_SEEDS = tuple(range(10))
FROZEN_PLANNING_SEEDS = tuple(range(5))
FROZEN_SOFT_TOKENS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument(
        "--allow-existing-results",
        action="store_true",
        help="Allow building after gated-attention result artifacts exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        result_glob=DEFAULT_RESULT_GLOB,
        allow_existing_results=args.allow_existing_results,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "probe_id": manifest["probe_id"],
                "report_output": str(args.report_output),
                "task_preset": manifest["task_preset"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(*, result_glob: str, allow_existing_results: bool = False) -> dict[str, object]:
    existing_results = sorted(str(path) for path in Path(".").glob(result_glob))
    if existing_results and not allow_existing_results:
        raise ValueError(
            "refusing gated-attention freeze after result artifacts exist: "
            + ", ".join(existing_results)
        )

    prior_artifacts = []
    for path in DEFAULT_PRIOR_RESULTS:
        prior_artifacts.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": _sha256(path) if path.exists() else None,
            }
        )

    return {
        "schema": "gated_attention_null_probe_freeze.v1",
        "generated_by": "experiments/build_gated_attention_null_probe_freeze.py",
        "probe_id": FROZEN_PROBE_ID,
        "task_preset": FROZEN_TASK_PRESET,
        "design_intent": (
            "Test whether the LSR random-prefix effect survives an architecture that uses "
            "gated-attention / Qwen3-Next-style layers instead of relying on ordinary "
            "attention sinks."
        ),
        "pre_result_boundary": {
            "existing_result_glob": result_glob,
            "existing_results": existing_results,
            "allow_existing_results": allow_existing_results,
            "no_gated_attention_outputs_seen": not existing_results,
        },
        "model_arms": {
            "baseline_family_matched": {
                "role": "non_gated_qwen_baseline",
                "model": "Qwen/Qwen3-4B",
                "quantization": "4bit_nf4_or_existing_repo_default",
                "source_status": "existing_repo_baseline",
            },
            "gated_primary": {
                "role": "gated_attention_transfer_candidate",
                "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
                "quantization": "GGUF_Q4_or_FP8_with_CPU_offload_if_needed",
                "source_status": "must_verify_snapshot_before_gpu_run",
                "architecture_requirement": "hybrid Gated DeltaNet plus Gated Attention",
            },
            "gated_small_control": {
                "role": "cheap_mechanics_control_if_primary_is_too_slow",
                "model": "QwQZh/gated_attention 1B gate/head variants or another documented gated-attention checkpoint",
                "quantization": "native_or_8bit",
                "source_status": "mechanics_only_not_claim_surface",
            },
        },
        "conditions": [
            {
                "id": "baseline_greedy",
                "soft_prompt": "none",
                "position_ids": "standard",
                "decode": "greedy_temperature_0",
            },
            {
                "id": "zero_prefix",
                "soft_prompt": "2 zero-valued embedding tokens",
                "position_ids": "standard_after_prefix",
                "decode": "greedy_temperature_0",
            },
            {
                "id": "random_prefix_n10",
                "soft_prompt": "2 RMS-matched random embedding tokens",
                "seeds": list(FROZEN_RANDOM_PREFIX_SEEDS),
                "position_ids": "standard_after_prefix",
                "decode": "greedy_temperature_0",
            },
            {
                "id": "position_shift_control",
                "soft_prompt": "none",
                "position_ids": "start_at_2_without_extra_embeddings",
                "decode": "greedy_temperature_0",
            },
        ],
        "planning_conditions": [
            {
                "id": "cache_debug_baseline",
                "soft_prompt": "none",
                "max_new_tokens": 2048,
                "decode": "greedy_temperature_0",
            },
            {
                "id": "cache_debug_random_prefix_n5",
                "soft_prompt": "2 RMS-matched random embedding tokens",
                "seeds": list(FROZEN_PLANNING_SEEDS),
                "max_new_tokens": 2048,
                "decode": "greedy_temperature_0",
            },
        ],
        "metrics": {
            "primary_report_first": [
                "mean last-integer accuracy by condition",
                "EOS/completion rate by condition",
                "true generated-token count from output_ids and prompt length",
                "strict final-answer accuracy",
                "answer-anywhere accuracy",
                "position-shift delta vs zero-prefix delta",
            ],
            "secondary_report_after_mean": [
                "oracle coverage at N=10",
                "pairwise error overlap/correlation",
                "planning rescue word count and judge score",
                "attention sink mass if output_attentions is exposed",
                "attention entropy if output_attentions is exposed",
            ],
        },
        "infrastructure_gates": {
            "inputs_embeds_token_count_bug_fixed": True,
            "full_raw_outputs_required": True,
            "path_equivalence_smoke_required": True,
            "reuse_exact_25_task_arithmetic_set": True,
            "include_cache_debugging_task": True,
            "no_article_claim_without_heldout_or_architecture_transfer_result": True,
            "source_snapshot_must_be_recorded": True,
        },
        "interpretation_gates": {
            "sink_dependent": {
                "oracle_coverage_lt": 0.60,
                "mean_lift_pp_lt": 5.0,
                "planning_rescue": False,
            },
            "trajectory_diversification_persists": {
                "oracle_coverage_gt": 0.80,
                "or_mean_lift_pp_gt": 10.0,
            },
            "ambiguous": {
                "oracle_coverage_range": [0.60, 0.80],
                "mean_lift_pp_range": [5.0, 10.0],
                "action": "extend tasks and run dose-response before changing the article",
            },
        },
        "commands": {
            "baseline_replay": (
                "python -u experiments/run_latent_sensitivity.py --model Qwen/Qwen3-4B "
                "--task-type nested --difficulty sweet_spot --n-tasks 25 --n-latents 10 "
                "--control-mode random_noise --num-soft-tokens 2 --quantization 4bit"
            ),
            "gated_primary_random_prefix": (
                "python -u experiments/run_latent_sensitivity.py "
                "--model Qwen/Qwen3-Next-80B-A3B-Instruct --task-type nested "
                "--difficulty sweet_spot --n-tasks 25 --n-latents 10 "
                "--control-mode random_noise --num-soft-tokens 2 --quantization 4bit "
                "--output eval_results/gated_attention/qwen3_next_random_prefix_n10_result.json"
            ),
            "gated_position_shift": (
                "python -u experiments/run_latent_sensitivity.py "
                "--model Qwen/Qwen3-Next-80B-A3B-Instruct --task-type nested "
                "--difficulty sweet_spot --n-tasks 25 --control-mode position_shift "
                "--num-soft-tokens 2 --quantization 4bit "
                "--output eval_results/gated_attention/qwen3_next_position_shift_control_result.json"
            ),
        },
        "prior_artifacts": prior_artifacts,
        "external_sources_checked": [
            {
                "url": "https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct",
                "note": "Current primary model card exists; exact quantized runtime must be rechecked before GPU run.",
            },
            {
                "url": "https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF",
                "note": "Current GGUF model card exists; selected quant file must be recorded in result metadata.",
            },
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    arms = manifest["model_arms"]
    gates = manifest["interpretation_gates"]
    commands = manifest["commands"]
    lines = [
        "# Gated Attention Null-Probe Freeze",
        "",
        "This file is generated by `experiments/build_gated_attention_null_probe_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze the architecture-transfer probe before any gated-attention outputs are "
            "generated. The result decides whether the random-prefix mechanism is mostly an "
            "attention-sink exploit or a broader deterministic-trajectory diversification effect."
        ),
        "",
        "## Frozen Surface",
        "",
        f"- Probe ID: `{manifest['probe_id']}`",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Soft tokens: `{FROZEN_SOFT_TOKENS}`",
        f"- Arithmetic random-prefix seeds: `{', '.join(str(seed) for seed in FROZEN_RANDOM_PREFIX_SEEDS)}`",
        f"- Planning random-prefix seeds: `{', '.join(str(seed) for seed in FROZEN_PLANNING_SEEDS)}`",
        f"- No gated-attention outputs seen at freeze time: `{manifest['pre_result_boundary']['no_gated_attention_outputs_seen']}`",
        "",
        "## Model Arms",
        "",
        f"- Baseline: `{arms['baseline_family_matched']['model']}`",
        f"- Primary gated candidate: `{arms['gated_primary']['model']}`",
        f"- Mechanics-only fallback: `{arms['gated_small_control']['model']}`",
        "",
        "## Mandatory Conditions",
        "",
        "| ID | Soft Prompt | Position IDs | Decode |",
        "|---|---|---|---|",
    ]
    for condition in manifest["conditions"]:
        lines.append(
            f"| `{condition['id']}` | {condition['soft_prompt']} | "
            f"{condition['position_ids']} | {condition['decode']} |"
        )

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "Report mean metrics before oracle metrics:",
            "",
        ]
    )
    lines.extend(f"- {metric}" for metric in manifest["metrics"]["primary_report_first"])
    lines.extend(["", "Then report oracle/mechanism diagnostics:", ""])
    lines.extend(f"- {metric}" for metric in manifest["metrics"]["secondary_report_after_mean"])

    lines.extend(
        [
            "",
            "## Interpretation Gates",
            "",
            (
                f"- Sink-dependent: oracle `< {gates['sink_dependent']['oracle_coverage_lt']}` "
                f"and mean lift `< {gates['sink_dependent']['mean_lift_pp_lt']}pp` "
                "and no planning rescue."
            ),
            (
                f"- Trajectory diversification persists: oracle `> {gates['trajectory_diversification_persists']['oracle_coverage_gt']}` "
                f"or mean lift `> {gates['trajectory_diversification_persists']['or_mean_lift_pp_gt']}pp`."
            ),
            (
                f"- Ambiguous: oracle `{gates['ambiguous']['oracle_coverage_range'][0]}-"
                f"{gates['ambiguous']['oracle_coverage_range'][1]}` or mean lift "
                f"`{gates['ambiguous']['mean_lift_pp_range'][0]}-"
                f"{gates['ambiguous']['mean_lift_pp_range'][1]}pp`; {gates['ambiguous']['action']}."
            ),
            "",
            "## GPU Commands",
            "",
            "Baseline replay:",
            "",
            f"```powershell\n{commands['baseline_replay']}\n```",
            "",
            "Primary gated random-prefix run:",
            "",
            f"```powershell\n{commands['gated_primary_random_prefix']}\n```",
            "",
            "Position-shift control:",
            "",
            f"```text\n{commands['gated_position_shift']}\n```",
            "",
            "## Run Blockers",
            "",
            "- Verify the current Qwen3-Next snapshot and selected quantized artifact before spending GPU time.",
            "- Keep full raw outputs and true generated-token counts; old inputs_embeds token counts are not acceptable.",
            "- Do not update the article or README claim strength from oracle numbers alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
