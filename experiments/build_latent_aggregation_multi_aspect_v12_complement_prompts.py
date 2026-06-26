"""Build the v12 complement-packet prompt artifact from label-free source rows.

Same core logic as v11 prompt builder but references v12 freeze, v12 label
raw/scores, and v12 output paths.  LLaDA-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.build_latent_aggregation_multi_aspect_v12_complement_freeze import (
    DEFAULT_PACKET_RAW,
    DEFAULT_PACKET_REPORT,
    DEFAULT_PACKET_SCORES,
    DEFAULT_REPLAY_OUTPUT,
    DEFAULT_LABEL_RAW,
    DEFAULT_LABEL_SCORES,
    DEFAULT_JSON_OUTPUT as DEFAULT_FREEZE,
    DEFAULT_V12_TASKS,
)
from experiments.build_latent_aggregation_multi_aspect_v5_freeze import _dict, _load_tasks, _sha256, _task_hash
from experiments.build_latent_aggregation_multi_aspect_v9_complement_source import _generation_command
from experiments.build_latent_aggregation_multi_aspect_v11_complement_prompts import (
    _label_free_rows,
    _list_of_dicts,
    _prompt_row,
    _record_task_id,
    _rows_by_task,
    _validate_label_source,
)
from experiments.run_latent_aggregation_multi_aspect_v2_replay import _read_jsonl

DEFAULT_TASKS = DEFAULT_V12_TASKS
DEFAULT_PROMPTS_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_complement_packet_prompts.jsonl"
)
DEFAULT_PROMPT_CONTRACT_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_complement_prompt_contract.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V12_COMPLEMENT_PROMPTS.md"
)

FORBIDDEN_LABEL_INPUTS = (
    "v12_complement_packet_raw",
    "v12_complement_packet_scores",
    "v12_complement_packet_replay",
    "v12_complement_packet_aspects",
    "v12_complement_packet_realized",
    "v12 replay labels",
    "v11 replay decisions",
    "rubric hit labels",
    "post-packet aggregation outcomes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--label-raw", type=Path, default=DEFAULT_LABEL_RAW)
    parser.add_argument("--label-scores", type=Path, default=DEFAULT_LABEL_SCORES)
    parser.add_argument("--prompts-output", type=Path, default=DEFAULT_PROMPTS_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_PROMPT_CONTRACT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--packet-raw", type=Path, default=DEFAULT_PACKET_RAW)
    parser.add_argument("--packet-scores", type=Path, default=DEFAULT_PACKET_SCORES)
    parser.add_argument("--packet-report", type=Path, default=DEFAULT_PACKET_REPORT)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY_OUTPUT)
    parser.add_argument("--allow-existing-packet-artifacts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, prompt_rows = build_complement_prompt_contract(
        freeze_path=args.freeze,
        tasks_path=args.tasks,
        label_raw_path=args.label_raw,
        label_scores_path=args.label_scores,
        prompts_output_path=args.prompts_output,
        packet_raw_path=args.packet_raw,
        packet_scores_path=args.packet_scores,
        packet_report_path=args.packet_report,
        replay_output_path=args.replay_output,
        allow_existing_packet_artifacts=args.allow_existing_packet_artifacts,
    )
    args.prompts_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.prompts_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in prompt_rows) + "\n",
        encoding="utf-8",
    )
    manifest["prompt_artifact"] = {
        "path": str(args.prompts_output),
        "row_count": len(prompt_rows),
        "sha256": _sha256(args.prompts_output),
    }
    args.json_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "prompt_rows": len(prompt_rows),
                "prompts_output": str(args.prompts_output),
                "report_output": str(args.report_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_complement_prompt_contract(
    *,
    freeze_path: Path,
    tasks_path: Path,
    label_raw_path: Path,
    label_scores_path: Path,
    prompts_output_path: Path,
    packet_raw_path: Path,
    packet_scores_path: Path,
    packet_report_path: Path,
    replay_output_path: Path,
    allow_existing_packet_artifacts: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    existing_packet_outputs = [
        path for path in (packet_raw_path, packet_scores_path, packet_report_path, replay_output_path) if path.exists()
    ]
    if existing_packet_outputs and not allow_existing_packet_artifacts:
        paths = ", ".join(str(path) for path in existing_packet_outputs)
        raise ValueError(f"refusing v12 prompt build after packet/replay artifacts exist: {paths}")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    _validate_freeze(freeze)
    tasks_by_id = _load_tasks(tasks_path)
    task_ids = [str(task_id) for task_id in freeze.get("task_ids", [])]
    missing_tasks = [task_id for task_id in task_ids if task_id not in tasks_by_id]
    if missing_tasks:
        raise ValueError(f"frozen task IDs are missing from {tasks_path}: {', '.join(missing_tasks)}")
    label_rows = _label_free_rows(_read_jsonl(label_raw_path), task_ids=task_ids)
    label_scores = json.loads(label_scores_path.read_text(encoding="utf-8"))
    _validate_label_source(label_scores, task_ids=task_ids)
    rows_by_task = _rows_by_task(label_rows)
    prompt_rows = [
        _prompt_row(task=tasks_by_id[task_id], task_rows=rows_by_task.get(task_id, [])) for task_id in task_ids
    ]
    empty_source = [row["task_id"] for row in prompt_rows if not row["source_candidates"]]
    if empty_source:
        raise ValueError(f"v12 prompt rows have no source candidates: {', '.join(empty_source)}")

    source_counts = [len(row["source_candidates"]) for row in prompt_rows]
    missing_counts = [len(row["missing_anchor_aspects"]) for row in prompt_rows]
    manifest = {
        "schema": "latent_aggregation_multi_aspect_v12_complement_prompt_contract.v1",
        "generated_by": "experiments/build_latent_aggregation_multi_aspect_v12_complement_prompts.py",
        "freeze": {
            "path": str(freeze_path),
            "sha256": _sha256(freeze_path),
            "schema": freeze.get("schema"),
            "task_preset": freeze.get("task_preset"),
        },
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "task_source": {
            "path": str(tasks_path),
            "sha256": _sha256(tasks_path),
            "task_hashes": {task_id: _task_hash(tasks_by_id[task_id]) for task_id in task_ids},
        },
        "source_inputs": {
            "label_raw_path": str(label_raw_path),
            "label_raw_sha256": _sha256(label_raw_path),
            "label_scores_path": str(label_scores_path),
            "label_scores_sha256": _sha256(label_scores_path),
            "allowed_fields": [
                "task prompt",
                "rubric item text",
                "generated source text",
                "candidate key",
                "schedule name",
                "generation stage",
                "stable trajectory id",
                "source-run content hash",
            ],
            "forbidden_inputs": list(FORBIDDEN_LABEL_INPUTS),
            "label_free_derivation": True,
        },
        "prompt_artifact": {"path": str(prompts_output_path), "row_count": len(prompt_rows), "sha256": None},
        "summary": {
            "anchor_policy": "first non-empty llada low-confidence candidate per task, fallback first non-empty llada row",
            "source_policy": "non-anchor llada rows ranked by missing expanded-aspect support, then source text length",
            "source_candidate_count_min": min(source_counts) if source_counts else 0,
            "source_candidate_count_max": max(source_counts) if source_counts else 0,
            "source_candidate_count_mean": sum(source_counts) / len(source_counts) if source_counts else 0.0,
            "missing_anchor_aspect_count_min": min(missing_counts) if missing_counts else 0,
            "missing_anchor_aspect_count_max": max(missing_counts) if missing_counts else 0,
            "missing_anchor_aspect_count_mean": sum(missing_counts) / len(missing_counts) if missing_counts else 0.0,
        },
        "packet_generation": {
            "command": _generation_command(
                prompts_output_path=prompts_output_path,
                raw_output_path=packet_raw_path,
                scores_output_path=packet_scores_path,
                source_report_output_path=packet_report_path,
            ),
            "expected_packet_samples_per_task": 3,
            "status": "prompt_artifact_ready_packet_generation_pending",
        },
    }
    return manifest, prompt_rows


def render_markdown(manifest: dict[str, object]) -> str:
    source_inputs = _dict(manifest.get("source_inputs"))
    summary = _dict(manifest.get("summary"))
    packet = _dict(manifest.get("packet_generation"))
    lines = [
        "# Latent Aggregation V12 Complement-Packet Prompts",
        "",
        "This file is generated by `experiments/build_latent_aggregation_multi_aspect_v12_complement_prompts.py`.",
        "",
        "## Decision",
        "",
        (
            "The v12 complement-packet prompt artifact is populated for the fresh 120-task "
            "LLaDA-only filtered replication slice. It is not a result claim; it is the "
            "pre-packet generation contract that converts fresh source rows into source-supported "
            "complement prompts."
        ),
        "",
        "## Leakage Boundary",
        "",
        f"- Label-free derivation: `{source_inputs.get('label_free_derivation')}`",
        f"- Allowed inputs: {', '.join(source_inputs.get('allowed_fields', []))}",
        f"- Forbidden inputs: {', '.join(source_inputs.get('forbidden_inputs', []))}",
        "",
        "## Prompt Artifact",
        "",
        f"- Path: `{_dict(manifest.get('prompt_artifact')).get('path')}`",
        f"- Rows: `{_dict(manifest.get('prompt_artifact')).get('row_count')}`",
        f"- SHA256: `{_dict(manifest.get('prompt_artifact')).get('sha256')}`",
        "",
        "## Source Policy",
        "",
        f"- Anchor policy: {summary.get('anchor_policy')}",
        f"- Source policy: {summary.get('source_policy')}",
        f"- Source candidates per prompt: `{summary.get('source_candidate_count_min')}` to `{summary.get('source_candidate_count_max')}`",
        f"- Missing anchor aspects per prompt: `{summary.get('missing_anchor_aspect_count_min')}` to `{summary.get('missing_anchor_aspect_count_max')}`",
        "",
        "## Next Command",
        "",
        "```powershell",
        str(packet.get("command", "")),
        "```",
        "",
    ]
    return "\n".join(lines)


def _validate_freeze(freeze: dict[str, object]) -> None:
    if freeze.get("schema") != "latent_aggregation_multi_aspect_v12_complement_freeze.v1":
        raise ValueError("v12 complement prompt builder requires the v12 complement freeze schema")
    transfer = _dict(freeze.get("transfer_contract"))
    if not freeze.get("task_ids"):
        raise ValueError("v12 complement freeze has no task IDs")


if __name__ == "__main__":
    raise SystemExit(main())
