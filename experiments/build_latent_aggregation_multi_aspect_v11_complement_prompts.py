"""Build the v11 complement-packet prompt artifact from label-free source rows.

Identical to the v10 prompt builder but references v11 freeze, v11 label
raw/scores, and v11 output paths.  LLaDA-only (no Dream candidate filter
change needed — the v10 builder already filters to llada-8b-instruct-hf).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.build_latent_aggregation_multi_aspect_v11_complement_freeze import (
    DEFAULT_PACKET_RAW,
    DEFAULT_PACKET_REPORT,
    DEFAULT_PACKET_SCORES,
    DEFAULT_REPLAY_OUTPUT,
    DEFAULT_LABEL_RAW,
    DEFAULT_LABEL_SCORES,
    DEFAULT_JSON_OUTPUT as DEFAULT_FREEZE,
)
from experiments.build_latent_aggregation_multi_aspect_v5_freeze import _dict, _load_tasks, _sha256, _task_hash
from experiments.build_latent_aggregation_multi_aspect_v9_complement_source import _generation_command
from experiments.latent_aggregation_expanded_aspects import EXPANDED_PLANNING_ASPECTS, expanded_aspect_scores
from experiments.run_latent_aggregation_inference_replay import _trajectory_id
from experiments.run_latent_aggregation_multi_aspect_v2_replay import _read_jsonl

DEFAULT_TASKS = Path("experiments/latent_aggregation_v11_planning_tasks.jsonl")
DEFAULT_PROMPTS_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v11_complement_packet_prompts.jsonl"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v11_complement_prompt_contract.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V11_COMPLEMENT_PROMPTS.md"
)

FORBIDDEN_LABEL_INPUTS = (
    "v11_complement_packet_raw",
    "v11_complement_packet_scores",
    "v11_complement_packet_replay",
    "v11_complement_packet_aspects",
    "v11_complement_packet_realized",
    "v11 replay labels",
    "v10 replay decisions",
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
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
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
        raise ValueError(f"refusing v11 prompt build after packet/replay artifacts exist: {paths}")

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
        raise ValueError(f"v11 prompt rows have no source candidates: {', '.join(empty_source)}")

    source_counts = [len(row["source_candidates"]) for row in prompt_rows]
    missing_counts = [len(row["missing_anchor_aspects"]) for row in prompt_rows]
    manifest = {
        "schema": "latent_aggregation_multi_aspect_v11_complement_prompt_contract.v1",
        "generated_by": "experiments/build_latent_aggregation_multi_aspect_v11_complement_prompts.py",
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
        "# Latent Aggregation V11 Complement-Packet Prompts",
        "",
        "This file is generated by `experiments/build_latent_aggregation_multi_aspect_v11_complement_prompts.py`.",
        "",
        "## Decision",
        "",
        (
            "The v11 complement-packet prompt artifact is populated for the fresh 96-task "
            "LLaDA-only transfer slice. It is not a result claim; it is the pre-packet "
            "generation contract that converts fresh source rows into source-supported "
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
        "## Boundary",
        "",
        (
            "Do not treat this artifact as evidence that complement packets work. The next "
            "valid evidence step is packet generation followed by frozen replay with no "
            "threshold, ontology, or realization-rule changes."
        ),
        "",
    ]
    return "\n".join(lines)


def _validate_freeze(freeze: dict[str, object]) -> None:
    if freeze.get("schema") != "latent_aggregation_multi_aspect_v11_complement_freeze.v1":
        raise ValueError("v11 complement prompt builder requires the v11 complement freeze schema")
    transfer = _dict(freeze.get("transfer_contract"))
    if transfer.get("policy") != "v10_complement_packet_policy_fixed_before_v11_labels":
        raise ValueError("v11 complement prompt builder requires the frozen v10 packet policy")
    if not freeze.get("task_ids"):
        raise ValueError("v11 complement freeze has no task IDs")


def _validate_label_source(label_scores: dict[str, object], *, task_ids: list[str]) -> None:
    if not label_scores.get("run_id"):
        raise ValueError("v11 label source scores must include a run_id")
    if int(label_scores.get("all_generation_count", 0)) <= 0:
        raise ValueError("v11 label source scores must include generated rows")
    comparison_ids = {str(row.get("task_id", "")) for row in _list_of_dicts(label_scores.get("comparison_rows"))}
    missing = [task_id for task_id in task_ids if task_id not in comparison_ids]
    if missing:
        raise ValueError(f"v11 label source scores are missing frozen task IDs: {', '.join(missing)}")


def _label_free_rows(rows: list[dict[str, object]], *, task_ids: list[str]) -> list[dict[str, object]]:
    allowed_ids = set(task_ids)
    filtered = []
    for row in rows:
        task_id = _record_task_id(row)
        if task_id not in allowed_ids:
            continue
        if str(row.get("candidate_key", "")) != "llada-8b-instruct-hf":
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        filtered.append(row)
    missing = sorted(allowed_ids - {_record_task_id(row) for row in filtered})
    if missing:
        raise ValueError(f"v11 label raw source has no non-empty LLaDA rows for: {', '.join(missing)}")
    return filtered


def _rows_by_task(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_task: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_task.setdefault(_record_task_id(row), []).append(row)
    return by_task


def _prompt_row(*, task: dict[str, object], task_rows: list[dict[str, object]]) -> dict[str, object]:
    task_id = str(task.get("task_id", ""))
    if not task_rows:
        raise ValueError(f"no v11 label source rows for {task_id}")
    prompt = str(task.get("prompt", ""))
    anchor = _anchor_row(task_rows)
    anchor_text = str(anchor.get("text", "")).strip()
    anchor_aspects = _present_aspects(anchor_text, prompt=prompt)
    missing = [aspect for aspect in EXPANDED_PLANNING_ASPECTS if aspect not in anchor_aspects]
    source_candidates = _source_candidates(
        task_rows=task_rows,
        anchor=anchor,
        prompt=prompt,
        missing_aspects=missing,
    )
    return {
        "anchor_candidate_key": str(anchor.get("candidate_key", "")),
        "anchor_generation_stage": str(anchor.get("generation_stage", "")),
        "anchor_schedule": str(_dict(anchor.get("schedule")).get("name", "")),
        "anchor_text": anchor_text,
        "anchor_trajectory_id": _trajectory_id(anchor, 0, stable=True),
        "label_free_derivation": True,
        "missing_anchor_aspects": missing,
        "present_anchor_aspects": anchor_aspects,
        "prompt": _complement_prompt(
            task_prompt=prompt,
            anchor_text=anchor_text,
            missing_aspects=missing,
            source_candidates=source_candidates,
        ),
        "rubric_items": list(task.get("rubric_items", [])),
        "source_candidates": source_candidates,
        "target_candidate_count": 3,
        "task_id": task_id,
    }


def _anchor_row(task_rows: list[dict[str, object]]) -> dict[str, object]:
    preferred = [
        row
        for row in task_rows
        if str(row.get("generation_stage", "")) == "candidate_generation"
        and str(_dict(row.get("schedule")).get("name", "")) == "low_confidence_32"
    ]
    if preferred:
        return preferred[0]
    candidates = [row for row in task_rows if str(row.get("text", "")).strip()]
    return candidates[0]


def _source_candidates(
    *,
    task_rows: list[dict[str, object]],
    anchor: dict[str, object],
    prompt: str,
    missing_aspects: list[str],
) -> list[dict[str, object]]:
    anchor_id = _trajectory_id(anchor, 0, stable=True)
    candidates = []
    for row in task_rows:
        trajectory_id = _trajectory_id(row, 0, stable=True)
        text = str(row.get("text", "")).strip()
        if not text or trajectory_id == anchor_id:
            continue
        present = _present_aspects(text, prompt=prompt)
        supported_missing = [aspect for aspect in missing_aspects if aspect in present]
        candidates.append(
            {
                "candidate_key": str(row.get("candidate_key", "")),
                "generation_stage": str(row.get("generation_stage", "")),
                "schedule": str(_dict(row.get("schedule")).get("name", "")),
                "supported_missing_aspects": supported_missing,
                "text": text,
                "trajectory_id": trajectory_id,
            }
        )
    candidates.sort(
        key=lambda row: (
            -len(row["supported_missing_aspects"]),
            -len(str(row["text"])),
            str(row["trajectory_id"]),
        )
    )
    return candidates[:3]


def _present_aspects(text: str, *, prompt: str) -> list[str]:
    scores = expanded_aspect_scores(text, prompt=prompt)
    return [
        aspect
        for aspect in EXPANDED_PLANNING_ASPECTS
        if float(_dict(scores.get(f"expanded::{aspect}")).get("support_score", 0.0)) > 0
    ]


def _complement_prompt(
    *,
    task_prompt: str,
    anchor_text: str,
    missing_aspects: list[str],
    source_candidates: list[dict[str, object]],
) -> str:
    aspect_text = ", ".join(missing_aspects) if missing_aspects else "any task-relevant aspect missing from the anchor"
    source_blocks = []
    for index, row in enumerate(source_candidates, start=1):
        supported = ", ".join(row["supported_missing_aspects"]) or "no explicit expanded-aspect hit"
        source_blocks.append(
            f"Source {index} ({row['trajectory_id']}; supports: {supported}):\n{_truncate(str(row['text']))}"
        )
    source_text = "\n\n".join(source_blocks) if source_blocks else "No auxiliary source text is available."
    return (
        f"Task:\n{task_prompt}\n\n"
        f"Current anchor answer:\n{anchor_text}\n\n"
        f"Auxiliary source text that may contain missing details:\n{source_text}\n\n"
        "Generate a complement packet, not a replacement final answer.\n"
        "Hard output rules:\n"
        "- Return raw JSON only; do not wrap it in markdown fences.\n"
        "- Return exactly 3 complement clauses.\n"
        "- Every clause must be one sentence, must add information absent from the anchor, "
        "and must be directly usable in a final answer.\n"
        "- Every clause must be grounded in the task or one of the auxiliary source texts.\n"
        "- Every `why_not_in_anchor` value must be non-empty and must identify the exact missing anchor detail.\n"
        "- Do not omit any object key from any clause.\n"
        "- Do not restate the anchor, do not contradict it, and do not invent facts outside the task.\n\n"
        f"Prioritize missing expanded-aspect types: {aspect_text}.\n\n"
        "Return this JSON shape exactly: "
        "{\"complement_clauses\":[{\"aspect_type\":\"...\",\"clause\":\"...\","
        "\"why_not_in_anchor\":\"...\"}]}.\n"
        "Example clause object: "
        "{\"aspect_type\":\"owner_assignment\",\"clause\":\"Name a directly responsible owner for the audit step.\","
        "\"why_not_in_anchor\":\"The anchor mentions the audit step but does not assign responsibility.\"}"
    )


def _record_task_id(record: dict[str, object]) -> str:
    task = _dict(record.get("task"))
    return str(task.get("task_id", record.get("task_id", "")))


def _truncate(text: str, limit: int = 900) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


if __name__ == "__main__":
    raise SystemExit(main())
