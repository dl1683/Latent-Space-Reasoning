"""Build the multi-aspect latent aggregation v7 task and ontology freeze."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.build_latent_aggregation_multi_aspect_v5_freeze import (
    _dict,
    _float,
    _format_float,
    _load_tasks,
    _sha256,
    _task_hash,
)

DEFAULT_TASKS = Path("experiments/general_reasoning_tasks_scout.jsonl")
DEFAULT_V6_REPLAY = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_replay.json")
DEFAULT_V6_COVERAGE = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_coverage_gap.json")
DEFAULT_V6_THRESHOLD = Path(
    "eval_results/diffusion_language/latent_aggregation_multi_aspect_v6_threshold_sensitivity.json"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_freeze.json")
DEFAULT_REPORT_OUTPUT = Path("docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V7_FREEZE.md")
DEFAULT_LABEL_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_raw.jsonl")
DEFAULT_ONTOLOGY_PROBE_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_ontology_probe_raw.jsonl")
DEFAULT_CROSS_LATENT_RAW = Path("eval_results/diffusion_language/latent_aggregation_multi_aspect_v7_cross_latent_raw.jsonl")

FROZEN_TASK_PRESET = "latent_aggregation_multi_aspect_v7_plan345_392"
FROZEN_TASK_IDS = tuple(f"plan_{index:03d}" for index in range(345, 393))
PRIOR_PLANNING_TASK_MAX = 344

EXPANDED_ASPECTS = (
    {
        "aspect_id": "owner_assignment",
        "definition": "names who acts, decides, or owns a follow-up",
        "false_positive_risk": "invented stakeholder roles",
    },
    {
        "aspect_id": "timeline_or_sequence",
        "definition": "orders actions by dependency, time, or prerequisite",
        "false_positive_risk": "generic step numbering",
    },
    {
        "aspect_id": "rollback_or_exit_criteria",
        "definition": "states when to stop, revert, escalate, or change course",
        "false_positive_risk": "boilerplate rollback text",
    },
    {
        "aspect_id": "evidence_or_measurement",
        "definition": "names an observation or metric that proves progress",
        "false_positive_risk": "vague metric mentions",
    },
    {
        "aspect_id": "scope_boundary",
        "definition": "limits where the plan applies or does not apply",
        "false_positive_risk": "unsupported narrowing",
    },
    {
        "aspect_id": "polarity_or_action_direction",
        "definition": "distinguishes do, avoid, defer, escalate, or rollback",
        "false_positive_risk": "hidden contradiction",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--v6-replay", type=Path, default=DEFAULT_V6_REPLAY)
    parser.add_argument("--v6-coverage", type=Path, default=DEFAULT_V6_COVERAGE)
    parser.add_argument("--v6-threshold", type=Path, default=DEFAULT_V6_THRESHOLD)
    parser.add_argument("--label-raw", type=Path, default=DEFAULT_LABEL_RAW)
    parser.add_argument("--ontology-probe-raw", type=Path, default=DEFAULT_ONTOLOGY_PROBE_RAW)
    parser.add_argument("--cross-latent-raw", type=Path, default=DEFAULT_CROSS_LATENT_RAW)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_freeze_manifest(
        tasks_path=args.tasks,
        v6_replay_path=args.v6_replay,
        v6_coverage_path=args.v6_coverage,
        v6_threshold_path=args.v6_threshold,
        label_raw_path=args.label_raw,
        ontology_probe_raw_path=args.ontology_probe_raw,
        cross_latent_raw_path=args.cross_latent_raw,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "source_command_status": manifest["source_family_contract"]["command_status"],
                "task_count": manifest["task_count"],
                "task_preset": manifest["task_preset"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_freeze_manifest(
    *,
    tasks_path: Path,
    v6_replay_path: Path,
    v6_coverage_path: Path,
    v6_threshold_path: Path,
    label_raw_path: Path,
    ontology_probe_raw_path: Path,
    cross_latent_raw_path: Path,
) -> dict[str, object]:
    existing_outputs = [
        path for path in (label_raw_path, ontology_probe_raw_path, cross_latent_raw_path) if path.exists()
    ]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise ValueError(f"refusing v7 freeze after output artifacts exist: {paths}")

    _assert_fresh_task_ids(FROZEN_TASK_IDS)
    tasks_by_id = _load_tasks(tasks_path)
    missing = [task_id for task_id in FROZEN_TASK_IDS if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"frozen task ids are missing from {tasks_path}: {', '.join(missing)}")

    v6_replay = json.loads(v6_replay_path.read_text(encoding="utf-8"))
    v6_coverage = json.loads(v6_coverage_path.read_text(encoding="utf-8"))
    v6_threshold = json.loads(v6_threshold_path.read_text(encoding="utf-8"))
    replay_gate = _dict(v6_replay.get("gate_evaluation"))
    replay_summary = _dict(v6_replay.get("summary"))
    coverage_summary = _dict(v6_coverage.get("summary"))
    threshold_summary = _dict(v6_threshold.get("summary"))
    if replay_gate.get("overall_status") != "failed":
        raise ValueError("v7 freeze requires the committed failed v6 replay")
    if int(_float(replay_summary.get("complement_coverage_count"))) != 27:
        raise ValueError("v7 freeze requires the committed v6 coverage failure")
    if bool(threshold_summary.get("threshold_can_explain_failure")):
        raise ValueError("v7 freeze requires threshold sensitivity to reject threshold-only explanation")
    if int(_float(threshold_summary.get("positive_floor_coverage_count"))) >= 36:
        raise ValueError("v7 freeze requires remaining positive-floor coverage shortfall")

    return {
        "schema": "latent_aggregation_multi_aspect_v7_freeze.v1",
        "generated_by": "experiments/build_latent_aggregation_multi_aspect_v7_freeze.py",
        "task_preset": FROZEN_TASK_PRESET,
        "task_ids": list(FROZEN_TASK_IDS),
        "task_count": len(FROZEN_TASK_IDS),
        "task_source": {
            "path": str(tasks_path),
            "sha256": _sha256(tasks_path),
            "task_hashes": {task_id: _task_hash(tasks_by_id[task_id]) for task_id in FROZEN_TASK_IDS},
        },
        "prior_evidence": {
            "boundary": (
                "v6 is a fresh negative coverage-targeting result: positive lift and clean "
                "safety, but failed coverage, win-count, and Wilson gates. V7 changes "
                "observable complement surfaces instead of lowering gates."
            ),
            "v6_replay": _diagnostic_ref(v6_replay_path, replay_summary),
            "v6_coverage_gap": _diagnostic_ref(v6_coverage_path, coverage_summary),
            "v6_threshold_sensitivity": _diagnostic_ref(v6_threshold_path, threshold_summary),
        },
        "freshness_contract": {
            "prior_planning_task_max": PRIOR_PLANNING_TASK_MAX,
            "rule": "all v7 planning IDs must be greater than every prior committed aggregation planning slice",
            "status": "passed",
        },
        "expanded_aspect_ontology": {
            "status": "frozen_before_v7_labels",
            "aspects": list(EXPANDED_ASPECTS),
            "support_rules": [
                "require source-span support",
                "generic process language does not count by itself",
                "prompt-term echoing does not count by itself",
                "report old-ontology and expanded-ontology coverage separately",
                "report false-positive examples before promotion",
            ],
        },
        "source_family_contract": {
            "command_status": "implementation_required_before_generation",
            "families": [
                "baseline_dream_llada_low_confidence_random",
                "ontology_probe",
                "cross_latent_perturbation",
            ],
            "required_outputs": {
                "label_raw_output": str(label_raw_path),
                "ontology_probe_raw_output": str(ontology_probe_raw_path),
                "cross_latent_raw_output": str(cross_latent_raw_path),
            },
            "implementation_requirements": [
                "add replay support for v7 expanded aspects before scoring v7 promotion",
                "add source-family mapping for ontology_probe and cross_latent_perturbation",
                "forbid task_score and rubric-hit labels as extractor inputs",
                "report duplicate/noise rate for every new source family",
            ],
        },
        "statistical_gates": {
            "minimum_task_count": len(FROZEN_TASK_IDS),
            "minimum_complement_coverage_count": 36,
            "minimum_complement_coverage_fraction": 0.75,
            "minimum_conditional_promoted_fraction": 0.50,
            "minimum_conditional_non_rubric_lift": 0.05,
            "minimum_all_task_mean_non_rubric_lift": 0.035,
            "minimum_aggregate_win_count": 30,
            "minimum_wilson_lower_bound": 0.60,
            "maximum_unsupported_addition_count": 0,
            "maximum_hard_contradiction_count": 0,
        },
        "v7_specific_gates": {
            "must_report_old_vs_expanded_ontology_coverage": True,
            "must_report_false_positive_aspect_audit": True,
            "must_report_length_normalized_complement_yield": True,
            "must_report_source_family_unique_coverage": True,
            "must_report_theme_bucket_concentration": True,
            "must_report_label_leakage_check": True,
        },
        "failure_taxonomy": [
            "expanded_ontology_adds_false_positive_coverage",
            "new_sources_duplicate_existing_complements",
            "ontology_probe_copies_prompt_terms_without_support",
            "cross_latent_source_adds_noise_without_unique_coverage",
            "coverage_passes_but_conditional_lift_fails",
            "mean_lift_carried_by_one_theme_bucket",
            "hard_contradictions_from_polarity_aspects",
            "length_growth_explains_score_gain",
            "label_leakage_in_expanded_extractor",
        ],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    prior = _dict(manifest.get("prior_evidence"))
    ontology = _dict(manifest.get("expanded_aspect_ontology"))
    source_contract = _dict(manifest.get("source_family_contract"))
    gates = _dict(manifest.get("statistical_gates"))
    lines = [
        "# Latent Aggregation Multi-Aspect V7 Freeze",
        "",
        "This file is generated by `experiments/build_latent_aggregation_multi_aspect_v7_freeze.py`.",
        "",
        "## Decision",
        "",
        (
            "Freeze the fresh v7 task slice and expanded planning-aspect ontology. "
            "This manifest does not authorize generation yet: v7 source-family and "
            "expanded-ontology replay support must be implemented before GPU runs."
        ),
        "",
        "## Frozen Tasks",
        "",
        f"- Task preset: `{manifest['task_preset']}`",
        f"- Task count: `{manifest['task_count']}`",
        f"- Task IDs: `{', '.join(manifest['task_ids'])}`",
        "",
        "## Prior Boundary",
        "",
        f"- Boundary: {prior.get('boundary')}",
        f"- V6 replay: `{_dict(prior.get('v6_replay')).get('path')}`",
        f"- V6 coverage gap: `{_dict(prior.get('v6_coverage_gap')).get('path')}`",
        f"- V6 threshold sensitivity: `{_dict(prior.get('v6_threshold_sensitivity')).get('path')}`",
        "",
        "## Expanded Aspect Ontology",
        "",
        f"- Status: `{ontology.get('status')}`",
        "",
        "| Aspect | Definition | False-Positive Risk |",
        "| --- | --- | --- |",
    ]
    for aspect in _list_of_dicts(ontology.get("aspects")):
        lines.append(
            "| "
            f"`{aspect['aspect_id']}` | "
            f"{aspect['definition']} | "
            f"{aspect['false_positive_risk']} |"
        )
    lines.extend(
        [
            "",
            "Support rules:",
            "",
            *[f"- {rule}" for rule in ontology.get("support_rules", [])],
            "",
            "## Source-Family Contract",
            "",
            f"- Command status: `{source_contract.get('command_status')}`",
            f"- Families: `{', '.join(source_contract.get('families', []))}`",
            "",
            "Implementation requirements:",
            "",
            *[f"- {item}" for item in source_contract.get("implementation_requirements", [])],
            "",
            "## Statistical Gates",
            "",
            f"- Minimum task count: `{gates.get('minimum_task_count')}`",
            f"- Minimum complement coverage count: `{gates.get('minimum_complement_coverage_count')}`",
            f"- Minimum complement coverage fraction: `{_format_float(gates.get('minimum_complement_coverage_fraction'))}`",
            f"- Minimum all-task mean non-rubric lift: `{_format_float(gates.get('minimum_all_task_mean_non_rubric_lift'))}`",
            f"- Minimum aggregate wins: `{gates.get('minimum_aggregate_win_count')}`",
            f"- Minimum Wilson lower bound: `{_format_float(gates.get('minimum_wilson_lower_bound'))}`",
            "- Unsupported additions and hard contradictions must remain `0`.",
            "",
            "## V7-Specific Gates",
            "",
            *[f"- `{name}`" for name, enabled in _dict(manifest.get("v7_specific_gates")).items() if enabled],
            "",
            "## Failure Taxonomy",
            "",
            *[f"- `{item}`" for item in manifest.get("failure_taxonomy", [])],
        ]
    )
    return "\n".join(lines) + "\n"


def _assert_fresh_task_ids(task_ids: tuple[str, ...]) -> None:
    stale = [
        task_id
        for task_id in task_ids
        if not task_id.startswith("plan_")
        or int(task_id.removeprefix("plan_")) <= PRIOR_PLANNING_TASK_MAX
    ]
    if stale:
        raise ValueError(f"v7 task ids must be fresh above plan_{PRIOR_PLANNING_TASK_MAX:03d}: {stale}")


def _diagnostic_ref(path: Path, summary: dict[str, object]) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "summary": summary}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
