"""Audit denoise-phase geometry for diffusion repairability source states."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_diffusion_three_arm_benchmark import (
    _normalize,
    _planning_quality_score,
    _prompt_constraint_gap_terms,
    _prompt_keyword_coverage,
)

DEFAULT_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_fixed_source_repairability_gate_fresh_v1_raw.jsonl"
)
DEFAULT_REPAIRABILITY_AUDIT = Path(
    "eval_results/diffusion_language/diffusion_repairability_geometry_audit.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/diffusion_denoise_phase_geometry.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_DENOISE_PHASE_GEOMETRY.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--repairability-audit", type=Path, default=DEFAULT_REPAIRABILITY_AUDIT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--coverage-floor", type=float, default=0.40)
    parser.add_argument("--quality-floor", type=float, default=0.25)
    parser.add_argument("--repair-gap-min", type=int, default=2)
    parser.add_argument("--repair-gap-max", type=int, default=9)
    return parser.parse_args()


def build_denoise_phase_audit(
    *,
    raw_path: Path,
    repairability_audit_path: Path,
    coverage_floor: float = 0.40,
    quality_floor: float = 0.25,
    repair_gap_min: int = 2,
    repair_gap_max: int = 9,
) -> dict[str, object]:
    repairability_audit = _read_json(repairability_audit_path)
    repairability_rows = _list_of_dicts(repairability_audit.get("rows"))
    records = _source_record_index(_load_raw_records(raw_path))
    rows = []
    for repair_row in repairability_rows:
        task_id = str(repair_row.get("task_id", ""))
        source_control = str(repair_row.get("source_control", ""))
        source_record = records.get((task_id, source_control), {})
        if not source_record:
            continue
        prompt = str(source_record.get("prompt", ""))
        final_text = str(source_record.get("text", ""))
        history_steps = _int(source_record.get("history_steps"))
        samples = _phase_samples(
            source_record,
            prompt=prompt,
            coverage_floor=coverage_floor,
            quality_floor=quality_floor,
        )
        final_quality = _source_quality(source_record, prompt)
        final_coverage = _prompt_keyword_coverage(prompt, _normalize(final_text))
        final_gap_count = len(_prompt_constraint_gap_terms(prompt, final_text))
        rows.append(
            {
                "classification": str(repair_row.get("classification", "")),
                "coverage_at_midpoint": _sample_at_fraction(samples, 0.50, "prompt_coverage"),
                "coverage_floor": coverage_floor,
                "eos_pressure_peak": _max(
                    sample["eos_pressure"]
                    for sample in samples
                    if _int(sample.get("step")) < history_steps and _int(sample.get("mask_count")) > 0
                ),
                "final_gap_count": final_gap_count,
                "final_planning_quality": final_quality,
                "final_prompt_coverage": final_coverage,
                "first_quality_step": _first_step_at_or_above(samples, "planning_quality", quality_floor),
                "first_skeleton_step": _first_skeleton_step(samples, coverage_floor=coverage_floor),
                "first_joint_phase_step": _first_joint_phase_step(
                    samples,
                    coverage_floor=coverage_floor,
                    quality_floor=quality_floor,
                ),
                "first_mask_free_step": _nested_int(source_record, ("trajectory_summary", "first_mask_free_step")),
                "history_steps": history_steps,
                "late_coverage_gain": final_coverage
                - (_sample_at_fraction(samples, 0.75, "prompt_coverage") or 0.0),
                "mask_count_increase_count": _nested_int(
                    source_record,
                    ("trajectory_summary", "mask_count_increase_count"),
                ),
                "peak_coverage": _max(sample["prompt_coverage"] for sample in samples),
                "peak_coverage_step": _step_of_max(samples, "prompt_coverage"),
                "peak_quality": _max(sample["planning_quality"] for sample in samples),
                "peak_quality_step": _step_of_max(samples, "planning_quality"),
                "phase": _phase_label(
                    final_quality=final_quality,
                    final_coverage=final_coverage,
                    final_gap_count=final_gap_count,
                    coverage_floor=coverage_floor,
                    quality_floor=quality_floor,
                    repair_gap_min=repair_gap_min,
                    repair_gap_max=repair_gap_max,
                ),
                "quality_floor": quality_floor,
                "sample_count": len(samples),
                "source_control": source_control,
                "task_id": task_id,
            }
        )
    return {
        "coverage_floor": coverage_floor,
        "generated_by": "experiments/analyze_diffusion_denoise_phase_geometry.py",
        "quality_floor": quality_floor,
        "raw_path": str(raw_path),
        "repair_gap_max": repair_gap_max,
        "repair_gap_min": repair_gap_min,
        "repairability_audit_path": str(repairability_audit_path),
        "rows": rows,
        "schema": "diffusion_denoise_phase_geometry.v1",
        "summary": _summary(rows),
    }


def render_markdown(audit: dict[str, object]) -> str:
    summary = _dict(audit.get("summary"))
    rows = _list_of_dicts(audit.get("rows"))
    lines = [
        "# Diffusion Denoise Phase Geometry",
        "",
        "This file is generated by `experiments/analyze_diffusion_denoise_phase_geometry.py`.",
        "It audits when useful constraint skeletons appear inside sampled diffusion denoise histories.",
        "",
        "## Research Link",
        "",
        (
            "Local research in `_meta` and `Market Reports/Open Exploration` frames the next "
            "reasoning system as energy-bounded attention plus detect-diagnose-repair loops. "
            "This audit translates that into a diffusion-native question: which denoise "
            "phases are worth spending extra repair compute on?"
        ),
        "",
        "## Summary",
        "",
        f"- Raw: `{audit.get('raw_path', '')}`",
        f"- Repairability audit: `{audit.get('repairability_audit_path', '')}`",
        f"- Rows: `{summary.get('row_count', 0)}`",
        f"- Phase counts: `{summary.get('phase_counts', {})}`",
        f"- Classification counts: `{summary.get('classification_counts', {})}`",
        f"- Repairable-phase precision/recall: `{_format_float(summary.get('repairable_phase_precision'))}` / `{_format_float(summary.get('repairable_phase_recall'))}`",
        f"- Mean first skeleton step by class: `{summary.get('mean_first_skeleton_step_by_class', {})}`",
        f"- Mean EOS pressure peak by class: `{summary.get('mean_eos_pressure_peak_by_class', {})}`",
        "",
        "## Phase Table",
        "",
        (
            "| Task | Class | Phase | Source | Final PQ | Final Coverage | Gap | "
            "First Skeleton | First Joint | Peak Coverage | EOS Peak | Late Cov Gain |"
        ),
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('task_id', '')} | "
            f"{row.get('classification', '')} | "
            f"{row.get('phase', '')} | "
            f"`{row.get('source_control', '')}` | "
            f"{_format_float(row.get('final_planning_quality'))} | "
            f"{_format_float(row.get('final_prompt_coverage'))} | "
            f"{int(row.get('final_gap_count', 0))} | "
            f"{_format_optional_int(row.get('first_skeleton_step'))} | "
            f"{_format_optional_int(row.get('first_joint_phase_step'))} | "
            f"{_format_float(row.get('peak_coverage'))} | "
            f"{_format_float(row.get('eos_pressure_peak'))} | "
            f"{_format_float(row.get('late_coverage_gain'))} |"
        )
    return "\n".join(lines) + "\n"


def _phase_samples(
    record: dict[str, object],
    *,
    prompt: str,
    coverage_floor: float,
    quality_floor: float,
) -> list[dict[str, object]]:
    samples = _summary_samples(record)
    rows = []
    for sample in samples:
        visible_text = str(sample.get("visible_text", "") or sample.get("text", ""))
        quality = _planning_quality_score({"text": visible_text}, prompt) if visible_text else 0.0
        coverage = _prompt_keyword_coverage(prompt, _normalize(visible_text)) if visible_text else 0.0
        gap_count = len(_prompt_constraint_gap_terms(prompt, visible_text)) if visible_text else 0
        eos_count = _int(sample.get("eos_count"))
        mask_count = _int(sample.get("mask_count"))
        rows.append(
            {
                "eos_pressure": eos_count / max(1, eos_count + mask_count),
                "is_joint_phase": coverage >= coverage_floor and quality >= quality_floor,
                "is_skeleton": coverage >= coverage_floor and _int(sample.get("visible_chars")) >= 20,
                "mask_count": mask_count,
                "planning_quality": quality,
                "prompt_coverage": coverage,
                "prompt_gap_count": gap_count,
                "step": _int(sample.get("step")),
                "visible_chars": _int(sample.get("visible_chars")),
            }
        )
    return rows


def _summary_samples(record: dict[str, object]) -> list[dict[str, object]]:
    trajectory_summary = _dict(record.get("trajectory_summary"))
    samples = _list_of_dicts(trajectory_summary.get("samples"))
    if samples:
        return samples
    return _list_of_dicts(record.get("history_samples"))


def _phase_label(
    *,
    final_quality: float,
    final_coverage: float,
    final_gap_count: int,
    coverage_floor: float,
    quality_floor: float,
    repair_gap_min: int,
    repair_gap_max: int,
) -> str:
    if final_quality >= quality_floor and final_gap_count < repair_gap_min:
        return "complete_source"
    if final_coverage < coverage_floor or final_gap_count > repair_gap_max:
        return "undercovered_or_overdiffuse"
    if final_quality < quality_floor:
        return "low_quality_repairable_skeleton"
    return "repairable_skeleton"


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    phase_counts = Counter(str(row.get("phase", "")) for row in rows)
    class_counts = Counter(str(row.get("classification", "")) for row in rows)
    skeleton_by_class: dict[str, list[object]] = defaultdict(list)
    eos_by_class: dict[str, list[object]] = defaultdict(list)
    for row in rows:
        classification = str(row.get("classification", ""))
        skeleton_by_class[classification].append(row.get("first_skeleton_step"))
        eos_by_class[classification].append(row.get("eos_pressure_peak"))
    repairable_rows = [row for row in rows if _is_repairable_phase(str(row.get("phase", "")))]
    productive_rows = [row for row in rows if str(row.get("classification", "")) == "productive_spend"]
    productive_repairable_rows = [
        row
        for row in productive_rows
        if _is_repairable_phase(str(row.get("phase", "")))
    ]
    return {
        "classification_counts": dict(sorted(class_counts.items())),
        "mean_eos_pressure_peak_by_class": {
            key: _rounded_mean(values) for key, values in sorted(eos_by_class.items())
        },
        "mean_first_skeleton_step_by_class": {
            key: _rounded_mean(values) for key, values in sorted(skeleton_by_class.items())
        },
        "phase_counts": dict(sorted(phase_counts.items())),
        "repairable_phase_count": len(repairable_rows),
        "repairable_phase_precision": (
            len(productive_repairable_rows) / len(repairable_rows) if repairable_rows else None
        ),
        "repairable_phase_recall": (
            len(productive_repairable_rows) / len(productive_rows) if productive_rows else None
        ),
        "row_count": len(rows),
    }


def _is_repairable_phase(phase: str) -> bool:
    return phase in {"repairable_skeleton", "low_quality_repairable_skeleton"}


def _source_record_index(records: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    index = {}
    for record in records:
        if str(record.get("generation_stage", "")) != "candidate_generation":
            continue
        task_id = _task_id(record)
        control = str(_dict(record.get("schedule")).get("name", ""))
        if task_id and control:
            index[(task_id, control)] = record
    return index


def _load_raw_records(path: Path) -> list[dict[str, object]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_quality(record: dict[str, object], prompt: str) -> float:
    value = record.get("planning_quality_score")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return _planning_quality_score(record, prompt)


def _task_id(record: dict[str, object]) -> str:
    task = _dict(record.get("task"))
    return str(task.get("task_id", record.get("task_id", "")))


def _sample_at_fraction(samples: list[dict[str, object]], fraction: float, key: str) -> float | None:
    if not samples:
        return None
    max_step = max(_int(sample.get("step")) for sample in samples)
    target_step = max(1, int(round(max_step * fraction)))
    candidates = [sample for sample in samples if _int(sample.get("step")) <= target_step]
    if not candidates:
        candidates = samples[:1]
    return _optional_float(candidates[-1].get(key))


def _first_skeleton_step(samples: list[dict[str, object]], *, coverage_floor: float) -> int | None:
    for sample in samples:
        if _float(sample.get("prompt_coverage")) >= coverage_floor and _int(sample.get("visible_chars")) >= 20:
            return _int(sample.get("step"))
    return None


def _first_joint_phase_step(
    samples: list[dict[str, object]],
    *,
    coverage_floor: float,
    quality_floor: float,
) -> int | None:
    for sample in samples:
        if (
            _float(sample.get("prompt_coverage")) >= coverage_floor
            and _float(sample.get("planning_quality")) >= quality_floor
        ):
            return _int(sample.get("step"))
    return None


def _first_step_at_or_above(
    samples: list[dict[str, object]],
    key: str,
    threshold: float,
) -> int | None:
    for sample in samples:
        if _float(sample.get(key)) >= threshold:
            return _int(sample.get("step"))
    return None


def _step_of_max(samples: list[dict[str, object]], key: str) -> int | None:
    if not samples:
        return None
    best = max(samples, key=lambda sample: _float(sample.get(key)))
    return _int(best.get("step"))


def _nested_int(record: dict[str, object], path: tuple[str, ...]) -> int | None:
    current: object = record
    for key in path:
        current = _dict(current).get(key)
    if isinstance(current, int) and not isinstance(current, bool):
        return current
    return None


def _max(values: Any) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, int | float)]
    if not numbers:
        return None
    return max(numbers)


def _rounded_mean(values: list[object]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, int | float)]
    if not numbers:
        return None
    return round(mean(numbers), 6)


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _float(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _format_float(value: object) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return ""
    return f"{float(value):.6f}"


def _format_optional_int(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return ""


def main() -> int:
    args = parse_args()
    audit = build_denoise_phase_audit(
        raw_path=args.raw,
        repairability_audit_path=args.repairability_audit,
        coverage_floor=args.coverage_floor,
        quality_floor=args.quality_floor,
        repair_gap_min=args.repair_gap_min,
        repair_gap_max=args.repair_gap_max,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(render_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "rows": audit["summary"]["row_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
