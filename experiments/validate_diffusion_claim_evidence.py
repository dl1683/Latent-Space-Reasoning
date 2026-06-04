"""Validate promoted diffusion claim evidence artifacts.

This is the hard gate for public diffusion-reasoning claims. It verifies that
the generated Markdown/JSON claim map is current with the promoted claim specs
and that every promoted score file has internally coherent coverage, budget,
win/loss, oracle, and raw-artifact counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

try:
    from experiments.build_diffusion_claim_evidence import (
        DEFAULT_CANONICAL_SLOTS,
        DEFAULT_CLAIMS,
        DEFAULT_INDEX_MARKDOWN_OUTPUT,
        DEFAULT_INDEX_OUTPUT,
        DEFAULT_JSON_OUTPUT,
        DEFAULT_OUTPUT,
        DEFAULT_PUBLIC_BENCHMARK_JSON_OUTPUT,
        DEFAULT_PUBLIC_BENCHMARK_OUTPUT,
        CanonicalClaimSlot,
        ClaimEvidence,
        ClaimSpec,
        RepairDiagnosticRequirement,
        build_claim_evidence,
        build_ground_truth_index,
        build_public_benchmark_summary,
        render_ground_truth_index_markdown,
        render_markdown,
        render_public_benchmark_markdown,
    )
    from experiments.scan_stale_diffusion_docs import (
        DEFAULT_DOC_PATHS,
        StaleDiffusionDocIssue,
        scan_stale_diffusion_docs,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from build_diffusion_claim_evidence import (
        DEFAULT_CANONICAL_SLOTS,
        DEFAULT_CLAIMS,
        DEFAULT_INDEX_MARKDOWN_OUTPUT,
        DEFAULT_INDEX_OUTPUT,
        DEFAULT_JSON_OUTPUT,
        DEFAULT_OUTPUT,
        DEFAULT_PUBLIC_BENCHMARK_JSON_OUTPUT,
        DEFAULT_PUBLIC_BENCHMARK_OUTPUT,
        CanonicalClaimSlot,
        ClaimEvidence,
        ClaimSpec,
        RepairDiagnosticRequirement,
        build_claim_evidence,
        build_ground_truth_index,
        build_public_benchmark_summary,
        render_ground_truth_index_markdown,
        render_markdown,
        render_public_benchmark_markdown,
    )
    from scan_stale_diffusion_docs import (
        DEFAULT_DOC_PATHS,
        StaleDiffusionDocIssue,
        scan_stale_diffusion_docs,
    )

FLOAT_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--index-output", default=str(DEFAULT_INDEX_OUTPUT))
    parser.add_argument("--index-markdown-output", default=str(DEFAULT_INDEX_MARKDOWN_OUTPUT))
    parser.add_argument("--public-output", default=str(DEFAULT_PUBLIC_BENCHMARK_OUTPUT))
    parser.add_argument(
        "--public-json-output",
        default=str(DEFAULT_PUBLIC_BENCHMARK_JSON_OUTPUT),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable validation summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issues = validate_claim_evidence(
        output=Path(args.output),
        index_markdown_output=Path(args.index_markdown_output),
        index_output=Path(args.index_output),
        json_output=Path(args.json_output),
        public_benchmark_output=Path(args.public_output),
        public_benchmark_json_output=Path(args.public_json_output),
    )
    if args.json:
        print(json.dumps({"issue_count": len(issues), "issues": issues}, indent=2))
    elif issues:
        for issue in issues:
            print(f"ERROR: {issue}", file=sys.stderr)
    else:
        print("Diffusion claim evidence validation passed.")
    return 1 if issues else 0


def validate_claim_evidence(
    *,
    output: Path = DEFAULT_OUTPUT,
    json_output: Path = DEFAULT_JSON_OUTPUT,
    index_output: Path = DEFAULT_INDEX_OUTPUT,
    index_markdown_output: Path = DEFAULT_INDEX_MARKDOWN_OUTPUT,
    claim_specs: tuple[ClaimSpec, ...] = DEFAULT_CLAIMS,
    canonical_slots: tuple[CanonicalClaimSlot, ...] = DEFAULT_CANONICAL_SLOTS,
    public_benchmark_output: Path | None = None,
    public_benchmark_json_output: Path | None = None,
    stale_doc_paths: tuple[Path, ...] = DEFAULT_DOC_PATHS,
) -> list[str]:
    issues: list[str] = []
    duplicate_ids = _duplicate_claim_ids(claim_specs)
    if duplicate_ids:
        issues.append(f"Duplicate claim IDs: {', '.join(duplicate_ids)}")

    claims: list[ClaimEvidence] = []
    for spec in claim_specs:
        try:
            evidence = build_claim_evidence(spec)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            issues.append(f"{spec.claim_id}: missing or invalid evidence: {exc}")
            continue
        claims.append(evidence)
        issues.extend(_validate_source_scores(spec, evidence))

    if not issues:
        public_benchmark_output = public_benchmark_output or output.parent / DEFAULT_PUBLIC_BENCHMARK_OUTPUT.name
        public_benchmark_json_output = (
            public_benchmark_json_output
            or json_output.parent / DEFAULT_PUBLIC_BENCHMARK_JSON_OUTPUT.name
        )
        issues.extend(_validate_generated_json(json_output, claims))
        issues.extend(_validate_generated_markdown(output, claims))
        issues.extend(_validate_ground_truth_index(index_output, index_markdown_output, claims, canonical_slots))
        issues.extend(
            _validate_public_benchmark_outputs(
                public_benchmark_output,
                public_benchmark_json_output,
                claims,
            )
        )
        issues.extend(_validate_public_doc_artifacts(index_output, stale_doc_paths))
    return issues


def _validate_source_scores(spec: ClaimSpec, evidence: ClaimEvidence) -> list[str]:
    issues: list[str] = []
    scores = json.loads(spec.scores_path.read_text(encoding="utf-8"))
    issues.extend(_validate_required_score_keys(spec.claim_id, scores))
    arms = scores.get("arms")
    if not isinstance(arms, dict):
        return [f"{spec.claim_id}: scores file is missing arms object"]
    for arm in ("fixed", "random", "repair_selected"):
        if arm not in arms:
            issues.append(f"{spec.claim_id}: scores file is missing {arm} arm")
    if issues:
        return issues

    repair_arm = arms["repair_selected"]
    fixed_arm = arms["fixed"]
    if not isinstance(repair_arm, dict) or not isinstance(fixed_arm, dict):
        return [f"{spec.claim_id}: fixed/repair arms must be objects"]

    repair_count = _int_value(repair_arm.get("count"))
    full_count = _int_value((arms.get("trajectory_selected") or fixed_arm).get("count"))
    eligible_count = _int_value(scores.get("repair_eligible_task_count"))
    all_generation_count = _int_value(scores.get("all_generation_count"))
    if repair_count <= 0:
        issues.append(f"{spec.claim_id}: repair count must be positive")
    if full_count <= 0:
        issues.append(f"{spec.claim_id}: full task count must be positive")
    if repair_count > full_count:
        issues.append(
            f"{spec.claim_id}: repair count {repair_count} exceeds full count {full_count}"
        )
    if eligible_count < repair_count:
        issues.append(
            f"{spec.claim_id}: eligible count {eligible_count} is below repair count {repair_count}"
        )
    if eligible_count > full_count:
        issues.append(
            f"{spec.claim_id}: eligible count {eligible_count} exceeds full count {full_count}"
        )
    if all_generation_count < full_count:
        issues.append(
            f"{spec.claim_id}: all_generation_count {all_generation_count} is below full count {full_count}"
        )

    issues.extend(_validate_deltas(spec.claim_id, scores, evidence))
    issues.extend(_validate_wins(spec.claim_id, scores, repair_count))
    issues.extend(_validate_oracle_headroom(spec.claim_id, scores, evidence))
    issues.extend(_validate_raw_line_count(spec.claim_id, Path(evidence.raw_path), all_generation_count))
    issues.extend(
        _validate_repair_diagnostic_requirements(
            spec.claim_id,
            scores,
            spec.required_repair_diagnostics,
        )
    )
    return issues


def _validate_required_score_keys(claim_id: str, scores: dict[str, object]) -> list[str]:
    required_keys = (
        "all_generation_count",
        "exact_task_trajectory_policy",
        "oracle_headroom_vs_repair",
        "repair_eligible_task_count",
        "repair_generation_budget_delta_vs_evolved",
        "repair_pack",
        "repair_task_delta_per_extra_generation_vs_evolved",
        "repair_task_delta_vs_evolved",
        "repair_task_delta_vs_fixed",
        "repair_task_delta_vs_random",
        "repair_wins_vs_evolved",
        "repair_wins_vs_fixed",
        "repair_wins_vs_random",
    )
    return [
        f"{claim_id}: scores file is missing required key {key}"
        for key in required_keys
        if key not in scores
    ]


def _validate_deltas(
    claim_id: str,
    scores: dict[str, object],
    evidence: ClaimEvidence,
) -> list[str]:
    issues: list[str] = []
    repair_score = evidence.repair_score
    expected_fixed = repair_score - _float_value(scores.get("repair_task_delta_vs_fixed"))
    expected_random = repair_score - _float_value(scores.get("repair_task_delta_vs_random"))
    if not _close(expected_fixed, evidence.fixed_repair_slice_score):
        issues.append(f"{claim_id}: fixed repair slice does not match repair delta")
    if not _close(expected_random, evidence.random_repair_slice_score):
        issues.append(f"{claim_id}: random repair slice does not match repair delta")

    delta_vs_evolved = _optional_float(scores.get("repair_task_delta_vs_evolved"))
    budget_delta = _optional_float(scores.get("repair_generation_budget_delta_vs_evolved"))
    gain = _optional_float(scores.get("repair_task_delta_per_extra_generation_vs_evolved"))
    if delta_vs_evolved is not None and delta_vs_evolved <= 0.0:
        issues.append(f"{claim_id}: promoted repair delta vs evolved must be positive")
    if budget_delta is not None and budget_delta <= 0.0:
        issues.append(f"{claim_id}: repair budget delta vs evolved must be positive")
    if delta_vs_evolved is not None and budget_delta is not None and gain is not None:
        expected_gain = delta_vs_evolved / budget_delta
        if not _close(gain, expected_gain, tolerance=1e-6):
            issues.append(
                f"{claim_id}: gain per extra generation {gain} != delta/budget {expected_gain}"
            )
    return issues


def _validate_wins(
    claim_id: str,
    scores: dict[str, object],
    repair_count: int,
) -> list[str]:
    issues: list[str] = []
    for key in ("repair_wins_vs_fixed", "repair_wins_vs_random", "repair_wins_vs_evolved"):
        value = scores.get(key)
        if not isinstance(value, dict):
            issues.append(f"{claim_id}: {key} must be an object")
            continue
        total = sum(_int_value(value.get(part)) for part in ("wins", "ties", "losses"))
        if total != repair_count:
            issues.append(f"{claim_id}: {key} total {total} != repair count {repair_count}")
    return issues


def _validate_oracle_headroom(
    claim_id: str,
    scores: dict[str, object],
    evidence: ClaimEvidence,
) -> list[str]:
    oracle_score = _optional_float(scores.get("oracle_task_score"))
    headroom = _optional_float(scores.get("oracle_headroom_vs_repair"))
    if headroom is None:
        return []
    if headroom < -FLOAT_TOLERANCE:
        return [f"{claim_id}: oracle headroom must be non-negative"]
    if oracle_score is None or evidence.repair_count != evidence.full_count:
        return []
    expected = oracle_score - evidence.repair_score
    if expected < -FLOAT_TOLERANCE:
        return [f"{claim_id}: oracle task score is below repair score"]
    if not _close(headroom, expected, tolerance=1e-6):
        return [f"{claim_id}: oracle headroom {headroom} != oracle-repair {expected}"]
    return []


def _validate_raw_line_count(
    claim_id: str,
    raw_path: Path,
    all_generation_count: int,
) -> list[str]:
    raw_lines = _nonempty_line_count(raw_path)
    if raw_lines < all_generation_count:
        return [
            f"{claim_id}: raw line count {raw_lines} is below all_generation_count {all_generation_count}"
        ]
    return []


def _validate_repair_diagnostic_requirements(
    claim_id: str,
    scores: dict[str, object],
    requirements: tuple[RepairDiagnosticRequirement, ...],
) -> list[str]:
    if not requirements:
        return []
    summary = scores.get("repair_candidate_summary")
    if not isinstance(summary, dict):
        return [f"{claim_id}: scores file is missing repair_candidate_summary"]

    issues: list[str] = []
    for requirement in requirements:
        candidate = summary.get(requirement.repair_name)
        label = f"{requirement.repair_name}.{requirement.metric}"
        if not isinstance(candidate, dict):
            issues.append(f"{claim_id}: repair diagnostic candidate is missing: {requirement.repair_name}")
            continue
        value = candidate.get(requirement.metric)
        if value is None:
            issues.append(f"{claim_id}: repair diagnostic metric is missing: {label}")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            issues.append(f"{claim_id}: repair diagnostic metric is not numeric: {label}")
            continue
        if requirement.min_value is not None and numeric < requirement.min_value - FLOAT_TOLERANCE:
            issues.append(
                f"{claim_id}: repair diagnostic {label}={numeric} "
                f"is below required minimum {requirement.min_value}"
            )
        if requirement.max_value is not None and numeric > requirement.max_value + FLOAT_TOLERANCE:
            issues.append(
                f"{claim_id}: repair diagnostic {label}={numeric} "
                f"exceeds required maximum {requirement.max_value}"
            )
    return issues


def _validate_generated_json(json_output: Path, claims: list[ClaimEvidence]) -> list[str]:
    if not json_output.exists():
        return [f"Missing generated JSON claim map: {json_output}"]
    actual = json.loads(json_output.read_text(encoding="utf-8"))
    expected = _json_ready([asdict(claim) for claim in claims])
    if actual != expected:
        return [f"Generated JSON claim map is stale or inconsistent: {json_output}"]
    return []


def _validate_generated_markdown(output: Path, claims: list[ClaimEvidence]) -> list[str]:
    if not output.exists():
        return [f"Missing generated Markdown claim map: {output}"]
    expected = render_markdown(claims)
    if output.read_text(encoding="utf-8") != expected:
        return [f"Generated Markdown claim map is stale or inconsistent: {output}"]
    return []


def _validate_ground_truth_index(
    index_output: Path,
    index_markdown_output: Path,
    claims: list[ClaimEvidence],
    canonical_slots: tuple[CanonicalClaimSlot, ...],
) -> list[str]:
    issues: list[str] = []
    try:
        expected_index = build_ground_truth_index(claims, canonical_slots=canonical_slots)
    except ValueError as exc:
        return [str(exc)]
    if not index_output.exists():
        issues.append(f"Missing generated ground truth index JSON: {index_output}")
    else:
        actual = json.loads(index_output.read_text(encoding="utf-8"))
        if actual != expected_index:
            issues.append(f"Generated ground truth index JSON is stale or inconsistent: {index_output}")
    if not index_markdown_output.exists():
        issues.append(f"Missing generated ground truth index Markdown: {index_markdown_output}")
    else:
        expected_markdown = render_ground_truth_index_markdown(expected_index)
        if index_markdown_output.read_text(encoding="utf-8") != expected_markdown:
            issues.append(
                "Generated ground truth index Markdown is stale or inconsistent: "
                f"{index_markdown_output}"
            )
    return issues


def _validate_public_benchmark_outputs(
    public_output: Path,
    public_json_output: Path,
    claims: list[ClaimEvidence],
) -> list[str]:
    issues: list[str] = []
    try:
        expected_summary = build_public_benchmark_summary(claims)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [f"Could not build public benchmark summary: {exc}"]
    if not public_json_output.exists():
        issues.append(f"Missing generated public benchmark JSON: {public_json_output}")
    else:
        actual = json.loads(public_json_output.read_text(encoding="utf-8"))
        if actual != expected_summary:
            issues.append(
                f"Generated public benchmark JSON is stale or inconsistent: {public_json_output}"
            )
    if not public_output.exists():
        issues.append(f"Missing generated public benchmark Markdown: {public_output}")
    else:
        expected_markdown = render_public_benchmark_markdown(expected_summary)
        actual_markdown = public_output.read_text(encoding="utf-8")
        if actual_markdown != expected_markdown:
            issues.append(
                f"Generated public benchmark Markdown is stale or inconsistent: {public_output}"
            )
        issues.extend(_validate_public_benchmark_language(public_output, actual_markdown))
    return issues


def _validate_public_benchmark_language(public_output: Path, text: str) -> list[str]:
    lowered = text.lower()
    issues: list[str] = []
    for internal_term in ("evolved", "oracle", "trajectory"):
        if internal_term in lowered:
            issues.append(
                f"Public benchmark Markdown exposes internal diagnostic term "
                f"{internal_term!r}: {public_output}"
            )
    return issues


def _validate_public_doc_artifacts(index_output: Path, doc_paths: tuple[Path, ...]) -> list[str]:
    if not doc_paths:
        return []
    try:
        issues = scan_stale_diffusion_docs(index_path=index_output, doc_paths=doc_paths)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [f"Could not scan public docs for stale diffusion artifacts: {exc}"]
    return [_format_stale_doc_issue(issue) for issue in issues]


def _format_stale_doc_issue(issue: StaleDiffusionDocIssue) -> str:
    return (
        f"{issue.path}:{issue.line}: {issue.reason}: {issue.artifact}"
    )


def _duplicate_claim_ids(claim_specs: tuple[ClaimSpec, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for spec in claim_specs:
        if spec.claim_id in seen:
            duplicates.append(spec.claim_id)
        seen.add(spec.claim_id)
    return duplicates


def _json_ready(value: object) -> object:
    return json.loads(json.dumps(value))


def _nonempty_line_count(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _int_value(value: object) -> int:
    return int(value or 0)


def _float_value(value: object) -> float:
    return float(value or 0.0)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _close(left: float, right: float, *, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return abs(left - right) <= tolerance


if __name__ == "__main__":
    raise SystemExit(main())
