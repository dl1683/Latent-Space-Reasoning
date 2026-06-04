"""Score an ARC-3 mechanistic pipeline run from its manifest and audit.

This is not an ARC-3 benchmark score. It is an internal progress metric for the
mechanistic reasoning substrate: did the run produce validated reusable rules,
did it expose contradictions, and did it produce repair requests?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.audit_arc3_mechanistic_manifest import audit_manifest


@dataclass(frozen=True)
class MechanisticRunScore:
    manifest: str
    audit_passed: bool
    transitions: int
    candidate_rules: int
    validated_rules: int
    rejected_rules: int
    contradictions: int
    contextual_rules: int
    contextual_rule_checks: int
    contextual_validated_rules: int
    contextual_rejected_rules: int
    contextual_contradictions: int
    repairs: int
    reuse_ratio: float
    contradiction_rate: float
    contextual_contradiction_rate: float
    status: str
    blockers: list[str]


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected manifest object: {path}")
    return payload


def _count(manifest: dict[str, Any], field: str) -> int:
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        return 0
    value = counts.get(field, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def score_run(manifest_path: Path) -> MechanisticRunScore:
    manifest = _read_manifest(manifest_path)
    findings = audit_manifest(manifest_path)
    blockers = [f"{finding.field}: {finding.detail}" for finding in findings if finding.status == "fail"]
    audit_passed = not blockers
    candidate_rules = _count(manifest, "candidate_rules")
    validated_rules = _count(manifest, "validated_rules")
    rejected_rules = _count(manifest, "rejected_rules")
    contradictions = _count(manifest, "contradictions")
    contextual_rules = _count(manifest, "contextual_rules")
    contextual_rule_checks = _count(manifest, "contextual_rule_checks")
    contextual_validated_rules = _count(manifest, "contextual_validated_rules")
    contextual_rejected_rules = _count(manifest, "contextual_rejected_rules")
    contextual_contradictions = _count(manifest, "contextual_contradictions")
    rule_checks = _count(manifest, "rule_checks")
    repairs = _count(manifest, "repairs")
    reuse_ratio = validated_rules / candidate_rules if candidate_rules else 0.0
    contradiction_rate = contradictions / rule_checks if rule_checks else 0.0
    contextual_contradiction_rate = (
        contextual_contradictions / contextual_rule_checks if contextual_rule_checks else 0.0
    )

    if not audit_passed:
        status = "invalid"
    elif validated_rules:
        status = "reusable"
    elif contradictions or repairs:
        status = "needs_repair"
    else:
        status = "observed_only"

    return MechanisticRunScore(
        manifest=str(manifest_path),
        audit_passed=audit_passed,
        transitions=_count(manifest, "transitions"),
        candidate_rules=candidate_rules,
        validated_rules=validated_rules,
        rejected_rules=rejected_rules,
        contradictions=contradictions,
        contextual_rules=contextual_rules,
        contextual_rule_checks=contextual_rule_checks,
        contextual_validated_rules=contextual_validated_rules,
        contextual_rejected_rules=contextual_rejected_rules,
        contextual_contradictions=contextual_contradictions,
        repairs=repairs,
        reuse_ratio=reuse_ratio,
        contradiction_rate=contradiction_rate,
        contextual_contradiction_rate=contextual_contradiction_rate,
        status=status,
        blockers=blockers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score = score_run(args.manifest)
    text = json.dumps(asdict(score), indent=2 if args.pretty else None, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    if not score.audit_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
