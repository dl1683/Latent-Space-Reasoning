"""Run a self-contained smoke test for the offline ARC-3 mechanistic pipeline.

This creates a tiny synthetic replay artifact and exercises the full cheap path:
pipeline, manifest audit, run score, and abstract planner evaluation.
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
from experiments.evaluate_arc3_rule_planner import evaluate_planner
from experiments.run_arc3_mechanistic_pipeline import run_pipeline
from experiments.score_arc3_mechanistic_run import score_run


@dataclass(frozen=True)
class MechanisticSmokeResult:
    passed: bool
    replay: str
    manifest: str
    score: dict[str, Any]
    planner_evaluation: dict[str, Any]
    audit_failures: list[str]


def _write_synthetic_replay(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "level": "synthetic",
                "trace": [
                    {
                        "step": 0,
                        "action": "enter_shape_pad",
                        "state_before": {"shape": 0, "color": 1, "position": [1, 1]},
                        "state_after": {"shape": 5, "color": 1, "position": [1, 1]},
                    },
                    {
                        "step": 1,
                        "action": "enter_shape_pad",
                        "state_before": {"shape": 0, "color": 1, "position": [2, 1]},
                        "state_after": {"shape": 5, "color": 1, "position": [2, 1]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def run_smoke(output_dir: Path) -> MechanisticSmokeResult:
    replay_path = output_dir / "synthetic_replay.json"
    pipeline_dir = output_dir / "pipeline"
    _write_synthetic_replay(replay_path)
    manifest = run_pipeline(replay_path, pipeline_dir, min_support=2, pretty=True)
    manifest_path = pipeline_dir / "manifest.json"
    audit_findings = audit_manifest(manifest_path)
    audit_failures = [f"{finding.field}: {finding.detail}" for finding in audit_findings if finding.status == "fail"]
    score = score_run(manifest_path)
    validated_rules = json.loads(Path(manifest["outputs"]["validated_rules"]).read_text(encoding="utf-8"))
    planner_eval = evaluate_planner(
        validated_rules,
        [
            {
                "id": "shape-goal",
                "initial_state": {"shape": 0, "color": 1},
                "goal_state": {"shape": 5},
                "expected_solved": True,
                "expected_actions": ["enter_shape_pad"],
            }
        ],
        max_depth=2,
    )
    passed = (
        not audit_failures
        and score.audit_passed
        and score.status == "reusable"
        and planner_eval.solved == 1
        and planner_eval.expected_solved_matches == 1
        and planner_eval.action_matches == 1
    )
    result = MechanisticSmokeResult(
        passed=passed,
        replay=str(replay_path),
        manifest=str(manifest_path),
        score=asdict(score),
        planner_evaluation=asdict(planner_eval),
        audit_failures=audit_failures,
    )
    (output_dir / "smoke_result.json").write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/mechanistic_rules/smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_smoke(args.output_dir)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
