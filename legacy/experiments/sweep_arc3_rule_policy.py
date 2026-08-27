"""Sweep ARC-3 learned-rule policy evaluation across train fractions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_arc3_rule_policy import RulePolicyScore, evaluate_rule_policy


@dataclass(frozen=True)
class RulePolicySweep:
    input: str
    fractions: list[float]
    runs: list[RulePolicyScore]
    mean_top1_action_accuracy: float
    mean_frequency_baseline_accuracy: float
    mean_top1_lift_over_frequency: float
    mean_modeled_transition_accuracy: float
    min_top1_action_accuracy: float
    min_modeled_transition_accuracy: float


def sweep_rule_policy(
    input_path: Path,
    fractions: list[float],
    min_support: int = 2,
) -> RulePolicySweep:
    runs = [
        evaluate_rule_policy(input_path, train_fraction=fraction, min_support=min_support)
        for fraction in fractions
    ]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    top1 = [run.top1_action_accuracy for run in runs]
    baseline = [run.frequency_baseline_accuracy for run in runs]
    lift = [run.top1_lift_over_frequency for run in runs]
    modeled = [run.modeled_transition_accuracy for run in runs]

    return RulePolicySweep(
        input=str(input_path),
        fractions=fractions,
        runs=runs,
        mean_top1_action_accuracy=mean(top1),
        mean_frequency_baseline_accuracy=mean(baseline),
        mean_top1_lift_over_frequency=mean(lift),
        mean_modeled_transition_accuracy=mean(modeled),
        min_top1_action_accuracy=min(top1) if top1 else 0.0,
        min_modeled_transition_accuracy=min(modeled) if modeled else 0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score = sweep_rule_policy(args.input, fractions=args.fractions, min_support=args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(score), indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(asdict(score), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
