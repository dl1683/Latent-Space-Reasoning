"""Run a small diffusion schedule sweep and score denoising trajectories.

This is the first diffusion-native search surface: schedules are candidates and
trajectory summaries provide the cheap selector signal before external judges.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.diffusion import (  # noqa: E402
    HFDiffusionBackend,
    attach_control_score,
    default_dream_schedules,
    default_llada_schedules,
    get_candidate,
    is_llada_family,
)

DEFAULT_PROMPT = (
    "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
    "the other tests a risky reasoning intervention. Decide which measurements to "
    "collect so tomorrow's result is publishable even if the intervention fails."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="dream-7b-instruct-hf")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-jsonl", default="eval_results/diffusion_language/schedule_sweep_raw.jsonl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = get_candidate(args.candidate)
    if candidate.backend != "hf_custom":
        raise SystemExit(f"{candidate.key} uses {candidate.backend}; schedule sweep expects hf_custom.")

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 32 if is_llada_family(candidate.family) else 64
    schedules = (
        default_llada_schedules(max_new_tokens)
        if is_llada_family(candidate.family)
        else default_dream_schedules(max_new_tokens)
    )
    if args.limit is not None:
        schedules = schedules[: args.limit]

    backend = HFDiffusionBackend(
        args.candidate,
        device=args.device,
        dtype=args.dtype,
        model_path=args.model_path,
    )
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for schedule in schedules:
        result = backend.generate(args.prompt, config=schedule.to_config())
        record = result.to_dict()
        record["created_at"] = datetime.now(timezone.utc).isoformat()
        record["schedule"] = schedule.to_dict()
        record = attach_control_score(record)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        records.append(record)
        score = record["trajectory_control_score"]["overall"]
        print(f"{schedule.name}: score={score:.3f} text={_preview(record.get('text', ''))}")

    best = max(records, key=lambda item: item["trajectory_control_score"]["overall"])
    print(
        json.dumps(
            {
                "records": len(records),
                "best_schedule": best["schedule"]["name"],
                "best_score": best["trajectory_control_score"]["overall"],
                "output_jsonl": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _preview(value: object, limit: int = 100) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
