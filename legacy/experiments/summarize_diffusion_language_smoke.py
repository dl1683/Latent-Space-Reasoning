"""Summarize diffusion-language smoke JSONL runs.

Example:
    python experiments/summarize_diffusion_language_smoke.py \
        --input eval_results/diffusion_language/smoke_raw.jsonl \
        --output eval_results/diffusion_language/smoke_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="eval_results/diffusion_language/smoke_raw.jsonl")
    parser.add_argument("--output", default="eval_results/diffusion_language/smoke_report.md")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, object]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def render_report(records: list[dict[str, object]]) -> str:
    lines = [
        "# Diffusion Language Smoke Report",
        "",
        f"Records: `{len(records)}`",
        "",
    ]
    if not records:
        lines.append("No records found.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| Run | Candidate | Schedule | Score | Steps | Temp | History | First visible | First final | Mask-free | Final chars | Text |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for index, record in enumerate(records, start=1):
        config = _dict(record.get("config"))
        summary = _dict(record.get("trajectory_summary"))
        schedule = _dict(record.get("schedule"))
        control_score = _dict(record.get("trajectory_control_score"))
        text = _preview(str(record.get("text", "")))
        lines.append(
            "| "
            f"{index} | "
            f"{record.get('candidate_key', '')} | "
            f"{schedule.get('name', '-')} | "
            f"{_format_score(control_score.get('overall'))} | "
            f"{config.get('steps', '')} | "
            f"{config.get('temperature', '')} | "
            f"{record.get('history_steps', '')} | "
            f"{_none_dash(summary.get('first_visible_step'))} | "
            f"{_none_dash(summary.get('first_final_text_step'))} | "
            f"{_none_dash(summary.get('first_mask_free_step'))} | "
            f"{summary.get('final_visible_chars', '')} | "
            f"{text} |"
        )

    latest = records[-1]
    lines.extend(["", "## Latest Trajectory Samples", ""])
    for sample in _list(_dict(latest.get("trajectory_summary")).get("samples")):
        lines.append(
            f"- step `{sample.get('step')}`: masks `{sample.get('mask_count')}`, "
            f"eos `{sample.get('eos_count')}`, visible chars `{sample.get('visible_chars')}`"
        )
    return "\n".join(lines) + "\n"


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _none_dash(value: object) -> object:
    return "-" if value is None else value


def _format_score(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return "-"


def _preview(text: str, limit: int = 96) -> str:
    normalized = " ".join(text.replace("|", "/").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    records = load_records(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(records), encoding="utf-8")
    print(f"Wrote {output_path} from {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
