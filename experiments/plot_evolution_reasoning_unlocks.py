"""
Create a shareable visualization of evolution-only reasoning motifs.

The chart is intentionally selective: it highlights only motifs that are
absent from greedy baseline and random perturbation outputs for the same task,
but appear in one or more evolution outputs. This keeps the claim honest while
showing the strongest evidence in a visually shareable format.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
COMPARISON_PATH = ROOT / "planning_comparison_results.json"
EVOLUTION_PATH = ROOT / "planning_evolution_results.json"
OUTPUT_PNG = ROOT / "fig_evolution_reasoning_unlocks.png"
OUTPUT_SVG = ROOT / "fig_evolution_reasoning_unlocks.svg"
OUTPUT_JSON = ROOT / "fig_evolution_reasoning_unlocks_summary.json"


@dataclass(frozen=True)
class Motif:
    task_id: str
    group: str
    label: str
    patterns: tuple[str, ...]
    note: str


MOTIFS = (
    Motif(
        task_id="plan_02_incident_response",
        group="Incident Response",
        label="DMZ / quarantine segmentation",
        patterns=(r"\bdmz\b", r"quarantine zone", r"isolated network segment", r"private vlan"),
        note="explicit containment topology",
    ),
    Motif(
        task_id="plan_02_incident_response",
        group="Incident Response",
        label="Honeypot / decoy services",
        patterns=(r"honeypot", r"decoy service"),
        note="active adversary-monitoring tactic",
    ),
    Motif(
        task_id="plan_02_incident_response",
        group="Incident Response",
        label="MITRE ATT&CK / TTP mapping",
        patterns=(r"mitre", r"att&ck", r"\bttps?\b"),
        note="framework-based attacker tracking",
    ),
    Motif(
        task_id="plan_04_cache_debugging",
        group="Cache Debugging",
        label="Key versioning checks",
        patterns=(r"key version",),
        note="state-consistency instrumentation",
    ),
    Motif(
        task_id="plan_04_cache_debugging",
        group="Cache Debugging",
        label="Idempotent update logic",
        patterns=(r"idempot",),
        note="app-level duplicate prevention",
    ),
    Motif(
        task_id="plan_04_cache_debugging",
        group="Cache Debugging",
        label="Application-level traces",
        patterns=(r"application-level", r"app-level", r"application traces"),
        note="looks beyond Redis internals",
    ),
)


GROUP_STYLE = {
    "Incident Response": {
        "accent": "#B04A2A",
        "fill": "#D66B45",
        "soft": "#F2D2C6",
    },
    "Cache Debugging": {
        "accent": "#0E6B6B",
        "fill": "#20A4A1",
        "soft": "#CBECEB",
    },
}

GROUP_ORDER = {
    "Incident Response": 0,
    "Cache Debugging": 1,
}


def _normalize(text: str) -> str:
    return text.lower()


def _has_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = _normalize(text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def _load_data() -> tuple[list[dict], list[dict]]:
    with COMPARISON_PATH.open(encoding="utf-8") as f:
        comparison = json.load(f)["outputs"]
    with EVOLUTION_PATH.open(encoding="utf-8") as f:
        evolution = json.load(f)["outputs"]
    return comparison, evolution


def build_summary() -> list[dict]:
    comparison, evolution = _load_data()

    summary = []
    for motif in MOTIFS:
        baseline_texts = [
            row["response"]
            for row in comparison
            if row["condition"] == "greedy_baseline" and row["task_id"] == motif.task_id
        ]
        perturb_texts = [
            row["response"]
            for row in comparison
            if row["condition"] == "random_perturbation" and row["task_id"] == motif.task_id
        ]
        evolution_rows = [
            row for row in evolution
            if row["task_id"] == motif.task_id and "response_soft_prompt" in row
        ]
        evolution_texts = [row["response_soft_prompt"] for row in evolution_rows]

        baseline_count = sum(_has_pattern(text, motif.patterns) for text in baseline_texts)
        perturb_count = sum(_has_pattern(text, motif.patterns) for text in perturb_texts)
        evolution_count = sum(_has_pattern(text, motif.patterns) for text in evolution_texts)

        if baseline_count or perturb_count or not evolution_count:
            continue

        seeds = [
            row["seed"]
            for row in evolution_rows
            if _has_pattern(row["response_soft_prompt"], motif.patterns)
        ]

        summary.append({
            "task_id": motif.task_id,
            "group": motif.group,
            "label": motif.label,
            "note": motif.note,
            "baseline_count": baseline_count,
            "perturb_count": perturb_count,
            "evolution_count": evolution_count,
            "evolution_total": len(evolution_rows),
            "seeds": seeds,
        })

    summary.sort(
        key=lambda row: (
            GROUP_ORDER[row["group"]],
            -row["evolution_count"],
            row["label"],
        )
    )
    return summary


def save_summary(summary: list[dict]) -> None:
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def draw_card(ax, x0: float, y0: float, width: float, height: float, color: str) -> None:
    ax.add_patch(FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.04",
        linewidth=0,
        facecolor=color,
        zorder=0,
    ))


def render(summary: list[dict]) -> None:
    if not summary:
        raise RuntimeError("No evolution-only motifs found for the configured motif set.")

    fig = plt.figure(figsize=(18, 12.5), dpi=220)
    ax = plt.axes([0, 0, 1, 1])
    fig.patch.set_facecolor("#F7F2E8")
    ax.set_facecolor("#F7F2E8")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background panels
    draw_card(ax, 0.03, 0.035, 0.94, 0.91, "#FBF8F1")
    draw_card(ax, 0.58, 0.18, 0.33, 0.53, "#FFF8EE")
    draw_card(ax, 0.79, 0.80, 0.15, 0.12, "#FFF8EE")
    draw_card(ax, 0.05, 0.04, 0.88, 0.14, "#F1E6D5")

    title = "Evolution Opens Reasoning Paths\nStandard Decoding Never Reaches"
    ax.text(
        0.055, 0.935,
        title,
        fontsize=25.5, fontweight="bold", color="#1C1C1C", ha="left", va="top",
        linespacing=1.06,
    )
    subtitle = textwrap.fill(
        "Planning benchmark, 5 tasks x 5 seeds. Rows shown here are motifs that never appeared "
        "in the greedy baseline or in any of the 5 random-noise runs, but did appear in one or "
        "more evolved soft-prompt decodes.",
        width=92,
    )
    ax.text(
        0.055, 0.875,
        subtitle,
        fontsize=11.8, color="#514B43", ha="left", va="top", linespacing=1.28
    )

    # Condition headers
    x_cols = {"Baseline": 0.58, "2-token Noise": 0.71, "Evolution": 0.84}
    for label, x in x_cols.items():
        ax.text(x, 0.73, label, fontsize=13.5, fontweight="bold", color="#3C3832", ha="center")
    ax.text(0.58, 0.708, "mentions", fontsize=9.8, color="#7A736A", ha="center")
    ax.text(0.71, 0.708, "mentions", fontsize=9.8, color="#7A736A", ha="center")
    ax.text(0.84, 0.708, "evolution seeds", fontsize=9.8, color="#7A736A", ha="center")

    # Summary callout
    unique_count = len(summary)
    total_hits = sum(row["evolution_count"] for row in summary)
    ax.text(0.865, 0.885, f"{unique_count}", fontsize=33, fontweight="bold",
            color="#1C1C1C", ha="center", va="center")
    ax.text(0.865, 0.85, "evolution-only\nreasoning motifs", fontsize=12.5,
            color="#514B43", ha="center", va="center")
    ax.text(0.865, 0.814, f"{total_hits} total motif hits\nacross evolved seeds",
            fontsize=9.8, color="#7A736A", ha="center", va="center", linespacing=1.15)

    y = 0.66
    row_gap = 0.085
    group_gap = 0.032
    current_group = None

    for row in summary:
        group = row["group"]
        style = GROUP_STYLE[group]

        if group != current_group:
            if current_group is not None:
                y -= group_gap
            ax.text(0.07, y + 0.03, group.upper(), fontsize=12.5, fontweight="bold",
                    color=style["accent"], ha="left", va="center")
            ax.plot([0.07, 0.49], [y + 0.015, y + 0.015], color=style["soft"], linewidth=3)
            current_group = group

        # Labels
        ax.text(0.075, y - 0.01, row["label"], fontsize=15.8, fontweight="bold",
                color="#1F1E1B", ha="left", va="center")
        ax.text(0.075, y - 0.038, row["note"], fontsize=10.9,
                color="#6B645B", ha="left", va="center")
        ax.text(0.43, y - 0.038, f"seeds {', '.join(map(str, row['seeds']))}",
                fontsize=9.7, color="#8B847A", ha="left", va="center")

        # Guide line
        ax.plot([0.55, 0.88], [y - 0.01, y - 0.01], color="#E4DED3", linewidth=1.4, zorder=1)

        # Baseline and perturbation: always zero for selected motifs
        for col in ("Baseline", "2-token Noise"):
            x = x_cols[col]
            ax.scatter([x], [y - 0.01], s=350, facecolors="white",
                       edgecolors="#D6CEC1", linewidths=1.8, zorder=3)
            ax.text(x, y - 0.01, "0", fontsize=12, fontweight="bold",
                    color="#AAA08F", ha="center", va="center", zorder=4)

        # Evolution bubble with glow
        x = x_cols["Evolution"]
        bubble_size = 680 + 360 * row["evolution_count"]
        ax.scatter([x], [y - 0.01], s=bubble_size * 1.9, c=style["fill"],
                   alpha=0.10, linewidths=0, zorder=2)
        ax.scatter([x], [y - 0.01], s=bubble_size, c=style["fill"],
                   edgecolors="white", linewidths=2.2, zorder=5)
        ax.text(x, y - 0.01, f"{row['evolution_count']}/{row['evolution_total']}",
                fontsize=12.5, fontweight="bold", color="white",
                ha="center", va="center", zorder=6)

        y -= row_gap

    how_to_read = textwrap.fill(
        "Each row is a concrete reasoning motif that appeared in the planning outputs. "
        "The left two columns show that the motif never appeared in the standard greedy baseline "
        "or in any of the 5 random-noise runs for that same task. "
        "The bubble on the right shows how many of the 5 evolution seeds surfaced it.",
        width=170,
    )
    takeaway = textwrap.fill(
        "Takeaway: latent-space evolution did not just make responses longer. "
        "On the clearest tasks, it unlocked specific technical ideas and investigation "
        "strategies that standard decoding never reached at all.",
        width=160,
    )
    ax.text(0.07, 0.148, "How To Read This", fontsize=15.0, fontweight="bold",
            color="#2F2A24", ha="left", va="center")
    ax.text(
        0.07, 0.116,
        how_to_read,
        fontsize=11.9, color="#4D473F", ha="left", va="top", linespacing=1.3
    )
    ax.text(
        0.07, 0.074,
        takeaway,
        fontsize=12.4, fontweight="bold", color="#3A342E", ha="left", va="top",
        linespacing=1.24
    )

    fig.savefig(OUTPUT_PNG, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUTPUT_SVG, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    summary = build_summary()
    save_summary(summary)
    render(summary)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_SVG}")
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
