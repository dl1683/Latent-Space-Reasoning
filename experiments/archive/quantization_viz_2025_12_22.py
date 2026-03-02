from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Manual answer-quality evaluation from tests 34-43 (best decode, 2048 tokens).
    # Counts are 4bit wins / ties / none wins per model.
    models = ["Qwen3-0.6B", "Qwen3-1.7B", "Total"]
    wins_4bit = np.array([2, 3, 5])
    ties = np.array([5, 5, 10])
    wins_none = np.array([3, 2, 5])

    # Efficiency is theoretical weight memory only (relative to fp16/none).
    eff_labels = ["fp16/none", "4bit"]
    eff_values = [1.0, 0.25]

    x = np.arange(len(models))

    fig, (ax_quality, ax_eff) = plt.subplots(
        2,
        1,
        figsize=(11, 8.5),
        gridspec_kw={"height_ratios": [2.2, 1]},
        constrained_layout=True,
    )

    # Quality panel (stacked bars).
    ax_quality.bar(x, wins_4bit, label="4bit wins", color="#2E8B57")
    ax_quality.bar(x, ties, bottom=wins_4bit, label="ties", color="#B0B0B0")
    ax_quality.bar(
        x,
        wins_none,
        bottom=wins_4bit + ties,
        label="none wins",
        color="#B22222",
    )
    ax_quality.set_title(
        "4bit vs none: answer-quality outcomes (best decode, tests 34-43)"
    )
    ax_quality.set_xticks(x, models)
    ax_quality.set_ylabel("Count of tests")
    ax_quality.set_ylim(0, 10.5)
    ax_quality.legend(loc="upper right", frameon=False)

    for i, total in enumerate(wins_4bit + ties + wins_none):
        ax_quality.text(i, total + 0.2, f"{int(total)} tests", ha="center", va="bottom")

    # Efficiency panel.
    ax_eff.bar(eff_labels, eff_values, color=["#4F81BD", "#9ACD32"])
    ax_eff.set_title("Efficiency: relative weight memory (theoretical)")
    ax_eff.set_ylabel("Relative size (fp16 = 1.0)")
    ax_eff.set_ylim(0, 1.1)
    ax_eff.text(
        1,
        eff_values[1] + 0.05,
        "approx 4x smaller",
        ha="center",
        va="bottom",
    )

    fig.suptitle(
        "Quantization tradeoff: quality parity with 4bit and large memory savings",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        "Notes: Quality is manual, answer-based judgment on tests 34-43. "
        "Efficiency is weight memory only (theoretical); actual savings depend on runtime.",
        ha="left",
        va="top",
        fontsize=9,
    )

    out_dir = Path("experiments") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "quantization_4bit_vs_none_2025_12_22.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
