"""Generate figures for README: scaling ladder and temperature comparison."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def plot_scaling_vs_perturbation():
    models = ["1.7B", "4B", "8B\n(4-bit)", "8B\n(8-bit)", "14B", "32B"]
    baseline = [28, 32, 24, 16, 36, 0]
    pert_mean = [29, 52, 25, 29, 40, 0]
    params_b = [1.7, 4.0, 8.0, 8.0, 14.0, 32.0]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(models))
    w = 0.32

    bars1 = ax.bar(x - w/2, baseline, w, label="Baseline (greedy)", color="#5a7d9a", zorder=3)
    bars2 = ax.bar(x + w/2, pert_mean, w, label="Perturbation mean", color="#e07b54", zorder=3)

    ax.axhline(72, color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
    ax.text(len(models) - 0.5, 73.5, "4B perturbation\nplurality@10 = 72%",
            ha="right", va="bottom", fontsize=9, color="#c0392b", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_xlabel("Qwen3 model size")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Parameter scaling is flat; perturbation is not", fontweight="bold", pad=12)
    ax.set_ylim(0, 85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=8, color="#5a7d9a")
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=8, color="#e07b54")

    fig.tight_layout()
    fig.savefig(OUT / "scaling_vs_perturbation.png", bbox_inches="tight")
    fig.savefig(OUT / "scaling_vs_perturbation.svg", bbox_inches="tight")
    plt.close(fig)
    print("Saved scaling_vs_perturbation.png/.svg")


def plot_temperature_vs_perturbation():
    methods = ["Greedy\nbaseline", "Temp 0.3\n×10", "Temp 0.6\n×10", "Temp 0.9\n×10", "Perturbation\n×10"]
    mean_acc = [32, 38, 41, 39, 52]
    plurality = [None, 64, 60, 48, 72]
    oracle = [None, 88, 100, 96, 100]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))

    colors = ["#888888", "#7fb3d8", "#5a9bc7", "#3a7db5", "#e07b54"]

    bars = ax.bar(x, mean_acc, 0.5, color=colors, zorder=3, label="Mean accuracy")

    for i, (p, o) in enumerate(zip(plurality, oracle)):
        if p is not None:
            ax.plot(i, p, "D", color="#2c3e50", markersize=8, zorder=4)
            ax.plot(i, o, "^", color="#27ae60", markersize=8, zorder=4)

    ax.plot([], [], "D", color="#2c3e50", markersize=8, label="Plurality@10")
    ax.plot([], [], "^", color="#27ae60", markersize=8, label="Oracle@10")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Perturbation vs temperature sampling (same model, same cost)",
                 fontweight="bold", pad=12)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "temperature_vs_perturbation.png", bbox_inches="tight")
    fig.savefig(OUT / "temperature_vs_perturbation.svg", bbox_inches="tight")
    plt.close(fig)
    print("Saved temperature_vs_perturbation.png/.svg")


def plot_text_gen_judge():
    """Win/loss chart for text generation blind judge comparisons."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Comparison 1: perturbation vs 4B baseline
    ax = axes[0]
    categories = ["Perturbation\nwins", "Baseline\nwins", "Tie"]
    counts = [7, 1, 7]
    colors = ["#e07b54", "#5a7d9a", "#bbbbbb"]
    bars = ax.bar(categories, counts, color=colors, width=0.55, zorder=3, edgecolor="white", linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(int(h)),
                ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.set_ylabel("Tasks (out of 15)")
    ax.set_title("4B perturbation vs 4B baseline", fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Comparison 2: perturbation vs 14B baseline
    ax = axes[1]
    categories = ["4B pert\nwins", "14B base\nwins", "Tie"]
    counts = [5, 7, 3]
    colors = ["#e07b54", "#2a78d6", "#bbbbbb"]
    bars = ax.bar(categories, counts, color=colors, width=0.55, zorder=3, edgecolor="white", linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(int(h)),
                ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.set_ylabel("Tasks (out of 15)")
    ax.set_title("4B perturbation vs 14B baseline", fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Text Generation: Blind LLM-as-Judge Evaluation",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "text_gen_judge_results.png", bbox_inches="tight")
    fig.savefig(OUT / "text_gen_judge_results.svg", bbox_inches="tight")
    plt.close(fig)
    print("Saved text_gen_judge_results.png/.svg")


def plot_text_gen_by_category():
    """Heatmap showing which categories perturbation helps most."""
    categories = ["Reasoning", "Analysis", "Explanation", "Creative", "Debugging", "Planning"]
    # vs 4B baseline: wins / total in category
    vs_base = [2/3, 2/3, 2/3, 1/2, 1/2, 0/2]
    # vs 14B baseline: wins / total
    vs_14b = [2/3, 2/3, 1/3, 0/2, 0/2, 0/2]

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(categories))
    w = 0.32

    bars1 = ax.bar(x - w/2, [v * 100 for v in vs_base], w,
                   label="Pert win rate vs 4B baseline", color="#e07b54", zorder=3)
    bars2 = ax.bar(x + w/2, [v * 100 for v in vs_14b], w,
                   label="Pert win rate vs 14B baseline", color="#2a78d6", zorder=3)

    ax.axhline(50, color="#999999", linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Perturbation win rate (%)")
    ax.set_title("Where perturbation helps: structured reasoning > open-ended planning",
                 fontweight="bold", pad=12)
    ax.set_ylim(0, 85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=8, color="#e07b54")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=8, color="#2a78d6")

    fig.tight_layout()
    fig.savefig(OUT / "text_gen_category_wins.png", bbox_inches="tight")
    fig.savefig(OUT / "text_gen_category_wins.svg", bbox_inches="tight")
    plt.close(fig)
    print("Saved text_gen_category_wins.png/.svg")


if __name__ == "__main__":
    plot_scaling_vs_perturbation()
    plot_temperature_vs_perturbation()
    plot_text_gen_judge()
    plot_text_gen_by_category()
