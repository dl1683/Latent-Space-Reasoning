"""Generate publication-quality figures for warm-start research.

Outputs to experiments/figures/. Run from repo root:
    python experiments/create_figures.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

EXPERIMENTS_DIR = Path(__file__).parent
FIGURES_DIR = EXPERIMENTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("colorblind")
COLOR_BASELINE = "#888888"
COLOR_WARM = PALETTE[0]   # blue
COLOR_ZERO = PALETTE[1]   # orange
COLOR_MEAN = PALETTE[2]   # green
COLOR_1TOK = PALETTE[3]   # red
COLOR_LATENT = PALETTE[4] # purple


def load_json(name):
    path = EXPERIMENTS_DIR / name
    if not path.exists():
        print(f"  [skip] {name} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ─── Load all data ───────────────────────────────────────────────────────────

main = load_json("sensitivity_sweet_spot_results.json")
zero = load_json("sensitivity_sweet_spot_zero_embedding_results.json")
mean = load_json("sensitivity_sweet_spot_mean_embedding_results.json")
t1 = load_json("sensitivity_sweet_spot_random_noise_t1_results.json")
noise = load_json("sensitivity_sweet_spot_random_noise_results.json")
nested = load_json("sensitivity_nested_easy_results.json")
nested_noise = load_json("sensitivity_easy_nested_random_noise_results.json")
t2 = load_json("sensitivity_sweet_spot_random_noise_t2_results.json")


# ─── Figure 1: Condition comparison bar chart ────────────────────────────────

def fig1_condition_comparison():
    """Bar chart: accuracy across all control conditions."""
    if not main:
        return

    baseline_acc = main["baseline_accuracy"]
    latent_accs = main.get("latent_accuracies", [])
    latent_mean = np.mean(latent_accs) if latent_accs else main.get("mean_accuracy", 0)
    latent_std = np.std(latent_accs) if latent_accs else main.get("std_accuracy", 0)

    conditions = ["Baseline\n(no prefix)"]
    means = [baseline_acc]
    stds = [0]
    colors = [COLOR_BASELINE]

    if zero:
        conditions.append("Zero\nembedding")
        means.append(zero["mean_accuracy"])
        stds.append(zero.get("std_accuracy", 0))
        colors.append(COLOR_ZERO)

    if mean:
        conditions.append("Mean\nembedding")
        means.append(mean["mean_accuracy"])
        stds.append(mean.get("std_accuracy", 0))
        colors.append(COLOR_MEAN)

    if t1:
        conditions.append("Random noise\n(1 token)")
        means.append(t1["mean_accuracy"])
        stds.append(t1.get("std_accuracy", 0))
        colors.append(COLOR_1TOK)

    if t2:
        conditions.append("Random noise\n(2 tokens)")
        means.append(t2["mean_accuracy"])
        stds.append(t2.get("std_accuracy", 0))
        colors.append(COLOR_1TOK)

    # 8-token random noise — extract from the random_noise results file
    if noise and "sensitivity_results" in noise:
        sr = noise["sensitivity_results"]
        if sr:
            accs = [e["accuracy"] for e in sr]
            conditions.append("Random noise\n(8 tokens)")
            means.append(np.mean(accs))
            stds.append(np.std(accs))
            colors.append(COLOR_WARM)
    elif noise and "noise_mean" in noise:
        conditions.append("Random noise\n(8 tokens)")
        means.append(noise["noise_mean"])
        stds.append(noise.get("noise_std", 0))
        colors.append(COLOR_WARM)

    conditions.append("W-projected\n(8 tokens)")
    means.append(latent_mean)
    stds.append(latent_std)
    colors.append(COLOR_LATENT)

    means = [m * 100 for m in means]
    stds = [s * 100 for s in stds]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.65)

    ax.axhline(y=baseline_acc * 100, color=COLOR_BASELINE, linestyle="--",
               alpha=0.5, linewidth=1)

    for i, (m, s) in enumerate(zip(means, stds)):
        label = f"{m:.1f}%"
        if s > 0.1:
            label += f"\n({s:.1f})"  # show SD only if meaningful
        ax.text(i, m + s + 1.5, label, ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Warm-Start Effect: Accuracy by Prefix Condition\n"
                 "Qwen3-4B (Q4), 25 sweet-spot arithmetic tasks, thinking mode",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0, max(means) + max(stds) + 12)
    ax.set_xlim(-0.6, len(conditions) - 0.4)

    # Annotations
    ax.annotate("+12pp", xy=(len(conditions) - 2, means[-2]),
                xytext=(len(conditions) - 2, means[-2] + 8),
                fontsize=10, color=COLOR_WARM, fontweight="bold",
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=COLOR_WARM, lw=1.5))

    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_condition_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig1_condition_comparison.png")


# ─── Figure 2: Per-task heatmap ─────────────────────────────────────────────

def fig2_task_heatmap():
    """Heatmap showing per-task correctness across conditions."""
    if not main:
        return

    bl_results = main["baseline_results"]
    sr_latent = main.get("sensitivity_results", [])
    task_ids = sorted(r["task_id"] for r in bl_results)

    bl_map = {r["task_id"]: r["correct"] for r in bl_results}

    # Build latent correctness: average across all latents
    latent_rates = {}
    for tid in task_ids:
        ok, total = 0, 0
        for entry in sr_latent:
            for r in entry.get("task_results", []):
                if r["task_id"] == tid:
                    total += 1
                    if r["correct"]:
                        ok += 1
        latent_rates[tid] = ok / max(total, 1)

    # Build condition columns
    cond_data = {}

    # Baseline
    cond_data["Baseline"] = [1 if bl_map.get(t, False) else 0 for t in task_ids]

    # Zero embedding
    if zero and zero.get("sensitivity_results"):
        z_map = {}
        for entry in zero["sensitivity_results"]:
            for r in entry.get("task_results", []):
                z_map[r["task_id"]] = 1 if r["correct"] else 0
        cond_data["Zero\nembed"] = [z_map.get(t, -1) for t in task_ids]

    # Mean embedding
    if mean and mean.get("sensitivity_results"):
        m_map = {}
        for entry in mean["sensitivity_results"]:
            for r in entry.get("task_results", []):
                m_map[r["task_id"]] = 1 if r["correct"] else 0
        cond_data["Mean\nembed"] = [m_map.get(t, -1) for t in task_ids]

    # 1-token random
    if t1 and t1.get("sensitivity_results"):
        t1_rates = {}
        for tid in task_ids:
            ok, total = 0, 0
            for entry in t1["sensitivity_results"]:
                for r in entry.get("task_results", []):
                    if r["task_id"] == tid:
                        total += 1
                        if r["correct"]:
                            ok += 1
            t1_rates[tid] = ok / max(total, 1)
        cond_data["1-tok\nrandom"] = [t1_rates.get(t, -1) for t in task_ids]

    # 8-token latent-projected (average rate)
    cond_data["8-tok\nW-proj"] = [latent_rates.get(t, -1) for t in task_ids]

    # Build matrix
    cond_names = list(cond_data.keys())
    matrix = np.array([cond_data[c] for c in cond_names]).T  # tasks x conditions

    fig, ax = plt.subplots(figsize=(6, 10))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels(cond_names, fontsize=9)
    ax.set_yticks(range(len(task_ids)))
    ax.set_yticklabels(task_ids, fontsize=8)

    # Add text annotations
    for i in range(len(task_ids)):
        for j in range(len(cond_names)):
            val = matrix[i, j]
            if val < 0:
                txt = "?"
            elif val == 0:
                txt = "X"
            elif val == 1:
                txt = "OK"
            else:
                txt = f"{val:.0%}"
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    fontweight="bold", color=color)

    ax.set_title("Per-Task Correctness by Condition\n(green = correct, red = wrong)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.5, label="Correct rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_task_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig2_task_heatmap.png")


# ─── Figure 3: Error redistribution (Sankey-style) ──────────────────────────

def fig3_error_redistribution():
    """Stacked bar showing which tasks got fixed vs regressed by 8-token prefix."""
    if not main:
        return

    bl_results = main["baseline_results"]
    sr_latent = main.get("sensitivity_results", [])
    bl_map = {r["task_id"]: r["correct"] for r in bl_results}
    task_ids = sorted(bl_map.keys())

    # Get per-task rates across all latents
    latent_rates = {}
    for tid in task_ids:
        ok, total = 0, 0
        for entry in sr_latent:
            for r in entry.get("task_results", []):
                if r["task_id"] == tid:
                    total += 1
                    if r["correct"]:
                        ok += 1
        latent_rates[tid] = ok / max(total, 1)

    categories = {
        "Fixed\n(wrong->right, >50%)": [],
        "Still broken": [],
        "Stable correct\n(100% across latents)": [],
        "Regressed\n(any latent wrong)": [],
    }

    for tid in task_ids:
        bl_ok = bl_map[tid]
        rate = latent_rates[tid]
        if not bl_ok and rate > 0.5:
            categories["Fixed\n(wrong->right, >50%)"].append(tid)
        elif not bl_ok:
            categories["Still broken"].append(tid)
        elif bl_ok and rate >= 1.0:
            categories["Stable correct\n(100% across latents)"].append(tid)
        else:
            categories["Regressed\n(any latent wrong)"].append(tid)

    cat_names = list(categories.keys())
    cat_counts = [len(categories[c]) for c in cat_names]
    cat_colors = ["#2ecc71", "#e74c3c", "#3498db", "#e67e22"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(cat_names, cat_counts, color=cat_colors, edgecolor="black",
                   linewidth=0.8, height=0.6)

    for bar, count, cat in zip(bars, cat_counts, cat_names):
        tids = categories[cat]
        label = f"  {count} tasks"
        if tids:
            label += f"  ({', '.join(tids[:4])}{'...' if len(tids) > 4 else ''})"
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)

    ax.set_xlabel("Number of tasks (out of 25)", fontsize=11)
    ax.set_title("Error Redistribution: 8-Token Random Prefix vs Baseline\n"
                 "The +12pp improvement is NOT clean: 3 fixed, 6 regressed",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(cat_counts) + 8)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_error_redistribution.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig3_error_redistribution.png")


# ─── Figure 4: Dose-response (token count) ──────────────────────────────────

def fig4_dose_response():
    """Token count vs accuracy: 0, 1, (2?), 8 tokens."""
    if not main:
        return

    baseline_acc = main["baseline_accuracy"] * 100

    tokens = [0]
    means = [baseline_acc]
    stds = [0]

    if t1:
        tokens.append(1)
        means.append(t1["mean_accuracy"] * 100)
        stds.append(t1.get("std_accuracy", 0) * 100)

    if t2:
        tokens.append(2)
        means.append(t2["mean_accuracy"] * 100)
        stds.append(t2.get("std_accuracy", 0) * 100)

    # 8-token data from main (W-projected, but noise is equivalent)
    latent_accs = main.get("latent_accuracies", [])
    if latent_accs:
        tokens.append(8)
        means.append(np.mean(latent_accs) * 100)
        stds.append(np.std(latent_accs) * 100)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(tokens, means, yerr=stds, fmt="o-", color=COLOR_WARM,
                markersize=10, capsize=6, linewidth=2, markeredgecolor="black",
                markeredgewidth=1)

    ax.axhline(y=baseline_acc, color=COLOR_BASELINE, linestyle="--",
               alpha=0.5, linewidth=1, label="Baseline (no prefix)")

    for t, m, s in zip(tokens, means, stds):
        offset = 2.5 if t == 0 else 2.0
        ax.text(t, m + s + offset, f"{m:.1f}%", ha="center", fontsize=10,
                fontweight="bold")

    # Mark the 89% annotation
    if len(tokens) >= 3 and 1 in tokens:
        full_effect = means[-1] - baseline_acc
        one_tok_effect = means[tokens.index(1)] - baseline_acc
        if full_effect > 0:
            pct = one_tok_effect / full_effect * 100
            ax.annotate(f"1 token captures\n{pct:.0f}% of effect",
                        xy=(1, means[tokens.index(1)]),
                        xytext=(2.5, means[tokens.index(1)] - 5),
                        fontsize=9, color=COLOR_1TOK,
                        arrowprops=dict(arrowstyle="->", color=COLOR_1TOK),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                  edgecolor=COLOR_1TOK))

    ax.set_xlabel("Number of random prefix tokens", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Dose-Response: Prefix Token Count\n"
                 "Threshold effect — 1 token captures most of the benefit",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(tokens)
    ax.set_ylim(baseline_acc - 5, max(means) + max(stds) + 8)
    ax.legend(fontsize=10)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_dose_response.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig4_dose_response.png")


# ─── Figure 5: Generation time vs correctness ───────────────────────────────

def fig5_generation_time():
    """Box plot: generation time for correct vs incorrect answers."""
    if not main:
        return

    bl_results = main["baseline_results"]
    sr_latent = main.get("sensitivity_results", [])

    # Collect times
    data = []
    for r in bl_results:
        data.append({
            "condition": "Baseline",
            "correct": "Correct" if r["correct"] else "Wrong",
            "time": r["time"]
        })
    for entry in sr_latent:
        for r in entry.get("task_results", []):
            data.append({
                "condition": "8-token prefix",
                "correct": "Correct" if r["correct"] else "Wrong",
                "time": r["time"]
            })

    if not data:
        return

    import pandas as pd
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["Correct", "Wrong"]
    hue_order = ["Baseline", "8-token prefix"]

    sns.boxplot(data=df, x="correct", y="time", hue="condition",
                order=order, hue_order=hue_order, ax=ax,
                palette=[COLOR_BASELINE, COLOR_WARM], width=0.6)

    ax.axhline(y=80, color="red", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(1.35, 81, "~max_new_tokens limit", fontsize=8, color="red", alpha=0.6)

    ax.set_xlabel("Answer correctness", fontsize=12)
    ax.set_ylabel("Generation time (seconds)", fontsize=12)
    ax.set_title("Wrong Answers Hit Token Budget\n"
                 "Correct answers finish in ~60s, wrong answers exhaust 1024 tokens (~80s)",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Condition", fontsize=9)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_generation_time.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig5_generation_time.png")


# ─── Figure 6: Cross-difficulty comparison ───────────────────────────────────

def fig6_cross_difficulty():
    """Grouped bar: warm-start effect at sweet-spot (32%) vs easy (92%) baseline."""
    if not (main and nested):
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    # Sweet-spot panel
    ax = axes[0]
    sweet_baseline = main["baseline_accuracy"] * 100
    sweet_latent = np.array(main.get("latent_accuracies", [])) * 100

    ax.axhline(y=sweet_baseline, color=COLOR_BASELINE, linestyle="--",
               linewidth=1.5, label=f"Baseline ({sweet_baseline:.0f}%)")
    parts = ax.violinplot([sweet_latent], positions=[1], showmeans=True,
                          showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(COLOR_WARM)
        pc.set_alpha(0.6)
    ax.scatter(np.ones(len(sweet_latent)) + np.random.uniform(-0.05, 0.05, len(sweet_latent)),
               sweet_latent, color=COLOR_WARM, s=40, zorder=5, edgecolor="black",
               linewidth=0.5)

    ax.set_title("Sweet-Spot Tasks\n(32% baseline)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks([1])
    ax.set_xticklabels(["W-projected\n(10 latents)"])
    ax.legend(fontsize=9)
    ax.set_ylim(20, 65)

    # Easy panel
    ax = axes[1]
    easy_baseline = nested["baseline_accuracy"] * 100
    easy_latent = np.array(nested.get("latent_accuracies", [])) * 100

    ax.axhline(y=easy_baseline, color=COLOR_BASELINE, linestyle="--",
               linewidth=1.5, label=f"Baseline ({easy_baseline:.0f}%)")
    parts = ax.violinplot([easy_latent], positions=[1], showmeans=True,
                          showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(COLOR_LATENT)
        pc.set_alpha(0.6)
    ax.scatter(np.ones(len(easy_latent)) + np.random.uniform(-0.05, 0.05, len(easy_latent)),
               easy_latent, color=COLOR_LATENT, s=40, zorder=5, edgecolor="black",
               linewidth=0.5)

    ax.set_title("Easy Tasks\n(92% baseline)", fontsize=12, fontweight="bold")
    ax.set_xticks([1])
    ax.set_xticklabels(["W-projected\n(10 latents)"])
    ax.legend(fontsize=9)
    ax.set_ylim(55, 100)

    fig.suptitle("Warm-Start Effect Varies by Task Difficulty\n"
                 "Helps on hard tasks, mostly hurts on easy tasks",
                 fontsize=13, fontweight="bold", y=1.02)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_cross_difficulty.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig6_cross_difficulty.png")


# ─── Figure 7: Mechanism evidence summary ────────────────────────────────────

def fig7_mechanism_summary():
    """Visual summary table of mechanism evidence."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    experiments = [
        ("Zero embedding (8 tokens)", "+4pp", "Embedding values matter,\nnot just length"),
        ("Mean embedding (8 identical)", "+4pp", "Token diversity doesn't help\nfor identical tokens"),
        ("Random noise (1 token)", "+10.7pp", "Threshold effect:\n1 token = 89% of benefit"),
        ("Random noise (8 tokens)", "+12pp", "Random = W-projected\n(p = 1.0)"),
        ("W-projected (8 tokens)", "+12.4pp", "Direction carries\nno signal"),
        ("No-think mode (any prefix)", "+0pp", "CoT is the mediating\nmechanism"),
    ]

    col_labels = ["Experiment", "Effect vs\nbaseline", "Interpretation"]
    cell_colors = []
    cell_text = []

    color_map = {
        "+0pp": "#ffcccc",
        "+4pp": "#ffe0b2",
        "+10.7pp": "#c8e6c9",
        "+12pp": "#a5d6a7",
        "+12.4pp": "#a5d6a7",
    }

    for exp, effect, interp in experiments:
        cell_text.append([exp, effect, interp])
        c = color_map.get(effect, "white")
        cell_colors.append([c, c, "white"])

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=cell_colors, loc="center",
                     cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Style header
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("Mechanism Evidence Summary\n"
                 "Random prefix tokens shift generation policy via trajectory perturbation",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_mechanism_summary.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig7_mechanism_summary.png")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    fig1_condition_comparison()
    fig2_task_heatmap()
    fig3_error_redistribution()
    fig4_dose_response()
    fig5_generation_time()
    fig6_cross_difficulty()
    fig7_mechanism_summary()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
