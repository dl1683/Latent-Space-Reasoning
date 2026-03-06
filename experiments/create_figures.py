"""Generate publication-quality figures for perturbation-gated reasoning paper.

Outputs to experiments/figures/. Run from repo root:
    python experiments/create_figures.py

Figures (per Codex 2026-03-06):
  1. Task-condition heatmap (strict categorization)
  2. Equalization/oracle figure (sensitive tasks)
  3. Invariance length vs token count
  4. Answer magnitude tradeoff
  5. Task-specific resonance windows (nest_005/021)
  6. Dose-response curve (DRAFT until 3-tok completes)
  7. Butterfly effect divergence examples
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
C_BASE = "#888888"
C_1TOK = PALETTE[3]
C_2TOK = PALETTE[0]
C_8TOK = PALETTE[4]
C_ZERO = PALETTE[1]
C_MEAN = PALETTE[2]


def load_json(name):
    path = EXPERIMENTS_DIR / name
    if not path.exists():
        print(f"  [skip] {name} not found")
        return None
    with open(path) as f:
        return json.load(f)


def get_task_correctness(data):
    """Return {latent_idx: {task_id: bool}} from sensitivity_results."""
    result = {}
    for sr in data.get("sensitivity_results", []):
        li = sr["latent_idx"]
        result[li] = {}
        for tr in sr["task_results"]:
            result[li][tr["task_id"]] = tr["correct"]
    return result


# ─── Load all data ───────────────────────────────────────────────────────────

d1 = load_json("sensitivity_sweet_spot_random_noise_t1_results.json")
d2 = load_json("sensitivity_sweet_spot_random_noise_t2_results.json")
d8 = load_json("sensitivity_sweet_spot_results.json")
dz = load_json("sensitivity_sweet_spot_zero_embedding_results.json")
dm = load_json("sensitivity_sweet_spot_mean_embedding_results.json")
d3 = load_json("sensitivity_sweet_spot_random_noise_t3_results.json")

# Build common task info
if d2:
    ANSWERS = {r["task_id"]: r["correct_answer"] for r in d2["baseline_results"]}
    BASELINE = {r["task_id"]: r["correct"] for r in d2["baseline_results"]}
    TASK_IDS = sorted(ANSWERS.keys())
elif d8:
    ANSWERS = {r["task_id"]: r["correct_answer"] for r in d8["baseline_results"]}
    BASELINE = {r["task_id"]: r["correct"] for r in d8["baseline_results"]}
    TASK_IDS = sorted(ANSWERS.keys())
else:
    ANSWERS, BASELINE, TASK_IDS = {}, {}, []

# Compute strict categorization
def compute_categories():
    """Strict: always = solved by ALL latents in ALL conditions + baseline.
    Never = unsolved by ALL. Sensitive = rest."""
    base_correct = {t for t, c in BASELINE.items() if c}
    always = set(TASK_IDS)
    all_oracle = set(base_correct)

    for dd in [d1, d2, d8, dz]:
        if not dd:
            continue
        tc = get_task_correctness(dd)
        for li, tasks in tc.items():
            for tid, correct in tasks.items():
                if correct:
                    all_oracle.add(tid)
                else:
                    always.discard(tid)
    always &= base_correct
    never = set(TASK_IDS) - all_oracle
    sensitive = sorted(set(TASK_IDS) - always - never)
    return sorted(always), sorted(never), sensitive

ALWAYS, NEVER, SENSITIVE = compute_categories()


# ─── Figure 1: Task-Condition Heatmap ────────────────────────────────────────

def fig1_task_heatmap():
    """Full task × condition heatmap showing per-latent correctness."""
    if not d2:
        return

    # Columns: baseline, zero, 1t×3, 2t×3, 8t×10
    col_labels = ["Base"]
    col_data = []

    # Baseline
    col_data.append([int(BASELINE.get(t, False)) for t in TASK_IDS])

    # Zero (use first latent, all identical)
    if dz:
        tc = get_task_correctness(dz)
        lats = sorted(tc.keys())
        col_labels.append("Zero")
        col_data.append([int(tc[lats[0]].get(t, False)) for t in TASK_IDS])

    # 1-tok latents
    if d1:
        tc = get_task_correctness(d1)
        for li in sorted(tc.keys()):
            col_labels.append(f"1t_{li}")
            col_data.append([int(tc[li].get(t, False)) for t in TASK_IDS])

    # 2-tok latents
    if d2:
        tc = get_task_correctness(d2)
        for li in sorted(tc.keys()):
            col_labels.append(f"2t_{li}")
            col_data.append([int(tc[li].get(t, False)) for t in TASK_IDS])

    # 8-tok (show first 5 for readability)
    if d8:
        tc = get_task_correctness(d8)
        for li in sorted(tc.keys())[:5]:
            col_labels.append(f"8t_{li}")
            col_data.append([int(tc[li].get(t, False)) for t in TASK_IDS])

    matrix = np.array(col_data).T  # tasks × conditions

    # Sort tasks by answer magnitude for better visual grouping
    sort_idx = sorted(range(len(TASK_IDS)),
                      key=lambda i: abs(ANSWERS.get(TASK_IDS[i], 0)))
    sorted_tasks = [TASK_IDS[i] for i in sort_idx]
    sorted_answers = [ANSWERS.get(t, 0) for t in sorted_tasks]
    matrix = matrix[sort_idx]

    # Mark always/never/sensitive
    row_colors = []
    for t in sorted_tasks:
        if t in ALWAYS:
            row_colors.append("#c8e6c9")  # green
        elif t in NEVER:
            row_colors.append("#ffcdd2")  # red
        else:
            row_colors.append("white")

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.6), 10))
    cmap = plt.cm.colors.ListedColormap(["#e74c3c", "#2ecc71"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")

    ylabels = [f"{t} ({sorted_answers[i]})" for i, t in enumerate(sorted_tasks)]
    ax.set_yticks(range(len(sorted_tasks)))
    ax.set_yticklabels(ylabels, fontsize=7)

    # Row background colors for always/never
    for i, c in enumerate(row_colors):
        if c != "white":
            ax.axhspan(i - 0.5, i + 0.5, color=c, alpha=0.15, zorder=0)

    ax.set_title("Task × Condition Correctness Matrix\n"
                 "Sorted by |answer|; green rows = always solved, red = never solved",
                 fontsize=11, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", label="Correct"),
        mpatches.Patch(facecolor="#e74c3c", label="Wrong"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_task_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig1_task_heatmap.png")


# ─── Figure 2: Equalization & Oracle ─────────────────────────────────────────

def fig2_equalization_oracle():
    """Bar chart: per-latent sensitive-task count + oracle rate by condition."""
    if not SENSITIVE:
        return

    conditions = []
    per_lat_counts = []
    oracle_counts = []
    colors = []

    # Baseline
    base_sens = sum(1 for t in SENSITIVE if BASELINE.get(t, False))
    conditions.append("Baseline")
    per_lat_counts.append([base_sens])
    oracle_counts.append(base_sens)
    colors.append(C_BASE)

    # Zero
    if dz:
        tc = get_task_correctness(dz)
        counts = [sum(1 for t in SENSITIVE if tc[li].get(t, False)) for li in sorted(tc.keys())]
        oracle = sum(1 for t in SENSITIVE if any(tc[li].get(t, False) for li in tc))
        conditions.append("Zero (8t)")
        per_lat_counts.append(counts)
        oracle_counts.append(oracle)
        colors.append(C_ZERO)

    # 1-tok
    if d1:
        tc = get_task_correctness(d1)
        counts = [sum(1 for t in SENSITIVE if tc[li].get(t, False)) for li in sorted(tc.keys())]
        oracle = sum(1 for t in SENSITIVE if any(tc[li].get(t, False) for li in tc))
        conditions.append("1-tok")
        per_lat_counts.append(counts)
        oracle_counts.append(oracle)
        colors.append(C_1TOK)

    # 2-tok
    if d2:
        tc = get_task_correctness(d2)
        counts = [sum(1 for t in SENSITIVE if tc[li].get(t, False)) for li in sorted(tc.keys())]
        oracle = sum(1 for t in SENSITIVE if any(tc[li].get(t, False) for li in tc))
        conditions.append("2-tok")
        per_lat_counts.append(counts)
        oracle_counts.append(oracle)
        colors.append(C_2TOK)

    # 8-tok
    if d8:
        tc = get_task_correctness(d8)
        counts = [sum(1 for t in SENSITIVE if tc[li].get(t, False)) for li in sorted(tc.keys())]
        oracle = sum(1 for t in SENSITIVE if any(tc[li].get(t, False) for li in tc))
        conditions.append("8-tok")
        per_lat_counts.append(counts)
        oracle_counts.append(oracle)
        colors.append(C_8TOK)

    n_sens = len(SENSITIVE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Per-latent counts (box/strip plot)
    ax = axes[0]
    for i, (cond, counts) in enumerate(zip(conditions, per_lat_counts)):
        x = np.full(len(counts), i) + np.random.uniform(-0.1, 0.1, len(counts))
        ax.scatter(x, counts, color=colors[i], s=50, edgecolor="black", linewidth=0.5, zorder=5)
        mean_c = np.mean(counts)
        std_c = np.std(counts)
        ax.errorbar(i, mean_c, yerr=std_c, fmt="_", color="black", markersize=15,
                     linewidth=2, capsize=5)
        ax.text(i, max(counts) + 0.8, f"std={std_c:.2f}", ha="center", fontsize=8)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel(f"Tasks solved (out of {n_sens} sensitive)", fontsize=10)
    ax.set_title("A. Per-Direction Solve Count\n"
                 "2-tok: perfect equalization (std=0.0)",
                 fontsize=11, fontweight="bold")

    # Panel B: Oracle rate
    ax = axes[1]
    oracle_pcts = [o / n_sens * 100 for o in oracle_counts]
    bars = ax.bar(range(len(conditions)), oracle_pcts, color=colors,
                  edgecolor="black", linewidth=0.8)
    for i, (pct, cnt) in enumerate(zip(oracle_pcts, oracle_counts)):
        ax.text(i, pct + 1.5, f"{cnt}/{n_sens}\n({pct:.0f}%)", ha="center",
                fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Oracle accuracy (%)", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title("B. Oracle (any 1 of k correct)\n"
                 "2-tok matches 8-tok with 3x fewer directions",
                 fontsize=11, fontweight="bold")

    fig.suptitle(f"Equalization and Oracle Efficiency on {n_sens} Sensitive Tasks\n"
                 f"(strict: {len(ALWAYS)} always-solved, {len(NEVER)} never-solved excluded)",
                 fontsize=12, fontweight="bold", y=1.02)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_equalization_oracle.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig2_equalization_oracle.png")


# ─── Figure 3: Invariance Length ─────────────────────────────────────────────

def fig3_invariance_length():
    """Invariance length vs token count with oracle/accuracy overlay."""
    datasets = []
    for label, dd, color in [("1-tok", d1, C_1TOK), ("2-tok", d2, C_2TOK), ("8-tok", d8, C_8TOK)]:
        if not dd:
            continue
        tc_resp = {}
        for sr in dd["sensitivity_results"][:3]:  # use first 3 latents for fair comparison
            li = sr["latent_idx"]
            for tr in sr["task_results"]:
                tid = tr["task_id"]
                if tid not in tc_resp:
                    tc_resp[tid] = {}
                tc_resp[tid][li] = tr.get("response", "")

        inv_lengths = []
        for tid in TASK_IDS:
            if tid not in tc_resp:
                continue
            r = tc_resp[tid]
            lats = sorted(r.keys())
            for i in range(len(lats)):
                for j in range(i + 1, len(lats)):
                    ra, rb = r[lats[i]], r[lats[j]]
                    inv = 0
                    for k in range(min(len(ra), len(rb))):
                        if ra[k] == rb[k]:
                            inv += 1
                        else:
                            break
                    inv_lengths.append(inv)

        datasets.append((label, inv_lengths, color))

    if not datasets:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    positions = []
    for i, (label, inv, color) in enumerate(datasets):
        pos = i
        positions.append(pos)
        parts = ax.violinplot([inv], positions=[pos], showmeans=True, showmedians=True,
                              widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("red")

        mean_v = np.mean(inv)
        med_v = np.median(inv)
        n_ident = sum(1 for x in inv if x >= 500)
        ax.text(pos, max(inv) + 15, f"mean={mean_v:.0f}\nmed={med_v:.0f}\nident={n_ident}",
                ha="center", fontsize=8, style="italic")

    ax.set_xticks(positions)
    ax.set_xticklabels([d[0] for d in datasets], fontsize=11)
    ax.set_ylabel("Pairwise invariance length (chars)", fontsize=11)
    ax.set_title("Deterministic Chaos: Invariance Length vs Perturbation Energy\n"
                 "More tokens = shorter invariance = earlier bifurcation (T=0, greedy decoding)",
                 fontsize=11, fontweight="bold")
    ax.axhline(500, color="gray", linestyle=":", alpha=0.5, label="Storage truncation (500 chars)")
    ax.legend(fontsize=8)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_invariance_length.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig3_invariance_length.png")


# ─── Figure 4: Answer Magnitude Tradeoff ─────────────────────────────────────

def fig4_magnitude_tradeoff():
    """Grouped bar showing accuracy by answer magnitude tier."""
    if not d2:
        return

    tiers = [
        ("Tiny\n(|a|<=10)", lambda a: abs(a) <= 10),
        ("Small\n(11-100)", lambda a: 11 <= abs(a) <= 100),
        ("Medium\n(101-1k)", lambda a: 101 <= abs(a) <= 1000),
        ("Large\n(1k-5k)", lambda a: 1001 <= abs(a) <= 5000),
        ("Huge\n(>5k)", lambda a: abs(a) > 5000),
    ]

    cond_data = {}  # cond_name -> [tier_accuracy, ...]
    cond_colors = {}

    for cname, dd, color in [("Base", None, C_BASE), ("1-tok", d1, C_1TOK),
                              ("2-tok", d2, C_2TOK), ("8-tok", d8, C_8TOK)]:
        tier_accs = []
        for tier_name, tier_fn in tiers:
            tasks_in_tier = [t for t in TASK_IDS if tier_fn(ANSWERS.get(t, 0))]
            if not tasks_in_tier:
                tier_accs.append(0)
                continue

            if cname == "Base":
                acc = sum(BASELINE.get(t, False) for t in tasks_in_tier) / len(tasks_in_tier)
            else:
                tc = get_task_correctness(dd)
                per_lat_accs = []
                for li in tc:
                    lat_acc = sum(tc[li].get(t, False) for t in tasks_in_tier) / len(tasks_in_tier)
                    per_lat_accs.append(lat_acc)
                acc = np.mean(per_lat_accs)
            tier_accs.append(acc * 100)

        cond_data[cname] = tier_accs
        cond_colors[cname] = color

    n_conds = len(cond_data)
    n_tiers = len(tiers)
    width = 0.18
    x = np.arange(n_tiers)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (cname, accs) in enumerate(cond_data.items()):
        offset = (i - n_conds / 2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=cname, color=cond_colors[cname],
                      edgecolor="black", linewidth=0.5)
        for j, (bar, acc) in enumerate(zip(bars, accs)):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{acc:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in tiers], fontsize=9)
    ax.set_ylabel("Mean accuracy (%)", fontsize=11)
    ax.set_xlabel("Answer magnitude tier", fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, ncol=4)
    ax.set_title("Accuracy by Answer Magnitude: Not a Generic Boost\n"
                 "Tiny answers benefit most (+75pp at 2-tok); "
                 "large answers REGRESS (100% -> 78%)",
                 fontsize=11, fontweight="bold")

    # Annotate regression
    large_idx = 3  # "Large" tier
    base_val = cond_data["Base"][large_idx]
    tok2_val = cond_data["2-tok"][large_idx]
    if base_val > tok2_val:
        ax.annotate("Overthinking\nregression",
                    xy=(large_idx + 0.5 * width, tok2_val),
                    xytext=(large_idx + 1.2, tok2_val + 15),
                    fontsize=9, color="#d62728", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))

    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_magnitude_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig4_magnitude_tradeoff.png")


# ─── Figure 5: Task-Specific Resonance Windows ──────────────────────────────

def fig5_resonance_windows():
    """Case study: nest_005 (1-tok only) and nest_021 (8-tok only)."""
    if not (d1 and d2 and d8):
        return

    case_tasks = ["nest_005", "nest_021"]
    case_labels = {
        "nest_005": f"nest_005 (answer={ANSWERS.get('nest_005', '?')})\nOnly solvable at 1-tok",
        "nest_021": f"nest_021 (answer={ANSWERS.get('nest_021', '?')})\nOnly solvable at 8-tok",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax_i, tid in enumerate(case_tasks):
        ax = axes[ax_i]

        # Collect per-latent correctness across conditions
        cond_results = {}
        cond_results["Base"] = [int(BASELINE.get(tid, False))]

        if dz:
            tc = get_task_correctness(dz)
            cond_results["Zero"] = [int(tc[li].get(tid, False)) for li in sorted(tc.keys())]

        for label, dd in [("1-tok", d1), ("2-tok", d2), ("8-tok", d8)]:
            tc = get_task_correctness(dd)
            cond_results[label] = [int(tc[li].get(tid, False)) for li in sorted(tc.keys())]

        cond_names = list(cond_results.keys())
        cond_colors_local = [C_BASE, C_ZERO, C_1TOK, C_2TOK, C_8TOK]

        for i, (cname, vals) in enumerate(cond_results.items()):
            solve_rate = sum(vals) / len(vals)
            color = cond_colors_local[i] if i < len(cond_colors_local) else "gray"
            ax.bar(i, solve_rate * 100, color=color, edgecolor="black", linewidth=0.8)
            ax.text(i, solve_rate * 100 + 3, f"{sum(vals)}/{len(vals)}",
                    ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(cond_names)))
        ax.set_xticklabels(cond_names, fontsize=9)
        ax.set_ylabel("Solve rate (%)", fontsize=10)
        ax.set_ylim(0, 120)
        ax.set_title(case_labels[tid], fontsize=10, fontweight="bold")

    fig.suptitle("Task-Specific Resonance Windows\n"
                 "Different perturbation energies unlock different tasks",
                 fontsize=12, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_resonance_windows.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig5_resonance_windows.png")


# ─── Figure 6: Dose-Response (DRAFT) ────────────────────────────────────────

def fig6_dose_response():
    """Token count vs accuracy. DRAFT until 3-tok completes."""
    tokens = [0]
    means = [BASELINE and sum(BASELINE.values()) / len(BASELINE) * 100 or 0]
    stds = [0]
    colors_pts = [C_BASE]

    for n_tok, dd, color in [(1, d1, C_1TOK), (2, d2, C_2TOK), (3, d3, "#999"),
                              (8, d8, C_8TOK)]:
        if not dd:
            continue
        accs = [sr["accuracy"] * 100 for sr in dd["sensitivity_results"]]
        tokens.append(n_tok)
        means.append(np.mean(accs))
        stds.append(np.std(accs))
        colors_pts.append(color)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(tokens, means, yerr=stds, fmt="o-", color=C_2TOK,
                markersize=10, capsize=6, linewidth=2, markeredgecolor="black",
                markeredgewidth=1, zorder=5)

    # Color individual points
    for t, m, c in zip(tokens, means, colors_pts):
        ax.plot(t, m, "o", color=c, markersize=10, markeredgecolor="black",
                markeredgewidth=1, zorder=6)

    ax.axhline(y=means[0], color=C_BASE, linestyle="--", alpha=0.5,
               linewidth=1, label=f"Baseline ({means[0]:.0f}%)")

    for t, m, s in zip(tokens, means, stds):
        offset = 2.5
        ax.text(t, m + s + offset, f"{m:.1f}%", ha="center", fontsize=10,
                fontweight="bold")

    # Find peak
    peak_idx = int(np.argmax(means))
    if peak_idx > 0:
        ax.annotate(f"Best at {tokens[peak_idx]} tokens",
                    xy=(tokens[peak_idx], means[peak_idx]),
                    xytext=(tokens[peak_idx] + 1.5, means[peak_idx] - 3),
                    fontsize=9, color="#d62728", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                              edgecolor="#d62728"))

    draft_label = " [DRAFT]" if not d3 else ""
    ax.set_xlabel("Number of random prefix tokens", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Dose-Response: Prefix Token Count{draft_label}\n"
                 "Best among tested conditions at 2 tokens",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(tokens)
    ax.set_ylim(min(means) - 8, max(means) + max(stds) + 10)
    ax.legend(fontsize=10)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_dose_response.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig6_dose_response.png")


# ─── Figure 7: Butterfly Effect Examples ─────────────────────────────────────

def fig7_butterfly_effect():
    """Text alignment showing divergence points between latent directions."""
    if not d2:
        return

    responses = {}
    correctness = {}
    for sr in d2["sensitivity_results"]:
        li = sr["latent_idx"]
        for tr in sr["task_results"]:
            tid = tr["task_id"]
            if tid not in responses:
                responses[tid] = {}
                correctness[tid] = {}
            responses[tid][li] = tr.get("response", "")
            correctness[tid][li] = tr["correct"]

    # Find best visible divergence examples
    examples = []
    for tid in TASK_IDS:
        if tid not in responses:
            continue
        c = correctness[tid]
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            if c.get(a) == c.get(b):
                continue
            ra = responses[tid].get(a, "")
            rb = responses[tid].get(b, "")
            inv = 0
            for i in range(min(len(ra), len(rb))):
                if ra[i] == rb[i]:
                    inv += 1
                else:
                    break
            if 100 < inv < 450:
                examples.append((inv, tid, a, b, c[a], c[b], ra, rb))

    examples.sort(key=lambda x: -x[0])
    best = examples[:3] if examples else []

    if not best:
        return

    fig, axes = plt.subplots(len(best), 1, figsize=(12, 3 * len(best)))
    if len(best) == 1:
        axes = [axes]

    for ax_i, (inv, tid, la, lb, ca, cb, ra, rb) in enumerate(best):
        ax = axes[ax_i]
        ax.axis("off")

        shared = ra[max(0, inv - 60):inv]
        div_a = ra[inv:inv + 100]
        div_b = rb[inv:inv + 100]

        text = (
            f"Task: {tid} (answer={ANSWERS.get(tid, '?')})  |  "
            f"Invariance: {inv} chars  |  "
            f"L{la}={'OK' if ca else 'WRONG'} vs L{lb}={'OK' if cb else 'WRONG'}\n\n"
            f'Shared: ...{shared}\n'
            f'L{la} ({"OK" if ca else "X"}): {div_a}\n'
            f'L{lb} ({"OK" if cb else "X"}): {div_b}'
        )
        ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Butterfly Effect: Deterministic Divergence Under Greedy Decoding (T=0)\n"
                 "Different random prefix embeddings produce byte-identical text, "
                 "then diverge to different outcomes",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_butterfly_effect.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig7_butterfly_effect.png")


def fig8_coverage_budget():
    """Coverage-vs-budget curve (Codex: recommended main figure).

    Shows oracle coverage as a function of number of perturbation runs,
    comparing 2-tok vs 8-tok efficiency.
    """
    # Load per-task data for 2-tok and 8-tok
    t2_data = load_json("sensitivity_sweet_spot_random_noise_t2_results.json")
    t8_data = load_json("sensitivity_sweet_spot_results.json")
    t1_data = load_json("sensitivity_sweet_spot_random_noise_t1_results.json")

    if not all([t2_data, t8_data, t1_data]):
        print("  [skip] fig8 - missing data files")
        return

    task_ids = [t["task_id"] for t in t2_data["baseline_results"]]
    n_tasks = len(task_ids)

    def build_matrix(data, n_lat):
        mat = np.zeros((n_tasks, n_lat))
        for li, sr in enumerate(data["sensitivity_results"][:n_lat]):
            for tr in sr["task_results"]:
                ti = task_ids.index(tr["task_id"])
                mat[ti, li] = 1 if tr.get("correct", tr.get("is_correct", False)) else 0
        return mat

    mat_2tok = build_matrix(t2_data, 3)
    mat_8tok = build_matrix(t8_data, 10)
    mat_1tok = build_matrix(t1_data, 3)

    # Compute oracle curves
    def oracle_curve(mat):
        n_lat = mat.shape[1]
        oracles = []
        for k in range(1, n_lat + 1):
            oracle = (mat[:, :k].max(axis=1) > 0).mean() * 100
            oracles.append(oracle)
        return oracles

    oc_2tok = oracle_curve(mat_2tok)
    oc_8tok = oracle_curve(mat_8tok)
    oc_1tok = oracle_curve(mat_1tok)

    # Independence null for each
    def independence_null(mat, n_perm=2000):
        n_lat = mat.shape[1]
        curves = []
        rng = np.random.default_rng(42)
        for _ in range(n_perm):
            perm_mat = np.zeros_like(mat)
            for li in range(n_lat):
                perm_col = mat[:, li].copy()
                rng.shuffle(perm_col)
                perm_mat[:, li] = perm_col
            curve = []
            for k in range(1, n_lat + 1):
                curve.append((perm_mat[:, :k].max(axis=1) > 0).mean() * 100)
            curves.append(curve)
        return np.array(curves)

    null_2tok = independence_null(mat_2tok)
    null_8tok = independence_null(mat_8tok)

    # Mean accuracy lines for reference
    mean_2tok = mat_2tok.mean() * 100
    mean_8tok = mat_8tok.mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    # 2-tok
    ax.plot(range(1, 4), oc_2tok, "o-", color=C_2TOK, linewidth=2.5,
            markersize=10, label=f"2-tok noise (mean={mean_2tok:.0f}%)", zorder=5)
    null_mean_2 = null_2tok.mean(axis=0)
    null_lo_2 = np.percentile(null_2tok, 2.5, axis=0)
    null_hi_2 = np.percentile(null_2tok, 97.5, axis=0)
    ax.fill_between(range(1, 4), null_lo_2, null_hi_2, alpha=0.15, color=C_2TOK)
    ax.plot(range(1, 4), null_mean_2, "--", color=C_2TOK, alpha=0.5,
            label="2-tok independence null")

    # 8-tok
    ax.plot(range(1, 11), oc_8tok, "s-", color=C_8TOK, linewidth=2,
            markersize=7, label=f"8-tok latent (mean={mean_8tok:.0f}%)", zorder=4)
    null_mean_8 = null_8tok.mean(axis=0)
    null_lo_8 = np.percentile(null_8tok, 2.5, axis=0)
    null_hi_8 = np.percentile(null_8tok, 97.5, axis=0)
    ax.fill_between(range(1, 11), null_lo_8, null_hi_8, alpha=0.12, color=C_8TOK)
    ax.plot(range(1, 11), null_mean_8, "--", color=C_8TOK, alpha=0.5,
            label="8-tok independence null")

    # 1-tok
    ax.plot(range(1, 4), oc_1tok, "^-", color=C_1TOK, linewidth=1.5,
            markersize=7, label=f"1-tok noise (mean={mat_1tok.mean()*100:.0f}%)", zorder=3)

    # Baseline
    base_acc = sum(1 for t in t2_data["baseline_results"]
                   if t.get("correct", t.get("is_correct", False))) / n_tasks * 100
    ax.axhline(y=base_acc, color=C_BASE, linestyle=":", linewidth=1.5,
               label=f"Baseline ({base_acc:.0f}%)")

    # 100% line
    ax.axhline(y=100, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Number of perturbation runs (k)", fontsize=12)
    ax.set_ylabel("Oracle coverage (%)", fontsize=12)
    ax.set_ylim(25, 105)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Oracle Coverage vs Perturbation Budget\n"
                 "2-tok reaches 88% in 3 runs; 8-tok needs 10 for 92%\n"
                 "Shaded: independence null (observed below = positive correlation)",
                 fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_coverage_budget.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig8_coverage_budget.png")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating paper figures...")
    print(f"  Strict categories: {len(ALWAYS)} always, {len(NEVER)} never, "
          f"{len(SENSITIVE)} sensitive")
    fig1_task_heatmap()
    fig2_equalization_oracle()
    fig3_invariance_length()
    fig4_magnitude_tradeoff()
    fig5_resonance_windows()
    fig6_dose_response()
    fig7_butterfly_effect()
    fig8_coverage_budget()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
