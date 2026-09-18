"""Quantitative eigenspace allocation analysis.

Entry 94 found qualitatively that rare, high-information tokens get top
eigenspaces while common tokens get the tail. This script makes that
QUANTITATIVE: compute eigenspace position for ALL ~49k tokens and
correlate with token properties (length, type, BPE prefix).

Test: is eigenspace allocation predictable from token information content?

Model: SmolLM2-135M (CPU-only, weight analysis, no inference)
"""

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import torch


def analyze_allocation():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading SmolLM2-135M...")
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    d = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(f"  d={d}, vocab={vocab_size}")

    W = model.model.embed_tokens.weight.detach().numpy()  # (vocab, d)

    # Eigendecomposition of frame operator S = W^T W / vocab
    print("Computing eigendecomposition...")
    S = W.T @ W / vocab_size
    evals, evecs = np.linalg.eigh(S)
    idx = np.argsort(-evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Per-token projections onto eigenspaces
    projections_sq = (W @ evecs) ** 2  # (vocab, d) — squared projections
    norms_sq = np.sum(projections_sq, axis=1)  # (vocab,) — ||w_i||^2
    norms = np.sqrt(norms_sq + 1e-15)

    # Normalized energy distribution per token
    energy_dist = projections_sq / (norms_sq[:, None] + 1e-15)

    # Key metrics per token
    eigenindices = np.arange(d)
    mean_eigenindex = energy_dist @ eigenindices  # weighted mean eigenindex
    top10_frac = energy_dist[:, :10].sum(axis=1)  # fraction in top 10
    top1_frac = energy_dist[:, 0]  # fraction in top 1

    # Effective rank (exp of entropy of energy distribution)
    ent = -np.sum(energy_dist * np.log(energy_dist + 1e-15), axis=1)
    effective_rank = np.exp(ent)

    # Token properties
    print("Analyzing token properties...")
    token_lengths = []
    is_word_start = []  # BPE Ġ prefix
    is_alpha = []
    is_digit_token = []
    is_special = []
    is_punct = []
    char_counts = []

    for i in range(vocab_size):
        tok_str = tokenizer.decode([i])
        raw_token = tokenizer.convert_ids_to_tokens(i) if i < len(tokenizer) else ""
        raw_token = raw_token or ""

        token_lengths.append(len(tok_str.strip()))
        is_word_start.append(1 if raw_token.startswith("Ġ") or raw_token.startswith("▁") else 0)
        stripped = tok_str.strip()
        is_alpha.append(1 if stripped.isalpha() and len(stripped) > 0 else 0)
        is_digit_token.append(1 if stripped.isdigit() else 0)
        is_special.append(1 if stripped in ["", " ", "\n", "\t", "<s>", "</s>", "<pad>", "<unk>"] else 0)
        is_punct.append(1 if len(stripped) > 0 and all(c in ".,;:!?-()[]{}'\"/\\@#$%^&*_+=<>~`|" for c in stripped) else 0)
        char_counts.append(len(stripped))

    token_lengths = np.array(token_lengths)
    is_word_start = np.array(is_word_start)
    is_alpha = np.array(is_alpha)
    is_digit_token = np.array(is_digit_token)
    is_special = np.array(is_special)
    is_punct = np.array(is_punct)
    char_counts = np.array(char_counts)

    # Information proxy: longer tokens are rarer and carry more information
    # BPE encodes frequent tokens as short, rare tokens as long
    # So token length IS an information proxy
    info_proxy = char_counts.astype(float)

    # Correlations
    mask = char_counts > 0  # exclude empty tokens
    n_valid = mask.sum()

    def corr(a, b, m=mask):
        a_, b_ = a[m].astype(float), b[m].astype(float)
        if np.std(a_) < 1e-10 or np.std(b_) < 1e-10:
            return 0.0
        return float(np.corrcoef(a_, b_)[0, 1])

    results = {
        "model": "SmolLM2-135M",
        "d": int(d),
        "vocab_size": int(vocab_size),
        "n_valid_tokens": int(n_valid),
        "eigenvalue_top10": evals[:10].tolist(),
        "eigenvalue_exponent": None,
        "correlations": {
            "norm_vs_length": corr(norms, info_proxy),
            "mean_eigenindex_vs_length": corr(mean_eigenindex, info_proxy),
            "top10_frac_vs_length": corr(top10_frac, info_proxy),
            "top1_frac_vs_length": corr(top1_frac, info_proxy),
            "effective_rank_vs_length": corr(effective_rank, info_proxy),
            "norm_vs_word_start": corr(norms, is_word_start),
            "mean_eigenindex_vs_alpha": corr(mean_eigenindex, is_alpha),
        },
    }

    # Power-law fit
    epos = evals[evals > 1e-10]
    n_fit = len(epos) // 2
    log_k = np.log(np.arange(1, n_fit + 1))
    log_e = np.log(epos[:n_fit])
    slope, _ = np.polyfit(log_k, log_e, 1)
    results["eigenvalue_exponent"] = float(-slope)

    # Category analysis (mean ± std of eigenspace metrics by token type)
    categories = {
        "alpha_short": (is_alpha == 1) & (char_counts <= 3) & mask,
        "alpha_long": (is_alpha == 1) & (char_counts >= 6) & mask,
        "alpha_verylong": (is_alpha == 1) & (char_counts >= 10) & mask,
        "digits": (is_digit_token == 1) & mask,
        "punctuation": (is_punct == 1) & mask,
        "word_start_alpha": (is_word_start == 1) & (is_alpha == 1) & mask,
        "continuation_alpha": (is_word_start == 0) & (is_alpha == 1) & mask,
        "special": (is_special == 1) & mask,
    }

    cat_stats = {}
    for cat_name, cat_mask in categories.items():
        n_cat = int(cat_mask.sum())
        if n_cat < 5:
            continue
        cat_stats[cat_name] = {
            "n": n_cat,
            "norm": {"mean": float(norms[cat_mask].mean()), "std": float(norms[cat_mask].std())},
            "mean_eigenindex": {"mean": float(mean_eigenindex[cat_mask].mean()), "std": float(mean_eigenindex[cat_mask].std())},
            "top10_frac": {"mean": float(top10_frac[cat_mask].mean()), "std": float(top10_frac[cat_mask].std())},
            "effective_rank": {"mean": float(effective_rank[cat_mask].mean()), "std": float(effective_rank[cat_mask].std())},
        }
    results["category_stats"] = cat_stats

    # Binned analysis: group by char_count, report mean eigenspace metrics
    bins = {}
    for length in range(1, 15):
        length_mask = (char_counts == length) & mask
        n_bin = int(length_mask.sum())
        if n_bin < 10:
            continue
        bins[str(length)] = {
            "n": n_bin,
            "norm_mean": float(norms[length_mask].mean()),
            "mean_eigenindex_mean": float(mean_eigenindex[length_mask].mean()),
            "top10_frac_mean": float(top10_frac[length_mask].mean()),
            "effective_rank_mean": float(effective_rank[length_mask].mean()),
        }
    results["by_char_count"] = bins

    # Print summary
    print("\n" + "=" * 70)
    print("EIGENSPACE ALLOCATION ANALYSIS")
    print("=" * 70)

    print(f"\nEigenvalue exponent: {results['eigenvalue_exponent']:.3f}")
    print(f"Top eigenvalue captures {evals[0]/evals.sum()*100:.1f}% of trace")

    print("\nCorrelations (token property vs eigenspace metric):")
    for k, v in results["correlations"].items():
        print(f"  {k:40s}: r = {v:+.4f}")

    print("\nCategory analysis:")
    print(f"  {'Category':25s} {'N':>6s} {'||w||':>7s} {'MeanIdx':>8s} {'Top10%':>7s} {'EffRank':>8s}")
    for cat_name in sorted(cat_stats.keys()):
        s = cat_stats[cat_name]
        print(f"  {cat_name:25s} {s['n']:6d} {s['norm']['mean']:7.2f} "
              f"{s['mean_eigenindex']['mean']:8.1f} {s['top10_frac']['mean']:7.3f} "
              f"{s['effective_rank']['mean']:8.1f}")

    print("\nBy character count (information proxy):")
    print(f"  {'Len':>4s} {'N':>6s} {'||w||':>7s} {'MeanIdx':>8s} {'Top10%':>7s} {'EffRank':>8s}")
    for length in sorted(bins.keys(), key=int):
        b = bins[length]
        print(f"  {length:>4s} {b['n']:6d} {b['norm_mean']:7.2f} "
              f"{b['mean_eigenindex_mean']:8.1f} {b['top10_frac_mean']:7.3f} "
              f"{b['effective_rank_mean']:8.1f}")

    # The key test: does token length predict eigenspace position?
    valid_lengths = char_counts[mask].astype(float)
    valid_mean_idx = mean_eigenindex[mask]
    r = corr(valid_lengths, valid_mean_idx, np.ones(n_valid, dtype=bool))
    print(f"\nKEY TEST: correlation(char_count, mean_eigenindex) = {r:+.4f}")
    if r < -0.3:
        print("  >> CONFIRMED: longer (rarer, higher-info) tokens get LOWER eigenindex (top eigenspaces)")
    elif r > 0.3:
        print("  >> REVERSED: longer tokens get HIGHER eigenindex (tail eigenspaces)")
    else:
        print("  >> WEAK: token length doesn't strongly predict eigenspace allocation")

    out_dir = Path(__file__).parent / "results" / "eigenspace_allocation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smollm2_135m_allocation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")

    return results


if __name__ == "__main__":
    analyze_allocation()
