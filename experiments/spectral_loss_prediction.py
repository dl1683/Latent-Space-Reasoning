"""Does spectral structure predict per-token loss?

Chain: Zipf data distribution -> power-law eigenspectrum -> per-token loss.

Design: Bigram prediction task. Each token x has a FIXED conditional P(y|x)
drawn from a Dirichlet prior. All conditionals have similar entropy. The task
is non-trivial: the model must learn each token's conditional distribution.

The question: do frequent tokens (which get more training examples) learn
better (lower loss), and does the spectral exponent predict the per-token
loss vs frequency relationship?

Setup: toy embeddings (d=64, n=1024) trained with different Zipf exponents.
CPU-only, ~2 min total.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def make_bigram_matrix(vocab_size, n_neighbors=20, seed=42):
    """Random bigram: each token has a peaked conditional over ~n_neighbors tokens."""
    rng = np.random.default_rng(seed)
    B = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for i in range(vocab_size):
        targets = rng.choice(vocab_size, n_neighbors, replace=False)
        weights = rng.dirichlet(np.ones(n_neighbors) * 2.0)
        B[i, targets] = weights
    return B


def train_and_analyze(vocab_size, d_model, zipf_s, bigram, n_steps=5000, batch_size=256, seed=42):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = ranks ** (-zipf_s) if zipf_s > 0 else np.ones(vocab_size, dtype=np.float64)
    probs /= probs.sum()

    embedding = nn.Embedding(vocab_size, d_model)
    head = nn.Linear(d_model, vocab_size)
    params = list(embedding.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    n_total = n_steps * batch_size
    inputs_np = rng.choice(vocab_size, size=n_total, p=probs)
    targets_np = np.array([
        rng.choice(vocab_size, p=bigram[x]) for x in inputs_np
    ])
    inputs = torch.tensor(inputs_np)
    targets = torch.tensor(targets_np)

    for step in range(n_steps):
        idx = slice(step * batch_size, (step + 1) * batch_size)
        x = embedding(inputs[idx])
        logits = head(x)
        loss = nn.functional.cross_entropy(logits, targets[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Per-token loss on a held-out test set
    n_test = 100000
    test_inputs_np = rng.choice(vocab_size, size=n_test, p=probs)
    test_targets_np = np.array([rng.choice(vocab_size, p=bigram[x]) for x in test_inputs_np])
    test_inputs = torch.tensor(test_inputs_np)
    test_targets = torch.tensor(test_targets_np)

    embedding.eval()
    head.eval()
    per_token_loss = np.zeros(vocab_size)
    per_token_count = np.zeros(vocab_size)
    with torch.no_grad():
        for start in range(0, n_test, 1024):
            end = min(start + 1024, n_test)
            x = embedding(test_inputs[start:end])
            logits = head(x)
            losses = nn.functional.cross_entropy(logits, test_targets[start:end], reduction='none')
            for i in range(end - start):
                tok = test_inputs[start + i].item()
                per_token_loss[tok] += losses[i].item()
                per_token_count[tok] += 1

    mask = per_token_count >= 5
    per_token_loss[mask] /= per_token_count[mask]

    # Bayes-optimal loss per token (conditional entropy)
    bayes_loss = np.array([
        -np.sum(bigram[i] * np.log(bigram[i] + 1e-15)) for i in range(vocab_size)
    ])

    # Excess loss = model loss - Bayes optimal
    excess_loss = per_token_loss - bayes_loss

    # Eigenspectrum
    W = embedding.weight.detach().numpy()
    S = W.T @ W / vocab_size
    evals = np.sort(np.linalg.eigvalsh(S))[::-1]
    evals_pos = evals[evals > 1e-10]
    log_k = np.log(np.arange(1, len(evals_pos) + 1))
    log_e = np.log(evals_pos)
    n_fit = len(evals_pos) // 2
    slope, _ = np.polyfit(log_k[:n_fit], log_e[:n_fit], 1)
    alpha = -slope

    freq_ranks = np.argsort(-probs) + 1

    # Correlations on tokens with enough samples
    enough = mask & (per_token_count >= 20)
    n_enough = int(np.sum(enough))

    results = {
        "zipf_s": zipf_s,
        "alpha": float(alpha),
        "n_tokens_enough": n_enough,
        "mean_loss": float(np.mean(per_token_loss[mask])),
        "mean_bayes": float(np.mean(bayes_loss)),
        "mean_excess": float(np.mean(excess_loss[mask])),
    }

    if n_enough > 20:
        log_rank = np.log(freq_ranks[enough])
        log_excess = np.log(np.maximum(excess_loss[enough], 1e-10))

        corr_rank_excess = float(np.corrcoef(log_rank, log_excess)[0, 1])
        excess_slope, _ = np.polyfit(log_rank, log_excess, 1)

        corr_rank_loss = float(np.corrcoef(log_rank, per_token_loss[enough])[0, 1])
        loss_slope, _ = np.polyfit(log_rank, np.log(per_token_loss[enough] + 1e-10), 1)

        results.update({
            "corr_rank_excess": corr_rank_excess,
            "excess_slope": float(excess_slope),
            "corr_rank_loss": corr_rank_loss,
            "loss_slope": float(loss_slope),
        })

    return results


def main():
    vocab_size = 1024
    d_model = 64
    zipf_values = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3]
    seeds = [42, 137, 256]

    print("Generating bigram matrix...")
    bigram = make_bigram_matrix(vocab_size, n_neighbors=20, seed=99)

    print("=" * 70)
    print("SPECTRAL-LOSS PREDICTION TEST (bigram task)")
    print(f"vocab={vocab_size}, d={d_model}, neighbors=20, 3 seeds per Zipf")
    print("=" * 70)

    all_results = {}
    for zipf_s in zipf_values:
        seed_results = []
        for seed in seeds:
            print(f"  Training s={zipf_s}, seed={seed}...", end=" ", flush=True)
            r = train_and_analyze(vocab_size, d_model, zipf_s, bigram, seed=seed)
            seed_results.append(r)
            print(f"alpha={r['alpha']:.3f}, loss={r['mean_loss']:.4f}")

        def mean_key(key):
            vals = [r.get(key) for r in seed_results if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        summary = {
            "zipf_s": zipf_s,
            "alpha_mean": mean_key("alpha"),
            "loss_slope_mean": mean_key("loss_slope"),
            "excess_slope_mean": mean_key("excess_slope"),
            "corr_rank_loss_mean": mean_key("corr_rank_loss"),
            "corr_rank_excess_mean": mean_key("corr_rank_excess"),
            "mean_loss": mean_key("mean_loss"),
            "mean_excess": mean_key("mean_excess"),
            "seeds": seed_results,
        }
        all_results[f"zipf_{zipf_s}"] = summary

        print(
            f"  >> s={zipf_s:.1f}: alpha={summary['alpha_mean']:.3f}, "
            f"loss~rank^{summary.get('loss_slope_mean', 0):.3f}, "
            f"excess~rank^{summary.get('excess_slope_mean', 0):.3f}, "
            f"r(rank,excess)={summary.get('corr_rank_excess_mean', 0):.3f}"
        )

    print("\n" + "=" * 70)
    print("CHAIN: Zipf s -> spectral alpha -> loss distribution")
    print("=" * 70)
    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(
            f"  s={r['zipf_s']:.1f} -> alpha={r['alpha_mean']:.3f} -> "
            f"excess~rank^{r.get('excess_slope_mean', 0):.3f}"
        )

    alphas = [all_results[k]["alpha_mean"] for k in sorted(all_results.keys())
              if all_results[k]["alpha_mean"] is not None]
    excess_slopes = [all_results[k].get("excess_slope_mean", 0) for k in sorted(all_results.keys())
                     if all_results[k].get("excess_slope_mean") is not None]

    if len(alphas) > 2 and np.std(alphas) > 0.01:
        corr = float(np.corrcoef(alphas, excess_slopes)[0, 1])
        print(f"\n  Correlation(alpha, excess_slope) = {corr:.4f}")
        if abs(corr) > 0.8:
            print("  >> STRONG: spectral exponent predicts how loss distributes across tokens")
        elif abs(corr) > 0.5:
            print("  >> MODERATE: partial relationship")
        else:
            print("  >> WEAK: spectral exponent doesn't predict loss distribution")

    out_dir = Path(__file__).parent / "results" / "spectral_loss"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bigram_prediction_test.json"

    def to_native(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=to_native)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
