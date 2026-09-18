"""Does the linearization of gated MLP composition improve with dimension?

Entry 109 showed ||DU||_F predicts SwiGLU permutation damage at R2~0.95.
This means the nonlinear computation approximately linearizes for damage
prediction. If this improves with d, it's a NATIVE HD phenomenon:
high dimensions suppress nonlinear effects via concentration.

Test: train gated MLPs at d=16,32,64,128,256,512. Apply partial permutations.
Measure how well the linear predictor ||DΔU||_F approximates the actual
nonlinear damage.

Prediction: R2(DΔU vs actual) should INCREASE with d.
Mechanism: gate values concentrate via CLT → silu becomes ~linear.

CPU-only, ~5 min total.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def silu_np(x):
    return x / (1 + np.exp(-np.clip(x, -20, 20)))


def train_gated_mlp(d, m, n_vocab, n_steps=3000, batch_size=128, zipf_s=0.5, seed=42):
    """Train a gated MLP (embedding → SwiGLU → head) on bigram prediction."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    ranks = np.arange(1, n_vocab + 1, dtype=np.float64)
    probs = ranks ** (-zipf_s)
    probs /= probs.sum()

    # Bigram matrix
    B = np.zeros((n_vocab, n_vocab))
    for i in range(n_vocab):
        targets = rng.choice(n_vocab, min(20, n_vocab), replace=False)
        weights = rng.dirichlet(np.ones(len(targets)) * 2.0)
        B[i, targets] = weights

    embed = nn.Embedding(n_vocab, d)
    gate_proj = nn.Linear(d, m, bias=False)
    up_proj = nn.Linear(d, m, bias=False)
    down_proj = nn.Linear(m, d, bias=False)
    head = nn.Linear(d, n_vocab)

    params = (list(embed.parameters()) + list(gate_proj.parameters()) +
              list(up_proj.parameters()) + list(down_proj.parameters()) +
              list(head.parameters()))
    optimizer = torch.optim.Adam(params, lr=1e-3)

    n_total = n_steps * batch_size
    inputs_np = rng.choice(n_vocab, size=n_total, p=probs)
    targets_np = np.array([rng.choice(n_vocab, p=B[x]) for x in inputs_np])
    inputs_t = torch.tensor(inputs_np)
    targets_t = torch.tensor(targets_np)

    for step in range(n_steps):
        idx = slice(step * batch_size, (step + 1) * batch_size)
        x = embed(inputs_t[idx])
        g = gate_proj(x)
        u = up_proj(x)
        h = down_proj(torch.nn.functional.silu(g) * u)
        logits = head(x + h)  # residual connection
        loss = nn.functional.cross_entropy(logits, targets_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        "gate": gate_proj.weight.detach().numpy(),  # (m, d)
        "up": up_proj.weight.detach().numpy(),        # (m, d)
        "down": down_proj.weight.detach().numpy(),    # (d, m)
        "embed": embed.weight.detach().numpy(),       # (n_vocab, d)
        "loss": float(loss.item()),
    }


def test_linearization(gate_w, up_w, down_w, inputs, n_perms=30, seed=42):
    """Test how well ||DΔU||_F predicts actual SwiGLU damage."""
    rng = np.random.default_rng(seed)
    m, d = up_w.shape

    fracs = [0.1, 0.3, 0.5, 0.7, 1.0]
    records = []

    for frac in fracs:
        for trial in range(n_perms // len(fracs)):
            perm = np.arange(m)
            n_swap = max(2, int(m * frac))
            swap_idx = rng.choice(m, n_swap, replace=False)
            shuffled = rng.permutation(swap_idx)
            perm[swap_idx] = shuffled

            up_perm = up_w[perm]
            delta_u = up_w - up_perm

            du_norm = float(np.linalg.norm(down_w @ delta_u, 'fro'))

            actual_errors = []
            for x in inputs:
                gx = gate_w @ x
                ux = up_w @ x
                ux_perm = up_perm @ x
                silu_gx = silu_np(gx)
                y_orig = down_w @ (silu_gx * ux)
                y_perm = down_w @ (silu_gx * ux_perm)
                actual_errors.append(float(np.linalg.norm(y_orig - y_perm)))

            mean_actual = float(np.mean(actual_errors))
            records.append({"frac": frac, "actual": mean_actual, "du_pred": du_norm})

    actuals = np.array([r["actual"] for r in records])
    preds = np.array([r["du_pred"] for r in records])

    if np.std(actuals) < 1e-10 or np.std(preds) < 1e-10:
        return {"r2": 0.0, "corr": 0.0}

    corr = float(np.corrcoef(actuals, preds)[0, 1])

    # R2 with optimal scaling
    slope = np.dot(preds, actuals) / (np.dot(preds, preds) + 1e-15)
    scaled = preds * slope
    ss_res = np.sum((actuals - scaled) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-15))

    # Gate value statistics
    gate_stds = []
    for x in inputs:
        gx = gate_w @ x
        gate_stds.append(float(np.std(gx)))
    mean_gate_std = float(np.mean(gate_stds))

    # Nonlinearity strength: how far is silu from linear?
    nonlin_ratios = []
    for x in inputs:
        gx = gate_w @ x
        silu_gx = silu_np(gx)
        linear_gx = gx * 0.5  # silu ≈ 0.5*x near 0
        nonlin = float(np.linalg.norm(silu_gx - linear_gx))
        scale = float(np.linalg.norm(silu_gx)) + 1e-10
        nonlin_ratios.append(nonlin / scale)
    mean_nonlin = float(np.mean(nonlin_ratios))

    return {
        "r2": r2,
        "corr": corr,
        "gate_std": mean_gate_std,
        "nonlin_ratio": mean_nonlin,
    }


def main():
    dimensions = [16, 32, 64, 128, 256, 512]
    n_vocab = 256  # small vocab to keep training fast
    expansion = 4  # m = 4*d
    seeds = [42, 137, 256]
    n_test_inputs = 20

    print("=" * 70)
    print("LINEARIZATION vs DIMENSION TEST")
    print(f"dims={dimensions}, vocab={n_vocab}, m=4d, 3 seeds")
    print("=" * 70)

    all_results = {}

    for d in dimensions:
        m = expansion * d
        seed_results = []
        for seed in seeds:
            print(f"  d={d:4d} (m={m:5d}), seed={seed}...", end=" ", flush=True)
            model = train_gated_mlp(d, m, n_vocab, seed=seed)

            # Generate test inputs from trained embedding
            rng = np.random.default_rng(seed + 1000)
            test_idx = rng.choice(n_vocab, n_test_inputs, replace=False)
            test_inputs = [model["embed"][i] for i in test_idx]

            result = test_linearization(
                model["gate"], model["up"], model["down"],
                test_inputs, n_perms=30, seed=seed + 2000
            )
            result["d"] = d
            result["m"] = m
            result["loss"] = model["loss"]
            seed_results.append(result)
            print(f"R2={result['r2']:.4f}, gate_std={result['gate_std']:.3f}, nonlin={result['nonlin_ratio']:.3f}")

        r2_mean = float(np.mean([r["r2"] for r in seed_results]))
        corr_mean = float(np.mean([r["corr"] for r in seed_results]))
        gate_std_mean = float(np.mean([r["gate_std"] for r in seed_results]))
        nonlin_mean = float(np.mean([r["nonlin_ratio"] for r in seed_results]))

        all_results[f"d{d}"] = {
            "d": d,
            "m": m,
            "r2_mean": r2_mean,
            "corr_mean": corr_mean,
            "gate_std_mean": gate_std_mean,
            "nonlin_mean": nonlin_mean,
            "seeds": seed_results,
        }

    print("\n" + "=" * 70)
    print("LINEARIZATION QUALITY vs DIMENSION")
    print("=" * 70)
    print(f"  {'d':>5s} {'m':>6s} {'R2':>7s} {'corr':>7s} {'gate_σ':>7s} {'nonlin':>7s}")
    for key in sorted(all_results.keys(), key=lambda x: all_results[x]["d"]):
        r = all_results[key]
        print(f"  {r['d']:5d} {r['m']:6d} {r['r2_mean']:7.4f} {r['corr_mean']:7.4f} "
              f"{r['gate_std_mean']:7.3f} {r['nonlin_mean']:7.3f}")

    dims = [all_results[k]["d"] for k in sorted(all_results.keys(), key=lambda x: all_results[x]["d"])]
    r2s = [all_results[k]["r2_mean"] for k in sorted(all_results.keys(), key=lambda x: all_results[x]["d"])]

    if len(dims) > 2:
        log_dims = np.log(np.array(dims, dtype=float))
        r2_arr = np.array(r2s)
        corr_d_r2 = float(np.corrcoef(log_dims, r2_arr)[0, 1])
        print(f"\n  Correlation(log(d), R2) = {corr_d_r2:.4f}")
        if corr_d_r2 > 0.7:
            print("  >> CONFIRMED: linearization improves with dimension (native HD phenomenon)")
        elif corr_d_r2 > 0.3:
            print("  >> MODERATE: partial dimension dependence")
        elif corr_d_r2 < -0.3:
            print("  >> REVERSED: linearization gets WORSE with dimension")
        else:
            print("  >> FLAT: linearization quality doesn't depend on dimension")

    out_dir = Path(__file__).parent / "results" / "linearization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dim_scaling.json"

    def to_native(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=to_native)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
