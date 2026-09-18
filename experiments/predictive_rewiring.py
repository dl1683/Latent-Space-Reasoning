"""Predictive rewiring challenge: which predictor best forecasts permutation damage?

Compares four predictors of row-permutation damage on held-out data:
1. Decorrelation null: sqrt(2) * ||y|| (random baseline)
2. DU change: ||D(U-PU)x|| (ignores gate)
3. Tensor change: ||D[(Gx) * (Ux - PUx)]|| (includes gate, linear-gated)
4. Activation-weighted: ||D[silu(Gx) * (Ux - PUx)]|| (includes silu)

Uses partial permutations (corrupt 10-100% of rows) to get a gradient of damage.
Train correlation on 10 prompts, evaluate prediction on 5 held-out prompts.

Decision rule:
- Static tensor predicts held-out: continue weight-only analysis
- Only activation metrics predict: pivot to data-dependent analysis
- Neither beats DU/decorrelation nulls: close this line

Model: SmolLM2-135M (CPU-only)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch


def partial_permutation(m, frac, rng):
    """Permute frac of rows randomly, keep the rest fixed."""
    perm = np.arange(m)
    n_swap = max(2, int(m * frac))
    swap_idx = rng.choice(m, n_swap, replace=False)
    shuffled = rng.permutation(swap_idx)
    perm[swap_idx] = shuffled
    return perm


def run_challenge():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading SmolLM2-135M...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    m = model.config.intermediate_size
    print(f"  Loaded in {time.time()-t0:.1f}s (d={d}, m={m}, layers={n_layers})")

    train_prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In quantum mechanics, the wave function",
        "2 + 2 = 4, therefore 3 + 3 =",
        "The quick brown fox jumps over the",
        "import numpy as np\nnp.array([1, 2, 3])",
        "According to Einstein's theory of",
        "Once upon a time in a land far",
        "The derivative of sin(x) is",
        "SELECT * FROM users WHERE",
    ]
    test_prompts = [
        "Machine learning algorithms can",
        "The largest planet in our solar system",
        "for i in range(10):",
        "In the year 2025, technology",
        "The sum of angles in a triangle",
    ]

    all_prompts = train_prompts + test_prompts
    n_train = len(train_prompts)

    print("\nExtracting hidden states...")
    model.eval()
    hidden_by_layer = {l: [] for l in range(n_layers)}
    with torch.no_grad():
        for prompt in all_prompts:
            tokens = tokenizer(prompt, return_tensors="pt")
            outputs = model(**tokens, output_hidden_states=True)
            for l in range(n_layers):
                hidden_by_layer[l].append(outputs.hidden_states[l][0, -1, :])
    del outputs
    print(f"  Got {len(all_prompts)} x {n_layers} hidden states")

    test_layers = [0, 7, 15, 22, 29]
    fracs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    n_perms_per_frac = 10
    rng = np.random.default_rng(42)

    results = {}

    for l in test_layers:
        gate_w = model.model.layers[l].mlp.gate_proj.weight.detach()
        up_w = model.model.layers[l].mlp.up_proj.weight.detach()
        down_w = model.model.layers[l].mlp.down_proj.weight.detach()

        records = []

        for frac in fracs:
            for trial in range(n_perms_per_frac):
                perm = partial_permutation(m, frac, rng)
                perm_t = torch.tensor(perm, dtype=torch.long)
                up_perm = up_w[perm_t]

                train_actual = []
                train_du = []
                train_tensor = []
                train_activ = []
                train_null = []

                test_actual = []
                test_du = []
                test_tensor = []
                test_activ = []
                test_null = []

                for idx, x in enumerate(hidden_by_layer[l]):
                    gx = gate_w @ x
                    ux = up_w @ x
                    ux_perm = up_perm @ x

                    silu_gx = gx * torch.sigmoid(gx)

                    y_orig = down_w @ (silu_gx * ux)
                    y_perm = down_w @ (silu_gx * ux_perm)

                    actual_err = torch.norm(y_orig - y_perm).item()
                    du_pred = torch.norm(down_w @ (ux - ux_perm)).item()
                    tensor_pred = torch.norm(down_w @ (gx * (ux - ux_perm))).item()
                    activ_pred = torch.norm(down_w @ (silu_gx * (ux - ux_perm))).item()
                    null_pred = np.sqrt(2) * torch.norm(y_orig).item()

                    if idx < n_train:
                        train_actual.append(actual_err)
                        train_du.append(du_pred)
                        train_tensor.append(tensor_pred)
                        train_activ.append(activ_pred)
                        train_null.append(null_pred)
                    else:
                        test_actual.append(actual_err)
                        test_du.append(du_pred)
                        test_tensor.append(tensor_pred)
                        test_activ.append(activ_pred)
                        test_null.append(null_pred)

                records.append({
                    "frac": frac,
                    "trial": trial,
                    "train_actual_mean": float(np.mean(train_actual)),
                    "train_du_mean": float(np.mean(train_du)),
                    "train_tensor_mean": float(np.mean(train_tensor)),
                    "train_activ_mean": float(np.mean(train_activ)),
                    "train_null_mean": float(np.mean(train_null)),
                    "test_actual_mean": float(np.mean(test_actual)),
                    "test_du_mean": float(np.mean(test_du)),
                    "test_tensor_mean": float(np.mean(test_tensor)),
                    "test_activ_mean": float(np.mean(test_activ)),
                    "test_null_mean": float(np.mean(test_null)),
                })

        # Compute correlations across all permutations
        test_actuals = [r["test_actual_mean"] for r in records]
        test_dus = [r["test_du_mean"] for r in records]
        test_tensors = [r["test_tensor_mean"] for r in records]
        test_activs = [r["test_activ_mean"] for r in records]
        test_nulls = [r["test_null_mean"] for r in records]

        def corr(a, b):
            a, b = np.array(a), np.array(b)
            if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        def r_squared(pred, actual):
            pred, actual = np.array(pred), np.array(actual)
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            if ss_tot < 1e-10:
                return 0.0
            return float(1 - ss_res / ss_tot)

        r_du = corr(test_dus, test_actuals)
        r_tensor = corr(test_tensors, test_actuals)
        r_activ = corr(test_activs, test_actuals)
        r_null = corr(test_nulls, test_actuals)

        # R² requires scaling — use train data to fit linear scaling
        train_actuals_all = [r["train_actual_mean"] for r in records]
        train_dus_all = [r["train_du_mean"] for r in records]
        train_tensors_all = [r["train_tensor_mean"] for r in records]
        train_activs_all = [r["train_activ_mean"] for r in records]

        def fit_scale_and_r2(train_pred, train_actual, test_pred, test_actual):
            tp, ta = np.array(train_pred), np.array(train_actual)
            if np.std(tp) < 1e-10:
                return 0.0
            slope = np.dot(tp, ta) / np.dot(tp, tp)
            scaled = np.array(test_pred) * slope
            ss_res = np.sum((np.array(test_actual) - scaled) ** 2)
            ss_tot = np.sum((np.array(test_actual) - np.mean(test_actual)) ** 2)
            if ss_tot < 1e-10:
                return 0.0
            return float(1 - ss_res / ss_tot)

        r2_du = fit_scale_and_r2(train_dus_all, train_actuals_all, test_dus, test_actuals)
        r2_tensor = fit_scale_and_r2(train_tensors_all, train_actuals_all, test_tensors, test_actuals)
        r2_activ = fit_scale_and_r2(train_activs_all, train_actuals_all, test_activs, test_actuals)

        results[f"L{l}"] = {
            "n_permutations": len(records),
            "corr_null": r_null,
            "corr_du": r_du,
            "corr_tensor": r_tensor,
            "corr_activ": r_activ,
            "r2_du": r2_du,
            "r2_tensor": r2_tensor,
            "r2_activ": r2_activ,
            "records": records,
        }

        print(
            f"  L{l:2d}: corr(null)={r_null:.4f}, "
            f"corr(DU)={r_du:.4f}, "
            f"corr(tensor)={r_tensor:.4f}, "
            f"corr(activ)={r_activ:.4f} | "
            f"R2(DU)={r2_du:.4f}, "
            f"R2(tensor)={r2_tensor:.4f}, "
            f"R2(activ)={r2_activ:.4f}"
        )

    print("\n" + "=" * 70)
    print("PREDICTIVE REWIRING CHALLENGE — HELD-OUT RESULTS")
    print("=" * 70)

    for l in test_layers:
        r = results[f"L{l}"]
        print(f"\n  Layer {l}:")
        print(f"    Decorrelation null:  corr = {r['corr_null']:.4f}")
        print(f"    DU change:           corr = {r['corr_du']:.4f}, R2 = {r['r2_du']:.4f}")
        print(f"    Tensor (gate-aware): corr = {r['corr_tensor']:.4f}, R2 = {r['r2_tensor']:.4f}")
        print(f"    Activation-weighted: corr = {r['corr_activ']:.4f}, R2 = {r['r2_activ']:.4f}")

    # Decision
    mean_r2_du = np.mean([results[f"L{l}"]["r2_du"] for l in test_layers])
    mean_r2_tensor = np.mean([results[f"L{l}"]["r2_tensor"] for l in test_layers])
    mean_r2_activ = np.mean([results[f"L{l}"]["r2_activ"] for l in test_layers])

    print(f"\n  Mean R2 across layers:")
    print(f"    DU:     {mean_r2_du:.4f}")
    print(f"    Tensor: {mean_r2_tensor:.4f}")
    print(f"    Activ:  {mean_r2_activ:.4f}")

    if mean_r2_tensor > mean_r2_du + 0.05:
        print(f"\n  >> TENSOR PREDICTS: gate info adds {mean_r2_tensor-mean_r2_du:.3f} R2")
        print(f"  >> Continue weight-only analysis")
    elif mean_r2_activ > mean_r2_du + 0.05:
        print(f"\n  >> ACTIVATION PREDICTS: silu info adds {mean_r2_activ-mean_r2_du:.3f} R2")
        print(f"  >> Pivot to data-dependent analysis")
    elif mean_r2_du > 0.5:
        print(f"\n  >> DU SUFFICES: simple DU change predicts damage well")
        print(f"  >> No tensor/activation info needed — close this line")
    else:
        print(f"\n  >> NO PREDICTOR WORKS WELL")
        print(f"  >> Close this line — damage is too stochastic to predict")

    summary = {
        "model": "SmolLM2-135M",
        "d": d, "m": m, "n_layers": n_layers,
        "test_layers": test_layers,
        "n_train_prompts": n_train,
        "n_test_prompts": len(test_prompts),
        "fracs": fracs,
        "n_perms_per_frac": n_perms_per_frac,
        "mean_r2_du": mean_r2_du,
        "mean_r2_tensor": mean_r2_tensor,
        "mean_r2_activ": mean_r2_activ,
        "per_layer": {k: {kk: vv for kk, vv in v.items() if kk != "records"}
                      for k, v in results.items()},
    }

    out_dir = Path(__file__).parent / "results" / "row_permutation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictive_rewiring.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    run_challenge()
