"""Experiment: Do Neural Networks Introduce Non-Transferability?

Tests whether neural networks add non-transferability beyond what the
underlying task requires, using NK fitness landscapes as controlled ground truth.

Design:
- NK landscapes: N=16, K in [0, 5, 15]
- Architectures: MLP (16->128->128->1), Transformer (embed+2-layer+pool->1)
- Transfer test: mutation profile similarity on train positions (0-7)
  vs test positions (8-15), Spearman rho over 200-genotype pairs
- Checkpoints every 100 steps for 2000 steps

Key comparison: ground-truth landscape rho vs network rho at convergence.
If network rho < landscape rho, the architecture introduces artificial
non-transferability. If network rho matches, it faithfully captures structure.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist


class NKLandscape:
    """NK fitness landscape with non-circular (clamped) neighbors."""

    def __init__(self, N, K, seed=42):
        self.N = N
        self.K = K
        rng = np.random.RandomState(seed)
        self.tables = []
        self.neighbors = []
        for i in range(N):
            nbrs = [min(i + j + 1, N - 1) for j in range(K)]
            nbrs = list(dict.fromkeys(nbrs))
            positions = [i] + nbrs
            n_combos = 2 ** len(positions)
            self.tables.append(rng.random(n_combos))
            self.neighbors.append(positions)

    def fitness(self, g):
        total = 0.0
        for i in range(self.N):
            positions = self.neighbors[i]
            idx = sum(int(g[p]) * (1 << j) for j, p in enumerate(positions))
            total += self.tables[i][idx]
        return total / self.N

    def fitness_batch(self, genotypes):
        return np.array([self.fitness(g) for g in genotypes])

    def mutation_profile(self, g):
        base = self.fitness(g)
        profile = np.zeros(self.N)
        for i in range(self.N):
            gm = g.copy()
            gm[i] = 1 - gm[i]
            profile[i] = self.fitness(gm) - base
        return profile


class MLPPredictor(nn.Module):
    def __init__(self, N, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TransformerPredictor(nn.Module):
    def __init__(self, N, dim=64, heads=2, layers=2):
        super().__init__()
        self.N = N
        self.value_embed = nn.Embedding(2, dim)
        self.pos_embed = nn.Parameter(torch.randn(N, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        x_int = x.long()
        emb = self.value_embed(x_int) + self.pos_embed.unsqueeze(0)
        out = self.encoder(emb)
        pooled = out.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def compute_transfer_rho(mutation_profiles, train_pos, test_pos):
    """Compute transfer rho from mutation profiles."""
    train_p = mutation_profiles[:, train_pos]
    test_p = mutation_profiles[:, test_pos]

    train_vars = np.var(train_p, axis=1)
    test_vars = np.var(test_p, axis=1)
    active = (train_vars > 1e-12) & (test_vars > 1e-12)

    if active.sum() < 5:
        return float('nan'), 0

    train_d = pdist(train_p[active], metric='euclidean')
    test_d = pdist(test_p[active], metric='euclidean')

    rho, _ = spearmanr(train_d, test_d)
    return float(rho), int(active.sum())


def get_nn_mutation_profiles(model, genotypes, landscape):
    """Compute mutation profiles using network predictions."""
    N = genotypes.shape[1]
    n = genotypes.shape[0]
    profiles = np.zeros((n, N))

    model.eval()
    with torch.no_grad():
        x_base = torch.tensor(genotypes, dtype=torch.float32)
        base_preds = model(x_base).numpy()

        for i in range(N):
            x_mut = genotypes.copy()
            x_mut[:, i] = 1 - x_mut[:, i]
            x_mut_t = torch.tensor(x_mut, dtype=torch.float32)
            mut_preds = model(x_mut_t).numpy()
            profiles[:, i] = mut_preds - base_preds

    return profiles


def train_and_measure(model, landscape, train_genotypes, train_fitness,
                      transfer_genotypes, gt_profiles,
                      train_pos, test_pos,
                      n_steps=2000, checkpoint_every=100, batch_size=128, lr=1e-3):
    """Train model and measure transfer rho at checkpoints."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_train = torch.tensor(train_genotypes, dtype=torch.float32)
    y_train = torch.tensor(train_fitness, dtype=torch.float32)

    n_train = len(train_genotypes)
    checkpoints = []

    # Step 0 (random)
    nn_profiles = get_nn_mutation_profiles(model, transfer_genotypes, landscape)
    rho, n_active = compute_transfer_rho(nn_profiles, train_pos, test_pos)

    model.eval()
    with torch.no_grad():
        val_pred = model(x_train).numpy()
        val_r = float(np.corrcoef(val_pred, train_fitness)[0, 1]) if np.std(val_pred) > 1e-10 else 0.0

    checkpoints.append({
        "step": 0, "transfer_rho": rho, "n_active": n_active,
        "train_r": val_r, "loss": float('nan'),
    })

    rng = np.random.RandomState(99)
    for step in range(1, n_steps + 1):
        model.train()
        idx = rng.choice(n_train, size=min(batch_size, n_train), replace=False)
        xb = x_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % checkpoint_every == 0 or step == n_steps:
            model.eval()
            nn_profiles = get_nn_mutation_profiles(model, transfer_genotypes, landscape)
            rho, n_active = compute_transfer_rho(nn_profiles, train_pos, test_pos)

            with torch.no_grad():
                val_pred = model(x_train).numpy()
                val_r = float(np.corrcoef(val_pred, train_fitness)[0, 1]) if np.std(val_pred) > 1e-10 else 0.0

            checkpoints.append({
                "step": step, "transfer_rho": rho, "n_active": n_active,
                "train_r": val_r, "loss": float(loss.item()),
            })

    return checkpoints


def main():
    N = 16
    K_VALUES = [0, 5, 15]
    TRAIN_POS = list(range(0, 8))
    TEST_POS = list(range(8, 16))
    N_TRAIN = 5000
    N_TRANSFER = 200
    N_STEPS = 2000
    CKPT_EVERY = 100
    BATCH = 128

    print("=" * 70)
    print("NK NEURAL NETWORK TRANSFER EXPERIMENT")
    print("Do neural networks introduce non-transferability beyond the task?")
    print("=" * 70)
    print(f"N={N}, K values={K_VALUES}")
    print(f"Train positions: {TRAIN_POS}, Test positions: {TEST_POS}")
    print(f"Training: {N_TRAIN} genotypes, {N_STEPS} steps, batch {BATCH}")
    print(f"Transfer test: {N_TRANSFER} genotypes\n")

    all_results = {}

    for K in K_VALUES:
        print(f"\n{'='*60}")
        print(f"K = {K}")
        print(f"{'='*60}")

        landscape = NKLandscape(N, K, seed=42 + K)

        rng = np.random.RandomState(42)
        train_genos = rng.randint(0, 2, size=(N_TRAIN, N))
        train_fitness = landscape.fitness_batch(train_genos)

        rng2 = np.random.RandomState(123)
        transfer_genos = rng2.randint(0, 2, size=(N_TRANSFER, N))

        gt_profiles = np.array([landscape.mutation_profile(g) for g in transfer_genos])
        gt_rho, gt_active = compute_transfer_rho(gt_profiles, TRAIN_POS, TEST_POS)
        print(f"  Ground truth transfer rho: {gt_rho:.3f} ({gt_active} active genotypes)")

        k_results = {"K": K, "ground_truth_rho": gt_rho, "architectures": {}}

        for arch_name, ModelClass in [("MLP", MLPPredictor), ("Transformer", TransformerPredictor)]:
            print(f"\n  --- {arch_name} ---")
            t0 = time.time()

            if arch_name == "MLP":
                model = MLPPredictor(N, hidden=128)
            else:
                model = TransformerPredictor(N, dim=64, heads=2, layers=2)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}")

            checkpoints = train_and_measure(
                model, landscape, train_genos, train_fitness,
                transfer_genos, gt_profiles, TRAIN_POS, TEST_POS,
                n_steps=N_STEPS, checkpoint_every=CKPT_EVERY,
                batch_size=BATCH, lr=1e-3,
            )

            elapsed = time.time() - t0
            final = checkpoints[-1]
            init = checkpoints[0]

            print(f"  Time: {elapsed:.1f}s")
            print(f"  Random init: rho={init['transfer_rho']:.3f}, train_r={init['train_r']:.3f}")
            print(f"  Converged:   rho={final['transfer_rho']:.3f}, train_r={final['train_r']:.3f}")
            print(f"  Ground truth: rho={gt_rho:.3f}")
            gap = gt_rho - final['transfer_rho'] if not (np.isnan(gt_rho) or np.isnan(final['transfer_rho'])) else float('nan')
            print(f"  Gap (GT - network): {gap:+.3f}")

            k_results["architectures"][arch_name] = {
                "n_params": n_params,
                "checkpoints": checkpoints,
                "final_rho": final["transfer_rho"],
                "init_rho": init["transfer_rho"],
                "gap_vs_gt": gap,
                "time_s": elapsed,
            }

        all_results[f"K={K}"] = k_results

    # --- Summary Table ---
    print(f"\n\n{'='*70}")
    print("SUMMARY: Transfer Rho Across Architectures and Task Complexity")
    print(f"{'='*70}")
    print(f"\n{'K':>3s} {'Ground Truth':>13s} {'MLP init':>10s} {'MLP final':>10s} {'MLP gap':>9s}"
          f" {'TF init':>9s} {'TF final':>9s} {'TF gap':>8s}")
    print("-" * 75)

    for K in K_VALUES:
        r = all_results[f"K={K}"]
        gt = r["ground_truth_rho"]
        mlp = r["architectures"]["MLP"]
        tf = r["architectures"]["Transformer"]

        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "N/A"

        print(f"{K:>3d} {fmt(gt):>13s} {fmt(mlp['init_rho']):>10s} {fmt(mlp['final_rho']):>10s}"
              f" {fmt(mlp['gap_vs_gt']):>9s} {fmt(tf['init_rho']):>9s} {fmt(tf['final_rho']):>9s}"
              f" {fmt(tf['gap_vs_gt']):>8s}")

    # Training dynamics
    print(f"\n--- Training Dynamics (rho at checkpoints) ---")
    for K in K_VALUES:
        r = all_results[f"K={K}"]
        gt = r["ground_truth_rho"]
        print(f"\nK={K} (ground truth rho={gt:.3f}):")
        print(f"  {'Step':>6s} {'MLP rho':>9s} {'MLP r':>7s} {'TF rho':>8s} {'TF r':>6s}")
        print(f"  {'-'*38}")
        mlp_ckpts = r["architectures"]["MLP"]["checkpoints"]
        tf_ckpts = r["architectures"]["Transformer"]["checkpoints"]
        for mc, tc in zip(mlp_ckpts, tf_ckpts):
            step = mc["step"]
            if step == 0 or step % 500 == 0 or step == N_STEPS:
                mr = f"{mc['transfer_rho']:.3f}" if not np.isnan(mc['transfer_rho']) else "N/A"
                tr = f"{tc['transfer_rho']:.3f}" if not np.isnan(tc['transfer_rho']) else "N/A"
                print(f"  {step:>6d} {mr:>9s} {mc['train_r']:>7.3f} {tr:>8s} {tc['train_r']:>6.3f}")

    # Key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    for K in K_VALUES:
        r = all_results[f"K={K}"]
        gt = r["ground_truth_rho"]
        mlp_final = r["architectures"]["MLP"]["final_rho"]
        tf_final = r["architectures"]["Transformer"]["final_rho"]
        mlp_init = r["architectures"]["MLP"]["init_rho"]
        tf_init = r["architectures"]["Transformer"]["init_rho"]

        print(f"\nK={K}:")
        print(f"  Ground truth rho = {gt:.3f}")

        if not np.isnan(mlp_final) and not np.isnan(gt):
            if abs(mlp_final - gt) < 0.05:
                print(f"  MLP matches ground truth ({mlp_final:.3f} vs {gt:.3f})")
            elif mlp_final < gt:
                print(f"  MLP BELOW ground truth ({mlp_final:.3f} vs {gt:.3f}) — network adds non-transferability")
            else:
                print(f"  MLP ABOVE ground truth ({mlp_final:.3f} vs {gt:.3f}) — network imposes artificial coupling")

        if not np.isnan(tf_final) and not np.isnan(gt):
            if abs(tf_final - gt) < 0.05:
                print(f"  Transformer matches ground truth ({tf_final:.3f} vs {gt:.3f})")
            elif tf_final < gt:
                print(f"  Transformer BELOW ground truth ({tf_final:.3f} vs {gt:.3f}) — network adds non-transferability")
            else:
                print(f"  Transformer ABOVE ground truth ({tf_final:.3f} vs {gt:.3f}) — network imposes artificial coupling")

        if not np.isnan(mlp_init) and not np.isnan(tf_init):
            print(f"  Random init: MLP={mlp_init:.3f}, Transformer={tf_init:.3f}")

    # Save results
    out_dir = Path("experiments/results/nk_nn_transfer")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    save_data = {}
    for key, val in all_results.items():
        save_data[key] = {
            "K": val["K"],
            "ground_truth_rho": val["ground_truth_rho"],
            "architectures": {}
        }
        for arch, adata in val["architectures"].items():
            save_data[key]["architectures"][arch] = {
                "n_params": adata["n_params"],
                "final_rho": adata["final_rho"],
                "init_rho": adata["init_rho"],
                "gap_vs_gt": adata["gap_vs_gt"],
                "time_s": adata["time_s"],
                "checkpoints": adata["checkpoints"],
            }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
