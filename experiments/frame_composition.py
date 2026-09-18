"""Frame composition experiment: does pairwise geometry predict composition?

Tests whether the marginal spectral law of two frame operators suffices to
predict their composition spectrum, or whether mixed moments (eigenvector
alignment) carry essential additional information.

Six ensembles, matched marginal spectra, varied eigenvector structure.
Pure CPU, deterministic seeds, ~10-30min total.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import linalg


def generate_haar_orthogonal(d, rng):
    """Random orthogonal matrix from Haar measure."""
    H = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def power_law_spectrum(d, alpha, normalize=True):
    """Power-law eigenvalues: lambda_k = k^{-alpha}, k=1..d."""
    lam = np.arange(1, d + 1, dtype=np.float64) ** (-alpha)
    if normalize:
        lam = lam / np.mean(lam)  # normalize so tr(S)/d = 1
    return lam


def make_frame_operator(eigenvalues, eigenvectors):
    """S = U @ diag(lambda) @ U^T."""
    return (eigenvectors * eigenvalues[np.newaxis, :]) @ eigenvectors.T


def generate_gaussian_pair(d, gamma, rng):
    """Two independent Gaussian frame operators. Marchenko-Pastur baseline."""
    m = int(d * gamma)
    V = rng.standard_normal((d, m)) / np.sqrt(m)
    W = rng.standard_normal((d, m)) / np.sqrt(m)
    S = V @ V.T
    T = W @ W.T
    return S, T


def generate_powerlaw_random_pair(d, alpha, rng):
    """Two power-law frame operators with INDEPENDENT random eigenvectors."""
    lam = power_law_spectrum(d, alpha)
    U1 = generate_haar_orthogonal(d, rng)
    U2 = generate_haar_orthogonal(d, rng)
    S = make_frame_operator(lam, U1)
    T = make_frame_operator(lam, U2)
    return S, T


def generate_powerlaw_shared_pair(d, alpha, rank_frac, rng):
    """Two power-law frame operators SHARING top-r eigenvectors."""
    lam = power_law_spectrum(d, alpha)
    r = max(1, int(d * rank_frac))

    # Shared eigenvectors for top-r eigenvalues
    U_shared = generate_haar_orthogonal(d, rng)[:, :r]

    # Independent eigenvectors for remaining d-r
    U1_rest = generate_haar_orthogonal(d, rng)[:, :d - r]
    U2_rest = generate_haar_orthogonal(d, rng)[:, :d - r]

    # Orthogonalize rest against shared subspace
    U1_rest = U1_rest - U_shared @ (U_shared.T @ U1_rest)
    U1_rest, _ = np.linalg.qr(U1_rest)
    U2_rest = U2_rest - U_shared @ (U_shared.T @ U2_rest)
    U2_rest, _ = np.linalg.qr(U2_rest)

    U1 = np.hstack([U_shared, U1_rest[:, :d - r]])
    U2 = np.hstack([U_shared, U2_rest[:, :d - r]])

    S = make_frame_operator(lam, U1)
    T = make_frame_operator(lam, U2)
    return S, T


def generate_tight_frame_pair(d, rng):
    """Two tight frame operators (uniform spectrum)."""
    lam = np.ones(d)
    U1 = generate_haar_orthogonal(d, rng)
    U2 = generate_haar_orthogonal(d, rng)
    S = make_frame_operator(lam, U1)
    T = make_frame_operator(lam, U2)
    return S, T


def generate_sparse_pair(d, gamma, rng):
    """Two sparse random frame operators (k nonzeros per column)."""
    m = int(d * gamma)
    k = max(2, int(np.sqrt(d)))

    def sparse_frame(d, m, k, rng):
        V = np.zeros((d, m))
        for j in range(m):
            rows = rng.choice(d, size=k, replace=False)
            vals = rng.standard_normal(k) / np.sqrt(k)
            V[rows, j] = vals
        return V

    V = sparse_frame(d, m, k, rng)
    W = sparse_frame(d, m, k, rng)
    S = V @ V.T / m
    T = W @ W.T / m
    return S, T


def generate_empirical_pair(d, empirical_spectrum, rng):
    """Power-law spectrum matching empirical data, random eigenvectors."""
    # Interpolate empirical spectrum to dimension d
    n_emp = len(empirical_spectrum)
    indices = np.linspace(0, n_emp - 1, d)
    lam = np.interp(indices, np.arange(n_emp), empirical_spectrum)
    lam = lam / np.mean(lam)  # normalize

    U1 = generate_haar_orthogonal(d, rng)
    U2 = generate_haar_orthogonal(d, rng)
    S = make_frame_operator(lam, U1)
    T = make_frame_operator(lam, U2)
    return S, T


def mixed_moments(S, T, k_max=3):
    """Compute mixed moments D_k(S,T) for k=1..k_max.

    D_k = (1/d) tr[(S - tau_S * I)(T - tau_T * I)]^k
    where tau_X = tr(X)/d.
    """
    d = S.shape[0]
    tau_S = np.trace(S) / d
    tau_T = np.trace(T) / d
    S_c = S - tau_S * np.eye(d)
    T_c = T - tau_T * np.eye(d)

    moments = {}
    # Build the alternating product
    M = S_c @ T_c  # (S-tau I)(T-tau I)
    for k in range(1, k_max + 1):
        if k == 1:
            moments[k] = np.trace(M) / d
        else:
            Mk = np.linalg.matrix_power(M, k)
            moments[k] = np.trace(Mk) / d
    return moments


def spectral_distance(spec1, spec2):
    """L2 Wasserstein distance between two empirical spectral distributions."""
    s1 = np.sort(spec1)
    s2 = np.sort(spec2)
    # Interpolate to common grid if different lengths
    if len(s1) != len(s2):
        n = max(len(s1), len(s2))
        q1 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s1)), s1)
        q2 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s2)), s2)
    else:
        q1, q2 = s1, s2
    return np.sqrt(np.mean((q1 - q2) ** 2))


def run_trial(d, gamma, alpha, rank_frac, rng, empirical_spectrum=None):
    """Run one trial: generate all ensemble pairs, compute all metrics."""
    results = {}

    # 1. Gaussian
    S, T = generate_gaussian_pair(d, gamma, rng)
    results["gaussian"] = analyze_pair(S, T, "gaussian")

    # 2. Power-law random
    S, T = generate_powerlaw_random_pair(d, alpha, rng)
    results["powerlaw_random"] = analyze_pair(S, T, "powerlaw_random")

    # 3. Power-law shared
    S, T = generate_powerlaw_shared_pair(d, alpha, rank_frac, rng)
    results["powerlaw_shared"] = analyze_pair(S, T, "powerlaw_shared")

    # 4. Tight frame
    S, T = generate_tight_frame_pair(d, rng)
    results["tight"] = analyze_pair(S, T, "tight")

    # 5. Sparse random
    S, T = generate_sparse_pair(d, gamma, rng)
    results["sparse"] = analyze_pair(S, T, "sparse")

    # 6. Empirical spectrum (if available)
    if empirical_spectrum is not None and d <= len(empirical_spectrum):
        S, T = generate_empirical_pair(d, empirical_spectrum, rng)
        results["empirical"] = analyze_pair(S, T, "empirical")

    return results


def analyze_pair(S, T, label):
    """Analyze one pair of frame operators."""
    d = S.shape[0]

    # Marginal spectra
    spec_S = np.sort(np.linalg.eigvalsh(S))[::-1]
    spec_T = np.sort(np.linalg.eigvalsh(T))[::-1]

    # Composition spectra
    spec_add = np.sort(np.linalg.eigvalsh((S + T) / 2))[::-1]
    spec_mult = np.sort(np.linalg.eigvalsh(S @ T))[::-1]

    # Average of marginal spectra (what you'd predict from marginals alone)
    avg_marginal = (spec_S + spec_T) / 2

    # Mixed moments
    mm = mixed_moments(S, T, k_max=3)

    # Spectral distances
    add_vs_avg = spectral_distance(spec_add, avg_marginal)

    # Frobenius inner product (measures alignment)
    frob_inner = np.trace(S @ T) / d

    return {
        "label": label,
        "d": d,
        "spec_S_summary": {
            "mean": float(np.mean(spec_S)),
            "std": float(np.std(spec_S)),
            "max": float(spec_S[0]),
            "min": float(spec_S[-1]),
            "condition": float(spec_S[0] / max(spec_S[-1], 1e-15)),
        },
        "mixed_moments": {str(k): float(v) for k, v in mm.items()},
        "composition_add_vs_avg_marginal": float(add_vs_avg),
        "frob_inner_product": float(frob_inner),
        "spec_add_summary": {
            "mean": float(np.mean(spec_add)),
            "std": float(np.std(spec_add)),
            "max": float(spec_add[0]),
            "min": float(spec_add[-1]),
        },
        "spec_mult_summary": {
            "mean": float(np.mean(spec_mult)),
            "std": float(np.std(spec_mult)),
            "max": float(spec_mult[0]),
            "min": float(spec_mult[-1]),
        },
    }


def main():
    dims = [64, 128, 256, 512, 1024]
    gammas = [2, 4]
    alpha = 0.8  # from entry 89
    rank_frac = 0.25  # share top 25% of eigenvectors
    n_trials = 30  # enough for statistics, fast enough for large d
    seed = 42

    rng_master = np.random.default_rng(seed)

    all_results = {
        "config": {
            "dims": dims,
            "gammas": gammas,
            "alpha": alpha,
            "rank_frac": rank_frac,
            "n_trials": n_trials,
            "seed": seed,
        },
        "trials": [],
    }

    total = len(dims) * len(gammas) * n_trials
    done = 0
    t0 = time.time()

    for d in dims:
        for gamma in gammas:
            trial_results = []
            for trial in range(n_trials):
                trial_seed = rng_master.integers(0, 2**31)
                rng = np.random.default_rng(trial_seed)
                result = run_trial(d, gamma, alpha, rank_frac, rng)
                trial_results.append(result)
                done += 1
                if done % 10 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    remaining = (total - done) / rate
                    print(
                        f"  [{done}/{total}] d={d} gamma={gamma} "
                        f"trial={trial+1}/{n_trials} "
                        f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
                    )

            # Aggregate statistics across trials
            agg = aggregate_trials(trial_results)
            all_results["trials"].append(
                {"d": d, "gamma": gamma, "n_trials": n_trials, "aggregate": agg}
            )

    elapsed = time.time() - t0
    all_results["runtime_seconds"] = elapsed
    print(f"\nDone in {elapsed:.1f}s")

    # Save results
    out_dir = Path(__file__).parent / "results" / "frame_composition"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "composition_test.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results: {out_path}")

    # Print summary table
    print_summary(all_results)


def aggregate_trials(trial_results):
    """Aggregate metrics across trials for each ensemble."""
    ensembles = set()
    for tr in trial_results:
        ensembles.update(tr.keys())

    agg = {}
    for ens in ensembles:
        ens_results = [tr[ens] for tr in trial_results if ens in tr]
        if not ens_results:
            continue

        mm1 = [r["mixed_moments"]["1"] for r in ens_results]
        mm2 = [r["mixed_moments"]["2"] for r in ens_results]
        mm3 = [r["mixed_moments"]["3"] for r in ens_results]
        add_dev = [r["composition_add_vs_avg_marginal"] for r in ens_results]
        frob = [r["frob_inner_product"] for r in ens_results]

        agg[ens] = {
            "D1_mean": float(np.mean(mm1)),
            "D1_std": float(np.std(mm1)),
            "D2_mean": float(np.mean(mm2)),
            "D2_std": float(np.std(mm2)),
            "D3_mean": float(np.mean(mm3)),
            "D3_std": float(np.std(mm3)),
            "add_deviation_mean": float(np.mean(add_dev)),
            "add_deviation_std": float(np.std(add_dev)),
            "frob_inner_mean": float(np.mean(frob)),
            "frob_inner_std": float(np.std(frob)),
        }

    return agg


def print_summary(results):
    """Print a concise summary table."""
    print("\n" + "=" * 90)
    print("FRAME COMPOSITION: PAIRWISE GEOMETRY VS MIXED MOMENTS")
    print("=" * 90)

    for trial_group in results["trials"]:
        d = trial_group["d"]
        gamma = trial_group["gamma"]
        agg = trial_group["aggregate"]

        print(f"\n--- d={d}, gamma={gamma} ---")
        print(
            f"{'Ensemble':<20} {'D1(mean±std)':<22} {'D2(mean±std)':<22} "
            f"{'Add Dev(mean±std)':<22}"
        )
        print("-" * 86)

        for ens in [
            "gaussian",
            "powerlaw_random",
            "powerlaw_shared",
            "tight",
            "sparse",
            "empirical",
        ]:
            if ens not in agg:
                continue
            a = agg[ens]
            print(
                f"{ens:<20} "
                f"{a['D1_mean']:>+9.5f}±{a['D1_std']:<9.5f} "
                f"{a['D2_mean']:>+9.5f}±{a['D2_std']:<9.5f} "
                f"{a['add_deviation_mean']:>+9.5f}±{a['add_deviation_std']:<9.5f}"
            )

    # Key comparison: shared vs random power-law
    print("\n" + "=" * 90)
    print("KEY TEST: Power-law shared vs random (does eigenvector alignment matter?)")
    print("=" * 90)
    print(f"{'d':<8} {'gamma':<8} {'D1_shared':<14} {'D1_random':<14} {'ratio':<10} {'separation?'}")
    print("-" * 70)

    for trial_group in results["trials"]:
        d = trial_group["d"]
        gamma = trial_group["gamma"]
        agg = trial_group["aggregate"]

        if "powerlaw_shared" in agg and "powerlaw_random" in agg:
            d1_shared = agg["powerlaw_shared"]["D1_mean"]
            d1_random = agg["powerlaw_random"]["D1_mean"]
            d1_shared_std = agg["powerlaw_shared"]["D1_std"]
            d1_random_std = agg["powerlaw_random"]["D1_std"]

            ratio = abs(d1_shared) / max(abs(d1_random), 1e-15)
            # Z-score for separation
            pooled_std = np.sqrt(d1_shared_std**2 + d1_random_std**2)
            z = abs(d1_shared - d1_random) / max(pooled_std, 1e-15)
            sep = "YES" if z > 3 else ("maybe" if z > 2 else "no")

            print(
                f"{d:<8} {gamma:<8} {d1_shared:>+12.6f} {d1_random:>+12.6f} "
                f"{ratio:>8.1f}x   {sep} (z={z:.1f})"
            )


if __name__ == "__main__":
    main()
