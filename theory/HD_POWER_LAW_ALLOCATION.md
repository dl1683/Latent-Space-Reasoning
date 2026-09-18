# HP-1: Equilibrium or Transient? The Spectral Law of Zipf-Trained Superposition Codes

Status: PRE-REGISTERED (frozen before any run in this cycle). Design gate: pending.
Date opened: 2026-09-18. Program: mathematics of high-dimensional spaces (STATE.md, ACTIVE).
Everything numeric below is frozen at the commit that first adds this file.

## 1. Question

Trained representations have power-law frame spectra with exponent α depending
on data frequency structure (Zipf s) and overcompleteness γ = n/d (entries 95,
101). Two facts make the current interpretation unstable:

1. The rate-distortion benchmark for usage-weighted readout noise gives
   α* = s/2 (entry 117), but the single-layer toy of entry 101 measured
   α ≈ 1.0–2.4 — up to 5x steeper — and real embeddings sit at α ≈ 0.35–0.57
   with an estimated effective s ≈ 0.5 (entry 95), i.e. also far from s/2.
2. Entry 101 called its 3000-step spectrum a "steady state", but its recorded
   loss (5.54 → 2.70 over 3000 steps, still decreasing, non-monotone tail) is
   not converged. No time-axis test exists anywhere in the record.

Cycle question, exactly: **is the power-law exponent of a superposition code an
equilibrium quantity α_eq(s, γ) of the training objective, or a transient
α(s, γ, t) frozen by finite optimization time?** If transient, the toy-vs-real
exponent gap is a training-time artifact and the real quantity of interest is
the relaxation law, not the endpoint.

## 2. Definitions

- Vocabulary [n], usage p_i = i^{-s}/H_{n,s} (Zipf(s)); H = Σ i^{-s}.
- Code: W ∈ R^{d×n}, columns w_i; γ = n/d > 1 (superposition regime).
- Frame spectrum: eigenvalues of S = W W^T ∈ R^{d×d}, sorted λ_1 ≥ … ≥ λ_d.
- Spectral exponent α: OLS slope of log λ_k vs log k over k = 1..⌊0.9 d⌋
  (same estimator as entries 88–101). Sign convention: λ_k ~ k^{-α}, α > 0 steep.
- QSAE (quadratic superposition autoencoder): minimize
  L(W) = Σ_i p_i ‖W^T W e_i − e_i‖²
       = Σ_i p_i (‖w_i‖² − 1)² + Σ_{i≠j} p_i (w_i·w_j)².
- XE model (entry-101 replication): single layer, tied embeddings, full
  softmax cross-entropy on Zipf draws (spec in §6).

## 3. Propositions (derived; assumptions explicit)

**P1 (mean-field endpoint has no power law).** Assume random mixing:
directions u_i = w_i/‖w_i‖ satisfy E(u_i·u_j)² = 1/d for i≠j, treated as
independent of norms. With x_i = ‖w_i‖², X = Σx_i, M = Σ p_i x_i:

  Φ(x) = Σ_i p_i (x_i−1)² + d^{-1} Σ_i p_i x_i (X − x_i) + O(d^{-1}, diag).

Stationarity in x_k gives x_k = c_0 − c_1/p_k with c_0 = 1 − X/(2d),
c_1 = M/(2d) ≥ 0, clipped at 0 (feasibility). Ranked by usage
(p_(k) ∝ k^{-s}): x_(k) = c_0 − c_1 H k^s up to the cutoff k_0 = (c_0/c_1 H)^{1/s}.
The norm profile is a **bulge with hard cutoff, not a power law**, and (with
mixed directions) the frame spectrum inherits no power law — consistent with
entry 95's Monte Carlo (norms + random directions = flat). Conclusion carried
into testing: *under mean-field mixing, neither equilibrium nor norm
allocation produces power-law spectra; a power law is evidence of beyond-
mean-field directional correlation created by optimization.*

**P2 (rate-distortion benchmark).** With spectral budget Σλ = P and token-wise
readout noise σ²/λ_(i) charged at usage p_i, Lagrange optimization of
E = σ² Σ_i p_i / λ_(i) gives λ_(i)* ∝ √p_(i), i.e. α* = s/2 (entry 117,
restated). Any fitted α exceeding s/2 by more than tolerance is NOT explained
by this allocation-optimum.

**P3 (gradient-flow cutoff law).** For QSAE from small symmetric init, each
token's norm follows to leading order dx_i/dt = 4 p_i x_i (1 + o(1)) while
x_i ≪ 1 (the (x_i−1)² term drives growth at rate proportional to usage), so
x_i(t) ≈ x_i^* (1 − e^{−κ p_i t}) toward its endpoint value. Saturated set
size k_sat(t) = #{i : p_i κ t ≳ 1} grows as t^{1/s}. Fitted exponent over the
full window at time t interpolates accordingly; in particular α_fit(t) is
NON-STATIONARY while k_sat(t) < n, and the spectrum shows a two-slope
(saturated head, growing tail) shape. Exact frozen curve: α_pred(t) computed
from the closed form x_i(t) = x_i^*(1−e^{−κ p_i t}) with κ fit ONLY on the
first checkpoint of each run (calibration), all later times prediction.

## 4. Pre-registered predictions (frozen)

Grid: s ∈ {0.5, 1.0}; γ ∈ {4, 16} via (n,d) ∈ {(256,64), (1024,64)}; also
(n,d) = (1024,256) for a γ=4 large-d check. GD full-batch, lr tuned only on
loss at s=1,γ=16 arm; 3 seeds; numpy CPU; T_max = 2×10⁵ steps.

- **P-A (norm profile law, QSAE, primary).** At convergence (loss slope
  < 10⁻⁴/1000 steps), the ranked norm profile fits x_(k) = c_0 − c k^s
  (unclipped region) with R² > 0.9 in ≥ 2/3 seeds at each (s,γ).
- **P-B (mean-field adequacy, primary).** Mean-field shape (P-A form fed
  through random directions) fits the converged frame spectrum as well as a
  power law: ΔR² = R²_pl − R²_mf < 0.05 in ≥ 2/3 seeds. REFUTATION
  (ΔR² ≥ 0.05): optimization creates beyond-mean-field correlation — promote
  directional-organization theory to the HP-2 target.
- **P-C (transient, primary).** α_fit(t) changes by more than 0.10 between
  t = 10⁴ and T_max in the QSAE at s=1, γ=16 (non-stationary regime), and the
  calibrated α_pred(t) tracks measured α_fit(t) within ±0.10 at ≥ 3 of 5
  checkpoints.
- **P-D (XE replication, primary).** Replicating entry 101 (n=1024, d=64,
  s=1, tied, full softmax) with t ∈ {3k, 12k, 48k, 192k}: |α(48k) − α(3k)|
  > 0.15 ⇒ entry 101's 3000-step "steady state" claim was premature
  (transient regime); ≤ 0.05 ⇒ equilibrium regime, and the α(s,γ) surface at
  t=192k becomes the HP-2 derivation target. Between: inconclusive, extend t.
- **Predicted direction (all arms):** α_fit rises in log t toward its
  endpoint; never decreases (monotone rich-get-richer). Any decrease > 0.05
  between consecutive checkpoints refutes P3's mechanism.

## 5. Null ladder (each run as an arm, cheapest first)

- **N2 uniform usage** (s=0): flat spectrum, α < 0.35 (entry-101 control).
- **N1 random directions + endpoint norms** (from P-A fit): spectrum stays
  flat/no power law (entry-95 calibration at these (n,d)).
- **N0 ridge endpoint**: closed-form ridge solution at matched effective
  regularization; if its spectrum equals the GD endpoint spectrum
  (per-eigenvalue max relative deviation < 5%), dynamics add nothing beyond
  the stationary problem — dynamics-story scope collapses accordingly.

## 6. Experiment spec

One canonical runner `experiments/run_hp1_spectral_law.py` (CPU numpy only,
config explores the frozen grid). QSAE GD: full-batch gradient on the exact
loss; checkpoints at t ∈ {10³, 10⁴, 5×10⁴, 10⁵, 2×10⁵}. XE arm: SGD batch
256, Adam 3e-3 (entry-101-like), same checkpoints; record loss curves.
Outputs: per-run JSON (exponents, R² of both fits, norm profiles, spectra at
checkpoints, nulls) to `experiments/results/hp1_spectral_law/`. No model
downloads, no GPU. Measurement floor: 3 seeds, spread reported; tolerance
criteria above are the verdicts (exact tables diagnostic only).

## 7. What each outcome licenses

- P-A & P-B hold: mean-field theory of superposition endpoints adequate;
  power laws in real/toy codes are created by optimization dynamics or
  softmax geometry, not by norm allocation — narrows HP-2 to those mechanisms.
- P-B refuted: learned directional correlation is the power-law source —
  HP-2 targets its characterization (the genuinely new-math branch).
- P-C/P-D transient: the law is α(s,γ,t); toy-vs-real exponent gap resolved
  as training-time artifact; real-model exponents must be read against
  convergence stage, not as equilibrium constants.
- P-D equilibrium: HP-2 derives α_eq(s,γ) for softmax codes; rate-distortion
  s/2 benchmark then quantitatively falsifiable as the wrong noise model.
