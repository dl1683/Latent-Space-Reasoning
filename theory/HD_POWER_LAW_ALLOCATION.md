# HP: Spectral Law of Zipf-Trained Superposition Codes (v2)

Status: PRE-REGISTERED v2, frozen before any HP-A/HP-B run. v1 was gated
NO-GO by an independent Droid design review (ledger `hp1_design_gate_droid`);
v2 incorporates its five required fixes and its strategic steer (analytic
foundation cycle first). v1 is preserved in git history.
Date: 2026-09-18. Program: mathematics of high-dimensional spaces (ACTIVE).

## 0. Structure

- **HP-A (this cycle): analytic + Monte-Carlo foundation.** No SGD. Exact
  mean-field stationarity of the superposition objective; the finite-d
  random-frame spectral map; estimator validity (when is a fitted power-law
  exponent a measurement rather than an artifact); exact early gradient-flow
  law with an independent pilot calibration.
- **HP-B (next, after HP-A): optimizer cycle.** Tests transient-vs-equilibrium
  on trained codes with the repaired design.

## 1. Question (unchanged in substance)

Is the power-law exponent of a superposition code an equilibrium quantity of
the training objective or a transient frozen by finite optimization time?
Entry 101's "steady state" claim is unsupported (its loss is still decreasing
at t=3000); toy exponents (≈1.0–2.4) sit far above the rate-distortion
benchmark s/2 and above real-model exponents (≈0.35–0.57).

## 2. Definitions

- p_i = i^{-s}/H_{n,s}; W ∈ R^{d×n}; γ = n/d; x_i = ‖w_i‖²; frame spectrum =
  eigenvalues of WW^T sorted desc; λ_k ~ k^{-α} by convention.
- QSAE: L(W) = Σ_i p_i (x_i−1)² + Σ_{i≠j} p_i x_i x_j (u_i·u_j)².
- Estimator battery (frozen, used everywhere): windows {top-50%, top-90%,
  full-d} of ranked eigenvalues; within each window two fits — power law
  (log λ = a − b log k) and curved alternative (log λ = a + b log k + c log²k);
  ΔR² = R²_pl − R²_quad. Primary window: top-90%.

## 3. Corrected propositions

**P1′ (exact mean-field stationarity).** Under random mixing (E(u_i·u_j)² =
1/d, i≠j, independent of norms), Φ(x) = Σ_i p_i(x_i−1)² + d^{-1}[M X − Σ_i
p_i x_i²] with X = Σx_i, M = Σ p_i x_i. Stationarity gives EXACTLY

  x_k = (d/(d−1)) · [1 − (X + M/p_k)/(2d)],  clipped at 0,

with (X, M) determined self-consistently (two-scalar fixed point). The optimal
norm profile is a bulge with a usage cutoff k₀ = (2d c₀/(c₁H))^{1/s}-shaped
boundary — NOT a power law. No flatness claim for the induced frame spectrum
is made here; that map is finite-d and measured in A2. Rate-distortion s/2 is
retired as a direct comparator (n tokens vs d modes; no token-to-mode
assignment is defined) and kept only as a labeled diagnostic.

**P3′ (exact early flow).** The exact QSAE gradient at small ‖W‖ is dominated
by the −4·W D_p term, so dx_i/dt = 8 p_i x_i (continuous time t = lr·steps):
growth is EXPONENTIAL at usage-proportional rates, saturating logistically
toward x_i^*: x_i(t) ≈ x_i^*/(1 + (x_i^*/x_i(0) − 1)e^{−8 p_i t}).
Saturation clock t_i ∝ ln(x_i^*/x_i(0))/(8p_i). HD sharpening: for Gaussian
init, x_i(0) concentrates (relative fluctuations O(√(2/d))), so ln x_i(0)
spreads only O(1/√d) — concentration of measure makes the formation transient
SHARP. k_sat(t) ∝ t^{1/s} up to those log-corrections. No monotone-α claim:
whether α_fit(t) is monotone is an outcome, not a prediction.

## 4. HP-A pre-registration (frozen)

**A1 (mean-field endpoint adequacy).** Solve the P1′ fixed point for
(s ∈ {0.5, 1.0}) × (n,d) ∈ {(64,16),(256,64),(512,128)}. Compare to L-BFGS
optima of the true QSAE (5 independent inits each, converged: gtol 1e-10).
Verdict on norm-profile shape R² (fit of x_(k) to the P1′ form on the
unclipped region, median over inits): > 0.9 mean-field adequate; 0.8–0.9
partial; < 0.8 refuted (optimal directional correlations beyond mean field —
promote to HP-C target).

**A2 (random-frame spectral map + estimator validity).** Monte Carlo (20
draws): norm profiles from (a) A1 bulge solutions, (b) power-law x_(k) ∝
k^{−β}, β ∈ {0.25, 0.5, 1.0}, (c) uniform; d ∈ {32,…,1024} ×6, γ ∈ {4, 16}.
Pre-registered verdicts:
- A2-i: PL input, d ≥ 256: |α̂ − β| ≤ 0.10 in top-90% window (medians).
- A2-ii: bulge input with k₀ < d/2: power-law R² > 0.95 in ≥ 1 of 3 windows
  at d ≤ 64 (manufactured power laws quantified) AND ΔR² flags curvature
  (R²_quad − R²_pl ≥ 0.02 in the same window).
- A2-iii: uniform input: α̂ ≤ 0.35 at all d ≥ 64 in all windows.
- A2-iv (new, window-stability rule): any later trained-code α claim requires
  |α̂(top50) − α̂(top90)| ≤ 0.15 and ΔR² ≥ −0.02 (curved not better).

**A3 (transient law + independent pilot).** Build α_pred(t) by mapping the
P3′ logistic norm trajectories (x^* from A1, x(0) from Gaussian init) through
the A2 spectral map at (d=64, γ=16, s=1). The single free rate constant is
calibrated ONLY on an independent pilot: direct Euler integration of the
exact QSAE gradient flow at (64,16,s=1), seeds 90–92, matching α at the
pilot's FIRST checkpoint only. Grid runs never touch the calibration.

**N1-t (adopted control, standing).** At any checkpoint of any trained code:
preserve learned column norms, redraw directions independently; compare α
trajectories. Separates norm-driven from direction-driven spectral evolution.

## 4b. A1 result and theorem (recorded 2026-09-18, before any HP-B run)

**A1 verdict: mean-field ansatz REFUTED at every configuration** (profile-fit
R² 0.00–0.08 vs thresholds 0.9/0.8; identity R² 0.10–0.53). The reason is a
theorem the L-BFGS optima exposed (loss matched Σ_{i>d} p_i to 6 decimals at
all six (s,n,d)):

**Theorem (sparse-exact optimum of the QSAE).** Let G = W^T W (rank ≤ d).
L = tr((G−I) P (G−I)) with P = diag(p). G is PSD with rank ≤ d, so G−I has
an (n−d)-dimensional kernel-inherited subspace on which it acts as −I
(eigenvalues of G on ker G are 0). Hence L ≥ tr(P|_{ker G}), minimized by
choosing ker G = span{e_i : i ∉ S} with S the top-d usage set, giving
L_min = Σ_{i∉S} p_i, attained by ANY orthonormal unit-norm set {w_i}_{i∈S}
(G = coordinate projector onto S). The optimal frame operator is W W^T = I_d:
**completely flat spectrum, condition number 1.**

Consequences (now part of the cycle's frozen expectations):
- **P-E (added before any HP-B run):** converged QSAE GD endpoints show
  α̂(top90) ≤ 0.35, ≈ d nonzero norms, and loss within 1e-6 of the tail mass.
- Power-law spectra CANNOT come from the quadratic objective's optimum. They
  must arise from (i) optimization dynamics (transients) or (ii) non-quadratic
  retrieval geometry (softmax margins). The XE arm is therefore the main event
  of HP-B; the QSAE arms test the dynamics-only branch.
- HD-native moral: the quadratic reconstruction loss has no superposition
  premium — with d ≪ n, interference is avoided by DROPPING tokens. The cost
  of superposition only bites when every token must remain retrievable
  (softmax), which is the real-embedding regime. The rate-distortion s/2
  benchmark and the mean-field bulge both describe the wrong object for this
  loss and are retired as comparators for QSAE.

## 5. HP-B design (repaired; frozen after HP-A lands)

- Grid: s ∈ {0.5,1.0}; (n,d) ∈ {(256,64),(1024,64),(1024,256)} plus (4096,64)
  s=1 only (γ=64); 3 seeds (0–2). No α(s,γ)-surface claim from this grid —
  mechanism claims only.
- lr frozen by independent loss-based pilot (seeds 90–92; choose lowest final
  loss among stable {0.002,0.01,0.05,0.2} at (1024,64,s=1), 2k steps).
- QSAE GD checkpoints {1e3,1e4,5e4,1e5,2e5}. Convergence gate (all three):
  |α(last two checkpoints)| ≤ 0.05 (top-90 window), loss slope < 1e-5/1k
  steps, grad-norm < 1e-3 × init grad-norm. Frozen extension schedule: ×4
  steps, up to 3 extensions, else verdict UNCONVERGED (no endpoint claim).
- P-A′ (endpoint profile): fit P1′ form on x_k > 1e-8·max x; verdict on
  median-of-seeds R² with the A1 thresholds.
- P-B′ (mean-field spectral adequacy): endpoint spectrum vs median of 5
  N1-t-style random-direction replicates with the fitted endpoint norms;
  ΔR²_identity form (log-actual on log-model); scoring rule frozen: adequate
  if R²_pl(actual) − R²_identity ≤ 0.05.
- P-C′ (transient): score α_pred(t) vs measured at the 4 NON-calibration
  checkpoints; pass = |dev| ≤ 0.10 at ≥ 3 of 4.
- P-D′ (XE replication, entry-101 config n=1024,d=64,s=1, tied softmax,
  Adam 3e-3, batch 256): checkpoints {3k,12k,48k,192k}; equilibrium gate =
  A2-iv window stability + the convergence gate (loss slope, grad norm);
  frozen extension ×4 up to 3×. Interpretations pre-registered both ways.
- Nulls: N2 uniform usage (flat, α̂ ≤ 0.35); N1 endpoint randomization;
  N0 L-BFGS multi-init vs GD endpoint (max per-eigenvalue rel dev < 5% ⇒
  objective-determined).

## 6. What each outcome licenses

- A1 refutes mean field at the optimum: directional correlation is created by
  the OBJECTIVE itself (not just dynamics) — HP-C targets its structure.
- A2-ii confirms manufactured power laws: a measurement-validity constraint
  retroactively qualifies every prior d ≤ 64 spectral reading in the record.
- HP-B transient wins: the law is α(t); toy-vs-real exponent gap is a
  training-time artifact. HP-B equilibrium wins: HP-C derives the softmax
  equilibrium surface α_eq(s,γ) on a widened grid.
