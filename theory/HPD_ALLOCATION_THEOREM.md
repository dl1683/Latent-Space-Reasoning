# HP-D: The Usage-Weighted Margin-Allocation Theorem

Status: OPENED 2026-09-19 (post review-2 pivot; the verified unoccupied gap).
Math-first: this document states the full specification, the target theorems,
and the first reduction BEFORE any HP-D computation. Prior-art scoping
(ledger `priorart_allocation_gap`): no published derivation of eigenvalue
allocation for CE/softmax-trained representations from usage in the
overcomplete regime; balanced neural collapse predicts FLAT spectra;
reverse water-filling is MSE-only.

## 0. Specification (the discipline the adversarial review demanded)

- **Ensemble.** Vocabulary [n]; usage p_i = i^{-s}/H_{n,s} (monotone; Zipf
  family). Code W ∈ R^{d×n}, columns w_i = √x_i v_i, ‖v_i‖ = 1; γ = n/d ≥ 1.
- **Task.** Tied retrieval: context I ~ p, query direction q = v_I; target
  J | I ~ (1−ρ)·unigram(p) + ρ·structured(P(·|I)). ρ = 0 case is solved (H1:
  rank-2 collapse). The theorem concerns the joint-retrievability constraint
  structure the ρ > 0 part imposes.
- **Decoder.** argmax_j z_j, z_j = w_j·q (logit-margin m bounds softmax CE
  error by e^{−m} up to constants; both stated where used).
- **Noise/interference model.** For target j, interference logit I_j =
  Σ_{k≠j} √x_k (v_k·q)(v_j·v_k·correction) — subgaussian with variance proxy
  σ_j² = Σ_{k≠j} x_k (v_j·v_k)² c(q-geometry). Tail control beyond Gaussian
  via degree-2 concentration (Hoeffding/chaos; T3 below).
- **Error criterion.** Per-token misretrieval ε_j = P(argmax ≠ j | target j).
  Objective: minimize Σ_j p_j ε_j subject to spectral budget Σ_j x_j = P.
- **Capacity claim protocol.** Any k(d) threshold statement must fix all of
  the above before it is a theorem (reviewer's rule, binding).

## 1. Target theorems

**T1 (reliability floor).** ε_j ≤ exp(−t_j²/2) with t_j = (√x_j a_j − m_j)/σ_j
(a_j = query-target alignment) — Gaussian-proxy form; T3 replaces it with a
chaos tail carrying explicit d-dependence.

**T2 (allocation — the main target).** The budget-optimal profile x* solving
the KKT system; characterize x*_(k)'s shape and the induced frame-spectrum
allocation. Two pre-registered outcomes:
- **T2-LOG:** under uniform coherence (μ), optimal allocation is
  x_(k) ≍ (m + c√log(1/p_k))² — logarithmic-squared, NOT a power law; the
  induced spectrum is curved-shallow; any measured power law then requires
  structured coherence (organization), data-correlation spectra, or dynamics.
- **T2-PL:** coherence structure μ(j,k) co-optimizing with usage admits a
  power-law allocation with exponent α(s, γ) — derive α or refute.

**T3 (hierarchy rigor).** For isotropic subgaussian directions, the t-body
interference terms concentrate with d^{-(t-1)/2} suppression (Hoeffding/
Wiener-chaos), yielding the tail constants in T1/T2 with explicit d, γ.

## 2. First reduction: uniform coherence (T2-LOG derivation sketch)

Set (v_j·v_k)² = μ for j ≠ k; σ_j² = μ(X − x_j), X = Σx. Misretrieval
ε_j ≈ Φ̄(t_j), t_j = (√x_j − m_j)/√(μ(X − x_j)). Lagrangian optimality on
Σx = P with objective Σp_j ε_j: the marginal condition equalizes
p_j·|∂ε_j/∂x_j| (reliability-allocation classic). With Φ̄(t) ≈ φ(t)/t:
∂ε_j/∂x_j ∝ ε_j·t_j/(2√x_j σ_j); the optimum satisfies
p_j ε_j t_j/(√x_j σ_j) = c ⇒ t_j² ≍ 2 log p_j + O(1) ⇒
√x_j ≍ m_j + t_j·√(μX) ⇒ **x_j ≍ (m + √(2μX·log(1/p_j)))²** — grows only as
log²(1/p_j) in usage rank. Consequences (to be stated as lemmas with proofs):
(i) induced norm profile is curved-log, its battery signature is
"curved-beats-power-law" — matching the XE interim (ΔR² < 0 everywhere);
(ii) NO exponent α(s, γ) exists in this specification; power-law spectra are
not explained by usage + uniform superposition;
(iii) the folk claim "superposition over Zipf features ⇒ power-law spectra"
is false in this (fully specified) model. What remains as power-law sources:
structured coherence co-optimized with usage (T2-PL), data-correlation
spectrum inheritance (H2 with gauge fixed — whitened/balanced observable,
full-shape comparison only), or optimization dynamics.

## 2b. Derivation progress notes (2026-09-19, pre-proof capture)

Two independent channels both yield LOG-type allocation; both must be
formalized as lemmas before T2-LOG is claimed:

**L1 (subgaussian-max channel).** Under uniform coherence, the binding
interference for retrieving pair (I, j) is the MAXIMUM competitor logit,
whose scale is √(μ X_max · 2 log n_eff) (n_eff = effective count of
near-max-norm competitors). Required: √x_j c_Ij ≥ √(μ X_max 2 log n_eff).
Consequences: (i) the margin constraint is (to leading order) norm-SCALE
free — it constrains geometry (c, μ) not budget; (ii) feasibility reads
2μ log n_eff ≤ c²_min — a coherence-capacity condition recovering the
interference-hierarchy bound in disguise (T3's job: make this exact with
chaos tails); (iii) where norms DO bind (alignment heterogeneity c_Ij),
x_j ∝ log(n_eff)/c²_Ij — logarithmic.

**L2 (CE logit-margin channel).** Softmax excess loss for pair (I, j) forces
z_j − z_k ≳ log(p_k/p_j)-type separations against every sizeable competitor
(continuous version of the union bound; exact statement via the classical
multinomial logistic margin bound). Under uniform coherence the separation
requirement enters norms as x_j ≳ (log usage-ratios)²/c² — again
logarithmic-squared in usage rank, consistent with the reliability-KKT
sketch in §2.

**Caveat to resolve (flagged, not hidden):** in the pure union-bound regime
the problem is nearly norm-indifferent (L1-i) — allocation is determined by
alignment heterogeneity and CE curvature, not raw norms. T2's formalization
must therefore fix WHICH loss variant defines the budget-constrained problem
(CE excess with explicit budget Σx = P and alignment profile c) before the
allocation law is stated. Suspicion to test: power-law allocation can only
enter through structured/heavy-tailed coherence (correlated interference) —
if L1+L2 are proved as stated, T2-LOG follows for ALL uniform-coherence
variants, and the folk "Zipf superposition ⇒ power law" claim is false
across the entire uniform family.

## 3. Planned falsification arm (design AFTER derivation, before running)

A margin-task simulation: hard-margin retrieval loss on Zipf tokens with
explicit coherence control (random vs organized directions), budget-matched;
pre-registered checks: (a) uniform-coherence arm reproduces T2-LOG shape
(log² profile; curved battery signature); (b) an organization arm (coherence
concentrated among high-usage tokens) is required for power-law-like
profiles. Numeric tolerances frozen at that point.

## 4. Provenance and status

Opened after review-2 (ledger `droid_review_2hr_1`) steering: capacity/
allocation theorems and interference rigor over slope matching. HP-C1
exponent-matching measurement stays PAUSED (gauge flaw; corpus asset kept).
HP-B grids continue in background (dynamics branch evidence). Nothing in
this document is claimed as proved until each lemma carries its proof and an
evidence-gate pass.
