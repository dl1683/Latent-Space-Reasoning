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

## 2c. CORRECTION AND REDONE DERIVATION (2026-09-19; review-3 caught a sign
## error in §2 — this section supersedes §2's monotonicity claim)

The §2 marginal condition p_j ε_j t_j/(√x_j σ_j) = c with ε_j ≈ φ(t_j)/t_j
gives, taking logs: t_j² = 2 log p_j + C(x, σ) — reliability margin grows
with USAGE, not rarity (§2's "x_j ≍ (m + √(2μX·log(1/p_j)))²" had the
monotonicity REVERSED; withdrawn). Consequences of the corrected condition:
- Frequent tokens: t_j² → C — bounded margins; x_j ≈ (m + √C σ)².
- Rare tokens: 2 log p_j + C < 0 — the tail linearization fails; the true
  optimum floors them at minimal functional norms (sacrificed in the
  usage-weighted sense; cf. minority collapse and the quadratic theorem's
  tail-dropping).
- Net: the optimal norm profile is FLAT UP TO LOG CORRECTIONS in this
  specification — no power law in either monotonicity. This coheres with
  all three toy results (sparse-exact optimum; XE normspread → 1.4; entry-95
  random-norm flatness). **T2-LOG is restated as T2-FLAT:** under uniform
  coherence and the reliability-KKT specification, budget-optimal allocation
  is flat-with-log-corrections; no power-law allocation is optimal.

## 2d. Repairs required before T2-FLAT is claimed (per review-3)

1. Complete the specification: one CONCRETE structured transition kernel —
   block model: vocabulary in B usage-ordered blocks; P(J|I) = within-block
   association ρ + unigram remainder; alignment a_Ij derived from it.
2. Derive the interference law FROM the decoder: misretrieval of (I, j) is
   the event ∃k∉S_I: √x_k (v_k·v_I) ≥ √x_j (v_j·v_I) − m; the competitor
   logit vector's joint subgaussian law (covariance from the coherence
   profile) replaces the hand-written sum in §0.
3. Prove allocation monotonicity (x* nondecreasing in p under the kernel).
4. **T4 (phase transition — the dimension-dependent invariant target):** for
   the block kernel, characterize the critical d*(n, ρ, s) below which
   usage-weighted retrieval error stays bounded — candidate scalings
   Θ(log n) (packing-like) vs Θ(√n) (coherent-like); a sharp threshold with
   explicit constants is the theorem this program wants.

## 5. The structured kernel and the T4 derivation (2026-09-19; repairs §2d)

**5.1 Kernel (concrete, planted block).** Usage p_i = i^{-s}/H_{n,s};
vocabulary split by rank into B equal blocks of size m = n/B. Transition:
P(j | i) = (1−ρ)·p_j + ρ·(1/m)·1[block(j) = block(i)]. ρ = structure
strength; ρ = 0 is the solved unigram case (H1). Target logits:
z*_j(i) = log P(j|i) + c_i.

**5.2 Decoder-derived logits (repair of §0's hand-written noise).** Tied
retrieval, z_j(i) = w_j·w_i exactly. Misretrieval of (i, j) is the event
∃k: z_k(i) ≥ z_j(i) − 0 (argmax) or, margin version, z_k(i) ≥ z_j(i) − m_ε.
ALL interference is competitor logits z_k(i) = w_k·w_i — no external noise
variable exists or is needed. Every statement below is about the joint law
of {w_k·w_i}_{k} under the code ensemble.

**5.3 Equilibrium ansatz (lemma to prove: optimality of decomposition).**
w_j = a_j·f + b_j·u_{b(j)}, with f ⊥ span{u_b}: the frequency direction f
carries the unigram part (a_j ∝ log-scale of p_j, per H1's mechanism) and
the B signature vectors u_b (unit norm) carry block membership. The task is
exactly representable iff d ≥ B+1; **the superposition regime is d < B.**

**5.4 Interference law (lemma).** For i in block b, out-of-block competitor
k in block b': z_k(i) = a_i a_k + b_i b_k (u_{b'}·u_b). Same-block target
advantage: Δ_ij = b_i b_j (1 − u_{b(j)}·u_{b(i)}) + (a-terms, favorable
since same-block targets need not beat the unigram part). With signature
coherence μ = max_{b≠b'} u_{b'}·u_b: b_i b_k μ is the worst structured
interference; the a-part is favorable (out-of-block frequent tokens have
lower required log-prob, rare ones lower a_k).

**5.5 Threshold route (T4a statement).** Usage-weighted error ≤ ε requires,
for typical (i,j same-block): b_i b_j(1 − μ) ≥ max_k [a_i a_k + b_i b_k μ]
+ σ·√(2 log(n/ε))-type subgaussian slack (exact constant from the chaos
analysis, T3). Two regimes fall out:
- **Welch regime:** B signatures in d dims force μ ≥ √((B−d)/(d(B−1)))
  (Welch bound; achievability caveats logged). Substituting gives the
  feasibility inequality in (d, B, ρ, s, ε, budget P) — solve for
  d*(B, ρ, s, ε, P). Conjectured form: d* linear-ish in B with a
  structure-strength factor — a genuinely dimension-dependent threshold.
- **Packing regime:** B > d(d+1)/2 — equiangular signatures unachievable;
  μ jumps; the transition sharpens (candidate: d* ≥ c·B^{1/2} vs linear —
  the phase boundary this theorem hunts).
Budget coupling: Σ_j b_j² enters through the slack term — the allocation
problem REDUX, now structured (b_j vs signature geometry jointly optimal),
which is where any non-flat allocation law must live if it exists at all.

**5.6 Proof obligations (open, ordered):** (i) prove 5.3's decomposition is
near-optimal for the CE objective (variational argument or counterexample);
(ii) prove 5.4's favorable-a-part claim exactly (it interacts with s);
(iii) chaos/Hoeffding tail for the max over n competitors (T3 machinery,
fixing normalization per review-3's point 2); (iv) solve the feasibility
inequality in both regimes; (v) pre-register the simulation that tests the
threshold before running it (planted-block retrieval at (n, B, d, ρ, s)
grid around the predicted boundary).

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

## 6. PAUSED for repair (2026-09-19, review-5)

Two defects must be repaired before T4 proceeds (no HP-D simulations):
1. **Observation model:** the kernel samples J|I but the decoder receives
   only q = v_I; sampled-target argmax error includes irreducible Bayes
   uncertainty (the conditional entropy of J|I), which is not interference.
   Repair: full-distribution CE objective — minimize CE(P(.|i) ||
   softmax(z(i))) — matching real training; error metric = CE excess over
   H(J|I).
2. **Representability:** tied logits W^T W are symmetric PSD; the target
   log-kernel is symmetrizable only in special cases, so "exactly
   representable iff d >= B+1" is WITHDRAWN pending proof. Repair: untied
   code (query matrix H, key matrix W; Z = W^T H general rank-d), with the
   tied/PSD variant stated as a separate constrained corollary; the gauge
   freedom noted by review-2 is then fixed by the CE optimum +
   regularization (the balancing principle), not assumed.
While paused, HP-E (theory/HPE_AGGREGATE_INTERFERENCE.md) carries the lead.

## 7. REPAIRED FORMALIZATION (2026-09-19; implements the review-5 repairs)

**Objective (full-distribution CE, matches real training).** Minimize over
untied (H, W) in R^{d x n} x R^{d x n}:
  L = -sum_i p_i sum_j P(j|i) log softmax_j(w_j . h_i).
Error metric: excess = L - H(P(.|I)) (conditional entropy floor). No sampled
target, no argmax, no external noise; all interference lives in the softmax
competition, decoder-derived by construction.

**Representability lemma (statement).** Zero-excess solutions exist iff some
row-constant shift of the log-kernel, log P - 1 c^T, has rank <= d (Z = W^T H
ranges over all rank-<=d matrices; softmax rows are shift-invariant).
Approximate form: excess CE <= g(tail energy of the centered log-kernel
beyond rank d), with g quadratic-ish via Pinsker-type bounds (to be pinned).

**Planted-block log-kernel is approximately rank-(B+2).** log[(1-rho)p_j +
rho/m 1_same] = logsumexp(log(1-rho)+log p_j, log rho - log m + M_ij) where
M = block indicator. When one argument dominates, the logsumexp correction is
additive-bounded; so log P = A_j + B-coeff*M_ij + E with ||E|| bounded by the
mixture-transition sharpness; A (unigram) contributes rank 1-2, M
(block-diagonal ones) contributes rank B. The representability defect at
d < B is governed by the weighted low-rank compression of M.

**Block-mass dichotomy (T4 in objective terms).** M has B equal singular
values (sqrt(m) each), but USAGE block-masses decay by Zipf rank (block b
carries mass W_b = sum of p over its ranks). The weighted rank-d compression
of the kernel faces exactly the dropping-vs-superposing dichotomy of the
quadratic theorem, now over blocks:
- when block masses decay steeply (large s), keep-top-d-blocks + drop tail is
  near-optimal (sparse-exact redux);
- when masses are comparable, superposition (soft mixing of the remaining
  B-d block directions) is forced, and the EXCESS CE of compression becomes
  the retrieval-phase quantity: T4 boundary = where compression excess
  crosses the entropy floor as a function of (B/d, rho, s, n).
Pre-registered T4' target: for fixed (rho, s), the excess-CE phase boundary
in (B, d) is sharp (concentration over the kernel ensemble); concrete
conjecture: critical d*(B) proportional to the participation ratio of the
block-mass sequence (bridging to the E-struct capacity law).
