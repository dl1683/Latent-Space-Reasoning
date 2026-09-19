# STATE

## CURRENT DIRECTION: Mathematics of High-Dimensional Spaces

**Status: ACTIVE (user ruling, 2026-09-18).** A mid-file "HD MATH DIRECTION:
CLOSED" verdict was a coding subagent overstepping its review mandate and is
void; the corrected block appears below. Adversarial reviews (headless
`droid exec`, every 2–3 hours during research sessions) are goal-aligned
adversaries: they attack the method and strategy to ensure the work is the
best path to the goal, diagnose what is going wrong, and prevent
overclaiming — assuming the axiom is true; they never close a
direction or void the axiom. On a failed cycle the required question is "this
is not working — what would work for exploring this problem statement?"

How does math change at high dimensions? This is the core research question.

---

## ACTIVE CYCLE HP: spectral law of Zipf-trained superposition codes (2026-09-18)

Spec (pre-registered, frozen): theory/HD_POWER_LAW_ALLOCATION.md (v2 after a
Droid design gate returned NO-GO on v1 with five fixes; both verdict and
redesign are in the ledger). Established so far:

**Theorem (licensed).** The quadratic superposition autoencoder
L = Σ_i p_i ‖W^T W e_i − e_i‖² with d < n has global optimum = orthonormal
exact storage of the top-d usage tokens, minimum loss Σ_{i>d} p_i (tail
mass), optimal frame operator I_d (perfectly flat spectrum). Proof:
L = tr((G−I)P(G−I)) with G PSD rank ≤ d; on ker G the factor acts as −I.
Empirically matched by L-BFGS multi-init optima to 6 decimals at 6/6 configs.
Mean-field (random-mixing) stationarity is REFUTED at the optimum
(profile R² 0.00–0.08); the optimum anti-mixes.

**Licensed consequence.** Power-law eigenspectra cannot be global optima of
this diagonal quadratic objective; dynamics or assumptions absent from that
objective (correlations, constraints, regularization, retrieval margins)
are required.

**XE result (3 seeds, licensed 2026-09-19):** unigram tied-softmax training
converges toward the H1 rank-2 collapse — top-2 eigenvalue share ≥ 0.984 at
every checkpoint from t=3k, norm uniformization (spread 1.9 → 1.4), curved
fits beat power law at 12/12 checkpoint-seeds (r²_pl only 0.54–0.71), and
measured α ≫ the norms-only control (N1-t ≈ 0.22) — the fitted slopes are
directional structure (collapse coherence), not norm allocation. Entry-101's
"Zipf + cross-entropy → power-law eigenspectra" is an estimator artifact on
a two-spike-plus-floor spectrum. Scoped statement: in the specified
diagonal QSAE optimum and the three tested unigram tied-softmax runs, usage
alone did not yield genuine power-law spectra — this does not cover
structured tasks or other objectives; both toy tasks are underconstrained
(quadratic permits tail-dropping; unigram retrieval has no input-dependent
requirement), a generalization boundary stated per review-4. HP-D
(theory/HPD_ALLOCATION_THEOREM.md) targets the structured-retrieval
phase-transition theorem T4 (critical d*(n, rho, s)); its earlier T2-LOG
conjecture was withdrawn for a sign error (review-3) and replaced by the
unproved T2-FLAT restatement. HP-B is CLOSED (2026-09-20): XE complete (rank-2
collapse, 3/3 seeds), QSAE 12/12 runs converged with flat endpoints (P-E
confirmed on every run); the heavy confirmatory arms were foregone with
review-backed provenance (reviews 3-4 demoted them; ~7h/run economics).

**HP-E results (2026-09-19, derived + numerically validated):**
1. Quadratic aggregate law: E[A_2] = k(k-1)||M||_F^2/d^2 proved and validated
   (P4 crossing at k = sqrt(d) confirmed; concentration confirmed). Scope:
   quadratic/projection readouts; the degree>=3 extension was REFUTED in its
   naive form (per-term law is ||T||^2/d^t, and no natural Theta(1) diagonal
   channel exists at degree>=3) — open with explicit obstruction.
2. E-struct (expectation level, validated; RESCOPED per review-6): for
   unit-energy features with covariance Sigma, per-pair interference =
   tr(Sigma^2)/(tr Sigma)^2 exactly; expected aggregate interference becomes
   order one at k = Theta(sqrt(PR(Sigma))) in the specified expectation-
   normalized quadratic model. A matching high-probability frontier is proved
   ONLY in the isotropic case; anisotropic concentration at the crossing is
   open. Power-law corollary restricted to 1/2 < alpha < 1: crossing at
   Theta(d^{1-alpha}); at alpha = 1 the scale is Theta(log d); alpha > 1
   Theta(1).
   Validated: per-pair exact to 0.4-2.4% at alpha <= 0.5; crossings bracketed
   in 8/8 cells (two registered formal fails carry diagnosed benign causes,
   logged). Pending rigor: concentration formalization and the matching
   lower bound with quantifier order. This is the lawful spectrum->capacity
   bridge the program sought (no fitted slopes involved).

**HP-E covariance-sufficiency result (2026-09-19, review-6 lead unit,
pre-registered):** at identical covariance, Gaussian-direction and
coordinate-sparse feature ensembles share the same expected aggregate
interference (to a few %) but the sparse ensemble's concentration collapses
(cv 21x, spread 14x at the crossing regime). The participation ratio is an
EXPECTATION-LEVEL invariant only; the high-probability frontier requires a
fourth-moment/localization functional of the direction distribution — the
 Next invariant target (candidate: coordinate-kurtosis law for cv(A_2)).

**HP-E localization law (2026-09-19, closed form):** the sharpness of the
frontier is controlled by two computable functionals — cv²(A₂) ≈
2(Ψ₁−1)/k² + 2(Ψ₂−1)/k with Ψ₁ = d²·trΣ⁴/(trΣ²)² and Ψ₂ = d·trΣ⁴/(trΣ²)²
for coordinate-sparse directions (exact at α=0 in both terms, both matching
measurement; 1.7× quantified residual at the worst inhomogeneous cell; no
free constants). Gaussian directions collapse both terms to the 2/k law.
Net program statement: the spectrum (participation ratio) sets the mean
interference frontier; for coordinate-sparse directions the sharpness is
governed by the two functionals of 3l (exact at the homogeneous case, 1.7x
residual elsewhere) — covariance alone does not suffice (settled by the
two-ensemble test).

**HP-D Orlicz-compression arc (2026-09-19/20, verified chain):** for softmax
rank-d compression, KL(q||softmax(z)) = log E_q[e^delta] - E_q[delta] — the
cumulant generating function of the logit residual (EXACT identity).
Consequences, all measured at settled optima: (1) the Fisher law is the
kappa_2 regime and holds where deviations are small (validated at s=0.5
far-below-knee, points within 1.2x); (2) at the tested finite configuration, the best reproducible stationary
solutions found accept ~7x the Fisher-optimal residual variance and exploit
higher-cumulant structure (12 seeds within 0.0005; gap 6.2x/4.7x); whether
these are global optima is open; (3) the variance-KL envelope is
DEGENERATE (inf KL = 0 at any fixed variance via rare huge negative jumps)
— polynomial moments do not govern KL compression; the one-sided
exponential moment does (Orlicz scale); (4) the settled optimum satisfies
the rank-d first-order condition EXACTLY (gradient field entirely in the
complement block — Eckart-Young-type orthogonality verified at 1.0000);
(5) the attempted SVD-based dual is unbounded without a norm constraint and
the tested regularized dual is vacuous; no useful global dual certificate is
currently known (simplex-native or regularized routes untested beyond these). T4 completion target stands as the tilted-SVD/two-fixed-point
program with verified first-order core.

**HP-E limit-law arc (2026-09-20, exact + validated):** for localized
directions the aggregate interference A_2 is a weighted birthday sum; its
EXACT fixed-k multinomial covariance (derived; validated 12/12 zero-parameter,
ratios 0.76–1.09) supersedes the refuted Poissonized model — the negative
cross-covariance −(k)₂(4k−6)p⁴ is load-bearing. Law class governed by
λ²·PR_W (density² × weight participation ratio): Gaussian concentration iff
it diverges; persistent O(1) fluctuations when O(1); boundary computable in
closed form. Poissonization itself was refuted 2/12 by a zero-parameter
pre-registered test — the cycle (prediction → refutation → exact derivation
→ validation) is the arc's method signature.

---

The scope is broad and fundamental: the native mathematics of spaces where
d >> 1. This includes but is not limited to neural network representations.
Traditional ML, deep learning, pure mathematics — any domain where
high-dimensional structure matters. The dimension range is open (d=100 to
arbitrarily large). LLM latent spaces (d=768-4096+) are one motivating
example, but the math applies to any high-dimensional system: kernel methods,
embedding spaces, random feature maps, high-dimensional statistics, and the
pure geometry/algebra/analysis of R^d when d >> 1.

**Axiom:** Math behaves differently in high dimensions. The phenomena that
emerge — concentration of measure, almost-orthogonal packing, interference,
volume collapse, distance uniformity — are not curiosities but the native
physics of these spaces. Understanding them is the goal.

**Prior work in this repo:** 86 experiments characterizing perturbation
sensitivity in small language models (one source of high-dimensional evidence),
omega modulus measurements, structured negatives about behavioral instruments.
This data is available for reanalysis within the broader frame.

---

## HISTORICAL: Perturbation Sensitivity Analysis (86 entries, ongoing data source)

86 NOTEBOOK entries analyzing how code perturbations affect small language model
behavior. Position >> operator >> sign >> identity >> meaning hierarchy established
across 3+ architecture families. C-group two-tier split (comment vs code prefixes).
Construct-validity question open (Droid review: is hierarchy probe-independent?).

---

## HISTORICAL (within HD-MATH): Theory-First Cycles — Closed (2026-09-18)

Three theory-first cycles attempted to connect HD math to behavioral data, all
killed. These are structured negatives about using behavioral instruments to
study dimension — not negatives about HD math itself.

**Open questions (initial):**
1. What algebraic structure, if any, do almost-orthogonal packings in d=512-4096 have?
2. Can interference between superposed features be predicted from geometry alone?
3. Does concentration of measure explain granularity-dependent transfer?
4. What are the native operations of these spaces — not R^n operations applied to them?
5. What does calculus look like when d >> 1 and the manifold hypothesis may not hold?

**Known landscape (from literature review, 2026-09-18):**
- Almost-orthogonal packing: ~exp(d) directions, no algebraic structure, d=512-4096
  poorly characterized (Luisto 2025 survey, arXiv:2510.23609)
- Computation in superposition: n neurons compute O(n²/log n) features, exponential
  gap over representation (Hänni et al., arXiv:2408.05451, 2409.15318)
- Spectral superposition: frame operator classifies toy-model geometry, not real
  networks (arXiv:2602.02224)
- Adversarial interference: geometry predicts vulnerability, transfers across models
  (arXiv:2510.11709). No composition rules.
- Platonic Representation Hypothesis: coarse convergence known (Huh et al., ICML 2024).
  Fine divergence known ("Convergence Without Understanding", arXiv:2605.23315).
- Formal superposition theory (arXiv:2608.27540, Aug 2026): frame theory + compressed
  sensing. Recovery for nearly tight, low-coherence dictionaries up to O(d/log n)
  sparsity. ETF recovery thresholds computed exactly. Active debate on computation
  in superposition vs compressed computation (Jun-Jul 2026).

**NEW EMPIRICAL FINDINGS (2026-09-18, entries 88-92):**

Frame-theoretic analysis of 3 trained models (SmolLM2-135M d=576, Qwen3-0.6B
d=1024, Phi-1.5 d=2048):

1. **Real models are 100x-10,000x from tight frame ideal.** Embedding condition
   numbers: 640 (Phi) to 11,971 (SmolLM2). MLP conditions: 180-25,888. The
   formal superposition theory's assumptions (nearly tight, low coherence)
   are massively violated.

2. **Coherence is 15-30x the Welch bound.** Embeddings: 0.75-1.00 vs Welch
   0.02-0.04. Features are far more aligned than optimal packing predicts.

3. **Eigenspectra follow power laws**, not Marchenko-Pastur (random) or uniform
   (tight frame). Exponents: -0.35 to -0.90 (full), steepening through depth.
   Training creates regularity neither random matrix theory nor toy superposition
   theory predicts.

4. **Embedding condition decreases with d** (11,971 → 1,175 → 640 as d goes
   576 → 1024 → 2048). Higher-d models have more uniform embeddings.

5. **Gate/up asymmetry is universal.** Gate_proj has higher condition, lower
   effective rank, higher coherence than up_proj. Sharper feature selection
   vs broader value computation.

6. **Amplification gradient, not tail vulnerability (entries 92, 118).**
   Entry 92 found natural perturbations with more tail eigenspace energy
   cause more damage (rho = -1.00 at L29). OVERTURNED by entry 118:
   controlled fixed-magnitude perturbations into individual eigenspaces
   show TOP eigenspaces are 3-14x MORE sensitive (beta = +0.25 to +0.41).
   Entry 92's correlation was confounded — the model's multi-layer
   processing absorbs top-eigenspace perturbations, leaving tail residuals.
   Controlled injection establishes greater LOCAL sensitivity in top
   eigenspaces (amplification gradient); downstream compensation through
   subsequent layers is one untested explanation for the natural-perturbation
   reversal, not an established mechanism.

7. **Information-theoretic eigenspace priority (entry 94; wording corrected
   2026-09-19).** In the measured embedding, selected rare morphological
   tokens (strugg-, authent-) had greater top-eigenspace energy than selected
   common tokens (observation on selected tokens only). The governing
   variable and any optimality are unestablished.

8. **Power-law spectra require directional structure, not just norm scaling
   (entry 95).** Monte Carlo test: random embedding directions with
   information-weighted norms produce FLAT spectra (exponent ~-0.1, condition
   ~1.5) indistinguishable from uniform random embeddings. The power law
   therefore comes from training organizing tokens into correlated directions.
   Directional organization alpha = (obs - random) / (greedy - random):
   SmolLM2 alpha=0.50, Qwen3 alpha=0.32. Alpha decreases with d — higher-d
   models need less directional organization. The Zipf sensitivity analysis
   suggests effective training Zipf exponent ~0.5 (not 1.0).

9. **First-principles: Zipf + cross-entropy → power law (entry 101).** A
   single-layer embedding trained with SGD on Zipfian tokens produces power-law
   eigenspectra. Uniform tokens → flat spectrum. The power law is a consequence
   of training on Zipfian data, not an architectural feature. Exponent is
   controlled by (Zipf s, n/d ratio): higher overcompleteness → flatter.
   Phi-1.5 (n/d=25) steeper than Qwen3 (n/d=148) explained. Alpha(d) monotone
   prediction FALSIFIED; alpha depends on n/d not d alone.

10. **Hierarchical eigenspace clustering (entry 99).** Tokens are concentrated
    into few eigenspaces: SmolLM2 median erank = 32/576 (5.6% of d), Qwen3
    median erank = 326/1024 (31.8%). Concentration ratio inversely tracks
    power-law steepness. Most concentrated tokens (erank ~3) are rare
    vocabulary entries aligned with eigenvector 0 at 77-83%.

**CORRECTED (2026-09-19, adversarial review; supersedes "RESOLVED"):** The
theory-practice gap between formal superposition theory (tight frames) and
real models (power-law-like fitted slopes) is NOT resolved. Finite-time
Zipf-trained models produced power-law-like fitted slopes, but the estimator
can manufacture such slopes (HP-A A2), and unigram-softmax equilibrium
evidence favors non-power-law structure (rank-2 collapse, HP-B XE interim).
The structured-data explanation remains open; no mechanism is licensed.
Retained context: fitted slopes co-vary with Zipf s and n/d (entry 101);
directional organization exists relative to random baselines (entry 95);
top-eigenspace allocation tracks token information (entry 94).

**CLOSED PROGRAM: Frame Composition and Mixed Moments (2026-09-18)**

Strategic direction from Droid review (entry 96): develop a predictive calculus
for composition and interference in structured HD systems.

**RESULTS (entries 97-98), narrowed per Droid review (entry 102):**

1. Gate and up input-side Gram operators exhibit positive centered Frobenius
   alignment at all layers (entry 98). D1_norm: SmolLM2-135M mean 0.22,
   Qwen3-0.6B mean 0.22. Sharing increases coarsely through depth zones.
   This is a real descriptive finding but not yet predictive or surprising.

2. Synthetic test (entry 97): D1 separates shared-eigenvector from random
   frames. Valid but elementary — that relative eigenbasis affects matrix
   sums is established linear algebra. The "pairwise insufficiency" framing
   overclaims; it shows only that Gram inner product detects eigenvector
   alignment, which is expected.

3. D1_norm ≈ 0.21 across SmolLM2 scale ladder (entry 100). Three checkpoints,
   but width, depth, and training are confounded. Not yet evidence of a
   universal budget.

**WITHDRAWN (per Droid review, entry 102):**
- Formal theorem (theory/mixed_moment_critical_exponent.md): multiple proof
  errors (log divergence at α=1/2, O(1) contribution below α=1/2, H(s,δ)
  divergence for s>1, measured exponents 1.1-1.8 outside α<1 derivation).
  Needs correct rederivation.
- "Effective sharing fraction" (2-15%): divides by unmatched benchmark.
- "Pairwise geometry is provably insufficient": overclaimed framing.
- D_k sufficiency for SwiGLU composition: KILLED by row-permutation argument.
  Permuting up_proj rows preserves all D_k, all spectra, but changes which
  gate/up neurons are multiplied → network function changes. D_k discards
  the row-pairing structure essential for SwiGLU.
- "Universal coupling budget" at 0.21: premature from 3 confounded checkpoints.

**CRITICAL INSIGHT (Droid):** The row-permutation argument shows that any
invariant based solely on W^T W (Gram operators) is fundamentally blind to
row pairing. SwiGLU computes σ(Wx) ⊙ (Vx), so the i-th row of W gates
the i-th row of V. A useful composition invariant must be row-aware.

**Row-permutation challenge DONE (entry 103):** Random permutation of up_proj
rows preserves all D_k (diff < 5e-08) but destroys MLP output (139% relative
error, range 1.04-1.64 across layers). Function-preserving control (same
permutation to gate+up, inverse to down cols) has zero error (2.5e-07).
Separation ratio: 5.5 million to 1. D_k is conclusively insufficient.

**Row-wise cosines are zero (entry 104):** The simplest row-aware statistic
— cos(gate_i, up_i) — is indistinguishable from random (mean -0.0009, frac
positive 0.499). In d=576, all pairwise cosines concentrate at zero. The row
pairing is essential but carries no per-pair directional signature.

**The pairing paradox:** Gram operators show population alignment (D1_norm=0.22).
Individual row pairs show no alignment (cos ≈ 0). Yet the specific combinatorial
pairing determines the function (139% error from mismatch). The compositional
information exists at no single natural level of analysis.

**Row-permutation series (entries 103-106, corrected by entry 107)**

Clean result: D_k sufficiency is killed (entry 103). Any invariant factoring
only through G^T G and U^T U cannot characterize a SwiGLU block. The 139%
error ≈ sqrt(2) (random decorrelation), mostly from DU disruption (additive
error 0.88). Entry 105 has a mathematical error (m×m cross-matrix SVD is
exactly permutation-invariant by construction). Entries 104-106 overclaimed;
the correct framing is hidden-unit gauge symmetry, not "tensor phenomenon."

**CLOSED: Tensor composition line (entries 97-109)**

Started: D_k mixed moments detect shared eigenvector structure (entries 97-98).
Killed: D_k sufficiency killed by row-permutation argument (entry 102-103).
Investigated: Row cosines, cross-matrix SVD, bilinear sensitivity (entries 104-106).
Overclaim corrected: Gauge symmetry, not tensor phenomenon (entry 107).
Final test: Weight-only tensor predictor ||ΔT||_F ≈ DU predictor ||DΔU||_F
(entry 109). Gate structure adds nothing for damage prediction. Line closed.

**What survives from this program:**
1. D_k sufficiency is cleanly killed (entry 103) — valid structured negative
2. Power-law-LIKE FITTED SLOPES from Zipfian training (entry 101) — status
   downgraded 2026-09-19: estimator-manufacture and transient/equlibrium
   structure both implicated (HP-A A2, HP-B XE interim); origin open
3. Information-theoretic eigenspace priority (entry 94) — positive finding
4. Gate/up Gram alignment exists descriptively (D1_norm ≈ 0.22) — but not compositionally meaningful
5. Permutation damage is predicted by ||DΔU|| — the down-up product, not gate-up coupling
6. Spectral-composition bridge REJECTED (entry 110) — spectral structure and
   compositional pairing are INDEPENDENT; training does not jointly organize them
7. Additive anti-pairing (entry 110) — additive MLPs develop z=-7.79 spectral
   anti-pairing (gate and up project with opposite signs onto top eigenvectors);
   gating DISRUPTS this, freeing pairing structure from spectral constraints.
   This explains WHY gated MLPs are pairing-sensitive (entry 106).
8. Output spectral pairing dipole (entry 111) — CONFIRMED CROSS-ARCHITECTURE.
   Near-output layers develop a sign-structured pairing pattern. SmolLM2 L28:
   top5_z=+2.48, 137/576 positive-significant. Qwen3 L27: top5_z=+1.80,
   240/1024 positive-significant. Both ~25-28% sig fraction. Preceding layers
   show anti-pairing (Qwen3 L26: z=-2.35, 122 negative). "Spectral dipole"
   at the output. Middle layers show z≈0 (aggregate p>0.29 in both models).

**ACTIVE DIRECTION: General Mathematics of High-Dimensional Spaces (2026-09-18)**

Pivot from model-specific experiments to GENERAL HD math. Scope: pure mathematics,
statistical physics, information theory, random matrix theory, kernel methods —
any domain where d >> 1. Models are ONE source of evidence, not the primary object.
Analytical derivations and proofs take priority over running code on specific models.

**Analytical results derived (2026-09-18):**

1. **Interference hierarchy (rescoped 2026-09-19 per adversarial review):**
   For isotropic random unit features and a suitably normalized fixed
   quadratic form, an individual off-diagonal bilinear term has RMS O(d^{-1/2})
   (shown). The general t-body and — crucially — the AGGREGATE k(d)-body laws
   (combinatorial multiplicity vs per-term suppression) remain conjectural and
   are the HP-E theorem target: candidate threshold k* = Theta(d^{(p-1)/p}).

2. **Packing vs recoverable capacity (corrected 2026-09-19 per adversarial
   review):** Packing permits exponentially many candidate directions.
   Recoverable simultaneous capacity depends on the decoder, task and error
   model; no universal linear or bilinear capacity law is established
   (orthonormal features permit a bilinear layer to compute k ≤ d diagonal
   quantities exactly). Any capacity theorem must specify ensemble, task,
   decoder, noise, error probability and k(d) before a threshold claim.

3. **Scalar allocation benchmark (corrected 2026-09-19):** The scalar
   resource-allocation problem (minimize σ² Σ p_i/λ_i s.t. Σλ_i = P) has
   optimizer λ_i ∝ √p_i. Its interpretation as an eigenspectrum law or a
   d → ∞ limit is unproved and, per the HP theorem, cannot describe quadratic-
   loss equilibria; retained only as a labeled diagnostic.

4. **Gauge symmetry of eigenspectra:** Embedding eigenspectra are GL(d) gauge-
   dependent. The training-chosen gauge has physical meaning through amplification
   (top = amplified more, per entry 118), but the spectrum itself is not a
   fundamental representation property. Entry 92's "tail vulnerability" was
   overturned — see finding 6 above.

5. **Capacity bounds (corrected 2026-09-19 per adversarial review):** Particular
   random-feature and error models may yield √d-type thresholds; no general
   coherent-case or depth-capacity law is established. Any threshold claim
   must fix ensemble, task, decoder, noise and error criterion first.

**Connections to established mathematics (to formalize):**
- Hoeffding decomposition (statistics): ANOVA-like variance decomposition on product spaces
- Wiener-Itô chaos expansion (probability): polynomial chaos in Gaussian Hilbert space
- Cluster expansion (statistical physics): systematic corrections to mean field
- 1/N expansion (gauge theory): planar diagram dominance in large-N limit
- All share the same structure: ordered correction hierarchy suppressed by a large parameter

**Open mathematical threads:**
1. Rigorous proof of interference hierarchy for general degree-p polynomials
2. Spectral calculus: closed-form evolution equations for eigenvalue distributions
   under gradient descent / stochastic dynamics in HD
3. Phase transitions in superposition capacity (sharp thresholds in k vs d)
4. HD calculus: gradient concentration, Hessian spectra, optimization landscape
5. Connection between interference hierarchy and neural network depth requirements
6. What makes structured (trained) features different from random features in the
   interference hierarchy — how does training break the incoherence assumption?

**Prior model-specific findings (available for reanalysis under new framework):**

**Open targets from prior program (updated by entry 118):**
1. Power-law spectra as computational structure (entries 92-95, 101, 118).
   Entry 118 overturns entry 92's tail vulnerability: controlled perturbations
   show TOP eigenspaces are more sensitive (beta=+0.3) — an amplification
   gradient. Whether later layers compensate (the natural-perturbation
   reversal) is untested; fitted power-law exponents themselves now carry
   estimator-bias caveats (HP-A A2) and their origin is open (HP cycle).
2. The additive anti-pairing discovery (entry 110) may have implications for
   understanding why gated MLPs outperform additive MLPs — the gating mechanism
   decouples pairing from spectral structure, enabling richer computation.
3. Depth-dependent pairing organization (entry 111) — does the penultimate-layer
   positive pairing generalize across model families?
4. Spectral steepening: DESCRIPTIVE, NOT PREDICTIVE (entries 112-115).
   Up_proj steepening confirmed across 3 models (rho < -0.65). Scale-invariant
   rate (~-0.15 normalized). Complementary specialization survives LN confound
   (post-LN rho=+0.45). BUT spectral exponent does NOT predict layer
   sensitivity (entry 115 NEGATIVE). Sensitivity is U-shaped (position-driven,
   not spectrum-driven). The depth gradient characterizes training dynamics,
   not functional importance.

**Finding 10 (entry 113): Complementary specialization — weights and representations anti-correlate.**
Hidden state spectra FLATTEN with depth (rho=+0.871, p<0.001) while weight
spectra STEEPEN (rho=-0.891) — oppositely moving fitted slopes in one model
and sampled corpus, after normalization. Causal relationship and functional
significance (accumulation, compensation, efficiency) are unestablished.

**Finding 9 (entry 112): Spectral depth gradient — CONFIRMED CROSS-ARCHITECTURE.**
MLP up_proj power-law exponent steepens monotonically with depth: SmolLM2 rho=-0.891,
Qwen3 rho=-0.649 (both p<0.001). Gate follows. Attention QK/OV ranks show no
depth trend. Creates a generalization-to-specialization gradient: early layers use
many spectral directions (flat spectra), late layers concentrate on few (steep spectra).
The pairing dipole (entry 111) sits at the extreme end of this gradient.

**Data from prior programs (available for reanalysis):**
- 7-model balanced kill: 69 cells × 24 perturbations, full-logit cosine distances
- 2×2 D-reliability diagnostic: within-op +0.632 >> cross-op +0.366 (Qwen3-0.6B)
- Omega modulus: ~28,000 readouts, 5 architectures, continuation-dependent sensitivity
- Scale ladder: SmolLM2 135M/360M/1.7B, Qwen3 0.6B/1.7B, base and instruct variants

**Empirical findings (2026-09-18, geometry probes, SmolLM2-1.7B d=2048):**

Finding 1: For diverse inputs (arith+NL+code), representation geometry shows
MONOTONIC CONCENTRATION through layers — no hunchback. eff_dim stays 1-3.

Finding 2: The "spread-then-concentrate" hunchback appears only when the
extraction-position token is IDENTICAL across inputs (context propagation artifact,
not a computational signature). Probe 1 shared "# Output: " suffix → identical
last tokens → apparent hunchback. Probes 2-3 with varying last tokens → no hunchback.

Finding 3: Cross-class cosine at mid-layers converges (arith-NL cos≈0.55 at L12-L20),
then the FINAL LAYER re-separates domains. Arith-code stay close (0.70), NL splits
from both (0.35-0.37). The final-layer cross-class cosine reflects next-token
vocabulary overlap.

Finding 4: Within-class final-layer cosine: arithmetic (0.96) > code (0.82) > NL (0.81).
Input homogeneity maps directly to representation homogeneity.

**Fable review (2026-09-18):** All geometric patterns (ID hunchback, trajectory
convergence, anisotropy) have extensive prior art (5+ papers 2024-2026). The
"interference tensor" concept has prior art. Category error identified: input-to-input
cosine ≠ feature superposition geometry. Actionable paths: fix methodology, engage
published frontier (Representational Capacity 2606.02765, Geometric Wall 2605.09887),
or build predictive models.

**RMT / Composition analysis (2026-09-18, KILLED):**

MP-deviation analysis: eigenvalue spikes above Tracy-Widom exist in all 7 models
(1-3 spikes each) but add NO predictive power over standard PCA for behavioral
transfer prediction. Kill criterion #2 triggered.

Interference composition: product rule r(i,k) ~ r(i,j)*r(j,k) shows positive
residual (+0.14 mean, all 7 models), but this is entirely explained by 3 shared
vulnerability factors (PCA). After factoring, residual flips negative.

**Fundamental constraint (2026-09-18):** The behavioral transfer data (69×24) is
intrinsically rank-3. Three architecture-family-specific factors capture 79-97% of
variance. No high-d math can extract structure that doesn't exist in the data.
The dataset is exhausted for high-d analysis. High-d investigation requires
genuinely high-d data (hidden states d=768-2048).

**Theory-first attempts (post Fable R3, 2026-09-18):**

Three theory-first cycles attempted, all killed:
1. Concentration-of-measure bound on behavioral rank → routing prediction killed
   (base vs instruct have identical attention routing despite 2× behavioral PR)
2. Weight spectral bridge → no correlation (all p>0.25; base/instruct have
   identical weight spectra but 98% different behavioral PR)
3. Jacobian derivation → Fable R4 KILL. Value/routing decomposition is backwards,
   perturbations not in Jacobian domain, and the one sharp prediction (d_add_zero
   should be A-like) is falsified in 6/7 models. Theory = QK/OV split (2020) +
   adversarial examples Jacobian (2015). See scratchpad/fable_jacobian_r4.md.

**d_add_zero dissociation (from R4 review, zero-cost):** A perturbation that
appends "+ 0" (touching zero original expression positions) does NOT behave like
an inert suffix. rho(add_zero, A) ranges 0.06-0.86 vs AA baseline 0.64-0.94.
Position of the edit does not determine behavior; whether appended tokens enter
the computation the readout must perform DOES.

**Fable R3 criterion #1 triggered (closure clause VOID per user ruling
2026-09-18):** 3 cycles, 0 surviving testable predictions. Next attempt must
state a numeric/ordinal prediction absent from existing data BEFORE examining
data. The "or the HD math direction closes entirely" consequence is void:
reviewers flag overclaims and approach problems; they do not close the mission.

**Computational-relevance test REJECTED (Fable design review, 2026-09-18):**
Pre-registered "syntactic context determines behavioral sensitivity" with 6 new
perturbation types. Fable REJECT: predictions already confirmed by R29 inline-inert
data and R31 `//`-disambiguation (both in NOTEBOOK), threshold fitted to calibration
data, confirmatory outcome is tautological ("model knows Python"), disconfirmatory
outcome has 3 untested alternatives the design cannot separate. Not counted as an
HD math attempt (behavioral sensitivity, not HD math, by its own admission).

**HD MATH DIRECTION: ACTIVE (user ruling, 2026-09-18).** The earlier "CLOSED"
verdict that stood here was a coding subagent overstepping its review mandate,
not a scientific result. Three theory-first cycles failing to produce surviving
predictions is evidence about those cycles, not license to retire the mission.
The user direction stands: explore the mathematics of high-dimensional spaces,
with LLM latent spaces as one later instance (they are too expensive on current
hardware and have too many moving parts; the general HD foundation comes first).
What the killed cycles DO establish, retained as method guidance for the next
attempt: (1) the rank-3 behavioral transfer matrices are exhausted — do not
re-mine them; (2) the open route is analytical/proof-first work on genuinely
high-dimensional objects (weight matrices, hidden-state geometry, random-matrix
and kernel regimes); (3) any empirical claim states a numeric/ordinal prediction
absent from existing data BEFORE examining data. On a failed cycle the required
question is "this is not working — what would work for exploring this problem
statement?", never closure. The behavioral sensitivity program's surviving
yield (multiplicative model, 4-level transfer hierarchy, d_add_zero
dissociation, R31 disambiguation, omega amplification) remains retained below.

**What survived (total yield of the kill experiment program):**
- Multiplicative sensitivity model d[i,j] ≈ f[i] × m[j], R²=0.82-0.88
- 4-level transfer hierarchy (operator type, difficulty, magnitude, cell×pert)
- Behavioral rank k=1-6 varying by model; instruction tuning quadruples rank
- d_add_zero dissociation: computational relevance, not position, determines sensitivity
- R31 `//` disambiguation: 10x effect from surface-dependent parse (a genuine finding)
- Omega modulus: cross-architecture amplification phenomenon (mechanism unresolved)

---

## HISTORICAL: OMT-0: Omega Modulus Transfer Test — structured positive (2026-09-17)

The omega modulus (modulus of continuity under behavioral distance) is a measurable,
cross-architecture phenomenon. ~24,000+ readouts across 55+ experiments,
5 architectures. Mechanism unresolved; prospective predictive value under test.

**Cross-architecture confirmation.** Three architecture families show amplification:
Qwen3 (0.6B/1.7B/4B), StableLM-2-1.6B, and Gemma-3-1b. Pythia-1.4B shows zero
(insufficient code training). The amplification pattern is model-dependent:
Qwen3 ranks identity highest; StableLM ranks neutral highest; Gemma ranks pass
highest. The universal finding: code continuations amplify behavioral distance
relative to no-op baselines, across architectures.

**Scaling.** Identity amplification scales monotonically within Qwen3:
0.003 (0.6B) -> 0.014 (1.7B) -> 0.051 (4B).

**Generalization axes confirmed:** cross-variable (83-92%), cross-pair leave-one-out
(+27.5pp at tau=0.01), cross-query format (100% at tau=0.02), cross-code-structure
(60-80% at tau=0.05), perturbation diversity (14/14 novel types amplify),
rank-based cross-pair (Spearman rho=0.52, predictor 71-86%).

**Held-out prediction.** Omega trained on standard continuations predicts novel
perturbation types at 83.3% (tau=0.05), +33pp over naive baselines.
Statistical significance: Cohen's d=3.49, permutation p=0.0004 for core effect.

**FRONTIER: Observed step (exp #31).** Omega shows a sharp jump at the first
identity statement: 0.003 (no identity) -> 0.047 (one identity) within the tested
suffix conditions. Non-monotonic with repetitions (rep_2 < rep_1, interference
effect). Pure neutral comments × 4 amplify MORE than identity (catalysis, 0.106).
Not an established phase transition in the formal sense.

**FRONTIER: NL omega comparable to code under full-vocab TV (exp #32, #37, #39,
#45).** The 11-bin projection concealed NL differences (ratio 18,000-64,000x).
With full-vocab TV (exp #45), NL omega amplification (5-17x on tested pairs)
reaches code-range levels. Passive voice pair: question mark amplifies 16.9x
(0.028->0.474), period 10.6x. house/home pair: period 5.6x (0.090->0.502). Code
pairs also lose signal in 11-bin when differences aren't in digit tokens
(boundary_pair: full/11bin ratio 3-21x). The small, unmatched panels do not
establish a general domain ordering; only one of four NL pairs meets full-vocab
baseline tau=0.05. The 11-bin readout is a code-specific optimization.

**FRONTIER: Base/instruct profile differences (exp #33, CORRECTED by exp #39).**
Base vs instruct omega profiles differ substantially on Qwen3-1.7B. On matched
cells (eligible in both models), heavy RISES from #3 (base, 0.034) to #1
(instruct, 0.158) — 4.6× stronger. The original exp #33 headline ("instruct
suppresses heavy") was an artifact of comparing different eligible cell sets.
Cross-architecture (StableLM): reshaping replicates but in architecture-dependent
direction. NOTE: base/instruct differences do not causally isolate RLHF from
other training differences (data mix, SFT, etc.).

**FRONTIER: Layer-wise profile (exp #35).** On 1.7B: omega is monotonically
increasing through model depth. Layers 0-16: omega≈0 (10^-5 to 10^-7). Layer 20:
first nonzero (0.0003). Layer 24: 0.010 (100× jump). Final: 0.049. Heavy is the
earliest amplifier (detectable at L12). No compression bottleneck. NOTE: layer
readout profiles do not causally localize a circuit or establish variable tracking;
they show where omega becomes measurable, not mechanism.

**FRONTIER: Quantization (exp #34).** INT8 dynamic quantization reshapes the omega
profile as dramatically as base/instruct differences. Identity drops from #1 (0.047)
to #5 (0.003). Mean TV(fp32, int8) = 0.378. Omega profiles measured at fp32 change
substantially under int8; certification transfer was not established. Omega is
precision-dependent.

**FRONTIER: Pair property (exp #44).** Tested 12 pairs to find what predicts
identity amplification vs preservation. REFUTES the "strong vs tight" framing:
identity amplifies only 2/8 eligible pairs (assert_pair I/N=1.5×, minimal_ok_good
I/N=2.5×) while PRESERVING for 6/8 — including "known strong" pairs (strong_assert
I/N=0.3×). Heavy is the strongest amplifier across all tested pairs (0.03-0.45).
No simple predictor works: not semantic category (value/structure/name change), not
baseline TV, not token diff count. The governing pair property remains an open question.

**Known limitations.** Cross-pair threshold prediction fails at tau=0.01-0.02
(omega is pair-dependent). Operational threshold: tau=0.05. Architecture-specific
amplification rankings mean omega magnitude is not directly comparable across models.
NL and code tau thresholds are on incomparable measurement scales (11-bin vs full-vocab TV).

**FRONTIER: Partial identity tokens (exp #38, BOUNDARY BUG IDENTIFIED by Fable R3).**
The "progressive buildup" narrative (x→x=→x=x→x=x\n) and "newline reduces 4×"
finding are ARTIFACTS of broken prompt assembly: continuations without trailing \n
get glued to the query (x becomes xf(), x= becomes x=f(), x=x becomes x=xf()).
Among VALID continuations only: x=None (0.367) > x=y (0.236) > type(x) (0.207) >
x=x\n (0.164) > x=1 (0.134) > #x (0.124) > _=x (0.077) > newline (0.070) > pass
(0.036) > nothing (0.007). The valid-continuation ranking still shows 23× amplification
for identity vs nothing, and any assignment amplifies. But the mechanistic claims
about progressive buildup and newline dampening are WITHDRAWN pending boundary fix
experiment (Fable Rank 1).

**FRONTIER: Repetition disentanglement (exp #40).** The interference from exp #31
(rep_2 < rep_1) is pair-dependent with OPPOSING directions. Strong pairs: additive
(II>I), last-statement-wins (NI>IN), neutrals accumulate (NNN>NN>N). Tight pairs:
OPPOSITE — suppressive (II<I), first-statement-wins (IN>NI), ending on identity
preserves. I_paraphrase (identity via temp swap) dramatically different from direct
I (0.001 vs 0.022) — syntactic form matters, not just semantics.

**FRONTIER: Layer scaling (exp #41).** 0.6B onset at L8 (vs L20 in 1.7B) — EARLIER
in smaller model. But weaker final layer (0.024 vs 0.049). The 0.6B distributes
omega across many layers; 1.7B concentrates sharply in L20-27. Larger models develop
more specialized late-layer variable-tracking circuits. 0.6B loses eligible cells
at L16-20 (representations less stable mid-network).

**Mechanistic (CORRECTED by exp #46/#47 boundary fixes + Fable R3 + Codex R1).**
The amplification mechanism is UNRESOLVED. "Continuation-dependent sensitivity" is
an accurate description: appended context can increase or decrease distinguishability
of selected histories under a specified next-token readout. The effect depends on
histories, continuation, query, model, and measurement channel. Exp #47 (192
readouts, proper prompt boundaries): on strong_assert_2, bare_var_nl (`x\n`) is
strongest (25x), nonvar_expr (`_ = 42\n`) amplifies 10x (same as identity_nl),
pass_stmt only 2.7x. Critical: glued_identity (11x) ≈ identity_nl (10x) — the #38
boundary confound does NOT explain away the identity effect. WITHDRAWN claims:
"progressive buildup" (boundary bug), "newline dampening" (boundary bug), "variable
context activation" (nonvar amplifies strongly), "computation context activation"
(still a hypothesis per Codex R1 — NL punctuation and neutral comments undermine any
claim that computation-adjacent syntax is necessary). Variable mention modulates
magnitude but is not required. Heavy is the strongest amplifier across all tested
pairs (exp #44). Remaining alternative explanations (Codex R1): readout
redistribution rather than recovered information; ordinary access to retained
history; position/tokenization/query interaction; confidence changes (softmax
saturation); selection effects from maximizing over continuations.

**FRONTIER: Authority framing (exp #43).** Adding source-frame comments before
continuations dramatically changes omega. On tight pairs (minimal_ok_good):
"# According to the documentation:" before neutral SUPPRESSES omega 97% (0.034x).
On wide pairs (strong_assert_2): same frame AMPLIFIES neutral 3.5x. The model
processes frames differently depending on the baseline distance landscape.

**FRONTIER: Comment taxonomy (exp #44).** Systematic test of 21 comment types across
7 semantic categories. On tight pairs: safety comments ("SAFETY: do not modify") are
the STRONGEST amplifier at 7.45x bare identity. Deprecation ("DEPRECATED: will be
removed") is the strongest suppressor at 0.15x. Category ranking: safety >
uncertainty > explanation > certainty > authority > deprecation. Safety implication:
a "do not modify" comment actually INCREASES model sensitivity to following code,
opposite of its intent.

**DATA: Baseline predictor (exp #46, analysis only).** Baseline TV correlates
with max omega globally (rho=0.738). Within-pair patterns are noisy: strong_assert_2
rho=-0.31 on n=6 (p=0.54, R5 WITHDRAWN as evidence of inversion). The global
correlation is dominated by the ineligible strong_assert pair.

**FRONTIER: Diagnostic set (exp #45, METHODOLOGICALLY FLAWED per Fable R3).** The
greedy selection score is circular — any 2-element subset trivially gets rho=1. The
reported rho=0.606 is NOT a valid coverage measure. Raw observations useful; the
compression/efficiency claim is withdrawn.

**FRONTIER: Referent crossover (exp #50, #51; REVISED by R4 + eligibility reanalysis).**
On eligible cells (12/18, tau=0.05), the gradient SURVIVES in a cleaner form:
baseline(0.009) < tracked(0.028) < none(0.068) < other(0.087). All three pairs
individually show tracked < other. The original "gradient reversal" was an artifact
of inflated baselines (0.296 from ineligible cells with TV 0.05-0.48). tracked_value
(`x = 1`) writes the answer into the prompt. "State confirmation anchor" and
mechanism labels are withdrawn. Exp #51 dual-variable: 100% eligible (all baseline
TV < 0.048); TSI = -0.008 at noise floor. #50 and #51 are one reproducibility check,
not independent evidence.

**FRONTIER: Revival (exp #42, 756 readouts; REVISED by R4 + eligibility reanalysis).**
R4 analyzed unfiltered data (including strong_assert, baseline TV=0.88) and found 1/25
chains exceeding control. Eligibility-filtered reanalysis (336/756 rows at
baseline≤0.05): 9/30 chains exceed no-contraction control — substantially stronger.
del_var consistently amplifies 3.3-3.8x across all probes on eligible cells (expected:
NameError context). none_assign+isinstance_check: 0.086 vs 0.042 control (still real).
blank_newlines+type_var: 0.094 vs 0.085 (marginal). strong_assert pair entirely
ineligible (0/252 cells, baseline=0.88). Revival is real but narrow: ~30% of chains
on eligible cells. LIMITATIONS: 11-bin only; del_var is trivially explained.

**FRONTIER: Consequence prediction (exp #49/#53, 576 readouts (not 864; #48=#49);
REVISED by Fable R4 + eligibility reanalysis).**
All-cells: 126 rows, 12 flips, AUC 0.939, r_pb=0.690. Eligible-only (tau=0.05):
84 rows, 7 flips, AUC=0.918, r_pb=0.639 — result SURVIVES eligibility filtering
with modest degradation. 12/18 cells eligible (5/6 strong_assert_2, 3/6 boundary,
4/6 minimal). Single-surface top-2 margin gets AUC 0.80-0.86 (R4); TV's 0.94 is
real pair-specific information but 80% of signal available from one surface. TV and
flip are computed from the same vectors (structural coupling). `heavy` scoring
contaminated (scores x+1 against pre-mutation value). Transfer across QUESTIONS
is untested — that is the real omega claim (exp #58 in progress). On BASE model
(exp #53): weak predictor (test F1=0.129). The strong result is instruct-specific.

**FRONTIER: Cross-model consequence (exp #55, 432 readouts, NEGATIVE).** tau=0.3
does NOT transfer across model sizes. TV-flip correlation is non-monotonic:
0.6B-Base r=0.307, 1.7B r=0.677, 4B-Base r=0.284. Best thresholds are model-
specific (0.6B: tau=0.05/F1=0.14; 4B: tau=0.15/F1=0.33). At tau=0.3, both 0.6B
and 4B have F1=0.000 (no pairs exceed TV=0.3). The pattern: 0.6B is too weak
(2.3% accuracy, near-chance — barely understands scoping); 4B is too robust
(91.2% accuracy, 2.8% flip rate — almost never flips). 1.7B sits in the
"vulnerable but capable" sweet spot. Consequence prediction via omega requires
a model in a specific capability regime for the task, not a universal property.

**CANCELLED: Identity paradox (exp #54; WITHDRAWN by Fable R4).** There is no
paradox. Identity has the lowest mean TV of all continuations (0.048) and the
highest margin (0.70). Zero flips at TV 0.05 is the expected outcome. The
"compression vs redistribution" taxonomy was derived from instruct-model data.
Cancel.

**FRONTIER: Autoregressive compounding (exp #56, 64 trajectories × 30 tokens;
DOWNGRADED by Fable R5 — pending E1 rerun).** Under greedy decoding on 32 surface
pairs (17 eligible), 11-bin cross-surface TV after 30 generated tokens is near 0
for 13 pairs and above 0.94 for 19; 28 of 32 pairs exceed TV 0.9 at some step.
The step-30 label is NOT stable across horizons — changing to step 20 or 40 changes
labels. Whether the two generated texts converge or merely phase-align was not
recorded (token ids not persisted). The rival explanation: coarse-readout phase
alignment under greedy decoding — two confident distributions give TV near 0 (same
bin) or near 1 (different bins); 81% of step-TVs lie outside (0.1, 0.9). Initial TV
predicts the step-30 label with AUC 0.68 on all pairs and 0.60 on eligible pairs.
E1 (token-persisted rerun with horizon stability and single-surface baselines) is
running to determine if this is a real dynamical phenomenon or a greedy artifact.

**FRONTIER: State reconstruction (exp #52, 240 readouts; REVISED by Fable R4).**
canonical_record and summary_nl produce TV=0 (tautological by design). Appended
overwrite shows target_query 0.006→0.036 — the "6x amplification" is a ratio of
near-zero numbers (absolute change 0.03, below tau=0.05). ortho_query 0.148→0.059
shows the overwrite doing what overwrites do. No valid repair arm exists in this
experiment design; finding should not be in STATE as established.

**DATA: nonvar_42 is NOT digit copying on base model (exp #59; MODEL MISMATCH with
#48).** R4's digit-copying finding was on Qwen/Qwen3-1.7B (post-trained, exp #48).
Exp #59 used Qwen/Qwen3-1.7B-Base. On Base: argmax is locked to the cell value
regardless of continuation literal (leading-digit match 9.7%, identical argmax census
36/36). On post-trained (#48): `_ = 42` gives argmax "4" in 20/36, empty already at
"9" in 23/36 — R4 was right for that model. These are DIFFERENT MODELS; neither
refutes the other. "Digit-copying DEAD" was a cross-model comparison error.
UNREPORTED #59 finding: identity (`x = x`) on Base flips argmax from correct outer
value to "9" (nested-99999 attractor) in 18/36 readouts across 5/6 cells.
Margin collapses from 0.42-0.81 to 0.01-0.42. Mechanism: two-basin competition
between outer value and innermost-scope attractor, tipped by continuations.

**SYSTEMIC: Eligibility filter dropped after #46 (Fable R4).** Experiments
#49/#50/#51/#42/#52 compute raw TV on all cells with no eligibility filter
(d_pre <= tau). Omega is only defined on pairs that start close. Roughly half the
cells have baseline TV 0.2-0.98. Several headline claims were eligibility artifacts.
All claims since #46 must be reassessed on eligible cells only.

~28,000+ readouts across 65+ experiments, 5 architectures. NOTE: #48 and #49 are
the same 288 readouts (double-counted in total). Effective n for consequence claims
is 18 cell-pairs on one model. CRITICAL: experiments #48 (post-trained) and #59
(Base) use different models — cross-model comparisons require matching model_id.

**Codex mechanism review R1 (2026-09-17):** "Continue the program, but pivot from
cataloguing amplifiers to testing whether omega predicts consequential failures
prospectively. The amplification phenomenon survives; the proposed mechanism remains
unidentified." Nine claims revised per review. Full review:
experiments/results/codex_mechanism_r1.md

**Fable strategic review R4 (2026-09-17):** Withdraws gradient reversal, revival
generality, 6x overwrite, identity paradox, digit-value-matters as eligibility or
selection artifacts. r=0.677 survives with CI [0.47, 0.81] but is only modestly
better than single-surface margin. Recommends cross-question transfer test and
digit-prior sweep before any cross-model replication. Full review:
experiments/results/fable_strategy_r4.md

**PROVISIONAL: Cross-question transfer hypothesis OPEN (exp #58, 35% complete).**
Partial data (200/576 readouts, 2/6 cells) yields a Spearman hint (ρ=0.651,
p=0.003, n=18 non-independent, single cell x=1) that TV ranks may correlate
between assert_eq and print_output. No cross-cell evidence. Continuation and
pair confounds unresolved (R7). Mean transfer AUC at chance (0.498). Fair margin
also fails at transfer (0.386). Same-Q certificate: 0/49 violations. Addition
shows no rank stability (ρ=-0.135, 0.057). Suffix-overlap rival ruled out.
One asymmetric transfer AUC (0.922) rests on 1 flip — not evidence of
discrimination. **Complete the experiment (remaining 4 cells) before concluding.**

**Fable strategic review R5 (2026-09-17):** Devastating provenance audit. #61 is
n=32 reanalysis of #56, NOT 108 pairs. #62 AUC delta is fit noise (7 flips, no CV,
F1=0 everywhere). Safety paradox WITHDRAWN (n=6, p=0.54, bugged panel). Bifurcation
DOWNGRADED: greedy-decoding/11-bin readout artifact pending token-persisted E1 rerun.
Length scaling downgraded (n=6/point, no CIs, heavy decline has cheap explanation).
Three priority experiments: E1 (token-persisted #56 rerun), E2 (null control +
temperature), E3 (answer-level consequence). Cross-question #58 remains the only
claim-bearing test. Full review: experiments/results/fable_strategy_r5.md

**FINDING: Margin dominates same-question flip prediction (exp #60b, #62b).** On
Qwen3-1.7B-Base, 120 (cell, continuation) rows from exp #53 with 7 argmax flips,
single-surface top-2 margin predicts same-readout flips with in-sample AUC 0.977
versus 0.838 for cross-surface TV. An in-sample logistic fit (no cross-validation,
F1=0 for all models) shows no incremental value of TV over margin plus continuation
type at this n. NOTE (R5): the AUC delta of -0.018 when adding omega is fit noise
on 7 positives, not evidence of negative information. The defensible conclusion:
omega does not add to margin for same-question prediction at this sample size.

**DATA: Length scaling (exp #57, n=6 cells per point, no CIs; DOWNGRADED by R5).**
Over 0-12 repetitions on 6 cells per point without confidence intervals: nonvar
(_=42) rises monotonically (0.029 to 0.121) — the one curve with no cheap
explanation. Heavy (x=x+1) rises then declines (0.126 at n=2 to 0.077 at n=12),
but repeated `x = x + 1` moves the correct answer to val+n (two digits for
val>=5 at n>=5), so both surfaces converging on "1" explains the decline without
dynamics. Identity (x=x) jumps at n=1, then wobbles 0.040-0.051. The nonvar
log-growth curve merits a CI before it merits a claim.

**DATA: Bifurcation predictor (exp #61, CORRECTED by R5: n=32 reanalysis of #56,
NOT 108 pairs).** On the 32 #56 trajectories, initial TV predicts the step-30 label
with AUC 0.676 (95% CI approx [0.48, 0.87]). The 108-pair design was never run.
Eligible-only (n=17, 8 grow): identity 3/3, heavy 1/1, empty 3/7, nonvar 1/6.
The reported "identity 88% grow" rests mostly on ineligible cells.

**WITHDRAWN: Inverted safety paradox (R5).** Within strong_assert_2 (n=6, exp #38
panel), baseline TV and max omega have Spearman rho=-0.31 (p=0.54). The source
panel has the R3 boundary bug, range restriction at the noise floor (baseline TVs
span 0.0002-0.01), and max_omega has winner's curse (max over 14 continuations).
Not established.

**FINDING: 11-bin ≡ full-vocab for code domain (exp #55b).** TV ratio 1.0004,
identical 6 flips, same correlations. The 11-bin readout loses no information vs
full vocabulary in code domain. No measurement artifact concern for existing claims.

**DATA: Continuation composition is non-commutative (exp #62, composition ordering).**
Chaining two continuations produces TV that depends strongly on ORDER.
neutral_then_identity (mean TV 0.100) is 2.1x identity alone (0.047) and 4x the
reverse order identity_then_neutral (0.025). verified_then_bare (0.040) is 7.3x
bare_then_verified (0.005). Amplification is real: several chains exceed the maximum
of either single continuation. This is static omega data (not trajectory), showing
that continuation sequence creates non-trivial interaction effects in the model's
state. N=11 cells per condition.

**CRITICAL: Exp #49 data unreproducible (model behavior mismatch, 2026-09-17).**
Fresh reproducer with identical code, model hash (ea980cb0), cell (x,1),
surface (minimal_ok), continuation (nonvar_42) gives P(1)=0.610 P(4)=0.206
(argmax=1, correct). #49 had P(1)=0.005 P(4)=0.760 (argmax=4, digit copying).
KV cache deepcopy verified intact (3 identical results across 36 intervening
calls). Full-forward matches KV (diff 1.31e-06). Tokenization verified identical.
Weights unchanged since Feb 17 2026. Cause: likely transformers library update
(now 5.12.1) changed attention computation. The AUC 0.918 result and the
digit-copying phenomenon from #49 are unreproducible artifacts. All post-#54
experiments use the current version and show consistent behavior (argmax = cell
value for all nonvar continuations). The 7 flips that drove AUC 0.918 were
likely version-dependent.

**FINDING: Neither TV nor margin transfers across questions.** On 500-readout
partial data from exp #58: cross-Q TV AUC 0.561 (near chance), cross-Q margin
AUC ~0.49 (also chance). Argmax agreement under Q1 has near-zero sensitivity for
Q2 flips. The transfer problem is fundamentally hard — no single-question
measurement predicts behavior under a different question. This is a deeper
negative than "omega fails."

Results: experiments/results/omega_*/. Reviews: experiments/results/fable_omega_review_{1,2,3,4}.md, fable_strategy_r4.md, fable_strategy_r5.md.
Synthesis: https://claude.ai/artifact/S3HWPNAvzPKNaWEvi7KJVk

---

## HISTORICAL: TRHO: Transfer Rho Spectrum — behavioral non-verifiability (2026-09-17)

**What d measures (R21 clarification):** For each (cell, perturbation) pair, the two forward
passes use the *minimal* surface (no header) and the *verbose* surface (`# Python code\n` header),
both with the perturbation applied. d[cell, pert] is the cosine distance between those two
full-logit vectors. The perturbation is the CONTEXT in which a fixed 4-token header edit is
measured — it is never compared to the unperturbed prompt. The 24 "perturbations" are 24
contexts; d is header-sensitivity-in-a-context.

Transfer rho quantifies how well behavioral distances measured under one set of
code perturbations predict behavioral distances under a different set. 5 model
families (Qwen3-1.7B, SmolLM2-1.7B, Phi-1.5, StableLM-2-1.6B, CodeGen-350M),
full-logit cosine readout, 40 expression-based cells, 24 perturbations (4 groups × 6),
fp32 CPU. Scale ladder complete: SmolLM2-135M, SmolLM2-360M, Qwen3-0.6B (2026-09-18).

**R21 multiple-testing note:** >70 hypothesis tests in this section; no family-wise or FDR
correction applied. Marginal results (p=0.007-0.05) should be interpreted with this in mind.

**Central finding (post-R19/R20/R21, corrected):** Behavioral sensitivity is predominantly
**multiplicative**: d[i,j] ≈ cell_fragility[i] × pert_magnitude[j]. R21 notes this is
first-order Taylor: median angle is 0.04-0.28 rad (small-angle regime), so d ~ theta²/2 where
theta ~ gain(cell) × push(context). Any smooth model is multiplicative to first order.
The CONTENT is entirely in the log-scale residual (12-25%).
On log scale, additive R²=0.82-0.88. **R²=1.000 claim RETRACTED:** the computation mixed
raw-scale residuals with log-scale total SS (62M× inflated denominator). Correct raw-scale
mult R² = 0.62-0.86; log-space R² = 0.82-0.88. That residual has **significant cell × group
interaction in 7/8 models** (nested F + group-label permutation, log scale, 8-model extension):
Qwen3 p=0.0003, SmolLM2 p=0.0007, StableLM p=0.0003, CodeGen p=0.009, Phi p=0.31.
Group D perturbations do NOT cohere internally (within-D rho 0.22-0.76); the
group-level interaction is not separable from edit locus in this design.
(Prior "5/5 additive, type-blind" claim RETRACTED — R20 found the permutation test
was a null operation: whole-row permutation leaves ISS invariant, all p-values were
floating-point roundoff. Prior "2/5 transfer" also retracted per R19.)

**Scale ladder results (2026-09-18, CORRECTED 2026-09-19):**
| Model | Params | Gap | 95% CI | Acc | P(digit) | Perm Z | p |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 135M | 0.121 | [0.065, 0.205] | 52.5% | 0.994 | 2.7 | 0.004 |
| SmolLM2-360M | 360M | 0.318 | [0.207, 0.431] | 90.0% | 0.993 | 6.0 | <0.001 |
| SmolLM2-1.7B | 1.7B | 0.281 | [0.197, 0.385] | 95.0% | — | 6.0 | <0.001 |
| SmolLM2-1.7B-Inst | 1.7B | 0.251 | [0.170, 0.339] | 94.4% | — | 3.2 | <0.001 |
| Qwen3-0.6B-Base | 0.6B | 0.087 | [0.052, 0.129] | 100% | 0.996 | 5.0 | <0.001 |
| Qwen3-0.6B (instruct) | 0.6B | 0.489 | [0.347, 0.628] | 100% | 1.000 | 8.8 | <0.001 |
| Qwen3-1.7B-Base | 1.7B | 0.452 | [0.315, 0.593] | 100% | — | 8.8 | <0.001 |
**SmolLM2 non-monotonic:** Gap peaks at 360M (0.318) then DECREASES at 1.7B (0.281).
Accuracy scales monotonically (52.5%→90%→95%). Gap-CIs overlap (R21 note: not
significant at n=3 scale points). Gradient exists at 135M with 52.5% accuracy (p=0.004).
**R23 CORRECTION (2026-09-19):** The 0.087 (Base) vs 0.489 (Instruct) comparison is a
definition artifact: "gap_fulllogit" was computed differently for the two files (within-minus-
between vs AA-minus-AD). Like-for-like comparison using the same metric (AA - AD): Base
0.324 → Instruct 0.489 = **1.5x** (paired CI [-0.004, +0.310]). The "5.6x" and "LARGEST
SINGLE EFFECT" claims are WITHDRAWN. **Replication on SmolLM2 (2026-09-19):** SmolLM2-1.7B
Base gap=0.281, Instruct gap=0.251 (0.89x). Gap amplification does NOT replicate —
the instruct effect is architecture-dependent.
**INSTRUCT CONFOUND — RESOLVED (2026-09-19).**
Base vs Instruct comparison (Qwen3-0.6B, same tokenizer, like-for-like gap):
**Base vs Instruct comparison (Qwen3-0.6B, same tokenizer):**
| Metric | Base | Instruct | Ratio |
|---|---|---|---|
| Gap (AA-AD, like-for-like) | 0.324 | 0.489 | 1.5x |
| Within-D rho | 0.394 | 0.169 | 2.3x |
| parens<->int_cast | 0.729 | 0.183 | 4.0x |
| Mean distance | 0.025 | 0.019 | 0.76x |
| PC1 share | 87.8% | 53.1% | — |
| Residual gap | 0.241 | 0.309 | — |
| Cross-variant fragility rho | 0.333 (p=0.036) | — | — |
Instruction tuning DESTROYS D-group internal coherence (0.394→0.169) and specifically
the parens<->int_cast coupling (0.729→0.183) despite identical tokenizer. Base model
has LARGER mean distances (0.025 vs 0.019) but LOWER gap — perturbation types are more
interchangeable in the Base model. Cross-variant fragility rho = 0.333: weak correlation
means instruction tuning reorganizes WHICH cells are fragile, not just how much.
**Residual gap (fragility-partialled) for scale ladder models:**
Qwen3-0.6B-Base: residual gap 0.241 [0.169, 0.307], PC1=87.8%. ALL GATES PASS.
  R21 SUPPRESSOR CAUTION: PC1=87.8% is high; residual gap (0.241) exceeds raw gap
  (0.087) by 2.8x — same suppressor pattern as SmolLM2-360M (PC1=90.5%).
Qwen3-0.6B (instruct): residual gap 0.309 [0.115, 0.487], PC1=53.1%.
Qwen3-1.7B: residual gap 0.230 [0.077, 0.430], PC1=51.0%.
SmolLM2-1.7B: residual gap 0.310 [0.091, 0.533], PC1=78.2%.
SmolLM2-360M: residual gap 0.384 [0.205, 0.480], PC1=90.5%. R21 SUPPRESSOR WARNING.
SmolLM2-135M: residual gap 0.105 [-0.015, 0.251], PC1=69.6%. FAILS (CI incl 0).
**PC1 ordering (scale):** Qwen3-1.7B (51%) < Qwen3-0.6B-instruct (53%) << SmolLM2-135M (70%)
< SmolLM2-1.7B (78%) < Qwen3-0.6B-Base (88%) < SmolLM2-360M (91%). Lower PC1 = more
complex sensitivity structure beyond simple fragility. Base model has HIGHER fragility
dominance than instruct — instruction tuning creates multi-dimensional sensitivity.

**Evidence for the surviving claim:**
- Same-type transfer (6 inert appends): rho 0.63-0.76, CIs exclude zero
- Novel inert held-out (3 new types): rho ~ 0, all p > 0.4
- Cross-type (surface -> semantic): Qwen3 CI [-0.03, +0.34], includes zero
- Within-semantic: Qwen3 CI [-0.004, +0.34], includes zero
- One-factor fragility model rejected in 4/5 model x readout combinations
- Cross-model fragility: rho=0.24 (code, NS) and -0.06 (NL, zero)

**WITHDRAWN claims (R13 triage):** "Qwen3 collapses in NL" (fp16 floor),
"SmolLM2 domain-stable" (fragility-dependent), "Phi gap erasure code-specific"
(precision mismatch), mac/sqrt(d_eff) (in-sample/tautological), "no negative
correlations in full-logit" (false in NL), biology >> LLM (no backing data),
d_eff as independent finding (restatement of correlation structure).

**RESOLVED: position confound (kill experiment, 2026-09-18).** Position contributes
~0.2 to the gap but does NOT explain it. Cross-position inerts (insertions before
the assignment) transfer at rho=0.611 vs same-position inerts at 0.785-0.811. The
remaining ~0.3 gap between cross-position inerts (0.611) and rewrites (0.333) is
genuine: touching the computation path vs not touching it. Rewrites barely transfer
to each other (0.224) — each probes a unique sensitivity dimension.

**Kill experiment results (Qwen3-1.7B, fp32 CPU, 40 cells x 24 perts; R14 REVIEWED):**
- Inert appends transfer as a class: a<->b = 0.811 (new inerts predict old inerts)
- Position is a PARTIAL confound (~35% of gap): a<->c = 0.611 vs a<->a = 0.785
- Gap IS real after position control: a<->d = 0.333 (rewrites don't transfer)
- Rewrites are NOT a coherent class: d<->d = 0.224 (std 0.250 > mean, includes negatives)
- Gradient: same-type (0.81) > cross-position (0.61) > cross-type (0.33) > within-rewrite (0.22)
- fp32 CONFIRMS 11-bin gradient (but doesn't fix full-logit magnitudes: 54% still < 1e-3)
- R14 CORRECTION: a<->b > a<->a (0.811 vs 0.785) is group-homogeneity noise, not evidence
  of "better than training" transfer — Group B is internally more homogeneous (std 0.064 vs 0.106)
- R14 CAVEAT: "computation path" interpretation is unfalsifiable as stated; conflated with
  syntactic novelty (Group D uses unusual syntax, A/B/C use common idioms). Needs syntactic
  novelty control experiment.
- Statistical significance: Mann-Whitney p=7.9e-13 (a<->b vs a<->d), p=2.1e-8 (a<->c vs a<->d)
- Bootstrap CIs (95%, cell resampling): a<->b [0.72, 0.86], a<->c [0.46, 0.71], a<->d [0.15, 0.47], d<->d [0.07, 0.37]
- Rewrites produce larger distances in 92% of cells (37/40) — not driven by a subset
- Within-rewrite negative correlations: d_temp_var<->d_add_zero = -0.328 (each rewrite probes a
  UNIQUE dimension). Within-inert all rhos > 0.55 (all p < 0.001) — inerts ARE one class

**Transfer hierarchy (11-bin readout, POSITION-CONTROLLED by kill experiment):**
- Within-surface rho: 0.63-0.74 (trivial perturbations self-consistent)
- Cross-position same-type: ~0.57 (position contributes ~0.2 of the original gap)
- Cross-type rho: 0.08-0.37 (surface cannot predict semantic — CONFIRMED with controls)
- Within-semantic rho: 0.14-0.35 (structural changes are unpredictable)

**Effective dimensionality (model-independent at 11-bin):**
- Surface: ~1.87-1.89 effective dimensions (participation ratio)
- Semantic: ~4.05-4.23 effective dimensions
- Ratio 2.2x is MODEL-INDEPENDENT across Qwen3 and SmolLM2
- PC3/PC4 of semantic are nearly orthogonal to surface (2-4% overlap)

**1/sqrt(d_eff) theoretical bound:**
- Matches surface rho within 2% (predicted 0.731, observed 0.718)
- Overshoots semantic by 3.7x (predicted 0.497, observed 0.135)
- The excess = adversarial structure in semantic perturbation interactions

**Mechanistic source — perturbation correlations:**
- Surface: all pairwise correlations positive (min 0.38-0.48)
- Qwen semantic: 3 NEGATIVE correlation pairs (min r=-0.30)
- SmolLM2 semantic: zero negative pairs but all weak (min 0.01)
- Negative correlation means learning one semantic effect MISLEADS about another

**Full-logit readout vs 11-bin — gap is MODEL-SPECIFIC:**
- Qwen3 full-logit: surf=0.661, sem=0.227 (gap PRESERVED, ratio 2.0x)
- SmolLM2 full-logit: surf=0.760, sem=0.511 (gap REDUCED, ratio 1.8x)
- Phi-1.5 full-logit: surf=0.696, sem=0.662 (gap ERASED, ratio 1.05x)
- Phi-1.5 (code-trained) has semantic dimensionality 1.07x surface
- Code-specific training may reduce semantic perturbation dimensionality

**Non-Euclidean geometry hypothesis: NEGATIVE.**
Triangle inequality satisfied. All perturbation angles acute (5.7-35.5 deg).
Gromov hyperbolicity test shows weakly positive curvature (not hyperbolic).
Cross-space distance Spearman 0.35-0.55 (moderate, not degenerate).
The non-transfer comes from signal structure (heterogeneous cell sensitivity
patterns), not from non-Euclidean geometry of the perturbation space.

**Bootstrap CIs (500 iterations, cell resampling):**
- Qwen cross-type 11-bin: [-0.031, 0.341] — includes zero
- SmolLM2 cross-type 11-bin: [0.185, 0.709] — excludes zero
- All within-surface CIs exclude zero
- Qwen full-logit: surf [0.427, 0.848], sem [0.032, 0.540]

**Noise calibration:** Deterministic TV=0.0, permutation p<0.001, 4.9sigma.

**NL generalization (Qwen3-1.7B-Base, GPU):**
NL math uses the same arithmetic problems in question format instead of code.
100% engagement (accuracy 1.0, P(digit)=0.999). ALL rho values collapse:
- 11-bin: surface 0.518 (vs 0.685 code), semantic 0.094 (vs 0.309 code)
- Full-logit: surface 0.175 (vs 0.661 code), semantic 0.048 (vs 0.227 code)
- NL surface dimensionality: 3.81 (vs 1.62 code) — every NL word is semantic
- Dimensionality ratio: 1.44x (vs 1.99x code) — NL surface is also high-dim
Implication: code's strict syntax creates a uniquely clean surface/semantic
separation. NL has NO genuinely trivial perturbation class. Code alignment is
partially certifiable; NL alignment is uncertifiable at this granularity.

**Perturbation transfer matrix (full-logit, zero-cost analysis):**
- Qwen sem-sem mean rho=0.361, worst pair varname_y<->assign_twice at 0.044
- SmolLM2 sem-sem mean rho=0.613; Phi-1.5 sem-sem mean rho=0.754
- Phi-1.5's gap erasure comes from HIGH within-semantic correlation, not low surface
- No negative correlations in full-logit; 11-bin negatives are readout artifacts
- func_call is universally the lowest-correlated semantic perturbation

**Common-factor null (fragility control):**
Fragility (per-cell mean TV) accounts for 55-90% of raw rho. After partialling
out fragility: Qwen gap AMPLIFIES (surf partial rho 0.39-0.73, sem 0.07 to -0.24);
SmolLM2 gap shrinks; Phi-1.5 semantic > surface. The confound was real but the
Qwen finding is STRONGER with the control.

**NL multi-model comparison (full-logit, confirmed):**
- Phi-1.5: gap erasure is CODE-SPECIFIC. Code ratio 1.05x -> NL ratio 1.82x
- SmolLM2: DOMAIN-STABLE. Code ratio 1.49x -> NL ratio 1.50x (unchanged!)
- Qwen3: COLLAPSES in NL. Surface 0.661->0.175, semantic 0.227->0.048
- Model ordering reverses in NL: SmolLM2 (0.492) > Phi-1.5 (0.338) > Qwen3 (0.048)
- Phi-1.5 has P(digit)=0.001 in NL (11-bin unreliable); full-logit valid

**Improved theoretical bound: rho ~ mac / sqrt(d_eff).**
mac = mean absolute pairwise Spearman correlation between perturbation effects.
mac/sqrt(d_eff) has mean error 6.7% vs 15.6% for 1/sqrt(d_eff) alone.
Qwen3 semantic: 11% error (vs 146% for 1/sqrt alone). The formula captures both
pathway-sharing structure (mac) and dimensionality (d_eff). All three predictors
(1/sqrt, mac, mac/sqrt) rank-order the models correctly (n=6).

**Cross-domain fragility decomposition (NL vs code):**
Fragility loading patterns are MODEL×DOMAIN specific — the same model allocates
fragility differently in code vs NL:
- Qwen3 CODE: fragility dominates both surface (R2=88%) and semantic (R2=83%)
- Qwen3 NL: fragility barely touches surface (R2=0.6%) but dominates semantic (R2=92%)
- SmolLM2 CODE: surface fragility (R2=94%), low semantic fragility (R2=30%)
- SmolLM2 NL: fragility dominates both (R2=80-90%)
After removing fragility, the partial gap is CONSERVED for Qwen3: +0.255 (code) vs
+0.256 (NL). The gap is an architectural invariant even though what fragility
captures changes completely between domains. SmolLM2 code retains genuine
semantic structure (partial rho=0.377); SmolLM2 NL collapses entirely.

**StableLM-2-1.6B (4th architecture family, PARTIAL):**
- Code full-logit: surf=0.918, cross=0.763, sem=0.827 (ratio 0.97x — NO GAP)
- Code 11-bin: surf=0.735, cross=0.419, sem=0.406 (ratio 1.36x)
- NL full-logit: surf=0.640, cross=0.443, sem=0.206 (ratio 1.91x — HUGE gap)
- NL 11-bin: surf=0.612, cross=0.091, sem=0.025 (ratio 1.55x)
- Pattern: code gap erasure (like Phi-1.5), NL gap massive (1.91x)
Falcon-H1-1.5B (hybrid SSM-attention, 5th family) running.

**SmolLM2-1.7B kill experiment replication (2026-09-18):**
- GRADIENT REPLICATES: same-type (0.86) > cross-position (0.67) > cross-type (0.58) > within-rewrite (0.44)
- Gap COMPRESSED vs Qwen3: a<->a minus a<->d = 0.281 (SmolLM2) vs 0.452 (Qwen3)
- ZERO distances below 1e-3 in full-logit — no fp16/fp32 floor concern whatsoever
- 95% accuracy (2 cells wrong)

**Phi-1.5 kill experiment (2026-09-18, Q0: NO GRADIENT; Q1: GRADIENT CONFIRMED — readout artifact):**
- Full-logit: a<->a=0.807, a<->d=0.752, d<->d=0.763. All rhos ~ 0.7-0.8. MWU p=0.52 (NS)
- Rewrites transfer to inerts AND to each other — no perturbation type boundary
- **P(digit) VERIFIED NEAR-ZERO (R15 condition resolved):** Sum P(any digit) = 0.0006-0.0028
  across test cells. Model assigns 97-99% probability to NEWLINE, not digits. The "92.5% accuracy"
  is argmax within the digit bins (correct ordering of <0.3% probability mass) — NOT meaningful
  task engagement. Phi does not predict digits; it predicts code continuation (newline)
- **11-bin INVALIDATED:** 84% distances < 1e-3 is EXPLAINED by P(digit) near zero (noise on noise).
  All Phi 11-bin rhos are unreliable and should be disregarded
- **Full-logit no-gradient SURVIVES:** Cosine distances (0.002-0.004 range) are on the full 51K-dim
  logit vector, dominated by non-digit tokens. The model IS sensitive to perturbations on the
  full distribution; it just doesn't differentiate between perturbation TYPES
- Phi's RII=0.763 (full-logit) measures consistency of perturbation effects on code-continuation
  behavior, not on digit prediction — a fundamentally different measurement than the other models
- R15: "code-specialized training" attribution is confounded with architecture (parallel attn),
  tokenizer (GPT-2 BPE, 51k vocab), model size (1.3B vs 1.7B), and training volume

**MODEL-DEPENDENT gradient (kill experiment, 5 architectures + Base/Instruct variant, IDENTICAL cells):**
- Qwen3-0.6B (instruct, 100% acc): STRONGEST gradient, gap 0.489 [0.347, 0.628], Z=8.8
- Qwen3-1.7B-Base (general, 100% acc): STRONG gradient, a<->d=0.333 << a<->a=0.785 (gap 0.452)
- CodeGen-350M-mono (code, 32.5% acc): GRADIENT, a<->d=0.444, a<->a=0.762 (gap 0.318, MWU p=2.3e-7)
- SmolLM2-1.7B (general, 95% acc): COMPRESSED gradient, a<->d=0.583, a<->a=0.864 (gap 0.281)
- StableLM-2-1.6B (general, 82.5% acc): MODERATE gradient, a<->d=0.628, a<->a=0.927 (gap 0.299, MWU p=4.5e-13)
- Phi-1.5 (code, 92.5%/95% acc Q0/Q1): NO gradient under Q0 (gap 0.055); **GRADIENT UNDER Q1
    (digit-predicting readout): gap 0.217 [0.123, 0.355], p=5.26e-11, Z=4.6, residual gap
    0.386 [0.171, 0.564]. ALL GATES PASS.** Q0 no-gradient was a READOUT ARTIFACT:
    P(digit)=0.001 under Q0, 0.980 under Q1. Model-specific: Q1 inadmissible on Qwen3
- **Qwen3-0.6B-Base (general, 100% acc): WEAK gradient, gap 0.087 [0.052, 0.129], Z=5.0.**
    LOWEST gap of any model with 100% accuracy. a<->a=0.910, a<->d=0.586, d<->d=0.394.
    Residual gap 0.241 [0.169, 0.307], PC1=87.8%. ALL GATES PASS.
    Like-for-like gap: 0.324 (1.5x lower than instruct 0.489; R23-corrected from 5.6x).
- All use the same 40 expression-based cells, 24 perturbations, fp32 CPU — no cell-design confound
- **Instruct effect (n=1 pair):** Qwen3-0.6B-Base (0.324 like-for-like) vs instruct (0.489).
  1.5x amplification.
  **SmolLM2-1.7B (second family):** Base gap=0.281, Instruct gap=0.251, ratio=0.89x.
  SmolLM2 shows NO gap amplification. All distances increase uniformly (1.9-2.8x).
  **Instruction tuning effect on gap is ARCHITECTURE-DEPENDENT, not general.**
  Cross-variant fragility: SmolLM2 rho=0.689 (strong) vs Qwen3 rho=0.333 (weak).
  SmolLM2 preserves its fragility fingerprint across instruction tuning; Qwen3 doesn't.
- CodeGen-350M 11-bin is dead (100% distances < 1e-3, near-zero digit probability);
  only full-logit valid. 32.5% accuracy means task incompetence is a major confound
- StableLM has highest a<->a (0.927) and a<->b (0.937) — strongest inert transfer,
  yet still drops to 0.628 for rewrites. Strong baseline ≠ strong cross-type transfer
- R15: "code-specialized → no gradient" REFUTED by CodeGen-350M (code-trained, gradient
  present). Phi's no-gradient is confounded with architecture, tokenizer, scale, accuracy,
  and training volume — cannot be attributed to code-specialization alone

**REWRITE INTERCHANGEABILITY INDEX (RII = within-group-D Spearman correlation):**
- Phi-1.5: RII=0.763 (but P(digit)≈0; measures code-continuation, not digit prediction)
- StableLM-2: RII=0.566
- SmolLM2: RII=0.443
- CodeGen-350M: RII=0.368 (P(digit)≈0 like Phi, but STILL shows gradient — refutes
  "not doing the task" as explanation for Phi's no-gradient)
- Qwen3: RII=0.224 (includes negatives)
- **MEASUREMENT SURFACE NOTE:** Phi and CodeGen put near-zero probability on digit tokens
  (P(digit) ≈ 0.001 and ≈ 0 respectively); their 11-bin readout is noise. Qwen3/SmolLM2/StableLM
  genuinely predict digits (11-bin median distance 0.015-0.094, 0% below 1e-3). However, the
  gradient and RII are computed on FULL-LOGIT cosine distance (full vocabulary vector), which
  captures whatever the model IS doing regardless of which tokens carry probability. The
  full-logit comparison is valid across all 5 models. The confound applies ONLY to 11-bin readout
- RII vs gradient (all 5 models, n=5): Pearson r=-0.944 (p=0.016), Spearman r=-0.900 (p=0.037)
- RII vs gradient (4 comparable 1.3-1.7B models, n=4): Pearson r=-0.952 (p=0.048)
- RII vs gradient (3 digit-predicting models only, n=3): Pearson r=-0.897 (p=0.29, underpowered)
- R16 E-corrected: RII is ~50-78% fragility share; this correlation should be read as
  fragility-share vs gradient. n=5, no CI. The GAP (not RII) is the perturbation-type-specific
  quantity that survives fragility removal
- R15 CORRECTION: causal attribution ("code-specialized → high RII") was OVERREACH.
  The correlation is empirical; the driver is unknown (training data, scale, architecture,
  measurement surface, or some combination). RII is a descriptive index, not a causal mechanism
- **RETRACTED: z-RII / FDI (R16 R2 — null artifact).** z-scored RII ≈ -0.18 is the expected
  pairwise Spearman after row-centering k=6 columns (-1/(k-1) = -0.20; pure noise gives -0.192).
  FDI = RII + 0.18 is a constant shift; FDI vs gap r=-0.952 is the existing RII vs gap relabeled.
  All z-RII/FDI claims withdrawn
- **FRAGILITY-PARTIALLED GROUP RHOS (R16 E-corrected, rank regression on per-cell mean distance):**
  PC1 (common fragility) share: Qwen3 51%, CodeGen 58%, Phi 73%, StableLM 77%, SmolLM2 78%.
  Residual RII collapses to 0.05-0.13 for all 5 models (vs raw 0.22-0.76, range 0.11 vs 0.54);
  analytic baseline -1/(k-1) ≈ -0.04 to -0.10 depending on PC1 share (not zero). Cross-model
  RII differences are mostly fragility-share differences. But the GAP SURVIVES partialling:
  **Raw gap CIs (bootstrap 95%, 1000 resamples):**
  Qwen3 +0.452 [+0.315, +0.593], CodeGen +0.318 [+0.202, +0.436], StableLM +0.299 [+0.191, +0.438],
  SmolLM2 +0.281 [+0.197, +0.385] — all exclude 0. Phi Q0 +0.055 [+0.005, +0.106] barely excludes 0.
  **Residual gap CIs (fragility-partialled):**
  Qwen3 +0.230 [+0.077, +0.430], CodeGen +0.277 [+0.100, +0.421], StableLM +0.560 [+0.256, +0.750],
  SmolLM2 +0.310 [+0.091, +0.533] — all exclude 0. Phi Q0 +0.005 [-0.076, +0.110] includes 0 (NS).
  Residual RII near zero for all models (CIs all include or touch 0): Qwen3 +0.062 [-0.037, +0.142],
  SmolLM2 +0.051 [-0.046, +0.164], StableLM +0.061 [-0.027, +0.218], CodeGen +0.078 [-0.025, +0.197].
  Exception: Phi +0.125 [+0.029, +0.263] — the only model with residual RII excluding 0.
  **Permutation null for residual RII:** Qwen3 bias-corrected +0.099 (>null 97.5th),
  SmolLM2 +0.071 (within null), StableLM +0.078 (within null), Phi +0.142 (>null 97.5th),
  CodeGen +0.118 (>null 97.5th). Phi's high residual RII is genuine, not a partialling artifact.
  The gradient is not a common-factor artifact; Phi's absent Q0 gradient is not fragility-masked
  but IS a readout artifact (see PHI READOUT SATURATION).
  **Phi Q1 fragility partialling COMPLETED:** residual gap +0.386 [+0.171, +0.564] excl 0.
  PC1 share 84.6%. Despite highest fragility share, Phi Q1 has SECOND HIGHEST residual gap.
  Residual RII under Q1: 0.219. All 5 models now have residual gap CIs excluding 0 under
  admissible readouts
- **PHI READOUT SATURATION — CONFIRMED (R16 Step 1 + R18 restructure, 2026-09-18):**
  Removing ONE trailing space from the query (`# Output: ` → `# Output:`) flips Phi from
  code continuation (P(digit)=0.001) to full digit prediction (P(digit)=0.980, accuracy 95%).
  Tokenizer mechanism: Q0 ends on space token (→ newline prediction), Q1 ends on colon token
  (→ digit prediction). Phi KNOWS the answer but outputs newline instead of digit under Q0.
  **PRIMARY RESULT (40-cell kill-matched, Q1 query, same cells as other 4 models):**
  gap=0.217, CI [0.123, 0.355], p=5.26e-11. P(digit)=0.980, accuracy=95%.
  a↔a=0.899, a↔d=0.682. Pre-registered gates PASS (gap ≥ 0.20 AND CI excludes 0.10).
  Q0 control on same 40 cells: gap=0.055 [0.010, 0.106] — EXACT MATCH to original kill.
  **SUPPLEMENTARY (20-cell, different cells/surfaces — NOT directly comparable to kill):**
  gap=0.335, CI [0.141, 0.494], p=8.76e-9. 20 varied-variable-name cells, surfaces
  "" vs "# init\n" (not kill surfaces). P(digit)=0.843, accuracy=85%. Larger gap partly
  explained by different cell design, surface pair, and lower accuracy. a<->a=0.728, a<->d=0.393.
  RII under Q1: 0.235 [0.051, 0.412].
  **STATUS: 5/5 CONFIRMED (R18 definitive test, 2026-09-18).** Phi Q1 40-cell passes ALL gates:
  (1) gap 0.217 >= 0.20 PASS, (2) CI [0.123, 0.355] lower > 0.10 PASS,
  (3) permutation null Z=4.6 (p<0.001) > 2 PASS, (4) residual gap 0.386 [0.171, 0.564] excl 0 PASS.
  PC1 share 84.6% — high fragility, but type-specific signal survives. Residual gap (0.386) is
  the SECOND HIGHEST after StableLM (0.560), ABOVE Qwen3 (0.230) and CodeGen (0.277).
  Raw gap ordering: Qwen3 (0.452) > CodeGen (0.318) > StableLM (0.299) > SmolLM2 (0.281) > Phi-Q1 (0.217).
  Residual ordering: StableLM (0.560) > Phi-Q1 (0.386) > SmolLM2 (0.310) > CodeGen (0.277) > Qwen3 (0.230).
  All 5 confirmed under their admissible readouts. Phi requires model-specific Q1 correction
  (inadmissible on Qwen3). "5/5 under model-specific admissible readouts" is the honest framing.
  **QWEN3 Q1 CONTROL:** Q1 is INADMISSIBLE on Qwen3 (P(digit)=0.019, 0% cells ≥ 0.5).
  The trailing-space effect is MODEL-SPECIFIC and OPPOSITE: Phi needs no-space (Q1) to predict
  digits; Qwen3 needs space (Q0). Cross-model magnitude comparison confounded by readout mode
- **PERMUTATION NULL (independent confirmation, 2026-09-18):**
  Shuffle perturbation labels within each cell (500 permutations). Under null: only fragility
  matters. Results (Q0): Qwen3 Z=8.8, SmolLM2 Z=6.0, StableLM Z=6.2, CodeGen Z=5.5 (all p<0.001).
  Phi Q0 Z=1.0 (p=0.168, NOT significant — readout artifact). **Phi Q1 Z=4.6 (p<0.001) — CONFIRMED.**
  All 5 models pass permutation null under their admissible readouts.
  Permutation 95th percentile approx 0.08-0.10 for all models; any real gap exceeds this by 2.5-5x
- **P(DIGIT) VERIFICATION (from 11-bin distance magnitudes, no model loading):**
  Qwen3: median 0.015 (0% < 1e-3, DIGIT). SmolLM2: 0.094 (0%, DIGIT). StableLM: 0.089 (0%, DIGIT).
  Phi: 0.0003 (83%, CODE CONTINUATION). CodeGen: 0.00001 (100%, CODE CONTINUATION).
  Clean binary split confirms 11-bin is INVALID for Phi/CodeGen; full-logit valid for all
- **DIGIT-CONCENTRATION HYPOTHESIS TEST (R18 Reading B, zero-cost, 2026-09-18):**
  If the gradient were purely P(digit)-driven concentration, 11-bin (digit-only) gap should be
  ≥ full-logit gap. Observed 11-bin/full-logit gap ratios: Qwen3 0.79 (11-bin SMALLER),
  SmolLM2 1.33, StableLM 1.13, Phi-Q0 1.57, CodeGen 0.60 (11-bin MUCH smaller).
  **REFUTED at comparative level:** CodeGen has full-logit gap 0.318 despite P(digit)≈0 — the
  gradient lives in the non-digit logit space. Qwen3 has full-logit gap (0.452) > 11-bin gap
  (0.359) — non-digit dimensions carry ADDITIONAL signal. The gradient is NOT purely
  digit-concentration. Definitive test (masking digit dims from logit vector) requires model loads.
- **CELL-EXCLUSION ROBUSTNESS (all 5 models, 2026-09-18):**
  Phi L3O exhaustive (9,880 triples): max gap 0.077 — no cell subset creates a gradient.
  Gradient models L3O (2,000 random triples): min gap Qwen3 0.387, SmolLM2 0.232,
  StableLM 0.238, CodeGen 0.255 — gradient never collapses under any cell exclusion
- **COPY CONTROL — GRADIENT IS SURFACE SENSITIVITY (2026-09-19):**
  40 literal-assignment cells (x=5, y=3, etc.) under same 24 perturbations per model.
  | Model | Copy Gap | CI | Expr Gap | Ratio | H_surface | Admissible |
  |---|---|---|---|---|---|---|
  | Qwen3-1.7B | 0.403 | [0.288, 0.516] | 0.452 | 89% | PASS (0.049) | YES |
  | SmolLM2-1.7B | 0.386 | [0.261, 0.527] | 0.281 | 137% | PASS (0.105) | YES |
  | StableLM-2-1.6B | 0.193 | [0.094, 0.315] | 0.299 | 65% | PASS (0.106) | YES |
  | CodeGen-350M | 0.241 | [0.163, 0.337] | 0.318 | 76% | PASS (0.077) | NO (70% acc) |
  | Phi-1.5 | 0.197 | [0.115, 0.281] | 0.217 | 91% | PASS (0.020) | NO (P(digit)=0.001) |
  **5/5 models tested. Series COMPLETE.** H_surface/H_compute binary thresholds DEPRECATED
  (see R22 correction below). StableLM: indeterminate (ratio CI [0.31, 1.05] includes both
  0.5 and 1.0). Phi copy control run under inadmissible Q0 readout (Q1 was available).
  **R22 CORRECTION (ratio bootstrap CIs):** All ratio CIs include 1.0:
  Qwen3 0.89 [0.64, 1.14], SmolLM2 1.37 [0.93, 1.87], StableLM 0.65 [0.31, 1.05].
  Copy and expression gaps are statistically INDISTINGUISHABLE at 95% for all 3 admissible
  models. SmolLM2 ratio>1 is NOT significant — "computation-as-shielding" unsupported.
  Binary H_surface/H_compute thresholds DEPRECATED in favor of CI overlap.
  Phi NOT admissible (P(digit)=0.001). CodeGen NOT admissible (70% accuracy).
  3/5 admissible models all show ratio CIs including 1.0.
- **POSITION vs CONTENT — R22 WITHDRAWN AS STATED (zero-cost, 2026-09-19):**
  Original: same-position/diff-content 0.705 vs same-content/diff-position 0.530 (+0.175).
  **R22 correction:** The +0.175 is entirely the A/B after-block cohesion (already known from
  R13-R14). Balanced before-only comparison: mean +0.02, position wins 2/7, content 2/7.
  Content-matched cross-position pairs are NO MORE correlated than unmatched ones in 5/7
  models (e.g. Qwen3-1.7B matched=0.607, unmatched=0.614). Content identity is UNDETECTABLE.
  Three confounds: header adjacency (C sits adjacent to measured edit), magnitude floor
  (C has smallest-magnitude perturbations), communality attenuation. "Position dominates
  content" WITHDRAWN; replaced with: "after-inerts form one class; content identity is
  undetectable; position vs header-adjacency not yet separable."
- **LOCUS ZERO-COST — R22 WITHDRAWN (zero-cost, 2026-09-19):**
  Original: same-line inerts +0.070 more D-correlated (7/7 positive, 2/7 significant).
  **R22 correction:** "Different-line" pool included the 6 C perturbations (whose low
  D-correlation is the best-established fact in the program). Against A/B inerts only:
  +0.037, 1/7 significant, exact-permutation floor p=0.018. Same-line inerts are within
  0.015 of a_empty (unperturbed baseline) in 7/9 models — they ARE the baseline, not
  "more D-like". For Qwen3, a_whitespace is token-identical to a_empty (R17). WITHDRAWN.
- **REPLAY NOISE FLOOR — REPLAY-DETERMINISTIC; NOT A TEST OF RANK STABILITY (2026-09-19, R22+R23 correction):**
  | Model | Noise max | Actual min | SNR | Verdict |
  |---|---|---|---|---|
  | Qwen3-1.7B | 3.1e-6 | 1.29e-4 | 316× | CLEAN |
  | SmolLM2-135M | 3.6e-7 | 1.56e-4 | 10,408× | CLEAN |
  Both largest (Qwen3) and smallest (SmolLM2-135M) models: 0% of kill experiment distances
  below replay noise. Smaller model has LOWER noise (fewer params → less fp32 non-determinism).
  All kill experiment measurements are well above fp32 replay noise.
- **POSITION SHIFT MECHANISM TEST — NULL RESULT (zero-cost, 7 models, 2026-09-19):**
  If absolute token-count shift caused the position effect, C pairs with similar shift
  (same tokens added) should correlate more. Test: Spearman(shift_diff, C-pair rho) across
  all 15 C pairs per model. Grand correlation r=0.010 (p=0.918, n=105 pooled pairs). NULL.
  5/7 models have NEGATIVE correlation. SmolLM2-1.7B: r=0.655 but highest C pair has LARGEST
  shift difference (shift=4, rho=0.875) — OPPOSITE of prediction. Position organization is
  QUALITATIVE (before vs after the assignment in causal attention window), not QUANTITATIVE
  (how many tokens shift the position). Mechanism is causal attention asymmetry: before-
  perturbations enter the assignment's attention context; after-perturbations don't.
- **INSTRUCTION TUNING EFFECT — R22 WITHDRAWN + REPLICATION FAILED (2026-09-19):**
  **Qwen3-0.6B** (original claim): Gap UP 0.324→0.489, A↔C UP, A↔D DOWN, D↔D DOWN.
  **R22 correction:** Paired-cell bootstrap: only decreases significant (A↔A -0.12, A↔D -0.28,
  D↔D -0.22); A↔C "up" is +0.09 [-0.11, +0.29] NS; gap marginal under one definition,
  null under the other. Base PC1=0.69 vs IT 0.59 → higher base A↔D is common-factor inflation.
  **SmolLM2-1.7B REPLICATION (this session):** ALL correlations DROP uniformly.
  | Metric | SmolLM2 Base | SmolLM2 IT | Qwen3 Base | Qwen3 IT |
  |---|---|---|---|---|
  | a↔a | 0.864 | 0.748 (DOWN) | 0.910 | 0.876 (DOWN) |
  | a↔c | 0.672 | 0.480 (DOWN) | 0.410 | 0.503 (UP?) |
  | a↔d | 0.583 | 0.497 (DOWN) | 0.586 | 0.301 (DOWN) |
  | d↔d | 0.443 | 0.323 (DOWN) | 0.394 | 0.169 (DOWN) |
  | Gap | 0.281 | 0.251 (DOWN) | 0.324 | 0.489 (UP) |
  SmolLM2 IT uniformly decorrelates; gap goes DOWN. OPPOSITE of Qwen3. "Selective reshaping"
  and "IT amplifies gradient" WITHDRAWN. Correct statement: "In both pairs, IT lowers most
  correlations; gap direction is model-specific (Qwen3 up, SmolLM2 down); n=2 pairs."
- **SPACE-TOKEN CONFOUND — DISMISSED (zero-cost, 7 models, 2026-09-19):**
  R21 flagged parens/int_cast share space-token deletion, top-3 D pair in 5/8 models.
  Test: Jaccard similarity of token-level changes vs behavioral D-pair correlation.
  Grand r=0.094 (p=0.340, n=105). NULL. Tokenization similarity does NOT predict behavioral
  correlation. Critical control: add_zero/zero_plus have Jaccard=1.0 (IDENTICAL token changes)
  but behavioral rho=0.023-0.777 — wildly inconsistent. If tokenization drove D-coherence,
  this pair should always be top. It's not. parens/int_cast coincidentally share both
  tokenization similarity and high behavioral correlation, but the pattern doesn't generalize.
  One exception: Qwen3-1.7B shows r=0.523 (p=0.045) — model-specific, not general.
- **RETRACTED: COPY-VS-COMPUTE CONTROL (R17 Finding 1 — cell-schema mismatch):**
  `copy_vs_compute.py` mapped rows using the 20-cell `make_cells()` schema from
  `run_transfer_spectrum.py`, but all five kill JSONs use 40 cells from `kill_experiment*.py`
  where ALL cells are `x = <arithmetic expr>` (zero literal-copy cells). The "simple/complex"
  split was an arbitrary answer-digit partition, not a computation-type split. The cell labels
  in STATE.md were factually false. Additionally, the 12-cell gap sits within the random-12
  null distribution (48th-88th percentile), and the MWU tested amplitude difference not the
  rho gap. All copy-vs-compute claims WITHDRAWN. Process violation: entered STATE.md without
  review pass, violating R16 recommendations

**TRANSFER RHO (zero-cost reanalysis of kill experiment data, 2026-09-18):**
**Definition of d (R21 correction):** d[cell, perturbation] is the cosine distance between
two forward passes: the minimal surface (no header) and the verbose surface (`# Python code\n`
header, exactly 4 tokens in every condition), BOTH with the perturbation applied. The
perturbation is a CONTEXT in which the fixed header edit is measured, not the edit being
measured. The 24 perturbations are 24 contexts; d measures how much the 4-token header
moves the readout given a cell and surrounding context.
Split: Group A (inert appends + unperturbed baseline) vs Group D (meaning-preserving rewrites).
Transfer rho = Spearman(A-sensitivity, D-sensitivity) across 40 cells.
- **Raw transfer rho: 0.56-0.92** — real and driven by common fragility (all methods agree).
  Cells that are cross-surface fragile are fragile under every perturbation family.
- **Same-locus inert exchangeability:** A→B residual 0.83-0.97 in all 5 models under every
  covariate. Appends of the same locus are practically interchangeable.
- **RETRACTED: "2/5 models show type-specific transfer" (R19 review, 2026-09-18).**
  The B+C-PC1 partialled results (Qwen3 0.536, StableLM 0.565) were a forking-paths artifact:
  (a) StableLM's 0.57 flips to -0.42 under rank partialling; one outlier cell (10%2, distance
      0.59, 6 SD leverage) drives the sign. Refit-BCa CI [-0.07, 0.87] includes 0.
  (b) Qwen3's 0.54 drops to 0.00 under rank + PC1+PC2. Within-operator strata show Simpson
      mixture: %: -0.89, +: 0.73. Cross-type exceeds within-D reliability (0.32).
  (c) The 2/5 vs 3/5 split is COVARIATE COMPOSITION: C-PC1 gives all 5 models [0.30, 0.65];
      B-PC1 gives all 5 [-0.12, 0.42]. B+C-PC1 is a per-model blend; which models pass depends
      on the blend, not on genuine transfer differences.
  (d) Group A contains a_empty (the unperturbed baseline); rho(a_empty, rest of A) = 0.87-0.99.
      "Inert-append sensitivity" IS baseline fragility. Not a perturbation type.
  (e) Power 0.43 at rho=0.3, n=40. Cannot claim null either. Need ~113 cells.
  (f) "Model-dependent" was never tested (best Fisher z: p~0.06, no correction).
  All-PC1 anti-transfer was also a suppression artifact (retracted earlier).
- **NEW FINDING: Fragility has a second edit-locus dimension** (8-17% z-scored variance). PC2
  loads as before-insert vs at/after-assignment. One-PC partialling leaves this behind.
  After removing both dimensions (PC1+PC2): zero of five models shows residual A→D transfer.
- **Defensible statement:** Cross-surface sensitivity is dominated by cell-level fragility
  (PC1, 62-73% variance) plus a second factor tied to edit locus (PC2, 8-17%). After removing
  both, no model shows a residual A→D correlation distinguishable from zero at n=40. The
  experiment as currently designed cannot establish or refute type-specific transfer; it
  establishes that same-locus inert appends are exchangeable (ρ 0.83-0.97).
- **Method instability table (A→D partial, B+C-PC1 unless noted):**
  Qwen3:   0.54 / rank 0.36 / log 0.33 / PC1+2 0.22 / +operator 0.14 / rank-PC1+2 0.00
  SmolLM2: -0.18 / rank 0.21 / log -0.11 / PC1+2 0.13 / +operator 0.05 (robust null)
  StableLM: 0.57 / rank -0.42 / log -0.11 / PC1+2 0.17 / +operator 0.55 (unstable)
  Phi:      0.16 / rank 0.41 / log 0.02 / PC1+2 -0.15 / +operator 0.39
  CodeGen:  0.15 / rank 0.07 / log 0.08 / PC1+2 0.58 / +operator 0.27
  No model ordering is stable across methods.
- **RETRACTED: Row-permutation interaction test (2026-09-19, retracted same day by R20).**
  An interaction permutation test permuted whole rows of the distance matrix. The interaction
  SS is invariant under row relabelling (row means travel with rows, column/grand means unchanged).
  All 20 p-values were floating-point artifacts (null_std ~ 1e-20). The "5/5 additive", "type-blind",
  "monolithic fragility" claims are all withdrawn. Process failure: null_std should have been a stop
  signal. See experiments/results/fable_strategy_r20.md for full derivation.
- **CORRECTLY SPECIFIED TEST (R20 nested F + group-label permutation, log scale):**
  **4/5 models show significant cell × perturbation-group interaction.**
  | Model      | cell% | pert% | resid% | cell×group share | nested F p | group-perm p |
  |------------|-------|-------|--------|------------------|------------|--------------|
  | Qwen3      | 19.3  | 66.1  | 14.6   | 0.31             | 1e-19      | 0.0003       |
  | SmolLM2    | 39.5  | 42.1  | 18.5   | 0.32             | 8e-21      | 0.0007       |
  | StableLM   | 45.7  | 37.1  | 17.2   | 0.40             | 9e-36      | 0.0003       |
  | CodeGen    | 12.6  | 74.9  | 12.4   | 0.26             | 3e-11      | 0.009        |
  | Phi Q0     | 31.8  | 55.5  | 12.7   | 0.15             | 0.17       | 0.31         |
  Under H0 (no cell×group), expected share is 117/897 = 0.13. Four models at 0.26-0.40.
  Phi at 0.15 (null). Power of this test: 0.79 at 5% of residual SS, 0.99 at 10%.
- **Within-group coherence (pairwise Spearman of cell orderings):**
  A: 0.76-0.93, B: 0.64-0.95, C: 0.34-0.75, **D: 0.22-0.76**.
  Group D does NOT cohere: in Qwen3, two D perturbations agree at rho=0.22, LESS than D agrees
  with A (0.33). "Type D sensitivity" is not a property of cells; individual D perturbations are.
  The group-level interaction is not separable from edit locus in this design.
- **Multiplicative structure (R21: first-order Taylor, deflated):**
  Median angle sqrt(2d) is 0.04-0.28 rad (small-angle regime). To first order, d ~ gain(cell)
  x push(context), so log-additivity is expected from smoothness alone (Tukey F=60-490 confirms
  the first-order expansion, not a finding). The content is entirely in the 12-25% log-space
  residual, which is REAL deterministic structure (fp32 CPU, no noise term).
- **Defensible statement (post-R20):** On log scale, cell identity accounts for 13-46% and
  perturbation identity for 37-75% of cross-surface distance variance. The remaining 12-19% is
  deterministic cell × perturbation structure. In 4/5 models this is organized by perturbation
  group (cell × group share 0.26-0.40 vs 0.13 expected; group-label perm p <= 0.009). Phi is
  at chance. Inert appends (A, B) rank cells consistently (within-group rho 0.64-0.95);
  meaning-preserving rewrites (D) do not (0.22-0.76). Whether this is semantic type vs edit
  locus is not separable in the current design.
- **Gradient perm Z re-audit (R20 signal, resolved):** The gradient permutation test shuffles
  column labels WITHIN each row (np.random.shuffle per cell), NOT whole rows. This correctly
  randomizes perturbation-group assignment while preserving cell structure. The row-permutation
  defect does NOT apply. Gradient perm Z > 2 for all 7 tested models STANDS.
- **NESTED F + GROUP-LABEL PERM (all 8 models, log scale, 2026-09-19):**
  | Model        | Cell% | Pert% | Res% | CG share | F     | perm p  | A    | B    | C    | D    |
  |--------------|-------|-------|------|----------|-------|---------|------|------|------|------|
  | SmolLM2-135M | 44.2  | 44.1  | 11.7 | 0.270    | 2.47  | 0.009   | 0.92 | 0.81 | 0.76 | 0.71 |
  | SmolLM2-360M | 26.6  | 48.7  | 24.8 | 0.303    | 2.89  | 0.0006  | 0.67 | 0.61 | 0.33 | 0.18 |
  | SmolLM2-1.7B | 39.5  | 42.1  | 18.5 | 0.319    | 3.12  | 0.0002  | 0.86 | 0.88 | 0.75 | 0.44 |
  | Qwen3-0.6B-I | 21.6  | 55.9  | 22.5 | 0.338    | 3.40  | 0.0002  | 0.79 | 0.75 | 0.46 | 0.17 |
  | Qwen3-0.6B-B | —     | —     | —    | —        | —     | pending | 0.91 | —    | —    | 0.39 |
  | Qwen3-1.7B   | 19.3  | 66.1  | 14.6 | 0.312    | 3.02  | 0.0002  | 0.78 | 0.84 | 0.59 | 0.22 |
  | StableLM     | 45.7  | 37.1  | 17.2 | 0.396    | 4.36  | 0.0002  | 0.93 | 0.95 | 0.51 | 0.57 |
  | CodeGen      | 12.6  | 74.9  | 12.4 | 0.255    | 2.29  | 0.0072  | 0.76 | 0.72 | 0.34 | 0.37 |
  | Phi Q0       | 31.8  | 55.5  | 12.7 | 0.145    | 1.14  | 0.30    | 0.81 | 0.64 | 0.62 | 0.76 |
  7/8 models: cell × group interaction significant (perm p < 0.01). Only Phi Q0 null.
  **R21 CRITICAL QUALIFIER:** The interaction is present among inert contexts alone:
  - AB vs C (D removed): p ≤ 0.004 in 7/8 models. Rewrites NOT NEEDED for interaction.
  - A vs B: null in 8/8 (negative control passing — two inert groups exchangeable).
  - Comment-vs-statement within A+B: significant in 7/8 (p=0.0006-0.07).
  - Replacing ABCD with comment/stmt/C/D lowers p in ALL 8 models (Phi: 0.30 → 0.002).
  The declared groups are not the axes along which models organize the residual.
  The interaction reflects CONTEXT FEATURES (locus, comment-vs-statement, token adjacency),
  demonstrably including features that have nothing to do with meaning preservation.
  **ATTENUATION RATIO (R21 central result):** observed A-D cross rho / sqrt(within-A × within-D)
  = 0.79-1.01 across 8 models. A single common cell factor with column-specific noise
  reproduces the A-D "gap" to within 0-20%. Inert appends predict each rewrite's cell ordering
  as well as other rewrites do (8/8 models). "Not predictable from inert" WITHDRAWN;
  "less reliable" is the supported phrase.
  **Phi Q1 (admissible readout):** Cell%=52.7, Pert%=33.8, Res%=13.5, CG share=0.275,
  perm p=0.001, A=0.90 B=0.85 C=0.76 D=0.71. SIGNIFICANT. 7/8 models significant under
  Q0; Phi additionally under model-specific Q1 (post-hoc readout; Q1 inadmissible on Qwen3).
  Phi Q0 null is a readout artifact.
  Within-group coherence (A, B columns): pairwise Spearman of cell orderings 0.61-0.95.
  Within-D coherence: 0.17-0.76 — GROUP D DOES NOT COHERE as a type.
- **SCALE-DEPENDENT D-COHERENCE (new finding):**
  SmolLM2 family: 135M D=0.71, 360M D=0.18, 1.7B D=0.44.
  At smallest scale (52% acc): all D perturbations are interchangeable (model can't distinguish).
  At medium scale (90% acc): D perturbations become HIGHLY DISTINCT (individual processing).
  At largest scale (95% acc): partial reconsolidation.
  Qwen3 (instruct): 0.6B D=0.17, 1.7B D=0.22 — consistently low.
  **Qwen3-0.6B-Base: D=0.39 — 2.3x HIGHER than instruct.** Instruction tuning CREATES the
  low D-coherence, not scale. Base model treats rewrites more interchangeably.
  Implication: models develop perturbation-specific processing as they scale up. The distinction
  between "equivalent rewrites" is a LEARNED property, not inherent. Instruction tuning
  accelerates this differentiation (Base 0.39 → Instruct 0.17 at same scale).
- **D-COHERENCE DECOMPOSITION (SmolLM2 scale ladder, zero-compute reanalysis):**
  *R21 CORRECTIONS: all rhos carry CIs (bootstrap n=40, B=2000). No mechanism assertions.*
  Mean D-coherence with CIs: 135M: 0.71 [0.55, 0.81], 360M: 0.18 [0.05, 0.31], 1.7B: 0.44 [0.27, 0.58].
  Scale dependence of mean D-coherence is REAL (CIs don't overlap).
  - **int_cast is the most isolated D context at 360M** (85% of cell bootstraps), uncorrelated
    (NOT anti-correlated — CIs include 0 and reach +0.15) with other D perts. Divergence driven
    by `-` and `%` cells: eta²(operator)=0.44 for int_cast residual. Effective n = 5 operator strata.
  - **cond_true is the most isolated at 1.7B in SmolLM2 only** (97% of bootstraps). Does NOT
    replicate in any other model. Driven by `*1` and `//1` identity cells (eta²(operator)=0.64).
  - **At 1.7B:** {int_cast, add_zero, zero_plus, parens} cluster (rho 0.56-0.78), {temp_var}
    and {cond_true} isolated. The first cluster has no arithmetic reading (parens is in it).
  - **parens and int_cast** both delete the space token before the first operand. They are a
    top-3 D pair in 5/8 models. Tokenization confound is untested.
  - **Cross-scale rank stability near zero (ALL groups, not D-specific):** Cell orderings
    reorganize with scale across all 24 columns: A mean rho=0.08, B=0.03, C=-0.30, D=0.11
    (135M→360M). Row-mean fragility also near-zero cross-scale. Group C sign flip (-0.30) is
    the strangest number and is unexplained. "Which cells are sensitive" is a checkpoint property.
  - **Cross-group coherence (A-D):** 0.80 (135M) -> 0.36 (360M) -> 0.58 (1.7B). At 135M the
    model doesn't distinguish groups; at 360M groups become very distinct; partial re-merge at 1.7B.
  - **Phi's residual is NOT unorganized** — it is organized along comment-vs-statement (p=0.002)
    and C-vs-D (p=0.011), just not along ABCD. Its D columns' nearest neighbours are inert contexts.
    "No group-specific processing" WITHDRAWN.
- **LOCUS CONFOUND ANALYSIS (all 8 models, zero-compute reanalysis):**
  *Is the cell × group interaction driven by edit POSITION or semantic TYPE?*
  Decompose the A-D coherence drop into locus effect (A-B minus A-C) and type effect (A-C minus A-D).
  | Model        | Locus | Type  | Total | Dominant   |
  |--------------|-------|-------|-------|------------|
  | SmolLM2-135M | 0.065 | -0.004| 0.061 | Neither    |
  | SmolLM2-360M | 0.437 | -0.146| 0.291 | LOCUS      |
  | SmolLM2-1.7B | 0.190 | 0.089 | 0.278 | Locus      |
  | Qwen3-0.6B   | 0.265 | 0.202 | 0.467 | Mixed      |
  | Qwen3-1.7B   | 0.200 | 0.279 | 0.479 | **TYPE**   |
  | StableLM-1.6B| 0.349 | -0.041| 0.309 | LOCUS      |
  | CodeGen-350M | 0.377 | -0.090| 0.287 | LOCUS      |
  | Phi-1.5-Q0   | 0.030 | -0.042| -0.013| Neither    |
  **5/8 models have negative type effect:** D rewrites are MORE coherent with A (post-assignment
  appends) than C (pre-assignment insertions) are. The cell × group interaction is primarily
  driven by WHERE the edit goes, not WHAT it does. Only Qwen3-1.7B shows genuine semantic
  type dominance (type 0.28 > locus 0.20).
  **Mean distance pattern:**
  SmolLM2/StableLM/CodeGen: A ~= B > D > C (before-assignment is LEAST disruptive).
  Qwen3: D > C ~= A ~= B (rewrites are MOST disruptive — architecture-specific).
  **B/C matched-pair test (same operation, different position):**
  Matched-pair rho (e.g., a_comment vs c_comment_before) vs unmatched A-C cross rho:
  difference < 0.06 for 7/8 models. Matching by operation provides NO additional predictive power.
  Cell ordering is determined by position, not by what specific inert operation is performed.
  **CONCLUSION:** The 7/8 cell × group interaction is REAL but primarily reflects edit POSITION,
  not the semantic content of the edit. This is TOTAL for A/B/C groups.
  **What IS NOT confounded (genuine type effects):**
  - Within-D structure (arithmetic vs control-flow clusters) — all D perts at the same locus
  - Scale-dependent D-coherence — position is constant across scale, so it can't explain changes
  - Qwen3-1.7B type effect (0.28 > locus 0.20) — small but genuine
  - D-coherence < C-coherence in 5/8 models — rewrites produce more heterogeneous cell orderings
    than before-assignment insertions, independent of cross-group position confound
  - Per-D-pert vs A-mean cell ordering: some D perts DIVERGE from the general cell-fragility
    hierarchy. zero_plus has r=-0.07 (ANTI-correlated) at Qwen3-0.6B and r=0.10 at SmolLM2-360M.
    These rewrites probe pathways ORTHOGONAL to append-based sensitivity. The interaction is genuine.
- **WITHIN-D DEEP ANALYSIS (all 8 models, zero-compute):**
  *Which D perturbations are most disruptive? Is there a universal ordering?*
  - **Kendall W = 0.152:** NO universal D-perturbation ordering. Which rewrite is most disruptive
    is completely model-dependent.
  - **Architecture preserves ranking:** Qwen3-0.6B vs 1.7B r=0.943 (p=0.005). Same architecture
    at different scales ranks rewrites the same way: temp_var > cond_true > zero_plus > add_zero.
  - **Scale CAN reverse ranking:** SmolLM2-135M vs 1.7B r=-0.771. The disruption hierarchy
    inverts with scale (parens most disruptive at 135M; add_zero most at 1.7B).
  - **Structural complexity does NOT predict disruption** (all Spearman rho NS). Adding more tokens
    doesn't make a rewrite more disruptive — the model's processing matters, not edit size.
  - **Cell fragility generalizes (multiplicative):** D-mean vs A-mean cell-level rho = 0.56-0.98.
    Cells that are sensitive to appends are also sensitive to rewrites. BUT:
  - **Some D perts are ORTHOGONAL to cell fragility:** Per-D-pert vs A-mean ranges from -0.07
    (zero_plus, Qwen3-0.6B) to 0.93 (cond_true, SmolLM2-135M). Perturbations that diverge
    most from A-ordering: zero_plus (r=0.10 at 360M), cond_true (r=0.11 at 360M), int_cast
    (r=0.22 at Qwen3-0.6B). These rewrites activate cell-specific pathways independent of
    general model sensitivity — they're the source of the genuine interaction term.
- **RESIDUAL DIRECTION STRUCTURE TEST (permutation-validated, 4 models):**
  Within-D log-additive residual: pairwise column correlations range -0.55 to +0.57.
  The mean (~-0.18) is explained by the row-sum-to-zero constraint (expected = -0.20).
  But the SPREAD of correlations exceeds the constraint-induced expectation:
  | Model        | Obs SD | Null SD | Z    | perm p | Verdict    |
  |--------------|--------|---------|------|--------|------------|
  | SmolLM2-360M | 0.338  | 0.167   | 5.82 | 0.0005 | SIGNIFICANT|
  | SmolLM2-1.7B | 0.280  | 0.157   | 3.85 | 0.001  | SIGNIFICANT|
  | Qwen3-0.6B   | 0.166  | 0.154   | 0.38 | 0.33   | Null       |
  | Qwen3-1.7B   | 0.247  | 0.157   | 2.91 | 0.007  | SIGNIFICANT|
  **In 3/4 models, D perturbations push logits in genuinely DIFFERENT DIRECTIONS, not just
  different amounts.** This is the cleanest evidence for perturbation-specific processing:
  it's within-D (same code position), in the residual (controls for multiplicative cell
  fragility x perturbation magnitude), and permutation-validated against the constraint null.
  Specific antagonistic pairs at SmolLM2-360M: int_cast vs add_zero (-0.55), parens vs
  zero_plus (-0.53). The {parens, int_cast} cluster opposes the {zero_plus, cond_true} cluster.
- **DIRECTION STRUCTURE CORRELATES WITH TRANSFER (algebraically expected, NOT a mechanism):**
  **R21 TAUTOLOGY WARNING:** rho_raw(i,j) is approximately a monotone function of r_resid(i,j)
  when var(cell) dominates: both measure how much two perturbation columns share cell-specific
  structure. The meta-r values (0.51-0.83) are algebraically expected from the decomposition,
  not an independent finding. Values below 1.0 reflect Spearman-vs-Pearson mismatch and
  variation in per-perturbation residual variance. This is a consistency check, not a mechanism.
  Meta-correlation between residual pairwise rho and raw pairwise within-D transfer:
  | Model        | Meta r | Meta p |
  |--------------|--------|--------|
  | Qwen3-0.6B   | 0.829  | 0.0001 |
  | SmolLM2-360M | 0.814  | 0.0002 |
  | Qwen3-1.7B   | 0.711  | 0.003  |
  | CodeGen-350M | 0.536  | 0.040  |
  | SmolLM2-1.7B | 0.514  | 0.050  |
  | StableLM-1.6B| 0.332  | 0.227  |
  **In 5/6 models, the residual direction correlates with pairwise transfer.** Pairs with
  anti-correlated residuals have low raw transfer rho. This is algebraically expected from
  the log-additive decomposition (see R21 tautology note above), NOT an independent mechanism.
  The observation is consistent with the decomposition but carries no additional content.
  Consistently worst-transferring D pairs (cross-model mean rho):
  - int_cast-cond_true: 0.146 (arithmetic vs control-flow divide)
  - zero_plus-cond_true: 0.208
  - int_cast-zero_plus: 0.243
  Best-transferring: parens-int_cast (0.552), parens-add_zero (0.428).
  Cell-fragility correlation: zero_plus consistently lowest (0.35-0.67). These are
  perturbations that LEAST follow the general sensitivity pattern — they activate
  perturbation-specific processing pathways independent of overall cell fragility.
  **R21 caveat:** inert appends predict each rewrite's cell ordering as well as other rewrites
  do (leave-one-D-out, 8/8 models). The attenuation ratio (0.79-1.01) shows a single common
  cell factor reproduces the A-D gap. The direction divergence finding (above) is real in the
  RESIDUAL (3/4 perm-validated), but the residual is 12-25% of total variance. The practical
  question is whether that 12-25% matters for fine-tuning/alignment robustness.
  Tokenization confound: parens and int_cast both delete the space token before the operand;
  they are a top-3 D pair in 5/8 models. Space-restoration test needed to distinguish
  tokenization artifact from semantic processing.
  **Surviving implication:** Sensitivity is predominantly about cell fragility × context magnitude
  (multiplicative). The residual direction structure is real but small, and its organization
  by surface features (locus, comment-vs-statement, token adjacency) rather than semantic
  type is the currently supported interpretation.
- **DIRECTION STRUCTURE IS UNIVERSAL ACROSS GROUPS (not D-specific):**
  Tested direction divergence (residual pairwise-rho SD vs row-permutation null) for each group:
  | Model        | A Z   | B Z   | C Z   | D Z   |
  |--------------|-------|-------|-------|-------|
  | SmolLM2-360M | 13.80 | 6.92  | 2.17  | 5.82  |
  | SmolLM2-1.7B | 9.45  | 4.15  | 1.50  | 3.85  |
  | Qwen3-1.7B   | 8.26  | 5.58  | 3.59  | 2.91  |
  **Groups A and B show STRONGER direction divergence than D.** Even different inert appends
  push logits in genuinely different directions. Direction divergence is a general property of
  how contexts modulate header sensitivity, not a signature of meaning-preserving rewrites.
  Group C (before-assignment insertions) shows weakest structure, consistent with the locus
  confound: positional adjacency to the header is the primary axis, not semantic content.
  This supports R21's reframe: residual structure tracks surface features (locus, comment-vs-
  statement, token adjacency), not whether the context is a meaning-preserving rewrite.
- **CROSS-MODEL DIRECTION PREDICTION FAILS (underpowered negative result):**
  **R21 POWER NOTE:** All comparisons at n=15 perturbation pairs. Critical rho ~0.52 at alpha=0.05.
  Leave-one-out: cross-mean 0.01-0.26, all NS. Self-prediction: 0.33-0.83 (partly NS: CodeGen 0.33).
  Within-family: SmolLM2-360M vs 1.7B = -0.37 (NS), Qwen3-0.6B vs 1.7B = -0.30 (NS).
  At n=15, we cannot distinguish "direction structure is model-specific" from "insufficient power."
  The "double non-transfer" framing is unsupported: self-prediction is partly NS, and cross-prediction
  NS results could reflect low power rather than genuine model specificity.
- **CROSS-MODEL FRAGILITY NOT DETECTABLY CORRELATED (zero-cost, 9 model-readouts, underpowered):**
  **R21 POWER NOTE:** At n=40, power to detect rho=0.3 is ~50%. 36 pairwise comparisons with no
  multiple-testing correction; expect 1.8 false positives at alpha=0.05. "Completely model-specific"
  is not testable at this sample size. Two significant negative pairs are reported below.
  Per-cell fragility rank-correlated between all model pairs:
  - **Within-family rho near ZERO:** Qwen3-0.6B vs 1.7B: rho=0.032 (p=0.84).
    SmolLM2-135M vs 360M: -0.123 (p=0.45). SmolLM2-360M vs 1.7B: 0.065 (p=0.69). All NS.
  - **Cross-family near ZERO:** Qwen3-1.7B vs SmolLM2-1.7B: 0.125 (p=0.44).
  - **Two significant NEGATIVE pairs:** Qwen3-0.6B vs SmolLM2-1.7B: -0.494. SmolLM2-1.7B vs
    Phi-Q0: -0.491. Cells most fragile in one are LEAST in the other.
  - **Consensus std (4.5) = random (4.7).** No cell is universally fragile or robust.
  - **Extreme reorganization:** Cell `1**5` rank 0 in Qwen3-1.7B, rank 35 in Qwen3-0.6B.
  **R21-corrected implication:** At n=40, no consistent cross-model fragility pattern is
  detectable (power ~50% at rho=0.3). The claim "vulnerability location is model-specific"
  is a power limitation, not a demonstrated biological fact. Needs n >= 200 diverse cells to
  be testable.
- **COMPLETE SCALE LADDER RESIDUAL ANALYSIS (2026-09-19):**
  | Model | Gap | Resid gap | Resid CI | PC1% | Gates |
  |---|---|---|---|---|---|
  | SmolLM2-135M | 0.121 | 0.105 | [-0.015, 0.251] | 69.6% | FAIL (CI incl 0) |
  | SmolLM2-360M | 0.318 | 0.384 | [0.205, 0.480] | 90.5% | ALL PASS |
  | SmolLM2-1.7B | 0.281 | 0.310 | [0.091, 0.533] | 78.2% | ALL PASS |
  | Qwen3-0.6B | 0.489 | 0.309 | [0.115, 0.487] | 53.1% | ALL PASS |
  | Qwen3-1.7B | 0.452 | 0.230 | [0.077, 0.430] | 51.0% | ALL PASS |
  **SmolLM2-360M residual (0.384) EXCEEDS raw gap (0.318).**
  **R21 SUPPRESSOR WARNING:** This occurs at the highest PC1 share (90.5%). When partialling out
  90.5% of variance, the residual gap computed on the remaining 9.5% can be inflated by
  suppressor artifacts (PC1 shares variance with both signal and noise). SmolLM2-135M (PC1=69.6%)
  contradicts the suppression story: lower PC1 but fails (CI incl 0). The pattern (residual >
  raw only at highest PC1) is consistent with a suppressor artifact, not evidence that fragility
  masks type-specificity. SmolLM2 raw gap CIs overlap (360M [0.207, 0.431] vs 1.7B [0.197, 0.385]);
  "non-monotonic scaling" is not significant at n=3 scale points.
- **D-GROUP INTERNAL STRUCTURE IS COMPLETELY MODEL-SPECIFIC (8 models, 2026-09-19):**
  Within-D rho (mean of 15 pairs) ranges from 0.169 (Qwen3-0.6B) to 0.712 (SmolLM2-135M).
  Cross-model consistency of D-pair rho vectors: 0/28 model pairs significant after correction.
  Median cross-model rho = +0.03, range = [-0.46, +0.52]. D-group internal structure is random
  across models. No D-perturbation is consistently "most anomalous":
  | Model | Most anomalous D-pert | Mean rho with others |
  |---|---|---|
  | Qwen3-1.7B | temp_var | -0.005 |
  | SmolLM2-1.7B | cond_true | +0.232 |
  | StableLM-2-1.6B | zero_plus | +0.411 |
  | CodeGen-350M | cond_true | +0.197 |
  | SmolLM2-360M | int_cast | -0.002 |
  | Qwen3-0.6B | int_cast | +0.086 |
  | Phi-1.5-Q1 | add_zero | +0.674 (ALL cohere) |
  | SmolLM2-135M | zero_plus | +0.662 (ALL cohere) |
  D-coherence anticorrelates with model capability: 135M and Phi-Q1 (high coherence) treat
  all rewrites similarly; Qwen3 and SmolLM2-360M (low coherence) differentiate rewrites.
  **D-coherence does NOT predict residual gap** (Spearman rho=+0.05, p=0.91, n=8).
  **Raw gap does NOT predict residual gap** (rho=-0.11, p=0.80). Fragility partialling
  completely reshuffles the model ordering.
  **INSTRUCT EFFECT ON D-COHERENCE (Qwen3-0.6B, same tokenizer):**
  Base within-D rho = 0.394 vs Instruct 0.169 (2.3x). Instruction tuning makes D
  perturbations LESS interchangeable. parens<->int_cast coupling: Base 0.729 vs Instruct
  0.183. This is NOT tokenization (same tokenizer) — instruction tuning reorganizes how
  the model processes meaning-preserving rewrites.
  **Combined with cross-model fragility (above):** both WHICH cells are fragile and HOW
  rewrites interact with cells are not detectably correlated across models (all underpowered
  at n=40). The sensitivity fingerprint appears unique to each model but confirmation
  requires n >= 200 cells (see R21 section 3.3).
- **Next steps (post-R21 design):**
  R21 specifies: NO NEW CLAIMS until a properly designed next experiment exists.
  (1) **Operator-balanced cells:** n >= 60, no identity expressions (`x*1`, `x//1`, `1**5`),
      balanced operator strata (12+ per class). Current effective n = 5 operator strata.
  (2) **Space-token restoration:** `x = ( e )` and `x = int( e )` (spaced variants, ~10 min/model).
      Tests whether parens/int_cast coupling is tokenization artifact.
  (3) **Before-locus rewrite:** `_f = lambda: e; x = _f()` — rewrite placed BEFORE assignment.
      Should behave like C if locus is the mechanism.
  (4) **Comment-vs-statement within D:** `x = e  # ok` vs `x = e; pass`. If this separates D,
      "type" has no residual role.
  (5) **k >= 3 surface instances** per context for proper error term.
  (6) **Locus-crossed design** for A/B/C confound separation.
  (7) **Qwen3-0.6B-Base** ~~to resolve instruct confound~~ — **RESOLVED (2026-09-19).** Gap 0.087
      vs instruct 0.489 (like-for-like 1.5x; R23 corrected from 5.6x definition artifact).
  (8) **Base model for other families** — SmolLM2, StableLM have no instruct variant tested.
      Only Qwen3 has the Base/Instruct comparison. Need ≥1 more family to generalize.
  Estimated cost: one CPU afternoon per model at 60+ cells.
- **BALANCED CELL V3 RESULTS (69 cells, no identity, operator-balanced, 2026-09-19):**
  | Model | Gap | CI 95% | Perm Z | Residual Gap | DD | AD |
  |---|---|---|---|---|---|---|
  | Qwen3-1.7B | **0.500** | [0.411, 0.588] | 8.7 | 0.462 (92%) | 0.337 | 0.337 |
  | SmolLM2-1.7B | **0.275** | [0.202, 0.362] | 4.3 | 0.389 (141%) | 0.531 | 0.669 |
  DD < AD in both. Gap survives on balanced cells in both models. SmolLM2 residual HIGHER
  than raw (operator strata absorb some gap; partialling increases A-D difference).
  Qwen3 residual 92% of raw (operator effect is small). SmolLM2 eta²(operator)=0.687.
  **R21 precondition MET.** Cross-model claims now admissible on balanced cells.
  **R21 falsifiable claim:** each context column = common cell factor + column-specific residual;
  the residual is organized by surface features (locus, comment-vs-statement, token adjacency)
  and NOT by whether the context is a meaning-preserving rewrite.
- **MEANING-CHANGE TRANSFER TEST (2 models, Group E, fp32 CPU, 2026-09-18; R24-CORRECTED):**
  Cell-level Spearman correlation between perturbation-group mean sensitivity profiles.
  **R24 CORRECTIONS (16/40 cells are identity expressions — dominant confound):**
  On non-identity cells only (n=24):
  | Transfer | Qwen3-1.7B | SmolLM2-1.7B |
  |---|---|---|
  | Inert (A+B) → E | 0.066 (NS) | 0.143 (NS) |
  | D (rewrites) → E | **0.455** (p=0.026) | 0.393 (p=0.057) |
  | add_zero+cond_true → E | **0.708** (p<0.001) | **0.438** (p=0.032) |
  | Other 4 D members → E | 0.123 (NS) | 0.277 (NS) |
  D→E > Inert→E holds (0.46/0.39 vs 0.07/0.14) but is driven by 2 of 6 D members
  (add_zero, cond_true) that share token-prefix form with E members (e_plus_one, e_minus_one).
  **RETRACTED:** SmolLM2 Inert→E significance (identity-cell artifact).
  **RETRACTED:** "Whether inert predicts E depends on model" (Fisher z=1.29, p=0.195, NS).
  **DOWNGRADED:** "Additive transfers, multiplicative doesn't" (confounded with prefix form).
  **WITHDRAWN:** "D captures proximity to genuine behavioral vulnerability" (no behavioral
  quantity measured; Qwen3 full-logit cosine cell-profile doesn't correlate with 11-bin at all).
  **TOKEN-PREFIX NULL:** d_add_zero → e_plus_one = 0.861 (SmolLM2): differ by one token.
  Prefix-breaking D members (temp_var, parens) → E ≈ 0 on non-identity cells.
  E-group partial coherence after removing form twins ≈ 0.
  **STRONGEST VERSION (R24):** Cross-surface logit-geometry sensitivity transfers between
  perturbations sharing a token prefix with the cell, regardless of meaning preservation.
  No evidence yet of a meaning-change factor beyond shared token form.
- **PREFIX-FORM CROSSOVER TEST (Qwen3-1.7B, 2x2 design, fp32 CPU, 2026-09-19):**
  2x2: D members (add_zero=`e+0`, zero_plus=`0+e`) x E members (plus_one=`e+1`, one_plus=`1+e`).
  Non-identity cells (n=24):
  | D \ E | plus_one (e+1) | one_plus (1+e) |
  |---|---|---|
  | add_zero (e+0) | **0.634** | -0.301 |
  | zero_plus (0+e) | 0.428 | 0.063 |
  Diagonal mean: 0.349, off-diagonal mean: 0.063. E-E rho: -0.143.
  **R25 REANALYSIS: Column main effect, not diagonal (see R25 below).**
  zero_plus→plus_one (0.428, diff prefix) > zero_plus→one_plus (0.063, same prefix) — breaks
  clean prefix prediction. Operator strata reverse Qwen3 diagonal within-stratum.
  **SmolLM2 replication (n=24):** add_zero→plus_one **0.676**, add_zero→one_plus -0.405,
  zero_plus→plus_one 0.390, zero_plus→one_plus -0.159. E-E rho: **-0.470 (p=0.021, SIG)**.
  All four signs replicate Qwen3. E-E anti-correlation significant in SmolLM2.
  **VARIANCE ANALYSIS (zero-cost):** add_zero has 3.5-4.3x more cross-cell variance than
  zero_plus. Full D×E table: d_temp_var REVERSES the column effect in both models.
  Mean D → plus_one: +0.311 (p=0.139, NS). Mean D → one_plus: -0.277 (p=0.191, NS).
  **NEITHER PREFIX NOR GRADIENT HYPOTHESIS SURVIVES** aggregation over all D members.
  **FULL-LOGIT-ONLY FINDINGS (RETRACTED AS GENERAL CLAIMS):**
  (1) E-E anti-correlation (full-logit only: -0.143 Qwen3, -0.470 SmolLM2). REVERSED on 11-bin.
  (2) add_zero→one_plus anti-transfer (-0.301/-0.405): full-logit only. On 11-bin: +0.531.
  (3) Member-specificity: full-logit only. On 11-bin, all four entries positive and significant.
- **11-BIN CROSSOVER REANALYSIS (Qwen3, 160 fwd passes, 2026-09-19):**
  | Cell | Full-logit | 11-bin |
  |---|---|---|
  | add_zero→plus_one | +0.634 | +0.610 |
  | add_zero→one_plus | -0.301 | **+0.531** |
  | zero_plus→plus_one | +0.428 | **+0.710** |
  | zero_plus→one_plus | +0.063 | **+0.423** |
  | **E-E rho** | -0.143 | **+0.758 (p<0.001)** |
  Diag mean: FL 0.349, 11-bin 0.517. Off-diag mean: FL 0.063, 11-bin 0.620.
  **TOKEN-PREFIX NULL REFUTED on answer distribution.** All additive perturbations transfer
  regardless of operand position on 11-bin. E-E flips from anti-correlated to r=+0.76.
  Off-diagonal > diagonal on 11-bin: gradient hypothesis wins over prefix hypothesis.
  **REFRAMING:** The gap measures D-DIVERSITY (how differently each D member modulates
  a common cell-sensitivity factor), not D-DISCONNECTION from E. D→E transfer is REAL
  on 11-bin (Qwen3).
- **11-BIN CROSSOVER REPLICATION (SmolLM2, 160 fwd passes, 2026-09-19):**
  | Cell | Full-logit | 11-bin |
  |---|---|---|
  | add_zero→plus_one | +0.676 | **+0.564** |
  | add_zero→one_plus | -0.405 | +0.089 (NS) |
  | zero_plus→plus_one | +0.390 | +0.170 (NS) |
  | zero_plus→one_plus | -0.159 | +0.129 (NS) |
  | **E-E rho** | -0.470 (p=0.021) | +0.124 (NS, p=0.563) |
  Diag mean: FL 0.258, 11-bin 0.347. Off-diag mean: FL -0.007, 11-bin 0.130.
  **PARTIAL REPLICATION:**
  (a) Anti-correlation artifact: CONFIRMED. E-E flips sign in both models (SmolLM2: -0.47→+0.12).
  (b) Broad positive transfer: QWEN3-SPECIFIC. SmolLM2 has 1/4 entries significant on 11-bin
      vs Qwen3's 4/4. Only add_zero→plus_one survives in both models.
  (c) add_zero is the "privileged" D member: highest variance (3.5-4.3x), only D member
      whose cross-perturbation transfer survives both readouts in both models.
  (d) Reframing qualified: D-diversity is real, but D→E transfer strength on 11-bin
      is model-dependent. The gap measures D-diversity; whether that diversity implies
      D→E transfer on answer distributions depends on the model.
- **FABLE R25 ADVERSARIAL REVIEW (2026-09-19) — MAJOR CORRECTIONS:**
  The 2×2 is a COLUMN main effect, not a diagonal/prefix effect.
  | Contrast (n=24) | Qwen3 | SmolLM2 |
  |---|---|---|
  | Column (plus_one − one_plus) | **+0.65** [+0.21, +1.01] | **+0.82** [+0.19, +1.28] |
  | Row (add_zero − zero_plus) | −0.08 | +0.02 |
  | Interaction (diag − offdiag) | +0.29 [−0.04, +0.61], p=0.085 | +0.27 [−0.04, +0.60], p=0.052 |
  Only the column CI excludes zero. Diagonal interaction NS in both models.
  **one_plus is a column anomaly:** anti-correlates with 17/18 inerts in Qwen3 (mean rho -0.44).
  `1 + e` is a GARDEN PATH for `* // %` cells (precedence override). Not equivalent to `e + 1`.
  **Operator strata reverse Qwen3 diagonal:** within-stratum ad→plu = −0.23. The 0.634 is
  between-operator structure on an unbalanced set. SmolLM2 ad→plu survives within-operator
  (+0.52) but interaction collapses to 0.04.
  **RETRACTIONS (R25):**
  - "Prefix form is partial mediator" → DOWNGRADE to "one append-twin transfers"
  - "E-E independence" → RETRACT (Qwen3 CI spans zero; SmolLM2 significantly negative)
  - "Meaning change determines which cells affected" → WITHDRAW (no meaning-preserving
    late twin measured; no 11-bin E readout on old cells)
  - "Token-prefix sharing" → NARROW to late-edit locality
  **MOST HONEST READING (R25):** Edit locality relative to `e` plus precedence artifact in
  `1 + e`. Append-after-e edits (add_zero, plus_one, minus_one, cond_true, times_two) cohere
  at rho 0.43-0.66. Prepend/wrap edits do not. Meaning change has NO demonstrated role —
  no meaning-preserving late twin (e-0, e*1) was ever measured.
  **LATE-TWIN DISAMBIGUATION (Qwen3, balanced v3, 828 fwd passes, 2026-09-19):**
  6 append-after-e perts: D (e+0, e-0) vs E (e+1, e-1, e+2, e+3). Both readouts.
  | Readout | D-D mean | D-E mean | E-E mean | D-D minus D-E |
  |---|---|---|---|---|
  | Full-logit | **+0.658** | +0.424 | +0.405 | **+0.234** |
  | 11-bin | +0.591 | +0.644 | **+0.781** | **-0.053** |
  **MEANING IS DEAD ON 11-BIN.** On the answer distribution, D-D < D-E < E-E. Meaning-changing
  perturbations are MORE internally coherent than meaning-preserving ones. e+0 correlates
  MORE with E members (0.73-0.83 on 11-bin) than with e-0 (0.59). All late-edit perts load
  on a common "surface sensitivity" factor. Meaning changes AMPLIFY this factor because
  answer shifts increase prediction uncertainty (E-E > D-E > D-D on 11-bin).
  Full-logit shows D-D > D-E because formatting geometry encodes operator identity (+/-)
  separate from meaning. This is the known readout disconnect.
  **Full-logit bootstrap CI on D-D minus D-E: [+0.082, +0.382]** (excludes zero, N=2000
  resamples). Meaning distinction is real on formatting geometry. 11-bin bootstrap pending.
  **SmolLM2 REPLICATION (828 fwd passes, 2026-09-19):**
  | | Qwen3 FL | Qwen3 11b | SmolLM2 FL | SmolLM2 11b |
  |---|---|---|---|---|
  | D-D | +0.658 | +0.591 | +0.704 | +0.365 |
  | D-E | +0.424 | +0.644 | +0.468 | +0.349 |
  | E-E | +0.405 | +0.781 | +0.541 | +0.415 |
  | D-D - D-E | +0.234 | -0.053 | +0.236 | +0.015 |
  | FL CI | [+0.082, +0.382] | — | [+0.106, +0.357] | — |
  | 11b CI | pending | — | [-0.171, +0.200] | — |
  **3-MODEL REPLICATION (Qwen3, SmolLM2, StableLM-2, 2026-09-19):**
  | Model | FL D-D-DE | FL CI | 11b D-D-DE | 11b CI |
  |---|---|---|---|---|
  | Qwen3-1.7B | +0.234 | [+0.082,+0.382]* | -0.053 | pending |
  | SmolLM2-1.7B | +0.236 | [+0.106,+0.357]* | +0.015 | [-0.171,+0.200] |
  | StableLM-2-1.6B | **-0.017** | [-0.170,+0.129] | -0.044 | [-0.201,+0.122] |
  *CI excludes zero. StableLM shows edit locality on BOTH readouts.
  **11-bin: 3/3 models show no meaning effect (CI spans zero where measured).**
  **Full-logit: 2/3 models show meaning matters; StableLM does not.** The FL meaning
  distinction is model-specific, likely reflecting operator token (+/-) encoding that varies
  by architecture. StableLM doesn't encode this distinction in formatting geometry.
  **ADVERSARIAL REVIEW (self, 2026-09-19):** Strongest attack: D has only 1 D-D pair.
  Correct claim: "fail to detect meaning effect" (null), not "meaning is dead" (positive).
  **EXPANDED-D LATE-TWIN (3 models, 2026-09-19):** 4 D perts (e+0, e-0, e*1, e**1) +
  6 E perts (e+1, e-1, e+2, e+3, e*2, e**2) = 6 D-D, 24 D-E, 15 E-E pairs.
  | Model | FL D-D-DE | FL CI | 11b D-D-DE | 11b CI |
  |---|---|---|---|---|
  | Qwen3-1.7B | +0.048 | [-0.050, +0.148] | **-0.111** | **[-0.186, -0.033]*** |
  | SmolLM2-1.7B | +0.180 | [+0.074, +0.285]* | +0.085 | [-0.017, +0.189] |
  | StableLM-2-1.6B | +0.095 | [+0.009, +0.194]* | +0.062 | [-0.010, +0.140] |
  | SmolLM2-360M | -0.031 | [-0.076, +0.009] | -0.035 | [-0.097, +0.024] |
  *CI excludes zero. **Qwen3 11-bin excludes zero NEGATIVE (REVERSED). 360M both REVERSED.**
  **11-bin: 0/4 positive meaning. 2/4 reversed (Qwen3, 360M).**
  **FL: 1/4 clear (SmolLM2-1.7B), 1/4 marginal (StableLM), 2/4 not detected (Qwen3, 360M).**
  Qwen3 FL REVERSAL from original late-twin: original CI [+0.082, +0.382], now [-0.050, +0.148].
  Adding e*1/e**1 to D broke additive-family coherence that drove original D-D > D-E.
  **OPERATOR FAMILY DECOMPOSITION (3 models, zero-cost, 2026-09-19):**
  Within same operator family, meaning on 11-bin = +0.002 (Qwen3), +0.008 (SmolLM2), -0.090
  (StableLM). ALL three show ZERO within-family meaning effect on answer distribution.
  Operator-family effect is +0.119 to +0.422 in SmolLM2/Qwen3 (real structure), near-zero in
  StableLM (+0.026). The FL "meaning matters" in SmolLM2 is partly family clustering.
  Same-family D-E can exceed same-family D-D (e+0~e+1 > e+0~e-0 in SmolLM2).
  **CROSS-POSITION ANALYSIS (zero-cost, full-logit, 2026-09-19):**
  A-A > A-C in both models (CI excludes zero). After-perts cohere within position.
  A-D ≈ C-D in both: rewrites equally low from both positions.
  **ADVERSARIAL REVIEW OF EXPANDED-D (Fable, 2026-09-20):** CRITICAL FINDING — expanded-D
  experiment is confounded. D={+0,-0,*1,**1} spans 3 operator families. D-D pairs are
  predominantly cross-family (penalized by family heterogeneity). D-E pairs include
  same-family matches (boosted by family coherence). SmolLM2-360M confirms: both surfaces
  reversed (D-E > D-D). Ratings: meaning-dead QUALIFIED (failure-to-reject not evidence-
  for-null), FL=clustering SOLID, family>meaning QUALIFIED (true by construction),
  expanded-D-fix FRAGILE (introduced worse confound). **FIX: same-family D expansion
  (additive-only, eliminates confound). Pending.**
  **REFRAMING (POST-ANSWER-IDENTITY-CONFOUND, 2026-09-20):** The gap measures POSITIONAL
  non-transfer and OPERATOR-FAMILY structure. Meaning preservation contributes NOTHING
  beyond answer identity.
  **ISOMORPHIC SAME-FAMILY D (4 models) — RETRACTED due to answer identity confound:**
  D={+0,-0,+(0),-(0)}, E={+1,-1,+(1),-(1)}. The DD-DE comparison is confounded:
  D perts ALL produce the same answer (x), E perts produce x+1 or x-1. Answer identity
  is perfectly collinear with the D/E label in DD-DE pairs. The correct isomorphic
  comparison is DD-EE, which shows no effect (SmolLM2-1.7B: FL p=0.071, 11b p=0.157).
  **ANSWER IDENTITY CONFOUND TEST (all 4 models × 2 surfaces):**
  Cleanest test: same-answer pairs (D-D vs E-E-same-sign). If meaning matters beyond
  answer identity, D-D > E-E-same-sign. Result: D-D <= E-E-same-sign in ALL 8 tests.
  | Model | Surface | D-D | EE-same-sign | Diff |
  |---|---|---|---|---|
  | SmolLM2-1.7B | FL | +0.782 | +0.770 | +0.012 (null) |
  | SmolLM2-1.7B | 11-bin | +0.489 | +0.538 | -0.048 (reverse) |
  | Qwen3-1.7B | FL | +0.563 | +0.588 | -0.025 (null) |
  | Qwen3-1.7B | 11-bin | +0.562 | +0.765 | -0.204 (reverse) |
  **MEANING: SMALL, MODEL-DEPENDENT (R29 correction of 62h retraction).**
  The original "D-D <= E-E-same-sign in ALL 8 tests" was biased: D-D averaged 6 pairs
  (4 cross-sign), E-E-same-sign averaged 2 (all same-sign). Corrected sign-and-form-matched
  comparison (DDsame vs EEsame, n=2 pairs per side): SmolLM2-1.7B +0.105 [0.025, 0.199] FL,
  +0.231 [0.116, 0.350] 11-bin (EXCLUDES 0). Qwen3-1.7B 11-bin: -0.132 (REVERSED). StableLM
  FL: +0.058 [0.001, 0.130]. Score: 4/8 D>E excl 0, 1/8 reversed, 3/8 null. Meaning is NOT
  zero — it is small, model-dependent, and fragile at n=2 pairs.
  Dominant factor is SIGN MATCHING (+0.19); meaning vs answer identity is non-identifiable
  in this arithmetic design (62i regression parametrization was a choice, not a finding).
  Factor ordering: position >> sign >> (form, answer, meaning non-identifiable below sign).
  R30b (corrected post-Fable review): op-stratified in-expression edits (J) are INTERMEDIATE
  (J-J=0.53-0.59 vs I-I=0.87, D-D=0.29), not D-like as initially pooled (0.413). Pooled
  statistic was dominated by `//`-comment artifact and between-operator block structure.
  Whitespace inside expression is inert; single-operand parenthesisation partially non-inert.
  "Rarity fits the ordering" WITHDRAWN — was a pooling artifact; op-stratified loadings are
  flat (0.53-0.60) across all AST-preserving in-expression edits despite 3x surprisal range.
  Three-level plateau: outside-expression (0.77-0.85), AST-preserving in-expression (0.53-0.60),
  computation-changing (-0.01 to 0.57). All cell-ranking rhos conditional on `# Output: ` readout.
- **R27 2×2 D-RELIABILITY DIAGNOSTIC (Qwen3-0.6B + SmolLM2-360M, 2026-09-20):**
  D unreliability is STRUCTURED — 2-model universal finding. 24 perts in 4 groups:
  A (inert), D (adjacent semantic ops), V (lexical variants of D), O (own-line versions of D).
  | Metric | Qwen3-0.6B | SmolLM2-360M |
  |---|---|---|
  | A-A | +0.696 | +0.780 |
  | O-O | +0.660 | +0.609 |
  | D-D | +0.392 | +0.412 |
  | Within-op D↔V | +0.632 | +0.912 |
  | Cross-op | +0.366 | +0.481 |
  **Universal (2 models):** (1) D unreliability is structured — 5-6/6 operations reliably measure
  different cell properties (within-op >> cross-op). (2) O-O >> D-D in both models:
  detaching semantic ops from the expression increases reliability.
  **Model-specific:** (1) In Qwen3, O ≈ A (A-O=0.552, O-O≈A-A); in 360M, O is BETWEEN A and D
  (A-O=0.409 < A-D=0.536). (2) temp_var: anti-correlated in Qwen3 (-0.346), structured in
  360M (+0.821).
  **R29 INLINE-INERT EXPERIMENT (2 models, 2026-09-20): WHICH-LINE EFFECT IS ZERO.**
  Added group I (in-line inerts: comment, semicolon, tab, extra space).
  DESIGN FLAW: i_trail_space = a_whitespace (bit-identical duplicate). Corrected numbers:
  | Metric | Qwen3-0.6B | SmolLM2-360M |
  |---|---|---|
  | A-A (corrected) | +0.826 | +0.746 |
  | I-I (corrected) | +0.878 | +0.789 |
  | D-D | +0.392 | +0.412 |
  | I-I − A-A | +0.052 | +0.044 |
  | I-I − D-D | +0.486 | +0.377 |
  **R30b IN-EXPRESSION INERTS — CORRECTED post-Fable review (Qwen3-0.6B, 2026-09-20):**
  In-expression edits (J group) pooled J-J = +0.413, but Fable review found 3 SERIOUS defects:
  (1) `//`-comment artifact: j_space_before_op on 15 `//` cells creates `x = 3  // 4` read
  as C-style comment by bare surface (10x distance inflation, dominates pooled rho).
  (2) j_spaces_both is pseudo-replicate of j_space_before_op (rho 0.957).
  (3) j_unary_first adds UnaryOp AST node — misclassified, belongs in D.
  Op-stratified (correct): J-J = 0.53-0.59 vs I-I = 0.87, D-D = 0.29. **J is INTERMEDIATE,
  not D-like.** Whitespace inside expression is inert (loadings 0.79-0.86). Single-operand
  parenthesisation partially non-inert (j_parens_first 0.40, j_parens_second 0.66) while
  whole-expression parens nearly inert (d_parens 0.77). Neither adjacency nor semantic change
  predicts this ordering; "rarity of resulting code form" fits but is untested.
  NEW FINDING: `a  // b` (2 spaces before `//`) read as C-comment on bare surface;
  `# Python code` header restores floor division — surface disambiguates language (10x).
  CAVEAT: cell rankings readout-specific (cross-readout correlation 0.04-0.72).
  A-A, I-I, D-D recomputed on bit-identical data; deltas reflect only 2 swapped perts.
  Note (R29): disattenuated AD is not D-specific; A is special (near-duplicate columns).
  **R31 Stage 1 — SPACING/PAREN PERTS, CORRECTED (Qwen3-0.6B, 2026-09-20):**
  5 new perts + 2 anchor re-runs (226s CPU, 98.8% accuracy, anchors reproduced to 4e-6).
  Fable adversarial review: 1/6 claims survive, 5 withdrawn.
  SURVIVING: p_double `((expr))` strat4=0.279 < d_parens 0.655, cell-level paired bootstrap
  diff -0.376 CI[-0.60,-0.15]. Robust to artifact removal (strat3 diff -0.35 CI[-0.61,-0.08]).
  WITHDRAWN: per-operator mechanism (// vs +/- within 1 SE), betwOp inversion (p=0.24 on n=6),
  token-context refutation (0.670≈0.671 manufactured by `%` artifact), rarity dead (untested
  as registered; both matched contrasts went rarity's way). CIs corrected (operator-level →
  cell-level stratified bootstrap). Anchor ceiling corrected (0.924 → LOO 0.881, true A-A ~0.80).
  LEADING ALTERNATIVE: header shielding — parens make the header redundant, depth-monotonically.
  On `-`: mean dist A=33.6 → (e)=24.4 → ((e))=14.0, SD 10.0→5.5→2.1.
  `%`-disambiguation artifact confirmed (same class as `//`-comment): `<digit>%` no-space triggers
  percent-literal reading (7-15x distance). Spacing perts near-inert once artifact excluded.
  R32 NESTING DEPTH: (((expr))) strat4=0.287 ≈ ((expr)) 0.279. Floor at depth 2.
  M-shield holds through depth 2 on header-sensitive operators but all 4 operators show small
  mean distance INCREASE at depth 3. Floor is real, not pure monotonic compression.
- **WITHIN-OPERATOR STRATUM ANALYSIS (balanced v3, zero-cost, 2026-09-19):**
  Gap SURVIVES and GROWS within operator strata:
  | Metric | Qwen3 | SmolLM2 |
  |---|---|---|
  | Raw gap | 0.500 | 0.275 |
  | Within-operator residual gap | **0.515** | **0.365** |
  Operator strata SUPPRESS the gap, not create it. DD < AD in 5/5 (Qwen3) and 3/5 (SmolLM2)
  strata. d_add_zero and d_zero_plus (crossover D members) have LOWEST AD correlations
  (0.068/-0.171 Qwen3) — they're the least representative D members, NOT the gap's drivers.
  d_cond_true highest (0.584/0.839). R25's "operator strata reverse diagonal" was about
  anomalous D members on old cells; the gap overall is robust within operator strata.
- **HEADER RELOCATION TEST (Qwen3-1.7B, R22 E1, fp32 CPU, 2026-09-18):**
  Moved `# Python code\n` from before code to after assignment. Tests whether C group's
  anomalous low correlations are header-adjacency or genuine position effect.
  | Group Rho | Original | Relocated | Delta |
  |---|---|---|---|
  | A_A | 0.786 | 0.584 | -0.202 |
  | B_C | 0.617 | 0.162 | -0.456 |
  | C_C | 0.589 | 0.453 | -0.135 |
  | A_D | 0.333 | 0.350 | +0.016 |
  | D_D | 0.224 | 0.239 | +0.015 |
  | Gap | 0.453 | 0.234 | -0.219 |
  **HEADER ADJACENCY REJECTED AS SOLE MECHANISM.** Simple adjacency predicted C_C increase;
  it decreased. Header in the middle disrupts ALL correlations, not selectively adjacent ones.
  D_D and A_D barely changed — D perturbations are header-position-independent.
  Gap halved from 0.453 to 0.234: ~50% header effect, ~50% genuine A-D structural difference.
- **READOUT METRIC COMPARISON (4 models, zero-cost reanalysis, 2026-09-18):**
  | Model | Full-logit gap | 11-bin gap | Cell-profile cross-readout rho (inert/D) |
  |---|---|---|---|
  | Qwen3-1.7B | 0.452 | 0.359 | 0.099 / -0.084 |
  | SmolLM2-1.7B | 0.281 | 0.374 | 0.652 / 0.653 |
  | StableLM-1.6B | 0.299 | 0.338 | 0.429 / 0.286 |
  | CodeGen-350M | 0.318 | 0.190 | -0.438 / -0.476 |
  Gap survives on 11-bin in ALL models. But Qwen3 and CodeGen have near-zero or negative
  cell-profile cross-readout rho — WHICH cells are sensitive differs between full-logit
  and 11-bin. Cell-level transfer claims must specify the readout.

**Cross-model perturbation rank correlation (full-logit mean distance):**
- Qwen3 vs SmolLM2: r=0.035 (near zero; 95% CI ~[-0.37, 0.43] at n=24, underpowered)
- Qwen3 vs Phi: r=0.203 (weak)
- Qwen3 vs StableLM: r=0.424 (moderate, p=0.039)
- SmolLM2 vs Phi: r=0.521 (moderate, p=0.009)
- SmolLM2 vs StableLM: r=0.523 (moderate, p=0.009)
- Phi vs StableLM: r=0.495 (moderate, p=0.014)
- Qwen3 is the outlier; SmolLM2/Phi/StableLM form a cluster (r≈0.5)
- Cell-level gradient correlates Qwen3-SmolLM2 (r=0.446, p=0.004) but not with Phi (r≈0)

**MULTIPLICATIVE FACTOR ASYMMETRY (8 models, zero-cost, 2026-09-19):**
The decomposition d[i,j] ~ f[i] * m[j] has fundamentally different cross-model structure
for each factor:
- **Perturbation magnitude m[j]:** cross-model rho = +0.397 [-0.17 to +0.77]. PARTIALLY SHARED.
  Most stable: c_desc_comment (rank 23-24 in all 8), c_assert_before (rank 20-23 in 7/8).
  C-group (before-position) always weakest. Group ordering: A ≈ B > D > C in 6/8.
  Exception: Qwen3 family has D > A (rewrites MORE disruptive than inerts).
- **Cell fragility f[i]:** cross-model rho = -0.022 [-0.49 to +0.35]. COMPLETELY MODEL-SPECIFIC.
  Even within-family: SmolLM2-135M vs 360M: -0.12. Qwen3-0.6B vs 1.7B: +0.03.
- **Implication:** you can partially transfer WHICH TYPES of changes matter (perturbation
  magnitude hierarchy is ~40% shared) but you CANNOT transfer WHICH INPUTS are vulnerable
  (cell fragility is ~0% shared). Alignment implication: general rules about change types
  may generalize; per-input vulnerability maps do not.

**TOKEN SEGMENTATION DOES NOT PREDICT FRAGILITY ORDERING (8 models, zero-cost, 2026-09-19;
R23-corrected heading):**
SmolLM2 (49K vocab) and Qwen3 (152K vocab) have DIFFERENT tokenizers (26% vocab overlap,
different token IDs) but produce IDENTICAL segmentation of all 40 assignment lines (7 tokens,
same boundaries). StableLM/CodeGen/Phi have CONSTANT token count (zero variance).
Feature regression: R²=0.41-0.84 per model (null expectation ~0.13-0.21 for 5+ predictors
at n=40). No feature has consistent sign across all 8 models, but P(all-8 agreement) ≈ 0.4
even under a true shared effect of rho=0.3 — the sign-consistency test is uninformative.
**R23 CAVEAT:** Fragility is 34-74% operator stratum (eta², KW p<.05 in 8/9 models).
Identity operations (n*1, n//1, 1**5) are confounded with operator class. Effective n ≈ 6
strata, not 40 cells. Double-operator feature was exploratory (post-hoc flag stands).

**CROSS-MODEL TRANSFER (8 models, zero-cost, 2026-09-19; R23-CORRECTED):**
Models: SmolLM2-{135M, 360M, 1.7B}, Qwen3-{0.6B-Inst, 1.7B}, StableLM-1.6B, CodeGen-350M,
Phi-1.5-Q1. Two of eight (CodeGen 32.5%, SmolLM2-135M 52.5%) fail admissibility. Qwen3-0.6B-Base
omitted from the 8 but included in Base/Instruct comparisons.
- **Cell fragility:** mean rho = -0.022 (model-level jackknife 95% CI [-0.09, +0.14]).
  Heterogeneous: SD=0.23 vs null SD=0.16; 4/28 pairs significantly negative, 1 positive.
  Mean flips sign with Phi-Q0 (+0.024). Qwen3-0.6B Base/Instruct (same architecture): 0.333
  (p=.036). **Ceiling problem:** same-weights Phi Q0 vs Q1 fragility rho = 0.45 — if the
  same model doesn't agree with itself across readouts beyond 0.45, cross-model 0.3 would
  already be near ceiling. "Zero" is not established; "small mean, large spread" is.
- **Perturbation magnitude:** mean rho = +0.397 raw; 0.170 after removing A/B/C/D group means;
  0.100 with C-group dropped. The shared component is the C-group (header-adjacent) magnitude
  floor already identified in R17/R22. 16/28 pairs significant at n=24.
- **Profile transfer (0.326) carries no cell-specific information** (R23 correction): per-cell
  profile rho against target model's mean profile (no cell info) = 0.356; against row-shuffled
  target = 0.318. Cell-matched excess: +0.008. This is the magnitude correlation restated, not
  a separate test.
**R23 NOTE:** All cross-model claims use cells with operator-stratum eta²=0.34-0.74 and
confounded identity operations; effective n ≈ 6 strata, not 40 cells. R21 required
operator-balanced cells (n≥60, no identity expressions) before any new cross-model claim.
That precondition was not met. The balanced cell set is the only admissible next step.

**WITHIN-FAMILY vs CROSS-FAMILY TRANSFER (8 models, zero-cost, 2026-09-19; R23-QUALIFIED):**
- **Magnitude transfer:** within-family rho = 0.678 ± 0.091 >> cross-family rho = 0.350 ± 0.260.
  **R23 CAVEAT:** After removing group means, raw 0.397 → 0.170; with C dropped → 0.100.
  The within-family advantage may also be C-group-floor dominated.
- **Fragility transfer:** within-family rho = 0.028 ± 0.095 ≈ cross-family rho = -0.030 ± 0.241.
  Both near zero at n=40 (effective n ≈ 6 operator strata). Family doesn't help.
  **R23 CAVEAT:** 4 pairs significantly negative — heterogeneous, not simply zero.
- No scaling law: param distance does not predict transfer (rho=-0.165, p=0.40).
- Qwen3-1.7B is magnitude-transfer outlier (0.174 vs others >0.29).
**R23 NOTE:** These patterns are on 40 cells with confounded operator strata. Cross-model
claims require the operator-balanced cell set (n≥60).

**INSTRUCTION TUNING SEPARABILITY (Qwen3-0.6B Base vs Instruct, zero-cost, 2026-09-19):**
Qwen3-0.6B-Base magnitude ranking has rho=0.905 with Qwen3-1.7B-Instruct and rho=0.737
with Qwen3-0.6B-Instruct. Base-to-non-Qwen3 transfer = 0.186, Instruct-to-non-Qwen3 = 0.285.
Instruction tuning has TWO SEPARABLE effects:
- **Magnitude hierarchy:** architecture-determined, UNCHANGED by instruction tuning.
  Qwen3 Base and Instruct agree on which perturbation types are strong/weak.
- **A-D gap amplification:** 1.5x increase (like-for-like; R23-corrected from 5.6x).
  Instruction tuning doesn't change WHICH changes matter, but amplifies the DIFFERENCE
  between meaning-preserving rewrites and inert modifications. n=1 pair; not yet replicated.
Qwen3 is a genuine architectural outlier in perturbation magnitude ranking (d_temp_var
ranked #1-2 vs #16 median in other families; b_dead_var ranked #18 vs #2 in others).
This explains why Qwen3 magnitude transfer to non-Qwen3 is low (0.08-0.29) while
within-Qwen3 is high (0.71-0.91).

**GAP MECHANISM (Qwen3-0.6B Base vs Instruct decomposition, 2026-09-19):**
| Component | Base | Instruct | Change |
|---|---|---|---|
| Within-A rho | 0.910 | 0.791 | -0.119 |
| A-D cross rho | 0.586 | 0.301 | -0.284 |
| Within-D rho | 0.394 | 0.169 | -0.225 |
| Gap (AA-AD) | 0.324 | 0.489 | +0.165 |
70% of gap increase from A-D cross-rho dropping. Instruction tuning makes D perturbations
(semantic rewrites) MORE IDIOSYNCRATIC — each rewrite affects cells in its own distinct
pattern. D-group absolute sensitivity DECREASES (0.69x) while within-D coherence drops
2.3x (0.394→0.169). The gap captures PROFILE DIVERSITY, not magnitude: instruction-tuned
models discriminate between meaning-preserving rewrites more finely, while treating inert
modifications more uniformly. Group-specific amplification: A=0.98x, B=0.94x, C=1.42x, D=0.69x.

**ROBUSTNESS: Tokenization-covariate regression (R17, zero cost):**
- Token-count delta between minimal/verbose surfaces is CONSTANT (4 tokens) across all 24
  perturbations for 4/5 models. The only surface difference (`# Python code\n`) always adds
  the same tokens regardless of perturbation. Qwen3 has 1 exception (c_newline_before, 3 tokens).
- R^2: Qwen3 0.347, all others 0.000 (degenerate — constant input). Below 0.5 threshold.
- **Per-perturbation distance variation is NOT a tokenizer artifact.** Variation is genuine model behavior.

**ROBUSTNESS: Operator-class test (R17, pre-registered, zero cost):**
- Pre-registered H1: D-sensitivity varies by operator class AND A-sensitivity does not.
- **H1 FAILED for all 5 models.** Both A and D sensitivity vary by operator (Kruskal-Wallis):
  Qwen3 (A p=0.003, D p=0.012), SmolLM2 (A<0.001, D=0.001), StableLM (A=0.002, D=0.024),
  Phi (A<0.001, D<0.001), CodeGen (A=0.022, D=0.319 NS).
- Operator class is COMMON FRAGILITY, not perturbation-type-specific. "Pattern structure" is empty.
- The gradient is amplitude-invariant by construction (Spearman). Group amplitude ordering is
  model-dependent (Qwen3 D>A, others D<A, all paired p<0.01) — a clarification, not a finding.

**R14 control experiment results (Qwen3-1.7B, fp32 CPU, 2026-09-18; CALIBRATED):**
- Groups: A (reference inert), E (unusual inert appends), F (inert assignment-line mods), D (rewrites)
- Full-logit ordering: A(0.785) > E(0.645) > F(0.471) > D(0.224)
- 11-bin ordering: A(0.686) > F(0.583) > E(0.442) > D(0.182)
- E and F swap between surfaces in point estimates, but **bootstrap CIs include zero on
  BOTH surfaces** (fulllogit: +0.171 [-0.013, 0.355]; 11-bin: -0.138 [-0.341, 0.048])
- Within-group heterogeneity dominates: f_semicolon (0.872) > any E member; e_list_pop
  (-0.021 on 11-bin) < any F member. Groups E/F are not coherent classes
- Neither "computation path" nor "syntactic novelty" can be cleanly distinguished
- Individual perturbation identity matters more than any group-level factor
- Both surfaces preserve the core A > D gradient — the gap is robust regardless of readout

**Known limitations:** 11-bin is code-specific. Position confound PARTIALLY real
(~0.2 contribution) but gap survives controls (kill experiment 2026-09-18).
Qwen3 full-logit has fp16 resolution floor (60% distances < 1e-3) but
SmolLM2 has ZERO sub-1e-3 distances, confirming the gradient is not a floor
artifact. Phi-1.5 code was fp32 but NL was fp16 (precision mismatch
invalidates cross-domain Phi comparison). Falcon-H1 (SSM-attention hybrid)
experiment interrupted; not restarted due to battery constraints.
Single-factor "computation path" and "syntactic novelty" interpretations are
BOTH refuted as sole explanations — the gradient is readout-dependent with
at least two contributing mechanisms (R14 controls, 2026-09-18).
R15 CORRECTIONS: Position ordinal as FL predictor (r=-0.538, p=0.021 uncorrected)
does NOT survive Bonferroni correction (10 tests). Mean distance as 11-bin predictor
(r=-0.703, p=0.001) survives correction but is tautological (rank-SNR effect).
"Two partially independent axes" claim downgraded to: one confirmed tautological
predictor, one suggestive uncorrected predictor. d_eff analysis confirms earlier
#19 dimensionality; not an independent finding (R13 already withdrew this).
RII significance: Pearson r=-0.944 (p=0.016) at n=5. z-RII/FDI claims RETRACTED
(null artifact per R16 R2). But gradient survives both fragility partialling
(residual gap 0.25-0.63 for 4 models, all CIs exclude 0; Phi 0.044 CI includes 0) and permutation null (Z=5.5-8.8,
all p<0.001; Phi p=0.168 NS). The gradient is real perturbation-type structure.

Results: experiments/results/transfer_spectrum/. Fable R12 complete (fragility
decomposition, held-out prediction). R13 complete (claim triage, kill experiment
design). Kill experiment COMPLETED across 4 architectures (fp32 CPU — 2026-09-18).

---

**Prior work (paused 2026-09-14).**
Evidence reaches S65; S58 completed an
independently reproduced CPU teaching precursor. Correct specialist pseudo-labels
improved a frozen-feature species head; soft targets offered no advantage over
hard labels under the locked recipe. The [canonical next-work boundary](theory/CUMULATIVE_CAPABILITY_ACQUISITION.md#next-work-boundary)
is authoritative for active allocation and closed/deferred episodes.

S59 declines an unmotivated hierarchical-classifier comparison. S60 adds external
evidence for a jointly taught representation supporting later trained readers.
S61 closes that source inquiry with behavioral coverage evidence and charged-reader
repair requirements. S62 completed the later-attribute comparison: mean pair scores
.631 for specialist features, .607 for full logits and .690 for existing recipient
features on fourteen pairs. Predictions were independently reproduced. Selected
visual checks qualify the labels' visibility. Close this panel; it does not justify
requiring specialist features, nor prove general logit sufficiency. Next identify
a useful contribution beyond existing recipient access and matched teaching.

S63 identifies an external temporal-service precedent and its attribution boundary:
object association, persistent extent, memory and trained consumers are distinct
contributions. Compare joint preparation against separate contributors with the
same useful consumer. The source inquiry is complete; no large local reproduction.

S64 rejects unmotivated temporal-state migration and distinguishes recovery from
composite segmentation gains. Existing-weight memory/search is a serious direct
remedy; retrospective trajectory selection is not an immediate decision contract.
The source selection is closed without local tracking reproduction.

S65 completed a three-case supplied-revision diagnostic. All 18 responses terminate;
original citizenship is correct in 2/3 and supplied citizenship in 3/3, but capital
prerequisites and supplied composed capital answers are both 0/3 (termination 3/3
in each arm). There are zero prerequisite-eligible cases. Manual reviewers confirm
country-for-capital substitution and one wrong-subject citizenship misattribution.
This is a scoped interface limitation, not missing knowledge or an intrinsic
composition failure. First-attempt serialization failure and successful retry are
preserved separately. The natural-question prerequisite remedy is unrun, not queued.

The shared-predictor branch needs useful second-donor acquisition evidence; its
isolation and conditional cache arguments are already worked out. Connected use,
structured consequence records and global policy revisions remain distinct routes.
No cumulative acquisition, local concept-correction consumer or intrinsic latent
space failure is established. Source-access failures do not establish model failure.

## Current direction: cumulative capability acquisition (2026-09-13)

Read the [canonical direction and decision map](theory/CUMULATIVE_CAPABILITY_ACQUISITION.md).
The enduring philosophy is native latent mathematics discovered from a denizen's
needs: identity, moves, effort, maps and laws. Native means coordinate-independent
and effective; existing mathematics is welcome. Cumulative acquisition is the
current direction within that mission, not its replacement.

Explore capability grafting, reasoning over encoder representations, and a
persistent system that combines successive pretrained donors while retaining
prior capabilities. Connected growth retains donor runtime; assimilation after
donor removal is a separate branch. Do not assume isolated capability modules,
a fixed shared vector space, training-free transfer or proprietary access.

**Mode on explicit user resumption:** strategy/discovery, authorized 2026-09-13 at
19:45 UTC. Definitions, primary literature and bounded CPU probes should resolve
decision-changing questions. No system-building, large-model loading or sustained
compute; per-run GPU approval remains. Consolidate before approximately two-hour
adversarial approach reviews. The first completed around 21:42 UTC (audit e1027);
the second completed around 23:03 UTC (e1039-e1040), and the third around
00:54 UTC on September 14 (e1061-e1062), after consolidation at 89b34ae.
Fourth review completed around 02:40 UTC (e1086-e1087), after e72f83d and a
separate entropy review. Strategy map review completed around 03:34 UTC
(e1095–e1096); retain the two-hour cadence
during subsequent substantive research.
The reviewer challenges approach and question selection, never the mission.

**Evidence status:** S58 establishes modest teacher-assisted species readout after
donor removal. No cumulative acquisition, transferable graft, encoder reasoning
artifact, new backbone knowledge or cost advantage is established.
The earlier four bounded probes sharpen the evidence requirements (S26 below);
current eight-block accounting is reported separately:

- **S6, retained exports:** legacy plan bundles are nonfinite or overwhelmingly
  zero and unsuitable for the proposed transfer analysis. Underlying model
  capabilities were not tested. Integrity audit e1011–e1012 upheld this scope.
- **S21, conditional port:** eight CPU forwards / 32 rows, no fitting. Four
  isolated conditional rectangles fall below registered tolerances. Fresh real
  Codex CLI audit e1021 found an additive preference class also fits the primary
  tolerance; no useful conditional mechanism or global law is established.
- **S22, cached prediction screen:** the locked equal-rank combination loses to
  the retrospectively better individual donor on four assays / 8,610 variants.
  Fresh real Codex CLI audit e1022 prompted exact tie handling, now corrected in
  the retained result and independently validated; the negative direction holds.
  No complementarity-impossibility, latent-information or biological-use claim.

Exact provenance, qualifications and corrected numbers live in the
[experiment log](experiments/EXPERIMENTS.md), [ledger](experiments/ledger.jsonl)
and canonical S6/S21/S22. S23 is proof-only: pairwise compatibility need not
imply joint compatibility, and compatible marginals need not identify new joint
answers. It is an obligation for applicable interfaces, not a selected architecture.

**Closed task cases:** [Canonical S8/S24](theory/CUMULATIVE_CAPABILITY_ACQUISITION.md) preserve the PFN1
endpoint/receiver eligibility failure and the sufficient scoped-record contract.
The former is not a whole-benchmark or trained-model attribution verdict; the
latter does not settle broader capability composition. Coordinate repair and
caller-supplied continuation laws remain integration obligations. Source readings,
independent checks and decision chronology stay in canonical theory, NOTEBOOK
and the ledger; neither case is queued for extension.

**S26 result, independently audited (e1044-e1045):** the single autonomous-continuation probe
completed in 10.1455 seconds, seven CPU forwards / 112 rows, on 16 fixed pairs
nested in four assay reference constructs. All fixed-position cases completed;
repeat probability drift was zero. Primary and independent pair recovery were
both 0/16; second-site recovery both 2/16, versus cyclic content control 3/16.
The primary first proposal was correct only 1/16, bounding attainable pair recovery
below the locked .10 gain threshold. Close this procedure on this support; weak
proposals prevent a strong conclusion against competent composition. No parameter
fitting, new model, GPU or panel extension. Root read all 16 decoded outputs;
the fresh real CLI audit independently reproduced all counts and code-path checks.
Reverse and confidence-first each correct one prediction and spoil one; primary
corrects/spoils none. The 10.1455-second timing starts at main, excluding interpreter
startup; an external 180-second process wait also returned normally. Exact receipts
and the correction of the audit prose's runtime transcription are in the ledger.
This completed probe is not reopened. Preserve the broader capability-composition objective.

**Historical S50 decision (superseded by paused S65 routing above):** retain S50/S51 as source and operational evidence, then pivot
to a specific testable benefit of sharing or assimilating useful computations.
The fresh periodic review around 15:30 UTC distinguishes unchanged connected
extension from modification of an established service; replay is not automatically
needed for the former. It rejects another availability-driven substitution.
Next investigate cumulative assimilation of
fresh-input operations using already studied teaching recipes; ordinary teaching
and connected growth remain valid alternatives. No trained AB endpoint was
bound in the limited source check; do not extend the hunt or train a substitute.
Old-operation prompts and prompts produced by an acquired operation need separate
coverage. Repeated questions over one memory are not successive acquisition.
S40 supplies a connected-use precedent; S49 supplies a later-question memory
consumer and a complete trained-reasoning comparator. Their unresolved combination
is not a selected architecture. Useful contribution may be information, computation,
accessibility or effort savings; novel information is not mandatory.

See the [canonical next-work boundary](theory/CUMULATIVE_CAPABILITY_ACQUISITION.md#next-work-boundary).
S37 stays parked, S43/S47 stay closed, and the S49 small-checkpoint branch is closed
without download or execution. No new model run or apparatus is queued.

**Session accounting:** 1710 apparatus / 0 reusable artifact-bearing code lines
(98 + 119 + 204 + 207 + 330 + 107 + 286 + 222 + 137); nine probe blocks / zero building rounds.
S37 has two failed execution attempts and zero scored forwards. There are 666 recorded
forward calls: 15 earlier batched, 540 single-image S58/S62 and 111 S65 retry calls.
S65's failed first attempt adds an unrecorded 1–16 calls: total bounded 667–682.
S58 fits twelve candidate linear heads in 2400 optimization steps; S62 prepares
45 ridge readers (36 nonconstant and nine constants). No GPU. No reusable acquisition
component was built. S43 separately inspects32 published
responses on16 matched queries; it adds no local model inference. The explicit strategy mandate
supersedes construction merely to change this ratio.

**Continuity and custody:** commit-seal rung 1 stays paused; closed lines stay
closed, and the strong-assert panel is a deferred option. Historical quota dates
are not current access checks. Root alone writes the blackboards because separate
MCP sessions reused IDs; reviewers return findings for provenance-preserving relay.
Mission synthesis is skipped under its known large-board hang workaround.
Real CLI result audits require a working fresh CLI. Under the latest restart
workaround, available independent strategy/data reviews are labeled accurately and
do not halt over inaccessible tooling. Dated session checkpoints remain
in NOTEBOOK.md and the ledger rather than being repeated in current routing.

## Historical state below

The following dated record preserves the prior direction and audit decisions.
Its next-step and current-direction language is historical; use the entry above
for present routing. Numerical claims remain subject to their latest corrections.

## Prior direction: Selective Write-Boundary Construction (2026-09-04)

**Pivot (Codex direction dialogue, 2026-09-04):** Close the pretrained-Qwen
path-dependence line. Skip CEG-1R — its outcomes produce no different program
decisions (both paths lead to construction). Measurement-to-artifact ratio was
infinite (506:0); governance required halt above 5:1.

**Central question:** If a model is given an internal "replace this fact"
operation (an endogenous keyed write with hard-masked overwrite), does
compositional reasoning become possible because dead history is truly dead?

**Anti-tautology constraint (Codex):** The write boundary alone is NOT the
artifact. Boundary satisfaction is by construction. The claim-bearing results
are: (1) learned use of writes for compositional reasoning, (2) predictive
compression — |Q_H| bounded by store cardinality not history length, (3)
transfer across held-out presentations, (4) causal substitution of store-
equivalent states, (5) advantage over matched append-only ablation.

**Theory:** theory/CONTINUATION_REFINEMENT.md, CR-10 through CR-14. Laws L1-L4
(overwrite absorption, cross-register commutativity, preservation, write
fidelity). The last-write normal form bounds the write orbit from one initial
place by ∏_r(|V_r|+1). With faithful native reads, an actual complete
initialization gives exactly ∏_r|V_r| places. These statements do not
bound the entire predictive carrier when preserved context varies. Five
pre-declared falsifiers.

**CR-14 revised (2026-09-04):** Original hard-register design was EAC
tautological (Codex correction). Replaced with commit-seal carrier: hard seal
(information bottleneck) + learned update U(m, address, proposal) → m+. The
model CAN fail — the learned update is the falsifiable object, not the
hard-wired register semantics. WB-1 validates the normal form math but is
architecturally tautological. Positive-control staircase: 1 address, 2 values,
2 commits first.

**Narrative gate:** "Give a model an internal act that genuinely means 'replace
this fact,' then test whether compositional reasoning becomes possible because
dead history is truly dead."

**Status relative to the central claim:** CR-15 supplies conditional
behavioral structure and sufficient certificates. It establishes no learned
realization, transfer, causal-substitution result in a learned model, or
advantage over an append-only alternative. Those remain the claim-bearing
obligations.

**Mode since 2026-09-05: analytical (compute-constrained).** Work is
derivation and proof; runs only for tiny models when a derivation needs a
concrete check. Rung 1 of the commit-seal staircase ran 2/5 seeds (both arms
passed every mechanism gate; stale partial deleted, weights not saved) and is
paused, not closed.

**CR-15 adopted (2026-09-05, theory/CONTINUATION_REFINEMENT.md):**
- Under A1-A3, each T-saturated subset is canonically B_T x S_T as a set
  equipped with its T-write action. The general write-orbit bound from one
  initial place is ∏_r(|V_r|+1). An actual complete initialization gives
  exactly ∏_r|V_r| places. Numerical initialization alone does not imply
  saturation, and preserved context B_T may be infinite.
- Transition-closed realization sets with η-accurate registered response
  laws give a history-independent 2η bound whenever the compared histories
  have the same abstract endpoint. Preserved abstract distinctions require
  margins exceeding 2η. Small local absorption errors alone do not provide
  a vanishing history-uniform error bound (cyclic-counter counterexample in
  CR-15.2). A greatest tolerance-compatible relation need not be an
  equivalence relation, and a coarsest tolerance-bounded congruence need
  not exist (three-state counterexample in CR-15.2).
- Contraction supplies invariant sets (surgeon certificate): per-value
  contraction with verified invariant domain and initialization coverage
  gives radius kappa D/(1-kappa) balls; selective-tube version for multiple
  registers requires exact preservation of unaddressed coordinates and
  contraction uniform over them. Fibered contraction with consistent fixed
  points does NOT imply commutation (explicit counterexample).
- Selective memory requires selective forgetting: a global behavioral
  contraction under every write is incompatible with indefinite preservation
  of an unaddressed register.
- A constant, address-independent sigmoid gate CAN realize exact selective
  overwrite (address selection in the candidate); keyed gating is not a
  theorem-level requirement.
- Withdrawn: "positive attention mass on dead tokens forces a nonzero
  absorption defect". Surviving restricted obstruction: with one bounded-
  logit softmax route the LIVE value's influence dilutes like e^B/(m+e^B)
  (fidelity, not absorption, fails with history).

**Status relative to the central claim:** CR-15 supplies conditional
behavioral structure and sufficient certificates. It establishes no learned
realization, transfer, causal-substitution result in a learned model, or
advantage over an append-only alternative. Those remain the claim-bearing
obligations.

**Accounting:** This is an analytical cycle with no runs. Zero runs does not
determine the apparatus-to-artifact code-line or round ratio; those
classifications must be reported separately.

**Audit #39 (2026-09-05): REVISE.** Mathematical core of CR-15 retained
(15.0, 15.2, 15.3, 15.5, 15.7 ADOPT; 15.1, 15.4, 15.6 REVISE, corrections
applied verbatim). Withdrawn from every surface: "distance from claim 0"
and "pretrained transformers lack an endogenous overwrite boundary".
Licensed sentence: under a fixed registered deterministic interface, exact
selective-write laws yield a context/store factorization on write-saturated
states, and a transition-closed, η-accurate realization of an abstract write
system bounds equal-endpoint histories by 2η under every registered
continuation; CR-15 supplies sufficient contraction certificates and an
exact constructed gated-cell example, without establishing learned
realization, transfer, or architectural advantage.

**CR-15.8 — audit #40: mathematical core ADOPT; parameter and attribution clarifications applied.** Fix one initial behavioral place and the registered deterministic interface. If its entire write orbit contains at most N exact behavioral places and absorption and commutation defects are uniformly at most ε in the full behavioral metric, equal-normal-form histories differ by at most B_N = min{1, (N−1)ε} under every common registered continuation. No additional factor two is required. With native own-write calibration error β, a sufficient live-value separation bound is Δ − 2(B_N + β). The worst-case error has order min{1, Nε}; the binary example preserves its live value exactly while accumulating nuisance-response error. The exact envelope is attained when singleton-valued registers are permitted, with zero cases handled separately. A full-behavioral packing bound gives min{1, Nτ + (N−1)ε}, and Nτ can be attained for arbitrarily small positive defects. Finite state, packing, contraction certificates, invariant relations, and ultrametric behavioral distance provide sufficient routes to uniform bounds; no exhaustive or disjoint classification is claimed. These results establish no learned realization, transfer, or architectural advantage. Corrected en route: \"identity cost c_τ = 0\" does not characterize overwrite, and append-only storage does not universally violate uniform reader accuracy.

**Next obligation

**Certificate check (2026-09-05; evidence gate REVISE):** the registered
single-cell attempt remains CERTIFICATE_INCONCLUSIVE. Separately, the sound
post-hoc confirmation procedure reports strict expansion for both writes on
D=[-1,1]^16 in the infinity norm. This is a scoped contraction-premise
failure, not a behavioral overwrite failure. Verifier defects (fixed after the
run, not rerun) prevented accepting the implementation as a general PASS
verifier. The eight-cell training-panel accuracy is 1.000, with maximum
immediate-read paired TV approximately 1.4e-8. Jacobian norms 0.987 and
0.539 are numerical diagnostics at 64-step candidates, not certified
contracting regions. No history-uniform behavioral bound is established.
Details: experiments/EXPERIMENTS.md.

**CR-15.9 adopted (2026-09-05, pending audit; register-free CR-15.8):** for
any deterministic action alphabet on a behavioral metric space, an orbit of at
most N exact places, nonexpansive actions, and a finite family of word
equations each holding to within ε on the orbit, two histories equal in the
least congruence generated by the equations are within min{1,(N−1)ε} under
every registered continuation. No registers, stores, or normal forms are
assumed. Finite orbit is sufficient, not necessary (an infinite-orbit
contracting world has arbitrarily small lifetime error).

**Re-contextualization ruling (Codex, 2026-09-05):** the program continues;
the selective-reset/state-budget certification line is NOT currently the
highest-leverage route to a real-model artifact ("not working yet" as a
route; the mathematics is retained as transferable constraint). The
"von Neumann trap" (imported addressed-register semantics certified as
discovery) is substantially real; native vocabulary = actions, response
distinctions and the continuations that reveal them, distinctions an action
erases/preserves/exposes, contextual equations with explicit error and
observation scope. Next cycle (direction dialogue, round 2 of 3): one
explicit rule for when an erased distinction stays irrelevant under
subsequent actions, anchored to the real-model CR-9 examples, with a
counterexample and an intervention consequence. Discrete-carrier testbed
deferred; no run.

**Locked direction (direction dialogue rounds 1-3 complete, 2026-09-05):
CR-15.10.6, ANALYTICAL_ONLY.** Interface-relative interchangeability with the
modulus ω_a^W(t) as organizing quantity. The next cycle is one bounded
analytical construction: (1) moduli for the contrast worlds (countdown
ω = 1 at every finite depth; doubling ω = min{1, 2t}; CR-15.5 selective
write θt on the addressed fiber; CR-15.9 decaying nuisance ½min{t, δ});
(2) the finite-panel bounding proposition; (3) the finite-observation
obstruction for unrestricted deterministic worlds; (4) one explicit
sufficient premise that lets finitely many questions yield an UPPER bound
on ω (the substantive gap: CR-15.2 bounds pairs in the same realization set,
not every W-close pair; CR-15.3 needs an observability premise in the other
direction, small d_W ⇒ small carrier separation). Real-model panel DEFERRED:
the historical CRC-0 panel has zero eligible W_H-close same-variable pairs
at τ = 0.01, P2's presentation is malformed (`func` vs `f`), and no model in
play is "tiny" under the directive; any acquisition needs explicit approval
and the locked criteria in CR-15.10.6 (τ = 0.01, numerical guard m, five
robust-close pairs over five task blocks, affine and length-only rivals
tested separately by interval-aware infeasibility, verdicts WITNESSED /
TOLERANCE_PRESERVED_ON_PANEL / INCONCLUSIVE).

**CR-15.10.7 (audit #41: REVISE, corrections applied):** finite-panel
enclosures, corrected declared-domain moduli, and the refinement identity
are retained. Response-only finite transcripts cannot establish a
nontrivial global bound over unrestricted extensions; independently
justified completeness or structural premises can. The contraction route
requires Lipschitz control of full question profiles and a uniform inverse
bound on the declared carrier domain. Saturation can degrade conditioning
without making inversion impossible. Local rank does not establish a global
inverse bound or PSR dimension. Neither the saved cell's retained-history
mechanism nor W_H's rank has been established. No new learned realization,
generalization result, or native mathematical novelty follows. Withdrawn:
the original obstruction theorem, the saturation impossibility, the global
rank bound and both dimension extrapolations, the cyclic parity claim.

**Question rotation (Devansh, 2026-09-06: never idle; rotate to a different
analytical question when a line closes).** Block 1 = CR-16, architectural
restrictions of the transformer world. Audit #42: RESTATEMENT; CR-16 requires corrections. Fixed-token causal site invariance and, with consecutive positions, the appended-token offset identity are retained. The exact two-route decomposition, content-independent prefix re-weighting, filler-effect subtraction and identification with CEG-1 RELAY-CUT are withdrawn. CR-15.10.6 already requires matched positions, valid tokenization boundaries and the registered H-versus-F rival. Close Q1 and rotate to one bounded Q4 reading of existing audited response tables under their original interface and claim limits, with no runs or new acquisition.

**Block 3 reading, evidence gate REVISE (2026-09-06):** on the archived
multifiller panel (24 roots, positions matched by construction, 11-bin
immediate-read TV, assumed floor) H and its matched successor filler
violate the 0.002 length-only prediction on 24, 24, 23 roots across three
filler variants: conditional LENGTH_ONLY_RIVAL_REJECTED at that channel;
unqualified locked-interface verdict INCONCLUSIVE. The "between-filler
spread" reference was confounded (each filler pair varies input and
successor together; off-diagonal endpoints absent), so "H is not
distinguishable from an arbitrary equal-length comment" and "content-
specific H disfavored" are withdrawn; generic readout sensitivity remains
unexcluded, not confirmed. Decision: future panels need several response-
independently selected successor fillers on fixed input pairs; the
deferred panel's discrimination against the readout-sensitivity rival is
unestablished. Ruling: stop zero-call rotation pending compute approval;
prioritized acquisition = position-matched, full-law, eligibility-first
panel on repaired strong-assert pairs (five task blocks; measure floor;
five robust W_H-close pairs required; then N/H successors and three fixed-
input successor fillers). Details: experiments/EXPERIMENTS.md.

**Block 2 (Q4) reading, evidence gate REVISE (2026-09-06):** on the archived
unequal-position svb_qwen3_congruence panel (Qwen3-1.7B-Base, 24 cells,
11-bin TV, assumed numerical guard), the neutral comment keeps all 32
initially close assert-type presentation pairs below 0.01 while the heavy
comment separates 29 (arithmetic verified exactly by the gate). Under the
locked CR-15.10.6 interface the verdict is INCONCLUSIVE (surfaces of a pair
differ by 1-3 tokens, so positions are unmatched; floor unmeasured; no
pinned revisions). Decision (amended): prioritize strong-assert as the
candidate family for a NEW position-matched full-law eligibility panel;
moderate-assert as boundary diagnostic; N-preservation and H-separation are
W_0-level hypotheses; the deferred panel is not replaced (only three
W_H-close pairs over two blocks exist in the archive). Q4 ruled decision-
relevant. Next block (Codex): the multifiller table's filler-versus-surface
comparison on its own roots, after verifying token matching and endpoint
availability; if non-actionable, rotation stops pending compute approval.
Details: experiments/EXPERIMENTS.md.

**Program ruling (audit #41):** the analytical cycle CR-15.10.6 STOPS after
this consolidation pass; wait for approval of an empirical step capable of
changing a program decision. Further solo theory expansion is not the
highest-leverage work; "this is not working as a route to a real-model
artifact." The valid mathematical residue is retained; the wider program
remains open.

**Withdrawn en route:** the "P vs R" mechanism dichotomy (not mutually
exclusive); the CV-of-ratio rival rule (b/d = g + c/d makes spread
uninformative); "immediate-read family closed under N".

**Standing check (Codex):** this is established behavioral mathematics, not
discovered native vocabulary; value must come from effective predictions
about a real model. A proven observation hole identifies an inadequate
interface, not an obstacle to structured reasoning in the model.

**So what:** find which differences between an AI's histories it can safely
ignore, and which later question makes those differences matter again.

**Closed lanes this cycle:** single-cell contraction certificate (evidence
gate: no further run; apparatus ratio 5.71:1); commit-seal rung 1 remains
paused (2/5 seeds, no saved weights); discrete-carrier testbed deferred.

**So what:** if a model has a proved limit on its distinct behavioral
situations, small update mistakes cannot accumulate beyond that limit times
the local error, even after a million updates.


---

## Qwen path-dependence line: CLOSED (2026-09-04)

CEG-1R remains a deferred diagnostic. The tested interfaces did not
establish a useful overwrite law. CEG-1 supports bounded attention-mediated
context dependence after a failed competence preflight; it does not prove
that pretrained transformers lack endogenous behavioral overwrite. The
construction program investigates whether an explicit write boundary
provides a useful alternative.

---

## CEG-1 result: REVISE / formal NO_INTERFACE — bounded context dependence (2026-09-04)

**CEG-1 (Codex strategic design):** Custom 4D attention masks on Qwen3-1.7B-Base
(eager attention, full forward passes). 16 fixed-slot histories × 4 identifier
sets, 6 intervention arms. 914 forwards, 259s CPU.

**Verdict (Codex Evidence Gate, audit #38): REVISE.** The original RELAY headline
does not pass. Competence preflight failed (83.6% < 95% threshold); the binding
design required NO_INTERFACE. RELAY-CUT Δ=0.000 is tautological (all downstream
positions see only pair-identical tokens once the cut disconnects the graph).

**Licensed sentence (Codex, verbatim):** In this fixed-slot Qwen3-1.7B-Base
panel, changing overwritten assignments altered normalized binary response
profiles, and after direct query-to-early-span attention was cut an indirect,
identifier-consistent excess over matched dummy cuts remained (mean max-channel
TV excess 0.101); however, the residual was channel-nonspecific and observed
after a failed competence preflight, so CEG-1 identifies bounded
attention-mediated context dependence, not semantic relay of dead information
in competent computation.

**Key numbers (post-audit):**
- OPEN Δ=0.406 (accuracy 83.6%; r0=71.9%, r1=95.3%)
- QUERY-CUT Δ=0.194 (accuracy 64.1%; channel specificity disappears)
- RELAY-CUT Δ=0.000 (accuracy 100%; tautological by mask design)
- LIVE-CUT Δ=0.934 (positive control)
- SYNCHRONIZER Δ=0.267 (accuracy 89.1%; one string rejected, not synchronization)
- Dummy OPEN Δ=0.105 (accuracy 96.1%)
- Paired excess (QUERY-CUT minus DUMMY-CUT): 0.101, bootstrap CI [0.050, 0.152]
- Target vs orthogonal excess under QUERY-CUT: 0.062 vs 0.061 (nonspecific)

**Audit corrections adopted verbatim:**
1. RELAY-CUT=0 is a tautology of the intervention, not an empirical finding.
2. The "52/48 pathway split" used arm-specific maxima across channels and cannot
   support an additive decomposition. Dummy-adjusted ratio is 33/67.
3. Under QUERY-CUT, register-specific dead-value effect disappears: target and
   orthogonal channels show equal excess over dummy. The residual is broadcast
   context sensitivity, not semantic relay.
4. Four "renderings" are one syntax with four identifier pairs — name robustness,
   not held-out template generalization.
5. SYNCHRONIZER rejects one string, not synchronization or irreversibility.
6. Design omitted target/orthogonal effects, empirical CDFs, clustered bootstrap
   intervals, full-vocabulary accuracy, replay noise, and model revision binding.

**Never say:** "Dead information persists through attention relay into live-write
representations; roughly 48% is relayed, and the failed synchronizer shows that
this relay is irreversible."

**Measurement-to-artifact ratio:** 506 apparatus lines to 0 construction lines
(infinite). One-more-test cap is binding.

**Line closed.** CEG-1R skipped (Codex direction dialogue: decision value nil).
Qwen path-dependence line closed. Transferable finding: the tested interface did not
establish behavioral overwrite; the broader architectural question remains
unresolved. Construction program opened (CR-10+).

---

## CRC-1 result: INVALID / bounded negative (2026-09-04)

**CRC-1 (Codex Architecture Theorist design):** Two binary registers, 32 roots
(8 per store), 4 literal actions, task-aligned {0,1} observer. 1508 forwards,
357s CPU. **INVALID** (competence 87.5% < 95% threshold). Identity partition
(Q_1=32), zero compression at all tolerances (min D_1=0.099), algebraic
relations 25.9% (idempotence 37.9%, absorption 20.7%, commutation 19.1%).

**Licensed sentence (Codex-corrected):** In the tested Qwen3-1.7B-Base
interface, semantically cofinal assignment histories induced distinct
conditional binary response profiles; because baseline competence was 87.5%
and the principal law comparisons were not position-matched, this does not
yet prove that correctly executed overwritten facts remain causally active.

**CRC verdict (combined CRC-0 + CRC-1, Codex strategic review):** Two
experiments, two task families, two observers. Both identity partitions.
CRC-1 removed CRC-0's strongest design excuses (fine observer, root-dependent
actions, no guaranteed collisions) and still landed far from compression.
CRC on Qwen3-1.7B-Base is **closed** as a discovery engine. Retained only as
a downstream assay if a future method discovers a plausible synchronizer.

**Codex adversarial corrections (adopted verbatim):**
1. "Competence failure makes it more decisive" is scientifically wrong —
   invalid data cannot strengthen claims about competent computation.
2. Normalized {0,1} is not "the coarsest possible observer."
3. Q_1=Q_2=32 is trivial, not meaningful tower stabilization.
4. Algebraic relation rates are confounded by length, position, competence.

**Cross-framework convergence:** CRC-1 (Qwen3-1.7B, Python) partly replicates
predictive_setter_algebra_v1 (Qwen3-0.6B, NL) primacy findings. Extends that
failure class to 1.7B and a probabilistic binary observer.

**Competence position analysis:** 100% correct when queried register's last
write is the final line, 75% when it's not. All 8 failures predict the
overwritten value. Positional structure, not yet causally localized.

**Next: CEG-1 (Causal Erasure Graph v1).** Characterize the causal support of
path dependence via attention mask interventions. Where does dead information
live: direct KV access, relay into later states, or no legal synchronizer?
16 histories × 6 intervention arms × 2 channels. ~800-1000 CPU forwards.
Codex design: `.codex_post_crc1.txt`.

---

## CRC-0 result (2026-09-04; interpretation revised by audit #41, 2026-09-05)

The historical CRC-0 panel (Qwen3-1.7B-Base, 36 roots, words ≤ 2 over
{N, H, R}; numbers in experiments/EXPERIMENTS.md and
docs/STATE_ARCHIVE_closed_lines.md) had no distinct same-variable
W_H-close pairs at τ = 0.01, giving no positive preservation evidence. The
recorded closure score had zero coverage and did not test successor W_H
profiles. Two retained same-variable witnesses violate immediate-tolerance
preservation under H. These finite numerical summaries establish neither
exact quotient stabilization nor absence of compact structure elsewhere.
Earlier wording ("no closure", "tower stabilizes", "apparatus works",
"almost compressible", task family exhausted) is withdrawn.

## Archived claims

Every section below this point in the original STATE.md was moved verbatim to
docs/STATE_ARCHIVE_closed_lines.md (2026-09-05). Index (name, dates, verdict
word as written in each section header):

- Prior campaign: Continuation-Refinement Calculus (2026-09-04)
- Prior campaign: Intensional Descent Criterion (2026-09-03 — 2026-09-04, CLOSED)
- Frozen-residual measurement lane (2026-09-01, CLOSED)
- Real-model line 2026-08-29 (historical record; all constructions closed under audits #27–#31)
- Direction (set 2026-08-27); Guiding question (Devansh, 2026-08-27); Live question; Constraints
- Prior program (closed 2026-08-27)
- Closed toy program — licensed wording (Rounds 36–37; ENDED 2026-08-29)
- NLM-007 — CLOSED (2026-08-29; audit #22 closing statement, verbatim)
- NLM-007 — adoption-order record (Rounds 1–34; superseded orders kept verbatim), including the NLM-001 through NLM-006b verdict/lock sections, Rounds 10–33, audits #2–#20, and the four-cell evidence gate (2026-08-27 — 2026-08-29)
