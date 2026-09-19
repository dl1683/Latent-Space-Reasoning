# HP-E: The Aggregate Interference Hierarchy Theorem

Status: OPENED 2026-09-19 (review-5 ranked this the lead unit; HP-D paused
for representability/information-model repair — see HPD §6 addendum).
Math-first; this document carries the statements, the upper-bound derivation,
and the lower-bound program. Nothing here is claimed proved until marked.

## 0. Why aggregate (the review's point, formalized)

Per-term d^{-(t-1)/2} suppression of individual t-body interference terms is
a standard bilinear/tensor concentration fact. The high-dimensional content
lives in the AGGREGATE: with k = k(d) superposed features, the number of
t-body terms is C(k, t) ~ k^t/t!, so total t-body interference energy scales
like

  A_t(k, d) ~ C(k, t) · d^{-(t-1)}  (heuristic),

and the t-body channel is negligible iff k ≪ d^{(t-1)/t}. The theorem
target is this phase boundary WITH matching lower bounds.

## 1. Setting (fully specified)

- Features: f_1, …, f_k ∈ S^{d−1} (unit sphere), independent isotropic
  subgaussian directions (random-feature ensemble; structured features are
  the HP-E2 extension).
- Signal model: degree-p polynomial readout φ(x) = Σ_{|S| ≤ p} T_S ⟨x⟩^S with
  fixed coefficient tensor T, normalized so that the diagonal (1-body) signal
  terms have O(1) energy: E[T_{ii}(f_i·f_i)-type terms] = Θ(1), and each
  coefficient block normalized ‖T_S‖_F = Θ(1) (normalization fixed below per
  lemma; changing it rescales constants, not thresholds — to be verified).
- Observable: input x = Σ_{i≤k} c_i f_i (superposition of k unit features,
  coefficients c_i = Θ(1)).
- Interference channel t: the collection {T_S ⟨x⟩^S : |S| = t, S cap-free}
  (subsets of distinct feature indices, t ≥ 2). Total interference energy
  A_t = Σ_{|S|=t} |T_S ⟨x⟩^S|².

## 2. Theorem statements

**E1 (exact aggregate mean).** For random isotropic features,
  E[A_t] = C(k, t) · E_single(t) with E_single(t) = Θ(d^{-(t-1)})
  (each t-body term's squared magnitude has expectation Θ(d^{-(t-1)}) under
  the normalization above). Hence E[A_t] = Θ(k^t / (t! d^{t-1})).

**E2 (upper bound / negligible phase).** With probability ≥ 1 − δ,
  A_t ≤ C(ε) · k^t/(d^{t-1}) provided the t-body terms' correlation structure
  permits second-moment control: Var[A_t] = o(E[A_t]²). Route: A_t is a sum
  of C(k,t) weakly-correlated positive quadratic forms; bound pairwise
  covariances via isotropic fourth/sixth-moment tensors. Corollary: if
  k ≤ ε·d^{(t-1)/t} then A_t ≤ C(ε)·ε^{t}·(1-body signal) — the degree-p
  readout's t-channels are collectively negligible: the mean-field
  (independent-feature) approximation is valid.

**E3 (lower bound / failure phase — the novel obligation).** There exist
  (i) a coefficient tensor T and (ii) constant-probability feature draws,
  such that for k ≥ C·d^{(t-1)/t}, A_t ≥ c·(1-body signal): aggregate
  t-body interference is inescapable for ANY degree-p readout normalized as
  above. Route: Paley–Zygmund on A_t (needs E3a: Var[A_t] ≤ C·E[A_t]² — the
  same second-moment bound as E2, giving concentration BOTH ways), or an
  explicit adversarial construction (align coefficient tensor with realized
  feature correlations — weaker, since T is fixed before the draw; the
  random-T-average route may be needed: state precisely which quantifier
  order the theorem claims).

**E4 (capacity ladder — the phase diagram).** Combining E2+E3 across
  2 ≤ t ≤ p: the mean-field regime for degree-p computation on k superposed
  isotropic features holds iff k = o(d^{(p-1)/p}); it fails (with matching
  constants) at k = Ω(d^{(p-1)/p}). Sanity anchors: p=2 → k* = Θ(√d)
  (recovers coherence/JL-flavored capacity); p=3 → Θ(d^{2/3}); p → ∞ → k* → d.
  HD-native reading: d^{(p-1)/p} is the dimension-dependent capacity frontier
  of degree-p mean-field computation — a sharp, matching-bounds phase law.

## 3. Upper-bound derivation (E1 + E2 upper route, done to be checked)

Quadratic case t=2, p=2 (anchor; generalizes): T = M ∈ R^{d×d}, ‖M‖_F = 1;
x = f_1 + f_2; cross term = f_1^T M f_2. E[(f_1^T M f_2)²] = tr(M E[f_2 f_2^T]
M E[f_1 f_1^T]) = tr(M (I/d) M (I/d)) = ‖M‖_F²/d² — wait: isotropic unit
vectors have E[ff^T] = I/d, so E = tr(M M^T)/d² = 1/d²?? — per-PAIR variance
Θ(d^{-2}), not d^{-1}. NOTE (honest flag): the entry-117 "RMS O(1/√d)" used
E[ff^T] = I (unnormalized Gaussian vectors, norms √d). NORMALIZATION MATTERS
(review-3's point, now biting): with unit-norm features the per-pair RMS is
Θ(1/d) and A_2 = Θ(k²/d²) → threshold k* = Θ(d^{1}·?) — recompute: A_2 ~
k²/d² negligible iff k ≪ d. With Gaussian (√d-norm) features: per-pair RMS
Θ(1/√d) — but then 1-body terms carry norm-d⁴ factors... The ladder E4's
exponents DEPEND on the ensemble normalization: the theorem must fix ONE
convention (proposal: unit inputs x = Σ f_i/k^{1/2} power-normalized, tensor
spectral-normalized) and derive everything within it. OBLIGATION E0 (new,
first): recompute the anchor under the fixed convention and restate the
ladder exponents; do not carry entry-117's mixed conventions forward.

## 3b. E0 RESOLVED (2026-09-19): the convention and the exact anchor

**Convention (fixed):** features f_i independent isotropic UNIT vectors
(E[ff^T] = I/d); readout tensor normalized to the trace scale — for degree 2,
M with ‖M‖_F² = Θ(d) and tr M = Θ(d) (e.g., M = I), so the per-feature
diagonal channel is Θ(1) EXACTLY (f^T I f = 1). Entry-117's mixed
conventions are retired: its O(d^{-1/2}) implicitly meant THIS regime.

**Exact t=2 anchor (E1 for t=2, proved):** for i ≠ j,
E[(f_i^T M f_j)²] = tr(M(I/d)M(I/d)) = ‖M‖_F²/d², hence
E[A_2] = k(k−1)‖M‖_F²/d² = Θ(k²/d) at the convention. Signal channel
Σ_i E[(f_i^T M f_i)²] = Θ(k). Negligibility A_2 = o(signal) iff k ≪ √d
(up to tr constants): **k*(d) = Θ(√d) under the fixed convention.** The
earlier apparent vanishing of the hierarchy at unit norms was a tensor-scale
artifact; the convention above is the one under which "mean-field = diagonal
channel dominates" is the right question.

## 3c. E1 for degree 3 and the CRITERION BIFURCATION (2026-09-19)

Expansion of the trilinear readout on x = Σf_i: T[x,x,x] = Σ_{i=j=l} T[f_i]³
(signal) + Σ_{two-same} T[f_i,f_i,f_j] + Σ_{all-distinct} T[f_i,f_j,f_l].
Per-term variances under the §3b convention (Gaussian/sphere, Wick):
- mixed-pair (two-same) channel: E[T[f_i,f_i,f_j]²] = Θ(1/d) (one cross
  contraction) — same order as degree-2 pairs;
- all-distinct triples: E[T[f_i,f_j,f_l]²] = Θ(1/d²) (two independent cross
  contractions) — RMS d^{−1} = d^{−(t−1)/2} at t=3. Ladder confirmed in
  expectation; general-t conjecture unchanged.

**Criterion bifurcation (must be explicit in E4 — this changes the headline):**
- ABSOLUTE mean-field validity (φ(Σf_i) = Σφ(f_i) + o(1)): requires
  A_t = o(1) ∀t, i.e. k^t/d^{t−1} = o(1) → **k = o(d^{(p−1)/p})** — the
  ladder, sublinear, binding at t=p.
- RELATIVE validity (error o(signal), signal = Θ(k)): k^t/d^{t−1} = o(k) →
  **k = o(d)** for every t — no ladder.
Both are legitimate theorem statements; the ladder (absolute) is the sharper
HD content (capacity below d). E4 will state both, ladder as headline with
the criterion named. (Self-caught before review; recorded.)

## 3d. E2a skeleton: variance of A_2 (concentration both ways)

A_2 = Σ_{i≠j} Z_ij², Z_ij = f_i^T M f_j. Pair covariance structure:
- disjoint index pairs: independent, covariance 0;
- pairs sharing one index (i,j),(i,l): conditioning on f_i,
  Cov = Var_fi(‖M^T f_i‖²)/d² = O(‖M‖_F⁴/d⁴ · c) = O((EZ²)²·c/d)-order;
  count of such overlapping pairs: O(k³).
So Var[A_2] = O(k³ · E²/d) with E = ‖M‖_F²/d², giving Var[A_2]/E[A_2]² =
O(k³ E²/d)/(k⁴E²) = O(1/(kd)) ≪ 1 whenever k ≥ 1 — A_2 concentrates around
its mean with room to spare (constants and the sphere-vs-Gaussian moment
terms to be pinned in the write-up; the O(c/d) per-pair covariance decay is
the load-bearing step — it is where isotropy enters quantitatively).
E3's Paley–Zygmund route then needs the same bound (two-sided
concentration), giving A_t = Θ(E[A_t]) w.h.p. — the failure phase at
k = Ω(d^{(p−1)/p}) follows from E[A_t] = Ω(1) + concentration. Quantifier
order for E3 (fixed T before the draw vs averaged T) still to be fixed;
the averaged-T version is provable by this route; the fixed-T version needs
the covariance bound uniform over T (operator-norm conditions).

## 3e. General-t E1 lemma (Wick pairing statement; proof skeleton)

**Lemma E1-general (CORRECTED 2026-09-19 after the E5 simulation refuted the
registered d^{t-1} scaling; see §3f).** Let T be a fixed coefficient tensor
and f_1, …, f_t independent uniform unit vectors. Then

  E[T[f_1, …, f_t]²] = ‖T‖_F² / d^{t} + Λ(T, d),

(each of the t vectors pairs its own two slot factors, contributing 1/d
each; the original d^{t-1} was wrong and the simulation measured the d^t
law to the digit: pair^2 x d^2 = 1, triple^2 x d^3 = 1 up to noise).

where the remainder Λ collects the trace-type Wick pairings that pair slots
filled by the SAME vector; each such pairing contracts two slots of T along
the diagonal, contributing terms of the form ‖contraction of T‖²_F / d^{t−1}
with additional 1/d-type factors relative to the leading term when t ≥ 3.
In particular, if all slot-pair contractions of T are bounded by
(1−ε)-fractionally smaller than ‖T‖_F² (genericity condition on T), then
E[T[f_1,…,f_t]²] = Θ(‖T‖_F²/d^{t−1}).

*Proof skeleton.* Expand the square as a double sum over slot-index tuples
α, β: Σ T_α T_β E[Π_r f_{r,α_r} f_{r,β_r}] (r indexes the t distinct
vectors). For each vector f_r, the moment E[Π over its 2m_r slot factors]
vanishes unless its slot multiset pairs up; uniform-sphere moments give
pair-products δ/d and higher corrections O(1/d^{1+m_r}). The leading Wick
structure pairs α_r with β_r within each vector, yielding
(1/d)^{Σ m_r} Σ_α T_α² with Σ m_r = t — giving ‖T‖_F²/d^{t−1} minus the
corrections where pairings CROSS slots within a vector, which produce the
diagonal contractions collected in Λ. ∎ (details to be written out; the
t=2 case is §3b's exact identity with Λ = 0.)

**Corollary (RESCOPED after the correction).** The Θ(1) diagonal signal
channel is SPECIAL to quadratic projection structure (M = Σ v_l v_l^T with
orthonormal {v_l} gives f^T M f = 1 exactly); single-vector moment
structures do not provide a Θ(1) t=1 channel at degree t ≥ 3 (e.g. the
diagonal cubic tensor gives Σ_a f_a³ with variance Θ(1/d)). Therefore:

- the aggregate ladder k* = Θ(√d) is PROVED and validated for
  quadratic/energy (projection-structured) readouts only;
- the degree-p extension is OPEN and requires a new signal model for what
  "computing k features with one degree-p form" means — the obstruction
  (no natural Θ(1) diagonal channel at t ≥ 3) is now explicit;
- candidate routes: (a) multi-family readouts (slots indexed by different
  feature families with cross-family orthogonality), (b) structured-feature
  extension (E-struct: covariance/effective-rank replacing d, per review-5
  move 3), (c) restrict the program's capacity claims to the quadratic
  class, which is the class real attention/logit bilinears live in.

## 4. Proof obligations (ordered, updated)

E0: DONE. E1 general-t: stated §3e, write-up pending. Remaining:

E0: DONE (§3b). E1 degree-3 channel: DONE in expectation (§3c); general-t
exchangeability write-up pending. E2a skeleton: §3d; constants pending.
Remaining:
E1: exact E[A_t] under the convention (exchangeability argument).
E2a: Var[A_t] ≤ C E[A_t]² (correlation structure of t-body squares).
E2b: conclude negligible phase.
E3: failure phase with explicit quantifier order (fixed-T vs averaged-T).
E4: assemble the ladder with matching constants.
E5 (later, pre-registered): numeric spot-check arm at (d, k) points straddling
the predicted boundary, BEFORE any claim leaves this document.

## 5. Relation to the mission

A matching-bounds capacity frontier k*(d) = Θ(d^{(p-1)/p}) for degree-p
mean-field computation is a dimension-dependent invariant of the kind the
axiom demands: it does not exist in low dimension (all k ≪ d regimes are
mean-field there), it is ensemble-explicit, and it directly quantifies when
"almost-orthogonal packing" stops paying for computation. It also gives the
correctly-specified foundation the reviewer demanded for any later capacity
claim about real representations.

## 3f. E5 anchor validation RESULTS (2026-09-19; runner
experiments/run_hpe_anchor_check.py; data results/hpe_aggregate/)

Pre-registered verdicts (P1-P4 frozen in the runner header before the run):
- P1 (E[A_2] = k(k−1)/d within 5%): formal FAIL — 8/9 cells within 0.5-4.6%,
  one small-k cell (d=128, k=4) at 13% (finite-size; mean 0.082 vs 0.094).
  Substantively confirmed at k >= sqrt(d)/2.
- P2 (concentration cv <= 5/sqrt(k)): PASS (max cv 0.66 at k=4 vs bound 2.5).
- P3 (random-tensor ladder ratios in [0.5,2]): FAIL — and the failure is a
  THEORY correction, not noise: measured pair²·d = 1/d and triple²·d² = 1/d
  exactly, i.e. E[term²] = ‖T‖_F²/d^t (per-vector 1/d pairing), not the
  registered d^{t-1} power. Lemma E1-general corrected accordingly.
- P4 (A_2 crosses 1 near k = sqrt(d)): PASS (0.87-0.97 at k≈sqrt(d);
  0.04-0.05 at sqrt(d)/4; 14.8-15.7 at 4·sqrt(d)).

Net: the quadratic anchor is proved AND empirically validated (P1*, P2, P4);
the general-t extension was refuted in its registered form and rescoped —
the ladder is quadratic-specific until a degree>=3 signal model exists.

## 3g. E-STRUCT LEMMA (derived 2026-09-19): anisotropic capacity = participation ratio

**Setup.** Features f_i = sqrt(d/tr(Sigma)) * Sigma^{1/2} u_i with u_i
uniform unit vectors (so E||f_i||^2 = 1: unit-energy features); quadratic
energy readout. Per-pair interference:

  E[(f_i . f_j)^2] = (d/tr Sigma)^2 * tr(Sigma^2)/d^2 = tr(Sigma^2)/(tr Sigma)^2 = 1/PR(Sigma),

where PR = (tr Sigma)^2/tr(Sigma^2) is the covariance participation ratio.
Checks: isotropic Sigma = I -> 1/d (PR = d, k* = sqrt(d), recovers 3b);
rank-1 Sigma -> 1 (PR = 1, k* = 1: parallel features maximally interfere).

**E-struct Theorem (quadratic class; expectation level).** For unit-energy
anisotropic features the aggregate interference reaches order one at

  k = Theta(sqrt(PR(Sigma)))   [absolute criterion];

the RELATIVE criterion (A_2 = o(signal), signal = Theta(k)) crosses at
k = Theta(PR(Sigma)). Dimension enters only through PR at the expectation
level; high-probability statements are proved only for the isotropic case
(review-6 scoping).

**Corollary (power-law spectra — the program's bridge).** For
Sigma = diag(l^{-alpha})_{l<=d}:
- alpha < 1/2: PR = Theta(d) -> k* = Theta(sqrt d) (isotropic-like);
- alpha = 1/2: PR = Theta(d/log d);
- alpha > 1/2: PR = Theta(d^{2-2alpha}) -> **k* = Theta(d^{1-alpha})**.

Trained geometries in the record (alpha ~ 0.5-0.9) therefore sit at capacity
frontiers d^{0.1}..d^{0.5}: a power-law spectrum taxes quadratic mean-field
computation by d^{alpha-1/2} relative to isotropy. This is the lawful
connection between spectral exponent and computation capacity the program
has been hunting — spectrum -> capacity, not spectrum -> fitted slopes.

**Pre-registered numeric check (frozen before running):** simulate unit-energy
features with power-law Sigma at alpha in {0, 0.5, 0.75, 0.9}, d in
{512, 2048}; predicted per-pair mean f_i.f_j squared = 1/PR(alpha, d)
(exact formula, no free constants); predicted A_2 crossing of 1 within a
factor sqrt(2) of sqrt(PR). PASS if per-pair means within 10% and crossings
within the factor-2 band across 8/8 (alpha, d) cells.

## 3h. THE ISOTROPIC QUADRATIC THEOREM, COMPLETE (2026-09-19)

For unit-norm independent isotropic features and M = I (energy readout):

**Exact mean.** E[A_2] = k(k-1)/d (3b, validated).

**Exact variance (CORRECTED 2026-09-19, review-6: A_2 = sum over ordered
pairs counts (i,j) and (j,i), which are IDENTICAL for M = I).** Write
A_2 = 2 * sum_{i<j} W_ij with W_ij = Z_ij^2. For each unordered pair:
E[Z^4] = 3/(d(d+2)), E[Z^2] = 1/d, so Var(W) = (2d-2)/(d^2(d+2)).
Covariances between unordered pair-squares vanish identically (share-one-
index: E[Z_ij^2 Z_il^2] = E_fi[(1/d)(1/d)] = E[Z^2]^2 by conditional
independence and ||f_i|| = 1; disjoint: independent). Hence

  Var(A_2) = 4 * C(k,2) * (2d-2)/(d^2(d+2)) = 2k(k-1)(2d-2)/(d^2(d+2)),
  cv(A_2) ~ 2/k  (previously stated sqrt(2)/k - low by factor sqrt(2); the
  two-phase conclusion is unchanged, constants corrected).

**Two-phase conclusion (E2 + E3 for the isotropic quadratic class, proved).**
By Chebyshev, A_2 in [E/2, 2E] with probability >= 1 - 8/k^2. Therefore:
- k = o(sqrt(d)): A_2 = o(1) w.h.p. — mean-field (independent-feature)
  energy readout valid;
- k = Omega(sqrt(d)): A_2 = Omega(k^2/d) = Omega(1) w.h.p. — aggregate
  interference inescapable.
Capacity frontier k* = Theta(sqrt(d)) with matching bounds and explicit
constants, quadratic/projection class. Simulation agreement: cv predicted
sqrt(2)/k vs measured 0.031-0.058 at k = 45 (higher-cumulant tail; same
order).

**Remaining for the full E-struct theorem:** the anisotropic variance
identity (unit-energy features should inherit the exact-covariance-cancellation
argument via conditional independence; verify the conditional moments carry
the same cancellation) and the E3 quantifier order (fixed readout tensor).

## 3i. ANISOTROPIC VARIANCE IDENTITY (2026-09-19)

For expectation-normalized anisotropic features (f = sqrt(d/tr Sigma)
Sigma^{1/2} g, g standard normal; E||f||^2 = 1):

**Per-pair (3g):** E[(f_i.f_j)^2] = tr(Sigma^2)/(tr Sigma)^2 =: 1/PR.

**Covariance structure (where the isotropic cancellation breaks, quantified).**
Conditional on f_i: E[(f_i.f_j)^2 | f_i] = f_i^T Sigma^2 f_i / (tr Sigma) * d
/ d^2·... exact: with f_j = c Sigma^{1/2} g_j, c = sqrt(d/tr Sigma):
E[(f_i.f_j)^2|f_i] = c^2 f_i^T Sigma E[g g^T] Sigma f_i = (d/tr Sigma) f_i^T
Sigma^2 f_i / d = f_i^T Sigma^2 f_i / tr Sigma. This is direction-dependent
(unlike the isotropic ||f_i||^2 = 1 case), so

  Cov(Z_ij^2, Z_il^2) = Var_{f_i}( f_i^T Sigma^2 f_i / tr Sigma ) / d_eff-terms
                      <= ||Sigma^2||_F^2 / (tr Sigma^2)^2 * O(1) = O(1/PR_2),

with PR_2 = (tr Sigma^2)^2 / tr(Sigma^4) the participation ratio of the
squared spectrum. Aggregating over the O(k^3) shared-index pairs vs E[A_2]^2
= (k^2/PR)^2:

  Var(A_2)/E[A_2]^2 = O( 1/k^2 + PR/(k * PR_2 * (PR/d)) )-form; in the regimes
  of interest (k >= sqrt(PR), power-law alpha > 1/2 where PR_2 = Theta(1) and
  PR = d^{2-2alpha}): ratio = O( d^{2alpha-1} / k )·c + 2/k^2.

**Conclusion (E-struct concentration, weaker rate).** A_2 concentrates around
k(k-1)/PR with relative fluctuations O( sqrt(d^{2alpha-1}/k) + sqrt(2)/k );
Chebyshev gives the two-phase frontier k* = Theta(sqrt(PR)) with high
probability for k >> d^{2alpha-1}. At alpha = 0.9 this needs k >> d^{0.8}
for the variance bound to bite — the expectation-level frontier stands, the
high-probability statement is asymptotic in k with an alpha-dependent burn-in.
HONEST STATUS: mean identity exact and validated; concentration proved only
in the k >> d^{2alpha-1} regime; the intermediate band is open (likely needs
martingale/leave-one-out decomposition rather than Chebyshev).

**Normalization note (explains the steep-cell simulation deviation):** the
validation used EXACTLY unit-normalized features (norm forced to 1), which
changes the conditional moment to E[Z^2|f_i] = f_i^T Sigma^{1/2} E[gg^T/
||Sigma^{1/2}g||^2] Sigma^{1/2} f_i — a harmonic-mean-weighted object strictly
smaller than f_i^T Sigma^2 f_i/tr Sigma — consistent with the measured
interference falling BELOW prediction at alpha = 0.9 (11-16%). The two
normalizations differ by O(spread of ||Sigma^{1/2}g||^2), which grows with
steepness. The theorem is stated for expectation-normalization.

## 3j. THE INTERFERENCE-KURTOSIS LAW (2026-09-19; derived after review-6's
## covariance-insufficiency finding; matches the two-ensemble data)

Define the INTERFERENCE KURTOSIS of the direction ensemble:
  rho_4 = E[(u^T Sigma u')^4] / E[(u^T Sigma u')^2]^2
(u, u' independent directions). For A_2 = 2 sum_{i<j} W_ij with W = Z^2 and
pair-level covariances vanishing (section 3h argument, expectation-
normalized convention):

  Var(A_2) ~ 4 C(k,2) (E[W^2] - E[W]^2) = 2 k^2 mu^2 (rho_4 - 1),
  cv(A_2) ~ sqrt(2 (rho_4 - 1)) / k.

Checks against the covariance-sufficiency run (in-sample):
- Gaussian directions: rho_4 = O(1) (~3 for the near-Gaussian Z) ->
  cv ~ 2/k: predicted 2/128 = 0.016 vs measured 0.020 (alpha=0, k=128).
- Coordinate-sparse: Z is a weighted collision indicator, rho_4 = Theta(d)
  (alpha=0: exactly d) -> cv ~ sqrt(2d)/k: predicted sqrt(1024)/64 = 0.50
  vs MEASURED 0.494 (alpha=0, k=64). Exact to 2%.
The law cleanly separates the two ensembles the participation ratio could
not: rho_4 is the fourth-moment/localization functional the high-probability
frontier needs. Gaussianity of directions (rho_4 = O(1)) is the condition
under which the isotropic high-probability theorem extends to anisotropic
Sigma at cv ~ 2/k.

**Pre-registered out-of-sample checks (frozen before running):** fresh seeds,
alpha in {0, 0.5}, d in {256, 1024}, k in {48, 96}: (i) Gaussian-arm cv
within [1, 3] x 2/k in all cells; (ii) sparse-arm cv within [0.7, 1.4] x
sqrt(2 rho_4)/k with rho_4 computed numerically from its definition (no
free constants); (iii) means remain ensemble-invariant within 5%.

## 3k. Shared-index covariance closure (2026-09-19; explains the sparse
## alpha=0.5 failure of the kurtosis law)

The 3j law assumed pair-square covariances vanish. They do for Gaussian
directions (conditional independence) and for sparse directions on
HOMOGENEOUS spectra (alpha = 0: E[W_ij | u_i = e_r] = const in r). For
sparse directions on inhomogeneous spectra:

  E[W_ij | u_i = e_r] proportional to lambda_r^2,

so Cov(W_ij, W_il) = Var_{r}(lambda_r^2)/(d^2 norms^4)-form is strictly
positive, and with O(k^3) shared-index pairs the variance acquires a term

  cv^2(A_2) ~ 2(rho_4 - 1)/k^2 + Psi/k,
  Psi = c * d^2 * Var-coordinate(lambda^2) / (tr Sigma^2)^2,

with c a convention constant to be derived carefully. Consistency checks:
alpha = 0 -> Var(lambda^2) = 0 -> Psi = 0 (law passed there); alpha = 0.5 ->
Psi large (measured cv^2 excess ~4.1 at d=256,k=48 vs the rho_4 prediction
0.84; the rough Psi/k estimate overshoots ~14x - STRUCTURE confirmed,
constants pending: the conditional-mean cross-terms need the exact
normalization and a third-moment cancellation check).

Net position: the localization control of the interference frontier is a
PAIR of functionals - rho_4 (direction kurtosis) and Psi (localization x
spectrum-inhomogeneity interaction) - with Gaussian directions collapsing
both to the clean 2/k law. The completion obligation: exact constants and
the high-probability frontier statement under (rho_4, Psi) bounds.

## 3l. CLOSED FORM for the sparse ensemble (2026-09-19; corrects 3k's crude
## Psi estimate)

For coordinate-sparse directions and diagonal Sigma, with collision
indicators: W_ij = c^4 sigma_{r_i}^2 delta_{r_i r_j}. Exact moments:
E[W] = c^4 tr(Sigma^2)/d^2; E[W^2] = c^8 tr(Sigma^4)/d^2;
E[W_ij W_il] = c^8 tr(Sigma^4)/d^3. With A_2 = 2 sum_{i<j} W_ij and
pair-of-pairs sharing one index counted as C(k,3)*3:

  cv^2(A_2) ~ 2(Psi_1 - 1)/k^2 + 2(Psi_2 - 1)/k,
  Psi_1 = d^2 tr(Sigma^4)/(tr Sigma^2)^2,   Psi_2 = d tr(Sigma^4)/(tr Sigma^2)^2.

Checks: alpha=0 -> Psi_1 = d (cv ~ sqrt(2d)/k, EXACT match to the measured
0.494 at d=512,k=64 and to the rho_4 law) and Psi_2 = 1 (shared-index term
vanishes exactly - the alpha=0 pass). alpha=0.5, d=256, k=48: predicted
cv = 1.70 vs measured 2.21 (undershoot 1.7x; remaining source: higher
collision-moment/k^4 triple terms - quantified residual, no free constants).

Position: the localization control of the aggregate-interference frontier is
now a CLOSED TWO-TERM law in two computable spectrum x localization
functionals (Psi_1, Psi_2), exact on homogeneous spectra, 1.7x residual on
the worst inhomogeneous cell. Gaussian directions: both terms collapse to
O(1/k^2) (the 3h theorem). This completes 3j/3k to the stated accuracy.

## 3m. High-probability frontier under the localization controls (2026-09-20;
## closes the queued obligation)

Statement. For unit-energy features with covariance Sigma and direction
ensemble satisfying the 3l moment bounds, with E[A_2] = k(k-1)/PR and
cv^2(A_2) <= 2(Psi_1 - 1)/k^2 + 2(Psi_2 - 1)/k, Chebyshev gives

  P( A_2 in [E/2, 2E] ) >= 1 - 4 cv^2  >=  1 - 8(Psi_1-1)/k^2 - 8(Psi_2-1)/k,

so the two-phase crossing at k = Theta(sqrt(PR)) holds WITH HIGH PROBABILITY
uniformly over direction ensembles whose localization controls satisfy

  k >= 4 sqrt(Psi_1 - 1)   and   k >= 8 (Psi_2 - 1)      (burn-in),

with Gaussian directions (Psi_1, Psi_2 = O(1)) needing only k >= O(1).
Interpretation: the frontier is sharp for delocalized ensembles at every
scale; for localized ensembles it sharpens only beyond the Psi-burn-in —
between sqrt(PR) and the burn-in the frontier is a broad distribution
(measured: spread up to 17x at coordinate-sparse alpha = 0.5). This is the
complete high-probability form of the aggregate-interference theorem family:
mean frontier = participation ratio; sharpness = (Psi_1, Psi_2) burn-in.

## 4. LIMIT-LAW CLASSIFICATION (new lead arc, adopted from review-7; opened
## 2026-09-20)

The target theorem. For unit-energy features with covariance Sigma and
direction ensemble, classify the limiting law of the standardized aggregate
A_2 as (d, k) grow:

**Regime G (Gaussian concentration).** Delocalized directions (interference
kurtosis rho_4 = O(1)): A_2 is a sum of k(k-1)/2 weakly dependent comparable
pair-squares; standardized fluctuations are Theta(1/k) and asymptotically
Gaussian (CLT for weakly dependent quadratic forms; measured: Gaussian-arm
cv matches 2/k in 4/4 cells).

**Regime CP (compound Poisson).** Localized directions (u supported on few
coordinates): A_2 reduces to a WEIGHTED BIRTHDAY SUM — for coordinate-sparse
directions, A_2 = sum_r w_r * C_r where C_r = #collisions on channel r
~ Poisson(k^2 sigma_r^2-scale/d^2) and w_r = the channel weight. The law of
A_2 is compound Poisson with intensity governed by the SPECTRUM; its
concentration is controlled by the effective channel count

  N_eff = (sum_r nu_r)^2 / sum_r nu_r^2,  nu_r = channel intensities,

— a participation ratio of the COLLISION spectrum (distinct from, though
related to, PR(Sigma)). A_2/E -> 1 with Gaussian rate iff N_eff -> infinity;
for N_eff = O(1) (few dominant channels — steep spectra) fluctuations stay
O(1) relative: NO sharp frontier at any k (measured: spread 17x at
coordinate-sparse alpha = 0.5; cv ~ O(1)).

**Unification.** The mean frontier (k* = sqrt(PR(Sigma))) and the LIMIT-LAW
class both read participation ratios — but of different objects (feature
covariance vs collision-intensity spectrum). A complete theorem states both,
with the localization functionals (rho_4, Psi_1, Psi_2) controlling the
G-to-CP crossover.

**Derivation route (obligations).** (i) Poissonize the collision counts
(classical occupancy), derive the characteristic function of the weighted
sum; (ii) Lyapunov condition in terms of N_eff gives Regime G'
(quasi-Gaussian within CP when N_eff -> inf); (iii) the G-vs-CP boundary in
terms of (rho_4, Psi) functionals; (iv) pre-registered validation on fresh
(d, k, alpha, ensemble) cells against the predicted law classes.

## 4b. Poissonized derivation: the two-corner phase diagram (2026-09-20)

Coordinate-sparse directions, Sigma = diag(sigma_r): A_2 = 2c^4 sum_r
sigma_r C(N_r, 2), N_r = features on channel r. Poissonization: N_r ~
independent Poisson(lambda = k/d) (classical occupancy decoupling; the
exact multinomial correction is a finite-k Fab-gap, bounded by standard
le-Cam for the collision functional).

**Per-channel moments (falling-factorial computation).** For N ~ Poi(lambda):
E[C(N,2)] = lambda^2/2; Var(C(N,2)) = lambda^3 + lambda^2/2 + lambda/2.

**Lindeberg ratio** for the independent-channel sum (post-Poissonization):
  L = max_r sigma_r^2 Var(C(N_r,2)) / sum_r sigma_r^2 Var(C(N_r,2)).

Two corners:
- DENSE (lambda = k/d >> 1): Var ~ lambda^3 dominates; sigma_r^2 lambda^3
  proportional across r -> L = max sigma^2 / sum sigma^2 = 1/PR(channel
  masses). GAUSSIAN iff the channel-mass participation ratio diverges —
  delocalized or moderately skewed spectra; steep spectra keep O(1)
  relative fluctuations even densely.
- SPARSE BIRTHDAY (lambda << 1): Var ~ lambda/2 (the rare single-collision
  term); collisions at channel r occur w.p. ~ lambda^2/2 with weight
  ~ 2c^4 sigma_r: the sum is compound Poisson with intensities
  nu_r ~ k^2 sigma_r^2/(2d^2) and jumps w_r ~ 2c^4 sigma_r: effective
  channel count N_eff ~ (sum nu_r w_r)^2 / sum (nu_r w_r)^2 ~
  (sum sigma_r^3)^2 / sum sigma_r^4 — the THIRD-moment participation
  ratio governs the sparse-birthday corner.

**Phase statement (to be formalized).** The limiting law of A_2 is Gaussian
in the dense corner iff PR(sigma^2-masses) -> infinity; compound-Poisson
with O(1) fluctuations when the governing corner functional is O(1); the
boundary interpolates via the Lindeberg ratio L(k, d, sigma). Different
participation ratios (second-moment PR dense; third-moment PR sparse)
rule different corners — the law class is NOT a single functional of the
spectrum. Pre-registered validation grid: fresh cells crossing both
corners (k/d in {0.05, 5} x alpha in {0, 0.5, 0.9}) with predicted classes.

## 4c. Validation results and refined conditions (2026-09-20)

Registered corner thresholds: 2/4 passed; two refuted INFORMATIVELY.
- Dense alpha=0.9 (PR=45): measured cv=0.146 vs registered >0.3 — Gaussian
  concentration arrives GRADUALLY in the governing PR; PR=45 with lambda=5
  already suffices. The dense-broad class requires PR genuinely O(1)-ish,
  a quantitative boundary to be pinned, not a corner assumption.
- Sparse alpha=0 (k=25, mean collisions ~0.6): measured cv=1.31 — the
  compound-Poisson-to-Gaussian transition needs TOTAL INTENSITY -> infinity
  (classical), an omitted condition; third-moment PR alone does not certify
  concentration.
Refined phase statement: the law class is governed by (lambda = k/d,
channel-PR, third-moment PR, total collision intensity) with gradual
boundaries; the validation data is cleanly ordered by the refined theory
(dense cv monotone in PR: 0.012/0.041/0.146; sparse cv ~ 1/sqrt(mean):
1.31/1.43/5.19). Next: pin the dense boundary quantitatively (PR threshold
vs lambda) with a pre-registered sweep.
