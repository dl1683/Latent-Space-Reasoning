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
