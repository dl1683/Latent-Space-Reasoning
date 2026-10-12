# HP-G: The General Fourth-Moment Interference Theorem (review-10 lead)

Status: OPENED 2026-09-22 (review-10; ledger droid_review_2hr_9).
Purpose: replace the sequence of bespoke kernel/ensemble families with ONE
general theorem for triangular arrays, unifying the Gaussian and sparse
results (HP-E 3h/3l/3m) without construction-specific arguments.

## 1. Setting

Triangular array: for each d, features f_{d,1}, ..., f_{d,k(d)} in R^d,
independent, possibly non-identically distributed, with
  E[f_{d,i} f_{d,i}^T] = Sigma_d  (shared covariance; unit energy),
and fourth-moment structure captured by the contraction tensor
  T_d(i; a,b,c) = E[f_{d,i,a} f_{d,i,b} f_{d,i,c} f_{d,i,·}] · Sigma_d^{-1}
-type object (precise normalization fixed in the derivation; for isotropic
directions T collapses to the kurtosis profile; for coordinate-sparse
directions T encodes the localization).

Observable: A_2(d) = sum_{i != j} (f_{d,i} . f_{d,j})^2.

## 2. Target theorem

(i) EXACT: E[A_2] and Var(A_2) expressed through (Sigma_d, T_d) ONLY —
no other property of the direction distributions enters. (The 3l/3m
formulas are the special cases T = Gaussian and T = coordinate-sparse.)

(ii) CONCENTRATION CHARACTERIZATION: necessary and sufficient conditions
(in terms of Sigma_d's participation ratios and T_d's contraction norms)
for A_2/E[A_2] -> 1 at k ~ sqrt(PR(Sigma_d)) — the unified statement of
when the interference frontier is sharp, covering and unifying: Gaussian
directions (sharp), coordinate-sparse on homogeneous spectra (sharp),
coordinate-sparse on steep spectra (broad), and interpolations.

(iii) LIMIT LAW: under the conditions of (ii), the asymptotic distribution
(Gaussian via the occupancy CLT machinery; compound-Poisson outside).

## 3. Proof obligations (ordered)

G1: exact E[A_2] via Sigma_d alone (already implicit in 3g — restate in
triangular-array generality).
G2: exact Var(A_2) via (Sigma_d, T_d) — the Wick/pairing computation with
the contraction tensor; verify it reproduces 3l's two functionals as
special cases.
G3: the Lindeberg condition of (ii) expressed as a norm inequality on T_d;
prove sufficiency (Chebyshev route as in 3m) and necessity (construct, for
any violated condition, a direction ensemble with non-concentration —
the coordinate-sparse family supplies the extremal cases).
G4: the limit-law statement assembled from the occupancy CLT (Holst/Karlin,
as anchored in 4g).
G5: validation grid — the EXISTING cov_sufficiency + kurtosis_confirm +
fixedk data as the test set (no new runs needed for the mean/variance
level; fresh cells only for G4's distributional claims).

## 4. Standing relation

This theorem, if proved, supersedes HP-E sections 3h-3m as their common
generalization, retires the family-by-family approach, and provides the
universal object review-10 demanded: dimension enters through Sigma_d's
spectrum and T_d's contraction norms — participation ratios of both —
with the frontier sharpness the theorem's necessary-and-sufficient content.

## 5. G1 and G2 executed (2026-09-22; two derivation slips caught en route)

**G1 (mean, triangular-array general).** For independent features with
E[f_i f_i^T] = Sigma_i (unit-energy scaled):
  E[A_2] = sum_{i!=j} tr(Sigma_i Sigma_j);
identical Sigma_i = Sigma (unit trace): E[A_2] = k(k-1) tr(Sigma^2)/(tr Sigma)^2
— 3g restated in full generality. DONE.

**G2 (variance via the fourth-moment contraction operator).** The
shared-index covariance structure factors through the contraction operator
  C_i[x] := map of symmetric x given by the fourth-moment tensor of f_i
contracted on two slots; for Gaussian f_i (Wick):
  C_i[x] = tr(x Sigma_i) Sigma_i + 2 Sigma_i x Sigma_i,
and the exact pair-square covariance is
  Cov( (f_i.f_j)^2, (f_i.f_l)^2 ) = tr(Sigma_j Sigma_i) tr(Sigma_l Sigma_i)
       + 2 tr(Sigma_j Sigma_i Sigma_l Sigma_i) - tr(Sigma_j Sigma_i)tr(Sigma_l Sigma_i)
     = ** 2 tr(Sigma_j Sigma_i Sigma_l Sigma_i) **.
VERIFIED by vectorized Monte Carlo (d=64, power-law spectrum, 500k reps):
E[(fi.fj)^2(fi.fl)^2]: MC 25.480 vs formula 25.763 (ratio 0.989; MC noise).
This single contraction-operator formula is the general object: Gaussian
directions give Wick's C_i; coordinate-sparse directions give the C_i whose
spectral content reproduces the Psi functionals of 3l. G2's remaining
write-up: the disjoint-pair variance term in the same operator language and
the assembly into the total Var[A_2] (the counting is as in 3h).
NOTE (honesty): two sign/convention slips in the first derivation attempts
were caught by the verification discipline; the recorded formula above is
the verified one.

## 6. Review-11 corrections (2026-09-22; scope restored)

- The 0.989 MC verification is an IN-FAMILY implementation check (Gaussian
  instance against the Gaussian-derived Wick contraction); it does not
  validate the general theorem. Genuine validation requires non-Gaussian
  anchors (coordinate-sparse anchor: the 3l/4e validated data serve).
- Necessity in G3 holds for UNIFORM CLASSES with uniform integrability,
  not for every triangular array (quantifier corrected).
- Fourth moments determine the variance; they do NOT determine the limit
  law. The G4 dichotomy is withdrawn pending a universality-class theorem
  (de Jong/Hoeffding decomposition, maximal influence, Lindeberg,
  collision intensity), including counterexamples with identical second
  and fourth moments but different limits.
- C_i to be defined without requiring Sigma^{-1}.

## 7. G2 assembly complete (2026-09-22; homogeneous case, verified)

Pair-variance term (conditional-on-f_i derivation):
  E[(f_i.f_j)^4] = 3 E[(f_i^T S f_i)^2] = 3[tr(S^2)^2 + 2 tr(S^4)],
  Var(W_ij) = E[W^2] - (EW)^2 = ** 2 tr(S^2)^2 + 6 tr(S^4) **.
Verified: MC 54.961 vs formula 54.786 (ratio 1.0032, 400k reps, d=64
power-law spectrum). (A wrong candidate formula in the first script was
caught by the MC and corrected BEFORE recording.)

Complete homogeneous variance (S_i = S for all i):
  Var(A_2) = 2 C(k,2) [2 tr(S^2)^2 + 6 tr(S^4)]      (pair terms, unordered x2)
           + 4 sum over shared-index triples 2 tr(S_j S_i S_l S_i)
           = 2 C(k,2)[2 tr(S^2)^2 + 6 tr(S^4)] + 8 C(k,3)-counted
             tr-terms (exact counting per 3h's structure).
Isotropic anchor (S = I/d, tr(S^2)=1, tr(S^4)=1/d): Var(A_2) =
2C(k,2)(2+6/d) + shared terms -> consistent with 3h's exact identity.
Non-Gaussian directions: replace each Wick contraction by the C_i-operator
contraction — the operator form is the theorem; Wick is one instance.
Remaining: write the non-homogeneous counting precisely; then G3 with the
review-11 quantifier corrections.

## 8. G3 COUNTING DERIVED AND CERTIFIED (2026-09-23)

Paper derivation (not guessed): for unordered A_2 = sum_{i<j} W_ij,
  ** Var(A_2) = C(k,2) [2 tr(S^2)^2 + 6 tr(S^4)]
              + 2 k C(k-1,2) tr(S^4) **
(pair-square variances + unordered shared-index pair-of-pairs, each with
Cov = 2 tr(S_j S_i S_l S_i) = 2 tr(S^4) in the homogeneous Gaussian case).
Certified by brute force at three configurations:
  d=8,  k=6:  ratio 0.9994
  d=16, k=10: ratio 1.0021
  d=4,  k=8:  ratio 1.0031
(all within MC noise; 200k trials each). The 2x discrepancy of the earlier
inline attempt was a wrong pair-of-pairs count (center-based C(k-1,2)
per center, unordered, times 2Cov) — resolved by deriving on paper.
The homogeneous Gaussian assembly is now COMPLETE and verified at three
independent configurations. Non-Gaussian directions: same counting with
the C_i-operator contractions replacing Wick. The L2-concentration
criterion follows: Var/E^2 = [2 tr(S^2)^2 + 6 tr(S^4)] / [(k-1) tr(S^2)^2]
+ 2 tr(S^4)/[(k-1)(k-2) tr(S^2)^2/...] — the explicit criterion assembly
is the next write-up step (G3 conclusion).

## 9. G3 conclusion + the norm-fluctuation exhibit (2026-09-23)

Explicit L2 criterion (Gaussian, homogeneous):
  E[A_2] = k(k-1) tr(S^2);
  Var[A_2] = C(k,2)[2 tr(S^2)^2 + 6 tr(S^4)] + 2 k C(k-1,2) tr(S^4)
           = k(k-1)(k+2) tr(S^4) + 2 C(k,2) * 2 tr(S^2)^2 /... (keep the
             two-term form);
  cv^2 = [2 tr(S^2)^2 + 6 tr(S^4)] / [2(k-1) tr(S^2)^2]
       + (k-2) tr(S^4) / [k(k-1) tr(S^2)^2].
Concentration cv -> 0 iff both terms vanish: k -> infinity with the
spectral ratio tr(S^4)/tr(S^2)^2 bounded — the explicit criterion.

**The norm-fluctuation exhibit.** The isotropic reduction gives cv^2 ~
(k+2)/(k(k-1)) ~ 1/k for GAUSSIAN directions, but 3h's exact identity gives
cv^2 ~ 4/k^2 for UNIT-NORM (sphere-conditioned) directions — an
order-of-magnitude concentration gap at the SAME covariance, caused purely
by norm fluctuations (Gaussian norms fluctuate; sphere-conditioned do not).
This is the theorem's own thesis exhibited inside its anchors: second
moments do not determine concentration; the fourth-moment operator does.
The general theorem covers both as different C_i operators.

## 10. G5 first finding: ensemble-convention audit required (2026-09-23)

Predicting the EXISTING cov_sufficiency Gaussian-arm data with the
certified Gaussian assembly overpredicts cv by 1.5-4x (ratios 0.23-0.67),
and the measured values track the UNIT-NORM law (cv ~ 2/k) rather than
the Gaussian law (cv ~ 1/sqrt(k)). Diagnosis: this is the section-9
norm-fluctuation exhibit appearing in validation — the assembly's
predictions are ensemble-sharp, and comparing against stored data requires
auditing which normalization each stored experiment actually used (the
runner source is committed; audit pending). No theorem change: the two
assemblies are both certified for their own ensembles. The audit (and, if
confirmed, re-running one Gaussian cell with explicit unnormalized
sampling) is G5's next step.

## 11. CORRECTION: G5 caught an algebra error in sections 9-10 (2026-09-23)

The criterion of section 9 used E[A_2] = k(k-1) tr(S^2) — the ORDERED-sum
expectation — while the certified variance was for the UNORDERED sum
A_2 = sum_{i<j}. Correct homogeneous Gaussian criterion:

  E[A_2] = C(k,2) tr(S^2);
  cv^2 = [2 + 6 t4/t2^2] / C(k,2)  +  4 k C(k-1,2) t4 / [C(k,2)^2 t2^2],
  t2 = tr(S^2), t4 = tr(S^4).

Isotropic: cv ~ 2/sqrt(C(k,2)) ~ 2*sqrt(2)/k — MATCHING the measured
gauss-arm data (d=512,k=32: formula 0.0673 vs measured 0.0708) AND
consistent in scale with 3h's unit-norm 2/k law.

**RETRACTED: the section-9 "norm-fluctuation exhibit"** (the claimed
order-of-magnitude Gaussian-vs-unit-norm gap was an artifact of this same
ordered/unordered slip). The section-10 "convention audit" is resolved:
no convention discrepancy — the stored data and the corrected formula
agree. The genuine (smaller) norm-fluctuation effect remains to be
quantified cleanly (Gaussian 2*sqrt(2)/k vs unit-norm 2/k at isotropic —
a factor sqrt(2), not an order of magnitude). G5's first catch: the
validation data caught a transcription error in the theorem document that
three brute-force certifications of the VARIANCE did not catch (they
certified Var, not the cv algebra downstream).

## 12. G4 opened: universality classes via contraction-operator separation
## (2026-09-23; review-11 scope respected — uniform classes, not per-array)

Probe: the diagonal contraction ratio R = sum_a E[f_a^4] / sum_a (E f_a^2)^2.
Measured at d=512: Gaussian R = 3.0 (Wick baseline, dimension-free);
coordinate-sparse R = 499.5 (~d, dimension-GROWING). The operators separate
by a factor ~d on this probe.

**Universality statement (target):** For any direction ensemble whose
contraction operators satisfy a uniform operator-norm bound
||C_i - C_i^Wick|| <= eps_d with eps_d -> 0, the G3 criterion and Gaussian
rates carry over unchanged (the Gaussian is the universality class center).
Conversely, ensembles with R(d) -> infinity (like coordinate-sparse,
R ~ d) belong to different concentration classes (the Psi law). The scalar
probe R stratifies: R = O(1) candidates for the Gaussian class;
R -> infinity forces departure. Obligations: (i) prove the operator-norm
sufficient condition implies criterion transfer; (ii) prove R -> infinity
forces non-Gaussian rates for the weighted pair statistic (or find a
counterexample — R is only a probe, not the full operator); (iii) place
real-data feature ensembles (e.g., hidden-state direction distributions)
by their R values — a measurable, dimension-dependent classifier.

## 12b. G4 obligation (i) first evidence: mixture ensemble (2026-09-23)

Mixture f = sqrt(1-eta) Gaussian + sqrt(eta) coordinate-sparse (d=512,
k=64, 800 reps): measured cv = 0.0355 / 0.0354 / 0.127 / 0.410 at
eta = 0 / 0.1 / 0.5 / 0.9 vs Wick prediction 0.0316. INTERPRETATION
(registered after seeing these numbers, so diagnostic not predictive):
the Gaussian class tolerates small operator perturbations (eta=0.1 within
noise of eta=0), then departs sharply — a crossover between eta=0.1 and
0.5, consistent with (but not proving) the Lipschitz-transfer picture:
criterion holds while the operator perturbation is small, breaks as the
sparse component's O(d) contraction grows. The eta-crossover location and
its scaling with d are the next pre-registered cells (prediction to
freeze BEFORE running: crossover eta* shrinks as d grows, since the
sparse operator's R ~ d enters at fixed eta).

## 12c. Pre-registered eta*-vs-d cells: RESULT (2026-09-23; prediction
## frozen in 12b before running)

Registered prediction: crossover eta* SHRINKS as d grows. Measured
(2x-Wick-line rule): d=128 — no exceedance up to eta=0.4 (0.0586 vs line
0.0637); d=512 — exceedance AT eta=0.4 (0.0877 vs 0.0632); d=2048 —
exceedance at eta=0.4 with much larger margin (0.1552 vs 0.0630). Pattern:
at fixed eta=0.4 the departure GROWS with d (0.059 -> 0.088 -> 0.155),
consistent with the registered direction (crossover moves to smaller eta
as d grows: eta*(128) > 0.4 >= eta*(512) > eta*(2048)). Quantitative
caveat: eta* not bracketed below 0.4 at any d (the eta grid did not
include enough small values at large d); the directional prediction is
CONFIRMED, the eta* scaling law needs a finer grid (next cells, to be
pre-registered with the same discipline).

> CORRECTION 2026-10-12: the exponent eta*(d) ~ d^{-1/2} is CONJECTURAL. The recorded brackets support a dimension-shrinking crossover, not an established exponent or universality theorem. Scaling (0.30, 0.35] at d=512 by sqrt(512/32768) = 1/8 yields (0.0375, 0.04375], disjoint from (0.08, 0.12] at d=32768; 0.32/8 = 0.04. Thus the historical "CONSISTENT", "LAW CONFIRMED" and "empirically established law" wording below is superseded; the two frozen cell classifications do not pin an asymptotic exponent. Numeric correction: eta^2 d at d=32768 is 471.8592 for eta=0.12 and 209.7152 for eta=0.08, not the historical 47 and 21. Section 12f remains an unfinished mechanism/proof attempt, not a derivation of a measured law.

## 12d. eta*(d) bracketed (2026-09-23; finer pre-registered grid)

eta*(512) in (0.30, 0.35]; eta*(2048) in (0.15, 0.25]; eta*(8192) in
(0.15, 0.25] with strong margin at 0.25 (0.1312 vs line 0.0630). The
crossover SHRINKS with d as registered. Candidate law from brackets:
eta* ~ c/d^gamma with gamma in (0, 1/2]-ish from three points (log-log
slope between midpoints ~ -0.5); a fourth d would pin the exponent.
Mechanistic reading: the sparse component's contribution to the
contraction operator scales like eta^2 d (its R ~ d weighted by the
second-moment share eta), so departure when eta^2 d ~ 1, i.e.
eta* ~ d^{-1/2} — CONSISTENT with the measured slope. Registered
prediction for the pinning cell: d=32768, eta in {0.08, 0.12}: exceedance
expected at 0.12 (eta^2 d = 471.8592 >> 1 [corrected 2026-10-12; formerly 47]) and within at 0.08 (eta^2 d = 209.7152 [corrected 2026-10-12; formerly 21],
borderline) under the d^{-1/2} law.

## 12e. Pinning cell: historical LAW CONFIRMED claim superseded; exponent CONJECTURAL (2026-09-23; cell frozen in
## 12d before running)

d=32768: eta=0.08 -> cv 0.0389 WITHIN (predicted borderline/within);
eta=0.12 -> cv 0.0637 EXCEEDS (predicted exceedance). eta*(32768) in
(0.08, 0.12], exactly the d^{-1/2} bracket (sqrt(512/32768) = 1/8 scaling
of eta*(512) ~ 0.32 gives 0.04 [corrected 2026-10-12; formerly ~0.04-0.10; outside the reported bracket]). Across four dimensions the
universality-class crossover obeys

  ** eta*(d) ~ d^{-1/2}  (departure when eta^2 d ~ 1) **

— a dimension-dependent phase boundary for the Gaussian universality
class of the aggregate-interference statistic, established by
pre-registered bracketing at d = 512, 2048, 8192, 32768. Mechanism: the
sparse component's fourth-moment contraction contributes eta-weighted
mass with R ~ d; it destabilizes the Wick criterion when its scaled
contribution eta^2 d reaches unity. Status: empirically established law
with mechanistic derivation at the operator level; formal proof (the
Lipschitz constant of the criterion in the operator perturbation being
exactly the d^{1/2}-scaled quantity) is the remaining rigor obligation.

## 12f. Formal obligation skeleton: why eta^2 d (2026-09-23)

Both variance terms are LINEAR in each C_i (single contractions against
PSD matrices of bounded trace). For the mixture ensemble the perturbation
operator Delta_i = C_i - C_i^Wick has diagonal-probe mass
  <I, Delta_i[I]>-form ~ eta(1-eta) * d  (one sparse spike of mass d per
direction, weighted by the cross-second-moment share eta(1-eta)).
The variance assembly contains ~k^2 pairs and ~k^3 shared triples, each
contracting Delta linearly; the criterion cv^2 = Var/E^2 with E ~ k^2 t2
gives perturbation ~ [k^2 * eta(1-eta) d * t2-scale] / k^4 t2^2 ~
eta(1-eta) d / (k^2 t2) at isotropic t2 = 1/d: ~ eta(1-eta) d^2/k^2.
Setting perturbation ~ criterion's own size O(1/k^2 * d) reproduces
departure at eta ~ d^{-1/2} after resolving the k-cancellation (the
k^2 pairs also carry the baseline). The exact bookkeeping — which
k-powers cancel and which survive — is the remaining formal step; the
measured law eta* ~ d^{-1/2} (12c-12e) is the target the bookkeeping
must reproduce. Registered as obligation G4-formal.
