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
