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
