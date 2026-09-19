# HP-F: The Overlap-Contrast Theorem (review-8 lead arc)

Status: OPENED 2026-09-21 after review-8 (ledger droid_review_2hr_7).
Motivation: the two-family observation (nested = Fisher regime; disjoint =
CGF regime) is construction-confounded — the shared-root floor bounds the
nested family's likelihood ratios regardless of overlap per se. This arc
replaces the qualitative "disjointness" with a quantitative parameter and
proves the actual gating functional.

## 1. The parameter

Per-context contrast (max log-likelihood ratio within a context):
  Lambda_i = max_j log P(j|i) - min_j log P(j|i);   Lambda = max_i Lambda_i.
Candidates for the theorem functional: Lambda itself, or a Renyi-type
averaged version D_1/2 or D_inf across context pairs.

## 2. Conjecture (to prove or refute)

The Fisher (second-order) compression law is accurate on a kernel family
iff the OPTIMAL residual's exponential-moment structure stays controlled,
which is gated by the family's contrast functional — bounded Lambda keeps
residuals in the quadratic regime; the CGF regime requires (and is entered
when) the compression must drop directions carrying contrasts of size
comparable to Lambda.

## 3. The falsifier family (pre-registered design BEFORE any run)

Overlapping, full-support kernels with DIVERGING Lambda:
  P(j|i) = (1-eps) * unigram(j) + eps * uniform-on-top-m(i),
m(i) = the m highest-probability associates of i, m shrinking with the
parameter (m -> 1 limit: near-deterministic preferences, FULL support
retained via the (1-eps) unigram floor). Lambda ~ log((eps/m + (1-eps)p)/
((1-eps)p_min)) -> infinity as m shrinks despite eps-level overlap
everywhere. Prediction to freeze: at fixed eps = 0.1, sweeping m in
{32, 8, 2}: the far-below-knee measured/Fisher ratio grows monotonically
and exceeds 2 at m = 2 (the CGF regime reached WITHOUT disjointness). If
ratios stay <= 1.4 for all m, overlap itself (not contrast) is the gate —
also decisive, in the opposite direction.

## 4. Obligations

(i) Prove the Fisher-approximation bound in terms of Lambda (or the Renyi
functional): excess bound |true excess - Fisher prediction| <= f(Lambda,
residual scale) with explicit f.
(ii) Run the pre-registered falsifier sweep (cheap; single kernel family,
the existing compression harness).
(iii) Integrate with the CGF machinery: the tilted fixed point's onset
condition expressed through the kernel's Lambda.
(iv) Out-of-family forward test (review-8 move 3): predict a real
co-occurrence kernel's gauge-safe excess-CE curve BEFORE measuring
(corpus assets staged), with Lambda measured from the kernel itself.

## 5. Residual-level remainder bound (review-9 repair, step 1; verified)

From the exact CGF identity and the pointwise Taylor inequality
0 <= e^t - 1 - t - t^2/2 <= |t|^3 e^{max(t,0)}/6, the compression remainder
satisfies (per row, after mean-centering corrections):

  | KL - Var_q(delta)/2 | <= (1/6) E_q[ |delta - E_q delta|^3
       e^{(delta - E_q delta)^+} ] + mean-shift terms.

VERIFIED at the settled s=1 optimum: R = KL - Var/2 = -0.0107 (negative —
the remainder subtracts, consistent with cumulant cancellation), bound =
0.1014 >= |R| with 9.5x slack. The bound lives ENTIRELY on the residual's
own norms — no kernel-level parameter appears. The identifiability question
(review-9 move 1, second half): can ANY kernel-level functional (Lambda,
Fisher spectrum, Renyi) control this residual quantity across kernels?
Matched-kernel construction (equal Lambda and equal Fisher spectrum,
different CGF tails) is the decisive test — if such kernels exist, the
compression regime is residual-level state, not kernel-level parameter,
and the gate must be stated on the OPTIMIZATION's solution, not the task.
