# Matrix-native information theory (owner insight, 2026-08-01)

Status: **LIVE RESEARCH FRAME.** Owner-originated (Devansh). This document is
canonical context for every future unit and subagent. Read it before proposing
any new measure, objective, or search design.

Labels used per repo method: PROVED / EMPIRICAL / CONJECTURE / SPECULATION.

## 0. The owner's framing (the point of this document)

> "A lot of our math primitives are still number-specific and number-native.
> Assume we didn't know numbers exist at all and all we had are matrices,
> naturally thought in terms of matrices — what would the equivalents of
> Kolmogorov complexity, Shannon entropy, information theory, and Bayesian
> statistics look like?"

The program's tools (rank, charts, wounds, certificates) are inherited from
scalar-native mathematics. The claim under investigation is that the right
primitives for cheaper matrix multiplication are *matrix-native* versions of
the information-theoretic measures, where the composition law is MATRIX
MULTIPLICATION rather than concatenation.

All four classical concepts share one skeleton: objects, a composition law, a
cost additive under composition, and a logarithm converting multiplication to
addition. Changing the composition law from concatenation to matrix product
changes each of them in a specific, computable way.

## 1. Kolmogorov complexity -> bilinear complexity (PROVED, by identification)

K(x) = length of the shortest program producing x. The matrix-native version:
the shortest *arithmetic circuit* producing a bilinear map from primitive
operators, with MULTIPLICATIONS as the cost unit.

This is exactly tensor rank / bilinear complexity. Therefore:

**Tensor rank IS Kolmogorov complexity in the multiply-counting cost model.**
The program's entire hunt is Kolmogorov compression of one specific object.

Open (CONJECTURE): the invariance theorem's matrix analogue. For strings, two
universal machines agree up to an ADDITIVE constant. For matrices, costs
compose multiplicatively, so the expected statement is agreement up to a
MULTIPLICATIVE constant depending on the instruction set. Proving this would
make "intrinsically expensive" a well-defined, algorithm-family-independent
property.

## 2. THE TWO-AXIS COMPLEXITY (the load-bearing new object)

Border rank is Kolmogorov complexity WITH A DISTORTION ALLOWANCE — the
approximate-description version. But approximation is not free: realizing a
border scheme of pole order h to accuracy eps costs ~ h*log(1/eps) bits of
precision.

**CONJECTURE (central):** the honest complexity of a matrix operation is a PAIR

        C(M) = (multiplications, precision bits)

and border rank does not remove cost — it MOVES cost from the first axis to the
second. The right objective is a rate-distortion curve over that pair, not a
single number.

EMPIRICAL support already in this repo:
- the kappa non-attainment law (cancellation grows lawfully as residual falls;
  results/kmw_als_descent_v1/captains_kappa_probe.jsonl);
- the arc law K ~ delta^(-M/h) derived in research/SMIRNOV_NORMAL_FORM_R4.md;
- the valuation cone criterion, whose (h, k, m) data ARE pole/precision data
  (research/VALUATION_CONE_LEMMA.md).

Consequences if true:
1. "Galactic" algorithms are galactic BECAUSE the second axis is ignored.
2. The cheapest USABLE scheme may not be the lowest-rank scheme. The mission's
   objective becomes minimizing (multiplications x precision), a target nobody
   is optimizing.
3. Our border-route results (Smirnov valuation gap, KMW kappa growth) are
   measurements on this curve, not merely failures.

## 3. Shannon entropy -> von Neumann entropy (PROVED, standard)

S(rho) = -tr(rho log rho) on the normalized spectrum (rho = M'M / tr(M'M)).
Non-commutativity is the entire content: eigenvalues supply the "probabilities"
but the BASIS matters.

Key inequality (PROVED, Schur concavity): S(rho) <= H(diag(rho)), equality iff
diagonal. Therefore:

**The ENTROPY DEFECT — how far a matrix sits below its own diagonal entropy —
is exactly its exploitable structure.**

Practical forms:
- effective rank exp(S) = the number of directions actually carrying mass;
  bounds low-rank approximability directly;
- structure-finding = finding the basis/transform minimizing entropy.

This is the computable "fingerprint" for the owner's fingerprint-plus-error
program (see section 6).

## 4. Bayesian statistics -> quantum relative entropy (PROVED, standard)

Prior/posterior over non-commuting hypotheses: rho -> sqrt(E) rho sqrt(E) /
tr(E rho). Bayesian surprise becomes S(rho||sigma) = tr(rho log rho - rho log
sigma), which satisfies the data-processing inequality.

IMMEDIATE PRACTICAL USE (adopt in unit design): every certified dead chart is
an OBSERVATION. Chart selection should maximize EXPECTED INFORMATION GAIN over
a posterior on where schemes live, replacing supervisor judgment with optimal
experiment design. Charts cost seconds; the space is astronomical; the binding
constraint is WHICH question to ask.

## 5. The barrier question (SPECULATION, honestly labeled)

Kolmogorov theory's power is incompressibility by counting: most strings are
complex, provably. Its dark twin is Chaitin incompleteness: a formal system can
prove "K(x) > L" for NO specific x once L exceeds the system's own description
length — you can prove most objects are hard, essentially never a named one.

The analogy to our wall: generic tensors have near-maximal rank by dimension
counting, yet lower bounds for the SPECIFIC matmul tensor stall near 3n^2.

HONEST CAVEAT: this is not literal undecidability. Rank of a fixed tensor over
an algebraically closed field is decidable in principle (quantifier
elimination), though astronomically expensive; decision problems for tensor
rank are NP-hard over finite fields and exists-R-complete over the reals. The
Chaitin analogy is therefore about PROOF COMPLEXITY, not computability.

The real precedent is algebraic-complexity barrier results (e.g. the
no-occurrence-obstructions theorem against the original geometric complexity
theory route). The research question this frame poses:

**Is there a barrier theorem explaining why matmul rank lower bounds resist —
i.e. a proof that a broad class of methods cannot reach beyond ~3n^2?**

That would be a genuine result and it is the honest version of "prove it is
impossible", which the owner asked about directly and which the global bound
itself is far out of reach of.

## 6. Connection to the fingerprint-plus-error program (owner, 2026-08-01)

Owner's parallel idea: map a matrix to a structured family plus an error term,
then multiply fast in the family. Literature anchors: displacement rank
(Toeplitz/Hankel/Cauchy-like), butterfly/Monarch/kaleidoscope matrices (with
CLOSED-FORM nearest-Monarch projection), hierarchical matrices (FMM), and the
Cohn-Umans group-algebra route (the exact, omega-relevant Fourier method).

Cost calculus (PROVED arithmetic): M = S + E costs fast(S) + ~4kn^2 for a
rank-k error against dense 2n^3, so the approach requires k << n. Generic
matrices fail this; PDE operators, covariances and trained weights often do not.

THE REGIME WHERE IT IS UNAMBIGUOUSLY RIGHT: fixed M applied many times
(inference). Preprocessing is then free in the amortized limit, and the
question becomes: given unlimited offline time on a fixed M, what is the
cheapest representation for repeated application at a stated error tolerance?

Under-served slices: (a) DISPATCH — a cheap detector deciding which family M is
near (the entropy-defect fingerprint of section 3); (b) COST-AWARE JOINT
FITTING — minimizing the multiply COST of (S + low-rank + sparse), not the
approximation error.

## 7. Standing instructions for units and subagents

1. Do not propose a new measure without stating its matrix-native form.
2. Report complexity as the PAIR (multiplications, precision), never
   multiplications alone, whenever a border/approximate object is involved.
3. When choosing the next chart/wound/channel, state the expected information
   gain, not only the cost.
4. Treat the entropy defect as the default structure detector before designing
   a bespoke one.
5. Label every claim PROVED / EMPIRICAL / CONJECTURE / SPECULATION, and keep
   the Chaitin analogy explicitly SPECULATION until a barrier statement is
   actually formulated.
