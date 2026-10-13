# Same Shannon entropy, same atom count, different erasure feasibility

Date: 2026-10-03. Rigor: PROVED AT SCOPE, independent arithmetic/proof PASS in
WEIGHTED_ROBUST_EVIDENCE_REVIEW.md after the depth-quantifier repair.
This is an algebraic consequence of the reviewed weighted theorem,
not a claim of priority or physical advantage. No experiment has been run here.

Take nine distinct complete input states, a constant output, and a common
denominator W=24. Distribution A has weights (4,4,4,4,4,1,1,1,1).
Distribution B has weights (8,2,2,2,2,2,2,2,2). Both sums are 24 and both have
nine atoms. Both sum w*log2(w)=40 exactly, hence initial Shannon entropy
log2(24)-40/24, identical zero final entropy and identical total loss.

For A, the heaviest-prefix threshold is L(16/24,4/24). The first five equal
weights have increasing absorption losses; the later weight-one losses are at
most L(23/24,1/24). To establish the displayed maximum, compare those exact
exponential ratios against R(16,4), rather than trust approximate logs.

For B, all absorptions after its first atom have weight two. Their losses
increase with accumulated mass, so its threshold is L(22/24,2/24).

At K=1/2, the candidate distinction is A CANNOT at any finite depth / B CAN
with a finite circuit (not at every depth):

    R(16,4) = 5^20 / 2^32 > 2^12 iff 5^20 > 2^44;
    R(22,2) = 2^48*3^24 / 11^22 < 2^12 iff 2^36*3^24 < 11^22.

These are finite exact integer inequalities to be independently checked, along
with A's maximum location. The total-loss depth lower bound cannot distinguish
these tasks even though neither initial entropy nor support size changed.

Fresh reviewer obligation: verify both integer comparisons, all prefix ratios,
the exact entropy identity, and application of the weighted theorem. Finite
support nine can be encoded in four input bits with seven zero-probability
encodings under the reviewed weighted contract. It is not the uniform full-input
contract and must never be presented as such. This proof-only unit is registered
here before its independent arithmetic attack; no new full experiment is implied.

Review disposition, 2026-10-03: the running independent reviewer identified the
original phrase "B CAN at every finite depth" as false. The correction above
preserves finite existence; total entropy rules out depths five or less at
K=1/2. The saved final report explicitly approves the corrected file. The later
weight-one tail is smaller by the exact inequality
2^104*3^24 < 5^20*23^23. This header/disposition promotion changes no proof claim.
