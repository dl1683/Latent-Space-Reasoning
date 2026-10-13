# Correlation changes critical local work by a growing factor

2026-10-03. PROVED AT SCOPE: fresh actual independent LOWER/charged UPPER PASS
in CORRELATION_LOCAL_REVIEW.md. Original candidate text below is provenance. Separate theory
unit; no running experiment/source/range is modified and no new FULL asserted.

## Same statistical marginals, explicit change of source promise

Clean AND n>=3,r>=4, X max atometa=1/8, R uniform marginalbeta=2^-r and returned
pointwise, fixed primitive NOT/CNOT/Toffoli/one-bit RESET, allscratch/dataclean.
Local capK=h2(beta). All finite source/program/width/gates charged, no oracle,
fresh random gate, fractional atoms or hidden unbounded precision. Compare:

P: ALLp_X cappedeta, with independent uniformR. Reviewed optimal elementary
work Theta(n+r+2^r*max(1,n/r)), MASKED_BLOCK_REVIEW.
C: ALLjoint laws with SAME capped-X and uniform-R marginal promises, arbitrary
correlation. Proposed optimal elementary work Theta(n*2^r+r).

This is a source-promise distinction, not a same-law hardware speed claim.
When n>=r and r grows, costs differ by a factorTheta(r). No claim for r2/3
is made here. The joint law is finite and its preparation/encoding is charged.

## Candidate universal lower for C

At a fixed RESET its target probability is a linear functional of the ORIGINAL
joint law, even after arbitrary prior coalescence. C is a convex transportation
polytope. Because K=h2(beta)<one, its attainable interval must lie entirely
in [0,beta] or[1-beta,1]. Choose this gate's globally rare target value.

Suppose TWOdifferent original R labels each admit SOMEoriginal X taking that
rare value at this gate. Put massbeta at the two selected (X,R) pairs. These
may even use the SAMEX. Their combined X-row mass is at most2beta<=eta,
because r>=4. Each selected R-column has exactly its required massbeta.
Fill all other R columns, each massbeta, from remaining X-row capacities
eta minus alreadyassigned masses. Their total capacity is2^n*eta-2beta,
at least1-2beta. Nonnegative rational row masses with that sum fit the
capacities; a finite transportation filling splits those masses over the
remaining columns. No other condition on conditional X|R exists in C.
The resulting admitted finite rational law has rare probability>=2beta,
contradicting the globally rare bound<=beta. Thus at MOSTONEoriginal R label
can ever take the rare target value at this RESET.

Pointwise returned R implies no complete-state class can contain different
original R labels. Consequently the RESET can collide only inside that one
R slice: in every other slice the target bit is constant. For the admitted
uniform-joint source its full-state entropy drop is beta times the conditional
drop in the active slice, which is at mostone for a binary RESET. Therefore
EVERY universally C-admissible circuit has per-RESET J<=beta on that source,
although its specified budget is the larger LOCALh2(beta).

Total uniform-source loss is n-h2(2^-n), so G>=[n-h2(2^-n)]/beta=Omega(n*2^r).
Input-targeting adds Omega(n+r). Physical R recoding cannot change original
label or convex-source arguments. Extra scratch/depth cannot avoid the bound.

## Candidate charged upper for C

Use the already reviewed context_threshold.fast_circuit(n,r). Copy/returnR,
compute/copy/uncompute AND, explicitly enumerate all full R contexts with
charged Gray prefix gates, and clear each data bit via the three-gate selected
RESET macro. Its flag is supported inside ONEoriginal R context, hence
probability<=beta for EVERYjoint law with that R marginal, irrespective of
correlation or eta. L<=h2(beta). Width2n+3r+4; workO(n*2^r+n+r), matching
the lower (n>=3). All gates/predicates/program generation are charged.

## Gate and useful meaning

Fresh actual zero-context reviewer must attack the convex entropy-tail argument,
two-column construction including SAMEX/cap boundary2beta=eta, normalization/
rational transportation filling, originalR separation under recoding,
conditional entropy at a binary RESET, total loss and charged same-contract
upper. Independent finite CPU transportation/label-layout probes welcome;
no edits, agents, GPU or FULL. Return LOWER/UPPER PASS/REPAIR/KILL, exact scope.

So what: a uniform returned marginal is insufficient to earn the factor-r
masked cleanup gain when workload/context correlation is unconstrained, even
with the same X atom cap. This would supply a quantitative audit rule for
which statistical promise a claimed capability actually consumes. Physical
calibration, prior-art priority and mission remain OPEN.
