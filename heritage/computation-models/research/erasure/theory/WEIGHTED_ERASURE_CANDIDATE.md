# Weighted finite erasure: candidate exact frontier

Date: 2026-10-03. Rigor: PROVED AT STATED FINITE RATIONAL DETERMINISTIC CLEAN SCOPE.
Fresh actual CLI WEIGHTED_ERASURE_REVIEW.md: lower PASS, finite-basis upper PASS,
partition-runner code PASS. The weighted gate schedule is specified by the proof;
the old lexicographic compiler is not a weighted-optimal implementation.
So what: compute the exact least possible worst entropy-loss step for a finite
clean task from its explicit probability masses, rather than total loss alone.
This is an audit/synthesis result candidate, not a new-paradigm verdict.

## Contract and leak audit

Use ATOMIC_ERASURE_PROOF's finite deterministic exact clean full-state machine,
but let the finite input support have positive rational masses p_x. The input
encodings, output table, rational numerators/denominators, and all construction
work are charged. No splitting a single atom into multiple indistinguishable
physical alternatives, fresh randomness, advice, or distribution oracle. An atom
means one possible complete initial state. Zero-probability encodings may be
present but impose no supported-input correctness requirement. Fixed finite
circuits use NOT/CNOT/Toffoli/RESET with reusable clean workspace.

Write L(a,b)=(a+b)log2(a+b)-a log2 a-b log2 b. It is increasing in each positive
argument; this fact and the largest-mass merge path are established antecedents,
not our inventions (Ebrahimi/Chen/Khisti 2024 Appendix A.3 and Algorithm 1).
The new candidate obligation is exact minimax optimality for a FIXED final fibre
partition, plus a full charged circuit realization, not their EBIM approximation.

## Candidate theorem

Within each output fibre y, sort its supported atom masses
p_(y,1)>=...>=p_(y,q_y)>0 and set P_(y,j)=sum_(i<=j)p_(y,i).
Let

    T(f,p)=max_y max_(2<=j<=q_y) L(P_(y,j-1),p_(y,j)),

where the maximum over an empty set is zero. Clean realizability at arbitrary
finite depth with per-step cap K is conjectured equivalent to K>=T(f,p).
The sorting/scan uses polynomially many arithmetic operations in the explicitly
supplied support size; no polynomial bound for exact-ratio bit work at arbitrary
rational precision is claimed. Circuit
generation remains generally exponential in Boolean input length. Comparisons
need no hidden precision oracle: for integer weights a,b with common denominator
W, 2^(W*L(a/W,b/W))=(a+b)^(a+b)/(a^a*b^b), an exact rational. Cap comparisons
against rational K also reduce to finite integer power comparisons. The bit work
is charged; no unit-cost claim for enormous integers.

## Candidate CANNOT proof

Fix y and j>=2 and the j heaviest atoms in that fibre. Deterministic complete
states only coarsen and cannot cross two final fibres. At the FIRST step putting
these j selected atoms in a single class, they previously occupy r>=2 classes.
Let a_1,...,a_r be the total selected-atom masses in those predecessor classes.
Every a_i>=p_(y,j) and sum a_i=P_(y,j). The actual predecessor classes may contain
additional atoms. Multiway merge loss is increasing in each class mass, since
its derivative in a_i is log2(sum a/a_i)>=0. Thus removing these extra atoms
can only lower the contribution of this merger.

Merge predecessor parts until only two remain. This reduces the multiway loss.
Both parts have mass at least p_(y,j). For fixed sum P_(y,j), concavity of binary
entropy minimizes the two-part loss at the endpoint masses p_(y,j) and
P_(y,j)-p_(y,j). Hence the actual step loses at least
L(P_(y,j-1),p_(y,j)). Other simultaneous merges add nonnegative loss. Apply to
every y,j. This is stronger than looking only at a fibre's FINAL merge: the
heaviest-subset first-coalescence obstruction also detects earlier large losses.

## Candidate CAN proof

In each fibre, choose its heaviest atom's encoding as representative. Absorb all
remaining singleton encodings in nonincreasing mass order, one at a time, using
the reviewed candidate flag-reset macro. The j-th absorption loses exactly
L(P_(y,j-1),p_(y,j)). Process fibres sequentially and finish by reversible table
XOR evaluation and representative erasure. All scratch is uncomputed; only f
remains. This gives peak T(f,p). The support table and generated gates are paid.

The uniform theorem is the specialization p_x=1/M. A biased three-atom example
(.49,.49,.02) shows why the lightest-last-merger bound alone is insufficient:
forming the pair of .49 atoms already costs .98 bits. This is a proof example,
not an experiment result or an independently approved theorem.

## Registered falsification experiment WEIGHTED-ERASURE-1

Precommitted BEFORE implementation or any numerical run. CPU, standard library.
Enumerate every integer partition of W=1..30 with 1..8 positive parts. Each vector
represents one final fibre with rational initial masses weights/W. Optimize over
ALL binary merge orders using exact rational edge ratios
(a+b)^(a+b)/(a^a*b^b), with an acyclic minimax dynamic program. Compare the exact
optimal ratio with the maximum descending singleton-absorption ratio. No logs,
floating tolerance, or fitted cases enter acceptance. A multiway merge can be
serialized into binary submerges, each losing no more than its original total
loss; this justifies the binary optimization for unrestricted-depth abstraction.

Smoke: W<=10, number of parts<=5. Full: frozen ranges above, only after fresh
independent code and proof review. Output records every initial partition,
optimal/predicted ratios, global memoized state count, source hash, and elapsed
CPU wall time. CONFIRM iff every exact ratio agrees; KILL on the first mismatch
(save a witness); VOID on a changed range, arithmetic, distribution, or objective.
No weighted gate-level validation is claimed by this experiment. The gate upper
proof and literature strict delta remain independent gates.

Next novelty search: minimax entropy aggregation trees, descending aggregation,
minimum-bottleneck partition paths, and prior VI metric results. Meila's original
Comparing Clusterings equation 28 already gives the uniform singleton split and
its minimality. Treat that as a direct antecedent, not a semantic loophole.

## Gate and execution standing

The fresh independent proof/code gate passed (WEIGHTED_ERASURE_REVIEW.md).
The full registered partition experiment returned CONFIRM for all 15,225 cases
and 15,225 DP states, using exact rational arithmetic, 0.32 seconds CPU wall.
Pre-run environment/hash: results/weighted_full_run_environment.json. Raw-output
evidence admission is pending; no finite run proves the theorem. No weighted
gate-compiler full validation or priority/efficiency claim is made.
