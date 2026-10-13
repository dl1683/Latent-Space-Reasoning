# Distribution-robust clean computation

Date: 2026-10-03. Rigor: PROVED AT STATED FINITE DETERMINISTIC CLEAN SCOPE.
Fresh actual CLI ROBUST_ERASURE_REVIEW.md: lower PASS, charged upper PASS,
registered partition code PASS. Full 428-case run returns CONFIRM, with raw
evidence admission pending. Novelty, efficiency, and physical interpretation
remain separate gates; the general compiler is exponential.
So what: an erasure claim must state how concentrated the input distribution may
be. A rare event under one prior cannot be priced as rare under every workload.

## Model, charged resources, and leak audit

Keep the finite NOT/CNOT/Toffoli/one-bit RESET model and exact clean output task
of ATOMIC_ERASURE_PROOF. An input-independent circuit must work correctly for
every Boolean input and respect the per-step joint entropy-loss cap for EVERY
distribution on M=2^n input states satisfying max_x p_x<=eta. Here 1/M<=eta<=1
is a supplied rational concentration limit. Zeros are allowed in distributions;
this never relaxes the requirement of exact functional correctness on all inputs.
No distribution oracle enters the machine. The gate schedule is the SAME for
all admitted distributions. Count width, gates, program/table, construction work,
and precision. No physical heat/clock assertion or classical speedup is made.

Let q=max_y |f^-1(y)|, phi as in the uniform proof, and h the binary entropy.

## Candidate exact threshold

For q=1 let R(q,eta)=0. For q>=2:

    R(q,eta) = eta*phi(q)       if eta<=1/2 and q*eta<=1;
             = h(eta)          if eta<=1/2 and q*eta>=1;
             = 1               if eta>=1/2.

The formulas agree on both boundaries. Candidate CAN/CANNOT: a fixed circuit
respecting all distributions exists at arbitrary finite depth iff K>=R(q,eta).
At eta=1/M this recovers the uniform atomic threshold. At eta>=1/2 EVERY
noninjective clean task requires a one-bit-loss step under some admitted input
distribution, whatever the available time or zero workspace. This shows why
unqualified low average-entropy claims can fail a workload-concentration audit.

## CAN: no access to the actual distribution

Use any fixed singleton-absorption order in each fibre, with the charged
selective flag-reset macro and reversible clean terminal conversion. At its k-th
merge, the singleton mass b<=eta and the accumulated mass a<=(k-1)eta,
a+b<=1. All reversible steps preserve entropy for every distribution.

If q*eta<=1, coordinatewise monotonicity of L gives loss<=L((q-1)eta,eta)
=eta*phi(q). If eta<=1/2 and q*eta>=1, loss<=L(1-b,b)=h(b)<=h(eta).
For eta>=1/2 any one-bit RESET loses at most one bit. Thus one explicit fixed
circuit meets the threshold simultaneously for every admitted prior. It uses
the same charged exponential truth-table compiler as the uniform theorem.

## CANNOT: a prior witnessing a forced merger

Choose a largest final fibre F. Complete-state determinism ensures intermediate
partitions refine final f-fibres. At the FIRST step forming all of F, its
predecessor nonempty blocks partition F into r>=2 parts.

If q*eta<=1, assign every atom in F mass eta and distribute the remaining
1-q*eta over outside atoms, each of capacity eta. This is possible because
M*eta>=1. At the forced step, the uniform-in-F convexity argument gives loss
at least eta*phi(q). The case q=M and q*eta=1 has no outside remainder.

If eta<=1/2 and q*eta>=1, concentrate total mass one within F. Let the smallest
predecessor block have d atoms, so 1<=d<=q/2. Assign this block total mass

    t=max(eta, 1-(q-d)*eta).

It obeys eta<=t<=1/2: the second term is at most 1/2 because
(q-d)*eta>=q*eta/2>=1/2. Also t<=d*eta, using q*eta>=1 and d>=1.
Give the complementary blocks total mass 1-t<= (q-d)*eta. Both totals can be
distributed among their atoms without exceeding eta. The forced multiway loss
is at least the binary grouping loss h(t)>=h(eta). This distribution can depend
on the fixed circuit; it need not defeat all circuits simultaneously to refute
the statement that a circuit works for every admitted distribution.

If eta>=1/2, assign mass 1/2 to two distinct inputs with the same f output and
zero elsewhere. Their first full-state merger loses one bit. No prior padding
or arbitrary reversible waiting can split those two initial alternatives.

## Precision and physical boundary

For eta=u/W rational, entropy thresholds can be compared by finite integer powers
after exponentiating W times each entropy expression. This is computable,
possibly expensive bit work, not a unit-cost real oracle. Distributional
robustness remains a mathematical assumption; inferring that a real device obeys
such a cap requires additional physical evidence. Full-state Shannon accounting
does not certify marginal reset costs, noise, dissipated energy, or cooling time.

## Pre-registered experiment ROBUST-ERASURE-1

CPU exact Fraction minimax optimization, registered before implementation/run.
For each support cardinality M=2..5, even denominator W in {2,4,6,8,10}, and
integer cap u=ceil(W/M)..W, enumerate EVERY bounded integer weight vector
0<=w_i<=u, sum w_i=W. For every integer partition of M, use a fixed contiguous
labelled target partition with those fibre sizes. Optimize over all serial binary
coarsening orders refining that target. Each edge cost is the MAX exact ratio
(a+b)^(a+b)/(a^a*b^b) over the weight vectors, with 0^0 interpreted as one.
The circuit order cannot adapt to the chosen vector. Compare the optimum ratio
with 2^(W*R(q,u/W)): use edge_ratio((q-1)u,u) below q*u<=W,
edge_ratio(W-u,u) above it for 2u<=W, and 2^W for 2u>=W (q=1 gives one).

Smoke: M=2..3, W in {2,4}, every registered u and target. Full ranges above
require fresh code/proof review. Output each task, cap, exact optimum/prediction,
counts of input distributions and states, source hashes, and first mismatch
witness if any. CONFIRM iff every exact ratio agrees; KILL on mismatch, VOID on
scope/arithmetic changes. Because all edge sublosses are nonnegative, serializing
a multiway merge cannot increase its worst prior loss, so arbitrary-depth binary
optimization is a valid abstraction control. No gate-level robustness run or
novelty certification is supplied by this finite enumeration.

Novelty gate: compare against robust information loss, generalized reversibility,
majorization/concentration constraints, and minimax quantization. Proof review
and finite agreement do not settle priority or mission usefulness.
