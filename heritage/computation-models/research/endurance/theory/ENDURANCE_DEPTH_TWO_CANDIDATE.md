# Non-power-of-two control: two probes cannot attain three-cell full lifetime

2026-10-04 CANDIDATE PROOF, ENDURANCE-DEPTH-TWO-1. Pure theory, no numerical
run, actual compiler or novelty admission. Applies to the admitted old/new
closed deterministic boundary kernels (INDEXED_MODEL_REVIEW transfers valid).
This probes the gap left by the general query law, not a new primary direction.

## Claim candidate

N3,k1,L2,H5=NL-1 requires worst-case cold exact QUERY depth AT LEAST3.
Johnson binary search has the ideal upper3, but its paid compiler remains
unadmitted. General probe law gives onlyceil(log2(N+1))=2 here. No all-N,
all-L strengthening or scalable CAN is claimed.

## Proof route

Assume worst QUERY has <=2 persistent singlebit observations on reachable
states. Prune unreachable transcript branches to arbitrary leaves. This
extends the promised answers to a total Boolean f on three physical bits
computed by a deterministic decision tree of depth<=2.

If f depends on at most2 cells, its essential-cell set P has size<=2. The
same init-plus-repeatedFLIP argument used in PROBE_LAW_REVIEW gives
H+1<=L|P|<=4, contradictingH5. Agreement on essential cells gives samef even
if P wasn't the literal observation-position union of the original program.

Otherwise f depends on ALL3 variables. Its decision-tree root is one variable
a. Both child subtrees must be nonconstant literals, on DISTINCT other
variables b and c. Root repeated at child is constant once restricted;
same child variable would leave only2 essential variables. Thus f is, up to
signs/branch ordering, a 3-bit multiplexer: at a0 literalb, at a1 literalc.
Independent complementations/permutations of physical coordinates preserve
hypercube adjacency, so its bichromatic-edge graph (one-bit changes that
change f) is isomorphic to that of ordinaryMUX. OrdinaryMUX has two disjoint
four-vertex paths P4: within a0 the only changing coordinate isb; within a1
onlyc; the a-edge exists exactly whenb!=c. Those six edges form two paths of
three edges each. This is a finite classification, not a solver result.

Initial raw values0/1 decode differently, so at least one initialized endpoint
is nonzero. Choose that raw input. Its initialization consumes >=1 actual
wear. Five promised FLIPs each consume >=1 (logical outputchanges). Total
wear allowance NL=6 forces EXACTLY1 initialization change and EXACTLY1
actual change per FLIP. No intermediate redundant changes are possible;
each promised FLIP is therefore a one-bit endpoint change along a bichromatic
edge. The update map T is stationary at every reached state beforeH because
all private/PC/code/source state is historyfree and wear unobservable.

Consequently those five FLIPs form a deterministic stationary walk of length5
inside one P4. A functional walk in a tree cannot enter a directed cycle of
length>2; no loop is possible since every move changes a bit/logicalvalue.
In a four-vertex path, at most2 moves precede entry to a two-vertex cycle.
Among five moves, at least3 traverse that cycle's SAME physical edge. Each
such traversal changes its SAME coordinate, exhausting L2 and invalidating
the promised walk. (If no repeated vertex by move4, four edges would require
five distinct vertices, also impossible.) This contradicts exact promised
termination/wear. Therefore query depth cannot be2.

Off-promise f may be arbitrary and stationaryT may be arbitrary outside
required reachability; neither adds promised moves. Only the actual five
FLIPs chosen above are used. The physical initialzero is not assumed to be
a valid query boundary. Initializer selection need not choose logical1.

## Gate contract

Registered before fresh actual independent proof attack. DEPTH2-PROOF-PASS
admits scoped CANNOT only. DEPTH2-REPAIR names literal flaw; DEPTH2-VOID if
fatal. Check reachable-only tree extension, nonlinear initializer/update,
depth2 classification with literal signs, graph P4, singleactualchange from
globalwear saturation, stationary functional walk, Hterminal obligation and
initialization coverage. No numerical grid, CAN/priority/mission admission.
CPUonly/GPUunused. Classical cycle ingredient remains attributed to TI.
