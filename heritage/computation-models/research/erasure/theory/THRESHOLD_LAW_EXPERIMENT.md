# THRESHOLD-LAW-1: registered bounded CPU experiment

2026-10-03. REGISTERED BEFORE IMPLEMENTATION / SMOKE / FULL. Scientific priority
is the genuine two-round decision in STEERING_RESTART_DECISION.md. Binding theory
contract: RETURNED_CONTEXT_THRESHOLD_CANDIDATE.md; fresh proof gate is running.
This experiment cannot establish novelty or certify an arbitrary concave maximum.

## Fixed contract, finite resources and ladder

Exact NOT/CNOT/Toffoli/one-bit-RESET, fixed schedule, complete state, clean AND
output and pointwise-returned product-uniform R. Unknown p_X has max atom 1/8.
CPU only; all gates/registers/program generation and supplied source promises
charged. No probability oracle, real precision, free predicates or fresh random
gates. Exact integer/Fraction arithmetic with factored entropy-ratio records.
The verified fair-marginal upper is a separate stronger clause, not a license
to replace the product prior for the subthreshold feasibility tests.

FULL ladder n=3..8 and (r,K) in the ordered list
(0,3/4),(1,3/4),(1,3/8),(2,3/8),(2,3/16): 30 circuits. Use the charged singleton
Gray constructor for the three subthreshold points. For the two other points,
compute-copy-uncompute AND and clear each data bit within full R contexts using
explicit Gray predicates. Each complete input must end at (AND(X),R), all else
zero. Require width 2n+3r+4, every RESET fan-in <=2, elementary gate validity,
and the registered K on the tested product laws. Subthreshold circuits must
merge at most three complete-state pairs per RESET; the proposed construction
actually merges at most one. Require class-count decline 2^r*(2^n-2).
Require G<=64*(2^(n+r)+n+r+1) for Gray, and
G<=64*(n*2^r+n+r+2^r+1) for context cleanup. Bounds are finite sanity ceilings,
not a proof of optimal constants. Record actual gates/width/RESET traces.

For every ladder circuit: full uniform support; at n3 the single eight-atom
prior; at n4 ALL 12870 eight-subset priors; at n5..8 sixteen eight-subsets drawn
by Python Random(20261003+n), plus three ten-atom priors with weights
(3,3,3,3,3,3,2,2,1,1)/24 placed on the first ten, last ten and a seeded random
ten-subset. Each atom mass <=1/8. Keep labelled records even if finite controls
coincide. This is 64735 non-full-support circuit/prior checks, plus 30 uniform
traces. Independent reconstruction may cache deterministic RESET state maps;
all original inputs and every elementary gate still receive full-state checks.

Additional tiny circuit controls: n3, r0..2 at K=2^-r for constant, parity,
three-bit majority and injective f=X: 12 controls. Charge the reversible
function implementation (constant has none; parity CNOTs; majority three
Toffolis; injective transfer/clear CNOTs). Noninjective f uses the same selected
cleanup. Injective transfer has zero RESET loss and exact G=2(n+r).

Excluded-contract negative: the n3,r1 Gray schedule at K3/8 with eight atoms
[3,2,0,1,13,12,14,15], weights 1/8. X and R both have uniform marginals but
are correlated. Require its cap violation; never call it a product-theorem
counterexample or an all-circuit correlated impossibility.

## RESET-layout attacks

A layout is a partition of original X atoms into predecessor classes, together
with a partial matching of those classes into RESET collision pairs. Unmatched
classes are unchanged. Generate all set partitions canonically and all partial
matchings; for M2,3,4 their counts are 3,11,49. One independent layout per R
slice captures earlier coalescence, overlapping X labels and parallel edges.

FULL: M2..4, r0,1, eta in the DISTINCT set {1/M,1/2,3/4,1}. Enumerate every
tuple of slice layouts: exactly 36+528+9800=10364 layout/eta cases. Use the
candidate matched-edge construction to produce one shared original p_X, verify
normalization/caps exactly, and calculate the resulting complete-state RESET
loss. Require loss >=2^-r*min(1,2*eta*B_total). These are finite constructive
witness checks, not optimized supremum certificates.

Also 1000 structured seeded cases, Random(20261003): cycle through M8,16,32,
r1,2,3, eta1/8. Include parallel singleton matchings, shifted matchings, shared
centres across slices, paired coalesced blocks and random partition/matching
cases in a fixed repeating five-pattern order. Record each complete layout,
constructed rational prior, B_total, exact ratio and required exponent.
Positive control r0 is included in exhaustive cases. Negative control drops
the 2^-r factor: M2,eta1/2,r1, only slice zero merges the two X singletons.
The capped p_X is uniquely uniform, so true loss is exactly 1/2 while the false
bound is one. Require exact rejection. This is a certified false strengthening,
not a numerical-search inference.

## Smoke, evidence and frozen tokens

Smoke: n3 only for all five ladder circuits and their one eight-atom prior;
all 12 tiny function controls; the excluded correlated control; all 36 M2
layout/eta cases and first 16 structured cases; certified negative control.
FULL waits on fresh code and theory gates. Sources must execute from captured
bytes, publishing an identified RUNNING before computation and KILL/VOID on
first failure. Save the separately tool-observed exit/source/attempt/inventory
receipt. A separate actual fresh raw auditor must reconstruct the FULL results.

TL-CONFIRM: independent lower/charged-upper clauses pass and frozen finite
controls pass/admit. TL-PARTIAL: retain passed clauses while others stay open;
failure of one constructed witness alone selects this, not TL-KILL-LAMBDA.
TL-KILL-LAMBDA requires a certified actual supremum below the candidate or a
mathematical counterexample. TL-KILL-CORE requires a counterexample to the fixed
returned-context pair. TL-OCCUPIED requires a contract/cost translation to an
existing equivalent. TL-VOID: changed promise/meter/basis/ranges/accounting or
source identity makes the unit inadmissible. Implementation/control failures
are recorded as KILL/VOID with their exact witness without elevating them to a
universal mathematical refutation.

So what: test the same-contract charged CAN and CANNOT across a resource
threshold and attack its shared-prior proof constructively. Statistical ensemble
time sharing, physical interpretation, preparation cost, prior translation and
useful-world standing remain open. No GPU, paper or mission completion claim.
