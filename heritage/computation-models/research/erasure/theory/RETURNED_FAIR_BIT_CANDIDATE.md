# Returned statistical context: candidate universal cleanup compiler

Date: 2026-10-03. Rigor: PROVED AT SCOPE for lower/upper, independently passed
in RETURNED_FAIR_BIT_REVIEW_1.md. CODE PASSED in RETURNED_FAIR_BIT_CODE_REVIEW.md.
FULL confirms 11 circuits and 4336 priors; independent raw admission PASSED in
RETURNED_FAIR_BIT_FULL_EVIDENCE_REVIEW.md.
This records the strongest composition attack on the reviewed prior-uncertainty
gap. Registered smoke, serialization regression and code-approved FULL have run.
No priority or physical-power claim.

## Statistical resource and design audit

Same elementary basis and complete finite-state accounting. n>=2 unknown data
bits X, one supplied bit R whose MARGINAL is known fair, distinct initially-zero
outputs (AND(X),R). All inputs/workspace end zero. R is returned pointwise; it
is never resampled or consumed. X and R may be correlated arbitrarily. Thus
independence is sufficient but not necessary for the candidate CAN. Charge the
finite statistical preparation guarantee, input bit, output bit, all elementary
gates, width and program generation/storage. The guarantee is an explicit input
promise, not something finite runs certify. No distribution oracle, timing,
infinite precision or uncharged random reservoir enters the model.

The model without R must work for all capped-X priors; adding an R marginal
constraint changes the resource/contract. It is not a same-promise simulation
separation, and a logical-entropy cap does not imply a device's thermal cap.

## Candidate construction and bound

Copy R to its output and clear its input with two CNOTs. Compute AND from the
original X with the charged balanced tree, copy output and reverse the tree.
Compute the predicate [returned R=0] using a sentinel and a one-bit prefix flag.
For every X bit apply Toffoli(predicate,X_bit,disposable),
CNOT(disposable,X_bit), RESET(disposable). Change the prefix to R=1 and repeat
for every X bit. Uncompute the prefix and sentinel. All data inputs and scratch
are zero; outputs are the original AND and R for every input, not only support.

The reset condition is recoverable from the preserved R output. Full-state
loss at each reset is at most

    P(R=context)*H(X_bit | remaining state, R=context) <=1/2.

The conditional quantity here refers to the current bit/state, after previous
resets. Correlation of initial X and R cannot change the marginal mass of a
pointwise-preserved R. NOT/CNOT/Toffoli lose no joint entropy. A one-bit fair
context therefore meets cap K=3/4 for EVERY input law, with 8n+11 charged gates
and width 2n+7 using the existing Circuit allocation (constants require review).
Every input and both distinct outputs must be targeted, so G>=n+3 and
width>=n+3. Work is Theta(n), width Theta(n), with no optimal-constant claim.

Without that returned resource, PRIOR_UNCERTAINTY_REVIEW.md independently proves
Theta(2^n) at capped-X atom mass 1/8 and cap 3/4. With the fair returned context,
even arbitrary correlated data laws admit the candidate linear construction.
This is a claimed composition law and a usefulness pressure test, not an
established new paradigm. Scalar entropy-loss accounting alone does not price
availability or authenticity of such a statistical context.

More generally a supplied pointwise-returned r-bit R with every context mass
<=K allows context-conditioned cleanup after computing any finite Boolean f.
Match contexts via charged Gray prefix updates, erase every X bit within each
context, clear predicates. No independence from X is needed. A succinct
reversible compute-copy-uncompute implementation of f gives a candidate bound
O(C_f+n*2^r+r+2^r) gates with separately charged workspace/program work. The
distribution constraint must hold for complete R CONTEXTS: fair one-bit
marginals do not imply uniform joint R. This generalization separately PASSED
in RETURNED_FAIR_BIT_REVIEW_1.md under its explicit full-context mass premise.

## Registered CPU experiment RETURNED-FAIR-BIT-1

Registered before implementation/execution. FULL: n=2..12, enumerate every x,R
through the explicit specialized AND compiler, check clean terminal state,
pointwise R, width 2n+7 and exact count 8n+11. Enumerate joint uniform-X/R,
R=X_low_bit with uniform X (fair but correlated), and R=NOT X_low_bit with
uniform X, computing exact multiplicity entropy traces. Peak must be <=1/2.

For n=2,3,4 enumerate ALL priors placing mass 1/4 on two distinct X inputs at
one fixed R value and mass 1/2 on one X at the opposite R value, both R choices.
This gives 48+448+3,840=4,336 explicit correlated priors; initial states may
share X across R but are distinct complete states. Require correct terminal
state and every RESET's ratio <=2^2=4 for denominator four. Record each peak.
These finite priors are controls, not an exhaustive universal-law certificate.

Smoke: n=2,3 for the full-support/uniform/correlated laws, and the first 16
lexicographic three-atom priors at each n. CONFIRM iff all frozen checks pass;
KILL on dirty/wrong output, unreturned bit, loss/count/width violation; VOID on
changed statistical promise, basis, ranges or arithmetic. Use exact integer/
Fraction comparisons, snapshot sources, successful-exit receipts and first
failure publication. Fresh code/proof gate before FULL, separate raw admission.

## Independent review and serialization repair

RETURNED_FAIR_BIT_REVIEW_1.md passes LOWER and UPPER, including arbitrary
correlation, exact gate/width counts and the general full-context upper. CODE
requires repair: n=11,12 valid uniform-law ratios have reduced integer numerators
with 6,778 and 14,793 decimal digits, exceeding the observed Python 4,300-digit
limit. The reviewer derived these without running FULL. No range is reduced.

The repaired artifact stores each exact Q as an integer-power product: factors
[base,exponent], negative exponents for denominator, empty product=1. The loss
is log2(Q) divided by the record's multiplicity denominator. All cap/peak
comparisons still use exact integer/Fraction arithmetic. This changes only the
serialization, not the registered laws, circuit or success criteria. Fresh
smoke/code gate PASSED before FULL; independent raw audit PASSED. Older smoke/source bytes remain in git
7a6e8d6/02005c6; no new format is attributed to their recorded old hashes.

## Narrative attack

CANDIDATE story: one returned fair bit can eliminate exponential robust-cleanup
work under this logical peak-loss meter. The strongest triviality dismissal is
expected-cost time sharing: the two contexts merely schedule half the ensemble
at a time. That may be the correct interpretation and a decisive limit of the
meter's usefulness. Any claimed physical saving, hidden independence assumption,
unpriced preparation or fresh-randomness consumption is rejected. The research
target is the exact role of statistical context, not an embellished speed claim.
