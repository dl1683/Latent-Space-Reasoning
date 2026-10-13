# Returned-context threshold: proof candidate

2026-10-03. Rigor: PROVED AT SCOPE for A, B and C in the independent
RETURNED_CONTEXT_THRESHOLD_REVIEW.md. The body preserves the submitted proof
wording as provenance; its candidate labels are superseded only at that review's
exact scope. Registered FULL controls are independently admitted in
THRESHOLD_FULL_EVIDENCE_REVIEW.md. This sharpens the
composition question during the ongoing genuine advisor/worker dialogue; it
has a settled next-gate priority in STEERING_RESTART_DECISION.md, with no
established novelty or completed mission.

## Contract and infinity audit

One finite deterministic fixed NOT/CNOT/Toffoli/one-bit-RESET circuit must clean
X in {0,1}^n, output AND(X), return R in {0,1}^r pointwise, and clear every input
and workspace bit. R is uniformly distributed and INDEPENDENT of X in this
theorem candidate. X's unknown prior ranges over p_x<=eta, with M=2^n and
M*eta>=1. The circuit is fixed before the prior. Each step's full-state Shannon
entropy drop is at most rational K. Count every bit, gate, program-generation
bit operation and the supplied statistical promise. No stochastic gates,
distribution oracle, real precision or fresh random reservoir. Product and
uniform-marginal-only contracts are explicitly distinguished below.

## A: multigraph RESET capacity lemma

Take any elementary RESET of any exact circuit in this contract, at any prefix.
No full-state class can contain inputs with two different original R labels:
deterministic continuations could not return both values. Thus every collision
pair of nonempty full-state classes is within one original R slice. At most two
predecessors map to one successor under a one-bit RESET.

Within each pair, let A,B be its disjoint subsets of original X and match every
member of the smaller set injectively to distinct members of the larger set.
Across all slices/pairs this gives a multigraph on original X vertices; parallel
edges across R slices are allowed. Let B_total be its number of edges, namely
the sum over pairs of min(|A|,|B|).

Assign edge weights 0<=t_e<=eta with total

    T=min(eta*B_total,1/2).

Define provisional p_x=max of incident edge weights (zero for isolated vertices).
Then p_x<=eta and sum_x p_x<=2*T<=1. Complete to a probability distribution using
remaining vertex capacities; capacity M*eta>=1 makes this possible. For each
original pair the matched endpoints on either side are distinct WITHIN THAT
PAIR, so both side probabilities are at least the sum of that pair's t_e.
Sharing vertices across slices never invalidates this individual inequality.

For binary merge entropy L(a,b), L(a,b)>=(2*min(a,b)): hold the smaller mass fixed
and use monotonicity in the larger mass, with L(b,b)=2*b. Product uniform R scales
every pair's masses by beta=2^-r. Summing their losses gives, for this admissible
prior, at least

    Lambda = beta*min(1,2*eta*B_total).

This prior may depend on the fixed circuit and attacked gate: that is permitted
by the universal-prior quantifier. It is one original-X prior, not a separate
prior chosen independently for every R slice. Earlier coalescence is allowed
because the matched vertices are original inputs, not current equiprobable states.

Candidate conclusion: for K<beta, every RESET has

    B_total <= floor(K/(2*eta*beta)).

Each collision decreases the complete-state class count by one and contributes
at least one to B_total. Initial/final class counts are M*2^r and 2*2^r for n>=2.
If b=floor(K/(2*eta*beta))>=1, work is at least

    ceil(2^r*(M-2)/b).

If b=0, exact cleanup is impossible. Arbitrary width, predicates, recoding of R,
reversible gates and previous RESETs do not escape this candidate bound.

## B: concrete same-contract CAN/CANNOT

At eta=1/8 and K=(3/4)*2^-r, n>=3, A implies at most three full-state collisions
per RESET and G>=ceil(2^r*(2^n-2)/3). The existing charged Gray singleton compiler,
run separately within returned R contexts, has O(2^(n+r)+n+r) gates and width
2n+3r+4. Each RESET absorbs at most one original X singleton, whose mass <=eta;
its context-scaled loss is <=beta*h2(eta)<(3/4)*beta. Predicate/tree workspace is
uncomputed as in the separately reviewed compiler; all R values return pointwise.
Therefore candidate optimal work is Theta(2^(n+r)), not just a failed fast search.

At K>=beta, compute-copy-uncompute AND and erase every X bit in each full R
context. Every selected RESET has loss <=P(R=context)=beta, even if X and R are
correlated. Charged Gray context predicates give O(n*2^r+r+2^r) gates. At fixed
r this is linear in n. The r=1 implementation/registered controls are in
RETURNED_FAIR_BIT_CANDIDATE.md, not yet independently proved in this record.

In the PRODUCT contract these two bounds give a candidate sharp efficiency
threshold at K=beta: below beta fixed eta and fixed K/beta<1 force exponential
work whenever feasible; at/above beta, fixed-r clean AND has linear work. This
does not assert that every sub-beta cap is feasible. In particular the reviewed
robust feasibility frontier remains load-bearing, not replaced by Lambda.

The upper at K>=beta also applies to the broader known-uniform-MARGINAL contract.
Its impossibility lower inherits the product subclass. Its sub-beta feasibility
upper does NOT follow: correlated laws need not bound P(X=x|R) by eta. Never
promote individual fair coordinate marginals to a uniform joint R context law.

## C: exact feasibility scaling, separate proof obligation

Let R_atomic(q,eta) be the reviewed ROBUST_ERASURE_CANDIDATE threshold for one
fixed circuit at bounded atom mass eta. Candidate exact threshold with returned
product-uniform R is beta*R_atomic(q,eta), q=2^n-1 for AND.

Upper: run the reviewed singleton absorption schedule separately in each R
context; only that context loses information at a RESET. Lower: original R is
a deterministic function of each reachable full state (because it must be
returned at termination). Thus each entropy drop is beta times the sum of the
conditional drops. Restrict any candidate circuit to one fixed R label, prepare
its constant bits by reversible NOTs, and clear constant returned bits by NOTs
afterwards. This becomes an exact no-helper cleanup circuit for all p_X. The
reviewed robust lower yields some prior and step with conditional drop at least
R_atomic; all other conditional RESET drops are nonnegative, giving beta scaling.
The restriction/initial constants and full-state scope require independent review.

## Falsification/usefulness obligations

Fresh proof attack must test overlapping X labels across slices, parallel edges,
fractional eta/normalization, previous coalescence, arbitrary finite widths and
the elementary RESET assumption. Charged upper must be evaluated, not inferred
from a truth-table-only primitive. Small optimization/search controls should
include a deliberately too-strong negative inequality so a vacuous search cannot
pass. Vertex sweeps alone cannot maximize concave entropy loss.

So what (candidate): statistical context availability can change the optimal
work needed to obey a logical peak-loss budget by an exponential factor. The
strongest dismissal is expected-cost ensemble time sharing. This may expose an
unphysical feature of the meter rather than a useful new computation paradigm.
Prior literature, preparation cost, meter choice and useful-world gates stay open.
