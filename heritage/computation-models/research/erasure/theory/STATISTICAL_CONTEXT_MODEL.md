# Statistical context as a computation resource: working model

2026-10-03. Rigor: DEFINITION / RESEARCH CANDIDATE, with theorem status delegated
to the cited fresh reviews. This is the October restart's common contract, not
an earned new-paradigm or physical-computing claim. Historical July bridge debts
are still open. No paper or submission artifact is being prepared.

## Finite machine and charged resources

A program is a finite word over NOT, CNOT, Toffoli and one-bit RESET on w binary
registers. The reversible gates have their ordinary exact Boolean semantics;
RESET sets its target to zero. Each gate counts as one unit of work. Every gate
name/address and program-generation bit operation is additionally charged.
Registers, program length and exact integer encodings are finite. No oracle,
real-valued instruction, random gate, unbounded advice or free truth-table lookup.
The atomic computation rule is ordinary finite-bit manipulation. The proposed
new first-class resource is a statistical operating contract and its peak loss.

Initialize data X, optional supplied context R, and every other register to zero.
After a fixed schedule, distinct output registers must contain f(X),R, with R
returned for each individual input, and EVERY other register zero. Correctness
holds for all X,R, including those assigned probability zero by a prior. Gates
conditional on a data/context bit are explicit reversible control macros; all
macros execute in the same fixed schedule and count toward work. No input-dependent
external timing or discarded controller is free. The full state includes all
input, output, predicate, temporary and controller registers.

An honest baseline uses the same cleanup/output contract, finite gate basis,
width/program accounting and supplied statistical resources. Merely retaining
the original input changes the task and is not an admitted comparison.

## Distribution contract and budget

Specify a prior class P on the finite initial labels before choosing the circuit.
Examples: one known uniform prior; unknown p_X with max_x p_x<=eta; product
p_X times a supplied uniform r-bit R; or arbitrary joint laws with a specified
uniform R MARGINAL. The latter two classes differ. Promise authenticity and
preparation are resources; finite samples never certify an exact distribution.
The circuit receives neither the numerical prior nor a distribution oracle.
Finite rational eta/K describe the constraints; quantifying over real-valued
finite distributions is a mathematical contract, not precision supplied to gates.

For state S_g after gate g define

    loss_p(g) = H_p(S_(g-1)) - H_p(S_g),
    peak_P(C) = sup_(p in P) max_g loss_p(g).

The circuit is admitted at rational budget K when peak_P(C)<=K. Reversible gates
have zero full-joint loss; one-bit RESET has loss in [0,1]. One circuit must obey
the budget for every promised prior. The optimization objective is elementary
gate work at specified width/statistical contract/K, with construction cost and
program size reported separately. Exact arithmetic in finite experiments is an
offline verifier resource and its bit work is not assumed constant-cost.

This is a logical ensemble meter. It is not a calibrated cost for a local reset
device, a worst individual execution, energy, thermal power or wall-clock speed.
Meter choice is a research question. A change of meter can destroy a theorem's
usefulness without refuting the theorem under its original definitions.

## Current theorem map

PROVED AT SCOPE, by the cited independent reviews:

- Uniform finite inputs: exact all-depth threshold phi(q)/M for largest output
  fibre q, with a fully charged exponential table compiler. See
  ATOMIC_COMPILER_REVIEW_2.md and ATOMIC_ERASURE_PROOF.md.
- Supplied positive rational masses: exact weighted heaviest-prefix frontier;
  unknown bounded-atom priors: exact minimax feasibility frontier. See
  WEIGHTED_ERASURE_REVIEW.md and ROBUST_ERASURE_REVIEW.md. The old implemented
  lexicographic compiler is not a weighted-optimal compiler.
- Clean AND with known uniform X and independent uniform returned R: optimal
  Theta(n+r+n/K) elementary work at every feasible rational cap, linear width,
  with the n=1 reversible exception. SUCCINCT_ERASURE_REVIEW_1.md.
- Clean AND without retained context, unknown max-atom 1/8: optimal Theta(2^n)
  work at cap 3/4 and O(n) work at cap one. PRIOR_UNCERTAINTY_REVIEW.md. The
  known-uniform comparison changes the statistical promise explicitly.

EMPIRICAL finite FULL controls for those statements are independently admitted
in ATOMIC_EVIDENCE_REVIEW.md, WEIGHTED_ROBUST_EVIDENCE_REVIEW.md and
SUCCINCT_PRIOR_FULL_EVIDENCE_REVIEW.md. Their agreement is not a universal proof.

PROVED AT SCOPE in RETURNED_FAIR_BIT_REVIEW_1.md: a returned fair marginal enables
linear cleanup even with arbitrary data correlation, with peak <=1/2. Its code
serialization is repaired and independently code-approved. FULL passes 11
circuits and 4336 correlated controls; independent raw admission PASSED in
RETURNED_FAIR_BIT_FULL_EVIDENCE_REVIEW.md.

PROVED AT SCOPE in RETURNED_CONTEXT_THRESHOLD_REVIEW.md: shared-prior RESET
capacity, exact product-context feasibility scaling and the exponential/linear
AND work threshold at full-context mass (linear in n for fixed r). That review
also proves marginal-only correlated laws can be infeasible at a cap where
product laws are feasible. Registered generalized CPU controls passed all gates
in THRESHOLD_CODE_REVIEW.md and THRESHOLD_FULL_EVIDENCE_REVIEW.md: 30 circuits,
64735 priors, 10364 exhaustive and1000 structured layouts. This is finite
EMPIRICAL admission, not an extension of the universal mathematical proof.

PROVED AT SCOPE in METER_REVIEW.md: changing J to local target entropy L shifts
the feasibility floor to h2(2^-r/8) and the fixed-r work cliff to h2(2^-r)
(r0:one). Worst-positive-branch meter V permits no finite universal cap for
n>=4 without a minimum atom/source-preparation restriction. The original
growing-r local work gap is closed by MASKED_BLOCK_REVIEW.md below. No general
end-to-end bit-work or hardware efficiency
theorem is established. UNIFORM_BRANCH_REVIEW.md now proves exact uniform V
floor log2(3) and optimal Theta(n+r) elementary work, including growing r.
MIN_ATOM_BRANCH_REVIEW.md bounds finite minimum-positive-atom V cap within an
additive3bits (dyadic scale); it is not a fixed probability-grid restriction.

PROVED AT SCOPE: MASKED_BLOCK_REVIEW.md closes growing-r local critical work
with returned-mask chunk cleanup and a matching original-label count lower:
Theta(n+r+2^r*max(1,n/r)), n>=3r>=1, uniform constants. It changes no registered
METER-1 program/source identity. METER_FULL_EVIDENCE_REVIEW independently
admits72circuits/360finite traces as EMPIRICAL controls. MASKED code gate PASS;
all116circuits FULL CONFIRM after429592644496 exact CPU transitions, observed
exit0 and fingerprinted receipt. MASKED_BLOCK_FULL_EVIDENCE_REVIEW independently
admits all116circuits at registered finite scope, including every original label,
q-vector entry and clean terminal; no universal proof follows from the sweep.

PROVED AT SCOPE in GENERAL_MASKED_REVIEW.md: arbitrary finite f with charged
clean computation cost C_f admits returned-mask cleanup at EXACT local marginal
beta under ALL unknown p_X. Every noninjective f has exact universal floor
h2(beta); injective maps have a separately charged reversible exception.
This uses independence, not merely uniform R marginals. No new generic FULL.

PROVED AT SCOPE in CORRELATION_LOCAL_REVIEW.md: n>=3r>=4, arbitrary correlation
with capped X and uniform returned R forces optimal critical local work
Theta(n*2^r+r). TOTAL_LOCAL_REVIEW.md proves product total local cost Theta(n+r)
simultaneously with optimal work, and correlated total cost Omega_C(n*r)
on uniform input under fixed work allowance G<=C*n*2^r. The latter does not
quantify over arbitrarily slow programs. These are logical resource theorems;
primary novelty, physical heat, source authentication and usefulness remain OPEN.

## Why this is being pursued, and what remains missing

PROVED AT SCOPE in ERASURE_OBSERVER_REVIEW.md: one fixed finite execution,
all unknown workloads, independent uniform private-owner returned r-bit key,
r>=2, local cap h2(2^-r), observer sees only public f(X) and pre-RESET taps.
Perfect two-probe privacy forces whole-transcript privacy. A clean realization
exists iff max output fibre<=2^r, charging finite ranking/inversion and cleanup.
AND therefore needs r>=n for two probes, despite one-probe small-key CAN. At cap
one, the separately charged linear short-key pairwise mask is a positive control.
Other internal wires, later key disclosure and safe repeated reuse are outside
this privacy claim. ERASURE_OBSERVER_SOURCES leaves priority/usefulness OPEN.

PROVED AT SCOPE in MASK_SOURCE_REVIEW.md: for ALL unknown p_X and independent
unknown Q_R with rational maximum atom tau in[beta,1/2], noninjective floor is
h2(tau). For tau<1/2 critical AND work retains the masked Theta order; at the
half endpoint it is linear. This family does not replace arbitrary correlation.
WINDOW_DESIGN_REVIEW.md extends the n>=r>=8 combined work/total distinction to
fixed rational design crossentropy under zero-extended two-gate windows. Its
padding claim explicitly preserves costs and adds two separately charged scratch
bits. Both are scoped mathematical results, no physical or source-testing claim.
MIN_ATOM_BRANCH_ERRATUM_REVIEW repairs the older direct shortcut to distinct R
outputs: G=3n+2r-1,width2n+2r, preserving the finite-source V enclosure.

The proposed usefulness is precise resource accounting: equal total entropy loss
can conceal a different atomic bottleneck; uncertainty and returned statistical
context may radically change the work required to obey a peak-loss constraint.
The hardest dismissal is that the helper merely time-shares an ensemble, so a
striking logical work law may disappear under a physically meaningful meter.

Priority comparisons already find occupied foundations in entropy aggregation,
generalized reversibility, energy-aware algorithms, private circuits, catalytic
space and complexity-constrained thermodynamics. ATOMIC_ERASURE_SOURCES.md records
specific primary passages and their limits. A different resource definition alone
does not establish a useful fundamental reframe. A strict literature delta, useful
same-contract theorem pair, meter assessment and fresh whole-repository adversary
remain required. No extended Church-Turing breach or completion is established.
