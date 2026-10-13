# Succinct AND: proposed optimal peak-loss / gate-work frontier

Date: 2026-10-03. Rigor: PROVED AT SCOPE (lower/upper independently passed
SUCCINCT_ERASURE_REVIEW_1.md); repaired code gate and FULL evidence still open.
So what: realize the useful succinct AND task at every admitted peak-loss cap
with asymptotically optimal elementary gate work and linear width, instead of
paying for a truth-table compiler. This attacks H1 in QUESTION_RESTART_RESULT.
Priority and new-paradigm standing remain unsettled; no physical claim.

## 1. Fixed common contract and pre-investment infinity audit

Uniform independent X in {0,1}^n and Z in {0,1}^r, n>=1,r>=0. Inputs occupy
distinct input registers; initially zero output registers return (AND(X),Z).
All input and workspace bits end zero. Z is returned POINTWISE, not merely with
the same marginal distribution. Every comparator, NOT, CNOT, Toffoli and one-bit
RESET is charged. Fixed finite schedule, full joint entropy, no fresh randomness,
data-dependent timing, arbitrary permutations, or wide gates. Charge gate work G,
width, program storage/generation and bit precision separately. The supplied cap
K in (0,1] is a finite positive rational; no entropy/admission oracle is used.

Constructor guarantee is for promised-feasible (n,r,K) triples. It need not decide
this promise to execute: AND always has an ordinary uniform constructor. The
promise affects only its resource guarantee, not an answer to a hidden decision
problem. Admission is computable with finite integer comparisons, whose audit
cost is separately charged and may be large. Construction uses only cheap
rational comparisons specified below, not huge exact entropy powers.

## 2. Candidate theorem pair

For n>=2 let q=2^n-1 and M=2^(n+r). Reviewed atomic feasibility forces

    K >= K_* = phi(q)/M.

CANNOT at every finite depth below K_*. Every admitted circuit satisfies

    G*K >= n-h2(2^-n),
    G >= n+2r+1,

the second because every input and nonconstant output wire must be targeted.
Thus G=Omega(n+r+n/K) for n>=2. Minimum width is Omega(n+r).

Candidate CAN: a uniform finite-basis constructor for EVERY feasible rational K
uses G=O(n+r+n/K), width 2n+3r+4, clears all workspace, and meets the cap.
At the exact symbolic boundary K_* an additional uniform boundary constructor
uses G=O(n+r+2^(n+r)), matching the lower bound. For n=1, AND is injective and
the task is entirely reversible in O(r+1) gates, at peak zero. The n>=2 work
law deliberately excludes that entropy-zero special case.

Gate work is the elementary-circuit resource. Program generation/storage is
O(G*poly(n+r+bitlength(K))), explicitly charged; no stronger construction bit-work
bound or end-to-end classical efficiency advantage is asserted.

## 3. Amortized charged Gray merge primitive

Let a,b be adjacent d-bit vertices. A moving representative class occupies a;
the next singleton/coarse class occupies b. A stable root predicate C selects
the relevant output/R context. Maintain d prefix flags

    P_i = C AND [the first i high-to-low data bits match a].

All flags are computed with NOT/Toffoli ladders. They cost O(d) to initialize.
If a->b flips bit at depth j (zero is highest), uncompute every P_i involving
that bit, in reverse order. Keep the higher matching prefix P_j (or C).
Using the now-zero suffix flag slots, compute a predicate matching all lower
bits too, EXCLUDING the flipped bit. This selects exactly vertices a and b.
No old-bit-dependent flag remains outside this temporary common predicate.

Conditionally reset the flipped bit to b's value, using a zero disposable flag:

    optionally NOT the flipped data bit, so b's value becomes zero;
    Toffoli(common_predicate, flipped_bit, disposable_flag);
    CNOT(disposable_flag, flipped_bit);
    RESET(disposable_flag);
    restore the optional NOT.

The common predicate is unchanged by this operation. It can be uncomputed, then
the prefix flags recomputed for pattern b. This merges only a and b in C's
context, with all flag information about their differing bit removed BEFORE
RESET. The scratch cannot retain the erased distinction. Everything outside the
context remains unchanged. Cost is O(d-j) constant-arity gates per merge.

Use the reflected Gray sequence g(i)=i XOR (i>>1), shifted by
(2^d-1) XOR 2^(d-1). It ends at all ones. Its edge at depth j occurs 2^j times,
so sum_edges(d-j) < 2*2^d. Maintaining predicates therefore costs O(2^d), not
O(d*2^d). The program is fixed and input-independent. A similar update maintains
matching prefixes for immutable returned Z as its public context pattern walks
through a Gray sequence, costing O(2^r+r) overall.

## 4. Boundary construction

Copy Z to its output register and reversibly clear its input. Compute AND with
a balanced Toffoli tree, copy its output Y, then reverse the tree. Reuse the
clean tree workspace for prefix flags. For each Z context, use stable root
[Z=context] AND [Y=0], computed with explicit gates.

Walk all 2^n X encodings in the shifted Gray order. The first 2^n-1 have AND=0;
absorb them into the moving representative one by one. The final all-ones vertex
has no Y=0 mass. Moving the accumulated representative there loses zero entropy
and leaves both Y=0 and Y=1 supported inputs encoded as all-ones X. Clear X
reversibly using the Z-context predicate. Uncompute every prefix and context
flag. Y and returned Z remain; all else is zero.

The true mergers have losses phi(k)/M, k=2..q, so peak is exactly K_*. Each
context costs O(2^n+n), all context predicates together O(2^r+r). Total
G=O(2^(n+r)+n+r). This uses AND's succinct structure and no acquired truth table.

## 5. All feasible rational caps: staged erasure

If K>=2^-r choose d=0. Otherwise choose the first d in 1..n-1 with

    U_d=(d+2)/2^(r+d) <= K.

If no such d exists choose d=n. These are finite rational/integer comparisons,
with polynomial bit work in n+r and cap encoding. They do not test K_*.
Let ell=n-d and s be the smallest integer with 2^-s<=K. Feasibility guarantees
s<=r+d; for d=0 the cutoff guarantees s<=r. For d=n no low-bit reset is needed.

First erase the ell low X bits only in Y=0 branches, conditioned on every value
of the first s high/context bits among Z and high X. Those conditioning bits
remain unchanged during this phase. Use matching-prefix maintenance on their
Gray order and the same disposable-flag reset. Each reset acts on probability
at most 2^-s<=K, and its condition remains recoverable from the successor state.
Cost O(ell*2^s+2^s+n+r)=O(n/K+n+r). Reversibly clear the low X bits of Y=1
branches with CNOTs from Y. Low bits are now all zero.

If d=0, finish. Otherwise in each returned-Z context the remaining Y=0 high-X
classes have masses 2^ell/M, except the all-ones high class of mass (2^ell-1)/M.
Run the charged Gray merge walk on the d high X bits, ending at the all-ones
class LAST. The first 2^d-1 masses are equal; the last is smaller (zero at d=n).
This is the reviewed descending weighted-absorption schedule. Its loss is at
most phi(2^d)/2^(r+d) < U_d. At d=n it is exactly K_* instead. No splitting of
previously merged atoms or free rebasing is involved. Clear all high X bits
reversibly in their Z context and uncompute all flags.

Minimal choice of d gives 2^(r+d)=O(n/K): when d>1, U_(d-1)>K implies
2^(r+d)<2(d+1)/K; when d=1 use the failed d=0 cutoff. The same bound holds for
fallback d=n because U_(n-1)>K (with n=2 handled by the cutoff or that comparison).
Thus residual work O(2^(r+d)+n+r) plus low-bit work is O(n+r+n/K).

## 6. Registered CPU experiment SUCCINCT-ERASURE-1

Registered BEFORE implementation/execution. Boundary cases n=1..6,r=0..3;
also every such pair with rational K in {1,1/2,1/4,1/8,1/16}. For n>=2 classify
feasibility by exact finite integer comparison against K_*; infeasible cases
must be recorded CANNOT and not represented as successful circuits. For each
feasible case enumerate every x,z through explicit gates, verify full clean
terminal state, returned-Z pointwise equality, prefix/macro invariants, width,
counts, and exact peak entropy-drop ratios from integer multiplicities.

Smoke: n=1..3,r=0..1, all listed caps and boundary modes. Full only after fresh
independent code/proof review. Boundary peak must equal q^q/(q-1)^(q-1) after
exponentiating M times entropy (q=1 gives one). For rational caps compare each
ratio by exact integer powers, no floating tolerance. Conservative count bounds:
boundary G<=64*(2^(n+r)+n+r+1); rational G<=512*(n+r+1+n/K). CONFIRM iff every
frozen feasible case meets every criterion and every infeasible case is recorded
honestly; KILL on a wrong output, dirty flag, cap/count/invariant violation;
VOID on changed ranges, contracts, basis, or arithmetic. Save first witness.

Independent gate must attack the invariant that scratch carries no erased bit,
the Gray amortization, Y-conditioned support, zero-mass final vertex, all-rational
cap constructor, promise/precision accounting, and claimed work optimality.
Nearest sources include Bennett, energy-efficient algorithms, conditional and
compositional generalized reversibility. A passed theorem is not priority proof.

## 7. Independent gate and failure-report repair

SUCCINCT_ERASURE_REVIEW_1.md passes the full theorem pair and H1 work conjecture,
but returns code REPAIR: an exception could leave an earlier CONFIRM at the
requested output path. FULL remains gated. The runner now publishes an identified
RUNNING attempt before case execution, saves the first failing case, completed
rows and exception diagnostic with KILL/VOID and exits nonzero; successful
publication requires every case and stable import/start/end source fingerprints.
Atomic replacement avoids partial final records. The frozen ranges, gate
constructor, arithmetic and proof are unchanged. Current-source smoke and an
injected case-two failure regression pass. A new fresh code reviewer must approve
the repaired source before FULL; saved raw output needs a separate evidence gate.

The second code report identified additional publication/confinement/source-byte
issues (SUCCINCT_ERASURE_REVIEW_2.md). Exact-byte snapshot launching and shared
exclusive-temp publication now address them. Registered dimensions, theorem,
constructor and acceptance thresholds are unchanged. Admitted commands use
`python -B experiments/run_snapshot.py succinct_erasure --output ...` (plus
`--smoke` for smoke). A separate observed-zero-exit receipt, matching attempt,
executed-source identity, artifact hash and full inventory is mandatory before
independent evidence admission; a JSON CONFIRM alone never establishes success.
Trusted Python/tool observer/launcher are explicit; no OS attacker claim is made.
