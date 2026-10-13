# Closed persistence: query locality versus wear

2026-10-04 CANDIDATE PROOF. ENDURANCE-PROBE-LAW-1. No novelty/positive compiler
or mission admission. Applies to the already design-admitted boundary model
SHA f0ab692cf629fd1f3c4f35d7e6600c3a09f5a6505b98f9ff776649b3c8d7ec97.
The indexed kernel remains unselected and is not needed for this proof.

## Scope and definition

N>=k>=1 initially zero/unworn binary cells; every actual bit change consumes
one of L changes, INCLUDING initialization/intermediates. Strict exhaustion
has no observable failure/saturation/wear feedback. Exact deterministic
stationary QUERY_i is read-only, starts from identical parameter-only private
state/current i, and all private/PC/port/output state is canonical at every
boundary. No retained random seed, external clock, past output, future input,
or history advice. Every raw x and every FLIP sequence through H is supported;
QUERY remains correct after H and does not alter the physical endpoint.

An observation is ANY acquisition of a worn-cell bit value, including a
direct Boolean/copy operand or bit output; not merely a READ mnemonic. Each
such acquisition exposes one binary value. Re-reading counts again. Other
computation/code/input-i access reveals no persistent/history information.
Let d_i be the maximum observation count of QUERY_i over all reachable
boundary states in the promise. Infinite d_i gives only a vacuous statement;
for finite d_i, deterministic control between observations contracts into a
finite binary decision tree restricted to reachable paths.

## Candidate theorem

For EVERY logical coordinate i:

    H+1 <= L min(N, 2^d_i - 1).

Equivalently d_i >= ceil(log2((H+1)/L + 1)); use real logarithms and integer
ceil exactly. In particular H=NL-1 forces d_i>=ceil(log2(N+1)) for every i.
For d_i=1, H<=L-1 regardless of redundancy. These are worst-case promise
depth laws, not an assertion that every single invocation has that many reads.

## Proof candidate

At each transcript of <=d_i observed values that occurs on a reachable query,
all subsequent nonpersistent work/control is deterministic. Hence the next
queried position or answer is fixed by this transcript. Two states with the
same transcript cannot diverge except on the next persistent observation.
Unreachable transcript branches may be cut into arbitrary terminal outputs;
reachable promised queries terminate and have observation depth <=d_i.
The resulting decision tree has <=2^d_i-1 internal nodes. Let P_i be the union
of all their positions. Then |P_i|<=min(N,2^d_i-1), and any two reachable
physical states agreeing on P_i give the same QUERY_i answer. This also covers
variable time/nonpersistent branches since control is deterministic from the
transcript and timing is not an additional wear oracle.

Initial logical inputs include both values of x_i. If EVERY initializer had
all-zero projection on P_i, all initial QUERY_i outputs would coincide, a
contradiction. Choose an initializer with at least one 1 in P_i. Since the
machine starts physical zero, initialization consumes >=1 wear in P_i.
Now request FLIP_i exactly H times. All these prefixes are promised. Consecutive
endpoints decode opposite logical bit i, so cannot agree on P_i. Thus every
FLIP consumes >=1 actual wear in P_i; intermediate writes can only add wear.
Total wear in P_i is >=H+1 but <=L|P_i|. QED candidate.

No all-zero off-promise QUERY execution is assumed. Initialization x chosen
for coordinate i need not maximize any other coordinate's wear. QUERY
interleavings are optional and do not supply history. Repeated FLIP_i ignores
updates of other coordinates, but universality includes this sequence.

## Construction hypothesis, NOT theorem admission

k=1: the classical Johnson/thermometer cycle has 2N states: prefixes of ones
from allzero to allone, followed by prefixes of zeros back to allzero.
Let p=last cell, then w_j=cell_j xor p is a prefix of ones followed by zeros,
with last w zero. Binary-search the first zero j; QUERY returns
parity(j) xor (p AND parity(N)); FLIP writes only cell_j to1-p.
Initialize raw0 allzero/raw1 cell0=1. Each FLIP advances one cycle step; cell
positions changed run0..N-1 repeatedly. Candidate guaranteed H=NL-1.
Worst QUERY probes <=1+ceil(log2 N); for powers of two N this equals the
candidate necessary ceil(log2(N+1)). No all-N tightness claim (N3 naive3 vs2).
Actual paid compiler, control space, source/preparation/work and cleanup are
owed by the separately selected new controller unit before CAN admission.
The cycle itself is occupied: TI CD4017B/CD4022B Johnson-counter datasheet
https://www.ti.com/lit/ds/symlink/cd4017b.pdf (primary checked 2026-10-04).
No free hardware parallel shift is used by the proposed one-bit update.

## Registered gate/decision contract

Independent theory-only branch. Fresh actual zero-context Codex attack of
definition/proof/model coverage required before PROVED. PROBE-PROOF-PASS only
means the CANNOT clause at scope. PROBE-REPAIR requires a named literal proof
fix; PROBE-VOID for fatal invalidity. No new numerical run registered here.
Primary source translation must test whether this precise law/task is already
known before any novelty admission. Pure theorem is independent of positive
journal/compiler repair, and does not authorize any calibration or code run.
CPU only, GPU unused. No finite solver result is a uniform CAN.
