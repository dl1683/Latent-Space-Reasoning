# Observer access as a new dial: rare erasure privacy has a two-probe wall

2026-10-03 CANDIDATE-PROOF / QUESTION-LOOP HYPOTHESIS. This is a different
observer-access dial, not a settled strategic pivot or a physical heat claim.
Fresh infinity-leak/design audit must pass before fresh proof investment.
No new experiment/FULL, no statement of primary novelty.

CURRENT STANDING: fresh ERASURE_OBSERVER_REVIEW passes DESIGN, LOWER, charged
UPPER and cap-one control. The exact two-probe feasibility and two-to-full
rigidity are PROVED AT THE STATED ONE-EXECUTION SCOPE. The candidate wording
below preserves the submitted proof. Reviewer finite checks are not a registered
FULL; ERASURE_OBSERVER_SOURCES records occupied foundations and OPEN priority.

## Finite model and leak audit

Finite fixed-schedule NOT/CNOT/Toffoli/one-bit RESET; initialize all scratch zero,
clean output (f(X),R), original inputs/scratch zero. Program/generation/width/
preparation are charged. X is arbitrary, ALL p_X admitted. Independent uniform
r-bit R, r>=2, is supplied and returned pointwise to its OWNER.

NEW explicit access contract: the observer sees public Y=f(X), the fixed word,
and at most t chosen pre-RESET target bits B_g. It sees neither R nor other
internal wires; returned R is private owner output. This is an erasure-channel
observer, not an all-wire ISW adversary and not a public-key-output protocol.
An adversary that later learns R is outside this privacy claim. Adaptive two-
probe strategies are included by requiring every fixed pair distribution to
be independent of X within each public output fibre.

Perfect t-probe privacy: for any set of <=t RESET positions, the joint law
of tapped bits over uniform R is identical for every X with the same f(X).
It therefore holds for every original workload law and conditionally on Y.
No computational indistinguishability, infinite key, fresh random gate, hidden
measurement precision or free oracle. All observer positions are finite; fixed
schedule excludes timing as a separate data channel. Logical local cap is
L_g<=theta=h2(beta), beta=2^-r, for EVERY p_X. No device interpretation.

## Candidate rigidity: two-probe privacy implies whole-transcript privacy

The reviewed unrestricted-prior local argument gives one globally rare target
value at each RESET, with probability<=beta for all p_X. In particular each
pure X has at most ONE original R label taking that value. Let U_g indicate
this globally rare value; complementing tapped B_g is a fixed public operation.

Fix a public output fibre F_y. One-probe privacy makes the number of rare R
labels (0 or1) identical for all x in F_y. Drop constant probes. Each remaining
gate g identifies exactly one R label z_g(x) for every x in this fibre.
For any pair g,h, their joint rare probability is beta when z_g(x)=z_h(x),
zero otherwise. Two-probe privacy therefore makes the equality partition of
the active gate indices independent of x.

The entire rare transcript, for a uniform R, equals the indicator of one block
of that fixed equality partition, with probabilitybeta per block. Unused R
labels produce the all-zero transcript with probability1-beta*numberofblocks.
These probabilities and bit patterns do not depend on x. Restore constant
probes and fixed complements: the WHOLE erased-bit transcript is private givenY.
Adaptive two probes follow from the same fixed-pair distributions; they cannot
restore a loophole. An arbitrary history-dependent observer has no extra access.

## Candidate all-time CANNOT from finite-key counting

Saving all RESET target values makes every gate invertible: reverse the word,
restore each erased target bit from its saved B_g, and undo reversible gates.
Thus original X is determined by (publicY, returned-privateR, wholeB).

If wholeB is private within an output fibre of sizeq, choose any positive-
probability transcript b. Every x in the fibre produces it on the SAME positive
integer a_b of key labels. For each fixed R, at mostone such x can produce b,
otherwise reversed computation is ambiguous. Thus q*a_b<=2^r, so q<=2^r.

Consequently, under the critical local cap, PERFECT TWO-PROBE privacy is
impossible at ANY finite depth/width when maxfibre(f)>2^r. For clean AND with
n>=3, its largest fibre2^n-1 forces r>=n. This is a computational privacy
obstruction, not a lower bound on the amount of heat dissipated.

## Candidate CAN and exact privacy feasibility

For arbitrary finite f with maxfibreq<=2^r, assign each x its finite rank within
its output fibre, using k=ceil(log2(q))<=r bits. Reversibly compute Y and rank;
the pair is injective, so a separately charged finite truth-table inverse
clears X without RESET. No function evaluation/inverse is free. Then apply
the reviewed generic returned-mask cleanup to those k rank bits in ONEchunk.
Returned low R XOR rank is uniform for every rank; high R is independent.
The complete RESET transcript (one incoming-pattern indication or no indication
in each original high context) has a law independent of rank, hence of x givenY.
Its RESET marginals are beta. Clean all rank/scratch, return R pointwise.
Thus maxfibreq<=2^r is proposed sufficient for whole-transcript privacy at captheta.
Injective f has a separately charged reversible zero-RESET exception.

For AND and r<n, the existing generic masked compiler is still ONE-probe
private for every workload, since each flag has exactbeta marginal for every
x. Charged gate work inherits Theta(n+r+2^r*max(1,n/r)) at critical cap;
this does not make it two-probe private. When r>=n, its single-chunk version
is whole-transcript private and has that same reviewed work order.

## Counter-control: a larger local cap permits short-key two-probe privacy

At local cap ONE, take r=ceil(log2(n+1)) and assign distinct nonzero r-bit
vectors a_i to the n data bits. Compute/copy/uncompute AND, transferR toprivate
output, then mask X_i by a_i dotR using charged CNOTs and RESET X_i. Each
tapped bit is x_i XOR a_i dotR. Every pair of distinct nonzero vectors over
F2 is linearly independent, so each tapped pair is uniform on four values,
for every x (also conditionally onY). WholeB need not be private.

G=3n+2r-1+sum_i weight(a_i)=O(n*r+n+r), width2n+2r. All pointwise outputs
and cleanup are exact; each RESET local entropy is one. No fresh coins or
hidden PRG. For n>=3, r<n, contrasting the critical-cap obstruction with this
perfect two-probe construction exposes the role of rare atom support.

So what: hiding every erased bit individually can be cheap in secret-key bits;
under an atom-sized rare-erasure cap, hiding even TWO erased bits forces enough
private key labels to distinguish an entire output fibre. This is a potential
resource/privacy audit with a CS access contract, without needing a heat model.
Shannon secrecy, probing security and masking are occupied foundations; whether
the rigidity/charged compilation is a strict useful delta remains OPEN.

Fresh actual reviewer: FIRST design/infinity-leak audit, including private R
access versus public output, all-prior/tail/source dependence, conditioningY,
key preparation and finite inverse costs. Only if coherent, attack rigidity,
constant probes/global complements, pair-to-adaptive relation, transcript
invertibility, q*key counting, onechunk generic privacy, exact maxfibre floor,
and short-key linear-mask control. Return design/PASS-or-KILL then theorem
LOWER/charged UPPER PASS/REPAIR/KILL separately. Bounded CPU prototypes welcome,
no edits/agents/GPU/FULL; no novelty/mission promotion from scoped proofs.
