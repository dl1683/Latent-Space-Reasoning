# Recall latency of occupied-memory translations: an exact lower target

2026-10-03 CANDIDATE-PROOF / generative question. A tactical foundational
recallability lemma, not a selected strategic direction or novelty assertion.
Fresh design/infinity-leak audit precedes proof attack. No implementation/FULL.

Finite immutable input X has C disjoint blocks of m bits, N=Cm. Owner bank B
has C arbitrary bits, no promise or initialization. At the exposed checkpoint
the bank is R_i=B_i XOR parity(X_block_i). The algorithm has at most S private
bits, INCLUDING all input-dependent controller/register/advice state. Program
and its generation are finite, public, charged and independent of X,B. The
owner may demand RECALL there: restore B exactly for every original X,B.
Recall can query input coordinates adaptively. Each single-bit input access
costs one query; all bank reads/writes, private operations, addresses and program
bits are separately charged. Give the lower-bound machine all of R for free,
which only strengthens it. No external undo log, readable output or copied input.

Precisely C,S>=0 integers,m>=1. For every fixed R, the COMPLETE nonpublic
checkpoint configuration has <=2^S possibilities, including input-dependent PC,
addresses, caches, allocations, length and continuation metadata. The admitted
theorem is deterministic. Its lower may grant unlimited fresh workspace AFTER
the checkpoint: that cannot supply missing information about original X,B.
Zero-error randomness is outside admission until seed/cost semantics are defined.

The founding finite machine can use reversible two-bit XOR from an immutable
input bit into a bank/private bit, counted as one input query and one target
update, plus ordinary bit READ/WRITE/XOR and charged private RESET. A fixed
unrolled schedule carries no input-dependent controller information; its program
length/address encoding is charged. Private scratch need not preserve past values
after an abort; clearing S private journal bits costs S writes. This is classical
CS access/permission accounting, not a thermodynamic meter or no-erase protocol.

The exact-S upper uses a public fixed synchronous schedule. Program position,
addresses and timing depend ONLY on C,m,S, never on X,B or journal contents.
Generation/storage/controller bits and execution are charged separately; public
is not free. No input-dependent choice of position/address/checkpoint time is
exempt from S. If S counts EVERY physical controller/address bit on a uniform
RAM, subtract that actual footprint before counting parity journal cells; the
exact-S upper is not claimed under that alternate resource convention.

Proposed all-algorithm theorem, exact worst-case input-query bound:

    Q_recall >= m * max(C-S,0).

Proof route: fix any current bank R. Then original B is R XOR F(X), so recovering
B recovers all C block parities F(X). Private state is at most S bits of arbitrary
advice depending on X (and that fixed R). For each fixed advice value, retain
the finite recall decision tree ON ITS VALID advice class; fill missing branches
with arbitrary leaves and bypass repeat queries with their deterministic effects.
Invalid input/advice pairs need not terminate or respect Q; completion preserves
all valid runs and gives a tree of depth<=Q. At a leaf querying d
distinct input bits, at most floor(d/m) complete blocks are revealed. Remaining
parities have rank k>=C-floor(Q/m) on unqueried input bits. A fixed returned B
therefore admits at most 2^(N-d-k) X values in that leaf's query cylinder.
Summing leaves with Kraft weight sum 2^-d<=1 and then <=2^S advice values gives
coverage <=2^(N+S-k). If Q<m(C-S), k>S, so some original input cannot restore.
Bank rewrites are determined by fixed R, advice and queried bits. They can retain
learned information but cannot manufacture unqueried information. Thus the proof
even permits unrestricted additional workspace during recall, with information
restriction at the checkpoint. Randomness extension remains outside admission.

Proposed charged CAN: journal the parities of min(S,C) blocks in private bits
before the checkpoint. On recall use those journals for the corresponding bank
bits, and input-controlled XOR each bit of every other block into its bank bit.
Clear private journals with charged writes. Exactly m*max(C-S,0) input queries;
O(Q+C+S) bit updates, finite program O((N+C+S)log(N+C+S)). Journal preparation
cost min(S,C)*m input queries and S initialized bits is explicitly charged.
The checkpoint translations themselves can be generated with N input-controlled
XORs; final endpoint restoration without a journal can always replay N XORs.

With j=min(S,C), exact data update counts: S initialization writes, jm journal
input reads/updates, N checkpoint input reads/updates, recall Q=m(C-j) input
reads and Q+2j updates (saved-to-bank XOR plus j RESETs). These are not TOTAL
bit work or recall latency; controller/fetch/address/generation additionally
charged. Zero input-query cost does not imply zero time.

So what: exact endpoint return of an arbitrary occupied bank supplies no fast
reclamation guarantee. For independent block translations, each saved clean
bit removes an entire m-query recall obligation, and no compressed/nonlinear/
adaptive journal can beat the remaining bound. This is a prefix restoration
limit, not a proof that catalytic computation is useless or that a dirty-bank
algorithm beats a fully granted ordinary incumbent. Static independent-payload
capacity and worst-case computation utility remain different questions.

Priority/mission OPEN. The counting ingredients overlap decision-tree/advice
lower bounds and memory checkpointing. A useful new-model achievement requires
scalable positive computation plus a same-contract universal limit. This simple
translation can be a calibration/control for catalytic prefix claims; it alone
does not supply matrix powering or the proposed recall-sorting upper.
