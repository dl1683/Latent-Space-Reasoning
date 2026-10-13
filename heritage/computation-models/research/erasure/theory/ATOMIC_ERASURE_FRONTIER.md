# Atomic erasure frontier: registered candidate and controls

Date: 2026-10-03. Rigor: CONJECTURE / DESIGN-GATE-CLEARED FOR FINITE FALSIFICATION.
This is research toward the owner's objective, not a mission-completion claim.

## 1. What survived the previous failure

EBC constrains average joint-state entropy loss per deterministic layer. The
telescoping bound charges total loss but does not decide whether any individual
legal step can fit a small cap. Finite deterministic inputs are indivisible:
time alone may not make an irreversible transition admissible.

Freeze M equally likely input atoms, an exact target f, and full finite machine
state with no unobserved environment, random coins, external archive, or advice.
An initial state uniquely identifies the input atom. A deterministic transition
can merge equivalence classes of atoms but cannot split them later. Clean final
state identifies f and nothing else. Thus every intermediate partition refines
the final f-fibre partition. All retained controller/input state is included.

Let q be the largest f-fibre and define, for integers k>=2,

    phi(k) = k log2(k) - (k-1) log2(k-1).

Candidate CANNOT: if q>=2, every finite exact clean realization has at least one
step losing >= phi(q)/M bits of joint entropy. Hence K<phi(q)/M forbids the task
at EVERY finite depth, even when the telescoping depth bound is finite.
For injective f, q=1 and the corresponding obstruction is zero.

Candidate CAN AT ABSTRACTION SCOPE: coalesce each fibre one singleton at a time,
in separate steps. This has peak phi(q)/M and at most M-|image(f)| merging steps.
This is NOT yet a bounded-gate or efficient-computation CAN. Arbitrary partition
merges are analyst descriptions; compiling them must charge comparisons, routing,
ancillas, program/table size, uniform construction, and final-state conversion.

## 2. Candidate proof obligations

For the first step creating the complete largest final fibre, its predecessor
blocks have positive integer sizes a_1,...,a_r, r>=2, sum a_i=q. Entropy loss
from that fibre is (q log2 q - sum a_i log2 a_i)/M. Convexity should maximize
the latter sum at q-1,1, giving phi(q)/M. Other merges cannot offset this loss.
The candidate upper path merges 1 with k-1: its loss is phi(k)/M. Monotonicity
of phi should make the final merge the largest. A review must attack the first-
formation argument, complete-state premise, transitions within/across fibres,
multiway/simultaneous merges, controller/time state, and finite output encoding.

## 3. Same-contract AND control

Use the fixed gate basis NOT, CNOT, Toffoli, RESET, with disjoint gate footprints
per layer, finite uniformly generated circuits, and all input/scratch/output bits
in the state. RESET maps one bit to zero. Inputs are uniform N bits. Clean final
state has only AND(X); dirty output is a separate explicitly named contract.

For integer K>=1, a reversible balanced tree, output copy, tree reversal, and
batches of at most K input resets give width 2N and depth
2 ceil(log2 N)+1+ceil(N/K). Each reset batch loses at most K joint entropy bits.
Every clean circuit obeys D*K >= N-h2(2^-N) and, for this bounded-input basis,
an output light-cone lower bound of ceil(log_3 N). For N>=2,K=0 the clean task
is impossible; dirty reversible output remains possible. These are candidate
control proofs using established reversible-computation ingredients, not novelty.

## 4. Design/infinity-leak audit

All enumeration inputs, states, partitions, gates, and output encodings are finite.
No physical cooling or speed law is inferred. Entropy is an analyst resource on
an explicitly frozen distribution, not an implementation-free energy meter.
Partition search is exponential and its elapsed time/state count are recorded.
No large-fanin merge is counted as a gate-level primitive. Initialized scratch
bits count in width. The CPU checks can falsify finite claims, not prove theorems.

## 5. Precommitted finite tests

Experiment RESET-AND-CONTROL-1: enumerate all inputs for N=1..12 and integer
K=1..N; validate output, cleanup, width, disjoint footprints, reversible layers,
per-layer entropy loss and the displayed depth/count. Dirty K=0 controls are
checked separately. CONFIRM only all frozen cases agree; KILL on a mismatch;
VOID if basis, distribution, cleanup, range, or resource contract is changed.

Experiment ATOMIC-ERASURE-1: exhaustively solve binary-merge minimax dynamic
programs over integer partitions for q=1..16. Normalize losses by a declared
M>=q (use M=q for the frozen table). Compare the optimal peak with phi(q)/M;
compare a balanced merge with singleton absorption; evaluate shortest merge
paths under caps immediately below/equal/above the predicted threshold.
CONFIRM if all finite cases agree within 1e-30 at 80-digit Decimal arithmetic;
KILL on a genuine disagreement; VOID if the objective, range, transitions, or
precision changes. Decimal agreement is EMPIRICAL, never an analytic proof.

Smoke: N<=3 and q<=5. Full runs require independent implementation review.
Report complete raw summaries, source hashes, commands, and execution time.

## 6. Narrative and novelty gate

HEURISTIC story: a finite deterministic machine may be unable to erase information
at all under an overly small per-step cap, however long it runs. The obstacle is
indivisible uncertainty, not total work. This is an exact structural question that
the previous telescoping theorem did not answer.

The strongest triviality attack is 'you are just optimizing a partition lattice.'
That may be correct. Nearest homes include entropy-constrained quantization,
information lattices, reversible computation, and bottleneck aggregation. No
novelty claim is earned by an unsuccessful literature search. The missing useful
gate-level CAN, compiler cost, and broad significance remain decisive. This unit
can become a reviewed component or be killed; either outcome feeds successor
search while the new-paradigm objective stays active.
