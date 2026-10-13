# Finite indexed closed-persistence machine: explicit serial controller

2026-10-04 NEW DEFINITION CANDIDATE, not yet design-admitted. Supersedes only
ENDURANCE_INDEXED_KERNEL_DRAFT.md as a proposal; old admitted kernel/hash and
exhausted cubic unit are unchanged. Actual two-round direction is
STEERING_CONTROLLER_DECISION.md. No theorem transfer or positive CAN admitted.

## Task, history and wear

Finite integers N>=k>=1,L>=1,0<=H<=NL; a=ceil(log2(N+1)),
q=ceil(log2(NL+2)), S=256q^2, B=65536q^2,
c=ceil(log2(S+1)), b=ceil(log2(B+1)),
e=ceil(log2(max(k,a+2)+1)). Exactly N initially zero/unworn physical binary
data cells. Each actual bit-value change consumes wear1, including raw
initialization and every intermediate write. Same-value write costs work,
wear0. Opposite-value write when wear already L is INVALID, not a failed
operation that can be observed. No saturation, exception recovery, timing,
readable wear or exhaustion flag can guide a promised execution.

Initialization consumes arbitrary raw x in{0,1}^k then loses access to it.
Only current QUERY(i)/FLIP(i) input is available, not a step number, past
output, external clock, randomness, future input, cache or retained history.
All initialization executions terminate; every promised QUERY through depthH
terminates exact/read-only; every FLIP beforeH terminates with exact logical
x_i toggled. Queries remain correct indefinitely afterH. No postH FLIP or
midcommand crash/overlap promise. Every completed command restores ALL
volatile/PC/control/port state to the same parameter-only canonical image.
All history-dependent surviving information is in the N worn data cells.

## Finite planes and uniform installation

Exactly S volatile bits include every mutable microstate/latch/register/bus/
PC/head marker/stack/source cursor/return/arithmetic field. CODE is exactly B
immutable installed bits, padded zeros, sealed BEFORE raw x is exposed.
Uniform parameter-only generator has S working bits including its complete
controller. It emits CODE append-only: no rereading or rewritable output
workspace before seal. No input/future/history advice or runtime-created ROM.
Report useful image length, source/parameter reads, exact CODE emission/padding/
installation, generator space/work and fixed controller description.
Only this definition's ports can read sealed CODE during commands.

The base controller is a FIXED parameter-independent finite transition
description, at most256 microstates and at most65536 description bits.
Description is a reported/prepared part of the abstract machine, not an
additional parameter-growing opcode lookup table. Its current encoding (8
bits) and all mutable signals belong to S. Its transition relation acts on
a bounded number of named onebit signals/serial cells and invokes at most
one named port/local action at a time. A multi-bit microstate change pays
each changed bit; finite local Boolean logic pays each gate. There is no
second interpreter for this declared finite base relation. A construction
must supply its actual transition inventory/state count/description, not
assume the numerical cap proves existence. Comparators obey the same rules.

## Literal serial lane layout (base primitives of this NEW model)

Ten disjoint lanes, in this order: DATA address A widtha, PRIVATE address P
widthc, CODE address C widthb, INPUT address I widthe, program counter PC
widthb, branch target TARGET widthb, and four temporary lanes widthc.
For widthm each lane has m+2 cells numbered0..m+1. Each cell has FOUR counted
volatile bits: DATA, HEAD marker, LEFT tag, RIGHT tag. Address bits occupy
DATA cells1..m, least significant first. Endcell DATA bits are zero. Canonical
HEAD is onehot atcell0; LEFT tag at0 and RIGHT tag at m+1; all other tag bits
zero. Tags are constant during legal execution. All lanes plus8 microstate
bits and16 named onebit latches occupy

    S_base=4(a+e+3b+5c+20)+24 =4(a+e+3b+5c)+104.

Remaining S-S_base bits are ordinary private scratch. This is nonnegative
for every q>=2: a<=q,e<=q+2,c<=2q+7,b<=2q+15 imply
S_base<=72q+432<=256q^2. Fixed layout offsets are uniform finite parameters;
copying their bits into buses is paid, not a native word operation.

LOCAL LANE READ/WRITE accesses ONE DATA bit at its HEAD (or reads an end tag)
for work1 plus an explicit latch copy if needed. MOVE LEFT/RIGHT moves HEAD
one adjacent cell, work2 for the two changed marker bits. Movement beyond an
end is INVALID. Head position is ONLY the counted onehot marker; there is no
unrecorded numeric head, cursor or external selector state. Every local
operation uses one of ten fixed lane names. Selection of its marked local
position is the declared TM-like local-head primitive, not a free arbitrary
private-address gate or a physical circuit theorem. Simultaneous/wide shifts
are unavailable. Legal execution maintains onehot markers and constant tags;
PRIVATE writes to these reserved marker/tag cells are INVALID. PRIVATE
access to lane DATA, latches, microstate or general scratch may only preserve
legal control encoding and cannot create extra heads or hidden state.

All address formation, serial ripple increment, comparison, target/PC copy,
stack/arithmetic and clearing are explicit onebit routines using these lane
actions and indexed ports. No native word addition/multiplication, wide copy,
atomic helper call/return, variable-width branch or parameter-sized opcode.

## Explicit indexed ports and elementary work

READ DATA uses formed A, selectedcell0..N-1, chargea+1; WRITE DATA similarly
uses one named latch bit, chargea+1 and wear as above. PRIVATE READ/WRITE uses
formed P, selectedbit0..S-1, chargec+1. CODE READ uses formed C, selectedbit
0..B-1, chargeb+1. Every selected operation acquires or changes one data bit;
the address-buffer consumption is the explicitly PAID nonlocal resource.
It supplies selected value only, never wear/history. Invalid indices are
INVALID with no recoverable information. Moving the returned bit to other
storage/latches, filling buses or forming addresses costs additional work.
The base finite transition relation may read/write its named16 latch bits
and8 state bits, but pays every Boolean gate/copy/change. General PRIVATE
indexed selection of a bit never installs a wide bus or PC value.

Current-input port is finite immutable CALL input: for initialization k raw
bits; otherwise2 opcode bits plus a index bits. INPUT READ at formed I returns
one bit at worke+1, plus copies; invalid index is INVALID. This is a literal
charged bit port, with address/head/cursor state inside S, accessible only
for the CURRENT invocation; it never carries count/history/wear. Initial
parameter-source reads during generation are likewise finite/pay per bit.
Random positional reads of current raw x before initialization ends are
permitted and charged; no persistent original-input oracle after init.

EMIT returns one logical bit for work1 plus source reads/copies, then that
output is inaccessible. RETURN is permitted only with exactly canonical
volatile image (including PC/head/source/flags/latches); it cannot auto-clear
anything. Canonical image has the listed HEAD/tags and zero other bits; a
parameter-only entry image may additionally contain explicitly reported
fixed parameters, never caller history. Constructor supplies a real clearing
tail, including that tail's own cursor/PC/buses/microstate. Termination at
canonical image is part of the task promise, not a free cleanup primitive.

Thus W_RAM=W_local+R_N(a+1)+R_S(c+1)+R_B(b+1)+R_I(e+1).
R counts selected BIT accesses, including every CODE bit. Local counts all
lane reads/writes/moves, gates/copies/changes/emission. Double counting convention:
port tariff includes selected value access, while movement to/from non-port
storage/latches is separate; each actual local operation appears once.

Optional exact SCAN tariff on SAME traces: PRIVATE charge4S(c+1), CODE
charge4B(b+1), everything else unchanged. Serial-local lane access remains
local. It is sensitivity accounting, NOT an old-machine/physical simulation,
not RAM-confirmation-gating. No energy/latency theorem follows.

## Pending admission and positive obligations

Fresh infinity/design gate must attack information storage, finite primitive
and source closure, validity, dimensions/canonical image, nonlocal addressing
and cost conventions. Fresh transfer must attack stationarity/endpoint coverage
and old universal A/B/C/query-law proofs under this exact definition. Old
admission hashes do not automatically cover it. No computation begins under
this machine before design review; tiny registered endpoint calibration uses
the OLD admitted relaxation and owes later separate compiled-machine gates.

New controller unit's targets are budgets, not established costs:
uniform O(Nq^6) preparation in same S; CODE fitsB; worstcommand O(q^4),
informational scanO(q^8). One compiler family and one named repair. Same
compiler first Johnson k1, then fixed-trie/two-journal N2k lifetime target.
Actual field allocation, fixed microstates, instruction image, emitter,
generation space, execution and cleanup coefficients required before CAN.
Primary novelty/full-contract incumbent comparison/tight loss/low-r OPEN.
CPU only, GPU unused, no papers/new-paradigm/mission completion.
