# Uniform cold bit-probe endurance model: candidate definition

Date: 2026-10-04. CANDIDATE / FRESH DESIGN-INFINITY-TRANSFER GATE PENDING.
New model, not a reinterpretation or repair of the indexed microcontroller.
No capability theorem, execution, novelty or mission admission in this file.

## Parameters, finite state and task

Integers N>=k>=1 and L>=1 are public, binary encoded parameters.
q=ceil(log2(NL+2)), a=ceil(log2(N+1)), S=256q^2. Logs in resource targets are
mathematical real logs; executable dimensions use integer bit lengths. The
algorithm need not evaluate a real logarithm or the target H0.

DATA contains exactly N persistent binary cells, initially all zero. Hidden
wear counters w_j initially zero count EVERY actual change, including INIT and
undone intermediate writes. Their finite analysis range is0..L. They are
resource bookkeeping, not readable storage, advice, clocks or input ports.
The entire physical resource includes N cells and their declared wear limits;
no claim identifies a real device's energy, analog state or durability law.

One fixed finite deterministic ordinary bit-program P is independent of the
parameter values. Its full finite source/provisioning size is reported for a
CAN; there is no parameter-growing program/policy ROM or noncomputable oracle.
P may read only public encoded N,k,L, the current input and DATA through probes.
Immutable program and parameter bits contain no raw input or caller history.

All mutable local information totals at most S bits at EVERY instant of
preparation, INIT and commands: control/PC, tape/input/program head positions,
physical/private addresses, latches, counters, arithmetic operands, path frames,
transcripts, buffers, allocation bookkeeping and cleanup state. No host stack,
local word, library scratch or implicit head position escapes this inventory.
P is an ordinary computable finite program; local steps are finite bit-level
operations. Free local work means price zero in the probe coordinates, not
unbounded space, an oracle, automatic halting or an implicit cleanup operation.

INIT receives only the current raw x in{0,1}^k on a finite read-only input tape.
After INIT that input is inaccessible. QUERY_i and FLIP_i receive only the
current opcode and i in0..k-1. Index means logical coordinate, never invocation
ordinal. Every promised command actually halts. No elapsed-time signal, stage,
output cursor, retained random seed, readable wear, prior input or output cache.

## Literal persistent probes and canonical boundary

READ(j), 0<=j<N, acquires exactly DATA[j] and costs one read probe. Every
acquisition and reread counts, including direct copy/Boolean operands. No
free word, bit-vector snapshot, search, rank, scan or indirect memory operand.

WRITE(j,b), b in{0,1}, sets DATA[j] to supplied b and costs one write probe,
including a same-value write. It returns no old value or status. A value
change increases w_j by one. If that would exceed L, execution is INVALID;
there is no caught failure, saturation, unchanged-write hint or feedback branch.
Read-modify-write therefore needs an explicit READ of the old value.
Invalid addresses are INVALID and provide no useful observation either.

Local operations manipulate only inventoried private bits/current immutable
input/parameters/program. Local bit-work is unpriced in R/W; ordinary local
execution work may be reported separately. Real latency/energy and a cheap
Turing-runtime theorem do not follow from a probe bound.

QUERY is DATA-read-only. Output is the current one-bit answer, emitted through
a write-only owner interface. Previous outputs and their positions cannot be
read. FLIP produces acknowledgement after finishing all persistent maintenance.
No QUERY, caller inspection or crash occurs inside a command; completed-boundary
durability is the only promise. No concurrent or mid-command crash theorem.

Before acknowledging INIT/FLIP or completing QUERY, P actually restores every
mutable local bit/head/control location to the same parameter-only canonical
image. It provides its finite clearing/return routine. A free automatic reset,
an uncharged scratch extension or retained allocator is NOT a model operation.
The canonical input interface exposes only that command's immutable input;
once it returns the next caller input replaces it. This carries no history.

## Uniform provisioning and ordinary preparation work

Provision one fixed P, then receive the encoded parameters. There is no
parameter-specific installed CODE image in this abstraction. The constant
program bits and public parameter bits are reported separately from mutable
space; their head/PC/address information is in S. P cannot call an external
generator/synthesizer during operation or consult its output as hidden advice.

Parameter preparation and INIT must use at most S mutable bits and, for the
registered CAN, ordinary bit-work <=2^20Nq^6. Count parameter/raw input reads,
local bit operations/control transitions, and each persistent probe as a step
in this preparation coordinate. Supplying the fixed program is reported finite
provisioning, not a growing execution table. Command probe targets alone do not
establish an ordinary command-time bound.

This is explicitly DIFFERENT from the old fixed256-microstate/65536-description-
bit, append-only CODE-generation kernel. None of those closed construction
claims transfers merely by releasing the restrictions. The new uniform fixed
program and total private-space/preparation obligations replace them openly.

## Correctness and resource quantifiers

A lifetime guarantee H is a nonnegative integer function of N,k,L. For EVERY
raw input and EVERY finite sequence of at most H FLIPs, with arbitrary finite
QUERY interleavings and arbitrarily many terminal queries afterward:
- INIT and every command halt legally and return canonical;
- each QUERY_i equals x_i xor parity of previous FLIP_i occurrences;
- no intermediate write exceeds any individual cell's wear L;
- the same P, source/provisioning and numerical resource bounds apply.

Let R_Q be maximum actual persistent read acquisitions over all promised
queries/inputs/histories. Let R_F,W_F be corresponding maxima for FLIPs.
Report INIT reads/writes separately, plus every-cell cumulative actual changes.
Same-value writes consume traffic even when wear zero. Maxima for R_F and W_F
may be attained by different histories; bounding their sum is the conservative
registered requirement. Data size N counts every surviving metadata bit.

## Requested first-gate transfers and infinity audit

REQUESTED, not admitted here: canonical deterministic entry and absence of
history/wear feedback induce stationary I(x), T_i(s), d_i(s) on promised states.
Extend arbitrary off-promise maps only for the endpoint relaxation; such an
extension supplies no actual executable program. Endpoint changes undercount
all intermediate wear, so the existing endpoint ceilings may transfer as CANNOT.

Fresh gate must separately attack:
- universal A/B/C, initialization/fibre/final-branch and floating-code transfers;
- H+1<=L min(N,2^r_i-1) for each bit's worst query depth r_i;
- candidate fixed-support/nonadaptive H+1<=L min(N,|P_i|), with no history-derived
  query address set supplied free; this is PENDING REVIEW, not new novelty;
- finite input/symbol/control/workspace and explicit halting; no infinity through
  free local computation, uniform source, immutability, fatigue or output timing;
- fair incumbents under the SAME cold/wear/storage/probe/space permissions.

## Candidate construction inventory and limitations

For the forthcoming crit-bit family, candidate available layout within S:
96q control/parameter/address bits; 4q(q+1) path-frame bits; eight node buffers
of v bits; 32q^2 arithmetic bits; 64q counters/cleanup bits. Since v<=5q+6,
this envelope is <=36q^2+204q+48<=256q^2 for q>=2. This is a reserved-capacity
envelope, NOT a supplied program or proof that every actual temporary fits it.
The construction must map all simultaneous live variables, program/input heads
and clearing routines into it and certify exact worst probes/preparation work.

No theorem is labeled PROVED until a fresh independent actual Codex attack
passes. Source ingredients are occupied; source OPEN cannot become DELTA.
Narrative: we are testing whether a fully cold finite controller can obtain
long lifetime and cheap persistent accesses. Until construction and comparator
proofs pass, the new-paradigm mission and terminal standing remain OPEN.
