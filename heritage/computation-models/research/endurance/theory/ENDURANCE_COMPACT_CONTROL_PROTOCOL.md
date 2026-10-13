# ENDURANCE-COMPACT-CONTROL-1: frozen logical CPU controls

Registered 2026-10-04, before control implementation, design/code admission,
smoke or outcome execution. Separate from the pure construction and conditional
adaptivity units. It consumes no new construction repair allowance.

## Purpose and exact scope

Exercise the final one-box crit-bit reference against an independent logical
vector and per-cell change monitor. Seek concrete bugs in pointer propagation,
leaf-copy filtering, cutoffs, publication, clearing and final wear. This is
finite EMPIRICAL model-level evidence, not a proof over all inputs/histories,
the ordinary private-space bound, compiled preparation work, novelty, actual
device performance or mission completion. Python heap is not S.

The construction worker must finish first. Freeze its final source hash and
construction hash in the control source receipt before fresh CODE admission.
Do not accept a source mutation after that gate without a fresh exact-source
gate. A mismatch between source and theory is reported explicitly; the control
cannot silently fix, weaken or reconfigure the frozen construction target.

## Frozen cases and finite coverage

Small direct-base controls: (N,k,L,H)=(2,1,1,0),(2,1,2,1),(4,2,4,3),(6,3,4,3).
For each tuple, independently enumerate EVERY raw vector and EVERY length-H
FLIP-index word; fresh zero DATA/wear for each raw/history execution. Check
all logical coordinates after INIT and each FLIP, plus two terminal query
passes. All shorter histories are covered as prefixes. N6/L4 is explicitly
outside the primary real-log domain and is only a fallback diagnostic.

Eight trie configurations: each N in {8192,16384,32768,65536}, k=N/2,
L=(log2N)^2 (integer powers-of-two dimensions), with two fixed pairings:
(all-zero raw, PATH-AND-OVERFLOW) and (all-one raw, SPREAD-AND-REUSE).
For each configuration, derive M from the frozen layout and execute H=M*L
FLIPs, completing every epoch and the final exchange. M is derived, not tuned.

PATH-AND-OVERFLOW cycles four epoch modes by epoch mod4:
0: all requests key0;
1: cycle [0,1,2,4,...,2^(d-1)];
2: first ceil(M/2) requests cycle that chain, then repeatedly request the last
   key from the first half;
3: alternate key0 and k-1.
All key positions are valid because k is a power of two. Full histories are
fixed functions of public case parameters and harness step, never controller
inputs beyond the current opcode/index.

SPREAD-AND-REUSE: epoch E (zero-based), step j (zero-based), m=M/2.
For j<m use ((E*m+j)*65537+17) mod k. For j>=m repeat the key from first-half
position (j-m) mod m. This deterministic sequence is not a controller seed or
a randomized guarantee. No adaptive choice of histories after observing results.

After every FLIP query the current key, key0, key(k-1), and (current key+1) mod k
(deduplicate). At epoch positions 1,M/2,M-1,M additionally query every key
touched in the current or immediately previous epoch. After FLIP H, query
every logical coordinate twice. Counts, duplicates and case order are fixed
before execution. No parameter grids, portfolios, extrapolation or H4 solver.

## Independent monitor and what it must verify

Each command receives a fresh ports facade exposing only read(address)->bit
and write(address,bit)->None. The controller never sees wear, expected logical
answers, history, previous outputs or monitor counters. Facade backing state is
harness evidence; changing the facade does not reset DATA or scratch.

The monitor records EVERY controller persistent acquisition and write,
same-value writes included. Every actual value change increments the individual
wear count, including INIT and changes undone later; exhausted opposite writes
raise a harness failure, never a controller-observable success/failure branch.
Addresses must be valid; queries must produce no DATA writes. Expected logical
state changes only by the current FLIP coordinate.

After each completed command, reused reference tape must be all zero. This is
an observed reference-tape check only, not evidence about every ordinary-program
head/control bit. Check highest individual wear<=L, actual R_Q and separate
max R_F+max W_F<=2^20q^3, plus source layout <=N and all actual outputs.
Do not turn combined per-command traffic into the sum-of-maxima certificate.

Independent observer traversal may inspect backing DATA without counting those
observations as controller probes. Keep observer and controller counters separate.
At each boundary check current published node/root prefixes, allocator<=3j,
valid pointer ranges, increasing critical positions, no current-tree cycles or
duplicate current leaves, and key/parity agreement with the epoch's independent
request vector. Frozen OLD is traversed only before clearing starts; after that
the observer must not demand a still-readable OLD tree. Check that QUERY at
j>=M/2 performs ZERO OLD acquisitions; clearing-half FLIPs likewise have no
OLD acquisitions. A folding FLIP that reaches j=M/2 may read OLD while folding.

Track all cells' change counts, arena and base maxima, ROLE exchanges and
STARTED changes. Record at least one exercised full-box copy, ancestor copy,
garbage leaf skip and active-leaf fold across the trie suite. If a requested
event never occurs, report coverage INCOMPLETE; do not add a new workload.

## Gates, fixed resource limits and evidence

Fresh actual independent DESIGN/code review must check the monitor's independence,
driver formulas, source identity, aggregate budget and outcome coverage. Allowed
smoke is the four small cases above plus the first TWO epochs of the N8192
all-zero PATH configuration with the same source/monitor; its partial horizon
is smoke only, never the FULL outcome. Code gate follows smoke, precedes FULL.

FULL is ONE fixed case suite in the stated order. CPU<=4 logical processors,
aggregate job RAM<=8GiB, elapsed wall<=3600s including child preparation, cases,
monitor checks, evidence publication and root receipt. Use Windows job limits
or an equivalently independently reviewed enforcement mechanism. No GPU.
No detached unconstrained child or wall reset per case. Cap/incomplete suite is
CONTROL-VOID, not success or theoretical refutation. A concrete counterexample
is CONTROL-FAIL and retains its complete finite reproduction.

Before launch freeze program, driver, monitor, runner, dependency/environment
and protocol hashes with Git HEAD. Save source snapshots, each case's command
and query outcomes/counters, per-cell final wear and final DATA, diagnostic
event counts and enough exact history formulas to independently regenerate
every exercised trace. Bind them through hashes and a root-observed exit
receipt. Preserve aborted and failing runs; no outcome-only latest pointer.

Fresh independent actual raw admission replays every finite case or validates
every emitted output/change sequence independently at frozen identities.
CONTROL-PASS only after complete suite and raw admission; no PROVED promotion.
If construction review is still pending, evidence remains control-only and
does not bypass it. Primary adaptive-incumbent comparison remains separate.

No reportable computation is launched by this registration alone. Next step
is design review after the final construction source is available.
