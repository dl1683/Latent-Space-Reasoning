# Cold-transducer runner: ports, meters, leak checklist, class-level translation

Roundtable BUILD_PLAN v1, unit 1 (distance 1). Status: SPECIFICATION of a test harness, not a theorem.
The normative machine definition stays `ENDURANCE_COMPACT_PROBE_MODEL_CANDIDATE.md`.

## 1. Ports (the only way a policy touches persistent state)
- `read(address) -> bit`: acquires one persistent bit. One probe, always counted.
- `write(address, bit)`: sets one persistent bit. One probe, always counted, even if the value is
  unchanged. A value-changing write adds one unit of wear to that cell. A same-value write adds none.
- Exhaustion: a value-changing write to a cell that already has L changes is INVALID. The runner raises
  `Exhausted`. It returns no flag, no saturation, no timing signal to the policy. The monitor records it as
  a violation of the tested instance.
- Wear counters, the logical vector and the oracle belong to the checker. A policy never sees them.
- No other state crosses the command boundary. The runner builds a FRESH policy object from (N, k, L) for
  every command, hands it a slot-only ports facade (exactly `read`, `write`), keeps wear, tape, oracle and a
  LATCHED violation flag in checker-private state, and deep-compares class-level attributes across the whole class hierarchy (MRO) between commands (plain data only: numbers, strings, bytes, None and lists/tuples/sets/dicts/bytearrays of them; callables, descriptors and dunders are excluded; any other class-level object is REJECTED fail-closed), routes every failure (construction, execution, boundary checks, non-contract exceptions) through one finalizer that reports the failing command's traffic, and checks that a declared scratch tape (attribute `tape`) is present, all zero and of unchanged length at the end of every command (so object replacement cannot hide a missing reference wipe).
  A policy that catches `Exhausted`/`QueryWrote` is still failed by the latch.
- Scope of the harness: it is for HONEST policy code. It catches accidents and makes leaks auditable. It is NOT
  a sandbox against an adversarial policy author (Python stack inspection, module globals and process-wide
  state remain reachable; a hostile policy needs process isolation or an interpreted policy interface).

## 2. Commands and tasks
`INIT(raw)` writes the raw k-bit vector exactly once, from an all-zero unworn tape. Wear from INIT counts.
`FLIP(i)` toggles logical bit i. `QUERY(i)` returns logical bit i and must not write. The oracle holds the
logical vector. After every QUERY the runner compares the answer. A query that performs any write is a
violation, regardless of whether the value changes.

## 2b. Required diagnostic (rev 2)
`two_epoch_all_ones`: INIT all ones; at least 2M+2 (and at least 12) FLIPs over a mix of cancelling pairs on
bit 0, distinct bits, a repeated bit and a hammered bit; QUERY every touched index after every FLIP; then
QUERY all k bits. M is the compact layout's record count, so the trace passes folding, clearing and bank
exchange. The self-test shows that a policy whose FLIP does nothing FAILS this diagnostic (the earlier
two-flip diagnostic let it pass). `hammer_bit0` runs separately from an all-zero tape past each policy's
lifetime to expose the first violation.

## 2c. Review status (honest)
Three Codex review rounds on the harness each found repairs (BUILD_PLAN rule: three consecutive repair rounds
without an admissible result is a stop condition). The harness is therefore ACCEPTED WITH DOCUMENTED LIMITS,
not declared clean: honest-policy scope only; caches held outside the policy class hierarchy are audited by
reading; the Python constructor/setup cost is not charged; no ordinary-work proxy. Further hardening is
deferred in favour of unit 3, the actual test of the central bet.

## 3. Meters (reported separately, per command and as worst case)
reads, writes, value-changing writes (= wear increments), maximum single-cell wear, lifetime H (number of
FLIPs survived before first violation), preparation cost (INIT probes), and ordinary command work.
**Ordinary command work is NOT measured by the Python reference.** The report carries the field
`ordinary_work = UNAVAILABLE` (no proxy interface exists); any future proxy must be labelled PROXY and is not an
ordinary-machine bit-work bound. The policy constructor (layout computation) is re-run every command and its work is NOT charged: the report says so in `setup`. There is no proxy interface. The report also separates INIT reads/writes/changes, a per-cell wear histogram, an optional replayable per-command trace (`keep_trace`), the traffic of the
failing command (preserved even when it raises), maximum single-cell wear, cells at maximum wear and total changes.
One address implementation is used for every contender (plain integer address, one probe per bit).

## 4. Outcome vocabulary
`PASS(policy,N,L)` means all checks held on the tested histories only; the runner emits it as the field `outcome`. `FAILED-CONSTRUCTION(policy,N,L)` is a
wrong answer, a query write, an exhaustion or a retained-state change. Neither is a theorem. `OPEN` is
reserved for incomplete incumbent translation. `ABSORBED` requires a faithful uniform transfer matching the
full target and is never produced by this harness.

## 5. Infinity-leak checklist (checked per policy; a policy fails the audit on any "yes")
L1. Stage-indexed or parameter-growing policy tables supplied as free ROM?
L2. Wear counters or other history readable by the policy?
L3. Mutable Python/host state retained across commands (heap objects, caches, counters, RNG state)?
L4. Addresses, heads or indices carrying history outside the inventoried workspace?
L5. Cleanup, wiping or setup delegated to the host rather than done by the policy?
L6. Endpoint-only wear (omitting intermediate changes later undone)?
L7. Free randomness, retained seeds, or multilevel/word cells treated as bits?
L8. Local computation declared free while only probes are counted, then read as a latency claim?
L9. Preparation outside INIT (precomputed layouts that grow with N beyond fixed parameter-derived constants)?
The runner mechanically enforces L2 (slot-only facade, checker-private state), L6 (all value changes counted)
and, for honest code, L3 within the policy class hierarchy (fresh object per command, MRO-wide deep class-state comparison, scratch-zero check). Caches held OUTSIDE the policy class hierarchy (module globals, unrelated classes, files) are NOT detected and are audited by reading each policy. L1, L4, L5, L7, L8, L9 are audited by reading each
policy and recorded in the incumbent translation ledger.

## 6. Class-level translation (the "is this a model or a data structure?" test)
**Observation, narrowly stated.** The no-retained-history convention (each operation starts from its
argument; all history sits in persistent memory) is the convention of the cell-probe / bit-probe dynamic
data-structure abstraction. Cold programs EMBED into that abstraction: any cold program is a cell-probe
algorithm whose probe count is its probe count. The embedding is not an equivalence. The cell-probe lower
bounds (Patrascu-Demaine, "Logarithmic Lower Bounds in the Cell-Probe Model"; Larsen's dynamic
data-structure work) allow unbounded internal state during an operation, free computation and state loss
afterward, and count only probes. The cold contract here ADDS obligations those models do not impose:
bounded inventoried workspace S, executable cleanup and canonical return, separately charged ordinary work,
paid preparation, and the per-cell write budget. So cold programs are cell-probe algorithms with extra
obligations: cell-probe lower bounds transfer to cold programs, not the reverse.
**Permissions of the endurance sources (specific, not "i.e.").** Chee et al. (endurance-limited memories)
model an exhausted cell as RETAINING its value on further programming attempts, not as INVALID, and vary what
the encoder/decoder know (count knowledge vs visible state vs neither); their encodings are stage-indexed.
Onodera-Shibuya (wear leveling) allow randomness, word-sized cells and retained wear-free memory. This spec's
contract: exhaustion INVALID and unobservable, bit cells, no randomness, no wear-free state, no stage index.
These sources therefore establish OCCUPATION of related settings, not faithful absorption of this target.
Each incumbent translation records the permission differences (exhaustion, wear visibility, operation
ordinal, retained state, cell width, guarantees).
**The candidate class.** Cell-probe/bit-probe dynamic problems with a per-cell write budget, bounded
workspace and executable canonical return. FLIP/QUERY is the first witness; a finite command/answer task is
any deterministic specification (finite logical state space, commands, answers, transition function)
implementable by ONE fixed finite bit-program parameterised only by (N, k, L).
**Obstruction (honest).** Every such task is implementable by an ordinary RAM that keeps the logical state in
ordinary memory with no endurance limit. The class adds a resource accounting, not a computability notion.
A "new model" claim must come from a lifetime/access/work frontier that survives translation of ELM, wear
leveling and rewriting-code constructions. If absorbed, the supported outcome is ABSORBED (unit 8).
**Cold-sensitivity (unit 3b).** Because the cold rule is the cell-probe convention, the sensitivity control is
"allow B retained controller bits across commands" and compare charged costs. A benefit that disappears at
small B means the frontier depends on an unusually strict controller, not on endurance.
