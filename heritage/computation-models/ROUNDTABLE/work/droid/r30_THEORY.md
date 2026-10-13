# Bulk-operation pricing and strongest-comparator attack (droid, 2026-10-09)

**Verdict: CONJECTURE (author-side), no mission-level survivor.** These
elementary reductions are transparent but **not** PROVED under AGENTS.md:
there has been no fresh independent proof attack. No physical energy lower
bound is claimed. Register: `r30_PREREG.md`; exact finite code:
`r30_exact.py`; occupancy: `r30_SOURCES.md`.

## Exact contract and what a step could mean

Task `CLEAR(n)`: starting from n arbitrary independently variable binary
physical cells, finish with all *those same cells* at zero, and reveal
termination. Distinct from a logically zero array that can be represented
by changing an epoch bit: the latter is cheaper for **both** machines,
until a future physical inspection exposes stale cells. Task `AND(n)`:
consume n independent bits and output their exact conjunction. `SUM(n)`:
output the exact integer sum in `ceil(log2(n+1))` bits. For a new request,
include preparation, distribution and readout at both ends.

Meter `(T,A,E,W,I)`: latency in synchronized local rounds; installed cell,
driver and wire area; active switching/driving energy or *capacity* at a
stated supply voltage; total wire length/bandwidth; number/timing of input
ports. E is not assumed equal to number of logically erased bits. The
strong ordinary comparator can build ordinary splitters, reduction gates,
parallel local controllers and memories up to the **same A,E,W,I**, and
uses the same class of interconnect/physical devices. Comparing with a
single-head sequential RAM instead is a weaker-comparator engineering
comparison.

**Infinity audit:** A fixed n-bit word is finite, but claiming a *uniform*
one-step arbitrary-n word without paying n-bit storage and ports imports
unbounded bandwidth as n grows. A one-wire broadcast to n sinks with
fixed input drive while assuming fixed reliability and zero added
capacitance imports unbounded drive. An analog sum with exact arbitrary-n
output but fixed readout resolution imports unbounded distinguishability;
even though there are only n+1 outcomes, output precision grows as
log(n+1). A repeatedly used command signal does not erase the n physical
endpoints' reset transitions from a write energy budget.

## Conditional equal-price arguments

1. **Fan-out-two CLEAR.** If a new command enters one designated cell
   and each informed source can retain one copy and forward one per round,
   the number of informed cells after t rounds is at most 2^t. Hence
   delivery to all n requires `ceil(log2 n)` rounds, plus whatever cell
   write settling the contract charges. A balanced broadcast tree has
   n-1 branch/relay events, O(n) nodes and `ceil(log2 n)` rounds. On a
   resident n-cell square nearest-neighbor grid, if signal speed is
   at most one edge/round and entry is the corner, the opposite corner
   is distance `2(s-1)` for n=s^2. Flooding by a breadth-first spanning
   tree attains that bound up to the final write, using n-1 unit edges.
   These are **different physical-placement contracts**. A true n-way
   constant-time driver must provision n recipients, their wires,
   signal/bandwidth/energy capacity, or an externally distributed field.
   At the same paid resources the ordinary device can use the same
   terminals and local controllers. If instead one source costs one
   round while the *ordinary* target gets only a serial port, the
   n-to-1 time ratio is caused by unequal access bandwidth.

2. **Bounded-fan-in AND.** An output after d layers of fan-in-two gates
   depends on at most 2^d independent inputs. AND depends on every input
   (fix all others to 1), so d >= `ceil(log2 n)`. A balanced tree
   reaches that depth with n-1 AND gates and n inputs. To argue
   *n-1 gates* for arbitrary AND circuits, note each binary gate can
   merge at most two previously disjoint input-dependence components
   on its way to the sole output; the exact subset-DP code treats
   AND-only read-once trees, not all Boolean circuits. This is a
   circuit-depth and area argument, NOT a bound for unrestricted
   geometrical chips. Under Jin/Williams/Young's different fixed-layer
   wire-delay VLSI contract, exact AND already has matching
   Theta(n^(1/3)) delay. In a stationary square grid with inputs
   resident on cells and output fixed at the corner, dependency
   propagation forces Omega(sqrt n), and a convergecast tree
   achieves O(sqrt n) for one-bit AND.

3. **SUM rather than AND.** A binary addition tree, using conventional
   O(log n)-depth adders for O(log n)-bit partial sums, needs at most
   O(log^2 n) depth and polynomial aggregate gate area; the actual
   total input work is at least n under independent input reading.
   Faster known carry-save reductions only improve this ordinary
   upper bound. With single-bit nearest-neighbor links, aggregate
   partial sums have O(log n) bits so an ordinary grid can achieve
   O(sqrt n log n) by local reduction and bit-serial routing with
   polynomially comparable n-cell hardware. That upper bound is
   enough to exclude a >polylog advantage for a **same-grid** bulk
   input (which itself must cross Omega(sqrt n) distance to a fixed
   output); no claim of an optimal grid SUM algorithm here.
   An analog crossbar may have excellent finite energy/latency,
   but exact n+1 distinguishable outputs, noise and ADC cost must
   be accounted for. Gonugondla et al. study this tradeoff, not a
   universal exact-sum impossibility.

4. **Energy is not automatically n kBT ln 2.** A physical reset of n
   independent uniformly random unknown bits, with no preserved side
   information, decreases logical entropy by n bits. Coupling that to
   dissipated energy needs temperature, reservoir, cycle/reliability
   and controller assumptions, which round 30 has **not** established.
   One worst-case initial pattern is not an ensemble heat law. For a
   fixed-voltage per-cell switching-capacity meter with a positive
   minimum charge per changed cell, the all-one input requires n
   charged endpoint events (uniform input expected n/2), regardless
   of whether they occur in parallel. That **assumed** meter kills a
   constant-**energy** SETALL, but permits O(1) time with n-way
   parallel write bandwidth. An ordinary design given the same n
   ports matches that time and energy order.

5. **Generic price-threshold inequality, under explicit access models.**
   Let `F>=2` be the maximum number of successor copies per informed
   signal per round, with all fan-out drivers and n terminal receivers
   counted in installed area. One injected command reaches at most
   `F^t` recipients at round t, so `T>=ceil(log_F n)`. The ordinary
   balanced F-ary distribution network reaches all n in
   `ceil(log_F n)` rounds and O(n) total devices/edges when F is
   bounded; for growing F it uses O(n) output edges plus suitable
   paid high-degree drivers. **This lower/upper pair bounds the
   *extra time ratio* by a constant inside a common F-ary access
   class, independently of source instruction names.** For an
   arbitrary `F=n` high-degree broadcast, both can use the same
   n-terminal access, while a bounded-fan-out ordinary replacement
   is slower by at most O(log n) *if* the source already provisions
   Ω(n) area/drive capacity. If the source gets an input-dependent
   field that the target cannot implement with its device alphabet,
   shared area/energy numbers alone do not prove a simulation;
   that is an explicit unmatched primitive, not a theorem of
   physical impossibility.

   For aggregation, replace fan-out F by maximum **fan-in** G>=2,
   with n separately presented input values and one output. Any
   depth-d G-input Boolean circuit has support at most G^d;
   a balanced G-ary AND tree achieves the corresponding logarithmic
   depth, provided the charged area supports n inputs and reduction
   gates. In an area-limited grid with bounded propagation speed,
   the output's backward causal cone is bounded by reachable
   positions; a resident input outside that cone cannot influence
   the result. Ordinary flooding/convergecast saturates the corner
   distance bound on the fixed square grid. **Do not apply the
   grid bound to Jin et al.'s time-multiplexed input ports or to
   devices with extra spatially distributed actuation.**

**Provisional threshold (CONJECTURE):** For CLEAR and AND in the above
local device families, there is no finite honest `A,E,I` price that
preserves a super-polylogarithmic latency ratio versus all ordinary
devices with equal priced resources. *Charging the n endpoints alone
does not kill a latency gap versus a deliberately serial target*;
granting the comparator matching fan-out/parallel ports or charging
distance does. Conversely, a gap could only appear if the source
can couple to n endpoints (or collect n independent inputs) in
sub-polylog rounds using a coupling **unavailable** to any
equal-price ordinary machine. That is a *new physical access
primitive* plus a matching-device lower bound, not a SETALL
instruction or proof. This conditional statement covers these
families, not every conceivable technology.

## Price-vector comparison (orders only, technology assumptions explicit)

| Contract / route | Bulk-like implementation `(T,A,E)` | Strong ordinary implementation at same *order* of A and E | Gap |
|---|---|---|---|
| Ideal shared-command PRAM, n writable cells and n local controllers | `(1, Θ(n), Θ(n))` active cell operations; free global broadcast is **PRAM abstraction** | n local processors, one round, n operations | None; serial CPU comparison gives n but changes processor/bandwidth count. |
| Bounded-fanout distributed pulse, unit-length wires, fixed per-toggle energy | `(Θ(log n), Θ(n), Θ(n))` including tree | Same fan-out tree and cells | None. |
| Fixed corner injection on s-by-s nearest-neighbor resident grid | `(Θ(s), Θ(n), Θ(n))` with worst-case all-one input and charged local writes; s=sqrt n | Ordinary BFS broadcast / reverse reduction over same grid | Constant-factor at this coarse meter. |
| High-degree star with ideal long-wire unit delay | `(1, >=Θ(n), >=Θ(n))` **conditional** on enough drive/wire capacity; Thompson prices wiring/nexus, not dynamic energy | Tree at O(log n) delay, Θ(n) area and switching capacity, or same high-degree driver when equal technology is permitted | At most log n in either formulation. |
| Large word width w=n, unit-cost CPU instruction | `(1, Θ(n) bits of register/ports, unknown physical E)` | Word-RAM with same w or banked n-way writes | No difference with equal instruction/port access; abstract CPU-versus-bit-probe is unequal width. |
| CAM / DRAM in-memory row operation of fixed finite row size | Few memory cycles per row, row hardware and switched capacitance not zero | A controller/ordinary machine authorized the same row primitive performs identical operation | None at equal access; practical win versus off-chip byte loop remains real. |
| Exact analog n-way reduction | Potential low latency, but n receivers and at least log(n+1) resolved output bits plus noise constraints | Binary reduction tree O(log n) for AND; polylog for exact SUM | A >polylog gap needs a technology-specific *matching lower bound* against ordinary devices using comparable coupling; none established. |

For the third and fourth rows, Θ(n) energy means the **specified
per-toggle capacity meter**, not a universally derived joule or Landauer
bound. For AND, no n-cell overwrite is required, so switching energy need
not be Θ(n) on every input. The grid comparison concerns *resident*
inputs/outputs; allowing time-multiplexed input ports changes its
Θ(sqrt n) statement (Jin et al. explicitly show Θ(n^(1/3)) in another
contract). For SETALL, a lazy-zero tag changes the output contract, and
an equal-interface ordinary implementation can also use that tag.

**Honest one-sentence so-what:** In-memory bulk initialization is useful
for avoiding needless CPU/memory-channel traffic, but it is an already
studied hardware improvement, not a new computation model with a
matched-price CAN/CANNOT advantage.
