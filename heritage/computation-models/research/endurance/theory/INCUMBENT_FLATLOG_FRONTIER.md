# Unit 3 / 3b / 3c report: flat-log incumbents vs the compact reference (BUILD_PLAN v1)

Rigor: EMPIRICAL, finite, Python reference. Command: `python -B experiments/cold_runner.py --frontier 8192,16384,32768`.
Raw output: `results/cold_runner/frontier_unit3_b.json` (rev b; `frontier_unit3_a.json` is the superseded first run).
Incumbents and their permission ledger: `experiments/incumbent_flatlog.py`. Harness: `COLD_RUNNER_SPEC.md`.
Nothing here is a theorem, an asymptotic claim, or an absorption/novelty verdict.

## Correction history (read first)
Rev a of this report compared the compact reference with a flat log restricted to O(log k) local workspace
(rescans) and found compact used fewer worst flip reads (1.9x, 3.8x, 12.9x). Gate-4 review (Codex, 2026-10-07)
showed this was an artifact of a SELF-IMPOSED restriction: the normative compact model allows S = 256 q^2
workspace bits, and a flat log whose fold buffer needs only M*d (108 to 462) bits fits easily. With the
permitted buffer the flat log uses FEWER worst flip reads than compact. The label "privileged" for the
buffered variant was wrong and the SAME entry for exhaustion semantics in the ledger was too strong. Both are
fixed in `incumbent_flatlog.py`.

## Contract and measurement
k = N/2, L = (log2 N)^2, binary cells, strict INVALID exhaustion. Two workloads per policy: HAMMER (bit 0
always; lifetime adversary) to first failure or a cap of 2*H0+10 flips (H0 = M*L, compact's guaranteed
lifetime) and SPREAD (distinct indices; worst case for folds). Lifetime = hammer lifetime; worst-command
probes = max over both workloads (so these are LOWER BOUNDS on true worst case: the 8192 compact two-epoch
diagnostic measured 289 query reads vs 211 here). Python constructor/setup cost and ordinary work are
uncharged. Flat logs are sized to the smallest M with M*floor(L/2) >= H0 (a minimum-lifetime control, not an
equal-capacity comparison: most of the free cells stay unused).

## Results (equal guaranteed lifetime)
| N | policy | lifetime | worst Q reads | worst F reads | worst F writes | workspace / retained bits |
|--:|---|--:|--:|--:|--:|--:|
| 8192 | direct | 169 | 1 | 1 | 1 | 0 |
| 8192 | compact | 678 | 211 | 857 | 704 | S<=256q^2 (not measured) |
| 8192 | flat log, O(log) workspace (rescans) | 764 | 106 | 1,656 | 139 | O(log k) |
| 8192 | flat log, buffered (cold) | 764 | 106 | 135 | 139 | 108 |
| 8192 | flat log, RETAINED cache (NOT cold) | 764 | 1 | 9 | 139 | 112 retained |
| 16384 | compact | 1,568 | 444 | 1,471 | 796 | |
| 16384 | flat log, buffered (cold) | 1,568 | 199 | 256 | 254 | 208 |
| 16384 | flat log, RETAINED cache | 1,568 | 1 | 16 | 254 | 213 retained |
| 32768 | compact | 3,608 | 383 | 1,919 | 915 | |
| 32768 | flat log, buffered (cold) | 3,728 | 454 | 561 | 543 | 462 |
| 32768 | flat log, RETAINED cache | 3,728 | 1 | 33 | 543 | 468 retained |
Full-size flat log at N=8192 (all free cells): lifetime >= 26,500 (reached 26,774 in an earlier run), worst
query 3,796 reads, worst spread fold 1,937,565 reads (O(log) workspace variant).

## What the data say (finite, tested instances only)
1. Against a workspace-matched cold flat log, the compact reference is dominated on all three read/write
   metrics at N = 8192 and 16384, and on flip reads and flip writes at N = 32768. Its only edge is worst
   query reads at N = 32768 (383 vs 454).
2. Scaling: the flat log's query cost is M*(d+1), linear in the log size (about N/512 at the construction's
   target lifetime NL/(1024 log N)); compact's claimed bound is polylog (32 q^2). A crude crossover is near
   N ~ 2^27 (consistent with the vacuity finding), and only against an UNINDEXED log. A sorted-run or
   persistent-dictionary incumbent (Chandran-Kanukurthi-Ostrovsky hierarchical buffer; DSST node-copying,
   with bit-priced pointers and wear-aware reclamation) could remove that gap. Untested.
3. UNIT 3b (cold-sensitivity): retaining a cache of 112 to 468 bits across commands (within the allowed
   S) cuts query reads to 1 and flip reads to 9 to 33. The compact design's cost is therefore the price of
   forbidding retained state; its benefit exists only where state is lost between commands (power-cycled
   devices). No application showing that need has been established (BUILD_PLAN unit 4 criterion iii).
4. The compact reference buys little lifetime at these sizes (large constants: b = 271 bits per record
   at N = 8192); the same cells in a full-size flat log give ~39x its lifetime at linear query cost.

## Status against BUILD_PLAN unit 4 (gate)
(i) monitored diagnostics survive: yes (reference scope). (ii) a specified plausible useful gap: NOT MET at
feasible sizes (the only edge is a linear-vs-polylog query scaling that appears around N ~ 2^27 and only
against an unindexed log). (iii) a named application needing the cold boundary and a regime where the gap is
useful: NOT MET. Therefore the single compact repair is NOT spent. Roundtable gate-4 verdict: unanimous
BUILD-STRONGER-INCUMBENT-AND-RUN-3b-FIRST (3b is done here). The stronger-incumbent build (sorted-run /
DSST transfer) remains OPEN.

## Limits
Two workloads only; worst-case figures are lower bounds; ordinary work and Python constructor cost
uncharged; compact is the frozen Python model, not the repaired bit-program; hammer lifetime is not
universal lifetime; one incumbent family; the retained-state control is a different contract.
