All three gates **PASS at the repaired contract**. No blocking defect survived the independent attack.

Reviewed [the repaired candidate](<LOCAL_PROJECTS>/new-computation-model/research/ADAPTIVE_ERASURE_FEASIBILITY_CANDIDATE.md:1), SHA-256 `3135ce537e5d594d62fc86a7c10c988894e7feb2cfaa82227550d3a883f72813`, after reading AGENTS, the full dual-loop skill and current orientation files, then REVIEW_1.

| Gate | Verdict | Supported scope |
|---|---|---|
| DESIGN | **PASS** | Explicit finite, private-seed, hidden-erasure READ contract; costs charged. |
| EXACT BENCHMARK | **PASS** | Rational minimax formula, including adaptive testers, revisits and the at-least-\(M\) promise. |
| FINITE-SEED TRANSFER | **PASS** | Dyadic lower bound and charged bounded-rejection approximation; no optimal-seed claim. |

**DESIGN.** The immutable source, hidden erased locations, intact first READ and erasure opportunities after every answered READ are now explicit. The adversary’s information excludes private seed/state and local timing. Oracle loading supplies access without revealing contents. These restrictions are load-bearing, and I found no hidden-infinity channel.

**EXACT BENCHMARK — PROVED at this contract.** Fresh uniform sampling gives conditional hit probability at least \(a_j/(N-j)\). Erasing live ones attains the product bound. For the universal lower bound, the ordered-subset hard distribution preserves symmetry on unseen addresses separately for zero and ⊥ histories. After \(s\) misses and \(d_s\) distinct queries, a fresh query has hazard
\[
\frac{a_s}{N-d_s}\le \frac{a_s}{N-s};
\]
a revisit has zero hazard. Every no-one transcript has a legal all-zero mirror. That mirror can use a predetermined erasure schedule, so it requires no seed access. Averaging produces a fixed input and deterministic eraser witnessing failure at least \(F\). Monotonicity is **nondecreasing**, and exact-\(M\) hard inputs remain admitted under the at-least-\(M\) promise.

**FINITE-SEED TRANSFER — PROVED at this contract.** Against each fixed input/deterministic eraser, failure lies on the \(2^{-r}\) grid, giving
\[
\operatorname{worst\ failure}\ge 2^{-r}\lceil2^rF\rceil.
\]
Bounded rejection gives exactly uniform ranks conditional on a successful round. For coupling, fix the original input and the finite tester’s causal erasure policy—including its program-dependent choices—and reuse that policy in the ideal reference. Histories agree until the first abort. Different stopping after abort is covered by its probability; conditioning globally on “no abort” is unnecessary. Thus failure is at most \(F+\eta\).

The lazy representation starts empty and stores at most \(h\) overrides; lookup, replacement and tail deletion require \(O(hb)\) bit work per round. This supports the stated conservative internal work/storage bounds without an \(N\)-entry initialization. Oracle loading/storage remain additional charges. For \(h=0\), accept with \(r=0,F=1\); for \(N-j=1\), rank zero needs no bits and cannot abort.

The controls recompute exactly:

| Parameters | Benchmark failure \(F\) | Finite-seed consequence |
|---|---:|---|
| \(16,4,1,q\ge4\) | \(216/455>1/3\) | Error \(1/3\) impossible. |
| \(8,4,1,4\) | \(16/105\) | \(r=72\), error \(\le33/140<1/3\). |
| \(16,8,1,4\) | \(128/1365\) | \(r=96\), error \(\le967/5460<1/3\). |

**EMPIRICAL bounded CPU attacks:** 252 posterior-DP cases, 252 uniform-sampler/adversary-DP cases, 51 exhaustive rejection kernels, 16,064 lazy Fisher–Yates transitions and 756 finite-kernel/adversary cases all matched. These check finite instances and representation behavior; they do not establish the general proofs or measure the bit-machine work bound.

The access model is already occupied, and zero-string randomness obstacles appear in primary literature. Exact-formula priority remains uncertified. [Kalemaj–Raskhodnikova–Varma](https://theoryofcomputing.org/articles/v019a001/v019a001.pdf), [Kelman–Linder–Raskhodnikova](https://drops.dagstuhl.de/storage/00lipics/lipics-vol325-itcs2025/LIPIcs.ITCS.2025.65/LIPIcs.ITCS.2025.65.pdf).

This admits a scoped feasibility control. Optimal seed complexity, novelty, terminal standing and mission completion remain open. No files were edited; no memory evidence, agents, GPU work or FULL run were used.