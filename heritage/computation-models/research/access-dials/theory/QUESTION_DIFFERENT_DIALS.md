**Priority for the steering dialogue: (1) adaptive audit access with a finite private seed, (2) recallable input mutation, (3) erasure-observer privacy.** These change, respectively, how future input reads survive intervention, which input cells may serve as temporary workspace, and what an observer can see. They are three different dials.

**Record status:** the bounded construction returned scoped NO-BINDING; the erasure results remain intact, novelty/usefulness remain unresolved, and July remains parked. [Construction record](<LOCAL_PROJECTS>/new-computation-model/research/OPERATING_BINDING_CONSTRUCTION.md:1), [mission review](<LOCAL_PROJECTS>/new-computation-model/research/OCTOBER_MILESTONE_REVIEW_1.md:3). Everything below is a **CONJECTURE, SPECULATION, or proposed test** unless explicitly identified as published prior. I performed reads and literature checks only.

**1. Adaptive audit access with a finite private seed — highest priority**

**Useful problem.** Audit a finite Boolean array for defects while its owner can hide entries in response to inspections. Distinguish an all-zero array from one containing at least \(\varepsilon N\) ones, with one-sided error at most \(1/3\). This models integrity checking of an evasive digital source; it needs no heat interpretation.

**Finite founding primitive.** `READ(i)` returns the original bit or ⊥ if that address has been erased. After each completed read, an adversary knowing the original array, program and public transcript may permanently erase up to \(t\) further addresses. It cannot see unused private seed bits.

One read costs one access operation. Constructing its \(\lceil\log_2N\rceil\)-bit address, all local bit operations and storage are charged separately. The entire randomness supply is an initially supplied independent uniform \(r\)-bit string; there are no subsequent random gates. Source loading, seed initialization and program generation are charged.

Let \(Q=\lceil1/\varepsilon\rceil\), \(t\ge1\), and initially restrict to \(N\ge(tQ)^4\). The **precise positive hypothesis** is a uniform tester with

\[
q=O(Q),\qquad
r=O\!\left(\log N+Q\log(t+1)\right),
\]

and bit work polynomial in \(q,\log N,\log t\).

**Fair incumbent.** Compare with independently sampled queries and existing online-erasure testers under exactly the same adversary, error, seed, access and generation contracts. Independent sampling uses \(O(Q\log N)\) seed bits here. An offline-erasure tester is a different contract. A public schedule is a negative control.

**Charged leak audit.** A finite seed specifies finitely many schedules; its apparent expansion creates no entropy. A workload-specific table of successful schedules is advice and must be generated and charged. The source cannot be queried during preparation for free. Seed independence is a mathematical input contract, not something finite sampling authenticates. Each new audit must account for its randomness anew.

**Nontrivial CAN route — derived candidate, pending review.** Start uniformly on an explicitly generated constant-degree expander. Between inspections, take \(k=\Theta(\log(tq^2))\) private walk steps, charging every neighbour computation and coin bit. Choose \(k\) so the powered transition satisfies

\[
\max_{u,v}P^k(u,v)\le \frac{1}{12tq^2}.
\]

Before inspection \(j\), at most \(t(j-1)\) addresses have been erased. Conditional on the public history, the probability of hitting one is consequently at most \(t(j-1)/(12tq^2)\). Couple this process with the unerased walk; a spectral hitting argument supplies the defect-detection route.

This gives a concrete **weaker candidate upper**

\[
r=O\!\left(\log N+Q\log(tQ)\right).
\]

The research target is removing the extra \(Q\log Q\) term without spending more inspections. Explicit expander generation is established machinery, not the proposed novelty. [Reingold–Vadhan–Wigderson](https://annals.math.princeton.edu/2002/155-1/p04).

**Universal CANNOT target.** Cover every finite-seed adaptive tester, not merely walk-based testers:

\[
r=\Omega\!\left(\log N+Q\log(t+1)\right)
\quad\text{when }q=O(Q).
\]

The online component is already substantially occupied: Kelman–Linder–Raskhodnikova’s published seed-elimination reduction converts an \(r\)-bit online tester into a standard tester with roughly \(r/\log(t+1)\) queries. Applied here, it gives the \(Q\log(t+1)\) obstruction. Separately, the all-zero execution paths visit at most \(q2^r\) addresses; insufficient coverage lets a defect set avoid every path. These arguments must be transferred with their exact error and timing conventions. [Primary theorem, Lemma 4.3](https://drops.dagstuhl.de/storage/00lipics/lipics-vol325-itcs2025/LIPIcs.ITCS.2025.65/LIPIcs.ITCS.2025.65.pdf).

**Kill test.** First print a fixed finite sampler and its constants. For \(N=8,16\), \(t=1\), and selected defect densities, enumerate seeds and defect sets and solve the adversary’s decision tree using exact probabilities. Return `SAMPLER-KILL` for any failure probability above \(1/3\). A bounded search timeout is `VOID`, not survival. Separately, translate the proposed improvement against the 2025 construction: an existing matching upper is `DELTA-KILL`.

**Novelty/usefulness attack.** “Adaptive access consumes randomness” is already known. The surviving possibility is an efficiently generated, matching resource law for a useful audit. Even success could be an algorithmic contribution inside an established model; it would not automatically establish a new paradigm.

**2. Recallable input mutation — second priority, greater construction risk**

**Useful problem.** Produce a sorted stream of \(n\) finite integers using little private workspace, while guaranteeing that an owner can interrupt and recover the original array promptly. The intended use is a library subroutine borrowing an owner’s input storage without making a full copy.

**Finite founding primitive.** A finite word RAM has \(w=\Theta(\log n)\)-bit words and ordinary single-word read, write and Boolean instructions. After any instruction, the owner may issue `RECALL`. The algorithm must restore the original input within \(D\) further instructions and terminate the borrowing session. Without recall, it must emit the correct sorted stream and restore the input at completion.

One instruction costs one word operation; its address and bit implementation are additionally charged. Private storage is \(S\) bits, including registers, stacks, undo information and continuation state. Output is write-only. There is no atomic rollback instruction.

**Precise hypothesis — SPECULATION.** For integers from a polynomial-size universe, characterize sorting by the joint resource law

\[
T=\widetilde{\Theta}\!\left(
n+\frac{n^2}{S+Dw}
\right),
\]

in an explicitly registered sublinear-storage regime. Tildes permit only specified logarithmic factors.

A concrete first target is \(S=O(\log^2 n)\), recall latency \(D=\widetilde O(\sqrt n)\), and total work \(T=\widetilde O(n^{3/2})\). This is a construction target, not an existing algorithm.

**Fair incumbent.** Every competitor receives the same mutation permission, recall requests, workspace, integer operations and output interface. Compare with read-only sorting, ordinary undo logs, and existing restore-model algorithms adapted to meet the same recall bound. An endpoint-restoring algorithm that blocks the owner for its entire run is not yet an admitted low-\(D\) competitor.

**Charged leak audit.** Do not attach original-position tags unless their bits and construction are charged. Original values cannot be stored in an external undo service. Neither output nor program code can become a readable scratch tape. Every compression must work on the stated worst-case inputs; a favourable input distribution is not free storage.

**Nontrivial CAN route — CONJECTURE.** Use reversible integer partitions and recoverable bit packing, keeping only a bounded frontier of temporarily transformed input regions. The decisive obligations are:

- restore every partially completed partition from its actual finite state;
- bound the outstanding restoration work by \(D\);
- preserve global sorted-output progress without retaining a private iterator for every region.

The existing restore model establishes that temporary mutation can support efficient sorting while recovering the original input. Its integer algorithms manipulate recoverable bits; indivisible-element sorting retains the familiar lower bound. These are positive controls, not this hypothesis’s delta. [Chan–Munro–Raman](https://tmc.web.engr.illinois.edu/restore_soda.pdf).

**Universal CANNOT target — CONJECTURE.** A configuration recallable in \(D\) single-word instructions can differ from the original array in at most \(D\) words. Simulate its mutable overlay using

\[
S+O(D(w+\log n))
\]

private bits over read-only input, then transfer a sorting time–space lower bound. Dictionary lookup and simulation work must be included; this currently supports a lower-bound route with logarithmic overhead, not an exact constant-factor theorem.

The quantifier covers all admitted programs, including compressed or nonlinear encodings.

**Kill test.** Instrument a concrete partition/restoration kernel on \(n=4,6\), testing permutations and duplicate-heavy arrays. Interrupt after every instruction. `RESTORE-KILL` fires for incorrect restoration, excess latency or uncharged state. Two different original arrays reaching the same *accessible restoration configuration* are a decisive collision witness. Next attack the proposed construction’s frontier bookkeeping: needing \(\Omega(n)\) outstanding transformed words kills the proposed sublinear-\(D\) route.

**Novelty/usefulness attack.** “You borrowed writable memory” is the strongest dismissal. The proposed delta must concern restoration on arbitrary recall, rather than rediscovering endpoint restoration or ordinary workspace tradeoffs. This is a fully deterministic CS hypothesis with no source-calibration dependency, but its fast upper is presently the least developed construction here.

**3. Erasure-observer access — third priority; presently an audit byproduct**

**Useful problem.** Cleanly compute a public summary while protecting distinctions within each output fibre from an observer restricted to erased-bit taps. This can support a precise disposal-channel privacy audit. It supplies no guarantee against arbitrary internal-wire observation.

**Finite founding primitive.** Preserve the candidate’s fixed-schedule NOT/CNOT/Toffoli/RESET machine. Each gate costs one elementary operation, with addresses, width and compilation charged. Independent uniform \(r\)-bit \(R\) is supplied and returned privately to its owner; the observer receives public \(f(X)\), the program and at most two chosen pre-RESET target bits.

Keep the existing cap \(\theta=h_2(2^{-r})\) fixed. The new dial is observer access. [Actual candidate](<LOCAL_PROJECTS>/new-computation-model/research/ERASURE_OBSERVER_CANDIDATE.md:8).

**Exact hypothesis — CONJECTURE, separate gate pending.**

\[
\text{Perfect two-probe privacy is feasible}
\iff
\max_y |f^{-1}(y)|\le2^r,
\]

with separately charged compilation and the injective exception.

**Fair incumbent.** Admit arbitrary finite masked circuits with the same private-key supply, observer access, cleanup and cap. Charge fresh randomness and encoding interfaces in any probing-security construction. Neither public \(R\) nor all-wire probing describes this contract.

**Charged leak audit.** Independence is essential. Returning \(R\) pointwise does not restore independence from future workloads or erase the observer’s knowledge. Privacy needs either a lifetime exposure bound or an explicit fresh-key contract across invocations. Later disclosure of \(R\) defeats the stated secrecy boundary. Fibre ranking and inversion must be computed, not supplied as advice.

**CAN route.** The candidate’s finite fibre-rank encoder followed by one-chunk masking supplies a proposed whole-transcript-private upper when the key-label condition holds. Its generic inverse can be exponentially expensive. The existing one-probe compiler and cap-one short-key pairwise mask remain important controls; the cap-one construction is not a same-cap advantage.

**Universal CANNOT route.** At the atomic rare cap, each nonconstant tapped event selects one key label for each fixed input. Pairwise privacy fixes the equality partition of those labels, hence the distribution of the entire erased transcript. Reversing the computation from transcript, public output and private key then supplies the fibre/key counting obstruction at every finite depth and width.

**Primary-prior assessment.** The resulting key-count bound is Shannon’s perfect-secrecy obstruction applied to a reconstructible transcript. Bounded probing and circuit masking are occupied by ISW. The potentially distinct lemma is the **rare-support collapse from two probes to full-transcript secrecy**, together with charged compilation—not either foundation. [Shannon, perfect secrecy](https://dkrizanc.web.wesleyan.edu/courses/351/papers/Communication_Theory_of_Secrecy_Systems_Shannon_1949.pdf), [ISW](https://people.eecs.berkeley.edu/~daw/papers/privcirc-crypto03.pdf).

**Kill test.** Enumerate small one-label-per-column incidence arrays and search for equal pair distributions but unequal transcript distributions. Include constant columns, complements and output-fibre conditioning. A witness is `RIGIDITY-KILL`. Separately verify finite rank compilation and the cap-one mask; reject free inverse lookup. A failed bounded search is not a proof.

**Candid verdict — HEURISTIC.** The current object is an audit byproduct. For AND, the proposed two-probe guarantee requires \(r\ge n\), and the available rare-cap upper is expensive. A restricted observer alone does not supply a useful strict advantage. The fresh gate could establish a valid lemma without changing that assessment.

**Strongest fresh positive angle — HEURISTIC.** Hypothesis 1 has the clearest constructive foothold: a finite, uniformly generated audit schedule with an explicit adaptive-erasure calculation and a universal published lower-bound mechanism. Its meaningful next question is whether generation and seed costs can match that lower without weakened access or extra queries.

**Negative lessons from the completed thread — record-backed synthesis.**

- A RESET implementation does not supply the receiver, reversible controls, source, bath and controller needed for an operational circuit claim. Noise changes lower-bound quantifiers as well as correctness.
- Returned resources retain history: pointwise return establishes neither independence nor safe reuse.
- Every access advantage needs an incumbent granted that same access. Ordinary masking, restore algorithms and adaptive testing must enter before a novelty claim.
- Failed constructions are not universal CANNOT results. The admitted 116-circuit sweep supports its finite contracts; its size cannot establish novelty or useful-world significance.
- Further meter expansion would repeat the unresolved bridge. These hypotheses instead make intervention, mutation permission or observation boundaries explicit.

**Narrative attack — HEURISTIC.** The strongest triviality attacks are “a seeded sampler,” “borrowed writable memory,” and “Shannon applied to sparse transcripts.” Each proposal needs a useful positive construction and a universal limit that survives those translations. None currently earns a fundamental-novelty or mission-completion claim.

This ranking is material for the actual advisor/architect dialogue. No new direction was settled, no artifacts were written, and no proof gate or computation was executed.