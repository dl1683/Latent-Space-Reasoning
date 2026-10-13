**PROBE-PROOF-PASS.** The CANNOT theorem
\[
H+1\le L\min(N,2^{d_i}-1)
\]
holds **for every coordinate \(i\)** under the stated deterministic, cold, exact QUERY contract. I found no counterexample within that model.

The [proof](/<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_PROBE_LAW_CANDIDATE.md:41) survives independent reconstruction:

1. Fix \(i\). Canonical entry state, fixed code and fixed current command make each observation address, subsequent control and final answer deterministic functions of the observed bit transcript. Contracting computation between observations gives a reachable-path binary tree of depth at most \(d_i\), with at most \(2^{d_i}-1\) observation nodes. Their cell-position union \(P_i\) therefore satisfies \(|P_i|\le\min(N,2^{d_i}-1)\). Reachable states agreeing on \(P_i\) produce identical answers.
2. Initial inputs include both values of \(x_i\). If every initialized state projected to zero on \(P_i\), all initial answers would coincide. Thus some initialization ends with a one in \(P_i\), consuming at least one change there from the supplied unworn zeros.
3. From that initialization, request \(H\) consecutive FLIP(i) commands. Every consecutive pair of endpoints has opposite decoded bit \(i\), hence differs somewhere in \(P_i\). Each update consumes at least one additional change there. Consequently \(H+1\le L|P_i|\), proving the claim. The initializer may differ between coordinates; the theorem needs no common worst-case input.

Coverage of the requested attacks:

- **Variable time and off-promise behavior:** deterministic internal delays add no information. Promised queries terminate. Unreachable branches can become arbitrary leaves without executing an off-promise query—including QUERY on the physical all-zero state.
- **Reachable-only pruning:** identical transcript prefixes determine identical next actions. A reachable terminating prefix cannot also require another observation on another reachable execution. Thus pruning preserves every promised answer.
- **Retained external state and transcript collisions:** excluded by the [model](/<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_BOUNDARY_MODEL_CANDIDATE.md:28). Raw input disappears after initialization; private state and PC reset; outputs become unreadable; clocks, invocation counters, caches and additional persistent storage are forbidden. For fixed QUERY(i), current opcode/index ports supply identical input. Distinct histories cannot use an external retained value to diverge after an identical transcript.
- **Observation accounting:** direct-copy, BOOL operands and direct bit outputs count whenever they acquire a worn-cell value. Repeated acquisitions count again. Multiple operand acquisitions can be expanded into successive binary observations; counting only READ mnemonics would invalidate the argument.
- **Nonlinear encoding and multiwrite updates:** no linearity, preferred encoding, \(I(0)=0\), or single-write assumption is used. Intermediate and undone changes increase consumption; they cannot refund endpoint changes.
- **\(H=0\), \(N=1\), and zero observations:** initialization alone supplies the \(+1\). Exact cold queries require \(d_i\ge1\). When \(N=1\), the bound gives \(H\le L-1\), regardless of repeated reads.
- **All FLIP sequences and post-\(H\) queries:** universality includes the repeated-coordinate sequence used separately for each \(i\). Correct QUERY after flip \(H\) supplies the final endpoint distinction. Arbitrarily interleaved read-only queries preserve cells and canonical entry.
- **Failure/saturation oracle:** exhausted opposite writes are invalid, and promised histories must avoid them. Hidden wear supplies no observable branching or timing channel.

One algebraic clarification: the complete numerical reformulation retains **both** \(H+1\le LN\) and
\[
d_i\ge\left\lceil\log_2\!\left((H+1)/L+1\right)\right\rceil.
\]
The logarithmic line alone omits the global capacity condition; this does not undermine the proof.

I hand-checked the Johnson cycle mathematically: its stated query parity matches cycle-step parity, updates change positions \(0,\ldots,N-1\) cyclically, and initialization plus \(NL-1\) flips respects each cell’s budget. **The compiler remains HYPOTHESIS; no CAN admission or executable proof follows.** The proposed probe bound matches the lower bound for powers of two only; no all-\(N\) tightness is admitted.

This establishes a scoped lifetime-versus-observation restriction. It establishes no novelty, physical advantage, or completed model pair, and is independent of the selected newmodel. Read-only throughout; no subagents, solver, experiments or source changes.