**LOWER: PASS. Charged UPPER: PASS.** The [candidate](<LOCAL_PROJECTS>/new-computation-model/research/CORRELATION_LOCAL_CANDIDATE.md:8) survives independent attack:
\[
G_C^*(n,r)=\Theta(n2^r+r),\qquad n\ge3,\ r\ge4.
\]
Acceptance covers elementary gate work under the fixed-schedule, clean-AND contract, local RESET cap \(L\le h_2(2^{-r})\), capped X marginal, and arbitrary correlation with pointwise-returned uniform R. No mathematical repair emerged.

**Lower reasoning.** Write \(N=2^n\), \(M=2^r\), \(\beta=1/M\), and \(\eta=1/8\). At any fixed RESET, its target bit is a deterministic function of the original \((X,R)\), including after previous coalescence. Its probability is therefore linear in the source law. The source polytope is convex, and the permitted entropy tails
\[
[0,\beta]\ \cup\ [1-\beta,1]
\]
are separated. Universal admissibility forces the attainable interval wholly into one tail. Thus this gate has a globally chosen rare value whose probability is at most \(\beta\). Rational mixtures alone suffice for this argument.

Suppose two distinct original R columns contain rare labels. Assign each selected cell mass \(\beta\). Even when both cells use the same X, their row mass satisfies
\[
2\beta\le\eta,
\]
with equality at \(r=4\). Completion is finite and dyadic: measure masses in units of \(\beta\). Each row has capacity \(M/8\ge2\), and total row capacity \(NM/8\ge M\). After pinning the two columns, assign every remaining column’s single unit to a row with available capacity. This preserves every R marginal exactly and respects every X cap. The resulting rare probability is at least \(2\beta\), contradicting its globally fixed bound. **At most one original R column can contain rare labels at this RESET.** The active column may change between gates.

Pointwise return defeats physical recoding: two original labels with different R cannot share a complete state at any stage, because deterministic continuation could not subsequently return both values. On every inactive R slice, the RESET target is constant, so resetting it is injective within that slice. Under the admitted uniform joint source, only one slice can lose entropy:
\[
J_g
=\sum_z\beta\bigl[H(S_{g-1}\mid R=z)-H(S_g\mid R=z)\bigr]
\le\beta.
\]
The active slice loses at most one bit through a binary RESET. This establishes the stronger joint-loss restriction despite the larger specified local cap \(h_2(\beta)\).

Initial entropy is \(n+r\); clean final entropy is \(r+h_2(2^{-n})\). Consequently,
\[
\#\mathrm{RESETs}\ge
\frac{n-h_2(2^{-n})}{\beta}
=\Omega(n2^r).
\]
Since \(r=O(n2^r)\) throughout the stated range, this proves the claimed lower bound. Extra scratch, depth, and recoding do not change the argument.

**Upper reasoning.** I inspected [fast_circuit](<LOCAL_PROJECTS>/new-computation-model/experiments/context_threshold.py:49) and its [selected RESET macro](<LOCAL_PROJECTS>/new-computation-model/experiments/succinct_erasure.py:93). AND scratch is uncomputed before cleanup. Context predicates depend only on returned R and remain valid while data are cleared. Each macro executes three explicit gates and restores its flag to zero.

At every RESET, flag one is supported inside one original R context. Therefore its probability is at most \(\beta\) under **every** admitted joint law, giving \(L\le h_2(\beta)\). The upper does not need the X atom cap.

There are \(nM\) selected RESET macros. Charged Gray suffix maintenance uses
\[
\sum_{\text{edges}}\ell=2^{r+1}-r-2,
\]
so context maintenance costs \(O(M+r)\). Including AND computation and R transfer gives \(O(nM+n+r)\) gates and width \(2n+3r+4\). Generation, serialization, and source preparation remain separately charged finite resources; this is not an end-to-end bit-work theorem.

**EMPIRICAL reviewer probes passed:** 1,664 exact dyadic transport witnesses, including 24 same-X tight-boundary witnesses; eight independently emitted circuits matched the source gate for gate; 2,557,824 label–gate transitions and 864 RESET support checks verified complete cleanup, returned R, and the uniform-source entropy accounting.

The factor-\(\Theta(r)\) comparison survives when \(n\ge r\) and r grows. Its useful meaning is precise: the masked cleanup gain consumes independence, beyond a uniform returned marginal. Physical calibration, priority, and mission completion remain **OPEN**.

Read-only throughout: no edits, agents, GPU, FULL invocation, or alteration of the running registered experiment.