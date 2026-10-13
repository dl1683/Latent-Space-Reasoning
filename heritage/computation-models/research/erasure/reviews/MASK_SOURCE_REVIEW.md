**LOWER: PASS. Charged UPPER: PASS. Critical AND work: PASS. Endpoint: PASS.** No counterexample or substantive proof repair emerged for [MASK_SOURCE_CANDIDATE.md](<LOCAL_PROJECTS>/new-computation-model/research/MASK_SOURCE_CANDIDATE.md:7).

Acceptance covers the stated unknown **independent product-source class**, fixed finite schedules, pointwise correctness and cleanup, and local RESET entropy. Work bounds concern elementary gates; generation, encoding, precision and source preparation remain separately charged.

**LOWER — PASS.** For each fixed gate, the target bit is a deterministic function of original \((x,z)\), even after coalescence. Interpolating \(p_X\) and \(Q\) separately preserves independence and gives a continuous, generally quadratic probability path. Rational path parameters produce admitted rational laws. If opposite entropy tails were attained, an open segment of this path would violate the cap and contain rational parameters.

Strictly, connectedness applies to the **real product polytope**, not its rational points. The candidate’s rational-path argument supplies the needed correction already.

For \(K<h_2(\tau)\), let \(a=h_2^{-1}(K)<\tau\). One target value is globally rare, with probability at most \(a\). Any original label taking that value permits the rational witness
\[
p_X=\delta_x,\qquad Q(z)=\tau,\qquad
Q(z')=\frac{1-\tau}{S-1}\quad(z'\ne z).
\]
The remaining atoms satisfy the cap precisely because \(S\tau\ge1\). This contradicts rarity. Consequently every RESET has a constant target on full original support and preserves distinguishability. A clean noninjective output cannot result. This includes \(\tau=\beta\), zero-mass laws, and \(\tau=1/2\) with \(K<1\). The separately charged reversible-inverse exception for injective \(f\) survives.

**Charged UPPER — PASS.** The generic compiler’s incoming pattern is unvisited. For each original \(x\), its selected high context and masked incoming chunk identify exactly one original mask label. Earlier chunk cleanup leaves the current chunk and returned mask untouched; earlier merges within the chunk move visited patterns to the representative.

The [incoming macro](<LOCAL_PROJECTS>/new-computation-model/experiments/meter_controls.py:26) also handles missing predecessors: its final advance moves the accumulated representative forward even when the incoming class is empty. Untouched output registers preserve distinct output fibres.

Thus under an arbitrary admitted law,
\[
\Pr(\text{flag}=1)=\sum_xp_X(x)Q(z_g(x))\le\tau.
\]
These marginals need **not** be uniform or workload independent. A pure workload and a mask law saturating its marked column attain \(\tau\). The finite compiler therefore attains the exact noninjective floor \(h_2(\tau)\), with
\[
G=O\!\left(C_f+n+r+S\max\{1,n/r\}\right).
\]
Gray suffix reuse, predicate uncomputation, short final chunks and \(r>n\) preserve this charged gate bound.

**Critical AND work — PASS**, uniformly for \(n\ge3,r\ge2,\beta\le\tau<1/2\). Global tail orientation forces rare probability \(\le\tau\). Two rare mask labels for one original \(x\) admit
\[
Q(z_1)=Q(z_2)=\tau,\qquad
Q(z)=\frac{1-2\tau}{S-2}
\]
elsewhere. All masses are rational, nonnegative and capped. Hence each original \(x\) has at most one rare mask label.

The important attack here is that **the two-label witness can itself land in the opposite allowed entropy tail**. Its entropy alone does not establish a violation. The proof survives because global orientation has already been established by rational interpolation.

Under the admitted uniform joint law, every RESET then has
\[
J=H(B\mid T)\le H(B)\le h_2(\beta),
\]
so entropy telescoping gives
\[
G\ge\frac{n-h_2(2^{-n})}{h_2(\beta)}.
\]
Each RESET collision has a distinct rare predecessor containing a distinct rare original label. Prior coalescence therefore does not invalidate the bound of \(2^n\) collisions per RESET. The required class decline gives
\[
G\ge S(1-2^{1-n}).
\]
Distinct output registers require targeting all original input registers, yielding \(G\ge n+r\). Combining these bounds with \(h_2(2^{-r})=\Theta((r+1)2^{-r})\) proves
\[
\boxed{G^*=\Theta\!\left(n+r+2^r\max\{1,n/r\}\right)}
\]
with constants independent of \(\tau,n,r\).

**Endpoint — PASS.** At \(\tau=1/2\), \(K=1\) removes the tail restriction. Transferring \(R\), computing/copying/uncomputing AND, and resetting \(X\) costs \(3n+2r-1\) elementary gates. Together with input targeting, this proves \(\Theta(n+r)\) under the referenced distinct-output-register contract. The feasibility floor remains one; the critical work changes.

**EMPIRICAL reviewer probes:** 20 bounded in-memory generic circuits passed 1,109,216 original-label gate transitions, 200 RESET checks and 800 exact product-law checks. Cases included constant, AND, parity, multioutput and injective functions, consecutive chunks, short chunks and \(r>n\). Biased marginals were observed.

Read-only throughout: no edits, agents, GPU or FULL execution. This admits the scoped mathematical statements only; novelty, physics, source authentication and useful-world standing remain open.