**LOWER: PASS. Charged UPPER: PASS.** No counterexample or mathematical repair emerged for the [generic candidate](<LOCAL_PROJECTS>/new-computation-model/research/GENERAL_MASKED_CANDIDATE.md:9), under independent uniform returned \(R\), unrestricted unknown \(p_X\), fixed schedules, full cleanup, and local RESET cost \(H(B)\).

**LOWER.** Set \(\beta=2^{-r}\). At any fixed RESET, its target probability is affine in the original data prior, even after previous coalescence. The full prior simplex therefore produces an interval. If \(K<h_2(\beta)\le1\), that interval must lie entirely in one entropy tail. Hence one target value is globally rare, with probability at most \(a<\beta\).

For every admitted pure prior \(\delta_x\), this rare probability is \(q_x\beta\), where \(q_x\) counts original \(R\) labels and is an integer. Thus \(q_x=0\) for every \(x\). The target is constant over the entire reachable support, so RESET is injective on that support. Consequently every gate preserves distinguishability of the original labels. Noninjective \(f\) requires reducing \(2^{n+r}\) initial states to \(2^r|\operatorname{im}f|\), a contradiction. This also covers \(r=1\).

The injective exception survives: compute \(Y=f(X)\), reversibly evaluate a finite inverse extension on \(Y\), XOR its result into \(X\), then uncompute the inverse workspace. No RESET is needed; inverse construction and work remain charged.

**UPPER.** The exact [incoming macro](<LOCAL_PROJECTS>/new-computation-model/experiments/meter_controls.py:26) maps both adjacent patterns \(a,b\) to \(b\), independently within every fixed combination of untouched registers. If only \(a\) exists, the final predicate-controlled advance moves it to \(b\). If only \(b\) exists, the macro returns it to \(b\) with clean scratch. Empty classes therefore cause no failure. Distinct output fibres cannot merge because \(Y\) remains untouched.

Before each RESET, the incoming pattern is unvisited. For each original \(x\), precisely one original mask satisfies
\[
R_{\mathrm{low}}=x_{\mathrm{chunk}}\oplus b,
\qquad
R_{\mathrm{high}}=\text{selected context}.
\]
Earlier chunks changed neither the current chunk nor returned \(R\). Earlier merges in this chunk moved visited patterns to the previous representative. Thus **\(q_x=1\) for every original \(x\)**, including after coalescence, and
\[
\Pr(B=1)=\sum_xp_x\,2^{-r}=2^{-r}.
\]
No output predicate or special output handling is required.

Uncomputing the final prefix before clearing the all-ones representative preserves scratch correctness. Gray suffix reuse gives \(O(2^d)\) work per context; the \(2^{r-d}\) contexts cost \(O(2^r)\) per chunk. Therefore the charged elementary-gate bound is
\[
G=O\!\left(C_f+n+r+2^r\max\{1,n/r\}\right).
\]
This includes \(C_f\). Construction, address encoding, program generation, and source preparation remain separately charged; this is not an end-to-end bit-work bound.

Together, these establish the **exact universal local feasibility floor \(h_2(2^{-r})\) for noninjective \(f\)**. The matching critical-work \(\Theta\) statement remains AND-specific; it does not extend automatically to every noninjective function.

**EMPIRICAL reviewer checks.** Independent in-memory emission matched the extracted source helpers and incoming macro. All 36 bounded circuits passed across constant, parity, majority, nonlinear injective, multioutput, and AND functions, including consecutive chunks, short final chunks, and \(r>n\):

- 3,914,444 original-label/gate transitions and 498 RESET checks.
- Exact \(q_x=1\), macro-boundary prefixes, returned \(R\), correct \(Y\), and complete cleanup.
- A separate nonlinear injective construction cleaned up with zero RESETs.
- Removing the representative advance broke cleanup; removing masks broke exact flag counts.
- Another 41,731 bounded count-vector checks supported the lower interval argument.

**Physical implication: conditional inference only.** A local reset design whose relevant input law is Bernoulli\((\beta)\) has zero workload mismatch because that marginal is identical for every \(p_X\). This does not fix workload-dependent correlations with other registers or establish finite-time physical realization. Preparation, precision, reversible-gate costs, rare-event energy, and primary novelty remain open.

The surviving result is a generic compiler theorem about workload-independent erasure marginals. Its useful physical advantage and computation-model novelty remain unestablished.

Read-only throughout: no edits, agents, GPU, FULL run, or invocation of the separate 116-circuit experiment.