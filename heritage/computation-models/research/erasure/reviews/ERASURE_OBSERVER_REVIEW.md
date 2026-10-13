**DESIGN: PASS. LOWER: PASS. Charged UPPER: PASS. Cap-one control: PASS.** No mandatory mathematical repair emerged for the [observer candidate](<LOCAL_PROJECTS>/new-computation-model/research/ERASURE_OBSERVER_CANDIDATE.md:8).

Acceptance is restricted to one finite, fixed-schedule execution, independent uniform \(r\)-bit \(R\) with \(r\ge2\), unrestricted \(p_X\), exact cleanup, and an observer seeing only public \(Y\), the program, and selected pre-RESET target bits.

**DESIGN — private-owner output and accounting.** Returning \(R\) to its private owner is consistent with excluding it from the observer’s view. Indeed, \((Y,R,B_{\mathrm{whole}})\) reconstructs \(X\), while \(B_{\mathrm{whole}}\) alone can be private given \(Y\). The access distinction is essential and explicitly stated.

The model hides no infinite resource: keys, registers, schedules, tables and inverse circuits are finite. Exact uniform independent key preparation is a supplied, charged assumption. The cap \(h_2(2^{-r})\) has a finite symbolic specification from \(r\); gates never consult an exact-real oracle. Function evaluation, ranking, inverse construction, program generation and address encoding remain charged. The generic existence result supplies no efficient inverse guarantee.

**LOWER — rigidity and arbitrary finite depth.** At a fixed RESET, its target probability is affine in the original \(p_X\), including after previous coalescence. All priors therefore generate an interval. Since
\[
h_2(\beta)<1,\qquad \beta=2^{-r},
\]
the cap forces that interval into one entropy tail. One globally fixed target value consequently has probability at most \(\beta\). Under each pure input \(x\), its probability is an integer multiple of \(\beta\), so its support contains at most one original key label.

Within a public output fibre, one-probe privacy fixes which probes are constant and which select a singleton key label. For active probes, pairwise intersections have probability either zero or \(\beta\). Two-probe privacy therefore fixes their equality partition. Each partition block produces its indicator transcript with probability \(\beta\); unused key labels produce zero. This fixes the whole transcript law. Fibre-dependent constant probes and globally fixed complements introduce no hole.

Adaptive two-probe strategies also survive the attack: each branch probability comes from a fixed singleton or pair law, including the chosen probe indices and stopping decision.

Saving every RESET target makes the computation reversible. For any positive-probability transcript \(b\), privacy gives the same positive integer key count \(a_b\) for every input in a fibre of size \(q\). Reconstruction permits at most one such input per fixed key:
\[
q\,a_b\le2^r,\qquad\text{hence }q\le2^r.
\]
This covers arbitrary finite depth and width. For AND, \(q=2^n-1\) forces \(r\ge n\). [Lower argument](<LOCAL_PROJECTS>/new-computation-model/research/ERASURE_OBSERVER_CANDIDATE.md:31).

**Charged UPPER — exact feasibility.** Ranking within each fibre makes \((f(x),\operatorname{rank}(x))\) injective. A finite inverse extension reversibly clears \(X\), with its construction and execution charged.

For \(k=\lceil\log_2q\rceil\le r\), the single-chunk cleanup’s transcript has at most one set flag. Each noninitial masked pattern in each high-key context produces a distinct singleton transcript with probability \(2^{-r}\); the zero transcript has probability \(2^{-k}\). These probabilities are rank-independent. Every RESET marginal is exactly \(\beta\), scratch is cleaned, and \(R\) returns pointwise.

Thus, at the stated cap,
\[
\boxed{\text{finite clean perfect two-probe realization exists}
\iff \max_y|f^{-1}(y)|\le2^r.}
\]
Injective functions retain the charged zero-RESET exception. Generic work includes the rank/output and inverse circuits, plus \(O(k+r+2^r)\) cleanup. The stated AND \(\Theta(n+r+2^r\max\{1,n/r\})\) bound concerns elementary gates; encoding, generation and preparation remain separate charges. [Upper construction](<LOCAL_PROJECTS>/new-computation-model/research/ERASURE_OBSERVER_CANDIDATE.md:69).

**Cap-one control — PASS.** Distinct nonzero binary mask vectors give uniform tapped pairs. Compute/copy/uncompute AND, key transfer, mask CNOTs and data RESETs yield exactly
\[
G=3n+2r-1+\sum_i\operatorname{weight}(a_i),\qquad w=2n+2r.
\]
Each RESET has entropy one. Full privacy fails in the controls, correctly distinguishing pair privacy from rare-support rigidity.

**EMPIRICAL bounded checks:** 7,186 singleton-support vectors; 22 rank/inverse constructions covering non-power-of-two fibres and injective cases; six generic AND controls; and six cap-one elementary circuits for \(n=3,\ldots,8\). All expected properties passed. Multi-chunk AND leaked through pairs when \(r<n\); single-chunk cases had private full transcripts.

Fresh read-only review, without private-memory evidence, edits, agents, GPU or FULL execution. These are scoped design and theorem verdicts; no novelty or mission admission.