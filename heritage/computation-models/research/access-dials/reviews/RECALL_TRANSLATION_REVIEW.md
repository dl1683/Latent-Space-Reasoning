**DESIGN: PASS. LOWER: PASS. CHARGED UPPER: PASS.** Independent attack found no remaining defect at the repaired scope.

The admitted deterministic optimum is
\[
Q_{\mathrm{recall}}^{*}=m\max(C-S,0).
\]

**Design.** For every fixed \(R\), the **complete nonpublic checkpoint configuration** has at most \(2^S\) possibilities. Input-dependent PCs, addresses, lengths, allocations and continuation metadata count. The upper’s fixed synchronous schedule depends only on \(C,m,S\); program generation, storage, controller, addresses and execution remain separately charged. Its direct two-bit XOR primitive needs no extra private accumulator. No hidden-information or infinity leak survived. [Definitions](<LOCAL_PROJECTS>/new-computation-model/research/RECALL_TRANSLATION_CANDIDATE.md:18)

**Lower.** Fixing \(R\) leaves every \(X\) possible because \(B=R\oplus F(X)\). For each advice value, retaining valid runs, adding dummy branches and bypassing repeated queries produces a bounded tree before applying Kraft. With \(k_0=C-\lfloor Q/m\rfloor\), one advice value covers at most \(2^{N-k_0}\) correctly restored inputs; all advice values cover at most \(2^{N+S-k_0}\). A violating \(Q<m(C-S)\) makes \(k_0>S\), contradicting coverage of all inputs. Deterministic bank rewrites and unlimited fresh postcheckpoint workspace cannot supply unqueried information. [Proof](<LOCAL_PROJECTS>/new-computation-model/research/RECALL_TRANSLATION_CANDIDATE.md:45)

**Charged upper.** With \(j=\min(S,C)\), the construction restores every arbitrary bank and clears its private cells:

| Phase | Input-bit reads | Data-bit updates |
|---|---:|---:|
| Initialize private cells | \(0\) | \(S\) |
| Prepare journals | \(jm\) | \(jm\) |
| Produce checkpoint | \(N\) | \(N\) |
| Recall and cleanup | \(q=m(C-j)\) | \(q+2j\) |

These exact counts measure the displayed operations, **not total bit work or latency**. Across these phases, input reads total \(2N\): journaling shifts reads into preparation. [Construction and charges](<LOCAL_PROJECTS>/new-computation-model/research/RECALL_TRANSLATION_CANDIDATE.md:62)

**EMPIRICAL:** Fresh bounded CPU probes checked all 32,768 binary advice partitions for \(C=m=2,S=1\), obtaining optimum \(2\), and 4,314 primitive executions with \(N\le6\). Restoration, journal clearing and phase counts passed, including \(S=0\) and \(S>C\).

Admission excludes randomness and the exact-\(S\) upper under an all-physical-controller RAM convention. This is a calibration lemma: no novelty, sorting/matrix-powering CAN or mission completion follows. No memory evidence, edits, agents, GPU or FULL run were used.