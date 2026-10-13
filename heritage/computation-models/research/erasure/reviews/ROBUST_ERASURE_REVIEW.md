**CODE: PASS. LOWER PROOF: PASS. UPPER PROOF: PASS**, at the stated deterministic, finite, clean-output scope. No mandatory repair found. The threshold characterizes robust finite realizability; it is more than a singleton-construction upper bound.

This was a read-only CPU review. I read AGENTS.md, the required dual-loop skill, and the required status surfaces. I did **not** run the registered FULL experiment or modify files.

| Gate | Finding |
|---|---|
| CODE | Registered ranges, exact arithmetic, fixed-schedule minimax, witnesses, and failure handling are consistent. |
| LOWER PROOF | Every fixed clean circuit has an admitted prior witnessing a step loss of at least \(R(q,\eta)\). |
| UPPER PROOF | One fixed, charged gate schedule attains loss at most \(R(q,\eta)\) simultaneously for every admitted prior. |

The [candidate proof](/<LOCAL_PROJECTS>/new-computation-model/research/ROBUST_ERASURE_CANDIDATE.md:21) therefore survives as
\[
\exists C\;\forall p\in\mathcal P_\eta\;\forall t:\Delta H_p(C,t)\le K
\quad\Longleftrightarrow\quad K\ge R(q,\eta).
\]
The general Boolean-machine theorem uses \(M=2^n\).

**Proof findings.** Complete-state equality can only coarsen, and exact clean correctness on every input forces every intermediate block to remain within a final fibre. These partitions include zero-mass inputs; zero probability does not waive functional correctness.

- **Sparse regime, \(q\eta\le1\):** assigning mass \(\eta\) to every atom of the largest fibre is feasible. Outside capacity satisfies \((M-q)\eta\ge1-q\eta\). Convexity at the first complete formation of that fibre gives loss at least \(\eta\phi(q)\).
- **Saturated regime, \(\eta\le1/2,\ q\eta\ge1\):** for the smallest predecessor block of size \(d\), the proposed \(t=\max(\eta,1-(q-d)\eta)\) satisfies \(\eta\le t\le1/2\), \(t\le d\eta\), and \(1-t\le(q-d)\eta\). Thus the allocation exists, and grouping the predecessor blocks gives loss at least \(h(t)\ge h(\eta)\).
- **High-cap regime, \(\eta\ge1/2\):** two same-output inputs with masses \(1/2,1/2\) witness one bit of loss at their first complete-state merger.
- Both boundaries agree. At \(\eta=1/M\), the uniform threshold is recovered; \(q=1\) has a reversible clean realization with zero loss.

The adversarial prior may depend on the **already fixed circuit**. A single prior defeating every circuit is unnecessary.

For the upper bound, the [selective flag-reset compiler](/<LOCAL_PROJECTS>/new-computation-model/research/ATOMIC_ERASURE_PROOF.md:82) merges only the accumulated representative class and the selected singleton. All other gates preserve complete-state entropy. The fixed singleton order obeys the three claimed bounds, and terminal conversion is reversible and clean. Width, generated gates, truth-table storage, and construction work remain charged; this is an exponential compiler.

**Code findings.** In [robust_erasure.py](/<LOCAL_PROJECTS>/new-computation-model/experiments/robust_erasure.py:26), sorting block masks removes only block-list ordering. Atom labels remain intact. Successors enumerate binary merges within target fibres, and target-size enumeration is justified by permutation symmetry of the prior class.

The recurrence computes
\[
\min_{\text{fixed paths}}\max_{\text{edges}}\max_{\text{priors}}\text{edge cost}.
\]
The two maxima commute; neither crosses the minimum. `weighted_erasure.optimum` is not used for robust optimization.

When either merged mass is zero, the ratio is exactly one, including the both-zero case. The positive-only dependency is correctly bypassed. Path witnesses identify maximizing priors for each edge; different edges may have different witnesses. First failure retains the evaluated prefix and failing task and exits unsuccessfully. Normal optimized execution and optimized code objects both fail closed.

**Actual probes**, using Python 3.13.7 and `-B`:

- 294 bounded-vector comparisons against direct Cartesian-product enumeration.
- Independent partition checks; exact replay of all **25 saved smoke rows**.
- **12 additional small tasks** checked by an independent labelled schedule enumerator, including optimum and every path witness, spanning all regimes and both boundaries.
- 1,321 predecessor-capacity checks, plus sparse-regime padding checks.
- Target-order symmetry checks and all seven target shapes for \(M=5,W=4,u=2\); state and edge counts matched independently derived counts.
- Zero-mass handling, `-O`, optimized code objects at levels 1 and 2, and an injected first-failure smoke simulation with filesystem writes mocked out.

Two explicit counterexamples delimit interpretation:

1. **Interchanging quantifiers changes the answer.** On the 31-prior grid \(M=4,W=8,u=3\), with one constant-output fibre:
   \[
   \min_C\max_p 2^{8\,\mathrm{peak}(C,p)}
   =\frac{16777216}{84375},
   \qquad
   \max_p\min_C 2^{8\,\mathrm{peak}(C,p)}
   =\frac{65536}{729}.
   \]
   The submitted DP computes the first quantity correctly.

2. **Extending the registered grid to odd denominators breaks exact agreement.** For \(M=2,W=3,u=2,q=2\), the grid optimum is \(27/4\), whereas the continuous-prior prediction is \(8\). The grid omits the balanced prior. This does not refute the theorem; the registration correctly requires even \(W\).

No counterexample to the stated theorem was found.

**Precision accounting passes.** For \(\eta=u/W\), exponentiating \(W R\) yields the stated exact rational ratio. For rational \(K=s/v\), comparison reduces to integer arithmetic after clearing denominators. Large powers and bit work must be charged. An arbitrary real \(K\) remains a mathematical parameter, not an exact-comparison runtime oracle.

**Full evidence admission requires:**

1. Matching reviewed source hashes before and after execution, recorded CPU environment and assertion mode, and unchanged registration.
2. Exactly **428 cases**: 40, 69, 130, and 189 for \(M=2,3,4,5\), respectively, with every registered denominator, cap, and target partition.
3. Independent verification of vector completeness, exact edge maxima, refinement-preserving paths, maximizing witnesses, and DP minima. A path alone certifies feasibility, not optimality.
4. Every exact optimum equals its prediction. A first mismatch admits **KILL** with its task and witness; scope or arithmetic changes require **VOID**. Interrupted output cannot admit CONFIRM.
5. Continued labeling as **EMPIRICAL exact finite abstraction evidence**. \(M=3,5\) are valid finite supports; this enumeration supplies no gate-level result or all-prior proof.

Reviewed SHA-256 fingerprints:

```text
ROBUST_ERASURE_CANDIDATE.md  68d9ed34ae2a55dc5594a3f05deae2247db94f7425878de0505799c68f989009
robust_erasure.py            c04022347ae4cfb6be1318df3582b41518a1e2c80d6fa74f779afe3ca229e104
weighted_erasure.py          9592d370d94f3b5df5649b92834e45bee9b8aa5e0b3a53176bb0a87c1dccdbf3
robust_erasure_smoke.json    5abd014dc92c27756a77a10eafbb4ec82df25981575e5dd6dd3ed2160ac1446f
EXPERIMENTS.md              988b792124888c5b60037e4477380c1716d5fff73655aebffde95cc5446a81d4
ledger.jsonl                41841afb81b317c4dbab3d35d31445c643b965cea08089ccb3c442f3344cb4f8
ATOMIC_ERASURE_PROOF.md      ffb0f7862fb326f1a92956a9afff9ef5cc923d26202d7f981e5190a2d6f3e1ed
```

The ledger received an unrelated weighted-run append during review; robust registration and smoke records stayed unchanged. The manifest gives the final inspected ledger hash.

**Novelty remains unsettled.** This gate does not establish a strict advance over Bennett, generalized reversibility, VI partition distances, or EBIM. An atom-mass cap supplies neither an efficiency gain nor a physical cooling law. The surviving result is an exact feasibility theorem under a stated prior class, not mission completion.