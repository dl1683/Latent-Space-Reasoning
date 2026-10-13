| Gate | Verdict |
|---|---|
| **CODE GATE** | **PASS** for registered weighted-partition falsification |
| **LOWER PROOF GATE** | **PASS** under the stated finite deterministic clean-state contract |
| **UPPER PROOF GATE** | **PASS** for the specified descending schedule and finite-basis construction |

Read-only, CPU only. I read AGENTS.md and the complete dual-loop skill. No files were edited and no registered full experiment ran.

The [lower proof](</<LOCAL_PROJECTS>/new-computation-model/research/WEIGHTED_ERASURE_CANDIDATE.md:44>) survives all requested attacks. For **every** heaviest prefix, its first coalescence has at least two predecessor classes containing selected atoms. Each selected class mass is at least \(p_{y,j}\). Removing additional atoms—and any merging classes containing no selected atoms—cannot increase merger loss. Combining the selected predecessor parts into two groups reduces that loss; both group masses remain at least \(p_{y,j}\). Binary-entropy concavity therefore gives the claimed endpoint bound. Other simultaneous mergers contribute nonnegative loss. Unequal final fibres require no renormalization, and arbitrary preceding depth cannot split already equal complete states.

The upper proof also survives. Choosing the heaviest encoding as representative and absorbing remaining singletons in descending mass order makes each flag RESET lose exactly the corresponding prefix quantity. Predicate computation, controlled input changes, and terminal conversion are reversible; scratch is cleared. The explicit table, rational comparisons, circuit generation, and gates are charged. **The existing lexicographic compiler does not implement this weighted schedule.** The proof specifies a sufficient construction; repository weighted gate-compiler validation remains outstanding.

The [code](</<LOCAL_PROJECTS>/new-computation-model/experiments/weighted_erasure.py:17>) correctly uses
\[
R(a,b)=2^{W L(a/W,b/W)}
\]
with one fixed denominator \(W\) throughout each DP. The minimax recurrence is acyclic and exact. Sorted mass states form a valid quotient for unrestricted within-fibre merging. Duplicate successors have the same removed mass pair and hence the same edge cost. Partition recursion is complete and duplicate-free. Frozen ranges, optimized-mode rejection, resolved-path confinement, and first-mismatch serialization passed inspection and probes.

**Actual cheap probes — EMPIRICAL:**

- Reproduced all **112 weighted smoke rows**, exact ratios and **112 memoized states**, with matching source hash.
- Independently verified **15,225** partitions across the full frozen enumeration range. This checked enumeration only.
- An independent labeled solver evaluated **6,145 binary/multiway transitions** and **7,030 first-coalescence prefix inequalities**, agreeing with the repository DP.
- Unequal-fibre solvers checked **145 coarsening transitions**, including simultaneous mergers, and agreed.
- Three in-memory mass-scheduled circuit probes achieved the exact predicted ratios and clean terminal states using an independent gate evaluator. On masses \((.02,.49,.49)\), descending scheduling achieved **0.98 bits**; the old compiler reached **0.99971144175281 bits**.
- Reproduced all **six uniform compiler smoke rows** after JSON normalization.
- `-O` and `-OO` rejected execution. Three escaping output paths and an attempted range override were rejected. An injected mismatch serialized the first witness before exit status 1; filesystem writes were mocked.

**Evidence admission conditions:** admit a future full `CONFIRM` only with the reviewed source hash, unchanged objective/ranges/arithmetic, normal Python execution, and all **15,225 unique cases** present with exact equality. A fresh-process complete run should have **15,225 DP states**. A mismatch admits its first witness as `KILL`; incomplete or changed-protocol execution supplies no confirmation. Weighted compiler evidence needs its own registered implementation and gate checks.

**Novelty remains unresolved.** Ebrahimi/Chen/Khisti already supplies the merge formula, monotonicity, descending largest-mass path, and its maximum-step expression in Algorithm 1 and Appendix A.3. The inspected theorem bounds EBIM approximation error; it does not establish the proposed fixed-final-fibre minimax optimum. [Primary source](https://arxiv.org/html/2410.21666#A3)

Meilă’s equation 28 and surrounding discussion already establish the uniform singleton-split formula and its minimality. Those ingredients earn no novelty credit. I did not identify a prior fixed-final-fibre weighted minimax theorem in the inspected sources; that is **not** evidence of its absence. [Primary source, §4.3](https://sites.stat.washington.edu/mmp/Papers/compare-jmva-revised.pdf)

SHA-256 hashes of reviewed sources and smoke artifacts:

```text
research/WEIGHTED_ERASURE_CANDIDATE.md
d377966f1133646d3b902aa868bc3e2d951a2d86d8f08edb77d196a801bb8d78
experiments/weighted_erasure.py
9592d370d94f3b5df5649b92834e45bee9b8aa5e0b3a53176bb0a87c1dccdbf3
research/ATOMIC_ERASURE_PROOF.md
b27440bb7723f0b6ef38cddceb0b2ec818ac330e7b0239f82f2481a093fbd5e7
experiments/atomic_compiler.py
40b24c94078b21a49ff6e4b7b73ff424ada096e06ea66bbb1ddcb86ba5327e4e
experiments/entropy_controls.py
c22765f9a084260d81dc274128939c57757545d23612f372f08ec10055044dad
results/weighted_erasure_smoke.json
53969e0ba2c5038745e6d7d12884093b8515791d9e9d12a9b31f6a52af39835a
results/atomic_compiler_smoke.json
4d4ceb95add1e25d8ca1a4dc1e1cb9b6102ad08ed3915c002c35d50195255695
```

The scoped mathematical result is an exact peak-loss budget for prescribed finite erasure. Mission achievement remains unestablished.