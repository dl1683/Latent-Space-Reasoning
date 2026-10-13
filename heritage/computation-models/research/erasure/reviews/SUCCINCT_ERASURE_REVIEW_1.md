| Gate | Verdict |
|---|---|
| **CODE** | **REPAIR** — failure reporting permits stale positive output |
| **LOWER PROOF** | **PASS / PROVED AT SCOPE** |
| **UPPER PROOF** | **PASS / PROVED AT SCOPE** |

**Yes: the mathematics resolves H1’s simultaneous gate-work conjecture under the stated contract.** For \(n\ge2\), every feasible rational cap admits
\[
G=\Theta(n+r+n/K),\qquad W=\Theta(n+r).
\]
This establishes neither physical nor classical efficiency advantage. FULL execution remains blocked by the code repair; evidence admission remains separate.

I read AGENTS.md, the required [dual-loop skill](<USER_HOME>/.claude/skills/dual-loop/SKILL.md), startup materials, candidate, implementation, smoke, registrations, H1, and approved atomic/weighted reports. This was read-only, CPU-only; no registered FULL, file edits, or additional agents.

**CODE — blocking finding.** At [succinct_erasure.py:308](</<LOCAL_PROJECTS>/new-computation-model/experiments/succinct_erasure.py:308>), all cases execute before output publication. An injected failure at case two:

- stopped before case three;
- produced no failure artifact or saved witness;
- performed zero output writes;
- left the mocked older `CONFIRM` unchanged.

Thus a failed attempt can leave an apparently positive artifact at its requested path. Before FULL, invalidate previous success for the current attempt, save the first failing case and diagnostic witness, exit unsuccessfully, and publish `CONFIRM` only after complete validation tied to the loaded source and attempt.

Otherwise, the inspected implementation passed the bounded audit: exact integer loss products and cap classification, valid gate arities/distinct indices, full terminal-state comparison, macro prefix/data invariants, frozen enumeration, optimized-execution guards, and resolved-path confinement. The end-of-run source hash still requires a stable source snapshot for evidence admission.

**LOWER PROOF — PASS.** Initially there are \(M=2^{n+r}\) distinguishable complete states. Final fibres have largest size \(q=2^n-1\). Determinism prevents class splitting or crossing final fibres, so first complete formation forces
\[
K\ge K_*=\phi(q)/M.
\]
Extra zero scratch, reversible recoding, or arbitrary preceding depth cannot evade this.

The terminal entropy is \(r+h_2(2^{-n})\); telescoping gives \(GK\ge n-h_2(2^{-n})\). Every input wire and every nonconstant output wire must be targeted, giving \(G\ge n+2r+1\). Together these imply the claimed work lower bound. Required registers already force linear width.

**UPPER PROOF — PASS.** The [construction](</<LOCAL_PROJECTS>/new-computation-model/research/SUCCINCT_ERASURE_CANDIDATE.md:47>) survives the requested attacks:

- Before each Gray RESET, every prefix flag depending on the flipped bit is uncomputed. Retained and temporary predicates exclude that bit; merging states differ only in the disposable flag. Scratch retains no erased distinction.
- Edges at depth \(j\) occur \(2^j\) times, and
  \[
  \sum_{j=0}^{d-1}(d-j)2^j=2^{d+1}-d-2.
  \]
  Predicate maintenance therefore costs \(O(2^d)\).
- AND is computed from the original input and preserved. Returned Z remains pointwise correct, its original input registers clear reversibly, and all workspace ends zero.
- At the boundary, the final all-ones vertex has zero \(Y=0\) mass. Moving the accumulated representative there loses zero entropy.
- Rational selection uses finite conservative comparisons. Feasibility guarantees \(s\le r+d\). Low-reset conditions depend only on unchanged high/context bits and Y, so remain recoverable.
- Residual coarse masses are exactly \(2^\ell/M\), except the last mass \((2^\ell-1)/M\). The descending weighted schedule meets \(U_d\); fallback \(d=n\) meets \(K_*\). Minimal selection bounds residual work by \(O(n/K)\).

Program generation/storage and precision are separately charged. Gate/width optimality supplies no optimal construction-bit-work claim.

**The \(n=1\) case is separate:** the implemented circuit uses exactly \(2r+2\) CNOTs, no RESET, and peak loss zero.

**Actual probes — EMPIRICAL:**

- Reproduced all **36 smoke rows**, including complete traces: **22 CONFIRM, 14 CANNOT**, matching the current source hash.
- Independent evaluator checked **18 circuits**, **645,668 atom/gate transitions**, **465 RESETs/macros**, and **666 complete-state collision pairs**; included intermediate rational branches and fallback.
- Checked all **258 directed adjacent Gray merges** through five bits over **25,488 complete-state assignments**, including inactive roots and unrelated scratch.
- Verified Gray counts/amortization through **12 dimensions**.
- Checked **6,650 rational classifications**, **6,155 feasible promise/selection cases**, and four residual-mass distributions.
- Rejected nine injected gate/output defects, three escaping paths, actual `-O`/`-OO` executions, and explicitly optimized compilation.
- Enumerated/classified the frozen FULL cases **without executing their circuits**: **144 unique cases**, comprising **114 feasible and 30 CANNOT**.

**Literature occupancy remains open.** Direct existing ingredients include Bennett’s compute/copy/reverse construction; Demaine et al.’s time/space/irreversibility framework; Frank’s support-sensitive zero-loss reversibility; and Chanavat–Srinivasan’s compositional conditional reversibility. Demaine’s §2.1 gate cost differs from this full-state distribution-dependent peak metric. I did not identify a direct equivalent of the complete H1 frontier in the inspected passages; this incomplete search earns no priority. [Bennett](https://www.cs.princeton.edu/courses/archive/fall06/cos576/papers/bennett73.html), [Demaine et al.](https://erikdemaine.org/papers/Energy_ITCS2016/paper.pdf), [Frank, Theorem 6](https://arxiv.org/pdf/1806.10183), [Chanavat–Srinivasan, §4.4](https://arxiv.org/html/2511.02425v2).

SHA-256 fingerprints, unchanged when rechecked after probes:

```text
research/SUCCINCT_ERASURE_CANDIDATE.md
115dc5c25353fd46e55eadbde670a087e147674fdaf76e971e7b474c1592e4bc
experiments/succinct_erasure.py
0ac7c9f1d3acf6312250c18c3b72cd149e5d5c7f0189745a90a764eaa88b56d6
results/succinct_erasure_smoke.json
f24961cd068bc2a99e71a5202dadb95cec9efdaf3d1be216d7557b332b41103e
EXPERIMENTS.md
d26350ce227b9b3f377eb359bde006b5990220fc51cc0ebde9e4cec9c4692585
experiments/ledger.jsonl
cee1c8f7f9b7fa772dc805eabeef319aa9b41331f5ae9ac13c054a5069e8b128
research/QUESTION_RESTART_RESULT.md
fccd3b2f9cc65598946b06c8e3cd862eb6d5f47e6788a9ef5ac56fe4d6c529a8
research/ATOMIC_COMPILER_REVIEW_2.md
87149955477d942958ce1c81e6f2838813a097dcf04f0577642a5c5cafdf057c
research/WEIGHTED_ERASURE_REVIEW.md
b47755616594841e767f6038666dee1d50031c0b6548a130f44f75b6e327d37b
research/ATOMIC_ERASURE_PROOF.md
ffb0f7862fb326f1a92956a9afff9ef5cc923d26202d7f981e5190a2d6f3e1ed
research/WEIGHTED_ERASURE_CANDIDATE.md
225fed71f44ec6f718220b9b03b102246653103eacf29e5305ad7a8218954633
```

After repair, require a new fingerprint, refreshed smoke and code gate, then separate admission of all 144 FULL records with independent reconstruction of outputs, cleanup, invariants, counts, and exact loss ratios. No FULL evidence or whole-mission conclusion is certified here.