# A real chronological test for operation discovery, not premise retrieval

28 November 2026. Classification: mission-directed evaluation design and public source-history audit. Reviewed LSR head `eaf15bfe`. This is a separate possible post-qualification development study, not a frozen panel modification, learned result, or authorization to run anything.

## Decision

Test whether a worker can **invent and retain a reusable operation before the later problem exists**, not just predict which existing premise will be retrieved. Formal software verification offers an independently authored evolving corpus, explicit dependencies and a potentially inexpensive relative-to-training external witness. A proof kernel can reject invalid generated computations; it cannot certify general natural-language semantics or economic advantage.

The [counterexample-driven latent library](COUNTEREXAMPLE_DRIVEN_LATENT_LIBRARY.md) already proposes learned inventors and local transition modules. The missing training/test ingredient is a genuine later operation demand, not another supplied arithmetic operator. Use chronological development histories for a bounded formal-domain study; retain the current ordinary-reader-first sequence.

## Prior art and strong comparators

- [HipSpec, Claessen et al., 2013](https://research.chalmers.se/en/publication/175485) generates equational properties of recursive programs by theory exploration, testing and inductive proof. This is a direct symbolic competitor to query-blind library invention; a worker that finds only supplied equational laws has not surpassed it.
- [Proof repair across type equivalences, Ringer et al., PLDI 2021](https://doi.org/10.1145/3453483.3454033) transforms Coq proof terms across type equivalences. This is an established example of symbolic transport across revisions, not evidence for neural transport or arbitrary semantic repair.
- [VeriSoftBench, Xin et al., Feb 2026, sections 3–4](https://arxiv.org/html/2602.18307v1) packages 500 proof obligations from 23 actual Lean repositories. Its curated setting uses the reference proof's direct dependencies and associated definitions. That context is an oracle-style access diagnostic, not a fair task-blind acquisition policy. Its full context is static and sometimes truncated. Neither condition proves chronological operation invention. The [release README](https://github.com/utopia-group/VeriSoftBench/blob/8f5e8b36ab5fbdec6fd8d8f4ee0580d29647097a/README.md) says the full Docker image is approximately 110 GB; do not download it to the current host or assume a full benchmark is feasible. The pinned [repository manifest](https://github.com/utopia-group/VeriSoftBench/blob/8f5e8b36ab5fbdec6fd8d8f4ee0580d29647097a/repos.json) includes a small MIT-licensed expression evaluator with no Mathlib dependency and records Lean 4.24.0 for the packaged environment. Inspect upstream toolchain and exact licence at each selected historical pin before running.

## Independently authored example: a reusable law survives a structural change

Inspected actual public Git history of [Paul-Passeron/LeanExprEvaluator](https://github.com/Paul-Passeron/LeanExprEvaluator), a repository named in the VeriSoftBench manifest. This is a development example selected retrospectively, not an independent held task. Neither the 500-task dataset nor a private LSR packet was opened.

| Upstream commit | Source-level observation |
|---|---|
| [79fa176](https://github.com/Paul-Passeron/LeanExprEvaluator/commit/79fa176b7fcf22e4de3db540f254391054b870ed) | Arithmetic-specific composition lemma `chasles_step_star` in ArStepLemmas.lean. |
| [e5f20c9](https://github.com/Paul-Passeron/LeanExprEvaluator/commit/e5f20c984aa6389d84dafd8065d507dcab381304) | Generic reflexive/transitive StepStar abstraction introduced. |
| [5325ffd](https://github.com/Paul-Passeron/LeanExprEvaluator/commit/5325ffdb63daca00742f0f653c7cc8f897f4abdc) | Composition generalized to arbitrary StepKind; one-step embedding lemma introduced. Boolean Not completeness already calls both. |
| [fe1e308](https://github.com/Paul-Passeron/LeanExprEvaluator/commit/fe1e308a285ada831fe7f5458b9a89f253016557) | Expressions and stepping machinery reorganized for mutual induction; arithmetic/Boolean files combined. |
| [6fbf4e1](https://github.com/Paul-Passeron/LeanExprEvaluator/commit/6fbf4e1bc0dd04481e7f4026755c205dc558aa16) | If constructor and conditional stepping introduced; some preservation cases still contain sorry. |
| [f838499](https://github.com/Paul-Passeron/LeanExprEvaluator/commit/f838499066ee275002b4efad05f81f9332f2daa4) | Completeness extended to conditionals using mutual recursion. In both conditional outcomes, composition combines the lifted condition trajectory with the selected branch trajectory. |
| [0aff744](https://github.com/Paul-Passeron/LeanExprEvaluator/commit/0aff7447dfb98c318b3f9f8fba4c427d2fa3920c) | Benchmark-manifest snapshot; both generic lemmas and both conditional composition calls remain. |

These are source/diff observations, not independently kernel-checked proofs. In particular the early refactor explicitly anticipates If; do not count its later use as wholly unforeseen if a worker receives that revision. A test of less predictable transfer must use a cutoff before those hints, and split whole repositories/families rather than just nearby commits.

A lexical audit of pinned ExprEval/Basic.lean (non-full-line-comment lines containing a whole-word identifier) found:

| Pin | Lines with either identifier | Lines with composition identifier | Lines with one-step embedding identifier |
|---|---:|---:|---:|
| 79fa176 | 3 | 3 | 0 |
| 5325ffd | 5 | 4 | 1 |
| fe1e308 | 19 | 16 | 7 |
| f838499 | 21 | 18 | 7 |
| 0aff744 | 21 | 18 | 7 |

The columns overlap. These are line counts, not unique proof uses, causal effects, compiler dependencies or savings. The two generic theorem blocks in Expr.lean are exactly text-identical at 5325ffd, f838499 and 0aff744. At f838499 the new If branch has exactly two lines calling the generic composition lemma, one for each branch; this is visible in [the later source](https://github.com/Paul-Passeron/LeanExprEvaluator/blob/f838499066ee275002b4efad05f81f9332f2daa4/ExprEval/Basic.lean). Source inspection thus distinguishes an invariant reusable law from domain-specific proofs repaired around it. There is no evidence that latent workers can discover either.

Reproduction: fetch the named files at the exact pins above (not main). This V8 audit was executed on the fetched text:
```js
function counts(s) {
  const a = s.split("\n").filter(x => !x.trim().startsWith("--"));
  const c = a.filter(x => /\bchasles_step_star\b/.test(x));
  const t = a.filter(x => /\bstep_to_stepstar\b/.test(x));
  return [a.filter(x => /\b(?:chasles_step_star|step_to_stepstar)\b/.test(x)).length,
          c.length,t.length];
}
function lemmaBlock(s) {
  const a=s.split("\n");
  const i=a.findIndex(x=>x.startsWith("theorem chasles_step_star"));
  const j=a.findIndex(x=>x.startsWith("theorem step_to_stepstar"));
  if(i<0||j<=i) throw Error("declarations absent");
  let k=j+1;
  while(k<a.length&&(!a[k].trim()||/^\s/.test(a[k]))) k++;
  return a.slice(i,k).join("\n").trim();
}
```
The first five rows above were inspected at those pins; equality of all three extracted declaration blocks was checked. No Lean installation or build was available; no model, proof search, learned worker, grader or gate run.

## Proposed discriminating study (not registered)

Choose several independent small projects with chronological histories, fixed eligible source universe and trusted proof checker. Use this public history for development only. Have an independent custodian select later proof obligations and source/rule edits, including a new operation family, after the opening corpus and budgets are set. A whole repository that already contains the target proof, or its downstream consequences, is not an admissible opening corpus: stripping the target proof alone leaks knowledge through later files and dependencies. Publish the exact older commit tree before exposing later targets; disallow future commits, PR descriptions, solution-bearing logs and proof artifacts. Also address pretrained contamination with fresh private problems built and graded independently, not public source-history claims alone.

1. At opening, the worker receives actual source definitions, existing lemmas, proofs and provenance, not the future theorem or a supplied operation registry. It proposes reusable typed latent transitions, applies them to new operands and challenges them on public developmental witnesses. A kernel-checked decoded statement/proof can be an *admission witness*; the task-time resident operation must still do useful computation in latent state. Record witness generation and checking costs. Exact symbolic proof artifacts may be retained by the comparator.
2. At revisions, supply the same source diffs to all arms. Separate changes of names/files from changes of semantic types, laws and admitted operand domains. Maintain historical pins, invalidate dependent results, and distinguish transport of a law from transport of its proof or proof strategy. Include unchanged useful laws and an actually invalidated law. The latter requires an independently verified failing or counterexample witness; a failed tactic alone is not a disproof.
3. Late tasks require application of the learned law to new operands, at least one changed consequence, and a protected unaffected obligation. Give the same reader/prover its normal task-time proof checker in all arms, with fixed candidate, time, source-open and verification budgets. Report first latent predictions separately from checker-assisted repair. A checker that simply constructs the proof for the reader is not evidence of latent execution.
4. Compare query-blind symbolic theory exploration, incremental proof reuse/repair, ordinary text sleep-worker, equal extra task-time search, no memory and latent worker. In a separate diagnostic expose the reference dependency closure equally to all relevant arms; if gains vanish, the measured contribution was premise selection. The warm symbolic arm may reuse generic lemmas and compiled proof objects; do not handicap it by requiring cold reproving.
5. Remove and restore only the acquired computation at the same source pin. Keep its source evidence and eligible read rights fixed. Match access and dispatch scaffolding and include an identity or wrong-operator control. A positive result requires valid new consequences and a source-sensitive revision effect, not only more theorem-name matches. Count the cost of all proposals, failed checker calls, build/toolchain, source bytes, retained latent and symbolic state, history, and downstream verification. Refuse to claim edge economics without actual matched-quality measurements.

Two complementary endpoints: (i) verified task success and revision correctness, with causal use under intervention; (ii) savings on repeated independent tasks against competent warm symbolic search. The exact formal library may be unbeatable on many cases; this is useful falsification. A pass supports only formal-domain learned reuse under the tested edit family, not natural-language reasoning, multimodal integration or general intelligence. Do not turn the project into a theorem prover: the reason to use this domain is a trustworthy independently authored test of query-blind learned computation. Transfer to semantic corpora remains the goal.

**Next decision:** reader qualification is still the first registered run. If qualified, use a learned semantic worker with crossed-world tests as planned; consider this formal-domain study only if an inexpensive independent verifier and genuinely later task custody can be arranged. The [system design's gates](LATENT_WORLD_SYSTEM_DESIGN.md#7-research-sequence-and-decision-gates) and [independent gate](INDEPENDENT_ORDINARY_GATE_AND_SCALAR_BOUNDARY.md) remain untouched.
