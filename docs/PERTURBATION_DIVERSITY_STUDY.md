# Is embedding perturbation a real diversity source?

**Verdict:** no. On a capability-limited benchmark, embedding perturbation is
statistically indistinguishable from applying no intervention at all, and it
significantly *degrades* per-generation accuracy. About 80% of the answer
diversity it appears to create is already produced for free by floating-point
nondeterminism in greedy decoding.

**Model:** `Qwen/Qwen3-4B`, bfloat16, unquantized, thinking mode disabled.
**Hardware:** NVIDIA GH200 480GB, torch 2.11.0+cu130, transformers 5.5.3.
**Drivers:** `experiments/run_capability_limited_diversity_study.py`,
`experiments/probe_greedy_trajectory_stability.py`
**Raw:** `experiments/capability_limited_diversity_wide_mult.json`,
`experiments/greedy_trajectory_stability.json`

---

## 1. Why a new benchmark was needed

Every perturbation comparison in this repository was run on a benchmark where
accuracy is termination rate in disguise. Of the Qwen3-4B generations that
terminated inside the 1024-token cap, **100 of 100 were correct**; truncated
ones scored 11-22%, the base rate for the last integer of a severed trace
happening to be right. Perturbation's entire measured benefit there was raising
the termination rate from 24% to 38%. See
[BENCHMARK_VALIDITY_ASSESSMENT.md](BENCHMARK_VALIDITY_ASSESSMENT.md).

Under that confound, perturbation-vs-temperature is uninterpretable: both arms
are scored on whether the model stopped talking in time.

## 2. Design

**Precondition, enforced in code.** The baseline must terminate on >=95% of
tasks and still be wrong. Thinking mode off; 2048-token budget against an
observed mean of 626. The driver aborts rather than reporting numbers if the
precondition fails, because a budget-limited baseline reproduces the very
artifact this study exists to avoid.

**Task tier.** `wide_mult` (added here): 4-digit x 4-digit multiplication cores
with only 2-4 operations. Existing tiers scale difficulty by piling on
operations, which lengthens traces roughly linearly; `frontier_nested` averages
~1250 tokens. Wide multiplication is a different axis — four partial products
and a carry-heavy sum is the most error-prone single step for a small model, but
takes a handful of lines. Harder *and* cheaper.

Calibration on Qwen3-4B, thinking off (all terminate; these are capability
measurements, not truncation):

| tier | accuracy | mean tokens |
| --- | ---: | ---: |
| `hard_nested` | 92% | 342 |
| `brutal_nested` | 96% | 517 |
| `frontier_nested` | 72% | 1247 |
| **`wide_mult`** | **64%** | **626** |

Note the generator documents `hard_nested` as "~8% baseline" and
`brutal_nested` as "~12%". Those labels were calibrated under truncation and
are wrong by an order of magnitude; `brutal_nested` is in fact *easier* than
`hard_nested`.

**Arms**, all at k=10, all batched across seeds (identical prompt per row, so no
padding is involved):

| arm | what varies |
| --- | --- |
| baseline | nothing; one greedy generation |
| **noise floor** | nothing — k byte-identical rows in one batch |
| perturbation | k random embedding soft prompts at native embedding RMS, greedy |
| temperature | k sampled generations at t=0.6 and t=1.0 |

The **noise floor is the null model** and is the control the repository never
had. Its k rows are mathematically identical, differing only in floating-point
reduction order. Any diversity it yields is free, and perturbation must beat it
to be doing anything at all.

**Headline metric.** For distillation on verifiable tasks you have a checker, so
plurality is not the quantity of interest — yield is. **Rescue rate** = oracle@k
restricted to the tasks the baseline fails: the label yield a distillation loop
actually consumes.

## 3. Greedy decoding here is not reproducible

Established first, because it determines what the null model is
(`probe_greedy_trajectory_stability.py`, 5 tasks, k=8, greedy throughout):

| Condition | Distinct completions | Distinct answers (5 tasks) |
| --- | ---: | ---: |
| A. same sequential call, run twice | 7 of 10 | 5 |
| B. identical rows, one batch (pure FP noise) | 13 of 40 | **9** |
| C. perturbed rows, one batch | 31 of 40 | **8** |

Condition A rules out a batching bug: the *sequential* path already fails to
reproduce itself on 2 of 5 tasks. Some tasks are perfectly stable (1/8 distinct)
while others explode (7/8) — the signature of chaotic sensitivity near decision
boundaries, not a code defect.

Perturbation produced more *textual* diversity than pure noise (31 vs 13) but no
more *answer* diversity (8 vs 9).

## 4. Results

25 `wide_mult` tasks, k=10. Baseline: **60%** accuracy, **96%** termination —
precondition met, so failures are capability failures.

| arm | mean acc | sd(seed) | plurality@10 | oracle@10 | **rescue** | tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| noise floor | **58.8%** | 5.7% | 68% | 88% | **80%** | 645 |
| perturbation | 52.8% | 6.2% | 68% | 88% | **80%** | 653 |
| temperature 0.6 | 45.6% | 5.1% | **80%** | **96%** | 90% | 522 |
| temperature 1.0 | 41.6% | 6.3% | 68% | 92% | 100% | 474 |

Mean distinct answers per task:

| arm | all tasks | baseline-failed tasks | tasks with all 10 outputs identical |
| --- | ---: | ---: | ---: |
| noise floor | 3.00 | 3.20 | 2/25 |
| perturbation | 3.72 | 3.40 | 0/25 |
| temperature 0.6 | 6.16 | 6.80 | 0/25 |
| temperature 1.0 | 6.36 | 7.40 | 0/25 |

### What is statistically supported

**Perturbation costs accuracy.** 52.8% vs the noise floor's 58.8%, Welch t on 10
seed-level accuracies, **p = 0.037**. Well powered (250 generations per arm).

**Perturbation is indistinguishable from the null on every ensemble metric.**
Identical point estimates — 68% plurality, 88% oracle, 80% rescue — with one
discordant task in each direction (McNemar p = 1.0). Two arms differing only in
whether a deliberate embedding perturbation was applied, landing on exactly the
same ensemble behaviour.

**Most of perturbation's diversity is free.** 3.00 of its 3.72 distinct answers
per task (~80%) are produced by floating-point noise alone.

### What is NOT statistically supported

**Temperature's ensemble advantage.** Directionally ahead on every ensemble
metric and ~25% cheaper in tokens, but oracle@10 of 96% vs 88% gives McNemar
p = 0.5, and rescue-rate CIs are [49%, 94%] against [72%, 100%]. With 25 tasks
and 10 baseline failures this comparison is underpowered. It is suggestive, not
established.

## 5. Consequence for the distillation motivation

If the goal is to run perturbation + voting offline at 10x token cost, harvest
correct answers, and distil them back for 1x inference, then the number that
matters is rescue rate — and perturbation's is **80%, exactly equal to running
the model ten times completely unchanged**. There is no label yield attributable
to the perturbation itself. The same training set is obtainable by sampling the
base model, which additionally produces twice the answer diversity for fewer
tokens.

The distillation idea is sound on its own terms. It just does not need, and does
not benefit from, embedding-space perturbation as its diversity source.

## 6. Limitations

- **One model, one task family.** Qwen3-4B on nested arithmetic. Perturbation
  could behave differently on other architectures or on tasks with a different
  answer structure.
- **n = 25 tasks, 10 baseline failures.** Ensemble metrics are underpowered; the
  per-generation accuracy comparison is not. Scaling to ~100 tasks would settle
  the temperature-vs-perturbation question.
- **The noise floor is hardware- and kernel-dependent.** Non-deterministic
  greedy decoding is a property of this GPU and this kernel stack. Its magnitude
  may differ on the machine where the original results were produced. This does
  not affect the finding that perturbation adds no answer diversity *here*, but
  the floor is not a universal constant.
- **Two temperature settings only** (0.6, 1.0), each k=10. Reporting the better
  of the two favours the alternative hypothesis, which is deliberate.

---

Contributed by Igor Rivin ([@igorrivin](https://github.com/igorrivin)) with Claude Code (Claude Opus 5).
All runs on an NVIDIA GH200 480GB; raw result JSON for every number is in `experiments/`.
