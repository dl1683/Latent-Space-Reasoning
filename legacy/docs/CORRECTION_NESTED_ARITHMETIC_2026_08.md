# Correction: nested-arithmetic perturbation and scaling claims (August 2026)

This is the canonical, permanent record of a correction to this repository's
published claims. External articles and posts that promoted the withdrawn
claims link here.

## 1. Claims withdrawn (quoted from the README as published)

1. "Perturbation Unlocks Capabilities That Scaling Cannot" — including
   "2 random embedding tokens raise Qwen3-4B from 32% to 72% (plurality@10) on
   nested arithmetic. Scaling from 1.7B to 32B is flat or worse (28%→36%→0%)."
2. "Perturbation beats temperature sampling at equal cost (72% vs best 64%)."
3. The cost-per-capability comparison: "At roughly the same FLOP budget as a
   single 32B pass, perturbation×10 on 4B achieves 72% vs 0%."
4. The interpretation that per-model deltas reflected a model-dependent
   computation/convergence mechanism.
5. The suggestion that the 32B result was "likely a quantization artifact."

## 2. What invalidated them

The nested-arithmetic benchmark conflated arithmetic capability with
termination under a 1024-token cap in Qwen thinking mode. From our own stored
result files in `experiments/`:

- **32B baseline** (`scaling_ladder_32b_4bit_baseline.json`): 25/25
  generations hit the cap, 0 terminated, 0% "accuracy". 14B: 19/25 at cap.
- **4B perturbation data** (`sensitivity_sweet_spot_random_noise_t2_results.json`):
  100/100 generations that terminated by EOS were correct (94 perturbation +
  6 baseline, zero exceptions); truncated generations scored 22.4%, the base
  rate for the last integer of a severed trace. The published +19.6pp was a
  termination-rate change (24% → 38%).
- **8B data**: same pattern (34/34 terminated generations correct;
  termination 8% → 13%).
- **DeepSeek-R1-1.5B** — the only model that terminated on 100% of tasks,
  hence the only capability measurement in the set — showed perturbation
  *reducing* termination (100% → 91%) and accuracy (−1.6pp).
- **The direct control was never run.** With thinking mode disabled — one
  chat-template flag — every model from 1.7B to 32B scores 96–100% on the
  same 25 tasks under the same cap at ~220–300 tokens
  ([BENCHMARK_VALIDITY_ASSESSMENT.md](BENCHMARK_VALIDITY_ASSESSMENT.md)).
  The difficulty-tier labels (`hard_nested` "~8%") measured token exhaustion.
- bf16 unquantized 32B with thinking on scores 4%: quantization was not the
  cause.

Our own word-problem perturbation run
(`sensitivity_random_noise_t2_results.json`) shows the identical coupling:
baseline 14 correct = 14 terminated; perturbation latents 14/16/14 correct with
15/16/16 terminated.

## 3. How the error was established

Controls contributed by Igor Rivin ([@igorrivin](https://github.com/igorrivin)),
run on an NVIDIA GH200 — PR #4 (four measurement bugs, Linux/multi-model
portability, 33 tests) and PR #5 (thinking-mode control, un-truncated scaling
ladder, cross-family evidence, a capability-limited diversity study with a
numerical-noise control) — together with our reanalysis of the stored result
files, which reproduces his findings exactly.

Our archived internal audit (`archive/tesla_session/`, spring 2026) had
identified the completion/extraction confound (P(correct|EOS) = 1.000), the
perturbation token-counting bug, and pre-registered budget and no-think
controls. We did not execute those controls, and the README claims remained
public. That process failure is ours; section 6 addresses it.

## 4. What survives, and what is open

- **Convergence diagnosis** — "the bottleneck is convergence, not
  knowledge": supported and strengthened. Under a binding cap in thinking
  mode, perturbation reliably shifts trajectories toward completion. On
  arithmetic this is moot (disabling thinking is cheaper and better); it is
  not evidence of improved reasoning.
- **Diversity source — stack-dependent.** On GH200 (non-deterministic greedy
  decoding), randomized prefixes matched a fixed-prefix numerical-noise
  control on all ensemble metrics; the reported per-generation accuracy
  deficit (Welch p = 0.037 over seed rows) is not established under
  task-level paired analysis (paired t p = 0.33, Wilcoxon p = 0.29). On our
  RTX 5090 / Windows / torch 2.11+cu128 stack, Igor's unmodified
  `probe_greedy_trajectory_stability.py` decodes byte-identical inputs
  identically across three independent processes (zero noise floor) while
  perturbed inputs diverge (2/8 and 7/8 distinct completions, up to 3
  distinct answers, tasks 1–2 of 5; the run could not be completed locally
  because of a hardware power-delivery fault, and will be completed on stable
  hardware). Perturbation is a causal intervention on deterministic stacks;
  its usefulness on capability-limited tasks is unresolved on either stack.
- **Judge-based planning, legal, and text-generation results** — suggestive,
  uncontrolled. The tasks terminated, so the truncation confound does not
  apply directly, but the baselines were single greedy thinking-mode
  generations; no no-think baseline, temperature-matched best-of-k, or null
  arm exists. Hypotheses until those controls are run.
- **Separatrix probe** — interpretation frozen pending determinism,
  endpoint-selection, and norm-preserving interpolation controls.
- **Diffusion latent repair and multi-latent aggregation** — separate
  research lines; unaffected by this correction.

## 5. Chronology

- 2026-03: nested-arithmetic perturbation results published in README.
- 2026-04 to 2026-06: internal audits identify the EOS/extraction confound
  and the token-accounting bug; controls pre-registered but not run; README
  unchanged.
- 2026-08-02: PRs #4 and #5 submitted.
- 2026-08-27: findings verified against stored data; both PRs merged; this
  correction published.

## 6. Process changes

1. **Termination gate.** No accuracy claim on generation tasks without the
   termination rate beside it; a benchmark with <95% baseline termination is
   invalid for capability claims.
2. **Direct control first.** Any intervention claim is compared against the
   cheapest direct remedy (template flags, budget, prompting) before any
   scaling or cost comparison.
3. **Null models.** Diversity claims require a fixed-input null on the same
   hardware, with stack determinism measured and reported.
4. **Task-clustered statistics.** No per-generation significance from
   task-nested samples without clustered or paired analysis. (Our own
   published McNemar values had this flaw.)
5. **Claim propagation.** Any internal audit that downgrades a public claim
   updates the README in the same working session.
6. Model revisions and chat-template versions are pinned in every result file.

## 7. Credit

The controls, the bug fixes, and both assessment documents are the
contribution of Igor Rivin. We thank him for the rigor and for the correction
this repository needed.
