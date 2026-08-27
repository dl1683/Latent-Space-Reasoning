# Latent Space Reasoning

This repository is a research lab for a different unit of inference.

Most reasoning systems ask a model for one completion, maybe sample a few more,
then pick a winner. This project asks a stronger question:

**Can a frozen model reason better if we treat its latent generation path as an
editable, inspectable search space instead of a one-shot text sample?**

The long-term target is a control layer for reasoning itself:

`sample -> inspect -> repair -> aggregate -> verify -> realize`

That means a completion is not the final object. It is evidence: a partial
reasoning artifact that can be compared, repaired, merged, rejected, or used as
source material for a stronger answer.

## Why This Matters

Winner-take-all inference wastes partial work.

One candidate may have the right plan but miss a constraint. Another may catch
the constraint but have weak structure. Another may surface a risk, preserve an
exact role, or expose an edge case. A selector can only choose one. A stronger
reasoning system should preserve the useful parts from several trajectories
without importing contradictions or unsupported claims.

This repo explores that family:

- token and prefix perturbations as cheap trajectory diversity;
- diffusion denoise histories as intermediate reasoning state;
- latent repair as local editable-state intervention;
- candidate promotion as spend and selection control;
- multi-latent aggregation as source-supported synthesis across trajectories;
- proof objects, gates, and cost accounting as the discipline that keeps the
  claims honest.

## Quick Start

Install from source:

```bash
git clone https://github.com/dl1683/Latent-Space-Reasoning.git
cd Latent-Space-Reasoning
pip install -e .
```

For quantized models (recommended):

```bash
pip install -e ".[quant]"
```

Compare baseline versus latent reasoning on any query:

```bash
latent-reason compare "How do I implement user authentication?"
latent-reason compare "Design a REST API" --encoder Qwen/Qwen3-4B
```

Or use the Python API:

```python
from latent_reasoning import reason, compare

result = reason("How do I implement caching?")
print(result.plan)

cmp = compare("How do I implement rate limiting?")
print(cmp["baseline"])
print(cmp["latent_reasoning"])
```

Other CLI commands:

```bash
latent-reason run "Design microservices" --encoder Qwen/Qwen3-1.7B
latent-reason models          # list recommended models
latent-reason check-gpu       # check GPU availability
```

Minimum hardware: ~2GB VRAM (Qwen3-0.6B). Recommended: ~8GB VRAM (Qwen3-4B). CPU-only works but runs slower.

## Correction — 2026-08-27: nested-arithmetic perturbation claims withdrawn

Earlier versions of this README claimed that soft-prefix embedding perturbation
unlocked arithmetic capability unavailable through model scaling
("Perturbation Unlocks Capabilities That Scaling Cannot"), outperformed
temperature sampling, and delivered superior cost-per-capability. **Those
conclusions are withdrawn.**

The nested-arithmetic benchmark they rested on was dominated by Qwen
thinking-trace truncation at a 1024-token cap. In our stored data, every
generation that terminated normally was correct (100/100), truncated
generations scored near the last-integer base rate, and the measured accuracy
gains were changes in termination rate. Disabling thinking mode — a control we
never ran — takes every model on the ladder, including the 32B we reported at
0%, to 96–100% on the same tasks at a fraction of the cost.

The error was established by controls contributed by Igor Rivin
([@igorrivin](https://github.com/igorrivin)) in PRs #4 and #5, together with
reanalysis of our own stored result files. Full record — withdrawn claims,
evidence, what survives, chronology, and process changes:
[docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md](docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md).


## Current Evidence

| Area | Status | What It Shows |
| --- | --- | --- |
| Diffusion latent repair | Promoted public result | A frozen diffusion language model can improve on a lean mixed benchmark by repairing latent generation state. |
| Benchmark validity review | **Independent controls — see [docs/BENCHMARK_VALIDITY_ASSESSMENT.md](docs/BENCHMARK_VALIDITY_ASSESSMENT.md) and [docs/PERTURBATION_DIVERSITY_STUDY.md](docs/PERTURBATION_DIVERSITY_STUDY.md)** | The nested-arithmetic rows below were run with Qwen3 thinking mode on against a 1024-token cap, and 76–100% of every rung was truncated. With `--no-think` at the same cap, Qwen3-32B scores 100% (published: 0%) and every rung from 1.7B up scores 96–100%. On a capability-limited tier, perturbation is statistically indistinguishable from a fixed-prefix numerical-noise control on that GH200 stack (see owner addendum in that document). Read those two documents alongside the rows below. |
| Token/prefix perturbation (nested arithmetic) | **Withdrawn 2026-08-27** — truncation confound; see the correction above | The published 32%→72% gain, the flat scaling ladder, and the temperature/cost comparisons measured termination under a 1024-token cap. What survives is a trajectory-completion effect and untested judge-based results on open-ended tasks — see "Embedding Perturbation: What Survives the Controls". |
| Multi-latent aggregation v5 | Clean local milestone | A predeclared 48-task aggregation replay passed stricter robustness gates on planning tasks. |
| Aggregation v6-v8 | Negative transfer evidence | More repair, probes, and targeted standalone repair did not reliably create aggregation-useful complements. |
| Aggregation v9 | Post-failure design breakthrough | Complement-first packet generation passed frozen numeric replay gates on the failed v7 surface, but remains diagnostic. |
| Aggregation v10 | Fresh transfer promotion — ALL 13 GATES PASSED | Complement-first packets on fresh `plan_393`-`plan_440` slice: 40/48 coverage, 38 promotions, 40/8/0 W/T/L, mean lift +0.077, zero contradictions. |
| Aggregation v11 | 2x replication — ALL 13 GATES PASSED | LLaDA-only 96-task replication on `plan_441`-`plan_536`: 87/96 coverage (90.6%), 87 promotions, 87/9/0 W/T/L, mean lift +0.100, Wilson lower 0.831, zero contradictions. Keyword audit RED — rubric gameable but packets are not keyword-stuffing. |
| Blinded pairwise evaluation | STATISTICAL_GO — preregistered confirmatory study | N=50, 4 arms, 3 blinded same-model judge calls. Task-specific clause-append preferred over generic boilerplate at 33/50 (66%, p=0.016, Wilson CI [52.2%, 77.6%]). Task-specificity confirmed: true > deranged 47/50. Wrong-task clauses hurt: deranged < anchor 64%. Single-model judge caveat; error gate fails narrowly (6/50 vs ≤5 threshold). |
| Separatrix probe (latent interpolation) | Exploratory — interesting structural signal | Interpolating between wrong and correct perturbation vectors reveals non-monotonic correctness landscape: 74% of tasks show interior correctness islands, 47% of transitions involve deep divergence (>50 shared tokens before branching). Two mechanisms coexist (trajectory-level and format-level). Needs controls before strong claims. |

## Embedding Perturbation: What Survives the Controls

Two random vectors injected into the embedding space before generation, scaled
to the model's native embedding RMS; no training, frozen model. The method is
unchanged. What changed is what the evidence supports.

### Withdrawn (see the correction above)

The nested-arithmetic scaling ladder, the perturbation-vs-scaling and
perturbation-vs-temperature comparisons on that benchmark, and the
cost-per-capability table. All were measurements of termination under a token
cap. The README's earlier guess that the 32B result was a quantization
artifact was also wrong: bf16 unquantized scores 4% with thinking on.

### What the data does show: a trajectory-completion effect

Under Qwen thinking mode with a binding token cap, perturbation reliably shifts
trajectories toward normal completion: on Qwen3-4B it raised the termination
rate from 24% to 38%, and every generation that completed was correct
(100/100). This is the "convergence, not knowledge" diagnosis, and it is the
one part of the original section that the controls confirm. It is moot on
arithmetic — disabling thinking mode is a cheaper and far more effective
remedy on every tier tested — and it is not evidence of improved reasoning.

### Diversity source: real but stack-dependent

On a GH200/CUDA-13 stack, randomized soft prefixes did not improve any
ensemble metric over a fixed-prefix numerical-noise control
([docs/PERTURBATION_DIVERSITY_STUDY.md](docs/PERTURBATION_DIVERSITY_STUDY.md);
see the owner addendum in that document for scope and statistics). On our
RTX 5090 / CUDA 12.8 stack, the same probe decodes byte-identical inputs
identically across independent processes (zero numerical noise floor) while
perturbed inputs diverge (up to 7/8 distinct completions, 3 distinct answers
on a wide-multiplication task). Perturbation is therefore a causal
intervention on deterministic stacks; whether that diversity is *useful* on
capability-limited tasks is unresolved on either stack.

### Open-ended domains: suggestive, controls pending

Judge-based studies on hard planning (5 tasks), legal reasoning (12 tasks), and
open-ended text generation (15 tasks, blind pairwise LLM-as-judge, best of 5
perturbation seeds vs greedy) favoured perturbation — e.g. 7 wins / 1 loss / 7
ties vs the same 4B model, and 5/15 wins vs a 14B model. These tasks
terminated, so the arithmetic truncation confound does not apply directly. But
the baselines were single greedy generations in thinking mode, and the
studies lack a no-think baseline, a temperature-matched best-of-k arm, and a
fixed-prefix null. Until those controls are run they are hypotheses, not
results. Figures and data remain available:
`docs/figures/text_gen_judge_results.png`,
`docs/figures/text_gen_category_wins.png`,
`experiments/text_generation_results.json`,
`experiments/text_gen_judge_results.json`,
`experiments/planning_3way_outputs.json`,
`experiments/legal_v2_full_clean.json`.

### Next study

The open question is whether perturbation preserves the benefit of thinking
mode where thinking is genuinely necessary — tasks on which thinking-on with a
generous budget beats thinking-off — at lower token cost than either. That
requires, on held-out tasks: thinking-off, thinking-on at generous and binding
budgets, temperature-matched best-of-k, a fixed-prefix null, and perturbation,
with task-clustered paired statistics. Design is pre-registered before any
run; results land here only after passing the correction document's process
gates.

Historical data for the withdrawn section:
`experiments/sensitivity_sweet_spot_random_noise_t2_results.json`,
`experiments/temperature_vs_perturbation_results.json`,
`experiments/scaling_ladder_14b_4bit_baseline.json`,
`experiments/scaling_ladder_32b_4bit_baseline.json`, and cross-model variants.


## Promoted Result

The current promoted public claim is diffusion-native latent repair on the lean
mixed benchmark:

| Public arm | Score | Relative GPU cost |
| --- | ---: | ---: |
| Greedy/fixed denoise | 0.412277 | 1.000000x |
| Random perturbation | 0.372125 | 1.000000x |
| Latent repair | 0.531116 | 2.625000x |

There is also a lower-cost controller point at `0.508705` and `2.375000x`.
Use the top-score point for the headline reasoning-lift claim and the
lower-cost point for the controller/cost claim.

Canonical evidence:

- [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md)
- [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md)
- [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md)
- [docs/DIFFUSION_THEORY_CLAIM_LEDGER.md](docs/DIFFUSION_THEORY_CLAIM_LEDGER.md)

## Active Frontier: Multi-Latent Aggregation

The aggregation line is trying to move beyond "pick the best sample." It asks
whether the system can combine source-supported, non-overlapping useful
components from multiple latent trajectories into an answer that beats every
individual candidate.

The clean aggregation milestone is v5:

- fresh `plan_249` through `plan_296` planning slice;
- `34/48` complement coverage;
- `34` online promotions;
- `34/14/0` wins/ties/losses;
- mean realized score `0.402750` versus anchor `0.340964`;
- `0` unsupported additions and `0` hard contradictions.

The design arc from v7 through v10:

- v7 failed with `24/48` coverage against a `36/48` gate.
- v8 showed targeted standalone repair did not create complement evidence.
- v9 changed the source family to explicit complement packets and passed the
  diagnostic replay with `47/48` coverage and `46` online promotions.
- v9 was still not a fresh promotion because the packet source was added after
  the v7/v8 failures.
- v10 is the fresh transfer test — and it **passed all 13 frozen gates**.

v10 result (fresh `plan_393` through `plan_440`, complement-packet policy
frozen before labels):

- Complement coverage: `40/48` (83%, gate was 75%)
- Online promotions: `38` (gate was 30)
- Conditional promotion rate: `95%` (gate was 50%)
- Wins/ties/losses: `40/8/0`
- Mean anchor score: `0.365` → mean realized: `0.442` (lift `+0.077`)
- Wilson 95% CI lower bound: `0.657` (gate was 0.600)
- Unsupported additions: `0`, hard contradictions: `0`
- Leave-one-out range: `0.073..0.078` (no single task dominates)
- Source-family ablation: without complement packets, coverage drops to `15/48`

v10 artifacts:

- Freeze: [docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_FREEZE.md](docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_FREEZE.md)
- Anchor/label source: [docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_LABEL_REPORT.md](docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_LABEL_REPORT.md)
- Complement prompts: [docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_PROMPTS.md](docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_PROMPTS.md)
- Complement packet report: [docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_PACKET_REPORT.md](docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_PACKET_REPORT.md)
- Replay: [docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_PACKET_REPLAY.md](docs/reports/diffusion/LATENT_AGGREGATION_MULTI_ASPECT_V10_COMPLEMENT_PACKET_REPLAY.md)

Blinded pairwise evaluation artifacts:

- Preregistration: [docs/reports/diffusion/CONFIRMATORY_STUDY_PREREGISTRATION.md](docs/reports/diffusion/CONFIRMATORY_STUDY_PREREGISTRATION.md)
- Results: [docs/reports/diffusion/CONFIRMATORY_STUDY_RESULTS.md](docs/reports/diffusion/CONFIRMATORY_STUDY_RESULTS.md)
- Placebo diagnostic: [docs/reports/diffusion/PLACEBO_DIAGNOSTIC.md](docs/reports/diffusion/PLACEBO_DIAGNOSTIC.md)
- Pilot v2: [docs/reports/diffusion/LATENT_AGGREGATION_BLINDED_PAIRWISE_PILOT_V2.md](docs/reports/diffusion/LATENT_AGGREGATION_BLINDED_PAIRWISE_PILOT_V2.md)

## Exploratory Frontier: Latent Space Geometry

The separatrix probe experiments explore what the behavioral landscape looks like
*between* perturbation endpoints. Rather than treating each perturbation as an
independent sample, they ask: what happens when you smoothly interpolate through
latent space?

Key structural finding: the landscape is fragmented, not smooth. Correctness
flickers on and off along interpolation paths, and many correctness transitions
involve long shared reasoning prefixes before the model diverges at what appears
to be a decision point. This suggests the model's reasoning has richer internal
structure than endpoint sampling reveals.

This is early exploratory work. The structural signal is interesting but does not
yet prove computational basins or answer-free detectability. Controls (null
interpolations, full trace audits, alternative projections) are needed before
making strong claims. See `experiments/EXPERIMENTS.md` for the full analysis.

Artifacts:

- Results: `eval_results/separatrix_probe_v2/probe_results_v2.json`
- Driver: `experiments/run_separatrix_probe.py`

## What Not To Overclaim

- The withdrawn nested-arithmetic results must not be cited as evidence of
  capability or of perturbation beating scaling or temperature; they measured
  termination under a token cap.
- Judge-based planning, legal, and text-generation results are uncontrolled
  (no no-think baseline, no temperature-matched best-of-k, no null) and are
  hypotheses until those arms exist.
- Perturbation-as-diversity-source is stack-dependent; do not generalize a
  result from one hardware/kernel stack to another without measuring the
  numerical noise floor there.
- Old token perturbation runs are historically important, but many were small
  exploratory slices.
- Oracle coverage is not a deployment result unless a selector or aggregator
  can realize the gain without unavailable labels.
- Aggregation does not count if it only writes a longer answer.
- Post-failure diagnostic evidence can guide the next freeze, but it is not a
  fresh promotion claim.
- Current aggregation evidence is planning-local until broader transfer slices
  are run.
- Separatrix probe results show interesting structure but do not yet establish
  computational basins or answer-free correctness detection. Controls are needed.

## How To Read This Repo

Start here:

1. [START_HERE.md](START_HERE.md) for a one-page orientation.
2. [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md) for the
   promoted score/cost claim.
3. [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md) for claim-to-artifact
   provenance.
4. [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md) for
   canonical hashes and raw artifact pointers.
5. [docs/NAVIGATION.md](docs/NAVIGATION.md) for the compact map through current
   and historical reports.

Then read by intent:

- Mechanism: [docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md](docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md)
- Review path: [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md)
- Aggregation doctrine: [docs/LATENT_TRAJECTORY_AGGREGATION.md](docs/LATENT_TRAJECTORY_AGGREGATION.md)
- Generated diffusion report archive: [docs/reports/diffusion/README.md](docs/reports/diffusion/README.md)

Generated reports and raw outputs are retained for auditability, but they are
not the first-read path.

## Evidence Discipline

The repo is intentionally bold about the research direction and strict about
what counts as evidence.

A claim should be treated as exploratory unless it has:

- a predeclared or clearly bounded task slice;
- generated raw artifacts;
- score reports;
- cost accounting;
- explicit failure modes;
- tests or validators for artifact builders;
- a statement of what would falsify the claim.

For aggregation, the minimum proof shape is stricter:

- aggregate score must beat the best single candidate;
- component or aspect gain must be reported separately;
- unsupported additions must be zero;
- hard contradictions must be zero;
- cost must include extra generation/probe/diversity/packet sources;
- source-family ablations and equal-budget controls must be reported;
- post-hoc diagnostic evidence must not be described as a frozen promotion.

## Repository Layout

```text
archive/        Historical notes and legacy snapshots; not first-read material
docs/           Reader guides, theory docs, runbooks, and consolidated status pages
experiments/    Experiment runners, report builders, validators, and analysis scripts
eval_results/   Generated run outputs, raw generations, ledgers, and score reports
meditations/    Question-first research notes for paradigm and doctrine work
paper/          Paper drafts and manuscript materials when present
src/            Shared package code used by the experiment stack
tests/          Unit and regression tests for runners, builders, validators, and controls
```

## Reproduce The Public Diffusion Evidence

```bash
python experiments/build_diffusion_claim_evidence.py
python experiments/validate_diffusion_claim_evidence.py
python experiments/validate_diffusion_theory_claim_ledger.py
```

For normal development:

```bash
python -m pytest tests -q
python -m compileall experiments src tests
```
