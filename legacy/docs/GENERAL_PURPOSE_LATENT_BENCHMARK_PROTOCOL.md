# General-Purpose Latent Reasoning Benchmark Protocol

This is the next execution protocol after pausing ARC-AGI work. The goal is to
test whether latent-space interventions improve general reasoning under a
small, GPU-bounded benchmark stack.

The protocol is intentionally narrow. Do not add temperature sampling, prompt
paraphrase baselines, self-consistency, or large benchmark sweeps until this
three-arm test shows a real signal.

## Core Claim Under Test

Continuous latent interventions can improve reasoning trajectories under fixed
model weights, fixed greedy decoding, and fixed token budgets.

The claim is considered supported only if random perturbation or latent
reasoning improves one or more of:

- exact-answer accuracy on objective tasks
- rubric score on short open-ended planning tasks
- completion rate under the same token budget
- collapse/repetition avoidance
- baseline-failure rescue rate

## Conditions

Every task must run the same three conditions.

| Condition | Description | Count per task |
| --- | --- | ---: |
| `greedy_baseline` | Standard greedy decode, no soft prompt, temperature `0`. | 1 |
| `random_prefix` | RMS-matched random soft prefix, default `2` soft tokens, greedy decode. | 5 seeds |
| `latent_reasoning` | Current latent-space reasoning path: selected, evolved, or scorer-guided soft prefix, greedy decode. | 1 final output |

If `latent_reasoning` internally evaluates multiple candidate latents, the run
must report both the candidate count and the number of full model generations.
The public comparison should separate cheap scorer evaluations from expensive
autoregressive generations.

## Benchmark Pack

Start with the scout pack. Move to the pilot pack only after the harness,
scoring, and artifacts are clean.

### Scout Pack: 25 Tasks

| Family | Tasks | Purpose |
| --- | ---: | --- |
| Short open-ended planning | 8 | Main qualitative showcase and collapse-rescue test. |
| Math / GSM-style exact answer | 8 | Objective numeric sanity check. |
| Symbolic reasoning / BBH-style | 6 | Logic, tracking, temporal, and structured reasoning. |
| Science QA / GPQA-style | 3 | Hard multi-step factual reasoning with exact choices. |

Expected generation count: `25 * (1 + 5 + 1) = 175` full generations, plus any
reported latent-candidate scorer passes.

### Pilot Pack: 50 Tasks

| Family | Tasks | Purpose |
| --- | ---: | --- |
| Short open-ended planning | 10 | Primary public-facing evidence. |
| Math / GSM-style exact answer | 20 | Objective accuracy and failure-rescue measurement. |
| Symbolic reasoning / BBH-style | 15 | Cross-domain reasoning transfer. |
| Science QA / GPQA-style | 5 | Hard reasoning probe without making the run expensive. |

Expected generation count: `50 * (1 + 5 + 1) = 350` full generations, plus any
reported latent-candidate scorer passes.

## Planning Task Requirements

Planning tasks should be short but hard. They should expose failure modes where
greedy decoding gives generic, incomplete, repetitive, or prematurely collapsed
plans.

Each planning prompt should:

- fit in one short paragraph
- require causal diagnosis or multi-step tradeoffs
- have 4-6 expected rubric items
- avoid legal citations or external facts
- avoid tasks where a generic checklist receives a high score
- use the same max token budget as the other conditions

Rubric dimensions:

- `completion`: finishes a usable answer instead of collapsing or truncating
- `causal_diagnosis`: identifies the important mechanism or root cause
- `specificity`: gives concrete steps, checks, or decisions
- `constraint_handling`: respects the prompt's constraints and tradeoffs
- `risk_awareness`: names failure modes, validation, or rollback where relevant

Each dimension is scored `0-2`, for a `10` point task score.

## Objective Task Scoring

Math, symbolic, and science tasks should use deterministic scoring wherever
possible.

Required fields per task:

- `task_id`
- `family`
- `prompt`
- `answer`
- `answer_type`: `integer`, `short_text`, `multiple_choice`, or `rubric`
- `max_new_tokens`
- `scorer`

For exact-answer tasks, report:

- final-answer accuracy
- answer-anywhere accuracy when safe to compute
- extracted answer
- parse failure rate

For multiple-choice tasks, report:

- selected choice
- final choice accuracy
- parse failure rate

## Required Metrics

Every run report must include:

- mean score by condition and task family
- win/loss/tie counts versus `greedy_baseline`
- baseline failures fixed by `random_prefix`
- baseline failures fixed by `latent_reasoning`
- regressions introduced by each intervention
- completion rate
- collapse/repetition rate
- average generated tokens
- tokens per correct answer for objective tasks
- oracle score for `random_prefix` across its 5 seeds
- selector regret: random-prefix oracle minus `latent_reasoning`

## Evidence Thresholds

Scout pack success requires at least two of:

- planning: `latent_reasoning` wins at least `5 / 8` planning tasks
- objective: `random_prefix` or `latent_reasoning` improves exact accuracy by at least `5` percentage points
- rescue: interventions fix at least `25%` of greedy baseline failures
- collapse: interventions cut collapse/repetition failures by at least half
- selector: `latent_reasoning` captures at least `50%` of the random-prefix oracle lift

Pilot pack success requires the same pattern without relying on one family only.
If the improvement appears only in planning, the public claim must be framed as
planning/trajectory completion, not general reasoning.

## Artifacts To Produce

Use stable, append-only result paths:

- `experiments/general_reasoning_tasks_scout.jsonl`
- `experiments/general_reasoning_tasks_pilot.jsonl`
- `eval_results/general_reasoning/scout_raw.jsonl`
- `eval_results/general_reasoning/scout_scores.json`
- `eval_results/general_reasoning/scout_report.md`
- `eval_results/general_reasoning/pilot_raw.jsonl`
- `eval_results/general_reasoning/pilot_scores.json`
- `eval_results/general_reasoning/pilot_report.md`

Raw records must include the full generated text, not truncated previews.

## Non-Goals For This Pass

Do not add these until after the pilot:

- temperature baselines
- prompt paraphrase baselines
- self-consistency
- large benchmark sweeps
- broad model-family comparisons
- legal-reasoning judge-heavy tasks
- ARC-AGI or game environments

Those can become follow-up controls only if the three-arm GPU-bounded protocol
produces a real effect.

## Immediate Implementation Order

1. Create the scout task manifest.
2. Add a three-arm runner that writes canonical raw JSONL.
3. Add deterministic scorers for exact-answer and multiple-choice tasks.
4. Add the planning rubric scorer and keep the rubric fixed before running.
5. Run the 25-task scout.
6. Generate `scout_report.md` with the required metrics.
7. Decide whether the 50-task pilot is justified.
