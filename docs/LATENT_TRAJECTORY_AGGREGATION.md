# Latent Trajectory Aggregation Doctrine

This document defines the next research frame after single-trajectory selection.
It is not a promoted result. It is the doctrine and experiment protocol for
testing whether multiple latent trajectories can be composed into a stronger
reasoning object than any individual trajectory.

## Core Thesis

Latent reasoning should not only select the best sampled trajectory.

The stronger hypothesis is:

`multi_trajectory_reasoning = expose diverse latent trajectories, extract their
non-overlapping useful structure, and synthesize a final answer that preserves
the best verified components from each trajectory`

Token perturbation, diffusion repair, denoise history, semantic anchors,
candidate promotion, verifier spans, and learned selectors are all members of
one family when treated as ways to expose or edit trajectory fragments.

## Why This Matters

Winner-take-all selection discards useful partial work.

Common failure pattern:

- Candidate A has the best high-level plan but misses a constraint.
- Candidate B catches the constraint but has weak structure.
- Candidate C finds an edge case.
- Candidate D preserves exact numbers or target roles.
- Candidate E has the strongest final answer but a thin explanation.

A selector can only choose one. An aggregator should preserve the useful parts.

The central research question becomes:

`Can aggregated latent trajectories produce an answer that beats the best
individual candidate because it composes non-overlapping useful content?`

## Unifying Intervention Surfaces

Each latent-control method exposes a different kind of useful trajectory
variation.

| Surface | Role In The Family | Main Risk |
| --- | --- | --- |
| Prefix/token perturbation | Early boundary-condition control; cheap trajectory diversity | Diversity without reliable selection or attribution |
| Attention/position perturbation | Routing-pressure control | Confounds with format or position effects |
| Diffusion denoise history | Mid-trajectory state exposure | History states can be repairable but not source-safe |
| Diffusion span repair | Local editable-state repair | Repair can destroy retained constraints |
| Semantic anchors | Compact constraint preservation | Anchor can leak meta text or overcompress meaning |
| Candidate promotion | Post-generation value estimation | Whole-candidate selection discards partial useful content |
| Component aggregation | Multi-trajectory synthesis | Fusion can introduce contradictions or unearned claims |

## Object Model

For a prompt `p`, let candidate trajectories be:

`tau_1, tau_2, ..., tau_k`

Each trajectory has intermediate states and a final output:

`tau_i = [x_i0, x_i1, ..., x_iT]`

Define component extraction:

`C(tau_i) = {c_ij}`

where each component can be a claim, constraint, calculation, subgoal, risk,
answer span, proof step, legal issue, code-fix hypothesis, or planning action.

Define component value:

`v(c_ij | p) = [support, correctness, novelty, constraint_retention, conflict_risk, cost]`

Define aggregation:

`A(p, {tau_i}) -> y`

where `y` is a synthesized answer whose component set should have higher
verified task value than the best individual candidate:

`S(p, y) > max_i S(p, x_iT)`

The proof target is not higher verbosity. The proof target is component-level
gain with contradiction control.

## Component Types

The aggregator should start with task families where component boundaries are
explicit.

- Planning: requirements, risks, root-cause hypotheses, mitigations, ordering,
  feasibility constraints, missing stakeholder concerns.
- Math and symbolic tasks: parsed quantities, equations, intermediate facts,
  final answer, provenance of each operation.
- Science or multiple-choice QA: answer choice, supporting fact, eliminated
  distractors, condition or exception.
- Legal or policy analysis: issues, rules, factual predicates, risks,
  counterarguments, missing facts.
- Code/debugging: failing behavior, suspected cause, patch idea, test evidence,
  regression risk.

## Aggregation Protocol V1

The first protocol should be deliberately simple and auditable.

1. Generate diverse trajectories.
   - Greedy/fixed baseline.
   - Random prefix/token perturbation.
   - Diffusion latent repair where available.
   - Optional temperature or self-consistency control.

2. Extract components.
   - Use deterministic parsers when possible.
   - Use rubric slots for planning tasks.
   - Use exact parsers for math/symbolic tasks.
   - Keep provenance: every component must cite the candidate and span it came
     from.

3. Score components.
   - Mark supported, contradicted, duplicate, missing, or unsupported.
   - Score component value separately from whole-answer value.
   - Penalize components that introduce new unsupported details.

4. Fuse components.
   - Keep the best supported non-duplicative components.
   - Resolve contradictions by verifier evidence, not by rhetorical confidence.
   - Preserve exact prompt constraints and target roles.

5. Realize the final answer.
   - Generate or template a final response from the fused component set.
   - Require the final answer to cite its component provenance internally in the
     artifact, even if the public answer hides it.

6. Validate against baselines.
   - Best single candidate.
   - Whole-candidate selector.
   - Majority/self-consistency.
   - Diffusion-only repair.
   - Prefix-only perturbation.
   - Aggregated multi-latent answer.

## Required Metrics

The aggregation claim only counts if these are reported separately:

- `best_single_score`: best whole candidate before aggregation.
- `aggregate_score`: final fused answer score.
- `component_gain`: rubric components present in aggregate but absent from the
  best single candidate.
- `component_loss`: useful components from the best single candidate lost during
  aggregation.
- `contradiction_count`: contradictions introduced by fusion.
- `unsupported_addition_count`: new claims not supported by any trajectory or
  prompt evidence.
- `source_diversity`: number of distinct trajectories contributing retained
  components.
- `cost`: total generation, repair, extraction, and fusion cost.

Minimum positive result:

`aggregate_score > best_single_score`

and

`component_gain > component_loss`

and

`contradiction_count = 0`

on a held-out or predeclared task slice.

## Statistical Discipline

The old token-perturbation work should re-enter only as a clean intervention
family, not as promoted historical proof.

Rules:

- Historical n=3 and small exploratory perturbation runs stay archived.
- New prefix/token perturbation experiments must use predeclared task manifests.
- Oracle coverage is not enough; aggregation must show selected or fused gains.
- Report means, variance, best-single, aggregate, and cost separately.
- Do not tune aggregation prompts or thresholds on the same slice used for the
  headline claim.
- Treat any first slice as `validated-local` at most.

## Failure Modes

Aggregation fails if:

- It merely writes a longer answer without increasing verified components.
- It imports false claims from low-quality candidates.
- It hides selection/judge information as if it came from the frozen model.
- It beats candidates only by using post-hoc labels unavailable at inference.
- It requires a human to identify all useful components manually.
- It cannot beat the best single candidate after contradiction penalties.

## First Experiment: Component Aggregation Scout

Use a small but predeclared scout before building a large system.

Current scaffold:

- Script: `experiments/analyze_latent_trajectory_aggregation.py`
- Fixture rows: `experiments/latent_trajectory_aggregation_scout_components.jsonl`
- JSON output: `eval_results/latent_trajectory_aggregation_scout.json`
- Report: `docs/reports/diffusion/LATENT_TRAJECTORY_AGGREGATION_SCOUT.md`
- Rubric replay script:
  `experiments/build_latent_aggregation_replay_from_rubric_hits.py`
- Rubric replay report:
  `docs/reports/diffusion/LATENT_AGGREGATION_RUBRIC_REPLAY.md`
- Inference validation freeze:
  `docs/reports/diffusion/LATENT_AGGREGATION_INFERENCE_V1_FREEZE.md`
- Inference replay harness:
  `experiments/run_latent_aggregation_inference_replay.py`
- Smoke replay report:
  `docs/reports/diffusion/LATENT_AGGREGATION_INFERENCE_SMOKE_REPLAY.md`
- Frozen GPU replay report:
  `docs/reports/diffusion/LATENT_AGGREGATION_INFERENCE_V1_REPLAY.md`

Status: protocol scaffold plus one negative frozen validation. The current
scout validates the accounting logic and gate behavior on deterministic
component fixtures. The rubric replay then applies the same accounting to
existing scored planning trajectories and finds post-hoc component-union
headroom. The frozen GPU replay is the first inference-time validation and
fails the predeclared promotion gates: `0/16` online promotions, low component
recall, and realized aggregate answers below the best single candidate on
average. It should be cited as a useful failure, not as a promoted aggregation
result.

The inference freeze is the next run contract. It fixes a 16-task planning
slice, trajectory families, extractor inputs, forbidden label fields, final
answer realization requirements, and statistical gates before labels exist.
The smoke replay is only a deterministic pipeline check: it proves the frozen
extractor/fuser/realizer accounting path runs end to end and keeps the
post-hoc scoring boundary intact, but it is not GPU evidence and cannot satisfy
the freeze gates.

The frozen GPU replay identifies the next implementation bottleneck: the
literal rubric-overlap extractor has high precision but misses most paraphrased
components, while the template realizer copies selected rubric labels rather
than synthesizing task-conditioned answers strong enough to beat the best LLaDA
repair candidate.

Task mix:

- 8 planning tasks with explicit rubric components.
- 8 math/symbolic tasks with exact answer and intermediate-operation checks.
- 4 science or multiple-choice tasks with answer plus support components.

Trajectory sources:

- Greedy/fixed baseline.
- 5 random prefix perturbation candidates.
- 1 diffusion repair candidate when the backend supports it.
- Optional temperature/self-consistency candidates as a control.

Aggregator:

- Extract components into JSONL rows.
- Select non-conflicting component winners by verifier score.
- Realize a final fused answer.
- Compare against best single candidate and whole-candidate selector.

Promotion condition:

- Aggregate beats best single candidate on score.
- Aggregate has positive component gain.
- Aggregate introduces zero hard contradictions.
- The result survives a second held-out slice without changing thresholds.

## Relationship To Current Diffusion Work

Diffusion repair remains the strongest current promoted result. Aggregation does
not replace it.

The next doctrine is broader:

- Diffusion repair edits one trajectory.
- Token perturbation exposes alternative starting basins.
- Candidate promotion chooses among generated branches.
- Aggregation composes useful verified fragments across branches.

The future system should route between all four:

`diverge -> attribute -> aggregate -> repair -> verify -> realize`

This is the path from latent trajectory control to latent trajectory
composition.
