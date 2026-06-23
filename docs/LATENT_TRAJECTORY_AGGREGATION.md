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

A post-hoc extractor diagnostic on the same frozen labels shows that the first
literal threshold was too conservative: a lower support threshold recovers the
frozen rubric components without false positives on this slice. That finding
must be frozen before reuse; it cannot retroactively promote the failed run.
When replayed post hoc at support threshold `0.1`, the mean realized aggregate
score rises above best-single, but the run still fails the frozen promotion
gate with only `1/16` online promotions because most recovered components do
not add measured component gain beyond the best single candidate.
A follow-up gain diagnostic shows the remaining bottleneck more sharply:
`12/16` tasks lift final score without positive component gain. This means the
next credible experiment is not just more sampling; it needs a complement-aware
selector that starts from the best single answer and explicitly targets
supported components absent from or weak in that answer.
A second score-dimension diagnostic sharpens that further: the current
component ontology only counts rubric items, but the scorer also rewards
causal diagnosis, specificity, constraint handling, and risk awareness. On the
threshold `0.1` replay, `12/16` score-lift/no-gain tasks improve through these
non-rubric dimensions, including `4/16` where the best single already has full
rubric coverage. The next family should therefore aggregate across multiple
latent aspect types, not only across rubric fragments.
That follow-up is frozen as multi-aspect v2 on held-out tasks
`plan_025..plan_048`, with aspect types for rubric items, causal diagnosis,
specificity, constraint handling, and risk awareness. Its promotion gates
require separate reporting of rubric gain and dimension gain so a score lift
cannot hide inside an underspecified component definition.
The first held-out v2 headroom diagnostic found modest candidate-level
complement material: `9/24` tasks have at least one complement aspect beyond
the best anchor, and `5/24` have dimension complements. The next replay should
therefore separate three failure modes: no complement material, selector found
complements but the realizer dropped them, and realized aggregate preserved
complements but still failed final scoring.
The first deterministic v2 replay preserved the no-complement control: tasks
without selected complements now return the anchor text unchanged. It locally
promotes `9/24` tasks and clears win-count, win-fraction, and Wilson gates, but
the full frozen gate still fails because mean non-rubric lift is `0.027083`,
below the predeclared `0.030000` threshold. This is evidence that multi-aspect
aggregation is viable but not yet strong enough to promote as a full result.
A post-replay failure analysis shows the miss is coverage-driven rather than
weak-complement-driven: the `9` complement tasks average `0.072222` non-rubric
lift and `0.087778` score lift, but `15/24` tasks have no complement material.
The next freeze should therefore improve complement discovery coverage or use
separate predeclared gates for coverage and conditional complement quality.
A coverage-gap diagnostic sharpens that prescription: `14/15` no-complement
tasks are anchor-dominance cases on the frozen aspect ontology, while only
`1/15` is a positive-but-below-threshold near miss. That means v3 should not be
primarily a threshold tweak; it should add targeted aspect-deficit probes or a
second candidate family conditioned on missing non-rubric dimensions.

V3 is now frozen as a pre-label task-extension contract over new `plan_201`-
`plan_224` rows. The freeze separates complement coverage from conditional
complement quality, requires probe cost and equal-budget best-of controls, and
keeps the targeted aspect-deficit probe as an explicit implementation boundary
if it is not available before labels.

The first v3 probe dry run is explicitly non-evidence: it used
`--limit-repair-candidates 0`, which skipped the runner's repair/probe gate and
therefore generated `0` counterfactual probes. The corrected probe command keeps
the diagnostic-only trigger but uses `--limit-repair-candidates 1` so the probe
source path is actually reached.

The corrected probe run generated `24` counterfactual probes. That validates the
probe measurement path, but not aggregation promotion: `23/24` probe texts were
stage-1 valid, every row kept `should_run=false`, and bounded probe text task
score averaged `-0.090247` below its source. V3 therefore has useful coverage
visibility, not yet useful fused-answer material.

The frozen v3 GPU label run is positive for the repair arm itself. It produced
`24/24` eligible repair coverage, selected latent repair task score `0.350000`,
task-score lift `+0.036042` over the selected trajectory/fixed baseline on
repair-covered tasks, and `10/13/1` wins/ties/losses against the evolved source.
This is repair-surface evidence, not yet a promoted aggregation result.

The deterministic frozen v3 aggregation replay now tests that question directly.
It keeps unsupported additions and hard contradictions at `0` under the template
audit and promotes every covered task locally (`6/6`), with conditional
non-rubric lift `0.079786`. The full v3 gate still fails because complement
coverage is only `6/24`, below the frozen `12/24` coverage gate, and the all-task
non-rubric lift is `0.019946` below the `0.030000` gate. The failure analysis
therefore preserves the same conclusion as v2 but more sharply: useful
multi-aspect fusion exists when complement material is found, but the next
experiment must increase complement discovery coverage rather than tune
thresholds or claim a promoted aggregation result from this slice.

Adding the corrected probe raw rows back into the replay as an extra latent
source improves coverage only from `6/24` to `7/24`. It also raises conditional
non-rubric lift to `0.090245` and reduces the aggregate-win/global-lift
shortfall to one covered task, but it still misses the frozen `12/24` coverage
gate. This means the current probe implementation is useful as a diagnostic and
occasionally supplies a real complement, but it is not yet the missing
high-coverage complement generator.

The v3 coverage-gap diagnostic rules out a simple threshold story. In the
baseline replay, all `18` no-complement tasks are anchor-dominance cases with
zero below-threshold near misses. In the probe-augmented replay, probes double
the mean non-anchor candidate count and add one covered task, but the remaining
`17` no-complement tasks are still anchor-dominance cases. The next credible
coverage experiment therefore needs either a complement-directed generation
policy that is conditioned on anchor deficits, or an expanded aspect ontology
that can expose useful differences invisible to the current rubric-plus-four-
dimension scorer projection.

A bounded GPU diversity-extension run tested the first option by adding LLaDA
evolved and revision schedules as extra raw sources, with no repair arm. When
those rows are replayed together with the original label rows and probe rows,
the augmented replay clears every numeric v3 gate: complement coverage rises to
`13/24`, local promotions to `13`, all-task non-rubric lift to `0.053741`, and
Wilson lower bound to `0.350749`. This is the strongest evidence so far for the
family-level thesis that multiple latent sources can compose into a richer
reasoning object. The boundary is important: because this diversity source was
added after the baseline v3 failure on the same task slice, it is diagnostic
design evidence for the next freeze, not the original predeclared v3 promotion.

V4 now tests that design cleanly on fresh tasks `plan_225` through `plan_248`.
It predeclared the baseline label run, probe run, LLaDA evolved/revision
diversity run, and combined replay before any v4 labels existed. The result
passes the frozen replay gates: complement coverage is `14/24`, all `14`
covered tasks locally promote, conditional non-rubric lift is `0.075352`,
all-task non-rubric lift is `0.043955`, Wilson lower bound is `0.388347`, and
unsupported additions and hard contradictions remain `0`.

This changes the aggregation status. The v3 diversity result was
hypothesis-generating because the diversity source was added after v3 failed.
The v4 replay is a fresh predeclared replication of that source mix on a new
planning slice. It is still bounded to this task family, aspect ontology, and
deterministic realization policy, but it is no longer merely post-failure
design evidence.

The remaining v4 coverage gap is explicit: `10/24` tasks still have no selected
complement material, and every no-complement task is an anchor-dominance case
under the current rubric-plus-dimension aspect ontology. The next experiment
should therefore move beyond adding more of the same sources. It should either
generate complements conditioned on anchor deficits, expand the aspect ontology
so useful differences become visible, or test whether another latent family
contributes aspects that LLaDA label/probe/diversity rows do not expose.

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
