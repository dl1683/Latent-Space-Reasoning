# V11 Complement-Packet Aggregation Pre-Registration

## Scope

Fresh 96-task planning replication using task IDs plan_441..plan_536 from
experiments/latent_aggregation_v11_planning_tasks.jsonl. No v11 model
generations, packet rows, replay outputs, or labels exist before this
document is committed.

## Prior Result

v10 passed all 13 frozen gates on plan_393..plan_440 (48 tasks):
complement coverage 40/48, online promotions 38/48, W/T/L 40/8/0,
mean score lift +0.077, zero contradictions.

v11 tests whether this result replicates at 2x scale on fresh held-out tasks.

## Primary Hypothesis

H1: Complement-packet aggregation over fresh planning tasks produces online
promotions above a strong local-replication null.

- H0: p_online_promotion <= 0.70
- Success: online_promotions >= 76/96 and one-sided exact binomial p < 0.05.

## Secondary Hypotheses

H2: Complement coverage exceeds the v10 gate-level null.
- H0: p_complement_coverage <= 0.75
- Success: complement_coverage >= 80/96 and one-sided exact binomial p < 0.05.

H3: Mean realized score lift is positive and practically meaningful.
- Success: paired bootstrap 95% lower bound for mean score lift > +0.04.

H4: Complement packets add source-family value.
- Success: removing complement_packet raw rows drops coverage by >= 20
  percentage points, and paired discordant-task exact test p < 0.01.

H5: Gains are not keyword-stuffing artifacts.
- Success: keyword-stuffing audit is green or yellow; red demotes
  automatic-score claims.

## Model And Source Policy

- Candidate model: LLaDA-8B-Instruct only (single-model claim).
- Complement packets: 3 samples/task, max_new_tokens=128, steps=128,
  block_length=32, entropy algorithm, bfloat16.
- No threshold, ontology, realization, or audit changes after generation begins.

## Allowed Inputs

Task prompt, anchor text, generated source text, candidate key, schedule
name, generation stage, stable trajectory ID, source-run content hash,
predeclared aspect ontology.

## Forbidden Inputs

Replay labels, packet outcomes, task score deltas, selected complement
decisions, realized aggregate scores, v10 blocked-task decisions except
as public prior motivation.

## Primary Endpoints

- complement_coverage_count
- online_promoted_task_count
- online_promoted Wilson interval
- mean and median score lift
- bootstrap lower bound for mean score lift
- wins/ties/losses
- unsupported_addition_count
- hard_contradiction_count
- source-family ablation
- keyword-stuffing audit result

## Safety Gates

- Failure if unsupported additions > 0.
- Failure if hard contradictions > 0.
- Failure if losses > 0 for headline promotion; if losses occur, result is
  diagnostic only.

## Stopping Rules

- Run all 96 tasks. Do not stop early for success or failure.
- If generation crashes, resume with the identical command and record the
  interruption.
- If more than 5% of tasks have no usable raw LLaDA output due to
  infrastructure failure, mark run invalid and rerun from the frozen command.
- No task removal after generation begins.

## Statistical Tests

- One-sided exact binomial for online promotions vs p0=0.70.
- One-sided exact binomial for complement coverage vs p0=0.75.
- Paired bootstrap over task score lifts, 10,000 resamples.
- Paired source-family ablation discordance test.
- Leave-one-out mean lift range.
- High-leverage task share.

## Failure Interpretation

- Coverage failure: complement discovery is not robust; next direction is
  blocked-task recovery, not scale.
- Conditional-promotion failure: complements exist but selector/realizer/scorer
  is weak.
- Keyword-audit red: automatic planning rubric is not trustworthy as a primary
  endpoint; switch to manual/LLM judge.
- Safety failure: stop aggregation promotion claims and tighten
  provenance/contradiction verifier.
- Ablation failure: complement packets are not uniquely responsible; reframe
  as generic multi-source aggregation.

## Estimated Compute

- Label/source run (LLaDA-only): ~190 generations for 96 tasks at 30s/gen = ~1.6h.
  With repair candidates: ~670 generations = ~5.6h.
- Complement packets: 96 * 3 = 288 generations = ~2.4h.
- CPU replay/audit: minutes.
- Total: ~8h.

## Publication Path

- v11 pass: arXiv preprint + workshop submission.
- v11 fail: diagnose, do not publish.
