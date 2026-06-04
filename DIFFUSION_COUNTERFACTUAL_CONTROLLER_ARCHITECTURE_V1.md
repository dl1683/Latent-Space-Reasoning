# Diffusion Counterfactual Controller Architecture V1

This is the next-controller contract after the v5-v9 spend failures. It is not
a promoted runner policy. It defines what has to be measured before another
live spend gate deserves GPU budget.

## Why This Exists

The accumulated evidence now has a stable split:

- Post-repair promotion transfers: `candidate_aware_promotion_v1` has zero
  errors on generated candidates in the v5-v9 target artifacts.
- Pre-repair spend does not transfer: simple rules, nonlinear source geometry,
  source-value signals, source degeneration, and pre-repair edge proxies all
  leave named false positives and false negatives.
- Frozen source features are exhausted as a standalone control surface:
  `DIFFUSION_PRE_REPAIR_EDGE_PROXY_V1.md` still has 12 errors for the best
  source+span proxy and explicitly keeps the gate at `do_not_promote`.

The architectural conclusion is that the controller needs new information, not
another threshold over the same frozen row features.

## Cross-Project Principles

This contract follows the AI Moonshots rule: intelligence should come from
better geometry and information allocation, not brute-force scale. The outside
research context points to the same discipline:

- Information geometry: useful measurements predict behavior early enough to
  change the intervention.
- Error correction: a serious system detects, diagnoses, repairs, and verifies;
  failure only counts as learning when it changes state.
- Search governance: search is useful only with memory, strict metrics,
  independent referees, and kill tests.
- Multi-scale intelligence: local proxy wins cannot be sold as integration
  wins; the controller has to report both local repair evidence and full-task
  cost/value.

For diffusion reasoning, that means every repair decision should be a value of
information decision.

## Controller Thesis

The next spend controller should not ask:

`Should I run the expensive repair candidate now?`

It should ask:

`What is the cheapest counterfactual observation that can reduce uncertainty
about repair value enough to justify or reject the expensive repair?`

The missing object is a counterfactual micro-probe:

`z_i = P_light(p_i, x_t, source_i, span_i)`

where `P_light` is a bounded, cheap observation that is not allowed to become a
full repair. It may be a short masked sketch, a small-span local completion, a
verifier-feature delta, or a compact plan skeleton generated under a strict
token and phase budget.

The controller then estimates:

`E[Delta_i | g_i, z_i] - lambda * C_repair - C_probe`

and spends on full repair only if the expected value remains positive after
probe cost.

## Three-Stage Policy

### Stage 0: Frozen Triage

Inputs:

- denoise phase diagnostics
- prompt gap and prompt coverage
- source quality and trajectory-relative source score
- source degeneration and meta leakage
- span target scores and source-relative preservation

Allowed output:

- `skip`
- `probe`

Forbidden output:

- `repair`

Rationale: the current evidence shows frozen features cannot safely promote a
live spend gate. They can only decide whether buying more information is worth
considering.

### Stage 1: Counterfactual Micro-Probe

The probe must be cheaper than full repair and must produce a verifier-facing
delta, not just more prose.

Minimum probe row:

| Field | Meaning |
| --- | --- |
| `task_id` | Planning task id. |
| `probe_policy` | Exact micro-probe operator. |
| `probe_cost_relative` | Extra generation fraction or token-normalized cost. |
| `pre_probe_features` | Frozen triage features used before probing. |
| `probe_text` | Bounded probe output or extracted sketch. |
| `probe_feature_delta` | Coverage, gap, retention, degeneration, and span deltas. |
| `probe_value_prediction` | Predicted repair lift after observing the probe. |
| `would_repair` | Probe-time spend decision. |

Hard rules:

- No post-repair candidate score can enter Stage 1 training features.
- Probe cost must be reported beside lift.
- Every probe must be replayable from raw artifacts.
- A probe that improves source text but misses task score is a failed proxy, not
  a hidden success.

### Stage 2: Full Repair and Candidate-Aware Promotion

Full repair remains the expensive action. Generated candidates are still judged
by `candidate_aware_promotion_v1` until a challenger beats its zero-error record
on generated-candidate rows.

The controller split is:

- spend head: `f(g_i, z_i) -> run full repair?`
- promotion head: `h(candidate_i) -> select generated candidate?`

Do not collapse these heads. The repo already has evidence that doing so hides
the real bottleneck.

## Training Surface

The next dataset should append probe rows to the existing v5-v9 target surface.

Required joined row:

| Family | Fields |
| --- | --- |
| frozen source | `source_quality`, `source_task_delta_vs_trajectory`, `prompt_gap_count`, source-value signals |
| degeneration | `degeneracy_score`, adjacent repeats, punctuation runs, meta leakage |
| span | gap term count, span target score, source-relative preservation |
| probe | probe cost, probe deltas, probe prediction, probe decision |
| repair | candidate lift, selected lift, repair selector edge |
| labels | profitable spend, promote vs trajectory, cost-adjusted utility |

The first useful model is not necessarily neural. A small monotone model,
calibrated tree, or Bayesian value-of-information rule is acceptable if it
reports uncertainty and named misses.

## Gate To Spend GPU

A live spend-gated GPU run is blocked until an offline challenger satisfies one
of these gates on accumulated target rows:

1. Preserve all profitable rows while removing at least five named no-lift rows.
2. Or explicitly trade away lift with a declared budget: missed positive lift
   must be lower than the saved repair cost under the selected `lambda`.
3. Report false positives and false negatives separately.
4. Keep `candidate_aware_promotion_v1` fixed unless the challenger beats its
   generated-candidate zero-error record.
5. Exclude post-candidate task scores and planning deltas from pre-repair
   features.

Anything weaker is diagnostic only.

## First Implementation Slice

The next code increment should be the smallest probe harness that can produce
real target rows:

1. Add a `counterfactual_micro_probe_v1` repair-spend mode that generates a
   bounded probe artifact but does not run full repair.
2. Run it only on the named counterexample rows:
   - false negatives: `plan_034`, `plan_044`, `plan_046`, `plan_061`,
     `plan_063`, `plan_070`, `plan_072`
   - false positives: `plan_045`, `plan_050`, `plan_064`, `plan_069`,
     `plan_071`
3. Build `diffusion_counterfactual_probe_targets_v1.json` with probe deltas,
   probe costs, and the existing repair/promotion labels.
4. Fit an offline value-of-information rule against those probe rows.
5. Keep the result offline unless it clears the GPU gate above.

The first replayable target sheet is now specified by
`DIFFUSION_COUNTERFACTUAL_PROBE_TARGETS_V1.md`. It is a deterministic scaffold,
not a measured GPU probe result: it names the rows, fields, labels, and cost
contract that a real `counterfactual_micro_probe_v1` run must replace.

The first offline fit over that scaffold is now specified by
`DIFFUSION_COUNTERFACTUAL_PROBE_POLICY_FIT_V1.md`. The best deterministic rule
keeps all seven profitable rows and removes four of five no-lift rows, but it
is still `diagnostic_only` because the decisive probe deltas are generated
predictions rather than measured cheap-probe observations.

The runner now exposes the diagnostic hook in
`DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_RUNNER_HOOK_V1.md` through
`--repair-spend-trigger counterfactual_micro_probe_v1`. That trigger records
probe fields in `repair_spend_gate_rows` while forcing `should_run=false`, so it
cannot promote full repair spend before measured probe deltas exist.

## Falsifiers

This architecture is wrong if any of the following happen:

- A frozen-feature-only controller beats the same gate without probe rows.
- Micro-probes cost nearly as much as full repair.
- Probe deltas predict source polish but not candidate lift.
- The probe policy introduces leakage from generated repair candidates.
- The best probe-aware rule still cannot reduce false positives without missing
  named positives.

## Current Decision

Do not spend another live GPU slice on a promoted spend gate yet. Spend the next
engineering increment on counterfactual probe rows. The goal is not to make the
current controller look better; it is to change the information geometry of the
decision.
