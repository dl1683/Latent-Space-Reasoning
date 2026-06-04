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

## Current Probe Evidence

The measured `counterfactual_micro_probe_v1` line has advanced through four
diagnostic instruments without clearing the spend gate:

- The legacy prose probe exposed value signal but produced malformed
  `FULL_REPAIR_AUTHORIZED=false` strings and weak slot fidelity.
- `strict_tomography_probe_v1` fixed the full sentinel but left only 8/12 rows
  valid for Stage 1 and missed three profitable invalid-positive rows.
- `key_value_tomography_probe_v2` removed placeholder exemplars but regressed
  raw diagnostic reliability to 6/12 valid rows and two malformed
  authorizations.
- `compact_tomography_probe_v3` restores 12/12 valid rows with zero malformed
  `Z=false` authorizations at a 48-token / 24-step probe budget, but the best
  validity-required Stage 1 rule still makes four false-positive spend
  decisions. This is a better measurement instrument, not a promoted spend
  controller.

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
`--repair-spend-trigger counterfactual_micro_probe_v1`. That trigger emits
bounded `counterfactual_probe` raw records for `would_probe=true` rows and
records the measured deltas in `repair_spend_gate_rows` while forcing
`should_run=false`, so it cannot promote full repair spend before the measured
probe policy clears the offline gate.

The first full named-counterexample measurement is
`DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_COUNTEREXAMPLES_V1.md`: seven measured
cheap probes, five skipped no-lift rows, zero triage errors, and zero full
repair authorizations. This is strong Stage 0 evidence, but still not a live
spend gate; the next promotion question is whether measured deltas support a
Stage 1 value-of-information rule that decides full repair after the probe.

The all-shadow measurement and fit in
`DIFFUSION_COUNTERFACTUAL_MEASURED_PROBE_VALUE_POLICY_V1.md` answers that Stage
1 question conservatively. It generates measured probe rows for all 12 named
counterexamples, including the five no-lift negatives. The best measured-only
post-probe rule still makes two false-positive errors. The only zero-error
all-feature rules are `prompt_gap_count_le_7` and `would_probe_score_ge_1`,
which are Stage 0 triage signals reappearing after the fact. Full repair spend
therefore remains blocked until measured probe features add a real post-probe
decision boundary.

The probe-text fidelity audit in
`DIFFUSION_COUNTERFACTUAL_PROBE_TEXT_FIDELITY_V1.md` makes the next design
constraint sharper: the cheap probe is currently too much like a short answer
and not enough like a stable diagnostic instrument. Four of 12 rows malformed
the explicit `FULL_REPAIR_AUTHORIZED=false` sentinel, five rows show punctuation
or spelling defects, and the best post-probe text rule still has two errors.
The next micro-probe should borrow the tomography discipline from the broader
Moonshots stack: perturb one constraint at a time, require stable slot fills,
score no-repair authorization as a hard validity check, and fit value only on
validated diagnostic rows.

The strict tomography policy in
`DIFFUSION_COUNTERFACTUAL_TOMOGRAPHY_PROBE_TEXT_FIDELITY_V1.md` is the first
measured response to that constraint. It adds
`--counterfactual-probe-policy strict_tomography_probe_v1`, raises the bounded
probe budget to `48` tokens / `24` denoise steps, and records slot-validity
fields in each gate row. On the same 12 named counterexamples, exact
`FULL_REPAIR_AUTHORIZED=false` improves to 12/12 with zero malformed
authorization rows. The gate still stays closed: only 8/12 rows pass strict
Stage 1 validity, and the best post-probe text rule still admits one no-lift
false positive.

`DIFFUSION_COUNTERFACTUAL_VALIDATED_PROBE_STAGE1_GATE_V1.md` then applies the
controller rule that invalid diagnostics are missing evidence, not weakly
usable evidence. With `valid_for_stage1` required before any measured value rule
can select a row, three profitable repairs (`plan_034`, `plan_046`,
`plan_070`) become invalid-positive misses with `0.185357` total missed lift.
The best validated Stage 1 rule still has five errors. This blocks full repair
spend more strongly than the raw text-fidelity result: the next probe must
first make profitable rows valid, then learn value.

The key-value v2 probe in
`DIFFUSION_COUNTERFACTUAL_KEY_VALUE_PROBE_TEXT_FIDELITY_V2.md` and
`DIFFUSION_COUNTERFACTUAL_KEY_VALUE_VALIDATED_PROBE_STAGE1_GATE_V2.md` tests
whether removing placeholder exemplars fixes that first bottleneck. It does not.
The validated value fit improves to two false negatives, but raw diagnostic
quality regresses to 6/12 valid rows, two malformed authorization strings, and
three generic slot rows. This is a controller-design lesson: a lower fit error
cannot compensate for a probe operator that is less reliable as an instrument.

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
