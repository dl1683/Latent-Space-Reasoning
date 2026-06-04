# Diffusion Reasoning Geometry Theory

This document states the current mathematical theory behind the diffusion
reasoning work. It is deliberately half formal and half empirical: the proofs
below are conditional on the abstractions and the current evidence ledger, not
claims that all language diffusion models now solve general reasoning.

## Objects

Let a prompt be `p`, a frozen diffusion language model be `M`, and a denoise
trajectory be:

`x_0, x_1, ..., x_T`

where `x_t` is the visible or partially visible text state after denoise step
`t`. A final answer is `x_T`, but the reasoning object is the whole trajectory.

For a task, define verifier features:

`phi(p, x_t) = [coverage, gap, contradiction, retention, source_quality, trajectory_score, ...]`

The exact coordinates differ by task family:

- Planning uses prompt coverage, missing constraint gaps, contradiction/risk
  penalties, source quality, and compact span targets.
- Exact tasks use answer parseability, arithmetic consistency, provenance,
  role/object/target checks, and symbolic proof traces.
- Diffusion repair uses first repairable denoise step, prompt-gap band,
  source quality, trajectory score, and source/final/history retention.

The current empirical score function is:

`S(p, x) in [0, 1]`

and the repair operator is:

`R(m, a, s): x_s -> x'_T`

where `m` is a mask/span policy, `a` is an anchor/control term, and `s` is a
source state such as final text, history text, or a phase-selected state.

## Information Loss

Define verifier information loss as:

`L_info(p, x_t) = d(phi_required(p), phi_observed(p, x_t)) + P_bad(p, x_t)`

where `d` measures missing required structure and `P_bad` penalizes
contradictions, irrelevant numbers, wrong target roles, unsupported proof
steps, or prompt-keyword loss.

This is not Shannon information in the raw-token sense. It is task-relevant
information: the part of the state needed to satisfy a verifier or judge.

The central claim is:

`reasoning improvement = controlled reduction of L_info along an editable trajectory`

not merely higher final-token likelihood.

## Proposition 1: Diffusion Exposes A Larger Intervention Set Than AR Decoding

For an autoregressive decoder, after token `y_i` is emitted, ordinary decoding
cannot directly edit `y_i` without restarting or using an external rewrite.
The available intervention set at future step `j > i` is restricted to suffix
coordinates:

`I_AR(j) subset {y_j, y_{j+1}, ...}`

For a diffusion decoder with remasking or inpainting, an intermediate state can
select a span or mask set `m_t` that includes earlier visible coordinates:

`I_diff(t) subset {positions selected by m_t}`

and `m_t` can overlap any unstable or verifier-relevant span. Therefore, when
`m_t` includes a wrong or missing verifier-critical region, diffusion has an
intervention available that standard left-to-right decoding does not.

Proof sketch: AR decoding has a causal commitment constraint: emitted prefix
tokens become conditioning context, not editable variables. Diffusion decoding
treats visible spans as a state over which masks can be reapplied. If a verifier
identifies a bad span before or after finalization, the diffusion repair
operator can make that span a variable again. Thus the reachable set of
post-diagnosis states is weakly larger for diffusion repair under the same
frozen model.

Evidence in this repo: the public latent repair arm improves the lean mixed
score from greedy/fixed `0.412277` and random `0.372125` to `0.531116` by
repairing denoise trajectories rather than selecting more final samples.

## Proposition 2: Useful Repair Is A Marginal Value Problem

For repair opportunity `i`, define:

`Delta_i = S(p_i, R_i(x)) - S(p_i, x)`

`c_i = extra relative GPU cost of the repair`

For a benchmark with `n` primary tasks and cost penalty `lambda`, the current
utility target is:

`U_i(lambda) = Delta_i / n - lambda * c_i / n`

A repair is cost-profitable when:

`U_i(lambda) > 0`

equivalently:

`lambda < Delta_i / c_i`

This gives a learned selector target: estimate `Delta_i / c_i` from
pre-repair trajectory features before spending another generation.

Proof sketch: total benchmark objective under a linear compute penalty is
additive across independent repair spends. The difference between spending and
not spending on task `i` is exactly the score lift minus cost penalty. Therefore
the optimal task-gated policy spends iff `U_i(lambda) > 0`.

Evidence in this repo: `DIFFUSION_BUDGET_POLICY_LOSS.md` implements this target.
At `lambda = 0.18`, the oracle task-gated selector keeps `plan_004`,
`plan_006`, and `plan_007`, scoring `0.508705` at `2.375000x`.

## Proposition 3: Phase Windows Create A Piecewise-Constant Cost Frontier

Let `tau_i` be the first denoise step where task `i` exposes a repairable
constraint skeleton. Let cap `k` permit repairs only when:

`tau_i <= k`

Then the selected repair set is:

`A(k) = {i : tau_i <= k and gate_i = true}`

The score and cost functions are piecewise constant in `k`, changing only at
observed onset steps `tau_i`.

Proof sketch: between two adjacent onset steps, no new repair opportunity
crosses the cap boundary. The active repair set is unchanged, so both aggregate
score and generation count are unchanged. When `k` reaches an onset step, one
or more repair opportunities enter the active set, changing score/cost by the
marginal contribution of those tasks.

Evidence in this repo: `DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md` derives four
validated regimes:

- cap `9`: no repair, `0.414598` at `2.000000x`
- cap `10-19`: three repairs, `0.472500` at `2.375000x`
- cap `20-30`: four repairs, `0.496607` at `2.500000x`
- cap `31+`: five repairs, `0.531116` at `2.625000x`

All fresh CUDA confirmations match the predicted score/cost map.

## Proposition 4: Current Repair Value Is Separable By Label-Free Geometry

For the current lean mixed planning targets at `lambda = 0.18`, define the
feature vector:

`g_i = [first_repairable_step_i, prompt_gap_i, source_quality_i, trajectory_score_i]`

The profitable repair set is:

`P = {plan_004, plan_006, plan_007}`

The runner-stable decision rule:

`first_repairable_step exists`

`source_needs_repair`

`prompt_gap_count <= 9`

`source_quality <= 0.31`

selects exactly `P` on the current targets.

Proof sketch: direct evaluation over the eight planning repair targets shows
zero false positives and zero false negatives. Inside the prompt-gap band,
profitable repairs have source quality at or below `0.301429`; the nearest
negative repairable in-band case starts at `0.324286`, leaving a separation gap
of `0.022857`. Low source quality alone is insufficient because `plan_005` is
outside the prompt-gap band at gap `10`, and `plan_008` lacks a repairable
denoise skeleton.

Evidence in this repo: `DIFFUSION_REPAIR_VALUE_GEOMETRY.md` records zero
regret for the runner proxy and the equivalent trajectory-score and
specific-or-late rules. The fresh CUDA confirmation is
`diffusion-a343e942cbfb0a93`: `0.508705` at `2.375000x`.

## Proposition 5: Judge Information Must Be Accounted For Separately

Let the complete inference system be:

`A = (M, G, V, J, C)`

where `M` is the frozen diffusion model, `G` is the generator/repair policy,
`V` is the verifier feature map, `J` is any judge or selector, and `C` is the
compute budget. A final selected output is:

`x* = select_J({G(M, p, z_k)}_k)`

The information used to choose `x*` is not only model information. It is:

`I_total = I_prompt + I_model + I_denoise_state + I_verifier + I_judge + I_anchor`

Therefore, any public claim that says "the model reasoned better" must specify
which of these channels was used and whether the channel is available at normal
inference time.

Proof sketch: if two candidate trajectories are generated and a selector
chooses one, the selector changes the output distribution even when the model
weights are frozen. That selection pressure is an information source. The same
is true for verifier rules and compact anchors. A claim that attributes all
gain to the model alone is under-specified unless it separately accounts for
these channels.

Evidence in this repo: exact-answer proposal ablations separate
proposal-attributable wins from diffusion inpainting wins, and the public
diffusion benchmark keeps the comparison to greedy/fixed, random perturbation,
and selected latent repair so the selection and repair machinery are visible.

## Proposition 6: Transfer Requires Feature Stability, Not Threshold Reuse

Let `g_i` be the label-free geometry vector used by a repair policy. A threshold
rule such as:

`source_quality <= 0.31`

is a current operating point, not a theorem. The transferable claim is weaker
and more useful:

`P(U_i(lambda) > 0 | g_i)` should be predictable from the same feature family
on new tasks.

Proof sketch: numeric thresholds depend on task mix, scoring scale, prompt
style, model family, and repair pack. If the feature family remains predictive
but the threshold shifts, the theory survives. If the feature family stops
separating profitable from wasteful repair opportunities, the current
mechanism claim fails to transfer.

Proof obligation: future benchmark slices should report whether
`first_repairable_step`, `prompt_gap_count`, `source_quality`,
`trajectory_score`, retention losses, and anchor-realization losses still
predict marginal repair value. Reusing `0.31` unchanged is not sufficient
evidence.

## Proposition 7: A Repair Operator Is Safe Only Under A Retention Constraint

Let `r = R(m, a, s)` be a repaired output. A repair is score-improving but
unsafe when:

`S(p, r) > S(p, s)` but `L_ret(p, s, r) > epsilon`

where `L_ret` measures loss of stable prompt terms, digits, target roles,
compact span obligations, or final/history length structure.

The safe repair objective is therefore constrained:

`maximize Delta S subject to L_ret <= epsilon`

or penalized:

`maximize Delta S - beta * L_ret`

Proof sketch: a repair can improve one visible dimension while destroying
another required constraint. If the benchmark scorer does not fully capture the
destroyed constraint, unconstrained repair can look locally useful while
degrading transfer. A retention term makes the hidden cost explicit.

Evidence in this repo: phase-source threshold sweeps show that weak history
replacement can regress score even when history looks repairable. Anchor
retention and realization-quality audits exist because compact controls can be
present in the prompt but fail to survive as useful final-answer content.

## Proposition 8: The Denoise Trajectory Is A Search Space With An Energy Budget

Let each intervention have cost `c`. The practical reasoning objective is not:

`maximize S`

It is:

`maximize S - lambda * C`

where `C` is total generation or repair cost. This makes diffusion reasoning an
energy-bounded search process over editable states.

Proof sketch: because each repair branch spends an additional generation, a
policy that repairs every possible weak state can improve raw score while
losing budget-normalized utility. If the objective includes cost, the optimal
policy must sometimes skip a real repair opportunity when its marginal value is
too low.

Evidence in this repo: repairing all phase-repairable states reaches the
top-score point, but the value-proxy policy reaches a lower-cost point by
skipping `plan_001` and `plan_003`, which are real but low-marginal repair
opportunities at `lambda = 0.18`.

## Where Information Can Be Gained

The model is frozen, so new task facts are not created inside the weights.
Useful information enters through controlled interfaces:

- Prompt information: task constraints, quantities, roles, and output format.
- Denoise-state information: partial skeletons reveal which constraints are
  present, missing, unstable, or overdiffuse.
- Verifier information: rule checks expose contradictions, missing roles,
  unsupported arithmetic, and proof gaps.
- Judge information: final or partial-state preference scores select which
  trajectory branch survives.
- Anchor information: compact semantic controls preserve obligations that the
  denoise process would otherwise drop.

The research risk is leakage: a judge or anchor can add information that is not
available at inference time. The repo therefore distinguishes label-free
pre-repair features from post-hoc task scores and records promoted claims in
the evidence ledger.

## Where Information Can Be Lost

Information loss appears as geometry failure:

- Undercoverage: prompt obligations disappear from the visible state.
- Overdiffusion: the state becomes verbose but misses compact constraints.
- Retention failure: a useful history or final-state constraint is damaged by
  repair.
- Role drift: the right number or object appears with the wrong task role.
- Anchor non-realization: a compact seed is present but not expressed in the
  final answer.
- Source mistrust: a denoise-history state looks repairable but is too weak to
  replace the final source.

The current theory treats these as measurable losses over `phi(p, x_t)`, not as
vague style differences.

## Candidate Error Functions

These are the loss families that should guide the next generation of the system.

The generated bridge from collected rows to loss design is
[DIFFUSION_ERROR_FUNCTION_GEOMETRY.md](../DIFFUSION_ERROR_FUNCTION_GEOMETRY.md).
It derives four current assertions from the repair-value and phase-source
target data:

- raw repair lift shrinks from five positive repair targets to three
  cost-profitable targets at lambda `0.18`
- earliest repairable denoise step `10` contains both profitable and
  unprofitable repair rows
- the current value proxy selects `plan_004`, `plan_006`, and `plan_007` with
  zero regret on the present targets
- naive repairable-history source trust creates four false positives, and even
  any-safe-history trust creates three

So the next system should optimize separate losses for repair value, source
trust, retention, and anchor realization rather than collapsing them into one
"repairable state" label.

The first direct selector comparison is
[DIFFUSION_DECOMPOSED_SELECTOR_AUDIT.md](../DIFFUSION_DECOMPOSED_SELECTOR_AUDIT.md).
On the current target rows, the one-label repairability controller has
composite shortfall `3.053730`, with three value false positives, four source
false positives, retention error `1.566063`, and realization error `0.573292`.
The decomposed controller has composite shortfall `0.186127`: zero value regret,
zero source error, zero retention error, and the remaining `0.186127` comes from
the active preservation-seed realization loss. It matches the target-label
oracle on value/source decisions: spend on `plan_004`, `plan_006`, and
`plan_007`; trust history source only on `plan_001`.

The first trainable target surface is
[DIFFUSION_COMPOSITE_SELECTOR_TARGETS.md](../DIFFUSION_COMPOSITE_SELECTOR_TARGETS.md).
It emits `diffusion_composite_selector_targets.jsonl` with eight task-level
rows for spend/source/retention heads and seven realization-policy rows for the
compact-anchor head. This is the handoff from proof-style geometry to a learned
controller.

The first fitted controller baseline is
[DIFFUSION_COMPOSITE_SELECTOR_FIT.md](../DIFFUSION_COMPOSITE_SELECTOR_FIT.md).
It fits zero-error local heads:

- spend: `first_repairable_step` exists, `prompt_gap_count <= 9`, and
  `source_quality <= 0.301429`
- source trust: `retention_safe_history`
- retention: `classification_safe_history_anchor`
- realization: `min_realization_policy_error`

This is the floor for learned selectors. A neural or regression-based
controller is only useful if it transfers better than this tiny rule fit.

The first local transfer check is
[DIFFUSION_SELECTOR_HOLDOUT_EVAL.md](../DIFFUSION_SELECTOR_HOLDOUT_EVAL.md).
It hides each task row, refits the spend/source/retention heads on the
remaining seven task rows, and tests the held-out labels. The decomposed heads
make `4` errors over `21` scored labels, while a single repairability controller
that predicts every head from `first_repairable_step` existence makes `12`
errors. This is not yet a fresh benchmark slice, but it upgrades the claim from
exact local fit to leave-one-task-out evidence that the decomposition captures
real geometry rather than only memorizing the full target surface.

The first runner-facing bridge is
[DIFFUSION_COMPOSITE_SELECTOR_RUNNER_POLICY.md](../DIFFUSION_COMPOSITE_SELECTOR_RUNNER_POLICY.md).
The benchmark CLI now accepts
`--repair-spend-trigger decomposed_four_head_selector`, which exposes the fitted
spend rule in live repair gating and writes the four head IDs into repair-spend
diagnostics. This matters because the theory now has an executable boundary:
fresh CUDA runs can test the same geometry-derived selector without translating
the loss family by hand.
Fresh run `diffusion-62476b492c9e592c` confirms that boundary: the trigger
repairs `plan_004`, `plan_006`, and `plan_007`, scores `0.508705` at
`2.375000x`, and records all four head IDs on every repair-spend gate row. It
does not beat the `0.531116` top-score frontier; it promotes the decomposed
selector as the lower-cost budget controller with explicit provenance.

The first independent spend-head transfer check is
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER.md).
It adds four new planning prompts, runs an all-repairable phase/final GPU pass,
and derives repair-value labels from repair-oracle lift. The boundary is
important: `plan_012` is a positive low-margin repair case even though the
selected repair arm can be held back by the promotion margin. A single
repairability trigger makes one false-positive spend on `plan_010`, while the
decomposed spend head skips `plan_010` and keeps `plan_012`, giving zero
repair-availability errors on the four-row transfer slice.

The expanded transfer check is
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V2.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V2.md).
It adds four more independent planning prompts. The result is stable but
narrow: `plan_012` remains the only positive repair-availability label, and
the decomposed spend head still has zero errors over eight independent planning
rows.

The fitted transfer rule report is
[DIFFUSION_SPEND_TRANSFER_RULE_FIT.md](../DIFFUSION_SPEND_TRANSFER_RULE_FIT.md).
After the oracle-lift correction, the best repair-availability rule is the
current decomposed spend rule, not the stricter `0.3075` source-task floor. The
mathematical assertion changed in a useful way: source quality and prompt-gap
geometry identify repairable low-quality states, while the promotion margin is
a separate cost-control layer. A source-task floor above `0.295357` becomes a
false negative because it skips the low-margin but positive `plan_012` repair.
The next proof question is how to learn two related functions: repair
availability and cost-adjusted promotion value.

[DIFFUSION_TRANSFER_PROMOTION_VALUE.md](../DIFFUSION_TRANSFER_PROMOTION_VALUE.md)
tests that split directly. On `lean_gpu_mixed_transfer_v2`, the corrected
spend gate runs only `plan_012`. The planning-quality seed-realization selector
generates the better repair but leaves `0.002500` mean repair headroom, while
the inherited planning-state selector selects it, raises repair-covered
planning score to `0.350938`, and reduces oracle headroom to `0.000000`.
This supports a two-head objective: one head predicts repair availability from
denoise/source geometry, and another predicts promotion value from the
post-repair state.

The runner now names the current promotion-value proxy as
`--repair-selector transfer_promotion_value`. It is an alias for inherited
planning-state repair selection, not yet a trained head. That distinction is
important for the theory: the repo has evidence that the promotion-value
surface is different from the repair-availability surface, but the current
implementation still uses an interpretable proxy until a learned selector is
trained.

[DIFFUSION_TRANSFER_HEAD_FIT.md](../DIFFUSION_TRANSFER_HEAD_FIT.md) turns that
split into the first fitted transfer-head artifact. The availability head
`availability_current_decomposed_spend` has `0` errors over 16 original plus
transfer rows. The promotion head `transfer_promotion_value` has `0` errors on
the expanded transfer policy rows, while the planning-quality promotion policy
has one false negative: the available `plan_012` repair.

[DIFFUSION_REASONING_PROOF_OBJECT.md](../DIFFUSION_REASONING_PROOF_OBJECT.md)
is the current canonical proof-object ledger. It ties six heads to target rows,
information channels, evidence files, falsifiers, and next GPU validations:
availability, promotion value, source trust, retention, realization, and cost.
This is the repo's concrete bridge from "geometry of reasoning space" to
testable error functions.

The first larger proof-object GPU slice is
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md).
It adds eight more planning prompts on top of the v2 transfer surface and
turns the availability head into a sharper geometric statement. Repair
availability is not only a property of the denoise skeleton and source-quality
band; it is relative to the already selected trajectory state. After the
repair-only label correction, v3 has two positive repair candidates
(`plan_018`, `plan_021`), single repairability makes `5` errors, the older
decomposed spend head makes `2`, and the trajectory-relative head makes `1`.
The historical executable CUDA policy run `diffusion-106f05c6dd5532ee` is still
useful execution evidence, but it also admitted stale positive `plan_012`; the
current label path treats that as availability evidence that must be checked by
post-repair promotion. The resulting spend-gate loss term is:

`L_avail = BCE(y_oracle_lift, h(denoise_phase, source_quality, prompt_gap, source_task - selected_task))`

This is a stricter information-accounting rule: a repair source can expose a
repairable denoise skeleton and still be the wrong source if another denoise
trajectory has already carried more task-relevant information forward.
[DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md](../DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md)
fits the first learned version of this head: `prompt_gap_count <= 8`,
`source_quality <= 0.256429`, and
`source_task_delta_vs_trajectory >= 0`. CUDA run
`diffusion-d0c2962992c50178` executes that learned trigger, selects the same
three repairs, and preserves `0.000000` oracle repair headroom.
The v4 transfer slice falsifies that fixed learned cutoff as a transfer-stable
availability model. `DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md` finds three
profitable repairs on `plan_026`, `plan_028`, and `plan_031`; all have
source-quality above the v3 learned ceiling, so
`learned_availability_predictor_v1` makes three false negatives. CUDA run
`diffusion-865e5acb0ee73e8a` confirms the executable trigger runs zero repairs
on that slice. The geometric lesson is that absolute source quality was a local
separator, not an invariant; the next availability loss should normalize source
quality against the slice, trajectory distribution, or task-local repairable
band.
The v5 slice then falsifies the idea that calibrated pre-repair availability is
enough. `calibrated_availability_predictor_v1` removes the absolute
source-quality ceiling, but the corrected repair-only labels show three v5
availability errors: it admits no-lift `plan_033` and `plan_038`, and drops
profitable `plan_037`, where source task is below the selected trajectory but
the repair candidate still improves over it. CUDA run `diffusion-c4f0d7bc21768f21`
still beats fixed and random, but leaves `0.011920` oracle headroom and trails
the all-repairable v5 run. `DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md` exposes
the missing term directly: `candidate_aware_promotion_v1` has zero errors over
the seven generated v5 repair candidates. The next loss therefore cannot be only
`L_avail(g_i)` over pre-repair geometry; it needs a candidate-aware term that
estimates realized repair promotion value after a repair candidate exists.

The v6 slice sharpens that conclusion. `DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md`
shows calibrated availability makes four spend errors: it admits no-lift
`plan_042` and `plan_045`, and misses positive repairs on `plan_046` and
`plan_048`. But `DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md` gives
`candidate_aware_promotion_v1` zero promotion errors over all eight generated
repair candidates. Rescoring the same raw generations with repairable-denoise
spend and candidate-aware promotion yields run `diffusion-ae7a4edd5c22ca20`:
`+0.086696` versus fixed, `+0.125929` versus random, `+0.068009` versus
trajectory, and `0.000000` oracle headroom. The theoretical update is that
promotion geometry is now the stable term; the unstable term is pre-repair spend
gating.

### Marginal Repair Value Loss

Target:

`y_i(lambda) = 1[Delta_i / c_i > lambda]`

Prediction:

`q_theta(g_i) ~= Delta_i / c_i`

Use this to decide whether a repair is worth another GPU generation.

### Constraint Retention Loss

For final source `f`, history source `h`, and repaired output `r`:

`L_ret = target_loss(p, r) + digit_loss(p, r) + keyword_loss(p, r) + length_drift(f, r)`

Use this to prevent a repair from destroying stable task constraints.

### Phase Source Trust Loss

Target whether history should replace final source:

`trust_history = 1[history_advantage > 0 and retention_safe(h, f)]`

This is why phase history should first be selector evidence, not automatically
the repair source.

### Anchor Realization Loss

Let anchor `a` encode a compact control obligation. Measure whether it becomes
directly expressed without leaking meta-instructions:

`L_anchor = missing_control(a, r) + meta_leak(a, r) + indirectness(a, r)`

Use this for compact semantic seeds.

### Energy-Aware Denoise Loss

Let `C(k)` be relative cost at phase cap `k`. Optimize:

`J = S(p, x_T) - lambda * C(k)`

with `k` or repair spend selected from denoise features. This prevents the
system from treating every repairable state as worth spending on.

### Verifier-Aligned Trajectory Loss

For verifier feature target `phi*`:

`L_traj = sum_t w_t d(phi*, phi(p, x_t))`

where later or more stable states can receive larger weights. This trains the
trajectory, not just the terminal string.

## Data-Derived Proof Workbench

The current mathematical work should be read as a proof workbench, not as a
finished theorem. The generated artifacts now let us state falsifiable
assertions about where information is lost or gained:

1. Repair availability is a geometric event: a denoise trajectory enters a
   region where a verifier-visible defect can be edited without discarding the
   stable task constraints.
2. Promotion value is a separate decision: a generated repair can be available
   yet not worth selecting unless its post-repair state improves the selector
   objective by enough margin.
3. Source trust is a retention-constrained decision: history states can contain
   useful information, but they are safe sources only when similarity,
   constraint retention, and source advantage agree.
4. Cost enters the geometry through phase windows: the same repair frontier is
   piecewise constant over denoise caps, so the right loss is marginal value per
   generation, not raw repairability.

The practical proof strategy is to turn each assertion into a target row:
availability labels from repair-oracle lift, promotion labels from selected
post-repair lift, retention labels from constraint preservation, and source
labels from final/history counterfactuals. A theorem-like claim should then
survive three checks: local fit, holdout over generated target rows, and a fresh
GPU slice without retuning thresholds.

## Current Theoretical Boundary

The strongest current assertion is:

Diffusion-native latent reasoning is useful when the system can diagnose a
verifier-relevant failure in an editable denoise state, estimate the marginal
value of repairing it, and apply a bounded repair that preserves stable
constraints.

The current work does not prove:

- broad benchmark domination
- a universal diffusion advantage for every task
- that source-quality thresholds will transfer unchanged
- that the learned controller exists without more training data

The next proof obligation is transfer: show that the same geometric variables
predict marginal repair value on new planning, exact-symbolic, and science
tasks without retuning thresholds per task.

## Claim Ledger

The theory claims above are tracked in
[DIFFUSION_THEORY_CLAIM_LEDGER.md](DIFFUSION_THEORY_CLAIM_LEDGER.md). Use that
ledger to distinguish `validated-local`, `supported-conditional`, `hypothesis`,
and `boundary` claims. A new theorem-like claim should not be treated as part
of the public theory until the ledger names its evidence, assumptions,
falsifier, and next proof obligation.
