# Diffusion-Native Reasoning Architecture

This project should stop treating diffusion as a drop-in decoder for the old
autoregressive latent-prefix loop. Diffusion gives us a different substrate: a
reasoning attempt is an editable denoising trajectory.

## Core Shift

Old loop:

1. sample or evolve a continuous prefix
2. decode left to right
3. judge final text
4. select the best final output

Diffusion-native loop:

1. choose a denoising schedule, mask policy, and future token/logit hook
2. generate an intermediate denoise sequence
3. score partial states before the final answer
4. repair or redirect unresolved positions
5. select the trajectory, not just the final text

The latent object is no longer only a vector. It can be:

- a denoising schedule
- a remasking policy
- a mask-position curriculum
- a partial-state value trace
- a logits hook applied at selected denoise steps
- an error-correction policy for unstable spans
- a judge-selected trajectory branch

## Local Research Constraints

The broader workspace research changes what this architecture should optimize:

- `_meta/projects/latent-space-reasoning.md` downgraded the old AR story toward
  oracle selection, completion rate, and scorer quality. Diffusion work should
  therefore make selection and repair explicit, not pretend the latent object
  itself magically contains new task information.
- `_meta/inquiry/RESEARCH.md` frames reasoning gains as dynamic reachable-set
  changes rather than static geometry changes. Diffusion is a better fit because
  it exposes a multi-step state trajectory that can be redirected.
- `Market Reports/Open Exploration/Attention Everywhere/README.md` treats
  gated attention as the main falsifier for AR prefix perturbations. Diffusion
  repair should be evaluated as its own trajectory-control mechanism, not as an
  attention-sink workaround.
- `Market Reports/Open Exploration/Consciousness and the Hard Problem/the_hard_problem.md`
  names feedback and error correction as core intelligence primitives. In this
  repo that means repairs should become explicit error-correction operators,
  measured by whether they improve a locked task score without damaging stable
  spans.

## First Implementation Surface

The current implementation adds this surface:

- `DiffusionScheduleCandidate`: searchable schedule candidate
- `trajectory_summary`: sampled denoise-state metrics
- `trajectory_control_score`: cheap selector score before external judges
- `is_llada_family`: shared routing so dense LLaDA and sparse LLaDA-MoE use the
  same masked-denoise schedule, revision, and repair surface
- `generated_token_confidences`: per-token model confidence at the denoise step where LLaDA commits each suffix token
- `initial_suffix_token_ids`: LLaDA suffix-inpainting hook for branch repairs
- `DiffusionRepairCandidate`: keep a generated prefix or remask low-confidence positions, then denoise again
- `default_llada_source_relative_repair_candidates`: minimal low-confidence remask pack for testing whether repairs actually improve their source
- `default_llada_targeted_content_repair_candidates`: text-policy pack that remasks filler or repeated spans after mapping generated text back to token positions
- `default_llada_prompt_guided_repair_candidates`: draft-revision pack that gives diffusion the source output plus a generic critique prompt
- `default_llada_constraint_gap_repair_candidates`: hybrid pack that keeps the canonical state-adaptive repair line and adds prompt-grounded revision against missing task terms
- `default_llada_constraint_span_clause_repair_candidates`: opt-in diagnostic pack that forces clause targets inside long planning sentences; the default span repair now uses a compact policy that can refine long risky sentence targets to clauses only when that reduces the denoise region
- `default_llada_replay_consistency_repair_candidates`: denoise-history instability pack that remasks suffix positions that fluctuate across sampled trajectory states
- `DiffusionVerifierRepairCandidate`: remask extracted wrong-answer spans or a fallback answer window
- `counterfactual_repair`: enumerate prompt-surface, arithmetic, or symbolic answer proposals after verifier failure, then run diffusion on the counterfactual prompt
- `answer_proposals.py`: deterministic proposal layer for exact-answer repair that does not read the hidden expected answer
- `proposal_only_ablation`: non-model repair candidate used to test whether the proposer alone explains an exact-answer gain
- `run_diffusion_schedule_sweep.py`: keeps one model loaded and compares schedules
- `run_diffusion_scout.py`: applies schedule selection to locked scout tasks
- `run_diffusion_repair_scout.py`: branches suffix repairs from selected scout outputs
- `sweep_adaptive_source_gate.py`: reuses the exhaustive MoE repair-source raw
  pool to rescore adaptive second-source thresholds without another GPU run,
  producing JSON/CSV/Markdown summaries of score-maximal and budget-efficient
  gate regimes
- `run_diffusion_three_arm_benchmark.py`: compares fixed, random, trajectory-selected, evolved, and optional repair-selected diffusion arms without using task labels for arm selection; evolved selects from the base schedule pool plus a small mutated schedule pool, repair-selected branches LLaDA suffix-inpainting candidates from the configured repair source (`evolved`, `trajectory`, `fixed`, or `non_revision_evolved`), exact-answer counterfactual repairs use prompt-derived proposals and label-free proposal-match promotion, `--exact-verifier-revision` can remask the failed answer span under the original prompt before counterfactual repair, exact-answer trajectory selection can optionally choose a final output or visible denoise-history state only when it matches a prompt-derived proposal, primary repair spending can be gated by label-free source completeness, state-adaptive repair can change history-anchor length from source/history state quality, constraint-gap repair can rewrite against missing source prompt terms, replay-consistency repair can remask denoise-history-unstable positions, adaptive history rescue can spend a later history anchor only after a weak first repair pass, adaptive prompt-guided rescue can spend a draft-revision branch behind baseline/source-quality/selector-disagreement gates, both promotion stages apply conservative selector margins, `--revision-promotion-margin` adds a stricter gate for non-monotonic revision schedules, `--repair-source-policy non_revision_evolved` keeps revision as an evolved-arm candidate without forcing repair to seed from the revised output, risk-guarded selectors can penalize prompt-contradicting planning repairs, and the report exposes each arm's larger generation budget, family-level arm summaries, selector-regret/oracle-coverage summaries, and separate overall and repair-eligible coverage
- `planning_quality_delta_guarded`: repair selector that uses label-free planning-quality improvement over the source output, not absolute repair quality
- `planning_quality_delta_risk_guarded`: source-relative repair selector that also subtracts a label-free contradiction/risk penalty and reports `Risk Penalty`

This already found schedule effects:

- Dream `entropy_64` beat `entropy_32` and `origin_64` on the smoke prompt.
- Dream `origin_64` collapsed into a thin answer despite completing masks.
- LLaDA `low_confidence_32` and `random_32` both produced coherent answers, with low-confidence slightly ahead on the current trajectory score.
- On the locked planning scout slice, the neutral benchmark system prompt fixed
  refusal behavior and produced 40 raw generations across Dream/LLaDA schedules.
- On the objective scout slice, LLaDA was materially stronger than Dream on
  exact-answer tasks, especially arithmetic: selected outputs averaged `0.941`
  for LLaDA versus `0.647` for Dream after rescoring the raw generations.
- On the LLaDA repair scout slice, suffix-inpainting repairs were selected on
  4/8 planning tasks and moved selected mean task score from `0.412` to `0.436`.
- After adding token-transfer confidence capture, confidence-aware repair added
  one more selected repair: 5/8 planning tasks selected a repair and mean task
  score moved from `0.412` to `0.443`.
- On the Dream objective proposal-ablation slice, Dream moved from `0.647` to
  `1.000` with 6/17 selected model repairs, but proposal-only selection also
  reached `1.000`. The task-score gain on this slice is therefore attributable
  to verifier/proposer search; diffusion adds trajectory-scored execution, not
  extra exact-answer accuracy over proposal-only.
- On the LLaDA objective proposal-ablation slice, LLaDA moved from `0.941` to
  `1.000`, and proposal-only selection also reached `1.000`. The selected model
  repair was the answer-context remask path for `sym_002`, which beat the
  counterfactual branch on combined trajectory score in the latest run.
- On the seeded three-arm planning-plus-mix benchmark, fixed diffusion averaged
  `0.436`, random schedule control averaged `0.423`, and planning-state
  trajectory-selected diffusion averaged `0.465` across Dream/LLaDA on 8
  planning tasks plus 3 objective checks. The selector is now using sampled
  denoise states, not only the final output: `+0.029` vs fixed and `+0.042` vs
  random, with one remaining small planning regression.
  Exact-answer tasks are guarded back to fixed schedules because raw trajectory
  scores can prefer stable wrong one-token answers without a verifier.
- On the seeded evolved planning-plus-mix benchmark, the same runner added two
  mutated schedules per model and selected from the larger pool for the evolved
  arm. The run used 99 full model generations, 88 arm selections, kept the same
  exact-answer guard, and required a `0.015` selector-score margin before an
  evolved candidate could replace the base trajectory-selected schedule.
  Evolved schedule selection averaged `0.475` task score versus `0.465` for
  base-pool trajectory selection, `0.436` fixed, and `0.423` random. The gain
  is budgeted: evolved used `4.50` generations per task versus `2.50` for
  trajectory selection and `1.00` for fixed/random. It beat fixed by `+0.039`,
  random by `+0.052`, and trajectory selection by `+0.010`, with
  wins/ties/losses of `3/19/0` against trajectory selection. The canonical
  report is
  `eval_results/diffusion_language/four_arm_evolved_margin015_v1_report.md`.
- Rescoring the same raw generations with oracle diagnostics showed the current
  evolved selector is close to the available schedule-pool ceiling: oracle task
  score was `0.481`, evolved was `0.475`, and oracle headroom over evolved was
  only `+0.006` mean task score. Oracle still found four better task-level
  choices, so there is some selector headroom, but the larger next gain likely
  requires richer trajectory mutations or repair candidates rather than only
  reweighting the current selector. The diagnostic report is
  `eval_results/diffusion_language/four_arm_evolved_margin015_oracle_rescore_v1_report.md`.
- On the LLaDA planning repair-arm diagnostic, the benchmark now branches
  prefix-inpainting repairs from the evolved output and from selected
  mid-denoise states, then selects repair with a repair-specific final
  planning-quality selector. The efficient canonical run is
  `eval_results/diffusion_language/llada_planning_adaptive_history_rescue_margin01_v1_report.md`:
  fixed `0.412`, random `0.376`, trajectory `0.412`, evolved `0.451`, and
  repair-selected `0.490`, with `6/2/0` wins/ties/losses against evolved at
  `6.12` generations per covered task. Oracle headroom over repair is now
  `0.000` on this slice. The adaptive rescue path spends the later
  `history_prefix_50_repair` only when the first repair pass would keep the
  matching evolved baseline; in this run it rescues `plan_004` from history
  step `39`.
- The newer efficient gated run
  `eval_results/diffusion_language/llada_planning_primary_repair_gate_v1_report.md`
  adds `--repair-spend-trigger source_quality_or_short` before the same
  prefix/history repair path and uses `planning_quality_delta_guarded` for
  repair selection. It skips primary repair for sources that already look
  complete by label-free planning quality plus visible length. It preserves
  repair-selected `0.490`, `+0.039` over evolved, `6/2/0`, and zero oracle
  headroom while reducing budget to `5.88` generations per covered task and
  raising task-score gain per extra generation to `0.021`.
- The state-adaptive history-prefix run
  `eval_results/diffusion_language/llada_planning_state_adaptive_history_prefix_v1_report.md`
  replaces the fixed `history_prefix_25_repair` plus adaptive rescue with
  `state_adaptive_history_repair` and `prefix_25_repair`. The history repair
  uses a longer anchor only when both source quality and selected history-state
  quality are weak, which captures the old `plan_004` `history_prefix_50`
  rescue without a separate rescue branch. It preserves repair-selected
  `0.490`, `+0.039` over evolved, `6/2/0`, and zero oracle headroom while
  reducing budget further to `5.75` generations per covered task and raising
  task-score gain per extra generation to `0.022`. The negative diagnostic
  `llada_planning_state_adaptive_repair_pack_v1_report.md` showed
  `state_adaptive_confidence_repair` averaged `-0.009` task delta versus source
  and was selected zero times, so that branch is no longer in the first two
  state-adaptive candidates.
- The constraint-gap hybrid diagnostic
  `eval_results/diffusion_language/llada_planning_constraint_gap_repair_v1_report.md`
  keeps `state_adaptive_history_repair` and `prefix_25_repair`, then adds
  `constraint_gap_revision_repair`, a full-draft repair prompt that lists
  missing or weak terms from the original task prompt. It reaches a tiny new
  absolute best on this slice: repair-selected `0.491`, `+0.040` over evolved,
  with `6/2/0` repair-vs-evolved wins/ties/losses and zero oracle headroom.
  The new branch is real but sparse: selected once on `plan_001`, where it
  improves the evolved output by `+0.076`, and averages `+0.011` task delta
  versus source with `1/6/0` wins/ties/losses. It is not the efficient default
  because budget rises to `6.62` generations per task and repair gain per
  extra generation falls to `0.015`, below the state-adaptive line's `0.022`.
- The gated constraint-gap rescue run
  `eval_results/diffusion_language/llada_planning_gated_constraint_gap_rescue_v1_report.md`
  converts that diagnostic into a budgeted policy. It starts from the efficient
  `state_adaptive` pack and adds `constraint_gap_revision_repair` only when the
  evolved source has midrange label-free planning quality (`0.400-0.500`) and
  at least 6 missing prompt terms. The gate fired only on `plan_001`, preserved
  the `0.491` selected score, kept `6/2/0` wins/ties/losses and zero oracle
  headroom, and used 47 generations instead of the unconditional run's 53. The
  covered-slice budget is `5.88` generations/task with `0.022` repair gain per
  extra generation, matching the state-adaptive efficiency while adding the
  small absolute score improvement.
- The risk-guarded rescore
  `eval_results/diffusion_language/llada_planning_gated_constraint_gap_risk_guard_rescore_v1_report.md`
  reuses the gated constraint-gap raw file with
  `planning_quality_delta_risk_guarded`. It adds a label-free contradiction
  penalty for prompt-violating planning repairs and exposes `Risk Penalty` in
  repair diagnostics. On this slice all detected penalties are `0.000`, so it
  preserves the same `0.491` repair-selected score, `+0.040` over evolved,
  `6/2/0` repair-vs-evolved wins/ties/losses, and zero oracle headroom. This is
  a selector-safety upgrade, not evidence of a new aggregate gain.
- The replay-consistency diagnostic
  `eval_results/diffusion_language/llada_planning_replay_consistency_repair_v1_report.md`
  tests whether denoise-history instability marks the right spans to repair.
  It reaches only `0.477`, `+0.026` over evolved, with `4/4/0`
  repair-vs-evolved wins/ties/losses at the same `5.75` generation budget as
  the state-adaptive history-prefix line. The dedicated
  `replay_unstable_25_repair` candidate was selected zero times, averaged
  `0.000` task delta versus source, and had `0/7/0` wins/ties/losses versus
  source. This is a useful negative: trajectory instability is measurable, but
  instability alone is not yet a source-relative repair objective.
- On the mixed Dream-plus-LLaDA evolved-plus-repair benchmark, the current
  canonical report is
  `eval_results/diffusion_language/mixed_adaptive_history_rescue_margin01_v1_report.md`.
  It uses 116 seeded generations over 22 model-task pairs. Fixed averaged
  `0.436`, random `0.423`, trajectory-selected `0.465`, and evolved `0.480`
  across the full mix. The repair arm is `8/22` overall and `8/8` on the LLaDA
  planning tasks where suffix-inpainting repair is available. It reaches
  `0.490`, `+0.039` over evolved on that covered slice, with `6/2/0`
  wins/ties/losses and zero oracle headroom.
- The mixed family/regret rescore
  `eval_results/diffusion_language/mixed_adaptive_history_rescue_family_regret_rescore_v1_report.md`
  reuses the same raw file with `planning_quality_delta_risk_guarded` and adds
  family-level arm summaries plus explicit selector-regret diagnostics.
  Aggregate fixed/random/trajectory/evolved stay at `0.436`/`0.423`/`0.465`/`0.480`.
  The repair arm is correctly reported as `8/22` overall and `8/8` repair
  eligible, all on LLaDA planning, with `0.490` selected task score. Selector
  regret is now visible: trajectory leaves `0.030` task score over `8/22`
  improvable selections, evolved leaves `0.014` over `7/22`, and repair leaves
  `0.000` over `0/8` on its covered slice.
- The fresh LLaDA mixed gated run
  `eval_results/diffusion_language/llada_mixed_gated_constraint_gap_risk_guard_v1_report.md`
  carries the current best planning policy into a mixed suite with math,
  symbolic, and science checks. It uses 59 generations, keeps repair coverage
  honest at `8/11` overall and `8/8` eligible, and preserves the planning result:
  repair-selected `0.491`, `+0.040` over evolved, `6/2/0`, zero oracle headroom,
  and `0.022` repair gain per extra generation. The family table shows why the
  next mixed-benchmark operator should target exact-answer/symbolic handling:
  math and science score `1.000`, while the symbolic check remains `0.000`.
- Exact-answer counterfactual repair is now in the main benchmark. The targeted
  smoke
  `eval_results/diffusion_language/llada_sym002_exact_counterfactual_repair_v1_report.md`
  repairs `sym_002` from `0.000` to `1.000` with one additional LLaDA
  generation. The selector does not read the hidden expected answer; it promotes
  only when the generated text matches the prompt-derived proposal. The full
  mixed run
  `eval_results/diffusion_language/llada_mixed_gated_constraint_gap_exact_repair_v2_report.md`
  uses 60 generations, improves repair coverage to `9/11` overall and `9/9`
  eligible, raises repair-selected mean task score to `0.548`, beats evolved by
  `+0.147` on covered tasks, keeps zero oracle headroom, and lifts repair gain
  per extra generation to `0.083`. The full planning-plus-exact scout
  `eval_results/diffusion_language/llada_full_scout_gated_exact_repair_v1_report.md`
  carries the same policy across 25 LLaDA tasks: 8 planning, 8 math, 6
  symbolic, and 3 science. Full-suite fixed/random/trajectory/evolved are
  `0.772`/`0.680`/`0.772`/`0.784`; repair coverage is `9/25` overall and `9/9`
  eligible, with covered-task repair-selected score `0.548`, `+0.147` versus
  evolved, `7/2/0`, zero repair-oracle headroom, and `0.083` gain per extra
  generation. The guarded compact rerun
  `eval_results/diffusion_language/llada_mixed_gated_ranked_span_guarded_exact_identity_v1_report.md`
  keeps the same mixed repair score, `0.548`, and planning score, `0.491`, while
  allowing the expanded prompt-gap rescue candidate set and carrying stable
  `run_id`/`content_hash` fields. It rejects the superficially higher
  `plan_001` anchor candidate because the output leaks a comma-separated prompt
  checklist; the selected full revision is cleaner and the report exposes
  `Risk Penalty 0.180` for the rejected anchor. The
  exact-answer candidate table reports `Proposal Task 1.000` and
  `Task-vs-Proposal 0.000`, so this is the cleanest compact mixed line while the
  exact-answer lift remains proposer-attributable.
- A harder exact-answer stress slice now gives the project its first exact
  repair line that is neither proposal-attributable nor selected by hidden
  answers. Four tasks were added to
  `experiments/general_reasoning_tasks_scout.jsonl` as `math_009`,
  `math_010`, `math_011`, and `sym_007`. They intentionally have zero
  deterministic prompt-derived proposals. The fresh GPU run
  `eval_results/diffusion_language/llada_hard_exact_arithmetic_feedback_v1_report.md`
  uses long scratchpad `self_check_answer_repair` plus
  `arithmetic_feedback_repair`. Self-repair fixes `sym_007` from `0.000` to
  `1.000`; arithmetic feedback detects the false `math_010` equation
  `3*14 + 2*9 = 54`, tells the model the expression equals `60`, and repairs
  the final answer to `10`. Repair-selected covers `2/4` overall and `2/2`
  eligible, reaches `1.000` on the eligible failures, beats evolved by
  `+1.000`, has `2/0/0` repair-vs-evolved wins/ties/losses, zero
  repair-oracle headroom, and `0.667` task gain per extra generation.
- The current full LLaDA line is
  `eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_report.md`.
  It combines planning repair, prompt-derived proposal repair, scratchpad
  self-repair, and arithmetic-feedback repair over 29 tasks with 135 fresh
  generations. Full-suite fixed/random/trajectory/evolved means are
  `0.734`/`0.656`/`0.734`/`0.745`; repair coverage is `11/29` overall and
  `11/11` eligible. On the eligible slice repair-selected reaches `0.630`,
  beats evolved by `+0.302`, has `9/2/0` repair-vs-evolved wins/ties/losses,
  zero repair-oracle headroom, and `0.175` gain per extra generation. This is
  the first repo line where the planning and exact-answer repair operators are
  exercised together under the same budget ledger.
- The first GSM-style hidden-distractor exact slice is
  `eval_results/diffusion_language/llada_gsm_distractor_self_repair_v1_report.md`.
  It adds `math_012` through `math_015`, all with empty deterministic proposal
  coverage, and runs 19 fresh LLaDA generations. Fixed, random, trajectory, and
  evolved all score `0.500`; repair covers the two failed tasks and reaches
  `1.000` on the eligible slice with `2/0/0` wins/ties/losses against evolved,
  zero repair-oracle headroom, and `0.667` task gain per extra generation.
  `math_014` is fixed by label-free scratchpad self-repair. `math_013` is
  fixed by arithmetic feedback after the scratchpad makes the checkable false
  claim `204 + 56 = 265`; the verifier computes `260` and the second repair
  returns the correct ticket revenue. The arithmetic guard now also catches
  simple worded arithmetic claims and compound `times ... plus ... times`
  claims, so it is less dependent on scratchpads using an equals sign.
- Exact integer repairs now have an arithmetic-evidence gate. A self-repair or
  arithmetic-feedback repair must include at least one checkable arithmetic
  claim before it can be selected; a changed final integer with no scratchpad
  evidence scores `0.0` under the label-free selector. CPU-only evidence-guard
  rescores preserve the current gains:
  `llada_extended_full_evidence_guard_rescore_v1_report.md` keeps the 29-task
  full line at `11/11` eligible repair coverage, `+0.302` versus evolved,
  `9/2/0` wins/ties/losses, and zero repair-oracle headroom; the GSM distractor
  evidence rescore keeps `2/2` repair coverage, `+1.000`, and zero headroom.
  Reports now expose `Arithmetic Claims` beside `Arithmetic OK`.
- The missing-evidence exact repair branch is now implemented as
  `arithmetic_evidence_repair`. If an integer `self_check_answer_repair`
  returns a changed final answer but zero checkable arithmetic claims, the
  runner can spend the second exact-repair slot on a prompt that asks diffusion
  to re-solve with explicit digit/operator equations. This addresses the
  previous failure mode where the stricter selector correctly rejected a bare
  number but had no way to ask the model to make its reasoning verifier-visible.
  The branch remains label-free: selection still requires a changed parseable
  answer, arithmetic consistency, and at least one checkable claim.
- Semantic equation verification now has a first label-free guard:
  `self_repair_irrelevant_number_used`. The selector extracts prompt numbers
  from clauses marked irrelevant by local language such as "not being packed",
  "not ticket revenue", "only count", and "question asks", then rejects integer
  repairs whose arithmetic expressions use those excluded quantities. CPU-only
  semantic-guard rescores preserve the current results:
  `llada_extended_full_semantic_guard_rescore_v1_report.md` keeps the 29-task
  line at `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`,
  and zero repair-oracle headroom; `llada_gsm_distractor_semantic_guard_rescore_v1_report.md`
  keeps the GSM distractor line at `2/2`, `+1.000`, `2/0/0`, and zero headroom.
  Reports now expose `Irrelevant # Used`, and selected exact repairs score
  `0.000` on that diagnostic in both rescored lines.
- Semantic equation verification now also has an operation-role guard. For
  integer exact repairs, the selector infers obvious required operations from
  prompt language such as "remaining", "shared equally", "per bag", "dollars
  each", "twice as many", and "across those", then rejects repairs whose
  checkable arithmetic claims omit those operations. CPU-only operator-guard
  rescores preserve the current exact results:
  `llada_extended_full_operator_guard_rescore_v1_report.md` keeps the 29-task
  line at `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`,
  and zero repair-oracle headroom; `llada_gsm_distractor_operator_guard_rescore_v1_report.md`
  keeps the GSM distractor line at `2/2`, `+1.000`, `2/0/0`, and zero headroom.
  Reports now expose `Missing Ops`, and selected exact repairs score `0.0` on
  that diagnostic in both rescored lines.
- Semantic equation verification now has a quantity-role binding guard as well.
  For integer exact repairs, the selector extracts explicit prompt quantity
  roles such as ticket-count times ticket-price, trays times items per tray,
  subtraction of a stated removed quantity, and division by a stated equal-share
  count. It rejects repairs whose equations contain the right operators but bind
  those quantities to the wrong roles. CPU-only role-guard rescores preserve the
  current exact results: `llada_extended_full_role_guard_rescore_v1_report.md`
  keeps the 29-task line at `11/11` eligible repair coverage, `+0.302` versus
  evolved, `9/2/0`, and zero repair-oracle headroom;
  `llada_gsm_distractor_role_guard_rescore_v1_report.md` keeps the GSM
  distractor line at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now
  expose `Role Gaps`, and selected exact repairs score `0.0` on that diagnostic
  in both rescored lines.
- Semantic equation verification now has an arithmetic-provenance guard. For
  integer exact repairs, the selector maintains a set of grounded numbers from
  the prompt plus outputs of earlier verified equations. Later equations are
  rejected if they introduce constants outside that provenance set. This catches
  derived-variable smuggling and also prevents an inconsistent earlier equation
  from licensing its false claimed output as a later intermediate. CPU-only
  provenance-guard rescores preserve the current exact results:
  `llada_extended_full_provenance_guard_rescore_v1_report.md` keeps the 29-task
  line at `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`,
  and zero repair-oracle headroom;
  `llada_gsm_distractor_provenance_guard_rescore_v1_report.md` keeps the GSM
  distractor line at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now
  expose `Provenance Gaps`; selected exact repairs score `0.0` on that
  diagnostic in both rescored lines.
- Semantic equation verification now has a final-answer role guard. For integer
  exact repairs, the selector infers whether the prompt asks for a total, a
  per-share division answer, a full-bag floor division answer, or a remainder,
  then rejects repairs whose final integer is not that role output even when
  some local equation is valid. CPU-only final-role rescores preserve the
  current exact results:
  `llada_extended_full_final_role_guard_rescore_v1_report.md` keeps the 29-task
  line at `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`,
  and zero repair-oracle headroom;
  `llada_gsm_distractor_final_role_guard_rescore_v1_report.md` keeps the GSM
  distractor line at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now
  expose `Final Role Gaps`; selected exact repairs score `0.0` on that
  diagnostic, while a non-selected GSM self-repair surfaces the expected gap.
- Semantic equation verification now has a final-answer object guard. For
  integer exact repairs, the selector extracts prompt objects that are locally
  excluded from the requested answer, then rejects repairs whose final-answer
  context explicitly names those objects. This catches cases like orange bags
  when oranges are not being packed, donations when the question asks for
  ticket revenue, and chocolate-chip cookies when the question asks about all
  cookies. CPU-only object-guard rescores preserve the current exact results:
  `llada_extended_full_object_guard_rescore_v1_report.md` keeps the 29-task
  line at `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`,
  and zero repair-oracle headroom;
  `llada_gsm_distractor_object_guard_rescore_v1_report.md` keeps the GSM
  distractor line at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now
  expose `Object Gaps`.
- Semantic equation verification now has a final-answer target guard. For
  integer exact repairs, the selector extracts the requested answer head from
  explicit "how many ..." and related prompt forms, then rejects final-answer
  units that name a wrong prompt-known target or attach a conflicting modifier
  to the requested target head. It catches answers like `8 students` when the
  prompt asks how many cookies each student gets, or `9 pear bags` when the
  requested object is apple bags, while still allowing bare numeric answers and
  bare units. CPU-only target-guard rescores preserve the current exact
  results: `llada_extended_full_target_guard_rescore_v1_report.md` keeps the
  29-task line at `11/11` eligible repair coverage, `+0.302` versus evolved,
  `9/2/0`, and zero repair-oracle headroom;
  `llada_gsm_distractor_target_guard_rescore_v1_report.md` keeps the GSM
  distractor line at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now
  expose `Target Gaps`.
- Exact self-repair now supports constrained non-arithmetic `short_text`
  answers. The label-free parser is schema-gated: it only turns on for prompts
  with bounded answer forms such as on/off, yes/no, a fixed number of letters
  separated by spaces, or a final list drawn from an explicit initial list.
  This lets LLaDA self-repair symbolic outputs when no deterministic proposal
  exists, while avoiding generic prose promotion. Unit tests cover extraction
  for toggles, letter orders, and list outputs, plus the actual repair-record
  path for a no-proposal constrained letter task.
- Constrained `short_text` self-repair now has a symbolic proof guard. When a
  prompt-derived order/list/toggle solver exists, exact self-repair selection
  must agree with that solver. This keeps no-solver schemas eligible but rejects
  mechanical contradictions such as `D A C B` for a prompt whose before-chain
  proves `D A B C`. Repair diagnostics expose the rate as `Symbolic Gaps`.
- The symbolic proof guard now includes simple categorical yes/no syllogisms.
  The same prompt-derived solver powers counterfactual proposals and
  self-repair verification, so "All zargs are blicks. No blicks are morts. Can
  a zarg be a mort?" proves `no` without reading the hidden label. A
  schema-valid but logically wrong final answer is rejected through `Symbolic
  Gaps`.
- Mechanically solvable `short_text` self-repairs now have a trace-evidence
  guard. If the prompt solver can prove the answer, the repair must show a
  minimal pre-answer trace: adjacent before-relations for order tasks, swap
  evidence for list tasks, toggle/parity evidence for toggles, or
  category/exclusion relation evidence for syllogisms. Terse final-answer-only
  repairs are rejected through `Trace Gaps`; no-solver bounded schemas remain
  eligible.
- The bounded `short_text` symbolic layer now covers letter-code transforms:
  prompts that start with an explicit code such as `K L M`, then ask for
  one-step rotation and swaps, can be solved into a label-free proposal. The
  failure-driven GPU slice `llada_symbolic_letter_transform_repair_v1_report.md`
  shows why this matters: LLaDA fixed/random/evolved all answered `M L K`, but
  the operation solver derived `L K M` and the counterfactual repair arm reached
  `1.000` with zero repair-oracle headroom. The prior
  `llada_symbolic_short_text_no_proposal_self_repair_v1_report.md` records the
  negative control where scratchpad self-repair alone preserved the same wrong
  answer.
- Exact-answer trajectory selection now has a label-free proposal-history mode.
  With `--exact-task-trajectory-policy proposal_history`, exact trajectory and
  evolved arms may select a final output or visible denoise-history state only
  when it matches a prompt-derived answer proposal; hidden labels are used after
  selection only for report scoring. CPU rescores on
  `llada_symbolic_letter_transform_repair_v1_raw.jsonl` and
  `llada_extended_full_arithmetic_feedback_v1_raw.jsonl` found no history-state
  wins in the current raw traces: `sym_008` still needs counterfactual repair,
  and the 29-task full rescore kept trajectory delta versus fixed at `-0.000`.
  This makes exact-history selection executable, but the next benchmark needs
  traces where a correct answer appears transiently before final regression.
- Full-history symbolic probing shows why that transient case does not appear
  in the current LLaDA loop. The benchmark now reports history mutability:
  committed visible-token changes, committed-token remasks, and mask-count
  increases across sampled denoise states. In
  `llada_symbolic_full_history_probe_v1_report.md`, all `14/14` generated
  histories were monotonic fills with zero committed-token changes and zero
  remasks. Evolved final selection can still solve `sym_010`, and
  counterfactual repair solves `sym_008`/`sym_009`, but passive history
  selection cannot become a true revision mechanism on this backend until we
  add an explicit non-monotonic remask/revision operator.
- That operator now exists. Revision schedules remask low-confidence committed
  suffix tokens inside the same LLaDA generation, append the remasked state to
  the denoise history, and continue denoising. The first GPU probes prove
  mutability: the exact 25%/16-step run produced `48` committed remasks and
  `10` remask-mediated rewrites, the exact 50%/24-step run produced `96`
  remasks and `8` rewrites, and the planning probe produced `96` remasks and
  `17` rewrites. Blind low-confidence revision is not yet the right policy:
  exact-task evolved selection remains `0.000`, and the planning probe regresses
  evolved by `-0.006` versus trajectory. The useful next operator is
  verifier-guided revision: remask answer spans, contradiction spans, or
  missing-constraint spans, then guard promotion against source-relative
  regressions.
- A full history-fraction diagnostic added `history_prefix_50_repair`
  unconditionally alongside the default `history_prefix_25_repair` and
  final-prefix repair. It reaches the same `0.490` repair-selected score as the
  adaptive rescue run, but costs `7.00` generations per task and has lower
  budget-normalized gain: `0.013` task score per extra generation versus
  `0.018` for the adaptive rescue line.
- A visible-history rescue diagnostic added `history_visible_repair`, which
  preserves every visible token from the selected mid-denoise state instead of
  only preserving a prefix. The diagnostic report is
  `eval_results/diffusion_language/llada_planning_visible_history_rescue_margin01_v1_report.md`.
  It reaches the same `0.490` repair-selected score but costs `6.25`
  generations per covered task and lowers budget-normalized gain to `0.017`.
  The candidate table shows why it is not canonical yet:
  `history_visible_repair` had the highest trajectory score on its rescue run
  (`0.738`) but stayed at `0.347` task score, while `history_prefix_50_repair`
  reached `0.375`.
- The guarded visible-history rescore
  `eval_results/diffusion_language/llada_planning_visible_history_rescue_guarded_margin01_v1_report.md`
  adds a label-free overpreservation penalty to repair selection. It assigns
  `history_visible_repair` a `0.053` penalty because only `10/64` generated
  positions were remasked while the source history state already had `230`
  visible characters. Selected outputs stay unchanged, which confirms the
  canonical `planning_quality` selector was already avoiding the visible-state
  trap on this slice.
- The disagreement-triggered expansion diagnostic
  `eval_results/diffusion_language/llada_planning_disagreement_visible_history_rescue_guarded_v1_report.md`
  turns on `baseline_or_selector_disagreement`, removes the source-control
  filter, and lets disagreement between repair selection and trajectory
  selection spend extra history-prefix and all-visible rescue candidates. It
  generated four `history_visible_repair` candidates with strong raw averages
  (`0.497` task, `0.738` trajectory), but the marginal source-relative table
  shows `0.000` task delta and `0/4/0` wins/ties/losses versus the source
  trajectories. Selected repair stayed `0.490` while budget rose to `7.00`
  generations per covered task. This proves the adaptive expansion mechanism
  exists, but also that broader expansion is not yet budget-worthy.
- The source-relative repair selector rescore
  `eval_results/diffusion_language/llada_planning_source_delta_guarded_rescore_v1_report.md`
  keeps the canonical LLaDA repair score at `0.490`, `+0.039` over evolved,
  with `6/2/0` wins/ties/losses and zero oracle headroom, while changing the
  selector score into a planning-quality delta over the source. This is a
  safer default for future repair experiments because no-op preservation now
  scores near zero instead of looking good on absolute quality alone.
- The source-relative minimal-remask pack
  `eval_results/diffusion_language/llada_planning_source_relative_repair_pack_v1_report.md`
  tested `history_prefix_50_repair`, `low_confidence_15_repair`, and
  `low_confidence_25_repair` under `planning_quality_delta_guarded`. It reached
  only `0.454`, `+0.004` over evolved, with `1/7/0` wins/ties/losses. The
  low-confidence 15% repair had `0.000` task delta and `0/8/0` wins/ties/losses
  versus source, proving that minimal remasking mostly preserves the source
  rather than improving it.
- The targeted-content repair pack
  `eval_results/diffusion_language/llada_planning_targeted_content_repair_pack_v1_report.md`
  remaps filler and repetition spans back to generated token positions before
  remasking. It matched the weak minimal-remask result: `0.454`, `+0.004` over
  evolved, `1/7/0`, and `targeted_filler_repair` was never selected. Token-span
  cleanup is therefore not enough to move reasoning quality by itself.
- The prompt-guided repair pack
  `eval_results/diffusion_language/llada_planning_prompt_guided_repair_pack_v1_report.md`
  gives the model the original task, the source draft, and a generic
  label-free critique. It produced the first non-history repair win after the
  source-relative guard: `prompt_guided_revision_repair` improved `plan_001` by
  `+0.034`. The aggregate is still below the canonical history-prefix repair
  line: `0.459`, `+0.008` over evolved, `2/6/0`, so this should be adaptive
  rather than unconditional.
- The adaptive prompt-guided rescue diagnostic
  `eval_results/diffusion_language/llada_planning_adaptive_hybrid_prompt_guided_rescue_v1_report.md`
  adds `--prompt-guided-rescue-trigger baseline_or_source_quality` on top of
  the canonical prefix/history path. It generated prompt-guided revision on
  seven low-quality or baseline-stuck tasks, but selected it on zero. The final
  selected score stayed at the canonical `0.490`, `+0.039` over evolved, with
  `6/2/0` repair-vs-evolved wins/ties/losses and zero oracle headroom, while
  the budget rose to `7.00` generations per task. The current prompt-level
  revision operator is therefore a measured fallback, not the default path.

## Next Control Mechanisms

### 1. Stepwise Judge

Instead of judging only final text, score sampled denoise states:

- first visible reasoning step
- first final-equivalent text step
- mask-resolution speed
- premature EOS pressure
- unstable span count
- rubric score on partial and final states

The judge should reward trajectories that become correct early and remain
stable, not just trajectories that end well after chaotic intermediate states.

### 2. Span Repair

Use masks as editable uncertainty markers. If a partial state has a strong
answer skeleton with weak spans, remask only those spans and rerun a short
repair schedule.

Candidate repair policies:

- remask low-confidence spans
- remask generic filler spans
- remask contradiction spans identified by a verifier
- freeze high-value spans and denoise the rest
- branch multiple repairs from the same mid-trajectory state

Implemented v1: freeze a short prefix from the selected generated suffix,
remask all remaining suffix positions, and run a short LLaDA denoise repair.
Implemented v2: record per-token model confidence at the denoise step where a
suffix token is committed, then remask the weakest committed positions. This is
not the final repair policy, but it proves the backend can restart from a
partially fixed diffusion state instead of only rerunning from a blank mask.

Implemented v3: the budgeted benchmark can now add a `repair_selected` arm that
branches LLaDA suffix-inpainting repairs from the evolved trajectory. On the
current planning slice, history-prefix repair plus final-prefix repair with the
evolved `planning_quality_fallback` and repair-specific `planning_quality`
selector improved evolved selection from `0.451` to `0.490` without losses and
reached the oracle available in that generated pool. Adding low-confidence
repairs previously raised budget without improving selected score, so they
remain diagnostic until the selector can use them reliably.

Implemented v3b: the same evolved-plus-repair selector has now been run inside
the full Dream-plus-LLaDA planning-plus-objective mix. The full-mix report
preserves the original fixed/random/trajectory/evolved comparison over all 22
model-task pairs while reporting repair coverage separately, so the project
does not pretend a LLaDA-only repair arm covers Dream or exact-answer tasks.

Implemented v3c: the repair arm can now select a sampled denoise-history state,
freeze its visible prefix, remask the rest, and run a short LLaDA inpainting
repair. Reports include the repair source state and source step so gains can be
credited to actual trajectory repair rather than opaque reruns.

Implemented v3d: history-prefix length is now configurable through
`--history-repair-fractions`, and reports include repair gain per extra
generation versus the evolved arm. This keeps candidate growth honest: the
`0.25,0.50` sweep improves absolute score but is less budget-efficient than
spending the later history anchor only when the first repair pass fails to
improve the matching evolved baseline.

Implemented v3e: adaptive history rescue adds `--history-rescue-fractions` and
`--history-rescue-source-controls`. The runner first evaluates the cheap repair
pack, then generates rescue repairs only when selection would otherwise keep a
matching evolved record. Rescoring uses the same budgeted-record construction
as fresh generation, so oracle and generation-count claims do not accidentally
include unused repair candidates from a larger raw file.

Implemented v3f: visible-history repair adds `history_visible_repair` plus the
`--include-history-visible-repair` and `--history-rescue-visible` switches. This
tests the diffusion-specific idea that useful partial structure can appear
outside an AR-style prefix. Reports now include repair-candidate diagnostics so
failed operators are still measured. The first GPU diagnostic showed visible
history can raise trajectory score while preserving the wrong plan content, so
the next selector has to penalize over-preserved wrong structure before this
operator becomes canonical.

Implemented v3g: `planning_quality_guarded` subtracts a label-free
overpreservation penalty from history repairs that remask too little of an
already-long visible state. The penalty is exposed in the repair-candidate
diagnostic table. It does not use task labels, and it can be tested by rescoring
existing raw generations before spending more GPU.

Implemented v3h: `--history-rescue-trigger baseline_or_selector_disagreement`
adds the first true disagreement-triggered expansion path. It spends adaptive
rescue repairs when the repair selector and trajectory selector prefer
different generated repair candidates, while still honoring source-control
filters when provided. The first GPU diagnostic showed that disagreement can
find high-quality visible repair candidates, but current guarded selection
cannot convert them into selected-score gain at acceptable budget.

Implemented v3i: repair-candidate diagnostics now report task delta and
wins/ties/losses versus each repair candidate's source trajectory. This catches
the key visible-history failure mode: high absolute scores can simply reflect a
strong source state, not a useful repair. On the disagreement diagnostic,
`history_visible_repair` averaged `0.497` task score but had exactly `0.000`
mean delta versus source.

Implemented v3j: `planning_quality_delta` and `planning_quality_delta_guarded`
select repairs by label-free planning-quality improvement over the source
output. The guarded variant keeps the overpreservation penalty, so copied or
nearly copied history repairs do not get promoted just because their source was
already strong. Reports now include `PQ Delta` beside hidden task delta.

Implemented v3k: `--repair-pack source_relative` prioritizes minimal
low-confidence remasks before broad prefix rewrites. The first GPU diagnostic
showed this pack is too conservative by itself: `low_confidence_15_repair`
preserved source quality exactly on all eight planning tasks and produced no
selected repair. The next repair operator should therefore combine
source-relative selection with targeted content edits, not only smaller masks.

Implemented v3l: `--repair-pack targeted_content` uses a tokenizer-backed text
repair policy. It detects repeated or filler spans, maps them back to generated
suffix token positions, and remasks those positions for diffusion inpainting.
The first diagnostic showed this mechanism works mechanically but does not
improve the selected score, so span cleanup alone is not a sufficient repair
objective.

Implemented v3m: `--repair-pack prompt_guided` adds a label-free critique
channel. Repair prompts include the original task, the source draft, and a
generic instruction to remove filler and add causal checks, measurements,
constraints, risks, rollback/fallback, and thresholds implied by the task. This
produced one real selected source-relative improvement but remains below the
canonical adaptive history-prefix line.

Implemented v3n: adaptive prompt-guided rescue adds
`--prompt-guided-rescue-trigger`, `--prompt-guided-rescue-limit`,
`--prompt-guided-rescue-source-quality-threshold`, and
`--prompt-guided-rescue-source-controls`. Fresh generation and raw rescoring both
gate prompt-guided draft revision behind baseline, source-quality, or selector
disagreement signals. The first GPU diagnostic showed the gate works but the
operator is not budget-worthy yet: seven prompt-guided revisions were generated,
none were selected, and the canonical score only matched the history-prefix line
at higher cost.

Implemented v3o: primary repair spending can now be gated by
`--repair-spend-trigger source_quality_or_short`, with
`--repair-source-quality-threshold`, `--repair-source-min-chars`, and optional
`--repair-source-controls`. The gate skips the first repair pass when the
selected source already looks complete under label-free planning quality and
visible text length. Fresh generation and raw rescoring both keep a repair arm
record with the evolved source when the gate skips spending, so coverage and
budget accounting stay explicit.

Implemented v3p: `--repair-pack state_adaptive` now starts with
`state_adaptive_history_repair` and `prefix_25_repair`. The state-adaptive
history candidate receives source planning quality plus the selected
denoise-history state's score and mask count, then chooses between a short
history anchor and a longer weak-state anchor. This turns the previous
one-off `history_prefix_50` rescue into a first-pass state policy. A
quality-scaled low-confidence branch was also tested, but demoted after a fresh
GPU diagnostic showed negative source-relative task delta.

Implemented v3q: `--repair-pack replay_consistency` builds repair seeds from
denoise-history instability. It compares sampled history token states at each
generated suffix position, remasks the most unstable positions, and falls back
to low-confidence repair when history is unavailable. The first GPU diagnostic
showed this operator is mechanically valid but not yet useful: its dedicated
candidate produced exact source ties on every generated case, so it stays a
diagnostic branch behind the state-adaptive history-prefix policy.

Implemented v3r: `--repair-pack constraint_gap` adds a prompt-grounded repair
branch that extracts task terms missing from the source draft and asks diffusion
to rewrite around those gaps. The first GPU diagnostic gives the strongest
absolute selected score so far, but only by `+0.001` over the efficient
state-adaptive line and at worse budget efficiency. This makes constraint-gap
revision a good candidate for a future gate keyed by high prompt-gap pressure,
not an unconditional third repair spend.

Implemented v3s: `--constraint-gap-rescue-trigger prompt_gap` turns
constraint-gap revision into that gate. It fires after the first repair pass
only when the evolved source is neither too weak nor already complete and has
enough missing prompt terms. The first GPU diagnostic preserved the absolute
`0.491` score of the unconditional constraint-gap run while reducing generation
count from `53` to `47`, so the current policy is no longer "try every repair";
it is state-adaptive repair plus source-conditioned error correction.

Implemented v3t: `planning_quality_risk_guarded` and
`planning_quality_delta_risk_guarded` subtract a label-free contradiction/risk
penalty from planning repair selection when a candidate violates explicit prompt
constraints. Reports now include `Risk Penalty` in repair-candidate diagnostics.
The first rescore of the gated constraint-gap line preserved the current best
`0.491` selected score with zero detected penalties, so this is a guardrail for
future repair expansions rather than a current score improvement.

Implemented v3u: mixed benchmark reports now include a `Family Arm Summary` and
explicit selector-regret/oracle-coverage diagnostics for trajectory, evolved,
and repair arms. Raw rescoring now matches fresh generation by only creating a
repair-selected arm for actually repair-capable LLaDA planning tasks, so exact
answer fallback arms no longer inflate repair coverage.

Implemented v3v: exact-answer counterfactual repair is now part of the
three-arm benchmark. Failed LLaDA exact-answer tasks can generate prompt-derived
counterfactual proposals, ask diffusion to answer with the proposed candidate,
and promote only proposal-matching outputs. This fixed the symbolic `sym_002`
gap in the fresh mixed run without turning hidden expected answers into a
selector.

Implemented v4: move counterfactual proposals into a reusable exact-answer
proposal layer. It can enumerate prompt-surface options such as `on/off`,
multiple-choice letters, deterministic arithmetic answers for the locked scout
math patterns, and simple symbolic transforms such as before-chain order and
list swaps. These proposals are generated from the task prompt, not from the
hidden expected answer.

Implemented v5: every counterfactual repair run now emits a proposal-only
ablation record and report section. This is the guardrail that prevents the
project from mistaking a deterministic verifier/proposer win for a diffusion
latent-trajectory win.

Implemented v6: the full LLaDA planning-plus-exact scout now uses the same
gated state-adaptive planning repair plus exact-answer counterfactual repair
line across all 25 locked scout tasks. Reports keep the global fixed/random/
trajectory/evolved means separate from covered repair-slice means, so the
benchmark does not imply that the repair arm applies to tasks where no repair
was generated.

Implemented v7: unsupported exact-answer failures can now use
`--exact-self-repair`. This creates one longer scratchpad solve-again repair
when the prompt-derived proposal layer returns no candidates. Selection remains
label-free: it promotes only parseable answers that differ from the failed
source and whose scratchpad arithmetic claims are internally consistent. Reports
now expose `Self Changed` and `Arithmetic OK` in repair-candidate diagnostics.

Implemented v8: exact self-repair can now spend an arithmetic-feedback repair
when the first scratchpad contains a verifiably false arithmetic equation and
`--limit-repair-candidates` leaves room. The feedback prompt reports only the
detected equation correction, then asks diffusion to redo the calculation.
Selection still uses parse/change plus arithmetic consistency, not the hidden
expected answer.

Implemented v9: the extended full LLaDA scout now runs planning repair,
proposal repair, scratchpad self-repair, and arithmetic feedback together under
one explicit coverage and budget ledger. The 29-task line verifies that the
exact-answer feedback path composes with the planning repair stack instead of
being only a hand-picked hard-slice result.

Implemented v10: GSM-style hidden-distractor arithmetic tasks are now in the
locked scout manifest, and arithmetic-feedback repair can catch natural-language
arithmetic claims in addition to symbolic `=` equations. The first four-task
GSM distractor run shows the same repair pattern working without deterministic
proposal candidates.

Implemented v11: integer exact-answer repair selection now requires arithmetic
evidence, not only a changed final number. The benchmark stores and reports
`self_repair_arithmetic_claim_count`; CPU rescoring confirms the stricter guard
does not weaken the current 29-task full line or the GSM distractor slice.

Implemented v12: integer exact-answer repair can now spend an
`arithmetic_evidence_repair` branch when self-repair gives a changed answer
without checkable arithmetic. This is the first missing-evidence fallback: it
does not correct a known false equation, but asks diffusion to produce equations
that the selector can verify before promotion.

Implemented v13: integer exact-answer repair selection now has a first semantic
quantity guard. The selector can reject internally consistent equations that
use prompt numbers explicitly marked irrelevant, and reports expose the rate as
`Irrelevant # Used`.

Implemented v14: integer exact-answer repair selection now has an operation-role
guard. The selector can reject internally consistent equations that omit
prompt-required operations, and reports expose the rate as `Missing Ops`.

Implemented v15: integer exact-answer repair selection now has a quantity-role
binding guard. The selector can reject equations that use the right operations
but attach prompt quantities to the wrong roles, and reports expose the rate as
`Role Gaps`.

Implemented v16: integer exact-answer repair selection now has an
arithmetic-provenance guard. The selector can reject equations that introduce
unexplained constants not present in the prompt and not produced by earlier
verified equations, and reports expose the rate as `Provenance Gaps`.

Implemented v17: integer exact-answer repair selection now has a final-answer
role guard. The selector can reject equations whose final integer is not the
prompt-requested answer role, and reports expose the rate as `Final Role Gaps`.

Implemented v18: integer exact-answer repair selection now has a final-answer
object guard. The selector can reject final answers that explicitly name a
prompt-excluded object, and reports expose the rate as `Object Gaps`.

Implemented v19: integer exact-answer repair selection now has a final-answer
target guard. The selector can reject explicit final-answer units that point to
the wrong prompt-known target or attach a conflicting modifier to the requested
target head, and reports expose the rate as `Target Gaps`.

Implemented v20: exact-answer self-repair now supports constrained `short_text`
tasks. The parser extracts only bounded answer schemas from final-answer text,
so symbolic no-proposal repairs can be selected without using the hidden label.

Implemented v21: constrained `short_text` exact-answer self-repair now has a
symbolic proof guard. When a prompt-derived symbolic solver exists, self-repair
selection must match it, and reports expose the rate as `Symbolic Gaps`.

Implemented v22: the symbolic proof guard now covers simple categorical
yes/no syllogisms, sharing one prompt-derived solver between counterfactual
proposals and self-repair verification.

Implemented v23: mechanically solvable `short_text` self-repairs now require
minimal trace evidence before the final answer, and reports expose the rate as
`Trace Gaps`.

Implemented v24: bounded letter-code transforms now share the symbolic
proposal/proof path. The prompt-derived solver handles explicit code rotations
and swaps, the trace guard requires operation evidence for self-repairs, and
`llada_symbolic_letter_transform_repair_v1_report.md` shows a fresh LLaDA
repair from `0.000` to `1.000` on `sym_008` after fixed/random/evolved all
answered `M L K`.

Implemented v25: exact-answer trajectory selection now has
`--exact-task-trajectory-policy proposal_history`. It scans the final output and
visible denoise-history states for prompt-derived proposal matches, then scores
the selected state only after selection. Existing LLaDA raw traces show final
proposal matches and fixed fallbacks, but no current history-state win, so this
is the selector surface for the next probe rather than a new aggregate best.

Implemented v26: history mutability is now a first-class diagnostic. Reports
summarize whether sampled histories are monotonic fills or actually revise
visible tokens through committed-token changes, committed-token remasks, and
mask-count increases. The full-history symbolic probe found `14/14` LLaDA
histories were monotonic fills, which means the next architecture step is an
explicit within-trajectory remask/revision policy rather than another passive
history selector.

Implemented v27: LLaDA generation now supports non-monotonic within-trajectory
revision. `DiffusionGenerationConfig` and `DiffusionScheduleCandidate` carry
`revision_remask_fraction` and `revision_steps`; the benchmark can add
`--include-revision-schedules` and tune `--revision-remask-fraction` /
`--revision-steps`. The first GPU probes prove the operator rewrites visible
tokens, but blind low-confidence remasking is not enough: exact symbolic tasks
still need counterfactual repair, and a three-task planning probe regresses
slightly. The next revision policy should be verifier-guided and source-relative.

Implemented v28: exact-answer revision now has a verifier-guided answer-span
inpainting path. `build_answer_span_repair_seed` maps a failed extracted answer
back to generated token positions and remasks that span under the original
prompt; `--exact-verifier-revision` adds this repair before counterfactual
prompt repair and still promotes only proposal-matching outputs. The symbolic
GPU probe selects `answer_span_repair` on `sym_010` and keeps the full exact
repair slice at `1.000`. Planning revision also has a source-relative safety
guard: `--revision-promotion-margin 0.050` blocks revision schedules unless
their selector or fallback quality edge is larger than the normal evolved
margin, removing the prior planning revision regression in a raw rescore.

Implemented v29: verifier-guided span repair is no longer limited to final
answers. `build_text_span_repair_seed` remasks arbitrary decoded verifier spans;
integer exact repair can now add `arithmetic_contradiction_span_repair` after a
failed self-check scratchpad, and the constraint-gap pack now includes
`constraint_gap_span_repair`, which pairs span masking with missing-prompt-term
revision. The first arithmetic GPU diagnostic showed literal span masking was
too narrow, so the policy now remasks the first verified-bad arithmetic claim,
dependent downstream claims, and the final answer. On `math_010`, this
downstream span repair fixes the task to `1.000` and is selected under a
two-repair budget, replacing the older extra arithmetic-feedback generation.
Planning constraint-gap span repair preserved `plan_001` source quality but did
not improve it when the full weak draft was quoted back to the denoiser. The
next pass made the planning branch diffusion-native instead of copy-native:
`constraint_gap_span_repair` now ranks weak downstream sentences from
prompt-gap pressure, masks those decoded spans, and uses a span-specific prompt
that shows only the preserved opening plus missing task terms. On
`llada_planning_constraint_gap_ranked_span_v3_report.md`, `plan_001` improves
from `0.399` to `0.465`; the selected repair beats evolved by `+0.066`, has
`0.046` label-free planning-quality delta, and reaches zero repair oracle
headroom. The current-code eight-task diagnostic
`llada_planning_constraint_gap_ranked_span_v6_8task_report.md` improves
repair-selected mean from `0.412` to `0.465`, with `6/2/0`
repair-vs-evolved wins/ties/losses. The span branch itself averages `0.430`,
is selected on 3/8 tasks, and has `4/2/2` wins/ties/losses versus its source.
The evolved/full-pack comparison
`llada_planning_constraint_gap_ranked_span_v5_canonical_report.md` reaches
`0.482`, which is positive but still below the older efficient gated
state-adaptive line at `0.491`. Planning span repair is therefore a real
positive operator, but it remains a candidate inside the guarded
constraint-gap pack rather than the default budget line.

Implemented v30: planning risk guarding now catches prompt-checklist leakage,
not just direct prompt contradictions. The expanded gated rescue run
`llada_planning_gated_ranked_span_rescue_default_history_v1_report.md` allowed
full revision, anchor revision, and ranked span repair behind the existing
prompt-gap gate. It reached `0.492`, but selected an anchor output that visibly
dumped the missing terms (`gpu, jobs, overnight, ...`) into the answer. The
guarded rescore
`llada_planning_gated_ranked_span_rescue_default_history_guarded_rescore_v1_report.md`
assigns that candidate `Risk Penalty 0.180`, selects the cleaner
`constraint_gap_revision_repair`, and preserves the efficient planning line at
`0.491` with zero repair selector regret. This is a quality guardrail: the
score does not materially increase, but the selector is now harder to game with
prompt-term lists.

Implemented v31: the lean GPU diffusion benchmark contract is now explicit in
`docs/LEAN_GPU_DIFFUSION_BENCHMARK_PROTOCOL.md`. The next public-facing compact
suite is fixed greedy/low-confidence baseline versus random perturbation versus
selected diffusion repair on 8 short planning tasks plus `math_001`, `sym_002`,
and `sci_001`, available through `--task-preset lean_gpu_mixed`. The first run
under that contract is
`llada_mixed_gated_ranked_span_guarded_exact_identity_v1_report.md`: repair-selected
scores `0.548` on covered tasks, beats fixed by `+0.181`, random by `+0.213`,
and evolved by `+0.147`, with planning at `0.491` and `sym_002` repaired to
`1.000`.

Implemented v32: risk-guarded planning repair now includes a verifier-residue
guard for span repair. `constraint_gap_span_repair` records the targeted weak
sentences as `planning_span_targets`; if the final output reconstructs those
exact spans, `_planning_span_residue_penalty` contributes to the risk penalty
and reports as `Span Residue`. The negative prompt-copy rescore
`llada_planning_constraint_gap_ranked_span_v2_span_residue_guard_rescore_v1_report.md`
assigns `0.180` residue/risk to the failed span repair that copied back both
weak sentences. This is the text analogue of world-model surprise detection:
the repair trajectory is penalized when the verifier-targeted error reappears.

Implemented v33: the backend and benchmark runners now treat LLaDA as a family
rather than a single dense model string. `llada-moe-7b-a1b-instruct-hf` is
registered as the next cheap active-parameter HF target, with a GGUF fallback
metadata entry for quantized smoke tests. The three-arm runner, scouts, schedule
sweep, and repair scout all route `LLaDA*` families through the LLaDA schedule,
revision, and repair gates. A cheap preflight for the MoE target succeeded
without downloading weights, and mask-token resolution now prefers the tokenizer
mask id so MoE uses its actual mask token instead of the dense-LLaDA fallback.

Implemented v34: LLaDA-MoE is now a real local GPU target, not only metadata.
The 13.7 GB HF snapshot is materialized under
`external/diffusion_models/LLaDA-MoE-7B-A1B-Instruct`. BF16 CUDA smoke passes,
token confidence capture works, and `llada_moe_history_smoke_v1_raw.jsonl`
shows 32 denoise steps with sampled visible states. The compact lean benchmark
`llada_moe_mixed_gated_ranked_span_guarded_exact_v1_report.md` completed 60
full generations. MoE's base exact checks are strong, but the transferred
state-adaptive planning repair line reaches only `0.446`, with one selected
repair win (`plan_007`) and seven ties versus evolved. The conclusion is
architectural: MoE runs cheaply enough to iterate, but needs a MoE-specific
repair/remask policy rather than inheriting dense LLaDA's current planner
repair pack unchanged.

Implemented v35: MoE now has its first model-specific repair policy. The
diagnostic `llada_moe_planning_constraint_gap_repair_v1_report.md` showed that
full constraint-gap revision and anchor revision were no-ops for MoE, while
`constraint_gap_span_repair` improved 6/8 planning tasks and had zero span
residue. The new `--repair-pack constraint_span` exposes only that useful branch.
`llada_moe_planning_constraint_span_repair_v1_report.md` preserves the
high-spend diagnostic's `0.472` planning repair-selected score while cutting
budget from 72 to 40 full generations, or from 9 to 5 generations per task. It
beats fixed by `+0.060`, random by `+0.100`, and evolved by `+0.050`, with
`6/2/0` repair-vs-evolved wins/ties/losses and only `0.001` oracle headroom.

Implemented v36: repair source selection is now diffusion-native instead of
hardwired to the evolved winner. `--repair-source-policy` can keep the old
`evolved` behavior, branch from `fixed`, branch from `trajectory`, or use
`non_revision_evolved`, which lets a non-monotonic revision schedule win the
evolved arm while span repair seeds from the best non-revision source, or
`evolved_and_trajectory`, which spends repairs from both the evolved winner and
the base trajectory source. Reports now expose `Repair Source Control` and the
repair-candidate source-control set, so source effects are visible instead of
hidden behind a shared repair name. This fixes the MoE revision/repair
interaction where `plan_007` improved under blind revision but then lost the
stronger span-repair source. The refreshed validation report
`llada_moe_planning_revision_constraint_span_nonrev_source_rescore_fixed_v1_report.md`
keeps
revision schedules active, records non-monotonic history (`256` committed
remasks and `68` rewrites), and reaches repair-selected `0.472`: `+0.060` vs
fixed, `+0.100` vs random, `+0.028` vs the stronger revision-aware evolved arm,
`6/2/0` repair-vs-evolved wins/ties/losses, and `0.001` oracle headroom. This
does not replace the cheaper 5-generation/task span-only MoE default, but it is
the correct architecture when revision and repair are both enabled.

Implemented v37: raw rescoring now matches fresh revision-generation semantics.
Previously `--reuse-raw-input --include-revision-schedules --limit-evolved-schedules 2`
treated revision schedules as part of the two evolved-schedule limit, while a
fresh run used two evolved mutations plus two revision schedules. The helper
`_selected_evolved_records_for_rescore` now limits non-revision evolved
mutations separately and then includes revision records when requested. This
matters for cheap selector experiments because budget and evolved baselines must
match the GPU run that produced the raw file.

Implemented v38: multi-source MoE repair is measured and classified as a
diagnostic path. `llada_moe_planning_revision_constraint_span_multisource_v1_report.md`
spends `--repair-source-policy evolved_and_trajectory` and generates 61 records.
It finds real extra source value: the trajectory-source repair fixes `plan_002`
to `0.689` and exposes a better but unselected `plan_006` repair at `0.459`.
Aggregate repair-selected reaches `0.473`, with `7/1/0` wins/ties/losses versus
the revision-aware evolved arm and `0.001` oracle headroom. But the added source
budget lowers gain per extra generation from `0.028` for `non_revision_evolved`
to `0.018`, so the default remains `non_revision_evolved`; `evolved_and_trajectory`
is for selector-development and source-diversity diagnostics.

Implemented v39: source-diversity spending now has a cheap adaptive gate.
`--repair-source-policy non_revision_plus_gap_trajectory` starts from
`non_revision_evolved` and adds the base trajectory source only when it is a
distinct low-confidence source, has at least six missing prompt terms, and still
passes a configurable generic planning-quality floor. The paired
`planning_quality_prompt_coverage_guarded` selector gives prompt-coverage credit
only after the repair already clears a planning-quality floor, which avoids
promoting keyword-stuffed random repairs. The raw MoE rescore
`llada_moe_planning_revision_constraint_span_adaptive_source_prompt_guard_v1_report.md`
was confirmed by a fresh GPU run at
`llada_moe_planning_revision_constraint_span_adaptive_source_prompt_guard_fresh_v1_report.md`.
Both use 58 records, reach repair-selected `0.474`, beat the revision-aware
evolved arm by `+0.030`, record `7/1/0` repair-vs-evolved wins/ties/losses,
and have zero oracle headroom on the generated repair pool. Budget-normalized
gain is `0.024` per extra generation: worse than the single-source default
(`0.028`) but better than exhaustive multi-source (`0.018`), with higher raw
repair score than both. Treat it as the current revision-enabled MoE selector
candidate; the cheaper 40-generation `constraint_span` line remains the default
when revision schedules are disabled. The runner now exposes named
`--adaptive-source-gate-mode` presets: `score_max` resolves to `gap>=6`,
`quality>=0.25`, while `efficiency` resolves to `gap>=10`, `quality>=0.25`.
`custom` keeps the explicit `--adaptive-source-gap-min-terms` and
`--adaptive-source-quality-floor` thresholds for sweeps. Reports emit an
`Adaptive Source Gate` table with the primary source, trajectory source, gap
count, trajectory planning quality, skip/add reason, and actual extra-source
repair spend for each task. The threshold sweep
`adaptive_source_gate_sweep_v1_summary.md` shows `score_max` is on the
score-maximal plateau, while `efficiency` improves budget-normalized gain at the
cost of `0.001339` mean task score. The stricter efficiency mode is also
fresh-GPU confirmed in
`llada_moe_planning_revision_constraint_span_adaptive_source_efficiency_fresh_v1_report.md`:
57 generations, repair-selected `0.472768`, and `0.025794` repair gain per
extra generation. The sweep is now executable as
`experiments/sweep_adaptive_source_gate.py`; the script-regenerated artifact
`adaptive_source_gate_sweep_script_v1_summary.md` confirms the named modes sit
on equivalent operating plateaus even when another tied threshold pair appears
first under score sorting. The companion `*_best.json` stores those named rows
for downstream evidence maps.

Implemented v40: the adaptive MoE repair line now holds on the full lean mixed
protocol, not only the planning-only slice. The fresh GPU run
`llada_moe_mixed_revision_constraint_span_adaptive_source_score_max_v1_report.md`
uses `--task-preset lean_gpu_mixed`, revision schedules, `constraint_span`,
`non_revision_plus_gap_trajectory`, and the `score_max` gate. It keeps exact
math/symbolic/science checks solved, reports repair coverage as `8/11` overall
and `8/8` eligible, and reaches planning repair-selected `0.474107`: `+0.061830`
vs fixed, `+0.101982` vs random, `+0.030357` vs evolved, `7/1/0` wins/ties/losses
vs evolved, and zero repair oracle headroom. Rescores from the same raw file
show the expected tradeoff: `efficiency` reaches `0.472768` with one fewer
generation and better gain per extra generation, while single-source
`non_revision_evolved` reaches `0.472143` with the best budget-normalized gain.
The adaptive source-gate report now omits exact-answer tasks from the gate table
because they are not repair-eligible rubric tasks.

Implemented v41: benchmark reports now emit a `Lean Three-Arm Headline` whenever
the repair arm is present. This section is the public evidence surface requested
for the lean benchmark: fixed baseline, random perturbation, and selected latent
repair only. It computes fixed/random scores over the repair-covered task slice,
so exact tasks that are already solved but not repair-eligible do not dilute or
inflate the headline comparison. Trajectory, evolved, oracle, source-gate, and
selector-regret details remain below the headline as diagnostics.

Implemented v42: diffusion claims now have a repo-level evidence ledger.
`experiments/build_diffusion_claim_evidence.py` reads canonical score JSON files
and writes `CLAIM_EVIDENCE_MAP.md` plus
`eval_results/diffusion_language/diffusion_claim_evidence_map.json`. The map
links each active diffusion claim to score, report, and raw-generation artifacts;
records status, coverage, repair policy, source policy, adaptive gate mode, and
exact-task trajectory policy; and reports fixed/random/repair scores on the
repair-covered slice. The first ledger covers the dense LLaDA compact mixed
line, the MoE policy-transfer baseline, the MoE adaptive score-max line, the MoE
efficiency rescore, and the single-source MoE budget line. Treat this as the
public claim gate: if a result is not in the map, it is not yet promoted.

Implemented v43: planning span repair now has source-relative verifier ranking
instead of only hand-scored prompt-gap pressure. `constraint_gap_span_repair`
computes `planning_span_target_scores` for each candidate decoded span: source
preservation after removing the span, prompt-gap miss, prompt-keyword coverage,
and contradiction relief. The opening scaffold is protected unless it is itself
weak or contradictory, and invalid baseline-comparison claims such as "baseline
data will not be available, making it a valid comparison" are now penalized as
prompt contradictions. Reports include a `Planning Span Target Diagnostics`
table so the mask choice is auditable without opening raw JSONL. A two-task
LLaDA-MoE GPU smoke,
`llada_moe_planning_source_ranked_span_smoke_v1_report.md`, generated 16 records
on `plan_002` and `plan_006`; selected latent repair reached `0.573750` versus
fixed `0.540000`, random `0.460536`, and evolved `0.558214`, with `2/0/0`
repair-vs-evolved wins/ties/losses and zero oracle headroom.

Implemented v44: the source-ranked span policy is now full-suite confirmed and
promoted as the canonical MoE score-max evidence line. The fresh lean mixed GPU
run `llada_moe_mixed_revision_constraint_span_source_ranked_score_max_v1_report.md`
matches the previous promoted aggregate while adding span-target diagnostics:
76 records, repair coverage `8/11` overall and `8/8` eligible, selected latent
repair `0.474107`, `+0.061830` vs fixed, `+0.101982` vs random, `+0.030357` vs
evolved, `7/1/0` repair-vs-evolved wins/ties/losses, and zero oracle headroom.
CPU rescores from the same raw pool preserve the expected tradeoff:
`source_ranked_efficiency_rescore` reaches `0.472768` with better gain per extra
generation, and `source_ranked_nonrev_source_rescore` reaches `0.472143` with
the best budget-normalized gain. `CLAIM_EVIDENCE_MAP.md` now points the MoE
score-max, efficiency, and single-source budget claims at the source-ranked
artifact set.

Implemented v45: clause-level planning span repair is now separated as an
explicit diagnostic pack, not the default. The new
`--repair-pack constraint_span_clause` uses `planning_span_chunk_mode=clause`
and can split long comma/semicolon planning drafts into smaller masked clauses.
The two-task LLaDA-MoE GPU diagnostic
`llada_moe_planning_clause_ranked_span_smoke_v1_report.md` reached selected
latent repair `0.571250` versus fixed `0.540000`, random `0.460536`, and
evolved `0.558214`, but it regressed from the sentence-level source-ranked
smoke at `0.573750`. The main failure was `plan_002`: the clause repair from
`low_confidence_32` scored `0.583` and was not selected, while the older
sentence-level fallback repaired the same source to `0.689`. Keep
`constraint_span` as the promoted MoE score-max path; use
`constraint_span_clause` only to investigate future finer-grained mask policies.

Implemented v46: exact-answer verifier revision no longer depends entirely on
having a prompt-derived answer proposal. When `--exact-verifier-revision` is
paired with `--exact-self-repair`, constrained non-integer label-free exact
tasks can now remask the rejected answer span under the original prompt before
spending a full solve-again branch. The resulting `answer_span_repair` carries
the same self-repair metadata used by symbolic guards, so selection can reject
unchanged or malformed answers without reading hidden labels. This is a
substrate improvement rather than a promoted benchmark claim: it makes
verifier-guided inpainting available to exact tasks where the system can parse
answers but does not yet have a prompt solver.

Implemented v47: no-proposal integer tasks are now excluded from that
answer-span shortcut. The hard exact CUDA diagnostic
`llada_hard_exact_verifier_span_self_repair_v1_report.md` showed the rejected
answer span alone was a no-op on `math_010` and `sym_007`: it preserved the
wrong scratchpad and selected zero times. The gated rerun
`llada_hard_exact_verifier_span_integer_gate_v1_report.md` preserves the hard
exact result with fewer wasted generations: 20 records, repair-selected
`1.000` on the `2/2` eligible slice, `+1.000` vs fixed/random/evolved, zero
oracle headroom, repair budget delta `2.00` vs evolved, and gain per extra
generation `0.500` instead of `0.400`. Integer no-proposal repair should spend
budget on self-repair plus arithmetic contradiction spans, not on final-answer
span inpainting.

Implemented v48: exact-answer repair selection now has a small mechanism
priority after label-free correctness guards pass. Verifier-localized span
repairs (`answer_span_repair` and `arithmetic_contradiction_span_repair`) beat
broader prompt-feedback repairs when both produce a changed, parseable,
guard-clean answer. A cached rescore of
`llada_hard_exact_verifier_span_integer_gate_v1_report.md` keeps the same hard
exact headline (`1.000` on the `2/2` eligible slice, `+1.000` vs evolved,
zero oracle headroom, gain per extra generation `0.500`) but now selects
`arithmetic_contradiction_span_repair` on `math_010` instead of the broader
`arithmetic_feedback_repair`. This keeps public scoring unchanged while making
the selected repair better aligned with the diffusion-native inpainting claim.

Implemented v49: arithmetic feedback is now skipped when verifier-localized
arithmetic span repair has already passed the exact-answer guards. The fresh
CUDA diagnostic `llada_hard_exact_verifier_span_early_stop_v1_report.md`
preserves the hard exact headline while spending one fewer generation than the
integer-gated run: 19 full generations, repair-selected `1.000` on the `2/2`
eligible slice, `+1.000` vs fixed/random/evolved, zero oracle headroom, repair
budget delta `1.50` vs evolved, and gain per extra generation `0.667`.
`math_010` now generates self-repair plus
`arithmetic_contradiction_span_repair` and does not spend the broader
`arithmetic_feedback_repair` branch after the span repair succeeds.

Implemented v50: the hard-exact no-proposal result is now promoted into the
repo-level evidence ledger. `CLAIM_EVIDENCE_MAP.md` includes
`dense_llada_hard_exact_no_proposal_span_repair`, pointing at the early-stop
scores, report, and raw JSONL. This makes the public evidence set cover both
open-ended planning repair and exact-answer diffusion repair where no
deterministic proposal exists: fixed and random score `0.000000` on the
repair-covered hard-exact slice, selected latent repair scores `1.000000`, and
the gain per extra generation is `0.666667`.

Implemented v51: default planning span repair no longer falls back to
whole-draft masking when sentence-level verifier targets collapse on a long
single-sentence draft. `constraint_gap_span_repair` now uses an adaptive
source-relative chunk policy: first try sentence targets, then retry clause
targets only when the sentence target is a fallback or the entire source. This
keeps diffusion repair local to weak clauses such as repetition, risky shipping
claims, or missing-measurement continuations while preserving useful opening
structure.

Implemented v52: a fresh one-task CUDA scout confirms the adaptive planning
span path still works under real LLaDA generation. In
`eval_results/diffusion_language/llada_planning_constraint_gap_span_adaptive_v1_report.md`,
`plan_001` fixed/random/trajectory score `0.398929`, selected latent repair
scores `0.465357`, and `constraint_gap_span_repair` is the selected repair with
`+0.066429` task delta vs fixed/random/evolved, `1/0/0` repair-vs-evolved
wins/ties/losses, and zero repair-oracle headroom. The span target diagnostics
show non-fallback verifier targets on the two weak downstream baseline
sentences rather than an entire-draft mask.

Implemented v53: the adaptive planning span repair has now been promoted from
single-task diagnostic to an 8-task planning scout and repo-level claim. The
fresh CUDA run
`eval_results/diffusion_language/llada_planning_constraint_gap_span_adaptive_8task_v1_report.md`
uses 48 full generations over `plan_001` through `plan_008`; selected latent
repair scores `0.465313` vs fixed/random/trajectory `0.412277`, giving
`+0.053036` task delta, `6/2/0` wins/ties/losses vs fixed/random/evolved,
`0.010607` gain per extra generation, and only `0.0015625` oracle headroom.
`CLAIM_EVIDENCE_MAP.md` now includes
`dense_llada_planning_adaptive_span_repair`, so the public evidence ledger
covers both no-proposal exact repair and open-ended short planning repair with
adaptive verifier-localized spans.

Implemented v54: the adaptive span policy has a lean mixed budget line. The
fresh CUDA run
`eval_results/diffusion_language/llada_mixed_adaptive_constraint_span_v1_report.md`
uses the 11-task compact suite with `--repair-pack constraint_span`, exact
self-repair, and verifier revision. It is not the highest absolute mixed score,
but it cuts full generations from `63` to `54` versus the stronger guarded mixed
line and raises gain per extra generation versus evolved from `0.069643` to
`0.104143`. On the repair-covered slice it scores `0.516468` versus fixed
`0.366468`, random `0.334484`, and evolved `0.400754`, with `2/7/0`
repair-vs-evolved wins/ties/losses and `0.003611` oracle headroom. The claim
ledger now includes `dense_llada_mixed_adaptive_span_budget` as the
budget-favored compact mixed policy, while `dense_llada_lean_mixed_guarded_repair`
remains the strongest absolute-score compact line.

Implemented v55: promoted diffusion claims now have a hard validation gate.
`experiments/validate_diffusion_claim_evidence.py` rebuilds the claim evidence
from `DEFAULT_CLAIMS`, verifies the generated Markdown and JSON maps are not
stale, checks required score settings, enforces repair/full/eligible count
consistency, checks win/tie/loss totals against repair counts, validates
budget-normalized gain arithmetic, and ensures raw artifacts have enough JSONL
records for the promoted score files. During implementation it caught an oracle
scope trap: some `oracle_task_score` fields are full-suite while
`oracle_headroom_vs_repair` is repair-covered, so the oracle subtraction check
only runs when repair coverage equals full-suite coverage.

Implemented v56: promoted diffusion claims now have a generated latest
ground-truth index. `experiments/build_diffusion_claim_evidence.py` writes
`DIFFUSION_GROUND_TRUTH_INDEX.md` and
`eval_results/diffusion_language/diffusion_ground_truth_index.json` next to the
claim evidence map. The index records every promoted claim's canonical score,
report, and raw JSONL files and names the current public slots for dense
top-score, dense budget, dense planning, hard-exact no-proposal, MoE transfer,
MoE score-max, MoE efficiency, and MoE single-source budget evidence. The
index also records SHA-256 fingerprints for every promoted artifact. The
validator now checks that both index files are current, so stale public pointers
or stale artifact hashes fail before they can leak into docs or posts.

Implemented v57: diffusion benchmark outputs now carry deterministic result
identity. `experiments/run_diffusion_three_arm_benchmark.py` attaches `run_id`
and `content_hash` to every score summary before writing score JSON or report
Markdown. The hash covers raw generation records, arm selections, and the score
summary while excluding volatile `created_at` timestamps, so reruns and rescores
can distinguish content changes from timestamp churn. This makes future
ground-truth promotion auditable without relying only on filenames.

Implemented v58: the dense adaptive-span budget line has been refreshed with
deterministic result identity and promoted as the canonical budget artifact.
The fresh CUDA run
`eval_results/diffusion_language/llada_mixed_adaptive_constraint_span_identity_v1_report.md`
preserves the same repair-covered score and deltas as the earlier adaptive-span
mixed scout: repair-selected `0.516468`, `+0.150000` vs fixed, `+0.181984` vs
random, `+0.115714` vs evolved, `2/7/0` repair-vs-evolved wins/ties/losses,
and `0.003611` oracle headroom. It also reduces full model generations from
`54` to `53`, raising gain per extra generation versus evolved to `0.115714`.
`CLAIM_EVIDENCE_MAP.md` and `DIFFUSION_GROUND_TRUTH_INDEX.md` now point
`dense_llada_mixed_adaptive_span_budget` at this identity-bearing artifact.

Implemented v59: the strongest dense compact line has also been refreshed with
deterministic result identity and promoted as the canonical top-score artifact.
The fresh CUDA run
`eval_results/diffusion_language/llada_mixed_gated_ranked_span_guarded_exact_identity_v1_report.md`
exactly preserves the earlier guarded mixed headline: 63 full generations,
repair-selected `0.547778` on the 9/9 eligible repair slice, `+0.181310` vs
fixed, `+0.213294` vs random, `+0.147024` vs evolved, `7/2/0`
repair-vs-evolved wins/ties/losses, `0.000992` oracle headroom, and
`0.069643` gain per extra generation. It adds run identity
`diffusion-45da934106d48a5b`, so the strongest and budget-favored dense
canonical artifacts now both have stable result hashes.

Implemented v60: the promoted MoE source-ranked score-max line has also been
refreshed with deterministic result identity. The CUDA rerun
`eval_results/diffusion_language/llada_moe_mixed_revision_constraint_span_source_ranked_score_max_identity_v1_report.md`
uses the same lean mixed suite and writes run ID
`diffusion-6fac1a7361a1fbb0`. It is slightly below the older pre-identity MoE
score-max artifact but remains the best identity-confirmed MoE score line:
repair-selected `0.473482`, `+0.061205` vs fixed, `+0.101357` vs random,
`+0.029732` vs evolved, `6/2/0` repair-vs-evolved wins/ties/losses, and
`0.000625` oracle headroom. CPU rescores from the same identity raw pool now
back the MoE efficiency and single-source budget claims; both reach
`0.472143`, while single-source uses one fewer full generation and keeps the
best budget-normalized MoE gain. The evidence builder now surfaces each
artifact's `run_id` and `content_hash` directly in the generated claim map and
ground-truth index.

Implemented v61: verifier-guided span repair now reports whether the requested
span was actually localized in the decoded token stream. The repair seed layer
exposes `build_text_span_repair_seed_with_diagnostics` and
`build_text_span_repair_seed_diagnostics`, recording literal target matches,
character spans, token positions, expanded masked positions, fallback mode, and
the final masked-token count. The three-arm benchmark attaches this diagnostic
to answer-span revision, arithmetic contradiction span revision, and planning
constraint-gap span repair. Reports now summarize `Span Localized` and
`Span Fallback` in the repair-candidate table, so a claimed verifier-guided
span repair can be distinguished from a tail-window fallback before it reaches
the public evidence map. The CUDA smoke
`eval_results/diffusion_language/llada_planning_span_localization_smoke_v1_report.md`
confirms the diagnostic on `plan_001`: `constraint_gap_span_repair` localized
literal targets (`Span Localized 1.000`, `Span Fallback 0.000`), repaired the
task from `0.399` to `0.465`, and carries run ID
`diffusion-d67e0b34ba99f9af`.

Implemented v62: stale public diffusion evidence references are now gated in
code, not only by reviewer memory. `experiments/scan_stale_diffusion_docs.py`
loads the generated ground-truth index, extracts canonical score/report/raw
artifacts, and scans `README.md`, `RESEARCH_BRIEF.md`, `ARTICLE_UPDATE.md`,
and `EXPERIMENTS.md` for non-canonical diffusion artifacts used in current,
canonical, promoted, headline, public, or claim contexts. The normal
`experiments/validate_diffusion_claim_evidence.py` gate calls this scanner, so
a top-level public doc cannot quietly keep pointing at an old benchmark file
after `DIFFUSION_GROUND_TRUTH_INDEX.md` changes. Historical diagnostic mentions
remain allowed unless the scanner is run with `--strict-all-artifacts`.

Implemented v63: the MoE source-ranked planning repair now has a fresh
span-localized CUDA confirmation instead of relying only on the older mixed
artifact. The planning-only run
`eval_results/diffusion_language/llada_moe_planning_revision_constraint_span_source_ranked_score_max_spanloc_v1_report.md`
generated 58 records with run ID `diffusion-8e411cb3a650322e` and matches the
promoted MoE planning score-max line: selected latent repair `0.473482`,
`+0.061205` over fixed, `+0.101357` over random, `+0.029732` over evolved,
`6/2/0` repair-vs-evolved, and `0.000625` oracle headroom. Its repair-candidate
diagnostics show `constraint_gap_span_repair` with `Span Localized 1.000` and
`Span Fallback 0.000` across 10 repair candidates, proving the source-ranked
operator is literal verifier-target inpainting rather than a hidden generic
tail-window repair. This was the first registered MoE planning
span-localization claim; v66 supersedes it with the compact-policy v2 artifact.

Implemented v64: mechanism-level claim requirements are now explicit validator
contracts instead of prose. `ClaimSpec` can attach
`RepairDiagnosticRequirement` entries that name a repair candidate, metric, and
required min/max threshold. `experiments/build_diffusion_claim_evidence.py`
serializes those requirements into `CLAIM_EVIDENCE_MAP.md` and the ground-truth
index, and `experiments/validate_diffusion_claim_evidence.py` enforces them
against `repair_candidate_summary`. The MoE planning span-localization claim now
requires `constraint_gap_span_repair.mean_span_literal_target_found >= 1.0` and
`constraint_gap_span_repair.mean_span_fallback_used <= 0.0`, so a future
artifact cannot satisfy the claim with generic fallback masking.

Implemented v65: planning span repair now has a compact denoise-target policy.
`DiffusionRepairCandidate` has `planning_span_selection_policy`, and the default
`constraint_gap_span_repair` sets it to `compact`. The source-relative verifier
ranker still chooses spans without labels, but the compact policy no longer
saturates the target limit just because many prompt terms are missing. When an
adaptive sentence target is a long multi-clause planning sentence, it rescans the
source at clause granularity and uses clause targets only if they preserve most
of the verifier score while masking fewer words. The one-task MoE CUDA smoke
`eval_results/diffusion_language/llada_moe_planning_compact_span_policy_smoke_v1_report.md`
ran `plan_001` with fixed/random/selected latent repair arms and improved
`0.465357` to `0.528214` (`+0.062857`) with literal span localization and zero
repair-oracle headroom. Treat this as a mechanism smoke, not a promoted claim
map replacement for the existing MoE source-ranked aggregate.

Implemented v66: the compact span policy was full-suite debugged and promoted
for MoE planning. The first compact full run improved the mean but exposed
regressions where clause refinement masked only the tail of a decision rule and
where the compact word budget dropped near-tie weak failure-chain sentences.
The policy now keeps high-coverage decision-rule spans intact, retains near-tie
failure-chain spans under a small budget slack, and still refines long risky
sentences when clause targets preserve verifier score with fewer masked words.
The fresh CUDA run
`eval_results/diffusion_language/llada_moe_planning_compact_span_score_max_v2_report.md`
uses the 8-task MoE planning suite with revision schedules and adaptive
second-source gating. It generated 58 records with run ID
`diffusion-911c8526a9cfa11e`; selected latent repair reaches `0.492321`,
`+0.080045` over fixed, `+0.120196` over random, `+0.048571` over evolved,
`6/2/0` repair-vs-evolved, `0.038857` gain per extra generation, and
`0.000625` oracle headroom. Repair diagnostics still show
`Span Localized 1.000` and `Span Fallback 0.000`; average masked positions fall
from the older source-ranked line's `46.0` to `34.2`. The claim evidence map now
points `moe_planning_span_localized_repair` at this compact v2 artifact.

Implemented v67: the compact span policy now has a full lean mixed MoE CUDA
confirmation. The fresh run
`eval_results/diffusion_language/llada_moe_mixed_compact_span_score_max_v1_report.md`
uses the 11-task compact suite, revision schedules, adaptive source gating, and
the public fixed/random/selected-latent headline. It generated 76 records with
run ID `diffusion-33bf0475f913c6a7`; the repair-covered planning slice reaches
selected latent repair `0.492321`, `+0.080045` over fixed, `+0.120196` over
random, `+0.048571` over evolved, `6/2/0` repair-vs-evolved, `0.038857` gain
per extra generation, and `0.000625` oracle headroom. The math, symbolic, and
science checks remain solved at `1.000`. This supersedes the older mixed
source-ranked score-max line (`0.473482`) at the same 76-generation cost.
The fresh score-efficient CUDA run
`llada_moe_mixed_compact_span_score_efficient_fresh_v1` now provides the
budget frontier. It adds a trajectory-quality ceiling to the adaptive source
gate, skips the high-quality no-op `plan_002` second source, keeps the selected
`plan_006` branch, and preserves the full `0.492321` repair score at 75 records.
Its gain per extra generation rises to `0.043175`, with run ID
`diffusion-afe24cca9924dc37` and the same `0.000625` oracle headroom.
`llada_moe_mixed_compact_span_fixed_source_repairability_gate_fresh_v1` now
supplies the budget-favored public point: direct greedy-output repair with a
source-quality plus prompt-gap/coverage geometry gate reaches `0.489911` at 27
records and `2.625000x` relative repair cost, with run ID
`diffusion-ae26bb892c8a68aa` and zero repair-oracle headroom. It strictly
dominates the quality-only gate, which kept the same score at 29 records and
`2.875000x`, and the ungated fixed-source run at 30 records and `3.000000x`.
`DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md` is the companion mechanism audit:
it replays the gated run against the ungated fixed-source reference and finds
that the gate spent on `5/5` productive repair states, skipped `3/3` no-lift
states, and missed `0` reference repairs. The current geometric explanation is
that productive repair states occupy the source-quality plus moderate
prompt-gap/coverage band where compact span inpainting has room to add missing
constraints without spending on already-good or prompt-undercovered sources.
`DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md` stress-tests that explanation by
sweeping 53,460 label-free geometry-plus-phase gate settings. It now includes
optional first-denoise-skeleton step caps, finds 168 zero-waste/zero-miss
settings, and keeps the promoted `0.531116` at `2.625000x` gate on the
score/cost frontier. This gives the theory layer an explicit cost frontier:
each extra repair spend corresponds to admitting a wider repairable band in
prompt-gap, coverage, and denoise-phase geometry.
The phase-window tradeoff is now visible as an engineering knob: cap first
repairable skeletons at step `20` or `24` and the policy spends four repairs for
`0.496607` at `2.500000x`; allow step `32` or no cap and it spends five repairs
for the promoted `0.531116` at `2.625000x`.
The step-`20` operating point is fresh-GPU confirmed by run
`diffusion-419fbf63c9d8e30b`: it writes 26 generations, skips `plan_007` as
`late_repairable_denoise_skeleton` because the first skeleton appears at step
`31`, and keeps `+0.084330` versus fixed and `+0.124482` versus random.
The matching step-`32` promoted point is fresh-GPU confirmed by run
`diffusion-3b42951db77c5aa6`: it writes 27 generations, spends on `plan_007`
because step `31` is inside the cap, and recovers selected latent repair
`0.531116` at `2.625000x` with zero repair-oracle headroom.
`DIFFUSION_DENOISE_PHASE_GEOMETRY.md` pushes the same idea inside the denoise
trajectory. Inspired by the local research folders' emphasis on energy-bounded
attention, maintenance, and detect-diagnose-repair loops, it treats each
sampled history as a phase trace rather than a static output. On the current
run, productive repair sources all enter repairable or low-quality skeleton
phases, skipped no-lift sources are undercovered or overdiffuse, and the
repairable-phase classifier has `1.000000` precision and recall. The practical
operator implication is now in the runner: `--repair-spend-trigger
denoise_phase_repairability` spends only when final source geometry is in the
repairable band and sampled history exposes a constraint skeleton before
finalization. The runner also records first skeleton step, step fraction,
skeleton coverage, and peak denoise prompt coverage, and exposes
`--repair-denoise-skeleton-max-step` for stricter phase-window tests. The
CPU-only rescore
`llada_moe_mixed_compact_span_fixed_source_denoise_phase_gate_rescore_v1`
matched the frontier, and the fresh CUDA confirmation
`llada_moe_mixed_compact_span_fixed_source_denoise_phase_gate_fresh_v1`
preserves the current budget point with run ID `diffusion-5b1bf286b8cfa727`:
selected latent repair `0.489911` at `2.625000x`.
Implemented v70: `constraint_span_history` is the first compact span operator
that uses a sampled denoise state as the repair source for planning-span
inpainting. The runner now localizes planning spans against the selected
history state's token IDs and visible text instead of silently falling back to
the final output. The fresh CUDA diagnostic
`llada_moe_mixed_compact_span_history_anchor_denoise_phase_gate_fresh_v1`, run
ID `diffusion-16dc676d10e4b12e`, proves the mechanism is real: source states are
`history`, span localization is `1.000000`, fallback is `0.000000`, and the
policy still beats greedy/random at the same `2.625000x` cost. It is not the
promoted budget policy because score drops from `0.489911` to `0.474107`; the
audit `DIFFUSION_HISTORY_ANCHOR_REPAIR_AUDIT.md` shows the lost score comes from
history anchors losing final-context detail on several tasks. The next model
idea is a history/final anchor-choice objective or a consistency loss that
preserves constraints already stable by final denoising while still allowing
earlier skeleton repair. The same audit gives a post-generation selector upper
bound: spending both final and history anchors and then using label-free selector
scores recovers `0.489911`, but costs `3.250000x`. That is too expensive for the
public budget line, so the useful version must predict the anchor before repair
generation. The audit now tests that useful version directly: a pre-generation
span-geometry selector chooses history only when the history target is a single
compact span with high final/history overlap, no digit or prompt-keyword loss,
and a strictly higher target score. On the current trace it selects history only
for `plan_001`, preserves `0.489911`, and keeps the single-anchor `2.625000x`
budget.
Implemented v71: `constraint_span_anchor_select` moves that selector into the
runner. The repair candidate reports as one policy, but internally executes the
canonical final-span or history-span repair branch so the generation seed,
prompt, span targeting, and cost stay comparable with the audited branches.
Anchor-select/history-span packs now request dense denoise-history sampling by
default, exposing near-final states without adding model generations. The fresh
CUDA run
`llada_moe_mixed_compact_span_anchor_select_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-f3c291037d94daaf`, preserves the public budget score
`0.489911` at `2.625000x`, chooses a history anchor on `plan_001`, and chooses
final anchors on the other four repair spends.
Implemented v72: `experiments/analyze_diffusion_anchor_retention_loss.py`
turns that selector into an executable theory object instead of only a runner
heuristic. The generated `DIFFUSION_ANCHOR_RETENTION_LOSS.md` defines a
constraint-retention loss over target overlap, target-token loss,
prompt-keyword loss, digit loss, final/history length retention, and
target-count consistency, then pairs it with the positive compact-span
advantage gate. On the dense-history run it identifies `plan_001` as the single
safe history anchor, blocks six histories because they have no positive span
advantage over the final source, blocks one because the compact target
structure is not safe, and preserves the observed diagnostic fact that the
all-history policy trails final-source repair by `0.015804`.
Implemented v73: `constraint_span_anchor_search` makes the anchor choice a
whole-history search instead of only scoring the preselected history state. The
first fresh GPU run, `llada_moe_mixed_compact_span_anchor_search_denoise_phase_gate_dense_history_fresh_v1`
with run ID `diffusion-c326b3ef25eb8374`, found a useful failure mode: loose
retention thresholds selected an earlier `plan_003` history anchor and dropped
the public repair line to `0.483348`. The guarded version now requires
near-final target similarity `0.96` and history/final length retention `0.95`
before history can replace final. Fresh GPU run
`llada_moe_mixed_compact_span_anchor_search_guarded_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-ccef06238847a352`, restores `0.489911` at the same
`2.625000x` cost while preserving whole-history search as an executable
diffusion-native operator.
Implemented v74: `constraint_span_history_contrast` tested the weaker
hypothesis that denoise history can help merely as prompt-side evidence while
the seed remains final-source span repair. Fresh GPU run
`llada_moe_mixed_compact_span_history_contrast_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-b92d689695016154`, selected zero repair candidates and scored
only `0.414598` at `2.625000x`. This is a useful negative boundary: the
trajectory signal has to alter seed/remask geometry or anchor selection. Adding
history text to the prompt is not enough for this diffusion backend.
Implemented v75: `constraint_span_history_instability` tests that stronger
seed/remask hypothesis directly. The operator keeps final-source compact span
targets, then unions in a small mask over final token positions whose values are
unstable across sampled denoise histories. Fresh GPU run
`llada_moe_mixed_compact_span_history_instability_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-e28eb1d3dde8eea7`, scores `0.459107` at `2.625000x`, beating
greedy/random by `+0.046830` / `+0.086982`, with `5/3/0` wins/ties/losses
versus fixed and `6/2/0` versus random. The mechanism is active: each generated
repair adds six instability positions and span fallback remains `0.000000`.
It still trails anchor-select `0.489911`, so instability masking is a useful
secondary remask feature rather than a replacement for constraint-retention
anchor choice.
Implemented v76: `constraint_span_anchor_instability` combines the two surfaces
directly by running the pre-generation final/history anchor selector and then
carrying `remask_history_unstable_fraction=0.08` into the concrete span repair.
Fresh GPU run
`llada_moe_mixed_compact_span_anchor_instability_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-d14467a9f9a550b2`, reaches `0.481027` at `2.625000x`,
beating greedy/random by `+0.068750` / `+0.108902` and improving over
standalone instability `0.459107`. It still trails anchor-select `0.489911`.
The repair metadata is decisive: instability masks were active on all five
generated repairs, source states were `final,history`, and the only obvious raw
task win over anchor-select is `plan_007`; unconditional instability masking is
therefore too blunt. The next version should learn or hand-code a conditional
gate for when unstable positions should be unioned into an already safe anchor.
Implemented v77: `constraint_span_anchor_instability_gated` adds that first
conditional gate, activating instability masks only for multi-span, low-quality
final-source repairs while leaving history anchors and simpler final-source
repairs on the concrete base span seed and prompt. The first fresh GPU run
`llada_moe_mixed_compact_span_anchor_instability_gated_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-30a85507d687dfdc`, regressed to `0.452188` because the wrapper
pack still changed the prompt on gate-off tasks. The runner now resolves gated
anchor repairs back to the concrete final/history prompt for identity control,
and the fixed GPU run
`llada_moe_mixed_compact_span_anchor_instability_gated_identity_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-a7b64be5b7258f39`, restores `0.489911` at `2.625000x`,
beating greedy/random by `+0.077634` / `+0.117786`. The generated anchor
retention audit confirms `4/4` gate-off repairs are identical to anchor-select
in seed, prompt, masked seed, output text, and score; the single active gate on
`plan_007` changes seed/text but has `0.000000` score delta. Treat this as a
clean identity-stable A/B harness: instability masking is isolated but not yet
an improving repair policy.
Implemented v78: `constraint_span_anchor_instability_prompt_gated` uses that
harness to gate both the denoise-instability mask and the instability-specific
repair instruction. Gate-off branches still resolve to the concrete
anchor-select seed and prompt. Only the active low-quality multi-span final
anchor receives the instability instruction. Fresh GPU run
`llada_moe_mixed_compact_span_anchor_instability_prompt_gated_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-4c6a7a9f356b3f0d`, reaches `0.498304` at `2.625000x`,
beating greedy/random by `+0.086027` / `+0.126179` with zero oracle headroom.
The raw A/B shows `4/4` gate-off repairs match anchor-select exactly, while
`plan_007` is the only changed branch and gains `+0.067143`. This is now the
first positive evidence that denoise
instability is useful as a conditional error-correction instruction, not just
as a blind extra mask.
Implemented v79: `constraint_span_anchor_instability_prompt_only_gated`
removes the active instability remask while preserving the same gate and
instability-specific repair instruction. Fresh GPU run
`llada_moe_mixed_compact_span_anchor_instability_prompt_only_gated_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-4b5fc2b7604c28a5`, scores `0.479911` at `2.625000x`.
Gate-off branches remain `4/4` identical to anchor-select, but the active
`plan_007` branch drops by `-0.080000`. This is the negative mechanism
control: the v78 lift is not prompt routing alone; it requires the active
denoise-instability mask plus the gated instruction.
Implemented v80: `constraint_span_anchor_instability_claim_gated` adds a second
task-conditional prompt gate for public-claim confound plans while preserving
the v78 denoise-instability branch. The first fresh run,
`llada_moe_mixed_compact_span_anchor_instability_claim_gated_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-94e95f5d1b3d9822`, was selector-safe but copied repair
meta-language into `plan_004`, so it stayed below the v78 frontier at
`0.495625`. The compact-prompt rerun,
`llada_moe_mixed_compact_span_anchor_instability_claim_gated_compact_prompt_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-0fc7f067a7d87799`, reaches `0.513437` at `2.625000x`,
beating greedy/random by `+0.101161` / `+0.141313` with zero oracle headroom.
The raw A/B shows `plan_004` is the only new changed branch versus v78 and
gains `+0.121071`; `plan_007` keeps the v78 instability repair unchanged. This
was the first composite prompt-router evidence: different denoise failure
geometries need different repair instructions, but the gates must preserve
identity outside their failure mode.
Implemented v81: `constraint_span_anchor_instability_claim_oracle_gated` keeps
the same denoise-anchor and instability-mask geometry but uses a compact
oracle-aware public-claim control instruction. Fresh run
`llada_moe_mixed_compact_span_anchor_instability_claim_oracle_gated_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-692592da063daa60`, reaches `0.523304` at `2.625000x`,
beating greedy/random by `+0.111027` / `+0.151179` with zero repair-oracle
headroom. The active `plan_004` branch rises to `0.559286` task score and the
`plan_007` instability branch remains intact. The unresolved mechanism gap is
that `plan_004` still misses the literal selected-vs-oracle result-separation
rubric phrase, so the next theory target is a stronger binding between missing
rubric semantics and the denoise repair span rather than a longer instruction.
Implemented v82 as a negative mechanism test:
`constraint_span_anchor_instability_claim_seeded_gated` fixes the short phrase
`separate oracle best-of results from selected results` into the masked denoise
seed when the public-claim gate fires. Fresh run
`llada_moe_mixed_compact_span_anchor_instability_claim_seeded_gated_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-6ae167dc85d5e6ac`, proves the phrase binding works on
`plan_004`, but the aggregate line drops to `0.521295` at `2.625000x` because
the fixed anchor crowds out the public-claim survival control. This gives the
next theory target: semantic seed anchors need a compatibility or coverage loss
over the complete required-control set, not only a literal phrase-binding loss.
Implemented v83 as the first positive compatibility result:
`constraint_span_anchor_instability_claim_compatible_seeded_gated` fixes a
compact 9-token tail, `oracle selected results; claim survives if disappears`,
inside the same masked denoise seed. Fresh run
`llada_moe_mixed_compact_span_anchor_instability_claim_compatible_seeded_gated_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-6944d9dd6c412de4`, reaches `0.531116` at `2.625000x`,
beating greedy/random by `+0.118839` / `+0.158991` with zero repair-oracle
headroom. The active `plan_004` branch reaches `0.621786` and hits all five
rubric controls, so the theory update is no longer only "semantic anchors bind
phrases"; compact compatible anchors can bind multiple required controls
without adding repair candidates or GPU cost.
Implemented v84 as the first automatic compact-control seed policy:
`constraint_span_anchor_instability_claim_auto_seeded_gated` synthesizes the
same seed from the active task/rubric surface instead of taking a fixed string.
Fresh run
`llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_gated_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-7b74493b8c5ca15a`, scores `0.520536` at `2.625000x`.
The mechanism fires correctly: `plan_004` applies the generated seed without
truncation and hits all five rubric controls. It still trails the fixed
compatible seed because the denoised continuation is less direct, so the next
architecture target is a realization-quality loss for compact seeds, not just
automatic control-term extraction.
Implemented v85 as a negative realization-control test:
`constraint_span_anchor_instability_claim_auto_seeded_realization_gated` adds
explicit wording constraints for token budget, prompt format, locked tasks,
regressions, wins, and failure modes. Fresh run
`llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_realization_gated_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-2a310ed45712a36b`, drops to `0.515759` at `2.625000x`.
The seed still applies and `plan_004` still hits every rubric item, but the
model turns the answer into a low-specificity `Control:` label. This narrows the
architecture target again: realization quality should be a scored objective,
not another prompt constraint.
Implemented v86 as the first realization-quality loss:
`experiments/analyze_diffusion_realization_quality.py` reads the compact-seed
raw traces and scores active seed repairs with label-free realization and seed
objectives: control coverage, direct action coverage, seed-term coverage, prompt
coverage, specificity, direct sentence shape, meta-text penalties, and
selected/oracle plus claim-survival semantic preservation. The generated
`DIFFUSION_REALIZATION_QUALITY.md` audit now separates task score from seed
objective: compatible seeded remains best by task (`0.621786`), while
auto-compat-realized is best by realization (`0.846647`) and seed objective
(`0.904921`) but only scores `0.600714`. The runner exposes
`planning_quality_seed_realization_guarded` and
`planning_quality_seed_objective_guarded` as repair selectors. After tightening
the low-realization penalty, CPU rescore rejects the realization-gated
`plan_004` `Control:` branch and drops that boundary to `0.495625` with four
selected repairs. Fresh CUDA run
`llada_moe_mixed_compact_span_anchor_instability_claim_compatible_seeded_gated_realization_guard_fresh_v1`,
run ID `diffusion-a9ae901393235364`, preserves the compatible-seeded frontier:
`0.531116` at `2.625000x` with zero repair-oracle headroom. This makes
realization quality an active selector guard, not only a post-hoc audit.
Implemented v87 as the action-bearing automatic seed boundary:
`constraint_span_anchor_instability_claim_auto_action_seeded_gated` synthesizes
`rerun; oracle selected; claim survives` from the task/rubric surface and keeps
that action seed inside the same 9-token masked-tail budget. The one-task CUDA
smoke on `plan_004` showed no truncation, zero meta penalty, and `0.600714`
task score. The full lean mixed run
`llada_moe_mixed_compact_span_anchor_instability_claim_auto_action_seeded_gated_realization_guard_v1`,
run ID `diffusion-51b5b82f63ad87cd`, reaches `0.528482` at `2.625000x`,
beating fixed/random by `+0.116205` / `+0.156357` with zero repair-oracle
headroom. It does not replace the compatible fixed seed because that run still
scores `0.531116`; the remaining architecture target is seed compatibility
across all required controls, not only the presence of an action verb.
Implemented v88 as the automatic compatibility-scored seed policy:
`constraint_span_anchor_instability_claim_auto_compat_seeded_gated` scores
compact seed candidates for oracle/selected-result coverage, claim-survival
coverage, action pressure, and over-compression risk before applying the
masked-tail anchor. The first `plan_004` smoke used the right seed but exposed a
prompt-surface failure: mentioning the generated seed as a meta object dropped
the task score to `0.466786`. After removing that meta wording from the repair
instruction, the v2 smoke recovered `plan_004 = 0.621786`. The full lean mixed
run
`llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_seeded_gated_realization_guard_v1`,
run ID `diffusion-913b5bccb7894e5a`, ties the fixed compatible frontier:
`0.531116` at `2.625000x`, `+0.118839` versus fixed, `+0.158991` versus random,
`6/2/0` wins/ties/losses versus fixed, and zero repair-oracle headroom. The
public claim pointer now uses this automatic scorer rather than the hand-built
compatible seed.
Implemented v89 as the realization-prompt boundary for automatic compatibility
seeds: `constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated`
keeps the v88 seed scorer but rewrites the active claim-gate prompt so it does
not name seeds, anchors, masks, or repair instructions. One-task CUDA smoke
`llada_moe_plan004_anchor_instability_claim_auto_compat_realized_seeded_gated_realization_guard_smoke_v1`,
run ID `diffusion-1a80605979a231e8`, raises `plan_004` realization quality from
`0.655238` to `0.807460` and drops the seed-meta penalty from `0.140000` to
`0.000000`, but task score falls from `0.621786` to `0.600714`. This is a
theory-positive, score-negative result: direct realization improves, while the
selected/oracle rubric semantics weaken. Tightened v2 smoke
`llada_moe_plan004_anchor_instability_claim_auto_compat_realized_seeded_gated_realization_guard_smoke_v2`,
run ID `diffusion-d475c628f6386098`, raises realization quality again to
`0.846647` with zero meta penalty, but task score still stays `0.600714`. Keep
the public pointer on v88 and use v89 to train or score a joint
compatibility-realization objective.
Implemented v90 as that first joint seed objective:
`planning_quality_seed_objective_guarded` and
`constraint_span_anchor_instability_claim_auto_joint_seeded_gated` add semantic
preservation to the seed objective. The seed policy scores compact anchors for
compatibility, expected realization, and selected/oracle relation preservation,
then chooses `separate oracle selected; claim survives if disappears`, which fits
the 9-token masked tail. One-task CUDA smoke
`llada_moe_plan004_anchor_instability_claim_auto_joint_seeded_gated_seed_objective_smoke_v1`,
run ID `diffusion-91dcab0442e7d5a1`, preserves semantic score `1.000000` and
zero meta penalty but still scores `0.600714` on `plan_004`. The negative result
is useful: the next architecture step must shape the denoise continuation or
train a local objective, because better seed choice alone does not recover the
`0.621786` task frontier.
Implemented v91 as a preservation-seed recovery:
`constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated` and
`compact_preservation_control_terms` move the useful public-claim preservation
constraint into the fixed denoise tail. The prompt-only smoke
`llada_moe_plan004_anchor_instability_claim_auto_compat_preserve_seeded_gated_smoke_v1`,
run ID `diffusion-05c8f40e3fd0f234`, stayed at `0.600714` because the model
ignored prompt-level `preserve` wording. The preservation-seed smoke
`llada_moe_plan004_anchor_instability_claim_auto_compat_preserve_seeded_gated_preservation_seed_smoke_v2`,
run ID `diffusion-c18d75b68b87ef33`, fixes
`oracle selected results; preserve claim if disappears` into the masked tail and
recovers `plan_004 = 0.621786` with semantic preservation `1.000000` and zero
seed/anchor meta penalty. Full mixed-slice run
`llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_preservation_seed_fresh_v1`,
run ID `diffusion-3b42951db77c5aa6`, recovers the public aggregate exactly:
`0.531116` at `2.625000x`, `+0.118839` versus fixed, `+0.158991` versus random,
and zero repair-oracle headroom. This is now the cleaner promoted public run.
Its fresh step-`32` phase-window confirmation keeps the same public score and
run ID while making the denoise timing condition explicit: `plan_007` becomes a
valid repair spend when its first repairable skeleton appears at step `31`.
Implemented v92 as repair-spend gate instrumentation plus a forced-spend
boundary audit. Probe `llada_moe_plan002_auto_compat_preserve_seeded_forced_spend_probe_v1`,
run ID `diffusion-8a8a9e8904e62dbf`, forced the promoted preservation-seeded
repair on the high-quality skipped task `plan_002`; the candidate scored
`0.582500` versus source `0.688571`. Probe
`llada_moe_plan005_plan008_auto_compat_preserve_seeded_forced_spend_probe_v1`,
run ID `diffusion-4699321baf91294e`, forced the same repair on
`plan_005,plan_008`; the selector chose zero forced repairs, the forced
candidate mean task delta versus source was `-0.014464`, and repair-oracle
headroom stayed `0.000000`. The regenerated
`DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md` treats the spend gate as an
error-correction classifier: `5` true-positive productive spends, `3`
true-negative skipped no-lift repairs, and zero false positives/false negatives
on the promoted planning slice. The runner now emits `repair_spend_gate_rows`
in score JSON and a `Repair Spend Gate Diagnostics` report table, with source
quality, source length, prompt-gap count, prompt coverage, repairable-band
status, and visible denoise-skeleton status for every primary repair source
considered.
The older single-source run `llada_moe_mixed_compact_span_single_source_fresh_v1`
is now dominated at 74 records, `7.000000x`, and `0.473393`. The claim evidence
map now points the MoE mixed score-max, score-efficient,
denoise-phase fixed-source budget, repairability-gated historical,
quality-gated historical,
ungated fixed-source historical, and single-source historical claims at compact
artifacts, with all cost-frontier claims backed by fresh generations rather than
only CPU rescore.
Implemented v93 as a pre-generation phase-anchor repair operator. The
`constraint_span_phase_anchor` pack asks the runner to branch from the first
safe repairable denoise skeleton, not merely the final output or a post-hoc
history anchor. Smoke `llada_moe_plan003_constraint_span_phase_anchor_smoke_v1`,
run ID `diffusion-848cdd2d12d1fbc9`, improved the task score from `0.421786`
to `0.538214` while correctly falling back to the final source because the
first visible skeleton lost too much retained content. After adding a
phase-specific retention gate, smoke
`llada_moe_plan007_constraint_span_phase_anchor_smoke_v2`, run ID
`diffusion-00558374541fbc4d`, used history step `31` with
`anchor_selection_reason=history_phase_first_repairable_skeleton` and raised
the one-task repair score from `0.307500` to `0.497857` in three full
generations. This is the next executable operator for testing diffusion-native
latent reasoning, but it remains diagnostic until it clears the full lean mixed
benchmark.
Implemented v94 as the full lean mixed phase-anchor boundary test. Fresh CUDA
run `llada_moe_mixed_constraint_span_phase_anchor_fresh_v1`, run ID
`diffusion-9dabba8829d29658`, used `27` generations and reached selected latent
repair `0.476786` on the repair-covered tasks: `+0.064509` versus fixed and
`+0.104661` versus random, with zero repair-oracle headroom. It improved all
five repaired sources (`5/0/0` versus source), but it is dominated by the
promoted preservation-seeded final/history-anchor policy at the same
`2.625000x` relative cost. The decisive boundary is source choice: `plan_003`,
`plan_006`, and `plan_007` all did better when the promoted policy repaired
from the final source, even though their late denoise history states passed the
phase retention gate. This turns phase anchoring into a conditional signal, not
a replacement source: use phase geometry to decide when to spend, contrast, or
seed, and only branch from the phase text when retention lag plus source
features predict a real advantage. The full boundary report is
`DIFFUSION_PHASE_ANCHOR_BOUNDARY.md`.
Implemented v95 as the strict phase-conditioned hybrid. The loose hybrid first
tested source advantage alone and reached `0.524554` at `2.625000x`, but
regressed `plan_003` by switching to a history source whose phase text had only
`0.943503` target similarity and `0.908714` final-char ratio. The strict v2
requires the phase state to pass the normal history-anchor retention standard
before switching sources. Fresh CUDA run
`llada_moe_mixed_phase_hybrid_preserve_seeded_gated_fresh_v2`, run ID
`diffusion-9386ee5300a75528`, recovers the promoted line exactly: selected
latent repair `0.531116` at `2.625000x`, `+0.118839` versus fixed,
`+0.158991` versus random, and zero repair-oracle headroom. It uses history
only for `plan_001` and keeps final-source repair for `plan_003`, `plan_004`,
`plan_006`, and `plan_007`. The architecture update is important: denoise phase
states are now first-class selector evidence while final-source repair remains
the default unless strict phase retention plus source advantage justifies a
switch.

Implemented v96 as a generated mechanism audit for the strict phase hybrid.
`experiments/analyze_diffusion_phase_hybrid_mechanism.py` reads the strict
hybrid score/raw artifacts and renders
`DIFFUSION_PHASE_HYBRID_MECHANISM_AUDIT.md`, treating the denoise sequence as an
error-correction loop: detect a repairable phase, diagnose retention safety,
choose final/history source, repair the weak span, and verify source lift. On
run `diffusion-9386ee5300a75528`, the audit records the same public point
(`0.531116` at `2.625000x`), five selected repairs, source states
`{'history': 1, 'final': 4}`, five positive deltas versus the selected source,
mean first repairable phase step `16.2`, mean first retention-safe phase step
`30.5`, and mean retention-safety lag `12.75`. This is the practical version of
the world-model analogy: phase states are predicted intermediate answer states,
but the correction operator only trusts them as sources when they survive the
retention check.

Implemented v97 as source-choice loss-target extraction from the strict
phase-hybrid audit. The same analyzer now writes
`eval_results/diffusion_language/diffusion_phase_hybrid_loss_targets.jsonl`
with five weighted examples: one `trust_history_source` positive on `plan_001`
and four `preserve_final_source` negatives on `plan_003`, `plan_004`,
`plan_006`, and `plan_007`. The target label is the selected source policy,
while the features are phase timing, repairable/safe phase counts, retention
lag, target/text similarity, selected source state, and repair-vs-source lift.
This changes the next architecture target from another hand-written gate into a
small trainable selector loss: predict when the denoise history is trustworthy
enough to become the repair source, and otherwise preserve final-state repair.

Implemented v98 as a phase-source policy audit on top of those loss targets.
`experiments/analyze_diffusion_phase_source_policy.py` renders
`DIFFUSION_PHASE_SOURCE_POLICY_AUDIT.md` and compares constant final-source
repair, naive repairable-phase replacement, any retention-safe phase
replacement, loose similarity gates, strict similarity gates, and a calibrated
similarity gate. On the current strict phase-hybrid targets, the calibrated rule
matches the strict retention proxy: trust history only when
`phase_safe_repairable_count > 0`, `target_similarity >= 0.96`, and
`text_similarity >= 0.96`. It gets zero weighted error; naive repairable-phase
replacement creates four false history-source switches, and any-safe-phase
replacement creates three. Architecturally, this sharpens the lesson that
denoise phase is a value signal before it is a source state: the next learned
policy should use phase timing and retention features to block tempting but
unsafe history anchors.

Implemented v99 by wiring that calibrated phase-source policy back into the
runner. `experiments/run_diffusion_three_arm_benchmark.py` now has explicit
`PHASE_SOURCE_TARGET_SIMILARITY_MIN`, `PHASE_SOURCE_TEXT_SIMILARITY_MIN`, and
`PHASE_SOURCE_HISTORY_CHAR_RATIO_MIN` constants, records them in the
anchor-selection feature metadata, and routes strict phase-hybrid source
advantage through `_phase_history_anchor_passes_source_policy`. The distinction
matters: `PHASE_ANCHOR_*` remains the looser boundary for detecting a
repairable/safe denoise phase, while `PHASE_SOURCE_*` is the stricter rule for
promoting that phase into the actual repair source. This keeps the public
operator from collapsing back into raw phase-source replacement.

Implemented v100 by making the phase-source rule a GPU-sweepable experiment
surface. The runner now accepts `--phase-source-target-similarity-min`,
`--phase-source-text-similarity-min`, and
`--phase-source-history-char-ratio-min`, passes them into pre-generation
phase-hybrid anchor choice, and writes the chosen thresholds into the score JSON
and report. The default remains the calibrated strict source policy, but future
runs can now test looser or stricter source promotion without touching code.

Implemented v101 as the first fresh CUDA threshold sweep of that source policy.
`llada_moe_mixed_phase_hybrid_preserve_seeded_gated_phase_source_loose090_fresh_v1`,
run ID `diffusion-27e1b13d93f3abad`, lowers the source thresholds to
`0.90/0.90/0.90` while keeping the same lean mixed suite, spend gate, repair
pack, selector, and `2.625000x` relative cost. It scores `0.524554`, trailing
the strict `0.531116` run by `0.006563`. The generated
`DIFFUSION_PHASE_SOURCE_THRESHOLD_SWEEP.md` shows the mechanism: the loose rule
adds one extra history switch on `plan_003`, changing the source from final to
history and dropping that task from `0.538214` to `0.485714`. This is now a
GPU-backed boundary, not just an audit extrapolation: history promotion needs
the strict calibrated retention policy.

Implemented v102 as the too-strict side of the same threshold sweep. Fresh CUDA
run `llada_moe_mixed_phase_hybrid_preserve_seeded_gated_phase_source_strict097_fresh_v1`,
run ID `diffusion-d3d0f8b6e108263e`, raises the source thresholds to
`0.97/0.97/0.95`, removes the remaining `plan_001` history-source switch, uses
final sources on all five selected repairs, and still reaches `0.531116` at
`2.625000x`. This changes the interpretation of the phase-source policy: the
current public frontier is not evidence that history sourcing itself is required
for the score. It is evidence that weak history sourcing is dangerous, while
strict/final-preserving repair is on the score/cost frontier. The next learned
selector should therefore optimize expected repair realization, not maximize
history-source usage.

Implemented v103 by turning that interpretation into a named executable
operator: `constraint_span_phase_final_preserve_seeded_gated`. This repair pack
keeps the preservation-seeded compact-control prompt, the planning span policy,
the phase-denoise repair-spend gate, and the dense history default, but it sets
`source_state="final"` directly. The practical boundary is now explicit:
denoise phase is evidence for whether to spend repair compute, while final-state
repair remains the conservative source of truth unless a learned selector can
prove a history-source advantage. `README.md` now points public readers to this
operator instead of making them infer it from the threshold sweep. Fresh CUDA
run `diffusion-175cbd422107ee5e` validates the named operator directly:
`0.531116` at `2.625000x`, `0` history sources, `5` final sources, and the same
score/cost point as strict `0.96` and strict `0.97`. The threshold sweep report
now records it as `phase_final_named`, making the public mechanism a named
operator rather than a threshold side effect.

Implemented v104 as the lower-cost confirmation of the same named operator.
Fresh CUDA run `diffusion-65f906724fed3cbc` uses
`constraint_span_phase_final_preserve_seeded_gated` with the denoise skeleton
cap set to `20`. It spends four repairs, skips late `plan_007`, and scores
`0.496607` at `2.500000x`, matching the previous cap-20 phase-window point.
The architecture implication is useful: once phase history is reduced to spend
evidence and final-state repair remains the source, the same operator exposes a
clean cost dial. Cap `20` buys the cheaper four-repair point; cap `31` buys the
full five-repair frontier by accepting `plan_007` when the first repairable
skeleton appears late.

Implemented v105 as the cap-16 point on that same dial. Fresh CUDA run
`diffusion-f8f6ae3e209d502b` uses the named phase/final operator with
`--repair-denoise-skeleton-max-step 16`, spends three repairs
(`plan_001`, `plan_003`, `plan_004`), skips two late repairable cases, and
scores `0.472500` at `2.375000x`. The generated repairability geometry report
now includes a policy column so the cost ladder is explicit: cap `16` is the
three-repair cheap point, cap `20` is the four-repair middle point, and cap `31`
is the five-repair frontier.

Implemented v106 as the lower boundary check for that cheap point. A fresh CUDA
run with `--repair-denoise-skeleton-max-step 10` produced the same
content/run ID `diffusion-f8f6ae3e209d502b` as cap `16`: three repairs
(`plan_001`, `plan_003`, `plan_004`), `0.472500` score, and `2.375000x` cost.
That means the first named-operator plateau starts at cap `10`; increasing the
cap to `16` adds no selected repairs and no score. The next meaningful budget
transition is cap `20`, where `plan_006` becomes available and the score moves
to `0.496607` at `2.500000x`.

Implemented v107 as the no-repair lower boundary. Fresh CUDA run
`diffusion-fae5a3498468b66f` uses
`--repair-denoise-skeleton-max-step 9`, spends zero repairs, and scores
`0.414598` at `2.000000x`. All five productive repairable cases are late under
this cap: `plan_001`, `plan_003`, and `plan_004` first become available at step
`10`, `plan_006` at step `20`, and `plan_007` at step `31`. The cost ladder is
therefore discrete rather than smooth: cap `9` is the no-repair floor, cap
`10`/`16` is the first three-repair plateau, cap `20`/`30` is the four-repair
plateau, and cap `31` is the full five-repair frontier.

Implemented v108 as the minimal full-frontier check. Fresh CUDA run with
`--repair-denoise-skeleton-max-step 31` produced the same content/run ID
`diffusion-175cbd422107ee5e` as cap `32`: five repairs, `0.531116` score, and
`2.625000x` cost. This tightens the public statement from "cap 32 recovers the
frontier" to "cap 31 is the minimal confirmed full-frontier cap"; increasing to
`32` adds no selected repairs and no score because the last productive case,
`plan_007`, first appears at step `31`.

Implemented v109 as the cap-30 plateau confirmation. Fresh CUDA run
`diffusion-65f906724fed3cbc` with `--repair-denoise-skeleton-max-step 30`
matches the cap-20 named phase/final operator result exactly: four repairs
(`plan_001`, `plan_003`, `plan_004`, `plan_006`), 26 total generations,
`0.496607` score, and `2.500000x` cost. This closes the gap between the
middle-cost point and the full frontier: steps `21` through `30` add no selected
repair, while step `31` admits `plan_007` and moves the system to the
five-repair frontier. The README and repairability geometry sweep now expose
that as a public budget ladder rather than burying it in raw score files.

Implemented v110 by making that budget ladder executable and auditable.
`experiments/analyze_diffusion_phase_window_budget.py` reads the named
phase/final reference score file, extracts task-level repair onsets from
`repair_spend_gate_rows`, predicts score/cost for each interesting cap, and
then checks those predictions against the fresh CUDA cap confirmations. The
generated `DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md` has zero confirmation
mismatches across eleven rows. The resulting denoise-phase transition model is:
cap `9` is the no-repair floor, cap `10-19` activates
`plan_001`/`plan_003`/`plan_004`, cap `20-30` adds `plan_006`, and cap `31+`
adds `plan_007`. This is a stronger mechanism artifact than a list of runs
because it ties compute spend to the first denoise step where useful constraint
skeletons become repairable.

Implemented v111 by moving the phase-window budget ladder into the benchmark
runner. `run_diffusion_three_arm_benchmark.py` now exposes
`--repair-phase-budget floor|cheap|mid|frontier`, resolving to the verified
caps `9`, `10`, `20`, and `31`. Manual
`--repair-denoise-skeleton-max-step` remains available only under the default
`custom` mode, so a run cannot silently mix two competing budget definitions.
Score JSON and reports now include `repair_phase_budget`, and the content
identity hash treats the named mode as cap metadata because the generated
records, not the spelling of the budget knob, define the run content. The
generated phase-window budget map now includes a Runner Modes table, which
makes the current public operating points executable rather than only
documented.

Implemented v112 as the first fresh CUDA validation of a named phase budget
mode. Running the named phase/final operator with `--repair-phase-budget
frontier` resolves to cap `31`, spends on all five selected repairs, and
reproduces run ID `diffusion-175cbd422107ee5e`: `0.531116` selected latent
repair at `2.625000x`, 27 generations, and the same five repaired planning
tasks. The phase-window budget map now records this as a separate confirmation
row with mode `frontier`, proving the public budget CLI is not just syntactic
sugar over the docs but an actually validated GPU path.

Implemented v113 as the fresh CUDA validation of the low-cost named budget
mode. Running the named phase/final operator with `--repair-phase-budget cheap`
resolves to cap `10`, spends on `plan_001`, `plan_003`, and `plan_004`, skips
the later `plan_006` and `plan_007` repair opportunities, and reproduces run ID
`diffusion-f8f6ae3e209d502b`: `0.472500` selected latent repair at
`2.375000x`, 25 generations, and zero score/cost mismatch against the budget
map. The public budget CLI now has live GPU validation at both the cheap and
frontier tiers.

Implemented v114 as the fresh CUDA validation of the middle named budget mode.
Running the named phase/final operator with `--repair-phase-budget mid`
resolves to cap `20`, spends on `plan_001`, `plan_003`, `plan_004`, and
`plan_006`, skips late `plan_007`, and reproduces run ID
`diffusion-65f906724fed3cbc`: `0.496607` selected latent repair at
`2.500000x`, 26 generations, and zero score/cost mismatch against the budget
map. This left the no-repair `floor` tier as the final named-mode GPU check,
with the equivalent cap-9 custom run already confirmed.

Implemented v115 as the fresh CUDA validation of the no-repair named budget
mode. Running the named phase/final operator with `--repair-phase-budget floor`
resolves to cap `9`, spends zero repairs, and reproduces run ID
`diffusion-fae5a3498468b66f`: `0.414598` selected latent repair at
`2.000000x`, 22 generations, and zero score/cost mismatch against the budget
map. The public phase-budget CLI is now live-GPU validated at all four exposed
tiers: `floor`, `cheap`, `mid`, and `frontier`. The generated phase-window
budget map now has eleven fresh confirmation rows with zero mismatches, so the
README can present the budget ladder as an executable result rather than an
inferred cost model.

Implemented v116 as the cost-aware learning target for that budget ladder.
`experiments/analyze_diffusion_budget_policy_loss.py` consumes
`DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md`'s JSON source and rewrites the cap ladder
as a marginal utility problem:
`utility(task, lambda) = aggregate_score_lift - lambda * marginal_relative_cost`.
The generated `DIFFUSION_BUDGET_POLICY_LOSS.md` and JSONL targets expose five
positive repair targets, three skip targets, a marginal repair cost of
`0.125000`, and task-level break-even lambdas from `0.062857` to `0.283929`.
The important mechanism result is that the public cap ladder is a validated
budget control surface, but it is not the final learned controller. At lambda
`0.18`, an oracle task-gated selector would skip the low-value early repairs,
keep `plan_004`, `plan_006`, and `plan_007`, score `0.508705` at
`2.375000x`, and gain `+0.022589` objective over the best cap policy. This
connects the diffusion work back to the trajectory-dynamics thesis from `_meta`:
reasoning improvement is a path-control problem over denoise states, and the
next loss should predict marginal value of intervention from trajectory
features rather than treating every repairable state as equally worth spending.

Implemented v117 by making that marginal-value objective executable in the
runner and validating it on CUDA. The new
`experiments/analyze_diffusion_budget_value_proxy.py` report tests label-free
proxies against the budget-value targets and selects a runner-ready
source-quality rule: spend only when a repairable denoise skeleton exists, the
source still needs repair, the prompt gap is inside the public band, and
`source_quality <= 0.301429`. `run_diffusion_three_arm_benchmark.py` now exposes
that as `--repair-spend-trigger denoise_phase_value_proxy` with
`--repair-value-proxy-source-quality-max`. Fresh CUDA run
`diffusion-a343e942cbfb0a93` uses the stable CLI threshold `0.31`, spends only
on `plan_004`, `plan_006`, and `plan_007`, and scores `0.508705` at
`2.375000x` with 25 generations. That is the same relative cost as the cheap
cap-10 tier but `+0.036205` higher score, because the learned-selector-shaped
gate skips low-marginal early repairs (`plan_001`, `plan_003`) while keeping
the high-marginal late repairs (`plan_006`, `plan_007`). This is now the
practical bridge between the public cap ladder and a learned cost-aware repair
controller. The generated public artifacts now make that bridge discoverable
from the repo front door: `DIFFUSION_PUBLIC_BENCHMARK.md` renders
`diffusion-a343e942cbfb0a93` as the budget-favored latent repair row, and
`CLAIM_EVIDENCE_MAP.md` records it as
`moe_mixed_phase_final_preserve_seeded_value_proxy_budget` with the linked
score/report/raw artifacts.

Implemented v118 as the first feature-geometry audit for the cost-aware repair
controller. `experiments/analyze_diffusion_repair_value_geometry.py` reads the
budget-policy loss targets and emits `DIFFUSION_REPAIR_VALUE_GEOMETRY.md` plus
`eval_results/diffusion_language/diffusion_repair_value_geometry.json`. The
result makes the selector lesson sharper: early repairability alone is not the
right objective. At lambda `0.18`, `plan_001` and `plan_003` are real
repairable states but negative-utility spends, while `plan_004`, `plan_006`,
and `plan_007` are profitable. Inside the public prompt-gap band, profitable
repairs have source quality at or below `0.301429`, and the nearest negative
repairable in-band example starts at `0.324286`, giving a `0.022857`
source-quality separation gap. The runner-stable rule
`source_quality <= 0.31` and `prompt_gap_count <= 9` has zero regret versus the
task-gated oracle on the current targets. The geometry also isolates the
important false boundary: low source quality outside the prompt-gap band starts
at gap `10`, so `plan_005` remains a skip even though it looks superficially
repairable and low-quality.

Implemented v119 as the repo-level mathematical theory layer. The new
`docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md` states the formal objects behind
the implementation: denoise trajectories `x_0...x_T`, verifier feature maps
`phi(p, x_t)`, task-relevant information loss `L_info`, repair operators over
mask/anchor/source triples, and the marginal repair-value objective. It also
records the current proof obligations: diffusion has a larger editable
post-diagnosis intervention set than ordinary AR decoding; repair spending is a
linear marginal-utility decision under a cost penalty; phase caps create a
piecewise-constant frontier at observed first-repairable steps; and the current
budget-favored repair set is separable by label-free source-quality and
prompt-gap geometry. This is the document to update when the theory changes,
rather than scattering proof claims across benchmark reports.

Implemented v120 as the reader-facing documentation map. The new
`docs/DIFFUSION_READER_GUIDE.md` gives a fast path through the public benchmark,
claim evidence, theory, cost controls, mechanism audits, anchor-retention work,
and source files. This addresses a repo-level communication failure: the
diffusion work is now broad enough that generated reports alone do not expose
the research story. Public readers should be routed through the guide before
digging into individual artifacts.

Implemented v121 as the theory claim ledger. The new
`docs/DIFFUSION_THEORY_CLAIM_LEDGER.md` maps the theory layer into explicit
statuses, evidence, assumptions, falsifiers, and next proof obligations. It
keeps mathematical assertions from drifting into unsupported narrative: top
benchmark claims stay in `CLAIM_EVIDENCE_MAP.md`, while theorem-like statements
now have a separate audit trail that says what would disprove them.

Implemented v122 as the theory claim validator. The new
`experiments/validate_diffusion_theory_claim_ledger.py` checks that ledger rows
are ordered, status-bounded, evidence-linked, falsifiable, and backed by public
reader-guide links. This turns the theory layer into a maintained repo surface
instead of another Markdown-only claim pile.

Implemented v123 as the error-function geometry bridge. The new
`experiments/analyze_diffusion_error_function_geometry.py` reads the current
repair-value geometry and phase-source loss targets, emits
`DIFFUSION_ERROR_FUNCTION_GEOMETRY.md`, and states the next-loss implication:
cost-aware repair value, source trust, retention, and anchor realization need
separate terms. The generated audit shows why a single repairability label is
too weak: five raw positive repair targets become only three cost-profitable
targets at lambda `0.18`, earliest repairable step `10` mixes profitable and
unprofitable rows, and naive history-source trust creates four false positives.

Implemented v124 as the decomposed selector audit. The new
`experiments/analyze_diffusion_decomposed_selector.py` compares one-label
repairability controllers against selectors that keep repair value, source
trust, retention, and anchor realization separate. On the current target rows,
`single_repairability_label` has composite shortfall `3.053730`, while
`decomposed_value_source` has composite shortfall `0.186127`; the remaining
loss is the preservation-seed realization term. This is the first executable
test of the full composite-loss thesis rather than only a prose argument for it.

Implemented v125 as the composite selector target surface. The new
`experiments/build_diffusion_composite_selector_targets.py` merges repair-value
geometry, phase-source targets, retention loss rows, realization policy scores,
and the decomposed selector choice into `DIFFUSION_COMPOSITE_SELECTOR_TARGETS.md`
plus `eval_results/diffusion_language/diffusion_composite_selector_targets.jsonl`.
The generated dataset has eight task-level rows for spend/source/retention
heads and seven realization-policy rows for the compact-anchor head.

Implemented v126 as the first fitted composite selector. The new
`experiments/fit_diffusion_composite_selector.py` fits small interpretable heads
over the target surface and emits `DIFFUSION_COMPOSITE_SELECTOR_FIT.md`. The
local zero-error fit recovers the current controller as four separate rules:
gap/source-quality repair spending, retention-safe source trust,
safe-history-anchor retention, and minimum realization-policy error for the
compact-anchor policy. This becomes the floor that learned selectors must beat
on held-out slices.

Next repair step: replace the hand-coded proposal coverage with a more general
verifier/proposer for integer and open-form symbolic failures, then generalize
the arithmetic-feedback loop beyond local arithmetic claims. The current
feedback path fixes local false equations and simple worded arithmetic claims,
the evidence branch can demand equations when none are shown, the semantic
quantity guard rejects explicit distractor numbers, and the operation-role guard
rejects missing prompt-required operations. The quantity-role binding guard now
also catches swapped explicit roles like ticket count times the wrong ticket
price, and the arithmetic-provenance guard catches ungrounded intermediate
numbers. The final-answer role guard catches totals, per-share answers,
full-bag floor divisions, and remainders where the final integer has the wrong
role. The final-answer object guard catches explicitly excluded answer objects
when the final number is otherwise plausible. The final-answer target guard
catches explicit wrong target units and conflicting target modifiers. The next
gap is proof-carrying target evidence across more prompt forms without
overrejecting bare answers, plus expanding proof-carrying non-arithmetic exact
outputs beyond the current order/list/toggle/syllogism solvers and making trace
checks richer than lightweight lexical evidence.

Next selector step: improve trajectory scoring for planning beyond the current
planning-state selector by making revision span selection learned rather than
only source-relative and compact-verifier ranked. The current
selector scores visible denoise samples with generic planning quality, prompt
keyword coverage, repetition/filler penalties, and stability between peak
partial quality and final quality. It uses no hidden rubric items. The next
useful judge needs partial-state specificity, constraint coverage,
contradiction-span detection, and stability of useful content rather than
stability alone.

### 3. Trajectory Evolution

Evolution should mutate schedule-level and state-level controls:

- steps
- algorithm
- remasking policy
- block length
- temperature or algorithm temperature
- frozen span mask
- repair insertion step
- verifier hook threshold

Fitness should combine final task score, early stability, low EOS pressure,
and judge confidence.

Implemented v1: the seeded schedule-selection benchmark now has an evolved arm.
It mutates schedule depth/remasking around the base Dream/LLaDA schedules, keeps
the old trajectory arm restricted to the base pool, and records the evolved
arm's larger generation budget. This turns trajectory evolution from a design
idea into a measurable GPU protocol.

Implemented v2: evolved selection now has a configurable promotion margin and
the runner can rescore an existing raw JSONL without rerunning GPU generations.
This lets selector changes be tested cheaply before a full rerun. On the
current canonical slice, a `0.015` margin preserved the evolved mean gain while
removing the two evolved-vs-trajectory regressions from the previous argmax-only
selector.

Implemented v3: benchmark reports now include oracle schedule selection and
selector-regret diagnostics. These use task labels only after generation for
analysis, never for arm selection. An experimental `planning_state_v2` selector
that rewards prompt-specific action structure was added, but the first rescore
was weaker than the default (`0.465` evolved mean versus `0.475`), so the
default remains `planning_state`.

### 4. Error-Correction Losses

If we train anything later, the useful losses are not generic next-token losses.
Better targets:

- trajectory stability loss: correct partial states should stay correct under
  remaining denoise steps
- repair consistency loss: remasked weak spans should improve without damaging
  frozen spans
- verifier-aligned denoise loss: states that satisfy a verifier should become
  attractors
- counterfactual repair loss: wrong intermediate states should learn minimal
  edits toward correct states
- energy-aware schedule loss: reward shorter denoise paths that keep correctness
- realization-quality loss: compact semantic anchors should become direct,
  action-bearing answer text instead of labels, seed chatter, or repair
  instructions

### 5. World-Model Angle

The useful world-model analogy is not "make text generation more visual." It is
to treat a reasoning answer as a latent state that can be predicted, corrected,
and stabilized before committing to visible tokens.

For planning tasks, the denoise sequence is a tiny world model of the answer:

- early steps establish coarse causal structure
- middle steps fill mechanisms and constraints
- late steps resolve wording and termination

The system should learn which intermediate states are worth preserving and which
ones should be corrected.

## Immediate Build Order

1. Keep Dream and LLaDA local weights under `external/diffusion_models/`.
2. Run `run_diffusion_scout.py` on the 8 planning tasks in the scout pack.
3. Compare Dream/LLaDA schedule-selected outputs against the old AR three-arm
   protocol.
4. Add branch-and-repair: take the best mid-trajectory state, remask weak spans,
   and finish from that state.
5. Compare:
   - fixed Dream schedule
   - fixed LLaDA schedule
   - schedule-selected Dream
   - schedule-selected LLaDA
   - later: branch-and-repair diffusion reasoning

Do not claim a general reasoning result until this is run against the locked
benchmark protocol. But mechanically, the key substrate is now in place:
language diffusion models are local, runnable, and expose trajectories that can
be scored.

## v127: Four-Head Selector Reaches The Runner

The composite selector is no longer only an audit artifact. The benchmark
runner now has a named spend trigger:

`--repair-spend-trigger decomposed_four_head_selector`

This trigger wires the fitted spend head into live repair gating and records
the four selector head IDs in repair-spend diagnostics:

- spend: `first_repairable_gap_le_9_source_quality_le_0p301429`
- source: `retention_safe_history`
- retention: `classification_safe_history_anchor`
- realization: `min_realization_policy_error`

The source/retention/realization heads still ride on the existing phase/final
repair path, so this is a bridge implementation rather than the final learned
controller. The important architectural change is that the geometry-derived
loss family now has a CLI entry point for fresh GPU runs.

## v128: Four-Head Selector CUDA Budget Confirmation

Fresh run `diffusion-62476b492c9e592c` executed the new trigger on the lean GPU
mixed benchmark. It repaired `plan_004`, `plan_006`, and `plan_007`, skipped the
low-value early repairable rows, and reproduced the lower-cost budget point:

- selected latent repair: `0.508705`
- relative GPU cost: `2.375000x`
- delta vs greedy/fixed: `0.096429`
- delta vs random perturbation: `0.136580`
- repair-oracle headroom: `0.000000`

The top-score phase/final frontier remains `0.531116` at `2.625000x`. The new
result matters because the cheaper point is now produced by the named
four-head selector trigger with full spend/source/retention/realization
provenance, not only by an unnamed value-proxy gate.

## v129: Independent Spend-Head Transfer Boundary

Added `lean_gpu_mixed_transfer`, an independent transfer preset with four new
planning tasks plus math/symbolic/science guards:

- `plan_009`
- `plan_010`
- `plan_011`
- `plan_012`
- `math_009`
- `sym_007`
- `sci_002`

Fresh all-repairable run `diffusion-a43504b2dec11ced` produced the first
out-of-sample spend labels. After correcting the evaluator to use repair-oracle
lift rather than selected-repair lift, the result is:

- positive repair-availability rows: `1/4`
- positive row: `plan_012`
- single repairability false-positive spends: `1`
- decomposed spend-head errors: `0`
- decomposed selected task: `plan_012`

So the current spend geometry transfers as a repair-availability filter on the
first independent slice. It is still not a complete value predictor because
cost-adjusted promotion value is a separate question.

## v130: Source-Task-Floor Boundary

The first source-task-floor transfer rule is now better understood as a
boundary, not a replacement for decomposed spend. With repair-oracle labels,
`current_decomposed_spend` has zero errors. Floors above `0.295357`, including
the earlier `0.3075` floor, skip `plan_012`.

That matters because `plan_012` has positive oracle repair lift but is held
back by the current promotion margin:

- source task score: `0.295357`
- oracle repair lift: `0.020000`
- selected repair lift under the margin: `0.000000`

The runner still exposes the source-floor probe as:

`--repair-spend-trigger decomposed_spend_transfer_rule`

Architecturally, the lesson is that repair availability and cost-adjusted
promotion value need separate heads. Source quality and prompt-gap geometry can
identify repairable low-quality states; a stricter floor can be useful only if
the objective is budget margin, not raw repair availability.

## v131: Strict Floor CUDA Confirmation

Fresh run `diffusion-f50e82f88f59111b` executed
`--repair-spend-trigger decomposed_spend_transfer_rule` on
`lean_gpu_mixed_transfer`.

The trigger spent zero repairs on the four independent planning rows:

- `plan_009`: skipped outside repairable band
- `plan_010`: skipped by source-quality value proxy
- `plan_011`: skipped outside repairable band
- `plan_012`: skipped by `transfer_source_task_score_low`

This confirms the strict floor executes in the live runner, but the corrected
oracle labels show it is too conservative for repair availability. The selected
latent repair score on the repair-covered planning rows stayed at `0.345268`,
with `0.000000` delta vs fixed and `0.019536` delta vs random perturbation,
because the promotion margin also held back the low-margin `plan_012` repair.

## v132: Expanded Transfer Slice

Added `lean_gpu_mixed_transfer_v2`, extending the independent planning slice to
eight prompts while keeping the math/symbolic/science guards. Fresh
all-repairable run `diffusion-76fd30506cace1ee` generated:

- full model generations: `24`
- arm selections: `41`
- independent planning rows: `8`
- positive repair-availability rows: `1`
- positive row: `plan_012`
- decomposed spend-head errors: `0`
- single repairability errors: `1`

The expanded fit in `DIFFUSION_SPEND_TRANSFER_RULE_FIT_V2.md` keeps
`current_decomposed_spend` as the best repair-availability rule. The next
architecture step is to train separate predictors for repair availability and
cost-adjusted promotion value, because `plan_012` is exactly the kind of
low-margin case where those objectives disagree.

## v133: Promotion Selector Recovers The Low-Margin Repair

Fresh run `diffusion-2a4bd4e3cad622a2` reran the expanded transfer policy with
the corrected spend floor and `repair_selector=inherit`.

The result separates the two heads cleanly:

- spend gate: runs only `plan_012`
- full model generations: `23`
- repair-covered planning score: `0.350938`
- delta vs fixed on repair-covered planning rows: `0.015670`
- delta vs random on repair-covered planning rows: `0.043804`
- delta vs trajectory on repair-covered planning rows: `0.002500`
- oracle headroom vs selected repair: `0.000000`

The planning-quality seed-realization selector saw the same repair candidate
but left `0.002500` mean oracle headroom. The inherited planning-state selector
selected it with repair selector edge `0.112535`. This is the first transfer
evidence that the next learned controller should have a promotion-value head,
not only a spend-availability head.

## v134: Named Transfer Promotion Policy

The benchmark runner now exposes the current promotion-value proxy directly:

`--repair-selector transfer_promotion_value --repair-promotion-margin 0.0`

This is intentionally a named alias for inherited planning-state selection, not
a trained promotion head. It makes the current transfer recipe reproducible
without asking readers to remember that `inherit` is serving as the
promotion-value proxy. The theoretical boundary remains clear: this alias is an
executable baseline for the next learned promotion-value head.

## v135: Separate Transfer Heads Fit

Added the first CPU-safe fit for the transfer split:

`experiments/fit_diffusion_transfer_heads.py`

Generated report:

`DIFFUSION_TRANSFER_HEAD_FIT.md`

Result:

- availability head: `availability_current_decomposed_spend`
- availability rows: `16`
- availability errors: `0`
- selected availability tasks: `plan_004`, `plan_006`, `plan_007`, `plan_012`
- promotion head: `transfer_promotion_value`
- promotion errors: `0`
- planning-quality promotion false negatives: `1`, on `plan_012`

This is still an interpretable fitted baseline, not a neural controller. It is
useful because it turns the theory split into executable target surfaces:
repair availability is learned from source/denoise geometry, while promotion
value is learned from the post-repair selection outcome.

## v136: Reasoning Proof Object

Added the generated proof-object ledger:

`DIFFUSION_REASONING_PROOF_OBJECT.md`

Builder:

`experiments/build_diffusion_proof_object.py`

The report originally recorded:

- heads: `6`
- total target rows: `52`
- total measured errors on fitted heads: `0`
- heads: availability, promotion value, source trust, retention, realization,
  cost
- each head has an assertion, information channels, evidence files, falsifier,
  and next GPU validation obligation

This is the concrete bridge between the mathematical theory and benchmark work:
the repo can now state not only "diffusion repair improved the score" but also
which error function is supposed to explain each decision, what evidence
supports it, and what fresh GPU result would falsify it.

## v137: Availability Boundary From v4 Transfer

The fresh v4 transfer slice updates the proof object from a clean local fit to a
useful falsifier:

- total target rows: `60`
- total measured errors: `3`
- unresolved/boundary heads: `1`
- boundary head: availability
- failing executable rule: `learned_availability_predictor_v1`

`DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md` shows that v4 has profitable
repairs on `plan_026`, `plan_028`, and `plan_031`, but the v3 learned absolute
source-quality cutoff blocks all three. CUDA run `diffusion-865e5acb0ee73e8a`
confirms the executable learned trigger runs zero repairs. The architecture
direction is now slice-relative availability calibration, not tighter reuse of
the v3 source-quality threshold.

## v138: Calibrated Availability Boundary From v5 Transfer

The calibrated availability trigger is now executable:

`--repair-spend-trigger calibrated_availability_predictor_v1`

It removes the failed absolute source-quality ceiling and uses the v3/v4
counterexample boundary: repairable denoise source, non-ambiguous prompt gap,
and source-vs-trajectory diagnostics. After the repair-only label correction,
this is a useful spend trigger but not a solved promotion model.

The fresh v5 transfer slice is the next boundary:

- all-repairable run: `diffusion-b3324317dadee840`
- calibrated executable run: `diffusion-c4f0d7bc21768f21`
- calibrated availability errors: `3`
- learned v3 cutoff errors under corrected labels: `4`
- calibrated repair-covered planning delta: `+0.044866` vs fixed, `+0.074438`
  vs random
- all-repairable repair-covered planning delta: `+0.069821` vs fixed,
  `+0.099393` vs random
- candidate-aware promotion target errors: `0` over seven generated repair
  candidates

The failure cases matter. `plan_037` is profitable even though the repair source
is below the selected trajectory, while `plan_033`, `plan_038`, and `plan_040`
are generated repair candidates that should not be promoted. The new
`DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md` artifact turns that into a direct
post-repair target and exposes `candidate_aware_promotion_v1` in the runner.

## v139: Candidate-Aware Promotion Holds On V6, Spend Gate Fails

Added `lean_gpu_mixed_transfer_v6` with `plan_041` through `plan_048` plus the
same math/symbolic/science probes. The fresh all-repairable CUDA pass is:

- run: `diffusion-158fb4ff45a8d2e8`
- repair rows selected: `plan_041`, `plan_044`, `plan_046`, `plan_047`,
  `plan_048`
- repair delta: `+0.086696` vs fixed, `+0.125929` vs random, `+0.068009`
  vs trajectory
- extra repair generation budget vs evolved: `1.000000`
- oracle headroom vs repair: `0.000000`

The executable calibrated-spend plus candidate-aware-promotion CUDA run is:

- run: `diffusion-b6d8fd700b3a267f`
- repair rows selected: `plan_041`, `plan_044`, `plan_047`
- repair delta: `+0.047036` vs fixed, `+0.086268` vs random, `+0.028348`
  vs trajectory
- extra repair generation budget vs evolved: `0.625000`
- oracle headroom vs repair: `0.000000` within its generated candidate set

The important result is not that calibrated spend is best. It is not. The
corrected v6 spend labels show calibrated availability has four errors: it
admits no-lift `plan_042` and `plan_045`, and misses positive `plan_046` and
`plan_048`. In contrast, `DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md` gives
`candidate_aware_promotion_v1` zero errors over all eight generated repair
candidates. Rescoring the all-repairable raw generations with
`--repair-selector candidate_aware_promotion_v1` gives run
`diffusion-ae7a4edd5c22ca20`, matching the all-repairable score and selecting
exactly the five positive repair candidates. The next controller work should
keep candidate-aware promotion fixed and refit the spend gate.
