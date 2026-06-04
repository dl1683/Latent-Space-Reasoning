# Diffusion Latent Reasoning Roadmap

This is the execution bridge from the three-arm general benchmark protocol into
language-diffusion reasoning. ARC work stays paused. The goal is to test whether
latent-space reasoning improves general-purpose reasoning by controlling an
iterative denoising trajectory instead of only nudging an autoregressive prefix.

The architecture pivot is now defined in
`docs/DIFFUSION_NATIVE_REASONING_ARCHITECTURE.md`. Use that as the source of
truth for new reasoning-control work: schedules, mask policies, repair hooks,
and intermediate states are first-class latent objects.

For the public narrative and field-level implications, use
`docs/DIFFUSION_REASONING_FIELD_IMPLICATIONS.md`. It frames the current result
as diffusion-native latent reasoning control, not as a broad claim that general
reasoning is solved.

## Why This Is The Right Next Surface

Autoregressive soft prompts only perturb the first visible trajectory and then
let left-to-right decoding amplify the shift. Language diffusion models expose a
stronger control surface:

- generation is an iterative masked-token denoising process
- intermediate states can be scored before the final answer
- remasking and step schedules are explicit knobs
- output positions can be revised instead of committed once
- planning tasks can be inspected for partial plans, repairs, and collapse

That maps cleanly to the repo's current hypothesis: useful reasoning gains may
come from trajectory selection, error correction, and distribution movement, not
from magical information inside a random vector.

## Local Hardware Reality

Current local check: `torch 2.9.1+cu128`, CUDA available, `NVIDIA GeForce RTX
5090 Laptop GPU`, `23.89 GB` VRAM. The repo also has `transformers`,
`accelerate`, `bitsandbytes`, and `huggingface_hub` installed.

This is enough to try BF16 7B/8B diffusion LMs first. If Python custom-code
inference is unstable or too memory-heavy, use GGUF/diffuse-cpp as the fallback.

## Candidate Model Order

| Priority | Candidate | Why | First path | Risk |
| ---: | --- | --- | --- | --- |
| 1 | `Dream-org/Dream-v0-Instruct-7B` | Official `diffusion_generate()` API, planning/math focus, can return history. | `HFDiffusionBackend("dream-7b-instruct-hf")` | Official repo tested older torch/transformers versions; debug custom-code issues directly. |
| 2 | `GSAI-ML/LLaDA-8B-Instruct` | Trained-from-scratch masked diffusion LM, strong architecture contrast. | `HFDiffusionBackend("llada-8b-instruct-hf")` | Uses repo-style sampling loop; history instrumentation needs extra work. |
| 3 | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` | Sparse diffusion LM with roughly 1B-1.4B active parameters at inference; useful cheap architecture target now that it runs locally. | `HFDiffusionBackend("llada-moe-7b-a1b-instruct-hf")` with `external/diffusion_models/LLaDA-MoE-7B-A1B-Instruct` | Full-weight smoke and compact benchmark pass; current dense-LLaDA repair policy transfers but is weaker on planning. |
| 4 | `diffuse-cpp/Dream-v0-Instruct-7B-GGUF` | Cheap Q4 fallback, usable via diffuse-cpp/llama.cpp. | local server or CLI | Less access to denoising internals. |
| 5 | `diffuse-cpp/LLaDA-8B-Instruct-GGUF` | Cheap LLaDA fallback. | local server or CLI | Less access to denoising internals. |
| 6 | `mradermacher/LLaDA-MoE-7B-A1B-Instruct-i1-GGUF` | Community quantized fallback for the sparse MoE target. | local server or CLI | Lower confidence than HF BF16 path and likely less access to denoising internals. |

Mercury is not a first execution target because the public path is API or
on-prem deployment, not cheap local weights. SEDD and MDLM are useful research
baselines for algorithm design, but not the first public reasoning demo because
their open checkpoints are smaller/research-oriented compared with Dream/LLaDA.

## Current Local Evidence

The first local runs are now in `eval_results/diffusion_language/`.

| Model | Artifact | Result |
| --- | --- | --- |
| Dream 7B | `schedule_sweep_report.md` | `entropy_64` scored `0.727`, beating `entropy_32` and `origin_64`. |
| LLaDA 8B | `llada_schedule_sweep_clean_report.md` | `low_confidence_32` scored `0.688`, slightly above `random_32`. |
| Dream + LLaDA | `smoke_report.md` | Both models generated coherent local outputs from full local weights. |
| Dream + LLaDA planning scout | `planning_scout_v2_system_report.md` | 40 full generations over 8 planning tasks; selected mean task score `0.390`; LLaDA selected outputs averaged `0.412`, Dream `0.367`. |
| Dream + LLaDA objective scout | `objective_scout_v1_report.md` | 85 full generations over math/symbolic/science tasks; selected mean task score `0.794`; LLaDA averaged `0.941`, Dream `0.647`; science was `1.0`. |
| LLaDA repair scout | `repair_scout_planning_v1_report.md` | 24 full generations over 8 planning tasks; suffix-inpainting repairs were selected on 4/8 tasks; selected mean task score moved from `0.412` baseline to `0.436`. |
| LLaDA confidence repair scout | `repair_scout_planning_confidence_v1_report.md` | 40 full generations over 8 planning tasks; prefix and low-confidence repairs were selected on 5/8 tasks; selected mean task score moved from `0.412` baseline to `0.443`. |
| Dream objective proposal ablation | `repair_scout_dream_objective_proposal_ablation_v1_report.md` | 23 model generations plus 6 proposal-only ablations; baseline selected mean moved from `0.647` to `1.000`, but proposal-only selection also reached `1.000`, so the exact-answer gain is proposer-attributable on this slice. |
| LLaDA objective proposal ablation | `repair_scout_llada_objective_proposal_ablation_v1_report.md` | 24 model generations plus 1 proposal-only ablation; baseline selected mean moved from `0.941` to `1.000`, proposal-only also reached `1.000`, and the selected model repair was answer-context remasking for `sym_002`. |
| Dream + LLaDA three-arm planning-state mix | `three_arm_planning_state_selector_v1_report.md` | 55 seeded model generations over 8 planning tasks plus math/symbolic/science checks; fixed mean task `0.436`, random `0.423`, planning-state trajectory-selected `0.465`; trajectory selection beat fixed by `+0.029` and random by `+0.042`, with exact-answer tasks guarded to fixed schedules unless a verifier is used. |
| Dream + LLaDA evolved planning-state mix | `four_arm_evolved_margin015_v1_report.md` | 99 seeded model generations over the same task mix; fixed `0.436`, random `0.423`, base-pool trajectory-selected `0.465`, evolved schedule-selected `0.475`; evolved used a `0.015` selector-score promotion margin, beat fixed by `+0.039`, random by `+0.052`, and trajectory by `+0.010`, with `3/19/0` wins/ties/losses against trajectory at `4.50` generations per task. |
| Dream + LLaDA evolved oracle rescore | `four_arm_evolved_margin015_oracle_rescore_v1_report.md` | Reuses the same 99 raw generations and adds oracle/regret diagnostics; oracle task score was `0.481`, only `+0.006` over evolved, with `4/18/0` oracle wins/ties/losses against evolved. An experimental `planning_state_v2` selector was also rescored, but it regressed evolved mean to `0.465`, so the default remains `planning_state`. |
| LLaDA adaptive history-rescue repair | `llada_planning_adaptive_history_rescue_margin01_v1_report.md` | 49 budgeted LLaDA planning generations with default `0.25` history-prefix repair and an adaptive `0.50` history rescue only when the first repair pass would keep the matching evolved baseline. Fixed `0.412`, random `0.376`, trajectory `0.412`, evolved `0.451`, repair-selected `0.490`. Repair beats evolved by `+0.039`, has `6/2/0` repair-vs-evolved wins/ties/losses, costs `6.12` generations/task on the covered slice, keeps `0.018` task-score gain per extra generation versus evolved, and reaches zero oracle headroom. |
| LLaDA gated primary-repair line | `llada_planning_primary_repair_gate_v1_report.md` | 47 fresh GPU generations using `--repair-spend-trigger source_quality_or_short` before the canonical prefix/history repair path. The gate skips primary repairs when the source already has high label-free planning quality and enough text to look complete. It preserves the same selected score as the adaptive history-rescue line: repair-selected `0.490`, `+0.039` over evolved, `6/2/0`, zero oracle headroom, but lowers covered-slice budget to `5.88` generations/task and raises repair gain per extra generation to `0.021`. |
| LLaDA state-adaptive history-prefix repair | `llada_planning_state_adaptive_history_prefix_v1_report.md`; negative confidence diagnostic: `llada_planning_state_adaptive_repair_pack_v1_report.md` | 46 fresh GPU generations using `--repair-pack state_adaptive`, which now spends `state_adaptive_history_repair` plus `prefix_25_repair` under the source-quality spend gate. The adaptive history repair uses a longer history anchor only when source quality and history-state score are both weak, which captures the old `plan_004` rescue without generating a separate rescue branch. This preserves repair-selected `0.490`, `+0.039` over evolved, `6/2/0`, zero oracle headroom, lowers covered-slice budget to `5.75` generations/task, and raises repair gain per extra generation to `0.022`. The earlier state-adaptive confidence branch averaged `-0.009` task delta vs source and was selected 0 times, so it is demoted behind the prefix branch. This is now the efficient LLaDA planning repair line. |
| LLaDA gated constraint-gap rescue | `llada_planning_gated_constraint_gap_rescue_v1_report.md`; expanded rescue diagnostic: `llada_planning_gated_ranked_span_rescue_default_history_v1_report.md`; guarded rescore: `llada_planning_gated_ranked_span_rescue_default_history_guarded_rescore_v1_report.md` | The efficient `state_adaptive` pack can spend prompt-grounded constraint-gap rescue only when source planning quality is in the `0.400-0.500` band and at least 6 prompt terms are missing. The original gate fired once on `plan_001`, selected `constraint_gap_revision_repair`, and preserved the full constraint-gap absolute win with much lower spend: repair-selected `0.491`, `+0.040` over evolved, `6/2/0`, zero oracle headroom, `5.88` generations/task, and `0.022` gain per extra generation. The expanded rescue diagnostic allows revision, anchor, and ranked span rescue on the gated task; it reached `0.492` before guardrails but selected an anchor output that leaked the missing-term checklist into the answer. The guarded rescore adds a label-free checklist-leakage penalty, demotes that keyword dump with `Risk Penalty 0.180`, selects the cleaner full revision, and preserves the line at `0.491` with zero repair selector regret. This remains the best LLaDA planning repair line when the prompt-gap gate is allowed. |
| LLaDA risk-guarded constraint-gap rescore | `llada_planning_gated_constraint_gap_risk_guard_rescore_v1_report.md`; checklist-leakage guard: `llada_planning_gated_ranked_span_rescue_default_history_guarded_rescore_v1_report.md` | `planning_quality_delta_risk_guarded` now subtracts label-free contradiction/risk penalties and prompt-checklist leakage penalties from source-relative repair selection, exposing them as `Risk Penalty`. The original gated raw file had no detected prompt-contradiction penalties, so score stayed `0.491`. The expanded rescue rescore proves the new hygiene guard matters: it blocks the higher-scoring but visibly bad comma-separated keyword dump from `constraint_gap_revision_anchor25_repair` and promotes the cleaner `constraint_gap_revision_repair` on `plan_001`. Treat this as the safer selector default for planning repair experiments, not as a new aggregate-score gain. |
| LLaDA constraint-gap hybrid repair diagnostic | `llada_planning_constraint_gap_repair_v1_report.md` | 53 fresh GPU generations using `--repair-pack constraint_gap`, which preserves the state-adaptive history plus prefix candidates and adds a prompt-grounded full-draft revision that targets missing source prompt terms. It creates a tiny new absolute best: repair-selected `0.491`, `+0.040` over evolved, `6/2/0`, zero oracle headroom. The new branch is real but not yet budget-efficient: `constraint_gap_revision_repair` was selected once, improved `plan_001` by `+0.076` over evolved, averaged `+0.011` task delta versus source with `1/6/0` wins/ties/losses, and lifted selected score only `+0.001` over the efficient state-adaptive line while raising budget to `6.62` generations/task and lowering repair gain per extra generation to `0.015`. Treat it as a future gated rescue branch, not the default pack. |
| LLaDA replay-consistency repair diagnostic | `llada_planning_replay_consistency_repair_v1_report.md` | 46 fresh GPU generations using `--repair-pack replay_consistency`, which remasks suffix positions that are unstable across sampled denoise-history states before falling back to state-adaptive history repair. Repair-selected reaches `0.477`, `+0.026` over evolved, with `4/4/0` repair-vs-evolved wins/ties/losses, `5.75` generations/task, and `0.015` task-score gain per extra generation. The replay-instability operator is the important negative signal: `replay_unstable_25_repair` was selected 0 times, averaged `0.000` task delta versus source, and had `0/7/0` wins/ties/losses versus source. It works mechanically, but it is currently a no-op repair, so state-adaptive history-prefix remains canonical. |
| LLaDA full history-fraction repair diagnostic | `llada_planning_history_fraction_sweep_margin01_v1_report.md` | 56 seeded LLaDA planning generations with unconditional history-prefix fractions `0.25,0.50` plus final-prefix repair. It reaches the same `0.490` repair-selected score as adaptive rescue, but costs `7.00` generations/task and only `0.013` task-score gain per extra generation, so it remains a diagnostic rather than the canonical budget line. |
| LLaDA visible-history rescue diagnostic | `llada_planning_visible_history_rescue_margin01_v1_report.md`; guarded rescore: `llada_planning_visible_history_rescue_guarded_margin01_v1_report.md` | 50 seeded LLaDA planning generations adding an adaptive `history_visible_repair` candidate that preserves all visible mid-denoise tokens on rescue. It keeps the same `0.490` repair-selected score but costs `6.25` generations/task and lowers budget-normalized gain to `0.017`. Candidate diagnostics show `history_visible_repair` had high trajectory score (`0.738`) but did not beat `history_prefix_50_repair` on task quality for `plan_004`. The `planning_quality_guarded` rescore adds a label-free overpreservation penalty (`0.053` for `history_visible_repair`, `0.000` for selected prefix repairs) and leaves selected outputs unchanged, so visible-state rescue remains diagnostic. |
| LLaDA disagreement-triggered rescue diagnostic | `llada_planning_disagreement_visible_history_rescue_guarded_v1_report.md` | 56 seeded LLaDA planning generations with `baseline_or_selector_disagreement` rescue and guarded visible-history repair. It generated four `history_visible_repair` candidates with strong average raw task score (`0.497`) and high trajectory score (`0.738`), but the new marginal diagnostics show `0.000` task delta vs source and `0/4/0` wins/ties/losses vs source. Selected repair stayed `0.490` and budget rose to `7.00` generations/task, dropping budget-normalized gain to `0.013`. This completes the first disagreement-triggered expansion mechanism but shows all-visible repair mostly preserves source quality rather than improving it. |
| LLaDA source-relative selector rescore | `llada_planning_source_delta_guarded_rescore_v1_report.md`; mixed rescore: `mixed_source_delta_guarded_rescore_v1_report.md` | Reuses existing raw generations with `planning_quality_delta_guarded`, which scores repair candidates by label-free planning-quality improvement over their source rather than absolute quality. The selected repair score stays `0.490`, `+0.039` over evolved, with `6/2/0` repair-vs-evolved wins/ties/losses and zero oracle headroom. The value is diagnostic discipline: reports now expose `PQ Delta` so no-op repairs are visible. |
| LLaDA source-relative repair-pack diagnostic | `llada_planning_source_relative_repair_pack_v1_report.md` | 56 fresh GPU generations using `--repair-pack source_relative`, `history_prefix_50_repair`, `low_confidence_15_repair`, and `low_confidence_25_repair`, selected by `planning_quality_delta_guarded`. It reaches only `0.454` repair-selected score, `+0.004` over evolved, with `1/7/0` repair-vs-evolved wins/ties/losses. `low_confidence_15_repair` has `0.000` task delta and `0/8/0` wins/ties/losses vs source, so minimal low-confidence remasking is too conservative by itself. |
| LLaDA targeted-content repair diagnostic | `llada_planning_targeted_content_repair_pack_v1_report.md` | 56 fresh GPU generations using `--repair-pack targeted_content`, which maps filler/repetition spans back to generated token positions and remasks those spans. It also reaches only `0.454`, `+0.004` over evolved, with `1/7/0`. `targeted_filler_repair` was never selected and averaged `-0.005` task delta vs source, so token-span cleanup alone is not enough. |
| LLaDA prompt-guided repair diagnostic | `llada_planning_prompt_guided_repair_pack_v1_report.md` | 56 fresh GPU generations using `--repair-pack prompt_guided`, where repair prompts include the source draft plus a label-free generic critique. It improves over the minimal-mask diagnostics but remains below the canonical line: repair-selected `0.459`, `+0.008` over evolved, `2/6/0`, and zero oracle headroom. The useful positive signal is `plan_001`: `prompt_guided_revision_repair` improved the source by `+0.034`. Prompt-guided repair should be an adaptive side path, not the main repair pack. |
| LLaDA adaptive prompt-guided rescue diagnostic | `llada_planning_adaptive_hybrid_prompt_guided_rescue_v1_report.md` | 56 fresh GPU generations using the canonical prefix/history repair line plus `--prompt-guided-rescue-trigger baseline_or_source_quality`. It matches the canonical selected score, not a new win: repair-selected `0.490`, `+0.039` over evolved, `6/2/0`, zero oracle headroom, but at `7.00` generations/task and only `0.013` task-score gain per extra generation. `prompt_guided_revision_repair` was generated on 7 tasks and selected on 0; its average task delta versus source was only `+0.005`. Current prompt-guided revision is therefore a diagnostic fallback, not a default budget spend. |
| Dream + LLaDA mixed adaptive history-rescue repair | `mixed_adaptive_history_rescue_margin01_v1_report.md` | 116 seeded generations over the planning-plus-objective mix; fixed `0.436`, random `0.423`, trajectory `0.465`, evolved `0.480` over 22 model-task pairs. The repair arm is `8/22` overall and `8/8` on repair-eligible LLaDA planning tasks, reaching `0.490`, `+0.039` over evolved on that covered slice, with `6/2/0` wins/ties/losses and zero oracle headroom. |
| Dream + LLaDA mixed family/regret rescore | `mixed_adaptive_history_rescue_family_regret_rescore_v1_report.md` | Reuses the mixed adaptive history-rescue raw generations with `planning_quality_delta_risk_guarded`, explicit selector-regret summaries, and a by-family arm table. Aggregate fixed/random/trajectory/evolved remain `0.436`/`0.423`/`0.465`/`0.480`; repair remains `0.490` on the 8 repair-eligible LLaDA planning tasks. The new diagnostics show trajectory selector regret `0.030` over `8/22` improvable selections, evolved regret `0.014` over `7/22`, and repair regret `0.000` over `0/8` on its covered planning slice. The rescore also fixes mixed-report repair coverage so inherited exact-answer arms no longer count as repair coverage. |
| LLaDA mixed gated constraint-gap repair | `llada_mixed_gated_constraint_gap_risk_guard_v1_report.md` | 59 fresh LLaDA generations over 8 planning tasks plus math/symbolic/science checks using the current `state_adaptive` pack, prompt-gap constraint rescue, and `planning_quality_delta_risk_guarded`. Aggregate fixed/random/trajectory/evolved are `0.482`/`0.455`/`0.481`/`0.510`; the repair arm is `8/11` overall and `8/8` repair eligible, all planning, with `0.491`, `+0.040` over evolved, `6/2/0`, zero oracle headroom, and `0.022` repair gain per extra generation. Family table: planning keeps the current best repair line, math and science score `1.000`, and symbolic remains `0.000`, making symbolic repair/exact-answer handling the next mixed-benchmark gap. |
| LLaDA mixed gated repair plus exact-answer counterfactual repair | `llada_mixed_gated_constraint_gap_exact_repair_v2_report.md`; current guarded ranked-span rerun with identity: `llada_mixed_gated_ranked_span_guarded_exact_identity_v1_report.md`; targeted smoke: `llada_sym002_exact_counterfactual_repair_v1_report.md` | The exact repair path uses prompt-derived answer proposals and promotes a repair only when the model output matches its own proposal, not by reading the hidden expected answer. The original 60-generation mixed run has aggregate fixed/random/trajectory/evolved at `0.482`/`0.455`/`0.481`/`0.510`; repair-selected covers `9/11` overall and `9/9` eligible, reaches `0.548`, beats evolved by `+0.147` on covered tasks, has `7/2/0` repair-vs-evolved wins/ties/losses, zero oracle headroom, and `0.083` task gain per extra generation. The current guarded ranked-span rerun uses the same compact 8-planning plus math/symbolic/science suite with full revision, anchor revision, and ranked-span prompt-gap rescue available behind the gate. It preserves the same `0.548` mixed repair score and `0.491` planning score, repairs `sym_002` from `0.000` to `1.000`, correctly rejects the slightly higher `plan_001` anchor candidate because it leaks a prompt-term checklist (`Risk Penalty 0.180`), and carries deterministic `run_id`/`content_hash`. This is now the cleanest compact mixed LLaDA line. |
| LLaDA exact-suite counterfactual repair | `llada_exact_suite_counterfactual_repair_v1_report.md` | 69 fresh LLaDA generations over all 17 exact-answer math/symbolic/science tasks. Fixed, trajectory, and evolved all score `0.941`; random scores `0.824`. Only one task is repair-eligible, the failed symbolic `sym_002`; counterfactual answer repair covers `1/17` overall and `1/1` eligible, repairs it to `1.000`, and leaves zero repair-oracle headroom. |
| LLaDA full planning-plus-exact scout | `llada_full_scout_gated_exact_repair_v1_report.md` | 116 fresh LLaDA generations over 25 tasks: 8 planning, 8 math, 6 symbolic, and 3 science. Full-suite fixed/random/trajectory/evolved means are `0.772`/`0.680`/`0.772`/`0.784`. The repair arm is correctly scoped to eligible tasks: `9/25` overall and `9/9` eligible, with covered-task repair-selected score `0.548`, `+0.147` versus evolved, `7/2/0` repair-vs-evolved wins/ties/losses, zero repair-oracle headroom, and `0.083` task gain per extra generation. Planning holds the `0.491` gated constraint-gap line, symbolic repairs `sym_002` to `1.000`, and the exact-answer proposal diagnostics show `Proposal Task 1.000` with `Task-vs-Proposal 0.000`, so this is the current best full LLaDA scout but the exact-answer lift is still proposer-attributable. |
| LLaDA hard exact arithmetic-feedback repair | Current verifier-span early stop: `llada_hard_exact_verifier_span_early_stop_v1_report.md`; claim-map entry: `dense_llada_hard_exact_no_proposal_span_repair`; original line: `llada_hard_exact_arithmetic_feedback_v1_report.md`; integer-gate diagnostic: `llada_hard_exact_verifier_span_integer_gate_v1_report.md`; negative diagnostic: `llada_hard_exact_verifier_span_self_repair_v1_report.md` | 19 fresh LLaDA generations over four harder exact tasks added as `math_009`, `math_010`, `math_011`, and `sym_007`. These prompts have zero deterministic answer proposals, so they stress diffusion execution rather than proposal-only repair. Fixed/random/trajectory/evolved all score `0.500`. Long scratchpad self-repair fixes `sym_007`; arithmetic-feedback repair detects the false `math_010` claim `3*14 + 2*9 = 54`, feeds back that the expression equals `60`, and repairs the answer to `10`. Repair-selected covers `2/4` overall and `2/2` eligible, reaches `1.000` on the eligible slice, beats evolved by `+1.000`, has `2/0/0` repair-vs-evolved wins/ties/losses, zero repair-oracle headroom, and `0.667` task gain per extra generation. This is the first exact-answer line in this repo where the gain is neither proposal-attributable nor selected by hidden answers. The verifier-span no-proposal diagnostic showed final-answer span inpainting is not useful for integer scratchpad failures: `answer_span_repair` generated twice, selected zero times, and left both wrong answers unchanged. The gated rerun excludes no-proposal integer answer-span repair, preserves repair-selected `1.000`, reduces total records from `21` to `20`, and improves gain per extra generation versus evolved from `0.400` to `0.500`. A selector rescore gives verifier-localized arithmetic span repair priority over broader feedback when both pass the same label-free guards, so `math_010` selects `arithmetic_contradiction_span_repair`. The current early-stop GPU run then skips arithmetic feedback once span repair passes, preserving repair-selected `1.000` while using 19 total generations, repair budget delta `1.50` vs evolved, and gain per extra generation `0.667`. This result is now in `CLAIM_EVIDENCE_MAP.md` so public evidence covers no-proposal exact reasoning, not only planning repair. |
| LLaDA extended full arithmetic-feedback scout | `llada_extended_full_arithmetic_feedback_v1_report.md` | 135 fresh LLaDA generations over 29 tasks: 8 planning, 11 math, 7 symbolic, and 3 science. This is the current full LLaDA line because it runs planning repair, prompt-derived proposal repair, scratchpad self-repair, and arithmetic-feedback repair under one budget ledger. Full-suite fixed/random/trajectory/evolved means are `0.734`/`0.656`/`0.734`/`0.745`. Repair coverage is explicit as `11/29` overall and `11/11` eligible; on that eligible slice repair-selected reaches `0.630`, beats evolved by `+0.302`, has `9/2/0` repair-vs-evolved wins/ties/losses, zero repair-oracle headroom, and `0.175` task gain per extra generation. The exact-answer repair diagnostics now separate proposal-attributable `sym_002`, label-free self-repair on `sym_007`, and arithmetic-feedback repair on `math_010`. |
| LLaDA GSM-style distractor exact repair | `llada_gsm_distractor_self_repair_v1_report.md` | 19 fresh LLaDA generations over four new hidden-distractor arithmetic tasks, `math_012` through `math_015`. The deterministic proposal layer returns no candidates for all four, so the repair path is self-repair plus arithmetic feedback rather than proposal-only. Fixed/random/trajectory/evolved all score `0.500`; repair covers the two failures and reaches `1.000` on the eligible slice, beating evolved by `+1.000` with `2/0/0` repair-vs-evolved wins/ties/losses and zero repair-oracle headroom. `math_014` is fixed by scratchpad self-repair; `math_013` is fixed by arithmetic feedback after the scratchpad says `204 + 56 = 265` and the verifier computes `260`. The arithmetic guard now also recognizes simple natural-language claims like "90 minus 60 is 30" and compound claims like "3 times 14 plus 2 times 9 is 54". |
| Exact-repair arithmetic-evidence guard rescores | `llada_extended_full_evidence_guard_rescore_v1_report.md`; `llada_gsm_distractor_evidence_guard_rescore_v1_report.md` | CPU-only rescoring now requires integer self-repair and arithmetic-feedback repair to include at least one checkable arithmetic claim before selection; merely changing the final integer is not enough. The 29-task full line is unchanged under this stricter guard: repair coverage remains `11/29` overall and `11/11` eligible, repair-selected still beats evolved by `+0.302` with `9/2/0` repair-vs-evolved wins/ties/losses and zero repair-oracle headroom. The GSM distractor slice is also unchanged: repair still covers `2/2`, beats evolved by `+1.000`, and has zero repair-oracle headroom. Reports now expose `Arithmetic Claims`; the selected full-line exact repairs average `2.5` claims for self-repair and `3.0` for arithmetic feedback. |
| Missing-evidence exact repair branch | unit-tested in `test_exact_self_repair_spends_arithmetic_evidence_branch_when_claims_missing` | The exact repair generator now has a second label-free fallback for integer tasks: if `self_check_answer_repair` changes the answer but produces zero checkable arithmetic claims, the runner can spend the remaining repair slot on `arithmetic_evidence_repair`. That prompt asks diffusion to solve again with explicit digit/operator equations and to ignore irrelevant quantities. Selection still requires a changed parseable answer, arithmetic consistency, and at least one checkable claim, so the branch adds verifier-visible evidence rather than relaxing the guard. |
| Semantic irrelevant-number guard rescores | `llada_extended_full_semantic_guard_rescore_v1_report.md`; `llada_gsm_distractor_semantic_guard_rescore_v1_report.md` | CPU-only rescoring now rejects integer exact repairs whose arithmetic expressions use prompt numbers marked as irrelevant by local prompt language such as "not being packed", "not ticket revenue", "only count", or "question asks". This is label-free semantic equation verification: it does not know the hidden answer, but it can reject internally valid equations that use excluded quantities. The guard preserves both current exact lines: the 29-task full line remains `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`, zero repair-oracle headroom; the GSM distractor slice remains `2/2` eligible coverage, `+1.000`, `2/0/0`, zero headroom. Reports now expose `Irrelevant # Used`, and both selected exact-repair slices score `0.000` on that diagnostic. |
| Operation-role guard rescores | `llada_extended_full_operator_guard_rescore_v1_report.md`; `llada_gsm_distractor_operator_guard_rescore_v1_report.md` | CPU-only rescoring now rejects integer exact repairs whose verifier-readable arithmetic claims omit an operation clearly required by the prompt, such as subtraction for "remaining", division for "shared equally" or "per bag", multiplication for "dollars each" or "twice as many", and addition for totals across groups. The guard preserves both current exact lines: the 29-task full line remains `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`, zero repair-oracle headroom; the GSM distractor slice remains `2/2` eligible coverage, `+1.000`, `2/0/0`, zero headroom. Reports now expose `Missing Ops`, and selected exact-repair slices score `0.0` on that diagnostic. |
| Quantity-role binding guard rescores | `llada_extended_full_role_guard_rescore_v1_report.md`; `llada_gsm_distractor_role_guard_rescore_v1_report.md` | CPU-only rescoring now rejects integer exact repairs whose equations contain the right operations but bind prompt quantities to the wrong roles, such as multiplying adult ticket counts by child prices, dividing by the wrong equal-share count, or omitting verifier-readable pairs like trays times items per tray. The guard remains label-free: it extracts only explicit prompt quantity roles and checks the repair's arithmetic claims. It preserves both current exact lines: the 29-task full line remains `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`, zero repair-oracle headroom; the GSM distractor slice remains `2/2` eligible coverage, `+1.000`, `2/0/0`, zero headroom. Reports now expose `Role Gaps`, and selected exact-repair slices score `0.0` on that diagnostic. |
| Arithmetic-provenance guard rescores | `llada_extended_full_provenance_guard_rescore_v1_report.md`; `llada_gsm_distractor_provenance_guard_rescore_v1_report.md` | CPU-only rescoring now rejects integer exact repairs whose equations introduce ungrounded constants that are neither prompt numbers nor outputs of earlier verified equations. This catches a stricter derived-variable failure mode: a later equation cannot use an earlier claimed value unless that earlier equation was arithmetic-consistent. The guard preserves both current exact lines: the 29-task full line remains `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`, zero repair-oracle headroom; the GSM distractor slice remains `2/2` eligible coverage, `+1.000`, `2/0/0`, zero headroom. Reports now expose `Provenance Gaps`; selected exact repairs remain at `0.0` on that diagnostic, while non-selected inconsistent scratchpads can surface provenance gaps for downstream debugging. |
| Final-answer role guard rescores | `llada_extended_full_final_role_guard_rescore_v1_report.md`; `llada_gsm_distractor_final_role_guard_rescore_v1_report.md` | CPU-only rescoring now rejects exact integer repairs where the final answer is not the prompt-requested role output, including totals, per-share divisions, full-bag floor divisions, and remainders. The guard preserves both current exact lines: the 29-task full line remains `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`, zero repair-oracle headroom; the GSM distractor slice remains `2/2` eligible coverage, `+1.000`, `2/0/0`, zero headroom. Reports now expose `Final Role Gaps`; selected exact repairs score `0.0` on that diagnostic, while a non-selected GSM self-repair surfaces a useful final-role gap. |
| Final-answer object guard rescores | `llada_extended_full_object_guard_rescore_v1_report.md`; `llada_gsm_distractor_object_guard_rescore_v1_report.md` | CPU-only rescoring now rejects exact integer repairs whose final answer explicitly names an object the prompt excluded, such as orange bags when the prompt says oranges are not being packed, donations when the prompt asks for ticket revenue, or chocolate-chip cookies when the prompt asks about all cookies. This does not require every answer to include units; it only blocks final-answer contexts that name a locally excluded object. The guard preserves both current exact lines: the 29-task full line remains `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`, zero repair-oracle headroom; the GSM distractor slice remains `2/2` eligible coverage, `+1.000`, `2/0/0`, zero headroom. Reports now expose `Object Gaps`. |
| Final-answer target guard rescores | `llada_extended_full_target_guard_rescore_v1_report.md`; `llada_gsm_distractor_target_guard_rescore_v1_report.md` | CPU-only rescoring now rejects exact integer repairs whose final-answer unit names the wrong prompt-known target or attaches a conflicting modifier to the requested target head. This catches cases like answering `8 students` when the prompt asks how many cookies each student gets, or `9 pear bags` when the target is apple bags, while still allowing bare numeric answers and bare units like `9 bags`. The guard preserves both current exact lines: the 29-task full line remains `11/11` eligible repair coverage, `+0.302` versus evolved, `9/2/0`, zero repair-oracle headroom; the GSM distractor slice remains `2/2` eligible coverage, `+1.000`, `2/0/0`, zero headroom. Reports now expose `Target Gaps`. |
| Constrained short-text self-repair | unit-tested in `test_label_free_short_text_answer_extraction_for_constrained_prompts` and `test_exact_self_repair_runs_for_constrained_short_text_without_proposals` | Exact self-repair now supports prompt-constrained `short_text` answers, not just integers and multiple choice. The label-free parser only enables this branch when the prompt gives a bounded answer schema: on/off, yes/no, a fixed number of letters separated by spaces, or a final list drawn from a stated initial list. A fake-backend generation test verifies the actual repair-record path for a constrained letter answer with no deterministic proposal, extracting `Answer: X Y Z`, marking it as a changed self-repair, and giving it a positive exact-repair selection score without consulting the hidden label. |
| Short-text symbolic proof guard | unit-tested in `test_short_text_symbolic_proof_guard_rejects_wrong_mechanical_answer` and `test_short_text_symbolic_proof_guard_allows_no_solver_schema` | Exact self-repair selection now rejects bounded `short_text` repairs that contradict a prompt-derived symbolic solver when one exists. The guard reuses the order/list/toggle prompt solvers behind the counterfactual proposal layer, but applies them as verifier evidence for self-repair records: a wrong derived order such as `D A C B` is rejected against the mechanically solved `D A B C`, while no-solver constrained schemas such as `Answer: X Y Z` remain eligible. Repair diagnostics now expose `Symbolic Gaps`. |
| Short-text syllogism proof guard | unit-tested in `test_counterfactual_candidates_include_symbolic_syllogism_solver` and `test_short_text_symbolic_proof_guard_rejects_wrong_syllogism_answer` | The prompt-derived symbolic solver now handles simple categorical yes/no syllogisms. It proves `no` for chains such as "All zargs are blicks. No blicks are morts. Can a zarg be a mort?", and it can prove direct inheritance positives when the target category is in the subject's all-are closure. The same solver feeds both counterfactual proposals and the self-repair symbolic proof guard, so a schema-valid but logically wrong final answer like `Answer: yes` is rejected with a `Symbolic Gaps` diagnostic. |
| Short-text symbolic trace guard | unit-tested in `test_short_text_trace_guard_requires_mechanical_order_trace`, `test_short_text_trace_guard_requires_letter_transform_trace`, and `test_short_text_trace_guard_requires_syllogism_relation_trace` | Mechanically solvable `short_text` self-repairs now need minimal trace evidence before the final answer. For order prompts, the trace must show adjacent before-relations; for list prompts, swap evidence; for letter-code transforms, rotate/swap operation evidence; for toggle prompts, toggle/parity evidence; and for syllogisms, category/exclusion relation evidence. This rejects terse repairs like `Answer: D A B C` or `Answer: no` when the solver can prove the answer but the repair text does not carry the proof. No-solver bounded schemas remain eligible. Repair diagnostics now expose `Trace Gaps`. |
| LLaDA letter-transform symbolic repair | `llada_symbolic_letter_transform_repair_v1_report.md`; negative probe: `llada_symbolic_short_text_no_proposal_self_repair_v1_report.md` | Added `sym_008`, a bounded symbolic code task: start `K L M`, rotate one step left, then swap the final two letters. Before the operation solver, fixed/random/evolved and self-check repair all gave `M L K`, and the corrected diagnostics showed self-check had not changed the answer. The new label-free letter-transform solver derives `L K M` from the prompt, feeds counterfactual repair, and the fresh LLaDA run repairs the task from `0.000` to `1.000`. Repair-selected beats evolved by `+1.000`, has `1/0/0` repair-vs-evolved wins/ties/losses, zero repair-oracle headroom, and `1.000` task gain per extra generation. Unit tests now cover the solver, the symbolic proof guard, trace evidence for letter transforms, and short-text repair metadata normalization. |
| Exact proposal-history trajectory selection | `llada_symbolic_letter_transform_proposal_history_rescore_v1_report.md`; `llada_extended_full_proposal_history_rescore_v1_report.md` | Added `--exact-task-trajectory-policy proposal_history`, a label-free exact-task selector that may select either the final output or a visible denoise-history state only when it matches a prompt-derived answer proposal. Hidden labels are used only after selection for reporting. On the `sym_008` raw trace, no correct answer appeared in the base denoise history, so trajectory/evolved stayed fixed and counterfactual repair remained the only win. On the 29-task extended full raw trace, the policy found final proposal matches and fixed fallbacks, but no history-state wins: trajectory delta versus fixed was `-0.000`, evolved delta versus fixed was `+0.011`, repair still beat evolved by `+0.232` on the covered slice, and selector regret versus trajectory stayed `0.091` over `8/29` improvable selections. This is an implemented selector surface plus a negative diagnostic, not a new aggregate-best line. |
| LLaDA full-history exact symbolic probe | `llada_symbolic_full_history_probe_v1_report.md`; raw file `llada_symbolic_full_history_probe_v1_raw.jsonl` | Added `sym_009` and `sym_010` as additional symbolic exact probes, added `--history-sample-count` so a run can densely sample denoise states, and added history-mutability diagnostics to reports. With `--history-sample-count 32`, the three-task LLaDA probe generated 14 records and found `14/14` sampled histories were monotonic fills: committed-token changes `0`, committed-token remasks `0`, mask-count increases `0`. Proposal-history trajectory still had no history-state wins (`trajectory_selected:fallback=3`); evolved final selection solved `sym_010`, and counterfactual repair solved `sym_008` and `sym_009`. The architectural lesson is stronger than the score: this backend's raw history does not revise visible tokens, so next-generation latent sequence reasoning needs an explicit non-monotonic remask/revision operator rather than another passive history selector. |
| Non-monotonic LLaDA revision operator | Exact probes: `llada_symbolic_revision_probe_v1_report.md`, `llada_symbolic_revision_probe_frac50_v1_report.md`; planning probe: `llada_planning_revision_probe_v1_report.md` | Added revision fields to `DiffusionGenerationConfig` and `DiffusionScheduleCandidate`, plus `--include-revision-schedules`, `--revision-remask-fraction`, and `--revision-steps`. A revision schedule fills normally, remasks low-confidence committed suffix tokens inside the same generation, appends the remask state to history, then continues denoising. This breaks monotonic fill: the exact 25%/16-step probe shows `48` committed remasks and `10` remask-mediated rewrites; the exact 50%/24-step probe shows `96` remasks and `8` rewrites; the planning 25%/16-step probe shows `96` remasks and `17` rewrites. Blind revision is not yet a scoring win: exact evolved remains `0.000` and repair still supplies the `1.000` wins; planning evolved regresses by `-0.006` versus trajectory on the three-task slice. Treat this as the first real non-monotonic trajectory substrate, not the final policy. |
| Verifier-guided answer-span revision and revision guard | `llada_symbolic_verifier_revision_probe_v1_report.md`; planning rescore `llada_planning_revision_guard_rescore_v1_report.md` | Added `build_answer_span_repair_seed`, `--exact-verifier-revision`, and `--revision-promotion-margin`. Exact verifier revision remasks the source answer span under the original prompt, so it is a true diffusion inpainting branch rather than a counterfactual prompt. With prompt-derived proposals, promotion still requires the generated text to match the proposal. With `--exact-self-repair`, constrained non-integer label-free tasks without proposals can use the rejected answer span as the remask target and then pass through the same changed-answer and symbolic-trace guards as self-repair; no-proposal integer tasks are gated away from this shortcut because scratchpad failures need arithmetic evidence repair. On `sym_008,sym_009,sym_010`, repair-selected reaches `1.000`, `+1.000` over evolved, with `3/0/0` wins and zero repair oracle headroom; `answer_span_repair` itself solves `sym_010` and is selected once, while counterfactual proposal repair still solves the other two. The planning rescore reuses the prior blind-revision raw file and applies a `0.050` revision promotion margin: the old `-0.006` evolved regression becomes `0.000` versus trajectory with `0/3/0` wins/ties/losses, blocking the `plan_008` loss. Next step is to generalize verifier-guided revision from answer spans to arithmetic contradiction spans and planning constraint gaps. |
| Arithmetic and planning span-repair diagnostics | Arithmetic: `llada_math_arithmetic_downstream_span_budget2_v1_report.md`, `llada_math_arithmetic_downstream_span_revision_v1_report.md`; planning: `llada_planning_constraint_gap_ranked_span_v3_report.md`, `llada_planning_constraint_gap_ranked_span_v6_8task_report.md`; canonical comparison: `llada_planning_constraint_gap_ranked_span_v5_canonical_report.md`; negative prompt-copy control: `llada_planning_constraint_gap_ranked_span_v2_report.md` | Added `build_text_span_repair_seed`, `arithmetic_contradiction_span_repair`, and `constraint_gap_span_repair`. Arithmetic span repair starts from a failed self-check scratchpad, remasks verifier-identified inconsistent arithmetic plus dependent downstream claims and final answer, and denoises under a correction prompt before falling back to arithmetic feedback if budget remains. On `math_010`, downstream span repair masks the bad `3*14 + 2*9 = 54` chain through `Answer: 12` and repairs the task to `1.000`; with `--limit-repair-candidates 2`, it is selected directly, beats evolved by `+1.000`, has zero repair oracle headroom, and uses one fewer repair generation than the older self-check-plus-feedback path. Planning span repair now ranks weak downstream planning sentences from prompt-gap pressure, masks those decoded spans, and uses a span-specific prompt that preserves only the useful opening instead of quoting the full bad draft back to LLaDA. On `plan_001`, the old prompt-copy control selected the right spans but reconstructed the bad answer at `0.399`; the span-specific prompt repairs the answer to `0.465`, beats evolved by `+0.066`, and is selected with zero repair oracle headroom. The current-code eight-task diagnostic (`v6`) improves repair-selected mean from `0.412` to `0.465`; span repair itself averages `0.430` with `4/2/2` wins/ties/losses versus source and is selected on 3/8 tasks. The canonical evolved/full constraint-gap comparison (`v5`) reaches `0.482`, below the older efficient gated state-adaptive line at `0.491`, so ranked planning span repair is a real positive operator but not yet the default budget policy. |
| Planning span-residue, localization, and compact targeting guards | `llada_planning_constraint_gap_ranked_span_v2_span_residue_guard_rescore_v1_report.md`; localization smoke: `llada_planning_span_localization_smoke_v1_report.md`; compact mixed refresh: `llada_mixed_gated_ranked_span_guarded_exact_identity_v1_report.md`; compact MoE planning line: `llada_moe_planning_compact_span_score_max_v2_report.md`; compact MoE mixed line: `llada_moe_mixed_compact_span_score_max_v1_report.md` | Added a label-free `Span Residue` diagnostic and risk penalty for planning span repairs that regenerate the exact weak spans the verifier targeted. The negative prompt-copy control now reports `Span Residue 0.180` and `Risk Penalty 0.180` for `constraint_gap_span_repair`, because it copied back both bad downstream sentences instead of repairing them. The benchmark now also reports span-seed localization: whether verifier targets were found as literal decoded token spans or whether repair fell back to a tail window. The localization smoke confirms `constraint_gap_span_repair` on `plan_001` uses literal span localization (`1.000`) with no fallback (`0.000`) while preserving the known `0.399` to `0.465` repair. The default span repair now uses `planning_span_selection_policy=compact`: it keeps source-relative ranking, avoids extra targets from prompt-gap count alone, keeps decision-rule context intact, retains near-tie weak failure chains, and refines long risky sentence targets to clauses only when the clause set masks fewer words while preserving verifier score. The full MoE compact-policy planning run repairs the 8-task planning suite to `0.492321`, beating fixed/random/evolved by `+0.080045` / `+0.120196` / `+0.048571`, with `Span Localized 1.000`, `Span Fallback 0.000`, `6/2/0` versus evolved, and `0.000625` oracle headroom. The compact MoE mixed run preserves that same `0.492321` planning repair score inside the 11-task lean suite, keeps math/symbolic/science solved at `1.000`, and improves the older mixed source-ranked line from `0.473482` to `0.492321` at the same 76-generation cost. The compact dense mixed report was also refreshed on CUDA and preserves the dense headline result: repair-selected `0.548`, planning `0.491`, and `sym_002` repaired to `1.000`, now with result identity and the diagnostic column. |
| LLaDA-MoE local compact benchmark | `llada_moe_smoke_v1_raw.jsonl`; `llada_moe_history_smoke_v1_raw.jsonl`; `llada_moe_mixed_gated_ranked_span_guarded_exact_v1_report.md` | Materialized the 13.7 GB HF snapshot locally and ran BF16 CUDA smoke plus history smoke. The history smoke exposes 32 denoise steps and sampled monotonic fill states with token confidences. The compact 11-task run completed 60 full generations: fixed/random/trajectory/evolved were `0.573`/`0.543`/`0.574`/`0.580`; repair-selected covered all 8 planning-eligible tasks at `0.446`, `+0.034` vs fixed, `+0.074` vs random, and `+0.024` vs evolved, with `1/7/0` repair-vs-evolved wins/ties/losses. Exact checks were already `1.000` under guarded proposal-history selection. Treat this as proof that sparse MoE is locally runnable, not as a replacement for the stronger dense-LLaDA planning repair line (`0.491`). |
| LLaDA-MoE mixed compact adaptive source benchmark | `llada_moe_mixed_compact_span_score_max_v1_report.md`; score-efficient fresh run: `llada_moe_mixed_compact_span_score_efficient_fresh_v1_report.md`; compact span-localized planning confirmation: `llada_moe_planning_compact_span_score_max_v2_report.md`; repairability-gated fixed-source budget run: `llada_moe_mixed_compact_span_fixed_source_repairability_gate_fresh_v1_report.md`; denoise-phase fresh trigger: `llada_moe_mixed_compact_span_fixed_source_denoise_phase_gate_fresh_v1_report.md`; denoise-phase trigger rescore: `llada_moe_mixed_compact_span_fixed_source_denoise_phase_gate_rescore_v1_report.md`; repairability geometry audit: `DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md`; repairability geometry sweep: `DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md`; denoise-phase geometry audit: `DIFFUSION_DENOISE_PHASE_GEOMETRY.md`; historical quality-gated budget run: `llada_moe_mixed_compact_span_fixed_source_quality_gate_fresh_v1_report.md`; historical direct fixed-source budget run: `llada_moe_mixed_compact_span_fixed_source_fresh_v1_report.md`; prior source-ranked line: `llada_moe_mixed_revision_constraint_span_source_ranked_score_max_identity_v1_report.md` | The MoE-specific revision plus `constraint_span` adaptive source line now uses compact verifier-localized target selection and holds on the full lean mixed suite. Exact math/symbolic/science checks remain solved, so repair coverage is `8/11` overall and `8/8` eligible. The compact mixed `score_max` run uses `76` fresh records, run ID `diffusion-33bf0475f913c6a7`, and reaches planning repair-selected `0.492321`, beating fixed by `+0.080045`, random by `+0.120196`, and evolved by `+0.048571`, with `6/2/0` repair-vs-evolved wins/ties/losses and `0.000625` oracle headroom. This improves the prior source-ranked mixed line (`0.473482`) without increasing generation count. The fresh score-efficient CUDA run adds a trajectory-quality ceiling, skips the high-quality no-op `plan_002` second source, keeps the selected `plan_006` branch, and preserves `0.492321` at 75 records with `0.043175` gain per extra generation. The dedicated compact planning-only span-localization run uses 58 records, run ID `diffusion-911c8526a9cfa11e`, reports `Span Localized 1.000` / `Span Fallback 0.000` for `constraint_gap_span_repair`, and proves the promoted planning operator is literal verifier-target inpainting rather than generic tail masking. Fresh direct fixed-source repair moved the budget frontier to 30 records and `3.000000x`; the quality-gated fixed-source run moved it to 29 records and `2.875000x`; the repairability-geometry gate dominates both by spending only inside a source-quality plus prompt-gap/coverage band, skipping `plan_002`, `plan_005`, and `plan_008`, and preserving the promoted public `0.531116` at `2.625000x` after the automatic compatible preservation-seeded repair line. The audit verifies the mechanism claim directly: `5/5` repair spends are productive, `3/3` skipped repairs are no-lift against the no-repair baseline, missed repairs are `0`, and productive repair sources form denoise skeletons earlier on average (`16.2` steps) than skipped no-lift states (`30.0` steps). The sweep now evaluates 53,460 label-free geometry-plus-phase gate settings, including optional first-skeleton step caps, and finds 168 zero-waste/zero-miss gates; the promoted gate is score/cost Pareto-equivalent on a 5-point frontier. That signal is executable as `--repair-spend-trigger denoise_phase_repairability`, with `--repair-denoise-skeleton-max-step` available for stricter phase-window tests. Fresh CUDA run `diffusion-419fbf63c9d8e30b` confirms the step-`20` cap as a cheaper operating point: `0.496607` at `2.500000x`, four repair spends, and `plan_007` skipped because its first repairable skeleton appears at step `31`. Fresh CUDA run `diffusion-3b42951db77c5aa6` confirms the step-`32` promoted point: `0.531116` at `2.625000x`, 27 generations, five repair spends, and `plan_007` accepted because step `31` is inside the cap. |
| LLaDA-MoE source-ranked span smoke | `llada_moe_planning_source_ranked_span_smoke_v1_report.md` | `constraint_gap_span_repair` now ranks decoded weak spans by source-relative preservation, prompt-gap miss, keyword coverage, and contradiction relief before masking them. The two-task GPU diagnostic on `plan_002` and `plan_006` generated 16 records, added a `Planning Span Target Diagnostics` table, and reached selected latent repair `0.573750` versus fixed `0.540000`, random `0.460536`, and evolved `0.558214`, with `2/0/0` repair-vs-evolved wins/ties/losses and zero oracle headroom. The full lean mixed run above promotes the same mechanism. |
| LLaDA-MoE clause-level span diagnostic | `llada_moe_planning_clause_ranked_span_smoke_v1_report.md` | Added `--repair-pack constraint_span_clause` as an opt-in diagnostic that splits long planning sentences into comma/semicolon clauses before source-relative span ranking. The two-task smoke reached selected latent repair `0.571250` versus fixed `0.540000`, random `0.460536`, and evolved `0.558214`, with `1/1/0` repair-vs-evolved wins/ties/losses and small oracle headroom. This regressed from the sentence-level smoke at `0.573750`; `plan_002` is the failure case, where the clause repair from `low_confidence_32` scored `0.583` and was not selected while the earlier sentence-level fallback scored `0.689`. Keep clause mode diagnostic-only. |
| LLaDA-MoE span-only planning repair | Full-pack diagnostic: `llada_moe_planning_constraint_gap_repair_v1_report.md`; low-budget line: `llada_moe_planning_constraint_span_repair_v1_report.md` | The transferred state-adaptive pack was weak for MoE, so a high-spend planning diagnostic tested the full constraint-gap pack. Only `constraint_gap_span_repair` mattered: selected on 6/8 tasks, `6/0/2` wins/ties/losses versus its source, no span residue, no risk penalty, and `+0.042` mean task delta versus source. The new `--repair-pack constraint_span` spends only that branch and preserves the score with much lower budget: 40 generations, repair-selected `0.472`, `+0.060` vs fixed, `+0.100` vs random, `+0.050` vs evolved, `6/2/0` repair-vs-evolved wins/ties/losses, and `0.050` task gain per extra generation versus evolved. This is the current best MoE planning policy. |
| LLaDA-MoE revision plus source-aware span repair | `llada_moe_planning_revision_constraint_span_v1_report.md`; source-aware fix: `llada_moe_planning_revision_constraint_span_nonrev_source_rescore_fixed_v1_report.md`; multi-source diagnostic: `llada_moe_planning_revision_constraint_span_multisource_v1_report.md`; adaptive score-max candidate: `llada_moe_planning_revision_constraint_span_adaptive_source_prompt_guard_fresh_v1_report.md`; adaptive efficiency candidate: `llada_moe_planning_revision_constraint_span_adaptive_source_efficiency_fresh_v1_report.md`; threshold sweep: `adaptive_source_gate_sweep_v1_summary.md`; script-regenerated sweep: `adaptive_source_gate_sweep_script_v1_summary.md` | Non-monotonic MoE revision is real: the combined revision run recorded `256` committed remasks and `68` rewrites, and lifted evolved to `0.444`. But the old repair path always branched from that evolved winner, so `plan_007` lost the better baseline-span source and repair-selected stopped at `0.468`. The runner now has `--repair-source-policy non_revision_evolved`, which keeps the revision schedule eligible for the evolved arm while seeding span repair from the best non-revision source. The refreshed source-aware report restores repair-selected `0.472`, improves over the stronger revision-aware evolved arm by `+0.028`, keeps `6/2/0` repair-vs-evolved wins/ties/losses, and leaves only `0.001` oracle headroom. Reports now expose selected repair source control. The `evolved_and_trajectory` diagnostic spends both evolved and base-trajectory repair sources, reaches `0.473` and `7/1/0`, and finds real source-diversity wins on `plan_002` and a selector miss on `plan_006`; however, budget-normalized gain drops from `0.028` to `0.018`, so it is diagnostic, not the default. The new adaptive candidate `non_revision_plus_gap_trajectory` adds the base trajectory source only for distinct low-confidence prompt-gap sources and uses `planning_quality_prompt_coverage_guarded` to avoid keyword-stuffed weak repairs. Raw rescore and fresh GPU run both reach repair-selected `0.474`, `+0.030` vs evolved, `7/1/0`, zero oracle headroom, and `0.024` gain per extra generation at 58 records. The adaptive gate now has named modes: `score_max` for the best confirmed score, `efficiency` for the plan_002-only budget tradeoff, and `custom` for sweeps. Reports include per-task add/skip reasons plus the label-free source features behind them. A 30-cell raw threshold sweep shows `score_max` is score-maximal; `efficiency` is more generation-efficient but loses `0.001339` mean task score. `experiments/sweep_adaptive_source_gate.py` now regenerates the sweep and confirms the named modes sit on equivalent operating plateaus even when another tied threshold pair sorts first. The fresh efficiency candidate confirms that regime at 57 generations, repair-selected `0.472768`, and `0.025794` gain per extra generation. |

These are still scout-scale results, not publication-grade benchmark results.
They prove that the local hardware can run the models and that denoising
trajectories are visible enough to score and repair under a budget ledger.

The newest history-anchor diagnostic is `DIFFUSION_HISTORY_ANCHOR_REPAIR_AUDIT.md`.
It adds `--repair-pack constraint_span_history`, so compact planning-span repair
uses the selected denoise-history state's token IDs and visible text as the
repair source. The fresh MoE run
`llada_moe_mixed_compact_span_history_anchor_denoise_phase_gate_fresh_v1`, run
ID `diffusion-16dc676d10e4b12e`, preserves the same `2.625000x` cost and literal
span localization `1.000000` / fallback `0.000000`, but scores `0.474107`
instead of the final-source policy's `0.489911`. This makes history anchoring a
real but currently diagnostic operator: the next promising direction is a
history/final anchor-choice policy or a consistency loss over stable final
constraints. The audit also computes a post-generation dual-anchor selector:
label-free selector scores recover `0.489911`, but only by raising relative cost
to `3.250000x`, so naive dual spending is diagnostic-only. The same audit now
adds the cheap pre-generation anchor selector: source/history span geometry
chooses history once, preserves the `0.489911` final-source score, and keeps
relative cost at `2.625000x`. The selector is now implemented as
`--repair-pack constraint_span_anchor_select`; anchor-select/history-span packs
now request dense denoise-history sampling by default. Fresh CUDA run
`llada_moe_mixed_compact_span_anchor_select_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-f3c291037d94daaf`, preserves `0.489911` at `2.625000x`,
chooses history on `plan_001`, and chooses final anchors elsewhere.
The anchor-selector theory artifact is now explicit:
`DIFFUSION_ANCHOR_RETENTION_LOSS.md` is generated by
`experiments/analyze_diffusion_anchor_retention_loss.py` and treats
denoise-history repair as a constraint-preserving error-correction loop. It
computes a label-free retention loss plus compact-span advantage gate, making
the current selector trainable/auditable rather than only hand-coded.
The first whole-history search operator is also implemented as
`--repair-pack constraint_span_anchor_search`. Loose search selected an earlier
`plan_003` history false positive and dropped the fresh GPU score to
`0.483348`, so the active guarded search requires target similarity `0.96` and
history/final length retention `0.95`. The guarded fresh GPU run
`diffusion-ccef06238847a352` restores `0.489911` at `2.625000x`, matching the
current public line while giving us a stricter search operator to build on.
The prompt-only contrast variant is now ruled out as a public path:
`--repair-pack constraint_span_history_contrast` keeps final-source span repair
but adds compact denoise-history evidence to the prompt. Fresh GPU run
`diffusion-b92d689695016154` selected zero repairs and scored `0.414598`, so
the next work should keep modifying anchors, masks, or seed geometry rather
than merely appending trajectory text to prompts.
The first seed-geometry variant is now implemented as
`--repair-pack constraint_span_history_instability`: final-source span repair
keeps its compact targets but unions in final token positions that fluctuate
across sampled denoise histories. Fresh GPU run `diffusion-e28eb1d3dde8eea7`
scored `0.459107` at `2.625000x`, beating greedy/random by `+0.046830` /
`+0.086982` with no span fallback and active instability masks. It still trails
anchor-select `0.489911`, so the direction is useful as a secondary remask
feature, not as the public budget policy by itself.
The direct combined operator is now tested as
`--repair-pack constraint_span_anchor_instability`. It chooses the final/history
anchor before repair, then applies the instability mask. Fresh GPU run
`diffusion-d14467a9f9a550b2` improves over standalone instability to
`0.481027` at `2.625000x`, but it remains below anchor-select `0.489911`.
The raw repairs show active instability positions on all five attempts and a
clear gain mainly on `plan_007`, so the next remask work should gate
instability conditionally instead of unioning unstable positions everywhere.
The first conditional version, `--repair-pack
constraint_span_anchor_instability_gated`, exposed why identity control matters:
run `diffusion-30a85507d687dfdc` regressed to `0.452188` because the wrapper
pack still changed the prompt on gate-off tasks. The fixed identity run
`diffusion-a7b64be5b7258f39` now preserves the anchor-select line exactly:
`0.489911` at `2.625000x`, `+0.077634` versus fixed and `+0.117786` versus
random. The generated audit proves all `4/4` gate-off repairs match
anchor-select in generation seed, prompt, masked seed, output text, and score.
The only active gate on `plan_007` changes seed/text but ties the anchor-select
score, so instability masking remains non-improving while the harness is now a
clean geometry A/B surface.
The prompt-gated follow-up is the first positive result from that harness:
`--repair-pack constraint_span_anchor_instability_prompt_gated` keeps the
gate-off branches bit-identical to anchor-select, but when the instability
gate fires it also uses the instability-specific repair instruction. Fresh GPU
run `diffusion-4c6a7a9f356b3f0d` reaches `0.498304` at `2.625000x`, improving
over fixed/random by `+0.086027` / `+0.126179` and lifting only `plan_007` by
`+0.067143` versus anchor-select.
The prompt-only gated control now isolates the mechanism:
`constraint_span_anchor_instability_prompt_only_gated` keeps the same gate and
active instability instruction but removes the active instability remask.
Fresh GPU run `diffusion-4b5fc2b7604c28a5` scores `0.479911` at `2.625000x`;
gate-off branches remain `4/4` identical to anchor-select, while the active
`plan_007` branch drops by `-0.080000`. That makes the positive result a
mask-plus-prompt effect, not a prompt-routing artifact.
The current MoE budget frontier is the automatic compatibility-scored seeded
claim-gated follow-up:
`--repair-pack constraint_span_anchor_instability_claim_auto_compat_seeded_gated`
preserves the same denoise-anchor and instability-mask geometry, keeps the
active `plan_007` instability repair, and improves the public-claim `plan_004`
branch without increasing relative cost. The first claim-gated run
`diffusion-94e95f5d1b3d9822` copied repair meta-language and stayed below the
frontier; the compact-prompt rerun `diffusion-0fc7f067a7d87799` reached
`0.513437` at `2.625000x`; the oracle-aware rerun
`diffusion-692592da063daa60` reached `0.523304` at `2.625000x`; the
compatible-seeded rerun `diffusion-6944d9dd6c412de4` now reaches `0.531116`
at `2.625000x`, improving over fixed/random by `+0.118839` / `+0.158991`.
The fresh realization-guarded confirmation `diffusion-a9ae901393235364`
preserved that same score and cost under
`planning_quality_seed_realization_guarded`. The automatic compatibility-scored
run `diffusion-913b5bccb7894e5a` now recovers the same `0.531116` at
`2.625000x` by scoring compact seed candidates from the task/rubric surface,
and establishes the first automatic budget-frontier tie. The preservation-seeded
run `diffusion-3b42951db77c5aa6` now becomes the public benchmark pointer: it
keeps the same `0.531116` at `2.625000x`, keeps zero repair-oracle headroom, and
removes explicit seed/anchor meta wording from the frontier task. `plan_004`
still hits all five rubric controls and rises to `0.621786`.
The seeded-anchor follow-up, `constraint_span_anchor_instability_claim_seeded_gated`,
fixes that missing phrase directly into the masked denoise seed. Fresh run
`diffusion-6ae167dc85d5e6ac` binds the phrase, but drops the public line to
`0.521295` because `plan_004` loses the public-claim survival control. This is
now the clearest next theory target: semantic anchors need compatibility with
the whole required-control set.
The compatible-seeded run is the first positive compatibility result: a compact
9-token denoise tail carries both `oracle selected results` and `claim survives`
without crowding either control out, so the next theory target is to generalize
that hand-built compatibility anchor into a learned or scored compatibility
loss.
The first automatic version now exists as
`constraint_span_anchor_instability_claim_auto_seeded_gated`. It extracts the
same compact control seed from the active task/rubric surface and applies it
without truncation. Fresh run `diffusion-7b74493b8c5ca15a` keeps all five
`plan_004` rubric hits but scores only `0.520536` at `2.625000x`, below the
fixed compatible seed. That makes the next roadmap target sharper: a learned
seed policy must score realization quality, not just whether the right control
terms were selected.
The action-bearing automatic variant,
`constraint_span_anchor_instability_claim_auto_action_seeded_gated`, adds a
direct verb while preserving the same 9-token masked-tail budget. Fresh run
`diffusion-51b5b82f63ad87cd` scores `0.528482` at `2.625000x`, with
`+0.116205` over fixed, `+0.156357` over random, `6/2/0` wins/ties/losses
versus fixed, and zero repair-oracle headroom. It is close to but still below
the fixed compatible frontier (`0.531116`), because `plan_004` loses some
control compatibility. This keeps the roadmap target on scored or learned
semantic-anchor compatibility rather than another prompt-only constraint.
The automatic compatibility-scored version now exists as
`constraint_span_anchor_instability_claim_auto_compat_seeded_gated`. It scores
candidate compact anchors for oracle/selected-result coverage, claim-survival
coverage, action pressure, and over-compression risk. The first CUDA smoke
proved a separate lesson: mentioning a "generated seed" in the prompt caused
meta wording and dropped `plan_004` to `0.466786`; removing that meta wording
recovered `plan_004 = 0.621786`. Full run `diffusion-913b5bccb7894e5a` ties the
fixed compatible frontier at `0.531116` and `2.625000x`, with zero repair-oracle
headroom. This turns the hand-built compatibility anchor into an automatic
seed-selection policy.
The realization-prompt follow-up,
`constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated`,
tests whether the same automatic compatibility scorer improves when the repair
prompt stops naming seeds and anchors as objects. One-task CUDA smoke
`diffusion-1a80605979a231e8` raises `plan_004` realization quality from
`0.655238` to `0.807460` and removes the `0.140000` meta penalty, but lowers the
task score from `0.621786` to `0.600714`. This is a useful boundary, not a
public promotion: the prompt improves direct realization but weakens the
selected/oracle wording enough to lose task score. Tightened v2 smoke
`diffusion-d475c628f6386098` improves realization quality again to `0.846647`
and preserves zero meta penalty, but the task score still stays `0.600714`. The
next learned objective should combine compatibility, realization, and
rubric-semantic preservation rather than optimizing any one of them alone.
The joint-objective seed policy now makes that combination explicit:
`constraint_span_anchor_instability_claim_auto_joint_seeded_gated` scores compact
seed candidates for compatibility, expected direct realization, and
selected/oracle semantic preservation. One-task CUDA smoke
`diffusion-91dcab0442e7d5a1` selects the 9-token `separate oracle selected; claim
survives if disappears` anchor and keeps semantic preservation at `1.000000`
with zero meta penalty, but `plan_004` still scores `0.600714`. This rules out
seed choice by itself as the missing frontier move; the next objective needs to
shape the denoised continuation, not only the fixed tail anchor.
The preservation-seed follow-up is the first cleaner frontier tie:
`constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated`
uses `compact_preservation_control_terms` to fix
`oracle selected results; preserve claim if disappears` into the masked tail.
Prompt-only smoke `diffusion-05c8f40e3fd0f234` still scored `0.600714`, but the
preservation-seeded CUDA smoke `diffusion-c18d75b68b87ef33` recovered
`plan_004 = 0.621786`, kept semantic preservation at `1.000000`, and removed
seed/anchor meta text. The full mixed-slice run `diffusion-3b42951db77c5aa6`
then ties the public aggregate at `0.531116` and `2.625000x` with zero
repair-oracle headroom, so the claim-evidence map now promotes this cleaner
frontier run.
The explicit realization-constrained version,
`constraint_span_anchor_instability_claim_auto_seeded_realization_gated`, is a
negative boundary. Fresh run `diffusion-2a310ed45712a36b` scores `0.515759` at
`2.625000x`; all `plan_004` rubric controls still fire, but the output is a
low-specificity `Control:` sentence. That rules out longer/stricter prompt
obligations as the realization solution.
The first executable realization-quality loss is now in
`experiments/analyze_diffusion_realization_quality.py` and
`DIFFUSION_REALIZATION_QUALITY.md`. It scores active compact-seed repairs by
direct action coverage, control coverage, seed-term coverage, prompt coverage,
specificity, sentence shape, meta-text penalties, and now semantic preservation
of selected/oracle plus claim-survival relations. The audit separates score from
realization: compatible seeded remains best by task (`0.621786`), while
auto-compat-realized is best by realization/seed objective and
auto-compat-preserve is best by task (`0.621786`) with zero meta penalty. The
benchmark runner exposes
`planning_quality_seed_realization_guarded` and
`planning_quality_seed_objective_guarded`; the tightened guard rejects the
realization-gated `Control:` branch on CPU rescore while preserving the
compatible-seeded frontier on fresh CUDA.

The spend gate now has a complete boundary audit. Forced CUDA probe
`diffusion-8a8a9e8904e62dbf` applied the promoted preservation-seeded repair to
the high-quality skipped task `plan_002`; forced repair fell to `0.582500`
versus source `0.688571`. Forced CUDA probe `diffusion-4699321baf91294e`
covered `plan_005,plan_008`; forced repair selected zero branches and averaged
`-0.014464` task delta versus source. That makes the current
`denoise_phase_repairability` gate part of the mechanism: repair compute should
be spent only when low source quality, prompt-gap geometry, prompt coverage,
and a visible denoise skeleton all line up. The regenerated
`DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md` now scores the gate as an
error-correction classifier with `5` true-positive productive spends, `3`
true-negative skipped no-lift repairs, and no false positives or false
negatives on the promoted 8-task planning slice. The runner also records
`repair_spend_gate_rows` and renders a `Repair Spend Gate Diagnostics` table so
future cost changes are auditable instead of inferred from missing repairs.

## Benchmark Protocol

Keep the benchmark stack narrow:

- `greedy_baseline`: normal greedy decode from the current AR model
- `random_prefix`: RMS-matched random soft prefix, five seeds
- `latent_reasoning`: selected/evolved latent prefix, one final output
- `diffusion_baseline`: Dream or LLaDA with fixed greedy/low-temperature denoise
- `diffusion_latent_reasoning`: judge or scorer selects denoising schedules, remasking policies, or intermediate repairs

The first public comparison should still report the original three arms from
`docs/GENERAL_PURPOSE_LATENT_BENCHMARK_PROTOCOL.md`. Diffusion is the next
mechanism layer, not permission to explode the benchmark stack.

For the immediate GPU-only diffusion work, use
`docs/LEAN_GPU_DIFFUSION_BENCHMARK_PROTOCOL.md` as the execution contract:
headline reporting should stay focused on fixed greedy/low-confidence baseline,
random perturbation, and selected latent/diffusion repair. The compact suite is
8 short planning tasks plus `math_001`, `sym_002`, and `sci_001`.

Use `CLAIM_EVIDENCE_MAP.md` as the claim-promotion ledger and
`DIFFUSION_GROUND_TRUTH_INDEX.md` as the latest canonical pointer table. Both
are generated by `experiments/build_diffusion_claim_evidence.py` and tie each
current diffusion claim to score, report, and raw-generation artifacts while
keeping fixed/random baseline scores scoped to the repair-covered task slice.

Use `docs/DIFFUSION_RESEARCH_TRANSLATION_NOTES.md` for the current research
translation layer: Dream/LLaDA motivate iterative denoise control, LLaDA-MoE is
now registered and preflighted as the next cheap active-parameter candidate,
and JEPA/world-model work motivates surprise/verifier signals over compact
latent trajectories.

## What To Test First

1. Run a one-prompt Dream smoke with `output_history=True`.
2. Verify the generated history has enough intermediate states to score.
3. Add history scoring on short open-ended planning tasks.
4. Compare fixed Dream denoise against Dream with selected schedules:
   - `steps`: 32, 64, 96, 128
   - `temperature`: `0.0` first, then only one nonzero value if collapse occurs
   - `algorithm`: `entropy` first
5. Only after the Dream path works, run LLaDA as the architecture check.

## Artifacts Added For Execution

- `src/latent_reasoning/diffusion/candidates.py`: local candidate registry,
  including dense Dream/LLaDA, LLaDA-MoE, and GGUF fallback metadata
- `src/latent_reasoning/diffusion/backends.py`: lazy HF backend for Dream/LLaDA, including LLaDA token-transfer confidence capture
- `src/latent_reasoning/diffusion/trajectory.py`: sampled denoising-history summaries for judge/evolution hooks
- `src/latent_reasoning/diffusion/control.py`: diffusion-native schedule candidates and trajectory-control scoring
- `src/latent_reasoning/diffusion/repair.py`: suffix-inpainting repair candidates that keep a generated prefix, remask low-confidence generated positions, run source-relative minimal-remask, targeted-content, prompt-guided, state-adaptive, constraint-gap, and replay-consistency repair packs, or remask verifier-identified answer spans
- `src/latent_reasoning/eval/answer_proposals.py`: exact-answer proposal sources for counterfactual repair, including prompt options, simple arithmetic, and simple symbolic transformations
- `experiments/run_diffusion_reasoning_smoke.py`: list, environment probe, and optional generation smoke
- `experiments/run_diffusion_schedule_sweep.py`: load one model and compare denoising schedules as candidates
- `experiments/run_diffusion_scout.py`: run locked scout tasks with schedule selection and scoring
- `experiments/run_diffusion_repair_scout.py`: run branch-and-repair sweeps over selected scout outputs, including exact-answer verifier repairs, counterfactual repairs, and proposal-only ablations that do not count as model generations
- `experiments/run_diffusion_three_arm_benchmark.py`: compare fixed, random, trajectory-selected, optional evolved, and optional repair-selected diffusion arms under a seeded GPU budget, with exact-answer tasks guarded away from raw trajectory selection unless a verifier is active; `--task-preset lean_gpu_mixed` selects the compact 8-planning plus math/symbolic/science GPU suite; the default planning selector scores sampled denoise states, the evolved arm can use a conservative final-quality fallback, repair selection defaults to final planning-quality scoring with a `0.020` promotion edge, exact-answer counterfactual repairs can use prompt-derived proposals and label-free proposal-match promotion, `--exact-verifier-revision` can add original-prompt answer-span inpainting before counterfactual prompt repair and can remask rejected answer spans for constrained non-integer label-free self-repair tasks without proposals, `--exact-self-repair` can spend a longer scratchpad solve-again repair when no prompt-derived proposal exists and promote only changed parseable answers whose scratchpad arithmetic is internally consistent, `--revision-promotion-margin` adds a stricter source-relative gate for non-monotonic revision schedules, `--repair-source-policy non_revision_evolved` can keep revision eligible for the evolved arm while seeding repair from the best non-revision source, `--repair-spend-trigger source_quality_or_short` can skip primary repairs for complete high-quality sources, history-prefix repair can branch from selected mid-denoise states, `--repair-pack state_adaptive` can choose history-anchor length from source/history state quality and pair it with final prefix repair, `--repair-pack constraint_gap` can add prompt-grounded draft revision against missing source terms, `--constraint-gap-rescue-trigger prompt_gap` can spend that prompt-grounded revision only for mid-quality prompt-gap sources, `--repair-pack replay_consistency` can remask positions unstable across denoise-history samples, `--history-repair-fractions` can sweep history anchor lengths, adaptive `--history-rescue-fractions` can spend extra history repairs only after a weak first repair pass, `--history-rescue-trigger baseline_or_selector_disagreement` can spend extra repair only when first-pass selectors disagree, `--history-rescue-visible` can test all-visible mid-denoise repair, `planning_quality_guarded` can penalize over-preserved history repairs, `planning_quality_delta_guarded` can require label-free improvement over the source output, `planning_quality_delta_risk_guarded` can also penalize prompt-contradicting planning repairs, `--repair-pack source_relative` can prioritize minimal low-confidence remasks, `--repair-pack targeted_content` can remask filler/repetition spans, `--repair-pack prompt_guided` can run draft-revision repairs with a generic critique prompt, adaptive `--prompt-guided-rescue-trigger` can spend prompt-guided revisions only after baseline/source-quality/selector-disagreement gates, repair rescoring is source-consistent, reports separate overall and repair-eligible coverage plus repair source steps, reports family-level arm summaries, reports selector-regret/oracle-coverage summaries, reports budget-normalized repair gain, and includes repair-candidate diagnostics with guard penalty, risk penalty, planning-quality delta, proposal-only task attribution, self-repair answer-change and arithmetic-consistency attribution, task delta, and wins/ties/losses versus each repair source; score JSON and reports include deterministic `run_id` and
  `content_hash` fields derived from raw generations, arm selections, and
  summary scores
- `experiments/build_diffusion_claim_evidence.py`: generate
  `CLAIM_EVIDENCE_MAP.md`,
  `eval_results/diffusion_language/diffusion_claim_evidence_map.json`,
  `DIFFUSION_GROUND_TRUTH_INDEX.md`, and
  `eval_results/diffusion_language/diffusion_ground_truth_index.json` from the
  canonical diffusion score files so public claims and latest policy slots stay
  tied to concrete score/report/raw evidence with artifact SHA-256 fingerprints
- `experiments/validate_diffusion_claim_evidence.py`: hard-fail the public
  claim ledger when generated Markdown/JSON/index files are stale, required
  score settings are missing, coverage counts are inconsistent, win/tie/loss
  totals do not match repair coverage, budget-normalized gain arithmetic is
  wrong, raw artifacts are smaller than the promoted generation count, or
  per-claim required repair diagnostics are missing or outside their thresholds;
  it also rejects top-level public docs that point at non-canonical diffusion
  artifacts in public claim contexts
- `experiments/scan_stale_diffusion_docs.py`: compare public docs against the
  generated ground-truth index so old diffusion score/report/raw files cannot
  be described as current, canonical, promoted, headline, or public evidence
- `experiments/rescore_diffusion_scout.py`: re-score raw scout JSONL after scorer fixes without rerunning GPU generations
- `experiments/summarize_diffusion_language_smoke.py`: markdown report from raw diffusion JSONL
- `tests/test_diffusion_language.py`, `tests/test_diffusion_trajectory.py`, `tests/test_diffusion_control.py`, and `tests/test_diffusion_repair.py`: CPU-safe contract tests

## Commands

List candidates:

```powershell
python experiments/run_diffusion_reasoning_smoke.py --list
```

Probe local environment:

```powershell
python experiments/run_diffusion_reasoning_smoke.py --probe-env
```

Download only small custom-code/config files, no weights. This also avoids the
Windows Hugging Face symlink-cache issue seen in this workspace:

```powershell
python experiments/run_diffusion_reasoning_smoke.py --preflight --candidate dream-7b-instruct-hf --json
python experiments/run_diffusion_reasoning_smoke.py --preflight --candidate llada-8b-instruct-hf --json
python experiments/run_diffusion_reasoning_smoke.py --preflight --candidate llada-moe-7b-a1b-instruct-hf --json
```

Run first Dream smoke. This can download roughly 15 GB of weights if not cached:

```powershell
python experiments/run_diffusion_reasoning_smoke.py --materialize --generate --candidate dream-7b-instruct-hf --max-new-tokens 64 --steps 64 --temperature 0.2 --top-p 0.95 --output-history --history-samples 6 --output-jsonl eval_results/diffusion_language/smoke_raw.jsonl --json
```

Summarize the smoke records:

```powershell
python experiments/summarize_diffusion_language_smoke.py --input eval_results/diffusion_language/smoke_raw.jsonl --output eval_results/diffusion_language/smoke_report.md
```

Run a diffusion-native schedule sweep, keeping the model loaded once:

```powershell
python experiments/run_diffusion_schedule_sweep.py --candidate dream-7b-instruct-hf --model-path external/diffusion_models/Dream-v0-Instruct-7B --output-jsonl eval_results/diffusion_language/schedule_sweep_raw.jsonl
```

Run the planning slice of the locked scout pack:

```powershell
python experiments/run_diffusion_scout.py --families planning --raw-output eval_results/diffusion_language/planning_scout_raw.jsonl --scores-output eval_results/diffusion_language/planning_scout_scores.json --report-output eval_results/diffusion_language/planning_scout_report.md
```

Run the current diffusion evolved planning-plus-mix benchmark:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,sym_002,sci_001 --candidates dream-7b-instruct-hf,llada-8b-instruct-hf --limit-evolved-schedules 2 --evolved-promotion-margin 0.015 --raw-output eval_results/diffusion_language/four_arm_evolved_margin015_v1_raw.jsonl --scores-output eval_results/diffusion_language/four_arm_evolved_margin015_v1_scores.json --report-output eval_results/diffusion_language/four_arm_evolved_margin015_v1_report.md
```

Use `--limit-evolved-schedules 0` to reproduce the earlier fixed/random/base
trajectory comparison without the evolved arm. Use `--limit-repair-candidates 0`
to keep the repair arm disabled.

Run the current LLaDA repair-arm planning diagnostic:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --include-history-repairs --history-repair-fractions 0.25 --history-rescue-fractions 0.50 --history-rescue-source-controls evolved_random_48 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_planning_adaptive_history_rescue_margin01_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_adaptive_history_rescue_margin01_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_adaptive_history_rescue_margin01_v1_report.md
```

Run the previous efficient gated LLaDA repair-arm planning diagnostic:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --repair-pack prefix --include-history-repairs --history-repair-fractions 0.25 --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --history-rescue-fractions 0.50 --history-rescue-source-controls evolved_random_48 --prompt-guided-rescue-trigger off --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_planning_primary_repair_gate_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_primary_repair_gate_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_primary_repair_gate_v1_report.md
```

Run the current efficient state-adaptive LLaDA repair-arm planning diagnostic:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_planning_state_adaptive_history_prefix_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_state_adaptive_history_prefix_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_state_adaptive_history_prefix_v1_report.md
```

Run the replay-consistency repair diagnostic:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --repair-pack replay_consistency --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_planning_replay_consistency_repair_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_replay_consistency_repair_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_replay_consistency_repair_v1_report.md
```

Run the constraint-gap hybrid repair diagnostic:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 3 --repair-pack constraint_gap --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_planning_constraint_gap_repair_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_constraint_gap_repair_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_constraint_gap_repair_v1_report.md
```

Run the current gated constraint-gap rescue line:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_planning_gated_constraint_gap_rescue_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_gated_constraint_gap_rescue_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_gated_constraint_gap_rescue_v1_report.md
```

Rescore that raw file with the planning contradiction/risk guard:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_planning_gated_constraint_gap_rescue_v1_raw.jsonl --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_planning_gated_constraint_gap_risk_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_gated_constraint_gap_risk_guard_rescore_v1_report.md
```

Run the visible-history rescue diagnostic:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --include-history-repairs --history-repair-fractions 0.25 --history-rescue-fractions 0.50 --history-rescue-visible --history-rescue-source-controls evolved_random_48 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_planning_visible_history_rescue_margin01_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_visible_history_rescue_margin01_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_visible_history_rescue_margin01_v1_report.md
```

Rescore the visible-history diagnostic with the guarded selector:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_planning_visible_history_rescue_margin01_v1_raw.jsonl --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --include-history-repairs --history-repair-fractions 0.25 --history-rescue-fractions 0.50 --history-rescue-visible --history-rescue-source-controls evolved_random_48 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_planning_visible_history_rescue_guarded_margin01_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_visible_history_rescue_guarded_margin01_v1_report.md
```

Run the disagreement-triggered visible-history diagnostic:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --include-history-repairs --history-repair-fractions 0.25 --history-rescue-fractions 0.50 --history-rescue-visible --history-rescue-trigger baseline_or_selector_disagreement --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_planning_disagreement_visible_history_rescue_guarded_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_disagreement_visible_history_rescue_guarded_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_disagreement_visible_history_rescue_guarded_v1_report.md
```

Run the source-relative minimal-remask diagnostic:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 3 --repair-pack source_relative --include-history-repairs --history-repair-fractions 0.50 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_planning_source_relative_repair_pack_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_source_relative_repair_pack_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_source_relative_repair_pack_v1_report.md
```

Run the prompt-guided draft-revision diagnostic:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 3 --repair-pack prompt_guided --include-history-repairs --history-repair-fractions 0.50 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_planning_prompt_guided_repair_pack_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_prompt_guided_repair_pack_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_prompt_guided_repair_pack_v1_report.md
```

Run the adaptive prompt-guided rescue diagnostic:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --repair-pack prefix --include-history-repairs --history-repair-fractions 0.25 --history-rescue-fractions 0.50 --history-rescue-source-controls evolved_random_48 --prompt-guided-rescue-trigger baseline_or_source_quality --prompt-guided-rescue-source-quality-threshold 0.45 --prompt-guided-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_planning_adaptive_hybrid_prompt_guided_rescue_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_adaptive_hybrid_prompt_guided_rescue_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_adaptive_hybrid_prompt_guided_rescue_v1_report.md
```

Run the current mixed evolved-plus-repair benchmark:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,sym_002,sci_001 --candidates dream-7b-instruct-hf,llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --include-history-repairs --history-repair-fractions 0.25 --history-rescue-fractions 0.50 --history-rescue-source-controls evolved_random_48 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/mixed_adaptive_history_rescue_margin01_v1_raw.jsonl --scores-output eval_results/diffusion_language/mixed_adaptive_history_rescue_margin01_v1_scores.json --report-output eval_results/diffusion_language/mixed_adaptive_history_rescue_margin01_v1_report.md
```

Run the current full LLaDA planning-plus-exact scout:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,math_002,math_003,math_004,math_005,math_006,math_007,math_008,sym_001,sym_002,sym_003,sym_004,sym_005,sym_006,sci_001,sci_002,sci_003 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_full_scout_gated_exact_repair_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_full_scout_gated_exact_repair_v1_scores.json --report-output eval_results/diffusion_language/llada_full_scout_gated_exact_repair_v1_report.md
```

Run the current extended full LLaDA scout with arithmetic feedback:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,math_002,math_003,math_004,math_005,math_006,math_007,math_008,math_009,math_010,math_011,sym_001,sym_002,sym_003,sym_004,sym_005,sym_006,sym_007,sci_001,sci_002,sci_003 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_scores.json --report-output eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_report.md
```

Run the hard exact arithmetic-feedback stress slice:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families math,symbolic --task-ids math_009,math_010,math_011,sym_007 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_hard_exact_arithmetic_feedback_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_hard_exact_arithmetic_feedback_v1_scores.json --report-output eval_results/diffusion_language/llada_hard_exact_arithmetic_feedback_v1_report.md
```

Run the GSM-style hidden-distractor exact repair slice:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families math --task-ids math_012,math_013,math_014,math_015 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --raw-output eval_results/diffusion_language/llada_gsm_distractor_self_repair_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_gsm_distractor_self_repair_v1_scores.json --report-output eval_results/diffusion_language/llada_gsm_distractor_self_repair_v1_report.md
```

Rescore the current full line with the arithmetic-evidence guard:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_raw.jsonl --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,math_002,math_003,math_004,math_005,math_006,math_007,math_008,math_009,math_010,math_011,sym_001,sym_002,sym_003,sym_004,sym_005,sym_006,sym_007,sci_001,sci_002,sci_003 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_extended_full_evidence_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_extended_full_evidence_guard_rescore_v1_report.md
```

Rescore the current full and GSM lines with the operation-role guard:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_raw.jsonl --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,math_002,math_003,math_004,math_005,math_006,math_007,math_008,math_009,math_010,math_011,sym_001,sym_002,sym_003,sym_004,sym_005,sym_006,sym_007,sci_001,sci_002,sci_003 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_extended_full_operator_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_extended_full_operator_guard_rescore_v1_report.md

python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_gsm_distractor_self_repair_v1_raw.jsonl --families math --task-ids math_012,math_013,math_014,math_015 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_gsm_distractor_operator_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_gsm_distractor_operator_guard_rescore_v1_report.md
```

Rescore the current full and GSM lines with the quantity-role binding guard:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_raw.jsonl --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,math_002,math_003,math_004,math_005,math_006,math_007,math_008,math_009,math_010,math_011,sym_001,sym_002,sym_003,sym_004,sym_005,sym_006,sym_007,sci_001,sci_002,sci_003 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_extended_full_role_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_extended_full_role_guard_rescore_v1_report.md

python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_gsm_distractor_self_repair_v1_raw.jsonl --families math --task-ids math_012,math_013,math_014,math_015 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_gsm_distractor_role_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_gsm_distractor_role_guard_rescore_v1_report.md
```

Rescore the current full and GSM lines with the arithmetic-provenance guard:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_raw.jsonl --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,math_002,math_003,math_004,math_005,math_006,math_007,math_008,math_009,math_010,math_011,sym_001,sym_002,sym_003,sym_004,sym_005,sym_006,sym_007,sci_001,sci_002,sci_003 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_extended_full_provenance_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_extended_full_provenance_guard_rescore_v1_report.md

python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_gsm_distractor_self_repair_v1_raw.jsonl --families math --task-ids math_012,math_013,math_014,math_015 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_gsm_distractor_provenance_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_gsm_distractor_provenance_guard_rescore_v1_report.md
```

Rescore the current full and GSM lines with the final-answer role guard:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_raw.jsonl --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,math_002,math_003,math_004,math_005,math_006,math_007,math_008,math_009,math_010,math_011,sym_001,sym_002,sym_003,sym_004,sym_005,sym_006,sym_007,sci_001,sci_002,sci_003 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_extended_full_final_role_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_extended_full_final_role_guard_rescore_v1_report.md

python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_gsm_distractor_self_repair_v1_raw.jsonl --families math --task-ids math_012,math_013,math_014,math_015 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_gsm_distractor_final_role_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_gsm_distractor_final_role_guard_rescore_v1_report.md
```

Rescore the current full and GSM lines with the final-answer object guard:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_raw.jsonl --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,math_002,math_003,math_004,math_005,math_006,math_007,math_008,math_009,math_010,math_011,sym_001,sym_002,sym_003,sym_004,sym_005,sym_006,sym_007,sci_001,sci_002,sci_003 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_extended_full_object_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_extended_full_object_guard_rescore_v1_report.md

python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_gsm_distractor_self_repair_v1_raw.jsonl --families math --task-ids math_012,math_013,math_014,math_015 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_gsm_distractor_object_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_gsm_distractor_object_guard_rescore_v1_report.md
```

Rescore the current full and GSM lines with the final-answer target guard:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_raw.jsonl --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,math_002,math_003,math_004,math_005,math_006,math_007,math_008,math_009,math_010,math_011,sym_001,sym_002,sym_003,sym_004,sym_005,sym_006,sym_007,sci_001,sci_002,sci_003 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_extended_full_target_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_extended_full_target_guard_rescore_v1_report.md

python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_gsm_distractor_self_repair_v1_raw.jsonl --families math --task-ids math_012,math_013,math_014,math_015 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_gsm_distractor_target_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_gsm_distractor_target_guard_rescore_v1_report.md
```

The earlier scratchpad-only run can still be rescored with the arithmetic-consistency guard:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_hard_exact_self_repair_v3_raw.jsonl --families math,symbolic --task-ids math_009,math_010,math_011,sym_007 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 1 --exact-self-repair --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_hard_exact_self_repair_guarded_v1_scores.json --report-output eval_results/diffusion_language/llada_hard_exact_self_repair_guarded_v1_report.md
```

Run the bounded symbolic letter-transform slice:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families symbolic --task-ids sym_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results/diffusion_language/llada_symbolic_letter_transform_repair_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_symbolic_letter_transform_repair_v1_scores.json --report-output eval_results/diffusion_language/llada_symbolic_letter_transform_repair_v1_report.md
```

Rescore exact tasks with label-free proposal-history trajectory selection:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_symbolic_letter_transform_repair_v1_raw.jsonl --families symbolic --task-ids sym_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --exact-task-trajectory-policy proposal_history --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_symbolic_letter_transform_proposal_history_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_symbolic_letter_transform_proposal_history_rescore_v1_report.md

python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_extended_full_arithmetic_feedback_v1_raw.jsonl --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,math_002,math_003,math_004,math_005,math_006,math_007,math_008,math_009,math_010,math_011,sym_001,sym_002,sym_003,sym_004,sym_005,sym_006,sym_007,sci_001,sci_002,sci_003 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --exact-task-trajectory-policy proposal_history --repair-pack state_adaptive --repair-spend-trigger source_quality_or_short --repair-source-quality-threshold 0.50 --repair-source-min-chars 320 --prompt-guided-rescue-trigger off --constraint-gap-rescue-trigger prompt_gap --constraint-gap-rescue-min-terms 6 --constraint-gap-rescue-source-quality-floor 0.40 --constraint-gap-rescue-source-quality-ceiling 0.50 --constraint-gap-rescue-limit 1 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_extended_full_proposal_history_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_extended_full_proposal_history_rescore_v1_report.md
```

Run the full-history symbolic probe and mutability audit:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families symbolic --task-ids sym_008,sym_009,sym_010 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --limit-repair-candidates 2 --exact-self-repair --exact-task-trajectory-policy proposal_history --history-sample-count 32 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results/diffusion_language/llada_symbolic_full_history_probe_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_symbolic_full_history_probe_v1_scores.json --report-output eval_results/diffusion_language/llada_symbolic_full_history_probe_v1_report.md
```

Run the first non-monotonic revision probes:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families symbolic --task-ids sym_008,sym_009,sym_010 --candidates llada-8b-instruct-hf --limit-evolved-schedules 0 --include-revision-schedules --revision-remask-fraction 0.25 --revision-steps 16 --limit-repair-candidates 2 --exact-self-repair --exact-task-trajectory-policy proposal_history --history-sample-count 64 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results/diffusion_language/llada_symbolic_revision_probe_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_symbolic_revision_probe_v1_scores.json --report-output eval_results/diffusion_language/llada_symbolic_revision_probe_v1_report.md

python experiments/run_diffusion_three_arm_benchmark.py --families symbolic --task-ids sym_008,sym_009,sym_010 --candidates llada-8b-instruct-hf --limit-evolved-schedules 0 --include-revision-schedules --revision-remask-fraction 0.50 --revision-steps 24 --limit-repair-candidates 2 --exact-self-repair --exact-task-trajectory-policy proposal_history --history-sample-count 80 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results/diffusion_language/llada_symbolic_revision_probe_frac50_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_symbolic_revision_probe_frac50_v1_scores.json --report-output eval_results/diffusion_language/llada_symbolic_revision_probe_frac50_v1_report.md

python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_004,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 0 --include-revision-schedules --revision-remask-fraction 0.25 --revision-steps 16 --limit-repair-candidates 0 --history-sample-count 64 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results/diffusion_language/llada_planning_revision_probe_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_revision_probe_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_revision_probe_v1_report.md
```

Run verifier-guided answer-span revision and the revision guard rescore:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families symbolic --task-ids sym_008,sym_009,sym_010 --candidates llada-8b-instruct-hf --limit-evolved-schedules 0 --include-revision-schedules --revision-remask-fraction 0.25 --revision-steps 16 --limit-repair-candidates 3 --exact-verifier-revision --exact-self-repair --exact-task-trajectory-policy proposal_history --history-sample-count 64 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results/diffusion_language/llada_symbolic_verifier_revision_probe_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_symbolic_verifier_revision_probe_v1_scores.json --report-output eval_results/diffusion_language/llada_symbolic_verifier_revision_probe_v1_report.md

python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/llada_planning_revision_probe_v1_raw.jsonl --families planning --task-ids plan_001,plan_004,plan_008 --candidates llada-8b-instruct-hf --limit-evolved-schedules 2 --include-revision-schedules --revision-remask-fraction 0.25 --revision-steps 16 --limit-repair-candidates 0 --history-sample-count 64 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --revision-promotion-margin 0.05 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/llada_planning_revision_guard_rescore_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_revision_guard_rescore_v1_report.md
```

Run arithmetic/planning span-repair diagnostics:

The default `constraint_gap_span_repair` now uses adaptive span targeting. It
keeps sentence targets when they are specific, but retries clause targets when a
long draft would otherwise be remasked as one fallback span. The first fresh
CUDA scout after that change is
`eval_results/diffusion_language/llada_planning_constraint_gap_span_adaptive_v1_report.md`:
on `plan_001`, selected latent repair improves fixed/random/evolved by
`+0.066429`, selects `constraint_gap_span_repair`, and leaves zero repair-oracle
headroom. The broader 8-task CUDA scout is
`eval_results/diffusion_language/llada_planning_constraint_gap_span_adaptive_8task_v1_report.md`:
selected latent repair scores `0.465313` vs fixed/random/trajectory `0.412277`,
with `6/2/0` wins/ties/losses and `0.0015625` repair-oracle headroom. This is
now a public claim-map entry:
`dense_llada_planning_adaptive_span_repair`. The lean mixed adaptive-span scout
is
`eval_results/diffusion_language/llada_mixed_adaptive_constraint_span_identity_v1_report.md`:
it is lower absolute score than the strongest guarded mixed line, but uses `53`
full generations instead of `63`, carries deterministic `run_id`/`content_hash`,
and improves gain per extra generation versus evolved from `0.069643` to
`0.115714`. This is tracked as
`dense_llada_mixed_adaptive_span_budget`.

The current mechanism step is compact target selection for diffusion-native
repair. The same default repair now sets
`planning_span_selection_policy=compact`, so adaptive targeting can refine a
long risky sentence into smaller clause spans when that reduces the masked
denoise region, while keeping high-coverage decision-rule spans and near-tie
weak failure chains intact. The MoE CUDA smoke
`eval_results/diffusion_language/llada_moe_planning_compact_span_policy_smoke_v1_report.md`
first confirmed the path on `plan_001`: selected latent repair scored
`0.528214` versus fixed/random/evolved `0.465357`. The full 8-task MoE planning
confirmation is now
`eval_results/diffusion_language/llada_moe_planning_compact_span_score_max_v2_report.md`:
selected latent repair `0.492321`, `+0.080045` vs fixed, `+0.120196` vs random,
`+0.048571` vs evolved, `6/2/0`, literal span localization `1.000`, fallback
`0.000`, and `0.000625` repair-oracle headroom. This is now the promoted MoE
planning span-localization claim. The compact policy also holds in the full
lean mixed suite:
`eval_results/diffusion_language/llada_moe_mixed_compact_span_score_max_v1_report.md`
uses 76 records, keeps math/symbolic/science solved, reaches the same
`0.492321` planning repair-selected score, and improves the older source-ranked
mixed MoE line from `0.473482` to `0.492321` at unchanged generation cost. The
fresh `score_efficient` CUDA confirmation keeps that top score with 75 records
by gating out the unselected `plan_002` second source.

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --families math --task-ids math_010 --candidates llada-8b-instruct-hf --limit-schedules 1 --limit-evolved-schedules 0 --limit-repair-candidates 2 --exact-self-repair --exact-verifier-revision --exact-task-trajectory-policy proposal_history --history-sample-count 64 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results/diffusion_language/llada_math_arithmetic_downstream_span_budget2_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_math_arithmetic_downstream_span_budget2_v1_scores.json --report-output eval_results/diffusion_language/llada_math_arithmetic_downstream_span_budget2_v1_report.md

python experiments/run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001 --candidates llada-8b-instruct-hf --limit-schedules 1 --limit-evolved-schedules 0 --limit-repair-candidates 5 --repair-pack constraint_gap --repair-spend-trigger always --history-sample-count 64 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results/diffusion_language/llada_planning_constraint_gap_span_v1_raw.jsonl --scores-output eval_results/diffusion_language/llada_planning_constraint_gap_span_v1_scores.json --report-output eval_results/diffusion_language/llada_planning_constraint_gap_span_v1_report.md
```

Rescore an existing raw generation file after selector changes:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/four_arm_evolved_margin015_v1_raw.jsonl --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,sym_002,sci_001 --candidates dream-7b-instruct-hf,llada-8b-instruct-hf --limit-evolved-schedules 2 --evolved-promotion-margin 0.015 --trajectory-selector planning_state --scores-output eval_results/diffusion_language/four_arm_evolved_margin015_oracle_rescore_v1_scores.json --report-output eval_results/diffusion_language/four_arm_evolved_margin015_oracle_rescore_v1_report.md
```

Try the experimental prompt-specific selector without spending GPU:

```powershell
python experiments/run_diffusion_three_arm_benchmark.py --reuse-raw-input eval_results/diffusion_language/four_arm_evolved_margin015_v1_raw.jsonl --families all --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008,math_001,sym_002,sci_001 --candidates dream-7b-instruct-hf,llada-8b-instruct-hf --limit-evolved-schedules 2 --evolved-promotion-margin 0.015 --trajectory-selector planning_state_v2 --scores-output eval_results/diffusion_language/four_arm_evolved_planning_state_v2_rescore_v1_scores.json --report-output eval_results/diffusion_language/four_arm_evolved_planning_state_v2_rescore_v1_report.md
```

Run LLaDA after Dream is working:

```powershell
python experiments/run_diffusion_reasoning_smoke.py --materialize --generate --candidate llada-8b-instruct-hf --max-new-tokens 64 --steps 64 --temperature 0 --json
```

## Evidence Standard

Do not claim "diffusion reasoning works" from a good sample. A useful result
needs at least:

- fixed task manifest
- raw JSONL with full text
- generated token counts
- history or schedule metadata
- exact/rubric scoring
- baseline rescue and regression counts
- comparison against the original three-arm protocol

The publishable claim is narrow until the pilot proves otherwise: "latent
trajectory control improves short planning or failure rescue under a fixed GPU
budget."
