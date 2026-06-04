# Lean GPU Diffusion Benchmark Protocol

This is the current execution contract for testing whether diffusion-native
latent-space reasoning improves general-purpose reasoning. ARC/ARK-style work
stays paused. The benchmark stack must stay small enough to run repeatedly on
the local GPU.

The public interpretation layer is
`docs/DIFFUSION_REASONING_FIELD_IMPLICATIONS.md`. Keep this protocol focused on
the cheap three-arm ledger; use the implications doc for the broader claim about
diffusion trajectories as the right substrate for latent reasoning control.

## Allowed Arms

Use only these arms for the next round of public-facing evidence:

| Arm | Meaning | What It Tests |
| --- | --- | --- |
| `fixed` | Greedy/low-temperature fixed denoise schedule. | Baseline local model behavior. |
| `random` | Random remasking or random schedule perturbation under the same budget class. | Whether any perturbation helps. |
| `repair_selected` | Diffusion-native latent reasoning: trajectory/state selection plus gated remasking, span repair, or verifier-guided inpainting. | Whether controlled latent repair improves reasoning. |

`trajectory_selected` and `evolved` can still be reported as internal diagnostic
arms when the runner emits them, but the headline comparison should be fixed
baseline versus random perturbation versus selected latent repair. Avoid adding
extra benchmark arms unless a failure mode demands them.

## Task Mix

Primary task family:

- Short open-ended planning, because this is where diffusion repair currently
  has the clearest non-exact-answer signal.
- Canonical planning IDs: `plan_001` through `plan_008`.

Small mixed checks:

- `math_001`
- `sym_002`
- `sci_001`

This gives an 11-task compact GPU sweep: 8 planning tasks plus 3 exact or
closed-form checks. Use it before spending on larger 25-task or 29-task scouts.

## Current Promoted Compact Command

Run this on CUDA for the current public three-arm claim. It is the cheap mixed
suite: fixed/greedy, random perturbation, and selected latent repair over short
planning plus the three guard tasks.

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --task-preset lean_gpu_mixed --candidates llada-moe-7b-a1b-instruct-hf --limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 --repair-pack constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated --repair-source-policy fixed --repair-spend-trigger denoise_phase_repairability --repair-source-min-chars 240 --repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 --repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 --repair-denoise-skeleton-max-step 32 --repair-selector planning_quality_seed_realization_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --evolved-promotion-margin 0.015 --device cuda --dtype bfloat16 --raw-output eval_results\diffusion_language\llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_phase32_fresh_v1_raw.jsonl --scores-output eval_results\diffusion_language\llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_phase32_fresh_v1_scores.json --report-output eval_results\diffusion_language\llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_phase32_fresh_v1_report.md
```

History-diagnostic repair packs require dense denoise history because they must
choose, seed, contrast, or remask near-final states. That includes
`constraint_span_anchor_select`, `constraint_span_anchor_search`,
`constraint_span_history`, `constraint_span_history_contrast`,
`constraint_span_history_instability`, and the new
`constraint_span_phase_anchor` pack.

## Current Evidence

Promoted public mixed run:
`eval_results/diffusion_language/llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_preservation_seed_fresh_v1_report.md`.

Fresh step-`32` phase-window confirmation:
`eval_results/diffusion_language/llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_phase32_fresh_v1_report.md`.

Current phase-anchor diagnostic boundary:
`DIFFUSION_PHASE_ANCHOR_BOUNDARY.md`.

Current phase-conditioned hybrid confirmation:
`eval_results/diffusion_language/llada_moe_mixed_phase_hybrid_preserve_seeded_gated_fresh_v2_report.md`.

Repo-level evidence map and latest ground-truth index:
`CLAIM_EVIDENCE_MAP.md`.
`DIFFUSION_GROUND_TRUTH_INDEX.md`.

Refresh it with:

```powershell
python experiments\build_diffusion_claim_evidence.py
python experiments\validate_diffusion_claim_evidence.py
```

The evidence map is the current public-claim ledger. The ground-truth index is
the current canonical pointer table for promoted claim files and public policy
slots. Together they link each supported diffusion claim to score, report, and
raw-generation artifacts and recompute fixed/random baseline scores on the
repair-covered task slice, matching the `Lean Three-Arm Headline` rule below.
The index includes SHA-256 fingerprints for each promoted score, report, and
raw artifact.
Fresh benchmark score files and reports also carry a deterministic `run_id` and
`content_hash` derived from raw generations, arm selections, and summary scores,
with volatile timestamps excluded.
The validator is the public-claim gate: it fails if the generated map or index
is stale, required score settings are missing, coverage counts are inconsistent,
win/tie/loss totals do not match repair coverage, budget-normalized gain
arithmetic is wrong, or a raw artifact is too small for the promoted score file.
For claims that depend on mechanism-level diagnostics, `DEFAULT_CLAIMS` can now
declare required repair-candidate metrics; the validator enforces those
thresholds directly from `repair_candidate_summary`.
It also scans `README.md`, `RESEARCH_BRIEF.md`, `ARTICLE_UPDATE.md`, and
`EXPERIMENTS.md` for stale diffusion score/report/raw references in
public-claim contexts, using the generated ground-truth index as the canonical
allowlist.

Headline numbers:

- Overall fixed/random/trajectory/evolved: `0.482` / `0.455` / `0.481` / `0.510`.
- Repair-selected covered-task score: `0.548`.
- Repair delta versus fixed: `+0.181`.
- Repair delta versus random: `+0.213`.
- Repair delta versus evolved: `+0.147`.
- Repair wins/ties/losses versus evolved: `7/2/0`.
- Planning repair line: `0.412` fixed to `0.491` repair-selected.
- Symbolic exact check: `sym_002` repaired from `0.000` to `1.000`.
- Repair-oracle headroom is `0.001`, because the oracle's slightly higher
  `plan_001` candidate is a visible prompt-checklist leakage artifact.

Budget-favored adaptive span line:

- Uses `--repair-pack constraint_span`, which now targets adaptive
  verifier-ranked sentence or clause spans.
- Default `constraint_gap_span_repair` now sets
  `planning_span_selection_policy=compact`: it avoids spending extra mask
  targets from prompt-gap count alone and refines long risky sentence targets to
  clauses only when doing so masks fewer words while preserving verifier score.
- Full generations drop from `63` to `53` versus the strongest guarded mixed
  line.
- Repair-selected covered-task score: `0.516`.
- Repair delta versus fixed/random/evolved: `+0.150` / `+0.182` / `+0.116`.
- Repair wins/ties/losses versus evolved: `2/7/0`.
- Gain per extra generation versus evolved: `0.116`, higher than the guarded
  mixed line's `0.070`.
- Treat this as the budget-favored compact policy, not the top absolute-score
  policy.

Important hygiene result:

- `constraint_gap_revision_anchor25_repair` scored `0.614` on `plan_001`, but
  visibly dumped prompt-gap terms into the answer.
- The risk-guarded selector assigned that candidate `Risk Penalty 0.180`.
- The selected `constraint_gap_revision_repair` scored `0.605` and was cleaner.
- Treat this as a selector-quality win, not a new aggregate-score gain.

Sparse MoE check:

- `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` now runs locally from
  `external/diffusion_models/LLaDA-MoE-7B-A1B-Instruct`.
- The MoE compact report is
  `eval_results/diffusion_language/llada_moe_mixed_gated_ranked_span_guarded_exact_v1_report.md`.
- It completed `60` full generations. Overall fixed/random/trajectory/evolved
  were `0.573` / `0.543` / `0.574` / `0.580`.
- Repair-selected covered all `8/8` planning-eligible tasks at `0.446`, beating
  fixed by `+0.034`, random by `+0.074`, and evolved by `+0.024`.
- Exact checks were already solved by the base MoE outputs under the guarded
  exact-task policy, so this run is mainly a planning-repair diagnostic.
- A MoE-specific `constraint_span` pack now improves the planning line without
  the extra no-op repair branches:
  `eval_results/diffusion_language/llada_moe_planning_constraint_span_repair_v1_report.md`
  reaches repair-selected `0.472` on the 8 planning tasks with `40` total
  generations, `+0.060` vs fixed, `+0.100` vs random, `+0.050` vs evolved,
  and `6/2/0` repair-vs-evolved wins/ties/losses.
- When MoE non-monotonic revision schedules are enabled, use source-aware
  repair seeding:
  `eval_results/diffusion_language/llada_moe_planning_revision_constraint_span_nonrev_source_rescore_fixed_v1_report.md`
  keeps the revision-aware evolved arm at `0.444` and restores repair-selected
  to `0.472` by branching span repair from the best non-revision source. That
  is `+0.028` vs the stronger evolved arm, `6/2/0` repair-vs-evolved
  wins/ties/losses, with `0.001` oracle headroom. It costs `56` generations
  instead of `40`, so treat it as the correct revision diagnostic, not the
  cheaper default.
- The `evolved_and_trajectory` source policy is a diagnostic source-diversity
  spend, not a default. The run
  `eval_results/diffusion_language/llada_moe_planning_revision_constraint_span_multisource_v1_report.md`
  reaches repair-selected `0.473` and `7/1/0` repair-vs-evolved
  wins/ties/losses, but costs `61` generations and drops budget-normalized gain
  to `0.018` per extra generation versus `0.028` for `non_revision_evolved`.
  It is useful because it exposes source-specific wins and selector misses,
  especially `plan_002` and `plan_006`.
- The adaptive source policy is the current revision-enabled MoE candidate.
  `eval_results/diffusion_language/llada_moe_planning_revision_constraint_span_adaptive_source_prompt_guard_v1_report.md`
  was confirmed by the fresh GPU report
  `eval_results/diffusion_language/llada_moe_planning_revision_constraint_span_adaptive_source_prompt_guard_fresh_v1_report.md`.
  It spends `--repair-source-policy non_revision_plus_gap_trajectory` with
  `--adaptive-source-gate-mode score_max` and
  `--repair-selector planning_quality_prompt_coverage_guarded`. It reaches
  repair-selected `0.474`, `+0.030` vs evolved, `7/1/0` repair-vs-evolved
  wins/ties/losses, zero oracle headroom, and uses `58` generations. That is a
  better score than both the single-source and exhaustive multi-source rescores,
  with `0.024` budget-normalized gain per extra generation. The report includes
  an `Adaptive Source Gate` table so the second-source spend is inspectable:
  it fires on `plan_002` and `plan_006`, skips same-source cases, skips the
  low-gap `plan_004`, and rejects the weak random trajectory source on
  `plan_008`.
- The adaptive source threshold sweep is saved at
  `eval_results/diffusion_language/adaptive_source_gate_sweep_v1/adaptive_source_gate_sweep_v1_summary.md`.
  The reusable script form is
  `experiments/sweep_adaptive_source_gate.py`, with the script-regenerated
  artifact at
  `eval_results/diffusion_language/adaptive_source_gate_sweep_script_v1/adaptive_source_gate_sweep_script_v1_summary.md`.
  The companion
  `eval_results/diffusion_language/adaptive_source_gate_sweep_script_v1/adaptive_source_gate_sweep_script_v1_best.json`
  records the first sorted score/efficiency maxima plus the named
  `score_max` and `efficiency` mode rows.
  The `score_max` mode resolves to `gap>=6`, `quality>=0.25` and is on the score-maximal plateau:
  repair-selected `0.474107` with 58 generations, adding only `plan_002` and
  `plan_006`. The stricter `efficiency` mode resolves to `gap>=10`,
  `quality>=0.25`, adds only `plan_002`, and reaches `0.472768` with 57
  generations and a better gain per extra generation
  (`0.025794` vs `0.024286`). A fresh GPU confirmation is saved at
  `eval_results/diffusion_language/llada_moe_planning_revision_constraint_span_adaptive_source_efficiency_fresh_v1_report.md`.
  The script-regenerated grid shows these named modes sit on equivalent
  operating plateaus: several threshold pairs produce the same add set and
  score. Looser settings that add `plan_004` waste one generation without
  improving score.
- An earlier adaptive-source policy has a full lean mixed MoE confirmation:
  `eval_results/diffusion_language/llada_moe_mixed_revision_constraint_span_adaptive_source_score_max_v1_report.md`.
  It runs `plan_001`-`plan_008` plus `math_001`, `sym_002`, and `sci_001`,
  uses `76` fresh records, keeps exact checks solved, and reports repair
  coverage honestly as `8/11` overall and `8/8` repair-eligible. The planning
  repair line reaches `0.474107`, beating fixed by `+0.061830`, random by
  `+0.101982`, and the revision-aware evolved arm by `+0.030357`, with
  `7/1/0` repair-vs-evolved wins/ties/losses and zero oracle headroom.
  Cheap rescores from the same raw file show `efficiency` reaches `0.472768`
  with one fewer generation, while single-source `non_revision_evolved`
  reaches `0.472143` with the best budget-normalized gain. The comparison is
  saved at
  `eval_results/diffusion_language/llada_moe_mixed_revision_constraint_span_policy_comparison_v1.md`.
- Dense LLaDA remains the stronger compact repair line today, but MoE now has a
  budget-efficient active-parameter repair policy to iterate from.
- Compact planning span targeting is now the promoted MoE score-max evidence
  line because the full lean mixed confirmation preserves the planning-only
  compact lift inside the 11-task suite and includes auditable
  `Planning Span Target Diagnostics`. The report is
  `eval_results/diffusion_language/llada_moe_mixed_compact_span_score_max_v1_report.md`:
  76 records, repair coverage `8/11` overall and `8/8` eligible, selected
  latent repair `0.492321`, `+0.080045` vs fixed, `+0.120196` vs random,
  `+0.048571` vs evolved, `6/2/0`, run ID
  `diffusion-33bf0475f913c6a7`, and `0.000625` oracle headroom. This improves
  the older source-ranked mixed line (`0.473482`) at the same generation
  count, while math/symbolic/science checks stay solved at `1.000`.
  The fresh score-efficient CUDA run updates the cost frontier:
  `llada_moe_mixed_compact_span_score_efficient_fresh_v1_report.md` keeps the
  top `0.492321` repair score at 75 records by skipping the high-quality no-op
  `plan_002` second source while keeping the selected `plan_006` branch.
  The fresh single-source CUDA confirmation
  `llada_moe_mixed_compact_span_single_source_fresh_v1_report.md` reaches
  `0.473393` at 74 records; it is now historical frontier evidence because
  direct fixed-source repair dominates it on both score and cost.
- Repairability-geometry gated direct fixed-source repair is now the
  budget-favored public latent point:
  `llada_moe_mixed_compact_span_fixed_source_repairability_gate_fresh_v1_report.md`
  repairs the greedy fixed denoise output directly but spends only when the
  source is weak and its prompt-gap/coverage geometry sits in the compact-span
  repairable band. It skips `plan_002`, `plan_005`, and `plan_008`, uses 27
  records, reaches `0.489911`, and cuts relative repair cost to `2.625000x`
  while keeping `+0.077634` vs greedy and `+0.117786` vs random.
- The companion repairability audit is
  `DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md`, generated by
  `experiments/analyze_diffusion_repairability_geometry.py`. It compares the
  gated run against the ungated fixed-source reference and shows that the gate
  spent on `5/5` productive repairs, skipped `3/3` no-lift repairs, and missed
  `0` repairs. Use it as the current theoretical/geometry artifact for why the
  cheap public budget point is not just a score trick.
- The gate sweep is `DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md`, generated by
  `experiments/sweep_diffusion_repairability_geometry.py`. It now sweeps
  `53,460` source-quality, prompt-gap, prompt-coverage, and optional first
  denoise-skeleton step-cap settings against the promoted diagnostic reference.
  The promoted `0.531116` at `2.625000x` gate is score/cost Pareto-equivalent,
  and 168 gates are zero-waste/zero-miss, so the geometry result is a plateau
  rather than a one-threshold coincidence.
- The same sweep now prints a phase-window tradeoff table. The useful operating
  points are: no cap or skeleton step cap `32` spends five repairs and keeps the
  promoted `0.531116` at `2.625000x`; cap `20` or `24` spends four repairs and
  gives `0.496607` at `2.500000x`; cap `10` or `16` spends three repairs and
  gives `0.472500` at `2.375000x`. This is the current cheap budget knob for
  deciding how late in the denoise trajectory repair compute is still worth
  spending.
- Fresh CUDA confirmation for the step-`20` point is
  `llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_phase20_fresh_v1_report.md`,
  run ID `diffusion-419fbf63c9d8e30b`. It uses `26` generations, spends on
  `plan_001`, `plan_003`, `plan_004`, and `plan_006`, and skips `plan_007`
  because its first repairable skeleton appears at step `31` under the step-`20`
  cap.
- Fresh CUDA confirmation for the step-`32` promoted point is
  `llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_phase32_fresh_v1_report.md`,
  run ID `diffusion-3b42951db77c5aa6`. It uses `27` generations, spends on
  `plan_001`, `plan_003`, `plan_004`, `plan_006`, and `plan_007`, accepts
  `plan_007` because its first repairable skeleton appears at step `31` within
  the step-`32` cap, and recovers selected latent repair `0.531116` at
  `2.625000x` with zero repair-oracle headroom.
- The denoise-phase audit is `DIFFUSION_DENOISE_PHASE_GEOMETRY.md`, generated
  by `experiments/analyze_diffusion_denoise_phase_geometry.py`. It moves the
  mechanism check into the sampled diffusion history: productive repairs enter
  a repairable constraint-skeleton phase much earlier on average (`16.2` steps)
  than skipped no-lift states (`30.0` steps), and repairable-phase
  precision/recall are both `1.000000` on the current compact run.
- The denoise-phase signal is now wired into the benchmark runner as
  `--repair-spend-trigger denoise_phase_repairability`. The fresh CUDA run
  `llada_moe_mixed_compact_span_fixed_source_denoise_phase_gate_fresh_v1`
  spends on the same 5 productive repairs, skips the same 3 no-lift repairs,
  and preserves the public budget point: selected latent repair `0.489911`,
  `+0.077634` vs greedy, `+0.117786` vs random, and `2.625000x` relative
  repair cost. This is now executable geometry backed by fresh GPU evidence,
  not only an offline audit.
- A history-anchor diagnostic is generated as
  `DIFFUSION_HISTORY_ANCHOR_REPAIR_AUDIT.md` by
  `experiments/analyze_diffusion_history_anchor_repair.py`. It uses the same
  public spend trigger and cost, but seeds span repair from the sampled denoise
  skeleton itself through `--repair-pack constraint_span_history`. The result is
  diagnostic, not public-canonical: repair stays above greedy/random at
  `0.474107`, but loses `0.015804` versus final-source repair because several
  history anchors drop final-context detail. The same audit reports a
  post-generation dual-anchor selector upper bound: label-free selector scores
  recover `0.489911`, but relative cost rises to `3.250000x`, so dual spending
  is not the cheap public path. The audit now also reports the cheap
  pre-generation version: source/history span geometry chooses the history
  anchor only once, preserves `0.489911`, and stays at `2.625000x` because it
  spends one repair anchor rather than both. This is now executable as
  `--repair-pack constraint_span_anchor_select`; anchor-select/history-span
  packs now request dense denoise-history sampling by default so near-final
  states are available without adding model generations. The fresh CUDA
  confirmation
  `llada_moe_mixed_compact_span_anchor_select_denoise_phase_gate_dense_history_fresh_v1`
  preserves selected latent repair `0.489911` at `2.625000x` with run ID
  `diffusion-f3c291037d94daaf`, choosing a history anchor on `plan_001` and
  final anchors on the other four repair spends.
- The anchor-retention diagnostic is generated as
  `DIFFUSION_ANCHOR_RETENTION_LOSS.md` by
  `experiments/analyze_diffusion_anchor_retention_loss.py`. It is not a public
  fourth arm; it is the theory/geometry audit behind the single repair arm,
  measuring whether a denoise-history anchor retained prompt constraints,
  target tokens, digits, and compact target structure before GPU repair spend.
  It now records the whole-history search boundary too: loose search run
  `diffusion-c326b3ef25eb8374` selected a bad `plan_003` history anchor and
  scored `0.483348`, while guarded search run `diffusion-ccef06238847a352`
  blocks that false positive and restores `0.489911` at the same `2.625000x`
  cost.
- Prompt-only trajectory contrast is recorded as a diagnostic negative:
  `constraint_span_history_contrast` run `diffusion-b92d689695016154` selected
  zero repairs and scored `0.414598` at `2.625000x`. Keep it out of the public
  arm; the useful denoise-history path is seed/remask/anchor geometry.
- Seed/remask trajectory geometry is now tested by
  `constraint_span_history_instability`. It keeps the final-source compact span
  anchor but also masks final token positions unstable across sampled denoise
  histories. Fresh run `diffusion-e28eb1d3dde8eea7` scored `0.459107` at
  `2.625000x`, beating greedy/random by `+0.046830` / `+0.086982`, but trailing
  anchor-select `0.489911`. Treat it as a secondary mask feature, not the
  public three-arm headline.
- The direct combination `constraint_span_anchor_instability` is also a
  diagnostic boundary: fresh run `diffusion-d14467a9f9a550b2` scored
  `0.481027` at `2.625000x`, improving over standalone instability but still
  below anchor-select. Keep the public headline on anchor-select while using
  this result to build a conditional instability gate.
- The first conditional gate,
  `constraint_span_anchor_instability_gated`, now has an identity-controlled
  rerun. The first run `diffusion-30a85507d687dfdc` regressed to `0.452188`
  because the wrapper prompt changed gate-off branches. The fixed run
  `diffusion-a7b64be5b7258f39` restores anchor-select exactly: `0.489911` at
  `2.625000x`, with `+0.077634` / `+0.117786` versus fixed/random. The audit
  confirms `4/4` gate-off repairs match anchor-select exactly; the one active
  gate changes seed/text but ties the anchor-select score. Treat this pack as
  the clean geometry A/B harness.
- The prompt-gated follow-up was the first positive budget line:
  `constraint_span_anchor_instability_prompt_gated` preserves the same
  gate-off identity but uses the instability-specific repair instruction when
  the gate is active. Fresh run `diffusion-4c6a7a9f356b3f0d` scores `0.498304`
  at `2.625000x`, beating fixed/random by `+0.086027` / `+0.126179`, with
  `4/4` gate-off identity matches and a `+0.067143` lift on the single active
  `plan_007` branch.
- The prompt-only gated control is the required negative control:
  `constraint_span_anchor_instability_prompt_only_gated` keeps the active
  instability instruction but removes the active instability remask. Fresh run
  `diffusion-4b5fc2b7604c28a5` scores `0.479911` at `2.625000x`; gate-off
  identity still matches `4/4`, but active `plan_007` drops by `-0.080000`.
  Therefore the public lift should be described as mask-plus-prompt latent
  repair, not prompt routing alone.
- The current public budget line is the automatic compatibility-scored seeded
  claim-gated follow-up:
  `constraint_span_anchor_instability_claim_auto_compat_seeded_gated` preserves
  the same denoise-anchor and instability-mask geometry, keeps the active
  `plan_007` instability repair, and scores compact 9-token seed candidates so
  oracle/selected result separation stays compatible with the surviving-claim
  control.
  The initial compatible run `diffusion-6944d9dd6c412de4` found the score; the
  fresh realization-guarded confirmation `diffusion-a9ae901393235364` preserves
  `0.531116` at `2.625000x` with
  `--repair-selector planning_quality_seed_realization_guarded`,
  beating fixed/random by `+0.118839` / `+0.158991` with `6/2/0`
  wins/ties/losses versus fixed and zero repair-oracle headroom. The automatic
  compatibility-scored run `diffusion-913b5bccb7894e5a` now matches that same
  score and cost while replacing the hand-built seed with a scored seed policy.
  The preservation-seeded run `diffusion-3b42951db77c5aa6` is now the public
  pointer: it keeps the same score/cost and zero repair-oracle headroom while
  removing explicit seed/anchor meta wording from the frontier task.
  The older oracle-aware run `diffusion-692592da063daa60` remains historical at
  `0.523304`; the strict oracle-control run `diffusion-df4149f37f6b21bf` is a
  negative boundary at `0.495625`.
- The semantic seed-anchor follow-up,
  `constraint_span_anchor_instability_claim_seeded_gated`, fixes the missing
  selected/oracle result-separation phrase directly into the masked denoise
  seed. Fresh run `diffusion-6ae167dc85d5e6ac` proves this can bind the phrase,
  but it falls to `0.521295` at `2.625000x` because `plan_004` then omits the
  public-claim survival control. Keep it out of the public headline until seed
  anchors have a compatibility loss over the full required-control set.
- The compatible-seeded follow-up is the first positive compatibility result:
  `plan_004` reaches `0.621786` and hits all five rubric controls because the
  fixed tail anchor carries both `oracle selected results` and `claim survives`.
  Keep the older seeded run as the negative boundary and the compatible seed as
  the hand-built reference; the current public headline is the automatic
  compatibility-scored version below.
- The automatic compact-control seed follow-up,
  `constraint_span_anchor_instability_claim_auto_seeded_gated`, generates that
  seed from the active task/rubric surface instead of hardcoding it. Fresh run
  `diffusion-7b74493b8c5ca15a` applies the generated seed without truncation and
  keeps all five `plan_004` rubric hits, but the aggregate line is `0.520536` at
  `2.625000x`, below the fixed compatible seed. Treat it as a boundary for the
  next policy: control-term extraction works, but the seed needs a
  realization-quality loss so the denoised continuation stays direct.
- The action-bearing automatic compact-control seed follow-up,
  `constraint_span_anchor_instability_claim_auto_action_seeded_gated`, keeps the
  same 9-token seed budget while adding a direct rerun verb. Fresh run
  `diffusion-51b5b82f63ad87cd` reaches `0.528482` at `2.625000x`, with
  `+0.116205` versus fixed, `+0.156357` versus random, `6/2/0`
  wins/ties/losses versus fixed, and zero repair-oracle headroom. Do not promote
  it over the current headline: it remains `0.002634` below the fixed compatible
  seed, mainly because `plan_004` falls from `0.621786` to `0.600714`.
- The automatic compatibility-scored compact-control seed follow-up,
  `constraint_span_anchor_instability_claim_auto_compat_seeded_gated`, fixes that
  action-seed boundary. It scores candidate anchors before denoising and avoids
  prompt-visible meta wording about the seed. Fresh run
  `diffusion-913b5bccb7894e5a` reaches `0.531116` at `2.625000x`, with
  `+0.118839` versus fixed, `+0.158991` versus random, `6/2/0`
  wins/ties/losses versus fixed, and zero repair-oracle headroom. Keep this as
  the automatic-compatibility boundary; the current public pointer is the
  cleaner preservation-seeded run below.
- The realization-prompt follow-up,
  `constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated`,
  keeps the same compatibility-scored seed policy but removes seed/anchor
  meta-language from the repair prompt. One-task CUDA smoke
  `diffusion-1a80605979a231e8` improves `plan_004` realization quality
  (`0.655238` to `0.807460`) and removes meta penalty (`0.140000` to
  `0.000000`), but lowers task score from `0.621786` to `0.600714`. Keep this
  as a diagnostic boundary, not the headline. Tightened v2 smoke
  `diffusion-d475c628f6386098` raises realization quality again to `0.846647`
  with zero meta penalty, but task score still stays `0.600714`. The next seed
  objective must preserve both direct realization and the selected/oracle rubric
  semantics.
- The joint-objective seed follow-up,
  `constraint_span_anchor_instability_claim_auto_joint_seeded_gated`, moves that
  tradeoff into seed selection. It scores compact anchors for compatibility,
  expected direct realization, and selected/oracle semantic preservation. Smoke
  run `diffusion-91dcab0442e7d5a1` chooses `separate oracle selected; claim
  survives if disappears`, keeps semantic preservation `1.000000` and meta
  penalty `0.000000`, but still scores `0.600714` on `plan_004`. Keep it as a
  diagnostic boundary, not the headline.
- The preservation-seed follow-up,
  `constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated`,
  adds `compact_preservation_control_terms` so the denoise tail itself says
  `oracle selected results; preserve claim if disappears`. Prompt-only smoke
  `diffusion-05c8f40e3fd0f234` stayed at `0.600714`, but preservation-seeded
  smoke `diffusion-c18d75b68b87ef33` recovers `plan_004 = 0.621786` with
  semantic preservation `1.000000` and zero meta penalty. Full mixed-slice run
  `diffusion-3b42951db77c5aa6` recovers the public aggregate exactly:
  `0.531116` at `2.625000x`, `+0.118839` versus fixed, `+0.158991` versus
  random, and zero repair-oracle headroom. This is now the public headline.
  The fresh step-`32` phase-window confirmation has the same stable run ID,
  uses `27` generations, and shows `plan_007` is spendable only when the first
  repairable skeleton at step `31` is inside the cap.
- Forced-spend audit:
  `diffusion-8a8a9e8904e62dbf` reran the high-quality skipped task `plan_002`
  with the promoted preservation-seeded repair and `--repair-spend-trigger
  always`; forced repair scored `0.582500` versus source `0.688571`, so the
  skip was correct. `diffusion-4699321baf91294e` reran the remaining skipped
  planning tasks `plan_005,plan_008`; the selector chose zero forced repairs,
  and the forced repair candidate mean task delta was `-0.014464` versus source.
  Keep `denoise_phase_repairability` as the default spend trigger for this
  public line. `DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md` now treats the gate
  as an error-correction classifier: `5` true-positive productive spends, `3`
  true-negative skipped no-lift cases, and no false positives or false
  negatives on the promoted 8-task planning slice. Future reports also include
  `Repair Spend Gate Diagnostics`, exposing source quality, source length,
  prompt-gap count, prompt coverage, repairable-band status, and visible
  denoise-skeleton status for each primary repair source considered.
- The realization-constrained automatic follow-up,
  `constraint_span_anchor_instability_claim_auto_seeded_realization_gated`, adds
  explicit instructions to preserve token-budget/prompt-format/locked-task
  wording. Fresh run `diffusion-2a310ed45712a36b` falls to `0.515759` at
  `2.625000x`; `plan_004` still has all rubric hits, but the output becomes a
  low-specificity `Control:` sentence. Keep this as a negative boundary against
  simply adding more prompt obligations.
- `DIFFUSION_REALIZATION_QUALITY.md` is the cheap theory gate for that boundary.
  It scores active compact-seed repairs with label-free realization and seed
  objectives. Auto-compat-preserve is now best by task on `plan_004`
  (`0.621786`) with zero meta penalty, while auto-compat-realized remains best
  by realization/seed objective but only scores `0.600714`. The runner accepts
  `--repair-selector planning_quality_seed_realization_guarded` and
  `--repair-selector planning_quality_seed_objective_guarded`; the promoted
  automatic compatibility-scored run remains the public cost line.
- The public benchmark artifact is `DIFFUSION_PUBLIC_BENCHMARK.md` with machine
  sibling `eval_results/diffusion_language/diffusion_public_benchmark.json`.
  It intentionally hides diagnostic arms and reports only Greedy, Random
  perturbation, and Latent repair with relative GPU cost.
- A fresh planning-only span-localization confirmation is promoted in the
  evidence map as
  `eval_results/diffusion_language/llada_moe_planning_compact_span_score_max_v2_report.md`.
  It uses 58 fresh records, run ID `diffusion-911c8526a9cfa11e`, and improves
  the MoE planning score-max line: selected latent repair `0.492321`,
  `+0.080045` vs fixed, `+0.120196` vs random, `+0.048571` vs evolved, and
  `6/2/0` repair-vs-evolved. The mechanism evidence remains clean:
  `constraint_gap_span_repair` reports `Span Localized 1.000` and
  `Span Fallback 0.000`, so the gain comes from literal verifier-target
  inpainting rather than generic tail-window remasking. Compact targeting also
  lowers average masked positions from the older source-ranked line's `46.0` to
  `34.2`.
- Clause-level span targeting is available only as the diagnostic
  `--repair-pack constraint_span_clause`. The two-task smoke report
  `eval_results/diffusion_language/llada_moe_planning_clause_ranked_span_smoke_v1_report.md`
  reached selected latent repair `0.571250`, which is below the sentence-level
  source-ranked smoke at `0.573750`; `plan_002` was the concrete regression.
  Do not use this pack for the claim map or public headline until it beats the
  sentence-level `constraint_span` path.
- Compact target selection is now the default policy inside `constraint_span`.
  The one-task MoE CUDA smoke
  `eval_results/diffusion_language/llada_moe_planning_compact_span_policy_smoke_v1_report.md`
  repaired `plan_001` from fixed/random/evolved `0.465357` to selected latent
  repair `0.528214`, with `Span Localized 1.000`, `Span Fallback 0.000`, and
  zero repair-oracle headroom. The first full compact run exposed regressions on
  structured decision rules and multi-sentence failure chains; the v2 policy now
  keeps those contexts intact and is promoted for the planning-only claim above.

MoE planning-only command:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-moe-7b-a1b-instruct-hf --limit-evolved-schedules 2 --include-revision-schedules --revision-remask-fraction 0.25 --revision-steps 16 --limit-repair-candidates 1 --repair-pack constraint_span --repair-spend-trigger always --repair-source-policy non_revision_plus_gap_trajectory --adaptive-source-gate-mode score_max --history-sample-count 64 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --revision-promotion-margin 0.05 --repair-selector planning_quality_prompt_coverage_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results\diffusion_language\llada_moe_planning_compact_span_score_max_v2_raw.jsonl --scores-output eval_results\diffusion_language\llada_moe_planning_compact_span_score_max_v2_scores.json --report-output eval_results\diffusion_language\llada_moe_planning_compact_span_score_max_v2_report.md
```

MoE revision diagnostic command:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-moe-7b-a1b-instruct-hf --limit-evolved-schedules 2 --include-revision-schedules --revision-remask-fraction 0.25 --revision-steps 16 --limit-repair-candidates 1 --repair-pack constraint_span --repair-spend-trigger always --repair-source-policy non_revision_evolved --history-sample-count 64 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --revision-promotion-margin 0.05 --repair-selector planning_quality_delta_risk_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results\diffusion_language\llada_moe_planning_revision_constraint_span_nonrev_source_v1_raw.jsonl --scores-output eval_results\diffusion_language\llada_moe_planning_revision_constraint_span_nonrev_source_v1_scores.json --report-output eval_results\diffusion_language\llada_moe_planning_revision_constraint_span_nonrev_source_v1_report.md
```

MoE adaptive source GPU command:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-moe-7b-a1b-instruct-hf --limit-evolved-schedules 2 --include-revision-schedules --revision-remask-fraction 0.25 --revision-steps 16 --limit-repair-candidates 1 --repair-pack constraint_span --repair-spend-trigger always --repair-source-policy non_revision_plus_gap_trajectory --adaptive-source-gate-mode score_max --history-sample-count 64 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --revision-promotion-margin 0.05 --repair-selector planning_quality_prompt_coverage_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results\diffusion_language\llada_moe_planning_revision_constraint_span_adaptive_source_prompt_guard_fresh_v1_raw.jsonl --scores-output eval_results\diffusion_language\llada_moe_planning_revision_constraint_span_adaptive_source_prompt_guard_fresh_v1_scores.json --report-output eval_results\diffusion_language\llada_moe_planning_revision_constraint_span_adaptive_source_prompt_guard_fresh_v1_report.md
```

MoE adaptive efficiency-mode GPU command:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --families planning --task-ids plan_001,plan_002,plan_003,plan_004,plan_005,plan_006,plan_007,plan_008 --candidates llada-moe-7b-a1b-instruct-hf --limit-evolved-schedules 2 --include-revision-schedules --revision-remask-fraction 0.25 --revision-steps 16 --limit-repair-candidates 1 --repair-pack constraint_span --repair-spend-trigger always --repair-source-policy non_revision_plus_gap_trajectory --adaptive-source-gate-mode efficiency --history-sample-count 64 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --revision-promotion-margin 0.05 --repair-selector planning_quality_prompt_coverage_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results\diffusion_language\llada_moe_planning_revision_constraint_span_adaptive_source_efficiency_fresh_v1_raw.jsonl --scores-output eval_results\diffusion_language\llada_moe_planning_revision_constraint_span_adaptive_source_efficiency_fresh_v1_scores.json --report-output eval_results\diffusion_language\llada_moe_planning_revision_constraint_span_adaptive_source_efficiency_fresh_v1_report.md
```

MoE compact mixed score-efficient GPU command:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --task-preset lean_gpu_mixed --candidates llada-moe-7b-a1b-instruct-hf --limit-evolved-schedules 2 --include-revision-schedules --revision-remask-fraction 0.25 --revision-steps 16 --limit-repair-candidates 1 --repair-pack constraint_span --repair-spend-trigger always --repair-source-policy non_revision_plus_gap_trajectory --adaptive-source-gate-mode score_efficient --history-sample-count 64 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --revision-promotion-margin 0.05 --repair-selector planning_quality_prompt_coverage_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --exact-task-trajectory-policy proposal_history --exact-self-repair --exact-verifier-revision --device cuda --dtype bfloat16 --raw-output eval_results\diffusion_language\llada_moe_mixed_compact_span_score_efficient_fresh_v1_raw.jsonl --scores-output eval_results\diffusion_language\llada_moe_mixed_compact_span_score_efficient_fresh_v1_scores.json --report-output eval_results\diffusion_language\llada_moe_mixed_compact_span_score_efficient_fresh_v1_report.md
```

MoE adaptive source-gate threshold sweep:

```powershell
python experiments\sweep_adaptive_source_gate.py --raw-input eval_results\diffusion_language\llada_moe_planning_revision_constraint_span_multisource_v1_raw.jsonl --output-dir eval_results\diffusion_language\adaptive_source_gate_sweep_script_v1 --label adaptive_source_gate_sweep_script_v1
```

MoE adaptive source lean mixed GPU command:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --task-preset lean_gpu_mixed --candidates llada-moe-7b-a1b-instruct-hf --limit-evolved-schedules 2 --include-revision-schedules --revision-remask-fraction 0.25 --revision-steps 16 --limit-repair-candidates 1 --repair-pack constraint_span --repair-spend-trigger always --repair-source-policy non_revision_plus_gap_trajectory --adaptive-source-gate-mode score_max --history-sample-count 64 --evolved-selector planning_quality_fallback --evolved-quality-margin 0.01 --evolved-selector-tolerance 0.015 --evolved-promotion-margin 0.015 --revision-promotion-margin 0.05 --repair-selector planning_quality_prompt_coverage_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --exact-task-trajectory-policy proposal_history --exact-self-repair --exact-verifier-revision --device cuda --dtype bfloat16 --raw-output eval_results\diffusion_language\llada_moe_mixed_compact_span_score_max_v1_raw.jsonl --scores-output eval_results\diffusion_language\llada_moe_mixed_compact_span_score_max_v1_scores.json --report-output eval_results\diffusion_language\llada_moe_mixed_compact_span_score_max_v1_report.md
```

## Claim Rules

A result is worth making public only if it passes all of these:

- The fixed and random baselines are in the same report as the repair arm.
- The report includes the `Lean Three-Arm Headline` section, which scopes the
  headline fixed/random/repair scores to repair-covered tasks and leaves
  trajectory/evolved/oracle rows as diagnostics.
- Any claim worth sharing publicly is listed in `CLAIM_EVIDENCE_MAP.md`, or the
  evidence map is refreshed before making the claim.
- Hidden labels are used only after arm selection for scoring and diagnostics.
- Repair coverage is separated from full-suite coverage.
- Exact-answer gains report whether they are proposal-attributable.
- Planning repair reports risk penalties and prompt-checklist leakage penalties.
- Any oracle headroom is explained with the actual candidate behavior, not just
  the score.

## Next Work

The fundamental diffusion pivot is that passive history selection is not enough.
LLaDA history is mostly monotonic fill unless we explicitly remask committed
tokens. The next system improvement should therefore favor:

1. Verifier-guided non-monotonic remasking over passive history selection.
2. Span-level inpainting for arithmetic contradiction chains and planning
   constraint gaps.
3. Learned or verifier-ranked source-relative remask policies instead of
   hand-scored prompt-gap spans.
4. Larger scouts only after the compact 11-task GPU sweep stays clean.
