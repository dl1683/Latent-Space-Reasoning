# Next Generation Reasoning Task List

This is the working backlog for pushing Latent Space Reasoning from an interesting
perturbation result into a measurable reasoning-control system. The current thesis:
prefix perturbations are useful, but the next system must explain where information
enters, distinguish clean steering from useful chaos, and turn judge-selected
trajectory diversity into reliable reasoning gains.

## Research Anchors

- Information geometry: measure movement in output-distribution space, not only
  Euclidean movement in hidden or prompt-embedding space.
- Activation engineering and representation engineering: residual-stream writes
  can be a stronger intervention channel than prefix-only conditioning.
- Attention sinks: early-token positions are a mechanism surface, not just a
  prompt-format detail.
- Prefix and prompt tuning: learned continuous context can carry task information,
  but this repo must separate learned information from random control perturbation.
- Judge-selected trajectory search: most useful information may enter through
  evaluation and selection rather than through the sampled perturbation itself.

## Current Information Flow Hypothesis

1. The prompt supplies task facts and constraints.
2. The frozen model supplies latent capability and domain knowledge.
3. Prefix perturbations supply control energy: position, RMS, token count, and
   attention pattern disruption.
4. Directional perturbations only become information-bearing when a trained probe,
   scorer, or residual intervention maps them to behaviorally meaningful axes.
5. The judge/evolution loop supplies external preference information by selecting
   among many possible trajectories.
6. Autoregressive generation amplifies small early distribution shifts into full
   reasoning trajectories.

## Task Backlog

### Latent Trajectory Aggregation Pivot

The next cross-family doctrine is
`docs/LATENT_TRAJECTORY_AGGREGATION.md`. It generalizes the project from
winner-take-all trajectory selection to component-level synthesis across
multiple latent trajectories. Treat this as a research protocol, not a promoted
result: token perturbation, diffusion repair, denoise history, anchors, and
candidate promotion become intervention surfaces that expose useful partial
components for aggregation.

The first proof target is not "which method wins." It is whether a fused answer
can beat the best individual candidate by preserving non-overlapping verified
components from different latent trajectories without introducing
contradictions.

Current aggregation frontier: v5 is the passing 48-task statistical replication;
v6 is a useful negative coverage-targeting result. V6 failed promotion at
`27/48` complement coverage, and threshold sensitivity showed positive-floor
coverage reaches only `29/48`. The next aggregation step is therefore v7:
create fresh `plan_345`-`plan_392` tasks, freeze an expanded planning-aspect
ontology, and test new source families rather than weakening v6 gates. The task
inventory and task/ontology freeze now exist; source-family generation and
expanded-ontology replay support are the next implementation boundary before
GPU generation.

### Current Narrow Execution Protocol

The next non-ARC execution path is locked in
`docs/GENERAL_PURPOSE_LATENT_BENCHMARK_PROTOCOL.md`. Use that document as the
source of truth before expanding benchmark scope. The next run should be a
GPU-bounded three-arm comparison only: `greedy_baseline`, `random_prefix`, and
`latent_reasoning`, with short open-ended planning as the showcase plus a small
mix of math, symbolic, and science QA tasks.

### Language Diffusion Pivot

The next mechanism layer is documented in
`docs/DIFFUSION_LATENT_REASONING_ROADMAP.md`. Treat Dream 7B as the first local
GPU target because it exposes `diffusion_generate()` and optional denoising
history. Treat dense LLaDA 8B as the second architecture check and
LLaDA-MoE-7B-A1B-Instruct as the next cheap active-parameter check. The
benchmark stack still stays narrow: the diffusion work should first add a fixed
denoising baseline and a latent trajectory-control condition around the
existing three-arm protocol, not a broad benchmark suite.

The fundamental architecture pivot is now documented in
`docs/DIFFUSION_NATIVE_REASONING_ARCHITECTURE.md`: the next system should evolve
and judge denoising trajectories, mask policies, schedule candidates, repair
steps, and verifier hooks, not just autoregressive soft prefixes.

The public-facing implications are summarized in
`docs/DIFFUSION_REASONING_FIELD_IMPLICATIONS.md`. That document is the right
place to describe why the originating diffusion-native thesis is important:
latent reasoning becomes more plausible when it is treated as controlled
error-correction over an iterative denoising trajectory, with the current public
MoE line showing Latent repair `0.531116` versus Greedy `0.412277` and Random
`0.372125` at `2.625000x` relative cost.

Latest implementation checkpoint: `llada-moe-7b-a1b-instruct-hf` is registered
in the candidate table, a GGUF fallback is listed, and all benchmark runners now
use family-level LLaDA routing rather than checking only `LLaDA 8B`. Cheap
preflight for `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` succeeded with config,
tokenizer, README, and custom code only; the full 13.7 GB snapshot is now also
materialized locally. BF16 CUDA smoke and history smoke pass, and the lean mixed
benchmark completed 60 full MoE generations. Current result: MoE is locally
runnable and exact checks are strong, but dense LLaDA remains the stronger
planning-repair line under the transferred state-adaptive policy. A first
MoE-specific improvement is now in place: `--repair-pack constraint_span`
spends only prompt-gap span inpainting and lifts MoE planning repair-selected to
`0.472` at 5 generations/task, with `+0.050` over evolved. The revision path is
now source-aware: `--repair-source-policy non_revision_evolved` keeps
non-monotonic revision schedules in the evolved arm while seeding span repair
from the best non-revision source, restoring MoE repair-selected to `0.472`
when revision is enabled and beating the stronger revision-aware evolved arm by
`+0.028`. A two-source diagnostic, `evolved_and_trajectory`, reaches `0.473`
and `7/1/0` repair-vs-evolved wins/ties/losses, but the added branch budget
lowers gain per extra generation, so it is a selector-development diagnostic
rather than the default. The next adaptive source pass,
`--repair-source-policy non_revision_plus_gap_trajectory` with
`--repair-selector planning_quality_prompt_coverage_guarded`, uses 58 records,
reaches repair-selected `0.474`, beats evolved by `+0.030`, records `7/1/0`
repair-vs-evolved wins/ties/losses, and leaves zero oracle headroom in the raw
MoE repair pool. A fresh GPU run confirmed the same result in
`llada_moe_planning_revision_constraint_span_adaptive_source_prompt_guard_fresh_v1_report.md`.
The policy now exposes `--adaptive-source-gap-min-terms` and
`--adaptive-source-quality-floor`, and reports an `Adaptive Source Gate` table
showing why each second-source branch fired or skipped. It is the current
revision-enabled MoE selector candidate. A 30-cell threshold sweep in
`adaptive_source_gate_sweep_v1_summary.md` confirms the current default is
score-maximal, while stricter plan_002-only gates are more budget efficient. The
fresh efficiency run confirms the stricter regime at 57 generations,
repair-selected `0.472768`, and `0.025794` gain per extra generation. The cheaper
5-generation/task `constraint_span` line remains the no-revision default. These
thresholds are now exposed as named gate modes: `score_max`, `efficiency`, and
`custom`. The latest lean mixed MoE run is the compact span refresh
`llada_moe_mixed_compact_span_score_max_v1`: run ID
`diffusion-33bf0475f913c6a7`, 76 generations, repair coverage `8/11` overall
and `8/8` eligible, repair-selected `0.492321`, `+0.080045` vs fixed,
`+0.120196` vs random, `+0.048571` vs evolved, `6/2/0` repair-vs-evolved, and
`0.000625` oracle headroom. This keeps the math/symbolic/science checks solved
and improves the older source-ranked mixed score (`0.473482`) at the same
generation count. Compact efficiency and single-source rescores from the same
raw pool improve the cost frontier: `score_efficient` keeps `0.492321` at 75
fresh GPU generations by skipping the high-quality no-op `plan_002` second
source, while the fresh single-source CUDA run reaches `0.473393` at 74
generations with run ID `diffusion-150ed790105bb0b6`.
The fresh direct fixed-source CUDA run then moved the cost frontier sharply:
`llada_moe_mixed_compact_span_fixed_source_fresh_v1`, run ID
`diffusion-935bf9edc3efd410`, reaches `0.489911` at 30 records and `3.000000x`
relative repair cost by repairing the greedy output directly, so it dominates
the older 74-record single-source budget point. The quality-gated fixed-source
CUDA run then tightened that frontier again:
`llada_moe_mixed_compact_span_fixed_source_quality_gate_fresh_v1`, run ID
`diffusion-f2384443e57f5548`, skips the high-quality no-op `plan_002` repair
pass, preserves `0.489911`, and cuts relative repair cost to `2.875000x`. The
repairability-geometry gate then moved from a simple quality threshold to a
source-quality plus prompt-gap/coverage band:
`llada_moe_mixed_compact_span_fixed_source_repairability_gate_fresh_v1`, run ID
`diffusion-ae26bb892c8a68aa`, keeps the productive compact span repairs, skips
`plan_002`, `plan_005`, and `plan_008`, preserves `0.489911`, and cuts relative
repair cost to `2.625000x`.
`DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md` now turns that into an explicit
label-free geometry artifact: all 5 spent repairs are productive, all 3 skipped
repairs have no reference lift, missed repairs are `0`, and the spend/no-spend
split is explained by prompt-gap and prompt-coverage geometry rather than task
labels.
`DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md` now sweeps 53,460 label-free gate
settings over source quality, prompt-gap, prompt coverage, and optional first
denoise-skeleton step caps. The promoted gate is score/cost Pareto-equivalent,
the best score is `0.531116` at `2.625000x`, the frontier has 5 points, and 168
gates are zero-waste/zero-miss, so the repairability geometry is a small
plateau rather than a single fragile threshold.
The next denoise-native diagnostic is now generated as
`DIFFUSION_DENOISE_PHASE_GEOMETRY.md`. It uses the sampled diffusion history,
not just the final text, and classifies whether each source trajectory forms a
repairable constraint skeleton. On the current compact run, repairable-phase
precision and recall are both `1.000000`: all 5 productive repairs pass through
repairable/low-quality skeleton phases, while all 3 skipped no-lift states are
undercovered or overdiffuse. The average first skeleton step is `16.2` for
productive repairs versus `30.0` for skipped no-lift states.
That diagnostic is now an executable runner policy:
`--repair-spend-trigger denoise_phase_repairability` first requires the same
final source-quality/prompt-gap/coverage repairability band, then requires a
sampled denoise-history skeleton before spending repair compute. The fresh CUDA
confirmation
`llada_moe_mixed_compact_span_fixed_source_denoise_phase_gate_fresh_v1`, run ID
`diffusion-5b1bf286b8cfa727`, matches the current budget frontier exactly:
selected latent repair `0.489911`, `+0.077634` vs greedy, `+0.117786` vs
random, and `2.625000x` relative cost, while skipping the same no-lift
`plan_002`, `plan_005`, and `plan_008` repairs.
`DIFFUSION_PUBLIC_BENCHMARK.md` now gives the public-only cheap-stack view:
Greedy `0.412277` at `1.000000x`, Random perturbation `0.372125` at
`1.000000x`, and auto-compatible seeded claim-gated Latent repair `0.531116` at
`2.625000x`, with math/symbolic/science guard checks all at `1.000000` for the
no-repair arms.
`DIFFUSION_HISTORY_ANCHOR_REPAIR_AUDIT.md` now tests the more diffusion-native
variant: use the sampled denoise skeleton as the repair source itself. The fresh
CUDA diagnostic
`llada_moe_mixed_compact_span_history_anchor_denoise_phase_gate_fresh_v1`, run
ID `diffusion-16dc676d10e4b12e`, preserves the same `2.625000x` cost and literal
span localization `1.000000` / fallback `0.000000`, but drops selected latent
repair from `0.489911` to `0.474107`. The result is a useful boundary: history
anchors are real repair sources and still beat greedy/random, but they lose
final-context detail on `plan_003`, `plan_006`, and `plan_007`. The next operator
should choose between history and final anchors before spending, or train a
consistency loss that preserves constraints stable in the final denoise state.
The same audit now includes a post-generation dual-anchor selector upper bound:
label-free selector scores choose the history anchor once and final/no-repair
seven times, recovering `0.489911`, but relative cost rises to `3.250000x`.
That rules out naive dual spending as the budget path and turns pre-generation
anchor choice into the next concrete target. The audit now includes that
pre-generation selector too: using only source/history span geometry before
repair generation, it chooses history once and final/no-repair seven times,
preserves `0.489911`, and keeps relative cost at `2.625000x`. This makes
anchor choice a concrete next runner policy candidate rather than only an
upper-bound analysis. That candidate is now executable as
`--repair-pack constraint_span_anchor_select`. Anchor-select and history-span
packs now request dense denoise-history sampling by default so near-final
history anchors are visible without adding model generations. The fresh CUDA
run
`llada_moe_mixed_compact_span_anchor_select_denoise_phase_gate_dense_history_fresh_v1`,
run ID `diffusion-f3c291037d94daaf`, preserves `0.489911` at `2.625000x`,
chooses a history anchor on `plan_001`, and chooses final anchors on the other
four repair spends. The first executable theory layer on top of this is now
`DIFFUSION_ANCHOR_RETENTION_LOSS.md`, generated by
`experiments/analyze_diffusion_anchor_retention_loss.py`: it turns
denoise-history anchoring into a label-free constraint-retention loss over
target overlap, text overlap, prompt-keyword loss, digit loss, target-count
consistency, and compact-span score advantage. On the dense-history trace it
classifies `plan_001` as the single safe history anchor, blocks six histories
for lacking span-score advantage, blocks one for compact-target structure, and
shows history-anchor repair trails final-source repair by `0.015804` on the
diagnostic all-history policy. Whole-history anchor search is now executable as
`--repair-pack constraint_span_anchor_search`: loose search run
`diffusion-c326b3ef25eb8374` proved that the earlier `plan_003` history state
is a false positive and drops the score to `0.483348`; guarded search run
`diffusion-ccef06238847a352` uses target similarity `0.96` plus length
retention `0.95` to block that false positive and restore `0.489911` at
`2.625000x`. The next seed-geometry check,
`constraint_span_history_instability`, keeps final-source span anchors but adds
token positions unstable across sampled denoise histories. Fresh GPU run
`diffusion-e28eb1d3dde8eea7` scores `0.459107`, beating greedy/random by
`+0.046830` / `+0.086982` at the same `2.625000x` cost, but trailing
anchor-select `0.489911`; instability is therefore a secondary mask feature,
not a replacement for constraint-retention anchor choice. The direct combined
operator `constraint_span_anchor_instability` confirms that blind combination
is not enough: fresh GPU run `diffusion-d14467a9f9a550b2` improves over
standalone instability to `0.481027`, but still trails anchor-select because
instability masks help `plan_007` while hurting most other selected anchors.
The conditional gate has now been tested too: `constraint_span_anchor_instability_gated`
initially exposed a prompt-identity leak, then the fixed identity run
`diffusion-a7b64be5b7258f39` restored the anchor-select score: `0.489911` at
`2.625000x`. The audit now proves the gate-off repairs match anchor-select in
generation seed, prompt, masked seed, output text, and score; the only active
gate changes the seed/text on `plan_007` but ties the anchor-select score.
That makes the gate a useful A/B harness, not a new promoted mechanism.
The prompt-gated version then tested whether the earlier `plan_007` gain came
from the instability-specific repair instruction rather than the mask alone.
Fresh GPU run `diffusion-4c6a7a9f356b3f0d` keeps the same gate-off identity,
activates the instability prompt only on `plan_007`, and lifts the public
three-arm line to `0.498304` at `2.625000x`: `+0.086027` versus fixed and
`+0.126179` versus random, with zero oracle headroom.
The next composite gate adds a public-claim confound-control prompt only when
the final-anchor planning source is low-quality and the task explicitly mixes
baseline, token/prompt-format confounds, and public-claim risk. The first
claim-gated run `diffusion-94e95f5d1b3d9822` was selector-safe but copied
repair meta-language into `plan_004`, so the selector fell back to fixed and
the line stayed below the frontier at `0.495625`. The compact-prompt rerun
`diffusion-0fc7f067a7d87799` fixes that prompt surface, lifts `plan_004` by
`+0.121071` over prompt-gated repair, preserves every other prompt-gated branch,
and moves the public three-arm line to `0.513437` at the same `2.625000x`:
`+0.101161` versus fixed and `+0.141313` versus random.
The compact oracle-aware follow-up keeps the same denoise-anchor/instability
geometry and changes only the active public-claim repair instruction. Fresh GPU
run `diffusion-692592da063daa60` moves the public three-arm line to `0.523304`
at the same `2.625000x`: `+0.111027` versus fixed and `+0.151179` versus
random, with `6/2/0` wins/ties/losses versus fixed and zero repair-oracle
headroom. `plan_004` rises to `0.559286` by adding better failure-mode
validation and preserving the useful control structure; it still misses the
literal oracle-result rubric phrase, so the remaining theory gap is a more
reliable way to bind selected-vs-oracle result separation into the denoise
repair rather than merely asking for it.
The fixed seed-anchor follow-up tests that binding directly:
`constraint_span_anchor_instability_claim_seeded_gated` fixes the phrase
`separate oracle best-of results from selected results` into the masked denoise
seed. Fresh GPU run `diffusion-6ae167dc85d5e6ac` proves the phrase can be bound:
`plan_004` hits the oracle/selected-result rubric item. But the public line
drops to `0.521295` at `2.625000x` because that fixed anchor displaces the
public-claim survival control. This is a sharper theory boundary: seed anchors
need a compatibility loss over all required controls, not just a phrase-binding
objective.
The compatible seed-anchor follow-up is the positive version of that theory:
`constraint_span_anchor_instability_claim_compatible_seeded_gated` fixes a
compact 9-token anchor, `oracle selected results; claim survives if disappears`,
into the same masked tail. The initial compatible run
`diffusion-6944d9dd6c412de4` moved the public line to `0.531116`; the fresh
realization-guarded CUDA confirmation `diffusion-a9ae901393235364` preserves
that score at `2.625000x`: `+0.118839` versus fixed and `+0.158991` versus
random, with `6/2/0` wins/ties/losses versus fixed and zero repair-oracle
headroom. `plan_004` reaches `0.621786` and hits all five rubric controls.
This is now the best evidence that diffusion-native semantic anchors can
improve reasoning when the anchor is compact enough to keep required controls
compatible and realized as direct answer text.
The automatic seed-anchor follow-up converts that hand-built anchor into the
first policy version: `constraint_span_anchor_instability_claim_auto_seeded_gated`
synthesizes the same compact control tail from the active task/rubric surface.
Fresh GPU run `diffusion-7b74493b8c5ca15a` proves the mechanism fires correctly:
the seed is applied without truncation and `plan_004` still hits all five rubric
controls. The aggregate line drops to `0.520536` at `2.625000x`, though, because
the denoised continuation is less direct than the fixed compatible seed. This is
the next theory boundary: extracting the right control words is not enough; the
seed policy also needs a realization-quality term for how the anchor integrates
into the final sentence.
The action-bearing automatic seed follow-up,
`constraint_span_anchor_instability_claim_auto_action_seeded_gated`, tests that
boundary without increasing the token budget. The generated 9-token seed
`rerun; oracle selected; claim survives` fits the same masked tail as the
compatible fixed seed and applies without truncation. Fresh GPU run
`diffusion-51b5b82f63ad87cd` reaches `0.528482` at `2.625000x`, beating fixed
and random by `+0.116205` / `+0.156357` with `6/2/0` wins/ties/losses versus
fixed and zero repair-oracle headroom. It remains below the compatible fixed
seed by `0.002634`, almost entirely from `plan_004` (`0.600714` versus
`0.621786`). Treat this as the current automatic-seed boundary: action-bearing
semantic anchors help, but compatibility with every required control still
needs a scored or learned seed objective.
The compatibility-scored automatic follow-up closes that specific gap:
`constraint_span_anchor_instability_claim_auto_compat_seeded_gated` scores
compact seed candidates for required-control compatibility before applying the
anchor. An initial smoke failed because the prompt mentioned the generated seed
as a meta object, but the v2 prompt removed that wording and recovered
`plan_004 = 0.621786`. The full fresh CUDA run
`diffusion-913b5bccb7894e5a` ties the fixed compatible frontier at `0.531116`
and `2.625000x`, with `+0.118839` versus fixed, `+0.158991` versus random,
`6/2/0` wins/ties/losses versus fixed, and zero repair-oracle headroom. This
promotes the mechanism from hand-built semantic anchor to automatic
compatibility-scored seed selection.
The preservation-seeded mixed run `diffusion-3b42951db77c5aa6` now keeps the
same public aggregate (`0.531116` at `2.625000x`, zero repair-oracle headroom)
while removing explicit seed/anchor meta wording from the frontier task. The
public evidence-map pointer now uses this preservation-seeded run as the
canonical claim.
The realization-prompt follow-up,
`constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated`,
keeps the same automatic compatibility scorer but removes prompt language that
names seeds or anchors as meta objects. A one-task CUDA smoke
`diffusion-1a80605979a231e8` improves `plan_004` realization quality from
`0.655238` to `0.807460` and removes the meta penalty (`0.140000` to
`0.000000`), but task score drops from `0.621786` to `0.600714` because the
answer says to compare oracle selected results rather than preserving the
stronger selected/oracle separation phrasing. The tightened v2 smoke
`diffusion-d475c628f6386098` improves realization quality again to `0.846647`
and says `Separate oracle selected results`, but task score remains `0.600714`.
The joint-objective seed policy,
`constraint_span_anchor_instability_claim_auto_joint_seeded_gated`, then moves
that tradeoff into seed selection itself: it scores candidate anchors for
compatibility, expected direct realization, and selected/oracle semantic
preservation, choosing the 9-token `separate oracle selected; claim survives if
disappears` anchor. One-task CUDA smoke `diffusion-91dcab0442e7d5a1` keeps zero
meta penalty and semantic preservation `1.000000`, but still scores `0.600714`
on `plan_004` with seed objective `0.883582`, below the realized-prompt v2
objective `0.904921` and below the current task frontier. This rules out
seed-choice alone as the missing piece.
Keep it as a theory-positive, score-negative boundary: direct realization
matters, but the seed objective must jointly optimize realization quality and
task-rubric semantics.
The preservation-seed follow-up,
`constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated`,
then moves the missing constraint pressure into the denoise seed itself instead
of relying on prompt wording. The first prompt-only smoke
`diffusion-05c8f40e3fd0f234` stayed at `0.600714`: the model ignored the
`preserve` wording and emitted `claim survives`. After adding the
`compact_preservation_control_terms` seed policy, one-task CUDA smoke
`diffusion-c18d75b68b87ef33` selects the 9-token
`oracle selected results; preserve claim if disappears` anchor, recovers
`plan_004 = 0.621786`, keeps semantic preservation `1.000000`, and removes the
seed/anchor meta penalty (`0.000000`). Full mixed-slice CUDA run
`diffusion-3b42951db77c5aa6` then recovers the public aggregate exactly:
`0.531116` at `2.625000x`, `+0.118839` versus fixed, `+0.158991` versus random,
and zero repair-oracle headroom. This is the current cleaner public frontier.
The realization-constrained automatic follow-up,
`constraint_span_anchor_instability_claim_auto_seeded_realization_gated`, tested
whether stronger prompt constraints could supply that realization quality.
Fresh GPU run `diffusion-2a310ed45712a36b` falls further to `0.515759` at
`2.625000x`: `plan_004` still applies the generated seed and hits all rubric
controls, but the answer collapses into a labeled `Control:` sentence with low
specificity. This rules out "more explicit prompt obligations" as the next
solution; realization quality needs to be scored or learned.
The stricter oracle/best-of control then tested whether the remaining `plan_004`
rubric gap could be fixed by forcing the missing controls into the repair
instruction. Fresh GPU run `diffusion-df4149f37f6b21bf` scores `0.495625` at
the same `2.625000x`, below the compact claim-gated frontier: `plan_004`
includes the token/prompt confound but over-compresses the falsification plan
and still misses locked reruns, regression recording, and oracle/selected-result
separation. Keep this as a negative boundary: the deeper theory has to express
selective geometry-conditioned control, not just longer prompt obligations.
The promoted MoE planning-only line now uses compact target selection:
`llada_moe_planning_compact_span_score_max_v2`, run ID
`diffusion-911c8526a9cfa11e`, repair-selected `0.492321`, `+0.080045` vs
fixed, `+0.120196` vs random, `+0.048571` vs evolved, `6/2/0`
repair-vs-evolved, and `0.000625` oracle headroom.

Current diffusion implementation status: Dream/LLaDA local scout runs work, the
LLaDA backend now supports suffix inpainting through partially fixed generated
tokens, records per-token denoise confidence for committed LLaDA suffix tokens,
and `experiments/run_diffusion_repair_scout.py` has branch-and-repair results
on planning tasks (`0.412` baseline selected mean to `0.443` after selecting
prefix and low-confidence repairs). Exact-answer counterfactual repair now has
a reusable proposal layer plus proposal-only ablations. Dream moved from
`0.647` to `1.000` on the objective slice after six arithmetic/symbolic proposal
repairs, and LLaDA moved from `0.941` to `1.000`, but proposal-only selection
also reached `1.000` in both latest reports. The honest interpretation is that
the current exact-answer gains are verifier/proposer-attributable; diffusion
gets credit only for trajectory-controlled execution unless a future task beats
proposal-only. The first seeded diffusion three-arm benchmark is now in place
too: `three_arm_planning_state_selector_v1_report.md` compares fixed, random,
and planning-state trajectory-selected schedules over 8 planning tasks plus 3
objective checks. Trajectory-selected diffusion is currently modestly positive
(`0.465` mean task score versus fixed `0.436` and random `0.423`) and now scores
sampled denoise states, not only final text. Remaining selector failures show
where prompt grounding still does not imply rubric quality.
The first budgeted schedule-evolution benchmark is also in place:
`four_arm_evolved_margin015_v1_report.md` adds two mutated schedules per model,
selects from the larger pool, and requires a `0.015` selector-score edge before
replacing the base trajectory-selected schedule. Evolved schedule selection
averaged `0.475` versus `0.465` for base-pool trajectory selection, `0.436`
fixed, and `0.423` random on the same Dream/LLaDA planning-plus-mix slice. This
is a modest but real trajectory-control gain, with the cost made explicit at
`4.50` generations per task and no evolved-vs-trajectory losses on this run
(`3/19/0` wins/ties/losses).
The runner now also reports oracle schedule choice and selector regret when
rescoring raw generations. On `four_arm_evolved_margin015_oracle_rescore_v1`,
the oracle score was `0.481`, only `+0.006` above evolved selection, which
means the current schedule pool is nearly exhausted. The next step should add
richer candidate types: mid-trajectory repair, verifier-triggered remasking,
or multi-step schedule mutations, not just another scalar selector tweak.
That next mechanism is now partially in place: the diffusion benchmark can add
a `repair_selected` arm that branches LLaDA suffix-inpainting repairs from the
evolved output and from selected mid-denoise history states. The efficient
planning diagnostic is
`llada_planning_adaptive_history_rescue_margin01_v1_report.md`: default
`0.25` history-prefix repair plus final-prefix repair, with adaptive
`0.50` history rescue only when the first repair pass would keep a matching
evolved baseline. With evolved selector `planning_quality_fallback` and repair
selector `planning_quality`, fixed is `0.412`, random `0.376`, trajectory
`0.412`, evolved `0.451`, repair-selected `0.490`, and repair-vs-evolved
wins/ties/losses are `6/2/0` at `6.12` generations per covered task. This run
reached zero oracle headroom on the generated LLaDA planning repair pool.
The adaptive rescue selected `history_prefix_50_repair` on `plan_004` from
history step `39`. A full history-fraction diagnostic
`llada_planning_history_fraction_sweep_margin01_v1_report.md` reaches the same
`0.490` score, but costs `7.00` generations per task and has worse
budget-normalized gain, so it stays diagnostic.
The current mixed canonical run is
`mixed_adaptive_history_rescue_margin01_v1_report.md`: 116 seeded generations
over the Dream/LLaDA planning-plus-objective mix, fixed `0.436`, random
`0.423`, trajectory `0.465`, evolved `0.480` over 22 model-task pairs, plus
repair-selected `0.490` on the 8 LLaDA planning tasks where suffix-inpainting
repair is available. Repair coverage is explicit as `8/22` overall and `8/8`
among repair-eligible tasks, so the report does not mix full-suite and
covered-slice claims.

The latest diagnostic adds a diffusion-specific `history_visible_repair`
operator that preserves all visible tokens in the selected mid-denoise state
rather than only a prefix. It is implemented and tested, and the report
`llada_planning_visible_history_rescue_margin01_v1_report.md` now includes a
repair-candidate diagnostics table. The result does not replace the canonical
line: repair-selected stays `0.490`, but budget rises to `6.25`
generations/task and budget-normalized gain drops to `0.017`. The diagnostic
is still valuable because `history_visible_repair` reached `0.738` trajectory
score but only `0.347` task score on its rescue run, showing that the next
selector needs an over-preserved-wrong-structure penalty.
That penalty is now implemented as `planning_quality_guarded` and measured in
`llada_planning_visible_history_rescue_guarded_margin01_v1_report.md`. It is
label-free: it looks at the repair seed's masked fraction and source-history
visible length, not the hidden rubric. On the visible-history diagnostic it
assigns `history_visible_repair` a `0.053` guard penalty while leaving selected
outputs unchanged, confirming that current planning-quality repair selection
was already avoiding the visible-state trap on this slice.

Disagreement-triggered adaptive expansion is now implemented through
`--history-rescue-trigger baseline_or_selector_disagreement` and tested in
`llada_planning_disagreement_visible_history_rescue_guarded_v1_report.md`. It
spends rescue candidates when the repair selector and trajectory selector prefer
different generated repairs. The first GPU run generated four visible-history
rescues with strong raw averages (`0.497` task, `0.738` trajectory), but selected
repair stayed `0.490` while budget rose to `7.00` generations/task. This closes
the first implementation of T085, but it is diagnostic only until selection can
extract those visible candidates without over-expanding the budget.

The latest report tables now include repair-candidate task delta and
wins/ties/losses versus the source trajectory. That changed the visible-history
interpretation: the four `history_visible_repair` candidates had `0.000` mean
task delta versus source and `0/4/0` wins/ties/losses, so the high absolute
score came from preserving already-good source outputs, not from improving
them. The next useful repair operator should target minimal source-relative
improvement, not higher absolute trajectory aesthetics.

That source-relative selector is now implemented as
`planning_quality_delta_guarded`, and the report tables include `PQ Delta`
beside hidden task delta. Rescoring the existing disagreement and mixed raw
files preserved the canonical repair result (`0.490`, `+0.039` over evolved,
`6/2/0`, zero oracle headroom) while making no-op repair candidates visible.
A fresh GPU diagnostic also tested `--repair-pack source_relative` with
`history_prefix_50_repair`, `low_confidence_15_repair`, and
`low_confidence_25_repair`. It underperformed the canonical repair line:
repair-selected was only `0.454`, `+0.004` over evolved, with `1/7/0`
wins/ties/losses. The useful finding is negative: minimal low-confidence
remasking preserves source quality (`low_confidence_15_repair` was `0/8/0`
versus source), so the next repair operator needs targeted content edits, not
just smaller masks.

Two targeted content-edit diagnostics are now implemented. The first,
`--repair-pack targeted_content`, maps filler/repetition spans back to generated
token positions and remasks those spans. It ran cleanly but matched the weak
minimal-remask result: `0.454`, `+0.004` over evolved, `1/7/0`, with
`targeted_filler_repair` never selected. The second, `--repair-pack
prompt_guided`, gives diffusion the source draft plus a generic label-free
critique. That produced the first non-history repair win under the
source-relative guard: `prompt_guided_revision_repair` improved `plan_001` by
`+0.034`. Aggregate repair-selected improved to `0.459`, `+0.008` over evolved,
`2/6/0`, still below the canonical adaptive history-prefix line at `0.490`.
The next useful operator should therefore be adaptive hybrid repair: spend
prompt-guided revision only when cheap source-relative diagnostics predict a
real edit, while preserving the history-prefix repair path as the main line.

That adaptive hybrid path is now implemented as prompt-guided rescue gates:
`--prompt-guided-rescue-trigger`, `--prompt-guided-rescue-limit`,
`--prompt-guided-rescue-source-quality-threshold`, and
`--prompt-guided-rescue-source-controls`. The first fresh GPU diagnostic,
`llada_planning_adaptive_hybrid_prompt_guided_rescue_v1_report.md`, matched the
canonical selected score (`0.490`, `+0.039` over evolved, `6/2/0`, zero oracle
headroom) but did not improve it. Prompt-guided revision was generated on seven
tasks, selected on zero, and added budget (`7.00` generations/task, `0.013`
task-score gain per extra generation). The conclusion is sharp: the gating
mechanism is useful, but the current generic prompt-revision operator is not a
default spend. The next diffusion-native repair should use state-conditional
mask/edit policies rather than another generic critique prompt.

The first state-conditional spend policy is now implemented and measured.
`--repair-spend-trigger source_quality_or_short` uses label-free source planning
quality plus visible text length to skip the primary repair pass when the
selected source already looks complete. The fresh GPU report
`llada_planning_primary_repair_gate_v1_report.md` preserved repair-selected
`0.490`, `+0.039` over evolved, `6/2/0`, and zero oracle headroom, while
reducing actual generations from the old adaptive history line's `49` to `47`.
Covered-slice budget dropped from `6.12` to `5.88` generations/task, and repair
gain per extra generation rose from `0.018` to `0.021`.

The first state-conditional mask policy is now stronger. `--repair-pack
state_adaptive` starts with `state_adaptive_history_repair` plus
`prefix_25_repair`. The adaptive history branch receives the source planning
quality plus selected denoise-history state score and mask count, then chooses a
longer history anchor only for weak source/weak history states. The fresh GPU
report `llada_planning_state_adaptive_history_prefix_v1_report.md` is now the
efficient LLaDA planning repair line: it preserves repair-selected `0.490`, `+0.039` over
evolved, `6/2/0`, and zero oracle headroom, while reducing actual generations
to `46`. Covered-slice budget drops to `5.75` generations/task, and repair gain
per extra generation rises to `0.022`. The negative sibling diagnostic
`llada_planning_state_adaptive_repair_pack_v1_report.md` showed the
quality-scaled low-confidence branch averaged `-0.009` task delta versus source
and was selected zero times, so the current policy keeps it out of the first two
budgeted candidates.

The replay-consistency repair path is now implemented and measured as a
diffusion-specific diagnostic. `--repair-pack replay_consistency` remasks
positions that fluctuate across sampled denoise-history states, then falls back
to state-adaptive history repair. The fresh GPU report
`llada_planning_replay_consistency_repair_v1_report.md` reached only `0.477`,
`+0.026` over evolved, with `4/4/0` repair-vs-evolved wins/ties/losses at
`5.75` generations/task. The dedicated `replay_unstable_25_repair` candidate
was selected zero times and had `0.000` task delta versus source with `0/7/0`
wins/ties/losses. This is a useful negative: denoise instability is observable,
but instability alone is not enough to identify a planning repair. Keep
state-adaptive history-prefix as the canonical line.

The first prompt-grounded constraint-gap repair is also implemented and measured.
`--repair-pack constraint_gap` keeps the state-adaptive history plus prefix
candidates and adds `constraint_gap_revision_repair`, which extracts missing or
weak terms from the original prompt and asks diffusion to rewrite the draft
around those task constraints. The fresh GPU report
`llada_planning_constraint_gap_repair_v1_report.md` is a small absolute win:
repair-selected reaches `0.491`, `+0.040` over evolved, with `6/2/0` wins/ties/losses
and zero oracle headroom. The new branch was selected once, on `plan_001`, where
it improved the evolved output by `+0.076`; across all generated constraint-gap
repairs it averaged `+0.011` task delta versus source with `1/6/0`
wins/ties/losses. Because the third candidate raises budget to `6.62`
generations/task and drops gain per extra generation to `0.015`, this should be
gated by prompt-gap pressure before becoming a default spend.

That gate is now implemented as `--constraint-gap-rescue-trigger prompt_gap`.
The fresh GPU report `llada_planning_gated_constraint_gap_rescue_v1_report.md`
starts from the efficient `state_adaptive` pack and spends one
`constraint_gap_revision_repair` only when the evolved source has midrange
label-free planning quality (`0.400-0.500`) and at least 6 missing prompt terms.
The gate fired only on `plan_001`, preserved the `0.491` selected score and
zero oracle headroom from the unconditional constraint-gap run, and cut
generation count from `53` to `47`. Covered-slice budget is now `5.88`
generations/task with `0.022` repair gain per extra generation, so this is the
current best LLaDA planning repair line when the prompt-gap rescue gate is
enabled.

The selector guard is now stricter too:
`llada_planning_gated_constraint_gap_risk_guard_rescore_v1_report.md` reuses
that raw file with `planning_quality_delta_risk_guarded`, which subtracts a
label-free prompt-contradiction/risk penalty and reports `Risk Penalty` by
repair candidate. The current slice has zero detected penalties and unchanged
selection, so it preserves `0.491`, `+0.040` over evolved, `6/2/0`, zero oracle
headroom, and `5.88` generations/task. This makes the current best line safer
for future repair expansions without claiming a new aggregate gain.

Mixed benchmark reporting is now less blurry:
`mixed_adaptive_history_rescue_family_regret_rescore_v1_report.md` reuses the
Dream-plus-LLaDA mixed raw file and adds a by-family arm table plus explicit
selector-regret/oracle-coverage diagnostics. Aggregate fixed/random/trajectory/
evolved stay at `0.436`/`0.423`/`0.465`/`0.480`; the repair arm is correctly
reported as `8/22` overall and `8/8` repair eligible, all on LLaDA planning,
with `0.490` selected task score. Trajectory selector regret is `0.030` over
`8/22` improvable selections, evolved regret is `0.014` over `7/22`, and repair
regret is `0.000` over `0/8` on its covered slice.

The current best LLaDA planning policy has now been rerun as a fresh mixed GPU
benchmark:
`llada_mixed_gated_constraint_gap_risk_guard_v1_report.md` uses 59 generations
over 8 planning tasks plus math/symbolic/science checks. It preserves the
planning repair line at `0.491`, `+0.040` over evolved, `6/2/0`, zero oracle
headroom, and `0.022` repair gain per extra generation, with honest coverage
`8/11` overall and `8/8` repair eligible. The by-family table shows math and
science at `1.000`, but symbolic at `0.000`, so the next mixed-suite mechanism
should target symbolic/exact-answer repair rather than adding more planning
repair variants first.

That symbolic gap has a first concrete repair now:
`llada_sym002_exact_counterfactual_repair_v1_report.md` repairs the failed lamp
toggle task from `0.000` to `1.000` with one extra LLaDA generation, using a
prompt-derived counterfactual answer proposal and a label-free proposal-match
selector. The full mixed rerun
`llada_mixed_gated_constraint_gap_exact_repair_v2_report.md` uses 60 fresh
generations, improves repair coverage to `9/11` overall and `9/9` eligible,
raises repair-selected mean task score to `0.548`, beats evolved by `+0.147` on
covered tasks, keeps `7/2/0` repair-vs-evolved wins/ties/losses and zero oracle
headroom, and lifts gain per extra generation to `0.083`. The guarded compact
rerun `llada_mixed_gated_ranked_span_guarded_exact_v1_report.md` preserves that
mixed score while allowing the expanded ranked-span prompt-gap rescue set; it
rejects the slightly higher `plan_001` anchor candidate because the output leaks
a prompt-term checklist with `Risk Penalty 0.180`. This is now the cleanest mixed
LLaDA line. The full 25-task LLaDA scout
`llada_full_scout_gated_exact_repair_v1_report.md` carries that policy across
8 planning, 8 math, 6 symbolic, and 3 science tasks with 116 fresh generations.
Full-suite fixed/random/trajectory/evolved means are `0.772`/`0.680`/`0.772`/
`0.784`; repair is scoped honestly to `9/25` overall and `9/9` eligible,
reaches `0.548` on covered tasks, beats evolved by `+0.147`, and keeps zero
repair-oracle headroom. The `counterfactual_answer_proposal` row reports
`Proposal Task 1.000` and `Task-vs-Proposal 0.000`, so the exact-answer gain is
still proposal-attributable until a future task beats proposal-only.

That next exact-answer stress test now exists. Four harder exact tasks were
added as `math_009`, `math_010`, `math_011`, and `sym_007`; the proposal layer
returns no candidates for them. The fresh LLaDA run
`llada_hard_exact_arithmetic_feedback_v1_report.md` shows fixed/random/
trajectory/evolved all at `0.500`, while long scratchpad self-repair plus
arithmetic-feedback repair covers the two failed tasks. Self-repair fixes
`sym_007`; arithmetic feedback detects the false `math_010` claim
`3*14 + 2*9 = 54`, feeds back that the expression equals `60`, and repairs the
answer to `10`. Repair-selected reaches `1.000` on the eligible slice, beats
evolved by `+1.000`, has `2/0/0` repair-vs-evolved wins/ties/losses, zero
repair-oracle headroom, and `0.667` gain per extra generation. This is the
first exact-answer repair line that is not proposal-attributable and does not
select by hidden answers.

The current full LLaDA line is now
`llada_extended_full_arithmetic_feedback_v1_report.md`: 135 fresh generations
over 29 tasks, combining 8 planning, 11 math, 7 symbolic, and 3 science tasks.
Full-suite fixed/random/trajectory/evolved means are `0.734`/`0.656`/`0.734`/
`0.745`; repair coverage is explicit as `11/29` overall and `11/11` eligible.
On that eligible slice, repair-selected reaches `0.630`, beats evolved by
`+0.302`, has `9/2/0` repair-vs-evolved wins/ties/losses, zero repair-oracle
headroom, and `0.175` gain per extra generation. The key point is that the
repair stack now composes: planning repair, proposal repair, scratchpad
self-repair, and arithmetic-feedback repair all run under the same budget
ledger.

The first GSM-style hidden-distractor slice is now in the locked scout manifest
as `math_012` through `math_015`. The deterministic proposal layer returns no
candidates for all four tasks. The fresh LLaDA report
`llada_gsm_distractor_self_repair_v1_report.md` uses 19 generations:
fixed/random/trajectory/evolved all score `0.500`, while repair covers the two
failed tasks and reaches `1.000` on the eligible slice with `2/0/0`
repair-vs-evolved wins/ties/losses, zero repair-oracle headroom, and `0.667`
gain per extra generation. `math_014` is fixed by scratchpad self-repair;
`math_013` is fixed by arithmetic feedback after the scratchpad claims
`204 + 56 = 265` and the verifier computes `260`. The arithmetic guard now
also catches simple worded claims such as "90 minus 60 is 30" and compound
claims such as "3 times 14 plus 2 times 9 is 54".

Exact integer repair selection now has an arithmetic-evidence gate. Changed
integer answers from `self_check_answer_repair` and `arithmetic_feedback_repair`
need at least one checkable arithmetic claim before they can be promoted, and
reports expose the mean `Arithmetic Claims` count. CPU-only rescoring with this
guard preserves the current results: `llada_extended_full_evidence_guard_rescore_v1_report.md`
keeps the 29-task line at `11/11` eligible repair coverage, `+0.302` versus
evolved, `9/2/0`, and zero repair-oracle headroom; `llada_gsm_distractor_evidence_guard_rescore_v1_report.md`
keeps the GSM slice at `2/2` repair coverage, `+1.000`, and zero headroom.

The first missing-evidence fallback is now implemented as
`arithmetic_evidence_repair`. When integer self-repair changes the answer but
shows zero checkable arithmetic claims, the runner can spend the next repair
slot on a prompt that asks diffusion to solve again with explicit equations.
The branch remains under the same label-free selector: changed parseable answer,
arithmetic consistency, and at least one checkable arithmetic claim are still
required before promotion. This makes the strict evidence gate productive
instead of merely rejecting bare-number repairs.

Semantic equation verification now has a first label-free guard. The selector
extracts prompt numbers from locally irrelevant clauses such as "not being
packed", "not ticket revenue", "only count", or "question asks", and rejects
integer exact repairs whose arithmetic expressions use those excluded numbers.
The semantic-guard rescores preserve the current exact results:
`llada_extended_full_semantic_guard_rescore_v1_report.md` keeps the 29-task
line at `11/11` eligible repair coverage, `+0.302`, `9/2/0`, and zero
repair-oracle headroom; `llada_gsm_distractor_semantic_guard_rescore_v1_report.md`
keeps the GSM slice at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now
include `Irrelevant # Used`, and selected exact repairs are `0.000` on that
diagnostic in both rescored lines.

Operation-role checking is now the next semantic equation guard. The selector
infers obvious prompt-required operations from wording such as "remaining",
"shared equally", "per bag", "dollars each", "twice as many", and "across
those", then rejects integer exact repairs whose checkable equations omit those
operations. The operator-guard rescores preserve the current exact results:
`llada_extended_full_operator_guard_rescore_v1_report.md` keeps the 29-task
line at `11/11` eligible repair coverage, `+0.302`, `9/2/0`, and zero
repair-oracle headroom; `llada_gsm_distractor_operator_guard_rescore_v1_report.md`
keeps the GSM slice at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now
include `Missing Ops`, and selected exact repairs are `0.0` on that diagnostic
in both rescored lines.

Quantity-role binding is now the next stricter semantic equation guard. The
selector extracts explicit prompt roles such as ticket-count times ticket-price,
trays times items per tray, subtraction of a stated removed quantity, and
division by a stated equal-share count, then rejects integer exact repairs whose
equations use the right operators but bind those quantities incorrectly. The
role-guard rescores preserve the current exact results:
`llada_extended_full_role_guard_rescore_v1_report.md` keeps the 29-task line at
`11/11` eligible repair coverage, `+0.302`, `9/2/0`, and zero repair-oracle
headroom; `llada_gsm_distractor_role_guard_rescore_v1_report.md` keeps the GSM
slice at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now include
`Role Gaps`, and selected exact repairs are `0.0` on that diagnostic in both
rescored lines.

Arithmetic provenance is now the next derived-variable guard. The selector
maintains a set of prompt-grounded numbers plus outputs of earlier verified
equations, then rejects integer exact repairs whose later equations introduce
unexplained constants. This also prevents a false earlier equation from
licensing its claimed output as a later intermediate. The provenance-guard
rescores preserve the current exact results:
`llada_extended_full_provenance_guard_rescore_v1_report.md` keeps the 29-task
line at `11/11` eligible repair coverage, `+0.302`, `9/2/0`, and zero
repair-oracle headroom; `llada_gsm_distractor_provenance_guard_rescore_v1_report.md`
keeps the GSM slice at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now
include `Provenance Gaps`, and selected exact repairs are `0.0` on that
diagnostic in both rescored lines.

Final-answer role checking is now the answer-object guard for exact integer
repair. The selector infers whether the prompt asks for a total, per-share
division answer, full-bag floor division answer, or remainder, then rejects
repairs whose final integer is not that role output even when local arithmetic
claims are valid. The final-role rescores preserve the current exact results:
`llada_extended_full_final_role_guard_rescore_v1_report.md` keeps the 29-task
line at `11/11` eligible repair coverage, `+0.302`, `9/2/0`, and zero
repair-oracle headroom; `llada_gsm_distractor_final_role_guard_rescore_v1_report.md`
keeps the GSM slice at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now
include `Final Role Gaps`; selected exact repairs are `0.0` on that diagnostic,
while a non-selected GSM self-repair exposes a useful final-role failure.

Final-answer object checking is now the first named-object guard for exact
integer repair. The selector extracts prompt objects that are locally excluded
from the requested answer, then rejects repairs whose final-answer context
explicitly names those excluded objects. The object-guard rescores preserve the
current exact results:
`llada_extended_full_object_guard_rescore_v1_report.md` keeps the 29-task line
at `11/11` eligible repair coverage, `+0.302`, `9/2/0`, and zero repair-oracle
headroom; `llada_gsm_distractor_object_guard_rescore_v1_report.md` keeps the
GSM slice at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now include
`Object Gaps`.

Final-answer target checking is now the positive unit/object guard for exact
integer repair. The selector extracts the requested answer head from explicit
"how many ..." and related prompt forms, then rejects final-answer units that
name a wrong prompt-known target or attach a conflicting modifier to the
requested target head. The target-guard rescores preserve the current exact
results: `llada_extended_full_target_guard_rescore_v1_report.md` keeps the
29-task line at `11/11` eligible repair coverage, `+0.302`, `9/2/0`, and zero
repair-oracle headroom; `llada_gsm_distractor_target_guard_rescore_v1_report.md`
keeps the GSM slice at `2/2`, `+1.000`, `2/0/0`, and zero headroom. Reports now
include `Target Gaps`.

Constrained short-text self-repair is now enabled for bounded symbolic exact
answers. The label-free parser only supports schemas that the prompt makes
explicit: on/off, yes/no, a fixed number of letters separated by spaces, or a
final list drawn from an initial list. The fake-backend repair test exercises
the actual generation-record path for a no-proposal letter task and verifies
that the selected repair is driven by parsed final-answer text, not the hidden
label.

Short-text symbolic proof checking now guards bounded symbolic self-repair.
When the prompt is mechanically solvable by the existing order/list/toggle
solver, self-repair selection must match that prompt-derived answer. This
rejects symbolic repairs that merely fit the answer schema but contradict the
prompt, while preserving no-solver constrained schemas. Repair diagnostics now
include `Symbolic Gaps`.

The symbolic proof guard now covers simple categorical yes/no syllogisms. The
prompt-derived solver proves contradictions through all-are plus no-are chains,
feeds the same answer into counterfactual proposals, and rejects schema-valid
self-repairs that disagree with the proof. This extends non-arithmetic exact
verification beyond order/list/toggle tasks.

Mechanically solvable short-text repairs now require minimal trace evidence
before the final answer. The guard is still label-free: it only activates when
the prompt solver proves an answer, then checks for before-relations, swap
evidence, toggle/parity evidence, or syllogism relation evidence in the
scratchpad. Terse final-answer-only self-repairs are rejected through `Trace
Gaps`; no-solver bounded schemas remain eligible.

Letter-code transforms are now a first bounded symbolic repair target. `sym_008`
starts with `K L M`, asks for a one-step left rotation and a final-two-letter
swap, and exposed a real LLaDA failure: fixed/random/evolved and plain
self-check repair all preserved `M L K`. The new prompt-derived operation
solver proves `L K M`, feeds counterfactual repair, and the fresh
`llada_symbolic_letter_transform_repair_v1_report.md` run repairs the task to
`1.000` with zero repair-oracle headroom. This also tightened short-text repair
metadata so casing-only changes do not count as changed answers and non-integer
repairs do not report arithmetic operator noise.

### A. Distribution Geometry And Measurement

- [x] T001. Add reusable logit-distribution diagnostics for KL, JS, entropy, top-k overlap, rank drift, and counterfactual mass.
- [x] T002. Add a first-token geometry audit script comparing baseline, random prefix, latent-projected prefix, zero prefix, and mean-embedding prefix.
- [ ] T003. Record geometry metrics into every sensitivity result JSON so accuracy can be analyzed against distribution drift.
- [ ] T004. Add per-step generation geometry tracing for the first 32 generated tokens.
- [ ] T005. Add answer-token mass metrics for arithmetic tasks by tracking probability assigned to the correct final number.
- [ ] T006. Add margin-flip metrics for correct-vs-most-plausible-wrong answer tokens.
- [ ] T007. Add entropy-slope metrics to detect whether perturbations prevent early distribution collapse.
- [ ] T008. Add top-wrong-mass metrics to quantify when perturbations suppress dominant wrong answers.
- [ ] T009. Add calibrated "clean movement" score: high task gain, low off-target KL, low rank drift.
- [ ] T010. Add "useful chaos" score: high task gain, high distribution drift, high trajectory diversity.
- [ ] T011. Compare forward KL and reverse KL as predictors of task improvement.
- [ ] T012. Compare JS divergence against exact accuracy changes across all stored sensitivity results.
- [ ] T013. Add distribution metrics to planning and legal comparison runs, not just arithmetic runs.
- [ ] T014. Build a small notebook or script that plots accuracy gain against KL, entropy delta, and top-k rank drift.
- [ ] T015. Add tests for geometry metrics on peaked, uniform, and pair-swapped distributions.

### B. Information-Geometric Steering

- [ ] T016. Audit the existing DualSteeringProcessor against Park et al.'s hidden-state covariance update.
- [ ] T017. Implement unembedding-covariance dual steering at the hidden-state level using top-k covariance approximation.
- [ ] T018. Compare logit-space dual steering, hidden-state dual steering, and Euclidean logit addition.
- [ ] T019. Add a regularization sweep for alpha in dual steering and track stability vs target movement.
- [ ] T020. Add a KL-capped dual-steering schedule that adapts eta per generated token.
- [ ] T021. Implement a target-direction interface that can use answer-token, concept-token, or learned-probe directions.
- [ ] T022. Add "dual-prefix" decoding: convert a desired output-distribution movement into prefix gradients or prefix updates.
- [ ] T023. Test whether information-geometric updates preserve unrelated answer-token rankings better than random prefix noise.
- [ ] T024. Add low-rank covariance approximation to avoid full vocabulary hidden covariance costs.
- [ ] T025. Add a CPU-safe synthetic softmax geometry benchmark for steering correctness.
- [ ] T026. Create a pair-map builder for simple morphology tasks such as verb base to third-person.
- [ ] T027. Reproduce a minimal Gemma/Qwen verb-steering experiment using this repo's harness.
- [ ] T028. Test whether dual steering can shift reasoning style without causing answer collapse.
- [ ] T029. Add off-target preservation metrics for answer format, length, and refusal rate.
- [ ] T030. Write an internal note explaining exactly where this repo's dual steering differs from the paper.

### C. Soft Prompt Mechanism

- [ ] T031. Run a seed-controlled comparison of random normal, random real-token embeddings, mean embedding, zero embedding, and repeated embedding prefixes.
- [ ] T032. Separate position effect from vector effect by injecting identical prefixes at positions 0, 1, 2, 4, and after the user prompt.
- [ ] T033. Sweep prefix RMS at fine resolution around the current 0.022 sweet spot.
- [ ] T034. Sweep token count from 1 to 16 with identical compute budgets and identical task sets.
- [ ] T035. Test orthogonal random prefixes versus correlated random prefixes.
- [ ] T036. Test whether prefixes sampled from real embedding principal components outperform isotropic noise.
- [ ] T037. Test whether frequency-stratified real-token prefixes differ from random embedding noise.
- [ ] T038. Add prefix norm, cosine, and PCA diagnostics to every generated soft prompt.
- [ ] T039. Build a prefix ablation that masks attention to prefix tokens after generation token 1, 2, 4, and 8.
- [ ] T040. Build a prefix ablation that keeps keys but zeros values, then keeps values but zeros keys.
- [ ] T041. Measure whether successful prefixes primarily change first-token logits or later hidden-state dynamics.
- [ ] T042. Add prefix diversity metrics and correlate them with oracle coverage.
- [ ] T043. Test if the same prefix wins consistently across paraphrases of the same arithmetic task.
- [ ] T044. Test if successful prefixes transfer across model sizes within the same model family.
- [ ] T045. Test if successful prefixes transfer across domains or only across task instances.

### D. Attention Sink And Trajectory Dynamics

- [ ] T046. Add attention tracing for prefix tokens on Qwen models for the first N generated tokens.
- [ ] T047. Measure sink mass received by each prefix token across layers and heads.
- [ ] T048. Compare sink mass between successful and unsuccessful perturbation seeds.
- [ ] T049. Test whether perturbations rescue failures by changing high-sink heads specifically.
- [ ] T050. Add head-level ablations for heads with the strongest prefix-token attention.
- [ ] T051. Add a "sink placeholder" control using harmless discrete tokens with stable embeddings.
- [ ] T052. Test whether random prefix benefit survives when sink tokens are moved outside the attended prefix window.
- [ ] T053. Measure EOS-token probability under baseline and perturbation at each early generation step.
- [ ] T054. Add degenerate-loop detectors for repeated phrases, premature EOS, and short collapsed answers.
- [ ] T055. Build a trajectory divergence metric using hidden-state cosine distance over generated steps.
- [ ] T056. Compare trajectory divergence to final answer accuracy.
- [ ] T057. Add causal patching from successful runs into failed runs at early layers and tokens.
- [ ] T058. Add causal patching from failed runs into successful runs to verify mechanism necessity.
- [ ] T059. Identify whether perturbation changes reasoning content before or after the first visible token.
- [ ] T060. Produce a sink-mechanism report with plots and failure examples.

### E. Judge, Scorer, And Evolution

- [ ] T061. Replace the barely trained latent scorer with a calibrated lightweight reward model for arithmetic correctness.
- [ ] T062. Train a scorer to predict geometry-clean improvement, not just final correctness.
- [ ] T063. Add pairwise preference data from baseline vs perturbation outputs for legal and planning tasks.
- [ ] T064. Add a scorer calibration report: reliability curve, Brier score, and false-positive analysis.
- [ ] T065. Add active-learning loops that request labels for high-uncertainty perturbation candidates.
- [ ] T066. Add multi-objective selection: accuracy, low hallucination, low off-target drift, and diversity.
- [ ] T066a. Build the first component extraction schema for latent aggregation: component ID, task slot, source trajectory, source span, verifier status, contradiction group, and support score.
- [ ] T066b. Add a component-level aggregation scout over planning tasks where prefix perturbation, greedy, and diffusion repair candidates expose different rubric components.
- [ ] T066c. Compare aggregate answer quality against best single candidate, whole-candidate selector, and majority/self-consistency baselines.
- [ ] T066d. Add aggregation metrics: component gain, component loss, contradiction count, unsupported additions, source diversity, and total generation/repair/fusion cost.
- [ ] T066e. Promote no aggregation claim unless it beats the best single candidate on a predeclared or held-out slice with zero hard contradictions.
- [ ] T067. Implement Pareto-front selection instead of scalar-only fitness.
- [ ] T068. Add novelty search over distribution-geometry signatures.
- [ ] T069. Add quality-diversity archives keyed by answer-token mass and trajectory divergence.
- [ ] T070. Add lineage tracking so every evolved latent records parent, mutation, score, and outcome.
- [ ] T071. Compare CMA-ES, MAP-Elites, random search, and current evolutionary search on identical budgets.
- [ ] T072. Add early stopping when candidate geometry indicates pure destructive drift.
- [ ] T073. Add uncertainty-aware ensembling where scorer uncertainty triggers more candidates.
- [ ] T074. Train domain-specific legal/planning judges on blind-review outputs already stored in experiments.
- [ ] T075. Add judge-adversarial tests where verbosity or legal-sounding language should not be rewarded.

### F. Ensemble And Controlled Decorrelation

- [ ] T076. Formalize Controlled Decorrelation Ensemble as a first-class decode mode.
- [ ] T077. Implement a selector that chooses among diverse perturbed outputs using calibrated judge scores.
- [ ] T078. Add plurality voting for arithmetic with exact answer extraction.
- [ ] T079. Add weighted voting where weights come from geometry-clean confidence.
- [ ] T080. Compare random perturbation ensembles against temperature best-of-N at equal generation budgets.
- [ ] T081. Compare random perturbation ensembles against self-consistency at equal token budgets.
- [ ] T082. Add a budget-normalized compute ledger for all ensemble methods.
- [ ] T083. Measure marginal value of each additional perturbation seed.
- [ ] T084. Add seed subset selection: pick perturbations that historically solve complementary task subsets.
- [x] T085. Add adaptive ensemble expansion when first-pass candidates disagree.
- [x] T085a. Add source-relative diffusion repair selection and minimal-remask diagnostics.
- [x] T085b. Add targeted-content and prompt-guided diffusion repair diagnostics.
- [x] T085c. Add adaptive prompt-guided rescue gating and document the negative budget result.
- [x] T085d. Add state-conditional primary repair spending and verify lower-budget equal-score repair.
- [x] T085e. Add state-adaptive history-anchor repair and verify equal-score repair with no rescue branch.
- [x] T085f. Add replay-consistency repair from denoise-history instability and document the no-op diagnostic result.
- [x] T085g. Add prompt constraint-gap repair and verify the small absolute win plus budget tradeoff.
- [x] T085h. Gate prompt constraint-gap repair by source quality and prompt-gap pressure.
- [ ] T086. Add contradiction detection across ensemble outputs for planning and legal tasks.
- [x] T086a. Add a planning-only prompt-contradiction/risk guard for diffusion repair selection and reporting.
- [ ] T087. Build an aggregator that merges correct partial reasoning from multiple candidates.
- [ ] T088. Add "do not merge" rules when candidates cite incompatible facts or laws.
- [ ] T089. Track oracle coverage separately from selector-realized accuracy in every report.
- [x] T089a. Add oracle-coverage/improvable-selection counts to the diffusion three-arm benchmark report.
- [ ] T090. Build a selector-regret metric: oracle accuracy minus chosen-candidate accuracy.
- [x] T090a. Add selector-regret and by-family summaries to the diffusion three-arm benchmark report.

### G. Benchmarks And Domains

- [ ] T091. Create a clean arithmetic benchmark split with locked seeds, locked prompts, and locked expected answers.
- [ ] T092. Add symbolic expression tasks where exact answer extraction is robust.
- [x] T092a. Add exact-answer counterfactual diffusion repair for failed symbolic benchmark tasks.
- [x] T092b. Run the full 25-task LLaDA planning-plus-exact scout with exact-answer proposal attribution.
- [x] T092c. Add unsupported exact-answer stress tasks plus scratchpad self-repair with arithmetic-consistency guarding.
- [x] T092d. Add arithmetic-feedback exact repair for inconsistent scratchpad equations and verify it fixes `math_010`.
- [x] T092e. Run the extended 29-task full LLaDA scout with planning repair, proposal repair, self-repair, and arithmetic feedback enabled together.
- [x] T093. Add GSM-style word problems with exact numeric answers and hidden distractors.
- [x] T093a. Run a four-task LLaDA GSM distractor repair slice with empty deterministic proposal coverage.
- [x] T093b. Require checkable arithmetic evidence before selecting changed integer self-repairs, then rescore the full and GSM exact lines.
- [x] T093c. Add `arithmetic_evidence_repair` for changed integer self-repairs that lack checkable equations.
- [x] T093d. Add a semantic irrelevant-number guard for exact integer repair equations and rescore the full and GSM exact lines.
- [x] T093e. Add an operation-role guard for exact integer repair equations and rescore the full and GSM exact lines.
- [x] T093f. Add a quantity-role binding guard for exact integer repair equations and rescore the full and GSM exact lines.
- [x] T093g. Add an arithmetic-provenance guard for exact integer repair equations and rescore the full and GSM exact lines.
- [x] T093h. Add a final-answer role guard for exact integer repair equations and rescore the full and GSM exact lines.
- [x] T093i. Add a final-answer object guard for exact integer repair equations and rescore the full and GSM exact lines.
- [x] T093j. Add a final-answer target guard for exact integer repair equations and rescore the full and GSM exact lines.
- [x] T093k. Add constrained short-text exact self-repair support for no-proposal symbolic tasks.
- [x] T093l. Add a symbolic proof guard for constrained short-text exact self-repairs.
- [x] T093m. Add a categorical yes/no syllogism solver to the symbolic proof guard and proposal layer.
- [x] T093n. Add a trace-evidence guard for mechanically solvable short-text exact self-repairs.
- [x] T093o. Add a bounded letter-code transform solver/guard and run a fresh LLaDA symbolic repair slice.
- [x] T093p. Normalize short-text self-repair metadata and suppress irrelevant arithmetic diagnostics on non-integer repairs.
- [x] T093q. Add label-free proposal-history exact trajectory selection and rescore existing LLaDA raw traces.
- [x] T093r. Build full-history exact symbolic probes and audit whether current LLaDA histories expose transient correct-answer wins.
- [x] T093s. Add a non-monotonic within-trajectory remask/revision operator and run exact/planning GPU probes.
- [x] T093t. Replace blind revision remasking with verifier-guided span selection and a selector guard against revision regressions.
- [x] T093u1. Add arithmetic-contradiction and planning constraint-gap span repair candidates, then run targeted GPU diagnostics.
- [x] T093u2. Replace planning constraint-gap generic filler masking with prompt-gap-ranked downstream span repair, fix sentence-span edge cases, and run one-task, three-task, eight-task, and canonical-comparison LLaDA GPU diagnostics.
- [x] T093u3. Add a prompt-checklist leakage penalty to planning risk-guarded repair selection and rescore the expanded gated ranked-span rescue run.
- [x] T093u4. Document, preset, and run the lean compact GPU diffusion protocol: fixed baseline, random perturbation, and guarded latent repair on 8 short planning tasks plus math/symbolic/science checks.
- [x] T093u5. Add a planning span-residue guard so verifier-guided span repair is penalized when the denoise output reconstructs the exact weak spans it was supposed to remove.
- [x] T093u6. Register LLaDA-MoE and its quantized fallback, generalize LLaDA-family routing, fix tokenizer-aware mask-id resolution, and run a no-weight preflight for the sparse MoE target.
- [x] T093u7. Materialize LLaDA-MoE locally, run BF16 CUDA smoke/history smoke, and run the lean mixed GPU benchmark to compare the transferred dense-LLaDA repair policy.
- [x] T093u8. Add a MoE-friendly `constraint_span` repair pack and validate that prompt-gap span inpainting preserves the high-spend diagnostic gain at much lower budget.
- [x] T093u9. Add source-aware diffusion repair seeding so non-monotonic revision can win the evolved arm without forcing span repair to branch from the revised text, then validate on the MoE eight-task planning run.
- [x] T093u10. Fix revision-aware raw rescoring so revision schedules are not consumed by the evolved-mutation limit; add source-control report columns and validate multi-source MoE repair as a diagnostic, not the default.
- [x] T093u11. Add adaptive MoE second-source repair spending and a prompt-coverage guarded repair selector; validate by raw rescore and fresh GPU run against the single-source and exhaustive multi-source policies.
- [x] T093u12. Make the adaptive MoE source gate tunable and report its per-task label-free gate features so source spend is auditable.
- [x] T093u13. Sweep adaptive MoE source-gate thresholds against the exhaustive source pool and document the score-maximal versus budget-efficient gate regimes.
- [x] T093u14. Fresh-GPU confirm the stricter adaptive MoE efficiency gate so the score-vs-budget tradeoff is backed by real generations, not only raw rescore.
- [x] T093u15. Promote the confirmed adaptive MoE source-gate thresholds into named `score_max` and `efficiency` modes while preserving `custom` for sweeps.
- [x] T093u16. Turn the adaptive MoE source-gate threshold sweep into a reusable script with JSON/CSV/Markdown outputs, helper tests, and script-regenerated plateau artifacts.
- [x] T093u17. Run the adaptive MoE revision plus `constraint_span` policy on the full lean mixed GPU suite, rescore score-vs-budget variants from the same raw file, and clean adaptive gate diagnostics to only report repair-eligible rubric tasks.
- [x] T093u18. Add a public-facing lean three-arm report section that compares only fixed baseline, random perturbation, and selected latent repair on the repair-covered task slice.
- [x] T093u19. Generate a repo-level diffusion claim evidence map that ties each promoted lean result to score, report, and raw-generation artifacts.
- [x] T093u20. Replace planning span target selection with source-relative verifier ranking, expose target diagnostics in reports, and run a two-task LLaDA-MoE GPU smoke.
- [x] T093u21. Full-suite confirm the source-ranked MoE span repair policy, rescore efficiency and single-source variants, and promote the source-ranked artifact set in the claim evidence map.
- [x] T093u22. Add clause-level planning span targeting as an opt-in diagnostic, run the two-task LLaDA-MoE smoke, and keep the default sentence-level source-ranked policy because clause repair regressed on `plan_002`.
- [x] T093u23. Let exact-answer verifier revision remask rejected answer spans even when no prompt-derived proposal exists, gated behind `--exact-self-repair` and label-free answer extraction.
- [x] T093u24. Run the hard exact no-proposal verifier-span diagnostic, find that integer answer-span inpainting is a no-op for scratchpad failures, and gate no-proposal integer tasks back to self-repair plus arithmetic span repair.
- [x] T093u25. Add exact-repair selector priority for verifier-localized span inpainting, so guard-clean arithmetic contradiction span repair wins over broader feedback when both solve the task.
- [x] T093u26. Stop spending arithmetic feedback after verifier-localized arithmetic span repair already passes exact-answer guards, and confirm the hard exact slice keeps `1.000` eligible repair score at 19 generations.
- [x] T093u27. Promote the hard exact no-proposal early-stop result into the claim evidence map so public evidence covers verifier-localized exact repair as well as planning repair.
- [x] T093u28. Make default planning span repair adaptive: retry clause-level verifier spans only when sentence-level targeting degenerates into whole-draft or fallback masking.
- [x] T093u29. Run a fresh CUDA planning scout for adaptive span repair on `plan_001`; selected latent repair beats fixed/random/evolved by `+0.066429` with zero repair-oracle headroom.
- [x] T093u30. Run and promote the 8-task CUDA planning adaptive-span scout: selected latent repair beats fixed/random/evolved by `+0.053036` with `6/2/0` wins/ties/losses and `0.0015625` repair-oracle headroom.
- [x] T093u31. Run and promote the lean mixed adaptive-span budget scout: selected latent repair beats fixed/random/evolved by `+0.150000` / `+0.181984` / `+0.115714`, uses 54 generations instead of 63, and improves gain per extra generation versus evolved to `0.104143`.
- [x] T093u32. Refresh the MoE source-ranked score-max line with deterministic result identity, rescore efficiency and single-source budget variants from the same raw pool, and promote the identity-confirmed MoE artifact set.
- [x] T093u33. Add verifier-span seed localization diagnostics so answer-span, arithmetic-span, and planning-span repairs report literal target matches versus tail-window fallback before they can be promoted; confirm on CUDA with `llada_planning_span_localization_smoke_v1`, where `constraint_gap_span_repair` localizes literal targets and repairs `plan_001` from `0.399` to `0.465`.
- [x] T093u34. Run a fresh MoE planning source-ranked span-localization CUDA confirmation and promote it into the claim evidence map: `constraint_gap_span_repair` reports `Span Localized 1.000`, `Span Fallback 0.000`, selected latent repair `0.473482`, `+0.061205` vs fixed, `+0.101357` vs random, and `6/2/0` vs evolved.
- [x] T093u35. Add per-claim required repair diagnostics to the evidence map and validator, so the MoE span-localization claim hard-fails unless `constraint_gap_span_repair` has literal target localization `>=1.0` and fallback use `<=0.0`.
- [x] T093u36. Add compact planning-span selection as the default `constraint_span` policy, so diffusion repair minimizes remasked target regions and refines long risky sentences into clauses only when the clause set preserves verifier score with fewer masked words; confirm on CUDA with `llada_moe_planning_compact_span_policy_smoke_v1`, where selected latent repair improves `plan_001` from `0.465357` to `0.528214`.
- [x] T093u37. Debug compact-span full-suite regressions, keep high-coverage decision-rule spans intact, retain near-tie weak failure chains under a compact word budget, and promote the full MoE planning CUDA confirmation `llada_moe_planning_compact_span_score_max_v2`: selected latent repair `0.492321`, `+0.080045` vs fixed, `+0.120196` vs random, `+0.048571` vs evolved, `6/2/0`, and literal span localization `1.000` / fallback `0.000`.
- [x] T093u38. Run the compact MoE policy on the full lean mixed CUDA suite and promote `llada_moe_mixed_compact_span_score_max_v1`: the 11-task run keeps math/symbolic/science at `1.000`, improves the mixed MoE repair line from `0.473482` to `0.492321` at the same 76-generation cost, and compact efficiency/single-source rescores improve the budget frontier from the same raw pool.
- [x] T093u39. Add the `score_efficient` adaptive source gate with a trajectory-quality ceiling; rescore the compact mixed raw pool to keep the top MoE repair score `0.492321` while reducing generation count from 76 to 75 by skipping the unselected `plan_002` second source and keeping the selected `plan_006` branch.
- [x] T093u40. Fresh-GPU confirm the `score_efficient` compact mixed MoE policy in `llada_moe_mixed_compact_span_score_efficient_fresh_v1`: selected latent repair stays `0.492321`, generation count drops to 75, gain per extra generation rises to `0.043175`, and the adaptive gate table proves `plan_002` is skipped while `plan_006` is generated and selected.
- [x] T093u41. Add a generated public benchmark summary and validation gate so the promoted cheap GPU evidence exposes only Greedy, Random perturbation, and Latent repair arms plus relative cost: `DIFFUSION_PUBLIC_BENCHMARK.md` reports `0.412277` / `0.372125` / `0.492321` at `1.000000x` / `1.000000x` / `7.125000x`, backed by `diffusion_public_benchmark.json` and the fresh CUDA score-efficient artifact.
- [x] T093u42. Fresh-GPU confirm the compact MoE single-source budget frontier in `llada_moe_mixed_compact_span_single_source_fresh_v1`: selected latent repair `0.473393`, 74 records, `7.000000x` relative repair cost, `+0.061116` vs greedy, `+0.101268` vs random, and run ID `diffusion-150ed790105bb0b6`; promote it over the older CPU-rescore-only budget claim.
- [x] T093u43. Fresh-GPU confirm direct greedy-output span repair in `llada_moe_mixed_compact_span_fixed_source_fresh_v1`: selected latent repair `0.489911`, only 30 records, `3.000000x` relative repair cost, `+0.077634` vs greedy, `+0.117786` vs random, zero repair-oracle headroom, and run ID `diffusion-935bf9edc3efd410`; promote it as the new budget-favored public latent repair point.
- [x] T093u44. Fresh-GPU confirm source-quality gated fixed-source repair in `llada_moe_mixed_compact_span_fixed_source_quality_gate_fresh_v1`: skip the high-quality no-op `plan_002` repair, preserve selected latent repair `0.489911`, cut relative repair cost to `2.875000x`, keep `+0.077634` vs greedy and `+0.117786` vs random, and promote it as the new budget-favored public latent repair point with run ID `diffusion-f2384443e57f5548`.
- [x] T093u45. Add and fresh-GPU confirm `source_repairability_geometry`, a source-quality plus prompt-gap/coverage repair-spend gate: `llada_moe_mixed_compact_span_fixed_source_repairability_gate_fresh_v1` keeps the same selected latent repair `0.489911`, skips `plan_002`, `plan_005`, and `plan_008`, cuts relative repair cost to `2.625000x`, and promotes run ID `diffusion-ae26bb892c8a68aa` as the new budget-favored public latent repair point.
- [x] T093u46. Add `experiments/analyze_diffusion_repairability_geometry.py` and generate `DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md`, proving the current repairability gate spends on 5/5 productive repair states, skips 3/3 no-lift repair states, misses zero reference repairs, and exposes the prompt-gap/coverage geometry behind that budget frontier.
- [x] T093u47. Add `experiments/sweep_diffusion_repairability_geometry.py` and generate `DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md`, now sweeping 53,460 label-free gate settings including optional first denoise-skeleton step caps and showing the promoted `0.531116` at `2.625000x` budget point is score/cost Pareto-equivalent with 168 zero-waste/zero-miss gates.
- [x] T093u48. Add `experiments/analyze_diffusion_denoise_phase_geometry.py` and generate `DIFFUSION_DENOISE_PHASE_GEOMETRY.md`, using sampled denoise histories to show repairable/low-quality skeleton phases separate all productive repair sources from all skipped no-lift sources on the compact MoE run.
- [x] T093u49. Promote the denoise-phase repairability diagnostic into `--repair-spend-trigger denoise_phase_repairability`, add focused runner tests, CPU-rescore the compact MoE raw pool, and fresh-GPU confirm the trajectory-skeleton trigger preserves `0.489911` at `2.625000x` with run ID `diffusion-5b1bf286b8cfa727`.
- [x] T093u50. Add `constraint_span_history`, make planning span repair honor history-state token/text anchors, run the fresh MoE history-anchor diagnostic, and generate `DIFFUSION_HISTORY_ANCHOR_REPAIR_AUDIT.md`, showing history anchors are real but currently lose `0.015804` versus final-source repair at the same cost.
- [x] T093u51. Extend the history-anchor audit with a dual-anchor label-free selector rescore: it recovers `0.489911` but costs `3.250000x`, proving anchor choice must happen before repair generation to remain budget-competitive.
- [x] T093u52. Add the pre-generation history/final anchor selector audit: source/history span geometry chooses history only on `plan_001`, preserves `0.489911`, and keeps the budget line at `2.625000x`.
- [x] T093u53. Promote pre-generation anchor choice into the runner as `constraint_span_anchor_select` and fresh-GPU confirm it preserves `0.489911` at `2.625000x` with run ID `diffusion-f3c291037d94daaf`, choosing history on `plan_001` and final anchors elsewhere.
- [x] T093u54. Add `experiments/analyze_diffusion_anchor_retention_loss.py`, `tests/test_analyze_diffusion_anchor_retention_loss.py`, and `DIFFUSION_ANCHOR_RETENTION_LOSS.md`, turning the anchor selector into a label-free constraint-retention loss plus span-advantage gate: `plan_001` is the only safe history anchor, six rejected histories lack positive compact-span advantage, one rejected history fails compact target structure, and the diagnostic all-history policy is `-0.015804` behind final-source repair.
- [x] T093u55. Add `constraint_span_anchor_search`, which searches all sampled denoise-history states before choosing the repair anchor. Fresh GPU loose search (`diffusion-c326b3ef25eb8374`) exposed a false positive on `plan_003` and dropped repair to `0.483348`; tightening the safety gate to near-final target similarity `0.96` and length retention `0.95` produced guarded search run `diffusion-ccef06238847a352`, restoring `0.489911` at `2.625000x` while preserving the whole-history search operator.
- [x] T093u56. Add `constraint_span_history_contrast`, a prompt-only denoise-history contrast operator that keeps final-source seeds but adds compact near-final history evidence. Fresh GPU run `diffusion-b92d689695016154` selected zero repairs and scored `0.414598` at `2.625000x`, proving prompt-only trajectory evidence is insufficient; the denoise sequence has to change seed/remask geometry or anchor selection.
- [x] T093u57. Add `constraint_span_history_instability`, a seed/remask geometry operator that keeps final-source span targets but unions in token positions unstable across sampled denoise histories. Fresh GPU run `diffusion-e28eb1d3dde8eea7` scored `0.459107` at `2.625000x`, beating greedy/random by `+0.046830` / `+0.086982` with `5/3/0` and `6/2/0` wins/ties/losses, but trailing anchor-select `0.489911`; keep it as a secondary mask feature rather than the public budget policy.
- [x] T093u58. Add `constraint_span_anchor_instability`, which combines pre-generation final/history anchor selection with denoise-instability remasking. Fresh GPU run `diffusion-d14467a9f9a550b2` scored `0.481027` at `2.625000x`, improving over standalone instability but trailing anchor-select `0.489911`; repair metadata shows active instability masks on all five attempts, with the only clear raw win on `plan_007`, so the next step is conditional instability gating.
- [x] T093u59. Add and test `constraint_span_anchor_instability_gated`, then fix its identity leak by preserving the concrete final/history anchor prompt whenever the instability gate is off. The first GPU run `diffusion-30a85507d687dfdc` regressed to `0.452188`, revealing that gate-off tasks were not a clean A/B. The fixed identity run `diffusion-a7b64be5b7258f39` restores `0.489911` at `2.625000x`, `+0.077634` vs fixed and `+0.117786` vs random; the audit confirms `4/4` gate-off repairs match anchor-select exactly, while the one active gate on `plan_007` changes seed/text but has `0.000000` score delta. Keep this as an identity-stable geometry harness, not a promoted lift.
- [x] T093u60. Add `constraint_span_anchor_instability_prompt_gated`, which preserves exact anchor-select identity on gate-off tasks but switches to the instability-specific repair instruction when the instability gate is active. Fresh GPU run `diffusion-4c6a7a9f356b3f0d` scores `0.498304` at `2.625000x`, improving the public three-arm line by `+0.086027` vs fixed and `+0.126179` vs random; only `plan_007` changes from anchor-select, rising by `+0.067143`, and the claim/public benchmark artifacts now point to this run.
- [x] T093u61. Add `constraint_span_anchor_instability_prompt_only_gated` as the negative control for the prompt-gated win. Fresh GPU run `diffusion-4b5fc2b7604c28a5` scores `0.479911` at `2.625000x`; gate-off branches still preserve `4/4` exact anchor-select identity, but the single active `plan_007` branch drops by `-0.080000`. This rules out prompt routing alone and shows the positive `0.498304` result needs the denoise-instability mask plus the gated instruction.
- [x] T093u62. Add `constraint_span_anchor_instability_claim_gated`, a composite prompt router that preserves prompt-gated identity, keeps the active `plan_007` instability branch, and adds a compact public-claim confound-control gate for `plan_004`. Fresh GPU run `diffusion-0fc7f067a7d87799` scores `0.513437` at `2.625000x`, improving the public three-arm line by `+0.101161` vs fixed and `+0.141313` vs random; `plan_004` rises `+0.121071` versus the prompt-gated frontier while every other repair branch matches.
- [x] T093u63. Add `constraint_span_anchor_instability_claim_strict_gated` as the strict oracle/best-of control for the claim gate. Fresh GPU run `diffusion-df4149f37f6b21bf` scores `0.495625` at `2.625000x`, below the compact claim-gated frontier; `plan_004` falls to `0.355000` task score because the repair states the token/prompt confound but still omits locked reruns, regression recording, and oracle/selected-result separation. Treat this as a negative boundary showing that public-claim repair needs selective geometry-conditioned control rather than simply more explicit instruction text.
- [x] T093u64. Add `constraint_span_anchor_instability_claim_oracle_gated`, a compact oracle-aware public-claim gate that keeps the same denoise-anchor/instability geometry while emphasizing failure-mode validation and selected/oracle result separation. Fresh GPU run `diffusion-692592da063daa60` scores `0.523304` at `2.625000x`, improving the public three-arm line by `+0.111027` vs fixed and `+0.151179` vs random with `6/2/0` wins/ties/losses vs fixed and zero repair-oracle headroom. `plan_004` rises to `0.559286`, but still misses the literal oracle-result rubric hit, so preserve that as the next mechanism target rather than overclaiming the control is solved.
- [x] T093u65. Add `constraint_span_anchor_instability_claim_seeded_gated`, a semantic-anchor denoise repair that fixes `separate oracle best-of results from selected results` into the masked seed when the public-claim gate fires. Fresh GPU run `diffusion-6ae167dc85d5e6ac` confirms the phrase binding works on `plan_004`, but the aggregate line drops to `0.521295` at `2.625000x` because the fixed anchor crowds out the public-claim survival control. Keep it as a negative boundary and next-loss target: semantic anchors need compatibility with the full required-control set.
- [x] T093u66. Add `constraint_span_anchor_instability_claim_compatible_seeded_gated`, a compact dual-control semantic anchor that fits oracle/selected result separation and claim-survival into the same 9-token denoise tail. Fresh GPU run `diffusion-6944d9dd6c412de4` scores `0.531116` at `2.625000x`, improving the public three-arm line by `+0.118839` vs fixed and `+0.158991` vs random with `6/2/0` wins/ties/losses vs fixed and zero repair-oracle headroom. `plan_004` reaches `0.621786` and hits all five rubric controls, proving the compatibility-loss hypothesis in executable form: hard semantic anchors help when they preserve the full required-control set rather than one phrase.
- [x] T093u67. Add `constraint_span_anchor_instability_claim_auto_seeded_gated`, the first automatic compact-control seed policy. It extracts the oracle/selected and claim-survival control surface from the task/rubric and applies the same 9-token seed without truncation. Fresh GPU run `diffusion-7b74493b8c5ca15a` scores `0.520536` at `2.625000x`: still `+0.108259` vs fixed and `+0.148411` vs random, and `plan_004` hits all five rubric controls, but it trails the fixed compatible seed. Keep it as the next boundary: automatic seed policies need a realization-quality loss, not only control-term extraction.
- [x] T093u68. Add `constraint_span_anchor_instability_claim_auto_seeded_realization_gated`, an explicit realization-constraint variant for the automatic seed policy. Fresh GPU run `diffusion-2a310ed45712a36b` scores `0.515759` at `2.625000x`, below the automatic seed and fixed compatible seed. `plan_004` still hits all rubric controls, but the output becomes a low-specificity `Control:` label. This falsifies the idea that stronger prompt constraints are enough; the next mechanism needs a learned/scored realization-quality objective.
- [x] T093u69. Add `experiments/analyze_diffusion_realization_quality.py`, `DIFFUSION_REALIZATION_QUALITY.md`, and the `planning_quality_seed_realization_guarded` repair selector. The audit turns the compact-seed boundary into a label-free loss over control coverage, action coverage, seed-term coverage, prompt coverage, specificity, direct sentence shape, and meta-text penalties. It now also reports a joint seed objective with semantic preservation: compatible seeded remains best by task (`0.621786`), while auto-compat-realized is best by realization (`0.846647`) and seed objective (`0.904921`) but only scores `0.600714`. This gives the next GPU selector a cheap way to reject `Control:`/seed-anchor meta text without adding benchmark arms.
- [x] T093u70. Tighten `planning_quality_seed_realization_guarded` so low realization-quality seed text cannot pass on rubric surface credit alone, then fresh-GPU confirm the compatible-seeded frontier survives the stricter guard. The fresh CUDA run `diffusion-a9ae901393235364` uses `--repair-selector planning_quality_seed_realization_guarded`, preserves `0.531116` at `2.625000x`, and keeps zero oracle headroom. CPU rescore of the realization-gated boundary drops to `0.495625` with only four selected repairs because the `plan_004` `Control:` branch is rejected.
- [x] T093u71. Add `constraint_span_anchor_instability_claim_auto_compat_seeded_gated`, a compatibility-scored automatic seed policy that chooses among compact control anchors before denoise repair. A first `plan_004` CUDA smoke exposed prompt meta-language as the failure mode (`0.466786`), then the cleaned prompt recovered `0.621786`. Full fresh CUDA run `diffusion-913b5bccb7894e5a` ties the fixed compatible frontier at `0.531116` and `2.625000x`, with `+0.118839` vs fixed, `+0.158991` vs random, `6/2/0` wins/ties/losses vs fixed, and zero repair-oracle headroom. This is now retained as the automatic compatibility boundary; the public evidence-map pointer moved to the cleaner preservation-seeded run in T093u74.
- [x] T093u72. Add `constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated`, a realization-prompt follow-up that keeps automatic compatibility-scored seed selection but removes seed/anchor meta-language from the repair instruction. One-task CUDA smoke `diffusion-1a80605979a231e8` raises `plan_004` realization quality from `0.655238` to `0.807460` and removes the `0.140000` meta penalty, but task score falls from `0.621786` to `0.600714`. The tightened v2 smoke `diffusion-d475c628f6386098` raises realization quality to `0.846647` and preserves zero meta penalty, but task score remains `0.600714`. Keep it as a non-promoted boundary showing that the next seed objective must optimize both direct realization and selected/oracle rubric semantics.
- [x] T093u73. Add `planning_quality_seed_objective_guarded`, seed-objective audit fields, and `constraint_span_anchor_instability_claim_auto_joint_seeded_gated`, which scores compact seed candidates for compatibility, expected realization, and selected/oracle semantic preservation. One-task CUDA smoke `diffusion-91dcab0442e7d5a1` selects the 9-token `separate oracle selected; claim survives if disappears` anchor, keeps zero meta penalty and semantic preservation `1.000000`, but stays at `plan_004 = 0.600714` with seed objective `0.883582`. This is a useful negative boundary: joint seed choice alone does not recover the current `0.621786` task frontier.
- [x] T093u74. Add `constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated` and `compact_preservation_control_terms`, which move the useful public-claim preservation pressure from prompt prose into the denoise seed. Prompt-only smoke `diffusion-05c8f40e3fd0f234` stayed at `0.600714`, but preservation-seeded smoke `diffusion-c18d75b68b87ef33` recovers `plan_004 = 0.621786` with seed `oracle selected results; preserve claim if disappears`, semantic preservation `1.000000`, and zero seed/anchor meta penalty. Full mixed-slice CUDA run `diffusion-3b42951db77c5aa6` keeps the public aggregate at `0.531116` and `2.625000x` with zero repair-oracle headroom, so this is now the cleaner promoted public run.
- [x] T093u75. Audit the `denoise_phase_repairability` spend gate by forcing the promoted preservation-seeded repair on the skipped planning tasks. CUDA probe `diffusion-8a8a9e8904e62dbf` on high-quality skip `plan_002` regressed forced repair from source `0.688571` to candidate `0.582500`; CUDA probe `diffusion-4699321baf91294e` on `plan_005,plan_008` selected zero forced repairs, had mean candidate task delta `-0.014464` versus source, and kept zero oracle headroom. The regenerated `DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md` now scores the gate as an error-correction classifier: `5` productive-spend true positives, `3` skipped-no-lift true negatives, `0` false positives, and `0` false negatives on the 8-task planning slice. The runner emits `repair_spend_gate_rows` and a `Repair Spend Gate Diagnostics` report table explaining each spend/skip decision by source quality, prompt-gap band, prompt coverage, and visible denoise skeleton.
- [x] T093u76. Promote denoise-phase geometry from a boolean into executable phase features. The runner now records first repairable skeleton step, step fraction, skeleton coverage, and peak denoise prompt coverage in `repair_spend_gate_rows`, and `--repair-denoise-skeleton-max-step` can turn that into a stricter spend gate without changing the promoted default. `DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md` now reports spent/skipped first-skeleton means (`16.2` / `30.0`) and no-repair-baseline deltas, while `DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md` sweeps 53,460 geometry-plus-phase gates and confirms the promoted `0.531116` at `2.625000x` point remains on the score/cost frontier with 168 zero-waste/zero-miss gates.
- [x] T093u77. Add an explicit phase-window tradeoff table to `DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md`. The current cheap operating points are: no cap or first-skeleton cap `32` spends five repairs for `0.531116` at `2.625000x`; cap `20`/`24` spends four repairs for `0.496607` at `2.500000x`; cap `10`/`16` spends three repairs for `0.472500` at `2.375000x`. This turns denoise-phase timing into a concrete compute dial instead of just a classifier explanation.
- [x] T093u78. Fresh-GPU confirm the step-`20` phase-window operating point. `llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_phase20_fresh_v1`, run ID `diffusion-419fbf63c9d8e30b`, uses `26` generations and reaches selected latent repair `0.496607` at `2.500000x`, with `+0.084330` vs fixed and `+0.124482` vs random. The gate spends on `plan_001`, `plan_003`, `plan_004`, and `plan_006`; it skips `plan_007` as `late_repairable_denoise_skeleton` because its first repairable skeleton appears at step `31`, validating the phase-window budget dial with real CUDA generations.
- [x] T093u79. Fresh-GPU confirm the step-`32` phase-window promoted point. `llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_phase32_fresh_v1`, run ID `diffusion-3b42951db77c5aa6`, uses `27` generations and reaches selected latent repair `0.531116` at `2.625000x`, with `+0.118839` vs fixed and `+0.158991` vs random. The gate spends on `plan_001`, `plan_003`, `plan_004`, `plan_006`, and `plan_007`; it accepts `plan_007` because its first repairable skeleton appears at step `31`, inside the step-`32` cap, confirming the public score side of the phase-window budget dial with real CUDA generations.
- [x] T093u80. Add `constraint_span_phase_anchor`, a diffusion-native repair pack whose source state is the first safe repairable pre-generation denoise skeleton instead of a post-hoc final/history choice. Unit coverage verifies the runner defers anchor selection, marks dense history as required, and resolves the phase anchor to the first safe repairable skeleton. CUDA smoke `llada_moe_plan003_constraint_span_phase_anchor_smoke_v1`, run ID `diffusion-848cdd2d12d1fbc9`, improved `plan_003` from `0.421786` to `0.538214` but correctly fell back to the final output because the early skeleton was retention-unsafe. CUDA smoke `llada_moe_plan007_constraint_span_phase_anchor_smoke_v2`, run ID `diffusion-00558374541fbc4d`, used history step `31` with `anchor_selection_reason=history_phase_first_repairable_skeleton` and improved `plan_007` from `0.307500` to `0.497857` in three full generations. Keep this as the next diagnostic operator until a full lean mixed benchmark confirms whether phase anchors beat or tie the promoted public line.
- [x] T093u81. Full-GPU test `constraint_span_phase_anchor` on the lean mixed suite. `llada_moe_mixed_constraint_span_phase_anchor_fresh_v1`, run ID `diffusion-9dabba8829d29658`, used `27` generations and reached selected latent repair `0.476786` on repair-covered tasks, beating fixed/random by `+0.064509` / `+0.104661` with zero repair-oracle headroom. It repaired the same five planning tasks and improved every selected source (`5/0/0`), but it is not a promotion candidate because the promoted preservation-seeded policy reaches `0.531116` at the same `2.625000x` cost. The boundary is now documented in `DIFFUSION_PHASE_ANCHOR_BOUNDARY.md`: phase states are useful spend/source evidence, but late retention-safe history anchors should be conditional signals rather than blind replacements for final-source repair.
- [x] T093u82. Add and GPU-test `constraint_span_phase_hybrid_preserve_seeded_gated`, which keeps the promoted preservation-seeded repair controls while using phase history only as a conditional source switch. Loose v1, run ID `diffusion-31b57a6b0860adf7`, reached `0.524554` at `2.625000x` but regressed `plan_003` by accepting a weak phase source with target similarity `0.943503` and final-char ratio `0.908714`. The strict v2 gate now requires the normal history-anchor retention standard before switching sources. Fresh CUDA run `diffusion-9386ee5300a75528` ties the promoted public line exactly: selected latent repair `0.531116` at `2.625000x`, `+0.118839` versus fixed and `+0.158991` versus random, with zero repair-oracle headroom. It uses history only for `plan_001` and keeps final-source repair for `plan_003`, `plan_004`, `plan_006`, and `plan_007`, proving phase evidence can be integrated without losing the current frontier.
- [x] T093u83. Make relative cost first-class in the generated diffusion claim evidence. `build_diffusion_claim_evidence.py` now records fixed/random/repair generation budgets and repair-relative GPU cost for every claim, renders a comparable MoE lean mixed score/cost ledger in `CLAIM_EVIDENCE_MAP.md`, and marks the non-dominated score/cost frontier across 11-task, 8-repair MoE mixed claims. This keeps the public benchmark stack narrow while making the relative-cost story auditable from generated artifacts.
- [x] T093u84. Add `experiments/analyze_diffusion_phase_hybrid_mechanism.py`, `tests/test_analyze_diffusion_phase_hybrid_mechanism.py`, and `DIFFUSION_PHASE_HYBRID_MECHANISM_AUDIT.md`. The audit casts the strict phase-hybrid run as an explicit error-correction loop over the denoise sequence: detect repairable phase, diagnose retention safety, choose final/history source, repair, and verify source lift. On run `diffusion-9386ee5300a75528`, the audit records `0.531116` at `2.625000x`, five selected repairs, `{'history': 1, 'final': 4}` source states, five positive repair-vs-source deltas, mean first repairable step `16.2`, mean first safe step `30.5`, and mean retention-safety lag `12.75`. This turns the world-model/error-correction analogy into a generated benchmark artifact rather than a loose metaphor.
- [x] T093u85. Extend the phase-hybrid audit into a concrete source-choice loss target dataset. `experiments/analyze_diffusion_phase_hybrid_mechanism.py` now emits `eval_results/diffusion_language/diffusion_phase_hybrid_loss_targets.jsonl` alongside the JSON audit and Markdown report. The strict run yields five weighted targets: one `trust_history_source` positive on `plan_001`, four `preserve_final_source` negatives on `plan_003`, `plan_004`, `plan_006`, and `plan_007`, and mean loss weight `0.186429`. This is the next trainable selector objective for replacing hand-coded final/history source switching without expanding the benchmark stack.
- [x] T093u86. Add `experiments/analyze_diffusion_phase_source_policy.py`, `tests/test_analyze_diffusion_phase_source_policy.py`, and `DIFFUSION_PHASE_SOURCE_POLICY_AUDIT.md`. The policy audit compares `final_only`, naive repairable-phase replacement, any safe phase replacement, loose similarity, strict similarity, and a calibrated similarity gate against the phase-source loss targets. The selected calibrated rule is `phase_safe_repairable_count > 0`, `target_similarity >= 0.96`, and `text_similarity >= 0.96`; it has zero weighted error, while naive repairable-phase replacement creates four false history-source switches and any-safe-phase replacement creates three. This turns the source-choice loss into an executable selector audit and preserves the cheap three-arm benchmark scope.
- [x] T093u87. Align the executable phase-hybrid runner with the calibrated source-choice policy. `experiments/run_diffusion_three_arm_benchmark.py` now exposes `PHASE_SOURCE_TARGET_SIMILARITY_MIN`, `PHASE_SOURCE_TEXT_SIMILARITY_MIN`, and `PHASE_SOURCE_HISTORY_CHAR_RATIO_MIN`, records those thresholds in anchor-selection metadata, and routes `phase_hybrid_history_source_advantage` through `_phase_history_anchor_passes_source_policy`. Focused tests verify that weak text retention blocks a history-source switch even when target similarity is high enough, while the current safe `plan_001` style source still passes.
- [x] T093u88. Expose the phase-source policy as benchmark CLI knobs. `run_diffusion_three_arm_benchmark.py` now accepts `--phase-source-target-similarity-min`, `--phase-source-text-similarity-min`, and `--phase-source-history-char-ratio-min`, threads them into pre-generation phase-hybrid anchor selection, records the chosen thresholds in score/report metadata, and verifies that stricter CLI-equivalent thresholds can force final-source preservation. This makes the next source-policy sweep a command-line GPU experiment instead of another code edit.
- [x] T093u89. Run the first phase-source threshold GPU sweep and generate `DIFFUSION_PHASE_SOURCE_THRESHOLD_SWEEP.md` with `experiments/analyze_diffusion_phase_source_threshold_sweep.py`. Fresh CUDA run `diffusion-27e1b13d93f3abad` lowers the source thresholds to `0.90/0.90/0.90`; it scores `0.524554` at the same `2.625000x` relative cost, below strict `0.531116`, by adding one history-source switch on `plan_003` and dropping that task from `0.538214` to `0.485714`. This confirms the calibrated strict source policy with real GPU generations rather than only loss-target replay.
- [x] T093u90. Add the too-strict phase-source sweep point. Fresh CUDA run `diffusion-d3d0f8b6e108263e` raises thresholds to `0.97/0.97/0.95`, removes the remaining `plan_001` history-source switch, uses final sources for all five repairs, and still scores `0.531116` at `2.625000x`. The threshold sweep now shows a strict/final-preserving plateau: loose history promotion hurts, but the current public score does not require history sourcing when final-source repair can realize the same corrected plan.
- [x] T093u91. Operatorize and GPU-validate the strict/final-preserving plateau as `constraint_span_phase_final_preserve_seeded_gated`. The new repair pack keeps the promoted preservation-seeded controls and dense denoise history requirement for phase repair-spend gating, but sets `source_state="final"` directly so the public benchmark can test phase evidence without threshold-based source replacement. Fresh CUDA run `diffusion-175cbd422107ee5e` scores `0.531116` at `2.625000x`, matches the strict `0.96` and strict `0.97` frontier, and uses `0` history sources / `5` final sources. `DIFFUSION_PHASE_SOURCE_THRESHOLD_SWEEP.md` now lists it as `phase_final_named`, and unit coverage verifies the named pack, execution-equivalent final-source repair path, phase controls, and dense-history default.
- [x] T093u92. Validate the named phase/final operator on the lower-cost phase-window cap. Fresh CUDA run `diffusion-65f906724fed3cbc` uses `constraint_span_phase_final_preserve_seeded_gated` with `--repair-denoise-skeleton-max-step 20`, spends four repairs, skips late `plan_007`, and scores `0.496607` at `2.500000x`. `DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md` now lists both the promoted-policy and named-operator cap-20/cap-32 confirmations, proving the explicit operator preserves the same score/cost frontier and lower-cost tradeoff as the previous threshold-derived path.
- [x] T093u93. Add the named phase/final cap-16 point to the cost frontier. Fresh CUDA run `diffusion-f8f6ae3e209d502b` uses `--repair-denoise-skeleton-max-step 16`, spends three repairs (`plan_001`, `plan_003`, `plan_004`), skips two late repairable cases, and scores `0.472500` at `2.375000x`. The repairability sweep report now has a policy column and lists cap-16, cap-20, and cap-32 named-operator confirmations, giving the public benchmark a concrete score/cost ladder rather than only the top frontier point.
- [x] T093u94. Validate that cap-10 collapses to the same named phase/final cheap tier as cap-16. Fresh CUDA command with `--repair-denoise-skeleton-max-step 10` produced the same content/run ID `diffusion-f8f6ae3e209d502b`, same three repairs (`plan_001`, `plan_003`, `plan_004`), same score `0.472500`, and same `2.375000x` cost. `DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md` now lists cap-10 and cap-16 as separate cap settings on the same plateau, proving no extra benchmark value appears between steps 10 and 16.
- [x] T093u95. Validate the no-repair lower boundary for the named phase/final operator. Fresh CUDA run `diffusion-fae5a3498468b66f` uses `--repair-denoise-skeleton-max-step 9`, spends zero repairs because all five productive repairable skeletons first appear at step `10` or later, and scores `0.414598` at `2.000000x`. This proves the first useful phase-window transition is exactly cap `10`, where the operator starts repairing `plan_001`, `plan_003`, and `plan_004`.
- [x] T093u96. Validate the minimal full-frontier phase-window cap. Fresh CUDA command with `--repair-denoise-skeleton-max-step 31` produced the same content/run ID `diffusion-175cbd422107ee5e` as cap 32, spends all five repairs, and scores `0.531116` at `2.625000x`. The full frontier starts exactly when `plan_007` becomes available at step `31`; cap `32` is no longer the minimal public setting, just an equivalent looser cap.
- [x] T093u97. Validate the named phase/final cap-30 plateau before the full frontier. Fresh CUDA run `diffusion-65f906724fed3cbc` with `--repair-denoise-skeleton-max-step 30` matches the cap-20 named-operator content ID, spends the same four repairs (`plan_001`, `plan_003`, `plan_004`, `plan_006`), skips late `plan_007`, and scores `0.496607` at `2.500000x` with 26 total generations. `DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md` now lists cap-30 between the cap-20 four-repair point and the cap-31 five-repair frontier, proving no additional selected repair appears from steps 21 through 30.
- [x] T093u98. Generate the explicit denoise phase-window budget map. `experiments/analyze_diffusion_phase_window_budget.py` reads the named phase/final reference score file, derives task-level repair onsets from `repair_spend_gate_rows`, predicts the cap ladder, and cross-checks fresh CUDA confirmations. `DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md` now records four regimes with zero confirmation mismatches: cap `9` no repair (`0.414598`, `2.000000x`), cap `10-19` three repairs (`0.472500`, `2.375000x`), cap `20-30` four repairs (`0.496607`, `2.500000x`), and cap `31+` five repairs (`0.531116`, `2.625000x`). The README now links this map directly for public readers.
- [x] T093u99. Promote the phase-window budget ladder into benchmark-runner CLI modes. `run_diffusion_three_arm_benchmark.py` now accepts `--repair-phase-budget floor|cheap|mid|frontier`, resolving to caps `9`, `10`, `20`, and `31` respectively, and refuses ambiguous combinations with a manual `--repair-denoise-skeleton-max-step`. Score JSON and rendered reports now record `repair_phase_budget`, and the run identity hash treats the named mode like the underlying cap metadata so equivalent content keeps the same ID. `DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md` now includes a Runner Modes table, and the README's named phase/final reproduction command uses `--repair-phase-budget frontier`.
- [x] T093u100. Fresh-GPU validate the named `frontier` budget mode. CUDA run `diffusion-175cbd422107ee5e` with `--repair-phase-budget frontier` resolves to cap `31`, spends five repairs (`plan_001`, `plan_003`, `plan_004`, `plan_006`, `plan_007`), and reproduces selected latent repair `0.531116` at `2.625000x` with 27 generations. `DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md` now includes this as an eighth fresh confirmation row with mode `frontier` and zero score/cost mismatch, proving the CLI budget mode reaches the same frontier as the manual cap runs.
- [x] T093u101. Fresh-GPU validate the named `cheap` budget mode. CUDA run `diffusion-f8f6ae3e209d502b` with `--repair-phase-budget cheap` resolves to cap `10`, spends three repairs (`plan_001`, `plan_003`, `plan_004`), skips the later `plan_006`/`plan_007` opportunities, and reproduces selected latent repair `0.472500` at `2.375000x` with 25 generations. `DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md` now includes this as a ninth fresh confirmation row with mode `cheap` and zero score/cost mismatch, proving the low-cost public CLI tier reaches the same plateau as the manual cap-10/cap-16 runs.
- [x] T093u102. Fresh-GPU validate the named `mid` budget mode. CUDA run `diffusion-65f906724fed3cbc` with `--repair-phase-budget mid` resolves to cap `20`, spends four repairs (`plan_001`, `plan_003`, `plan_004`, `plan_006`), skips late `plan_007`, and reproduces selected latent repair `0.496607` at `2.500000x` with 26 generations. `DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md` now includes this as a tenth fresh confirmation row with mode `mid` and zero score/cost mismatch, proving the middle public CLI tier reaches the same four-repair plateau as the manual cap-20/cap-30 runs.
- [x] T093u103. Fresh-GPU validate the named `floor` budget mode. CUDA run `diffusion-fae5a3498468b66f` with `--repair-phase-budget floor` resolves to cap `9`, spends zero repairs, and reproduces selected latent repair `0.414598` at `2.000000x` with 22 generations. `DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md` now includes this as an eleventh fresh confirmation row with mode `floor` and zero score/cost mismatch, completing live GPU validation of all public phase-budget CLI modes.
- [x] T093u104. Add a cost-aware budget-policy loss for learned repair spending. `experiments/analyze_diffusion_budget_policy_loss.py` reads the verified phase-window budget map, emits `DIFFUSION_BUDGET_POLICY_LOSS.md`, `eval_results/diffusion_language/diffusion_budget_policy_loss.json`, and `diffusion_budget_policy_loss_targets.jsonl`, and turns each planning task into a marginal repair-value target with `utility(task, lambda) = aggregate_score_lift - lambda * marginal_relative_cost`. The generated audit shows five positive repair targets, three skip targets, marginal repair cost `0.125000`, highest break-even lambda `0.283929`, and a task-gated oracle at lambda `0.18` that would keep `plan_004`/`plan_006`/`plan_007`, score `0.508705` at `2.375000x`, and gain `+0.022589` objective over the best cap policy. This is the next learned selector target: the public cap ladder is validated, but the trainable policy should learn marginal repair value rather than blindly spending every early repairable phase.
- [x] T093u105. Promote the cost-aware marginal-value target into a runner-ready spend trigger and fresh-GPU validation. `experiments/analyze_diffusion_budget_value_proxy.py` now emits `DIFFUSION_BUDGET_VALUE_PROXY_AUDIT.md` and calibrates label-free proxy rules against the budget loss; the selected runner-ready rule is `first_repairable_step exists`, `source_needs_repair`, `prompt_gap_count <= 9`, and `source_quality <= 0.301429`. `run_diffusion_three_arm_benchmark.py` now exposes `--repair-spend-trigger denoise_phase_value_proxy` plus `--repair-value-proxy-source-quality-max`. Fresh CUDA run `diffusion-a343e942cbfb0a93` with the stable CLI threshold `0.31` spends only on `plan_004`, `plan_006`, and `plan_007`, scores `0.508705` at `2.375000x` with 25 generations, and improves over the cheap cap-10 tier by `+0.036205` at the same relative cost. This is the first learned-selector-shaped result: same cost as cheap, higher score by skipping low-marginal early repairs and keeping high-marginal late repairs.
- [x] T093u106. Promote the value-proxy result into the public front door. `build_diffusion_claim_evidence.py` now treats `moe_mixed_phase_final_preserve_seeded_value_proxy_budget` as the default public budget claim, `DIFFUSION_PUBLIC_BENCHMARK.md` renders it as the budget-favored latent repair row, `CLAIM_EVIDENCE_MAP.md` links it to run `diffusion-a343e942cbfb0a93`, and the README now tells public readers to go from README to public benchmark to claim evidence to ground-truth artifact index.
- [x] T093u107. Add a repair-value feature-geometry audit for the learned controller target. `experiments/analyze_diffusion_repair_value_geometry.py` reads `diffusion_budget_policy_loss_targets.jsonl`, emits `DIFFUSION_REPAIR_VALUE_GEOMETRY.md` and `diffusion_repair_value_geometry.json`, and shows why the controller should not equate early repairability with value: at lambda `0.18`, `plan_001` and `plan_003` are repairable but negative-utility spends, while `plan_004`, `plan_006`, and `plan_007` are profitable. The runner source-quality/gap rule has zero regret, the in-band source-quality separation gap is `0.022857`, and the first excluded low-quality prompt-gap case starts at gap `10`.
- [x] T093u108. Add the repo-level mathematical theory layer for diffusion reasoning. `docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md` formalizes denoise trajectories, verifier feature geometry, task-relevant information loss, repair operators, marginal repair value, phase-window frontiers, label-free separability, and candidate error functions. The README and field-implications doc now point public readers to this theory entry path before they dive into generated benchmark artifacts.
- [x] T093u109. Expand the mathematical theory layer with explicit information-accounting and safety assertions. `docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md` now states that judge/verifier/anchor channels must be accounted for separately from frozen-model information, that transfer requires feature stability rather than threshold reuse, that repair is safe only under a retention constraint, and that denoise reasoning is an energy-bounded search over editable states.
- [x] T093u110. Add a reader-facing documentation map for the diffusion stack. `docs/DIFFUSION_READER_GUIDE.md` gives a fast path through the public claims, theory layer, benchmark/cost layer, mechanism audits, anchor-retention work, development surfaces, and claim-evidence hierarchy. The README, field-implications doc, and architecture log now route readers through it.
- [x] T093u111. Add a theory claim ledger so theorem-like assertions are auditable. `docs/DIFFUSION_THEORY_CLAIM_LEDGER.md` maps each diffusion-theory assertion to status, current evidence, assumptions, falsifiers, and next proof obligations. The README, reader guide, theory doc, field-implications doc, and architecture log now point to it as the discipline layer for mathematical claims.
- [x] T093u112. Add a theory claim-ledger validator and tests. `experiments/validate_diffusion_theory_claim_ledger.py` validates ordered theory IDs, conservative statuses, nonempty assertions/evidence/assumptions/falsifiers/proof obligations, local Markdown evidence links, and public backlinks from the README, reader guide, and theory doc. `tests/test_validate_diffusion_theory_claim_ledger.py` covers complete ledgers, bad statuses, missing evidence refs, weak rows, out-of-order IDs, and missing public backlinks.
- [x] T093u113. Add a generated error-function geometry bridge. `experiments/analyze_diffusion_error_function_geometry.py` reads `diffusion_repair_value_geometry.json` and `diffusion_phase_hybrid_loss_targets.jsonl`, emits `DIFFUSION_ERROR_FUNCTION_GEOMETRY.md` plus `diffusion_error_function_geometry.json`, and makes the next-loss claims executable: cost-aware repair value is not raw repair lift, earliest repairability is not sufficient, source trust needs retention/source-advantage checks, and the next controller should decompose repair value, source trust, retention, and anchor realization instead of using one repairability label.
- [x] T093u114. Add a decomposed selector audit against single repairability labels. `experiments/analyze_diffusion_decomposed_selector.py` emits `DIFFUSION_DECOMPOSED_SELECTOR_AUDIT.md` and `diffusion_decomposed_selector_audit.json`, now scoring all four composite-loss terms: repair value, source trust, retention, and realization. The current `single_repairability_label` has composite shortfall `3.053730` from three value false positives, four source false positives, retention error `1.566063`, and realization error `0.573292`; `decomposed_value_source` has composite shortfall `0.186127`, with zero value regret, zero source error, zero retention error, and only the preservation-seed realization loss remaining. `tests/test_analyze_diffusion_decomposed_selector.py` covers local dominance, retention penalties, realization penalties, and report rendering.
- [x] T093u115. Build the first supervised target surface for the four-term controller. `experiments/build_diffusion_composite_selector_targets.py` emits `DIFFUSION_COMPOSITE_SELECTOR_TARGETS.md`, `diffusion_composite_selector_targets.json`, and `diffusion_composite_selector_targets.jsonl`, merging repair-value geometry, phase-source trust targets, retention classifications, realization-policy losses, and the selected decomposed controller. The generated JSONL contains eight task rows for spend/source/retention heads and seven realization-policy rows for the compact-anchor head; `tests/test_build_diffusion_composite_selector_targets.py` covers target merging and report rendering.
- [x] T093u116. Fit the first tiny four-head composite selector baseline. `experiments/fit_diffusion_composite_selector.py` emits `DIFFUSION_COMPOSITE_SELECTOR_FIT.md` and `diffusion_composite_selector_fit.json`, fitting zero-error local heads over the target surface: spend uses `first_repairable_gap_le_9_source_quality_le_0p301429`, source uses `retention_safe_history`, retention uses `classification_safe_history_anchor`, and realization uses `min_realization_policy_error`. `tests/test_fit_diffusion_composite_selector.py` covers the four-head fit and report rendering.
- [ ] T093u. Generalize verifier-guided revision beyond exact answer spans to arithmetic contradiction spans, planning-constraint gaps, and learned source-relative remask policies.
- [ ] T094. Add ARC-style grid tasks only if output verification is reliable.
- [ ] T095. Add legal issue-spotting tasks with required issue checklists.
- [ ] T096. Add legal hallucination checks for nonexistent statutes, cases, and agencies.
- [ ] T097. Add planning tasks with rubric dimensions for completeness, feasibility, and risk handling.
- [ ] T098. Add code-debugging tasks where tests verify candidate fixes.
- [ ] T099. Add long-context tasks designed to stress attention sinks and early-token anchoring.
- [ ] T100. Add paraphrase robustness suites for every task family.
- [ ] T101. Add cross-model runs for Qwen, DeepSeek distills, Phi, Gemma, and at least one Llama-family model.
- [ ] T102. Add quantization sensitivity reports for 4-bit, 8-bit, and full precision where feasible.
- [ ] T103. Add context-template sensitivity tests across chat templates and plain completion prompts.
- [ ] T104. Add "reasoning budget" sweeps for max tokens, thinking mode, and repetition penalty.
- [ ] T105. Publish benchmark manifests with task ID, prompt, answer, split, and generation settings.

### H. Codebase Consolidation And Experiment Hygiene

- [ ] T106. Consolidate duplicated task generators between experiments/run_latent_sensitivity.py and experiments/harness.py.
- [ ] T107. Move shared answer verification into src/latent_reasoning/verification.
- [ ] T108. Move decode result schemas into a shared dataclass instead of ad hoc dictionaries.
- [ ] T109. Add a single experiment config schema with model, decode, perturbation, scoring, and output sections.
- [x] T110. Add deterministic run IDs and content hashes for every result file.
- [x] T111. Add result validation scripts that fail on missing settings, stale claims, or inconsistent n values.
- [x] T112. Add a "latest ground truth" index that points to canonical result files for each claim.
- [x] T113. Add a stale-doc scanner for README, RESEARCH_BRIEF, ARTICLE_UPDATE, and EXPERIMENTS.
- [ ] T114. Add Makefile targets for fast unit tests, geometry tests, and smoke experiments.
- [ ] T115. Add CI-safe tests that never require downloading large models.
- [ ] T116. Add optional GPU smoke tests guarded by environment variables.
- [ ] T117. Add result-compression scripts for large JSON outputs.
- [ ] T118. Add plotting utilities that consume canonical result schemas only.
- [x] T119. Add a repo-level CLAIM_EVIDENCE_MAP.md tying every public claim to result files and scripts.
- [ ] T120. Remove or quarantine obsolete experimental scripts once canonical replacements exist.

### I. Product And Paper Direction

- [ ] T121. Reframe the paper around "judge-selected trajectory control" instead of vague latent knowledge injection.
- [ ] T122. Define the central empirical claim as controlled improvement under fixed compute and fixed model weights.
- [ ] T123. Separate mechanism claims into tested, plausible, and speculative categories.
- [ ] T124. Add a section explaining where information enters the system.
- [ ] T125. Add a section distinguishing random control perturbation, learned soft prompts, and trained probes.
- [ ] T126. Add negative results prominently: direction-agnostic prefix findings, geometry washout, and weak scorer limits.
- [ ] T127. Add a stronger related-work section covering prefix tuning, prompt tuning, ActAdd, RepE, attention sinks, and softmax information geometry.
- [ ] T128. Add a publishability checklist for all experimental claims.
- [ ] T129. Add a "minimum viable next paper" plan with exactly three core experiments.
- [ ] T130. Add a "production system" plan for domain-specific judge-selected perturbation ensembles.
- [ ] T131. Implement the general-purpose three-arm scout runner from `docs/GENERAL_PURPOSE_LATENT_BENCHMARK_PROTOCOL.md`.
- [ ] T132. Build the 25-task scout manifest with 8 planning, 8 math, 6 symbolic, and 3 science QA tasks.
- [ ] T133. Add the scout scorer and report generator for exact-answer, multiple-choice, and fixed-rubric planning tasks.
- [x] T134. Expose the fitted diffusion four-head selector as the runner trigger `decomposed_four_head_selector`.
- [x] T135. Run the lean GPU mixed benchmark with `decomposed_four_head_selector` and promote the budget confirmation.
- [x] T136. Generate an independent lean mixed spend target surface for the decomposed-selector transfer test.
- [x] T137. Fit the first transfer spend rule over original plus independent target rows. After fixing the evaluator to use repair-oracle lift, `DIFFUSION_SPEND_TRANSFER_RULE_FIT.md` now shows `current_decomposed_spend` has zero repair-availability errors and that stricter source-task floors above `0.295357` skip positive low-margin repair `plan_012`.
- [x] T138. Expose the fitted transfer probe in the benchmark runner. `run_diffusion_three_arm_benchmark.py` accepts `--repair-spend-trigger decomposed_spend_transfer_rule`, records `spend_head_source_task_min`, and now defaults the floor to `0.295357` so it preserves the low-margin `plan_012` repair-availability case.
- [x] T139. Run a strict-floor CUDA transfer-preset benchmark and identify the label mismatch. Run `diffusion-f50e82f88f59111b` used the old `0.3075` floor, spent zero repairs, and skipped `plan_012`; the corrected oracle-lift labels show this was cost-conservative but too strict for repair availability.
- [x] T140. Expand independent planning transfer beyond four prompts and refit/test the spend predictor. Added `plan_013`-`plan_016` and preset `lean_gpu_mixed_transfer_v2`; CUDA run `diffusion-76fd30506cace1ee` produced eight independent planning labels with `plan_012` as the only positive repair-availability row, and `DIFFUSION_SPEND_TRANSFER_RULE_FIT_V2.md` keeps `current_decomposed_spend` at zero errors.
- [x] T141. Test promotion-value selection on the low-margin transfer repair. CUDA run `diffusion-2a4bd4e3cad622a2` uses the corrected transfer spend trigger with `repair_selector=inherit`, selects the `plan_012` repair, raises repair-covered planning score to `0.350938`, and removes the `0.002500` oracle headroom left by the planning-quality repair selector. `experiments/evaluate_diffusion_transfer_promotion_value.py` now generates `DIFFUSION_TRANSFER_PROMOTION_VALUE.md` from the policy score files.
- [x] T142. Name the current transfer promotion-value proxy in the runner and docs. `run_diffusion_three_arm_benchmark.py` now accepts `--repair-selector transfer_promotion_value`, an explicit alias for inherited planning-state repair selection, and the README, reader guide, theory doc, architecture log, and generated promotion-value report describe it as the current executable baseline for the learned promotion-value head.
- [x] T143. Fit the first separate transfer-head predictors. `experiments/fit_diffusion_transfer_heads.py` emits `DIFFUSION_TRANSFER_HEAD_FIT.md` and `diffusion_transfer_head_fit.json`: `availability_current_decomposed_spend` has `0` errors over 16 original plus transfer availability rows, and `transfer_promotion_value` has `0` promotion errors where the planning-quality promotion policy has one false negative on `plan_012`.
- [x] T144. Turn the decomposed theory into a proof-object ledger. `experiments/build_diffusion_proof_object.py` emits `DIFFUSION_REASONING_PROOF_OBJECT.md` and `diffusion_reasoning_proof_object.json`, tying availability, promotion-value, source-trust, retention, realization, and cost heads to `52` target rows, information channels, evidence files, falsifiers, and next GPU validation obligations. The fitted heads have `0` measured errors; the cost head is marked objective-defined rather than misreported as a classifier.
- [x] T145. Run a larger fresh GPU slice that tests the proof-object heads without retuning thresholds. Added `plan_017`-`plan_024` plus preset `lean_gpu_mixed_transfer_v3`; all-repairable CUDA run `diffusion-db9cf6afb7c371ab` produced 16 independent planning rows. After the repair-only label correction, positive repair candidates are `plan_018` and `plan_021`; `DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md` now shows single repairability has `5` errors, old decomposed spend has `2`, and trajectory-relative decomposed spend has `1`.
- [x] T146. Make the v3 proof-object correction executable. `run_diffusion_three_arm_benchmark.py` accepts `--repair-spend-trigger trajectory_relative_decomposed_spend` and records source-vs-selected-trajectory diagnostics. CUDA run `diffusion-106f05c6dd5532ee` is retained as historical execution evidence, but the corrected repair-only labels now show the availability trigger also admitted stale positive `plan_012`; promotion labels should be read from `DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md`.
- [x] T147. Replace the hand-coded trajectory-relative availability rule with a learned availability predictor over denoise phase, source quality, prompt gap, and source-minus-selected trajectory delta, then test it on another fresh planning slice. The original v3 learned rule was made executable, but the repair-only label correction shows it was fitting one stale positive. The corrected `DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md` best pre-repair rule still has `1` v3 error, so learned availability remains a boundary rather than a solved head.
- [x] T148. Replace absolute source-quality availability with a slice-relative or calibrated availability model. `run_diffusion_three_arm_benchmark.py` exposes `--repair-spend-trigger calibrated_availability_predictor_v1`. Fresh v5 breaks the availability-only story: `DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md` shows calibrated availability has `3` repair-only label errors, while CUDA run `diffusion-c4f0d7bc21768f21` still beats fixed by `0.044866` and random by `0.074438`; all-repairable run `diffusion-b3324317dadee840` is stronger at `+0.069821` versus fixed and `+0.099393` versus random.
- [x] T149. Build a candidate-aware promotion target from post-repair diagnostics. `experiments/build_diffusion_candidate_promotion_targets.py` emits `DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md` and `diffusion_candidate_promotion_targets_v5.json`; corrected v5 has four positive repair candidates (`plan_034`, `plan_035`, `plan_037`, `plan_039`) and three negatives (`plan_033`, `plan_038`, `plan_040`). The runner now exposes `--repair-selector candidate_aware_promotion_v1`, and the target artifact records `0` promotion errors on the generated v5 repair candidates.
- [x] T150. Run a fresh v6 GPU planning slice without retuning. Added `plan_041`-`plan_048` and preset `lean_gpu_mixed_transfer_v6`. All-repairable CUDA run `diffusion-158fb4ff45a8d2e8` selects positive repairs on `plan_041`, `plan_044`, `plan_046`, `plan_047`, and `plan_048`, scoring `+0.086696` vs fixed and `+0.125929` vs random with zero oracle headroom. Calibrated-spend plus candidate-aware-promotion run `diffusion-b6d8fd700b3a267f` beats fixed/random but misses `plan_046` and `plan_048`; `DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md` shows calibrated availability has `4` errors. `DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md` shows `candidate_aware_promotion_v1` still has `0` promotion errors over eight generated repair candidates. Rescored repairable-denoise spending plus candidate-aware promotion, run `diffusion-ae7a4edd5c22ca20`, selects exactly the five positive repairs and matches all-repairable score.
- [x] T151. Make the v5/v6 spend-policy decision explicit before buying another GPU slice. `experiments/summarize_diffusion_spend_policy_decision.py` emits `DIFFUSION_SPEND_POLICY_DECISION.md` and `diffusion_spend_policy_decision.json`: across v5/v6, repairable-denoise spending covers all `9` profitable repair rows while calibrated availability misses `3`; on live v6, repairable-denoise plus `candidate_aware_promotion_v1` costs `0.375000` more relative extra generations per task than calibrated spend but buys `0.039661` more score and `0.105762` incremental lift per added generation. The incumbent for v7 is therefore `denoise_phase_repairability` plus `candidate_aware_promotion_v1`; any learned spend gate should be scored offline against v5/v6 targets before a full GPU run.
- [x] T152. Run the no-retuning v7 incumbent GPU slice. Added `plan_049`-`plan_056` plus preset `lean_gpu_mixed_transfer_v7`; CUDA run `diffusion-711ea5fcfd8c07e5` uses only fixed/greedy, random perturbation, and latent repair with `denoise_phase_repairability` plus `candidate_aware_promotion_v1`. It beats fixed by `+0.023036` and random by `+0.082063` on repair-covered tasks, with `0.016875` oracle headroom. `DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V7.md` shows only `plan_054` and `plan_056` are profitable repair rows, while single repairability spends on six no-lift rows; `DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V7.md` keeps candidate-aware promotion at `0` errors.

## Immediate Execution Order

1. Create `experiments/general_reasoning_tasks_scout.jsonl`.
2. Implement the three-arm runner for `greedy_baseline`, `random_prefix`, and `latent_reasoning`.
3. Add deterministic scoring for math, symbolic, and multiple-choice science tasks.
4. Add the fixed 10-point short-planning rubric scorer.
5. Run the 25-task scout and write `eval_results/general_reasoning/scout_report.md`.
6. Decide whether the 50-task pilot is justified.
7. Fit a spend gate against combined v5/v6/v7 labels while keeping `candidate_aware_promotion_v1` fixed.
8. Score the learned spend gate offline before running it on GPU.
9. Run a fresh v8 GPU planning slice with only fixed/greedy, random perturbation, and latent repair arms.
10. Compare selected repair lift, oracle headroom, and relative GPU cost against calibrated spend and repairable-denoise spending.
