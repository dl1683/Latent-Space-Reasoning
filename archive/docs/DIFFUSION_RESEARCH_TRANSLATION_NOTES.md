# Diffusion Research Translation Notes

Last updated: 2026-05-21.

This file records how current diffusion/world-model research and the local
research folders should influence this repo's executable reasoning system.

## External Signals

- Dream 7B frames language diffusion as iterative parallel sequence refinement,
  with arbitrary-order generation, infilling, planning flexibility, and a
  quality-speed trade-off. That supports treating denoise states and remasked
  spans as first-class reasoning objects rather than only final text.
  Source: https://arxiv.org/abs/2508.15487
- The official LLaDA line is still the main local masked-diffusion substrate.
  The LLaDA repo now points to LLaDA-MoE-7B-A1B-Instruct, which keeps 7B total
  capacity while activating roughly 1B to 1.4B parameters at inference and
  claims stronger diffusion-LM efficiency. It is now registered, materialized
  locally, and validated through BF16 CUDA planning benchmarks as
  `llada-moe-7b-a1b-instruct-hf`.
  Source: https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct,
  https://github.com/ML-GSAI/LLaDA, and https://arxiv.org/abs/2509.24389
- LeWorldModel moves the world-model thread toward compact latent prediction:
  two-loss JEPA training, low parameter count, faster planning, and surprise
  evaluation for physically implausible events. The direct translation for text
  diffusion is not "make a video world model"; it is to add surprise/verifier
  signals over intermediate text states and repair trajectories.
  Source: https://arxiv.org/abs/2603.19312

## Local Research Signals

- `_meta/projects/latent-space-reasoning.md` says the old random-prefix story
  must be down-scoped: the scorer/judge is carrying the useful signal, and the
  threat is architecture-specific attention-sink behavior. Diffusion should
  therefore be tested as a new substrate, not claimed as continuity with the
  old AR effect.
- `_meta/insights/duplication.md` and `_meta/experiments/proposed.md` point to
  `judge_kit` as the missing shared infrastructure. In this repo, that means
  every repair gain needs a verifier-facing diagnostic: risk penalty, span
  residue, arithmetic consistency, proof trace, proposal attribution, and
  selector regret.
- `Market Reports/Open Exploration/Biological AI/README.md` and
  `Biological AI/neuroscience_meets_deep_learning.md` emphasize feedback,
  prediction error, repair, selection pressure, and energy constraints. For
  diffusion reasoning, that maps to budget-normalized repair gain, verifier
  surprise, non-monotonic remasking, and stopping rules.

## Translation Into Operators

1. Verifier-guided non-monotonic remasking is the main path. Passive history
   selection is only a diagnostic when the sampled history is monotonic fill.
2. Every remask operator should expose a verifier residue score: did the model
   remove the exact weak span, false equation, unsupported answer, or checklist
   artifact it was asked to repair?
3. Budget-normalized gain matters as much as raw task score. The local hardware
   constraint should favor compact 8-planning plus 3-check sweeps before full
   scouts.
4. Exact-answer gains must be separated into proposal-attributable, verifier
   inpainting, self-repair, and arithmetic/proof-feedback buckets.
5. LLaDA-MoE-7B-A1B-Instruct should use the same LLaDA-family schedules,
   non-monotonic revision hooks, and repair gates as dense LLaDA, but repair
   source selection must be explicit because revision can be a good evolved arm
   and a bad seed for the next repair branch.

## Implemented Translation

- `--task-preset lean_gpu_mixed` makes the compact GPU suite executable without
  hand-copying task IDs.
- `planning_quality_delta_risk_guarded` now includes prompt-checklist leakage
  and planning span-residue penalties.
- `Span Residue` appears in repair-candidate diagnostics. The negative
  prompt-copy rescore
  `eval_results/diffusion_language/llada_planning_constraint_gap_ranked_span_v2_span_residue_guard_rescore_v1_report.md`
  assigns `0.180` span residue to the failed `constraint_gap_span_repair` that
  reconstructed both verifier-targeted weak sentences.
- The compact mixed report
  `eval_results/diffusion_language/llada_mixed_gated_ranked_span_guarded_exact_v1_report.md`
  preserves the current headline result while exposing the new diagnostic.
- LLaDA-MoE support is registered in the candidate table and the runner now
  routes every `LLaDA*` family through the LLaDA masked-denoise schedule,
  revision, and repair surface. Cheap preflight succeeded for
  `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` and downloaded only `README.md`,
  config, tokenizer, and custom code files under
  `external/diffusion_preflight/LLaDA-MoE-7B-A1B-Instruct`.
- Full LLaDA-MoE materialization and CUDA validation now pass. The history smoke
  exposes a 32-step monotonic denoise sequence; the compact mixed benchmark
  completed 60 full generations and shows that the sparse model is usable, but
  the dense-LLaDA state-adaptive planning repair policy transfers weakly
  (`0.446` planning repair-selected versus dense LLaDA's `0.491` line).
- A MoE-specific lesson is now implemented: the useful branch is prompt-gap span
  inpainting, not full-draft revision or inherited history repair. The new
  `constraint_span` repair pack spends only `constraint_gap_span_repair` and
  reaches `0.472` planning repair-selected at 5 generations/task, with `6/2/0`
  repair-vs-evolved wins/ties/losses.
- Source-aware repair seeding is now implemented for the revision case.
  `--repair-source-policy non_revision_evolved` lets revision schedules remain
  eligible for the evolved arm while seeding span repair from the best
  non-revision source. On the MoE eight-task planning run with revision enabled,
  this restores repair-selected `0.472`, beats the stronger revision-aware
  evolved arm by `+0.028`, and records the non-monotonic substrate directly:
  `256` committed remasks and `68` remask-mediated rewrites.
- Source diversity is now measurable but not free. `evolved_and_trajectory`
  repair seeding found additional useful span repairs, but the aggregate MoE
  gain was only `0.473` at 61 generations and lower budget-normalized return.
  This supports the next research translation: learn or verify source choice
  before spending extra branches, rather than treating more source seeds as a
  default form of reasoning.
- The first hand-coded source-choice verifier is now implemented. Adaptive
  source repair adds the trajectory source only for distinct low-confidence
  outputs with large prompt gaps and acceptable planning quality, then uses a
  prompt-coverage guarded selector that withholds keyword credit from weak
  repairs. On the MoE revision raw rescore and matching fresh GPU run, this
  reaches `0.474` repair-selected at 58 generations, `+0.030` over evolved and
  zero oracle headroom. That is the strongest MoE revision-enabled result so far;
  the source-gate features are now tunable and reported per task. The next
  raw threshold sweep shows the score-vs-budget tradeoff explicitly: current
  settings preserve the best score by adding `plan_002` and `plan_006`, while
  stricter gates add only `plan_002` and improve gain per extra generation. The
  stricter efficiency regime is now fresh-GPU confirmed, not only raw-rescored:
  57 generations, repair-selected `0.472768`, and `0.025794` gain per extra
  generation. The runner now exposes these as named gate modes (`score_max`,
  `efficiency`, and `custom`) so future GPU runs can choose an operating point
  without copying raw thresholds. The next research question is whether a
  learned/verifier-driven source gate can preserve the score while recovering
  that better budget-normalized return.
