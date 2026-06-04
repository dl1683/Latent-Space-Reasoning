# Diffusion Generated-Repair Value Hook V1

## Decision

Implement `--repair-selector generated_repair_value_v1` as the first runner-level generated-repair value hook.

The hook is label-free at selection time. It scores only generated repair candidates whose planning-quality signal improves over the recorded repair source. Direct source wins, no-lift repairs, and source-preservation-only evidence score zero.

## Mechanism

- Selector: `generated_repair_value_v1`
- Source signal: `repair.source_planning_quality_score`
- Candidate signal: label-free planning quality of the generated repair text
- Gate: `candidate_planning_quality - source_planning_quality > 0`
- Score: positive planning-quality delta plus a small existing seed-realization guard
- Live status: implemented as a named runner selector, but still requires fresh runner-level validation before public promotion language

## Evidence

- Fit boundary: [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V18.md](DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V18.md)
- Replay input: `eval_results\diffusion_language\generated_repair_v18_label_raw.jsonl`
- Hook rescore: `eval_results\diffusion_language\generated_repair_v18_value_hook_scores.json`
- Hook report: `eval_results\diffusion_language\generated_repair_v18_value_hook_report.md`

The no-generation rescore uses the committed v18 raw artifact with `--repair-selector generated_repair_value_v1`.

## Result

- Run ID: `diffusion-d4a90959bf5734b2`
- Full generations: `26` from the reused raw artifact
- Selected generated repairs: `plan_137`, `plan_139`
- Rejected no-lift generated repairs: `plan_141`, `plan_144`
- Repair-covered task delta versus fixed: `+0.035223`
- Repair-covered task delta versus random: `+0.058652`
- Task delta per extra generation: `+0.071`
- Oracle headroom versus repair: `0.000`

## Boundary

This is the first implementation step after the v18 replay boundary, not a final controller claim. The next gate must run `generated_repair_value_v1` on a fresh slice before treating it as a promoted live selector.
