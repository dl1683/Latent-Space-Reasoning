# Diffusion Phase-Anchor Boundary

This note records the first full lean mixed benchmark for the pre-generation
phase-anchor repair operator. It is a diagnostic boundary, not the promoted
public claim.

## Full Mixed Run

Run ID: `diffusion-9dabba8829d29658`

Artifacts:

- `eval_results/diffusion_language/llada_moe_mixed_constraint_span_phase_anchor_fresh_v1_raw.jsonl`
- `eval_results/diffusion_language/llada_moe_mixed_constraint_span_phase_anchor_fresh_v1_scores.json`
- `eval_results/diffusion_language/llada_moe_mixed_constraint_span_phase_anchor_fresh_v1_report.md`

Command:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --task-preset lean_gpu_mixed --candidates llada-moe-7b-a1b-instruct-hf --limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 --repair-pack constraint_span_phase_anchor --repair-source-policy fixed --repair-spend-trigger denoise_phase_repairability --repair-source-min-chars 240 --repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 --repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 --repair-selector planning_quality_prompt_coverage_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --evolved-promotion-margin 0.015 --device cuda --dtype bfloat16 --raw-output eval_results\diffusion_language\llada_moe_mixed_constraint_span_phase_anchor_fresh_v1_raw.jsonl --scores-output eval_results\diffusion_language\llada_moe_mixed_constraint_span_phase_anchor_fresh_v1_scores.json --report-output eval_results\diffusion_language\llada_moe_mixed_constraint_span_phase_anchor_fresh_v1_report.md
```

## Score

| Run | Repair-covered task score | Delta vs fixed | Delta vs random | Relative cost |
| --- | ---: | ---: | ---: | ---: |
| Fixed/greedy baseline | `0.412277` | `0.000000` | `0.040152` | `1.000000x` |
| Random perturbation | `0.372125` | `-0.040152` | `0.000000` | `1.000000x` |
| Phase-anchor repair | `0.476786` | `+0.064509` | `+0.104661` | `2.625000x` |
| Promoted preservation-seeded repair | `0.531116` | `+0.118839` | `+0.158991` | `2.625000x` |

The phase-anchor run used `27` full model generations, matching the promoted
cap-`32` generation count. It repaired the same five planning tasks and had zero
repair-oracle headroom. It is a real controlled improvement over fixed/random,
but it is dominated by the promoted final/history anchor plus preservation-seed
policy at the same cost.

## Per-Task Boundary

| Task | Phase source | Phase score | Promoted source | Promoted score | Boundary |
| --- | --- | ---: | --- | ---: | --- |
| `plan_001` | history step `30` | `0.528214` | history step `31` | `0.528214` | Tie. |
| `plan_003` | history step `30` | `0.485714` | final | `0.538214` | Safe late phase text is worse than final-source repair. |
| `plan_004` | final fallback | `0.359286` | final | `0.621786` | The generic phase pack lacks the public-claim seed/gate. |
| `plan_006` | history step `31` | `0.550357` | final | `0.584286` | Safe late phase text is still weaker than final-source repair. |
| `plan_007` | history step `31` | `0.497857` | final | `0.583571` | The one-task smoke gain does not beat the seeded final-source policy. |

Phase-anchor source metadata:

- `4` selected repairs used history: `history_phase_first_repairable_skeleton`.
- `1` selected repair fell back to final: `phase_anchor_not_retention_safe`.
- Selected history anchors were late: steps `30`, `30`, `31`, and `31`.
- Every selected repair improved its source (`5/0/0` versus source), but the
  phase source was not the best source for the strongest known repair policy.

## Mechanism Takeaway

The first repairable denoise skeleton is useful evidence, but it is not yet a
better repair source by itself. The current failure mode is retention-safety
lag: early skeletons expose task structure, but the first text state safe enough
to preserve is usually near the final denoise step. At that point, replacing the
final output with the late history text can remove useful realized details.

The next operator should not blindly replace the final repair source with the
first safe phase source. Better directions:

- Use phase features as a spend gate and source-quality signal, while keeping
  final-source repair when the phase anchor is late or semantically weaker.
- Use the phase state as a conditioning contrast or seed constraint, not as the
  whole repair source.
- Add a retention-lag feature: first repairable skeleton step versus first
  retention-safe repair source step.
- Test a hybrid pack that keeps the promoted preservation-seeded controls and
  only switches to phase-source repair when phase geometry predicts an actual
  source advantage.

## Hybrid Follow-Up

The next implementation tested that last direction directly.

Run ID: `diffusion-9386ee5300a75528`

Artifacts:

- `eval_results/diffusion_language/llada_moe_mixed_phase_hybrid_preserve_seeded_gated_fresh_v2_raw.jsonl`
- `eval_results/diffusion_language/llada_moe_mixed_phase_hybrid_preserve_seeded_gated_fresh_v2_scores.json`
- `eval_results/diffusion_language/llada_moe_mixed_phase_hybrid_preserve_seeded_gated_fresh_v2_report.md`

The new `constraint_span_phase_hybrid_preserve_seeded_gated` pack keeps the
promoted preservation-seeded repair controls, scans phase history, records phase
timing features, and only switches to a history source when the phase state also
passes the stricter normal history-anchor retention standard. This rejects weak
late skeletons like `plan_003`, where the looser v1 hybrid saw a tiny
span-score edge but only `0.943503` target similarity and `0.908714` final-char
ratio.

| Run | Repair-covered task score | Delta vs fixed | Delta vs random | Relative cost |
| --- | ---: | ---: | ---: | ---: |
| Phase-anchor repair | `0.476786` | `+0.064509` | `+0.104661` | `2.625000x` |
| Loose phase hybrid v1 | `0.524554` | `+0.112277` | `+0.152429` | `2.625000x` |
| Strict phase hybrid v2 | `0.531116` | `+0.118839` | `+0.158991` | `2.625000x` |
| Promoted preservation-seeded repair | `0.531116` | `+0.118839` | `+0.158991` | `2.625000x` |

The strict hybrid ties the promoted score and cost while making phase evidence
explicit. It chooses history only for `plan_001`; it keeps final-source repair
for `plan_003`, `plan_004`, `plan_006`, and `plan_007`.

Strict hybrid source metadata:

- `plan_001`: `history`, reason `phase_hybrid_history_source_advantage`, score
  `0.528214`, lag `20`.
- `plan_003`: `final`, reason `phase_hybrid_final_no_source_advantage`, score
  `0.538214`; rejected the weak phase source that v1 used.
- `plan_004`: `final`, reason `phase_hybrid_final_no_safe_repairable_skeleton`,
  score `0.621786`.
- `plan_006`: `final`, reason `phase_hybrid_final_no_source_advantage`, score
  `0.584286`.
- `plan_007`: `final`, reason `phase_hybrid_final_no_source_advantage`, score
  `0.583571`.

This gives the current mechanism target: preserve the public score while making
the denoise phase trace an explicit latent-state selector. The phase sequence is
not discarded; it becomes evidence for repair spend, source switching, and
future learned selectors.
