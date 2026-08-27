# LLaDA-MoE Mixed Adaptive Source Policy Comparison

Raw source:
`llada_moe_mixed_revision_constraint_span_adaptive_source_score_max_v1_raw.jsonl`

Suite: `lean_gpu_mixed` (`plan_001`-`plan_008`, `math_001`, `sym_002`,
`sci_001`).

## Result

| Policy | Report | Records | Repair Score | Delta vs Fixed | Delta vs Random | Delta vs Evolved | Extra Budget vs Evolved | Gain / Extra Gen | W/T/L vs Evolved | Oracle Headroom |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Adaptive `score_max` | `llada_moe_mixed_revision_constraint_span_adaptive_source_score_max_v1_report.md` | 76 | 0.474107 | +0.061830 | +0.101982 | +0.030357 | 1.250 | 0.024286 | 7/1/0 | 0.000 |
| Adaptive `efficiency` rescore | `llada_moe_mixed_revision_constraint_span_adaptive_source_efficiency_rescore_v1_report.md` | 75 | 0.472768 | +0.060491 | +0.100643 | +0.029018 | 1.125 | 0.025794 | 7/1/0 | 0.000 |
| Single-source `non_revision_evolved` rescore | `llada_moe_mixed_revision_constraint_span_nonrev_source_rescore_v1_report.md` | 74 | 0.472143 | +0.059866 | +0.100018 | +0.028393 | 1.000 | 0.028393 | 6/2/0 | 0.001 |

## Interpretation

The adaptive source policy now holds in the full lean mixed protocol, not only
the planning-only slice. Exact math, symbolic, and science checks are already
solved by the base MoE generations, so repair coverage remains `8/11` overall
and `8/8` repair-eligible.

Compared with the earlier transferred dense-LLaDA repair policy
(`llada_moe_mixed_gated_ranked_span_guarded_exact_v1_report.md`), the new
revision plus MoE-specific `constraint_span` line lifts mixed planning repair
from `0.446` to `0.474107` and improves repair-vs-evolved wins/ties/losses
from `1/7/0` to `7/1/0`.

Use `score_max` when the goal is best observed planning repair score. Use
`efficiency` when the goal is almost the same score with one fewer generation.
Use single-source `non_revision_evolved` when budget-normalized gain matters
more than the last `0.002` score.
