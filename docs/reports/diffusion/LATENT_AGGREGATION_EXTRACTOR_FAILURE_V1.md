# Latent Aggregation Extractor Failure Diagnostic

This diagnostic is post-hoc over the frozen inference replay labels.
Post-hoc diagnostic over frozen labels. It may guide the next frozen extractor, but it is not a promoted online aggregation result.

## Summary

- Components: `560`
- Replay online promotions: `0`
- Replay component precision: `1.000000`
- Replay component recall: `0.169565`
- Best threshold by F1: `0.1`
- Best-threshold precision: `1.000000`
- Best-threshold recall: `1.000000`
- Best-threshold F1: `1.000000`

## Threshold Sweep

| Threshold | Precision | Recall | F1 | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.100000 | 1.000000 | 1.000000 | 1.000000 | 230 | 0 | 0 |
| 0.200000 | 1.000000 | 0.760870 | 0.864198 | 175 | 0 | 55 |
| 0.300000 | 1.000000 | 0.495652 | 0.662791 | 114 | 0 | 116 |
| 0.400000 | 1.000000 | 0.278261 | 0.435374 | 64 | 0 | 166 |
| 0.500000 | 1.000000 | 0.169565 | 0.289963 | 39 | 0 | 191 |
| 0.600000 | 1.000000 | 0.082609 | 0.152610 | 19 | 0 | 211 |
| 0.700000 | 1.000000 | 0.030435 | 0.059072 | 7 | 0 | 223 |
| 0.800000 | 1.000000 | 0.017391 | 0.034188 | 4 | 0 | 226 |
| 0.900000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 230 |

## False-Negative Examples

| Task | Rubric Item | Best Literal Score | Source Span |
| --- | --- | ---: | --- |
| `plan_009` | check the original source text before trusting summaries | 0.142857 | Make a decision within the one hour before the customer review. |
| `plan_010` | state what evidence would justify keeping the upgrade | 0.142857 | Compare the results of the new run with the results of the run before the upgrade to determine if the gain is real. |
| `plan_019` | include user-facing or reviewer-facing acceptance criteria | 0.142857 | Additionally, ensure that the repair system improves the average score despite the trade-off of creating longer answers that users find harder to audit. |
| `plan_022` | choose logs that expose the suspected failure mechanism | 0.142857 | Choose logging option: - Final answers. |
| `plan_017` | state whether the claim should be narrowed or withdrawn | 0.166667 | If it passes without the hidden examples, the improvement is real and should likely be kept. |
| `plan_021` | separate checklist coverage from causal usefulness | 0.166667 | To audit the automated judge, create a set of questions that cover all checklist items. |
| `plan_021` | state what failure would invalidate the judge | 0.166667 | To audit the automated judge, create a set of questions that cover all checklist items. |
| `plan_024` | measure false refusals and missed harmful cases | 0.166667 | To validate the new refusal rule, conduct a small test run with a subset of the data data that includes harmless edge cases. |
| `plan_017` | measure regressions as well as the headline gain | 0.200000 | To determine if the improvement is real, conduct a quick test by removing the hidden examples from the prompt template and running the benchmark again. |
| `plan_018` | compare old and new rules on the same locked cases | 0.200000 | the the the the if the the if the the same in the same the the the the same the the the same as the same name the same the the same name the a the the same the the the same't the s |
| `plan_020` | report family-level gains and regressions | 0.200000 | This approach allows for a balanced assessment of the schedule's performance across different types of tasks and avoids overfitting to a single family. |
| `plan_021` | add a penalty for contradictions or irrelevant checklist dumping | 0.200000 | To audit the automated judge, create a set of questions that cover all checklist items. |

## Interpretation

The frozen literal extractor is not mainly hallucinating components; it is missing low-overlap components. On this frozen slice, lowering the literal threshold to 0.1 recovers all labeled components without false positives. Because that threshold was found after labels existed, it should become a diagnostic replay or a predeclared threshold for a new slice, not a retroactive promotion of the failed v1 run.
