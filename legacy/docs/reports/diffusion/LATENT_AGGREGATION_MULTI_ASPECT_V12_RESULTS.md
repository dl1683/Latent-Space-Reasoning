# Latent Aggregation Multi-Aspect V12 — Filtered Replication Results

**Verdict: NO-GO on the Gemini judge family.** The filtered task-specific
clause arm did not beat the task-aware generic arm.

## What was run

- Frozen plan: [V12 complement freeze](LATENT_AGGREGATION_MULTI_ASPECT_V12_COMPLEMENT_FREEZE.md)
  — 120 fresh tasks (`plan_537`–`plan_656`), five arms, four judge models
  across three families.
- Executed: **116 tasks** (`n_tasks` in the study manifest; the four-task
  difference from the frozen plan is not documented in the manifest and is
  recorded here as an open discrepancy), **one judge model**
  (`gemini-2.5-pro`) — the Anthropic and OpenAI judges were not run for lack of
  API access at the time. 348 judge calls, 2 parse failures (99.4% success).
- Arms: `anchor`, `fixed_generic`, `task_aware_generic`,
  `true_clause_unfiltered`, `true_clause_filtered`.

## Primary endpoint

`true_clause_filtered` vs `task_aware_generic`: **53 vs 63 wins** (0 ties,
n = 116), win rate 45.7%, binomial p = 0.85. Gate was win rate ≥ 60% with
p < 0.05. Not met. Judge-family agreement gate (≥ 2 of 3 families) could not be
evaluated with one family.

## Full pairwise table (gemini-2.5-pro, n = 116)

| Pair | Wins A | Wins B | Ties |
| --- | ---: | ---: | ---: |
| anchor vs task_aware_generic | 5 | 110 | 1 |
| anchor vs fixed_generic | 2 | 113 | 1 |
| fixed_generic vs task_aware_generic | 52 | 63 | 1 |
| task_aware_generic vs true_clause_filtered | 63 | 53 | 0 |
| task_aware_generic vs true_clause_unfiltered | 62 | 54 | 0 |
| fixed_generic vs true_clause_filtered | 63 | 40 | 13 |
| fixed_generic vs true_clause_unfiltered | 49 | 52 | 15 |

Hierarchy on this judge: task_aware_generic > fixed_generic ≳ true_clause_unfiltered > true_clause_filtered ≫ anchor.

## Reading

- Every clause arm beats the anchor by a wide margin: the clause-append
  mechanism works.
- Generic clauses beat true task-specific clauses. This is the opposite of the
  V11/confirmatory hypothesis.
- The defect filter hurt rather than helped (filtered 40 vs unfiltered 52
  against fixed_generic).
- Same-vendor confound: the task-aware generic clauses and the defect filter
  were generated with a Gemini model and judged by Gemini Pro. A cross-family
  judge is required before this result is treated as more than a
  single-family NO-GO.

## Artifacts

- `eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_study_manifest.json` (hash `74ba4b8997533708`)
- `eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_judge_results.json`
- `eval_results/diffusion_language/latent_aggregation_multi_aspect_v12_raw.jsonl`, `_scores.json`
- Prompts and packets: [V12 complement prompts](LATENT_AGGREGATION_MULTI_ASPECT_V12_COMPLEMENT_PROMPTS.md), [V12 packet report](LATENT_AGGREGATION_MULTI_ASPECT_V12_COMPLEMENT_PACKET_REPORT.md), [V12 label report](LATENT_AGGREGATION_MULTI_ASPECT_V12_LABEL_REPORT.md)

## Next step

Run one non-Gemini judge family on the same manifest. If it also returns
NO-GO, the filtered-clause hypothesis is closed; if it reverses, complete all
families.
