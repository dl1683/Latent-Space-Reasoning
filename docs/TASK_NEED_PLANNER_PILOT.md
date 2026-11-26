# Task-need planner pilot

**Status:** stopped after three output-contract failures; no full-cohort planner result.

## Question

When a task has two relevant source passages, does explicitly representing
separate evidence needs help a small planner find the labelled passages beyond
what an equally informed query expander can find?

This tests the task-conditioned half of the goal in [GOAL.md](../GOAL.md). It
does not test query-blind maintenance, persistent world revision, answer
correctness, or a latent advantage. The corpus provides passage relevance, not
a complete evidence ontology.

## Registered comparison

- **BM25:** rank the 5,090 source passages by the original question and take
  eight.
- **Query expansion:** retrieve the same initial four passages, then ask
  Qwen3-0.6B for up to three ordinary search queries. Fuse the original and
  generated BM25 rankings and keep four additional passages.
- **Evidence-need plan:** give the same question and the same four initial
  passages to the same model. Ask for separate targets, relations, evidence
  descriptions, and one search query per need. Use the same output cap, query
  cap, fusion, and eight-passage total as query expansion.

Both generated arms use greedy decoding, one model call per question, the same
256-token output cap, identical initial evidence, and up to three extra
full-corpus BM25 scans. Outputs are constrained to concise JSON. Queries and
qrels are loaded separately; only the question and the first four passages
reach the model. The model does not see qrels or answer labels. Any full cohort
with a non-terminating or invalid model output is rejected before scoring.

The original 128-token protocol was executed once and failed because every
generation reached the token cap. Its fallback retrieval scores are invalid.
The follow-up 256-token canary also failed for the planner arm. No full cohort
is authorized by this protocol; further work should use the broader experiment
in [WHOLE_WORLD_EVALUATION.md](WHOLE_WORLD_EVALUATION.md), where need-level
outcomes are measured rather than using query generation as a proxy.

## Measures and decision rule

Report relevant-passage recall@8, mean per-question recall, and complete
two-passage sets@8. Report paired 95% intervals from 10,000 bootstrap draws
over connected components of questions that share relevant passages. Also
charge index construction, model loading, actual prompt/output tokens, model
time, BM25 score-document visits, retrieval time, parsing failures, and GPU
memory. The shared opening BM25 pass is charged once per arm in the per-system
comparison.

The planner earns a follow-up only if it improves labelled-passage recall over
both direct BM25 and query expansion on this fixed query set while preserving
a measured cost/quality tradeoff. A tie, a loss, fragile parsing, or a gain
erased by model and search costs ends this planner recipe. A positive result
would justify a larger, held-out task study with a true iterative-RAG control;
it would not establish task-level answer quality or a persistent latent world.

## Scope and inputs

- Dataset: local `zeta-alpha-ai/NanoHotpotQA`, snapshot
  `d79c0cdda980aba54842756770928035e1b61a51`, CC-BY-4.0. The selected snapshot
  has 5,090 corpus passages, 50 questions, and 100 qrels (two passages per
  question). Corpus SHA256 `ba6b24a43534da33144886762af4984dadcd54c8969deeb3cf6689d9fd3174b9`,
  query SHA256 `6b65aa944e0d12f9e9996a78fe6f3348c06c169da42cc85d6d4004eff9811890`,
  qrels SHA256 `22cdc93f1f37ad3c39b1a94f902c115e79d2f0fe349d77ab960d80cdd5444be5`.
- Model: local `Qwen/Qwen3-0.6B`, snapshot
  `c1899de289a04d12100db370d81485cdf75e47ca`; BF16, greedy, Transformers SDPA,
  CUDA, local files only. No model or corpus bytes are written into the repo.
- Hardware preflight: RTX 5070 Laptop GPU, 8,151 MiB total and 7,891 MiB free,
  53 C and 0% utilization at inspection. Available system RAM was 40.2 GiB.
  The script enforces a 12 GiB RAM floor and 4 GiB free-GPU-memory floor, logs
  GPU telemetry every ten completed questions, and checkpoints the call log.
- The existing BM25 implementation is reused from
  [`musique_dev.py`](../experiments/musique_dev.py). Model-call records and
  generated summaries go to a caller-selected directory outside the repo.
- Run the resource and prompt-length preflight first with
  `python -m experiments.task_need_planner_pilot --preflight-only`, then run the
  registered comparison with the same dataset/model paths and an output
  directory outside the repository.

## What this cannot establish

This 50-question retrieval set is a narrow proxy for task-directed evidence
planning. It cannot show that the model inferred every necessary relationship,
that selected passages entail an answer, that unknowns are calibrated, or that
results generalize to broad tasks such as strategy preparation. It has no
revision stream or continuous maintenance worker. Public benchmark exposure in
the pretrained checkpoint is unknown. A measured retrieval improvement is a
candidate-generation signal only; the next required comparator is a competent
iterative-RAG reader under the same full lifecycle and context budget.

## Instrument failure and amendment

Version 1 used Qwen3's default thinking template and a 128-token generation
cap. Across 50 questions, all 100 model calls (50 per arm) produced exactly
128 tokens, none terminated, and none parsed as complete JSON. The apparent
shared score—90/100 relevant passages and 40/50 complete support sets for each
arm—came from the script falling back to the same original BM25 ranking. It is
not a planner or query-expansion result and must not be used as one. The v1
run manifest records script SHA256
`4031881ca1b4d297c48b1eef2d43c2348f68dc4f90f771d01d4135196fc9b2cb`.

A one-question output-contract canary after disabling thinking still failed:
both model arms produced 128/128 tokens without EOS or valid complete JSON.
The failed canary used question `5ae5669755429960a22e02ec`. Its outputs and
calls are in `%TEMP%\lsr_task_need_planner_canary_20261126_02`. No qrels were
loaded by this canary.

The 256-token canary disabled thinking, requested concise JSON and used an
equal 256-token cap. The planner terminated after 79 tokens but left the root
JSON object unclosed (1/1 invalid); the query expander terminated after 81
tokens with valid JSON and three queries (1/1 valid). The output is saved in
`%TEMP%\lsr_task_need_planner_canary_v2_20261126`. The v2 script used SHA256
`38071b5d8725b4b73f1adbd608fbaffe1876246bfdd1cd23a5897a685757dc10`. This
shows that raising the cap alone does not fix the planner's output protocol.
The v2 script fails closed on malformed or non-terminating output; no
50-question v2 cohort ran.
