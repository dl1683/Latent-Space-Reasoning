# Fresh 0.6B reader-competence screen

**Run date:** 27 November 2026
**Model:** Qwen3-0.6B, snapshot `c1899de289a04d12100db370d81485cdf75e47ca`
**Outcome:** 0/18 complete, correct, source-grounded answers.

This is a small synthetic reader-calibration result. The model received the
full pinned source packet in all sessions. It did not use the latent memory,
retrieval, a maintenance worker, or any same-pin removal/restoration arm. The
result is not evidence for or against latent maintenance and does not establish
a general model ceiling.

The exact scope, per-panel results, interpretation and limitations are in the
[results ledger](../../../../docs/RESULTS.md#fresh-06b-reader-competence-screen-27-november-2026).
The task contract is specified in
[`FRESH_REASONER_COMPETENCE_DIAGNOSTIC.md`](../../../../docs/FRESH_REASONER_COMPETENCE_DIAGNOSTIC.md).
The runner and frozen calibration inputs are in
[`experiments/memory`](../../).

## What is preserved

- `sessions.jsonl` contains the exact prompts, generations, token accounting,
  termination/schema checks and device samples for all 18 sessions.
- `blind_outputs.jsonl` is the arm-label-omitted view used for content scoring.
- `blind_scoring_initial.json` preserves the first per-session grade. A second
  AI content review found that r16 did not explicitly connect Q-17 to H13, so
  the corrected `blind_scoring.json` reports 8/9 explicit package-to-schedule
  links instead of 9/9. `second_content_review.json` records the recheck.
- `assignment.json` stores the task-pin and U/N/E assignments.
- `plan.json` records prompt hashes and input token counts; `preflight.json`
  records the tokenizer, budget and resource check; `run_manifest.json` records
  the model and run settings.
- `schema_mask_trial/` stores the abandoned token-mask pilot. It is a single
  partial session and has no semantic grade. The pilot was stopped after 774
  generated tokens took 319 seconds (about 2.4 tokens/second).

The model runner did not load the evaluator gold file. That separation was not
enforced by an OS-level access boundary. The run's randomized arm schedule was
visible in the preflight transcript before scoring, so the scoring view omitted
arm labels but the grading process was not fully blind. The second AI reviewer
also opened preflight but did not use or report arm assignments. Tasks, gold
and initial qualitative scores were authored or assigned by the working agent
and were not independently adjudicated. The second review is a content recheck,
not human certification. Treat the panel as instrument-development evidence.
Do not tune on these six task-pins or use them as held-out evaluation cases.

## Hashes

SHA-256 digests for the captured model inputs and primary run records:

| File | SHA-256 |
|---|---|
| `fresh_reasoner_competence_screen.py` | `ad99f8423c1ce99f2d678ba7ea9a7b6a4aca9b7faa823d8da099d44043c57045` |
| `fresh_reasoner_diagnostic_sources.json` | `18a6752ec71a39fa4190795fa76f5128924b2c753092493e48d1606cdd5097f0` |
| `fresh_reasoner_diagnostic_tasks.json` | `242d9d3fc06f9f135dbcbffd9014262ae125d02978f1fe5681e5f05e6deabd9f` |
| `fresh_reasoner_diagnostic_neutral.json` | `5e98548bbdab3020355d33b3b013208088838e3d6b2d98c2e9aa3d932ef47665` |
| `fresh_reasoner_diagnostic_markers.json` | `3a20343f1eb8a90ed1fbc2b0bfaaf83b55ed83fef5b62b693172c790766d0321` |
| `assignment.json` | `d5021fcb60eacde9beb31754125296cfe6a46eb9b9d4509071b99e90ea8b36d2` |
| `plan.json` | `c60ee0dc78f0a0615612f6950ef77fabd8b59f15f87cbf4ce65047035c0b692f` |
| `preflight.json` | `625d38480de68758c07c3c15c6533908234362a0ef7673317d16ad8cdb37a8d9` |
| `sessions.jsonl` | `4d222040a637a089df78878444c902617999d30c216832c287df4ac234a8ccde` |
| `blind_outputs.jsonl` | `81730eaa55719678d1c731dcde088eb8906e9c7436b8accb3d31159b65536d83` |
| `blind_scoring_initial.json` | `16e81f38b2012866b73844f87d4e2df4fd7109e0007b289dde3d09a2185184dc` |
| `blind_scoring.json` | `473c0fabf0f185dcc516b68e405e982f08936aa2e0b71e15294b678e43ae3c42` |
| `second_content_review.json` | `7f3f42c9886c2491e6ba0b7bc3d7e250e87b33825f6612e6bd018dbdb95d5271` |
| `run_manifest.json` | `7d90c8157edd724d79323ebfd69741cf315b17ff45f74b9b486397bcac75fa3c` |
| `schema_mask_trial/interrupted_run_note.json` | `beb50c72d4fb4f20708a637389c1ef2043066043f10b3ddbcda45fffbfea0665` |
| `schema_mask_trial/sessions.jsonl.pending` | `eb9824cf6310e916ceabac36c0cde03a0c02d46fae40c45f05360f4413767145` |
