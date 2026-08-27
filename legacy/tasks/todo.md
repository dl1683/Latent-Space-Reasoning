# Benchmark validity work (GH200, Linux)

## CURRENT: capability-limited diversity study

Assessment complete — see `docs/BENCHMARK_VALIDITY_ASSESSMENT.md`. The
nested-arithmetic benchmark measures termination, not reasoning, so
perturbation-vs-temperature was never actually tested. Now testing it properly.

Driver: `experiments/run_capability_limited_diversity_study.py`.

Precondition, enforced in code (the run ABORTS if violated): baseline must
terminate on >=95% of tasks and still be wrong. Thinking mode off, 2048-token
budget.

Calibrating Qwen3-4B (bf16, no-think) to find a tier in the 30-60% band:

| tier | accuracy | terminated | mean tokens |
| --- | ---: | ---: | ---: |
| `hard_nested` | 92% | 25/25 | 342 |
| `brutal_nested` | pending | | |
| `frontier_nested` | pending | | |

Headline metric is RESCUE RATE = oracle@k over baseline-failed tasks: the label
yield a distillation loop actually consumes. Plurality@k is reported as the
no-verifier fallback. Temperature is run at more than one setting, and taking
the best of them favours the alternative hypothesis — deliberately.

---

# Cross-family scaling ladder on Gemma 4 31B (GH200, Linux)

## Question

The published ladder in `README.md` concludes "parameter scaling is flat". Every
rung of it is a Qwen3 model and every rung is 4-bit quantized, so two candidate
explanations are not separated:

1. scaling genuinely does not help on these tasks; or
2. 4-bit NF4 degrades the larger models, and the 32B = 0% rung is a broken
   decode rather than a capability measurement.

Gemma 4 31B at bfloat16 shares neither confound and is locally available.

## Status of the port (done)

The experiment code was Windows-only and Qwen-shaped. Fixed, all in
`experiments/` and `src/latent_reasoning/`:

- `utils/architecture.py` (new) — resolve the text decoder stack and hidden size
  structurally. Gemma 4 is `Gemma4ForConditionalGeneration`: no
  `config.hidden_size`, decoder at `model.language_model.layers`.
- `utils/quantization.py` — `resolve_load_dtype`. Unquantized loads honour the
  checkpoint's declared dtype; forcing a bfloat16-native model into float16
  overflows Gemma's attention logits and `final_logit_softcapping`.
- `harness.py` — `stop_token_ids`. Gemma ends a turn with `<turn|>` (106), not
  `<eos>` (1). The perturbation decode path was passing only
  `tokenizer.eos_token_id` to `generate`, which *overrode* the checkpoint's stop
  set, so the model ran past its answer and last-integer extraction picked up
  trailing text.
- `harness.py` — `split_after_reasoning`. Strips a closed reasoning trace by
  token id, covering both Qwen `</think>` and Gemma `<channel|>`.
- `run_cost_comparison_experiments.py` — `sys.executable` instead of a
  hardcoded `.venv/Scripts/python.exe`.
- `run_cross_family_scaling_ladder.py` (new) — the driver.

Environment: conda env `gemma4` (torch 2.11.0+cu130, transformers 5.5.3).
Model pinned to revision `842da379…` (2026-07-09 canonical chat template).

## RESULT (see docs/CROSS_FAMILY_SCALING_LADDER.md)

Gemma 4 31B bf16 baseline, 25 tasks per row, greedy, 1024 tokens:

| task set | accuracy |
| --- | ---: |
| sweet_spot | 100% |
| hard_nested | 100% |
| brutal_nested | 100% |
| planning | 100% |
| word_problem | 80% nominal, 96% real (4/5 failures are a decimal-parsing artifact) |

Qwen3-32B at 4-bit scores 0% on the same 25 sweet_spot tasks. The 0% rung is
therefore not about parameter count. Quantization is the remaining suspect.

Perturbation arm is blocked: nothing to improve on at 100%.

### Follow-up work (all four approved)

1. **Qwen3-32B bf16 control** — weights downloaded (62 GB, revision
   `9216db5781bf21249d130ec9da846c4624c16137`). Baseline running at the
   identical 1024-token cap so quantization is the only variable vs the
   published 0% rung.
   - Note: `snapshot_download` with an explicit `revision=` does not write
     `refs/main`, so offline loads fail until it is created by hand.
2. **Frontier task tier** — `frontier_nested` added to
   `generate_nested_tasks`: 4-digit x 3-digit multiplicands, 9-11 operations,
   three levels of nesting, modulo operands non-negative by construction.
   Needs calibration on Gemma before the perturbation ladder runs.
3. **Decimal scorer fix** — DONE. `_parse_numbers` treats a decimal as one
   number; `verify_answer` compares the last *number*; `extract_answer` returns
   None when that number is not integral rather than falling back to an
   intermediate integer. `dense_score` aligned. 7 new tests.
4. **README revision** — pending the Qwen control result.

Test suite: 1109 passing. The 5 remaining failures (gated-attention plan,
ls20 planner safety) pre-date this work — verified by running the suite on a
stashed tree.

### Original decisions (now resolved)

1. Harder task tier, or a ceiling-free task family (text generation, judged)?
2. Run the Qwen3-32B bf16 control (~62 GB download)? It converts "quantization
   is the suspect" into "quantization was the cause".
3. Fix `_parse_integers` decimal splitting? Audit says zero published results
   flip, but it is the V11+ canonical scorer.
4. README's "parameter scaling is flat" claim needs revision — left alone
   pending a call, since it is a published claim.

## Open design question: headroom (RESOLVED — no headroom anywhere)

`sweet_spot` difficulty was calibrated so Qwen3-4B scores ~60%. A 3-task probe
put Gemma 4 31B at 3/3. If it ceilings on the full 25, the perturbation arm has
nothing to show — though the *ladder* claim is still answered, since a 31B
scoring near 100% where Qwen3-32B-4bit scored 0% settles which explanation is
right.

Plan: calibrate first (baseline only, 25 tasks) at `sweet_spot`,
`hard_nested`, `brutal_nested`; then run the full 11-arm protocol at
`sweet_spot` (comparability) plus whichever level lands baseline in ~30–70%
(headroom).

## Run protocol (held identical to the published ladder)

25 seeded nested-arithmetic tasks, greedy, 1024 new tokens, perturbation =
2 random embedding tokens at the model's native embedding RMS
(`--control-mode random_noise --num-soft-tokens 2`), 10 seeds.

Measured throughput: ~9–10 tok/s for 31B bf16 unbatched. Budget ~3 h per
11-arm run.

## Known caveats to state with any result

- Gemma bf16 vs Qwen 4-bit differ in family *and* quantization. The clean
  control is Qwen3-32B at bf16, which fits on this GPU but is not cached
  (~62 GB download). `--include-qwen32b-bf16` runs it.
- The published Qwen numbers were produced on a different machine; only the
  within-Gemma baseline-vs-perturbation contrast is same-hardware.
- `stop_token_ids` slightly changes the Qwen perturbation path too (it now
  passes the full configured stop set). Any re-run of Qwen arms is not
  bit-comparable to the published numbers.
