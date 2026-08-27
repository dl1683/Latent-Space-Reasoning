# Benchmark validity assessment: the nested-arithmetic scaling and perturbation claims

**Verdict:** the `sweet_spot` nested-arithmetic benchmark has no dynamic range.
Every published number derived from it — the scaling ladder, the
perturbation-beats-scaling comparison, and the cost-per-capability table —
measures whether the model stops generating inside a 1024-token cap, not whether
it can do arithmetic. Three of the repository's headline claims do not survive
the controls below.

**Hardware:** NVIDIA GH200 480GB, Linux, conda env `gemma4`
(torch 2.11.0+cu130, transformers 5.5.3). All new runs at bfloat16, unquantized.
**Drivers:** `experiments/run_true_scaling_ladder.py`,
`experiments/run_cross_family_scaling_ladder.py`
**Raw results:** `experiments/true_ladder_*.json`,
`experiments/cross_family_ladder_*.json`

---

## Reproduce the headline in one command

```bash
python experiments/run_latent_sensitivity.py --model Qwen/Qwen3-32B \
  --quantization none --dtype bfloat16 --no-think --task-type nested \
  --difficulty sweet_spot --calibrate --n-calibrate 25 --max-new-tokens 1024
```

Published result for Qwen3-32B on these 25 tasks: **0%**.
That command: **100%**. Same tasks, same token cap; thinking mode off.

---

## 1. The claim under test

README.md reports:

> 2 random embedding tokens raise Qwen3-4B from 32% to 72% (plurality@10) on
> nested arithmetic. Scaling from 1.7B to 32B is flat or worse (28%→36%→0%).

and concludes that perturbation "unlocks capabilities that scaling cannot."

Two features of the published ladder invite suspicion before any new compute.
Every rung is a Qwen3 model and every rung is 4-bit quantized, so family and
numeric precision are confounded with scale. And the README's own description of
the 32B failure — "verbose natural-language explanations, exhausts the
1024-token budget, never states an answer" — describes a truncated decode, which
is not a capability measurement.

## 2. Every published rung was truncated

Read directly out of the stored result files. No new compute.

| Published rung (4-bit) | Accuracy | Mean tokens | Hit 1024 cap | Terminated by EOS |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 28% | 990 | 80% | 20% |
| Qwen3-4B | 32% | 934 | 76% | 24% |
| Qwen3-8B | 24% | 1020 | 96% | 4% |
| Qwen3-14B | 36% | 983 | 76% | 24% |
| Qwen3-32B | 0% | 1024 | 100% | 0% |

Three quarters to all of every rung failed to finish generating. Accuracy is
then scored by taking the last integer in whatever text survived the cut.

## 3. The controls

### 3.1 It is not quantization

`Qwen/Qwen3-32B` at bfloat16, unquantized, same 25 tasks, same 1024-token cap:
**4%** (vs 0% at 4-bit). Removing quantization changes nothing. Across the whole
ladder the bfloat16 thinking-on arm reproduces the published 4-bit numbers
(section 3.3). Quantization was never the cause.

### 3.2 It is not scale — it is the thinking trace

Same model, same weights, same tasks, same cap. The only change is whether the
chat template opens a thinking block.

| Qwen3-32B bf16 | Accuracy | Mean tokens | Hit cap | Terminated |
| --- | ---: | ---: | ---: | ---: |
| thinking on (default) | 4% | 1024 | 25/25 | 0/25 |
| thinking off (`--no-think`) | **100%** | 302 | 0/25 | 25/25 |

0% → 100% on one template flag. The model was never unable to do the
arithmetic; it was unable to finish saying so. On `nest_000` it correctly
derives `(34 + 23) = 57` and `(45 − 25) = 20`, then asks itself "How do I
approach multiplying fifty-seven times twenty efficiently?" and is truncated
mid-sentence. The extracted answer is `50`, scraped from the phrase "30 and 20
make 50" inside its own reasoning.

### 3.3 The full ladder, both arms

25 `sweet_spot` tasks, bfloat16, 1024-token cap. The thinking-on arm reproduces
the published protocol on this hardware; the thinking-off arm removes the
truncation.

| Model | Published (4-bit) | Thinking on (bf16) | Truncated | **Thinking off** | Truncated | Mean tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 28% | 20% | 88% | **96%** | 0% | 220 |
| Qwen3-4B | 32% | 20% | 84% | **100%** | 0% | 221 |
| Qwen3-8B | 24% | 16% | 92% | **96%** | 0% | 293 |
| Qwen3-14B | 36% | 36% | 84% | **100%** | 0% | 261 |
| Qwen3-32B | 0% | 16% | 92% | **100%** | 0% | 301 |

Un-truncated, every model from 1.7B up scores 96–100% using about a quarter of
the token budget it was supposedly exhausting.

### 3.4 Cross-family: Gemma 4 31B

`google/gemma-4-31B-it` at bfloat16 (revision `842da379…`), whose chat template
defaults `enable_thinking` to false:

| Task set | Accuracy | Mean tokens | Terminated |
| --- | ---: | ---: | ---: |
| `sweet_spot` | 100% (25/25) | 298 | 100% |
| `hard_nested` | 100% (25/25) | 443 | 100% |
| `brutal_nested` | 100% (25/25) | 650 | 100% |
| `planning` | 100% (25/25) | 121 | 100% |
| `frontier_nested` (new tier, added here) | 96% (24/25) | 1011 | 100% |
| `word_problem` | 80% nominal, 96% real (section 5.2) | 52 | 100% |

Outputs were read, not just counted: correct step-by-step derivations with
correct modular reductions (`427 ≡ 7 (mod 21)`, `561 % 14 = 1`).
`frontier_nested` — 4-digit × 3-digit multiplicands, 9–11 operations, three
levels of nesting — was written specifically to break a frontier model and
yielded exactly one genuine arithmetic error in 25.

### 3.5 The difficulty tiers are mislabelled

`generate_nested_tasks` documents its tiers by expected baseline accuracy:
`sweet_spot` "~60%", `medium_nested` "~42%", `hard_nested` "~8%",
`brutal_nested` "~12%". Those figures were calibrated under truncation.

Qwen3-4B at bfloat16 with thinking off, 2048-token budget, on `hard_nested` —
the tier labelled ~8%:

| | Accuracy | Terminated | Hit cap | Mean tokens |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B, thinking off | **92%** | 25/25 | 0/25 | 342 |

A tier documented as near-impossible is solved 92% of the time by a 4B model.
Every difficulty label in the generator describes how often the model ran out of
tokens, not how hard the arithmetic is.

## 4. What this does to the three claims

**"Scaling from 1.7B to 32B is flat or worse."** True of the published numbers,
vacuous as evidence. Truncated, the ladder measures verbosity against a fixed
cap; un-truncated, it is saturated at 1.7B. Neither regime has the dynamic range
to detect a scaling effect. The benchmark cannot answer the question it is
being used to answer.

**"Perturbation unlocks capabilities that scaling cannot."** There is no
capability to unlock: a 1.7B model scores 96% once it is allowed to finish. The
measured effect is real but is a termination effect. On the 4B perturbation
data, **every generation that terminated was correct — 100 of 100, both arms,
no exceptions**; truncated generations score 11–22%, which is the base rate for
the last integer of a severed trace happening to be right.

| Qwen3-4B, 25 tasks, 1024 cap | Accuracy | Mean tokens | Truncated | Terminated |
| --- | ---: | ---: | ---: | ---: |
| baseline (greedy) | 32% | 934 | 76% | 24% |
| perturbation ×10 | 52% | 909 | 62% | **38%** |

Accuracy tracks termination rate. Perturbation's entire benefit is raising the
chance the model stops in time, from 24% to 38%.

**The cost-per-capability table.** It compares perturbation×10 on 4B (72%)
against a 32B baseline (0%) at matched FLOPs and concludes perturbation buys
more capability per dollar. The correct comparison on the same model and budget
is that disabling thinking mode takes the 4B from 32% to **100% at 1× cost**.
Perturbation×10 does not beat parameter scaling; it loses to a one-line template
change by 28 points while spending ten times the tokens. This control was never
run.

The README's *diagnosis* is correct and worth preserving — "the bottleneck is
not knowledge, it is convergence," models "fail because greedy decoding locks
them into a verbose reasoning path that exhausts the token budget." What does
not follow is adopting an expensive intervention as the remedy without testing
the direct one.

## 5. Measurement defects found and fixed

### 5.1 Stop tokens were overridden (fixed)

`decode_with_raw_soft_prompt` passed `eos_token_id=tokenizer.eos_token_id` to
`generate`, overriding the checkpoint's configured stop set. Gemma ends a turn
with `<turn|>` (106); its `eos_token` is `<eos>` (1). The model therefore ran
past its answer, and last-integer scoring read off the trailing text: task
`nest_001` ran 961 tokens and scored `2` against an expected `159`; with the
stop set restored it terminates at 190 tokens and scores `159`.
Fixed by `harness.stop_token_ids`.

### 5.2 `_parse_integers` split decimals (fixed)

`extract_answer` took the last *integer* while the regex matched bare digit
runs, so `"412.5"` parsed as `[412, 5]` and extracted as `5`. This is the entire
`word_problem` gap above: the generator expects integer division without saying
so, Gemma answers `412.5` where `413` is expected, and the parser scores it `5`.
Four of five failures are this artifact; only one is a real error.

Fixed on the last *number*, not the last integer: `verify_answer` compares
against it, and `extract_answer` returns `None` when it is not integral rather
than falling back to an earlier integer in the working — which would report an
intermediate step as the answer. `dense_score` carried the same bug and is
aligned.

**Blast radius on published results: none.** An audit of every result JSON in
`experiments/` found 20 decimal-final responses out of 1650, of which **zero**
would flip from wrong to correct.

### 5.3 Token accounting on the perturbation path (fixed)

`generate(inputs_embeds=…)` returns only the newly generated tokens — which the
code relies on two lines later when it decodes the whole tensor as the
completion — but it also subtracted the prompt length when counting. Every
perturbation generation was under-counted by ~60 tokens, making
`generated_tokens >= max_new_tokens` unsatisfiable, so truncation was invisible
in that arm's recorded diagnostics. Accuracy is unaffected; token and throughput
figures were wrong. Stored data is recoverable because `prompt_tokens` was
recorded.

## 6. Portability changes (the code was Windows- and Qwen-only)

- `src/latent_reasoning/utils/architecture.py` (new) — resolve the text decoder
  stack and hidden size structurally. Gemma 4 loads as
  `Gemma4ForConditionalGeneration`: no `config.hidden_size` (it is under
  `text_config`), decoder at `model.language_model.layers`. The text path is
  probed before any bare `layers` fallback so a vision tower is never mistaken
  for the decoder.
- `src/latent_reasoning/utils/quantization.py` — `resolve_load_dtype`.
  Unquantized loads honour the checkpoint's declared dtype; forcing a
  bfloat16-native checkpoint into float16 costs 5 bits of exponent range, which
  Gemma's `final_logit_softcapping: 30.0` can overflow.
- `src/latent_reasoning/decode/steering.py` — uses the same resolver.
- `experiments/harness.py` — `stop_token_ids`, `split_after_reasoning`,
  `_parse_numbers`.
- `experiments/run_latent_sensitivity.py` — `--dtype`, `frontier_nested` tier,
  token-count fix, Gemma-aware reasoning-trace stripping.
- `experiments/run_cost_comparison_experiments.py` — `sys.executable` instead of
  a hardcoded `.venv/Scripts/python.exe`.

Tests: `tests/test_cross_model_portability.py` (27),
`tests/test_soft_prompt_token_accounting.py` (6). No GPU or download required.
The 5 remaining suite failures (gated-attention plan, ls20 planner safety)
pre-date this work, verified against a stashed tree.

## 7. What would actually test the underlying idea

The perturbation hypothesis is not refuted in general — it is untested, because
the benchmark used to support it cannot distinguish reasoning from termination.
A valid test needs a task set that is **capability-limited rather than
budget-limited**, and the precondition must be verified before measuring:
baseline terminates ~100% of the time and is still wrong.

Concretely, with what is already on disk: Qwen3-4B with thinking off (which pins
termination near 100%), calibrated across `hard_nested` / `brutal_nested` /
`frontier_nested` to find the tier where it lands in the 30–60% band. Then three
arms at matched token budget — baseline ×1, perturbation ×10, temperature ×10 —
reporting oracle@10 over baseline-failed tasks (label yield, the quantity that
matters if the goal is distillation) and plurality@10 (the no-verifier
fallback).

If perturbation beats temperature there, the diversity-source claim is real. If
they converge once nothing is truncated, the effect was always the token cap.

## 8. A note on revisions

A model id is not a fixed object. Mid-session, upstream published a new commit of
`google/gemma-4-31B-it` changing only the chat template — the 2026-07-09
canonical fix for "tool-calling loops, turn closures, and thinking
content-ordering," which also flips `enable_thinking` to default `false`. Both
weight shards are byte-identical across revisions (verified by sha256), but the
prompt the model sees is not, and on this benchmark that flag is worth 96
accuracy points. Runs here are pinned:
Gemma `842da3794eaa0b77d5f08bae87a17459d91ff475`,
Qwen3-32B `9216db5781bf21249d130ec9da846c4624c16137`.

---

Contributed by Igor Rivin ([@igorrivin](https://github.com/igorrivin)) with Claude Code (Claude Opus 5).
All runs on an NVIDIA GH200 480GB; raw result JSON for every number is in `experiments/`.
