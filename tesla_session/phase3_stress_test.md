# Tesla Mode Phase 3: Stress Test — Phase A Design

## Purpose
Brutally stress-test Phase A before any implementation begins. Find every assumption that could be wrong, every edge case that would break the system, every place where complexity is hiding.

Phase A is: `SpecGenerator → RawSoftPrefixApplicator → FullGeneration → CandidateLogger → OutputValidator → AtlasLogger`

---

## Stress Test 1: The Sequence Alignment Problem

**The assumption**: When using `inputs_embeds`, generated token IDs in `output.sequences[0]` can be cleanly sliced to extract only the generated portion (not the prompt portion).

**What could break**:
- HuggingFace `generate()` with `inputs_embeds` returns sequences that start at position 0 of the generated tokens (prompt tokens are NOT returned in sequences when `inputs_embeds` is used — they were never tokenized). OR it returns sequences including dummy tokens. The behavior is model-dependent and has changed across transformers versions.
- `encoder.py:969` skips `prompt_len` tokens from the output, suggesting the existing codebase assumes prompt tokens ARE included in output sequences. If Phase A does the opposite, early features will be computed on the wrong tokens.
- At window w=1: logprob slope is undefined (need ≥2 points for a slope). The EarlyFeatures contract says "slope behavior for n_observed < 2" is undefined — this must be handled explicitly (return None, not 0.0).

**The edge case that breaks this**:
- Token count = 2 prefix tokens. Prompt = 50 tokens. Output: sequences[0] could be length 50+2+N or just N, depending on transformers version. If Phase A assumes N and gets 52+N, every window is off by 52. MI analysis breaks silently.

**Required fix**: Smoke test — generate 1 candidate with inputs_embeds, compare decoded `output.sequences[0]` to expected output text character-by-character. Log `generation_start_index` (the first token position that belongs to generated output). This is the most important correctness check.

---

## Stress Test 2: The Calibrated RMS Interaction with Quantization

**The assumption**: `effective_rms = embedding_rms × rms_multiplier` works the same way for 4-bit quantized models as for full-precision models.

**What could break**:
- In 4-bit quantization, the embedding layer is typically NOT quantized (it remains float16/bfloat16). The `get_input_embeddings()` return should be at full precision. BUT the forward pass casts to the quantized compute dtype. If the RMS calibration is done on the full-precision embedding matrix, but the model internally renormalizes or casts the soft prefix differently, the effective energy is wrong.
- `embedding_rms` is calibrated from the token embedding matrix at load time. But if quantization changes the embedding scale (possible with QLoRA-style quantization), the calibration is stale.
- Qwen3 uses tied embeddings (input = output embeddings). Does quantization affect the tied embedding differently than an untied one?

**Required fix**: After applying the soft prefix, compute `actual_rms = modified_embeds[:, :token_count, :].square().mean().sqrt()`. Assert it matches `effective_rms` within 1%. If not, abort. Log `actual_prefix_rms` in atlas.

---

## Stress Test 3: The Atlas Logging Race Condition

**The assumption**: Each candidate is logged atomically to the JSONL atlas. If the run crashes mid-candidate, we can resume from the last complete record.

**What could break**:
- Partial JSON writes: if a crash happens mid-record, the last line of the JSONL is malformed. On resume, the entire JSONL is unreadable past the corruption point.
- Truncation of `full_output` vs `full_token_ids`: if `full_output` is stored as UTF-8 text but `full_token_ids` is a separate array, they can desync if the tokenizer decodes differently than expected (BOS/EOS handling, special tokens).
- The `soft_prefix_hash` is computed from the noise tensor. But if `torch.manual_seed(seed)` is called before the noise is generated AND other random operations happen in between (e.g., model sampling, dataloader shuffling), the seed produces a different tensor than expected. The hash would be wrong.

**Required fix**:
- Write each record as a single JSON line terminated by `\n`. Validate on read: skip malformed last lines.
- Compute `soft_prefix_hash` IMMEDIATELY after generating the noise tensor, before any other torch operations. Store the hash AND the seed for reproducibility verification.
- On resume: validate that re-generating with the stored seed produces the same hash.

---

## Stress Test 4: The Output Validator — Arithmetic Extract_Answer Correctness

**The assumption**: `extract_answer` reliably extracts the final integer from arithmetic outputs.

**What could break**:
- Qwen3's thinking mode outputs: `<think>...42...28...computation...56...</think>Answer: 56`. The last integer in the full output may be 56 (correct), but if `extract_answer` uses a simple "last integer in the string" rule, it could extract 56 from inside the `<think>` block on a different format.
- What counts as "last integer"? If the output is `...The answer is 56. Let me verify: 14 × 4 = 56. ✓`, the last integer is 56 but appears after repeated computations.
- Negative numbers: does `extract_answer` handle `-42`? The current `last-integer-wins` grading only extracts non-negative integers (based on past audit findings).
- The `answer_anywhere_correct` flag: does it search the entire output including the `<think>` block? If so, it will conflate "found the right intermediate value in reasoning" with "solved the problem."

**Required fix**:
- Run `extract_answer` on 10 existing outputs and manually verify correctness.
- Document exactly what `extract_answer` searches: full output or only output-of-`</think>` section?
- For `answer_anywhere_correct`: define whether `<think>` blocks are included or excluded. Log this as `answer_anywhere_search_scope` in atlas.

---

## Stress Test 5: The MI Analysis — Degenerate Feature Cases

**The assumption**: The EarlyFeatures extracted at early windows (w=1, 4, 8) carry meaningful signal about final correctness.

**What could actually happen**:
- At w=1: the first token is almost always `<think>` (>99.99% probability per existing experiment). There is near-zero variance in the first token. MI from token-based features at w=1 will be essentially zero. Only logprob-based features (e.g., entropy of the first-token distribution) can carry signal.
- At w=4: all candidates may produce nearly identical early tokens (`<think>\n\n`) regardless of prefix. The diversity of trajectories may not manifest until w=32-64 when the model starts actually computing.
- `repetition_rate` at w=4: with only 4 tokens, repetition rate is meaningless (need at least 4 tokens to form a 4-gram, let alone a repeat). This feature will be 0 for all candidates at w≤8.
- `logprob_slope` at w=2: requires at least 2 points. At w=1, this is undefined. The slope of 2 points is just the difference — not a robust trend estimate.

**The degenerate case that corrupts MI**:
- If all candidates at all windows produce near-identical features until w=64, the MI estimator sees zero variance in features and reports near-zero MI. This is correct but would wrongly kill Phase B even if there IS signal in text-space features not captured by our feature set.
- Conversely: if easy tasks consistently produce high-confidence early tokens (high logprobs) and hard tasks produce low-confidence early tokens, `mean_logprob` will correlate with task difficulty, not with candidate-level correctness. This inflates within-task MI, which is exactly why task-held-out evaluation is required.

**Required fix**: Add `feature_variance` statistics to the MI analysis report. If any feature has near-zero variance across candidates within task groups, report it as a degenerate feature and exclude from the router.

---

## Stress Test 6: The Baseline Candidate Definition

**The assumption**: The no-prefix baseline (`token_count=0`) uses `input_ids` directly (not `inputs_embeds`), and this produces the same output as the original experiments.

**What could break**:
- The original experiments use `tokenizer.apply_chat_template()` with specific formatting. If Phase A's baseline uses a different prompt template or system prompt, the baseline is not comparable to the original 32% arithmetic result. The MI analysis would compare Phase A perturbation candidates against a different baseline than the published result.
- `token_count=0` spec means no `inputs_embeds` — but the Phase A pipeline is built around `inputs_embeds`. Switching to `input_ids` for the baseline creates a code fork that could have subtle differences (different attention mask handling, different position IDs).
- The baseline result should be IDENTICAL to running the model without any Phase A code. If it's not, Phase A has introduced a confound.

**Required fix**: Run the baseline spec and compare output to a direct `model.generate(input_ids=...)` call with identical parameters. Assert outputs are byte-for-byte identical. If not, the baseline spec has a bug.

---

## Stress Test 7: VRAM Budget for Phase A Data Collection

**The assumption**: 100 task groups × 11 candidates (baseline + 10 prefix) × max_new_tokens=1024, with `output_scores=True`, `output_attentions=True`, `output_hidden_states=True`, on RTX 5090 (~24GB VRAM), is feasible per-candidate.

**What could break**:
- `output_attentions=True` returns attention tensors for EVERY layer at EVERY generated step. For Qwen3-4B: 32 layers × 32 heads × (prefix_len + gen_len) × (prefix_len + gen_len) attention matrix = massive memory. For 1024 generated tokens with 50-token prompt: 32 × 32 × 1074 × 1074 × float16 ≈ 2.4 GB per candidate. This will OOM immediately.
- `output_hidden_states=True`: 32 layers × 1074 tokens × 2560 dims × float16 ≈ 180 MB per candidate. Manageable per-step, but if stored for all 1024 steps: 32 × 1024 × 2560 × float16 ≈ 167 MB total. Multiplied by batch — manageable.
- `output_scores=True`: vocab_size × generated_tokens = 152064 × 1024 × float16 ≈ 300 MB per candidate. This is feasible if processed step-by-step and discarded after feature extraction.

**The real constraint**: We don't need full attention tensors for all 1024 tokens. We only need early window features (w ≤ 128). Plan: collect attentions and hidden states for the first 128 generated tokens only, then disable collection. This reduces VRAM by 8x.

**Required fix**: Implement a generation hook that collects telemetry for the first `max_observe_tokens=128` steps, then switches to a lean generation mode for the remaining tokens. Don't try to store full 1024-step tensors in memory.

---

## Stress Test 8: The Oracle-Relative Label Degeneracy

**The assumption**: The oracle winner (judge-selected best candidate) in each task group represents a meaningful quality signal, not just "least bad among bad outputs."

**What could break**:
- If all 10 candidates for a task produce garbage outputs (e.g., all truncated, all wrong), the oracle winner is still labeled `correct=1`. The MI analysis then learns to predict "which garbage output the judge preferred," not "which output is actually good."
- For legal tasks: the judge may have strong length/style biases. If long outputs consistently win, `length_forecast` at w=32 will have high MI not because it predicts quality but because it predicts output length → judge preference.
- Tie breaking: if 2 candidates score identically, which gets `correct=1`? If both do, the MI analysis has multi-label targets within a task group.

**Required fix**:
- Require a minimum score margin for oracle winner selection: oracle winner must score ≥ 0.5 points above median candidate on the judging rubric. Otherwise, the task group is marked "no oracle winner" and excluded from MI training data.
- Log `oracle_margin` (winner score - median score) in atlas. Report distribution of margins in MI analysis.
- For ties: label both as `correct=1` and use ranked-oracle metric instead of binary.

---

## Stress Test 9: Generation Config Drift

**The assumption**: `do_sample=False, temperature=None, top_p=None` guarantees byte-for-byte reproducibility given the same seed and inputs.

**What could break**:
- Across HuggingFace versions, `do_sample=False` with certain kwargs has silently introduced bugs where temperature or top_p defaults are applied. If `temperature=None` is interpreted as `temperature=1.0` internally in some versions, sampling is applied.
- CUDA non-determinism: even with `do_sample=False`, some GPU operations are non-deterministic by default (e.g., atomic operations in attention). `torch.use_deterministic_algorithms(True)` is needed for exact reproducibility across runs.
- The `soft_prefix_hash` is used to verify reproducibility. But if CUDA non-determinism causes slightly different outputs, the hash won't match on re-run, falsely flagging contamination.

**Required fix**: Add a determinism smoke test — generate the same candidate twice with the same seed and assert `output.sequences` are identical. If not, document which CUDA ops are non-deterministic and add a "determinism_verified" flag to atlas records.

---

## Summary: The 5 Highest-Risk Assumptions

| # | Assumption | Risk Level | Failure Mode |
|---|---|---|---|
| 1 | `inputs_embeds` sequence slicing is correct | CRITICAL | Every early feature computed on wrong tokens — silent data corruption |
| 2 | `output_attentions=True` is VRAM-feasible for 1024 tokens | HIGH | OOM kills data collection; changes plan fundamentally |
| 3 | Oracle-relative label is meaningful (not "best garbage") | HIGH | MI measures judge preference for length/style, not reasoning quality |
| 4 | Baseline spec produces same output as native `model.generate()` | HIGH | Baseline is not the published baseline — comparisons invalid |
| 5 | Early features (w≤32) have non-zero variance within task groups | MEDIUM | MI estimator reports zero signal — Phase B killed by feature degeneracy, not true absence of signal |

**The single most dangerous failure**: Silent sequence misalignment in inputs_embeds. If `generation_start_index` is off by even 1 token, all window-sliced features are wrong, and the entire MI analysis is built on corrupted data. This must be verified empirically before any data collection.
