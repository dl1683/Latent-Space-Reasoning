# Native world addressing p2 gate

This is the auditable output of a one-task visible synthetic pilot run on
`research-review` at base commit `145f548`. The source-only publisher ran before
the reader opened task `glass-arch-p2`. Both inference runs kept evaluator gold
closed; separate scoring commands opened it after generation.

The ordinary direct-text follow-up ran at base commit `2faefad` using runner
SHA-256 `7d26287b62249cde198e09f9528f83997f0692091aae874b4acfc7e8d3c23476`.

## Frozen inputs and runtime

- Model: local Qwen3-0.6B snapshot `c1899de289a04d12100db370d81485cdf75e47ca`.
- Runtime fingerprint: `9d923088bd5b130d88e33bef4a3818cfc3ff03aad227f696bf94dc921ec88a38`.
- Tokenizer chat-template SHA-256: `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
- Source stream SHA-256: `3e9cf471218fe8c03a1bc00b62bf8a1edafbaf9deca1f667737fa0c55209144b`.
- Task set SHA-256: `f8bbc20263efa1166fcce587ea5e5368dc9770dc001d2dfde07f1ea8857c25d3`.
- Gold SHA-256: `98efcbe4c8d91427c42b14402782f0634e7166565c1461246375d9305461a6ac`.

The neighboring `source_feeds/` and `publications/` preserve all three source
pins and their native numeric state. `reads.jsonl` contains the four paired
reader outputs and selection receipts; `run.json` and `publication_index.json`
record limits and usage; `score.json` records exact scoring denominators and
the hash of the read records.

## Observed result

Query-blind maintenance installed 1 continuation at p0, 0 at corrected p1,
and 1 at restored p2. The p2 task required three critical sources. The active
arm selected two, including the event record and sector assignment, but not the
current record-to-sector mapping. Its derived-bank receipt was `empty` at both
latent steps, so it routed and applied 0 derived steps.

The gold answer was `1` (Copper). Active, derived-off and memory-omitted each
predicted `2`; the full-text arm received all five source bodies, covered all
three critical source IDs, and predicted `3` (unresolved). Each arm scored 0/1.
The full-text arm used the same native reader with the corpus appended; the
ordinary Qwen comparison is recorded separately below.

The separate ordinary Qwen full-context run used the exact 336-token p2 prompt,
the same model revision, tokenizer/template and two-token label-then-EOS
constraint. It returned `2` (Amber), which scored **0/1** against gold `1`
(Copper). The output IDs were `[17, 151645]`; greedy generation used two
backbone calls, 337 input positions, and 3.04 seconds on CPU. It did not use
the native recurrence or memory. Transformers warned that the attention mask
could not be inferred because pad and EOS IDs match; inspection of the pinned
installed generation code showed it fell back to all ones. This prompt has no
padding, so the fallback matches its intended mask. The raw prompt IDs, hashes,
run record and post-generation score are in `direct_text_baseline.json` and
`direct_text_score.json`.

One earlier attempt on the same task was excluded: the script passed the
literal string `default` as a chat template, producing a one-token prompt. The
corrected run checked that the decoded prompt retained the exact task; the
task-only input was 63 tokens and the full-text input was 336.

This one visible relation family is a falsification/interface diagnostic.
Neither ordinary Qwen full-text nor the native full-text arm solved it, and the
active derived route was unused. It cannot compare latent memory against a
competent text reasoner. Stop this fixture here. A later broad T8 one-shot also
missed its critical needs, so the competence gate remains unmet; see the
[separate T8 record](https://github.com/dl1683/Latent-Space-Reasoning/blob/b2dbbcc8d9d5f3e70c153cdb546f860cc5fc6b13/experiments/memory/results/whole_world_t8_direct_text_2026-11-26/README.md).
