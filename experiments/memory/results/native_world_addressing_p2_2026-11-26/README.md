# Native world addressing p2 gate

This is the auditable output of a one-task visible synthetic pilot run on
`research-review` at base commit `145f548`. The source-only publisher ran before
the reader opened task `glass-arch-p2`; the evaluator file was opened only by
the final scoring command.

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
The full-text arm used the same native reader with the corpus appended; it is
not a qualified conventional RAG or direct-generation baseline.

One earlier attempt on the same task was excluded: the script passed the
literal string `default` as a chat template, producing a one-token prompt. The
corrected run checked that the decoded prompt retained the exact task; the
task-only input was 63 tokens and the full-text input was 336.

This one visible relation family is a falsification/interface diagnostic. Its
incompetent full-text control and absent derived route do not establish a
latent-world advantage or a general task-reasoning result. Do not scale this
protocol until the text control can solve the task and the task-addressed read
uses the required evidence.
