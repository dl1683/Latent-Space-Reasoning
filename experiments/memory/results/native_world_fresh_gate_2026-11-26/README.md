# Fresh world task forced-choice smoke gate

This is a one-case forced-choice smoke gate for a new synthetic cold-chain
task. It is separate from the retired p2 fixture and does not use T8. The 0.6B
reader saw the complete pinned task plus all five source bodies, with one of
three answer labels permitted. Each option prewrites the full recommendation
and source-title list. The task therefore checks candidate selection only; it
does not test inferred obligations, generated findings or claim-level
grounding. The unrelated stock detail shares the handoff source.

The pinned Qwen3-0.6B model answered `1`; the frozen answer key was `2`. It
therefore selected East Clinic even though the p0 source chain maps S-4's
red seal to M-18 and then to North Clinic. The output was a normally
terminated, constrained one-token answer. The ordinary full-text baseline
received all five critical source IDs in a 784-token prompt and used 3.85
seconds of CPU generation after a 0.47-second model load.

The forced-choice gate failed **0/1**. Native publication, maintenance,
derived-off, memory-omitted and native full-text arms were deliberately not
run on this fixture: comparing them before a competent full-text answer would
not isolate a latent-world contribution. The score opened gold only after
inference. Raw prompt token IDs, source/task hashes, model revision, resource
preflight and timings are in [`direct_text.json`](direct_text.json); the
post-inference score and hashes are in [`score.json`](score.json).

This single visible task is a stop/continue signal only. It does not establish
a general Qwen3-0.6B limitation, answer quality on natural corpora, or any
effect of maintained latent state. Do not tune or rerun this prompt. A future
maintenance comparison needs a separate task for which a competent text or
iterative-RAG control succeeds, then same-pin operator removal and restoration
with all source-selection traces retained.

Each option already contains its complete answer and an aggregate source-title
list. The label-level score does not verify individual answer claims or
citations. The stock note also shares a source with the red-alert status, so
this gate does not test discovery of a disconnected side fact. These limits
are why the result is recorded as a smoke test rather than a task-level
competence result. The local snapshot path and model/tokenizer hashes are
preserved in [`model_custody.json`](model_custody.json). The exact runner
source used for this captured run is [`runner_at_run.py`](runner_at_run.py).
After the run, its docstring was corrected to call this a forced-choice smoke
test; the captured implementation bytes are preserved above.
