# Fresh error hunt: N-control (slot-index query) on Qwen3-8B (STRICTLY READ-ONLY; do not edit any file)

Read (file:line): experiments/cloud/lsr_jobs/bindq.py (slot mode + ref reader), configs/bindq_8b_slotctl.json and
bindq_8b_smoke.json, the N-control registration at the end of theory/LATENT_BRANCH_CAPACITY.md, and
experiments/results/cloud/bindq_8b_slotctl_summary.jsonl plus bindq_8b_smoke_summary.jsonl and
readout_8b_full_summary.jsonl.
Observation: same states/splits as the N smoke. Slot-query reader k=4: 0.99-1.00 in all quads incl. renamed names
(hist-mode 0.53); k=16: 0.62-0.73 (hist 0.38). R's per-slot linear reference on these states: k=4 0.97-0.99,
k=16 0.60-0.71. Name-query readers (N smoke) stayed <= hist-mode. Earlier R (20 training names) gave unseen-name
k=4 0.80 and k=16 0.43.
Answer in under 300 words: 1. Bugs/overclaims (could the slot reader exploit something trivial, e.g. slot-specific
token positions encoding colour via length? colours are single tokens; is the one-hot query path correct?).
2. Licensed sentence. 3. Does this indicate that R's held-out-name gap came from having only 20 training names?
4. Highest-leverage next step.
