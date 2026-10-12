# Fresh error hunt: Registration N 8B smoke (STRICTLY READ-ONLY; do not edit any file)

Read in detail (file:line): experiments/cloud/lsr_jobs/bindq.py, configs/bindq_8b_smoke.json, Registration N (+
amendments) at the end of theory/LATENT_BRANCH_CAPACITY.md, docs/council_2026_10_11/codex_bindq_code.md, and
experiments/results/cloud/bindq_8b_smoke_summary.jsonl. Compare with the Registration R slot-reader results
(experiments/results/cloud/readout_8b_full_summary.jsonl).
Observation: all 4 cells (W1/W2 x k=4,16) pass gates (pc 1.0; colour-free null at chance) but are inconclusive_reader:
entity-query reader (MLP chosen) stays at/below histogram-mode in every quad and layer (k4 0.47-0.51 vs 0.53; k16
0.36 vs 0.38); superposition diagnostic 0.69-0.77 (k4), 0.49-0.56 (k16). The R slot reader on the same kind of
closing states reached 0.80-0.83 unseen-name at k4 (20 names).
Questions (answer in under 300 words): 1. Bugs that could make the entity reader fail artificially (e.g. q(n) at the
wrong layer/position, reader training, the h-only path)? 2. Licensed sentence. 3. Is the full run unwarranted under
the registration? 4. The single most decisive next control (e.g. same architecture with a slot-index query instead of
a name query on the same closing states, to separate "information is slot-ordered, not name-keyed" from "reader
inadequate"; or more training names/collections).
