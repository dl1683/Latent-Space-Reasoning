**Instrument PASS; interpretation limited.** All 24 cells pass the defined gates (positive control 1.0). Independently recomputed E and `r_req` match every summary. No extraction, changed-slot denominator, or split-scoring bug found. No files edited; no model rerun.

1. **Bugs and overclaims.** Layer selection correctly maximizes dev accuracy; k=1 ties select the lowest layer. Selection favors seen-name transfer: 8B W1 k=16 reaches **0.529 at layer 18**, versus 0.427 selected. Its histogram margin there exceeds 0.10, but binding was not scored there; “all layers fail E” is unsupported ([summary:3](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/cloud/readout_8b_full_summary.jsonl:3), [selection:183](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/readout.py:183)).

   Seen k=4 denominators are **2,047/2,046**, after collision filtering. Reconstruction found one train/dev duplicate in W2 k=4; none survives into seen tests. At k=1, every seen test repeats a training fact; unseen tests remain name-disjoint ([splits:146](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/readout.py:146)).

   The 32B shuffle is 0.127/0.129. It passes the **one-sided** gate, but is anomalous, not validated “chance.” Repeated k=1 inputs and shuffled-label fluctuations could induce anticorrelation; inspect confusion matrices and repeat shuffle seeds before interpreting it ([gate:185](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/readout.py:185)). Bootstrap intervals condition on fitted readers and fixed name pools.

2. **Prediction scores, 8B W1/W2.** Registered criteria: [R:422](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/LATENT_BRANCH_CAPACITY.md:422), [Amendment:481](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/LATENT_BRANCH_CAPACITY.md:481).

   | Prediction | W1 / W2 |
   |---|---|
   | k4 E and margin ≥0.15 | PASS / PASS |
   | k16 margin ≥0.05 | FAIL (.0425) / PASS (.0629) |
   | k16 worst slot near chance | Descriptive PASS (.245) / PASS (.288); tolerance unspecified |
   | unseen k4 rank ≥8 | FAIL (6) / PASS (12) |
   | unseen k16 rank ≥2×k4 | **Not testable:** E fails; cannot honestly assign numerical PASS/FAIL |
   | E_seen k4 and k16 | PASS / PASS |
   | k16 gap ≥.15, interval excludes zero | PASS / PASS |
   | seen k16 rank ≥32 | PASS (48) / PASS (48) |
   | unseen k16 E fails | PASS / PASS |

3. **Licensed sentences.** “8B’s registered readers recover slot-ordered colour information beyond histogram information at k4; selected-layer unseen k16 fails E, while seen k16 passes.” “Larger-model smokes show the same qualitative split pattern.” These are **not scaling evidence**: training counts, rank grids and reader seeds differ; 72B also changes model generation ([8B config:40](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/configs/readout_8b_full.json:40), [72B config:26](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/configs/readout_72b_smoke.json:26)).

4. **Alternative mechanism and decisive control.** Additive name nuisance, token-length/position shifts, name-conditioned colour codes, and reader overfitting can explain the advantage without name-addressable binding. **Names plus positions alone cannot predict fresh IID colours** ([generator:31](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/native.py:31)). Run a counterbalanced, token-length-matched **rename × reorder** test, preserving entity–colour assignments, with fresh colours and an entity-query reader receiving only closing state plus queried name—not slot routing. Include a colour-free nuisance null.

5. **Continue.** This control is higher leverage than another size sweep. Linear probing remains Rⁿ scaffolding; the denizen-facing target is reliable identity and retrieval under renaming and movement.

