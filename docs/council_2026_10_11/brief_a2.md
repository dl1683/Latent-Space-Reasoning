# Amendment 2 request (2026-10-11 ~19:10Z): stalled gate, registered curriculum arm

Read theory/LATENT_BRANCH_CAPACITY.md (incl. Amendment 1) and docs/council_2026_10_11/codex_a1.md + droid_a1.md, where you both REJECTED a staged curriculum "now" but said a stalled cell could motivate "a separately registered, matched-budget curriculum experiment".

Evidence (all dev, non-claim-bearing; gs://sc-bench-iqidis-artifacts/lsr/runs/):
- gate-q06 (frozen concurrent recipe, Qwen3-0.6B, 3 seeds, H100): at step 250/4687 the train loss is 0.698 (chance 0.693) and EVERY dev cell (widths 64/256/1024 x updates 0/1/2 x k 1/4/16) equals the dev base rate in all 3 seeds: the model emits a constant answer and ignores the state.
- diag-k1 (proximal rung i only: k=1, u=0; same trainer, lr 2e-4): the TEXT positive control (no bottleneck, full text in the answer context) reaches dev 0.984 by step 400; the STATE variant (width 1024 bottleneck, prompt and KV discarded) reaches 0.926 by step 250 and rising. At lr 1e-3 the state variant collapses to chance (text variant at 1e-3 reaches 0.98).
So the interface is learnable; the concurrent all-k all-update mixture stalls.

Proposed Amendment 2 (operator draft):
"The frozen concurrent recipe continues unchanged as the registered gate arm (seeds 11, 23, 47). A separately registered CURRICULUM arm is added with matched optimizer, lr, LoRA, widths, batch, step count and total presentations (at most 300,000): stage boundaries at fractions 0, .10, .20, .30, .45, .60 of the steps with (k set, update set) = ({1},{0}), ({1,2},{0,1}), ({1,2,4},{0,1,2}), ({1..8 powers},{0,1,2}), ({1..32 powers},{0,1,2}), ({1..64 powers},{0,1,2}) thereafter; global and branch-addressed toggles 50/50 as frozen. Checkpoint selection on the same dev score, after the last stage boundary only. The gate verdict is computed separately for each arm; the flagship escalates on whichever arm passes the gate, and the arm used is reported. The curriculum arm uses the same three seeds. No other definition changes."

Answer in under 250 words: SIGN as written, or give exact replacement sentences. Also say if anything in the evidence suggests a bug rather than a learning-dynamics stall (e.g. constant output) that must be ruled out first.
