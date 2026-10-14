# Execution spec: E1 alignment and hook smoke (separate positive-control smoke; DESIGN / NOT READY)

2026-10-14 council (`internal/supervision/council_20261014T1226Z/r3_codex_out.md` section 5; Droid r2 section 6).
Queue row: `E1-align-hook-2026-10-14` in `experiments/cloud/RESEARCH_QUEUE.md`. Parent design:
`dependency_continuation_v1.md` (section 4A). This is an INSTRUMENT smoke on the pinned stem. It makes no
E1 sufficiency, memory or economic claim, and it does not replace the generic Gemma 4 controls
(`experiments/results/cloud/gemma4_controls_v2_summary.jsonl`), which validated only the layer-15 colour
instrument and say nothing about E1 names, ledger syntax, snapshot semantics, replay features or receiver-level noise.

## Why a separate smoke
AGENTS.md evidence gates: every costly registered experiment first passes its claimed operation's positive
control as a separate cheap smoke (token alignment, hooks, actual intervention effects, fixed-input noise and
batch-shape effects) before a sweep spends money. The full E1 workload (2,112+ replay forwards at the draft
corpus, 120-epoch reader fits, several billed hours, unmeasured) must not be the first thing that touches
the real tokenizer.

## Hypotheses and checks (each is an assertion that stops the run)
1. **Actual-tokenizer alignment before bulk work.** On a finite development-only corpus (matched alternate
   families, E1 ledger/name/query/replay syntax; no confirmation data): record-end positions, record
   spans, two-token package names in-family and standalone, isolated query keys (exactly two tokens), right
   padding indices and replay last-real-token positions all pass retokenize-through checks (e1_ledger.py:418-470,
   552-565). If too few names satisfy the two-token contract, stop before weight loading; never relax the assertion.
2. **Hook equivalence and a live intervention.** Identity/no-op hook reproduces base outputs exactly; a SET at the
   registered record position changes the reader-visible feature; no KV retained (`use_cache=False`); decoder
   hook shape matches (`gemma_controls.py` `decoder_layers`, `LayerHook`).
3. **Fixed-input noise over the whole deployed input path.** Repeat, singleton, reordered and mixed-length
   batches over ledger slots, isolated name keys and replay features, plus the combined deployed input; PROPOSED
   tolerance 100% decision agreement on the finite preflight set; finite features and logits; raw differences reported.
   (Droid C: the current check perturbs slots only, e1_ledger.py:803-813.)
4. **Placement.** Record the device map and both GPUs' peak allocated and reserved memory; reject unexpected
   CPU/disk offload; `max_memory` 34 GiB per GPU as a pilot setting; extraction batch 1 then 2 only after measured
   headroom; `logits_to_keep=1` where logits are discarded; right-padded replay scoring addresses the last real
   token; readers on CPU.
5. **Timed replay development pilot.** A small independently selected replay recipe on development data with
   timing; reports throughput so the full workload can be budgeted. Replay competence >= 0.95 per view is
   reported, not required for this smoke's pass (it is required later for replay-relative interpretation).
6. **Terminal receipt.** Explicit `instrument_pass` true/false with every check's value; nonzero exit on failure.

## Pins and configuration (PROPOSED; file `experiments/cloud/configs/e1_gemma4_alignment_hook_v1.json` does not exist yet)
- Kind `e1_ledger`, entry `alignment_hook_smoke` (both to be implemented by Z Code; `run.py` KINDS lacks `e1_ledger` today).
- Model `google/gemma-4-31B`, revision `5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89` (model and tokenizer), BF16,
  `device_map: auto` with `max_memory`; decoder layer 15 as an explicit integer; one layer only.
- Machine `a2-highgpu-2g` Spot, `accelerator: null`, disk 300 GB, zone `europe-west4-a` only (no fallback), planning
  rate 4.42 USD/h, one attempt, unique run name `e1-align-<UTC>-a01`, no automatic retry or relaunch.
- Deadline: `cost_cap_usd: 4.42` makes `lsr_cloud.py create_vm` set `--max-run-duration` to 3,600 s (hours = cap/rate);
  the job stops work at 3,300 s and syncs. `idle_min 10`, `grace_min 30`; loading must touch the heartbeat, and a
  stalled load with no progress for 10 minutes is a stop condition.
- Requirements: the dispatcher installs `requirements` ranges at boot (`vm_startup.sh` pip install), so the job must write
  `pip freeze`, torch/transformers/accelerate/tokenizers/huggingface-hub/numpy/jinja2 versions, image name and
  tokenizer/template hash into its receipt; exact versions recovered from the Gemma v1/v2 logs are
  UNVERIFIED-until-matched (NEXT_STEPS Agy A2). Do not invent versions.
- Code identity: `upload_bundle` uploads the WORKING-TREE `lsr_jobs/` directory. Bind the review and the run to the
  bytes: Agy's manifest (NEXT_STEPS A1) records HEAD, the tracked diff and the SHA-256 of every uploaded file
  including untracked `e1_ledger.py`; the VM reads the manifest back before executing; any change between review
  and upload voids the review.

## Duration, cost, monitoring (estimates, not measurements)
15-30 billed minutes if loading is favorable (about 1.11-2.21 USD at 4.42 USD/h); cold weight download and load could
use the full 60-minute lifetime (about 4.42 USD). Operator monitoring and stop conditions: RESEARCH_QUEUE.md
"Cloud operator hour plan", Path A.

## Outcomes
- All checks pass: record `instrument_pass`; the full E1 smoke may then be proposed (still needs its own fixes, frozen
  corpus and review). Nothing about sufficiency or memory follows.
- Alignment, name-contract or hook failure: repair on CPU and the stub tokenizer where possible, then a new attempt with a new name
  and review; no further stem minutes meanwhile.
- Noise failure: change batch policy or precision (a registered change); it is not evidence against memory.
- Placement or OOM failure: adjust `max_memory`/batch and re-review; do not change precision, quantize or swap weights as an unregistered repair.

## Non-claims
No E1 pass, no latent-memory result, no replay or contextual competence conclusion, no economic result.
