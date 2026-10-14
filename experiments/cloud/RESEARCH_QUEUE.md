# Research handoff to the external cloud agent

Owner direction, 2026-10-15: Google Cloud experiments are authorized; all GPU
work runs on the cloud, compute credit only. Per-program spend caps are removed;
the "Scale Y1 - AI" credit expires 2026-10-17. This does not revive historical
crons, watchdogs, job queues or allocations. Model weights stay on Google Cloud.

Current responsibilities (owner-directed 2026-10-15):
- **Supervisor — Claude coordinator:** decides what runs when, keeps this queue
  current, and directs documentation and commit updates.
- **Codex / Droid:** thinking, design dialogue, proof review and fresh error hunts.
- **Z Code:** implementation; owns the E1 must-fix list.
- **Antigravity (agy):** grunt work, custody and blackboard updates.
- **Cloud operator:** executes only `ready` entries under the supervisor's direction.
Neither entry below inherits `FIRST_CONSTRUCTION_PROPOSAL.md` budgets or model exclusions.

## Execution contract

Each ready entry supplies a hypothesis, decision-changing control, pinned inputs,
canonical runner/config, separate positive-control smoke, resource ceiling, stop
conditions and output location. Pin the executed Git commit and all dependency,
model, tokenizer/template and data revisions in the returned receipt. Record full
construction/training/inference/retrieval/readout/fallback costs and failures.
Return results to the specified repository location; retain raw evidence in a
versioned external store with hashes and recovery paths. Never put credentials here.

Read root STATE and the entry status before launching. `design` is not runnable;
`ready` means its smoke/runner and execution bounds are supplied. The execution
agent records `running`, attempt identity, allocation and then `completed` or
`failed`, preserving prior attempts. A returned run is not an admitted claim until
its controls and independent audit pass. No automatic restart of inherited work.

## Current entries

| ID | Status | Purpose | Runner / output | Resources |
|---|---|---|---|---|
| response-exact-2026-10-14 | completed locally (12 cases PASS); no cloud needed | Finite rational witnesses for the correction-ready response design | `experiments/analysis/correction_capsule_exact.py`, adjacent fixed JSON; output to `experiments/results/analysis/correction_capsule_exact.json` | Stdlib CPU, 12 cases, dimension at most 8; executed locally |
| E1-dependency-continuation-sufficiency | design; prioritized (not ready / running / allocated) | Determine what survives construction: whether indexed per-record frozen contextual states over a versioned package-dependency ledger preserve transitive dependency consequences under late policy reveal and supporting-record retraction | Design from Droid r4 (`internal/analysis_2026_10_14/droid_r4_cloud.md`). 12 128-dim FP16 slots + 1 policy slot (3,328 payload bytes). Evaluated against exact incremental graph evaluator, full-text replay, same-budget noncontextual features, and shuffled/omitted controls. Gate: lower clustered 95% CI >= 0.90, replay deficit <= 0.05. Cheap smoke: 256 train / 64 dev / 64 held-out clusters, 5 packages/6 records | Bounded adapter/config and externally approved time/spend limits required; no allocation yet |
| E2-cheap-continuation | design; conditional on E1 pass (not ready / running / allocated) | Determine whether recurring backbone computation can disappear: learn an updater operating directly on E1 states vs reconstruction/re-encoding and ordinary incremental evaluation, with no correction payloads or action logs at query time | Design from Droid r4 (`internal/analysis_2026_10_14/droid_r4_cloud.md`). Freezes competent E1 snapshot reader; tests shuffled/no-op actions, state swaps, query-after-update, horizons 1/2/4, retract/reinsert, policy replacement. Complete cost accounting (training, ingestion, action encoding, maintenance, readout, fallback) at declared reuse count | Conditional on E1 passing; no allocation yet |
| gemma4-controls-2026-10-15 | COMPLETED 2026-10-15 (v1 and v2 all pass: identity exact, hook live, donor-follow 1.00 n=64, token-difference 0/64 failures, batch-shape max logit diff 0.375 with argmax agreement 1.0, receiver-self-SET exactly 0; results `experiments/results/cloud/gemma4_controls_summary.jsonl`, `gemma4_controls_v2_summary.jsonl`); formerly: OWNER GO-AHEAD 2026-10-15 for this instrument control only (launch now, cap $25, compute credit only). LAUNCHED 2026-10-15: sc-bench-lsr-gemma4-controls-2026-10-15-000, europe-west4-a, a2-highgpu-2g Spot (no A100-80GB capacity in the zones tried; 2 x A100-40GB, device_map auto). Instrument validity alone does not establish the correction construction; its separate proof review passed with listed fixes applied | Amendment G4 instrument-validity smoke for Gemma 4 31B: identity-hook exactness, live-hook check, value-token donor-SET control (>= 0.95) | `experiments/cloud/lsr_jobs/gemma_controls.py`, `experiments/cloud/configs/gemma4_controls.json` (kind `gemma_controls`); output `experiments/results/cloud/gemma4_controls*` | a2-ultragpu-1g Spot, planning rate ~$5.1/h (verify), cap $25, ~1.5 h; stop on any assertion failure (identity != 0, hook not live, revision mismatch, donor-follow < 0.95); no latent-state claim |
| E1-gemma4-smoke-2026-10-15 | NOT READY: runner drafted (`experiments/cloud/lsr_jobs/e1_ledger.py`, `experiments/cloud/configs/e1_gemma4_smoke.json`, both uncommitted); independent Codex review verdict NOT-READY with MUST-fix list in `docs/council_2026_10_15_e1_runner_codex.md`; implementation owner: Z Code. Launch only after those fixes and a second Codex review say LAUNCHABLE | E1's cheapest cloud smoke on the pinned Gemma 4 31B stem (design: `internal/analysis_2026_10_14/droid_r4_cloud.md`) | runner/config above; register kind `e1_ledger` in `run.py` | a2-highgpu-2g Spot ~$4.42/h; recommended $25 total across attempts, 5 h maximum; stop on exact-input competence < 0.95, oracle pair non-separation, alignment failure, omitted-state leakage |

The bridge must compare a learned constructor with an equally exposed ordinary
structured-state constructor and exact response engine. It must separate encoding
defect, update defect, recovery cost and actual recurring backbone work. Matching
today's output, lowering a latent loss or testing multiple models alone is not the
success criterion. Choose the smallest discriminating construction before a sweep.

E1 and E2 originate from the Droid round 4 design dialogue (`internal/analysis_2026_10_14/droid_r4_cloud.md`).
They replace pooled Horn-bit memory with an addressed dependency object over an explicit versioned
package-dependency ledger (minimal pair: $s\to a\to t$ and $b\to t$ with $x\to b$ vs $s\to b$).
Neither entry inherits `FIRST_CONSTRUCTION_PROPOSAL.md` budgets or model exclusions. Neither entry
is ready, running, or allocated.

This queue will be updated as analytical decisions become executable. Cloud
project, region, accelerator availability and spend envelope belong in the
external execution agent's current allocation record; access alone supplies none
of those values. The cloud operator may launch only `ready` entries under the supervisor's direction and the owner authorization above; this documentation sync launches none.
