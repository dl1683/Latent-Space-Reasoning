# Execution spec: Q1 learned quadratic continuation state (CPU only; DESIGN)

2026-10-14 council (Droid r2 section 5 "learned quadratic continuation state"; Codex r3 section 3 adopts it in parallel
with E1 repair). Queue row: `Q1-learned-quadratic-continuation-cpu` in `experiments/cloud/RESEARCH_QUEUE.md`.
Mathematics: `theory/CORRECTION_READY_RESPONSE.md` (identities at lines 52-66, witness 68-76, `G+/-` mixing witness
137-149, realizability 143-149, control and next question 202-240). Nothing here is registered or launched. No cloud
and no GPU: a CPU-only entry cannot trigger the cloud authorization path.

## Hypothesis and why this is the direct bridge
Preserving today's answer does not preserve intervention consequences. A learned constructor/updater/reader that
retains the susceptibility `G` (not only the response `S, F`) can execute a named correction and keep unrelated worlds
and a paid historical checkpoint correct without reopening the world. This is the direct learned counterpart of the
exact capsule; it makes no transfer claim to the dependency ledger (`CORRECTION_READY_RESPONSE.md:235-241`) and is not an
E1 prerequisite. If the exact dynamic Schur/resolvent engine with identical access wins or ties, that is the result
to record, and the question becomes what harder operation or reuse would justify a learned carrier.

## World and state
- Two independent four-port rational SPD susceptibility worlds, `S=1`, `F=e1^T`, finite rational grid, legal
  cumulative corrections `0 <= D <= I` on the declared dyadic grid. Embed the existing `G+/-` witness
  (`G+/- = [[1, +/-1/32],[+/-1/32, 1]]` with `D = (1/2)(1,1)(1,1)^T`, responses 81/65 vs 79/63, separated by 32/4095) in the four ports: equal initial
  response, different susceptibility, the same mixing correction separates them.
- Retained learned state Z per world plus charged world IDs, event custody (named events, so retraction is by event, not
  by subtracting a supplied matrix) and paid historical checkpoints. Horizon: at most two insertions and one named retraction.
- Constructor sees the declared numeric world once; updater sees Z and an explicit structured action; reader sees the
  updated Z and a late-selected world/version query. No pretrained-resource reuse claim.

## Arms and controls (identical episodes, same source/action/history/precision permissions)
1. Learned carrier (constructor, updater, reader).
2. Exact resolvent/Schur state engine using the efficient formulas, not a deliberately slow solve (the control that can tie).
3. Reconstruction from the retained rational source (charged).
4. Response-only state (S, F without G) -- must fail on the equal-response worlds.
Controls on the learned carrier: no-op action, wrong-world/wrong-event action, whole-state swap (swap the complete declared
world state, not mismatched custody), named-event retraction, unrelated-world read, paid historical checkpoint read.

## Positive-control smoke (before any fitting)
Exact-rational agreement between the directly reconstructed Schur response and the resolvent response; witness separation
(81/65 vs 79/63); singular and no-op corrections; legal-grid validation (`I/3` rejected on a dyadic grid); wrong-event refusal with
state unchanged; unrelated and historical preservation. Reuse `mat_inv`, matrix operations and Fraction parsing from
`experiments/analysis/correction_capsule_exact.py` (lines ~32-114); leave its fixed historical fixtures and JSON unchanged.
New event/history semantics need their own checks, because they extend the source contract.

## Quality target and outputs
PROPOSED: absolute response error <= 1/4096 on held-out registered traces, which separates the mixing witness; report
numerical error and witness separation separately, plus quality and economics separately (timing of the optimized exact
engine versus learned inference, identical batch regime, including setup and history cost). Output directory
`experiments/results/analysis/quadratic_continuation_cpu_v1/<attempt-id>/` with the reloadable state artifact, predictions,
exact targets, failures, timing and resource receipts. Canonical runner: one new file under `experiments/memory/` or
`experiments/analysis/` with a literal JSON config; no framework.

## Bounds and stops
CPU only, <= 4 threads, <= 30 minutes including data construction and fitting; one declared recipe and seed initially; no
threshold relaxation, no fit search. STOP on any exact-control disagreement, custody violation, nonfinite learned state, any
attempted CUDA or model download, or 30 minutes elapsed.

## Realistic in one hour, and ownership
Realistic: the exact event/history wrapper, the exact-control smoke, a serializable learned-state skeleton, possibly one bounded
fit. Not realistic: assured learned quality, a general quadratic-world claim, pretrained reuse or an economic result.
Implementation needs a SEPARATE worker from Z Code (who owns E1 Z1-Z6); if none is assigned, Z Code finishes the E1 CPU
boundary and persistent snapshot first and then does the exact-control smoke here. The supervisor compares the two tracks'
returned persistent artifacts and actual construction effort after one hour (AGENTS.md distance-from-claim: this entry is
distance 0 on a different carrier).

## Outcomes
- Exact controls pass, learned carrier meets the target and beats the response-only state, with swap/no-op/wrong-action controls
  behaving: a scoped learned-continuation result for this world family only; next is a harder correction family and the
  strong-comparator cost comparison.
- Exact controls pass, learned carrier misses: one targeted remedy (constructor/addressing or objective), then preserve the negative.
- Exact engine ties or wins on cost and quality: record it; identify the harder operation or extraction reuse that could justify the
  learned carrier. The mission is unchanged.
- Exact controls fail: repair event/history semantics before learning.
