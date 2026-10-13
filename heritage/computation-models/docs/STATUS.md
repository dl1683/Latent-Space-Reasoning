# Status (2026-10-07)

**Execution is paused** (Devansh, 2026-10-04). On 2026-10-07 the project rules were
reset (see [../AGENTS.md](../AGENTS.md)): the dual-loop method is retired, the GPU is
available, and there are no fallback consolation tiers. The goal is a new model of
computation with a capability theorem and an impossibility theorem. No verified
new-paradigm pair exists yet. The saved terminal-gate verdict is NO.

## Where things stand

- **Established:** a body of reviewed, scoped theorems about erasure cost, returned
  statistical context and endurance-limited storage ([RESULTS.md](RESULTS.md)). They are
  correct at their stated scopes, and each is either partly occupied by prior work or a
  calibration.
- **Not established:** a new model of computation; a useful-world advantage over the
  standard model; strict primary-literature novelty; any physical heat or energy claim.
  The operational attempt to tie the erasure meter to a real device returned NO-BINDING.
- **Audit view, 2026-10-07 (from the readers' reports, not a gate):** the record shows no
  phase that built a runnable central artifact able to fail the project's central bet.
  The strongest line (erasure) restricts the standard model rather than extending it.
  The live line (endurance) is far from "new model of computation". See
  [LESSONS.md](LESSONS.md).

## The live unit and its one blocker

`ENDURANCE-COMPACT-PROBE-CAN-1` (cold one-box crit-bit parity dictionary in a bit-probe
model). Its persistent algorithm survived independent attack. The ordinary-machine
realization did not. The single named repair is **LOCAL-STATE-LOWERING-AND-RETURN**
(repairs used 0 of 1, across the whole unit):

1. Distinct live locations for QUERY's retained answer and LOOKUP's returned parity,
   preserved across nested calls.
2. Complete argument locations, call clobbers, loop-counter preservation and
   head/selector transitions.
3. Executable final pocket erasure, counter/selector cleanup and serial-head homing.
4. Recount simultaneous space, provisioning and preparation work.
5. Preserve the original submission, log the repair, get a fresh independent proof review.

Frozen target: N = 2k, ceil((log2 N)^2) <= L <= N, q = ceil(log2(NL+2)),
H >= max(L-1, floor(NL/(1024 log2 N))), R_Q <= 2^20 q^3, separate R_F + W_F <= 2^20 q^3,
mutable state S <= 256 q^2, preparation plus INIT <= 2^20 N q^6.

Registered but unrun: ENDURANCE-COMPACT-CONTROL-1 (four small exhaustive tuples and eight
trie configurations; the design review passed, nothing is implemented). Specification:
`research/endurance/theory/ENDURANCE_COMPACT_CONTROL_PROTOCOL.md` and
`research/endurance/reviews/ENDURANCE_COMPACT_CONTROL_DESIGN_REVIEW.md`.

Frozen artifact hashes (SHA-256) at pause:

| Artifact | Hash |
|---|---|
| Compact model | 217c867350cd8d8d5be1820269639d9698cc6f0e5aa675f34242e6f03a1cf00d |
| Compact theory protocol | 8059b17fd88f34d82f7e9d6f61ccd474a94c7a5385c06815c78d80dc5ab108db |
| Original construction | 38619efa3c0568e4f11575333c28fee34912aa8c1cbb6ca4de16d947b7385ec8 |
| Original reference code | f4df03c354829b09e1acf504f2bde601142b8ea716d0f9525e9510d8d87aa645 |
| Sixteen-row construction receipt | c1a69cbfc61af99fa41a23bf5a6d3d451e58d935fff92d61062d77cb9fd3d456 |
| Control protocol | 84080b7baf4dc42885d59974c303e7222a66d49173f40284812af98f6fa2e60c |

## Other open items

- ADAPTIVE: an efficient seed generator surviving exposed addresses; novelty of the exact
  benchmark.
- RECALL: a promptly decodable sparse directory for interruptible sorting.
- CAT: an input-aware recursive inverse; the universal task limit.
- Erasure: a same-promise separation, or an operational meaning for the peak-loss cap.
- Endurance: H = 4 achievability at N=4, k=2, L=2; the conjectured tight-loss bound
  H <= C NL / log2 k; a cheap-command, fully paid controller for the flat log.
- Observer privacy: usefulness delta.
- Bookkeeping: DELTA-TESTS, CAT, RECALL and ADAPTIVE have no rows in
  `experiments/ledger.jsonl`; `experiments/EXPERIMENTS.md` is not in reverse
  chronological order.

## Practical notes for resuming

- Old receipts and a few frozen scripts cite the pre-reorganization flat paths such as
  `research/ENDURANCE_TINY_CERT_PROTOCOL.md`. The scripts (`experiments/run_endurance_tiny*.py`,
  `endurance_compact_reference.py`, `masked_block.py`, `plot_erasure_controls.py`) are
  hash-frozen provenance. Re-running them needs those paths pointed at the new locations
  ([FILE_MAP.md](FILE_MAP.md)); doing so changes the source hash, which needs a new
  registration.
- Large evidence files stay in `results/` (about 950 MB, mostly one DRUP proof and two
  full-run JSONs). They are the receipts behind the admitted finite results.
- The raw DRUP proof streams are git-ignored locally; git stores gzip archives.
- Work stays in this repo. The sibling repo `../matrix-native-math/` is separate.
- The superseded pause handoff is `archive/history/PAUSE_HANDOFF_2026_10_04.md`.

## Update 2026-10-08 (after the roundtable exploration)
- Rules reset on 2026-10-07 (AGENTS.md): dual-loop retired, GPU free, no fallback tiers. A three-engine exploration loop then ran for about a day (records: `ROUNDTABLE/`, summarized in
  `docs/ATTEMPTS.md` Phase 5). The endurance repair was NOT spent: the repair gate failed on evidence.
- Five more lines closed by equal-resource translation (endurance, disturbance lease, restoration-window catalytic space, shared KV-prefix scheduling, proof-complexity lemma-interface throughput). Still no new model of computation and no
  CAN/CANNOT pair that survives translation into a known model.
- Assets that remain useful: a calibrated finite-model generator and replay toolchain (`ROUNDTABLE/work/codex`, `ROUNDTABLE/work/droid`), an independent-monitor cold-command runner
  (`experiments/cold_runner.py`), a checked lease-degree theorem candidate (side result, not promoted), and an engine reliability record (`ROUNDTABLE/FACTCHECK_gen1.md`).
- Open decision for Devansh: change what we search for (see the escalation in `ROUNDTABLE/HEARTBEAT.log` and the end of `ROUNDTABLE/LEASE_LINE_CLOSEOUT.md`). No new exploration round is running.

- Update 2026-10-08 (later): about 160 further pre-registered units (rounds 9-12) ended with no survivor; exact large-scale spectra (n = 3..5 for dozens of models) and 39-configuration scaling checks agree. The only reviewed
  mathematics are side results around the lease-degree theorem and its abelianization generalization (see docs/ATTEMPTS.md Phase 5 addendum); an independent proof review is running. The standing instruction is to keep the exploration loop running
  autonomously (never wait for direction).
