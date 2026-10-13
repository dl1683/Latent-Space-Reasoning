**CODE: REPAIR. FULL remains gated.** The prior lower/upper proof passes remain intact: the constructor’s AST and proof/registration sections 1–6 are unchanged. Only terminal failure diagnostics changed inside `check_case`.

Four findings block CODE PASS:

1. **Failed or interrupted attempts can publish current `CONFIRM`.** At [succinct_erasure.py:352](</<LOCAL_PROJECTS>/new-computation-model/experiments/succinct_erasure.py:352>), publication precedes the final stdout write and process completion. Actual subprocess probes produced:
   - injected `BrokenPipeError`: exit **1**, artifact **CONFIRM**;
   - injected interruption immediately after replacement: exit **3221225786**, artifact **CONFIRM**.
   
   Admission must require an independently observed successful exit, matching attempt identity, source identity and complete frozen case inventory. JSON `CONFIRM` alone is insufficient.

2. **Output confinement has a boundary defect.** [Line 312](</<LOCAL_PROJECTS>/new-computation-model/experiments/succinct_erasure.py:312>) accepts `--output results`. Its temporary path becomes `repo/results.<attempt>.tmp`, outside `results`, before replacement fails. Require a strict descendant file target.

3. **Temporary writes follow links.** [Lines 323–326](</<LOCAL_PROJECTS>/new-computation-model/experiments/succinct_erasure.py:323>) reuse a predictable temporary name after publishing its attempt ID and open it with ordinary truncating writes. A modeled symlink planted after `RUNNING` redirected final output outside `results`. Exclusive temporary creation and protection against link replacement are required. Existing output/directory links resolving outside were rejected.

4. **The fingerprint identifies disk bytes, not necessarily loaded bytes.** [Line 19](</<LOCAL_PROJECTS>/new-computation-model/experiments/succinct_erasure.py:19>) hashes a separate file read after compilation. A modeled compilation/source-read race executed a 24-case snapshot while publishing the reviewed disk hash and `CONFIRM`; all subsequent hash checks passed. Bind execution to an immutable, hashed source snapshot.

The finite CPU probe record is retained here and in the tool transcript; filesystem mutations were mocked.

| Probe | Observed result |
|---|---|
| Successful smoke | 36 rows: **22 CONFIRM, 14 CANNOT**; every saved row and trace reproduced |
| AssertionError at case two | Actual exit **1**; `KILL`, one completed row, first failing case and diagnostic saved; no third case |
| Infrastructure error at case two | Return **1**, `VOID`; no third case |
| Source change before cases / after smoke | `VOID`, return **1** |
| Source read failure | `VOID`, zero cases |
| Case-two KeyboardInterrupt / SystemExit | `RUNNING`; execution stopped |
| Initial invalidation write/replace failure | Zero cases; older `CONFIRM` survives |
| Failure-record write/replace failure | `RUNNING`; failing witness not persisted |
| Final write/replace failure before replacement completes | `RUNNING`; uncaught error |
| Actual `-O`, `-OO`; optimized code objects | Rejected before publication |
| Frozen FULL inventory | 144 unique cases enumerated; **zero FULL circuits executed** |

Replacement ordering was tested through mocked file I/O; live filesystem atomicity and real symlink swaps were not exercised.

SHA-256 fingerprints, stable through the probes:

```text
experiments/succinct_erasure.py
2d2b0c6b9ebc77de056a4af11f4692b8aac29322090afe19743b0356f346cc7e
research/SUCCINCT_ERASURE_CANDIDATE.md
6c936982f886cea9a3e49c5b5e5d85acdba1b524d4a6f4df640789a0de483a1d
research/SUCCINCT_ERASURE_REVIEW_1.md
aa2754fe31b9b3e8bb17c8f60b284aa194bec6bfef91c6cdd7ec9bca27b26895
results/succinct_erasure_smoke.json
682ce58d606d22305c0feb67000ef3ab78b3a75b27aa65e91550949a5a483a72
results/succinct_failure_regression.json
ad8beaa07f0edad3720cdcf9fe18b310f166b72c306d743c0a361065b122a9f4
AGENTS.md
e9b7400be1f4bf550fc46ee4cc0fe495b20d32d7dda5c30d370e1922ee75b6b7
dual-loop/SKILL.md
334a7d272c572b4dbc8c36a5fde15385fa7962df5c234ba8772c199628af0151
```

The registries changed externally during review; the succinct protocol remained unchanged:

```text
EXPERIMENTS.md — start
d26350ce227b9b3f377eb359bde006b5990220fc51cc0ebde9e4cec9c4692585
EXPERIMENTS.md — end
241430b0586af39b4c22c73fc09c293f83cde02347a9dfcba59a8d9d697361b7
experiments/ledger.jsonl — start
b2c0afc08ae3f78cba237ed7e8bc7e82a09a1f3c415119944ca02f9ced7527aa
experiments/ledger.jsonl — end
c940b112f429634b8c3f70e5e8525138c834788498b73a748190cec993890c85
```

**Alternate interpretation:** completed mathematical checks could remain valid despite a later reporting failure. That does not satisfy the stipulated attempt-admission rule; failed and interrupted probes are inadmissible as current CONFIRM.

Ordinary case-failure handling is repaired; publication and provenance still need repair. This is a process component, with no priority or CS/math moonshot conclusion. No edits, additional agents, GPU use, or FULL run occurred.