**TINY-CODE-PASS — final reviewed revisions only.** This passes the endpoint code gate, with no raw-evidence, physical-policy, compiler CAN, full-repository, or source-priority admission.

The validator and launcher changed externally during this read-only review. I reread and reprobed the final versions; the earlier assertion and wall-accounting defects were repaired.

- **Formula:** finite bitvector Array domains admit every arbitrary decoder `d[16]`, initializer `I[4]`, and transition table `T[2][16]`. No preferred initialization, linearity, symmetry, or single-cell restriction. Integer wear remains separate for every history, including histories sharing physical states. Coordinate `j` is integer bit `j`; initialization charges its set bits, transitions charge physical XOR, and command `i` flips logical bit `i`. Decoding at all **508 records** represents **1016 bit queries**, with **504 update edges** and no additional obligation after H.
- **Validator:** independently restarts each concrete word from its raw input and checks initialization, every prefix’s decoder/wear, and both endpoint queries. It imports no Z3. Final checks survive optimization, reject malformed tables, and the CLI enforces the frozen FULL parameters.
- **Resources:** live Windows queries confirmed structure sizes **64/48/144 bytes**, flags `0x2218`, affinity mask `15`, two active processes, and **8,589,934,592 bytes aggregate job memory**. The child belonged to that exact job and inherited the four-CPU mask. Handle prototypes avoid pointer truncation; default integer returns are compatible with these BOOL APIs.
- **Launcher:** one FULL configuration uses seed 0, four Z3 threads, soft memory limit 7000 MB, and timeout `(remaining−5)s`. A watchdog starts before preparation and covers publication. Unique attempts, source snapshots/hashes, git/environment, formula, logs, result hashes, and child-exit receipts are persisted on normal completion. SAT remains pending; UNKNOWN, nonzero worker exits, caps, errors, and uncertified UNSAT produce VOID.

**Failure-path finding:** receipt/latest publication failures can leave a pending-SAT receipt or `RUNNING` pointer while the launcher exits nonzero. Therefore the **root-observed launcher exit must govern raw admission**: any nonzero root exit is VOID regardless of intermediate artifacts. Receipt elapsed is sampled after hashing; publication is covered by the watchdog. Solver build/check timing excludes launcher preparation, startup, and later output work.

Exact probes completed:

- Four frozen controls rerun: SAT/UNSAT/SAT/UNSAT; both stored SAT replay reports matched.
- **256** fixed-policy SMT/replay comparisons: every N1k1 table combination at H=0,1,2,3.
- **24** N2k2 relabelings accepted, including **18 with `I₀≠0`**; one nonlinear N3k1 policy with `I=[7,4]` accepted.
- Concrete **N4k2L7H6** control replayed 508 records/504 edges/1016 queries; eight invalid-witness mutations rejected. **No FULL solver check occurred.**
- Thirteen mocked launcher failure paths checked; eleven completed as VOID/exit 2, and two publication failures raised nonzero errors.
- Actual short watchdog/job-close control exited **124 in 0.294 seconds**, closing the inherited child pipe before its three-second sleep completed.

Final reviewed SHA256 bindings at git `268a2cf64189b30be46b602a80e9fedb907a7c54`:

```text
endurance_tiny.py          1ffdecfcb1b9de5c68a092a925e9f5d185358662df0d67dff8fad9971188e21e
validate_endurance_tiny.py 4f4ab3f2e4771d0bd848eaa0b870f48c2db8dfe9497587148848ea5ad894e7bb
run_endurance_tiny.py      7ffc31cc9a711913c0e4f190a35ff119601e9f508126d19b8122186c960322e1
ENDURANCE_TINY_6_PROTOCOL  e4b63a3b7d94ac8bae649f24aae493d854f5d224c9eadc4505a80a4c17685e08
```

No files changed by me; no GPU, agents, or FULL run. All actual-machine and compiler CAN claims remain false.