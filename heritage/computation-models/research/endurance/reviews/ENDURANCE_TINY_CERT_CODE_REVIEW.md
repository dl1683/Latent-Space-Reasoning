**CERT-CODE-PASS** at HEAD `47925b40ec3d5ad3013e0b05825a8d33273c3128`, for the exact bytes below. No FULL N4 search ran.

A negative-DIMACS-dimension defect found during this review was repaired externally. I fully reread and reprobed the revised checker and worker, including the new UCRT-only flush and `finally` cleanup. The pass applies to those repaired bytes.

- **Formula review — PASS at registered scope.** Full Boolean equivalences, onehot selection, exact wear thresholds, initialization cost, overflow exclusion, separate hidden histories, stationary shared policies, and terminal decoding are consistent with the endpoint relaxation.
- **EMPIRICAL controls — PASS.** Entire registered smoke executed with memory-backed artifact paths and anonymous proof pipes: six ordinary cases, 256 fixed-policy comparisons, 24 Boolean truth-table cases. Additional checks covered 32 wear assignments, 12 onehot controls, concrete replay agreement, nonpreferred initialization and overflow.
- **Native proof handling — PASS.** Fresh Glucose3 returned `[]` before flushing and nine lines afterward. The installed extension’s loaded `fflush` import exactly matches UCRT. Normal and injected-error cleanup exited zero, including optimized Python.
- **Independent checker — PASS.** Both saved proofs and fresh smoke proofs validated with MiniSat22 search methods disabled. Checks covered malformed/non-RUP/empty/truncated proofs, identity mismatch, root units, original empty clauses, deletion retention and failed-assumption restoration. Forced empty-proof fallback on a root-consistent formula was rejected.
- **Resources/launcher — PASS within tested scope.** Live job queries confirmed CPUs 0–3, aggregate 8 GiB and two active processes; a third process was rejected. Watchdog exit 124 terminated its inherited child. Fourteen launcher scenarios under optimized Python used a mocked filesystem/process boundary; failures included preparation, timeout, memory, UUID, snapshots and publication. Nonzero root exit overrides any pending positive receipt.
- **Artifact binding — PASS.** All six saved CNFs/mappings matched regeneration; four SAT policies replayed and both UNSAT certificates revalidated. `.gitattributes` reports `text: unset` for frozen artifacts; reviewed bytes match HEAD. The smoke receipt binds current sources and records ordinary and exception-cleanup root exits **0**.

N4 generation only reproduced H6 **18,864 variables / 259,824 clauses**, with **508/504/1016** records/edges/queries; H5 **9,648 / 130,544**, with **252/248/504**. Source order remains fixed H6 then H5.

Exact SHA-256 bindings:

```text
experiments/endurance_tiny_cnf.py
312e4927833150aeb1b43655f10bc23eb07f8b1fb8f06dac8c78cce887377ff5
experiments/check_endurance_drup.py
7efee3295addb6d6e72826c96d73daa7582eead3a9bb834f47e33084030d07ae
experiments/endurance_tiny_cert.py
19a94bdc4e8eeadbe479b3a510cd7d56594e94f79b47fd2145bde249b821e569
experiments/run_endurance_tiny_cert.py
868e7155a70c7525522b4ff75a5b423ac51f01a7b3d814d1aede26069e8723c1
experiments/run_endurance_tiny.py
59974ac0015a71cc40bb754469bb4d5c401386772b6c8b470f0a80c4b13df10f
experiments/validate_endurance_tiny.py
4f4ab3f2e4771d0bd848eaa0b870f48c2db8dfe9497587148848ea5ad894e7bb
experiments/endurance_tiny.py
1ffdecfcb1b9de5c68a092a925e9f5d185358662df0d67dff8fad9971188e21e
research/ENDURANCE_TINY_CERT_PROTOCOL.md
4e622f4e89ae706cc6cd84acd1d3962d86fdf4f7212a172f470dc2c715c604aa
results/endurance_tiny_cert_smoke/smoke.json
07842ae4a208456aba687884e53eab9560420f25354b42f2231028fa27e2086d
results/endurance_tiny_cert_smoke/root_exit.json
16f2d33388ece782556fff5ce69af1962e535ec59035c1da48ff312460777c77
.gitattributes
f47701d90d51f26fefea9e28c98bde9c7b6994030b87b4c7672221b217601d17
installed pysolvers.cp313-win_amd64.pyd
1019bacdbb9400cc54fa89aa39294fefe1c63d5a67fdab35f473364529ec72dd
installed pysat/solvers.py
253654d8efabae650a0d136ad2f2e6d30b57206b1fb70846c714197468a28f7e
```

Reading coverage: full AGENTS, prescribed startup surfaces/sections and dual-loop skill; CERT protocol/design review/smoke failure/flush repair/code packet/failure review; all seven source files; complete saved smoke CNFs, mappings, proofs, results and root receipt; relevant installed wrappers, exported binders and PE import table.

Regular-file evidence was rechecked; fresh execution substituted memory paths/pipes to preserve read-only operation. No source writes, GPU or agents. Concurrent untracked compiler/microcode files are outside this gate. This grants no FULL result, endpoint optimum, compiled-machine CAN, novelty or mission admission.