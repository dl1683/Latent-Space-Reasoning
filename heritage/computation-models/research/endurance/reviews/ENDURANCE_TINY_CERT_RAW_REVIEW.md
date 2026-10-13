**CERT-RAW-PASS — CERT-CANNOT-ONLY**, limited to attempt `4eb62907-a878-42e6-83fa-a1f8ee195e01` and the frozen **N4/k2/L2 stationary endpoint relaxation**.

Both H6 and H5 are independently checked UNSAT. By prefix restriction, the supported finite ceiling is **H ≤ 4**. No optimum or achievability at H4 is established.

| Horizon | Variables | Clauses | Records / edges / queries | Validated additions | Ignored deletions |
|---|---:|---:|---:|---:|---:|
| H6 | 18,864 | 259,824 | 508 / 504 / 1,016 | 65,169 | 42,872 |
| H5 | 9,648 | 130,544 | 252 / 248 / 504 | 1,316,067 | 1,236,869 |

I processed **all 108,041 H6 proof lines and all 2,552,936 H5 proof lines** against the original DIMACS bytes using different-backend **MiniSat22 unit propagation only**. Search methods and native MiniSat22 search binders were disabled; **zero search calls** occurred.

Neither original formula had a root conflict. First addition-induced root conflicts occurred at H6 addition **65,168**, line **108,040**, and H5 addition **1,316,065**, line **2,552,934**. Every addition passed RUP before retention, including subsequent additions justified by established root inconsistency. Required empty conclusions were validated at the final additions **65,169** and **1,316,067**.

Both formulas and complete mappings regenerated **byte-for-byte in memory** from frozen code. Coverage includes every raw input and FLIP word, separate history wear, paid initialization, terminal decoding, arbitrary nonlinear stationary `I/T/d`, full Tseitin equivalences, and exact unary wear thresholds excluding overflow. No SAT case exists in this attempt, so no witness replay was applicable.

The [root exit record](/<LOCAL_PROJECTS>/new-computation-model/results/endurance_tiny_cert_4eb62907-a878-42e6-83fa-a1f8ee195e01/root_exit.json) records observed exit **0**, chunk `f9b12b`, launcher session `47696`. Result, receipt, environment, attempt identity, and hashes were consistent before execution and at final recheck.

- **Registered experiment:** 2,302.4657716 seconds; peak aggregate job memory **2,674,728,960 bytes**; frozen caps four CPUs, 8 GiB, 3,600 seconds.
- **Fresh certificate review:** H6 **8.2476565 seconds**, H5 **1,511.4858015 seconds**; total **1,519.7471100 seconds**; peak review job memory **677,523,456 bytes**; CPUs 0–3; review process exit **0**.

Seven source files and the protocol matched snapshots, current files, launch commit `245721479e0358e0c632d052ac9b8b86dcbe57ed`, and code-review hashes. Final HEAD was `cf907c2d8c8e720a3a273a856cbf29ecd8b05528`; later provenance did not alter frozen source identity. PySAT **1.9.dev7**, Python **3.13.7**, platform, native extension, and wrapper identities matched the environment record.

Verified SHA-256 bindings:

```text
H6 original CNF
e977ae1e7caa4f5f8837377acd2ac9645a75cd893cfcd65641b1658adf5173a9
H6 mapping
5588dc4574977ec2b685e441cfcf2bb8d797715e3bfac3c201298fe95049e970
H6 original proof — 23,148,563 bytes
96f88cb79ab09e7eca6eb982092b315d097565815d21af9a1ef23425c469504c
H5 original CNF
8c7f52b70c57b1422cf5e5e645ff6f625d347a779604f546a2e21ab63158447d
H5 mapping
5e576f545b1359460a0758f76c3716e6b3f92f5b9b4470580bd1b10292417e1c
H5 original proof — 595,175,368 bytes
9eef442dd4896289aac1ab02f49f84aa5c957c61c32cf341bd26c25c529fafb0

result.json
1eafae38ca99dcfbbf8cc6541dcffa4ce5ea88e6a8a37bdc5931892225c7e5c5
receipt.json
7df85722c08ada818eb3813a8d9b5c926d949b8ccff6d50b416ca81091e7339d
environment.json
49de8c297e6727a5fea776326fa0eac2c69c260810a08655dde96516beed8717
root_exit.json
b053798d3d5f5046eb5b2e43bdd3df6539ec78a0e1036fdcc50f47f83c64b4a8

experiments/run_endurance_tiny_cert.py
868e7155a70c7525522b4ff75a5b423ac51f01a7b3d814d1aede26069e8723c1
experiments/endurance_tiny_cert.py
19a94bdc4e8eeadbe479b3a510cd7d56594e94f79b47fd2145bde249b821e569
experiments/endurance_tiny_cnf.py
312e4927833150aeb1b43655f10bc23eb07f8b1fb8f06dac8c78cce887377ff5
experiments/check_endurance_drup.py
7efee3295addb6d6e72826c96d73daa7582eead3a9bb834f47e33084030d07ae
experiments/run_endurance_tiny.py
59974ac0015a71cc40bb754469bb4d5c401386772b6c8b470f0a80c4b13df10f
experiments/validate_endurance_tiny.py
4f4ab3f2e4771d0bd848eaa0b870f48c2db8dfe9497587148848ea5ad894e7bb
experiments/endurance_tiny.py
1ffdecfcb1b9de5c68a092a925e9f5d185358662df0d67dff8fad9971188e21e
research/ENDURANCE_TINY_CERT_PROTOCOL.md
4e622f4e89ae706cc6cd84acd1d3962d86fdf4f7212a172f470dc2c715c604aa

installed pysolvers.cp313-win_amd64.pyd
1019bacdbb9400cc54fa89aa39294fefe1c63d5a67fdab35f473364529ec72dd
installed pysat/solvers.py
253654d8efabae650a0d136ad2f2e6d30b57206b1fb70846c714197468a28f7e
```

Actual reading coverage: full AGENTS and dual-loop skill; prescribed startup surfaces/sections; complete CERT protocol and exact design/code reviews; seven source files and matching snapshots; environment, both case results, aggregate result, receipt, root exit, empty worker log; complete mappings, DIMACS, and proof streams. Relevant installed wrappers were inspected. Both gzip archives also passed complete decompression/hash verification.

This is **independent software certificate verification**, with native C++ and hardware correctness trusted. No GPU, agents, writes, Glucose3 rerun, or old SMT FULL occurred. It grants no paid indexed-machine CAN, scalable law, lifetime-loss theorem, novelty, primary delta, mission completion, or whole-repository terminal PASS.