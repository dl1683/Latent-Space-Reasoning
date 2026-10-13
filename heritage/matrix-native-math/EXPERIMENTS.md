# Experiments index

## 2026-09-29 Record corrections 2 (re-measurements of three mislabelled wins; beyond_bilinear_v1 close-out)

* Fresh-eyes review findings verified by single-thread re-measurement (`code/record_corrections_2_checks.py`, raw
  `results/record_corrections_2/`): E11 the workload pipeline's 2.8-14x is a win over a dense GEMM chain only
  (np.fft.fft2 is 56-620x faster than its compiled kernel; the structure is domain-obvious); E12 the Wedderburn
  emitter's 1.9-5x is likewise over a dense chain (FFT + batched matmul is 2.8-3.3x faster than the emitter);
  E13 the labeling withdrawal's "joint diagonalization dominates" compared an O(n^3) GEMM materialisation with an
  O(n^2) decode (single-thread crossover n ~ 150); E14 AGENTS.md 7.1 exponent-only wording. Details: Errata E11-E14 in
  `RESEARCH_MONOGRAPH.md`.
* `beyond_bilinear_v1` closed-partial (13 of 16 rows; 7 of 13 violate the 5% native-drift rule; the vendor cuBLAS
  emulation is a control; own Ozaki-I is slower than it in every matched row), superseded by `ozaki_fair_engine_v1`.
  `stratum_rational_v1` parked with a status note (moduli count 141 + 6, not 141 + 2).
* Ledger rows `record-corrections-2-*`, `beyond-bilinear-v1-closeout`, `stratum-rational-v1-parked`.

## 2026-09-28 Record corrections after outside audit (no new experiment; regression tests added)

* Outside audit of the monograph (`research/external_review_2026_09/`) plus in-repo
  checks corrected: r_pf = 49 does not imply rank(MM_4) = 48; registry theorem's
  storage claim false (affine counterexample); DPS48 slowdown is 65% not 39%; the
  DPS48 960-ops/node obstruction was an implementation artifact (transforms applied
  composed per node instead of once). Details: Errata in `RESEARCH_MONOGRAPH.md`.
* New regression tests: `code/tests/test_errata_2026_09_28.py` (5 pass): affine registry
  counterexample, Haar-mean identity E S = 2||Q-Phi(Q)||^2 with sup S <= 2 E S on a
  finite group, bounded-spread aperiodic swap chain generating S_7, contraction
  family with spread <= 1 generating M_n, product-form baselines.
* Cleanup: deleted dead code (`dps48_race.py`, `approx_matmul.py`, `structure_detector.py`,
  `product_form_search.py`) and duplicate manuscript files; no evidence receipts removed.
* Ledger row `record-corrections-2026-09-28`.

## 2026-09-18..19 Theorem arc (indexed retroactively; ledger rows 1035-1232)

* Walk dichotomy (transfer chain, mixing side, junction), counting theorem, truncation
  closure dichotomy, separable-schemes theorem, 27/23/4 alignment classification,
  product-form baselines (`code/product_form_baselines.py`, r_pf <= 49 certified),
  labeling recovery (Middle candidate withdrawn; reason corrected, E13), market probes (2/14 robust),
  workload pipeline v2 [vs dense chain only, E11], Wedderburn instrument [vs dense chain only, E12],
  DPS48 race (PRACTICAL_NEGATIVE for the
  per-node-composed implementation; see erratum). Verification scripts:
  `code/mixing_side_checks.py`, `code/periodic_chain_checks.py`,
  `code/n150_prediction_run.py` (n=150: 559,895 vs predicted 562,475, 99.54%, 8 reps,
  no error bar). Rigor labels and receipts live in the ledger and the units' docs.

## 2026-09-06 Approximate decomposition with structured correction (reopen probe)

* Question: can T = S + E with rank(S) < rank(T) leave a residual E that is
  cheaper than multiplications?  In the bilinear (block-recursive) model the
  answer is no by subadditivity (proved locally); for scalar commutative entries
  the 7-multiplication floor for 2x2 is Winograd 1971 (cited).  The run then
  quantified it: best-found upper bounds on d(T_2, rank<=r), 512 restarts, no
  global certificate: sqrt7, sqrt6, 2.23477, 1.89362*, sqrt2*, 1, 0 for r = 1..7
  (* = reached only along diverging factors; r=1,2 are the analytic values).
  The best-found r=6 point is attained and its residual is numerically rank 1;
  separately, T_2 minus the monomial a11*b11->c11 has an exact integer rank-6
  decomposition, so the best-case correction is one product: 6+1 = Strassen.
  The r=5 optimum is Bini's border-rank-5 tensor plus its two missing terms.
* T_3 probe (ALS only, 240 restarts): budget-limited upper bounds, nothing
  beyond the literature (border rank <= 20, cited not verified).
* Code: `code/experiments/approx_decomp.py`; evidence `results/approx_decomp_v1/`;
  ledger row `approx-decomp-v1-result`.  Negative result under every model
  considered; Codex evidence gate applied (reporting relabelled, no code defects).

## 2026-08-05 DPS48 exact circuits V6

* Exact finite-class circuit MILPs over union DAGs plus every useful signed
  first-level augmentation certified U/V/W optima 52/44/76; no improvement
  on L=172. Generated/verified circuits: W 4,096/4,096, U 4,096/4,096, and
  optional V 8,192/8,192.
* Exact quotient enumeration proved the fixed W outside-span minimum is 12,
  globally excluding new rank directions at weights 7 through 11.
* Canonical report: research/DPS48_EXACT_CIRCUITS_V6.md. Machine evidence is
  under results/dps48_exact_circuits_v6/; ledger row dps48-exact-circuits-v6-result.

## 2026-08-04 DPS48 isotropy V3

* All six exact slot variants, structured-isotropy cancellation, and
  20,000-restart circuit saturation left the authenticated best at L=174.
* Canonical report: research/DPS48_ISOTROPY_V3.md. Machine evidence is under
  results/dps48_isotropy_v3/; ledger row dps48-isotropy-v3-result.

## 2026-07-20 resumption

* B393: geometric program foundations; certificates in
  `research/verify_b393_geometry.py` (no benchmark claims). Ledger row
  `b393-unit1-computational-geometry-foundations`.

# Experiments at closeout

The complete chronological ledger is [`experiments/ledger.jsonl`](experiments/ledger.jsonl).
This file is only the current evidence index; it is not a second ledger.

## Current practical captures

* B382/B383: diagonal and real Walsh dense-output controls; see
  `research/CURRENT_EVIDENCE.md`.
* B384/B389: block-cyclic complex and real controls; B389 is the strongest
  current CPU result.
* B387/B388: channel controls with pinned/default and matched-specialist data.
* B392: exact Clifford candidate with a negative CPU dense-interface gate.

## Re-run commands

See `README.md` for the focused test command. Benchmark scripts live under
`research/` and write JSON captures under `results/`; do not overwrite a
canonical capture without adding a new dated ledger row.

## 2026-07-23 B439 exact geometry captures

* B439 collision census: `research/verify_b439_collision_protection.py`
  writes `results/b439_collision_protection.json`; exact 729-Brent replay
  for 15 schemes and symbolic HKS zero collision ideals.
* B439 pole table: `research/verify_b439_poles.py` writes
  `results/b439_poles.json`; all 69 factor valuations per curve, exact
  leading vectors, complete Laurent identities, and principal relation
  ranks/supports.

## 2026-07-23 B440 boundary and surgery capture

* `research/verify_b440_pole_census.py` writes
  `results/b440_pole_census.json`: six exact labeled pole stacks, all eight
  `621 x 27` ranks, collision/circuit/Schmidt data, regular and 36 shuffled
  controls, exact symmetry quotienting, bounded diagonal-gauge feasibility,
  and curve-2/curve-4 order-by-order scalar-lift obstruction certificates.

## 2026-07-23 B442 pressure-targeted motif search

* `research/run_b442_motif_search.py` ran 20 exact paired guided/vanilla seeds
  for 1,489,700 moves per arm at the KMW rank-153 anchor, CPU only.
* Terminal tokens: `NO_STRICT_FUSION_WITHIN_BUDGET` and
  `INCONCLUSIVE_LOW_EVENT_RATE`. Both arms produced zero target events; no
  direct base improvement or square-certificate candidate was found.
* Canonical capture: `results/b442_motif_search/final.json`; explicit 8h and
  20h checkpoints live in the same directory. Conservative compute was
  28.0116 core-hours and the 4-hour certification pool was unused.

## 2026-07-26 U23 border-obstruction V1 pilot

* `research/verify_u23_border_obstruction_v1_pilot.py` writes
  `results/u23_border_obstruction_v1_pilot.json`. It replays the registered
  good fibre, constructs concrete compressed A0 charts and all five
  weightwise contraction blocks, and emits candidate-only chart-family
  outcomes.
* Three cheapest families candidate-closed by exact modular unit identities;
  the global cover remains open and no border-rank token fired. The frozen
  schedule and external-only disposition are in
  `research/U23_BORDER_OBSTRUCTION_EXECUTION_PLAN.md`.

## 2026-07-27 R10 transport arithmetic fast restart

- `research/verify_r10_transport_arithmetic.py` executed the owner-frozen P2 without re-running the path selector and without enumerating the 55,296-chain witness Cartesian product.
- P2 stage: `R10_T2_P2_FROZEN_CHAIN_EXACT_Q_LIFT_EXISTS_CANDIDATE`; one deterministic chain passes mod 4/8/16 and exact-Q rank-7 replay.
- T2 geometry: common L is eight dimension-9 cells, 3,648 points, full 24-dimensional hull. ZERO is `ZERO_BASEPOINT_BRANCHING_DIAGNOSTIC` with no token. Complete LIVE_DOMAIN H11/H12/H22 emit `R10_T2_LIVE_DOMAIN_BRANCHING_CANDIDATE` for the two named routes.
- Prefix census emits `R10_T2_P1_P2_PREFIX_LIVE_GEOMETRY_PROFILE_COMPLETE_CANDIDATE`.
- Optional T4 arm: `KILL_R10_TA_RELATION_REPLAY_SOUNDNESS`; exact state/fiber/frame/direct-elimination controls pass, but the composed source projection is a strict subset of D8. No T4 candidate fires and there is no reduction coverage.
- Final candidate packet: `results/r10_transport_arithmetic.json`, SHA-256 `822ad5abf0edefc39fc5d199c0669cdeb375120bfd9ee69c6f90072ee0b72d4f`. Cumulative charged CPU is 5.956905382 core-hours including the prior 5.9-hour implementation failure; peak RSS is 72.88 MiB; GPU untouched. The six-core-hour amendment was not invoked.
## 2026-07-27 U23 conormal numerical-discovery pilot

- `results/spitball_probes/u23_conormal_pilot_exploratory.py` writes the candidate-only exact row packet `u23_conormal_pilot_exploratory.json`.
- The complete ordinary-Koszul sigma_23 differential supply has exact fibre rank zero; eight reconstructed exact branch normals yield 0/8 provenance passes and token `KILL_U23_CONORMAL_EXISTING_SUPPLY`.
- `results/spitball_probes/u23_conormal_pilot_audit_exploratory.py` independently replays tensor contractions, row independence, the six-class supply census, and the circuit unsaturation certificates.
- Scope: source kill only; no smoothness or border-rank inference.

## 2026-07-27 T4 registered-diamond radius-three reduction preflight

- `research/preflight_t4_radius3_reductions.py` exhaustively enumerated one legal
  outgoing F2 flip from each of the 1,008 registered eligible radius-two diamond
  endpoints, then tested complete `FlipEngine` reductions on every child.
- Exact scoped result: 562,464 raw arcs, 557,736 canonical unique children, and
  zero raw or unique-child reduction records. This does not cover all 164,736
  radius-two states, characteristic zero, plus seeds, or deeper/mixed paths.
- The fresh read-only evidence audit passed. Result:
  `results/t4_radius3_reduction_preflight.json`; review:
  `research/REVIEW_T4_RADIUS3_PREFLIGHT.md`.
- This was an unregistered exploratory preflight. It emits no scientific token
  and opens no dependency. The tested T4 targeting predicate remains empty.
- Compute: 28.78125 process CPU-seconds, 28.7964242 wall-seconds, peak RSS
  329,834,496 bytes, one worker, BLAS one, GPU untouched.

## 2026-07-27 T4 global plus-free radius-three premise

- `research/run_t4_global_r3_frontier.py` exhaustively reconstructed all 164,736 canonical exact-distance-two schoolbook-T4 states and streamed all 90,979,200 legal outgoing third-flip arcs over F2.
- Exact result: zero deterministic `FlipEngine` reduction-positive parents and zero reduction-positive third-flip children. Confirmed classification: `T4_GLOBAL_R3_PLUS_FREE_DESCENT_PREDICATE_EMPTY_COMPLETE`.
- Fresh review independently regenerated the corpus byte-for-byte, replayed all 323,712 stored routes, and independently tested reduction existence with GF(2) span maps rather than `FlipEngine._circuits`.
- Result: `results/t4_global_r3_frontier.json`, SHA-256 `abfb750e48893e48de2e7ec9f826d5436394fbe392e1ad2e382d6ef8a650bb9d`. Review: `research/REVIEW_T4_GLOBAL_R3_ZEROTRUST.md`.
- Compute: 2,457.203125 charged CPU-seconds, peak RSS 272,236,544 bytes, one worker, BLAS one, GPU untouched.
- Scope: F2, plus-free, schoolbook T4, radius at most three only. The result selects the plus-seeded fallback but does not open T4 targeting or support a cheaper-multiplication claim.

## 2026-07-27 T2 global target-pullback oracle candidate

- `research/run_t2_pullback_oracle.py` compiled the complete frozen shortest-DAG relation battery and evaluated all 248 registered path/predicate/level cases, CPU only.
- Exact candidate result: backward pullback, source-retaining compose/intersect, and the independent monolithic joint solve agree in all 248 cases; all 124 mandatory first-edge direct controls agree.
- P1 and P2 reproduce their frozen mod-4/mod-8/mod-16/exact-Q ground truths. P1 is the q7 singleton; P2 is a distinct point in the authenticated 512-point gauge orbit.
- Final artifact: `results/t2_pullback_oracle.json`, SHA-256 `74d59347755085bc58539e37ebebf207bf6d844b36dd42c9cc76820eb88794bc`. Candidate token: `T2_PB_COMPLETE_SHORTEST_DAG_ORACLE_VALIDATED_CANDIDATE`.
- This is pending fresh zero-trust confirmation and opens no T4 dependency. Review boundary: `research/REVIEW_T2_PULLBACK_ORACLE.md`.
- Conservative CPU accounting: 3,118.875 seconds against 4 core-hours; peak RSS 90,087,424 bytes under 2 GiB; one worker, BLAS one, GPU untouched; no formatter.

## 2026-07-27 T4 plus-seeded predicate v1

- `research/run_t4_plus_seeded_predicate_v1.py` implements the frozen complete plus/mandatory-bridge/zero-or-one-extra-flip/zero-to-two-reduction census with exact-Q replay and durable resource bindings.
- Authenticated 64-seed projection: 63.125 CPU-seconds and 63.328964799999085 wall-seconds, projecting 11,930.625 CPU-seconds and 11,969.174347199827 wall-seconds for all 12,096 seeds.
- Both projections exceed the frozen nontransferable 5,400-second Stage B caps. Fresh review approved `VOID_T4_PLUS_CENSUS_RESOURCE` and rejected launching the full census.
- The incomplete subset observed zero successful reduction paths, which supports no population claim. The required confirmed plus-predicate and relation-replay tokens did not fire, so `t4-transport-targeting-v2` remains gated.
- Durable closeout: `results/t4_plus_seeded_predicate_v1/stage_B.json`; review: `research/REVIEW_T4_PLUS_SEEDED_PREDICATE_V1_PRERUN_GO.md`. CPU only; GPU untouched; no formatter.

## 2026-07-27 T4 step-two-wall v1 mandatory-control resource void

- Fresh pre-run review passed for executor SHA-256 `75097a8d8ae7943d519035540a1eb09ee35394a2a5294018bdab1db3e5de199c`.
- Static authentication reproduced exactly 768 distinct frozen rows (512 Stage-B plus 256 Round-6), with row digest `8c6268c948c8575482b511a09c85f754668bdace502623eaa67e13acaf5c1305`.
- The mandatory exact synthetic two-reduction control did not complete before the frozen one-hour cumulative census cap. Verdict: `VOID_STEP2_CENSUS_RESOURCE`.
- The guard fired at 3,600.0646072999953 cumulative charged seconds; the finalized append-only ledger totals 3,600.091330799987 seconds. Control peak RSS was 53,657,600 bytes; one CPU worker, BLAS one, GPU untouched.
- No natural seed outcome was evaluated. The 1,065,420 occurrence count, distinct-state census, radius-0/1 closure, quarantined triage, and theorem/counterexample stage did not run. The void supports no wall, theorem, or counterexample inference.
- Durable summary: `results/t4_step_two_wall_v1.json`, SHA-256 `c2bd82ea6d5062d203e816052f65b9c4414957b99b529d76584e1963147d5f70`; closeout: `research/T4_STEP_TWO_WALL_V1_CLOSEOUT.md`.
## 2026-07-27 B463 two-wall deep exploratory run: instrumentation banked, soundness kill

- Executor/review: `research/run_b463_walls_deep_v1.py` at SHA-256 `706e95faf43a2eb64cde4e701ab660a4aa20842f5a4b84e333a2a0569d75aa40`; fresh pre-run decision GO, zero blockers. Registration SHA-256: `dec10452f50a4b9c2187758d05276eb27f17573d3d13953f00692696d6cea00f`.
- Two parallel CPU-only, single-thread workers ran under 1.5 GiB reservations; GPU untouched. Both exact quotient left-folds remained in the forward third-factor reduction through 3,600 seconds.
- EMPIRICAL, INSTRUMENTATION ONLY: at 900/1,800/3,600 seconds, each target retained basis size 3 and degree histogram `{1: 3}`, with zero completed durable factors and no exposed solver frontier. The 3,600-second records contain two operational-flatline intervals, not basis-flatness exits. M1/M2 peak worker-tree RSS: 474,066,944 / 471,633,920 bytes. M1/M2 charged science: 3,600.258989 / 3,600.299632 seconds.
- Provisional per-wall outcome was `VOID_RESOURCE` with reason `OPERATIONAL_FLATLINE_1800_TO_3600`, but mandatory evidence replay failed because the sealed compiler environment required `psutil 7.1.0` and the replay subprocess resolved `psutil 6.1.1`. Frozen source hashes still matched.
- Final disposition: `KILL_B463_DEEP_SOUNDNESS`; zero scientific tokens. The provisional VOID records must not be cited as wall classifications. The trajectory is retained solely for future resource budgeting.
- Canonical artifact: `results/b463_walls_deep_v1.json`, SHA-256 `397bcda63a9e1ab5fbadd7848bd0876a347ef180e294473c4d9b8f8ea75b22ea`. Emergency token commit SHA-256: `75fe3b7e0268911afb9ab3f61dabdb58595b2b8126b9123eab46b25666adfcb6`.

## t4-step-two-control-repair continuation â€” baseline resource void (2026-07-27)

The minimal repaired positive control passed both production-detector branches: 32 exact radius-0 two-reduction chains and a detector-found radius-1 retuning path in 27.8827316999959 charged seconds. This is synthetic plumbing evidence only.

The frozen natural baseline then emitted VOID_STEP2_CENSUS_RESOURCE at 3600.08805899999 cumulative charged seconds. Its first 64-seed transaction did not seal, so atomic rollback left zero durably evaluated natural seeds. No 1,065,420-occurrence reproduction, distinct-state count, closure result, theorem, counterexample, or wall inference exists from this continuation.
## T4 step-two wall owner-resume implementation gate (2026-07-27)

Operational only: owner-resume amendment and runner passed fresh R2 review after an R1 `NO_GO` repair cycle. Exact 16-seed smoke sealed two 8-seed shards with 1,941 occurrence rows and 1,724 raw canonical states, then passed full representative replay. Peak RSS was 72,925,184 bytes under 2 GiB; GPU untouched. No population inference is permitted. Full baseline and closure remain pending.
## 2026-07-28 KMW chart sweep V1 certified negative atlas

- The immutable ladder comprised 76 chart labels and 40 distinct exact CNFs: 36 wedge-target/base-hull charts, 36 site-local nearest-coordinate `+1/+2` widenings, balanced charts opening 24/30/36 pair-block terms, and one chart opening all pairs plus nine elementary remnants.
- Terminal verdict: `KMW_CHART_SWEEP_V1_ALL_76_F2_UNSAT_CERTIFIED_ATLAS`. CaDiCaL 1.9.5 and Glucose 4.1 independently returned UNSAT on every chart. Every chart has a persisted additions-only RUP ending empty and freshly verified against its rebuilt exact CNF.
- Proof topology: 69 charts have hard constant contradictions; 72 have one-addition root RUPs. The four longer proofs have 542, 109, 109, and 117 additions. A fresh-process audit rebound both solver statuses and accepted all 949 additions across all 76 certificates.
- No SAT chart, Q lift, exact-Q candidate, timeout, resource VOID, or hard chart occurred. The result proves only the exact registered F2 atlas, not unrestricted rank-152 or rational nonexistence.
- Final atlas: `results/kmw_chart_sweep_v1_atlas.json`, SHA-256 `03b7fb98f7248960368217af9769c4cd4f2bbffa924b4c8b1674763c98bfcabd`. Fresh audit: `results/kmw_chart_sweep_v1_atlas_zero_trust.out.json`, SHA-256 `b3f84329abff46faf7753b29fff8f1717e001998d100d5ea22a2cc358ff40af7`. Result note: `research/KMW_CHART_SWEEP_V1_RESULT.md`, SHA-256 `2a6a87ff7285daf814b6ff22735a6c9ebd2216d32c1ef8f20d034b2c39addbd7`.
- Retry invocation wall: 28.006 seconds; sealed chart-wall sum: 23.973 seconds; conservative failed-attempt charge: 120 seconds; total charged sweep wall: 148.006 of 1,800 seconds. Peak aggregate RSS: 98,828,288 bytes under 2 GiB. One CPU worker, BLAS one, GPU untouched, PID 2520 and census paths untouched.

## 2026-07-28 KMW GF3 and fully free F2 moonshot V1

- Final token: `KMW_GF3_AND_MOONSHOT_V1_SEALED_NO_CANDIDATE`. Both scientific stages are VOID; neither stage exhausted its fresh 4 GiB or wall-time cap.
- Why GF(3) was live: the F2 atlas's class-parity/coupled-Hensel obstruction is characteristic-two-specific and does not exist as stated over GF(3). This opened a field-change search, not a prediction of SAT.
- Encoding choice: exact sparse GF(3) expansion around the KMW anchor, Gaussian row reduction over GF(3) applied to the full linear-plus-nonlinear polynomial rows, then two-rail mod-3 gadgets for every retained residual and product definition.
- The GF(3) planted 18-site control passed Glucose and CaDiCaL, all 924,036 clauses, and selected/full replay of all 46,656 coordinates modulo 3. Scientific 18-site measured 148,776 Boolean variables, 924,036 clauses, and 21,461,624 DIMACS bytes.
- Scientific 18-site returned two UNSAT statuses, but the required proof did not exist: Glucose's trace was empty, Lingeling's was malformed, and the fallback process exited `0xC0000409`. Verdict remains VOID. The all-36 GF(3) chart was not opened.
- The fully free F2 encoder completed exactly at 3,221,748 variables, 11,286,864 clauses, and 256,314,596 DIMACS bytes. PySAT Kissat404 then exited `0xC0000005` before a solver status. Conflicts, decisions, propagations, and restarts are unavailable and recorded as null.
- Stage 1 peak/job wall: 299,950,080 bytes / 13.171 seconds. Stage 2 peak/job wall: 1,570,123,776 bytes / 12.173 seconds. CPU only, one worker, BLAS one, GPU untouched, PID 2520 untouched.
- No finite-field SAT, p=3 lift, exact-Q replay, rank-at-most-152 candidate, candidate zero-trust certificate, or disclosure path opened.
- Result: `research/KMW_GF3_AND_MOONSHOT_V1_RESULT.md`, SHA-256 `de2d5cf510e72b61ad2a8112a6e5808b5893bba99230bd67c45ffc1fc323749d`. Final seal: `results/kmw_gf3_and_moonshot_v1/final.seal.out.json`, SHA-256 `e46e21f54f5b94560f90ce8702a74b0fd6179cc3776daa0b5aef2286d1eb7c76`.
- Closeout integrity audit: `results/kmw_gf3_and_moonshot_v1_closeout_integrity.out.json`, PASS, SHA-256 `baa28b925001cd97c9420e27bb62a9fb978b727f772e22adb0c3be8350c21a7a`.

## 2026-07-28 KMW GF3 and fully free F2 moonshot V2

- Final token: `KMW_GF3_AND_MOONSHOT_V2_SEALED_NO_CANDIDATE`. Stage 1 is implementation VOID; Stage 2A is probe VOID; Stage 2B did not open.
- Terminal r2 GO bound runner SHA-256 `91f52c420790b91c2efc381c05a5179a22f649b892f7004510f0ea25eae657f4`, but the Stage 1 result-record path then read absent registration key `stage_1.encoding.choice`. The worker exited before any GF3 solver; neither scientific chart has a status or certificate.
- The retained 3,221,748-variable / 11,286,864-clause / 256,314,596-byte DIMACS remained SHA-256 `eb2e7451b14c2cb69d4b58398093cd54e7f7617a827c3656f499ecb2548fdd53`. CaDiCaL loaded all clauses in 13.726 seconds.
- CaDiCaL did not return from the timed interrupt before the exact 600-second Job close. No solver result or search-stat snapshot survived; probe PASS failed. Peak Job memory was 3,831,132,160 bytes under 4 GiB. The three-hour attempt remained closed.
- No SAT, UNSAT certificate, GF3 `3 -> 9 -> 27` lift, F2 lift, exact-Q candidate, global stop, zero-trust candidate, or disclosure path opened. The parity-dodge axis remains unresolved.
- Result: `research/KMW_GF3_AND_MOONSHOT_V2_RESULT.md`, SHA-256 `b6bebf522435965f532415f6d8ee03f07ca65980249e137ad7132fd130c5fadc`. Final seal: `results/kmw_gf3_and_moonshot_v2/final.seal.out.json`, SHA-256 `888e483688eea965f7050f840e3ec011dd9cb35f3cd62b8a102d0eec3829446f`.
- Closeout integrity audit: `results/kmw_gf3_and_moonshot_v2_closeout_integrity.out.json`, PASS, SHA-256 `bef6d850fe0dd3323af4bb1ac352000648346b05c98a2a96641d976ffd4efebc`.
## KMW_JETS_CHARTS_V1 â€” sealed 2026-07-29

The planted F2 dual-number control passed. All seven amended soft-orbit wound charts (`61,63,65,69,73,75,77`) are CaDiCaL+Glucose UNSAT with fresh RUP certificates checked against the persisted DIMACS. Canonical result: `research/KMW_JETS_CHARTS_V1_RESULT.md`; evidence root: `results/kmw_jets_charts_v1/`; final token: `KMW_JETS_CHARTS_V1_ALL_SOFT_ORBIT_UNSAT_CERTIFIED`. Exact scope only: no global rank or border-rank inference.

## 2026-07-29 KMW moonshot portfolio V1

- Exact retained GF3 18-site pair-block chart: `PROVED` UNSAT. CaDiCaL and Glucose agreed; fresh replay accepted all 2,754 additions in the persisted CaDiCaL RUP trace. Certified seal SHA-256 `f1c7432d7f537150fff50234b4cd6366412090d954ba18b4f58fcf01fb97155b`.
- Exact generated GF3 all-36 chart: resource `VOID` after the 5,433,486,162-byte DIMACS completed and solver load crossed the 4 GiB Job boundary; no solver status exists.
- Fully-free F2 portfolio: resource `VOID`, not UNSAT. CaDiCaL default/SAT-phase/UNSAT-phase and Glucose each loaded all 11,286,864 clauses and exhausted independent 5,400-second slices, charging exactly 21,600 seconds without a verdict.
- Terminal token: `KMW_MOONSHOT_PORTFOLIO_V1_SEALED_NO_CANDIDATE`. No F2 model/proof, lift, exact-Q candidate, lower bound, zero-trust candidate, or disclosure artifact exists.
- Canonical result: `research/KMW_MOONSHOT_PORTFOLIO_V1_RESULT.md`, SHA-256 `ad0dd0822b6fbad177fbcc2a8ff60857fae746dd3d4ba4d18083e8cf9c206bae`; Stage 2/final seals SHA-256 `4330a025...` / `7e73e30d...`; closeout integrity PASS SHA-256 `a3355334...`.

## 2026-08-03 M4 symmetric flip V1

- Exact anchor and implementation gates passed after a fresh NO-GO/repair/GO
  review cycle.
- The registered unrestricted positive control completed 18 walks plus one
  wall-censored partial walk in 1,800.006626 seconds. Its 4,199,777 accepted
  transitions never improved rank 49, so it emitted
  `CONTROL_FAIL_M4_SYMMETRIC_FLIP_V1_NO_R47_WITHIN_BOX`.
- The frozen policy withheld the symmetric arm:
  `NOT_RUN_M4_SYMMETRIC_FLIP_V1_AFTER_CONTROL_FAILURE`. This is a bounded
  calibration failure only, with no matrix-rank or reachability inference.
- Canonical report: `research/M4_SYMMETRIC_FLIP_V1.md`; machine result:
  `results/m4_symmetric_flip_v1/result.json`. CPU only; GPU and protected
  census tree untouched.

## 2026-08-03 M4 schoolbook flip V2

- Rebuilt the positive control around the cyclic schoolbook anchor: rank 64, zero Brent violations, cyclic orbit histogram `{1:4, 3:20}`, 288 flip-eligible unordered pairs, and trace contraction 4.
- The compiled Rust bitset engine passed exact move/reduction fixtures and two final independent pre-run reviews. Single-thread throughput trials were 2.848M, 2.846M, and 2.928M accepted flips/s; median 2.848M, earning `M4_SCHOOLBOOK_FLIP_V2_THROUGHPUT_GE_1M_PASS`.
- The registered 900-second, eight-worker ordinary control recorded 907,698,133 accepted flips from admitted completed segments and reached first minima through rank 49, but did not reach the published rank-47 calibration. Final token: `CONTROL_FAIL_M4_SCHOOLBOOK_FLIP_V2_NO_R47_WITHIN_BOX`.
- Per the frozen control gate, the cyclic-symmetric arm was not launched: `NOT_RUN_M4_SCHOOLBOOK_FLIP_V2_AFTER_CONTROL_FAILURE`. This is a bounded calibration/search failure only, with no matrix-rank, cyclic-locus, or component inference.
- A post-run independent receipt audit passed 30 consistency checks; it is not proof against an unrecorded stochastic hit.
- Canonical report: `research/M4_SCHOOLBOOK_FLIP_V2.md`, SHA-256 `b223b4e97c5193840b9c28c4dca8b805a5a347109a61970dd84fbb028cede961`. Final handoff receipt SHA-256 `6c60340b5426260457306dffcef52814f2a41c2c72b742c5707967aa3fd22b4e`. CPU only; GPU, formatter, and protected census tree untouched. No git write attempted. Intended supervisor commit: `research: record M4 schoolbook flip V2 calibration failure`.

## 2026-08-03 M4 symmetric ladder V4

- Resumed the still-live, hash-bound V3 symmetric process without rebuilding the
  engine or rerunning calibration. The inherited plain control remains sealed at
  published rank 47 after 772,467,477,478 accepted flips.
- The registered symmetric box completed 15,000.179562 seconds on 24 CPU
  workers, accepted 225,336,488,679 symmetric moves, and reached first minima
  `64 -> 61 -> 58 -> 55 -> 52`. Marginal rung costs were
  `1,977 / 722,001,364 / 1,313,889,258 / 9,189,606,986`.
- Final pools were `{52:4378,55:20000,58:20000,61:20000,64:1}`. Rank 52 did
  not reach the 20,000-state promotion threshold, so source-rank-52 search for
  the 52 -> 49 rung never opened. No rank-49 or rank-46 state occurred.
- Final token:
  `SEARCH_M4_SYMMETRIC_LADDER_V4_NO_R46_WITHIN_REGISTERED_BOX`. This is only
  a bounded search report, never evidence of unreachability, locus absence, or
  component separation.
- A post-run exact replay/self-test receipt passed on the frozen binary. A fresh
  adversarial reviewer independently verified all four minimum witnesses against
  the target tensor, cyclic closure, `f=4`, and trace sum 4, failed the first
  report draft for rigor/provenance defects, then passed the repaired closeout.
- Canonical report: `research/M4_SYMMETRIC_LADDER_V4.md`, SHA-256
  `c4b7a1c9250b02d07b6b55fe7b84350c5567d4063e1710e969f5f565f9786c45`.
  Review receipt SHA-256:
  `55c44f2c5369d7081dbfee30ed9a4d0dd3d8aa6b077d64c4b1fe55d67ec8ef94`.
  CPU only; GPU and formatter unused. One initial read-only status scan was
  denied at the protected lock and made no mutation; all subsequent work
  excluded the protected tree. No git write attempted.

## 2026-08-03 M4 ladder promotion V5

- Changed the calibrated symmetric engine's search eligibility setting only:
  the retained pool limit remains 20,000 and the frontier promotion threshold is
  2,000. Search semantics, f=4 chart, pure orbit-removal moves, replay, worker
  seeds, segment size, and rank-46 stop protocol remain unchanged; accounting
  instrumentation now records source rank plus disjoint opening/search flips.
- The registered 15,000-second box completed at 15,000.145852 seconds on 24 CPU
  workers with 275,806,142,527 accepted symmetric flips, best rank 52, and
  source rank 52.
- Rank 52 promoted at 4,103.707392 seconds with 10,896.292608 seconds left
  before the registered deadline. Exact completed phase totals were
  68,266,042,733 opening flips and 207,540,099,794 search flips.
- Rung statuses: 64->61, 61->58, 58->55, 55->52, and 52->49 were ATTEMPTED.
  The 52->49 rung received 207,527,000,000 genuine source-rank-52 search flips
  and produced no rank-49 hit. Rank 49 was never reached, so 49->46 was NOT
  OPENED.
- Verdict:
  `SEARCH_M4_LADDER_PROMOTION_V5_NO_R46_WITHIN_REGISTERED_BOX`, EMPIRICAL and
  bounded. It makes no reachability, invariant-locus, or component inference.
  The executable's non-identical emitted token is disclosed in the canonical
  report as a reporting-label defect.
- All 259 registered telemetry records passed field, conservation, monotonicity,
  hash, final-result, and milestone checks. The post-run replay self-test
  passed. Fresh adversarial review returned NO-GO on four documentation defects,
  then GO after exact repair.
- Canonical report: `research/M4_LADDER_PROMOTION_V5.md`, SHA-256
  `10c8682bd2cce30e22ba90aa22211687df6f9ed0888bb89522573bcbb175f646`.
  Review receipt SHA-256:
  `bd89fddee34e88e21e78e6567c97d169fbb16391efa7256271e010ab6961821f`.
  Audit receipt SHA-256:
  `61caa08ea8d64fa381043446d87c169b01e9cee7b6108b055c4be6ed54a705e2`.
  CPU only; GPU, formatter, and protected census tree untouched. No git write
  attempted. Intended commit: `Record M4 promotion-threshold ladder search`.

## 2026-08-04 M4 staged ladder V6

- Reused the M6 staged symmetric engine through an exactly normalized 4x4 port;
  ran 15,000 seconds on 24 CPU workers with pool/promotion `20,000/2,000`.
- `64->61` SEARCH used 135,179 accepted transitions and 119,184 estimated
  distinct states, 5.319643x below the unconstrained V5 SEARCH counter.
- The exact rank-61 pool plateaued at 1,198, below promotion. Consequently
  `61->58` was `POOL-FILLED ONLY` and all deeper rungs were `NOT OPENED`.
- Final: 29,652,813,144 accepted transitions, 80,542,510 HLL distinct estimate,
  best 61, source 64, no candidate. Deep cost is indeterminate; this is not a
  reachability or invariant-locus claim.
- Canonical report: `research/M4_STAGED_LADDER_V6.md`, SHA-256
  `6a4aefef5540ddd75d50822375b66a61ddc3f9bb78c85ff1e83f6041628292b0`.
  Audit: `results/m4_staged_ladder_v6/postrun_telemetry_audit.json`, SHA-256
  `fa12ca9484031a3817d750bc03ec70ec8c766905096840f5d830ddb149fa9622`.
  Fresh review could not initialize (OS error 5); no independent review claimed.
  CPU only; GPU and formatter unused. One initial read-only git status was denied at the protected census lock with zero mutation; all later commands excluded the tree. No git write.

## 2026-08-04 M4 adaptive-gate ladder V7

- Reused the staged M4 V6 engine and changed the promotion gate only, from 2,000
  to the precommitted 750, with an m=3 transition/wall/promotion escape to m=4.
- Promoted ranks 61, 58, and 55; reached rank 52 but retained only 603/750
  states. Therefore 52 -> 49 was POOL-FILLED ONLY with zero staged SEARCH work,
  and the deep staged price remains unmeasured.
- Final: 32,239,065,130 accepted transitions, 849,484,577 HLL global distinct
  estimate, best rank 52, source rank 55, stop reason wall. Rank-52 opening cost
  was 31,836,288,116 transitions and 512,624,626 estimated distinct states.
- The full distinct/dependency series has 258 durable records. All accounting
  and hash checks passed; no plateau predicate window fired. Reduction moments
  averaged 41.972170 dependencies per slot versus 36.515539 for fixed-cadence
  live samples, but stable interval correlation was -0.1159.
- This is a bounded search report only, with no rank-49/rank-46 reachability
  inference. No rank-at-most-46 candidate appeared.
- Canonical report: research/M4_ADAPTIVE_GATE_LADDER_V7.md, SHA-256
  997d3bd449fff635d29eed48f58bd737ad49ae5b9f606cb6394d71f5ca601d40.
  Audit: results/m4_adaptive_gate_ladder_v7/postrun_telemetry_audit.json,
  SHA-256 9e54893dee22523b5b7c4fa17164157e79afcadec540558d81c5b9b52557e029.
  Fresh review could not initialize (OS error 5); no independent review claimed.
  CPU only; GPU, formatter, protected census tree, and git writes untouched.

## 2026-08-05 DPS48 rank excursions V8

- Exact donor-factor rank excursions admitted 466 plus states and 14,658 novel
  rank-49 flip states under a 1,800.03125 process-CPU-second cap.
- All 1,232 detected reductions were applied and exact-T4 verified. Every
  rank-48 landing canonically equaled DPS48; zero new decompositions, rank-47
  candidates, or armed L149 candidates appeared.
- Rank-48 tier-1 remained the singleton `{497: 1}`. Fresh standing-certificate
  replay passed all circuits, 4,096 Brent coordinates, and 32 exact-rational
  pairs; L remains 172.
- This is bounded: 468/6,768 root donor descriptors were considered. It closes
  the owner-defined DPS48 arc, not the full rank-49 graph or all excursions.
- Canonical report: `research/DPS48_EXCURSION_V8.md`. Machine evidence:
  `results/dps48_excursion_v8/production/`; ledger row
  `dps48-excursion-v8-result`.

## 2026-09-07 representation-side restart (theory, no experiment run)

- `representation-frontier-v2`: independent proof and delta GO establish a scoped
  exact-complex-bilinear comparison. For ambient order 24 and dimension 145,
  the square-zero algebra uses 289 products; every semisimple competitor needs
  at least 291 by the imported Blaser bound. This is not a total-operation or
  runtime advantage. The constructions are classical controls.
- `girs-representation-obstruction-v1`: registered proof candidate, independent
  review pending. On Q6 all matrices are GIRS-1, while width-two DV has 2496 raw
  scalar parameters versus 4096 matrix entries. No numerical experiment run.
- Reports, source limits, registrations, and review receipts are under `research/`;
  exact events are appended to `experiments/ledger.jsonl`.

The GIRS proof gate subsequently passed; see its proof receipt and independent
review. The separate tree-composition proof candidate is registered and pending
review, with no experiment or performance run yet.

Tree composition implementation: the registered two-case smoke passed, then
passed after numeric validation. The subsequent directed-width smoke passed four
cases after a pre-full-run amendment preserved the original 24 and added two
asymmetric holdouts. An exact signed/unsigned promotion regression also passed.
Full correctness execution remains pending code delta GO. All smoke attempts and
source hashes are in `results/tree_composition_v1/` and the JSONL ledger.

## 2026-09-07 tree composition V1: reviewed correctness control

The registered full run passed 26 fixtures and 260 comparisons, including 14
exact integer/rational cases and 12 dyadic floating cases. Ten invalid contracts
and the large signed/unsigned integer regression passed. Fresh independent
evidence review reproduced every saved row and checked a second global-state
oracle. Result hash c6feef8682da643c1a583b6fc9c6e2fe3a942dfd55c0af55c376516317ea9790;
evidence receipt research/TREE_COMPOSITION_V1_EVIDENCE_RECEIPT.json. No timing,
general floating stability, new-class, or specialist advantage is established.
Native reduction is a separately registered theory candidate; no reduction
implementation or experiment has been run.

## 2026-09-07 separator support: reviewed theory only

The global polynomial support obstruction passed independent proof review.
No numerical experiment or benchmark was run. The proof and source renders are
bound in research/SEPARATOR_SUPPORT_OBSTRUCTION_V1_PROOF_RECEIPT.json.

## 2026-09-08: dps48-retained-reconciliation-v1   EXECUTION_VOID

The one authorized attempt failed at its initial child-process monitor before source replay. Independent review confirmed no admissible candidate/comparison or H0/H1 result. Child PID37876 was recorded without identity. Frozen inputs, code and failure receipts are preserved; no replay authority remains. See research/DPS48_RETAINED_RECONCILIATION_V1_DISPOSITION.md and results/dps48_retained_reconciliation_v1/completion.json. Full mission unchanged.


## 2026-09-08: M4 finite compact comparison, reviewed

EMPIRICAL / COST_RESULT_SCOPED. The historical M4 comparison completed84 cases
and504 phase records with independent evidence review. Encoding/conversion costs
are separately scoped; fixture construction includes Python arithmetic outside
field counters. No useful consumer or uniform compact-comparator advantage was
established. The ordered converter was not measured in this historical run.
See [M4 disposition](research/bruhat_product_build_v1/M4_DISPOSITION_V1.md).
The run and its raw receipts remain frozen; no replay is authorized.

## 2026-09-09: ordered comparison terminal, evidence review pending

Registered run: `results/ordered_comparison_v1/ordered-20260909t004834z`.
Design and independent core/harness software gates passed;22 local regression
groups and independent attacks cover the repaired protocol. The57-binding
execution packet validated, actual admission completed within its deadline,
and owned controller43328/worker45316 completed with exit0. All84 cases and504
route records are preserved; complete inventory and analysis recomputation match.

EMPIRICAL, PROVISIONAL: ordered conversion used fewer field multiplications than
R2 on this finite domain, while h1 costs missed the registered per-row heuristic.
All misses and controls remain included. Native construction/preparation/native
phases retain their separate accounting, and SSS-only output remains a different
contract. No compiled-specialist or useful-workload speedup is established.

Independent evidence review is running against156 frozen bindings. Its eventual
receipt controls whether COST_RESULT_SCOPED is granted. See [live status](research/ordered_comparison_v1/current_status.json),
[provisional counts](research/ordered_comparison_v1/PROVISIONAL_COUNTS.json), and
[heuristic explanation](research/ordered_comparison_v1/HEURISTIC_MISS_ANALYSIS_V1.md).
No campaign replay or result-based selection is authorized.

A prior controller preflight failed on a missing authorization packet after
provenance extraction rejected a log containing two different session headers.
It created no run directory or worker. The original-header extraction and full
raw-log binding were corrected before the one registered campaign above; the
failure receipt remains preserved in the unit and JSONL ledger.

## 2026-09-09: associated theory gates, not numerical experiments

The [full batched pipeline](research/batched_pipeline_v1/DISPOSITION.md) has
CONDITIONAL_PROOF_GO for exact compact-input sum/product under its explicit
F(h), workspace, input and output contract. It removes the inspected sum
algorithm's field-operation logarithm; index/bit/runtime costs remain scoped.
The existing implementation is classical and does not realize a fast backend.

The [normalized-output reduction](research/normalized_conversion_reduction_v1/DISPOSITION.md)
has SCOPED_PROOF_GO for one-call dense-product extraction from the specified
normalized single-block converter. It grants no arbitrary-output or direct-sum
lower bound and no dense multiplication exponent. Neither theorem completes a
mission tier. These proof gates used bounded corroborating checks, not campaigns.

## 2026-09-09: ordered comparison evidence accepted

The independent review completed COST_RESULT_SCOPED with all156 bindings verified and no required repairs. The prior pending entry is superseded by [the disposition](research/ordered_comparison_v1/DISPOSITION.md). Multiplication-count gains coexist with owned-memory and preparation costs; no useful-workload or mission-tier claim follows. Frozen raw records and provisional notes remain unchanged.

### NumPy batched V1 worker correctness smokes

Full native compact arithmetic component implemented.648 conversion/backend checks,48 binary/backend/operation checks and72 separate n17 holdout checks pass. Original holdout sources and later accounting repair are preserved separately. Independent integrated software gate is running; no broad timing or useful speedup claim. See research/numpy_batched_v1/IMPLEMENTATION_REPORT.md.

### NumPy batched V1 independent software disposition

Fresh actual review15060 exited0 with SCOPED_SOFTWARE_GO. The accepted scope and additional adversarial checks are in research/numpy_batched_v1/SOFTWARE_DISPOSITION.md. The earlier pending entry is superseded; no timing campaign or useful-speedup result was run.


### 2026-09-09T02:34:06.147881+00:00

Theory gate: echelon-contract reductions accepted by independent actual reviewer 8918, exit 0. Seven bindings match. Reviewer used 18 fresh deterministic checks (not a registered timing campaign or holdout replay); proof scope and limits: research/echelon_contract_v1/DISPOSITION.md.
