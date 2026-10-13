| FULL admission | Verdict | Independently reconstructed |
|---|---|---|
| SUCCINCT-ERASURE-1 | **PASS — finite controls only** | All 144 cases: 114 feasible, 30 below-cap |
| PRIOR-UNCERTAINTY-1 | **PASS — finite controls only** | All 16 circuits, 55 adverse witnesses, and 12,871 subset priors |

**Succinct:** 12,119,612 atom/gate transitions matched every saved row, including all 3,721 RESET ratios and macro invariants. Complete terminal states, pointwise returned Z, width, gate counts, and all 24 exact boundary equalities passed.

**Prior:** 22,765,520 full-support transitions and 25,740,968 subset-prior transitions passed. All 2,144 complete-state RESET traces matched. Every adverse witness independently produced eight distinct predecessors, four successors, and exact ratio **256 > 64**. Robust RESETs merged at most one pair. All saved subset peaks matched exactly:

- `823543/46656`: 6,436 priors.
- `16777216/823543`: 6,435 priors.

Both [completion receipts](</<LOCAL_PROJECTS>/new-computation-model/results/succinct_full_exit_receipt.json>) ([prior receipt](</<LOCAL_PROJECTS>/new-computation-model/results/prior_uncertainty_full_exit_receipt.json>)) bind recorded exit **0** to the correct commands, artifacts, inventories, and attempts:

```text
Succinct: 540facdc-d5eb-4239-8e79-170ab735b6f1
Prior:    6931faf8-b536-4d8e-9770-7c4656c4ef51
```

Executed sources match reviewed hashes and exact bytes at `6bf05cc`. Frozen protocol sections are unchanged. Current smoke rows match the corresponding FULL subsets. The saved failed attempt remains inadmissible; 18 synthetic completion/inventory defects were rejected.

My first prior audit stopped before circuit evaluation because I flattened the inventory encoding. The corrected grouped encoding matches the documented fingerprint; the completed independent audits exited **0**, taking approximately **5.12** and **18.26** seconds.

Core SHA-256 fingerprints:

```text
succinct_erasure.py
a8c1d33a98822de83792f344842c9ac90b709134eb6cb10d012d71b6f3f0e71c
prior_uncertainty.py
881ca25349ab1ab7222cb212118ec0a1ce6f9e21cfb962613a590fab5fa90afc
evidence_io.py
2bf4310800821fabf69f9a95bb71561ddf1a595d9263ec6f8a3fa90d1675386f
run_snapshot.py
2681e422d7cb5ba7b50f61a7bb973b6e308cd7a4f9b8da7386437c48038406a1

succinct_erasure_full.json
06e54a9bc1939e0ad59ef9fd98f492d48fbe9644e7c9a52c9ef7dab0bd03a861
succinct_full_run_environment.json
a58c1877d48b0e7659d5e3923b4960f6a4d83fe85cb7b0300e721c4265fc8f90
succinct_full_exit_receipt.json
f72cc62f32a0182be4d46057be8e87e25d321e019be3a2c1ac3ff46ad0bc8fde

prior_uncertainty_full.json
a5ce8392cc59d679c0d54d301c733bb1a9fd88919002119fe72bea1e178c3dd4
prior_uncertainty_full_environment.json
8f97b7d3d45c413d2f78cecf8faad281678584b15f157eb0bb70b9d0df3bcf14
prior_uncertainty_full_exit_receipt.json
fee5da44a5a6c6de01c18b09eae1c6590ce0f5f997c73ae24cb533c608849554

Succinct FULL inventory
131055feef2387943be17833df562b81d5d931154c9c0e1c3019a3422000ff43
Prior FULL inventory
7d38ff241c6b8187a70b326535faad000dfdea4457bc8e9ca5ddad102ee5077a

Independent succinct auditor source
75faf1861cb0be19984abc8fbfceaf34de6ef9ad329fdfc55945bed110802a90
Independent corrected prior auditor source
77feff8f9fac30e9241801e8b421482f37be3817f10c67e87842ea5fff8c5648
```

Probe code, actual outputs, complete inventories, and all 23 checked fingerprints are retained in this turn’s tool transcript. All 23 remained stable at final recheck. No files were edited, runner `main()` called, agents added, or GPU used. Arithmetic used exact integers/Fractions throughout.

These are **EMPIRICAL finite admissions**. They add no general proof, priority, physical interpretation, or whole-mission pass. The reviewed no-helper theorem remains PROVED at scope; the unknown-prior and uniform promises differ. Returned-fair-bit capability remains a separate registered candidate. Both invariants and the CS/math mission remain active.