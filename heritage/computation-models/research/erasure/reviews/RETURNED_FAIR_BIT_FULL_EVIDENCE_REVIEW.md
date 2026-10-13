**PASS — finite FULL evidence admission only.** No blocking defect survived.

Independently rebuilt each gate stream and evaluated complete states without calling runner `main()` or using its validation functions as evidence.

| Inventory | Transitions | RESET distributions | Terminal states checked |
|---|---:|---:|---:|
| 33 laws across n=2..12 | 3,243,856 | 462 | 32,752 |
| 4,336 correlated priors | 728,384 | 33,600 | 17,344 |
| **Total** | **3,972,240** | **34,062** | **50,096** |

All inventories match: eleven circuits, three laws each, and **48 + 448 + 3,840 priors**. Every circuit has exactly **8n+11 gates**, **2n+7 width**, clean workspace, correct AND output, and pointwise-returned R. Every RESET loses **≤½ bit**. All **462 saved RESET certificates and 4,369 peak certificates** match independent arithmetic.

All **4,831 certificates** decoded and round-tripped at the unchanged **4,300-digit limit**. The n=11/12 uniform peaks independently reproduce numerator/denominator lengths **6,778/6,162** and **14,793/13,560**; their decimal conversions fail while factor decoding succeeds.

The [artifact](<LOCAL_PROJECTS>/new-computation-model/results/fair_bit_cleanup_full.json), pre-run environment, and [observed-zero-exit receipt](<LOCAL_PROJECTS>/new-computation-model/results/fair_bit_cleanup_full_exit_receipt.json) bind attempt **`a6be3594-988b-497d-88db-1c9417cca122`**, its command, inventory, and executed sources. Attribution uses revision **`68e6e33f1169cb08d9ac9d6e5e04ece7e2558abc`** and its frozen launcher. The later launcher is excluded.

Failure-contract inspection confirmed RUNNING invalidation, startup/end source checks, first-failure publication, KILL/VOID classification, and nonzero failure returns. Six optimized-code probes and three invalid-output probes rejected execution/publication. This admission uses the stated trusted interpreter/launcher/root receipt model.

Exact SHA-256 fingerprints, rechecked:

```text
experiments/fair_bit_cleanup.py
3b2cbbd3158e6d6d89264fc6b1003a5f0f05fe1fe19c00ec1d462ef408cf8603
experiments/succinct_erasure.py
a8c1d33a98822de83792f344842c9ac90b709134eb6cb10d012d71b6f3f0e71c
experiments/evidence_io.py
2bf4310800821fabf69f9a95bb71561ddf1a595d9263ec6f8a3fa90d1675386f
experiments/run_snapshot.py — frozen
c1e6877d532cd4ec7e17323a1a88af50bde0c73842df98a87027ad808610035b

results/fair_bit_cleanup_full.json
f94d4dd3244e95643be8d5ee0399083f61a8c513798ae153f1889fcc02356b3d
results/fair_bit_cleanup_full_environment.json
397410fdbfaca01fd41941284e890a4ba2bfccd74e1504007e27eab48a591e2e
results/fair_bit_cleanup_full_exit_receipt.json
add65f978510dfcf28cdfc4a3e59b459b1051e2a647e648dfafcb4cba91dee7e

research/RETURNED_FAIR_BIT_CANDIDATE.md
79fcfa9f85bc670d961e6abe4dc5575c20f3e27b386bdcc63191f13688b1e8cc
research/RETURNED_FAIR_BIT_REVIEW_1.md
d2f8c56a4e9c8c36f475a98b1696d644831c0ab7ea2f9a3781892bd9e37438a2
research/RETURNED_FAIR_BIT_CODE_REVIEW.md
d477c04b99d55137e35d9f5e1794cc266a7019ec97b4aa954465677504fa8d53
```

**EMPIRICAL:** these finite controls corroborate the compiler. The independently reviewed universal theorem remains separate. Supplied fair R changes the statistical resource; finite checks cannot authenticate its promise. Ensemble scheduling remains the strongest triviality objection. Physical-meter validity, priority, useful-world value, new-paradigm standing, and whole-repository completion remain **OPEN**.

Read-only CPU review; no agents, edits, GPU use, or FULL runner calls.