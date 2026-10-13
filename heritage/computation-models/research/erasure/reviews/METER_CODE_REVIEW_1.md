**CODE REPAIR — missing registered provenance. No mathematical implementation defect found.**

[METER_EXPERIMENT.md:76](/<LOCAL_PROJECTS>/new-computation-model/research/METER_EXPERIMENT.md:76) requires runtime fingerprints and a recorded pre-run environment. Neither the successful smoke artifact nor its exit receipt contains them, and no METER environment artifact exists. Record that provenance, bind it to the attempt/receipt, and repeat the smoke before requesting PASS. This is an evidence repair, not METER-KILL.

The substantive checks passed:

- Independently rebuilt incoming macro: **61,896 valid encodings**. Dirty-predicate, retained-flag, and missing-advance mutants failed.
- Independent replay: **27 circuits, 14,171,624 transitions, 2,790 RESETs, 27 weighted traces**. Every saved smoke RESET record matched.
- Exact LP controls: **51,480 vertices**, shared-X/original-R counting, denominator \(8·2^r\), capped rational mixtures, and an interior entropy maximum exceeding both endpoints.
- Old/incoming ordered J and post-macro signatures matched. Incoming singleton/final-empty flags and local floor passed. Old n3/r0 reached local peak one with final J zero/L positive. Context-linear bounds passed, including n≥4 equality without imposing it at n3.
- Six additional source `inspect` calls passed N=128 arithmetic and JSON encoding. Weighted complete-state X0/R telescopes equalled \(2^N\); no astronomical weighted entropy products were evaluated.
- In-memory failures confirmed RUNNING invalidation, first-failure termination, source-change rejection, and inventory rejection. Output confinement and five optimization guards passed.
- Registered inventory remains **72 circuits/360 traces**; saved smoke is **12/12**. Failed and successful receipts match their artifacts; the earlier scalar-moment defect is corrected.

Verified SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `METER_EXPERIMENT.md` | `6f2fa67cf0ad0ac67fc58424ee118bd64deac641b62507d715ffde1668624bf7` |
| `meter_controls.py` | `c0ec80f180faef292c9d20997665eed48e0341bdb38d15034ab44370c5784079` |
| `run_snapshot.py` | `4e1028c8e340d7ca2e90c3ac29e123d5bb56707493b7ceca36f6b7932c8b1881` |
| `evidence_io.py` | `2bf4310800821fabf69f9a95bb71561ddf1a595d9263ec6f8a3fa90d1675386f` |
| `succinct_erasure.py` | `a8c1d33a98822de83792f344842c9ac90b709134eb6cb10d012d71b6f3f0e71c` |
| `fair_bit_cleanup.py` | `3b2cbbd3158e6d6d89264fc6b1003a5f0f05fe1fe19c00ec1d462ef408cf8603` |
| `context_threshold.py` | `2756a1ade91b0668ef1de6fde31c6ceb0ce624f9fe5b94d77f299098571ef519` |
| Successful smoke | `963ec96a4fdb4c70adf5ab336687bb681d0f48bc9c03b006fb98bd5d0ac253e2` |

Residual trust covers Python/standard-library arithmetic, launcher/publication machinery, filesystem behavior, and the historical external exit observer. Reviewer runtime was Python 3.13.7 with assertions enabled; that does **not** recover the smoke’s historical environment. Fixed dyadic preparation remains separately charged. These finite probes establish neither universal impossibility nor physical/novelty claims.

No edits, agents, GPU use, or registered FULL main invocation.