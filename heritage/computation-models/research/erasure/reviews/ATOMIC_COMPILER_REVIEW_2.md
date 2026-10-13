**1. CODE GATE: PASS for the current revisions below.**

```text
entropy_controls.py
c22765f9a084260d81dc274128939c57757545d23612f372f08ec10055044dad

atomic_compiler.py
40b24c94078b21a49ff6e4b7b73ff424ada096e06ea66bbb1ddcb86ba5327e4e
```

The current guards check both `sys.flags.optimize` and `not __debug__`. The latter closes a reproduced bypass: explicitly compiling source with `optimize=1` removes assertions even when process optimization flags remain zero.

Passed read-only probes:

- `-O`, `-OO`, `PYTHONOPTIMIZE=1/2`, and explicitly optimized code objects all rejected.
- Incorrect clean circuits, missing resets, wrong outputs, dirty scratch, overlapping gates, and invalid indices rejected.
- Outside/traversal output paths rejected before computation; simulated directory and permission failures emitted no success.
- Independent simulation passed **310 selective merges, 30 XOR predicates, and 50 small truth tables**, including unequal input/output widths. Gate counts independently agreed.
- Independent 100-digit entropy calculations differed from floating calculations by at most **4.86e-16**, comfortably below `1e-12`.

Both current smoke artifacts match current source hashes and in-memory reruns, excluding elapsed time. **No current smoke provenance debt remains.** Neither FULL experiment was executed; I edited no files.

**2. PROOF GATE: lower PASS; charged upper PASS, at the exact stated scope.**

The lower bound survives arbitrary simultaneous and multiway coalescence. Determinism prevents full-state classes from splitting or crossing final fibres. At first complete formation of a largest fibre, convexity forces loss at least `phi(q)/M`; other merges cannot offset it. Independent small minimax searches allowing all coarsenings also agreed.

The upper construction survives the selective-reset attack: the predicate ladder is uncomputed before input changes; controlled XOR moves only the flagged singleton; flag reset merges precisely it with the accumulated representative class.

Terminal conversion is valid: reversible XOR evaluation produces `f(rep(f(x)))=f(x)`, then reversible representative XOR clears the input. Scratch returns to zero. No arbitrary permutation or overwrite is a primitive.

The charged bounds `width = n+m+max(n,m)+1` and `O(2^n n m)` elementary gates hold. Truth-table storage, program encoding, and construction work remain additional charged resources; these outputs do not measure exact construction bit-operation cost.

This conclusion applies to **deterministic, uniform, exact clean computation**, excluding fresh randomness, information-bearing timing, and external information channels.

**3. Evidence admission after FULL execution**

Require:

- Unchanged reviewed hashes, matching smoke provenance, recorded commands/environment, and complete successful execution.
- Exactly **78 clean AND cases, 12 dirty controls, 16 atomic cases**, and **308 compiler cases** under the frozen registrations.
- Independent reconstruction from tables and integer block traces of correctness, cleanup, gate counts, and entropy losses.
- Explicit treatment of the Decimal equality probe’s `1e-60` slack; numerical agreement remains **EMPIRICAL**, while exact equality rests on the proof.
- Separate accounting for table/program/construction costs in any resource claim.

The runners hash files after computation. Admission therefore needs a stable source snapshot; an end-of-run hash alone cannot establish which bytes were loaded. **Unseen FULL output is not certified.**

**4. Novelty/usefulness standing**

The useful scoped result is an exact feasibility threshold distinguishing tasks with identical total entropy loss. Novelty remains unestablished: [entropy aggregation](https://arxiv.org/abs/1805.05375) and [conditional reversibility](https://arxiv.org/abs/1806.10183) are substantial adjacent areas.

The arbitrary-table compiler is exponential in input length. This establishes **no efficiency separation, physical energy advantage, priority claim, or achievement of the new-paradigm goal**.