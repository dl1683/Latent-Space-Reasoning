All 7 Round 6 fixes are correctly applied in the amended Blueprint.

Checked:

1. `generation_start_index` is no longer saved as a fixed scalar; `boundary_mode` is saved and start index is computed per call from `combined_length`.
2. Zero-prefix baseline detection uses `len(zero_output.scores)`.
3. Component 2 asserts lean rerun prefix-equivalence against the observed token prefix.
4. H1 is defined as MI of scalar `routing_score`, with per-feature MI only descriptive.
5. Within-task z-scoring is per task group, and `std < 1e-6` is treated as degenerate for that feature/group.
6. Arithmetic validator is explicitly placed before Phase 1/2 data collection.
7. Phase B edge cases are specified: inclusive thresholds, non-significant/ambiguous H1 extension, H5 ties, and all-wrong groups excluded.

No new substantive architectural concerns.

**Design is converged. Blueprint is ready for engineering handoff.**