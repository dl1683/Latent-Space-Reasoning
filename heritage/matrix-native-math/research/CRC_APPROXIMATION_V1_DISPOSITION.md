# CRC approximation proof disposition

PROVED / APPROXIMATION_CONSTRUCTION_REVIEWED. Independent real Codex review
terminated with exit zero and turn.completed; all retained claims received
PROOF GO, with no mandatory fixes. The worker accepts the derivation and scope.

For the registered row-snake CRC family with column parameters at most 1/2,
the finite-range construction has relative spectral error at most 972*2^(-L),
width at most (L+1)(2L+2) per flow and sparse arithmetic O(N L^2).
The reviewed draft gives the full offset masks, legal neighboring-state
equations, boundary cases and rational descriptor-bit accounting.
The tolerance choice L=ceil(log2(972/eps)) is independent of dimension.

The review independently checked the inherited conditioning argument, both
flow orientations, zero displacement with nonzero intermediate offsets,
invalid intermediate rows with valid endpoints, reversed parameters,
missing neighbors and L>=s. No finding requires changing the frozen draft.

Original Roesser/Kung comparisons and applicable DFZ obligations remain open.
There is no LITERATURE_SETTLES or novelty finding, dense-input recognition,
floating-point stability, exact bit-runtime, or general closure claim.

Narrative: exact state growth in this family does not force growing width
at fixed accuracy. This is a representation component; the retained exact
CRC word already applies in O(N), so no computational speedup is earned.
The full mission remains unfinished. Useful-workload nomination continues.

Evidence: [review](reviews/crc_approximation_v1/final.txt),
[reviewed draft](CRC_APPROXIMATION_V1_DRAFT.md),
[receipt](CRC_APPROXIMATION_V1_PROOF_RECEIPT.json).
