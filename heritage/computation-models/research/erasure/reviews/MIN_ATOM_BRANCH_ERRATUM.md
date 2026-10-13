# Distinct-output context transfer in the min-atom shortcut

2026-10-03 PROVED AT SCOPE correction: fresh independent actual UPPER/enclosure
PASS in MIN_ATOM_BRANCH_ERRATUM_REVIEW.md. Original correction text retained.
OCTOBER_MILESTONE_REVIEW_1 found a nonfatal contract mismatch in the earlier
MIN_ATOM_BRANCH_REVIEW shortcut. That original report is retained verbatim.
Its theorem-level universal V enclosure and first-positive-merge lower are
not changed by this correction. No registered scientific source is modified.

The common clean-output contract requires original R copied into DISTINCT
output registers and original R input registers cleared. Leaving R in place
does not satisfy that contract: r1R1 leaves a dirty input and an unset output.

Correct finite word:
1. For each i, CNOT(Rin_i,Rout_i); CNOT(Rout_i,Rin_i), 2r gates.
2. Balanced AND tree on n X bits into n-1 initialized-zero scratch bits;
   CNOT(tree_root,Y); uncompute the tree. This is 2n-1 gates.
3. RESET each original X bit, n gates.

All X/R input and tree scratch are zero, output is (AND(X),original R), no
other live register. Total G=3n+2r-1, width=n+r+1+r+(n-1)=2n+2r. Gates are
NOT/CNOT/Toffoli/one-bit RESET only; source/preparation, program generation and
encoding are separately finite charged resources. No inverse/oracle/precision leak.

R remains recoverable at every step and classes cannot mix its original values.
For independent uniform R, any positive incoming class has probability at least
beta*delta and any post class at most beta; thus every positive branch V<=
log2(1/delta). Reversible transfer/tree gates have Vzero. All preimages are
finite; zero-probability original atoms remain covered by pointwise correctness.
The original first-positive-merge lower on the finite nine-atom prior is unchanged.

Fresh actual read-only zero-context reviewer: attack full clean-output contract,
width/exact work, n>=4/r0 and finite-r boundary, branch support including zeros,
post/pre bound and unchanged lower/enclosure. Bounded exact CPU probes welcome;
no edits/agents/GPU/FULL. Return corrected UPPER/unchanged enclosure PASS/REPAIR/KILL.
