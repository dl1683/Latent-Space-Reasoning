**Lower PASS. Upper PASS at elementary gate-work scope.** The candidate’s exact threshold survives independent attack:

\[
\inf_C \operatorname{peak}_V(C)=\log_2 3
\]

for every \(n\ge2\) and finite integer \(r\ge0\), under uniform X, independent uniform R returned pointwise, and the finite fixed-schedule clean-output contract. No mathematical counterexample or proof repair was found. [Candidate](<LOCAL_PROJECTS>/new-computation-model/research/UNIFORM_BRANCH_CANDIDATE.md:9)

First shell was explicitly `C:/Windows/System32/cmd.exe`, `login=false`. I read AGENTS, the full dual-loop skill, required orientation files, machine contract, METER_REVIEW, and source dependencies. This review used read-only CPU probes, with no agents, edits, GPU use, or FULL runner.

**The universal lower bound is valid.** Each live complete-state class initially contains one original atom. Reversible gates preserve class cardinalities and cannot split previously coalesced labels. A physical one-bit RESET has at most two predecessor states for each successor, regardless of logical recoding or correlated scratch encodings.

Suppose all current class sizes are powers of two. If a RESET merges unequal positive sizes \(a<b\), then \(b\ge2a\), and the smaller predecessor incurs

\[
V=\log_2\frac{a+b}{a}\ge\log_2 3.
\]

Below that cap, only equal-sized classes can merge; their sum remains a power of two. Singleton transitions preserve cardinality. Zero-mass physical states contribute no positive predecessor branch.

Complete states containing different original R labels cannot coalesce: identical states have identical deterministic futures, contradicting pointwise R return. The same argument prevents mixing final AND values. Consequently, each final AND-zero/R state must contain exactly \(2^n-1\) atoms. For \(n\ge2\), this is not a power of two. The contradiction covers arbitrary finite width, depth, earlier coalescence, and temporary recoding.

**The actual upper construction is valid.** [fast_circuit](<LOCAL_PROJECTS>/new-computation-model/experiments/context_threshold.py:49) transfers R with two CNOTs per bit. Its AND tree uses \(n-1\) Toffolis, copies the result, and reverses the tree, clearing computation scratch before cleanup.

The selected-reset source is exactly

```text
TOFFOLI(predicate, X_bit, flag)
CNOT(flag, X_bit)
RESET(flag)
```

The flag starts clean. On the selected slice it stores the erased bit while X becomes zero; on other slices both remain unchanged. The predicate depends only on returned R and remains valid through data erasure. RESET clears the flag. Context predicates are subsequently uncomputed, and the sentinel returns to zero. There are no additional hidden scratch RESETs. [Macro and AND source](<LOCAL_PROJECTS>/new-computation-model/experiments/succinct_erasure.py:93)

After erasing \(j\) data bits in a selected context, an ordinary AND-zero suffix class contains \(2^j\) atoms. The all-ones suffix contains \(2^j-1\), because its missing atom belongs to the separate AND-one state. The next RESET therefore merges:

- ordinary children of sizes \(2^j,2^j\), giving ratio 2;
- exceptional children of sizes \(2^j,2^j-1\).

At \(j=0\), the exceptional second child is absent, so that transition has ratio 1. For \(j\ge1\), its largest ratio is

\[
\frac{2^{j+1}-1}{2^j-1}
=2+\frac1{2^j-1}\le3.
\]

Equality occurs at \(j=1\), which every \(n\ge2\) construction reaches. The AND-one class stays singleton. Other contexts undergo no collision. Thus the actual peak is **exactly** \(\log_2 3\).

The charged gate count is

\[
G=2r+(2n-1)+2+3n2^r+8(2^r-1).
\]

Context maintenance contributes the final term: prefix level \(z\) executes \(2^{z+2}\) literal macros, equally divided between costs one and three. Width is \(2n+3r+4\). This establishes \(O(n2^r+n+r)\); at \(r=0\), \(G=5n+1\).

**Bounded independent evidence — EMPIRICAL only:** 28 actual-source circuits covering \(2\le n\le8\), \(0\le r\le3\); 1,348,084 elementary transitions; 525 RESETs; and 7,410 binary collisions. Independent simulation checked complete states, macro boundaries, exceptional child counts, final cleanup, and exact integer branch ratios.

| \(n,r\) | Gates | Width | Maximum branch ratio |
|---|---:|---:|---:|
| 2,0 | 11 | 8 | 3 |
| 3,1 | 35 | 13 | 3 |
| 4,2 | 85 | 18 | 3 |
| 8,3 | 271 | 29 | 3 |

Candidate, model, source, and dependency hashes remained unchanged. Universality follows from the proofs above.

**Additional mathematical finding:** context enumeration is unnecessary for V. Transfer R, compute/copy/uncompute AND, then use the sentinel to select every context simultaneously for the same n cleanup macros. Each complete-state collision remains within its returned-R fibre, with unchanged branch ratios. This construction uses exactly \(5n+2r+1\) gates. Together with the input-targeting lower bound \(\Omega(n+r)\), optimal feasible elementary work is **\(\Theta(n+r)\)**, including growing r. Six additional bounded circuits passed this construction.

Acceptance concerns elementary gates. Program encoding, generation bit work, and preparation/authentication remain separately charged; no end-to-end linear bit-work theorem follows. If budgets must be rational, feasible budgets are exactly rational \(K>\log_2 3\); the symbolic boundary is attained by peak ratio 3.

The honest useful result is an exact source-sensitive branch constraint with efficient cleanup. Returned uniform context provides no asymptotic V advantage here. Physical calibration, a useful same-contract advantage, preparation costs, novelty/priority, and the whole-repository mission gate remain unresolved. **This passes the scoped theorem gate; it does not pass mission completion.**