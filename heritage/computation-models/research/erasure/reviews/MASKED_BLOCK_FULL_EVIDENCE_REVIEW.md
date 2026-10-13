**EMPIRICAL PASS** for the frozen [MASKED-BLOCK-1 protocol](<LOCAL_PROJECTS>/new-computation-model/research/MASKED_BLOCK_EXPERIMENT.md) and [FULL artifact](<LOCAL_PROJECTS>/new-computation-model/results/masked_block_full.json). No circuit, certificate or binding defect found.

I independently authored both word constructors and a CPU interpreter, compared every elementary word with the frozen source, and evaluated every original label through every gate. Source validators, runner `main` and recorded `CONFIRM` were not used as independent evidence.

| Independently checked inventory | Circuits | Gates | Original labels | Label–gate transitions | RESETs |
|---|---:|---:|---:|---:|---:|
| n=3..16, r=1..8 | 112 | 217,781 | 66,842,640 | 426,760,924,048 | 11,998 |
| n=3, r=9..12 | 4 | 122,036 | 61,440 | 2,831,720,448 | 6,720 |
| **FULL total** | **116** | **339,817** | **66,904,080** | **429,592,644,496** | **18,718** |

Gate totals are 46,281 CNOT, 134,384 Toffoli, 140,434 NOT and 18,718 RESET. The five smoke circuits additionally passed independent scalar/packed agreement at every gate: **160,240 transitions and 45 RESETs**.

Original label order was preserved as `x + (R << n)`. The interpreter packed 64 labels into independent Boolean lanes; it sampled nothing. Every terminal register matched
`(R << (n+r+1)) | (AND(x) << (n+r))`.
Thus returned R, clean AND and every other register being zero passed for all **66,904,080 labels**, covering **3,685,441,584 terminal register-bit comparisons**. Final class counts give total decline **66,874,440**.

All operands had legal arity, in-range indices and distinct controls/target. Widths matched `2n+3r+4`, with maxima **60/46**. Source controlled shifts first reduce to zero or one, so shifting to targets below these widths cannot truncate. RESET masks are finite unsigned 64-bit complements. Additional probes passed all 64 RESET masks and 4,032 nonalias CNOT bit-position combinations.

At every RESET, disjointness checks across **original R** and exact reconstruction over **original X** yielded `q=1`, except `q[2^n−1]=0`. All **161,571,496 q entries**, recorded hashes, histograms, flag totals and gate positions matched; total marked labels were **161,552,778**.

All 116 LP certificates and their application to every RESET passed. Eight-atom endpoint priors attain numerators `[7,7]` for n=3 and `[7,8]` otherwise, with denominator `D=8·2^r`. Factored certificates exactly encode
`H = log₂(D^D / (k^k(D−k)^(D−k)))/D`.
Probability ordering lies within `[0,½]`, where binary entropy is increasing, establishing the registered cap comparison without expanding huge products.

Chunk lists, RESET-count formulas, ceilings and transition counts all matched. The largest witness, n=16,r=8, has width 60, chunks `[8,8]`, **8,757 gates, 510 RESETs and 146,918,080,512 transitions**.

The independent n=128,r=8 **WORDONLY** reconstruction matched **69,937 new gates versus 100,617 old gates**, both width **284**. Neither was input-executed. The comparison retains the same source/output/cap contract. Masked counts are worse in **64 FULL cases**, which is permitted by registration.

Byte identities, attempt `df30cbd6-d950-4a7b-b8cf-4192f2eaa574`, inventory, pre-run environment and historical root-observed exit-0 receipt agree. FULL sources bind to committed `0d6dd939`; they also match current `91a882ba`. Smoke’s source/launcher absence at `23335c2` is correctly recorded and separately byte-bound. CPython 3.13.7, NumPy 2.3.5, psutil 6.1.1, module paths, 24 logical cores and enabled assertions match.

Verified SHA-256 identities:

```text
FULL artifact  3965acdfaaad10007e28e5b5e2d9e69d63458d70182bcbedb0502c4d21e6d7cc
Environment    e0bb7b845d3cbaff0d60931dfbf605552409e148a4ebb25da0f13fb433874d94
Exit receipt   feeb6a173ed12e0c9d563446ca1c4b3ed2dd0a0665a958519323655283f650ad
Protocol       cc1b6b47ed93a5065cb0485050e5e5e0ffbb60992970264f54bf7948bb27ced2
Masked source  fe926d6922ebaacdeee9780b4c248e0b7b0ac7f6e99ca509e6087b5c4dd1ab1d
Launcher       ed8fd5aa87af4951d748ca6fcfeabef70bb3ca07a15073e22c534bc05be1cfe7
Independent result
               f4d025b2d3641f3d8377a6647db8c9214f4c4de590a56adc3ab9f21ebee737b8
```

Independent evaluation took **12.070 seconds**, with maximum packed-plane storage **125,829,120 bytes** and peak working set **194,437,120 bytes**. These are reviewer resources, not circuit timing evidence. No edits, agents, GPU use, intermediate class-partition enumeration or runner FULL execution occurred.

This closes the registered finite raw-admission gate only. Universal optimality remains separately theorem-reviewed. Historical observer, interpreter, libraries and filesystem remain trusted control machinery. Preparation/authentication, physical heat, primary novelty, end-to-end bit work and mission-completion gates remain unadmitted; terminal Gate 2 remains **NO**.