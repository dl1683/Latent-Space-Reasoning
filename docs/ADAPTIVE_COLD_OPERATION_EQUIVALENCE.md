# Adaptive cold learning removes most of the apparent revision advantage

27 November 2026. Classification: mathematical foundation / negative public system-identification component result. Base f38fec53. No pretrained or held gate execution.

## Why this comparison is necessary

The previous [UTF-8 quotient result](REVISION_STABLE_OPERATION_EQUIVALENCE.md) saved 71–76% of revised membership queries against an unabstracted spectral cold fit, but its simple cold signature comparator failed. The relevant question is whether persistent structure beats a cold learner that can invent the same abstraction. This experiment isolates that question on a new executable family.

Prior art already includes state/alphabet co-refinement: [Steffen, Howar and Isberner (2012)](https://proceedings.mlr.press/v21/steffen12a.html), building on [Angluin (1987)](https://doi.org/10.1016/0890-5401(87)90052-6). A newer compositional approach also automatically learns alphabet decompositions ([Henry et al., 2025 preprint](https://arxiv.org/abs/2504.16624)). Do not present counterexample-driven alphabet refinement as novel. This is a deliberately simple local implementation, not a benchmark against LearnLib or CoalA.

## Executed experiment

Opaque membership interface: CPython 3.13.5's binascii.a2b_base64 with strict_mode true/false, and base64.b32decode default, casefold, and casefold plus map01=I. Input alphabet is restricted to 128 ASCII bytes; output is 1 for acceptance and 0 for rejection, not the decoded bytes. The underlying code, grammar and hidden states are not provided to the learner. The development and evaluation generators are codec-aware (valid encodings, local mutations, random words); both use the same generator with separate seeds. 600 generator iterations for development, 2,500 for evaluation, deduplicated. No evaluation feedback influences the fit.

Three arms: (1) full-alphabet observation-table learner; (2) adaptive quotient cold, starting with all symbols in one class; (3) the same adaptive quotient procedure starting from a prior learned equivalence partition exported from the opening full-alphabet fit. For the quotient arms, test each development word against its canonical form; on disagreement, replace its symbols one at a time until the binary outcome changes, and split the offending class by querying every member in that context. Refit the state machine on the representatives, using development counterexamples. All distinct oracle reads across class validation, splitting and fitting are counted. No exact equivalence oracle: agreement is on a finite panel. Export equal transition signatures as the next candidate prior; no global certification. Cold and warm use the same procedure, generator, and output obligation.

| Target | Full cold reads | Adaptive cold reads | Learned states |
| --- | ---: | ---: | ---: |
| Base64 strict | 10,307 | 4,764 | 7 |
| Base64 permissive | 8,528 | 4,674 | 7 |
| Base32 strict | 46,341 | 6,679 | 15 |
| Base32 casefold | 46,341 | 5,925 | 15 |
| Base32 casefold + map01 | 33,886 | 5,697 | 15 |

| Revision | Adaptive warm reads | Adaptive cold reads | Warm splits |
| --- | ---: | ---: | ---: |
| Base64 strict → permissive | 4,552 | 4,674 | 0 |
| Base64 permissive → strict | 4,573 | 4,764 | 0 |
| Base32 strict → casefold | 7,085 | 5,925 | 1 |
| Base32 strict → casefold + map01 | 7,177 | 5,697 | 1 |
| Base32 casefold → strict | 7,042 | 6,679 | 1 |
| Base32 casefold → casefold + map01 | 6,292 | 5,697 | 1 |
| Base32 casefold + map01 → strict | 7,073 | 6,679 | 1 |
| Base32 casefold + map01 → casefold | 6,276 | 5,925 | 1 |

All 18 arms had zero disagreements on their respective 14,313/15,298 unique evaluation words, and on the subsets excluded from their acquisition cache and development list (14,128–14,204 and 15,120–15,200, respectively). The test is exploratory and not a proof of correctness on all strings. Repeated edit directions reuse the same target pools, so the eight revisions are not independent samples.

**Decision:** On this family, the adaptive cold comparator eliminates most of the apparent benefit of structural retention. Warm saves only 2.6–4.0% of revised reads on the two Base64 changes; it costs more on every Base32 change (up to 26%). Warm Base32 splits temporarily into four classes while cold finds three, and the exported transition-equivalent partition collapses back to three. That is a plausible mechanism for extra work, not an isolated causal decomposition. No lifecycle advantage follows, even ignoring the opening full fit that created the prior. The exact codecs remain the practical incumbents. Counts exclude scoring-only oracle calls and no runtime/energy or storage comparison is claimed. This does not overturn the earlier UTF-8 result on its own benchmark; it invalidates the inference that a win against full-alphabet cold is adequate evidence.

## More important architectural boundary: equivalence is observation-dependent

The acceptance oracle deliberately projects away decoded content. In an actual CPython check, b'1AAAAAAA' decoded with map01=b'I' gives hex 4000000000; with map01=b'L' it gives 5800000000. Both accept. The map choices are indistinguishable by acceptance on this input, despite differing decoded output. In fact the two map01 choices admit the same Base32 alphabet and padding language in this standard-library setting. No amount of success predicting acceptance alone establishes preservation of the decoded content. A learned class is therefore conditional on the readout family, not a universal semantic identity.

For n concatenated valid unpadded eight-character Base32 blocks, there are 2^(40n) possible decoded contents; acceptance is 1 for all. An exact content reconstruction from retained state alone must distinguish those 2^(40n) cases, requiring at least 40n bits in the worst case (fixed-size state or maximum fixed-length representation). This is the ordinary counting bound, not an LSR-specific impossibility theorem. It does not prohibit a compact state for a narrower question family or selective source reopening.

WorldPin recommendation: preserve the readout/action family and witness provenance under which an equivalence was admitted. A new readout or rule can invalidate the abstraction even if every old observable remains unchanged. Treat newly relevant distinctions as an explicit maintenance problem, rather than assuming that a zero-error old predictor can support any future task. Avoid overinvesting in retaining cheap-to-relearn partitions.

**Next decision:** leave codec acceptance experiments here. Move to genuinely multi-output, corpus-derived operations with multiple independently chosen future readouts, partial access to raw evidence, and an equal-budget adaptive cold symbolic/text baseline. Optimize which distinctions the worker retains or reopens under a cost and uncertainty budget; then test causal reuse of those distinctions in late semantic tasks. The independent ordinary-reader qualification remains binding and unmodified. No learned-language or whole-corpus benefit established.

## Reproduction

Run python experiments/memory/adaptive_operation_quotient_pilot.py from this branch (standard library only). The executed file matches blob fb1fa89f17c01d5c2055e49d9886088acb8ddf5e. Python 3.13.5. All 18 arms completed and py_compile passed. Direct git checkout was blocked by DNS; no integration suite was run. The script uses finite counterexample checks, not a qualified exact equivalence teacher.
