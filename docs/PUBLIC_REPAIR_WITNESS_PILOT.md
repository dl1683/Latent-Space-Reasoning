# A semantic witness must do more than rank plausible repairs

27 November 2026. Classification: mission-directed **component falsification** and architecture decision. Base 349312d1. Separate public diagnostic; no pretrained model, private input, grader, launch authorization or registered receiver was touched.

## Question and decision

The proposed inventor/adversary/witness loop needs a cheap resident verifier that can reject semantic damage before writing a repair into a persistent world. Before building another receiver, test a deliberately small, trainable candidate discriminator on actual independently curated natural-language/formula revisions. Is the cross-corpus transfer signal from alignment and structure strong enough to justify using it as an admission witness?

No. Even with both the old and the human-corrected candidate supplied, cheap learned scores are unreliable. Do not train another confidence head to serve as the witness. Its output can only prioritize investigation. Instead, train the proposer and adversary to generate *distinguishing consequences* and evidence obligations; publication must require independent evidence for the consequential changed interpretation. This is a design decision based on a narrow pilot, not a general claim about neural verifiers.

## Public data and controls

Source: [Brunello et al., Fixing FOLIO and MALLS, v2](https://arxiv.org/abs/2606.02837v2), [released repository](https://github.com/dslab-uniud/Fixing-FOLIO-and-MALLS/tree/bac88aa1ca01c4b2fcf7ca05dad0a2c5623b2588), license CC BY 4.0 for the data. Retrieved pinned UTF-8 JSONL via GitHub:

- FOLIO: 275 rows, 73 story groups; Git blob `e5428ca9a4efbe9460bc8a833d9e5f01ff3d92d9`.
- MALLS: 100 rows; Git blob `c42ef1680ce6b21cbaf33a429faa93644e011c37`.

Paths are `core/dataset/datasets/{folio,malls}/dataset_rawpairs.jsonl`. Public development data only, not an independent blinded evaluation; no data redistributed here. The previous document proposed paired annotations; this pilot makes that concrete without a new deterministic world extension.

Eligibility: annotation says original incorrect, not marked ambiguous, distinct old/new strings. Excludes correct originals (including normalization), ambiguous annotations, and 1 otherwise eligible FOLIO record(s) without a distinct candidate. This gives 79 FOLIO pairs from 41 story groups and 23 MALLS pairs. FOLIO and MALLS have different task distributions, and the same authors curate both. Cross-dataset holdout prevents sharing rows/stories within an arm but is not independence from the authors or proof of generalization. Pairs are not independent draws; no significance claim.

Only features of the natural language and candidate formula are used. No correction explanations, story IDs, flags, labels, ontology answers, revised-formula identity or solver outcomes enter the rank score. The label is the direction of each supplied training pair. Test-time candidate generation is privileged; the model never generates a repair or establishes that a repair is necessary.

Fit a ridge-regularized pairwise logistic ranker (1000 full-batch gradient steps, rate .05, ridge .1, training-only RMS scaling with .1 floor). Structure-only: formula length, identifier counts, bracket imbalance/depth, logical operator counts. Joint: add source/predicate stem overlap, length ratio, a single-implication phrase-order heuristic and counts/interactions of simple NL cue words with operators. No pretrained embeddings. Compare shorter and better-balanced formula baselines. Use opposite corpus for training in each direction, no in-corpus hyperparameter search. Evaluate strictly positive/zero/negative signed margin, ties separate. Balance is only a parenthesis check, not parser validity. No threshold chosen after seeing the test. The feature design is exploratory, not preregistered.

## Results

Counts are corrected preferred / tie / original preferred. All eligible pairs retained.

| Train -> evaluate | N | Formula structure | Joint source/formula | Shorter | Better parentheses |
|---|---:|---:|---:|---:|---:|
| MALLS -> FOLIO | 79 | 41/1/37 | 35/0/44 | 20/4/55 | 10/69/0 |
| FOLIO -> MALLS | 23 | 15/0/8 | 15/0/8 | 5/3/15 | 0/23/0 |

Restricting to pairs with both candidates parenthesis-balanced does not rescue transfer: FOLIO (69) structure 36/1/32 vs joint 31/0/38; MALLS (23) structure and joint both 15/0/8. On the FOLIO subset of 19 eligible pairs whose published valid solver label remains unchanged, joint prefers the repair only 8/19, structure 7/19 with one tie. The source-aware learner is not reliably better than the cheap structural control here. This is not a bound on a pretrained learner or the full candidate-invention architecture. It is not a fair head-to-head with the authors' stronger LLM-assisted audit pipeline, which should be retained as an explicit competitor.

A separate audit of the full FOLIO release counts 58/63 original-incorrect conclusions with identical old/new human NLI labels; 23/41 original-incorrect conclusions with valid old/new published Z3 labels have identical solver labels. These are annotation record counts and supplied solver outputs, not a fresh solver run. A fixed endpoint can miss known corrections; it is not a complete witness. Neither equality nor inequality of these labels alone establishes semantic faithfulness. The prior worked example was not an isolated curiosity in this dataset.

## Architecture change: witness a distinction, not a confidence score

Keep the proposer/adversary/witness split, but define the witness's unit as an intervention obligation rather than approval of a candidate module:

1. The proposer supplies two competing interpretations of the same source, not a preferred score. The adversary predicts a counterexample that makes them disagree about an observable downstream condition. Train on repaired/unchanged predictions; hide evaluation probe identities. Penalize redundant probes and predictions unsupported by the actual source.
2. The independent checker validates that the two candidates really disagree under the same assumed facts. A solver can establish *formal* disagreement, never semantic truth. A source witness or human adjudicator selects or refuses the interpretation. If both are consistent with the text, preserve alternatives rather than force a winner. Record interpretation generation, evidence and scope separately from source generation.
3. Admission becomes conditional: a locally generated latent operator may be used where the distinction was resolved; outside that region retain uncertainty and a recovery route. Reopen only the evidence needed for the contested distinction. Keep old pins, and requalify downstream operators when their dependency on the disputed interpretation matters. A local learned confidence ranker can schedule these checks, not replace them.
4. The actual resident computation should learn to transfer the *repair*, not the donor answer or source-specific proof. Test donor-fixed/receiver-varying and receiver-fixed/donor-varying interventions with same-pin ablation, as specified in the earlier design. A warm text-to-FOL compiler with incremental invalidation and the authors' LLM audit are the serious comparators. Count candidate-generation and witness/adjudication costs, all source access and storage.

Next experiment: build the public ambiguity-aware countermodel/probe dataset from pinned annotations and an independently checked solver (not this ranker), separating syntax failure, unsatisfiable theories, semantic disagreement and genuinely ambiguous readings. For each correction, produce novel consequences with old/new truth status, check them independently, then test cross-story operation transfer with a learned semantic encoder under an explicit separate development protocol. If the strongest symbolic or text repair arm reaches the same quality at equal or lower lifecycle cost, retire the latent repair mechanism for this workload. The frozen four-case competence gate still requires owner-authorized private input-only handoff and adequate RAM; do not reinterpret this public diagnostic as its result.

## Reproduction and verification

Run `node experiments/memory/public_repair_witness_pilot.mjs /path/to/pinned/folio.jsonl /path/to/pinned/malls.jsonl` with both raw pinned blobs (the CLI verifies Git blob hashes). The same pilot function was executed in V8 over retrieved exact blobs twice, reproduced the primary counts, checked finite margins, 204 swapped-order antisymmetry checks per execution (two modes x 102 pairs), and checked result denominators. Machine-readable counts: [public_repair_witness_pilot.json](../experiments/memory/results/public_repair_witness_pilot.json). Direct git checkout was DNS-blocked; no integration suite or Node CLI execution is claimed. No model inference, external grading, formal parser execution or runtime/edge superiority measured.
