# First construction proposal (2026-10-14): correction-closed rule state over a frozen stem

**Current-authority pointer:** This historical proposal's Gemma exclusion is superseded by
[amendment G4](LATENT_BRANCH_CAPACITY.md#amendment-g4-2026-10-15-admit-the-gemma-4-family).
Its model exclusions, budgets and local-cache wording do not govern current work;
[STATE.md](../STATE.md) and the 2026-10-15 owner authorization govern execution,
with model weights on Google Cloud only. The proposal below remains historical.

Status: PROPOSAL ONLY. Nothing is registered, launched or spent. Thresholds below are proposed and freeze only on
the owner's approval. Selection follows theory/RESEARCH_PROGRAM.md, "How we choose the next work".

## 1. Track, object, central bet

- Track 1 (latent memory), realised through the Track 2 model-construction option: a persistent,
  future-addressable state with a competent action interface, learned over a declared frozen stem.
- Central bet tested: one retained state keeps several unresolved consequences, a late-revealed rule can be applied
  to it, and a later correction revises only the affected consequences while preserving unaffected answers and the
  historical record.
- Why this question: it is the LSR central bet in its most direct form, and the last cycles tested it only
  indirectly (readouts, patches, probes). It also avoids the traps recorded in experiments/EXPERIMENTS.md
  (closing-position transplant, lexical value-token effects, name-query inadequacy).

## 2. Declared object (the contract)

- Stem: frozen Mistral-7B (open weights, non-Chinese, already in the frozen panel of theory/LATENT_BRANCH_CAPACITY.md;
  checkpoint revision and licence to be pinned before any launch). Gemma and OpenAI models stay excluded.
- Episode: k entities (pseudo-names, length-matched as in bindq.py) each with a random subset of 4 opaque predicate
  symbols (random syllable tokens per episode, so no semantics leak). Operations:
  1. INGEST: "E has p_j" facts.
  2. REVEAL RULE (after ingestion): one Horn rule "if E has p_a and p_b then E has p_c".
  3. QUERY: "Does E have p_c?" (current view).
  4. CORRECT: retract one ingested fact.
  5. QUERY-AFTER: affected entities (consequence depended on the retracted fact), unaffected entities, and a
     historical query "Before the correction, did E have p_c?".
- Construction: E_theta(H) pools the frozen stem's states over the ingested history into r retained slots Z (fixed
  size, charged). U_theta(Z, action) updates Z on REVEAL-RULE and CORRECT actions. R_theta(Z, query) answers.
  Learned: E, U, R (stem frozen). Action and query tokenisation is an explicit, charged component.
- Resource accounting reported separately: ingest tokens, stored state bytes, update cost, query cost, and the
  cost of any replay.

## 3. Arms (all on identical episodes)

| Arm | What it is | Role |
|---|---|---|
| C | Explicit versioned fact store plus rule evaluator, exact | Conventional comparator and task positive control |
| A | Learned E, U, R as in section 2 | The construction under test |
| A-store | E and R only, no U (state never revised) | Tests whether correction needs the updater |
| A-shuf | Full A with actions shuffled across episodes | Tests whether the action changes the consequence |
| B | Frozen stem plus LoRA reading the full text, equally exposed | Full-replay comparator, reports tokens |

## 4. Smallest decisive experiment

Phase 0 positive controls (must pass before any A claim):
- C reaches >= 0.99 on all query types.
- A retrieves ingested facts without any rule: >= 0.90 at k=1 and >= 0.80 at k=4.
- If A fails retrieval, the failure is a seat or competence failure. Change the stem layer or pooling once,
  pre-registered, and do not add modules.

Phase 1 decisive run: k in {1, 4}, 3 seeds, held-out entity names, held-out rule-predicate combinations.
Proposed thresholds (bootstrapped over episodes; intervals must exclude 0 where a difference is claimed):
- A post-correction accuracy on affected entities exceeds A-store by >= 0.20.
- A post-correction accuracy on affected entities exceeds A-shuf by >= 0.20.
- Unaffected retention (post-correction answers relative to pre-correction) >= 0.90.
- Historical-query accuracy >= 0.80.

Falsifier: the central bet is falsified for this construction if A is within 0.10 of A-store on affected queries
or within 0.10 of A-shuf, or if A fails the retrieval competence bar. Then the state is not computing the
correction, and the result is reported as a negative for this seat.

Phase 2 (only on Phase 1 pass): replicate on a second non-Chinese stem (OLMo 2 7B, in the frozen panel) and on a
held-out rule family. Only then propose a larger model or budget.

## 5. Compute ceiling and budget

- Window: 3 days, $2,000 total for LSR (owner's instruction, 2026-10-15). Rates from experiments/cloud configs
  (a2-highgpu-1g Spot, $2.21/h).
- Phase 0 smoke: cap $100 (about 3 GPU-hours).
- Phase 1 decisive: cap $700 (estimated 12-15 GPU-hours, including the B LoRA arm; caps protect against preemption
  and reruns, they are not expected spend).
- Phase 2 replication: cap $600, released only on Phase 1 pass.
- Reserve: $600, for pivot, preemption and price change.
- Engineering: cache one selected stem layer's token states once (about 20 GB for 8k episodes), train only the
  small E, U, R heads on the cache.

## 6. Stop and pivot rules

- Stop the whole line if total LSR spend would exceed $2,000.
- Instrument stop: if C fails its positive control, the task generator or comparator is wrong. Fix it and make no
  claim.
- Seat pivot: if A fails retrieval after the one pre-registered layer or pooling change, the state seat is wrong.
  Pivot the seat or receiver (Droid's direction), not the central bet.
- Falsifier stop: if the falsifier fires in Phase 1, stop this construction and report the negative with its
  diagnosis (which arm matched which). Do not rescue it by adding modules or changing the loss.
- Escalate only on Phase 1 and Phase 2 passes.

## 7. What this does not claim

- The task is synthetic. It illustrates the revision contract in the program, and it is not a registered benchmark.
- A pass would show a learned state that supports correction in this family. It would not show latent reasoning in
  general, a capacity law, or an efficiency advantage over an equally resourced conventional system.
- Comparison with conventional systems follows the program's rule: competent indexed versioned evaluation with the
  same evidence.

## 8. Decisions needed from the owner before launch

1. Approve this construction (or choose a different scoped first construction from the program's table).
2. Confirm the stem policy: Mistral-7B primary, OLMo 2 7B for replication, Gemma excluded.
3. Approve the caps above, or set lower ones.
4. Approve freezing the thresholds in section 4 for registration.
