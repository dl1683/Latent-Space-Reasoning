# Whole-world planner: evidence-access feasibility before inference

**26 November 2026 — instrument diagnosis, not model or latent capability evidence.**
Pinned review: `ec787cad37c61f5b8eb74b8b97504cf2814a7445` (including the five-need v7 preflight and source-anchor intervention). No change to the registered packet, budgets, decoder, runtime authorization, or gold. The earlier one-need preflight and the later five-need stress preflight answer different questions.

## A second, non-token bottleneck

The five-need [contract](../experiments/memory/whole_world_planner_contract.py) has an evidence-sharing restriction absent from the common episode host and the [evaluation protocol](WHOLE_WORLD_EVALUATION.md). In `_receipt_index`, verified read IDs are available both globally and by need. But `schema_for_frame` passes only `read_ids_by_need` to `_finish_schema`, which constructs each finding with the enum of read IDs acquired **for that same need**. The host's `submission` instead calls `self._session._citations(finding["citations"])` without a need restriction for ordinary findings. The protocol explicitly says a finding may cite any verified read, including one acquired for another need; the evaluator judges relevance. This is distinct from the legitimate same-need constraint for opening search candidates and binding specialized role/joint receipts.

Let A be the number of evidence-action calls, S the search allowance, R the individually charged read allowance, and K the number of distinct registered need IDs whose findings can carry at least one verified citation. On the v7 search/read-only contract, starting with no receipts, each such ID requires one search call and one read call for that ID. Batches are confined to one need. Hence

```
K <= min(floor(A / 2), S, R).
```

With A=5, S=3, R=4, K<=2. This upper bound remains two even with unlimited prompt and output tokens, perfect selection, and a perfectly competent model. The cap is tight: search and read for two needs consumes four action calls. The fifth can search a third need but cannot also read for it. A later need can be informed by a prior read through `discovered_from`, yet the final schema still forbids citing that read unless acquired for the later need. The host rejects ordinary supports/defeats/conflicted findings with no verified citations. Citation-free unknowns with an open question and hypothetical findings are exceptions, not evidence-bearing resolution.

Scope: this is a bound on distinct *registered need IDs with nonempty citations* under this specific wire contract, not a bound on semantic obligations resolved by a single finding, nor proof that every calibration task is infeasible under every acceptable decomposition. It does not apply to a different contract allowing global citations or additional evidence operations. No held gold was accessed.

## Audit of all 22 recorded episodes

Read the archived [v7 ledgers](../experiments/results/whole_world_planner_preflight_2026-11-26_v7/) at the pinned commit via the repository API. All 22 (11 source pins, full and compact delivery) declared five need IDs, performed four successful byte-verified reads (two each under n1 and n2), cited only n1 and n2, and ended with five scripted unknown findings. Across the archive:

| Quantity | Count |
| --- | ---: |
| Completed stress episodes | 22 |
| Declared need instances | 110 |
| Need instances with at least one citation | 44 |
| Need instances with no citation | 66 |
| Scripted unknown findings | 110 |

These are structural counts, not quality or need-recall scores. The stress strings are not task-semantic needs. The archive already explicitly fails the registered token ceilings; this audit exposes an independent obstruction to interpreting a future token-only repair as instrument readiness.

## Verification

Static inspection of the exact pinned schema, host submission and documented citation rule. Enumerated every reachable pair of search/read need masks with a relaxed scheduler (all five IDs initially available, unlimited relevant candidates, one- or two-candidate same-need reads). This relaxation cannot reduce the actual reachable set. It found 121 mask pairs at up to five calls, maximum two read-bearing needs; at six calls the maximum became three. The two-need schedule is achievable under the actual initial declaration rule.

Reproduction (standard-library Python):

```python
from functools import lru_cache

@lru_cache(None)
def reach(k, s, r, S, R):
    states = {(S, R)}
    if not k:
        return frozenset(states)
    for i in range(5):
        b = 1 << i
        if s:
            states |= reach(k-1, s-1, r, S | b, R)
        if S & b:
            for n in (1, 2):
                if r >= n:
                    states |= reach(k-1, s, r-n, S, R | b)
    return frozenset(states)

assert len(reach(5, 3, 4, 0, 0)) == 121
assert max(r.bit_count() for s, r in reach(5, 3, 4, 0, 0)) == 2
assert max(r.bit_count() for s, r in reach(6, 3, 4, 0, 0)) == 3
```

A local standard JSON Schema check using isolated copies of the pinned `_string`, `_array`, `_finding_schema` and `_finish_schema` bodies also compared 27 combinations of empty, verified and nonexistent citation IDs across three findings. All 54 old/proposed-schema expectations passed. For a read ID 0 acquired only for n1, the old schema rejects n2 citing 0; the proposed global-ID enum accepts it and rejects nonexistent ID 1. This is a standard JSON Schema test, **not** a pinned LMFE parser or full host execution test. Empty-citation semantic validation remains the host's responsibility.

## Highest-leverage correction for the next version

Restore the registered semantics: make all verified read IDs at the same pinned episode available to ordinary final findings (the existing global `verified_read_ids` index is sufficient for the enum). Keep same-need candidate/read authorization, exact receipt/pin verification, role-specific provenance and independent assessment of whether each citation actually supports the claim. Do not treat sharing as independent corroboration or automatic relevance. No repeated source open should be needed merely to reuse the same receipt in another finding.

This is a proposal for the next coordinated contract revision, not a change to the frozen v7 hash. First test one read supporting two distinct declared needs, a nonexistent ID, wrong-pin provenance, and an unrelated but valid receipt that the independent semantic evaluator must reject. Run these through the actual pinned parser and host, and regenerate the token preflight for all arms. Include a separate capacity case with three distinct evidence-bearing needs, each requiring a distinct source; shared citation IDs alone do not guarantee retrieval access to disjoint evidence. If necessary, revise the evidence-action allowance or a matched multi-need search design rather than silently collapsing the obligation denominator.

The next scientific milestone remains unchanged: a competent ordinary task episode, then a query-blind acquired latent computation that causally helps two independent late questions before/after a real revision, with targeted withdrawal/restoration, memory omitted and warm text/exact controls at full cost. This audit removes a measurement confound, not a limitation of latent computation itself. No model weights, tokenizer rerun, LMFE execution, full test suite or evaluator gold were used.
