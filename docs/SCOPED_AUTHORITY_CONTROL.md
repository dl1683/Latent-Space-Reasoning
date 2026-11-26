# Scoped authority is not source freshness

Development-only mathematical/CPU control responding to [the scope and
supersession review](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7185414466).
The world must maintain not only observations but the conditions under which
an observation may govern a later inference.

Three distinctions: **knowledge pin** (observed source generations),
**effective scope** (entity/relation/valid interval), and **authority**
(the evidence or declared policy licensing a replacement). Exact source
pointers establish bytes, not which assertion governs. Two terminal documents
can remain unchanged while a separate order reverses the governing assertion.
At the same pin the answer outside the order's interval can remain conflicted.

For a given scope s and effective time t, let A be the applicable assertions.
Let E be the directed edges (winner, loser) from applicable authorized orders.
Under this **declared fixture policy**, refuse if an order endpoint is missing
or the graph has a directed cycle; otherwise retain the zero-indegree
assertions, reporting agreement or conflict together with the orders and
suppressed assertions. This is not a universal supersession rule: no
issuer-priority hierarchy, exceptions or open-world completeness is inferred.
The fixed trusted issuer policy is supplied as source metadata, not learned.

## Executed development control

Run `node experiments/memory/scoped_authority_control.js`. The same function
was executed before publication in an isolated JavaScript runtime.
128 deterministic generated ASCII prose worlds, half in each opening order
direction, use two terminal documents with two subject clauses each.
A separate order governs opposite terminals for each subject, only during
[10,20). Correcting that order swaps the associations, leaving the terminal
documents unchanged. The eight stages include initial conflict, discovery,
correction, an untrusted opposite order, competing authorized order/cycle,
missing endpoint, withdrawal and restoration. The late reads check both
subjects at 5,10,15,19,20,25.

Both the scope index and full-scan reference receive identical source text,
trusted metadata, parsed events, revisions and effective-time inputs, not
answer labels. They use the same resolver; explicit expected verdicts
independently check its outputs for each fixture stage. This is not
independent human annotation or held-out evaluation.

| Measure | Count |
| --- | ---: |
| Generated worlds | 128 |
| Stage/scope/time checks | 12288 |
| Incremental vs full scan receipt parity | 12288 |
| Historical snapshot checks | 12288 |
| Forced latest-assertion errors | 9984 |
| Scope-blind order errors | 3072 |
| Time-blind order errors | 4608 |
| Incremental parsed ASCII bytes | 111840 |
| Full rebuild parsed ASCII bytes | 292248 |

The index stores versioned scope buckets and exact clause offsets. Historical
pins are in-memory snapshots, not a durable journal. The test checks the
returned pointers against the pinned in-memory bytes. Parse-byte counts
exclude storage, verification, allocation and query overhead. The full scan
is a reference, not the strongest performance baseline: the incremental
index already achieves correctness parity. The three deliberately broken
ablations are negative controls, not competent semantic baselines. No latent
representation, learned acquisition, edge performance or efficiency advantage
is established.

## Consequence for the whole-world evaluation

- Give **both** the latent arm and incremental join the same predicted
  validity events: scope, effective interval, winner and loser assertion
  IDs, exact order witness, uncertainty and issuer evidence. Charge inference
  of validity to each arm using it. Do not give the latent arm adjudicated
  authority labels while the join receives only terminal tuples.
- Cross the knowledge pin with effective time. Ask about an earlier effective
  time at a later pin. Keep terminals fixed, correct or withdraw only the
  governing source, add an unauthorized opposite order, and remove one
  endpoint. Freeze the authority policy and completeness universe independently;
  expose only ordinary evidence and the applicable public policy.
- Require provenance for the supersession itself, not just the surviving
  terminal. Preserve defeated and contrary assertions. Unknown authority,
  incomplete coverage and contradictory orders do not license false closure.
- Current `TaskInvestigator._reconcile` checks an open role receipt and
  verified citations, not that those citations actually license reconciliation.
  `run_episode` likewise checks citation existence, not entailment. This
  is a policy boundary, not a claim of semantic validation. A typed
  reconciliation receipt with scope and authority witnesses would make the
  evidence auditable. The separate evaluator must still judge sufficiency.

Next integrate predicted validity as versioned dependencies of the installed
latent operation. Test whether an authority-only correction changes a later
hidden path while both terminal observations stay fixed; compare with the
equal-information incremental join and same-pin operator-off intervention.
The new exact witness-span tool surface is a useful prerequisite. This
development control does not authorize a model run.
