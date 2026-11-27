# Archived native addressing: worker-to-task geometry, not a new model trial

**26 November 2026. Classification: mathematical / architectural diagnosis.**
This is an offline read of the retired p2 fixture, not a rerun, a prompt change,
a new fixture, or a learned capability result. It narrows a necessary
transition in Issue #6: a source-only worker's computation must be addressable
by a later question in the same evolving world. Do not tune this fixture.

## Finding

The existing record says the restored p2 publication contains one derived row
but the late reader sees an empty derived bank selection at both steps.
Replaying the **actual signed integer inner products**, including negative
scores that the selection receipt suppresses, identifies the immediate
mechanism:

| Archived p2 query | Highest source score | Score of the operation's seed source | Acquired operation address score | Address cosine |
| --- | ---: | ---: | ---: | ---: |
| Step 1 | 1,374,027 | 773,746 | -653,974 | -0.054825 |
| Step 2 | 801,315 | 354,807 | -203,070 | -0.016904 |

Both derived scores are strictly below the positive cutoff of 1. This is
neither a top-k displacement nor a tie/cap failure: the derived bank has one
row. Normalizing the same signed vectors by their positive lengths cannot
change either sign. Increasing top-k cannot admit them under the registered
positive-score rule.

The seed is source `0a4f82d1c39547bfa62e104d8c7b935e`, the current
record-to-sector mapping, which the already-opened evaluator identifies as
critical and the active source reader missed. It is fifth out of five source
keys on both archived task queries. All five ordinary source-key scores are
positive on both steps. The same two archived active and derived-off queries
are byte-identical, with identical source selections and output IDs; the
derived read did not alter this trajectory.

Contrast: the retained derived address has **positive** dot products on all
10 source-only maintenance query steps preserved at p2, including the
eight carried from older epochs (range 1,449,189 to
11,301,814). These are not held-out queries, and the original source-only
steps did not route against the acquired row. The comparison is geometric
only. It shows that this address is compatible with those worker-state
directions while incompatible in sign with these two later task-state
directions. It does not demonstrate that the value encodes the required
relation. In particular, the value has a different geometry from the address.

## Independent archive replay

Pin: [2faefadf](https://github.com/dl1683/Latent-Space-Reasoning/commit/2faefadf8ca2a1af3fbeae7e0ee4f24fb56de83e);
archive as fetched at [9a20409d](https://github.com/dl1683/Latent-Space-Reasoning/commit/9a20409d5ff1dde406fffbe21d1cd49d9630c159).
Inputs: the three manifests and rows.bin files and reads.jsonl in
[the retired addressing archive](../experiments/memory/results/native_world_addressing_p2_2026-11-26/).
The publication index declares identity query/key/value maps. The archived
selector uses signed int8 grid keys, int64 products, nearest-even rounding at
127 levels, top-k=1 and a strictly positive score floor.

An independent JavaScript decoder fetched the actual base64 payloads from
GitHub and parsed the little-endian float64 and signed int8 blocks in the
archived format. No backbone or evaluator executed. Results:

- All three binary layouts consumed exactly their declared sizes (62,464,
  46,080, 62,464 bytes). Width 1,024; five source rows per pin; derived row
  counts 1, 0, 1.
- 15,360/15,360 stored source key coordinates matched re-quantization with
  the identity map.
- 30/30 archived source-maintenance selections (10 per pin, including
  reused receipts) matched their selected row indices and integer scores.
- Both p2 task source maxima matched their receipts; both derived scores
  reproduced their empty receipts. Both active/derived-off queries matched.
- The p2-derived row scored positive on 10/10 archived p2 maintenance
  queries and negative on 2/2 archived p2 task queries.

To reproduce the arithmetic without a model, read each manifest's ordered
documents. For each document consume `n*1024*8` bytes as little-endian f64
source rows, then `n*1024` bytes as signed int8 keys. Each remaining
continuation contributes an address and value (each 1024 little-endian f64);
the p2 archive has only one continuation, so ordering is unambiguous.
Quantize that address by nearest-even rounding of `127*address`.
For each archived query hex string, reinterpret the bytes as signed int8
(not uint8), promote both vectors to int64, and take the dot product.
Maintenance queries are entry 4 of each step in manifest.attempted; task
queries are in the active read's source_steps and derived_steps. Compute
cosine as dot divided by both Euclidean norms of the quantized vectors.
Do not replay p0/p1 using p2 queries and label the result as an actual task
read.

## Can reweighting recover the two archived task inputs?

**27 November 2026. Exact read-only replay.** The restored publication
contains five ordinary keys and one acquired continuation key. I granted
a solver access to all six keys with arbitrary signed coefficients and no
shortlist. It still left 98.64% and 99.52% of the two task queries' L2
norms unexplained. The projection captures 2.69% and 0.96% of the squared
query norms. Six independent columns cover six directions of the
1,024-dimensional space at this pin.

| Active step | Query L2 | Acquired key alone | Seed source key alone | Best residual across all six |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3,548.19 | 3,542.85 | 3,539.37 | 3,500.06 |
| 2 | 3,573.35 | 3,572.84 | 3,571.51 | 3,556.08 |

Here $q$ is the saved integer query and $K$ holds the integer keys as
columns. The replay solves $(K^\top K)c=K^\top q$ over rational numbers
and compares the exact squared residual with the bound. Displayed
norms are rounded. Rounding each clipped input to the archived identity
grid incurs at most 16 L2 units over 1,024 coordinates. The residual
vector of a fit with $\|c\|_1\le4$ can therefore differ from its pre-rounding
equivalent by at most $16(1+4)=80$ L2 units. The all-six residuals
exceed that upper bound by over 43 times at both steps. The actual
prototype requires reconstruction within $10^{-7}$ per grid coordinate.
The archive predates that prototype. This replay grants all six keys
to each fixed query for a lower bound on its error. The second query
belongs to the original trajectory. A live change to the first read
could produce a different second query.

The same six-key projection exactly reconstructs one of the ten
preserved source-only query steps. It is the freshly computed seed
query that equals the retained address. The next step has a positive
score of 3,750,450 against that address and leaves a residual of
3,412.62. The worker did not route against the acquired row on either step. The other eight worker
steps were carried forward from earlier source epochs, and all exceed
the rounding bound against this bank. Those cross-epoch checks measure
geometry of carried traces, not reads executed at the restored pin.

Reproduce with `python -m experiments.memory.archived_native_span_control`.
It checks the pinned index, trace, manifest, source descriptor and
payload digests before using saved Q8 states. Only the JSONL trace
normalizes checkout line endings. It opens no model or
grading data. The result is confined to one exposed task, one restored
publication and one coordinate system. Delivering the acquired payload
by another address could still help the task. Query reconstruction
alone cannot judge that effect.
[Exact replay](../experiments/memory/archived_native_span_control.py)

## Decision and next discriminator

The independently observed failure of ordinary direct text (p2 and T8)
means addressing is not the only bottleneck. The eight-case learned
planner/action/final calibration remains the next authorized decision gate;
the newer matched verified read batching is useful but scripted only.
No new pretrained execution is authorized here.

For the first new, independently graded late-task panel after competence is
established, retain the signed scores (or bounded summaries of signed
extrema) of the relevant derived entries at each applied/declined step.
Separate four outcomes: no viable acquired operation; viable operation
unaddressed; addressed but no material hidden change; material hidden change
but no correct grounded answer. A complete-evidence or known-operation
diagnostic may be used separately to distinguish task-addressing from
operation/readout defects, not as an unhinted success. Do not fit a key,
threshold or sign rule to this retired p2 case.

If the worker-to-task addressing failure persists, the architecture needs a
source-only acquired operation with a task-compatible key, not another
maintenance fixture. One falsifiable design contrast is to separate the
learned key from the continuation value and qualify a source-derived
multi-context address on source-disjoint calibration. Hold the reader,
source information, proposal/fit budget, retained bytes and exact verification
opportunity fixed. Compare a source-anchor address (a cheap control) with
a learned task-compatible address and the current hidden-state address.
An anchor match is routing, not reasoning. A key must be frozen before the
late tasks; charge all key construction and additional reads. No credit for
routing alone: the same acquired operation must improve independently
graded answers on at least two distinct late questions, lose that improvement
under its same-pin removal and regain it on restoration. Compare warm
iterative text and incremental exact computation. If more than one derived
row exists, an all-derived-off toggle is not an attribution to a single
operation; use the singleton case or a correctly pinned targeted intervention.

For intuition only, the seed source key has a positive score on both fixed
p2 task queries, so it is a plausible cheap address control on a future
panel. Replacing a key could change the next hidden state and hence the next
query. We did **not** run that counterfactual, choose a new key on p2, verify
an operation's semantic content or demonstrate an answer improvement.

The result identifies a concrete sign incompatibility in one archived
example. It is not a model-wide distribution, a general incompatibility
theorem, a transfer-learning result, or evidence of an edge speedup.
