# Whole-loop information boundary: negative controls

Status: executed visible synthetic control; **evaluation infrastructure**, not a
learned latent-world result. Reviewed branch head `89733c2d` on 26 November
2026 and issue #6 comments 7194030546, 7194431595 and 7194886192. The scoped
Pint case is a useful mechanical control; it does not establish acquisition.

## Failure that must be excluded

Same-pin worker-off / restore and answer-changing revisions establish a
dependency, not the origin of that dependency or the use of hidden reasoning.
A route supplied by the fixture, a worker that sees the task early, an exact
verifier that fills the answer, or a source-only materialized answer cache can
all pass the naive causal gate. The last is a legitimate conventional
comparator, not a protocol violation. The first three must not be credited
as evidence of endogenous latent acquisition.

The existing visible `_scope_world` test in
[`test_unit_program.py`](../experiments/memory/test_unit_program.py)
explicitly maps case prose to coordinate axes, assigns observation keys and
values, registers context source IDs and supplies the late unit name and
readout. This is appropriate for testing transport and scope mechanics,
not for testing whether a worker discovers useful relationships. Do not
reuse that route construction as the held evaluation's encoder.

## Executed adversarial canary

Run `node experiments/memory/whole_loop_boundary_control.js`.
The standalone JS function was executed, including the final version after
strengthening each arm's off and restore calls. It uses 64 deterministic
randomized functional three-edge worlds (four possible opening states),
each with initial, correction, withdrawal and restoration pins. A separate
enumeration oracle grades the late reads and downstream donor interventions.
The code is public, and its tasks are not held out.

| Arm | Answers matching oracle | Off/restore on determinate reads |
|---|---:|---:|
| Source-only indexed transitions | 1024/1024 | 960/960 |
| Supplied route replay | 1024/1024 | 960/960 |
| Task-visible worker | 1024/1024 | 960/960 |
| Answer-filling verifier | 1024/1024 | 960/960 |
| Source-only materialized answer table | 1024/1024 | 960/960 |

An ordinary exact source-only index and a precomputed table are sufficient
here; no model or uniquely latent mechanism is needed. The negative arms
deliberately violate different information boundaries.

- The minimal task schema rejected all 1,024 explicit route fields.
- Across 256 pins and three alternative tasks per pin, all 768 comparisons
  of task-visible publications differed; the deterministic source-only
  publication remained unchanged across tasks and independent republishes.
- The answer-filling verifier's raw reasoner resolved zero cases; a
  check-only gate cannot promote an open proposal into a resolved answer.
- In 768 interventions, a reachable donor midpoint at the same pin and
  depth was substituted while holding task and supplied route fixed. The
  independent suffix oracle predicted the opposite result. The indexed
  transition matched 768/768; route replay matched 0/768.
- Corrections changed 128 answers; withdrawals opened 64; restorations
  restored 64 with a new generation; all 256 historical reads passed.

The off arm here removes the stand-in publication, the only knowledge
channel in this toy. The evaluator keeps the same source rows; the core
has no separate observation bank. Consequently this is not a test of the
production journal or of surgical removal while preserving a real
continuous observation bank. The midpoint substitution tests exact state
dependence, not semantic validity of arbitrary off-manifold neural states.
No learned acquisition, process isolation, model-grade task, text baseline,
wall-clock measurement or edge advantage is claimed. The schema check only
detects an explicit route field; paired publication equality cannot prove
noninterference against a malicious or output-invariant leak.

## Admission protocol for the real whole loop

1. Freeze the model, encoder, generic operator grammar, budget, source
   manifest and maintenance policy before independently selecting tasks.
   Worker process may receive the revision stream, not task files,
   task-specific labels, answer axes, coordinate maps, endpoint IDs,
   preselected witnesses, evaluation-specific source filters or evaluator
   callbacks. Generic schemas and all source-derived observations are
   allowed and recorded. Maintain the same broad source scope for controls.
   A generic task input may name its actual entity and quantity, not a
   fixture-derived intermediate route.
2. Separate the corpus/worker from the task/evaluator process and file
   permissions. Seal content-addressed publications and log their
   dependencies before releasing the task. An independent evaluator
   selects a fresh question and readout after publication. Fork the same
   source history under alternative unreleased task selections with fixed
   random seeds; compare publication hashes. Also audit construction of
   the encoder, initial seed and readout. Hash equality is necessary for
   deterministic fork comparisons, not sufficient proof of isolation.
3. At the same source pin and fixed core, input, raw source observation
   bank, budget and readout, remove only the acquired contribution, then
   restore its exact bytes. Keep memory-omitted as a separate arm. Test
   matched same-pin, same-depth reachable donor states with independently
   predictable downstream consequences; fix route metadata and disallow
   silently recomputing the removed state. A stale-pin rejection alone
   cannot demonstrate useful numerical/semantic intervention. Count
   unresolved, incorrect and correct outcomes separately.
4. Record the reasoner's proposed finding and required witnesses before
   exact verification. An external checker may reject or withhold; if it
   computes a new result and feeds it to the reader, label and charge that
   as an explicit exact executor arm. Do not let a checker give the model
   a task label via a retry or proof-completion side channel. Separate
   worker proposal, gate verification, hidden-state contribution and
   final source receipts in the trace.
5. Require current, revised, unrelated-edit and historical tasks as well
   as novel compositions whose particular routes were not supplied in
   acquisition or readout. Compare a warm exact incremental solver
   (including materialized answers when useful), a strong iterative
   text agent with equal access/caching, links-only, and memory-omitted.
   Charge every arm for acquisition, all attempted maintenance including
   unread regions, proof checking, stored history, inference and readout,
   source reopening and fallbacks. A text baseline need not replay the
   entire corpus to be a fair comparator.

The falsification rule: a supplied-route, task-visible or verifier-only
arm passing the whole-loop admission is a failure of the protocol; a
source-only exact arm matching the latent arm at lower cost is a failure
to establish a latent advantage, not a failure of source maintenance.
Alder and Pint remain visible calibration. Do not open the held panel
or change the earlier paused model status because this canary passes.
