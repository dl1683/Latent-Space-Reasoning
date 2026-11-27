# Preserve operation semantics, not every hidden coordinate

**Classification:** mathematical foundation and finite experimental brief for
issue #6. Reviewed remote head `3b25e031d26047fa70f58b60391fb09feb7170df`
on 27 November 2026. A later
deterministic task-receipt integration control is recorded in
[`RESULTS.md`](RESULTS.md#binding-composed-operator-uses-into-text-task-receipts-27-november-2026);
it does not qualify natural-language task addressing. No model execution was
run and no held outcome, late binding, grader or revision envelope was opened. This
responds to the independent-screen and routing/operand requests in
[#7212018618](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7212018618),
[#7212213956](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7212213956),
[#7212527391](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7212527391)
and [#7212718288](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7212718288).

## Architectural decision: semantic sufficiency instead of reconstruction

The archived shared-coordinate obstruction is real under its stated
assumptions. It does not establish that both extra raw hidden directions
carry useful operand information. Reconstructing a raw task vector is
stronger than preserving its operation-relevant content. Conversely,
discarding a single direction to make a geometric fit possible can
destroy a distinction needed after a source revision.

Let X be a linear span of admissible semantic inputs and let L_1,...,L_m
be the candidate linear consequence maps that must remain available
across the permitted revisions. Stack their outputs as S. A shared
linear operand encoder R:X -> R^d supports exact consequence maps
D_j after the encoder, i.e. L_j = D_j R on X for all j, **if and only if**

    ker(R|X) is contained in the intersection of ker(L_j|X).

Necessity: if R x = 0 then each L_j x = D_j 0 = 0.
Sufficiency: define D_j on im(R) by D_j(Rx) = L_j x; the kernel
inclusion makes this well defined and linear; extend to the rest of
the codomain arbitrarily. Therefore the minimum linear operand rank
is rank(S|X), attained by coordinates for the quotient of X by ker S.
No invertible map of the raw hidden space is required. An affine version
follows by subtracting a common reference input and retaining its
output offsets. This is not a bound on arbitrary nonlinear embedding
dimension. The nonlinear necessary and sufficient condition on a
finite domain is that every fiber of R has the same consequence
signature for all required maps; it says nothing about how cheaply
those maps can be learned or evaluated.

For revision, one operator's null space is not generally safe:
compressing under today's winning operator alone can lose a direction
needed by tomorrow's. An independently learned semantic projection
should preserve distinctions required by source-supported alternatives,
not reconstruct every latent coordinate or simply maximize retrieval
similarity. Additional operand directions are useful only if they
restore a missing consequence distinction.

## Verified public-opening coverage

The existing independent packet (public blob
`e60def056cf26032e5d3637a86cc6f5563f43478`) is a valid screen of the
two source-defined execution orders. It is not identification of an
unrestricted learned operator.

Use abstract 0/1 labels solely for the controller's public manual audit:
p = Primary, r = Reserve, o = requested override. Exchange then
override gives E(p,r,o)=(o,p); override then exchange gives
O(p,r,o)=(r,o). Each map has rank 2. Their stack has rank 3, and the
joint consequence signatures distinguish all eight binary inputs.
If a world retains a common role representation across both orders,
dropping either channel is unsafe even if today's order ignores it.

All four public event cards have p=r. Their input span has rank 2,
not 3. An incorrect role-binding rule E'=(o,r), O'=(p,o) matches the
corresponding intended map on every public card. It differs on every
off-diagonal input. These impostors violate the manual, so they are
not equally source-supported rival laws. They model a failure to
encode the manual's role distinction, not additional candidates the
evaluator should admit.

| Check | Exact result |
| --- | ---: |
| Public event input tuples | 4 |
| Neutral / order-separating cards | 2 / 2 |
| Rank of public input span | 2 |
| Joint signatures over the complete abstract cube | 8 / 8 |
| Intended/impostor matches on public cards (two maps) | 8 / 8 |
| Intended/impostor disagreements off the public diagonal (two maps) | 8 / 8 |
| Distinct public calibration + worker IDs | 10 |

Verified directly from the pinned public JSON and exhaustive binary
enumeration in an independent JavaScript check; no model, hidden
vectors or capsule data. Reproduce the central checks with standard
Python, from the repository root:

```python
import itertools, json, re
from pathlib import Path
packet = json.loads(Path("docs/independent_operation_opening.json").read_text())
cards = [c["source"] for c in packet["event_cards"]]
xs = [tuple(int(v == "READY") for v in re.search(
    r"Primary=(READY|HOLD); Reserve=(READY|HOLD)\. The override request sets Primary=(READY|HOLD)\.",
    c["text"]).groups()) for c in cards]
E = lambda x: (x[2], x[0])
O = lambda x: (x[1], x[2])
Ef = lambda x: (x[2], x[1])
Of = lambda x: (x[0], x[2])
cube = list(itertools.product((0, 1), repeat=3))
assert set(xs) == {(0,0,0), (0,0,1), (1,1,0), (1,1,1)}
assert {c["id"] for c,x in zip(cards,xs) if E(x)==O(x)} == set(packet["neutral_only_control_ids"])
assert all(E(x)==Ef(x) and O(x)==Of(x) for x in xs)
assert len({E(x)+O(x) for x in cube}) == 8
assert all(E(x)!=Ef(x) and O(x)!=Of(x) for x in cube if x[0]!=x[1])
ids = [s["id"] for s in packet["calibration_sources"] + packet["opening_sources"] + cards]
assert len(ids)==len(set(ids))==10
```

**Do not change the frozen packet or release its held content.** This
audit does not inspect the withheld inputs, say whether a particular
late task exposes an impostor, or downgrade a correctly scoped
two-family result. For a subsequent independent panel, include
off-diagonal role contrasts, input order/label permutations and
direct A, direct B and A-to-B compositions. The controller should
check semantic separation on disjoint calibration examples before
testing acquired operations; no role axes, target vectors, cohort
membership or evaluator coordinates may be injected into the learned
arm. A supplied-coordinate positive control is separate. Prefer
a small independently learned role projection over another general
rotation fit to the retired p2 queries. If projection loses a
distinction, compare an extra source-acquired direction; only
retain it if it changes independent grounded conclusions.

## Finite execution brief for the current held packet

Proposed ceilings for a *new explicit authorization*, not granted
execution rights or proven token feasibility. The repository's 0.6B
model preference and Rank 1 stop remain unchanged. No larger
checkpoint without owner approval. The 0/18 fresh reader result
rules out repeating the same direct full-text setup on those tasks.
Do not tune on them.

1. **Ordinary competence gate:** use four new, independently authored
   calibration tasks outside the held packet: two direct, two composed
   (one with a changed or missing premise). Try one frozen iterative
   text/RAG policy, at most four model calls/task, 4,096 input and
   512 generated tokens/call, no retries after grading. Require all
   four complete grounded answers including unresolved status where
   appropriate; record every refusal. This is a small operational
   qualification, not a general competence claim. If the policy fails,
   stop the held test; no latent advantage can be attributed.
2. **Custody / preflight:** the independent custodian receives the
   existing private capsule out of band. The worker sees only the
   source bodies and opaque IDs (three opening sources plus four
   event cards), with the three disjoint calibration sources available
   to the encoder before freezing it. Verify the exact source spans,
   tokenizer feasibility, checkpoint/license, RAM/thermal reserve and
   resource ceilings before loading weights. Do not give the worker
   the wrapper, release hashes, neutral list, gold or any late binding.
   A separate neutral-only run uses only its two cards, no outcomes.
3. **Source-only acquisition gate:** compare no maintenance, ordinary
   recurrence, source exclusion and bounded deflation, separately
   initialized at the same opening. Freeze a common source seed list
   and horizon. Limit each arm to 32 source seed attempts, including
   refused/repeated attempts, 4 latent steps/attempt and at most
   384 backbone calls for opening maintenance plus causal checks
   (stop at the first exceeded limit). Fix the deflation shift from
   disjoint calibration. If coverage requires more than the cap,
   report incomplete coverage; never quietly substitute an easier
   candidate. Preserve actual candidate IDs, source-row counts,
   external-source selections, state changes and work. If no arm
   acquires a viable source-supported operation, stop before releasing
   any outcome. A matched irrelevant outside input is a diagnostic,
   not a new headline arm. Source-only exact operation discovery is
   a warm comparator with the same source/caching rights.
4. **Event selection:** for each surviving arm require a complete
   prediction of both candidate outcomes from each eligible card,
   prospective commitment of selected card, predictions and opening
   pin, then release that exact event only from the independent
   custodian. Charge event prediction and verification within a
   separately fixed cap of 16 model calls/arm, 4,096 input and 512
   output tokens/call. The neutral-only arm must abstain, without
   spending a release. The proposer exits before resolver publication.
   Distinguish learned acquisition from geometry-only basis
   discovery. Require genuine source-induced state change, not just
   a new row or address hit.
5. **Late use and revision:** only after publication, release the two
   late tasks; preserve each arm's first answers. Test opening,
   correction, withdrawal and restoration, two tasks per stage.
   Historical-pin and unrelated-edit controls must preserve prior
   answers. At opening, run same-pin targeted removal, restoration,
   memory omission and fixed-later-observation wrong-role substitution.
   Include a warm exact control and competent text control given the
   same released evidence. Cap each reader arm at 24 task sessions
   (including interventions and history), at most four calls/session
   under the same per-call token limits; failed/aborted sessions
   consume their allocation. Cap each post-opening revision's
   maintenance separately at the opening maintenance limit. If a
   cap cannot cover the prespecified controls, stop before releasing
   the next stage. Keep raw evidence access fixed for the causal
   wrong-role comparison. Do not hand known operation IDs or operands
   to the ordinary unassisted reader; use a separately labeled
   diagnostic on a fresh calibration case for known-ID addressing.

Charge all retained state, source bytes, encodings, exact source
checks, rejected attempts, cold reopening, model calls/tokens,
elapsed time and peak memory, including controls; model-call ceilings
are not an edge-cost result. Freeze the actual runner, resource
limits and exact release order before execution. Any required
parser, multimodal encoder or baseline that cannot pass preflight
is a blocker, not a reason to improvise on the held packet.

**Success criterion:** a source-only worker acquires a reusable
computation before the tasks, the same operation improves both
independent late answers with complete provenance, targeted removal
loses the effect, restoration recovers it, and legitimate changes
alter its conclusions. If the conventional exact baseline is faster
at equivalent quality, the learned behavior might still support
the mission but does not establish edge economics. If the ordinary
receiver fails, no learned latent benefit comparison is admitted.
This packet alone cannot establish unrestricted operation learning,
whole-corpus completeness, or multi-operator composition. No new
software fixture or model result is claimed here.
