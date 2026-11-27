# Independent ordinary competence gate and scalar-extension boundary

27 November 2026. Reviewed `1f6e8e06`, the four live commits since
`e63f8d78`, issue #6 and the five recent comments. There are no open
PRs; the four commit-comment endpoints returned empty. This responds
to [the requested independent calibration](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7213486096)
and [the installed observation control](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7213831437).

Classification: necessary experimental foundation (ordinary reader
qualification) and mathematical foundation (composition under partial
measurement). No learned execution, new authorization or release of
the previous independent operation packet.

## Four fresh ordinary-reader cases: frozen out of band

Independently authored two direct and two composed cases, including a
replacement, a withdrawal and lookup keys absent from the question.
Neither the cases nor the grading obligations are placed in GitHub.
The two private artifacts are separate; the owner can transfer the
input alone to the execution controller and retain the grader with an
independent custodian. The original operation packet is unchanged.

SHA-256 over the exact UTF-8 file bytes:

- `lsr_ordinary_calibration_input.json`: `c738edd6961215f9aecae0438495280f91253edd6a46aac10933602accac2fd5`
- `lsr_ordinary_calibration_PRIVATE_grader.json`: `d9f8962afb70f258f1f098c7af5d45d9b8fb665d7b3833ab1fadb2146ff3ffc6`

The grader is salted, contains the input digest, obligations and
required current source identities/hashes. Do not give it to the
runner, source index, worker, model, or research engine before the
first answers are frozen. The input is not an operation-discovery
panel and must not be added to the original held corpus.

### Frozen ordinary access contract

The input contains an opaque case ID, question, chronological
source-delta dictionaries and a requested read epoch for each case.
Each case is a separate source world. Convert text to UTF-8 bytes
and preserve null withdrawals. Use the public source feed writer
and TaskWorld.from_source_epoch with the previous delta and world
as appropriate. Open the last requested epoch, not a concatenation
of old and current documents. Start fresh model sessions; do not
show prior raw versions in current frames. An explicitly historical
task would require a different declared access contract.

Passage width 512 bytes, maximum 16 passages, 65,536 feed bytes
and 65,536 retained source bytes per case; TaskBudget steps=8,
candidates_per_search=1, evidence_bytes=2048, source_bytes=2048.
No memory, receiver or factor pin in the ordinary arm. The native
and incremental exact comparators must receive the same eligible
source bytes, identities and revisions, with equal source-read
rights for each corresponding task. Count their construction
and maintenance work separately. Do not let the ordinary arm
open a stale source or grant a latent arm an extra witness. This
is a prospective parity rule, not a measured cross-arm result.

Use run_iterative_text_competence without changing its prompt,
first-answer behavior or 4-call/4096-input/512-output ceilings.
No retries or grader-guided query repair. Pin actual model,
tokenizer, resource guard and generation adapter before a run.
Independent grading is all four complete, supported answers,
including the appropriate unresolved premise when evidence is
missing. A valid handle alone is not a semantic pass.

### Offline access feasibility (not competence)

I independently replayed the current BM25 formula (including
positive-only matching, sorted source IDs, seen-span exclusion
and top-one opening), source-delta semantics and exact JSON frame
content from the pinned implementation, with privately scripted
searches. The 14 live documents each fit a single 512-byte passage.
All four cases reached the required current evidence along a
feasible route, with 11 calls including the reserved final answers,
6 reads, and a maximum serialized system + newline + user payload
of 1,048 UTF-8 bytes before chat-template overhead. No actual model
outputs, tokenizer counts or full integration test are claimed.
These are path-existence checks, not proof an independent model
will discover the keys. Alternative search replies may exhaust
the budget or produce different frames; use the actual tokenizer
to preflight and enforce the input cap. The local environment
has about 4.6 GiB available RAM and no observed Hugging Face cache;
the registered resource and execution gates are not satisfied.

Keep the rubric and scripted paths in independent custody. The
next authorized experiment should run this gate once, and stop
the original held operation trial if it fails. Do not tune on its
first outputs.

## One scalar witness does not identify a full-vector extension

The new CPU control is valuable: an acquired measurement now
changes the operator actually used by the native reader, with
same-pin removal and restoration. The claimed measured-coordinate
result is not disputed. But in native_operand_probe.py,
_prepare_observed_operand copies the base-projected output,
replaces one measured coordinate, and explicitly treats all
unmeasured residual coordinates as zero. The installed row is then
passed to the vector selector. A full vector containing one
measured coordinate is not itself a fully witnessed vector.

Let the original operand span be U=span(e1) in R^2, with output
L(e1)=(1,0), and a prospective observation of the first coordinate
of L(e2) equal to 1. Consider two lawful completions:

    L+(a,b)=(a+b,b)
    L-(a,b)=(a+b,-b)

Both have exactly the same base observations and the same new scalar
witness. Both even satisfy a squared row gain cap of 2 for *each*
output row. The zero-residual extension implemented by the chosen
completion convention gives L0(a,b)=(a+b,0). On U, all three agree.
For any b != 0 the first coordinate is exactly right but the
second coordinate is wrong in both lawful worlds. A second
operation B(y)=y2 distinguishes the two, and composing B with
the zero-filled estimate gives neither correct nonzero answer.
This is a counterexample to inferring full-vector or composition
validity from scalar qualification, not a failure of the scoped
fixture and not a claim about the actual hidden semantics.

Exact independent integer enumeration: all 289 inputs in
[-8,8]^2 preserve the measured coordinate; all 272 off-span
inputs disagree in the unmeasured coordinate and both differ
from the installed zero-residual completion. Reproduction:

    for a in range(-8, 9):
        for b in range(-8, 9):
            p, m, z = (a+b,b), (a+b,-b), (a+b,0)
            assert p[0] == m[0] == z[0]
            assert (p == m == z) == (b == 0)
            if b:
                assert p[1]*m[1] < 0 and z[1] not in (p[1],m[1])

More generally let r be the new operand residual orthogonal to U,
and P the projection onto the set of measured output coordinates.
The observation identifies P L r, not (I-P)Lr. Under an unrestricted
linear residual law, a downstream linear map B is uniquely
identified from this observation for the new direction iff
B annihilates the admissible unmeasured variation space. With
additional qualified restrictions on L this space can be smaller.
Otherwise composition requires another independent observation,
a certified consequence interval, or abstention. A learned prior
can predict the missing values but is not verification.

### Decision for the next independent panel

Do not replace the frozen operation panel or spend another round
fitting the retired raw coordinate experiment. For a separate,
future role-sensitive panel, precommit an operation, measured
coordinate, role contrast and consequence thresholds before the
witness. Keep the two late operands distinct from the calibration
card and include a direct measured output, a composition using an
unmeasured distinction, and an operand outside the enlarged span.
Use two revisions whose composition differs while the measured
coordinate is identical. The exact control gets the same evidence:
where the law is underdetermined it must abstain rather than receive
hidden coordinates. Count all refusals, not just successful outputs.

Compare the scalar extension against a reused recurrent block and
joint-state revision. Freeze each prediction before the same witness
arrives. If a learned arm correctly predicts the unmeasured
consequence, then updates on an independently released witness,
and causal removal/restoration controls its first grounded answer,
that is stronger than copying the measured direction. If it
only succeeds with raw source reopening, report navigation rather
than retained computation. If none can transfer, stop and identify
whether the lost distinction is an encoder collision or an operator
failure, using disjoint calibration and a supplied-coordinate
control. No oracle role axes in the learned arm. Do not grade a
full-vector output as witnessed solely because its measured scalar
has zero uncertainty.

This preserves the full mission: workers can improve a persistent
world while carrying uncertainty into subsequent reasoning, instead
of introducing unobserved facts as if they were verified. The
highest priority remains a competent ordinary reader and the
first independently graded learned causal benefit. This document
does not establish edge economics or learned reasoning.
