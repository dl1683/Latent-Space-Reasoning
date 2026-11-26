from dataclasses import replace

import numpy as np
import pytest

from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace, _replay_source_transport, _transport_event
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory, StaleCorpusRevision
from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.recurrent_world import RecurrentWorld
from experiments.memory.role_validity import RoleSupersession
from experiments.memory.verified_source_tool import VerifiedSourceTool
from experiments.memory.world_transport import SourceTransportPolicy


def _publish(workspace, changes):
    return workspace.publish(workspace.prepare_changes(workspace.snapshot(), changes))


def _transport_world(tmp_path):
    axes = np.eye(8)
    records = {
        "entry": b"Start the deployment review at the local release profile.",
        "calibration-a": b"Local release profile activations transport to the service profile.",
        "calibration-b": b"Service profile activations transport to the rollout profile.",
        "correction": b"Service profile activations transport to the quarantine profile.",
        "other-law": b"A separate profile transports to itself.",
        "incomplete": b"The right-hand calibration is missing its destination.",
        "rollout": b"Rollout is permitted. The audit remains incomplete.",
        "changed-rollout": b"Rollout is prohibited. The audit is now complete.",
        "quarantine": b"Quarantine is required. The audit is complete.",
        "local": b"A separate local read stays available.",
        "midpoint": b"The local review advanced to its intermediate state.",
    }
    laws = {
        records["calibration-a"]: (0, 1),
        records["calibration-b"]: (1, 2),
        records["correction"]: (1, 3),
        records["other-law"]: (6, 6),
    }
    observations = {
        records["entry"]: 0, records["rollout"]: 2, records["quarantine"]: 3,
        records["local"]: 6, records["changed-rollout"]: 7, records["midpoint"]: 1,
    }
    calls = []

    def encode(raw):
        calls.append(raw)
        scope = (0, len(raw))
        if raw == records["incomplete"]:
            return (BindingMention("transport_entry", axes[1], 1., scope),)
        if raw in laws:
            entry, exit = laws[raw]
            return (BindingMention("transport_entry", axes[entry], 1., scope),
                    BindingMention("transport_exit", axes[exit], 1., scope))
        return (BindingMention("record", axes[observations[raw]], 1., scope),)

    frame = CorpusBindingMemory.prepare(("record", "transport_entry", "transport_exit"), axes, {}, {},
                                        encode=encode, encoder_version="transport-fixture", min_similarity=1., gap=0.)
    path = tmp_path / "transport.sqlite"
    owner = JournaledCorpusWorkspace(path, frame, create=True)
    base = _publish(owner, {name: records[name] for name in ("entry", "calibration-a", "calibration-b",
                     "other-law", "rollout", "quarantine", "local")})
    policy = SourceTransportPolicy(("calibration-a",), ("calibration-b", "other-law", "rival"), ("entry",),
                                   ("rollout", "quarantine"), gain=2., max_terms=1)
    values, carry, keys = np.zeros_like(axes), np.zeros_like(axes), axes.copy()
    values[2] = 2. * (axes[4] - axes[5])
    values[3] = -2. * (axes[4] - axes[5])
    values[7] = values[3]
    carry[6] = axes[6]
    keys[7] = axes[2]
    kernel = FeedbackKernel(query=30. * axes, carry=carry, observation_key=keys, observation_value=values,
                            feedback_key=axes, feedback_value=np.zeros_like(axes), max_steps=2)
    world = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=4,
                           feedback=True, transport_policy=policy)
    return owner, frame, base, world, records, calls, axes


def _read(world, bank, seed):
    return world.reason_joint(seed, steps=2, max_hypotheses=1, null_logit=8., margin=1., memory=bank)


def test_query_blind_transport_is_published_before_late_readouts_and_survives_restart(tmp_path):
    owner, frame, base, world, records, calls, axes = _transport_world(tmp_path)
    old = world.prepare(base, packed=False)
    published = world.maintain_transports(owner, base)
    assert published.revision == base.revision + 1 and published.source_revision == base.source_revision
    assert published.memory is base.memory and not published.changed_sources
    assert world.maintain_transports(owner, published) is published
    assert len(calls) == 7

    # The ordinary late task and its readout directions arrive after publication.
    seed = .8 * axes[0] + .1 * axes[5]
    enabled = world.prepare(published, previous=old, packed=False)
    ablated = world.prepare(published, previous=enabled, packed=False, operators=False)
    assert enabled._sources is ablated._sources and enabled.spans == ablated.spans == old.spans
    assert enabled.projected_rows == enabled.packed_rows == 0
    late = _read(world, enabled, seed)
    assert late.read(axes[4], floor=.8).status == "supported"
    assert late.read(axes[5], floor=.8).status == "opposed"
    assert _read(world, ablated, seed).read(axes[4], floor=.8).status == "missing"
    assert _read(world, old, seed).read(axes[4], floor=.8).status == "missing"
    branch, = late.branches
    assert tuple(w.span.source_id for w in branch.witnesses) == ("entry", "rollout")
    used, = branch.transport_uses
    assert used.publication_sha256 == published.journal_sha256 and used.source_row and used.task_row
    assert tuple((w.span.source_id, w.ordinal) for w in used.evidence) == (
        ("calibration-a", 0), ("calibration-a", 1), ("calibration-b", 0), ("calibration-b", 1),
    )
    for witness in used.evidence:
        assert witness.span == base.memory.source_mention_span(witness.span.source_id, witness.ordinal)
        assert witness.generation == base._source_revisions[witness.span.source_id]
    np.testing.assert_array_equal(_read(world, enabled, axes[6]).branches[0].hidden,
                                  _read(world, ablated, axes[6]).branches[0].hidden)
    assert not _read(world, enabled, axes[6]).branches[0].transport_uses
    premature = world.reason_joint(seed, steps=1, max_hypotheses=1, null_logit=8., margin=1., memory=enabled)
    assert premature.branches[0].source_hidden[2] > .8
    assert premature.read(axes[2], floor=.8).status == "unresolved"
    assert world.continue_joint(premature, steps=1, max_hypotheses=1).read(axes[4], floor=.8).status == "supported"

    changed_pin = _publish(owner, {"rollout": records["changed-rollout"]})
    changed = world.prepare(changed_pin, previous=enabled, packed=False)
    repaired = world.repair_joint(late, changed, max_hypotheses=1)
    assert repaired.read(axes[4], floor=.8).status == "opposed" and repaired.reused_writes == 1
    assert repaired.branches[0].transport_uses == (used,)

    def forbid_encoder(_):
        pytest.fail("transport replay re-entered source ingestion")

    reopened = JournaledCorpusWorkspace(tmp_path / "transport.sqlite", replace(frame, _encode=forbid_encoder))
    world_copy = RecurrentWorld(world._kernel, encoder_version=frame.encoder_version, max_observations=4,
                                feedback=True, transport_policy=world._transport_policy)
    before = reopened.snapshot_at(base.revision, max_events=10, max_payload_bytes=100000)
    history = reopened.snapshot_at(published.revision, max_events=10, max_payload_bytes=100000)
    assert _read(world_copy, world_copy.prepare(before), seed).read(axes[4], floor=.8).status == "missing"
    replay = _read(world_copy, world_copy.prepare(history), seed)
    np.testing.assert_array_equal(replay.branches[0].history, branch.history)
    assert replay.branches[0].transport_uses == (used,)
    assert len(calls) == 8


def test_transport_invalidates_on_authority_changes_and_requires_republication(tmp_path):
    owner, frame, base, world, records, _, axes = _transport_world(tmp_path)
    rival_handle = JournaledCorpusWorkspace(tmp_path / "transport.sqlite", frame)
    stale = rival_handle.snapshot()
    installed = world.maintain_transports(owner, base)
    with pytest.raises(StaleCorpusRevision):
        world.maintain_transports(rival_handle, stale)
    original = world.prepare(installed)
    live = _read(world, original, axes[0])

    conflict = _publish(owner, {"rival": records["correction"]})
    assert world.maintain_transports(owner, conflict) is conflict
    open_bank = world.prepare(conflict, previous=original)
    assert _read(world, open_bank, axes[0]).read(axes[4], floor=.8).status == "missing"
    with pytest.raises(ValueError, match="fresh frontier"):
        world.repair_joint(live, open_bank, max_hypotheses=1)

    incomplete = _publish(owner, {"rival": records["incomplete"]})
    assert world.maintain_transports(owner, incomplete) is incomplete
    with pytest.raises(ValueError, match="does not compile"):
        event = _transport_event(world._transport_policy, world._kernel_sha256)
        _replay_source_transport(event, incomplete, "a" * 64)

    corrected = _publish(owner, {"rival": None, "calibration-b": records["correction"]})
    assert _read(world, world.prepare(corrected), axes[0]).read(axes[4], floor=.8).status == "missing"
    republished = world.maintain_transports(owner, corrected)
    assert _read(world, world.prepare(republished), axes[0]).read(axes[4], floor=.8).status == "opposed"

    withdrawn = _publish(owner, {"calibration-b": None})
    assert world.maintain_transports(owner, withdrawn) is withdrawn
    restored = _publish(owner, {"calibration-b": records["calibration-b"]})
    assert _read(world, world.prepare(restored), axes[0]).read(axes[4], floor=.8).status == "missing"
    fresh = world.maintain_transports(owner, restored)
    later = _read(world, world.prepare(fresh), axes[0])
    assert later.read(axes[4], floor=.8).status == "supported"
    new_evidence = later.branches[0].transport_uses[0].evidence
    old_evidence = live.branches[0].transport_uses[0].evidence
    assert new_evidence[-1].span == old_evidence[-1].span
    assert new_evidence[-1].generation != old_evidence[-1].generation

    # An unauthorized relay can produce a score but cannot discharge the destination.
    false_relay = _publish(owner, {"rollout": None, "false-rollout": records["rollout"]})
    unresolved = _read(world, world.prepare(false_relay), axes[0])
    assert unresolved.branches[0].source_hidden[4] > .8
    assert unresolved.read(axes[4], floor=.8).status == "missing"

    # A numerical alias in a different source cannot activate the still-live term.
    alias = _publish(owner, {"local": None, "false-entry": records["entry"]})
    aliased = world.reason_joint(axes[0], steps=1, max_hypotheses=2, null_logit=8., margin=1.,
                                 memory=world.prepare(alias))
    assert {b.witnesses[0].span.source_id: bool(b.transport_uses) for b in aliased.branches} == {
        "entry": True, "false-entry": False,
    }
    changed_context = _publish(owner, {"entry": None})
    assert world.prepare(changed_context)._source_transports is None
    assert _read(world, world.prepare(changed_context), axes[0]).read(axes[4], floor=.8).status == "missing"
    other = RecurrentWorld(world._kernel._with_observation_value(np.zeros((8, 8))),
                           encoder_version=frame.encoder_version, max_observations=4,
                           feedback=True, transport_policy=world._transport_policy)
    with pytest.raises(ValueError, match="exact policy and kernel"):
        other.prepare(fresh)
    with pytest.raises(ValueError, match="scoped evidence"):
        world.reason(axes[0], steps=2, memory=world.prepare(fresh))


def test_task_only_transport_use_has_provenance_without_source_evidence(tmp_path):
    owner, frame, _, prior, records, _, axes = _transport_world(tmp_path)
    changed = _publish(owner, {"midpoint": records["midpoint"]})
    query, values, feedback = 30. * axes, np.zeros_like(axes), np.zeros_like(axes)
    query[3] = 30. * axes[0]
    values[6], values[1] = 2. * axes[1], 2. * axes[3]
    feedback[1] = 2. * axes[0]
    kernel = FeedbackKernel(query=query, carry=None, observation_key=axes, observation_value=values,
                            feedback_key=axes, feedback_value=feedback, max_steps=3)
    world = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=5,
                           feedback=True, transport_policy=prior._transport_policy)
    published = world.maintain_transports(owner, changed)
    result = world.reason_joint(axes[6], steps=3, max_hypotheses=1, null_logit=8., margin=1.,
                                memory=world.prepare(published))
    branch, = result.branches
    use, = branch.transport_uses
    assert use.task_row and not use.source_row and use.publication_sha256 == published.journal_sha256
    assert branch.hidden[2] > .8 and branch.source_hidden[2] == 0.
    assert result.read(axes[2], floor=.8).status == "undecided"


def test_effective_time_cutover_cannot_leave_an_overlapping_transport_live(tmp_path):
    owner, frame, _, prior, records, _, axes = _transport_world(tmp_path)
    entered = _publish(owner, {"replacement": records["correction"]})
    linked = owner.publish_program(owner.prepare_role_links(
        entered, ("transport_entry", "transport_exit"), min_similarity=1., compile_work_limit=32,
    ))
    world = RecurrentWorld(prior._kernel, encoder_version=frame.encoder_version, max_observations=6,
                           feedback=True, transport_policy=prior._transport_policy)
    published = world.maintain_transports(owner, linked)
    old = world.prepare(published)
    assert _read(world, old, axes[0]).read(axes[4], floor=.8).status == "supported"

    documents = {"calibration-b": records["calibration-b"], "replacement": records["correction"],
                 "other-law": records["other-law"]}
    leaves = tmp_path / "leaves"
    leaves.mkdir()
    for source, raw in documents.items():
        (leaves / (published.memory.source_digests[source] + ".txt")).write_bytes(raw)
    verifier = VerifiedSourceTool(tmp_path, published.memory.source_digests)
    spans = tuple(published.memory.source_mention_span(source, 0) for source in documents)
    rule = RoleSupersession(*spans, effective_at=20)
    cutover = owner.publish_program(owner.prepare_role_validity(published, (rule,), as_of=20, verifier=verifier))
    assert cutover.source_revision == published.source_revision
    with pytest.raises(ValueError, match="effective-time role rules"):
        world.prepare(cutover, previous=old)
    with pytest.raises(ValueError, match="effective-time role rules"):
        world.maintain_transports(owner, cutover)
    assert _read(world, old, axes[0]).read(axes[4], floor=.8).status == "supported"
