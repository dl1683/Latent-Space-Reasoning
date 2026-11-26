from dataclasses import replace
import json

import numpy as np
import pytest

from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace, _replay_source_transport, _transport_event
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory, StaleCorpusRevision
from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.recurrent_world import RecurrentWorld
from experiments.memory.role_validity import RoleSupersession
from experiments.memory.verified_source_tool import VerifiedSourceTool, sha256
from experiments.memory.world_transport import (
    DiscoveredTransportPolicy, SourceTransportPolicy, TransportAlignment, transport_action_rows,
)


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


def test_transport_discovers_across_opaque_source_ids_then_revises_without_a_task(tmp_path):
    _, scoped_frame, _, fixed, records, _, axes = _transport_world(tmp_path)
    wrong_role = b"The rollout profile is named in an unrelated directory."
    new_context = b"Review the local release profile for the newly filed case."

    def encode(raw):
        if raw == wrong_role:
            return (BindingMention("record", axes[2], 1., (0, len(raw))),)
        if raw == new_context:
            return (BindingMention("context", axes[0], 1., (0, len(raw))),)
        mentions = scoped_frame._encode(raw)
        if raw == records["entry"]:
            return (replace(mentions[0], predicate="context"),)
        if raw in (records["rollout"], records["quarantine"], records["changed-rollout"]):
            return (replace(mentions[0], predicate="relay"),)
        if raw == records["calibration-b"]:
            direction = mentions[0].vector.copy()
            direction[0] = -0.
            return replace(mentions[0], vector=direction), mentions[1]
        return mentions

    frame = CorpusBindingMemory.prepare(("record", "context", "relay", "transport_entry", "transport_exit"),
                                        axes, {}, {}, encode=encode, encoder_version="transport-discovery-fixture",
                                        min_similarity=1., gap=0.)
    path = tmp_path / "discovered.sqlite"
    owner = JournaledCorpusWorkspace(path, frame, create=True)
    policy = DiscoveredTransportPolicy("context", "relay", gain=2., max_terms=4, max_sources=20)
    world = RecurrentWorld(fixed._kernel, encoder_version=frame.encoder_version,
                           max_observations=8, feedback=True, transport_policy=policy)
    opening = _publish(owner, {"61b": records["entry"], "03a": records["calibration-a"],
                              "994": records["rollout"], "ea1": records["quarantine"],
                              "567": records["local"], "10d": records["other-law"],
                              "f39": records["midpoint"]})
    assert world.maintain_transports(owner, opening) is opening
    second = _publish(owner, {"2eb": records["calibration-b"]})
    first_publication = world.maintain_transports(owner, second)
    assert first_publication.source_revision == second.source_revision
    assert first_publication.journal_sha256 != second.journal_sha256
    assert world.maintain_transports(owner, first_publication) is first_publication

    # These ordinary records share a projected key, but the new observation
    # has a different coordinate and cannot discharge the transported relay.
    false_relay = _publish(owner, {"994": records["changed-rollout"]})
    assert world.maintain_transports(owner, false_relay) is false_relay
    ineligible_relay = _publish(owner, {"994": wrong_role})
    assert world.maintain_transports(owner, ineligible_relay) is ineligible_relay
    corrected_relay = _publish(owner, {"994": records["rollout"]})
    assert world.maintain_transports(owner, corrected_relay) is corrected_relay

    # A newly eligible source competes at the intermediate direction. A
    # retained earlier publication cannot be used on the edited pin.
    rival = _publish(owner, {"f39": records["correction"]})
    with pytest.raises(StaleCorpusRevision):
        world.maintain_transports(owner, first_publication)
    assert world.maintain_transports(owner, rival) is rival
    revised = _publish(owner, {"2eb": None})
    second_publication = world.maintain_transports(owner, revised)
    assert second_publication.journal_sha256 != revised.journal_sha256
    invalid = _publish(owner, {"101": records["incomplete"]})
    assert world.maintain_transports(owner, invalid) is invalid
    restored = _publish(owner, {"101": None, "2eb": records["calibration-b"], "f39": None})
    final = world.maintain_transports(owner, restored)
    benign = _publish(owner, {"61b": None, "303": new_context, "5ac": records["midpoint"]})
    assert world.maintain_transports(owner, benign) is benign

    # Neither the task encoding nor its readout direction was passed to
    # maintenance. Use the same computation and input for each intervention.
    seed, target = .8 * axes[0] + .1 * axes[5], axes[4]
    enabled = world.prepare(first_publication, packed=False)
    ablated = world.prepare(first_publication, previous=enabled, packed=False, operators=False)
    assert enabled._sources is ablated._sources
    results = (opening, second, false_relay, ineligible_relay, corrected_relay, rival, second_publication, invalid,
               final, benign)
    expected = ("unresolved", "unresolved", "missing", "missing", "supported", "unresolved", "opposed", "unresolved",
                "supported", "supported")
    for pin, status in zip(results, expected, strict=True):
        assert _read(world, world.prepare(pin, packed=False), seed).read(target, floor=.8).status == status
    wrong = _read(world, world.prepare(false_relay, packed=False), seed).branches[0]
    assert wrong.witnesses[-1].span.source_id == "994"
    assert wrong.transport_uses[0].publication_sha256 == first_publication.journal_sha256
    outcome = _read(world, enabled, seed)
    assert outcome.read(target, floor=.8).status == "supported"
    assert _read(world, ablated, seed).read(target, floor=.8).status == "missing"
    omitted = _read(world, None, seed)
    assert omitted.read(target, floor=.8).status == "memory_omitted"
    np.testing.assert_array_equal(omitted.branches[0].hidden, np.zeros(len(seed)))
    assert not omitted.branches[0].transport_uses
    assert _read(world, enabled, seed).read(target, floor=.8).status == "supported"
    for pin, publication, context, authority, terminal in (
        (first_publication, first_publication, "61b", "2eb", "994"),
        (second_publication, second_publication, "61b", "f39", "ea1"),
        (benign, final, "303", "2eb", "994"),
    ):
        branch, = _read(world, world.prepare(pin, packed=False), seed).branches
        use, = branch.transport_uses
        assert use.publication_sha256 == publication.journal_sha256
        assert [w.span.source_id for w in branch.witnesses] == [context, terminal]
        assert [w.span.source_id for w in use.evidence] == ["03a", "03a", authority, authority]
        for witness in (*branch.witnesses, *use.evidence):
            source = witness.span.source_id
            assert witness.span == pin.memory.source_mention_span(source, witness.ordinal)
            assert witness.generation == pin._source_revisions[source]

    # An ordinary fallback at a contested pin still has a complete numerical
    # read. The enabled arm must disclose the dispute on that same path.
    values = fixed._kernel._observation_value.copy()
    values[0] = 2. * axes[6]
    kernel = FeedbackKernel(query=30. * axes, carry=fixed._kernel._carry,
                            observation_key=fixed._kernel._observation_key, observation_value=values,
                            feedback_key=axes, feedback_value=np.zeros_like(axes), max_steps=2)
    fallback = RecurrentWorld(kernel, encoder_version=frame.encoder_version,
                              max_observations=8, feedback=True, transport_policy=policy)
    disputed = _read(fallback, fallback.prepare(rival), seed)
    held = disputed.read(axes[6], floor=.7)
    assert held.status == "unresolved" and not held.support and not held.opposition
    assert {w.span.source_id for w in disputed.uncertain_routes} == {"61b", "03a", "2eb", "f39"}
    no_operator = _read(fallback, fallback.prepare(rival, operators=False), seed)
    assert no_operator.read(axes[6], floor=.7).status == "supported"
    np.testing.assert_array_equal(disputed.branches[0].history, no_operator.branches[0].history)

    tampered = _transport_event(policy, world._kernel_sha256, discovery_sha256="a" * 64)
    with pytest.raises(ValueError, match="fingerprint does not match replay"):
        _replay_source_transport(tampered, second, first_publication.journal_sha256)

    def forbid_encoder(_):
        pytest.fail("recovery re-encoded a source")

    reopened = JournaledCorpusWorkspace(path, replace(frame, _encode=forbid_encoder))
    history = reopened.snapshot_at(first_publication.revision, max_events=20, max_payload_bytes=100000)
    recovered = _read(world, world.prepare(history, packed=False), seed)
    assert recovered.branches[0].transport_uses == outcome.branches[0].transport_uses
    np.testing.assert_array_equal(recovered.branches[0].history, outcome.branches[0].history)
    assert _read(world, world.prepare(reopened.snapshot()), seed).read(target, floor=.8).status == "supported"

    bounded = replace(policy, max_mentions=2)
    with pytest.raises(ValueError, match="mention allowance"):
        owner.maintain_source_transports(benign, bounded, kernel_sha256=world._kernel_sha256)
    assert owner.snapshot() is benign

    # An old publication's smaller capacity cannot prevent a valid current
    # policy from examining a newer pin and exposing the unresolved law.
    tight = replace(policy, max_sources=len(benign.memory.source_digests))
    tight_world = RecurrentWorld(world._kernel, encoder_version=frame.encoder_version,
                                 max_observations=16, feedback=True, transport_policy=tight)
    small = tight_world.maintain_transports(owner, benign)
    assert small is not benign
    larger = _publish(owner, {"2eb": None, "767": records["midpoint"], "768": records["midpoint"]})
    assert not small._source_transports.active_on(larger)
    assert _read(world, world.prepare(larger), seed).read(target, floor=.8).status == "unresolved"

    pending_bank = fallback.prepare(larger, packed=False)
    pending = _read(fallback, pending_bank, seed)
    assert pending.read(axes[6], floor=.7).status == "unresolved"
    retired = _publish(owner, {"03a": None})
    repaired_bank = fallback.prepare(retired, previous=pending_bank, packed=False)
    repaired = fallback.repair_joint(pending, repaired_bank, max_hypotheses=1)
    assert repaired.reused_writes == 2 and repaired.observation_scores == 0
    assert repaired.read(axes[6], floor=.7).status == "supported"
    assert not repaired.uncertain_routes
    np.testing.assert_array_equal(repaired.branches[0].history, pending.branches[0].history)


def test_scoped_discovery_reads_mixed_documents_by_original_ordinal(tmp_path):
    axes = dict(zip("ABCDEFX", np.eye(7), strict=True))
    sources = {
        "A": b"law A B\ncontext self\nlaw D E\n",
        "B": b"law B C\ncontext self\n",
        "C": b"relay C\n",
        "D": b"context self\n",
        "E": b"law E F\n",
        "F": b"relay F\n",
        "X": b"context self\n",
    }

    def encode(source, raw):
        offset = 0
        for line in raw.splitlines(keepends=True):
            role, *names = line.decode("ascii").split()
            tags = {"law": ("transport_entry", "transport_exit"), "entry": ("transport_entry",),
                    "exit": ("transport_exit",),
                    "law-relay": ("transport_entry", "transport_exit", "relay"),
                    "law-context": ("transport_entry", "transport_exit", "context")}.get(role, (role,))
            if role in ("law-context", "law-relay"):
                names.append(names[0] if role == "law-context" else names[-1])
            end = offset + len(line.rstrip())
            for tag, name in zip(tags, names, strict=True):
                yield BindingMention(tag, axes[source if name == "self" else name], 1., (offset, end))
            offset += len(line)

    # The same bytes in two different path-addressed sources receive their own coordinates.
    anchors = np.asarray(tuple(axes.values()))
    frame = CorpusBindingMemory.prepare(("context", "relay", "transport_entry", "transport_exit"),
                                        anchors, {}, {}, encode=encode, encoder_version="mixed-file-fixture",
                                        min_similarity=1., gap=0., source_aware=True)
    eye, zeros = np.eye(7), np.zeros((7, 7))
    values = zeros.copy()
    values[2], values[5] = 2. * axes["C"], 2. * axes["F"]
    kernel = FeedbackKernel(query=30. * eye, carry=None, observation_key=eye, observation_value=values,
                            feedback_key=eye, feedback_value=zeros, max_steps=2)
    path = tmp_path / "scoped.sqlite"
    owner = JournaledCorpusWorkspace(path, frame, create=True)
    opening = _publish(owner, sources)
    assert opening.memory.source_digests["D"] == opening.memory.source_digests["X"]
    assert not np.array_equal(opening.memory.source_mentions("D")[0].vector,
                              opening.memory.source_mentions("X")[0].vector)
    plain = DiscoveredTransportPolicy("context", "relay", 2., 4, 10)
    old = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=8,
                         feedback=True, transport_policy=plain)
    assert old.maintain_transports(owner, opening) is opening

    policy = replace(plain, scoped_clauses=True)
    world = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=8,
                           feedback=True, transport_policy=policy)
    published = world.maintain_transports(owner, opening)
    bank = world.prepare(published, packed=False)
    assert bank.projected_rows == 6 and bank.packed_rows == 0
    # Keep the two authority clauses in A but only its ordinal 2 in the ordinary bank.
    finding = _read(world, bank, axes["A"]).read(axes["C"], floor=.8)
    assert finding.status == "supported"
    branch, = finding.support
    assert [(w.span.source_id, w.ordinal) for w in branch.witnesses] == [("A", 2), ("C", 0)]
    use, = branch.transport_uses
    assert [(w.span.source_id, w.ordinal) for w in use.evidence] == [("A", 0), ("A", 1), ("B", 0), ("B", 1)]
    for witness in (*branch.witnesses, *use.evidence):
        span = witness.span
        assert span == published.memory.source_mention_span(span.source_id, witness.ordinal)
        assert witness.generation == published._source_revisions[span.source_id]
        assert span.span_sha256 == sha256(sources[span.source_id][span.start_byte:span.end_byte])
    assert _read(world, bank, axes["D"]).read(axes["F"], floor=.8).status == "supported"
    off = world.prepare(published, previous=bank, packed=False, operators=False)
    assert off._sources is bank._sources and off.projected_rows == 0
    assert _read(world, off, axes["A"]).read(axes["C"], floor=.8).status == "missing"

    swapped = _publish(owner, {"A": b"law A E\ncontext self\nlaw D B\n"})
    swapped = world.maintain_transports(owner, swapped)
    swapped_bank = world.prepare(swapped, previous=bank)
    assert _read(world, swapped_bank, axes["A"]).read(axes["F"], floor=.8).status == "supported"
    assert _read(world, swapped_bank, axes["D"]).read(axes["C"], floor=.8).status == "supported"
    _publish(owner, {"A": sources["A"]})

    dependent = _publish(owner, {"B": sources["B"] + b"relay C\n", "C": None})
    dependent = world.maintain_transports(owner, dependent)
    result = _read(world, world.prepare(dependent), axes["A"])
    assert result.branches[0].witnesses[-1].span.source_id == "B"
    assert result.read(axes["C"], floor=.8).status == "missing"

    shared_span = _publish(owner, {"B": b"law-relay B C\n"})
    shared_span = world.maintain_transports(owner, shared_span)
    shared_bank = world.prepare(shared_span)
    assert "B" not in shared_bank._sources
    assert _read(world, shared_bank, axes["A"]).read(axes["C"], floor=.8).status == "missing"

    self_context = _publish(owner, {"A": b"law-context A B\nlaw D E\n",
                                    "B": sources["B"], "C": sources["C"]})
    self_context = world.maintain_transports(owner, self_context)
    assert "A" not in world.prepare(self_context)._sources
    _publish(owner, {"A": sources["A"]})

    # A malformed, located clause competes with B's valid one without closing
    # the entire publication. Unrelated D remains usable.
    contested = _publish(owner, {"B": sources["B"] + b"entry B\n"})
    pending = world.prepare(contested, previous=bank, packed=False)
    assert pending._source_transports is None
    corrected = world.maintain_transports(owner, contested)
    revised = world.prepare(corrected, previous=pending, packed=False)
    a = _read(world, revised, axes["A"])
    assert a.read(axes["C"], floor=.8).status == "unresolved"
    assert ("B", 3) in {(w.span.source_id, w.ordinal) for w in a.uncertain_routes}
    assert _read(world, revised, axes["D"]).read(axes["F"], floor=.8).status == "supported"
    assert revised._sources["D"] is bank._sources["D"]

    # An exit scoped to a different line cannot be paired with the entry.
    split = _publish(owner, {"B": b"entry B\nexit C\ncontext self\n"})
    assert world.maintain_transports(owner, split) is split
    assert _read(world, world.prepare(split), axes["D"]).read(axes["F"], floor=.8).status == "unresolved"
    restored = world.maintain_transports(owner, _publish(owner, {"B": sources["B"]}))
    assert _read(world, world.prepare(restored), axes["A"]).read(axes["C"], floor=.8).status == "supported"

    def forbid_encoder(*_):
        pytest.fail("historical replay reopened the encoder")

    reopened = JournaledCorpusWorkspace(path, replace(frame, _encode=forbid_encoder))
    archive = reopened.snapshot_at(published.revision, max_events=20, max_payload_bytes=200000)
    replayed, = _read(world, world.prepare(archive), axes["A"]).branches
    assert replayed.transport_uses == branch.transport_uses
    np.testing.assert_array_equal(replayed.history, branch.history)
    assert reopened.snapshot().journal_sha256 == restored.journal_sha256


@pytest.mark.parametrize("route_changes", [True, False], ids=["changed-witness", "same-witness"])
def test_reachable_midpoint_changes_the_joint_suffix_after_a_discovered_transport(tmp_path, route_changes):
    identity = np.eye(9)
    axes = dict(zip(("A", "B", "C", "D", "E", "F", "G", "H", "answer"), identity, strict=True))
    sources = {
        "case": b"context A",
        "first": b"law A B",
        "second": b"law B C",
        "completion": b"relay C",
        "donor": b"record D",
        "donor-mid": b"record E",
    }
    if route_changes:
        sources.update(positive=b"record F", opposite=b"record G")
    else:
        sources.update(common=b"record H", opposite=b"record G")

    def encode(raw):
        role, *symbols = raw.decode("ascii").split()
        predicates = ("transport_entry", "transport_exit") if role == "law" else (role,)
        return tuple(BindingMention(predicate, axes[name], 1., (0, len(raw)))
                     for predicate, name in zip(predicates, symbols, strict=True))

    frame = CorpusBindingMemory.prepare(("context", "relay", "record", "transport_entry", "transport_exit"),
                                        identity, {}, {}, encode=encode, encoder_version="midpoint-fixture",
                                        min_similarity=1., gap=0.)
    values, carry, keys = np.zeros_like(identity), np.zeros_like(identity), identity.copy()
    values[3], values[4], values[6] = 2. * axes["E"], 2. * axes["G"], -2. * axes["answer"]
    if route_changes:
        values[2], values[5] = 2. * axes["F"], 2. * axes["answer"]
    else:
        keys[7] = axes["F"]
        carry[2], carry[5] = 2. * axes["F"], 32. * axes["answer"]
        values[7] = -32. * .915 * axes["answer"]
    kernel = FeedbackKernel(query=30. * identity, carry=carry, observation_key=keys, observation_value=values,
                            feedback_key=identity, feedback_value=identity, max_steps=3)
    policy = DiscoveredTransportPolicy("context", "relay", gain=2., max_terms=4, max_sources=12)
    world = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=8,
                           feedback=True, transport_policy=policy)
    owner = JournaledCorpusWorkspace(tmp_path / "midpoint.sqlite", frame, create=True)
    opening = _publish(owner, sources)
    published = world.maintain_transports(owner, opening)
    assert published is not opening
    bank = world.prepare(published)

    # Tasks and their judgment direction are supplied only after publication.
    def prefix(seed, depth, memory=bank, *, magnitude=1.):
        return world.reason_joint(magnitude * axes[seed], steps=depth, max_hypotheses=1,
                                  null_logit=8., margin=1., memory=memory)

    recipient = prefix("A", 2)
    donor = prefix("D", 2) if route_changes else prefix("A", 2, magnitude=.4)
    route, = recipient.branches
    use, = route.transport_uses
    assert use.source_row and use.publication_sha256 == published.journal_sha256
    if not route_changes:
        assert donor.branches[0].witnesses == route.witnesses
        for input_scale, sign, branch in ((1., 1., route), (.4, -1., donor.branches[0])):
            expected_midpoint = np.tanh(2. * np.tanh(2. * input_scale))
            np.testing.assert_allclose(branch.source_hidden @ axes["F"], expected_midpoint, rtol=0., atol=1.e-14)
            expected_outcome = np.tanh(32. * (expected_midpoint - .915))
            assert sign * expected_outcome > .8
    direction = axes["answer"]
    params = dict(steps=1, max_hypotheses=1, direction=direction, floor=.8)
    result = world.compare_joint_state(recipient, donor, **params)
    assert (result.ordinary_status, result.hypothetical_status) == ("supported", "opposed")
    assert result.journal_sha256 == result.transport_sha256 == published.journal_sha256
    assert result.depth == 2
    assert result.recipient_cut_score == result.donor_cut_score == 0.
    assert result.recipient_prefix == route.witnesses
    assert result.donor_prefix == donor.branches[0].witnesses
    assert result.ordinary_paths[0][:2] == result.hypothetical_paths[0][:2] == route.witnesses
    actual_relays = result.ordinary_paths[0][-1].span.source_id, result.hypothetical_paths[0][-1].span.source_id
    assert actual_relays == (("positive", "opposite") if route_changes else ("common", "common"))
    assert result.first_route_change == (3 if route_changes else None)
    assert result.ordinary_block_evaluations == result.hypothetical_block_evaluations == 2
    source_only = world.compare_joint_state(recipient, donor, channel="source", **params)
    task_only = world.compare_joint_state(recipient, donor, channel="task", **params)
    assert source_only.hypothetical_status == "opposed" and source_only.hypothetical_paths == result.hypothetical_paths
    assert task_only.hypothetical_status == "supported" and task_only.hypothetical_paths == result.ordinary_paths
    ordinary = world.continue_joint(recipient, steps=1, max_hypotheses=1)
    assert ordinary.read(direction, floor=.8).status == "supported"
    same = world.compare_joint_state(recipient, recipient, **params)
    assert same.ordinary_status == same.hypothetical_status == "supported"
    assert same.ordinary_paths == same.hypothetical_paths and same.first_route_change is None
    operator_off = prefix("A", 3, world.prepare(published, previous=bank, operators=False))
    assert operator_off.memory._sources is bank._sources
    assert operator_off.read(direction, floor=.8).status == "missing"
    assert prefix("A", 3).read(direction, floor=.8).status == "supported"

    # The donor cannot take the outstanding relay requirement with it. A
    # transplant before the relay completes runs the changed numerical path
    # without closing the recipient obligation.
    early = world.compare_joint_state(prefix("A", 1), prefix("D", 1), **{**params, "steps": 2})
    assert (early.ordinary_status, early.hypothetical_status) == ("supported", "missing")
    assert early.hypothetical_paths[0][1].span.source_id == "donor-mid"
    assert early.hypothetical_paths[0][2].span.source_id == "opposite"
    assert early.first_route_change == 2

    with pytest.raises(ValueError, match="unresolved source obligations"):
        world.compare_joint_state(prefix("D", 1), prefix("A", 1), **params)
    uncertain = world.reason_joint(axes["D"] + (8. / 30.) * axes["A"], steps=1,
                                   max_hypotheses=1, null_logit=8., margin=1., memory=bank)
    assert uncertain.branches[0].witnesses == prefix("D", 1).branches[0].witnesses
    assert uncertain.uncertain_routes[0].span.source_id == "case"
    with pytest.raises(ValueError, match="unresolved source obligations"):
        world.compare_joint_state(prefix("D", 1), uncertain, **params)
    with pytest.raises(ValueError, match="prepared memory"):
        world.compare_joint_state(recipient, prefix("D", 2, world.prepare(published, operators=False)), **params)
    with pytest.raises(ValueError, match="same committed depth"):
        world.compare_joint_state(recipient, prefix("D", 1), **params)
    with pytest.raises(ValueError, match="outside the joint frontier"):
        world.compare_joint_state(recipient, donor, donor_branch=1, **params)
    with pytest.raises(ValueError, match="committed frontier"):
        world.compare_joint_state(recipient, replace(donor, branches=recipient.branches), **params)
    with pytest.raises(ValueError, match="different reasoner"):
        world.continue_joint(result, steps=1, max_hypotheses=1)


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


def _alignment_world(tmp_path):
    axes = np.eye(10)
    first_held = (axes[0] + axes[1]) / np.sqrt(2.)
    first_image = (axes[2] + axes[3]) / np.sqrt(2.)
    rotated_a = .8 * axes[2] + .6 * axes[3]
    rotated_b = -.6 * axes[2] + .8 * axes[3]
    second_image = .8 * axes[4] - .6 * axes[5]
    context_a = .6 * axes[0] + .8 * axes[1]
    context_b = -.8 * axes[0] + .6 * axes[1]
    relay_a = .96 * axes[4] + .28 * axes[5]
    relay_b = -.28 * axes[4] + .96 * axes[5]

    # Inputs are supplied observations. The two middle bases do not share a row.
    laws = {
        "first-a": (axes[0], axes[2]),
        "first-b": (axes[1], axes[3]),
        "second-a": (rotated_a, axes[4]),
        "second-b": (rotated_b, axes[5]),
        "first-check": (first_held, first_image),
        "first-check-2": ((-axes[0] + axes[1]) / np.sqrt(2.), (-axes[2] + axes[3]) / np.sqrt(2.)),
        "second-check": (axes[2], second_image),
        "second-check-2": (axes[3], .6 * axes[4] + .8 * axes[5]),
        "chain-check": (.8 * axes[0] + .6 * axes[1], axes[4]),
        "chain-check-2": (-.6 * axes[0] + .8 * axes[1], axes[5]),
        "wrong-chain": (.8 * axes[0] + .6 * axes[1], -axes[4]),
        "wrong-check": (axes[2], -second_image),
        "duplicate-fit": (axes[0], axes[2]),
        "swapped-a": (axes[0], axes[3]),
        "swapped-b": (axes[1], axes[2]),
        "shifted-a": (.8 * axes[2] + .6 * axes[6], axes[4]),
        "shifted-b": (-.6 * axes[2] + .8 * axes[6], axes[5]),
        "shifted-check": (axes[6], .6 * axes[4] + .8 * axes[5]),
    }
    observations = {
        "entry-a": context_a, "entry-b": context_b, "off-plane": axes[6],
        "relay-a": relay_a, "relay-b": relay_b, "relay-contrary": axes[9],
        "bad-relay": relay_a + .001 * axes[9],
    }
    encodings = {name: (("transport_entry", a), ("transport_exit", b)) for name, (a, b) in laws.items()}
    encodings.update((name, (("record", row),)) for name, row in observations.items())
    raw = {name: ("Scoped observation in " + name + ".").encode() for name in encodings}
    by_raw = {raw[name]: encoding for name, encoding in encodings.items()}

    def encode(data):
        return tuple(BindingMention(predicate, row, 1., (0, len(data))) for predicate, row in by_raw[data])

    anchors = np.stack(tuple({row.tobytes(): row for rows in encodings.values() for _, row in rows}.values()))
    frame = CorpusBindingMemory.prepare(("record", "transport_entry", "transport_exit"), anchors, {}, {},
                                        encode=encode, encoder_version="alignment-fixture", min_similarity=.99, gap=0.)
    owner = JournaledCorpusWorkspace(tmp_path / "alignment.sqlite", frame, create=True)
    base = _publish(owner, {name: data for name, data in raw.items()
                           if name not in ("wrong-check", "duplicate-fit", "bad-relay", "swapped-a", "swapped-b", "relay-contrary",
                                           "shifted-a", "shifted-b", "shifted-check", "wrong-chain")})
    checks = TransportAlignment(("first-check", "first-check-2"), ("second-check", "second-check-2"),
                                ("chain-check", "chain-check-2"))
    policy = SourceTransportPolicy(("first-a", "first-b"), ("second-a", "second-b"),
                                   ("entry-a", "entry-b", "off-plane"), ("relay-a", "relay-b", "relay-contrary"),
                                   gain=2., max_terms=2, alignment=checks)
    values = 2. * np.outer(relay_a - relay_b - axes[9], axes[7])
    keys = axes.copy()
    keys[9] = relay_a
    kernel = FeedbackKernel(query=30. * axes, carry=None, observation_key=keys, observation_value=values,
                            feedback_key=axes, feedback_value=np.zeros_like(axes), max_steps=2)
    world = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=6,
                           feedback=True, transport_policy=policy)
    return owner, frame, base, world, raw, observations, axes


def _aligned_read(world, bank, seed):
    return world.reason_joint(seed, steps=2, max_hypotheses=2, null_logit=22., margin=1., memory=bank)


def test_query_blind_alignment_reaches_relays_from_unseen_directions(tmp_path):
    owner, frame, base, world, raw, rows, axes = _alignment_world(tmp_path)
    exact = RecurrentWorld(world._kernel, encoder_version=frame.encoder_version, max_observations=9,
                           feedback=True, transport_policy=replace(world._transport_policy, alignment=None))
    assert exact.maintain_transports(owner, base) is base
    published = world.maintain_transports(owner, base)
    assert published.revision == base.revision + 1 and published.source_revision == base.source_revision
    assert len(published._source_transports.operations) == 2
    enabled = world.prepare(published, packed=False)
    ablated = world.prepare(published, previous=enabled, packed=False, operators=False)
    assert ablated._sources is enabled._sources

    # A direct, warm numerical composer gets the same four paired fit rows.
    maps = []
    policy = world._transport_policy
    for sources in (policy.first, policy.second):
        pairs = [tuple(mention.vector for mention in base.memory.source_mentions(source)) for source in sources]
        entry_rows, exit_rows = (np.stack(column) for column in zip(*pairs, strict=True))
        maps.append(np.linalg.lstsq(entry_rows, exit_rows, rcond=None)[0])
    direct = maps[0] @ maps[1]
    for entry, relay, status in (("entry-a", "relay-a", "supported"), ("entry-b", "relay-b", "opposed")):
        expected = rows[entry] @ direct
        matches = [source for source in policy.relays if source in base.memory.source_digests
                   and np.linalg.norm(rows[source] - expected) <= policy.alignment.tolerance]
        assert matches == [relay]
        result = _aligned_read(world, enabled, rows[entry])
        assert result.read(axes[7], floor=.8).status == status
        branch, = result.branches
        assert tuple(w.span.source_id for w in branch.witnesses) == (entry, relay)
        use, = branch.transport_uses
        assert use.publication_sha256 == published.journal_sha256 and use.source_row
        assert {w.span.source_id for w in use.evidence} == {"first-a", "first-b", "second-a", "second-b"}
        assert {w.span.source_id for w in use.validation} == {
            "first-check", "first-check-2", "second-check", "second-check-2", "chain-check", "chain-check-2",
        }
        assert _aligned_read(world, ablated, rows[entry]).read(axes[7], floor=.8).status == "missing"

    unsupported = _aligned_read(world, enabled, rows["off-plane"])
    assert unsupported.read(axes[7], floor=.8).status == "missing"
    assert not unsupported.branches[0].transport_uses

    def forbid_ingestion(_):
        pytest.fail("aligned replay re-entered the source encoder")

    reopened = JournaledCorpusWorkspace(tmp_path / "alignment.sqlite", replace(frame, _encode=forbid_ingestion))
    recovered = _aligned_read(world, world.prepare(reopened.snapshot()), rows["entry-a"])
    original = _aligned_read(world, enabled, rows["entry-a"])
    np.testing.assert_array_equal(recovered.branches[0].history, original.branches[0].history)
    assert recovered.branches[0].transport_uses == original.branches[0].transport_uses

    actions = transport_action_rows(published._source_transports)
    record = json.loads(_transport_event(world._transport_policy, world._kernel_sha256, actions))
    record[4][0] = "a" * 64
    replay_bad = json.dumps(record, separators=(",", ":")).encode("ascii")
    with pytest.raises(ValueError, match="action fingerprint"):
        _replay_source_transport(replay_bad, base, "b" * 64)

    key, expected, output = actions[0]
    padded = np.frombuffer(bytes.fromhex(expected), dtype="<f8").copy()
    padded[8] = 5.e-14
    adjacent_actions = ((key, padded.tobytes().hex(), output), *actions[1:])
    event = _transport_event(world._transport_policy, world._kernel_sha256, adjacent_actions)
    compatible = _replay_source_transport(event, base, "b" * 64)
    assert transport_action_rows(compatible._source_transports) == adjacent_actions
    padded[8] = .01
    incompatible = ((key, padded.tobytes().hex(), output), *actions[1:])
    event = _transport_event(world._transport_policy, world._kernel_sha256, incompatible)
    with pytest.raises(ValueError, match="disagrees with replay"):
        _replay_source_transport(event, base, "b" * 64)


def test_alignment_rejects_a_wrong_check_a_singular_fit_and_a_near_relay(tmp_path):
    owner, frame, base, world, raw, rows, axes = _alignment_world(tmp_path)
    published = world.maintain_transports(owner, base)
    changed = _publish(owner, {"relay-a": raw["bad-relay"]})
    wrong_relay = _aligned_read(world, world.prepare(changed), rows["entry-a"])
    assert wrong_relay.branches[0].source_hidden[7] > .8
    assert wrong_relay.read(axes[7], floor=.8).status == "missing"
    assert wrong_relay.branches[0].transport_uses[0].publication_sha256 == published.journal_sha256

    competing = _publish(owner, {"relay-a": raw["relay-a"], "relay-contrary": raw["relay-contrary"]})
    conflict = _aligned_read(world, world.prepare(competing), rows["entry-a"]).read(axes[7], floor=.8)
    assert conflict.status == "unresolved"
    assert conflict.support[0].witnesses[-1].span.source_id == "relay-a"
    assert not conflict.opposition

    failed = _publish(owner, {"first-b": raw["duplicate-fit"], "relay-contrary": None})
    assert world.maintain_transports(owner, failed) is failed
    wrong_pairing = _publish(owner, {"first-a": raw["swapped-a"], "first-b": raw["swapped-b"]})
    assert world.maintain_transports(owner, wrong_pairing) is wrong_pairing
    # Each shifted stage still has an independently valid calibration, but the
    # first image contains a direction missing from the second domain.
    shifted = _publish(owner, {"first-a": raw["first-a"], "first-b": raw["first-b"],
                               "second-a": raw["shifted-a"], "second-b": raw["shifted-b"],
                               "second-check-2": raw["shifted-check"]})
    assert world.maintain_transports(owner, shifted) is shifted
    wrong_chain = _publish(owner, {"second-a": raw["second-a"], "second-b": raw["second-b"],
                                    "second-check-2": raw["second-check-2"], "chain-check": raw["wrong-chain"]})
    assert world.maintain_transports(owner, wrong_chain) is wrong_chain
    failed = _publish(owner, {"chain-check": raw["chain-check"], "second-check": raw["wrong-check"]})
    assert world.maintain_transports(owner, failed) is failed
    assert world.prepare(failed)._source_transports is None
    restored = _publish(owner, {"second-check": raw["second-check"]})
    assert world.prepare(restored)._source_transports is None
    fresh = world.maintain_transports(owner, restored)
    assert fresh is not restored and fresh.journal_sha256 != published.journal_sha256
    result = _aligned_read(world, world.prepare(fresh), rows["entry-a"])
    assert result.read(axes[7], floor=.8).status == "supported"

    def forbid_ingestion(_):
        pytest.fail("historical reconstruction re-entered source ingestion")

    reopened = JournaledCorpusWorkspace(tmp_path / "alignment.sqlite", replace(frame, _encode=forbid_ingestion))
    earlier = reopened.snapshot_at(published.revision, max_events=32, max_payload_bytes=2000000)
    original = _aligned_read(world, world.prepare(earlier), rows["entry-a"])
    assert original.read(axes[7], floor=.8).status == "supported"
    assert original.branches[0].transport_uses[0].publication_sha256 == published.journal_sha256

    copied = _publish(owner, {"off-plane": raw["first-a"]})
    assert world.maintain_transports(owner, copied) is copied
    with pytest.raises(ValueError, match="declared calibration source"):
        world.prepare(copied)
