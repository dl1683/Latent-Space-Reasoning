from hashlib import sha256
from itertools import product

import numpy as np
import pytest

from experiments.memory.binding_memory import BindingCapacityError, BindingGate, BindingMemory, BindingProof
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory, CorpusBindingRead


_GATES = {"release": BindingGate("disjoint_and", ("inspection", "permit"))}
_PREDICATES = ("inspection", "permit")


def _text_vector(name):
    digest = sha256(name.encode()).digest()
    return np.frombuffer(digest, dtype=np.uint8).astype(float) - 127.5


def _sensor(vector):
    return b"\x00" + np.asarray(vector, dtype="<f8").tobytes()


def _source_fixture(raw):
    if raw.startswith(b"\x00"):
        return [BindingMention("inspection", np.frombuffer(raw[1:], dtype="<f8"), .7)]
    name, predicate, grade = raw.decode().split("|")
    return [BindingMention(predicate, _text_vector(name), float(grade))]


def test_encoded_source_correction_changes_the_late_consequence_without_reencoding_history():
    events = []

    def encode(raw):
        events.append(raw)
        return _source_fixture(raw)

    anchors = np.stack([_text_vector(name) for name in ("Aster", "Birch", "Cedar")])
    original_sensor, corrected_sensor = _sensor(anchors[1]), _sensor(anchors[0])
    sources = {
        "permit-A": b"Aster|permit|0.9",
        "permit-B": b"Birch|permit|0.9",
        "sensor": original_sensor,
        "ins-C": b"Cedar|inspection|0.8",
        "permit-C": b"Cedar|permit|0.9",
    }
    base = CorpusBindingMemory.prepare(
        _PREDICATES, anchors, _GATES, sources, encode=encode, encoder_version="fixture-1",
        min_similarity=.5, gap=.05,
    )
    assert len(events) == 5
    assert base._certain is base._possible
    base = base.remember("release", 2)
    assert base.resolve("release", 0) == CorpusBindingRead(None, None)
    assert base.resolve("release", 1) == CorpusBindingRead(BindingProof(.7, frozenset({"permit-B", "sensor"})), .7)
    changed = base.revise({"sensor": corrected_sensor, "permit-A": sources["permit-A"]})
    assert len(events) == 6 and events[-1] == corrected_sensor
    assert changed.revise({"sensor": corrected_sensor}) is changed
    assert changed.resolve("release", 0) == CorpusBindingRead(BindingProof(.7, frozenset({"permit-A", "sensor"})), .7)
    assert changed.resolve("release", 1) == CorpusBindingRead(None, None)
    assert changed.resolve("release", 2, work_limit=1) == base.resolve("release", 2, work_limit=1)
    assert base.resolve("release", 0) == CorpusBindingRead(None, None)
    assert changed.source_digests["sensor"] == sha256(corrected_sensor).hexdigest()
    assert base.source_digests["sensor"] == sha256(original_sensor).hexdigest()
    assert len(events) == 6
    assert changed.revise({"sensor": None}).resolve("release", 0) == CorpusBindingRead(None, None)
    restored = changed.revise({"sensor": original_sensor})
    assert restored.resolve("release", 1) == base.resolve("release", 1)


def _axis_fixture(raw):
    fields = raw.decode().splitlines()
    mentions = []
    for line in fields:
        predicate, x, y, strength = line.split(",")
        mentions.append(BindingMention(predicate, np.array([float(x), float(y)]), float(strength)))
    return mentions


def test_ambiguous_and_outlying_vectors_preserve_possible_support_without_committing():
    anchors = np.eye(2)
    bank = CorpusBindingMemory.prepare(_PREDICATES, anchors, _GATES, {
        "p": b"permit,1,0,0.9",
        "q": b"permit,0,1,0.9",
        "sensor": b"inspection,1,1,0.8",
    }, encode=_axis_fixture, encoder_version="axis-fixture", min_similarity=.6, gap=.05)
    anchors[:] = 0
    for binding in range(2):
        unresolved = bank.resolve("release", binding)
        assert unresolved == CorpusBindingRead(None, .8)
        assert not unresolved.is_exact
        assert bank.resolve("release", binding, without_sources=iter(["sensor"])) == CorpusBindingRead(None, None)

    outlier = bank.revise({"sensor": b"inspection,-1,-1,0.8"})
    assert outlier.resolve("release", 1) == CorpusBindingRead(None, .8)
    assert outlier.outlier_sources == {"sensor"}
    assert not bank.outlier_sources
    resolved = outlier.revise({"sensor": b"inspection,1,0,0.8"})
    assert not resolved.outlier_sources
    assert resolved.resolve("release", 0).is_exact
    assert resolved.resolve("release", 0).certain == BindingProof(.8, frozenset({"p", "sensor"}))
    assert resolved.resolve("release", 1) == CorpusBindingRead(None, None)
    assert not bank.resolve("release", 0).is_exact


def test_candidate_envelope_keeps_shared_origin_disjoint_and_supported_zero_distinct():
    bank = CorpusBindingMemory.prepare(_PREDICATES, np.eye(2), _GATES, {
        "one": b"inspection,1,1,0.8\npermit,1,0,0.9",
    }, encode=_axis_fixture, encoder_version="axis-fixture", min_similarity=.6, gap=.05)
    assert bank.resolve("release", 0) == CorpusBindingRead(None, None)
    alternative = bank.revise({"other": b"permit,1,0,0.4"})
    assert alternative.resolve("release", 0) == CorpusBindingRead(None, .4)
    assert alternative.resolve("release", 0, without_sources=("other",)) == CorpusBindingRead(None, None)
    supported = bank.revise({"second": b"inspection,1,0,0"})
    assert supported.resolve("release", 0) == CorpusBindingRead(BindingProof(0, frozenset({"one", "second"})), 0)
    assert supported.resolve("release", 0).is_exact


def test_encoding_validation_and_possible_bank_failure_leave_prior_snapshot_readable():
    bank = CorpusBindingMemory.prepare(_PREDICATES, np.eye(2), _GATES, {
        "p": b"permit,1,0,1",
        "q": b"inspection,1,0,0.7",
    }, encode=_axis_fixture, encoder_version="axis-fixture", min_similarity=.6, gap=.05, term_limit=1)
    with pytest.raises(BindingCapacityError):
        bank.revise({"p": b"permit,1,0,0.9", "extra": b"inspection,1,1,0.3"})
    with pytest.raises(ValueError, match="nonzero"):
        bank.revise({"p": b"permit,0,0,0.9"})
    with pytest.raises(ValueError, match="strength"):
        bank.revise({"extra": b"inspection,1,1,1.3"})
    assert bank.resolve("release", 0) == CorpusBindingRead(BindingProof(.7, frozenset({"p", "q"})), .7)
    assert bank.source_digests["p"] == sha256(b"permit,1,0,1").hexdigest()
    assert "extra" not in bank.source_digests


def test_bounds_enclose_each_assignment_of_ambiguous_source_mentions():
    source_rows = {
        "p": b"inspection,1,1,0.6\npermit,1,0,0.9",
        "q": b"permit,1,1,0.8",
        "r": b"inspection,0,1,0.7",
    }
    bank = CorpusBindingMemory.prepare(
        _PREDICATES, np.eye(2), _GATES, source_rows,
        encode=_axis_fixture, encoder_version="axis-fixture", min_similarity=.6, gap=.05,
    )
    for assigned_inspection, assigned_permit in product(range(2), repeat=2):
        rows = {
            "p": [("inspection", assigned_inspection, .6), ("permit", 0, .9)],
            "q": [("permit", assigned_permit, .8)],
            "r": [("inspection", 1, .7)],
        }
        direct = BindingMemory.prepare(_PREDICATES, 2, _GATES, rows)
        for binding in range(2):
            result = bank.resolve("release", binding)
            proof = direct.resolve("release", binding)
            floor = -np.inf if result.certain is None else result.certain.strength
            ceiling = -np.inf if result.possible_strength is None else result.possible_strength
            actual = -np.inf if proof is None else proof.strength
            assert floor <= actual <= ceiling


def test_candidate_expansion_limit_is_batch_scoped_and_metadata_changes_survive_noop_rows():
    bank = CorpusBindingMemory.prepare(_PREDICATES, np.eye(2), _GATES, {
        "p": b"permit,1,0,0.9",
    }, encode=_axis_fixture, encoder_version="axis-fixture", min_similarity=.6, gap=.05, expanded_limit=2)
    with pytest.raises(BindingCapacityError, match="expansion limit"):
        bank.revise({"p": b"permit,1,0,0.90", "extra": b"inspection,1,1,0.8"})
    new = bank.revise({"p": b"permit,1,0,0.90"})
    assert new.source_digests["p"] != bank.source_digests["p"]
    assert new._certain is bank._certain and new._possible is bank._possible
    assert bank.revise({"extra": b"inspection,1,1,0.8"}).resolve("release", 0).possible_strength == .8
