import codecs
from copy import copy, deepcopy
from dataclasses import replace
import json
import subprocess

import numpy as np
import pytest

from experiments.memory import repository_import_memory as imports
from experiments.memory.disjoint_feedback import (
    DisjointFeedback, DisjointFeedbackWorkspace, RelationGuardFailed, RelationOutcomeGuard, StaleRelationRevision,
    TerminalRelationEvidence, TerminalRelationGrade,
)
from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.repository_import_readout import (
    RetainedImportRead, prepare_import_readout, prepare_published_import_readout,
)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo, files):
    for name, source in files.items():
        path = repo / "src" / "pkg" / name
        if source is None:
            path.unlink()
        else:
            path.write_text(source, encoding="utf-8")
    _git(repo, "add", "--", "src/pkg")
    _git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Snapshot")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    first = _commit(tmp_path, {
        "__init__.py": "", "start.py": "from .relay import worker\nimport pkg.other\n",
        "relay.py": "# line one\nfrom pkg.before import get\n", "before.py": "", "other.py": "import pkg.always\n",
        "always.py": "", "decoy.py": "import pkg.before\n",
    })
    second = _commit(tmp_path, {"relay.py": "from .added import get\n", "added.py": ""})
    third = _commit(tmp_path, {"relay.py": None})
    return tmp_path, first, second, third


def test_repository_memory_uses_pinned_blobs_and_reuses_unchanged_imports(history, monkeypatch):
    repo, first, second, third = history
    (repo / "src" / "pkg" / "start.py").write_text("broken !", encoding="utf-8")
    monkeypatch.setenv("GIT_DIR", str(repo / "unrelated.git"))
    monkeypatch.setenv("GIT_LITERAL_PATHSPECS", "1")
    before = imports.RepositoryImportMemory.prepare(repo, first, package="pkg")
    assert before.destinations("pkg.start") == ("pkg.always", "pkg.before")
    assert before.prove("pkg.start", "pkg.before") == (
        imports.ImportEvidence("src/pkg/start.py", 1, "pkg.relay"),
        imports.ImportEvidence("src/pkg/relay.py", 2, "pkg.before"),
    )
    original = imports._module_imports
    parsed = []

    def watch(content, path, name, package):
        parsed.append(name)
        return original(content, path, name, package)

    monkeypatch.setattr(imports, "_module_imports", watch)
    after = before.revise(second)
    assert set(parsed) == {"pkg.relay", "pkg.added"} and len(parsed) == 2
    assert after.destinations("pkg.start") == ("pkg.added", "pkg.always")
    assert after.prove("pkg.start", "pkg.before") is None
    assert after.prove("pkg.start", "pkg.added")[-1].line == 1
    assert after._modules["pkg.start"] is before._modules["pkg.start"]
    assert all(after._coordinates[name] == coord for name, coord in before._coordinates.items())
    unrelated = before._coordinates["pkg.other"]
    assert after._operator._middles[unrelated] is before._operator._middles[unrelated]

    parsed.clear()
    removed = after.revise(third)
    assert removed.destinations("pkg.start") == ("pkg.always",)
    assert removed.prove("pkg.start", "pkg.added") is None
    assert parsed == []
    with pytest.raises(ValueError, match="outside this committed"):
        removed.prove("pkg.start", "pkg.relay")
    assert before.destinations("pkg.start") == ("pkg.always", "pkg.before")
    assert after.destinations("pkg.start") == after.revise(first).revise(second).destinations("pkg.start")


def test_cli_reports_evidence_at_one_commit_and_absence_at_the_next(history, capsys):
    repo, first, second, _ = history
    imports.main(["--repo", str(repo), "--package", "pkg", "--revision", first, "--revision", second,
                  "--start", "pkg.start", "--destination", "pkg.before"])
    before, after = json.loads(capsys.readouterr().out)
    assert before["commit"] == first and before["evidence"][-1]["imported_module"] == "pkg.before"
    assert after["commit"] == second and after["evidence"] is None


def test_tied_witness_is_stable_after_coordinate_growth(history):
    repo, first, _, _ = history
    _git(repo, "checkout", "--detach", first)
    second = _commit(repo, {"ahead.py": "import pkg.before\n",
                            "start.py": "from .relay import worker\nimport pkg.ahead\n"})
    initial = imports.RepositoryImportMemory.prepare(repo, first, package="pkg")
    incremented = initial.revise(second)
    fresh = imports.RepositoryImportMemory.prepare(repo, second, package="pkg")
    assert incremented._coordinates["pkg.ahead"] > incremented._coordinates["pkg.relay"]
    assert incremented.prove("pkg.start", "pkg.before") == fresh.prove("pkg.start", "pkg.before") == (
        imports.ImportEvidence("src/pkg/start.py", 2, "pkg.ahead"),
        imports.ImportEvidence("src/pkg/ahead.py", 1, "pkg.before"),
    )


class _FixtureTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and not return_dict
        return list(json.dumps(messages, ensure_ascii=True).encode())


def _published_import_workspace(memory, start, destination, middles, *, bypass=False, strength=.75):
    width = len(memory.relation_coordinates)
    keys = np.zeros((width, 1))
    zeros = np.zeros((width, width))
    identity = np.eye(width)
    kernel = FeedbackKernel(query=keys, carry=zeros, observation_key=keys, observation_value=zeros,
                            feedback_key=keys, feedback_value=zeros, max_steps=2 if bypass else 1)
    receiver = DisjointFeedback(kernel, query=identity, presence_value=zeros, grade_value=identity, floor=.25)
    pin = memory.attach_feedback(receiver, identity)
    seed = np.zeros(width)
    seed[memory.relation_coordinates[start]] = strength
    state = receiver.start(seed, pin)
    if bypass:
        state = receiver.step(receiver.step(state, use_operator=False), relation_query=seed)
    else:
        state = receiver.step(state)
    paths = [memory.relation_path(start, middle) for middle in middles]
    outcome = RelationOutcomeGuard(evidence=[TerminalRelationEvidence(memory.relation_coordinates[destination], paths)])
    workspace = DisjointFeedbackWorkspace(receiver, pin, outcome_guards=[(state, outcome)])
    return receiver, workspace


def test_published_import_readout_preserves_witness_handoff_and_source_only_revisions(history, monkeypatch):
    repo, first, _, _ = history
    _git(repo, "checkout", "--detach", first)
    base = _commit(repo, {"other.py": "# second route\nfrom .before import get\n"})
    memory = imports.RepositoryImportMemory.prepare(repo, base, package="pkg")
    start, destination = "pkg.start", "pkg.before"
    receiver, workspace = _published_import_workspace(memory, start, destination, ("pkg.relay", "pkg.other"))
    opening = workspace.pin()
    assert memory.prove(start, destination)[0].imported_module == "pkg.other"
    tokenizer = _FixtureTokenizer()
    layout = dict(context_tokens=4096, max_new_tokens=32)

    def read(source, publication):
        return prepare_published_import_readout(source, publication, 0, destination, tokenizer, **layout)

    def unexpected(*args, **kwargs):
        pytest.fail("published readout must not select a new import witness")

    monkeypatch.setattr(imports.RepositoryImportMemory, "prove", unexpected)
    original = read(memory, opening)
    explicit = prepare_import_readout(memory, start, destination, tokenizer,
                                      edges=[(start, "pkg.relay"), ("pkg.relay", destination)], **layout)
    assert original.witness.strength == .5 and original.input_ids == explicit.input_ids
    assert original.check('{"middle":"pkg.relay","citations":["P0","P1"]}', terminated=True) == "pass"
    assert original.quotes[1].text == "from pkg.before import get"

    later = memory.revise(_commit(repo, {"relay.py": "# withdrawn declaration\n"}))
    handoff = workspace.publish(opening, later.feedback_revision(memory, receiver, opening.memory))
    new = read(later, handoff)
    assert new.witness.strength == .5 and new.commit == later.commit and new.quotes[1].line == 2
    assert new.quotes[1].text == "from .before import get"
    assert new.check('{"middle":"pkg.relay","citations":["P0","P1"]}', terminated=True) == "wrong_middle"
    assert new.check('{"middle":"pkg.other","citations":["P0","P1"]}', terminated=True) == "pass"

    commented = later.revise(_commit(repo, {"other.py": "# extra line\n# second route\nfrom .before import get\n"}))
    assert commented._operator is later._operator
    source_only = commented.feedback_revision(later, receiver, handoff.memory)
    assert deepcopy(source_only, {id(source_only.memory): opening.memory}) is source_only
    tip = workspace.publish(handoff, source_only)
    assert tip is not handoff and tip.protected[0].feedback is handoff.protected[0].feedback
    assert deepcopy(tip.protected[0], {id(tip.memory): opening.memory}) is tip.protected[0]
    (repo / "src/pkg/other.py").write_text("invalid working copy !", encoding="utf-8")
    latest = read(commented, tip)
    assert latest.commit == commented.commit and latest.quotes[1].line == 3
    assert latest.quotes[1].blob != new.quotes[1].blob and latest.quotes[1].text == new.quotes[1].text
    assert read(memory, opening) == original and read(later, handoff) == new
    with pytest.raises(ValueError, match="exact repository source snapshot"):
        read(commented, handoff)
    with pytest.raises(ValueError, match="trajectory index"):
        prepare_published_import_readout(commented, tip, -1, destination, tokenizer, **layout)
    with pytest.raises(TypeError, match="through guarded publication"):
        replace(tip, witnesses=opening.witnesses)
    with pytest.raises(TypeError, match="through prepare or revise"):
        replace(commented, commit=later.commit)
    assert copy(commented) is deepcopy(commented) is commented
    assert copy(tip) is deepcopy(tip) is tip
    for record in (commented, tip, tip.memory, latest, latest.witness, latest.quotes[1]):
        with pytest.raises(TypeError, match="__dict__"):
            vars(record)


def test_repository_adoption_requires_growth_anchors_and_rejects_mixed_drafts_and_evidence_loss(history):
    repo, first, second, _ = history
    memory = imports.RepositoryImportMemory.prepare(repo, first, package="pkg")
    start, destination = "pkg.start", "pkg.before"
    receiver, workspace = _published_import_workspace(memory, start, destination, ["pkg.relay"])
    parent = workspace.pin()
    with pytest.raises(ValueError, match="changed repository coordinate frame"):
        memory.revise(second).feedback_revision(memory, receiver, parent.memory)
    with pytest.raises(ValueError, match="direct repository successor"):
        imports.RepositoryImportMemory.prepare(repo, first, package="pkg").feedback_revision(memory, receiver, parent.memory)

    _git(repo, "checkout", "--detach", first)
    withdrawn = memory.revise(_commit(repo, {"relay.py": "# path withdrawn\n"}))
    adoption = withdrawn.feedback_revision(memory, receiver, parent.memory)
    noop = receiver.revise_memory(parent.memory, {})
    with pytest.raises(ValueError, match="must publish alone"):
        workspace.publish(parent, adoption, noop)
    with pytest.raises(RelationGuardFailed, match="violated protected outcome"):
        workspace.publish(parent, adoption)
    assert workspace.pin() is parent

    _git(repo, "checkout", "--detach", first)
    (repo / "src/pkg/relay").mkdir()
    moved = memory.revise(_commit(repo, {"relay.py": None, "relay/__init__.py": "from pkg.before import get\n"}))
    moved_revision = moved.feedback_revision(memory, receiver, parent.memory)
    replayed, _ = receiver.rebase(parent.protected[0], moved_revision)
    np.testing.assert_array_equal(replayed.grades, parent.protected[0].grades)
    with pytest.raises(RelationGuardFailed, match="violated protected outcome"):
        workspace.publish(parent, moved_revision)
    assert workspace.pin() is parent

    _git(repo, "checkout", "--detach", first)
    deleted = memory.revise(_commit(repo, {"before.py": None}))
    assert deleted._operator is memory._operator
    deleted_revision = deleted.feedback_revision(memory, receiver, parent.memory)
    replayed, _ = receiver.rebase(parent.protected[0], deleted_revision)
    assert receiver.prove_terminal(replayed, memory.relation_coordinates[destination]).strength == .5
    with pytest.raises(RelationGuardFailed, match="violated protected outcome"):
        workspace.publish(parent, deleted_revision)
    assert workspace.pin() is parent

    layout = dict(context_tokens=4096, max_new_tokens=32)
    numeric = DisjointFeedbackWorkspace(receiver, parent.memory, protected=parent.protected)
    with pytest.raises(ValueError, match="no approved witness"):
        prepare_published_import_readout(memory, numeric.pin(), 0, destination, _FixtureTokenizer(), **layout)

    extra = receiver.revise_memory(parent.memory, {"other origin": [(0, 0, 0, .1)]})
    changed = workspace.publish(parent, extra)
    with pytest.raises(ValueError, match="exact repository source snapshot"):
        prepare_published_import_readout(memory, changed, 0, destination, _FixtureTokenizer(), **layout)
    with pytest.raises(StaleRelationRevision, match="before evaluation"):
        workspace.publish(parent, adoption)


def test_appended_module_keeps_old_addresses_and_supports_a_new_published_read(history):
    repo, first, _, _ = history
    _git(repo, "checkout", "--detach", first)
    memory = imports.RepositoryImportMemory.prepare(repo, first, package="pkg")
    start, destination = "pkg.start", "pkg.before"
    receiver, workspace = _published_import_workspace(memory, start, destination, ["pkg.relay"], bypass=True)
    parent = workspace.pin()
    grown = memory.revise(_commit(repo, {"new.py": "from .relay import worker\n"}))
    width = len(memory.relation_coordinates)
    assert grown.relation_coordinates == {**memory.relation_coordinates, "pkg.new": width}

    with pytest.raises(ValueError, match="names must match"):
        grown.feedback_revision(memory, receiver, parent.memory, added_anchors={"pkg.decoy": np.zeros(width)})
    with pytest.raises(ValueError, match="one added anchor row"):
        grown.feedback_revision(memory, receiver, parent.memory, added_anchors={"pkg.new": np.zeros(width + 1)})
    with pytest.raises(ValueError, match="must be finite"):
        grown.feedback_revision(memory, receiver, parent.memory, added_anchors={"pkg.new": np.full(width, np.nan)})
    anchor = np.zeros(width)
    entry = memory.relation_coordinates["pkg.decoy"]
    anchor[entry] = 1.
    revision = grown.feedback_revision(memory, receiver, parent.memory, added_anchors={"pkg.new": anchor})
    anchor[:] = 0.
    before = parent.protected[0]
    replayed, reused = receiver.rebase(before, revision)
    assert reused == 1 and replayed._feedback._history[0] is before._feedback._history[0]
    np.testing.assert_array_equal(replayed.grades[:width], before.grades)
    assert replayed.grades[width] == -np.inf

    strict = DisjointFeedbackWorkspace(receiver, parent.memory, protected=[before])
    assert strict.publish(strict.pin(), revision).protected[0].grades.shape == (width + 1,)
    tip = workspace.publish(parent, revision)
    layout = dict(context_tokens=4096, max_new_tokens=32)
    old_evidence = prepare_published_import_readout(memory, parent, 0, destination, _FixtureTokenizer(), **layout)
    moved_evidence = prepare_published_import_readout(grown, tip, 0, destination, _FixtureTokenizer(), **layout)
    assert old_evidence.witness == moved_evidence.witness
    assert old_evidence.quotes == moved_evidence.quotes and moved_evidence.commit == grown.commit

    query = np.zeros(width)
    query[entry] = .75
    state = receiver.step(receiver.start(query, tip.memory))
    guard = RelationOutcomeGuard(evidence=[TerminalRelationEvidence(grown.relation_coordinates[destination],
                                                                    [grown.relation_path("pkg.new", "pkg.relay")])])
    new_workspace = DisjointFeedbackWorkspace(receiver, tip.memory, outcome_guards=[(state, guard)])
    new_evidence = prepare_published_import_readout(grown, new_workspace.pin(), 0, destination, _FixtureTokenizer(), **layout)
    assert new_evidence.witness.start == width and new_evidence.witness.strength == .5
    assert new_evidence.quotes[0].declaring_module == "pkg.new"
    assert new_evidence.check('{"middle":"pkg.relay","citations":["P0","P1"]}', terminated=True) == "pass"


@pytest.mark.parametrize(("strength", "grade"), [(.75, .5), (.25, 0.)])
def test_strict_growth_refuses_new_support_even_when_hidden_rows_match(history, strength, grade):
    repo, first, _, _ = history
    _git(repo, "checkout", "--detach", first)
    memory = imports.RepositoryImportMemory.prepare(repo, first, package="pkg")
    receiver, workspace = _published_import_workspace(memory, "pkg.start", "pkg.before", ["pkg.relay"], strength=strength)
    parent = workspace.pin()
    grown = memory.revise(_commit(repo, {"new.py": "import pkg.bridge_new\n",
                                         "bridge_new.py": "import pkg.newdest\n", "newdest.py": ""}))
    width = len(memory.relation_coordinates)
    suffix = {name: np.zeros(width) for name in ("pkg.newdest", "pkg.new", "pkg.bridge_new")}
    suffix["pkg.new"][memory.relation_coordinates["pkg.start"]] = 1.
    revision = grown.feedback_revision(memory, receiver, parent.memory, added_anchors=suffix)
    before = parent.protected[0]
    candidate, reused = receiver.rebase(before, revision)
    destination = grown.relation_coordinates["pkg.newdest"]
    assert reused == 0 and candidate.grades[destination] == grade
    np.testing.assert_array_equal(candidate.feedback.history, before.feedback.history)
    np.testing.assert_array_equal(candidate.grades[:before.grades.size], before.grades)

    strict = DisjointFeedbackWorkspace(receiver, parent.memory, protected=[before])
    with pytest.raises(RelationGuardFailed, match="changed protected trajectory"):
        strict.publish(strict.pin(), revision)
    tip = workspace.publish(parent, revision)
    assert tip.protected[0].grades[destination] == grade

    withdrawn = grown.revise(_commit(repo, {"bridge_new.py": "# path withdrawn\n"}))
    same_frame = withdrawn.feedback_revision(grown, receiver, tip.memory)
    repaired, reused = receiver.rebase(tip.protected[0], same_frame)
    assert reused == 0 and np.isneginf(repaired.grades[destination])
    assert np.isneginf(workspace.publish(tip, same_frame).protected[0].grades[destination])


def test_first_named_coordinate_replaces_empty_placeholder_anchor(history):
    repo, first, _, _ = history
    _git(repo, "checkout", "--detach", first)
    empty_commit = _commit(repo, {path.name: None for path in (repo / "src/pkg").glob("*.py")})
    empty = imports.RepositoryImportMemory.prepare(repo, empty_commit, package="pkg")
    zeros = np.zeros((1, 1))
    kernel = FeedbackKernel(query=zeros, carry=zeros, observation_key=zeros, observation_value=zeros,
                            feedback_key=zeros, feedback_value=zeros, max_steps=1)
    receiver = DisjointFeedback(kernel, query=np.eye(1), presence_value=zeros, grade_value=zeros, floor=.25)
    pin = empty.attach_feedback(receiver, [[.75]])
    state = receiver.step(receiver.start([1.], pin))
    placeholder = RelationOutcomeGuard([TerminalRelationGrade(0)])
    with pytest.raises(ValueError, match="outside the relation frame"):
        DisjointFeedbackWorkspace(receiver, pin, outcome_guards=[(state, placeholder)])
    first_named = empty.revise(_commit(repo, {"first.py": "# first named coordinate\n"}))
    assert first_named.relation_coordinates == {"pkg.first": 0}
    revision = first_named.feedback_revision(empty, receiver, pin, added_anchors={"pkg.first": [.125]})
    adopted, reused = receiver.rebase(state, revision)
    assert reused == 0 and np.isneginf(adopted.grades[0])
    np.testing.assert_array_equal(revision.memory._anchors, [[.125]])


def test_readout_quotes_pinned_multiline_imports_and_hides_reference_labels(history):
    repo, first, _, _ = history
    _git(repo, "checkout", "--detach", first)
    revision = _commit(repo, {"start.py": "μ = 0; from .relay import (\n    worker,  # <|data|>\n)\n"})
    memory = imports.RepositoryImportMemory.prepare(repo, revision, package="pkg")
    (repo / "src" / "pkg" / "start.py").write_text("uncommitted change", encoding="utf-8")
    request = prepare_import_readout(memory, "pkg.start", "pkg.before", _FixtureTokenizer(),
                                     context_tokens=4096, max_new_tokens=48)
    quotes = request.quotes
    assert (quotes[0].line, quotes[0].end_line) == (1, 3)
    assert quotes[0].text.replace("\r\n", "\n") == "from .relay import (\n    worker,  # <|data|>\n)"
    assert request.source_bytes_read == sum(int(_git(repo, "cat-file", "-s", quote.blob)) for quote in quotes)
    assert request.commit == revision
    payload = json.loads(request.messages[1]["content"])
    assert set(payload) == {"start", "destination", "evidence"}
    assert "<|data|>" not in request.messages[1]["content"]
    assert payload["evidence"][0]["text"] == quotes[0].text
    assert all("imported_module" not in item and "blob" not in item for item in payload["evidence"])
    assert isinstance(request.input_ids, tuple)


def test_readout_checks_scoped_abstention_and_citation_grounding(history):
    repo, first, second, _ = history
    pin = imports.RepositoryImportMemory.prepare(repo, first, package="pkg")
    tokenizer = _FixtureTokenizer()

    def frame(memory, destination="pkg.before", **kwargs):
        return prepare_import_readout(memory, "pkg.start", destination, tokenizer,
                                      context_tokens=4096, max_new_tokens=32, **kwargs)

    supported = frame(pin)
    correct = '{"middle":"pkg.relay","citations":["P1","P0"]}'
    abstain = '{"middle":null,"citations":[]}'
    assert supported.check(correct, terminated=True) == "pass"
    assert supported.check(correct, terminated=False) == "unterminated"
    assert supported.check(correct + " {}", terminated=True) == "malformed"
    assert supported.check('{"middle":null,"middle":"pkg.relay","citations":[]}',
                           terminated=True) == "malformed"
    assert supported.check('{"middle":"pkg.relay","citations":["P0","P0"]}',
                           terminated=True) == "wrong_citations"

    missing_first = frame(pin, edges=[("pkg.relay", "pkg.before")], passage_ids=["P1"])
    missing_second = frame(pin, edges=[("pkg.start", "pkg.relay")], passage_ids=["P0"])
    disconnected = frame(pin, edges=[("pkg.start", "pkg.relay"), ("pkg.decoy", "pkg.before")])
    revised = frame(pin.revise(second))
    assert len(disconnected.quotes) == len(supported.quotes) == 2
    assert revised.quotes == () and revised.source_bytes_read == 0
    for negative in (missing_first, missing_second, disconnected, revised):
        assert negative.check(abstain, terminated=True) == "pass"
        assert negative.check(correct, terminated=True) == "wrong_middle"
    assert json.loads(missing_first.messages[1]["content"])["evidence"][0]["passage_id"] == "P1"
    reversed_quotes = frame(pin, edges=[("pkg.relay", "pkg.before"), ("pkg.start", "pkg.relay")],
                            passage_ids=["Z8", "K2"])
    assert reversed_quotes.check('{"middle":"pkg.relay","citations":["K2","Z8"]}', terminated=True) == "pass"
    changed = _commit(repo, {"start.py": "import pkg.decoy\n"})
    other = frame(pin.revise(changed))
    assert other.check(correct, terminated=True) == "wrong_middle"
    assert other.check('{"middle":"pkg.decoy","citations":["P0","P1"]}', terminated=True) == "pass"

    with pytest.raises(ValueError, match="no declared import"):
        frame(pin, edges=[("pkg.relay", "pkg.always")])
    with pytest.raises(ValueError, match="no truncation"):
        prepare_import_readout(pin, "pkg.start", "pkg.before", tokenizer,
                               context_tokens=len(supported.input_ids) + supported.max_new_tokens - 1,
                               max_new_tokens=supported.max_new_tokens)


def test_readout_counts_other_aliases_visible_in_the_quoted_declaration(history):
    repo, first, _, _ = history
    _git(repo, "checkout", "--detach", first)
    revision = _commit(repo, {"start.py": "import pkg.relay\nimport pkg.other, pkg.relay, json, pkg.start\n"})
    pin = imports.RepositoryImportMemory.prepare(repo, revision, package="pkg")
    request = prepare_import_readout(pin, "pkg.start", "pkg.before", _FixtureTokenizer(),
                                     context_tokens=4096, max_new_tokens=32,
                                     edges=[("pkg.start", "pkg.other"), ("pkg.relay", "pkg.before")])
    assert request.quotes[0].text == "import pkg.other, pkg.relay, json, pkg.start"
    assert request.quotes[0].declared_targets == ("pkg.other", "pkg.relay", "json", "pkg.start")
    assert request.quotes[0].line == 2
    assert request.check('{"middle":"pkg.relay","citations":["P0","P1"]}', terminated=True) == "pass"
    self_import = prepare_import_readout(pin, "pkg.start", "pkg.relay", _FixtureTokenizer(),
                                         context_tokens=4096, max_new_tokens=32,
                                         edges=[("pkg.start", "pkg.other"), ("pkg.start", "pkg.relay")])
    assert self_import.check('{"middle":null,"citations":[]}', terminated=True) == "pass"


@pytest.mark.parametrize(("header", "newline", "encoding"), [
    (["# coding: latin-1"], "\r", "latin-1"),
    (["# café", "# coding: latin-1 café"], "\r\n", "latin-1"),
    (["# coding: cp1252"], "\r\n", "cp1252"),
    (["# coding: utf-8"], "\r", "utf-8-sig"),
])
def test_quote_decoding_matches_the_committed_python_parser(history, header, newline, encoding):
    repo, first, _, _ = history
    _git(repo, "checkout", "--detach", first)
    _git(repo, "config", "core.autocrlf", "false")
    expected = newline.join(("from .relay import (", "    worker,  # café", ")"))
    source = newline.join((*header, expected, ""))
    (repo / "src/pkg/start.py").write_bytes(source.encode(encoding))
    revision = _commit(repo, {})
    memory = imports.RepositoryImportMemory.prepare(repo, revision, package="pkg")
    retained = RetainedImportRead.prepare(memory, "pkg.start", "pkg.before")
    quote = retained.quotes[0]
    assert quote.text == expected
    assert (quote.line, quote.end_line) == (len(header) + 1, len(header) + 3)


@pytest.mark.parametrize("newline", [b"\r", b"\r\n", b"\n"])
def test_revision_refuses_a_codec_that_joins_source_lines(history, newline):
    repo, first, _, _ = history
    _git(repo, "checkout", "--detach", first)
    _git(repo, "config", "core.autocrlf", "false")
    memory = imports.RepositoryImportMemory.prepare(repo, first, package="pkg")
    source = newline.join((b"# coding: unicode_escape", b"# join\\", b"# padding padding", b"import pkg.relay", b""))
    (repo / "src/pkg/start.py").write_bytes(source)
    revision = _commit(repo, {})
    with pytest.raises(ValueError, match="source codec does not preserve committed line boundaries"):
        memory.revise(revision)
    assert memory.prove("pkg.start", "pkg.before")[0].line == 1


def test_codec_whole_decode_must_agree_with_its_incremental_line_check():
    class IncrementalAscii(codecs.IncrementalDecoder):
        def decode(self, contents, final=False):
            return contents.decode("ascii")

    def decode(contents, errors="strict"):
        text = bytes(contents).decode("ascii", errors)
        return text.replace("# pad\n", "# pad\n# added line\n"), len(contents)

    def lookup(name):
        if name == "lsr_fixture_codec":
            return codecs.CodecInfo(None, decode, incrementaldecoder=IncrementalAscii, name=name)

    source = b"# coding: lsr_fixture_codec\r# pad\rimport pkg.relay\rimport pkg.decoy\r"
    codecs.register(lookup)
    try:
        with pytest.raises(ValueError, match="source codec does not preserve committed line boundaries"):
            imports._module_imports(source, "src/pkg/start.py", "pkg.start", "pkg")
    finally:
        codecs.unregister(lookup)


def test_retained_import_read_reuses_quotes_and_changes_the_canonical_witness(history, monkeypatch):
    repo, first, _, _ = history
    _git(repo, "checkout", "--detach", first)
    base = _commit(repo, {"start.py": "import pkg.ahead, pkg.relay, pkg.zz\n", "zz.py": "import pkg.before\n"})
    memory = imports.RepositoryImportMemory.prepare(repo, base, package="pkg")
    layout = dict(context_tokens=4096, max_new_tokens=32)
    start, destination = "pkg.start", "pkg.before"
    tokenizer = _FixtureTokenizer()
    initial = RetainedImportRead.prepare(memory, start, destination)
    assert initial.source_bytes_read > 0 and initial.quotes[0].imported_module == "pkg.relay"

    later = memory.revise(_commit(repo, {"zz.py": "import pkg.always\n", "before.py": "# unrelated edit\n"}))
    fresh = prepare_import_readout(later, start, destination, tokenizer, **layout)

    def unexpected(*args, **kwargs):
        pytest.fail("a valid retained read should require neither a fresh proof nor Git blob access")

    with monkeypatch.context() as patch:
        patch.setattr(imports.RepositoryImportMemory, "prove", unexpected)
        patch.setattr(imports.RepositoryImportMemory, "read_imports", unexpected)
        carried = initial.advance(later)
        framed = prepare_import_readout(later, start, destination, tokenizer, retained=carried, **layout)
    assert carried.reused and carried.quotes is initial.quotes
    assert framed.commit == later.commit and framed.source_bytes_read == carried.source_bytes_read == 0
    assert framed.quotes == fresh.quotes and framed.input_ids == fresh.input_ids and framed.messages == fresh.messages
    assert framed.check('{"middle":"pkg.relay","citations":["P0","P1"]}', terminated=True) == "pass"

    earlier_winner = later.revise(_commit(repo, {"ahead.py": "import pkg.before\n"}))
    for prior in (initial, carried):
        changed = prior.advance(earlier_winner)
        readout = prepare_import_readout(earlier_winner, start, destination, tokenizer, retained=changed, **layout)
        assert not changed.reused and changed.source_bytes_read > 0
        assert readout.check('{"middle":"pkg.ahead","citations":["P0","P1"]}', terminated=True) == "pass"
        assert readout.quotes[1].path == "src/pkg/ahead.py"
        assert prior.quotes[0].imported_module == "pkg.relay"
    with pytest.raises(ValueError, match="match the snapshot"):
        prepare_import_readout(earlier_winner, start, destination, tokenizer, retained=carried, **layout)
    with pytest.raises(ValueError, match="cannot override evidence"):
        prepare_import_readout(later, start, destination, tokenizer, retained=carried, edges=[], **layout)


def test_retained_import_absence_depends_on_every_candidate_and_current_endpoints(history):
    repo, first, second, _ = history
    start, destination = "pkg.start", "pkg.before"
    old = imports.RepositoryImportMemory.prepare(repo, first, package="pkg")
    supported = RetainedImportRead.prepare(old, start, destination)
    absent_pin = old.revise(second)
    absent = supported.advance(absent_pin)
    assert not absent.reused and not absent.quotes

    _git(repo, "checkout", "--detach", second)
    current = absent_pin.revise(_commit(repo, {"before.py": None}))
    with pytest.raises(ValueError, match="outside this committed"):
        absent.advance(current)
    restored = current.revise(_commit(repo, {"before.py": "# restored endpoint\n"}))
    assert absent.advance(restored).reused

    alternate = restored.revise(_commit(repo, {"relay.py": "# no link\n", "other.py": "import pkg.before\n"}))
    changed = absent.advance(alternate)
    assert not changed.reused and changed.quotes[0].imported_module == "pkg.other"
    earlier = alternate.revise(_commit(repo, {"other.py": "\nimport pkg.before\n"}))
    relocated = changed.advance(earlier)
    assert not relocated.reused and relocated.quotes[1].line == 2
    fresh = imports.RepositoryImportMemory.prepare(repo, earlier.commit, package="pkg")
    rebuilt = relocated.advance(fresh)
    assert not rebuilt.reused and rebuilt.quotes == relocated.quotes
