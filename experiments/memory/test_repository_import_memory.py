import json
import subprocess

import pytest

from experiments.memory import repository_import_memory as imports
from experiments.memory.repository_import_readout import prepare_import_readout


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
