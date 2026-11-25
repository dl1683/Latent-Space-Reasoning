import json
import subprocess

import pytest

from experiments.memory import repository_import_memory as imports


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
        "always.py": "",
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
