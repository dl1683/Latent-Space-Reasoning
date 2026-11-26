"""Reproduce the visible four-pin publication control with late questions."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from random import Random
import subprocess
import sys
from tempfile import TemporaryDirectory

from experiments.memory.verified_source_tool import sha256


_ENTRY = "experiments.memory.repository_transport_control"
_DRIVER = "experiments.memory.repository_publication_trial"
_PACKAGE = "experiments.memory"
_PINS = ("0cfeae34", "c79f894e", "c393123f", "0cfeae34")
_SEED = 80164211
_ARMS = ("on", "operators-off", "memory-omitted", "on")


def _author(repo, question_path, gold_path):
    from experiments.memory.repository_import_memory import RepositoryImportMemory
    from experiments.memory.repository_transport_control import _expected

    rng = Random(_SEED)
    questions, gold = [], {}

    def ask(pin, start, target, status, middle=None):
        name = f"q{len(questions) + 1}"
        questions.append(dict(id=name, pin=pin, start=start, target=target))
        gold[name] = dict(status=status, middle=middle)

    reference = RepositoryImportMemory.prepare(repo, _PINS[0], package=_PACKAGE, source_root=".")
    for pin, revision in enumerate(_PINS):
        if pin:
            reference = reference.revise(revision)
        adjacency = {name: tuple(row.imports) for name, row in reference._modules.items()}
        destinations = {start: _expected(start, adjacency) for start in sorted(adjacency)}
        positives = sorted(name for name, target in destinations.items() if target is not None)
        negatives = sorted(name for name, target in destinations.items() if target is None)
        for start in rng.sample(positives, 2):
            target = destinations[start]
            middle, = adjacency[start]
            ask(pin, start, target, "supported", middle=middle)
            ask(pin, start, rng.choice(sorted(adjacency.keys() - {target, start})), "missing")
        for start in rng.sample(negatives, 2):
            ask(pin, start, rng.choice(sorted(adjacency.keys() - {start})), "missing")
    added = "experiments.memory._latent_factor_clusters"
    leaf = "experiments.memory.weight_program"
    ask(2, added, leaf, "supported", middle="experiments.memory.latent_feedback")
    ask(3, added, leaf, "outside_pin")
    question_path.write_text(json.dumps(questions), encoding="utf8")
    gold_path.write_text(json.dumps(gold), encoding="utf8")


def _refuse_subprocess(event, _):
    if event == "subprocess.Popen":
        raise AssertionError("offline reader attempted to launch a subprocess")


def _reader(arguments):
    sys.addaudithook(_refuse_subprocess)
    from experiments.memory.repository_transport_control import main
    main(["--read", *arguments])


def _witnesses(row, middle, contents):
    branch, = row["branches"]
    transport, = branch["transports"]
    observations = branch["witnesses"]
    authority = transport["evidence"]
    assert [w["span"]["source_id"] for w in observations] == [row["start"], row["target"]]
    assert [w["span"]["source_id"] for w in authority] == [row["start"], row["start"], middle, middle]
    for witness in (*observations, *authority):
        span = witness["span"]
        raw = contents[span["source_id"]]
        start, end = span["start_byte"], span["end_byte"]
        assert 0 <= start < end <= len(raw)
        assert sha256(raw[start:end]) == span["span_sha256"]
    return len(observations) + len(authority)


def _grade(repo, outputs, questions, gold, seal):
    from experiments.memory.repository_transport_control import _load_changes, _tree

    request_bytes = questions.read_bytes()
    requests = json.loads(request_bytes)
    expected = json.loads(gold.read_text(encoding="utf8"))
    assert set(expected) == {task["id"] for task in requests}
    assert outputs[-1] == outputs[0]
    assert all(result["questions_sha256"] == sha256(request_bytes) and result["seal"] == seal
               for result in outputs)
    pins = [_tree(repo, revision, _PACKAGE) for revision in _PINS]
    cache = {}
    checked = 0
    for question, normal, off, omitted in zip(requests, *(result["results"] for result in outputs[:3]), strict=True):
        name = question["id"]
        pin = question["pin"]
        commit, tree = pins[pin]
        for row, arm in zip((normal, off, omitted), _ARMS[:3], strict=True):
            assert all(row[key] == value for key, value in question.items())
            assert row["commit"] == commit and row["journal"] == normal["journal"] and row["arm"] == arm
        status = expected[name]["status"]
        if status == "missing":
            assert normal["status"] in {"missing", "undecided", "unresolved"}, (question, normal["status"])
        else:
            assert normal["status"] == status, (question, normal["status"], status)
        assert off["status"] == ("outside_pin" if status == "outside_pin" else "missing") and not off["branches"]
        assert omitted["status"] == ("outside_pin" if status == "outside_pin" else "memory_omitted")
        assert not omitted["branches"]
        if status == "supported":
            if commit not in cache:
                cache[commit], _ = _load_changes(repo, tree, {})
            checked += _witnesses(normal, expected[name]["middle"], cache[commit])
        else:
            assert not normal["branches"]
    return checked


def run(repo):
    checkout = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(checkout), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    repo = repo.resolve()
    with TemporaryDirectory(prefix="lsr-late-questions-") as directory:
        root = Path(directory)

        def child(module, *arguments):
            return subprocess.check_output([sys.executable, "-m", module, *map(str, arguments)],
                                           cwd=root, env=env, text=True, timeout=120)

        bundle = root / "publication"
        questions, gold = root / "questions.json", root / "answers.json"
        receipt = json.loads(child(_ENTRY, "--publish", "--repo", repo, "--output", bundle, *_PINS))
        retry = json.loads(child(_ENTRY, "--publish", "--repo", repo, "--output", root / "retry", *_PINS))
        assert retry["seal"] != receipt["seal"] and retry["content_sha256"] == receipt["content_sha256"]
        assert not questions.exists() and not gold.exists()
        initial = {path.name: sha256(path.read_bytes()) for path in bundle.iterdir()}
        child(_DRIVER, "--author", repo, questions, gold)
        seal = receipt["seal"]
        outputs = [json.loads(child(_DRIVER, "--reader", bundle, seal, questions, "--arm", arm)) for arm in _ARMS]
        checked = _grade(repo, outputs, questions, gold, seal)
        assert {path.name: sha256(path.read_bytes()) for path in bundle.iterdir()} == initial
        return dict(**receipt, questions_sha256=outputs[0]["questions_sha256"], seed=_SEED,
                    journal_bytes=(bundle / "corpus.sqlite").stat().st_size,
                    attached_statuses=Counter(row["status"] for row in outputs[0]["results"]),
                    checked_attached_witnesses=checked,
                    validated_reads=sum(len(result["results"]) for result in outputs))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--author"]:
        _author(*map(Path, argv[1:]))
    elif argv[:1] == ["--reader"]:
        _reader(argv[1:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--repo", type=Path, default=Path.cwd())
        parser.add_argument("--isolated-output", type=Path, help="fresh private directory for Linux stage artifacts")
        options = parser.parse_args(argv)
        if options.isolated_output is None:
            result = run(options.repo)
        else:
            from experiments.memory.repository_publication_isolation import run_isolated
            result = run_isolated(options.repo, options.isolated_output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
