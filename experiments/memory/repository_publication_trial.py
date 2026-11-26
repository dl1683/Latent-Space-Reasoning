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


def _bundle_hashes(root):
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes())
        for path in root.rglob("*") if path.is_file()
    }


def _reader(arguments):
    sys.addaudithook(_refuse_subprocess)
    from experiments.memory.repository_transport_control import main
    main(["--read", *arguments])


def _task_world_reader(directory, seal, pin_index):
    sys.addaudithook(_refuse_subprocess)
    from experiments.memory.recurrent_task_bridge import RecurrentTaskBridge, RecurrentTaskBridgeLimits
    from experiments.memory.repository_transport_control import open_task_world
    from experiments.memory.task_evidence import EvidenceRead

    start = "experiments.memory._latent_factor_clusters"
    target = "experiments.memory.weight_program"
    with open_task_world(directory, seal, pin_index, arm="on") as opened:
        pin, task_world = opened["pin"], opened["task_world"]
        reasoner, memory, encoder = opened["reasoner"], opened["memory"], opened["encoder"]
        source_span, = task_world._spans_by_source[start]
        source = task_world._tool.verify(source_span, max_source_bytes=2 * 1024 * 1024)
        if source["status"] != "VERIFIED_SOURCE_BYTES":
            raise AssertionError("published task seed did not reopen from the source archive")
        mentions = pin.memory.source_mentions(start)
        ordinal = next(i for i, mention in enumerate(mentions) if mention.predicate == "context")
        read = EvidenceRead(0, "n1", source_span, "VERIFIED_SOURCE_BYTES", source["text"],
                            source["source_bytes_read"], tuple(mention.predicate for mention in mentions))
        limits = RecurrentTaskBridgeLimits(max_steps=2, max_hypotheses=1)

        def route(memory_state, *, allow_removed=False):
            bridge = RecurrentTaskBridge(
                task_world, reasoner, memory_state,
                readout_direction=encoder.code[target], readout_floor=.1,
                null_logit=8., margin=1., limits=limits,
                allow_ablated_operators=allow_removed,
            )
            return bridge.read(read, need_id="n1", ordinal=ordinal, steps=2)

        attached = route(memory)
        removed = route(memory.without_source_transport(), allow_removed=True)
        restored = route(memory)
        if (attached.readout_status != "supported" or removed.readout_status != "unresolved"
                or removed.completed_steps != 2 or removed.overflow
                or any(branch.complete for branch in removed.branches)
                or restored.receipt_sha256 != attached.receipt_sha256
                or attached.memory_state_sha256 != removed.memory_state_sha256):
            def summarize(result):
                branches = tuple((branch.complete, branch.readout,
                                  tuple(None if witness is None else witness.ordinal
                                        for witness in branch.witnesses))
                                 for branch in result.branches)
                return (f"{result.readout_status} steps={result.completed_steps}/{result.requested_steps} "
                        f"overflow={result.overflow} branches={branches}")

            raise AssertionError(
                "published task bridge failed the same-pin operator intervention: "
                f"attached={summarize(attached)}, removed={summarize(removed)}, "
                f"restored={summarize(restored)}, "
                f"receipt_equal={restored.receipt_sha256 == attached.receipt_sha256}, "
                f"memory_equal={attached.memory_state_sha256 == removed.memory_state_sha256}"
            )
        checked = 0
        for witness in attached.witness_packet:
            opened_witness = task_world._tool.verify(witness.span, max_source_bytes=2 * 1024 * 1024)
            if opened_witness["status"] != "VERIFIED_SOURCE_BYTES":
                raise AssertionError("task bridge returned a witness outside the published source archive")
            checked += 1
        return dict(pin=pin.journal_sha256, task_source_count=len(task_world._spans),
                    task_setup_source_bytes=task_world.setup_source_bytes,
                    status=attached.readout_status, removed_status=removed.readout_status,
                    restored_receipt=restored.receipt_sha256,
                    checked_witness_spans=checked,
                    memory_fingerprint_bytes=attached.memory_fingerprint_bytes)


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
        initial = _bundle_hashes(bundle)
        child(_DRIVER, "--author", repo, questions, gold)
        seal = receipt["seal"]
        outputs = [json.loads(child(_DRIVER, "--reader", bundle, seal, questions, "--arm", arm)) for arm in _ARMS]
        task_world = json.loads(child(_DRIVER, "--task-world", bundle, seal, 2))
        checked = _grade(repo, outputs, questions, gold, seal)
        assert _bundle_hashes(bundle) == initial
        return dict(**receipt, questions_sha256=outputs[0]["questions_sha256"], seed=_SEED,
                    journal_bytes=(bundle / "corpus.sqlite").stat().st_size,
                    attached_statuses=Counter(row["status"] for row in outputs[0]["results"]),
                    checked_attached_witnesses=checked,
                    validated_reads=sum(len(result["results"]) for result in outputs),
                    task_world_bridge=task_world)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--author"]:
        _author(*map(Path, argv[1:]))
    elif argv[:1] == ["--reader"]:
        _reader(argv[1:])
    elif argv[:1] == ["--task-world"]:
        print(json.dumps(_task_world_reader(argv[1], argv[2], int(argv[3])), indent=2))
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
