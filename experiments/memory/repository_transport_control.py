"""Visible raw-source control using committed Python declarations.

The combined runner checks an exact comparator. Publish and read also
support a separate question process with a digest-pinned source handoff.
"""

import argparse
from contextlib import closing, contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import sqlite3
import sys
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import numpy as np

from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace, _frame_digest, _json
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory
from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.recurrent_world import RecurrentWorld
from experiments.memory.repository_import_memory import (
    RepositoryImportMemory, _decode_python, _git, _module_imports, _module_tree, _read_blobs,
)
from experiments.memory.verified_source_tool import require_digest, sha256
from experiments.memory.world_transport import DiscoveredTransportPolicy


_MAX_COORDINATES = 256
_MAX_REVISIONS = 8
_MAX_JOURNAL_BYTES = 32 * 1024 * 1024
_MAX_JSON_BYTES = 1024 * 1024
_PUBLICATION_SCHEMA = "lsr-committed-imports-publication-v1"
_REASONING_ARGS = dict(steps=2, max_hypotheses=1, null_logit=8., margin=1.)
_READOUT_FLOOR = .1
_READ_CONTRACT = dict(_REASONING_ARGS, floor=_READOUT_FLOOR)


def _tree(repo, rev, package):
    commit = _git(repo, "rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}").decode().strip()
    modules = _module_tree(repo, commit, PurePosixPath("."), package)
    if not 1 <= len(modules) <= 256:
        raise ValueError("visible control requires between 1 and 256 tracked modules")
    return commit, modules


def _load_changes(repo, tree, previous):
    changed = {name: blob for name, (_, blob) in tree.items() if tree[name] != previous.get(name)}
    blobs = sorted(set(changed.values()))
    removed = dict.fromkeys(previous.keys() - tree.keys())
    if not blobs:
        return removed, 0
    data = "\n".join(blobs).encode("ascii") + b"\n"
    sizes = _git(repo, "cat-file", "--batch-check=%(objecttype) %(objectsize)", data=data).splitlines()
    if len(sizes) != len(blobs) or any(not row.startswith(b"blob ") for row in sizes):
        raise ValueError("revision must contain pinned blobs")
    size = sum(int(row.split()[1]) for row in sizes)
    if size > 8 * 1024 * 1024:
        raise ValueError("source revision exceeds the byte allowance")
    contents = _read_blobs(repo, blobs)
    return {**removed, **{name: contents[blob] for name, blob in changed.items()}}, size


class _ImportEncoder:
    """Append coordinates from source revisions alone before preparing a journal event."""

    def __init__(self, package):
        self.package = package
        self.axes = np.eye(_MAX_COORDINATES)
        self.code = {}
        self._paths = {}
        self._staged = {}

    def stage(self, tree, changes):
        pending = {}
        names = set(tree)
        for name, raw in changes.items():
            if raw is None:
                continue
            path, _ = tree[name]
            if name in self._paths and self._paths[name] != path:
                raise ValueError("control requires stable paths for retained module names")
            # AST columns are UTF-8 bytes. Keep every scope inside this byte contract.
            if (not raw or raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw.replace(b"\r\n", b"")
                    or _decode_python(raw) != raw.decode("utf8")):
                raise ValueError("control requires nonempty UTF-8 files without a BOM or bare CR")
            edges = _module_imports(raw, path, name, self.package)
            pending[name] = (raw, edges)
            names.update(edges)
        additions = sorted(names - self.code.keys())
        if len(self.code) + len(additions) > len(self.axes):
            raise ValueError("source-derived coordinates exceed the fixed frame")
        self.code.update(zip(additions, self.axes[len(self.code):len(self.code) + len(additions)], strict=True))
        self._paths.update({name: path for name, (path, _) in tree.items()})
        self._staged = pending

    def __call__(self, name, raw):
        source, edges = self._staged[name]
        if source != raw:
            raise ValueError("encoding differs from the staged source bytes")
        lines = raw.splitlines(keepends=True)
        starts = np.cumsum((0, *map(len, lines)))
        authority = []
        for target, decl in edges.items():
            span = (int(starts[decl.lineno - 1]) + decl.col_offset,
                    int(starts[decl.end_lineno - 1]) + decl.end_col_offset)
            authority.append(span)
            yield BindingMention("transport_entry", self.code[name], 1., span)
            yield BindingMention("transport_exit", self.code[target], 1., span)
        role = "context" if edges else "relay"
        for start, end in zip(starts[:-1], starts[1:], strict=True):
            if all(end <= first or start >= last for first, last in authority):
                yield BindingMention(role, self.code[name], 1., (int(start), int(end)))
                return
        raise ValueError("module observation requires a line outside the import declarations")


def _frame(encoder):
    axes = encoder.axes
    frame = CorpusBindingMemory.prepare(("context", "relay", "transport_entry", "transport_exit"),
                                        axes, {}, {}, encode=encoder, encoder_version="committed-imports-v1",
                                        min_similarity=1., gap=0., source_aware=True)
    zeros = np.zeros_like(axes)
    kernel = FeedbackKernel(query=30. * axes, carry=.15 * axes, observation_key=axes, observation_value=zeros,
                            feedback_key=axes, feedback_value=zeros, max_steps=2)
    policy = DiscoveredTransportPolicy("context", "relay", gain=2., max_terms=len(axes), max_sources=len(axes),
                                       scoped_clauses=True)
    world = RecurrentWorld(kernel, encoder_version=frame.encoder_version, max_observations=len(axes),
                           feedback=False, transport_policy=policy)
    return frame, world


def _expected(source, adjacency):
    first = adjacency[source]
    if len(first) != 1:
        return None
    middle, = first
    second = adjacency.get(middle, ())
    if len(second) != 1:
        return None
    target, = second
    leaf = adjacency.get(target)
    return target if len({source, middle, target}) == 3 and leaf == () else None


def _evaluate(world, pin, bank, code, reference, contents):
    graph = {name: tuple(module.imports) for name, module in reference._modules.items()}
    off = world.prepare(pin, previous=bank, packed=False, operators=False)
    assert off._sources is bank._sources
    outcomes, source_scores, checked_witnesses = {}, 0, 0
    for name, start in code.items():
        finding = world.reason_joint(start, memory=bank, **_REASONING_ARGS)
        source_scores += finding.observation_scores
        expected = _expected(name, graph)
        supported = [target for target, direction in code.items()
                     if finding.read(direction, floor=_READOUT_FLOOR).status == "supported"]
        if supported != ([] if expected is None else [expected]):
            raise AssertionError(f"exact comparison disagrees for {name}: {expected!r} versus {supported!r}")
        if not supported:
            continue
        branch, = finding.branches
        use, = branch.transport_uses
        middle, = graph[name]
        assert tuple(w.span.source_id for w in use.evidence) == (name, name, middle, middle)
        assert tuple(w.span.source_id for w in branch.witnesses) == (name, expected)
        for witness in (*branch.witnesses, *use.evidence):
            span = witness.span
            raw = contents[span.source_id]
            assert pin.memory.source_mention_span(span.source_id, witness.ordinal) == span
            assert pin._source_revisions[span.source_id] == witness.generation
            assert sha256(raw[span.start_byte:span.end_byte]) == span.span_sha256
            checked_witnesses += 1
        ablated = world.reason_joint(start, memory=off, **_REASONING_ARGS)
        assert ablated.read(code[expected], floor=_READOUT_FLOOR).status == "missing"
        outcomes[name] = expected
    return outcomes, source_scores, checked_witnesses


def _name(name):
    return type(name) is str and all(part.isidentifier() for part in name.split("."))


def _publications(repo, revisions, package, encoder, world, workspace):
    if not 1 <= len(revisions) <= _MAX_REVISIONS or not _name(package):
        raise ValueError("expected a Python package and no more than eight pinned revisions")
    previous = {}
    for rev in revisions:
        commit, tree = _tree(repo, rev, package)
        changes, byte_count = _load_changes(repo, tree, previous)
        encoder.stage(tree, changes)
        source = workspace.publish(workspace.prepare_changes(workspace.snapshot(), changes))
        pin = world.maintain_transports(workspace, source)
        previous = tree
        yield commit, pin, changes, byte_count


def _forbid_encoding(*_):
    raise AssertionError("publication replay accessed the encoder")


def _history(workspace, revision):
    return workspace.snapshot_at(revision, max_events=2 * _MAX_REVISIONS + 1,
                                 max_payload_bytes=_MAX_JOURNAL_BYTES)


def run(repo, revisions, *, package):
    encoder = _ImportEncoder(package)
    frame, world = _frame(encoder)
    raw, report, exact, bank = {}, [], None, None
    with TemporaryDirectory(prefix="lsr-import-control-") as directory:
        path = Path(directory) / "corpus.sqlite"
        workspace = JournaledCorpusWorkspace(path, frame, create=True)
        for commit, pin, changes, byte_count in _publications(repo, revisions, package, encoder, world, workspace):
            for name, contents in changes.items():
                if contents is None:
                    raw.pop(name)
                else:
                    raw[name] = contents
            bank = world.prepare(pin, previous=bank, packed=False)
            # The combined comparator constructs tasks only after each publication.
            exact = (RepositoryImportMemory.prepare(repo, commit, package=package, source_root=".")
                     if exact is None else exact.revise(commit))
            if set(exact._modules) != set(pin.memory.source_digests):
                raise AssertionError("source publication differs from the exact comparator's tracked module set")
            active_code = {name: encoder.code[name] for name in sorted(exact._modules)}
            outcomes, scores, witnesses = _evaluate(world, pin, bank, active_code, exact, raw)
            report.append(dict(commit=commit, journal=pin.journal_sha256, journal_revision=pin.revision,
                               modules=len(active_code), changed_modules=len(changes), distinct_blob_bytes=byte_count,
                               projected_rows=bank.projected_rows, late_reads=len(active_code),
                               candidate_readouts=len(active_code) ** 2, operator_off_reads=len(outcomes),
                               verified_witnesses=witnesses, supported=outcomes, primary_observation_scores=scores))
        reopened = JournaledCorpusWorkspace(path, replace(frame, _encode=_forbid_encoding))
        assert reopened.snapshot().journal_sha256 == report[-1]["journal"]
        for item in report:
            historical = _history(reopened, item["journal_revision"])
            assert historical.journal_sha256 == item["journal"]
            restored = world.prepare(historical, packed=False)
            for source, target in item["supported"].items():
                read = world.reason_joint(encoder.code[source], memory=restored, **_REASONING_ARGS)
                assert read.read(encoder.code[target], floor=_READOUT_FLOOR).status == "supported"
    return report


def _bounded_read(path, limit):
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{path.name} exceeds its byte allowance")
    return raw


def _content_digest(journal, manifest):
    """Hash the ordered source-derived event payloads independent of the stream UUID."""
    metadata = {key: manifest[key] for key in ("schema", "package", "coordinates", "frame_sha256",
                                                "kernel_sha256", "read_contract")}
    metadata["pins"] = [pin[:2] for pin in manifest["pins"]]
    digest = hashlib.sha256(_json(metadata))
    with closing(sqlite3.connect(f"{journal.resolve().as_uri()}?mode=ro", uri=True)) as db:
        for revision, payload in db.execute("SELECT revision, payload FROM corpus_events ORDER BY revision"):
            digest.update(revision.to_bytes(8, "big"))
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def publish(repo, revisions, destination, *, package):
    """Write a source-only publication and return its custody and content digests."""
    encoder = _ImportEncoder(package)
    frame, world = _frame(encoder)
    destination = Path(destination)
    destination.mkdir()
    journal = destination / "corpus.sqlite"
    workspace = JournaledCorpusWorkspace(journal, frame, create=True)
    pins = [[commit, pin.revision, pin.journal_sha256]
            for commit, pin, _, _ in _publications(repo, revisions, package, encoder, world, workspace)]
    if any(journal.with_name(journal.name + suffix).exists() for suffix in ("-journal", "-wal", "-shm")):
        raise ValueError("publication requires a standalone finalized journal")
    data = dict(schema=_PUBLICATION_SCHEMA, package=package, coordinates=list(encoder.code),
                journal_sha256=sha256(_bounded_read(journal, _MAX_JOURNAL_BYTES)),
                frame_sha256=_frame_digest(frame).hex(), kernel_sha256=world._kernel_sha256,
                read_contract=_READ_CONTRACT, pins=pins)
    data["content_sha256"] = _content_digest(journal, data)
    manifest = _json(data)
    if len(manifest) > _MAX_JSON_BYTES:
        raise ValueError("publication manifest exceeds its byte allowance")
    # The manifest is written last. The caller records its digest separately.
    staging = destination / "publication.pending"
    with staging.open("xb") as handle:
        handle.write(manifest)
    staging.rename(destination / "publication.json")
    return {"seal": sha256(manifest), "content_sha256": data["content_sha256"]}


def _read_manifest(directory, seal):
    require_digest(seal)
    raw = _bounded_read(directory / "publication.json", _MAX_JSON_BYTES)
    if sha256(raw) != seal:
        raise ValueError("publication seal mismatch")
    data = json.loads(raw)
    fields = {"schema", "package", "coordinates", "pins", "journal_sha256", "kernel_sha256",
              "frame_sha256", "content_sha256", "read_contract"}
    if (type(data) is not dict or data.keys() != fields or data["schema"] != _PUBLICATION_SCHEMA
            or not _name(data["package"]) or data["read_contract"] != _READ_CONTRACT or _json(data) != raw):
        raise ValueError("invalid publication manifest")
    names, pins = data["coordinates"], data["pins"]
    if (type(names) is not list or not 1 <= len(names) <= _MAX_COORDINATES
            or not all(_name(name) for name in names) or len(set(names)) != len(names)
            or type(pins) is not list or not 1 <= len(pins) <= _MAX_REVISIONS):
        raise ValueError("invalid publication coordinate or pin inventory")
    previous = 0
    for pin in pins:
        if (type(pin) is not list or len(pin) != 3 or not _git_commit(pin[0])
                or type(pin[1]) is not int or not max(1, previous) <= pin[1] <= 2 * _MAX_REVISIONS):
            raise ValueError("invalid publication pin")
        previous = pin[1]
        require_digest(pin[2])
    for field in ("journal_sha256", "kernel_sha256", "frame_sha256", "content_sha256"):
        require_digest(data[field])
    return data


def _git_commit(value):
    return type(value) is str and len(value) in (40, 64) and all(c in "0123456789abcdef" for c in value)


def _questions(path, pin_count):
    def unique_object(pairs):
        row = dict(pairs)
        if len(row) != len(pairs):
            raise ValueError("duplicate question field")
        return row

    raw = _bounded_read(path, _MAX_JSON_BYTES)
    questions = json.loads(raw, object_pairs_hook=unique_object)
    if type(questions) is not list or not 1 <= len(questions) <= 4096:
        raise ValueError("expected a bounded list of named questions")
    ids = set()
    for row in questions:
        if (type(row) is not dict or row.keys() != {"id", "pin", "start", "target"}
                or type(row["id"]) is not str or not 1 <= len(row["id"]) <= 128 or row["id"] in ids
                or type(row["pin"]) is not int or not 0 <= row["pin"] < pin_count
                or not _name(row["start"]) or not _name(row["target"])):
            raise ValueError("questions may contain only a unique id, pin index, start and target")
        ids.add(row["id"])
    return questions, sha256(raw)


def _verify_names(pin, code):
    for name in pin.memory.source_digests:
        if name not in code:
            raise ValueError("publication named frame omits an active source")
        observations = pin.memory.source_mentions(name) or ()
        ordinary = tuple(row for row in observations if row.predicate in ("context", "relay"))
        entries = tuple(row for row in observations if row.predicate == "transport_entry")
        if len(ordinary) != 1 or any(not np.array_equal(row.vector, code[name]) for row in (*ordinary, *entries)):
            raise ValueError("publication named frame disagrees with the source observation")


def _verified_pin(workspace, code, record):
    _, revision, digest = record
    pin = _history(workspace, revision)
    if pin.journal_sha256 != digest:
        raise ValueError("publication journal pin does not match the manifest")
    _verify_names(pin, code)
    return pin


def _read_questions(workspace, world, code, pins, questions, arm):
    prepared, reads, results = {}, {}, []
    for question in questions:
        number = question["pin"]
        commit, _, digest = pins[number]
        if number not in prepared:
            pin = _verified_pin(workspace, code, pins[number])
            bank = None if arm == "memory-omitted" else world.prepare(pin, packed=False, operators=arm == "on")
            prepared[number] = (pin, bank)
        pin, bank = prepared[number]
        start, target = question["start"], question["target"]
        report = dict(question, commit=commit, journal=digest, arm=arm)
        if start not in code or target not in code:
            report.update(status="unknown_name", branches=[])
        elif start not in pin.memory.source_digests or target not in pin.memory.source_digests:
            report.update(status="outside_pin", branches=[])
        else:
            key = number, start
            if key not in reads:
                reads[key] = world.reason_joint(code[start], memory=bank, **_REASONING_ARGS)
            outcome = reads[key].read(code[target], floor=_READOUT_FLOOR)
            report.update(status=outcome.status, branches=[
                dict(witnesses=[asdict(w) for w in branch.witnesses],
                     transports=[asdict(use) for use in branch.transport_uses])
                for branch in outcome.support if outcome.status == "supported"
            ])
        results.append(report)
    return results


@contextmanager
def _replay_publication(directory, manifest):
    journal = _bounded_read(directory / "corpus.sqlite", _MAX_JOURNAL_BYTES)
    if sha256(journal) != manifest["journal_sha256"]:
        raise ValueError("publication journal digest mismatch")
    encoder = _ImportEncoder(manifest["package"])
    frame, world = _frame(encoder)
    if (world._kernel_sha256 != manifest["kernel_sha256"]
            or _frame_digest(frame).hex() != manifest["frame_sha256"]):
        raise ValueError("publication frame or kernel does not match the reader")
    code = dict(zip(manifest["coordinates"], encoder.axes[:len(manifest["coordinates"])], strict=True))
    # Replay a private copy of the bytes just checked. No later read of the
    # publisher's working file can alter the observed lineage.
    with TemporaryDirectory(prefix="lsr-publication-read-") as scratch:
        copy = Path(scratch) / "corpus.sqlite"
        copy.write_bytes(journal)
        if _content_digest(copy, manifest) != manifest["content_sha256"]:
            raise ValueError("publication source content digest mismatch")
        workspace = JournaledCorpusWorkspace(copy, replace(frame, _encode=_forbid_encoding))
        tip = workspace.snapshot()
        pins = manifest["pins"]
        if (tip.revision, tip.journal_sha256) != tuple(pins[-1][1:]):
            raise ValueError("publication journal tip does not match the manifest")
        yield workspace, world, code, pins


def verify_publication(directory, seal):
    """Check all retained pins before a controller releases tasks."""
    directory = Path(directory)
    manifest = _read_manifest(directory, seal)
    with _replay_publication(directory, manifest) as (workspace, _, code, pins):
        for record in pins:
            _verified_pin(workspace, code, record)


def read_publication(directory, seal, questions_path, *, arm="on"):
    """Read late questions from a pinned publication with no repository or encoder input."""
    if arm not in ("on", "operators-off", "memory-omitted"):
        raise ValueError("invalid publication read arm")
    directory = Path(directory)
    manifest = _read_manifest(directory, seal)
    questions, questions_digest = _questions(Path(questions_path), len(manifest["pins"]))
    with _replay_publication(directory, manifest) as (workspace, world, code, pins):
        results = _read_questions(workspace, world, code, pins, questions, arm)
        return {"seal": seal, "questions_sha256": questions_digest, "results": results}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    action, *rest = argv or [""]
    parser = argparse.ArgumentParser(description=__doc__)
    if action == "--read":
        parser.add_argument("publication", type=Path)
        parser.add_argument("seal")
        parser.add_argument("questions", type=Path)
        parser.add_argument("--arm", choices=("on", "operators-off", "memory-omitted"), default="on")
        args = parser.parse_args(rest)
        result = read_publication(args.publication, args.seal, args.questions, arm=args.arm)
    else:
        if action == "--publish":
            parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("revisions", nargs="+")
        parser.add_argument("--repo", default=".", type=Path)
        parser.add_argument("--package", default="experiments.memory")
        args = parser.parse_args(rest if action == "--publish" else argv)
        if action == "--publish":
            result = publish(args.repo, args.revisions, args.output, package=args.package)
        else:
            result = run(args.repo, args.revisions, package=args.package)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
