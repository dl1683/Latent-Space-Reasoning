"""Visible raw-source control using committed Python declarations.

Tasks are generated after each publication in one trusted process.
This control neither trains a model nor certifies Python runtime imports.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import numpy as np

from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory
from experiments.memory.latent_feedback import FeedbackKernel
from experiments.memory.recurrent_world import RecurrentWorld
from experiments.memory.repository_import_memory import (
    RepositoryImportMemory, _decode_python, _git, _module_imports, _module_tree, _read_blobs,
)
from experiments.memory.verified_source_tool import sha256
from experiments.memory.world_transport import DiscoveredTransportPolicy


_MAX_COORDINATES = 256


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
        params = dict(steps=2, max_hypotheses=1, null_logit=8., margin=1.)
        finding = world.reason_joint(start, memory=bank, **params)
        source_scores += finding.observation_scores
        expected = _expected(name, graph)
        supported = [target for target, direction in code.items()
                     if finding.read(direction, floor=.1).status == "supported"]
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
        ablated = world.reason_joint(start, memory=off, **params)
        assert ablated.read(code[expected], floor=.1).status == "missing"
        outcomes[name] = expected
    return outcomes, source_scores, checked_witnesses


def run(repo, revisions, *, package):
    if not 1 <= len(revisions) <= 8 or not all(part.isidentifier() for part in package.split(".")):
        raise ValueError("expected a Python package and no more than eight pinned revisions")
    encoder = _ImportEncoder(package)
    frame, world = _frame(encoder)
    raw, previous, report, exact, bank = {}, {}, [], None, None
    with TemporaryDirectory(prefix="lsr-import-control-") as directory:
        path = Path(directory) / "corpus.sqlite"
        workspace = JournaledCorpusWorkspace(path, frame, create=True)
        for rev in revisions:
            commit, tree = _tree(repo, rev, package)
            changes, byte_count = _load_changes(repo, tree, previous)
            encoder.stage(tree, changes)
            for name, contents in changes.items():
                if contents is None:
                    raw.pop(name)
                else:
                    raw[name] = contents
            source = workspace.publish(workspace.prepare_changes(workspace.snapshot(), changes))
            pin = world.maintain_transports(workspace, source)
            bank = world.prepare(pin, previous=bank, packed=False)
            # Read tasks and exact comparisons are constructed only after the publication.
            exact = (RepositoryImportMemory.prepare(repo, commit, package=package, source_root=".")
                     if exact is None else exact.revise(commit))
            active_code = {name: encoder.code[name] for name in sorted(tree)}
            outcomes, scores, witnesses = _evaluate(world, pin, bank, active_code, exact, raw)
            report.append(dict(commit=commit, journal=pin.journal_sha256, journal_revision=pin.revision,
                               modules=len(active_code), changed_modules=len(changes), distinct_blob_bytes=byte_count,
                               projected_rows=bank.projected_rows, late_reads=len(active_code),
                               candidate_readouts=len(active_code) ** 2, operator_off_reads=len(outcomes),
                               verified_witnesses=witnesses, supported=outcomes, primary_observation_scores=scores))
            previous = tree

        def forbid_encoding(*_):
            raise AssertionError("recovery accessed the encoder")

        reopened = JournaledCorpusWorkspace(path, replace(frame, _encode=forbid_encoding))
        assert reopened.snapshot().journal_sha256 == report[-1]["journal"]
        for item in report:
            historical = reopened.snapshot_at(item["journal_revision"], max_events=2 * len(report) + 1,
                                              max_payload_bytes=16 * 1024 * 1024)
            assert historical.journal_sha256 == item["journal"]
            restored = world.prepare(historical, packed=False)
            for source, target in item["supported"].items():
                read = world.reason_joint(encoder.code[source], memory=restored, steps=2, max_hypotheses=1,
                                          null_logit=8., margin=1.)
                assert read.read(encoder.code[target], floor=.1).status == "supported"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revisions", nargs="+")
    parser.add_argument("--repo", default=".", type=Path)
    parser.add_argument("--package", default="experiments.memory")
    args = parser.parse_args()
    print(json.dumps(run(args.repo, args.revisions, package=args.package), indent=2))


if __name__ == "__main__":
    main()
