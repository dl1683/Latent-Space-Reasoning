"""Visible warm exact relation control over committed Python PEP documents."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType
from uuid import uuid4

from experiments.memory.git_source import git_read, read_blobs
from experiments.memory.source_revision_feed import write_source_delta
from experiments.memory.verified_source_tool import SourceSpan, sha256
from experiments.memory.versioned_relations import ExactRelationReader, Relation, RelationIndex, RelationStep


_PATH = re.compile(rb"peps/pep-([0-9]{4})\.rst")
_HEADER_END = re.compile(rb"\r?\n[ \t]*\r?\n")
_FIELD = re.compile(rb"(?m)^([A-Za-z-]+):[ \t]*([^\r\n]*(?:\r?\n[ \t]+[^\r\n]+)*)")
_PEP_LIST = re.compile(r"[0-9]+(?:[ ,]+[0-9]+)*")
_RENAME = re.compile(
    rb"\x60\x60([A-Z][A-Z0-9_]+)\x60\x60\s+(?:[A-Za-z-]+\s+){0,3}"
    rb"is\s+renamed\s+to\s+\x60\x60([A-Z][A-Z0-9_]+)\x60\x60"
)
_PEP_ROLES = {b"Requires": "requires", b"Replaces": "replaces", b"Superseded-By": "superseded_by"}
_VALUE_ROLES = {b"Title": "title", b"Status": "status", b"Python-Version": "python_version"}


def _tree(repo, revision, *, allow_empty=False):
    commit = git_read(repo, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}").decode("ascii").strip()
    files = {}
    for entry in git_read(repo, "ls-tree", "-rz", "--full-tree", commit, "--", "peps").split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        match = _PATH.fullmatch(path)
        mode, kind, blob = meta.split()
        if match and kind == b"blob" and mode in (b"100644", b"100755"):
            files[path.decode("ascii")] = int(match[1]), blob.decode("ascii")
    if not files and not allow_empty:
        raise ValueError("opening revision has no tracked PEP text documents")
    if len({number for number, _ in files.values()}) != len(files):
        raise ValueError("revision has multiple source files for one PEP number")
    return commit, files


def _read_document(source, number, raw):
    boundary = _HEADER_END.search(raw)
    if boundary is None:
        raise ValueError(f"missing PEP header boundary in {source}")
    fields = list(_FIELD.finditer(raw, 0, boundary.start()))
    identifiers = [match[2].strip() for match in fields if match[1] == b"PEP"]
    if len(identifiers) != 1 or not identifiers[0].isdigit() or int(identifiers[0]) != number:
        raise ValueError(f"PEP header disagrees with the committed path {source}")
    digest = sha256(raw)
    pep = f"pep:{number}"
    relations = []

    def add(subject, role, target, bounds):
        start, end = bounds
        pointer = SourceSpan(source, digest, start, end, sha256(raw[start:end]))
        relations.append(Relation(subject, role, target, pointer))

    for match in fields:
        field = match[1]
        if field not in _PEP_ROLES and field not in _VALUE_ROLES:
            continue
        value = " ".join(match[2].decode("utf8").split())
        if field in _PEP_ROLES:
            if not _PEP_LIST.fullmatch(value):
                raise ValueError(f"unrecognized {field.decode('ascii')} list in {source}")
            for target in dict.fromkeys(int(n) for n in re.findall(r"[0-9]+", value)):
                add(pep, _PEP_ROLES[field], f"pep:{target}", match.span())
        elif value:
            role = _VALUE_ROLES[field]
            add(pep, role, f"{role}:{value}", match.span())

    for match in _RENAME.finditer(raw, boundary.end()):
        earlier, later = (name.decode("ascii") for name in match.groups())
        add(f"symbol:{number}:{earlier}", "renamed_to", f"symbol:{number}:{later}", match.span())
    return tuple(relations)


@dataclass(frozen=True, slots=True)
class PepRelationCorpus:
    """Warm typed view of one Git pin, with exact reopening available separately."""

    repo: Path
    commit: str
    memory: RelationIndex
    _blobs: MappingProxyType
    records_read: int
    source_bytes: int

    @classmethod
    def prepare(cls, repo, revision):
        return cls(Path(repo).resolve(), "", RelationIndex.empty(), MappingProxyType({}), 0, 0).revise(revision)

    def revise(self, revision):
        commit, documents = _tree(self.repo, revision, allow_empty=bool(self.commit))
        written = {source: (number, blob) for source, (number, blob) in documents.items()
                   if self._blobs.get(source) != blob}
        blobs = read_blobs(self.repo, sorted({blob for _, blob in written.values()}))
        changes = {source: None for source in self._blobs.keys() - documents.keys()}
        for source, (number, blob) in written.items():
            changes[source] = _read_document(source, number, blobs[blob])
        return PepRelationCorpus(self.repo, commit, self.memory.revise(changes),
                                 MappingProxyType({source: blob for source, (_, blob) in documents.items()}),
                                 len(written), sum(map(len, blobs.values())))

    @property
    def scope_sha256(self):
        manifest = json.dumps(dict(self._blobs), sort_keys=True, separators=(",", ":"))
        return sha256(manifest.encode("ascii"))

    def verify(self, spans):
        """Read exact pinned blobs and verify the named proof bytes."""
        spans = tuple(dict.fromkeys(spans))
        if any(type(span) is not SourceSpan or span.source_id not in self._blobs for span in spans):
            raise ValueError("proof span is outside the pinned source scope")
        blobs = read_blobs(self.repo, sorted({self._blobs[span.source_id] for span in spans}))
        versions = {blob: sha256(raw) for blob, raw in blobs.items()}
        for span in spans:
            blob = self._blobs[span.source_id]
            raw = blobs[blob]
            if (versions[blob] != span.version_sha256 or span.end_byte > len(raw)
                    or sha256(raw[span.start_byte:span.end_byte]) != span.span_sha256):
                raise ValueError("proof span does not match the pinned source")
        return dict(checked_spans=len(spans), bytes_read=sum(map(len, blobs.values())))


def export_source_feed(repo, revisions, output):
    """Write source-only epoch inputs and a separate controller custody catalogue."""
    output = Path(output)
    output.mkdir(mode=0o700)
    epoch_root = output / "epochs"
    epoch_root.mkdir()
    identities, previous, feed, reports, custody = {}, {}, None, [], []
    for revision in revisions:
        commit, tree = _tree(repo, revision, allow_empty=feed is not None)
        changed = {source: blob for source, (_, blob) in tree.items() if blob != previous.get(source)}
        blobs = read_blobs(repo, sorted(set(changed.values())))
        for source in sorted(tree.keys() - identities.keys()):
            identities[source] = uuid4().hex
        changes = {identities[source]: None for source in previous.keys() - tree.keys()}
        changes.update({identities[source]: blobs[blob] for source, blob in changed.items()})
        epoch = len(reports) + 1
        feed = write_source_delta(epoch_root / f"{epoch:04d}", changes, previous=feed)
        reports.append(dict(epoch=epoch, seal=feed.seal, sources=len(feed.manifest),
                            changed_sources=len(changes), unique_source_bytes_read=sum(map(len, blobs.values()))))
        custody.append(dict(commit=commit, seal=feed.seal,
                            sources={identities[source]: dict(path=source, git_blob=blob)
                                     for source, (_, blob) in tree.items()}))
        previous = {source: blob for source, (_, blob) in tree.items()}
    (output / "custody.json").write_text(json.dumps(custody, sort_keys=True, separators=(",", ":")), encoding="utf8")
    return reports


def _queries(arguments):
    return [(start, tuple(RelationStep(role.removeprefix("^"), role.startswith("^")) for role in roles))
            for start, *roles in arguments]


def run(repo, revisions, queries, *, max_edges=4096, distinct_sources=False, cache_size=128):
    reader = ExactRelationReader(cache_size=cache_size)
    snapshot = None
    report = []
    for revision in revisions:
        snapshot = (PepRelationCorpus.prepare(repo, revision) if snapshot is None else snapshot.revise(revision))
        answers, evidence = [], []
        for start, steps in queries:
            read = reader.read(start, steps, memory=snapshot.memory, max_edges=max_edges,
                               distinct_sources=distinct_sources)
            answers.append(dict(start=start, steps=[asdict(step) for step in steps], **asdict(read)))
            evidence.extend(relation.span for path in read.paths for relation in path.evidence)
        report.append(dict(commit=snapshot.commit, scope_sha256=snapshot.scope_sha256,
                           sources=snapshot.memory.source_count,
                           assertions=snapshot.memory.assertion_count, source_records_read=snapshot.records_read,
                           unique_source_bytes_read=snapshot.source_bytes, updated_postings=snapshot.memory.updated_postings,
                           reads=answers, verified=snapshot.verify(evidence)))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("revisions", nargs="+")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--query", nargs="+", action="append", metavar=("START", "ROLE"))
    action.add_argument("--source-output", type=Path, help="new directory for raw epochs and separate custody data")
    parser.add_argument("--max-edges", type=int, default=4096)
    parser.add_argument("--cache-size", type=int, default=128, help="retained reads, or zero for direct traversal")
    parser.add_argument("--distinct-sources", action="store_true")
    options = parser.parse_args(argv)
    if options.source_output is not None:
        results = export_source_feed(options.repo, options.revisions, options.source_output)
    else:
        if any(len(query) < 2 for query in options.query):
            parser.error("each --query needs a starting entity and at least one role")
        results = run(options.repo, options.revisions, _queries(options.query),
                      max_edges=options.max_edges, distinct_sources=options.distinct_sources,
                      cache_size=options.cache_size)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
