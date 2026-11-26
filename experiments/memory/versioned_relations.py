"""Exact, incrementally maintained typed relations with pinned source receipts."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from threading import Lock
from types import MappingProxyType

from experiments.memory.verified_source_tool import SourceSpan


_EMPTY = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Relation:
    subject: str
    role: str
    object: str
    span: SourceSpan

    def __post_init__(self):
        if any(type(value) is not str or not value.strip() for value in (self.subject, self.role, self.object)):
            raise ValueError("relation requires a subject, role and object")
        if type(self.span) is not SourceSpan:
            raise TypeError("relation requires a pinned source span")


@dataclass(frozen=True, slots=True)
class RelationStep:
    role: str
    reverse: bool = False

    def __post_init__(self):
        if type(self.role) is not str or not self.role.strip() or type(self.reverse) is not bool:
            raise ValueError("step requires a role and a boolean direction")


@dataclass(frozen=True, slots=True)
class RelationPath:
    endpoint: str
    evidence: tuple[Relation, ...]


@dataclass(frozen=True, slots=True)
class RelationRead:
    status: str
    paths: tuple[RelationPath, ...]
    edge_visits: int
    posting_visits: int
    cache_checks: int = 0
    cached: bool = False


_PostingKey = tuple[str, bool, str]


def _relation_order(row):
    span = row.span
    return (row.subject, row.role, row.object, span.source_id, span.version_sha256, span.start_byte,
            span.end_byte, span.span_sha256, span.start_character is not None,
            span.start_character or 0, span.end_character or 0)


@dataclass(frozen=True, slots=True, eq=False)
class RelationIndex:
    """An immutable exact index. Source edits replace all assertions from that origin."""

    _sources: Mapping[str, tuple[Relation, ...]]
    _postings: Mapping[_PostingKey, tuple[Relation, ...]]
    updated_postings: int

    @classmethod
    def empty(cls):
        return cls(_EMPTY, _EMPTY, 0)

    def revise(self, changes):
        sources = dict(self._sources)
        edits = {}

        def bucket(key):
            if key not in edits:
                edits[key] = set(self._postings.get(key, ()))
            return edits[key]

        for source, replacement in changes.items():
            if type(source) is not str or not source.strip():
                raise ValueError("source identity is required")
            old = sources.pop(source, ())
            if replacement is None:
                new = ()
            else:
                new = tuple(replacement)
                if (any(type(row) is not Relation or row.span.source_id != source for row in new)
                        or len({row.span.version_sha256 for row in new}) > 1 or len(set(new)) != len(new)):
                    raise ValueError("source assertions require one version and distinct proofs")
                sources[source] = new
            for rows, remove in ((set(old) - set(new), True), (set(new) - set(old), False)):
                for row in rows:
                    for key in ((row.role, False, row.subject), (row.role, True, row.object)):
                        members = bucket(key)
                        if remove:
                            members.remove(row)
                        else:
                            members.add(row)

        if not edits:
            return RelationIndex(MappingProxyType(sources), self._postings, 0)
        postings = dict(self._postings)
        for key, rows in edits.items():
            if rows:
                postings[key] = tuple(sorted(rows, key=_relation_order))
            else:
                del postings[key]
        return RelationIndex(MappingProxyType(sources), MappingProxyType(postings), len(edits))

    @property
    def source_count(self):
        return len(self._sources)

    @property
    def assertion_count(self):
        return sum(map(len, self._sources.values()))


@dataclass(frozen=True, slots=True)
class _CachedRead:
    result: RelationRead
    dependencies: tuple[tuple[_PostingKey, tuple[Relation, ...]], ...]

    def check(self, snapshot):
        for count, (key, posting) in enumerate(self.dependencies, 1):
            if snapshot._postings.get(key, ()) is not posting:
                return False, count
        return True, len(self.dependencies)


class ExactRelationReader:
    """Enumerate full bounded proofs and retain results while their postings survive."""

    def __init__(self, *, cache_size=128):
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache size must be a nonnegative integer")
        self._cache_size = cache_size
        self._cache = OrderedDict()
        self._lock = Lock()

    def read(self, start, steps, *, memory=None, max_edges=4096, distinct_sources=False):
        steps = tuple(steps)
        if (type(start) is not str or not start.strip() or not steps
                or any(type(step) is not RelationStep for step in steps)
                or type(max_edges) is not int or max_edges < 0 or type(distinct_sources) is not bool):
            raise ValueError("a read needs a start, typed steps and valid work bounds")
        if memory is None:
            return RelationRead("memory_omitted", (), 0, 0)
        if type(memory) is not RelationIndex:
            raise TypeError("read memory must be a relation index")
        key = start, steps, max_edges, distinct_sources
        with self._lock:
            cached = self._cache.get(key)
            checks = 0
            if cached is not None:
                valid, checks = cached.check(memory)
                if valid:
                    self._cache.move_to_end(key)
                    return replace(cached.result, edge_visits=0, posting_visits=0,
                                   cache_checks=checks, cached=True)
            result, dependencies = self._enumerate(start, steps, memory, max_edges, distinct_sources)
            if self._cache_size and result.status != "work_limit":
                self._cache[key] = _CachedRead(result, tuple(dependencies.items()))
                self._cache.move_to_end(key)
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return replace(result, cache_checks=checks)

    @staticmethod
    def _enumerate(start, steps, memory, max_edges, distinct_sources):
        paths = (RelationPath(start, ()),)
        dependencies = {}
        edges = postings = 0
        for step in steps:
            next_paths = []
            for path in paths:
                key = step.role, step.reverse, path.endpoint
                rows = memory._postings.get(key, ())
                # Record empty lookups so later competitors and filled gaps invalidate a miss.
                dependencies[key] = rows
                postings += 1
                for row in rows:
                    if edges == max_edges:
                        return RelationRead("work_limit", (), edges, postings), dependencies
                    edges += 1
                    if distinct_sources and any(item.span.source_id == row.span.source_id for item in path.evidence):
                        continue
                    endpoint = row.subject if step.reverse else row.object
                    next_paths.append(RelationPath(endpoint, (*path.evidence, row)))
            paths = tuple(next_paths)
        return RelationRead("matched" if paths else "no_path", paths, edges, postings), dependencies
