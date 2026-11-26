"""Query-blind acquisition of scoped binary rules from registered sources."""

from dataclasses import dataclass, replace
import json
import re
from types import MappingProxyType

import numpy as np

from experiments.memory.corpus_bindings import BinaryFactorClause, BindingMention, CorpusBindingMemory, CorpusRevision
from experiments.memory.latent_factor_field import LatentFactorField, LatentFactorPin
from experiments.memory.verified_source_tool import sha256


_SETTING = r"([A-Z][A-Z0-9_]*)"
_FLAGS = re.ASCII | re.IGNORECASE
_LEVEL = r"(on|off|enabled|disabled)"
_BETWEEN = re.compile(
    rf"Setting {_SETTING} must (be the same as|match|be different from|not match) setting {_SETTING}\.", _FLAGS,
)
_PAIR = re.compile(rf"Settings {_SETTING} and {_SETTING} must (match|agree|differ)\.", _FLAGS)
_UNARY = re.compile(rf"Setting {_SETTING} must be {_LEVEL}\.", _FLAGS)
_IF = re.compile(rf"If setting {_SETTING} is {_LEVEL}, setting {_SETTING} must be {_LEVEL}\.", _FLAGS)
_NAME = re.compile(r"[A-Z][A-Z0-9_]*", re.ASCII)
_VECTOR = (1.,)


def _ids(value):
    if isinstance(value, str):
        raise ValueError("declare source IDs as a collection")
    items = tuple(value)
    if any(type(item) is not str or not item for item in items) or len(set(items)) != len(items):
        raise ValueError("source IDs must be distinct nonempty strings")
    return frozenset(items)


def _code(word):
    return int(word.lower() in ("on", "enabled"))


@dataclass(frozen=True, slots=True)
class FactorAcquisition:
    pin: LatentFactorPin
    refused: MappingProxyType
    installed_batches: int
    installed_clauses: int


class RegisteredFactorCompiler:
    """Use a fixed codebook and source authority registry before tasks exist."""

    def __init__(self, settings, *, authoritative_sources, reference_sources=()):
        if isinstance(settings, str):
            raise ValueError("declare settings in their stable axis order")
        settings = tuple(settings)
        if (not settings or any(type(name) is not str or _NAME.fullmatch(name) is None for name in settings)
                or len(set(settings)) != len(settings)):
            raise ValueError("settings must be distinct uppercase ASCII identifiers")
        authority, references = _ids(authoritative_sources), _ids(reference_sources)
        if not authority or authority & references:
            raise ValueError("authority and reference source registries must be disjoint")
        patterns = [(pattern.pattern, pattern.flags) for pattern in (_BETWEEN, _PAIR, _UNARY, _IF)]
        wire = json.dumps([settings, sorted(authority), sorted(references), patterns],
                          ensure_ascii=True, separators=(",", ":")).encode("ascii")
        self.compiler_id = "registered-binary-clauses-v1:" + sha256(wire)
        self._axis = MappingProxyType({name: i for i, name in enumerate(settings)})
        self._authority, self._references = authority, references
        self.field = LatentFactorField((2,) * len(settings), compiler_id=self.compiler_id)

    def frame(self):
        """Create the source encoder frame recorded by the corpus journal."""
        return CorpusBindingMemory.prepare(
            ("hard_clause", "unresolved_clause"), [[1.]], {}, {}, encode=self._encode,
            encoder_version=self.compiler_id, min_similarity=1., gap=0.,
        )

    def _encode(self, raw):
        offset = 0
        for line in raw.splitlines(keepends=True):
            content = line.rstrip(b" \t\r\n")
            clause = content.lstrip(b" \t")
            if clause:
                start = offset + len(content) - len(clause)
                try:
                    text = clause.decode("utf8")
                except UnicodeDecodeError:
                    yield BindingMention("unresolved_clause", _VECTOR, 1.)
                else:
                    factor = self._parse(text)
                    predicate = "hard_clause" if factor is not None else "unresolved_clause"
                    yield BindingMention(predicate, _VECTOR, 1., (start, offset + len(content)), factor)
            offset += len(line)

    def _parse(self, text):
        try:
            matched = _BETWEEN.fullmatch(text)
            if matched is not None:
                left, relation, right = matched.groups()
                mask = 9 if relation.lower() in ("be the same as", "match") else 6
                return self._pair(left, right, mask)
            matched = _PAIR.fullmatch(text)
            if matched is not None:
                left, right, relation = matched.groups()
                return self._pair(left, right, 6 if relation.lower() == "differ" else 9)
            matched = _UNARY.fullmatch(text)
            if matched is not None:
                slot, value = matched.groups()
                return BinaryFactorClause((self._axis[slot.upper()],), 1 << _code(value))
            matched = _IF.fullmatch(text)
            if matched is not None:
                left, premise, right, conclusion = matched.groups()
                excluded = 2 * _code(premise) + (1 - _code(conclusion))
                return self._pair(left, right, 15 ^ (1 << excluded))
        except KeyError:
            # An unregistered identity never becomes an approximate match.
            return None
        return None

    def _pair(self, left, right, mask):
        first, second = self._axis[left.upper()], self._axis[right.upper()]
        if first == second:
            return BinaryFactorClause((first,), (mask & 1) | ((mask & 8) >> 2))
        return BinaryFactorClause((first, second), mask)

    def prepare(self, corpus, *, previous=None, max_batches=1024, max_clauses=8192,
                max_updates=1024, max_tensor_cells=1_000_000):
        """Install recorded clauses on a source pin without reopening any source."""
        if type(corpus) is not CorpusRevision or corpus.journal_sha256 is None:
            raise ValueError("registered acquisition requires a journaled source pin")
        if corpus.memory.encoder_version != self.compiler_id:
            raise ValueError("the source frame does not match the registered compiler")
        if any(type(value) is not int or value < 0 for value in (max_batches, max_clauses)):
            raise ValueError("acquisition allowances must be nonnegative integers")
        self.field._require_pin(previous)
        if previous is not None and previous.corpus._owner is not corpus._owner:
            raise ValueError("corpus revision belongs to a different workspace")
        if previous is not None and previous._versions.keys() - previous._recorded_sources:
            raise ValueError("registered acquisition cannot inherit supplied factor batches")
        prior = {} if previous is None else previous._versions
        batches, refused = [], {}
        installed = 0
        for source in sorted(corpus.memory.source_digests):
            if prior.get(source) == self.field._source_version(corpus, source):
                continue
            if source in self._references:
                clauses = ()
            elif source not in self._authority:
                refused[source] = "authority_unregistered"
                continue
            else:
                clauses = corpus.memory._factor_clauses.get(source)
                mentions = corpus.memory._mentions.get(source)
                if mentions is None:
                    refused[source] = "acquisition_unavailable"
                    continue
                if not mentions or any(mention[0] == "unresolved_clause" for mention in mentions):
                    refused[source] = "clause_unresolved"
                    continue
                if clauses is None or any(clause is None or mention[:2] != ("hard_clause", 1.)
                                          for clause, mention in zip(clauses, mentions, strict=True)):
                    refused[source] = "acquisition_unavailable"
                    continue
            if len(batches) >= max_batches or installed + len(clauses) > max_clauses:
                refused[source] = "acquisition_limit"
                continue
            batch = self.field.compile_source(corpus, source, self._entries(clauses))
            batches.append(replace(batch, _recorded=True))
            installed += len(clauses)
        pin = self.field.prepare(corpus, batches, previous=previous, max_updates=max_updates,
                                 max_tensor_cells=max_tensor_cells)
        return FactorAcquisition(pin, MappingProxyType(refused), len(batches), installed)

    def _entries(self, clauses):
        for ordinal, clause in enumerate(clauses):
            width = len(clause.axes)
            values = [(clause.allowed >> code) & 1 for code in range(1 << width)]
            yield ordinal, clause.axes, np.asarray(values, dtype=float).reshape((2,) * width)
