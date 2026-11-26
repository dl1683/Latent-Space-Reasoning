"""Query-blind acquisition of scoped binary rules from registered sources."""

from dataclasses import dataclass, replace
import json
import re
from types import MappingProxyType

import numpy as np

from experiments.memory.corpus_bindings import (
    BinaryFactorClause, BindingMention, CorpusBindingMemory, CorpusRevision, NamedBinaryFactorClause, _factor_identities,
)
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

    @property
    def field(self):
        """The numerical field on this acquisition's identity schema."""
        return self.pin._field


class RegisteredFactorCompiler:
    """Acquire scoped hard factors from a trusted source authority registry."""

    def __init__(self, settings, *, authoritative_sources, reference_sources=(), allow_identity_growth=False):
        if type(allow_identity_growth) is not bool:
            raise ValueError("identity growth must be boolean")
        settings = _factor_identities(settings)
        authority, references = _ids(authoritative_sources), _ids(reference_sources)
        if not authority or authority & references:
            raise ValueError("authority and reference source registries must be disjoint")
        patterns = [(pattern.pattern, pattern.flags) for pattern in (_BETWEEN, _PAIR, _UNARY, _IF)]
        wire = json.dumps([settings, sorted(authority), sorted(references), patterns],
                          ensure_ascii=True, separators=(",", ":")).encode("ascii")
        scheme = "registered-binary-clauses-growth-v1" if allow_identity_growth else "registered-binary-clauses-v1"
        self.compiler_id = scheme + ":" + sha256(wire)
        self._settings = settings
        self._grow = allow_identity_growth
        self._axis = MappingProxyType({name: i for i, name in enumerate(settings)})
        self._authority, self._references = authority, references
        self.field = LatentFactorField((2,) * len(settings), compiler_id=self.compiler_id)

    def frame(self):
        """Create the source encoder frame recorded by the corpus journal."""
        frame = CorpusBindingMemory.prepare(
            ("hard_clause", "unresolved_clause"), [[1.]], {}, {}, encode=self._encode,
            encoder_version=self.compiler_id, min_similarity=1., gap=0.,
        )
        return replace(frame, _factor_registry_seed=self._settings) if self._grow else frame

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
                return self._clause((slot,), 1 << _code(value))
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
        if left.upper() == right.upper():
            return self._clause((left,), (mask & 1) | ((mask & 8) >> 2))
        return self._clause((left, right), mask)

    def _clause(self, names, mask):
        keys = tuple(name.upper() for name in names)
        if self._grow:
            return NamedBinaryFactorClause(keys, mask)
        return BinaryFactorClause(tuple(self._axis[key] for key in keys), mask)

    def _field_id(self, names):
        if names == self._settings:
            return self.compiler_id
        wire = json.dumps(names, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        return self.compiler_id + ":" + sha256(wire)

    def _field_for(self, corpus, previous):
        if not self._grow:
            self.field._require_pin(previous)
            return self.field, previous, self._settings
        names = corpus.factor_identities
        if names is None or names[:len(self._settings)] != self._settings:
            raise ValueError("factor identities must extend the compiler seed")
        if previous is None:
            field = (self.field if names == self._settings
                     else LatentFactorField((2,) * len(names), compiler_id=self._field_id(names)))
            return field, previous, names
        field = previous._field
        prior_names = previous.corpus.factor_identities
        if (prior_names is None or field._compiler_id != self._field_id(prior_names)
                or names[:len(prior_names)] != prior_names):
            raise ValueError("previous factor pin does not use a prefix of this identity registry")
        if len(names) > len(prior_names):
            field, previous = field._extend_binary_slots(previous, len(names), self._field_id(names))
        return field, previous, names

    def prepare(self, corpus, *, previous=None, max_batches=1024, max_clauses=8192,
                max_updates=1024, max_tensor_cells=1_000_000):
        """Install recorded clauses on a source pin without reopening any source."""
        if type(corpus) is not CorpusRevision or corpus.journal_sha256 is None:
            raise ValueError("registered acquisition requires a journaled source pin")
        if corpus.memory.encoder_version != self.compiler_id:
            raise ValueError("the source frame does not match the registered compiler")
        if any(type(value) is not int or value < 0 for value in (max_batches, max_clauses)):
            raise ValueError("acquisition allowances must be nonnegative integers")
        if previous is not None:
            if type(previous) is not LatentFactorPin or previous.corpus._owner is not corpus._owner:
                raise ValueError("corpus revision belongs to a different workspace")
            if previous._versions.keys() - previous._recorded_sources:
                raise ValueError("registered acquisition cannot inherit supplied factor batches")
        field, previous, names = self._field_for(corpus, previous)
        axes = {name: slot for slot, name in enumerate(names)}
        prior = {} if previous is None else previous._versions
        batches, refused = [], {}
        installed = 0
        for source in sorted(corpus.memory.source_digests):
            if prior.get(source) == field._source_version(corpus, source):
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
            if self._grow and any(name not in axes for clause in clauses for name in clause.axes):
                refused[source] = "identity_unregistered"
                continue
            if len(batches) >= max_batches or installed + len(clauses) > max_clauses:
                refused[source] = "acquisition_limit"
                continue
            batch = field.compile_source(corpus, source, self._entries(clauses, axes))
            batches.append(replace(batch, _recorded=True))
            installed += len(clauses)
        pin = field.prepare(corpus, batches, previous=previous, max_updates=max_updates,
                                 max_tensor_cells=max_tensor_cells)
        return FactorAcquisition(pin, MappingProxyType(refused), len(batches), installed)

    def _entries(self, clauses, axes):
        for ordinal, clause in enumerate(clauses):
            width = len(clause.axes)
            values = [(clause.allowed >> code) & 1 for code in range(1 << width)]
            slots = tuple(axes[name] for name in clause.axes) if self._grow else clause.axes
            yield ordinal, slots, np.asarray(values, dtype=float).reshape((2,) * width)
