"""Project frozen source encodings into certain and possible binding banks."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from numbers import Real
from operator import index
from threading import Lock
from types import MappingProxyType

import numpy as np

from experiments.memory.binding_links import BindingLinkCondition, BindingLinkRead, BindingLinkStep, _RoleLinks, _role_hops
from experiments.memory.binding_memory import (
    BindingCapacityError, BindingMemory, BindingProof, _exclusions, _ProofWork, _proof_work_limit,
)
from experiments.memory.latent_feedback import _frozen, _real_array
from experiments.memory.role_validity import _RoleValidity
from experiments.memory.verified_source_tool import SourceSpan, VerifiedSourceTool


@dataclass(frozen=True)
class BindingMention:
    predicate: str
    vector: np.ndarray
    strength: float
    byte_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class CorpusBindingRead:
    certain: BindingProof | None
    possible_strength: float | None

    @property
    def is_exact(self):
        """Whether both modeled bounds agree, including modeled absence."""
        lower = None if self.certain is None else self.certain.strength
        return lower == self.possible_strength


_BindingRows = tuple[tuple[str, int, float], ...]
_MentionScope = tuple[int, int, str]
_StoredMention = tuple[str, float, bytes, int, _MentionScope | None]
_SOURCE_VECTOR_BYTES = 64 * 1024 * 1024


def _source_vector_limit(width):
    return _SOURCE_VECTOR_BYTES // (width * 8)


@dataclass(frozen=True)
class _EncodedChange:
    source: str
    digest: str | None
    certain: _BindingRows | None
    possible: _BindingRows | None
    outlier: bool
    mentions: tuple[_StoredMention, ...] | None = None


def _unit_rows(rows):
    scale = np.max(np.abs(rows), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("binding vectors must be nonzero")
    with np.errstate(under="ignore"):
        scaled = rows / scale
        return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


class _SourceObservationRefusal(BindingCapacityError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, eq=False)
class CorpusBindingMemory:
    """A revision of source digests, continuous observations and compiled evidence."""

    encoder_version: str
    source_digests: Mapping[str, str]
    outlier_sources: frozenset[str]
    _encode: Callable[[bytes], Iterable[BindingMention]]
    _anchors: np.ndarray
    _min_similarity: float
    _gap: float
    _expanded_limit: int
    _certain: BindingMemory
    _possible: BindingMemory
    _mentions: Mapping[str, tuple[_StoredMention, ...]]
    _role_links: _RoleLinks | None = None
    _scoped_sources: frozenset[str] = frozenset()

    @classmethod
    def prepare(cls, predicates, anchors, gates, sources, *, encode, encoder_version,
                min_similarity, gap, term_limit=64, retained_limit=64, expanded_limit=65536):
        anchors = _real_array(anchors)
        if anchors.ndim != 2 or not all(anchors.shape):
            raise ValueError("binding anchors must be a nonempty matrix")
        if not callable(encode) or type(encoder_version) is not str or not encoder_version:
            raise ValueError("a named encoder is required")
        if not -1 <= min_similarity <= 1 or not 0 <= gap <= 2:
            raise ValueError("invalid cosine similarity policy")
        expanded_limit = index(expanded_limit)
        if expanded_limit < 1:
            raise ValueError("expanded row limit must be positive")
        anchors = _frozen(_unit_rows(anchors))
        bank = BindingMemory.prepare(predicates, len(anchors), gates, {}, materialize=False,
                                     term_limit=term_limit, retained_limit=retained_limit)
        empty = cls(encoder_version, MappingProxyType({}), frozenset(), encode, anchors,
                    float(min_similarity), float(gap), expanded_limit, bank, bank, MappingProxyType({}))
        return empty.revise(sources)

    def _project(self, raw, remaining, vector_limit):
        certain, possible, mentions = [], [], []
        scopes = {}
        outlier = False
        for mention in self._encode(raw):
            if not isinstance(mention, BindingMention):
                raise ValueError("the encoder must emit binding mentions")
            if mention.predicate not in self._certain._predicates:
                raise ValueError("source rows must target declared predicates")
            if not isinstance(mention.strength, Real) or not 0 <= mention.strength <= 1:
                raise ValueError("binding strengths must be real numbers in [0, 1]")
            predicate, strength = str(mention.predicate), float(mention.strength)
            vector = _real_array(mention.vector)
            if vector.shape != (self._anchors.shape[1],):
                raise ValueError("mention vector must match the anchor dimension")
            unit = _unit_rows(vector[None, :])[0]
            scores = np.clip(self._anchors @ unit, -1, 1)
            coordinates = np.flatnonzero(
                (scores >= self._min_similarity) & (scores >= float(scores.max()) - self._gap)
            )
            if coordinates.size == 1:
                certain.append((predicate, int(coordinates[0]), strength))
            outlier |= coordinates.size == 0
            choices = coordinates if coordinates.size else range(len(self._anchors))
            remaining -= len(choices)
            if remaining < 0:
                raise BindingCapacityError("encoded binding expansion limit exceeded")
            possible.extend((predicate, int(coordinate), strength) for coordinate in choices)
            if len(mentions) >= vector_limit:
                raise BindingCapacityError("source vector batch byte limit exceeded")
            span = mention.byte_span
            scope = None
            if span is not None:
                if (type(span) is not tuple or len(span) != 2 or any(type(bound) is not int for bound in span)
                        or not 0 <= span[0] < span[1] <= len(raw)):
                    raise ValueError("binding mention requires a byte span inside its source")
                if span not in scopes:
                    piece = raw[span[0]:span[1]]
                    piece.decode("utf8")
                    scopes[span] = (*span, sha256(piece).hexdigest())
                scope = scopes[span]
            mentions.append((predicate, strength, np.asarray(unit, dtype="<f8").tobytes(),
                             int(coordinates.size), scope))
        return tuple(certain), tuple(possible), outlier, tuple(mentions), remaining

    def revise(self, changes):
        """Encode changed byte records and publish both banks together."""
        return self._apply_changes(self._prepare_changes(changes))

    def _prepare_changes(self, changes):
        staged = []
        remaining = self._expanded_limit
        vector_limit = _source_vector_limit(self._anchors.shape[1])
        for source, raw in changes.items():
            if type(source) is not str or not source:
                raise ValueError("source IDs must be nonempty strings")
            previous = self.source_digests.get(source)
            if raw is None:
                if previous is not None:
                    staged.append(_EncodedChange(source, None, None, None, False))
                continue
            if type(raw) is not bytes:
                raise ValueError("source records must be bytes")
            digest = sha256(raw).hexdigest()
            if digest != previous:
                certain, possible, outlier, mentions, remaining = self._project(raw, remaining, vector_limit)
                vector_limit -= len(mentions)
                staged.append(_EncodedChange(source, digest, certain, possible, outlier, mentions))
        return tuple(staged)

    def _apply_changes(self, staged):
        if not staged:
            return self
        digests = dict(self.source_digests)
        mentions = dict(self._mentions)
        scoped_sources = set(self._scoped_sources)
        outliers = set(self.outlier_sources)
        certain, possible = {}, {}
        for edit in staged:
            source = edit.source
            certain[source], possible[source] = edit.certain, edit.possible
            if edit.digest is None:
                del digests[source]
            else:
                digests[source] = edit.digest
            if edit.mentions is None:
                mentions.pop(source, None)
            else:
                mentions[source] = edit.mentions
            if any(mention[4] is not None for mention in edit.mentions or ()):
                scoped_sources.add(source)
            else:
                scoped_sources.discard(source)
            if edit.outlier:
                outliers.add(source)
            else:
                outliers.discard(source)
        lower = self._certain.revise(certain)
        upper = lower if self._certain is self._possible and certain == possible else self._possible.revise(possible)
        links = self._role_links
        if links is not None:
            links = links.revise((edit.source for edit in staged), digests, mentions)
        return replace(self, source_digests=MappingProxyType(digests), outlier_sources=frozenset(outliers),
                       _certain=lower, _possible=upper, _mentions=MappingProxyType(mentions), _role_links=links,
                       _scoped_sources=frozenset(scoped_sources))

    def _reprogram_role_links(self, roles, min_similarity, uncertainty, compile_work_limit, transport="fixed",
                              bridge_key=None, local_bridges=False, factored_bridges=False):
        if type(local_bridges) is not bool or type(factored_bridges) is not bool:
            raise ValueError("bridge maintenance options must be boolean")
        local_bridges = local_bridges or factored_bridges
        if local_bridges and bridge_key is None:
            raise ValueError("local bridge maintenance requires a bridge key")
        if roles is None:
            if bridge_key is not None:
                raise ValueError("bridge matching requires a compiled role program")
            return self if self._role_links is None else replace(self, _role_links=None)
        if isinstance(roles, str):
            raise ValueError("role links require two distinct declared predicates")
        roles = tuple(roles)
        if (len(roles) != 2 or any(type(role) is not str or role not in self._certain._predicates for role in roles)
                or roles[0] == roles[1]):
            raise ValueError("role links require two distinct declared predicates")
        if bridge_key is not None and (type(bridge_key) is not str or bridge_key in roles
                                       or bridge_key not in self._certain._predicates):
            raise ValueError("bridge matching requires a separate declared key predicate")
        floor, uncertainty = float(min_similarity), float(uncertainty)
        if not 0. < floor <= 1. or not 0. <= uncertainty <= min(floor, 1. - floor):
            raise ValueError("invalid role-link similarity threshold or uncertainty band")
        if type(transport) is not str or transport not in ("fixed", "reflection", "rotation"):
            raise ValueError("role-link transport must be fixed, reflection or rotation")
        limit = _proof_work_limit(compile_work_limit)
        links = self._role_links
        if links is not None and links.validity is not None and links.roles != roles:
            raise ValueError("installed role validity belongs to a different role grammar")
        if links is not None and (
            links.roles, links.floor, links.uncertainty, links.compile_limit, links.transport,
            links.bridge_key, links.local_bridges, links.factored_bridges
        ) == (
            roles, floor, uncertainty, limit, transport, bridge_key, local_bridges, factored_bridges,
        ):
            if not links.bridge_pending:
                return self
            if links.bridge_index is not None:
                return replace(self, _role_links=links.bridge_index.matches(links))
        compiled = _RoleLinks.prepare(
            roles, floor, uncertainty, limit, transport, self.source_digests, self._mentions, bridge_key,
            local_bridges, factored_bridges,
        )
        if links is not None and links.validity is not None:
            validity = links.validity.revise(compiled.scoped_pairs, self.source_digests, self._mentions)
            compiled = replace(compiled, validity=validity)
        return replace(self, _role_links=compiled)

    def _with_role_validity(self, transitions, as_of):
        links = self._role_links
        if links is None:
            raise ValueError("role validity needs a compiled role program")
        validity = _RoleValidity.prepare(as_of, transitions, links.scoped_pairs, self.source_digests,
                                         self._mentions, links.compile_limit, links.validity)
        return self if validity is links.validity else replace(self, _role_links=replace(links, validity=validity))

    def at_role_time(self, as_of):
        """Read the same source and authority snapshot at another effective time."""
        links = self._role_links
        if links is None or links.validity is None:
            raise ValueError("role time needs installed supersessions")
        return self._with_role_validity(links.validity.transitions, as_of)

    def read_role_links(self, direction, *, outcomes, work_limit, hops=1, condition=None):
        """Apply installed directions and then compare against declared terminal predicates."""
        return self._read_role_links(direction, outcomes=outcomes, work_limit=work_limit,
                                     hops=hops, condition=condition)

    def _read_role_links(self, direction, *, outcomes, work_limit, hops, condition,
                         prior_steps=(), prior_strength=1., branch_limit=None, retain_dependencies=False):
        hops = _role_hops(hops)
        if condition is not None and type(condition) is not BindingLinkCondition:
            raise TypeError("role-link condition must be a BindingLinkCondition")
        links = self._role_links
        if links is None:
            return BindingLinkRead("no_role_program", hops=hops, condition=condition)
        labels = self._role_outcomes(outcomes)
        read = links.read(direction, outcomes=frozenset(labels), mentions=self._mentions,
                          limit=_proof_work_limit(work_limit), width=self._anchors.shape[1], hops=hops,
                          condition=condition, prior_steps=prior_steps, prior_strength=prior_strength,
                          branch_limit=branch_limit, scoped_sources=self._scoped_sources,
                          retain_dependencies=retain_dependencies)
        if not self._scoped_sources and not retain_dependencies:
            return read
        return replace(read, witnesses=tuple(self._witness_with_spans(witness) for witness in read.witnesses),
                       dependency_witnesses=tuple(self._witness_with_spans(witness)
                                                  for witness in read.dependency_witnesses))

    def _role_outcomes(self, outcomes):
        if isinstance(outcomes, str):
            raise ValueError("role-link outcomes require distinct nonstructural predicates")
        labels = tuple(outcomes)
        if (not labels or any(type(label) is not str or label not in self._certain._predicates
                              or self._role_links.structural(label) for label in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError("role-link outcomes require distinct nonstructural predicates")
        return labels

    def _witness_with_spans(self, witness):
        final = BindingLinkStep(witness.link_source, witness.entry_ordinal, witness.exit_ordinal,
                                witness.exit_source, witness.key_ordinals, witness.publication_sha256)
        positions = []
        authorities = {}
        validity = self._role_links.validity
        for step in (*witness.prefix, final):
            target = step.exit_source if step.exit_source is not None else step.source
            positions.extend(((step.source, step.entry_ordinal), (target, step.exit_ordinal)))
            if step.key_ordinals is not None:
                entry_key, exit_key = step.key_ordinals
                positions.extend(((step.source, entry_key), (target, exit_key)))
            if validity is not None:
                for span in validity.authorities.get((step.source, step.entry_ordinal), ()):
                    authorities[span] = None
        positions.append((witness.terminal_source, witness.terminal_ordinal))
        spans, unscoped = {}, {}
        for source, ordinal in positions:
            span = self.source_mention_span(source, ordinal)
            if span is None:
                unscoped[source] = None
            else:
                spans[span] = None
        return replace(witness, source_spans=tuple(spans), validity_spans=tuple(authorities),
                       unscoped_sources=tuple(unscoped))

    def source_mentions(self, source):
        """Return a source's ordered normalized observations, or None for a legacy row-only source."""
        if source not in self.source_digests:
            raise KeyError(source)
        mentions = self._mentions.get(source)
        if mentions is None:
            return None
        return tuple(BindingMention(predicate, np.frombuffer(vector, dtype="<f8"), strength,
                                    None if scope is None else scope[:2])
                     for predicate, strength, vector, _, scope in mentions)

    def source_mention_span(self, source, ordinal):
        """Return the exact verified-source pointer for one retained mention, when scoped."""
        digest = self.source_digests[source]
        mentions = self._mentions.get(source)
        if mentions is None:
            return None
        if type(ordinal) is not int or not 0 <= ordinal < len(mentions):
            raise IndexError("mention ordinal is outside the source")
        scope = mentions[ordinal][4]
        return None if scope is None else SourceSpan(source, digest, *scope)

    def _observation_rows(self, max_mentions):
        limit = _proof_work_limit(max_mentions)
        if len(self._mentions) != len(self.source_digests):
            raise _SourceObservationRefusal(
                "source_observations_unavailable", "continuous observations are unavailable for legacy sources",
            )
        count = sum(len(mentions) for mentions in self._mentions.values())
        if count > limit:
            raise _SourceObservationRefusal(
                "source_observation_limit", "continuous observation limit exceeded",
            )
        rows = np.empty((count, self._anchors.shape[1]), dtype=np.float64)
        position = 0
        for source in sorted(self._mentions):
            for _, _, vector, _, _ in self._mentions[source]:
                rows[position] = np.frombuffer(vector, dtype="<f8")
                position += 1
        return rows

    def resolve(self, node, binding, *, without_sources=(), work_limit=65536):
        """Read both modeled bounds, with a work allowance for each bank."""
        excluded = _exclusions(without_sources)
        lower = self._certain.resolve(node, binding, without_sources=excluded, work_limit=work_limit)
        upper = lower if self._certain is self._possible else self._possible.resolve(
            node, binding, without_sources=excluded, work_limit=work_limit,
        )
        return CorpusBindingRead(lower, None if upper is None else upper.strength)

    def resolve_disjoint(self, addresses, *, without_sources=(), work_limit=65536):
        """Bound one joint proof across addresses, with one work allowance per bank."""
        work_limit = _proof_work_limit(work_limit)
        roots = self._certain._joint_keys(addresses, work_limit)
        excluded = _exclusions(without_sources, limit=work_limit)
        return self._disjoint_bounds(roots, excluded, work_limit, work_limit)

    def _disjoint_bounds(self, roots, excluded, certain_work, possible_work):
        lower = self._certain._resolve_disjoint(roots, excluded, certain_work)
        upper = lower if self._certain is self._possible else self._possible._resolve_disjoint(
            roots, excluded, possible_work,
        )
        return CorpusBindingRead(lower, None if upper is None else upper.strength)

    def reprogram(self, gates):
        """Evaluate different wiring against the same encoded source rows."""
        gates = dict(gates)
        lower = self._certain.reprogram(gates)
        if lower is self._certain:
            return self
        upper = lower if self._certain is self._possible else self._possible.reprogram(dict(lower._gates))
        return replace(self, _certain=lower, _possible=upper)

    def remember(self, node, binding, *, without_sources=(), work_limit=65536):
        """Retain the completed bounds on a new snapshot."""
        excluded = _exclusions(without_sources)
        lower = self._certain.remember(node, binding, without_sources=excluded, work_limit=work_limit)
        upper = lower if self._certain is self._possible else self._possible.remember(
            node, binding, without_sources=excluded, work_limit=work_limit,
        )
        if lower is self._certain and upper is self._possible:
            return self
        return replace(self, _certain=lower, _possible=upper)


class _DisjointBranchReads:
    """One decision's alternative extensions of a nonempty prefix on pinned banks."""

    def __init__(self, memory, prefix, work_limit, *, reuse_matching_failures=False, reuse_prefix_matching=False,
                 reuse_weighted_matching=False):
        self._memory, self._prefix = memory, tuple(prefix)
        reuse_matching = reuse_matching_failures or reuse_prefix_matching or reuse_weighted_matching
        certain_scope = (memory._certain, self._prefix) if reuse_matching else None
        self._certain = _ProofWork(work_limit, reuse_rows=True, matching_scope=certain_scope,
                                   reuse_prefix_matching=reuse_prefix_matching,
                                   reuse_weighted_matching=reuse_weighted_matching)
        possible_scope = (memory._possible, self._prefix) if reuse_matching else None
        self._possible = (self._certain if memory._certain is memory._possible
                          else _ProofWork(work_limit, reuse_rows=True, matching_scope=possible_scope,
                                          reuse_prefix_matching=reuse_prefix_matching,
                                          reuse_weighted_matching=reuse_weighted_matching))

    def resolve(self, address):
        roots = self._memory._certain._joint_keys((*self._prefix, address), self._certain.limit)
        return self._memory._disjoint_bounds(roots, frozenset(), self._certain, self._possible)

    @property
    def proof_work(self):
        if self._certain is self._possible:
            return (self._certain.used,)
        return self._certain.used, self._possible.used


class StaleCorpusRevision(ValueError):
    """A proposal no longer has the dependencies it was prepared against."""


_ReceiverProgram = tuple[str, str, int | str | None, tuple]


@dataclass(frozen=True, eq=False)
class CorpusRevision:
    revision: int
    memory: CorpusBindingMemory
    _source_revisions: Mapping[str, int]
    _owner: object
    _program_revision: int = 0
    _source_program_revision: int = 0
    _receiver_programs: tuple[_ReceiverProgram, ...] = ()
    source_revision: int = 0
    changed_sources: tuple[str, ...] = ()
    journal_sha256: str | None = None


@dataclass(frozen=True, eq=False)
class CorpusSourceProposal:
    base: CorpusRevision
    _dependencies: frozenset[str]
    _staged: tuple[_EncodedChange, ...]
    _require_tip: bool
    _allow_program_rebase: bool


@dataclass(frozen=True, eq=False)
class CorpusProgramProposal:
    base: CorpusRevision
    _preview: CorpusBindingMemory
    _prepared_on: CorpusRevision | None = None

    @property
    def preview(self):
        """Tentative memory for testing consequences before installation."""
        return self._preview


class CorpusBindingWorkspace:
    """Publish source-local edits and tip-dependent programs and proofs."""

    def __init__(self, memory):
        if type(memory) is not CorpusBindingMemory:
            raise ValueError("expected a corpus binding snapshot")
        self._lock = Lock()
        self._owner = object()
        self._tip = CorpusRevision(0, memory, MappingProxyType({}), self._owner)

    def snapshot(self):
        """Pin the current revision for an independent read or preparation."""
        with self._lock:
            return self._tip

    def _check_owner(self, base):
        if type(base) is not CorpusRevision or base._owner is not self._owner:
            raise ValueError("corpus revision belongs to another workspace")

    def prepare_changes(self, base, changes, *, depends_on=(), require_tip=False, allow_program_rebase=False):
        """Encode from a pin. Enable program rebase only for ingress independent of the graph."""
        self._check_owner(base)
        if type(require_tip) is not bool:
            raise ValueError("require_tip must be a boolean")
        if type(allow_program_rebase) is not bool or (require_tip and allow_program_rebase):
            raise ValueError("program rebase must be boolean and cannot require the whole tip")
        if require_tip and self.snapshot() is not base:
            raise StaleCorpusRevision("corpus workspace tip changed")
        if isinstance(depends_on, str):
            raise ValueError("depends_on must contain source IDs")
        changes = dict(changes)
        dependencies = frozenset(depends_on) | frozenset(changes)
        if any(type(source) is not str or not source for source in dependencies):
            raise ValueError("dependencies must be nonempty source IDs")
        staged = base.memory._prepare_changes(changes)
        return CorpusSourceProposal(base, dependencies, staged, require_tip, allow_program_rebase)

    def publish(self, proposal):
        """Check the dependencies and apply the prepared sources to the current tip."""
        self._check_proposal(proposal)
        with self._lock:
            self._check_dependencies(self._tip, proposal)
            self._tip = self._apply_proposal(self._tip, proposal._staged)
            return self._tip

    def _check_proposal(self, proposal):
        if type(proposal) is not CorpusSourceProposal:
            raise ValueError("expected a corpus source proposal")
        self._check_owner(proposal.base)

    def _check_dependencies(self, tip, proposal):
        if proposal._require_tip and tip is not proposal.base:
            raise StaleCorpusRevision("corpus workspace tip changed")
        if not proposal._allow_program_rebase and tip._source_program_revision != proposal.base._source_program_revision:
            raise StaleCorpusRevision("corpus binding program changed")
        if any(tip._source_revisions.get(source, 0) != proposal.base._source_revisions.get(source, 0)
               for source in proposal._dependencies):
            raise StaleCorpusRevision("corpus source dependency changed")

    def _apply_proposal(self, tip, staged):
        if not staged:
            return tip
        memory = tip.memory._apply_changes(staged)
        revision = tip.revision + 1
        source_revisions = dict(tip._source_revisions)
        source_revisions.update((edit.source, revision) for edit in staged)
        return replace(tip, revision=revision, memory=memory, _source_revisions=MappingProxyType(source_revisions),
                       source_revision=revision, changed_sources=tuple(sorted(edit.source for edit in staged)))

    def prepare_program(self, base, gates):
        """Compile alternate wiring for independent checks on a pinned memory."""
        self._check_owner(base)
        return CorpusProgramProposal(base, base.memory.reprogram(gates), base)

    def prepare_role_links(self, base, roles, *, min_similarity=None, uncertainty=0., compile_work_limit=None,
                           transport="fixed", bridge_key=None, local_bridges=False, factored_bridges=False):
        """Compile declared roles and optionally match fragments across sources on this pin."""
        self._check_owner(base)
        memory = base.memory._reprogram_role_links(roles, min_similarity, uncertainty, compile_work_limit, transport,
                                                   bridge_key, local_bridges, factored_bridges)
        return CorpusProgramProposal(base, memory, base)

    def prepare_role_validity(self, base, transitions, *, as_of, verifier=None):
        """Prepare a horizon and explicit role replacements on one pinned source snapshot."""
        self._check_owner(base)
        memory = base.memory._with_role_validity(transitions, as_of)
        prior, validity = base.memory._role_links.validity, memory._role_links.validity
        known = frozenset() if prior is None else frozenset(prior.transitions)
        new_rules = () if validity is None else tuple(rule for rule in validity.transitions if rule not in known)
        if new_rules:
            if type(verifier) is not VerifiedSourceTool:
                raise ValueError("new role supersessions need a source verifier")
            for pointer in dict.fromkeys(span for rule in new_rules
                                         for span in (rule.previous, rule.replacement, rule.authority)):
                if verifier.verify(pointer)["status"] != "VERIFIED_SOURCE_BYTES":
                    raise ValueError("role supersession evidence span did not verify")
        return CorpusProgramProposal(base, memory, base)

    def publish_program(self, proposal):
        """Install a checked preview only if it still represents the whole tip."""
        self._check_program(proposal)
        with self._lock:
            self._tip = self._apply_program(self._tip, proposal)
            return self._tip

    def _check_program(self, proposal):
        if type(proposal) is not CorpusProgramProposal:
            raise ValueError("expected a corpus program proposal")
        self._check_owner(proposal.base)

    def _apply_program(self, tip, proposal):
        if tip is not proposal.base:
            raise StaleCorpusRevision("corpus workspace tip changed")
        if (type(proposal.preview) is not CorpusBindingMemory
                or proposal.preview.source_digests is not tip.memory.source_digests
                or proposal.preview._mentions is not tip.memory._mentions):
            raise StaleCorpusRevision("program preview belongs to a different source snapshot")
        if proposal._prepared_on is not tip:
            raise StaleCorpusRevision("program preview was not prepared on the current revision")
        return self._replace_program(tip, proposal.preview)

    def _replace_program(self, tip, memory, *, updates_source_program=True):
        if memory is tip.memory:
            return tip
        revision = tip.revision + 1
        return replace(tip, revision=revision, memory=memory, _program_revision=revision,
                       _source_program_revision=revision if updates_source_program else tip._source_program_revision)

    def remember(self, base, node, binding, *, without_sources=(), work_limit=65536):
        """Retain a completed read only if its pinned revision is still current."""
        self._check_owner(base)
        if self.snapshot() is not base:
            raise StaleCorpusRevision("corpus workspace tip changed")
        memory = base.memory.remember(node, binding, without_sources=without_sources, work_limit=work_limit)
        with self._lock:
            if self._tip is not base:
                raise StaleCorpusRevision("corpus workspace tip changed")
            if memory is base.memory:
                return base
            self._tip = replace(base, revision=base.revision + 1, memory=memory)
            return self._tip
