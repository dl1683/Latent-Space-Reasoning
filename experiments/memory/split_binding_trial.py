"""Fit a two-step feedback receiver and define a split-binding revision panel."""

from dataclasses import dataclass, replace
from math import isfinite
from random import Random
import re
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import FeedbackKernel, ObservationMemory
from experiments.memory.weight_program import _frozen_matrix, _matrix


def _unit_rows(rows, name):
    rows = _matrix(rows, name)
    if not rows.shape[1]:
        raise ValueError(f"{name} requires a positive width")
    peak = np.max(np.abs(rows), axis=1)
    if np.any(peak == 0):
        raise ValueError(f"{name} contains a zero direction")
    scaled = rows / peak[:, None]
    return scaled / np.linalg.norm(scaled, axis=1)[:, None]


@dataclass(frozen=True)
class ReadLimits:
    min_logit: float
    min_route_gap: float
    min_answer_score: float
    min_answer_gap: float
    min_source_mass: float = .5

    def __post_init__(self):
        for value in (self.min_logit, self.min_route_gap, self.min_answer_score, self.min_answer_gap):
            if not isfinite(value) or value <= 0:
                raise ValueError("read floors and gaps must be positive and finite")
        if not 0 < self.min_source_mass <= 1:
            raise ValueError("source mass floor must be in (0, 1]")


@dataclass(frozen=True)
class TextSource:
    slot: str
    text: str


@dataclass(frozen=True)
class AnswerChoice:
    label: str
    text: str


@dataclass(frozen=True, eq=False)
class LearnedTextMemory:
    _receiver: object
    _encode: object
    _entries: MappingProxyType
    _memory: ObservationMemory

    @property
    def sources(self):
        return tuple(entry[0] for entry in self._entries.values())


@dataclass(frozen=True)
class RouteRead:
    logit: float
    gap: float | None
    source_mass: float
    observation_mass: float


@dataclass(frozen=True, eq=False)
class SplitBindingRead:
    answer: str | None
    status: str
    history: tuple[np.ndarray, ...]
    routes: tuple[RouteRead, ...]
    answer_gap: float | None


@dataclass(frozen=True, eq=False)
class LearnedSplitBindingReceiver:
    """Fit one-hop maps, then compose by feeding back the first generated state."""

    kernel: FeedbackKernel
    width: int
    limits: ReadLimits
    state_scale: float

    @classmethod
    def fit(cls, observations, starts, successors, stages, *, ridge, query_gain, state_scale, limits):
        observations = _unit_rows(observations, "training source activations")
        starts = _unit_rows(starts, "training starting activations")
        successors = _unit_rows(successors, "training successor activations")
        if observations.shape != starts.shape or observations.shape != successors.shape or not len(starts):
            raise ValueError("one-hop fit requires aligned nonempty sources, starts and successors")
        stages = np.asarray(stages)
        if stages.shape != (len(starts),) or stages.dtype.kind not in "iu" or set(stages) != {0, 1}:
            raise ValueError("one-hop fit requires aligned examples of both relation types")
        if not isfinite(ridge) or ridge <= 0 or not isfinite(query_gain) or query_gain <= 0:
            raise ValueError("ridge and query gain must be positive and finite")
        if not 0 < state_scale < 1 or type(limits) is not ReadLimits:
            raise ValueError("state scale must be inside the tanh range and limits must be fixed")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            left, singular, right = np.linalg.svd(observations, full_matrices=False)
            damped_inverse = (right.T * (singular / (singular**2 + ridge))) @ left.T
            stage_keys = np.eye(2)[stages]
            next_phase = np.column_stack((np.zeros(len(starts)), stage_keys[:, 0]))
            keys = damped_inverse @ np.column_stack((starts, stage_keys))
            values = damped_inverse @ np.arctanh(state_scale * np.column_stack((successors, next_phase)))
        width = observations.shape[1]
        operator_width = width + 2
        zeros = np.zeros((operator_width, operator_width))
        kernel = FeedbackKernel(query=query_gain * np.eye(operator_width), carry=None,
                                observation_key=np.pad(keys, ((0, 2), (0, 0))),
                                observation_value=np.pad(values, ((0, 2), (0, 0))),
                                feedback_key=zeros, feedback_value=zeros, max_steps=2)
        return cls(kernel, width, limits, state_scale)

    def _snapshot(self, entries, encode):
        arrays = np.array([entry[1] for entry in entries.values()]).reshape((-1, self.width))
        rows = np.pad(arrays, ((0, 0), (0, 2)))
        return LearnedTextMemory(self, encode, MappingProxyType(entries), self.kernel.prepare(rows))

    def prepare(self, sources, *, encode):
        if not callable(encode):
            raise ValueError("prepare requires an encoder callable")
        sources = tuple(sources)
        if len({source.slot for source in sources}) != len(sources):
            raise ValueError("source slots must be distinct")
        return self.revise(self._snapshot({}, encode), {source.slot: source.text for source in sources})

    def revise(self, memory, changes):
        """Encode changed text and keep the earlier snapshot for historical reads."""
        if memory._receiver is not self:
            raise ValueError("memory belongs to a different receiver")
        entries = dict(memory._entries)
        pending = []
        for slot, text in changes.items():
            if type(slot) is not str or not slot or (text is not None and (type(text) is not str or not text)):
                raise ValueError("source changes require named slots and nonempty text or deletion")
            if text is None:
                entries.pop(slot, None)
            elif slot not in entries or entries[slot][0].text != text:
                pending.append(TextSource(slot, text))
        if pending:
            rows = _unit_rows(memory._encode(tuple(source.text for source in pending)), "encoded sources")
            if rows.shape != (len(pending), self.width):
                raise ValueError("encoder output does not match the changed source slots")
            for source, row in zip(pending, rows, strict=True):
                entries[source.slot] = (source, _frozen_matrix(row))
        if not pending and len(entries) == len(memory._entries):
            return memory
        return self._snapshot(entries, memory._encode)

    def candidate_value(self, memory, weight):
        """Return a proposed operator on the same observation pin for guarded replay."""
        if memory._receiver is not self:
            raise ValueError("memory belongs to a different receiver")
        receiver = replace(self, kernel=self.kernel._with_observation_value(weight))
        return receiver, replace(memory, _receiver=receiver)

    def read(self, seed, candidates, labels, *, memory=None, replace_first=None):
        return self._read(seed, candidates, labels, memory=memory, replace_first=replace_first, forced=False)

    def generated_continuation(self, seed, candidates, labels, *, memory):
        """Diagnose the second hop while bypassing only the first route's refusal."""
        return self._read(seed, candidates, labels, memory=memory, replace_first=None, forced=True)

    def teacher_forced_read(self, seed, intermediate, candidates, labels, *, memory):
        """Evaluate hop two with an externally supplied relay, even if hop one would refuse."""
        entity = _unit_rows(np.asarray(intermediate)[None, :], "teacher relay activation")
        if entity.shape[1] != self.width:
            raise ValueError("teacher relay activation must match the receiver width")
        replacement = self.state_scale * np.r_[entity[0], 0., 1.]
        return self._read(seed, candidates, labels, memory=memory, replace_first=replacement, forced=True)

    def _read(self, seed, candidates, labels, *, memory, replace_first, forced):
        seed = _unit_rows(np.asarray(seed)[None, :], "query activation")
        candidates = _unit_rows(candidates, "answer activations")
        labels = tuple(labels)
        if seed.shape[1] != self.width or candidates.shape[1] != self.width:
            raise ValueError("query and answer activations must match the receiver width")
        if len(candidates) != len(labels) or len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("the read requires at least two distinct answer choices")
        if memory is not None and memory._receiver is not self:
            raise ValueError("memory belongs to a different receiver")
        if memory is None or not memory._entries:
            return SplitBindingRead(None, "unsupported_route", (), (), None)

        state = self.kernel.start(self.state_scale * np.r_[seed[0], 1., 0.], memory._memory)
        routes = []
        for hop in range(2):
            if hop == 0 and forced:
                state = self.kernel.step(state)
                if replace_first is not None:
                    state = self.kernel.intervene(state, 0, replace_first)
                continue
            diagnostics = {}
            next_state, _, _ = self.kernel._advance(state, diagnostics=diagnostics)
            logits = diagnostics["logits"][:len(memory._entries)]
            probabilities = diagnostics["probabilities"][:len(memory._entries)]
            best = int(np.argmax(logits))
            score, mass = float(logits[best]), float(probabilities[best])
            gap = float(score - np.partition(logits, -2)[-2]) if len(logits) > 1 else None
            routes.append(RouteRead(score, gap, mass, float(probabilities.sum())))
            if score < self.limits.min_logit:
                return SplitBindingRead(None, "unsupported_route", state.history, tuple(routes), None)
            if gap is not None and gap < self.limits.min_route_gap:
                return SplitBindingRead(None, "ambiguous_route", state.history, tuple(routes), None)
            if mass < self.limits.min_source_mass:
                return SplitBindingRead(None, "diffuse_route", state.history, tuple(routes), None)
            state = next_state
            if hop == 0 and replace_first is not None:
                state = self.kernel.intervene(state, 0, replace_first)

        scores = candidates @ state.latest[:self.width]
        order = np.argsort(scores)
        gap = float(scores[order[-1]] - scores[order[-2]])
        if scores[order[-1]] < self.limits.min_answer_score or gap < self.limits.min_answer_gap:
            return SplitBindingRead(None, "undecided_answer", state.history, tuple(routes), gap)
        return SplitBindingRead(labels[order[-1]], "answered", state.history, tuple(routes), gap)


_TEXT_FORMATS = (
    ("The relay for {} is {}.", "The route served by {} reaches {}."),
    ("Reports name {} as the origin and {} as its relay.", "Traffic from relay {} arrives at {}."),
)
_TEXT_PATTERNS = tuple(
    (stage, re.compile(re.escape(template).replace(r"\{\}", r"(E[0-9A-F]+)")))
    for wording in _TEXT_FORMATS
    for stage, template in enumerate(wording)
)


def _parse_source(text):
    for stage, pattern in _TEXT_PATTERNS:
        if found := pattern.fullmatch(text):
            return stage, *found.groups()
    raise ValueError("source falls outside the indexed comparator grammar")


@dataclass(frozen=True, eq=False)
class IndexedSplitBinding:
    """Exact comparator updating postings from the same text changes."""

    _source_rows: MappingProxyType
    _postings: MappingProxyType

    @classmethod
    def prepare(cls, sources):
        sources = tuple(sources)
        if len({source.slot for source in sources}) != len(sources):
            raise ValueError("source slots must be distinct")
        return cls._empty().revise({source.slot: source.text for source in sources})

    @classmethod
    def _empty(cls):
        return cls(MappingProxyType({}), MappingProxyType({}))

    def revise(self, changes):
        return self._revise_edges({slot: None if text is None else _parse_source(text)
                                   for slot, text in changes.items()})

    def _revise_edges(self, changes):
        rows = dict(self._source_rows)
        postings = dict(self._postings)
        for slot, current in changes.items():
            previous = rows.pop(slot, None)
            for direction, edge in ((-1, previous), (1, current)):
                if edge is None:
                    continue
                stage, start, end = edge
                key = (stage, start)
                values = dict(postings.get(key, {}))
                amount = values.get(end, 0) + direction
                if amount:
                    values[end] = amount
                else:
                    values.pop(end, None)
                if values:
                    postings[key] = MappingProxyType(values)
                else:
                    postings.pop(key, None)
            if current is not None:
                rows[slot] = current
        return IndexedSplitBinding(MappingProxyType(rows), MappingProxyType(postings))

    def relay(self, origin):
        first = self._postings.get((0, origin), {})
        return next(iter(first)) if len(first) == 1 else None

    def read(self, origin):
        middle = self.relay(origin)
        second = self._postings.get((1, middle), {})
        return next(iter(second)) if len(second) == 1 else None


_SPLITS = (
    "receiver-fit", "native-competence", "calibration-fit", "calibration-validation",
    "source-repair", "edit-guard", "destination-revisions", "software-control", "native-development",
)


def entity_prompt(label):
    return f"Entity {label}"


@dataclass(frozen=True)
class RevisionPlan:
    label: str
    changes: tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class SplitBindingWorld:
    split: str
    number: int
    sources: tuple[TextSource, ...]
    origins: tuple[str, str]
    choices: tuple[AnswerChoice, ...]
    entity_catalog: tuple[AnswerChoice, ...]
    training_edges: tuple[tuple[int, str, str], ...]
    revisions: tuple[RevisionPlan, ...]
    expected: tuple[tuple[str | None, str | None], ...]


def build_split_binding_world(split, number):
    """Build text sources and a separate truth record for a trial caller."""
    if split not in _SPLITS or type(number) is not int or not 0 <= number < 65536:
        raise ValueError("unknown split or world number outside its reserved range")
    namespace = (1 + _SPLITS.index(split)) << 24 | number << 8
    random = Random(namespace ^ 0x6DE829)
    a, b, shared, other, stale, updated, unrelated, decoy = (
        f"E{namespace | suffix:07X}" for suffix in random.sample(range(256), 8)
    )
    wording = _TEXT_FORMATS[split != "receiver-fit"]
    slots = [f"S{value:04X}" for value in random.sample(range(65536), 6)]
    edges = ((0, a, shared), (1, shared, stale), (0, b, other), (1, other, unrelated),
             (1, decoy, shared), (1, a, decoy))
    facts = [(TextSource(slot, wording[stage].format(start, end)), (stage, start, end))
             for slot, (stage, start, end) in zip(slots, edges, strict=True)]
    relay = facts[0][0].text
    corrected = wording[1].format(shared, updated)
    random.shuffle(facts)
    choices = [AnswerChoice(label, entity_prompt(label)) for label in (stale, updated, unrelated, decoy, shared, other)]
    random.shuffle(choices)
    catalog = choices + [AnswerChoice(label, entity_prompt(label)) for label in (a, b)]
    random.shuffle(catalog)
    sources, edges = zip(*facts, strict=True)
    revisions = (
        RevisionPlan("correct", ((slots[1], corrected),)),
        RevisionPlan("remove-first", ((slots[0], None),)),
        RevisionPlan("remove-second", ((slots[1], None),)),
        RevisionPlan("restore", ((slots[0], relay), (slots[1], corrected))),
    )
    expected = ((stale, unrelated), (updated, unrelated), (None, unrelated), (None, unrelated),
                (updated, unrelated), (updated, unrelated))
    training = edges if split == "receiver-fit" else ()
    return SplitBindingWorld(split, number, sources, (a, b), tuple(choices), tuple(catalog),
                             training, revisions, expected)


def fit_text_receiver(worlds, *, encode, ridge, query_gain, state_scale, limits):
    worlds = tuple(worlds)
    if not worlds or any(world.split != "receiver-fit" for world in worlds):
        raise ValueError("only the receiver-fit split can supervise the native receiver")
    triples = [(record.text, *edge) for world in worlds
               for record, edge in zip(world.sources, world.training_edges, strict=True)]
    records, phases, starts, ends = zip(*triples, strict=True)
    return LearnedSplitBindingReceiver.fit(
        encode(records), encode(tuple(map(entity_prompt, starts))), encode(tuple(map(entity_prompt, ends))),
        phases, ridge=ridge, query_gain=query_gain, state_scale=state_scale, limits=limits,
    )


def revision_snapshots(opening, world, *, revise):
    """Both premise-removal branches start from the corrected pin."""
    correction, remove_first, remove_second, restore = world.revisions
    corrected = revise(opening, dict(correction.changes))
    without_first = revise(corrected, dict(remove_first.changes))
    without_second = revise(corrected, dict(remove_second.changes))
    restored_first = revise(without_first, dict(restore.changes))
    restored_second = revise(without_second, dict(restore.changes))
    return opening, corrected, without_first, without_second, restored_first, restored_second


@dataclass(frozen=True, eq=False)
class NativeWorldReplay:
    reads: tuple[tuple[SplitBindingRead, SplitBindingRead], ...]
    indexed: tuple[tuple[str | None, str | None], ...]
    historical: tuple[SplitBindingRead, SplitBindingRead]
    indexed_historical: tuple[str | None, str | None]


def replay_native_world(receiver, world, *, encode):
    """Prepare before querying and replay both arms without using the truth record."""
    opening = receiver.prepare(world.sources, encode=encode)
    indexed_opening = IndexedSplitBinding.prepare(world.sources)
    seeds, candidates, labels = _world_questions(receiver, world, encode)
    replay, _, _ = _replay_world(receiver, world, opening, indexed_opening, seeds, candidates, labels)
    return replay


def _world_questions(receiver, world, encode):
    labels = tuple(choice.label for choice in world.choices)
    queries = tuple(map(entity_prompt, world.origins))
    answer_texts = tuple(choice.text for choice in world.choices)
    encoded = _matrix(encode(queries + answer_texts), "encoded queries and choices")
    if encoded.shape != (len(queries) + len(labels), receiver.width):
        raise ValueError("encoder output does not match the queries and answer choices")
    return encoded[:len(queries)], encoded[len(queries):], labels


def _replay_world(receiver, world, opening, indexed_opening, seeds, candidates, labels):
    def read(bank):
        return tuple(receiver.read(seed, candidates, labels, memory=bank) for seed in seeds)

    def exact(index):
        return tuple(index.read(origin) for origin in world.origins)

    first_read, first_exact = read(opening), exact(indexed_opening)
    snapshots = revision_snapshots(opening, world, revise=receiver.revise)
    indexes = revision_snapshots(indexed_opening, world, revise=IndexedSplitBinding.revise)
    replay = NativeWorldReplay((first_read, *map(read, snapshots[1:])),
                               (first_exact, *map(exact, indexes[1:])),
                               read(opening), exact(indexed_opening))
    return replay, snapshots, indexes


@dataclass(frozen=True)
class RoleDecodeLimits:
    min_phase_gap: float
    min_entity_score: float
    min_entity_gap: float

    def __post_init__(self):
        values = (self.min_phase_gap, self.min_entity_score, self.min_entity_gap)
        if not all(isfinite(value) and value > 0 for value in values):
            raise ValueError("role decoding requires positive, finite score and gap floors")


@dataclass(frozen=True)
class SourceRoleRead:
    phase: int | None
    start: str | None
    successor: str | None

    @property
    def edge(self):
        if None in (self.phase, self.start, self.successor):
            return None
        return self.phase, self.start, self.successor


@dataclass(frozen=True)
class SourceRoleCheck:
    slot: str
    predicted: SourceRoleRead
    exact: tuple[int, str, str]


@dataclass(frozen=True)
class WorldAnswerCounts:
    correct_supported: int
    wrong_supported: int
    missed_supported: int
    false_supported: int
    abstained_unsupported: int
    ineligible: int = 0


def count_world_answers(expected, predicted):
    expected = tuple(tuple(pair) for pair in expected)
    predicted = tuple(tuple(pair) for pair in predicted)
    if not expected or len(expected) != len(predicted) or any(len(a) != 2 or len(b) != 2
                                                               for a, b in zip(expected, predicted)):
        raise ValueError("prediction and truth must cover the same fixed pairs of world reads")
    return _count_answers((target, answer) for reference, answers in zip(expected, predicted, strict=True)
                          for target, answer in zip(reference, answers, strict=True))


def _count_answers(pairs, *, ineligible=0):
    correct = wrong = missed = false_supported = abstained = 0
    for target, answer in pairs:
        if target is None:
            if answer is None:
                abstained += 1
            else:
                false_supported += 1
        elif answer is None:
            missed += 1
        elif target == answer:
            correct += 1
        else:
            wrong += 1
    return WorldAnswerCounts(correct, wrong, missed, false_supported, abstained, ineligible)


@dataclass(frozen=True, eq=False)
class CompetenceWorldReplay:
    native: NativeWorldReplay
    decoded_index: tuple[tuple[str | None, str | None], ...]
    generated_continuation: tuple[tuple[SplitBindingRead, SplitBindingRead], ...]
    teacher_forced: tuple[tuple[SplitBindingRead | None, SplitBindingRead | None], ...]
    source_roles: tuple[SourceRoleCheck, ...]


@dataclass(frozen=True)
class CompetenceWorldScores:
    native: WorldAnswerCounts
    decoded_index: WorldAnswerCounts
    generated_continuation: WorldAnswerCounts
    teacher_forced: WorldAnswerCounts


def score_competence_world(world, replay):
    expected = world.expected
    if len(expected) != 6 or any(len(pair) != 2 for pair in expected):
        raise ValueError("competence scoring requires the six fixed truth pairs")
    reference = (*expected, expected[0])
    native = (*replay.native.reads, replay.native.historical)
    if len(replay.teacher_forced) != len(reference) or any(len(pair) != 2 for pair in replay.teacher_forced):
        raise ValueError("teacher diagnostic must retain all scheduled cells")
    teacher_pairs = [(target, read.answer)
                     for pair, reads in zip(reference, replay.teacher_forced, strict=True)
                     for target, read in zip(pair, reads, strict=True) if read is not None]
    ineligible = 2 * len(reference) - len(teacher_pairs)
    return CompetenceWorldScores(
        count_world_answers(reference, ((read.answer for read in pair) for pair in native)),
        count_world_answers(reference, replay.decoded_index),
        count_world_answers(reference, ((read.answer for read in pair) for pair in replay.generated_continuation)),
        _count_answers(teacher_pairs, ineligible=ineligible),
    )


def _decoded_entity(scores, labels, limits):
    best = int(np.argmax(scores))
    gap = scores[best] - np.partition(scores, -2)[-2]
    if scores[best] < limits.min_entity_score or gap < limits.min_entity_gap:
        return None
    return labels[best]


def _probe_snapshot(receiver, memory, inventory, labels, limits, cache):
    keys = [(source, row.tobytes()) for source, row in memory._entries.values()]
    pending = [(position, key) for position, key in enumerate(keys) if key not in cache]
    if pending:
        positions, new_keys = zip(*pending, strict=True)
        rows = np.asarray(positions)
        width = receiver.width
        projected = memory._memory._keys[rows]
        with np.errstate(over="raise", invalid="raise"):
            values = np.tanh(memory._memory._observations[rows] @ receiver.kernel._dense_observation_value())
            start_scores = projected[:, :width] @ inventory.T
            successor_scores = (values[:, :width] / receiver.state_scale) @ inventory.T
        for key, phase, starts, ends in zip(new_keys, projected[:, width:], start_scores, successor_scores,
                                            strict=True):
            route = int(np.argmax(phase)) if abs(phase[0] - phase[1]) >= limits.min_phase_gap else None
            cache[key] = SourceRoleRead(route, _decoded_entity(starts, labels, limits),
                                       _decoded_entity(ends, labels, limits))
    inferred = {key[0].slot: cache[key].edge for key in keys}
    return IndexedSplitBinding._empty()._revise_edges(inferred)


def replay_competence_world(receiver, world, *, encode, role_limits, catalog_extras=()):
    """Compare native feedback, inferred-role indexing and a teacher-forced diagnostic."""
    if type(role_limits) is not RoleDecodeLimits:
        raise ValueError("provide independently fixed role decoding limits")
    opening = receiver.prepare(world.sources, encode=encode)
    exact_opening = IndexedSplitBinding.prepare(world.sources)
    catalog = (*world.entity_catalog, *catalog_extras)
    inventory_labels = tuple(item.label for item in catalog)
    if len(inventory_labels) < 2 or len(set(inventory_labels)) != len(inventory_labels):
        raise ValueError("role probing requires a unique advertised entity inventory")
    inventory = _unit_rows(encode(tuple(item.text for item in catalog)), "role probe inventory")
    if inventory.shape != (len(catalog), receiver.width):
        raise ValueError("encoder output does not match the advertised inventory")
    entities = dict(zip(inventory_labels, inventory, strict=True))
    cached_roles = {}
    root_index = _probe_snapshot(receiver, opening, inventory, inventory_labels, role_limits, cached_roles)
    origins = world.origins
    labels = tuple(choice.label for choice in world.choices)
    if not all(label in entities for label in (*origins, *labels)):
        raise ValueError("queries and answer choices must be present in the advertised inventory")
    seeds = np.stack([entities[label] for label in origins])
    candidates = np.stack([entities[label] for label in labels])
    native, snapshots, exact_indexes = _replay_world(
        receiver, world, opening, exact_opening, seeds, candidates, labels,
    )
    decoded, generated, teacher = [], [], []
    for snapshot, exact in zip((*snapshots, opening), (*exact_indexes, exact_opening), strict=True):
        index = root_index if snapshot is opening else _probe_snapshot(
            receiver, snapshot, inventory, inventory_labels, role_limits, cached_roles,
        )
        outputs = tuple(index.read(origin) for origin in origins)
        decoded.append(outputs)
        generated.append(tuple(receiver.generated_continuation(seed, candidates, labels, memory=snapshot)
                               for seed in seeds))

        forced = []
        for origin, seed in zip(origins, seeds, strict=True):
            relay = exact.relay(origin)
            forced.append(receiver.teacher_forced_read(seed, entities[relay], candidates, labels, memory=snapshot)
                          if relay is not None and relay in entities else None)
        teacher.append(tuple(forced))

    checked = tuple(SourceRoleCheck(source.slot, prediction, _parse_source(source.text))
                    for (source, _), prediction in cached_roles.items())
    return CompetenceWorldReplay(native, tuple(decoded), tuple(generated), tuple(teacher), checked)
