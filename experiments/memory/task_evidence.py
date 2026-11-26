"""Bounded task-time evidence investigation over a pinned source world."""

from copy import copy
from dataclasses import asdict, dataclass, replace
import json
from math import isfinite
from operator import index
from types import MappingProxyType
from typing import Callable, Mapping

from experiments.memory.binding_feedback import BindingFeedback, BindingRoleAdvance, BindingRoleDecision
from experiments.memory.binding_links import BindingLinkCondition, BindingLinkRead
from experiments.memory.binding_memory import BindingCapacityError
from experiments.memory.binding_replay import _receiver_digest
from experiments.memory.compressed_routing_map import BM25
from experiments.memory.corpus_bindings import CorpusBindingMemory, CorpusRevision
from experiments.memory.verified_source_tool import SourceSpan, VerifiedSourceTool, sha256


@dataclass(frozen=True, slots=True)
class TaskPin:
    source_snapshot: str
    span_catalogue_sha256: str
    binding_revision: int | None
    binding_source_revision: int | None
    binding_program_revision: int | None
    binding_journal_sha256: str | None
    receiver_sha256: str | None = None
    as_of: int | None = None


@dataclass(frozen=True, slots=True, init=False)
class TaskWorld:
    """Associate verified source text and an optional retained computation."""

    _tool: VerifiedSourceTool
    _spans: tuple[SourceSpan, ...]
    _spans_by_source: Mapping[str, tuple[SourceSpan, ...]]
    _index: BM25 | None
    _memory: CorpusBindingMemory | None
    _receiver: BindingFeedback | None
    pin: TaskPin
    setup_source_bytes: int

    def __init__(self, tool, spans, *, bindings=None, receiver=None, as_of=None):
        if type(tool) is not VerifiedSourceTool:
            raise TypeError("a verified source tool is required")
        if bindings is not None and (
            type(bindings) is not CorpusRevision or bindings.memory.source_digests != tool.versions
        ):
            raise ValueError("retained computation and source catalogue must use the same sources and versions")
        if bindings is not None and bindings.journal_sha256 is None:
            raise ValueError("a retained computation requires a journaled revision")
        if as_of is not None and type(as_of) is not int:
            raise ValueError("task effective time must be an integer")
        memory = None if bindings is None else bindings.memory
        if type(receiver) is BindingFeedback:
            receiver = copy(receiver)
            receiver._kernel = copy(receiver._kernel)
        receiver_sha = None if receiver is None else _receiver_digest(receiver)
        if receiver is not None and bindings is not None and bindings._receiver_programs:
            if receiver_sha != bindings._receiver_programs[-1][1]:
                raise ValueError("task receiver does not match the journaled program")
        validity = None if memory is None or memory._role_links is None else memory._role_links.validity
        if validity is not None:
            if as_of is not None:
                memory = memory.at_role_time(as_of)
            else:
                as_of = validity.as_of
        tool = VerifiedSourceTool(tool.root, tool.versions)
        spans = tuple(spans)
        if any(type(span) is not SourceSpan for span in spans) or len(set(spans)) != len(spans):
            raise ValueError("search catalogue requires distinct source spans")
        texts, setup_bytes = [], 0
        by_source = {}
        for span in spans:
            opened = tool.verify(span)
            if opened["status"] != "VERIFIED_SOURCE_BYTES":
                raise ValueError("search catalogue contains an unverifiable span")
            texts.append(opened["text"])
            setup_bytes += opened["source_bytes_read"]
            by_source.setdefault(span.source_id, []).append(span)
        catalogue = json.dumps([asdict(span) for span in spans], sort_keys=True, separators=(",", ":")).encode()
        pin = TaskPin(tool.snapshot_sha256, sha256(catalogue), None if bindings is None else bindings.revision,
                      None if bindings is None else bindings.source_revision,
                      None if bindings is None else bindings._program_revision,
                      None if bindings is None else bindings.journal_sha256, receiver_sha, as_of)
        object.__setattr__(self, "_tool", tool)
        object.__setattr__(self, "_spans", spans)
        object.__setattr__(self, "_spans_by_source", MappingProxyType({key: tuple(rows) for key, rows in by_source.items()}))
        object.__setattr__(self, "_index", BM25(texts) if texts else None)
        object.__setattr__(self, "_memory", memory)
        object.__setattr__(self, "_receiver", receiver)
        object.__setattr__(self, "pin", pin)
        object.__setattr__(self, "setup_source_bytes", setup_bytes)

    def _search(self, query, limit):
        if self._index is None:
            return (), 0
        scores = self._index.score(query)
        matching = [i for i, score in enumerate(scores) if score > 0]
        matching.sort(key=lambda i: (-scores[i], i))
        return tuple(self._spans[i] for i in matching[:limit]), len(matching)


@dataclass(frozen=True, slots=True)
class TaskBudget:
    steps: int = 24
    candidates_per_search: int = 8
    evidence_bytes: int = 8192
    source_bytes: int = 2 * 1024 * 1024
    role_work: int = 512
    role_branches: int = 4
    role_transitions: int = 16
    role_candidate_spans: int = 256
    bound_context: bool = False

    def __post_init__(self):
        if type(self.bound_context) is not bool:
            raise ValueError("context completion policy must be boolean")
        for name in ("steps", "candidates_per_search", "evidence_bytes", "source_bytes", "role_work",
                     "role_branches", "role_transitions", "role_candidate_spans"):
            value = index(getattr(self, name))
            if value < 1:
                raise ValueError("task allowances must be positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class EvidenceNeed:
    id: str
    question: str
    rationale: str
    parent: str | None = None
    discovered_from: tuple[int, ...] = ()
    target_or_event: str | None = None
    relation: str | None = None
    time_version_scope: str | None = None
    evidence_sought: tuple[str, ...] = ()
    support_requirements: tuple[str, ...] = ()
    defeat_requirements: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    unresolved_conditions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchEvidence:
    need: str
    query: str


@dataclass(frozen=True, slots=True)
class OpenEvidence:
    need: str
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class ProbeRoleLinks:
    need: str
    direction: tuple[float, ...] | None
    outcomes: tuple[str, ...]
    hops: int = 1
    condition: BindingLinkCondition | None = None
    source_observation: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ExploreRolePaths:
    need: str
    source_observation: tuple[int, int]
    outcomes: tuple[str, ...]
    stages: tuple[int, ...]
    terminal_hops: int = 1


@dataclass(frozen=True, slots=True)
class RoleSupport:
    receipt: int
    outcome: str


@dataclass(frozen=True, slots=True)
class EvidenceCase:
    need: str
    stance: str
    claim: str
    inference: str
    citations: tuple[int, ...]
    role_support: RoleSupport | None = None


@dataclass(frozen=True, slots=True)
class ReconcileProbe:
    need: str
    probe: int
    rationale: str
    citations: tuple[int, ...]


TaskAction = EvidenceNeed | SearchEvidence | OpenEvidence | ProbeRoleLinks | ExploreRolePaths | EvidenceCase | ReconcileProbe


@dataclass(frozen=True, slots=True)
class EvidenceSearch:
    need: str
    query: str
    candidates: tuple[SourceSpan, ...]
    matched: int


@dataclass(frozen=True, slots=True)
class EvidenceRead:
    id: int
    need: str
    span: SourceSpan
    status: str
    text: str | None
    source_bytes_read: int
    observation_predicates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoleEvidence:
    request: ProbeRoleLinks | ExploreRolePaths
    read: BindingLinkRead | BindingRoleDecision
    direction: tuple[float, ...] | None
    reconciliation: ReconcileProbe | None = None


@dataclass(frozen=True, slots=True)
class RejectedAction:
    step: int
    action: str
    reason: str


@dataclass(frozen=True, slots=True)
class NeedFinding:
    need: EvidenceNeed
    status: str
    cases: tuple[EvidenceCase, ...]
    open_children: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskDossier:
    task: str
    pin: TaskPin
    needs: tuple[NeedFinding, ...]
    searches: tuple[EvidenceSearch, ...]
    reads: tuple[EvidenceRead, ...]
    role_evidence: tuple[RoleEvidence, ...]
    rejected_actions: tuple[RejectedAction, ...]
    budget: TaskBudget
    stop_reason: str
    steps: int
    source_bytes_read: int
    returned_evidence_bytes: int
    role_work: int
    lexical_rows_scored: int
    assessment_scope: str = "PLANNER_ASSESSED"
    role_transitions: int = 0


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def _optional_text(value, label):
    if value is not None:
        _text(value, label)
    return value


def _text_items(values, label):
    if values is None:
        return ()
    if isinstance(values, str) or not isinstance(values, (tuple, list)):
        raise ValueError(f"{label} must be a list or tuple of nonempty text")
    frozen = tuple(values)
    for value in frozen:
        _text(value, label)
    return frozen


def _role_terminal_reads(read):
    return read.terminal_reads if type(read) is BindingRoleDecision else (read,)


def _read_covers_span(read, raw, pointer):
    span = read.span
    if (span.source_id != pointer.source_id or span.version_sha256 != pointer.version_sha256
            or not span.start_byte <= pointer.start_byte < pointer.end_byte <= span.end_byte):
        return False
    start = pointer.start_byte - span.start_byte
    end = pointer.end_byte - span.start_byte
    return sha256(raw[start:end]) == pointer.span_sha256


class TaskInvestigator:
    """Run a configured policy that sees task text and bounded tool results."""

    def __init__(self, advance: Callable[[TaskDossier], TaskAction | None], *, budget=TaskBudget()):
        if not callable(advance) or type(budget) is not TaskBudget:
            raise TypeError("a planner policy and task budget are required")
        self._advance, self._budget = advance, budget

    def plan(self, task, world):
        _text(task, "task")
        if type(world) is not TaskWorld:
            raise TypeError("a pinned task world is required")
        session = _Investigation(task, world, self._budget)
        for _ in range(self._budget.steps):
            action = self._advance(session.dossier("in_progress"))
            if action is None:
                return session.dossier("policy_stopped")
            try:
                session.apply(action)
            except (TypeError, ValueError, OverflowError) as error:
                session.rejected.append(RejectedAction(session.steps + 1, type(action).__name__, str(error)))
            session.steps += 1
        return session.dossier("step_limit")


class _Investigation:
    def __init__(self, task, world, budget):
        self.task, self.world, self.budget = task, world, budget
        self.needs, self.cases, self.children = {}, {}, {}
        self.searches, self.reads, self.role_evidence = [], [], []
        self.rejected = []
        self.investigated, self.exhausted, self.unavailable = set(), set(), set()
        self.steps = self.source_bytes = self.evidence_bytes = self.role_work = 0
        self.lexical_rows = self.role_transitions = 0

    def _citations(self, values):
        ids = tuple(values)
        if (any(type(i) is not int or not 0 <= i < len(self.reads)
                or self.reads[i].status != "VERIFIED_SOURCE_BYTES" for i in ids) or len(ids) != len(set(ids))):
            raise ValueError("citations must identify distinct verified reads from this investigation")
        return ids

    def _verify_role_support(self, support, stance, citations, need, open_need_ids=None,
                             has_open_question=False):
        if (type(support) is not RoleSupport or type(support.receipt) is not int
                or not 0 <= support.receipt < len(self.role_evidence)
                or type(support.outcome) is not str or not support.outcome):
            raise ValueError("role support requires an existing receipt and outcome")
        role = self.role_evidence[support.receipt]
        if role.request.need != need:
            raise ValueError("role support requires a receipt for this need")
        if type(role.request) is not ExploreRolePaths or type(role.read) is not BindingRoleDecision:
            raise ValueError("role support requires a joint role-path receipt")
        status = role.read.status
        if status == "supported":
            if stance not in ("supports", "defeats") or role.read.answer != support.outcome:
                raise ValueError("role support requires a closed receipt matching its finding and outcome")
        elif status == "conflict":
            if stance != "conflicted" or support.outcome != "conflict":
                raise ValueError("conflicting role support must be reported as conflicted")
        elif stance != "unknown" or support.outcome != status:
            raise ValueError("an open role receipt must be reported as unknown with its exact status")
        if role.request.source_observation is None or role.request.source_observation[0] not in citations:
            raise ValueError("role support must cite the verified seed observation")
        terminal_reads = _role_terminal_reads(role.read)
        role_links = None if self.world._memory is None else self.world._memory._role_links
        has_time_rules = role_links is not None and role_links.validity is not None
        if has_time_rules and any(read.as_of != self.world.pin.as_of for read in terminal_reads):
            raise ValueError("role support must match the effective-time pin")
        if not terminal_reads and stance != "unknown":
            raise ValueError("decisive and conflicting role support requires terminal reads")

        exact, whole = set(), set()
        for read in terminal_reads:
            for witness in read.dependency_witnesses:
                whole.update(witness.unscoped_sources)
                exact.update(witness.source_spans)
                exact.update(witness.validity_spans)
        if status == "conflict":
            witnessed = {witness.outcome for read in terminal_reads for witness in read.dependency_witnesses}
            if len(witnessed) < 2:
                raise ValueError("a conflicted role finding requires witnesses for distinct outcomes")
            if role.read.open_coverage:
                if open_need_ids is not None and need not in open_need_ids:
                    raise ValueError("a partial role conflict requires a same-need open question")
                if open_need_ids is None and not has_open_question:
                    raise ValueError("a partial role conflict requires an open question")
        if stance in ("supports", "defeats") and any(not read.dependency_witnesses for read in terminal_reads):
            raise ValueError("decisive role support requires dependencies for every terminal path")
        verified = tuple(self.reads[i] for i in citations)
        verified_whole = {read.span.source_id for read in verified
                          if read.span.start_byte == 0 and read.span.end_byte == read.source_bytes_read}
        texts = tuple((read, read.text.encode("utf8")) for read in verified)
        if not whole <= verified_whole or any(
            not any(_read_covers_span(read, raw, pointer) for read, raw in texts) for pointer in exact
        ):
            raise ValueError("role support citations must cover the premises, terminal and authority spans")

    def apply(self, action):
        if type(action) is EvidenceNeed:
            self._add_need(action)
        else:
            if type(action) not in (SearchEvidence, OpenEvidence, ProbeRoleLinks, ExploreRolePaths,
                                    EvidenceCase, ReconcileProbe):
                raise TypeError("invalid planner action")
            if action.need not in self.needs:
                raise ValueError("action requires an existing evidence need")
            if type(action) is SearchEvidence:
                _text(action.query, "search query")
                candidates, matched = self.world._search(action.query, self.budget.candidates_per_search)
                self.lexical_rows += len(self.world._spans)
                self.searches.append(EvidenceSearch(action.need, action.query, candidates, matched))
                self.investigated.add(action.need)
            elif type(action) is OpenEvidence:
                self._open(action)
            elif type(action) is ProbeRoleLinks:
                self._probe(action)
            elif type(action) is ExploreRolePaths:
                self._explore(action)
            elif type(action) is ReconcileProbe:
                self._reconcile(action)
            else:
                self._record(action)

    def _add_need(self, need):
        _text(need.id, "need id")
        _text(need.question, "need question")
        _text(need.rationale, "need rationale")
        if need.id in self.needs or (need.parent is not None and need.parent not in self.needs):
            raise ValueError("need must be new and its parent must exist")
        self.needs[need.id] = replace(
            need,
            discovered_from=self._citations(need.discovered_from),
            target_or_event=_optional_text(need.target_or_event, "target or event"),
            relation=_optional_text(need.relation, "relation"),
            time_version_scope=_optional_text(need.time_version_scope, "time/version scope"),
            evidence_sought=_text_items(need.evidence_sought, "evidence sought"),
            support_requirements=_text_items(need.support_requirements, "support requirements"),
            defeat_requirements=_text_items(need.defeat_requirements, "defeat requirements"),
            alternatives=_text_items(need.alternatives, "alternatives"),
            unresolved_conditions=_text_items(need.unresolved_conditions, "unresolved conditions"),
        )
        self.cases[need.id], self.children[need.id] = [], []
        if need.parent is not None:
            self.children[need.parent].append(need.id)

    def _open(self, action):
        span = action.span
        if type(span) is not SourceSpan:
            raise TypeError("source span required")
        self.investigated.add(action.need)
        if span.end_byte - span.start_byte > self.budget.evidence_bytes - self.evidence_bytes:
            self.exhausted.add(action.need)
            self.reads.append(EvidenceRead(len(self.reads), action.need, span, "EVIDENCE_LIMIT", None, 0))
            return
        opened = self.world._tool.verify(span, max_source_bytes=self.budget.source_bytes - self.source_bytes)
        self.source_bytes += opened["source_bytes_read"]
        self.evidence_bytes += opened.get("returned_utf8_bytes", 0)
        if opened["status"] == "SOURCE_LIMIT":
            self.exhausted.add(action.need)
        elif opened["status"] != "VERIFIED_SOURCE_BYTES":
            self.unavailable.add(action.need)
        observations = ()
        memory = self.world._memory
        if (memory is not None and opened["status"] == "VERIFIED_SOURCE_BYTES"
                and span.start_byte == 0 and span.end_byte == opened["source_bytes_read"]):
            observations = tuple(mention.predicate for mention in memory.source_mentions(span.source_id) or ())
        self.reads.append(EvidenceRead(len(self.reads), action.need, span, opened["status"], opened.get("text"),
                                       opened["source_bytes_read"], observations))

    def _probe(self, action):
        outcomes = tuple(action.outcomes)
        hops = index(action.hops)
        if (hops < 1 or isinstance(action.outcomes, str) or not outcomes
                or any(type(item) is not str or not item for item in outcomes)
                or len(set(outcomes)) != len(outcomes)
                or (action.condition is not None and type(action.condition) is not BindingLinkCondition)):
            raise ValueError("invalid role probe request")
        memory = self.world._memory
        origin = None
        if action.source_observation is None:
            direction = tuple(map(float, action.direction)) if action.direction is not None else ()
            if not direction or not all(map(isfinite, direction)):
                raise ValueError("role probe requires a finite direction or a verified source observation")
        else:
            if action.direction is not None:
                raise ValueError("choose one role probe input")
            origin = tuple(action.source_observation)
            direction = self._observation_direction(origin)
        request = replace(action, direction=direction if origin is None else None, outcomes=outcomes, hops=hops,
                          source_observation=origin)
        remaining = self.budget.role_work - self.role_work
        if memory is None:
            read = BindingLinkRead("no_memory", outcomes=outcomes, hops=hops, condition=action.condition)
        elif not remaining:
            read = BindingLinkRead("work_limit", outcomes=outcomes, hops=hops, condition=action.condition)
        else:
            read = memory.read_role_links(direction, outcomes=outcomes, hops=hops,
                                          work_limit=remaining, condition=action.condition,
                                          bound_context=self.budget.bound_context)
            self.role_work += read.work
        if read.status == "work_limit":
            self.exhausted.add(action.need)
        self.investigated.add(action.need)
        self.role_evidence.append(RoleEvidence(request, read, direction))

    def _observation_direction(self, origin):
        read_id, ordinal = origin
        self._citations((read_id,))
        evidence = self.reads[read_id]
        if (type(ordinal) is not int or ordinal < 0 or evidence.span.start_byte != 0
                or evidence.span.end_byte != evidence.source_bytes_read):
            raise ValueError("role source must identify an observation in a verified complete source")
        if self.world._memory is None:
            return None
        if ordinal >= len(evidence.observation_predicates):
            raise ValueError("role observation is unavailable on this pin")
        mentions = self.world._memory.source_mentions(evidence.span.source_id)
        return tuple(map(float, mentions[ordinal].vector))

    def _explore(self, action):
        origin, stages, outcomes = tuple(action.source_observation), tuple(action.stages), tuple(action.outcomes)
        if (not stages or any(type(stage) is not int or stage < 0 for stage in stages)
                or type(action.terminal_hops) is not int or action.terminal_hops < 1
                or not outcomes or isinstance(action.outcomes, str)
                or any(type(outcome) is not str or not outcome for outcome in outcomes)
                or len(outcomes) != len(set(outcomes))):
            raise ValueError("invalid joint role request")
        memory, receiver = self.world._memory, self.world._receiver
        if memory is not None and memory._role_links is not None:
            outcomes = memory._role_outcomes(outcomes)
        direction = self._observation_direction(origin)
        work = self.budget.role_work - self.role_work
        transitions = self.budget.role_transitions - self.role_transitions
        branches = min(self.budget.role_branches, transitions // len(stages))
        if memory is None:
            result = BindingRoleDecision("no_memory", (), 0, 0, True)
        elif receiver is None:
            result = BindingRoleDecision("no_receiver", (), 0, 0, True)
        elif not work or not branches:
            status = "work_limit" if not work else "transition_limit"
            result = BindingRoleDecision(status, (), 0, 0, True)
        else:
            if len(stages) > receiver._kernel._max_steps:
                raise ValueError("role stages must fit the receiver horizon")
            operations = tuple(None if stage == 0 else BindingRoleAdvance(stage) for stage in stages)
            try:
                state = receiver.start(direction, memory)
                result = receiver.reason_role_links(
                    state, stages=operations, outcomes=outcomes, work_limit=work,
                    branch_limit=branches, terminal_hops=action.terminal_hops, retain_dependencies=True,
                    bound_context=self.budget.bound_context,
                )
            except BindingCapacityError as error:
                result = BindingRoleDecision(getattr(error, "status", "source_observation_limit"), (), 0, 0, True)
            except FloatingPointError:
                result = BindingRoleDecision("numerical_failure", (), work, branches * len(stages), True)
        self.role_work += result.work
        self.role_transitions += result.transitions
        if result.status in ("work_limit", "branch_limit", "transition_limit"):
            self.exhausted.add(action.need)
        self.investigated.add(action.need)
        request = replace(action, source_observation=origin, stages=stages, outcomes=outcomes)
        self.role_evidence.append(RoleEvidence(request, result, direction))

    def _reconcile(self, action):
        _text(action.rationale, "probe reconciliation rationale")
        if type(action.probe) is not int or not 0 <= action.probe < len(self.role_evidence):
            raise ValueError("probe reconciliation requires an existing role receipt")
        role = self.role_evidence[action.probe]
        if (role.request.need != action.need or role.read.status not in ("conflict", "unresolved")
                or role.reconciliation is not None):
            raise ValueError("probe reconciliation requires an open role receipt for this need")
        if (type(role.read) is BindingRoleDecision or role.read.supersessions or role.read.validity_open
                or role.read.as_of is not None):
            raise ValueError("an open joint or temporal role receipt needs new admissible computation")
        citations = self._citations(action.citations)
        if not citations:
            raise ValueError("probe reconciliation requires verified source evidence")
        self.role_evidence[action.probe] = replace(role, reconciliation=replace(action, citations=citations))

    def _record(self, case):
        _text(case.claim, "case claim")
        _text(case.inference, "case inference")
        if case.stance not in ("supports", "defeats", "unknown", "hypothetical"):
            raise ValueError("invalid evidence stance")
        citations = self._citations(case.citations)
        if not citations and case.stance != "hypothetical":
            raise ValueError("an observed case requires verified evidence")
        if case.role_support is not None:
            self._verify_role_support(case.role_support, case.stance, citations, case.need)
        self.cases[case.need].append(replace(case, citations=citations))

    def dossier(self, stop_reason):
        findings = {}
        open_roles = {role.request.need for role in self.role_evidence if role.reconciliation is None and
                      (role.read.status in ("conflict", "unresolved") or
                       (type(role.read) is BindingRoleDecision and role.read.open_coverage))}
        for key, need in reversed(tuple(self.needs.items())):
            cases = tuple(self.cases[key])
            stances = {case.stance for case in cases}
            opened = tuple(child for child in self.children[key] if findings[child].status != "supported")
            if {"supports", "defeats"} <= stances:
                status = "conflicted"
            elif opened or key in open_roles or "unknown" in stances:
                status = "underdetermined"
            elif "supports" in stances:
                status = "supported"
            elif "defeats" in stances:
                status = "defeated"
            elif key in self.exhausted:
                status = "budget_exhausted"
            elif key in self.unavailable:
                status = "source_unavailable"
            else:
                status = "underdetermined" if key in self.investigated else "not_investigated"
            findings[key] = NeedFinding(need, status, cases, opened)
        return TaskDossier(self.task, self.world.pin, tuple(findings[key] for key in self.needs),
                           tuple(self.searches), tuple(self.reads), tuple(self.role_evidence), tuple(self.rejected),
                           self.budget, stop_reason, self.steps,
                           self.source_bytes, self.evidence_bytes, self.role_work, self.lexical_rows,
                           role_transitions=self.role_transitions)
