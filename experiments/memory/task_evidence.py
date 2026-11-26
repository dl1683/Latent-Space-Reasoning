"""Bounded task-time evidence investigation over a pinned source world."""

from dataclasses import asdict, dataclass, replace
import json
from math import isfinite
from operator import index
from typing import Callable

from experiments.memory.binding_links import BindingLinkCondition, BindingLinkRead
from experiments.memory.compressed_routing_map import BM25
from experiments.memory.corpus_bindings import CorpusRevision
from experiments.memory.verified_source_tool import SourceSpan, VerifiedSourceTool, sha256


@dataclass(frozen=True, slots=True)
class TaskPin:
    source_snapshot: str
    span_catalogue_sha256: str
    binding_revision: int | None
    binding_source_revision: int | None
    binding_program_revision: int | None
    binding_journal_sha256: str | None


@dataclass(frozen=True, slots=True, init=False)
class TaskWorld:
    """Associate verified source text and an optional retained computation."""

    _tool: VerifiedSourceTool
    _spans: tuple[SourceSpan, ...]
    _index: BM25 | None
    _bindings: CorpusRevision | None
    pin: TaskPin
    setup_source_bytes: int

    def __init__(self, tool, spans, *, bindings=None):
        if type(tool) is not VerifiedSourceTool:
            raise TypeError("a verified source tool is required")
        if bindings is not None and (
            type(bindings) is not CorpusRevision or bindings.memory.source_digests != tool.versions
        ):
            raise ValueError("retained computation and source catalogue must use the same sources and versions")
        if bindings is not None and bindings.journal_sha256 is None:
            raise ValueError("a retained computation requires a journaled revision")
        tool = VerifiedSourceTool(tool.root, tool.versions)
        spans = tuple(spans)
        if any(type(span) is not SourceSpan for span in spans) or len(set(spans)) != len(spans):
            raise ValueError("search catalogue requires distinct source spans")
        texts, setup_bytes = [], 0
        for span in spans:
            opened = tool.verify(span)
            if opened["status"] != "VERIFIED_SOURCE_BYTES":
                raise ValueError("search catalogue contains an unverifiable span")
            texts.append(opened["text"])
            setup_bytes += opened["source_bytes_read"]
        catalogue = json.dumps([asdict(span) for span in spans], sort_keys=True, separators=(",", ":")).encode()
        pin = TaskPin(tool.snapshot_sha256, sha256(catalogue), None if bindings is None else bindings.revision,
                      None if bindings is None else bindings.source_revision,
                      None if bindings is None else bindings._program_revision,
                      None if bindings is None else bindings.journal_sha256)
        object.__setattr__(self, "_tool", tool)
        object.__setattr__(self, "_spans", spans)
        object.__setattr__(self, "_index", BM25(texts) if texts else None)
        object.__setattr__(self, "_bindings", bindings)
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

    def __post_init__(self):
        for name in ("steps", "candidates_per_search", "evidence_bytes", "source_bytes", "role_work"):
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
class EvidenceCase:
    need: str
    stance: str
    claim: str
    inference: str
    citations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReconcileProbe:
    need: str
    probe: int
    rationale: str
    citations: tuple[int, ...]


TaskAction = EvidenceNeed | SearchEvidence | OpenEvidence | ProbeRoleLinks | EvidenceCase | ReconcileProbe


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
    request: ProbeRoleLinks
    read: BindingLinkRead
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


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


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
        self.lexical_rows = 0

    def _citations(self, values):
        ids = tuple(values)
        if (any(type(i) is not int or not 0 <= i < len(self.reads)
                or self.reads[i].status != "VERIFIED_SOURCE_BYTES" for i in ids) or len(ids) != len(set(ids))):
            raise ValueError("citations must identify distinct verified reads from this investigation")
        return ids

    def apply(self, action):
        if type(action) is EvidenceNeed:
            self._add_need(action)
        else:
            if type(action) not in (SearchEvidence, OpenEvidence, ProbeRoleLinks, EvidenceCase, ReconcileProbe):
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
        self.needs[need.id] = replace(need, discovered_from=self._citations(need.discovered_from))
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
        bindings = self.world._bindings
        if (bindings is not None and opened["status"] == "VERIFIED_SOURCE_BYTES"
                and span.start_byte == 0 and span.end_byte == opened["source_bytes_read"]):
            observations = tuple(mention.predicate for mention in bindings.memory.source_mentions(span.source_id) or ())
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
        bindings = self.world._bindings
        origin = None
        if action.source_observation is None:
            direction = tuple(map(float, action.direction)) if action.direction is not None else ()
            if not direction or not all(map(isfinite, direction)):
                raise ValueError("role probe requires a finite direction or a verified source observation")
        else:
            if action.direction is not None:
                raise ValueError("choose one role probe input")
            read_id, ordinal = action.source_observation
            self._citations((read_id,))
            evidence = self.reads[read_id]
            if (type(ordinal) is not int or ordinal < 0 or evidence.span.start_byte != 0
                    or evidence.span.end_byte != evidence.source_bytes_read):
                raise ValueError("role source must identify an observation in a verified complete source")
            origin, direction = (read_id, ordinal), None
            if bindings is not None:
                if ordinal >= len(evidence.observation_predicates):
                    raise ValueError("role observation is unavailable on this pin")
                mentions = bindings.memory.source_mentions(evidence.span.source_id)
                direction = tuple(map(float, mentions[ordinal].vector))
        request = replace(action, direction=direction if origin is None else None, outcomes=outcomes, hops=hops,
                          source_observation=origin)
        remaining = self.budget.role_work - self.role_work
        if bindings is None:
            read = BindingLinkRead("no_memory", outcomes=outcomes, hops=hops, condition=action.condition)
        elif not remaining:
            read = BindingLinkRead("work_limit", outcomes=outcomes, hops=hops, condition=action.condition)
        else:
            read = bindings.memory.read_role_links(direction, outcomes=outcomes, hops=hops,
                                                   work_limit=remaining, condition=action.condition)
            self.role_work += read.work
        if read.status == "work_limit":
            self.exhausted.add(action.need)
        self.investigated.add(action.need)
        self.role_evidence.append(RoleEvidence(request, read, direction))

    def _reconcile(self, action):
        _text(action.rationale, "probe reconciliation rationale")
        if type(action.probe) is not int or not 0 <= action.probe < len(self.role_evidence):
            raise ValueError("probe reconciliation requires an existing role receipt")
        role = self.role_evidence[action.probe]
        if (role.request.need != action.need or role.read.status not in ("conflict", "unresolved")
                or role.reconciliation is not None):
            raise ValueError("probe reconciliation requires an open role receipt for this need")
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
        self.cases[case.need].append(replace(case, citations=citations))

    def dossier(self, stop_reason):
        findings = {}
        open_roles = {role.request.need for role in self.role_evidence
                      if role.read.status in ("conflict", "unresolved") and role.reconciliation is None}
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
                           self.source_bytes, self.evidence_bytes, self.role_work, self.lexical_rows)
