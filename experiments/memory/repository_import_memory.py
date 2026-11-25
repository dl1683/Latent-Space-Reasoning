"""Trace two explicit Python imports on a pinned, revisable Git snapshot."""

import argparse
import ast
from codecs import BOM_UTF8, getincrementaldecoder
from collections.abc import Mapping
from dataclasses import InitVar, asdict, dataclass, field
from importlib.util import resolve_name
from io import BytesIO
import json
from numbers import Real
from operator import index
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
from tokenize import detect_encoding
from types import MappingProxyType

import numpy as np

from experiments.memory.disjoint_feedback import (
    RelationGuardFailed, RelationOutcomeGuard, TerminalRelationEvidence, TerminalRelationGrade,
    _EvidenceEdit, _RELATION_GRADE_ATOL, _RelationWorkspacePin, _approved_evidence,
    _check_retraction_absence, _matching_witness, _same_supported_grade,
)
from experiments.memory.disjoint_transition import DisjointTransitionMemory, DisjointTransitionPath, propagate_disjoint
from experiments.memory.weight_program import program_weight


_SOURCE_LINE = re.compile(rb"[^\r\n]*(?:\r\n?|\n|$)")
_ASCII_HEADER = bytes(range(128)) + b"\x7f" * 128
_CREATION_KEY = object()


@dataclass(frozen=True)
class ImportEvidence:
    path: str
    line: int
    imported_module: str


@dataclass(frozen=True, slots=True)
class ImportQuote:
    declaring_module: str
    imported_module: str
    path: str
    blob: str
    line: int
    end_line: int
    text: str
    declared_targets: tuple[str, ...]


@dataclass(frozen=True, order=True, slots=True)
class _Declaration:
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    targets: tuple[str, ...] = field(compare=False)


@dataclass(frozen=True, slots=True)
class _Module:
    path: str
    blob: str
    imports: Mapping[str, _Declaration]


def _git(repo, *args, data=None):
    env = dict(os.environ)
    for name in ("GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS",
                 "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS"):
        env.pop(name, None)
    env["GIT_NO_LAZY_FETCH"] = "1"
    result = subprocess.run(["git", "--no-replace-objects", "--literal-pathspecs", "-C", str(repo), *args],
                            input=data, capture_output=True, check=False, env=env)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or "Git command failed")
    return result.stdout


def _module_tree(repo, commit, root, package):
    scope = root.joinpath(*package.split("."))
    listing = _git(repo, "ls-tree", "-rz", "--full-tree", commit, "--", f"{scope}/")
    modules = {}
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        header, path_bytes = entry.split(b"\t", 1)
        mode, kind, blob = header.decode("ascii").split()
        if kind != "blob" or mode not in ("100644", "100755") or not path_bytes.endswith(b".py"):
            continue
        path = PurePosixPath(path_bytes.decode("utf-8"))
        parts = path.relative_to(root).with_suffix("").parts
        parts = parts[:-1] if parts[-1] == "__init__" else parts
        if not all(part.isidentifier() for part in parts):
            continue
        name = ".".join(parts)
        if name in modules:
            raise ValueError(f"ambiguous tracked Python module: {name}")
        modules[name] = str(path), blob
    return modules


def _read_blobs(repo, blobs):
    if not blobs:
        return {}
    response = BytesIO(_git(repo, "cat-file", "--batch", data="\n".join((*blobs, "")).encode("ascii")))
    contents = {}
    for blob in blobs:
        header = response.readline().decode("ascii").split()
        if len(header) != 3 or header[:2] != [blob, "blob"]:
            raise ValueError("could not read a pinned Python blob")
        size = int(header[-1])
        contents[blob] = response.read(size)
        if len(contents[blob]) != size or response.read(1) != b"\n":
            raise ValueError("truncated pinned Python blob")
    return contents


def _module_imports(contents, path, name, package):
    imports = {}
    enclosing_package = name if path.endswith("/__init__.py") else name.rpartition(".")[0]
    syntax = ast.parse(contents, filename=path)
    _decode_python(contents)
    for node in ast.walk(syntax):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            try:
                target = resolve_name("." * node.level + node.module, enclosing_package)
            except ImportError:
                continue
            targets = [target]
        else:
            continue
        scoped = tuple(target for target in targets
                       if target != name and (target == package or target.startswith(package + ".")))
        if not scoped:
            continue
        span = _Declaration(node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, tuple(targets))
        for target in scoped:
            imports[target] = min(span, imports.get(target, span))
    return MappingProxyType(imports)


def _decode_python(contents):
    if contents.startswith(BOM_UTF8):
        return contents.decode("utf-8-sig")
    # The parser has already accepted this blob. The encoding cookie is ASCII,
    # but comments on the first two physical lines can use the declared codec.
    lines = _SOURCE_LINE.finditer(contents)
    encoding, _ = detect_encoding(lambda: next(lines).group().translate(_ASCII_HEADER))
    if encoding in {"utf-8", "ascii", "iso-8859-1"}:
        return contents.decode(encoding)

    # CPython applies the declared codec to universal newlines before parsing.
    # A codec such as unicode_escape can join or insert physical lines. The AST
    # offsets cannot safely identify committed text if those boundaries differ.
    missing_alignment = "source codec does not preserve committed line boundaries"
    normalized = contents.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    *lines, tail = normalized.split(b"\n")
    decoder = getincrementaldecoder(encoding)()
    parser_rows = []
    for line in lines:
        row = decoder.decode(line + b"\n")
        if not row.endswith("\n") or "\n" in row[:-1] or "\r" in row:
            raise ValueError(missing_alignment)
        parser_rows.append(row)
    last = decoder.decode(tail, final=True)
    if "\n" in last or "\r" in last:
        raise ValueError(missing_alignment)
    parser_rows.append(last)
    parser_text = normalized.decode(encoding)
    if parser_text != "".join(parser_rows):
        raise ValueError(missing_alignment)
    decoded = contents.decode(encoding)
    if decoded.replace("\r\n", "\n").replace("\r", "\n") != parser_text:
        raise ValueError(missing_alignment)
    return decoded


def _normalize_feedback_edits(edits):
    if not isinstance(edits, (tuple, list)) or not edits:
        raise ValueError("evidence batch needs a nonempty sequence of edits")
    normalized = {}
    for row in edits:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ValueError("an evidence edit needs an operation, protection index and destination")
        operation, protected_index, destination = row
        if (type(operation) is not str or operation not in ("handoff", "retraction", "restoration")
                or type(destination) is not str or not destination):
            raise ValueError("invalid evidence edit operation or destination")
        key = index(protected_index), destination
        if key in normalized:
            raise ValueError("evidence edits must name distinct protected destinations")
        normalized[key] = operation
    return tuple((normalized[key], *key) for key in sorted(normalized))


@dataclass(frozen=True, eq=False, slots=True)
class RepositoryImportMemory:
    """An in-memory projection of committed import declarations and source lines."""

    commit: str
    _repo: Path
    _root: PurePosixPath
    _package: str
    _modules: Mapping[str, _Module]
    _coordinates: Mapping[str, int]
    _operator: DisjointTransitionMemory
    _previous: object | None = None
    _changed_modules: frozenset[str] = frozenset()
    _identity: object = field(default_factory=object, init=False, repr=False)
    _creation_key: InitVar[object | None] = None

    def __post_init__(self, _creation_key):
        if _creation_key is not _CREATION_KEY:
            raise TypeError("construct repository import snapshots through prepare or revise")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @classmethod
    def prepare(cls, repo, revision, *, package, source_root="src"):
        """Compile a local Git commit under one repo-relative Python package."""
        root = PurePosixPath(source_root)
        if (root.is_absolute() or PureWindowsPath(source_root).drive or ".." in root.parts or "\\" in str(root)
                or not all(part.isidentifier() for part in package.split("."))):
            raise ValueError("expected a repo-relative source root and a dotted Python package")
        empty = MappingProxyType({})
        return cls("", Path(repo).resolve(), root, package, empty, empty,
                   DisjointTransitionMemory.prepare(1, {}), _creation_key=_CREATION_KEY).revise(revision)

    def revise(self, revision):
        """Apply the declared imports from another commit and preserve this snapshot."""
        commit = _git(self._repo, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}").decode().strip()
        if commit == self.commit:
            return self
        tree = _module_tree(self._repo, commit, self._root, self._package)
        modules, pending = {}, {}
        for name, (path, blob) in tree.items():
            old = self._modules.get(name)
            if old is not None and (path, blob) == (old.path, old.blob):
                modules[name] = old
            else:
                pending[name] = path, blob
        contents = _read_blobs(self._repo, sorted({blob for _, blob in pending.values()}))
        for name, (path, blob) in pending.items():
            modules[name] = _Module(path, blob, _module_imports(contents[blob], path, name, self._package))

        coordinates = dict(self._coordinates)
        names = set(modules)
        for name in pending:
            names.update(modules[name].imports)
        for name in sorted(names - coordinates.keys()):
            coordinates[name] = len(coordinates)

        retained_paths = {module.path for module in modules.values()}
        changed_modules = set(pending)
        changes = {}
        for name, module in self._modules.items():
            if name not in modules:
                changed_modules.add(name)
            if module.path not in retained_paths:
                changes[module.path] = None
        for name in pending:
            module = modules[name]
            changes[module.path] = [(hop, coordinates[name], coordinates[target], 1.)
                                    for target in module.imports for hop in (0, 1)]
        operator = self._operator.revise(changes, width=max(1, len(coordinates)))
        return type(self)(commit, self._repo, self._root, self._package,
                          MappingProxyType(modules), MappingProxyType(coordinates), operator,
                          self._identity, frozenset(changed_modules), _creation_key=_CREATION_KEY)

    @property
    def relation_coordinates(self):
        """Names and retained coordinates for placing feedback anchor rows."""
        return self._coordinates

    def relation_path(self, start, middle):
        """Name a candidate witness by the two declaring modules on this pin."""
        self._require_module(start)
        self._require_module(middle)
        return DisjointTransitionPath(self._coordinates[start], self._coordinates[middle],
                                      (self._modules[start].path, self._modules[middle].path))

    def _witness_edges(self, destination, witness):
        return self._path_edges(destination, witness.path)

    def _declares_path(self, destination, path):
        try:
            self._path_edges(destination, path)
        except ValueError:
            return False
        return True

    def _path_edges(self, destination, path):
        names = tuple(self._coordinates)
        coordinates = path.start, path.middle, destination
        if any(not 0 <= coordinate < len(names) for coordinate in coordinates):
            raise ValueError("published witness is outside the repository coordinate frame")
        start, middle, end = (names[coordinate] for coordinate in coordinates)
        self._require_module(end)
        if self.relation_path(start, middle) != path:
            raise ValueError("published witness origins do not match the committed declaring modules")
        edges = (start, middle), (middle, end)
        if any(imported not in self._modules[declaring].imports for declaring, imported in edges):
            raise ValueError("committed declarations do not explain the published witness")
        return edges

    def attach_feedback(self, receiver, anchors):
        """Attach this exact source snapshot to a relation feedback receiver."""
        return receiver._prepare(self._operator, anchors, source=self)

    def _feedback_names(self, previous, receiver, memory):
        receiver._require_memory(memory)
        if (type(previous) is not RepositoryImportMemory or memory._source is not previous
                or memory._transitions is not previous._operator):
            raise ValueError("feedback memory must be attached to the previous source snapshot")
        if self is not previous and self._previous is not previous._identity:
            raise ValueError("feedback revision needs a direct repository successor")
        if any(self._coordinates.get(name) != coordinate for name, coordinate in previous._coordinates.items()):
            raise ValueError("feedback revision must retain all earlier named coordinates")
        return sorted(self._coordinates.keys() - previous._coordinates.keys(), key=self._coordinates.__getitem__)

    def feedback_revision(self, previous, receiver, memory, *, added_anchors=None):
        """Adopt a direct revision, with anchor rows keyed by appended module names."""
        names = self._feedback_names(previous, receiver, memory)
        prior_count = len(previous._coordinates)
        if added_anchors is None:
            if names:
                raise ValueError("a changed repository coordinate frame needs added feedback anchors")
            return receiver._adopt_memory(memory, self._operator, source=self)
        if not isinstance(added_anchors, Mapping) or added_anchors.keys() != set(names):
            raise ValueError("added anchor names must match the new repository coordinates")
        if not names:
            return receiver._adopt_memory(memory, self._operator, source=self)
        suffix = np.asarray([added_anchors[name] for name in names])
        if suffix.shape != (len(names), receiver._kernel._width):
            raise ValueError("expected one added anchor row per new coordinate")
        return receiver._adopt_memory(memory, self._operator, source=self, anchor_rows=suffix,
                                      replace_placeholder=not prior_count)

    def fit_feedback_revision(self, previous, receiver, pin, *, targets, max_update_norm):
        """Draft appended anchors from named (hidden query, seed grade) targets."""
        return self._fit_feedback_revision(previous, receiver, pin, targets, max_update_norm)

    def _fit_feedback_revision(self, previous, receiver, pin, targets, max_update_norm, handoff=None):
        if type(pin) is not _RelationWorkspacePin:
            raise ValueError("anchor fitting needs an issued workspace pin")
        names = self._feedback_names(previous, receiver, pin.memory)
        if not isinstance(targets, Mapping) or not targets or not targets.keys() <= set(names):
            raise ValueError("anchor targets must name at least one appended repository coordinate")
        requests = []
        for name in names:
            if name in targets:
                hidden, grade = targets[name]
                if not isinstance(grade, Real) or not 0. <= grade <= 1.:
                    raise ValueError("target seed grades must be in [0, 1]")
                requests.append((name, receiver._kernel._vector(hidden), float(grade)))
        released_terminal = None if handoff is None else handoff[0]
        opening_inputs = [receiver._relation_read_input(state, step)
                          for protection, state in enumerate(pin.protected)
                          for step, read in enumerate(state._trace)
                          if read is not None and (protection != released_terminal or step + 1 != len(state._trace))]
        if handoff is not None:
            replayed = handoff[1]
            opening_inputs.extend(receiver._relation_read_input(replayed, step)
                                  for step, read in enumerate(replayed._trace[:-1]) if read is not None)
        hidden_width = receiver._kernel._width
        atol = _RELATION_GRADE_ATOL
        with np.errstate(over="raise", invalid="raise"):
            inputs = np.stack([hidden @ receiver._query for _, hidden, _ in requests])
            preserve = np.asarray([hidden @ receiver._query for hidden in opening_inputs]).reshape(-1, hidden_width)
            scores = np.diag([receiver._floor + grade for _, _, grade in requests])
        initial = np.zeros((hidden_width, len(requests)))
        fit = None
        if released_terminal is not None:
            minimal = program_weight(initial, inputs, scores, max_update_norm=max_update_norm, atol=atol)
            with np.errstate(over="raise", invalid="raise"):
                if (preserve @ minimal.weight < receiver._floor).all():
                    fit = minimal
        if fit is None:
            fit = program_weight(initial, inputs, scores, preserve_inputs=preserve,
                                 max_update_norm=max_update_norm, atol=atol)
        anchors = {name: np.zeros(hidden_width) for name in names}
        anchors.update({name: row for (name, _, _), row in zip(requests, fit.weight.T, strict=True)})
        revision = self.feedback_revision(previous, receiver, pin.memory, added_anchors=anchors)
        prior_count = len(previous._coordinates)
        for hidden in opening_inputs:
            _, active = receiver._relation_seed(revision.memory, hidden)
            if active[prior_count:].any():
                raise ValueError("installed anchors activate new entries on a protected opening query")
        for name, hidden, grade in requests:
            seed, active = receiver._relation_seed(revision.memory, hidden)
            coordinate = self._coordinates[name]
            match = seed[coordinate] == 0. if grade == 0. else abs(seed[coordinate] - grade) <= atol
            if not active[coordinate] or np.count_nonzero(active[prior_count:]) != 1 or not match:
                raise ValueError("installed anchors miss the requested exclusive seed grade")
        return revision

    def draft_feedback_growth(self, previous, receiver, pin, probe, *, destination, grade, max_update_norm):
        """Find one appended entry for an ordinary miss and draft its future protection."""
        if type(pin) is not _RelationWorkspacePin:
            raise ValueError("growth discovery needs an issued workspace pin")
        names = self._feedback_names(previous, receiver, pin.memory)
        receiver._require_state(probe)
        if (probe._memory is not pin.memory or probe._grades is None
                or any(read is None or read.query is not None for read in probe._trace)):
            raise ValueError("growth discovery needs an ordinary terminal read on the opening pin")
        self._require_module(destination)
        coordinate = self._coordinates[destination]
        if coordinate < len(previous._coordinates) and np.isfinite(probe._grades[coordinate]):
            raise ValueError("opening trajectory already supports the destination")

        entry, paths = self._unique_appended_route(names, destination)
        control = self._zero_anchor_revision(previous, receiver, pin.memory, names)
        missed, _ = receiver.rebase(probe, control)
        if np.isfinite(missed._grades[coordinate]):
            raise ValueError("source adoption alone already supports the destination")
        hidden = receiver._relation_read_input(missed, len(missed._trace) - 1)
        revision = self.fit_feedback_revision(previous, receiver, pin, targets={entry: (hidden, grade)},
                                              max_update_norm=max_update_norm)
        grade = float(grade)
        bounds = ((0., 0.) if grade == 0. else
                  (max(0., grade - _RELATION_GRADE_ATOL), min(1., grade + _RELATION_GRADE_ATOL)))
        guard = RelationOutcomeGuard(grades=[TerminalRelationGrade(coordinate, bounds)],
                                     evidence=[TerminalRelationEvidence(coordinate, paths)])
        return revision._with_required_outcome(pin, probe, guard)

    def _unique_appended_route(self, names, destination):
        paths = ()
        for start in names:
            if start not in self._modules:
                continue
            candidates = tuple(self.relation_path(start, middle) for middle in sorted(self._modules[start].imports)
                               if middle != start and middle in self._modules
                               and destination in self._modules[middle].imports)
            if candidates:
                if paths:
                    raise ValueError("multiple appended entries explain the destination")
                entry, paths = start, candidates
        if not paths:
            raise ValueError("no appended entry explains the destination")
        return entry, paths

    def _zero_anchor_revision(self, previous, receiver, memory, names):
        zeros = np.zeros(receiver._kernel._width)
        return self.feedback_revision(previous, receiver, memory, added_anchors=dict.fromkeys(names, zeros))

    def draft_feedback_handoff(self, previous, receiver, pin, protected_index, *, destination, max_update_norm):
        """Replace one lost committed witness clause under the same protected outcome."""
        if type(pin) is not _RelationWorkspacePin:
            raise ValueError("evidence handoff needs an issued workspace pin")
        names = self._feedback_names(previous, receiver, pin.memory)
        protected_index = index(protected_index)
        if not 0 <= protected_index < len(pin.protected) or pin._outcomes[protected_index] is None:
            raise ValueError("evidence handoff needs an existing protected outcome")
        self._require_module(destination)
        coordinate = self._coordinates[destination]
        outcome = pin._outcomes[protected_index]
        outcome._evidence_at(coordinate)
        incumbent = dict(pin.witnesses[protected_index])[coordinate]
        probe = pin.protected[protected_index]
        control = self._zero_anchor_revision(previous, receiver, pin.memory, names)
        replayed, _ = receiver.rebase(probe, control)
        replacement = self._source_handoff(receiver, pin, protected_index, coordinate, replayed)
        if replacement is not None and outcome._with_replaced_evidence(replacement)._evaluate(replayed) is not None:
            return control._with_evidence_handoff(pin, protected_index, replacement)

        entry, paths = self._unique_appended_route(names, destination)
        hidden = receiver._relation_read_input(replayed, len(replayed._trace) - 1)
        revision = self._fit_feedback_revision(previous, receiver, pin, {entry: (hidden, incumbent.strength)},
                                               max_update_norm, handoff=(protected_index, replayed))
        return revision._with_evidence_handoff(pin, protected_index, TerminalRelationEvidence(coordinate, paths))

    def _source_handoff(self, receiver, pin, protected_index, coordinate, replayed):
        original = pin._outcomes[protected_index]._evidence_at(coordinate)
        seed = receiver._terminal_seed(replayed)
        if _approved_evidence(replayed, seed, original) is not None:
            raise ValueError("an existing path still certifies the handoff destination")
        return self._replacement_handoff(pin, protected_index, coordinate, replayed, seed)

    def _replacement_handoff(self, pin, protected_index, coordinate, replayed, seed):
        existing = _matching_witness(replayed, seed, coordinate, None)
        previous = dict(pin.witnesses[protected_index])[coordinate]
        if existing is not None and _same_supported_grade(previous.strength, existing.strength):
            return TerminalRelationEvidence(coordinate, (existing.path,))
        return None

    def draft_feedback_retraction(self, previous, receiver, pin, protected_index, *, destination):
        """Propose withdrawing one lost committed witness and guarding the resulting absence."""
        if type(pin) is not _RelationWorkspacePin:
            raise ValueError("evidence retraction needs an issued workspace pin")
        names = self._feedback_names(previous, receiver, pin.memory)
        if self is previous:
            raise ValueError("evidence retraction needs a committed source correction")
        protected_index = index(protected_index)
        if not 0 <= protected_index < len(pin.protected) or pin._outcomes[protected_index] is None:
            raise ValueError("evidence retraction needs an existing protected outcome")
        previous._require_module(destination)
        coordinate = previous._coordinates[destination]
        pin._outcomes[protected_index]._evidence_at(coordinate)
        revision = self._zero_anchor_revision(previous, receiver, pin.memory, names)
        return revision._with_evidence_retraction(pin, protected_index, coordinate)

    def draft_feedback_restoration(self, previous, receiver, pin, protected_index, *, destination):
        """Propose a new committed witness for one currently retracted destination."""
        if type(pin) is not _RelationWorkspacePin:
            raise ValueError("evidence restoration needs an issued workspace pin")
        names = self._feedback_names(previous, receiver, pin.memory)
        if self is previous:
            raise ValueError("evidence restoration needs a committed source correction")
        protected_index = index(protected_index)
        if not 0 <= protected_index < len(pin.protected):
            raise ValueError("evidence restoration needs an existing protected outcome")
        self._require_module(destination)
        coordinate = self._coordinates[destination]
        receipt = pin._active_retraction(protected_index, coordinate)
        if receipt is None:
            raise ValueError("evidence restoration needs an active retraction")
        revision = self._zero_anchor_revision(previous, receiver, pin.memory, names)
        candidate, _ = receiver.rebase(pin.protected[protected_index], revision)
        evidence = self._source_restoration(candidate, receiver._terminal_seed(candidate), coordinate, receipt)
        return revision._with_evidence_restoration(pin, receipt, evidence)

    def _source_restoration(self, candidate, seed, coordinate, receipt):
        old_path = receipt.previous_witness.path
        witness = _matching_witness(candidate, seed, coordinate, (old_path,))
        if witness is None:
            witness = _matching_witness(candidate, seed, coordinate, None)
        if witness is None or not _same_supported_grade(receipt.previous_witness.strength, witness.strength):
            raise ValueError("the corrected source does not restore the certified terminal grade")
        self._witness_edges(coordinate, witness)
        return TerminalRelationEvidence(coordinate, (witness.path,))

    def draft_feedback_batch(self, previous, receiver, pin, *, edits):
        """Draft coupled source-only evidence edits against one opening pin and successor."""
        if type(pin) is not _RelationWorkspacePin:
            raise ValueError("evidence batch needs an issued workspace pin")
        names = self._feedback_names(previous, receiver, pin.memory)
        if self is previous:
            raise ValueError("evidence batch needs a committed source correction")
        requested = _normalize_feedback_edits(edits)
        revision = self._zero_anchor_revision(previous, receiver, pin.memory, names)
        replays = {}
        changes = []
        for operation, protected_index, destination in requested:
            outcome = pin._protected_outcome(protected_index)
            domain = previous if operation == "retraction" else self
            domain._require_module(destination)
            coordinate = domain._coordinates[destination]
            if operation == "retraction":
                outcome._evidence_at(coordinate)
                changes.append(_EvidenceEdit(operation, protected_index, coordinate))
                continue
            if protected_index not in replays:
                replays[protected_index], _ = receiver.rebase(pin.protected[protected_index], revision)
            candidate = replays[protected_index]
            if operation == "handoff":
                replacement = self._source_handoff(receiver, pin, protected_index, coordinate, candidate)
                if replacement is None:
                    raise ValueError("batch handoff needs source-only support at the certified terminal grade")
                changes.append(_EvidenceEdit(operation, protected_index, coordinate, replacement=replacement))
            else:
                receipt = pin._active_retraction(protected_index, coordinate)
                if receipt is None:
                    raise ValueError("evidence restoration needs an active retraction")
                replacement = self._source_restoration(candidate, receiver._terminal_seed(candidate), coordinate, receipt)
                changes.append(_EvidenceEdit(operation, protected_index, coordinate, replacement, receipt))
        return revision._with_evidence_edits(pin, changes)

    def draft_feedback_maintenance(self, previous, receiver, pin):
        """Find source-only evidence edits and return their draft and canonical target list."""
        if type(pin) is not _RelationWorkspacePin:
            raise ValueError("evidence maintenance needs an issued workspace pin")
        names = self._feedback_names(previous, receiver, pin.memory)
        revision = self._zero_anchor_revision(previous, receiver, pin.memory, names)
        if self is previous:
            return revision, ()

        # A receipt can be active only while its destination has an absence guard.
        closed = {id(result.retraction) for result in pin.restorations}
        active = {}
        for receipt in pin.retractions:
            if id(receipt) not in closed:
                active.setdefault(receipt.protected_index, []).append(receipt)

        coordinate_names = tuple(self._coordinates)
        changes = []
        for i, (state, outcome) in enumerate(zip(pin.protected, pin._outcomes, strict=True)):
            receipts = active.get(i, ())
            clauses = () if outcome is None else outcome.evidence
            if not clauses and not receipts:
                continue
            candidate, _ = receiver.rebase(state, revision)
            seed = receiver._terminal_seed(candidate)
            for clause in clauses:
                coordinate = clause.destination
                if _approved_evidence(candidate, seed, clause) is not None:
                    continue
                if np.isneginf(candidate._grades[coordinate]):
                    if any(self._declares_path(coordinate, path) for path in clause.paths):
                        raise RelationGuardFailed(
                            f"maintenance cannot retract {coordinate_names[coordinate]} at protection {i}: "
                            "a declared source path survives"
                        )
                    _check_retraction_absence(state, candidate, coordinate)
                    changes.append(_EvidenceEdit("retraction", i, coordinate))
                    continue
                replacement = self._replacement_handoff(pin, i, coordinate, candidate, seed)
                if replacement is None:
                    raise RelationGuardFailed(
                        f"maintenance needs source-only support at the certified terminal grade for "
                        f"{coordinate_names[coordinate]} at protection {i}"
                    )
                self._path_edges(coordinate, replacement.paths[0])
                changes.append(_EvidenceEdit("handoff", i, coordinate, replacement=replacement))
            for receipt in receipts:
                coordinate = receipt.destination
                if np.isneginf(candidate._grades[coordinate]):
                    continue
                evidence = self._source_restoration(candidate, seed, coordinate, receipt)
                changes.append(_EvidenceEdit("restoration", i, coordinate, evidence, receipt))
        if not changes:
            return revision, ()
        edits = _normalize_feedback_edits([
            (edit.operation, edit.protected_index, coordinate_names[edit.destination]) for edit in changes
        ])
        return revision._with_evidence_edits(pin, changes), edits

    def _seed(self, start):
        self._require_module(start)
        seed = np.full(self._operator.width, -np.inf)
        seed[self._coordinates[start]] = 1.
        return seed

    def _require_module(self, name):
        if name not in self._modules:
            raise ValueError(f"module is outside this committed Python snapshot: {name}")

    def destinations(self, start):
        """List tracked destinations supported by exactly two import declarations."""
        scores = propagate_disjoint(self._seed(start), memory=self._operator)
        return tuple(sorted(name for name in self._modules if scores[self._coordinates[name]] == 1.))

    def prove(self, start, destination):
        """Return the two source lines supporting a path, or None for modeled absence."""
        self._require_module(destination)
        names = tuple(self._coordinates)
        witness = self._operator.prove(self._seed(start), self._coordinates[destination], middle_key=names.__getitem__)
        if witness is None:
            return None
        left, right = witness.origins
        middle = names[witness.middle]
        return (ImportEvidence(left, self._modules[start].imports[middle].lineno, middle),
                ImportEvidence(right, self._modules[middle].imports[destination].lineno, destination))

    def read_imports(self, edges):
        """Read exact declarations for named edges, charging each pinned blob once."""
        requested = []
        for declaring, imported in edges:
            self._require_module(declaring)
            self._require_module(imported)
            module = self._modules[declaring]
            if imported not in module.imports:
                raise ValueError(f"no declared import from {declaring} to {imported}")
            requested.append((declaring, imported, module, module.imports[imported]))
        contents = _read_blobs(self._repo, sorted({module.blob for _, _, module, _ in requested}))
        decoded = {blob: _decode_python(raw) for blob, raw in contents.items()}
        quotes = []
        for declaring, imported, module, span in requested:
            fragment = ast.get_source_segment(decoded[module.blob], span)
            if not fragment:
                raise ValueError("pinned import declaration could not be quoted")
            quotes.append(ImportQuote(declaring, imported, module.path, module.blob,
                                      span.lineno, span.end_lineno, fragment, span.targets))
        return tuple(quotes), sum(map(len, contents.values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--package", required=True)
    parser.add_argument("--source-root", default="src")
    parser.add_argument("--revision", required=True, action="append")
    parser.add_argument("--start", required=True)
    parser.add_argument("--destination")
    args = parser.parse_args(argv)
    first, *later = args.revision
    try:
        memory = RepositoryImportMemory.prepare(args.repo, first, package=args.package, source_root=args.source_root)
        snapshots = [memory]
        for revision in later:
            snapshots.append(snapshots[-1].revise(revision))
        output = []
        for pin in snapshots:
            row = {"commit": pin.commit, "destinations": pin.destinations(args.start)}
            if args.destination is not None:
                evidence = pin.prove(args.start, args.destination)
                row["evidence"] = None if evidence is None else [asdict(item) for item in evidence]
            output.append(row)
    except (OSError, SyntaxError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
