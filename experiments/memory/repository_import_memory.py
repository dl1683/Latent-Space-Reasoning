"""Trace two explicit Python imports on a pinned, revisable Git snapshot."""

import argparse
import ast
from codecs import BOM_UTF8, getincrementaldecoder
from collections.abc import Mapping
from dataclasses import InitVar, asdict, dataclass, field
from importlib.util import resolve_name
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
from tokenize import detect_encoding
from types import MappingProxyType

import numpy as np

from experiments.memory.disjoint_transition import DisjointTransitionMemory, DisjointTransitionPath, propagate_disjoint


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
        names = tuple(self._coordinates)
        coordinates = witness.start, witness.middle, destination
        if any(not 0 <= coordinate < len(names) for coordinate in coordinates):
            raise ValueError("published witness is outside the repository coordinate frame")
        start, middle, end = (names[coordinate] for coordinate in coordinates)
        self._require_module(end)
        if self.relation_path(start, middle) != witness.path:
            raise ValueError("published witness origins do not match the committed declaring modules")
        edges = (start, middle), (middle, end)
        if any(imported not in self._modules[declaring].imports for declaring, imported in edges):
            raise ValueError("committed declarations do not explain the published witness")
        return edges

    def attach_feedback(self, receiver, anchors):
        """Attach this exact source snapshot to a relation feedback receiver."""
        return receiver._prepare(self._operator, anchors, source=self)

    def feedback_revision(self, previous, receiver, memory):
        """Adopt a directly revised repository snapshot with the same coordinate mapping."""
        receiver._require_memory(memory)
        if (type(previous) is not RepositoryImportMemory or memory._source is not previous
                or memory._transitions is not previous._operator):
            raise ValueError("feedback memory must be attached to the previous source snapshot")
        if self is not previous and self._previous is not previous._identity:
            raise ValueError("feedback revision needs a direct repository successor")
        if self._coordinates != previous._coordinates:
            raise ValueError("a changed repository coordinate frame needs fresh feedback anchors")
        return receiver._adopt_memory(memory, self._operator, source=self)

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
