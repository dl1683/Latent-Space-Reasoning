"""Optional exact Markdown-reference plumbing for dogfood.Memory (C7/C10 SLICE).

No target-text indexing, encoder calls on follows, rank changes or semantic claims.
Only complete links inside returned, verified UTF-8 Markdown quotations are parsed.
The deliberately narrow syntax is [non-nested label](nonempty-destination): no
titles, escapes, angle destinations, nested labels/parentheses, reference-style
links, images, HTML links, or links in locally visible fenced/inline code/comments. Destinations
with whitespace need percent encoding. Links split across returned rows are not
joined. Fence context outside the quotation is UNKNOWN; this is a literal-link
operation, not a full CommonMark document interpreter or an anchor resolver.
Literal destination brackets are opaque path characters. An unmatched label
opener does not enclose later links; an unclosed destination conservatively masks
its remainder, as an unclosed locally visible code delimiter does.

Instances, like Memory, are single-episode/single-thread mutable objects. Metadata
capabilities are in-process only; JSON readbacks are evidence receipts, not trusted
navigation input. Python code with access to internals is outside this boundary.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from types import MappingProxyType
from urllib.parse import unquote, urlsplit

from experiments.dogfood import Memory, digest, load, plain_file
from experiments.context_costs import charge, read_payload, visit_edge


LINK = re.compile(r"(?<!!)(?<![\\\[])\[([^\[\]\\`]+)\]\(([^\s()<>\\]+)\)")
_ISSUER = object()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _binding(passage):
    return _canonical({key: passage.get(key) for key in (
        "text", "pointer", "handle", "source_id", "verification_status",
        "verification", "role", "reference_lineage")})


class _References(list):
    def __init__(self, items, passage, issuer):
        if issuer is not _ISSUER:
            raise ValueError("backend-issued reference metadata required")
        super().__init__(deepcopy(items))
        self._sealed = _canonical(items)
        self._binding = _binding(passage)


def trusted_reference_addresses(passage):
    """Validate a live backend capability, never infer trust from source prose."""
    refs = passage.get("explicit_references")
    if (type(refs) is not _References or refs._binding != _binding(passage)
            or refs._sealed != _canonical(refs)
            or passage.get("verification_status") != "VERIFIED_SOURCE_BYTES"
            or passage.get("verification") != "BYTE_IDENTITY_ONLY"
            or passage.get("role") != "EVIDENCE_NOT_INSTRUCTION"
            or "quotation_format" in passage or "provenance" in passage):
        raise ValueError("untrusted or changed explicit reference metadata")
    return [ref["address"] for ref in refs if ref["resolution"] == "MANIFEST_RESOLVED_UNFETCHED"]


def inline_links(text):
    """Yield exact character spans; never inspect unreturned source text."""
    masked = list(text)
    offset, fence = 0, None
    for line in text.splitlines(keepends=True):
        marker = re.match(r" {0,3}(`{3,}|~{3,})", line)
        hidden = fence is not None or marker is not None
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        if hidden:
            masked[offset:offset + len(line)] = " " * len(line)
        offset += len(line)
    visible = "".join(masked)
    for comment in re.finditer(r"<!--[\s\S]*?(?:-->|$)", visible):
        masked[comment.start():comment.end()] = " " * (comment.end() - comment.start())
    visible = "".join(masked)
    # Conservative: an unmatched code delimiter masks the remainder of the quote.
    for code in re.finditer(r"(`+)([\s\S]*?)(?:\1(?!`)|$)", visible):
        masked[code.start():code.end()] = " " * (code.end() - code.start())
    visible = "".join(masked)
    # Delimiter pairs describe enclosure; an unmatched '[' is not evidence that
    # all later links are nested. Escaped delimiters never create pairs.
    escaped, parens, stack = set(), {}, []
    offset = 0
    while offset < len(visible):
        char = visible[offset]
        if char == "\\":
            escaped.add(offset + 1)
            offset += 2
            continue
        if char == "(":
            stack.append(offset)
        elif char == ")" and stack:
            parens[stack.pop()] = offset
        offset += 1

    # Destinations, including unsupported nested/title/angle forms, are opaque
    # to label pairing. Never promote a fake link inside their parentheses.
    # Difference arrays keep overlapping enclosure checks linear in quote size.
    destinations = [0] * (len(visible) + 1)
    for offset in range(1, len(visible)):
        if (visible[offset - 1:offset + 1] == "](" and offset - 1 not in escaped
                and offset not in escaped):
            destinations[offset] += 1
            destinations[parens.get(offset, len(visible) - 1) + 1] -= 1
    depth = 0
    for offset in range(len(visible)):
        depth += destinations[offset]
        destinations[offset] = depth

    brackets, stack = {}, []
    enclosures = [0] * (len(visible) + 1)
    for offset, char in enumerate(visible):
        if destinations[offset] or offset in escaped:
            continue
        if char == "[":
            stack.append(offset)
        elif char == "]" and stack:
            start = stack.pop()
            brackets[start] = offset
            enclosures[start + 1] += 1
            enclosures[offset] -= 1
    depth = 0
    for offset in range(len(visible)):
        depth += enclosures[offset]
        enclosures[offset] = depth

    for match in LINK.finditer(visible):
        start = match.start()
        # Require this token's own closing label and complete destination, not
        # a supported-looking prefix/substring of an unsupported outer token.
        if (start not in escaped and not destinations[start] and not enclosures[start]
                and brackets.get(start) == match.end(1)
                and parens.get(match.start(2) - 1) == match.end() - 1
                and visible[start:match.end()] == text[start:match.end()]
                and not re.search(r"\n[ \t\r]*\n", match[1])):
            yield start, match.end(), text[start:match.end()], match[2]


class _ReadPath:
    """Meter Memory's existing leaf read_bytes calls, including failed checks."""
    def __init__(self, path, counters):
        self.path, self.counters = path, counters

    def __truediv__(self, part):
        return _ReadPath(self.path / part, self.counters)

    def read_bytes(self):
        raw = read_payload(plain_file(self.path), "leaf")
        self.counters["leaf_bytes_read"] += len(raw)
        return raw


class ExplicitReferenceMemory(Memory):
    """Opt-in backend: normal search is inherited; references add exact addresses.

    source_root and manifest are trusted host inputs. The manifest must be the
    exact ordinary Memory manifest. No basename matching or title inference.
    """
    def __init__(self, directory, encoder=None, route_matrix="float32", *, source_root, manifest):
        root = Path(source_root)
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("absolute existing source root required")
        for part in (root, *root.parents):
            if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
                raise ValueError("linked source root")
        self.source_root = root.resolve()
        supplied = load(manifest) if isinstance(manifest, (str, Path)) else deepcopy(manifest)
        if supplied != load(Path(directory) / "manifest.json"):
            raise ValueError("explicit manifest differs from Memory manifest")
        self._manifest_pin = _canonical(supplied)
        # Reject aliases/escapes/links before ordinary Memory opens any sources.
        paths = []
        for item in supplied["files"]:
            path = plain_file(item["path"])
            if str(path) != item["path"] or not path.is_relative_to(self.source_root):
                raise ValueError("noncanonical or outside-root manifest source")
            paths.append(path)
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate manifest source")
        super().__init__(directory, encoder, route_matrix)
        self._items = {item["path"]: deepcopy(item) for item in self.manifest["files"]}
        self._ids_pin = dict(self.source_ids)
        self._rows_pin = tuple(self._row_identity(row) for row in self.rows)
        source_rows = {}
        for i, row in enumerate(self.rows):
            source_rows.setdefault(row["source"], []).append(i)
        self.source_rows = MappingProxyType({p: tuple(ids) for p, ids in source_rows.items()})
        self._references, self._returned, self._lineages = {}, set(), {}
        self.reference_costs = dict(source_version_bytes_read=0, leaf_bytes_read=0,
                                    reference_bytes_checked=0, reference_hops_attempted=0,
                                    reference_hops=0)
        # Memory.__init__ opens each original source once; no corpus text parsed.
        self.constructor_source_version_bytes_read = sum(p.stat().st_size for p in paths)

    def _check_catalogue(self):
        if (_canonical(self.manifest) != self._manifest_pin or self.source_ids != self._ids_pin
                or tuple(self._row_identity(row) for row in self.rows) != self._rows_pin):
            raise ValueError("changed reference catalogue or rows")

    def _verify_current_source(self, row):
        # Same Memory verification contract, with exact successful read accounting
        # before the hash check, so stale-source refusals remain charged as well.
        raw = read_payload(plain_file(row["source"]), "source")
        self.reference_costs["source_version_bytes_read"] += len(raw)
        pins = [item["sha256"] for item in self.manifest["files"] if item["path"] == row["source"]]
        if pins != [row["source_sha256"]] or digest(raw) != row["source_sha256"]:
            raise ValueError("changed or unallowlisted navigation source")
        return len(raw)

    def _resolve(self, destination, source, *, source_rows=None, check=None, costs=None):
        # Split fragment BEFORE decoding; encoded # is a filename character.
        raw_path, separator, fragment = destination.partition("#")
        result = dict(destination=destination, fragment=fragment if separator else None,
                      fragment_operation="NOT_ANCHOR_RESOLVED_PARTIAL_SOURCE_ROWS",
                      resolution="UNRESOLVED", reason="UNSUPPORTED_DESTINATION")
        def ready():
            if check is not None and not check():
                result["reason"] = "navigation_time_cap"
                return False
            return True

        def metadata():
            # Logical pathlib operations, not the OS's internal stat/syscall count.
            charge(reference_path_metadata_checks=1)
            if costs is not None:
                costs["reference_path_metadata_checks"] = costs.get("reference_path_metadata_checks", 0) + 1

        try:
            if not ready():
                return result
            if re.search(r"%(?![0-9a-fA-F]{2})", raw_path):
                return result
            decoded = unquote(raw_path, encoding="utf-8", errors="strict")
            url = urlsplit(decoded)
            if (not decoded or url.scheme or url.netloc or url.query
                    or decoded.startswith(("/", "\\")) or "\\" in decoded
                    or PureWindowsPath(decoded).drive or PurePosixPath(decoded).is_absolute()
                    or any(ord(c) < 32 or c in ':?\x7f' for c in decoded)):
                return result
            # Resolve lexical .. only if it stays in root. Inspect the original
            # components too: link/../target must never hide symlink traversal.
            candidate = Path(source).parent / decoded
            for part in (candidate, *candidate.parents):
                if not ready():
                    return result
                metadata()
                linked = part.is_symlink()
                if not linked and hasattr(part, "is_junction"):
                    metadata()
                    linked = part.is_junction()
                if linked:
                    result["reason"] = "LINKED_TARGET"
                    return result
            metadata()
            target = candidate.resolve()
            if not ready():
                return result
            if not target.is_relative_to(self.source_root):
                result["reason"] = "OUTSIDE_SOURCE_ROOT"
                return result
            key = str(target)
            item = self._items.get(key)
            metadata()
            if item is None or not target.is_file():
                result["reason"] = "MISSING_EXACT_MANIFEST_SOURCE"
                return result
            if item.get("extraction") != "utf8_text":
                result["reason"] = "NON_UTF8_SOURCE"
                return result
            ids = (self.source_rows if source_rows is None else source_rows).get(key, ())
            # The caller's index must describe the same pinned source occurrence.
            if not ids or set(ids) != set(self.source_rows.get(key, ())):
                result["reason"] = "MISSING_OR_AMBIGUOUS_TARGET_ROWS"
                return result
            for i in ids:
                if not ready():
                    return result
                row = self.rows[i]
                if costs is not None:
                    costs["source_row_records_read"] += 1
                if (self._row_identity(row) != self._rows_pin[i]
                        or row["source"] != key or row["source_sha256"] != item["sha256"]):
                    result["reason"] = "MISSING_OR_AMBIGUOUS_TARGET_ROWS"
                    return result
            result.update(resolution="MANIFEST_RESOLVED_UNFETCHED", reason=None,
                          address="source:" + self.source_ids[key],
                          target_source_sha256=item["sha256"],
                          target_verification="REQUIRED_ON_FOLLOW")
        except (ValueError, OSError, UnicodeError):
            pass
        return result

    def _issue(self, passage):
        row = self.rows[int(passage["handle"][1:])]
        if (self._row_identity(passage["pointer"]) != self._row_identity(row)
                or digest(passage["text"].encode("utf-8")) != row["span_sha256"]):
            raise ValueError("returned reference origin mismatch")
        refs = []
        if (self._items[row["source"]].get("extraction") == "utf8_text"
                and Path(row["source"]).suffix.lower() in (".md", ".markdown")):
            for a, b, literal, destination in inline_links(passage["text"]):
                visit_edge((a, b, literal, destination))
                start = row["start_byte"] + len(passage["text"][:a].encode("utf-8"))
                raw = literal.encode("utf-8")
                refs.append(dict(origin_handle=passage["handle"], origin_source_id=passage["source_id"],
                    origin_pointer=deepcopy(passage["pointer"]),
                    link_coordinate_space="VERIFIED_SOURCE_LEAF_UTF8", link_start_byte=start,
                    link_end_byte=start + len(raw), link_sha256=digest(raw), link_text=literal,
                    **self._resolve(destination, row["source"])))
        passage["explicit_references"] = _References(refs, passage, _ISSUER)
        return refs

    def _recheck(self, edge):
        visit_edge(edge)
        row = self.rows[int(edge["origin_handle"][1:])]
        if (self._row_identity(row) != self._row_identity(edge["origin_pointer"])
                or getattr(self, "navigation_anchors", {}).get(int(edge["origin_handle"][1:])) != self._row_identity(row)):
            raise ValueError("changed or unreturned reference origin")
        self._verify_current_source(row)
        data = (self.directory / "leaves" / (row["leaf_sha256"] + ".txt")).read_bytes()
        a, b = row["start_byte"], row["end_byte"]
        x, y = edge["link_start_byte"], edge["link_end_byte"]
        if (not 0 <= a <= x < y <= b <= len(data)
                or digest(data) != row["leaf_sha256"] or digest(data[a:b]) != row["span_sha256"]
                or digest(data[x:y]) != edge["link_sha256"]
                or data[x:y] != edge["link_text"].encode("utf-8")):
            raise ValueError("reference origin leaf/span/link verification failed")
        self.reference_costs["reference_bytes_checked"] += y - x
        current = self._resolve(edge["destination"], row["source"])
        if current != {key: edge[key] for key in current}:
            raise ValueError("changed reference target identity")

    def search(self, query, k, arm="hybrid", novelty=False, **kwargs):
        if not kwargs.get("source_navigation", False):
            return super().search(query, k, arm, novelty, **kwargs)
        before_cost = dict(self.reference_costs)
        seen, anchors = set(self.seen), dict(getattr(self, "navigation_anchors", {}))
        directory, edge = self.directory, None
        anchor_lineage = []
        try:
            self._check_catalogue()
            self.directory = _ReadPath(directory, self.reference_costs)
            if query.startswith("source:"):
                if query in self._references:
                    edge, parent_lineage = self._references[query]
                    self.reference_costs["reference_hops_attempted"] += 1
                    self._recheck(edge)
                elif query not in {"source:" + name for name in self._returned}:
                    raise ValueError("unreturned source/reference address")
            elif re.fullmatch(r"(?:next|prev):S(0|[1-9][0-9]*)", query):
                anchor_lineage = deepcopy(self._lineages.get(query.split(":", 1)[1], []))
            response = super().search(query, k, arm, novelty, **kwargs)
            pending, lineages = {}, {}
            for passage in response["passages"]:
                lineage = (parent_lineage + [deepcopy(edge)]) if edge else (
                    anchor_lineage or deepcopy(self._lineages.get(passage["handle"], [])))
                if lineage:
                    passage["reference_lineage"] = lineage
                lineages[passage["handle"]] = lineage
                for ref in self._issue(passage):
                    if ref["resolution"] == "MANIFEST_RESOLVED_UNFETCHED":
                        pending.setdefault(ref["address"], (deepcopy(ref), deepcopy(lineage)))
            # Publication only after all target rows AND metadata pass.
            self._references.update({a: e for a, e in pending.items() if a not in self._references})
            self._returned.update(p["source_id"] for p in response["passages"])
            self._lineages.update(lineages)
            if edge:
                fetched = bool(response["passages"])
                self.reference_costs["reference_hops"] += int(fetched)
                # Empty returns retain the attempt/read charges, but neither
                # complete a hop nor extend the completed reference lineage.
                response["reference_follow"] = dict(edge=deepcopy(edge), fragment_resolved=False,
                    operation="PARTIAL_SOURCE_ROWS", status="COMPLETED" if fetched else "ATTEMPTED",
                    fetched=fetched, lineage=parent_lineage + [deepcopy(edge)] if fetched else deepcopy(parent_lineage))
            delta = {key: value - before_cost[key] for key, value in self.reference_costs.items()}
            response["reference_costs"] = delta
            references = [ref for p in response["passages"] for ref in p["explicit_references"]]
            response["reference_counts"] = dict(returned=len(references),
                resolved=sum(ref["resolution"] == "MANIFEST_RESOLVED_UNFETCHED" for ref in references),
                unresolved=sum(ref["resolution"] == "UNRESOLVED" for ref in references))
            response["source_version_bytes_read"] = delta["source_version_bytes_read"]
            response["navigation_verification_bytes_read"] = delta["leaf_bytes_read"]
            response["source_bytes_read"] = delta["source_version_bytes_read"] + delta["leaf_bytes_read"]
            self.last_reference_attempt = dict(
                status="ATTEMPTED" if edge and not response["passages"] else "RETURNED", **delta)
            return response
        except Exception:
            self.seen, self.navigation_anchors = seen, anchors
            self.last_reference_attempt = dict(status="REFUSED", **{
                key: value - before_cost[key] for key, value in self.reference_costs.items()})
            raise
        finally:
            self.directory = directory
