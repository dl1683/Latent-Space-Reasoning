"""Source-defined sections, with conservative whole-source fallback.

Boundaries are structural, never query-sized windows. PDF furniture stays in the
quotation, so rows crossing projected pages and repeated headers stay together.
Completeness describes the extracted section, not all referenced accounting notes.
"""
import re

from experiments.dogfood import digest, plain_file, ram_guard
from experiments.context_costs import read_payload, stage


@stage("verification_reads")
def leaf_span(tool, span_id):
    ram_guard()
    original = tool._open(span_id)[0]
    row = original["pointer"]
    raw = read_payload(plain_file(tool.memory.directory / "leaves" / (
        row["leaf_sha256"] + ".txt")), "leaf")
    tool.memory.reference_costs["leaf_bytes_read"] += len(raw)
    if digest(raw) != row["leaf_sha256"]:
        raise ValueError("changed source leaf")
    span = tool._project(original)
    span.update(id=tool.corpus_id + ":L" + original["handle"][1:],
                origin_id=span_id, start_byte=0, end_byte=len(raw),
                sha256=digest(raw), text=raw.decode("utf-8"),
                unit="COMPLETE_EXTRACTED_LEAF", navigation=None)
    return span


def origin_id(tool, span_id):
    match = re.fullmatch(re.escape(tool.corpus_id) + r":U(\d+)_(\d+)_(\d+)", span_id)
    if match:
        original = tool.corpus_id + ":S" + match[1]
        tool._index(original)
        return original
    prefix = tool.corpus_id + ":L"
    if isinstance(span_id, str) and span_id.startswith(prefix):
        original = tool.corpus_id + ":S" + span_id[len(prefix):]
        tool._index(original)
        return original
    tool._index(span_id)
    return span_id


def source_leaf_ids(tool, span_id):
    """Keep every indexed page of a source together, including cross-page tables."""
    row = tool.memory.rows[tool._index(span_id)]
    selected = {}
    for i in tool.memory._source_order(row["source"], row["source_sha256"]):
        other = tool.memory.rows[i]
        selected.setdefault((other["page"], other["leaf_sha256"]), tool.corpus_id + f":S{i}")
    return list(selected.values())


def section_ranges(text):
    """Recognize sibling prose headings and explicitly closed financial statements.

    A table ends only at its accompanying-notes footer (including following local
    footnotes), not a page break or another matched row. Other numeric tables are
    deliberately NOT split using the prose-heading heuristic.
    """
    lines = text.splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line.encode("utf-8"))
    headings = []
    for i, line in enumerate(lines):
        label = line.strip()
        is_table = bool(re.fullmatch(r"CONSOLIDATED (?:STATEMENTS OF .+|BALANCE SHEETS)", label))
        if not label or len(label) > 100 or (i and lines[i - 1].strip() and not is_table):
            continue
        if is_table:
            headings.append((i, "table", label))
        elif (re.fullmatch(r"[A-Z][A-Z -]{8,}", label)
              or (len(line) - len(line.lstrip()) <= 4
                  and re.fullmatch(r"(?:[A-Z][A-Za-z’'-]*|and|of|the|to|&|\d{4})"
                                   r"(?: (?:[A-Z][A-Za-z’'-]*|and|of|the|to|&|\d{4}))*:?", label))):
            # PDF furniture and table labels are not sibling section headings.
            if label not in {"Investor Relations", "PDF Version", "News", "DATADOG, INC."}:
                headings.append((i, "section", label))
    # Table interiors can contain uppercase subheads and continuation titles.
    # They are never independent boundaries.
    table_ends = {}
    for i, kind, title in headings:
        if kind == "table":
            footer = next((j for j in range(i + 1, len(lines))
                           if re.search(r"See accompanying notes to .*financial statements", lines[j], re.I)), None)
            if footer is not None:
                table_ends[i] = footer
    headings = [h for h in headings if not any(a < h[0] <= b for a, b in table_ends.items())]
    # A short, local Notes/Footnotes heading closes over the table rather than
    # becoming an independently rankable section. Full accounting-note chapters
    # ("Notes to Consolidated Financial Statements") remain sibling sections.
    local_notes = re.compile(r"(?:Notes?|Footnotes?)(?: to (?:the )?(?:table|statement))?:?", re.I)
    attached = []
    table_open = False
    for heading in headings:
        i, kind, title = heading
        if table_open and kind == "section" and local_notes.fullmatch(title):
            continue
        attached.append(heading)
        table_open = kind == "table" and i in table_ends
    headings = attached
    result = []
    for n, (i, kind, title) in enumerate(headings):
        end = headings[n + 1][0] if n + 1 < len(headings) else len(lines)
        if kind == "table":
            footer = table_ends.get(i)
            if footer is None:
                continue
            # Keep footer, local footnotes and PDF furniture through next heading.
            start = i - 1 if i and lines[i - 1].strip() == "DATADOG, INC." else i
        else:
            start = i
            body = "".join(lines[i + 1:end])
            # Unknown numeric tables may extend beyond a guessed heading. Withhold
            # rather than presenting their fragment as a complete prose section.
            if re.search(r"\S[ \t]{3,}[$(]?\d[\d,.]*[ \t]{2,}[$(]?\d", body):
                continue
        result.append((offsets[start], offsets[end] if end < len(lines) else total,
                       "COMPLETE_TABLE" if kind == "table" else "COMPLETE_SECTION", title))
    return result


_ATX_HEADING = re.compile(r"^#{1,6}(?:[ \t]+(.*)|$)")
_INDENTED_HEADING = re.compile(r"^ {1,3}#{1,6}(?:[ \t]|$)")
_SETEXT_LINE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_MARKDOWN_CONTAINER = re.compile(r"^(?: {4}|\t| {0,3}(?:>|[-+*][ \t]+|\d+[.)][ \t]+))")
_STRUCTURAL_SPLIT_BYTES = 12000
_MARKDOWN_FENCE = re.compile(r"^ {0,3}(\x60{3,}|~{3,})(.*)$")
_RAW_HTML = re.compile(r"^ {0,3}<(?:!--|[!?]|/?[A-Za-z])")
_TITLED_PROSE = re.compile(r"^\*\*([^*]+[.?!:])\*\*[ \t]+\S.*$")
_TABLE_LINE = re.compile(
    r"(?m)^[ \t]*(?:\||:?-+:?[ \t]*\|)"
    r"|\S[ \t]{3,}[$(]?\d[\d,.]*[ \t]{2,}[$(]?\d"
)


def _markdown_ranges(text):
    headings, paragraphs, offset, fence = [], [], 0, None
    legacy_ambiguous = False
    lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$", text)
    if lines and lines[0].lstrip("\ufeff").strip() == "---":
        return None
    for i, line in enumerate(lines):
        content = line.rstrip("\r\n")
        if i == 0:
            content = content.removeprefix("\ufeff")
        fence_match = _MARKDOWN_FENCE.fullmatch(content)
        if fence:
            if (fence_match and fence_match[1][0] == fence[0]
                    and len(fence_match[1]) >= len(fence) and not fence_match[2].strip()):
                fence = None
        elif fence_match:
            if fence_match[1][0] == "`" and "`" in fence_match[2]:
                return None
            legacy_ambiguous = True
            fence = fence_match[1]
        elif ("<!--" in content or _RAW_HTML.match(content) or _INDENTED_HEADING.match(content)
              or (_SETEXT_LINE.fullmatch(content) and i and lines[i - 1].strip())):
            return None
        elif match := _ATX_HEADING.fullmatch(content):
            title = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", match[1] or "").strip()
            headings.append((offset, title))
        elif ((i == 0 or not lines[i - 1].strip())
              and (i + 1 == len(lines) or not lines[i + 1].strip())
              and (match := _TITLED_PROSE.fullmatch(content))):
            paragraphs.append((offset, offset + len(line.encode("utf-8")), "MARKDOWN_TITLED_PROSE", match[1]))
        elif _MARKDOWN_CONTAINER.match(content):
            legacy_ambiguous = True
        offset += len(line.encode("utf-8"))
    if fence or (legacy_ambiguous and not headings):
        return None
    blocks = [(start, headings[i + 1][0] if i + 1 < len(headings) else offset,
               "MARKDOWN_HEADING_BLOCK", title)
              for i, (start, title) in enumerate(headings)]
    return blocks, paragraphs


def _source_ranges(leaf, *, titled_prose=False):
    if leaf["file"].casefold().endswith((".md", ".markdown")):
        found = _markdown_ranges(leaf["text"])
        if found is None:
            return []
        blocks, paragraphs = found
        if blocks:
            if not titled_prose:
                return blocks
            raw = leaf["text"].encode("utf-8")
            large_prose = [(a, b) for a, b, _, _ in blocks
                           if b - a > _STRUCTURAL_SPLIT_BYTES and not _TABLE_LINE.search(raw[a:b].decode("utf-8"))]
            return blocks + [r for r in paragraphs if any(a <= r[0] < r[1] <= b for a, b in large_prose)]
    return section_ranges(leaf["text"])


def _verified_unit(leaf, span_id, boundary):
    a, b, kind, title = boundary
    raw = leaf["text"].encode("utf-8")[a:b]
    if kind == "MARKDOWN_HEADING_BLOCK":
        basis = "explicit Markdown heading to next heading outside fenced blocks"
        scope = "heading body only; nested headings and linked content excluded"
    elif kind == "MARKDOWN_TITLED_PROSE":
        basis = "source paragraph with a bold lead title and blank-line boundaries"
        scope = "paragraph only; surrounding section and linked content excluded"
    else:
        basis = "source heading to sibling heading; tables require notes footer"
        scope = "extracted unit only; referenced notes not expanded"
    return dict(leaf, id=span_id, start_byte=a, end_byte=b, sha256=digest(raw), text=raw.decode("utf-8"),
                unit=kind, section_title=title, boundary_basis=basis, completeness_scope=scope)


def reopen_verified_unit(leaf, span_id, *, leaf_count):
    """Reopen a structural unit on an already verified leaf."""
    if leaf_count != 1 or leaf["end_byte"] <= _STRUCTURAL_SPLIT_BYTES:
        raise ValueError("unknown structural unit boundary")
    _, a, b = span_id.rsplit(":U", 1)[1].split("_")
    bounds = int(a), int(b)
    boundary = next((r for r in _source_ranges(leaf, titled_prose=True) if r[:2] == bounds), None)
    if boundary is None:
        raise ValueError("unknown structural unit boundary")
    return _verified_unit(leaf, span_id, boundary)


@stage("verification_reads")
def source_units(tool, span_id, question, *, anchors=()):
    """Verify a source first; choose only independently bounded units inside it."""
    leaves = [leaf_span(tool, i) for i in source_leaf_ids(tool, span_id)]
    return units_from_leaves(tool, span_id, question, leaves, anchors=anchors)


def units_from_leaves(tool, span_id, question, leaves, *, anchors=()):
    """One structural policy for every strategy, after its charged byte reader."""
    # Anchors are exact leaf/page/byte coordinates, not relevance hints. Reject
    # invalid coordinates; never substitute an unrelated query-ranked section.
    for leaf_hash, page, a, b in anchors:
        if not any(s["leaf_sha256"] == leaf_hash and s["page"] == page
                   and s["start_byte"] <= a < b <= s["end_byte"] for s in leaves):
            raise ValueError("missing bridge parent or witness")
    # Unmapped multi-leaf boundaries stay atomic. Small sources remain intact.
    if len(leaves) != 1 or leaves[0]["end_byte"] <= _STRUCTURAL_SPLIT_BYTES:
        return [leaves]
    whole = leaves[0]
    raw = whole["text"].encode()
    ranges = _source_ranges(whole, titled_prose=not anchors)
    if not ranges:
        return [leaves]
    from experiments.latent_context import words
    terms = set(words(question))
    years = set(re.findall(r"\b20\d{2}\b", question))
    scored = []
    seed = tool.memory.rows[tool._index(span_id)]
    for a, b, kind, title in ranges:
        anchored = any(leaf_hash == whole["leaf_sha256"] and page == whole["page"]
                       and a <= start < end <= b for leaf_hash, page, start, end in anchors)
        if anchors and not anchored:
            continue
        body = raw[a:b].decode()
        seen = set(words(body))
        score = len(terms & seen) + 2 * len(terms & set(words(title)))
        if title and title.casefold().rstrip(":") in question.casefold():
            score += 8
        score += len(years & set(re.findall(r"\b20\d{2}\b", body)))
        if kind == "COMPLETE_TABLE" and terms & {"table", "tables"}:
            score += 8
            if "net income" in question.casefold() and "OPERATIONS" in title:
                score += 12
            if "cash flow" in question.casefold() and "CASH FLOWS" in title:
                score += 12
        if "risk" in question.casefold() and "RISK FACTORS SUMMARY" in title:
            score += 16
        if anchored or (score and terms & seen):
            original = whole["origin_id"]
            unit_id = tool.corpus_id + ":U" + original.rsplit(":S", 1)[1] + f"_{a}_{b}"
            span = _verified_unit(whole, unit_id, (a, b, kind, title))
            overlap = (seed["leaf_sha256"] == whole["leaf_sha256"]
                       and a < seed["end_byte"] and seed["start_byte"] < b)
            priority = int(kind == "MARKDOWN_TITLED_PROSE") * (2 if overlap else 1)
            scored.append((priority, score, a, [span]))
    if anchors:
        selected = [s for _, _, _, unit in scored for s in unit]
        if any(not any(s["start_byte"] <= a < b <= s["end_byte"] for s in selected)
               for _, _, a, b in anchors):
            # An indexed chunk crossing sibling boundaries stays whole/atomic.
            return [leaves]
        return [unit for _, _, _, unit in sorted(scored, key=lambda x: x[2])]
    ranked = sorted(scored, key=lambda x: (-x[0], -x[1], x[2]))
    return _choose_units(ranked, question) or [leaves]


def _choose_units(ranked, question):
    from experiments.latent_context import facets_for, words

    prose = [unit for priority, _, _, unit in ranked if priority]
    other = [unit for priority, _, _, unit in ranked if not priority]
    if not prose:
        return other
    tagged = [(unit, set(words(unit[0]["text"])), set(words(unit[0]["section_title"]))) for unit in prose]
    picked = [prose[0]]
    for facet in facets_for(question):
        terms = set(words(facet))
        unit, covered, _ = max(tagged, key=lambda item: len(terms & item[1]) + 2 * len(terms & item[2]))
        if terms & covered and unit not in picked:
            picked.append(unit)

    missing = set(words(question)) - set().union(*(set(words(unit[0]["text"])) for unit in picked))
    while missing:
        unit, terms, _ = max(tagged, key=lambda item: len(missing & item[1]) + 2 * len(missing & item[2]))
        gain = missing & terms
        if not gain:
            break
        picked.append(unit)
        missing -= gain
    return picked + other


def reopen_unit(tool, span_id):
    original = origin_id(tool, span_id)
    return reopen_verified_unit(leaf_span(tool, original), span_id,
                                leaf_count=len(source_leaf_ids(tool, original)))


def omission(memory, span_id, reason, dimensions, **extra):
    """Catalog identity is verified metadata, never a quotation or support claim."""
    tool = memory.tool
    row = tool.memory.rows[tool._index(origin_id(tool, span_id))]
    path = tool.memory.source_ids[row["source"]]
    if tool.file_pins.get(row["source"]) != row["source_sha256"]:
        raise ValueError("catalog identity mismatch")
    quote = lambda s: "'" + str(s).replace("'", "''") + "'"
    fallback = ("python -B -m experiments.latent_context_app read --corpus " +
                quote(memory.config) + " --file " + quote(path) + " --start-byte 0")
    return dict(span_id=span_id, reason=reason, blocked_dimensions=dimensions,
                catalog_identity=dict(role="CATALOG_METADATA_NOT_EVIDENCE", file=path,
                    source_version=row["source_sha256"], snapshot=tool.snapshot),
                exact_reader=fallback, fallback_scope="bounded raw bytes, NOT a complete table",
                **extra)
