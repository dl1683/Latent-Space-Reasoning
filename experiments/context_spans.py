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
    if len(leaves) != 1 or len(leaves[0]["text"].encode()) <= 12000:
        return [leaves]
    whole = leaves[0]
    raw = whole["text"].encode()
    ranges = section_ranges(whole["text"])
    if not ranges:
        return [leaves]
    from experiments.latent_context import words
    terms = set(words(question))
    years = set(re.findall(r"\b20\d{2}\b", question))
    scored = []
    for a, b, kind, title in ranges:
        anchored = any(leaf_hash == whole["leaf_sha256"] and page == whole["page"]
                       and a <= start < end <= b for leaf_hash, page, start, end in anchors)
        if anchors and not anchored:
            continue
        body = raw[a:b].decode()
        seen = set(words(body))
        score = len(terms & seen) + 2 * len(terms & set(words(title)))
        if title.casefold().rstrip(":") in question.casefold():
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
            span = dict(whole, id=tool.corpus_id + ":U" + original.rsplit(":S", 1)[1] + f"_{a}_{b}",
                        origin_id=original, start_byte=a, end_byte=b, sha256=digest(raw[a:b]),
                        text=body, unit=kind, section_title=title,
                        boundary_basis="source heading to sibling heading; tables require notes footer",
                        completeness_scope="extracted unit only; referenced notes not expanded")
            scored.append((score, a, [span]))
    if anchors:
        selected = [s for _, _, unit in scored for s in unit]
        if any(not any(s["start_byte"] <= a < b <= s["end_byte"] for s in selected)
               for _, _, a, b in anchors):
            # An indexed chunk crossing sibling boundaries stays whole/atomic.
            return [leaves]
        return [unit for _, _, unit in sorted(scored, key=lambda x: x[1])]
    return [s for _, _, s in sorted(scored, key=lambda x: (-x[0], x[1]))] or [leaves]


def reopen_unit(tool, span_id):
    original = origin_id(tool, span_id)
    whole = leaf_span(tool, original)
    _, a, b = span_id.rsplit(":U", 1)[1].split("_")
    a, b = int(a), int(b)
    ranges = section_ranges(whole["text"])
    boundary = next((r for r in ranges if r[:2] == (a, b)), None)
    if boundary is None:
        raise ValueError("unknown structural unit boundary")
    raw = whole["text"].encode()[a:b]
    return dict(whole, id=span_id, start_byte=a, end_byte=b, text=raw.decode(),
                sha256=digest(raw), unit=boundary[2], section_title=boundary[3])


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
