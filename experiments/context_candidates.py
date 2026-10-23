"""Shared source-unit preparation for one-shot, PPR and one-hop expansion."""
from experiments.context_spans import source_units
from experiments.context_costs import stage


@stage("verification_reads")
def prepare_units(memory, navigation, proposal, question, edge=None):
    """Return atomic span groups, or None when charged access could not finish."""
    nodes = proposal["parents"] if isinstance(proposal, dict) else [proposal]
    if edge and navigation:
        # A multi-source pair is atomic too. Refuse a permanently unaffordable
        # pair rather than issuing a cursor that reopens its first parent forever.
        sources = {}
        for node in nodes:
            row = memory.tool.memory.rows[memory.tool._index(node)]
            sources.setdefault(row["source"], node)
        needed = sum(navigation.source_read_size(node) for node in sources.values())
        if needed > navigation.access.limits.verification_bytes:
            navigation.access.stop_reason = "atomic_source_exceeds_verification_budget"
            navigation.refused_reads += 1
            return None
    parents = []
    source_cache = {}
    anchors = {}
    if edge:
        for number, node in enumerate(nodes):
            row = memory.tool.memory.rows[memory.tool._index(node)]
            a, b = row["start_byte"], row["end_byte"]
            if number == 0 and edge.witness_span:
                a, b = edge.witness_span.start, edge.witness_span.end
            anchors.setdefault(row["source"], []).append((row["leaf_sha256"], row["page"], a, b))
    for number, node in enumerate(nodes):
        row = memory.tool.memory.rows[memory.tool._index(node)]
        if row["source"] not in source_cache:
            options = dict(anchors=tuple(anchors[row["source"]])) if edge else {}
            source_cache[row["source"]] = (navigation.source_units(node, question, **options) if navigation
                                          else source_units(memory.tool, node, question, **options))
        units = source_cache[row["source"]]
        if units is None:
            return None
        if edge:
            a, b = row["start_byte"], row["end_byte"]
            if number == 0 and edge.witness_span:
                a, b = edge.witness_span.start, edge.witness_span.end
            # A structural section can earn a parent only if it actually contains
            # the endpoint/witness. Never attach an unrelated top-ranked section.
            chosen = next((unit for unit in units if any(
                s["source_path"] == row["source"] and s["source_version"] == row["source_sha256"]
                and s["leaf_sha256"] == row["leaf_sha256"] and s["page"] == row["page"]
                and s["start_byte"] <= a < b <= s["end_byte"] for s in unit)), None)
            if chosen is None:
                raise ValueError("missing bridge parent or witness")
            parents.extend(chosen)
        else:
            return units
    return [list({s["id"]: s for s in parents}.values())]


def proposal_id(proposal):
    if isinstance(proposal, dict):
        return proposal["parents"][0]
    if isinstance(proposal, list):
        return proposal[0]
    return proposal
