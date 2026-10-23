"""Check source identity/literal binding, never assert a semantic evidence join."""
from pathlib import Path

from experiments.memory.explicit_reference_memory import inline_links
from experiments.context_costs import stage


def compatible_bindings(*bindings):
    """Reject incompatible exact bindings. Semantic compatibility is not computed."""
    result = {}
    for binding in bindings:
        for key, value in binding:
            if key in result and result[key] != value:
                raise ValueError("incompatible bridge binding: " + key)
            result[key] = value
    return result


@stage("verification_reads")
def bridge_group(navigation, edge, spans):
    """Require BOTH exact parents and the literal witness in this output page."""
    mapped, tool = navigation.map, navigation.tool
    if edge.type not in ("explicit_reference", "scoped_association", "adjacency"):
        raise ValueError("unsupported bridge type")
    origin, target = mapped.spans[edge.from_id], mapped.spans[edge.to_id]
    rows = tool.memory.rows
    left, right = rows[origin.vector_row], rows[target.vector_row]
    expected = (("origin_version", origin.source_version), ("target_version", target.source_version))
    constraints = compatible_bindings(edge.binding_constraints, expected)

    def parent(record, a, b):
        return next((s for s in spans if s["source_path"] == record["source"]
                     and s["source_version"] == record["source_sha256"]
                     and s["leaf_sha256"] == record["leaf_sha256"] and s["page"] == record["page"]
                     and s["start_byte"] <= a < b <= s["end_byte"]), None)

    witness = edge.witness_span
    a, b = (witness.start, witness.end) if witness else (origin.start, origin.end)
    origin_span = parent(left, a, b)
    target_span = parent(right, target.start, target.end)
    if origin_span is None or target_span is None:
        raise ValueError("missing bridge parent or witness")
    if witness:
        if (witness.source_version != origin.source_version or witness.leaf_hash != origin.leaf_hash
                or witness.scope != origin.scope or witness.coordinate_space != origin.coordinate_space
                or not origin.start <= a < b <= origin.end):
            raise ValueError("wrong bridge witness binding")
        raw = origin_span["text"].encode()[a-origin_span["start_byte"]:b-origin_span["start_byte"]]
        if not navigation._hash_matches(raw, witness.span_hash):
            raise ValueError("corrupt bridge witness")
        literal = raw.decode()
        if edge.type == "scoped_association":
            scope = str(Path(left["source"]).parent)
            compatible_bindings(constraints.items(), (("directory", scope),))
            title = right["text"].splitlines()[0].strip("# ")[:240]
            if Path(right["source"]).parent != Path(left["source"]).parent or literal.casefold() != title.casefold():
                raise ValueError("wrong title binding")
        else:
            links = list(inline_links(literal))
            if len(links) != 1 or links[0][2] != literal:
                raise ValueError("wrong reference witness")
            resolved = tool.memory._resolve(links[0][3], left["source"],
                source_rows=mapped.source_rows, check=navigation.access.check,
                costs=navigation.access.costs)
            if (resolved["resolution"] != "MANIFEST_RESOLVED_UNFETCHED"
                    or resolved.get("address") != "source:" + tool.memory.source_ids[right["source"]]):
                raise ValueError("wrong reference binding")
        witness_record = dict(span_id=origin_span["id"], start_byte=a, end_byte=b,
                              sha256=witness.span_hash, text=literal, byte_verified=True)
    else:
        if origin.source_version != target.source_version:
            raise ValueError("incompatible adjacency versions")
        compatible_bindings(constraints.items(), (("source_version", origin.source_version),))
        witness_record = dict(basis="pinned source row adjacency", byte_verified=True)
    return dict(topic="Provisional source bridge", kind="PROVISIONAL_SOURCE_BRIDGE",
                parents=[origin_span["id"], target_span["id"]], edge_type=edge.type,
                witness=witness_record, binding_constraints=dict(edge.binding_constraints),
                semantic_support="READER_MUST_ASSESS", join_status="not computed",
                obligations=["semantic relevance/entailment and any missing premises remain unresolved"],
                spans=[])
