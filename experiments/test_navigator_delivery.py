"""Navigator delivery SLICE: anchored units and graceful access/output stops."""
import json

import numpy as np
import pytest

from experiments.latent_context import ContextMemory
from experiments.memory.evidence_map import EvidenceMap
from experiments.memory.ppr_context import PPRNavigation
from experiments.test_expand_context import (
    bridge_corpus, bridges, invoke, literal_edge, spans,
)
from experiments.test_latent_mcp import put, sha


def repin(memory, source, index, changes):
    """Replace authored fixture sources, keeping real byte verification enabled."""
    rows = json.loads((index / "rows.json").read_bytes())
    manifest = json.loads((index / "manifest.json").read_bytes())
    for number, (text, chunk) in changes.items():
        raw, excerpt = text.encode(), chunk.encode()
        path = source / f"{number:02}.txt"
        path.write_bytes(raw)
        (index / "leaves" / (sha(raw) + ".txt")).write_bytes(raw)
        start = raw.index(excerpt)
        rows[number].update(source_sha256=sha(raw), leaf_sha256=sha(raw),
                            start_byte=start, end_byte=start + len(excerpt),
                            span_sha256=sha(excerpt), text=chunk)
        manifest["files"][number].update(sha256=sha(raw), bytes=len(raw))
    put(index / "rows.json", rows)
    np.save(index / "vectors.npy", np.ones((len(rows), 3), dtype=np.float32))
    manifest.update(rows_sha256=sha((index / "rows.json").read_bytes()),
                    vector_sha256=sha((index / "vectors.npy").read_bytes()))
    put(index / "manifest.json", manifest)
    from pathlib import Path
    config = Path(memory.config)
    settings = json.loads(config.read_bytes())
    settings["manifest_sha256"] = sha((index / "manifest.json").read_bytes())
    put(config, settings)
    result = ContextMemory(config)
    result._evidence_map = EvidenceMap(result.tool)
    return result


@pytest.mark.parametrize("table", [False, True])
def test_zero_overlap_anchored_target_survives_question_filter(bridge_corpus, table):
    memory, source, index = bridge_corpus
    if table:
        target = (
            "CONSOLIDATED STATEMENTS OF OPERATIONS\n(in thousands)\n"
            "2025     2024\nNebula Port\nHarbor dues       29       21\n\n"
            "CONSOLIDATED STATEMENTS OF OPERATIONS\n(continued)\n"
            "Final row        42       32\n\n"
            "See accompanying notes to consolidated financial statements.\n\n"
            "Footnotes\n\n(1) Provisional amounts exclude tax.\n\n"
        )
        chunk = "Nebula Port\nHarbor dues       29       21"
    else:
        target = "Nebula Port\n\nMira administers vessels.\n\n"
        chunk = target.rstrip()
    preface = "Preface\n\nBeacon registry catalog.\n\n"
    text = preface + "Background\n\n" + "Unrelated prose. " * 1000 + "\n\n" + target
    memory = repin(memory, source, index, {11: (text, chunk)})
    pack = invoke(memory)
    assert bridges(pack), pack["unresolved"]
    destination = next(s for s in spans(pack) if s["file"] == "11.txt")
    assert destination["text"] == target
    assert destination["unit"] == ("COMPLETE_TABLE" if table else "COMPLETE_SECTION")
    assert not {"beacon", "registry"} & set(destination["text"].lower().split())
    assert destination["byte_verified"] and sha(target.encode()) == destination["sha256"]
    assert set(bridges(pack)[0]["parents"]) <= {s["id"] for s in spans(pack)}


@pytest.mark.parametrize("reason", ["navigation_time_cap", "verification_byte_cap"])
def test_stopped_access_never_attempts_uncached_source_read(bridge_corpus, monkeypatch, reason):
    memory, _, _ = bridge_corpus
    original = PPRNavigation.source_units
    read = PPRNavigation.source_spans
    endpoint = literal_edge(memory).to_id

    def stop_target(nav, node, question, **kwargs):
        if node == endpoint:
            nav.access.stop_reason = reason
            if reason == "navigation_time_cap":
                nav.access.clock = lambda: nav.access.started + nav.access.limits.seconds
            return None
        return original(nav, node, question, **kwargs)

    def guarded(nav, ids):
        assert nav.access.stop_reason != reason, "attempted read after access stop"
        return read(nav, ids)

    monkeypatch.setattr(PPRNavigation, "source_units", stop_target)
    monkeypatch.setattr(PPRNavigation, "source_spans", guarded)
    pack = invoke(memory)
    assert spans(pack) and spans(pack)[0]["file"] == "00.txt"
    assert not bridges(pack) and pack["continuation"]
    assert pack["costs"]["expand"]["access"]["source_opens"] == 1


def test_zero_overlap_witness_parent_is_reopened_before_ranking(bridge_corpus, monkeypatch):
    memory, source, index = bridge_corpus
    parent = "Departure Ledger\n\nVessels sail to Nebula Port.\n\n"
    text = ("Preface\n\nBeacon registry catalog.\n\nBackground\n\n" +
            "Unrelated prose. " * 1000 + "\n\n" + parent)
    memory = repin(memory, source, index, {0: (text, parent.rstrip())})
    monkeypatch.setattr("experiments.memory.expand_context.query_seeds",
                        lambda *a, **k: {memory.tool.corpus_id + ":S0": 1.0})
    pack = invoke(memory)
    assert bridges(pack), pack["unresolved"]
    origin = next(s for s in spans(pack) if s["id"] == bridges(pack)[0]["parents"][0])
    assert origin["text"] == parent and origin["sha256"] == sha(parent.encode())
    assert bridges(pack)[0]["witness"]["text"] == "Nebula Port"


def test_oversized_seed_does_not_displace_affordable_bridge_on_first_page(bridge_corpus, monkeypatch):
    memory, source, index = bridge_corpus
    text = "Oversized Catalog\n\n" + "Unrelated prose. " * 20000
    memory = repin(memory, source, index, {1: (text, "Unrelated prose.")})
    import experiments.memory.expand_context as expand
    propose = expand.ExpandNavigation.propose

    def oversized_first(nav, question, progress=None):
        candidates, trace, duplicates = propose(nav, question, progress)
        return [nav.candidate(memory.tool.corpus_id + ":S1"), *candidates], trace, duplicates

    monkeypatch.setattr(expand.ExpandNavigation, "propose", oversized_first)
    pack = invoke(memory)
    assert bridges(pack), "oversized seed displaced the affordable pair"
    assert {"00.txt", "11.txt"} <= {s["file"] for s in spans(pack)}
    assert pack["omission_ledger"]["total"] > 0
    assert memory.meter.fits(pack, pack["costs"]["output_limits"])
