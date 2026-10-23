import tempfile
from pathlib import Path

import numpy as np
import pytest

from experiments.memory.latent_frontend import LatentFrontend, expand, source_rows, title_graph


def fixture(tmp_path):
    ps = [
        dict(idx=0, title="Harbor", paragraph_text="The route leads to Mountain Station."),
        dict(idx=1, title="Mountain Station", paragraph_text="The beacon is turquoise."),
        dict(idx=2, title="Elsewhere", paragraph_text="The unrelated beacon is orange."),
        dict(idx=3, title="Registry", paragraph_text="The record number is 29."),
    ]
    x = np.asarray([[1, 0], [.6, .8], [-1, 0], [0, -1]], dtype=np.float32)
    return LatentFrontend.build(tmp_path/"map", ps, x, encoder_identity="fixture"), x


def test_route_graph_verify_and_costs(tmp_path):
    m, x = fixture(tmp_path)
    r = m.retrieve("harbor", x[0], k=2)
    assert [p["idx"] for p in r["evidence"]] == [0, 1]
    assert r["reference_edges_visited"] == 1
    assert r["exact_vector_bytes_touched"] > 0
    assert r["map_bytes_touched"] == m.map.payload_bytes
    assert r["source_bytes_touched"] == sum(len(p["text"].encode()) for p in r["evidence"])
    assert r["corpus_reencodes"] == r["reader_calls"] == 0
    assert r["completeness"] == r["semantic_absence"] == "UNKNOWN"
    assert all(p["verification"] == "BYTE_IDENTITY_ONLY" for p in r["evidence"])


def test_tamper_refused(tmp_path):
    m, x = fixture(tmp_path)
    p = next((tmp_path/"map"/"leaves").glob("*.txt"))
    # Select all sources so the altered leaf must be checked.
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source version/span changed"):
        m.retrieve("harbor", x[0], k=4)


def test_restart_and_no_reencode(tmp_path):
    m, x = fixture(tmp_path)
    a = m.retrieve("station", x[1], k=3, graph=False)
    b = LatentFrontend(tmp_path/"map").retrieve("station", x[1], k=3, graph=False)
    assert a["evidence"] == b["evidence"]
    assert b["selected_indices"] == list(np.argsort(-(x@x[1]), kind="stable")[:3])


@pytest.mark.parametrize("arm", ["latent", "dense", "hybrid", "bm25", "packing"])
def test_all_arms_verified(tmp_path, arm):
    m, x = fixture(tmp_path)
    r = m.retrieve("Harbor", x[0], k=3, arm=arm)
    assert len(r["evidence"]) == 3
    assert len(set(r["selected_indices"])) == 3


def test_no_labels_and_ambiguous_title_edges():
    with pytest.raises(ValueError):
        source_rows([dict(idx=0, title="a", paragraph_text="b", answer="CANARY")])
    ps = [dict(idx=0, title="Origin", paragraph_text="See Port."),
          dict(idx=1, title="Port", paragraph_text="One"),
          dict(idx=2, title="Port", paragraph_text="Two")]
    g = title_graph(ps)
    assert [e["target"] for e in g[0]] == [1, 2]
    assert expand([0, 1, 2], g, 3)[0] == [0, 1, 2]


def test_unicode_edge_byte_offsets():
    ps = [dict(idx=0, title="Origin", paragraph_text="é → Port"),
          dict(idx=1, title="Port", paragraph_text="One")]
    e = title_graph(ps)[0][0]
    assert ps[0]["paragraph_text"].encode()[e["start_byte"]:e["end_byte"]] == b"Port"


def test_wrong_vector_and_changed_manifest(tmp_path):
    m, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        m.retrieve("x", np.zeros(3, np.float32))
    (tmp_path/"map"/"graph.json").write_text("[]")
    with pytest.raises(ValueError, match="artifact changed"):
        LatentFrontend(tmp_path/"map")


def test_changed_vector_file_after_windows_mapping_release(tmp_path):
    m, _ = fixture(tmp_path)
    path = m.directory/"vectors.npy"
    # Windows correctly prevents truncating a live memory-mapped bank.
    m.bank._mmap.close()
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="artifact changed"):
        LatentFrontend(tmp_path/"map")

def test_title_graph_selected_origins_preserve_eager_edges_and_unicode():
    ps = [dict(title=t, paragraph_text=body) for t, body in (
        ('Origin', 'é → Port Portside port. PORT'),
        ('Port', 'Origin Port'), ('Port', 'PORT'), ('port', 'Port'),
        ('Port Port', 'Port Port'), ('x', 'Origin'))]
    eager = title_graph(ps)
    assert title_graph(ps, origins=[3, 0, 1]) == [eager[3], eager[0], eager[1]]
    for i, edges in enumerate(eager):
        for edge in edges:
            literal = ps[i]['paragraph_text'].encode()[edge['start_byte']:edge['end_byte']].decode()
            assert literal.lower() == ps[edge['target']]['title'].lower()
