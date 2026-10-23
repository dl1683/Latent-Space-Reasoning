import hashlib
import numpy as np
import pytest

from experiments.memory.compressed_routing_map import (
    BM25, PackedMap, chunk_text, fetch_leaf, normalize, pack_rows, projection,
    construction_binding, load_pointers, reciprocal_rank_fusion,
    require_smoke_binding, save_pointers,
    support_metrics, unique_documents, unpack_rows)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_physical_bit_packing(bits):
    x = np.random.default_rng(2).normal(size=(7, 33)).astype(np.float32)
    codes, scales = pack_rows(x, bits)
    restored = unpack_rows(codes, scales, 33, bits)
    assert codes.nbytes == 7 * ((33 * bits + 7) // 8)
    assert scales.nbytes == 7 * 4
    assert np.max(np.abs(x - restored) / scales[:, None]) <= 1 / (2**bits - 1) + 1e-6
    again = pack_rows(restored, bits)
    assert np.array_equal(again[0], codes)


def test_quantization_validation_and_zero():
    with pytest.raises(ValueError):
        pack_rows(np.array([[np.nan]]), 4)
    for bits in (1, 2, 4, 8):
        x, scale = pack_rows(np.zeros((2, 8)), bits)
        assert np.max(np.abs(unpack_rows(x, scale, 8, bits))) <= 1e-12


def test_fetch_refuses_source_mutation(tmp_path):
    raw = "Exact clause 12.\n".encode()
    p = tmp_path / "leaf.txt"
    p.write_bytes(raw)
    chunk = dict(file="leaf.txt", source_sha256=hashlib.sha256(raw).hexdigest(),
                 start=6, end=12)
    assert fetch_leaf(tmp_path, chunk) == ("clause", len(raw))
    p.write_text("New source", encoding="utf8")
    with pytest.raises(ValueError, match="source changed"):
        fetch_leaf(tmp_path, chunk)


def test_support_metrics_not_any_hit():
    chunks = [dict(document="a"), dict(document="a"), dict(document="b")]
    assert unique_documents([0, 1, 2], chunks, 2) == [0, 2]
    assert support_metrics([0], chunks, ["a", "b"]) == dict(recall=.5, complete=False)
    assert support_metrics([0, 2], chunks, ["a", "b"])["complete"]


def test_lexical_fusion_and_projection_are_fixed():
    bm = BM25(["orchard apples", "stellar telescope"])
    assert bm.score("apple orchard").argmax() == 0
    assert np.array_equal(projection(32), projection(32))
    assert projection(384) is None
    assert np.isfinite(normalize(np.zeros((2, 4)))).all()
    fusion = reciprocal_rank_fusion(np.array([1., 0]), np.array([0., 1]))
    assert fusion[0] == fusion[1]


def test_windows_cover_all_characters():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return dict(offset_mapping=[(i, i + 1) for i in range(len(text))])
    text = "x" * 103
    spans = chunk_text(text, Tokenizer(), width=20, stride=16)
    covered = set()
    for a, b in spans:
        covered.update(range(a, b))
    assert covered == set(range(len(text)))


def test_cold_map_never_needs_source_bank(tmp_path):
    x = normalize(np.random.default_rng(2).normal(size=(3, 384)).astype(np.float32))
    codes, scale = pack_rows(x, 8)
    np.savez(tmp_path / "d384b8.npz", codes=codes, scale=scale)
    chunks = [dict(document=str(i)) for i in range(3)]
    tool = PackedMap(tmp_path, "d384b8", chunks)
    assert tool.search(x[1], k=1)["selected_chunks"] == [1]
    assert tool.search(x[1])["status"] == "UNKNOWN_UNTIL_SOURCE_VERIFICATION"


def test_pointer_hashes_are_shared_and_round_trip(tmp_path):
    key = "ab" * 32
    chunks = [dict(document=key, source_sha256=key, file=f"leaves/{key}.txt",
                   start=a, end=a + 10) for a in (0, 8)]
    path = tmp_path / "ptr.npz"
    save_pointers(path, chunks)
    assert load_pointers(path) == chunks
    with np.load(path) as state:
        assert state["source_hashes"].nbytes == 32
        assert state["spans"].nbytes == 24


def test_smoke_rejects_unbound_or_changed_inputs(tmp_path):
    import json
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.txt"):
        (model / name).write_bytes(b"fixture")
    cfg = dict(model=str(model))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    receipt = tmp_path / "smoke.json"
    receipt.write_text(json.dumps(dict(passed=True)))
    with pytest.raises(RuntimeError, match="does not qualify"):
        require_smoke_binding(tmp_path, cfg)
    receipt.write_text(json.dumps(dict(passed=True,
        binding=construction_binding(tmp_path, cfg))))
    require_smoke_binding(tmp_path, cfg)
    (model / "vocab.txt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="does not qualify"):
        require_smoke_binding(tmp_path, cfg)
