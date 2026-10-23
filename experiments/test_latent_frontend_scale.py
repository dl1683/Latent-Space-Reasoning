import json

import numpy as np
import pytest

from experiments import latent_frontend_scale as scale


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    class Encoder:
        max_seq_length = 256
        def tokenizer(self, text, **kwargs):
            return {"input_ids": [1]}
        def get_sentence_embedding_dimension(self):
            return 2
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 2), dtype=np.float32)
    monkeypatch.setattr(scale, "encoder_at", lambda path: Encoder())
    monkeypatch.setattr(scale, "guard", lambda: None)
    source = tmp_path / "source.jsonl"
    source.write_text("source fixture", encoding="utf8")
    config = tmp_path / "config.json"
    scale.save(config, dict(encoder="fixture", scale_tokenizer="fixture",
                           scale_sources=[str(source)]))
    root = tmp_path / "scale"
    root.mkdir()
    raw = b"source text"
    (root / "leaves.bin").write_bytes(raw * 65)
    pointers = [dict(start=i*len(raw), end=(i+1)*len(raw),
                     span_sha256=scale.sha(raw), source_sha256="fixture")
                for i in range(65)]
    scale.save(root / "pointers.json", pointers)
    scale.save(root / "token_counts.json", [2] * len(pointers))
    # An earlier uncheckpointed bank must never be overwritten or treated as valid.
    (root / "vectors.npy").write_bytes(b"preserve partial original")
    return config, root


def test_checkpoint_resumes_after_memory_refusal(prepared, monkeypatch):
    config, root = prepared
    checks = iter([None, MemoryError("authored floor")])
    def guard():
        result = next(checks)
        if result:
            raise result
    monkeypatch.setattr(scale, "guard", guard)
    with pytest.raises(MemoryError):
        scale.resume_build(config, root)
    assert scale.read(root / "ENCODING.json")["completed"] == 32
    monkeypatch.setattr(scale, "guard", lambda: None)
    scale.resume_build(config, root)
    assert scale.read(root / "ENCODING.json")["completed"] == 65
    build = scale.read(root / "BUILD.json")
    assert build["actual_phi_chunk_tokens"] == 130
    assert np.load(root / build["vector_file"]).shape == (65, 2)
    assert (root / "vectors.npy").read_bytes() == b"preserve partial original"


def test_checkpoint_rejects_changed_inputs(prepared, monkeypatch):
    config, root = prepared
    monkeypatch.setattr(scale, "guard",
                        lambda: (_ for _ in ()).throw(MemoryError("authored floor")))
    with pytest.raises(MemoryError):
        scale.resume_build(config, root)
    (root / "leaves.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="binding"):
        scale.resume_build(config, root)


def test_completed_build_is_not_overwritten(prepared):
    config, root = prepared
    scale.resume_build(config, root)
    with pytest.raises(FileExistsError):
        scale.resume_build(config, root)
