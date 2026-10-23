"""Synthetic contract regressions. No real data, model, encoder or network.
Originally authored by the independent CLI tester and executed by an independent tester.
Assertions state required behavior; not xfail.
"""
from pathlib import Path

import numpy as np
import pytest

from experiments import musique_dev as shared
from experiments import latent_frontend_scale as scale
from experiments.memory.latent_frontend import LatentFrontend
from experiments.musique_latent import SharedFrontend


@pytest.fixture
def frontend_config(tmp_path):
    row = dict(id="synthetic", question="Where is Alpha?", paragraphs=[
        dict(idx=1, title="Alpha", paragraph_text="Alpha mentions Bravo."),
        dict(idx=2, title="Bravo", paragraph_text="Bravo is a location.")])
    generation = tmp_path / "generation.jsonl"
    generation.write_bytes(shared.encoded(row) + b"\n")
    work = tmp_path / "memory"
    work.mkdir()
    LatentFrontend.build(work / "000", row["paragraphs"],
                         np.eye(2, dtype=np.float32), encoder_identity={"synthetic": True})
    np.save(work / "queries.npy", np.array([[1, 0]], dtype=np.float32))
    shared.write(work / "PREPARED.json", dict(
        generation_sha256=shared.sha(generation.read_bytes()),
        query_sha256=shared.sha((work / "queries.npy").read_bytes())))
    return dict(generation=str(generation), work=str(work), top_k=2, graph=True)


def test_routing_dependency_change_changes_resume_identity(frontend_config, monkeypatch):
    before = SharedFrontend(frontend_config).identity
    # Emulate changed dependency bytes without editing any implementation file.
    dependency = Path(shared.__file__).parent / "memory" / "compressed_routing_map.py"
    original = Path.read_bytes
    def changed(path):
        raw = original(path)
        return raw + b"\n# synthetic changed routing dependency\n" if path.resolve() == dependency.resolve() else raw
    monkeypatch.setattr(Path, "read_bytes", changed)
    after = SharedFrontend(frontend_config).identity
    assert after != before, "BM25/RRF implementation changed without changing resume identity"


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    class SyntheticEncoder:
        max_seq_length = 32
        def tokenizer(self, text, **kwargs):
            return {"input_ids": [1]}
        def get_sentence_embedding_dimension(self):
            return 2
        def encode(self, texts, **kwargs):
            return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(texts), 1))
    monkeypatch.setattr(scale, "encoder_at", lambda _: SyntheticEncoder())
    source = tmp_path / "synthetic_source.txt"
    source.write_bytes(b"synthetic original source")
    config = tmp_path / "config.json"
    shared.write(config, dict(encoder="synthetic", scale_tokenizer="synthetic",
                              scale_sources=[str(source)]))
    out = tmp_path / "checkpoint"
    out.mkdir()
    raw = b"Synthetic chunk."
    (out / "leaves.bin").write_bytes(raw * 65)
    shared.write(out / "pointers.json", [dict(start=i*len(raw), end=(i+1)*len(raw),
        span_sha256=shared.sha(raw), source_sha256=shared.sha(source.read_bytes())) for i in range(65)])
    shared.write(out / "token_counts.json", [2]*65)
    checks = 0
    def interrupt():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise MemoryError("synthetic interruption after first durable batch")
    monkeypatch.setattr(scale, "guard", interrupt)
    with pytest.raises(MemoryError):
        scale.resume_build(config, out)
    assert shared.load(out / "ENCODING.json")["completed"] == 32
    monkeypatch.setattr(scale, "guard", lambda: None)
    return config, out, source


@pytest.mark.parametrize("replacement", [0.5, float("nan")], ids=["finite-substitution", "nan-substitution"])
def test_resume_refuses_changed_completed_vectors(checkpoint, replacement):
    config, out, _ = checkpoint
    bank = np.load(out / "vectors_checkpointed.npy", mmap_mode="r+")
    bank[0, 0] = replacement
    bank.flush()
    del bank
    with pytest.raises(ValueError):
        scale.resume_build(config, out)
    assert not (out / "BUILD.json").exists()


def test_resume_refuses_impossible_checkpoint_cursor(checkpoint):
    config, out, _ = checkpoint
    state = shared.load(out / "ENCODING.json")
    state["completed"] = 66  # only 65 rows exist
    shared.write(out / "ENCODING.json", state)
    with pytest.raises(ValueError):
        scale.resume_build(config, out)
    assert not (out / "BUILD.json").exists()


def test_resume_refuses_changed_source_attribution(checkpoint):
    config, out, source = checkpoint
    source.write_bytes(b"synthetic replacement source unrelated to encoded chunks")
    with pytest.raises(ValueError):
        scale.resume_build(config, out)
    assert not (out / "BUILD.json").exists()


def test_memory_failure_after_call_remains_charged(tmp_path, monkeypatch):
    row = dict(question="Where is Alpha?", paragraphs=[
        dict(idx=1, title="Alpha", paragraph_text="Alpha is here.")])
    rows = [dict(row, id=str(i)) for i in range(150)]
    paths = {}
    for name, values in [
        ("generation", rows),
        ("routes", [dict(id=r["id"], order=[0]) for r in rows]),
        ("oracle_indices", [dict(id=r["id"], indices=[1]) for r in rows])]:
        path = tmp_path / (name + ".jsonl")
        path.write_bytes(b"".join(shared.encoded(r) + b"\n" for r in values))
        paths[name] = str(path)
    config = tmp_path / "config.json"
    shared.write(config, dict(paths, readers=[dict(id="synthetic", backend="transformers")],
        arms=["bm25"], input_sha256={k: shared.sha(Path(v).read_bytes()) for k, v in paths.items()}))
    class SyntheticReader:
        def __init__(self, *args):
            self.counter, self.minimum_ram, self.metadata = 0, 100, {}
        def call(self, *args, **kwargs):
            self.counter += 1
            raise MemoryError("synthetic failure after attempted call")
        def close(self):
            pass
    monkeypatch.setattr(shared, "TransformerReader", SyntheticReader)
    monkeypatch.setattr(shared, "memory_check", lambda *args: 100)
    monkeypatch.setattr(shared, "smoke", lambda _: {"passed": True})
    out = tmp_path / "run"
    with pytest.raises(MemoryError):
        shared.run(config, out)
    predictions = shared.read_rows(out / "predictions.jsonl") if (out / "predictions.jsonl").exists() else []
    assert len(predictions) == 1, "attempt disappeared at MemoryError; resume will start again at attempt 1"
    assert predictions[0]["status"] == "FAILED"
    assert predictions[0]["calls"] == 1
    assert predictions[0].get("attempt", 1) == 1
