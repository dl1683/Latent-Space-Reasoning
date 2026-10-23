import sys
from types import SimpleNamespace

import pytest

from experiments.memory.prior_art_gpu import (
    windows, select, suffix, bert_state_dict, check, run, stream_model,
)


def test_complete_window_coverage():
    for n in (1, 177, 178, 280, 281, 1000):
        out = windows(list(range(n)))
        assert set().union(*(set(range(a, b)) for a, b, _ in out)) == set(range(n))
        assert all(seq == list(range(a, b)) for a, b, seq in out)
        assert out[-1][1] == n


def test_evidence_budget_and_ties():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)
    chunks = [{"rendered": x} for x in ["abc", "de", "fghi"]]
    assert select([1, 1, 0], chunks, Tokenizer(), budget=7) == ([0, 1], 7)
    assert select([0, 1, 2], chunks, Tokenizer(), budget=3) == ([1], 2)


def test_question_is_data_not_a_tool_call():
    assert suffix("What is stated?") == "\n\nQuestion: What is stated?[/INST]"


def test_checkpoint_pooler_membership_preserved():
    state = {"bert.pooler.dense.weight": 1, "bert.pooler.dense.bias": 2,
             "bert.embeddings.position_ids": 3, "linear.weight": 4}
    assert bert_state_dict(state) == {
        "pooler.dense.weight": 1, "pooler.dense.bias": 2}


def test_joined_evidence_separator_counts_against_budget():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)
    chunks = [{"rendered": x} for x in ["abc", "de"]]
    assert select([1, 1], chunks, Tokenizer(), budget=5) == ([0], 3)


def test_worker_checks_external_safety_reserve(monkeypatch):
    fake = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(
        available=20.49 * 2**30))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    with pytest.raises(RuntimeError, match="20.5 GiB safety reserve"):
        check()
    fake.virtual_memory = lambda: SimpleNamespace(available=20.5 * 2**30)
    check()


@pytest.mark.parametrize("phase", ["smoke", "panel"])
def test_withdrawn_reader_phases_refuse_before_resource_checks(monkeypatch, phase):
    def forbidden():
        raise AssertionError("resource checks must not precede withdrawal")
    monkeypatch.setattr("experiments.memory.prior_art_gpu.check", forbidden)
    with pytest.raises(RuntimeError, match="WITHDRAWN"):
        run({}, phase)


def test_legacy_loader_refuses_without_reading_model_path(tmp_path):
    with pytest.raises(RuntimeError, match="WITHDRAWN: unsupported legacy 7B reader loader"):
        stream_model(tmp_path / "nonexistent")
