import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.memory.independent_ordinary_input import LIMITS, SCHEMA, SCOPE, decode_cases
from experiments.memory.run_independent_ordinary_gate import _bind_generation_stops, _outside_repo


def _packet(cases=None, **overrides):
    payload = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "limits": dict(LIMITS),
        "cases": cases if cases is not None else [
            {"id": f"case-{index}", "task": f"Task {index}",
             "source_deltas": [{"a" * 32: f"revision {index}"}, {"a" * 32: None}], "read_epoch": 2}
            for index in range(4)
        ],
    }
    payload.update(overrides)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def test_input_adapter_normalizes_only_the_four_frozen_cases():
    cases = decode_cases(_packet())
    assert len(cases) == 4
    assert cases[0].case_id == "case-0"
    assert cases[0].read_epoch == 2
    assert cases[0].source_deltas == ({"a" * 32: "revision 0"}, {"a" * 32: None})


@pytest.mark.parametrize("packet", (
    _packet(grader={"answers": []}),
    _packet(limits={**LIMITS, "max_new_tokens": 513}),
    _packet(cases=[{"id": "duplicate", "task": "x", "source_deltas": [{}], "read_epoch": 1}] * 4),
))
def test_input_adapter_rejects_extra_grade_data_wrong_limits_and_duplicate_ids(packet):
    with pytest.raises(ValueError):
        decode_cases(packet)


def test_input_adapter_rejects_boolean_epoch_and_duplicate_json_keys():
    payload = json.loads(_packet())
    payload["cases"][0]["read_epoch"] = True
    with pytest.raises(ValueError, match="source delta history"):
        decode_cases(json.dumps(payload).encode())
    with pytest.raises(ValueError, match="duplicate JSON key"):
        decode_cases(b'{"schema":"wrong","schema":"wrong"}')


def test_gate_adapter_pins_one_eos_and_the_tokenizer_padding_id():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[151645, 151643], pad_token_id=151643))
    tokenizer = SimpleNamespace(eos_token_id=151645, pad_token_id=151643)
    assert _bind_generation_stops(model, tokenizer) == (151645, 151643)
    assert model.generation_config.eos_token_id == 151645
    assert model.generation_config.pad_token_id == 151643


def test_gate_paths_keep_packets_and_outputs_outside_the_checkout(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="outside the source checkout"):
        _outside_repo(repository / "private-input.json", "input packet")
    assert _outside_repo(tmp_path, "output directory") == tmp_path.resolve()