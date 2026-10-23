import copy
import json

import pytest

from experiments import musique_dev as m
from experiments import musique_dev_eval as e


@pytest.fixture
def row():
    return dict(id="authored", question="What number belongs to Neris?", paragraphs=[
        dict(idx=7, title="Neris", paragraph_text="Neris is a city. Its number is 83."),
        dict(idx=9, title="Other", paragraph_text="Other has number 29. Different town.")])


def test_projection_rejects_gold(row):
    assert m.validate_row(row) == row
    with pytest.raises(ValueError):
        m.validate_row(dict(row, answer="83"))
    row["paragraphs"][0]["is_supporting"] = True
    with pytest.raises(ValueError):
        m.validate_row(row)


def test_multibyte_spans(row):
    row["paragraphs"][0]["paragraph_text"] = "Néris is a city. Its number is 83."
    evidence = m.spans(row["paragraphs"], [0, 1], split=True)
    m.verify_spans(row, evidence)
    assert "".join(s["text"] for s in evidence if s["idx"] == 7) == row["paragraphs"][0]["paragraph_text"]
    bad = copy.deepcopy(evidence)
    bad[0]["text"] += " wrong"
    with pytest.raises(ValueError):
        m.verify_spans(row, bad)


def test_quote_match_not_fabrication(row):
    evidence = m.spans(row["paragraphs"], [0])
    good, bad = m.attach_quotes(evidence, ["Its number is 83.", "number is 29.", ""])
    assert len(good) == 1 and bad == ["number is 29.", ""]
    assert good[0]["matches"][0]["idx"] == 7


def test_bm25_and_fusion(row):
    assert m.LexicalIndex(row["paragraphs"]).rank("Neris")[0] == 0
    assert m.fuse([1, 0], [0, 1]) == [0, 1]


@pytest.mark.parametrize(("prediction", "answers", "expected"), [
    ("The Nile", ["Nile"], (1, 1)), ("red blue", ["blue green"], (0, .5)),
    ("", [""], (1, 1)), ("UNKNOWN", ["83"], (0, 0)),
    ("New York", ["NYC", "New York"], (1, 1)),
    ("A\tred\u2003fox", ["red fox"], (1, 1)),
])
def test_metrics(prediction, answers, expected):
    result = e.score(prediction, answers)
    assert (result["em"], result["f1"]) == expected


def test_bootstrap_paired_difference():
    result = e.bootstrap([.2] * 150, iterations=100)
    assert result["mean"] == pytest.approx(.2)
    assert result["ci95"] == pytest.approx([.2, .2])
    assert e.bootstrap([]) is None


class Reader:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def call(self, messages, **kwargs):
        self.messages.append(messages)
        return json.dumps(next(self.outputs))


def test_all_arms_receive_only_own_sources(row):
    for arm in ("bm25", "hybrid", "packed", "oracle"):
        reader = Reader([dict(answer="83", quotes=["Its number is 83."])])
        result = m.episode(reader, row, arm, dict(order=[1, 0]), {}, [7])
        m.verify_spans(row, result["evidence"])
        assert result["answer"] == "83"
        assert result["citations"]


def test_host_map_compose_bounds_and_unverified_extraction(row):
    reader = Reader([
        dict(facets=["Where is Neris?", "What is the number?"]),
        dict(answer="Fabricated place", quotes=["Not in source"]),
        dict(answer="83", quotes=["Its number is 83."]),
        dict(answer="83", quotes=["Its number is 83."])])
    result = m.episode(reader, row, "map_compose", dict(order=[0, 1]), {})
    assert len(reader.messages) == 4
    assert result["observations"][0]["answer"] == "UNKNOWN"
    assert "Fabricated place" not in json.dumps(reader.messages[-1])
    assert result["answer"] == "83"


def test_source_control_independent_of_benchmark():
    pc = m.smoke(Reader([dict(answer="83", quotes=[]), dict(answer="29", quotes=[])]))
    assert pc["passed"]


def test_partial_scores_and_pairs_stay_null(tmp_path, row):
    rows = [dict(row, id=f"q{i}") for i in range(150)]
    generation = tmp_path / "generation.jsonl"
    generation.write_bytes(b"".join(m.encoded(r) + b"\n" for r in rows))
    gold = tmp_path / "gold.jsonl"
    gold.write_bytes(b"".join(m.encoded(dict(id=r["id"], answer="83",
        answer_aliases=[], supporting=[7])) + b"\n" for r in rows))
    config = tmp_path / "config.json"
    m.write(config, dict(generation=str(generation), readers=[dict(id="fake")],
                         arms=["bm25", "hybrid"]))
    result = dict(reader="fake", arm="bm25", id="q0", status="COMPLETED",
                  answer="83", evidence=m.spans(row["paragraphs"], [0]), calls=1)
    m.append(tmp_path / "predictions.jsonl", result)
    summary = e.evaluate(config, tmp_path, gold)
    assert summary[0]["completed"] == 1
    assert summary[0]["conditional_f1"] == 1
    assert summary[0]["f1"] is None
    scores = m.load(tmp_path / "scores.json")
    assert scores["paired"][0]["delta_f1"] is None
    assert sum(r["f1"] is None for r in scores["per_question"]) == 299


def test_duplicate_prediction_refused(tmp_path, row):
    generation, gold, config = [tmp_path / f"{name}.json" for name in ("gen", "gold", "config")]
    generation.write_bytes(m.encoded(row) + b"\n")
    gold.write_bytes(m.encoded(dict(id=row["id"], answer="83", answer_aliases=[],
                                   supporting=[7])) + b"\n")
    m.write(config, dict(generation=str(generation), readers=[dict(id="fake")], arms=["bm25"]))
    prediction = dict(reader="fake", arm="bm25", id=row["id"], status="FAILED", calls=1)
    m.append(tmp_path / "predictions.jsonl", prediction)
    m.append(tmp_path / "predictions.jsonl", prediction)
    with pytest.raises(ValueError, match="duplicate"):
        e.evaluate(config, tmp_path, gold)


def test_failed_attempts_remain_charged():
    failed = dict(reader="r", arm="bm25", id="q", status="FAILED", calls=1)
    recovered = dict(failed, status="COMPLETED", attempt=2, calls=1)
    assert list(m.latest_attempts([failed, recovered]).values()) == [recovered]
    with pytest.raises(ValueError, match="duplicate"):
        m.latest_attempts([failed, recovered, recovered])


def test_resume_failed_rows_reaches_cpu_factory(tmp_path, monkeypatch, row):
    rows = [dict(row, id=str(i)) for i in range(150)]
    paths = {}
    for key, values in (
        ("generation", rows),
        ("routes", [dict(id=r["id"], order=[0, 1]) for r in rows]),
        ("oracle_indices", [dict(id=r["id"], indices=[7]) for r in rows])):
        path = tmp_path / f"{key}.jsonl"
        path.write_bytes(b"".join(m.encoded(v) + b"\n" for v in values))
        paths[key] = str(path)
    config = dict(paths, readers=[dict(id="stub", backend="transformers")],
                  arms=["bm25"], input_sha256={k: m.sha(open(v, "rb").read()) for k, v in paths.items()})
    cfg = tmp_path / "config.json"
    m.write(cfg, config)
    out = tmp_path / "run"
    out.mkdir()
    m.write(out / "run.json", dict(config_sha256=m.sha(cfg.read_bytes()),
        code_sha256=m.sha(m.Path(m.__file__).read_bytes())))
    for r in rows:
        m.append(out / "predictions.jsonl", dict(
            reader="stub", arm="bm25", id=r["id"], status="FAILED", calls=1))
    constructed = []
    def blocked(*args):
        constructed.append(True)
        raise RuntimeError("authored fixture, no model")
    monkeypatch.setattr(m, "TransformerReader", blocked)
    m.run(cfg, out, resume=True)
    assert constructed
