import copy
import json

import numpy as np
import pytest

from experiments import musique_dev as shared
from experiments.memory.latent_frontend import LatentFrontend
from experiments.musique_latent import SharedFrontend
from experiments.retrieval_benchmark import sha


@pytest.fixture
def fixture(tmp_path):
    row = dict(id="source-only", question="Which registry number belongs to Néris?",
               paragraphs=[
                   dict(idx=7, title="Néris", paragraph_text="Néris links to Registry."),
                   dict(idx=9, title="Registry", paragraph_text="Néris has registry number 83."),
                   dict(idx=3, title="Other", paragraph_text="Other has registry number 29.")])
    generation = tmp_path / "generation.jsonl"
    generation.write_text(json.dumps(row) + "\n", encoding="utf8")
    work = tmp_path / "memory"
    work.mkdir()
    LatentFrontend.build(work / "000", row["paragraphs"], np.eye(3, dtype=np.float32),
                         encoder_identity={"fixture": True})
    np.save(work / "queries.npy", np.array([[1, 0, 0]], dtype=np.float32))
    shared.write(work / "PREPARED.json", dict(
        generation_sha256=sha(generation.read_bytes()),
        query_sha256=sha((work / "queries.npy").read_bytes())))
    return row, dict(generation=str(generation), work=str(work), top_k=2, graph=True)


@pytest.mark.parametrize("arm", SharedFrontend.arms)
def test_shared_exact_spans_and_equal_budget(fixture, arm):
    row, config = fixture
    frontend = SharedFrontend(config)
    result = frontend.retrieve(row, arm)
    shared.verify_spans(row, result["evidence"])
    assert len(result["evidence"]) == (3 if arm == "packed" else 2)
    assert result["costs"]["corpus_reencodes"] == 0
    assert result["costs"]["source_bytes_touched"] > 0


def test_shared_episode_uses_same_reader_prompt(fixture):
    row, config = fixture
    class Reader:
        def call(self, messages, **kwargs):
            assert messages[0]["content"] == shared.SYSTEM
            assert json.loads(messages[1]["content"])["question"] == row["question"]
            return json.dumps(dict(answer="83", quotes=["Néris has registry number 83."]))
    result = shared.episode(Reader(), row, "latent", {}, config,
                            retriever=SharedFrontend(config))
    assert result["answer"] == "83"
    assert result["citations"][0]["matches"][0]["idx"] == 9
    assert result["retrieval"]["map_bytes_touched"] > 0


def test_changed_query_or_sources_refused(fixture):
    row, config = fixture
    frontend = SharedFrontend(config)
    changed = copy.deepcopy(row)
    changed["question"] += " malicious substitution"
    with pytest.raises(ValueError, match="allow-list"):
        frontend.retrieve(changed, "latent")
    changed = copy.deepcopy(row)
    changed["paragraphs"][0]["paragraph_text"] += " changed"
    with pytest.raises(ValueError, match="allow-list"):
        frontend.retrieve(changed, "latent")


def test_generation_or_queries_tampering_refused(fixture):
    row, config = fixture
    path = shared.Path(config["generation"])
    path.write_text(json.dumps(dict(row, question="Changed")), encoding="utf8")
    with pytest.raises(ValueError, match="binding"):
        SharedFrontend(config)


def test_unknown_arm_and_evaluator_fields_refused(fixture):
    row, config = fixture
    frontend = SharedFrontend(config)
    with pytest.raises(ValueError):
        frontend.retrieve(row, "oracle")
    with pytest.raises(ValueError):
        frontend.retrieve(dict(row, answer="83"), "latent")


def test_failed_readout_retains_retrieval_costs(fixture):
    row, config = fixture
    class Reader:
        def call(self, *args, **kwargs):
            raise RuntimeError("authored reader failure")
    reader = Reader()
    with pytest.raises(RuntimeError):
        shared.episode(reader, row, "latent", {}, config,
                       retriever=SharedFrontend(config))
    assert reader.last_retrieval["source_bytes_touched"] > 0


def test_shared_score_charges_failed_attempts(fixture, tmp_path):
    from experiments.musique_latent_score import report
    row, config = fixture
    config.update(readers=[dict(id="fixture")], arms=["bm25"],
                  input_sha256={"generation": sha(
                      shared.Path(config["generation"]).read_bytes())})
    cfg = tmp_path / "config.json"
    shared.write(cfg, config)
    gold = shared.Path(config["generation"]).parent / "gold.jsonl"
    gold.write_bytes(shared.encoded(dict(id=row["id"], answer="83",
                                        answer_aliases=[], supporting=[9])) + b"\n")
    out = tmp_path / "run"
    out.mkdir()
    shared.write(out / "run.json", dict(latent_component=True))
    failed = dict(reader="fixture", arm="bm25", id=row["id"], status="FAILED",
                  calls=1, retrieval=dict(source_bytes_touched=17),
                  reader_usage=dict(request_id="failed", prompt_tokens=7,
                                    output_tokens=3, done_reason="length"))
    completed = dict(failed, attempt=2, status="COMPLETED", answer="83",
                     evidence=shared.spans(row["paragraphs"], [0, 1]), citations=[],
                     retrieval=dict(source_bytes_touched=19),
                     reader_usage=dict(request_id="completed", prompt_tokens=11,
                                       output_tokens=2, done_reason="stop"))
    for value in (failed, completed):
        shared.append(out / "predictions.jsonl", value)
    result = report(cfg, out, gold)["summary"][0]
    assert result["reader_calls"] == 2
    assert result["prompt_eval_tokens"] == 18
    assert result["output_tokens"] == 5
    assert result["retrieval_bytes"]["source_bytes_touched"] == 36
    assert result["termination_rate"] == .5
    assert result["capability_interpretation"] == "WITHHELD"
