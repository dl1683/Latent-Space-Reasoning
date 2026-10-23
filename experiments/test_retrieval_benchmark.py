import json
from pathlib import Path

import numpy as np
import pytest

from experiments import retrieval_benchmark as b


def test_metric_exact_aliases_empty_and_overlap():
    assert b.metrics("The Port!", ["port"]) == (1, 1)
    assert b.metrics("", [""]) == (1, 1)
    assert b.metrics("UNKNOWN", ["turquoise"]) == (0, 0)
    assert b.metrics("north port", ["port"]) == (0, 2/3)
    assert b.metrics("three", ["3", "three"]) == (1, 1)


def test_paired_bootstrap_stable():
    assert b.bootstrap([0, 0, 0]) == [0, 0]
    assert b.bootstrap([1, 1, 1]) == [1, 1]
    assert b.bootstrap([-.5, .25, 0]) == b.bootstrap([-.5, .25, 0])


def test_generation_rejects_gold(tmp_path):
    p = tmp_path/"generation.jsonl"
    row = dict(id="q", question="What?", paragraphs=[
        dict(idx=0, title="T", paragraph_text="Source")], answer="CANARY")
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="generation-only"):
        b.generation_at(p)
    del row["answer"]
    row["paragraphs"][0]["is_supporting"] = True
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="source-only"):
        b.generation_at(p)


def test_request_excludes_provenance_and_uses_same_template():
    evidence = [dict(idx=2, title="T", paragraph_text="Body", pointer={"x":"y"},
                     verification="BYTE_IDENTITY_ONLY")]
    m = b.request("What?", evidence)
    assert m[0]["content"] == b.FINAL
    assert json.loads(m[1]["content"]) == dict(question="What?", evidence=[
        dict(idx=2, title="T", paragraph_text="Body")])


def test_save_is_exclusive_and_never_publishes_partial_json(tmp_path):
    path = tmp_path/"result.json"
    with pytest.raises(ValueError):
        b.save(path, {"invalid":float("nan")})
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
    b.save(path, {"valid":True})
    with pytest.raises(FileExistsError):
        b.save(path, {"valid":False})
    assert json.loads(path.read_bytes()) == {"valid":True}


def test_clustered_bootstrap_and_shared_source_union():
    rows = [
        dict(id="a", question="?", paragraphs=[dict(idx=0,title="T",paragraph_text="shared")]),
        dict(id="b", question="?", paragraphs=[dict(idx=1,title="T",paragraph_text="shared")]),
        dict(id="c", question="?", paragraphs=[dict(idx=2,title="U",paragraph_text="other")])]
    gold={r["id"]:dict(support_indices=[r["paragraphs"][0]["idx"]]) for r in rows}
    groups=b.source_clusters(rows,gold)
    assert groups["a"] == groups["b"] != groups["c"]
    assert b.bootstrap([0,1],clusters=[1,1]) is None
