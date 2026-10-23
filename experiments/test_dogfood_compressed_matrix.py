"""Portable builder-authored canonical matrix integration fixtures.

Exact independent-tester recommended ranges; private config checks stay private.
"""
import json
from pathlib import Path
import numpy as np
import pytest
from experiments import dogfood as d
from experiments.dogfood import Memory as RealMemory
from experiments.memory.compressed_route_matrix import CompressedRouteMatrix

@pytest.fixture
def bank(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("alpha beta gamma", encoding="utf-8")
    leaf = source.read_bytes()
    sha = d.digest(leaf)
    (tmp_path / "leaves").mkdir()
    (tmp_path / "leaves" / (sha + ".txt")).write_bytes(leaf)
    rows = [dict(text=text, source=str(source), leaf_sha256=sha,
                 start_byte=0, end_byte=len(leaf), span_sha256=sha)
            for text in ("alpha alpha", "beta beta", "gamma gamma")]
    d.write(tmp_path / "rows.json", rows)
    vectors = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
    np.save(tmp_path / "vectors.npy", vectors, allow_pickle=False)
    d.write(tmp_path / "manifest.json", dict(files=[dict(path=str(source),sha256=sha)],
        rows_sha256=d.file_digest(tmp_path / "rows.json"),
        vector_sha256=d.file_digest(tmp_path / "vectors.npy")))
    return tmp_path

@pytest.fixture
def fake_encode(monkeypatch):
    calls=[]
    def encode(encoder, texts):
        calls.extend(texts)
        return np.tile(np.array([1,.2,.1],dtype=np.float32), (len(texts),1))
    monkeypatch.setattr(d,"encode",encode)
    return calls

def test_default_parity_and_explicit_packed(bank, fake_encode):
    a=d.Memory(bank,object()); b=d.Memory(bank,object(),route_matrix="float32")
    c=d.Memory(bank,object(),route_matrix="affine8")
    assert isinstance(a.vectors,np.ndarray)
    assert isinstance(c.vectors,CompressedRouteMatrix)
    assert a.route_matrix_retained_numeric_bytes==36
    assert c.route_matrix_retained_numeric_bytes==33
    assert a.route_matrix_conversion_seconds==0
    assert c.route_matrix_conversion_seconds>=0
    for arm in ("dense","hybrid","bm25"):
        aa=a.search("alpha",3,arm); bb=b.search("alpha",3,arm); cc=c.search("alpha",3,arm)
        assert aa==bb
        assert [p["handle"] for p in aa["passages"]]==[p["handle"] for p in cc["passages"]]
        assert all(p["verification_status"]=="VERIFIED_SOURCE_BYTES" for p in cc["passages"])
        if arm=="hybrid":
            scores,dense,lex=c._scores("alpha",arm)
            np.testing.assert_array_equal(scores,d.reciprocal_rank_fusion(dense,lex))
    c.seen.clear(); first=c.search("alpha",1,"hybrid")["passages"][0]
    second=c.search("alpha",1,"hybrid",novelty=True)["passages"][0]
    assert first["handle"]!=second["handle"]
    assert first["pointer"]==a.search("alpha",1,"hybrid")["passages"][0]["pointer"]
    fake_encode.clear(); c.search("alpha",1,"bm25"); assert fake_encode==[]

@pytest.mark.parametrize("matrix",["float32","affine8"])
@pytest.mark.parametrize("fault",["vectors","rows","source","span","length"])
def test_verification_not_bypassed(bank,matrix,fault,fake_encode):
    if fault=="vectors":
        with (bank/"vectors.npy").open("ab") as f:f.write(b"x")
    elif fault=="rows":
        with (bank/"rows.json").open("ab") as f:f.write(b" ")
    elif fault=="source":(bank/"source.txt").write_text("changed")
    elif fault=="length":
        np.save(bank/"vectors.npy",np.ones((2,3),dtype=np.float32))
        m=d.load(bank/"manifest.json");m["vector_sha256"]=d.file_digest(bank/"vectors.npy")
        (bank/"manifest.json").write_text(json.dumps(m))
    else:
        memory=d.Memory(bank,None,route_matrix=matrix)
        next((bank/"leaves").iterdir()).write_text("changed")
        with pytest.raises(ValueError,match="source leaf/span"):memory.search("alpha",1,"bm25")
        return
    with pytest.raises(ValueError):d.Memory(bank,None,route_matrix=matrix)

@pytest.mark.parametrize("matrix", ["float32", "affine8"])
@pytest.mark.parametrize("arm", ["bm25", "dense", "hybrid"])
def test_run_one_real_matrix_and_result(bank, matrix, arm, fake_encode, monkeypatch):
    c=dict(schema=1, seed=1729, encoder="unused", readers=[dict(id="fake", origin="test",parameters=100)],
           caps=[1], num_ctx=4352, max_output_tokens=512, chunk_chars=1200, top_k=2,
           protocol="v2", question_prior=False, route_matrix=matrix,
           tasks=[dict(id="task", benchmark="test", allowed_paths=[str(bank/"source.txt")],
                       source_roots=[str(bank)], questions=[dict(id="q",type="test",question="Find alpha")])])
    config=bank/"config.json"; d.write(config,c)
    index=bank/"index"; index.mkdir()
    # Use the actual verified bank, without copying or rebuilding any corpus.
    monkeypatch.setattr(d,"Memory",lambda directory,encoder,**kw: RealMemory(bank,encoder,**kw))
    d.write(index/"build.json",dict(config_sha256=d.file_digest(config)))
    d.write(index/"task"/"manifest.json",dict(test_only=True))
    encoder_loads=[]
    monkeypatch.setattr(d,"encoder_at",lambda path: encoder_loads.append(path) or object())
    monkeypatch.setattr(d,"reader_info",lambda reader:dict(test_only=True))
    monkeypatch.setattr(d,"ram_guard",lambda:30*2**30)
    actions=iter([dict(query="alpha"),dict(answer="alpha [S0]")])
    monkeypatch.setattr(d,"reader_call",lambda *a,**kw:dict(message=dict(content=json.dumps(next(actions))),
        done_reason="stop",prompt_eval_count=50,eval_count=10))
    output=bank/"result.json"
    d.run_one(config,index,output,"task","q","fake",1,arm=arm)
    result=d.load(output)
    assert encoder_loads==([] if arm=="bm25" else ["unused"])
    assert result["terminated"] and not result["errors"]
    assert result["route_matrix"]==matrix
    assert result["route_matrix_retained_numeric_bytes"]==(33 if matrix=="affine8" else 36)
    assert result["route_matrix_conversion_seconds"]>=0
