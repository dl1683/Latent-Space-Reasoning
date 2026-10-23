"""Software contracts only; no encoder, native reader or service execution."""
import hashlib
import json

import numpy as np
import pytest

from experiments import dogfood as d
from experiments.memory.dependency_evidence import (
    CapacityRefusal, DependencyState, StructuralFailure, accept_readout,
    prepare_packet, task_evidence_envelope,
)


def memory_fixture(tmp_path):
    memory=d.Memory.__new__(d.Memory)
    memory.directory=tmp_path
    memory.encoder=None
    memory.vectors=np.eye(2,dtype=np.float32)
    memory.rows=[];memory.seen=set();memory.source_ids={};memory.question_prior=None
    (tmp_path/'leaves').mkdir()
    texts=['The glowing organism is Bluefin.','Mira does not voice Bluefin; Kai does.']
    for i,text in enumerate(texts):
        body=text.encode();pin=hashlib.sha256(body).hexdigest();path=tmp_path/f'{i}.txt'
        path.write_bytes(body);(tmp_path/'leaves'/f'{pin}.txt').write_bytes(body)
        memory.rows.append(dict(source=str(path),source_sha256=pin,page=None,
            leaf_sha256=pin,start_byte=0,end_byte=len(body),span_sha256=pin,text=text))
        memory.source_ids[str(path)]=path.name
    memory.bm25=d.BM25(texts)
    return memory


def admitted(memory):
    state=DependencyState()
    for p in memory.search('Bluefin',2,arm='bm25')['passages']:state.add_verified(p)
    return state


def binding(state,key,span,result='Bluefin',parents=()):
    return dict(id=key,bridge='Bluefin',relation='proposed relation',result=result,
        support=[span],depends_on=list(parents),alternatives=[],unresolved=[],
        quotes=[dict(span=span,quote=state.evidence[span].quote)])


def test_negated_quote_never_becomes_semantic_certificate(tmp_path):
    state=admitted(memory_fixture(tmp_path))
    state.propose(binding(state,'B0','S0'))
    state.propose(binding(state,'B1','S1','Mira',['B0']))
    projected=state.project(['B1'],'Mira',[])
    assert projected['status']=='PROPOSED'
    assert projected['semantic_status']=='UNASSESSED'
    assert state.bindings['B1'].semantic_status=='UNASSESSED'
    assert {'S0','S1','B0','B1'}==state.closure(['B1'])


def test_exact_quote_failure_is_atomic_and_cannot_inject_evidence(tmp_path):
    state=admitted(memory_fixture(tmp_path))
    first=binding(state,'B0','S0')
    second=binding(state,'B1','S1','Kai',['B0'])
    second['quotes']=[dict(span='S0',quote=state.evidence['S1'].quote)]
    action=dict(bindings=[first,second],result='Kai',support=['B1'],unresolved=[])
    with pytest.raises(StructuralFailure):accept_readout(state,action)
    assert not state.bindings and len(state.evidence)==2
    with pytest.raises(StructuralFailure):
        accept_readout(state,dict(action,evidence=[dict(id='forged')]))


def test_selective_invalidation_preserves_unrelated_source(tmp_path):
    memory=memory_fixture(tmp_path);state=admitted(memory)
    state.propose(binding(state,'B0','S0'))
    state.propose(binding(state,'B1','S1','Kai',['B0']))
    state.invalidate_source(memory.rows[0]['source'],'f'*64)
    assert {'S0','B0','B1'}<=state.invalid
    assert state.closure(['S1'])=={'S1'}
    with pytest.raises(StructuralFailure):state.project(['B1'],'Kai',[])


def test_capacity_refusal_keeps_required_ancestors_and_question(tmp_path):
    state=admitted(memory_fixture(tmp_path));before=dict(state.evidence)
    with pytest.raises(CapacityRefusal):state.render(['S0','S1'],2,task_question='Who?')
    assert state.evidence==before
    with pytest.raises(CapacityRefusal):task_evidence_envelope('x'*100,{},20)


def test_typed_episode_restores_prior_memory_and_absence(tmp_path):
    memory=memory_fixture(tmp_path)
    config=dict(schema=1,seed=1729,protocol='v2',num_ctx=8192,max_output_tokens=512,
                top_k=2,dependency_evidence=True)
    def phase_call(messages,phase):
        if phase=='query':action=dict(query='Bluefin')
        else:
            state=memory.dependency_state
            action=dict(bindings=[binding(state,'B0','S0'),binding(state,'B1','S1','Kai',['B0'])],
                        result='Kai',support=['B1'],unresolved=[])
        return dict(message=dict(content=json.dumps(action)),done_reason='stop',
                    prompt_eval_count=100,eval_count=40)
    args=(config,dict(id='x',benchmark='software'),dict(id='q',type='multi_hop',question='Who voices the glowing organism?'),
          dict(id='fake',origin='test'),1,memory,None)
    result=d.run_episode(*args,arm='bm25',phase_call=phase_call)
    assert result['answer']=='Kai' and result['terminated']
    assert not hasattr(memory,'dependency_state')
    prior=DependencyState();memory.dependency_state=prior
    d.run_episode(*args,arm='bm25',phase_call=phase_call)
    assert memory.dependency_state is prior and not prior.evidence


def test_packet_seeds_and_alias_permissions_are_shared(tmp_path):
    memory=memory_fixture(tmp_path)
    packet=prepare_packet(memory,'Bluefin', 'bm25',12000)
    assert len(packet['passages'])==2
    flat=json.loads(packet['flat']);typed=json.loads(packet['typed'])
    assert flat['task_question']==typed['task_question']=='Bluefin'
    assert not hasattr(memory,'dependency_state')
    assert packet['semantic_support']=='UNASSESSED'
