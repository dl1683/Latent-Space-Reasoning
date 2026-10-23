"""Query-only scheduling and bounded model residency; no model/service calls."""
import io
import json

import pytest

from experiments import dogfood as d


def run_policy(minimum=None, cap=3, protocol='v2'):
    config=dict(schema=1, seed=1729, num_ctx=4352, max_output_tokens=512,
                top_k=2, protocol=protocol)
    if minimum is not None:
        config['minimum_memory_queries']=minimum
    phases=[]
    class Memory:
        def search(self, query, k, **kwargs):
            return dict(passages=[dict(text='Synthetic source fact.',handle='S0',
                source_id='fixture.txt',pointer=dict(source='fixture.txt'))])
    def phase_call(messages, phase):
        phases.append(phase)
        action={'query':'next bridge'} if phase=='query' else {'answer':'Bounded answer [S0]'}
        return dict(message=dict(content=json.dumps(action)),done_reason='stop',
                    prompt_eval_count=100,eval_count=10)
    result=d.run_episode(config,dict(id='fixture',benchmark='software-only'),
        dict(id='q',question='Follow the two source links.',type='multi_hop'),
        dict(id='fixture',origin='test'),cap,Memory(),None,phase_call=phase_call)
    return result,phases


def test_default_and_opt_in_query_only_phases():
    default,phases=run_policy()
    explicit,explicit_phases=run_policy(1)
    assert phases==explicit_phases==['query','choose']
    assert default['trace']==explicit['trace']
    two,two_phases=run_policy(2)
    assert two_phases==['query','query','choose']
    assert two['interactions_used']==2 and two['terminated']
    assert all('Search now.' in e['messages'][-1]['content'] for e in two['trace'][:2])


def test_requested_minimum_cannot_expand_cap_or_change_legacy_mode():
    for cap,phases in ((0,['answer']),(1,['query','answer']),(3,['query']*3+['answer'])):
        result,actual=run_policy(99,cap)
        assert actual==phases and result['interactions_used']==cap
        assert result['terminated']
    default,phases=run_policy(protocol='v1')
    minimum,other_phases=run_policy(99,protocol='v1')
    assert phases==other_phases==['choose']
    assert default['trace']==minimum['trace']


@pytest.mark.parametrize('key,value',[
    ('minimum_memory_queries',True),('minimum_memory_queries',0),
    ('minimum_memory_queries',1.5),('reader_keep_alive_seconds',True),
    ('reader_keep_alive_seconds',-1),('reader_keep_alive_seconds',301),
    ('reader_keep_alive_seconds','300'),
])
def test_invalid_policy_cannot_enter_reader_config(tmp_path,key,value):
    config=dict(schema=1,seed=1729,tasks=[],readers=[],caps=[0,1,3],num_ctx=4352,
                max_output_tokens=512,chunk_chars=1200,top_k=2)
    config[key]=value
    path=tmp_path/'config.json'
    path.write_text(json.dumps(config),encoding='utf-8')
    with pytest.raises(ValueError,match=key):
        d.config_at(path)


def test_residency_changes_only_existing_wire_lifetime(monkeypatch):
    payloads=[]
    def fake_open(request,timeout):
        payloads.append(json.loads(request.data))
        assert timeout==180
        return io.BytesIO((json.dumps(dict(message=dict(content='{"answer":"fixture"}'),
            done=True,done_reason='stop',prompt_eval_count=2,eval_count=3))+'\n').encode())
    monkeypatch.setattr(d.urllib.request,'urlopen',fake_open)
    monkeypatch.setattr(d,'ram_guard',lambda:30*2**30)
    config=dict(seed=1729,num_ctx=4352,max_output_tokens=512,num_batch=128)
    messages=[dict(role='user',content='Synthetic question')]
    reader=dict(id='fixture')
    for extra in ({},{'reader_keep_alive_seconds':0},{'reader_keep_alive_seconds':300}):
        d.reader_call(dict(config,**extra),reader,messages)
    assert payloads[0]==payloads[1]
    assert payloads[0]['keep_alive']==0 and payloads[2]['keep_alive']==300
    baseline=dict(payloads[0]);warm=dict(payloads[2])
    baseline.pop('keep_alive');warm.pop('keep_alive')
    assert baseline==warm
