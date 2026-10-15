"""Finite FIX1 regressions; trusted offline CPU instruments, never readiness."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(directory, cfg, connector, memory, question):
    from experiments.cloud.lsr_jobs import s0_envelope as env
    from . import reader, reader_model, semantic
    directory = Path(directory)
    directory.mkdir()
    handle = semantic.load_handle(cfg)
    stem_cfg = cfg['semantic']['stem']
    recipe = {'kind': 'tiny-random' if stem_cfg['kind'] == 'tiny-random' else 'staged-real',
              'stem': {'initialization_seed': stem_cfg['seed'], 'config': handle['receiver'].stem.config.to_dict()} if stem_cfg['kind'] == 'tiny-random' else {k:v for k,v in stem_cfg['receiver_config'].items() if k in reader_model.REAL_FIELDS},
              'tokenizer_specials': {k: getattr(handle['tok'], k) for k in ('pad_token', 'eos_token', 'unk_token')}}
    (directory/'config.json').write_text(json.dumps(recipe), encoding='utf-8')
    (directory/'tokenizer.json').write_text(handle['tok'].backend_tokenizer.to_str(), encoding='utf-8')
    (directory/'connector.safetensors').write_bytes(Path(connector).read_bytes())
    (directory/'memory.json').write_text(env.serialize_envelope(memory, expected_width=handle['stem_info']['D_memory'],
                     expected_schema=env.WIDTH_SCHEMA_VERSION), encoding='utf-8')
    (directory/'query.json').write_text(json.dumps({'question':question,'budget':8}), encoding='utf-8')
    package = Path(__file__).resolve().parent
    repo = package.parents[2]
    code = reader_model.code_paths(package, repo)
    manifest={'inputs':{n:sha(directory/n) for n in ('config.json','connector.safetensors','memory.json','query.json','tokenizer.json')},
              'code':{str(p.resolve()):sha(p) for p in code if p.exists()}}
    (directory/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return Path(reader.__file__).resolve()


def _reader_command(script, *args):
    # Predeclare the isolated working directory as tempfile's runtime location.
    # This prevents a cold library import from probing/writing global temp paths;
    # the reader's audited section still forbids all retained file writes.
    bootstrap = "import os,sys,tempfile,runpy; tempfile.tempdir=os.getcwd(); script=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(script,run_name='__main__')"
    return [sys.executable, '-B', '-I', '-c', bootstrap, str(script), *args]


def child(script, directory, *args):
    result = subprocess.run(_reader_command(script, *args), cwd=directory,
                   capture_output=True,text=True,encoding='utf-8',timeout=120)
    (Path(directory)/'child_stderr.log').write_text(result.stderr,encoding='utf-8')
    (Path(directory)/'child_stdout.log').write_text(result.stdout,encoding='utf-8')
    if result.returncode:
        raise AssertionError(f'reader exit {result.returncode}: {result.stderr[-1800:]}')
    return json.loads(result.stdout.strip().splitlines()[-1])


def reader_device_policy(script, directory, out):
    """Execute the outer entrypoint; raising loader mocks never load real assets."""
    import shutil
    checks = {}
    for kind in ('tiny-random', 'cpu', 'cuda:0', 'multi-gpu', 'invalid-real'):
        target = Path(out) / ('device-' + kind.replace(':', '-'))
        shutil.copytree(directory, target)
        recipe = json.loads((target/'config.json').read_text())
        if kind != 'tiny-random':
            # Empty synthetic mirror metadata, no model weights or downloads.
            root = target/'mock-mirror'
            root.mkdir()
            (root/'_manifest_sha256.txt').write_text('')
            (root/'_COMPLETE').write_text('mock-only')
            devices = {'':kind} if kind in ('cpu','cuda:0') else {'model':'cuda:0','lm_head':'cuda:1'}
            recipe.update(kind='staged-real', stem={
                'model':'google/gemma-4-E2B', 'revision':'a'*40, 'tokenizer_revision':'b'*40,
                'model_local':str(root.resolve()), 'allow_hf_download':False,
                'weights_manifest_sha256':sha(root/'_manifest_sha256.txt'),
                'placement':{'device_map':devices, 'measurement_id':'offline-raising-mock', 'min_free_fraction':.2},
                'dtype':'float32', 'attn_implementation':'eager', 'memory_dim':96,
                'attention_dim':64, 'num_heads':4})
            if kind == 'invalid-real':
                recipe['stem']['unregistered_field'] = 1
        (target/'config.json').write_text(json.dumps(recipe), encoding='utf-8')
        manifest=json.loads((target/'manifest.json').read_text())
        manifest['inputs']['config.json']=sha(target/'config.json')
        (target/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        probe = r'''
import json, os, runpy, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[2])
tempfile.tempdir=os.getcwd()
import torch
from experiments.memory.prototype import reader_model
from experiments.cloud.lsr_jobs import gemma4_receiver
recipe=json.loads(Path('config.json').read_text())
kind=sys.argv[3]
class LoaderReached(Exception): pass
def no_gpu(*a, **k): raise AssertionError('GPU operation attempted')
torch.cuda._lazy_init=no_gpu
def stop(cfg, *a):
    expected='' if kind=='tiny-random' else 'operator-visibility-sentinel'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==expected, 'outer visibility mismatch'
    assert cfg==(recipe if kind=='tiny-random' else recipe['stem']), 'receiver config changed'
    assert os.environ['HF_HUB_OFFLINE']=='1' and os.environ['TRANSFORMERS_OFFLINE']=='1'
    raise LoaderReached()
if kind=='tiny-random': reader_model.load_handle=stop
else: gemma4_receiver.load_receiver=stop
script=sys.argv[1]
sys.argv=[script]
try:
    runpy.run_path(script,run_name='__main__')
except LoaderReached:
    assert kind!='invalid-real', 'invalid real recipe reached loader'
    print('PASS outer loader config/visibility '+kind)
except ValueError as exc:
    assert kind=='invalid-real' and 'recipe fields differ' in str(exc), str(exc)
    print('PASS invalid real recipe rejected before loader')
else: raise AssertionError('raising loader mock was bypassed')
'''
        environment=dict(os.environ, CUDA_VISIBLE_DEVICES='operator-visibility-sentinel')
        result=subprocess.run([sys.executable,'-B','-I','-c',probe,str(script),
                              str(Path(__file__).resolve().parents[3]),kind],
                             cwd=target,env=environment,capture_output=True,text=True,encoding='utf-8',timeout=120)
        (target/'device_stdout.log').write_text(result.stdout,encoding='utf-8')
        (target/'device_stderr.log').write_text(result.stderr,encoding='utf-8')
        if result.returncode or 'PASS ' not in result.stdout:
            raise AssertionError('outer device policy mock failed: '+result.stderr[-1800:])
        checks[kind]=True
    return checks


def reader_boundary(cfg, artifact, out, _test_stager=True):
    from . import semantic
    from experiments.cloud.lsr_jobs import s0_envelope as env
    handle=semantic.load_handle(cfg)
    bound=json.loads((Path(artifact)/'result.json').read_text())
    if semantic.state_hash(handle['receiver'].stem) != bound['fit']['frozen_stem_sha256']:
        raise ValueError('connector artifact stem identity mismatch')
    if sha(Path(artifact)/'connector.safetensors') != bound['connector_sha256']:
        raise ValueError('connector artifact hash mismatch')
    from safetensors.torch import load_file
    handle['connector'].load_state_dict(load_file(str(Path(artifact)/'connector.safetensors')))
    docs=semantic.slice_documents('dev',2,cfg['seed'])
    native=semantic.states(handle,docs)
    doc=docs[0]
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    for number,(op,snapshot) in enumerate((('late_target','v1'),('revised_target','v2'),('historical','v1'))):
        read=next(r for r in doc['reads'] if r['op']==op)
        memory=native[doc['key']][snapshot]
        directory=out/f'{number:02d}'
        script=stage(directory,cfg,Path(artifact)/'connector.safetensors',memory,read['query']['text'])
        answer=child(script,directory)
        direct=semantic.generate(handle,read['query']['text'],memory)
        if answer['output'] != direct:
            raise AssertionError('fresh process/recompute mismatch')
        blocked = [x['path'] for x in answer['opens'] if not x['allowed']]
        if blocked != answer['denied']:
            raise AssertionError('denied attempts differ from blocked audit events')
        if any(n.rsplit('.',1)[-1] in {'data','semantic','writer'} for n in answer['imports']):
            raise AssertionError('generator imported by reader')
        records.append({'op':op,'answer':answer['output']['generated'],'expected':read['answer'],
                        'opens':len(answer['opens']),'all_rejected_attempts_blocked':blocked==answer['denied'], 'blocked_attempts':len(blocked), 'imports':answer['imports']})
    directory=out/'03'
    memory=native[doc['key']]['v1']
    script=stage(directory,cfg,Path(artifact)/'connector.safetensors',memory,doc['reads'][0]['query']['text'])
    probe=child(script,directory,'--probe-forbidden-open')
    if not probe['forbidden_open_rejected']:
        raise AssertionError('source reopening not rejected')
    # Reseal a malicious numeric sidecar inside otherwise valid provenance.
    # Canonical envelope accepts arbitrary locator strings, so the reader MUST
    # independently compare metadata rather than just trust a valid digest.
    origins=copy.deepcopy(memory['meta']['origin'])
    origins[0]['locator']='encoded:1:2:3:4'
    altered=env.make_envelope(env.to_torch(memory['payload']),origin=origins,
                  temporal_horizon=memory['meta']['temporal_horizon'],expected_width=handle['stem_info']['D_memory'],
                  expected_schema=env.WIDTH_SCHEMA_VERSION)
    bad_dir=out/'04'
    stage(bad_dir,cfg,Path(artifact)/'connector.safetensors',altered,doc['reads'][0]['query']['text'])
    bad=subprocess.run(_reader_command(script),cwd=bad_dir,capture_output=True,
                       text=True,encoding='utf-8',timeout=120)
    (bad_dir/'child_stderr.log').write_text(bad.stderr,encoding='utf-8')
    if bad.returncode==0 or 'value-free metadata differs' not in bad.stderr:
        raise AssertionError('resealed encoded provenance not rejected: '+bad.stderr[-700:])
    import_dir=out/'05'
    stage(import_dir,cfg,Path(artifact)/'connector.safetensors',memory,doc['reads'][0]['query']['text'])
    import_probe=child(script,import_dir,'--probe-forbidden-import')
    if not import_probe['forbidden_import_rejected']:
        raise AssertionError('generator import not rejected')
    # These carry recoverable source facts in structurally valid integer/ID
    # fields. Recompute both envelope digest and manifest: hashes alone cannot
    # establish value-free metadata.
    fact=b'value is amber here'
    numeric=int.from_bytes(fact,'big')
    assert numeric.to_bytes(len(fact),'big') == fact
    attacks={}
    for channel in ('numeric-budget','source-id','numeric-time'):
        origins=copy.deepcopy(memory['meta']['origin'])
        if channel=='source-id':
            origins[0]['source_sha256']=fact.hex().ljust(64,'0')
        if channel=='numeric-time':
            origins[0]['event_time']=-numeric
        altered=env.make_envelope(env.to_torch(memory['payload']),origin=origins,
                    temporal_horizon=memory['meta']['temporal_horizon'],
                    latent_steps=numeric if channel=='numeric-budget' else 0,
                    expected_width=handle['stem_info']['D_memory'],expected_schema=env.WIDTH_SCHEMA_VERSION)
        attack_dir=out/channel
        stage(attack_dir,cfg,Path(artifact)/'connector.safetensors',altered,doc['reads'][0]['query']['text'])
        result=subprocess.run(_reader_command(script),cwd=attack_dir,capture_output=True,
                              text=True,encoding='utf-8',timeout=120)
        (attack_dir/'child_stderr.log').write_text(result.stderr,encoding='utf-8')
        if result.returncode==0 or 'value-free metadata differs' not in result.stderr:
            raise AssertionError('resealed '+channel+' not rejected: '+result.stderr[-700:])
        attacks[channel]={'exit_code':result.returncode,'rejected':True}
    device_checks=reader_device_policy(script,out/'00',out)
    if _test_stager:
        from unittest.mock import patch
        original_stage=stage
        calls=[]
        def leaking_stage(directory,cfg,connector,memory,question):
            calls.append(str(directory))
            leaked=env.make_envelope(env.to_torch(memory['payload']),origin=memory['meta']['origin'],
                    temporal_horizon=memory['meta']['temporal_horizon'],latent_steps=numeric,
                    expected_width=handle['stem_info']['D_memory'],expected_schema=env.WIDTH_SCHEMA_VERSION)
            return original_stage(directory,cfg,connector,leaked,question)
        with patch.dict(globals(),stage=leaking_stage):
            try:
                reader_boundary(cfg,artifact,out/'leaking-stager',_test_stager=False)
            except AssertionError as exc:
                if len(calls)!=1 or 'reader exit' not in str(exc) or 'value-free metadata differs' not in str(exc):
                    raise AssertionError('stager mutant failed for an unrelated reason') from exc
            else:
                raise AssertionError('leaking stager received false PASS')
    # Third process alone scores answers; no scorer inputs enter the reader.
    score=subprocess.run([sys.executable,'-B','-m','experiments.memory.prototype.regressions','--check','score-stdin'],
                  input=json.dumps(records),capture_output=True,text=True,encoding='utf-8',timeout=20)
    if score.returncode:
        raise AssertionError(score.stderr)
    return {'status':'PASS','fresh_readers':records,'third_process_score':json.loads(score.stdout),
            'forbidden_open_rejected':True,'forbidden_import_rejected':True,'resealed_encoded_metadata_rejected':True,
            'resealed_channels':attacks,'leaking_stager_rejected':_test_stager,'outer_device_policy':device_checks,
            'scope':'source-excluded tiny random generation mechanics; semantic correctness not required'}


def production_contract(cfg):
    import torch
    from . import semantic, writer
    from experiments.cloud.lsr_jobs import s0_envelope as env, s0_memory_port as port
    handle=semantic.load_handle(cfg)
    docs=semantic.slice_documents('train',16,cfg['seed'])
    assert len(docs)==16
    native=semantic.states(handle,docs)
    a,b=docs[:2]
    ma,mb=native[a['key']]['v1'],native[b['key']]['v1']
    checks={'exactly_16_train_documents':len(docs)==16,'paired_metadata_identical':ma['meta']==mb['meta'],
            'donor_digest_recomputed':ma['digest']!=mb['digest']}
    for memory in (ma,mb):
        env.verify_envelope(memory,expected_width=64,expected_schema=env.WIDTH_SCHEMA_VERSION)
    # Same fixed query selects the donor binding, not recipient/another slot.
    correct=total=0
    dev=semantic.slice_documents('dev',128,cfg['seed'])
    for d in dev:
        partner=next(x for x in dev if x['key']==d['key'][:-1]+('b' if d['key'][-1]=='a' else 'a'))
        read=d['reads'][0]
        wanted=next(r['value'] for r in partner['document']['records'] if r['slot']==read['query']['slot'])
        donor_read=partner['reads'][0]
        correct+=wanted==donor_read['answer']
        total+=1
    checks['competent_fixed_query_donor_128']=correct==total==128
    old_rows=env.to_torch(ma['payload'])
    revised=env.to_torch(native[a['key']]['v2']['payload'])
    rid=a['document']['revision']['record_id']
    keep=[i for i in range(16) if i//4!=rid]
    checks['unaffected_rows_identical']=torch.equal(old_rows[keep],revised[keep])
    checks['revision_changes_target']=not torch.equal(old_rows[rid*4:(rid+1)*4],revised[rid*4:(rid+1)*4])
    # Writer signature has no question/answer channel; records only.
    import inspect
    checks['writer_question_blind_signature']=list(inspect.signature(writer.ContextualWriter.write).parameters)==['self','records']
    inputs=semantic.answer_batch(handle,[(a['reads'][0]['query']['text'],a['reads'][0]['answer'])])
    for name,memory in [('ragged',torch.zeros(1,17,64)),('wrong_width',torch.zeros(1,16,63))]:
        try:
            port.forward(handle,inputs,memory)
            checks[name+'_rejected']=False
        except ValueError:
            checks[name+'_rejected']=True
    try:
        port.forward(handle,dict(inputs,use_cache=True),ma)
        checks['training_cache_rejected']=False
    except ValueError:
        checks['training_cache_rejected']=True
    # A forged PASS flag cannot substitute for a separate bound receipt.
    with tempfile.TemporaryDirectory() as tmp:
        smoke=Path(tmp)/'smoke.json'
        smoke.write_text(json.dumps({'status':'FAIL'}))
        try:
            semantic.fit(handle,cfg,docs,native,smoke)
            checks['failed_smoke_blocks_fit']=False
        except ValueError:
            checks['failed_smoke_blocks_fit']=True
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
            'donor_oracle':[correct,total]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',required=True)
    parser.add_argument('--config',default='experiments/memory/prototype/contract.json')
    parser.add_argument('--artifact')
    parser.add_argument('--out')
    args=parser.parse_args()
    if args.check=='score-stdin':
        rows=json.load(sys.stdin)
        print(json.dumps({'correct':sum(r['answer']==r['expected'] for r in rows),'total':len(rows)}))
        return 0
    if args.check not in ('production-contract','reader-boundary'):
        print('UNKNOWN_CHECK')
        return 2
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cfg=json.loads(Path(args.config).read_text())
    result=production_contract(cfg) if args.check=='production-contract' else reader_boundary(cfg,args.artifact,args.out)
    print(json.dumps(result),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
