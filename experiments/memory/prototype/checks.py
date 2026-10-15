"""Decisive false-PASS and resealed persistence regressions; no neural fitting."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

from . import data, evaluate as e, store, writer


SMOKE_CHECKS = {'one_resampling_call','tensor_loss','answer_eos_only','finite_nonzero_gradients',
                'frozen_stem','content_effect','none_empty_parity','disconnected_loss_rejected'}


def production_smoke_valid(receipt):
    """Require all live measurements; a status string cannot stand in for them."""
    import math
    checks = receipt.get('checks', {})
    return (receipt.get('status') == 'PASS' and isinstance(checks, dict)
            and SMOKE_CHECKS <= set(checks) and all(checks[k] is True for k in SMOKE_CHECKS)
            and all(type(receipt.get(k)) in (int,float) and math.isfinite(receipt[k]) for k in ('noise','effect'))
            and receipt['noise'] >= 0 and receipt['effect'] > max(10*receipt['noise'], 1e-7))


def smoke_gate_regressions():
    base = {'status':'PASS','checks':dict.fromkeys(SMOKE_CHECKS,True),'noise':0.0,'effect':0.1}
    checks = {'healthy_production_smoke':production_smoke_valid(base)}
    for key in SMOKE_CHECKS:
        bad = copy.deepcopy(base)
        del bad['checks'][key]
        checks['smoke_missing_' + key] = not production_smoke_valid(bad)
        bad['checks'][key] = False
        checks['smoke_false_' + key] = not production_smoke_valid(bad)
    for key in ('noise','effect'):
        bad = copy.deepcopy(base)
        del bad[key]
        checks['smoke_missing_' + key] = not production_smoke_valid(bad)
    for name,key,value in [('nan_effect','effect',float('nan')),('negative_noise','noise',-1),
                           ('noise_dominates','noise',1),('zero_effect','effect',0)]:
        bad = copy.deepcopy(base)
        bad[key] = value
        checks['smoke_' + name] = not production_smoke_valid(bad)
    return checks


def gate_regressions():
    corpus=data.build_corpus()
    truth=data.truth_table(corpus,'test')
    donor=data.donor_truth_table(corpus,'test')
    novel={str(p['pair']) for p in corpus['splits']['test'] if p['novel']}
    rows=[]
    for arm in e.REQUIRED_ARMS:
        for (doc,op),answer in (donor if arm=='donor' else truth).items():
            word=answer if arm not in ('omission','untrained') else data.WORDS[(data.WORDS.index(answer)+1)%32]
            rows.append(dict(arm=arm,doc=doc,op=op,generated=word,terminated=True,
                tokens=[7,1],eos_token_id=1,token_count=2,max_new_tokens=8,finish_cause='eos'))
    base=e.evaluate_all(rows,truth,novel,donor)
    checks={'healthy_scoring':base['scoring_pass'],
            'no_receipts_no_final_pass':not base['all_pass'] and base['acceptance_status']=='INCOMPLETE'}
    def rejected(name,mutate):
        changed=copy.deepcopy(rows)
        mutate(changed)
        result=e.evaluate_all(changed,truth,novel,donor)
        checks[name]=not result['scoring_pass'] and not result['all_pass']
    rejected('dictionary_wrong',lambda r:[x.update(generated='wrong') for x in r if x['arm']=='dictionary'])
    rejected('missing_arm',lambda r:r.__setitem__(slice(None),[x for x in r if x['arm']!='storage']))
    rejected('empty_table',lambda r:r.clear())
    rejected('duplicate_donor',lambda r:r.append(copy.deepcopy(next(x for x in r if x['arm']=='donor'))))
    rejected('partial_donor',lambda r:r.pop(next(i for i,x in enumerate(r) if x['arm']=='donor')))
    rejected('missing_termination',lambda r:[x.pop('terminated',None) for x in r])
    rejected('eos_lie',lambda r:[x.update(tokens=[7,8]) for x in r if x['arm']=='candidate'])
    rejected('crashed_row',lambda r:r[0].update(error='crashed'))
    rejected('constant_candidate',lambda r:[x.update(generated='amber') for x in r if x['arm']=='candidate'])
    rejected('foreign_key',lambda r:r[0].update(doc='999a'))
    rejected('budget_caps',lambda r:[x.update(terminated=False,tokens=[7]*8,token_count=8,finish_cause='budget') for x in r if x['arm']=='storage'])
    checks['64_cluster_units']=len({r['doc'][:-1] for r in rows})==64 and base['summary']['storage_cluster_all']['unit']=='world_pair'
    checks['fixed_cluster_bootstrap_10000']=base['summary']['storage_cluster_all']['n_boot']==10000
    doc=corpus['splits']['train'][0]['doc_a']
    native=writer.write_document(doc)
    instance=store.MemoryStore(readers=[store.dictionary_reader()])
    sid=instance.write(doc['records'],native)
    original=instance.snapshot_bytes(sid)
    clock=instance._time
    try:
        instance.revise(sid,doc['revision']['record_id'],doc['revision'])
        checks['missing_revision_rows']=False
    except store.StoreError:
        checks['missing_revision_rows']=instance.snapshot_bytes(sid)==original and instance._time==clock
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'store.json'
        instance.save(str(path))
        raw=json.loads(path.read_text())
        attacks=[('resealed_ragged',lambda p:p['snapshots'][0]['native_state']['rows'].__setitem__(0,[1.0])),
                 ('resealed_width',lambda p:p['snapshots'][0]['native_state'].__setitem__('width',99)),
                 ('encoded_counter',lambda p:p['snapshots'][0]['native_state']['sidecars'].__setitem__('n_records',31)),
                 ('bad_version',lambda p:p['snapshots'][0].__setitem__('version',7))]
        for name,mutate in attacks:
            bad=copy.deepcopy(raw)
            mutate(bad)
            for snap in bad['snapshots']:
                snap['digest']=store._digest({k:v for k,v in snap.items() if k!='digest'})
            bad['payload_digest']=store._digest({k:v for k,v in bad.items() if k!='payload_digest'})
            attack=Path(td)/(name+'.json')
            attack.write_text(json.dumps(bad))
            before=attack.read_bytes()
            try:
                store.MemoryStore.load(str(attack))
                checks[name]=False
            except store.StoreError:
                checks[name]=attack.read_bytes()==before
    checks.update(smoke_gate_regressions())
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
            'generator_sha256':data._hash(corpus['generator']), 'scoring_gate_count':len(base['gates'])}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',required=True)
    args=parser.parse_args(argv)
    if args.check!='false-pass':
        print('UNKNOWN_CHECK')
        return 2
    result=gate_regressions()
    print(json.dumps(result))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
