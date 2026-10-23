"""Four fixed full-dimension scalar maps on the attention probe's saved vectors.

CPU scoring only; no model, fitting or corpus re-encoding. Source snapshots are
copied/generated probe blocks, not a claim of original-document completeness.
"""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from experiments.memory.compressed_routing_map import pack_rows,unpack_rows,normalize

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(root):
    root=Path(root);out=root/'map_curve';out.mkdir(exist_ok=False)
    resource=json.loads((root/'panel_resources.json').read_text())
    assert resource['status']=='COMPLETED' and resource['cleanup']
    measurements=json.loads((root/'panel_measurements.json').read_text())
    assert sha(root/'panel_measurements.json')==resource['artifact_sha256']['panel_measurements.json']
    assert sha(root/'dense_vectors.npz')==measurements['dense_vectors_sha256']
    assert sha(root/'dense.json')==resource['artifact_sha256']['dense.json']
    data=json.loads((root/'data_v3/reader.json').read_text())
    labels=json.loads((root/'data_v3/labels.json').read_text())['questions']
    with np.load(root/'dense_vectors.npz',allow_pickle=False)as z:
        bank=z['bank'];owners=z['owners'];queries=z['queries'];ids=z['question_ids'].tolist()
    assert ids==[q['id']for q in labels] and bank.shape[1]==queries.shape[1]==384
    assert len(bank)==len(owners) and len(queries)==60 and np.isfinite(bank).all() and np.isfinite(queries).all()
    dense=json.loads((root/'dense.json').read_text())
    assert dense['execution_id']==resource['execution_id'] and dense['pins_sha256']==resource['pins_sha256']
    # Exact reference pointers name immutable copied/generated source snapshots.
    payload=bytearray();spans=[];hashes=[]
    for text in data['blocks']:
        raw=text.encode('utf8');start=len(payload);payload.extend(raw)
        spans.append([start,len(payload)]);hashes.append(list(hashlib.sha256(raw).digest()))
    leaves=out/'leaves.bin';leaves.write_bytes(payload)
    np.savez(out/'pointers.npz',spans=np.asarray(spans,dtype=np.uint64),hashes=np.asarray(hashes,dtype=np.uint8),source_version=np.frombuffer(hashlib.sha256(payload).digest(),dtype=np.uint8))
    pointer_bytes=(out/'pointers.npz').stat().st_size
    def scores(values,query):
        window=values@query
        result=np.full(60,-2.,dtype=np.float32)
        np.maximum.at(result,owners,window)
        return result
    reference=[scores(bank,q)for q in queries]
    reproduction=max(float(np.max(np.abs(x-np.asarray(dense['scores'][key]))))for key,x in zip(ids,reference))
    assert reproduction<=2e-5,'saved bank does not reproduce measured GPU dense scores'
    specs=['fp32',1,2,4,8];rows=[]
    for bits in specs:
        started=time.perf_counter();name='fp32'if bits=='fp32'else f'bits{bits}'
        path=out/(name+'.npz')
        if bits=='fp32':np.savez(path,bank=bank,owners=owners)
        else:
            packed,scale=pack_rows(bank,bits);np.savez(path,codes=packed,scales=scale,owners=owners)
        setup=time.perf_counter()-started
        timings=[];ranks=[];checks=0;fetched=0
        with np.load(path,allow_pickle=False)as state:
            for query in queries:
                start=time.perf_counter()
                # This small reference implementation reconstructs map coordinates
                # per query and charges that work; it never re-encodes source text.
                values=state['bank']if bits=='fp32'else normalize(unpack_rows(state['codes'],state['scales'],384,bits))
                order=np.argsort(-scores(values,query),kind='stable')[:3].tolist()
                timings.append(time.perf_counter()-start);ranks.append(order)
                with leaves.open('rb')as source:
                    for block in order:
                        a,b=spans[block];source.seek(a);raw=source.read(b-a)
                        assert list(hashlib.sha256(raw).digest())==hashes[block]
                        checks+=1;fetched+=len(raw)
        totals={split:{tier:sum(q['gold_block']in ranks[i]for i,q in enumerate(labels)if q['split']==split and q['tier']==tier)for tier in 'ABC'}for split in ['dev','test']}
        rows.append(dict(bits=bits,map_file_bytes=path.stat().st_size,total_map_and_pointer_bytes=path.stat().st_size+pointer_bytes,bytes_per_block=(path.stat().st_size+pointer_bytes)/60,recall3_counts=totals,n_per_tier_split=10,verification_passed=checks,verification_proposals=checks,verified_leaf_bytes_fetched=fetched,mean_verified_leaf_bytes_per_query=fetched/60,setup_seconds=setup,median_scoring_seconds=float(np.median(timings)),ranks=ranks,map_sha256=sha(path)))
    baseline=rows[0]['recall3_counts']
    candidates=[r for r in rows[1:]if all(r['recall3_counts']['dev'][t]>=baseline['dev'][t]for t in 'ABC')]
    chosen=min(candidates,key=lambda r:r['total_map_and_pointer_bytes'])if candidates else None
    # Stored pointer hashes are byte checks only; one corrupted leaf control.
    raw=bytes(payload[spans[0][0]:spans[0][1]]);changed=bytes([raw[0]^1])+raw[1:]
    assert list(hashlib.sha256(changed).digest())!=hashes[0]
    result=dict(scope='C5/C8 routing and C7 byte-verification supporting SLICE',condition='60x180 Phi-token copied/generated blocks; MiniLM384 vectors; same60questions; four preregistered scalar widths',rows=rows,dev_selected_bits=chosen['bits']if chosen else None,selection='smallest tested file retaining FP32 DEV counts in everytier; TEST not used to choose',source_leaf_bytes=len(payload),pointer_file_bytes=pointer_bytes,pointer_sha256=sha(out/'pointers.npz'),source_version_sha256=sha(leaves),dense_vector_sha256=sha(root/'dense_vectors.npz'),dense_cpu_gpu_max_error=reproduction,corrupt_leaf_control_rejected=True,encoder_training=False,corpus_encoder_passes=1,model_answer_parity='UNMEASURED for routed leaves; full-context reader ceiling is a separate diagnostic',model_calls_saved='UNMEASURED',model_tokens_saved='UNMEASURED; native cache incumbent avoids the same corpus prefill',semantic_absence='UNKNOWN',ambiguity_exploration='UNRUN task draft separately; not graded by exact needle scoring',costs='Includes actual NPZ headers, scales, owners and pointer NPZ; excludes source-leaf bytes reported separately and inherited90.9MB encoder weights; encoder/setup complete stage timing unavailable, whole source/resume phases charged separately',runtime='CPU reference reconstructs map coordinates per query; transient N*384 FP32 values counted conceptually, no persistent exact bank for packed lookup; loaded calibration bank exists only in this experiment process',verification='Fetch exact copied/generated snapshot span and SHA256; pass is byte identity, not relevance/truth/authority/completeness',execution_id=resource['execution_id'],pins_sha256=resource['pins_sha256'],numpy_version=np.__version__)
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf8');return result
if __name__=='__main__':print(json.dumps(run(sys.argv[1])))
