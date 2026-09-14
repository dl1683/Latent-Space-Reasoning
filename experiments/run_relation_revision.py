"""Fixed S65 direct-revision consumer probe; supervised CPU execution only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/relation_revision_v1'
WORK = ROOT / 'internal/relation_revision_cache'
ATTEMPT = 1
RECEIPT = OUT / 'attempt.json'
RESULT = OUT / 'result.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def worker():
    import importlib.metadata
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for _ in range(100):
        receipt = read(RECEIPT)
        if receipt['status'] == 'running':
            break
        time.sleep(.01)
    assert receipt['supervisor_pid'] == os.getppid() and receipt['status'] == 'running'
    design = read(OUT/'design.json')
    assert receipt['design_sha256'] == sha(OUT/'design.json')
    torch.set_num_threads(design['cpu_threads'])
    torch.set_num_interop_threads(1)
    snapshot = Path(design['snapshot'])
    for filename, digest in design['model_file_sha256'].items():
        assert sha(snapshot/filename) == digest, filename
    result = dict(design_sha256=receipt['design_sha256'], status='loading', rows=[],
                  versions={k: importlib.metadata.version(k) for k in ['torch','transformers','tokenizers']},
                  forward_calls=0)
    save(RESULT, result)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model, loading = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
        dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
    assert not loading.get('missing_keys') and not loading.get('mismatched_keys'), loading
    model.eval()
    assert all(p.device.type == 'cpu' for p in model.parameters())
    loading = {k: sorted(v) if isinstance(v, set) else v for k, v in loading.items()}
    result.update(loading_info=loading, parameters=sum(p.numel() for p in model.parameters()),
                  status='generating')
    save(RESULT, result)  # Validate metadata serialization before any generation.
    def count(module, args):
        result['forward_calls'] += 1
    handle = model.register_forward_pre_hook(count)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    with torch.inference_mode():
        for row in design['requests']:
            started = time.monotonic()
            rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                         add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False)
            assert inputs['input_ids'].shape[1] <= design['max_input_tokens']
            before = result['forward_calls']
            output = model.generate(**inputs, do_sample=False,
                max_new_tokens=design['max_new_tokens'], pad_token_id=tokenizer.pad_token_id)
            new = output[0, inputs['input_ids'].shape[1]:].tolist()
            result['rows'].append(dict(id=row['id'], case_id=row['case_id'], arm=row['arm'],
                rendered_prompt=rendered, input_ids=inputs['input_ids'][0].tolist(),
                output_ids=new, output=tokenizer.decode(new, skip_special_tokens=True),
                terminated=bool(new and new[-1] in eos), forward_calls=result['forward_calls']-before,
                wall_seconds=time.monotonic()-started))
            save(RESULT, result)
            print(row['id'], flush=True)
    handle.remove()
    result['status'] = 'complete'
    save(RESULT, result)


def supervise():
    import psutil
    assert not RECEIPT.exists(), 'No automatic repeat'
    design = read(OUT/'design.json')
    WORK.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='starting', supervisor_pid=os.getpid(), started_unix=time.time(),
                   design_sha256=sha(OUT/'design.json'), runner_sha256=sha(Path(__file__)))
    save(RECEIPT, receipt)
    started = time.monotonic()
    suffix = '' if ATTEMPT == 1 else '_retry'
    with (WORK/f'stdout{suffix}.log').open('w') as stdout, (WORK/f'stderr{suffix}.log').open('w') as stderr:
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
        child = subprocess.Popen([sys.executable, __file__, 'worker', '--attempt', str(ATTEMPT)], env=env,
            stdout=stdout, stderr=stderr, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        receipt.update(status='running', child_pid=child.pid, peak_rss_bytes=0)
        save(RECEIPT, receipt)
        while child.poll() is None:
            try:
                rss = psutil.Process(child.pid).memory_info().rss
                receipt['peak_rss_bytes'] = max(receipt['peak_rss_bytes'], rss)
            except psutil.NoSuchProcess:
                continue
            if time.monotonic()-started > design['deadline_seconds'] or rss > design['max_rss_bytes']:
                receipt['limit'] = 'time' if time.monotonic()-started > design['deadline_seconds'] else 'memory'
                child.kill()
                break
            time.sleep(.1)
        code = child.wait()
    receipt.update(exit_code=code, wall_seconds=time.monotonic()-started,
                   status='terminated' if 'limit' in receipt else ('complete' if code == 0 else 'failed'))
    save(RECEIPT, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['supervise','worker'])
    parser.add_argument('--attempt', type=int, choices=[1,2], default=1)
    args = parser.parse_args()
    ATTEMPT = args.attempt
    if ATTEMPT == 2:
        RECEIPT, RESULT = OUT/'attempt_retry.json', OUT/'result_retry.json'
    globals()[args.phase]()
