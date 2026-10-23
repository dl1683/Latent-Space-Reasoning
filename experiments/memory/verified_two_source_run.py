"""Consume prepared verified-passage requests under the existing GPU resource_guard.

This is a source-use diagnostic, not a retrieval or memory-quality benchmark.
No model or CUDA initialization occurs on import. RUN_DIR PHASE is the same
worker interface as the attention owner's controller; this file creates no lease.
A stdlib-only terminal seal records partial-output custody on NORMAL and
EXCEPTION exits; it is not a parent attestation, and a hard-killed worker
leaves no seal behind.
"""
from pathlib import Path
import base64
import hashlib
import json
import math
import os
import re
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _reject_constant(value):
    raise ValueError('nonfinite JSON constant: ' + value)


def thermal_json(path, *, observation=None):
    """Tolerate only a brief Windows sharing conflict; freshness checks follow."""
    until = time.monotonic() + .075
    attempts = 0
    while True:
        started = time.monotonic()
        attempts += 1
        if observation is not None:
            observation.setdefault('first_read_started', started)
            observation.update(read_started=started, read_attempts=attempts)
        try:
            try:
                raw = Path(path).read_bytes()
            except Exception as error:
                if observation is not None:
                    observation.update(read_failed_at=time.monotonic(),
                        read_error=type(error).__name__ + ': ' + str(error),
                        raw_bytes_available=False)
                raise
            finished = time.monotonic()
            if observation is not None:
                observation.update(read_started=started, read_finished=finished,
                    raw_bytes_available=True,
                    raw_base64=base64.b64encode(raw).decode('ascii'),
                    raw_sha256=hashlib.sha256(raw).hexdigest())
            return json.loads(raw.decode('utf8'), parse_constant=_reject_constant)
        except PermissionError:
            if time.monotonic() >= until:
                raise
            time.sleep(.002)


def thermal_predicates(row, phase, execution, gpu_uuid, checked_at):
    """Original physical-row limits; malformed numeric inputs now fail closed."""
    for name, value in [('checked_at', checked_at)] + [
            (key, row[key]) for key in ('monotonic', 'vendor_age', 'temperature')]:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('invalid finite numeric field: ' + name)
    if row['temperature'] < 0 or row['vendor_age'] < 0:
        raise ValueError('negative temperature or vendor age')
    age = checked_at - row['monotonic'] + row['vendor_age']
    return dict(phase=row['phase'] != phase,
                execution=row['execution_id'] != execution,
                uuid=row['uuid'] != gpu_uuid,
                freshness=not 0 <= age <= 1.25,
                temperature=row['temperature'] >= 72), age


def checked_thermal(path, phase, execution, gpu_uuid, *, rejection_path,
                    pins_sha256=None):
    """Capture rejected read bytes and clocks; never accept a failed predicate.

    No rejection I/O occurs on the passing path. Recording failure still raises;
    this is diagnostic custody, not a new resource policy or a retry of stale data.
    """
    observation = dict(path=str(Path(path)), phase=phase, execution_id=execution,
        pins_sha256=pins_sha256, worker_pid=os.getpid(), expected_uuid=gpu_uuid,
        maximum_effective_age=1.25, stop_temperature=72)
    try:
        row = thermal_json(path, observation=observation)
        checked_at = time.monotonic()
        observation.update(parsed=row, checked_at=checked_at)
        failed, age = thermal_predicates(row, phase, execution, gpu_uuid, checked_at)
        observation.update(parsed=row, checked_at=checked_at, effective_age=age,
            failed_predicates=failed, expected_uuid=gpu_uuid,
            maximum_effective_age=1.25, stop_temperature=72)
        if not any(failed.values()):
            return row
        reason = 'thermal identity/freshness/stop: ' + ','.join(
            name for name, value in failed.items() if value)
    except Exception as error:
        observation.update(evaluation_error=type(error).__name__ + ': ' + str(error))
        reason = 'thermal read/evaluation rejected: ' + str(error)
    try:
        with checked_output_path(rejection_path).open('a', encoding='utf8') as output:
            output.write(json.dumps(observation, sort_keys=True) + '\n')
    except Exception as error:
        raise RuntimeError(reason + '; rejection record failed: ' + str(error)
                           + '; rejected_frame_base64='
                           + observation.get('raw_base64', 'UNAVAILABLE')) from error
    raise RuntimeError(reason)


def format_pass(row, expected):
    """Literal output/ID check only; semantic judgment is a separate task."""
    if (expected.get('literal_check_eligible', True) is not True
            or not isinstance(expected.get('answer'), str)):
        return False
    text = row['answer']
    if not row['terminated'] or not isinstance(text, str) or not text.strip():
        return False
    if expected['answer'] == 'INSUFFICIENT_EVIDENCE':
        return text.strip().rstrip('.') == 'INSUFFICIENT_EVIDENCE'
    cited = set(re.findall(r'\bP\d+\b', text))
    phrase = re.search(r'(?<!\w)' + re.escape(expected['answer'].casefold())
                       + r'(?!\w)', text.casefold())
    return (phrase is not None
            and cited == set(expected['citations'])
            and 'INSUFFICIENT_EVIDENCE' not in text)


def literal_verdict(root, phase, resource):
    """Host callback for the existing controller; never launches a worker.

    A passing literal verdict still needs the independent evaluator's source-use judgment before
    the conditional panel. Attention's score-array smoke verdict is unrelated.
    """
    root = Path(root)
    measured_path = root / (phase + '_measurements.json')
    measured = json.loads(measured_path.read_text(encoding='utf8'))
    pins = json.loads((root / 'pins.json').read_text(encoding='utf8'))
    for name in ('requests.json', 'expectations.json'):
        if pins.get(str((root / name).resolve())) != digest(root / name):
            raise ValueError('unbound evaluation inputs')
    if (resource['status'] != 'COMPLETED' or not resource['cleanup_verified']
            or resource['execution_id'] != measured['execution_id']
            or resource['pins_sha256'] != measured['pins_sha256']
            or resource['pins_sha256'] != digest(root / 'pins.json')
            or resource['phase'] != phase or measured['phase'] != phase
            or resource['artifact_sha256'][measured_path.name] != digest(measured_path)):
        raise ValueError('no matching terminal operation receipt')
    requests = json.loads((root / 'requests.json').read_text(encoding='utf8'))
    selected = [r['id'] for r in requests if r['phase'] == phase]
    rows = measured['rows']
    if not selected or [r['id'] for r in rows] != selected:
        raise ValueError('missing, reordered or duplicated operation rows')
    expected = json.loads((root / 'expectations.json').read_text(encoding='utf8'))
    checks = [dict(id=r['id'], literal_pass=format_pass(r, expected[r['id']])) for r in rows]
    return dict(passed_literal_only=all(r['literal_pass'] for r in checks), checks=checks,
                terminated=sum(r['terminated'] for r in rows),
                empty=sum(not r['answer'].strip() for r in rows),
                distinct_answers=len({r['answer'] for r in rows}),
                semantic_judgment='PENDING_EVALUATOR_NOT_ESTABLISHED_BY_LITERAL_CHECKS',
                model_calls=sum(r['model_calls'] for r in rows),
                input_tokens=sum(r['input_tokens'] for r in rows),
                output_tokens=sum(r['output_tokens'] for r in rows))


def main(root, phase):
    if os.environ.get('LSR_GPU_GUARDED') != '1':
        raise RuntimeError('existing external GPU resource_guard required')
    root = Path(root).resolve()
    execution = os.environ['LSR_EXECUTION_ID']
    pin_digest = digest(root / 'pins.json')
    if pin_digest != os.environ['LSR_PIN_DIGEST']:
        raise ValueError('resource_guard pin mismatch')
    pins = json.loads((root / 'pins.json').read_text(encoding='utf8'))
    cfg = json.loads((root / 'config.json').read_text(encoding='utf8'))
    model_root = Path(cfg['reader']).resolve()
    required = {str(Path(__file__).resolve()), str((root / 'requests.json').resolve()),
                str(Path(__file__).with_name('small_reader_gpu.py').resolve()),
                str((root / 'config.json').resolve())}
    required.update(str(p.resolve()) for p in model_root.iterdir()
                    if p.is_file() and p.suffix in ('.json', '.safetensors', '.txt', '.model'))
    if not required.issubset(pins):
        raise ValueError('worker/request/config not pinned')
    for name, expected in pins.items():
        if digest(name) != expected:
            raise ValueError('changed pinned input: ' + name)
    requests = json.loads((root / 'requests.json').read_text(encoding='utf8'))
    selected = [r for r in requests if r['phase'] == phase]
    if not selected:
        raise ValueError('no registered requests for phase')
    if any((root / (phase + suffix)).exists()
           or (root / (phase + suffix)).is_symlink()
           for suffix in ('_rows.jsonl', '_attempts.jsonl', '_measurements.json',
                          '_guard_rejections.jsonl', '_worker_terminal.json')):
        raise FileExistsError('preserve prior attempt; use a new registered run directory')

    import psutil
    import torch
    from transformers import AutoTokenizer, DynamicCache
    from experiments.memory.small_reader_gpu import stream_model

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    burst = started

    def check():
        row = checked_thermal(root / (phase + '_external_thermal.json'), phase,
            execution, os.environ['LSR_GPU_UUID'],
            rejection_path=root / (phase + '_guard_rejections.jsonl'),
            pins_sha256=pin_digest)
        available = psutil.virtual_memory().available
        if available < 20 * 2**30:
            raise MemoryError('RAM floor: available_bytes=' + str(available)
                              + ' minimum_bytes=' + str(20 * 2**30))
        return row['temperature']

    def cool():
        nonlocal burst
        temperature = check()
        if time.monotonic() - burst < 20 and temperature < 66:
            return
        torch.cuda.synchronize()
        limit = time.monotonic() + 150
        since = None
        while time.monotonic() < limit:
            temperature = check()
            if temperature <= 60:
                since = since or time.monotonic()
                if time.monotonic() - since >= 30:
                    burst = time.monotonic()
                    return
            else:
                since = None
            time.sleep(.5)
        raise TimeoutError('bounded burst cooldown')

    check()
    tokenizer = AutoTokenizer.from_pretrained(cfg['reader'], local_files_only=True)
    model = stream_model(cfg['reader'], checkpoint=cool)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    rows = []

    def record_attempt(value, request):
        with checked_output_path(root / (phase + '_attempts.jsonl')).open('a', encoding='utf8') as output:
            output.write(json.dumps(bind_context(value, request, execution, pin_digest)) + '\n')

    for request in selected:
        ids = tokenizer.apply_chat_template(request['messages'], tokenize=True,
            add_generation_prompt=True, return_dict=False)
        if (ids != request['input_ids'] or len(ids) + 128 > 12000
                or request['budget_omissions'] or request['max_new_tokens'] != 128):
            raise ValueError('prepared request changed or evidence omitted')
        cool()
        cache = DynamicCache(config=model.config)
        generated = []
        begin = time.monotonic()
        record_attempt(dict(id=request['id'], event='STARTED',
                            planned_input_tokens=len(ids),
                            interrupted_inflight_consumption='UNKNOWN'), request)
        with torch.inference_mode():
            pending = torch.tensor([ids], device='cuda')
            for _ in range(128):
                cool()
                output = model(input_ids=pending, past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
                token = int(output.logits[0, -1].argmax().item())
                generated.append(token)
                record_attempt(dict(id=request['id'], event='FORWARD_RETURNED',
                                    forward_calls=len(generated),
                                    prompt_tokens=len(ids),
                                    forward_input_tokens=int(pending.shape[-1]),
                                    output_token_id=token), request)
                del output, pending
                if token in eos:
                    break
                pending = torch.tensor([[token]], device='cuda')
        row = bind_context(dict(id=request['id'], answer=tokenizer.decode(generated, skip_special_tokens=True),
                   output_ids=generated, terminated=generated[-1] in eos,
                   input_tokens=len(ids), output_tokens=len(generated),
                   model_calls=1, forward_calls=len(generated),
                   seconds=time.monotonic()-begin), request, execution, pin_digest)
        rows.append(row)
        # Append-only per-query checkpoint survives any later thermal/resource stop.
        with checked_output_path(root / (phase + '_rows.jsonl')).open('a', encoding='utf8') as output:
            output.write(json.dumps(row) + '\n')
        del cache
        if digest(root / 'pins.json') != pin_digest:
            raise ValueError('pins changed during operation')
    with checked_output_path(root / (phase + '_measurements.json')).open('x', encoding='utf8') as output:
        json.dump(dict(phase=phase, execution_id=execution, pins_sha256=pin_digest,
                       parameters=model.lsr_parameter_count, rows=rows,
                       seconds=time.monotonic()-started,
                       scope='SUPPLIED_SOURCE_OPERATION_NOT_RETRIEVAL_OR_SAVINGS'), output)


def bind_context(value, request, execution, pins_sha256):
    """Bind partial events and completed rows to the same full request bytes."""
    return dict(value, execution_id=execution, pins_sha256=pins_sha256,
        request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest())


def _phase_component(phase):
    if (not isinstance(phase, str) or not phase
            or any(c in phase for c in '\\/:<>"|?*')
            or any(ord(c) < 32 for c in phase)):
        raise ValueError('phase must be a single filename component')
    return phase


def checked_output_path(path):
    """Refuse existing/dangling symlink outputs before opening for any write.

    Run directories are trusted and private. This is not protection against an
    adversary replacing the directory between this check and the OS open.
    """
    path = Path(path).absolute()
    if path.is_symlink() or not path.resolve().is_relative_to(path.parent.resolve()):
        raise ValueError('output symlink or path escape refused')
    return path


def seal_worker_exit(root, phase, status, error=None):
    """Child self-seal only: parent binding and hard-kill custody remain external."""
    phase = _phase_component(phase)
    root = Path(root).resolve()
    files = {}
    for name in ['pins.json', 'requests.json', 'config.json'] + [phase + suffix for suffix in (
            '_rows.jsonl', '_attempts.jsonl', '_guard_rejections.jsonl', '_measurements.json')]:
        path = root / name
        if path.is_file():
            if not path.resolve().is_relative_to(root):
                raise ValueError('terminal artifact escapes worker directory')
            files[name] = dict(sha256=digest(path), bytes=path.stat().st_size)
    receipt = dict(schema='verified_source_worker_terminal.v1', phase=phase,
        status=status, execution_id=os.environ.get('LSR_EXECUTION_ID'),
        expected_pins_sha256=os.environ.get('LSR_PIN_DIGEST'), worker_pid=os.getpid(),
        observed_pins_sha256=files.get('pins.json', {}).get('sha256'),
        error=None if error is None else dict(type=type(error).__name__, message=str(error)),
        monotonic=time.monotonic(), unix=time.time(), artifacts=files,
        scope='CHILD_SELF_SEAL_NOT_PARENT_ATTESTATION_OR_HARD_KILL_GUARANTEE')
    with checked_output_path(root / (phase + '_worker_terminal.json')).open('x', encoding='utf8') as output:
        json.dump(receipt, output, indent=2, allow_nan=False)


def run_worker(root, phase):
    """Preserve normal/exception custody; never turn an unsealed exit into success."""
    phase = _phase_component(phase)
    root = Path(root).resolve()
    if ((root / (phase + '_worker_terminal.json')).exists()
            or (root / (phase + '_worker_terminal.json')).is_symlink()):
        raise FileExistsError('preserve prior terminal seal; use a new run directory')
    try:
        main(root, phase)
    except BaseException as error:
        try:
            seal_worker_exit(root, phase, 'FAILED', error)
        except BaseException as seal_error:
            error.add_note('worker terminal seal failed: ' + str(seal_error))
        raise
    else:
        seal_worker_exit(root, phase, 'COMPLETED')


if __name__ == '__main__':
    run_worker(sys.argv[1], sys.argv[2])
