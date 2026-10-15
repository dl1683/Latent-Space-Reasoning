"""Isolated frozen-stem reader. Fixed filenames; audited, hashed file allowlist.

Run with python -B -I /absolute/path/reader.py from a fresh directory. The
parent must stage five fixed input files and manifest.json. No oracle/source
or previous answers are staged. This is Python-level enforcement for the
pinned trusted implementation, not an OS sandbox against hostile native code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def validate_recipe(recipe):
    """Validate the real recipe before preserving any operator GPU visibility."""
    import math
    import re
    if set(recipe) != {'kind', 'stem', 'tokenizer_specials'}:
        raise ValueError('reader recipe fields differ')
    specials = recipe['tokenizer_specials']
    if set(specials) != {'pad_token', 'eos_token', 'unk_token'} or not all(isinstance(v, str) for v in specials.values()):
        raise ValueError('reader tokenizer specials differ')
    if recipe['kind'] == 'tiny-random':
        return
    if recipe['kind'] != 'staged-real':
        raise ValueError('unknown reader stem kind')
    cfg = recipe['stem']
    required = {'model', 'revision', 'tokenizer_revision', 'model_local', 'allow_hf_download',
                'weights_manifest_sha256', 'placement', 'attn_implementation', 'dtype'}
    if not required <= set(cfg) or set(cfg) - required - {'memory_dim', 'attention_dim', 'num_heads'}:
        raise ValueError('real reader recipe fields differ')
    if cfg['model'] not in {'google/gemma-4-31B', 'google/gemma-4-E2B', 'google/gemma-4-E4B'}:
        raise ValueError('unadmitted real reader model')
    for key in ('revision', 'tokenizer_revision', 'weights_manifest_sha256'):
        size = 64 if key == 'weights_manifest_sha256' else 40
        if not isinstance(cfg[key], str) or re.fullmatch('[0-9a-f]{%d}' % size, cfg[key]) is None:
            raise ValueError('real reader requires immutable pins')
    if cfg['model'] == 'google/gemma-4-31B' and cfg['revision'] != '5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89':
        raise ValueError('wrong primary revision')
    if cfg['allow_hf_download'] is not False or not Path(cfg['model_local']).is_absolute():
        raise ValueError('real reader requires absolute local mirror and disabled downloads')
    if cfg['dtype'] not in {'float32', 'float16', 'bfloat16'} or cfg['attn_implementation'] not in {'eager', 'sdpa'}:
        raise ValueError('unsupported real reader dtype/backend')
    for key in ('memory_dim', 'attention_dim', 'num_heads'):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] <= 0):
            raise ValueError('invalid real reader dimension')
    placement = cfg['placement']
    if set(placement) != {'device_map', 'measurement_id', 'min_free_fraction'}:
        raise ValueError('real reader placement fields differ')
    devices = placement['device_map']
    if not isinstance(devices, dict) or not devices or not all(
            isinstance(k, str) and isinstance(v, str) and (v == 'cpu' or re.fullmatch(r'cuda:[0-9]+', v))
            for k, v in devices.items()) or ('cpu' in devices.values() and len(set(devices.values())) > 1):
        raise ValueError('invalid real reader placement')
    headroom = placement['min_free_fraction']
    if not isinstance(placement['measurement_id'], str) or not placement['measurement_id'].strip() or \
            type(headroom) not in (int, float) or not math.isfinite(headroom) or not .1 <= headroom < 1:
        raise ValueError('unmeasured real reader placement')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-forbidden-open', action='store_true')
    parser.add_argument('--probe-forbidden-import', action='store_true')
    args = parser.parse_args()
    cwd = Path.cwd().resolve()
    repo = Path(__file__).resolve().parents[3]
    package = Path(__file__).resolve().parent
    manifest_path = cwd / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_names = {'config.json', 'connector.safetensors', 'memory.json', 'query.json', 'tokenizer.json'}
    if set(manifest['inputs']) != expected_names or set(manifest) != {'inputs', 'code'}:
        raise ValueError('reader manifest has unexpected fields/filenames')
    allowed = {manifest_path}
    for name, wanted in manifest['inputs'].items():
        path = cwd / name
        if path.is_symlink() or path.resolve().parent != cwd:
            raise ValueError('reader input escapes isolated directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
            raise ValueError('reader input hash mismatch')
        allowed.add(path.resolve())
    sys.path.insert(0, str(repo))
    required_code = [package / n for n in ('__init__.py', 'reader.py', 'reader_model.py')]
    required_code += [repo / 'experiments' / '__init__.py', repo / 'experiments' / 'memory' / '__init__.py', repo / 'experiments' / 'cloud' / '__init__.py', repo / 'experiments' / 'cloud' / 'lsr_jobs' / '__init__.py']
    required_code += [repo / 'experiments' / 'cloud' / 'lsr_jobs' / n for n in ('s0_envelope.py', 's0_memory_port.py', 'gemma4_receiver.py')]
    existing = {str(p.resolve()): p for p in required_code if p.exists()}
    if set(manifest['code']) != set(existing):
        raise ValueError('reader code roster mismatch')
    for name, path in existing.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['code'][name]:
            raise ValueError('reader code hash mismatch')
        allowed.add(path.resolve())
    # Cloud-only real stems retain the production loader's complete pinned
    # mirror contract. Permit only manifest-listed files; never a whole root.
    recipe = json.loads((cwd / 'config.json').read_text(encoding='utf-8'))
    validate_recipe(recipe)
    if recipe.get('kind') == 'staged-real':
        import re
        root = Path(recipe['stem']['model_local'])
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise ValueError('real reader requires an existing absolute local mirror')
        asset_manifest = root / '_manifest_sha256.txt'
        if hashlib.sha256(asset_manifest.read_bytes()).hexdigest() != recipe['stem']['weights_manifest_sha256']:
            raise ValueError('real reader asset manifest pin mismatch')
        allowed.update({asset_manifest.resolve(), (root / '_COMPLETE').resolve()})
        seen = set()
        for line in asset_manifest.read_text(encoding='utf-8').splitlines():
            match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
            if match is None:
                raise ValueError('malformed real reader asset manifest')
            name = match[2].removeprefix('./')
            if not name or name in seen or '\\' in name or ':' in name or any(x in ('', '.', '..') for x in name.split('/')):
                raise ValueError('unsafe real reader asset path')
            seen.add(name)
            target = root / name
            if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('real reader asset escapes local mirror')
            allowed.add(target.resolve())
    runtime = [Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()]
    # Installed wheels may live outside the interpreter root; enumerate their
    # import search directories, excluding the repository and current directory.
    for entry in sys.path:
        if entry and 'site-packages' in str(entry).lower():
            runtime.append(Path(entry).resolve())
    import tempfile
    tempfile.tempdir = str(cwd)
    opens = []
    denied = []

    def audit(event, values):
        if event == 'import' and values[0].startswith('experiments.memory.prototype.') and values[0].rsplit('.', 1)[-1] in {'data','semantic','writer','trainer','evaluate','store','resident','demo','regressions','runner','checks'}:
            raise PermissionError('reader generator import forbidden: ' + values[0])
        if event in ('socket.connect', 'subprocess.Popen', 'os.system'):
            raise PermissionError('reader external effects denied')
        if event != 'open' or isinstance(values[0], int):
            return
        path = Path(values[0]).resolve()
        flags = values[2] if len(values) > 2 else 0
        writing = isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        # Libraries probe the null device in read/write mode. It retains no
        # source or answers; permit that exact device independently of flags.
        null_device = path == Path(os.devnull).resolve()
        ok = null_device or (not writing and (path in allowed or any(path.is_relative_to(root) for root in runtime)))
        opens.append({'path': str(path), 'allowed': ok, 'write': writing})
        if not ok:
            denied.append(str(path))
            raise PermissionError('reader file is outside hashed input/code/runtime allowlist: ' + str(path))

    sys.addaudithook(audit)
    if args.probe_forbidden_import:
        try:
            __import__('experiments.memory.prototype.data')
        except PermissionError:
            print(json.dumps({'status':'PASS', 'forbidden_import_rejected':True}))
            return 0
        raise AssertionError('reader generator imported')
    if args.probe_forbidden_open:
        try:
            (cwd.parent / 'source.txt').read_text()
        except PermissionError:
            print(json.dumps({'status': 'PASS', 'forbidden_open_rejected': True, 'opens': opens}))
            return 0
        raise AssertionError('forbidden source opened')
    if recipe['kind'] == 'tiny-random':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(repo))
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    from experiments.memory.prototype import reader_model
    from experiments.cloud.lsr_jobs import s0_envelope as env
    from safetensors.torch import load_file
    handle = reader_model.load_handle(recipe, cwd / 'tokenizer.json')
    handle['connector'].load_state_dict(load_file(str(cwd / 'connector.safetensors')), strict=True)
    memory = env.deserialize_envelope((cwd / 'memory.json').read_text(encoding='utf-8'),
                 expected_width=handle['stem_info']['D_memory'], expected_schema=env.WIDTH_SCHEMA_VERSION)
    # Metadata is exactly independently generated address-only provenance.
    expected = [{'origin_type':'observation', 'modality':'text',
          'source_sha256':hashlib.sha256(f'address:{i//4}'.encode()).hexdigest(),
          'locator':f'record:{i//4}:row:{i%4}', 'event_time':0, 'ingestion_time':0,
          'valid_from':0, 'valid_until':None, 'branch':'main', 'dependencies':[], 'permissions':['read']}
          for i in range(16)]
    expected_meta = {'schema':env.WIDTH_SCHEMA_VERSION, 'type':'memory-data/1', 'branch':'main',
                     'origin':expected, 'n_positions':16, 'width':handle['stem_info']['D_memory'],
                     'temporal_horizon':{'as_of':0,'max_age':None}, 'compute_budget':{'latent_steps':0}}
    # Compare the complete closed schema, including JSON scalar types (False is
    # not the canonical integer 0). The envelope's structural/hash check ran above.
    if json.dumps(memory['meta'], sort_keys=True, allow_nan=False) != json.dumps(expected_meta, sort_keys=True) \
            or memory['mask'] != [1]*16 or memory['index_map'] != list(range(16)):
        raise ValueError('value-free metadata differs from independent address schema')
    query = json.loads((cwd / 'query.json').read_text(encoding='utf-8'))
    if set(query) != {'question', 'budget'} or not isinstance(query['question'], str) \
            or type(query['budget']) is not int or query['budget'] not in (8, 16):
        raise ValueError('reader query schema violation')
    output = reader_model.generate(handle, query['question'], memory, budget=query['budget'])
    roster = sorted(n for n in sys.modules if n.startswith('experiments.'))
    if any(n.startswith('experiments.memory.prototype.') and n.rsplit('.',1)[-1] in reader_model.FORBIDDEN_MODULES for n in roster):
        raise AssertionError('generator module in reader import roster')
    print(json.dumps({'status':'PASS', 'output':output, 'opens':opens, 'denied':denied,
                      'imports':roster, 'scope':handle['stem_info'].get('scope', 'staged-real') + ', trusted-code Python file-open boundary'}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
