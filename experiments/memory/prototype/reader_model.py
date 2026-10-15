"""Reader-only frozen stem/tokenizer and generation. No corpus or writer access."""
import json
from pathlib import Path
import torch
from experiments.cloud.lsr_jobs import s0_memory_port as port
from experiments.cloud.lsr_jobs.gemma4_receiver import Gemma4Receiver, MemoryResampler, ResamplerConfig

REAL_FIELDS = {'model','revision','tokenizer_revision','model_local','allow_hf_download','weights_manifest_sha256','placement','attn_implementation','dtype','memory_dim','attention_dim','num_heads'}

FORBIDDEN_MODULES = {'data', 'semantic', 'writer', 'trainer', 'evaluate', 'store', 'resident', 'demo', 'regressions', 'runner', 'checks'}

def code_paths(package, repo):
    paths = [package / n for n in ('__init__.py', 'reader.py', 'reader_model.py')]
    paths += [repo / 'experiments' / '__init__.py', repo / 'experiments' / 'memory' / '__init__.py', repo / 'experiments' / 'cloud' / '__init__.py', repo / 'experiments' / 'cloud' / 'lsr_jobs' / '__init__.py']
    paths += [repo / 'experiments' / 'cloud' / 'lsr_jobs' / n for n in ('s0_envelope.py', 's0_memory_port.py', 'gemma4_receiver.py')]
    return [p for p in paths if p.exists()]

def load_handle(recipe, tokenizer_path):
    if set(recipe) != {'kind', 'stem', 'tokenizer_specials'}:
        raise ValueError('reader recipe fields differ')
    from transformers import PreTrainedTokenizerFast, Gemma4ForCausalLM, Gemma4TextConfig
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **recipe['tokenizer_specials'])
    if recipe['kind'] == 'tiny-random':
        if set(recipe['stem']) != {'initialization_seed', 'config'}:
            raise ValueError('random stem recipe fields differ')
        torch.manual_seed(recipe['stem']['initialization_seed'])
        model_cfg = Gemma4TextConfig.from_dict(recipe['stem']['config'])
        model_cfg._attn_implementation = 'eager'
        with torch.device('cpu'):
            stem = Gemma4ForCausalLM(model_cfg).float().eval()
            width = model_cfg.hidden_size
            connector = MemoryResampler(ResamplerConfig(width, width, attention_dim=64, num_heads=4)).float()
        return {'tok': tok, 'receiver': Gemma4Receiver(stem, connector).eval(), 'connector': connector,
                'stem_info': {'D_memory': width, 'D_stem': width, 'scope': 'tiny-random'}}
    if recipe['kind'] != 'staged-real':
        raise ValueError('unknown reader stem kind')
    from experiments.cloud.lsr_jobs.gemma4_receiver import load_receiver
    cfg = recipe['stem']
    if set(cfg) - REAL_FIELDS:
        raise ValueError('unexpected real reader recipe fields')
    if cfg.get('allow_hf_download') is not False or not Path(cfg['model_local']).is_absolute():
        raise ValueError('real reader requires absolute local mirror and disabled downloads')
    handle = load_receiver(cfg)
    if handle['tok'].backend_tokenizer.to_str() != tok.backend_tokenizer.to_str():
        raise ValueError('staged tokenizer differs from frozen reader tokenizer')
    handle['tok'] = tok
    return handle

def prompt(question, evidence=None):
    base = '' if evidence is None else f'evidence: {evidence}\n'
    return base + question + '\nanswer:'

def generate(handle, query, memory=None, evidence=None, budget=8):
    ids = handle['tok'](prompt(query, evidence), add_special_tokens=False, return_tensors='pt')['input_ids']
    result = port.generate(handle, {'input_ids': ids}, memory,
                          budget={'max_new_tokens': budget, 'max_total_forward_calls': budget},
                          eos_ids=(handle['tok'].eos_token_id,))
    result['generated'] = handle['tok'].decode(result['tokens'], skip_special_tokens=True).strip()
    result['finish_cause'] = 'eos' if result['terminated'] else 'budget'
    result['token_count'] = len(result['tokens'])
    result['max_new_tokens'] = budget
    result['eos_token_id'] = handle['tok'].eos_token_id
    return result

