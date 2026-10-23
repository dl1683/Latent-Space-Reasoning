"""Guarded, cached-only Phi3 reader loader; no CUDA/model work on import."""
import json
import os
from pathlib import Path


def stream_model(path, *, checkpoint=None):
    """Count the actual meta model before CUDA, then stream BF16 parameters."""
    if os.environ.get('LSR_GPU_GUARDED') != '1':
        raise RuntimeError('external GPU resource_guard required')
    import psutil
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    root = Path(path).resolve()
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if config.model_type != 'phi3' or config.tie_word_embeddings:
        raise ValueError('finite loader supports untied built-in Phi3 only')
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa',
                                                torch_dtype=torch.bfloat16, trust_remote_code=False)
    names = dict(model.named_parameters())
    count = sum(value.numel() for value in names.values())
    if not 0 < count < 4_000_000_000 or any(not value.is_meta for value in names.values()):
        raise ValueError('actual meta model must contain fewer than four billion parameters')
    print(json.dumps(dict(stage='small_reader_meta_count', parameters=count,
                          model_type=config.model_type)), flush=True)
    index = root / 'model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_text())['weight_map']
    else:
        with safe_open(root / 'model.safetensors', framework='pt', device='cpu') as source:
            mapping = {name: 'model.safetensors' for name in source.keys()}
    if set(mapping) != set(names):
        raise ValueError('checkpoint/meta-model parameter membership mismatch')
    if any(Path(shard).name != shard for shard in mapping.values()):
        raise ValueError('checkpoint shard outside pinned snapshot')
    # No CUDA call occurs above the actual meta-model size assertion.
    import uuid
    expected = uuid.UUID(os.environ['LSR_GPU_UUID'].removeprefix('GPU-')).hex
    actual = torch.cuda.get_device_properties(0).uuid
    actual = uuid.UUID(bytes=actual).hex if isinstance(actual, bytes) else uuid.UUID(str(actual).removeprefix('GPU-')).hex
    if expected != actual:
        raise ValueError('actual CUDA UUID differs from resource_guard')
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    if torch.cuda.mem_get_info()[0] < 20 * 2**30:
        raise MemoryError('initial CUDA free VRAM reserve')
    for name in names:
        if checkpoint is not None:
            checkpoint()
        if psutil.virtual_memory().available < 20 * 2**30:
            raise MemoryError('available RAM below 20 GiB')
        with safe_open(root / mapping[name], framework='pt', device='cpu') as source:
            tensor = source.get_tensor(name)
            if tensor.shape != names[name].shape or not tensor.is_floating_point():
                raise ValueError('checkpoint/meta parameter shape or dtype differs')
            value = tensor.to(device='cuda:0', dtype=torch.bfloat16)
            del tensor
        parent, attr = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), attr, torch.nn.Parameter(value, requires_grad=False))
        del value
    for name, value in list(model.named_buffers()):
        if checkpoint is not None:
            checkpoint()
        if value.is_meta:
            raise ValueError('unmaterialized model buffer')
        parent, attr = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), attr, value.to('cuda:0'))
    if any(value.is_meta or value.device.type != 'cuda' or value.dtype != torch.bfloat16
           for value in model.parameters()):
        raise ValueError('incomplete CUDA BF16 checkpoint assignment')
    model.generation_config = GenerationConfig.from_pretrained(root, local_files_only=True)
    model.lsr_parameter_count = count
    return model.eval()
