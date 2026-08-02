"""Architecture introspection helpers.

HuggingFace wrappers disagree about where the text decoder lives. A plain causal
LM (``Qwen3ForCausalLM``) exposes its decoder as ``model.model.layers`` and its
text hyperparameters directly on ``config``. A multimodal wrapper such as Gemma
4's ``Gemma4ForConditionalGeneration`` nests the decoder under
``model.language_model.layers`` and moves every text hyperparameter into
``config.text_config``, leaving ``config.hidden_size`` undefined.

These helpers resolve both layouts by structure rather than by model family, so
adding a model does not mean adding a special case.
"""

from __future__ import annotations

from typing import Any

from torch import nn


def text_config(obj: Any) -> Any:
    """Return the config carrying the *text* hyperparameters.

    Accepts either a model or a config. A config that declares ``hidden_size``
    itself already *is* the text config, and is returned unchanged -- descending
    is only correct for composite configs that hoist the text hyperparameters
    into a sub-config (Gemma 4, and multimodal wrappers generally).

    Checking the flat case first also keeps this honest against duck-typed
    stand-ins, where an auto-created ``get_text_config`` would otherwise answer
    for a config that never had a sub-config at all.
    """
    config = getattr(obj, "config", obj)
    if getattr(config, "hidden_size", None) is not None:
        return config
    # transformers >= 4.46 exposes this on every composite config.
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            resolved = getter()
        except Exception:
            resolved = None
        if resolved is not None and getattr(resolved, "hidden_size", None) is not None:
            return resolved
    return getattr(config, "text_config", config)


def hidden_size(obj: Any) -> int:
    """Hidden width of the text decoder."""
    size = getattr(text_config(obj), "hidden_size", None)
    if size is None:
        raise ValueError(
            f"Could not determine hidden_size for {type(obj).__name__}; "
            "neither config.hidden_size nor config.text_config.hidden_size is set."
        )
    return int(size)


def decoder_layers(model: Any) -> nn.ModuleList | None:
    """Return the text decoder's layer stack, or ``None`` if it cannot be found.

    Searches the known nesting paths in order of specificity. The vision tower's
    layers are never returned: ``language_model`` is probed before the bare
    ``model.layers`` fallback, and only attributes literally named ``layers`` on
    the text path are considered.
    """
    candidates = (
        ("model", "language_model", "layers"),  # Gemma 4 / multimodal wrappers
        ("language_model", "model", "layers"),  # older multimodal layouts
        ("model", "layers"),                    # plain causal LM
        ("transformer", "h"),                   # GPT-2 / phi-style
        ("layers",),
    )
    for path in candidates:
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if isinstance(node, nn.ModuleList) and len(node) > 0:
            return node
    return None


def num_decoder_layers(model: Any) -> int:
    """Number of text decoder layers, or 0 if the stack cannot be located."""
    layers = decoder_layers(model)
    return len(layers) if layers is not None else 0
