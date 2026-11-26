"""Feed versioned source moments into a borrowed causal backbone's hidden steps."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import chain, islice
from math import isfinite
from pathlib import Path
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import _frozen
from experiments.memory.linear_feedback import AssociationMemory, LinearFeedbackKernel, MomentMemory
from experiments.memory.source_revision_feed import SourceDelta, read_source_delta
from experiments.memory.verified_source_tool import SourceSpan, sha256


@dataclass(frozen=True, slots=True)
class _EncodedSpan:
    span: SourceSpan
    row: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class NativeSourceMemory:
    """One retained source scope and its pre-question numerical contribution."""

    _owner: object
    _scope: SourceDelta
    _bank: MomentMemory
    _sources: Mapping[str, tuple[_EncodedSpan, ...]]
    omitted_sources: frozenset[str] = frozenset()
    _derived: AssociationMemory | None = None
    _seed_limit: int = 0
    derived_seeds: tuple[SourceSpan, ...] = ()

    @property
    def seal(self):
        return self._scope.seal

    @property
    def source_count(self):
        return len(self._scope.manifest)

    @property
    def rows(self):
        return self._bank.count

    def source_spans(self, source):
        return tuple(item.span for item in self._sources[source])


@dataclass(frozen=True, slots=True)
class NativeLatentRead:
    text: str
    token_ids: tuple[int, ...]
    terminated: bool
    latent_steps: int
    routed_memory_steps: int
    applied_memory_steps: int
    routed_derived_steps: int
    source_seal: str | None
    omitted_sources: frozenset[str]


def _require_cpu_eval(model):
    tensors = chain(model.parameters(), model.buffers())
    if any(layer.training for layer in model.modules()) or any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("the entire native backbone must be in eval mode on CPU")


class NativeLatentModel:
    """Use one causal backbone for source encoding, continuous steps and readout.

    The caller supplies an immutable eval-mode model, tokenizer and routing maps.
    The backbone must implement the Transformers causal cache protocol and use a
    vocabulary head that directly reads its final normalized hidden vectors.
    """

    def __init__(self, model, tokenizer, moments, *, source_chars, context_tokens, latent_steps,
                 output_tokens, representation_scale, memory_gain, null_weight, eos_ids,
                 relay_gain=1.0, derived_gain=0.5):
        _require_cpu_eval(model)
        embedding = model.get_input_embeddings()
        if (embedding.weight.device.type != "cpu" or model.get_output_embeddings() is None
                or not isinstance(moments, LinearFeedbackKernel) or len(moments._query) != embedding.weight.shape[1]):
            raise ValueError("expected an eval-mode CPU causal model and matching source moments")
        caps = source_chars, context_tokens, latent_steps, output_tokens
        if any(type(cap) is not int or cap <= 0 for cap in caps) or latent_steps + output_tokens >= context_tokens:
            raise ValueError("provide positive bounds with room for a task and its answer")
        native_limit = getattr(model.config, "max_position_embeddings", None)
        if native_limit is not None and context_tokens > native_limit:
            raise ValueError("context bound exceeds the backbone's position limit")
        eos_ids = frozenset(eos_ids)
        gains = representation_scale, memory_gain, null_weight, relay_gain, derived_gain
        if (any(not isfinite(gain) or gain <= 0 for gain in gains)
                or not eos_ids or any(type(token) is not int or token < 0 for token in eos_ids)):
            raise ValueError("provide finite positive gains and explicit stop tokens")
        self._model, self._tokenizer, self._moments = model, tokenizer, moments
        self._source_chars, self._context_tokens = source_chars, context_tokens
        self._steps, self._output_tokens = latent_steps, output_tokens
        self._scale, self._memory_gain, self._null_weight = representation_scale, memory_gain, null_weight
        self._relay_gain, self._derived_gain = relay_gain, derived_gain
        self._eos = eos_ids
        self._owner = object()

    def _embed(self, text, limit):
        import torch

        ids = self._tokenizer.encode(text, add_special_tokens=True, truncation=False)
        lookup = self._model.get_input_embeddings()
        if not ids or len(ids) > limit:
            raise ValueError("input must tokenize to a nonempty sequence inside its context bound")
        if any(type(token) is not int or not 0 <= token < lookup.num_embeddings for token in ids):
            raise ValueError("tokenizer returned an invalid native input ID")
        return lookup(torch.tensor([ids], dtype=torch.long, device=lookup.weight.device))

    def _backbone(self, embeddings, cache=None, *, retain_cache):
        import torch

        prefix = 0 if cache is None else int(cache.get_seq_length())
        result = self._model.base_model(inputs_embeds=embeddings, past_key_values=cache,
                                        use_cache=retain_cache, return_dict=True)
        hidden = result.last_hidden_state[:, -1:, :]
        next_cache = result.past_key_values
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("nonfinite native hidden state")
        if retain_cache and (next_cache is None or int(next_cache.get_seq_length()) != prefix + embeddings.shape[1]):
            raise ValueError("backbone did not retain the expected causal prefix")
        return hidden, next_cache

    def _encode_source(self, source, raw, version):
        text = raw.decode("utf8")
        offset = 0
        encoded = []
        for start in range(0, len(text), self._source_chars):
            chunk = text[start:start + self._source_chars]
            data = chunk.encode("utf8")
            stop = offset + len(data)
            span = SourceSpan(source, version.sha256, offset, stop, sha256(data))
            hidden, _ = self._backbone(self._embed(chunk, self._context_tokens), retain_cache=False)
            row = _frozen(self._encode_hidden(hidden))
            encoded.append(_EncodedSpan(span, row))
            offset = stop
        return tuple(encoded)

    def _encode_hidden(self, hidden):
        with np.errstate(over="ignore"):
            return np.tanh(hidden[0, 0].double().numpy() / self._scale)

    def _check_memory(self, memory):
        if type(memory) is not NativeSourceMemory or memory._owner is not self._owner:
            raise ValueError("source memory belongs to a different native reader")

    def publish(self, epoch_directory: Path, seal: str, *, previous=None):
        """Encode only changed sources from one verified, query-blind feed epoch."""
        import torch

        _require_cpu_eval(self._model)
        if previous is not None:
            self._check_memory(previous)
            if previous.omitted_sources:
                raise ValueError("an ablated contribution view cannot publish a new epoch")
        delta = read_source_delta(epoch_directory, seal, previous=None if previous is None else previous._scope)
        sources = {} if previous is None else dict(previous._sources)
        bank = self._moments._empty if previous is None else previous._bank
        retire, add = [], []
        with torch.inference_mode():
            for source, raw in delta.changes.items():
                retire.extend(item.row for item in sources.pop(source, ()))
                if raw is not None:
                    encoded = self._encode_source(source, raw, delta.manifest[source])
                    sources[source] = encoded
                    add.extend(item.row for item in encoded)
        if retire or add:
            bank = self._moments.revise(bank, remove=retire or None, append=add or None)
        scope = replace(delta, changes=MappingProxyType({}))
        return NativeSourceMemory(self._owner, scope, bank, MappingProxyType(sources))

    def without_contribution(self, memory, source):
        """Disable one source's numerical contribution while keeping its source pin."""
        self._check_memory(memory)
        if memory._derived is not None:
            raise ValueError("disable derived associations before intervening on source observations")
        if source in memory.omitted_sources:
            raise ValueError("source contribution is already disabled")
        rows = [item.row for item in memory._sources[source]]
        bank = self._moments.revise(memory._bank, remove=rows) if rows else memory._bank
        return replace(memory, _bank=bank, omitted_sources=memory.omitted_sources | {source})

    def without_derived(self, memory):
        """Keep the source observations and disable their acquired associations."""
        self._check_memory(memory)
        return replace(memory, _derived=None, _seed_limit=0, derived_seeds=())

    def _memory_delta(self, hidden, value, mass, gain):
        import torch

        weight = mass / (mass + self._null_weight)
        delta = torch.tensor(value, dtype=hidden.dtype, device=hidden.device)
        return (gain * weight) * delta

    def _continue(self, hidden, cache, memory):
        import torch

        routed_steps = applied_steps = derived_steps = 0
        for _ in range(self._steps):
            relay = hidden * self._relay_gain
            thought = relay
            if memory is not None:
                query = self._encode_hidden(hidden)
                value, mass = self._moments.read_observations(query, memory._bank)
                if mass > 0:
                    thought = thought + self._memory_delta(hidden, value, mass, self._memory_gain)
                    routed_steps += 1
                if memory._derived is not None:
                    association, mass = self._moments.read_associations(query, memory._derived)
                    if mass > 0:
                        thought = thought + self._memory_delta(hidden, association, mass, self._derived_gain)
                        derived_steps += 1
                applied_steps += bool(torch.any(thought != relay))
            if not torch.isfinite(thought).all():
                raise FloatingPointError("nonfinite continuous input")
            hidden, cache = self._backbone(thought, cache, retain_cache=True)
        return hidden, cache, routed_steps, applied_steps, derived_steps

    def maintain(self, memory, *, max_seeds):
        """Propose and install bounded source-only continuations before tasks arrive."""
        import torch

        _require_cpu_eval(self._model)
        self._check_memory(memory)
        if type(max_seeds) is not int or max_seeds <= 0 or memory.omitted_sources:
            raise ValueError("maintenance needs a positive seed budget and an intact source bank")
        if memory._seed_limit == max_seeds:
            return memory
        source_only = self.without_derived(memory)
        rows = (item for source in sorted(memory._sources) for item in memory._sources[source])
        lookup = self._model.get_input_embeddings().weight
        seeds, addresses, values = [], [], []
        with torch.inference_mode():
            for item in islice(rows, max_seeds):
                seed = torch.tensor(item.row, dtype=lookup.dtype, device=lookup.device)[None, None, :]
                hidden, cache = self._backbone(seed * self._relay_gain, retain_cache=True)
                address = self._encode_hidden(hidden)
                hidden, _, _, applied, _ = self._continue(hidden, cache, source_only)
                if not applied:
                    continue
                seeds.append(item.span)
                addresses.append(address)
                values.append(self._encode_hidden(hidden))
        bank = self._moments.prepare_associations(addresses, values) if seeds else None
        return replace(source_only, _derived=bank, _seed_limit=max_seeds, derived_seeds=tuple(seeds))

    def __call__(self, task: str, *, memory=None):
        """Read raw task text through the same recurrent call with optional source memory."""
        import torch

        _require_cpu_eval(self._model)
        if type(task) is not str or not task.strip():
            raise ValueError("task must be nonempty raw text")
        if memory is not None:
            self._check_memory(memory)
        limit = self._context_tokens - self._steps - self._output_tokens
        with torch.inference_mode():
            hidden, cache = self._backbone(self._embed(task, limit), retain_cache=True)
            hidden, cache, routed_steps, applied_steps, derived_steps = self._continue(hidden, cache, memory)
            ids = []
            for step in range(self._output_tokens):
                logits = self._model.get_output_embeddings()(hidden)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite native vocabulary scores")
                token = int(logits[0, -1].argmax())
                ids.append(token)
                if token in self._eos or step + 1 == self._output_tokens:
                    break
                lookup = self._model.get_input_embeddings()
                embedded = lookup(torch.tensor([[token]], dtype=torch.long, device=hidden.device))
                hidden, cache = self._backbone(embedded, cache, retain_cache=True)
        return NativeLatentRead(self._tokenizer.decode(ids, skip_special_tokens=True), tuple(ids),
                                ids[-1] in self._eos, self._steps, routed_steps, applied_steps, derived_steps,
                                None if memory is None else memory.seal,
                                frozenset() if memory is None else memory.omitted_sources)
