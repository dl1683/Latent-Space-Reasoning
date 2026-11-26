"""Feed versioned source moments into a borrowed causal backbone's hidden steps."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import chain, islice
from math import isfinite
from pathlib import Path
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import _apply_value, _frozen
from experiments.memory.linear_feedback import AssociationMemory, LinearFeedbackKernel, MomentMemory
from experiments.memory.selective_continuations import (
    RetainedSelector, RowShard, SelectionReceipt, SelectionStep, frontier_survives,
)
from experiments.memory.source_revision_feed import SourceDelta, read_source_delta
from experiments.memory.verified_source_tool import SourceSpan, require_digest, sha256


@dataclass(frozen=True, slots=True)
class _EncodedSpan:
    span: SourceSpan
    row: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class _Continuation:
    address: np.ndarray
    value: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class _SelectiveWorld:
    shards: Mapping[str, RowShard]
    acquired: Mapping[SourceSpan, _Continuation]
    attempted: Mapping[SourceSpan, tuple[SelectionStep, ...]]
    priority: tuple[str, ...]
    derived: tuple[RowShard, ...]
    parents: Mapping[SourceSpan, SourceSpan]


@dataclass(slots=True)
class _LatentRun:
    hidden: object
    cache: object
    routed: int = 0
    applied: int = 0
    derived: int = 0
    applied_derived: int = 0
    source_steps: tuple[SelectionStep, ...] = ()
    derived_reads: tuple[SelectionReceipt, ...] = ()
    derived_steps: tuple[SelectionStep, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class NativeSourceMemory:
    """One retained source scope and its pre-question numerical contribution."""

    _owner: object
    _scope: SourceDelta
    _bank: MomentMemory | None
    _sources: Mapping[str, tuple[_EncodedSpan, ...]]
    omitted_sources: frozenset[str] = frozenset()
    _derived: AssociationMemory | None = None
    _seed_limit: int = 0
    derived_seeds: tuple[SourceSpan, ...] = ()
    _selective: _SelectiveWorld | None = None
    _publication_seal: str | None = None
    _derived_off: bool = False
    _publication_runtime_sha256: str | None = None
    _reader_fingerprint_sha256: str | None = None

    @property
    def seal(self):
        return self._scope.seal

    @property
    def source_count(self):
        return len(self._scope.manifest)

    @property
    def rows(self):
        if self._selective is not None:
            return sum(len(shard.spans) for shard in self._selective.shards.values())
        return self._bank.count

    def source_spans(self, source):
        return tuple(item.span for item in self._sources[source])

    def continuation_reads(self, seed):
        if self._selective is None:
            raise ValueError("continuation receipts require selective memory")
        if seed not in self._selective.acquired:
            raise KeyError(seed)
        return tuple(frontier.receipt for frontier in self._selective.attempted[seed])

    def continuation_parent(self, seed):
        """Return the earlier continuation that affected this computation, if any."""
        if self._selective is None:
            raise ValueError("continuation parents require selective memory")
        if seed not in self._selective.acquired:
            raise KeyError(seed)
        return self._selective.parents.get(seed)

    def evidence_spans(self, read):
        """Collect candidate source spans for an attached read of this publication."""
        if (not isinstance(read, NativeLatentRead) or self._publication_seal is None
                or read.publication_seal != self._publication_seal or read.source_seal != self.seal
                or read.derived_omitted != self._derived_off or read.omitted_sources != self.omitted_sources):
            raise ValueError("evidence spans require a read from the matching reopened publication")

        def selected(receipts):
            return (hit.span for receipt in receipts for hit in receipt.selected)

        spans = dict.fromkeys(selected(read.source_reads))
        visited = set()
        for seed in selected(read.derived_reads):
            while seed is not None and seed not in visited:
                visited.add(seed)
                spans[seed] = None
                spans.update(dict.fromkeys(selected(self.continuation_reads(seed))))
                seed = self.continuation_parent(seed)
        return tuple(spans)

    @property
    def pending_seeds(self):
        if self._selective is None or self.omitted_sources:
            return None
        return self.rows - len(self._selective.attempted)


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
    source_reads: tuple[SelectionReceipt, ...] = ()
    derived_reads: tuple[SelectionReceipt, ...] = ()
    applied_derived_steps: int = 0
    publication_seal: str | None = None
    derived_omitted: bool = False
    source_steps: tuple[SelectionStep, ...] = ()
    derived_steps: tuple[SelectionStep, ...] = ()


@dataclass(frozen=True, slots=True)
class NativeGreedyControl:
    task_ids: tuple[int, ...]
    native: NativeLatentRead
    reference_ids: tuple[int, ...]

    @property
    def matches(self):
        return self.native.token_ids == self.reference_ids


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
                 relay_gain=1.0, derived_gain=0.5, top_k=None, tie_limit=16, temperature=0.1,
                 score_margin=0, task_template=None, source_ingress=None, source_ingress_sha256=None):
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
        if task_template is not None:
            if (type(task_template) is not str or not task_template.strip()
                    or not callable(getattr(tokenizer, "apply_chat_template", None))
                    or not callable(getattr(tokenizer, "get_chat_template", None))):
                raise ValueError("provide a nonempty task template and a chat tokenizer")
            task_template = tokenizer.get_chat_template(task_template)
            if type(task_template) is not str or not task_template.strip():
                raise ValueError("task template must resolve to a nonempty body")
        if source_ingress is not None and not callable(source_ingress):
            raise ValueError("source ingress must be callable")
        if source_ingress is not None or source_ingress_sha256 is not None:
            require_digest(source_ingress_sha256)
        self._model, self._tokenizer, self._moments = model, tokenizer, moments
        self._task_template = task_template
        self._source_ingress, self._source_ingress_sha256 = source_ingress, source_ingress_sha256
        self._source_chars, self._context_tokens = source_chars, context_tokens
        self._steps, self._output_tokens = latent_steps, output_tokens
        self._scale, self._memory_gain, self._null_weight = representation_scale, memory_gain, null_weight
        self._relay_gain, self._derived_gain = relay_gain, derived_gain
        self._eos = eos_ids
        self._owner = object()
        self._selector = (None if top_k is None else RetainedSelector(
            moments, top_k=top_k, tie_limit=tie_limit, temperature=temperature, score_margin=score_margin,
        ))

    def _check_input_ids(self, ids, limit):
        if not ids or len(ids) > limit:
            raise ValueError("input must tokenize to a nonempty sequence inside its context bound")
        vocabulary = self._model.get_input_embeddings().num_embeddings
        if any(type(token) is not int or not 0 <= token < vocabulary for token in ids):
            raise ValueError("tokenizer returned an invalid native input ID")

    def _embed_ids(self, ids):
        import torch

        lookup = self._model.get_input_embeddings()
        return lookup(torch.tensor([ids], dtype=torch.long, device=lookup.weight.device))

    def _embed(self, text, limit):
        ids = self._tokenizer.encode(text, add_special_tokens=True, truncation=False)
        self._check_input_ids(ids, limit)
        return self._embed_ids(ids)

    def _step_budget(self, steps):
        steps = self._steps if steps is None else steps
        if type(steps) is not int or not 0 <= steps <= self._steps:
            raise ValueError("requested latent steps exceed configured bounds")
        return steps

    def task_token_ids(self, task, *, steps=None):
        """Return the exact task prefix used by the native reader, before latent steps."""
        if type(task) is not str or not task.strip():
            raise ValueError("task must be nonempty raw text")
        limit = self._context_tokens - self._step_budget(steps) - self._output_tokens
        if self._task_template is None:
            ids = self._tokenizer.encode(task, add_special_tokens=True, truncation=False)
        else:
            from experiments.memory.verified_reader_bridge import encode_messages

            if self._tokenizer.get_chat_template(self._task_template) != self._task_template:
                raise ValueError("resolved task template changed")
            ids = encode_messages(self._tokenizer, [{"role": "user", "content": task}],
                                  chat_template=self._task_template)
        self._check_input_ids(ids, limit)
        return tuple(ids)

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

    def _encode_source(self, source, raw, version, *, binary):
        if binary:
            projected = self._source_ingress(source, raw)
            if projected is None:
                raise ValueError("binary source needs a compatible source ingress")
            row, = self._moments._bounded_rows((projected,))
            span = SourceSpan(source, version.sha256, 0, len(raw), version.sha256)
            return (_EncodedSpan(span, _frozen(row)),)
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

    def publish(self, epoch_directory: Path, seal: str, *, previous=None, max_source_bytes=None):
        """Encode only changed sources from one verified, query-blind feed epoch."""
        import torch

        _require_cpu_eval(self._model)
        if previous is not None:
            self._check_memory(previous)
            if previous.omitted_sources or previous._derived_off:
                raise ValueError("an ablated contribution view cannot publish a new epoch")
        if self._source_ingress_sha256 is not None and self._source_ingress is None:
            raise ValueError("publication requires the configured source ingress")
        delta = read_source_delta(epoch_directory, seal, previous=None if previous is None else previous._scope,
                                  max_bytes=max_source_bytes, allow_binary=self._source_ingress is not None)
        sources = {} if previous is None else dict(previous._sources)
        bank = None
        retire, add = [], []
        with torch.inference_mode():
            for source, raw in delta.changes.items():
                old_rows = sources.pop(source, ())
                if self._selector is None:
                    retire.extend(item.row for item in old_rows)
                if raw is not None:
                    encoded = self._encode_source(source, raw, delta.manifest[source],
                                                  binary=source in delta.binary_sources)
                    sources[source] = encoded
                    if self._selector is None:
                        add.extend(item.row for item in encoded)
        if self._selector is None:
            bank = self._moments._empty if previous is None else previous._bank
            if retire or add:
                bank = self._moments.revise(bank, remove=retire or None, append=add or None)
        scope = replace(delta, changes=MappingProxyType({}))
        memory = NativeSourceMemory(self._owner, scope, bank, MappingProxyType(sources))
        if self._selector is None:
            return memory
        selective = self._revise_selection(None if previous is None else previous._selective, sources, delta.changes)
        return replace(memory, _selective=selective, derived_seeds=tuple(selective.acquired))

    def _derived_rows(self, acquired):
        if not acquired:
            return ()
        seeds = sorted(acquired, key=lambda span: (span.source_id, span.start_byte, span.end_byte))
        return (self._selector.shard(seeds, [acquired[seed].address for seed in seeds],
                                     [acquired[seed].value for seed in seeds]),)

    def _revise_selection(self, previous, sources, changed):
        shards = {} if previous is None else dict(previous.shards)
        revised = []
        for source in changed:
            shards.pop(source, None)
            rows = sources.get(source)
            if rows:
                vectors = [row.row for row in rows]
                shard = self._selector.shard([row.span for row in rows], vectors, vectors)
                shards[source] = shard
                revised.append(shard)
        attempted, acquired, parents, previous_priority = {}, {}, {}, ()
        if previous is not None:
            previous_priority = previous.priority
            for seed, frontiers in previous.attempted.items():
                if seed.source_id not in changed and all(
                        frontier_survives(frontier, changed, revised) for frontier in frontiers):
                    attempted[seed] = frontiers
            for seed, row in previous.acquired.items():
                parent = previous.parents.get(seed)
                if seed in attempted and (parent is None or parent in acquired):
                    acquired[seed] = row
            for seed, parent in previous.parents.items():
                if seed in attempted:
                    if parent in acquired:
                        parents[seed] = parent
                    else:
                        del attempted[seed]
        derived = (previous.derived if previous is not None and len(acquired) == len(previous.acquired)
                   else self._derived_rows(acquired))
        ordered = MappingProxyType(dict(sorted(shards.items())))
        order = dict.fromkeys((*previous_priority, *sorted(sources)))
        priority = self._pending_groups(order, sources, attempted)
        return _SelectiveWorld(ordered, MappingProxyType(acquired), MappingProxyType(attempted),
                               priority, derived, MappingProxyType(parents))

    @staticmethod
    def _pending_groups(order, sources, attempted):
        return tuple(source for source in order if any(item.span not in attempted for item in sources.get(source, ())))

    def without_contribution(self, memory, source):
        """Disable one source's numerical contribution while keeping its source pin."""
        self._check_memory(memory)
        if memory.derived_seeds:
            raise ValueError("disable derived associations before intervening on source observations")
        if source in memory.omitted_sources:
            raise ValueError("source contribution is already disabled")
        rows = [item.row for item in memory._sources[source]]
        bank = memory._bank
        if rows and bank is not None:
            bank = self._moments.revise(bank, remove=rows)
        selective = memory._selective
        if selective is not None:
            shards = MappingProxyType({key: row for key, row in selective.shards.items() if key != source})
            selective = replace(selective, shards=shards)
        return replace(memory, _bank=bank, _selective=selective, omitted_sources=memory.omitted_sources | {source})

    def without_derived(self, memory):
        """Keep the source observations and disable their acquired associations."""
        self._check_memory(memory)
        selective = memory._selective
        if selective is not None:
            selective = replace(selective, acquired=MappingProxyType({}), attempted=MappingProxyType({}),
                                priority=tuple(sorted(selective.shards)), derived=(), parents=MappingProxyType({}))
        return replace(memory, _derived=None, _seed_limit=0, derived_seeds=(),
                       _selective=selective, _derived_off=True)

    def _memory_delta(self, hidden, value, mass, gain):
        import torch

        weight = mass / (mass + self._null_weight)
        delta = torch.tensor(value, dtype=hidden.dtype, device=hidden.device)
        return (gain * weight) * delta

    def _continue(self, hidden, cache, memory, steps):
        import torch

        run = _LatentRun(hidden, cache)
        source_steps, derived_reads, derived_steps = [], [], []
        selective = None if memory is None else memory._selective
        shards = () if selective is None else tuple(selective.shards.values())
        for _ in range(steps):
            relay = hidden * self._relay_gain
            thought = relay
            if memory is not None:
                query = self._encode_hidden(hidden)
                if selective is None:
                    value, mass = self._moments.read_observations(query, memory._bank)
                else:
                    value, mass, trace = self._selector.read(query, shards)
                    if value is not None:
                        value = _apply_value(self._moments._observation_value, value)
                if mass > 0:
                    thought = thought + self._memory_delta(hidden, value, mass, self._memory_gain)
                    run.routed += 1
                if selective is not None:
                    receipt = replace(trace.receipt, observed_source_seal=memory.seal)
                    source_steps.append(replace(trace, receipt=receipt, applied=bool(torch.any(thought != relay))))
                if memory._derived is not None:
                    association, mass = self._moments.read_associations(query, memory._derived)
                    derived_trace = None
                elif selective is not None and not memory._derived_off:
                    association, mass, trace = self._selector.read(query, selective.derived)
                    receipt = replace(trace.receipt, observed_source_seal=memory.seal)
                    derived_reads.append(receipt)
                    derived_trace = replace(trace, receipt=receipt)
                else:
                    mass = 0
                    derived_trace = None
                if mass > 0:
                    previous = thought
                    thought = thought + self._memory_delta(hidden, association, mass, self._derived_gain)
                    run.derived += 1
                    derived_applied = bool(torch.any(thought != previous))
                    run.applied_derived += derived_applied
                else:
                    derived_applied = False
                if derived_trace is not None:
                    derived_steps.append(replace(derived_trace, applied=derived_applied))
                run.applied += bool(torch.any(thought != relay))
            if not torch.isfinite(thought).all():
                raise FloatingPointError("nonfinite continuous input")
            hidden, cache = self._backbone(thought, cache, retain_cache=True)
        run.hidden, run.cache = hidden, cache
        run.source_steps = tuple(source_steps)
        run.derived_reads = tuple(derived_reads)
        run.derived_steps = tuple(derived_steps)
        return run

    def _run_seed(self, item, memory, lookup):
        import torch

        seed = torch.tensor(item.row, dtype=lookup.dtype, device=lookup.device)[None, None, :]
        hidden, cache = self._backbone(seed * self._relay_gain, retain_cache=True)
        address = self._encode_hidden(hidden)
        return address, self._continue(hidden, cache, memory, self._steps)

    def maintain(self, memory, *, max_seeds, parent=None):
        """Install bounded continuations with one optional parent from an earlier pass."""
        import torch

        _require_cpu_eval(self._model)
        self._check_memory(memory)
        if type(max_seeds) is not int or max_seeds <= 0 or memory.omitted_sources:
            raise ValueError("maintenance needs a positive seed budget and an intact source bank")
        if parent is not None and (not isinstance(parent, SourceSpan) or memory._selective is None
                                   or parent not in memory._selective.acquired):
            raise ValueError("maintenance parent must be an installed continuation at this pin")
        if memory._selective is not None:
            return self._maintain_selection(memory, max_seeds, parent)
        if memory._seed_limit == max_seeds:
            return memory
        source_only = self.without_derived(memory)
        rows = (item for source in sorted(memory._sources) for item in memory._sources[source])
        lookup = self._model.get_input_embeddings().weight
        seeds, addresses, values = [], [], []
        with torch.inference_mode():
            for item in islice(rows, max_seeds):
                address, run = self._run_seed(item, source_only, lookup)
                if not run.applied:
                    continue
                seeds.append(item.span)
                addresses.append(address)
                values.append(self._encode_hidden(run.hidden))
        bank = self._moments.prepare_associations(addresses, values) if seeds else None
        return replace(source_only, _derived=bank, _seed_limit=max_seeds, derived_seeds=tuple(seeds),
                       _derived_off=False, _publication_seal=None)

    def _maintain_selection(self, memory, max_seeds, parent):
        import torch

        selective = memory._selective
        rows = (row for source in selective.priority for row in memory._sources[source]
                if row.span not in selective.attempted)
        batch = tuple(islice(rows, max_seeds))
        if not batch:
            return memory
        # Freeze a single parent for the entire pass, including attempts that do not select it.
        inputs = () if parent is None else self._derived_rows({parent: selective.acquired[parent]})
        worker_input = replace(memory, _selective=replace(selective, derived=inputs))
        lookup = self._model.get_input_embeddings().weight
        acquired, attempted = dict(selective.acquired), dict(selective.attempted)
        parents = dict(selective.parents)
        with torch.inference_mode():
            for item in batch:
                address, run = self._run_seed(item, worker_input, lookup)
                attempted[item.span] = run.source_steps
                used_parent = parent if run.applied_derived else None
                if used_parent is not None:
                    parents[item.span] = used_parent
                if self._installable(run.source_steps, item.span.source_id, used_parent):
                    acquired[item.span] = _Continuation(_frozen(address), _frozen(self._encode_hidden(run.hidden)))
        derived = selective.derived if len(acquired) == len(selective.acquired) else self._derived_rows(acquired)
        worked = {item.span.source_id for item in batch}
        order = sorted(selective.priority, key=worked.__contains__)
        priority = self._pending_groups(order, memory._sources, attempted)
        selective = replace(selective, acquired=MappingProxyType(acquired), derived=derived,
                            attempted=MappingProxyType(attempted), priority=priority, parents=MappingProxyType(parents))
        return replace(memory, _selective=selective, derived_seeds=tuple(acquired),
                       _derived_off=False, _publication_seal=None)

    @staticmethod
    def _installable(steps, seed_source, parent=None):
        if any(step.receipt.status not in ("ready", "empty") for step in steps):
            return False
        return parent is not None or any(hit.span.source_id != seed_source for step in steps
                                         if step.applied for hit in step.receipt.selected)

    def __call__(self, task: str, *, memory=None, steps=None):
        """Read raw task text through the same recurrent call with optional source memory."""
        _require_cpu_eval(self._model)
        if memory is not None:
            self._check_memory(memory)
        steps = self._step_budget(steps)
        return self._read_ids(self.task_token_ids(task, steps=steps), memory, steps)

    def compare_greedy(self, task: str):
        """Compare zero-step native readout with the model's ordinary cached forward."""
        import torch

        _require_cpu_eval(self._model)
        ids = self.task_token_ids(task, steps=0)
        device = self._model.get_input_embeddings().weight.device
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        reference, cache, prefix = [], None, 0
        # Use input IDs and the full model head. GenerationConfig processors can
        # otherwise change the greedy reference before the stop rule even runs.
        with torch.inference_mode():
            for _ in range(self._output_tokens):
                state = self._model(input_ids=inputs, past_key_values=cache, use_cache=True, return_dict=True)
                prefix += inputs.shape[1]
                cache = state.past_key_values
                if cache is None or int(cache.get_seq_length()) != prefix:
                    raise ValueError("ordinary forward did not retain the expected causal prefix")
                logits = state.logits[0, -1]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite reference vocabulary scores")
                token = int(logits.argmax())
                reference.append(token)
                if token in self._eos:
                    break
                inputs = inputs.new_tensor([[token]])
        return NativeGreedyControl(ids, self._read_ids(ids, None, 0), tuple(reference))

    def _read_ids(self, input_ids, memory, steps):
        return self._read_ids_with_limit(input_ids, memory, steps, self._output_tokens)

    def read_token_ids(self, input_ids, *, memory=None, steps=None, max_new_tokens=None):
        """Read an exact, already-tokenized prompt with a caller-owned output cap.

        This is the episode adapter boundary: it consumes the exact prepared
        token sequence rather than decoding and retokenizing its chat frame.
        """
        _require_cpu_eval(self._model)
        if memory is not None:
            self._check_memory(memory)
        steps = self._step_budget(steps)
        output_limit = self._output_tokens if max_new_tokens is None else max_new_tokens
        if type(output_limit) is not int or not 1 <= output_limit <= self._output_tokens:
            raise ValueError("output allowance exceeds the configured native bound")
        input_ids = tuple(input_ids)
        self._check_input_ids(input_ids, self._context_tokens - steps - output_limit)
        return self._read_ids_with_limit(input_ids, memory, steps, output_limit)

    def _read_ids_with_limit(self, input_ids, memory, steps, output_limit):
        import torch

        with torch.inference_mode():
            hidden, cache = self._backbone(self._embed_ids(input_ids), retain_cache=True)
            run = self._continue(hidden, cache, memory, steps)
            hidden, cache = run.hidden, run.cache
            ids = []
            for step in range(output_limit):
                logits = self._model.get_output_embeddings()(hidden)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite native vocabulary scores")
                token = int(logits[0, -1].argmax())
                ids.append(token)
                if token in self._eos or step + 1 == output_limit:
                    break
                lookup = self._model.get_input_embeddings()
                embedded = lookup(torch.tensor([[token]], dtype=torch.long, device=hidden.device))
                hidden, cache = self._backbone(embedded, cache, retain_cache=True)
        return NativeLatentRead(self._tokenizer.decode(ids, skip_special_tokens=True), tuple(ids),
                                ids[-1] in self._eos, steps, run.routed, run.applied, run.derived,
                                None if memory is None else memory.seal,
                                frozenset() if memory is None else memory.omitted_sources,
                                tuple(step.receipt for step in run.source_steps), run.derived_reads,
                                run.applied_derived, None if memory is None else memory._publication_seal,
                                False if memory is None else memory._derived_off,
                                run.source_steps, run.derived_steps)
