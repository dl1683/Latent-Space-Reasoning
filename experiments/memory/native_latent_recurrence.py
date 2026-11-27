"""Feed versioned source moments into a borrowed causal backbone's hidden steps."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from itertools import chain, islice
from math import isfinite
from pathlib import Path
from threading import Lock
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import _apply_value, _frozen
from experiments.memory.linear_feedback import (
    AssociationMemory, LinearFeedbackKernel, MomentMemory, _quantize,
)
from experiments.memory.native_forecast_set import (
    NativeForecastObservation, NativeForecastSet, NativeRevisionWitness, NativeWeightedRevisionWitness,
    _ForecastAlternative, _ForecastSetCheck, _MAX_ARMS, _RevisionWitnessOpening, _WeightedWitnessOpening,
    _forecast_error, _forecast_fields, _forecast_values, _measure_witness, _payload_digest, _validate_witnesses,
    _witness_prediction,
)
from experiments.memory.selective_continuations import (
    RetainedSelector, RowShard, SelectionReceipt, SelectionStep, frontier_survives,
)
from experiments.memory.source_revision_feed import SourceDelta, _OPAQUE_ID, _encode, read_source_delta
from experiments.memory.verified_source_tool import SourceSpan, require_digest, sha256


_GEAR = tuple(int.from_bytes(bytes.fromhex(sha256(bytes((value,))))[:8], "little") for value in range(256))
_GEAR_MASK = (1 << 64) - 1


def _content_chunks(text, width):
    """Cut on rolling text landmarks with a hard character cap."""
    minimum = max(1, width // 2)
    anchor_mask = (1 << max(0, (width // 4).bit_length() - 1)) - 1
    rolling = start = 0
    for end, char in enumerate(text, start=1):
        # Keep the rolling state across cuts so an insertion can resynchronize.
        rolling = ((rolling << 1) + _GEAR[ord(char) & 255]) & _GEAR_MASK
        length = end - start
        if length == width or (length >= minimum and rolling & anchor_mask == 0):
            yield text[start:end]
            start = end
    if start < len(text):
        yield text[start:]


@dataclass(frozen=True, slots=True)
class _EncodedSpan:
    span: SourceSpan
    row: np.ndarray
    input_sha256: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class _Continuation:
    address: np.ndarray
    value: np.ndarray


@dataclass(frozen=True, slots=True)
class _ForecastCheck:
    opening_source: str
    admission_epoch: int
    witness: SourceSpan
    features: tuple[int, ...]
    tolerance: int
    baseline: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class _SelectiveWorld:
    shards: Mapping[str, RowShard]
    acquired: Mapping[SourceSpan, _Continuation]
    attempted: Mapping[SourceSpan, tuple[SelectionStep, ...]]
    priority: tuple[str, ...]
    derived: tuple[RowShard, ...]
    parents: Mapping[SourceSpan, SourceSpan]
    forecasts: Mapping[SourceSpan, _ForecastCheck] = field(default_factory=lambda: MappingProxyType({}))
    forecast_sets: Mapping[SourceSpan, _ForecastSetCheck] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class NativeValueWithholding:
    seed: SourceSpan
    step: int | None = None
    changed_input: bool | None = None
    input_max_abs_delta: float | None = None


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
    value_withholding: NativeValueWithholding | None = None


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
    _omitted_continuation: SourceSpan | None = None
    _source_anchor_seed: SourceSpan | None = None

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

    def _is_continuation_leaf(self, seed):
        selective = self._selective
        return seed in selective.acquired and not any(
            child in selective.acquired and parent == seed for child, parent in selective.parents.items()
        )

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

    def continuation_witness(self, seed):
        """Return a separately observed admission witness, when one exists."""
        if self._selective is None or seed not in self._selective.acquired:
            raise KeyError(seed)
        check = self._selective.forecasts.get(seed)
        return None if check is None else check.witness

    def continuation_witnesses(self, seed):
        """Return the source observations required to keep this continuation live."""
        single = self.continuation_witness(seed)
        check = self._selective.forecast_sets.get(seed)
        return check.witness_spans if check is not None else ((single,) if single is not None else ())

    def evidence_spans(self, read):
        """Collect candidate source spans for an attached read of this publication."""
        if (not isinstance(read, NativeLatentRead) or self._publication_seal is None
                or read.publication_seal != self._publication_seal or read.source_seal != self.seal
                or read.derived_omitted != self._derived_off or read.omitted_sources != self.omitted_sources
                or read.omitted_continuation != self._omitted_continuation
                or read.source_anchor_seed != self._source_anchor_seed
                or read.value_withholding is not None):
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


@dataclass(frozen=True, slots=True, eq=False)
class NativeContinuationForecast:
    seed: SourceSpan
    parent: SourceSpan
    witness_source: str
    features: tuple[int, ...]
    tolerance: int
    baseline: tuple[int, ...]
    prediction: tuple[int, ...]
    _opening: NativeSourceMemory = field(repr=False)
    _continuation: _Continuation = field(repr=False)
    _steps: tuple[SelectionStep, ...] = field(repr=False)
    _baseline_steps: tuple[SelectionStep, ...] = field(repr=False)


@dataclass(frozen=True, slots=True, eq=False)
class NativeForecastResolution:
    memory: NativeSourceMemory
    status: str
    observation: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class NativeForecastSetResolution:
    memory: NativeSourceMemory
    status: str
    survivors: tuple[SourceSpan, ...] = ()
    observations: tuple[NativeForecastObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class _ForecastCursor:
    memory: NativeSourceMemory
    payloads: tuple[_Continuation, ...]
    witness_spans: tuple[SourceSpan, ...] = ()
    observations: tuple[NativeForecastObservation, ...] = ()
    observation_seals: tuple[str, ...] = ()


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

    omitted_continuation: SourceSpan | None = None
    source_anchor_seed: SourceSpan | None = None
    value_withholding: NativeValueWithholding | None = None

    def json_record(self):
        """Serialize a read receipt, preserving packed queries as hexadecimal."""
        record = asdict(self)
        record["omitted_sources"] = sorted(self.omitted_sources)
        if self.omitted_continuation is None:
            del record["omitted_continuation"]
        if self.source_anchor_seed is None:
            del record["source_anchor_seed"]
        if self.value_withholding is None:
            del record["value_withholding"]
        for name in ("source_steps", "derived_steps"):
            for step in record[name]:
                step["query"] = step["query"].hex()
        return record


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
                 score_margin=0, task_template=None, source_ingress=None, source_ingress_sha256=None,
                 source_chunking="fixed", source_tokenizer_sha256=None):
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
        if type(source_chunking) is not str or source_chunking not in ("fixed", "content"):
            raise ValueError("source chunking must be fixed or content")
        if source_chunking == "content" and top_k is None:
            raise ValueError("content-defined source chunking requires a selective reader")
        if source_chunking == "content" and source_tokenizer_sha256 is None:
            raise ValueError("content chunking needs an explicit tokenizer identity")
        if source_tokenizer_sha256 is not None:
            require_digest(source_tokenizer_sha256)
        if source_ingress is not None and not callable(source_ingress):
            raise ValueError("source ingress must be callable")
        if source_ingress is not None or source_ingress_sha256 is not None:
            require_digest(source_ingress_sha256)
        self._model, self._tokenizer, self._moments = model, tokenizer, moments
        self._task_template = task_template
        self._source_ingress, self._source_ingress_sha256 = source_ingress, source_ingress_sha256
        self._source_chars, self._context_tokens = source_chars, context_tokens
        self._source_chunking = source_chunking
        self._source_tokenizer_sha256 = source_tokenizer_sha256
        self._steps, self._output_tokens = latent_steps, output_tokens
        self._scale, self._memory_gain, self._null_weight = representation_scale, memory_gain, null_weight
        self._relay_gain, self._derived_gain = relay_gain, derived_gain
        self._eos = eos_ids
        self._owner = object()
        self._forecast_lock = Lock()
        self._pending_forecast = None
        self._forecast_cursor = None
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

    def _encode_source(self, source, raw, version, *, binary, previous_text=()):
        if binary:
            projected = self._source_ingress(source, raw)
            if projected is None:
                raise ValueError("binary source needs a compatible source ingress")
            row, = self._moments._bounded_rows((projected,))
            span = SourceSpan(source, version.sha256, 0, len(raw), version.sha256)
            return (_EncodedSpan(span, _frozen(row)),)
        text = raw.decode("utf8")
        width = self._source_chars
        chunks = (_content_chunks(text, width) if self._source_chunking == "content"
                  else (text[start:start + width] for start in range(0, len(text), width)))
        retained = {(item.span.span_sha256, item.input_sha256): item.row for item in previous_text}
        tokenizer_options = {"padding": False} if self._source_chunking == "content" else {}
        offset = 0
        encoded = []
        for chunk in chunks:
            data = chunk.encode("utf8")
            stop = offset + len(data)
            digest = sha256(data)
            span = SourceSpan(source, version.sha256, offset, stop, digest)
            ids = tuple(self._tokenizer.encode(chunk, add_special_tokens=True, truncation=False, **tokenizer_options))
            self._check_input_ids(ids, self._context_tokens)
            input_sha256 = sha256(_encode(ids)) if self._source_chunking == "content" else None
            key = digest, input_sha256
            row = retained.get(key)
            if row is None:
                hidden, _ = self._backbone(self._embed_ids(ids), retain_cache=False)
                row = _frozen(self._encode_hidden(hidden))
                if input_sha256 is not None:
                    retained[key] = row
            encoded.append(_EncodedSpan(span, row, input_sha256))
            offset = stop
        return tuple(encoded)

    def _encode_hidden(self, hidden):
        with np.errstate(over="ignore"):
            return np.tanh(hidden[0, 0].double().numpy() / self._scale)

    def _check_memory(self, memory, *, for_read=False):
        if type(memory) is not NativeSourceMemory or memory._owner is not self._owner:
            raise ValueError("source memory belongs to a different native reader")
        if not for_read and (memory._omitted_continuation is not None or memory._source_anchor_seed is not None):
            raise ValueError("an intervention view is read-only")

    def publish(self, epoch_directory: Path, seal: str, *, previous=None, max_source_bytes=None):
        """Encode only changed sources from one verified, query-blind feed epoch."""
        _require_cpu_eval(self._model)
        delta = self._source_revision(epoch_directory, seal, previous, max_source_bytes)
        return self._apply_source_revision(delta, previous)

    def _source_revision(self, directory, seal, previous, max_bytes):
        if previous is not None:
            self._check_memory(previous)
            if previous.omitted_sources or previous._derived_off:
                raise ValueError("an ablated contribution view cannot publish a new epoch")
        if self._source_ingress_sha256 is not None and self._source_ingress is None:
            raise ValueError("publication requires the configured source ingress")
        return read_source_delta(directory, seal, previous=None if previous is None else previous._scope,
                                 max_bytes=max_bytes, allow_binary=self._source_ingress is not None)

    def _apply_source_revision(self, delta, previous):
        import torch

        fingerprint = None
        if self._source_chunking == "content":
            from experiments.memory.native_publication import _reader_digest

            fingerprint = _reader_digest(self)
            if previous is not None and fingerprint != previous._reader_fingerprint_sha256:
                raise ValueError("content-defined source reuse requires the unchanged opening reader")
        sources = {} if previous is None else dict(previous._sources)
        bank = None
        retire, add = [], []
        with torch.inference_mode():
            for source, raw in delta.changes.items():
                old_rows = sources.pop(source, ())
                if self._selector is None:
                    retire.extend(item.row for item in old_rows)
                if raw is not None:
                    was_text = previous is not None and source not in previous._scope.binary_sources
                    reusable = old_rows if fingerprint is not None and was_text else ()
                    encoded = self._encode_source(source, raw, delta.manifest[source],
                                                  binary=source in delta.binary_sources, previous_text=reusable)
                    sources[source] = encoded
                    if self._selector is None:
                        add.extend(item.row for item in encoded)
        if self._selector is None:
            bank = self._moments._empty if previous is None else previous._bank
            if retire or add:
                bank = self._moments.revise(bank, remove=retire or None, append=add or None)
        scope = replace(delta, changes=MappingProxyType({}))
        memory = NativeSourceMemory(self._owner, scope, bank, MappingProxyType(sources),
                                    _reader_fingerprint_sha256=fingerprint)
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
        forecasts, forecast_sets = {}, {}
        if previous is not None:
            previous_priority = previous.priority
            for seed, frontiers in previous.attempted.items():
                forecast = previous.forecasts.get(seed)
                if (seed.source_id not in changed and (forecast is None or forecast.witness.source_id not in changed)
                        and all(frontier_survives(frontier, changed, revised) for frontier in frontiers)):
                    attempted[seed] = frontiers
            for seed, row in previous.acquired.items():
                parent = previous.parents.get(seed)
                check = previous.forecast_sets.get(seed)
                if check is not None:
                    if (any(span.source_id in changed for span in check.witness_spans)
                            or any(rival not in acquired for rival in check.forecast.parents)
                            or not check.forecast._frontiers_survive(changed, revised)):
                        attempted.pop(seed, None)
                        continue
                    if seed in attempted:
                        forecast_sets[seed] = check
                if seed in attempted and (parent is None or parent in acquired):
                    acquired[seed] = row
                    if seed in previous.forecasts:
                        forecasts[seed] = previous.forecasts[seed]
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
                               priority, derived, MappingProxyType(parents), MappingProxyType(forecasts),
                               MappingProxyType(forecast_sets))

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

    def _without_continuation(self, memory, seed):
        self._check_memory(memory)
        if (memory._selective is None or memory._derived_off or memory.omitted_sources
                or not isinstance(seed, SourceSpan) or not memory._is_continuation_leaf(seed)):
            raise ValueError("disabling a continuation requires an installed leaf and intact source bank")
        selective = memory._selective
        shard, = selective.derived
        index = shard.spans.index(seed)
        kept = [position for position in range(len(shard.spans)) if position != index]
        bank = (replace(shard, spans=shard.spans[:index] + shard.spans[index + 1:],
                        keys=_frozen(shard.keys[kept]), values=_frozen(shard.values[kept])),) if kept else ()
        return replace(memory, _selective=replace(selective, derived=bank), _omitted_continuation=seed)

    def _source_anchor_keys(self, memory, seed):
        self._check_memory(memory)
        if (memory._publication_seal is None or memory._selective is None or memory._derived_off
                or memory.omitted_sources or not isinstance(seed, SourceSpan)
                or not memory._is_continuation_leaf(seed)):
            raise ValueError("source-anchor addressing requires a sealed publication and an installed leaf")
        derived, = memory._selective.derived
        source = memory._selective.shards[seed.source_id]
        position = derived.spans.index(seed)
        return derived, position, source.keys[source.spans.index(seed)]

    def _with_source_anchor_address(self, memory, seed):
        """Change one derived selector key for a read-only counterfactual."""
        derived, position, anchor = self._source_anchor_keys(memory, seed)
        keys = derived.keys.copy()
        keys[position] = anchor
        bank = (replace(derived, keys=_frozen(keys)),)
        return replace(memory, _selective=replace(memory._selective, derived=bank), _source_anchor_seed=seed)

    def without_derived(self, memory):
        """Keep the source observations and disable their acquired associations."""
        self._check_memory(memory)
        selective = memory._selective
        if selective is not None:
            selective = replace(selective, acquired=MappingProxyType({}), attempted=MappingProxyType({}),
                                priority=tuple(sorted(selective.shards)), derived=(), parents=MappingProxyType({}),
                                forecasts=MappingProxyType({}), forecast_sets=MappingProxyType({}))
        return replace(memory, _derived=None, _seed_limit=0, derived_seeds=(),
                       _selective=selective, _derived_off=True)

    def _memory_delta(self, hidden, value, mass, gain):
        import torch

        weight = mass / (mass + self._null_weight)
        delta = torch.tensor(value, dtype=hidden.dtype, device=hidden.device)
        return (gain * weight) * delta

    def _continue(self, hidden, cache, memory, steps, *, withhold_continuation_value=None):
        import torch

        run = _LatentRun(hidden, cache)
        if withhold_continuation_value is not None:
            run.value_withholding = NativeValueWithholding(withhold_continuation_value)
        source_steps, derived_reads, derived_steps = [], [], []
        selective = None if memory is None else memory._selective
        shards = () if selective is None else tuple(selective.shards.values())
        for step in range(1, steps + 1):
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
                masked_association = None
                if memory._derived is not None:
                    association, mass = self._moments.read_associations(query, memory._derived)
                    derived_trace = None
                elif selective is not None and not memory._derived_off:
                    if withhold_continuation_value is None:
                        association, mass, trace = self._selector.read(query, selective.derived)
                    else:
                        association, mass, trace, masked_association = self._selector.read_withheld_value(
                            query, selective.derived, withhold_continuation_value,
                        )
                    receipt = replace(trace.receipt, observed_source_seal=memory.seal)
                    derived_reads.append(receipt)
                    derived_trace = replace(trace, receipt=receipt)
                else:
                    mass = 0
                    derived_trace = None
                if mass > 0:
                    previous = thought
                    thought = previous + self._memory_delta(hidden, association, mass, self._derived_gain)
                    if masked_association is not None:
                        masked = previous + self._memory_delta(hidden, masked_association, mass, self._derived_gain)
                        change = float(torch.max(torch.abs(thought.double() - masked.double())))
                        if not isfinite(change):
                            raise FloatingPointError("nonfinite value-withholding effect")
                        run.value_withholding = NativeValueWithholding(
                            withhold_continuation_value, step, change > 0, change,
                        )
                        thought = masked
                        withhold_continuation_value = None
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

    def _check_forecast_reader(self, memory):
        from experiments.memory.native_publication import _reader_digest

        if (memory._publication_seal is None or memory._reader_fingerprint_sha256 is None
                or _reader_digest(self) != memory._reader_fingerprint_sha256):
            raise ValueError("forecast requires an attached publication with an unchanged reader")

    def forecast_next(self, memory, *, witness_source, features, tolerance, max_parent_scores):
        """Draft one continuation before its declared validation source exists."""
        import torch

        _require_cpu_eval(self._model)
        self._check_memory(memory)
        selective = memory._selective
        if selective is None or memory.omitted_sources or memory._derived_off:
            raise ValueError("a forecast needs an intact selective world")
        if self._pending_forecast is not None:
            raise ValueError("this reader already holds a pending forecast")
        self._check_forecast_reader(memory)
        if (type(witness_source) is not str or not _OPAQUE_ID.fullmatch(witness_source)
                or witness_source in memory._scope.manifest):
            raise ValueError("a forecast witness needs a currently absent source identity")
        lookup = self._model.get_input_embeddings().weight
        features = tuple(features)
        _forecast_fields(features, tolerance, lookup.shape[1])
        if type(max_parent_scores) is not int or max_parent_scores <= 0:
            raise ValueError("parent discovery needs a positive score budget")
        if not selective.derived or not selective.priority:
            return None
        if max_parent_scores < len(selective.acquired) * self._steps:
            raise ValueError("parent score budget cannot cover one pending seed")
        source = selective.priority[0]
        item = next(row for row in memory._sources[source] if row.span not in selective.attempted)
        source_only = replace(memory, _selective=replace(selective, derived=()))
        with torch.inference_mode():
            address, baseline = self._run_seed(item, source_only, lookup)
            candidate = self._discover_continuation(item, memory, lookup, address, baseline)
            if candidate is None:
                return None
        parent, run, value, baseline_value = candidate
        control = _forecast_values(baseline_value, features)
        prediction = _forecast_values(value, features)
        if prediction == control:
            return None
        forecast = NativeContinuationForecast(item.span, parent, witness_source, features, tolerance,
                                              control, prediction, memory,
                                              _Continuation(_frozen(address), _frozen(value)),
                                              run.source_steps, baseline.source_steps)
        with self._forecast_lock:
            if self._pending_forecast is not None:
                raise ValueError("this reader already holds a pending forecast")
            self._pending_forecast = forecast
        return forecast

    def cancel_forecast(self, forecast):
        """Release the opening state and invalidate its uncommitted forecast."""
        with self._forecast_lock:
            if forecast is None or self._pending_forecast is not forecast:
                raise ValueError("forecast is no longer pending on this reader")
            self._pending_forecast = None
            self._forecast_cursor = None

    def resolve_forecast(self, memory, forecast, epoch_directory, seal, *, max_source_bytes=None):
        """Apply one verified source epoch and admit its already-drafted operation if confirmed."""
        _require_cpu_eval(self._model)
        if type(forecast) is not NativeContinuationForecast or self._pending_forecast is not forecast:
            raise ValueError("forecast is no longer pending on this reader")
        if forecast._opening is not memory:
            raise ValueError("forecast belongs to a different opening world")
        self._check_forecast_reader(memory)
        delta = self._source_revision(epoch_directory, seal, memory, max_source_bytes)
        self.cancel_forecast(forecast)
        revised = self._apply_source_revision(delta, memory)
        witnesses = revised._sources.get(forecast.witness_source, ())
        if not witnesses:
            return NativeForecastResolution(revised, "witness_absent")
        if len(witnesses) != 1:
            return NativeForecastResolution(revised, "unscorable_witness")
        witness, = witnesses
        actual = _forecast_values(witness.row, forecast.features)
        if _forecast_error(forecast.baseline, actual) <= forecast.tolerance:
            return NativeForecastResolution(revised, "baseline_confirmed", actual)
        if _forecast_error(forecast.prediction, actual) > forecast.tolerance:
            return NativeForecastResolution(revised, "disconfirmed", actual)
        selective = revised._selective
        changed = delta.changes
        changed_shards = tuple(selective.shards[source] for source in changed if source in selective.shards)
        frontiers = chain(forecast._baseline_steps, forecast._steps)
        if (forecast.seed.source_id in changed or forecast.parent not in selective.acquired
                or any(not frontier_survives(step, changed, changed_shards) for step in frontiers)):
            return NativeForecastResolution(revised, "dependencies_changed", actual)
        check = _ForecastCheck(memory.seal, delta.epoch, witness.span, forecast.features,
                               forecast.tolerance, forecast.baseline)
        admitted = self._retain_forecast(revised, forecast.seed, forecast.parent, forecast._steps,
                                          forecast._continuation, check)
        return NativeForecastResolution(admitted, "admitted", actual)

    def forecast_set(self, memory, *, witnesses, max_candidates, max_parent_scores):
        """Freeze multiple latent trajectories before the declared witnesses arrive."""
        import torch

        _require_cpu_eval(self._model)
        self._check_memory(memory)
        selective = memory._selective
        if selective is None or memory.omitted_sources or memory._derived_off:
            raise ValueError("a forecast set needs an intact selective world")
        self._check_forecast_reader(memory)
        lookup = self._model.get_input_embeddings().weight
        witnesses = tuple(witnesses)
        _validate_witnesses(witnesses, lookup.shape[1])
        openings = tuple(self._witness_opening(memory, spec) for spec in witnesses)
        revision_openings = openings if any(opening is not None for opening in openings) else ()
        if (type(max_candidates) is not int or not 2 <= max_candidates <= _MAX_ARMS
                or type(max_parent_scores) is not int or not 1 <= max_parent_scores < 2 ** 63
                or max_parent_scores < len(selective.acquired) * self._steps):
            raise ValueError("forecast set needs a bounded complete parent scan and two to eight candidate slots")

        with self._forecast_lock, torch.inference_mode():
            if self._pending_forecast is not None:
                raise ValueError("this reader already holds a pending forecast")
            if not selective.priority or len(selective.acquired) < 2:
                return None
            source = selective.priority[0]
            item = next(row for row in memory._sources[source] if row.span not in selective.attempted)
            source_only = replace(memory, _selective=replace(selective, derived=()))
            address, baseline = self._run_seed(item, source_only, lookup)
            bank, = selective.derived
            parents = self._selector.parent_candidates(item.span, baseline.source_steps, bank, max_candidates)
            baseline_value = self._encode_hidden(baseline.hidden)
            control = tuple(_witness_prediction(spec, baseline_value) for spec in witnesses)
            alternatives, payloads = [], []
            for parent in parents:
                replay = self._replay_parent(item, memory, lookup, address, baseline_value, parent,
                                             require_complete_lineage=True)
                if replay is None:
                    continue
                run, value = replay
                payload = _Continuation(_frozen(address), _frozen(value))
                predictions = tuple(_witness_prediction(spec, value) for spec in witnesses)
                alternatives.append(_ForecastAlternative(parent, _payload_digest(selective.acquired[parent]),
                                                          _payload_digest(payload), predictions, run.source_steps))
                payloads.append(payload)
            if len(alternatives) < 2:
                return None
            forecast = NativeForecastSet(item.span, witnesses, memory.seal, memory._scope.epoch,
                                         memory._publication_seal, memory._reader_fingerprint_sha256,
                                         control, baseline.source_steps, tuple(alternatives),
                                         max_candidates, max_parent_scores, revision_openings)
            self._pending_forecast = forecast
            self._forecast_cursor = _ForecastCursor(memory, tuple(payloads))
            return forecast

    @staticmethod
    def _witness_opening(memory, spec):
        source = spec.source_id
        version = memory._scope.manifest.get(source)
        if not isinstance(spec, NativeRevisionWitness):
            if version is not None:
                raise ValueError("new-source witnesses must be absent at the opening")
            return None
        if version is None or source in memory._scope.binary_sources:
            raise ValueError("a revision witness needs an existing text source")
        rows = memory._sources[source]
        if len(rows) > spec.max_opening_rows:
            raise ValueError("revision witness exceeds its opening row allowance")
        if isinstance(spec, NativeWeightedRevisionWitness):
            if not rows:
                raise ValueError("weighted revision witness needs a nonempty opening source")
            return _WeightedWitnessOpening(
                version.sha256, version.generation, tuple(item.span.span_sha256 for item in rows),
                _measure_witness(spec, rows),
            )
        return _RevisionWitnessOpening(version.sha256, version.generation, tuple(
            (item.span.span_sha256, _forecast_values(item.row, spec.features)) for item in rows
        ))

    def advance_forecast_set(self, memory, forecast, epoch_directory, seal, *, commitment, max_source_bytes=None):
        """Compare one direct source revision against the opening forecast set."""
        _require_cpu_eval(self._model)
        with self._forecast_lock:
            if type(forecast) is not NativeForecastSet or self._pending_forecast is not forecast:
                raise ValueError("forecast set is no longer pending on this reader")
            cursor = self._forecast_cursor
            if cursor.memory is not memory or commitment != forecast.commitment:
                raise ValueError("forecast set opening or commitment differs")
            from experiments.memory.native_publication import _reader_digest

            if _reader_digest(self) != forecast.reader_sha256:
                raise ValueError("forecast set reader changed")
            delta = self._source_revision(epoch_directory, seal, memory, max_source_bytes)
            revised = self._apply_source_revision(delta, memory)
            result, next_cursor = self._score_forecast_set(forecast, cursor, revised, delta)
            self._forecast_cursor = next_cursor
            if next_cursor is None:
                self._pending_forecast = None
            return result

    def _score_forecast_set(self, forecast, cursor, revised, delta):
        index = len(cursor.observations)
        spec = forecast.witnesses[index]
        observations = cursor.observations
        selective = revised._selective
        shards = tuple(selective.shards[source] for source in delta.changes if source in selective.shards)
        if (forecast.seed.source_id in delta.changes or any(span.source_id in delta.changes for span in cursor.witness_spans)
                or any(parent not in selective.acquired for parent in forecast.parents)
                or not forecast._frontiers_survive(delta.changes, shards)):
            return NativeForecastSetResolution(revised, "dependencies_changed", (), observations), None
        if any(later.source_id in delta.changes for later in forecast.witnesses[index + 1:]):
            return NativeForecastSetResolution(revised, "witness_out_of_order", (), observations), None
        opening = forecast._opening_at(index)
        status, rows = self._observe_witness(cursor.memory, revised, delta, spec, opening)
        if status is not None:
            return NativeForecastSetResolution(revised, status, (), observations), None
        observations += (_measure_witness(spec, rows),)
        survivors = forecast._survivors(observations)
        witness_spans = (*cursor.witness_spans, *(item.span for item in rows))
        observation_seals = (*cursor.observation_seals, delta.seal)
        if not survivors:
            return NativeForecastSetResolution(revised, "disconfirmed", survivors, observations), None
        if len(observations) < len(forecast.witnesses):
            progress = _ForecastCursor(revised, cursor.payloads, witness_spans, observations, observation_seals)
            return NativeForecastSetResolution(revised, "pending", survivors, observations), progress
        if forecast._matches(forecast.baseline, observations):
            return NativeForecastSetResolution(revised, "baseline_confirmed", survivors, observations), None
        if forecast._persistence_matches(observations):
            return NativeForecastSetResolution(revised, "persistence_confirmed", survivors, observations), None
        if len(survivors) != 1:
            return NativeForecastSetResolution(revised, "ambiguous", survivors, observations), None
        position = forecast.parents.index(survivors[0])
        winner = forecast.alternatives[position]
        seals = observation_seals if forecast._revision_openings else ()
        admitted = self._retain_forecast(revised, forecast.seed, winner.parent, winner.steps, cursor.payloads[position],
                                          _ForecastSetCheck(forecast, witness_spans, seals))
        return NativeForecastSetResolution(admitted, "admitted", survivors, observations), None

    @staticmethod
    def _observe_witness(previous, revised, delta, spec, opening):
        source = spec.source_id
        current = revised._sources.get(source, ())
        if opening is None:
            if not current:
                return "witness_absent", ()
            return (None, current) if len(current) == 1 else ("unscorable_witness", ())
        version = previous._scope.manifest.get(source)
        if (version is None or version.generation != opening.generation
                or version.sha256 != opening.version_sha256):
            return "dependencies_changed", ()
        if source not in delta.changes:
            return "witness_unchanged", ()
        if delta.changes[source] is None:
            return "witness_withdrawn", ()
        if source in delta.binary_sources:
            return "unscorable_witness", ()
        rows = opening.introduced_rows(current, spec.max_changed_rows)
        if rows is None:
            return "witness_change_limit", ()
        return (None, rows) if rows else ("unscorable_witness", ())

    def _retain_forecast(self, memory, seed, parent, steps, continuation, check):
        selective = memory._selective
        attempted = {**selective.attempted, seed: steps}
        acquired = {**selective.acquired, seed: continuation}
        parents = {**selective.parents, seed: parent}
        order = sorted(selective.priority, key=lambda source: source == seed.source_id)
        priority = self._pending_groups(order, memory._sources, attempted)
        field = "forecasts" if isinstance(check, _ForecastCheck) else "forecast_sets"
        checks = MappingProxyType({**getattr(selective, field), seed: check})
        updated = replace(selective, attempted=MappingProxyType(attempted), acquired=MappingProxyType(acquired),
                          parents=MappingProxyType(parents), derived=self._derived_rows(acquired), priority=priority,
                          **{field: checks})
        return replace(memory, _selective=updated, derived_seeds=tuple(acquired))

    def maintain(self, memory, *, max_seeds, parent=None, max_parent_scores=None):
        """Install continuations with one fixed or source-discovered parent per seed."""
        import torch

        _require_cpu_eval(self._model)
        self._check_memory(memory)
        if type(max_seeds) is not int or max_seeds <= 0 or memory.omitted_sources:
            raise ValueError("maintenance needs a positive seed budget and an intact source bank")
        if parent is not None and (not isinstance(parent, SourceSpan) or memory._selective is None
                                   or parent not in memory._selective.acquired):
            raise ValueError("maintenance parent must be an installed continuation at this pin")
        if max_parent_scores is not None:
            if type(max_parent_scores) is not int or max_parent_scores <= 0:
                raise ValueError("parent discovery needs a positive score budget")
            if parent is not None or memory._selective is None or memory._derived_off:
                raise ValueError("parent discovery needs an intact selective world and no fixed parent")
        if memory._selective is not None:
            return self._maintain_selection(memory, max_seeds, parent, max_parent_scores)
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

    def _maintain_selection(self, memory, max_seeds, parent, max_parent_scores):
        import torch

        selective = memory._selective
        discover = max_parent_scores is not None and bool(selective.derived)
        if discover:
            # Each admitted seed can scan the whole opening bank once per latent step.
            max_seeds = min(max_seeds, max_parent_scores // (len(selective.acquired) * self._steps))
            if max_seeds == 0 and selective.priority:
                raise ValueError("parent score budget cannot cover one pending seed")
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
                used_parent = parent if run.applied_derived else None
                value = None
                if discover:
                    candidate = self._discover_continuation(item, memory, lookup, address, run)
                    if candidate is not None:
                        used_parent, run, value, _ = candidate
                attempted[item.span] = run.source_steps
                if used_parent is not None:
                    parents[item.span] = used_parent
                if self._installable(run.source_steps, item.span.source_id, used_parent):
                    if value is None:
                        value = self._encode_hidden(run.hidden)
                    acquired[item.span] = _Continuation(_frozen(address), _frozen(value))
        derived = selective.derived if len(acquired) == len(selective.acquired) else self._derived_rows(acquired)
        worked = {item.span.source_id for item in batch}
        order = sorted(selective.priority, key=worked.__contains__)
        priority = self._pending_groups(order, memory._sources, attempted)
        selective = replace(selective, acquired=MappingProxyType(acquired), derived=derived,
                            attempted=MappingProxyType(attempted), priority=priority, parents=MappingProxyType(parents))
        return replace(memory, _selective=selective, derived_seeds=tuple(acquired),
                       _derived_off=False, _publication_seal=None)

    def _discover_continuation(self, item, memory, lookup, address, baseline):
        selective = memory._selective
        bank, = selective.derived
        parent = self._selector.discover_parent(item.span, baseline.source_steps, bank)
        if parent is None:
            return None
        baseline_value = self._encode_hidden(baseline.hidden)
        replay = self._replay_parent(item, memory, lookup, address, baseline_value, parent)
        return None if replay is None else (parent, *replay, baseline_value)

    def _replay_parent(self, item, memory, lookup, address, baseline_value, parent, *,
                       require_complete_lineage=False):
        selective = memory._selective
        if self._parent_conflicts(selective, parent, item.span.source_id,
                                  require_complete_lineage=require_complete_lineage):
            return None
        inputs = self._derived_rows({parent: selective.acquired[parent]})
        worker_input = replace(memory, _selective=replace(selective, derived=inputs))
        candidate_address, candidate = self._run_seed(item, worker_input, lookup)
        if (not candidate.applied_derived
                or not self._installable(candidate.source_steps, item.span.source_id, parent)
                or not np.array_equal(_quantize(address), _quantize(candidate_address))):
            return None
        candidate_value = self._encode_hidden(candidate.hidden)
        if np.array_equal(_quantize(baseline_value), _quantize(candidate_value)):
            return None
        return candidate, candidate_value

    @staticmethod
    def _parent_conflicts(selective, seed, source_id, *, require_complete_lineage=False,
                          changed_sources=(), revised_shards=()):
        pending, seen = [seed], set()
        while pending:
            seed = pending.pop()
            if seed in seen:
                continue
            seen.add(seed)
            forecast = selective.forecasts.get(seed)
            check = selective.forecast_sets.get(seed)
            if (seed.source_id == source_id or (forecast is not None
                    and (require_complete_lineage or forecast.witness.source_id == source_id))):
                return True
            steps = selective.attempted[seed]
            if check is not None:
                if any(witness.source_id == source_id or witness.source_id in changed_sources
                       for witness in check.witness_spans):
                    return True
                steps = (*steps, *check.forecast._baseline_steps,
                         *(step for arm in check.forecast.alternatives for step in arm.steps))
                pending.extend(check.forecast.parents)
            elif seed in selective.parents:
                pending.append(selective.parents[seed])
            if changed_sources and (seed.source_id in changed_sources or any(
                    not frontier_survives(step, changed_sources, revised_shards) for step in steps)):
                return True
            if any(hit.span.source_id == source_id for step in steps if step.applied for hit in step.receipt.selected):
                return True
        return False

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
            self._check_memory(memory, for_read=True)
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

    def read_token_ids(self, input_ids, *, memory=None, steps=None, max_new_tokens=None,
                       allowed_next_tokens=None, withhold_continuation_value=None):
        """Read an exact, already-tokenized prompt with a caller-owned output cap.

        This is the episode adapter boundary: it consumes the exact prepared
        token sequence rather than decoding and retokenizing its chat frame.
        An optional allowlist callback receives the full prompt-plus-generated
        prefix and returns the vocabulary IDs permitted at the next step.
        For a diagnostic with an intact sealed publication, withhold one leaf's
        value at its first ready selection. Later steps read memory normally.
        """
        _require_cpu_eval(self._model)
        if memory is not None:
            self._check_memory(memory, for_read=withhold_continuation_value is None)
        if withhold_continuation_value is not None and (
                memory is None or memory._publication_seal is None or memory._selective is None
                or memory._derived_off or memory.omitted_sources
                or not isinstance(withhold_continuation_value, SourceSpan)
                or not memory._is_continuation_leaf(withhold_continuation_value)):
            raise ValueError("withholding a value requires an installed leaf in an intact sealed publication")
        steps = self._step_budget(steps)
        output_limit = self._output_tokens if max_new_tokens is None else max_new_tokens
        if type(output_limit) is not int or not 1 <= output_limit <= self._output_tokens:
            raise ValueError("output allowance exceeds the configured native bound")
        if allowed_next_tokens is not None and not callable(allowed_next_tokens):
            raise TypeError("next-token allowlist must be callable")
        input_ids = tuple(input_ids)
        self._check_input_ids(input_ids, self._context_tokens - steps - output_limit)
        return self._read_ids_with_limit(input_ids, memory, steps, output_limit,
                                         allowed_next_tokens=allowed_next_tokens,
                                         withhold_continuation_value=withhold_continuation_value)

    def _read_ids_with_limit(self, input_ids, memory, steps, output_limit, *, allowed_next_tokens=None,
                             withhold_continuation_value=None):
        import torch

        with torch.inference_mode():
            hidden, cache = self._backbone(self._embed_ids(input_ids), retain_cache=True)
            run = self._continue(hidden, cache, memory, steps, withhold_continuation_value=withhold_continuation_value)
            hidden, cache = run.hidden, run.cache
            ids = []
            for step in range(output_limit):
                logits = self._model.get_output_embeddings()(hidden)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite native vocabulary scores")
                scores = logits[0, -1]
                if allowed_next_tokens is not None:
                    allowed = allowed_next_tokens((*input_ids, *ids))
                    if isinstance(allowed, (str, bytes)):
                        raise TypeError("next-token allowlist must contain token IDs")
                    try:
                        allowed = tuple(allowed)
                    except TypeError as error:
                        raise TypeError("next-token allowlist must be iterable") from error
                    if (not allowed or any(type(token) is not int or not 0 <= token < scores.shape[0]
                                           for token in allowed)
                            or len(set(allowed)) != len(allowed)):
                        raise ValueError("next-token allowlist must contain unique in-vocabulary IDs")
                    allowed = tuple(sorted(allowed))
                    choices = torch.tensor(allowed, dtype=torch.long, device=scores.device)
                    token = allowed[int(scores.index_select(0, choices).argmax())]
                else:
                    token = int(scores.argmax())
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
                                run.source_steps, run.derived_steps,
                                None if memory is None else memory._omitted_continuation,
                                None if memory is None else memory._source_anchor_seed,
                                run.value_withholding)
