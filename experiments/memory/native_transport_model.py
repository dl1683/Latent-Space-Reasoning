"""Opt-in direct native read that applies a signed fit to retained worker rows."""

from dataclasses import replace

from experiments.memory.native_latent_recurrence import NativeLatentModel
from experiments.memory.selective_continuations import RetainedSelector, _TransportShard


class NativeTransportModel(NativeLatentModel):
    """Keep the source-only worker intact and solve only on the late derived read."""

    def __init__(self, model, tokenizer, moments, *, derived_transport_l1, **options):
        super().__init__(model, tokenizer, moments, **options)
        original = self._selector
        if original is None or derived_transport_l1 is None:
            raise ValueError("derived transport requires selective memory and an explicit L1 bound")
        self._selector = RetainedSelector(
            moments, top_k=original._k, tie_limit=original._tie_limit,
            temperature=original._temperature, score_margin=original._margin,
            transport_l1=derived_transport_l1,
        )

    def _read_ids_with_limit(self, input_ids, memory, steps, output_limit, **options):
        if memory is not None:
            self._check_memory(memory, for_read=True)
            selective = memory._selective
            if selective is not None and selective.derived:
                shards = tuple(_TransportShard(row.spans, row.keys, row.values) for row in selective.derived)
                memory = replace(memory, _selective=replace(selective, derived=shards))
        return super()._read_ids_with_limit(input_ids, memory, steps, output_limit, **options)
