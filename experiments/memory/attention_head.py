"""Reuse one bias-free grouped-query head in a continuous-feedback kernel."""

from math import sqrt
from operator import index

from experiments.memory.latent_feedback import FactoredValue, FeedbackKernel, _real_array


def feedback_from_attention_head(*, query, key, value, output, query_heads,
                                 key_value_heads, head, max_steps):
    """Extract one head from complete Linear weight arrays of one layer.

    Weights use (out_features, in_features) axes. The query and key/value
    groups are contiguous. The resulting kernel uses raw projection geometry,
    with no positional rotation, causal mask, residual, or sum across heads.
    Its pre-tanh contribution is the selected head evaluated on supplied
    coordinates at equal relative positions, including earlier feedback rows.
    """
    query_heads, key_value_heads, head = map(index, (query_heads, key_value_heads, head))
    if (query_heads <= 0 or key_value_heads <= 0 or query_heads % key_value_heads
            or not 0 <= head < query_heads):
        raise ValueError("invalid grouped-query head selection")
    query, key, value, output = map(_real_array, (query, key, value, output))
    if query.ndim != 2 or not all(query.shape) or query.shape[0] % query_heads:
        raise ValueError("invalid grouped-query projection dimensions")
    head_width = query.shape[0] // query_heads
    state_width = query.shape[1]
    kv_shape = (key_value_heads * head_width, state_width)
    if key.shape != kv_shape or value.shape != kv_shape or output.shape != query.shape[::-1]:
        raise ValueError("inconsistent grouped-query projection dimensions")

    group = head // (query_heads // key_value_heads)
    query_rows = slice(head * head_width, (head + 1) * head_width)
    kv_rows = slice(group * head_width, (group + 1) * head_width)
    query_map = query[query_rows].T / sqrt(head_width)
    key_map = key[kv_rows].T
    value_map = FactoredValue(value[kv_rows].T, output[:, query_rows].T)
    return FeedbackKernel(
        query=query_map, carry=None,
        observation_key=key_map, observation_value=value_map,
        feedback_key=key_map, feedback_value=value_map, max_steps=max_steps,
    )
