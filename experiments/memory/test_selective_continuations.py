import numpy as np

from experiments.memory.linear_feedback import LinearFeedbackKernel
from experiments.memory.selective_continuations import RetainedSelector, frontier_survives
from experiments.memory.verified_source_tool import SourceSpan


def _selector(width, *, top_k, tie_limit=16):
    maps = np.eye(width)
    kernel = LinearFeedbackKernel(query=maps, carry=None, observation_key=maps, observation_value=maps,
                                  feedback_key=maps, feedback_value=maps, max_steps=1)
    return kernel, RetainedSelector(kernel, top_k=top_k, tie_limit=tie_limit, temperature=0.04)


def _span(source, offset=0):
    return SourceSpan(source, "1" * 64, offset, offset + 1, "2" * 64)


def test_exact_selection_preserves_the_planted_value_in_24000_retained_rows():
    rng = np.random.default_rng(17)
    width, count = 64, 24_000
    keys = rng.normal(size=(count, width))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    target = count // 2
    clean = keys[target]
    direction = rng.normal(size=width)
    noise = direction - (direction @ clean) * clean
    query = 0.8 * clean + 0.6 * noise / np.linalg.norm(noise)
    values = np.zeros_like(keys)
    values[target, 0] = 1
    kernel, selector = _selector(width, top_k=8)
    spans = tuple(_span("synthetic-document", offset) for offset in range(count))
    shard = selector.shard(spans, keys, values)
    read, _, trace = selector.read(query, (shard,))
    assert trace.receipt.status == "ready" and trace.receipt.rows_scored == count
    assert trace.receipt.selected[0].span == spans[target]
    assert trace.receipt.eligible == 8 and read[0] > 0.99
    dense, _ = kernel.read_associations(query, kernel.prepare_associations(keys, values))
    assert 0 < dense[0] < 0.0002


def test_cutoff_ties_carry_witnesses_and_refuse_overflow():
    _, selector = _selector(2, top_k=1, tie_limit=2)
    addresses, values = [[1, 0], [1, 0], [1, 0]], [[0, 1], [0, -1], [0, 1]]
    shards = tuple(selector.shard([_span(str(i))], [a], [v])
                   for i, (a, v) in enumerate(zip(addresses, values, strict=True)))
    value, mass, trace = selector.read([1, 0], shards[:2])
    assert value is None and mass == 0
    assert trace.receipt.status == "competing" and trace.receipt.eligible == 2
    assert [hit.span.source_id for hit in trace.receipt.selected] == ["0", "1"]
    value, mass, trace = selector.read([1, 0], shards)
    assert value is None and mass == 0
    assert trace.receipt.status == "overflow" and trace.receipt.eligible == 3
    assert len(trace.receipt.selected) == 2


def test_overflow_preserves_strict_winners_and_margin_checks_near_cutoff():
    kernel, selector = _selector(2, top_k=2, tie_limit=2)
    source_a = [_span(str(i)) for i in range(3)]
    weaker = selector.shard(source_a, [[0.5, 0]] * 3, [[0, 1]] * 3)
    stronger = selector.shard([_span("last")], [[1, 0]], [[0, -1]])
    _, _, overflow = selector.read([1, 0], (weaker, stronger))
    assert overflow.receipt.status == "overflow"
    assert [hit.span.source_id for hit in overflow.receipt.selected] == ["last", "0"]

    guarded = RetainedSelector(kernel, top_k=1, tie_limit=2, temperature=0.04, score_margin=127)
    neighbor = selector.shard([_span("neighbor")], [[126 / 127, 0]], [[0, 1]])
    _, _, opening = guarded.read([1, 0], (stronger,))
    assert not frontier_survives(opening, {"neighbor"}, [neighbor])
    _, mass, contested = guarded.read([1, 0], (stronger, neighbor))
    assert contested.receipt.status == "competing" and mass == 0
    assert [hit.span.source_id for hit in contested.receipt.selected] == ["last", "neighbor"]


def test_parent_discovery_compares_distinct_parents_across_queries_and_refuses_a_tie():
    kernel, selector = _selector(2, top_k=1)
    own, early, late, placebo = [_span(source) for source in ("own", "early", "late", "placebo")]
    _, _, first_step = selector.read([1, 0], ())
    _, _, second_step = selector.read([0, 1], ())
    opening = selector.shard((own, early, late), [[1, 0], [0.5, 0], [0, 0.75]], [[0, 1]] * 3)
    assert selector.discover_parent(own, (first_step,), opening) == early
    assert selector.discover_parent(own, (first_step, second_step), opening) == late

    tied = selector.shard((early, late, placebo), [[0.5, 0], [0, 0.75], [0, 0.75]],
                         [[0, 1], [1, 0], [-1, 0]])
    assert selector.discover_parent(own, (first_step, second_step), tied) is None
    conservative = RetainedSelector(kernel, top_k=1, tie_limit=2, temperature=0.1, score_margin=5_000)
    assert conservative.discover_parent(own, (first_step, second_step), opening) is None

    _, _, conflict = selector.read([0, 1], (tied,))
    assert conflict.receipt.status == "competing"
    assert selector.discover_parent(own, (first_step, conflict), opening) is None


def test_revision_keeps_selected_world_only_when_changed_rows_stay_below_old_cutoff():
    _, selector = _selector(2, top_k=2)
    initial = selector.shard([_span("a"), _span("b")], [[1, 0], [0, 1]], [[1, 1], [-1, -1]])
    _, _, step = selector.read([1, 0], (initial,))
    assert step.cutoff == 1 and step.receipt.eligible == 1
    neutral = selector.shard([_span("c")], [[0, -1]], [[0, 1]])
    weak = selector.shard([_span("c")], [[1 / 127, 0]], [[0, 1]])
    assert frontier_survives(step, {"c"}, [neutral])
    assert not frontier_survives(step, {"c"}, [weak])
    assert not frontier_survives(step, {"a"}, [])
    _, _, empty = selector.read([-1, 0], (initial,))
    assert empty.cutoff == 1 and empty.receipt.status == "empty"
    assert frontier_survives(empty, {"c"}, [weak])

    _, top_one = _selector(2, top_k=1)
    value, mass, full = top_one.read([1, 0], (initial,))
    assert full.cutoff == 127 ** 2 and frontier_survives(full, {"c"}, [weak])
    rebuilt, new_mass, new_step = top_one.read([1, 0], (initial, weak))
    np.testing.assert_array_equal(rebuilt, value)
    assert new_mass == mass and new_step.receipt.selected == full.receipt.selected
