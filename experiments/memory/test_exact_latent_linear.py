from fractions import Fraction

import pytest

from experiments.memory.exact_latent_linear import LinearOperatorEvidence, dot, gram, solve_gram


def test_one_scalar_preserves_only_supported_downstream_consequences():
    first = (1, 0)
    second = (0, 1)
    combined = (1, 1)
    opening = LinearOperatorEvidence(2, 2, 3).observe(first, first, 1).observe(first, second, 0)
    scalar = opening.observe(second, first, 1)
    query = (2, 1)
    assert opening.residual_reduction(query, combined, candidate_operand=second,
                                       candidate_output_form=(2, -1)) == Fraction(1, 5)
    measured = scalar.bound(query, first)
    assert measured.center == 3 and measured.radius_squared == 0
    downstream = scalar.bound(query, combined)
    assert (downstream.center, downstream.radius_squared, downstream.residual_squared) == (3, 1, 1)
    assert downstream.strict_side(3) == downstream.strict_side(2) == "unresolved"
    assert downstream.strict_side(Fraction(3, 2)) == "above"

    assert scalar.residual_reduction(query, combined, candidate_operand=second,
                                     candidate_output_form=first) == 0
    assert scalar.residual_reduction(query, combined, candidate_operand=second,
                                     candidate_output_form=combined) == 1
    positive = scalar.observe(second, combined, 2)
    negative = scalar.observe(second, combined, 0)
    for evidence, center, side in ((positive, 4, "above"), (negative, 2, "below")):
        closed = evidence.bound(query, combined)
        assert closed.center == center and closed.radius_squared == 0
        assert closed.strict_side(3) == side
        assert evidence.bound(query, first) == measured
    assert scalar.rank == 3 and opening.rank == 2
    assert scalar.bound(query, combined) == downstream
    with pytest.raises(ValueError, match="declared norm cap"):
        scalar.observe(second, combined, 3)


def test_exhausted_norm_cap_can_settle_an_off_span_read():
    evidence = LinearOperatorEvidence(2, 1, 1).observe((1, 0), (1,), 1)
    off_span = evidence.bound((0, 1), (1,))
    assert (off_span.center, off_span.radius_squared, off_span.residual_squared) == (0, 0, 1)
    assert off_span.strict_side(Fraction(1, 2)) == "below"


def test_incremental_gram_matches_a_dense_exact_map_with_nonorthogonal_observations():
    operator = ((2, -1, 1), (1, 2, 3))
    probes = (((1, 1, 0), (1, 1)), ((0, 1, 1), (1, 1)),
              ((1, 0, 1), (2, -1)), ((2, -1, 1), (1, 2)))

    def value(operand, output):
        return dot(output, tuple(dot(row, operand) for row in operator))

    evidence = LinearOperatorEvidence(3, 2, 20)
    for operand, output in probes:
        evidence = evidence.observe(operand, output, value(operand, output))

    def flattened(operand, output):
        return tuple(weight * coordinate for weight in output for coordinate in operand)

    operand, output = (Fraction(1, 2), -2, 3), (2, 3)
    query = flattened(operand, output)
    rows = [flattened(*probe) for probe in probes]
    matrix = gram(rows)
    values = [value(*probe) for probe in probes]
    cross = [dot(row, query) for row in rows]
    minimum = dot(values, solve_gram(matrix, values))
    residual = dot(query, query) - dot(cross, solve_gram(matrix, cross))
    bound = evidence.bound(operand, output)
    assert evidence.minimum_norm_squared == minimum
    assert bound.center == dot(solve_gram(matrix, values), cross)
    assert bound.radius_squared == (20 - minimum) * residual

    assert evidence.observe((1, 2, 1), (1, 1), value((1, 2, 1), (1, 1))) is evidence
    with pytest.raises(ValueError, match="contradicts"):
        evidence.observe((1, 2, 1), (1, 1), value((1, 2, 1), (1, 1)) + 1)
