"""Conditional scalar decisions outside a retained operation's exact span."""

from dataclasses import dataclass, replace
from fractions import Fraction
from struct import unpack

from experiments.memory.exact_latent_linear import dot, gram, solve_gram
from experiments.memory.linear_feedback import _LEVELS
from experiments.memory.native_operator_inventory import NativeOperatorInventoryModel, _entries
from experiments.memory.verified_source_tool import SourceSpan


@dataclass(frozen=True, slots=True)
class NativeOperandConsequence:
    status: str
    publication_seal: str
    source_seal: str
    address: tuple[int, ...]
    query_address: tuple[int, ...]
    query: bytes
    coordinate: int
    threshold: Fraction
    gain_squared: Fraction
    omitted_continuation: SourceSpan | None = None
    derived_omitted: bool = False
    cohort_spans: tuple[SourceSpan, ...] = ()
    witness: SourceSpan | None = None
    projected_l1: Fraction | None = None
    residual_squared: Fraction | None = None
    minimum_gain_squared: Fraction | None = None
    center: Fraction | None = None
    radius_squared: Fraction | None = None


def check_native_operand_consequence(reader, memory, *, address, query, coordinate, threshold, gain_squared):
    """Bound one witnessed output coordinate for a known address, in Q8 grid units.

    The caller supplies a prior squared Euclidean cap on the linear output row.
    The packed query comes from the existing read boundary. The known address
    can differ from the query address to separate operand checks from routing.
    """
    result, problem = _consequence_problem(reader, memory, address=address, query=query, coordinate=coordinate,
                                           threshold=threshold, gain_squared=gain_squared)
    if problem is None:
        return result
    return _bound_consequence(result, *problem, coefficient_limit=reader._selector._transport_l1)


def _consequence_problem(reader, memory, *, address, query, coordinate, threshold, gain_squared):
    if type(reader) is not NativeOperatorInventoryModel:
        raise TypeError("operand consequence requires the native inventory reader")
    reader._check_memory(memory, for_read=True)
    if memory._publication_seal is None or memory.omitted_sources or memory._source_anchor_seed is not None:
        raise ValueError("operand consequence requires an attached original source and address view")
    axes = reader._selector._operand_address_axes
    width = reader._moments._query.shape[1]
    if (type(address) is not tuple or len(address) != len(axes)
            or any(type(value) is not int or abs(value) > _LEVELS for value in address)
            or type(query) is not bytes or len(query) != width or 128 in query
            or type(coordinate) is not int or not 0 <= coordinate < reader._moments._width):
        raise ValueError("operand consequence requires a Q8 address, query and valid output coordinate")
    if type(threshold) not in (int, Fraction) or type(gain_squared) not in (int, Fraction) or gain_squared < 0:
        raise ValueError("operand consequence requires a rational threshold and nonnegative squared gain")
    packed = unpack(f"{width}b", query)
    result = NativeOperandConsequence(
        "unavailable", memory._publication_seal, memory.seal, address,
        tuple(packed[axis] for axis in axes), query, coordinate, Fraction(threshold), Fraction(gain_squared),
        omitted_continuation=memory._omitted_continuation, derived_omitted=memory._derived_off,
    )
    entries = _entries(memory._selective)
    entry = next((entry for entry in entries if entry.address == address), None)
    groups = reader._operand_groups(memory, include_observations=False)
    prepared = groups[0].by_address.get(address) if groups else None
    if entry is None or not prepared:
        return result, None
    if coordinate not in entry.admission.draft.coordinates:
        return replace(result, status="unwitnessed_coordinate"), None
    inventory, = prepared
    if len(inventory.groups) != 1:
        return replace(result, status="unsupported_basis"), None
    group, = inventory.groups
    if tuple(group.address) != address or group.coefficient_map is None:
        return replace(result, status="unsupported_basis"), None

    operand = tuple(packed[axis] for axis in reader._selector._operands)
    keys = tuple(tuple(map(int, column)) for column in group.keys.T)
    values = tuple(map(int, group.values[:, coordinate]))
    result = replace(result, cohort_spans=group.spans, witness=entry.admission.witness)
    return result, (operand, keys, values)


def _bound_consequence(result, operand, keys, values, *, coefficient_limit):
    system = gram(keys)
    rhs = [dot(key, operand) for key in keys]
    coefficients = solve_gram(system, rhs)
    projected_l1 = sum(map(abs, coefficients))
    residual_squared = dot(operand, operand) - dot(rhs, coefficients)
    result = replace(result, projected_l1=projected_l1, residual_squared=residual_squared)
    if projected_l1 > Fraction(coefficient_limit):
        return replace(result, status="coefficient_limit")

    minimum_gain_squared = dot(values, solve_gram(system, values))
    result = replace(result, minimum_gain_squared=minimum_gain_squared)
    headroom = result.gain_squared - minimum_gain_squared
    if headroom < 0:
        return replace(result, status="inconsistent_gain")
    center = dot(values, coefficients)
    radius_squared = headroom * residual_squared
    margin = center - result.threshold
    status = "unresolved"
    if margin ** 2 > radius_squared:
        status = "above" if margin > 0 else "below"
    return replace(result, status=status, center=center, radius_squared=radius_squared)
