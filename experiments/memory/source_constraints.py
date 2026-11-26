"""Compile exact, scoped linear consequences before a task enters the latent world."""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import re

from experiments.memory.relational_feedback import Expression, combine
from experiments.memory.verified_source_tool import SourceSpan
from experiments.memory.world_transport import WorldWitness, _source_ids


MAX_CONSTRAINT_SOURCE_BYTES = 4096
_MAX_AXES = 16
_MAX_SOURCES = 32
_RATIONAL = re.compile(r"-?[0-9]{1,7}(/[1-9][0-9]{0,6}|\.[0-9]{1,7})?")
_SURROGATE = re.compile(r"[\ud800-\udfff]")


@dataclass(frozen=True)
class SourceConstraintPolicy:
    """Bound one declared set of input coordinates and downstream channels."""

    scope: tuple[str, str, str, str, str]
    input_axes: tuple[int, ...]
    output_axes: tuple[int, ...]
    middle_width: int
    authorities: tuple[str, ...]
    contexts: tuple[str, ...]

    def __post_init__(self):
        if (type(self.scope) is not tuple or len(self.scope) != 5
                or any(type(item) is not str or not item or len(item) > 128 or _SURROGATE.search(item)
                       for item in self.scope)):
            raise ValueError("constraint scope requires identity, input frame, intermediate frame, mode and unit")
        for axes in (self.input_axes, self.output_axes):
            if (type(axes) is not tuple or not 1 <= len(axes) <= _MAX_AXES
                    or any(type(axis) is not int or axis < 0 for axis in axes)
                    or len(set(axes)) != len(axes)):
                raise ValueError("constraint axes must be distinct nonnegative coordinates")
        if type(self.middle_width) is not int or not 1 <= self.middle_width <= _MAX_AXES:
            raise ValueError("invalid constraint intermediate width")
        for name in ("authorities", "contexts"):
            group = _source_ids(getattr(self, name))
            if len(group) > _MAX_SOURCES:
                raise ValueError("constraint source allowance exceeded")
            object.__setattr__(self, name, group)
        if not set(self.authorities).isdisjoint(self.contexts):
            raise ValueError("constraint authorities and activating contexts must be disjoint")

    def wire(self):
        return (self.scope, self.input_axes, self.output_axes, self.middle_width, self.authorities, self.contexts)

    def require_frame(self, pin):
        if max(*self.input_axes, *self.output_axes) >= pin.memory._anchors.shape[1]:
            raise ValueError("constraint coordinates exceed the latent frame")
        links = pin.memory._role_links
        validity = None if links is None else links.validity
        if validity is not None and not validity.dependencies.isdisjoint((*self.authorities, *self.contexts)):
            raise ValueError("effective-time role rules overlap source constraint scope")


@dataclass(frozen=True)
class ConstraintUse:
    publication_sha256: str
    # Each modeled channel carries the law and its contributing measurements.
    evidence: tuple[tuple[int, tuple[WorldWitness, ...]], ...]
    open_axes: tuple[int, ...]
    inconsistent_measurements: tuple[WorldWitness, ...] = ()
    unreconciled_scope: tuple[WorldWitness, ...] = ()


@dataclass(frozen=True, eq=False)
class SourceConstraintPublication:
    policy: SourceConstraintPolicy
    kernel_sha256: str
    source_versions: tuple[tuple[str, str | None, int | None], ...]
    # Input coefficients and shared nuisance coefficients, in exact arithmetic.
    relations: tuple[tuple[int, tuple[Fraction, ...], tuple[Fraction, ...]], ...]
    contexts: frozenset[str]
    use: ConstraintUse

    def active_on(self, pin):
        return self.source_versions == constraint_versions(pin, self.policy)

    def additions(self, rows):
        inputs = tuple(rows[axis] for axis in self.policy.input_axes)
        nuisance = tuple(Expression.unknown() for _ in range(self.policy.middle_width))
        additions = {axis: combine((*inputs, *nuisance), (*weights, *remainder))
                     for axis, weights, remainder in self.relations}
        return tuple(additions[axis] if axis in additions else
                     Expression.unknown() if axis in self.policy.output_axes else Expression()
                     for axis in range(len(rows)))


def constraint_versions(pin, policy):
    return tuple((source, pin.memory.source_digests.get(source), pin._source_revisions.get(source))
                 for source in (*policy.authorities, *policy.contexts))


def _rational_row(values, width):
    if type(values) is not list or len(values) != width:
        raise ValueError("constraint row has the wrong width")
    row = []
    for value in values:
        if type(value) is int and abs(value) <= 9_999_999:
            row.append(Fraction(value))
        elif type(value) is str and _RATIONAL.fullmatch(value):
            row.append(Fraction(value))
        else:
            raise ValueError("constraint coefficients must be bounded exact rationals")
    return row


def _declaration(raw, policy):
    try:
        data = json.loads(raw.decode("utf8"))
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid numeric source declaration") from exc
    if (type(data) is not list or len(data) != 5 or data[0] != "lsr-linear-v1"
            or type(data[1]) is not list or len(data[1]) != 5
            or any(type(item) is not str or not item for item in data[1])):
        raise ValueError("invalid numeric source declaration")
    if data[1][0] != policy.scope[0]:
        return None
    if tuple(data[1]) != policy.scope:
        return "unreconciled_scope", None, None
    kind, left, right = data[2:]
    if kind == "measurement":
        p = _rational_row(left, policy.middle_width)
        if not any(p):
            raise ValueError("measurement must observe an intermediate coordinate")
        return kind, p, _rational_row(right, len(policy.input_axes))
    if kind == "consequence" and type(left) is int and left in policy.output_axes:
        return kind, left, _rational_row(right, policy.middle_width)
    if kind == "context" and left is None and right is None:
        return kind, left, right
    raise ValueError("unknown numeric declaration in constraint scope")


def _subtract(row, coefficient, basis):
    return [x - coefficient * y for x, y in zip(row, basis, strict=True)]


def _measurement_basis(measurements, width):
    """Reduce [P | K | source coefficients] without rounding or losing witnesses."""
    basis = {}
    for ordinal, (left, right, _) in enumerate(measurements):
        row = [*left, *right, *(Fraction(i == ordinal) for i in range(len(measurements)))]
        for pivot, direction in sorted(basis.items()):
            row = _subtract(row, row[pivot], direction)
        column = next((i for i, value in enumerate(row[:width]) if value), None)
        if column is None:
            if any(row[width:width + len(right)]):
                return (), _contributing_measurements(row[width + len(right):], measurements)
        else:
            basis[column] = [value / row[column] for value in row]
    return tuple(sorted(basis.items())), ()


def _contributing_measurements(coefficients, measurements):
    return tuple(row[2] for row, coefficient in zip(measurements, coefficients, strict=True) if coefficient)


def _reduce_law(law, basis, input_width, measurements):
    residual = [*law, *([Fraction(0)] * (input_width + len(measurements)))]
    for pivot, row in basis:
        residual = _subtract(residual, residual[pivot], row)
    width = len(law)
    weights = tuple(-value for value in residual[width:width + input_width])
    witnesses = _contributing_measurements(residual[width + input_width:], measurements)
    return weights, tuple(residual[:width]), witnesses


def compile_source_constraints(pin, policy, kernel_sha256, records, publication_sha256):
    """Validate a complete registered authority snapshot and compile its observable rows."""
    policy.require_frame(pin)
    versions = constraint_versions(pin, policy)
    active = tuple((source, digest, generation) for source, digest, generation in versions if digest is not None)
    if tuple(source for source, _ in records) != tuple(source for source, _, _ in active):
        raise ValueError("constraint records must cover every current registered source")

    measurements, laws = [], defaultdict(list)
    contexts = set()
    unreconciled = []
    for (source, raw), (_, digest, generation) in zip(records, active, strict=True):
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_CONSTRAINT_SOURCE_BYTES:
            raise ValueError("numeric declaration exceeds its byte allowance")
        if sha256(raw).hexdigest() != digest:
            raise ValueError("numeric declaration disagrees with its pinned source bytes")
        declaration = _declaration(raw, policy)
        if declaration is None:
            continue
        kind, left, right = declaration
        span = SourceSpan(source, digest, 0, len(raw), digest)
        witness = WorldWitness(span, generation, 0)
        if kind == "unreconciled_scope":
            if source in policy.authorities:
                unreconciled.append(witness)
            continue
        if (kind == "context") != (source in policy.contexts):
            raise ValueError("numeric declaration disagrees with its registered source role")
        if kind == "context":
            mentions = pin.memory._mentions.get(source, ())
            if len(mentions) == 1 and mentions[0][3] == 1 and pin.memory.source_mention_span(source, 0) == span:
                contexts.add(source)
            continue
        if kind == "measurement":
            measurements.append((left, right, witness))
        else:
            laws[left].append((right, witness))

    basis, conflicts = _measurement_basis(measurements, policy.middle_width)
    relations, evidence, exact_axes = [], [], set()
    if not conflicts and not unreconciled:
        for axis in policy.output_axes:
            candidates = laws[axis]
            if len(candidates) != 1:
                continue
            direction, law_witness = candidates[0]
            weights, remainder, measurement_witnesses = _reduce_law(
                direction, basis, len(policy.input_axes), measurements,
            )
            relations.append((axis, weights, remainder))
            evidence.append((axis, (law_witness, *measurement_witnesses)))
            if not any(remainder):
                exact_axes.add(axis)

    open_axes = tuple(axis for axis in policy.output_axes if axis not in exact_axes)
    use = ConstraintUse(publication_sha256, tuple(evidence), open_axes, conflicts, tuple(unreconciled))
    return SourceConstraintPublication(policy, kernel_sha256, versions, tuple(relations), frozenset(contexts), use)
