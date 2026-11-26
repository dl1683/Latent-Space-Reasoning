"""Compile exact, scoped linear consequences before a task enters the latent world."""

from collections import defaultdict
from dataclasses import dataclass, field, replace
from fractions import Fraction
from hashlib import sha256
import json
import re
from types import MappingProxyType

from experiments.memory.relational_feedback import Expression, _FLOAT_MAX, _rounded, combine
from experiments.memory.unit_program import MAX_REGISTRY_BYTES, UnitConversion, UnitProgram, compile_unit_scopes, compile_units
from experiments.memory.verified_source_tool import SourceSpan
from experiments.memory.world_transport import WorldWitness, _source_ids


MAX_CONSTRAINT_SOURCE_BYTES = 4096
UNIT_ACTION_GRAMMAR = "multiplicative-units-v1"
UNIT_CATALOG_GRAMMAR = "named-unit-factors-v1"
UNIT_SCOPE_GRAMMAR = "scoped-unit-factors-v1"
_UNIT_CATALOG_TERMS = 65_536
_MAX_AXES = 16
_MAX_SOURCES = 32
_RATIONAL = re.compile(r"-?[0-9]{1,7}(/[1-9][0-9]{0,6}|\.[0-9]{1,7})?")
_SURROGATE = re.compile(r"[\ud800-\udfff]")
_NORMAL_MIN = Fraction(1, 2**1022)


@dataclass(frozen=True)
class SourceConstraintPolicy:
    """Bound one declared set of input coordinates and downstream channels."""

    scope: tuple[str, str, str, str, str]
    input_axes: tuple[int, ...]
    output_axes: tuple[int, ...]
    middle_width: int
    authorities: tuple[str, ...]
    contexts: tuple[str, ...]
    unit_registry: str | None = None
    unit_catalog: bool = False
    scoped_units: bool = False

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
        if self.unit_registry is not None and (
            type(self.unit_registry) is not str or self.unit_registry not in self.authorities
            or len(self.input_axes) != 1 or len(self.output_axes) != 1 or self.middle_width != 1
            or not self.scope[4].isidentifier()
        ):
            raise ValueError("a unit registry requires scalar channels and one named target unit")
        if (type(self.unit_catalog) is not bool or type(self.scoped_units) is not bool
                or self.scoped_units and not self.unit_catalog
                or self.unit_catalog and (self.unit_registry is None or not self.scoped_units and len(self.contexts) != 1)):
            raise ValueError("unit catalogs require registered conversion contexts")

    def wire(self):
        linear = (self.scope, self.input_axes, self.output_axes, self.middle_width, self.authorities, self.contexts)
        grammar = UNIT_CATALOG_GRAMMAR if self.unit_catalog else UNIT_ACTION_GRAMMAR
        if self.scoped_units:
            grammar = UNIT_SCOPE_GRAMMAR
        return linear if self.unit_registry is None else (*linear, self.unit_registry, grammar)

    def byte_limit(self, source):
        return (MAX_REGISTRY_BYTES if self.unit_registry is not None and source in self.authorities
                else MAX_CONSTRAINT_SOURCE_BYTES)

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
    withheld_conversion: tuple[str, tuple[WorldWitness, ...]] | None = None
    source_units: tuple[str, ...] = ()
    unit_scope: str | None = None


@dataclass(frozen=True, eq=False)
class ConstraintAction:
    # Input coefficients and shared nuisance coefficients, in exact arithmetic.
    relations: tuple[tuple[int, tuple[Fraction, ...], tuple[Fraction, ...]], ...]
    use: ConstraintUse

    def project(self, policy, rows):
        inputs = tuple(rows[axis] for axis in policy.input_axes)
        nuisance = tuple(Expression.unknown() for _ in range(policy.middle_width))
        additions = {axis: combine((*inputs, *nuisance), (*weights, *remainder))
                     for axis, weights, remainder in self.relations}
        use = self.use
        if policy.unit_registry is not None and additions:
            output, = policy.output_axes
            exact = additions[output].constant
            if exact and not _NORMAL_MIN <= abs(exact) <= _FLOAT_MAX / 2:
                _, witnesses = use.evidence[0]
                use = replace(use, evidence=(), open_axes=(output,), withheld_conversion=("numeric_range", witnesses))
                additions.clear()
            elif exact:
                additions[output] = _rounded(additions[output], abs(exact), 1)
        projected = tuple(additions[axis] if axis in additions else
                          Expression.unknown() if axis in policy.output_axes else Expression()
                          for axis in range(len(rows)))
        return projected, use


@dataclass(frozen=True, eq=False)
class ScopedUnitCatalog:
    program: UnitProgram | None
    context: WorldWitness
    group: str
    headers: tuple[SourceSpan, ...]


@dataclass(frozen=True, eq=False)
class SourceConstraintPublication:
    policy: SourceConstraintPolicy
    kernel_sha256: str
    source_versions: tuple[tuple[str, str | None, int | None], ...]
    actions: MappingProxyType
    catalog: UnitProgram | None = None
    catalog_context: WorldWitness | None = None
    publication_sha256: str = ""
    scoped_catalogs: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    def active_on(self, pin):
        return self.source_versions == constraint_versions(pin, self.policy)

    def unit_actions(self, name):
        generations = {source: generation for source, _, generation in self.source_versions}
        target, output = self.policy.scope[4], self.policy.output_axes[0]
        if self.catalog is not None:
            result = _catalog_conversion(self.catalog, name, target)
            return {self.catalog_context.span.source_id:
                    _conversion_action(result, output, generations, self.publication_sha256)}
        actions = {}
        for source, catalog in self.scoped_catalogs.items():
            result = (UnitConversion("unit_scope") if catalog.program is None
                      else _catalog_conversion(catalog.program, name, target))
            spans = tuple(dict.fromkeys((catalog.context.span, *catalog.headers, *result.spans)))
            actions[source] = _conversion_action(replace(result, spans=spans), output, generations,
                                                 self.publication_sha256, unit_scope=catalog.group)
        return actions

    def context_witness(self, source):
        if self.catalog is not None:
            return self.catalog_context
        selected = self.scoped_catalogs.get(source)
        return None if selected is None else selected.context


def _catalog_conversion(catalog, name, target):
    result = catalog.convert(name, target)
    if name not in catalog:
        return replace(result, status="outside_grammar")
    if target not in catalog:
        return replace(result, status="target_unit")
    return result


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
    if policy.unit_registry is not None:
        if policy.unit_catalog:
            valid_group = (type(left) is str and left.isidentifier() if policy.scoped_units else left is None)
            if kind == "catalog" and valid_group and right is None:
                return kind, left, right
            raise ValueError("invalid catalog context declaration")
        if kind == "conversion" and all(type(unit) is str and unit.isidentifier() for unit in (left, right)):
            return kind, left, right
        raise ValueError("numeric context must declare a conversion between registry names")
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


def _context_matches(pin, source, span):
    mentions = pin.memory._mentions.get(source, ())
    return len(mentions) == 1 and mentions[0][3] == 1 and pin.memory.source_mention_span(source, 0) == span


def _linear_actions(pin, policy, records, publication_sha256):
    measurements, laws = [], defaultdict(list)
    contexts = set()
    unreconciled = []
    for source, raw, witness in records:
        declaration = _declaration(raw, policy)
        if declaration is None:
            continue
        kind, left, right = declaration
        if kind == "unreconciled_scope":
            if source in policy.authorities:
                unreconciled.append(witness)
            continue
        if (kind == "context") != (source in policy.contexts):
            raise ValueError("numeric declaration disagrees with its registered source role")
        if kind == "context":
            if _context_matches(pin, source, witness.span):
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
    operation = ConstraintAction(tuple(relations), use)
    return {source: operation for source in sorted(contexts)}


def _conversion_action(result, output, generations, publication_sha256, *, unit_scope=None):
    evidence = tuple(WorldWitness(span, generations[span.source_id], 0) for span in result.spans)
    if result.coefficient is None:
        use = ConstraintUse(publication_sha256, (), (output,), withheld_conversion=(result.status, evidence),
                            unit_scope=unit_scope)
        return ConstraintAction((), use)
    use = ConstraintUse(publication_sha256, ((output, evidence),), (), unit_scope=unit_scope)
    return ConstraintAction(((output, (result.coefficient,), (Fraction(),)),), use)


def _unit_actions(pin, policy, records, publication_sha256):
    authorities = {source: raw for source, raw, _ in records if source in policy.authorities}
    if set(authorities) != set(policy.authorities):
        return {}, None, None
    program = compile_units(policy.unit_registry, authorities,
                            max_terms=_UNIT_CATALOG_TERMS if policy.unit_catalog else None)
    output, = policy.output_axes
    actions = {}
    for source, raw, witness in records:
        if source not in policy.contexts or not _context_matches(pin, source, witness.span):
            continue
        declaration = _declaration(raw, policy)
        if declaration is not None and declaration[0] == "catalog":
            return {}, program, witness
        if declaration is None or declaration[0] != "conversion":
            continue
        if declaration[2] != policy.scope[4]:
            use = ConstraintUse(publication_sha256, (), (output,), withheld_conversion=("target_unit", (witness,)))
            actions[source] = ConstraintAction((), use)
            continue
        result = program.convert(*declaration[1:])
        actions[source] = _conversion_action(result, output, pin._source_revisions, publication_sha256)
    return actions, None, None


def _scoped_unit_catalogs(pin, policy, records):
    authorities = {source: raw for source, raw, _ in records if source in policy.authorities}
    if set(authorities) != set(policy.authorities):
        return {}
    contexts = {}
    for source, raw, witness in records:
        if source in policy.contexts and _context_matches(pin, source, witness.span):
            declaration = _declaration(raw, policy)
            if declaration is not None and declaration[0] == "catalog":
                contexts[source] = declaration[1], witness
    views = compile_unit_scopes(policy.unit_registry, authorities, (group for group, _ in contexts.values()),
                               max_terms=_UNIT_CATALOG_TERMS)
    return {source: ScopedUnitCatalog(views[group][0], witness, group, views[group][1])
            for source, (group, witness) in contexts.items()}


def bind_constraint_publication(publication, digest):
    bound = {action: replace(action, use=replace(action.use, publication_sha256=digest))
             for action in dict.fromkeys(publication.actions.values())}
    actions = MappingProxyType({source: bound[action] for source, action in publication.actions.items()})
    return replace(publication, actions=actions, publication_sha256=digest)


def unit_action_digest(publication):
    def spans(witnesses):
        return [(w.span.source_id, w.span.version_sha256, w.span.start_byte, w.span.end_byte,
                 w.span.span_sha256, w.generation) for w in witnesses]

    rows = []
    if publication.policy.unit_catalog and publication.catalog is not None:
        rows.append(("catalog", publication.catalog.fingerprint(), spans((publication.catalog_context,))))
    for source, catalog in sorted(publication.scoped_catalogs.items()):
        fingerprint = None if catalog.program is None else catalog.program.fingerprint()
        headers = [(span.source_id, span.version_sha256, span.start_byte, span.end_byte, span.span_sha256)
                   for span in catalog.headers]
        rows.append(("scope", source, catalog.group, fingerprint, spans((catalog.context,)), headers))
    for source, action in sorted(publication.actions.items()):
        if action.use.withheld_conversion is None:
            axis, (coefficient,), _ = action.relations[0]
            rows.append((source, "resolved", axis, str(coefficient), spans(action.use.evidence[0][1])))
        else:
            reason, evidence = action.use.withheld_conversion
            rows.append((source, reason, spans(evidence)))
    return sha256(json.dumps(rows, separators=(",", ":")).encode("ascii")).hexdigest()


def compile_source_constraints(pin, policy, kernel_sha256, records, publication_sha256):
    """Validate a complete registered authority snapshot and compile context actions."""
    policy.require_frame(pin)
    versions = constraint_versions(pin, policy)
    active = tuple((source, digest, generation) for source, digest, generation in versions if digest is not None)
    if tuple(source for source, _ in records) != tuple(source for source, _, _ in active):
        raise ValueError("constraint records must cover every current registered source")
    verified = []
    for (source, raw), (_, digest, generation) in zip(records, active, strict=True):
        if type(raw) is not bytes or not 0 < len(raw) <= policy.byte_limit(source):
            raise ValueError("numeric declaration exceeds its byte allowance")
        if sha256(raw).hexdigest() != digest:
            raise ValueError("numeric declaration disagrees with its pinned source bytes")
        span = SourceSpan(source, digest, 0, len(raw), digest)
        verified.append((source, raw, WorldWitness(span, generation, 0)))
    catalog = context = None
    scoped = {}
    if policy.unit_registry is None:
        actions = _linear_actions(pin, policy, verified, publication_sha256)
    elif policy.scoped_units:
        actions = {}
        scoped = _scoped_unit_catalogs(pin, policy, verified)
    else:
        actions, catalog, context = _unit_actions(pin, policy, verified, publication_sha256)
    return SourceConstraintPublication(policy, kernel_sha256, versions, MappingProxyType(actions),
                                       catalog, context, publication_sha256, MappingProxyType(scoped))
