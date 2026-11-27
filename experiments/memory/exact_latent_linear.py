"""Exact arithmetic for small Gram systems over quantized latent rows."""

from dataclasses import dataclass, field
from fractions import Fraction


def dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def gram(vectors):
    return [[dot(vector, other) for other in vectors] for vector in vectors]


def solve_gram(matrix, rhs):
    """Solve an invertible Gram system over rationals."""
    system = [[*map(Fraction, row), Fraction(target)] for row, target in zip(matrix, rhs, strict=True)]
    for column in range(len(system)):
        pivot = next((index for index in range(column, len(system)) if system[index][column]), None)
        if pivot is None:
            raise ValueError("expected an independent Gram basis")
        system[column], system[pivot] = system[pivot], system[column]
        divisor = system[column][column]
        system[column] = [value / divisor for value in system[column]]
        for index, row in enumerate(system):
            if index != column and row[column]:
                factor = row[column]
                system[index] = [value - factor * ref for value, ref in zip(row, system[column], strict=True)]
    return [row[-1] for row in system]


def _exact(value):
    if type(value) not in (int, Fraction):
        raise TypeError("linear evidence requires exact rational coordinates and values")
    return Fraction(value)


@dataclass(frozen=True, slots=True)
class LinearConsequenceBound:
    center: Fraction
    radius_squared: Fraction
    residual_squared: Fraction

    def strict_side(self, threshold):
        """Classify only if the entire closed interval is on one strict side."""
        margin = self.center - _exact(threshold)
        if margin ** 2 <= self.radius_squared:
            return "unresolved"
        return "above" if margin > 0 else "below"


@dataclass(frozen=True, slots=True)
class _OperatorFunctional:
    operand: tuple[Fraction, ...]
    output_form: tuple[Fraction, ...]

    def inner(self, other):
        return dot(self.operand, other.operand) * dot(self.output_form, other.output_form)


@dataclass(frozen=True, slots=True)
class LinearOperatorEvidence:
    """Immutable exact observations of a single, norm-bounded linear operator."""

    input_width: int
    output_width: int
    frobenius_cap_squared: int | Fraction
    _basis: tuple[_OperatorFunctional, ...] = field(default=(), init=False)
    _inverse: tuple[tuple[Fraction, ...], ...] = field(default=(), init=False)
    _coefficients: tuple[Fraction, ...] = field(default=(), init=False)
    minimum_norm_squared: Fraction = field(default=Fraction(0), init=False)

    def __post_init__(self):
        if any(type(width) is not int or width < 1 for width in (self.input_width, self.output_width)):
            raise ValueError("linear evidence requires positive input and output widths")
        if _exact(self.frobenius_cap_squared) < 0:
            raise ValueError("linear evidence requires a nonnegative squared norm cap")

    @property
    def rank(self):
        return len(self._basis)

    def _project(self, operand, output_form):
        probe = _OperatorFunctional(tuple(map(_exact, operand)), tuple(map(_exact, output_form)))
        if len(probe.operand) != self.input_width or len(probe.output_form) != self.output_width:
            raise ValueError("linear evidence functional has the wrong width")
        cross = tuple(probe.inner(item) for item in self._basis)
        weights = tuple(dot(row, cross) for row in self._inverse)
        residual = probe.inner(probe) - dot(cross, weights)
        return probe, cross, weights, residual

    def observe(self, operand, output_form, value):
        """Return a successor constrained by the exact value of output_form @ L @ operand."""
        probe, cross, weights, residual = self._project(operand, output_form)
        innovation = _exact(value) - dot(self._coefficients, cross)
        if residual == 0:
            if innovation != 0:
                raise ValueError("dependent linear measurement contradicts retained evidence")
            return self
        adjustment = innovation / residual
        updated_norm = self.minimum_norm_squared + innovation * adjustment
        if updated_norm > self.frobenius_cap_squared:
            raise ValueError("new linear measurement exceeds the declared norm cap")
        # Schur update for one new independent probe.
        inverse = tuple(
            tuple(entry + wi * wj / residual for entry, wj in zip(row, weights, strict=True)) + (-wi / residual,)
            for row, wi in zip(self._inverse, weights, strict=True)
        )
        inverse += (tuple(-weight / residual for weight in weights) + (1 / residual,),)
        coefficients = tuple(coefficient - adjustment * weight for coefficient, weight in
                             zip(self._coefficients, weights, strict=True)) + (adjustment,)
        successor = LinearOperatorEvidence(self.input_width, self.output_width, self.frobenius_cap_squared)
        object.__setattr__(successor, "_basis", (*self._basis, probe))
        object.__setattr__(successor, "_inverse", inverse)
        object.__setattr__(successor, "_coefficients", coefficients)
        object.__setattr__(successor, "minimum_norm_squared", updated_norm)
        return successor

    def bound(self, operand, output_form):
        """Return the sharp interval in center and squared-radius form."""
        headroom = self.frobenius_cap_squared - self.minimum_norm_squared
        _, cross, _, residual = self._project(operand, output_form)
        return LinearConsequenceBound(dot(self._coefficients, cross), headroom * residual, residual)

    def residual_reduction(self, operand, output_form, *, candidate_operand, candidate_output_form):
        """Score a possible measurement before release, regardless of its value."""
        target, cross, _, _ = self._project(operand, output_form)
        candidate, _, weights, residual = self._project(candidate_operand, candidate_output_form)
        if residual == 0:
            return Fraction(0)
        aligned = target.inner(candidate) - dot(cross, weights)
        return aligned ** 2 / residual
