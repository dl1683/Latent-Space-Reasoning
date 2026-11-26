"""Retain shared latent dependencies through linear maps and coordinate nonlinearities."""

from dataclasses import dataclass
from fractions import Fraction
from math import inf, nextafter, tanh

import numpy as np

_EPSILON = Fraction(1, 2**52)
_SUBNORMAL = Fraction(1, 2**1074)
_FLOAT_MAX = Fraction(float(np.finfo(float).max))


@dataclass(frozen=True, eq=False)
class _Atom:
    center: Fraction
    # None denotes an unconstrained real nuisance variable.
    bounds: tuple[Fraction, Fraction] | None


@dataclass(frozen=True)
class Expression:
    constant: Fraction = Fraction()
    terms: tuple[tuple[_Atom, Fraction], ...] = ()

    def center(self):
        return self.constant + sum(weight * atom.center for atom, weight in self.terms)

    def bounds(self):
        lower = upper = self.constant
        for atom, weight in self.terms:
            if atom.bounds is None:
                return None
            a, b = (endpoint * weight for endpoint in atom.bounds)
            lower += min(a, b)
            upper += max(a, b)
        return lower, upper

    def score_bounds(self):
        bounds = self.bounds()
        return (-inf, inf) if bounds is None else bounds

    @classmethod
    def unknown(cls):
        return cls(terms=((_Atom(Fraction(), None), Fraction(1)),))


def point(row):
    return tuple(Expression(Fraction(float(value))) for value in row)


def combine(rows, weights, constant=Fraction()):
    terms = {}
    for expression, weight in zip(rows, weights, strict=True):
        if not weight:
            continue
        weight = weight if type(weight) is Fraction else Fraction(float(weight))
        constant += expression.constant * weight
        for atom, coefficient in expression.terms:
            terms[atom] = terms.get(atom, Fraction()) + coefficient * weight
    # Identity is only an in-process ordering rule. Exact sums do not depend on it.
    return Expression(constant, tuple(sorted(((atom, value) for atom, value in terms.items() if value),
                                             key=lambda term: id(term[0]))))


def _rounded(expression, scale, operations):
    if scale is None or scale > _FLOAT_MAX / 2:
        return Expression.unknown()
    radius = 8 * operations * _EPSILON * scale + operations * _SUBNORMAL
    error = _Atom(Fraction(), (-radius, radius))
    return combine((expression, Expression(terms=((error, Fraction(1)),))), (1, 1))


def _round_linear(rows, weights, representative):
    row = tuple((expression, Fraction(float(weight)))
                for expression, weight in zip(rows, weights, strict=True) if weight)
    if not any(expression.terms for expression, _ in row):
        return Expression(Fraction(float(representative)))
    scale = Fraction()
    for expression, weight in row:
        bounds = expression.bounds()
        if bounds is None:
            return Expression.unknown()
        scale += abs(weight) * max(map(abs, bounds))
    return _rounded(combine(rows, weights), scale, len(rows))


def linear(rows, weights):
    """Apply the receiver's Float64 linear map with uncertainty and roundoff."""
    values = representatives(rows) @ weights
    return tuple(_round_linear(rows, column, value) for column, value in zip(weights.T, values, strict=True))


def dot(rows, weights):
    return _round_linear(rows, weights, representatives(rows) @ weights)


def scores(query, keys):
    values = keys @ representatives(query)
    return (_round_linear(query, key, value).score_bounds()
            for key, value in zip(keys, values, strict=True))


def add(left, right):
    # This also permits injecting a source expression into an empty coordinate.
    if not left.terms and not left.constant:
        return right
    if not right.terms and not right.constant:
        return left
    if not left.terms and not right.terms:
        return Expression(Fraction(float(left.constant) + float(right.constant)))
    a, b = left.bounds(), right.bounds()
    scale = None if a is None or b is None else max(map(abs, a)) + max(map(abs, b))
    return _rounded(combine((left, right), (1, 1)), scale, 1)


def scale(expression, weight):
    if weight == 0:
        return Expression()
    if weight == 1:
        return expression
    if weight == -1:
        return combine((expression,), (-1,))
    if not expression.terms:
        return Expression(Fraction(float(expression.constant) * float(weight)))
    bounds = expression.bounds()
    size = None if bounds is None else max(map(abs, bounds)) * abs(Fraction(float(weight)))
    return _rounded(combine((expression,), (weight,)), size, 1)


def _tanh(value):
    return tanh(float(min(Fraction(40), max(Fraction(-40), value))))


def _tanh_bounds(expression):
    bounds = expression.bounds()
    if bounds is None:
        return Fraction(-1), Fraction(1)
    lower, upper = map(_tanh, bounds)
    # Pad conversion and libm at the enclosing float boundary.
    for _ in range(4):
        lower, upper = nextafter(lower, -inf), nextafter(upper, inf)
    return Fraction(max(-1., lower)), Fraction(min(1., upper))


def activate(rows):
    """Share the same tanh atom for opposite preactivations."""
    atoms = {}
    activated = []
    for row in rows:
        if not row.terms:
            activated.append(Expression(Fraction(float(np.tanh(float(row.constant))))))
            continue
        leading = row.constant or row.terms[0][1]
        sign = 1 if leading > 0 else -1
        canonical = row if sign == 1 else combine((row,), (-1,))
        atom = atoms.get(canonical)
        if atom is None:
            atom = _Atom(Fraction(_tanh(canonical.center())), _tanh_bounds(canonical))
            atoms[canonical] = atom
        activated.append(Expression(terms=((atom, Fraction(sign)),)))
    return tuple(activated)


def hinge(expression, threshold):
    if threshold is None:
        return expression
    shifted = add(expression, Expression(-Fraction(float(threshold))))
    bounds = shifted.bounds()
    if bounds is None:
        return Expression.unknown()
    lo, hi = bounds
    if lo >= 0:
        return shifted
    if hi <= 0:
        return Expression()
    return Expression(terms=((_Atom(max(Fraction(), shifted.center()), (Fraction(), hi)), Fraction(1)),))


def representatives(rows):
    return np.asarray([float(row.center()) for row in rows], dtype=np.float64)
