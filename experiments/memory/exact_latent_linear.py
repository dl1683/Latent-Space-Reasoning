"""Exact arithmetic for small Gram systems over quantized latent rows."""

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
