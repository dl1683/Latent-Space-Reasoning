"""Persistent ordered source operators for a late-selected latent continuation."""

from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import _frozen, _real_array


_PATH_REDUCTIONS = {"max_plus": np.maximum, "log_sum_exp": np.logaddexp}


@dataclass(frozen=True, eq=False)
class _Product:
    operator: np.ndarray | None
    left: "_Product | None" = None
    right: "_Product | None" = None


def _multiply(left, right):
    with np.errstate(over="raise", invalid="raise"):
        result = left @ right
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite transition product")
    return result


def _path_product(left, right, *, reduce):
    """Compose or apply log-weighted transfers through their shared coordinates."""
    shape = (left.shape[0], *right.shape[1:])
    column_shape = (left.shape[0],) + (1,) * (right.ndim - 1)
    result = np.full(shape, -np.inf)
    paths = np.empty(shape)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        for middle in range(left.shape[1]):
            np.add(left[:, middle].reshape(column_shape), right[middle], out=paths)
            # The internal logaddexp gap may overflow even when the sum is finite.
            with np.errstate(over="ignore"):
                reduce(result, paths, out=result)
    if np.isposinf(result).any():
        raise FloatingPointError("positive infinity in path product")
    return result


def _product_for(algebra):
    if algebra == "linear":
        return _multiply
    return partial(_path_product, reduce=_PATH_REDUCTIONS[algebra])


def _coordinates(value, algebra):
    if algebra == "linear":
        return _real_array(value)
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError("expected real path scores")
    with np.errstate(over="raise", under="ignore"):
        array = array.astype(np.float64, copy=False)
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError("path scores must be finite or negative infinity")
    return array


def _join(left, right, multiply):
    product = None
    if left.operator is not None and right.operator is not None:
        product = _frozen(multiply(right.operator, left.operator))
    return _Product(product, left, right)


def _build(sources, first, stop, multiply):
    if stop - first == 1:
        return _Product(sources.get(first))
    split = (first + stop) // 2
    return _join(_build(sources, first, split, multiply), _build(sources, split, stop, multiply), multiply)


def _replace(node, first, stop, slots, operators, lo, hi, multiply):
    if lo == hi:
        return node
    if stop - first == 1:
        revised = operators[lo]
        if revised is node.operator or np.array_equal(revised, node.operator):
            return node
        return _Product(revised)
    split = (first + stop) // 2
    middle = bisect_left(slots, split, lo, hi)
    left = _replace(node.left, first, split, slots, operators, lo, middle, multiply)
    right = _replace(node.right, split, stop, slots, operators, middle, hi, multiply)
    if left is node.left and right is node.right:
        return node
    return _join(left, right, multiply)


def _cover(node, first, stop, start, end):
    if start <= first and stop <= end:
        yield node.operator
        return
    split = (first + stop) // 2
    if start < split:
        yield from _cover(node.left, first, split, start, end)
    if split < end:
        yield from _cover(node.right, split, stop, start, end)


def _source_operators(slots, width, changes, algebra):
    operators = {}
    for source, supplied in changes.items():
        if source not in slots:
            raise ValueError("unknown transition source")
        value = None if supplied is None else _coordinates(supplied, algebra)
        if value is not None and value.shape != (width, width):
            raise ValueError("source operators must be square with the declared width")
        operators[slots[source]] = None if value is None else _frozen(value)
    return operators


@dataclass(frozen=True, eq=False)
class TransitionMemory:
    """An immutable product tree over a fixed, explicitly ordered source layout."""

    source_ids: tuple[str, ...]
    width: int
    _slots: Mapping[str, int]
    _root: _Product
    algebra: str = "linear"

    @classmethod
    def prepare(cls, source_ids, width, sources, *, algebra="linear"):
        """Compile a bank before any particular seed or query range is chosen."""
        source_ids, width = tuple(source_ids), index(width)
        if (width < 1 or not source_ids or any(type(source) is not str or not source for source in source_ids)
                or len(set(source_ids)) != len(source_ids)):
            raise ValueError("transition memory needs unique nonempty source IDs and a positive width")
        if type(algebra) is not str or (algebra != "linear" and algebra not in _PATH_REDUCTIONS):
            raise ValueError("unknown transition algebra")
        slots = {source: position for position, source in enumerate(source_ids)}
        operators = _source_operators(slots, width, sources, algebra)
        root = _build(operators, 0, len(slots), _product_for(algebra))
        return cls(source_ids, width, MappingProxyType(slots), root, algebra)

    def revise(self, changes):
        """Replace named source operators, using None to remove a prerequisite."""
        updates = _source_operators(self._slots, self.width, changes, self.algebra)
        if not updates:
            return self
        slots, operators = zip(*sorted(updates.items()))
        root = _replace(self._root, 0, len(self.source_ids), slots, operators, 0, len(slots),
                        _product_for(self.algebra))
        return self if root is self._root else replace(self, _root=root)


def continue_latent(seed, *, memory=None, start=0, stop=None):
    """Continue a column vector through an ordered interval, or pass it through without memory."""
    if memory is not None and type(memory) is not TransitionMemory:
        raise ValueError("expected transition memory")
    algebra = "linear" if memory is None else memory.algebra
    seed = _coordinates(seed, algebra)
    if seed.ndim != 1 or not seed.size:
        raise ValueError("latent seed must be a nonempty vector")
    if memory is None:
        if index(start) != 0 or stop is not None:
            raise ValueError("an explicit transition range requires memory")
        return _frozen(seed)
    if seed.shape != (memory.width,):
        raise ValueError("latent seed must match the transition memory width")
    size = len(memory.source_ids)
    start, stop = index(start), size if stop is None else index(stop)
    if not 0 <= start <= stop <= size:
        raise ValueError("transition range is outside the source layout")
    if start == stop:
        return _frozen(seed)
    operators = tuple(_cover(memory._root, 0, size, start, stop))
    if any(operator is None for operator in operators):
        return None
    multiply = _product_for(algebra)
    for operator in operators:
        seed = multiply(operator, seed)
    return _frozen(seed)
