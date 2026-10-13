"""Exact local composition in the existing tree quasiseparable (TQS) class.

See research/TREE_COMPOSITION_FRONTIER_V1.md for formulas, proof and costs.
Raw state widths add; this module performs no rank reduction or discovery.
Object arrays of Python integers/Fractions support exact arithmetic. Integer
arrays are promoted to object to avoid fixed-width overflow. Floating arithmetic
has no general stability guarantee. Composition never materializes a dense matrix.
"""

from types import MappingProxyType
from numbers import Number, Rational

import numpy as np


class TreeOperator:
    """Square physical blocks on nodes 0,...,K-1; directed states may differ.

    P[i,j] maps node i input to state i->j. Q[i,j] maps state j->i
    to node i output. T[i,j,k] maps state k->i to state i->j (j != k).
    Arrays are copied on construction. Graph metadata uses O(K+E) space.
    """

    def __init__(self, adjacency, D, P, Q, T):
        adjacency = tuple(tuple(row) for row in adjacency)
        count = len(adjacency)
        if not count:
            raise ValueError("A nonempty tree is required")
        for i, row in enumerate(adjacency):
            if len(set(row)) != len(row) or i in row:
                raise ValueError("Duplicate neighbors and self edges are forbidden")
            for j in row:
                if not isinstance(j, (int, np.integer)) or not 0 <= j < count:
                    raise ValueError("Neighbor labels must be node indices")
                if i not in adjacency[j]:
                    raise ValueError("Adjacency must be symmetric")
        adjacency = tuple(tuple(sorted(row)) for row in adjacency)
        if sum(map(len, adjacency)) != 2 * (count - 1):
            raise ValueError("The graph must be a tree")
        parent = {0: None}
        order = [0]
        for i in order:
            for j in adjacency[i]:
                if j == parent[i]:
                    continue
                if j in parent:
                    raise ValueError("The graph contains a cycle")
                parent[j] = i
                order.append(j)
        if len(order) != count:
            raise ValueError("The graph is disconnected")

        edges = {(i, j) for i, row in enumerate(adjacency) for j in row}
        transfers = {(i, j, k) for i, row in enumerate(adjacency)
                     for j in row for k in row if j != k}
        if set(D) != set(range(count)) or set(P) != edges or set(Q) != edges:
            raise ValueError("D/P/Q generator keys do not match the tree")
        if set(T) != transfers:
            raise ValueError("Transfer keys must contain precisely the nonreversing pairs")
        raw = [np.asarray(v) for group in (D, P, Q, T) for v in group.values()]
        if any(v.ndim != 2 or v.dtype.kind not in "biufcO" for v in raw):
            raise ValueError("Generators must be numeric two-dimensional arrays")
        for arr in raw:
            if arr.dtype.kind == "O":
                for value in arr.flat:
                    if not isinstance(value, Number):
                        raise ValueError("Object generators must contain numbers")
                    if not isinstance(value, Rational) and not np.isfinite(value):
                        raise ValueError("Generators must be finite")
            elif not np.isfinite(arr).all():
                raise ValueError("Generators must be finite")
        # Determine integer-only input before NumPy mixes signed and unsigned
        # dtypes: int64 + uint64 otherwise promotes to lossy float64.
        dtype = (np.dtype(object) if all(v.dtype.kind in "biu" for v in raw)
                 else np.result_type(*(v.dtype for v in raw)))

        def copy_group(group):
            result = {}
            for key, value in group.items():
                arr = np.array(value, dtype=dtype, copy=True)
                arr.setflags(write=False)
                result[key] = arr
            return MappingProxyType(result)

        self.D, self.P, self.Q, self.T = map(copy_group, (D, P, Q, T))
        sizes = tuple(self.D[i].shape[0] for i in range(count))
        for i in range(count):
            if self.D[i].shape != (sizes[i], sizes[i]):
                raise ValueError("Physical diagonal blocks must be square")
            for j in adjacency[i]:
                width = self.P[i, j].shape[0]
                if self.P[i, j].shape != (width, sizes[i]):
                    raise ValueError("Input generator shape mismatch")
                if self.Q[j, i].shape != (sizes[j], width):
                    raise ValueError("Output generator shape mismatch")
                for k in adjacency[i]:
                    if k != j and self.T[i, j, k].shape != (width, self.P[k, i].shape[0]):
                        raise ValueError("Transfer generator shape mismatch")
        offsets = np.cumsum((0,) + sizes)
        self.slices = tuple(slice(int(offsets[i]), int(offsets[i + 1])) for i in range(count))
        self.adjacency = adjacency
        self.sizes = sizes
        self.shape = (sum(sizes), sum(sizes))
        self.dtype = dtype
        self.schedule = tuple((i, parent[i]) for i in reversed(order[1:])) + tuple(
            (parent[i], i) for i in order[1:])

    @property
    def scalar_count(self):
        return sum(arr.size for group in (self.D, self.P, self.Q, self.T)
                   for arr in group.values())

    def apply(self, x):
        """Apply to a vector or a matrix of column vectors by two tree sweeps."""
        x = np.asarray(x)
        vector = x.ndim == 1
        if x.ndim not in (1, 2) or x.shape[0] != self.shape[1]:
            raise ValueError("Input must have the operator's row dimension")
        if vector:
            x = x[:, None]
        messages = {}
        for i, j in self.schedule:
            value = self.P[i, j] @ x[self.slices[i]]
            for k in self.adjacency[i]:
                if k != j:
                    value = value + self.T[i, j, k] @ messages[k, i]
            messages[i, j] = value
        blocks = []
        for i, row in enumerate(self.adjacency):
            value = self.D[i] @ x[self.slices[i]]
            for k in row:
                value = value + self.Q[i, k] @ messages[k, i]
            blocks.append(value)
        result = np.concatenate(blocks, axis=0)
        return result[:, 0] if vector else result

    def compose(self, other):
        """Return the descriptor of self * other, without applying either input.

        No minimality, numerical stability, or performance advantage is implied.
        """
        if not isinstance(other, TreeOperator):
            raise TypeError("Composition requires another TreeOperator")
        if self.adjacency != other.adjacency or self.sizes != other.sizes:
            raise ValueError("Composition requires the same tree and physical partition")
        a, b = self, other
        dtype = np.result_type(a.dtype, b.dtype)
        reverse = {}
        for i, j in a.schedule:
            value = a.P[i, j] @ b.Q[i, j]
            for k in a.adjacency[i]:
                if k != j:
                    value = value + a.T[i, j, k] @ reverse[k, i] @ b.T[i, k, j]
            reverse[i, j] = value

        D, P, Q, T = {}, {}, {}, {}
        for i, row in enumerate(a.adjacency):
            diag = a.D[i] @ b.D[i]
            for k in row:
                diag = diag + a.Q[i, k] @ reverse[k, i] @ b.P[i, k]
            D[i] = diag
            for j in row:
                inp = a.P[i, j] @ b.D[i]
                out = a.D[i] @ b.Q[i, j]
                for k in row:
                    if k != j:
                        inp = inp + a.T[i, j, k] @ reverse[k, i] @ b.P[i, k]
                        out = out + a.Q[i, k] @ reverse[k, i] @ b.T[i, k, j]
                P[i, j] = np.concatenate((inp, b.P[i, j]), axis=0)
                Q[i, j] = np.concatenate((a.Q[i, j], out), axis=1)
                for ell in row:
                    if ell == j:
                        continue
                    cross = a.P[i, j] @ b.Q[i, ell]
                    for k in row:
                        if k != j and k != ell:
                            cross = cross + a.T[i, j, k] @ reverse[k, i] @ b.T[i, k, ell]
                    zero = np.zeros((b.P[i, j].shape[0], a.P[ell, i].shape[0]), dtype=dtype)
                    T[i, j, ell] = np.block([[a.T[i, j, ell], cross],
                                              [zero, b.T[i, j, ell]]])
        return TreeOperator(a.adjacency, D, P, Q, T)
