"""Conventional bounded search of a quantized map, not semantic completeness.

The builder may inspect corpus vectors, never queries or labels. Serving keeps
only codes and tree bounds. Numeric bounds are conservatively padded and tested,
not a universal floating-point certificate. Source truth still needs the leaf.
"""
from dataclasses import dataclass
import heapq
from pathlib import Path

import numpy as np


def scores(vectors, query):
    return np.sum(np.asarray(vectors, dtype=np.float64)
                  * np.asarray(query, dtype=np.float64), axis=1)


def rank(values):
    return np.argsort(-np.asarray(values), kind="stable")


@dataclass(frozen=True)
class HierarchicalMap:
    codes: np.ndarray
    scales: np.ndarray
    order: np.ndarray
    nodes: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def build(cls, vectors, leaf_size=32, boxes=False):
        x = np.asarray(vectors)
        if (x.dtype != np.float32 or x.ndim != 2 or min(x.shape) < 1
                or not np.isfinite(x).all() or type(leaf_size) is not int
                or leaf_size < 1 or type(boxes) is not bool):
            raise ValueError("finite nonempty FP32 vectors and positive leaf size required")
        scale = np.maximum(np.max(np.abs(x), axis=1), 1e-12).astype(np.float32)
        codes = np.clip(np.rint((x/scale[:, None]+1)*127.5), 0, 255).astype(np.uint8)
        obj = cls(codes, scale, *([np.empty(0)]*6))
        values = obj.decode(np.arange(len(x)))
        order, nodes, centers, radii, lows, highs = [], [], [], [], [], []

        def split(ids):
            number = len(nodes)
            nodes.append(None)
            block = values[ids]
            center = block.mean(axis=0, dtype=np.float64).astype(np.float32)
            radius = np.linalg.norm(block.astype(np.float64)-center, axis=1).max()
            centers.append(center)
            radii.append(np.nextafter(radius+1e-12, np.inf))
            if boxes:
                lows.append(block.min(axis=0))
                highs.append(block.max(axis=0))
            start = len(order)
            if len(ids) <= leaf_size:
                order.extend(sorted(ids.tolist()))
                left = right = -1
            else:
                axis = np.ptp(block, axis=0).argmax()
                sorted_ids = ids[np.lexsort((ids, block[:, axis]))]
                midpoint = len(ids)//2
                left, right = split(sorted_ids[:midpoint]), split(sorted_ids[midpoint:])
            nodes[number] = (start, len(ids), left, right)
            return number

        split(np.arange(len(x)))
        obj = cls(codes, scale, np.asarray(order, dtype=np.int32),
                  np.asarray(nodes, dtype=np.int32), np.asarray(centers, dtype=np.float32),
                  np.asarray(radii, dtype=np.float64),
                  np.asarray(lows, dtype=np.float32).reshape(-1, x.shape[1]),
                  np.asarray(highs, dtype=np.float32).reshape(-1, x.shape[1]))
        obj.validate()
        return obj

    def decode(self, ids):
        values = ((2*self.codes[ids].astype(np.float32)/255-1)
                  * self.scales[ids, None])
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    @property
    def payload_bytes(self):
        return sum(getattr(self, key).nbytes for key in self.__dataclass_fields__)

    @property
    def flat_payload_bytes(self):
        return self.codes.nbytes+self.scales.nbytes

    def validate(self):
        n, d = self.codes.shape
        m = len(self.nodes)
        if (n < 1 or d < 1 or self.codes.dtype != np.uint8
                or self.scales.shape != (n,) or self.scales.dtype != np.float32
                or not np.isfinite(self.scales).all() or (self.scales <= 0).any()
                or self.order.dtype != np.int32
                or not np.array_equal(np.sort(self.order), np.arange(n))
                or self.nodes.dtype != np.int32 or self.nodes.shape != (m, 4)
                or m < 1 or self.centers.shape != (m, d)
                or self.centers.dtype != np.float32 or self.radii.shape != (m,)
                or self.radii.dtype != np.float64 or (self.radii < 0).any()
                or not np.isfinite(self.centers).all()
                or not np.isfinite(self.radii).all()
                or self.lower.shape not in ((0, d), (m, d))
                or self.upper.shape != self.lower.shape
                or self.lower.dtype != np.float32 or self.upper.dtype != np.float32
                or not np.isfinite(self.lower).all()
                or not np.isfinite(self.upper).all()
                or (self.lower > self.upper).any()):
            raise ValueError("invalid hierarchy fields")
        reached = set()

        def visit(i, start, count):
            if i in reached or not 0 <= i < m:
                raise ValueError("cycle, repeated or missing node")
            reached.add(i)
            a, b, left, right = map(int, self.nodes[i])
            if (a, b) != (start, count) or count < 1 or start+count > n:
                raise ValueError("invalid node coverage")
            if left == right == -1:
                return
            if not (i < left < m and i < right < m):
                raise ValueError("invalid child")
            left_count = int(self.nodes[left, 1])
            if not 0 < left_count < count:
                raise ValueError("invalid partition")
            visit(left, start, left_count)
            visit(right, start+left_count, count-left_count)

        visit(0, 0, n)
        if len(reached) != m:
            raise ValueError("orphan node")
        # Bounds are trusted builder output bound by an external file digest.
        # Structural validation does not authenticate adversarial numeric bounds.

    def save(self, path):
        self.validate()
        with Path(path).open("xb") as handle:
            np.savez(handle, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(cls.__dataclass_fields__):
                raise ValueError("missing or foreign hierarchy field")
            obj = cls(**{k: data[k] for k in cls.__dataclass_fields__})
        obj.validate()
        return obj

    def search(self, query, k=10):
        q = np.asarray(query)
        if (q.dtype != np.float32 or q.shape != (self.codes.shape[1],)
                or not np.isfinite(q).all() or type(k) is not int
                or not 1 <= k <= len(self.codes)):
            raise ValueError("finite aligned FP32 query and valid k required")
        q64 = q.astype(np.float64)
        norm = np.linalg.norm(q64)
        slack = 1e-11*(1+norm)*len(q)
        bounded_nodes = 0

        def bound(node):
            nonlocal bounded_nodes
            bounded_nodes += 1
            center_score = scores(self.centers[node:node+1], q64)[0]
            result = center_score + self.radii[node]*norm
            if len(self.lower):
                corner = np.where(q64 >= 0, self.upper[node], self.lower[node])
                result = min(result, scores(corner[None], q64)[0])
            return result+slack

        pending, best = [(-bound(0), 0)], []
        decoded, leaf_visits = [], 0
        while pending:
            negative_bound, node = heapq.heappop(pending)
            if len(best) == k and -negative_bound < best[0][0]:
                break
            start, count, left, right = map(int, self.nodes[node])
            if left != -1:
                heapq.heappush(pending, (-bound(left), left))
                heapq.heappush(pending, (-bound(right), right))
                continue
            ids = self.order[start:start+count]
            values = self.decode(ids)
            current = scores(values, q64)
            decoded.extend(ids.tolist())
            leaf_visits += 1
            for row, score in zip(ids, current):
                item = (float(score), -int(row))
                if len(best) < k:
                    heapq.heappush(best, item)
                elif item > best[0]:
                    heapq.heapreplace(best, item)
        answer = sorted(best, reverse=True)
        node_bytes = (self.centers.shape[1]*4+8+16
                      + (self.centers.shape[1]*8 if len(self.lower) else 0))
        return dict(indices=[-row for _, row in answer], scores=[s for s, _ in answer],
                    decoded_rows=len(decoded), leaf_visits=leaf_visits,
                    bounded_nodes=bounded_nodes,
                    logical_bytes=bounded_nodes*node_bytes+len(decoded)*(
                        self.codes.shape[1]+4+4),
                    semantic_absence="UNKNOWN", completeness="UNKNOWN",
                    decoded_ids=decoded)
