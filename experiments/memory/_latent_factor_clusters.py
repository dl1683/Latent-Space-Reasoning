"""Bounded elimination trees for source-pinned latent factor fields."""

from collections import deque
from dataclasses import dataclass
from math import prod
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import _frozen


def _relative_log(log_values):
    peak = np.max(log_values)
    return log_values if peak == -np.inf else log_values - peak


def _expand(values, axes, target):
    """Broadcast a tensor into a containing scope in the target's order."""
    order = tuple(axes.index(slot) for slot in target if slot in axes)
    shape = tuple(values.shape[axes.index(slot)] if slot in axes else 1 for slot in target)
    return values.transpose(order).reshape(shape)


def _project(values, axes, target):
    order = tuple(axes.index(slot) for slot in target)
    trailing = tuple(i for i, slot in enumerate(axes) if slot not in target)
    if not trailing:
        return _relative_log(values.transpose(order))
    shape = tuple(values.shape[i] for i in order)
    slab = values.transpose((*order, *trailing)).reshape((*shape, -1))
    return _relative_log(np.logaddexp.reduce(slab, axis=-1))



@dataclass(frozen=True, eq=False)
class _ClusterPlan:
    signature: tuple
    bags: tuple
    widths: tuple
    owner: MappingProxyType
    host: MappingProxyType
    groups: tuple
    adjacent: tuple
    separators: MappingProxyType
    schedule: tuple
    peak_cells: int
    retained_cells: int

    @classmethod
    def build(cls, widths, signature, *, max_bag_cells, max_retained_cells):
        neighbors = {}
        for _, axes in signature:
            if prod(widths[slot] for slot in axes) > max_bag_cells:
                return None
            for slot in axes:
                neighbors.setdefault(slot, set()).update(axes)
                neighbors[slot].discard(slot)

        def size(slot):
            return widths[slot] * prod(widths[other] for other in neighbors[slot])

        order, bags, bag_cells = [], [], []
        retained = 0
        while neighbors:
            slot = min(neighbors, key=lambda slot: (size(slot), slot))
            size_needed = size(slot)
            retained += size_needed
            if size_needed > max_bag_cells or retained > max_retained_cells:
                return None
            separator = neighbors.pop(slot)
            for other in separator:
                neighbors[other].remove(slot)
                neighbors[other].update(separator - {other})
            order.append(slot)
            bags.append(tuple(sorted({slot, *separator})))
            bag_cells.append(size_needed)

        # Singleton dimensions cost one cell but still count against NumPy's
        # version-dependent array rank. Check using a scalar view only.
        try:
            np.broadcast_to(0., (1,) * max(map(len, bags)))
        except ValueError:
            return None

        owner = {slot: i for i, slot in enumerate(order)}
        host = {key: min(owner[slot] for slot in axes) for key, axes in signature}
        groups, adjacent = [[] for _ in order], [[] for _ in order]
        for key, bag in host.items():
            groups[bag].append(key)

        separators, upward = {}, []
        for bag, (slot, axes) in enumerate(zip(order, bags, strict=True)):
            separator = tuple(other for other in axes if other != slot)
            if not separator:
                continue
            parent = min(owner[other] for other in separator)
            retained += 2 * prod(widths[other] for other in separator)
            if retained > max_retained_cells:
                return None
            adjacent[bag].append(parent)
            adjacent[parent].append(bag)
            separators[bag, parent] = separators[parent, bag] = separator
            upward.append((bag, parent))
        downward = tuple((parent, bag) for bag, parent in reversed(upward))
        return cls(signature, tuple(bags), widths, MappingProxyType(owner), MappingProxyType(host),
                   tuple(map(tuple, groups)), tuple(map(tuple, adjacent)), MappingProxyType(separators),
                   (*upward, *downward), max(bag_cells), retained)

    def potential(self, bag, factors):
        axes = self.bags[bag]
        result = np.zeros(tuple(self.widths[slot] for slot in axes))
        for key in self.groups[bag]:
            factor = factors[key]
            result += _expand(factor.potential, factor.axes, axes)
        return _frozen(_relative_log(result))

    def invalidate(self, dirty):
        agenda = deque((bag, target) for bag in dirty for target in self.adjacent[bag])
        invalid = set()
        while agenda:
            address = agenda.popleft()
            if address in invalid:
                continue
            invalid.add(address)
            current, downstream = address
            agenda.extend((downstream, target) for target in self.adjacent[downstream] if target != current)
        return invalid


@dataclass(frozen=True, eq=False)
class _ClusterBank:
    plan: _ClusterPlan
    locals: tuple[np.ndarray, ...]
    messages: MappingProxyType

    @classmethod
    def initialize(cls, plan, factors):
        return cls(plan, tuple(plan.potential(bag, factors) for bag in range(len(plan.bags))), MappingProxyType({}))

    def refresh(self, factors, changed_keys):
        dirty = {self.plan.host[key] for key in changed_keys if key in self.plan.host}
        if not dirty:
            return self
        locals_ = tuple(self.plan.potential(bag, factors) if bag in dirty else previous
                        for bag, previous in enumerate(self.locals))
        invalid = self.plan.invalidate(dirty)
        reusable = {address: value for address, value in self.messages.items() if address not in invalid}
        return _ClusterBank(self.plan, locals_, MappingProxyType(reusable))

    @property
    def ready(self):
        return len(self.messages) == len(self.plan.schedule)

    def can_read(self, bag):
        return all((origin, bag) in self.messages for origin in self.plan.adjacent[bag])

    def mass(self, bag, drives, messages, *, omit=None):
        axes = self.plan.bags[bag]
        result = self.locals[bag].copy()
        for slot in axes:
            if self.plan.owner[slot] == bag and slot in drives:
                result += _expand(drives[slot], (slot,), axes)
        for origin in self.plan.adjacent[bag]:
            if origin != omit:
                address = origin, bag
                result += _expand(messages[address], self.plan.separators[address], axes)
        return result

    def settle(self, drives, changed, limits):
        max_updates, max_cells = limits
        invalid = self.plan.invalidate({self.plan.owner[slot] for slot in changed if slot in self.plan.owner})
        if self.ready and not invalid:
            return self, 0, 0
        messages = {address: value for address, value in self.messages.items() if address not in invalid}
        updates = cells = 0
        for address in self.plan.schedule:
            if address in messages:
                continue
            if updates == max_updates:
                break
            origin, target = address
            size = self.locals[origin].size
            if size > max_cells - cells:
                continue
            if any((other, origin) not in messages for other in self.plan.adjacent[origin] if other != target):
                continue
            mass = self.mass(origin, drives, messages, omit=target)
            messages[address] = _frozen(_project(mass, self.plan.bags[origin], self.plan.separators[address]))
            updates += 1
            cells += size
        return _ClusterBank(self.plan, self.locals, MappingProxyType(messages)), updates, cells

    def project(self, bag, axes, drives):
        mass = self.mass(bag, drives, self.messages)
        return _project(mass, self.plan.bags[bag], axes)


def _signature(graph, root):
    return tuple((key, graph.factors[key].axes) for key in sorted(graph.groups[root]))


def _refresh_clusters(clusters, graph, changed_keys):
    return MappingProxyType({
        root: bank.refresh(graph.factors, changed_keys)
        for root, bank in clusters.items()
        if root in graph.loopy and bank.plan.signature == _signature(graph, root)
    })


def _settle_clusters(clusters, drives, changed, limits):
    remaining_updates, remaining_cells = limits
    updates = cells = 0
    banks = {}
    for root, bank in clusters.items():
        current, used, scanned = bank.settle(drives, changed, (remaining_updates, remaining_cells))
        banks[root] = current
        remaining_updates -= used
        remaining_cells -= scanned
        updates += used
        cells += scanned
    return MappingProxyType(banks), updates, cells
