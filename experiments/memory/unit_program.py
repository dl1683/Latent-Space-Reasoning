"""Exact, source-scoped conversion programs for multiplicative unit registries."""

import ast
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import re
from types import MappingProxyType

from experiments.memory.verified_source_tool import SourceSpan


MAX_REGISTRY_BYTES = 65_536
_MAX_BITS = 8192
_DIMENSION = re.compile(r"\[[a-zA-Z_][a-zA-Z0-9_]*\]|\[\]")
_TOKENS = re.compile(r"\s+|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
                     r"|[^\W\d]\w*|\*\*|[*/()+-]")
_REFERENCE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.txt")


@dataclass(frozen=True)
class _Product:
    scale: Fraction
    powers: tuple[tuple[str, int], ...] = ()

    def power(self, exponent):
        if abs(exponent) > 20 or any(abs(power * exponent) > 80 for _, power in self.powers):
            raise ValueError("unit exponent exceeds the supported range")
        return _product(self.scale ** exponent, {key: power * exponent for key, power in self.powers})

    def multiply(self, other, sign=1):
        powers = dict(self.powers)
        for key, power in other.powers:
            powers[key] = powers.get(key, 0) + sign * power
        return _product(self.scale * other.scale ** sign, powers)


def _product(scale, powers):
    if max(scale.numerator.bit_length(), scale.denominator.bit_length()) > _MAX_BITS:
        raise ValueError("unit scale exceeds the supported range")
    if any(abs(exponent) > 80 for exponent in powers.values()):
        raise ValueError("unit exponent exceeds the supported range")
    return _Product(scale, tuple(sorted(powers.items())))


def _unit_expression(text):
    if _DIMENSION.fullmatch(text):
        return _Product(Fraction(1), ((text, 1),) if text != "[]" else ())
    if len(text) > 512:
        raise ValueError("unit expression exceeds the supported range")
    names, tokens, previous, end, last_token_end = {}, [], None, 0, 0
    for match in _TOKENS.finditer(text):
        token = match.group()
        if end != match.start():
            raise ValueError("unsupported unit expression")
        end = match.end()
        if token.isspace():
            continue
        if token.isidentifier() or token[0].isdigit() or token[0] == ".":
            encoded = f"v{len(names)}"
            if token.isidentifier():
                kind = "name"
                names[encoded] = _Product(Fraction(1), ((token, 1),))
            else:
                kind = "number"
                if len(token) > 64 or ("e" in token.lower() and abs(int(token.lower().split("e")[1])) > 128):
                    raise ValueError("unit literal exceeds the supported range")
                names[encoded] = _Product(Fraction(token))
        else:
            kind = encoded = token
        if previous in ("number", "name", ")") and kind in ("number", "name", "("):
            if previous != "number" or kind != "name" or match.start() == last_token_end:
                raise ValueError("unsupported implicit unit product")
            tokens.append("*")
        tokens.append(encoded)
        previous, last_token_end = kind, end
    if end != len(text):
        raise ValueError("unsupported unit expression")

    def reduce(node):
        if type(node) is ast.Name and node.id in names:
            return names[node.id]
        if type(node) is ast.UnaryOp and type(node.op) in (ast.UAdd, ast.USub):
            number = reduce(node.operand)
            if not number.powers and type(node.op) is ast.USub:
                return _Product(-number.scale)
            if not number.powers:
                return number
        if type(node) is ast.BinOp and type(node.op) in (ast.Mult, ast.Div, ast.Pow):
            left, right = reduce(node.left), reduce(node.right)
            if type(node.op) is ast.Pow:
                if not right.powers and right.scale.denominator == 1:
                    return left.power(int(right.scale))
            else:
                return left.multiply(right, -1 if type(node.op) is ast.Div else 1)
        raise ValueError("unsupported unit operation")

    try:
        result = reduce(ast.parse(" ".join(tokens), mode="eval").body)
    except (SyntaxError, RecursionError, ZeroDivisionError) as exc:
        raise ValueError("unsupported unit expression") from exc
    if result.scale <= 0:
        raise ValueError("unit scale must be positive")
    return result


@dataclass(frozen=True)
class UnitConversion:
    status: str
    coefficient: Fraction | None = None
    spans: tuple[SourceSpan, ...] = ()


@dataclass(frozen=True)
class _Definition:
    expression: _Product | None
    span: SourceSpan


@dataclass(frozen=True)
class _Resolved:
    status: str
    value: _Product | None = None


def _references(definition):
    if definition.expression is not None:
        yield from (symbol for symbol, _ in definition.expression.powers if not _DIMENSION.fullmatch(symbol))


def _read_registry(source, raw, includes, definitions):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_REGISTRY_BYTES:
        raise ValueError("unit registry bytes exceed the supported range")
    digest, offset, block = sha256(raw).hexdigest(), 0, None
    for line in raw.splitlines(keepends=True):
        raw_prefix = line.split(b"#", 1)[0]
        fragment = raw_prefix.strip()
        start = offset + len(raw_prefix) - len(raw_prefix.lstrip())
        offset += len(line)
        if not fragment:
            continue
        text = fragment.decode("utf8")
        if text.startswith("@"):
            key, _, argument = text.partition(" ")
            name = key.split("(", 1)[0]
            if name == "@end" and block is not None and not argument:
                block = None
            elif block is None and name in ("@group", "@defaults", "@context", "@system"):
                block = name
            elif block is None and key == "@import" and _REFERENCE.fullmatch(argument.strip()):
                includes(argument.strip())
            else:
                raise ValueError("unsupported unit registry directive")
            continue
        if block not in (None, "@group") or "=" not in text:
            continue
        name, expression, *aliases = (part.strip() for part in text.split("="))
        if not name.isidentifier() or name == "_":
            continue
        try:
            product = _unit_expression(expression)
        except ValueError:
            product = None
        definition = _Definition(product, SourceSpan(source, digest, start, start + len(fragment),
                                                     sha256(fragment).hexdigest()))
        definitions.setdefault(name, []).append(definition)
        for alias in dict.fromkeys(aliases):
            if alias and alias not in ("_", name) and alias.isidentifier():
                reference = _Product(Fraction(1), ((name, 1),))
                definitions.setdefault(alias, []).append(_Definition(reference, definition.span))
    if block is not None:
        raise ValueError("unclosed unit registry directive")


@dataclass(frozen=True, eq=False)
class UnitProgram:
    """Evaluate all explicit multiplicative units at one source snapshot."""

    _values: MappingProxyType
    _definitions: MappingProxyType

    @property
    def statuses(self):
        return MappingProxyType({name: result.status for name, result in self._values.items()})

    def convert(self, source, target):
        left, right = (self._values.get(name, _Resolved("missing")) for name in (source, target))
        spans = self._evidence(source, target)
        for status in ("conflict", "open", "missing"):
            if status in (left.status, right.status):
                return UnitConversion(status, spans=spans)
        if left.value.powers != right.value.powers:
            return UnitConversion("incompatible", spans=spans)
        return UnitConversion("resolved", left.value.scale / right.value.scale, spans)

    def _evidence(self, source, target):
        pending, visited, spans = [target, source], set(), {}
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            referenced = []
            for definition in self._definitions.get(name, ()):
                spans[definition.span] = None
                referenced.extend(_references(definition))
            pending.extend(reversed(referenced))
        return tuple(spans)


def _propagate_conflicts(values, definitions):
    """A DFS can encounter a cycle before another branch proves a conflict."""
    dependents = defaultdict(set)
    for name, candidates in definitions.items():
        for candidate in candidates:
            for referenced in _references(candidate):
                dependents[referenced].add(name)
    pending = [name for name, result in values.items() if result.status == "conflict"]
    while pending:
        for dependent in dependents[pending.pop()]:
            if values[dependent].status != "conflict":
                values[dependent] = _Resolved("conflict")
                pending.append(dependent)


def compile_units(entry, records):
    """Compile exact dependencies within the supplied, closed import catalogue."""
    definitions, read, importing = {}, set(), set()

    def include(source):
        if source in importing or source not in records:
            raise ValueError("unit registry import is missing or cyclic")
        if source not in read:
            importing.add(source)
            _read_registry(source, records[source], include, definitions)
            importing.remove(source)
            read.add(source)

    include(entry)
    if read != set(records):
        raise ValueError("unit registry contains records outside its import graph")

    values, resolving = {}, set()

    def expand(expression):
        components, powers, statuses = [], {}, set()
        for symbol, exponent in expression.powers:
            if _DIMENSION.fullmatch(symbol):
                powers[symbol] = powers.get(symbol, 0) + exponent
            else:
                part = resolve(symbol)
                statuses.add(part.status)
                if part.status == "resolved":
                    components.append((part.value, exponent))
        if statuses - {"resolved"}:
            return _Resolved("conflict" if "conflict" in statuses else "open")
        scale = _Product(expression.scale)
        try:
            for value, exponent in components:
                scale = scale.multiply(_Product(value.scale).power(exponent))
                for dimension, power in value.powers:
                    powers[dimension] = powers.get(dimension, 0) + power * exponent
            return _Resolved("resolved", _product(scale.scale, {k: v for k, v in powers.items() if v}))
        except ValueError:
            return _Resolved("open")

    def resolve(name):
        if name in values:
            return values[name]
        if name in resolving or len(resolving) >= 64:
            return _Resolved("open")
        if name not in definitions:
            return _Resolved("missing")
        resolving.add(name)
        candidates = tuple(_Resolved("open") if definition.expression is None else expand(definition.expression)
                           for definition in definitions[name])
        resolving.remove(name)
        keys = {candidate.value for candidate in candidates if candidate.status == "resolved"}
        if len(keys) > 1 or any(candidate.status == "conflict" for candidate in candidates):
            result = _Resolved("conflict")
        elif any(candidate.status != "resolved" for candidate in candidates):
            result = _Resolved("open")
        else:
            result = _Resolved("resolved", keys.pop())
        values[name] = result
        return result

    for name in sorted(definitions):
        resolve(name)
    _propagate_conflicts(values, definitions)
    return UnitProgram(MappingProxyType(values),
                       MappingProxyType({name: tuple(candidates) for name, candidates in definitions.items()}))
