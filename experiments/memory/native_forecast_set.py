"""A frozen comparison of latent continuations against later source observations."""

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import chain

import numpy as np

from experiments.memory.linear_feedback import _LEVELS, _quantize
from experiments.memory.selective_continuations import SelectionStep, frontier_survives
from experiments.memory.source_revision_feed import _OPAQUE_ID, _encode
from experiments.memory.verified_source_tool import SourceSpan, sha256

_MAX_ARMS = 8
_MAX_WITNESSES = 8


def _forecast_fields(features, tolerance, width):
    if (not features or any(type(index) is not int or not 0 <= index < width for index in features)
            or tuple(sorted(set(features))) != features
            or type(tolerance) is not int or not 0 <= tolerance <= 2 * _LEVELS):
        raise ValueError("forecast needs distinct sorted row coordinates and an integer tolerance")


def _forecast_values(row, features):
    return tuple(map(int, _quantize(row[list(features)])))


def _forecast_error(prediction, observation):
    return max(abs(left - right) for left, right in zip(prediction, observation, strict=True))


@dataclass(frozen=True, slots=True)
class NativeForecastWitness:
    source_id: str
    features: tuple[int, ...]
    tolerance: int


@dataclass(frozen=True, slots=True)
class NativeRevisionWitness(NativeForecastWitness):
    """Observe a correction to one already published text source."""

    max_opening_rows: int
    max_changed_rows: int


@dataclass(frozen=True, slots=True)
class _RevisionWitnessOpening:
    version_sha256: str
    generation: int
    rows: tuple[tuple[str, tuple[int, ...]], ...]

    def introduced_rows(self, current, limit):
        """Subtract the opening digest multiset in current document order."""
        remaining = Counter(digest for digest, _ in self.rows)
        introduced = []
        for item in current:
            digest = item.span.span_sha256
            if remaining[digest]:
                remaining[digest] -= 1
            elif len(introduced) == limit:
                return None
            else:
                introduced.append(item)
        return tuple(introduced)


NativeForecastObservation = tuple[int, ...] | tuple[tuple[int, ...], ...]


def _validate_witnesses(specs, width):
    if not 1 <= len(specs) <= _MAX_WITNESSES:
        raise ValueError("forecast set needs one to eight witness specifications")
    seen = set()
    for spec in specs:
        if (type(spec) not in (NativeForecastWitness, NativeRevisionWitness) or type(spec.source_id) is not str
                or not _OPAQUE_ID.fullmatch(spec.source_id) or spec.source_id in seen
                or type(spec.features) is not tuple):
            raise ValueError("forecast set needs distinct source identities")
        _forecast_fields(spec.features, spec.tolerance, width)
        if isinstance(spec, NativeRevisionWitness) and any(
            type(limit) is not int or not 1 <= limit < 2 ** 63
            for limit in (spec.max_opening_rows, spec.max_changed_rows)
        ):
            raise ValueError("revision witness needs positive opening and changed row bounds")
        seen.add(spec.source_id)


def _payload_digest(continuation):
    return sha256(np.asarray([continuation.address, continuation.value], dtype="<f8").tobytes())


def _trace_description(steps):
    return [[asdict(step.receipt), step.query.hex(), step.cutoff, step.applied] for step in steps]


@dataclass(frozen=True, slots=True)
class _ForecastAlternative:
    parent: SourceSpan
    parent_sha256: str
    payload_sha256: str
    predictions: tuple[tuple[int, ...], ...]
    steps: tuple[SelectionStep, ...]


@dataclass(frozen=True, slots=True, eq=False)
class NativeForecastSet:
    seed: SourceSpan
    witnesses: tuple[NativeForecastWitness | NativeRevisionWitness, ...]
    opening_source: str
    opening_epoch: int
    opening_publication: str
    reader_sha256: str
    baseline: tuple[tuple[int, ...], ...]
    _baseline_steps: tuple[SelectionStep, ...]
    alternatives: tuple[_ForecastAlternative, ...]
    _arm_limit: int
    _score_limit: int
    _revision_openings: tuple[_RevisionWitnessOpening | None, ...] = ()

    def _opening_at(self, index):
        return self._revision_openings[index] if self._revision_openings else None

    @property
    def commitment(self):
        """Return the digest the controller records before releasing source events."""
        preimage = [
            "lsr-native-forecast-set-v1", asdict(self.seed), self.opening_source, self.opening_epoch,
            self.opening_publication, self.reader_sha256, self._arm_limit, self._score_limit,
            [asdict(spec) for spec in self.witnesses], self.baseline, _trace_description(self._baseline_steps),
            [[asdict(arm.parent), arm.parent_sha256, arm.payload_sha256, arm.predictions,
              _trace_description(arm.steps)] for arm in self.alternatives],
        ]
        if self._revision_openings:
            preimage[0] = "lsr-native-forecast-set-v2"
            preimage.append([None if opening is None else asdict(opening) for opening in self._revision_openings])
        return sha256(_encode(preimage))

    @property
    def parents(self):
        return tuple(arm.parent for arm in self.alternatives)

    def _matches(self, predictions, observations):
        count = len(observations)
        for spec, predicted, observation in zip(self.witnesses[:count], predictions[:count], observations, strict=True):
            rows = observation if isinstance(spec, NativeRevisionWitness) else (observation,)
            if any(_forecast_error(predicted, row) > spec.tolerance for row in rows):
                return False
        return True

    def _persistence_matches(self, observations):
        """Can the opening document already supply every revised projection?"""
        checked = False
        for index, observation in enumerate(observations):
            opening = self._opening_at(index)
            if opening is None:
                continue
            checked = True
            tolerance = self.witnesses[index].tolerance
            for row in observation:
                if not any(_forecast_error(old, row) <= tolerance for _, old in opening.rows):
                    return False
        return checked

    def _survivors(self, observations):
        return tuple(arm.parent for arm in self.alternatives if self._matches(arm.predictions, observations))

    def _frontiers_survive(self, changes, shards):
        frontiers = chain(self._baseline_steps, *(arm.steps for arm in self.alternatives))
        return all(frontier_survives(step, changes, shards) for step in frontiers)


@dataclass(frozen=True, slots=True)
class _ForecastSetCheck:
    forecast: NativeForecastSet
    witness_spans: tuple[SourceSpan, ...]
    observation_seals: tuple[str, ...] = ()
