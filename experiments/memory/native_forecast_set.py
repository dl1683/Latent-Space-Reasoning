"""A frozen comparison of latent continuations against later source observations."""

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


def _validate_witnesses(specs, width):
    if not 1 <= len(specs) <= _MAX_WITNESSES:
        raise ValueError("forecast set needs one to eight witness specifications")
    seen = set()
    for spec in specs:
        if (type(spec) is not NativeForecastWitness or type(spec.source_id) is not str
                or not _OPAQUE_ID.fullmatch(spec.source_id) or spec.source_id in seen
                or type(spec.features) is not tuple):
            raise ValueError("forecast set needs distinct source identities")
        _forecast_fields(spec.features, spec.tolerance, width)
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
    witnesses: tuple[NativeForecastWitness, ...]
    opening_source: str
    opening_epoch: int
    opening_publication: str
    reader_sha256: str
    baseline: tuple[tuple[int, ...], ...]
    _baseline_steps: tuple[SelectionStep, ...]
    alternatives: tuple[_ForecastAlternative, ...]
    _arm_limit: int
    _score_limit: int

    @property
    def commitment(self):
        """Return the digest the controller records before releasing source events."""
        return sha256(_encode([
            "lsr-native-forecast-set-v1", asdict(self.seed), self.opening_source, self.opening_epoch,
            self.opening_publication, self.reader_sha256, self._arm_limit, self._score_limit,
            [asdict(spec) for spec in self.witnesses], self.baseline, _trace_description(self._baseline_steps),
            [[asdict(arm.parent), arm.parent_sha256, arm.payload_sha256, arm.predictions,
              _trace_description(arm.steps)] for arm in self.alternatives],
        ]))

    @property
    def parents(self):
        return tuple(arm.parent for arm in self.alternatives)

    def _matches(self, predictions, observations):
        count = len(observations)
        return all(_forecast_error(prediction, actual) <= spec.tolerance
                   for spec, prediction, actual in zip(self.witnesses[:count], predictions[:count],
                                                        observations, strict=True))

    def _survivors(self, observations):
        return tuple(arm.parent for arm in self.alternatives if self._matches(arm.predictions, observations))

    def _frontiers_survive(self, changes, shards):
        frontiers = chain(self._baseline_steps, *(arm.steps for arm in self.alternatives))
        return all(frontier_survives(step, changes, shards) for step in frontiers)


@dataclass(frozen=True, slots=True)
class _ForecastSetCheck:
    forecast: NativeForecastSet
    witness_spans: tuple[SourceSpan, ...]
