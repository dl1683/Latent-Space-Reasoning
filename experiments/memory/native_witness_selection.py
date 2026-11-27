"""Choose a prospective source witness from frozen native predictions."""

from dataclasses import dataclass, replace
from itertools import combinations

from experiments.memory.native_forecast_set import _forecast_error
from experiments.memory.native_workspace import NativeWorkspaceForecastDraft


@dataclass(frozen=True, slots=True)
class NativeWitnessSelection:
    draft: NativeWorkspaceForecastDraft | None
    roster_commitment: str | None
    options: tuple[tuple[str, int, tuple[tuple[int, ...], ...]], ...]


def witness_separation(option):
    """Return the least gap between rival tolerance boxes in one descriptor."""
    _, tolerance, predictions = option
    return min(_forecast_error(left, right) for left, right in combinations(predictions, 2)) - 2 * tolerance


def choose_forecast_witness(options):
    """Maximize the worst rival separation over outcome-free descriptors."""
    margin, source = min((-witness_separation(option), option[0]) for option in options)
    return source if margin < 0 else None


def draft_discriminating_forecast(workspace, reader, pin, *, witnesses, max_candidates,
                                  max_parent_scores, choose=choose_forecast_witness):
    """Draft once across a roster, then bind one selected source before release.

    The chooser receives immutable (source ID, tolerance, rival predictions)
    records. Return the selected draft and the outcome-free selection record.
    """
    opening = workspace._opening(reader, pin)
    roster = reader.forecast_set(opening, witnesses=witnesses, max_candidates=max_candidates,
                                 max_parent_scores=max_parent_scores)
    if roster is None:
        return NativeWitnessSelection(None, None, ())

    installed = False
    try:
        commitment = NativeWorkspaceForecastDraft(pin, roster, opening).commitment
        options = {
            witness.source_id: (witness.source_id, witness.tolerance,
                                tuple(arm.predictions[index] for arm in roster.alternatives))
            for index, witness in enumerate(roster.witnesses)
        }
        chooser_input = tuple(options.values())
        selected = choose(chooser_input)
        if selected is None:
            return NativeWitnessSelection(None, commitment, chooser_input)
        if type(selected) is not str or selected not in options or witness_separation(options[selected]) <= 0:
            raise ValueError("chosen forecast witness does not separate its rivals")

        index = next(i for i, witness in enumerate(roster.witnesses) if witness.source_id == selected)
        revision_opening = roster._opening_at(index)
        forecast = replace(
            roster,
            witnesses=(roster.witnesses[index],),
            _revision_openings=(revision_opening,) if revision_opening is not None else (),
            baseline=(roster.baseline[index],),
            alternatives=tuple(replace(arm, predictions=(arm.predictions[index],)) for arm in roster.alternatives),
        )
        draft = NativeWorkspaceForecastDraft(pin, forecast, opening)
        with reader._forecast_lock:
            cursor = reader._forecast_cursor
            if (reader._pending_forecast is not roster or cursor.memory is not opening
                    or cursor.observations or cursor.witness_spans or cursor.observation_seals):
                raise ValueError("forecast witness roster is no longer pending at its opening")
            workspace._expect(pin, workspace.head())
            reader._pending_forecast = forecast
            installed = True
        return NativeWitnessSelection(draft, commitment, chooser_input)
    finally:
        if not installed:
            with reader._forecast_lock:
                if reader._pending_forecast is roster:
                    reader._pending_forecast = None
                    reader._forecast_cursor = None
