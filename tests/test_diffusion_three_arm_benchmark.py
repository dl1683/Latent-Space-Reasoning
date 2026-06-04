"""Tests for the diffusion three-arm benchmark runner."""

from argparse import Namespace
from dataclasses import replace

from experiments.run_diffusion_three_arm_benchmark import (
    _anchor_selected_execution_repair,
    _answer_text_matches_proposal,
    _apply_planning_seed_suffix_anchor,
    _arithmetic_claim_count,
    _arithmetic_claim_inconsistencies,
    _arithmetic_claims_consistent,
    _arithmetic_inconsistency_span_targets,
    _choose_pre_generation_repair_anchor,
    _constraint_gap_rescue_candidates,
    _evolved_schedules_for_candidate,
    _exact_answer_repair_selection_score,
    _float_csv,
    _format_fraction_list,
    _generate_exact_answer_repair_records,
    _generate_repair_records,
    _history_instability_gate_decision,
    _history_rescue_candidates,
    _label_free_exact_answer_from_text,
    _label_free_exact_answer_supported,
    _phase_history_anchor_has_source_advantage,
    _phase_history_anchor_passes_source_policy,
    _planning_constraint_gap_span_target_scores,
    _planning_constraint_gap_span_targets,
    _planning_contradiction_penalty,
    _planning_prompt_gate_decision,
    _planning_prompt_gate_seed_suffix_text,
    _planning_quality_prompt_coverage_guarded_score,
    _planning_repair_chunks,
    _planning_span_history_contrast,
    _planning_span_repair_prompt_override,
    _planning_span_residue_penalty,
    _planning_span_target_rows,
    _primary_repair_gate_diagnostics,
    _prompt_constraint_gap_terms,
    _prompt_excluded_final_answer_terms,
    _prompt_final_answer_target_spec,
    _prompt_guided_rescue_candidates,
    _prompt_irrelevant_numbers,
    _prompt_quantity_role_requirements,
    _prompt_required_arithmetic_operators,
    _repair_arithmetic_provenance_gaps,
    _repair_candidates,
    _repair_final_answer_object_gaps,
    _repair_final_answer_role_gaps,
    _repair_final_answer_target_gaps,
    _repair_pack_needs_dense_history,
    _repair_prompt_override,
    _repair_quantity_role_gaps,
    _repair_records_for_source,
    _repair_selection_score,
    _repair_short_text_symbolic_gaps,
    _repair_short_text_trace_gaps,
    _resolve_adaptive_source_gate_mode,
    _resolve_repair_phase_budget,
    _revision_schedules_for_candidate,
    _schedules_for_candidate,
    _seed_objective_score,
    _seed_realization_quality_components,
    _seed_realization_quality_score,
    _select_repair_source_record,
    _select_repair_source_records,
    _select_tasks,
    _selected_evolved_records_for_rescore,
    _selected_history_repair_sample,
    _short_text_answer_schema,
    _should_run_adaptive_history_rescue,
    _should_run_constraint_gap_rescue,
    _should_run_exact_answer_repairs,
    _should_run_history_rescue,
    _should_run_primary_repair_pass,
    _should_run_prompt_guided_rescue,
    _should_run_repairs,
    _should_run_selector_disagreement_rescue,
    _with_history_sample_count,
    render_report,
    select_evolved_record,
    select_repair_record,
    select_three_arm_records,
    summarize_three_arm_scores,
)
from latent_reasoning.diffusion import (
    DiffusionGenerationConfig,
    DiffusionGenerationResult,
    DiffusionRepairCandidate,
    DiffusionScheduleCandidate,
)
from latent_reasoning.eval.general_reasoning import GeneralReasoningTask


def _record(candidate: str, task_id: str, schedule: str, task_score: float, trajectory_score: float):
    combined = 0.75 * task_score + 0.25 * trajectory_score
    return {
        "candidate_key": candidate,
        "task": {"task_id": task_id, "family": "planning", "answer_type": "rubric"},
        "schedule": {"name": schedule},
        "task_score": {"score": task_score},
        "trajectory_control_score": {"overall": trajectory_score},
        "combined_selection_score": combined,
        "text": schedule,
    }


def test_history_sample_count_override_updates_schedule_candidates():
    schedule = DiffusionScheduleCandidate(
        name="low_confidence_32",
        steps=32,
        max_new_tokens=32,
        algorithm="low_confidence",
        history_sample_count=6,
    )

    overridden = _with_history_sample_count((schedule,), 32)

    assert overridden[0].history_sample_count == 32
    assert schedule.history_sample_count == 6
    assert _with_history_sample_count((schedule,), None)[0].history_sample_count == 6


def test_repair_phase_budget_modes_resolve_to_public_cap_ladder():
    assert _resolve_repair_phase_budget("custom", 0) == 0
    assert _resolve_repair_phase_budget("custom", 24) == 24
    assert _resolve_repair_phase_budget("floor", 0) == 9
    assert _resolve_repair_phase_budget("cheap", 0) == 10
    assert _resolve_repair_phase_budget("mid", 0) == 20
    assert _resolve_repair_phase_budget("frontier", 0) == 31

    try:
        _resolve_repair_phase_budget("frontier", 32)
    except SystemExit as exc:
        assert "--repair-phase-budget cannot be combined" in str(exc)
    else:
        raise AssertionError("expected conflicting phase budget settings to exit")


def test_lean_gpu_mixed_task_preset_selects_planning_plus_small_mixed_suite():
    args = Namespace(
        tasks="experiments/general_reasoning_tasks_scout.jsonl",
        families="planning",
        task_ids=None,
        task_preset="lean_gpu_mixed",
        limit_tasks=None,
    )

    tasks = _select_tasks(args)

    assert [task.task_id for task in tasks] == [
        "plan_001",
        "plan_002",
        "plan_003",
        "plan_004",
        "plan_005",
        "plan_006",
        "plan_007",
        "plan_008",
        "math_001",
        "sym_002",
        "sci_001",
    ]


def test_lean_gpu_mixed_transfer_task_preset_selects_independent_transfer_suite():
    args = Namespace(
        tasks="experiments/general_reasoning_tasks_scout.jsonl",
        families="planning",
        task_ids=None,
        task_preset="lean_gpu_mixed_transfer",
        limit_tasks=None,
    )

    tasks = _select_tasks(args)

    assert [task.task_id for task in tasks] == [
        "plan_009",
        "plan_010",
        "plan_011",
        "plan_012",
        "math_009",
        "sym_007",
        "sci_002",
    ]
    assert {task.family for task in tasks} == {"planning", "math", "symbolic", "science"}


def test_lean_gpu_mixed_transfer_v2_task_preset_extends_independent_planning_suite():
    args = Namespace(
        tasks="experiments/general_reasoning_tasks_scout.jsonl",
        families="planning",
        task_ids=None,
        task_preset="lean_gpu_mixed_transfer_v2",
        limit_tasks=None,
    )

    tasks = _select_tasks(args)

    assert [task.task_id for task in tasks] == [
        "plan_009",
        "plan_010",
        "plan_011",
        "plan_012",
        "plan_013",
        "plan_014",
        "plan_015",
        "plan_016",
        "math_009",
        "sym_007",
        "sci_002",
    ]
    assert {task.family for task in tasks} == {"planning", "math", "symbolic", "science"}


def test_lean_gpu_mixed_transfer_v3_task_preset_extends_proof_object_slice():
    args = Namespace(
        tasks="experiments/general_reasoning_tasks_scout.jsonl",
        families="planning",
        task_ids=None,
        task_preset="lean_gpu_mixed_transfer_v3",
        limit_tasks=None,
    )

    tasks = _select_tasks(args)

    assert [task.task_id for task in tasks] == [
        "plan_009",
        "plan_010",
        "plan_011",
        "plan_012",
        "plan_013",
        "plan_014",
        "plan_015",
        "plan_016",
        "plan_017",
        "plan_018",
        "plan_019",
        "plan_020",
        "plan_021",
        "plan_022",
        "plan_023",
        "plan_024",
        "math_009",
        "sym_007",
        "sci_002",
    ]
    assert {task.family for task in tasks} == {"planning", "math", "symbolic", "science"}


def test_lean_gpu_mixed_transfer_v4_task_preset_adds_fresh_learned_predictor_slice():
    args = Namespace(
        tasks="experiments/general_reasoning_tasks_scout.jsonl",
        families="planning",
        task_ids=None,
        task_preset="lean_gpu_mixed_transfer_v4",
        limit_tasks=None,
    )

    tasks = _select_tasks(args)

    assert [task.task_id for task in tasks] == [
        "plan_025",
        "plan_026",
        "plan_027",
        "plan_028",
        "plan_029",
        "plan_030",
        "plan_031",
        "plan_032",
        "math_009",
        "sym_007",
        "sci_002",
    ]
    assert {task.family for task in tasks} == {"planning", "math", "symbolic", "science"}


def test_lean_gpu_mixed_transfer_v5_task_preset_adds_calibrated_predictor_slice():
    args = Namespace(
        tasks="experiments/general_reasoning_tasks_scout.jsonl",
        families="planning",
        task_ids=None,
        task_preset="lean_gpu_mixed_transfer_v5",
        limit_tasks=None,
    )

    tasks = _select_tasks(args)

    assert [task.task_id for task in tasks] == [
        "plan_033",
        "plan_034",
        "plan_035",
        "plan_036",
        "plan_037",
        "plan_038",
        "plan_039",
        "plan_040",
        "math_009",
        "sym_007",
        "sci_002",
    ]
    assert {task.family for task in tasks} == {"planning", "math", "symbolic", "science"}


def test_lean_gpu_mixed_transfer_v6_task_preset_adds_candidate_promotion_slice():
    args = Namespace(
        tasks="experiments/general_reasoning_tasks_scout.jsonl",
        families="planning",
        task_ids=None,
        task_preset="lean_gpu_mixed_transfer_v6",
        limit_tasks=None,
    )

    tasks = _select_tasks(args)

    assert [task.task_id for task in tasks] == [
        "plan_041",
        "plan_042",
        "plan_043",
        "plan_044",
        "plan_045",
        "plan_046",
        "plan_047",
        "plan_048",
        "math_009",
        "sym_007",
        "sci_002",
    ]
    assert {task.family for task in tasks} == {"planning", "math", "symbolic", "science"}


def test_lean_gpu_mixed_transfer_v7_task_preset_adds_fresh_incumbent_slice():
    args = Namespace(
        tasks="experiments/general_reasoning_tasks_scout.jsonl",
        families="planning",
        task_ids=None,
        task_preset="lean_gpu_mixed_transfer_v7",
        limit_tasks=None,
    )

    tasks = _select_tasks(args)

    assert [task.task_id for task in tasks] == [
        "plan_049",
        "plan_050",
        "plan_051",
        "plan_052",
        "plan_053",
        "plan_054",
        "plan_055",
        "plan_056",
        "math_009",
        "sym_007",
        "sci_002",
    ]
    assert {task.family for task in tasks} == {"planning", "math", "symbolic", "science"}


def test_revision_schedules_enable_non_monotonic_llada_revision_config():
    schedule = DiffusionScheduleCandidate(
        name="low_confidence_32",
        steps=32,
        max_new_tokens=32,
        algorithm="low_confidence",
        remasking="low_confidence",
    )

    revisions = _revision_schedules_for_candidate(
        "LLaDA 8B",
        (schedule,),
        revision_remask_fraction=0.50,
        revision_steps=24,
    )

    assert [schedule.name for schedule in revisions] == [
        "evolved_revision_low_confidence_32",
        "evolved_revision_random_32",
    ]
    assert revisions[0].revision_remask_fraction == 0.50
    assert revisions[0].revision_steps == 24
    assert revisions[0].to_config().revision_remask_fraction == 0.50


def test_sparse_llada_moe_uses_llada_schedule_and_repair_surface():
    family = "LLaDA MoE 7B-A1B"
    schedules = _schedules_for_candidate(family)

    assert [schedule.name for schedule in schedules] == ["low_confidence_32", "random_32"]
    assert schedules[0].max_new_tokens == 64

    evolved = _evolved_schedules_for_candidate(family, schedules, limit=2)
    assert [schedule.name for schedule in evolved] == [
        "evolved_low_confidence_48",
        "evolved_random_48",
    ]

    revisions = _revision_schedules_for_candidate(
        family,
        schedules,
        revision_remask_fraction=0.50,
        revision_steps=24,
    )
    assert [schedule.name for schedule in revisions] == [
        "evolved_revision_low_confidence_32",
        "evolved_revision_random_32",
    ]

    rubric_task = GeneralReasoningTask(
        task_id="plan",
        family="planning",
        prompt="Choose an overnight GPU plan.",
        answer_type="rubric",
        scorer="planning_quality",
        max_new_tokens=64,
        rubric_items=("baseline", "intervention"),
    )
    assert _should_run_repairs(family, rubric_task, 1)

    exact_task = GeneralReasoningTask(
        task_id="sym",
        family="symbolic",
        prompt="A lamp starts off. It is toggled 5 times. Is it on or off at the end? Answer only on or off.",
        answer_type="short_text",
        scorer="exact_short_text",
        answer="on",
        max_new_tokens=16,
    )
    source = _record("llada-moe-7b-a1b-instruct-hf", "sym", "baseline", task_score=0.0, trajectory_score=0.4)
    source["task_score"]["extracted_answer"] = "off"
    assert _should_run_exact_answer_repairs(family, exact_task, 2, source)


def _record_with_history(
    candidate: str,
    task_id: str,
    schedule: str,
    task_score: float,
    trajectory_score: float,
    text: str,
    samples: list[str],
):
    record = _record(candidate, task_id, schedule, task_score, trajectory_score)
    record["text"] = text
    record["trajectory_summary"] = {
        "samples": [
            {"step": index + 1, "visible_text": sample}
            for index, sample in enumerate(samples)
        ]
    }
    return record


class _FakeExactRepairBackend:
    def __init__(self, outputs: list[str], tokenizer=None, token_id_outputs: list[list[int]] | None = None) -> None:
        self.outputs = list(outputs)
        self.token_id_outputs = list(token_id_outputs or [])
        self.prompts: list[str] = []
        self.configs = []
        self.tokenizer = tokenizer

    def generate(self, prompt: str, config=None) -> DiffusionGenerationResult:
        self.prompts.append(prompt)
        self.configs.append(config)
        text = self.outputs.pop(0)
        token_ids = self.token_id_outputs.pop(0) if self.token_id_outputs else []
        return DiffusionGenerationResult(
            text=text,
            prompt=prompt,
            candidate_key="llada-8b-instruct-hf",
            model_id="fake-llada",
            config=config.to_dict() if hasattr(config, "to_dict") else {},
            generated_token_ids=token_ids,
            generated_token_count=len(token_ids),
        )


class _TokenPiecesTokenizer:
    def __init__(self, pieces: dict[int, str]) -> None:
        self.pieces = pieces

    def decode(self, token_ids, **_kwargs) -> str:
        return "".join(self.pieces[int(token_id)] for token_id in token_ids)


def test_three_arm_selection_uses_fixed_random_and_trajectory_without_task_score():
    records = [
        _record("model", "task", "fixed", task_score=1.0, trajectory_score=0.1),
        _record("model", "task", "middle", task_score=0.5, trajectory_score=0.4),
        _record("model", "task", "best_trajectory", task_score=0.0, trajectory_score=0.9),
    ]

    selected = select_three_arm_records(records, seed=1, candidate_key="model", task_id="task")

    assert selected["fixed"]["schedule"]["name"] == "fixed"
    assert selected["trajectory_selected"]["schedule"]["name"] == "best_trajectory"
    assert selected["trajectory_selected"]["task_score"]["score"] == 0.0
    assert selected["random"] in records


def test_exact_answer_trajectory_selection_uses_fixed_guard_by_default():
    records = [
        _record("model", "task", "fixed", task_score=1.0, trajectory_score=0.1),
        _record("model", "task", "unstable_wrong", task_score=0.0, trajectory_score=0.9),
    ]

    selected = select_three_arm_records(
        records,
        seed=1,
        candidate_key="model",
        task_id="task",
        task_answer_type="integer",
    )

    assert selected["trajectory_selected"]["schedule"]["name"] == "fixed"


def test_exact_answer_trajectory_selection_can_be_forced_to_raw_trajectory():
    records = [
        _record("model", "task", "fixed", task_score=1.0, trajectory_score=0.1),
        _record("model", "task", "unstable_wrong", task_score=0.0, trajectory_score=0.9),
    ]

    selected = select_three_arm_records(
        records,
        seed=1,
        candidate_key="model",
        task_id="task",
        task_answer_type="integer",
        exact_task_trajectory_policy="trajectory",
    )

    assert selected["trajectory_selected"]["schedule"]["name"] == "unstable_wrong"


def test_exact_answer_evolved_selection_uses_fixed_guard_by_default():
    records = [
        _record("model", "task", "fixed", task_score=1.0, trajectory_score=0.1),
        _record("model", "task", "evolved_wrong", task_score=0.0, trajectory_score=0.95),
    ]

    selected = select_evolved_record(records, task_answer_type="integer")

    assert selected["schedule"]["name"] == "fixed"


def test_evolved_selection_can_choose_mutated_schedule_pool():
    records = [
        _record("model", "plan", "fixed", task_score=0.0, trajectory_score=0.2),
        _record("model", "plan", "base_best", task_score=0.0, trajectory_score=0.5),
        _record("model", "plan", "evolved_best", task_score=0.0, trajectory_score=0.9),
    ]

    selected = select_evolved_record(
        records,
        task_answer_type="rubric",
        trajectory_selector="generic",
    )

    assert selected["schedule"]["name"] == "evolved_best"


def test_evolved_selection_keeps_base_when_selector_edge_is_tiny():
    baseline = _record("model", "plan", "base_trajectory", task_score=0.5, trajectory_score=0.5)
    evolved = _record("model", "plan", "tiny_edge", task_score=0.0, trajectory_score=0.51)

    selected = select_evolved_record(
        [baseline, evolved],
        baseline_record=baseline,
        task_answer_type="rubric",
        trajectory_selector="generic",
        promotion_margin=0.02,
    )

    assert selected["schedule"]["name"] == "base_trajectory"


def test_evolved_selection_promotes_when_selector_edge_clears_margin():
    baseline = _record("model", "plan", "base_trajectory", task_score=0.5, trajectory_score=0.5)
    evolved = _record("model", "plan", "clear_edge", task_score=0.0, trajectory_score=0.53)

    selected = select_evolved_record(
        [baseline, evolved],
        baseline_record=baseline,
        task_answer_type="rubric",
        trajectory_selector="generic",
        promotion_margin=0.02,
    )

    assert selected["schedule"]["name"] == "clear_edge"


def test_revision_evolved_selection_uses_stricter_promotion_margin():
    baseline = _record("model", "plan", "base_trajectory", task_score=0.5, trajectory_score=0.50)
    revision = _record(
        "model",
        "plan",
        "evolved_revision_low_confidence_32",
        task_score=0.0,
        trajectory_score=0.53,
    )

    selected = select_evolved_record(
        [baseline, revision],
        baseline_record=baseline,
        task_answer_type="rubric",
        trajectory_selector="generic",
        promotion_margin=0.02,
        revision_promotion_margin=0.05,
    )

    assert selected["schedule"]["name"] == "base_trajectory"


def test_revision_evolved_selection_promotes_after_revision_margin():
    baseline = _record("model", "plan", "base_trajectory", task_score=0.5, trajectory_score=0.50)
    revision = _record(
        "model",
        "plan",
        "evolved_revision_low_confidence_32",
        task_score=0.0,
        trajectory_score=0.56,
    )

    selected = select_evolved_record(
        [baseline, revision],
        baseline_record=baseline,
        task_answer_type="rubric",
        trajectory_selector="generic",
        promotion_margin=0.02,
        revision_promotion_margin=0.05,
    )

    assert selected["schedule"]["name"] == "evolved_revision_low_confidence_32"


def test_rescore_evolved_record_limit_keeps_revision_schedules_separate_from_mutation_limit():
    records = [
        _record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.50),
        _record("model", "plan", "evolved_low_confidence_48", task_score=0.0, trajectory_score=0.62),
        _record("model", "plan", "evolved_random_48", task_score=0.0, trajectory_score=0.40),
        _record("model", "plan", "evolved_revision_low_confidence_32", task_score=0.0, trajectory_score=0.55),
        _record("model", "plan", "evolved_revision_random_32", task_score=0.0, trajectory_score=0.56),
    ]

    selected = _selected_evolved_records_for_rescore(
        records,
        limit_evolved_schedules=2,
        include_revision_schedules=True,
    )

    assert [record["schedule"]["name"] for record in selected] == [
        "evolved_low_confidence_48",
        "evolved_random_48",
        "evolved_revision_low_confidence_32",
        "evolved_revision_random_32",
    ]

    revision_only = _selected_evolved_records_for_rescore(
        records,
        limit_evolved_schedules=0,
        include_revision_schedules=True,
    )

    assert [record["schedule"]["name"] for record in revision_only] == [
        "evolved_revision_low_confidence_32",
        "evolved_revision_random_32",
    ]


def test_repair_source_policy_preserves_evolved_default():
    selected = {
        "fixed": _record("model", "plan", "fixed", task_score=0.0, trajectory_score=0.10),
        "random": _record("model", "plan", "random", task_score=0.0, trajectory_score=0.20),
        "trajectory_selected": _record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.50),
    }
    evolved = _record(
        "model",
        "plan",
        "evolved_revision_random_32",
        task_score=0.0,
        trajectory_score=0.70,
    )

    source = _select_repair_source_record(
        "evolved",
        selected_records=selected,
        evolved_record=evolved,
        candidate_records=[*selected.values(), evolved],
        trajectory_selector="generic",
    )

    assert source is evolved


def test_repair_source_policy_can_fall_back_to_best_non_revision_evolved_source():
    selected = {
        "fixed": _record("model", "plan", "fixed", task_score=0.0, trajectory_score=0.10),
        "random": _record("model", "plan", "random", task_score=0.0, trajectory_score=0.20),
        "trajectory_selected": _record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.50),
    }
    non_revision_evolved = _record(
        "model",
        "plan",
        "evolved_low_confidence_48",
        task_score=0.0,
        trajectory_score=0.62,
    )
    revision = _record(
        "model",
        "plan",
        "evolved_revision_random_32",
        task_score=0.0,
        trajectory_score=0.70,
    )

    source = _select_repair_source_record(
        "non_revision_evolved",
        selected_records=selected,
        evolved_record=revision,
        candidate_records=[*selected.values(), non_revision_evolved, revision],
        trajectory_selector="generic",
        evolved_promotion_margin=0.02,
    )

    assert source["schedule"]["name"] == "evolved_low_confidence_48"


def test_repair_source_policy_keeps_non_revision_evolved_source():
    selected = {
        "fixed": _record("model", "plan", "fixed", task_score=0.0, trajectory_score=0.10),
        "random": _record("model", "plan", "random", task_score=0.0, trajectory_score=0.20),
        "trajectory_selected": _record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.50),
    }
    evolved = _record("model", "plan", "evolved_low_confidence_48", task_score=0.0, trajectory_score=0.62)

    source = _select_repair_source_record(
        "non_revision_evolved",
        selected_records=selected,
        evolved_record=evolved,
        candidate_records=[*selected.values(), evolved],
        trajectory_selector="generic",
    )

    assert source is evolved


def test_repair_source_policy_can_spend_from_evolved_and_trajectory_sources():
    selected = {
        "fixed": _record("model", "plan", "fixed", task_score=0.0, trajectory_score=0.10),
        "random": _record("model", "plan", "random", task_score=0.0, trajectory_score=0.20),
        "trajectory_selected": _record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.50),
    }
    evolved = _record("model", "plan", "evolved_low_confidence_48", task_score=0.0, trajectory_score=0.62)

    sources = _select_repair_source_records(
        "evolved_and_trajectory",
        selected_records=selected,
        evolved_record=evolved,
        candidate_records=[*selected.values(), evolved],
        trajectory_selector="generic",
    )

    assert [source["schedule"]["name"] for source in sources] == [
        "evolved_low_confidence_48",
        "low_confidence_32",
    ]


def test_adaptive_gap_trajectory_source_policy_spends_only_specific_low_confidence_gap_source():
    task_prompt = (
        "Measure baseline, intervention, rollback, threshold, latency, accuracy, owner, risk, "
        "customer, dashboard, migration, and root cause."
    )
    selected = {
        "fixed": _record("model", "plan", "fixed", task_score=0.0, trajectory_score=0.10),
        "random": _record("model", "plan", "random", task_score=0.0, trajectory_score=0.20),
        "trajectory_selected": {
            **_record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.50),
            "text": "baseline output",
        },
    }
    evolved = _record("model", "plan", "evolved_low_confidence_48", task_score=0.0, trajectory_score=0.62)

    no_gap = _select_repair_source_records(
        "non_revision_plus_gap_trajectory",
        selected_records=selected,
        evolved_record=evolved,
        candidate_records=[*selected.values(), evolved],
        task_prompt=task_prompt,
        trajectory_selector="generic",
    )

    assert [source["schedule"]["name"] for source in no_gap] == ["evolved_low_confidence_48"]

    selected["trajectory_selected"]["text"] = (
        "Measure baseline and rollback, define a decision rule, name risk, assign owner, "
        "validate failure cause, and add a fallback check."
    )
    with_gap = _select_repair_source_records(
        "non_revision_plus_gap_trajectory",
        selected_records=selected,
        evolved_record=evolved,
        candidate_records=[*selected.values(), evolved],
        task_prompt=task_prompt,
        trajectory_selector="generic",
    )

    assert [source["schedule"]["name"] for source in with_gap] == [
        "evolved_low_confidence_48",
        "low_confidence_32",
    ]

    strict_gap = _select_repair_source_records(
        "non_revision_plus_gap_trajectory",
        selected_records=selected,
        evolved_record=evolved,
        candidate_records=[*selected.values(), evolved],
        task_prompt=task_prompt,
        trajectory_selector="generic",
        adaptive_source_gap_min_terms=9,
    )

    assert [source["schedule"]["name"] for source in strict_gap] == ["evolved_low_confidence_48"]

    strict_quality = _select_repair_source_records(
        "non_revision_plus_gap_trajectory",
        selected_records=selected,
        evolved_record=evolved,
        candidate_records=[*selected.values(), evolved],
        task_prompt=task_prompt,
        trajectory_selector="generic",
        adaptive_source_quality_floor=0.90,
    )

    assert [source["schedule"]["name"] for source in strict_quality] == ["evolved_low_confidence_48"]

    quality_ceiling = _select_repair_source_records(
        "non_revision_plus_gap_trajectory",
        selected_records=selected,
        evolved_record=evolved,
        candidate_records=[*selected.values(), evolved],
        task_prompt=task_prompt,
        trajectory_selector="generic",
        adaptive_source_quality_ceiling=0.0,
    )

    assert [source["schedule"]["name"] for source in quality_ceiling] == ["evolved_low_confidence_48"]

    selected["trajectory_selected"]["schedule"]["name"] = "random_32"
    random_source = _select_repair_source_records(
        "non_revision_plus_gap_trajectory",
        selected_records=selected,
        evolved_record=evolved,
        candidate_records=[*selected.values(), evolved],
        task_prompt=task_prompt,
        trajectory_selector="generic",
    )

    assert [source["schedule"]["name"] for source in random_source] == ["evolved_low_confidence_48"]


def test_prompt_coverage_guarded_repair_selector_requires_minimum_planning_quality_for_bonus():
    task_prompt = "Fix the customer dashboard timezone migration today and do root cause later."
    prompt_specific = {
        **_record("model", "plan", "prompt_specific", task_score=0.0, trajectory_score=0.3),
        "text": "Fix the customer dashboard timezone migration today and do root cause later.",
    }
    keyword_stuffed = {
        **_record("model", "plan", "keyword_stuffed", task_score=0.0, trajectory_score=0.3),
        "text": "customer dashboard timezone migration today root cause",
    }

    assert _planning_quality_prompt_coverage_guarded_score(
        prompt_specific,
        task_prompt,
    ) > _planning_quality_prompt_coverage_guarded_score(
        {**prompt_specific, "text": "Analyze the result and document a plan."},
        task_prompt,
    )
    assert _planning_quality_prompt_coverage_guarded_score(keyword_stuffed, task_prompt) < 0.30


def test_adaptive_source_gate_modes_resolve_fresh_confirmed_thresholds():
    assert _resolve_adaptive_source_gate_mode(
        "custom",
        gap_min_terms=3,
        quality_floor=0.4,
    ) == (3, 0.4, None)
    assert _resolve_adaptive_source_gate_mode(
        "score_max",
        gap_min_terms=99,
        quality_floor=0.99,
    ) == (6, 0.25, None)
    assert _resolve_adaptive_source_gate_mode(
        "efficiency",
        gap_min_terms=1,
        quality_floor=0.0,
    ) == (10, 0.25, None)
    assert _resolve_adaptive_source_gate_mode(
        "score_efficient",
        gap_min_terms=99,
        quality_floor=0.99,
    ) == (6, 0.25, 0.50)


def test_summary_reports_adaptive_source_gate_features():
    task_prompt = (
        "Measure baseline, intervention, rollback, threshold, latency, accuracy, owner, risk, "
        "customer, dashboard, migration, and root cause."
    )
    fixed = {**_record("model", "plan", "low_confidence_32", 0.10, 0.10), "prompt": task_prompt}
    random = {**_record("model", "plan", "random_32", 0.10, 0.10), "prompt": task_prompt}
    trajectory = {
        **_record("model", "plan", "low_confidence_32", 0.30, 0.50),
        "prompt": task_prompt,
        "text": (
            "Measure baseline and rollback, define a decision rule, name risk, assign owner, "
            "validate failure cause, and add a fallback check."
        ),
    }
    evolved = {
        **_record("model", "plan", "evolved_low_confidence_48", 0.35, 0.52),
        "prompt": task_prompt,
    }
    repair = {
        **_record("model", "plan", "constraint_gap_span_repair", 0.40, 0.55),
        "prompt": task_prompt,
        "repair": {
            "name": "constraint_gap_span_repair",
            "source_control": "low_confidence_32",
            "source_state": "final",
        },
    }
    exact_fixed = {**_record("model", "sym", "low_confidence_32", 1.0, 0.10), "prompt": "A lamp is toggled."}
    exact_random = {**_record("model", "sym", "random_32", 1.0, 0.10), "prompt": "A lamp is toggled."}
    exact_trajectory = {
        **_record("model", "sym", "low_confidence_32", 1.0, 0.10),
        "prompt": "A lamp is toggled.",
    }
    exact_evolved = {
        **_record("model", "sym", "evolved_random_48", 1.0, 0.10),
        "prompt": "A lamp is toggled.",
    }
    for record in (exact_fixed, exact_random, exact_trajectory, exact_evolved):
        record["task"]["family"] = "symbolic"
        record["task"]["answer_type"] = "short_text"
    arm_records = [
        {**fixed, "arm": "fixed", "arm_generation_budget_per_task": 1},
        {**random, "arm": "random", "arm_generation_budget_per_task": 1},
        {**trajectory, "arm": "trajectory_selected", "arm_generation_budget_per_task": 2},
        {**evolved, "arm": "evolved", "arm_generation_budget_per_task": 6},
        {**repair, "arm": "repair_selected", "arm_generation_budget_per_task": 7},
        {**exact_fixed, "arm": "fixed", "arm_generation_budget_per_task": 1},
        {**exact_random, "arm": "random", "arm_generation_budget_per_task": 1},
        {**exact_trajectory, "arm": "trajectory_selected", "arm_generation_budget_per_task": 2},
        {**exact_evolved, "arm": "evolved", "arm_generation_budget_per_task": 6},
    ]

    scores = summarize_three_arm_scores(
        [
            fixed,
            random,
            trajectory,
            evolved,
            repair,
            exact_fixed,
            exact_random,
            exact_trajectory,
            exact_evolved,
        ],
        arm_records,
        repair_source_policy="non_revision_plus_gap_trajectory",
        adaptive_source_gap_min_terms=6,
        adaptive_source_quality_floor=0.25,
    )

    assert scores["adaptive_source_gap_min_terms"] == 6
    assert scores["adaptive_source_quality_floor"] == 0.25
    assert scores["adaptive_source_quality_ceiling"] is None
    assert len(scores["adaptive_source_gate_rows"]) == 1
    assert scores["adaptive_source_gate_rows"][0]["task_id"] == "plan"
    assert scores["adaptive_source_gate_rows"][0]["add"] is True
    assert scores["adaptive_source_gate_rows"][0]["reason"] == "add"
    assert scores["adaptive_source_gate_rows"][0]["prompt_gap_term_count"] >= 6
    assert scores["adaptive_source_gate_rows"][0]["generated_repair_count"] == 1
    assert scores["adaptive_source_gate_rows"][0]["selected_repair_count"] == 1

    report = render_report(scores)
    assert "## Lean Three-Arm Headline" in report
    assert "fixed baseline | repair-covered tasks | 0.100000" in report
    assert "random perturbation | repair-covered tasks | 0.100000" in report
    assert "selected latent repair | repair-covered tasks | 0.400000" in report
    assert "Trajectory/evolved/oracle rows below are diagnostics" in report


def test_summary_run_identity_is_deterministic_and_ignores_created_at():
    fixed = {**_record("model", "plan", "low_confidence_32", 0.10, 0.10), "created_at": "a"}
    random = {**_record("model", "plan", "random_32", 0.15, 0.20), "created_at": "a"}
    trajectory = {**_record("model", "plan", "low_confidence_64", 0.30, 0.50), "created_at": "a"}
    arm_records = [
        {**fixed, "arm": "fixed", "arm_generation_budget_per_task": 1, "created_at": "a"},
        {**random, "arm": "random", "arm_generation_budget_per_task": 1, "created_at": "a"},
        {
            **trajectory,
            "arm": "trajectory_selected",
            "arm_generation_budget_per_task": 3,
            "created_at": "a",
        },
    ]
    changed_timestamps = [
        {**fixed, "created_at": "b"},
        {**random, "created_at": "b"},
        {**trajectory, "created_at": "b"},
    ]
    changed_arm_timestamps = [{**record, "created_at": "b"} for record in arm_records]

    scores = summarize_three_arm_scores([fixed, random, trajectory], arm_records)
    changed_scores = summarize_three_arm_scores(changed_timestamps, changed_arm_timestamps)
    changed_phase_budget = summarize_three_arm_scores(
        [fixed, random, trajectory],
        arm_records,
        repair_phase_budget="frontier",
        repair_denoise_skeleton_max_step=31,
    )
    changed_content = summarize_three_arm_scores(
        [{**fixed, "text": "changed"}, random, trajectory],
        arm_records,
    )
    changed_gate_diagnostics = summarize_three_arm_scores(
        [fixed, random, trajectory],
        arm_records,
        repair_spend_gate_rows=[
            {
                "candidate_key": "model",
                "task_id": "plan",
                "source_control": "low_confidence_64",
                "should_run": False,
                "reason": "diagnostic_only",
            }
        ],
    )

    assert scores["run_id"].startswith("diffusion-")
    assert len(scores["content_hash"]) == 64
    assert changed_scores["run_id"] == scores["run_id"]
    assert changed_scores["content_hash"] == scores["content_hash"]
    assert changed_phase_budget["run_id"] == scores["run_id"]
    assert changed_phase_budget["content_hash"] == scores["content_hash"]
    assert changed_phase_budget["repair_phase_budget"] == "frontier"
    assert changed_phase_budget["repair_denoise_skeleton_max_step"] == 31
    assert "Repair phase budget: `frontier`" in render_report(changed_phase_budget)
    assert changed_gate_diagnostics["run_id"] == scores["run_id"]
    assert changed_gate_diagnostics["content_hash"] == scores["content_hash"]
    assert changed_content["run_id"] != scores["run_id"]
    assert f"Run ID: `{scores['run_id']}`" in render_report(scores)


def test_evolved_planning_quality_fallback_can_promote_near_tie_mutation():
    baseline = {
        **_record("model", "plan", "base_trajectory", task_score=0.0, trajectory_score=0.50),
        "text": "Assess the result with generic reasoning quality.",
    }
    evolved = {
        **_record("model", "plan", "evolved_near_tie", task_score=0.0, trajectory_score=0.505),
        "text": (
            "Measure the baseline and intervention, compare metrics, preserve rollback, "
            "and define a decision threshold."
        ),
    }

    selected = select_evolved_record(
        [baseline, evolved],
        baseline_record=baseline,
        task_prompt="Compare a baseline and intervention, preserve rollback, and define a decision threshold.",
        task_answer_type="rubric",
        trajectory_selector="generic",
        evolved_selector="planning_quality_fallback",
        promotion_margin=0.02,
    )

    assert selected["schedule"]["name"] == "evolved_near_tie"


def test_evolved_planning_quality_fallback_ignores_low_state_score_mutation():
    baseline = {
        **_record("model", "plan", "base_trajectory", task_score=0.0, trajectory_score=0.50),
        "text": "Assess the result with generic reasoning quality.",
    }
    evolved = {
        **_record("model", "plan", "evolved_low_state", task_score=0.0, trajectory_score=0.40),
        "text": (
            "Measure the baseline and intervention, compare metrics, preserve rollback, "
            "and define a decision threshold."
        ),
    }

    selected = select_evolved_record(
        [baseline, evolved],
        baseline_record=baseline,
        task_prompt="Compare a baseline and intervention, preserve rollback, and define a decision threshold.",
        task_answer_type="rubric",
        trajectory_selector="generic",
        evolved_selector="planning_quality_fallback",
        promotion_margin=0.02,
    )

    assert selected["schedule"]["name"] == "base_trajectory"


def test_repair_selection_uses_same_margin_gate_as_evolved_selection():
    baseline = _record("model", "plan", "evolved", task_score=0.5, trajectory_score=0.5)
    repair = _record("model", "plan", "repair", task_score=0.0, trajectory_score=0.51)
    repair["repair"] = {"name": "low_confidence_25_repair"}
    repair["schedule"] = None

    selected = select_repair_record(
        [baseline, repair],
        baseline_record=baseline,
        task_answer_type="rubric",
        trajectory_selector="generic",
        repair_selector="inherit",
        promotion_margin=0.02,
    )

    assert selected["schedule"]["name"] == "evolved"


def test_transfer_promotion_value_reuses_inherited_planning_state_score():
    prompt = "Compare a baseline and intervention, preserve rollback, and define a decision threshold."
    baseline = {
        **_record("model", "plan", "evolved", task_score=0.5, trajectory_score=0.2),
        "text": "Compare the baseline and intervention, then record a threshold.",
    }
    repair = {
        **_record("model", "plan", "repair", task_score=0.0, trajectory_score=0.2),
        "schedule": None,
        "repair": {"name": "prefix_25_repair"},
        "text": (
            "Compare the baseline and intervention metrics, preserve rollback, "
            "monitor risk, and define a decision threshold."
        ),
    }

    inherited_score = _repair_selection_score(
        repair,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="planning_state",
        repair_selector="inherit",
    )
    transfer_score = _repair_selection_score(
        repair,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="planning_state",
        repair_selector="transfer_promotion_value",
    )
    selected = select_repair_record(
        [baseline, repair],
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="planning_state",
        repair_selector="transfer_promotion_value",
        promotion_margin=0.0,
    )

    assert transfer_score == inherited_score
    assert selected["repair"]["name"] == "prefix_25_repair"


def test_repair_selection_can_use_planning_quality_instead_of_trajectory_score():
    baseline = {
        **_record("model", "plan", "evolved", task_score=0.5, trajectory_score=0.9),
        "text": "Assess the result with generic reasoning quality.",
    }
    repair = {
        **_record("model", "plan", "repair", task_score=0.0, trajectory_score=0.2),
        "schedule": None,
        "repair": {"name": "prefix_25_repair"},
        "text": (
            "Measure the baseline and intervention, compare metrics, preserve rollback, "
            "and define a decision threshold."
        ),
    }

    selected = select_repair_record(
        [baseline, repair],
        baseline_record=baseline,
        task_prompt="Compare a baseline and intervention, preserve rollback, and define a decision threshold.",
        task_answer_type="rubric",
        trajectory_selector="generic",
        repair_selector="planning_quality",
        promotion_margin=0.02,
    )

    assert selected["repair"]["name"] == "prefix_25_repair"


def test_exact_answer_repair_selection_uses_proposal_match_without_task_label():
    baseline = _record("llada-8b-instruct-hf", "sym", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "sym", "family": "symbolic", "answer_type": "short_text"}
    repair = {
        **_record("llada-8b-instruct-hf", "sym", "counterfactual", task_score=0.0, trajectory_score=0.2),
        "task": {"task_id": "sym", "family": "symbolic", "answer_type": "short_text"},
        "text": "on",
        "schedule": None,
        "repair": {"name": "counterfactual_answer_proposal", "proposal": "on"},
    }

    selected = select_repair_record(
        [baseline, repair],
        baseline_record=baseline,
        task_answer_type="short_text",
        exact_task_trajectory_policy="fixed",
        promotion_margin=0.02,
    )

    assert selected["repair"]["name"] == "counterfactual_answer_proposal"
    assert _exact_answer_repair_selection_score(repair, "short_text") > 1.0


def test_exact_answer_repair_selection_rejects_unmatched_proposal():
    baseline = _record("llada-8b-instruct-hf", "sym", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "sym", "family": "symbolic", "answer_type": "short_text"}
    repair = {
        **_record("llada-8b-instruct-hf", "sym", "counterfactual", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "sym", "family": "symbolic", "answer_type": "short_text"},
        "text": "off",
        "schedule": None,
        "repair": {"name": "counterfactual_answer_proposal", "proposal": "on"},
    }

    selected = select_repair_record(
        [baseline, repair],
        baseline_record=baseline,
        task_answer_type="short_text",
        exact_task_trajectory_policy="fixed",
        promotion_margin=0.02,
    )

    assert selected is baseline


def test_exact_answer_self_repair_promotes_changed_parseable_answer():
    baseline = _record("llada-8b-instruct-hf", "math", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "math", "family": "math", "answer_type": "integer"}
    baseline["task_score"]["extracted_answer"] = 40
    repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.2),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "20 + 21 = 41. Answer: 41",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 40,
            "self_repair_extracted_answer": "41",
        },
    }

    selected = select_repair_record(
        [baseline, repair],
        baseline_record=baseline,
        task_answer_type="integer",
        exact_task_trajectory_policy="fixed",
        promotion_margin=0.02,
    )

    assert selected["repair"]["name"] == "self_check_answer_repair"


def test_exact_answer_self_repair_rejects_integer_without_arithmetic_evidence():
    baseline = _record("llada-8b-instruct-hf", "math", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "math", "family": "math", "answer_type": "integer"}
    baseline["task_score"]["extracted_answer"] = 40
    repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "After checking again, Answer: 41",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 40,
            "self_repair_extracted_answer": "41",
        },
    }

    selected = select_repair_record(
        [baseline, repair],
        baseline_record=baseline,
        task_answer_type="integer",
        exact_task_trajectory_policy="fixed",
        promotion_margin=0.02,
    )

    assert selected is baseline
    assert _exact_answer_repair_selection_score(repair, "integer") == 0.0


def test_exact_answer_self_repair_rejects_unchanged_parseable_answer():
    baseline = _record("llada-8b-instruct-hf", "math", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "math", "family": "math", "answer_type": "integer"}
    baseline["task_score"]["extracted_answer"] = 40
    repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "40",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 40,
            "self_repair_extracted_answer": "40",
        },
    }

    selected = select_repair_record(
        [baseline, repair],
        baseline_record=baseline,
        task_answer_type="integer",
        exact_task_trajectory_policy="fixed",
        promotion_margin=0.02,
    )

    assert selected is baseline


def test_exact_answer_self_repair_rejects_invalid_scratchpad_arithmetic():
    baseline = _record("llada-8b-instruct-hf", "math", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "math", "family": "math", "answer_type": "integer"}
    baseline["task_score"]["extracted_answer"] = 8
    repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "3*14 + 2*9 = 54. Answer: 12",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 8,
            "self_repair_extracted_answer": "12",
            "self_repair_arithmetic_consistent": False,
        },
    }

    selected = select_repair_record(
        [baseline, repair],
        baseline_record=baseline,
        task_answer_type="integer",
        exact_task_trajectory_policy="fixed",
        promotion_margin=0.02,
    )

    assert selected is baseline
    assert not _arithmetic_claims_consistent("3*14 + 2*9 = 54. Answer: 12")
    assert _arithmetic_claims_consistent("9*3 + 1 = 28. 28 / 2 = 14. Answer: 14")
    assert _arithmetic_claim_inconsistencies("3*14 + 2*9 = 54. Answer: 12") == [
        {"expression": "3*14 + 2*9", "claimed": 54.0, "computed": 60.0}
    ]
    assert _arithmetic_claim_inconsistencies(
        "Three talks and two breaks: 3 times 14 plus 2 times 9 is 54. Answer: 12"
    ) == [{"expression": "3 * 14 + 2 * 9", "claimed": 54.0, "computed": 60.0}]
    assert _arithmetic_claim_count("3 times 14 plus 2 times 9 is 54. Answer: 12") == 1
    assert _arithmetic_claims_consistent("45 \u00f7 5 = 9. Answer: 9")
    assert _arithmetic_claims_consistent(
        "Remaining time: 90 minus 60 is 30. Each session: 30 divided by 3 is 10."
    )


def test_exact_answer_arithmetic_feedback_repair_promotes_consistent_changed_answer():
    baseline = _record("llada-8b-instruct-hf", "math", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "math", "family": "math", "answer_type": "integer"}
    baseline["task_score"]["extracted_answer"] = 8
    feedback = {
        **_record("llada-8b-instruct-hf", "math", "feedback", task_score=0.0, trajectory_score=0.7),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "3*14 + 2*9 = 60. 90 - 60 = 30. 30 / 3 = 10. Answer: 10",
        "schedule": None,
        "repair": {
            "name": "arithmetic_feedback_repair",
            "source_extracted_answer": 8,
            "self_repair_extracted_answer": "10",
            "self_repair_arithmetic_consistent": True,
        },
    }

    selected = select_repair_record(
        [baseline, feedback],
        baseline_record=baseline,
        task_answer_type="integer",
        exact_task_trajectory_policy="fixed",
        promotion_margin=0.02,
    )

    assert selected["repair"]["name"] == "arithmetic_feedback_repair"


def test_exact_answer_repair_prefers_verifier_span_over_feedback_when_both_pass():
    baseline = _record("llada-8b-instruct-hf", "math", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "math", "family": "math", "answer_type": "integer"}
    baseline["task_score"]["extracted_answer"] = 12
    span = {
        **_record("llada-8b-instruct-hf", "math", "span", task_score=0.0, trajectory_score=0.1),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "3*14 + 2*9 = 60. 90 - 60 = 30. 30 / 3 = 10. Answer: 10",
        "schedule": None,
        "repair": {
            "name": "arithmetic_contradiction_span_repair",
            "source_extracted_answer": 12,
            "self_repair_extracted_answer": "10",
            "self_repair_arithmetic_consistent": True,
        },
    }
    feedback = {
        **_record("llada-8b-instruct-hf", "math", "feedback", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "3*14 + 2*9 = 60. 90 - 60 = 30. 30 / 3 = 10. Answer: 10",
        "schedule": None,
        "repair": {
            "name": "arithmetic_feedback_repair",
            "source_extracted_answer": 12,
            "self_repair_extracted_answer": "10",
            "self_repair_arithmetic_consistent": True,
        },
    }

    selected = select_repair_record(
        [baseline, feedback, span],
        baseline_record=baseline,
        task_answer_type="integer",
        exact_task_trajectory_policy="fixed",
        promotion_margin=0.02,
    )

    assert selected["repair"]["name"] == "arithmetic_contradiction_span_repair"
    assert _exact_answer_repair_selection_score(span, "integer") > _exact_answer_repair_selection_score(
        feedback,
        "integer",
    )


def test_exact_answer_arithmetic_evidence_repair_promotes_consistent_changed_answer():
    baseline = _record("llada-8b-instruct-hf", "math", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "math", "family": "math", "answer_type": "integer"}
    baseline["task_score"]["extracted_answer"] = 40
    evidence = {
        **_record("llada-8b-instruct-hf", "math", "evidence", task_score=0.0, trajectory_score=0.7),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "20 + 21 = 41. Answer: 41",
        "schedule": None,
        "repair": {
            "name": "arithmetic_evidence_repair",
            "source_extracted_answer": 40,
            "self_repair_extracted_answer": "41",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 1,
        },
    }

    selected = select_repair_record(
        [baseline, evidence],
        baseline_record=baseline,
        task_answer_type="integer",
        exact_task_trajectory_policy="fixed",
        promotion_margin=0.02,
    )

    assert selected["repair"]["name"] == "arithmetic_evidence_repair"


def test_exact_answer_repair_rejects_irrelevant_prompt_number_in_equation():
    prompt = (
        "Nia is packing 48 apples. A note on the table says there are 9 oranges, "
        "but the oranges are not being packed. She sets aside 3 damaged apples, "
        "then puts the remaining apples into bags with 5 apples per bag. "
        "How many full apple bags can she make? Answer with one integer."
    )
    baseline = _record("llada-8b-instruct-hf", "math", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "math", "family": "math", "answer_type": "integer"}
    baseline["task_score"]["extracted_answer"] = 8
    repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "48 - 3 + 9 = 54. 54 / 5 = 10.8. Answer: 10",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 8,
            "self_repair_extracted_answer": "10",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 2,
        },
    }

    selected = select_repair_record(
        [baseline, repair],
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="integer",
        exact_task_trajectory_policy="fixed",
        promotion_margin=0.02,
    )

    assert selected is baseline
    assert _prompt_irrelevant_numbers(prompt) == {"9"}
    assert _exact_answer_repair_selection_score(repair, "integer", prompt) == 0.0


def test_prompt_required_arithmetic_operators_for_gsm_distractors():
    prompts = {
        "math_012": (
            "Nia is packing 48 apples. A note on the table says there are 9 oranges, "
            "but the oranges are not being packed. She sets aside 3 damaged apples, "
            "then puts the remaining apples into bags with 5 apples per bag. "
            "How many full apple bags can she make? Answer with one integer."
        ),
        "math_013": (
            "A club sold 17 adult tickets for 12 dollars each and 8 child tickets for "
            "7 dollars each. A sponsor also donated 25 dollars, but that donation is "
            "not ticket revenue. How many dollars came from ticket sales? Answer with one integer."
        ),
        "math_014": (
            "Ravi read 18 pages on Monday. On Tuesday he read twice as many pages as Monday. "
            "On Wednesday he read 7 fewer pages than Tuesday. The book has 200 pages, but only "
            "count the pages read across those three days. How many pages did Ravi read? "
            "Answer with one integer."
        ),
        "math_015": (
            "There are 6 trays with 14 cookies on each tray. A label mentions 9 chocolate-chip "
            "cookies, but the question asks about all cookies. After 20 cookies are eaten, the "
            "remaining cookies are shared equally by 8 students. How many cookies does each "
            "student get? Answer with one integer."
        ),
    }

    assert _prompt_required_arithmetic_operators(prompts["math_012"]) == {"-", "/"}
    assert _prompt_required_arithmetic_operators(prompts["math_013"]) == {"*", "+"}
    assert _prompt_required_arithmetic_operators(prompts["math_014"]) == {"*", "-", "+"}
    assert _prompt_required_arithmetic_operators(prompts["math_015"]) == {"*", "-", "/"}


def test_exact_answer_repair_rejects_missing_required_prompt_operation():
    prompt = (
        "There are 6 trays with 14 cookies on each tray. After 20 cookies are eaten, "
        "the remaining cookies are shared equally by 8 students. How many cookies does "
        "each student get? Answer with one integer."
    )
    baseline = _record("llada-8b-instruct-hf", "math", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "math", "family": "math", "answer_type": "integer"}
    baseline["task_score"]["extracted_answer"] = 7
    repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "6 * 14 = 84. 84 - 20 = 64. Answer: 64",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 7,
            "self_repair_extracted_answer": "64",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 2,
        },
    }

    selected = select_repair_record(
        [baseline, repair],
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="integer",
        exact_task_trajectory_policy="fixed",
        promotion_margin=0.02,
    )

    assert selected is baseline
    assert _exact_answer_repair_selection_score(repair, "integer", prompt) == 0.0


def test_prompt_quantity_role_requirements_for_gsm_distractors():
    ticket_prompt = (
        "A club sold 17 adult tickets for 12 dollars each and 8 child tickets for "
        "7 dollars each. A sponsor also donated 25 dollars, but that donation is "
        "not ticket revenue. How many dollars came from ticket sales? Answer with one integer."
    )
    cookie_prompt = (
        "There are 6 trays with 14 cookies on each tray. A label mentions 9 chocolate-chip "
        "cookies, but the question asks about all cookies. After 20 cookies are eaten, the "
        "remaining cookies are shared equally by 8 students. How many cookies does each "
        "student get? Answer with one integer."
    )
    pages_prompt = (
        "Ravi read 18 pages on Monday. On Tuesday he read twice as many pages as Monday. "
        "On Wednesday he read 7 fewer pages than Tuesday. The book has 200 pages, but only "
        "count the pages read across those three days. How many pages did Ravi read? "
        "Answer with one integer."
    )

    ticket_roles = _prompt_quantity_role_requirements(ticket_prompt)
    cookie_roles = _prompt_quantity_role_requirements(cookie_prompt)
    pages_roles = _prompt_quantity_role_requirements(pages_prompt)

    assert ticket_roles["multiply_pairs"] == {("12", "17"), ("7", "8")}
    assert cookie_roles["multiply_pairs"] == {("6", "14")}
    assert cookie_roles["subtraction_right_values"] == {"20"}
    assert cookie_roles["division_right_values"] == {"8"}
    assert pages_roles["multiply_pairs"] == {("2", "18")}
    assert pages_roles["subtraction_right_values"] == {"7"}


def test_exact_answer_repair_rejects_wrong_quantity_role_binding():
    prompt = (
        "A club sold 17 adult tickets for 12 dollars each and 8 child tickets for "
        "7 dollars each. A sponsor also donated 25 dollars, but that donation is "
        "not ticket revenue. How many dollars came from ticket sales? Answer with one integer."
    )
    baseline = _record("llada-8b-instruct-hf", "math", "baseline", task_score=0.0, trajectory_score=0.4)
    baseline["task"] = {"task_id": "math", "family": "math", "answer_type": "integer"}
    baseline["task_score"]["extracted_answer"] = 260
    wrong_binding = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "17 * 7 = 119. 8 * 12 = 96. 119 + 96 = 215. Answer: 215",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 260,
            "self_repair_extracted_answer": "215",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 3,
        },
    }
    correct_binding = {
        **_record("llada-8b-instruct-hf", "math", "feedback", task_score=0.0, trajectory_score=0.7),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "17 * 12 = 204. 8 * 7 = 56. 204 + 56 = 260. Answer: 260",
        "schedule": None,
        "repair": {
            "name": "arithmetic_feedback_repair",
            "source_extracted_answer": 215,
            "self_repair_extracted_answer": "260",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 3,
        },
    }

    assert _repair_quantity_role_gaps(wrong_binding, prompt) == {"mul:12*17", "mul:7*8"}
    assert _exact_answer_repair_selection_score(wrong_binding, "integer", prompt) == 0.0
    assert _repair_quantity_role_gaps(correct_binding, prompt) == set()
    assert _exact_answer_repair_selection_score(correct_binding, "integer", prompt) > 0.0


def test_exact_answer_repair_rejects_wrong_equal_share_divisor():
    prompt = (
        "There are 6 trays with 14 cookies on each tray. After 20 cookies are eaten, "
        "the remaining cookies are shared equally by 8 students. How many cookies does "
        "each student get? Answer with one integer."
    )
    repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "6 * 14 = 84. 84 - 20 = 64. 64 / 4 = 16. Answer: 16",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 8,
            "self_repair_extracted_answer": "16",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 3,
        },
    }

    assert _prompt_required_arithmetic_operators(prompt) == {"*", "-", "/"}
    assert _repair_quantity_role_gaps(repair, prompt) == {"div:8"}
    assert _exact_answer_repair_selection_score(repair, "integer", prompt) == 0.0


def test_exact_answer_repair_rejects_ungrounded_arithmetic_intermediate():
    prompt = (
        "A club sold 17 adult tickets for 12 dollars each and 8 child tickets for "
        "7 dollars each. How many dollars came from ticket sales? Answer with one integer."
    )
    repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "17 * 12 = 204. 8 * 7 = 56. 999 + 56 = 1055. Answer: 1055",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 260,
            "self_repair_extracted_answer": "1055",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 3,
        },
    }

    assert _repair_quantity_role_gaps(repair, prompt) == set()
    assert _repair_arithmetic_provenance_gaps(repair, prompt) == {"999 + 56:999"}
    assert _exact_answer_repair_selection_score(repair, "integer", prompt) == 0.0


def test_arithmetic_provenance_allows_verified_intermediates_and_word_constants():
    prompt = (
        "A code starts at 9. If the current number is odd, triple it and add 1; "
        "if it is even, halve it. Apply this rule twice. What number results? "
        "Answer with one integer."
    )
    repair = {
        **_record("llada-8b-instruct-hf", "sym", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "sym", "family": "symbolic", "answer_type": "integer"},
        "text": "9 * 3 + 1 = 28. 28 / 2 = 14. Answer: 14",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 10,
            "self_repair_extracted_answer": "14",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 2,
        },
    }

    assert _repair_arithmetic_provenance_gaps(repair, prompt) == set()
    assert _exact_answer_repair_selection_score(repair, "integer", prompt) > 0.0


def test_exact_answer_repair_rejects_final_answer_that_is_not_prompted_total():
    prompt = (
        "A club sold 17 adult tickets for 12 dollars each and 8 child tickets for "
        "7 dollars each. How many dollars came from ticket sales? Answer with one integer."
    )
    repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "17 * 12 = 204. 8 * 7 = 56. 204 + 56 = 260. Answer: 204",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 260,
            "self_repair_extracted_answer": "204",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 3,
        },
    }

    assert _repair_final_answer_role_gaps(repair, prompt) == {"sum:final_not_sum"}
    assert _exact_answer_repair_selection_score(repair, "integer", prompt) == 0.0


def test_final_answer_role_accepts_prompted_division_and_floor_division():
    each_prompt = (
        "There are 6 trays with 14 cookies on each tray. After 20 cookies are eaten, "
        "the remaining cookies are shared equally by 8 students. How many cookies does "
        "each student get? Answer with one integer."
    )
    full_bags_prompt = (
        "Nia is packing 48 apples. She sets aside 3 damaged apples, then puts the "
        "remaining apples into bags with 5 apples per bag. How many full apple bags "
        "can she make? Answer with one integer."
    )
    each_repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "6 * 14 = 84. 84 - 20 = 64. 64 / 8 = 8. Answer: 8",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 7,
            "self_repair_extracted_answer": "8",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 3,
        },
    }
    wrong_full_bag_repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "48 - 3 = 45. 45 / 5 = 9. Answer: 45",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 8,
            "self_repair_extracted_answer": "45",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 2,
        },
    }

    assert _repair_final_answer_role_gaps(each_repair, each_prompt) == set()
    assert _exact_answer_repair_selection_score(each_repair, "integer", each_prompt) > 0.0
    assert _repair_final_answer_role_gaps(wrong_full_bag_repair, full_bags_prompt) == {
        "floor_division:final_not_floor_division"
    }
    assert _exact_answer_repair_selection_score(wrong_full_bag_repair, "integer", full_bags_prompt) == 0.0


def test_exact_answer_repair_rejects_quotient_for_leftover_prompt():
    prompt = (
        "A store sells notebooks in packs of 6. Mara buys 4 packs, gives 7 notebooks "
        "away, then splits the rest equally among 5 bins. How many notebooks are left "
        "over after making equal bins? Answer with one integer."
    )
    repair = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "4 * 6 = 24. 24 - 7 = 17. 17 / 5 = 3.4. Answer: 3",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 4,
            "self_repair_extracted_answer": "3",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 3,
        },
    }

    assert _repair_final_answer_role_gaps(repair, prompt) == {"remainder:final_not_remainder"}
    assert _exact_answer_repair_selection_score(repair, "integer", prompt) == 0.0


def test_prompt_excluded_final_answer_terms_for_distractor_objects():
    apple_prompt = (
        "Nia is packing 48 apples. A note on the table says there are 9 oranges, "
        "but the oranges are not being packed. She sets aside 3 damaged apples, "
        "then puts the remaining apples into bags with 5 apples per bag. "
        "How many full apple bags can she make? Answer with one integer."
    )
    ticket_prompt = (
        "A club sold 17 adult tickets for 12 dollars each and 8 child tickets for "
        "7 dollars each. A sponsor also donated 25 dollars, but that donation is "
        "not ticket revenue. How many dollars came from ticket sales? Answer with one integer."
    )
    cookie_prompt = (
        "There are 6 trays with 14 cookies on each tray. A label mentions 9 chocolate-chip "
        "cookies, but the question asks about all cookies. After 20 cookies are eaten, the "
        "remaining cookies are shared equally by 8 students. How many cookies does each "
        "student get? Answer with one integer."
    )

    assert _prompt_excluded_final_answer_terms(apple_prompt) == {"orange", "oranges"}
    assert _prompt_excluded_final_answer_terms(ticket_prompt) == {"donation", "donations", "sponsor", "sponsors"}
    assert _prompt_excluded_final_answer_terms(cookie_prompt) == {"chocolate chip", "chocolate chips"}


def test_exact_answer_repair_rejects_final_answer_named_excluded_object():
    prompt = (
        "Nia is packing 48 apples. A note on the table says there are 9 oranges, "
        "but the oranges are not being packed. She sets aside 3 damaged apples, "
        "then puts the remaining apples into bags with 5 apples per bag. "
        "How many full apple bags can she make? Answer with one integer."
    )
    wrong_object = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "48 - 3 = 45. 45 / 5 = 9. Answer: 9 orange bags",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 8,
            "self_repair_extracted_answer": "9",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 2,
        },
    }
    correct_object = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "48 - 3 = 45. 45 / 5 = 9. Answer: 9 apple bags, not orange bags",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 8,
            "self_repair_extracted_answer": "9",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 2,
        },
    }

    assert _repair_final_answer_object_gaps(wrong_object, prompt) == {"excluded:orange"}
    assert _exact_answer_repair_selection_score(wrong_object, "integer", prompt) == 0.0
    assert _repair_final_answer_object_gaps(correct_object, prompt) == set()
    assert _exact_answer_repair_selection_score(correct_object, "integer", prompt) > 0.0


def test_prompt_final_answer_target_spec_for_gsm_units():
    apple_prompt = (
        "Nia is packing 48 apples. She sets aside 3 damaged apples, then puts the "
        "remaining apples into bags with 5 apples per bag. How many full apple bags "
        "can she make? Answer with one integer."
    )
    cookie_prompt = (
        "There are 6 trays with 14 cookies on each tray. After 20 cookies are eaten, "
        "the remaining cookies are shared equally by 8 students. How many cookies does "
        "each student get? Answer with one integer."
    )
    capacity_prompt = (
        "A tank is 3/5 full. After adding 18 liters, it is 4/5 full. "
        "What is the full capacity of the tank in liters? Answer with one integer."
    )

    assert _prompt_final_answer_target_spec(apple_prompt)["heads"] == {"bag", "bags"}
    assert _prompt_final_answer_target_spec(apple_prompt)["modifiers"] >= {"apple"}
    assert _prompt_final_answer_target_spec(cookie_prompt)["heads"] == {"cookie", "cookies"}
    assert _prompt_final_answer_target_spec(capacity_prompt)["heads"] == {"liter", "liters"}


def test_exact_answer_repair_rejects_wrong_final_target_unit():
    prompt = (
        "There are 6 trays with 14 cookies on each tray. After 20 cookies are eaten, "
        "the remaining cookies are shared equally by 8 students. How many cookies does "
        "each student get? Answer with one integer."
    )
    wrong_target = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "6 * 14 = 84. 84 - 20 = 64. 64 / 8 = 8. Answer: 8 students",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 7,
            "self_repair_extracted_answer": "8",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 3,
        },
    }
    correct_target = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "6 * 14 = 84. 84 - 20 = 64. 64 / 8 = 8. Answer: 8 cookies",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 7,
            "self_repair_extracted_answer": "8",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 3,
        },
    }

    assert _repair_final_answer_target_gaps(wrong_target, prompt) == {"wrong_target:students"}
    assert _exact_answer_repair_selection_score(wrong_target, "integer", prompt) == 0.0
    assert _repair_final_answer_target_gaps(correct_target, prompt) == set()
    assert _exact_answer_repair_selection_score(correct_target, "integer", prompt) > 0.0


def test_exact_answer_repair_rejects_conflicting_final_target_modifier():
    prompt = (
        "Nia is packing 48 apples. She sets aside 3 damaged apples, then puts the "
        "remaining apples into bags with 5 apples per bag. How many full apple bags "
        "can she make? Answer with one integer."
    )
    wrong_modifier = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "48 - 3 = 45. 45 / 5 = 9. Answer: 9 pear bags",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 8,
            "self_repair_extracted_answer": "9",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 2,
        },
    }
    bare_unit = {
        **_record("llada-8b-instruct-hf", "math", "self", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "math", "family": "math", "answer_type": "integer"},
        "text": "48 - 3 = 45. 45 / 5 = 9. Answer: 9 bags",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": 8,
            "self_repair_extracted_answer": "9",
            "self_repair_arithmetic_consistent": True,
            "self_repair_arithmetic_claim_count": 2,
        },
    }

    assert _repair_final_answer_target_gaps(wrong_modifier, prompt) == {"conflicting_modifier:pear_bags"}
    assert _exact_answer_repair_selection_score(wrong_modifier, "integer", prompt) == 0.0
    assert _repair_final_answer_target_gaps(bare_unit, prompt) == set()
    assert _exact_answer_repair_selection_score(bare_unit, "integer", prompt) > 0.0


def test_exact_self_repair_spends_arithmetic_evidence_branch_when_claims_missing(tmp_path):
    task = GeneralReasoningTask(
        task_id="math_missing_evidence",
        family="math",
        prompt=(
            "A small puzzle has 21 useful tokens and then 20 more useful tokens. "
            "How many useful tokens are there? Answer with one integer."
        ),
        answer_type="integer",
        scorer="exact_integer",
        answer=41,
        max_new_tokens=16,
    )
    source = _record(
        "llada-8b-instruct-hf",
        "math_missing_evidence",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["task"] = {"task_id": task.task_id, "family": "math", "answer_type": "integer"}
    source["task_score"]["extracted_answer"] = 40
    backend = _FakeExactRepairBackend(
        [
            "Answer: 41",
            "21 + 20 = 41. Answer: 41",
        ]
    )

    records = _generate_exact_answer_repair_records(
        backend,
        task,
        source_record=source,
        limit=2,
        exact_self_repair=True,
        generation_seed_base=7,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
    )

    assert [record["repair"]["name"] for record in records] == [
        "self_check_answer_repair",
        "arithmetic_evidence_repair",
    ]
    assert "did not show checkable arithmetic" in backend.prompts[1]
    assert records[0]["repair"]["self_repair_arithmetic_claim_count"] == 0
    assert records[1]["repair"]["self_repair_arithmetic_claim_count"] == 1


def test_exact_verifier_revision_repairs_arithmetic_contradiction_span(tmp_path):
    task = GeneralReasoningTask(
        task_id="math_bad_claim",
        family="math",
        prompt=(
            "A puzzle has 3 groups of 14 and 2 groups of 9. "
            "The total is shared by 6 teams. How many per team? "
            "Answer with one integer."
        ),
        answer_type="integer",
        scorer="exact_integer",
        answer=10,
        max_new_tokens=6,
    )
    source = _record(
        "llada-8b-instruct-hf",
        "math_bad_claim",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["task"] = {"task_id": task.task_id, "family": "math", "answer_type": "integer"}
    source["task_score"]["extracted_answer"] = 12
    backend = _FakeExactRepairBackend(
        [
            "3*14 + 2*9 = 54. 54 / 6 = 9. Answer: 9",
            "3*14 = 42. 2*9 = 18. 42 + 18 = 60. 60 / 6 = 10. Answer: 10",
        ],
        tokenizer=_TokenPiecesTokenizer(
            {
                10: "3*14 + ",
                11: "2*9 = ",
                12: "54",
                13: ". ",
                14: "54 / 6 = 9. ",
                15: "Answer: 9",
            }
        ),
        token_id_outputs=[
            [10, 11, 12, 13, 14, 15],
            [],
        ],
    )

    records = _generate_exact_answer_repair_records(
        backend,
        task,
        source_record=source,
        limit=2,
        exact_self_repair=True,
        generation_seed_base=7,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
        exact_verifier_revision=True,
    )

    assert [record["repair"]["name"] for record in records] == [
        "self_check_answer_repair",
        "arithmetic_contradiction_span_repair",
    ]
    assert backend.configs[1].initial_suffix_token_ids == (None, None, None, None, None, None)
    assert records[1]["repair"]["uses_arithmetic_span_revision"] is True
    assert records[1]["repair"]["arithmetic_span_targets"] == [
        "3*14 + 2*9",
        "3*14 + 2*9 = 54",
        "54 / 6",
        "54 / 6 = 9",
        "Answer: 9",
    ]
    assert records[1]["repair"]["seed_masked_positions"] == 6
    assert _exact_answer_repair_selection_score(records[1], "integer", task.prompt) > 0.0


def test_exact_verifier_revision_skips_feedback_after_successful_arithmetic_span(tmp_path):
    task = GeneralReasoningTask(
        task_id="math_bad_claim_budget",
        family="math",
        prompt=(
            "A puzzle has 3 groups of 14 and 2 groups of 9. "
            "The total is shared by 6 teams. How many per team? "
            "Answer with one integer."
        ),
        answer_type="integer",
        scorer="exact_integer",
        answer=10,
        max_new_tokens=6,
    )
    source = _record(
        "llada-8b-instruct-hf",
        "math_bad_claim_budget",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["task"] = {"task_id": task.task_id, "family": "math", "answer_type": "integer"}
    source["task_score"]["extracted_answer"] = 12
    backend = _FakeExactRepairBackend(
        [
            "3*14 + 2*9 = 54. 54 / 6 = 9. Answer: 9",
            "3*14 = 42. 2*9 = 18. 42 + 18 = 60. 60 / 6 = 10. Answer: 10",
            "This feedback branch should not run. Answer: 10",
        ],
        tokenizer=_TokenPiecesTokenizer(
            {
                10: "3*14 + ",
                11: "2*9 = ",
                12: "54",
                13: ". ",
                14: "54 / 6 = 9. ",
                15: "Answer: 9",
            }
        ),
        token_id_outputs=[
            [10, 11, 12, 13, 14, 15],
            [],
            [],
        ],
    )

    records = _generate_exact_answer_repair_records(
        backend,
        task,
        source_record=source,
        limit=3,
        exact_self_repair=True,
        generation_seed_base=7,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
        exact_verifier_revision=True,
    )

    assert [record["repair"]["name"] for record in records] == [
        "self_check_answer_repair",
        "arithmetic_contradiction_span_repair",
    ]
    assert backend.outputs == ["This feedback branch should not run. Answer: 10"]


def test_arithmetic_inconsistency_span_targets_include_downstream_claims():
    inconsistencies = [
        {"expression": "3*14 + 2*9", "claimed": 54.0, "computed": 60.0},
    ]

    targets = _arithmetic_inconsistency_span_targets(
        inconsistencies,
        (
            "Total time: 3*14 + 2*9 = 54. "
            "Remaining: 90 - 54 = 36. "
            "Each: 36 / 3 = 12. Answer: 12"
        ),
    )

    assert targets == [
        "3*14 + 2*9",
        "3*14 + 2*9 = 54",
        "90 - 54",
        "90 - 54 = 36",
        "36 / 3",
        "36 / 3 = 12",
        "Answer: 12",
    ]


def test_exact_answer_repairs_run_for_failed_llada_symbolic_task():
    task = GeneralReasoningTask(
        task_id="sym",
        family="symbolic",
        prompt="A lamp starts off. It is toggled 5 times. Is it on or off at the end? Answer only on or off.",
        answer_type="short_text",
        scorer="exact_short_text",
        answer="on",
        max_new_tokens=16,
    )
    source = _record("llada-8b-instruct-hf", "sym", "baseline", task_score=0.0, trajectory_score=0.4)
    source["task_score"]["extracted_answer"] = "off"

    assert _should_run_exact_answer_repairs("LLaDA 8B", task, 2, source)


def test_exact_verifier_revision_masks_source_answer_span_before_counterfactual_prompt(tmp_path):
    task = GeneralReasoningTask(
        task_id="sym",
        family="symbolic",
        prompt="A lamp starts off. It is toggled 5 times. Is it on or off at the end? Answer only on or off.",
        answer_type="short_text",
        scorer="exact_short_text",
        answer="on",
        max_new_tokens=4,
    )
    source = _record("llada-8b-instruct-hf", "sym", "baseline", task_score=0.0, trajectory_score=0.4)
    source["task"] = {"task_id": task.task_id, "family": "symbolic", "answer_type": "short_text"}
    source["text"] = "Scratch. Answer: off."
    source["generated_token_ids"] = [10, 11, 12, 13]
    source["task_score"]["extracted_answer"] = "off"
    backend = _FakeExactRepairBackend(
        ["Scratch. Answer: on.", "Answer: on"],
        tokenizer=_TokenPiecesTokenizer(
            {
                10: "Scratch. ",
                11: "Answer: ",
                12: "off",
                13: ".",
            }
        ),
    )

    records = _generate_exact_answer_repair_records(
        backend,
        task,
        source_record=source,
        limit=2,
        exact_self_repair=False,
        exact_verifier_revision=True,
        generation_seed_base=17,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
    )

    assert [record["repair"]["name"] for record in records] == [
        "answer_span_repair",
        "counterfactual_answer_proposal",
    ]
    assert backend.prompts[0] == task.prompt
    assert "Evaluate the alternative candidate" in backend.prompts[1]
    assert backend.configs[0].initial_suffix_token_ids == (10, 11, None, 13)
    assert records[0]["repair"]["proposal"] == "on"
    assert records[0]["repair"]["uses_verifier_answer_span_revision"] is True
    assert _exact_answer_repair_selection_score(records[0], "short_text", task.prompt) > 1.0


def test_label_free_short_text_answer_extraction_for_constrained_prompts():
    toggle_task = GeneralReasoningTask(
        task_id="toggle",
        family="symbolic",
        prompt="A lamp starts off. It is toggled 5 times. Is it on or off at the end? Answer only on or off.",
        answer_type="short_text",
        scorer="exact_short_text",
        answer="on",
        max_new_tokens=16,
    )
    order_task = GeneralReasoningTask(
        task_id="order",
        family="symbolic",
        prompt="If D is before A, A is before B, and B is before C, what is the full order from first to last? Answer with the four letters separated by spaces.",
        answer_type="short_text",
        scorer="exact_short_text",
        answer="D A B C",
        max_new_tokens=16,
    )
    list_task = GeneralReasoningTask(
        task_id="list",
        family="symbolic",
        prompt="Start with the list red, blue, green. What is the final list?",
        answer_type="short_text",
        scorer="exact_short_text",
        answer="red blue green",
        max_new_tokens=16,
    )

    assert _short_text_answer_schema(toggle_task.prompt) == {"kind": "choice", "choices": ["on", "off"]}
    assert _label_free_exact_answer_supported(toggle_task)
    assert _label_free_exact_answer_from_text(toggle_task, "Scratch. Answer: on") == "on"
    assert _label_free_exact_answer_from_text(order_task, "Final answer: D A B C") == "D A B C"
    assert _label_free_exact_answer_from_text(order_task, "Final answer: D A B B") is None
    assert _label_free_exact_answer_from_text(list_task, "Final list: green red blue") == "green red blue"


def test_exact_self_repair_runs_for_constrained_short_text_without_proposals(tmp_path):
    task = GeneralReasoningTask(
        task_id="letters_no_solver",
        family="symbolic",
        prompt=(
            "A display code should be X Y Z. What is the display code? "
            "Answer with the three letters separated by spaces."
        ),
        answer_type="short_text",
        scorer="exact_short_text",
        answer="X Y Z",
        max_new_tokens=16,
    )
    source = _record("llada-8b-instruct-hf", "letters_no_solver", "baseline", task_score=0.0, trajectory_score=0.4)
    source["task"] = {"task_id": task.task_id, "family": "symbolic", "answer_type": "short_text"}
    source["task_score"]["extracted_answer"] = "X Z Y"
    backend = _FakeExactRepairBackend(["Reasoning. Answer: X Y Z"])

    records = _generate_exact_answer_repair_records(
        backend,
        task,
        source_record=source,
        limit=2,
        exact_self_repair=True,
        generation_seed_base=13,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
    )

    assert [record["repair"]["name"] for record in records] == ["self_check_answer_repair"]
    assert records[0]["repair"]["self_repair_extracted_answer"] == "X Y Z"
    assert _exact_answer_repair_selection_score(records[0], "short_text", task.prompt) > 0.0


def test_exact_verifier_revision_can_remask_answer_span_without_prompt_proposal(tmp_path):
    task = GeneralReasoningTask(
        task_id="letters_no_solver_span",
        family="symbolic",
        prompt=(
            "A display code should be X Y Z. What is the display code? "
            "Answer with the three letters separated by spaces."
        ),
        answer_type="short_text",
        scorer="exact_short_text",
        answer="X Y Z",
        max_new_tokens=5,
    )
    source = _record(
        "llada-8b-instruct-hf",
        "letters_no_solver_span",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["task"] = {"task_id": task.task_id, "family": "symbolic", "answer_type": "short_text"}
    source["text"] = "Scratch. Answer: X Z Y"
    source["generated_token_ids"] = [10, 11, 12, 13, 14]
    source["task_score"]["extracted_answer"] = "X Z Y"
    backend = _FakeExactRepairBackend(
        [
            "Scratch. Answer: X Y Z",
            "Reasoning. Answer: X Y Z",
        ],
        tokenizer=_TokenPiecesTokenizer(
            {
                10: "Scratch. ",
                11: "Answer: ",
                12: "X ",
                13: "Z ",
                14: "Y",
            }
        ),
    )

    records = _generate_exact_answer_repair_records(
        backend,
        task,
        source_record=source,
        limit=2,
        exact_self_repair=True,
        exact_verifier_revision=True,
        generation_seed_base=19,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
    )

    assert [record["repair"]["name"] for record in records] == [
        "answer_span_repair",
        "self_check_answer_repair",
    ]
    assert backend.configs[0].initial_suffix_token_ids == (10, 11, None, None, None)
    assert records[0]["repair"]["proposal"] is None
    assert records[0]["repair"]["uses_label_free_verifier_span_revision"] is True
    assert records[0]["repair"]["self_repair_extracted_answer"] == "X Y Z"
    assert _exact_answer_repair_selection_score(records[0], "short_text", task.prompt) > 0.0


def test_exact_self_repair_metadata_normalizes_short_text_case_and_skips_arithmetic(tmp_path):
    task = GeneralReasoningTask(
        task_id="letters_no_solver",
        family="symbolic",
        prompt=(
            "A display code should be X Y Z. A note says one spare sticker is left, "
            "but the code is unchanged. What is the display code? "
            "Answer with the three letters separated by spaces."
        ),
        answer_type="short_text",
        scorer="exact_short_text",
        answer="X Y Z",
        max_new_tokens=16,
    )
    source = _record("llada-8b-instruct-hf", "letters_no_solver", "baseline", task_score=0.0, trajectory_score=0.4)
    source["task"] = {"task_id": task.task_id, "family": "symbolic", "answer_type": "short_text"}
    source["task_score"]["extracted_answer"] = "x y z"
    backend = _FakeExactRepairBackend(["Reasoning. Answer: X Y Z"])

    records = _generate_exact_answer_repair_records(
        backend,
        task,
        source_record=source,
        limit=1,
        exact_self_repair=True,
        generation_seed_base=13,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
    )

    repair = records[0]["repair"]
    assert repair["self_repair_changed_answer"] is False
    assert repair["self_repair_required_operators"] == []
    assert repair["self_repair_missing_required_operators"] == []


def test_short_text_symbolic_proof_guard_rejects_wrong_mechanical_answer():
    prompt = (
        "If D is before A, A is before B, and B is before C, "
        "what is the full order from first to last? Answer with the four letters separated by spaces."
    )
    wrong = {
        **_record("llada-8b-instruct-hf", "order", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "order", "family": "symbolic", "answer_type": "short_text"},
        "text": "D is first, then A, then C, then B. Answer: D A C B",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "A B C D",
            "self_repair_extracted_answer": "D A C B",
            "self_repair_arithmetic_consistent": True,
        },
    }
    correct = {
        **_record("llada-8b-instruct-hf", "order", "self", task_score=1.0, trajectory_score=0.8),
        "task": {"task_id": "order", "family": "symbolic", "answer_type": "short_text"},
        "text": "D before A before B before C. Answer: D A B C",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "A B C D",
            "self_repair_extracted_answer": "D A B C",
            "self_repair_arithmetic_consistent": True,
        },
    }

    assert _repair_short_text_symbolic_gaps(wrong, prompt) == {"symbolic:expected_d_a_b_c"}
    assert _exact_answer_repair_selection_score(wrong, "short_text", prompt) == 0.0
    assert _repair_short_text_symbolic_gaps(correct, prompt) == set()
    assert _exact_answer_repair_selection_score(correct, "short_text", prompt) > 0.0


def test_short_text_symbolic_proof_guard_allows_no_solver_schema():
    prompt = (
        "A display code should be X Y Z. What is the display code? "
        "Answer with the three letters separated by spaces."
    )
    repair = {
        **_record("llada-8b-instruct-hf", "letters_no_solver", "self", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "letters_no_solver", "family": "symbolic", "answer_type": "short_text"},
        "text": "Answer: X Y Z",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "X Z Y",
            "self_repair_extracted_answer": "X Y Z",
            "self_repair_arithmetic_consistent": True,
        },
    }

    assert _repair_short_text_symbolic_gaps(repair, prompt) == set()
    assert _exact_answer_repair_selection_score(repair, "short_text", prompt) > 0.0


def test_short_text_symbolic_proof_guard_rejects_wrong_letter_transform():
    prompt = (
        "A display starts with the code K L M. Rotate the code one step left, "
        "then swap the final two letters. What code should be displayed? "
        "Answer with the three letters separated by spaces."
    )
    wrong = {
        **_record("llada-8b-instruct-hf", "letters", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "letters", "family": "symbolic", "answer_type": "short_text"},
        "text": "Rotate left gives M L K, then swap the final two letters. Answer: M L K",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "M L K",
            "self_repair_extracted_answer": "M L K",
            "self_repair_arithmetic_consistent": True,
        },
    }
    correct = {
        **_record("llada-8b-instruct-hf", "letters", "self", task_score=1.0, trajectory_score=0.8),
        "task": {"task_id": "letters", "family": "symbolic", "answer_type": "short_text"},
        "text": "Rotate left gives L M K, then swap the final two letters. Answer: L K M",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "M L K",
            "self_repair_extracted_answer": "L K M",
            "self_repair_arithmetic_consistent": True,
        },
    }

    assert _repair_short_text_symbolic_gaps(wrong, prompt) == {"symbolic:expected_l_k_m"}
    assert _exact_answer_repair_selection_score(wrong, "short_text", prompt) == 0.0
    assert _repair_short_text_symbolic_gaps(correct, prompt) == set()
    assert _exact_answer_repair_selection_score(correct, "short_text", prompt) > 0.0


def test_short_text_symbolic_proof_guard_rejects_wrong_syllogism_answer():
    prompt = "All zargs are blicks. No blicks are morts. Can a zarg be a mort? Answer yes or no."
    wrong = {
        **_record("llada-8b-instruct-hf", "syllogism", "self", task_score=0.0, trajectory_score=0.9),
        "task": {"task_id": "syllogism", "family": "symbolic", "answer_type": "short_text"},
        "text": "A zarg is a blick, and blicks cannot be morts. Answer: yes",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "maybe",
            "self_repair_extracted_answer": "yes",
            "self_repair_arithmetic_consistent": True,
        },
    }
    correct = {
        **_record("llada-8b-instruct-hf", "syllogism", "self", task_score=1.0, trajectory_score=0.8),
        "task": {"task_id": "syllogism", "family": "symbolic", "answer_type": "short_text"},
        "text": "A zarg must be a blick, and no blick is a mort. Answer: no",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "yes",
            "self_repair_extracted_answer": "no",
            "self_repair_arithmetic_consistent": True,
        },
    }

    assert _repair_short_text_symbolic_gaps(wrong, prompt) == {"symbolic:expected_no"}
    assert _exact_answer_repair_selection_score(wrong, "short_text", prompt) == 0.0
    assert _repair_short_text_symbolic_gaps(correct, prompt) == set()
    assert _exact_answer_repair_selection_score(correct, "short_text", prompt) > 0.0


def test_short_text_trace_guard_requires_mechanical_order_trace():
    prompt = (
        "If D is before A, A is before B, and B is before C, "
        "what is the full order from first to last? Answer with the four letters separated by spaces."
    )
    terse = {
        **_record("llada-8b-instruct-hf", "order", "self", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "order", "family": "symbolic", "answer_type": "short_text"},
        "text": "Answer: D A B C",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "A B C D",
            "self_repair_extracted_answer": "D A B C",
            "self_repair_arithmetic_consistent": True,
        },
    }
    traced = {
        **_record("llada-8b-instruct-hf", "order", "self", task_score=1.0, trajectory_score=0.8),
        "task": {"task_id": "order", "family": "symbolic", "answer_type": "short_text"},
        "text": "D before A before B before C. Answer: D A B C",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "A B C D",
            "self_repair_extracted_answer": "D A B C",
            "self_repair_arithmetic_consistent": True,
        },
    }

    assert _repair_short_text_symbolic_gaps(terse, prompt) == set()
    assert _repair_short_text_trace_gaps(terse, prompt) == {"order:missing_trace"}
    assert _exact_answer_repair_selection_score(terse, "short_text", prompt) == 0.0
    assert _repair_short_text_trace_gaps(traced, prompt) == set()
    assert _exact_answer_repair_selection_score(traced, "short_text", prompt) > 0.0


def test_short_text_trace_guard_requires_letter_transform_trace():
    prompt = (
        "A display starts with the code K L M. Rotate the code one step left, "
        "then swap the final two letters. What code should be displayed? "
        "Answer with the three letters separated by spaces."
    )
    terse = {
        **_record("llada-8b-instruct-hf", "letters", "self", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "letters", "family": "symbolic", "answer_type": "short_text"},
        "text": "Answer: L K M",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "M L K",
            "self_repair_extracted_answer": "L K M",
            "self_repair_arithmetic_consistent": True,
        },
    }
    traced = {
        **_record("llada-8b-instruct-hf", "letters", "self", task_score=1.0, trajectory_score=0.8),
        "task": {"task_id": "letters", "family": "symbolic", "answer_type": "short_text"},
        "text": "Rotate left gives L M K, then swap the final two letters. Answer: L K M",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "M L K",
            "self_repair_extracted_answer": "L K M",
            "self_repair_arithmetic_consistent": True,
        },
    }

    assert _repair_short_text_symbolic_gaps(terse, prompt) == set()
    assert _repair_short_text_trace_gaps(terse, prompt) == {"letter_transform:missing_trace"}
    assert _exact_answer_repair_selection_score(terse, "short_text", prompt) == 0.0
    assert _repair_short_text_trace_gaps(traced, prompt) == set()
    assert _exact_answer_repair_selection_score(traced, "short_text", prompt) > 0.0


def test_short_text_trace_guard_requires_syllogism_relation_trace():
    prompt = "All zargs are blicks. No blicks are morts. Can a zarg be a mort? Answer yes or no."
    terse = {
        **_record("llada-8b-instruct-hf", "syllogism", "self", task_score=1.0, trajectory_score=0.9),
        "task": {"task_id": "syllogism", "family": "symbolic", "answer_type": "short_text"},
        "text": "Answer: no",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "yes",
            "self_repair_extracted_answer": "no",
            "self_repair_arithmetic_consistent": True,
        },
    }
    traced = {
        **_record("llada-8b-instruct-hf", "syllogism", "self", task_score=1.0, trajectory_score=0.8),
        "task": {"task_id": "syllogism", "family": "symbolic", "answer_type": "short_text"},
        "text": "All zargs are blicks, and no blicks are morts. Answer: no",
        "schedule": None,
        "repair": {
            "name": "self_check_answer_repair",
            "source_extracted_answer": "yes",
            "self_repair_extracted_answer": "no",
            "self_repair_arithmetic_consistent": True,
        },
    }

    assert _repair_short_text_symbolic_gaps(terse, prompt) == set()
    assert _repair_short_text_trace_gaps(terse, prompt) == {"syllogism:missing_trace"}
    assert _exact_answer_repair_selection_score(terse, "short_text", prompt) == 0.0
    assert _repair_short_text_trace_gaps(traced, prompt) == set()
    assert _exact_answer_repair_selection_score(traced, "short_text", prompt) > 0.0


def test_exact_self_repair_runs_for_unsupported_failed_integer_task_when_enabled():
    task = GeneralReasoningTask(
        task_id="math_hard",
        family="math",
        prompt=(
            "A store sells notebooks in packs of 6. Mara buys 4 packs, gives 7 "
            "notebooks away, then splits the rest equally among 5 bins. How many "
            "notebooks are left over? Answer with one integer."
        ),
        answer_type="integer",
        scorer="exact_integer",
        answer=2,
        max_new_tokens=16,
    )
    source = _record("llada-8b-instruct-hf", "math_hard", "baseline", task_score=0.0, trajectory_score=0.4)
    source["task_score"]["extracted_answer"] = 3

    assert not _should_run_exact_answer_repairs("LLaDA 8B", task, 2, source)
    assert _should_run_exact_answer_repairs(
        "LLaDA 8B",
        task,
        2,
        source,
        exact_self_repair=True,
    )


def test_exact_verifier_revision_skips_no_proposal_integer_answer_span(tmp_path):
    task = GeneralReasoningTask(
        task_id="math_hard_no_span",
        family="math",
        prompt=(
            "A store sells notebooks in packs of 6. Mara buys 4 packs, gives 7 "
            "notebooks away, then splits the rest equally among 5 bins. How many "
            "notebooks are left over? Answer with one integer."
        ),
        answer_type="integer",
        scorer="exact_integer",
        answer=2,
        max_new_tokens=4,
    )
    source = _record("llada-8b-instruct-hf", "math_hard_no_span", "baseline", task_score=0.0, trajectory_score=0.4)
    source["task"] = {"task_id": task.task_id, "family": "math", "answer_type": "integer"}
    source["text"] = "Scratch. Answer: 3"
    source["generated_token_ids"] = [10, 11, 12, 13]
    source["task_score"]["extracted_answer"] = 3
    backend = _FakeExactRepairBackend(
        ["6*4 = 24. 24 - 7 = 17. 17 / 5 leaves remainder 2. Answer: 2"],
        tokenizer=_TokenPiecesTokenizer(
            {
                10: "Scratch. ",
                11: "Answer: ",
                12: "3",
                13: ".",
            }
        ),
    )

    records = _generate_exact_answer_repair_records(
        backend,
        task,
        source_record=source,
        limit=1,
        exact_self_repair=True,
        exact_verifier_revision=True,
        generation_seed_base=23,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
    )

    assert [record["repair"]["name"] for record in records] == ["self_check_answer_repair"]
    assert backend.configs[0].initial_suffix_token_ids is None
    assert records[0]["repair"]["self_repair_extracted_answer"] == "2"


def test_answer_text_matches_exact_repair_proposals():
    assert _answer_text_matches_proposal("Final answer: 540.", "540", "integer")
    assert _answer_text_matches_proposal("Answer: B", "B", "multiple_choice")
    assert _answer_text_matches_proposal("Final: green, red, blue.", "green red blue", "short_text")
    assert not _answer_text_matches_proposal("Final: off", "on", "short_text")


def test_guarded_repair_selection_penalizes_overpreserved_history_state():
    baseline = {
        **_record("model", "plan", "evolved", task_score=0.5, trajectory_score=0.4),
        "text": "Compare the baseline and intervention, then record a decision threshold.",
    }
    visible_repair = {
        **_record("model", "plan", "history_visible_repair", task_score=0.0, trajectory_score=0.9),
        "config": {"max_new_tokens": 64},
        "schedule": None,
        "repair": {
            "name": "history_visible_repair",
            "source_state": "history",
            "seed_masked_positions": 6,
            "source_history_visible_chars": 240,
        },
        "text": (
            "Compare the baseline and intervention, preserve rollback, record metrics, "
            "and define a decision threshold."
        ),
    }

    prompt = "Compare a baseline and intervention, preserve rollback, and define a decision threshold."
    unguarded_score = _repair_selection_score(
        visible_repair,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="generic",
        repair_selector="planning_quality",
    )
    guarded_score = _repair_selection_score(
        visible_repair,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="generic",
        repair_selector="planning_quality_guarded",
    )

    assert unguarded_score > guarded_score
    assert unguarded_score - guarded_score > 0.05


def test_source_relative_repair_selection_requires_quality_delta():
    prompt = "Compare a baseline and intervention, preserve rollback, and define a decision threshold."
    baseline = {
        **_record("model", "plan", "evolved", task_score=0.5, trajectory_score=0.4),
        "text": "Compare the baseline and intervention, preserve rollback, and define a threshold.",
    }
    copied_repair = {
        **_record("model", "plan", "history_visible_repair", task_score=0.5, trajectory_score=0.95),
        "config": {"max_new_tokens": 64},
        "schedule": None,
        "repair": {
            "name": "history_visible_repair",
            "source_state": "history",
            "seed_masked_positions": 6,
            "source_history_visible_chars": 240,
        },
        "text": "Compare the baseline and intervention, preserve rollback, and define a threshold.",
    }
    improved_repair = {
        **_record("model", "plan", "prefix_25_repair", task_score=0.0, trajectory_score=0.3),
        "schedule": None,
        "repair": {"name": "prefix_25_repair"},
        "text": (
            "First compare the baseline and intervention metrics because the effect may be a "
            "regression. Then preserve rollback, validate with logs, monitor risk, and define "
            "a decision threshold before release."
        ),
    }

    copied_score = _repair_selection_score(
        copied_repair,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="generic",
        repair_selector="planning_quality_delta_guarded",
    )
    improved_score = _repair_selection_score(
        improved_repair,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="generic",
        repair_selector="planning_quality_delta_guarded",
    )
    selected = select_repair_record(
        [baseline, copied_repair, improved_repair],
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="generic",
        repair_selector="planning_quality_delta_guarded",
        promotion_margin=0.02,
    )

    assert copied_score <= 0.0
    assert improved_score > 0.02
    assert selected["repair"]["name"] == "prefix_25_repair"


def test_planning_contradiction_penalty_catches_prompt_violations():
    baseline_skip = {
        **_record("model", "plan", "bad", task_score=0.0, trajectory_score=0.3),
        "text": "Skip the baseline and spend both jobs on risky variants.",
    }
    checkpoint_delete = {
        **_record("model", "plan", "bad", task_score=0.0, trajectory_score=0.3),
        "text": "Delete old files immediately, then resume training.",
    }

    assert _planning_contradiction_penalty(
        baseline_skip,
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, the other tests a risky intervention.",
    ) >= 0.16
    assert _planning_contradiction_penalty(
        checkpoint_delete,
        "Checkpoint writes are failing. Give a recovery plan that preserves reproducibility and avoids corrupting the best checkpoint.",
    ) >= 0.14


def test_planning_contradiction_penalty_catches_prompt_checklist_leakage():
    leaked = {
        **_record("model", "plan", "constraint_gap_revision_anchor25_repair", task_score=0.0, trajectory_score=0.3),
        "repair": {
            "name": "constraint_gap_revision_anchor25_repair",
            "prompt_constraint_gap_terms": [
                "gpu",
                "jobs",
                "overnight",
                "gives",
                "reliable",
                "other",
                "tests",
                "risky",
                "intervention",
                "measurements",
                "tomorrow",
            ],
        },
        "text": (
            "Collect only the reliable baseline before running the risky reasoning intervention. "
            "The, gpu, jobs, overnight, gives, reliable baseline, other, tests, risky, "
            "intervention, measurements, tomorrow."
        ),
    }

    assert _planning_contradiction_penalty(
        leaked,
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline.",
    ) >= 0.18


def test_planning_span_residue_penalty_catches_reconstructed_weak_span():
    weak_span = "If the baseline job fails, the baseline data will not be available, making it a valid comparison."
    record = {
        **_record("model", "plan", "constraint_gap_span_repair", task_score=0.0, trajectory_score=0.3),
        "repair": {
            "name": "constraint_gap_span_repair",
            "planning_span_targets": [weak_span],
        },
        "text": f"Run the baseline job first. {weak_span}",
    }

    assert _planning_span_residue_penalty(record) == 0.12
    assert _planning_contradiction_penalty(
        record,
        "Collect a reliable baseline measurement and explain failure modes.",
    ) >= 0.12


def test_risk_guarded_source_relative_selector_penalizes_bad_repairs():
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention."
    )
    baseline = {
        **_record("model", "plan", "evolved", task_score=0.5, trajectory_score=0.4),
        "text": "Run the baseline, then compare one risky intervention.",
    }
    risky_repair = {
        **_record("model", "plan", "constraint_gap_revision_repair", task_score=0.0, trajectory_score=0.4),
        "schedule": None,
        "repair": {"name": "constraint_gap_revision_repair"},
        "text": (
            "Skip the baseline and spend both jobs on risky variants because the "
            "intervention is the only thing that matters."
        ),
    }

    unguarded = _repair_selection_score(
        risky_repair,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="generic",
        repair_selector="planning_quality_delta_guarded",
    )
    risk_guarded = _repair_selection_score(
        risky_repair,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="generic",
        repair_selector="planning_quality_delta_risk_guarded",
    )

    assert risk_guarded < unguarded


def test_repair_records_for_source_filters_incompatible_rescore_repairs():
    source = _record("model", "plan", "evolved_a", task_score=0.0, trajectory_score=0.5)
    compatible = {
        **_record("model", "plan", "repair_a", task_score=0.0, trajectory_score=0.5),
        "schedule": None,
        "repair": {"name": "prefix_25_repair", "source_control": "evolved_a"},
    }
    incompatible = {
        **_record("model", "plan", "repair_b", task_score=0.0, trajectory_score=0.5),
        "schedule": None,
        "repair": {"name": "prefix_50_repair", "source_control": "evolved_b"},
    }

    assert _repair_records_for_source([compatible, incompatible], source) == [compatible]


def test_history_repair_fraction_helpers_parse_and_render_cli_values():
    fractions = _float_csv("0.25,0.5")

    assert fractions == (0.25, 0.5)
    assert _format_fraction_list(list(fractions)) == "0.25,0.50"


def test_history_rescue_candidates_exclude_primary_repairs():
    primary = (DiffusionRepairCandidate(name="history_prefix_25_repair", source_state="history"),)

    rescue = _history_rescue_candidates(
        history_rescue_fractions=(0.25, 0.5),
        include_history_rescue_visible=False,
        existing_repairs=primary,
    )

    assert [repair.name for repair in rescue] == ["history_prefix_50_repair"]


def test_history_rescue_candidates_can_include_visible_state_once():
    primary = (DiffusionRepairCandidate(name="history_visible_repair", source_state="history"),)

    rescue = _history_rescue_candidates(
        history_rescue_fractions=(0.5,),
        include_history_rescue_visible=True,
        existing_repairs=primary,
    )

    assert [repair.name for repair in rescue] == ["history_prefix_50_repair"]


def test_primary_repair_candidates_can_include_visible_history_state():
    repairs = _repair_candidates(
        include_history_repairs=True,
        history_repair_fractions=(0.25,),
        include_history_visible_repair=True,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "history_prefix_25_repair",
        "history_visible_repair",
        "prefix_25_repair",
    ]


def test_source_relative_repair_pack_prioritizes_minimal_low_confidence_repairs():
    repairs = _repair_candidates(
        repair_pack="source_relative",
        include_history_repairs=True,
        history_repair_fractions=(0.50,),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "history_prefix_50_repair",
        "low_confidence_15_repair",
        "low_confidence_25_repair",
    ]


def test_targeted_content_repair_pack_prioritizes_text_repairs():
    repairs = _repair_candidates(
        repair_pack="targeted_content",
        include_history_repairs=True,
        history_repair_fractions=(0.50,),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "history_prefix_50_repair",
        "targeted_filler_repair",
        "targeted_filler_wide_repair",
    ]


def test_prompt_guided_repair_pack_can_be_budgeted_after_history_repair():
    repairs = _repair_candidates(
        repair_pack="prompt_guided",
        include_history_repairs=True,
        history_repair_fractions=(0.50,),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "history_prefix_50_repair",
        "prompt_guided_revision_repair",
        "prompt_guided_revision_anchor25_repair",
    ]


def test_constraint_gap_repair_pack_keeps_state_adaptive_line_and_adds_gap_revision():
    repairs = _repair_candidates(
        repair_pack="constraint_gap",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "state_adaptive_history_repair",
        "prefix_25_repair",
        "constraint_gap_revision_repair",
    ]


def test_constraint_span_repair_pack_spends_only_span_candidate():
    repairs = _repair_candidates(
        repair_pack="constraint_span",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_repair"]
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_constraint_span_history_repair_pack_spends_history_span_candidate():
    repairs = _repair_candidates(
        repair_pack="constraint_span_history",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_history_repair"]
    assert repairs[0].source_state == "history"
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_constraint_span_anchor_select_pack_defers_anchor_choice_to_runner():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_select",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_select_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_constraint_span_phase_anchor_pack_defers_first_skeleton_choice_to_runner():
    repairs = _repair_candidates(
        repair_pack="constraint_span_phase_anchor",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_phase_anchor_repair"]
    assert repairs[0].source_state == "pre_generation_phase_anchor"
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_constraint_span_phase_hybrid_pack_keeps_promoted_controls_with_hybrid_source():
    repairs = _repair_candidates(
        repair_pack="constraint_span_phase_hybrid_preserve_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_phase_hybrid_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_preservation_control_terms"


def test_constraint_span_phase_final_pack_keeps_phase_controls_with_final_source():
    repairs = _repair_candidates(
        repair_pack="constraint_span_phase_final_preserve_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_phase_final_preserve_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "final"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_preservation_control_terms"
    assert "selected denoise anchor" in str(repairs[0].prompt_repair_instruction)


def test_constraint_span_anchor_instability_pack_defers_anchor_and_masks_instability():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_instability_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_constraint_span_anchor_instability_gated_pack_defers_anchor_and_gates_mask():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_instability_gated_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_constraint_span_anchor_instability_prompt_gated_pack_defers_anchor_and_prompt_gate():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_prompt_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_instability_prompt_gated_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].history_instability_gate_prompt_policy == "active_instability_instruction"
    assert "unstable across sampled denoise history" in str(repairs[0].prompt_repair_instruction)


def test_constraint_span_anchor_instability_claim_gated_pack_defers_anchor_and_claim_gate():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_instability_claim_gated_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].history_instability_gate_prompt_policy == "active_instability_instruction"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert "public claim survives" in str(repairs[0].planning_prompt_gate_instruction)


def test_constraint_span_anchor_instability_claim_strict_gated_pack_forces_oracle_split():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_strict_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_strict_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].history_instability_gate_prompt_policy == "active_instability_instruction"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert "separate oracle best-of results from selected results" in str(
        repairs[0].planning_prompt_gate_instruction
    )


def test_constraint_span_anchor_instability_claim_oracle_gated_pack_keeps_compact_oracle_split():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_oracle_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_oracle_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].history_instability_gate_prompt_policy == "active_instability_instruction"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    instruction = str(repairs[0].planning_prompt_gate_instruction)
    assert "Because extra tokens and a different prompt format are confounds" in instruction
    assert "separately report oracle best-of results and selected results" in instruction


def test_constraint_span_anchor_instability_claim_seeded_gated_pack_adds_seed_anchor():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text == (
        " separate oracle best-of results from selected results."
    )


def test_constraint_span_anchor_instability_claim_compatible_seeded_gated_pack_adds_dual_seed_anchor():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_compatible_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_compatible_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text == (
        " oracle selected results; claim survives if disappears."
    )


def test_constraint_span_anchor_instability_claim_auto_seeded_gated_pack_adds_seed_policy():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_control_terms"


def test_constraint_span_anchor_instability_claim_auto_action_seeded_gated_pack_adds_seed_policy():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_action_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_action_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_action_control_terms"


def test_constraint_span_anchor_instability_claim_auto_compat_seeded_gated_pack_adds_seed_policy():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_compat_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_compat_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_compatibility_control_terms"


def test_constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated_pack_adds_realization_prompt():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_compat_realized_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_compatibility_control_terms"
    assert "selected-run results" in str(repairs[0].planning_prompt_gate_instruction)


def test_constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_pack_adds_direct_preservation_prompt():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_preservation_control_terms"
    instruction = str(repairs[0].planning_prompt_gate_instruction)
    assert "preserve only the public claim" in instruction
    assert "seed anchor" not in instruction


def test_constraint_span_anchor_instability_claim_auto_joint_seeded_gated_pack_adds_joint_seed_policy():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_joint_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_joint_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_joint_control_terms"
    assert "selected-run results" in str(repairs[0].planning_prompt_gate_instruction)


def test_constraint_span_anchor_instability_claim_auto_seeded_realization_gated_pack_adds_seed_policy():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_seeded_realization_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_seeded_realization_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_control_terms"
    assert "token budget" in str(repairs[0].planning_prompt_gate_instruction)


def test_compact_control_seed_suffix_policy_synthesizes_dual_control_anchor():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    seed_text, diagnostics = _planning_prompt_gate_seed_suffix_text(
        repair,
        task_prompt=(
            "A research result looks impressive, but the baseline used more tokens "
            "and a different prompt format. Design a quick falsification plan before "
            "anyone writes a public claim."
        ),
        prompt_constraint_gap_terms=["baseline", "prompt", "claim"],
        rubric_items=(
            "separate oracle best-of results from selected results",
            "state what claim survives if the effect disappears",
        ),
    )

    assert seed_text == " oracle selected results; claim survives if disappears."
    assert diagnostics["seed_suffix_policy"] == "compact_control_terms"
    assert diagnostics["seed_suffix_policy_reason"] == "oracle_selected_claim_survival"


def test_compact_action_control_seed_suffix_policy_synthesizes_action_anchor():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_action_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    seed_text, diagnostics = _planning_prompt_gate_seed_suffix_text(
        repair,
        task_prompt=(
            "A research result looks impressive, but the baseline used more tokens "
            "and a different prompt format. Design a quick falsification plan before "
            "anyone writes a public claim."
        ),
        prompt_constraint_gap_terms=["baseline", "intervention", "locked tasks", "claim"],
        rubric_items=(
            "separate oracle best-of results from selected results",
            "state what claim survives if the effect disappears",
        ),
    )

    assert seed_text == " rerun; oracle selected; claim survives."
    assert diagnostics["seed_suffix_policy"] == "compact_action_control_terms"
    assert diagnostics["seed_suffix_policy_reason"] == "action_oracle_selected_claim_survival"


def test_compact_compatibility_control_seed_suffix_policy_prefers_full_control_anchor():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_compat_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    seed_text, diagnostics = _planning_prompt_gate_seed_suffix_text(
        repair,
        task_prompt=(
            "A research result looks impressive, but the baseline used more tokens "
            "and a different prompt format. Design a quick falsification plan before "
            "anyone writes a public claim."
        ),
        prompt_constraint_gap_terms=["baseline", "intervention", "locked tasks", "claim"],
        rubric_items=(
            "separate oracle best-of results from selected results",
            "state what claim survives if the effect disappears",
        ),
    )

    assert seed_text == " oracle selected results; claim survives if disappears."
    assert diagnostics["seed_suffix_policy"] == "compact_compatibility_control_terms"
    assert diagnostics["seed_suffix_policy_reason"] == "compatibility_oracle_selected_claim_survival"
    scores = diagnostics["seed_suffix_candidate_scores"]
    assert scores[0]["seed_suffix_text"] == seed_text
    assert scores[0]["has_results"]
    assert scores[0]["has_disappear_condition"]
    assert any(row["reason"] == "action_oracle_selected_claim_survival" for row in scores)


def test_compact_joint_control_seed_suffix_policy_prefers_realizable_semantic_anchor():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_joint_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    seed_text, diagnostics = _planning_prompt_gate_seed_suffix_text(
        repair,
        task_prompt=(
            "A research result looks impressive, but the baseline used more tokens "
            "and a different prompt format. Design a quick falsification plan before "
            "anyone writes a public claim."
        ),
        prompt_constraint_gap_terms=["baseline", "intervention", "locked tasks", "claim"],
        rubric_items=(
            "separate oracle best-of results from selected results",
            "state what claim survives if the effect disappears",
        ),
    )

    assert seed_text == " separate oracle selected; claim survives if disappears."
    assert diagnostics["seed_suffix_policy"] == "compact_joint_control_terms"
    assert diagnostics["seed_suffix_policy_reason"] == "joint_separate_oracle_selected_claim_survival"
    scores = diagnostics["seed_suffix_candidate_scores"]
    assert scores[0]["seed_suffix_text"] == seed_text
    legacy = next(row for row in scores if row["reason"] == "oracle_selected_claim_survival")
    assert scores[0]["score"] > legacy["score"]
    assert scores[0]["expected_realization_score"] > legacy["expected_realization_score"]
    assert scores[0]["semantic_intent_score"] > legacy["semantic_intent_score"]


def test_compact_preservation_control_seed_suffix_policy_prefers_preserve_claim_anchor():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    seed_text, diagnostics = _planning_prompt_gate_seed_suffix_text(
        repair,
        task_prompt=(
            "A research result looks impressive, but the baseline used more tokens "
            "and a different prompt format. Design a quick falsification plan before "
            "anyone writes a public claim."
        ),
        prompt_constraint_gap_terms=["baseline", "intervention", "locked tasks", "claim"],
        rubric_items=(
            "separate oracle best-of results from selected results",
            "state what claim survives if the effect disappears",
        ),
    )

    assert seed_text == " oracle selected results; preserve claim if disappears."
    assert diagnostics["seed_suffix_policy"] == "compact_preservation_control_terms"
    assert diagnostics["seed_suffix_policy_reason"] == "preservation_oracle_selected_results_preserve_claim"
    scores = diagnostics["seed_suffix_candidate_scores"]
    assert scores[0]["seed_suffix_text"] == seed_text
    legacy = next(row for row in scores if row["reason"] == "oracle_selected_claim_survival")
    assert scores[0]["score"] > legacy["score"]
    assert scores[0]["preservation_action_score"] == 1.0


def test_seed_realization_quality_rewards_direct_control_plan_over_meta_label():
    prompt = (
        "A research result looks impressive, but the baseline used more tokens "
        "and a different prompt format. Design a quick falsification plan before "
        "anyone writes a public claim."
    )
    anchor = {
        "active": True,
        "seed_suffix_text": " oracle selected results; claim survives if disappears.",
    }
    direct = {
        **_record("model", "plan_004", "direct", task_score=0.0, trajectory_score=0.4),
        "prompt": prompt,
        "text": (
            "Equalize token budget and prompt format, rerun baseline and intervention "
            "on locked tasks, record regressions and wins, validate failure modes, "
            "report oracle selected results, and state the claim survives if it "
            "disappears."
        ),
        "repair": {"planning_seed_suffix_anchor": anchor},
    }
    meta = {
        **_record("model", "plan_004", "meta", task_score=0.0, trajectory_score=0.4),
        "prompt": prompt,
        "text": (
            "Control: token budget, prompt format, locked tasks, regressions, wins "
            "and failure modes; use generated compact seed anchor as oracle selected "
            "results; claim survives if disappears."
        ),
        "repair": {"planning_seed_suffix_anchor": anchor},
    }

    direct_components = _seed_realization_quality_components(direct, prompt)
    meta_components = _seed_realization_quality_components(meta, prompt)

    assert direct_components["active_seed_anchor"]
    assert direct_components["realization_quality_score"] > meta_components["realization_quality_score"]
    assert direct_components["meta_penalty"] < meta_components["meta_penalty"]
    assert _seed_realization_quality_score(direct, prompt) > _seed_realization_quality_score(meta, prompt)
    assert _seed_objective_score(direct, prompt) > _seed_objective_score(meta, prompt)


def test_seed_objective_rewards_semantic_separation_over_compare_wording():
    prompt = (
        "A research result looks impressive, but the baseline used more tokens "
        "and a different prompt format. Design a quick falsification plan before "
        "anyone writes a public claim."
    )
    anchor = {
        "active": True,
        "seed_suffix_text": " oracle selected results; claim survives if disappears.",
    }
    compare = {
        **_record("model", "plan_004", "compare", task_score=0.0, trajectory_score=0.4),
        "prompt": prompt,
        "text": (
            "Equalize token budget and prompt format, rerun both versions on locked "
            "tasks, record regressions and wins, validate failure modes, and compare "
            "oracle selected results; claim survives if disappears."
        ),
        "repair": {"planning_seed_suffix_anchor": anchor},
    }
    separate = {
        **_record("model", "plan_004", "separate", task_score=0.0, trajectory_score=0.4),
        "prompt": prompt,
        "text": (
            "Equalize token budget and prompt format, rerun baseline and intervention "
            "on locked tasks, record regressions and wins, validate failure modes. "
            "Separate oracle selected results; claim survives if disappears."
        ),
        "repair": {"planning_seed_suffix_anchor": anchor},
    }

    compare_components = _seed_realization_quality_components(compare, prompt)
    separate_components = _seed_realization_quality_components(separate, prompt)

    assert separate_components["semantic_preservation_score"] > compare_components["semantic_preservation_score"]
    assert separate_components["seed_objective_score"] > compare_components["seed_objective_score"]


def test_seed_realization_guarded_repair_selector_penalizes_seed_meta_text():
    prompt = (
        "A research result looks impressive, but the baseline used more tokens "
        "and a different prompt format. Design a quick falsification plan before "
        "anyone writes a public claim."
    )
    baseline = {
        **_record("model", "plan_004", "baseline", task_score=0.0, trajectory_score=0.4),
        "text": "Rerun a baseline and intervention before making a public claim.",
    }
    anchor = {
        "active": True,
        "seed_suffix_text": " oracle selected results; claim survives if disappears.",
    }
    direct = {
        **_record("model", "plan_004", "direct", task_score=0.0, trajectory_score=0.4),
        "text": (
            "Equalize token budget and prompt format, rerun on locked tasks, record "
            "regressions and wins, validate failure modes, report oracle selected "
            "results, and state the claim survives if it disappears."
        ),
        "repair": {"planning_seed_suffix_anchor": anchor},
    }
    meta = {
        **_record("model", "plan_004", "meta", task_score=0.0, trajectory_score=0.4),
        "text": (
            "Control: token budget, prompt format, locked tasks, regressions, wins "
            "and failure modes; use generated compact seed anchor as oracle selected "
            "results; claim survives if disappears."
        ),
        "repair": {"planning_seed_suffix_anchor": anchor},
    }

    direct_score = _repair_selection_score(
        direct,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="planning_state",
        repair_selector="planning_quality_seed_realization_guarded",
    )
    meta_score = _repair_selection_score(
        meta,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="planning_state",
        repair_selector="planning_quality_seed_realization_guarded",
    )

    assert direct_score > meta_score
    direct_objective_score = _repair_selection_score(
        direct,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="planning_state",
        repair_selector="planning_quality_seed_objective_guarded",
    )
    meta_objective_score = _repair_selection_score(
        meta,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="planning_state",
        repair_selector="planning_quality_seed_objective_guarded",
    )
    assert direct_objective_score > meta_objective_score


def test_candidate_aware_promotion_alias_matches_seed_realization_guarded_selector():
    prompt = (
        "A research result looks impressive, but the baseline used more tokens "
        "and a different prompt format. Design a quick falsification plan."
    )
    baseline = {
        **_record("model", "plan_004", "baseline", task_score=0.0, trajectory_score=0.4),
        "text": "Rerun the baseline and intervention before making a public claim.",
    }
    repair = {
        **_record("model", "plan_004", "repair", task_score=0.0, trajectory_score=0.4),
        "text": (
            "Equalize token budget and prompt format, rerun on locked tasks, record "
            "regressions and wins, validate failure modes, and state the surviving claim."
        ),
        "repair": {
            "planning_seed_suffix_anchor": {
                "active": True,
                "seed_suffix_text": " rerun tasks; record wins.",
            }
        },
    }

    guarded_score = _repair_selection_score(
        repair,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="planning_state",
        repair_selector="planning_quality_seed_realization_guarded",
    )
    alias_score = _repair_selection_score(
        repair,
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="planning_state",
        repair_selector="candidate_aware_promotion_v1",
    )

    assert alias_score == guarded_score


def test_seed_realization_guarded_selection_rejects_low_quality_seed_meta_text():
    prompt = (
        "A research result looks impressive, but the baseline used more tokens "
        "and a different prompt format. Design a quick falsification plan before "
        "anyone writes a public claim."
    )
    baseline = {
        **_record("model", "plan_004", "baseline", task_score=0.0, trajectory_score=0.4),
        "text": (
            "Rerun the baseline and intervention on locked tasks before making a "
            "public claim."
        ),
    }
    meta = {
        **_record("model", "plan_004", "meta", task_score=0.0, trajectory_score=0.4),
        "text": (
            "Control: token budget, prompt format, locked tasks, regressions, wins "
            "and failure modes; use generated compact seed anchor as oracle selected "
            "results; claim survives if disappears."
        ),
        "repair": {
            "planning_seed_suffix_anchor": {
                "active": True,
                "seed_suffix_text": " oracle selected results; claim survives if disappears.",
            }
        },
    }

    selected = select_repair_record(
        [baseline, meta],
        baseline_record=baseline,
        task_prompt=prompt,
        task_answer_type="rubric",
        trajectory_selector="planning_state",
        repair_selector="planning_quality_seed_realization_guarded",
        promotion_margin=0.02,
    )

    assert selected is baseline


def test_planning_seed_suffix_anchor_fixes_tokens_into_masked_tail():
    config = DiffusionGenerationConfig(
        max_new_tokens=8,
        initial_suffix_token_ids=(10, 11, None, None, None, None, 98, 99),
    )
    updated, diagnostics = _apply_planning_seed_suffix_anchor(
        config,
        seed_suffix_text=" oracle results",
        token_encoder=lambda text: [31, 32],
        active=True,
    )

    assert updated.initial_suffix_token_ids == (10, 11, None, None, 31, 32, 98, 99)
    assert diagnostics["active"] is True
    assert diagnostics["anchor_positions"] == [4, 5]
    assert diagnostics["anchor_token_count"] == 2


def test_constraint_span_anchor_instability_prompt_only_gated_pack_defers_anchor_and_prompt_gate():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_prompt_only_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_instability_prompt_only_gated_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction is None
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].history_instability_gate_prompt_policy == "active_instability_instruction"
    assert "denoise history shows instability" in str(repairs[0].prompt_repair_instruction)


def test_anchor_instability_execution_repair_preserves_remask_after_resolution():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    final_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="final",
    )
    history_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="history",
    )

    assert final_repair.name == "constraint_gap_span_anchor_instability_repair"
    assert final_repair.source_state == "final"
    assert final_repair.remask_history_unstable_fraction == 0.08
    assert history_repair.name == "constraint_gap_span_anchor_instability_repair"
    assert history_repair.source_state == "history"
    assert history_repair.remask_history_unstable_fraction == 0.08


def test_gated_anchor_instability_execution_repair_keeps_base_seed_name():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    final_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="final",
    )

    assert final_repair.name == "constraint_gap_span_repair"
    assert final_repair.remask_history_unstable_fraction == 0.08
    assert final_repair.history_instability_gate_policy == "multi_span_low_quality"
    assert final_repair.prompt_repair_instruction == _repair_candidates(
        repair_pack="constraint_span",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0].prompt_repair_instruction


def test_prompt_gated_anchor_instability_execution_preserves_base_prompt_until_gate_fires():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_prompt_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    final_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="final",
    )

    assert final_repair.name == "constraint_gap_span_repair"
    assert final_repair.remask_history_unstable_fraction == 0.08
    assert final_repair.history_instability_gate_policy == "multi_span_low_quality"
    assert final_repair.history_instability_gate_prompt_policy == "active_instability_instruction"
    assert final_repair.prompt_repair_instruction == _repair_candidates(
        repair_pack="constraint_span",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0].prompt_repair_instruction


def test_claim_gated_anchor_instability_execution_preserves_base_prompt_until_gate_fires():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    final_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="final",
    )

    assert final_repair.name == "constraint_gap_span_repair"
    assert final_repair.remask_history_unstable_fraction == 0.08
    assert final_repair.history_instability_gate_policy == "multi_span_low_quality"
    assert final_repair.history_instability_gate_prompt_policy == "active_instability_instruction"
    assert final_repair.planning_prompt_gate_policy == "public_claim_confound_control"
    assert "public claim survives" in str(final_repair.planning_prompt_gate_instruction)
    assert final_repair.prompt_repair_instruction == _repair_candidates(
        repair_pack="constraint_span",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0].prompt_repair_instruction


def test_seeded_claim_gated_anchor_instability_execution_preserves_seed_suffix_anchor():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    final_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="final",
    )

    assert final_repair.name == "constraint_gap_span_repair"
    assert final_repair.planning_prompt_gate_policy == "public_claim_confound_control"
    assert final_repair.planning_prompt_gate_seed_suffix_text == (
        " separate oracle best-of results from selected results."
    )


def test_auto_seeded_claim_gated_anchor_instability_execution_preserves_seed_suffix_policy():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    final_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="final",
    )

    assert final_repair.name == "constraint_gap_span_repair"
    assert final_repair.planning_prompt_gate_policy == "public_claim_confound_control"
    assert final_repair.planning_prompt_gate_seed_suffix_text is None
    assert final_repair.planning_prompt_gate_seed_suffix_policy == "compact_control_terms"


def test_auto_seeded_realization_claim_gated_anchor_instability_execution_preserves_seed_suffix_policy():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_auto_seeded_realization_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    final_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="final",
    )

    assert final_repair.name == "constraint_gap_span_repair"
    assert final_repair.planning_prompt_gate_policy == "public_claim_confound_control"
    assert final_repair.planning_prompt_gate_seed_suffix_text is None
    assert final_repair.planning_prompt_gate_seed_suffix_policy == "compact_control_terms"
    assert "failure modes" in str(final_repair.planning_prompt_gate_instruction)


def test_prompt_only_gated_anchor_instability_execution_preserves_base_prompt_until_gate_fires():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_prompt_only_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    final_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="final",
    )

    assert final_repair.name == "constraint_gap_span_repair"
    assert final_repair.remask_history_unstable_fraction is None
    assert final_repair.history_instability_gate_policy == "multi_span_low_quality"
    assert final_repair.history_instability_gate_prompt_policy == "active_instability_instruction"
    assert final_repair.prompt_repair_instruction == _repair_candidates(
        repair_pack="constraint_span",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0].prompt_repair_instruction


def test_gated_anchor_instability_history_execution_preserves_base_history_prompt():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    history_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="history",
    )

    assert history_repair.name == "constraint_gap_span_history_repair"
    assert history_repair.remask_history_unstable_fraction == 0.08
    assert history_repair.history_instability_gate_policy == "multi_span_low_quality"
    assert history_repair.prompt_repair_instruction == _repair_candidates(
        repair_pack="constraint_span_history",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0].prompt_repair_instruction


def test_history_instability_gate_only_allows_low_quality_multi_span_final_anchor():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]
    final_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="final",
    )

    active = _history_instability_gate_decision(
        final_repair,
        planning_span_targets=["a", "b", "c"],
        source_quality_score=0.24,
        source_state="final",
    )
    too_few = _history_instability_gate_decision(
        final_repair,
        planning_span_targets=["a", "b"],
        source_quality_score=0.24,
        source_state="final",
    )
    high_quality = _history_instability_gate_decision(
        final_repair,
        planning_span_targets=["a", "b", "c"],
        source_quality_score=0.31,
        source_state="final",
    )
    history_anchor = _history_instability_gate_decision(
        final_repair,
        planning_span_targets=["a", "b", "c"],
        source_quality_score=0.24,
        source_state="history",
    )

    assert active == {
        "active": True,
        "policy": "multi_span_low_quality",
        "reason": "multi_span_low_quality",
    }
    assert too_few["active"] is False
    assert too_few["reason"] == "too_few_planning_spans"
    assert high_quality["active"] is False
    assert high_quality["reason"] == "source_quality_above_gate"
    assert history_anchor["active"] is False
    assert history_anchor["reason"] == "history_anchor_skip"


def test_planning_prompt_gate_only_allows_public_claim_confound_repairs():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]
    final_repair = _anchor_selected_execution_repair(
        repair,
        configured_source_state="pre_generation_anchor",
        resolved_source_state="final",
    )
    task_prompt = (
        "A research result looks impressive, but the baseline used more tokens "
        "and a different prompt format. Design a quick falsification plan before "
        "anyone writes a public claim."
    )

    active = _planning_prompt_gate_decision(
        final_repair,
        task_prompt=task_prompt,
        prompt_constraint_gap_terms=["looks", "used"],
        planning_span_targets=["weak downstream draft"],
        source_quality_score=0.28,
        source_state="final",
    )
    high_quality = _planning_prompt_gate_decision(
        final_repair,
        task_prompt=task_prompt,
        prompt_constraint_gap_terms=["looks", "used"],
        planning_span_targets=["weak downstream draft"],
        source_quality_score=0.40,
        source_state="final",
    )
    no_claim = _planning_prompt_gate_decision(
        final_repair,
        task_prompt="A model training run diverges after an optimizer change.",
        prompt_constraint_gap_terms=["gpu", "cause"],
        planning_span_targets=["weak downstream draft"],
        source_quality_score=0.28,
        source_state="final",
    )
    history_anchor = _planning_prompt_gate_decision(
        final_repair,
        task_prompt=task_prompt,
        prompt_constraint_gap_terms=["looks", "used"],
        planning_span_targets=["weak downstream draft"],
        source_quality_score=0.28,
        source_state="history",
    )

    assert active == {
        "active": True,
        "policy": "public_claim_confound_control",
        "reason": "public_claim_confound_control",
    }
    assert high_quality["active"] is False
    assert high_quality["reason"] == "source_quality_above_gate"
    assert no_claim["active"] is False
    assert no_claim["reason"] == "no_public_claim"
    assert history_anchor["active"] is False
    assert history_anchor["reason"] == "history_anchor_skip"


def test_public_claim_prompt_gate_uses_direct_prompt_without_mask_meta_language():
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_instability_claim_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]
    active_repair = replace(repair, prompt_repair_instruction=repair.planning_prompt_gate_instruction)
    prompt = _planning_span_repair_prompt_override(
        "A research result looks impressive, but the baseline used more tokens and a different prompt format. "
        "Design a quick falsification plan before anyone writes a public claim.",
        "Create a quick falsification plan. This weak downstream sentence should be replaced.",
        ["looks", "used"],
        ["This weak downstream sentence should be replaced."],
        active_repair,
        planning_prompt_gate_active=True,
    )

    assert "Write only the completed falsification plan" in prompt
    assert "Equalize token budget and prompt format" in prompt
    assert "masked in the seed" not in prompt
    assert "weak downstream draft" not in prompt


def test_constraint_span_anchor_search_pack_defers_history_search_to_runner():
    repairs = _repair_candidates(
        repair_pack="constraint_span_anchor_search",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_search_repair"]
    assert repairs[0].source_state == "pre_generation_anchor_search"
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_constraint_span_history_contrast_pack_keeps_final_seed_with_history_prompt():
    repairs = _repair_candidates(
        repair_pack="constraint_span_history_contrast",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_history_contrast_repair"]
    assert repairs[0].source_state == "final"
    assert repairs[0].prompt_history_contrast is True
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_constraint_span_history_instability_pack_masks_unstable_final_positions():
    repairs = _repair_candidates(
        repair_pack="constraint_span_history_instability",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_history_instability_repair"]
    assert repairs[0].source_state == "final"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_anchor_select_and_history_span_packs_request_dense_history_by_default():
    assert _repair_pack_needs_dense_history("constraint_span_anchor_search")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_select")
    assert _repair_pack_needs_dense_history("constraint_span_phase_anchor")
    assert _repair_pack_needs_dense_history("constraint_span_phase_hybrid_preserve_seeded_gated")
    assert _repair_pack_needs_dense_history("constraint_span_phase_final_preserve_seeded_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_claim_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_claim_oracle_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_claim_seeded_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_claim_compatible_seeded_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_claim_auto_seeded_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_claim_auto_action_seeded_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_claim_auto_compat_seeded_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_claim_auto_joint_seeded_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_claim_auto_seeded_realization_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_claim_strict_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_prompt_only_gated")
    assert _repair_pack_needs_dense_history("constraint_span_anchor_instability_prompt_gated")
    assert _repair_pack_needs_dense_history("constraint_span_history_contrast")
    assert _repair_pack_needs_dense_history("constraint_span_history_instability")
    assert _repair_pack_needs_dense_history("constraint_span_history")
    assert not _repair_pack_needs_dense_history("constraint_span")


def test_constraint_span_clause_repair_pack_uses_clause_chunks():
    repairs = _repair_candidates(
        repair_pack="constraint_span_clause",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )

    assert [repair.name for repair in repairs] == ["constraint_gap_span_clause_repair"]
    assert repairs[0].planning_span_chunk_mode == "clause"


def test_state_adaptive_repair_pack_starts_with_conditional_history_and_confidence_repairs():
    repairs = _repair_candidates(
        repair_pack="state_adaptive",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=2,
    )

    assert [repair.name for repair in repairs] == [
        "state_adaptive_history_repair",
        "prefix_25_repair",
    ]


def test_replay_consistency_repair_pack_starts_with_replay_instability():
    repairs = _repair_candidates(
        repair_pack="replay_consistency",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=2,
    )

    assert [repair.name for repair in repairs] == [
        "replay_unstable_25_repair",
        "state_adaptive_history_repair",
    ]


def test_prompt_guided_repair_prompt_keeps_original_task_and_draft():
    repair = _repair_candidates(
        repair_pack="prompt_guided",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    prompt = _repair_prompt_override(
        "Original task prompt.",
        "Draft answer with repeated repeated filler.",
        repair,
    )

    assert prompt is not None
    assert "Original task prompt." in prompt
    assert "Draft answer with repeated repeated filler." in prompt
    assert "Rewrite the draft answer directly." in prompt


def test_constraint_gap_repair_prompt_names_missing_prompt_terms():
    repair = _repair_candidates(
        repair_pack="constraint_gap",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=3,
    )[2]

    prompt = _repair_prompt_override(
        "Compare baseline accuracy, rollback threshold, and customer risk.",
        "Measure accuracy only.",
        repair,
    )

    assert prompt is not None
    assert "Missing or weak task terms to cover:" in prompt
    assert "baseline" in prompt
    assert "rollback" in prompt
    assert "customer" in prompt
    assert "Draft answer to repair:" in prompt


def test_planning_constraint_gap_span_targets_bad_downstream_sentences():
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    source = (
        "Run the baseline job first. "
        "If the baseline job fails, you can still run the intervention job, but the "
        "baseline data will not be available, making it a valid comparison. "
        "If the baseline job succeeds, you can then run the intervention job, ensuring "
        "you have a publishable result even if the intervention fails."
    )

    targets = _planning_constraint_gap_span_targets(
        prompt,
        source,
        ["gpu", "jobs", "overnight", "risky", "reasoning", "measurements", "collect", "tomorrow"],
    )

    assert targets == [
        (
            "If the baseline job fails, you can still run the intervention job, but the "
            "baseline data will not be available, making it a valid comparison."
        ),
        (
            "If the baseline job succeeds, you can then run the intervention job, ensuring "
            "you have a publishable result even if the intervention fails."
        ),
    ]


def test_planning_constraint_gap_span_target_scores_are_source_relative():
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    source = (
        "Run the baseline job first. "
        "If the baseline job fails, you can still run the intervention job, but the "
        "baseline data will not be available, making it a valid comparison. "
        "Collect baseline and intervention metrics, record failure modes, and define a "
        "rollback threshold."
    )

    ranked = _planning_constraint_gap_span_target_scores(
        prompt,
        source,
        ["gpu", "jobs", "overnight", "risky", "reasoning", "measurements", "collect", "tomorrow"],
    )

    assert ranked
    assert "valid comparison" in ranked[0]["span"]
    assert ranked[0]["source_relative_preservation"] > 0.0
    assert ranked[0]["contradiction_relief"] > 0.0
    assert all("Run the baseline job first" not in target["span"] for target in ranked)


def test_planning_repair_chunks_split_long_clause_like_drafts():
    source = (
        "Run a stress test with a controlled dataset of 10,000 records to reproduce the failure, "
        "then analyze the logs for patterns such as timeouts, retries, and malformed payloads, "
        "then compare the result to the demo baseline before shipping."
    )

    chunks = _planning_repair_chunks(source, chunk_mode="clause")

    assert len(chunks) == 3
    assert chunks[0].startswith("Run a stress test")
    assert chunks[1].startswith("then analyze")
    assert chunks[2].startswith("then compare")


def test_planning_constraint_gap_span_targets_avoid_whole_sentence_fallback_for_long_draft():
    prompt = (
        "A customer pipeline fails once every thousand noisy hours. Decide what to collect "
        "and compare so the team can isolate root cause without delaying the demo."
    )
    source = (
        "Run a stress test with a controlled dataset of 10,000 records to reproduce the failure, "
        "then analyze the logs for patterns such as timeouts, timeouts, or timeouts, "
        "then compare the result to the demo baseline before shipping."
    )

    ranked = _planning_constraint_gap_span_target_scores(
        prompt,
        source,
        ["pipeline", "fails", "once", "thousand", "noisy", "hours", "customer"],
        chunk_mode="clause",
    )

    assert ranked
    assert all(not target.get("fallback") for target in ranked)
    assert all(target["span"] != source for target in ranked)
    assert any("timeouts" in target["span"] or "shipping" in target["span"] for target in ranked)


def test_planning_span_targets_default_to_sentence_chunks_for_long_draft():
    prompt = (
        "A customer pipeline fails once every thousand noisy hours. Decide what to collect "
        "and compare so the team can isolate root cause without delaying the demo."
    )
    source = (
        "Run a stress test with a controlled dataset of 10,000 records to reproduce the failure, "
        "then analyze the logs for patterns such as timeouts, timeouts, or timeouts, "
        "then compare the result to the demo baseline before shipping."
    )

    ranked = _planning_constraint_gap_span_target_scores(
        prompt,
        source,
        ["pipeline", "fails", "once", "thousand", "noisy", "hours", "customer"],
    )

    assert ranked == [
        {
            "span": source,
            "index": 0,
            "score": 0.0,
            "sentence_surface": 0.0,
            "without_surface": 0.0,
            "source_relative_preservation": 0.0,
            "contradiction_relief": 0.0,
            "prompt_gap_miss": 0.0,
            "keyword_coverage": 0.0,
            "fallback": True,
        }
    ]


def test_planning_span_targets_adapt_to_clause_chunks_for_long_draft():
    prompt = (
        "A customer pipeline fails once every thousand noisy hours. Decide what to collect "
        "and compare so the team can isolate root cause without delaying the demo."
    )
    source = (
        "Run a stress test with a controlled dataset of 10,000 records to reproduce the failure, "
        "then analyze the logs for patterns such as timeouts, timeouts, or timeouts, "
        "then compare the result to the demo baseline before shipping."
    )

    ranked = _planning_constraint_gap_span_target_scores(
        prompt,
        source,
        ["pipeline", "fails", "once", "thousand", "noisy", "hours", "customer"],
        chunk_mode="adaptive",
    )

    assert ranked
    assert all(not target.get("fallback") for target in ranked)
    assert all(target["span"] != source for target in ranked)
    assert any("timeouts" in target["span"] or "shipping" in target["span"] for target in ranked)


def test_compact_planning_span_targets_refine_long_risky_sentence_to_clauses():
    prompt = (
        "A customer pipeline fails once every thousand noisy hours. Decide what to collect "
        "and compare, including rollback thresholds, so the team can isolate root cause "
        "without delaying the demo."
    )
    long_sentence = (
        "If the baseline looks fine, then skip noisy-hour telemetry and log capture, then "
        "ship immediately without a rollback threshold, then call the comparison valid "
        "even if the intermittent failure appears again."
    )
    source = f"Keep the reliable demo baseline intact. {long_sentence}"
    gap_terms = ["customer", "pipeline", "thousand", "noisy", "collect", "compare", "rollback", "thresholds"]

    ranked = _planning_constraint_gap_span_target_scores(
        prompt,
        source,
        gap_terms,
        chunk_mode="adaptive",
        selection_policy="compact",
    )

    assert ranked
    assert all(target["span"] != long_sentence for target in ranked)
    assert any("ship immediately" in target["span"] or "comparison valid" in target["span"] for target in ranked)
    assert sum(len(str(target["span"]).split()) for target in ranked) < len(long_sentence.split())


def test_compact_planning_span_targets_keep_decision_rule_context():
    prompt = (
        "An ML model improves offline accuracy but triples production latency. Decide what "
        "measurement and rollback rule to use before release."
    )
    decision_rule = (
        "Decision rule: If accuracy improves by 10% or latency increases by <50%, ship; "
        "if accuracy improves by <10% and latency increases by >50%, rollback; "
        "if accuracy > 0 and latency > 0, gate."
    )
    source = f"Measure accuracy improvement and latency increase. {decision_rule}"

    targets = _planning_constraint_gap_span_targets(
        prompt,
        source,
        ["model", "offline", "triples", "production", "release", "rollback"],
        chunk_mode="adaptive",
        selection_policy="compact",
    )

    assert targets == [decision_rule]


def test_compact_planning_span_targets_keep_near_tie_failure_chain():
    prompt = (
        "A GPU training run diverges after an optimizer change. Plan the cheapest sequence "
        "to isolate the cause on one free debugging slot."
    )
    source = (
        "Run the model with the original optimizer first, then with the new one. "
        "If the divergence occurs only with the change, the issue is with the optimizer. "
        "If it occurs with both, the problem may lie in the model architecture or training loop. "
        "This experiment is sufficient to attribute the divergence to the optimizer change."
    )

    targets = _planning_constraint_gap_span_targets(
        prompt,
        source,
        ["gpu", "diverges", "free", "debugging", "cheapest", "sequence", "isolate", "cause"],
        chunk_mode="adaptive",
        selection_policy="compact",
    )

    assert targets == [
        "If the divergence occurs only with the change, the issue is with the optimizer.",
        "If it occurs with both, the problem may lie in the model architecture or training loop.",
        "This experiment is sufficient to attribute the divergence to the optimizer change.",
    ]


def test_planning_span_target_rows_expose_selected_source_relative_scores():
    record = {
        **_record("model", "plan", "constraint_gap_span_repair", task_score=0.4, trajectory_score=0.5),
        "repair": {
            "name": "constraint_gap_span_repair",
            "source_control": "low_confidence_32",
            "planning_span_target_scores": [
                {
                    "span": "Weak downstream sentence.",
                    "score": 2.5,
                    "source_relative_preservation": 0.8,
                    "prompt_gap_miss": 1.0,
                    "contradiction_relief": 0.16,
                    "keyword_coverage": 0.1,
                }
            ],
        },
    }

    rows = _planning_span_target_rows([record], [record])

    assert rows == [
        {
            "candidate_key": "model",
            "task_id": "plan",
            "repair": "constraint_gap_span_repair",
            "source_control": "low_confidence_32",
            "selected": True,
            "span": "Weak downstream sentence.",
            "score": 2.5,
            "source_relative_preservation": 0.8,
            "prompt_gap_miss": 1.0,
            "contradiction_relief": 0.16,
            "keyword_coverage": 0.1,
            "fallback": False,
        }
    ]


def test_planning_constraint_gap_span_targets_skip_numbered_list_markers():
    targets = _planning_constraint_gap_span_targets(
        "Compare baseline accuracy, prompt format, regressions, and public claim risk.",
        (
            "To falsify the result, 1. Increase the number of tokens used in the experiment. "
            "2. Compare the results to the original baseline to ensure the improvement is genuine."
        ),
        ["baseline", "prompt", "regressions", "claim", "risk"],
    )

    assert "2." not in targets
    assert targets


def test_planning_constraint_gap_span_targets_keep_abbreviations_intact():
    source = (
        "Measure offline accuracy and production latency. "
        "Decision rule: If the improvement is significant (e.g., above a threshold) "
        "and latency is acceptable, ship. Otherwise, gate the release."
    )

    targets = _planning_constraint_gap_span_targets(
        "Decide what to measure and what decision rule to use for a gated release.",
        source,
        ["decision", "rule", "gate"],
    )

    assert all(target != "g." for target in targets)
    assert all(not target.startswith(", above") for target in targets)
    assert any("e.g., above a threshold" in target for target in targets)


def test_constraint_gap_span_repair_masks_planning_gap_targets(tmp_path):
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    source_text = (
        "Run the baseline job first. "
        "If the baseline job fails, you can still run the intervention job, but the "
        "baseline data will not be available, making it a valid comparison. "
        "If the baseline job succeeds, you can then run the intervention job, ensuring "
        "you have a publishable result even if the intervention fails."
    )
    task = GeneralReasoningTask(
        task_id="plan_gap",
        family="planning",
        prompt=prompt,
        answer_type="rubric",
        scorer="planning_rubric_v1",
        max_new_tokens=8,
        rubric_items=("record enough metrics to explain failure modes",),
    )
    source = _record(
        "llada-8b-instruct-hf",
        "plan_gap",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["text"] = source_text
    source["generation_stage"] = "candidate_generation"
    source["generated_token_ids"] = [1, 2, 3, 4, 5, 6, 7, 8]
    source["generated_token_confidences"] = [0.9] * 8
    backend = _FakeExactRepairBackend(
        ["Run the baseline job first. Collect baseline metrics and failure-mode notes."],
        tokenizer=_TokenPiecesTokenizer(
            {
                1: "Run the baseline job first.",
                2: " If the baseline job fails,",
                3: " you can still run the intervention job,",
                4: " but the baseline data will not be available,",
                5: " making it a valid comparison.",
                6: " If the baseline job succeeds,",
                7: " you can then run the intervention job,",
                8: " ensuring you have a publishable result even if the intervention fails.",
            }
        ),
    )
    repair = _repair_candidates(
        repair_pack="constraint_gap",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=5,
    )[4]

    records = _generate_repair_records(
        backend,
        task,
        source_record=source,
        repairs=(repair,),
        generation_seed_base=11,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
    )

    assert backend.configs[0].initial_suffix_token_ids == (1, None, None, None, None, None, None, None)
    assert "Preserved opening from the previous draft:" in backend.prompts[0]
    assert "Draft answer to repair:" not in backend.prompts[0]
    assert "Do not claim the comparison is valid" in backend.prompts[0]
    assert records[0]["repair"]["uses_planning_span_revision"] is True
    assert records[0]["repair"]["seed_masked_positions"] == 7
    assert records[0]["repair"]["span_localization_mode"] == "literal_span"
    assert records[0]["repair"]["span_literal_target_found"] is True
    assert records[0]["repair"]["span_fallback_used"] is False
    assert records[0]["repair"]["span_seed_diagnostics"]["masked_positions"] == [1, 2, 3, 4, 5, 6, 7]
    assert records[0]["repair"]["planning_span_targets"] == _planning_constraint_gap_span_targets(
        prompt,
        source_text,
        records[0]["repair"]["prompt_constraint_gap_terms"],
    )
    assert records[0]["repair"]["planning_span_target_scores"]
    assert records[0]["repair"]["planning_span_target_scores"][0]["score"] > 0.0


def test_phase_final_span_repair_runs_without_history_source(tmp_path):
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    source_text = (
        "Run the baseline job first. If it works, run the intervention. If it fails, "
        "stop and publish the baseline note."
    )
    task = GeneralReasoningTask(
        task_id="plan_phase_final",
        family="planning",
        prompt=prompt,
        answer_type="rubric",
        scorer="planning_rubric_v1",
        max_new_tokens=8,
        rubric_items=("measurements", "fallback", "failure modes"),
    )
    source = _record(
        "llada-8b-instruct-hf",
        "plan_phase_final",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["text"] = source_text
    source["generation_stage"] = "candidate_generation"
    source["generated_token_ids"] = [1, 2, 3, 4, 5, 6, 7, 8]
    source["generated_token_confidences"] = [0.9] * 8
    backend = _FakeExactRepairBackend(
        ["Run the baseline first, collect measurements, and preserve a fallback."],
        tokenizer=_TokenPiecesTokenizer(
            {
                1: "Run the baseline job first.",
                2: " If it works,",
                3: " run the intervention.",
                4: " If it fails,",
                5: " stop and publish",
                6: " the baseline note.",
                7: " Missing detail.",
                8: " Missing fallback.",
            }
        ),
    )
    repair = _repair_candidates(
        repair_pack="constraint_span_phase_final_preserve_seeded_gated",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    records = _generate_repair_records(
        backend,
        task,
        source_record=source,
        repairs=(repair,),
        generation_seed_base=11,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
    )

    assert records[0]["repair"]["source_state"] == "final"
    assert records[0]["repair"]["source_history_step"] is None
    assert records[0]["repair"]["uses_planning_span_revision"] is True
    assert records[0]["repair"]["execution_repair_name"] == "constraint_gap_span_repair"
    assert records[0]["repair"]["generation_seed_repair_name"] == "constraint_gap_span_repair"
    assert records[0]["repair"]["seed_masked_positions"] > 0


def test_constraint_gap_span_history_repair_uses_history_state_tokens_and_text(tmp_path):
    prompt = (
        "Run a baseline and risky intervention overnight. Collect measurements, "
        "set rollback thresholds, preserve failure evidence, and keep a fallback."
    )
    final_text = (
        "Run the baseline, compare the intervention, collect measurements, set "
        "rollback thresholds, preserve failure evidence, and keep a fallback."
    )
    history_text = (
        "Run the baseline first. Compare the intervention vaguely. Stop after the jobs."
    )
    task = GeneralReasoningTask(
        task_id="plan_history_gap",
        family="planning",
        prompt=prompt,
        answer_type="rubric",
        scorer="planning_rubric_v1",
        max_new_tokens=6,
        rubric_items=("measurements", "rollback thresholds", "fallback"),
    )
    source = _record(
        "llada-8b-instruct-hf",
        "plan_history_gap",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["text"] = final_text
    source["generation_stage"] = "candidate_generation"
    source["generated_token_ids"] = [10, 11, 12, 13, 14, 15]
    source["generated_token_confidences"] = [0.9] * 6
    source["history_samples"] = [
        {"step": 8, "generated_token_ids": [1, 2, 3, 126336, 126336, 126336]}
    ]
    source["trajectory_summary"] = {
        "samples": [
            {
                "step": 8,
                "mask_count": 3,
                "visible_chars": len(history_text),
                "visible_text": history_text,
            }
        ]
    }
    backend = _FakeExactRepairBackend(
        ["Run the baseline first. Add measurements, rollback thresholds, and fallback evidence."],
        tokenizer=_TokenPiecesTokenizer(
            {
                1: "Run the baseline first.",
                2: " Compare the intervention vaguely.",
                3: " Stop after the jobs.",
                10: "Final token ten.",
                11: " Final token eleven.",
                12: " Final token twelve.",
                13: " Final token thirteen.",
                14: " Final token fourteen.",
                15: " Final token fifteen.",
                126336: "",
            }
        ),
    )
    repair = _repair_candidates(
        repair_pack="constraint_span_history",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    records = _generate_repair_records(
        backend,
        task,
        source_record=source,
        repairs=(repair,),
        generation_seed_base=11,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
    )

    seed = backend.configs[0].initial_suffix_token_ids
    assert seed[:3] != tuple(source["generated_token_ids"][:3])
    assert seed[0] == 1
    assert records[0]["repair"]["source_state"] == "history"
    assert records[0]["repair"]["source_history_step"] == 8
    assert records[0]["repair"]["repair_source_text_chars"] == len(history_text)
    assert records[0]["repair"]["planning_span_targets"] == _planning_constraint_gap_span_targets(
        prompt,
        history_text,
        records[0]["repair"]["prompt_constraint_gap_terms"],
    )
    assert "Run the baseline first" in backend.prompts[0]
    assert "Final token" not in backend.prompts[0]


def test_pre_generation_anchor_selector_prefers_clean_single_span_history_anchor():
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    final_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either the baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    history_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful ( the baseline or the intervention) "
        "can be published tomorrow, ensuring a publishable result even if the "
        "intervention fails."
    )
    source = _record(
        "llada-8b-instruct-hf",
        "plan_anchor_select",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["text"] = final_text
    source["generated_token_ids"] = [10, 11, 12]
    source["history_samples"] = [{"step": 31, "generated_token_ids": [1, 2, 126336]}]
    source["trajectory_summary"] = {
        "samples": [
            {
                "step": 31,
                "mask_count": 1,
                "visible_chars": len(history_text),
                "visible_text": history_text,
            }
        ]
    }

    choice = _choose_pre_generation_repair_anchor(source, prompt)

    assert choice["anchor_choice"] == "history"
    assert choice["reason"] == "history_single_span_score_advantage"
    assert choice["features"]["history_target_count"] == 1


def test_pre_generation_phase_hybrid_uses_history_only_with_source_advantage():
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    final_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either the baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    history_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful ( the baseline or the intervention) "
        "can be published tomorrow, ensuring a publishable result even if the "
        "intervention fails."
    )
    source = _record(
        "llada-8b-instruct-hf",
        "plan_phase_hybrid",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["text"] = final_text
    source["generated_token_ids"] = [10, 11, 12]
    source["history_samples"] = [{"step": 31, "generated_token_ids": [1, 2, 126336]}]
    source["trajectory_summary"] = {
        "samples": [
            {
                "step": 31,
                "mask_count": 1,
                "visible_chars": len(history_text),
                "visible_text": history_text,
            }
        ]
    }

    choice = _choose_pre_generation_repair_anchor(source, prompt, phase_hybrid=True)

    assert choice["anchor_choice"] == "history"
    assert choice["reason"] == "phase_hybrid_history_source_advantage"
    assert choice["features"]["history_span_score_delta"] > 0
    assert choice["features"]["phase_first_repairable_step"] == 31
    assert choice["features"]["phase_first_safe_repairable_step"] == 31
    assert choice["features"]["phase_source_target_similarity_min"] == 0.96
    assert choice["features"]["phase_source_text_similarity_min"] == 0.96
    assert choice["history_sample"]["step"] == 31

    stricter_choice = _choose_pre_generation_repair_anchor(
        source,
        prompt,
        phase_hybrid=True,
        phase_source_text_similarity_min=1.01,
    )

    assert stricter_choice["anchor_choice"] == "final"
    assert stricter_choice["reason"] == "phase_hybrid_final_no_source_advantage"
    assert stricter_choice["features"]["phase_source_text_similarity_min"] == 1.01


def test_phase_hybrid_source_advantage_requires_strict_retention():
    loose_phase_features = {
        "history_repairable_denoise_skeleton": True,
        "history_target_count": 1,
        "final_target_count": 1,
        "text_similarity": 0.94,
        "target_similarity": 0.943503,
        "history_to_final_char_ratio": 0.908714,
        "lost_digit_token_count": 0,
        "lost_prompt_keyword_count": 0,
        "history_span_score_delta": 0.006615,
    }
    strict_phase_features = {
        **loose_phase_features,
        "text_similarity": 0.979969,
        "target_similarity": 0.960486,
        "history_to_final_char_ratio": 0.960725,
    }
    weak_text_features = {
        **strict_phase_features,
        "text_similarity": 0.94,
    }

    assert _phase_history_anchor_has_source_advantage(loose_phase_features) is False
    assert _phase_history_anchor_has_source_advantage(strict_phase_features) is True
    assert _phase_history_anchor_passes_source_policy(weak_text_features) is False
    assert _phase_history_anchor_has_source_advantage(weak_text_features) is False


def test_pre_generation_anchor_search_scans_all_history_for_retention_safe_anchor():
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    final_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either the baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    unsafe_history_text = "Collect the baseline measurement first. Publish a generic answer tomorrow."
    safe_history_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful ( the baseline or the intervention) "
        "can be published tomorrow, ensuring a publishable result even if the "
        "intervention fails."
    )
    source = _record(
        "llada-8b-instruct-hf",
        "plan_anchor_search",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["text"] = final_text
    source["generated_token_ids"] = [10, 11, 12]
    source["history_samples"] = [
        {"step": 12, "generated_token_ids": [1, 2, 126336]},
        {"step": 30, "generated_token_ids": [3, 4, 126336]},
    ]
    source["trajectory_summary"] = {
        "samples": [
            {
                "step": 12,
                "mask_count": 20,
                "visible_chars": len(unsafe_history_text),
                "visible_text": unsafe_history_text,
            },
            {
                "step": 30,
                "mask_count": 2,
                "visible_chars": len(safe_history_text),
                "visible_text": safe_history_text,
            },
        ]
    }

    choice = _choose_pre_generation_repair_anchor(source, prompt, search_history=True)

    assert choice["anchor_choice"] == "history"
    assert choice["reason"] == "history_search_retention_loss_minimum"
    assert choice["features"]["history_step"] == 30
    assert choice["features"]["history_retention_loss"] == 0.289514
    assert choice["history_sample"]["step"] == 30


def test_pre_generation_phase_anchor_uses_first_safe_repairable_skeleton():
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    final_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either the baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    early_safe_history_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    later_safe_history_text = final_text.replace("result (either", "result, either")
    source = _record(
        "llada-8b-instruct-hf",
        "plan_phase_anchor",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["text"] = final_text
    source["generated_token_ids"] = [10, 11, 12]
    source["history_samples"] = [
        {"step": 12, "generated_token_ids": [1, 2, 126336]},
        {"step": 30, "generated_token_ids": [3, 4, 126336]},
    ]
    source["trajectory_summary"] = {
        "samples": [
            {
                "step": 12,
                "mask_count": 20,
                "visible_chars": len(early_safe_history_text),
                "visible_text": early_safe_history_text,
            },
            {
                "step": 30,
                "mask_count": 2,
                "visible_chars": len(later_safe_history_text),
                "visible_text": later_safe_history_text,
            },
        ]
    }

    choice = _choose_pre_generation_repair_anchor(source, prompt, phase_anchor=True)

    assert choice["anchor_choice"] == "history"
    assert choice["reason"] == "history_phase_first_repairable_skeleton"
    assert choice["features"]["history_step"] == 12
    assert choice["features"]["history_repairable_denoise_skeleton"] is True
    assert choice["features"]["history_prompt_coverage"] >= 0.4
    assert choice["features"]["history_span_score_delta"] == 0.0
    assert choice["features"]["phase_repairable_sample_count"] == 2
    assert choice["features"]["phase_safe_repairable_sample_count"] == 2
    assert choice["features"]["phase_first_repairable_step"] == 12
    assert choice["features"]["phase_first_safe_repairable_step"] == 12
    assert choice["features"]["phase_retention_safety_lag"] == 0
    assert choice["history_sample"]["step"] == 12


def test_pre_generation_phase_hybrid_keeps_final_without_source_advantage():
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    final_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either the baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    early_safe_history_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    source = _record(
        "llada-8b-instruct-hf",
        "plan_phase_hybrid",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["text"] = final_text
    source["generated_token_ids"] = [10, 11, 12]
    source["history_samples"] = [{"step": 12, "generated_token_ids": [1, 2, 126336]}]
    source["trajectory_summary"] = {
        "samples": [
            {
                "step": 12,
                "mask_count": 20,
                "visible_chars": len(early_safe_history_text),
                "visible_text": early_safe_history_text,
            }
        ]
    }

    choice = _choose_pre_generation_repair_anchor(source, prompt, phase_hybrid=True)

    assert choice["anchor_choice"] == "final"
    assert choice["reason"] == "phase_hybrid_final_no_source_advantage"
    assert choice["features"]["history_repairable_denoise_skeleton"] is True
    assert choice["features"]["history_span_score_delta"] == 0.0
    assert choice["features"]["phase_first_repairable_step"] == 12
    assert choice["features"]["phase_first_safe_repairable_step"] == 12
    assert choice["features"]["phase_retention_safety_lag"] == 0


def test_pre_generation_phase_anchor_records_retention_safety_lag():
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    final_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either the baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    early_repairable_but_unsafe_text = (
        "Run two GPU jobs overnight: collect baseline measurements, test the risky "
        "intervention, and make tomorrow publishable if the intervention fails."
    )
    later_safe_history_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    source = _record(
        "llada-8b-instruct-hf",
        "plan_phase_anchor",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["text"] = final_text
    source["generated_token_ids"] = [10, 11, 12]
    source["history_samples"] = [
        {"step": 10, "generated_token_ids": [1, 2, 126336]},
        {"step": 30, "generated_token_ids": [3, 4, 126336]},
    ]
    source["trajectory_summary"] = {
        "samples": [
            {
                "step": 10,
                "mask_count": 20,
                "visible_chars": len(early_repairable_but_unsafe_text),
                "visible_text": early_repairable_but_unsafe_text,
            },
            {
                "step": 30,
                "mask_count": 2,
                "visible_chars": len(later_safe_history_text),
                "visible_text": later_safe_history_text,
            },
        ]
    }

    choice = _choose_pre_generation_repair_anchor(source, prompt, phase_anchor=True)

    assert choice["anchor_choice"] == "history"
    assert choice["reason"] == "history_phase_first_repairable_skeleton"
    assert choice["features"]["history_step"] == 30
    assert choice["features"]["phase_repairable_sample_count"] == 2
    assert choice["features"]["phase_safe_repairable_sample_count"] == 1
    assert choice["features"]["phase_first_repairable_step"] == 10
    assert choice["features"]["phase_first_safe_repairable_step"] == 30
    assert choice["features"]["phase_retention_safety_lag"] == 20
    assert choice["history_sample"]["step"] == 30


def test_planning_span_history_contrast_returns_compact_near_final_history():
    prompt = "Compare baseline accuracy, latency, rollback threshold, and publishable evidence."
    source_text = (
        "Run the baseline first. Decision rule: if accuracy improves by 10% or latency "
        "increases by under 50%, ship; otherwise use the rollback threshold."
    )
    weak_text = "Run a test and decide later."
    contrast_text = (
        "Run the baseline first. Decision rule: if accuracy improves by 10% or latency "
        "increases by under 50%, ship; otherwise use the rollback threshold and record "
        "publishable evidence."
    )
    source = _record(
        "llada-8b-instruct-hf",
        "plan_history_contrast",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["history_samples"] = [
        {"step": 4, "generated_token_ids": [1, 2, 126336]},
        {"step": 30, "generated_token_ids": [3, 4, 126336]},
    ]
    source["trajectory_summary"] = {
        "samples": [
            {
                "step": 4,
                "mask_count": 30,
                "visible_chars": len(weak_text),
                "visible_text": weak_text,
            },
            {
                "step": 30,
                "mask_count": 2,
                "visible_chars": len(contrast_text),
                "visible_text": contrast_text,
            },
        ]
    }

    contrast = _planning_span_history_contrast(
        source,
        task_prompt=prompt,
        source_text=source_text,
        span_targets=["Decision rule: if accuracy improves by 10%"],
    )

    assert contrast.startswith("history step 30:")
    assert "rollback threshold" in contrast
    assert weak_text not in contrast


def test_anchor_select_span_repair_resolves_to_history_tokens_when_geometry_prefers_history(tmp_path):
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    final_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either the baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    history_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful ( the baseline or the intervention) "
        "can be published tomorrow, ensuring a publishable result even if the "
        "intervention fails."
    )
    task = GeneralReasoningTask(
        task_id="plan_anchor_select",
        family="planning",
        prompt=prompt,
        answer_type="rubric",
        scorer="planning_rubric_v1",
        max_new_tokens=6,
        rubric_items=("baseline", "measurements", "intervention"),
    )
    source = _record(
        "llada-8b-instruct-hf",
        "plan_anchor_select",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["text"] = final_text
    source["generation_stage"] = "candidate_generation"
    source["generated_token_ids"] = [10, 11, 12, 13, 14, 15]
    source["generated_token_confidences"] = [0.9] * 6
    source["history_samples"] = [
        {"step": 31, "generated_token_ids": [1, 2, 3, 4, 5, 126336]}
    ]
    source["trajectory_summary"] = {
        "samples": [
            {
                "step": 31,
                "mask_count": 1,
                "visible_chars": len(history_text),
                "visible_text": history_text,
            }
        ]
    }
    backend = _FakeExactRepairBackend(
        ["Collect the baseline measurement first, then repair the weak span."],
        tokenizer=_TokenPiecesTokenizer(
            {
                1: "Collect the baseline measurement first.",
                2: " If it is successful, proceed to run the risky intervention job.",
                3: " If the baseline fails, do the intervention job instead.",
                4: " This way, at least one successful ( the baseline or the intervention)",
                5: " can be published tomorrow, ensuring a publishable result even if the intervention fails.",
                10: "Final token ten.",
                11: " Final token eleven.",
                12: " Final token twelve.",
                13: " Final token thirteen.",
                14: " Final token fourteen.",
                15: " Final token fifteen.",
                126336: "",
            }
        ),
    )
    repair = _repair_candidates(
        repair_pack="constraint_span_anchor_select",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    records = _generate_repair_records(
        backend,
        task,
        source_record=source,
        repairs=(repair,),
        generation_seed_base=11,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
    )

    assert records[0]["repair"]["configured_source_state"] == "pre_generation_anchor"
    assert records[0]["repair"]["execution_repair_name"] == "constraint_gap_span_history_repair"
    assert records[0]["repair"]["generation_seed_repair_name"] == "constraint_gap_span_history_repair"
    assert records[0]["repair"]["source_state"] == "history"
    assert records[0]["repair"]["source_history_step"] == 31
    assert records[0]["repair"]["anchor_selection_reason"] == "history_single_span_score_advantage"
    assert "Final token" not in backend.prompts[0]


def test_phase_anchor_span_repair_resolves_to_first_repairable_skeleton(tmp_path):
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    final_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either the baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    history_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful ( the baseline or the intervention) "
        "can be published tomorrow, ensuring a publishable result even if the "
        "intervention fails."
    )
    task = GeneralReasoningTask(
        task_id="plan_phase_anchor",
        family="planning",
        prompt=prompt,
        answer_type="rubric",
        scorer="planning_rubric_v1",
        max_new_tokens=6,
        rubric_items=("baseline", "measurements", "intervention"),
    )
    source = _record(
        "llada-8b-instruct-hf",
        "plan_phase_anchor",
        "baseline",
        task_score=0.0,
        trajectory_score=0.4,
    )
    source["text"] = final_text
    source["generation_stage"] = "candidate_generation"
    source["generated_token_ids"] = [10, 11, 12, 13, 14, 15]
    source["generated_token_confidences"] = [0.9] * 6
    source["history_samples"] = [
        {"step": 12, "generated_token_ids": [1, 2, 3, 4, 5, 126336]}
    ]
    source["trajectory_summary"] = {
        "samples": [
            {
                "step": 12,
                "mask_count": 1,
                "visible_chars": len(history_text),
                "visible_text": history_text,
            }
        ]
    }
    backend = _FakeExactRepairBackend(
        ["Collect the baseline measurement first, then repair the weak phase span."],
        tokenizer=_TokenPiecesTokenizer(
            {
                1: "Collect the baseline measurement first.",
                2: " If it is successful, proceed to run the risky intervention job.",
                3: " If the baseline fails, do the intervention job instead.",
                4: " This way, at least one successful ( the baseline or the intervention)",
                5: " can be published tomorrow, ensuring a publishable result even if the intervention fails.",
                10: "Final token ten.",
                11: " Final token eleven.",
                12: " Final token twelve.",
                13: " Final token thirteen.",
                14: " Final token fourteen.",
                15: " Final token fifteen.",
                126336: "",
            }
        ),
    )
    repair = _repair_candidates(
        repair_pack="constraint_span_phase_anchor",
        include_history_repairs=False,
        history_repair_fractions=(),
        include_history_visible_repair=False,
        limit=1,
    )[0]

    records = _generate_repair_records(
        backend,
        task,
        source_record=source,
        repairs=(repair,),
        generation_seed_base=11,
        raw_output=tmp_path / "raw.jsonl",
        all_records=[],
    )

    assert records[0]["repair"]["configured_source_state"] == "pre_generation_phase_anchor"
    assert records[0]["repair"]["execution_repair_name"] == "constraint_gap_span_history_repair"
    assert records[0]["repair"]["source_state"] == "history"
    assert records[0]["repair"]["source_history_step"] == 12
    assert records[0]["repair"]["anchor_selection_reason"] == "history_phase_first_repairable_skeleton"
    assert records[0]["repair"]["anchor_selection_features"]["history_repairable_denoise_skeleton"] is True
    assert records[0]["repair"]["anchor_selection_features"]["phase_first_repairable_step"] == 12
    assert records[0]["repair"]["anchor_selection_features"]["phase_first_safe_repairable_step"] == 12
    assert records[0]["repair"]["anchor_selection_features"]["phase_retention_safety_lag"] == 0
    assert "Final token" not in backend.prompts[0]


def test_constraint_gap_terms_are_empty_for_non_gap_repair():
    terms = _prompt_constraint_gap_terms(
        "Compare baseline and rollback threshold.",
        "Measure accuracy only.",
        DiffusionRepairCandidate(name="plain"),
    )

    assert terms == []


def test_prompt_guided_rescue_candidates_exclude_existing_repairs():
    existing = (
        DiffusionRepairCandidate(name="prompt_guided_revision_repair"),
        DiffusionRepairCandidate(name="targeted_filler_repair"),
    )

    rescue = _prompt_guided_rescue_candidates(existing_repairs=existing, limit=2)

    assert [repair.name for repair in rescue] == ["prompt_guided_revision_anchor25_repair"]


def test_constraint_gap_rescue_candidates_only_include_gap_repairs():
    existing = (
        DiffusionRepairCandidate(name="state_adaptive_history_repair"),
        DiffusionRepairCandidate(name="prefix_25_repair"),
    )

    rescue = _constraint_gap_rescue_candidates(existing_repairs=existing, limit=3)

    assert [repair.name for repair in rescue] == [
        "constraint_gap_revision_repair",
        "constraint_gap_revision_anchor25_repair",
        "constraint_gap_span_repair",
    ]


def test_primary_repair_gate_skips_complete_high_quality_source():
    source = {
        **_record("model", "plan", "evolved_low_confidence_48", task_score=0.0, trajectory_score=0.5),
        "text": (
            "Compare baseline and intervention, record metrics, preserve rollback, "
            "define a threshold, monitor risk, and include fallback actions. "
            "This complete plan has enough operational detail for selection."
        ),
    }

    assert not _should_run_primary_repair_pass(
        trigger="source_quality_or_short",
        source_record=source,
        source_controls=[],
        task_prompt="Compare a baseline and intervention, preserve rollback, record metrics, and define a threshold.",
        task_answer_type="rubric",
        source_quality_threshold=0.40,
        source_min_chars=80,
    )


def test_primary_repair_gate_runs_for_short_or_low_quality_source():
    short_source = {
        **_record("model", "plan", "evolved_low_confidence_48", task_score=0.0, trajectory_score=0.5),
        "text": "Compare baseline and intervention.",
    }
    weak_source = {
        **_record("model", "plan", "evolved_low_confidence_48", task_score=0.0, trajectory_score=0.5),
        "text": "Generic plan. More generic notes that avoid metrics, rollback, risks, or thresholds.",
    }

    assert _should_run_primary_repair_pass(
        trigger="source_quality_or_short",
        source_record=short_source,
        source_controls=[],
        task_prompt="Compare a baseline and intervention, preserve rollback, record metrics, and define a threshold.",
        task_answer_type="rubric",
        source_quality_threshold=0.40,
        source_min_chars=80,
    )
    assert _should_run_primary_repair_pass(
        trigger="source_quality_or_short",
        source_record=weak_source,
        source_controls=[],
        task_prompt="Compare a baseline and intervention, preserve rollback, record metrics, and define a threshold.",
        task_answer_type="rubric",
        source_quality_threshold=0.95,
        source_min_chars=40,
    )


def test_primary_repair_geometry_gate_requires_repairable_prompt_gap_band():
    prompt = (
        "Run two GPU jobs overnight: one reliable baseline and one risky reasoning "
        "intervention. Collect measurements, failure evidence, and a publishable fallback."
    )
    productive_source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.5),
        "text": (
            "Collect the baseline measurement first, then run the intervention. "
            "If the intervention fails, record failure evidence and keep a fallback."
        ),
    }
    under_grounded_source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.5),
        "text": "Make a generic plan with a few notes.",
    }
    overloaded_source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.5),
        "text": "Collect the baseline.",
    }

    assert _should_run_primary_repair_pass(
        trigger="source_repairability_geometry",
        source_record=productive_source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
    )
    assert not _should_run_primary_repair_pass(
        trigger="source_repairability_geometry",
        source_record=under_grounded_source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
    )
    assert not _should_run_primary_repair_pass(
        trigger="source_repairability_geometry",
        source_record=overloaded_source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=4,
        source_prompt_coverage_min=0.05,
        source_prompt_coverage_max=1.00,
    )


def test_primary_repair_denoise_phase_gate_requires_history_skeleton():
    prompt = (
        "Run two GPU jobs overnight: one reliable baseline and one risky reasoning "
        "intervention. Collect measurements, failure evidence, and a publishable fallback."
    )
    final_text = (
        "Collect the baseline measurement first, then run the intervention. "
        "If the intervention fails, record failure evidence and keep a fallback."
    )
    source_with_skeleton = {
        **_record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.5),
        "text": final_text,
        "trajectory_summary": {
            "samples": [
                {"step": 1, "visible_chars": 0, "visible_text": ""},
                {
                    "step": 8,
                    "visible_chars": 55,
                    "visible_text": "Collect the baseline measurement and failure evidence.",
                },
            ]
        },
    }
    source_without_history = {
        **_record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.5),
        "text": final_text,
    }
    source_without_skeleton = {
        **source_with_skeleton,
        "trajectory_summary": {
            "samples": [
                {"step": 1, "visible_chars": 0, "visible_text": ""},
                {"step": 8, "visible_chars": 31, "visible_text": "Write a short generic response."},
            ]
        },
    }

    assert _should_run_primary_repair_pass(
        trigger="denoise_phase_repairability",
        source_record=source_with_skeleton,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
    )
    assert _should_run_primary_repair_pass(
        trigger="denoise_phase_repairability",
        source_record=source_with_skeleton,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=8,
    )
    assert not _should_run_primary_repair_pass(
        trigger="denoise_phase_repairability",
        source_record=source_with_skeleton,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=7,
    )
    assert not _should_run_primary_repair_pass(
        trigger="denoise_phase_repairability",
        source_record=source_without_history,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
    )
    assert not _should_run_primary_repair_pass(
        trigger="denoise_phase_repairability",
        source_record=source_without_skeleton,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
    )


def test_primary_repair_gate_diagnostics_explain_denoise_phase_skip():
    prompt = (
        "Run two GPU jobs overnight: one reliable baseline and one risky reasoning "
        "intervention. Collect measurements, failure evidence, and a publishable fallback."
    )
    source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.5),
        "text": (
            "Collect the baseline measurement first, then run the intervention. "
            "If the intervention fails, record failure evidence and keep a fallback."
        ),
        "trajectory_summary": {
            "samples": [
                {"step": 8, "visible_chars": 31, "visible_text": "Write a short generic response."},
            ]
        },
    }

    diagnostics = _primary_repair_gate_diagnostics(
        trigger="denoise_phase_repairability",
        source_record=source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
    )

    assert diagnostics["should_run"] is False
    assert diagnostics["reason"] == "no_repairable_denoise_skeleton"
    assert diagnostics["source_needs_repair"] is True
    assert diagnostics["in_repairable_band"] is True
    assert diagnostics["has_repairable_denoise_skeleton"] is False
    assert diagnostics["prompt_gap_count"] >= 2
    assert diagnostics["prompt_coverage"] >= 0.30
    assert diagnostics["first_repairable_denoise_skeleton_step"] is None
    assert diagnostics["peak_denoise_prompt_coverage"] < 0.30


def test_primary_repair_gate_diagnostics_track_denoise_phase_features():
    prompt = (
        "Run two GPU jobs overnight: one reliable baseline and one risky reasoning "
        "intervention. Collect measurements, failure evidence, and a publishable fallback."
    )
    source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.5),
        "history_steps": 16,
        "text": (
            "Run baseline and intervention, collect measurements and failure evidence."
        ),
        "trajectory_summary": {
            "samples": [
                {"step": 4, "visible_chars": 24, "visible_text": "Collect the baseline."},
                {
                    "step": 8,
                    "visible_chars": 55,
                    "visible_text": "Collect the baseline measurement and failure evidence.",
                },
            ]
        },
    }

    diagnostics = _primary_repair_gate_diagnostics(
        trigger="denoise_phase_repairability",
        source_record=source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
    )

    assert diagnostics["should_run"] is True
    assert diagnostics["reason"] == "denoise_phase_repairable"
    assert diagnostics["denoise_history_steps"] == 16
    assert diagnostics["first_repairable_denoise_skeleton_step"] == 8
    assert diagnostics["first_repairable_denoise_skeleton_step_fraction"] == 0.5
    assert diagnostics["first_repairable_denoise_skeleton_visible_chars"] == len(
        "Collect the baseline measurement and failure evidence."
    )
    assert diagnostics["denoise_skeleton_within_max_step"] is True


def test_primary_repair_gate_diagnostics_apply_denoise_value_proxy():
    prompt = (
        "Run two GPU jobs overnight: one reliable baseline and one risky reasoning "
        "intervention. Collect measurements, failure evidence, and a publishable fallback."
    )
    source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.5),
        "history_steps": 16,
        "text": (
            "Collect the baseline measurement first, then run the intervention. "
            "If the intervention fails, record failure evidence and keep a fallback."
        ),
        "trajectory_summary": {
            "samples": [
                {"step": 4, "visible_chars": 24, "visible_text": "Collect the baseline."},
                {
                    "step": 8,
                    "visible_chars": 55,
                    "visible_text": "Collect the baseline measurement and failure evidence.",
                },
            ]
        },
    }

    run_diagnostics = _primary_repair_gate_diagnostics(
        trigger="denoise_phase_value_proxy",
        source_record=source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=1.0,
    )
    skip_diagnostics = _primary_repair_gate_diagnostics(
        trigger="denoise_phase_value_proxy",
        source_record=source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=0.0,
    )

    assert run_diagnostics["should_run"] is True
    assert run_diagnostics["reason"] == "denoise_phase_value_proxy"
    assert run_diagnostics["value_proxy_source_quality_max"] == 1.0
    assert skip_diagnostics["should_run"] is False
    assert skip_diagnostics["reason"] == "value_proxy_source_quality_high"
    assert skip_diagnostics["source_quality"] > 0.0


def test_primary_repair_gate_diagnostics_apply_decomposed_four_head_selector():
    prompt = (
        "Run two GPU jobs overnight: one reliable baseline and one risky reasoning "
        "intervention. Collect measurements, failure evidence, and a publishable fallback."
    )
    source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.5),
        "history_steps": 16,
        "text": (
            "Collect the baseline measurement first, then run the intervention. "
            "If the intervention fails, record failure evidence and keep a fallback."
        ),
        "trajectory_summary": {
            "samples": [
                {"step": 4, "visible_chars": 24, "visible_text": "Collect the baseline."},
                {
                    "step": 8,
                    "visible_chars": 55,
                    "visible_text": "Collect the baseline measurement and failure evidence.",
                },
            ]
        },
    }

    run_diagnostics = _primary_repair_gate_diagnostics(
        trigger="decomposed_four_head_selector",
        source_record=source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=1.0,
    )
    skip_diagnostics = _primary_repair_gate_diagnostics(
        trigger="decomposed_four_head_selector",
        source_record=source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=0.0,
    )
    outside_band_diagnostics = _primary_repair_gate_diagnostics(
        trigger="decomposed_four_head_selector",
        source_record=source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=20,
        source_prompt_gap_max=30,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=1.0,
    )

    assert run_diagnostics["should_run"] is True
    assert run_diagnostics["reason"] == "decomposed_four_head_selector"
    assert run_diagnostics["composite_selector_id"] == "decomposed_four_head_selector"
    assert run_diagnostics["spend_head_rule_id"] == (
        "first_repairable_gap_le_9_source_quality_le_0p301429"
    )
    assert run_diagnostics["source_head_rule_id"] == "retention_safe_history"
    assert run_diagnostics["retention_head_rule_id"] == "classification_safe_history_anchor"
    assert run_diagnostics["realization_head_rule_id"] == "min_realization_policy_error"
    assert run_diagnostics["spend_head_prediction"] is True
    assert skip_diagnostics["should_run"] is False
    assert skip_diagnostics["reason"] == "value_proxy_source_quality_high"
    assert skip_diagnostics["spend_head_prediction"] is False
    assert outside_band_diagnostics["should_run"] is False
    assert outside_band_diagnostics["reason"] == "outside_repairable_band"
    assert outside_band_diagnostics["composite_selector_id"] == "decomposed_four_head_selector"
    assert outside_band_diagnostics["spend_head_rule_id"] == (
        "first_repairable_gap_le_9_source_quality_le_0p301429"
    )
    assert outside_band_diagnostics["spend_head_prediction"] is False


def test_primary_repair_gate_diagnostics_apply_decomposed_spend_transfer_rule():
    prompt = (
        "Run two GPU jobs overnight: one reliable baseline and one risky reasoning "
        "intervention. Collect measurements, failure evidence, and a publishable fallback."
    )
    source_base = {
        "history_steps": 16,
        "text": "Run baseline and intervention, collect measurements and failure evidence.",
        "trajectory_summary": {
            "samples": [
                {"step": 4, "visible_chars": 24, "visible_text": "Collect the baseline."},
                {
                    "step": 8,
                    "visible_chars": 55,
                    "visible_text": "Collect the baseline measurement and failure evidence.",
                },
            ]
        },
    }
    high_source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.40, trajectory_score=0.5),
        **source_base,
    }
    low_source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.20, trajectory_score=0.5),
        **source_base,
    }

    run_diagnostics = _primary_repair_gate_diagnostics(
        trigger="decomposed_spend_transfer_rule",
        source_record=high_source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=1.0,
        transfer_source_task_min=0.295357,
    )
    skip_diagnostics = _primary_repair_gate_diagnostics(
        trigger="decomposed_spend_transfer_rule",
        source_record=low_source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=1.0,
        transfer_source_task_min=0.295357,
    )

    assert run_diagnostics["should_run"] is True
    assert run_diagnostics["reason"] == "decomposed_spend_transfer_rule"
    assert run_diagnostics["composite_selector_id"] == "decomposed_spend_transfer_rule"
    assert run_diagnostics["spend_head_rule_id"] == "current_decomposed_spend_source_task_ge_0p295357"
    assert run_diagnostics["spend_head_source_task_min"] == 0.295357
    assert run_diagnostics["source_task_score"] == 0.40
    assert run_diagnostics["spend_head_prediction"] is True
    assert skip_diagnostics["should_run"] is False
    assert skip_diagnostics["reason"] == "transfer_source_task_score_low"
    assert skip_diagnostics["source_task_score"] == 0.20
    assert skip_diagnostics["spend_head_prediction"] is False


def test_primary_repair_gate_diagnostics_apply_trajectory_relative_spend_rule():
    prompt = (
        "Run two GPU jobs overnight: one reliable baseline and one risky reasoning "
        "intervention. Collect measurements, failure evidence, and a publishable fallback."
    )
    source_base = {
        "history_steps": 16,
        "text": "Run baseline and intervention, collect measurements and failure evidence.",
        "trajectory_summary": {
            "samples": [
                {"step": 4, "visible_chars": 24, "visible_text": "Collect the baseline."},
                {
                    "step": 8,
                    "visible_chars": 55,
                    "visible_text": "Collect the baseline measurement and failure evidence.",
                },
            ]
        },
    }
    source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.32, trajectory_score=0.5),
        **source_base,
    }
    better_trajectory = _record(
        "model",
        "plan",
        "random_32",
        task_score=0.40,
        trajectory_score=0.5,
    )
    equal_trajectory = _record(
        "model",
        "plan",
        "low_confidence_32",
        task_score=0.32,
        trajectory_score=0.5,
    )

    skip_diagnostics = _primary_repair_gate_diagnostics(
        trigger="trajectory_relative_decomposed_spend",
        source_record=source,
        trajectory_record=better_trajectory,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=1.0,
        transfer_source_task_min=0.295357,
    )
    run_diagnostics = _primary_repair_gate_diagnostics(
        trigger="trajectory_relative_decomposed_spend",
        source_record=source,
        trajectory_record=equal_trajectory,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=1.0,
        transfer_source_task_min=0.295357,
    )

    assert skip_diagnostics["should_run"] is False
    assert skip_diagnostics["reason"] == "source_below_trajectory_selected"
    assert skip_diagnostics["spend_head_prediction"] is False
    assert skip_diagnostics["source_task_delta_vs_trajectory"] < 0.0
    assert run_diagnostics["should_run"] is True
    assert run_diagnostics["reason"] == "trajectory_relative_decomposed_spend"
    assert run_diagnostics["spend_head_rule_id"] == (
        "current_decomposed_spend_source_task_ge_0p295357_source_ge_trajectory"
    )
    assert run_diagnostics["spend_head_prediction"] is True


def test_primary_repair_gate_diagnostics_apply_learned_availability_predictor():
    prompt = (
        "Run two GPU jobs overnight: one reliable baseline and one risky reasoning "
        "intervention. Collect measurements, failure evidence, and a publishable fallback."
    )
    source_base = {
        "history_steps": 16,
        "text": "Run baseline and intervention, collect measurements and failure evidence.",
        "trajectory_summary": {
            "samples": [
                {"step": 4, "visible_chars": 24, "visible_text": "Collect the baseline."},
                {
                    "step": 8,
                    "visible_chars": 55,
                    "visible_text": "Collect the baseline measurement and failure evidence.",
                },
            ]
        },
    }
    source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.32, trajectory_score=0.5),
        **source_base,
    }
    trajectory = _record(
        "model",
        "plan",
        "low_confidence_32",
        task_score=0.32,
        trajectory_score=0.5,
    )

    diagnostics = _primary_repair_gate_diagnostics(
        trigger="learned_availability_predictor_v1",
        source_record=source,
        trajectory_record=trajectory,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=0.0,
        transfer_source_task_min=0.295357,
    )

    assert diagnostics["should_run"] is True
    assert diagnostics["reason"] == "learned_availability_predictor_v1"
    assert diagnostics["spend_head_rule_id"] == (
        "learned_gap_le_8_source_quality_le_0p256429_source_ge_trajectory"
    )
    assert diagnostics["spend_head_prediction"] is True

    weak_source = {
        **source,
        "text": "Collect the baseline, then run the intervention.",
    }
    skip_diagnostics = _primary_repair_gate_diagnostics(
        trigger="learned_availability_predictor_v1",
        source_record=weak_source,
        trajectory_record=trajectory,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=12,
        source_prompt_coverage_min=0.0,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=1.0,
        transfer_source_task_min=0.295357,
    )

    assert skip_diagnostics["should_run"] is False
    assert skip_diagnostics["reason"] == "learned_availability_prompt_gap_high"
    assert skip_diagnostics["spend_head_prediction"] is False


def test_primary_repair_gate_diagnostics_apply_calibrated_availability_predictor():
    prompt = (
        "Run two GPU jobs overnight: one reliable baseline and one risky reasoning "
        "intervention. Collect measurements, failure evidence, and a publishable fallback."
    )
    source_base = {
        "history_steps": 16,
        "text": "Run baseline and intervention, collect measurements and failure evidence.",
        "trajectory_summary": {
            "samples": [
                {"step": 4, "visible_chars": 24, "visible_text": "Collect the baseline."},
                {
                    "step": 8,
                    "visible_chars": 55,
                    "visible_text": "Collect the baseline measurement and failure evidence.",
                },
            ]
        },
    }
    source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.32, trajectory_score=0.5),
        **source_base,
    }
    trajectory = _record(
        "model",
        "plan",
        "low_confidence_32",
        task_score=0.32,
        trajectory_score=0.5,
    )

    diagnostics = _primary_repair_gate_diagnostics(
        trigger="calibrated_availability_predictor_v1",
        source_record=source,
        trajectory_record=trajectory,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=0.0,
        transfer_source_task_min=0.295357,
    )

    assert diagnostics["should_run"] is True
    assert diagnostics["reason"] == "calibrated_availability_predictor_v1"
    assert diagnostics["spend_head_rule_id"] == "calibrated_gap_not_7_source_ge_trajectory"
    assert diagnostics["spend_head_prediction"] is True

    gap_seven_prompt = "ALPHA BETA GAMMA DELTA EPSILON ZETA ETA THETA"
    gap_seven_source = {
        **source,
        "text": "ALPHA",
        "trajectory_summary": {
            "samples": [
                {
                    "step": 4,
                    "visible_chars": 31,
                    "visible_text": "ALPHA placeholder text sample",
                },
                {
                    "step": 8,
                    "visible_chars": 35,
                    "visible_text": "ALPHA BETA placeholder text sample",
                },
            ]
        },
    }
    skip_gap = _primary_repair_gate_diagnostics(
        trigger="calibrated_availability_predictor_v1",
        source_record=gap_seven_source,
        trajectory_record=trajectory,
        source_controls=[],
        task_prompt=gap_seven_prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=1,
        source_prompt_gap_min=2,
        source_prompt_gap_max=12,
        source_prompt_coverage_min=0.0,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=1.0,
        transfer_source_task_min=0.295357,
    )
    stronger_trajectory = {
        **trajectory,
        "task_score": {"score": 0.40},
    }
    skip_trajectory = _primary_repair_gate_diagnostics(
        trigger="calibrated_availability_predictor_v1",
        source_record=source,
        trajectory_record=stronger_trajectory,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
        value_proxy_source_quality_max=1.0,
        transfer_source_task_min=0.295357,
    )

    assert skip_gap["should_run"] is False
    assert skip_gap["reason"] == "calibrated_availability_prompt_gap_ambiguous"
    assert skip_gap["spend_head_prediction"] is False
    assert skip_trajectory["should_run"] is False
    assert skip_trajectory["reason"] == "calibrated_availability_source_below_trajectory"
    assert skip_trajectory["spend_head_prediction"] is False


def test_primary_repair_gate_diagnostics_explain_late_denoise_skeleton():
    prompt = (
        "Run two GPU jobs overnight: one reliable baseline and one risky reasoning "
        "intervention. Collect measurements, failure evidence, and a publishable fallback."
    )
    source = {
        **_record("model", "plan", "low_confidence_32", task_score=0.0, trajectory_score=0.5),
        "text": (
            "Collect the baseline measurement first, then run the intervention. "
            "If the intervention fails, record failure evidence and keep a fallback."
        ),
        "trajectory_summary": {
            "samples": [
                {
                    "step": 18,
                    "visible_chars": 55,
                    "visible_text": "Collect the baseline measurement and failure evidence.",
                },
            ]
        },
    }

    diagnostics = _primary_repair_gate_diagnostics(
        trigger="denoise_phase_repairability",
        source_record=source,
        source_controls=[],
        task_prompt=prompt,
        task_answer_type="rubric",
        source_quality_threshold=0.99,
        source_min_chars=40,
        source_prompt_gap_min=2,
        source_prompt_gap_max=8,
        source_prompt_coverage_min=0.30,
        source_prompt_coverage_max=1.00,
        denoise_skeleton_max_step=12,
    )

    assert diagnostics["should_run"] is False
    assert diagnostics["reason"] == "late_repairable_denoise_skeleton"
    assert diagnostics["has_repairable_denoise_skeleton"] is True
    assert diagnostics["denoise_skeleton_within_max_step"] is False


def test_report_includes_repair_spend_gate_diagnostics():
    fixed = _record("model", "plan", "low_confidence_32", 0.10, 0.20)
    random = _record("model", "plan", "random_32", 0.15, 0.20)
    trajectory = _record("model", "plan", "low_confidence_64", 0.20, 0.30)
    repair = {
        **_record("model", "plan", "constraint_gap_span_repair", 0.40, 0.35),
        "repair": {
            "name": "constraint_gap_span_repair",
            "source_control": "low_confidence_32",
            "source_state": "final",
        },
    }
    arm_records = [
        {**fixed, "arm": "fixed", "arm_generation_budget_per_task": 1},
        {**random, "arm": "random", "arm_generation_budget_per_task": 1},
        {**trajectory, "arm": "trajectory_selected", "arm_generation_budget_per_task": 2},
        {**repair, "arm": "repair_selected", "arm_generation_budget_per_task": 3},
    ]

    scores = summarize_three_arm_scores(
        [fixed, random, trajectory, repair],
        arm_records,
        repair_spend_gate_rows=[
            {
                "candidate_key": "model",
                "task_id": "plan",
                "source_control": "low_confidence_32",
                "should_run": False,
                "reason": "no_repairable_denoise_skeleton",
                "source_task_score": 0.10,
                "source_quality": 0.25,
                "source_chars": 120,
                "source_needs_repair": True,
                "prompt_gap_count": 4,
                "prompt_coverage": 0.50,
                "in_repairable_band": True,
                "has_repairable_denoise_skeleton": False,
            }
        ],
    )

    report = render_report(scores)

    assert "## Repair Spend Gate Diagnostics" in report
    assert "model | plan | low_confidence_32 | False | no_repairable_denoise_skeleton" in report


def test_primary_repair_gate_respects_source_controls():
    source = {
        **_record("model", "plan", "evolved_low_confidence_48", task_score=0.0, trajectory_score=0.5),
        "text": "Compare baseline and intervention.",
    }

    assert not _should_run_primary_repair_pass(
        trigger="always",
        source_record=source,
        source_controls=["evolved_random_48"],
        task_prompt="Compare a baseline and intervention.",
        task_answer_type="rubric",
        source_quality_threshold=0.40,
        source_min_chars=80,
    )


def test_prompt_guided_rescue_runs_on_low_source_quality():
    baseline = {
        **_record("model", "plan", "evolved_random_48", task_score=0.0, trajectory_score=0.2),
        "generation_seed": 42,
        "text": "Generic plan.",
    }
    selected = {
        **_record("model", "plan", "prefix_25_repair", task_score=0.0, trajectory_score=0.3),
        "generation_seed": 43,
        "schedule": None,
        "repair": {"name": "prefix_25_repair"},
        "text": "Still generic.",
    }

    assert _should_run_prompt_guided_rescue(
        trigger="source_quality",
        selected_repair=selected,
        baseline_record=baseline,
        repair_pool=[baseline, selected],
        source_controls=["evolved_random_48"],
        task_prompt="Compare a baseline and intervention, preserve rollback, record metrics, and define a threshold.",
        task_answer_type="rubric",
        exact_task_trajectory_policy="fixed",
        trajectory_selector="planning_state",
        source_quality_threshold=0.99,
    )


def test_prompt_guided_rescue_respects_source_controls():
    baseline = {
        **_record("model", "plan", "evolved_low_confidence_48", task_score=0.0, trajectory_score=0.2),
        "generation_seed": 42,
        "text": "Generic plan.",
    }

    assert not _should_run_prompt_guided_rescue(
        trigger="source_quality",
        selected_repair=baseline,
        baseline_record=baseline,
        repair_pool=[baseline],
        source_controls=["evolved_random_48"],
        task_prompt="Compare a baseline and intervention, preserve rollback, record metrics, and define a threshold.",
        task_answer_type="rubric",
        exact_task_trajectory_policy="fixed",
        trajectory_selector="planning_state",
        source_quality_threshold=0.99,
    )


def test_constraint_gap_rescue_runs_on_prompt_gap_pressure():
    baseline = {
        **_record("model", "plan", "evolved_random_48", task_score=0.0, trajectory_score=0.2),
        "generation_seed": 42,
        "text": "Compare the baseline and risky intervention.",
    }

    assert _should_run_constraint_gap_rescue(
        trigger="prompt_gap",
        selected_repair=baseline,
        baseline_record=baseline,
        source_controls=["evolved_random_48"],
        task_prompt=(
            "Two GPU jobs run overnight. Compare the baseline and intervention, "
            "record measurements, preserve rollback, define tomorrow's threshold, "
            "and state what remains publishable."
        ),
        task_answer_type="rubric",
        min_terms=4,
        source_quality_floor=0.0,
        source_quality_ceiling=1.0,
    )


def test_constraint_gap_rescue_respects_quality_band_and_controls():
    baseline = {
        **_record("model", "plan", "evolved_low_confidence_48", task_score=0.0, trajectory_score=0.2),
        "generation_seed": 42,
        "text": "Compare the baseline and risky intervention.",
    }

    assert not _should_run_constraint_gap_rescue(
        trigger="prompt_gap",
        selected_repair=baseline,
        baseline_record=baseline,
        source_controls=["evolved_random_48"],
        task_prompt="Compare baseline, intervention, rollback, threshold, measurements, and publishable result.",
        task_answer_type="rubric",
        min_terms=2,
        source_quality_floor=0.0,
        source_quality_ceiling=1.0,
    )
    assert not _should_run_constraint_gap_rescue(
        trigger="prompt_gap",
        selected_repair=baseline,
        baseline_record=baseline,
        source_controls=[],
        task_prompt="Compare baseline, intervention, rollback, threshold, measurements, and publishable result.",
        task_answer_type="rubric",
        min_terms=2,
        source_quality_floor=0.9,
        source_quality_ceiling=1.0,
    )


def test_history_rescue_runs_only_when_primary_repair_keeps_matching_baseline():
    baseline = {
        **_record("model", "plan", "evolved_random_48", task_score=0.0, trajectory_score=0.5),
        "generation_seed": 42,
    }
    promoted_repair = {
        **_record("model", "plan", "prefix_25_repair", task_score=0.0, trajectory_score=0.5),
        "generation_seed": 43,
        "schedule": None,
        "repair": {"name": "prefix_25_repair"},
    }

    assert _should_run_history_rescue(
        selected_repair=baseline,
        baseline_record=baseline,
        source_controls=["evolved_random_48"],
    )
    assert not _should_run_history_rescue(
        selected_repair=promoted_repair,
        baseline_record=baseline,
        source_controls=["evolved_random_48"],
    )
    assert not _should_run_history_rescue(
        selected_repair=baseline,
        baseline_record=baseline,
        source_controls=["evolved_low_confidence_48"],
    )


def test_selector_disagreement_rescue_runs_when_repair_selectors_disagree():
    baseline = {
        **_record("model", "plan", "evolved_random_48", task_score=0.0, trajectory_score=0.2),
        "generation_seed": 42,
        "text": "Compare baseline and intervention.",
    }
    quality_repair = {
        **_record("model", "plan", "prefix_25_repair", task_score=0.0, trajectory_score=0.3),
        "generation_seed": 43,
        "schedule": None,
        "repair": {"name": "prefix_25_repair"},
        "text": (
            "Compare the baseline and intervention, preserve rollback, "
            "record metrics, and define a decision threshold."
        ),
    }
    trajectory_repair = {
        **_record("model", "plan", "history_visible_repair", task_score=0.0, trajectory_score=0.95),
        "generation_seed": 44,
        "schedule": None,
        "repair": {"name": "history_visible_repair"},
        "text": "Generic plan.",
    }

    assert _should_run_selector_disagreement_rescue(
        selected_repair=quality_repair,
        baseline_record=baseline,
        repair_pool=[baseline, quality_repair, trajectory_repair],
        source_controls=["evolved_random_48"],
        task_prompt="Compare a baseline and intervention, preserve rollback, and define a decision threshold.",
        task_answer_type="rubric",
        exact_task_trajectory_policy="fixed",
        trajectory_selector="generic",
    )
    assert _should_run_adaptive_history_rescue(
        trigger="baseline_or_selector_disagreement",
        selected_repair=quality_repair,
        baseline_record=baseline,
        repair_pool=[baseline, quality_repair, trajectory_repair],
        source_controls=["evolved_random_48"],
        task_prompt="Compare a baseline and intervention, preserve rollback, and define a decision threshold.",
        task_answer_type="rubric",
        exact_task_trajectory_policy="fixed",
        trajectory_selector="generic",
    )
    assert not _should_run_adaptive_history_rescue(
        trigger="baseline",
        selected_repair=quality_repair,
        baseline_record=baseline,
        repair_pool=[baseline, quality_repair, trajectory_repair],
        source_controls=["evolved_random_48"],
        task_prompt="Compare a baseline and intervention, preserve rollback, and define a decision threshold.",
        task_answer_type="rubric",
        exact_task_trajectory_policy="fixed",
        trajectory_selector="generic",
    )


def test_selector_disagreement_rescue_respects_source_controls():
    baseline = {
        **_record("model", "plan", "evolved_low_confidence_48", task_score=0.0, trajectory_score=0.2),
        "generation_seed": 42,
    }
    selected = {
        **_record("model", "plan", "prefix_25_repair", task_score=0.0, trajectory_score=0.3),
        "generation_seed": 43,
        "schedule": None,
        "repair": {"name": "prefix_25_repair"},
    }
    trajectory_repair = {
        **_record("model", "plan", "history_visible_repair", task_score=0.0, trajectory_score=0.95),
        "generation_seed": 44,
        "schedule": None,
        "repair": {"name": "history_visible_repair"},
    }

    assert not _should_run_selector_disagreement_rescue(
        selected_repair=selected,
        baseline_record=baseline,
        repair_pool=[baseline, selected, trajectory_repair],
        source_controls=["evolved_random_48"],
        task_prompt="",
        task_answer_type="rubric",
        exact_task_trajectory_policy="fixed",
        trajectory_selector="generic",
    )


def test_selected_history_repair_sample_prefers_prompt_grounded_partial_state():
    record = {
        **_record("llada-8b-instruct-hf", "plan", "evolved", task_score=0.0, trajectory_score=0.5),
        "history_samples": [
            {
                "step": 4,
                "generated_token_ids": [101, 126336, 126336, 126081],
                "text": "Generic plan<|mdm_mask|><|mdm_mask|>",
            },
            {
                "step": 8,
                "generated_token_ids": [201, 202, 126336, 126081],
                "text": "Compare baseline<|mdm_mask|>",
            },
            {
                "step": 12,
                "generated_token_ids": [301, 302, 303, 126081],
                "text": "Compare baseline metrics",
            },
        ],
        "trajectory_summary": {
            "samples": [
                {"step": 4, "mask_count": 2, "visible_chars": 12, "visible_text": "Generic plan"},
                {"step": 8, "mask_count": 1, "visible_chars": 16, "visible_text": "Compare baseline"},
                {"step": 12, "mask_count": 0, "visible_chars": 24, "visible_text": "Compare baseline metrics"},
            ]
        },
    }

    selected = _selected_history_repair_sample(
        record,
        "Compare a baseline and intervention, record metrics, and preserve rollback.",
    )

    assert selected is not None
    assert selected["step"] == 8
    assert selected["generated_token_ids"] == [201, 202, 126336, 126081]


def test_planning_prompt_selector_can_override_raw_trajectory_score():
    records = [
        {
            **_record("model", "plan", "stable_generic", task_score=0.0, trajectory_score=0.9),
            "text": "Use complex nuanced tasks to assess deep understanding.",
        },
        {
            **_record("model", "plan", "prompt_grounded", task_score=0.0, trajectory_score=0.4),
            "text": "Compare the baseline and intervention, record metrics, and define a rollback threshold.",
        },
    ]

    selected = select_three_arm_records(
        records,
        seed=1,
        candidate_key="model",
        task_id="plan",
        task_prompt="Compare a baseline and intervention, record metrics, and set a rollback threshold.",
        task_answer_type="rubric",
        trajectory_selector="planning_prompt",
    )

    assert selected["trajectory_selected"]["schedule"]["name"] == "prompt_grounded"


def test_planning_state_selector_uses_visible_denoise_states():
    records = [
        _record_with_history(
            "model",
            "plan",
            "weak_history",
            task_score=0.0,
            trajectory_score=0.8,
            text="Assess deep understanding with generic tasks.",
            samples=["", "Assess deep understanding.", "Assess deep understanding with generic tasks."],
        ),
        _record_with_history(
            "model",
            "plan",
            "useful_history",
            task_score=0.0,
            trajectory_score=0.4,
            text="Compare baseline and intervention, record metrics, and preserve rollback.",
            samples=[
                "Compare baseline and intervention.",
                "Compare baseline and intervention, record metrics.",
                "Compare baseline and intervention, record metrics, and preserve rollback.",
            ],
        ),
    ]

    selected = select_three_arm_records(
        records,
        seed=1,
        candidate_key="model",
        task_id="plan",
        task_prompt="Compare a baseline and intervention, record metrics, and preserve rollback.",
        task_answer_type="rubric",
        trajectory_selector="planning_state",
    )

    assert selected["trajectory_selected"]["schedule"]["name"] == "useful_history"


def test_exact_proposal_history_policy_selects_verified_denoise_state():
    record = _record_with_history(
        "llada-8b-instruct-hf",
        "sym",
        "low_confidence_32",
        task_score=0.0,
        trajectory_score=0.2,
        text="M L K",
        samples=["", "Answer: L K M", "M L K"],
    )
    record["task"] = {
        "task_id": "sym",
        "family": "symbolic",
        "answer_type": "short_text",
        "scorer": "exact_short_text",
        "answer": "L K M",
    }
    record["task_score"] = {"score": 0.0, "extracted_answer": "m l k"}

    selected = select_three_arm_records(
        [record],
        seed=1,
        candidate_key="llada-8b-instruct-hf",
        task_id="sym",
        task_prompt=(
            "A display starts with the code K L M. Rotate the code one step left, "
            "then swap the final two letters. What code should be displayed? "
            "Answer with the three letters separated by spaces."
        ),
        task_answer_type="short_text",
        exact_task_trajectory_policy="proposal_history",
    )

    trajectory = selected["trajectory_selected"]
    assert trajectory["text"] == "Answer: L K M"
    assert trajectory["task_score"]["score"] == 1.0
    assert trajectory["schedule"]["name"] == "low_confidence_32:history_step_2"
    assert trajectory["exact_trajectory_selection"]["source"] == "history"


def test_exact_proposal_history_policy_keeps_fixed_without_verified_state():
    record = _record_with_history(
        "llada-8b-instruct-hf",
        "sym",
        "low_confidence_32",
        task_score=0.0,
        trajectory_score=0.2,
        text="M L K",
        samples=["", "M L", "M L K"],
    )
    record["task"] = {
        "task_id": "sym",
        "family": "symbolic",
        "answer_type": "short_text",
        "scorer": "exact_short_text",
        "answer": "L K M",
    }
    record["task_score"] = {"score": 0.0, "extracted_answer": "m l k"}

    selected = select_three_arm_records(
        [record],
        seed=1,
        candidate_key="llada-8b-instruct-hf",
        task_id="sym",
        task_prompt=(
            "A display starts with the code K L M. Rotate the code one step left, "
            "then swap the final two letters. What code should be displayed? "
            "Answer with the three letters separated by spaces."
        ),
        task_answer_type="short_text",
        exact_task_trajectory_policy="proposal_history",
    )

    assert selected["trajectory_selected"] is record


def test_planning_state_v2_rewards_prompt_specific_action_structure():
    records = [
        {
            **_record("model", "plan", "generic", task_score=0.0, trajectory_score=0.8),
            "text": "Analyze the issue and make a plan with high quality reasoning.",
        },
        {
            **_record("model", "plan", "specific", task_score=0.0, trajectory_score=0.4),
            "text": (
                "1. Confirm the customer dashboard totals. 2. Ship a temporary fix today. "
                "3. Preserve logs for a later root cause analysis."
            ),
        },
    ]

    selected = select_three_arm_records(
        records,
        seed=1,
        candidate_key="model",
        task_id="plan",
        task_prompt=(
            "A customer dashboard shows wrong totals after a timezone migration. "
            "The team needs a fix today and a deeper root-cause analysis later."
        ),
        task_answer_type="rubric",
        trajectory_selector="planning_state_v2",
    )

    assert selected["trajectory_selected"]["schedule"]["name"] == "specific"


def test_three_arm_summary_tracks_generation_budget_and_deltas():
    fixed = _record("model", "task", "fixed", task_score=0.25, trajectory_score=0.1)
    random = _record("model", "task", "random", task_score=0.5, trajectory_score=0.2)
    trajectory = _record("model", "task", "trajectory", task_score=0.75, trajectory_score=0.9)
    arm_records = [
        {**fixed, "arm": "fixed", "arm_generation_budget_per_task": 1},
        {**random, "arm": "random", "arm_generation_budget_per_task": 1},
        {**trajectory, "arm": "trajectory_selected", "arm_generation_budget_per_task": 3},
    ]

    scores = summarize_three_arm_scores([fixed, random, trajectory], arm_records)

    assert scores["all_generation_count"] == 3
    assert scores["arms"]["trajectory_selected"]["mean_generation_budget_per_task"] == 3
    assert scores["by_family_arm"]["planning"]["trajectory_selected"]["count"] == 1
    assert scores["trajectory_task_delta_vs_fixed"] == 0.5
    assert scores["trajectory_task_delta_vs_random"] == 0.25
    assert scores["trajectory_wins_vs_fixed"] == {"wins": 1, "ties": 0, "losses": 0}
    assert scores["selector_regret_vs_trajectory"] == {
        "count": 1,
        "mean_task_regret": 0.0,
        "improvable_count": 0,
        "improvable_fraction": 0.0,
        "wins_vs_selected": {"wins": 0, "ties": 1, "losses": 0},
    }


def test_summary_tracks_exact_proposal_history_selection_sources():
    fixed = _record("model", "task_a", "fixed", task_score=0.0, trajectory_score=0.1)
    random = _record("model", "task_a", "random", task_score=0.0, trajectory_score=0.1)
    history = {
        **_record("model", "task_a", "history", task_score=1.0, trajectory_score=0.2),
        "arm": "trajectory_selected",
        "arm_generation_budget_per_task": 3,
        "exact_trajectory_selection": {"source": "history"},
    }
    final = {
        **_record("model", "task_b", "final", task_score=1.0, trajectory_score=0.3),
        "arm": "evolved",
        "arm_generation_budget_per_task": 4,
        "exact_trajectory_selection": {"source": "final"},
    }
    fallback = {
        **_record("model", "task_c", "fallback", task_score=0.0, trajectory_score=0.4),
        "arm": "trajectory_selected",
        "arm_generation_budget_per_task": 3,
        "arm_selection_reason": "exact_answer_proposal_history_no_match_kept_fixed",
    }

    scores = summarize_three_arm_scores(
        [
            fixed,
            random,
            history,
            final,
            fallback,
        ],
        [
            {**fixed, "arm": "fixed", "arm_generation_budget_per_task": 1},
            {**random, "arm": "random", "arm_generation_budget_per_task": 1},
            history,
            final,
            fallback,
        ],
        exact_task_trajectory_policy="proposal_history",
    )

    assert scores["exact_trajectory_selection_source_counts"] == {
        "evolved:final": 1,
        "trajectory_selected:fallback": 1,
        "trajectory_selected:history": 1,
    }


def test_summary_tracks_history_mutability_diagnostics():
    monotonic = {
        **_record("model", "task_a", "mono", task_score=0.0, trajectory_score=0.1),
        "trajectory_summary": {
            "sampled_history_is_monotonic_fill": True,
            "committed_token_change_count": 0,
            "committed_token_remask_count": 0,
            "remasked_token_rewrite_count": 0,
            "mask_count_increase_count": 0,
        },
    }
    revision = {
        **_record("model", "task_b", "revision", task_score=0.0, trajectory_score=0.1),
        "trajectory_summary": {
            "sampled_history_is_monotonic_fill": False,
            "committed_token_change_count": 2,
            "committed_token_remask_count": 1,
            "remasked_token_rewrite_count": 1,
            "mask_count_increase_count": 1,
        },
    }

    scores = summarize_three_arm_scores(
        [monotonic, revision],
        [
            {**monotonic, "arm": "fixed", "arm_generation_budget_per_task": 1},
            {**revision, "arm": "trajectory_selected", "arm_generation_budget_per_task": 2},
        ],
    )

    assert scores["history_mutability"] == {
        "count": 2,
        "monotonic_fill_count": 1,
        "committed_token_change_count": 2,
        "committed_token_remask_count": 1,
        "remasked_token_rewrite_count": 1,
        "mask_count_increase_count": 1,
    }


def test_summary_tracks_evolved_arm_budget_and_deltas():
    fixed = _record("model", "task", "fixed", task_score=0.25, trajectory_score=0.1)
    random = _record("model", "task", "random", task_score=0.5, trajectory_score=0.2)
    trajectory = _record("model", "task", "trajectory", task_score=0.75, trajectory_score=0.9)
    evolved = _record("model", "task", "evolved", task_score=1.0, trajectory_score=0.8)
    arm_records = [
        {**fixed, "arm": "fixed", "arm_generation_budget_per_task": 1},
        {**random, "arm": "random", "arm_generation_budget_per_task": 1},
        {**trajectory, "arm": "trajectory_selected", "arm_generation_budget_per_task": 3},
        {
            **evolved,
            "arm": "evolved",
            "arm_generation_budget_per_task": 5,
            "arm_selection_reason": "max_generic_score_evolved_pool",
            "arm_selector_score": 0.8,
        },
    ]

    scores = summarize_three_arm_scores([fixed, random, trajectory, evolved], arm_records)

    assert scores["all_generation_count"] == 4
    assert scores["arms"]["evolved"]["mean_generation_budget_per_task"] == 5
    assert scores["evolved_task_delta_vs_fixed"] == 0.75
    assert scores["evolved_task_delta_vs_random"] == 0.5
    assert scores["evolved_task_delta_vs_trajectory"] == 0.25
    assert scores["evolved_wins_vs_trajectory"] == {"wins": 1, "ties": 0, "losses": 0}
    assert scores["selector_regret_vs_evolved"]["mean_task_regret"] == 0.0


def test_summary_tracks_repair_arm_budget_and_deltas():
    fixed = _record("llada-8b-instruct-hf", "task", "fixed", task_score=0.25, trajectory_score=0.1)
    random = _record("llada-8b-instruct-hf", "task", "random", task_score=0.5, trajectory_score=0.2)
    trajectory = _record("llada-8b-instruct-hf", "task", "trajectory", task_score=0.75, trajectory_score=0.9)
    evolved = _record("llada-8b-instruct-hf", "task", "evolved", task_score=1.0, trajectory_score=0.8)
    repair = _record("llada-8b-instruct-hf", "task", "repair", task_score=1.0, trajectory_score=0.85)
    evolved["planning_quality_score"] = 0.75
    repair["planning_quality_score"] = 1.0
    repair["schedule"] = None
    repair["repair"] = {
        "name": "low_confidence_25_repair",
        "source_history_step": 8,
        "source_state": "history",
        "seed_masked_positions": 3,
        "source_task_score": 0.75,
        "source_planning_quality_score": 0.75,
    }
    arm_records = [
        {**fixed, "arm": "fixed", "arm_generation_budget_per_task": 1},
        {**random, "arm": "random", "arm_generation_budget_per_task": 1},
        {**trajectory, "arm": "trajectory_selected", "arm_generation_budget_per_task": 3},
        {**evolved, "arm": "evolved", "arm_generation_budget_per_task": 5},
        {
            **repair,
            "arm": "repair_selected",
            "arm_generation_budget_per_task": 7,
            "arm_selection_reason": "max_generic_score_repair_pool",
            "arm_selector_score": 0.85,
        },
    ]

    scores = summarize_three_arm_scores(
        [fixed, random, trajectory, evolved, repair],
        arm_records,
        prompt_guided_rescue_trigger="baseline_or_source_quality",
        prompt_guided_rescue_limit=1,
        prompt_guided_rescue_source_quality_threshold=0.45,
        prompt_guided_rescue_source_controls=["evolved_random_48"],
        repair_spend_trigger="source_quality_or_short",
        repair_source_policy="non_revision_evolved",
        repair_source_quality_threshold=0.50,
        repair_source_min_chars=320,
        repair_source_controls=["evolved_low_confidence_48"],
    )

    assert scores["arms"]["repair_selected"]["mean_generation_budget_per_task"] == 7
    assert scores["repair_spend_trigger"] == "source_quality_or_short"
    assert scores["repair_source_policy"] == "non_revision_evolved"
    assert scores["repair_source_controls"] == ["evolved_low_confidence_48"]
    assert scores["prompt_guided_rescue_trigger"] == "baseline_or_source_quality"
    assert scores["prompt_guided_rescue_source_controls"] == ["evolved_random_48"]
    assert scores["repair_eligible_task_count"] == 1
    assert scores["repair_task_delta_vs_evolved"] == 0.0
    assert scores["repair_generation_budget_delta_vs_evolved"] == 2.0
    assert scores["repair_task_delta_per_extra_generation_vs_evolved"] == 0.0
    assert scores["repair_wins_vs_evolved"] == {"wins": 0, "ties": 1, "losses": 0}
    assert scores["selector_regret_vs_repair"]["mean_task_regret"] == 0.0
    assert scores["comparison_rows"][0]["repair_control"] == "low_confidence_25_repair"
    assert scores["comparison_rows"][0]["repair_source_state"] == "history"
    assert scores["comparison_rows"][0]["repair_source_history_step"] == "8"
    assert scores["repair_candidate_summary"]["low_confidence_25_repair"] == {
        "count": 1,
        "selected_count": 1,
        "source_controls": "",
        "source_states": "history",
        "mean_seed_masked_positions": 3.0,
        "mean_span_literal_target_found": 0.0,
        "mean_span_fallback_used": 0.0,
        "mean_overpreservation_penalty": 0.0,
        "mean_contradiction_penalty": 0.0,
        "mean_planning_span_residue_penalty": 0.0,
        "mean_seed_realization_quality": 0.0,
        "mean_seed_objective_score": 0.0,
        "mean_seed_realization_meta_penalty": 0.0,
        "mean_seed_realization_control_coverage": 0.0,
        "mean_seed_semantic_preservation": 0.0,
        "mean_planning_quality_delta_vs_source": 0.25,
        "mean_task_delta_vs_source": 0.25,
        "mean_proposal_task_score": 0.0,
        "mean_task_delta_vs_proposal": 0.0,
        "mean_self_repair_changed_answer": 0.0,
        "mean_self_repair_arithmetic_consistent": 0.0,
        "mean_self_repair_arithmetic_claim_count": 0.0,
        "mean_self_repair_irrelevant_number_used": 0.0,
        "mean_self_repair_missing_required_operator_count": 0.0,
        "mean_self_repair_quantity_role_gap_count": 0.0,
        "mean_self_repair_arithmetic_provenance_gap_count": 0.0,
        "mean_self_repair_final_answer_role_gap_count": 0.0,
        "mean_self_repair_final_answer_object_gap_count": 0.0,
        "mean_self_repair_final_answer_target_gap_count": 0.0,
        "mean_self_repair_short_text_symbolic_gap_count": 0.0,
        "mean_self_repair_short_text_trace_gap_count": 0.0,
        "wins_vs_source": {"wins": 1, "ties": 0, "losses": 0},
        "mean_task_score": 1.0,
        "mean_trajectory_score": 0.85,
        "mean_combined_score": 0.9625,
    }


def test_summary_tracks_oracle_headroom_without_using_it_for_selection():
    fixed = _record("model", "task", "fixed", task_score=0.25, trajectory_score=0.1)
    trajectory = _record("model", "task", "trajectory", task_score=0.5, trajectory_score=0.9)
    oracle = _record("model", "task", "oracle", task_score=0.75, trajectory_score=0.2)
    arm_records = [
        {**fixed, "arm": "fixed", "arm_generation_budget_per_task": 1},
        {**fixed, "arm": "random", "arm_generation_budget_per_task": 1},
        {**trajectory, "arm": "trajectory_selected", "arm_generation_budget_per_task": 3},
    ]

    scores = summarize_three_arm_scores([fixed, trajectory, oracle], arm_records)

    assert scores["oracle_generation_budget_per_task"] == 3
    assert scores["oracle_task_score"] == 0.75
    assert scores["oracle_headroom_vs_trajectory"] == 0.25
    assert scores["selector_regret_vs_trajectory"] == {
        "count": 1,
        "mean_task_regret": 0.25,
        "improvable_count": 1,
        "improvable_fraction": 1.0,
        "wins_vs_selected": {"wins": 1, "ties": 0, "losses": 0},
    }
    assert scores["comparison_rows"][0]["oracle_schedule"] == "oracle"
    assert scores["comparison_rows"][0]["oracle_delta_vs_trajectory"] == 0.25


def test_summary_tracks_family_arm_breakdown():
    planning_fixed = _record("model", "plan", "fixed", task_score=0.25, trajectory_score=0.1)
    planning_random = _record("model", "plan", "random", task_score=0.5, trajectory_score=0.2)
    planning_trajectory = _record("model", "plan", "trajectory", task_score=0.75, trajectory_score=0.9)
    symbolic_fixed = {
        **_record("model", "sym", "fixed", task_score=1.0, trajectory_score=0.1),
        "task": {"task_id": "sym", "family": "symbolic", "answer_type": "short_text"},
    }
    symbolic_random = {
        **_record("model", "sym", "random", task_score=0.0, trajectory_score=0.2),
        "task": {"task_id": "sym", "family": "symbolic", "answer_type": "short_text"},
    }
    symbolic_trajectory = {
        **_record("model", "sym", "trajectory", task_score=1.0, trajectory_score=0.3),
        "task": {"task_id": "sym", "family": "symbolic", "answer_type": "short_text"},
    }
    arm_records = [
        {**planning_fixed, "arm": "fixed", "arm_generation_budget_per_task": 1},
        {**planning_random, "arm": "random", "arm_generation_budget_per_task": 1},
        {**planning_trajectory, "arm": "trajectory_selected", "arm_generation_budget_per_task": 3},
        {**symbolic_fixed, "arm": "fixed", "arm_generation_budget_per_task": 1},
        {**symbolic_random, "arm": "random", "arm_generation_budget_per_task": 1},
        {**symbolic_trajectory, "arm": "trajectory_selected", "arm_generation_budget_per_task": 3},
    ]

    scores = summarize_three_arm_scores(
        [
            planning_fixed,
            planning_random,
            planning_trajectory,
            symbolic_fixed,
            symbolic_random,
            symbolic_trajectory,
        ],
        arm_records,
    )

    assert scores["by_family_arm"]["planning"]["trajectory_selected"]["mean_task_score"] == 0.75
    assert scores["by_family_arm"]["symbolic"]["trajectory_selected"]["mean_task_score"] == 1.0
