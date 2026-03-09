"""
Audit whether last-integer-wins grading creates a length-accuracy confound.

Key questions:
1. Do perturbed responses contain more unique integers?
2. Does the correct answer appear more often ANYWHERE in perturbed responses?
3. How often is the correct answer NOT the last integer but IS present somewhere?
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def parse_integers(text: str) -> list[int]:
    """Extract all integers from text (same regex as harness)."""
    raw = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)", text)
    return [int(s.replace(",", "")) for s in raw]


def audit_response(response: str, expected: int) -> dict:
    """Audit a single response for grading confounds."""
    numbers = parse_integers(response)
    unique_numbers = set(numbers)

    last_is_correct = numbers[-1] == expected if numbers else False
    answer_anywhere = expected in unique_numbers
    answer_count = numbers.count(expected)

    # Position of correct answer (if present)
    positions = [i for i, n in enumerate(numbers) if n == expected]
    last_position = positions[-1] if positions else -1
    is_only_last = last_is_correct and (len(positions) == 1)

    return {
        'n_integers': len(numbers),
        'n_unique': len(unique_numbers),
        'last_correct': last_is_correct,
        'answer_anywhere': answer_anywhere,
        'answer_count': answer_count,
        'answer_last_position': last_position,
        'total_positions': len(numbers),
        'answer_only_at_end': is_only_last,
    }


def audit_file(path: str, label: str):
    """Audit a results file for grading confounds."""
    with open(path) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"GRADING CONFOUND AUDIT: {label}")
    print(f"Model: {data.get('model', '?')}, Baseline acc: {data['baseline_accuracy']:.0%}")
    print(f"{'='*70}")

    # Baseline audit
    baseline_audits = []
    for task in data['baseline_results']:
        response = task.get('response', '') or task.get('response_raw', '')
        expected = task.get('expected') or task.get('correct_answer')
        if response:
            a = audit_response(response, expected)
            a['correct'] = task['correct']
            a['task_id'] = task['task_id']
            baseline_audits.append(a)

    # Perturbed audit
    perturbed_audits = []
    for li_data in data.get('sensitivity_results', []):
        for task in li_data['task_results']:
            response = task.get('response', '') or task.get('response_raw', '')
            expected = task.get('expected') or task.get('correct_answer')
            if response:
                a = audit_response(response, expected)
                a['correct'] = task['correct']
                a['task_id'] = task['task_id']
                perturbed_audits.append(a)

    def avg(lst, key):
        vals = [x[key] for x in lst]
        return sum(vals) / len(vals) if vals else 0

    def avg_by(lst, key, filter_key, filter_val):
        vals = [x[key] for x in lst if x[filter_key] == filter_val]
        return sum(vals) / len(vals) if vals else 0

    print(f"\n--- Integer Count Comparison ---")
    print(f"  Baseline:  mean {avg(baseline_audits, 'n_integers'):.1f} integers, "
          f"{avg(baseline_audits, 'n_unique'):.1f} unique")
    print(f"  Perturbed: mean {avg(perturbed_audits, 'n_integers'):.1f} integers, "
          f"{avg(perturbed_audits, 'n_unique'):.1f} unique")

    print(f"\n  Baseline by outcome:")
    print(f"    Correct: {avg_by(baseline_audits, 'n_integers', 'correct', True):.1f} integers, "
          f"{avg_by(baseline_audits, 'n_unique', 'correct', True):.1f} unique")
    print(f"    Wrong:   {avg_by(baseline_audits, 'n_integers', 'correct', False):.1f} integers, "
          f"{avg_by(baseline_audits, 'n_unique', 'correct', False):.1f} unique")

    print(f"\n  Perturbed by outcome:")
    print(f"    Correct: {avg_by(perturbed_audits, 'n_integers', 'correct', True):.1f} integers, "
          f"{avg_by(perturbed_audits, 'n_unique', 'correct', True):.1f} unique")
    print(f"    Wrong:   {avg_by(perturbed_audits, 'n_integers', 'correct', False):.1f} integers, "
          f"{avg_by(perturbed_audits, 'n_unique', 'correct', False):.1f} unique")

    # Key confound check: answer present anywhere but NOT last
    print(f"\n--- Confound Check: Answer Present but NOT Last ---")
    base_anywhere_not_last = sum(1 for a in baseline_audits
                                  if a['answer_anywhere'] and not a['last_correct'])
    pert_anywhere_not_last = sum(1 for a in perturbed_audits
                                  if a['answer_anywhere'] and not a['last_correct'])
    print(f"  Baseline:  {base_anywhere_not_last}/{len(baseline_audits)} "
          f"({base_anywhere_not_last/len(baseline_audits)*100:.0f}%) have answer somewhere but NOT last")
    print(f"  Perturbed: {pert_anywhere_not_last}/{len(perturbed_audits)} "
          f"({pert_anywhere_not_last/len(perturbed_audits)*100:.0f}%) have answer somewhere but NOT last")

    # When grading says correct, is the answer ONLY at the end?
    print(f"\n--- When Graded Correct: Is Answer ONLY at End? ---")
    base_correct = [a for a in baseline_audits if a['last_correct']]
    pert_correct = [a for a in perturbed_audits if a['last_correct']]
    base_only_end = sum(1 for a in base_correct if a['answer_only_at_end'])
    pert_only_end = sum(1 for a in pert_correct if a['answer_only_at_end'])
    if base_correct:
        print(f"  Baseline:  {base_only_end}/{len(base_correct)} "
              f"({base_only_end/len(base_correct)*100:.0f}%) answer appears ONLY at end")
    if pert_correct:
        print(f"  Perturbed: {pert_only_end}/{len(pert_correct)} "
              f"({pert_only_end/len(pert_correct)*100:.0f}%) answer appears ONLY at end")

    # Average answer count in correct responses
    print(f"\n--- Answer Frequency in Correct Responses ---")
    if base_correct:
        print(f"  Baseline:  mean {avg_by(baseline_audits, 'answer_count', 'last_correct', True):.1f} "
              f"times answer appears")
    if pert_correct:
        print(f"  Perturbed: mean {avg_by(perturbed_audits, 'answer_count', 'last_correct', True):.1f} "
              f"times answer appears")

    # Would a different grading rule change results?
    print(f"\n--- Hypothetical: 'Answer Anywhere' Grading ---")
    base_anywhere = sum(1 for a in baseline_audits if a['answer_anywhere'])
    pert_anywhere = sum(1 for a in perturbed_audits if a['answer_anywhere'])
    print(f"  Baseline 'anywhere' accuracy:  {base_anywhere}/{len(baseline_audits)} "
          f"({base_anywhere/len(baseline_audits)*100:.0f}%)")
    print(f"  Perturbed 'anywhere' accuracy: {pert_anywhere}/{len(perturbed_audits)} "
          f"({pert_anywhere/len(perturbed_audits)*100:.0f}%)")
    print(f"  vs last-integer accuracy: base={data['baseline_accuracy']:.0%}, "
          f"pert={sum(1 for a in perturbed_audits if a['last_correct'])/len(perturbed_audits):.0%}")


if __name__ == "__main__":
    result_dir = Path(__file__).parent
    files = [
        ("Qwen3-4B 2-tok n=10",
         result_dir / "sensitivity_sweet_spot_random_noise_t2_results.json"),
        ("DeepSeek 2-tok n=10",
         result_dir / "sensitivity_sweet_spot_random_noise_t2_deepseekr1distillqwen1.5b_results.json"),
        ("phi-2 2-tok n=3",
         result_dir / "sensitivity_sweet_spot_random_noise_t2_phi2_results.json"),
    ]
    for label, path in files:
        if path.exists():
            try:
                audit_file(str(path), label)
            except Exception as e:
                print(f"Error: {label}: {e}")
