"""
Analyze reasoning QUALITY (not just accuracy) across perturbation conditions.

Key question: Does perturbation change the STRUCTURE and COHERENCE of reasoning
chains, independent of whether the final answer is correct?

This addresses the concern that we've been measuring arithmetic accuracy
when the original goal was planning/reasoning quality.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, str(Path(project_root) / "src"))
from latent_reasoning.core.heuristic_scorer import HeuristicScorer


def analyze_response_quality(response: str) -> dict:
    """Analyze a single response for reasoning quality indicators."""
    # Strip <think>...</think> to get just reasoning chain
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    reasoning = think_match.group(1) if think_match else response

    # Get the final answer part (after </think>)
    answer_part = response.split('</think>')[-1] if '</think>' in response else ''

    words = reasoning.split()
    word_count = len(words)

    # Reasoning chain indicators
    step_markers = len(re.findall(r'(?:step|first|then|next|finally|so|therefore|thus|because|since|let me|I need to|now)', reasoning.lower()))

    # Self-correction indicators
    corrections = len(re.findall(r'(?:wait|actually|no,|oops|let me re|I made|mistake|error|hmm|correct)', reasoning.lower()))

    # Computation steps (actual math being done)
    computations = len(re.findall(r'\d+\s*[+\-*/×÷=]\s*\d+', reasoning))

    # Structured breakdown (numbered steps, bullets)
    numbered_steps = len(re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|Step \d+)', reasoning))

    # LaTeX/boxed answers (structured output)
    latex_elements = len(re.findall(r'\\(?:boxed|frac|times|div|text|mathbf)', reasoning))

    # Logical connectives
    logical_flow = len(re.findall(r'\b(?:therefore|thus|hence|so|because|since|if|then|given|implies|means)\b', reasoning.lower()))

    # Paragraph structure
    paragraphs = [p.strip() for p in reasoning.split('\n\n') if p.strip()]

    # Token budget usage
    hit_token_cap = word_count > 900  # rough proxy for 1024 token cap

    return {
        'word_count': word_count,
        'step_markers': step_markers,
        'corrections': corrections,
        'computations': computations,
        'numbered_steps': numbered_steps,
        'latex_elements': latex_elements,
        'logical_flow': logical_flow,
        'paragraphs': len(paragraphs),
        'hit_token_cap': hit_token_cap,
        'answer_part_length': len(answer_part.split()),
    }


def load_and_analyze(result_path: str, label: str = "") -> dict:
    """Load a results JSON and analyze reasoning quality across conditions."""
    with open(result_path) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"REASONING QUALITY ANALYSIS: {label or result_path}")
    print(f"Model: {data.get('model', 'unknown')}, Task: {data.get('task_type', 'unknown')}")
    print(f"Baseline accuracy: {data['baseline_accuracy']:.0%}")
    print(f"{'='*70}")

    # Analyze baseline responses
    baseline_results = data['baseline_results']
    baseline_quality = []
    for task in baseline_results:
        response = task.get('response', '') or task.get('response_raw', '')
        if response:
            q = analyze_response_quality(response)
            q['correct'] = task['correct']
            q['task_id'] = task['task_id']
            baseline_quality.append(q)

    # Analyze perturbed responses
    sensitivity_results = data.get('sensitivity_results', [])
    perturbed_quality = defaultdict(list)

    for li, latent_data in enumerate(sensitivity_results):
        for task in latent_data['task_results']:
            response = task.get('response', '') or task.get('response_raw', '')
            if response:
                q = analyze_response_quality(response)
                q['correct'] = task['correct']
                q['task_id'] = task['task_id']
                perturbed_quality[li].append(q)

    # Aggregate and compare
    def avg_metric(quality_list, metric):
        vals = [q[metric] for q in quality_list]
        return sum(vals) / len(vals) if vals else 0

    def avg_metric_by_correct(quality_list, metric, correct_val):
        vals = [q[metric] for q in quality_list if q['correct'] == correct_val]
        return sum(vals) / len(vals) if vals else 0

    print(f"\n--- Baseline Reasoning Quality (n={len(baseline_quality)}) ---")
    metrics = ['word_count', 'step_markers', 'corrections', 'computations',
               'numbered_steps', 'logical_flow', 'paragraphs']

    for m in metrics:
        val = avg_metric(baseline_quality, m)
        val_correct = avg_metric_by_correct(baseline_quality, m, True)
        val_wrong = avg_metric_by_correct(baseline_quality, m, False)
        print(f"  {m:20s}: {val:6.1f}  (correct={val_correct:5.1f}, wrong={val_wrong:5.1f})")

    cap_rate = sum(1 for q in baseline_quality if q['hit_token_cap']) / len(baseline_quality)
    print(f"  {'token_cap_rate':20s}: {cap_rate:.0%}")

    print(f"\n--- Perturbed Reasoning Quality ({len(perturbed_quality)} directions) ---")
    all_perturbed = []
    for li in sorted(perturbed_quality.keys()):
        pq = perturbed_quality[li]
        all_perturbed.extend(pq)
        acc = sum(1 for q in pq if q['correct']) / len(pq)
        wc = avg_metric(pq, 'word_count')
        steps = avg_metric(pq, 'step_markers')
        corr = avg_metric(pq, 'corrections')
        comp = avg_metric(pq, 'computations')
        logic = avg_metric(pq, 'logical_flow')
        cap = sum(1 for q in pq if q['hit_token_cap']) / len(pq)
        print(f"  L{li}: acc={acc:.0%}, words={wc:.0f}, steps={steps:.1f}, "
              f"corrections={corr:.1f}, computations={comp:.1f}, "
              f"logical_flow={logic:.1f}, cap_rate={cap:.0%}")

    # Delta analysis
    print(f"\n--- Delta: Perturbed vs Baseline ---")
    for m in metrics:
        base_val = avg_metric(baseline_quality, m)
        pert_val = avg_metric(all_perturbed, m)
        delta = pert_val - base_val
        pct = (delta / base_val * 100) if base_val != 0 else 0
        direction = "+" if delta > 0 else ""
        print(f"  {m:20s}: {direction}{delta:5.1f} ({direction}{pct:.0f}%)")

    # Key insight: does perturbation change quality differently for
    # tasks it FIXES vs tasks it BREAKS?
    print(f"\n--- Quality Change by Outcome Change ---")

    baseline_correct_set = {q['task_id'] for q in baseline_quality if q['correct']}

    fixed_quality = []  # wrong→right
    broken_quality = []  # right→wrong
    maintained_quality = []  # same outcome

    for li in perturbed_quality:
        for pq in perturbed_quality[li]:
            bq = next((q for q in baseline_quality if q['task_id'] == pq['task_id']), None)
            if bq is None:
                continue

            was_correct = bq['correct']
            now_correct = pq['correct']

            quality_delta = {m: pq[m] - bq[m] for m in metrics}

            if not was_correct and now_correct:
                fixed_quality.append(quality_delta)
            elif was_correct and not now_correct:
                broken_quality.append(quality_delta)
            else:
                maintained_quality.append(quality_delta)

    for category, deltas, label in [
        (fixed_quality, fixed_quality, "FIXED (wrong->right)"),
        (broken_quality, broken_quality, "BROKEN (right->wrong)"),
        (maintained_quality, maintained_quality, "MAINTAINED (same)"),
    ]:
        if not deltas:
            print(f"  {label}: no instances")
            continue
        print(f"  {label} (n={len(deltas)}):")
        for m in ['word_count', 'step_markers', 'corrections', 'computations', 'logical_flow']:
            vals = [d[m] for d in deltas]
            avg_delta = sum(vals) / len(vals)
            direction = "+" if avg_delta > 0 else ""
            print(f"    {m:20s}: {direction}{avg_delta:.1f}")

    # Heuristic scorer (planning-oriented)
    scorer = HeuristicScorer()

    print(f"\n--- Heuristic Scorer (Planning Quality) ---")
    base_scores = []
    for task in baseline_results:
        response = task.get('response', '') or task.get('response_raw', '')
        if response:
            s = scorer.score(response)
            base_scores.append(s)

    pert_scores = []
    for li in sensitivity_results:
        for task in li['task_results']:
            response = task.get('response', '') or task.get('response_raw', '')
            if response:
                s = scorer.score(response)
                pert_scores.append(s)

    if base_scores and pert_scores:
        for attr in ['structure_score', 'depth_score', 'action_score', 'coherence_score', 'overall_score']:
            b = sum(getattr(s, attr) for s in base_scores) / len(base_scores)
            p = sum(getattr(s, attr) for s in pert_scores) / len(pert_scores)
            delta = p - b
            direction = "+" if delta > 0 else ""
            print(f"  {attr:20s}: base={b:.3f}, pert={p:.3f}, delta={direction}{delta:.3f}")

    return {
        'baseline_quality': baseline_quality,
        'perturbed_quality': perturbed_quality,
        'all_perturbed': all_perturbed,
    }


if __name__ == "__main__":
    import glob

    # Find all result files
    result_dir = Path(__file__).parent

    # Priority: 2-tok results (our main condition)
    key_files = [
        # Qwen3-4B 2-tok n=10 (primary)
        ("Qwen3-4B 2-tok n=10",
         result_dir / "sensitivity_sweet_spot_random_noise_t2_results.json"),
        # DeepSeek 2-tok n=10
        ("DeepSeek 2-tok n=10",
         result_dir / "sensitivity_sweet_spot_random_noise_t2_deepseekr1distillqwen1.5b_results.json"),
        # phi-2
        ("phi-2 2-tok n=3",
         result_dir / "sensitivity_sweet_spot_random_noise_t2_phi2_results.json"),
    ]

    for label, path in key_files:
        if path.exists():
            try:
                analyze = load_and_analyze(str(path), label)
            except Exception as e:
                print(f"Error analyzing {label}: {e}")
