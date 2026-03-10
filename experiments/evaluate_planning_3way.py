"""
Evaluate 3-way planning comparison by reading actual outputs.

Merges baseline + perturbation (from planning_comparison_results.json) with
evolution (from planning_evolution_results.json) and presents outputs for
quality comparison.
"""

import json
import re
from pathlib import Path


def strip_think_block(text):
    """Remove <think>...</think> blocks and 'assistant' prefix."""
    text = text.strip()
    if text.startswith("assistant"):
        text = text[len("assistant"):].strip()
    # Remove complete think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If still starts with unclosed <think>, remove everything up to </think> or first heading
    if text.startswith("<think>"):
        # Try to find end
        end = text.find("</think>")
        if end >= 0:
            text = text[end + len("</think>"):].strip()
        else:
            # Find first markdown heading or numbered list
            for pattern in [r"\n#{1,4} ", r"\n\d+\.", r"\n\*\*"]:
                m = re.search(pattern, text)
                if m:
                    text = text[m.start():].strip()
                    break
    return text


def extract_think_block(text):
    """Extract just the think block content."""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # Unclosed think block
    if "<think>" in text:
        start = text.find("<think>") + len("<think>")
        return text[start:].strip()
    return ""


def main():
    base_dir = Path(__file__).parent

    # Load baseline + perturbation
    with open(base_dir / "planning_comparison_results.json") as f:
        bp_data = json.load(f)

    # Load evolution
    evo_path = base_dir / "planning_evolution_results.json"
    if not evo_path.exists():
        print(f"ERROR: {evo_path} not found. Run run_evolution_planning.py first.")
        return

    with open(evo_path) as f:
        evo_data = json.load(f)

    tasks = bp_data["tasks"]

    print(f"{'='*80}")
    print(f"3-WAY PLANNING COMPARISON: ACTUAL OUTPUT QUALITY")
    print(f"{'='*80}")
    print(f"Tasks: {len(tasks)}")
    print(f"Baseline: max_new_tokens={bp_data['metadata']['max_new_tokens']}")
    print(f"Evolution: max_new_tokens={evo_data['metadata']['max_new_tokens']}")
    print()

    for task_id in tasks:
        print(f"\n{'#'*80}")
        print(f"# TASK: {task_id}")
        print(f"{'#'*80}")

        # Baseline
        baseline = [o for o in bp_data["outputs"]
                    if o["condition"] == "greedy_baseline" and o["task_id"] == task_id][0]
        b_full = baseline["response"]
        b_plan = strip_think_block(b_full)
        b_think = extract_think_block(b_full)

        print(f"\n--- BASELINE (greedy, temp=0) ---")
        print(f"Total: {baseline['word_count']}w | Plan (after think): {len(b_plan.split())}w | Think: {len(b_think.split())}w")
        print(f"\nPLAN OUTPUT:")
        print(b_plan[:2000] if len(b_plan) > 2000 else b_plan)
        if len(b_plan) > 2000:
            print(f"... [{len(b_plan) - 2000} more chars]")

        # Perturbation (show best and summary)
        perturbs = [o for o in bp_data["outputs"]
                    if o["condition"] == "random_perturbation" and o["task_id"] == task_id]
        print(f"\n--- PERTURBATION (2-tok noise, temp=0, {len(perturbs)} seeds) ---")
        for i, p in enumerate(perturbs):
            p_plan = strip_think_block(p["response"])
            p_think = extract_think_block(p["response"])
            print(f"  Seed {p['seed']}: {p['word_count']}w total | Plan: {len(p_plan.split())}w | Think: {len(p_think.split())}w")

        # Show first perturbation plan
        p0_plan = strip_think_block(perturbs[0]["response"])
        print(f"\nPERTURBATION seed={perturbs[0]['seed']} PLAN OUTPUT:")
        print(p0_plan[:2000] if len(p0_plan) > 2000 else p0_plan)
        if len(p0_plan) > 2000:
            print(f"... [{len(p0_plan) - 2000} more chars]")

        # Evolution
        evo_outputs = [o for o in evo_data["outputs"]
                       if o["task_id"] == task_id and "error" not in o]
        if evo_outputs:
            print(f"\n--- EVOLUTION (trained scorer + soft prompt decode, temp=0, {len(evo_outputs)} seeds) ---")
            for e in evo_outputs:
                # Standard decode (likely same as baseline)
                evo_std = strip_think_block(e.get("response_evo_decode", ""))
                # Soft prompt decode (should differ)
                evo_soft = strip_think_block(e.get("response_soft_prompt", ""))
                print(f"  Seed {e['seed']}: score={e.get('evolution_score', 0):.3f} | "
                      f"std_decode={e.get('word_count_evo', 0)}w | "
                      f"soft_prompt={e.get('word_count_soft', 0)}w | "
                      f"gens={e.get('evolution_generations', 0)}")

            # Check if standard decode is same as baseline
            e0_std = strip_think_block(evo_outputs[0].get("response_evo_decode", ""))
            if e0_std == b_plan:
                print("\n  NOTE: Standard decode is IDENTICAL to baseline (expected at temp=0)")

            # Show first soft prompt plan
            e0_soft = strip_think_block(evo_outputs[0].get("response_soft_prompt", ""))
            print(f"\nEVOLUTION seed={evo_outputs[0]['seed']} SOFT PROMPT PLAN OUTPUT:")
            print(e0_soft[:2000] if len(e0_soft) > 2000 else e0_soft)
            if len(e0_soft) > 2000:
                print(f"... [{len(e0_soft) - 2000} more chars]")
        else:
            evo_errors = [o for o in evo_data["outputs"]
                          if o["task_id"] == task_id and "error" in o]
            if evo_errors:
                print(f"\n--- EVOLUTION: {len(evo_errors)} ERRORS ---")
                for e in evo_errors:
                    print(f"  Seed {e['seed']}: {e['error']}")
            else:
                print(f"\n--- EVOLUTION: No outputs for this task ---")

    print(f"\n{'='*80}")
    print("READY FOR LLM-AS-JUDGE EVALUATION")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
