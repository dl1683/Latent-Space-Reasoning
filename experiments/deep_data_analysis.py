"""
Deep data analysis -- Tesla workflow phase.
Mine existing experimental data for patterns we haven't examined.
"""
import json
import re
import math
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent

def load_json(name):
    p = DATA_DIR / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

# ── Load all relevant datasets ──
sweet_spot = load_json("sensitivity_sweet_spot_results.json")
sweet_noise = load_json("sensitivity_sweet_spot_random_noise_results.json")
sweet_t1 = load_json("sensitivity_sweet_spot_random_noise_t1_results.json")
sweet_t2 = load_json("sensitivity_sweet_spot_random_noise_t2_results.json")
sweet_zero = load_json("sensitivity_sweet_spot_zero_embedding_results.json")
sweet_mean = load_json("sensitivity_sweet_spot_mean_embedding_results.json")
nested_easy = load_json("sensitivity_nested_easy_results.json")
nested_easy_noise = load_json("sensitivity_easy_nested_random_noise_results.json")
calibration = load_json("calibration_nested_results.json")

print("=" * 80)
print("DEEP DATA ANALYSIS -- TESLA WORKFLOW")
print("=" * 80)

# ── 1. Per-task structural analysis ──
print("\n\n1. PER-TASK STRUCTURAL ANALYSIS")
print("-" * 60)

if sweet_spot:
    baseline = {r["task_id"]: r for r in sweet_spot["baseline_results"]}

    # Extract task properties from calibration data
    task_props = {}
    if calibration:
        for r in calibration.get("results", []):
            tid = r.get("task_id", "")
            task_props[tid] = {
                "expression": r.get("expression", ""),
                "correct_answer": r.get("correct_answer", 0),
                "n_steps": r.get("n_steps", 0),
            }

    # Build per-task profile across ALL conditions
    task_profiles = {}
    for r in sweet_spot["baseline_results"]:
        tid = r["task_id"]
        task_profiles[tid] = {
            "baseline_correct": r["correct"],
            "baseline_time": r["time"],
            "correct_answer": r["correct_answer"],
            "n_steps": r.get("n_steps", 0),
            "answer_magnitude": abs(r["correct_answer"]),
            "latent_results": [],  # (correct, time) per latent
            "noise_results": [],
            "t1_results": [],
            "t2_results": [],
        }

    # Helper to extract per-task results from sensitivity_results structure
    def extract_task_results(data, key="sensitivity_results"):
        """Extract per-latent per-task results from JSON data."""
        results_by_task = defaultdict(list)
        for lr in data.get(key, []):
            for r in lr.get("task_results", []):
                tid = r["task_id"]
                results_by_task[tid].append({
                    "correct": r["correct"],
                    "time": r.get("time", 0),
                })
        return results_by_task

    # Add latent-projected results (8-token)
    lat_by_task = extract_task_results(sweet_spot)
    for tid, results in lat_by_task.items():
        if tid in task_profiles:
            task_profiles[tid]["latent_results"] = results

    # Add 8-token noise results
    if sweet_noise:
        noise_by_task = extract_task_results(sweet_noise)
        for tid, results in noise_by_task.items():
            if tid in task_profiles:
                task_profiles[tid]["noise_results"] = results

    # Add 1-token results
    if sweet_t1:
        t1_by_task = extract_task_results(sweet_t1)
        for tid, results in t1_by_task.items():
            if tid in task_profiles:
                task_profiles[tid]["t1_results"] = results

    # Add 2-token results
    if sweet_t2:
        t2_by_task = extract_task_results(sweet_t2)
        for tid, results in t2_by_task.items():
            if tid in task_profiles:
                task_profiles[tid]["t2_results"] = results

    # ── Analyze task-level patterns ──
    print("\nTask-by-task profile across all conditions:")
    print(f"{'Task':<10} {'Base':>5} {'8-lat':>6} {'8-noi':>6} {'1-tok':>6} {'2-tok':>6} {'Ans':>8} {'Steps':>5} {'BTime':>6}")
    print("-" * 70)

    for tid in sorted(task_profiles.keys()):
        tp = task_profiles[tid]
        b = "Y" if tp["baseline_correct"] else "N"

        lat_rate = sum(1 for r in tp["latent_results"] if r["correct"]) / max(len(tp["latent_results"]), 1)
        noi_rate = sum(1 for r in tp["noise_results"] if r["correct"]) / max(len(tp["noise_results"]), 1)
        t1_rate = sum(1 for r in tp["t1_results"] if r["correct"]) / max(len(tp["t1_results"]), 1)
        t2_rate = sum(1 for r in tp["t2_results"] if r["correct"]) / max(len(tp["t2_results"]), 1)

        print(f"{tid:<10} {b:>5} {lat_rate:>6.0%} {noi_rate:>6.0%} {t1_rate:>6.0%} {t2_rate:>6.0%} {tp['answer_magnitude']:>8} {tp['n_steps']:>5} {tp['baseline_time']:>6.1f}")

    # ── 2. Answer magnitude vs susceptibility ──
    print("\n\n2. ANSWER MAGNITUDE vs SUSCEPTIBILITY TO PERTURBATION")
    print("-" * 60)

    # Group tasks by whether they're helped or hurt
    helped_mags = []
    hurt_mags = []
    unchanged_mags = []

    for tid, tp in task_profiles.items():
        if not tp["latent_results"]:
            continue
        lat_rate = sum(1 for r in tp["latent_results"] if r["correct"]) / len(tp["latent_results"])
        base = 1.0 if tp["baseline_correct"] else 0.0
        delta = lat_rate - base

        if delta > 0.2:
            helped_mags.append(tp["answer_magnitude"])
        elif delta < -0.2:
            hurt_mags.append(tp["answer_magnitude"])
        else:
            unchanged_mags.append(tp["answer_magnitude"])

    print(f"Helped tasks (n={len(helped_mags)}): answer magnitudes = {sorted(helped_mags)}")
    print(f"  Mean: {sum(helped_mags)/max(len(helped_mags),1):.0f}, Median: {sorted(helped_mags)[len(helped_mags)//2] if helped_mags else 'N/A'}")
    print(f"Hurt tasks (n={len(hurt_mags)}): answer magnitudes = {sorted(hurt_mags)}")
    print(f"  Mean: {sum(hurt_mags)/max(len(hurt_mags),1):.0f}, Median: {sorted(hurt_mags)[len(hurt_mags)//2] if hurt_mags else 'N/A'}")
    print(f"Unchanged tasks (n={len(unchanged_mags)}): answer magnitudes = {sorted(unchanged_mags)}")

    # ── 3. Timing distribution analysis ──
    print("\n\n3. TIMING DISTRIBUTION ANALYSIS")
    print("-" * 60)

    # Baseline timing patterns
    baseline_correct_times = [r["time"] for r in sweet_spot["baseline_results"] if r["correct"]]
    baseline_wrong_times = [r["time"] for r in sweet_spot["baseline_results"] if not r["correct"]]

    print(f"Baseline correct (n={len(baseline_correct_times)}):")
    print(f"  Mean: {sum(baseline_correct_times)/max(len(baseline_correct_times),1):.1f}s")
    print(f"  Min: {min(baseline_correct_times):.1f}s, Max: {max(baseline_correct_times):.1f}s")
    print(f"  Sorted: {[f'{t:.1f}' for t in sorted(baseline_correct_times)]}")

    print(f"Baseline wrong (n={len(baseline_wrong_times)}):")
    print(f"  Mean: {sum(baseline_wrong_times)/max(len(baseline_wrong_times),1):.1f}s")
    print(f"  Min: {min(baseline_wrong_times):.1f}s, Max: {max(baseline_wrong_times):.1f}s")
    print(f"  Sorted: {[f'{t:.1f}' for t in sorted(baseline_wrong_times)]}")

    # 2-token timing
    if sweet_t2:
        t2_correct_times = []
        t2_wrong_times = []
        for lr in sweet_t2.get("latent_results", []):
            for r in lr.get("results", []):
                if r["correct"]:
                    t2_correct_times.append(r["time"])
                else:
                    t2_wrong_times.append(r["time"])

        print(f"\n2-token correct (n={len(t2_correct_times)}):")
        print(f"  Mean: {sum(t2_correct_times)/max(len(t2_correct_times),1):.1f}s")
        print(f"  Min: {min(t2_correct_times) if t2_correct_times else 0:.1f}s, Max: {max(t2_correct_times) if t2_correct_times else 0:.1f}s")
        print(f"2-token wrong (n={len(t2_wrong_times)}):")
        print(f"  Mean: {sum(t2_wrong_times)/max(len(t2_wrong_times),1):.1f}s")
        print(f"  Min: {min(t2_wrong_times) if t2_wrong_times else 0:.1f}s, Max: {max(t2_wrong_times) if t2_wrong_times else 0:.1f}s")

    # ── 4. Response style quantitative analysis ──
    print("\n\n4. RESPONSE STYLE METRICS")
    print("-" * 60)

    def analyze_response(text):
        """Extract quantitative features from response text."""
        # Count LaTeX markers
        latex_count = text.count("$$") + text.count("\\times") + text.count("\\div") + text.count("\\boxed")
        # Count markdown headers
        header_count = len(re.findall(r'\*\*Step', text)) + len(re.findall(r'#+\s', text))
        # Count numbers
        numbers = re.findall(r'\d+', text)
        num_count = len(numbers)
        # Count arithmetic operations in text
        ops = len(re.findall(r'[+\-*/÷×]', text))
        # Response length
        char_len = len(text)
        word_len = len(text.split())
        # Informality markers
        informal = len(re.findall(r'\b(ok|okay|let me|hmm|wait|so|well)\b', text, re.I))
        # Formal markers
        formal = len(re.findall(r'\b(Step|Therefore|Thus|Hence|Final Answer|Compute)\b', text, re.I))

        return {
            "latex": latex_count,
            "headers": header_count,
            "numbers": num_count,
            "operations": ops,
            "chars": char_len,
            "words": word_len,
            "informal_markers": informal,
            "formal_markers": formal,
        }

    # Analyze baseline responses
    base_correct_styles = [analyze_response(r["response"]) for r in sweet_spot["baseline_results"] if r["correct"]]
    base_wrong_styles = [analyze_response(r["response"]) for r in sweet_spot["baseline_results"] if not r["correct"]]

    def avg_style(styles):
        if not styles:
            return {}
        keys = styles[0].keys()
        return {k: sum(s[k] for s in styles) / len(styles) for k in keys}

    avg_correct = avg_style(base_correct_styles)
    avg_wrong = avg_style(base_wrong_styles)

    print("Baseline correct vs wrong response style:")
    print(f"{'Metric':<20} {'Correct':>10} {'Wrong':>10} {'Ratio':>10}")
    for k in avg_correct:
        c = avg_correct[k]
        w = avg_wrong[k]
        ratio = c / w if w > 0 else float('inf')
        print(f"{k:<20} {c:>10.1f} {w:>10.1f} {ratio:>10.2f}")

    # ── 5. 2-token vs 8-token: per-task comparison ──
    print("\n\n5. 2-TOKEN vs 8-TOKEN: PER-TASK COMPARISON")
    print("-" * 60)

    if sweet_t2 and sweet_spot:
        print(f"{'Task':<10} {'Base':>5} {'2-tok':>6} {'8-lat':>6} {'Delta 2v8':>10} {'Pattern':>15}")
        print("-" * 60)

        patterns = defaultdict(int)
        for tid in sorted(task_profiles.keys()):
            tp = task_profiles[tid]
            b = tp["baseline_correct"]

            lat_rate = sum(1 for r in tp["latent_results"] if r["correct"]) / max(len(tp["latent_results"]), 1)
            t2_rate = sum(1 for r in tp["t2_results"] if r["correct"]) / max(len(tp["t2_results"]), 1)

            delta = t2_rate - lat_rate

            # Classify pattern
            if b and t2_rate >= 0.67 and lat_rate >= 0.5:
                pattern = "stable_correct"
            elif not b and t2_rate >= 0.67:
                pattern = "2tok_recovers"
            elif b and t2_rate < 0.5:
                pattern = "2tok_regresses"
            elif not b and t2_rate < 0.33 and lat_rate < 0.33:
                pattern = "always_broken"
            elif not b and lat_rate >= 0.5 and t2_rate < 0.5:
                pattern = "8tok_better"
            elif not b and t2_rate >= 0.33:
                pattern = "partial_help"
            else:
                pattern = "other"

            patterns[pattern] += 1
            b_str = "Y" if b else "N"
            print(f"{tid:<10} {b_str:>5} {t2_rate:>6.0%} {lat_rate:>6.0%} {delta:>+10.0%} {pattern:>15}")

        print(f"\nPattern distribution:")
        for p, c in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"  {p}: {c}")

    # ── 6. Zero variance investigation ──
    print("\n\n6. ZERO VARIANCE AT 2 TOKENS -- DEEP DIVE")
    print("-" * 60)

    if sweet_t2:
        latent_results = sweet_t2.get("latent_results", [])
        print(f"Number of 2-token latent vectors tested: {len(latent_results)}")

        for i, lr in enumerate(latent_results):
            results = lr.get("results", [])
            correct = sum(1 for r in results if r["correct"])
            times = [r["time"] for r in results]
            avg_time = sum(times) / len(times) if times else 0

            correct_tasks = [r["task_id"] for r in results if r["correct"]]
            wrong_tasks = [r["task_id"] for r in results if not r["correct"]]

            print(f"\nLatent {i}: {correct}/{len(results)} correct ({correct/len(results)*100:.0f}%)")
            print(f"  Avg time: {avg_time:.1f}s")
            print(f"  Correct: {sorted(correct_tasks)}")
            print(f"  Wrong: {sorted(wrong_tasks)}")

        # Check if EXACT same tasks are correct across all latents
        if len(latent_results) >= 2:
            correct_sets = []
            for lr in latent_results:
                results = lr.get("results", [])
                correct_set = frozenset(r["task_id"] for r in results if r["correct"])
                correct_sets.append(correct_set)

            all_same = all(s == correct_sets[0] for s in correct_sets)
            print(f"\nAre the EXACT same tasks correct across all latents? {'YES' if all_same else 'NO'}")

            if not all_same:
                # Show the differences
                for i, s in enumerate(correct_sets):
                    for j, s2 in enumerate(correct_sets):
                        if i < j:
                            only_i = s - s2
                            only_j = s2 - s
                            print(f"  Latent {i} vs {j}: {len(only_i)} differ (only in {i}: {only_i}, only in {j}: {only_j})")
            else:
                # This is the remarkable finding - investigate WHY these tasks
                print("\n  REMARKABLE: All 3 independent random vectors produce exactly the same correct/wrong split")
                print("  This suggests the task structure (not the latent direction) determines the outcome")
                print("  The 2-token perturbation creates a DETERMINISTIC attractor for each task")

    # ── 7. Operation type analysis ──
    print("\n\n7. OPERATION TYPE vs PERTURBATION SUSCEPTIBILITY")
    print("-" * 60)

    if calibration:
        # Parse expressions to identify operation types
        for r in calibration.get("results", []):
            expr = r.get("expression", "")
            tid = r.get("task_id", "")
            if tid in task_profiles:
                has_mult = "*" in expr or "×" in expr
                has_div = "/" in expr or "÷" in expr
                has_add = "+" in expr
                has_sub = "-" in expr
                nesting = expr.count("(")

                task_profiles[tid]["has_mult"] = has_mult
                task_profiles[tid]["has_div"] = has_div
                task_profiles[tid]["has_add"] = has_add
                task_profiles[tid]["has_sub"] = has_sub
                task_profiles[tid]["nesting"] = nesting
                task_profiles[tid]["expression"] = expr

        # Correlate with recovery/regression
        print(f"{'Task':<10} {'Expr':<30} {'Base':>5} {'2tok':>5} {'Nest':>5} {'Ops':>10}")
        print("-" * 70)
        for tid in sorted(task_profiles.keys()):
            tp = task_profiles[tid]
            if "expression" not in tp:
                continue
            b = "Y" if tp["baseline_correct"] else "N"
            t2_rate = sum(1 for r in tp["t2_results"] if r["correct"]) / max(len(tp["t2_results"]), 1)
            ops = []
            if tp.get("has_mult"): ops.append("*")
            if tp.get("has_div"): ops.append("/")
            if tp.get("has_add"): ops.append("+")
            if tp.get("has_sub"): ops.append("-")

            expr_short = tp.get("expression", "")[:28]
            print(f"{tid:<10} {expr_short:<30} {b:>5} {t2_rate:>5.0%} {tp.get('nesting',0):>5} {''.join(ops):>10}")

    # ── 8. Cross-condition correlation matrix ──
    print("\n\n8. CROSS-CONDITION TASK AGREEMENT")
    print("-" * 60)
    print("Do different conditions agree on which tasks are easy/hard?")

    conditions = {}
    for tid in sorted(task_profiles.keys()):
        tp = task_profiles[tid]
        conditions.setdefault("baseline", {})[tid] = 1.0 if tp["baseline_correct"] else 0.0

        if tp["latent_results"]:
            conditions.setdefault("8-latent", {})[tid] = sum(1 for r in tp["latent_results"] if r["correct"]) / len(tp["latent_results"])
        if tp["t1_results"]:
            conditions.setdefault("1-token", {})[tid] = sum(1 for r in tp["t1_results"] if r["correct"]) / len(tp["t1_results"])
        if tp["t2_results"]:
            conditions.setdefault("2-token", {})[tid] = sum(1 for r in tp["t2_results"] if r["correct"]) / len(tp["t2_results"])

    # Compute pairwise agreement (Pearson correlation)
    cond_names = sorted(conditions.keys())
    task_ids = sorted(task_profiles.keys())

    def pearson(x, y):
        n = len(x)
        if n < 3:
            return float('nan')
        mx = sum(x) / n
        my = sum(y) / n
        sx = math.sqrt(sum((xi - mx)**2 for xi in x) / n)
        sy = math.sqrt(sum((yi - my)**2 for yi in y) / n)
        if sx == 0 or sy == 0:
            return float('nan')
        return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n * sx * sy)

    print(f"\n{'':>12}", end="")
    for cn in cond_names:
        print(f"{cn:>12}", end="")
    print()

    for cn1 in cond_names:
        print(f"{cn1:>12}", end="")
        for cn2 in cond_names:
            vals1 = [conditions[cn1].get(t, 0) for t in task_ids if t in conditions[cn1] and t in conditions[cn2]]
            vals2 = [conditions[cn2].get(t, 0) for t in task_ids if t in conditions[cn1] and t in conditions[cn2]]
            r = pearson(vals1, vals2)
            print(f"{r:>12.3f}", end="")
        print()

    # ── 9. Transition matrix: what happens to each task category across conditions ──
    print("\n\n9. TASK TRANSITION ANALYSIS: BASELINE -> 2-TOKEN")
    print("-" * 60)
    print("For each task, track how it moves through conditions:")

    # Categorize tasks
    categories = {
        "baseline_correct_2tok_correct": [],
        "baseline_correct_2tok_wrong": [],
        "baseline_wrong_2tok_correct": [],
        "baseline_wrong_2tok_wrong": [],
    }

    for tid in sorted(task_profiles.keys()):
        tp = task_profiles[tid]
        if not tp["t2_results"]:
            continue

        b = tp["baseline_correct"]
        t2_rate = sum(1 for r in tp["t2_results"] if r["correct"]) / len(tp["t2_results"])
        t2_correct = t2_rate >= 0.5  # majority vote

        key = f"baseline_{'correct' if b else 'wrong'}_2tok_{'correct' if t2_correct else 'wrong'}"
        categories[key].append(tid)

    for cat, tasks in categories.items():
        print(f"\n{cat} (n={len(tasks)}):")
        for tid in tasks:
            tp = task_profiles[tid]
            t2_rate = sum(1 for r in tp["t2_results"] if r["correct"]) / len(tp["t2_results"])
            expr = tp.get("expression", "?")
            print(f"  {tid}: {expr[:40]:<42} answer={tp['correct_answer']:<8} t2_rate={t2_rate:.0%}")

    # ── 10. Information-theoretic analysis ──
    print("\n\n10. ENTROPY ANALYSIS")
    print("-" * 60)
    print("How much uncertainty does each condition have?")

    def binary_entropy(p):
        if p <= 0 or p >= 1:
            return 0
        return -(p * math.log2(p) + (1-p) * math.log2(1-p))

    for cond_name, cond_data in conditions.items():
        vals = list(cond_data.values())
        mean_rate = sum(vals) / len(vals)
        per_task_entropy = [binary_entropy(v) for v in vals]
        mean_entropy = sum(per_task_entropy) / len(per_task_entropy)

        print(f"{cond_name:>12}: mean_rate={mean_rate:.2f}, mean_entropy={mean_entropy:.3f}")

    print("\nPer-task entropy (higher = more variable across latents within condition):")
    for cond_name, cond_data in conditions.items():
        if cond_name == "baseline":
            continue  # baseline is deterministic
        print(f"\n{cond_name}:")
        for tid in sorted(task_profiles.keys()):
            if tid in cond_data:
                rate = cond_data[tid]
                ent = binary_entropy(rate)
                if ent > 0.5:  # high uncertainty tasks
                    print(f"  {tid}: rate={rate:.2f}, entropy={ent:.3f} <- HIGH UNCERTAINTY")

print("\n\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
