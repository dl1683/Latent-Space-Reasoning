"""Error taxonomy analysis across warm-start conditions.

sensitivity_results structure: list of dicts, each with:
  - latent_idx, accuracy, n_correct, n_total, task_results (list of per-task dicts)
"""
import json
import os

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))

files = {
    # Main file has both baseline AND 10-latent (latent-projected) sensitivity_results
    'main': os.path.join(EXPERIMENTS_DIR, 'sensitivity_sweet_spot_results.json'),
    'zero_emb': os.path.join(EXPERIMENTS_DIR, 'sensitivity_sweet_spot_zero_embedding_results.json'),
    'mean_emb': os.path.join(EXPERIMENTS_DIR, 'sensitivity_sweet_spot_mean_embedding_results.json'),
    '1_token': os.path.join(EXPERIMENTS_DIR, 'sensitivity_sweet_spot_random_noise_t1_results.json'),
}

data = {}
for name, path in files.items():
    if os.path.exists(path):
        with open(path) as f:
            data[name] = json.load(f)

# Build baseline map from main file
bl = data['main']['baseline_results']
bl_map = {r['task_id']: r for r in bl}
task_ids = sorted(bl_map.keys())

# The main file also has the 10-latent (latent-projected) sensitivity_results
sr_latent_projected = data['main'].get('sensitivity_results', [])


def get_per_task_correctness(sr_list, latent_idx=0):
    """Extract task correctness from sensitivity_results list."""
    if not sr_list or latent_idx >= len(sr_list):
        return {}
    entry = sr_list[latent_idx]
    task_results = entry.get('task_results', [])
    return {r['task_id']: r['correct'] for r in task_results}


def get_all_latent_correctness(sr_list):
    """Get correctness across ALL latents for each task."""
    counts = {}  # tid -> [ok, total]
    for entry in sr_list:
        for r in entry.get('task_results', []):
            tid = r['task_id']
            if tid not in counts:
                counts[tid] = [0, 0]
            counts[tid][1] += 1
            if r['correct']:
                counts[tid][0] += 1
    return counts


# Get sensitivity_results for each condition
cond_sr = {}
for name in ['zero_emb', 'mean_emb', '1_token']:
    if name in data:
        cond_sr[name] = data[name].get('sensitivity_results', [])
    else:
        cond_sr[name] = []

# 8-token = the original 10-latent latent-projected data
counts_8tok = get_all_latent_correctness(sr_latent_projected)

# Per-task correctness matrix
print("=== TASK CORRECTNESS MATRIX ===")
print(f"{'Task':<10} {'BL':>4} {'Zero':>5} {'Mean':>5} {'1tok':>5}   8tok(10 latents)")
print("-" * 65)

for tid in task_ids:
    bl_ok = bl_map[tid]['correct']
    vals = ['OK' if bl_ok else ' X']
    for c in ['zero_emb', 'mean_emb', '1_token']:
        cm = get_per_task_correctness(cond_sr[c], 0)
        if tid in cm:
            vals.append('OK' if cm[tid] else ' X')
        else:
            vals.append('--')

    ok, tot = counts_8tok.get(tid, (0, 0))
    print(f"{tid:<10} {vals[0]:>4} {vals[1]:>5} {vals[2]:>5} {vals[3]:>5}   {ok}/{tot}")

# Failure analysis
baseline_wrong = [tid for tid in task_ids if not bl_map[tid]['correct']]
baseline_right = [tid for tid in task_ids if bl_map[tid]['correct']]

print(f"\nBaseline: {len(baseline_right)}/25 correct, {len(baseline_wrong)}/25 wrong")

print("\n=== BASELINE FAILURES: Recovery rate with 8 tokens (10 latents) ===")
for tid in baseline_wrong:
    ok, tot = counts_8tok.get(tid, (0, 0))
    pct = 100 * ok / max(tot, 1)
    label = "FIXED" if pct > 50 else "still_broken" if pct == 0 else "partial"
    print(f"  {tid}: {ok}/{tot} ({pct:.0f}%) [{label}]")

print("\n=== BASELINE SUCCESSES: Regression rate with 8 tokens (10 latents) ===")
for tid in baseline_right:
    ok, tot = counts_8tok.get(tid, (0, 0))
    if ok < tot:
        pct = 100 * ok / max(tot, 1)
        print(f"  {tid}: {ok}/{tot} ({pct:.0f}%) [REGRESSION]")

# Examine actual errors
print("\n=== SAMPLE ERROR ANALYSIS (baseline failures) ===")
for tid in baseline_wrong[:5]:
    r = bl_map[tid]
    resp = r.get('response', '')
    resp_safe = resp[:300].encode('ascii', 'replace').decode('ascii')
    print(f"\n{tid} (expected={r['correct_answer']}, n_steps={r['n_steps']}):")
    print(f"  Response: {resp_safe}...")

# Compare a recovered task: baseline wrong -> 8tok right
print("\n=== RECOVERED TASK COMPARISON (baseline wrong -> fixed by tokens) ===")
shown = 0
for tid in baseline_wrong:
    ok, tot = counts_8tok.get(tid, (0, 0))
    if ok > tot / 2:  # Majority fixed
        print(f"\n{tid}: Baseline WRONG, 8-tok fixes {ok}/{tot}")
        print(f"  Expected: {bl_map[tid]['correct_answer']}")
        bl_resp = bl_map[tid].get('response', '')[:300]
        bl_safe = bl_resp.encode('ascii', 'replace').decode('ascii')
        print(f"  Baseline response: {bl_safe}...")
        # Find a correct 8-tok response
        for entry in sr_latent_projected:
            for r in entry.get('task_results', []):
                if r['task_id'] == tid and r['correct']:
                    lat_resp = r.get('response', '')[:300]
                    lat_safe = lat_resp.encode('ascii', 'replace').decode('ascii')
                    print(f"  8-tok correct response: {lat_safe}...")
                    break
            else:
                continue
            break
        shown += 1
        if shown >= 3:
            break

# Overall summary
print("\n=== SUMMARY ===")
n_fixed = sum(1 for tid in baseline_wrong
              if counts_8tok.get(tid, (0, 0))[0] > counts_8tok.get(tid, (0, 0))[1] / 2)
n_broken = sum(1 for tid in baseline_right
               if counts_8tok.get(tid, (0, 0))[0] < counts_8tok.get(tid, (0, 0))[1])
print(f"Baseline correct: {len(baseline_right)}/25")
print(f"Baseline failures recovered by 8-tok (>50%): {n_fixed}/{len(baseline_wrong)}")
print(f"Baseline successes with regressions: {n_broken}/{len(baseline_right)}")
print(f"Net effect: +{n_fixed} recovered - {n_broken} regressed")
