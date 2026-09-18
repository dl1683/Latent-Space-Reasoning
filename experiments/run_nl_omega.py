"""Experiment #83: Natural Language Omega.

Does the code-domain omega protocol transfer to natural language?
Same model (Qwen3-1.7B-Base), same 11-bin digit readout, but NL word
problems instead of code prefixes.

STRUCTURE MAPPING:
  Code domain          ->  NL domain
  Variable binding     ->  Quantity established in narrative
  Nested scope         ->  Multi-step word problem
  Surface comment      ->  Narrative description
  Identity (var=var)   ->  Restate the answer
  Heavy (var=var+1)    ->  Modify the quantity
  Collision (_=val)    ->  Mention confusing number
  Query (print(var))   ->  Ask for the number

Design: 3 cells x 2 pairs x 4 conditions x 2 surfaces = 48 readouts
+ non-commutativity test: 3 cells x 2 orders x 2 surfaces = 12 readouts
Total: 60 readouts, ~30s after model load.
"""
import gc
import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter
from omega import tv

MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "UNUSED",
    "templates": {},
}

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]

CELLS = [
    {
        "name": "birds",
        "prefix": "There were 5 birds on a wire. 2 flew away.\n",
        "query": "Birds remaining: ",
        "answer": 3,
        "original_n": 5,
        "identity": "3 birds are still on the wire.\n",
        "heavy": "Then 1 more flew away.\n",
        "collision": "5 birds were first counted.\n",
    },
    {
        "name": "fish",
        "prefix": "A pond had 9 fish. A fisherman caught 2.\n",
        "query": "Fish in pond: ",
        "answer": 7,
        "original_n": 9,
        "identity": "7 fish remain in the pond.\n",
        "heavy": "Then 1 more was caught.\n",
        "collision": "The fisherman wanted 9 fish.\n",
    },
    {
        "name": "stars",
        "prefix": "Tonight 8 stars were visible. Clouds covered 3.\n",
        "query": "Stars still visible: ",
        "answer": 5,
        "original_n": 8,
        "identity": "5 stars can still be seen.\n",
        "heavy": "Then 1 more was covered.\n",
        "collision": "Last night there were 8 stars.\n",
    },
]

SURFACES = {
    "narrative_a": {
        "birds": "The rest stayed calmly.\n",
        "fish": "The water was still.\n",
        "stars": "The sky was clear otherwise.\n",
    },
    "narrative_b": {
        "birds": "They kept sitting quietly.\n",
        "fish": "The pond was peaceful.\n",
        "stars": "It was a calm night.\n",
    },
    "boundary_a": {
        "birds": "\n", "fish": "\n", "stars": "\n",
    },
    "boundary_b": {
        "birds": "...\n", "fish": "...\n", "stars": "...\n",
    },
}

EQUIV_PAIRS = [
    ("narrative_a", "narrative_b", "narrative_pair"),
    ("boundary_a", "boundary_b", "boundary_pair"),
]


def run():
    print("=== Exp #83: Natural Language Omega ===")
    print("Testing code-domain omega protocol on NL word problems.\n")

    adapter = ModelAdapter(MODEL_CFG)
    import torch
    adapter.mdl.half()
    gc.collect()
    print("Model loaded (float16)")

    # VALIDATION: Does the model predict the right digit for bare prefix+query?
    print("\n--- Validation: bare prefix+query ---")
    for cell in CELLS:
        prefix_text = cell["prefix"]
        query = cell["query"]
        prefix_state = adapter.get_state_after_prefix(prefix_text)
        dist = adapter.get_dist_from_state(prefix_state, query)
        predicted = DIGIT_LABELS[int(np.argmax(dist))]
        correct = str(cell["answer"])
        prob = float(dist[cell["answer"]])
        print(f"  {cell['name']}: predicted={predicted} (P={prob:.3f}), expected={correct}"
              f"  {'OK' if predicted == correct else 'WRONG'}")
        del prefix_state
        gc.collect()

    t0 = time.time()
    all_dists = {}
    n_done = 0

    # Main grid
    for cell in CELLS:
        prefix_text = cell["prefix"]
        query = cell["query"]
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        conditions = {
            "empty": "",
            "identity": cell["identity"],
            "heavy": cell["heavy"],
            "collision": cell["collision"],
        }

        for surf_name in SURFACES:
            surf_text = SURFACES[surf_name][cell["name"]]
            for cond_name, cond_text in conditions.items():
                full_suffix = surf_text + cond_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                all_dists[(cell["name"], surf_name, cond_name)] = dist
                n_done += 1

        # Non-commutativity: identity then collision vs collision then identity
        for surf_name in ["narrative_a", "narrative_b"]:
            surf_text = SURFACES[surf_name][cell["name"]]
            # Order A: identity then collision
            suf_a = surf_text + cell["identity"] + cell["collision"] + query
            dist_a = adapter.get_dist_from_state(prefix_state, suf_a)
            all_dists[(cell["name"], surf_name, "id_then_col")] = dist_a
            n_done += 1

            # Order B: collision then identity
            suf_b = surf_text + cell["collision"] + cell["identity"] + query
            dist_b = adapter.get_dist_from_state(prefix_state, suf_b)
            all_dists[(cell["name"], surf_name, "col_then_id")] = dist_b
            n_done += 1

        del prefix_state
        gc.collect()
        elapsed = time.time() - t0
        print(f"  Cell {cell['name']} done, {n_done} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} readouts in {elapsed:.1f}s")

    # Compute TV distances for equiv pairs
    results = []
    for cell in CELLS:
        cn = cell["name"]
        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            for cond in ["empty", "identity", "heavy", "collision"]:
                ka = (cn, surf_a, cond)
                kb = (cn, surf_b, cond)
                if ka in all_dists and kb in all_dists:
                    cell_tv = float(tv(all_dists[ka], all_dists[kb]))
                    flip = (DIGIT_LABELS[int(np.argmax(all_dists[ka]))] !=
                            DIGIT_LABELS[int(np.argmax(all_dists[kb]))])
                    results.append({
                        "cell": cn,
                        "pair": pair_name,
                        "continuation": cond,
                        "tv": cell_tv,
                        "flip": flip,
                    })

    # Non-commutativity results
    noncomm_results = []
    for cell in CELLS:
        cn = cell["name"]
        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            if pair_name != "narrative_pair":
                continue
            for order in ["id_then_col", "col_then_id"]:
                ka = (cn, surf_a, order)
                kb = (cn, surf_b, order)
                if ka in all_dists and kb in all_dists:
                    cell_tv = float(tv(all_dists[ka], all_dists[kb]))
                    noncomm_results.append({
                        "cell": cn,
                        "order": order,
                        "tv": cell_tv,
                    })

    # Print main table
    for pair_name in ["narrative_pair", "boundary_pair"]:
        print(f"\n=== NL OMEGA: {pair_name} ===")
        print(f"{'Cell':>10s} {'empty':>8s} {'identity':>10s} {'id_amp':>8s}"
              f" {'heavy':>8s} {'hvy_amp':>8s} {'collision':>10s} {'col_amp':>8s}")
        print("-" * 75)

        for cell in CELLS:
            cn = cell["name"]
            e = np.mean([r["tv"] for r in results if r["cell"] == cn and r["pair"] == pair_name and r["continuation"] == "empty"]) or 0.001
            i = np.mean([r["tv"] for r in results if r["cell"] == cn and r["pair"] == pair_name and r["continuation"] == "identity"]) or 0
            h = np.mean([r["tv"] for r in results if r["cell"] == cn and r["pair"] == pair_name and r["continuation"] == "heavy"]) or 0
            c = np.mean([r["tv"] for r in results if r["cell"] == cn and r["pair"] == pair_name and r["continuation"] == "collision"]) or 0
            e = max(e, 0.001)
            print(f"{cn:>10s} {e:8.4f} {i:10.4f} {i/e:7.1f}x"
                  f" {h:8.4f} {h/e:7.1f}x {c:10.4f} {c/e:7.1f}x")

        # Averages
        e_all = np.mean([r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "empty"]) or 0.001
        i_all = np.mean([r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "identity"]) or 0
        h_all = np.mean([r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "heavy"]) or 0
        c_all = np.mean([r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "collision"]) or 0
        e_all = max(e_all, 0.001)
        print(f"{'MEAN':>10s} {e_all:8.4f} {i_all:10.4f} {i_all/e_all:7.1f}x"
              f" {h_all:8.4f} {h_all/e_all:7.1f}x {c_all:10.4f} {c_all/e_all:7.1f}x")

    # Non-commutativity
    print(f"\n=== NON-COMMUTATIVITY TEST ===")
    print(f"{'Cell':>10s} {'id->col TV':>12s} {'col->id TV':>12s} {'ratio':>8s} {'commutes?':>10s}")
    print("-" * 55)
    for cell in CELLS:
        cn = cell["name"]
        ic = [r["tv"] for r in noncomm_results if r["cell"] == cn and r["order"] == "id_then_col"]
        ci = [r["tv"] for r in noncomm_results if r["cell"] == cn and r["order"] == "col_then_id"]
        if ic and ci:
            ic_tv = np.mean(ic)
            ci_tv = np.mean(ci)
            ratio = ic_tv / ci_tv if ci_tv > 0.001 else float('inf')
            commutes = "YES" if 0.7 < ratio < 1.3 else "NO"
            print(f"{cn:>10s} {ic_tv:12.4f} {ci_tv:12.4f} {ratio:7.2f}x  {commutes:>10s}")

    # Code domain comparison
    print(f"\n=== COMPARISON: NL vs CODE DOMAIN ===")
    print("Code domain results (from exp #73, depth3_nested template):")
    print("  strong_assert: empty=0.004, identity=13.5x, heavy=12.9x, collision=11.2x")
    print("  boundary_pair: empty=0.013, identity=6.0x, heavy=2.7x, collision=2.8x")
    print("\nNL domain results (this experiment):")
    for pair_name in ["narrative_pair", "boundary_pair"]:
        e = np.mean([r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "empty"]) or 0.001
        i = np.mean([r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "identity"]) or 0
        h = np.mean([r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "heavy"]) or 0
        c = np.mean([r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "collision"]) or 0
        e = max(e, 0.001)
        print(f"  {pair_name}: empty={e:.4f}, identity={i/e:.1f}x, heavy={h/e:.1f}x, collision={c/e:.1f}x")

    # Save
    summary = {
        "experiment": 83,
        "name": "nl_omega",
        "n_readouts": n_done,
        "results": results,
        "noncomm_results": noncomm_results,
    }
    result_dir = Path("experiments/results/omega_nl")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
