"""Omega State Reconstruction vs Appended Repair (Experiment #52).

Fable R3 Rank 3: can a real information boundary prevent revival?

Tests whether replacing textual history by an explicit state record
eliminates presentation-specific divergence while preserving answers.

Design: 6 cells x 2 pair families x 5 treatments x 2 probes = 120 obs.
Each observation = 2 readouts -> 240 total readouts.

Treatments:
  1. original: full program text as-is
  2. summary_nl: natural language summary of program state
  3. appended_overwrite: original + explicit variable reassignment
  4. canonical_record: fresh context with only the declared bindings
  5. incomplete_record: canonical but missing one binding (control)

Probes:
  1. target_query: query the tracked variable (standard)
  2. ortho_query: query a DIFFERENT variable (capability check)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "templates": {
        "depth4_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        def h():\n"
            "            {var} = 9999\n            def k():\n"
            "                {var} = 99999\n                return {var}\n"
            "            k()\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
    },
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

EQUIV_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("\n", "pass\n", "boundary_pair"),
]

PROBES = {
    "target_query": "f()\nprint({var})  # Output: ",
    "ortho_query": "print(type({var}))  # Output: ",
}


def make_treatments(prefix_text, surface, var, val):
    original = prefix_text + surface

    summary = (f"# The variable {var} is currently {val}.\n"
               f"# Functions f, g, h, k are defined and shadow {var}.\n")

    appended = (prefix_text + surface +
                f"{var} = {val}\n")

    canonical = (f"# State record:\n"
                 f"# {var} = {val} (outer scope)\n"
                 f"# f() returns {val}\n")

    incomplete = f"# State record:\n# f() is defined.\n"

    return {
        "original": original,
        "summary_nl": summary,
        "appended_overwrite": appended,
        "canonical_record": canonical,
        "incomplete_record": incomplete,
    }


def run_state_reconstruct():
    adapter = ModelAdapter(MODEL_CFG)
    observations = []
    t0 = time.time()
    n_readouts = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)

        for surf_a_tmpl, surf_b_tmpl, pair_name in EQUIV_PAIRS:
            surf_a = surf_a_tmpl.replace("{var}", var)
            surf_b = surf_b_tmpl.replace("{var}", var)

            treatments_a = make_treatments(prefix_text, surf_a, var, val)
            treatments_b = make_treatments(prefix_text, surf_b, var, val)

            for treat_name in treatments_a:
                for probe_name, probe_tmpl in PROBES.items():
                    probe = probe_tmpl.replace("{var}", var)

                    text_a = treatments_a[treat_name] + probe
                    text_b = treatments_b[treat_name] + probe

                    prefix_state_a = adapter.get_state_after_prefix(
                        treatments_a[treat_name])
                    prefix_state_b = adapter.get_state_after_prefix(
                        treatments_b[treat_name])
                    dist_a = adapter.get_dist_from_state(
                        prefix_state_a, probe)
                    dist_b = adapter.get_dist_from_state(
                        prefix_state_b, probe)
                    d = tv(dist_a, dist_b)
                    n_readouts += 2

                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name,
                        "treatment": treat_name,
                        "probe": probe_name,
                        "tv": float(d),
                    })

                    if n_readouts % 40 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n_readouts}/240 readouts, {elapsed:.0f}s",
                              flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done: {n_readouts} readouts in {elapsed:.1f}s")
    del adapter
    return observations, elapsed


def analyze(observations):
    print("\n=== STATE RECONSTRUCTION ANALYSIS ===\n")

    for pair_name in ["strong_assert_2", "boundary_pair"]:
        pair_obs = [o for o in observations if o["pair"] == pair_name]
        if not pair_obs:
            continue
        print(f"\n--- {pair_name} ---")

        for probe_name in ["target_query", "ortho_query"]:
            print(f"\n  Probe: {probe_name}")
            for treat_name in ["original", "summary_nl", "appended_overwrite",
                                "canonical_record", "incomplete_record"]:
                tvs = [o["tv"] for o in pair_obs
                       if o["treatment"] == treat_name
                       and o["probe"] == probe_name]
                if tvs:
                    print(f"    {treat_name:25s}: mean={np.mean(tvs):.6f} "
                          f"max={np.max(tvs):.6f}")

    print("\n=== KEY QUESTION: Does canonical record eliminate divergence? ===")
    for pair_name in ["strong_assert_2", "boundary_pair"]:
        pair_obs = [o for o in observations if o["pair"] == pair_name]
        orig_target = [o["tv"] for o in pair_obs
                       if o["treatment"] == "original"
                       and o["probe"] == "target_query"]
        canon_target = [o["tv"] for o in pair_obs
                        if o["treatment"] == "canonical_record"
                        and o["probe"] == "target_query"]
        if orig_target and canon_target:
            ratio = np.mean(canon_target) / np.mean(orig_target) if np.mean(orig_target) > 0.0001 else float("inf")
            print(f"  {pair_name}:")
            print(f"    original:  mean={np.mean(orig_target):.6f}")
            print(f"    canonical: mean={np.mean(canon_target):.6f}")
            print(f"    ratio:     {ratio:.3f}x "
                  f"({'ELIMINATES' if ratio < 0.1 else 'REDUCES' if ratio < 0.5 else 'PERSISTS'})")

    print("\n=== CAPABILITY CHECK: Does canonical preserve ortho answers? ===")
    for pair_name in ["strong_assert_2", "boundary_pair"]:
        pair_obs = [o for o in observations if o["pair"] == pair_name]
        orig_ortho = [o["tv"] for o in pair_obs
                      if o["treatment"] == "original"
                      and o["probe"] == "ortho_query"]
        canon_ortho = [o["tv"] for o in pair_obs
                       if o["treatment"] == "canonical_record"
                       and o["probe"] == "ortho_query"]
        if orig_ortho and canon_ortho:
            print(f"  {pair_name}:")
            print(f"    original ortho:  mean={np.mean(orig_ortho):.6f}")
            print(f"    canonical ortho: mean={np.mean(canon_ortho):.6f}")

    return {
        "type": "state_reconstruction",
        "n_observations": len(observations),
        "treatments": ["original", "summary_nl", "appended_overwrite",
                        "canonical_record", "incomplete_record"],
    }


if __name__ == "__main__":
    out_dir = Path("experiments/results/omega_state_reconstruct")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Omega State Reconstruction vs Repair (Exp #52) ===")
    print("Testing: can a real information boundary prevent revival?\n")

    observations, elapsed = run_state_reconstruct()

    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f, indent=2)

    summary = analyze(observations)
    summary["n_readouts"] = len(observations) * 2
    summary["elapsed_s"] = elapsed

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
