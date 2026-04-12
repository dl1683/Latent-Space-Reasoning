"""
Legal Reasoning Comparison: Baseline vs Perturbation vs Evolution

3 conditions on 12 legal analysis tasks:
  1. greedy_baseline  — standard autoregression, temp=0
  2. random_perturbation — 2-token embedding noise + greedy, 5 seeds
  3. evolution        — trained latent scorer guiding search, 5 seeds

Output format is designed for transparent human review:
  - Every record contains the full query text
  - Conditions are grouped per task in the saved JSON
  - A companion human-readable report (.txt) is written alongside the JSON
    so anyone can open it and see query + all responses side-by-side

Evaluation: Codex CLI blind review after run completes.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from latent_reasoning.core.encoder import LLMEncoder
from experiments.harness import auto_calibrate
from experiments.run_latent_sensitivity import decode_with_raw_soft_prompt


# =============================================================================
# Legal Task Definitions
# 12 tasks across 5 legal reasoning categories
# =============================================================================

LEGAL_TASKS = [
    # --- Contract Analysis ---
    {
        "id": "legal_01_contract_ambiguity",
        "category": "contract_analysis",
        "prompt": (
            "A SaaS vendor contract contains the following clause: 'The vendor shall provide "
            "99.9% uptime, calculated monthly, excluding scheduled maintenance windows announced "
            "at least 24 hours in advance. Downtime credits shall be applied at 10% of monthly "
            "fees per hour of excess downtime, up to a maximum of one month's fees.' "
            "The vendor experienced 6 hours of unannounced downtime in March plus 4 hours of "
            "announced maintenance. The client claims they are owed credits for all 10 hours. "
            "The vendor says only the 6 unannounced hours qualify and the credit cap means they "
            "owe at most one month's fees regardless. Analyze the contract clause, identify "
            "ambiguities, determine which interpretation is more defensible, and advise the "
            "client on their strongest legal argument."
        ),
    },
    {
        "id": "legal_02_indemnification_scope",
        "category": "contract_analysis",
        "prompt": (
            "Review this indemnification clause and identify its risks and gaps: "
            "'Party A shall indemnify, defend, and hold harmless Party B from and against any "
            "and all claims, damages, losses, costs, and expenses (including reasonable "
            "attorneys fees) arising out of or relating to Party A's breach of this Agreement "
            "or Party A's negligence.' "
            "Party B is a large enterprise client and Party A is a small startup. "
            "Analyze: (1) what claims this clause covers and excludes, (2) the risk allocation "
            "between parties, (3) whether 'arising out of or relating to' is materially "
            "different from 'arising out of', (4) what a sophisticated Party B would demand "
            "be changed, and (5) whether the clause is mutual or one-sided and implications."
        ),
    },
    {
        "id": "legal_03_force_majeure",
        "category": "contract_analysis",
        "prompt": (
            "A logistics company failed to deliver goods on time due to a combination of: "
            "(1) a cyberattack on their warehouse management system (3-day disruption), and "
            "(2) a regional truck driver shortage caused by a competing company's labor strike. "
            "The contract contains a force majeure clause covering 'acts of God, war, terrorism, "
            "government action, and other events beyond the reasonable control of the parties.' "
            "The buyer is claiming breach of contract and consequential damages of $2.3M. "
            "Analyze whether either event qualifies as force majeure under this clause, how "
            "courts have treated cyberattacks and third-party labor actions as force majeure, "
            "and what arguments the logistics company should make in their defense."
        ),
    },

    # --- Statutory Interpretation ---
    {
        "id": "legal_04_statutory_ambiguity",
        "category": "statutory_interpretation",
        "prompt": (
            "A state consumer protection statute prohibits 'unfair or deceptive acts or "
            "practices in the conduct of any trade or commerce.' A fintech company automatically "
            "enrolled users in a premium subscription tier after a free trial without sending "
            "a separate pre-renewal notification — the enrollment terms were disclosed in the "
            "original 47-page terms of service. The state AG is investigating. "
            "Analyze: (1) whether the conduct likely constitutes an 'unfair' practice under "
            "the FTC Act three-part test commonly adopted by state courts, (2) whether it "
            "constitutes a 'deceptive' practice, (3) what intent standard applies, (4) how "
            "the buried disclosure in the ToS affects the analysis, and (5) what defenses "
            "the company could plausibly raise."
        ),
    },
    {
        "id": "legal_05_preemption_analysis",
        "category": "statutory_interpretation",
        "prompt": (
            "A city ordinance requires all app-based food delivery platforms operating in the "
            "city to: (a) disclose their algorithm-based surge pricing formula to the city "
            "council, (b) cap delivery fees at 15% of order value, and (c) classify delivery "
            "workers as employees rather than independent contractors for workers within city "
            "limits. The platforms argue federal preemption under the Federal Aviation "
            "Administration Authorization Act (FAAAA), which preempts state/local laws 'related "
            "to a price, route, or service of any motor carrier.' "
            "Analyze the preemption argument for each of the three requirements, identifying "
            "which are most and least vulnerable to FAAAA preemption based on existing case law."
        ),
    },

    # --- Case Law Reasoning ---
    {
        "id": "legal_06_negligence_analysis",
        "category": "case_law_reasoning",
        "prompt": (
            "A social media platform's recommendation algorithm repeatedly surfaced extremist "
            "content to a user over six months. The user subsequently committed a violent act, "
            "and victims are suing the platform for negligence. The platform invokes Section 230 "
            "of the Communications Decency Act as a complete defense. "
            "Analyze: (1) whether Section 230 immunizes algorithmic recommendation decisions "
            "or only publisher/speaker liability, distinguishing the Ninth Circuit's ruling in "
            "Force v. Facebook from the Second Circuit's approach, (2) the elements of a "
            "negligence claim the plaintiffs must establish, (3) the foreseeability question "
            "and what evidence would be necessary, and (4) how the Supreme Court's 2023 "
            "Gonzalez v. Google decision (or the Court's avoidance of the Section 230 question) "
            "affects the analysis."
        ),
    },
    {
        "id": "legal_07_ip_fair_use",
        "category": "case_law_reasoning",
        "prompt": (
            "An AI company trained a large language model on a dataset that included "
            "copyrighted books, news articles, and academic papers scraped from the web "
            "without license. The model can generate text in the style of specific authors "
            "and can reproduce verbatim passages when prompted. Publishers are suing for "
            "copyright infringement. The AI company claims fair use. "
            "Apply the four-factor fair use analysis: (1) purpose and character of use "
            "(commercial vs. transformative), (2) nature of the copyrighted works, "
            "(3) amount and substantiality of the portion used, (4) market effect. "
            "Address how Authors Guild v. Google and the Andy Warhol Foundation v. Goldsmith "
            "decision affect the transformativeness analysis, and assess the strength of the "
            "fair use defense."
        ),
    },

    # --- Regulatory Compliance ---
    {
        "id": "legal_08_gdpr_breach",
        "category": "regulatory_compliance",
        "prompt": (
            "A US-based e-commerce company with EU customers suffered a data breach exposing "
            "the names, email addresses, hashed passwords, and purchase histories of 340,000 "
            "EU residents. The breach was discovered internally on a Monday; forensics suggest "
            "it began 11 days earlier. The company has a UK subsidiary that processes the data "
            "under a Data Processing Agreement. "
            "Advise on: (1) GDPR Article 33 notification obligations — to which supervisory "
            "authority, within what timeframe, and what the 72-hour clock started, (2) Article "
            "34 obligations to notify affected individuals and what threshold triggers it, "
            "(3) whether the UK GDPR applies separately post-Brexit and creates parallel "
            "obligations, (4) the maximum exposure under GDPR and UK GDPR, and (5) the three "
            "most important steps to take in the next 48 hours."
        ),
    },
    {
        "id": "legal_09_securities_disclosure",
        "category": "regulatory_compliance",
        "prompt": (
            "A publicly traded tech company's CFO learned in a board meeting that the company's "
            "largest customer (35% of revenue) notified them that it would not renew its "
            "contract expiring in 4 months. Three days later, before any public disclosure, "
            "the CFO sold $4.2M in company stock. The company disclosed the non-renewal two "
            "weeks later; the stock dropped 28%. "
            "Analyze: (1) whether the non-renewal information constitutes material non-public "
            "information under SEC Rule 10b-5, (2) the elements of insider trading liability "
            "the SEC must prove, (3) whether a 10b5-1 trading plan would have protected the "
            "CFO and why it does not here, (4) potential criminal exposure under 18 U.S.C. § "
            "1348, and (5) the company's own disclosure liability under Regulation FD."
        ),
    },

    # --- Legal Strategy & Risk Analysis ---
    {
        "id": "legal_10_litigation_strategy",
        "category": "legal_strategy",
        "prompt": (
            "Your client is a mid-size software company being sued by a patent troll holding "
            "a broad software patent on 'optimizing database queries using machine learning.' "
            "The troll has already settled with 14 other defendants for an average of $800K "
            "each. Your client's annual revenue is $12M. An initial review suggests the patent "
            "may be invalidated on prior art grounds (there is a 2009 academic paper and a 2011 "
            "open-source project that appear to predate the patent's claims). "
            "Design a comprehensive litigation strategy covering: (1) immediate defensive steps, "
            "(2) whether to file an IPR petition at the USPTO vs. litigating invalidity in "
            "district court, with pros/cons of each, (3) the discovery strategy, (4) settlement "
            "negotiation leverage points given the prior art, and (5) how to evaluate the "
            "make-vs-settle decision with specific financial thresholds."
        ),
    },
    {
        "id": "legal_11_employment_termination",
        "category": "legal_strategy",
        "prompt": (
            "A company wants to terminate a senior engineer who has been underperforming for "
            "8 months. The engineer is 58 years old, is a member of a religious minority, and "
            "filed an internal HR complaint about a manager 3 months ago. The company has "
            "documented the performance issues with two written warnings, but the documentation "
            "is inconsistent — one warning references 'attitude issues' alongside technical "
            "failures, and the manager who issued the warnings is the same one the employee "
            "complained about. "
            "Advise the company on: (1) the legal exposure — ADEA, Title VII retaliation, "
            "and state law claims — and which risks are highest, (2) whether the existing "
            "documentation is adequate or creates additional risk, (3) what process to follow "
            "before termination to reduce exposure, (4) what a separation agreement should "
            "include, and (5) the OWBPA requirements that apply given the employee's age."
        ),
    },
    {
        "id": "legal_12_m_and_a_due_diligence",
        "category": "legal_strategy",
        "prompt": (
            "You are conducting legal due diligence for a $45M acquisition of a 6-year-old "
            "B2B SaaS company. During review you discover: (a) the company never obtained "
            "written IP assignment agreements from its first three engineers (who are now "
            "at competitors), (b) one of the company's core patents was filed 18 months after "
            "the feature was publicly demonstrated at a conference, (c) the company has been "
            "auto-renewing contracts with customers but the original MSAs contained one-sided "
            "termination-for-convenience clauses that favor customers, and (d) the company "
            "processes EU personal data but has no Data Processing Agreements with its "
            "sub-processors. "
            "For each issue: assess the legal risk severity (high/medium/low), explain the "
            "specific risk, identify what additional information you need, and recommend "
            "whether each should be a deal-breaker, price adjustment, or post-closing "
            "remediation with specific steps."
        ),
    },
]


# =============================================================================
# Condition Runners (same mechanics as planning comparison)
# =============================================================================

def run_greedy_baseline(encoder, task, max_new_tokens=2048):
    """Standard autoregression — no intervention."""
    start = time.time()
    response = encoder.generate_baseline(
        query=task["prompt"],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    elapsed = time.time() - start
    return {
        "condition": "greedy_baseline",
        "task_id": task["id"],
        "category": task["category"],
        "query": task["prompt"],
        "seed": None,
        "response": response,
        "response_length": len(response),
        "word_count": len(response.split()),
        "elapsed_seconds": round(elapsed, 2),
    }


def run_random_perturbation(encoder, task, seed, calibration, max_new_tokens=2048):
    """2-token embedding noise + greedy decode."""
    n_tokens = 2
    embed_dim = calibration["embed_dim"]
    rms = calibration["embedding_rms"]

    rng = torch.Generator().manual_seed(seed)
    sp = torch.randn(1, n_tokens, embed_dim, generator=rng) * rms

    start = time.time()
    response, gen_meta = decode_with_raw_soft_prompt(
        encoder, sp, task["prompt"],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    elapsed = time.time() - start
    return {
        "condition": "random_perturbation",
        "task_id": task["id"],
        "category": task["category"],
        "query": task["prompt"],
        "seed": seed,
        "response": response,
        "response_length": len(response),
        "word_count": len(response.split()),
        "elapsed_seconds": round(elapsed, 2),
        "generated_tokens": gen_meta.get("generated_tokens"),
        "terminated_by_eos": gen_meta.get("terminated_by_eos"),
    }


def run_evolution(encoder, task, seed, checkpoint_path, max_new_tokens=2048):
    """Trained latent scorer guiding evolutionary search."""
    from latent_reasoning.config import Config, ScorerConfig
    from latent_reasoning.engine import Engine

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = Config()
    config.encoder.model = "Qwen/Qwen3-4B"
    config.encoder.quantization = "4bit"
    config.encoder.latent_dim = encoder.latent_dim

    config.judges.scorers = [ScorerConfig(
        type="trained_latent",
        checkpoint=str(checkpoint_path),
        latent_dim=encoder.latent_dim,
    )]
    config.judges.modifiers = []

    config.evolution.chains = 6
    config.evolution.generations = 10
    config.evolution.temperature = 0.5
    config.evolution.temperature_decay = 0.95

    config.synthesis.max_tokens = max_new_tokens
    config.synthesis.temperature = 0.7
    config.synthesis.decode_strategy = "best"
    config.output.verbosity = "minimal"

    engine = Engine(config=config, encoder=encoder, verbosity="minimal")

    start = time.time()
    result = engine.run(task["prompt"])
    elapsed = time.time() - start

    return {
        "condition": "evolution",
        "task_id": task["id"],
        "category": task["category"],
        "query": task["prompt"],
        "seed": seed,
        "response": result.plan,
        "response_length": len(result.plan),
        "word_count": len(result.plan.split()),
        "elapsed_seconds": round(elapsed, 2),
        "evolution_generations": result.generations,
        "evolution_evaluations": result.evaluations,
        "evolution_score": result.confidence,
        "stop_reason": result.stop_reason,
    }


# =============================================================================
# Human-readable report writer
# =============================================================================

def write_readable_report(all_results, report_path):
    """Write a plain-text file where every task shows query + all responses.

    Designed so anyone can open this file and immediately see what was asked
    and what each condition produced — no JSON parsing required.
    """
    tasks_meta = {t["id"]: t for t in LEGAL_TASKS}
    outputs_by_task = {}
    for rec in all_results["outputs"]:
        tid = rec["task_id"]
        if tid not in outputs_by_task:
            outputs_by_task[tid] = []
        outputs_by_task[tid].append(rec)

    sep = "=" * 80
    thin = "-" * 80

    lines = []
    lines.append(sep)
    lines.append("LEGAL REASONING COMPARISON — FULL OUTPUT REPORT")
    lines.append(f"Model: {all_results['metadata']['model']}")
    lines.append(f"Quantization: {all_results['metadata']['quantization']}")
    lines.append(f"Generated: {all_results['metadata']['timestamp']}")
    lines.append(f"Tasks: {all_results['metadata']['n_tasks']}  |  "
                 f"Seeds per condition: {all_results['metadata']['n_seeds']}")
    lines.append(sep)
    lines.append(
        "HOW TO READ THIS FILE:\n"
        "  Each task section shows the full query followed by every response.\n"
        "  Conditions: greedy_baseline (standard LLM) vs random_perturbation vs evolution.\n"
        "  Multiple seeds are shown for perturbation/evolution to illustrate variance.\n"
        "  Judge this yourself — ignore any numeric scores, read the actual text."
    )
    lines.append(sep)

    for task_id, records in outputs_by_task.items():
        task_meta = tasks_meta.get(task_id, {})
        lines.append(f"\n{sep}")
        lines.append(f"TASK: {task_id}")
        lines.append(f"CATEGORY: {task_meta.get('category', 'unknown')}")
        lines.append(sep)

        # Print query once (grab from first record that has it)
        query = next((r["query"] for r in records if "query" in r), task_meta.get("prompt", ""))
        lines.append("QUERY:")
        lines.append(thin)
        lines.append(query)
        lines.append(thin)

        # Group by condition
        conditions_order = ["greedy_baseline", "random_perturbation", "evolution"]
        by_condition = {}
        for rec in records:
            cond = rec.get("condition", "unknown")
            by_condition.setdefault(cond, []).append(rec)

        for cond in conditions_order:
            recs = by_condition.get(cond, [])
            if not recs:
                continue
            lines.append(f"\n{'—'*80}")
            lines.append(f"CONDITION: {cond.upper()}")
            lines.append(f"{'—'*80}")

            for rec in recs:
                seed_label = f"  seed={rec['seed']}" if rec.get("seed") is not None else ""
                lines.append(
                    f"\n[{cond}{seed_label}]  "
                    f"{rec.get('word_count', '?')} words  |  "
                    f"{rec.get('elapsed_seconds', '?')}s"
                )
                lines.append(thin)
                lines.append(rec.get("response", "(no response)"))
                lines.append(thin)

    lines.append(f"\n{sep}")
    lines.append("END OF REPORT")
    lines.append(sep)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Human-readable report: {report_path}", flush=True)


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Legal reasoning 3-way comparison")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-evolution", action="store_true")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated task IDs to run (default: all)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    output_path = args.output or str(out_dir / "legal_comparison_results.json")
    report_path = output_path.replace(".json", "_readable.txt")
    seeds = list(range(42, 42 + args.n_seeds))

    tasks = LEGAL_TASKS
    if args.tasks:
        selected = set(args.tasks.split(","))
        tasks = [t for t in LEGAL_TASKS if t["id"] in selected]
        if not tasks:
            print(f"ERROR: No tasks matched {args.tasks}")
            sys.exit(1)

    print(f"{'='*70}")
    print(f"LEGAL REASONING 3-WAY COMPARISON")
    print(f"Model: {args.model} ({args.quantization})")
    print(f"Tasks: {len(tasks)}  |  Seeds: {seeds}  |  Max tokens: {args.max_new_tokens}")
    print(f"Output JSON:   {output_path}")
    print(f"Output report: {report_path}")
    print(f"{'='*70}", flush=True)

    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(
        model_name=args.model,
        device_preference="auto",
        quantization=args.quantization,
    )
    calibration = auto_calibrate(encoder)
    print(f"Calibration: embed_dim={calibration['embed_dim']}, "
          f"rms={calibration['embedding_rms']:.4f}", flush=True)

    checkpoint_path = (
        Path(__file__).parent.parent
        / "checkpoints" / "latent_scorer" / "final_model.pt"
    )
    if not checkpoint_path.exists():
        print(f"WARNING: No trained scorer at {checkpoint_path} — skipping evolution.",
              flush=True)
        args.skip_evolution = True

    all_results = {
        "metadata": {
            "experiment": "legal_reasoning_comparison",
            "model": args.model,
            "quantization": args.quantization,
            "n_seeds": args.n_seeds,
            "seeds": seeds,
            "max_new_tokens": args.max_new_tokens,
            "n_tasks": len(tasks),
            "calibration": calibration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "conditions": (
                ["greedy_baseline", "random_perturbation"]
                + ([] if args.skip_evolution else ["evolution"])
            ),
            "note": (
                "Each output record includes the full query text. "
                "See *_readable.txt for human-reviewable query+response pairs."
            ),
        },
        "tasks": [
            {"id": t["id"], "category": t["category"], "query": t["prompt"]}
            for t in tasks
        ],
        "outputs": [],
    }

    for task_idx, task in enumerate(tasks):
        print(f"\n{'#'*70}")
        print(f"# TASK {task_idx+1}/{len(tasks)}: {task['id']}")
        print(f"# Category: {task['category']}")
        print(f"{'#'*70}", flush=True)

        # Condition 1: greedy baseline
        print(f"\n  [1/3] Greedy baseline...", end="", flush=True)
        result = run_greedy_baseline(encoder, task, args.max_new_tokens)
        all_results["outputs"].append(result)
        print(f" {result['word_count']} words, {result['elapsed_seconds']}s", flush=True)

        # Condition 2: random perturbation (n seeds)
        for seed in seeds:
            print(f"  [2/3] Perturbation seed={seed}...", end="", flush=True)
            try:
                result = run_random_perturbation(
                    encoder, task, seed, calibration, args.max_new_tokens
                )
                all_results["outputs"].append(result)
                print(f" {result['word_count']} words, {result['elapsed_seconds']}s",
                      flush=True)
            except Exception as e:
                print(f" ERROR: {e}", flush=True)
                all_results["outputs"].append({
                    "condition": "random_perturbation",
                    "task_id": task["id"],
                    "category": task["category"],
                    "query": task["prompt"],
                    "seed": seed,
                    "error": str(e),
                })

        # Condition 3: evolution (n seeds)
        if not args.skip_evolution:
            for seed in seeds:
                print(f"  [3/3] Evolution seed={seed}...", end="", flush=True)
                try:
                    result = run_evolution(
                        encoder, task, seed, checkpoint_path, args.max_new_tokens
                    )
                    all_results["outputs"].append(result)
                    print(
                        f" {result['word_count']} words, "
                        f"score={result['evolution_score']:.3f}, "
                        f"{result['elapsed_seconds']}s",
                        flush=True,
                    )
                except Exception as e:
                    print(f" ERROR: {e}", flush=True)
                    all_results["outputs"].append({
                        "condition": "evolution",
                        "task_id": task["id"],
                        "category": task["category"],
                        "query": task["prompt"],
                        "seed": seed,
                        "error": str(e),
                    })

        # Checkpoint after each task so partial runs are recoverable
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Write readable report incrementally too
        write_readable_report(all_results, report_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final save
    print(f"\n{'='*70}")
    print(f"COMPLETE — {len(all_results['outputs'])} outputs saved.")
    print(f"  JSON:   {output_path}")
    print(f"  Report: {report_path}")
    print(f"{'='*70}", flush=True)

    # Summary table
    conditions = sorted(set(
        o["condition"] for o in all_results["outputs"] if "error" not in o
    ))
    for cond in conditions:
        recs = [o for o in all_results["outputs"]
                if o["condition"] == cond and "error" not in o]
        wc = [o["word_count"] for o in recs]
        print(f"\n{cond}:  n={len(recs)}  "
              f"words mean={sum(wc)//len(wc)}  "
              f"min={min(wc)}  max={max(wc)}")

    print(f"\nReady for Codex blind review.", flush=True)


if __name__ == "__main__":
    main()
