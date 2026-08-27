"""
Legal Reasoning v2 — Redesigned Task Set

Changes from v1 based on Codex quality review:
  - Citation pressure REMOVED: hallucinated citations are disqualifying for legal AI.
    Every task now instructs the model not to fabricate case names.
  - Legal frameworks PROVIDED in the prompt: model applies a named, defined standard
    rather than needing to recall it from training. Quality is then verifiable.
  - Quality signals are COUNTABLE: issue spotting (count issues identified),
    factor application (how many factors of the test applied?), risk tiers.
  - Tasks favor strategic analysis, risk stratification, and framework application
    over encyclopedic recall — that's what a legal AI assistant actually does.

The hallucination problem in v1: evolution's scorer rewards confident-sounding
legal text; confident legal text invents citations. These tasks remove that trap.

Same infrastructure: 3 conditions × 5 seeds × N tasks.
"""

import gc
import json
import os
import re
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

# Prefix applied to every task to suppress citation hallucination
NO_CITE_PREFIX = (
    "[INSTRUCTION: Analyze the legal question below using legal principles and reasoning. "
    "Do NOT fabricate or guess case names, statute numbers, or legal citations. "
    "If you refer to a legal standard or test, name it generically (e.g. 'the three-part "
    "unfairness test') rather than citing a specific case. Focus on the quality of your "
    "legal reasoning, issue identification, and practical advice.]\n\n"
)

# =============================================================================
# v2 Legal Task Definitions — 12 tasks, framework-anchored
# =============================================================================

LEGAL_TASKS_V2 = [

    # --- Framework Application: provided standard, model applies it ---

    {
        "id": "v2_01_ftc_unfairness",
        "category": "framework_application",
        "prompt": NO_CITE_PREFIX + (
            "The FTC three-part unfairness test requires that a practice must: "
            "(1) cause or be likely to cause substantial injury to consumers, "
            "(2) not be outweighed by countervailing benefits to consumers or competition, and "
            "(3) not be reasonably avoidable by consumers themselves. "
            "\n\nScenario: A subscription fitness app charges users $99/year. After the "
            "free trial, users are auto-enrolled and charged without a pre-billing reminder. "
            "The enrollment was disclosed in the original signup flow on page 3 of 5 of the "
            "terms of service, in 8pt font, in a section titled 'Renewal Policy.' The company "
            "argues users could have cancelled at any time via a 6-step cancellation process "
            "accessible through Settings > Account > Subscription > Manage > Cancel > Confirm. "
            "\n\nApply the three-part FTC unfairness test to this scenario in full. "
            "For each factor: state whether it is met, explain why with specific reference "
            "to the facts, identify the strongest counterargument, and assess its weight. "
            "Conclude with an overall unfairness finding and explain what changes would "
            "eliminate the unfairness."
        ),
    },

    {
        "id": "v2_02_gdpr_controller_processor",
        "category": "framework_application",
        "prompt": NO_CITE_PREFIX + (
            "Under GDPR, a 'controller' determines the purposes and means of processing "
            "personal data. A 'processor' processes data only on behalf of the controller "
            "and on documented instructions. A controller-processor relationship requires "
            "a Data Processing Agreement (DPA) per Article 28. Joint controllers must have "
            "a joint controller arrangement per Article 26. Processors cannot engage "
            "sub-processors without controller authorization (Art. 28(2))."
            "\n\nScenario: CompanyA (US, no EU establishment) runs a SaaS HR platform "
            "used by CompanyB (UK) and CompanyC (Germany) to manage their employee records. "
            "CompanyA stores data on AWS EU-West servers. CompanyA uses a US-based analytics "
            "firm SubVendor to run engagement scoring on all HR data across all its customers "
            "to improve its own product. CompanyB and CompanyC did not authorize SubVendor "
            "and were not informed of it."
            "\n\nFor each entity (CompanyA, CompanyB, CompanyC, SubVendor, AWS): "
            "(1) classify them as controller, processor, or joint controller with reasoning, "
            "(2) identify every GDPR obligation that flows from that classification, "
            "(3) identify every GDPR violation present in this scenario, "
            "(4) rank the violations by severity and explain the maximum exposure for each."
        ),
    },

    {
        "id": "v2_03_employment_disparate_impact",
        "category": "framework_application",
        "prompt": NO_CITE_PREFIX + (
            "The disparate impact theory of employment discrimination requires a plaintiff to: "
            "(1) identify a specific employment practice, "
            "(2) show that it causes a statistically significant disparate impact on a "
            "protected class, and "
            "(3) the employer must then demonstrate the practice is job-related and consistent "
            "with business necessity. Even if the employer meets that burden, the plaintiff "
            "can still prevail by showing a less discriminatory alternative exists."
            "\n\nScenario: A tech company uses an AI-based resume screening tool that filters "
            "candidates based on 'cultural fit' signals derived from social media activity, "
            "writing style analysis, and university prestige rankings. An internal audit shows: "
            "the tool advances 34% of white applicants to phone screens but only 18% of Black "
            "applicants and 21% of Hispanic applicants. The company says the tool is validated "
            "by positive correlation with 6-month performance reviews."
            "\n\nApply the disparate impact framework: walk through each step, assess "
            "the statistical evidence, evaluate whether the business necessity defense holds "
            "given the validation method, identify what a plaintiff's less-discriminatory "
            "alternative argument would be, and give a risk assessment for the company."
        ),
    },

    # --- Issue Spotting: quality = how many distinct issues identified ---

    {
        "id": "v2_04_saas_contract_issues",
        "category": "issue_spotting",
        "prompt": NO_CITE_PREFIX + (
            "You are reviewing a SaaS Master Service Agreement on behalf of an enterprise "
            "customer. Identify and analyze EVERY legal issue in the following clauses. "
            "For each issue: name it, explain the specific risk to the customer, "
            "and state what the clause should say instead."
            "\n\nClause 1 — Liability Cap: 'Vendor's total liability for any claims arising "
            "under this Agreement shall not exceed the fees paid by Customer in the three "
            "months preceding the claim.'"
            "\n\nClause 2 — IP Ownership: 'All work product, customizations, and deliverables "
            "created by Vendor under this Agreement shall be the exclusive property of Vendor.'"
            "\n\nClause 3 — Data: 'Customer grants Vendor a perpetual, irrevocable license "
            "to use Customer data to improve Vendor's products and services.'"
            "\n\nClause 4 — Termination: 'Either party may terminate this Agreement for "
            "convenience upon 90 days written notice. Upon termination, Customer shall pay "
            "all fees through the end of the current contract term.'"
            "\n\nClause 5 — Governing Law: 'This Agreement shall be governed by the laws of "
            "the State of Delaware. Any disputes shall be resolved by binding arbitration "
            "in Wilmington, Delaware under AAA Commercial Rules, with no class actions.'"
            "\n\nBe exhaustive. A good lawyer finds every issue, not just the obvious ones."
        ),
    },

    {
        "id": "v2_05_startup_acquisition_issues",
        "category": "issue_spotting",
        "prompt": NO_CITE_PREFIX + (
            "You are conducting legal due diligence for a $30M acquisition of a 4-year-old "
            "B2B SaaS startup. You have been given the following facts. Identify and explain "
            "EVERY material legal issue. For each: classify as deal-breaker / price-reduction "
            "/ post-closing remedy, explain the risk and its magnitude, and identify what "
            "additional information you need."
            "\n\nFacts discovered:"
            "\n1. The founding CTO left 18 months ago under disputed circumstances; no IP "
            "assignment or separation agreement was signed at departure."
            "\n2. The company has 3 open-source dependencies licensed under AGPL. The product "
            "is delivered as SaaS (not distributed), but one AGPL component was modified and "
            "the modifications were never released."
            "\n3. Customer contracts auto-renew annually but the company has been accepting "
            "verbal mid-term pricing changes not reflected in written amendments."
            "\n4. The company processes health-adjacent data (employee wellness surveys) for "
            "enterprise clients but has no BAAs in place and has not conducted a HIPAA analysis."
            "\n5. Two former sales employees filed EEOC charges 8 months ago alleging "
            "commission clawbacks were applied discriminatorily. No litigation has been filed."
            "\n6. The company has $2.1M in deferred revenue on its balance sheet from "
            "multi-year prepaid contracts, but the contracts contain broad refund rights "
            "if the product is 'materially modified.'"
        ),
    },

    # --- Risk Stratification: quality = completeness + accuracy of risk ranking ---

    {
        "id": "v2_06_data_breach_risk_triage",
        "category": "risk_stratification",
        "prompt": NO_CITE_PREFIX + (
            "A company just discovered a data breach. You have 2 hours before the CEO "
            "needs a legal risk briefing. Using the facts below, produce a prioritized "
            "risk register: for each regulatory/legal exposure, state (a) the specific "
            "legal obligation triggered, (b) the deadline if any, (c) the maximum financial "
            "exposure, (d) whether immediate action is required in the next 24 hours, and "
            "(e) the single most important thing to do for that risk right now."
            "\n\nFacts: The company is a US healthcare staffing firm with operations in "
            "California, New York, and Texas. It has EU clients (8% of revenue) and "
            "processes their EU employee contact data. The breach exposed: full names, "
            "SSNs, dates of birth, home addresses, and employment history of 85,000 "
            "individuals (70,000 US, 15,000 EU). The breach appears to have lasted 19 days. "
            "A ransomware group has already posted a sample of 500 records publicly. "
            "The company has cyber insurance with a $5M limit and $500K retention."
            "\n\nCover all applicable frameworks: HIPAA, state breach notification laws, "
            "GDPR, FTC Act, and any others that apply. Prioritize by deadline urgency."
        ),
    },

    {
        "id": "v2_07_ip_risk_portfolio",
        "category": "risk_stratification",
        "prompt": NO_CITE_PREFIX + (
            "A software company is preparing for a Series B raise. Investors have asked "
            "for a clean IP risk assessment. Review the following facts and produce a "
            "complete IP risk register. For each risk: identify the IP right implicated, "
            "explain why it is a risk, assess likelihood (high/medium/low) and potential "
            "impact (high/medium/low), and recommend a specific remediation step."
            "\n\nFacts:"
            "\n- Core algorithm was developed by two co-founders while employed at BigCorp. "
            "Their employment agreements had standard IP assignment clauses but they claim "
            "the work was done on personal time and equipment."
            "\n- The product's UI is strikingly similar to a competitor's (CompetitorX) "
            "product that launched 8 months before this company's. No copying occurred but "
            "both founders used CompetitorX daily in their prior jobs."
            "\n- The company has 2 pending US patent applications filed 14 months ago. "
            "The core feature was demo'd at a conference 20 months ago."
            "\n- The codebase uses 47 open-source packages. A recent audit found 3 packages "
            "with GPL licenses being used in the core product (not LGPL, not MIT — GPL)."
            "\n- One key engineer contributed code before signing an IP assignment agreement "
            "(the agreement was signed 3 months after they started)."
        ),
    },

    # --- Strategic Legal Analysis: multi-party, adversarial reasoning ---

    {
        "id": "v2_08_negotiation_leverage",
        "category": "strategic_analysis",
        "prompt": NO_CITE_PREFIX + (
            "You represent a mid-size software company (the 'Licensor') negotiating a "
            "source code escrow and license agreement with a Fortune 500 enterprise client "
            "(the 'Licensee'). The Licensee is demanding: (1) full source code escrow with "
            "release triggers including 'failure to provide adequate support' and "
            "'material decline in product quality,' (2) a perpetual license to modify and "
            "maintain the source code if released, (3) the right to hire away the Licensor's "
            "engineers if escrow is triggered, (4) audit rights over the Licensor's "
            "development practices, and (5) a most-favored-nation pricing clause."
            "\n\nFor each demand: (a) explain why the Licensee wants it and what legitimate "
            "business concern it addresses, (b) explain the specific risk to the Licensor, "
            "(c) propose a compromise position that protects the Licensor's core interests "
            "while giving the Licensee meaningful protection, and (d) identify the single "
            "sentence you would use to explain the compromise to the Licensee's general counsel."
            "\n\nThen: rank these five demands in order of how hard to push back on, "
            "and explain the overall negotiation strategy."
        ),
    },

    {
        "id": "v2_09_regulatory_response_strategy",
        "category": "strategic_analysis",
        "prompt": NO_CITE_PREFIX + (
            "A state attorney general has sent your client, a consumer lending platform, "
            "a Civil Investigative Demand (CID) requesting: all internal communications "
            "about underwriting model design, all model documentation and validation reports, "
            "all consumer complaint records for the past 3 years, and a 30(b)(6) deposition "
            "of the Chief Risk Officer."
            "\n\nThe AG's cover letter states the investigation concerns 'potential violations "
            "of the state Consumer Financial Protection Act, including but not limited to "
            "unfair, deceptive, or abusive acts or practices, and potential disparate impact "
            "in credit decisions.'"
            "\n\nDesign a complete response strategy: "
            "(1) What to produce, what to withhold and under what privilege, "
            "(2) How to prepare the CRO for deposition — what topics to address, what lines "
            "not to cross, what not to say about model decisions, "
            "(3) Whether to seek a meet-and-confer and what to propose, "
            "(4) Whether to engage proactively with the AG vs. purely defensive posture — "
            "pros and cons of each, "
            "(5) What internal remediation (if any) to initiate before responding, and "
            "whether doing so helps or creates more exposure."
        ),
    },

    # --- Scenario Analysis: complex multi-issue, realistic legal problems ---

    {
        "id": "v2_10_contractor_misclassification",
        "category": "scenario_analysis",
        "prompt": NO_CITE_PREFIX + (
            "A technology company has 340 'independent contractors' who: work exclusively "
            "for the company, use company-provided laptops and tools, work set hours "
            "(9-5 Monday-Friday), attend mandatory weekly team meetings, have managers who "
            "direct their daily work, have been working in this arrangement for an average "
            "of 2.3 years, and are prohibited by contract from working for competitors. "
            "The company pays them via 1099 with no benefits."
            "\n\nThe company operates in California, New York, and Texas and has contractors "
            "in all three states. An employment plaintiff's firm has sent a demand letter "
            "threatening class action litigation."
            "\n\nProvide a complete analysis: "
            "(1) Apply the relevant worker classification tests for each state — "
            "California ABC test, New York economic reality test, Texas common law test — "
            "and give a classification verdict for each jurisdiction with reasoning, "
            "(2) If they are misclassified employees, enumerate every category of liability "
            "(back wages, benefits, taxes, penalties) and give a rough magnitude estimate, "
            "(3) Assess the federal exposure (FLSA, IRS), "
            "(4) Recommend an immediate triage plan — what to do in the next 30 days "
            "to reduce exposure without triggering additional liability."
        ),
    },

    {
        "id": "v2_11_corporate_liability_shield",
        "category": "scenario_analysis",
        "prompt": NO_CITE_PREFIX + (
            "ParentCorp (Delaware C-corp) owns 100% of SubCorp (also Delaware). SubCorp "
            "has $500K in assets. SubCorp has a $4M judgment against it from a products "
            "liability case. The plaintiff is seeking to pierce the corporate veil and "
            "reach ParentCorp's assets ($45M)."
            "\n\nFacts: ParentCorp and SubCorp share the same CFO and office space. "
            "SubCorp never held formal board meetings and has no board minutes for "
            "the past 3 years. SubCorp's bank account was occasionally used to pay "
            "ParentCorp's vendors when ParentCorp had a cash timing issue (6 instances, "
            "all repaid within 30 days). SubCorp had no independent legal counsel — "
            "all contracts were reviewed by ParentCorp's GC. SubCorp was adequately "
            "capitalized when formed 5 years ago but ParentCorp stopped making capital "
            "contributions 2 years ago as SubCorp's liabilities grew."
            "\n\nAnalyze the veil-piercing claim: identify every fact that supports "
            "piercing and every fact that cuts against it, apply the alter ego / "
            "instrumentality tests, assess the overall probability of success for "
            "the plaintiff, and recommend what ParentCorp should do right now to "
            "strengthen its position before trial."
        ),
    },

    {
        "id": "v2_12_whistleblower_retaliation",
        "category": "scenario_analysis",
        "prompt": NO_CITE_PREFIX + (
            "An engineer at a publicly traded company reported to the SEC (via the "
            "whistleblower program) that the company was overstating ARR in its "
            "earnings reports. The report was made 6 months ago. Since then: "
            "the engineer received a 'needs improvement' performance review (first ever "
            "negative review in 4 years), was excluded from a promotion cycle, was "
            "moved to a less visible project, had their remote work accommodation "
            "revoked citing 'team cohesion,' and was denied a conference speaking slot "
            "they had been approved for."
            "\n\nThe company claims all decisions were made by different managers with "
            "no knowledge of the SEC report. The SEC submission was anonymous but "
            "the engineer disclosed it to HR 3 months ago when asking about retaliation "
            "protections."
            "\n\nAnalyze: (1) Which federal and state anti-retaliation protections apply "
            "and what each requires to prove retaliation, (2) evaluate each adverse action "
            "as evidence of retaliation — how strong is each individually and collectively, "
            "(3) assess the 'different managers, no knowledge' defense and what discovery "
            "would be needed to challenge it, (4) advise the engineer on their strongest "
            "claims and the right sequence of actions (OSHA complaint, SEC complaint, "
            "civil suit), and (5) advise the company on what it must do immediately "
            "to avoid making this worse."
        ),
    },
]


# =============================================================================
# Condition Runners — identical to v1, no changes needed
# =============================================================================

def run_greedy_baseline(encoder, task, max_new_tokens=2048):
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

    # Strip thinking tokens for fair comparison (baseline strips them too)
    response_raw = response
    if "<think>" in response and "</think>" in response:
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
    elif response.startswith("<think>"):
        response = ""

    return {
        "condition": "random_perturbation",
        "task_id": task["id"],
        "category": task["category"],
        "query": task["prompt"],
        "seed": seed,
        "response": response,
        "response_raw": response_raw,
        "response_length": len(response),
        "word_count": len(response.split()),
        "elapsed_seconds": round(elapsed, 2),
        "generated_tokens": gen_meta.get("generated_tokens"),
        "terminated_by_eos": gen_meta.get("terminated_by_eos"),
    }


def run_evolution(encoder, task, seed, checkpoint_path, max_new_tokens=2048):
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
# Human-readable report
# =============================================================================

def write_readable_report(all_results, report_path):
    tasks_meta = {t["id"]: t for t in LEGAL_TASKS_V2}
    outputs_by_task = {}
    for rec in all_results["outputs"]:
        tid = rec["task_id"]
        outputs_by_task.setdefault(tid, []).append(rec)

    sep = "=" * 80
    thin = "-" * 80

    lines = []
    lines.append(sep)
    lines.append("LEGAL REASONING v2 — FULL OUTPUT REPORT")
    lines.append(f"Model: {all_results['metadata']['model']}")
    lines.append(f"Quantization: {all_results['metadata']['quantization']}")
    lines.append(f"Generated: {all_results['metadata']['timestamp']}")
    lines.append(f"Tasks: {all_results['metadata']['n_tasks']}  |  "
                 f"Seeds per condition: {all_results['metadata']['n_seeds']}")
    lines.append(sep)
    lines.append(
        "HOW TO READ THIS FILE:\n"
        "  Each section shows the full query then every response side-by-side.\n"
        "  Conditions: greedy_baseline (standard LLM), random_perturbation, evolution.\n"
        "  Evaluate by: issues identified, frameworks applied, reasoning depth.\n"
        "  Do NOT judge by length alone. A concise correct answer beats a long wrong one.\n"
        "  Tasks are designed so quality is countable: count issues spotted, factors applied."
    )
    lines.append(sep)

    for task_id, records in outputs_by_task.items():
        task_meta = tasks_meta.get(task_id, {})
        lines.append(f"\n{sep}")
        lines.append(f"TASK: {task_id}")
        lines.append(f"CATEGORY: {task_meta.get('category', 'unknown')}")
        lines.append(sep)

        query = next((r["query"] for r in records if "query" in r),
                     task_meta.get("prompt", ""))
        lines.append("QUERY:")
        lines.append(thin)
        lines.append(query)
        lines.append(thin)

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
                seed_label = (f"  seed={rec['seed']}"
                              if rec.get("seed") is not None else "")
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

    print(f"  Report: {report_path}", flush=True)


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Legal reasoning v2 comparison")
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
    output_path = args.output or str(out_dir / "legal_v2_comparison_results.json")
    report_path = output_path.replace(".json", "_readable.txt")
    seeds = list(range(42, 42 + args.n_seeds))

    tasks = LEGAL_TASKS_V2
    if args.tasks:
        selected = set(args.tasks.split(","))
        tasks = [t for t in LEGAL_TASKS_V2 if t["id"] in selected]
        if not tasks:
            print(f"ERROR: No tasks matched {args.tasks}")
            sys.exit(1)

    print(f"{'='*70}")
    print(f"LEGAL REASONING v2 — 3-WAY COMPARISON")
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
            "experiment": "legal_reasoning_v2_comparison",
            "version": "v2",
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
            "design_notes": (
                "v2 redesign: citation pressure removed (hallucinated authority was "
                "disqualifying in v1); frameworks provided in-prompt; quality is "
                "countable (issues spotted, factors applied, risk tiers)."
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

        print(f"\n  [1/3] Greedy baseline...", end="", flush=True)
        result = run_greedy_baseline(encoder, task, args.max_new_tokens)
        all_results["outputs"].append(result)
        print(f" {result['word_count']} words, {result['elapsed_seconds']}s", flush=True)

        for seed in seeds:
            print(f"  [2/3] Perturbation seed={seed}...", end="", flush=True)
            try:
                result = run_random_perturbation(
                    encoder, task, seed, calibration, args.max_new_tokens)
                all_results["outputs"].append(result)
                print(f" {result['word_count']} words, {result['elapsed_seconds']}s",
                      flush=True)
            except Exception as e:
                print(f" ERROR: {e}", flush=True)
                all_results["outputs"].append({
                    "condition": "random_perturbation",
                    "task_id": task["id"], "category": task["category"],
                    "query": task["prompt"], "seed": seed, "error": str(e),
                })

        if not args.skip_evolution:
            for seed in seeds:
                print(f"  [3/3] Evolution seed={seed}...", end="", flush=True)
                try:
                    result = run_evolution(
                        encoder, task, seed, checkpoint_path, args.max_new_tokens)
                    all_results["outputs"].append(result)
                    print(
                        f" {result['word_count']} words, "
                        f"score={result['evolution_score']:.3f}, "
                        f"{result['elapsed_seconds']}s", flush=True)
                except Exception as e:
                    print(f" ERROR: {e}", flush=True)
                    all_results["outputs"].append({
                        "condition": "evolution",
                        "task_id": task["id"], "category": task["category"],
                        "query": task["prompt"], "seed": seed, "error": str(e),
                    })

        # Checkpoint after each task
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        write_readable_report(all_results, report_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"COMPLETE — {len(all_results['outputs'])} outputs saved.")
    print(f"  JSON:   {output_path}")
    print(f"  Report: {report_path}")
    print(f"{'='*70}", flush=True)

    conditions = sorted(set(
        o["condition"] for o in all_results["outputs"] if "error" not in o))
    for cond in conditions:
        recs = [o for o in all_results["outputs"]
                if o["condition"] == cond and "error" not in o]
        wc = [o["word_count"] for o in recs]
        print(f"\n{cond}:  n={len(recs)}  words mean={sum(wc)//len(wc)}  "
              f"min={min(wc)}  max={max(wc)}")

    print(f"\nReady for Codex blind review.", flush=True)


if __name__ == "__main__":
    main()
