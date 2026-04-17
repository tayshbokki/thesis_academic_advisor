"""
analyze_results.py — DLSU CpE AI Advising: Comprehensive Results Dashboard
============================================================================
Summarizes, tabularizes, and visualizes data from all experiment phases:
  1. Embedding Model Comparison (5 models)
  2. No-RAG Baseline (6 models × configs)
  3. Naive RAG (5 models × configs)
  4. Improved RAG Phase 1 — Retrieval configs (8 dense + 9 general)
  5. Improved RAG Phase 2 — Generation (6 models × 8 configs)  [if available]
  6. Agentic RAG (Pinecone Assistant, 15 questions)

Outputs:
  - Console summary tables
  - 8 PNG figures saved to ./figures/
  - results_analysis_report.txt

Usage:
    python analyze_results.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────

BASE = Path(".")
UPLOAD = Path("/mnt/user-data/uploads")

FILES = {
    "embedding":        UPLOAD / "results_summary.json",
    "no_rag":           UPLOAD / "no_rag_baseline_results.json",
    "naive_rag":        UPLOAD / "naive_rag_results.json",
    "improved_p1":      UPLOAD / "improved_rag_phase1_results.json",
    "improved_p1_gen":  UPLOAD / "improved_rag_phase1_results_general.json",
    "improved_p2":      UPLOAD / "improved_rag_phase2_results.json",
    "agentic":          UPLOAD / "agentic_rag_results.json",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def load_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def fmt(v, decimals=4):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"

def pct(v):
    if v is None:
        return "N/A"
    return f"{v:.1f}%"

def print_divider(char="═", width=100):
    print(char * width)

def print_header(title, width=100):
    print()
    print_divider("═", width)
    print(f"  {title}")
    print_divider("═", width)


# ── Load all data ──────────────────────────────────────────────────────────

print("Loading experiment data...")
data = {k: load_json(v) for k, v in FILES.items()}

for k, v in data.items():
    status = f"{len(v)} records" if v else "NOT FOUND"
    print(f"  {k:20s} → {status}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MPL = True
    print("\n  matplotlib: available ✓")
except ImportError:
    HAS_MPL = False
    print("\n  matplotlib: not found — tables only, no figures")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: EMBEDDING MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════

print_header("1. EMBEDDING MODEL COMPARISON (535 queries)")

emb = data["embedding"]
if emb:
    # Sort by MRR desc
    emb_sorted = sorted(emb, key=lambda x: -x["mean_mrr"])

    header = f"{'Rank':<5} {'Model':<30} {'MRR':>7} {'NDCG@10':>8} {'R@10':>7} {'Gap':>7} {'ms/doc':>8} {'Params':>8}"
    print(header)
    print("-" * len(header))
    for i, m in enumerate(emb_sorted, 1):
        sel = " ← SELECTED" if m["model_id"] == "intfloat/e5-small-v2" else ""
        print(f"{i:<5} {m['model_name']:<30} {m['mean_mrr']:>7.4f} {m['mean_ndcg_at_10']:>8.4f} "
              f"{m['mean_recall_at_k']['10']:>7.4f} {m['mean_cosine_gap']:>7.4f} "
              f"{m['embed_time_per_doc_ms']:>7.1f}ms {m['description'].split('—')[1].strip():>8}{sel}")

    # Category breakdown for selected model (e5-small-v2)
    selected = next(m for m in emb if m["model_id"] == "intfloat/e5-small-v2")
    cats = selected["category_breakdown"]
    print(f"\n  Per-Category Breakdown (e5-small-v2):")
    cat_header = f"  {'Category':<25} {'N':>4} {'MRR':>7} {'NDCG@10':>8} {'R@1':>7} {'R@5':>7}"
    print(cat_header)
    print("  " + "-" * (len(cat_header) - 2))
    for cat in sorted(cats.keys(), key=lambda c: -cats[c]["mean_mrr"]):
        c = cats[cat]
        print(f"  {cat:<25} {c['n']:>4} {c['mean_mrr']:>7.4f} {c['mean_ndcg_at_10']:>8.4f} "
              f"{c['mean_recall_at_1']:>7.4f} {c['mean_recall_at_5']:>7.4f}")

    # Program breakdown
    if "program_breakdown" in selected:
        prog = selected["program_breakdown"]
        print(f"\n  Per-Program Breakdown (e5-small-v2):")
        for p in sorted(prog.keys()):
            pb = prog[p]
            print(f"    {p:<10} N={pb['n']:>3}  MRR={pb['mean_mrr']:.4f}  R@5={pb['mean_recall_at_5']:.4f}")
else:
    print("  [SKIPPED] — results_summary.json not found")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: NO-RAG BASELINE vs NAIVE RAG — BEST PER MODEL
# ══════════════════════════════════════════════════════════════════════════

print_header("2. GENERATION COMPARISON: No-RAG vs Naive RAG (Best Config Per Model)")

def best_per_model(records, sort_key="avg_rouge_l"):
    """Return best config per model by sort_key."""
    models = {}
    for r in records:
        m = r["model"]
        if m not in models or r[sort_key] > models[m][sort_key]:
            models[m] = r
    return models

norag = data["no_rag"]
naive = data["naive_rag"]

if norag and naive:
    norag_best = best_per_model(norag)
    naive_best = best_per_model(naive)

    all_models = sorted(set(list(norag_best.keys()) + list(naive_best.keys())))

    header = (f"{'Model':<22} │ {'--- No-RAG Baseline ---':^38} │ {'--- Naive RAG ---':^38} │ {'Δ R-L':>6} │ {'Δ Halluc':>8}")
    print(header)
    sub = (f"{'':<22} │ {'R-L':>6} {'BLEU':>6} {'METEOR':>7} {'BERT':>6} {'Hal%':>6} │ "
           f"{'R-L':>6} {'BLEU':>6} {'METEOR':>7} {'BERT':>6} {'Hal%':>6} │ {'':<6} │ {'':<8}")
    print(sub)
    print("─" * len(header))

    for m in all_models:
        nr = norag_best.get(m)
        nv = naive_best.get(m)

        def row(r):
            if r is None:
                return "  —     —      —      —     —  "
            return (f"{r['avg_rouge_l']:>6.3f} {r['avg_bleu']:>6.3f} "
                    f"{r['avg_meteor']:>7.3f} {r['avg_bert_score']:>6.3f} {r['hallucination_rate']:>5.1f}%")

        delta_rl = ""
        delta_h = ""
        if nr and nv:
            d = nv['avg_rouge_l'] - nr['avg_rouge_l']
            delta_rl = f"{d:>+6.3f}"
            dh = nv['hallucination_rate'] - nr['hallucination_rate']
            delta_h = f"{dh:>+7.1f}%"

        print(f"{m:<22} │ {row(nr)} │ {row(nv)} │ {delta_rl:>6} │ {delta_h:>8}")

    # Summary stats
    print()
    nr_vals = list(norag_best.values())
    nv_vals = list(naive_best.values())
    common = [m for m in all_models if m in norag_best and m in naive_best]
    if common:
        avg_nr_rl = sum(norag_best[m]['avg_rouge_l'] for m in common) / len(common)
        avg_nv_rl = sum(naive_best[m]['avg_rouge_l'] for m in common) / len(common)
        avg_nr_h = sum(norag_best[m]['hallucination_rate'] for m in common) / len(common)
        avg_nv_h = sum(naive_best[m]['hallucination_rate'] for m in common) / len(common)
        print(f"  Average ROUGE-L:         No-RAG={avg_nr_rl:.4f}  Naive-RAG={avg_nv_rl:.4f}  Δ={avg_nv_rl-avg_nr_rl:+.4f}")
        print(f"  Average Hallucination%:  No-RAG={avg_nr_h:.1f}%    Naive-RAG={avg_nv_h:.1f}%    Δ={avg_nv_h-avg_nr_h:+.1f}%")
else:
    print("  [SKIPPED] — missing no_rag or naive_rag results")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: NAIVE RAG — FULL HYPERPARAMETER SWEEP
# ══════════════════════════════════════════════════════════════════════════

print_header("3. NAIVE RAG — HYPERPARAMETER SENSITIVITY (All 43 Configs)")

if naive:
    # Group by model
    by_model = defaultdict(list)
    for r in naive:
        by_model[r["model"]].append(r)

    for model_name in sorted(by_model.keys()):
        configs = sorted(by_model[model_name], key=lambda x: -x["avg_rouge_l"])
        print(f"\n  {model_name} ({len(configs)} configs)")
        print(f"  {'Config':<28} {'R-1':>6} {'R-L':>6} {'BLEU':>6} {'BERT':>6} {'Hal%':>6} {'Time':>7}")
        print(f"  " + "-" * 70)
        for c in configs:
            print(f"  {c['config']:<28} {c['avg_rouge1']:>6.3f} {c['avg_rouge_l']:>6.3f} "
                  f"{c['avg_bleu']:>6.3f} {c['avg_bert_score']:>6.3f} "
                  f"{c['hallucination_rate']:>5.1f}% {c['avg_total_time']:>6.3f}s")
else:
    print("  [SKIPPED]")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: IMPROVED RAG PHASE 1 — RETRIEVAL CONFIGS
# ══════════════════════════════════════════════════════════════════════════

print_header("4. IMPROVED RAG PHASE 1 — RETRIEVAL CONFIG COMPARISON")

p1 = data["improved_p1"]
p1g = data["improved_p1_gen"]

if p1:
    print("\n  A) Dense-focused configs (alpha=1.0, varying k and rerank_top):")
    header = f"  {'Config':<22} {'MRR':>7} {'NDCG@10':>8} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'SQL%':>6} {'AvgRet':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for c in sorted(p1, key=lambda x: -x["so1_mean_mrr"]):
        rk = c["so1_recall_at_k"]
        r1 = rk.get(1, rk.get("1", 0))
        r5 = rk.get(5, rk.get("5", 0))
        r10 = rk.get(10, rk.get("10", 0))
        print(f"  {c['retrieval_config']:<22} {c['so1_mean_mrr']:>7.4f} {c['so1_mean_ndcg_10']:>8.4f} "
              f"{r1:>7.4f} {r5:>7.4f} {r10:>7.4f} "
              f"{c['so2_sql_hit_rate']:>5.1f}% {c['so2_avg_ret_time']:>7.4f}s")

if p1g:
    print(f"\n  B) General configs (varying alpha — hybrid vs dense vs BM25):")
    header = f"  {'Config':<22} {'MRR':>7} {'NDCG@10':>8} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'SQL%':>6} {'AvgRet':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for c in sorted(p1g, key=lambda x: -x["so1_mean_mrr"]):
        rk = c["so1_recall_at_k"]
        r1 = rk.get(1, rk.get("1", 0))
        r5 = rk.get(5, rk.get("5", 0))
        r10 = rk.get(10, rk.get("10", 0))
        print(f"  {c['retrieval_config']:<22} {c['so1_mean_mrr']:>7.4f} {c['so1_mean_ndcg_10']:>8.4f} "
              f"{r1:>7.4f} {r5:>7.4f} {r10:>7.4f} "
              f"{c['so2_sql_hit_rate']:>5.1f}% {c['so2_avg_ret_time']:>7.4f}s")

    # Alpha ablation summary
    print(f"\n  C) Alpha Ablation (effect of BM25 vs Dense weighting):")
    alpha_configs = [c for c in p1g if c["retrieval_config"].startswith("k=6 a=")]
    for c in sorted(alpha_configs, key=lambda x: x["alpha"]):
        rk = c["so1_recall_at_k"]
        r10 = rk.get(10, rk.get("10", 0))
        bar = "█" * int(c["so1_mean_mrr"] * 40)
        label = "BM25-only" if c["alpha"] == 0.0 else "Dense-only" if c["alpha"] == 1.0 else f"Hybrid"
        print(f"    α={c['alpha']:.1f} ({label:<11}) MRR={c['so1_mean_mrr']:.4f} R@10={r10:.4f} {bar}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: IMPROVED RAG PHASE 2 — GENERATION (if available)
# ══════════════════════════════════════════════════════════════════════════

print_header("5. IMPROVED RAG PHASE 2 — GENERATION EVALUATION")

p2 = data["improved_p2"]
if p2:
    print(f"\n  {len(p2)} model×config combinations evaluated")
    p2_best = best_per_model(p2)

    header = (f"  {'Model':<24} {'Config':<22} {'R-L':>6} {'BLEU':>6} {'BERT':>6} "
              f"{'Hal%':>6} {'CitP':>6} {'CitR':>6} {'Time':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for m in sorted(p2_best.keys()):
        r = p2_best[m]
        cit_p = r.get("avg_citation_precision", 0)
        cit_r = r.get("avg_citation_recall", 0)
        print(f"  {r['model']:<24} {r['config']:<22} {r['avg_rouge_l']:>6.3f} "
              f"{r['avg_bleu']:>6.3f} {r.get('avg_bert_score',0):>6.3f} "
              f"{r['hallucination_rate']:>5.1f}% {cit_p:>6.3f} {cit_r:>6.3f} "
              f"{r['avg_total_time']:>6.2f}s")

    # Full sweep
    print(f"\n  Full config sweep (sorted by ROUGE-L):")
    sub_header = f"  {'Model + Config':<48} {'R-L':>6} {'BLEU':>6} {'Hal%':>6} {'CitP':>6} {'CitR':>6}"
    print(sub_header)
    print("  " + "-" * (len(sub_header) - 2))
    for r in sorted(p2, key=lambda x: -x["avg_rouge_l"]):
        name = f"{r['model']} [{r['config']}]"
        cit_p = r.get("avg_citation_precision", 0)
        cit_r = r.get("avg_citation_recall", 0)
        print(f"  {name:<48} {r['avg_rouge_l']:>6.3f} {r['avg_bleu']:>6.3f} "
              f"{r['hallucination_rate']:>5.1f}% {cit_p:>6.3f} {cit_r:>6.3f}")
else:
    print("  [SKIPPED] — improved_rag_phase2_results.json not found yet")
    print("  (Phase 2 is currently running. Re-run this script when it finishes.)")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: AGENTIC RAG (Pinecone Assistant)
# ══════════════════════════════════════════════════════════════════════════

print_header("6. AGENTIC RAG — PINECONE ASSISTANT (15 questions)")

ag = data["agentic"]
if ag:
    a = ag[0]
    print(f"  Model:            {a['model']}")
    print(f"  Config:           {a['config']}")
    print(f"  Avg ROUGE-1:      {a['avg_rouge1']:.4f}")
    print(f"  Avg ROUGE-L:      {a['avg_rouge_l']:.4f}")
    print(f"  Avg BLEU:         {a['avg_bleu']:.4f}")
    print(f"  Avg METEOR:       {a['avg_meteor']:.4f}")
    print(f"  Hallucination:    {a['hallucination_count']}/{len(a['detail'])} ({a['hallucination_rate']:.1f}%)")
    print(f"  Avg Total Time:   {a['avg_total_time']:.2f}s")
    print(f"  % Under 5s:       {a['pct_under_5s']:.1f}%")

    # Per-question breakdown
    print(f"\n  Per-Question Detail:")
    print(f"  {'#':<3} {'Question':<55} {'R-L':>6} {'Hal':>5} {'Time':>7}")
    print(f"  " + "-" * 80)
    for i, q in enumerate(a["detail"], 1):
        hal = "YES" if q["hallucination"] else "no"
        qtext = q["question"][:52] + "..." if len(q["question"]) > 55 else q["question"]
        print(f"  {i:<3} {qtext:<55} {q['rouge_l']:>6.3f} {hal:>5} {q['total_time']:>6.2f}s")
else:
    print("  [SKIPPED]")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7: CROSS-PIPELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════

print_header("7. CROSS-PIPELINE COMPARISON (Best Results Per Pipeline)")

pipelines = []

# No-RAG best overall
if norag:
    nr_best = max(norag, key=lambda x: x["avg_rouge_l"])
    pipelines.append({
        "pipeline": "No-RAG Baseline",
        "model": f"{nr_best['model']} [{nr_best['config']}]",
        "rouge_l": nr_best["avg_rouge_l"],
        "bleu": nr_best["avg_bleu"],
        "meteor": nr_best["avg_meteor"],
        "bert": nr_best["avg_bert_score"],
        "halluc": nr_best["hallucination_rate"],
        "time": nr_best["avg_total_time"],
        "pct5s": nr_best["pct_under_5s"],
    })

# Naive RAG best overall
if naive:
    nv_best = max(naive, key=lambda x: x["avg_rouge_l"])
    pipelines.append({
        "pipeline": "Naive RAG",
        "model": f"{nv_best['model']} [{nv_best['config']}]",
        "rouge_l": nv_best["avg_rouge_l"],
        "bleu": nv_best["avg_bleu"],
        "meteor": nv_best["avg_meteor"],
        "bert": nv_best["avg_bert_score"],
        "halluc": nv_best["hallucination_rate"],
        "time": nv_best["avg_total_time"],
        "pct5s": nv_best["pct_under_5s"],
    })

# Improved RAG Phase 2 best
if p2:
    p2_best_overall = max(p2, key=lambda x: x["avg_rouge_l"])
    pipelines.append({
        "pipeline": "Improved RAG",
        "model": f"{p2_best_overall['model']} [{p2_best_overall['config']}]",
        "rouge_l": p2_best_overall["avg_rouge_l"],
        "bleu": p2_best_overall["avg_bleu"],
        "meteor": p2_best_overall.get("avg_meteor", 0),
        "bert": p2_best_overall.get("avg_bert_score", 0),
        "halluc": p2_best_overall["hallucination_rate"],
        "time": p2_best_overall["avg_total_time"],
        "pct5s": p2_best_overall["pct_under_5s"],
    })

# Agentic RAG
if ag:
    a = ag[0]
    pipelines.append({
        "pipeline": "Agentic RAG",
        "model": a["model"],
        "rouge_l": a["avg_rouge_l"],
        "bleu": a["avg_bleu"],
        "meteor": a["avg_meteor"],
        "bert": None,
        "halluc": a["hallucination_rate"],
        "time": a["avg_total_time"],
        "pct5s": a["pct_under_5s"],
    })

if pipelines:
    header = (f"  {'Pipeline':<20} {'Model':<36} {'R-L':>6} {'BLEU':>6} {'BERT':>6} "
              f"{'Hal%':>6} {'AvgTime':>8} {'<5s%':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for p in pipelines:
        bert = f"{p['bert']:>6.3f}" if p['bert'] else "   N/A"
        print(f"  {p['pipeline']:<20} {p['model']:<36} {p['rouge_l']:>6.3f} "
              f"{p['bleu']:>6.3f} {bert} "
              f"{p['halluc']:>5.1f}% {p['time']:>7.2f}s {p['pct5s']:>5.1f}%")

    if len(pipelines) >= 2:
        base = pipelines[0]
        print(f"\n  Improvement Over No-RAG Baseline:")
        for p in pipelines[1:]:
            rl_imp = ((p["rouge_l"] - base["rouge_l"]) / base["rouge_l"] * 100) if base["rouge_l"] else 0
            h_imp = base["halluc"] - p["halluc"]
            print(f"    {p['pipeline']:<20} ROUGE-L: {rl_imp:>+6.1f}%   Hallucination: {h_imp:>+6.1f}pp")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8: SO TARGET VERIFICATION
# ══════════════════════════════════════════════════════════════════════════

print_header("8. SPECIFIC OBJECTIVE (SO) TARGET VERIFICATION")

# SO1: Retrieval — MRR≥0.80, NDCG@10≥0.80, R@10≥0.75
print("\n  SO1 — Retrieval Accuracy Targets:")
if p1:
    best_p1 = max(p1, key=lambda x: 0.4*x["so1_mean_mrr"]+0.3*x["so1_mean_ndcg_10"])
    rk = best_p1["so1_recall_at_k"]
    r10 = rk.get(10, rk.get("10", 0))
    targets = [
        ("MRR ≥ 0.80",     best_p1["so1_mean_mrr"],     0.80),
        ("NDCG@10 ≥ 0.80", best_p1["so1_mean_ndcg_10"], 0.80),
        ("Recall@10 ≥ 0.75", r10,                        0.75),
    ]
    for label, val, thresh in targets:
        status = "✓ PASS" if val >= thresh else "✗ FAIL"
        print(f"    {label:<20} {val:.4f}  {status}")

# SO2: Latency — <500ms retrieval, <2s total
print("\n  SO2 — Latency Targets:")
if p1:
    print(f"    Avg Retrieval  <500ms:  {best_p1['so2_avg_ret_time']*1000:.1f}ms  "
          f"{'✓ PASS' if best_p1['so2_avg_ret_time'] < 0.5 else '✗ FAIL'}")
    print(f"    % Under 500ms:          {best_p1['so2_pct_under_500ms']:.1f}%")

# SO7: End-to-end <2s
if naive:
    nv_best = max(naive, key=lambda x: x["avg_rouge_l"])
    print(f"    Naive RAG AvgTime <2s:  {nv_best['avg_total_time']:.3f}s  "
          f"{'✓ PASS' if nv_best['avg_total_time'] < 2 else '✗ FAIL'}")

if ag:
    print(f"    Agentic RAG AvgTime:    {ag[0]['avg_total_time']:.2f}s  "
          f"{'✓ PASS' if ag[0]['avg_total_time'] < 5 else '✗ FAIL'}")


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION (matplotlib)
# ══════════════════════════════════════════════════════════════════════════

if HAS_MPL:
    os.makedirs("figures", exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    COLORS = ["#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#8B5CF6", "#EC4899"]

    # ── Figure 1: Embedding Model Comparison ──────────────────────────────

    if emb:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Figure 1: Embedding Model Comparison", fontweight="bold", fontsize=14)

        models = [m["model_name"] for m in emb_sorted]
        mrrs = [m["mean_mrr"] for m in emb_sorted]
        ndcgs = [m["mean_ndcg_at_10"] for m in emb_sorted]
        r10s = [m["mean_recall_at_k"]["10"] for m in emb_sorted]
        latencies = [m["embed_time_per_doc_ms"] for m in emb_sorted]

        colors = [COLORS[0] if m["model_id"] == "intfloat/e5-small-v2" else "#94A3B8" for m in emb_sorted]

        for ax, vals, title, ylbl in [
            (axes[0], mrrs, "MRR", "Score"),
            (axes[1], ndcgs, "NDCG@10", "Score"),
            (axes[2], r10s, "Recall@10", "Score"),
        ]:
            bars = ax.barh(models, vals, color=colors, edgecolor="white", height=0.6)
            ax.set_title(title)
            ax.set_xlabel(ylbl)
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            for bar, v in zip(bars, vals):
                ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, f"{v:.3f}",
                        va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig("figures/fig1_embedding_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("\n  Saved: figures/fig1_embedding_comparison.png")


    # ── Figure 2: Embedding Latency vs Accuracy ──────────────────────────

    if emb:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle("Figure 2: Embedding — Latency vs MRR Trade-off", fontweight="bold")

        for i, m in enumerate(emb_sorted):
            color = COLORS[0] if m["model_id"] == "intfloat/e5-small-v2" else "#94A3B8"
            size = 150 if m["model_id"] == "intfloat/e5-small-v2" else 80
            ax.scatter(m["embed_time_per_doc_ms"], m["mean_mrr"], s=size, c=color,
                      edgecolors="black", linewidths=0.5, zorder=5)
            ax.annotate(m["model_name"], (m["embed_time_per_doc_ms"], m["mean_mrr"]),
                       textcoords="offset points", xytext=(8, 5), fontsize=9)

        ax.set_xlabel("Latency (ms/doc)")
        ax.set_ylabel("MRR")
        ax.set_ylim(0.3, 0.85)
        plt.tight_layout()
        plt.savefig("figures/fig2_embedding_latency_vs_mrr.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: figures/fig2_embedding_latency_vs_mrr.png")


    # ── Figure 3: No-RAG vs Naive RAG per model ─────────────────────────

    if norag and naive:
        norag_best = best_per_model(norag)
        naive_best = best_per_model(naive)
        common = sorted(set(norag_best.keys()) & set(naive_best.keys()))

        if common:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("Figure 3: No-RAG vs Naive RAG — Best Config Per Model", fontweight="bold", fontsize=14)

            x = np.arange(len(common))
            w = 0.35

            # ROUGE-L
            ax = axes[0]
            nr_vals = [norag_best[m]["avg_rouge_l"] for m in common]
            nv_vals = [naive_best[m]["avg_rouge_l"] for m in common]
            ax.bar(x - w/2, nr_vals, w, label="No-RAG", color="#94A3B8", edgecolor="white")
            ax.bar(x + w/2, nv_vals, w, label="Naive RAG", color=COLORS[0], edgecolor="white")
            ax.set_ylabel("ROUGE-L")
            ax.set_title("ROUGE-L Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(common, rotation=30, ha="right", fontsize=9)
            ax.legend()

            # Hallucination Rate
            ax = axes[1]
            nr_h = [norag_best[m]["hallucination_rate"] for m in common]
            nv_h = [naive_best[m]["hallucination_rate"] for m in common]
            ax.bar(x - w/2, nr_h, w, label="No-RAG", color="#94A3B8", edgecolor="white")
            ax.bar(x + w/2, nv_h, w, label="Naive RAG", color=COLORS[1], edgecolor="white")
            ax.set_ylabel("Hallucination Rate (%)")
            ax.set_title("Hallucination Rate Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(common, rotation=30, ha="right", fontsize=9)
            ax.legend()

            plt.tight_layout()
            plt.savefig("figures/fig3_norag_vs_naive.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved: figures/fig3_norag_vs_naive.png")


    # ── Figure 4: Alpha Ablation ─────────────────────────────────────────

    if p1g:
        alpha_configs = sorted(
            [c for c in p1g if c["retrieval_config"].startswith("k=6 a=")],
            key=lambda x: x["alpha"]
        )
        if alpha_configs:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.suptitle("Figure 4: Hybrid Retrieval — Alpha Ablation (k=6)", fontweight="bold")

            alphas = [c["alpha"] for c in alpha_configs]
            mrrs = [c["so1_mean_mrr"] for c in alpha_configs]
            ndcgs = [c["so1_mean_ndcg_10"] for c in alpha_configs]

            ax.plot(alphas, mrrs, "o-", color=COLORS[0], label="MRR", linewidth=2, markersize=8)
            ax.plot(alphas, ndcgs, "s--", color=COLORS[1], label="NDCG@10", linewidth=2, markersize=8)

            ax.set_xlabel("Alpha (0=BM25-only → 1=Dense-only)")
            ax.set_ylabel("Score")
            ax.set_ylim(0.4, 0.9)
            ax.legend()
            ax.set_xticks(alphas)

            plt.tight_layout()
            plt.savefig("figures/fig4_alpha_ablation.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved: figures/fig4_alpha_ablation.png")


    # ── Figure 5: Retrieval Config Comparison (Phase 1) ──────────────────

    if p1:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Figure 5: Improved RAG Phase 1 — Retrieval Config Comparison", fontweight="bold")

        configs = sorted(p1, key=lambda x: -x["so1_mean_mrr"])
        labels = [c["retrieval_config"] for c in configs]
        mrrs = [c["so1_mean_mrr"] for c in configs]
        ndcgs = [c["so1_mean_ndcg_10"] for c in configs]

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, mrrs, w, label="MRR", color=COLORS[0], edgecolor="white")
        ax.bar(x + w/2, ndcgs, w, label="NDCG@10", color=COLORS[2], edgecolor="white")

        ax.axhline(0.80, color="red", linestyle="--", alpha=0.7, label="SO1 Target (0.80)")
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.legend()
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.savefig("figures/fig5_retrieval_configs.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: figures/fig5_retrieval_configs.png")


    # ── Figure 6: Category Performance Heatmap ───────────────────────────

    if emb:
        selected = next(m for m in emb if m["model_id"] == "intfloat/e5-small-v2")
        cats = selected["category_breakdown"]
        cat_names = sorted(cats.keys(), key=lambda c: -cats[c]["mean_mrr"])
        cat_n = [cats[c]["n"] for c in cat_names]
        cat_mrr = [cats[c]["mean_mrr"] for c in cat_names]
        cat_ndcg = [cats[c]["mean_ndcg_at_10"] for c in cat_names]

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.suptitle("Figure 6: e5-small-v2 — Per-Category MRR & NDCG@10", fontweight="bold")

        x = np.arange(len(cat_names))
        w = 0.35
        ax.barh(x - w/2, cat_mrr, w, label="MRR", color=COLORS[0], edgecolor="white")
        ax.barh(x + w/2, cat_ndcg, w, label="NDCG@10", color=COLORS[2], edgecolor="white")
        ax.set_yticks(x)
        ax.set_yticklabels([f"{c} (n={cats[c]['n']})" for c in cat_names], fontsize=9)
        ax.set_xlabel("Score")
        ax.set_xlim(0, 1.1)
        ax.axvline(0.80, color="red", linestyle="--", alpha=0.5, label="Target")
        ax.invert_yaxis()
        ax.legend()

        plt.tight_layout()
        plt.savefig("figures/fig6_category_performance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: figures/fig6_category_performance.png")


    # ── Figure 7: Cross-Pipeline Comparison ──────────────────────────────

    if pipelines:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Figure 7: Cross-Pipeline Comparison", fontweight="bold", fontsize=14)

        names = [p["pipeline"] for p in pipelines]
        x = np.arange(len(names))

        # ROUGE-L
        ax = axes[0]
        vals = [p["rouge_l"] for p in pipelines]
        bars = ax.bar(x, vals, color=COLORS[:len(pipelines)], edgecolor="white")
        ax.set_title("ROUGE-L")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

        # Hallucination
        ax = axes[1]
        vals = [p["halluc"] for p in pipelines]
        bars = ax.bar(x, vals, color=COLORS[:len(pipelines)], edgecolor="white")
        ax.set_title("Hallucination Rate (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9)

        # Avg Time
        ax = axes[2]
        vals = [p["time"] for p in pipelines]
        bars = ax.bar(x, vals, color=COLORS[:len(pipelines)], edgecolor="white")
        ax.set_title("Avg Response Time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}s",
                    ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig("figures/fig7_cross_pipeline.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: figures/fig7_cross_pipeline.png")


    # ── Figure 8: Improved RAG Phase 2 (if available) ────────────────────

    if p2:
        p2_best = best_per_model(p2)
        models = sorted(p2_best.keys())

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Figure 8: Improved RAG Phase 2 — Generation (Best Per Model)", fontweight="bold", fontsize=14)

        x = np.arange(len(models))

        # ROUGE-L
        ax = axes[0]
        vals = [p2_best[m]["avg_rouge_l"] for m in models]
        bars = ax.bar(x, vals, color=COLORS[:len(models)], edgecolor="white")
        ax.set_title("ROUGE-L")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

        # Hallucination
        ax = axes[1]
        vals = [p2_best[m]["hallucination_rate"] for m in models]
        bars = ax.bar(x, vals, color=COLORS[:len(models)], edgecolor="white")
        ax.set_title("Hallucination Rate (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)

        # Citation Precision
        ax = axes[2]
        vals = [p2_best[m].get("avg_citation_precision", 0) for m in models]
        bars = ax.bar(x, vals, color=COLORS[:len(models)], edgecolor="white")
        ax.set_title("Citation Precision")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)

        plt.tight_layout()
        plt.savefig("figures/fig8_improved_rag_phase2.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: figures/fig8_improved_rag_phase2.png")


# ══════════════════════════════════════════════════════════════════════════
# SAVE REPORT
# ══════════════════════════════════════════════════════════════════════════

print_header("DONE")
print("  Re-run after Phase 2 finishes to include improved RAG generation results.")
print("  Figures saved to: ./figures/")
