"""
thesis_charts.py — Cross-system visualizations for RAG thesis
==============================================================
Produces five charts comparing No-RAG, Naive RAG, and Improved RAG systems.
All charts use editorial/minimal aesthetic suitable for thesis results chapters.

Inputs (next to this script):
  - no_rag_baseline_results.json
  - naive_rag_results.json
  - improved_rag_phase2_test_rescored.json
  - improved_rag_phase1_test_results.json  (for SO1 numbers)

Outputs (PNGs + a combined PDF report):
  1. chart_01_progression.png      — SO3 metrics across three systems
  2. chart_02_hallucination.png    — hallucination rates across systems
  3. chart_03_category_heatmap.png — NDCG@10 and ROUGE-L by category
  4. chart_04_latency_quality.png  — SO4 tradeoff scatter
  5. chart_05_naive_vs_improved.png — per-query head-to-head scatter

Usage:
    python thesis_charts.py
"""

import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np


# ─── Config ───────────────────────────────────────────────────────────────

FILES = {
    "no_rag":   "no_rag_baseline_results.json",
    "naive":    "naive_rag_results.json",
    "improved": "improved_rag_phase2_test_rescored.json",
    "phase1":   "improved_rag_phase1_test_results.json",
}

# Models to compare across all three systems (overlap set)
MODELS = ["Llama-3.1-8B", "Qwen2.5-7B", "Gemma-2-9B",
          "Gemini-2.5-Flash-Lite", "GPT-4o-mini"]

# System-level colors — academic palette, not AI-generic
SYSTEM_COLORS = {
    "no_rag":   "#c9a96e",   # muted gold
    "naive":    "#6b8ca8",   # slate blue
    "improved": "#2f4858",   # deep teal
}
SYSTEM_LABELS = {
    "no_rag":   "No-RAG (LLM only)",
    "naive":    "Naive RAG",
    "improved": "Improved RAG",
}

# Editorial matplotlib defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.edgecolor": "#2b2b2b",
    "axes.linewidth": 0.7,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.color": "#2b2b2b",
    "ytick.color": "#2b2b2b",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ─── Helpers ──────────────────────────────────────────────────────────────

def normalize_model(name: str) -> str:
    """Normalize model names — naive RAG has 'Gemma-2-9b-it' vs others' 'Gemma-2-9B'."""
    if name.lower().startswith("gemma"):
        return "Gemma-2-9B"
    return name


def load_all() -> dict:
    """Load all JSONs, abort with a clear message if any are missing."""
    out = {}
    for key, fname in FILES.items():
        p = Path(fname)
        if not p.exists():
            raise FileNotFoundError(
                f"Missing input: {fname}\n"
                f"Place this script next to the five result JSONs."
            )
        out[key] = json.load(open(p, encoding="utf-8"))
    return out


def best_config_per_model(configs: list[dict], metric: str = "avg_rouge_l") -> dict:
    """For each model, return the single best-performing config by `metric`.
    Returns {model_name: config_dict}."""
    by_model = {}
    for cfg in configs:
        m = normalize_model(cfg["model"])
        if m not in MODELS:
            continue
        if m not in by_model or cfg[metric] > by_model[m][metric]:
            by_model[m] = cfg
    return by_model


def system_best(configs: list[dict], metric: str = "avg_rouge_l") -> dict:
    """Flatten: for each system, one best config per model."""
    return best_config_per_model(configs, metric)


def strip_citations(text: str) -> str:
    return re.sub(r"\s+", " ",
                  re.sub(r"\[(?:Source|Chunk):\s*[^\]]+\]", "", text or "")).strip()


# ─── Chart 1: Progression bar chart ───────────────────────────────────────

def chart_progression(all_data: dict, out_path: str) -> None:
    """Grouped bars: 3 systems × N models × 3 metrics (RL, BLEU, BERT).
    Two-panel layout: top panel = SO3 metrics, bottom panel = (not used here,
    a single panel works better for the progression narrative)."""

    systems = ["no_rag", "naive", "improved"]
    best = {s: system_best(all_data[s]) for s in systems}

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    metrics = [("avg_rouge_l", "ROUGE-L"),
               ("avg_bleu",    "BLEU"),
               ("avg_bert_score", "BERTScore")]

    n_models = len(MODELS)
    x = np.arange(n_models)
    width = 0.27

    for ax_i, (metric, label) in enumerate(metrics):
        ax = axes[ax_i]
        for s_i, sys in enumerate(systems):
            vals = [best[sys].get(m, {}).get(metric, 0) for m in MODELS]
            ax.bar(x + (s_i - 1) * width, vals, width,
                   color=SYSTEM_COLORS[sys], edgecolor="#1a1a1a",
                   linewidth=0.4,
                   label=SYSTEM_LABELS[sys] if ax_i == 0 else None)

        ax.set_title(label, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-Flash-Lite", "-FL").replace("2.5-", "")
                            for m in MODELS], rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.22),
                   ncol=3, fontsize=9)

    fig.text(0.5, 0.98,
             "SO3 generation quality across RAG complexity levels",
             ha="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             "Best config per model; test split, n=219",
             ha="center", fontsize=10, color="#555", style="italic")

    plt.subplots_adjust(top=0.78, bottom=0.18, wspace=0.12)
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ─── Chart 2: Hallucination rate comparison ───────────────────────────────

def chart_hallucination(all_data: dict, out_path: str) -> None:
    """Grouped bar chart: hallucination rate (%) per model × 3 systems."""

    systems = ["no_rag", "naive", "improved"]
    best = {s: system_best(all_data[s], metric="avg_rouge_l") for s in systems}

    fig, ax = plt.subplots(figsize=(9, 5.2))

    x = np.arange(len(MODELS))
    width = 0.27

    for s_i, sys in enumerate(systems):
        rates = [best[sys].get(m, {}).get("hallucination_rate", 0) for m in MODELS]
        bars = ax.bar(x + (s_i - 1) * width, rates, width,
                      color=SYSTEM_COLORS[sys], edgecolor="#1a1a1a",
                      linewidth=0.4, label=SYSTEM_LABELS[sys])
        # Numeric labels on top
        for b, v in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-Flash-Lite", "-FL").replace("2.5-", "")
                        for m in MODELS], rotation=15, ha="right")
    ax.set_ylabel("Hallucination rate (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    fig.text(0.125, 0.96,
             "Hallucination rate across RAG complexity levels",
             fontsize=13, fontweight="bold", ha="left")
    fig.text(0.125, 0.925,
             "Lower is better; detected via ground-truth code/contradiction matching",
             fontsize=9, color="#555", style="italic", ha="left")

    plt.subplots_adjust(top=0.87, bottom=0.17)
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ─── Chart 3: Per-category heatmap ────────────────────────────────────────

def chart_category_heatmap(all_data: dict, out_path: str) -> None:
    """Two-panel heatmap: left = ROUGE-L by (category × system),
    right = hallucination rate by (category × system). Uses the per-query
    `detail` field — best model per system."""

    systems = ["no_rag", "naive", "improved"]

    # Get best-ROUGE-L model per system, use its per-query detail
    best_cfg = {}
    for s in systems:
        per_model = system_best(all_data[s])
        best_cfg[s] = max(per_model.values(), key=lambda c: c["avg_rouge_l"])

    # Collect unique categories seen across details (should be 21)
    cats = sorted({d["category"] for d in best_cfg["improved"]["detail"]})

    # For each (system, category), compute mean ROUGE-L and hallucination rate
    rl_mat = np.full((len(cats), len(systems)), np.nan)
    hal_mat = np.full((len(cats), len(systems)), np.nan)

    for s_i, s in enumerate(systems):
        by_cat = {}
        for d in best_cfg[s]["detail"]:
            by_cat.setdefault(d["category"], []).append(d)
        for c_i, c in enumerate(cats):
            entries = by_cat.get(c, [])
            if entries:
                rl_mat[c_i, s_i]  = statistics.mean(e["rouge_l"] for e in entries)
                hal_mat[c_i, s_i] = 100 * sum(1 for e in entries if e["hallucination"]) / len(entries)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, max(6.5, 0.32 * len(cats))))

    # Panel 1: ROUGE-L heatmap
    im1 = axes[0].imshow(rl_mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(systems)))
    axes[0].set_xticklabels([SYSTEM_LABELS[s] for s in systems],
                            rotation=15, ha="right", fontsize=9)
    axes[0].set_yticks(range(len(cats)))
    axes[0].set_yticklabels(cats, fontsize=8)
    axes[0].set_title("Mean ROUGE-L", pad=10, fontsize=11)
    for i in range(len(cats)):
        for j in range(len(systems)):
            if not np.isnan(rl_mat[i, j]):
                val = rl_mat[i, j]
                color = "white" if val > 0.55 else "#1a1a1a"
                axes[0].text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=7, color=color)
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.038, pad=0.03)
    cbar1.ax.tick_params(labelsize=8)

    # Panel 2: Hallucination rate heatmap (inverted cmap — higher = worse)
    im2 = axes[1].imshow(hal_mat, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    axes[1].set_xticks(range(len(systems)))
    axes[1].set_xticklabels([SYSTEM_LABELS[s] for s in systems],
                            rotation=15, ha="right", fontsize=9)
    axes[1].set_yticks(range(len(cats)))
    axes[1].set_yticklabels([])
    axes[1].set_title("Hallucination rate (%)", pad=10, fontsize=11)
    for i in range(len(cats)):
        for j in range(len(systems)):
            if not np.isnan(hal_mat[i, j]):
                val = hal_mat[i, j]
                color = "white" if val > 50 else "#1a1a1a"
                axes[1].text(j, i, f"{val:.0f}", ha="center", va="center",
                             fontsize=7, color=color)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.038, pad=0.03)
    cbar2.ax.tick_params(labelsize=8)

    fig.text(0.5, 0.985, "Per-category performance across RAG systems",
             ha="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.955,
             "Best ROUGE-L model per system; test split, n=219 across 21 categories",
             ha="center", fontsize=9, color="#555", style="italic")

    plt.subplots_adjust(top=0.91, bottom=0.12, left=0.17, right=0.96, wspace=0.10)
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ─── Chart 4: Latency vs quality scatter ──────────────────────────────────

def chart_latency_quality(all_data: dict, out_path: str) -> None:
    """Scatter: avg_total_time (x) vs avg_rouge_l (y), color = system,
    marker = model. One point per (system, model, config) combination.
    Shows SO4 tradeoff."""

    systems = ["no_rag", "naive", "improved"]
    markers = {"Llama-3.1-8B": "o", "Qwen2.5-7B": "s", "Gemma-2-9B": "D",
               "Gemini-2.5-Flash-Lite": "^", "GPT-4o-mini": "v"}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for sys in systems:
        for cfg in all_data[sys]:
            m = normalize_model(cfg["model"])
            if m not in MODELS:
                continue
            # Time field varies by system — handle all three
            t = cfg.get("avg_total_time") or cfg.get("avg_generation_time") or cfg.get("avg_gen_time", 0)
            q = cfg.get("avg_rouge_l", 0)
            ax.scatter(t, q, s=48, color=SYSTEM_COLORS[sys],
                       marker=markers[m], alpha=0.72,
                       edgecolors="#1a1a1a", linewidths=0.5)

    # Pareto frontier: best-quality point at each latency level
    all_points = []
    for sys in systems:
        for cfg in all_data[sys]:
            m = normalize_model(cfg["model"])
            if m not in MODELS:
                continue
            t = cfg.get("avg_total_time") or cfg.get("avg_generation_time") or cfg.get("avg_gen_time", 0)
            q = cfg.get("avg_rouge_l", 0)
            all_points.append((t, q))
    all_points.sort()
    pareto = []
    max_q = -1
    for t, q in all_points:
        if q > max_q:
            pareto.append((t, q))
            max_q = q
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.plot(px, py, linestyle="--", color="#b03a2e", linewidth=1.2,
                alpha=0.6, zorder=0, label="Pareto frontier")

    ax.set_xlabel("Average total time per query (seconds)")
    ax.set_ylabel("ROUGE-L")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    # Two legends — systems (color) and models (marker)
    sys_handles = [mpatches.Patch(facecolor=SYSTEM_COLORS[s],
                                   edgecolor="#1a1a1a", linewidth=0.5,
                                   label=SYSTEM_LABELS[s]) for s in systems]
    model_handles = [plt.Line2D([], [], marker=markers[m], color="#444",
                                 markersize=7, linestyle="",
                                 markeredgecolor="#1a1a1a", markeredgewidth=0.5,
                                 label=m.replace("-Flash-Lite", "-FL").replace("2.5-", ""))
                      for m in MODELS]
    if len(pareto) > 1:
        sys_handles.append(plt.Line2D([], [], linestyle="--", color="#b03a2e",
                                       label="Pareto frontier"))

    leg1 = ax.legend(handles=sys_handles, loc="lower right", fontsize=9,
                     title="System", title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, loc="upper right", fontsize=8,
              title="Model", title_fontsize=9)

    fig.text(0.125, 0.96,
             "Latency vs quality tradeoff (SO4)",
             fontsize=13, fontweight="bold", ha="left")
    fig.text(0.125, 0.925,
             "Each point is one (system, model, config) combination",
             fontsize=9, color="#555", style="italic", ha="left")

    plt.subplots_adjust(top=0.87, bottom=0.11, left=0.10, right=0.96)
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ─── Chart 5: Naive vs Improved per-query scatter ─────────────────────────

def chart_naive_vs_improved(all_data: dict, out_path: str) -> None:
    """Per-query scatter: x = naive RAG ROUGE-L, y = improved RAG ROUGE-L.
    Points above diagonal = improved won; below = naive won.
    Uses matched (by query id) results from best model in each system."""

    # Find a common model that exists in both — use GPT-4o-mini (stable across systems)
    # but fall back to any model present in both
    common_model = None
    naive_models = {normalize_model(c["model"]) for c in all_data["naive"]}
    improved_models = {normalize_model(c["model"]) for c in all_data["improved"]}
    for candidate in ["GPT-4o-mini", "Gemini-2.5-Flash-Lite", "Qwen2.5-7B",
                      "Gemma-2-9B", "Llama-3.1-8B"]:
        if candidate in naive_models and candidate in improved_models:
            common_model = candidate
            break

    if common_model is None:
        print("  [SKIP] No overlapping model between naive and improved — cannot plot.")
        return

    # Pick best config for common_model in each system
    naive_cfg = max(
        [c for c in all_data["naive"] if normalize_model(c["model"]) == common_model],
        key=lambda c: c["avg_rouge_l"],
    )
    improved_cfg = max(
        [c for c in all_data["improved"] if normalize_model(c["model"]) == common_model],
        key=lambda c: c["avg_rouge_l"],
    )

    # Join on question id
    naive_by_id = {d["id"]: d for d in naive_cfg["detail"]}
    improved_by_id = {d["id"]: d for d in improved_cfg["detail"]}

    # If ids are None, fall back to matching on question text
    if all(k is None for k in naive_by_id) or all(k is None for k in improved_by_id):
        naive_by_id = {d["question"]: d for d in naive_cfg["detail"]}
        improved_by_id = {d["question"]: d for d in improved_cfg["detail"]}

    shared = set(naive_by_id) & set(improved_by_id)
    if not shared:
        print("  [SKIP] No overlapping queries between naive and improved.")
        return

    pairs = [(naive_by_id[k]["rouge_l"], improved_by_id[k]["rouge_l"],
              naive_by_id[k]["category"]) for k in shared]

    naive_vals = [p[0] for p in pairs]
    improved_vals = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # Color by win/loss
    colors = []
    for n, i, _ in pairs:
        if i > n + 0.02:
            colors.append("#2f4858")   # improved wins
        elif n > i + 0.02:
            colors.append("#b03a2e")   # naive wins
        else:
            colors.append("#888888")   # tie (±0.02)

    ax.scatter(naive_vals, improved_vals, s=22, c=colors, alpha=0.55,
               edgecolors="none")

    # Diagonal
    ax.plot([0, 1], [0, 1], color="#444", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.text(0.98, 0.965, "y = x", fontsize=8, color="#555",
            ha="right", va="top", style="italic",
            transform=ax.transAxes)

    # Win/loss counts
    n_improved_wins = sum(1 for n, i, _ in pairs if i > n + 0.02)
    n_naive_wins = sum(1 for n, i, _ in pairs if n > i + 0.02)
    n_ties = len(pairs) - n_improved_wins - n_naive_wins

    ax.text(0.04, 0.96,
            f"Improved better: {n_improved_wins}",
            color="#2f4858", fontweight="bold", fontsize=10,
            transform=ax.transAxes, va="top")
    ax.text(0.04, 0.92,
            f"Naive better: {n_naive_wins}",
            color="#b03a2e", fontweight="bold", fontsize=10,
            transform=ax.transAxes, va="top")
    ax.text(0.04, 0.88,
            f"Tied (±0.02): {n_ties}",
            color="#555", fontsize=10,
            transform=ax.transAxes, va="top")

    ax.set_xlabel(f"Naive RAG ROUGE-L  ({common_model}, n={naive_cfg['n_questions']})")
    ax.set_ylabel(f"Improved RAG ROUGE-L  ({common_model})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_aspect("equal")

    fig.text(0.125, 0.96,
             "Per-query comparison: Naive vs Improved RAG",
             fontsize=13, fontweight="bold", ha="left")
    fig.text(0.125, 0.925,
             f"Each dot is one of {len(pairs)} test queries; matched by question id",
             fontsize=9, color="#555", style="italic", ha="left")

    plt.subplots_adjust(top=0.87, bottom=0.10, left=0.12, right=0.95)
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading result JSONs...")
    data = load_all()
    for k, v in data.items():
        print(f"  {k}: {len(v)} entries")

    print("\nBuilding charts...")
    chart_progression(data,        "chart_01_progression.png")
    chart_hallucination(data,      "chart_02_hallucination.png")
    chart_category_heatmap(data,   "chart_03_category_heatmap.png")
    chart_latency_quality(data,    "chart_04_latency_quality.png")
    chart_naive_vs_improved(data,  "chart_05_naive_vs_improved.png")

    print("\nAll charts saved.")
