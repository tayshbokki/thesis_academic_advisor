"""
embedding_charts.py — Visualizations for the embedding model comparison
========================================================================
Produces three charts documenting the embedding experiment that selected
intfloat/e5-small-v2 as the retrieval model for the RAG pipeline.

Inputs (next to this script):
  - results_summary.json  (one entry per model with aggregate + breakdowns)

Outputs:
  1. embed_01_model_comparison.png   — SO1 metrics across 5 models
  2. embed_02_speed_quality.png      — latency vs NDCG@10 tradeoff
  3. embed_03_category_heatmap.png   — NDCG@10 per (category × model)

Usage:
    python embedding_charts.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


INPUT = "results_summary.json"

# Editorial matplotlib defaults (matches thesis_charts.py)
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

# Deliberately distinct palette for 5 models — ordered light→dark so the
# selected model (e5-small-v2) sits in a visually distinguishable position.
MODEL_COLORS = {
    "all-MiniLM-L6-v2":           "#d4a373",  # sand
    "all-mpnet-base-v2":          "#b56576",  # muted rose
    "multi-qa-mpnet-base-dot-v1": "#8d99ae",  # cool grey
    "e5-small-v2":                "#2f4858",  # deep teal (selected)
    "e5-base-v2":                 "#5b8291",  # mid teal
}

SELECTED = "e5-small-v2"


def load() -> list[dict]:
    p = Path(INPUT)
    if not p.exists():
        raise FileNotFoundError(f"Missing {INPUT} — place next to this script.")
    return json.load(open(p, encoding="utf-8"))


# ─── Chart 1: Model comparison bars ───────────────────────────────────────

def chart_model_comparison(data: list[dict], out_path: str) -> None:
    """Grouped bars: 4 SO1 metrics × 5 models. Selected model highlighted."""

    # Order models by composite (MRR + NDCG + R@10 - latency_penalty), selected last
    models = sorted(data, key=lambda m: (m["model_name"] == SELECTED,
                                          m["mean_mrr"]))
    names = [m["model_name"] for m in models]

    metrics = [
        ("mean_mrr",                      "MRR"),
        ("mean_ndcg_at_10",               "NDCG@10"),
        ("r10",                           "Recall@10"),
        ("r5",                            "Recall@5"),
    ]

    # Precompute values
    def val(m, key):
        if key == "r10": return m["mean_recall_at_k"]["10"]
        if key == "r5":  return m["mean_recall_at_k"]["5"]
        return m[key]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    n_metrics = len(metrics)
    n_models = len(names)
    group_width = 0.82
    bar_width = group_width / n_models

    x = np.arange(n_metrics)
    for m_i, m in enumerate(models):
        vals = [val(m, k) for k, _ in metrics]
        offset = (m_i - (n_models - 1) / 2) * bar_width
        color = MODEL_COLORS[m["model_name"]]
        is_selected = m["model_name"] == SELECTED
        bars = ax.bar(x + offset, vals, bar_width, color=color,
                      edgecolor="#b03a2e" if is_selected else "#1a1a1a",
                      linewidth=1.5 if is_selected else 0.4,
                      label=m["model_name"] + (" (SELECTED)" if is_selected else ""))
        # Numeric labels on the selected model
        if is_selected:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, color="#b03a2e", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, ncol=1)

    fig.text(0.125, 0.96,
             "SO1 retrieval metrics across candidate embedding models",
             fontsize=13, fontweight="bold", ha="left")
    fig.text(0.125, 0.925,
             f"n=535 queries; red outline = selected model; "
             f"red labels show selected values",
             fontsize=9, color="#555", style="italic", ha="left")

    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.97)
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ─── Chart 2: Speed vs quality scatter ────────────────────────────────────

def chart_speed_quality(data: list[dict], out_path: str) -> None:
    """Scatter: embed_time_per_doc_ms (x, log) vs NDCG@10 (y). The selected
    model should be in the upper-left region — fast AND accurate."""

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for m in data:
        name = m["model_name"]
        x = m["embed_time_per_doc_ms"]
        y = m["mean_ndcg_at_10"]
        is_selected = name == SELECTED

        ax.scatter(x, y, s=220 if is_selected else 140,
                   color=MODEL_COLORS[name],
                   edgecolor="#b03a2e" if is_selected else "#1a1a1a",
                   linewidth=2 if is_selected else 0.5,
                   zorder=4 if is_selected else 3,
                   label=name + (" (SELECTED)" if is_selected else ""))

        # Annotation — offset based on quadrant
        dx, dy = 4, 0.008
        if name == "all-mpnet-base-v2":     # low performer, avoid collision
            dx, dy = 4, -0.025
        elif name == "all-MiniLM-L6-v2":
            dx, dy = 4, -0.025
        elif name == "e5-base-v2":
            dx, dy = -4, -0.025                # overlap with e5-small-v2 NDCG
        elif name == "multi-qa-mpnet-base-dot-v1":
            dx, dy = 6, 0.010

        ax.annotate(name, (x, y), xytext=(x + dx, y + dy),
                    fontsize=8.5, color="#333",
                    fontweight="bold" if is_selected else "normal")

    # Shade the "desired region" — fast & accurate
    ax.axhspan(0.75, 1.0, xmin=0, xmax=0.35, alpha=0.06, color="#2f4858",
               zorder=0)
    ax.text(18, 0.97, "desired region\n(fast + accurate)",
            fontsize=8, color="#2f4858", alpha=0.7, style="italic",
            ha="center", va="top")

    ax.set_xscale("log")
    ax.set_xlabel("Embedding time per document (ms, log scale)")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0.3, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    fig.text(0.125, 0.96,
             "Embedding model selection: speed vs retrieval quality",
             fontsize=13, fontweight="bold", ha="left")
    fig.text(0.125, 0.925,
             "Upper-left is ideal; e5-small-v2 selected for best "
             "composite score",
             fontsize=9, color="#555", style="italic", ha="left")

    plt.subplots_adjust(top=0.88, bottom=0.11, left=0.10, right=0.97)
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ─── Chart 3: Per-category NDCG@10 heatmap ────────────────────────────────

def chart_category_heatmap(data: list[dict], out_path: str) -> None:
    """Rows = categories, Columns = models. Cell = NDCG@10.
    Shows where the zero-score categories are (consistent across all models,
    supporting the 'data pipeline issue' explanation in your thesis)."""

    # Canonical model order — put selected in a prominent position
    model_order = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1",
        "e5-base-v2",
        "e5-small-v2",   # selected, last column
    ]
    data_by_name = {m["model_name"]: m for m in data}
    models = [data_by_name[n] for n in model_order if n in data_by_name]

    # Collect categories — union across models
    all_cats = set()
    for m in models:
        all_cats.update(m["category_breakdown"].keys())
    # Sort by number of queries (desc) so the visually-dominant categories
    # appear at top — matches how a reader scans
    cat_counts = {c: models[0]["category_breakdown"].get(c, {}).get("n", 0)
                  for c in all_cats}
    cats = sorted(all_cats, key=lambda c: -cat_counts[c])

    # Build matrix
    mat = np.full((len(cats), len(models)), np.nan)
    for c_i, c in enumerate(cats):
        for m_i, m in enumerate(models):
            stats = m["category_breakdown"].get(c)
            if stats:
                mat[c_i, m_i] = stats["mean_ndcg_at_10"]

    fig, ax = plt.subplots(figsize=(9.5, max(6, 0.34 * len(cats))))

    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)

    ax.set_xticks(range(len(models)))
    labels = []
    for m in models:
        name = m["model_name"]
        marker = " (SELECTED)" if name == SELECTED else ""
        labels.append(name + marker)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)

    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([f"{c}  (n={cat_counts[c]})" for c in cats], fontsize=8)

    # Cell text
    for i in range(len(cats)):
        for j in range(len(models)):
            if not np.isnan(mat[i, j]):
                val = mat[i, j]
                color = "white" if val > 0.55 else "#1a1a1a"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color)

    # Highlight selected column with a red outline
    sel_idx = next((i for i, m in enumerate(models)
                    if m["model_name"] == SELECTED), None)
    if sel_idx is not None:
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((sel_idx - 0.5, -0.5), 1, len(cats),
                                fill=False, edgecolor="#b03a2e",
                                linewidth=2, zorder=5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("NDCG@10", fontsize=9)

    fig.text(0.5, 0.985, "Per-category NDCG@10 across embedding models",
             ha="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.955,
             "Zero-score categories persist across all 5 models — "
             "indicates data pipeline issue, not model limitation",
             ha="center", fontsize=9, color="#555", style="italic")

    plt.subplots_adjust(top=0.92, bottom=0.16, left=0.25, right=0.94)
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading embedding results...")
    data = load()
    print(f"  {len(data)} models")

    print("\nBuilding charts...")
    chart_model_comparison(data, "embed_01_model_comparison.png")
    chart_speed_quality(data,    "embed_02_speed_quality.png")
    chart_category_heatmap(data, "embed_03_category_heatmap.png")
    print("\nAll charts saved.")
