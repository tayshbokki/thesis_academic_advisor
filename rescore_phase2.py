"""
rescore_phase2.py — Post-hoc correction of Phase 2 evaluation metrics
=====================================================================
Applies three fixes to an existing Phase 2 results JSON without re-running
any LLM calls:

  1. CITATION PARSER FIX
     The original parser looked only for [Source: ...] markers, but the
     models primarily produce [Chunk: ...] markers (the same format used
     in the context prompt). Both formats are now accepted.

  2. HALLUCINATION DETECTOR FIX
     The original detector's WRONG_CODES rule matched any uppercase
     3–8 letter word as a potential hallucinated course code. This
     false-positives on ordinals ("SEVENTH", "TENTH") and grade letters
     ("INC") that legitimately appear in generated answers. An expanded
     noise list suppresses these.

  3. ROUGE / BLEU / METEOR FIX
     The original metrics scored generated answers including [Chunk: X]
     and [Source: X] markers, which inflate answer length and break
     n-gram overlap with citation-free ground truth. Citation markers
     are stripped from the generated answer before re-scoring.
     (BERTScore is NOT recomputed — it's batched, GPU-heavy, and the
     semantic similarity it measures is much less sensitive to a few
     extra tokens. We note its limitation rather than rerun it.)

Inputs:
  - improved_rag_phase2_{split}_results.json  (from improved_rag.py)

Outputs:
  - improved_rag_phase2_{split}_rescored.json   (full detail, corrected)
  - improved_rag_phase2_{split}_rescored.txt    (summary table)

Usage:
    python rescore_phase2.py --in  improved_rag_phase2_test_results.json
    python rescore_phase2.py --in  improved_rag_phase2_test_results.json \\
                             --out improved_rag_phase2_test_rescored.json
"""

import argparse
import json
import re
from pathlib import Path

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("omw-1.4",   quiet=True)


# ─── Fix 1: expanded citation parser ─────────────────────────────────────────

_CITATION_RE = re.compile(r"\[(?:Source|Chunk):\s*([^\]]+)\]", re.IGNORECASE)


def parse_citations_fixed(answer: str) -> list[str]:
    """Extract cited IDs from both [Source: ...] and [Chunk: ...] markers.

    Splits multi-source markers like [Chunk: SQL_RESULT, faq] into
    individual citations — some models group them.
    """
    raw = _CITATION_RE.findall(answer or "")
    out = []
    for item in raw:
        for c in item.split(","):
            c = c.strip()
            if c:
                out.append(c)
    return out


# ─── Fix 2: expanded hallucination noise list ────────────────────────────────
# Original list from improved_rag.py, plus ordinals and grade codes that
# showed up as false positives in the test run.

_NOISE_CODES = {
    # Original noise tokens
    "I", "H", "S", "C", "OK", "ONLY", "DLSU", "NO", "YES",
    "BOTH", "TERM", "AY", "NOTE", "GPA", "CUM", "AND", "OR",
    "NOT", "ALL", "IF", "THE", "FOR", "SAS", "OJT", "PE",
    "GE", "CPE", "ECE", "IT", "CS", "LET",
    # Ordinals written in uppercase (Gemini flags these as course codes)
    "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH",
    "SEVENTH", "EIGHTH", "NINTH", "TENTH", "ELEVENTH", "TWELFTH",
    # Grade-related uppercase tokens
    "INC", "DRP", "WP", "WF", "PASS", "FAIL", "PASSED", "FAILED",
    # Common hedge / explanation tokens
    "SQL", "FAQ", "CHUNK", "SOURCE", "CHECKLIST", "POLICY",
    "HANDBOOK", "CURRICULUM", "SEMESTER", "TRIMESTER",
    # Program / document phrases
    "BSCPE", "BSECE", "MOU", "ID", "NSTP", "ROTC",
}


def detect_hallucination_fixed(ground_truth: str, answer: str) -> tuple[bool, str]:
    """Same logic as improved_rag.py's detect_hallucination but with the
    expanded noise list that filters ordinal false positives.

    Also strips citation markers before scanning for course codes so that
    citation IDs (which are not hallucinations) don't trigger WRONG_CODES.
    """
    gt, ans = ground_truth.lower(), answer.lower()

    # RETRIEVAL_MISS — model said "no info" when the ground truth exists
    no_info = ["don't have that information", "cannot find", "no information",
               "i'm not sure", "i do not have", "consult your adviser",
               "unable to provide", "not in the context", "context does not"]
    if any(p in ans for p in no_info) and len(ground_truth) > 30:
        return True, "RETRIEVAL_MISS"

    # CONTRADICTION — ground truth says "no/cannot" but answer says "yes/can"
    gt_neg = any(w in gt for w in ["no.", "cannot", "must not", "not allowed", "invalid"])
    ans_pos = any(w in ans for w in ["yes,", "yes.", "you can", "is allowed", "is permitted"])
    ans_agr = any(w in ans for w in ["no,", "no.", "cannot", "must not", "not allowed"])
    if gt_neg and ans_pos and not ans_agr:
        return True, "CONTRADICTION"

    # WRONG_CODES — strip citation markers first so IDs aren't falsely flagged
    answer_no_cites = _CITATION_RE.sub(" ", answer)
    gt_codes  = set(re.findall(r"\b[A-Z]{3,8}\d*[A-Z]?\b", ground_truth))
    ans_codes = set(re.findall(r"\b[A-Z]{3,8}\d*[A-Z]?\b", answer_no_cites))
    wrong = ans_codes - gt_codes - _NOISE_CODES
    if wrong:
        return True, f"WRONG_CODES:{wrong}"

    return False, "OK"


# ─── Fix 3: strip citation markers before lexical metrics ────────────────────

_rouge  = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method4


def strip_citations(text: str) -> str:
    """Remove all [Source: ...] and [Chunk: ...] markers from a generated
    answer. This produces the 'answer content' that should be compared to
    a citation-free ground truth. Extra whitespace is collapsed."""
    return re.sub(r"\s+", " ", _CITATION_RE.sub("", text or "")).strip()


def rescore_lexical(ground_truth: str, answer: str) -> dict:
    """Recompute ROUGE / BLEU / METEOR on the citation-stripped answer."""
    stripped = strip_citations(answer)
    r = _rouge.score(ground_truth, stripped)
    ref = [ground_truth.lower().split()]
    hyp = stripped.lower().split()
    bleu = sentence_bleu(ref, hyp, smoothing_function=_smooth)
    met  = meteor_score([ground_truth.lower().split()], stripped.lower().split())
    return {
        "rouge1":  round(r["rouge1"].fmeasure, 4),
        "rouge_l": round(r["rougeL"].fmeasure, 4),
        "bleu":    round(bleu, 4),
        "meteor":  round(met, 4),
    }


# ─── Citation precision/recall (same formula, corrected inputs) ──────────────

def compute_citation_metrics(cited: list[str], chunk_ids: list[str],
                              relevant_ids: list[str], had_sql: bool) -> dict:
    available = set(chunk_ids)
    if had_sql:
        available.add("SQL_RESULT")

    if not cited:
        return {"citation_precision": 0.0, "citation_recall": 0.0, "n_citations": 0}

    valid_citations = sum(
        1 for c in cited
        if c in available or any(c.lower() in a.lower() for a in available)
    )
    precision = valid_citations / len(cited) if cited else 0.0

    relevant_set = set(relevant_ids)
    if relevant_set:
        cited_set = set(cited)
        cited_relevant = sum(
            1 for r in relevant_set
            if r in cited_set or any(r.lower() in c.lower() for c in cited_set)
        )
        recall = cited_relevant / len(relevant_set)
    else:
        recall = 0.0

    return {
        "citation_precision": round(precision, 4),
        "citation_recall":    round(recall, 4),
        "n_citations":        len(cited),
    }


# ─── Main rescoring loop ─────────────────────────────────────────────────────

def rescore(in_path: str, out_path: str) -> list[dict]:
    data = json.load(open(in_path, encoding="utf-8"))
    print(f"Loaded {len(data)} configs from {in_path}")

    rescored = []
    for cfg_idx, cfg in enumerate(data):
        details = cfg["detail"]
        n = len(details)

        # Per-query rescoring
        new_detail = []
        new_halluc = 0
        sum_r1 = sum_rl = sum_bleu = sum_met = 0.0
        sum_cp = sum_cr = sum_nc = 0.0
        halluc_reasons = []

        for d in details:
            gt  = d["ground_truth"]
            gen = d["generated"]

            # Fix 1: citations
            cited_fixed = parse_citations_fixed(gen)

            # Fix 3: strip markers, recompute lexical metrics
            lex = rescore_lexical(gt, gen)

            # Fix 2: hallucination with expanded noise list
            is_halluc_new, reason_new = detect_hallucination_fixed(gt, gen)

            # Citation precision/recall using the corrected citation list
            cit = compute_citation_metrics(
                cited_fixed,
                d["retrieved_ids"],
                d.get("relevant_ids", []),   # may be absent in Phase 2 JSON
                d.get("sql_context", False),
            )

            new_d = dict(d)
            new_d.update({
                "rouge1":  lex["rouge1"],
                "rouge_l": lex["rouge_l"],
                "bleu":    lex["bleu"],
                "meteor":  lex["meteor"],
                "citations":           cited_fixed,
                "citation_precision":  cit["citation_precision"],
                "citation_recall":     cit["citation_recall"],
                "n_citations":         cit["n_citations"],
                "hallucination":       is_halluc_new,
                "halluc_reason":       reason_new,
                # Keep original values for traceability
                "_original_rouge_l":        d["rouge_l"],
                "_original_bleu":           d["bleu"],
                "_original_hallucination":  d["hallucination"],
                "_original_halluc_reason":  d["halluc_reason"],
                "_original_citations":      d["citations"],
            })
            new_detail.append(new_d)

            sum_r1   += lex["rouge1"]
            sum_rl   += lex["rouge_l"]
            sum_bleu += lex["bleu"]
            sum_met  += lex["meteor"]
            sum_cp   += cit["citation_precision"]
            sum_cr   += cit["citation_recall"]
            sum_nc   += cit["n_citations"]
            if is_halluc_new:
                new_halluc += 1
                halluc_reasons.append(reason_new)

        # Build updated summary (keep BERTScore as-is — see rationale in docstring)
        new_cfg = dict(cfg)
        new_cfg.update({
            "avg_rouge1":               round(sum_r1 / n, 4),
            "avg_rouge_l":              round(sum_rl / n, 4),
            "avg_bleu":                 round(sum_bleu / n, 4),
            "avg_meteor":               round(sum_met / n, 4),
            "avg_citation_precision":   round(sum_cp / n, 4),
            "avg_citation_recall":      round(sum_cr / n, 4),
            "avg_citations_per_answer": round(sum_nc / n, 2),
            "hallucination_count":      new_halluc,
            "hallucination_rate":       round(new_halluc / n * 100, 1),
            "detail":                   new_detail,
            # Original values for provenance
            "_original_avg_rouge_l":             cfg["avg_rouge_l"],
            "_original_avg_bleu":                cfg["avg_bleu"],
            "_original_hallucination_count":     cfg["hallucination_count"],
            "_original_avg_citation_precision":  cfg["avg_citation_precision"],
            "_original_avg_citation_recall":     cfg["avg_citation_recall"],
        })
        rescored.append(new_cfg)

        label = f"{cfg['model']} [{cfg['config']}]"
        print(f"  [{cfg_idx+1:2d}/{len(data)}] {label:<48} "
              f"RL:{cfg['avg_rouge_l']:.3f}->{new_cfg['avg_rouge_l']:.3f}  "
              f"Hal:{cfg['hallucination_count']:>3}->{new_halluc:>3}  "
              f"CitP:{cfg['avg_citation_precision']:.3f}->{new_cfg['avg_citation_precision']:.3f}")

    # Save JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rescored, f, indent=2, ensure_ascii=False)
    print(f"\nRescored JSON saved: {out_path}")

    # Save summary table
    txt_path = out_path.replace(".json", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 130 + "\n")
        f.write("IMPROVED RAG — RESCORED (citation parser + ordinal filter + "
                "citation-stripped lexical metrics)\n")
        f.write("=" * 130 + "\n")
        f.write(f"{'Model + Config':<48} {'RL':>6} {'BLEU':>6} {'BERT':>6} "
                f"{'Halluc':>8} {'CitP':>6} {'CitR':>6}  "
                f"({'originals in parens'})\n")
        f.write("-" * 130 + "\n")

        for s in sorted(rescored, key=lambda x: -x["avg_rouge_l"]):
            name = f"{s['model']} [{s['config']}]"
            f.write(
                f"{name:<48} "
                f"{s['avg_rouge_l']:>6.3f} "
                f"{s['avg_bleu']:>6.3f} "
                f"{s['avg_bert_score']:>6.3f} "
                f"{s['hallucination_count']:>4}/{s['n_questions']} "
                f"{s['avg_citation_precision']:>6.3f} "
                f"{s['avg_citation_recall']:>6.3f}  "
                f"(RL was {s['_original_avg_rouge_l']:.3f}, "
                f"Hal was {s['_original_hallucination_count']}, "
                f"CitP was {s['_original_avg_citation_precision']:.3f})\n"
            )
    print(f"Summary table saved: {txt_path}")

    return rescored


def print_comparison_summary(rescored: list[dict]) -> None:
    """Print a per-model before/after summary to stdout."""
    by_model = {}
    for cfg in rescored:
        m = cfg["model"]
        by_model.setdefault(m, []).append(cfg)

    print("\n" + "=" * 100)
    print("PER-MODEL SUMMARY — best config (by rescored ROUGE-L)")
    print("=" * 100)
    print(f"{'Model':<25} {'RL orig':>9} {'RL new':>9} {'Δ RL':>8}   "
          f"{'Hal orig':>9} {'Hal new':>9} {'Δ Hal':>8}   "
          f"{'CitP orig':>10} {'CitP new':>10}")
    print("-" * 100)
    for model, cfgs in sorted(by_model.items()):
        best = max(cfgs, key=lambda x: x["avg_rouge_l"])
        rl_delta   = best["avg_rouge_l"] - best["_original_avg_rouge_l"]
        hal_delta  = best["hallucination_count"] - best["_original_hallucination_count"]
        print(f"{model:<25} "
              f"{best['_original_avg_rouge_l']:>9.3f} "
              f"{best['avg_rouge_l']:>9.3f} "
              f"{rl_delta:>+8.3f}   "
              f"{best['_original_hallucination_count']:>9d} "
              f"{best['hallucination_count']:>9d} "
              f"{hal_delta:>+8d}   "
              f"{best['_original_avg_citation_precision']:>10.3f} "
              f"{best['avg_citation_precision']:>10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path",
                        default="improved_rag_phase2_test_results.json",
                        help="Phase 2 results JSON to rescore.")
    parser.add_argument("--out", dest="out_path", default=None,
                        help="Output path. Default: <in>_rescored.json")
    args = parser.parse_args()

    in_path = args.in_path
    if not Path(in_path).exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    if args.out_path:
        out_path = args.out_path
    else:
        # Replace _results.json with _rescored.json
        if in_path.endswith("_results.json"):
            out_path = in_path.replace("_results.json", "_rescored.json")
        else:
            out_path = in_path.replace(".json", "_rescored.json")

    rescored = rescore(in_path, out_path)
    print_comparison_summary(rescored)
