"""
standard_baseline.py — No-RAG Baseline Evaluation
===================================================
Tests LLMs on DLSU CpE academic advising queries WITHOUT retrieval-augmented
generation.  The purpose is to establish a lower-bound baseline so that the
subsequent RAG systems (Naive, Improved, Agentic) can be compared against
pure parametric knowledge.

Test cases are sampled from advising_dataset.xlsx (593 Q&A pairs) to ensure
coverage across all question categories and programs.

Metrics: ROUGE-1, ROUGE-L, BLEU, METEOR, BERTScore
Also: hallucination detection, response-time compliance (< 5 s threshold)

Usage:
    pip install openpyxl nltk rouge-score bert-score huggingface-hub
    # Optional: pip install google-genai openai
    python standard_baseline.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

import os, re, json, time, random
import nltk
import openpyxl
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from huggingface_hub import InferenceClient

nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("omw-1.4",   quiet=True)
nltk.download("stopwords", quiet=True)

# NLI-based context grounding scorer — not applicable for standard baseline
# (no retrieved context exists). Defined here for API consistency with RAG scripts.
_NLI_SCORER = None


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD TEST CASES FROM advising_dataset.xlsx
# ═══════════════════════════════════════════════════════════════════════════

DATASET_PATH = "dataset_test.xlsx"

def load_dataset(path: str) -> list[dict]:
    """Load all Q&A pairs from the advising dataset.

    Reads columns by name so the loader is robust to column-order changes
    across dataset versions (e.g. dataset-query.xlsx vs advising_dataset.xlsx).
    """
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    wb.close()

    if not rows:
        return []

    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    col = {h: i for i, h in enumerate(headers)}

    dataset = []
    for row in rows[1:]:
        q = row[col["question"]] if "question" in col else None
        a = row[col["answer"]]   if "answer"   in col else None
        if not q or not a:
            continue
        dataset.append({
            "id":          row[col["qa_id"]]    if "qa_id"          in col else None,
            "program":     row[col["program"]]  if "program"        in col else "GENERAL",
            "category":    row[col["category"]] if "category"       in col else "unknown",
            "term_no":     row[col["term_no"]]  if "term_no"        in col else None,
            "question":    str(q).strip(),
            "answer":      str(a).strip(),
            "keywords":    str(row[col["keywords"]]         if "keywords"         in col else "") or "",
            "source_file": str(row[col["source_doc_title"]] if "source_doc_title" in col else "") or "",
        })
    return dataset


def sample_test_cases(
    dataset: list[dict],
    n: int = 60,
    seed: int = 42,
) -> list[dict]:
    """
    Stratified sample: pick proportionally from each category so the
    baseline evaluation covers prerequisite, corequisite, policy, planning,
    OJT, grading, etc.  Falls back to the full dataset if n >= len(dataset).
    """
    if n >= len(dataset):
        return dataset

    rng = random.Random(seed)

    # Group by category
    by_cat: dict[str, list[dict]] = {}
    for item in dataset:
        by_cat.setdefault(item["category"], []).append(item)

    sampled = []
    total = len(dataset)

    for cat, items in by_cat.items():
        k = max(1, round(len(items) / total * n))
        k = min(k, len(items))
        sampled.extend(rng.sample(items, k))

    # If rounding left us short/over, adjust
    if len(sampled) < n:
        remaining = [d for d in dataset if d not in sampled]
        sampled.extend(rng.sample(remaining, min(n - len(sampled), len(remaining))))
    elif len(sampled) > n:
        sampled = rng.sample(sampled, n)

    rng.shuffle(sampled)
    return sampled


# ═══════════════════════════════════════════════════════════════════════════
# 2. MODELS TO TEST
# ═══════════════════════════════════════════════════════════════════════════

MODELS_TO_TEST = [
    # — Hugging Face (needs HF_TOKEN) —
    {
        "provider": "hf",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "label":    "Llama-3.1-8B",
    },
    {
        "provider": "hf",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "label":    "DeepSeek-R1-8B",
    },
    {
        "provider": "hf",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "label":    "Qwen2.5-7B",
    },
    {
        "provider":    "hf",
        "model_id":    "google/gemma-2-9b-it:featherless-ai",
        "label":       "Gemma-2-9b-it",
    },
    # — Google Gemini (needs GEMINI_API_KEY) —
    {
        "provider": "gemini",
        "model_id": "gemini-2.5-flash-lite",
        "label":    "Gemini-2.5-Flash-Lite",
    },
    # — OpenAI (needs OPENAI_API_KEY) —
    {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "label":    "GPT-4o-mini",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# 3. PARAMETER CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

PARAM_CONFIGS = [
    # ── Temperature × Token-limit grid (3 × 2 = 6 combos) ──
    # Temperature controls randomness: 0.0 = greedy/deterministic,
    # 0.1 = near-deterministic, 0.3 = mild creativity.
    # Token limit controls max response length.
    {"temperature": 0.0, "max_tokens": 200, "top_p": 1.0, "label": "t=0.0 tok=200"},
    {"temperature": 0.0, "max_tokens": 400, "top_p": 1.0, "label": "t=0.0 tok=400"},
    {"temperature": 0.1, "max_tokens": 200, "top_p": 1.0, "label": "t=0.1 tok=200"},
    {"temperature": 0.1, "max_tokens": 400, "top_p": 1.0, "label": "t=0.1 tok=400"},
    {"temperature": 0.3, "max_tokens": 200, "top_p": 1.0, "label": "t=0.3 tok=200"},
    {"temperature": 0.3, "max_tokens": 400, "top_p": 1.0, "label": "t=0.3 tok=400"},
    # ── Top-p (nucleus sampling) variations ──
    # top_p restricts the token pool to the smallest set whose cumulative
    # probability ≥ p.  Lower top_p = more focused outputs.
    # Tested at t=0.1 (our best near-deterministic temp) with both token limits.
    {"temperature": 0.1, "max_tokens": 200, "top_p": 0.9, "label": "t=0.1 tok=200 p=0.9"},
    {"temperature": 0.1, "max_tokens": 400, "top_p": 0.9, "label": "t=0.1 tok=400 p=0.9"},
]


# ═══════════════════════════════════════════════════════════════════════════
# 4. GENERATION — No retrieval; pure parametric knowledge
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an academic adviser for De La Salle University (DLSU) "
    "Computer Engineering (CpE) and Electronics Engineering (ECE) students. "
    "Answer questions about the curriculum, prerequisites, co-requisites, "
    "academic policies, and student handbook rules as accurately as you can "
    "based on your training knowledge. "
    "Keep answers concise and factual. "
    "If you are not sure, say: "
    "'I don't have that information — please consult your adviser.'"
)


def _call_with_retry(fn, max_retries=5):
    """Wrapper that handles HF 402/429 and Gemini 503 by waiting and retrying."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            if "402" in err:
                wait = 310  # 5 min + 10s buffer
                print(f"\n  [HF 402] Budget depleted — waiting {wait}s for reset "
                      f"(attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            elif "429" in err:
                wait = 60
                print(f"\n  [429] Rate limited — waiting {wait}s...")
                time.sleep(wait)
            elif "503" in err or "UNAVAILABLE" in err:
                wait = 30
                print(f"\n  [503] Service unavailable — waiting {wait}s "
                      f"(attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def generate(client: dict, model_id: str,
             question: str, config: dict) -> str:
    """Send the question directly to the LLM — no retrieved context."""

    top_p = config.get("top_p", 1.0)

    if client["provider"] == "hf":
        def _hf_call():
            resp = client["instance"].chat_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": question},
                ],
                max_tokens=config["max_tokens"],
                temperature=max(config["temperature"], 1e-7),
                top_p=top_p,
            )
            content = resp.choices[0].message.content
            if content is None:
                content = getattr(resp.choices[0].message, 'reasoning_content', None)
            return (content or "").strip()
        answer = _call_with_retry(_hf_call)

    elif client["provider"] == "gemini":
        from google import genai as google_genai
        from google.genai import types

        def _gemini_call():
            gemini_client = google_genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )
            resp = gemini_client.models.generate_content(
                model=model_id,
                contents=f"{SYSTEM_PROMPT}\n\n{question}",
                config=types.GenerateContentConfig(
                    max_output_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    top_p=top_p,
                ),
            )
            return resp.text.strip()
        answer = _call_with_retry(_gemini_call)

    elif client["provider"] == "openai":
        response = client["instance"].chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            top_p=top_p,
        )
        answer = response.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown provider: {client['provider']}")

    # Strip reasoning-model thinking blocks (e.g. DeepSeek)
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    # Handle empty responses (DeepSeek sometimes exhausts tokens on thinking)
    if not answer:
        answer = "[NO RESPONSE — model returned empty output]"

    return answer


# ═══════════════════════════════════════════════════════════════════════════
# 5. SCORING — BLEU, METEOR, ROUGE-1, ROUGE-L, BERTScore
# ═══════════════════════════════════════════════════════════════════════════

_rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method4


def score_response(ground_truth: str, answer: str) -> dict:
    """Compute ROUGE, BLEU, METEOR (fast, per-question). BERTScore is batched separately."""
    r = _rouge.score(ground_truth, answer)

    ref = [ground_truth.lower().split()]
    hyp = answer.lower().split()
    bleu = sentence_bleu(ref, hyp, smoothing_function=_smooth)
    met  = meteor_score([ground_truth.lower().split()], answer.lower().split())

    return {
        "rouge1":     round(r["rouge1"].fmeasure, 4),
        "rouge_l":    round(r["rougeL"].fmeasure, 4),
        "bleu":       round(bleu, 4),
        "meteor":     round(met, 4),
    }


def batch_bert_score(answers: list[str], references: list[str]) -> list[float]:
    """Compute BERTScore for all answers at once — loads model only once."""
    P, R, F1 = bert_score_fn(
        answers, references,
        lang="en",
        model_type="roberta-large",
        verbose=True,
        batch_size=32,
    )
    return [round(f.item(), 4) for f in F1]


# ═══════════════════════════════════════════════════════════════════════════
# 6. HALLUCINATION DETECTION — multi-condition framework
# ═══════════════════════════════════════════════════════════════════════════
#
# Five conditions evaluated per answer:
#   C1  RETRIEVAL_MISS       — model refuses when ground truth is substantive
#   C2  CONTRADICTION        — polarity flip (GT negates, answer affirms)
#   C3  WRONG_CODES          — fabricated course codes not in ground truth
#   C4  NUMERIC_FABRICATION  — invented numbers (units, GPA, %, years) not in GT
#   C5  HIGH_CLAIM_RATE      — ≥60% of answer sentences are unsupported vs GT
#
# Outputs per question:
#   hallucination (bool), halluc_reason (str),
#   halluc_claim_rate (float)    — C5 unsupported-sentence ratio [0..1]
#   factual_consistency (float)  — 1 − halluc_claim_rate  (thesis metric)
#   completeness_score (float)   — required entities covered in answer [0..1]
#   align_score (float|None)     — NLI entailment score vs context (RAG only; None here)
# ═══════════════════════════════════════════════════════════════════════════

_NOISE_CODES = {
    "I", "H", "S", "C", "OK", "ONLY", "DLSU", "NO", "YES",
    "BOTH", "TERM", "AY", "NOTE", "GPA", "CUM", "AND", "OR",
    "NOT", "ALL", "IF", "THE", "FOR", "SAS", "OJT", "PE",
    "GE", "CPE", "ECE", "IT", "CS", "LET",
}

_NO_INFO_PHRASES = [
    "don't have that information", "cannot find", "no information",
    "i'm not sure", "i do not have", "consult your adviser",
    "unable to provide", "not available in my",
    "not in the context", "context does not",
]

_ROUGE_SENT = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_CLAIM_THRESHOLD = 0.15   # ROUGE-L below this → sentence is unsupported


def _claim_level_rate(ground_truth: str, answer: str) -> float:
    """
    Sentence-level unsupported-claim ratio.
    Implements: Hallucination Rate = (unsupported claims / total claims).
    A generated sentence is 'unsupported' if its max ROUGE-L against any
    ground-truth sentence is below _CLAIM_THRESHOLD.
    """
    gt_sents  = [s.strip() for s in sent_tokenize(ground_truth) if s.strip()]
    ans_sents = [s.strip() for s in sent_tokenize(answer)       if s.strip()]
    if not ans_sents:
        return 0.0
    unsupported = 0
    for asent in ans_sents:
        best = max(
            (_ROUGE_SENT.score(gs, asent)["rougeL"].fmeasure for gs in gt_sents),
            default=0.0,
        )
        if best < _CLAIM_THRESHOLD:
            unsupported += 1
    return round(unsupported / len(ans_sents), 4)


def _numeric_fabrication(ground_truth: str, answer: str) -> set[str]:
    """Numbers present in answer but absent from ground truth (≥2 significant digits)."""
    pat = r"\b\d+(?:\.\d+)?\b"
    gt_nums  = set(re.findall(pat, ground_truth))
    ans_nums = set(re.findall(pat, answer))
    return {n for n in (ans_nums - gt_nums) if len(n.replace(".", "")) >= 2}


def _completeness_score(ground_truth: str, answer: str) -> float:
    """
    Fraction of required elements (course codes + key numbers from GT) that
    appear in the answer. Returns 1.0 when GT has no extractable elements.
    """
    gt_codes = set(re.findall(r"\b[A-Z]{3,8}\d*[A-Z]?\b", ground_truth)) - _NOISE_CODES
    gt_nums  = {n for n in re.findall(r"\b\d+(?:\.\d+)?\b", ground_truth)
                if len(n.replace(".", "")) >= 2}
    required = gt_codes | gt_nums
    if not required:
        return 1.0
    found = sum(1 for e in required if e in answer)
    return round(found / len(required), 4)


def _nli_align_score(nli_scorer, context: str, answer: str) -> float:
    """
    Score how well the answer is grounded in the retrieved context using NLI.
    Uses cross-encoder/nli-deberta-v3-small via sentence-transformers.
    Returns entailment probability [0..1].
    Truncates context to first 400 words to stay within model limits.
    """
    ctx_truncated = " ".join(context.split()[:400])
    result = nli_scorer.predict([(ctx_truncated, answer)])
    # cross-encoder NLI label order: contradiction=0, entailment=1, neutral=2
    import torch, torch.nn.functional as F
    scores = F.softmax(torch.tensor(result), dim=-1)
    entailment_prob = float(scores[0][1])
    return round(entailment_prob, 4)


def detect_hallucination(
    ground_truth: str,
    answer: str,
    context: str | None = None,
) -> dict:
    """
    Run all hallucination conditions. Returns a metrics dict — no longer a
    bare (bool, str) tuple so all downstream code should unpack via key access.
    """
    gt  = ground_truth.lower()
    ans = answer.lower()
    fired = "OK"

    # C1: Retrieval miss
    if any(p in ans for p in _NO_INFO_PHRASES) and len(ground_truth) > 30:
        fired = "RETRIEVAL_MISS"

    # C2: Contradiction
    if fired == "OK":
        gt_neg  = any(w in gt  for w in ["no.", "cannot", "must not", "not allowed", "invalid"])
        ans_pos = any(w in ans for w in ["yes,", "yes.", "you can", "is allowed", "is permitted"])
        ans_agr = any(w in ans for w in ["no,", "no.", "cannot", "must not", "not allowed"])
        if gt_neg and ans_pos and not ans_agr:
            fired = "CONTRADICTION"

    # C3: Wrong course codes
    if fired == "OK":
        gt_codes  = set(re.findall(r"\b[A-Z]{3,8}\d*[A-Z]?\b", ground_truth))
        ans_codes = set(re.findall(r"\b[A-Z]{3,8}\d*[A-Z]?\b", answer))
        wrong = ans_codes - gt_codes - _NOISE_CODES
        if wrong:
            fired = f"WRONG_CODES:{wrong}"

    # C4: Numeric fabrication
    if fired == "OK":
        fab = _numeric_fabrication(ground_truth, answer)
        if fab:
            fired = f"NUMERIC_FABRICATION:{fab}"

    # C5: Claim-level rate (always computed; also triggers flag if majority unsupported)
    claim_rate = _claim_level_rate(ground_truth, answer)
    if fired == "OK" and claim_rate >= 0.60:
        fired = f"HIGH_CLAIM_RATE:{claim_rate:.2f}"

    # NLI grounding check — only when retrieved context provided (RAG scripts)
    align_sc = None
    if context and _NLI_SCORER is not None:
        try:
            align_sc = _nli_align_score(_NLI_SCORER, context, answer)
        except Exception:
            align_sc = None

    return {
        "hallucination":       fired != "OK",
        "halluc_reason":       fired,
        "halluc_claim_rate":   claim_rate,
        "factual_consistency": round(1.0 - claim_rate, 4),
        "completeness_score":  _completeness_score(ground_truth, answer),
        "align_score":         align_sc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model_label: str,
    model_id: str,
    config: dict,
    client: dict,
    test_cases: list[dict],
) -> dict:
    """Run all test cases through one model+config combination."""

    print(f"\n  --- Config: {config['label']} ---")

    all_scores         = []
    generation_times   = []
    total_times        = []
    halluc_flags       = []
    all_halluc_details = []
    results_log        = []

    # Collect for batched BERTScore
    all_answers    = []
    all_references = []

    for i, tc in enumerate(test_cases):
        t_start = time.time()

        t_gen   = time.time()
        answer  = generate(client, model_id, tc["question"], config)
        gen_time = time.time() - t_gen

        total_time = time.time() - t_start

        scores = score_response(tc["answer"], answer)
        halluc = detect_hallucination(tc["answer"], answer, context=None)

        all_scores.append(scores)
        generation_times.append(gen_time)
        total_times.append(total_time)
        halluc_flags.append(halluc["hallucination"])
        all_halluc_details.append(halluc)
        all_answers.append(answer)
        all_references.append(tc["answer"])

        h_tag = "[HALLUC]" if halluc["hallucination"] else "[OK]"
        t_tag = "[OK]" if total_time < 5 else "[OVER 5s]"

        print(f"  Q{i+1:03d}: {tc['question'][:60]}")
        print(f"        Ans: {answer[:100]}...")
        print(f"        R1:{scores['rouge1']:.3f} RL:{scores['rouge_l']:.3f} "
              f"BL:{scores['bleu']:.3f} MT:{scores['meteor']:.3f} | "
              f"{gen_time:.2f}s {t_tag} | {h_tag} {halluc['halluc_reason']} "
              f"ClaimRt:{halluc['halluc_claim_rate']:.2f} "
              f"Compl:{halluc['completeness_score']:.2f}")

        results_log.append({
            "id":              tc["id"],
            "category":        tc["category"],
            "program":         tc["program"],
            "question":        tc["question"],
            "ground_truth":    tc["answer"],
            "generated":       answer,
            **scores,
            **halluc,
            "generation_time": round(gen_time, 3),
            "total_time":      round(total_time, 3),
        })

    # Batch BERTScore — loads roberta-large ONCE for all 593 answers
    print(f"  Computing BERTScore for {len(all_answers)} answers (batched)...")
    bert_scores = batch_bert_score(all_answers, all_references)
    for i, bs in enumerate(bert_scores):
        all_scores[i]["bert_score"] = bs
        results_log[i]["bert_score"] = bs

    n            = len(test_cases)
    halluc_count = sum(halluc_flags)
    def avg(key): return round(sum(s[key] for s in all_scores) / n, 4)
    def avgh(key): return round(sum(h[key] for h in all_halluc_details) / n, 4)

    summary = {
        "model":                   model_label,
        "provider":                client["provider"],
        "config":                  config["label"],
        "n_questions":             n,
        "avg_rouge1":              avg("rouge1"),
        "avg_rouge_l":             avg("rouge_l"),
        "avg_bleu":                avg("bleu"),
        "avg_meteor":              avg("meteor"),
        "avg_bert_score":          avg("bert_score"),
        "hallucination_count":     halluc_count,
        "hallucination_rate":      round(halluc_count / n * 100, 1),
        "avg_halluc_claim_rate":   avgh("halluc_claim_rate"),
        "avg_factual_consistency": avgh("factual_consistency"),
        "avg_completeness_score":  avgh("completeness_score"),
        "avg_align_score":         None,   # no context in standard baseline
        "avg_generation_time":     round(sum(generation_times) / n, 3),
        "avg_total_time":          round(sum(total_times) / n, 3),
        "pct_under_5s":            round(sum(1 for t in total_times if t < 5) / n * 100, 1),
        "detail":                  results_log,
    }

    print(f"  SUMMARY [{config['label']}]")
    print(f"  SO3 | R1:{summary['avg_rouge1']:.3f} RL:{summary['avg_rouge_l']:.3f} "
          f"BLEU:{summary['avg_bleu']:.3f} METEOR:{summary['avg_meteor']:.3f} "
          f"BERTScore:{summary['avg_bert_score']:.3f}")
    print(f"  SO3 | Hallucinations:{halluc_count}/{n} ({summary['hallucination_rate']:.0f}%) | "
          f"ClaimRate:{summary['avg_halluc_claim_rate']:.3f} | "
          f"Consistency:{summary['avg_factual_consistency']:.3f} | "
          f"Completeness:{summary['avg_completeness_score']:.3f}")
    print(f"  TIME | Avg:{summary['avg_total_time']:.2f}s | <5s:{summary['pct_under_5s']:.0f}%")

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # --- Load & sample test data ---
    print("Loading advising dataset...")
    full_dataset = load_dataset(DATASET_PATH)
    print(f"  Total Q&A pairs: {len(full_dataset)}")

    # Use the full dataset — all 593 Q&A pairs.
    # No sampling: every category gets full representation, which makes
    # per-category breakdowns statistically meaningful and eliminates
    # reviewer concerns about sampling bias.
    test_cases = full_dataset
    random.Random(42).shuffle(test_cases)  # shuffle for consistent ordering
    print(f"  Using full dataset: {len(test_cases)} questions")

    cats = {}
    for tc in test_cases:
        cats[tc["category"]] = cats.get(tc["category"], 0) + 1
    print(f"  Categories covered: {len(cats)}")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    print()

    # --- Initialize API clients ---
    print("Checking API keys...")
    print(f"  HF_TOKEN:       {'SET' if os.getenv('HF_TOKEN')       else 'NOT SET'}")
    print(f"  GEMINI_API_KEY:  {'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")
    print(f"  OPENAI_API_KEY:  {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print()

    clients = {}

    if os.getenv("HF_TOKEN"):
        clients["hf"] = {
            "provider": "hf",
            "instance": InferenceClient(token=os.getenv("HF_TOKEN")),
        }
        # Featherless-AI client for models that need it (e.g. Gemma)
        clients["hf_featherless"] = {
            "provider": "hf",
            "instance": InferenceClient(
                provider="featherless-ai",
                api_key=os.getenv("HF_TOKEN"),
            ),
        }

    if os.getenv("GEMINI_API_KEY"):
        try:
            from google import genai as google_genai
            clients["gemini"] = {"provider": "gemini", "instance": None}
            print("  Gemini client initialized")
        except ImportError:
            print("  [SKIP] Gemini — pip install google-genai")

    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            clients["openai"] = {
                "provider": "openai",
                "instance": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
            }
            print("  OpenAI client initialized")
        except ImportError:
            print("  [SKIP] OpenAI — pip install openai")

    # --- Run evaluations ---
    all_summaries = []

    for model_cfg in MODELS_TO_TEST:
        provider = model_cfg["provider"]
        model_id = model_cfg["model_id"]
        label    = model_cfg["label"]
        hf_prov  = model_cfg.get("hf_provider")

        # Pick the right client — featherless for Gemma, default for others
        if hf_prov and f"hf_{hf_prov.replace('-','_')}" in clients:
            client_key = f"hf_{hf_prov.replace('-','_')}"
        elif provider in clients:
            client_key = provider
        else:
            print(f"\n[SKIP] {label} — {provider} client not available")
            continue

        print(f"\n{'='*70}")
        print(f"MODEL: {label}  |  Provider: {provider}"
              f"{f' ({hf_prov})' if hf_prov else ''}")
        print(f"{'='*70}")

        for param_cfg in PARAM_CONFIGS:
            try:
                summary = run_evaluation(
                    label, model_id, param_cfg, clients[client_key], test_cases,
                )
                all_summaries.append(summary)
            except Exception as e:
                print(f"  [SKIPPED] {param_cfg['label']} — {e}\n")
                continue

    # --- Final comparison ---
    if all_summaries:
        print(f"\n{'='*140}")
        print("NO-RAG BASELINE — FINAL COMPARISON")
        print(f"{'='*140}")
        print(f"{'Model + Config':<42} {'R-1':>6} {'R-L':>6} {'BLEU':>6} "
              f"{'METEOR':>7} {'BERT':>7} {'Halluc':>8} {'ClmRt':>6} "
              f"{'Consist':>8} {'Compl':>6} {'AvgTime':>8} {'<5s':>5}")
        print("-" * 140)

        for s in sorted(all_summaries, key=lambda x: -x["avg_rouge_l"]):
            name = f"{s['model']} [{s['config']}]"
            print(
                f"{name:<42} "
                f"{s['avg_rouge1']:>6.3f} "
                f"{s['avg_rouge_l']:>6.3f} "
                f"{s['avg_bleu']:>6.3f} "
                f"{s['avg_meteor']:>7.3f} "
                f"{s['avg_bert_score']:>7.3f} "
                f"{s['hallucination_count']:>4}/{s['n_questions']} "
                f"{s['avg_halluc_claim_rate']:>6.3f} "
                f"{s['avg_factual_consistency']:>8.3f} "
                f"{s['avg_completeness_score']:>6.3f} "
                f"{s['avg_total_time']:>7.2f}s "
                f"{s['pct_under_5s']:>4.0f}%"
            )

        # Save results
        out_path = "no_rag_baseline_results_gemma.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")
    else:
        print("\nNo results — check your API keys and model configurations.")
