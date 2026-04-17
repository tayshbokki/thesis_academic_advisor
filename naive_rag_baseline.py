"""
naive_rag_baseline.py — Naive RAG Evaluation (Dense Retrieval Only)
====================================================================
Tests LLMs with simple semantic retrieval from the ChromaDB vector store
built by chunking_pipeline.py.  This is the "Naive RAG" system described
in the thesis: embed query → cosine search → top-k context → generate.

No BM25, no reranking, no hybrid fusion — just pure dense retrieval with
e5-small-v2 embeddings (as justified by embedding_experiment.py).

Test cases: all 593 Q&A pairs from advising_dataset.xlsx
SO1 metrics: Precision@K, Recall@K, MRR, NDCG@10, cosine similarity
SO3 metrics: ROUGE-1, ROUGE-L, BLEU, METEOR, BERTScore
Also:       hallucination detection, retrieval time, response-time compliance

Usage:
    pip install openpyxl nltk rouge-score bert-score huggingface-hub chromadb
    # First: run chunking_pipeline.py to build the ChromaDB collection
    python naive_rag_baseline.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

import os, re, json, time, random, math
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

# ── NLI context grounding scorer — loaded once at startup ───────────────────
# Uses cross-encoder/nli-deberta-v3-small (sentence-transformers) to score
# how well the generated answer is entailed by the retrieved context.
# No torch version conflicts — works with your existing cu124 install.
# Install: pip install sentence-transformers  (likely already present)
_NLI_SCORER = None
try:
    from sentence_transformers.cross_encoder import CrossEncoder
    print("[NLI Scorer] Loading cross-encoder/nli-deberta-v3-small...")
    _NLI_SCORER = CrossEncoder(
        "cross-encoder/nli-deberta-v3-small",
        device="cuda",
    )
    print("[NLI Scorer] Ready.")
except Exception as _ne:
    print(f"[NLI Scorer] Not available ({_ne}). align_score will be None.\n"
          f"  Install with: pip install sentence-transformers")


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD TEST CASES FROM advising_dataset.xlsx
# ═══════════════════════════════════════════════════════════════════════════

DATASET_PATH = "dataset_test.xlsx"  # held-out 20% split — run dataset_split.py first


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


# ═══════════════════════════════════════════════════════════════════════════
# 1b. SO1 — RELEVANCE RESOLUTION & RETRIEVAL METRICS
# ═══════════════════════════════════════════════════════════════════════════
# Reuses the same relevance judgment logic from embedding_experiment.py.
# Each query maps to a set of "relevant IDs" (course codes or doc_type tags)
# that should appear in the retrieved chunk IDs.

SOURCE_TO_DOC_TYPE = {
    "OJT": "ojt_policy", "Practicum": "ojt_policy", "GCOE UG": "ojt_policy",
    "advising": "advising_guidelines", "Advising": "advising_guidelines",
    "Best Practices": "advising_best_practices",
    "Thesis": "thesis_policies", "Thesis Policies": "thesis_policies",
    "retention": "retention_policy", "Retention": "retention_policy",
    "load": "load_policy", "overload": "load_policy",
    "lab": "lab_lecture_policy", "lecture": "lab_lecture_policy",
    "crediting": "crediting_process",
}

CHECKLIST_CATS = {
    "prerequisite", "corequisite", "prerequisite_lookup", "eligibility_check",
    "ambiguous_counterpart", "term_plan", "curriculum_overview", "curriculum_summary",
    "curriculum_rule", "course_completion", "checklist_rule", "planning_guidance",
    "program_info",
}
POLICY_CATS = {
    "ojt_policy", "enrollment_policy", "grading_policy", "attendance_policy",
    "withdrawal_policy", "course_credit", "edge_case", "student_query_variant",
    "discipline_policy", "student_policy",  # added in dataset-query.xlsx
}


def resolve_relevant_ids(tc: dict) -> list[str]:
    """Map a test case to its expected relevant chunk IDs."""
    cat = tc["category"]
    kw  = tc.get("keywords", "")
    src = tc.get("source_file", "")

    if cat in CHECKLIST_CATS:
        codes = [k.strip() for k in kw.split(",") if k.strip()]
        valid = [c for c in codes if re.match(r"^[A-Z]{2,8}\d*[A-Z]?$", c)]
        return valid[:3] if valid else []

    if cat in POLICY_CATS:
        for key, dt in SOURCE_TO_DOC_TYPE.items():
            if key in src:
                return [dt]
        if "Handbook" in src or "handbook" in src:
            return ["load_policy", "retention_policy", "lab_lecture_policy"]
        return []

    # Fallback: try course codes + source mapping
    relevant = []
    codes = [k.strip() for k in kw.split(",") if k.strip()]
    valid = [c for c in codes if re.match(r"^[A-Z]{2,8}\d*[A-Z]?$", c)]
    relevant.extend(valid[:2])
    for key, dt in SOURCE_TO_DOC_TYPE.items():
        if key in src and dt not in relevant:
            relevant.append(dt)
            break
    return relevant


def compute_mrr(ranked_ids: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def compute_precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in relevant) / k


def compute_recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = ranked_ids[:k]
    return sum(1 for d in top_k if d in relevant) / len(relevant)


def compute_ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, d in enumerate(ranked_ids[:k], 1)
        if d in relevant
    )
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return min(dcg / idcg, 1.0) if idcg else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. CHROMADB — Connect to 3 collections built by chunking_pipeline.py
# ═══════════════════════════════════════════════════════════════════════════

CHROMA_BASE_DIR  = "./chroma_store"
EMBEDDING_MODEL  = "intfloat/e5-small-v2"
E5_QUERY_PREFIX  = "query: "

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embedding model...")
_embedding_fn = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)


def connect_vector_stores() -> dict:
    """
    Connect to all 3 ChromaDB collections built by chunking_pipeline.py:
      checklist (course rows), policies (handbook sections), faqs (Q&A pairs)
    """
    stores = {}
    for name in ["checklist", "policies", "faqs"]:
        path = f"{CHROMA_BASE_DIR}/{name}"
        stores[name] = Chroma(
            collection_name=name,
            embedding_function=_embedding_fn,
            persist_directory=path,
        )
        count = stores[name]._collection.count()
        print(f"  [{name}] {count} chunks loaded")
    total = sum(s._collection.count() for s in stores.values())
    print(f"  Total: {total} chunks across 3 collections")
    return stores


def _extract_doc_id(metadata: dict, collection_name: str) -> str:
    """Build a consistent doc_id from metadata for relevance matching."""
    if collection_name == "checklist":
        return metadata.get("course_code", "unknown")
    elif collection_name == "policies":
        return metadata.get("doc_type", "policy")
    else:
        cat = metadata.get("category", "General")
        ci  = metadata.get("chunk_index", 0)
        return f"faq_{cat}_{ci}"


def retrieve(stores: dict, query: str, top_k: int = 5) -> dict:
    """
    Pure dense retrieval across all 3 collections — cosine similarity only.
    e5 models require 'query: ' prefix for query embeddings.
    Returns dict with context text, retrieved IDs, and distances.
    """
    prefixed = f"{E5_QUERY_PREFIX}{query}"
    all_results = []

    for name, store in stores.items():
        try:
            hits = store.similarity_search_with_score(prefixed, k=top_k)
            for doc, dist in hits:
                doc_id = _extract_doc_id(doc.metadata, name)
                all_results.append((doc_id, doc.page_content, dist, doc.metadata))
        except Exception as e:
            print(f"  [WARN] {name} search failed: {e}")

    # Sort by distance (lower = more similar for L2)
    all_results.sort(key=lambda x: x[2])

    # Take top_k overall
    top = all_results[:top_k]

    return {
        "context":   "\n\n".join(r[1] for r in top),
        "ids":       [r[0] for r in top],
        "distances": [r[2] for r in top],
        "metadatas": [r[3] for r in top],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. MODELS TO TEST
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
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "label":    "Qwen2.5-7B",
    },
    {
        "provider":    "hf",
        "model_id":    "google/gemma-2-9b-it:featherless-ai",
        "label":       "Gemma-2-9B",
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
# 4. PARAMETER CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

PARAM_CONFIGS = [
    # ── Temperature × Token-limit grid (3 × 2 = 6 combos) ──
    # All use top_k=5 (retrieval) and top_p=1.0 (generation) as defaults.
    {"temperature": 0.0, "max_tokens": 200, "top_p": 1.0, "top_k": 5, "label": "t=0.0 tok=200"},
    {"temperature": 0.0, "max_tokens": 400, "top_p": 1.0, "top_k": 5, "label": "t=0.0 tok=400"},
    {"temperature": 0.1, "max_tokens": 200, "top_p": 1.0, "top_k": 5, "label": "t=0.1 tok=200"},
    {"temperature": 0.1, "max_tokens": 400, "top_p": 1.0, "top_k": 5, "label": "t=0.1 tok=400"},
    {"temperature": 0.3, "max_tokens": 200, "top_p": 1.0, "top_k": 5, "label": "t=0.3 tok=200"},
    {"temperature": 0.3, "max_tokens": 400, "top_p": 1.0, "top_k": 5, "label": "t=0.3 tok=400"},
    # ── Top-p (nucleus sampling) variations ──
    {"temperature": 0.1, "max_tokens": 200, "top_p": 0.9, "top_k": 5, "label": "t=0.1 tok=200 p=0.9"},
    {"temperature": 0.1, "max_tokens": 400, "top_p": 0.9, "top_k": 5, "label": "t=0.1 tok=400 p=0.9"},
    # ── Retrieval top_k variations ──
    # top_k=3: fewer chunks = less noise but may miss relevant context.
    # top_k=5: default — balanced.
    # top_k=10: more chunks = better recall but risks diluting with irrelevant context.
    # Tested at best-performing generation params (t=0.1, tok=200, p=1.0).
    {"temperature": 0.1, "max_tokens": 200, "top_p": 1.0, "top_k": 3,  "label": "t=0.1 tok=200 k=3"},
    {"temperature": 0.1, "max_tokens": 200, "top_p": 1.0, "top_k": 10, "label": "t=0.1 tok=200 k=10"},
]


# ═══════════════════════════════════════════════════════════════════════════
# 5. GENERATION — RAG: retrieved context + question → LLM
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an academic adviser for De La Salle University (DLSU) "
    "Computer Engineering (CpE) and Electronics Engineering (ECE) students. "
    "Use ONLY the provided context to answer questions about the curriculum, "
    "prerequisites, co-requisites, academic policies, and student handbook rules. "
    "Keep answers concise and factual. "
    "If the answer is not in the context, say: "
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
                wait = 310
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


def generate(client: dict, model_id: str, context: str,
             question: str, config: dict) -> str:
    """Send retrieved context + question to the LLM."""

    user_msg = f"Context:\n{context}\n\nQuestion: {question}"
    top_p = config.get("top_p", 1.0)

    if client["provider"] == "hf":
        def _hf_call():
            resp = client["instance"].chat_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
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
                contents=f"{SYSTEM_PROMPT}\n\n{user_msg}",
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
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            top_p=top_p,
        )
        answer = response.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown provider: {client['provider']}")

    # Strip reasoning-model thinking blocks
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    # Handle empty responses (DeepSeek sometimes exhausts tokens on thinking)
    if not answer:
        answer = "[NO RESPONSE — model returned empty output]"

    return answer


# ═══════════════════════════════════════════════════════════════════════════
# 6. SCORING — BLEU, METEOR, ROUGE-1, ROUGE-L, BERTScore
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
# 7. HALLUCINATION DETECTION — multi-condition framework
# ═══════════════════════════════════════════════════════════════════════════
#
# Five conditions evaluated per answer:
#   C1  RETRIEVAL_MISS       — model refuses when ground truth is substantive
#   C2  CONTRADICTION        — polarity flip (GT negates, answer affirms)
#   C3  WRONG_CODES          — fabricated course codes not in ground truth
#   C4  NUMERIC_FABRICATION  — invented numbers (units, GPA, %, years) not in GT
#   C5  HIGH_CLAIM_RATE      — ≥60% of answer sentences are unsupported vs GT
#
# RAG-specific grounding check:
#   NLI grounding score — answer faithfulness to retrieved context
#
# Outputs per question:
#   hallucination (bool), halluc_reason (str),
#   halluc_claim_rate (float)    — C5 unsupported-sentence ratio [0..1]
#   factual_consistency (float)  — 1 − halluc_claim_rate  (thesis metric)
#   completeness_score (float)   — required entities covered in answer [0..1]
#   align_score (float|None)     — NLI entailment score vs context [0..1]
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
    "not in the context", "not mentioned in the context", "context does not",
]

_ROUGE_SENT = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_CLAIM_THRESHOLD = 0.15


def _claim_level_rate(ground_truth: str, answer: str) -> float:
    """
    Sentence-level unsupported-claim ratio.
    Implements: Hallucination Rate = (unsupported claims / total claims).
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
    """Fraction of required elements (course codes + key numbers from GT) in answer."""
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
    Score answer grounding in retrieved context via NLI entailment probability.
    cross-encoder label order: contradiction=0, entailment=1, neutral=2
    Truncates context to first 400 words to stay within model limits.
    """
    ctx_truncated = " ".join(context.split()[:400])
    result = nli_scorer.predict([(ctx_truncated, answer)])
    import torch, torch.nn.functional as F
    scores = F.softmax(torch.tensor(result), dim=-1)
    return round(float(scores[0][1]), 4)


def detect_hallucination(
    ground_truth: str,
    answer: str,
    context: str | None = None,
) -> dict:
    """Run all hallucination conditions. Returns a metrics dict."""
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

    # C5: Claim-level rate
    claim_rate = _claim_level_rate(ground_truth, answer)
    if fired == "OK" and claim_rate >= 0.60:
        fired = f"HIGH_CLAIM_RATE:{claim_rate:.2f}"

    # NLI grounding check — scores answer entailment vs. retrieved context
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
# 8. EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model_label: str,
    model_id: str,
    config: dict,
    stores: dict,
    client: dict,
    test_cases: list[dict],
) -> dict:
    """Run all test cases through one model+config with naive RAG retrieval.
    Computes both SO1 (retrieval) and SO3 (generation) metrics."""

    print(f"\n  --- Config: {config['label']} ---")

    all_scores         = []
    retrieval_times    = []
    generation_times   = []
    total_times        = []
    halluc_flags       = []
    all_halluc_details = []
    results_log        = []

    # Collect for batched BERTScore
    all_answers    = []
    all_references = []

    # SO1 accumulators
    all_mrr      = []
    all_p_at_k   = {k: [] for k in [1, 3, 5, 10]}
    all_r_at_k   = {k: [] for k in [1, 3, 5, 10]}
    all_ndcg     = []
    all_cos_sim  = []
    so1_evaluated = 0

    for i, tc in enumerate(test_cases):
        t_start = time.time()

        # RETRIEVE — pure dense search (naive RAG)
        top_k    = config.get("top_k", 5)
        t_ret    = time.time()
        ret_result = retrieve(stores, tc["question"], top_k=max(top_k, 10))
        ret_time   = time.time() - t_ret

        # Context for generation uses only top_k chunks
        context = "\n\n".join(ret_result["ids"][:top_k]
                              and ret_result["context"].split("\n\n")[:top_k]) \
                  if top_k < 10 else ret_result["context"]
        # Simpler: just re-join the top_k documents
        all_docs = ret_result["context"].split("\n\n")
        context  = "\n\n".join(all_docs[:top_k])

        # --- SO1: Retrieval quality metrics ---
        relevant_ids = resolve_relevant_ids(tc)
        if relevant_ids:
            relevant_set = set(relevant_ids)
            ranked_ids   = ret_result["ids"]  # up to 10

            q_mrr = compute_mrr(ranked_ids, relevant_set)
            all_mrr.append(q_mrr)

            for k in [1, 3, 5, 10]:
                all_p_at_k[k].append(compute_precision_at_k(ranked_ids, relevant_set, k))
                all_r_at_k[k].append(compute_recall_at_k(ranked_ids, relevant_set, k))

            all_ndcg.append(compute_ndcg_at_k(ranked_ids, relevant_set, 10))

            # Cosine similarity: ChromaDB returns L2 distances by default;
            # convert to approximate cosine sim = 1 - (d^2 / 2) for normalized vecs
            if ret_result["distances"]:
                cos_sims = [1.0 - (d**2 / 2.0) for d in ret_result["distances"][:top_k]]
                all_cos_sim.append(sum(cos_sims) / len(cos_sims))

            so1_evaluated += 1

        # GENERATE — LLM with retrieved context
        t_gen    = time.time()
        answer   = generate(client, model_id, context, tc["question"], config)
        gen_time = time.time() - t_gen

        total_time = time.time() - t_start

        # --- SO3: Generation quality metrics ---
        scores = score_response(tc["answer"], answer)
        halluc = detect_hallucination(tc["answer"], answer, context=context)

        all_scores.append(scores)
        retrieval_times.append(ret_time)
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
              f"ret:{ret_time:.3f}s gen:{gen_time:.2f}s {t_tag} | "
              f"{h_tag} {halluc['halluc_reason']} "
              f"ClaimRt:{halluc['halluc_claim_rate']:.2f} "
              f"Align:{halluc['align_score'] if halluc['align_score'] is not None else 'N/A'}")

        results_log.append({
            "id":              tc["id"],
            "category":        tc["category"],
            "program":         tc["program"],
            "question":        tc["question"],
            "ground_truth":    tc["answer"],
            "generated":       answer,
            "retrieved_ids":   ret_result["ids"][:top_k],
            "relevant_ids":    relevant_ids,
            **scores,
            **halluc,
            "retrieval_time":  round(ret_time, 4),
            "generation_time": round(gen_time, 3),
            "total_time":      round(total_time, 3),
        })

    # Batch BERTScore — loads roberta-large ONCE for all answers
    print(f"  Computing BERTScore for {len(all_answers)} answers (batched)...")
    bert_scores = batch_bert_score(all_answers, all_references)
    for idx, bs in enumerate(bert_scores):
        all_scores[idx]["bert_score"] = bs
        results_log[idx]["bert_score"] = bs

    n            = len(test_cases)
    halluc_count = sum(halluc_flags)
    def avg(key): return round(sum(s[key] for s in all_scores) / n, 4)
    def avgh(key): return round(sum(h[key] for h in all_halluc_details) / n, 4)
    def avgh_opt(key):
        vals = [h[key] for h in all_halluc_details if h[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # SO1 aggregates
    so1_metrics = {}
    if so1_evaluated > 0:
        so1_metrics = {
            "so1_n_evaluated":  so1_evaluated,
            "so1_mean_mrr":     round(sum(all_mrr) / so1_evaluated, 4),
            "so1_mean_ndcg_10": round(sum(all_ndcg) / so1_evaluated, 4),
            "so1_mean_cosine":  round(sum(all_cos_sim) / len(all_cos_sim), 4) if all_cos_sim else None,
            "so1_precision_at_k": {
                k: round(sum(v) / so1_evaluated, 4) for k, v in all_p_at_k.items()
            },
            "so1_recall_at_k": {
                k: round(sum(v) / so1_evaluated, 4) for k, v in all_r_at_k.items()
            },
        }

    summary = {
        "model":                   model_label,
        "provider":                client["provider"],
        "config":                  config["label"],
        "n_questions":             n,
        "retrieval":               f"dense_only (e5-small-v2, cosine, top_k={config.get('top_k', 5)})",
        # SO3 — generation quality
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
        "avg_align_score":         avgh_opt("align_score"),
        # SO1 — retrieval quality
        **so1_metrics,
        # Timing
        "avg_retrieval_time":      round(sum(retrieval_times) / n, 4),
        "avg_generation_time":     round(sum(generation_times) / n, 3),
        "avg_total_time":          round(sum(total_times) / n, 3),
        "pct_under_5s":            round(sum(1 for t in total_times if t < 5) / n * 100, 1),
        "detail":                  results_log,
    }

    print(f"  SUMMARY [{config['label']}]")
    print(f"  SO3 | R1:{summary['avg_rouge1']:.3f} RL:{summary['avg_rouge_l']:.3f} "
          f"BLEU:{summary['avg_bleu']:.3f} METEOR:{summary['avg_meteor']:.3f} "
          f"BERTScore:{summary['avg_bert_score']:.3f}")
    align_str = f"{summary['avg_align_score']:.3f}" if summary["avg_align_score"] is not None else "N/A"
    print(f"  SO3 | Hallucinations:{halluc_count}/{n} ({summary['hallucination_rate']:.0f}%) | "
          f"ClaimRate:{summary['avg_halluc_claim_rate']:.3f} | "
          f"Consistency:{summary['avg_factual_consistency']:.3f} | "
          f"Completeness:{summary['avg_completeness_score']:.3f} | "
          f"NLI-Align:{align_str}")
    if so1_metrics:
        print(f"  SO1 | MRR:{so1_metrics['so1_mean_mrr']:.4f} "
              f"NDCG@10:{so1_metrics['so1_mean_ndcg_10']:.4f} "
              f"R@5:{so1_metrics['so1_recall_at_k'][5]:.4f} "
              f"R@10:{so1_metrics['so1_recall_at_k'][10]:.4f} "
              f"(n={so1_evaluated})")
    print(f"  TIME | Ret:{summary['avg_retrieval_time']:.4f}s "
          f"Tot:{summary['avg_total_time']:.2f}s <5s:{summary['pct_under_5s']:.0f}%")

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # --- Load test data (all 593 Q&A pairs) ---
    print("Loading advising dataset...")
    full_dataset = load_dataset(DATASET_PATH)
    print(f"  Total Q&A pairs: {len(full_dataset)}")

    test_cases = full_dataset
    random.Random(42).shuffle(test_cases)
    print(f"  Using full dataset: {len(test_cases)} questions")

    cats = {}
    for tc in test_cases:
        cats[tc["category"]] = cats.get(tc["category"], 0) + 1
    print(f"  Categories covered: {len(cats)}")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    print()

    # --- Connect to ChromaDB ---
    stores = connect_vector_stores()
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

        # Pick the right client
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
                    label, model_id, param_cfg,
                    stores, clients[client_key], test_cases,
                )
                all_summaries.append(summary)
            except Exception as e:
                print(f"  [SKIPPED] {param_cfg['label']} — {e}\n")
                continue

    # --- Final comparison ---
    if all_summaries:
        # SO3 table
        print(f"\n{'='*150}")
        print("NAIVE RAG — SO3 GENERATION QUALITY")
        print(f"{'='*150}")
        print(f"{'Model + Config':<42} {'R-1':>6} {'R-L':>6} {'BLEU':>6} "
              f"{'METEOR':>7} {'BERT':>7} {'Halluc':>8} {'ClmRt':>6} "
              f"{'Consist':>8} {'Compl':>6} {'Align':>6} {'AvgRet':>8} {'AvgTot':>8} {'<5s':>5}")
        print("-" * 150)

        for s in sorted(all_summaries, key=lambda x: -x["avg_rouge_l"]):
            name = f"{s['model']} [{s['config']}]"
            align_str = f"{s['avg_align_score']:>6.3f}" if s.get("avg_align_score") is not None else "   N/A"
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
                f"{align_str} "
                f"{s['avg_retrieval_time']:>7.4f}s "
                f"{s['avg_total_time']:>7.2f}s "
                f"{s['pct_under_5s']:>4.0f}%"
            )

        # SO1 table (retrieval metrics are the same across models for same config,
        # but printing per-run for completeness)
        so1_runs = [s for s in all_summaries if s.get("so1_mean_mrr") is not None]
        if so1_runs:
            print(f"\n{'='*110}")
            print("NAIVE RAG — SO1 RETRIEVAL QUALITY")
            print(f"{'='*110}")
            print(f"{'Model + Config':<42} {'MRR':>7} {'NDCG@10':>8} "
                  f"{'P@5':>6} {'R@5':>6} {'R@10':>6} {'CosSim':>7} {'n':>5}")
            print("-" * 110)

            # SO1 retrieval is model-independent (same query→same chunks),
            # so just print unique configs
            seen_configs = set()
            for s in so1_runs:
                cfg_key = s["config"]
                if cfg_key in seen_configs:
                    continue
                seen_configs.add(cfg_key)
                name = f"[{s['config']}]"
                cos_str = f"{s['so1_mean_cosine']:.4f}" if s.get("so1_mean_cosine") else "  N/A"
                print(
                    f"{name:<42} "
                    f"{s['so1_mean_mrr']:>7.4f} "
                    f"{s['so1_mean_ndcg_10']:>8.4f} "
                    f"{s['so1_precision_at_k'][5]:>6.4f} "
                    f"{s['so1_recall_at_k'][5]:>6.4f} "
                    f"{s['so1_recall_at_k'][10]:>6.4f} "
                    f"{cos_str:>7} "
                    f"{s['so1_n_evaluated']:>5}"
                )

            # SO1 target verification
            best = so1_runs[0]
            print(f"\n  SO1 Target Verification (config: {best['config']}):")
            mrr_ok  = best["so1_mean_mrr"] >= 0.8
            ndcg_ok = best["so1_mean_ndcg_10"] >= 0.8
            r10_ok  = best["so1_recall_at_k"][10] >= 0.75
            print(f"    MRR ≥ 0.80:      {'PASS' if mrr_ok else 'FAIL'} ({best['so1_mean_mrr']:.4f})")
            print(f"    NDCG@10 ≥ 0.80:  {'PASS' if ndcg_ok else 'FAIL'} ({best['so1_mean_ndcg_10']:.4f})")
            print(f"    Recall@10 ≥ 0.75:{'PASS' if r10_ok else 'FAIL'} ({best['so1_recall_at_k'][10]:.4f})")

        # Save results
        out_path = "naive_rag_results_gemma.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")
    else:
        print("\nNo results — check your API keys and model configurations.")
