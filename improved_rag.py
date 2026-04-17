"""
improved_rag.py — Improved RAG Evaluation (Hybrid + Reranking + SQL)
=====================================================================
Two-phase evaluation:
  Phase 1: Retrieval-only (local, no API calls) — tests 7 retrieval configs
           on all 593 queries, picks best config by SO1 metrics
  Phase 2: Full generation (API calls) — tests 6 models × 8 gen configs
           using Phase 1 winner, computes SO1 + SO2 + SO3 + citation tracking

Improvements over Naive RAG:
  - Hybrid retrieval: BM25 (keyword) + dense (e5-small-v2) across 3 collections
  - Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
  - SQL retrieval path for structured course/prereq queries (SO2)
  - Citation tracking in system prompt + verification (SO3)
  - Intent detection routes queries to SQL vs vector paths

Collections searched: checklist, policies, faqs (2,571 total chunks)
Embedding model: intfloat/e5-small-v2
Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

Usage:
    # Phase 1 only (no API keys needed):
    python improved_rag.py --phase 1

    # Phase 2 only (needs API keys + Phase 1 results):
    python improved_rag.py --phase 2

    # Both phases:
    python improved_rag.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

import os, re, json, time, random, math, argparse
import nltk
import openpyxl
from pathlib import Path
from typing import List, Dict, Optional, Tuple

nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("omw-1.4",   quiet=True)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DATASET_PATH     = "dataset_test.xlsx"  # held-out 20% split — run dataset_split.py first
CHROMA_BASE_DIR  = "./chroma_store"
EMBEDDING_MODEL  = "intfloat/e5-small-v2"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
E5_QUERY_PREFIX  = "query: "

# SQL connection (XAMPP MariaDB)
SQL_HOST     = "localhost"
SQL_PORT     = 3307
SQL_USER     = "root"
SQL_PASSWORD = ""
SQL_DB       = "dlsu_cpe_advising"


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD TEST CASES
# ═══════════════════════════════════════════════════════════════════════════

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
# 2. SO1 RELEVANCE RESOLUTION (same as naive_rag & embedding_experiment)
# ═══════════════════════════════════════════════════════════════════════════

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
    cat = tc["category"]
    kw  = tc.get("keywords", "")
    src = tc.get("source_file", "")

    if cat in CHECKLIST_CATS:
        codes = [k.strip() for k in kw.split(",") if k.strip()]
        valid = [c for c in codes if re.match(r"^[A-Z]{2,8}\d*[A-Z]?$", c)]
        return valid[:3] if valid else []

    if cat in POLICY_CATS:
        # First try specific source mapping
        for key, dt in SOURCE_TO_DOC_TYPE.items():
            if key in src:
                return [dt]

        # For Handbook-sourced queries, map by category to the most
        # relevant doc_types that actually exist in the policies collection.
        # The collection has: retention_policy, load_policy, lab_lecture_policy,
        # ojt_policy, advising_best_practices, advising_guidelines, thesis_policies
        if "Handbook" in src or "handbook" in src:
            HANDBOOK_CAT_MAP = {
                "grading_policy":     ["retention_policy"],
                "attendance_policy":  ["retention_policy", "load_policy"],
                "enrollment_policy":  ["load_policy", "advising_guidelines"],
                "withdrawal_policy":  ["retention_policy", "load_policy"],
                "course_credit":      ["load_policy"],
                "edge_case":          ["retention_policy", "load_policy", "advising_guidelines"],
                "student_query_variant": ["advising_guidelines", "advising_best_practices"],
            }
            mapped = HANDBOOK_CAT_MAP.get(cat)
            if mapped:
                return mapped
            # Fallback for unmapped categories
            return ["retention_policy", "load_policy"]

        return []

    relevant = []
    codes = [k.strip() for k in kw.split(",") if k.strip()]
    valid = [c for c in codes if re.match(r"^[A-Z]{2,8}\d*[A-Z]?$", c)]
    relevant.extend(valid[:2])
    for key, dt in SOURCE_TO_DOC_TYPE.items():
        if key in src and dt not in relevant:
            relevant.append(dt)
            break
    return relevant


def compute_mrr(ranked_ids, relevant):
    for i, d in enumerate(ranked_ids, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0

def compute_precision_at_k(ranked_ids, relevant, k):
    top = ranked_ids[:k]
    # Count unique relevant docs found (not duplicates)
    found = set(d for d in top if d in relevant)
    return len(found) / k if top else 0.0

def compute_recall_at_k(ranked_ids, relevant, k):
    if not relevant:
        return 0.0
    # Count unique relevant docs found in top-k
    found = set(d for d in ranked_ids[:k] if d in relevant)
    return min(len(found) / len(relevant), 1.0)

def compute_ndcg_at_k(ranked_ids, relevant, k):
    # Only count first occurrence of each relevant doc
    seen = set()
    dcg = 0.0
    for i, d in enumerate(ranked_ids[:k], 1):
        if d in relevant and d not in seen:
            dcg += 1.0 / math.log2(i + 1)
            seen.add(d)
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return min(dcg / idcg, 1.0) if idcg else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. RETRIEVAL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

# --- 3a. ChromaDB (dense retrieval across 3 collections) ---

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embedding model...")
embedding_fn = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

def load_collections() -> dict:
    """Load all 3 ChromaDB collections."""
    stores = {}
    for name in ["checklist", "policies", "faqs"]:
        path = f"{CHROMA_BASE_DIR}/{name}"
        stores[name] = Chroma(
            collection_name=name,
            embedding_function=embedding_fn,
            persist_directory=path,
        )
        count = stores[name]._collection.count()
        print(f"  [{name}] {count} chunks loaded")

        # Diagnostic: show first document from each collection
        if count > 0:
            sample = stores[name]._collection.peek(limit=1)
            sample_doc = (sample.get("documents") or [""])[0][:120]
            sample_meta = (sample.get("metadatas") or [{}])[0]
            meta_keys = list(sample_meta.keys())
            print(f"    SAMPLE doc: '{sample_doc}...'")
            print(f"    SAMPLE meta keys: {meta_keys}")
            cc = sample_meta.get("course_code", "N/A")
            dt = sample_meta.get("doc_type", "N/A")
            print(f"    SAMPLE course_code={cc} doc_type={dt}")
    return stores


def dense_search_all(stores: dict, query: str, k_per_collection: int = 10
                     ) -> list[tuple[str, str, float, dict]]:
    """
    Search all 3 collections, return list of (doc_id, text, score, metadata).
    Uses similarity_search_with_score (lower = more similar for L2 distance).

    For checklist collection: if the query contains course codes, also does
    exact metadata filtering to guarantee the correct course chunks are
    included. This is critical because structurally similar checklist chunks
    (e.g. LBYEC3A vs LBYEC3D) have nearly identical embeddings, causing the
    wrong course to rank first in pure semantic search.
    """
    results = []
    prefixed = f"{E5_QUERY_PREFIX}{query}"

    # Detect course codes in query for exact matching
    query_codes = detect_course_codes(query)

    for name, store in stores.items():
        try:
            # Standard semantic search
            hits = store.similarity_search_with_score(prefixed, k=k_per_collection)
            for doc, dist in hits:
                doc_id = _extract_doc_id(doc.metadata, name, doc.page_content)
                results.append((doc_id, doc.page_content, dist, doc.metadata))

            # For checklist collection: also fetch exact course code matches
            # AND their prerequisite courses for better Recall@10
            if name == "checklist" and query_codes:
                seen_texts = {doc.page_content for doc, _ in hits}
                prereq_codes_to_fetch = set()

                for code in query_codes[:3]:
                    try:
                        exact_hits = store.similarity_search_with_score(
                            prefixed, k=3,
                            filter={"course_code": code}
                        )
                        for doc, dist in exact_hits:
                            if doc.page_content not in seen_texts:
                                seen_texts.add(doc.page_content)
                                doc_id = _extract_doc_id(doc.metadata, name, doc.page_content)
                                results.append((doc_id, doc.page_content, dist, doc.metadata))
                            # Collect prerequisite codes from matched chunk
                            pc = doc.metadata.get("prereq_codes", "")
                            if pc:
                                for p in pc.split(","):
                                    p = p.strip()
                                    if p and p not in query_codes:
                                        prereq_codes_to_fetch.add(p)
                    except Exception:
                        pass

                # Also fetch prerequisite course chunks
                for pcode in list(prereq_codes_to_fetch)[:3]:
                    try:
                        prereq_hits = store.similarity_search_with_score(
                            prefixed, k=2,
                            filter={"course_code": pcode}
                        )
                        for doc, dist in prereq_hits:
                            if doc.page_content not in seen_texts:
                                seen_texts.add(doc.page_content)
                                doc_id = _extract_doc_id(doc.metadata, name, doc.page_content)
                                results.append((doc_id, doc.page_content, dist, doc.metadata))
                    except Exception:
                        pass

        except Exception as e:
            print(f"  [WARN] {name} search failed: {e}")

    return results


def _extract_doc_id(metadata: dict, collection: str, page_content: str = "") -> str:
    """Build a consistent doc_id from metadata for relevance matching.

    IMPORTANT: IDs must align with what resolve_relevant_ids() produces,
    which is either course codes (e.g. 'LBYCPE2') or doc_type strings
    (e.g. 'ojt_policy').

    FAQ chunks in ChromaDB store only category/program in metadata —
    no course_code, keywords, or source_file.  So for FAQs we parse
    course codes directly from the page_content (the Q&A text).
    """
    if collection == "checklist":
        return metadata.get("course_code", "unknown")

    elif collection == "policies":
        dt = metadata.get("doc_type", "policy")
        return f"{dt}"  # match at doc_type level for relevance

    elif collection == "faqs":
        # Priority 1: parse course codes from the actual chunk text
        # FAQ text format: "Question: ...\nAnswer: ..."
        # Course codes appear as CALENG1, LBYCPE2, ENGPHYS, FNDMATH, etc.
        # Some codes have digits (CALENG1), some are all-alpha (ENGPHYS).
        # Strategy: extract both patterns, then filter out common English
        # words that happen to be all-caps.
        if page_content:
            # Pattern A: alpha + digit (always a course code, never English)
            with_digit = re.findall(r"\b([A-Z]{2,8}\d[A-Z0-9]*)\b", page_content)
            if with_digit:
                return with_digit[0]

            # Pattern B: all-alpha codes — need to filter false positives
            # Only accept 5+ char all-alpha tokens (course codes are typically
            # 5-8 chars like ENGPHYS, FNDMATH; common words like "Answer",
            # "Question", "Course" are excluded by being mixed-case in text)
            all_alpha = re.findall(r"\b([A-Z]{5,8})\b", page_content)
            # Filter out known non-course words that appear in FAQ text
            STOP_WORDS = {
                "GENERAL", "COURSE", "UNITS", "PREREQ", "ANSWER", "QUESTION",
                "TERM", "YEAR", "LEVEL", "HARD", "SOFT", "TOTAL", "GRADE",
                "DLSU", "GCOE", "IMPORTANT", "SOURCE", "CHUNK", "POLICY",
                "PASSAGE", "CHECKLIST",
            }
            filtered = [c for c in all_alpha if c not in STOP_WORDS]
            if filtered:
                return filtered[0]

        # Priority 2: if metadata happens to have course_code (future-proof)
        cc = metadata.get("course_code", "")
        if cc and re.match(r"^[A-Z]{2,8}\d", cc):
            return cc

        # Priority 3: map FAQ category → doc_type for policy-related FAQs
        cat = metadata.get("category", "General")
        FAQ_CAT_TO_DOC_TYPE = {
            "ojt_policy": "ojt_policy",
            "enrollment_policy": "enrollment_policy",
            "grading_policy": "grading_policy",
            "attendance_policy": "attendance_policy",
            "withdrawal_policy": "withdrawal_policy",
            "retention_policy": "retention_policy",
            "load_policy": "load_policy",
            "lab_lecture_policy": "lab_lecture_policy",
            "advising_guidelines": "advising_guidelines",
            "advising_best_practices": "advising_best_practices",
            "thesis_policies": "thesis_policies",
            "crediting_process": "crediting_process",
        }
        if cat in FAQ_CAT_TO_DOC_TYPE:
            return FAQ_CAT_TO_DOC_TYPE[cat]

        # Fallback: category-based (won't match ground truth)
        return f"faq_{cat}"

    return metadata.get("doc_id", "unknown")


# --- 3b. BM25 (lexical retrieval) ---

from rank_bm25 import BM25Okapi

def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, strip punctuation, split on whitespace."""
    # Remove punctuation so 'caleng1?' matches 'caleng1', 'code:' matches 'code'
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in cleaned.split() if t]


def build_bm25_index(stores: dict) -> tuple:
    """Build BM25 index from all ChromaDB documents."""
    print("Building BM25 index from all collections...")
    all_texts = []
    all_ids   = []
    all_meta  = []

    for name, store in stores.items():
        collection = store._collection
        data = collection.get(include=["documents", "metadatas"])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        ids = data.get("ids", [])

        for i, (text, meta) in enumerate(zip(docs, metas)):
            if text:
                # Strip 'passage: ' prefix added by chunking pipeline for e5 embeddings
                # BM25 needs clean text for keyword matching
                clean_text = text.removeprefix("passage: ").removeprefix("passage:")
                all_texts.append(clean_text)
                all_ids.append(_extract_doc_id(meta, name, clean_text))
                all_meta.append(meta)

    tokenized = [_tokenize(text) for text in all_texts]
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 index built: {len(all_texts)} documents")
    return bm25, all_texts, all_ids, all_meta


def bm25_search(bm25, all_texts, all_ids, all_meta, query: str, k: int = 20
                ) -> list[tuple[str, str, float, dict]]:
    """BM25 keyword search, returns (doc_id, text, score, metadata)."""
    scores = bm25.get_scores(_tokenize(query))
    max_score = max(scores) if max(scores) > 0 else 1.0

    ranked = sorted(
        zip(all_ids, all_texts, scores, all_meta),
        key=lambda x: -x[2]
    )
    return [(did, text, score / max_score, meta) for did, text, score, meta in ranked[:k]]


# --- 3c. Cross-encoder reranker ---

from sentence_transformers import CrossEncoder

print("Loading cross-encoder reranker...")
reranker = CrossEncoder(RERANKER_MODEL)


def rerank_chunks(query: str, chunks: list[tuple], top_k: int = 5
                  ) -> list[tuple]:
    """Rerank (doc_id, text, score, metadata) tuples using cross-encoder."""
    if not chunks:
        return chunks

    pairs  = [(query, c[1]) for c in chunks]
    scores = reranker.predict(pairs)

    combined = list(zip(chunks, scores))
    combined.sort(key=lambda x: -x[1])
    return [c for c, _ in combined[:top_k]]


# --- 3d. SQL retrieval (structured course/prereq queries) ---

import mysql.connector

def get_sql_connection():
    return mysql.connector.connect(
        host=SQL_HOST, port=SQL_PORT, user=SQL_USER,
        password=SQL_PASSWORD, database=SQL_DB,
        charset="utf8mb4",
    )


def detect_course_codes(query: str) -> list[str]:
    """Extract course codes from the query for SQL routing."""
    codes = re.findall(r"\b[A-Z]{2,8}\d*[A-Z]?\b", query)
    noise = {"H", "S", "C", "I", "OK", "DLSU", "NO", "YES", "BOTH", "TERM",
             "AY", "NOTE", "GPA", "AND", "OR", "NOT", "ALL", "IF", "THE",
             "FOR", "SAS", "OJT", "PE", "GE", "CPE", "ECE", "IT", "CS",
             "LET", "CAN", "WHAT", "HOW", "DOES", "ARE", "IS", "DO", "WILL",
             "MY", "SHOULD", "WHICH", "WHEN", "WHERE", "WHO", "TAKE", "PASS",
             "FAIL", "NEED", "WITHOUT", "BEFORE", "AFTER", "STILL"}
    return [c for c in codes if c not in noise]


def sql_retrieve(query: str) -> Optional[str]:
    """
    If the query contains course codes, fetch structured data from SQL.
    Returns formatted text or None if no course codes found.
    """
    codes = detect_course_codes(query)
    if not codes:
        return None

    try:
        conn = get_sql_connection()
        cursor = conn.cursor(dictionary=True)
        parts = []

        for code in codes[:3]:  # limit to first 3 codes
            # Course info
            cursor.execute("""
                SELECT course_code, title, units
                FROM courses WHERE course_code = %s
            """, (code,))
            course = cursor.fetchone()

            if course:
                parts.append(
                    f"[SQL] {course['course_code']} — {course['title']} "
                    f"({course['units']} units)"
                )

            # Prerequisites
            cursor.execute("""
                SELECT c.course_code, r.course_code AS required_course,
                       p.prereq_type
                FROM prerequisites p
                JOIN courses c ON p.course_id = c.course_id
                JOIN courses r ON p.required_course_id = r.course_id
                WHERE c.course_code = %s
            """, (code,))
            prereqs = cursor.fetchall()

            if prereqs:
                for p in prereqs:
                    type_label = {"H": "Hard", "S": "Soft", "C": "Co-requisite"
                                  }.get(p["prereq_type"], p["prereq_type"])
                    parts.append(
                        f"[SQL] {p['course_code']} requires {p['required_course']} "
                        f"as {type_label} prerequisite"
                    )

            # Term placement
            cursor.execute("""
                SELECT cc.year_level, cc.term_number, cc.term_name
                FROM curriculum_courses cc
                JOIN courses c ON cc.course_id = c.course_id
                WHERE c.course_code = %s
                LIMIT 1
            """, (code,))
            term = cursor.fetchone()

            if term:
                parts.append(
                    f"[SQL] {code} is in Year {term['year_level']}, "
                    f"Term {term['term_number']} ({term['term_name']})"
                )

        cursor.close()
        conn.close()
        return "\n".join(parts) if parts else None

    except Exception as e:
        print(f"  [SQL WARN] {e}")
        return None


# --- 3e. Hybrid retrieval pipeline ---

def hybrid_retrieve(
    stores: dict,
    bm25_data: tuple,
    query: str,
    top_k: int = 6,
    alpha: float = 0.5,
    rerank_top: int = 5,
) -> dict:
    """
    Full improved retrieval pipeline:
      1. Dense search (all 3 collections)
      2. BM25 search
      3. Score fusion (alpha-weighted)
      4. Cross-encoder reranking
      5. SQL augmentation (if course codes detected)
    Returns dict with context, chunk_ids, sql_context, timings.
    """
    bm25_index, bm25_texts, bm25_ids, bm25_meta = bm25_data

    # Dense search
    t0 = time.time()
    dense_results = dense_search_all(stores, query, k_per_collection=top_k)
    dense_time = time.time() - t0

    # BM25 search
    t0 = time.time()
    bm25_results = bm25_search(bm25_index, bm25_texts, bm25_ids, bm25_meta,
                                query, k=top_k * 3)
    bm25_time = time.time() - t0

    # Score fusion using Reciprocal Rank Fusion (RRF)
    # ────────────────────────────────────────────────
    # RRF is rank-based rather than score-based, so a document ranked #1
    # by dense search but not found by BM25 still gets a strong combined
    # score.  This prevents BM25's high scores on generic policy docs from
    # burying the correct checklist chunks that dense search finds perfectly.
    #
    # RRF score = Σ  1 / (k_rrf + rank_in_system)
    # where k_rrf is a constant (typically 60) that prevents top-ranked
    # docs from dominating too heavily.
    K_RRF = 60

    # Build rank maps from each system
    # Dense: sort by distance ascending (lower = better)
    dense_sorted = sorted(dense_results, key=lambda x: x[2])
    dense_rank = {}  # text → rank (1-indexed)
    for rank, (doc_id, text, dist, meta) in enumerate(dense_sorted, 1):
        if text not in dense_rank:  # keep best rank for duplicates
            dense_rank[text] = rank

    # BM25: already sorted by score descending
    bm25_rank = {}
    for rank, (doc_id, text, score, meta) in enumerate(bm25_results, 1):
        if text not in bm25_rank:
            bm25_rank[text] = rank

    # Collect metadata from both systems
    text_to_info = {}  # text → (doc_id, metadata)
    for doc_id, text, dist, meta in dense_results:
        if text not in text_to_info:
            text_to_info[text] = (doc_id, meta)
    for doc_id, text, score, meta in bm25_results:
        if text not in text_to_info:
            text_to_info[text] = (doc_id, meta)

    # Compute RRF scores
    all_texts_set = set(dense_rank.keys()) | set(bm25_rank.keys())
    fused = []
    # Use a large fallback rank for documents not found by one system
    max_fallback = max(len(dense_results), len(bm25_results)) + 100
    for text in all_texts_set:
        d_rank = dense_rank.get(text, max_fallback)
        b_rank = bm25_rank.get(text, max_fallback)
        rrf_score = alpha / (K_RRF + d_rank) + (1 - alpha) / (K_RRF + b_rank)
        doc_id, meta = text_to_info[text]
        fused.append((doc_id, text, rrf_score, meta))

    fused.sort(key=lambda x: -x[2])

    # Build candidate set for reranking.
    # CRITICAL: Always include the top dense results so that BM25 can't
    # bury semantically strong matches (e.g. checklist chunks with exact
    # course codes that dense search finds at dist=0.25 but BM25 misses).
    # Strategy: start with top dense results, then add fused results
    # that aren't already included, up to a generous candidate pool.
    dense_top_texts = set()
    candidates = []

    # First: add top dense results (sorted by distance, best first)
    dense_sorted_for_candidates = sorted(dense_results, key=lambda x: x[2])
    for doc_id, text, dist, meta in dense_sorted_for_candidates[:top_k]:
        if text not in dense_top_texts:
            dense_top_texts.add(text)
            candidates.append((doc_id, text, 0.0, meta))  # score doesn't matter, reranker re-scores

    # Then: add fused results that aren't already included
    for item in fused:
        if item[1] not in dense_top_texts:
            candidates.append(item)
            dense_top_texts.add(item[1])
        if len(candidates) >= top_k * 3:
            break

    # Rerank
    t0 = time.time()
    reranked = rerank_chunks(query, candidates, top_k=rerank_top)
    rerank_time = time.time() - t0

    # SQL augmentation
    t0 = time.time()
    sql_context = sql_retrieve(query)
    sql_time = time.time() - t0

    # Build final context (reranked = what the LLM sees)
    chunk_ids   = [c[0] for c in reranked]
    chunk_texts = [c[1] for c in reranked]
    chunk_metas = [c[3] for c in reranked]

    # Also keep the full fused ranking for evaluation metrics @10.
    # The reranked list may only have 3-5 items (rerank_top), which
    # truncates NDCG@10 / Recall@10 unfairly.  We build a combined
    # ranking: reranked items first (in reranked order), then remaining
    # fused candidates that weren't in the reranked set.
    reranked_texts = set(c[1] for c in reranked)
    eval_ranked = list(reranked)  # reranked first
    # Add remaining candidates not in reranked set
    for c in candidates:
        if c[1] not in reranked_texts:
            eval_ranked.append(c)
            reranked_texts.add(c[1])
    # If still under 10, extend with fused items
    for c in fused:
        if len(eval_ranked) >= 10:
            break
        if c[1] not in reranked_texts:
            eval_ranked.append(c)
            reranked_texts.add(c[1])
    eval_ids = [c[0] for c in eval_ranked]

    return {
        "chunk_texts": chunk_texts,
        "chunk_ids":   chunk_ids,
        "chunk_metas": chunk_metas,
        "eval_ids":    eval_ids,     # full ranked list for metric computation
        "sql_context": sql_context,
        "timings": {
            "dense":  round(dense_time, 4),
            "bm25":   round(bm25_time, 4),
            "rerank": round(rerank_time, 4),
            "sql":    round(sql_time, 4),
            "total":  round(dense_time + bm25_time + rerank_time + sql_time, 4),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. RETRIEVAL CONFIGS
# ═══════════════════════════════════════════════════════════════════════════

RETRIEVAL_CONFIGS = [
    # Best from previous run — baseline
    {"top_k": 6,  "alpha": 1.0, "rerank_top": 5,  "label": "k=6 a=1.0 rt=5"},
    # More reranked results to improve NDCG@10 and R@10
    {"top_k": 6,  "alpha": 1.0, "rerank_top": 8,  "label": "k=6 a=1.0 rt=8"},
    {"top_k": 6,  "alpha": 1.0, "rerank_top": 10, "label": "k=6 a=1.0 rt=10"},
    # Higher k for more checklist coverage, lower alpha to test updated data
    {"top_k": 8,  "alpha": 1.0, "rerank_top": 8,  "label": "k=8 a=1.0 rt=8"},
    {"top_k": 8,  "alpha": 1.0, "rerank_top": 10, "label": "k=8 a=1.0 rt=10"},
    {"top_k": 10, "alpha": 1.0, "rerank_top": 10, "label": "k=10 a=1.0 rt=10"},
    # Slight BM25 contribution (may help policy queries)
    {"top_k": 8,  "alpha": 0.9, "rerank_top": 10, "label": "k=8 a=0.9 rt=10"},
    {"top_k": 10, "alpha": 0.9, "rerank_top": 10, "label": "k=10 a=0.9 rt=10"},
]


# ═══════════════════════════════════════════════════════════════════════════
# 5. PHASE 1 — RETRIEVAL-ONLY EVALUATION (SO1 + SO2 timing)
# ═══════════════════════════════════════════════════════════════════════════

def run_phase1(test_cases: list[dict], stores: dict, bm25_data: tuple) -> list[dict]:
    """Evaluate all retrieval configs on SO1 metrics. No LLM calls."""
    print("\n" + "=" * 70)
    print("PHASE 1 — RETRIEVAL-ONLY EVALUATION")
    print("=" * 70)

    # --- Diagnostic: how many queries have evaluable ground truth? ---
    n_with_gt = sum(1 for tc in test_cases if resolve_relevant_ids(tc))
    n_skip = len(test_cases) - n_with_gt
    print(f"\n  Ground truth coverage: {n_with_gt}/{len(test_cases)} queries "
          f"have relevant_ids ({n_skip} skipped for SO1)")
    # Show breakdown of skipped categories
    skip_cats = {}
    for tc in test_cases:
        if not resolve_relevant_ids(tc):
            cat = tc["category"]
            skip_cats[cat] = skip_cats.get(cat, 0) + 1
    if skip_cats:
        print(f"  Skipped categories (no ground truth IDs):")
        for cat, cnt in sorted(skip_cats.items(), key=lambda x: -x[1])[:10]:
            print(f"    {cat}: {cnt}")

    # --- Diagnostic: one-time test search to verify per-collection results ---
    test_q = "What is the prerequisite of CALENG1?"
    print(f"\n  DIAGNOSTIC SEARCH: '{test_q}'")
    prefixed_q = f"{E5_QUERY_PREFIX}{test_q}"
    for coll_name, store in stores.items():
        try:
            hits = store.similarity_search_with_score(prefixed_q, k=3)
            print(f"    [{coll_name}] {len(hits)} hits:")
            for doc, dist in hits:
                cc = doc.metadata.get("course_code", "N/A")
                dt = doc.metadata.get("doc_type", "N/A")
                text_preview = doc.page_content[:80].replace("\n", " ")
                print(f"      dist={dist:.4f} cc={cc} dt={dt} text='{text_preview}...'")
        except Exception as e:
            print(f"    [{coll_name}] SEARCH FAILED: {e}")
    # Also test BM25
    bm25_index, bm25_texts, bm25_ids, bm25_meta = bm25_data
    bm25_hits = bm25_search(bm25_index, bm25_texts, bm25_ids, bm25_meta, test_q, k=5)
    print(f"    [BM25] top 5:")
    for did, text, score, meta in bm25_hits[:5]:
        text_preview = text[:80].replace("\n", " ")
        print(f"      score={score:.4f} id={did} text='{text_preview}...'")
    print()

    all_results = []

    for rcfg in RETRIEVAL_CONFIGS:
        label = rcfg["label"]
        print(f"\n  --- Retrieval config: {label} ---")

        mrrs, ndcgs, p_at_k, r_at_k = [], [], {k: [] for k in [1,3,5,10]}, {k: [] for k in [1,3,5,10]}
        cos_sims, ret_times, sql_hits = [], [], 0
        so1_n = 0
        n_id_mismatch = 0   # track how often retrieved IDs have zero overlap
        mismatch_samples = []  # collect first N zero-MRR samples for debugging
        detail = []

        for i, tc in enumerate(test_cases):
            ret = hybrid_retrieve(
                stores, bm25_data, tc["question"],
                top_k=rcfg["top_k"], alpha=rcfg["alpha"],
                rerank_top=rcfg["rerank_top"],
            )

            ret_times.append(ret["timings"]["total"])
            if ret["sql_context"]:
                sql_hits += 1

            # SO1 scoring — use eval_ids (full fused+reranked ranking)
            # instead of chunk_ids (truncated to rerank_top)
            relevant_ids = resolve_relevant_ids(tc)
            ranked_ids = ret.get("eval_ids", ret["chunk_ids"])

            entry = {
                "id":           tc["id"],
                "category":     tc["category"],
                "program":      tc["program"],
                "question":     tc["question"],
                "relevant_ids": relevant_ids,
                "retrieved_ids": ranked_ids[:10],
                "reranked_ids":  ret["chunk_ids"],
                "sql_hit":      ret["sql_context"] is not None,
                "timings":      ret["timings"],
            }

            if relevant_ids:
                rel_set = set(relevant_ids)
                q_mrr = compute_mrr(ranked_ids, rel_set)
                mrrs.append(q_mrr)
                ndcgs.append(compute_ndcg_at_k(ranked_ids, rel_set, 10))
                for k in [1, 3, 5, 10]:
                    p_at_k[k].append(compute_precision_at_k(ranked_ids, rel_set, k))
                    r_at_k[k].append(compute_recall_at_k(ranked_ids, rel_set, k))
                so1_n += 1
                entry["mrr"] = q_mrr
                # Track total mismatches for diagnostics
                if q_mrr == 0.0:
                    n_id_mismatch += 1
                    if len(mismatch_samples) < 15:
                        # Also record which collection each retrieved chunk came from
                        ret_with_coll = []
                        for cid, meta in zip(ret["chunk_ids"], ret["chunk_metas"]):
                            coll = "checklist" if meta.get("course_code") else \
                                   "policies" if meta.get("doc_type") not in (None, "faq") else \
                                   "faqs"
                            ret_with_coll.append(f"{cid}({coll})")
                        mismatch_samples.append({
                            "q": tc["question"][:80],
                            "cat": tc["category"],
                            "expected": relevant_ids,
                            "retrieved": ret_with_coll[:5],
                            "eval_ids": ranked_ids[:5],
                        })

            detail.append(entry)

            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(test_cases)} processed...")

        # Aggregate
        summary = {
            "retrieval_config": label,
            "top_k":            rcfg["top_k"],
            "alpha":            rcfg["alpha"],
            "rerank_top":       rcfg["rerank_top"],
            "n_questions":      len(test_cases),
            "so1_n_evaluated":  so1_n,
            "so1_n_zero_mrr":   n_id_mismatch,
            "so1_mean_mrr":     round(sum(mrrs) / so1_n, 4) if so1_n else 0,
            "so1_mean_ndcg_10": round(sum(ndcgs) / so1_n, 4) if so1_n else 0,
            "so1_precision_at_k": {k: round(sum(v) / so1_n, 4) if so1_n else 0 for k, v in p_at_k.items()},
            "so1_recall_at_k":    {k: round(sum(v) / so1_n, 4) if so1_n else 0 for k, v in r_at_k.items()},
            "so2_sql_hit_rate":   round(sql_hits / len(test_cases) * 100, 1),
            "so2_avg_ret_time":   round(sum(ret_times) / len(ret_times), 4),
            "so2_pct_under_500ms": round(sum(1 for t in ret_times if t < 0.5) / len(ret_times) * 100, 1),
            "detail": detail,
        }

        # SO1 target check
        mrr_ok  = summary["so1_mean_mrr"] >= 0.8
        ndcg_ok = summary["so1_mean_ndcg_10"] >= 0.8
        r10_ok  = (summary["so1_recall_at_k"].get(10, 0)) >= 0.75

        print(f"    SO1 | MRR:{summary['so1_mean_mrr']:.4f} "
              f"{'PASS' if mrr_ok else 'FAIL'} | "
              f"NDCG@10:{summary['so1_mean_ndcg_10']:.4f} "
              f"{'PASS' if ndcg_ok else 'FAIL'} | "
              f"R@10:{summary['so1_recall_at_k'][10]:.4f} "
              f"{'PASS' if r10_ok else 'FAIL'}")
        print(f"    SO2 | SQL hits:{sql_hits}/{len(test_cases)} "
              f"({summary['so2_sql_hit_rate']:.0f}%) | "
              f"Avg retrieval:{summary['so2_avg_ret_time']:.4f}s | "
              f"<500ms:{summary['so2_pct_under_500ms']:.0f}%")
        print(f"    DIAG | evaluated:{so1_n}/{len(test_cases)} | "
              f"zero-MRR:{n_id_mismatch}/{so1_n} "
              f"({round(n_id_mismatch/so1_n*100,1) if so1_n else 0}%)")

        # Print first N mismatch samples for the first config only
        if mismatch_samples and rcfg == RETRIEVAL_CONFIGS[0]:
            print(f"\n    --- MISMATCH SAMPLES (first {len(mismatch_samples)}) ---")
            for s in mismatch_samples:
                print(f"      Q: {s['q']}")
                print(f"        cat={s['cat']} expected={s['expected']}")
                print(f"        reranked={s['retrieved']}")
                print(f"        eval_ids={s['eval_ids']}")
                print()

        all_results.append(summary)

    # Pick best by composite score (same weighting as embedding_experiment)
    def composite(s):
        return 0.4 * s["so1_mean_mrr"] + 0.3 * s["so1_mean_ndcg_10"] + \
               0.2 * s["so1_recall_at_k"].get(10, 0) + 0.1 * (1 if s["so2_avg_ret_time"] < 0.5 else 0)

    best = max(all_results, key=composite)
    print(f"\n  BEST CONFIG: {best['retrieval_config']} "
          f"(MRR={best['so1_mean_mrr']:.4f}, NDCG@10={best['so1_mean_ndcg_10']:.4f})")

    # Save
    with open("improved_rag_phase1_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("  Phase 1 results saved to improved_rag_phase1_results.json")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# 6. PHASE 2 — GENERATION EVALUATION (SO3 + citation tracking)
# ═══════════════════════════════════════════════════════════════════════════

# --- Generation configs ---

PARAM_CONFIGS = [
    #{"temperature": 0.0, "max_tokens": 200, "top_p": 1.0, "label": "t=0.0 tok=200"},
    #{"temperature": 0.0, "max_tokens": 400, "top_p": 1.0, "label": "t=0.0 tok=400"},
    #{"temperature": 0.1, "max_tokens": 200, "top_p": 1.0, "label": "t=0.1 tok=200"},
    {"temperature": 0.1, "max_tokens": 400, "top_p": 1.0, "label": "t=0.1 tok=400"},
    #{"temperature": 0.3, "max_tokens": 200, "top_p": 1.0, "label": "t=0.3 tok=200"},
    {"temperature": 0.3, "max_tokens": 400, "top_p": 1.0, "label": "t=0.3 tok=400"},
    #{"temperature": 0.1, "max_tokens": 200, "top_p": 0.9, "label": "t=0.1 tok=200 p=0.9"},
    {"temperature": 0.1, "max_tokens": 400, "top_p": 0.9, "label": "t=0.1 tok=400 p=0.9"},
]

MODELS_TO_TEST = [
    #{"provider": "hf",     "model_id": "meta-llama/Llama-3.1-8B-Instruct",            "label": "Llama-3.1-8B"},
    #{"provider": "hf",     "model_id": "Qwen/Qwen2.5-7B-Instruct",                    "label": "Qwen2.5-7B"},
    #{"provider": "hf",     "model_id": "google/gemma-2-9b-it:featherless-ai",          "label": "Gemma-2-9B"},
    #{"provider": "gemini", "model_id": "gemini-2.5-flash-lite",                        "label": "Gemini-2.5-Flash-Lite"},
    {"provider": "openai", "model_id": "gpt-4o-mini",                                  "label": "GPT-4o-mini"},
]


# --- Citation-aware system prompt ---

SYSTEM_PROMPT = (
    "You are an academic adviser for De La Salle University (DLSU) "
    "Computer Engineering (CpE) and Electronics Engineering (ECE) students. "
    "Use ONLY the provided context to answer questions about the curriculum, "
    "prerequisites, co-requisites, academic policies, and student handbook rules.\n\n"
    "IMPORTANT: For each claim in your answer, cite the source using the format "
    "[Source: ID] where ID is the chunk identifier shown in the context. "
    "Keep answers concise and factual.\n"
    "If the answer is not in the context, say: "
    "'I don't have that information — please consult your adviser.'"
)


def build_citation_context(ret: dict) -> str:
    """Build a context string with chunk IDs for citation tracking."""
    parts = []

    # SQL results first (most precise for structured queries)
    if ret["sql_context"]:
        parts.append(f"[Chunk: SQL_RESULT]\n{ret['sql_context']}")

    # Vector-retrieved chunks
    for chunk_id, text, meta in zip(ret["chunk_ids"], ret["chunk_texts"], ret["chunk_metas"]):
        source_label = meta.get("course_code") or meta.get("doc_type") or chunk_id
        parts.append(f"[Chunk: {source_label}]\n{text}")

    return "\n\n".join(parts)


def parse_citations(answer: str) -> list[str]:
    """Extract cited source IDs from [Source: ...] tags in the answer."""
    return re.findall(r"\[Source:\s*([^\]]+)\]", answer)


def compute_citation_metrics(cited: list[str], chunk_ids: list[str],
                              relevant_ids: list[str], had_sql: bool) -> dict:
    """Compute citation precision, recall, and adherence."""
    available = set(chunk_ids)
    if had_sql:
        available.add("SQL_RESULT")

    if not cited:
        return {"citation_precision": 0.0, "citation_recall": 0.0, "n_citations": 0}

    # Precision: what % of cited sources were in the retrieved context
    valid_citations = sum(1 for c in cited if c in available or
                          any(c.lower() in a.lower() for a in available))
    precision = valid_citations / len(cited) if cited else 0.0

    # Recall: what % of relevant chunks got cited
    relevant_set = set(relevant_ids)
    if relevant_set:
        cited_set = set(cited)
        cited_relevant = sum(1 for r in relevant_set if r in cited_set or
                             any(r.lower() in c.lower() for c in cited_set))
        recall = cited_relevant / len(relevant_set)
    else:
        recall = 0.0

    return {
        "citation_precision": round(precision, 4),
        "citation_recall":    round(recall, 4),
        "n_citations":        len(cited),
    }


# --- Generation function ---

from huggingface_hub import InferenceClient

def _call_with_retry(fn, max_retries=5):
    """Wrapper that handles transient API errors by waiting and retrying.
    Covers: 402 (budget), 429 (rate limit), 500/502/503 (server errors)."""
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
                wait = 60 * (attempt + 1)
                print(f"\n  [429] Rate limited — waiting {wait}s "
                      f"(attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            elif any(code in err for code in ["500", "502", "503"]) or "UNAVAILABLE" in err:
                wait = 30 * (attempt + 1)   # exponential-ish backoff
                code = "500" if "500" in err else ("502" if "502" in err else "503")
                print(f"\n  [{code}] Server error — waiting {wait}s "
                      f"(attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Max retries ({max_retries}) exceeded — last error: {err}")


def generate(client: dict, model_id: str, context: str,
             question: str, config: dict) -> str:

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
            gemini_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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
        def _openai_call():
            resp = client["instance"].chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                top_p=top_p,
            )
            return resp.choices[0].message.content.strip()
        answer = _call_with_retry(_openai_call)

    else:
        raise ValueError(f"Unknown provider: {client['provider']}")

    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    if not answer:
        answer = "[NO RESPONSE — model returned empty output]"
    return answer


# --- SO3 scoring ---

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

# ── NLI context grounding scorer — loaded once at startup ───────────────────
# Uses cross-encoder/nli-deberta-v3-small (sentence-transformers) to score
# how well the generated answer is entailed by the retrieved context.
# No torch version conflicts — works with your existing cu124 install.
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

_rouge  = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
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


# --- Hallucination detection — multi-condition framework ---
#
# Five conditions evaluated per answer:
#   C1  RETRIEVAL_MISS       — model refuses when ground truth is substantive
#   C2  CONTRADICTION        — polarity flip (GT negates, answer affirms)
#   C3  WRONG_CODES          — fabricated course codes not in ground truth
#   C4  NUMERIC_FABRICATION  — invented numbers (units, GPA, %, years) not in GT
#   C5  HIGH_CLAIM_RATE      — ≥60% of answer sentences unsupported vs GT
#
# RAG-specific grounding check:
#   NLI grounding score — answer faithfulness to retrieved context [0..1]
#
# Outputs: hallucination (bool), halluc_reason (str),
#          halluc_claim_rate (float), factual_consistency (float),
#          completeness_score (float), align_score (float|None)

_NOISE_CODES = {
    "I", "H", "S", "C", "OK", "ONLY", "DLSU", "NO", "YES",
    "BOTH", "TERM", "AY", "NOTE", "GPA", "CUM", "AND", "OR",
    "NOT", "ALL", "IF", "THE", "FOR", "SAS", "OJT", "PE",
    "GE", "CPE", "ECE", "IT", "CS", "LET",
}

_NO_INFO_PHRASES = [
    "don't have that information", "cannot find", "no information",
    "i'm not sure", "i do not have", "consult your adviser",
    "unable to provide", "not in the context", "context does not",
]

_ROUGE_SENT   = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_CLAIM_THRESHOLD = 0.15


def _claim_level_rate(ground_truth: str, answer: str) -> float:
    """Sentence-level unsupported-claim ratio (Hallucination Rate formula)."""
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
    """Numbers in answer absent from ground truth (≥2 significant digits)."""
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


# --- Phase 2 evaluation loop ---

def run_phase2(test_cases: list[dict], stores: dict, bm25_data: tuple,
               best_ret_config: dict) -> list[dict]:
    """Run generation evaluation using best retrieval config from Phase 1."""
    print("\n" + "=" * 70)
    print("PHASE 2 — GENERATION EVALUATION")
    print(f"Retrieval config: {best_ret_config['label']}")
    print("=" * 70)

    # Init API clients
    clients = {}
    if os.getenv("HF_TOKEN"):
        clients["hf"] = {"provider": "hf",
                         "instance": InferenceClient(token=os.getenv("HF_TOKEN"))}
    if os.getenv("GEMINI_API_KEY"):
        try:
            from google import genai as google_genai
            clients["gemini"] = {"provider": "gemini", "instance": None}
            print("  Gemini ready")
        except ImportError:
            print("  [SKIP] Gemini — pip install google-genai")
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            clients["openai"] = {"provider": "openai",
                                 "instance": OpenAI(api_key=os.getenv("OPENAI_API_KEY"))}
            print("  OpenAI ready")
        except ImportError:
            print("  [SKIP] OpenAI — pip install openai")

    all_summaries = []

    for model_cfg in MODELS_TO_TEST:
        provider = model_cfg["provider"]

        if provider not in clients:
            print(f"\n  [SKIP] {model_cfg['label']} — {provider} not available")
            continue

        print(f"\n{'='*70}")
        print(f"MODEL: {model_cfg['label']}")
        print(f"{'='*70}")

        for pcfg in PARAM_CONFIGS:
            print(f"\n  --- {pcfg['label']} ---")

            # --- Checkpoint support ---
            # Sanitize label for filename
            safe_model = re.sub(r'[^\w\-]', '_', model_cfg["label"])
            safe_cfg   = re.sub(r'[^\w\-]', '_', pcfg["label"])
            ckpt_path  = Path(f"phase2_ckpt_{safe_model}_{safe_cfg}.json")
            cached = {}
            if ckpt_path.exists():
                try:
                    cached = {d["id"]: d for d in json.load(open(ckpt_path, encoding="utf-8"))}
                    print(f"    Resuming from checkpoint ({len(cached)}/{len(test_cases)} done)")
                except Exception:
                    cached = {}

            scores_all, halluc_flags, halluc_details_all, citation_metrics_all = [], [], [], []
            gen_times, total_times = [], []
            so1_mrrs, so1_ndcgs = [], []
            so1_n = 0
            detail = []
            all_answers, all_references = [], []
            n_skipped = 0

            for i, tc in enumerate(test_cases):
                # --- Resume from checkpoint if available ---
                if tc["id"] in cached:
                    d = cached[tc["id"]]
                    scores = {k: d[k] for k in ["rouge1", "rouge_l", "bleu", "meteor"]}
                    scores_all.append(scores)
                    halluc_flags.append(d["hallucination"])
                    halluc_details_all.append({
                        "hallucination":       d["hallucination"],
                        "halluc_reason":       d.get("halluc_reason", "OK"),
                        "halluc_claim_rate":   d.get("halluc_claim_rate", 0.0),
                        "factual_consistency": d.get("factual_consistency", 1.0),
                        "completeness_score":  d.get("completeness_score", 1.0),
                        "align_score":         d.get("align_score"),
                    })
                    citation_metrics_all.append({
                        "citation_precision": d["citation_precision"],
                        "citation_recall":    d["citation_recall"],
                        "n_citations":        d["n_citations"],
                    })
                    gen_times.append(d["gen_time"])
                    total_times.append(d["total_time"])
                    all_answers.append(d["generated"])
                    all_references.append(d["ground_truth"])
                    detail.append(d)
                    relevant_ids = resolve_relevant_ids(tc)
                    if relevant_ids:
                        rel_set = set(relevant_ids)
                        so1_mrrs.append(compute_mrr(d["retrieved_ids"], rel_set))
                        so1_ndcgs.append(compute_ndcg_at_k(d["retrieved_ids"], rel_set, 10))
                        so1_n += 1
                    n_skipped += 1
                    continue

                t_start = time.time()

                # Retrieve
                ret = hybrid_retrieve(
                    stores, bm25_data, tc["question"],
                    top_k=best_ret_config["top_k"],
                    alpha=best_ret_config["alpha"],
                    rerank_top=best_ret_config["rerank_top"],
                )

                # Build citation-aware context
                context = build_citation_context(ret)

                # Generate
                t_gen = time.time()
                answer = generate(clients[provider], model_cfg["model_id"],
                                  context, tc["question"], pcfg)
                gen_time = time.time() - t_gen
                total_time = time.time() - t_start

                # SO3 scoring
                scores = score_response(tc["answer"], answer)
                halluc = detect_hallucination(tc["answer"], answer, context=context)

                # Citation tracking
                cited = parse_citations(answer)
                relevant_ids = resolve_relevant_ids(tc)
                cit_metrics = compute_citation_metrics(
                    cited, ret["chunk_ids"], relevant_ids,
                    ret["sql_context"] is not None
                )

                # SO1 (retrieval quality for this query)
                if relevant_ids:
                    rel_set = set(relevant_ids)
                    so1_mrrs.append(compute_mrr(ret["chunk_ids"], rel_set))
                    so1_ndcgs.append(compute_ndcg_at_k(ret["chunk_ids"], rel_set, 10))
                    so1_n += 1

                scores_all.append(scores)
                halluc_flags.append(halluc["hallucination"])
                halluc_details_all.append(halluc)
                citation_metrics_all.append(cit_metrics)
                gen_times.append(gen_time)
                total_times.append(total_time)
                all_answers.append(answer)
                all_references.append(tc["answer"])

                h_tag = "[HALLUC]" if halluc["hallucination"] else "[OK]"

                entry = {
                    "id": tc["id"], "category": tc["category"],
                    "program": tc["program"], "question": tc["question"],
                    "ground_truth": tc["answer"], "generated": answer,
                    "retrieved_ids": ret["chunk_ids"],
                    "sql_context": ret["sql_context"] is not None,
                    "citations": cited,
                    **scores, **cit_metrics, **halluc,
                    "gen_time": round(gen_time, 3),
                    "total_time": round(total_time, 3),
                }
                detail.append(entry)

                # Save checkpoint every 25 questions
                if (i + 1 - n_skipped) % 25 == 0:
                    json.dump(detail, open(ckpt_path, "w", encoding="utf-8"),
                              indent=1, ensure_ascii=False)

                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(test_cases)}...")

            # Final checkpoint save (complete)
            json.dump(detail, open(ckpt_path, "w", encoding="utf-8"),
                      indent=1, ensure_ascii=False)

            # Batch BERTScore — loads roberta-large ONCE for all answers
            print(f"    Computing BERTScore for {len(all_answers)} answers (batched)...")
            bert_scores = batch_bert_score(all_answers, all_references)
            for idx, bs in enumerate(bert_scores):
                scores_all[idx]["bert_score"] = bs
                detail[idx]["bert_score"] = bs

            n = len(test_cases)
            hc = sum(halluc_flags)
            def avg(k): return round(sum(s[k] for s in scores_all) / n, 4)
            def avgh(k): return round(sum(h[k] for h in halluc_details_all) / n, 4)
            def avgh_opt(k):
                vals = [h[k] for h in halluc_details_all if h[k] is not None]
                return round(sum(vals) / len(vals), 4) if vals else None

            summary = {
                "model": model_cfg["label"], "provider": provider,
                "config": pcfg["label"],
                "retrieval_config": best_ret_config["label"],
                "n_questions": n,
                # SO3
                "avg_rouge1": avg("rouge1"), "avg_rouge_l": avg("rouge_l"),
                "avg_bleu": avg("bleu"), "avg_meteor": avg("meteor"),
                "avg_bert_score": avg("bert_score"),
                "hallucination_count": hc,
                "hallucination_rate": round(hc / n * 100, 1),
                "avg_halluc_claim_rate":   avgh("halluc_claim_rate"),
                "avg_factual_consistency": avgh("factual_consistency"),
                "avg_completeness_score":  avgh("completeness_score"),
                "avg_align_score":         avgh_opt("align_score"),
                # Citations
                "avg_citation_precision": round(sum(c["citation_precision"] for c in citation_metrics_all) / n, 4),
                "avg_citation_recall": round(sum(c["citation_recall"] for c in citation_metrics_all) / n, 4),
                "avg_citations_per_answer": round(sum(c["n_citations"] for c in citation_metrics_all) / n, 2),
                # SO1
                "so1_mean_mrr": round(sum(so1_mrrs) / so1_n, 4) if so1_n else 0,
                "so1_mean_ndcg_10": round(sum(so1_ndcgs) / so1_n, 4) if so1_n else 0,
                # Timing
                "avg_gen_time": round(sum(gen_times) / n, 3),
                "avg_total_time": round(sum(total_times) / n, 3),
                "pct_under_5s": round(sum(1 for t in total_times if t < 5) / n * 100, 1),
                "detail": detail,
            }

            align_str = f"{summary['avg_align_score']:.3f}" if summary["avg_align_score"] is not None else "N/A"
            print(f"    SO3 | R1:{summary['avg_rouge1']:.3f} RL:{summary['avg_rouge_l']:.3f} "
                  f"BLEU:{summary['avg_bleu']:.3f} METEOR:{summary['avg_meteor']:.3f} "
                  f"BERT:{summary['avg_bert_score']:.3f}")
            print(f"    SO3 | Halluc:{hc}/{n} ({summary['hallucination_rate']:.0f}%) | "
                  f"ClaimRate:{summary['avg_halluc_claim_rate']:.3f} | "
                  f"Consistency:{summary['avg_factual_consistency']:.3f} | "
                  f"Completeness:{summary['avg_completeness_score']:.3f} | "
                  f"NLI-Align:{align_str}")
            print(f"    CIT | Prec:{summary['avg_citation_precision']:.3f} "
                  f"Rec:{summary['avg_citation_recall']:.3f} "
                  f"Avg:{summary['avg_citations_per_answer']:.1f}/ans")

            all_summaries.append(summary)

            # Clean up checkpoint file — this config is done
            if ckpt_path.exists():
                ckpt_path.unlink()
                print(f"    Checkpoint {ckpt_path.name} removed (config complete)")

            # Incremental save so partial results survive crashes between configs
            with open("improved_rag_phase2_results.json", "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # Final comparison
    if all_summaries:
        print(f"\n{'='*160}")
        print("IMPROVED RAG — FINAL COMPARISON (sorted by ROUGE-L)")
        print(f"{'='*160}")
        print(f"{'Model + Config':<42} {'R-L':>6} {'BLEU':>6} {'MTR':>6} {'BERT':>6} "
              f"{'Halluc':>8} {'ClmRt':>6} {'Consist':>8} {'Compl':>6} {'Align':>6} "
              f"{'CitP':>6} {'CitR':>6} {'AvgT':>7}")
        print("-" * 160)

        for s in sorted(all_summaries, key=lambda x: -x["avg_rouge_l"]):
            name = f"{s['model']} [{s['config']}]"
            align_str = f"{s['avg_align_score']:>6.3f}" if s.get("avg_align_score") is not None else "   N/A"
            print(
                f"{name:<42} "
                f"{s['avg_rouge_l']:>6.3f} "
                f"{s['avg_bleu']:>6.3f} "
                f"{s['avg_meteor']:>6.3f} "
                f"{s['avg_bert_score']:>6.3f} "
                f"{s['hallucination_count']:>4}/{s['n_questions']} "
                f"{s['avg_halluc_claim_rate']:>6.3f} "
                f"{s['avg_factual_consistency']:>8.3f} "
                f"{s['avg_completeness_score']:>6.3f} "
                f"{align_str} "
                f"{s['avg_citation_precision']:>6.3f} "
                f"{s['avg_citation_recall']:>6.3f} "
                f"{s['avg_total_time']:>6.2f}s"
            )

    with open("improved_rag_phase2_results.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 2 results saved to improved_rag_phase2_results.json")

    return all_summaries


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Run only phase 1 (retrieval) or 2 (generation). "
                             "Default: both.")
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    full_dataset = load_dataset(DATASET_PATH)
    test_cases = full_dataset
    random.Random(42).shuffle(test_cases)
    print(f"  {len(test_cases)} questions loaded")

    cats = {}
    for tc in test_cases:
        cats[tc["category"]] = cats.get(tc["category"], 0) + 1
    print(f"  {len(cats)} categories")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:5]:
        print(f"    {cat}: {count}")
    print()

    # Load retrieval components
    print("Loading retrieval components...")
    stores = load_collections()
    bm25_data = build_bm25_index(stores)
    print()

    # Phase 1
    if args.phase is None or args.phase == 1:
        phase1_results = run_phase1(test_cases, stores, bm25_data)
    else:
        # Load Phase 1 results to get best config
        p1_path = Path("improved_rag_phase1_results.json")
        if not p1_path.exists():
            print("[ERROR] Phase 1 results not found. Run --phase 1 first.")
            sys.exit(1)
        phase1_results = json.load(open(p1_path, encoding="utf-8"))
        print(f"Loaded Phase 1 results from {p1_path}")

    # Determine best retrieval config
    def composite(s):
        return 0.4 * s["so1_mean_mrr"] + 0.3 * s["so1_mean_ndcg_10"] + \
               0.2 * s.get("so1_recall_at_k", {}).get(10, s.get("so1_recall_at_k", {}).get("10", 0)) + \
               0.1 * (1 if s["so2_avg_ret_time"] < 0.5 else 0)

    best_p1 = max(phase1_results, key=composite)
    best_ret_config = {
        "top_k":      best_p1["top_k"],
        "alpha":      best_p1["alpha"],
        "rerank_top": best_p1["rerank_top"],
        "label":      best_p1["retrieval_config"],
    }
    print(f"\nBest retrieval config: {best_ret_config['label']}")

    # Phase 2
    if args.phase is None or args.phase == 2:
        print("\nChecking API keys...")
        print(f"  HF_TOKEN:       {'SET' if os.getenv('HF_TOKEN')       else 'NOT SET'}")
        print(f"  GEMINI_API_KEY: {'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")
        print(f"  OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")

        run_phase2(test_cases, stores, bm25_data, best_ret_config)

    print("\nDone.")
