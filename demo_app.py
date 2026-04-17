"""
Run:
    python demo_app.py
Then open:
    http://localhost:5000
"""

import os, re, time
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Retrieval stack ──────────────────────────────────────────────────────────
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import mysql.connector

# ── Generation ───────────────────────────────────────────────────────────────
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — best retrieval config from Phase 1
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_BASE_DIR = "./chroma_store"
EMBEDDING_MODEL = "intfloat/e5-small-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
E5_QUERY_PREFIX = "query: "

TOP_K      = 6
ALPHA      = 1.0        # dense-only (alpha=1.0, issues with BM25 TO BE FIXED)
RERANK_TOP = 10

GEN_MODEL       = "gpt-4o-mini" #best performing config per data, subject to change
GEN_TEMPERATURE = 0.1
GEN_MAX_TOKENS  = 400
GEN_TOP_P       = 0.9

SQL_HOST     = "localhost"
SQL_PORT     = 3307
SQL_USER     = "root"
SQL_PASSWORD = ""
SQL_DB       = "dlsu_cpe_advising"

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

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (once at startup)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading embedding model (e5-small-v2)...")
embedding_fn = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},   # switch to "cuda" if GPU available
    encode_kwargs={"normalize_embeddings": True},
)

print("Loading cross-encoder reranker...")
reranker = CrossEncoder(RERANKER_MODEL)

print("Loading ChromaDB collections...")
stores = {}
for name in ["checklist", "policies", "faqs"]:
    path = f"{CHROMA_BASE_DIR}/{name}"
    stores[name] = Chroma(
        collection_name=name,
        embedding_function=embedding_fn,
        persist_directory=path,
    )
    count = stores[name]._collection.count()
    print(f"  [{name}] {count} chunks")

print("Building BM25 index...")
all_bm25_texts, all_bm25_ids, all_bm25_meta = [], [], []

def _extract_doc_id(metadata, collection, page_content=""):
    if collection == "checklist":
        return metadata.get("course_code", "unknown")
    elif collection == "policies":
        return metadata.get("doc_type", "policy")
    elif collection == "faqs":
        if page_content:
            with_digit = re.findall(r"\b([A-Z]{2,8}\d[A-Z0-9]*)\b", page_content)
            if with_digit:
                return with_digit[0]
            all_alpha = re.findall(r"\b([A-Z]{5,8})\b", page_content)
            STOP = {"GENERAL","COURSE","UNITS","PREREQ","ANSWER","QUESTION",
                    "TERM","YEAR","LEVEL","HARD","SOFT","TOTAL","GRADE",
                    "DLSU","GCOE","IMPORTANT","SOURCE","CHUNK","POLICY",
                    "PASSAGE","CHECKLIST"}
            filtered = [c for c in all_alpha if c not in STOP]
            if filtered:
                return filtered[0]
        cat = metadata.get("category","General")
        return cat
    return metadata.get("doc_id","unknown")

for name, store in stores.items():
    data = store._collection.get(include=["documents","metadatas"])
    for text, meta in zip(data.get("documents",[]), data.get("metadatas",[])):
        if text:
            clean = text.removeprefix("passage: ").removeprefix("passage:")
            all_bm25_texts.append(clean)
            all_bm25_ids.append(_extract_doc_id(meta, name, clean))
            all_bm25_meta.append(meta)

_tokenize = lambda t: [w for w in re.sub(r"[^\w\s]"," ",t.lower()).split() if w]
bm25_index = BM25Okapi([_tokenize(t) for t in all_bm25_texts])
print(f"  BM25 index: {len(all_bm25_texts)} documents")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("Ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def detect_course_codes(query):
    codes = re.findall(r"\b[A-Z]{2,8}\d*[A-Z]?\b", query)
    NOISE = {"H","S","C","I","OK","DLSU","NO","YES","BOTH","TERM","AY","NOTE",
             "GPA","AND","OR","NOT","ALL","IF","THE","FOR","SAS","OJT","PE",
             "GE","CPE","ECE","IT","CS","LET","CAN","WHAT","HOW","DOES","ARE",
             "IS","DO","WILL","MY","SHOULD","WHICH","WHEN","WHERE","WHO",
             "TAKE","PASS","FAIL","NEED","WITHOUT","BEFORE","AFTER","STILL"}
    return [c for c in codes if c not in NOISE]


def dense_search_all(query, k_per_collection=10):
    results = []
    prefixed = f"{E5_QUERY_PREFIX}{query}"
    query_codes = detect_course_codes(query)

    for name, store in stores.items():
        try:
            hits = store.similarity_search_with_score(prefixed, k=k_per_collection)
            for doc, dist in hits:
                doc_id = _extract_doc_id(doc.metadata, name, doc.page_content)
                results.append((doc_id, doc.page_content, dist, doc.metadata))

            if name == "checklist" and query_codes:
                seen = {doc.page_content for doc, _ in hits}
                for code in query_codes[:3]:
                    try:
                        exact = store.similarity_search_with_score(
                            prefixed, k=3, filter={"course_code": code})
                        for doc, dist in exact:
                            if doc.page_content not in seen:
                                seen.add(doc.page_content)
                                doc_id = _extract_doc_id(doc.metadata, name, doc.page_content)
                                results.append((doc_id, doc.page_content, dist, doc.metadata))
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [WARN] {name} search failed: {e}")
    return results


def bm25_search(query, k=20):
    scores = bm25_index.get_scores(_tokenize(query))
    max_s  = max(scores) if max(scores) > 0 else 1.0
    ranked = sorted(
        zip(all_bm25_ids, all_bm25_texts, scores, all_bm25_meta),
        key=lambda x: -x[2])
    return [(did, txt, s/max_s, meta) for did, txt, s, meta in ranked[:k]]


def hybrid_retrieve(query):
    dense_results = dense_search_all(query, k_per_collection=TOP_K)
    bm25_results  = bm25_search(query, k=TOP_K * 3)

    K_RRF = 60
    dense_sorted = sorted(dense_results, key=lambda x: x[2])
    dense_rank = {}
    for rank, (_, text, _, _) in enumerate(dense_sorted, 1):
        if text not in dense_rank:
            dense_rank[text] = rank

    bm25_rank = {}
    for rank, (_, text, _, _) in enumerate(bm25_results, 1):
        if text not in bm25_rank:
            bm25_rank[text] = rank

    text_to_info = {}
    for doc_id, text, _, meta in dense_results:
        if text not in text_to_info:
            text_to_info[text] = (doc_id, meta)
    for doc_id, text, _, meta in bm25_results:
        if text not in text_to_info:
            text_to_info[text] = (doc_id, meta)

    all_texts_set = set(dense_rank) | set(bm25_rank)
    max_fallback  = max(len(dense_results), len(bm25_results)) + 100
    fused = []
    for text in all_texts_set:
        d_rank = dense_rank.get(text, max_fallback)
        b_rank = bm25_rank.get(text, max_fallback)
        rrf = ALPHA / (K_RRF + d_rank) + (1 - ALPHA) / (K_RRF + b_rank)
        doc_id, meta = text_to_info[text]
        fused.append((doc_id, text, rrf, meta))
    fused.sort(key=lambda x: -x[2])

    seen = set()
    candidates = []
    dense_sorted_for_cands = sorted(dense_results, key=lambda x: x[2])
    for doc_id, text, dist, meta in dense_sorted_for_cands[:TOP_K]:
        if text not in seen:
            seen.add(text)
            candidates.append((doc_id, text, 0.0, meta))

    for item in fused:
        if item[1] not in seen:
            candidates.append(item)
            seen.add(item[1])
        if len(candidates) >= TOP_K * 3:
            break

    # Rerank
    if candidates:
        pairs  = [(query, c[1]) for c in candidates]
        scores = reranker.predict(pairs)
        combined = sorted(zip(candidates, scores), key=lambda x: -x[1])
        reranked = [c for c, _ in combined[:RERANK_TOP]]
    else:
        reranked = candidates[:RERANK_TOP]

    # SQL augmentation
    sql_context = sql_retrieve(query)

    chunk_ids   = [c[0] for c in reranked]
    chunk_texts = [c[1] for c in reranked]
    chunk_metas = [c[3] for c in reranked]

    return {
        "chunk_ids":   chunk_ids,
        "chunk_texts": chunk_texts,
        "chunk_metas": chunk_metas,
        "sql_context": sql_context,
    }


def sql_retrieve(query):
    codes = detect_course_codes(query)
    if not codes:
        return None
    try:
        conn   = mysql.connector.connect(
            host=SQL_HOST, port=SQL_PORT, user=SQL_USER,
            password=SQL_PASSWORD, database=SQL_DB, charset="utf8mb4")
        cursor = conn.cursor(dictionary=True)
        parts  = []
        for code in codes[:3]:
            cursor.execute(
                "SELECT course_code, title, units FROM courses WHERE course_code=%s",
                (code,))
            c = cursor.fetchone()
            if c:
                parts.append(f"[SQL] {c['course_code']} — {c['title']} ({c['units']} units)")
            cursor.execute("""
                SELECT c.course_code, r.course_code AS req, p.prereq_type
                FROM prerequisites p
                JOIN courses c ON p.course_id = c.course_id
                JOIN courses r ON p.required_course_id = r.course_id
                WHERE c.course_code=%s""", (code,))
            for p in cursor.fetchall():
                t = {"H":"Hard","S":"Soft","C":"Co-requisite"}.get(p["prereq_type"],p["prereq_type"])
                parts.append(f"[SQL] {p['course_code']} requires {p['req']} as {t} prerequisite")
            cursor.execute("""
                SELECT cc.year_level, cc.term_number, cc.term_name
                FROM curriculum_courses cc
                JOIN courses c ON cc.course_id = c.course_id
                WHERE c.course_code=%s LIMIT 1""", (code,))
            term = cursor.fetchone()
            if term:
                parts.append(f"[SQL] {code} is in Year {term['year_level']}, Term {term['term_number']} ({term['term_name']})")
        cursor.close()
        conn.close()
        return "\n".join(parts) if parts else None
    except Exception as e:
        print(f"  [SQL WARN] {e}")
        return None


def build_context(ret):
    parts = []
    if ret["sql_context"]:
        parts.append(f"[Chunk: SQL_RESULT]\n{ret['sql_context']}")
    for cid, text, meta in zip(ret["chunk_ids"], ret["chunk_texts"], ret["chunk_metas"]):
        label = meta.get("course_code") or meta.get("doc_type") or cid
        parts.append(f"[Chunk: {label}]\n{text}")
    return "\n\n".join(parts)


def parse_citations(answer):
    return list(set(re.findall(r"\[Source:\s*([^\]]+)\]", answer)))


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "frontend.html")


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400

    t0 = time.time()
    # Retrieve
    ret     = hybrid_retrieve(message)
    context = build_context(ret)

    # Generate
    try:
        completion = openai_client.chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {message}"},
            ],
            max_tokens=GEN_MAX_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
        )
        answer = completion.choices[0].message.content.strip()
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Citations for display
    raw_citations = parse_citations(answer)
    citations = [{"label": c, "source": c} for c in raw_citations]

    # Retrieval metadata to surface in UI
    sources = []
    if ret["sql_context"]:
        sources.append({"id": "SQL_RESULT", "type": "sql", "preview": ret["sql_context"][:120]})
    for cid, text, meta in zip(ret["chunk_ids"], ret["chunk_texts"], ret["chunk_metas"]):
        label = meta.get("course_code") or meta.get("doc_type") or cid
        sources.append({
            "id":      cid,
            "type":    "checklist" if meta.get("course_code") else "policy",
            "label":   label,
            "preview": text[:120],
        })

    elapsed = round(time.time() - t0, 2)
    return jsonify({
        "response":  answer,
        "citations": citations,
        "sources":   sources[:8],
        "elapsed":   elapsed,
        "config": {
            "retrieval": f"k={TOP_K} α={ALPHA} rt={RERANK_TOP}",
            "model":     GEN_MODEL,
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
