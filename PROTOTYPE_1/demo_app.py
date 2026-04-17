#DEMO USING GPT4omini

from dotenv import load_dotenv
load_dotenv()

import re
import os
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

app = Flask(__name__, static_folder=".")

# client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# DATABASE (PLEASE ADD/FIX WHEN POSSIBLE)
import json

with open("knowledge_base.json", encoding="utf-8") as f:
    DOCUMENTS = json.load(f)

print(f"Loaded {len(DOCUMENTS)} knowledge chunks.")

# chromadb database vector
print("Building ChromaDB vector store...")
chroma_client = chromadb.Client()
embedding_fn  = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
try:
    chroma_client.delete_collection(name="dlsu_cpe_demo")
except Exception:
    pass
collection = chroma_client.create_collection(
    name="dlsu_cpe_demo",
    embedding_function=embedding_fn,
)
collection.add(
    ids=[doc["id"] for doc in DOCUMENTS],
    documents=[doc["text"] for doc in DOCUMENTS],
    metadatas=[doc["metadata"] for doc in DOCUMENTS],
)
print(f"Stored {len(DOCUMENTS)} chunks.")

# bm25
tokenized  = [doc["text"].lower().split() for doc in DOCUMENTS]
bm25_index = BM25Okapi(tokenized)

# cross-encoder
print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Ready.\n")

SYSTEM_PROMPT = (
    "You are an academic adviser for DLSU Computer Engineering students. "
    "Use ONLY the provided context to answer questions about the CpE curriculum "
    "AY 2022-2023. Be concise and accurate. "
    "If the answer is not in the context, say: "
    "'I don't have that information — please consult your academic adviser.'"
)

def retrieve_hybrid(query: str, top_k: int = 6, alpha: float = 0.5) -> list:
 
    # dense retrieval
    dense_results = collection.query(
        query_texts=[query],
        n_results=min(top_k * 2, len(DOCUMENTS)),  # safe cap
        include=["documents", "distances", "metadatas"]
    )
    dense_docs  = dense_results["documents"][0]
    dense_dists = dense_results["distances"][0]
 
    max_dist = max(dense_dists) if dense_dists else 1
    dense_scores = {
        doc: 1 - (dist / max_dist)
        for doc, dist in zip(dense_docs, dense_dists)
    }
 
    # bm25
    tokenized_query = query.lower().split()
    bm25_scores_raw = bm25_index.get_scores(tokenized_query)
 
    max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
 
    bm25_top_indices = sorted(
        range(len(bm25_scores_raw)),
        key=lambda i: -bm25_scores_raw[i]
    )[:top_k * 2]
    bm25_doc_scores = {
        DOCUMENTS[i]["text"]: bm25_scores_raw[i] / max_bm25
        for i in bm25_top_indices
    }
 
    # course-code boost
    query_codes = set(re.findall(r'\b[A-Z]{3,8}\d*[A-Z]?\b', query))
 
    # combine
    all_docs = set(dense_scores.keys()) | set(bm25_doc_scores.keys())
    combined = {}
    for doc in all_docs:
        d_score = dense_scores.get(doc, 0)
        b_score = bm25_doc_scores.get(doc, 0)
        base    = alpha * d_score + (1 - alpha) * b_score
        boost   = 0.3 if any(code in doc for code in query_codes) else 0
        combined[doc] = base + boost
 
    top_docs = sorted(combined.items(), key=lambda x: -x[1])[:top_k]
    return [doc for doc, _ in top_docs]

def rerank(query: str, chunks: list, top_k: int = 3) -> list:
    if not chunks:
        return chunks
    pairs  = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
    return [chunk for chunk, _ in ranked[:top_k]]

def ask(question: str) -> str:
    chunks   = retrieve_hybrid(question)
    reranked = rerank(question, chunks)
    context  = "\n\n".join(reranked)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=300,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

# FLASK 
@app.route("/")
def index():
    return send_from_directory(".", "frontend.html")

@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json()
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"error": "Empty message"}), 400
    try:
        answer = ask(question)
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting DLSU CpE Advising Chatbot...")
    print("Open: http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
