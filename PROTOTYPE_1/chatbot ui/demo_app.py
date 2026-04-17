"""
DLSU CpE Academic Advising Chatbot — Demo Backend
Improved RAG: Hybrid Search (BM25 + Dense) + Reranking + GPT-4o-mini

Setup:
  pip install flask chromadb sentence-transformers rank-bm25 openai python-dotenv

Run:
  python demo_app.py
  Open: http://localhost:5000
"""

from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

app = Flask(__name__, static_folder=".")

# ── OpenAI client ──────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Knowledge base ─────────────────────────────────────────────────────────
DOCUMENTS = [
    {"id": "term1_overview", "text": "TERM 1 - First Term courses: NSTP101 (National Service Training Program-General Orientation, 0 units), FNDMATH (Foundation in Math FOUN, 5 units), BASCHEM (Basic Chemistry, 3 units), BASPHYS (Basic Physics, 3 units), FNDSTAT (Foundation in Statistics FOUN, 3 units), GEARTAP (Art Appreciation 2A, 3 units). Total: 17 units. No prerequisites required for First Term.", "metadata": {"term": "1", "type": "overview"}},
    {"id": "term2_overview", "text": "TERM 2 - Second Term courses: NSTPCW1 (National Service Training Program 1 2D, 3 units), GEMATMW (Mathematics in the Modern World 2A, 3 units), CALENG1 (Differential Calculus 1A, 3 units) requires FNDMATH as hard prerequisite, COEDISC (Computer Engineering as a Discipline 1E, 1 unit), PROLOGI (Programming Logic and Design Lecture 1E, 2 units), LBYCPA1 (Programming Logic and Design Laboratory 1E, 2 units) requires PROLOGI as co-requisite, LBYEC2A (Computer Fundamentals and Programming 1, 1 unit), GESTSOC (Science Technology and Society 2A, 3 units), GERIZAL (Life and Works of Rizal 2B, 3 units). Total: 18 units.", "metadata": {"term": "2", "type": "overview"}},
    {"id": "caleng1_prereq", "text": "CALENG1 (Differential Calculus) has FNDMATH as a hard prerequisite. Students must pass FNDMATH before enrolling in CALENG1.", "metadata": {"term": "2", "type": "prerequisite", "course": "CALENG1"}},
    {"id": "lbycpa1_coreq", "text": "LBYCPA1 (Programming Logic and Design Laboratory) requires PROLOGI as a co-requisite. Both PROLOGI and LBYCPA1 must be taken in the same term.", "metadata": {"term": "2", "type": "corequisite", "course": "LBYCPA1"}},
    {"id": "term3_overview", "text": "TERM 3 - Third Term courses: NSTPCW2 (National Service Training Program 2 2D, 3 units) requires NSTPCW1 as hard prerequisite, LCLSONE (Lasallian Studies 1, 1 unit), SAS1000 (Student Affairs Service 1000 LS, 0 units), LASARE1 (Lasallian Recollection 1, 0 units), ENGPHYS (Physics for Engineers 1B, 3 units) requires CALENG1 as soft/hard prerequisite and BASPHYS, LBYPH1A (Physics for Engineers Laboratory 1B, 1 unit) requires ENGPHYS as co-requisite, CALENG2 (Integral Calculus 1A, 3 units) requires CALENG1 as hard prerequisite, LBYCPEI (Object Oriented Programming Laboratory 1E, 2 units) requires PROLOGI as hard prerequisite, GEPCOMM (Purposive Communications 2A, 3 units), LCFAITH (Faith Worth Living, 3 units), GELECSP (Social Science and Philosophy 2B, 3 units). Total: 19 units.", "metadata": {"term": "3", "type": "overview"}},
    {"id": "caleng2_prereq", "text": "CALENG2 (Integral Calculus) requires CALENG1 as a hard prerequisite. Students must pass CALENG1 before enrolling in CALENG2.", "metadata": {"term": "3", "type": "prerequisite", "course": "CALENG2"}},
    {"id": "engphys_prereq", "text": "ENGPHYS (Physics for Engineers) requires CALENG1 as a soft/hard prerequisite and BASPHYS. LBYPH1A (Physics for Engineers Laboratory) requires ENGPHYS as a co-requisite.", "metadata": {"term": "3", "type": "prerequisite", "course": "ENGPHYS"}},
    {"id": "term4_overview", "text": "TERM 4 - Fourth Term courses: CALENG3 (Differential Equations 1A, 3 units) requires CALENG2 as hard prerequisite, DATSRAL (Data Structures and Algorithms Lecture 1E, 1 unit) requires LBYCPEI as hard prerequisite, LBYCPA2 (Data Structures and Algorithms Laboratory 1E, 2 units) requires DATSRAL as co-requisite, DISCRMT (Discrete Mathematics 1E, 3 units) requires CALENG1 as hard prerequisite, FUNDCKT (Fundamentals of Electrical Circuits Lecture 1D, 3 units) requires ENGPHYS as hard prerequisite, LBYEC2M (Fundamentals of Electrical Circuits Lab 1D, 1 unit) requires FUNDCKT as co-requisite, ENGCHEM (Chemistry for Engineers 1B, 3 units) requires BASCHEM as hard prerequisite, LBYCH1A (Chemistry for Engineers Laboratory 1B, 1 unit) requires ENGCHEM as co-requisite, GEFTWEL (Physical Fitness and Wellness 2C, 2 units). Total: 19 units.", "metadata": {"term": "4", "type": "overview"}},
    {"id": "datsral_prereq", "text": "DATSRAL (Data Structures and Algorithms Lecture) requires LBYCPEI as a hard prerequisite. LBYCPA2 (Data Structures and Algorithms Laboratory) requires DATSRAL as a co-requisite.", "metadata": {"term": "4", "type": "prerequisite", "course": "DATSRAL"}},
    {"id": "fundckt_prereq", "text": "FUNDCKT (Fundamentals of Electrical Circuits Lecture) requires ENGPHYS as a hard prerequisite. LBYEC2M (Fundamentals of Electrical Circuits Lab) requires FUNDCKT as a co-requisite.", "metadata": {"term": "4", "type": "prerequisite", "course": "FUNDCKT"}},
    {"id": "term5_overview", "text": "TERM 5 - Fifth Term courses: ENGDATA (Engineering Data Analysis 1A, 3 units) requires CALENG2 or FNDSTAT as soft/hard prerequisite, NUMMETS (Numerical Methods 1E, 3 units) requires CALENG3 as hard prerequisite, FUNDLEC (Fundamentals of Electronic Circuits Lecture 1D, 3 units) requires FUNDCKT as hard prerequisite, LBYCPC2 (Fundamentals of Electronic Circuits Laboratory 1D, 1 unit) requires FUNDLEC as co-requisite, SOFDESG (Software Design Lecture 1E, 3 units) requires LBYCPA2 as hard prerequisite, LBYCPD2 (Software Design Laboratory 1E, 1 unit) requires SOFDESG as co-requisite, ENGENVI (Environmental Science and Engineering, 3 units) requires ENGCHEM as hard prerequisite, GEDANCE (Physical Fitness and Wellness in Dance 2C, 2 units), SAS2000 (Student Affairs Series 2, 0 units). Total: 19 units.", "metadata": {"term": "5", "type": "overview"}},
    {"id": "fundlec_prereq", "text": "FUNDLEC (Fundamentals of Electronic Circuits Lecture) requires FUNDCKT as a hard prerequisite. LBYCPC2 (Fundamentals of Electronic Circuits Laboratory) requires FUNDLEC as a co-requisite.", "metadata": {"term": "5", "type": "prerequisite", "course": "FUNDLEC"}},
    {"id": "sofdesg_prereq", "text": "SOFDESG (Software Design Lecture) requires LBYCPA2 as a hard prerequisite. LBYCPD2 (Software Design Laboratory) requires SOFDESG as a co-requisite.", "metadata": {"term": "5", "type": "prerequisite", "course": "SOFDESG"}},
    {"id": "term6_overview", "text": "TERM 6 - Sixth Term courses: LCLSTWO (Lasallian Studies 2, 1 unit), LASARE2 (Lasallian Recollection 2, 0 units), MXSIGFN (Fundamentals of Mixed Signals and Sensors 1E, 3 units) requires FUNDLEC as hard prerequisite, LOGDSGN (Logic Circuits and Design Lecture 1E, 3 units) requires FUNDLEC as hard prerequisite, LBYCPG3 (Logic Circuits and Design Laboratory 1E, 1 unit) requires LOGDSGN as co-requisite, FDCNSYS (Feedback and Control Systems 1E, 3 units) requires NUMMETS as hard prerequisite, LBYCPC3 (Feedback and Control System Laboratory 1E, 1 unit) requires FDCNSYS as co-requisite, LBYME1C (Computer-Aided Drafting CAD for ECE and CpE 1C, 1 unit), GELACAH (Arts and Humanities 2B, 3 units), GESPORT (Physical Fitness and Wellness in Individual Sports 2C, 2 units). Total: 17 units.", "metadata": {"term": "6", "type": "overview"}},
    {"id": "logdsgn_prereq", "text": "LOGDSGN (Logic Circuits and Design Lecture) requires FUNDLEC as a hard prerequisite. LBYCPG3 (Logic Circuits and Design Laboratory) requires LOGDSGN as a co-requisite.", "metadata": {"term": "6", "type": "prerequisite", "course": "LOGDSGN"}},
    {"id": "fdcnsys_prereq", "text": "FDCNSYS (Feedback and Control Systems) requires NUMMETS as a hard prerequisite. LBYCPC3 (Feedback and Control System Laboratory) requires FDCNSYS as a co-requisite.", "metadata": {"term": "6", "type": "prerequisite", "course": "FDCNSYS"}},
    {"id": "term7_overview", "text": "TERM 7 - Seventh Term courses: GEETHIC (Ethics 2A, 3 units), MICPROS (Microprocessors Lecture 1E, 3 units) requires LOGDSGN as hard prerequisite, LBYCPA3 (Microprocessors Laboratory 1E, 1 unit) requires MICPROS as co-requisite, LBYCPB3 (Computer Engineering Drafting and Design Laboratory 1E, 1 unit) requires FUNDLEC and LOGDSGN as hard prerequisites, LBYEC3B (Intelligent Systems for Engineering, 1 unit) requires LBYEC2A and ENGDATA as hard prerequisites, LBYCPF2 (Introduction to HDL Laboratory 1E, 1 unit) requires FUNDLEC as hard prerequisite, DIGDACM (Data and Digital Communications 1E, 3 units) requires FUNDLEC as hard prerequisite, GETEAMS (Physical Fitness and Wellness in Team Sports 2C, 2 units), LBYCPG2 (Basic Computer Systems Administration, 1 unit). Total: 16 units.", "metadata": {"term": "7", "type": "overview"}},
    {"id": "micpros_prereq", "text": "MICPROS (Microprocessors Lecture) requires LOGDSGN as a hard prerequisite. LBYCPA3 (Microprocessors Laboratory) requires MICPROS as a co-requisite. MICPROS is taken in the Seventh Term.", "metadata": {"term": "7", "type": "prerequisite", "course": "MICPROS"}},
    {"id": "term8_overview", "text": "TERM 8 - Eighth Term courses: CSYSARC (Computer Architecture and Organization Lecture 1E, 3 units) requires MICPROS as hard prerequisite, LBYCPD3 (Computer Architecture and Organization Laboratory 1E, 1 unit) requires CSYSARC as co-requisite, EMBDSYS (Embedded Systems Lecture 1E, 3 units) requires MICPROS as hard prerequisite, LBYCPM3 (Embedded Systems Laboratory 1E, 1 unit) requires EMBDSYS as co-requisite, GELECST (Science and Technology 2B, 3 units), REMETHS (Methods of Research for CpE 1E, 3 units) requires ENGDATA, GEPCOMM, and LOGDSGN as hard prerequisites, OPESSYS (Operating Systems Lecture 1E, 3 units) requires LBYCPA2 as hard prerequisite, LBYCPO1 (Operating Systems Laboratory 1E, 1 unit) requires OPESSYS as co-requisite. Total: 8 units.", "metadata": {"term": "8", "type": "overview"}},
    {"id": "embdsys_prereq", "text": "EMBDSYS (Embedded Systems Lecture) requires MICPROS as a hard prerequisite. LBYCPM3 (Embedded Systems Laboratory) requires EMBDSYS as a co-requisite.", "metadata": {"term": "8", "type": "prerequisite", "course": "EMBDSYS"}},
    {"id": "csysarc_prereq", "text": "CSYSARC (Computer Architecture and Organization Lecture) requires MICPROS as a hard prerequisite. LBYCPD3 (Computer Architecture and Organization Laboratory) requires CSYSARC as a co-requisite.", "metadata": {"term": "8", "type": "prerequisite", "course": "CSYSARC"}},
    {"id": "remeths_prereq", "text": "REMETHS (Methods of Research for CpE) requires ENGDATA, GEPCOMM, and LOGDSGN as hard prerequisites.", "metadata": {"term": "8", "type": "prerequisite", "course": "REMETHS"}},
    {"id": "term9_overview", "text": "TERM 9 - Ninth Term courses: LCLSTRI (Lasallian Studies 3, 1 unit), LCASEAN (The Filipino and ASEAN, 3 units), LASARE3 (Lasallian Recollection 3, 0 units), DSIGPRO (Digital Signal Processing Lecture 1E, 3 units) requires FDCNSYS and EMBDSYS as hard/soft prerequisite, LBYCPA4 (Digital Signal Processing Laboratory 1E, 1 unit) requires DSIGPRO as co-requisite, OCHESAF (Basic Occupational Health and Safety 1E, 3 units) requires EMBDSYS as hard prerequisite, THSCP4A (CpE Practice and Design 1 1E, 1 unit) requires EMBDSYS and REMETHS as hard prerequisites, CPEPRAC (CpE Laws and Professional Practice 1E, 2 units) requires EMBDSYS as hard prerequisite, CPECOG1 (CpE Elective 1 Lecture 1F, 2 units) requires EMBDSYS and THSCP4A as hard/co prerequisite, LBYCPF3 (CpE Elective 1 Laboratory 1F, 1 unit) requires CPECOG1 as co-requisite. Total: 16 units.", "metadata": {"term": "9", "type": "overview"}},
    {"id": "thscp4a_prereq", "text": "THSCP4A (CpE Practice and Design 1) requires both EMBDSYS and REMETHS as hard prerequisites. This is a capstone/thesis preparation course taken in Term 9.", "metadata": {"term": "9", "type": "prerequisite", "course": "THSCP4A"}},
    {"id": "dsigpro_prereq", "text": "DSIGPRO (Digital Signal Processing Lecture) requires FDCNSYS as a hard prerequisite and EMBDSYS as a soft prerequisite. LBYCPA4 (Digital Signal Processing Laboratory) requires DSIGPRO as a co-requisite.", "metadata": {"term": "9", "type": "prerequisite", "course": "DSIGPRO"}},
    {"id": "term10_overview", "text": "TERM 10 - Tenth Term courses: LCENWRD (Encountering the Word in the World, 3 units), EMERTEC (Emerging Technologies in CpE 1E, 3 units) requires EMBDSYS as hard prerequisite, THSCP4B (CpE Practice and Design 2 1E, 1 unit) requires THSCP4A as hard prerequisite, ENGTREP (Technopreneurship 101 1C, 3 units) requires EMBDSYS as hard prerequisite, CONETSC (Computer Networks and Security Lecture 1E, 3 units) requires DIGDACM as hard prerequisite, LBYCPB4 (Computer Networks and Security Laboratory 1E, 1 unit) requires CONETSC as co-requisite, CPECAPS (Operational Technologies, 2 units) requires LBYCPB3 and LBYCPB4 as co/co requisite, CPECOG2 (CpE Elective 2 Lecture 1F, 2 units) requires THSCP4A as soft prerequisite, LBYCPH3 (CpE Elective 2 Laboratory 1F, 1 unit) requires CPECOG2 as co-requisite, SAS3000 (Student Affairs Series 3, 0 units) requires SAS2000 as hard prerequisite.", "metadata": {"term": "10", "type": "overview"}},
    {"id": "thscp4b_prereq", "text": "THSCP4B (CpE Practice and Design 2) requires THSCP4A as a hard prerequisite. This is the second part of the capstone/thesis sequence taken in Term 10.", "metadata": {"term": "10", "type": "prerequisite", "course": "THSCP4B"}},
    {"id": "conetsc_prereq", "text": "CONETSC (Computer Networks and Security Lecture) requires DIGDACM as a hard prerequisite. LBYCPB4 (Computer Networks and Security Laboratory) requires CONETSC as a co-requisite.", "metadata": {"term": "10", "type": "prerequisite", "course": "CONETSC"}},
    {"id": "term11_overview", "text": "TERM 11 - Eleventh Term: PRCGECP (Practicum for CpE 1E, 3 units) requires REMETHS as hard prerequisite. Total: 3 units. This is the practicum/internship term.", "metadata": {"term": "11", "type": "overview"}},
    {"id": "term12_overview", "text": "TERM 12 - Twelfth Term courses: GERPHIS (Readings in the Philippine History 2A, 3 units), GEWORLD (The Contemporary World 2A, 3 units), THSCP4C (CpE Practice and Design 3 1E, 1 unit) requires THSCP4B as hard prerequisite, CPECOG3 (CpE Elective 3 Lecture 1F, 2 units) requires THSCP4A as soft prerequisite, LBYCPC4 (CpE Elective 3 Laboratory 1F, 1 unit) requires CPECOG3 as co-requisite, CPETRIP (Seminars and Field Trips for CpE 1E, 1 unit) requires EMBDSYS and CPECAPS as hard prerequisites, ECNOMIC (Engineering Economics for CpE 1C, 3 units) requires CALENG1 as soft prerequisite, ENGMANA (Engineering Management, 2 units) requires CALENG1 as soft prerequisite, GEUSELF (Understanding the Self 2A, 3 units). Total: 19 units.", "metadata": {"term": "12", "type": "overview"}},
    {"id": "thscp4c_prereq", "text": "THSCP4C (CpE Practice and Design 3) requires THSCP4B as a hard prerequisite. The full thesis sequence is THSCP4A (Term 9) then THSCP4B (Term 10) then THSCP4C (Term 12).", "metadata": {"term": "12", "type": "prerequisite", "course": "THSCP4C"}},
    {"id": "prereq_legend", "text": "Prerequisite Legend: H = Hard Pre-Requisite (must be passed before enrolling), S = Soft Pre-Requisite (should be enrolled before, not needed to be passed; not following will cause the course to be INVALIDATED), C = Co-Requisite (must be taken in the same term). This checklist is for freshmen who started AY 2022-2023.", "metadata": {"term": "all", "type": "policy"}},
    {"id": "checklist_warning", "text": "Students should not enroll without passing their respective hard prerequisites. Students may still proceed to a course even if they fail its soft prerequisite, as long as they enrolled in it. If a student does not take the soft prerequisite at all, the subsequent course will be INVALIDATED. This checklist is tentative and subject to change.", "metadata": {"term": "all", "type": "policy"}},
    {"id": "thesis_sequence", "text": "The CpE thesis/capstone sequence is: THSCP4A (Term 9, requires EMBDSYS and REMETHS) then THSCP4B (Term 10, requires THSCP4A) then THSCP4C (Term 12, requires THSCP4B). Students must complete this sequence to graduate.", "metadata": {"term": "all", "type": "policy"}},
    {"id": "nstp_sequence", "text": "The NSTP sequence is: NSTP101 (Term 1, General Orientation) then NSTPCW1 (Term 2) then NSTPCW2 (Term 3, requires NSTPCW1 as hard prerequisite).", "metadata": {"term": "all", "type": "policy"}},
    {"id": "lasallian_sequence", "text": "The Lasallian Studies sequence is: LCLSONE (Term 3) then LCLSTWO (Term 6) then LCLSTRI (Term 9). Lasallian Recollections: LASARE1 (Term 3), LASARE2 (Term 6), LASARE3 (Term 9).", "metadata": {"term": "all", "type": "policy"}},
    {"id": "sas_sequence", "text": "Student Affairs Series: SAS1000 (Term 3, 0 units) then SAS2000 (Term 5, 0 units) then SAS3000 (Term 10, 0 units, requires SAS2000 as hard prerequisite).", "metadata": {"term": "all", "type": "policy"}},
]

# ── Build vector store ─────────────────────────────────────────────────────
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

# ── BM25 index ─────────────────────────────────────────────────────────────
tokenized  = [doc["text"].lower().split() for doc in DOCUMENTS]
bm25_index = BM25Okapi(tokenized)

# ── Cross-encoder reranker ─────────────────────────────────────────────────
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

def retrieve_hybrid(query: str, top_k: int = 6) -> str:
    dense_results = collection.query(
        query_texts=[query],
        n_results=top_k * 2,
        include=["documents", "distances"]
    )
    dense_docs  = dense_results["documents"][0]
    dense_dists = dense_results["distances"][0]
    max_dist    = max(dense_dists) if dense_dists else 1
    dense_scores = {
        doc: 1 - (dist / max_dist)
        for doc, dist in zip(dense_docs, dense_dists)
    }
    tokenized_query = query.lower().split()
    bm25_raw        = bm25_index.get_scores(tokenized_query)
    max_bm25        = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_scores     = {DOCUMENTS[i]["text"]: bm25_raw[i] / max_bm25 for i in range(len(DOCUMENTS))}
    all_docs = set(dense_scores.keys()) | set(bm25_scores.keys())
    combined = {
        doc: 0.5 * dense_scores.get(doc, 0) + 0.5 * bm25_scores.get(doc, 0)
        for doc in all_docs
    }
    top_docs = sorted(combined.items(), key=lambda x: -x[1])[:top_k]
    return [doc for doc, _ in top_docs]

def rerank(query: str, chunks: list, top_k: int = 4) -> list:
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

# ── Flask routes ───────────────────────────────────────────────────────────
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
