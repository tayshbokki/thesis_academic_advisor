# DLSU CpE AI Academic Advising System
### A Retrieval-Augmented Generation (RAG) System for Academic Advising
**De La Salle University — Computer Engineering & Electronics Engineering Programs**

> Thesis project evaluating three systems — No-RAG Baseline, Naive RAG, and Improved RAG — against a dataset of 1,093 academic advising Q&A pairs across 23 question categories.

---

## Repository Structure

```
thesis_academic_advisor/
│
├── Data & Evaluation Dataset
│   ├── dataset-query.xlsx           # Full dataset — 1,093 Q&A pairs, 23 categories
│   ├── dataset_train.xlsx           # Training split — 874 Q&A pairs (80%)
│   ├── dataset_test.xlsx            # Test split — 219 Q&A pairs (20%)
│   ├── parse_report.json            # Parser output summary (17 checklists, 4 policy docs)
│   └── results_summary.json         # Embedding experiment benchmark results
│
├── Pipeline (run in order)
│   ├── source_data.py               # Step 0 — Raw handbook text & curated FAQs
│   ├── batch_parser.py              # Step 1 — Parse PDFs/DOCX → JSON
│   ├── chunking_pipeline.py         # Step 2 — Chunk & ingest into ChromaDB
│   ├── embedding_experiment.py      # Step 3 — Benchmark 5 embedding models
│   ├── setup_database.py            # Step 4a — Create MySQL schema
│   └── seed_database.py             # Step 4b — Populate MySQL from parsed data
│
├── Evaluation Systems
│   ├── standard_baseline.py         # System 1 — No-RAG baseline
│   ├── naive_rag_baseline.py        # System 2 — Naive RAG (dense retrieval only)
│   └── improved_rag.py              # System 3 — Improved RAG (hybrid + reranking + SQL)
│
└── Thesis Document
    └── AISL-1-2526-C1-THSCP4B.pdf  # Thesis manuscript
```

---

## Prerequisites

### API Keys (`.env` file)
```env
HF_TOKEN=your_huggingface_token
GEMINI_API_KEY=your_gemini_key        # optional
OPENAI_API_KEY=your_openai_key        # optional
```

### Python Dependencies
```bash
pip install openpyxl nltk rouge-score bert-score huggingface-hub chromadb
pip install sentence-transformers langchain langchain-chroma langchain-huggingface
pip install pdfplumber python-docx rank-bm25 mysql-connector-python pandas numpy
```

### Infrastructure
- **ChromaDB** — local vector store, auto-created at `./chroma_store/`
- **XAMPP MySQL** — running on `localhost:3307`, database `dlsu_cpe_advising`, user `root`, no password

---

## File-by-File Reference

---

### `dataset-query.xlsx` _(replaces `advising_dataset.xlsx`)_
The full evaluation dataset — ground truth for all three evaluation systems. Contains **1,093 Q&A pairs** spanning **23 categories** for CpE, ECE, EES, and GENERAL programs. This is the combined pool from which the train/test split was derived.

| Column | Description |
|--------|-------------|
| `qa_id` | Unique question ID (currently unpopulated — all `None`) |
| `question` | Student question |
| `answer` | Ground-truth adviser answer |
| `category` | Question type (prerequisite, eligibility_check, ojt_policy, etc.) |
| `program` | `CPE`, `ECE`, `EES`, or `GENERAL` |
| `term_no` | Term the question applies to |
| `difficulty` | `basic` or `complex` |
| `keywords` | Relevant keywords for relevance judgment |
| `source_section` | Section within the source document |
| `source_doc_title` | Source document the answer comes from |
| `created_by` | Author of the Q&A entry |
| `verified` | Whether the entry has been verified |
| `created_at` / `updated_at` | Timestamps (currently unpopulated) |

**Category breakdown (top 10):** prerequisite (340), eligibility_check (172), corequisite (120), grading_policy (62), prerequisite_lookup (56), discipline_policy (47), ojt_policy (44), term_plan (42), attendance_policy (40), withdrawal_policy (27)

---

### `dataset_train.xlsx`
Training split — **874 Q&A pairs** (80% of the full dataset).

| Program | Count | Difficulty | Count |
|---------|-------|------------|-------|
| CPE | 296 | basic | 752 (86%) |
| ECE | 187 | complex | 122 (14%) |
| EES | 136 | | |
| GENERAL | 255 | | |

Covers 23 categories. Used as the primary training/development set for the RAG systems.

---

### `dataset_test.xlsx`
Test split — **219 Q&A pairs** (20% of the full dataset).

| Program | Count | Difficulty | Count |
|---------|-------|------------|-------|
| CPE | 71 | basic | 185 (84%) |
| ECE | 61 | complex | 34 (16%) |
| EES | 20 | | |
| GENERAL | 67 | | |

Covers 21 categories (two rare categories — `program_info`, `student_policy` — fell entirely into train). Used as the held-out evaluation set for final metric reporting.

---

### `source_data.py`
**Step 0 of the pipeline.** Contains two static data structures that feed directly into the chunking pipeline:

- **`HANDBOOK_SECTIONS`** — Official policy text from the DLSU Student Handbook and official memos, structured as ingestion-ready objects. Each entry has a `doc_type` field used for filtered ChromaDB retrieval (e.g., `retention_policy`, `grading`, `leave_of_absence`). Includes the updated Retention Policy (effective Term 2 AY 2025–2026 per Provost memo).
- **`FAQ_LIST`** — Adviser-written FAQ entries derived from real CpE adviser communications, ready for ingestion via `chunk_and_ingest_faqs()`.

**Usage:**
```python
from source_data import HANDBOOK_SECTIONS, FAQ_LIST
```

---

### `batch_parser.py`
**Step 1 of the pipeline.** Parses all source documents into clean Python objects and saves them as JSON for use in the embedding experiment — _before_ any chunking or ChromaDB ingestion.

**Documents handled:**
- 17 checklist PDFs (CPE ID 118–125, ECE ID 118–125, EES ID 125)
- 4 policy files: `GCOE_UG_OJT_Policy.pdf`, `Guidelines-for-Academic-Advising-2024.pdf`, `GCOE_Academic_Advising_Best_Practices.docx`, `Thesis_Policies_and_Guidelines_latest.docx`
- Handbook sections from `source_data.py`

**Outputs** (saved to `./parsed_data/`):

| File | Contents |
|------|----------|
| `checklist_rows.json` | All course rows across all programs and batches |
| `policy_sections.json` | All policy content as plain-text sections (9 sections total) |
| `parse_report.json` | Summary stats and parse warnings |

**Run:**
```bash
python batch_parser.py
python batch_parser.py --data-dir /your/custom/path
```

> **Known issue:** A repeated heading in the Academic Advising Best Practices section is a `parse_policy_docx()` DOCX parser artifact. The parsed output is otherwise structurally valid.

---

### `parse_report.json`
Output of `batch_parser.py`. Summarizes what was successfully parsed and any warnings.

- **17 checklist files** parsed — 2,234 total course rows extracted
- **4 policy files** parsed — 9 policy sections produced; one duplicate PDF detected and skipped
- **3 warnings** total: one ECE ID 124 page with no tables, one advising guidelines page with no extractable text, one duplicate PDF skip

Use this file to verify your source documents were parsed correctly before running the embedding experiment or chunking pipeline.

---

### `chunking_pipeline.py`
**Step 2 of the pipeline.** Chunks all parsed documents and ingests them into three ChromaDB vector store collections using `intfloat/e5-small-v2` embeddings.

**Collections:**

| Collection | Source | Chunking Strategy |
|------------|--------|-------------------|
| `checklist` | Checklist PDFs | One chunk per course row |
| `policies` | Policy PDFs, DOCX, handbook text | Section-aware recursive character splitting |
| `faqs` | FAQ list from `source_data.py` | One chunk per Q&A pair |

**Key design choices:**
- Embedding model: `intfloat/e5-small-v2` (selected via `embedding_experiment.py`)
- E5 prefix convention: `"passage: "` prepended to all corpus documents; `"query: "` prepended to all search queries
- ChromaDB persist directory: `./chroma_store/`

**Ingestion functions:**

| Function | Purpose |
|----------|---------|
| `chunk_and_ingest_checklist(pdf_path)` | Parses checklist PDF → ChromaDB |
| `chunk_and_ingest_policy(file_path)` | Parses policy PDF or DOCX → ChromaDB |
| `chunk_and_ingest_policy_text(**section)` | Ingests pre-extracted policy text → ChromaDB |
| `chunk_and_ingest_faqs(faq_list)` | Ingests FAQ list → ChromaDB |

---

### `embedding_experiment.py`
**Step 3 of the pipeline.** Benchmarks 5 embedding models on the actual parsed corpus using stratified sampling across all 21 question categories.

**Models benchmarked:**

| Model | Params | Notes |
|-------|--------|-------|
| `all-MiniLM-L6-v2` | 22M | Fast lightweight baseline |
| `all-mpnet-base-v2` | 110M | Strong general-purpose |
| `multi-qa-mpnet-base-dot-v1` | 110M | Trained for Q&A retrieval |
| `e5-small-v2` | 33M | **Selected** — best composite score |
| `e5-base-v2` | 109M | Stronger but slower |

**Metrics:** Recall@K (K=1,3,5,10), MRR, NDCG@10, cosine gap, embedding speed

**Outputs** (saved to `./embedding_experiment/`):
- `results_raw.json` — per-query per-model results
- `results_summary.json` — aggregated metrics per model
- `results_report.txt` — human-readable comparison table

**Run:**
```bash
python embedding_experiment.py
python embedding_experiment.py --sample-size 0   # use all 1093 queries
```

> **Note on zero scores:** Categories `ojt_policy` and `curriculum_summary` showed zero retrieval scores across all five models. This is a data pipeline issue — source documents either absent or embedded as single oversized vectors — not a model deficiency. This is consistent methodology across all models.

---

### `setup_database.py`
**Step 4a.** Creates the `dlsu_cpe_advising` MySQL database and applies `dlsu_cpe_schema.sql`.

**Run:**
```bash
python setup_database.py
python setup_database.py --password yourpassword
python setup_database.py --host localhost --port 3306
```

Defaults: `localhost:3306`, user `root`, no password. Change `--port 3307` if using XAMPP MariaDB on a non-default port.

---

### `seed_database.py`
**Step 4b.** Populates the MySQL database from all parsed source data.

**Tables seeded:**

| Table | Source |
|-------|--------|
| `curricula` | `checklist_rows.json` |
| `courses` | `checklist_rows.json` (unique course codes) |
| `curriculum_courses` | `checklist_rows.json` (course placements per term) |
| `prerequisites` | `checklist_rows.json` (H/S/C prerequisite rules) |
| `documents` | `policy_sections.json` + `source_data.py` |
| `faq_items` | `dataset-query.xlsx` + `source_data.FAQ_LIST` |
| `embedding_meta` | Metadata for all embedded documents |

Tables not seeded here (populated at runtime): `users`, `students`, `advisers`, `enrollments`, `advising_sessions`, `advising_queries`, `advising_responses`, `plan_courses`, `metric_runs`, `metric_results`.

**Run:**
```bash
python seed_database.py
python seed_database.py --port 3307
python seed_database.py --port 3307 --clear   # wipe and reseed
```

---

### `standard_baseline.py`
**Evaluation System 1 — No-RAG Baseline.**

Tests LLMs on DLSU CpE advising queries using _no retrieval_ — pure parametric knowledge only. Establishes the lower-bound baseline that Naive RAG and Improved RAG are compared against.

**Test set:** All 219 Q&A pairs from `dataset_test.xlsx` (held-out split)

**Models tested:**

| Label | Model ID | Provider |
|-------|----------|----------|
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | HF |
| DeepSeek-R1-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | HF |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | HF |
| Gemma-2-9B | `google/gemma-2-9b-it` | featherless-ai via HF |
| Gemini-2.5-Flash-Lite | `gemini-2.5-flash-lite` | Gemini |
| GPT-4o-mini | `gpt-4o-mini` | OpenAI |

**Parameter configurations (8 total):** Temperature × max_tokens grid (t ∈ {0.0, 0.1, 0.3} × tokens ∈ {200, 400}) plus top_p variations (p=0.9 at t=0.1).

**Metrics (SO3):** ROUGE-1, ROUGE-L, BLEU, METEOR, BERTScore (`roberta-large`, batched), hallucination rate, response time compliance (<5s threshold)

**Output:** `no_rag_baseline_results.json`

**Run:**
```bash
python standard_baseline.py
```

---

### `naive_rag_baseline.py`
**Evaluation System 2 — Naive RAG Baseline.**

Adds dense semantic retrieval to generation. Query is embedded → cosine search against ChromaDB → top-k chunks prepended as context → LLM generates answer. No BM25, no reranking, no hybrid fusion.

**Retrieval:** All three ChromaDB collections (`checklist`, `policies`, `faqs`), top_k tested as a variable (k=3, 5, 10). Test set: 219 Q&A pairs from `dataset_test.xlsx`.

**Models & parameter configs:** Same as `standard_baseline.py`

**Metrics:**
- **SO1 (Retrieval):** Precision@K, Recall@K, MRR, NDCG@10, cosine similarity
- **SO3 (Generation):** ROUGE-1, ROUGE-L, BLEU, METEOR, BERTScore, hallucination rate, retrieval time, response time

**Output:** `naive_rag_results.json`

**Run:**
```bash
# Requires ChromaDB to be populated first (chunking_pipeline.py)
python naive_rag_baseline.py
```

---

### `improved_rag.py`
**Evaluation System 3 — Improved RAG.**

Two-phase evaluation architecture with hybrid retrieval, cross-encoder reranking, SQL routing, and citation tracking.

**Phase 1 — Retrieval Only (local, no API calls)**
Tests 7 retrieval configurations across all 1,093 queries (full dataset-query.xlsx). Best config selected by SO1 metrics.

| Config variable | Options tested |
|----------------|---------------|
| BM25 + dense fusion (alpha weighting) | α ∈ {0.3, 0.5, 0.7} |
| Cross-encoder reranking | on / off |
| Collections searched | all three |

- BM25 model: Okapi BM25 (`rank-bm25`)
- Dense model: `intfloat/e5-small-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Phase 2 — Full Generation (uses best Phase 1 config)**
Tests 4 models × 8 generation configs using the winning retrieval config, with:
- **SO2:** SQL routing via intent detection — structured course/prereq queries routed to MySQL; unstructured queries to vector search
- **SO3:** Citation tracking in system prompt + post-generation verification
- Document timestamp logging via `documents` table

**SQL connection:** `localhost:3307`, `dlsu_cpe_advising`, `root`/no password

**Outputs:**
- `improved_rag_phase1_results.json` — retrieval config benchmark
- `improved_rag_phase2_results.json` — full generation results

**Run:**
```bash
python improved_rag.py              # both phases
python improved_rag.py --phase 1   # retrieval only (no API keys needed)
python improved_rag.py --phase 2   # generation only (needs Phase 1 results)
```

---

### `results_summary.json`
Output of `embedding_experiment.py`. Contains aggregated SO1 metrics per embedding model across all question categories, used to justify the selection of `intfloat/e5-small-v2` as the system embedding model.

---

### `AISL-1-2526-C1-THSCP4B.pdf`
The thesis manuscript document.

---

## Running the Full Pipeline

```bash
# 1. Parse all source documents
python batch_parser.py

# 2. Benchmark embedding models (optional — selection already made)
python embedding_experiment.py

# 3. Chunk and ingest into ChromaDB
#    (edit chunking_pipeline.py to call ingestion functions for your source files)
python chunking_pipeline.py

# 4. Set up MySQL database
python setup_database.py --port 3307
python seed_database.py --port 3307

# 5. Run evaluations
python standard_baseline.py
python naive_rag_baseline.py
python improved_rag.py --phase 1   # find best retrieval config first
python improved_rag.py --phase 2   # then run full generation
```

> **Tip:** Run scripts from a standalone PowerShell window (not inside VS Code terminal) to prevent process loss if VS Code crashes during long evaluation runs.

---

## Specific Objectives Mapping

| Objective | Description | Scripts |
|-----------|-------------|---------|
| **SO1** | Retrieval quality — MRR, NDCG@10, Precision@K, Recall@K, cosine similarity | `embedding_experiment.py`, `naive_rag_baseline.py`, `improved_rag.py` |
| **SO2** | SQL integration — structured query routing via intent detection | `improved_rag.py` (Phase 2) |
| **SO3** | Generation quality — ROUGE, BLEU, METEOR, BERTScore, citation tracking | All three evaluation scripts |
| **SO4** | System performance — response time, throughput, <5s compliance | All three evaluation scripts |

---

## Hardware & Environment

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce GTX 1650 Ti (CUDA 12.4) |
| PyTorch | Installed with `cu124` |
| Embedding inference | CPU (switch `device` to `"cuda"` in `chunking_pipeline.py` if desired) |
| Inference provider | Hugging Face serverless + featherless-ai (Gemma-2-9B) |
| Evaluation model | `roberta-large` via `bert_score`, batched |
