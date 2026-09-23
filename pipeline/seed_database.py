# ============================================================
# DLSU CpE AI Academic Advising System
# Database Seeder
# ============================================================
# Populates the MySQL database from parsed source data.
#
# Tables seeded and their sources:
#   curricula           <- checklist_rows.json (one per program+batch)
#   courses             <- checklist_rows.json (unique course codes)
#   curriculum_courses  <- checklist_rows.json (course placements)
#   prerequisites       <- checklist_rows.json (H/S/C rules)
#   documents           <- policy_sections.json + source_data.py
#   faq_items           <- dataset_train.xlsx + source_data.FAQ_LIST
#   embedding_meta      <- policy_sections.json + source_data.py
#
# Tables NOT seeded here (populated at runtime by the system):
#   users, students, advisers   <- created when users register
#   enrollments                 <- created when students log grades
#   advising_sessions           <- created per advising interaction
#   advising_queries            <- created per student question
#   advising_responses          <- created per AI/adviser answer
#   plan_courses                <- created per advising session plan
#   metric_runs, metric_results <- created during SO1-SO5 evaluation
#
# Run:
#   pip install mysql-connector-python openpyxl pandas
#   python seed_database.py
#   python seed_database.py --port 3307
#   python seed_database.py --port 3307 --clear   # wipe and reseed
# ============================================================

import uuid
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import mysql.connector
from mysql.connector import Error

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_HOST     = "localhost"
DEFAULT_PORT     = 3307          # XAMPP MariaDB
DEFAULT_USER     = "root"
DEFAULT_PASSWORD = ""
DEFAULT_DB       = "dlsu_cpe_advising"

PARSED_DIR   = Path("./parsed_data")
DATASET_PATH = Path("./dataset_train.xlsx")  # train split only — run dataset_split.py first

# Embedding model confirmed by experiment
EMBEDDING_MODEL = "intfloat/e5-small-v2"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection(host, port, user, password, database):
    conn = mysql.connector.connect(
        host=host, port=port, user=user,
        password=password, database=database,
        charset="utf8mb4",
    )
    conn.autocommit = False
    return conn


# ============================================================
# HELPERS
# ============================================================

def new_uuid() -> str:
    return str(uuid.uuid4())


def batch_insert(cursor, table: str, columns: List[str], rows: List[tuple],
                 batch_size: int = 500) -> int:
    """Insert rows in batches. Returns total inserted count."""
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(columns))
    col_str      = ", ".join(columns)
    sql          = f"INSERT IGNORE INTO {table} ({col_str}) VALUES ({placeholders})"
    total        = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        cursor.executemany(sql, batch)
        total += cursor.rowcount
    return total


# ============================================================
# SEEDER 1: CURRICULA
# One row per unique (program, batch_id) combination.
# ============================================================

def seed_curricula(cursor, rows: List[Dict]) -> Dict[str, str]:
    """
    Seed the curricula table.
    Returns a dict mapping (program, batch_id) -> curriculum_id.
    """
    seen    = {}
    inserts = []

    for row in rows:
        key = (row["program"], row["batch_id"])
        if key in seen:
            continue
        cid = new_uuid()
        seen[key] = cid
        ay         = row.get("academic_year", "")
        start_year = int(ay.split("-")[0]) if ay and "-" in ay else row["batch_id"] + 1900
        inserts.append((
            cid,
            row["program"],
            row["batch_id"],
            ay,
            start_year,
            None,         # end_year NULL — still active
            None,         # description
        ))

    count = batch_insert(cursor, "curricula",
        ["curriculum_id","program","batch_id","academic_year",
         "start_year","end_year","description"],
        inserts)
    print(f"  [curricula]           {count:>5} rows inserted  "
          f"({len(seen)} unique curricula)")
    return seen


# ============================================================
# SEEDER 2: COURSES
# One row per unique course_code across all programs.
# ============================================================

def seed_courses(cursor, rows: List[Dict]) -> Dict[str, str]:
    """
    Seed the courses table.
    Returns a dict mapping course_code -> course_id.
    """
    seen    = {}
    inserts = []

    for row in rows:
        code = row["course_code"]
        if code in seen:
            continue
        cid = new_uuid()
        seen[code] = cid
        inserts.append((
            cid,
            code,
            row.get("title", ""),
            row.get("units", 3),
            None,         # description
        ))

    count = batch_insert(cursor, "courses",
        ["course_id","course_code","title","units","description"],
        inserts)
    print(f"  [courses]             {count:>5} rows inserted  "
          f"({len(seen)} unique courses)")
    return seen


# ============================================================
# SEEDER 3: CURRICULUM_COURSES
# Maps each course to its term position in a specific curriculum.
# ============================================================

def seed_curriculum_courses(cursor, rows: List[Dict],
                             curriculum_map: Dict, course_map: Dict) -> int:
    """Seed the curriculum_courses junction table."""
    inserts = []
    seen    = set()

    for row in rows:
        curr_key = (row["program"], row["batch_id"])
        curr_id  = curriculum_map.get(curr_key)
        course_id = course_map.get(row["course_code"])

        if not curr_id or not course_id:
            continue

        key = (curr_id, course_id)
        if key in seen:
            continue
        seen.add(key)

        inserts.append((
            new_uuid(),
            curr_id,
            course_id,
            row.get("year_level") or 1,
            row.get("term_number") or 1,
            row.get("term_name", ""),
            "major",      # default category — checklist doesn't encode GE separately
        ))

    count = batch_insert(cursor, "curriculum_courses",
        ["cc_id","curriculum_id","course_id","year_level",
         "term_number","term_name","category"],
        inserts)
    print(f"  [curriculum_courses]  {count:>5} rows inserted")
    return count


# ============================================================
# SEEDER 4: PREREQUISITES
# Encodes H (hard), S (soft), C (co-requisite) rules.
# ============================================================

def seed_prerequisites(cursor, rows: List[Dict],
                        course_map: Dict) -> int:
    """Seed the prerequisites table."""
    inserts = []
    seen    = set()

    for row in rows:
        course_id = course_map.get(row["course_code"])
        if not course_id:
            continue

        for prereq in row.get("prerequisites", []):
            req_code = prereq.get("course_code", "").strip()
            req_type = prereq.get("type", "unknown")

            # Skip standing requirements and unknown codes
            if req_type == "standing" or not req_code:
                continue

            req_id = course_map.get(req_code)
            if not req_id:
                # Prerequisite course not in our corpus — skip
                continue

            key = (course_id, req_id, req_type)
            if key in seen:
                continue
            seen.add(key)

            inserts.append((
                new_uuid(),
                course_id,
                req_id,
                req_type,
                None,     # min_grade not specified in checklist
            ))

    count = batch_insert(cursor, "prerequisites",
        ["prereq_id","course_id","required_course_id","prereq_type","min_grade"],
        inserts)
    print(f"  [prerequisites]       {count:>5} rows inserted")
    return count


# ============================================================
# SEEDER 5: DOCUMENTS
# One row per policy file + one per handbook section.
# Returns doc_type -> document_id mapping for faq_items and
# embedding_meta.
# ============================================================

def seed_documents(cursor, policy_sections: List[Dict]) -> Dict[str, str]:
    """
    Seed the documents table from policy sections.
    Returns a dict mapping doc_type -> document_id.
    """
    inserts  = []
    type_map = {}   # doc_type -> document_id

    for section in policy_sections:
        doc_type = section.get("doc_type", "policy")
        if doc_type in type_map:
            # Multiple sections of same doc_type share one document record
            continue

        did = new_uuid()
        type_map[doc_type] = did
        inserts.append((
            did,
            section.get("title", doc_type),
            doc_type,
            section.get("source", "DLSU GCOE"),
            None,         # file_path
            "2024",       # version
            None,         # effective_date
        ))

    count = batch_insert(cursor, "documents",
        ["document_id","title","doc_type","source",
         "file_path","version","effective_date"],
        inserts)
    print(f"  [documents]           {count:>5} rows inserted  "
          f"({len(type_map)} unique doc types)")
    return type_map


# ============================================================
# SEEDER 6: FAQ_ITEMS
# Seeds from two sources:
#   a. dataset_train.xlsx (train split only — run dataset_split.py first)
#   b. source_data.FAQ_LIST  (13 adviser-written FAQs)
# ============================================================

def seed_faq_items(cursor, dataset_path: Path,
                   doc_type_map: Dict) -> int:
    """Seed the faq_items table."""
    inserts = []

    # --- Source A: dataset_train.xlsx ---
    if dataset_path.exists():
        df = pd.read_excel(dataset_path)
        # Normalise column names — robust to old and new dataset layouts
        df.columns = [str(c).strip() for c in df.columns]
        cat_col  = "category"        if "category"        in df.columns else None
        prog_col = "program"         if "program"         in df.columns else None
        # New dataset uses source_doc_title; old used source_file
        src_col  = "source_doc_title" if "source_doc_title" in df.columns else \
                   "source_file"      if "source_file"      in df.columns else None
        for _, row in df.iterrows():
            q = row.get("question", "")
            a = row.get("answer", "")
            if not q or not a or str(q).strip() == "nan" or str(a).strip() == "nan":
                continue
            src    = str(row[src_col]) if src_col else ""
            doc_id = None
            for key, did in doc_type_map.items():
                if key.lower() in src.lower() or src.lower() in key.lower():
                    doc_id = did
                    break
            inserts.append((
                new_uuid(),
                doc_id,
                str(q).strip(),
                str(a).strip(),
                str(row[cat_col])  if cat_col  else "General",
                str(row[prog_col]) if prog_col else "GENERAL",
                "basic",
                True,
                "dataset_train",
            ))
    else:
        print(f"  [WARN] {dataset_path} not found — skipping dataset FAQs")

    # --- Source B: source_data.FAQ_LIST ---
    try:
        sys.path.insert(0, str(Path(".").resolve()))
        from source_data import FAQ_LIST
        for faq in FAQ_LIST:
            if not faq.get("verified", True):
                continue
            inserts.append((
                new_uuid(),
                None,       # no linked document for hand-written FAQs
                faq["question"],
                faq["answer"],
                faq.get("category", "General"),
                faq.get("program", "GENERAL"),
                faq.get("difficulty", "basic"),
                True,
                "source_data",
            ))
    except ImportError:
        print("  [WARN] source_data.py not found — skipping hand-written FAQs")

    count = batch_insert(cursor, "faq_items",
        ["faq_id","document_id","question","answer","category",
         "program","difficulty","verified","created_by"],
        inserts)
    print(f"  [faq_items]           {count:>5} rows inserted  "
          f"({len(inserts)} total FAQs)")
    return count


# ============================================================
# SEEDER 7: EMBEDDING_META
# One row per chunk that will be stored in ChromaDB.
# Tracks source document, chunk position, and model used.
# ============================================================

def seed_embedding_meta(cursor, policy_sections: List[Dict],
                         doc_type_map: Dict) -> int:
    """
    Seed the embedding_meta table with chunk metadata.
    Uses the same 256-char chunking logic as the corpus builder
    so chunk counts match what ChromaDB will actually store.
    """
    import re
    inserts = []

    for section in policy_sections:
        doc_type = section.get("doc_type", "policy")
        doc_id   = doc_type_map.get(doc_type)
        if not doc_id:
            continue

        text  = section.get("text", "").strip()
        title = section.get("title", doc_type)

        # Split into 256-char chunks (matches corpus builder logic)
        paras  = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        chunks = []
        buf    = ""
        for para in paras:
            if len(buf) + len(para) < 256:
                buf = (buf + "\n\n" + para).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = para
        if buf:
            chunks.append(buf)

        total = len(chunks)
        for i, chunk in enumerate(chunks):
            # Use section title + chunk index as identifier
            section_id = f"{title[:50]}_chunk_{i}"
            # ChromaDB reference format: collection/doc_type/chunk_index
            chroma_ref = f"policies/{doc_type}/{i}"

            inserts.append((
                new_uuid(),
                doc_id,
                section_id,
                i,
                total,
                EMBEDDING_MODEL,
                chroma_ref,
            ))

    count = batch_insert(cursor, "embedding_meta",
        ["embedding_id","document_id","section_identifier",
         "chunk_index","chunk_total","model_name","chromadb_ref"],
        inserts)
    print(f"  [embedding_meta]      {count:>5} rows inserted  "
          f"({count} policy chunks tracked)")
    return count


# ============================================================
# CLEAR TABLES (for --clear flag)
# Clears seeded tables in reverse FK order.
# ============================================================

def clear_seeded_tables(cursor) -> None:
    """Delete all seeded data in reverse foreign key order."""
    tables = [
        "embedding_meta",
        "faq_items",
        "documents",
        "prerequisites",
        "curriculum_courses",
        "courses",
        "curricula",
    ]
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    for table in tables:
        cursor.execute(f"DELETE FROM {table}")
        print(f"  [cleared] {table}")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
    print("  All seeded tables cleared.")


# ============================================================
# VERIFY SEEDED DATA
# ============================================================

def verify_seed(cursor) -> None:
    """Print row counts for all seeded tables."""
    tables = [
        "curricula", "courses", "curriculum_courses",
        "prerequisites", "documents", "faq_items", "embedding_meta",
    ]
    print("\n[Verification] Row counts:")
    all_ok = True
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        status = "✓" if count > 0 else "✗ EMPTY"
        print(f"  {status}  {table:<25} {count:>6} rows")
        if count == 0:
            all_ok = False

    if all_ok:
        print("\n[OK] All tables seeded successfully.")
    else:
        print("\n[WARN] Some tables are empty — check warnings above.")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Seed the DLSU CpE advising database."
    )
    parser.add_argument("--host",       default=DEFAULT_HOST)
    parser.add_argument("--port",       default=DEFAULT_PORT, type=int)
    parser.add_argument("--user",       default=DEFAULT_USER)
    parser.add_argument("--password",   default=DEFAULT_PASSWORD)
    parser.add_argument("--database",   default=DEFAULT_DB)
    parser.add_argument("--parsed-dir", default=str(PARSED_DIR), type=Path)
    parser.add_argument("--dataset",    default=str(DATASET_PATH), type=Path)
    parser.add_argument("--clear",      action="store_true",
                        help="Clear seeded tables before inserting")
    args = parser.parse_args()

    print("=" * 55)
    print("DLSU CpE Advising System — Database Seeder")
    print("=" * 55)
    print(f"Host      : {args.host}:{args.port}")
    print(f"Database  : {args.database}")
    print(f"Parsed dir: {args.parsed_dir.resolve()}")
    print(f"Dataset   : {args.dataset.resolve()}")
    print("=" * 55)

    # Load parsed data
    checklist_path = args.parsed_dir / "checklist_rows.json"
    policy_path    = args.parsed_dir / "policy_sections.json"

    if not checklist_path.exists():
        print(f"[ERROR] {checklist_path} not found. Run batch_parser.py first.")
        sys.exit(1)
    if not policy_path.exists():
        print(f"[ERROR] {policy_path} not found. Run batch_parser.py first.")
        sys.exit(1)

    print("\nLoading parsed data...")
    checklist_rows   = json.load(open(checklist_path, encoding="utf-8"))
    policy_sections  = json.load(open(policy_path,    encoding="utf-8"))
    print(f"  {len(checklist_rows)} course rows loaded")
    print(f"  {len(policy_sections)} policy sections loaded")

    try:
        conn   = get_connection(
            args.host, args.port, args.user, args.password, args.database
        )
        cursor = conn.cursor()
        print(f"\n[OK] Connected to {args.database} at {args.host}:{args.port}")

        if args.clear:
            print("\nClearing existing seeded data...")
            clear_seeded_tables(cursor)
            conn.commit()

        print("\nSeeding tables...")

        # Seed in dependency order
        curriculum_map = seed_curricula(cursor, checklist_rows)
        conn.commit()

        course_map = seed_courses(cursor, checklist_rows)
        conn.commit()

        seed_curriculum_courses(cursor, checklist_rows, curriculum_map, course_map)
        conn.commit()

        seed_prerequisites(cursor, checklist_rows, course_map)
        conn.commit()

        doc_type_map = seed_documents(cursor, policy_sections)
        conn.commit()

        seed_faq_items(cursor, args.dataset, doc_type_map)
        conn.commit()

        seed_embedding_meta(cursor, policy_sections, doc_type_map)
        conn.commit()

        # Verify
        verify_seed(cursor)

        cursor.close()
        conn.close()
        print("\n[Done] Database seeded successfully.")

    except Error as e:
        print(f"\n[ERROR] {e}")
        print("Make sure XAMPP MySQL is running and setup_database.py was run first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
