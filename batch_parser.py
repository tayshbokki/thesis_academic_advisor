# ============================================================
# DLSU CpE AI Academic Advising System
# Batch Document Parser
# ============================================================
# Parses ALL source documents into clean Python objects.
# Output is saved as JSON for use in the embedding comparison
# experiment BEFORE any chunking or ingestion into ChromaDB.
#
# Documents handled:
#   1. Checklists (17 PDFs)
#      - CPE ID 118-125 CHECKLIST.pdf
#      - ECE ID 118-125 CHECKLIST.pdf
#      - EES ID 125 CHECKLIST.pdf
#
#   2. Policy files (4 files)
#      - GCOE_UG_OJT_Policy.pdf
#      - Guidelines-for-Academic-Advising-2024.pdf
#      - GCOE_Academic_Advising_Best_Practices.docx
#      - Thesis_Policies_and_Guidelines_latest.docx
#
#   3. Handbook sections (from source_data.py)
#      - Already structured, just normalized here
#
# Output files (saved to ./parsed_data/):
#   - checklist_rows.json      : all course rows across all programs/batches
#   - policy_sections.json     : all policy chunks as plain text sections
#   - parse_report.json        : summary stats and any parse warnings
#
# Run:
#   python batch_parser.py
#   python batch_parser.py --data-dir /your/custom/path
# ============================================================

import re
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import pdfplumber                          # pip install pdfplumber
from docx import Document as DocxDocument  # pip install python-docx


# ============================================================
# CONFIGURATION
# Update DATA_DIR to match your actual folder structure:
#
# DATA_DIR/
# ├── CPE CHECKLISTS/
# │   ├── CPE ID 118 CHECKLIST.pdf
# │   └── ...
# ├── ECE CHECKLISTS/
# │   ├── ECE ID 118 CHECKLIST.pdf
# │   └── ...
# ├── EES CHECKLIST/
# │   └── EES ID 125 CHECKLIST.pdf
# └── POLICY DOCUMENTS/
#     ├── GCOE_UG_OJT_Policy.pdf
#     ├── Guidelines-for-Academic-Advising-2024.pdf
#     ├── GCOE_Academic_Advising_Best_Practices.docx
#     └── Thesis_Policies_and_Guidelines_latest.docx
# ============================================================

DATA_DIR   = Path("./data")
OUTPUT_DIR = Path("./parsed_data")


# ============================================================
# FILENAME PARSING
# Extracts program and batch year from checklist filenames.
#
# Supported formats:
#   "CPE ID 118 CHECKLIST.pdf"  -> program="CPE", batch_id=118, ay="2018-2019"
#   "ECE ID 122 CHECKLIST.pdf"  -> program="ECE", batch_id=122, ay="2022-2023"
#   "EES ID 125 CHECKLIST.pdf"  -> program="EES", batch_id=125, ay="2025-2026"
# ============================================================

# Maps batch ID number to academic year string
def batch_id_to_ay(batch_id: int) -> str:
    """
    Convert a DLSU batch ID number to its academic year string.
    ID 118 = AY 2018-2019, ID 119 = AY 2019-2020, etc.
    """
    start = 2000 + (batch_id - 100)
    return f"{start}-{start + 1}"


def parse_checklist_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Extract program, batch_id, and academic_year from a checklist filename.

    Returns None if the filename does not match the expected pattern.
    """
    pattern = re.compile(
        r"^(CPE|ECE|EES|CPES)\s+ID\s+(\d{3})\s+CHECKLIST\.pdf$",
        re.IGNORECASE
    )
    match = pattern.match(filename.strip())
    if not match:
        return None

    program  = match.group(1).upper()
    batch_id = int(match.group(2))
    return {
        "program":       program,
        "batch_id":      batch_id,
        "academic_year": batch_id_to_ay(batch_id),
    }


# ============================================================
# CHECKLIST TABLE PARSING
# Handles the two-column table layout used across all
# CPE, ECE, and EES checklists.
# ============================================================

TERM_MAP = {
    "FIRST TERM":    (1, 1),  "SECOND TERM":  (1, 2),  "THIRD TERM":   (1, 3),
    "FOURTH TERM":   (2, 1),  "FIFTH TERM":   (2, 2),  "SIXTH TERM":   (2, 3),
    "SEVENTH TERM":  (3, 1),  "EIGHTH TERM":  (3, 2),  "NINTH TERM":   (3, 3),
    "TENTH TERM":    (4, 1),  "ELEVENTH TERM":(4, 2),  "TWELFTH TERM": (4, 3),
}

SKIP_PATTERNS = re.compile(
    r"^(COURSE|COURSE TITLE|UNITS|PREREQUISITES|TOTAL|LEGEND|"
    r"Please|This checklist|Chair|GCOE|Prepared|Approved|Noted|"
    r"None|N/A|BS |Bachelor).*",
    re.IGNORECASE
)

COURSE_CODE_PATTERN = re.compile(r"^[A-Z]{2,8}\d*[A-Z]?$")


def parse_prerequisite_string(raw: str) -> List[Dict[str, str]]:
    """
    Parse raw prerequisite cell text into a structured list.

    Handles:
        "H CALENG1"              -> [{"type": "H", "course_code": "CALENG1"}]
        "S/H CALENG1/BASPHYS"    -> [{"type": "S", "course_code": "CALENG1"},
                                      {"type": "H", "course_code": "BASPHYS"}]
        "H FDCNSYS; MICPROS"     -> [{"type": "H", "course_code": "FDCNSYS"},
                                      {"type": "H", "course_code": "MICPROS"}]
        "C DATSTRAL"             -> [{"type": "C", "course_code": "DATSTRAL"}]
        "3rd Yr Standing"        -> [{"type": "standing", "course_code": "3rd Yr Standing"}]
        ""                       -> []
    """
    if not raw or not raw.strip():
        return []

    raw = raw.strip()

    if "standing" in raw.lower():
        return [{"type": "standing", "course_code": raw}]

    prerequisites = []
    groups = [g.strip() for g in raw.split(";")]

    for group in groups:
        type_match = re.match(r"^([A-Z](?:/[A-Z])*)\s+(.+)$", group)
        if type_match:
            types = type_match.group(1).split("/")
            codes = [c.strip() for c in type_match.group(2).split("/")]
            for i, code in enumerate(codes):
                req_type = types[i] if i < len(types) else types[-1]
                if code and COURSE_CODE_PATTERN.match(code):
                    prerequisites.append({
                        "type":        req_type,
                        "course_code": code,
                    })
        else:
            # No type prefix — treat bare course codes as unknown type
            for code in re.split(r"[/,]", group):
                code = code.strip()
                if code and COURSE_CODE_PATTERN.match(code):
                    prerequisites.append({
                        "type":        "unknown",
                        "course_code": code,
                    })

    return prerequisites


def extract_course_from_cells(
    cells: List[str],
    start: int,
    term_name: Optional[str],
    year_level: Optional[int],
    term_number: Optional[int],
    program: str,
    batch_id: int,
    academic_year: str,
) -> Optional[Dict[str, Any]]:
    """
    Try to extract one course entry from a slice of table cells.

    Expected layout from `start`:
        [course_code, title, units, prereq_type_flag, prereq_codes]

    Returns None if the slice does not look like a valid course row.
    """
    try:
        code      = cells[start].strip()   if start   < len(cells) else ""
        title     = cells[start+1].strip() if start+1 < len(cells) else ""
        units_raw = cells[start+2].strip() if start+2 < len(cells) else ""
        p_type    = cells[start+3].strip() if start+3 < len(cells) else ""
        p_codes   = cells[start+4].strip() if start+4 < len(cells) else ""
    except IndexError:
        return None

    # Validate course code
    if not COURSE_CODE_PATTERN.match(code):
        return None

    # Skip obvious header/footer content
    if SKIP_PATTERNS.match(code):
        return None

    # Parse units — handle bracketed optional units like "(3)"
    units = 0
    units_match = re.search(r"\d+", units_raw)
    if units_match:
        units = int(units_match.group())

    prerequisites = parse_prerequisite_string(f"{p_type} {p_codes}".strip())

    return {
        "course_code":   code,
        "title":         title,
        "units":         units,
        "year_level":    year_level,
        "term_number":   term_number,
        "term_name":     term_name or "",
        "prerequisites": prerequisites,
        "program":       program,
        "batch_id":      batch_id,
        "academic_year": academic_year,
    }


def parse_single_checklist(
    file_path: Path,
    program: str,
    batch_id: int,
    academic_year: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Parse one checklist PDF into a list of course row dicts.

    Returns:
        (courses, warnings)
        courses  : list of parsed course dicts
        warnings : list of warning messages for the parse report
    """
    courses  = []
    warnings = []
    current_term = current_year_level = current_term_number = None

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()

                if not tables:
                    warnings.append(
                        f"{file_path.name} page {page_num}: no tables found"
                    )
                    continue

                for table in tables:
                    for row in table:
                        # Clean cells — pdfplumber returns None for empty cells
                        cleaned = [
                            str(c).strip() if c else "" for c in row
                        ]

                        if all(c == "" for c in cleaned):
                            continue

                        full_row = " ".join(c for c in cleaned if c).upper()

                        # Detect term headers
                        for term_name, (yr, tm) in TERM_MAP.items():
                            if term_name in full_row:
                                current_term        = term_name
                                current_year_level  = yr
                                current_term_number = tm
                                break

                        # Skip header/footer rows
                        if cleaned and SKIP_PATTERNS.match(cleaned[0]):
                            continue

                        # Extract left column course (columns 0-4)
                        left = extract_course_from_cells(
                            cleaned, 0,
                            current_term, current_year_level,
                            current_term_number,
                            program, batch_id, academic_year,
                        )
                        if left:
                            courses.append(left)

                        # Extract right column course (columns 4+)
                        if len(cleaned) > 4:
                            right = extract_course_from_cells(
                                cleaned, 4,
                                current_term, current_year_level,
                                current_term_number,
                                program, batch_id, academic_year,
                            )
                            if right:
                                courses.append(right)

    except Exception as e:
        warnings.append(f"{file_path.name}: PARSE ERROR — {str(e)}")

    return courses, warnings


# ============================================================
# POLICY FILE PARSING
# Handles both PDF and DOCX policy documents.
# ============================================================

# Maps keyword tuples to doc_type — ANY keyword match wins.
# Uses word-level matching so spaces vs underscores don't matter.
POLICY_DOC_TYPE_KEYWORDS = [
    (["OJT", "Practicum", "Undergraduate"],          "ojt_policy"),
    (["Academic", "Advising", "Best", "Practices"],  "advising_best_practices"),
    (["Advising", "Guidelines"],                      "advising_guidelines"),
    (["Thesis", "Policies", "Guidelines"],            "thesis_policies"),
]


def file_content_hash(file_path: Path) -> str:
    """
    Compute an MD5 hash of a file's raw bytes.
    Used to detect and skip duplicate files regardless of filename.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_doc_type(filename: str) -> str:
    """
    Infer doc_type from filename using keyword matching.
    Word-level matching handles spaces vs underscores in filenames.
    Checks from most specific to least specific rule.
    """
    # Tokenize filename into uppercase words for matching
    words = set(re.sub(r"[_\-\.()]", " ", filename).upper().split())
    for keywords, doc_type in POLICY_DOC_TYPE_KEYWORDS:
        if all(k.upper() in words for k in keywords):
            return doc_type
    # Fallback: partial keyword match (any single keyword)
    for keywords, doc_type in POLICY_DOC_TYPE_KEYWORDS:
        if any(k.upper() in words for k in keywords[:2]):
            return doc_type
    return "policy"


def parse_policy_pdf(file_path: Path) -> Tuple[str, List[str]]:
    """Extract full text from a policy PDF. Returns (text, warnings)."""
    parts    = []
    warnings = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    parts.append(text.strip())
                else:
                    warnings.append(
                        f"{file_path.name} page {page_num}: no text extracted"
                    )
    except Exception as e:
        warnings.append(f"{file_path.name}: PARSE ERROR — {str(e)}")

    return "\n\n".join(parts), warnings


def parse_policy_docx(file_path: Path) -> Tuple[str, List[str]]:
    """Extract full text from a policy DOCX. Returns (text, warnings)."""
    parts    = []
    warnings = []
    NS       = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    try:
        doc = DocxDocument(str(file_path))

        for element in doc.element.body:
            tag = element.tag.split("}")[-1]

            if tag == "p":
                for p in doc.paragraphs:
                    if p._element is element:
                        text = p.text.strip()
                        if text:
                            prefix = "\n\n" if p.style.name.startswith("Heading") else ""
                            parts.append(f"{prefix}{text}")
                        break

            elif tag == "tbl":
                for row in element.findall(f".//{{{NS}}}tr"):
                    row_texts = []
                    for cell in row.findall(f"{{{NS}}}tc"):
                        cell_text = " ".join(
                            n.text for n in cell.iter()
                            if n.text and n.text.strip()
                        ).strip()
                        if cell_text:
                            row_texts.append(cell_text)
                    if row_texts:
                        parts.append(" | ".join(row_texts))

    except Exception as e:
        warnings.append(f"{file_path.name}: PARSE ERROR — {str(e)}")

    return "\n".join(parts), warnings


def parse_single_policy(file_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """
    Parse one policy file (PDF or DOCX) into a policy section dict.

    Returns:
        (section, warnings)
        section  : dict with title, doc_type, source, text
        warnings : list of warning messages
    """
    suffix   = file_path.suffix.lower()
    doc_type = resolve_doc_type(file_path.name)
    title    = file_path.stem.replace("_", " ").replace("-", " ")

    if suffix == ".pdf":
        text, warnings = parse_policy_pdf(file_path)
    elif suffix in (".docx", ".doc"):
        text, warnings = parse_policy_docx(file_path)
    else:
        return {}, [f"{file_path.name}: unsupported file type '{suffix}'"]

    section = {
        "title":    title,
        "doc_type": doc_type,
        "source":   file_path.name,
        "text":     text,
    }
    return section, warnings


# ============================================================
# BATCH RUNNER
# Discovers and parses all files under DATA_DIR.
# ============================================================

def discover_checklist_files(data_dir: Path) -> List[Path]:
    """
    Recursively find all checklist PDFs under data_dir.
    Matches files named like: 'CPE ID 118 CHECKLIST.pdf'
    """
    pattern = re.compile(
        r"^(CPE|ECE|EES|CPES)\s+ID\s+\d{3}\s+CHECKLIST\.pdf$",
        re.IGNORECASE
    )
    return sorted([
        f for f in data_dir.rglob("*.pdf")
        if pattern.match(f.name)
    ])


def discover_policy_files(data_dir: Path) -> List[Path]:
    """
    Recursively find all policy PDFs and DOCX files under data_dir.
    Excludes checklist files.
    """
    checklist_pattern = re.compile(
        r"^(CPE|ECE|EES|CPES)\s+ID\s+\d{3}\s+CHECKLIST\.pdf$",
        re.IGNORECASE
    )
    policy_files = []
    for ext in ("*.pdf", "*.docx", "*.doc"):
        for f in data_dir.rglob(ext):
            if not checklist_pattern.match(f.name):
                policy_files.append(f)
    return sorted(policy_files)


def run_batch_parse(data_dir: Path, output_dir: Path) -> None:
    """
    Parse all documents under data_dir and save results to output_dir.

    Output files:
        checklist_rows.json   : all parsed course rows
        policy_sections.json  : all parsed policy text sections
        parse_report.json     : summary and warnings
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_courses         = []
    all_policy_sections = []
    all_warnings        = []
    parse_report        = {
        "checklist_files":  [],
        "policy_files":     [],
        "total_courses":    0,
        "total_policies":   0,
        "warnings":         [],
    }

    # ----------------------------------------------------------
    # 1. Parse all checklist files
    # ----------------------------------------------------------
    checklist_files = discover_checklist_files(data_dir)
    print(f"\nFound {len(checklist_files)} checklist file(s).")

    for file_path in checklist_files:
        meta = parse_checklist_filename(file_path.name)
        if not meta:
            warning = f"Could not parse filename: {file_path.name} — skipping"
            print(f"  [WARN] {warning}")
            all_warnings.append(warning)
            continue

        program       = meta["program"]
        batch_id      = meta["batch_id"]
        academic_year = meta["academic_year"]

        print(f"  Parsing {file_path.name} [{program} | ID {batch_id} | AY {academic_year}]...")
        courses, warnings = parse_single_checklist(
            file_path, program, batch_id, academic_year
        )

        all_courses.extend(courses)
        all_warnings.extend(warnings)

        file_report = {
            "file":          file_path.name,
            "program":       program,
            "batch_id":      batch_id,
            "academic_year": academic_year,
            "courses_found": len(courses),
            "warnings":      warnings,
        }
        parse_report["checklist_files"].append(file_report)
        print(f"    -> {len(courses)} courses parsed. {len(warnings)} warning(s).")

    # ----------------------------------------------------------
    # 2. Parse all policy files
    # ----------------------------------------------------------
    policy_files = discover_policy_files(data_dir)
    print(f"\nFound {len(policy_files)} policy file(s).")

    seen_hashes: Set[str] = set()   # tracks content hashes to skip duplicates

    for file_path in policy_files:
        # Deduplicate by file content hash — catches files with different
        # names but identical content (e.g. the two advising guidelines PDFs)
        content_hash = file_content_hash(file_path)
        if content_hash in seen_hashes:
            msg = (f"  [SKIP] {file_path.name} — duplicate content "
                   f"(hash {content_hash[:8]}...), skipping.")
            print(msg)
            all_warnings.append(msg.strip())
            parse_report["policy_files"].append({
                "file":     file_path.name,
                "doc_type": "duplicate — skipped",
                "skipped":  True,
            })
            continue
        seen_hashes.add(content_hash)

        print(f"  Parsing {file_path.name}...")
        section, warnings = parse_single_policy(file_path)

        if section:
            all_policy_sections.append(section)
            all_warnings.extend(warnings)

            file_report = {
                "file":         file_path.name,
                "doc_type":     section.get("doc_type", "unknown"),
                "text_length":  len(section.get("text", "")),
                "content_hash": content_hash[:8],   # short hash for the report
                "warnings":     warnings,
            }
            parse_report["policy_files"].append(file_report)
            print(f"    -> {len(section.get('text', ''))} chars extracted. "
                  f"{len(warnings)} warning(s).")

    # ----------------------------------------------------------
    # 3. Add handbook sections from source_data.py
    # ----------------------------------------------------------
    try:
        from source_data import HANDBOOK_SECTIONS
        print(f"\nLoading {len(HANDBOOK_SECTIONS)} handbook section(s) from source_data.py...")
        for section in HANDBOOK_SECTIONS:
            all_policy_sections.append({
                "title":    section["title"],
                "doc_type": section["doc_type"],
                "source":   section.get("source", "DLSU Student Handbook"),
                "text":     section["text"],
            })
        print(f"  -> {len(HANDBOOK_SECTIONS)} handbook sections loaded.")
    except ImportError:
        warning = "source_data.py not found — handbook sections skipped"
        print(f"  [WARN] {warning}")
        all_warnings.append(warning)

    # ----------------------------------------------------------
    # 4. Save outputs
    # ----------------------------------------------------------
    checklist_out = output_dir / "checklist_rows.json"
    policy_out    = output_dir / "policy_sections.json"
    report_out    = output_dir / "parse_report.json"

    with open(checklist_out, "w", encoding="utf-8") as f:
        json.dump(all_courses, f, indent=2, ensure_ascii=False)

    with open(policy_out, "w", encoding="utf-8") as f:
        json.dump(all_policy_sections, f, indent=2, ensure_ascii=False)

    parse_report["total_courses"]  = len(all_courses)
    parse_report["total_policies"] = len(all_policy_sections)
    parse_report["warnings"]       = all_warnings

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(parse_report, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------
    # 5. Print summary
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("PARSE COMPLETE")
    print("="*60)
    print(f"  Checklist files parsed : {len(parse_report['checklist_files'])}")
    print(f"  Policy files parsed    : {len(parse_report['policy_files'])}")
    print(f"  Total course rows      : {len(all_courses)}")
    print(f"  Total policy sections  : {len(all_policy_sections)}")
    print(f"  Total warnings         : {len(all_warnings)}")
    print(f"\nOutput files saved to '{output_dir}/':")
    print(f"  {checklist_out.name}")
    print(f"  {policy_out.name}")
    print(f"  {report_out.name}")

    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  [WARN] {w}")
    print("="*60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch parse all DLSU CpE checklist and policy documents."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Root directory containing all document folders (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to save parsed JSON output (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    print(f"Data directory  : {args.data_dir.resolve()}")
    print(f"Output directory: {args.output_dir.resolve()}")

    if not args.data_dir.exists():
        print(f"\n[ERROR] Data directory not found: {args.data_dir.resolve()}")
        print("Please update DATA_DIR at the top of this file or use --data-dir.")
        exit(1)

    run_batch_parse(args.data_dir, args.output_dir)
