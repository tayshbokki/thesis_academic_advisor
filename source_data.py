# ============================================================
# DLSU CpE AI Academic Advising System
# Source Data — Handbook Sections & Curated FAQs
# ============================================================
# This file contains:
#   1. HANDBOOK_SECTIONS : official policy text from the DLSU
#      Student Handbook and official memos, structured for
#      ingestion by chunk_and_ingest_policy_text() in the
#      chunking pipeline.
#
#   2. FAQ_LIST : real adviser-written FAQ entries derived from
#      actual adviser communications to CpE advisees. Ready for
#      ingestion by chunk_and_ingest_faqs() in the pipeline.
#
# Usage:
#   from source_data import HANDBOOK_SECTIONS, FAQ_LIST
#   from chunking_pipeline import (
#       chunk_and_ingest_policy_text,
#       chunk_and_ingest_faqs,
#   )
#   for section in HANDBOOK_SECTIONS:
#       chunk_and_ingest_policy_text(**section)
#   chunk_and_ingest_faqs(FAQ_LIST)
# ============================================================


# ============================================================
# HANDBOOK SECTIONS
# Each entry is one logical policy document or memo.
# The 'text' field is the raw content that gets chunked.
# 'doc_type' maps to the ChromaDB metadata tag used for
# filtered retrieval (e.g. retrieve only retention policy
# chunks when a student asks about failures).
# ============================================================

HANDBOOK_SECTIONS = [

    # ----------------------------------------------------------
    # Retention Policy — updated via Provost memo Nov 2025
    # Effective Term 2 AY 2025-2026 (ArchersHub cutover)
    # ----------------------------------------------------------
    {
        "title":    "Retention Policy for Undergraduates (Revised Nov 2025)",
        "doc_type": "retention_policy",
        "version":  "Term 2 AY 2025-2026",
        "source":   "Office of the Provost — provost@dlsu.edu.ph",
        "text": """
Revision of Retention Policy for Undergraduates
Office of the Provost | 29 November 2025
Effective: Term 2 AY 2025-2026

Section 10.17.1
Students of any undergraduate degree program who have accumulated 36 units of failure
in academic courses at the end of any trimester are ineligible to continue studies in
the current program.

Section 10.17.3
Any failure incurred is automatically added to previous accumulated units of failure.
Only students who have not reached the maximum allowable accumulated failures may
re-enroll any failed course.

Note: The following clause has been REMOVED from the old Section 10.17.3 effective
Term 2 AY 2025-2026:
"If a student receives a grade of 2.5 or higher in the re-enrolled failed course,
the original failure will not be counted in the accumulation of the number of units
failed. However, all failures will still be counted for purposes of GPA computation
and will be reflected on the transcript of records."

Section 2.5.2 (unchanged)
When a student shifts to another program, failures in courses that are not part of
the curriculum of the new program shall not be included in the computation of
accumulated units of failures. These failures shall, however, still be reflected in
the transcript of records, and included in the computation of the CGPA.

Background:
These revisions arose from discussions of a trisectoral committee composed of USG,
AFED and DLSU Administration representatives in March 2023. They were approved by
the Academics Council in May 2023. The revisions took effect in Term 2 AY 2025-2026
upon the University's cutover from MLS/Animosys to ArchersHub (the new student
information system).
        """.strip(),
    },

    # ----------------------------------------------------------
    # Academic Load and Overload Policy
    # Sections 10.2, 10.2.1, 10.2.2
    # ----------------------------------------------------------
    {
        "title":    "Academic Load and Overload Policy",
        "doc_type": "load_policy",
        "version":  "Current",
        "source":   "DLSU Student Handbook",
        "text": """
Academic Load and Overload Policy
DLSU Student Handbook — Section 10.2

Section 10.2
For regular terms, the maximum academic load for undergraduate students is 15 units,
or the number of units indicated on the program checklist. A student may enroll more
than the maximum allowable load during a regular term under the following circumstances:

Section 10.2.1
The additional units pertain to or the enrolled courses includes a PE course.

Section 10.2.2
The student is in their last term and the overload does not exceed 6 units.

Note for CpE students:
The program checklist may specify a term load higher than 15 units. In such cases,
the checklist load takes precedence. Overload beyond the checklist load still requires
adviser and registrar approval under Sections 10.2.1 and 10.2.2.
        """.strip(),
    },

    # ----------------------------------------------------------
    # Lecture and Laboratory Course Policies
    # Sections 10.10.1–10.10.5
    # ----------------------------------------------------------
    {
        "title":    "Lecture and Laboratory Course Policies",
        "doc_type": "lab_lecture_policy",
        "version":  "Current",
        "source":   "DLSU Student Handbook",
        "text": """
Lecture and Laboratory Course Policies
DLSU Student Handbook — Sections 10.10.1 to 10.10.5

Section 10.10.1
The laboratory course is a co-requisite of the corresponding lecture course.
Both should be taken during the same term.

Section 10.10.2
Separate grades are given for the lecture and laboratory courses unless specified
otherwise by the department concerned.

Section 10.10.3
If a student drops the lecture course, they must also drop the laboratory course.
However, a student may drop the laboratory course without dropping the lecture course
unless specified otherwise by the department concerned.

Section 10.10.4
If the student fails in either the laboratory or lecture course, they should
re-enroll only in the failed subject.

Section 10.10.5
The student has to pass both the lecture and laboratory courses to proceed to the
succeeding course(s) unless specified otherwise by the college concerned.

Examples in CpE curriculum:
- MICPROS (lecture) and LBYCPA3 (lab) must be taken in the same term.
- LOGDSGN (lecture) and LBYCPG4 (lab) must be taken in the same term.
- Failing LBYCPA3 does not require re-enrolling MICPROS; only LBYCPA3 is retaken.
        """.strip(),
    },

    # ----------------------------------------------------------
    # Course Crediting Process
    # Based on adviser communication to CpE advisees
    # ----------------------------------------------------------
    {
        "title":    "Course Crediting Process",
        "doc_type": "crediting_process",
        "version":  "Current",
        "source":   "DECEE Department — Adviser Communication",
        "text": """
Course Crediting Process
Department of Electronics, Computer, and Electrical Engineering (DECEE)

Step 1: Submit the Course Crediting Form to the DECEE Chair.

Step 2: The DECEE Chair (currently Dr. Bandala) reviews and approves the course
crediting request.

Step 3: Once approved by the Department Chair, follow up with the Office of the
University Registrar (OUR) at registrar@dlsu.edu.ph to confirm whether the course
has been officially credited in the student information system.

Contact:
Office of the University Registrar (OUR): registrar@dlsu.edu.ph
DECEE Department Chair: Dr. Bandala
        """.strip(),
    },

    # ----------------------------------------------------------
    # Academic Tutorial and Reviewer Resources
    # Based on adviser communication to CpE advisees
    # ----------------------------------------------------------
    {
        "title":    "Academic Reviewer and Tutorial Resources",
        "doc_type": "academic_resources",
        "version":  "Current",
        "source":   "Adviser Communication to CpE Advisees",
        "text": """
Academic Reviewer and Tutorial Resources for CpE Students

A. SYNTAX Reviewers
Available at: https://dlsusyntax.wixstudio.com/dlsu-syntax/reviewers
SYNTAX is a student organization providing reviewers for CpE-related courses.

B. ACCESS Tutorials and Materials
Available at: https://accessdlsu.wixsite.com/access-dlsu/academics
ACCESS provides academic materials and tutorials for engineering students.

C. ECES Tutorials and Materials
Contact the ECES officers directly for tutorial schedules and materials.

D. SHINE Tutorials (English Language)
Free online English language tutorials offered by DLSU.
Sessions are held weekly. No sessions on holidays or when classes are canceled.
Interested students may fill out the SHINE registration form to participate.
For inquiries contact:
  Mr. Amir Austria — amir.austria@dlsu.edu.ph
  Dr. Ron Resurreccion — ron.resurreccion@dlsu.edu.ph
        """.strip(),
    },
]


# ============================================================
# FAQ LIST
# Derived from real adviser communications to CpE advisees.
# Each entry maps directly to chunk_and_ingest_faqs() format.
# Categories align with query_type in the advising schema:
#   "Retention", "Enrollment", "Prerequisites", "OJT",
#   "Crediting", "Lab/Lecture", "Resources", "Advising"
# ============================================================

FAQ_LIST = [

    # --- Retention Policy ---
    {
        "question":   "How many units of failure will make me ineligible to continue my program?",
        "answer":     (
            "Under Section 10.17.1 (revised effective Term 2 AY 2025-2026), students who "
            "have accumulated 36 units of failure in academic courses at the end of any "
            "trimester are ineligible to continue studies in their current program."
        ),
        "category":   "Retention",
        "program":    "BS Computer Engineering",
        "difficulty": "basic",
        "verified":   True,
    },
    {
        "question":   "If I retake a failed course and pass it, does the original failure get removed from my accumulated failures?",
        "answer":     (
            "No. Under the revised Section 10.17.3 (effective Term 2 AY 2025-2026), any "
            "failure incurred is automatically added to previous accumulated units of failure "
            "and stays there permanently. The old rule that allowed a passing grade of 2.5 or "
            "higher to remove the original failure from the count has been removed. All failures "
            "remain in the accumulation regardless of retake grades."
        ),
        "category":   "Retention",
        "program":    "BS Computer Engineering",
        "difficulty": "complex",
        "verified":   True,
    },
    {
        "question":   "If I shift to another program, will my failures from my old program count against me?",
        "answer":     (
            "Under Section 2.5.2, failures in courses that are not part of the curriculum of "
            "your new program will not be included in the computation of accumulated units of "
            "failures for retention purposes. However, those failures will still appear on your "
            "transcript of records and will be included in the computation of your CGPA."
        ),
        "category":   "Retention",
        "program":    "BS Computer Engineering",
        "difficulty": "complex",
        "verified":   True,
    },
    {
        "question":   "When did the new retention policy take effect?",
        "answer":     (
            "The revised retention policy took effect in Term 2 AY 2025-2026, coinciding with "
            "the University's cutover from MLS/Animosys to the new student information system "
            "called ArchersHub. The revisions were approved by the Academics Council in May 2023 "
            "following discussions by a trisectoral committee of USG, AFED, and DLSU Administration."
        ),
        "category":   "Retention",
        "program":    "BS Computer Engineering",
        "difficulty": "basic",
        "verified":   True,
    },

    # --- Overload / Academic Load ---
    {
        "question":   "What is the maximum number of units I can enroll in per term?",
        "answer":     (
            "Under Section 10.2, the maximum academic load for undergraduate students in "
            "regular terms is 15 units, or the number of units indicated on the program "
            "checklist — whichever is applicable. For CpE students, the checklist often "
            "specifies higher loads per term (e.g. 18 or 19 units), and those checklist "
            "loads take precedence."
        ),
        "category":   "Enrollment",
        "program":    "BS Computer Engineering",
        "difficulty": "basic",
        "verified":   True,
    },
    {
        "question":   "Under what conditions can I overload beyond the maximum load?",
        "answer":     (
            "Under Sections 10.2.1 and 10.2.2, a student may enroll beyond the maximum "
            "allowable load in two cases: (1) the additional units pertain to or include a "
            "PE course, or (2) the student is in their last term and the overload does not "
            "exceed 6 units beyond the maximum. Both conditions require adviser approval."
        ),
        "category":   "Enrollment",
        "program":    "BS Computer Engineering",
        "difficulty": "complex",
        "verified":   True,
    },

    # --- Lecture and Laboratory ---
    {
        "question":   "Can I take a laboratory course without taking its lecture course?",
        "answer":     (
            "No. Under Section 10.10.1, the laboratory course is a co-requisite of the "
            "corresponding lecture course — both must be taken during the same term. "
            "For example, MICPROS and LBYCPA3 must be enrolled together."
        ),
        "category":   "Prerequisites",
        "program":    "BS Computer Engineering",
        "difficulty": "basic",
        "verified":   True,
    },
    {
        "question":   "If I drop my lecture course, do I also have to drop the lab?",
        "answer":     (
            "Yes. Under Section 10.10.3, if you drop the lecture course you must also drop "
            "the corresponding laboratory course. However, you may drop the laboratory course "
            "alone without dropping the lecture course, unless the department specifies otherwise."
        ),
        "category":   "Enrollment",
        "program":    "BS Computer Engineering",
        "difficulty": "basic",
        "verified":   True,
    },
    {
        "question":   "I failed my laboratory course but passed the lecture. Do I need to retake both?",
        "answer":     (
            "No. Under Section 10.10.4, if you fail either the lecture or the laboratory "
            "course, you only need to re-enroll in the subject you failed. You do not need "
            "to retake the course you passed. However, under Section 10.10.5, you must pass "
            "both the lecture and laboratory before you can proceed to the succeeding courses."
        ),
        "category":   "Prerequisites",
        "program":    "BS Computer Engineering",
        "difficulty": "complex",
        "verified":   True,
    },
    {
        "question":   "Do lecture and laboratory courses have separate grades?",
        "answer":     (
            "Yes. Under Section 10.10.2, separate grades are given for lecture and laboratory "
            "courses unless the department specifies otherwise. Both grades appear independently "
            "on your transcript."
        ),
        "category":   "Enrollment",
        "program":    "BS Computer Engineering",
        "difficulty": "basic",
        "verified":   True,
    },

    # --- Course Crediting ---
    {
        "question":   "How do I get a course credited in my curriculum?",
        "answer":     (
            "First, submit the Course Crediting Form to the DECEE Chair. Once Dr. Bandala "
            "(the current Department Chair) approves the crediting, follow up with the Office "
            "of the University Registrar (OUR) at registrar@dlsu.edu.ph to confirm the course "
            "has been officially credited in your academic record."
        ),
        "category":   "Crediting",
        "program":    "BS Computer Engineering",
        "difficulty": "basic",
        "verified":   True,
    },
    {
        "question":   "Who approves course crediting requests for CpE students?",
        "answer":     (
            "Course crediting requests are reviewed and approved by the DECEE Department Chair, "
            "currently Dr. Bandala. After departmental approval, the Office of the University "
            "Registrar (OUR) at registrar@dlsu.edu.ph processes the official crediting."
        ),
        "category":   "Crediting",
        "program":    "BS Computer Engineering",
        "difficulty": "basic",
        "verified":   True,
    },

    # --- Academic Resources ---
    {
        "question":   "Where can I find reviewers or tutorial materials for my CpE courses?",
        "answer":     (
            "Several resources are available: (1) SYNTAX reviewers at "
            "https://dlsusyntax.wixstudio.com/dlsu-syntax/reviewers, (2) ACCESS tutorials "
            "and materials at https://accessdlsu.wixsite.com/access-dlsu/academics, "
            "(3) ECES tutorials — contact the ECES officers directly for schedules, and "
            "(4) SHINE free online English language tutorials — fill out the SHINE "
            "registration form. For SHINE inquiries contact Mr. Amir Austria at "
            "amir.austria@dlsu.edu.ph or Dr. Ron Resurreccion at ron.resurreccion@dlsu.edu.ph."
        ),
        "category":   "Resources",
        "program":    "BS Computer Engineering",
        "difficulty": "basic",
        "verified":   True,
    },
]
