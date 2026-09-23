-- ============================================================
-- DLSU CpE AI Academic Advising System
-- Relational Database Schema (MySQL)
-- ============================================================
-- Follows professor's style:
--   - UUID primary keys (CHAR(36))
--   - Inline comments on every field
--   - Clean flat naming (snake_case)
--   - Explicit DEFAULT values
--   - Foreign keys with ON DELETE rules
--
-- Clusters:
--   1. User & Roles
--   2. Curriculum & Courses
--   3. Enrollment & Advising
--   4. Knowledge Base & FAQs
--   5. Embeddings
--   6. Metrics & Evaluation
-- ============================================================

SET FOREIGN_KEY_CHECKS = 0;
SET NAMES utf8mb4;


-- ============================================================
-- CLUSTER 1: USER & ROLES
-- Stores authentication and role information.
-- A user is either a student, an adviser, or an admin.
-- Student and adviser profiles are linked back to this table.
-- ============================================================

CREATE TABLE users (
    user_id       CHAR(36)     PRIMARY KEY,                  -- UUID
    username      VARCHAR(50)  NOT NULL UNIQUE,              -- login handle
    email         VARCHAR(100) NOT NULL UNIQUE,              -- must be DLSU email
    password_hash VARCHAR(255) NOT NULL,                     -- bcrypt hash, never plaintext
    role          VARCHAR(20)  NOT NULL,                     -- "student", "adviser", "admin"
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,        -- soft-disable without deleting
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE students (
    student_id      CHAR(36)    PRIMARY KEY,                 -- UUID
    user_id         CHAR(36)    NOT NULL UNIQUE,             -- one-to-one with users
    student_number  VARCHAR(20) NOT NULL UNIQUE,             -- e.g. "12345678"
    program         VARCHAR(50) NOT NULL,                    -- e.g. "BS Computer Engineering"
    curriculum_year VARCHAR(10) NOT NULL,                    -- e.g. "2022" (AY start year)
    year_level      TINYINT     NOT NULL DEFAULT 1,          -- 1 to 4
    status          VARCHAR(20) NOT NULL DEFAULT 'regular',
    created_at      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE advisers (
    adviser_id  CHAR(36)    PRIMARY KEY,                     -- UUID
    user_id     CHAR(36)    NOT NULL UNIQUE,                 -- one-to-one with users
    department  VARCHAR(100) NOT NULL,                       -- e.g. "Electronics and Computer Engineering"
    rank        VARCHAR(50)  NOT NULL,                       -- e.g. "Assistant Professor"
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);


-- ============================================================
-- CLUSTER 2: CURRICULUM & COURSES
-- Models the academic program structure.
-- CURRICULA holds one entry per batch year per program.
-- COURSES holds one entry per unique course code.
-- CURRICULUM_COURSES maps courses to a specific curriculum
-- with term, year, and category information.
-- PREREQUISITES encodes hard, soft, and co-requisite rules.
-- ============================================================

CREATE TABLE curricula (
    curriculum_id   CHAR(36)    PRIMARY KEY,                 -- UUID
    program         VARCHAR(50) NOT NULL,                    -- e.g. "BS Computer Engineering"
    batch_id        SMALLINT    NOT NULL,                    -- e.g. 122 for ID 122
    academic_year   VARCHAR(10) NOT NULL,                    -- e.g. "2022-2023"
    start_year      SMALLINT    NOT NULL,                    -- e.g. 2022
    end_year        SMALLINT    NULL,                        -- NULL if still active
    description     TEXT        NULL,                        -- optional notes
    created_at      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_curriculum (program, batch_id)             -- one curriculum per program per batch
);

CREATE TABLE courses (
    course_id   CHAR(36)     PRIMARY KEY,                    -- UUID
    course_code VARCHAR(20)  NOT NULL UNIQUE,                -- e.g. "CALENG1", "CONETSC"
    title       VARCHAR(150) NOT NULL,                       -- full course title
    units       TINYINT      NOT NULL DEFAULT 3,             -- credit units
    description TEXT         NULL,                           -- optional course description
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE curriculum_courses (
    cc_id         CHAR(36)    PRIMARY KEY,                   -- UUID
    curriculum_id CHAR(36)    NOT NULL,                      -- FK to curricula
    course_id     CHAR(36)    NOT NULL,                      -- FK to courses
    year_level    TINYINT     NOT NULL,                      -- 1 to 4
    term_number   TINYINT     NOT NULL,                      -- 1, 2, or 3
    term_name     VARCHAR(20) NOT NULL,                      -- e.g. "FIRST TERM"
    category      VARCHAR(20) NOT NULL DEFAULT 'major',
    created_at    TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_cc (curriculum_id, course_id),             -- no duplicate entries per curriculum
    FOREIGN KEY (curriculum_id) REFERENCES curricula(curriculum_id) ON DELETE CASCADE,
    FOREIGN KEY (course_id)     REFERENCES courses(course_id)     ON DELETE CASCADE
);

CREATE TABLE prerequisites (
    prereq_id          CHAR(36)    PRIMARY KEY,              -- UUID
    course_id          CHAR(36)    NOT NULL,                 -- the course that has a prerequisite
    required_course_id CHAR(36)    NOT NULL,                 -- the course that must be taken first
    prereq_type        VARCHAR(10) NOT NULL,                 -- "H" (hard), "S" (soft), "C" (co-requisite)
    min_grade          VARCHAR(5)  NULL,                     -- minimum passing grade if specified
    created_at         TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_prereq (course_id, required_course_id, prereq_type),
    FOREIGN KEY (course_id)          REFERENCES courses(course_id) ON DELETE CASCADE,
    FOREIGN KEY (required_course_id) REFERENCES courses(course_id) ON DELETE CASCADE
);


-- ============================================================
-- CLUSTER 3: ENROLLMENT & ADVISING
-- ENROLLMENTS tracks the student's full course history.
-- ADVISING_SESSIONS captures each advising interaction.
-- ADVISING_QUERIES stores individual questions per session.
-- ADVISING_RESPONSES stores AI or human answers per query.
-- PLAN_COURSES stores the course plan produced per session.
-- ============================================================

CREATE TABLE enrollments (
    enrollment_id CHAR(36)    PRIMARY KEY,                   -- UUID
    student_id    CHAR(36)    NOT NULL,                      -- FK to students
    course_id     CHAR(36)    NOT NULL,                      -- FK to courses
    term_number   TINYINT     NOT NULL,                      -- 1, 2, or 3
    school_year   VARCHAR(10) NOT NULL,                      -- e.g. "2024-2025"
    status        VARCHAR(20) NOT NULL DEFAULT 'in_progress',
    grade         VARCHAR(5)  NULL,                          -- final grade, NULL if in progress
    created_at    TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_enrollment (student_id, course_id, term_number, school_year),
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (course_id)  REFERENCES courses(course_id)   ON DELETE CASCADE
);

CREATE TABLE advising_sessions (
    session_id  CHAR(36)    PRIMARY KEY,
    student_id  CHAR(36)    NOT NULL,
    adviser_id  CHAR(36)    NULL DEFAULT NULL,
    channel     VARCHAR(20) NOT NULL DEFAULT 'AI',
    started_at  TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at    TIMESTAMP   NULL DEFAULT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (adviser_id) REFERENCES advisers(adviser_id) ON DELETE SET NULL
);

CREATE TABLE advising_queries (
    query_id    CHAR(36)     PRIMARY KEY,                    -- UUID
    session_id  CHAR(36)     NOT NULL,                       -- FK to advising_sessions
    query_text  TEXT         NOT NULL,                       -- raw student question
    query_type  VARCHAR(30)  NOT NULL,                       -- "FAQ", "term_plan", "graduation_check",
                                                             -- "overload", "retake", "prerequisite"
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES advising_sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE advising_responses (
    response_id   CHAR(36)     PRIMARY KEY,                  -- UUID
    query_id      CHAR(36)     NOT NULL,                     -- FK to advising_queries
    response_text TEXT         NOT NULL,                     -- full generated or human-written response
    source        VARCHAR(20)  NOT NULL DEFAULT 'AI',
    is_final      BOOLEAN      NOT NULL DEFAULT FALSE,       -- TRUE = response shown to student
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES advising_queries(query_id) ON DELETE CASCADE
);

CREATE TABLE plan_courses (
    plan_id           CHAR(36)    PRIMARY KEY,               -- UUID
    session_id        CHAR(36)    NOT NULL,                  -- FK to advising_sessions
    course_id         CHAR(36)    NOT NULL,                  -- FK to courses
    target_term       TINYINT     NOT NULL,                  -- 1, 2, or 3
    school_year       VARCHAR(10) NOT NULL,                  -- e.g. "2025-2026"
    is_recommended    BOOLEAN     NOT NULL DEFAULT TRUE,     -- FALSE = flagged or blocked
    constraint_status VARCHAR(30) NOT NULL DEFAULT 'ok',
    explanation       TEXT        NULL,                      -- reason for constraint status
    created_at        TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES advising_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (course_id)  REFERENCES courses(course_id)  ON DELETE CASCADE
);


-- ============================================================
-- CLUSTER 4: KNOWLEDGE BASE & FAQs
-- DOCUMENTS holds metadata for all institutional documents
-- ingested into the RAG knowledge base.
-- FAQ_ITEMS holds curated Q&A pairs linked to source documents.
-- These feed directly into the ChromaDB vector store.
-- ============================================================

CREATE TABLE documents (
    document_id   CHAR(36)     PRIMARY KEY,                  -- UUID
    title         VARCHAR(200) NOT NULL,                     -- human-readable document title
    doc_type      VARCHAR(50)  NOT NULL,                     -- "handbook", "checklist", "ojt_policy",
                                                             -- "advising_guidelines", "thesis_policies", etc.
    source        VARCHAR(200) NOT NULL,                     -- origin, e.g. "Office of the Registrar"
    file_path     VARCHAR(300) NULL,                         -- local or cloud path to source file
    version       VARCHAR(20)  NULL,                         -- document version or effectivity date
    effective_date DATE        NULL,                         -- date the document took effect
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE faq_items (
    faq_id      CHAR(36)     PRIMARY KEY,
    document_id CHAR(36)     NULL DEFAULT NULL,
    question    TEXT         NOT NULL,
    answer      TEXT         NOT NULL,
    category    VARCHAR(50)  NOT NULL,
    program     VARCHAR(50)  NOT NULL DEFAULT 'GENERAL',
    difficulty  VARCHAR(10)  NOT NULL DEFAULT 'basic',
    verified    BOOLEAN      NOT NULL DEFAULT TRUE,
    created_by  VARCHAR(100) NULL DEFAULT NULL,
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE SET NULL
);


-- ============================================================
-- CLUSTER 5: EMBEDDINGS
-- Tracks metadata for text chunks embedded into ChromaDB.
-- Links each embedding back to its source document so the
-- RAG pipeline can return proper citations (SO3).
-- The actual embedding vectors live in ChromaDB, not here.
-- ============================================================

CREATE TABLE embedding_meta (
    embedding_id       CHAR(36)     PRIMARY KEY,             -- UUID
    document_id        CHAR(36)     NOT NULL,                -- FK to documents
    section_identifier VARCHAR(100) NOT NULL,                -- e.g. "Section 10.17.1" or "CALENG1"
    chunk_index        SMALLINT     NOT NULL DEFAULT 0,      -- position of this chunk in the document
    chunk_total        SMALLINT     NOT NULL DEFAULT 1,      -- total chunks from this document
    model_name         VARCHAR(100) NOT NULL,                -- embedding model used, e.g. "all-MiniLM-L6-v2"
    chromadb_ref       VARCHAR(200) NOT NULL,                -- ChromaDB collection + document ID reference
    created_at         TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);


-- ============================================================
-- CLUSTER 6: METRICS & EVALUATION
-- Stores results from SO1-SO5 evaluation runs.
-- METRIC_RUNS captures one evaluation experiment.
-- METRIC_RESULTS stores per-item scores within that run.
-- Supports retrieval metrics (MRR, NDCG, Recall),
-- generation metrics (BLEU, ROUGE-L, BERTScore),
-- and usability metrics (SUS, NASA-TLX).
-- ============================================================

CREATE TABLE metric_runs (
    run_id      CHAR(36)     PRIMARY KEY,                    -- UUID
    metric_type VARCHAR(50)  NOT NULL,                       -- e.g. "MRR", "NDCG", "BLEU", "SUS"
    so_target   VARCHAR(10)  NOT NULL,                       -- SO this run evaluates, e.g. "SO1", "SO3"
    run_date    DATE         NOT NULL,                       -- date the evaluation was conducted
    description TEXT         NULL,                           -- notes on configuration or test conditions
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE metric_results (
    result_id    CHAR(36)      PRIMARY KEY,                  -- UUID
    run_id       CHAR(36)      NOT NULL,                     -- FK to metric_runs
    item_type    VARCHAR(20)   NOT NULL,                     -- "query", "response", "user", "session"
    item_id      VARCHAR(100)  NOT NULL,                     -- ID of the evaluated item
    metric_value DECIMAL(10,6) NOT NULL,                     -- numeric score
    threshold    DECIMAL(10,6) NULL,                         -- target threshold for pass/fail check
    passed       BOOLEAN       NULL,                         -- TRUE if metric_value meets threshold
    notes        TEXT          NULL,                         -- optional per-item annotation
    created_at   TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES metric_runs(run_id) ON DELETE CASCADE
);


-- ============================================================
-- RE-ENABLE FOREIGN KEY CHECKS
-- ============================================================

SET FOREIGN_KEY_CHECKS = 1;