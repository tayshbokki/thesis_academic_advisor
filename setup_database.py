# ============================================================
# DLSU CpE AI Academic Advising System
# Database Setup Script
# ============================================================
# Creates the database and runs dlsu_cpe_schema.sql against
# your local XAMPP MySQL server.
#
# Requirements:
#   pip install mysql-connector-python
#
# Usage:
#   python setup_database.py
#   python setup_database.py --password yourpassword
#   python setup_database.py --host localhost --port 3306
# ============================================================

import argparse
import mysql.connector
from mysql.connector import Error
from pathlib import Path

# ============================================================
# CONFIGURATION
# XAMPP MySQL defaults — change only if you modified them
# ============================================================
DEFAULT_HOST     = "localhost"
DEFAULT_PORT     = 3306
DEFAULT_USER     = "root"
DEFAULT_PASSWORD = ""              # XAMPP default is no password
DEFAULT_DB       = "dlsu_cpe_advising"
SCHEMA_FILE      = Path("dlsu_cpe_schema.sql")


def create_database(cursor, db_name: str) -> None:
    """Create the database if it does not already exist."""
    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
    )
    cursor.execute(f"USE `{db_name}`;")
    print(f"[OK] Database '{db_name}' ready.")


def run_schema(cursor, schema_path: Path) -> None:
    """
    Read the SQL schema file and execute each statement.
    Splits on semicolons, skips empty statements and
    SET FOREIGN_KEY_CHECKS lines (handled separately).
    """
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_path.resolve()}\n"
            f"Make sure dlsu_cpe_schema.sql is in the same folder as this script."
        )

    sql = schema_path.read_text(encoding="utf-8")

    # Split into individual statements
    statements = [s.strip() for s in sql.split(";")]
    executed = 0
    skipped  = 0

    for stmt in statements:
        # Skip empty lines and comment-only blocks
        lines = [l for l in stmt.splitlines() if l.strip() and not l.strip().startswith("--")]
        if not lines:
            skipped += 1
            continue

        clean = "\n".join(lines).strip()
        if not clean:
            skipped += 1
            continue

        try:
            cursor.execute(clean)
            executed += 1

            # Extract table name for feedback
            if clean.upper().startswith("CREATE TABLE"):
                words = clean.split()
                table = words[2].strip("`") if len(words) > 2 else "unknown"
                print(f"  [created] {table}")

        except Error as e:
            # Warn but continue — don't abort on non-fatal errors
            # (e.g. table already exists)
            print(f"  [WARN] {e.msg} — statement skipped")
            skipped += 1

    print(f"\n[OK] Schema applied: {executed} statements executed, {skipped} skipped.")


def verify_tables(cursor, db_name: str) -> None:
    """List all tables in the database as a verification step."""
    cursor.execute(f"USE `{db_name}`;")
    cursor.execute("SHOW TABLES;")
    tables = [row[0] for row in cursor.fetchall()]

    expected = [
        "users", "students", "advisers",
        "curricula", "courses", "curriculum_courses", "prerequisites",
        "enrollments", "advising_sessions", "advising_queries",
        "advising_responses", "plan_courses",
        "documents", "faq_items",
        "embedding_meta",
        "metric_runs", "metric_results",
    ]

    print(f"\n[Verification] Tables in '{db_name}':")
    all_good = True
    for t in expected:
        found = t in tables
        status = "✓" if found else "✗ MISSING"
        print(f"  {status}  {t}")
        if not found:
            all_good = False

    if all_good:
        print(f"\n[OK] All {len(expected)} tables created successfully.")
    else:
        print(f"\n[WARN] Some tables are missing — check the warnings above.")


def main():
    parser = argparse.ArgumentParser(
        description="Set up the DLSU CpE advising database on XAMPP MySQL."
    )
    parser.add_argument("--host",     default=DEFAULT_HOST)
    parser.add_argument("--port",     default=DEFAULT_PORT, type=int)
    parser.add_argument("--user",     default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--database", default=DEFAULT_DB)
    parser.add_argument("--schema",   default=str(SCHEMA_FILE), type=Path)
    args = parser.parse_args()

    print("=" * 55)
    print("DLSU CpE Advising System — Database Setup")
    print("=" * 55)
    print(f"Host     : {args.host}:{args.port}")
    print(f"User     : {args.user}")
    print(f"Database : {args.database}")
    print(f"Schema   : {args.schema.resolve()}")
    print("=" * 55)

    try:
        # Connect without specifying a database first
        conn = mysql.connector.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
        )
        cursor = conn.cursor()
        print(f"\n[OK] Connected to MySQL at {args.host}:{args.port}")

        # Create database
        create_database(cursor, args.database)

        # Run schema
        print(f"\nRunning schema from '{args.schema}'...")
        run_schema(cursor, args.schema)

        # Commit and verify
        conn.commit()
        verify_tables(cursor, args.database)

        cursor.close()
        conn.close()
        print("\n[Done] Database is ready to use.")

    except Error as e:
        print(f"\n[ERROR] Could not connect to MySQL: {e}")
        print("\nTroubleshooting:")
        print("  1. Open XAMPP Control Panel and make sure MySQL is running (green)")
        print("  2. If you set a MySQL root password, run:")
        print("     python setup_database.py --password yourpassword")
        print("  3. Default XAMPP MySQL port is 3306 — if you changed it, run:")
        print("     python setup_database.py --port yourport")


if __name__ == "__main__":
    main()
