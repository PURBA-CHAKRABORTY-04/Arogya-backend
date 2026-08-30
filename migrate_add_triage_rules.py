"""
One-shot migration: adds the `triage_rules` TEXT column to the triage_sessions table.
Safe to run multiple times — the column is only added if it does not already exist.

Run with:
    python migrate_add_triage_rules.py
"""
import os
import re
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing in .env")

# Strip channel_binding param if present (not supported by psycopg2 driver)
if "channel_binding" in DATABASE_URL:
    DATABASE_URL = re.sub(r"[&?]channel_binding=[^&]*", "", DATABASE_URL)
    DATABASE_URL = re.sub(r"[&?]$", "", DATABASE_URL)

engine = create_engine(DATABASE_URL)

MIGRATION_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'triage_sessions'
          AND column_name = 'triage_rules'
    ) THEN
        ALTER TABLE triage_sessions ADD COLUMN triage_rules TEXT;
        RAISE NOTICE 'Column triage_rules added to triage_sessions.';
    ELSE
        RAISE NOTICE 'Column triage_rules already exists — skipping.';
    END IF;
END
$$;
"""

with engine.begin() as conn:
    conn.execute(text(MIGRATION_SQL))

print("Migration complete.")
