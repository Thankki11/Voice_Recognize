"""
PostgreSQL connection helper.
Tự động tạo schema (2 bảng) khi kết nối lần đầu.
"""

import psycopg2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH  = PROJECT_ROOT / "database" / "schema.sql"

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "voice_search",
    "user":     "postgres",
    "password": "1",
}


def get_connection(**overrides) -> psycopg2.extensions.connection:
    cfg  = {**DB_CONFIG, **overrides}
    conn = psycopg2.connect(**cfg)

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(schema)
    conn.commit()

    return conn
